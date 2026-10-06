"""CI eval gate: replay a small fixed subset through the engine and fail the build on a quality/cache regression.

    python -m eval.ci_gate                         # gate against eval/results/ci_baseline.json (exit 1 on regression)
    python -m eval.ci_gate --update-baseline       # rewrite the baseline (do this in the PR that changes the subset)
    python -m eval.ci_gate --record --backend anthropic --estimate-only   # pre-flight: new calls + est. $, no key
    python -m eval.ci_gate --record --backend anthropic --yes   # (re)record the CI cassette with a real model

Input: eval/data/ci_subset.jsonl (~30 eval-set rows + ~20 trap pairs + a few legitimate repeats), replayed twice in
trace order: arm A0 (all levers off) and the gate arm (balanced mode exactly as configured in configs/policy.yaml,
plus COSTGUARD_TAU_OVERRIDE if set). Upstream calls come from the committed cassette eval/cassettes/<backend>_ci.jsonl
in replay mode (no key needed; COSTGUARD_CI_BACKEND picks the backend, default anthropic). Without that cassette the
gate runs on the deterministic mock backend. Quality uses the heuristic judge by default (deterministic, keyless);
`--judge model` replays eval/cassettes/judge.jsonl instead.

Checks (all deterministic):
  (a)  no trap pair produces an exact or semantic cache hit between its two rows (zero trap false hits)
  (a') false hits over all rows do not increase vs the baseline
  (b)  mean quality of the gate arm >= baseline - quality margin
  (c)  savings vs A0 (paired, %) >= baseline - savings margin
  plus: replay complete (no cassette misses, including a cheap-tier miss the pipeline papered over by falling back
  to strong) and the baseline matches this backend/judge/subset.
Writes eval/results/ci_gate.json and, if $GITHUB_STEP_SUMMARY is set, a markdown table to it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Optional

from costguard.config import ROOT, Settings, load_policy

from .build_trace import CI_SUBSET, file_sha256
from .judge import LOCAL_BACKENDS, HeuristicJudge, get_judge
from .run_ab import (CASSETTES, HIT, apply_overrides, arm_policy, keyless_provider, make_engine, run_rows,
                     set_provider_prompt_cache)
from .stats import ratio_bootstrap

RESULTS = ROOT / "eval" / "results"
CI_BASELINE = RESULTS / "ci_baseline.json"
CI_RESULT = RESULTS / "ci_gate.json"
QUALITY_MARGIN = 0.03        # absolute, on the 0..1 grade scale
SAVINGS_MARGIN = 2.0         # percentage points


def resolve_backend(requested: str) -> tuple[str, Optional[Path]]:
    """('mock', None) or (backend, its committed CI cassette)."""
    if requested == "mock":
        return "mock", None
    cands = [requested] if requested != "auto" else [os.environ.get("COSTGUARD_CI_BACKEND", "anthropic"), "mlx"]
    for b in cands:
        p = CASSETTES / f"{b}_ci.jsonl"
        if p.exists() and p.stat().st_size > 0:
            return b, p
    if requested not in ("auto",):
        raise SystemExit(f"no CI cassette for backend {requested!r} at {CASSETTES / (requested + '_ci.jsonl')}; "
                         f"record it with --record --backend {requested}")
    return "mock", None


def _load(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _stage_cassette_misses(r: dict) -> list[str]:
    """Cassette misses swallowed by a fail-open stage, e.g. a cheap-tier miss that fell back to strong
    (stage_errors["upstream_cheap"]): the request succeeded, but not with the recorded calls."""
    return [f"{k}: {v}" for k, v in (r.get("stage_errors") or {}).items() if "CassetteMiss" in str(v)]


def evaluate(engine, rows: list[dict], judge) -> dict:
    """Run A0 and the gate arm (balanced as configured + overrides) and compute the gate metrics."""
    if getattr(engine, "_ci_base_policy", None) is None:     # run_rows swaps engine.policy; keep the original
        engine._ci_base_policy = engine.policy
    base, overrides = apply_overrides(engine._ci_base_policy)
    mode_a0, pol_a0 = arm_policy(base, "A0")
    a0 = run_rows(engine, pol_a0, mode_a0, "ci-A0", rows)
    gate = run_rows(engine, base, "balanced", "ci-balanced", rows)
    by_pos = {r["pos"]: r for r in rows}
    ok = [r for r in gate if not r.get("error")]
    a0_ok = {r["pos"]: r for r in a0 if not r.get("error")}
    errors = [r for r in a0 + gate if r.get("error")]
    stage_misses = [r for r in a0 + gate if not r.get("error") and _stage_cassette_misses(r)]
    hits = [r for r in ok if r["cache_status"] in HIT]
    false_hits = [r for r in hits if r.get("false_hit")]
    trap_fh = [r for r in false_hits if r.get("trap_false_hit")]
    paired = [(r["cost"], a0_ok[r["pos"]]["cost"], r["cluster_id"]) for r in ok if r["pos"] in a0_ok]
    ratio, lo, hi = ratio_bootstrap([p[0] for p in paired], [p[1] for p in paired], clusters=[p[2] for p in paired])
    grades, grades0 = {}, {}
    for r in ok:
        grades[r["pos"]] = judge.grade(r["query"], r["answer"], r.get("reference"))
    for p, r in a0_ok.items():
        grades0[p] = judge.grade(r["query"], r["answer"], r.get("reference"))
    g = [v for v in grades.values() if v is not None]
    both = [(grades[p], grades0[p]) for p in grades if grades[p] is not None and grades0.get(p) is not None]
    tau = base.modes["balanced"].tau
    return {
        "n": len(rows), "n_ok": len(ok), "errors": len(errors), "stage_cassette_misses": len(stage_misses),
        "error_samples": ([f"{r['pos']}: {r['error']}" for r in errors[:3]]
                          + [f"{r['pos']}: {_stage_cassette_misses(r)[0]}" for r in stage_misses[:3]]),
        "hits": {"exact": sum(1 for r in hits if r["cache_status"] == "exact"),
                 "semantic": sum(1 for r in hits if r["cache_status"] == "semantic")},
        "hit_rate_pct": round(100 * len(hits) / len(gate), 2) if gate else 0.0,     # ÷ attempted requests
        "false_hits": len(false_hits), "trap_false_hits": len(trap_fh),
        "false_hit_examples": [{"pos": r["pos"], "query": r["query"], "served_answer_of": r.get("hit_from_query"),
                                "similarity": r.get("cache_similarity"), "kind": r["cache_status"],
                                "trap": bool(r.get("trap_false_hit"))}
                               # every trap false hit is named first (they are what fails check (a)), then the rest
                               for r in sorted(false_hits, key=lambda r: not r.get("trap_false_hit"))[:25]],
        "savings_pct": round(100 * (1 - ratio), 2) if paired else None,
        "savings_ci": [round(100 * (1 - hi), 2), round(100 * (1 - lo), 2)] if paired else None,
        "cost_usd": round(sum(p[0] for p in paired), 6), "a0_cost_usd": round(sum(p[1] for p in paired), 6),
        "quality_mean": round(sum(g) / len(g), 4) if g else None, "ungraded": len(grades) - len(g),
        "quality_retained_pct": (round(100 * sum(b[0] for b in both) / sum(b[1] for b in both), 2)
                                 if both and sum(b[1] for b in both) else None),
        "n_correct": sum(1 for v in g if v >= 0.75),
        "tau": tau, "config_hash": base.config_hash, "overrides": overrides,
    }


def compare(cur: dict, base: Optional[dict], q_margin: float, s_margin: float, meta: dict) -> list[dict]:
    checks = []

    def add(name, value, baseline, limit, ok, note=""):
        checks.append({"check": name, "value": value, "baseline": baseline, "limit": limit,
                       "status": "PASS" if ok else "FAIL", "note": note})

    incomplete = cur["errors"] + cur.get("stage_cassette_misses", 0)     # failed requests + swallowed misses
    add("replay complete (no upstream/cassette errors)", incomplete, 0, "== 0", incomplete == 0,
        "; ".join(cur["error_samples"]) + (" -> re-record the CI cassette" if incomplete else ""))
    add("(a) trap false hits", cur["trap_false_hits"], 0, "== 0", cur["trap_false_hits"] == 0)
    if base is None:
        add("baseline present", "missing", "-", "exists", False, "run: python -m eval.ci_gate --update-baseline")
        return checks
    bm = base["metrics"]
    same = all(base.get(k) == meta.get(k) for k in ("backend", "judge", "subset_sha256"))
    add("baseline matches backend/judge/subset", f"{meta['backend']}/{meta['judge']}/{meta['subset_sha256'][:8]}",
        f"{base.get('backend')}/{base.get('judge')}/{str(base.get('subset_sha256'))[:8]}", "equal", same,
        "" if same else "stale baseline: run python -m eval.ci_gate --update-baseline in this PR")
    add("(a') false hits, all rows", cur["false_hits"], bm["false_hits"], f"<= {bm['false_hits']}",
        cur["false_hits"] <= bm["false_hits"])
    qv, qb = cur["quality_mean"], bm.get("quality_mean")
    q_ok = qv is not None and qb is not None and qv >= qb - q_margin and cur["ungraded"] == 0
    add("(b) quality, mean grade (0-1)", qv, qb, f">= {round(qb - q_margin, 4) if qb is not None else '?'}", q_ok,
        f"{cur['ungraded']} ungraded" if cur["ungraded"] else "")
    sv, sb = cur["savings_pct"], bm.get("savings_pct")
    s_ok = sv is not None and sb is not None and sv >= sb - s_margin
    add("(c) savings vs A0 (%)", sv, sb, f">= {round(sb - s_margin, 2) if sb is not None else '?'}", s_ok)
    return checks


def render(cur: dict, checks: list[dict], meta: dict, markdown: bool = False) -> str:
    passed = all(c["status"] == "PASS" for c in checks)
    title = (f"CostGuard CI eval gate: {'PASS' if passed else 'FAIL'}  (backend={meta['backend']}, "
             f"judge={meta['judge']}, rows={cur['n']}, tau={cur['tau']}, config={cur['config_hash']})")
    info = (f"hit rate {cur['hit_rate_pct']}% (exact {cur['hits']['exact']}, semantic {cur['hits']['semantic']}), "
            f"savings {cur['savings_pct']}% {cur['savings_ci']}, quality retained {cur['quality_retained_pct']}%")
    fh = cur["false_hit_examples"]
    if markdown:
        lines = [f"### {'✅' if passed else '❌'} {title}", "", "| check | value | baseline | limit | status |",
                 "|---|---|---|---|---|"]
        lines += [f"| {c['check']} | {c['value']} | {c['baseline']} | {c['limit']} | **{c['status']}** |"
                  + (f" {c['note']}" if c["note"] else "") for c in checks]
        lines += ["", info]
        if fh:
            lines += ["", "**False hits (request ← served the cached answer of):**", ""]
            lines += [f"- {'TRAP ' if e['trap'] else ''}`{e['query']}` ← `{e['served_answer_of']}` "
                      f"({e['kind']}, sim {e['similarity']})" for e in fh]
        return "\n".join(lines) + "\n"
    w = [48, 30, 30, 14, 6]
    lines = [title, "-" * sum(w), "".join(h.ljust(n) for h, n in zip(["check", "value", "baseline", "limit", ""], w))]
    for c in checks:
        lines.append("".join(str(x).ljust(n) for x, n in zip(
            [c["check"], c["value"], c["baseline"], c["limit"], c["status"]], w)) + (f"  {c['note']}" if c["note"] else ""))
    lines += ["-" * sum(w), info]
    for e in fh:
        lines.append(f"  {'TRAP ' if e['trap'] else ''}false hit: {e['query']!r} <- served answer of "
                     f"{e['served_answer_of']!r} ({e['kind']}, sim {e['similarity']})")
    return "\n".join(lines)


def run_gate(subset: Path = CI_SUBSET, baseline: Path = CI_BASELINE, out: Optional[Path] = CI_RESULT,
             backend: str = "auto", judge_kind: str = "auto", engine=None, update_baseline: bool = False,
             quality_margin: float = QUALITY_MARGIN, savings_margin: float = SAVINGS_MARGIN,
             record: bool = False, step_summary: Optional[str] = None,
             provider_prompt_cache: bool = False) -> tuple[int, dict]:
    rows = _load(subset)
    settings = Settings.from_env()
    prompt_cache = None                 # provider prompt caching: only the anthropic adapter has it
    if engine is None:
        if record:
            if backend in ("auto", "mock"):
                raise SystemExit("--record needs a real --backend (e.g. anthropic or mlx)")
            b, cassette = backend, CASSETTES / f"{backend}_ci.jsonl"
        else:
            b, cassette = resolve_backend(backend)
        if "anthropic" in (b, os.environ.get("COSTGUARD_JUDGE_BACKEND") or b):
            prompt_cache = set_provider_prompt_cache(provider_prompt_cache)    # before any provider is built
        settings.backend = b
        if record:
            settings.cassette, settings.cassette_mode = cassette, "auto"
            engine = make_engine(settings)
        else:
            engine = (make_engine(settings) if b == "mock"
                      else make_engine(settings, provider=keyless_provider(b, cassette, "replay")))
    else:
        b = engine.settings.backend
    if judge_kind == "model" and b != "mock":
        judge = get_judge(settings, provider=engine.provider)
    else:
        judge = HeuristicJudge()
    cur = evaluate(engine, rows, judge)
    if record:     # also record the miss path of every row, so a PR that raises tau still replays
        base, _ = apply_overrides(engine._ci_base_policy)
        run_rows(engine, base, "balanced", "ci-record-nocache", rows, no_cache=True)
    meta = {"backend": b, "judge": judge.label, "subset": str(Path(subset).name),
            "subset_sha256": file_sha256(Path(subset)), "provider_prompt_cache": prompt_cache,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    if update_baseline:
        Path(baseline).parent.mkdir(parents=True, exist_ok=True)
        Path(baseline).write_text(json.dumps({**meta, "metrics": cur, "quality_margin": quality_margin,
                                              "savings_margin": savings_margin}, indent=2, ensure_ascii=False) + "\n")
        print(f"baseline written to {baseline}: savings {cur['savings_pct']}%, quality {cur['quality_mean']}, "
              f"false hits {cur['false_hits']} (trap {cur['trap_false_hits']})")
        if cur["trap_false_hits"]:
            print("WARNING: the baseline itself has trap false hits; the gate will fail until they are fixed.")
        return 0, {"meta": meta, "metrics": cur}
    base_doc = json.loads(Path(baseline).read_text()) if Path(baseline).exists() else None
    checks = compare(cur, base_doc, quality_margin, savings_margin, meta)
    passed = all(c["status"] == "PASS" for c in checks)
    result = {"meta": meta, "passed": passed, "checks": checks, "metrics": cur}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(render(cur, checks, meta))
    step_summary = step_summary if step_summary is not None else os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(render(cur, checks, meta, markdown=True))
    return (0 if passed else 1), result


def estimate_record(subset: Path, backend: str) -> dict:
    """Pre-flight for `--record`: the calls the three recording passes make (A0, balanced as configured, balanced with
    no cache) that the backend's CI cassette lacks, with a list-price estimate. Runs the real pipeline on a stand-in
    upstream (eval.run_ab.DryRunProvider), so it needs no key and calls nothing."""
    from costguard.pricing import PriceBook

    from .run_ab import DryRunProvider, _alias_for, _read_cassette
    rows = _load(subset)
    settings = Settings.from_env()
    settings.backend = backend
    cassette = CASSETTES / f"{backend}_ci.jsonl"
    store = _read_cassette(cassette)
    seen: dict[str, list[int]] = {}
    for c in store.values():
        seen.setdefault(c.get("model", ""), []).append(int(c.get("usage", {}).get("output_tokens", 0)))

    def out_est(model: str, max_tokens: int) -> int:     # same rule as run_ab's pre-flight
        xs = seen.get(model, [])
        return int(sum(xs) / len(xs)) if len(xs) >= 5 else int(0.6 * max_tokens)

    dry = DryRunProvider(backend, store, out_est)
    engine = make_engine(settings, provider=dry)
    base, _ = apply_overrides(engine.policy)
    mode_a0, pol_a0 = arm_policy(base, "A0")
    run_rows(engine, pol_a0, mode_a0, "ci-A0", rows)
    run_rows(engine, base, "balanced", "ci-balanced", rows)
    run_rows(engine, base, "balanced", "ci-record-nocache", rows, no_cache=True)
    prices = PriceBook(settings.prices_path, base.billing_for(backend))
    by_model: dict[str, list] = {}
    for _, model, _, _, n_in, n_out in dry.new.values():
        n, usd = by_model.get(model, [0, 0.0])
        by_model[model] = [n + 1, usd + prices.cost(_alias_for(base, backend, model), n_in, n_out)]
    return {"backend": backend, "rows": len(rows), "cassette": cassette.name, "cassette_entries": len(store),
            "new_generations": len(dry.new), "est_usd": round(sum(v[1] for v in by_model.values()), 4),
            "by_model": {m: {"calls": v[0], "est_usd": round(v[1], 4)} for m, v in by_model.items()}}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CI eval gate (see module docstring).")
    ap.add_argument("--subset", default=str(CI_SUBSET))
    ap.add_argument("--baseline", default=str(CI_BASELINE))
    ap.add_argument("--out", default=str(CI_RESULT))
    ap.add_argument("--backend", default="auto", help="auto | mock | <backend with a committed *_ci.jsonl cassette>")
    ap.add_argument("--judge", choices=["auto", "heuristic", "model"], default="auto",
                    help="auto/heuristic: deterministic token-overlap judge; model: replay eval/cassettes/judge.jsonl")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--record", action="store_true", help="record eval/cassettes/<backend>_ci.jsonl with a real model")
    ap.add_argument("--yes", action="store_true", help="allow --record on a paid backend")
    ap.add_argument("--estimate-only", action="store_true",
                    help="with --record: print the new generations and estimated $ for --backend, call nothing")
    ap.add_argument("--quality-margin", type=float, default=QUALITY_MARGIN)
    ap.add_argument("--savings-margin", type=float, default=SAVINGS_MARGIN)
    ap.add_argument("--provider-prompt-cache", action="store_true",
                    help="keep Anthropic prompt caching on when recording (default off; docs/EVALUATION.md section 2)")
    args = ap.parse_args(argv)
    if args.estimate_only:
        if not args.record or args.backend in ("auto", "mock"):
            ap.error("--estimate-only goes with --record --backend <real backend>")
        est = estimate_record(Path(args.subset), args.backend)
        print(f"CI cassette pre-flight ({est['backend']}, {est['rows']} subset rows, {est['cassette']} has "
              f"{est['cassette_entries']} entries): NEW generations {est['new_generations']}, est. ${est['est_usd']:.4f} "
              f"{est['by_model']}; judge: heuristic, no API calls (unless --judge model)")
        return 0
    if args.record and args.backend not in LOCAL_BACKENDS and not args.yes:
        print(f"--record on paid backend {args.backend!r} spends money; see the estimate with --estimate-only, "
              "then add --yes", file=sys.stderr)
        return 2
    code, _ = run_gate(Path(args.subset), Path(args.baseline), Path(args.out) if args.out else None, args.backend,
                       args.judge, update_baseline=args.update_baseline or args.record,
                       quality_margin=args.quality_margin, savings_margin=args.savings_margin, record=args.record,
                       provider_prompt_cache=args.provider_prompt_cache)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
