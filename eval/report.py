"""Regenerate docs/RESULTS.md from eval/results/*.json, and the README's results block from the same files.

    python -m eval.report                     # -> docs/RESULTS.md + README.md block between the results markers
    python -m eval.report --out -             # print RESULTS.md to stdout (README untouched)

Reads whichever exist: ab_summary.json (falls back to ab_summary_mlx.json, then a mock summary, clearly bannered),
ab_summary_dup*.json (duplicate-rate sensitivity), ci_gate.json + ci_baseline.json, threshold_sweep.json,
compression_eval.json, router_gate.json (or router_gate_dryrun.json), loadtest.json (or loadtest_quick.json).
Each section renders defensively: if another workstream changes its schema, the section falls back to a generic
key/value table instead of failing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

from costguard.config import ROOT

RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "RESULTS.md"
README = ROOT / "README.md"
README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


# ------------------------------------------------------------------------------------------- formatting
def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def f(x: Any, nd: int = 2, suffix: str = "") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if _num(x):
        return f"{x:,.{nd}f}{suffix}" if isinstance(x, float) else f"{x:,}{suffix}"
    return str(x)


def ci(v: Any, nd: int = 1, suffix: str = "") -> str:
    if not isinstance(v, (list, tuple)) or len(v) != 2 or v[0] is None:
        return ""
    return f" [{f(v[0], nd)}, {f(v[1], nd)}]{suffix}"


def usd(x: Any, nd: int = 4) -> str:
    return "–" if not _num(x) else f"${x:,.{nd}f}"


def table(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load(name: str) -> Optional[dict]:
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def generic(d: dict, title: str) -> str:
    rows = [[k, f(v) if not isinstance(v, (dict, list)) else f"`{json.dumps(v)[:120]}`"] for k, v in d.items()
            if not isinstance(v, list) or len(json.dumps(v)) < 200]
    return f"{title}\n\n" + table(["key", "value"], rows[:40])


def safe(fn: Callable[..., str], *args, title: str) -> str:
    try:
        return fn(*args)
    except Exception as e:  # never let one workstream's schema change break the report
        d = args[0] if args and isinstance(args[0], dict) else {}
        return generic(d, f"{title}\n\n_(structured render failed: {type(e).__name__}: {e}; generic view below)_")


# ------------------------------------------------------------------------------------------- A/B
def incomplete(a: dict) -> str:
    """'' for a complete arm (or a summary written before the flag existed), else a visible warning."""
    if a.get("complete", True):
        return ""
    c = a.get("coverage") or {}
    pw = f" / {c['pairwise_errors']} pairwise judge errors" if c.get("pairwise_errors") else ""
    return f" **INCOMPLETE: {c.get('failed', 0)} failed / {c.get('ungraded') or 0} ungraded{pw}**"


def headline_arm(arms: dict) -> tuple[Optional[str], list[str]]:
    """(the furthest cumulative arm that is complete, the later arms skipped because they are incomplete).
    Nothing qualifies when A0 is incomplete: every paired number is measured against it."""
    cands = [a for a in ("A5", "A4", "A3", "A2", "A1") if a in arms]
    if incomplete(arms.get("A0", {})):
        return None, cands
    for i, a in enumerate(cands):
        if not incomplete(arms[a]):
            return a, cands[:i]
    return None, cands


def pick_ab() -> tuple[Optional[dict], str]:
    for name in ("ab_summary.json", "ab_summary_mlx.json"):
        d = load(name)
        if d:
            return d, name
    mocks = sorted(RESULTS.glob("ab_summary*mock*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    mocks = [p for p in mocks if "_dup" not in p.name]
    if mocks:
        return json.loads(mocks[0].read_text()), mocks[0].name
    return None, ""


def render_ab(s: dict, name: str) -> str:
    m, arms = s["meta"], s["arms"]
    parts = []
    if m.get("backend") == "mock":
        parts.append("> **MOCK BACKEND — placeholder numbers.** Answers are synthetic and the judge is a token-overlap "
                     "heuristic, so quality numbers mean nothing; costs are list-price equivalents of mock token "
                     "counts. Re-run `python -m eval.run_ab --backend anthropic --yes` for real results.")
    if m.get("limit"):
        parts.append(f"> Partial run: first {m['limit']} of {m.get('trace_rows_total')} trace rows.")
    last, skipped = headline_arm(arms)
    if skipped or incomplete(arms.get("A0", {})):
        bad = [a for a in ["A0", *skipped] if a in arms and incomplete(arms[a])]
        parts.append("> " + "; ".join(f"**{a}**{incomplete(arms[a])}" for a in bad)
                     + (". Not used as the headline (failed requests and missing grades bias rates and quality)."
                        if last else ". No headline: re-run the missing requests and judgments."))
    if last and arms[last].get("savings_pct") is not None:
        a, q = arms[last], arms[last].get("quality", {})
        parts.append(
            f"**Headline ({last}, {a['description']}):** {f(a['savings_pct'], 1)}% cost saved vs the A0 baseline"
            f"{ci(a.get('savings_ci'), 1)} (95% cluster-bootstrap CI), quality retained {f(q.get('retained'), 1)}%"
            f"{ci(q.get('retained_ci'), 1)}, cache hit rate {f(a['hit_rate']['total'], 1)}%, false-hit rate "
            f"{f(a['false_hit_rate'], 2)}% of requests, on {a['n']} requests ({m['backend']}, judge "
            f"{(m.get('judge') or {}).get('judge', '–')}).")
    rows = []
    for arm, a in arms.items():
        q = a.get("quality", {})
        pw = q.get("pairwise")
        rows.append([f"**{arm}**{incomplete(a)}", a["description"], a["n"], usd(a["cost_usd"]),
                     f(a.get("savings_pct"), 1) + ci(a.get("savings_ci"), 1), f(a.get("est_savings_pct"), 1),
                     f"{f(a['hit_rate']['exact'], 1)} / {f(a['hit_rate']['semantic'], 1)}",
                     f(a["false_hit_rate"], 2), f(q.get("mean"), 3),
                     ("100 (ref)" if arm == "A0" and q else f(q.get("retained"), 1) + ci(q.get("retained_ci"), 1)),
                     f"{pw['win']}/{pw['tie'] + pw['identical']}/{pw['loss']}" if pw else "–",
                     usd(q.get("cost_per_correct_usd"), 5),
                     f"{f(a['latency_ms']['e2e']['p50'], 0)} / {f(a['latency_ms']['e2e']['p99'], 0)}"])
    parts.append("### Cumulative ablation\n\n" + table(
        ["arm", "levers", "n", "cost", "savings % vs A0 [95% CI]", "est. savings %", "hit % exact / sem",
         "false-hit %", "quality", "quality retained % [CI]", "W/T/L vs A0", "$ / correct", "p50 / p99 ms"], rows))
    parts.append("Savings are paired: each arm's per-item cost against A0's actual cost on the same items. "
                 "*est. savings* uses the per-request estimated baseline (strong tier, full prompt). False-hit rate is "
                 "wrong cache hits ÷ all requests. W/T/L counts identical answers as ties.")
    if s.get("waterfall"):
        w0 = s["waterfall"][0]
        same = (f"\n\nEvery step is computed on the same {w0['n_items']} items: those that succeeded in every arm"
                + (f" ({w0['n_excluded']} excluded)." if w0.get("n_excluded") else ".")) if "n_items" in w0 else ""
        parts.append("### Savings waterfall (each lever's increment)\n\n" + table(
            ["step", "lever", "saved $", "% of A0 cost", "input tokens saved", "output tokens saved"],
            [[f"{w['from']} → {w['to']}", w["lever"], usd(w["saved_usd"]), f(w["saved_pct_of_a0"], 1),
              f(w["saved_input_tokens"]), f(w["saved_output_tokens"])] for w in s["waterfall"]]) + same)
    lat_rows = []
    for arm, a in arms.items():
        L = a["latency_ms"]
        lat_rows.append([arm, f"{f(L['e2e_hit']['p50'], 1)} / {f(L['e2e_hit']['p99'], 1)} (n={L['e2e_hit']['n']})",
                         f"{f(L['e2e_miss']['p50'], 0)} / {f(L['e2e_miss']['p99'], 0)} (n={L['e2e_miss']['n']})",
                         f"{f(L['overhead']['p50'], 1)} / {f(L['overhead']['p99'], 1)}"])
    parts.append("### Latency (ms, p50 / p99)\n\n" + table(["arm", "cache hit", "cache miss (incl. upstream)",
                                                            "CostGuard overhead"], lat_rows)
                 + "\n\nMiss latency = CostGuard overhead + the recorded upstream generation time.")
    ops = []
    for arm, a in arms.items():
        ops.append([arm, ", ".join(f"{k} {v}" for k, v in sorted(a.get("route_mix", {}).items())),
                    f(a["compression"]["mean_ratio"], 2) + (f" (n={a['compression']['n']})" if a["compression"]["n"] else ""),
                    f"{f(a['context']['mean_docs_kept'], 1)} of {f(a['context']['mean_docs_in'], 1)}",
                    a.get("trap_false_hits", 0), a.get("guard_rejections", 0)])
    parts.append("### Route mix, compression, context, guards\n\n" + table(
        ["arm", "route mix", "mean compression ratio", "docs kept", "trap false hits", "guard rejections"], ops))
    prov = [["summary file", f"`eval/results/{name}`"], ["backend / models", f"{m['backend']} {m.get('models')}"],
            ["billed as", f"{m.get('billing')} (prices checked {m.get('prices_checked_on')})"],
            ["trace", f"`{m['trace']}` sha256 `{str(m['trace_sha256'])[:16]}…`, {m['rows_used']} rows"],
            ["policy config_hash", f"`{m['config_hash']}` {m.get('overrides') or ''}"],
            ["components", ", ".join(f"{k}={v}" for k, v in (m.get("components") or {}).items())],
            ["judge", f"{(m.get('judge') or {}).get('judge')} ({m.get('judge_mode')}, pairwise ≤ {m.get('pairwise_max')}"
                      f"/arm, correct = grade ≥ {m.get('correct_threshold')})"],
            ["git / generated", f"{m.get('git_commit')} / {m.get('generated_at')}"]]
    if s.get("judge_human_agreement"):
        h = s["judge_human_agreement"]
        prov.append(["judge-human agreement", f"{f(h.get('agreement'), 3)} incl. ties, {f(h.get('agreement_excl_ties'), 3)}"
                                              f" excl. ties, κ {f(h.get('kappa'), 3)} (n={h.get('n_pairwise')})"])
    parts.append("### Provenance\n\n" + table(["", ""], prov))
    return "\n\n".join(parts)


def render_sensitivity() -> Optional[str]:
    files = sorted(RESULTS.glob("ab_summary_dup*.json"))
    real = [p for p in files if "mock" not in p.name]
    files = real or files
    main, _ = pick_ab()
    entries = [(json.loads(p.read_text()), p.name) for p in files]
    if main and (not real or main["meta"]["backend"] != "mock"):
        entries.append((main, "headline"))
    if len(entries) < 2:
        return None
    rows = []
    for s, name in sorted(entries, key=lambda e: e[0]["meta"].get("trace_stats", {}).get("dup_rate", 0)):
        arms = s["arms"]
        last = next((a for a in ("A5", "A4", "A3", "A2") if a in arms), None)
        if not last:
            continue
        a = arms[last]
        miss = a.get("savings_on_misses_pct")
        rows.append([f(100 * s["meta"]["trace_stats"]["dup_rate"], 0) + "%", last + incomplete(a),
                     f(a.get("savings_pct"), 1) + ci(a.get("savings_ci"), 1), f(miss, 1), f(a["hit_rate"]["total"], 1),
                     f(a["false_hit_rate"], 2), f(a.get("quality", {}).get("retained"), 1), f"`{name}`"])
    return "### Sensitivity to the duplicate rate\n\n" + table(
        ["dup rate", "arm", "savings % [CI]", "savings on cache misses %", "hit %", "false-hit %",
         "quality retained %", "source"], rows) + ("\n\nAt 0% duplicates every saving must come from context trimming, "
                                                   "compression and routing (the honesty check).")


# ------------------------------------------------------------------------------------------- other results
def render_ci(g: dict, b: Optional[dict]) -> str:
    v = lambda x: f(x, 4) if isinstance(x, float) and abs(x) < 1 else f(x)  # noqa: E731
    rows = [[c["check"], v(c["value"]), v(c["baseline"]), c["limit"], f"**{c['status']}**"] for c in g["checks"]]
    m = g["metrics"]
    txt = (f"**{'PASS' if g['passed'] else 'FAIL'}** on `{g['meta']['subset']}` ({m['n']} rows, backend "
           f"{g['meta']['backend']}, judge {g['meta']['judge']}, τ = {m['tau']}, config `{m['config_hash']}`).\n\n"
           + table(["check", "value", "baseline", "limit", "status"], rows)
           + f"\n\nHit rate {f(m['hit_rate_pct'], 1)}%, savings {f(m['savings_pct'], 1)}%{ci(m.get('savings_ci'), 1)}, "
             f"false hits {m['false_hits']} (traps {m['trap_false_hits']}).")
    if b:
        txt += f" Baseline generated {b.get('generated_at')}."
    return txt


def render_threshold(t: dict) -> str:
    rows = []
    for mode, r in t["recommended"].items():
        nog = (t.get("recommended_without_guards") or {}).get(mode, {})
        for rend in ("templated", "filled"):
            if rend in r:
                x = r[rend]
                rows.append([mode, f(r.get("budget_false_hit_rate"), 3), f(r.get("tau"), 2), rend,
                             f(100 * x["hit_rate"], 1), f(100 * x["false_hit_rate"], 2)
                             + (ci([100 * v for v in x["false_hit_rate_ci95"]], 2) if x.get("false_hit_rate_ci95") else ""),
                             f(100 * x.get("precision_per_hit", float("nan")), 1), f(nog.get("tau"), 2)])
    return (f"Embedding model `{t.get('embedding_model')}`; per-request false-hit budgets per mode.\n\n"
            + table(["mode", "false-hit budget", "τ (with guards)", "rendering", "hit %", "false-hit % [CI]",
                     "precision per hit %", "τ without guards"], rows))


def render_compression(c: dict) -> str:
    rows = [[k, f(v.get("ratio"), 2), f(100 * v["kept_share"], 1) if _num(v.get("kept_share")) else "–",
             f(100 * v["evidence_retention"], 1) if _num(v.get("evidence_retention")) else "–",
             f(v.get("latency_ms_p50"), 1)] for k, v in c["methods"].items()]
    rec = [[mode, r.get("context_budget_tokens"), f(r.get("compression_rate")), f(r.get("ratio"), 2),
            f(100 * r["evidence_retention"], 1) + (ci([100 * v for v in r["retention_ci95"]], 1)
                                                   if r.get("retention_ci95") else ""), f(r.get("bar"))]
           for mode, r in c.get("recommendation", {}).items()]
    return (table(["method", "compression ratio", "tokens kept %", "evidence retained %", "p50 ms"], rows)
            + "\n\nRecommended settings:\n\n"
            + table(["mode", "context budget", "compression rate", "ratio", "evidence retained % [CI]", "bar"], rec))


def render_router(r: dict) -> str:
    rows = []
    for cat, c in r["categories"].items():
        pw = c.get("pairwise") or {}
        rows.append([cat, "**yes**" if c.get("allow") else "no", c.get("n"), f(c.get("diff"), 2) + ci(c.get("ci"), 2),
                     f(100 * pw["non_inferior_rate"], 0) + ci([100 * v for v in pw["non_inferior_ci"]], 0, "")
                     if pw.get("non_inferior_rate") is not None else "–",
                     f(100 * c["savings_if_allowed"], 0) if _num(c.get("savings_if_allowed")) else "–",
                     f(100 * c["traffic_share"], 1) if _num(c.get("traffic_share")) else "–", c.get("reason", "")])
    head = (f"{'**Dry run** — ' if r.get('dry_run') else ''}backend {r.get('backend')}, judge {r.get('judge')}, "
            f"metric: {r.get('metric')}, margin {r.get('margin')}. Allowed to downshift: "
            f"{', '.join(r.get('allowed') or []) or 'none'}.")
    return head + "\n\n" + table(["category", "downshift", "n", "quality diff [CI]", "non-inferior % [CI]",
                                  "saving if allowed %", "traffic %", "reason"], rows)


def render_loadtest(lt: dict) -> str:
    st = lt.get("setup", {})
    e2e, hit, miss = lt.get("end_to_end_ms", {}), lt.get("hit_path", {}), lt.get("miss_path", {})
    rows = [["throughput", f"{f(lt.get('throughput_rps'), 1)} req/s ({st.get('users')} users, {st.get('duration')}, "
                           f"mock upstream {st.get('mock_latency_ms')} ms)"],
            ["requests / failures", f"{f(lt.get('requests'))} / {f(lt.get('failures'))}"],
            ["end-to-end p50 / p99", f"{f(e2e.get('p50'), 1)} / {f(e2e.get('p99'), 1)} ms"],
            ["cache-hit path p50 / p99", f"{f(hit.get('client_ms', {}).get('p50'), 1)} / "
                                         f"{f(hit.get('client_ms', {}).get('p99'), 1)} ms"],
            ["miss path p50 / p99", f"{f(miss.get('client_ms', {}).get('p50'), 1)} / "
                                    f"{f(miss.get('client_ms', {}).get('p99'), 1)} ms"],
            ["CostGuard overhead on misses p50 / p99", f"{f(miss.get('overhead_ms', {}).get('p50'), 1)} / "
                                                       f"{f(miss.get('overhead_ms', {}).get('p99'), 1)} ms"]]
    return table(["", ""], rows)


# ------------------------------------------------------------------------------------------- main
def build() -> str:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    parts = ["# Results", f"_Generated by `python -m eval.report` from `eval/results/*.json` at {now}. "
                          "Do not edit by hand; re-run the report instead._"]
    ab, name = pick_ab()
    parts.append("## A/B: cumulative ablation on the frozen trace")
    parts.append(safe(render_ab, ab, name, title="") if ab else "_No A/B summary yet: run `python -m eval.run_ab`._")
    sens = render_sensitivity() if ab else None
    if sens:
        parts.append(sens)
    g = load("ci_gate.json")
    if g:
        parts.append("## CI eval gate")
        parts.append(safe(render_ci, g, load("ci_baseline.json"), title=""))
    t = load("threshold_sweep.json")
    if t:
        parts.append("## Semantic-cache threshold calibration")
        parts.append("> Hit rates come from a paraphrase-heavy replay (Bitext: 27 intents, many phrasings each), so treat them "
                     "as an **upper bound**. Savings claims come from the frozen-trace A/B, not from this sweep. False-hit rates "
                     "are per request (wrong hits / all requests).")
        parts.append(safe(render_threshold, t, title=""))
    c = load("compression_eval.json")
    if c:
        parts.append("## Context trimming and compression")
        parts.append(safe(render_compression, c, title=""))
    r = load("router_gate.json") or load("router_gate_dryrun.json")
    if r:
        parts.append("## Router eval gate")
        parts.append(safe(render_router, r, title=""))
    lt = load("loadtest.json") or load("loadtest_quick.json")
    if lt:
        parts.append("## Load test (mock upstream)")
        parts.append(safe(render_loadtest, lt, title=""))
    return "\n\n".join(p for p in parts if p) + "\n"


def _billing(b: Any) -> str:
    if isinstance(b, dict):
        return " / ".join(str(b[k]) for k in ("strong", "cheap") if k in b) + " (strong / cheap)"
    return str(b)


def readme_block() -> str:
    """Compact headline for the README: the A/B arms, the router gate verdict and proxy overhead. Generated, never typed."""
    ab, name = pick_ab()
    if not ab:
        return "_No A/B results yet: run `python -m eval.run_ab`, then `python -m eval.report`._"
    m, arms = ab["meta"], ab["arms"]
    parts = []
    if m.get("backend") == "mock":
        parts.append("> **Mock backend: placeholder numbers.** Synthetic answers and a token-overlap judge; real runs "
                     "replace this block.")
    elif m.get("backend") == "mlx":
        parts.append("> **Local stand-in models.** Qwen2.5-7B (strong) and Qwen2.5-1.5B (cheap) on Apple silicon, billed "
                     f"at {_billing(m.get('billing'))} list prices (a ~12× price gap; Sonnet 5.5 → Haiku 4.5 is 2×, so router "
                     "savings on Anthropic will be smaller). Cache savings do not depend on the model pair.")
    judge = (m.get("judge") or {}).get("judge", "–")
    rows = []
    for arm, a in arms.items():
        q = a.get("quality", {})
        rows.append([f"**{arm}**{incomplete(a)}", a["description"],
                     f(a.get("savings_pct"), 1) + ci(a.get("savings_ci"), 1),
                     f(a["hit_rate"]["total"], 1), f(a["false_hit_rate"], 2),
                     "100 (ref)" if arm == "A0" and q else f(q.get("retained"), 1) + ci(q.get("retained_ci"), 1)])
    parts.append(table(["arm", "levers on (cumulative)", "cost saved % vs A0 [95% CI]", "cache hit %",
                        "false-hit % of requests", "quality retained % [95% CI]"], rows))
    parts.append(f"{m['rows_used']} requests from the frozen trace (sha256 `{str(m['trace_sha256'])[:12]}…`), backend "
                 f"`{m['backend']}`, judge `{judge}`, paired cluster bootstrap. Source: `eval/results/{name}`.")
    other = load("ab_summary_mlx.json") if name == "ab_summary.json" else None
    if other and other.get("arms", {}).get("A5"):
        a5, q5 = other["arms"]["A5"], other["arms"]["A5"].get("quality", {})
        parts.append(f"Also measured on local stand-in models ({other['meta'].get('backend')}: Qwen2.5-7B / 1.5B, billed at "
                     f"{_billing(other['meta'].get('billing'))} prices): A5 saved {f(a5.get('savings_pct'), 1)}%"
                     f"{ci(a5.get('savings_ci'), 1)} at {f(q5.get('retained'), 1)}% quality retained"
                     f"{ci(q5.get('retained_ci'), 1)}. Source: `eval/results/ab_summary_mlx.json`.")
    r = load("router_gate.json") or load("router_gate_dryrun.json")
    if r:
        allowed = ", ".join(r.get("allowed") or []) or "none"
        line = (f"**Router gate** ({'dry run, ' if r.get('dry_run') else ''}backend `{r.get('backend')}`): "
                f"categories allowed to downshift to the cheap tier: {allowed}.")
        diffs = [c["diff"] for c in r.get("categories", {}).values() if _num(c.get("diff"))]
        if diffs and not r.get("dry_run"):
            line += (f" Cheap − strong quality per category: {f(min(diffs), 1)} to {f(max(diffs), 1)} points; a category "
                     f"needs n ≥ {r.get('min_n')} and a 95% CI lower bound ≥ −{f(r.get('margin'), 0)} points "
                     f"(judge `{r.get('judge')}`).")
        parts.append(line)
    lt = load("loadtest.json")
    if lt:
        miss, st = lt.get("miss_path", {}), lt.get("setup", {})
        parts.append(f"**Proxy overhead** (load test, {st.get('users')} users, mock upstream): "
                     f"{f(lt.get('throughput_rps'), 0)} req/s, 0 failures of {f(lt.get('requests'))}" if not lt.get("failures")
                     else f"**Proxy overhead** (load test): {f(lt.get('failures'))} failures of {f(lt.get('requests'))}")
        parts[-1] += (f"; CostGuard's own overhead on a cache miss p50 / p99 = {f(miss.get('overhead_ms', {}).get('p50'), 1)}"
                      f" / {f(miss.get('overhead_ms', {}).get('p99'), 1)} ms.")
    parts.append("Full tables, latency, waterfall and provenance: [docs/RESULTS.md](docs/RESULTS.md).")
    return "\n\n".join(parts)


def update_readme(path: Path = README) -> bool:
    if not path.exists():
        return False
    text = path.read_text()
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    block = safe(readme_block, title="")
    path.write_text(f"{head}{README_START}\n<!-- generated by `python -m eval.report`; do not edit by hand -->\n\n"
                    f"{block}\n\n{README_END}{tail}")
    return True


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate docs/RESULTS.md from eval/results/*.json")
    ap.add_argument("--out", default=str(OUT), help="output path, or - for stdout")
    ap.add_argument("--no-readme", action="store_true", help="do not touch the README results block")
    args = ap.parse_args(argv)
    md = build()
    if args.out == "-":
        print(md)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f"wrote {args.out} ({len(md.splitlines())} lines)")
        if not args.no_readme and update_readme():
            print(f"updated results block in {README.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
