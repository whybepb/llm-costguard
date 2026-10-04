"""Offline measurement of context optimisation and compression (no LLM calls by default).

    python -m eval.compression_eval              # full run, writes eval/results/compression_eval.json
    python -m eval.compression_eval --fast       # skip LLMLingua-2 (no torch model load)
    python -m eval.compression_eval --with-llm   # also generate + judge a small subset (needs eval/judge.py and a backend)

For every seed question in `eval.kb.kb_questions()` we retrieve the naive top-8 context and run each method on it,
measuring tokens before/after (o200k), the compression ratio and the stage latency (p50/p99).

Quality proxy, evidence retention: the share of answerable questions whose key facts (the numbers and short phrases
the reference answer depends on) still appear in the optimised context, among questions whose facts were present
in the full context. If the fact is cut, the answer must degrade, so retention is an upper bound on quality kept.
It is cheap, deterministic and LLM-free. It does not see paraphrase or reasoning failures, and it scores token-dropping
methods harshly when they split a phrase. Confirm the chosen setting with the judge (`--with-llm`, or the A/B harness).

The recommendation picks, per mode, the most aggressive (fewest tokens sent) (context_budget_tokens, compression_rate)
pair from a grid whose retention clears the mode's bar: quality >= 98%, balanced >= 95%, economy >= 90%.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from costguard.config import ROOT
from costguard.context.compress import HeuristicCompressor, LLMLingua2Compressor, PassthroughCompressor
from costguard.context.optimizer import (CROSS_ENCODER_MODEL, DEFAULT_GAP, CrossEncoderScorer,
                                         RerankContextOptimizer, Scorer)
from costguard.pipeline import format_docs
from costguard.tokens import count_text
from eval.kb import facts_present, kb_questions, retrieve

OUT = ROOT / "eval" / "results" / "compression_eval.json"
MODE_BARS = {"quality": 0.98, "balanced": 0.95, "economy": 0.90}
MODE_MIN_TOKENS = {"quality": 400, "balanced": 400, "economy": 250}   # compression_min_tokens in policy.yaml
GRID_BUDGETS = [None, 2000, 1600, 1200, 1000, 800, 600, 400]
GRID_RATES = [None, 0.7, 0.5, 0.33]


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)   # peak RSS


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    try:
        from eval.stats import proportion_ci   # owned by the eval workstream, if present
        return proportion_ci(k, n)
    except Exception:
        if n == 0:
            return (float("nan"),) * 3
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return p, max(0.0, c - h), min(1.0, c + h)


class CachedScorer(Scorer):
    """Wraps the cross-encoder so the budget/rate grid re-uses one scoring pass per question."""
    kind = "cross-encoder"

    def __init__(self, inner: Scorer):
        super().__init__(inner.name)
        self.inner, self.cache = inner, {}

    def load(self):
        return self.inner.load()

    def score(self, query, docs):
        key = (query, tuple(docs))
        if key not in self.cache:
            self.cache[key] = self.inner.score(query, docs)
        return self.cache[key]


# ----------------------------------------------------------------------------------------------- methods

def truncate_naive(docs: list[str], budget: int) -> list[str]:
    """Baseline without a reranker: keep docs in retrieval order while they fit."""
    out = []
    for d in docs:
        if count_text(format_docs(out + [d])) > budget:
            break
        out.append(d)
    return out or docs[:1]


def _timed_truncate(docs: list[str], budget: int) -> tuple[str, float, dict]:
    t = time.perf_counter()
    kept = truncate_naive(docs, budget)
    return format_docs(kept), (time.perf_counter() - t) * 1000, {"docs_kept": len(kept)}


def run_pipeline(q: str, docs: list[str], opt: Optional[RerankContextOptimizer], budget: Optional[int],
                 comp, rate: Optional[float], min_tokens: int = 0) -> tuple[str, float, dict]:
    """Mimic pipeline stages 3+4 exactly: optimise docs, format, compress the block if long enough."""
    t = time.perf_counter()
    sent = docs
    extra = {}
    if opt is not None:
        r = opt.optimize(q, docs, budget, None)
        sent = r.docs
        extra["docs_kept"] = len(sent)
    block = format_docs(sent)
    if comp is not None and rate is not None and count_text(block) >= min_tokens:
        block = comp.compress(block, rate, q).text
    return block, (time.perf_counter() - t) * 1000, extra


# ----------------------------------------------------------------------------------------------- evaluation

def evaluate(items: list[dict], fn: Callable[[dict], tuple[str, float, dict]]) -> dict:
    tb, ta, lat, rows = [], [], [], []
    for it in items:
        text, ms, extra = fn(it)
        before, after = it["tokens_before"], count_text(text)
        tb.append(before)
        ta.append(after)
        lat.append(ms)
        ok = facts_present(it["key_facts"], text) if it["answerable"] else None
        ok_len = facts_present(it["key_facts"], text, lenient=True) if it["answerable"] else None
        rows.append({"id": it["id"], "type": it["type"], "before": before, "after": after,
                     "retained": ok, "retained_lenient": ok_len, **extra})
    elig = [r for r, it in zip(rows, items) if it["answerable"] and it["baseline_ok"]]
    k = sum(bool(r["retained"]) for r in elig)
    p, lo, hi = wilson(k, len(elig))
    by_type = {}
    for t in ("single", "multi_hop"):
        sub = [r for r in elig if r["type"] == t]
        by_type[t] = round(sum(bool(r["retained"]) for r in sub) / len(sub), 4) if sub else None
    return {
        "tokens_before": int(sum(tb)), "tokens_after": int(sum(ta)),
        "tokens_after_mean": round(float(np.mean(ta)), 1),
        "ratio": round(sum(tb) / max(1, sum(ta)), 3),
        "kept_share": round(sum(ta) / max(1, sum(tb)), 4),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 2),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 2),
        "evidence_retention": round(p, 4), "retention_ci95": [round(lo, 4), round(hi, 4)],
        "evidence_retention_lenient": round(sum(bool(r["retained_lenient"]) for r in elig) / max(1, len(elig)), 4),
        "retained": k, "eligible": len(elig), "retention_by_type": by_type,
        "missed": [r["id"] for r in elig if not r["retained"]],
        "unanswerable_tokens_after_mean": round(float(np.mean([r["after"] for r in rows
                                                                if r["type"] == "unanswerable"] or [0])), 1),
    }


def build_items(k: int) -> list[dict]:
    items = []
    for q in kb_questions():
        docs = retrieve(q["query"], k)
        block = format_docs(docs)
        answerable = bool(q["key_facts"])
        items.append({**q, "docs": docs, "block": block, "tokens_before": count_text(block),
                      "answerable": answerable,
                      "baseline_ok": answerable and facts_present(q["key_facts"], block)})
    return items


def recommend(grid: list[dict]) -> dict:
    out = {}
    for mode, bar in MODE_BARS.items():
        ok = [g for g in grid if g["mode_min"] == MODE_MIN_TOKENS[mode] and g["evidence_retention"] >= bar]
        if not ok:
            out[mode] = None
            continue
        pick = lambda pool: min(pool, key=lambda g: (g["tokens_after"], -g["evidence_retention"]))  # noqa: E731
        best = pick(ok)
        out[mode] = {"context_budget_tokens": best["budget"], "compression": best["rate"] is not None,
                     "compression_rate": best["rate"], "compression_min_tokens": MODE_MIN_TOKENS[mode],
                     "evidence_retention": best["evidence_retention"], "retention_ci95": best["retention_ci95"],
                     "kept_share": best["kept_share"], "ratio": best["ratio"], "bar": bar}
        no_comp = [g for g in ok if g["rate"] is None]
        if no_comp and best["rate"] is not None:   # the conservative alternative: rerank + budget only
            alt = pick(no_comp)
            out[mode]["alternative_without_compression"] = {
                "context_budget_tokens": alt["budget"], "evidence_retention": alt["evidence_retention"],
                "kept_share": alt["kept_share"], "ratio": alt["ratio"]}
    return out


def maybe_llm_check(items: list[dict], configs: dict, n: int) -> Optional[dict]:
    """Optional: generate answers with the engine's provider for a small subset and grade them with eval.judge."""
    try:
        from eval.judge import get_judge
    except Exception as e:
        print(f"--with-llm skipped: eval/judge.py not available ({type(e).__name__})")
        return None
    from costguard.config import Settings, load_policy
    from costguard.pipeline import build_messages
    from costguard.providers import make_provider
    settings = Settings.from_env()
    policy = load_policy(settings.policy_path)
    provider = make_provider(settings)
    model = policy.model_id(settings.backend, "strong")
    judge = get_judge(settings)
    sub = [it for it in items if it["answerable"]][:n]
    res = {}
    for name, fn in configs.items():
        scores = []
        for it in sub:
            block = fn(it)[0]
            msgs = build_messages(policy.system_prompt, [], it["query"], block)
            ans = provider.complete(msgs, model, policy.default_max_tokens, 0.0).text
            scores.append(judge.grade(it["query"], ans, it["reference"]))
        res[name] = {"n": len(scores), "mean_grade": round(float(np.mean(scores)), 4)}
    base = res.get("passthrough", {}).get("mean_grade") or float("nan")
    for v in res.values():
        v["quality_retained"] = round(v["mean_grade"] / base, 4) if base else None
    return {"backend": settings.backend, "model": model, "results": res}


def markdown_tables(out: dict) -> str:
    """Render the results JSON as the markdown tables used in docs/components/context_and_compression.md."""
    lines = ["| Method | Tokens kept | Ratio | Evidence retention (95% CI) | Lenient | p50 ms | p99 ms |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for name, r in out["methods"].items():
        lo, hi = r["retention_ci95"]
        lines.append(f"| {name} | {r['kept_share']:.0%} | {r['ratio']:.2f}x | {r['evidence_retention']:.1%} "
                     f"({lo:.0%}-{hi:.0%}) | {r['evidence_retention_lenient']:.1%} | {r['latency_ms_p50']:.1f} | "
                     f"{r['latency_ms_p99']:.1f} |")
    lines += ["", "| Mode | Bar | context_budget_tokens | compression_rate | Tokens kept | Retention (95% CI) |",
              "|---|---:|---:|---|---:|---|"]
    for mode, r in out["recommendation"].items():
        if r:
            lines.append(f"| {mode} | {r['bar']:.0%} | {r['context_budget_tokens']} | "
                         f"{r['compression_rate'] if r['compression'] else 'off'} | {r['kept_share']:.0%} | "
                         f"{r['evidence_retention']:.1%} ({r['retention_ci95'][0]:.0%}-{r['retention_ci95'][1]:.0%}) |")
    lines += ["", "| Gap | Retention | Ratio | Docs kept (mean of 8) |", "|---:|---:|---:|---:|"]
    for g in out["gap_sweep"]:
        lines.append(f"| {g['gap']} | {g['evidence_retention']:.1%} | {g['ratio']:.2f}x | {g['docs_kept_mean']} |")
    return "\n".join(lines)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fast", action="store_true", help="skip LLMLingua-2")
    ap.add_argument("--no-cpu-timing", action="store_true", help="skip the extra LLMLingua-2 CPU latency run")
    ap.add_argument("--llmlingua-large", action="store_true",
                    help="also run the xlm-roberta-large LLMLingua-2 model (~2.2 GB download)")
    ap.add_argument("--with-llm", action="store_true", help="also generate+judge a subset (needs eval/judge.py)")
    ap.add_argument("--llm-n", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--markdown", action="store_true", help="only print markdown tables from an existing --out JSON")
    a = ap.parse_args(argv)
    if a.markdown:
        print(markdown_tables(json.loads(a.out.read_text())))
        return {}

    t_all = time.perf_counter()
    rss0 = _rss_mb()
    items = build_items(a.k)
    rss_retrieval = _rss_mb()
    ce = CrossEncoderScorer()
    t = time.perf_counter()
    ce.load()
    ce_load_ms = (time.perf_counter() - t) * 1000
    rss_ce = _rss_mb()
    opt = RerankContextOptimizer(scorer=ce, fallbacks=[])
    heur = {s: HeuristicCompressor(scorer=s) for s in ("lexical", "hybrid")}
    print(f"{len(items)} questions, {sum(i['answerable'] for i in items)} answerable, "
          f"{sum(i['baseline_ok'] for i in items)} with all key facts in the top-{a.k} context; "
          f"context tokens median {int(np.median([i['tokens_before'] for i in items]))}")

    configs: dict[str, Callable] = {
        "passthrough": lambda it: (it["block"], 0.0, {}),
    }
    for b in (400, 800, 1200):
        configs[f"truncate@{b} (no rerank)"] = lambda it, b=b: _timed_truncate(it["docs"], b)
    configs[f"rerank gap={opt.gap:g}"] = lambda it: run_pipeline(it["query"], it["docs"], opt, None, None, None)
    for b in (400, 800, 1200):
        configs[f"rerank+budget@{b}"] = lambda it, b=b: run_pipeline(it["query"], it["docs"], opt, b, None, None)
    for s, hc in heur.items():
        for r in ((0.33, 0.5, 0.7) if s == "lexical" else (0.33, 0.5)):
            configs[f"heuristic-{s}@{r}"] = lambda it, hc=hc, r=r: run_pipeline(it["query"], it["docs"], None, None, hc, r)
    configs["rerank@1200+heuristic@0.5"] = lambda it: run_pipeline(it["query"], it["docs"], opt, 1200, heur["lexical"], 0.5)
    configs["rerank@800+heuristic@0.33"] = lambda it: run_pipeline(it["query"], it["docs"], opt, 800, heur["lexical"], 0.33)

    ll_meta = None
    if not a.fast:
        ll = LLMLingua2Compressor()
        lld = LLMLingua2Compressor(force_reserve_digit=True)
        t = time.perf_counter()
        ll.load()
        lld._pc, lld.device, lld.load_ms = ll._pc, ll.device, ll.load_ms   # share one model instance
        ll_meta = {"model": ll.model_name, "device": ll.device, "load_ms": round((time.perf_counter() - t) * 1000, 1),
                   "peak_rss_mb_after_load": _rss_mb()}
        ll.compress(items[0]["block"], 0.5)   # warm-up (first call compiles kernels)
        for r in (0.33, 0.5):
            configs[f"llmlingua2@{r}"] = lambda it, r=r: run_pipeline(it["query"], it["docs"], None, None, ll, r)
        configs["llmlingua2+digits@0.5"] = lambda it: run_pipeline(it["query"], it["docs"], None, None, lld, 0.5)
        configs["rerank@1200+llmlingua2@0.5"] = lambda it: run_pipeline(it["query"], it["docs"], opt, 1200, ll, 0.5)
        if a.llmlingua_large:
            big = LLMLingua2Compressor(model="large")
            t = time.perf_counter()
            big.load()
            big.compress(items[0]["block"], 0.5)
            ll_meta["large"] = {"model": big.model_name, "device": big.device,
                                "load_ms": round((time.perf_counter() - t) * 1000, 1), "peak_rss_mb_after_load": _rss_mb()}
            for r in (0.33, 0.5):
                configs[f"llmlingua2-large@{r}"] = lambda it, r=r: run_pipeline(it["query"], it["docs"], None, None, big, r)

    results = {}
    for name, fn in configs.items():
        results[name] = evaluate(items, fn)
        r = results[name]
        print(f"  {name:32s} ratio {r['ratio']:5.2f}x  retention {r['evidence_retention']:.3f} "
              f"(lenient {r['evidence_retention_lenient']:.3f})  "
              f"p50 {r['latency_ms_p50']:7.1f} ms  p99 {r['latency_ms_p99']:7.1f} ms")

    if ll_meta and not a.no_cpu_timing and ll.device != "cpu":
        cpu = LLMLingua2Compressor(device="cpu")
        t = time.perf_counter()
        cpu.load()
        load = (time.perf_counter() - t) * 1000
        cpu.compress(items[0]["block"], 0.5)
        lat = []
        for it in items:
            cpu.compress(it["block"], 0.5)
            lat.append(cpu.last_ms)
        ll_meta["cpu"] = {"load_ms": round(load, 1), "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
                          "latency_ms_p99": round(float(np.percentile(lat, 99)), 1)}
        del cpu

    # gap sweep for the relative dynamic-k rule (no budget), cross-encoder scores computed once
    cached = CachedScorer(ce)
    gap_sweep = []
    for g in (2, 3, 4, 5, 6, 8, 10):
        o = RerankContextOptimizer(scorer=cached, gap=g, fallbacks=[])
        r = evaluate(items, lambda it: run_pipeline(it["query"], it["docs"], o, None, None, None))
        gap_sweep.append({"gap": g, "evidence_retention": r["evidence_retention"], "ratio": r["ratio"],
                          "docs_kept_mean": round(float(np.mean([len(o.optimize(it["query"], it["docs"]).docs)
                                                                 for it in items])), 2)})

    # recommendation grid: (budget, heuristic rate) under each mode's compression_min_tokens
    grid = []
    og = RerankContextOptimizer(scorer=cached, fallbacks=[])
    for mode_min in sorted(set(MODE_MIN_TOKENS.values())):
        for b in GRID_BUDGETS:
            for rate in GRID_RATES:
                r = evaluate(items, lambda it: run_pipeline(it["query"], it["docs"], og, b, heur["lexical"], rate,
                                                            mode_min))
                grid.append({"mode_min": mode_min, "budget": b, "rate": rate,
                             **{k: r[k] for k in ("tokens_after", "kept_share", "ratio", "evidence_retention",
                                                  "retention_ci95")}})
    rec = recommend(grid)

    llm = maybe_llm_check(items, {k: configs[k] for k in ("passthrough", "rerank@1200+heuristic@0.5",
                                                           "rerank@800+heuristic@0.33") if k in configs},
                          a.llm_n) if a.with_llm else None

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "python -m eval.compression_eval",
        "setup": {"questions": len(items), "answerable": sum(i["answerable"] for i in items),
                  "eligible": sum(i["baseline_ok"] for i in items), "k": a.k,
                  "retriever": "fastembed BAAI/bge-small-en-v1.5 top-k (eval/kb.py)",
                  "reranker": CROSS_ENCODER_MODEL, "reranker_load_ms": round(ce_load_ms, 1),
                  "default_gap": DEFAULT_GAP["cross-encoder"], "tokenizer": "o200k_base",
                  "context_tokens_median": int(np.median([i["tokens_before"] for i in items])),
                  "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
                  "peak_rss_mb": {"start": rss0, "after_retrieval_index": rss_retrieval,
                                  "after_cross_encoder": rss_ce},
                  "metric": "evidence_retention = share of answerable questions (key facts present in the full "
                            "top-k context) whose key facts all survive in the optimised context"},
        "methods": results,
        "llmlingua2": ll_meta,
        "gap_sweep": gap_sweep,
        "grid": grid,
        "recommendation": rec,
        "llm_check": llm,
        "runtime_s": round(time.perf_counter() - t_all, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\nRecommended per-mode settings (most aggressive meeting the evidence-retention bar):")
    for mode, r in rec.items():
        if r is None:
            print(f"  {mode:9s} no grid point meets {MODE_BARS[mode]:.0%}")
            continue
        comp = f"rate {r['compression_rate']}" if r["compression"] else "off"
        print(f"  {mode:9s} budget {r['context_budget_tokens']}, compression {comp} -> keeps {r['kept_share']:.0%} "
              f"of context tokens ({r['ratio']}x), retention {r['evidence_retention']:.1%} "
              f"(95% CI {r['retention_ci95'][0]:.1%}-{r['retention_ci95'][1]:.1%}; bar {r['bar']:.0%})")
    shown = a.out.relative_to(ROOT) if a.out.is_relative_to(ROOT) else a.out
    print(f"\nwrote {shown} in {out['runtime_s']} s")
    return out


if __name__ == "__main__":
    main()
