"""Cumulative-ablation A/B on the frozen trace.

    python -m eval.run_ab --backend mock --limit 50                      # seconds, $0 (heuristic judge)
    python -m eval.run_ab --backend anthropic --estimate-only            # pre-flight: new generations + est. $
    python -m eval.run_ab --backend anthropic --yes --workers 4          # the real run (paid; needs --yes)
    python -m eval.run_ab --trace eval/data/trace_v1_dup00.jsonl --arms A0,A5 --backend anthropic --yes
    python -m eval.run_ab --backend mlx --replay --out-dir /tmp/ab           # reproduce a committed run from cassettes, $0

Arms (cumulative, ordered by quality risk; each is the balanced mode with later levers switched off):
  A0 baseline: everything off, strong tier    A1 + exact cache    A2 + semantic cache (policy tau, guards)
  A3 + context rerank / dynamic-k             A4 + compression    A5 + router (gated) = the full balanced mode
  optional: B = economy mode, Q = quality mode
Method:
  - one engine (costguard.factory.build_engine); per arm a deep-copied policy with modes[...] overridden, caches
    flushed, requests replayed sequentially in trace order with costguard options from the trace row
  - upstream calls go through eval/cassettes/<backend>_ab.jsonl (auto mode): calls identical across arms are made
    once, and re-runs cost $0. Every TraceRecord is logged to eval/results/ab_<backend>.sqlite.
  - savings are PAIRED: each arm's per-item cost vs A0's actual per-item cost on the same items (A0 has real usage).
    The record-level baseline_cost_usd (estimated strong-tier cost of the full prompt) is kept as a secondary column.
  - quality: judge.grade against the item reference for every arm (quality retained = arm / A0, paired cluster
    bootstrap CI) plus judge.pairwise(A0, arm) with position swap on up to --pairwise-max differing items per arm.
    Identical answers are not judged (guaranteed tie).
  - a served cache hit is a false hit when the cached entry came from a request in a different trace cluster.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

from costguard import factory
from costguard.config import ROOT, ModePolicy, Policy, Settings, load_policy
from costguard.pricing import PriceBook
from costguard.providers.cassette import CassetteProvider, call_key
from costguard.schemas import ChatMessage, ChatRequest, Completion, CostGuardOptions, Usage
from costguard.tokens import count_messages, count_text

from .judge import LOCAL_BACKENDS, HeuristicJudge, Judge, ReplayOnlyProvider, get_judge, human_agreement
from .stats import paired_bootstrap, percentile, proportion_ci, ratio_bootstrap

RESULTS = ROOT / "eval" / "results"
CASSETTES = ROOT / "eval" / "cassettes"
DEFAULT_TRACE = ROOT / "eval" / "data" / "trace_v1.jsonl"
HIT = ("exact", "semantic")
CORRECT_AT = 0.75          # grade >= 4 on the 1-5 rubric counts as a correct answer

CUMULATIVE = ["A0", "A1", "A2", "A3", "A4", "A5"]
ARMS: dict[str, tuple[str, str, dict]] = {      # arm -> (description, mode slot, ModePolicy overrides)
    "A0": ("baseline: all levers off, strong tier", "balanced",
           dict(exact_cache=False, semantic_cache=False, context=False, compression=False, router=False)),
    "A1": ("+ exact cache", "balanced", dict(exact_cache=True, semantic_cache=False, context=False, compression=False,
                                             router=False)),
    "A2": ("+ semantic cache (policy tau, guards)", "balanced",
           dict(exact_cache=True, semantic_cache=True, context=False, compression=False, router=False)),
    "A3": ("+ context rerank / dynamic-k", "balanced",
           dict(exact_cache=True, semantic_cache=True, context=True, compression=False, router=False)),
    "A4": ("+ compression", "balanced",
           dict(exact_cache=True, semantic_cache=True, context=True, compression=True, router=False)),
    "A5": ("+ router (gated) = full balanced mode", "balanced",
           dict(exact_cache=True, semantic_cache=True, context=True, compression=True, router=True)),
    "B": ("economy mode (as configured)", "economy", {}),
    "Q": ("quality mode (as configured)", "quality", {}),
}
LEVER = {"A1": "exact cache", "A2": "semantic cache", "A3": "context trim", "A4": "compression", "A5": "router"}
PROMPT_CACHE_ENV = "COSTGUARD_ANTHROPIC_PROMPT_CACHE"


def set_provider_prompt_cache(enabled: bool) -> bool:
    """Turn the Anthropic adapter's prompt caching on/off for this process; call before any provider is built.

    Eval runs default to off: a cassette stores one usage per call key, so replay would repeat the first call's
    cache-write usage for every identical call, and cache warmth depends on call order (parallel prefetch, which
    arm ran first). Provider prompt-cache savings are reported separately, not mixed into the paired A/B."""
    os.environ[PROMPT_CACHE_ENV] = "1" if enabled else "0"
    return enabled


# ------------------------------------------------------------------------------------------- policy + engine
def apply_overrides(policy: Policy) -> tuple[Policy, dict]:
    """Deep copy of the policy with env overrides applied (never edits policy.yaml).

    COSTGUARD_TAU_OVERRIDE=<float> sets the balanced-mode semantic-cache threshold (used by the CI-gate demo/tests)."""
    pol = copy.deepcopy(policy)
    ov: dict = {}
    tau = os.environ.get("COSTGUARD_TAU_OVERRIDE")
    if tau:
        pol.modes["balanced"].tau = float(tau)
        ov["balanced.tau"] = float(tau)
    if ov:
        pol.config_hash = hashlib.sha256((policy.config_hash + json.dumps(ov, sort_keys=True)).encode()).hexdigest()[:12]
    return pol, ov


def arm_policy(base: Policy, arm: str) -> tuple[str, Policy]:
    """(mode name to request, policy copy whose mode slot is configured for this arm)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {', '.join(ARMS)}")
    _, slot, fields = ARMS[arm]
    pol = copy.deepcopy(base)
    pol.modes[slot] = dataclasses.replace(base.modes[slot], **fields)
    return slot, pol


def make_engine(settings: Settings, provider=None, with_logger: bool = False):
    """build_engine(settings), optionally with a pre-built upstream provider (keyless replay / dry run)."""
    if provider is None:
        return factory.build_engine(settings, with_logger=with_logger)
    orig = factory.make_provider
    factory.make_provider = lambda s: provider
    try:
        return factory.build_engine(settings, with_logger=with_logger)
    finally:
        factory.make_provider = orig


class RecordedCountProvider(ReplayOnlyProvider):
    """Replay-only upstream for a local backend whose token counts come from the cassette. A local provider's pre-call
    count of a prompt is exactly the input usage recorded for that prompt's call, so the counts (and with them the
    estimated baseline and the router's hardness signals) replay exactly, with no tokenizer or model weights loaded."""

    def __init__(self, name: str, store: dict[str, dict], max_tokens: int):
        super().__init__(name, "replay only", ratio=1.0)
        self.store, self.max_tokens = store, max_tokens

    def count_tokens(self, messages, model):
        rec = self.store.get(call_key(messages, model, self.max_tokens, 0.0))
        if rec is not None:
            return int(rec["usage"]["input_tokens"])
        return super().count_tokens(messages, model)


def keyless_provider(backend: str, cassette: Path, mode: str = "replay", max_tokens: Optional[int] = None):
    """Cassette replay without constructing the real upstream (no key / no MLX needed). With `max_tokens` (the policy
    default every eval request uses), a local backend counts tokens from the cassette (RecordedCountProvider)."""
    if backend in LOCAL_BACKENDS and max_tokens:
        inner = RecordedCountProvider(backend, _read_cassette(cassette), max_tokens)
    else:
        inner = ReplayOnlyProvider(backend, "replay only", ratio=1.15 if backend == "anthropic" else 1.0)
    return CassetteProvider(inner, cassette, mode)


def make_request(row: dict, arm: str, mode: str, no_cache: bool = False) -> ChatRequest:
    return ChatRequest(model="strong", messages=[ChatMessage(role="user", content=row["query"])], temperature=0.0,
                       costguard=CostGuardOptions(mode=mode, arm=arm, trace_pos=row["pos"], item_id=row.get("item_id"),
                                                  category=row.get("category"), context=list(row.get("context") or []),
                                                  no_cache=no_cache))


def run_rows(engine, policy: Policy, mode: str, arm: str, rows: list[dict],
             progress: Optional[Callable[[int, list], None]] = None, no_cache: bool = False) -> list[dict]:
    """Replay rows sequentially through `engine` under `policy` (caches flushed first). One result dict per row."""
    engine.policy = policy
    engine.reset_caches()
    by_request: dict[str, dict] = {}
    by_query: dict[str, dict] = {}
    out: list[dict] = []
    for i, row in enumerate(rows):
        base = {"pos": row["pos"], "item_id": row.get("item_id"), "cluster_id": row.get("cluster_id"),
                "intent": row.get("intent"), "category": row.get("category"), "source": row.get("source"),
                "dup_kind": row.get("dup_kind"), "pair_id": row.get("pair_id"), "query": row["query"],
                "reference": row.get("reference")}
        try:
            comp, rec = engine.handle(make_request(row, arm, mode, no_cache))
        except Exception as e:
            out.append({**base, "error": f"{type(e).__name__}: {str(e)[:200]}", "cost": 0.0, "cache_status": "error"})
            continue
        res = {**base, "request_id": rec.request_id, "answer": rec.response_text, "cache_status": rec.cache_status,
               "cache_similarity": rec.cache_similarity, "cache_guard": rec.cache_guard,
               "model_used": rec.model_used or "cache", "route_reason": rec.route_reason, "cost": rec.cost_usd,
               "est_baseline": rec.baseline_cost_usd, "in_orig": rec.input_tokens_original,
               "in_sent": rec.input_tokens_sent, "out_tokens": rec.output_tokens, "cached_in": rec.cached_input_tokens,
               "cache_write": rec.cache_write_tokens,
               "compression_ratio": rec.compression_ratio, "docs_in": rec.context_docs_in,
               "docs_kept": rec.context_docs_kept, "latency_ms": rec.latency_ms, "overhead_ms": rec.overhead_ms,
               "upstream_ms": rec.upstream_latency_ms, "stage_errors": dict(rec.stage_errors), "error": rec.error,
               "false_hit": False, "false_hit_intent": False, "hit_from_pos": None}
        if rec.cache_status in HIT:
            src = by_request.get((comp.raw or {}).get("entry_id")) or by_query.get(rec.cache_neighbor or "")
            if src is None:
                res["hit_unattributed"] = True
            else:
                res["hit_from_pos"], res["hit_from_query"] = src["pos"], src["query"]
                res["false_hit"] = src.get("cluster_id") != row.get("cluster_id")
                res["false_hit_intent"] = src.get("intent") != row.get("intent")
                res["trap_false_hit"] = res["false_hit"] and "trap" in (row.get("source"), src.get("source"))
        by_request[rec.request_id] = row
        by_query[row["query"]] = row
        out.append(res)
        if progress:
            progress(i + 1, out)
    return out


# ------------------------------------------------------------------------------------------- pre-flight
class DryRunProvider:
    """Upstream stand-in for the pre-flight: replays what the cassette already has and records every other call
    (with its exact messages) as a new generation. The real pipeline runs, so caches/routing/compression decide
    which calls happen exactly as in the real run (they depend on queries and context, not on answers)."""

    def __init__(self, backend: str, store: dict[str, dict], out_estimate: Callable[[str, int], int]):
        self.name, self.store, self.out_estimate = backend, store, out_estimate
        self.ratio = 1.15 if backend == "anthropic" else 1.0
        self.new: dict[str, tuple] = {}
        self.hits = self.misses = 0

    def count_tokens(self, messages, model):
        return int(round(count_messages(messages) * self.ratio))

    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        key = call_key(messages, model, max_tokens, temperature)
        if key in self.store:
            self.hits += 1
            return Completion(**self.store[key])
        if key not in self.new:
            self.misses += 1
            self.new[key] = (list(messages), model, max_tokens, temperature, self.count_tokens(messages, model),
                             self.out_estimate(model, max_tokens))
        _, _, _, _, n_in, n_out = self.new[key]
        return Completion(text=f"[dry-run {key[:16]}]", model=model, usage=Usage(input_tokens=n_in, output_tokens=n_out),
                          latency_ms=0.0)


def _read_cassette(path: Path) -> dict[str, dict]:
    store: dict[str, dict] = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                store[r["key"]] = r["completion"]
    return store


def _alias_for(policy: Policy, backend: str, model: str) -> str:
    return next((a for a in ("strong", "cheap") if policy.backends.get(backend, {}).get(a) == model), model)


def preflight(settings: Settings, base: Policy, rows: list[dict], arms: list[str], cassette: Path,
              judge_mode: str, pairwise_max: int, judge_model: Optional[str]) -> dict:
    store = _read_cassette(cassette)
    outs: dict[str, list[int]] = defaultdict(list)
    for c in store.values():
        outs[c.get("model", "")].append(int(c.get("usage", {}).get("output_tokens", 0)))

    def out_est(model: str, max_tokens: int) -> int:
        seen = outs.get(model, [])
        return int(sum(seen) / len(seen)) if len(seen) >= 5 else int(0.6 * max_tokens)

    dry = DryRunProvider(settings.backend, store, out_est)
    engine = make_engine(settings, provider=dry)
    prices = PriceBook(settings.prices_path, base.billing_for(settings.backend))
    per_arm, answers = {}, {}
    for arm in arms:
        before_new, before_hits = len(dry.new), dry.hits
        mode, pol = arm_policy(base, arm)
        res = run_rows(engine, pol, mode, arm, rows)
        answers[arm] = {r["pos"]: r.get("answer") for r in res}
        per_arm[arm] = {"upstream_calls": sum(1 for r in res if r.get("cache_status") not in HIT and not r.get("error")),
                        "new_generations": len(dry.new) - before_new, "replayed": dry.hits - before_hits}
    gen_cost = defaultdict(float)
    for _, model, _, _, n_in, n_out in dry.new.values():
        gen_cost[model] += prices.cost(_alias_for(base, settings.backend, model), n_in, n_out)
    # judge upper bound: distinct (item, answer) pairs to grade + pairwise on differing answers (capped per arm)
    ref = {r["pos"]: r for r in rows}
    grade_keys, n_pair, j_in = set(), 0, 0
    a0 = answers.get("A0", {})
    for arm in arms:
        for pos, ans in answers[arm].items():
            if ans is None:
                continue
            if judge_mode in ("grade", "both") and (pos, ans) not in grade_keys:
                grade_keys.add((pos, ans))
                j_in += 200 + count_text(ref[pos]["query"]) + count_text(ref[pos].get("reference") or "") + out_est("", 256)
        if arm != "A0" and judge_mode in ("pairwise", "both") and a0:
            k = min(pairwise_max, sum(1 for p, a in answers[arm].items() if a is not None and a != a0.get(p)))
            n_pair += k
            j_in += 2 * k * (160 + 2 * out_est("", 256) + 60)
    judge_cost = 0.0
    n_judge = len(grade_keys) + 2 * n_pair
    if settings.backend not in ("mock",) and judge_mode != "none":
        jb = os.environ.get("COSTGUARD_JUDGE_BACKEND") or settings.backend
        if jb != "mock":
            jm = judge_model or base.model_id(jb, "strong")
            try:
                judge_cost = PriceBook(settings.prices_path, base.billing_for(jb)).cost(
                    _alias_for(base, jb, jm), int(j_in * (1.15 if jb == "anthropic" else 1.0)), 2 * n_judge)
            except KeyError:
                judge_cost = float("nan")
    return {"backend": settings.backend, "rows": len(rows), "arms": per_arm, "cassette_entries": len(store),
            "new_generations": len(dry.new), "new_calls": list(dry.new.values()),
            "est_generation_usd": round(sum(gen_cost.values()), 4),
            "est_generation_usd_by_model": {m: round(v, 4) for m, v in gen_cost.items()},
            "est_judge_calls_max": n_judge, "est_judge_usd_max": round(judge_cost, 4)}


def print_preflight(p: dict) -> None:
    print(f"\nPre-flight ({p['backend']}, {p['rows']} trace rows, cassette has {p['cassette_entries']} entries)")
    for arm, a in p["arms"].items():
        print(f"  {arm:3s} upstream calls {a['upstream_calls']:5d}   new generations {a['new_generations']:5d}   "
              f"reused (cassette or an earlier arm) {a['upstream_calls'] - a['new_generations']:5d}")
    print(f"  NEW generations: {p['new_generations']}  est. ${p['est_generation_usd']:.2f} "
          f"{p['est_generation_usd_by_model']}")
    print(f"  judge calls <= {p['est_judge_calls_max']}  est. <= ${p['est_judge_usd_max']:.2f} "
          f"(judge cassette hits make this cheaper)")
    tot = p["est_generation_usd"] + (p["est_judge_usd_max"] if not math.isnan(p["est_judge_usd_max"]) else 0)
    print(f"  TOTAL est. <= ${tot:.2f}\n", flush=True)


# ------------------------------------------------------------------------------------------- judging
def judge_results(judge: Judge, results: dict[str, list[dict]], mode: str, pairwise_max: int, workers: int,
                  seed: int = 0) -> dict:
    """Grades (pos -> score) per arm and pairwise verdicts vs A0 (pos -> 'win'|'tie'|'loss'|'error'), judged in
    parallel. 'error' = the judgment failed (either order unparseable or the call failed): missing, never a tie."""
    grades: dict[str, dict[int, Optional[float]]] = {a: {} for a in results}
    pair: dict[str, dict[int, str]] = {a: {} for a in results if a != "A0"}
    identical: dict[str, int] = {a: 0 for a in results if a != "A0"}
    tasks: list[tuple] = []
    if mode in ("grade", "both"):
        for arm, res in results.items():
            for r in res:
                if r.get("answer") is not None and not r.get("error"):
                    tasks.append(("g", arm, r["pos"], r["query"], r["answer"], r.get("reference")))
    a0 = {r["pos"]: r for r in results.get("A0", []) if r.get("answer") is not None and not r.get("error")}
    if mode in ("pairwise", "both") and a0:
        for arm, res in results.items():
            if arm == "A0":
                continue
            diff = []
            for r in res:
                b = a0.get(r["pos"])
                if b is None or r.get("answer") is None or r.get("error"):
                    continue
                if r["answer"].strip() == b["answer"].strip():
                    identical[arm] += 1
                    pair[arm][r["pos"]] = "tie"
                else:
                    diff.append((r, b))
            random.Random(f"{seed}:{arm}").shuffle(diff)
            for r, b in diff[:pairwise_max]:
                tasks.append(("p", arm, r["pos"], r["query"], b["answer"], r["answer"], r.get("reference")))

    def do(t):
        if t[0] == "g":
            return t, judge.grade(t[3], t[4], t[5])
        v = judge.pairwise(t[3], t[4], t[5], t[6])            # A = A0 answer, B = arm answer
        return t, {"A": "loss", "B": "win", "tie": "tie"}.get(v, "error")

    if workers > 1 and not isinstance(judge, HeuristicJudge):
        with ThreadPoolExecutor(workers) as ex:
            done = list(ex.map(do, tasks))
    else:
        done = [do(t) for t in tasks]
    for t, v in done:
        if t[0] == "g":
            grades[t[1]][t[2]] = v
        else:
            pair[t[1]][t[2]] = v
    return {"grades": grades, "pairwise": pair, "identical": identical, "mode": mode}


# ------------------------------------------------------------------------------------------- summary
def _pct(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(100 * x, nd)


def _lat(xs: list[float]) -> dict:
    return {"p50": round(percentile(xs, 50), 2) if xs else None, "p99": round(percentile(xs, 99), 2) if xs else None,
            "n": len(xs)}


def summarize_arm(arm: str, res: list[dict], a0: Optional[list[dict]], q: Optional[dict], n_boot: int = 2000) -> dict:
    ok = [r for r in res if not r.get("error")]
    n = len(ok)
    n_all = len(res)       # attempted requests, failed ones included: the denominator of every per-request rate
    hits = Counter(r["cache_status"] for r in ok if r["cache_status"] in HIT)
    fh = sum(1 for r in ok if r.get("false_hit"))
    out: dict = {"arm": arm, "description": ARMS.get(arm, ("",))[0], "n": n, "n_attempted": n_all,
                 "n_errors": n_all - n,
                 "cost_usd": round(sum(r["cost"] for r in ok), 6),
                 "est_baseline_usd": round(sum(r["est_baseline"] for r in ok), 6)}
    eb = out["est_baseline_usd"]
    out["est_savings_pct"] = _pct(1 - out["cost_usd"] / eb) if eb else None
    # ---- paired savings vs A0's actual per-item cost
    if a0 is not None:
        base = {r["pos"]: r for r in a0 if not r.get("error")}
        paired = [(r, base[r["pos"]]) for r in ok if r["pos"] in base]
        cost = [r["cost"] for r, _ in paired]
        c0 = [b["cost"] for _, b in paired]
        cl = [r["cluster_id"] for r, _ in paired]
        ratio, lo, hi = ratio_bootstrap(cost, c0, n=n_boot, clusters=cl)
        m, mlo, mhi = paired_bootstrap([b - a for a, b in zip(cost, c0)], n=n_boot, clusters=cl)
        miss = [(r, b) for r, b in paired if r["cache_status"] not in HIT]
        s0m = sum(b["cost"] for _, b in miss)
        out.update(a0_cost_usd=round(sum(c0), 6), n_paired=len(paired),
                   savings_pct=_pct(1 - ratio), savings_ci=[_pct(1 - hi), _pct(1 - lo)],
                   saved_per_request_usd=[round(m, 8), round(mlo, 8), round(mhi, 8)],
                   savings_on_misses_pct=_pct(1 - sum(r["cost"] for r, _ in miss) / s0m) if s0m else None)
    out["tokens"] = {"input_original": sum(r["in_orig"] for r in ok), "input_sent": sum(r["in_sent"] for r in ok),
                     "output": sum(r["out_tokens"] for r in ok), "cached_input": sum(r["cached_in"] for r in ok),
                     "cache_write": sum(r.get("cache_write", 0) for r in ok)}
    p, lo, hi = proportion_ci(fh, n_all) if n_all else (float("nan"), 0, 1)
    out.update(
        hit_rate={"exact": _pct(hits["exact"] / n_all) if n_all else None,
                  "semantic": _pct(hits["semantic"] / n_all) if n_all else None,
                  "total": _pct(sum(hits.values()) / n_all) if n_all else None},
        hits=dict(hits), false_hits=fh, false_hit_rate=_pct(p), false_hit_ci=[_pct(lo), _pct(hi)],
        false_hit_rate_intent=_pct(sum(1 for r in ok if r.get("false_hit_intent")) / n_all) if n_all else None,
        trap_false_hits=sum(1 for r in ok if r.get("trap_false_hit")),
        unattributed_hits=sum(1 for r in ok if r.get("hit_unattributed")),
        guard_rejections=sum(1 for r in ok if r.get("cache_guard") and r["cache_status"] not in HIT),
        route_mix=dict(Counter(r["model_used"] for r in ok)),
        route_reasons=dict(Counter(r["route_reason"] for r in ok).most_common(8)),
    )
    comp = [r["compression_ratio"] for r in ok if r.get("compression_ratio")]
    ctx = [r for r in ok if r.get("docs_in")]
    ctx_sent = [r for r in ctx if r["cache_status"] not in HIT]
    out["compression"] = {"n": len(comp), "mean_ratio": round(sum(comp) / len(comp), 3) if comp else None}
    out["context"] = {"n_with_docs": len(ctx), "mean_docs_in": round(sum(r["docs_in"] for r in ctx) / len(ctx), 2) if ctx else None,
                      "mean_docs_kept": (round(sum(r["docs_kept"] for r in ctx_sent) / len(ctx_sent), 2)
                                         if ctx_sent else None)}
    hit_r = [r for r in ok if r["cache_status"] in HIT]
    miss_r = [r for r in ok if r["cache_status"] not in HIT]
    modelled = [r["latency_ms"] if r["cache_status"] in HIT else r["overhead_ms"] + r["upstream_ms"] for r in ok]
    out["latency_ms"] = {
        "e2e": _lat(modelled), "e2e_hit": _lat([r["latency_ms"] for r in hit_r]),
        "e2e_miss": _lat([r["overhead_ms"] + r["upstream_ms"] for r in miss_r]),
        "overhead": _lat([r["overhead_ms"] for r in ok]), "overhead_miss": _lat([r["overhead_ms"] for r in miss_r]),
        "wall": _lat([r["latency_ms"] for r in ok]),
        "note": "e2e = CostGuard overhead + recorded upstream generation time (cassette replays are instant)"}
    errs = Counter(k for r in ok for k in r.get("stage_errors", {}))
    if errs:
        out["stage_errors"] = dict(errs)
    if q is not None:
        out["quality"] = quality_block(arm, ok, a0, q, n_boot)
    # complete = every request succeeded and, when grading was requested, every answer has a grade and every
    # pairwise judgment is valid. Incomplete arms are flagged by eval.report and never used as the headline.
    qb = out.get("quality") or {}
    graded = q is not None and q.get("mode", "both") in ("grade", "both")
    out["coverage"] = {"attempted": n_all, "failed": n_all - n, "ungraded": qb.get("n_ungraded", 0) if graded else None,
                       "pairwise_errors": (qb.get("pairwise") or {}).get("errors", 0)}
    out["complete"] = not (out["coverage"]["failed"] or out["coverage"]["ungraded"]
                           or out["coverage"]["pairwise_errors"])
    return out


def quality_block(arm: str, ok: list[dict], a0: Optional[list[dict]], q: dict, n_boot: int) -> dict:
    g = q["grades"].get(arm, {})
    scored = [(r, g[r["pos"]]) for r in ok if g.get(r["pos"]) is not None]
    blk: dict = {"n_graded": len(scored), "n_ungraded": sum(1 for r in ok if g.get(r["pos"]) is None)}
    if scored:
        n_ok = sum(1 for _, s in scored if s >= CORRECT_AT)
        cost_scored = sum(r["cost"] for r, _ in scored)
        blk.update(mean=round(sum(s for _, s in scored) / len(scored), 4), n_correct=n_ok,
                   correct_rate=_pct(n_ok / len(scored)),
                   cost_per_correct_usd=round(cost_scored / n_ok, 8) if n_ok else None)
    if a0 is not None and arm != "A0":
        g0 = q["grades"].get("A0", {})
        both = [(g[r["pos"]], g0[r["pos"]], r["cluster_id"]) for r in ok
                if g.get(r["pos"]) is not None and g0.get(r["pos"]) is not None]
        if both:
            ratio, lo, hi = ratio_bootstrap([b[0] for b in both], [b[1] for b in both], n=n_boot,
                                            clusters=[b[2] for b in both])
            d, dlo, dhi = paired_bootstrap([b[0] - b[1] for b in both], n=n_boot, clusters=[b[2] for b in both])
            blk.update(a0_mean_paired=round(sum(b[1] for b in both) / len(both), 4), retained=_pct(ratio),
                       retained_ci=[_pct(lo), _pct(hi)], score_diff=[round(d, 4), round(dlo, 4), round(dhi, 4)])
        pw = q["pairwise"].get(arm, {})
        if pw:
            ident = q["identical"].get(arm, 0)
            c = Counter(pw.values())
            w, t, l = c["win"], c["tie"] - ident, c["loss"]
            n_j = w + t + l                    # valid judgments only: a failed judgment is missing, not a tie
            ni, nlo, nhi = proportion_ci(w + t, n_j) if n_j else (float("nan"), 0.0, 1.0)
            blk["pairwise"] = {"identical": ident, "judged": n_j, "win": w, "tie": t, "loss": l, "errors": c["error"],
                               "non_inferior_rate_judged": _pct(ni), "non_inferior_ci": [_pct(nlo), _pct(nhi)]}
    return blk


def waterfall(summary: dict[str, dict], results: Optional[dict[str, list[dict]]] = None) -> list[dict]:
    """Each lever's increment over the previous arm. With `results`, every step is computed on the SAME item set:
    the items that succeeded in every cumulative arm (n_items / n_excluded on each step say so)."""
    arms = [a for a in CUMULATIVE if a in summary]
    common: Optional[set] = None
    if results is not None and arms:
        common = set.intersection(*({r["pos"] for r in results[a] if not r.get("error")} for a in arms))

    def tot(arm: str) -> tuple[float, int, int]:
        if common is None:
            s = summary[arm]
            return s["cost_usd"], s["tokens"]["input_sent"], s["tokens"]["output"]
        rs = [r for r in results[arm] if r["pos"] in common]
        return round(sum(r["cost"] for r in rs), 6), sum(r["in_sent"] for r in rs), sum(r["out_tokens"] for r in rs)

    steps, prev = [], None
    for arm in arms:
        if prev is not None:
            (pc, pin, pout), (sc, sin, sout) = tot(prev), tot(arm)
            if common is not None and "A0" in arms:
                a0c = tot("A0")[0]
            else:
                a0c = summary[arm].get("a0_cost_usd") or summary.get("A0", {}).get("cost_usd") or 0
            dc = pc - sc
            step = {"from": prev, "to": arm, "lever": LEVER.get(arm, arm), "saved_usd": round(dc, 6),
                    "saved_pct_of_a0": _pct(dc / a0c) if a0c else None,
                    "saved_input_tokens": pin - sin, "saved_output_tokens": pout - sout}
            if common is not None:
                step.update(n_items=len(common), n_excluded=max(len(results[a]) for a in arms) - len(common))
            steps.append(step)
        prev = arm
    return steps


# ------------------------------------------------------------------------------------------- CLI
def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _tag_for(trace: Path) -> str:
    stem = Path(trace).stem
    return "" if stem == "trace_v1" else stem.replace("trace_v1_", "")


def print_table(summary: dict, file=sys.stdout) -> None:
    hdr = (f"{'arm':4s} {'n':>4s} {'cost $':>10s} {'savings % [95% CI]':>24s} {'hit% ex/sem':>12s} "
           f"{'false-hit%':>10s} {'quality ret % [CI]':>22s} {'W/T/L':>11s} {'p50/p99 ms':>14s}")
    print(hdr, file=file)
    print("-" * len(hdr), file=file)
    for arm, s in summary["arms"].items():
        sv = (f"{s.get('savings_pct', 0):6.1f} [{s['savings_ci'][0]:.1f}, {s['savings_ci'][1]:.1f}]"
              if s.get("savings_ci") and s["savings_ci"][0] is not None else "n/a")
        q = s.get("quality", {})
        qr = (f"{q['retained']:6.1f} [{q['retained_ci'][0]:.1f}, {q['retained_ci'][1]:.1f}]"
              if q.get("retained") is not None else ("100 (ref)" if arm == "A0" and q else "n/a"))
        pw = q.get("pairwise")
        wtl = f"{pw['win']}/{pw['tie'] + pw['identical']}/{pw['loss']}" if pw else "-"
        hr = s["hit_rate"]
        lat = s["latency_ms"]["e2e"]
        print(f"{arm:4s} {s['n']:4d} {s['cost_usd']:10.4f} {sv:>24s} {hr['exact'] or 0:5.1f}/{hr['semantic'] or 0:<5.1f} "
              f"{s['false_hit_rate'] or 0:10.2f} {qr:>22s} {wtl:>11s} "
              f"{(lat['p50'] or 0):6.0f}/{(lat['p99'] or 0):<7.0f}", file=file)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cumulative-ablation A/B on the frozen trace (see module docstring).")
    ap.add_argument("--trace", default=str(DEFAULT_TRACE))
    ap.add_argument("--backend", default=os.environ.get("COSTGUARD_BACKEND", "mock"))
    ap.add_argument("--arms", default=",".join(CUMULATIVE))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N trace rows")
    ap.add_argument("--judge", choices=["auto", "heuristic", "model", "none"], default="auto",
                    help="auto: heuristic on mock, the backend's strong tier otherwise")
    ap.add_argument("--judge-mode", choices=["grade", "pairwise", "both"], default="both")
    ap.add_argument("--pairwise-max", type=int, default=150, help="max differing items judged pairwise per arm")
    ap.add_argument("--workers", type=int, default=None, help="parallel upstream prefetch + judging (default 4 paid, 1 local)")
    ap.add_argument("--cassette", default=None, help="default eval/cassettes/<backend>_ab.jsonl (none for mock)")
    ap.add_argument("--out", default=None, help="summary path (default eval/results/ab_summary[...].json)")
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--estimate-only", action="store_true", help="print the pre-flight estimate and exit")
    ap.add_argument("--yes", action="store_true", help="allow spending on a paid backend")
    ap.add_argument("--replay", action="store_true",
                    help="cassette replay only ($0): never generates and never builds the real upstream (no key, no "
                         "MLX); a request or judgment missing from the cassettes fails and marks its arm incomplete")
    ap.add_argument("--no-prefetch", action="store_true", help="don't pre-generate known upstream calls in parallel")
    ap.add_argument("--agreement", action="store_true", help="also report judge-vs-human agreement (human_labels.jsonl)")
    ap.add_argument("--provider-prompt-cache", action="store_true",
                    help="keep Anthropic prompt caching on (default off for headline numbers: replay would freeze "
                         "the first call's cache-write usage; see docs/EVALUATION.md section 2)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args(argv)

    t_start = time.time()
    trace = Path(args.trace)
    all_rows = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
    rows = all_rows[: args.limit] if args.limit else all_rows
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            ap.error(f"unknown arm {a}")
    backend = args.backend
    paid = backend not in LOCAL_BACKENDS
    workers = args.workers or (4 if paid else 1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _tag_for(trace)
    suffix = (f"_{tag}" if tag else "") + (f"_{backend}" if not paid else "") + (f"_limit{args.limit}" if args.limit else "")
    summary_path = Path(args.out) if args.out else out_dir / f"ab_summary{suffix}.json"
    db_path = out_dir / f"ab_{backend}{('_' + tag) if tag else ''}{('_limit' + str(args.limit)) if args.limit else ''}.sqlite"
    cassette = Path(args.cassette) if args.cassette else (None if backend == "mock" else CASSETTES / f"{backend}_ab.jsonl")
    if args.replay and cassette is None:
        ap.error("--replay needs a cassette backend (or --cassette)")

    settings = Settings.from_env()
    settings.backend = backend
    settings.cassette = cassette
    settings.cassette_mode = "replay" if args.replay else os.environ.get("COSTGUARD_CASSETTE_MODE", "auto")
    if args.replay:
        os.environ["COSTGUARD_JUDGE_CASSETTE_MODE"] = "replay"
    settings.db_path = db_path
    base, overrides = apply_overrides(load_policy(settings.policy_path))
    print(f"run_ab: backend={backend} trace={trace.name} rows={len(rows)} arms={','.join(arms)} "
          f"config={base.config_hash}{' overrides=' + json.dumps(overrides) if overrides else ''}", flush=True)

    # ---- pre-flight (any backend that is not free mock)
    pre = None
    if backend != "mock":
        pre = preflight(settings, base, rows, arms, cassette, "none" if args.judge == "none" else args.judge_mode,
                        args.pairwise_max, os.environ.get("COSTGUARD_JUDGE_MODEL"))
        print_preflight(pre)
        if args.estimate_only:
            return 0
        if args.replay and pre["new_generations"]:
            print(f"WARNING: --replay: {pre['new_generations']} upstream calls are not in {cassette.name}; those "
                  "requests will fail and their arms will be marked incomplete", file=sys.stderr)
        spend = pre["est_generation_usd"] + (pre["est_judge_usd_max"] if not math.isnan(pre["est_judge_usd_max"]) else 0)
        if paid and not args.replay and (pre["new_generations"] > 0 or pre["est_judge_calls_max"] > 0) and not args.yes:
            print(f"Paid backend '{backend}': re-run with --yes to spend up to ~${spend:.2f} "
                  "(already-recorded calls and judgements are free).", file=sys.stderr)
            return 2
    elif args.estimate_only:
        print("mock backend: $0, nothing to estimate")
        return 0

    # ---- engine (real upstream, cassette-wrapped) + logger
    if db_path.exists():
        db_path.unlink()
    prompt_cache = None                 # provider prompt caching: only the anthropic adapter has it
    if "anthropic" in (backend, os.environ.get("COSTGUARD_JUDGE_BACKEND") or backend):
        prompt_cache = set_provider_prompt_cache(args.provider_prompt_cache)
    replay_provider = keyless_provider(backend, cassette, "replay", base.default_max_tokens) if args.replay else None
    engine = make_engine(settings, provider=replay_provider, with_logger=True)
    if pre and pre["new_calls"] and not args.no_prefetch and not args.replay and workers > 1:
        calls = pre["new_calls"]
        print(f"prefetching {len(calls)} upstream generations with {workers} workers ...", flush=True)
        errors, lock = Counter(), threading.Lock()

        def gen(c):
            try:
                engine.provider.complete(c[0], c[1], c[2], c[3])
            except Exception as e:
                with lock:              # updated from worker threads
                    errors[type(e).__name__] += 1
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(gen, calls))
        if errors:
            print(f"  prefetch errors (will retry inline): {dict(errors)}", flush=True)
    if args.judge == "none":
        judge = None
    elif args.judge == "heuristic" or (args.judge == "auto" and backend == "mock"):
        judge = HeuristicJudge()
    else:
        judge = get_judge(settings, provider=engine.provider)

    results: dict[str, list[dict]] = {}
    up = engine.provider if isinstance(engine.provider, CassetteProvider) else None
    upstream: dict[str, dict] = {}
    for arm in arms:
        mode, pol = arm_policy(base, arm)
        h0, m0, t0 = (up.hits, up.misses, time.time()) if up else (0, 0, time.time())
        step = max(50, len(rows) // 6)

        def progress(i, out, arm=arm):
            if i % step == 0 or i == len(rows):
                c = sum(r.get("cost", 0) for r in out)
                h = sum(1 for r in out if r.get("cache_status") in HIT)
                print(f"  {arm} {i}/{len(rows)}  cost ${c:.4f}  hits {h}", flush=True)
        results[arm] = run_rows(engine, pol, mode, arm, rows, progress=progress if len(rows) >= 100 else None)
        upstream[arm] = {"replayed": (up.hits - h0) if up else None, "new": (up.misses - m0) if up else None,
                         "seconds": round(time.time() - t0, 2)}
        print(f"{arm}: done in {upstream[arm]['seconds']}s (cassette replayed {upstream[arm]['replayed']}, "
              f"new {upstream[arm]['new']})", flush=True)

    for h in getattr(engine, "hooks", []):          # RequestLogger writes asynchronously: make the sqlite complete
        if callable(getattr(h, "flush", None)):
            try:
                h.flush()
            except Exception as e:  # never lose the summary over a logging hiccup
                print(f"warning: hook flush failed: {e}", file=sys.stderr)

    q = None
    if judge is not None:
        print(f"judging with {judge.label} ({args.judge_mode}) ...", flush=True)
        q = judge_results(judge, results, args.judge_mode, args.pairwise_max, workers)

    a0 = results.get("A0")
    per_arm = {arm: summarize_arm(arm, res, a0, q, args.bootstrap) for arm, res in results.items()}
    for arm in per_arm:
        per_arm[arm]["upstream"] = upstream[arm]
    from .build_trace import file_sha256, trace_stats
    prices = engine.prices
    summary = {
        "meta": {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "git_commit": _git_commit(), "backend": backend, "models": base.backends.get(backend),
                 "billing": base.billing_for(backend), "prices_checked_on": prices.checked_on,
                 "config_hash": base.config_hash, "overrides": overrides, "trace": str(trace.relative_to(ROOT))
                 if trace.is_relative_to(ROOT) else str(trace), "trace_sha256": file_sha256(trace),
                 "trace_rows_total": len(all_rows), "rows_used": len(rows), "limit": args.limit,
                 "trace_stats": trace_stats(rows), "cassette": str(cassette.relative_to(ROOT))
                 if cassette and cassette.is_relative_to(ROOT) else (str(cassette) if cassette else None),
                 "components": getattr(engine, "component_status", {}),
                 "judge": judge.describe() if judge else None, "judge_mode": args.judge_mode,
                 "pairwise_max": args.pairwise_max, "correct_threshold": CORRECT_AT,
                 "provider_prompt_cache": prompt_cache, "replay_only": bool(args.replay),
                 "savings_definition": "1 - sum(arm cost) / sum(A0 actual cost) on the same items (paired); "
                                       "est_savings_pct uses the record-level estimated baseline",
                 "preflight": {k: v for k, v in pre.items() if k != "new_calls"} if pre else None,
                 "runtime_s": round(time.time() - t_start, 1)},
        "arms": per_arm,
        "waterfall": waterfall(per_arm, results),
    }
    cache_tok = sum(a["tokens"]["cached_input"] + a["tokens"]["cache_write"] for a in per_arm.values())
    if prompt_cache is False and cache_tok:
        summary["meta"]["prompt_cache_warning"] = (
            f"{cache_tok} prompt-cache read/write tokens replayed although provider prompt caching is off: the "
            "cassette was recorded with caching on; re-record it for headline numbers")
        print("WARNING: " + summary["meta"]["prompt_cache_warning"], file=sys.stderr)
    if args.agreement and judge is not None:
        summary["judge_human_agreement"] = human_agreement(judge=judge)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print()
    print_table(summary)
    for arm, a in per_arm.items():
        if not a["complete"]:
            c = a["coverage"]
            print(f"WARNING: {arm} INCOMPLETE: {c['failed']} failed / {c['ungraded'] or 0} ungraded / "
                  f"{c['pairwise_errors']} pairwise judge errors of {c['attempted']} requests "
                  "(eval.report will not use it as the headline)", file=sys.stderr)
    print(f"\nwaterfall: " + "; ".join(f"{w['lever']} -${w['saved_usd']:.4f} ({w['saved_pct_of_a0']}% of A0)"
                                      for w in summary["waterfall"]))
    print(f"wrote {summary_path.relative_to(ROOT) if summary_path.is_relative_to(ROOT) else summary_path} "
          f"and {db_path.name} in {summary['meta']['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
