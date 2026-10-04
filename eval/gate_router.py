"""Offline eval gate for model downshift: which categories may the `gated` router send to the cheap tier?

    python -m eval.gate_router --dry-run                      # mock backend: exercises everything, allows nothing
    COSTGUARD_BACKEND=anthropic python -m eval.gate_router    # prints the plan + estimated $, then stops
    COSTGUARD_BACKEND=anthropic python -m eval.gate_router --yes            # spend it
    COSTGUARD_BACKEND=anthropic python -m eval.gate_router --replay         # re-score from cassettes only, $0

Method (per category, decided in advance, see docs/components/router_and_gate.md):
  1. For every eval item, generate the strong-tier and the cheap-tier answer with the serving prompt (engine in
     mode "off", so no cache / trimming / routing interferes). Calls go through a cassette, so re-runs are free.
  2. Keep the items the router would actually downshift: those with no hardness signal (hard ones stay strong).
  3. Score both answers with the shared judge (eval.judge): absolute grade 0..1 against the reference, and a
     position-swapped pairwise verdict. Per-item quality difference d = 100 * (grade_cheap - grade_strong) points.
  4. Paired (cluster) bootstrap 95% CI of mean(d). ALLOW the category only if
        judge coverage >= 95% of its routable items  and  n >= min_n (30)  and
        CI lower bound >= -margin (5 points)  and  measured saving > min_saving.
     A failed or unparseable judgment is missing evidence (pairwise "error"), never a tie or a zero difference.
     Everything else, including categories with no or too little data, stays on strong.
  5. Write configs/router_gate.json (read live by the router) and eval/results/router_gate*.json.

Paid backends (anything except mock/mlx) print the number of new generations/judge calls and an estimated $
first, and only run with --yes. --limit N caps the item count (round-robin across categories) for smoke runs.
Dry and partial runs never overwrite a full gate file unless --force.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from costguard.config import ROOT, Settings, load_policy
from costguard.pipeline import build_messages, format_docs
from costguard.pricing import PriceBook
from costguard.router import features
from costguard.router.classifier import BITEXT, CATEGORIES, load_bitext_rows
from costguard.schemas import ChatMessage, ChatRequest, CostGuardOptions, RouteInput
from costguard.tokens import count_messages, count_text

GATE_PATH = ROOT / "configs" / "router_gate.json"
RESULTS_DIR = ROOT / "eval" / "results"
EVALSET_DIR = ROOT / "eval" / "data" / "evalset"
TRACE_PATH = ROOT / "eval" / "data" / "trace_v1.jsonl"
GATE_CASSETTE = ROOT / "eval" / "cassettes" / "gate.jsonl"
LOCAL_BACKENDS = ("mock", "mlx")
SKIP = ("exact_cache", "semantic_cache", "context_optimizer", "compressor", "router")
# Minimum cacheable prefix (tokens) per billed model, from the research notes (Anthropic prompt-caching docs,
# fetched 2026-10-03). OpenAI GPT-5.x caches from 1,024 tokens; Gemini 2.5 from 2,048.
CACHE_MIN_PREFIX = {"claude-sonnet-5.5": 512, "claude-haiku-4.5": 4096, "gpt-5.4-mini": 1024, "gpt-5-nano": 1024,
                    "gemini-2.5-flash": 2048, "gemini-2.5-flash-lite": 2048}
TOKEN_RATIO = {"anthropic": 1.15}       # backend tokens per o200k token, for pre-run estimates only
MIN_JUDGE_COVERAGE = 0.95               # share of a category's routable items that must carry a valid judgment


def _rel(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


# ============================================================================================ data
def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def _norm(r: dict, source: str, i: int) -> Optional[dict]:
    cat = (r.get("category") or "").strip().lower()
    if not r.get("query") or cat not in CATEGORIES:
        return None
    return {"id": r.get("id") or r.get("item_id") or f"{source}-{i:04d}", "category": cat, "query": r["query"],
            "reference": r.get("reference"), "needs_context": bool(r.get("needs_context")),
            "context": list(r.get("context") or []), "type": r.get("type") or "answerable",
            "pair_id": r.get("pair_id") or r.get("cluster_id"), "source": r.get("source") or source,
            "author": r.get("author")}


def bitext_sample(per_category: int, seed: int) -> list[dict]:
    rows = load_bitext_rows()
    by_cat: dict[str, list] = defaultdict(list)
    for q, cat, intent, resp in rows:
        by_cat[cat].append((q, intent, resp))
    rng = random.Random(seed)
    out = []
    for cat in CATEGORIES:
        xs = sorted(by_cat.get(cat, []))
        rng.shuffle(xs)
        for j, (q, intent, resp) in enumerate(xs[:per_category]):
            out.append({"id": f"bitext-{cat}-{j:03d}", "category": cat, "query": q, "reference": resp,
                        "needs_context": False, "context": [], "type": "answerable", "pair_id": None,
                        "source": f"bitext:{intent}", "author": "bitext"})
    return out


def kb_seed_items() -> list[dict]:
    try:
        from eval.kb import kb_questions
    except Exception:
        return []
    return [x for i, r in enumerate(kb_questions()) if (x := _norm({**r, "source": "kb-seed"}, "kb", i))]


def load_items(args) -> tuple[list[dict], str, list[str]]:
    notes: list[str] = []
    if args.data:
        items = [x for i, r in enumerate(_read_jsonl(Path(args.data))) if (x := _norm(r, "data", i))]
        return items, f"file:{args.data}", notes
    files = sorted(EVALSET_DIR.glob("*.jsonl")) if EVALSET_DIR.exists() else []
    if args.source in ("auto", "evalset") and files:
        items = [x for p in files for i, r in enumerate(_read_jsonl(p)) if (x := _norm(r, p.stem, i))]
        if items:
            return items, "evalset:" + ",".join(p.name for p in files), notes
    if args.source == "evalset":
        raise SystemExit(f"no eval set rows found in {EVALSET_DIR}")
    notes.append(f"No hand-written eval set in {_rel(EVALSET_DIR)}: falling back to a fixed-seed sample "
                 f"of public Bitext rows ({args.per_category}/category, seed {args.seed}). Bitext references are "
                 f"generic chatbot replies, not ShopNest policy, so this gate is weaker evidence than the eval set.")
    items = bitext_sample(args.per_category, args.seed)
    src = f"bitext-sample:{BITEXT}"
    if args.source in ("auto", "bitext+kb"):
        kb = kb_seed_items()
        if kb:
            items += kb
            src += "+kb-seed"
            notes.append(f"Added {len(kb)} KB seed questions (eval.kb.kb_questions, with retrieved context): "
                         f"Bitext has no returns or product rows.")
    return items, src, notes


def round_robin_limit(items: list[dict], limit: Optional[int]) -> list[dict]:
    if not limit or limit >= len(items):
        return items
    by_cat: dict[str, list] = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    out, k = [], 0
    while len(out) < limit:
        for cat in CATEGORIES:
            if k < len(by_cat[cat]) and len(out) < limit:
                out.append(by_cat[cat][k])
        k += 1
    return out


def attach_context(items: list[dict], notes: list[str], quiet: bool) -> None:
    need = [it for it in items if it["needs_context"] and not it["context"]]
    if not need:
        return
    try:
        from eval.kb import retrieve
    except Exception as e:
        notes.append(f"{len(need)} items need context but eval.kb is unavailable ({type(e).__name__}); sent without it.")
        return
    log(f"retrieving KB context for {len(need)} items ...", quiet)
    for it in need:
        it["context"] = retrieve(it["query"])


# ============================================================================================ plan / estimate
def serving_messages(policy, it: dict) -> list[ChatMessage]:
    return build_messages(policy.system_prompt, [], it["query"], format_docs(it["context"]))


def cassette_keys(path: Optional[Path]) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    keys = set()
    for line in Path(path).read_text().splitlines():
        if line.strip():
            try:
                keys.add(json.loads(line)["key"])
            except Exception:
                pass
    return keys


JUDGE_OVERHEAD_TOKENS = {"grade": 230, "pairwise": 200}   # system + rubric text of eval.judge prompts (o200k)


def estimate(items: list[dict], settings: Settings, policy, prices: PriceBook, scorer: str, judge_backend: str) -> dict:
    """New upstream calls and an estimated $ (upper bound: every output at max_tokens; judge calls all new)."""
    from costguard.providers.cassette import call_key
    backend, max_tok = settings.backend, policy.default_max_tokens
    keys = cassette_keys(settings.cassette)
    ratio = TOKEN_RATIO.get(backend, 1.0)
    new = Counter()
    usd_max = usd_typ = 0.0
    for it in items:
        msgs = serving_messages(policy, it)
        tin = int(count_messages(msgs) * ratio)
        for alias in ("strong", "cheap"):
            if call_key(msgs, policy.model_id(backend, alias), max_tok, 0.0) in keys:
                continue
            new[alias] += 1
            p = prices.price_for(alias)
            usd_max += (p.input * tin + p.output * max_tok) / 1e6
            usd_typ += (p.input * tin + p.output * max_tok * 0.5) / 1e6
    judge_calls, judge_usd = 0, 0.0
    if judge_backend not in ("mock", "proxy"):
        jprices = PriceBook(settings.prices_path, policy.billing_for(judge_backend))
        pj = jprices.price_for("strong")
        jr = TOKEN_RATIO.get(judge_backend, 1.0)
        for it in items:
            base = count_text(it["query"]) + count_text(it.get("reference") or "")
            ans = max_tok * 0.6
            calls = []
            if scorer in ("grade", "both"):
                calls += [JUDGE_OVERHEAD_TOKENS["grade"] + base + ans] * 2
            if scorer in ("pairwise", "both"):
                calls += [JUDGE_OVERHEAD_TOKENS["pairwise"] + base + 2 * ans] * 2
            judge_calls += len(calls)
            judge_usd += sum((pj.input * t * jr + pj.output * 5) / 1e6 for t in calls)
    return {"items": len(items), "generations_total": 2 * len(items), "generations_new": dict(new),
            "generations_cached": 2 * len(items) - sum(new.values()), "judge_calls_upper_bound": judge_calls,
            "usd_generation_upper_bound": round(usd_max, 4), "usd_generation_typical": round(usd_typ, 4),
            "usd_judge_upper_bound": round(judge_usd, 4),
            "usd_total_upper_bound": round(usd_max + judge_usd, 4)}


# ============================================================================================ engine + scoring
def make_engine(settings: Settings, replay: bool):
    from costguard.factory import build_engine
    try:
        return build_engine(settings, with_logger=False, skip=SKIP)
    except Exception as e:
        if not replay:
            raise
        # replay without a key: serve from the cassette only (a miss raises and the item is skipped)
        from costguard.pipeline import CostGuard
        from costguard.providers.cassette import CassetteProvider
        from eval.judge import ReplayOnlyProvider
        policy = load_policy(settings.policy_path)
        prov = CassetteProvider(ReplayOnlyProvider(settings.backend, str(e)[:100], TOKEN_RATIO.get(settings.backend, 1.0)),
                                settings.cassette, "replay")
        return CostGuard(policy, settings, prov, PriceBook(settings.prices_path, policy.billing_for(settings.backend)))


def generate(engine, it: dict, alias: str) -> dict:
    req = ChatRequest(model=alias, messages=[ChatMessage(role="user", content=it["query"])], temperature=0.0,
                      costguard=CostGuardOptions(mode="off", category=it["category"], context=it["context"],
                                                 arm=f"gate-{alias}", item_id=it["id"], no_cache=True))
    comp, rec = engine.handle(req)
    return {"text": comp.text, "model": rec.model_id, "cost_usd": rec.cost_usd, "input_tokens": rec.input_tokens_sent,
            "input_tokens_original": rec.input_tokens_original, "output_tokens": rec.output_tokens,
            "cached_input_tokens": rec.cached_input_tokens, "latency_ms": round(comp.latency_ms, 1)}


class ProxyJudge:
    """Fallback scorer, used only when eval.judge is unavailable or --judge proxy: similarity of the answer to the
    reference (bge-small cosine; token-overlap F1 if the embedder can't load). A PROXY, not a quality judgement."""

    def __init__(self):
        from costguard.router.classifier import get_embedder
        self.model = get_embedder()
        self.label = "proxy:embedding-similarity" if self.model is not None else "proxy:lexical-f1"
        self.stats = Counter()

    def _sim(self, a: str, b: str) -> float:
        if self.model is not None:
            from costguard.router.classifier import embed
            v = embed(self.model, [a or " ", b or " "])
            return float(max(0.0, min(1.0, v[0] @ v[1])))
        from eval.judge import overlap_f1
        return overlap_f1(a, b)

    def grade(self, question, answer, reference=None):
        self.stats["grades"] += 1
        return round(self._sim(answer, reference or question), 4)

    def pairwise(self, question, answer_a, answer_b, reference=None):
        self.stats["pairwise"] += 1
        a, b = self.grade(question, answer_a, reference), self.grade(question, answer_b, reference)
        return "tie" if abs(a - b) < 0.02 else ("A" if a > b else "B")

    def describe(self):
        return {"judge": self.label, "stats": dict(self.stats), "cost_new_usd": 0.0}


def make_judge(kind: str, settings: Settings, engine):
    if kind == "proxy":
        return ProxyJudge(), "proxy"
    try:
        from eval.judge import get_judge
        j = get_judge(settings, provider=getattr(engine, "provider", None))
        return j, getattr(j, "label", type(j).__name__)
    except Exception as e:
        log(f"WARN: eval.judge unavailable ({type(e).__name__}: {e}); using the labelled proxy scorer")
        return ProxyJudge(), "proxy"


# ============================================================================================ statistics
def bootstrap(diffs: list[float], clusters: Optional[list], n: int, seed: int) -> tuple[float, float, float]:
    try:
        from eval.stats import paired_bootstrap
        return paired_bootstrap(diffs, n=n, seed=seed, clusters=clusters)
    except ImportError:
        d = np.asarray(diffs, float)
        if d.size == 0:
            return (math.nan, math.nan, math.nan)
        rng = np.random.default_rng(seed)
        labels = clusters or list(range(d.size))
        groups: dict[Any, list[float]] = defaultdict(list)
        for x, c in zip(d, labels):
            groups[c].append(x)
        g = list(groups.values())
        boots = []
        for _ in range(n):
            pick = rng.integers(0, len(g), len(g))
            vals = [x for i in pick for x in g[i]]
            boots.append(float(np.mean(vals)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)


def wilson(k: int, n: int) -> tuple[float, float, float]:
    try:
        from eval.stats import proportion_ci
        return proportion_ci(k, n)
    except ImportError:
        if n == 0:
            return (math.nan, 0.0, 1.0)
        z, p = 1.959963984540054, k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return p, max(0.0, c - h), min(1.0, c + h)


def n_needed(sd: float, mean: float, margin: float) -> Optional[int]:
    """Items needed for the 95% CI lower bound to clear -margin at the observed mean and sd (normal approx.)."""
    slack = mean + margin
    if not (sd > 0) or slack <= 0:
        return None
    return int(math.ceil((1.96 * sd / slack) ** 2))


# ============================================================================================ economics
def price_gap(prices: PriceBook, measured: Optional[dict] = None) -> dict:
    s, c = prices.price_for("strong"), prices.price_for("cheap")
    ref_in, ref_out = 1500, 400

    def per_req(p, tin, tout):
        return (p.input * tin + p.output * tout) / 1e6

    out = {"strong": {"billed_as": prices.billing.get("strong"), "input": s.input, "output": s.output,
                      "cached_input": s.cached_input},
           "cheap": {"billed_as": prices.billing.get("cheap"), "input": c.input, "output": c.output,
                     "cached_input": c.cached_input},
           "reference_request": {"input_tokens": ref_in, "output_tokens": ref_out,
                                 "strong_usd": round(per_req(s, ref_in, ref_out), 6),
                                 "cheap_usd": round(per_req(c, ref_in, ref_out), 6)}}
    ratio = per_req(c, ref_in, ref_out) / per_req(s, ref_in, ref_out)
    out["cost_ratio_cheap_over_strong"] = round(ratio, 4)
    out["saving_per_downshifted_request"] = round(1 - ratio, 4)
    out["router_saving_at_50pct_routed"] = round(0.5 * (1 - ratio), 4)
    out["cascade_break_even_escalation"] = round(1 - ratio, 4)
    out["cascade_saving_at_20pct_escalation"] = round(1 - ratio - 0.2, 4)
    if measured and measured.get("strong_usd") and measured.get("cheap_usd"):
        out["measured_ratio"] = round(measured["cheap_usd"] / measured["strong_usd"], 4)
        out["measured_saving_per_downshifted_request"] = round(1 - out["measured_ratio"], 4)
    # cache-minimum effect: a cached shared prefix of P tokens on strong but (if P < cheap's minimum) not on cheap
    ks, kc = prices.billing.get("strong"), prices.billing.get("cheap")
    ms, mc = CACHE_MIN_PREFIX.get(ks), CACHE_MIN_PREFIX.get(kc)
    if ms and mc:
        P, extra, O = 3000, 500, 400
        strong_cached = (s.cached_input * P + s.input * extra + s.output * O) / 1e6 if P >= ms else per_req(s, P + extra, O)
        cheap_cached = (c.cached_input * P + c.input * extra + c.output * O) / 1e6 if P >= mc else per_req(c, P + extra, O)
        out["cache_minimum_prefix"] = {"strong": ms, "cheap": mc}
        out["example_3000_token_cached_prefix"] = {
            "request": "3,000-token shared prefix (cached where the model allows) + 500 new input + 400 output",
            "strong_usd": round(strong_cached, 6), "cheap_usd": round(cheap_cached, 6),
            "saving_per_downshifted_request": round(1 - cheap_cached / strong_cached, 4)}
    return out


def all_pairs(settings: Settings, policy) -> dict:
    out = {}
    for b in policy.backends:
        try:
            pb = PriceBook(settings.prices_path, policy.billing_for(b))
            g = price_gap(pb)
            out[b] = {"strong": pb.billing.get("strong"), "cheap": pb.billing.get("cheap"),
                      "cost_ratio": g["cost_ratio_cheap_over_strong"],
                      "saving_per_downshifted_request": g["saving_per_downshifted_request"]}
        except Exception:
            continue
    return out


def print_economics(gap: dict, pairs: dict, backend: str, system_tokens: int, quiet: bool) -> list[str]:
    s, c, r = gap["strong"], gap["cheap"], gap["reference_request"]
    lines = [
        f"Price gap ({backend}): strong={s['billed_as']} ${s['input']}/${s['output']} vs cheap={c['billed_as']} "
        f"${c['input']}/${c['output']} per 1M tokens (configs/prices.yaml).",
        f"  1,500-in/400-out request: strong ${r['strong_usd']:.5f} vs cheap ${r['cheap_usd']:.5f} -> cheap costs "
        f"{gap['cost_ratio_cheap_over_strong']:.0%} of strong; each downshifted request saves "
        f"{gap['saving_per_downshifted_request']:.0%}.",
        f"  Router: savings ~= r x {gap['saving_per_downshifted_request']:.0%} (r = share routed cheap); routing 50% "
        f"saves ~{gap['router_saving_at_50pct_routed']:.0%} of that traffic's spend.",
        f"  Cascade (cheap first, escalate on failure): cost ~= p_cheap + e x p_strong; breaks even at "
        f"e = {gap['cascade_break_even_escalation']:.0%} escalation (before verifier cost); at e=20% it saves "
        f"{gap['cascade_saving_at_20pct_escalation']:.0%}.",
    ]
    if "measured_saving_per_downshifted_request" in gap:
        lines.append(f"  Measured on this run (actual tokens, both tiers): each downshifted request saves "
                     f"{gap['measured_saving_per_downshifted_request']:.0%}.")
    if "example_3000_token_cached_prefix" in gap:
        ex, mins = gap["example_3000_token_cached_prefix"], gap["cache_minimum_prefix"]
        lines.append(f"  Prompt caching: minimum cacheable prefix {mins['strong']} (strong) vs {mins['cheap']} (cheap) "
                     f"tokens. With a 3,000-token cached prefix the saving per downshifted request drops to "
                     f"{ex['saving_per_downshifted_request']:.0%}. Today's system prompt is ~{system_tokens} tokens, "
                     f"below both minimums, so neither tier caches it yet.")
    lines.append("  Other configured pairs: " + "; ".join(
        f"{b}: {v['cheap']}/{v['strong']} = {v['cost_ratio']:.0%} -> saves {v['saving_per_downshifted_request']:.0%}"
        for b, v in pairs.items()))
    for ln in lines:
        log(ln, quiet)
    return lines


# ============================================================================================ main run
def _is_full_gate(path: Path) -> bool:
    try:
        d = json.loads(path.read_text())
        return not d.get("dry_run") and not d.get("partial")
    except Exception:
        return False


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)                 # atomic: the router never sees a half-written gate file


def run(args) -> dict:
    t0 = time.time()
    settings = Settings.from_env()
    if args.backend:
        settings.backend = args.backend
    policy = load_policy(settings.policy_path)
    backend = settings.backend
    if backend not in policy.backends:
        raise SystemExit(f"backend {backend!r} has no model pair in {settings.policy_path}")
    paid = backend not in LOCAL_BACKENDS
    dry_run = bool(args.dry_run)
    if backend == "mock" and not dry_run:
        log("NOTE: mock answers carry no quality signal -> running as --dry-run (nothing will be allowed).", args.quiet)
        dry_run = True
    if args.cassette:
        settings.cassette = Path(args.cassette)
    elif settings.cassette is None and backend != "mock":
        settings.cassette = GATE_CASSETTE
    if args.replay:
        if settings.cassette is None:
            raise SystemExit("--replay needs a cassette (set --cassette or COSTGUARD_CASSETTE)")
        settings.cassette_mode = "replay"
        os.environ["COSTGUARD_JUDGE_CASSETTE_MODE"] = "replay"
    prices = PriceBook(settings.prices_path, policy.billing_for(backend))
    hcfg = features.HardnessConfig.from_dict(((policy.raw or {}).get("router") or {}).get("hardness"))

    # ---------------------------------------------------------------- items
    items, source, notes = load_items(args)
    partial = bool(args.limit) and args.limit < len(items)
    items = round_robin_limit(items, args.limit)
    attach_context(items, notes, args.quiet)
    for n in notes:
        log("NOTE: " + n, args.quiet)
    cats_present = Counter(it["category"] for it in items)
    log(f"{len(items)} items from {source}: " + ", ".join(f"{c}={cats_present.get(c, 0)}" for c in CATEGORIES), args.quiet)

    # ---------------------------------------------------------------- plan + spend guard
    judge_backend = "proxy" if args.judge == "proxy" else (os.environ.get("COSTGUARD_JUDGE_BACKEND") or backend)
    est = estimate(items, settings, policy, prices, args.scorer, judge_backend)
    log(f"Plan: {est['generations_total']} generations ({est['generations_cached']} cached, new: "
        f"{est['generations_new'] or 0}) + up to {est['judge_calls_upper_bound']} judge calls on {judge_backend}. "
        f"Estimated cost: <= ${est['usd_total_upper_bound']:.2f} (generation typical ~${est['usd_generation_typical']:.2f}, "
        f"judge <= ${est['usd_judge_upper_bound']:.2f})" + ("." if paid else " - list-price equivalent; this "
                                                             "local backend costs $0."), args.quiet)
    judge_paid = judge_backend not in LOCAL_BACKENDS + ("proxy",)
    needs_money = (paid and sum(est["generations_new"].values()) > 0) or (judge_paid and est["judge_calls_upper_bound"] > 0)
    if needs_money and not args.replay and not args.yes:
        log(f"Paid backend ({backend}, judge {judge_backend}): re-run with --yes to spend up to "
            f"${est['usd_total_upper_bound']:.2f}, or --replay to score from cassettes only ($0). Nothing was called.")
        return {"status": "needs_confirmation", "estimate": est}

    # ---------------------------------------------------------------- generate + judge (sequential)
    prompt_cache = None                 # provider prompt caching: only the anthropic adapter has it
    if "anthropic" in (backend, judge_backend):
        from eval.run_ab import set_provider_prompt_cache
        prompt_cache = set_provider_prompt_cache(args.provider_prompt_cache)
    engine = make_engine(settings, args.replay)
    judge, judge_label = make_judge(args.judge, settings, engine)
    log(f"judge: {judge_label}", args.quiet)
    rows, errors = [], Counter()
    for k, it in enumerate(items, 1):
        t1 = time.time()
        row = {k2: it[k2] for k2 in ("id", "category", "type", "source", "pair_id", "query")}
        row["reference"] = it.get("reference")
        row["context_docs"] = len(it["context"])
        try:
            st = generate(engine, it, "strong")
            ch = generate(engine, it, "cheap")
        except Exception as e:
            errors["generation"] += 1
            log(f"[{k}/{len(items)}] {it['category']:<8} {it['id']}: generation failed ({type(e).__name__}: {str(e)[:120]})", args.quiet)
            continue
        rin = RouteInput(query=it["query"], category=it["category"], input_tokens=st["input_tokens_original"],
                         has_context=bool(it["context"]), history_turns=0)
        h = features.from_route_input(rin, hcfg)
        row.update(strong=st, cheap=ch, hard=h.hard, hard_reasons=h.reasons)
        ref = it.get("reference")
        if args.scorer in ("grade", "both"):
            gs, gc = judge.grade(it["query"], st["text"], ref), judge.grade(it["query"], ch["text"], ref)
            row["grade_strong"], row["grade_cheap"] = gs, gc
            row["diff_grade"] = None if gs is None or gc is None else round(100 * (gc - gs), 3)
            if row["diff_grade"] is None:
                errors["grade_missing"] += 1
        if args.scorer in ("pairwise", "both"):
            v = judge.pairwise(it["query"], st["text"], ch["text"], ref)       # A = strong, B = cheap
            row["pairwise"] = {"A": "strong", "B": "cheap", "tie": "tie"}.get(v, "error")
            row["diff_pairwise"] = {"cheap": 100.0, "strong": -100.0, "tie": 0.0}.get(row["pairwise"])
            if row["diff_pairwise"] is None:                 # failed judgment: missing, never a tie
                errors["pairwise_missing"] += 1
        rows.append(row)
        log(f"[{k}/{len(items)}] {it['category']:<8} {it['id']:<22} {'HARD:' + h.primary if h.hard else 'easy':<26} "
            f"grade s/c={row.get('grade_strong')}/{row.get('grade_cheap')} pw={row.get('pairwise', '-')} "
            f"${st['cost_usd'] + ch['cost_usd']:.5f} {time.time() - t1:.1f}s", args.quiet)

    # ---------------------------------------------------------------- per-category gate
    decision_metric = args.decision if args.scorer == "both" else args.scorer
    dkey = f"diff_{decision_metric}"
    traffic = _traffic_mix(items)
    cats, warnings = {}, []
    for cat in CATEGORIES:
        all_c = [r for r in rows if r["category"] == cat]
        rt = [r for r in all_c if not r["hard"]] if args.subset == "routable" else all_c
        scored = [r for r in rt if r.get(dkey) is not None]
        d = [r[dkey] for r in scored]
        clusters = [r.get("pair_id") or r["id"] for r in scored]
        mean, lo, hi = bootstrap(d, clusters, args.bootstrap, args.seed) if d else (math.nan,) * 3
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else math.nan
        base = sum(r["strong"]["cost_usd"] for r in all_c)
        served = sum(r["cheap"]["cost_usd"] if (not r["hard"]) else r["strong"]["cost_usd"] for r in all_c)
        saving = (1 - served / base) if base > 0 else 0.0
        n = len(d)
        coverage = n / len(rt) if rt else math.nan
        if dry_run:
            allow, why = False, "dry-run"
        elif not rt:
            allow, why = False, "no data"
        elif coverage < MIN_JUDGE_COVERAGE:
            allow, why = False, f"judge-coverage {100 * coverage:.1f}% < {100 * MIN_JUDGE_COVERAGE:.0f}%"
        elif n < args.min_n:
            allow, why = False, f"n={n} < min_n={args.min_n}"
        elif not (lo >= -args.margin):
            allow, why = False, f"CI lower bound {lo:.1f} < -{args.margin:g}"
        elif not (saving > args.min_saving):
            allow, why = False, f"measured saving {saving:.1%} <= {args.min_saving:.0%}"
        else:
            allow, why = True, f"CI lower bound {lo:.1f} >= -{args.margin:g}"
        if 0 < n < args.min_n or (n == 0 and cats_present.get(cat)):
            warnings.append(f"WARN: {cat}: n={n} routable+scored items < {args.min_n}: the 95% CI is too wide to allow "
                            f"a downshift; it stays on strong.")
        if rt and coverage < MIN_JUDGE_COVERAGE:
            warnings.append(f"WARN: {cat}: {len(rt) - n} of {len(rt)} routable items have no valid judgment "
                            f"(failed or unparseable): judge-coverage {100 * coverage:.1f}% < "
                            f"{100 * MIN_JUDGE_COVERAGE:.0f}%, so it stays on strong.")
        entry = {"allow": allow, "n": n, "n_items": len(all_c), "n_routable": len(rt),
                 "judge_coverage": _r(coverage, 4), "n_unjudged": len(rt) - n,
                 "hard_share": round(1 - len(rt) / len(all_c), 3) if all_c and args.subset == "routable" else 0.0,
                 "diff": _r(mean), "ci": [_r(lo), _r(hi)], "sd": _r(sd), "reason": why,
                 "n_needed": n_needed(sd, mean, args.margin) if n > 1 else None,
                 "savings_if_allowed": round(saving, 4), "traffic_share": round(traffic.get(cat, 0.0), 4)}
        if args.scorer == "both" or args.scorer == "pairwise":
            pw = Counter(r.get("pairwise") for r in rt if r.get("pairwise"))
            npw = pw["cheap"] + pw["tie"] + pw["strong"]           # valid judgments only; errors are missing
            nir = wilson(pw["cheap"] + pw["tie"], npw) if npw else (math.nan, 0.0, 1.0)
            entry["pairwise"] = {"cheap_wins": pw["cheap"], "ties": pw["tie"], "strong_wins": pw["strong"],
                                 "errors": pw["error"], "coverage": _r(npw / len(rt), 4) if rt else None,
                                 "non_inferior_rate": _r(nir[0], 4), "non_inferior_ci": [_r(nir[1], 4), _r(nir[2], 4)]}
        if args.scorer == "both":
            alt = "pairwise" if decision_metric == "grade" else "grade"
            da = [r[f"diff_{alt}"] for r in rt if r.get(f"diff_{alt}") is not None]
            m2, l2, h2 = bootstrap(da, None, args.bootstrap, args.seed) if da else (math.nan,) * 3
            entry[f"diff_{alt}"] = {"mean": _r(m2), "ci": [_r(l2), _r(h2)]}
        if args.subset == "routable":
            da = [r[dkey] for r in all_c if r.get(dkey) is not None]
            m3, l3, h3 = bootstrap(da, [r.get("pair_id") or r["id"] for r in all_c if r.get(dkey) is not None],
                                   args.bootstrap, args.seed) if da else (math.nan,) * 3
            entry["all_items_diff"] = {"n": len(da), "mean": _r(m3), "ci": [_r(l3), _r(h3)]}
        cats[cat] = entry
    for w in warnings:
        log(w, args.quiet)

    # ---------------------------------------------------------------- economics + summary
    strong_usd = sum(r["strong"]["cost_usd"] for r in rows)
    cheap_usd = sum(r["cheap"]["cost_usd"] for r in rows)
    gap = price_gap(prices, {"strong_usd": strong_usd, "cheap_usd": cheap_usd})
    pairs = all_pairs(settings, policy)
    econ_lines = print_economics(gap, pairs, backend, int(count_text(policy.system_prompt) * TOKEN_RATIO.get(backend, 1.0)), args.quiet)
    allowed = [c for c, e in cats.items() if e["allow"]]
    w = {c: cats[c]["traffic_share"] * (sum(r["strong"]["cost_usd"] for r in rows if r["category"] == c) /
                                        max(1, cats[c]["n_items"])) for c in CATEGORIES}
    tot = sum(w.values())
    expected = sum(w[c] * cats[c]["savings_if_allowed"] for c in allowed) / tot if tot > 0 else 0.0
    hard_by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        hard_by_type[r["type"]][0] += int(r["hard"])
        hard_by_type[r["type"]][1] += 1

    judge_desc = judge.describe() if hasattr(judge, "describe") else {"judge": judge_label}
    created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    gate = {
        "version": 1, "created": created, "margin": args.margin, "min_n": args.min_n, "min_saving": args.min_saving,
        "judge": judge_label, "metric": f"{decision_metric} diff (cheap - strong), points", "subset": args.subset,
        "dry_run": dry_run, "partial": partial, "backend": backend,
        "models": {a: policy.model_id(backend, a) for a in ("strong", "cheap")},
        "billing": {a: prices.billing.get(a) for a in ("strong", "cheap")},
        "data_source": source, "policy_hash": policy.config_hash, "provider_prompt_cache": prompt_cache,
        "min_judge_coverage": MIN_JUDGE_COVERAGE,
        "categories": {c: {k: cats[c][k] for k in ("allow", "n", "judge_coverage", "diff", "ci", "reason",
                                                   "savings_if_allowed")} for c in CATEGORIES},
    }
    results = {
        **{k: v for k, v in gate.items() if k != "categories"},
        "allowed": allowed, "expected_routing_savings_if_deployed": round(expected, 4),
        "traffic_source": "trace_v1" if TRACE_PATH.exists() else "eval items",
        "categories": cats, "warnings": warnings, "notes": notes, "errors": dict(errors),
        "n_items": len(items), "n_scored": len(rows),
        "hard_signal_rate": {t: {"flagged": a, "n": b, "rate": round(a / b, 3)} for t, (a, b) in hard_by_type.items()},
        "estimate": est, "spend": {"generation_usd_list_price": round(strong_usd + cheap_usd, 6),
                                   "judge": judge_desc}, "price_gap": gap, "price_gap_all_backends": pairs,
        "economics_summary": econ_lines, "seconds": round(time.time() - t0, 1),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }

    # ---------------------------------------------------------------- write
    suffix = "_dryrun" if dry_run else ("_partial" if partial else "")
    gate_out = Path(args.gate_out) if args.gate_out else GATE_PATH
    res_out = Path(args.results_out) if args.results_out else RESULTS_DIR / f"router_gate{suffix}.json"
    items_out = res_out.with_name(res_out.stem + "_items.json")
    _write(res_out, results)
    _write(items_out, {"created": created, "judge": judge_label, "items": rows})
    wrote_gate = None
    if (dry_run or partial) and not args.gate_out and gate_out.exists() and _is_full_gate(gate_out) and not args.force:
        log(f"Kept the existing full gate file {gate_out} (this run is {'dry' if dry_run else 'partial'}; --force to replace).", args.quiet)
    else:
        _write(gate_out, gate)
        wrote_gate = gate_out
    # repo-relative, so committed results never carry a local absolute path (/Users/<name>/...)
    results["written"] = {"gate": _rel(wrote_gate) if wrote_gate else None, "results": _rel(res_out),
                          "items": _rel(items_out)}
    _write(res_out, results)

    log("", args.quiet)
    log(f"{'category':<9} {'n':>4} {'judged':>7} {'diff':>7} {'95% CI':>17} {'save':>6}  decision", args.quiet)
    for c in CATEGORIES:
        e = cats[c]
        ci = f"[{_fmt(e['ci'][0])}, {_fmt(e['ci'][1])}]"
        cov = "-" if e["judge_coverage"] is None else f"{e['judge_coverage']:.0%}"
        log(f"{c:<9} {e['n']:>4} {cov:>7} {_fmt(e['diff']):>7} {ci:>17} {e['savings_if_allowed']:>6.0%}  "
            f"{'ALLOW' if e['allow'] else 'strong'} ({e['reason']})", args.quiet)
    log(f"allowed: {allowed or 'none'}; expected routing savings if deployed: {expected:.1%}; judge {judge_label}; "
        f"{'DRY RUN; ' if dry_run else ''}{'PARTIAL; ' if partial else ''}gate -> {wrote_gate or '(unchanged)'}; "
        f"results -> {res_out}", args.quiet)
    return results


def _traffic_mix(items: list[dict]) -> dict:
    if TRACE_PATH.exists():
        c = Counter(r.get("category") for r in _read_jsonl(TRACE_PATH) if r.get("category") in CATEGORIES)
    else:
        c = Counter(it["category"] for it in items)
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


def _r(x: float, nd: int = 2) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _fmt(x) -> str:
    return "-" if x is None else f"{x:+.1f}"


def parse_args(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(prog="python -m eval.gate_router", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", help="override COSTGUARD_BACKEND (mock | mlx | anthropic | openai | gemini)")
    ap.add_argument("--dry-run", action="store_true", help="exercise everything; write a gate marked dry_run that allows nothing")
    ap.add_argument("--yes", action="store_true", help="confirm spending on a paid backend")
    ap.add_argument("--replay", action="store_true", help="cassette replay only (guaranteed $0; misses are skipped)")
    ap.add_argument("--limit", type=int, help="cap the number of items (round-robin across categories)")
    ap.add_argument("--data", help="jsonl of eval rows (default: eval/data/evalset/*.jsonl, else Bitext + KB seeds)")
    ap.add_argument("--source", default="auto", choices=["auto", "evalset", "bitext", "bitext+kb"])
    ap.add_argument("--per-category", type=int, default=40, help="Bitext fallback sample size per category")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scorer", default="both", choices=["grade", "pairwise", "both"])
    ap.add_argument("--decision", default="grade", choices=["grade", "pairwise"], help="metric the gate decides on")
    ap.add_argument("--judge", default="auto", choices=["auto", "proxy"])
    ap.add_argument("--subset", default="routable", choices=["routable", "all"],
                    help="decide on the items the router would downshift (no hardness signal), or on all items")
    ap.add_argument("--margin", type=float, default=5.0, help="non-inferiority margin, quality points (0-100 scale)")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--min-saving", type=float, default=0.0, help="required measured saving on the category's traffic")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--cassette", help="generation cassette (default eval/cassettes/gate.jsonl on real backends)")
    ap.add_argument("--provider-prompt-cache", action="store_true",
                    help="keep Anthropic prompt caching on (default off: cassette replay would freeze the first "
                         "call's cache-write usage; see docs/EVALUATION.md section 2)")
    ap.add_argument("--gate-out", help=f"gate file (default {_rel(GATE_PATH)})")
    ap.add_argument("--results-out", help="results json (default eval/results/router_gate[_dryrun|_partial].json)")
    ap.add_argument("--force", action="store_true", help="let a dry/partial run replace a full gate file")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    res = run(parse_args(argv))
    return 2 if res.get("status") == "needs_confirmation" else 0


if __name__ == "__main__":
    sys.exit(main())
