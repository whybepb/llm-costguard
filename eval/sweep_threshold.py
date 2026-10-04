"""Semantic-cache threshold calibration -> eval/results/threshold_sweep.json

    python -m eval.sweep_threshold                 # full run (~5-10 min on a laptop CPU, no LLM calls)
    python -m eval.sweep_threshold --quick         # 1,500-request stream, primary model only (smoke test)

Two views, both for tau = 0.70 ... 0.99 (cosine *similarity*, step 0.01), with and without the guards:

1. Pairwise curve on the labelled pairs (eval/data/cache_pairs/pairs_v1.jsonl): precision / recall / false-positive rate
   per source (bitext, qqp, trap, seed). Useful for comparing models, but it is NOT the operating metric: a cache never
   sees a balanced set of pairs, it sees the nearest neighbour of each new query among everything cached so far.

2. Cache simulation (the one the recommendation uses). A replay stream of Bitext queries with Zipf-like intent
   popularity and ~30% repeats is fed, in arrival order, through the real SemanticCacheImpl (same code as serving;
   only the embeddings are precomputed). A hit is correct when the served entry has the same (refined) intent and the
   same specifics as the query (see eval/cache_pairs.py). Metrics, per the project contract:
       hit rate          = hits / requests
       false-hit rate    = wrong hits / requests        (per request, not per hit)
       precision per hit = correct hits / hits
   Two renderings of the same stream: "templated" keeps Bitext's {{Order Number}} placeholders (so two order-number
   queries carry the same slot), "filled" substitutes sampled order numbers / tiers / cities / names, as real
   customers would type them.

Recommendation per mode: the tau with the highest hit rate whose false-hit rate stays within the mode's budget
(quality 0.5%, balanced 1%, economy 3%) with guards on, required in BOTH renderings (we take the stricter tau).
The coordinator copies the values into configs/policy.yaml; this script never edits it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from costguard.cache import guards as G
from costguard.cache.embedder import DEFAULT_MODEL, get_embedder
from costguard.cache.semantic import MemoryStore, SemanticCacheImpl
from costguard.config import ROOT
from costguard.schemas import CacheEntry
from eval import cache_pairs as CP

RESULTS = ROOT / "eval" / "results"
OUT = RESULTS / "threshold_sweep.json"
COMPARE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TAUS = [round(t, 2) for t in np.arange(0.70, 0.9901, 0.01)]
BUDGETS = {"quality": 0.005, "balanced": 0.01, "economy": 0.03}
SEED = 13

# ============================================================================================ replay stream

FILL_POOLS = {
    "Order Number": [f"SN-{n}" for n in np.random.default_rng(1).choice(np.arange(10000, 99999), 300, replace=False)],
    "Invoice Number": [f"INV-{n}" for n in np.random.default_rng(2).choice(np.arange(100000, 999999), 100, replace=False)],
    "Refund Amount": ["49.99", "120", "15.50", "899", "2,499", "75", "1,299", "350", "64.20", "18", "4,999", "230",
                      "99", "560", "12.75", "1,850", "42", "310", "7.99", "999"],
    "Currency Symbol": ["₹", "$"],
    "Account Type": ["free", "freemium", "premium", "gold", "platinum", "pro", "standard", "business"],
    "Account Category": ["free", "freemium", "premium", "gold", "platinum", "pro", "standard", "business"],
    "Delivery City": ["Pune", "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Nagpur",
                      "Indore", "Lucknow", "Surat", "Kochi", "Bhopal", "Patna", "Austin", "Denver", "Leeds", "Lyon",
                      "Osaka", "Toronto", "Dubai", "London", "Sydney"],
    "Delivery Country": ["India", "Canada", "Germany", "Australia", "Brazil", "Japan", "Kenya", "Norway", "Chile",
                         "Vietnam", "Mexico", "Spain", "Singapore", "Ireland", "Egypt", "Poland"],
    "Person Name": ["Priya Sharma", "John Smith", "Aisha Khan", "Rahul Verma", "Maria Garcia", "Chen Wei",
                    "Fatima Ali", "Arjun Nair", "Emma Brown", "Lucas Silva", "Sara Cohen", "Kenji Sato",
                    "Neha Gupta", "Omar Haddad", "Olivia Jones", "Ravi Iyer"],
}


_NAMED_SLOTS = {"Person Name", "Delivery City", "Delivery Country"}


@dataclass
class Item:
    text: str
    label: str
    key: tuple          # (label, specifics): two items with equal keys may share one cached answer
    dup_of: Optional[int]


def build_skeleton(df: pd.DataFrame, n: int, dup_rate: float, zipf_s: float, seed: int) -> list[tuple[int, Optional[int]]]:
    """[(row_index, dup_of_position)] -- fresh draws follow Zipf over intents; repeats re-ask an earlier query."""
    rng = np.random.default_rng(seed)
    labels = sorted(df["label"].unique())
    order = list(rng.permutation(labels))
    w = 1.0 / np.arange(1, len(order) + 1) ** zipf_s
    p = w / w.sum()
    pools = {l: list(rng.permutation(np.flatnonzero(df["label"].to_numpy() == l))) for l in labels}
    cursor = Counter()
    sk: list[tuple[int, Optional[int]]] = []
    for pos in range(n):
        if sk and rng.random() < dup_rate:
            j = int(rng.integers(len(sk)))
            sk.append((sk[j][0], j if sk[j][1] is None else sk[j][1]))
        else:
            l = order[int(rng.choice(len(order), p=p))]
            pool = pools[l]
            sk.append((int(pool[cursor[l] % len(pool)]), None))
            cursor[l] += 1
    return sk


def render(sk, df: pd.DataFrame, mode: str, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed + 7)
    texts, labels = df["instruction"].tolist(), df["label"].tolist()
    items: list[Item] = []
    for row, dup in sk:
        if dup is not None:
            o = items[dup]
            items.append(Item(o.text, o.label, o.key, dup))
            continue
        t = texts[row]
        spec = set(CP.specifics(t))
        if mode == "filled":
            filled = {}

            def sub(m):
                name = m.group(1)
                pool = FILL_POOLS.get(name)
                if not pool:
                    return m.group(0)
                v = pool[int(rng.integers(len(pool)))]
                if name in _NAMED_SLOTS:          # numbers and tiers are picked up by CP.specifics() below
                    filled.setdefault(name, []).append(v.lower())
                return v
            t = CP._PH.sub(sub, t)
            spec = set(CP.specifics(t)) | {f"{k}={v}" for k, vs in filled.items() for v in vs}
        items.append(Item(t, labels[row], (labels[row], frozenset(spec)), None))
    return items


# ============================================================================================ simulation


class PrecomputedEmbedder:
    """Duck-types costguard.cache.embedder.Embedder with vectors computed once up front (same model)."""

    def __init__(self, vecs: dict[str, np.ndarray], model_name: str):
        self.vecs, self.model_name = vecs, model_name
        self.dim = next(iter(vecs.values())).shape[0]

    def embed_one(self, text: str) -> np.ndarray:
        return self.vecs[text]

    def embed(self, texts):
        return np.vstack([self.vecs[t] for t in texts])


def simulate(items: list[Item], emb: PrecomputedEmbedder, tau: float, cfg: G.GuardConfig,
             checkpoint_every: int = 500) -> dict:
    cache = SemanticCacheImpl(emb, MemoryStore(), ttl_seconds=None, guards=cfg)
    hits = wrong = exact_text = 0
    rej = defaultdict(lambda: [0, 0])          # reason -> [would_be_wrong (true reject), would_be_correct (false reject)]
    wrong_pairs = Counter()
    cps = []
    for i, it in enumerate(items):
        h = cache.lookup(it.text, "p", tau)
        for reason, rec in cache.last_lookup.get("rejected", []):
            j = rec.entry.metadata["pos"]
            rej[reason.split(":")[0]][0 if items[j].key != it.key else 1] += 1
        if h.entry is not None:
            hits += 1
            j = h.entry.metadata["pos"]
            if items[j].text == it.text:
                exact_text += 1
            if items[j].key != it.key:
                wrong += 1
                wrong_pairs[f"{items[j].label} -> {it.label}" if items[j].label != it.label else f"{it.label} (specifics)"] += 1
        else:
            cache.insert(it.text, "p", CacheEntry(entry_id=str(i), query_text=it.text, response_text="",
                                                  model_alias="strong", input_tokens=0, output_tokens=0,
                                                  metadata={"pos": i}))
        if (i + 1) % checkpoint_every == 0:
            cps.append([i + 1, round(hits / (i + 1), 5), round(wrong / (i + 1), 5)])
    n = len(items)
    return {"tau": tau, "n": n, "hits": hits, "wrong_hits": wrong, "exact_text_hits": exact_text,
            "hit_rate": hits / n, "false_hit_rate": wrong / n, "precision_per_hit": (hits - wrong) / hits if hits else None,
            "entries": cache.store.count(), "guard_rejections": {k: {"true_reject": v[0], "false_reject": v[1]} for k, v in rej.items()},
            "top_wrong": wrong_pairs.most_common(6), "checkpoints": cps,
            "search_ms_p50": float(np.percentile(cache.search_ms_history, 50)),
            "search_ms_p99": float(np.percentile(cache.search_ms_history, 99))}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    try:
        from eval.stats import proportion_ci   # shared helper (eval workstream), same Wilson interval
        _, lo, hi = proportion_ci(k, n)
        return lo, hi
    except Exception:
        if not n:
            return 0.0, 1.0
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
        return max(0.0, c - h), min(1.0, c + h)


def recommend(runs: list[dict], budget: float) -> Optional[dict]:
    ok = [r for r in runs if r["false_hit_rate"] <= budget]
    if not ok:
        return None
    return max(ok, key=lambda r: (round(r["hit_rate"], 6), r["tau"]))


# ============================================================================================ pairwise


def auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pairwise(rows: list[dict], sims: np.ndarray, verdicts: list[dict]) -> dict:
    out = {}
    first = [next((v for v in vd.values() if v), None) for vd in verdicts]
    for src in sorted({r["source"] for r in rows}) + ["all"]:
        idx = [i for i, r in enumerate(rows) if src == "all" or r["source"] == src]
        y = np.array([rows[i]["label"] for i in idx])
        s = sims[idx]
        g_ok = np.array([first[i] is None for i in idx])
        cur = defaultdict(list)
        for t in TAUS:
            for tag, pred in (("", s >= t), ("_guarded", (s >= t) & g_ok)):
                tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
                fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
                cur["precision" + tag].append(round(tp / (tp + fp), 4) if tp + fp else None)
                cur["recall" + tag].append(round(tp / (tp + fn), 4) if tp + fn else None)
                cur["fpr" + tag].append(round(fp / (fp + tn), 4) if fp + tn else None)
        # per-guard verdicts over pairs that clear 0.80 (each guard evaluated on its own)
        by_guard = {}
        for g in ("numbers", "negation", "entities", "content"):
            hit = [i for i in idx if sims[i] >= 0.80 and verdicts[i][g]]
            by_guard[g] = {"rejects_label0": sum(rows[i]["label"] == 0 for i in hit),
                           "rejects_label1": sum(rows[i]["label"] == 1 for i in hit)}
        out[src] = {"n": len(idx), "positives": int((y == 1).sum()), "negatives": int((y == 0).sum()),
                    "roc_auc": None if src == "trap" else (round(auc(s, y), 4) if auc(s, y) is not None else None),
                    "curve": dict(cur), "per_guard_at_0.80": by_guard}
    traps = [i for i, r in enumerate(rows) if r["source"] == "trap"]
    out["trap_detail"] = [{"a": rows[i]["query_a"], "b": rows[i]["query_b"], "type": rows[i].get("trap_type"),
                           "sim": round(float(sims[i]), 4), "guard": first[i]} for i in traps]
    return out


# ============================================================================================ latency


def latency_profile(stream_texts: list[str], n: int = 400) -> dict:
    emb = get_embedder(DEFAULT_MODEL)
    emb.warmup()
    uniq = list(dict.fromkeys(stream_texts))
    probe = uniq[:n]
    ms = []
    for t in probe:
        t0 = time.perf_counter()
        emb.embed_one(t)
        ms.append((time.perf_counter() - t0) * 1000)
    # guard cost on cold analysis caches (worst case: both texts unseen)
    G.analyse.cache_clear()
    rng = np.random.default_rng(0)
    gms = []
    for _ in range(1000):
        a, b = rng.choice(len(uniq), 2)
        t0 = time.perf_counter()
        G.check(uniq[a], uniq[b])
        gms.append((time.perf_counter() - t0) * 1000)
    # end-to-end lookup() on a real cache (real embedder) holding 2,000 entries
    fill = uniq[n: n + 2000]
    cache = SemanticCacheImpl(PrecomputedEmbedder(dict(zip(fill, emb.embed(fill, batch_size=256))), emb.model_name),
                              MemoryStore(), ttl_seconds=None)
    for t in fill:
        cache.insert(t, "p", CacheEntry(entry_id=t[:8], query_text=t, response_text="", model_alias="strong",
                                        input_tokens=0, output_tokens=0))
    cache.embedder = emb      # lookups below pay the real embedding cost
    lk = []
    for t in probe[:200]:
        t0 = time.perf_counter()
        cache.lookup(t, "p", 0.9)
        lk.append((time.perf_counter() - t0) * 1000)

    def q(xs):
        return {"p50": round(float(np.percentile(xs, 50)), 3), "p95": round(float(np.percentile(xs, 95)), 3),
                "p99": round(float(np.percentile(xs, 99)), 3), "n": len(xs)}
    return {"model": emb.model_name, "model_load_ms": round(emb.load_ms, 1), "embed_one_ms": q(ms),
            "guard_check_ms_cold": q(gms), "lookup_end_to_end_ms_2k_entries": q(lk),
            "cost_usd_per_lookup": 0.0, "note": "CPU, single query per call, onnxruntime via fastembed"}


# ============================================================================================ main


def run_model(model: str, rows, pair_texts, streams: dict[str, list[Item]], guard_modes, log) -> dict:
    texts = sorted(set(pair_texts) | {it.text for its in streams.values() for it in its})
    t0 = time.perf_counter()
    E = CP.embed_texts(texts, model)
    log(f"  embedded {len(texts):,} texts with {model} in {time.perf_counter() - t0:.1f}s")
    vec = {t: E[i] for i, t in enumerate(texts)}
    emb = PrecomputedEmbedder(vec, model)
    sims = np.array([float(vec[r["query_a"]] @ vec[r["query_b"]]) for r in rows])
    verdicts = [G.check_all(r["query_a"], r["query_b"]) for r in rows]
    res = {"pairwise": pairwise(rows, sims, verdicts), "simulation": {}, "guard_effect": {}, "recommended": {}}
    for name, items in streams.items():
        res["simulation"][name] = {}
        for gname, cfg in guard_modes:
            t1 = time.perf_counter()
            runs = [simulate(items, emb, t, cfg) for t in TAUS]
            log(f"  sim {name:9s} guards={gname:3s}: {time.perf_counter() - t1:.1f}s")
            res["simulation"][name][f"guards_{gname}"] = {
                k: [r[k] for r in runs] for k in ("tau", "hits", "wrong_hits", "exact_text_hits", "hit_rate",
                                                  "false_hit_rate", "precision_per_hit", "entries")} | {
                "guard_rejections": [r["guard_rejections"] for r in runs], "top_wrong": [r["top_wrong"] for r in runs],
                "cumulative": [r["checkpoints"] for r in runs], "search_ms_p50": [round(r["search_ms_p50"], 4) for r in runs],
                "search_ms_p99": [round(r["search_ms_p99"], 4) for r in runs], "_runs": runs}
        sim = res["simulation"][name]
        if "guards_on" in sim and "guards_off" in sim:
            on, off = sim["guards_on"], sim["guards_off"]
            res["guard_effect"][name] = {
                "tau": TAUS,
                # net effect of running the same stream with guards on vs off. correct_hits_change can be positive:
                # a refused near-hit is answered fresh and cached, so later paraphrases find a correct neighbour.
                "false_hits_removed": [b - a for a, b in zip(on["wrong_hits"], off["wrong_hits"])],
                "correct_hits_change": [(a - aw) - (b - bw) for a, aw, b, bw in
                                        zip(on["hits"], on["wrong_hits"], off["hits"], off["wrong_hits"])],
                "false_hit_rate_off": off["false_hit_rate"], "false_hit_rate_on": on["false_hit_rate"],
                "hit_rate_off": off["hit_rate"], "hit_rate_on": on["hit_rate"]}
    for gname, _ in guard_modes:
        rec = {}
        for mode, budget in BUDGETS.items():
            per = {}
            for name in streams:
                r = recommend(res["simulation"][name][f"guards_{gname}"]["_runs"], budget)
                per[name] = r["tau"] if r else None
            taus = [t for t in per.values() if t is not None]
            tau = max(taus) if len(taus) == len(per) else None
            entry = {"tau": tau, "budget_false_hit_rate": budget, "per_rendering_tau": per}
            if tau is not None:
                k = TAUS.index(tau)
                for name in streams:
                    r = res["simulation"][name][f"guards_{gname}"]["_runs"][k]
                    lo, hi = wilson(r["wrong_hits"], r["n"])
                    entry[name] = {"hit_rate": round(r["hit_rate"], 4), "false_hit_rate": round(r["false_hit_rate"], 4),
                                   "false_hit_rate_ci95": [round(lo, 4), round(hi, 4)],
                                   "precision_per_hit": round(r["precision_per_hit"], 4) if r["precision_per_hit"] else None,
                                   "exact_text_hit_rate": round(r["exact_text_hits"] / r["n"], 4)}
            rec[mode] = entry
        res["recommended"][f"guards_{gname}"] = rec
    for name in streams:
        for g in res["simulation"][name].values():
            g.pop("_runs", None)
    return res


def _round(o):
    if isinstance(o, float):
        return round(o, 5)
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    return o


def print_table(out: dict) -> None:
    m = out["models"][out["embedding_model"]]
    t_on, t_off = m["simulation"]["templated"]["guards_on"], m["simulation"]["templated"].get("guards_off")
    f_on = m["simulation"]["filled"]["guards_on"]
    pw = m["pairwise"]["bitext"]["curve"]
    print(f"\nembedding model: {out['embedding_model']}  (cosine similarity)   stream n={out['datasets']['stream']['n']}")
    print(" tau | templated hit / false-hit (guards on) | guards off false-hit | filled hit / false-hit (on) | bitext pair P/R")
    for k, t in enumerate(TAUS):
        if k % 2 and t not in {v["tau"] for v in out["recommended"].values() if v.get("tau")}:
            continue
        off = f"{t_off['false_hit_rate'][k]:.2%}" if t_off else "-"
        print(f"{t:.2f} | {t_on['hit_rate'][k]:6.1%} / {t_on['false_hit_rate'][k]:5.2%}               | {off:>7}"
              f"              | {f_on['hit_rate'][k]:6.1%} / {f_on['false_hit_rate'][k]:5.2%}            "
              f"| {pw['precision_guarded'][k] or 0:.2f}/{pw['recall_guarded'][k] or 0:.2f}")
    print("\nrecommended tau (guards on; budget = per-request false-hit rate, must hold in both renderings):")
    for mode, r in out["recommended"].items():
        if r["tau"] is None:
            print(f"  {mode:9s} none meets {r['budget_false_hit_rate']:.1%}: {r['per_rendering_tau']}")
            continue
        print(f"  {mode:9s} tau={r['tau']:.2f}  budget {r['budget_false_hit_rate']:.1%}  "
              f"templated hit {r['templated']['hit_rate']:.1%} / false-hit {r['templated']['false_hit_rate']:.2%}  "
              f"filled hit {r['filled']['hit_rate']:.1%} / false-hit {r['filled']['false_hit_rate']:.2%}")
    cmp = out["models"].get(COMPARE_MODEL)
    if cmp:
        print(f"\ncomparison model {COMPARE_MODEL}: recommended tau (guards on) = "
              + ", ".join(f"{k} {v['tau']}" for k, v in cmp["recommended"]["guards_on"].items()))
    lat = out["latency_ms"]
    print(f"\nlatency: embed_one p50 {lat['embed_one_ms']['p50']} ms / p99 {lat['embed_one_ms']['p99']} ms; "
          f"lookup end-to-end p50 {lat['lookup_end_to_end_ms_2k_entries']['p50']} ms; guards p50 "
          f"{lat['guard_check_ms_cold']['p50']} ms; cost $0")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="semantic-cache threshold sweep")
    ap.add_argument("--n", type=int, default=5000, help="replay stream length")
    ap.add_argument("--dup-rate", type=float, default=0.30)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quick", action="store_true", help="1,500 requests, primary model only")
    ap.add_argument("--no-compare", action="store_true", help="skip the MiniLM comparison")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if a.quick:
        a.n, a.no_compare = 1500, True
    t_start = time.perf_counter()
    log = lambda s: print(s, flush=True)  # noqa: E731

    rows = CP.load_pairs()
    pairs_sha = hashlib.sha256(CP.PAIRS_PATH.read_bytes()).hexdigest()
    pair_texts = [t for r in rows for t in (r["query_a"], r["query_b"])]
    df = CP.load_bitext()
    sk = build_skeleton(df, a.n, a.dup_rate, a.zipf, a.seed)
    streams = {"templated": render(sk, df, "templated", a.seed), "filled": render(sk, df, "filled", a.seed)}
    dup_actual = sum(1 for _, d in sk if d is not None) / len(sk)
    log(f"pairs: {len(rows):,}  stream: {a.n:,} requests, {dup_actual:.1%} repeats, "
        f"{len({it.label for it in streams['templated']})} intent labels")

    models = {}
    log(f"model {DEFAULT_MODEL}")
    models[DEFAULT_MODEL] = run_model(DEFAULT_MODEL, rows, pair_texts, streams,
                                      [("on", G.ALL_ON), ("off", G.ALL_OFF)], log)
    if not a.no_compare:
        try:
            log(f"model {COMPARE_MODEL}")
            models[COMPARE_MODEL] = run_model(COMPARE_MODEL, rows, pair_texts, streams,
                                              [("on", G.ALL_ON), ("off", G.ALL_OFF)], log)
        except Exception as e:
            log(f"!! comparison model skipped: {type(e).__name__}: {e}")
    lat = latency_profile([it.text for it in streams["templated"]])
    prim = models[DEFAULT_MODEL]
    by_src = Counter(r["source"] for r in rows)
    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "python -m eval.sweep_threshold" + (" --quick" if a.quick else ""),
        "embedding_model": DEFAULT_MODEL,
        "similarity_unit": "cosine similarity of L2-normalised embeddings (1 = identical); not a distance",
        "taus": TAUS, "seed": a.seed, "budgets_false_hit_rate": BUDGETS,
        "metric_definitions": {"hit_rate": "hits / requests", "false_hit_rate": "wrong hits / requests (per request)",
                               "precision_per_hit": "correct hits / hits",
                               "correct_hit": "served entry has the same refined intent and the same specifics"},
        "datasets": {
            "pairs": {"file": str(CP.PAIRS_PATH.relative_to(ROOT)), "sha256": pairs_sha, "rows": len(rows),
                      "by_source": dict(by_src)},
            "stream": {"source": "bitext", "n": a.n, "dup_rate_target": a.dup_rate, "dup_rate_actual": round(dup_actual, 4),
                       "zipf_s": a.zipf, "intent_labels": len({it.label for it in streams["templated"]}),
                       "unique_texts_templated": len({it.text for it in streams["templated"]}),
                       "unique_texts_filled": len({it.text for it in streams["filled"]}),
                       "renderings": ["templated", "filled"]},
            "bitext_rows": int(len(df)), "licences": CP.LICENCES,
        },
        "recommended": prim["recommended"]["guards_on"],
        "recommended_without_guards": prim["recommended"]["guards_off"],
        "models": models,
        "latency_ms": lat,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    out = _round(out)
    path = __import__("pathlib").Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print_table(out)
    print(f"\nwrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} ({path.stat().st_size / 1e3:.0f} kB) "
          f"in {out['runtime_s']}s")
    return out


if __name__ == "__main__":
    main()
