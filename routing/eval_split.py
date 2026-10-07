"""Score the router on cluster train/test splits (5 seeds).

Usage: .venv/bin/python -m routing.eval_split

Label: needs large = not small_model_ok. Context features come from hybrid-800
windowing of shuffled attachments (small contexts pass through unwindowed).

Threshold: on the train clusters, out-of-fold P(large) from group k-fold by cluster;
pick the highest threshold whose complex->small rate is <= TARGET_COMPLEX_TO_SMALL.
Then fit on all train clusters and report on the test clusters.

Cost per query = input (system + query + context after windowing) and output
(reference answer length as a stand-in) at each model's price. Compared against
always-large, always-small, a simple rules router, and the labels themselves (oracle).

Writes results/routing__*.csv (see results/README.md).
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold

from demo_data.build_dataset import load_corpus, render_case
from routing.router import LARGE, SMALL, Router, cost, text_features
from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences, select_hybrid

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SEEDS = [0, 1, 2, 3, 4]
SYSTEM_TOKENS = 200  # assumed fixed system prompt size
TARGET_COMPLEX_TO_SMALL = 0.0
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def rules_route(query, docs_kept):
    """Generic baseline: large for multi-part or assumption-laden questions."""
    q = query.lower()
    multi = q.count("?") > 1 or " instead" in q or " or " in q or q.rstrip().endswith("right?")
    return LARGE if multi or docs_kept > 1 else SMALL


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate(routes, ps):
    n = len(ps)
    complex_ = [p for p in ps if p["needs_large"]]
    easy = [p for p in ps if not p["needs_large"]]
    c2s = sum(1 for p in complex_ if routes[p["row"]["id"]] == SMALL)
    e2l = sum(1 for p in easy if routes[p["row"]["id"]] == LARGE)
    total = sum(cost(routes[p["row"]["id"]], p["input_tokens"], p["output_tokens"]) for p in ps)
    return {
        "queries": n,
        "complex_to_small": c2s,
        "complex_to_small_pct": round(c2s / max(1, len(complex_)), 4),
        "easy_to_large": e2l,
        "easy_to_large_pct": round(e2l / max(1, len(easy)), 4),
        "share_routed_small": round(sum(1 for p in ps if routes[p["row"]["id"]] == SMALL) / n, 4),
        "cost_per_1k_queries_usd": round(1000 * total / n, 4),
    }


def main():
    rows = [json.loads(line) for line in QUERIES.open()]
    docs = load_corpus()
    cases = json.loads(CASES.read_text())
    boilerplate = repeated_sentences(docs)
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = {c["id"]: c for d in docs for c in chunk_docs(docs, [{"doc_id": d}])}
    ids = list(all_chunks)
    chunk_vec = dict(zip(ids, model.encode([all_chunks[i]["text"] for i in ids], normalize_embeddings=True)))
    qvecs = model.encode([r["query"] for r in rows], normalize_embeddings=True)

    clusters = sorted({r["cluster_id"] for r in rows})
    chosen, fixed, per_query = [], [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        prepared = []
        for r, qv in zip(rows, qvecs):
            case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
            chunks = chunk_docs(docs, rng.sample(r["attachments"], len(r["attachments"])))
            full = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
            if full >= MIN_CONTEXT_TOKENS:
                chunks, _ = select_hybrid(chunks, qv, [chunk_vec[c["id"]] for c in chunks], boilerplate)
            ctx = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
            docs_kept = len({c["doc_id"] for c in chunks})
            prepared.append({
                "row": r, "qvec": qv, "needs_large": int(not r["small_model_ok"]),
                "feats": text_features(r["query"], docs_kept, ctx), "docs_kept": docs_kept,
                "input_tokens": SYSTEM_TOKENS + approx_tokens(r["query"]) + ctx,
                "output_tokens": approx_tokens(r["reference_answer"]),
            })
        cl = clusters[:]
        rng.shuffle(cl)
        train_cl = set(cl[: len(cl) // 2])
        train = [p for p in prepared if p["row"]["cluster_id"] in train_cl]
        test = [p for p in prepared if p["row"]["cluster_id"] not in train_cl]

        def arrays(ps):
            return (np.stack([p["qvec"] for p in ps]), np.array([p["feats"] for p in ps], dtype=float),
                    np.array([p["needs_large"] for p in ps]))

        Xe, Xf, y = arrays(train)
        groups = [p["row"]["cluster_id"] for p in train]
        oof = np.zeros(len(train))
        for tr, va in GroupKFold(n_splits=min(5, len(set(groups)))).split(Xe, y, groups):
            oof[va] = Router().fit(Xe[tr], Xf[tr], y[tr]).p_large(Xe[va], Xf[va])
        ok = [t for t in THRESHOLDS
              if sum(1 for p, s in zip(train, oof) if p["needs_large"] and s < t) / max(1, y.sum()) <= TARGET_COMPLEX_TO_SMALL]
        threshold = max(ok) if ok else min(THRESHOLDS)

        router = Router(threshold).fit(Xe, Xf, y)
        Te, Tf, _ = arrays(test)
        p_large = router.p_large(Te, Tf)
        strategies = {
            "router": {p["row"]["id"]: (LARGE if s >= threshold else SMALL) for p, s in zip(test, p_large)},
            "always_large": {p["row"]["id"]: LARGE for p in test},
            "always_small": {p["row"]["id"]: SMALL for p in test},
            "rules": {p["row"]["id"]: rules_route(p["row"]["query"], p["docs_kept"]) for p in test},
            "oracle_labels": {p["row"]["id"]: (LARGE if p["needs_large"] else SMALL) for p in test},
        }
        for name, routes in strategies.items():
            chosen.append({"seed": seed, "strategy": name, "threshold": threshold if name == "router" else "",
                           **evaluate(routes, test)})
        for t in THRESHOLDS:
            fixed.append({"seed": seed, "threshold": t,
                          **evaluate({p["row"]["id"]: (LARGE if s >= t else SMALL) for p, s in zip(test, p_large)}, test)})
        for p, s in zip(test, p_large):
            per_query.append({"seed": seed, "id": p["row"]["id"], "cluster_id": p["row"]["cluster_id"],
                              "difficulty": p["row"]["difficulty"], "label": LARGE if p["needs_large"] else SMALL,
                              "p_large": round(float(s), 4), "threshold": threshold,
                              "routed": strategies["router"][p["row"]["id"]],
                              "correct": int(strategies["router"][p["row"]["id"]] == (LARGE if p["needs_large"] else SMALL)),
                              "input_tokens": p["input_tokens"], "query": p["row"]["query"]})

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "routing__strategies_test_per_seed.csv", chosen)
    write_csv(OUT / "routing__router_fixed_thresholds_test.csv", fixed)
    write_csv(OUT / "routing__router_test_per_query.csv", per_query)
    print(json.dumps(chosen[:5], indent=1))


if __name__ == "__main__":
    main()
