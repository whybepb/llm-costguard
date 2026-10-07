"""Compare section-first and doc-first window management on cluster train/test splits.

Usage: .venv/bin/python -m window.eval_split

For each seed: shuffle attachments, split the clusters in half, pick each approach's
best config on the train clusters (most queries with all gold facts kept, then fewest
tokens), and report it on the test clusters, which the tuning never saw.

Only queries with large context (>= MIN_CONTEXT_TOKENS) are scored.
Writes results/window_split_eval__*.csv (see results/README.md).
"""

import csv
import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from demo_data.build_dataset import load_corpus, render_case
from window.manager import (MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences,
                            select, select_docs, select_hybrid)

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"
MODEL = "BAAI/bge-small-en-v1.5"
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = (
    [("section_first", {"budget": b}) for b in [400, 600, 800, 1000, 1200]]
    + [("doc_first", {"method": m, "gap": g}) for m in ["max", "mean"] for g in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15]]
    + [("hybrid", {"method": m, "budget": b}) for m in ["max", "mean"] for b in [600, 700, 800, 1000, 1200]]
)
APPROACHES = ["section_first", "doc_first", "hybrid"]


def facts_in(facts, text):
    low = text.lower()
    return [f for f in facts if f.lower() in low]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def config_name(approach, cfg):
    return approach + "(" + ",".join(f"{k}={v}" for k, v in cfg.items()) + ")"


def main():
    rows = [json.loads(line) for line in QUERIES.open()]
    docs = load_corpus()
    cases = json.loads(CASES.read_text())
    boilerplate = repeated_sentences(docs)
    model = SentenceTransformer(MODEL)

    # Chunk text doesn't depend on attachment order, so embed each section once.
    all_chunks = {c["id"]: c for d in docs for c in chunk_docs(docs, [{"doc_id": d}])}
    ids = list(all_chunks)
    vec_of = dict(zip(ids, model.encode([all_chunks[i]["text"] for i in ids], normalize_embeddings=True)))

    large = []
    for r in rows:
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
        chunks = chunk_docs(docs, r["attachments"])
        full_tokens = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
        if full_tokens < MIN_CONTEXT_TOKENS:
            continue
        full_text = case_text + "\n" + "\n".join(c["text"] for c in chunks)
        large.append({"row": r, "case_text": case_text, "full_tokens": full_tokens,
                      "facts": facts_in(r["gold_facts"], full_text)})
    qvecs = model.encode([p["row"]["query"] for p in large], normalize_embeddings=True)
    for p, qv in zip(large, qvecs):
        p["qvec"] = qv

    def run(p, approach, cfg, rng_attachments):
        chunks = chunk_docs(docs, rng_attachments)
        cvecs = [vec_of[c["id"]] for c in chunks]
        if approach == "section_first":
            kept, _ = select(chunks, p["qvec"], cvecs, cfg["budget"])
        elif approach == "doc_first":
            kept, _ = select_docs(chunks, p["qvec"], cvecs, boilerplate, **cfg)
        else:
            kept, _ = select_hybrid(chunks, p["qvec"], cvecs, boilerplate, **cfg)
        text = p["case_text"] + "\n" + "\n".join(c["text"] for c in kept)
        found = facts_in(p["facts"], text)
        r = p["row"]
        return {
            "all_facts_kept": int(len(found) == len(p["facts"])),
            "gold_docs_kept": int(set(r["gold_doc_ids"]) <= {c["doc_id"] for c in kept}),
            "docs_kept": len({c["doc_id"] for c in kept}),
            "tokens_after": approx_tokens(p["case_text"]) + sum(c["tokens"] for c in kept),
            "tokens_before": p["full_tokens"],
            "missing_facts": " | ".join(f for f in p["facts"] if f not in found),
            "sections_kept": "|".join(c["id"] for c in kept),
        }

    def score(results):
        n = len(results)
        before = sum(x["tokens_before"] for x in results)
        after = sum(x["tokens_after"] for x in results)
        return {
            "queries": n,
            "all_facts_kept_pct": round(sum(x["all_facts_kept"] for x in results) / n, 4),
            "gold_docs_kept_pct": round(sum(x["gold_docs_kept"] for x in results) / n, 4),
            "avg_docs_kept": round(sum(x["docs_kept"] for x in results) / n, 2),
            "avg_tokens_after": round(after / n),
            "tokens_saved_pct": round(1 - after / before, 4),
        }

    clusters = sorted({p["row"]["cluster_id"] for p in large})
    all_configs, per_seed, test_rows = [], [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        shuffled = {p["row"]["id"]: rng.sample(p["row"]["attachments"], len(p["row"]["attachments"])) for p in large}
        cl = clusters[:]
        rng.shuffle(cl)
        train_clusters = set(cl[: len(cl) // 2])
        splits = {
            "train": [p for p in large if p["row"]["cluster_id"] in train_clusters],
            "test": [p for p in large if p["row"]["cluster_id"] not in train_clusters],
        }
        results = {}
        for approach, cfg in CONFIGS:
            name = config_name(approach, cfg)
            for split, ps in splits.items():
                res = [run(p, approach, cfg, shuffled[p["row"]["id"]]) for p in ps]
                results[(name, split)] = res
                all_configs.append({"seed": seed, "split": split, "approach": approach, "config": name, **score(res)})

        for approach in APPROACHES:
            names = [config_name(a, c) for a, c in CONFIGS if a == approach]
            # Best on train: most queries fully covered, then fewest tokens.
            best = max(names, key=lambda n: (score(results[(n, "train")])["all_facts_kept_pct"],
                                             -score(results[(n, "train")])["avg_tokens_after"]))
            tr, te = score(results[(best, "train")]), score(results[(best, "test")])
            per_seed.append({
                "seed": seed, "approach": approach, "chosen_config": best,
                "train_clusters": "|".join(sorted(train_clusters)),
                "train_all_facts_kept_pct": tr["all_facts_kept_pct"],
                "test_queries": te["queries"],
                "test_all_facts_kept_pct": te["all_facts_kept_pct"],
                "test_gold_docs_kept_pct": te["gold_docs_kept_pct"],
                "test_avg_docs_kept": te["avg_docs_kept"],
                "test_avg_tokens_after": te["avg_tokens_after"],
                "test_tokens_saved_pct": te["tokens_saved_pct"],
            })
            for p, x in zip(splits["test"], results[(best, "test")]):
                test_rows.append({"seed": seed, "approach": approach, "config": best, "id": p["row"]["id"],
                                  "cluster_id": p["row"]["cluster_id"], "difficulty": p["row"]["difficulty"],
                                  "gold_docs": "|".join(p["row"]["gold_doc_ids"]), **x, "query": p["row"]["query"]})

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "window_split_eval__all_configs_train_and_test.csv", all_configs)
    write_csv(OUT / "window_split_eval__chosen_config_per_seed.csv", per_seed)
    write_csv(OUT / "window_split_eval__test_per_query.csv", test_rows)
    print(json.dumps(per_seed, indent=1))


if __name__ == "__main__":
    main()
