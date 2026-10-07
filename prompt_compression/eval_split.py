"""Score sentence-pruning compression on top of hybrid window management (budgets 800 and 1200).

Usage: .venv/bin/python -m prompt_compression.eval_split

Same protocol as window/eval_split.py: shuffled attachments, 5 seeds, clusters split
in half. Each config is reported on the test clusters. The config chosen on train is
the cheapest one whose train coverage is within MARGIN of no compression, so tuning
can't pick a setting that only works by luck on the train half.

Writes results/compression_split_eval__*.csv (see results/README.md).
"""

import csv
import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from prompt_compression.compressor import clean, compress
from demo_data.build_dataset import load_corpus, render_case
from window.manager import (MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences,
                            select_hybrid, sentences)

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"
MODEL = "BAAI/bge-small-en-v1.5"
SEEDS = [0, 1, 2, 3, 4]
MARGIN = 0.02
WINDOW_BUDGETS = [800, 1200]
CONFIGS = [
    {"window_budget": w, "keep_ratio": k, "neighbors": n}
    for w in WINDOW_BUDGETS
    for k, n in [(1.0, 0)] + [(k, n) for k in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9] for n in [0, 1]]
]


def facts_in(facts, text):
    low = text.lower()
    return [f for f in facts if f.lower() in low]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def name(cfg):
    comp = "no_compression" if cfg["keep_ratio"] == 1.0 else f"keep={cfg['keep_ratio']},neighbors={cfg['neighbors']}"
    return f"window={cfg['window_budget']},{comp}"


def main():
    rows = [json.loads(line) for line in QUERIES.open()]
    docs = load_corpus()
    cases = json.loads(CASES.read_text())
    boilerplate = repeated_sentences(docs)
    model = SentenceTransformer(MODEL)

    all_chunks = {c["id"]: c for d in docs for c in chunk_docs(docs, [{"doc_id": d}])}
    ids = list(all_chunks)
    chunk_vec = dict(zip(ids, model.encode([all_chunks[i]["text"] for i in ids], normalize_embeddings=True)))
    all_sentences = sorted({clean(s) for c in all_chunks.values() for s in sentences(c["text"].split("\n", 1)[1])})
    sent_vec = dict(zip(all_sentences, model.encode(all_sentences, normalize_embeddings=True)))

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
    for p, qv in zip(large, model.encode([p["row"]["query"] for p in large], normalize_embeddings=True)):
        p["qvec"] = qv

    def run(p, cfg, attachments):
        chunks = chunk_docs(docs, attachments)
        windowed, _ = select_hybrid(chunks, p["qvec"], [chunk_vec[c["id"]] for c in chunks], boilerplate,
                                    budget=cfg["window_budget"])
        kept = (compress(windowed, p["qvec"], lambda s: sent_vec[s], keep_ratio=cfg["keep_ratio"], neighbors=cfg["neighbors"])
                if cfg["keep_ratio"] < 1 else windowed)
        text = p["case_text"] + "\n" + "\n".join(c["text"] for c in kept)
        found = facts_in(p["facts"], text)
        case_tokens = approx_tokens(p["case_text"])
        return {
            "all_facts_kept": int(len(found) == len(p["facts"])),
            "tokens_original": p["full_tokens"],
            "tokens_after_window": case_tokens + sum(c["tokens"] for c in windowed),
            "tokens_after_compression": case_tokens + sum(c["tokens"] for c in kept),
            "missing_facts": " | ".join(f for f in p["facts"] if f not in found),
        }

    def score(res):
        n = len(res)
        orig = sum(x["tokens_original"] for x in res)
        win = sum(x["tokens_after_window"] for x in res)
        comp = sum(x["tokens_after_compression"] for x in res)
        return {
            "queries": n,
            "all_facts_kept_pct": round(sum(x["all_facts_kept"] for x in res) / n, 4),
            "avg_tokens_after_window": round(win / n),
            "avg_tokens_after_compression": round(comp / n),
            "compression_saved_pct_of_windowed": round(1 - comp / win, 4),
            "total_saved_pct_of_original": round(1 - comp / orig, 4),
        }

    clusters = sorted({p["row"]["cluster_id"] for p in large})
    all_configs, chosen, test_rows = [], [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        shuffled = {p["row"]["id"]: rng.sample(p["row"]["attachments"], len(p["row"]["attachments"])) for p in large}
        cl = clusters[:]
        rng.shuffle(cl)
        train_clusters = set(cl[: len(cl) // 2])
        splits = {"train": [p for p in large if p["row"]["cluster_id"] in train_clusters],
                  "test": [p for p in large if p["row"]["cluster_id"] not in train_clusters]}
        results = {}
        for cfg in CONFIGS:
            for split, ps in splits.items():
                res = [run(p, cfg, shuffled[p["row"]["id"]]) for p in ps]
                results[(name(cfg), split)] = res
                all_configs.append({"seed": seed, "split": split, "config": name(cfg), **cfg, **score(res)})

        for w in WINDOW_BUDGETS:
            none = name({"window_budget": w, "keep_ratio": 1.0, "neighbors": 0})
            base = score(results[(none, "train")])["all_facts_kept_pct"]
            cands = [name(c) for c in CONFIGS if c["window_budget"] == w]
            ok = [n for n in cands if score(results[(n, "train")])["all_facts_kept_pct"] >= base - MARGIN]
            best = min(ok, key=lambda n: score(results[(n, "train")])["avg_tokens_after_compression"])
            te = score(results[(best, "test")])
            chosen.append({"seed": seed, "window_budget": w, "chosen_config": best, "train_coverage_no_compression": base,
                           "train_coverage_chosen": score(results[(best, "train")])["all_facts_kept_pct"],
                           "test_coverage_no_compression": score(results[(none, "test")])["all_facts_kept_pct"],
                           **{f"test_{k}": v for k, v in te.items()}})
            for p, x in zip(splits["test"], results[(best, "test")]):
                test_rows.append({"seed": seed, "config": best, "id": p["row"]["id"], "cluster_id": p["row"]["cluster_id"],
                                  **x, "query": p["row"]["query"]})

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "compression_split_eval__all_configs_train_and_test.csv", all_configs)
    write_csv(OUT / "compression_split_eval__chosen_config_per_seed.csv", chosen)
    write_csv(OUT / "compression_split_eval__test_per_query.csv", test_rows)
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
