"""Score LLMLingua-2 compression on top of hybrid window management (budgets 800 and 1200).

Usage: .venv/bin/python -m prompt_compression.eval_lingua

Same protocol as prompt_compression/eval_split.py (shuffled attachments, 5 seeds,
cluster halves); every config is reported on the test clusters. LLMLingua-2 rewords,
so two fact checks are reported:
  exact  the gold fact string appears verbatim
  loose  the fact's content words appear in order (see lingua.fact_kept_loose)
Loose can hide real damage; an LLM answer check is the final word.

Writes results/compression_lingua2__all_configs_test.csv and
results/compression_lingua2__test_per_query.csv.
"""

import csv
import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from demo_data.build_dataset import load_corpus, render_case
from prompt_compression.lingua import Lingua, fact_kept_loose
from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences, select_hybrid

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [(w, r) for w in [800, 1200] for r in [1.0, 0.8, 0.7, 0.6, 0.5]]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = [json.loads(line) for line in QUERIES.open()]
    docs = load_corpus()
    cases = json.loads(CASES.read_text())
    boilerplate = repeated_sentences(docs)
    model = SentenceTransformer(EMBED_MODEL)
    lingua = Lingua()

    all_chunks = {c["id"]: c for d in docs for c in chunk_docs(docs, [{"doc_id": d}])}
    ids = list(all_chunks)
    chunk_vec = dict(zip(ids, model.encode([all_chunks[i]["text"] for i in ids], normalize_embeddings=True)))

    large = []
    for r in rows:
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
        chunks = chunk_docs(docs, r["attachments"])
        full_tokens = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
        if full_tokens < MIN_CONTEXT_TOKENS:
            continue
        full = (case_text + "\n" + "\n".join(c["text"] for c in chunks)).lower()
        large.append({"row": r, "case_text": case_text, "full_tokens": full_tokens,
                      "facts": [f for f in r["gold_facts"] if f.lower() in full]})
    for p, qv in zip(large, model.encode([p["row"]["query"] for p in large], normalize_embeddings=True)):
        p["qvec"] = qv

    clusters = sorted({p["row"]["cluster_id"] for p in large})
    summary, per_query = [], []
    for seed in SEEDS:
        rng = random.Random(seed)
        shuffled = {p["row"]["id"]: rng.sample(p["row"]["attachments"], len(p["row"]["attachments"])) for p in large}
        cl = clusters[:]
        rng.shuffle(cl)
        test = [p for p in large if p["row"]["cluster_id"] not in set(cl[: len(cl) // 2])]
        for window, rate in CONFIGS:
            res = []
            for p in test:
                chunks = chunk_docs(docs, shuffled[p["row"]["id"]])
                win, _ = select_hybrid(chunks, p["qvec"], [chunk_vec[c["id"]] for c in chunks], boilerplate, budget=window)
                kept = lingua.compress(win, rate) if rate < 1 else win
                text = p["case_text"] + "\n" + "\n".join(c["text"] for c in kept)
                exact = [f for f in p["facts"] if f.lower() in text.lower()]
                loose = [f for f in p["facts"] if fact_kept_loose(f, text)]
                case_tokens = approx_tokens(p["case_text"])
                x = {
                    "all_facts_exact": int(len(exact) == len(p["facts"])),
                    "all_facts_loose": int(len(loose) == len(p["facts"])),
                    "tokens_original": p["full_tokens"],
                    "tokens_after_window": case_tokens + sum(c["tokens"] for c in win),
                    "tokens_after_compression": case_tokens + sum(c["tokens"] for c in kept),
                    "missing_loose": " | ".join(f for f in p["facts"] if f not in loose),
                }
                res.append(x)
                if seed == SEEDS[0]:
                    per_query.append({"seed": seed, "window_budget": window, "rate": rate, "id": p["row"]["id"],
                                      **x, "query": p["row"]["query"],
                                      "compressed_docs": " || ".join(c["text"].replace("\n", ": ") for c in kept)})
            n = len(res)
            orig = sum(x["tokens_original"] for x in res)
            summary.append({
                "seed": seed, "window_budget": window, "lingua_rate": rate, "test_queries": n,
                "all_facts_exact_pct": round(sum(x["all_facts_exact"] for x in res) / n, 4),
                "all_facts_loose_pct": round(sum(x["all_facts_loose"] for x in res) / n, 4),
                "avg_tokens_after_window": round(sum(x["tokens_after_window"] for x in res) / n),
                "avg_tokens_after_compression": round(sum(x["tokens_after_compression"] for x in res) / n),
                "total_saved_pct_of_original": round(1 - sum(x["tokens_after_compression"] for x in res) / orig, 4),
            })

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "compression_lingua2__all_configs_test.csv", summary)
    write_csv(OUT / "compression_lingua2__test_per_query.csv", per_query)
    print(json.dumps(summary[:10], indent=1))


if __name__ == "__main__":
    main()
