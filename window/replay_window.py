"""Score context window management on demo_data/queries.jsonl.

Usage: .venv/bin/python -m window.replay_window

Attachments are shuffled (fixed seed) before selection: in the raw data the gold
doc is listed first in 147 of 150 rows, so position would otherwise give it away.

Per query and budget it checks:
  gold_docs_kept   every gold doc has at least one section kept
  gold_facts_kept  every gold fact string appears in the kept text (case + kept sections)
Facts missing from the full, unwindowed context are reported and excluded.

Writes results/window_bge-cosine__per_query_all_budgets.csv and
results/window_bge-cosine__budget_sweep.csv.
"""

import csv
import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from demo_data.build_dataset import load_corpus, render_case
from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, select

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"
MODEL = "BAAI/bge-small-en-v1.5"
BUDGETS = [100, 200, 300, 400, 500, 600, 800, 1000, 1200]
SEED = 7


def facts_in(facts, text):
    low = text.lower()
    return [f for f in facts if f.lower() in low]


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = [json.loads(line) for line in QUERIES.open()]
    docs = load_corpus()
    cases = json.loads(CASES.read_text())
    model = SentenceTransformer(MODEL)
    rng = random.Random(SEED)

    prepared = []
    for r in rows:
        attachments = r["attachments"][:]
        rng.shuffle(attachments)
        chunks = chunk_docs(docs, attachments)
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
        full_text = case_text + "\n" + "\n".join(c["text"] for c in chunks)
        prepared.append({
            "row": r,
            "chunks": chunks,
            "case_text": case_text,
            "full_tokens": approx_tokens(case_text) + sum(c["tokens"] for c in chunks),
            # Only facts that exist in the full context can be lost by windowing.
            "checkable_facts": facts_in(r["gold_facts"], full_text),
        })

    query_vecs = model.encode([p["row"]["query"] for p in prepared], normalize_embeddings=True)
    all_chunks = [c for p in prepared for c in p["chunks"]]
    chunk_vecs = model.encode([c["text"] for c in all_chunks], normalize_embeddings=True)
    pos = 0
    for p in prepared:
        p["chunk_vecs"] = chunk_vecs[pos:pos + len(p["chunks"])]
        pos += len(p["chunks"])

    per_query, sweep = [], []
    for budget in BUDGETS:
        stats = []
        for p, qv in zip(prepared, query_vecs):
            r = p["row"]
            windowed = p["full_tokens"] >= MIN_CONTEXT_TOKENS
            if windowed:
                kept, scores = select(p["chunks"], qv, p["chunk_vecs"], budget)
            else:
                kept, scores = p["chunks"], {}
            kept_text = p["case_text"] + "\n" + "\n".join(c["text"] for c in kept)
            kept_docs = {c["doc_id"] for c in kept}
            found = facts_in(p["checkable_facts"], kept_text)
            missing = [f for f in p["checkable_facts"] if f not in found]
            kept_tokens = approx_tokens(p["case_text"]) + sum(c["tokens"] for c in kept)
            # Rank of the best-scoring section from a gold doc (1 = top).
            ranked = sorted(scores, key=lambda k: -scores[k])
            gold_rank = next((i + 1 for i, k in enumerate(ranked) if k.split("#")[0] in r["gold_doc_ids"]), "")
            row = {
                "budget": budget,
                "id": r["id"],
                "cluster_id": r["cluster_id"],
                "difficulty": r["difficulty"],
                "windowed": int(windowed),
                "gold_docs": "|".join(r["gold_doc_ids"]),
                "gold_docs_kept": int(set(r["gold_doc_ids"]) <= kept_docs),
                "best_gold_section_rank": gold_rank,
                "facts_checkable": len(p["checkable_facts"]),
                "facts_kept": len(found),
                "all_facts_kept": int(not missing),
                "missing_facts": " | ".join(missing),
                "facts_not_in_full_context": " | ".join(f for f in r["gold_facts"] if f not in p["checkable_facts"]),
                "sections_kept": "|".join(c["id"] for c in kept) if windowed else "(all, not windowed)",
                "tokens_before": p["full_tokens"],
                "tokens_after": kept_tokens,
                "tokens_saved_pct": round(1 - kept_tokens / p["full_tokens"], 4),
                "query": r["query"],
            }
            per_query.append(row)
            stats.append(row)

        w = [s for s in stats if s["windowed"]]
        sweep.append({
            "budget": budget,
            "windowed_queries": len(w),
            "gold_docs_kept_pct": round(sum(s["gold_docs_kept"] for s in w) / len(w), 4),
            "all_facts_kept_pct": round(sum(s["all_facts_kept"] for s in w) / len(w), 4),
            "facts_kept_pct": round(sum(s["facts_kept"] for s in w) / max(1, sum(s["facts_checkable"] for s in w)), 4),
            "all_facts_kept_easy_pct": round(sum(s["all_facts_kept"] for s in w if s["difficulty"] == "easy") / max(1, sum(1 for s in w if s["difficulty"] == "easy")), 4),
            "all_facts_kept_complex_pct": round(sum(s["all_facts_kept"] for s in w if s["difficulty"] == "complex") / max(1, sum(1 for s in w if s["difficulty"] == "complex")), 4),
            "gold_section_rank1_pct": round(sum(1 for s in w if s["best_gold_section_rank"] == 1) / len(w), 4),
            "avg_tokens_before": round(sum(s["tokens_before"] for s in w) / len(w)),
            "avg_tokens_after": round(sum(s["tokens_after"] for s in w) / len(w)),
            "tokens_saved_pct_windowed": round(1 - sum(s["tokens_after"] for s in w) / sum(s["tokens_before"] for s in w), 4),
            "tokens_saved_pct_all_queries": round(1 - sum(s["tokens_after"] for s in stats) / sum(s["tokens_before"] for s in stats), 4),
        })

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / "window_bge-cosine__per_query_all_budgets.csv", per_query)
    write_csv(OUT / "window_bge-cosine__budget_sweep.csv", sweep)
    print(json.dumps(sweep, indent=1))


if __name__ == "__main__":
    main()
