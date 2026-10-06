"""Replay demo_data/queries.jsonl through ExactCache and score it.

Usage: python3 -m cache.replay_exact  (from the repo root)
Writes results/exact__per_query.csv and results/exact__summary.json.
"""

import csv
import json
import time
from pathlib import Path

from cache.exact import ExactCache, cache_key
from demo_data.build_dataset import load_corpus, render_attachment, render_case

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"


def main():
    rows = [json.loads(line) for line in QUERIES.open()]
    docs = load_corpus()
    cases = json.loads(CASES.read_text())
    cache = ExactCache()
    answered_by = {}  # row id -> id of the row whose answer was served
    out_rows = []

    for r in rows:
        # Rebuild the text production would send; ids are only used to look it up.
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else None
        doc_texts = [render_attachment(docs, a) for a in r["attachments"]]
        key = cache_key(r["query"], case_text, doc_texts)
        t0 = time.perf_counter()
        entry = cache.get(key)
        lookup_us = (time.perf_counter() - t0) * 1e6

        hit = entry is not None
        if hit:
            answered_by[r["id"]] = entry["source_id"]
        else:
            # Stand-in for an LLM call: store the reference answer.
            cache.put(key, {"source_id": r["id"], "answer": r["reference_answer"]})
            answered_by[r["id"]] = r["id"]

        expected_hit = r["expected_cache"] == "exact_hit"
        if hit and expected_hit:
            outcome = "true_hit" if entry["source_id"] == r["exact_of"] else "wrong_source"
        elif hit:
            outcome = "false_hit"
        elif expected_hit:
            outcome = "missed_hit"
        else:
            outcome = "true_miss"

        out_rows.append({
            "id": r["id"],
            "cluster_id": r["cluster_id"],
            "expected_cache": r["expected_cache"],
            "variant": r["variant"],
            "hit": int(hit),
            "outcome": outcome,
            "served_from": entry["source_id"] if hit else "",
            "exact_of": r["exact_of"] or "",
            "context_tokens": r["approx_context_tokens"],
            "tokens_saved": r["approx_context_tokens"] if hit else 0,
            "lookup_us": round(lookup_us, 2),
            "query": r["query"],
        })

    def count(o):
        return sum(1 for x in out_rows if x["outcome"] == o)

    exp_hits = sum(1 for r in rows if r["expected_cache"] == "exact_hit")
    total_tokens = sum(r["approx_context_tokens"] for r in rows)
    saved = sum(x["tokens_saved"] for x in out_rows)
    summary = {
        "queries": len(rows),
        "expected_exact_hits": exp_hits,
        "hits": cache.hits,
        "misses": cache.misses,
        "true_hits": count("true_hit"),
        "false_hits": count("false_hit"),
        "wrong_source": count("wrong_source"),
        "missed_hits": count("missed_hit"),
        "true_misses": count("true_miss"),
        "recall": round(count("true_hit") / exp_hits, 4) if exp_hits else None,
        "precision": round(count("true_hit") / cache.hits, 4) if cache.hits else None,
        "hit_rate": round(cache.hits / len(rows), 4),
        "cache_entries": len(cache),
        "context_tokens_total": total_tokens,
        "context_tokens_saved": saved,
        "tokens_saved_pct": round(saved / total_tokens, 4),
        "llm_calls_avoided": cache.hits,
        "passed": count("false_hit") == 0 and count("missed_hit") == 0 and count("wrong_source") == 0,
    }

    OUT.mkdir(exist_ok=True)
    with (OUT / "exact__per_query.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    (OUT / "exact__summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
