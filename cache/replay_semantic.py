"""Replay demo_data/queries.jsonl through ExactCache, then SemanticCache, in each mode.

Usage: .venv/bin/python -m cache.replay_semantic [--model NAME] [--threshold T]

Modes (each adds a check to the previous one):
  cosine_only                      query cosine >= threshold
  cosine_context                   + cached answer's depends_on must be in the new context
  cosine_context_numbers_negation  + numbers and negation guards on the query text
  cosine_context_numbers_negation_judge
                                   + Haiku 4.5 judge on what's left (only with --judge;
                                     needs ANTHROPIC_API_KEY; no threshold sweep, to limit calls)

depends_on stands in for "what the answer used": hashes of the row's gold_doc_ids
(as attached) plus the full case text. In production this comes from the chunks
window management kept, or from the LLM's citations.

A served answer is correct when its answer_id matches the row's answer_id.
See results/README.md for what each output file holds.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from cache.exact import ExactCache, cache_key
from cache.guards import GUARDS
from cache.semantic import SemanticCache
from demo_data.build_dataset import load_corpus, render_attachment, render_case

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "demo_data" / "queries.jsonl"
CASES = ROOT / "demo_data" / "cases.json"
OUT = ROOT / "results"
MODELS = {"BAAI/bge-small-en-v1.5": "bge-small", "sentence-transformers/all-MiniLM-L6-v2": "minilm"}
THRESHOLDS = [round(0.70 + 0.01 * i, 2) for i in range(30)]
TRAPS = ["tricky_number", "tricky_negation", "tricky_intent", "tricky_case", "tricky_state", "edge"]
MODES = {
    "cosine_only": {"check_context": False, "guards": []},
    "cosine_context": {"check_context": True, "guards": []},
    "cosine_context_numbers_negation": {"check_context": True, "guards": GUARDS},
}
JUDGE_MODE = "cosine_context_numbers_negation_judge"
REASONS = ["low_cosine", "context_mismatch", "numbers_mismatch", "negation_mismatch", "judge_mismatch"]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_context(rows, docs, cases):
    """Per row: exact-cache key, hashes of everything attached, and depends_on hashes."""
    out = []
    for r in rows:
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else None
        doc_texts = {a["doc_id"]: render_attachment(docs, a) for a in r["attachments"]}
        doc_hash = {d: sha(t) for d, t in doc_texts.items()}
        case_hash = {"case:" + sha(case_text)} if case_text else set()
        missing = set(r["gold_doc_ids"]) - set(doc_hash)
        assert not missing, f"{r['id']}: gold docs not attached: {missing}"
        out.append({
            "key": cache_key(r["query"], case_text, list(doc_texts.values())),
            "context": frozenset(doc_hash.values()) | case_hash,
            "depends_on": frozenset(doc_hash[d] for d in r["gold_doc_ids"]) | case_hash,
        })
    return out


def embed(model_name, texts):
    model = SentenceTransformer(model_name)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def replay(rows, ctx, vecs, threshold, mode):
    exact = ExactCache()
    sem = SemanticCache(threshold, **MODES[mode])
    out = []
    for r, c, vec in zip(rows, ctx, vecs):
        score, reason, entry = None, "", exact.get(c["key"])
        if entry is not None:
            layer = "exact"
        else:
            entry, score, reason = sem.lookup(vec, r["query"], c["context"])
            layer = "semantic" if entry is not None else "llm"
        if entry is None:
            # Stand-in for an LLM call: store the reference answer.
            value = {"source_id": r["id"], "answer_id": r["answer_id"]}
            exact.put(c["key"], value)
            sem.put(vec, value, r["query"], c["depends_on"])

        expected_hit = r["expected_cache"] in ("exact_hit", "semantic_hit")
        if entry is not None:
            outcome = "correct_hit" if entry["answer_id"] == r["answer_id"] else "wrong_hit"
        else:
            outcome = "missed_hit" if expected_hit else "true_miss"

        out.append({
            "id": r["id"],
            "cluster_id": r["cluster_id"],
            "expected_cache": r["expected_cache"],
            "variant": r["variant"],
            "layer": layer,
            "outcome": outcome,
            "miss_reason": "" if layer != "llm" else reason,
            "cosine": "" if score is None else round(score, 4),
            "served_from": entry["source_id"] if entry else "",
            "expected_source": r["exact_of"] or r["semantic_of"] or "",
            "confusable_with": r["confusable_with"] or "",
            "tokens_saved": r["approx_context_tokens"] if outcome == "correct_hit" else 0,
            "context_tokens": r["approx_context_tokens"],
            "query": r["query"],
        })
    return out


def summarize(out, model_name, threshold, mode):
    def n(**kw):
        return sum(1 for x in out if all(x[k] == v for k, v in kw.items()))

    wrong = [x for x in out if x["outcome"] == "wrong_hit"]
    traps = [x for x in out if x["expected_cache"] == "miss"]
    paraphrases = [x for x in out if x["expected_cache"] == "semantic_hit"]
    saved = sum(x["tokens_saved"] for x in out)
    total = sum(x["context_tokens"] for x in out)
    return {
        "mode": mode,
        "model": model_name,
        "threshold": threshold,
        "queries": len(out),
        "exact_hits": n(layer="exact"),
        "semantic_hits": n(layer="semantic"),
        "paraphrases_caught": n(expected_cache="semantic_hit", outcome="correct_hit"),
        "paraphrases_total": len(paraphrases),
        "paraphrase_recall": round(n(expected_cache="semantic_hit", outcome="correct_hit") / len(paraphrases), 4),
        "wrong_hits": len(wrong),
        "wrong_hits_on_traps": sum(1 for x in wrong if x["expected_cache"] == "miss"),
        "traps_total": len(traps),
        "wrong_hits_on_cold": sum(1 for x in wrong if x["expected_cache"] == "cold"),
        **{f"wrong_{v}": sum(1 for x in wrong if x["variant"] == v) for v in TRAPS},
        **{f"traps_stopped_by_{r}": sum(1 for x in traps if x["miss_reason"] == r) for r in REASONS},
        **{f"paraphrases_blocked_by_{r}": sum(1 for x in paraphrases if x["miss_reason"] == r) for r in REASONS},
        "missed_hits": n(outcome="missed_hit"),
        "tokens_saved_correct_hits_only": saved,
        "tokens_saved_pct": round(saved / total, 4),
    }


def pairs(rows, vecs, model_name):
    idx = {r["id"]: i for i, r in enumerate(rows)}
    out = []
    for i, r in enumerate(rows):
        other = r["semantic_of"] if r["expected_cache"] == "semantic_hit" else r["confusable_with"] if r["expected_cache"] == "miss" else None
        if not other:
            continue
        out.append({
            "model": model_name,
            "id": r["id"],
            "kind": "paraphrase" if r["expected_cache"] == "semantic_hit" else "trap",
            "variant": r["variant"],
            "other_id": other,
            "cosine": round(float(vecs[i] @ vecs[idx[other]]), 4),
            "query": r["query"],
            "other_query": rows[idx[other]]["query"],
        })
    return out


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--judge", action="store_true", help="also run the Haiku 4.5 judge mode")
    args = ap.parse_args()
    judge = None
    if args.judge:
        from cache.judge import Judge
        judge = Judge(OUT / "judge_verdicts_haiku-4-5.json")
        MODES[JUDGE_MODE] = {"check_context": True, "guards": [*GUARDS, ("judge", judge.same)]}
    tag = f"{MODELS.get(args.model, args.model.split('/')[-1])}_t{args.threshold:.2f}"

    rows = [json.loads(line) for line in QUERIES.open()]
    ctx = build_context(rows, load_corpus(), json.loads(CASES.read_text()))
    queries = [r["query"] for r in rows]
    vecs = {m: embed(m, queries) for m in dict.fromkeys([args.model, *MODELS])}

    OUT.mkdir(exist_ok=True)
    comparison = []
    for mode in MODES:
        per_query = [{"mode": mode, "model": args.model, "threshold": args.threshold, **x}
                     for x in replay(rows, ctx, vecs[args.model], args.threshold, mode)]
        summary = summarize(per_query, args.model, args.threshold, mode)
        if mode == JUDGE_MODE:
            summary.update({
                "judge_model": "claude-haiku-4-5",
                "judge_verdicts_used": len(judge.used),
                "judge_api_calls_this_run": judge.calls,
                "judge_input_tokens": sum(v["input_tokens"] for v in judge.used),
                "judge_output_tokens": sum(v["output_tokens"] for v in judge.used),
            })
        else:
            sweep = [summarize(replay(rows, ctx, v, t, mode), m, t, mode) for m, v in vecs.items() for t in THRESHOLDS]
            write_csv(OUT / f"semantic_{mode}__threshold_sweep_all_models.csv", sweep)
        comparison.append(summary)
        write_csv(OUT / f"semantic_{mode}__{tag}__per_query.csv", per_query)
        (OUT / f"semantic_{mode}__{tag}__summary.json").write_text(json.dumps(summary, indent=2))

    keys = list(dict.fromkeys(k for s in comparison for k in s))
    write_csv(OUT / f"semantic_mode_comparison__{tag}.csv", [{k: s.get(k, "") for k in keys} for s in comparison])
    write_csv(OUT / "embedding_pair_cosine_all_models.csv",
              [p for m, v in vecs.items() for p in pairs(rows, v, m)])
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
