"""Deterministic quality gates: no API calls, a few seconds on CPU.

Usage: .venv/bin/python -m gates.deterministic   (also run by tests/test_gates.py in CI)

  exact     replay demo_data through the exact cache: hits and false hits
  semantic  replay demo_data through exact -> semantic cache with context check, numbers
            and negation guards and the memoized Haiku judge verdicts (offline: a new
            query pair fails the gate instead of calling the API)
  window    hybrid window at the configured budget on every large-context demo query,
            attachments shuffled (seed 0) so the gold doc isn't first: share of queries
            that keep all gold facts

Bounds come from [gates.deterministic] in config/costguard.toml.
"""

import json
import os
import random
from functools import lru_cache
from pathlib import Path

from config import CFG
from demo_data.build_dataset import load_corpus, render_case

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo_data"
JUDGE_MEMO = ROOT / "results" / "judge_verdicts_haiku-4-5.json"


@lru_cache(maxsize=1)
def _data():
    rows = [json.loads(line) for line in (DEMO / "queries.jsonl").open()]
    return rows, load_corpus(), json.loads((DEMO / "cases.json").read_text())


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(CFG["models"]["embedding"])


def exact_metrics():
    from cache.replay_exact import ExactCache, cache_key
    from demo_data.build_dataset import render_attachment
    rows, docs, cases = _data()
    cache, hits, false_hits = ExactCache(), 0, 0
    for r in rows:
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else None
        key = cache_key(r["query"], case_text, [render_attachment(docs, a) for a in r["attachments"]])
        entry = cache.get(key)
        if entry is None:
            cache.put(key, {"answer_id": r["answer_id"]})
        elif entry["answer_id"] == r["answer_id"] and r["expected_cache"] == "exact_hit":
            hits += 1
        else:
            false_hits += 1
    return {"exact_hits": hits, "exact_false_hits": false_hits}


def semantic_metrics():
    os.environ["COSTGUARD_OFFLINE"] = "1"  # judge verdicts must come from the memo
    from cache.guards import GUARDS
    from cache.judge import Judge
    from cache.replay_semantic import MODES, build_context, replay
    rows, docs, cases = _data()
    judge = Judge(JUDGE_MEMO)
    MODES["gate"] = {"check_context": True, "guards": [*GUARDS, ("judge", judge.same)]}
    vecs = _model().encode([r["query"] for r in rows], normalize_embeddings=True)
    out = replay(rows, build_context(rows, docs, cases), vecs, CFG["cache"]["semantic_threshold"], "gate")
    return {
        "semantic_wrong_hits": sum(1 for x in out if x["outcome"] == "wrong_hit"),
        "semantic_paraphrase_hits": sum(1 for x in out if x["expected_cache"] == "semantic_hit"
                                        and x["outcome"] == "correct_hit"),
    }


def window_metrics():
    from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences, select_hybrid
    rows, docs, cases = _data()
    model, rng = _model(), random.Random(0)
    boilerplate = repeated_sentences(docs)
    kept_all, n, tokens = 0, 0, 0
    for r in rows:
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
        chunks = chunk_docs(docs, rng.sample(r["attachments"], len(r["attachments"])))
        if approx_tokens(case_text) + sum(c["tokens"] for c in chunks) < MIN_CONTEXT_TOKENS:
            continue
        qv = model.encode([r["query"]], normalize_embeddings=True)[0]
        cv = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
        kept, _ = select_hybrid(chunks, qv, cv, boilerplate, budget=CFG["window"]["budget"])
        text = (case_text + "\n" + "\n".join(c["text"] for c in kept)).lower()
        n += 1
        kept_all += all(f.lower() in text for f in r["gold_facts"])
        tokens += approx_tokens(case_text) + sum(c["tokens"] for c in kept)
    return {"window_all_facts_kept": round(kept_all / n, 4), "window_queries": n,
            "window_mean_tokens": round(tokens / n)}


def check(metrics):
    """List of (name, value, bound, ok)."""
    g = CFG["gates"]["deterministic"]
    return [
        ("exact_false_hits", metrics["exact_false_hits"], f"<= {g['exact_false_hits_max']}",
         metrics["exact_false_hits"] <= g["exact_false_hits_max"]),
        ("exact_hits", metrics["exact_hits"], f">= {g['exact_hits_min']}", metrics["exact_hits"] >= g["exact_hits_min"]),
        ("semantic_wrong_hits", metrics["semantic_wrong_hits"], f"<= {g['semantic_wrong_hits_max']}",
         metrics["semantic_wrong_hits"] <= g["semantic_wrong_hits_max"]),
        ("semantic_paraphrase_hits", metrics["semantic_paraphrase_hits"], f">= {g['semantic_paraphrase_hits_min']}",
         metrics["semantic_paraphrase_hits"] >= g["semantic_paraphrase_hits_min"]),
        ("window_all_facts_kept", metrics["window_all_facts_kept"], f">= {g['window_all_facts_kept_min']}",
         metrics["window_all_facts_kept"] >= g["window_all_facts_kept_min"]),
    ]


def main():
    metrics = {**exact_metrics(), **semantic_metrics(), **window_metrics()}
    rows = check(metrics)
    for name, value, bound, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:28s} {value!s:>8}  (bound {bound})")
    print(json.dumps(metrics))
    raise SystemExit(0 if all(ok for *_, ok in rows) else 1)


if __name__ == "__main__":
    main()
