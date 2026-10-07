"""End-to-end cost vs quality: baseline Sonnet 5.5 on full context vs the CostGuard pipeline.

Usage: set -a; . ./.env; set +a; .venv/bin/python -m e2e.run [--data eval_data]

--data demo_data (default): the 150 tuning queries. Router cross-fitted by cluster;
  semantic-cache depends_on from the gold docs. Writes results/e2e__*.csv.
--data eval_data: the held-out set. Router trained on the 150 demo queries only;
  depends_on from the docs the 800 window keeps (as the live service does), so no
  label of the eval set is used. Writes results/e2e_evalset__*.csv.

Pipelines (all 150 queries, replayed in order):
  baseline                     Sonnet 5.5, full context (case + every attachment), no cache
  optimized_w800               exact cache -> semantic cache (Haiku 4.5 judge) -> on a miss:
                               hybrid window 800 -> router -> Haiku 5.5 or Sonnet 5.5
  optimized_w1200              same, window 1200
  optimized_w800_lingua0.7     same as w800, then LLMLingua-2 at rate 0.7
  optimized_w1200_lingua0.7    same as w1200, then LLMLingua-2 at rate 0.7

A cache hit returns the real answer generated earlier for the query it was served from.
The router is cross-fitted by cluster: each query is routed by a model trained on the
other half of the clusters, with its threshold set from out-of-fold predictions.

Every answer is graded blind by Sonnet 5.5 against the dataset's reference answer,
gold facts and must-not-claim list. Costs come from each response's token usage.
All API calls are memoized in results/e2e__api_memo.json.
"""

import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import GroupKFold

from config import CFG, prompt as load_prompt
from cache.exact import ExactCache, cache_key
from cache.guards import GUARDS
from cache.judge import Judge
from cache.semantic import SemanticCache
from demo_data.build_dataset import assemble, load_corpus, render_attachment, render_case
from e2e.llm import LLM, cost
from routing.router import LARGE, SMALL, Router, text_features
from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, repeated_sentences, select_hybrid

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
EMBED_MODEL = CFG["models"]["embedding"]
SEMANTIC_THRESHOLD = CFG["cache"]["semantic_threshold"]
JUDGE_MODEL = CFG["models"]["eval_judge"]
BASELINE_MODEL = CFG["models"]["baseline"]
VARIANTS = {"optimized_w800": (800, None), "optimized_w1200": (1200, None),
            "optimized_w800_lingua0.7": (800, 0.7), "optimized_w1200_lingua0.7": (1200, 0.7)}
WORKERS = 8

ANSWER_SYSTEM = load_prompt("answer_system")
GRADER_SYSTEM = load_prompt("eval_grader_system")


def prompt(query, context):
    return f"<context>\n{context}\n</context>\n\nCustomer message:\n{query}"


def optimized_context(case_text, chunks):
    parts = ["## Case\n" + case_text] if case_text else []
    for c in chunks:
        header, body = c["text"].split("\n", 1)
        parts.append(f"## {header}\n{body}")
    return "\n\n".join(parts)


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo_data")
    ap.add_argument("--prefix", default=None, help="output file prefix (default e2e / e2e_evalset)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated subset of VARIANTS")
    args = ap.parse_args()
    held_out = args.data != "demo_data"
    variants = {v: VARIANTS[v] for v in args.variants.split(",")}
    prefix = args.prefix or ("e2e_evalset" if held_out else "e2e")
    rows = [json.loads(line) for line in (ROOT / args.data / "queries.jsonl").open()]
    docs = load_corpus()
    cases = json.loads((ROOT / args.data / "cases.json").read_text())
    boilerplate = repeated_sentences(docs)
    embed = SentenceTransformer(EMBED_MODEL)
    llm = LLM(OUT / "e2e__api_memo.json")
    cache_judge = Judge(OUT / "judge_verdicts_haiku-4-5.json")

    all_chunks = {c["id"]: c for d in docs for c in chunk_docs(docs, [{"doc_id": d}])}
    ids = list(all_chunks)
    chunk_vec = dict(zip(ids, embed.encode([all_chunks[i]["text"] for i in ids], normalize_embeddings=True)))
    qvecs = embed.encode([r["query"] for r in rows], normalize_embeddings=True)

    base = []
    for r in rows:
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
        doc_texts = {a["doc_id"]: render_attachment(docs, a) for a in r["attachments"]}
        sha = lambda t: __import__("hashlib").sha256(t.encode()).hexdigest()
        case_hash = {"case:" + sha(case_text)} if case_text else set()
        base.append({
            "row": r, "case_text": case_text,
            "full_context": assemble(docs, cases, r),
            "key": cache_key(r["query"], case_text or None, list(doc_texts.values())),
            "context_hashes": frozenset(sha(t) for t in doc_texts.values()) | case_hash,
            "depends_on": frozenset(sha(doc_texts[d]) for d in r["gold_doc_ids"]) | case_hash,
        })
        if held_out:  # no gold labels: the answer depends on the docs the 800 window keeps
            qv = qvecs[len(base) - 1]
            chunks = chunk_docs(docs, r["attachments"])
            if approx_tokens(case_text) + sum(c["tokens"] for c in chunks) >= MIN_CONTEXT_TOKENS:
                chunks, _ = select_hybrid(chunks, qv, [chunk_vec[c["id"]] for c in chunks], boilerplate, budget=800)
            base[-1]["depends_on"] = frozenset(sha(doc_texts[c["doc_id"]]) for c in chunks) | case_hash

    # --- Cache decisions (same for every variant: they don't depend on answers) ---
    exact = ExactCache()
    sem = SemanticCache(SEMANTIC_THRESHOLD, check_context=True, guards=[*GUARDS, ("judge", cache_judge.same)])
    for b, qv in zip(base, qvecs):
        r = b["row"]
        hit = exact.get(b["key"])
        if hit is not None:
            b["layer"], b["served_from"] = "exact_cache", hit["source_id"]
            continue
        hit, _, _ = sem.lookup(qv, r["query"], b["context_hashes"])
        if hit is not None:
            b["layer"], b["served_from"] = "semantic_cache", hit["source_id"]
            continue
        b["layer"], b["served_from"] = "llm", ""
        value = {"source_id": r["id"], "answer_id": r["answer_id"]}
        exact.put(b["key"], value)
        sem.put(qv, value, r["query"], b["depends_on"])
    cache_judge_cost = sum(cost("claude-haiku-4-5", v["input_tokens"], v["output_tokens"]) for v in cache_judge.used)
    misses = [b for b in base if b["layer"] == "llm"]

    # --- Per-variant context and routing for cache misses ---
    lingua = None
    clusters = sorted({r["cluster_id"] for r in rows})
    rng = random.Random(0)
    rng.shuffle(clusters)
    fold_of = {c: i % 2 for i, c in enumerate(clusters)}
    plans = {}
    for variant, (budget, rate) in variants.items():
        items = []
        for b in base:
            qv = qvecs[rows.index(b["row"])]
            chunks = chunk_docs(docs, b["row"]["attachments"])
            full = approx_tokens(b["case_text"]) + sum(c["tokens"] for c in chunks)
            if full >= MIN_CONTEXT_TOKENS:
                chunks, _ = select_hybrid(chunks, qv, [chunk_vec[c["id"]] for c in chunks], boilerplate, budget=budget)
                if rate:
                    if lingua is None:
                        from prompt_compression.lingua import Lingua
                        lingua = Lingua()
                    chunks = lingua.compress(chunks, rate)
            ctx_tokens = approx_tokens(b["case_text"]) + sum(c["tokens"] for c in chunks)
            items.append({"b": b, "qv": qv, "context": optimized_context(b["case_text"], chunks),
                          "feats": text_features(b["row"]["query"], len({c["doc_id"] for c in chunks}), ctx_tokens),
                          "y": int(not b["row"]["small_model_ok"])})
        if held_out:  # router trained on the demo set only, with this variant's window budget
            from service.router_model import train
            demo_cases = json.loads((ROOT / "demo_data" / "cases.json").read_text())
            router = train(docs, demo_cases, lambda t: list(embed.encode(list(t), normalize_embeddings=True)),
                           lambda c: chunk_vec[c["id"]], boilerplate, budget=budget)
            p = router.p_large(np.stack([i["qv"] for i in items]), np.array([i["feats"] for i in items], float))
            for i, s in zip(items, p):
                i["model"], i["p_large"], i["threshold"] = (LARGE if s >= router.threshold else SMALL), float(s), router.threshold
        # Cross-fit router: train on one fold of clusters, route the other.
        for fold in ((0, 1) if not held_out else ()):
            tr = [i for i in items if fold_of[i["b"]["row"]["cluster_id"]] != fold]
            te = [i for i in items if fold_of[i["b"]["row"]["cluster_id"]] == fold]
            Xe, Xf, y = np.stack([i["qv"] for i in tr]), np.array([i["feats"] for i in tr], float), np.array([i["y"] for i in tr])
            groups = [i["b"]["row"]["cluster_id"] for i in tr]
            oof = np.zeros(len(tr))
            for a, v in GroupKFold(n_splits=min(5, len(set(groups)))).split(Xe, y, groups):
                oof[v] = Router().fit(Xe[a], Xf[a], y[a]).p_large(Xe[v], Xf[v])
            ok = [t for t in [round(0.05 * k, 2) for k in range(1, 20)]
                  if not any(yy and s < t for yy, s in zip(y, oof))]
            threshold = max(ok) if ok else 0.05
            router = Router(threshold).fit(Xe, Xf, y)
            p = router.p_large(np.stack([i["qv"] for i in te]), np.array([i["feats"] for i in te], float))
            for i, s in zip(te, p):
                i["model"], i["p_large"], i["threshold"] = (LARGE if s >= threshold else SMALL), float(s), threshold
        plans[variant] = {i["b"]["row"]["id"]: i for i in items}

    # --- Generate answers (baseline for all; variants for cache misses) ---
    jobs = [("baseline", b["row"]["id"], BASELINE_MODEL, prompt(b["row"]["query"], b["full_context"])) for b in base]
    for variant, plan in plans.items():
        for b in misses:
            i = plan[b["row"]["id"]]
            jobs.append((variant, b["row"]["id"], i["model"], prompt(b["row"]["query"], i["context"])))
    with ThreadPoolExecutor(WORKERS) as ex:
        answers = dict(zip([(v, q) for v, q, _, _ in jobs],
                           ex.map(lambda j: {**llm.call(j[2], ANSWER_SYSTEM, j[3]), "model": j[2]}, jobs)))

    results = []
    for b in base:
        r = b["row"]
        a = answers[("baseline", r["id"])]
        results.append({"pipeline": "baseline", "b": b, "layer": "llm", "model": BASELINE_MODEL, "served_from": "",
                        "answer": a, "cost": cost(BASELINE_MODEL, a["input_tokens"], a["output_tokens"])})
    for variant in variants:
        for b in base:
            r = b["row"]
            if b["layer"] == "llm":
                a = answers[(variant, r["id"])]
                results.append({"pipeline": variant, "b": b, "layer": "llm", "model": a["model"], "served_from": "",
                                "answer": a, "cost": cost(a["model"], a["input_tokens"], a["output_tokens"])})
            else:
                a = answers[(variant, b["served_from"])]
                results.append({"pipeline": variant, "b": b, "layer": b["layer"], "model": "", "served_from": b["served_from"],
                                "answer": a, "cost": 0.0})

    # --- Grade every answer blind ---
    def grade(res):
        r, b = res["b"]["row"], res["b"]
        user = (f"Customer message:\n{r['query']}\n\nCase details:\n{b['case_text'] or '(none)'}\n\n"
                f"Reference answer:\n{r['reference_answer']}\n\nFacts a correct answer relies on:\n- "
                + "\n- ".join(r["gold_facts"]) + "\n\nMust not claim:\n- " + "\n- ".join(r["must_not_claim"] or ["(none)"])
                + f"\n\nCandidate answer:\n{res['answer']['text'] or '(empty)'}")
        g = llm.call(JUDGE_MODEL, GRADER_SYSTEM, user, max_tokens=2000, effort="low")
        m = re.search(r"\{.*\}", g["text"], re.S)
        try:
            parsed = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            parsed = {}
        return {"verdict": parsed.get("verdict", "unparsed"), "score": parsed.get("score", ""),
                "reason": parsed.get("reason", g["text"][:200]),
                "grader_cost": cost(JUDGE_MODEL, g["input_tokens"], g["output_tokens"])}

    with ThreadPoolExecutor(WORKERS) as ex:
        grades = list(ex.map(grade, results))

    per_query = []
    for res, g in zip(results, grades):
        r = res["b"]["row"]
        plan = plans.get(res["pipeline"], {}).get(r["id"], {})
        per_query.append({
            "pipeline": res["pipeline"], "id": r["id"], "cluster_id": r["cluster_id"], "variant": r["variant"],
            "difficulty": r["difficulty"], "layer": res["layer"], "model": res["model"], "served_from": res["served_from"],
            "routed_label": (LARGE if not r["small_model_ok"] else SMALL),
            "router_p_large": round(plan.get("p_large", 0), 4) if res["layer"] == "llm" and plan else "",
            "input_tokens": res["answer"]["input_tokens"] if res["layer"] == "llm" else 0,
            "output_tokens": res["answer"]["output_tokens"] if res["layer"] == "llm" else 0,
            "cost_usd": round(res["cost"], 6), "verdict": g["verdict"], "score": g["score"], "grader_reason": g["reason"],
            "query": r["query"], "answer": res["answer"]["text"],
        })

    summary, by_layer = [], []
    for pipeline in ["baseline", *variants]:
        rs = [p for p in per_query if p["pipeline"] == pipeline]
        n = len(rs)
        total = sum(p["cost_usd"] for p in rs) + (cache_judge_cost if pipeline != "baseline" else 0)
        scores = [p["score"] for p in rs if isinstance(p["score"], (int, float))]
        summary.append({
            "pipeline": pipeline, "queries": n,
            "total_cost_usd": round(total, 4),
            "cost_per_1k_queries_usd": round(1000 * total / n, 3),
            "cache_judge_cost_usd": round(cache_judge_cost, 4) if pipeline != "baseline" else 0,
            "correct_pct": round(sum(p["verdict"] == "correct" for p in rs) / n, 4),
            "partial_pct": round(sum(p["verdict"] == "partial" for p in rs) / n, 4),
            "wrong_pct": round(sum(p["verdict"] == "wrong" for p in rs) / n, 4),
            "unparsed": sum(p["verdict"] not in ("correct", "partial", "wrong") for p in rs),
            "mean_score": round(sum(scores) / max(1, len(scores)), 3),
            "served_by_cache": sum(p["layer"] != "llm" for p in rs),
            "routed_haiku": sum(p["model"] == SMALL for p in rs),
            "routed_sonnet": sum(p["model"] == LARGE for p in rs),
            "input_tokens_total": sum(p["input_tokens"] for p in rs),
        })
        for layer in sorted({(p["layer"], p["model"]) for p in rs}):
            ls = [p for p in rs if (p["layer"], p["model"]) == layer]
            by_layer.append({"pipeline": pipeline, "layer": layer[0], "model": layer[1], "queries": len(ls),
                             "correct_pct": round(sum(p["verdict"] == "correct" for p in ls) / len(ls), 4),
                             "wrong_pct": round(sum(p["verdict"] == "wrong" for p in ls) / len(ls), 4),
                             "cost_usd": round(sum(p["cost_usd"] for p in ls), 4)})
    base_cost = summary[0]["total_cost_usd"]
    for s in summary:
        s["cost_vs_baseline_pct"] = round(s["total_cost_usd"] / base_cost, 4)

    OUT.mkdir(exist_ok=True)
    write_csv(OUT / f"{prefix}__per_query_all_pipelines.csv", per_query)
    write_csv(OUT / f"{prefix}__cost_vs_quality_summary.csv", summary)
    write_csv(OUT / f"{prefix}__quality_by_layer.csv", by_layer)
    grader_total = sum(g["grader_cost"] for g in grades)
    print(json.dumps({"summary": summary, "grader_cost_usd": round(grader_total, 4),
                      "new_api_calls": llm.new_calls}, indent=1))


if __name__ == "__main__":
    main()
