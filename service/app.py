"""CostGuard demo service.

Run: ./run_demo.sh   (or: set -a; . ./.env; set +a; HF_HUB_OFFLINE=1 .venv/bin/uvicorn service.app:app --port 8000)

  GET  /                 demo page
  POST /v1/answer        run the pipeline on one request
  GET  /metrics          Prometheus metrics
  GET  /api/demo         demo corpus doc ids and demo queries (to prefill the form)
  GET  /api/recent       last requests
  GET  /api/cache        cache entry counts;  DELETE /api/cache?tenant=...  clears it
  GET  /healthz
"""

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import CFG
from demo_data.build_dataset import load_corpus
from service import metrics
from service.pipeline import Pipeline
from service.router_model import load_or_train
from service.store import CacheStore, tenant_slug
from window.manager import chunk_docs, repeated_sentences

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "service_data"
EMBED_MODEL = CFG["models"]["embedding"]

app = FastAPI(title="LLM CostGuard")
state = {}


class CustomDoc(BaseModel):
    title: str = ""
    text: str


class Attachment(BaseModel):
    doc_id: str
    sections: list[str] | None = None


LLM_MODES = set(os.environ.get("COSTGUARD_LLM_MODES", "live,replay,mock").split(","))


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    tenant: str = "default"
    doc_ids: list[str] = []
    attachments: list[Attachment] = []  # doc ids with optional sections; overrides doc_ids
    custom_docs: list[CustomDoc] = []
    case: dict | str | None = None
    window_budget: int = Field(800, ge=200, le=8000)
    shadow: bool = True
    use_cache: bool = True
    llm_mode: str = "live"  # live | replay (memoized answers, free) | mock (no API, load tests)


@app.on_event("startup")
def startup():
    t = time.perf_counter()
    corpus = load_corpus()
    model = SentenceTransformer(EMBED_MODEL, device="cpu")  # the Mac GPU goes cold within seconds of idle
    cases = json.loads((ROOT / "demo_data" / "cases.json").read_text())
    store = CacheStore(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    embed = lambda texts: list(model.encode(list(texts), normalize_embeddings=True))
    vec_of = {}
    for d in corpus:
        for c in chunk_docs(corpus, [{"doc_id": d}]):
            vec_of[c["id"]] = c["text"]
    chunk_vecs = dict(zip(vec_of, embed(vec_of.values())))
    router = load_or_train(DATA / "router.joblib", corpus, cases, embed,
                           lambda c: chunk_vecs[c["id"]], repeated_sentences(corpus))
    results = ROOT / "results"
    state["pipeline"] = Pipeline(
        store, model, corpus, router, results / "judge_verdicts_haiku-4-5.json",
        memo_path=results / "e2e__api_memo.json",
        grade_paths=[results / "e2e__per_query_all_pipelines.csv", results / "e2e_evalset__per_query_all_pipelines.csv"])
    state["store"] = store
    state["corpus"] = corpus
    state["cases"] = cases
    state["queries"] = [json.loads(line) for line in (ROOT / "demo_data" / "queries.jsonl").open()]
    metrics.load_offline_eval(ROOT / "results" / "e2e__cost_vs_quality_summary.csv")
    print(f"CostGuard ready in {time.perf_counter() - t:.1f}s; router threshold {router.threshold}")


@app.on_event("shutdown")
def shutdown():
    state["pipeline"].tracer.flush()  # send any buffered Langfuse spans


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/healthz")
def healthz():
    return {"redis": bool(state["store"].ping()), "router_threshold": state["pipeline"].router.threshold}


@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/demo")
def demo():
    corpus = state["corpus"]
    return {
        "docs": [{"id": d, "title": corpus[d]["title"]} for d in corpus],
        "cases": state["cases"],
        "queries": [{"id": q["id"], "query": q["query"], "variant": q["variant"], "case_id": q["case_id"],
                     "doc_ids": [a["doc_id"] for a in q["attachments"]]} for q in state["queries"]],
    }


@app.post("/v1/answer")
def answer(req: AnswerRequest):
    if not (req.doc_ids or req.attachments) and not any(d.text.strip() for d in req.custom_docs):
        raise HTTPException(400, "attach at least one policy doc")
    if req.llm_mode not in LLM_MODES:
        raise HTTPException(400, f"llm_mode must be one of {sorted(LLM_MODES)}")
    try:
        r = state["pipeline"].answer(
            req.query, tenant=req.tenant, doc_ids=req.doc_ids,
            custom_docs=[d.model_dump() for d in req.custom_docs], case=req.case,
            window_budget=req.window_budget, shadow=req.shadow, use_cache=req.use_cache,
            attachments=[a.model_dump(exclude_none=True) for a in req.attachments] or None, llm_mode=req.llm_mode)
    except Exception as e:
        metrics.ERRORS.labels(type(e).__name__).inc()
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    metrics.record(r)
    state["store"].log_request({k: r[k] for k in (
        "request_id", "tenant", "query", "layer", "model", "cost_usd", "baseline_cost_usd", "savings_pct",
        "latency_s", "created", "llm_mode")} | {"quality": r["quality"]["verdict"] if r["quality"] else None})
    return r


@app.get("/api/recent")
def recent(n: int = 50):
    return state["store"].recent(n)


@app.get("/api/cache")
def cache_counts(tenant: str | None = None):
    return state["store"].counts(tenant_slug(tenant) if tenant else None)


@app.delete("/api/cache")
def cache_clear(tenant: str | None = None):
    return {"deleted": state["store"].clear(tenant_slug(tenant) if tenant else None)}
