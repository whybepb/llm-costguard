"""OpenAI-compatible HTTP front door. Drop-in: point any OpenAI SDK client's base_url at http://host:8000/v1.

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
    client.chat.completions.create(model="strong", messages=[...], extra_body={"costguard": {"mode": "balanced"}})
"""
from __future__ import annotations

import importlib
import statistics
import time
from typing import Any, Optional

import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from .factory import build_engine
from .obs.logger import RequestLogger
from .schemas import ChatRequest


def _pct(xs: list[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1)))))
    return round(xs[k], 2)


def _api_keys() -> dict[str, str]:
    """COSTGUARD_API_KEYS="key1:tenantA,key2:tenantB". Empty -> open dev mode (single 'default' tenant)."""
    out = {}
    for pair in filter(None, (os.environ.get("COSTGUARD_API_KEYS", "")).split(",")):
        k, _, t = pair.partition(":")
        if k.strip() and t.strip():
            out[k.strip()] = t.strip()
    return out


def resolve_tenant(authorization: Optional[str], keys: dict[str, str]) -> Optional[str]:
    """Service-side identity: the caller's key decides the tenant. Returns None in open dev mode."""
    if not keys:
        return None
    token = (authorization or "").removeprefix("Bearer ").strip()
    for k, tenant in keys.items():
        if token and hmac.compare_digest(token, k):
            return tenant
    raise HTTPException(status_code=401, detail="invalid or missing API key")


def create_app(engine=None) -> FastAPI:
    engine = engine or build_engine()
    app = FastAPI(title="LLM CostGuard", version="0.1.0")
    app.state.engine = engine
    req_logger = next((h for h in engine.hooks if isinstance(h, RequestLogger)), None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "backend": engine.backend, "provider": engine.provider.name,
                "config_hash": engine.policy.config_hash, "default_mode": engine.policy.default_mode,
                "components": getattr(engine, "component_status", {})}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, response: Response, authorization: Optional[str] = Header(default=None)) -> dict[str, Any]:
        tenant = resolve_tenant(authorization, _api_keys())
        if tenant is not None:           # never trust tenant from the body when keys are configured
            req.costguard.tenant = tenant
        try:
            comp, rec = engine.handle(req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"upstream error: {e}")
        for k, v in {"cache": rec.cache_status, "similarity": rec.cache_similarity, "route": rec.model_used or "cache",
                     "cost-usd": f"{rec.cost_usd:.8f}", "baseline-cost-usd": f"{rec.baseline_cost_usd:.8f}",
                     "saved-usd": f"{rec.saved_usd:.8f}", "overhead-ms": f"{rec.overhead_ms:.1f}",
                     "config-hash": rec.config_hash, "request-id": rec.request_id}.items():
            if v is not None:
                response.headers[f"x-costguard-{k}"] = str(v)
        return {
            "id": f"chatcmpl-{rec.request_id}", "object": "chat.completion", "created": int(rec.ts),
            "model": comp.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": comp.text},
                         "finish_reason": comp.finish_reason}],
            "usage": {"prompt_tokens": rec.input_tokens_sent, "completion_tokens": rec.output_tokens,
                      "total_tokens": rec.input_tokens_sent + rec.output_tokens},
            "costguard": rec.model_dump(include={"request_id", "mode", "cache_status", "cache_similarity", "model_used",
                                                 "route_reason", "input_tokens_original", "input_tokens_sent",
                                                 "compression_ratio", "context_docs_in", "context_docs_kept", "cost_usd",
                                                 "baseline_cost_usd", "saved_usd", "latency_ms", "overhead_ms",
                                                 "config_hash", "stage_errors"}),
        }

    @app.get("/v1/stats")
    def stats(since_s: float = 3600 * 24) -> dict[str, Any]:
        if req_logger is None:
            return {"error": "request logging disabled"}
        rows = req_logger.rows("WHERE ts >= ?", (time.time() - since_s,))
        n = len(rows)
        if not n:
            return {"requests": 0}
        hits = [r for r in rows if r["cache_status"] in ("exact", "semantic")]
        lat = [r["latency_ms"] for r in rows]
        cost, base = sum(r["cost_usd"] for r in rows), sum(r["baseline_cost_usd"] for r in rows)
        return {"requests": n, "cache_hit_rate": round(len(hits) / n, 4),
                "semantic_hits": sum(r["cache_status"] == "semantic" for r in rows),
                "exact_hits": sum(r["cache_status"] == "exact" for r in rows),
                "cheap_routed": sum(r["model_used"] == "cheap" for r in rows),
                "cost_usd": round(cost, 6), "baseline_cost_usd": round(base, 6),
                "saved_pct": round(100 * (1 - cost / base), 2) if base else None,
                "latency_ms_p50": _pct(lat, 50), "latency_ms_p99": _pct(lat, 99),
                "overhead_ms_p50": _pct([r["overhead_ms"] for r in rows], 50),
                "mean_latency_ms": round(statistics.fmean(lat), 2)}

    try:  # Prometheus /metrics if the observability module is present
        importlib.import_module("costguard.obs.metrics").mount(app)
    except (ImportError, AttributeError):
        pass
    return app


app = None  # created lazily by `uvicorn costguard.server:get_app --factory`


def get_app() -> FastAPI:
    return create_app()
