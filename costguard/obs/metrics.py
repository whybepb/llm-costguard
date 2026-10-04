"""Prometheus metrics for CostGuard: one registry, one hook that turns each TraceRecord into counters/histograms,
and `mount(app)` which adds `GET /metrics` plus the drift endpoints (`GET /v1/drift`, `POST /v1/drift/baseline`).

All metric names start with `costguard_`. Label values are clamped to small, known sets so a misbehaving client
cannot explode series cardinality (no tenant, query or request-id labels; those live in the SQLite log).

Useful PromQL (also in docs/RUNBOOK.md):
  savings %        1 - sum(rate(costguard_cost_usd_total[1h])) / sum(rate(costguard_baseline_cost_usd_total[1h]))
  hit rate         sum(rate(costguard_requests_total{cache_status=~"exact|semantic"}[5m])) / sum(rate(costguard_requests_total[5m]))
  p99 overhead     histogram_quantile(0.99, sum by (le) (rate(costguard_overhead_ms_bucket[5m])))
  error rate       sum(rate(costguard_request_errors_total[5m])) / sum(rate(costguard_requests_total[5m]))
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any, Optional

from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest)

from ..schemas import TraceRecord

log = logging.getLogger("costguard.obs")

CATEGORIES = ("order", "shipping", "returns", "refund", "payment", "account", "product", "other")
MODES = ("off", "quality", "balanced", "economy")
CACHE_STATUSES = ("miss", "exact", "semantic", "bypass", "disabled")
TIERS = ("strong", "cheap")

# ---------------------------------------------------------------- buckets
LATENCY_MS = (1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)
OVERHEAD_MS = (0.5, 1, 2, 5, 10, 20, 35, 50, 75, 100, 200, 500, 1000)       # 50 ms = hit-path p99 SLO
STAGE_MS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 1000)
TOKENS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
SIMILARITY = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99, 1.0)
COMPRESSION = (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 20)
COST_USD = (1e-6, 5e-6, 1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2)

_SAFE = re.compile(r"[^a-z0-9_]+")


def _clamp(value: Optional[str], allowed: tuple[str, ...], none: str = "none", other: str = "other") -> str:
    if value is None or value == "":
        return none
    v = str(value).lower()
    return v if v in allowed else other


def _slug(value: Optional[str], limit: int = 32) -> str:
    """Free-form label (stage names, guard reasons): keep the token before ':' and only [a-z0-9_]."""
    if not value:
        return "none"
    v = _SAFE.sub("_", str(value).split(":")[0].strip().lower()).strip("_")
    return (v or "other")[:limit]


class CostGuardMetrics:
    """All CostGuard series on a private registry (so tests and multiple apps can each own one)."""

    def __init__(self, registry: Optional[CollectorRegistry] = None, *, process_metrics: bool = False):
        self.registry = r = registry or CollectorRegistry(auto_describe=True)
        if process_metrics:
            try:  # CPU / RSS / fds of the serving process (Linux); harmless no-op elsewhere
                from prometheus_client import GCCollector, PlatformCollector, ProcessCollector
                ProcessCollector(registry=r)
                PlatformCollector(registry=r)
                GCCollector(registry=r)
            except Exception:  # pragma: no cover - platform dependent
                pass
        # --- traffic mix
        self.requests = Counter("costguard_requests", "Requests handled",
                                ["mode", "cache_status", "model_used", "category"], registry=r)
        self.requests_by_config = Counter("costguard_requests_by_config", "Requests per policy config_hash "
                                          "(shows canary/rollback traffic split)", ["config_hash"], registry=r)
        self.request_errors = Counter("costguard_request_errors", "Requests that failed upstream (5xx to client)",
                                      ["mode"], registry=r)
        self.stage_errors = Counter("costguard_stage_errors", "Optional stages that raised and failed open",
                                    ["stage"], registry=r)
        self.cache_guard = Counter("costguard_cache_guard_rejections", "Semantic near-hits refused by a guard",
                                   ["reason"], registry=r)
        # --- money
        self.cost = Counter("costguard_cost_usd", "Actual list-price cost (USD)", ["mode", "model_used"], registry=r)
        self.baseline_cost = Counter("costguard_baseline_cost_usd", "Cost of the same requests on the strong tier, "
                                     "full prompt, no cache (USD)", ["mode"], registry=r)
        self.saved = Counter("costguard_saved_usd", "Baseline minus actual cost, per request clamped at >= 0 (USD)",
                             ["mode", "cache_status"], registry=r)
        self.negative_savings = Counter("costguard_negative_savings", "Requests that cost MORE than the baseline",
                                        ["mode"], registry=r)
        self.request_cost = Histogram("costguard_request_cost_usd", "Per-request cost (USD); alert on >5x median",
                                      ["mode"], buckets=COST_USD, registry=r)
        # --- latency
        self.latency = Histogram("costguard_latency_ms", "End-to-end latency inside CostGuard (ms)",
                                 ["cache_status"], buckets=LATENCY_MS, registry=r)
        self.overhead = Histogram("costguard_overhead_ms", "Latency added by CostGuard stages = total - upstream (ms)",
                                  ["cache_status"], buckets=OVERHEAD_MS, registry=r)
        self.upstream = Histogram("costguard_upstream_latency_ms", "Provider call latency (ms)", ["model_used"],
                                  buckets=LATENCY_MS, registry=r)
        self.stage = Histogram("costguard_stage_ms", "Per-stage latency (ms)", ["stage"], buckets=STAGE_MS,
                               registry=r)
        # --- tokens / levers
        self.input_tokens = Histogram("costguard_input_tokens", "Input tokens per request: original (full prompt) "
                                      "vs sent (billed upstream; 0 on a cache hit)", ["kind"], buckets=TOKENS,
                                      registry=r)
        self.output_tokens = Histogram("costguard_output_tokens", "Output tokens generated upstream", ["model_used"],
                                       buckets=TOKENS, registry=r)
        self.provider_cache_tokens = Counter("costguard_provider_cache_tokens", "Provider prompt-cache tokens "
                                             "(Anthropic cache reads / writes); read/sent = prefix-cache hit ratio",
                                             ["kind"], registry=r)
        self.sent_tokens = Counter("costguard_sent_input_tokens", "Input tokens billed upstream", registry=r)
        self.similarity = Histogram("costguard_cache_similarity", "Best semantic-cache cosine similarity seen",
                                    ["outcome"], buckets=SIMILARITY, registry=r)
        self.compression = Histogram("costguard_compression_ratio", "Context tokens before / after compression",
                                     buckets=COMPRESSION, registry=r)
        self.context_docs = Counter("costguard_context_docs", "Retrieved context documents in / kept", ["kind"],
                                    registry=r)
        self.last_request = Gauge("costguard_last_request_timestamp_seconds", "Unix time of the last request",
                                  registry=r)
        self.build_info = Gauge("costguard_build_info", "Serving config (value is always 1)",
                                ["config_hash", "backend", "provider", "default_mode"], registry=r)
        # --- drift (W3S2 PSI) + hook health
        self.drift_psi = Gauge("costguard_drift_psi", "Population stability index vs the baseline window",
                               ["feature"], registry=r)
        self.drift_status = Gauge("costguard_drift_status", "-1 insufficient data, 0 stable (<0.10), "
                                  "1 investigate (0.10-0.25), 2 alert (>0.25)", ["feature"], registry=r)
        self.drift_window = Gauge("costguard_drift_window_requests", "Requests in the current drift window",
                                  registry=r)
        self.drift_baseline = Gauge("costguard_drift_baseline_requests", "Requests in the frozen drift baseline",
                                    registry=r)
        self.hook_errors = Counter("costguard_hook_errors", "Exceptions swallowed by observability hooks", ["hook"],
                                   registry=r)
        self.hook_dropped = Counter("costguard_hook_dropped", "Telemetry events dropped (queue full / sampled out)",
                                    ["hook", "reason"], registry=r)

    # ------------------------------------------------------------ recording
    def observe(self, rec: TraceRecord) -> None:
        mode = _clamp(rec.mode, MODES)
        cache = _clamp(rec.cache_status, CACHE_STATUSES)
        tier = _clamp(rec.model_used, TIERS, none="cache")
        cat = _clamp(rec.category, CATEGORIES)

        self.requests.labels(mode, cache, tier, cat).inc()
        self.requests_by_config.labels(_slug(rec.config_hash, 16)).inc()
        self.last_request.set(rec.ts or time.time())
        if rec.error:
            self.request_errors.labels(mode).inc()
        for stage in rec.stage_errors or {}:
            self.stage_errors.labels(_slug(stage)).inc()
        if rec.cache_guard:
            self.cache_guard.labels(_slug(rec.cache_guard)).inc()

        cost, base = max(0.0, float(rec.cost_usd or 0)), max(0.0, float(rec.baseline_cost_usd or 0))
        self.cost.labels(mode, tier).inc(cost)
        self.baseline_cost.labels(mode).inc(base)
        saved = float(rec.saved_usd or 0)
        if saved >= 0:
            self.saved.labels(mode, cache).inc(saved)
        else:
            self.negative_savings.labels(mode).inc()
        self.request_cost.labels(mode).observe(cost)

        self.latency.labels(cache).observe(max(0.0, rec.latency_ms))
        self.overhead.labels(cache).observe(max(0.0, rec.overhead_ms))
        if rec.model_used:  # upstream was actually called
            self.upstream.labels(tier).observe(max(0.0, rec.upstream_latency_ms))
            self.output_tokens.labels(tier).observe(max(0, rec.output_tokens))
        for stage, ms in (rec.stage_ms or {}).items():
            self.stage.labels(_slug(stage)).observe(max(0.0, float(ms)))

        self.input_tokens.labels("original").observe(max(0, rec.input_tokens_original))
        self.input_tokens.labels("sent").observe(max(0, rec.input_tokens_sent))
        self.sent_tokens.inc(max(0, rec.input_tokens_sent))
        if rec.cached_input_tokens:
            self.provider_cache_tokens.labels("read").inc(rec.cached_input_tokens)
        writes = int(getattr(rec, "cache_write_tokens", 0) or 0)   # not on TraceRecord yet; forward-compatible
        if writes:
            self.provider_cache_tokens.labels("write").inc(writes)
        if rec.cache_similarity is not None:
            outcome = "hit" if rec.cache_status == "semantic" else ("guarded" if rec.cache_guard else "miss")
            self.similarity.labels(outcome).observe(float(rec.cache_similarity))
        if rec.compression_ratio:
            self.compression.observe(float(rec.compression_ratio))
        if rec.context_docs_in:
            self.context_docs.labels("in").inc(rec.context_docs_in)
            self.context_docs.labels("kept").inc(rec.context_docs_kept)

    def set_build_info(self, engine: Any) -> None:
        try:
            pol = engine.policy
            self.build_info.labels(_slug(pol.config_hash, 16), _slug(engine.backend), _slug(engine.provider.name),
                                   _slug(pol.default_mode)).set(1)
        except Exception:
            pass

    def set_drift(self, report: dict) -> None:
        codes = {"insufficient_data": -1, "stable": 0, "investigate": 1, "alert": 2}
        for name, f in (report.get("features") or {}).items():
            if f.get("psi") is not None:
                self.drift_psi.labels(name).set(f["psi"])
            self.drift_status.labels(name).set(codes.get(f.get("band"), -1))
        self.drift_window.set(report.get("window_requests", 0) or 0)
        self.drift_baseline.set(report.get("baseline_requests", 0) or 0)

    def render(self) -> bytes:
        return generate_latest(self.registry)


_default: Optional[CostGuardMetrics] = None
_default_lock = threading.Lock()


def get_default_metrics() -> CostGuardMetrics:
    """Process-wide metrics (one Prometheus registry per process, as Prometheus expects)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = CostGuardMetrics(process_metrics=True)
        return _default


class MetricsHook:
    """Engine hook: TraceRecord -> Prometheus series. Never raises."""

    name = "metrics"

    def __init__(self, metrics: Optional[CostGuardMetrics] = None):
        self.metrics = metrics or get_default_metrics()

    def __call__(self, rec: TraceRecord) -> None:
        try:
            self.metrics.observe(rec)
        except Exception as e:  # observability must never break serving
            log.debug("metrics hook failed: %s", e)
            try:
                self.metrics.hook_errors.labels(self.name).inc()
            except Exception:
                pass


def _unwrap(h: Any) -> Any:
    return getattr(h, "inner", h)


def _find_hook(engine: Any, cls: type) -> Optional[Any]:
    for h in getattr(engine, "hooks", None) or []:
        h = _unwrap(h)
        if isinstance(h, cls):
            return h
    return None


def mount(app) -> None:
    """Add GET /metrics, GET /v1/drift and POST /v1/drift/baseline. Called by costguard.server.create_app."""
    from fastapi import Header, HTTPException, Response

    engine = getattr(app.state, "engine", None)
    mh = _find_hook(engine, MetricsHook)
    metrics = mh.metrics if mh else get_default_metrics()
    metrics.set_build_info(engine)
    try:
        from .drift import DriftHook
        dh = _find_hook(engine, DriftHook)
    except Exception:  # pragma: no cover
        dh = None
    app.state.metrics = metrics
    app.state.drift = dh

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        if dh is not None:
            try:
                metrics.set_drift(dh.monitor.report())
            except Exception:
                pass
        return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/drift")
    def drift_report() -> dict[str, Any]:
        if dh is None:
            return {"status": "disabled", "reason": "drift hook not installed (COSTGUARD_DRIFT=0 or custom engine)"}
        rep = dh.monitor.report()
        metrics.set_drift(rep)
        return rep

    @app.post("/v1/drift/baseline")
    def drift_rebaseline(x_costguard_admin_token: Optional[str] = Header(default=None)) -> dict[str, Any]:
        """Freeze the current window as the new baseline (after an intended change, e.g. a kb_version bump)."""
        token = os.environ.get("COSTGUARD_ADMIN_TOKEN")
        if token and x_costguard_admin_token != token:
            raise HTTPException(status_code=403, detail="admin token required")
        if dh is None:
            raise HTTPException(status_code=404, detail="drift monitoring disabled")
        dh.monitor.freeze_baseline(from_window=True)
        return dh.monitor.report()
