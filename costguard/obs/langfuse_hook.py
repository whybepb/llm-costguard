"""Optional Langfuse tracing. Asynchronous: the hook only enqueues the finished TraceRecord (non-blocking put); a
daemon thread builds the client and ships traces, so serving never waits on the network.

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set AND the `langfuse` package is importable
(`pip install langfuse`; optional, not a core dependency). Otherwise `build_langfuse_hook` returns None and
CostGuard runs without it.

Span attributes follow the OpenTelemetry GenAI semantic conventions where one exists (`gen_ai.usage.input_tokens`,
`gen_ai.usage.cache_read.input_tokens`, ...). Cost and CostGuard decisions have no standard attribute, so they sit
under `costguard.*` (`costguard.cost_usd`, `costguard.cache.status`, ...).

Quota: Langfuse Cloud Hobby = 50k units/month; one request here = 1 trace + 1-2 observations. Keep sweeps and CI
off Langfuse with COSTGUARD_LANGFUSE_SAMPLE=0 (or unset the keys); use e.g. 0.1 for a 10% sample in production.
"""
from __future__ import annotations

import atexit
import logging
import os
import queue
import random
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

log = logging.getLogger("costguard.obs")

_STOP = object()


def langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY"))


def otel_attributes(rec: Any) -> dict[str, Any]:
    """TraceRecord -> flat span attributes (OTel GenAI names + costguard.* extensions). Drops None values."""
    tier = rec.model_used or "cache"
    a: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": rec.provider or None,
        "gen_ai.request.model": rec.model_requested,
        "gen_ai.response.model": rec.model_id or f"cache:{rec.cache_status}",
        "gen_ai.usage.input_tokens": int(rec.input_tokens_sent),
        "gen_ai.usage.output_tokens": int(rec.output_tokens),
        "gen_ai.usage.cache_read.input_tokens": int(rec.cached_input_tokens),
        "gen_ai.usage.cache_creation.input_tokens": int(getattr(rec, "cache_write_tokens", 0) or 0),
        "costguard.request_id": rec.request_id,
        "costguard.tenant": rec.tenant,
        "costguard.mode": rec.mode,
        "costguard.category": rec.category,
        "costguard.arm": rec.arm,
        "costguard.config_hash": rec.config_hash,
        "costguard.cache.status": rec.cache_status,
        "costguard.cache.similarity": rec.cache_similarity,
        "costguard.cache.guard": rec.cache_guard,
        "costguard.route.tier": tier,
        "costguard.route.reason": rec.route_reason or None,
        "costguard.input_tokens.original": int(rec.input_tokens_original),
        "costguard.context.docs_in": int(rec.context_docs_in),
        "costguard.context.docs_kept": int(rec.context_docs_kept),
        "costguard.compression.ratio": rec.compression_ratio,
        "costguard.cost_usd": float(rec.cost_usd),
        "costguard.baseline_cost_usd": float(rec.baseline_cost_usd),
        "costguard.saved_usd": float(rec.saved_usd),
        "costguard.latency_ms": round(float(rec.latency_ms), 3),
        "costguard.overhead_ms": round(float(rec.overhead_ms), 3),
        "costguard.upstream_latency_ms": round(float(rec.upstream_latency_ms), 3),
        "error.type": (rec.error or "").split(":")[0] or None,
    }
    for stage, ms in (rec.stage_ms or {}).items():
        a[f"costguard.stage_ms.{stage}"] = float(ms)
    for stage, err in (rec.stage_errors or {}).items():
        a[f"costguard.stage_error.{stage}"] = str(err)[:200]
    return {k: v for k, v in a.items() if v is not None}


class LangfuseHook:
    """Never raises. Drops (and counts) events when the queue is full rather than slowing requests down."""

    name = "langfuse"

    def __init__(self, client: Any = None, *, client_factory: Optional[Callable[[], Any]] = None,
                 sample_rate: float = 1.0, max_queue: int = 2000, flush_every: int = 50, metrics: Any = None):
        self._client, self._factory = client, client_factory
        self.sample_rate = max(0.0, min(1.0, float(sample_rate)))
        self.flush_every = max(1, int(flush_every))
        self.metrics = metrics
        self.enqueued = self.sent = self.failed = self.dropped = 0
        self.disabled_reason: Optional[str] = None
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="costguard-langfuse", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # ------------------------------------------------------------ hot path (O(1), non-blocking)
    def __call__(self, rec: Any) -> None:
        try:
            if self.disabled_reason or (self.sample_rate < 1.0 and random.random() >= self.sample_rate):
                self._drop("disabled" if self.disabled_reason else "sampled")
                return
            self._q.put_nowait(rec)
            self.enqueued += 1
        except queue.Full:
            self._drop("queue_full")
        except Exception:
            self._drop("error")

    def _drop(self, reason: str) -> None:
        self.dropped += 1
        try:
            if self.metrics is not None:
                self.metrics.hook_dropped.labels(self.name, reason).inc()
        except Exception:
            pass

    # ------------------------------------------------------------ background worker
    def _run(self) -> None:
        client = self._client
        if client is None and self._factory is not None:
            try:
                client = self._factory()
            except Exception as e:  # bad keys/host/SDK: disable quietly, keep serving
                self.disabled_reason = f"client init failed: {type(e).__name__}: {e}"[:300]
                log.warning("Langfuse disabled: %s", self.disabled_reason)
        self._client = client
        since_flush = 0
        while True:
            item = self._q.get()
            if item is _STOP:
                break
            if client is None:
                self.failed += 1
                continue
            try:
                send_trace(client, item)
                self.sent += 1
            except Exception as e:
                self.failed += 1
                log.debug("langfuse send failed: %s", e)
            since_flush += 1
            if since_flush >= self.flush_every or self._q.empty():
                since_flush = 0
                _safe_flush(client)
        _safe_flush(client)

    def close(self, timeout: float = 2.0) -> None:
        try:
            if self._thread.is_alive():
                self._q.put(_STOP, timeout=timeout)
                self._thread.join(timeout)
        except Exception:
            pass


def _safe_flush(client: Any) -> None:
    try:
        if client is not None and hasattr(client, "flush"):
            client.flush()
    except Exception:
        pass


def _io(rec: Any) -> tuple[dict, dict]:
    return {"query": rec.query, "context_docs": rec.context_docs_in}, {"response": rec.response_text}


def send_trace(client: Any, rec: Any) -> None:
    """Ship one TraceRecord as a trace (+ a generation when the upstream model was called). Supports the
    OTel-based Langfuse SDK v3 (`start_observation` / `start_span`) and the legacy v2 (`trace`)."""
    attrs = otel_attributes(rec)
    tags = [f"mode:{rec.mode}", f"cache:{rec.cache_status}", f"route:{rec.model_used or 'cache'}",
            f"config:{rec.config_hash}"]
    inp, out = _io(rec)
    usage = {"input": max(0, int(rec.input_tokens_sent) - int(rec.cached_input_tokens)),
             "output": int(rec.output_tokens), "cache_read_input_tokens": int(rec.cached_input_tokens)}
    trace_meta = {"costguard.saved_usd": rec.saved_usd, "costguard.baseline_cost_usd": rec.baseline_cost_usd,
                  "costguard.config_hash": rec.config_hash}

    if hasattr(client, "start_observation") or hasattr(client, "start_span"):          # SDK v3+
        if hasattr(client, "start_observation"):
            root = client.start_observation(name="costguard.chat", as_type="span", input=inp, output=out,
                                            metadata=attrs)
        else:
            root = client.start_span(name="costguard.chat", input=inp, output=out, metadata=attrs)
        _set_otel(root, attrs)
        try:
            root.update_trace(name="costguard.chat", user_id=rec.tenant, session_id=rec.arm, tags=tags,
                              metadata=trace_meta)
        except Exception:
            pass
        if rec.model_used:
            kw = dict(name="upstream", model=rec.model_id, input=inp, output=out, usage_details=usage,
                      cost_details={"total": float(rec.cost_usd)}, metadata=attrs)
            gen = (root.start_observation(as_type="generation", **kw) if hasattr(root, "start_observation")
                   else root.start_generation(**kw))
            _set_otel(gen, attrs)
            gen.end()
        root.end()
        return

    if hasattr(client, "trace"):                                                         # SDK v2
        start = datetime.fromtimestamp(rec.ts, tz=timezone.utc)
        end = datetime.fromtimestamp(rec.ts + rec.latency_ms / 1000, tz=timezone.utc)
        t = client.trace(id=rec.request_id, name="costguard.chat", user_id=rec.tenant, session_id=rec.arm,
                         tags=tags, input=inp, output=out, metadata=attrs, timestamp=start)
        if rec.model_used:
            t.generation(name="upstream", model=rec.model_id, start_time=start, end_time=end, input=inp, output=out,
                         usage={"input": usage["input"], "output": usage["output"], "unit": "TOKENS",
                                "total_cost": float(rec.cost_usd)}, metadata=attrs)
        return
    raise RuntimeError("unsupported langfuse client (no start_observation/start_span/trace)")


def _set_otel(obs: Any, attrs: dict) -> None:
    """Langfuse v3 observations wrap an OTel span; also set the attributes natively (best effort)."""
    span = getattr(obs, "_otel_span", None)
    if span is None:
        return
    try:
        for k, v in attrs.items():
            span.set_attribute(k, v)
    except Exception:
        pass


def build_langfuse_hook(settings: Any = None, policy: Any = None, metrics: Any = None) -> Optional[LangfuseHook]:
    """Returns None (silently degrading) when keys are missing, the SDK isn't installed or sampling is 0."""
    if not langfuse_configured():
        return None
    try:
        sample = float(os.environ.get("COSTGUARD_LANGFUSE_SAMPLE", "1.0"))
    except ValueError:
        sample = 1.0
    if sample <= 0:
        return None
    try:
        import langfuse  # noqa: F401
        from langfuse import Langfuse
    except Exception as e:  # ImportError, or a broken install
        log.info("Langfuse keys set but the `langfuse` package is unavailable (%s); tracing disabled. "
                 "`pip install langfuse` to enable.", type(e).__name__)
        return None

    def factory() -> Any:  # runs in the worker thread: SDK init never blocks startup or requests
        kw = {"public_key": os.environ["LANGFUSE_PUBLIC_KEY"], "secret_key": os.environ["LANGFUSE_SECRET_KEY"]}
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
        if host:
            kw["host"] = host
        return Langfuse(**kw)

    return LangfuseHook(client_factory=factory, sample_rate=sample, metrics=metrics)
