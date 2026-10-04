"""Observability hooks the engine calls with every finished TraceRecord (costguard/factory.py -> build_hooks).

    metrics   Prometheus counters/histograms (costguard/obs/metrics.py)        always (COSTGUARD_METRICS=0 disables)
    drift     online PSI drift monitor (costguard/obs/drift.py)                 always (COSTGUARD_DRIFT=0 disables)
    langfuse  async Langfuse traces (costguard/obs/langfuse_hook.py)            only if LANGFUSE_PUBLIC_KEY and
                                                                                LANGFUSE_SECRET_KEY are set and the
                                                                                `langfuse` package is installed

Hooks run synchronously at the end of a request, so each is O(1) and in-memory; anything that touches the network
goes through a queue to a background thread. Every hook is wrapped in SafeHook, and build_hooks itself never raises:
observability must never break serving.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

log = logging.getLogger("costguard.obs")


def _on(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "no", "off")


class SafeHook:
    """Wraps a hook so an exception is counted and swallowed, never propagated into the request."""

    def __init__(self, inner: Callable[[Any], None], name: str = "", metrics: Any = None):
        self.inner = inner
        self.name = name or getattr(inner, "name", type(inner).__name__)
        self.metrics = metrics
        self.errors = 0

    def __call__(self, rec: Any) -> None:
        try:
            self.inner(rec)
        except Exception as e:
            self.errors += 1
            log.debug("hook %s raised: %s", self.name, e)
            try:
                if self.metrics is not None:
                    self.metrics.hook_errors.labels(self.name).inc()
            except Exception:
                pass

    def __repr__(self) -> str:
        return f"SafeHook({self.name})"


def build_hooks(settings: Any = None, policy: Any = None) -> list[Callable[[Any], None]]:
    hooks: list[tuple[str, Callable]] = []
    metrics = None
    try:
        from .metrics import MetricsHook, get_default_metrics
        metrics = get_default_metrics()
        if _on("COSTGUARD_METRICS"):
            hooks.append(("metrics", MetricsHook(metrics)))
    except Exception as e:
        log.warning("metrics hook unavailable: %s", e)
    try:
        if _on("COSTGUARD_DRIFT"):
            from .drift import DriftHook, DriftMonitor
            hooks.append(("drift", DriftHook(DriftMonitor.from_env(), metrics=metrics)))
    except Exception as e:
        log.warning("drift hook unavailable: %s", e)
    try:
        from .langfuse_hook import build_langfuse_hook, langfuse_configured
        if langfuse_configured():
            lf = build_langfuse_hook(settings, policy, metrics=metrics)
            if lf is not None:
                hooks.append(("langfuse", lf))
    except Exception as e:
        log.warning("langfuse hook unavailable: %s", e)
    return [SafeHook(h, name, metrics) for name, h in hooks]
