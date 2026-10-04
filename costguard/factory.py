"""Wires the engine from settings + policy. Each stage module exposes a `build_*` function; if a module is missing
or fails to build, the engine falls back to that stage's no-op (and /health reports it)."""
from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

from .config import Settings, load_policy
from .obs.logger import RequestLogger
from .pipeline import CostGuard
from .pricing import PriceBook
from .providers import make_provider

log = logging.getLogger("costguard")

# module path, builder function name (signature: builder(settings, policy) -> component)
COMPONENTS = {
    "exact_cache": ("costguard.cache.exact", "build_exact_cache"),
    "semantic_cache": ("costguard.cache.semantic", "build_semantic_cache"),
    "context_optimizer": ("costguard.context.optimizer", "build_context_optimizer"),
    "compressor": ("costguard.context.compress", "build_compressor"),
    "router": ("costguard.router.router", "build_router"),
}


def _build(name: str, settings: Settings, policy) -> tuple[Optional[Any], str]:
    mod, fn = COMPONENTS[name]
    try:
        builder = getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        return None, f"not available ({e.__class__.__name__})"
    try:
        comp = builder(settings, policy)
        return comp, type(comp).__name__
    except Exception as e:  # keep serving with the no-op stage
        log.warning("component %s failed to build: %s", name, e)
        return None, f"build failed: {e}"


def build_engine(settings: Optional[Settings] = None, *, with_logger: bool = True,
                 skip: tuple[str, ...] = ()) -> CostGuard:
    settings = settings or Settings.from_env()
    policy = load_policy(settings.policy_path)
    prices = PriceBook(settings.prices_path, policy.billing_for(settings.backend))
    provider = make_provider(settings)
    built, status = {}, {}
    for name in COMPONENTS:
        if name in skip:
            built[name], status[name] = None, "skipped"
            continue
        built[name], status[name] = _build(name, settings, policy)
    hooks = []
    if with_logger:
        hooks.append(RequestLogger(settings.db_path))
    try:
        hooks.extend(importlib.import_module("costguard.obs.hooks").build_hooks(settings, policy))
    except (ImportError, AttributeError):
        pass
    engine = CostGuard(policy, settings, provider, prices, hooks=hooks, **built)
    engine.component_status = status  # type: ignore[attr-defined]
    return engine
