"""The CostGuard engine. One request flows through the stages in order of quality risk (safest first):

  1 exact cache -> 2 semantic cache -> 3 context rerank/trim -> 4 compression -> 5 model router -> 6 upstream call
  -> 7 cache write-back -> 8 one TraceRecord (cost, baseline cost, savings, latency breakdown)

Every optional stage fails open: if it raises, the request continues un-optimised and the error is logged.
The system prompt is never modified (keeps provider prefix-caching intact); only retrieved context is trimmed/compressed.
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from typing import Callable, Optional

from .config import Policy, Settings
from .interfaces import (AlwaysRequestedRouter, NoCompressor, NoExactCache, NoSemanticCache, PassthroughContext)
from .pricing import PriceBook
from .schemas import (CacheEntry, ChatMessage, ChatRequest, Completion, RouteDecision, RouteInput, TraceRecord)
from .tokens import count_text

TIERS = ("strong", "cheap")


def _h(s: str, n: int = 12) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.! ")


def format_docs(docs: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {d.strip()}" for i, d in enumerate(docs))


def build_messages(system: str, history: list[ChatMessage], query: str, context_block: str) -> list[ChatMessage]:
    user = query if not context_block else f"Store policy context:\n{context_block}\n\nCustomer question: {query}"
    return [ChatMessage(role="system", content=system), *history, ChatMessage(role="user", content=user)]


class CostGuard:
    def __init__(self, policy: Policy, settings: Settings, provider, prices: PriceBook, *, exact_cache=None,
                 semantic_cache=None, context_optimizer=None, compressor=None, router=None,
                 hooks: Optional[list[Callable[[TraceRecord], None]]] = None):
        self.policy, self.settings, self.provider, self.prices = policy, settings, provider, prices
        self.exact = exact_cache or NoExactCache()
        self.semantic = semantic_cache or NoSemanticCache()
        self.context = context_optimizer or PassthroughContext()
        self.compressor = compressor or NoCompressor()
        self.router = router or AlwaysRequestedRouter()
        self.hooks = hooks or []

    # ------------------------------------------------------------------ helpers
    @property
    def backend(self) -> str:
        return self.settings.backend

    def model_id(self, alias: str) -> str:
        return self.policy.model_id(self.backend, alias)

    def reset_caches(self) -> None:
        self.exact.clear()
        self.semantic.clear()

    def _emit(self, rec: TraceRecord) -> None:
        for h in self.hooks:
            try:
                h(rec)
            except Exception:  # observability must never break serving
                pass

    # ------------------------------------------------------------------ main entry
    def handle(self, req: ChatRequest) -> tuple[Completion, TraceRecord]:
        t0 = time.perf_counter()
        opts = req.costguard
        mode_name, mp = self.policy.mode(opts.mode)
        requested = req.model if req.model in TIERS else "strong"
        max_tokens = req.max_tokens or self.policy.default_max_tokens

        sys_msgs = [m for m in req.messages if m.role == "system"]
        system = sys_msgs[0].content if sys_msgs else self.policy.system_prompt
        convo = [m for m in req.messages if m.role != "system"]
        if not convo or convo[-1].role != "user":
            raise ValueError("the last message must have role 'user'")
        query, history, docs = convo[-1].content, convo[:-1], list(opts.context)

        rec = TraceRecord(request_id=uuid.uuid4().hex, ts=time.time(), arm=opts.arm, trace_pos=opts.trace_pos,
                          item_id=opts.item_id, tenant=opts.tenant, mode=mode_name, category=opts.category,
                          config_hash=self.policy.config_hash, provider=self.provider.name,
                          model_requested=requested, query=query, context_docs_in=len(docs))
        stage = rec.stage_ms

        def timed(name: str, fn, default=None):
            s = time.perf_counter()
            try:
                return fn()
            except Exception as e:  # fail open
                rec.stage_errors[name] = f"{type(e).__name__}: {e}"[:300]
                return default
            finally:
                stage[name] = round((time.perf_counter() - s) * 1000, 3)

        orig_block = format_docs(docs)
        orig_msgs = build_messages(system, history, query, orig_block)
        strong_id = self.model_id("strong")
        rec.input_tokens_original = timed("count", lambda: self.provider.count_tokens(orig_msgs, strong_id), 0) or 0

        ccfg = self.policy.cache
        cacheable = (not opts.no_cache and req.temperature <= float(ccfg.get("max_temperature", 0.3))
                     and not (ccfg.get("single_turn_only", True) and history))
        partition = f"{opts.tenant}|{_h(system, 8)}|{self.policy.kb_version}|{'ctx' if docs else 'noctx'}"
        exact_key = _h(partition + "\x00" + normalize_query(query) + "\x00" + _h(orig_block), 32)
        caching_on = mp.exact_cache or mp.semantic_cache
        rec.cache_status = "miss" if (caching_on and cacheable) else ("bypass" if caching_on else "disabled")

        # 1. exact cache
        if mp.exact_cache and cacheable:
            entry = timed("exact_cache", lambda: self.exact.get(exact_key))
            if entry is not None and self._entry_ok(entry, mode_name):
                return self._serve_cached(rec, entry, "exact", t0)

        # 2. semantic cache
        if mp.semantic_cache and cacheable:
            hit = timed("semantic_cache", lambda: self.semantic.lookup(query, partition, mp.tau))
            if hit is not None:
                rec.cache_similarity, rec.cache_neighbor, rec.cache_guard = hit.similarity, hit.neighbor_query, hit.guard_rejected
                if hit.entry is not None:
                    if self._entry_ok(hit.entry, mode_name):
                        return self._serve_cached(rec, hit.entry, "semantic", t0)
                    rec.cache_guard = "tier_mismatch"

        # 3. context rerank / trim
        sent_docs = docs
        if docs and mp.context:
            res = timed("context", lambda: self.context.optimize(query, docs, mp.context_budget_tokens, mp.context_min_score))
            if res is not None:
                sent_docs = res.docs
        rec.context_docs_kept = len(sent_docs)
        block = format_docs(sent_docs)

        # 4. compression (retrieved context only; never the system prompt or the question)
        if block and mp.compression and count_text(block) >= mp.compression_min_tokens:
            cr = timed("compression", lambda: self.compressor.compress(block, mp.compression_rate, query))
            if cr is not None and cr.text.strip():
                rec.compression_ratio = round(cr.tokens_before / max(1, cr.tokens_after), 3)
                block = cr.text

        # 5. route
        decision = RouteDecision(alias=requested, reason="router-off")
        if mp.router:
            rin = RouteInput(query=query, category=opts.category, input_tokens=rec.input_tokens_original,
                             has_context=bool(docs), history_turns=len(history), requested_alias=requested)
            d = timed("router", lambda: self.router.route(rin, mp.router_policy))
            if d is not None and d.alias in TIERS:
                decision = d
        rec.model_used, rec.route_reason = decision.alias, decision.reason

        # 6. upstream call (cheap-tier failure falls back to strong once)
        sent = build_messages(system, history, query, block)
        rec.model_id = self.model_id(decision.alias)
        s = time.perf_counter()
        try:
            comp = self.provider.complete(sent, rec.model_id, max_tokens, req.temperature)
        except Exception as e:
            if decision.alias == "strong":
                rec.error = f"{type(e).__name__}: {e}"[:300]
                rec.latency_ms = (time.perf_counter() - t0) * 1000
                self._emit(rec)
                raise
            rec.stage_errors["upstream_cheap"] = f"{type(e).__name__}: {e}"[:300]
            rec.model_used, rec.route_reason, rec.model_id = "strong", "fallback-after-cheap-error", strong_id
            comp = self.provider.complete(sent, strong_id, max_tokens, req.temperature)
        stage["upstream"] = round((time.perf_counter() - s) * 1000, 3)
        rec.upstream_latency_ms = comp.latency_ms

        # 7. write-back
        if cacheable and caching_on and comp.text.strip() and comp.finish_reason != "error":
            entry = CacheEntry(entry_id=rec.request_id, query_text=query, response_text=comp.text,
                               model_alias=rec.model_used, input_tokens=rec.input_tokens_original,
                               output_tokens=comp.usage.output_tokens, metadata={"partition": partition})
            if mp.exact_cache:
                timed("exact_write", lambda: self.exact.put(exact_key, entry))
            if mp.semantic_cache:
                timed("semantic_write", lambda: self.semantic.insert(query, partition, entry))

        # 8. cost + record
        u = comp.usage
        rec.input_tokens_sent, rec.output_tokens, rec.cached_input_tokens = u.input_tokens, u.output_tokens, u.cached_input_tokens
        rec.cost_usd = self.prices.cost(rec.model_used, u.input_tokens, u.output_tokens, u.cached_input_tokens) + rec.overhead_cost_usd
        rec.baseline_cost_usd = self.prices.cost("strong", rec.input_tokens_original or u.input_tokens, u.output_tokens)
        rec.saved_usd = rec.baseline_cost_usd - rec.cost_usd
        rec.response_text = comp.text
        rec.latency_ms = (time.perf_counter() - t0) * 1000
        rec.overhead_ms = max(0.0, rec.latency_ms - stage.get("upstream", 0.0))
        self._emit(rec)
        return comp, rec

    # ------------------------------------------------------------------ cache serving
    @staticmethod
    def _entry_ok(entry: CacheEntry, mode_name: str) -> bool:
        # quality mode never serves an answer that a cheaper tier produced
        return not (mode_name == "quality" and entry.model_alias != "strong")

    def _serve_cached(self, rec: TraceRecord, entry: CacheEntry, kind: str, t0: float) -> tuple[Completion, TraceRecord]:
        from .schemas import Usage
        rec.cache_status = kind
        rec.model_used, rec.model_id, rec.route_reason = "", "", f"{kind}-cache-hit"
        rec.cache_neighbor = rec.cache_neighbor or entry.query_text
        rec.input_tokens_sent = rec.output_tokens = 0
        rec.cost_usd = rec.overhead_cost_usd
        rec.baseline_cost_usd = self.prices.cost("strong", rec.input_tokens_original or entry.input_tokens, entry.output_tokens)
        rec.saved_usd = rec.baseline_cost_usd - rec.cost_usd
        rec.response_text = entry.response_text
        rec.latency_ms = (time.perf_counter() - t0) * 1000
        rec.overhead_ms = rec.latency_ms
        comp = Completion(text=entry.response_text, model=f"cache:{entry.model_alias}", usage=Usage(),
                          latency_ms=rec.latency_ms, raw={"cache": kind, "entry_id": entry.entry_id})
        self._emit(rec)
        return comp, rec
