"""Native Anthropic adapter (official `anthropic` SDK) — the real-API backend.

Key: COSTGUARD_ANTHROPIC_API_KEY (preferred) or ANTHROPIC_API_KEY, from the environment or the repo's .env file.
Endpoint: always https://api.anthropic.com unless COSTGUARD_ANTHROPIC_BASE_URL is set. We deliberately do NOT honour
ANTHROPIC_BASE_URL, which other tools (e.g. coding agents) set for their own proxies — the SDK would otherwise read
it silently and send your key there.

Usage mapping: Anthropic's `input_tokens` excludes cache reads/writes, so
  total input = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
and PriceBook bills cache reads at the cached price and cache writes at the cache-write price.

Token counting before a call (used only for routing/baseline estimates — billing always uses returned usage) is a
local o200k estimate scaled by a ratio learnt from real responses, so the hot path makes no extra API call.
Set COSTGUARD_ANTHROPIC_EXACT_COUNT=1 to use the (free, rate-limited) count_tokens endpoint instead.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time

from ..schemas import ChatMessage, Completion, Usage
from ..tokens import count_messages

DEFAULT_BASE_URL = "https://api.anthropic.com"


def _api_key() -> str | None:
    return os.environ.get("COSTGUARD_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, client=None, timeout_s: float = 60.0, max_retries: int = 4):
        if client is None:
            from anthropic import Anthropic
            key = _api_key()
            if not key:
                raise RuntimeError("No Anthropic key: set COSTGUARD_ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY) in .env")
            client = Anthropic(api_key=key, base_url=os.environ.get("COSTGUARD_ANTHROPIC_BASE_URL", DEFAULT_BASE_URL),
                               timeout=timeout_s, max_retries=max_retries)
        self.client = client
        self.cache_system = os.environ.get("COSTGUARD_ANTHROPIC_PROMPT_CACHE", "1") == "1"
        self.exact_count = os.environ.get("COSTGUARD_ANTHROPIC_EXACT_COUNT", "0") == "1"
        self._ratio = 1.15            # Anthropic tokens per o200k token; learnt online from real usage
        self._lock = threading.Lock()
        self._count_cache: dict[str, int] = {}

    # ------------------------------------------------------------------ helpers
    def _split(self, messages: list[ChatMessage]):
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        msgs = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        if self.cache_system and system:
            # cache breakpoint on the static prefix; silently ignored below the model's minimum cacheable length
            sys_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            sys_param = system or None
        return sys_param, msgs

    def count_tokens(self, messages: list[ChatMessage], model: str) -> int:
        if self.exact_count:
            key = hashlib.sha256((model + repr([(m.role, m.content) for m in messages])).encode()).hexdigest()
            if key not in self._count_cache:
                sys_param, msgs = self._split(messages)
                kw = {"system": sys_param} if sys_param else {}
                self._count_cache[key] = int(self.client.messages.count_tokens(model=model, messages=msgs, **kw).input_tokens)
            return self._count_cache[key]
        return int(round(count_messages(messages) * self._ratio))

    # ------------------------------------------------------------------ call
    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        sys_param, msgs = self._split(messages)
        kw = {"system": sys_param} if sys_param else {}
        t0 = time.perf_counter()
        r = self.client.messages.create(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature, **kw)
        dt = (time.perf_counter() - t0) * 1000
        u = r.usage
        read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        total_in = int(u.input_tokens) + read + write
        est = count_messages(messages)
        if est > 0 and total_in > 0:          # learn the tokenizer ratio for pre-call estimates
            with self._lock:
                self._ratio = 0.8 * self._ratio + 0.2 * (total_in / est)
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        usage = Usage(input_tokens=total_in, output_tokens=int(u.output_tokens), cached_input_tokens=read,
                      cache_write_tokens=write)
        return Completion(text=text, model=model, usage=usage, latency_ms=dt, finish_reason=r.stop_reason or "stop",
                          raw={"id": r.id, "stop_reason": r.stop_reason})
