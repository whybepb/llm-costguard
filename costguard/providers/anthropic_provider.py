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
import json
import os
import threading
import time

from ..schemas import ChatMessage, Completion, Usage
from ..tokens import count_messages

DEFAULT_BASE_URL = "https://api.anthropic.com"
# Models that reject non-default sampling parameters with a 400 (temperature, top_p, top_k): Claude Sonnet 5 / 5.5,
# Opus 4.7 and later, Fable / Mythos 5. Haiku 4.5 and the 4.6 models still accept temperature.
NO_SAMPLING_PREFIXES = ("claude-sonnet-5", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-fable-5",
                        "claude-mythos-5")
# Claude Sonnet 5.5 runs adaptive thinking unless told otherwise, and thinking tokens count toward max_tokens (256 for
# answers, 5 for the judge). `between_tools` is its lowest setting: no extended thinking ("disabled" is a 400 there).
THINKING_OFF = {"claude-sonnet-5-5": {"type": "between_tools"}}


def accepts_sampling(model: str) -> bool:
    return not model.startswith(NO_SAMPLING_PREFIXES)


# Hard spend cap for real API calls: set COSTGUARD_SPEND_CAP_USD. Every completed call appends its list-price cost to a
# ledger file, and a new call is refused once the ledger total reaches the cap. The ledger is shared by every provider
# in the process and survives across processes (gate, CI recording, A/B), so one cap covers a whole session of runs.
# Run one paid process at a time: a second process only sees the ledger as it was when that process started.
LIST_PRICES = {  # USD per 1M tokens: input, output, cache read, cache write (mirrors configs/prices.yaml)
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
}
UNKNOWN_MODEL_PRICE = (5.00, 25.00, 0.50, 6.25)   # priced high on purpose, so the cap stays conservative


class SpendCapExceeded(RuntimeError):
    pass


class SpendLedger:
    def __init__(self, cap_usd: float, path):
        from pathlib import Path
        self.cap, self.path = float(cap_usd), Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.spent = 0.0
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.spent += float(json.loads(line).get("usd", 0.0))

    @staticmethod
    def cost(model: str, uncached_in: int, out: int, read: int, write: int) -> float:
        p = next((v for k, v in LIST_PRICES.items() if model.startswith(k)), UNKNOWN_MODEL_PRICE)
        return (uncached_in * p[0] + out * p[1] + read * p[2] + write * p[3]) / 1e6

    def check(self) -> None:
        with self._lock:
            if self.spent >= self.cap:
                raise SpendCapExceeded(f"spend cap reached: ${self.spent:.4f} of ${self.cap:.2f} "
                                       f"(ledger {self.path}); raise COSTGUARD_SPEND_CAP_USD to continue")

    def add(self, model: str, uncached_in: int, out: int, read: int, write: int) -> float:
        usd = self.cost(model, uncached_in, out, read, write)
        with self._lock:
            self.spent += usd
            with self.path.open("a") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "model": model, "input": uncached_in, "output": out,
                                    "cache_read": read, "cache_write": write, "usd": round(usd, 8)}) + "\n")
        return usd


_LEDGERS: dict[str, SpendLedger] = {}


def spend_ledger() -> "SpendLedger | None":
    cap = os.environ.get("COSTGUARD_SPEND_CAP_USD")
    if not cap:
        return None
    from ..config import ROOT
    path = os.environ.get("COSTGUARD_SPEND_LEDGER") or str(ROOT / "eval" / "results" / "logs" / "anthropic_spend.jsonl")
    if path not in _LEDGERS:
        _LEDGERS[path] = SpendLedger(float(cap), path)
    return _LEDGERS[path]


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
        self.ledger = spend_ledger()

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
        if accepts_sampling(model):        # on the others even temperature=0 is a 400; they run at the API default
            # anthropic SDK 1.x dropped `temperature` from messages.create(); the API still honours it on these models
            kw["extra_body"] = {"temperature": temperature}
        if model in THINKING_OFF:
            kw["thinking"] = THINKING_OFF[model]
        if self.ledger is not None:
            self.ledger.check()
        t0 = time.perf_counter()
        r = self.client.messages.create(model=model, messages=msgs, max_tokens=max_tokens, **kw)
        dt = (time.perf_counter() - t0) * 1000
        u = r.usage
        read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        total_in = int(u.input_tokens) + read + write
        if self.ledger is not None:
            self.ledger.add(model, int(u.input_tokens), int(u.output_tokens), read, write)
        est = count_messages(messages)
        if est > 0 and total_in > 0:          # learn the tokenizer ratio for pre-call estimates
            with self._lock:
                self._ratio = 0.8 * self._ratio + 0.2 * (total_in / est)
        text = "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "") == "text")
        usage = Usage(input_tokens=total_in, output_tokens=int(u.output_tokens), cached_input_tokens=read,
                      cache_write_tokens=write)
        return Completion(text=text, model=model, usage=usage, latency_ms=dt, finish_reason=r.stop_reason or "stop",
                          raw={"id": r.id, "stop_reason": r.stop_reason})
