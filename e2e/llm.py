"""Memoized Claude calls for the end-to-end evaluation.

Every request is keyed by its full content and saved to disk, so re-running the
evaluation never re-bills a call. Reads ANTHROPIC_API_KEY from the environment.

COSTGUARD_OFFLINE=1 makes a memo miss an error instead of an API call (CI, replay).
COSTGUARD_MAX_NEW_COST_USD caps what new calls in one process may spend.
Sonnet 5.5 calls opt into server-side refusal fallbacks ("default" routing);
Haiku 5.5 has no server-side fallback, so a refusal is returned as-is.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import anthropic

from config import CFG, price

PRICES = {m: price(m) for m in CFG["prices"]}  # $ per 1M input, output tokens
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class MemoMiss(RuntimeError):
    """An offline run needed a call that isn't memoized."""


class BudgetExceeded(RuntimeError):
    pass


def offline():
    return os.environ.get("COSTGUARD_OFFLINE") == "1"


def cost(model, input_tokens, output_tokens):
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6


class LLM:
    def __init__(self, memo_path):
        self.client = None if offline() else anthropic.Anthropic(max_retries=6)
        self.max_new_cost = float(os.environ.get("COSTGUARD_MAX_NEW_COST_USD", "inf"))
        self.new_cost = 0.0
        self.path = Path(memo_path)
        self.memo = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.lock = threading.Lock()
        self.inflight = {}  # key -> Event, so identical concurrent requests make one call
        self.new_calls = 0

    @staticmethod
    def key(model, system, user, max_tokens=CFG["llm"]["max_tokens"], effort=None):
        req = {"model": model, "system": system, "user": user, "max_tokens": max_tokens, "effort": effort}
        return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()

    def get(self, model, system, user, max_tokens=CFG["llm"]["max_tokens"], effort=None):
        """Memoized response or None; never calls the API."""
        return self.memo.get(self.key(model, system, user, max_tokens, effort))

    def call(self, model, system, user, max_tokens=CFG["llm"]["max_tokens"], effort=None):
        key = self.key(model, system, user, max_tokens, effort)
        while True:
            with self.lock:
                if key in self.memo:
                    return self.memo[key]
                waiting = self.inflight.get(key)
                if waiting is None:
                    self.inflight[key] = threading.Event()
                    break
            waiting.wait()
        try:
            return self._call(key, model, system, user, max_tokens, effort)
        finally:
            with self.lock:
                self.inflight.pop(key).set()

    def _call(self, key, model, system, user, max_tokens, effort):
        with self.lock:
            if self.client is None:
                raise MemoMiss(f"offline and not memoized: {model} call ({len(user)} chars of input)")
            if self.new_cost >= self.max_new_cost:
                raise BudgetExceeded(f"new-call spend reached ${self.new_cost:.4f}")
        kwargs = {"model": model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]}
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if model == "claude-sonnet-5-5":
            msg = self.client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            msg = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        out = {"text": text, "stop_reason": msg.stop_reason, "model_used": msg.model,
               "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        with self.lock:
            self.memo[key] = out
            self.new_calls += 1
            self.new_cost += cost(model, out["input_tokens"], out["output_tokens"])
            self.path.write_text(json.dumps(self.memo))
        return out
