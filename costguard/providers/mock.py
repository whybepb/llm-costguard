"""Deterministic fake upstream: for unit tests, CI and load tests (measures CostGuard's own overhead, $0)."""
from __future__ import annotations

import hashlib
import time

from ..schemas import ChatMessage, Completion, Usage
from ..tokens import count_messages, count_text

_WORDS = ("please check your order status in your account settings we can help with returns refunds shipping "
          "and payment options our team is available every day thank you for shopping with us").split()


class MockProvider:
    name = "mock"

    def __init__(self, latency_ms: float = 0.0):
        self.latency_ms = latency_ms
        self.calls = 0

    def count_tokens(self, messages: list[ChatMessage], model: str) -> int:
        return count_messages(messages)

    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        self.calls += 1
        t0 = time.perf_counter()
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        query = next((m.content for m in reversed(messages) if m.role == "user"), "")
        h = int(hashlib.sha256((model + query).encode()).hexdigest(), 16)
        n = min(max_tokens, 20 + h % 60)
        body = " ".join(_WORDS[(h >> i) % len(_WORDS)] for i in range(n))
        text = f"[{model}] {body}"
        usage = Usage(input_tokens=self.count_tokens(messages, model), output_tokens=count_text(text))
        return Completion(text=text, model=model, usage=usage, latency_ms=(time.perf_counter() - t0) * 1000)
