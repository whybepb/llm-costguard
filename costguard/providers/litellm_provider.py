"""Hosted APIs (OpenAI, Gemini, Anthropic, Groq...) through the LiteLLM SDK. Needs the provider's API key in env."""
from __future__ import annotations

import time

from ..schemas import ChatMessage, Completion, Usage


class LiteLLMProvider:
    def __init__(self, name: str = "litellm", timeout_s: float = 60.0):
        self.name = name
        self.timeout_s = timeout_s

    def count_tokens(self, messages: list[ChatMessage], model: str) -> int:
        import litellm
        return int(litellm.token_counter(model=model, messages=[m.model_dump() for m in messages]))

    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        import litellm
        t0 = time.perf_counter()
        r = litellm.completion(model=model, messages=[m.model_dump() for m in messages], max_tokens=max_tokens,
                               temperature=temperature, timeout=self.timeout_s, seed=7)
        dt = (time.perf_counter() - t0) * 1000
        u = r.usage
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        usage = Usage(input_tokens=int(u.prompt_tokens), output_tokens=int(u.completion_tokens), cached_input_tokens=cached)
        choice = r.choices[0]
        return Completion(text=choice.message.content or "", model=model, usage=usage, latency_ms=dt,
                          finish_reason=choice.finish_reason or "stop",
                          raw={"system_fingerprint": getattr(r, "system_fingerprint", None)})
