"""Real generations on Apple silicon via MLX (free). Token counts come from the model's own tokenizer.
Billing still uses the list price of the API model each tier stands in for (configs/policy.yaml -> billing)."""
from __future__ import annotations

import threading
import time

from ..schemas import ChatMessage, Completion, Usage


class MLXProvider:
    name = "mlx"

    def __init__(self):
        self._models: dict[str, tuple] = {}
        self._lock = threading.Lock()      # MLX/Metal: one generation at a time

    def _load(self, repo: str):
        if repo not in self._models:
            from mlx_lm import load
            self._models[repo] = load(repo)
        return self._models[repo]

    def _prompt(self, tok, messages: list[ChatMessage]) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def count_tokens(self, messages: list[ChatMessage], model: str) -> int:
        with self._lock:
            _, tok = self._load(model)
            return len(tok.encode(self._prompt(tok, messages)))

    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        with self._lock:
            m, tok = self._load(model)
            prompt = self._prompt(tok, messages)
            t0 = time.perf_counter()
            text = generate(m, tok, prompt=prompt, max_tokens=max_tokens, sampler=make_sampler(temp=temperature), verbose=False)
            dt = (time.perf_counter() - t0) * 1000
            usage = Usage(input_tokens=len(tok.encode(prompt)), output_tokens=len(tok.encode(text)))
        return Completion(text=text.strip(), model=model, usage=usage, latency_ms=dt)
