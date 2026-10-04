"""Fallback token counting (o200k_base). Providers count with their own tokenizer when they can; billing always
uses the provider's returned usage. This counter is for pre-call estimates (routing, budgets, baselines)."""
from __future__ import annotations

from functools import lru_cache

import tiktoken

from .schemas import ChatMessage


@lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("o200k_base")


def count_text(text: str) -> int:
    return len(_enc().encode(text or ""))


def count_messages(messages: list[ChatMessage]) -> int:
    # ~4 tokens of chat-format overhead per message, as in OpenAI's accounting guidance
    return sum(count_text(m.content) + 4 for m in messages) + 2


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    ids = _enc().encode(text or "")
    return text if len(ids) <= max_tokens else _enc().decode(ids[:max_tokens])
