from __future__ import annotations

from ..config import Settings
from .cassette import CassetteProvider
from .mock import MockProvider


def make_provider(settings: Settings):
    b = settings.backend
    if b == "mock":
        p = MockProvider(latency_ms=settings.mock_latency_ms)
    elif b == "mlx":
        from .mlx_local import MLXProvider
        p = MLXProvider()
    elif b == "anthropic":
        from .anthropic_provider import AnthropicProvider
        p = AnthropicProvider()
    elif b in ("openai", "gemini", "groq", "litellm"):
        from .litellm_provider import LiteLLMProvider
        p = LiteLLMProvider(name=b)
    else:
        raise ValueError(f"unknown backend {b!r}")
    if settings.cassette:
        p = CassetteProvider(p, settings.cassette, settings.cassette_mode)
    return p
