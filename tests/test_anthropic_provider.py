"""Anthropic adapter tests with a fake client (no network, no key)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from costguard.config import Settings, load_policy
from costguard.pricing import PriceBook
from costguard.providers.anthropic_provider import DEFAULT_BASE_URL, AnthropicProvider
from costguard.schemas import ChatMessage


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return NS(id="msg_1", stop_reason="end_turn", content=[NS(type="text", text="Returns are accepted within 30 days.")],
                  usage=NS(input_tokens=40, output_tokens=12, cache_read_input_tokens=100, cache_creation_input_tokens=0))


def test_usage_mapping_and_system_cache_breakpoint():
    fake = NS(messages=FakeMessages())
    p = AnthropicProvider(client=fake)
    msgs = [ChatMessage(role="system", content="You are ShopNest support."), ChatMessage(role="user", content="Return window?")]
    c = p.complete(msgs, "claude-haiku-4-5-20251001", 128, 0.0)
    call = fake.messages.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"] == [{"role": "user", "content": "Return window?"}]
    assert c.usage.input_tokens == 140 and c.usage.cached_input_tokens == 100 and c.usage.output_tokens == 12
    assert c.text.startswith("Returns")


def test_cost_prices_cache_reads_and_writes():
    s = Settings(backend="anthropic")
    pol = load_policy(s.policy_path)
    pb = PriceBook(s.prices_path, pol.billing_for("anthropic"))
    full = pb.cost("strong", 1000, 100)                       # sonnet 5.5: 1000*2 + 100*10 per 1M
    assert full == pytest.approx((1000 * 2 + 100 * 10) / 1e6)
    cached = pb.cost("strong", 1000, 100, cached_input_tokens=800)
    assert cached == pytest.approx((200 * 2 + 800 * 0.2 + 100 * 10) / 1e6)
    written = pb.cost("strong", 1000, 100, cache_write_tokens=800)
    assert written == pytest.approx((200 * 2 + 800 * 2.5 + 100 * 10) / 1e6)
    assert pb.cost("cheap", 1000, 100) == pytest.approx((1000 * 1 + 100 * 5) / 1e6)   # haiku 4.5


def test_ignores_inherited_anthropic_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://some-other-tools-proxy.local")
    monkeypatch.setenv("COSTGUARD_ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("COSTGUARD_ANTHROPIC_BASE_URL", raising=False)
    p = AnthropicProvider()
    assert str(p.client.base_url).rstrip("/") == DEFAULT_BASE_URL


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("COSTGUARD_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No Anthropic key"):
        AnthropicProvider()
