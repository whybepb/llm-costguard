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


def test_request_shape_per_model():
    """Sonnet 5.5 rejects non-default sampling parameters (temperature=0 is a 400) and thinks adaptively unless told
    otherwise; Haiku 4.5 accepts temperature and takes no thinking field."""
    fake = NS(messages=FakeMessages())
    p = AnthropicProvider(client=fake)
    msgs = [ChatMessage(role="system", content="You are ShopNest support."), ChatMessage(role="user", content="Return window?")]
    pol = load_policy(Settings(backend="anthropic").policy_path)
    strong, cheap = pol.model_id("anthropic", "strong"), pol.model_id("anthropic", "cheap")
    p.complete(msgs, strong, 256, 0.0)
    p.complete(msgs, cheap, 256, 0.0)
    s, c = fake.messages.calls
    assert strong == "claude-sonnet-5-5" and "temperature" not in s and "extra_body" not in s
    assert s["thinking"] == {"type": "between_tools"}
    assert c["extra_body"] == {"temperature": 0.0} and "thinking" not in c   # SDK 1.x: sent via extra_body


def test_text_is_read_by_block_type():
    fake = NS(messages=NS(create=lambda **kw: NS(
        id="m", stop_reason="end_turn", content=[NS(type="thinking", thinking=""), NS(type="text", text="B")],
        usage=NS(input_tokens=5, output_tokens=1, cache_read_input_tokens=None, cache_creation_input_tokens=None))))
    c = AnthropicProvider(client=fake).complete([ChatMessage(role="user", content="A or B?")], "claude-sonnet-5-5", 5, 0.0)
    assert c.text == "B" and c.usage.input_tokens == 5 and c.usage.cached_input_tokens == 0


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


def test_spend_cap_refuses_calls_once_the_ledger_reaches_the_cap(tmp_path, monkeypatch):
    import costguard.providers.anthropic_provider as ap
    monkeypatch.setattr(ap, "_LEDGERS", {})
    monkeypatch.setenv("COSTGUARD_SPEND_CAP_USD", "0.0012")
    monkeypatch.setenv("COSTGUARD_SPEND_LEDGER", str(tmp_path / "spend.jsonl"))
    fake = NS(messages=FakeMessages())          # each call: 40 in, 12 out, 100 cache-read tokens
    p = AnthropicProvider(client=fake)
    msgs = [ChatMessage(role="user", content="Return window?")]
    for _ in range(2):                           # sonnet: (40*2 + 12*10 + 100*0.2) / 1e6 = $0.00022 per call
        p.complete(msgs, "claude-sonnet-5-5", 64, 0.0)
    assert p.ledger.spent == pytest.approx(2 * 0.00022)
    p.ledger.spent = 0.0012                      # cap reached: the next call is refused before the API is hit
    with pytest.raises(ap.SpendCapExceeded):
        p.complete(msgs, "claude-sonnet-5-5", 64, 0.0)
    assert len(fake.messages.calls) == 2
    monkeypatch.setattr(ap, "_LEDGERS", {})      # a new process re-reads the ledger file
    assert ap.spend_ledger().spent == pytest.approx(2 * 0.00022)
