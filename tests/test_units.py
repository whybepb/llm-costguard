import os

import pytest

from cache.exact import cache_key
from cache.guards import negation_match, numbers_match
from config import CFG, config_version, prompt


def test_exact_key_ignores_doc_order_but_not_content():
    a = cache_key("Can I return it?", "case", ["doc A", "doc B"])
    assert a == cache_key("Can I return it?", "case", ["doc B", "doc A"])
    assert a != cache_key("Can I return it?", "case", ["doc A", "doc B edited"])
    assert a != cache_key("Can I return it?", "other case", ["doc A", "doc B"])
    assert a != cache_key("can I return it?", "case", ["doc A", "doc B"])  # no normalization


@pytest.mark.parametrize("a,b,same", [
    ("delivered 10 days ago", "delivered 40 days ago", False),
    ("cart is $62", "cart is $42", False),
    ("delivered 10 days ago", "it arrived 10 days ago", True),
    ("code ABCD-1234", "code ABCD-1235", False),
])
def test_numbers_guard(a, b, same):
    assert numbers_match(a, b) is same


@pytest.mark.parametrize("a,b,same", [
    ("Can I return it?", "I can't return it, right?", False),
    ("The status has not changed. Is it on my card?", "The status has not changed. Can I start a return?", True),
    ("Should I contact support?", "I should not contact support yet, right?", False),
])
def test_negation_guard(a, b, same):
    assert negation_match(a, b) is same


def test_prompts_and_config_load():
    assert prompt("answer_system").startswith("You are the customer support assistant")
    assert CFG["models"]["small"] != CFG["models"]["large"]
    assert len(config_version()) == 12


def test_offline_llm_never_calls_api(tmp_path, monkeypatch):
    monkeypatch.setenv("COSTGUARD_OFFLINE", "1")
    from e2e.llm import LLM, MemoMiss
    llm = LLM(tmp_path / "memo.json")
    with pytest.raises(MemoMiss):
        llm.call(CFG["models"]["small"], "system", "never memoized")
