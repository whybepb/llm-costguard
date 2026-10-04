"""Regression tests for review findings #11 (numeric guards drop currency, ID prefixes and number roles) and #16
(near-duplicate removal erases contradictory facts). Pure Python: no embedding or reranker model is loaded.
"""
from __future__ import annotations

import pytest

from costguard.cache import guards as G
from costguard.context.compress import HeuristicCompressor
from costguard.context.optimizer import RerankContextOptimizer, Scorer
from costguard.pipeline import format_docs


class FixedScorer(Scorer):
    kind = "cross-encoder"

    def __init__(self, scores):
        super().__init__("fixed")
        self.scores = list(scores)

    def score(self, query, docs):
        return self.scores[: len(docs)]


# ============================================================================================ #11 numbers guard


@pytest.mark.parametrize("a,b,reason", [
    ("Can I get a refund of $10?", "Can I get a refund of ₹10?", "number_mismatch"),
    ("Can I get a refund of $10?", "Can I get a refund of 10?", "number_mismatch"),        # one-sided currency: strict
    ("What is the status of order AB12345?", "What is the status of order CD12345?", "number_mismatch"),
    ("Where is order 4821A?", "Where is order 4821B?", "number_mismatch"),
    ("Move money from account 111 to 222", "Move money from account 222 to 111", "number_order_mismatch"),
    ("Transfer 500 from account 111 to account 222", "Transfer 500 from account 222 to account 111",
     "number_order_mismatch"),
    ("Move money from 111 to 222", "Move money to 111 from 222", "number_order_mismatch"),  # same order, roles swap
    ("Move 111 to 222", "Move 222 to 111", "number_order_mismatch"),
    ("Cancel orders 4821 and 4822", "Cancel orders 4822 and 4821", "number_order_mismatch"),  # no cue: position
])
def test_numbers_guard_rejects_currency_id_and_role_changes(a, b, reason):
    assert G.check(a, b) == reason
    assert G.check(b, a) == reason          # symmetric


@pytest.mark.parametrize("a,b", [
    ("Can I get a refund of $10?", "can i get a refund of 10 dollars"),
    ("I paid Rs. 2,499 for this item", "I paid ₹2499 for this item"),
    ("Please cancel order SN-48213", "please cancel order sn48213"),
    ("Move money from account 111 to 222", "Move money to account 222 from account 111"),   # role-aware, not order
    ("Transfer 500 from 111 to 222", "Transfer 500 to 222 from 111"),
    ("I want to move 111 to 222", "move 111 to 222 please"),        # infinitive "to" is not a destination
])
def test_numbers_guard_allows_paraphrases(a, b):
    assert G.check(a, b) is None
    assert G.check(b, a) is None


# ============================================================================================ #16 dedup veto


@pytest.mark.parametrize("a,b,reason", [
    # the cache's hedge rule reads "can't" as "unable to"; dedup must not
    ("You can return electronics within 30 days.", "You cannot return electronics within 30 days.",
     "negation_mismatch"),
    ("Plus members pay the reverse-pickup fee.", "Plus members never pay the reverse-pickup fee.", "negation_mismatch"),
    ("Electronics can be returned within 30 days.", "Electronics can be returned within 7 days.", "number_mismatch"),
    ("Laptops and phones can be returned within 10 days.", "Laptops and tablets can be returned within 10 days.",
     "entity_mismatch:product"),                                   # overlapping, not disjoint: still a different fact
])
def test_dedup_veto_flags_different_facts(a, b, reason):
    assert G.dedup_veto(a, b) == reason
    assert G.dedup_veto(b, a) == reason


def test_dedup_veto_allows_rewording():
    assert G.dedup_veto("UPI refunds take 1-3 business days.", "UPI refunds usually take 1-3 business days.") is None


@pytest.mark.parametrize("second", [
    "You cannot return electronics within 30 days of delivery for a full refund.",      # negated
    "You can return furniture within 30 days of delivery for a full refund.",           # different product
])
def test_compression_keeps_contradictory_near_duplicates(second):
    first = "You can return electronics within 30 days of delivery for a full refund."
    hc = HeuristicCompressor()
    r = hc.compress(format_docs([first, second]), 0.95, "Can I return electronics within 30 days?")
    assert first in r.text and second in r.text
    assert hc.last_stats["duplicate"] == 0


def test_compression_still_drops_true_near_duplicates():
    hc = HeuristicCompressor()
    r = hc.compress(format_docs(["UPI refunds take 1-3 business days after we receive the item.",
                                 "UPI refunds usually take 1-3 business days after we receive the item."]),
                    0.95, "How long do UPI refunds take?")
    assert hc.last_stats["duplicate"] == 1 and r.text.count("1-3 business days") == 1


_POLICY = ("Returns policy. Electronics such as laptops, phones, tablets and headphones {verb} returned within {n} "
           "days of delivery if they are unused, in the original packaging, with all accessories, manuals and the "
           "invoice. "
           "Refunds are issued to the original payment method once the item passes inspection at our warehouse.")


@pytest.mark.parametrize("other", [
    _POLICY.format(verb="can be", n=7),                  # 30 vs 7 days
    _POLICY.format(verb="cannot be", n=30),              # negated
])
def test_optimizer_keeps_near_duplicate_docs_with_different_facts(other):
    docs = [_POLICY.format(verb="can be", n=30), other]
    res = RerankContextOptimizer(scorer=FixedScorer([9, 8]), gap=100).optimize("electronics return window", docs)
    assert sorted(res.kept_indices) == [0, 1] and "dup 0" in res.note


def test_optimizer_still_drops_reworded_duplicate_doc():
    doc = _POLICY.format(verb="can be", n=30)
    docs = [doc, doc.replace("Returns policy.", "Returns.")]
    res = RerankContextOptimizer(scorer=FixedScorer([9, 8]), gap=100).optimize("electronics return window", docs)
    assert res.kept_indices == [0] and "dup 1" in res.note
