"""Context optimiser (rerank / dynamic-k / budget), compressors, the ShopNest KB, and an engine integration test.

Unit tests use a deterministic fake scorer. The KB-retrieval and engine tests load the small fastembed ONNX models
(bge-small, ms-marco-MiniLM-L-6-v2, cached under models/fastembed). LLMLingua-2 is opt-in: RUN_SLOW=1.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from costguard.config import Settings, load_policy
from costguard.context.compress import (HeuristicCompressor, LLMLingua2Compressor, PassthroughCompressor,
                                        build_compressor, split_docs, structured_kind)
from costguard.context.optimizer import (LexicalScorer, RerankContextOptimizer, Scorer, build_context_optimizer,
                                         edge_order)
from costguard.pipeline import CostGuard, format_docs
from costguard.pricing import PriceBook
from costguard.providers.mock import MockProvider
from costguard.schemas import ChatMessage, ChatRequest, CostGuardOptions
from costguard.tokens import count_text
from eval.kb import fact_present, facts_present, kb_questions, load_kb, retrieve

slow = pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="LLMLingua-2 model load; set RUN_SLOW=1")


class FixedScorer(Scorer):
    kind = "cross-encoder"

    def __init__(self, scores):
        super().__init__("fixed")
        self.scores = list(scores)

    def score(self, query, docs):
        return self.scores[: len(docs)]


class BrokenScorer(Scorer):
    kind = "cross-encoder"

    def __init__(self):
        super().__init__("broken")

    def score(self, query, docs):
        raise RuntimeError("model download failed")


def make_docs(n=8, words=25):
    return [f"Policy {i}. " + " ".join(f"term{i}x{j}" for j in range(words)) + "." for i in range(n)]


def kb_chunk(cid: str) -> str:
    return {c["id"]: c["text"] for c in load_kb()}[cid]


EIGHT_IDS = ["returns-policy#0", "size-guide#0", "privacy-policy#0", "gift-cards#0", "installation-services#0",
             "bulk-business-orders#0", "contact-escalation#0", "refunds#0"]


# ----------------------------------------------------------------------------------------------- optimiser

def test_budget_respected_with_whole_docs_only():
    docs = make_docs()
    opt = RerankContextOptimizer(scorer=FixedScorer([5, 9, 1, 7, 3, 8, 2, 6]), gap=100)
    res = opt.optimize("q", docs, budget_tokens=500, min_score=None)
    assert count_text(format_docs(res.docs)) <= 500 and res.tokens_after <= 500
    assert 1 < len(res.docs) < len(docs)
    assert [docs[i] for i in res.kept_indices] == res.docs          # every kept doc is an untouched input doc
    assert res.tokens_before == count_text(format_docs(docs)) and res.tokens_after == count_text(format_docs(res.docs))
    assert opt.last_ms > 0


def test_budget_drops_lowest_scores_first():
    docs = make_docs()
    res = RerankContextOptimizer(scorer=FixedScorer([5, 9, 1, 7, 3, 8, 2, 6]), gap=100, order="score").optimize(
        "q", docs, 500, None)
    assert res.kept_indices == [1, 5, 3, 7][: len(res.kept_indices)]
    assert res.scores == sorted(res.scores, reverse=True)


def test_top1_always_kept_even_when_it_alone_exceeds_budget_or_floor():
    docs = make_docs()
    opt = RerankContextOptimizer(scorer=FixedScorer([5, 9, 1, 7, 3, 8, 2, 6]))
    assert opt.optimize("q", docs, budget_tokens=5, min_score=None).kept_indices == [1]
    assert opt.optimize("q", docs, budget_tokens=None, min_score=999.0).kept_indices == [1]


def test_dynamic_k_relative_gap_and_absolute_floor():
    docs = make_docs(6)
    sc = FixedScorer([9.0, 8.5, 1.0, 0.0, -5.0, -9.0])
    assert sorted(RerankContextOptimizer(scorer=sc, gap=2).optimize("q", docs, None, None).kept_indices) == [0, 1]
    assert sorted(RerankContextOptimizer(scorer=sc, gap=2).optimize("q", docs, None, 0.5).kept_indices) == [0, 1, 2]


def test_lost_in_the_middle_order():
    assert edge_order([7, 3, 5, 1]) == [7, 5, 1, 3]
    assert edge_order([2, 4]) == [2, 4]
    res = RerankContextOptimizer(scorer=FixedScorer([1, 9, 5, 7]), gap=100).optimize("q", make_docs(4), None, None)
    assert res.kept_indices[0] == 1 and res.kept_indices[-1] == 3     # best first, second-best last


def test_near_duplicate_docs_are_dropped():
    docs = make_docs(4)
    docs[2] = docs[0].replace("term0x5 ", "term0x5 extra ")             # same chunk retrieved twice, one word apart
    res = RerankContextOptimizer(scorer=FixedScorer([9, 1, 8, 2]), gap=100).optimize("q", docs, None, None)
    assert 2 not in res.kept_indices and 0 in res.kept_indices and "dup 1" in res.note


def test_scorer_failure_falls_back_and_says_so():
    opt = RerankContextOptimizer(scorer=BrokenScorer(), fallbacks=[LexicalScorer])
    docs = ["Electronics can be returned within 10 days.", "We ship to Nepal.", "Gift cards last 12 months."]
    res = opt.optimize("return electronics days", docs, None, None)
    assert "FALLBACK" in res.note and "lexical" in res.note and res.docs[0] == docs[0]


def test_empty_context():
    assert RerankContextOptimizer(scorer=FixedScorer([])).optimize("q", [], 100, None).docs == []


# ----------------------------------------------------------------------------------------------- heuristic compressor

def test_dedup_and_boilerplate_removed_but_different_numbers_kept():
    block = format_docs([
        "UPI refunds take 1–3 business days. For more information, visit the Help Centre. We value your business.",
        "UPI refunds take 1–3 business days. Card refunds take 5–7 business days. Thank you for shopping with us.",
        "UPI refunds take 3–5 business days in rare bank outages.",
    ])
    r = HeuristicCompressor().compress(block, 0.95, "How long do UPI refunds take?")
    assert r.text.count("UPI refunds take 1–3 business days.") == 1
    assert "3–5 business days" in r.text                 # near-duplicate with a different number is a different fact
    for boiler in ("For more information", "We value your business", "Thank you for shopping"):
        assert boiler not in r.text
    assert r.tokens_after < r.tokens_before and r.method.startswith("heuristic")


@pytest.mark.parametrize("rate", [0.33, 0.5, 0.7])
def test_heuristic_hits_target_rate_on_kb_context(rate):
    block = format_docs([kb_chunk(i) for i in EIGHT_IDS])
    r = HeuristicCompressor().compress(block, rate, "What's the return window for a laptop?")
    assert r.tokens_before == count_text(block) and r.tokens_after == count_text(r.text)
    assert abs(r.tokens_after / r.tokens_before - rate) <= 0.1


def test_heuristic_keeps_query_relevant_numbers_and_markers():
    block = format_docs([kb_chunk(i) for i in EIGHT_IDS])
    hc = HeuristicCompressor()
    r = hc.compress(block, 0.2, "How many days do I have to return a laptop?")
    assert "10 days" in r.text
    assert all(re.match(r"^\[\d+\] ", d) for d in r.text.split("\n\n"))       # [n] markers survive
    r2 = hc.compress(block, 0.2, "How long does a UPI refund take?")
    assert fact_present("UPI 1-3 business days", r2.text)                        # the table row, with its header
    assert "Refund timeline" in r2.text


def test_structured_content_passes_through_verbatim():
    js = json.dumps({"order_id": "SN-20461183", "refund": 1499, "status": "initiated"})
    block = format_docs([js, kb_chunk("size-guide#0"), kb_chunk("privacy-policy#0")])
    assert structured_kind(js) == "json" and structured_kind("```py\nx = 1\n```") == "code"
    r = HeuristicCompressor().compress(block, 0.3, "Where is my refund?")
    assert js in r.text


def test_split_docs_round_trip():
    docs = ["alpha\nbeta", "gamma"]
    assert split_docs(format_docs(docs)) == [("[1] ", "alpha\nbeta"), ("[2] ", "gamma")]
    assert split_docs("no markers") == [("", "no markers")]


def test_build_compressor_from_env(monkeypatch):
    s = Settings()
    monkeypatch.delenv("COSTGUARD_COMPRESSOR", raising=False)
    assert isinstance(build_compressor(s, None), HeuristicCompressor)
    monkeypatch.setenv("COSTGUARD_COMPRESSOR", "none")
    c = build_compressor(s, None)
    assert isinstance(c, PassthroughCompressor) and c.compress("a b c", 0.5).tokens_after == count_text("a b c")
    monkeypatch.setenv("COSTGUARD_COMPRESSOR", "llmlingua2")
    ll = build_compressor(s, None)
    assert isinstance(ll, LLMLingua2Compressor) and ll._pc is None          # lazy: nothing loaded yet


@slow
def test_llmlingua2_compresses_and_keeps_markers():
    block = format_docs([kb_chunk(i) for i in EIGHT_IDS])
    ll = LLMLingua2Compressor()
    r = ll.compress(block, 0.5, "What's the return window for a laptop?")
    assert r.method == "llmlingua2" and r.tokens_after < 0.75 * r.tokens_before
    assert [m for m, _ in split_docs(r.text)] == [f"[{i}] " for i in range(1, 9)]
    assert ll.load_ms and ll.last_ms > 0


# ----------------------------------------------------------------------------------------------- KB

def test_kb_chunks_and_seed_questions():
    chunks = load_kb()
    ids = [c["id"] for c in chunks]
    assert len(chunks) >= 40 and len(set(ids)) == len(ids)
    assert not any(c["doc_id"].lower() == "readme" for c in chunks)
    toks = [count_text(c["text"]) for c in chunks]
    assert min(toks) >= 80 and max(toks) <= 330
    docs = {}
    for c in chunks:
        docs[c["doc_id"]] = docs.get(c["doc_id"], "") + "\n" + c["text"]
    assert len(docs) >= 20
    qs = kb_questions()
    assert 55 <= len(qs) <= 70 and all(q["author"] == "seed" for q in qs)
    assert {q["type"] for q in qs} == {"single", "multi_hop", "unanswerable"}
    for q in qs:
        assert set(q["doc_ids"]) <= set(docs)
        for f in q["key_facts"]:
            assert any(fact_present(f, docs[d]) for d in q["doc_ids"]), (q["id"], f)


def test_fact_matching_normalises_rupees_dashes_and_grouping():
    assert fact_present("₹1,499", "costs Rs. 1499 only") and fact_present("1-3 business days", "1–3 business days")
    assert fact_present("₹1,00,000", "up to ₹100000") and not fact_present("10 days", "110 days")
    assert facts_present([], "anything")


def test_retrieve_is_generous_top_k():
    docs = retrieve("How long does a UPI refund take?", k=8)
    assert len(docs) == 8 and any("UPI" in d and "Refund" in d for d in docs[:3])


# ----------------------------------------------------------------------------------------------- engine integration

class CapturingProvider(MockProvider):
    def complete(self, messages, model, max_tokens, temperature):
        self.sent = messages
        return super().complete(messages, model, max_tokens, temperature)


def test_engine_balanced_mode_sends_fewer_docs_and_tokens(tmp_path, monkeypatch):
    monkeypatch.delenv("COSTGUARD_COMPRESSOR", raising=False)
    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    pol = load_policy(s.policy_path)
    prov = CapturingProvider()
    eng = CostGuard(pol, s, prov, PriceBook(s.prices_path, pol.billing),
                    context_optimizer=build_context_optimizer(s, pol), compressor=build_compressor(s, pol))
    q = "What's the return window for a laptop?"
    docs = [kb_chunk(i) for i in EIGHT_IDS]
    _, rec = eng.handle(ChatRequest(messages=[ChatMessage(role="user", content=q)],
                                    costguard=CostGuardOptions(mode="balanced", context=docs)))
    assert not rec.stage_errors
    assert rec.context_docs_in == 8 and rec.context_docs_kept < 8
    assert rec.input_tokens_sent < rec.input_tokens_original
    assert rec.saved_usd > 0 and rec.cost_usd < rec.baseline_cost_usd
    assert prov.sent[0].role == "system" and prov.sent[0].content == pol.system_prompt   # prefix never touched
    user = prov.sent[-1].content
    assert user.endswith(f"Customer question: {q}") and "10 days" in user                # question verbatim, fact kept
