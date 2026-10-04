"""Regression tests for the core review findings (#6 prices in config_hash, #7-#10 cache identity, #12 auth config,
#13 cheap-tier override, #15 fail-open/fail-safe). Mock provider and tiny in-test fakes only."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import costguard.pipeline as pipeline
from costguard.config import Settings, load_policy
from costguard.pipeline import CostGuard
from costguard.pricing import PriceBook
from costguard.providers.mock import MockProvider
from costguard.schemas import (ChatMessage, ChatRequest, ContextResult, CostGuardOptions, RouteDecision, SemanticHit)
from costguard.server import create_app


class DictCache:
    def __init__(self):
        self.d = {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, e):
        self.d[k] = e

    def clear(self):
        self.d.clear()


class FakeSemantic:
    """Any entry in the same partition is a neighbour at a fixed similarity; records the partitions it saw."""

    def __init__(self, sim: float = 1.0):
        self.sim, self.items, self.seen = sim, [], []

    def lookup(self, query, partition, threshold):
        self.seen.append(partition)
        for p, q, e in self.items:
            if p == partition and self.sim >= threshold:
                return SemanticHit(entry=e, similarity=self.sim, neighbor_query=q)
        return SemanticHit()

    def insert(self, query, partition, entry):
        self.items.append((partition, query, entry))

    def clear(self):
        self.items.clear()


class CheapRouter:
    def __init__(self):
        self.calls = 0

    def route(self, inp, policy):
        self.calls += 1
        return RouteDecision(alias="cheap", reason="test")


class FinishProvider(MockProvider):
    def __init__(self, finish: str):
        super().__init__()
        self.finish = finish

    def complete(self, *a, **k):
        c = super().complete(*a, **k)
        c.finish_reason = self.finish
        return c


def make(tmp_path: Path, **kw) -> CostGuard:
    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    pol = load_policy(s.policy_path)
    return CostGuard(pol, s, kw.pop("provider", MockProvider()), PriceBook(s.prices_path, pol.billing), **kw)


def req(q: str, mode: str | None = "balanced", model: str | None = None, max_tokens: int | None = None,
        system: str | None = None, **opts) -> ChatRequest:
    msgs = ([ChatMessage(role="system", content=system)] if system else []) + [ChatMessage(role="user", content=q)]
    return ChatRequest(model=model, max_tokens=max_tokens, messages=msgs, costguard=CostGuardOptions(mode=mode, **opts))


LONG_CTX = ["Returns are accepted within 30 days of delivery for unused items in original packaging. " * 40]


# ---------------------------------------------------------------- #8 full system-prompt hash
def test_partition_uses_full_system_prompt_hash(tmp_path):
    a, b = "System instruction 30943", "System instruction 80067"      # same first 8 hex chars of sha256
    assert pipeline._h(a, 8) == pipeline._h(b, 8)
    sem = FakeSemantic()
    eng = make(tmp_path, exact_cache=DictCache(), semantic_cache=sem)
    eng.handle(req("hi there", system=a))
    _, rec = eng.handle(req("hi there", system=b))
    assert rec.cache_status == "miss"
    assert pipeline._h(b, 64) in sem.seen[-1]


# ---------------------------------------------------------------- #7 context identity in the partition
def test_semantic_partition_includes_context_digest(tmp_path):
    sem = FakeSemantic()
    eng = make(tmp_path, semantic_cache=sem)
    eng.handle(req("What is the return window?", context=["Returns within 30 days.", "Refunds take 5 days."]))
    _, other = eng.handle(req("What is the return window?", context=["Returns within 7 days."]))
    assert other.cache_status == "miss"                                  # different documents -> different partition
    _, reordered = eng.handle(req("What is the return window?", context=["Refunds take 5 days.", "Returns within 30 days."]))
    assert reordered.cache_status == "semantic"                          # same documents, other order -> same partition
    assert sem.seen[0].split("|")[-1] == "ctx:" + pipeline.context_digest(["Returns within 30 days.", "Refunds take 5 days."])
    eng.handle(req("Where is my order?"))
    assert sem.seen[-1].endswith("|noctx")


# ---------------------------------------------------------------- #10 generation limits + truncation
def test_max_tokens_is_part_of_cache_identity(tmp_path):
    eng = make(tmp_path, exact_cache=DictCache())
    eng.handle(req("Where is my order?", max_tokens=1))
    assert eng.handle(req("Where is my order?", max_tokens=256))[1].cache_status == "miss"
    assert eng.handle(req("Where is my order?", max_tokens=256))[1].cache_status == "exact"


@pytest.mark.parametrize("finish", ["length", "max_tokens"])
def test_truncated_completions_are_not_cached(tmp_path, finish):
    cache, sem = DictCache(), FakeSemantic()
    eng = make(tmp_path, provider=FinishProvider(finish), exact_cache=cache, semantic_cache=sem)
    eng.handle(req("Where is my order?"))
    assert eng.handle(req("Where is my order?"))[1].cache_status == "miss"
    assert not cache.d and not sem.items


def test_cache_hit_keeps_stored_finish_reason(tmp_path):
    eng = make(tmp_path, provider=FinishProvider("end_turn"), exact_cache=DictCache())
    eng.handle(req("Where is my order?"))
    comp, rec = eng.handle(req("Where is my order?"))
    assert rec.cache_status == "exact" and comp.finish_reason == "end_turn"


# ---------------------------------------------------------------- #9 promotion keeps the semantic similarity
def test_promoted_semantic_hit_is_rechecked_against_the_current_tau(tmp_path):
    cache, sem = DictCache(), FakeSemantic(sim=0.94)                     # balanced tau 0.93 < 0.94 < quality tau 0.95
    eng = make(tmp_path, exact_cache=cache, semantic_cache=sem)
    eng.handle(req("How do I return a jacket?"))
    _, r1 = eng.handle(req("how can I return my jacket"))
    assert r1.cache_status == "semantic"
    promoted = [e for e in cache.d.values() if "promoted_similarity" in e.metadata]
    assert len(promoted) == 1 and promoted[0].metadata["promoted_similarity"] == pytest.approx(0.94)
    assert "promoted_similarity" not in sem.items[0][2].metadata     # the semantic tier's entry is not mutated
    _, r2 = eng.handle(req("how can I return my jacket", mode="quality"))
    assert r2.cache_status == "miss"                                     # not served as exact under the stricter tau
    assert not [e for e in cache.d.values() if "promoted_similarity" in e.metadata]   # its own answer replaced it
    _, r3 = eng.handle(req("how can I return my jacket", mode="quality"))
    assert r3.cache_status == "exact" and r3.cache_entry_id == r2.request_id


def test_promoted_entry_served_as_exact_when_tau_allows(tmp_path):
    eng = make(tmp_path, exact_cache=DictCache(), semantic_cache=FakeSemantic(sim=0.94))
    eng.handle(req("How do I return a jacket?"))
    eng.handle(req("how can I return my jacket"))
    assert eng.handle(req("how can I return my jacket", mode="economy"))[1].cache_status == "exact"


# ---------------------------------------------------------------- #12 malformed COSTGUARD_API_KEYS
@pytest.mark.parametrize("bad", ["k-support", ":shopnest-support", "k-support:", "k1:a,k2", "k1:a,k1:b", ",,"])
def test_malformed_api_keys_fail_at_app_creation(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("COSTGUARD_API_KEYS", bad)
    with pytest.raises(ValueError, match="COSTGUARD_API_KEYS"):
        create_app(make(tmp_path))


def test_malformed_api_keys_after_startup_refuse_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("COSTGUARD_API_KEYS", "k-support:shopnest-support,")   # trailing comma is fine
    client = TestClient(create_app(make(tmp_path)))
    body = {"messages": [{"role": "user", "content": "Where is my order?"}]}
    assert client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer k-support"}).status_code == 200
    monkeypatch.setenv("COSTGUARD_API_KEYS", "k-support")
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 500 and "COSTGUARD_API_KEYS" in r.json()["detail"]


# ---------------------------------------------------------------- #13 requested cheap tier is an override
def test_requested_cheap_ignored_for_locked_tenant(tmp_path):
    eng = make(tmp_path)
    eng.policy.raw["tenants"]["locked-quality"] = {"mode": "quality", "allow_mode_override": False}
    _, rec = eng.handle(req("Where is my order?", mode=None, model="cheap", tenant="locked-quality"))
    assert rec.mode == "quality" and rec.model_requested == "strong" and rec.model_used == "strong"
    _, rec = eng.handle(req("Where is my order?", mode=None, model="cheap", tenant="shopnest-support"))
    assert rec.model_used == "strong"
    _, rec = eng.handle(req("Where is my order?", mode="off", model="cheap"))   # default tenant allows overrides
    assert rec.model_used == "cheap"


def test_gate_harness_still_gets_the_cheap_tier(tmp_path):
    from eval.gate_router import generate
    eng = make(tmp_path)
    it = {"id": "x-1", "query": "Do you ship to Pune?", "category": "shipping", "context": []}
    assert generate(eng, it, "cheap")["model"] == "mock-cheap"
    assert generate(eng, it, "strong")["model"] == "mock-strong"


# ---------------------------------------------------------------- #15 fail open, then fail safe
def test_compression_eligibility_count_is_inside_the_fail_open_boundary(tmp_path, monkeypatch):
    def boom(_text):
        raise ValueError("tokenizer exploded")
    monkeypatch.setattr(pipeline, "count_text", boom)
    comp, rec = make(tmp_path).handle(req("What is the return window?", context=LONG_CTX))
    assert comp.text and "compression" in rec.stage_errors
    assert rec.model_used == "strong" and rec.route_reason == "fail-safe:stage-error"


class TrimToFirst:
    def optimize(self, query, docs, budget_tokens, min_score):
        return ContextResult(docs=docs[:1], kept_indices=[0], note="trimmed")


class BoomCompressor:
    def compress(self, *a, **k):
        raise RuntimeError("compressor exploded")


def test_stage_error_skips_router_and_uses_strong_but_keeps_earlier_trim(tmp_path):
    router = CheapRouter()
    eng = make(tmp_path, context_optimizer=TrimToFirst(), compressor=BoomCompressor(), router=router)
    _, rec = eng.handle(req("What is the return window?", context=LONG_CTX + ["Shipping is free over Rs 499."]))
    assert "compression" in rec.stage_errors and router.calls == 0
    assert rec.model_used == "strong" and rec.route_reason == "fail-safe:stage-error"
    assert rec.context_docs_in == 2 and rec.context_docs_kept == 1      # compression's input passed through unchanged
    _, ok = make(tmp_path, router=CheapRouter()).handle(req("Do you ship to Pune?"))
    assert ok.model_used == "cheap" and not ok.stage_errors             # no error -> the router decides as before


# ---------------------------------------------------------------- #6 prices are part of config_hash
def test_config_hash_covers_prices(tmp_path):
    s = Settings()
    base = load_policy(s.policy_path, s.prices_path).config_hash
    same = tmp_path / "same.yaml"
    same.write_text(s.prices_path.read_text())
    assert load_policy(s.policy_path, same).config_hash == base
    changed = tmp_path / "changed.yaml"
    changed.write_text(s.prices_path.read_text() + "\n# price bump\n")
    assert load_policy(s.policy_path, changed).config_hash != base
