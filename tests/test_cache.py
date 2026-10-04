"""Cache tiers: exact (TTL/LRU), guards, semantic (memory + Qdrant local), and the engine wired with both.

The embedding model (bge-small, ~130 MB ONNX) loads once per session; everything else is in-memory and fast.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from costguard.cache import guards as G
from costguard.cache.exact import InMemoryExactCache, build_exact_cache
from costguard.cache.semantic import MemoryStore, SemanticCacheImpl, build_semantic_cache
from costguard.config import Settings, load_policy
from costguard.pipeline import CostGuard
from costguard.pricing import PriceBook
from costguard.providers.mock import MockProvider
from costguard.schemas import CacheEntry, ChatMessage, ChatRequest, CostGuardOptions


class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def entry(q: str, alias: str = "strong", text: str | None = None) -> CacheEntry:
    return CacheEntry(entry_id=q[:10], query_text=q, response_text=text or f"answer to: {q}", model_alias=alias,
                      input_tokens=50, output_tokens=20)


@pytest.fixture(scope="session")
def embedder():
    from costguard.cache.embedder import get_embedder
    try:
        return get_embedder().warmup()
    except Exception as e:  # no model weights and no network
        pytest.skip(f"embedding model unavailable: {type(e).__name__}: {e}")


@pytest.fixture
def cache(embedder):
    return SemanticCacheImpl(embedder, MemoryStore(), ttl_seconds=3600, clock=Clock())


# ============================================================================================ exact tier


def test_exact_ttl_expiry():
    clk = Clock()
    c = InMemoryExactCache(ttl_seconds=10, clock=clk)
    c.put("k", entry("q"))
    assert c.get("k").query_text == "q"
    clk.advance(11)
    assert c.get("k") is None and c.expired == 1 and c.size() == 0


def test_exact_lru_cap_evicts_least_recently_used():
    c = InMemoryExactCache(ttl_seconds=100, max_entries=2)
    c.put("a", entry("a"))
    c.put("b", entry("b"))
    assert c.get("a") is not None           # touch a -> b becomes LRU
    c.put("c", entry("c"))
    assert c.get("b") is None and c.get("a") is not None and c.get("c") is not None and c.evictions == 1


def test_exact_never_downgrades_strong_to_cheap():
    c = InMemoryExactCache(ttl_seconds=100)
    c.put("k", entry("q", alias="strong", text="strong answer"))
    c.put("k", entry("q", alias="cheap", text="cheap answer"))
    assert c.get("k").response_text == "strong answer"
    c.clear()
    assert c.get("k") is None


def test_build_exact_cache_uses_policy_ttl():
    s = Settings()
    pol = load_policy(s.policy_path)
    c = build_exact_cache(s, pol)
    assert c.ttl == float(pol.cache["ttl_seconds"])


# ============================================================================================ guards


@pytest.mark.parametrize("a,b,reason", [
    ("I want to cancel my order #4821", "I want to cancel my order #4822", "number_mismatch"),
    ("Please cancel order SN-48213, it hasn't shipped yet.", "Please cancel order SN-48231, it hasn't shipped yet.",
     "number_mismatch"),
    ("Can I return an item after 30 days?", "Can I return an item after 60 days?", "number_mismatch"),
    ("I want to cancel my order #4821", "I don't want to cancel my order #4821", "negation_mismatch"),
    ("Can I return an item without the receipt?", "Can I return an item with the receipt?", "negation_mismatch"),
    ("How do I subscribe to the newsletter?", "How do I unsubscribe from the newsletter?", "negation_mismatch"),
    ("What is the return window for laptops?", "What is the return window for phones?", "entity_mismatch:product"),
    ("I want a refund for my jacket", "I want an exchange for my jacket", "entity_mismatch:action"),
    ("Can I pay with PayPal?", "Can I pay with UPI?", "entity_mismatch:payment"),
    ("Can I switch to the premium plan?", "Can I switch to the free plan?", "entity_mismatch:tier"),
    ("Do you ship to Nagpur?", "Do you ship to Indore?", "content_mismatch"),
])
def test_guard_rejects_lookalikes(a, b, reason):
    assert G.check(a, b) == reason
    assert G.check(b, a) == reason          # symmetric


@pytest.mark.parametrize("a,b", [
    ("How do I cancel my order #4821?", "how can i cancel order 4821"),
    ("I can't log in to my account", "Help me log in to my account"),
    ("I don't know how to track my order", "track my order"),           # hedge, not a negation
    ("I want to close my gold account", "how do I cancel a gold account"),
    ("Where is my package?", "Can you tell me where my parcel is?"),
    ("i need help trackng order {{Order Number}}", "I need help tracking purchase {{Order Number}}"),
])
def test_guard_allows_paraphrases(a, b):
    assert G.check(a, b) is None


def test_guard_toggles():
    a, b = "I want to cancel my order #4821", "I want to cancel my order #4822"
    assert G.check(a, b, G.GuardConfig(numbers=False)) is None
    assert G.check(a, b, G.ALL_OFF) is None
    cfg = G.GuardConfig.from_mapping({"numbers": False, "negation": True, "bogus": 1})
    assert cfg.numbers is False and cfg.negation and cfg.entities and cfg.content


# ============================================================================================ semantic tier (memory)


def test_paraphrase_hit(cache):
    cache.insert("How do I return a jacket?", "t|s|kb|noctx", entry("How do I return a jacket?"))
    h = cache.lookup("how can I return my jacket", "t|s|kb|noctx", 0.90)
    assert h.entry is not None and h.entry.query_text == "How do I return a jacket?"
    assert h.similarity >= 0.90 and h.neighbor_query == "How do I return a jacket?" and h.guard_rejected is None
    assert cache.last_embed_ms >= 0 and cache.model_name == "BAAI/bge-small-en-v1.5"


def test_unrelated_miss_still_reports_neighbour(cache):
    cache.insert("How do I return a jacket?", "p", entry("How do I return a jacket?"))
    h = cache.lookup("Do you sell gift cards?", "p", 0.85)
    assert h.entry is None and h.similarity is not None and h.similarity < 0.85
    assert h.neighbor_query == "How do I return a jacket?" and h.guard_rejected is None


def test_partition_isolation(cache):
    cache.insert("How do I return a jacket?", "tenantA|sys1|kb-v1|noctx", entry("How do I return a jacket?"))
    assert cache.lookup("How do I return a jacket?", "tenantB|sys1|kb-v1|noctx", 0.5).similarity is None
    assert cache.lookup("How do I return a jacket?", "tenantA|sys1|kb-v2|noctx", 0.5).entry is None   # kb bump
    assert cache.lookup("How do I return a jacket?", "tenantA|sys1|kb-v1|noctx", 0.5).entry is not None


def test_semantic_ttl_expiry(embedder):
    clk = Clock()
    c = SemanticCacheImpl(embedder, MemoryStore(), ttl_seconds=60, clock=clk)
    c.insert("How do I return a jacket?", "p", entry("How do I return a jacket?"))
    assert c.lookup("how can I return my jacket", "p", 0.9).entry is not None
    clk.advance(61)
    h = c.lookup("how can I return my jacket", "p", 0.9)
    assert h.entry is None and h.similarity is None


@pytest.mark.parametrize("q,reason", [
    ("I don't want to cancel my order #4821", "negation_mismatch"),
    ("I want to cancel my order #4822", "number_mismatch"),
])
def test_guard_rejection_turns_near_hit_into_miss(cache, q, reason):
    cache.insert("I want to cancel my order #4821", "p", entry("I want to cancel my order #4821"))
    h = cache.lookup(q, "p", 0.90)
    assert h.similarity >= 0.90                 # the embedding alone would have served it
    assert h.entry is None and h.guard_rejected == reason


def test_entity_guard_in_cache(cache):
    cache.insert("How do I return a jacket?", "p", entry("How do I return a jacket?"))
    h = cache.lookup("How do I exchange a jacket?", "p", 0.80)
    assert h.entry is None and h.guard_rejected == "entity_mismatch:action"


def test_guarded_cache_serves_the_next_valid_candidate(cache):
    cache.insert("I want to cancel my order #4821", "p", entry("I want to cancel my order #4821"))
    cache.insert("I want to cancel my order #4822", "p", entry("I want to cancel my order #4822"))
    h = cache.lookup("please cancel my order #4822", "p", 0.90)
    assert h.entry is not None and h.entry.query_text.endswith("#4822")


def test_dedup_and_tier_upgrade(cache):
    q = "I want to cancel my order #4821"
    cache.insert(q, "p", entry(q, alias="cheap", text="cheap"))
    cache.insert("i would like to cancel my order #4821", "p", entry(q, alias="cheap", text="cheap2"))  # near-dup
    assert cache.counters["dedup_skips"] == 1 and cache.store.count() == 1
    cache.insert(q, "p", entry(q, alias="strong", text="strong"))     # strong answer replaces the cheap one
    assert cache.counters["tier_upgrades"] == 1 and cache.store.count() == 1
    assert cache.lookup(q, "p", 0.9).entry.response_text == "strong"
    cache.insert("I want to cancel my order #4822", "p", entry("#4822"))   # lookalike, not a duplicate
    assert cache.store.count() == 2
    cache.clear()
    assert cache.store.count() == 0 and cache.lookup(q, "p", 0.5).entry is None


def test_guards_can_be_disabled(embedder):
    c = SemanticCacheImpl(embedder, MemoryStore(), guards=G.ALL_OFF)
    c.insert("I want to cancel my order #4821", "p", entry("I want to cancel my order #4821"))
    assert c.lookup("I don't want to cancel my order #4821", "p", 0.90).entry is not None   # the false hit


def test_build_semantic_cache_default_backend(embedder):
    s = Settings()
    c = build_semantic_cache(s, load_policy(s.policy_path))
    assert isinstance(c, SemanticCacheImpl) and isinstance(c.store, MemoryStore) and c.guards == G.ALL_ON


# ============================================================================================ semantic tier (qdrant)


def test_qdrant_local_backend(embedder, tmp_path: Path):
    try:
        from costguard.cache.semantic import QdrantStore
        store = QdrantStore(dim=384, collection="test_semcache", path=tmp_path / "qdrant")
    except Exception as e:
        pytest.skip(f"qdrant local mode unavailable: {type(e).__name__}: {e}")
    clk = Clock()
    c = SemanticCacheImpl(embedder, store, ttl_seconds=60, clock=clk)
    try:
        c.insert("How do I return a jacket?", "A|s|kb|noctx", entry("How do I return a jacket?"))
        h = c.lookup("how can I return my jacket", "A|s|kb|noctx", 0.9)
        assert h.entry is not None and h.entry.response_text == "answer to: How do I return a jacket?"
        assert c.lookup("how can I return my jacket", "B|s|kb|noctx", 0.5).entry is None      # partition filter
        h = c.lookup("How do I exchange a jacket?", "A|s|kb|noctx", 0.8)
        assert h.entry is None and h.guard_rejected == "entity_mismatch:action"
        clk.advance(61)
        assert c.lookup("how can I return my jacket", "A|s|kb|noctx", 0.5).entry is None      # TTL
        c.clear()
        assert store.count() == 0
    finally:
        store.close()


# ============================================================================================ engine integration


def _req(q: str, mode: str = "balanced", **opts) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=q)], costguard=CostGuardOptions(mode=mode, **opts))


def test_engine_semantic_hit_and_guarded_trap(tmp_path, embedder):
    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    pol = load_policy(s.policy_path)
    pol.modes["balanced"].tau = 0.90            # pinned so the test does not move when policy.yaml is recalibrated
    provider = MockProvider()
    eng = CostGuard(pol, s, provider, PriceBook(s.prices_path, pol.billing),
                    exact_cache=build_exact_cache(s, pol), semantic_cache=build_semantic_cache(s, pol))

    _, r1 = eng.handle(_req("I want to cancel my order #4821"))
    _, r2 = eng.handle(_req("Please cancel my order #4821"))                   # paraphrase -> semantic hit
    _, r3 = eng.handle(_req("I want to cancel my order #4821"))                # identical -> exact hit
    _, r4 = eng.handle(_req("I don't want to cancel my order #4821"))          # negated trap -> guarded miss
    _, r5 = eng.handle(_req("Please cancel my order #4821", tenant="other"))  # other tenant -> miss

    assert r1.cache_status == "miss"
    assert r2.cache_status == "semantic" and r2.response_text == r1.response_text
    assert r2.cost_usd == 0 and r2.saved_usd > 0 and r2.cache_similarity >= 0.90
    assert r3.cache_status == "exact"
    assert r4.cache_status == "miss" and r4.cache_guard == "negation_mismatch" and r4.cache_similarity >= 0.90
    assert r4.response_text != r1.response_text
    assert r5.cache_status == "miss"
    assert provider.calls == 3


def test_engine_quality_mode_skips_cheap_semantic_entry(tmp_path, embedder):
    from costguard.schemas import RouteDecision

    class CheapRouter:
        def route(self, inp, policy):
            return RouteDecision(alias="cheap", reason="test")

    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    pol = load_policy(s.policy_path)
    pol.modes["economy"].tau = pol.modes["quality"].tau = 0.90
    eng = CostGuard(pol, s, MockProvider(), PriceBook(s.prices_path, pol.billing),
                    semantic_cache=build_semantic_cache(s, pol), router=CheapRouter())
    eng.handle(_req("How do I return a jacket?", mode="economy"))                   # cached by the cheap tier
    _, rq = eng.handle(_req("how can I return my jacket", mode="quality"))
    assert rq.cache_status == "miss" and rq.cache_guard == "tier_mismatch" and rq.model_used == "strong"
    _, rq2 = eng.handle(_req("how can I return my jacket?", mode="quality"))        # strong answer now cached
    assert rq2.cache_status == "semantic"
