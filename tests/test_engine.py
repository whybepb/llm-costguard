"""Engine-level tests with the mock provider and tiny in-test fakes (stage modules have their own tests)."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from costguard.config import Settings, load_policy
from costguard.pipeline import CostGuard
from costguard.pricing import PriceBook
from costguard.providers.cassette import CassetteMiss, CassetteProvider
from costguard.providers.mock import MockProvider
from costguard.schemas import ChatMessage, ChatRequest, CostGuardOptions, RouteDecision, SemanticHit
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


class Boom:
    def optimize(self, *a, **k):
        raise RuntimeError("stage exploded")


class CheapRouter:
    def route(self, inp, policy):
        return RouteDecision(alias="cheap", reason="test")


def make(tmp_path: Path, **kw) -> CostGuard:
    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    pol = load_policy(s.policy_path)
    return CostGuard(pol, s, kw.pop("provider", MockProvider()), PriceBook(s.prices_path, pol.billing), **kw)


def req(q: str, mode: str = "balanced", **opts) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=q)], costguard=CostGuardOptions(mode=mode, **opts))


def test_passthrough_off_mode_costs_equal_baseline(tmp_path):
    eng = make(tmp_path)
    comp, rec = eng.handle(req("Where is my order?", mode="off"))
    assert comp.text and rec.cache_status == "disabled" and rec.model_used == "strong"
    assert rec.cost_usd == pytest.approx(rec.baseline_cost_usd) and rec.saved_usd == pytest.approx(0)


def test_exact_cache_hit_second_time(tmp_path):
    eng = make(tmp_path, exact_cache=DictCache())
    _, r1 = eng.handle(req("How do I return a jacket?"))
    _, r2 = eng.handle(req("how do I return a jacket"))   # normalised
    assert r1.cache_status == "miss" and r2.cache_status == "exact"
    assert r2.cost_usd == 0 and r2.saved_usd > 0 and r2.response_text == r1.response_text


def test_cache_bypassed_for_multiturn_and_high_temperature(tmp_path):
    eng = make(tmp_path, exact_cache=DictCache())
    r = ChatRequest(messages=[ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello"),
                              ChatMessage(role="user", content="track my order")], costguard=CostGuardOptions(mode="balanced"))
    assert eng.handle(r)[1].cache_status == "bypass"
    hot = req("write a poem about shoes")
    hot.temperature = 0.9
    assert eng.handle(hot)[1].cache_status == "bypass"


def test_failing_stage_fails_open(tmp_path):
    eng = make(tmp_path, context_optimizer=Boom())
    comp, rec = eng.handle(req("What is the return window?", context=["Returns accepted within 30 days."]))
    assert comp.text and "context" in rec.stage_errors and rec.context_docs_kept == 1


def test_router_downshift_is_cheaper(tmp_path):
    eng = make(tmp_path, router=CheapRouter())
    _, rec = eng.handle(req("Do you ship to Pune?"))
    assert rec.model_used == "cheap" and rec.cost_usd < rec.baseline_cost_usd


def test_quality_mode_refuses_cheap_tier_cache_entries(tmp_path):
    cache = DictCache()
    eng = make(tmp_path, exact_cache=cache, router=CheapRouter())
    eng.handle(req("Can I change my address?", mode="economy"))
    _, rec = eng.handle(req("Can I change my address?", mode="quality"))
    assert rec.cache_status == "miss" and rec.model_used == "strong"


def test_cassette_replay_is_free_and_identical(tmp_path):
    cas = tmp_path / "c.jsonl"
    rec_p = CassetteProvider(MockProvider(), cas, "record")
    eng = make(tmp_path, provider=rec_p)
    a, _ = eng.handle(req("Is there a warranty?", mode="off"))
    rep = CassetteProvider(MockProvider(), cas, "replay")
    eng2 = make(tmp_path, provider=rep)
    b, _ = eng2.handle(req("Is there a warranty?", mode="off"))
    assert a.text == b.text and rep.hits == 1
    with pytest.raises(CassetteMiss):
        eng2.handle(req("never recorded", mode="off"))


def test_http_endpoint_openai_shape(tmp_path):
    from costguard.obs.logger import RequestLogger
    eng = make(tmp_path, exact_cache=DictCache())
    eng.hooks.append(RequestLogger(tmp_path / "log.sqlite"))
    client = TestClient(create_app(eng))
    body = {"model": "strong", "messages": [{"role": "user", "content": "Where is my refund?"}],
            "costguard": {"mode": "balanced"}}
    r1 = client.post("/v1/chat/completions", json=body)
    r2 = client.post("/v1/chat/completions", json=body)
    assert r1.status_code == 200 and r1.json()["choices"][0]["message"]["content"]
    assert r2.headers["x-costguard-cache"] == "exact"
    st = client.get("/v1/stats").json()
    assert st["requests"] == 2 and st["cache_hit_rate"] == 0.5
    assert client.get("/health").json()["status"] == "ok"


def test_service_side_tenant_from_api_key_and_locked_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("COSTGUARD_API_KEYS", "k-support:shopnest-support,k-internal:shopnest-internal")
    eng = make(tmp_path)
    client = TestClient(create_app(eng))
    body = {"messages": [{"role": "user", "content": "Where is my order?"}],
            "costguard": {"mode": "off", "tenant": "spoofed-tenant"}}
    assert client.post("/v1/chat/completions", json=body).status_code == 401           # no key -> rejected
    r = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer k-support"})
    cg = r.json()["costguard"]
    assert r.status_code == 200 and cg["mode"] == "balanced"                          # body override ignored
    r2 = client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer k-internal"})
    assert r2.json()["costguard"]["mode"] == "off"                                     # this tenant may override


class DownProvider(MockProvider):
    """Every upstream call fails (both tiers)."""
    def complete(self, *a, **k):
        raise ConnectionError("provider down")


def test_double_upstream_failure_is_logged_and_counted(tmp_path):
    seen = []
    eng = make(tmp_path, provider=DownProvider(), router=CheapRouter(), hooks=[seen.append])
    with pytest.raises(ConnectionError):
        eng.handle(req("Where is my order?"))
    assert len(seen) == 1 and seen[0].error.startswith("ConnectionError")
    assert seen[0].stage_errors["upstream_cheap"].startswith("ConnectionError")


def test_stream_true_is_rejected_not_silently_empty(tmp_path):
    client = TestClient(create_app(make(tmp_path)))
    body = {"messages": [{"role": "user", "content": "Where is my order?"}], "stream": True}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400 and "stream" in r.json()["detail"]


def test_shutdown_flushes_request_log(tmp_path):
    from costguard.obs.logger import RequestLogger
    log = RequestLogger(tmp_path / "log.sqlite", flush_interval_s=5.0)   # slow batching: rows sit in the queue
    eng = make(tmp_path)
    eng.hooks.append(log)
    with TestClient(create_app(eng)) as client:                          # context manager runs the lifespan
        for i in range(3):
            client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": f"order {i}?"}]})
    assert log._q.unfinished_tasks == 0                                  # drained by shutdown, not by a reader
    assert len(log.rows()) == 3
