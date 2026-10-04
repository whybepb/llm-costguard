"""Router, hardness features, category classifier and the offline eval gate (mock backend only, no keys, no MLX)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from costguard.config import Settings, load_policy
from costguard.interfaces import Router
from costguard.router import features
from costguard.router.classifier import CATEGORIES, CategoryClassifier, keyword_category
from costguard.router.router import GatedRouter, build_router
from costguard.schemas import ChatMessage, ChatRequest, CostGuardOptions, RouteInput


# ---------------------------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def clf() -> CategoryClassifier:
    return CategoryClassifier().warmup()


def write_gate(path: Path, allow: dict[str, bool], **extra) -> Path:
    cats = {c: {"allow": a, "n": 40, "diff": 1.0, "ci": [-2.0, 4.0]} for c, a in allow.items()}
    path.write_text(json.dumps({"version": 1, "created": "test", "margin": 5.0, "judge": "test", "categories": cats,
                                **extra}))
    return path


def rin(q: str, category=None, requested="strong", **kw) -> RouteInput:
    return RouteInput(query=q, category=category, requested_alias=requested, **kw)


HARD_QUERIES = [
    ("Why was I charged twice for order #48213?", "reasoning-keywords"),
    ("Compare the AeroBeat Pro vs the AeroBeat Lite for running", "reasoning-keywords"),
    ("Can you explain the difference between exchange and return?", "reasoning-keywords"),
    ("If I return 2 shirts at $20 and keep 1 at $35, what is my total refund?", "arithmetic"),
    ("Where is my order? And when will my refund arrive?", "multi-question"),
    ("I want a refund. Also, can you change my address?", "multi-question"),
    ("THIS IS RIDICULOUS, I WANT MY MONEY BACK NOW", "escalation"),
    ("I will file a chargeback and talk to my lawyer", "escalation"),
    ("This looks like fraud, there is a charge I never made", "escalation"),
    ("¿Dónde está mi pedido? Lo necesito hoy", "non-english"),
    ("mera refund kab aayega bhai", "non-english"),
    ("Checkout shows TypeError: undefined is not a function", "code"),
    ("My script does `curl -X POST /api/orders` and gets HTTP error 500", "code"),
]
EASY_QUERIES = ["How do I return a jacket?", "Do you ship to Pune?", "refund not received", "I forgot my password",
                "Is the air fryer in stock?", "I bought it on 2026-09-12, can I still return it?",
                "Thanks! Also, do you ship to Pune?", "where is my package"]


# ---------------------------------------------------------------------------------------------- features
@pytest.mark.parametrize("q,reason", HARD_QUERIES)
def test_hard_signals_detected(q, reason):
    h = features.extract(q)
    assert h.hard and reason in h.reasons, (q, h.reasons)


@pytest.mark.parametrize("q", EASY_QUERIES)
def test_easy_queries_not_flagged(q):
    h = features.extract(q)
    assert not h.hard, (q, h.reasons)


def test_size_signals_and_config():
    assert features.extract("hi", input_tokens=9000).reasons == ["long-input"]
    assert features.extract("hi", history_turns=6).reasons == ["deep-history"]
    assert features.extract("hi", context_docs=12).reasons == ["many-context-docs"]
    assert "long-query" in features.extract("please help " * 100).reasons
    cfg = features.HardnessConfig.from_dict({"max_history_turns": 10, "unknown_key": 1})
    assert not features.extract("hi", history_turns=6, cfg=cfg).hard


# ---------------------------------------------------------------------------------------------- router
def test_router_satisfies_protocol_and_exposes_latency(tmp_path, clf):
    r = GatedRouter(tmp_path / "none.json", clf)
    assert isinstance(r, Router)
    d = r.route(rin("Do you ship to Pune?"), "gated")
    assert d.alias in ("strong", "cheap") and r.last_ms >= 0 and r.last["reason"] == d.reason


@pytest.mark.parametrize("policy", ["gated", "aggressive"])
@pytest.mark.parametrize("q,reason", HARD_QUERIES[:6])
def test_hard_signals_force_strong(tmp_path, clf, policy, q, reason):
    gate = write_gate(tmp_path / "g.json", {c: True for c in CATEGORIES})       # everything allowed
    d = GatedRouter(gate, clf).route(rin(q, category="returns"), policy)
    assert d.alias == "strong" and d.reason.startswith("hard:"), d


def test_gated_without_gate_file_stays_strong(tmp_path, clf):
    d = GatedRouter(tmp_path / "missing.json", clf).route(rin("How do I return a jacket?", "returns"), "gated")
    assert (d.alias, d.reason) == ("strong", "gated:no-gate-file")


def test_gated_allowed_category_goes_cheap(tmp_path, clf):
    gate = write_gate(tmp_path / "g.json", {"returns": True, "refund": False})
    r = GatedRouter(gate, clf)
    d = r.route(rin("How do I return a jacket?", "returns"), "gated")
    assert (d.alias, d.reason, d.category) == ("cheap", "gated:returns-allowed", "returns")
    assert r.route(rin("refund not received", "refund"), "gated").reason == "gated:refund-blocked"
    assert r.route(rin("Is the air fryer in stock?", "product"), "gated").reason == "gated:product-not-evaluated"
    assert r.route(rin("Do you ship to Pune?", "billing"), "gated").reason == "gated:unknown-category"
    assert r.route(rin("How do I return a jacket?", " Returns "), "gated").alias == "cheap"   # normalised


def test_gated_infers_category_only_when_confident(tmp_path, clf):
    gate = write_gate(tmp_path / "g.json", {"refund": True, "returns": True, "other": True, "product": True})
    r = GatedRouter(gate, clf)
    d = r.route(rin("refund not received"), "gated")
    if clf.backend == "embedding":
        assert (d.alias, d.reason) == ("cheap", "gated:refund-allowed") and r.last["category_source"] == "embedding"
    else:  # keyword-only inference is never trusted for a downshift
        assert d.reason == "gated:unclassified"
    off = r.route(rin("write me a poem about cats"), "gated")
    assert (off.alias, off.reason) == ("strong", "gated:unclassified")


def test_dry_run_or_bad_gate_stays_strong(tmp_path, clf):
    dry = write_gate(tmp_path / "dry.json", {"returns": True}, dry_run=True)
    assert GatedRouter(dry, clf).route(rin("How do I return a jacket?", "returns"), "gated").reason == "gated:dry-run-gate"
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert GatedRouter(bad, clf).route(rin("How do I return a jacket?", "returns"), "gated").reason == "gated:bad-gate-file"


def test_gate_for_other_models_fails_safe(tmp_path, clf):
    pair = {"backend": "mlx", "models": {"strong": "qwen-7b", "cheap": "qwen-1.5b"}}
    gate = write_gate(tmp_path / "g.json", {"returns": True}, **pair)
    q = rin("How do I return a jacket?", "returns")
    assert GatedRouter(gate, clf, expect=pair).route(q, "gated").alias == "cheap"          # same pair -> applied
    other = {"backend": "anthropic", "models": {"strong": "claude-sonnet-5-5", "cheap": "claude-haiku-4-5-20251001"}}
    d = GatedRouter(gate, clf, expect=other).route(q, "gated")
    assert (d.alias, d.reason) == ("strong", "gated:gate-for-other-models")               # foreign gate -> strong
    mock = {"backend": "mock", "models": {"strong": "mock-strong", "cheap": "mock-cheap"}}
    assert GatedRouter(gate, clf, expect=mock).route(q, "gated").alias == "cheap"          # mock: warn only


def test_gate_file_hot_reload_is_the_rollback(tmp_path, clf):
    gate = write_gate(tmp_path / "g.json", {"returns": True})
    r = GatedRouter(gate, clf)
    assert r.route(rin("How do I return a jacket?", "returns"), "gated").alias == "cheap"
    write_gate(gate, {"returns": False})
    os.utime(gate, (time.time() + 5, time.time() + 5))                 # ensure a new mtime on coarse filesystems
    assert r.route(rin("How do I return a jacket?", "returns"), "gated").alias == "strong"
    gate.unlink()
    assert r.route(rin("How do I return a jacket?", "returns"), "gated").reason == "gated:no-gate-file"


def test_rollout_stages_shadow_and_canary(tmp_path, clf):
    gate = tmp_path / "g.json"
    gate.write_text(json.dumps({"categories": {"returns": {"allow": True, "rollout": "shadow"},
                                               "refund": {"allow": True, "rollout": "canary:0.25"},
                                               "shipping": {"allow": True, "rollout": "sideways"}}}))
    r = GatedRouter(gate, clf)
    d = r.route(rin("How do I return a jacket?", "returns"), "gated")
    assert (d.alias, d.reason) == ("strong", "gated:returns-shadow")
    qs = [f"refund {i} not received for order #{1000 + i}" for i in range(400)]
    arms = [r.route(rin(q, "refund"), "gated").reason for q in qs]
    share = arms.count("gated:refund-canary") / len(qs)
    assert set(arms) == {"gated:refund-canary", "gated:refund-holdout"} and 0.18 < share < 0.32
    assert r.route(rin(qs[0], "refund"), "gated").reason == arms[0]          # sticky per query
    assert r.route(rin("Do you ship to Pune?", "shipping"), "gated").reason == "gated:shipping-bad-rollout"


@pytest.mark.parametrize("q", EASY_QUERIES)
def test_aggressive_downshifts_easy_queries(tmp_path, clf, q):
    d = GatedRouter(tmp_path / "missing.json", clf).route(rin(q), "aggressive")   # no gate needed
    assert (d.alias, d.reason) == ("cheap", "aggressive:easy")


def test_explicit_cheap_request_is_honoured(tmp_path, clf):
    r = GatedRouter(tmp_path / "missing.json", clf)
    d = r.route(rin("Why was I charged twice?", requested="cheap"), "gated")
    assert (d.alias, d.reason) == ("cheap", "requested-cheap")


# ---------------------------------------------------------------------------------------------- classifier
@pytest.mark.parametrize("q,ok", [("where is my package", {"shipping", "order"}), ("refund not received", {"refund"}),
                                  ("How do I return a jacket?", {"returns"}), ("I forgot my password", {"account"}),
                                  ("my card got declined at checkout", {"payment"})])
def test_classifier_maps_obvious_queries(clf, q, ok):
    assert clf.classify(q).category in ok
    assert keyword_category(q) in ok                                      # the fallback agrees


def test_classifier_keyword_fallback_and_off_topic(clf):
    kw = CategoryClassifier(use_embeddings=False)
    assert kw.backend == "keywords" and kw.classify("refund not received").source == "keywords"
    assert kw.classify("write me a poem about cats").category is None
    if clf.backend == "embedding":
        assert clf.classify("write me a poem about cats").category is None   # below the similarity floor


def test_build_router_from_factory_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("COSTGUARD_ROUTER_GATE", str(write_gate(tmp_path / "g.json", {"returns": True})))
    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    r = build_router(s, load_policy(s.policy_path))
    assert r.gate.path == tmp_path / "g.json" and r.status()["allowed"] == ["returns"]


# ---------------------------------------------------------------------------------------------- engine integration
def test_engine_balanced_mode_downshifts_allowed_category(tmp_path, monkeypatch):
    from costguard.factory import build_engine
    monkeypatch.setenv("COSTGUARD_ROUTER_GATE", str(write_gate(tmp_path / "g.json", {"returns": True})))
    eng = build_engine(Settings(backend="mock", db_path=tmp_path / "t.sqlite"), with_logger=False,
                       skip=("exact_cache", "semantic_cache", "context_optimizer", "compressor"))
    assert eng.component_status["router"] == "GatedRouter"

    def ask(q, cat, mode="balanced"):
        return eng.handle(ChatRequest(messages=[ChatMessage(role="user", content=q)],
                                      costguard=CostGuardOptions(mode=mode, category=cat, no_cache=True)))[1]

    rec = ask("How do I return a jacket?", "returns")
    assert rec.model_used == "cheap" and rec.route_reason == "gated:returns-allowed"
    assert rec.cost_usd < rec.baseline_cost_usd and rec.saved_usd > 0
    assert ask("Why can't I return a jacket after 30 days?", "returns").model_used == "strong"   # hard
    assert ask("Do you ship to Pune?", "shipping").route_reason == "gated:shipping-not-evaluated"
    assert ask("How do I return a jacket?", "returns", mode="quality").model_used == "strong"   # router off


# ---------------------------------------------------------------------------------------------- eval gate
def _gate_data(tmp_path: Path) -> Path:
    rows = []
    for cat in ("returns", "refund", "shipping"):
        for i in range(4):
            rows.append({"id": f"{cat}-{i}", "category": cat, "query": f"question {i} about {cat} for my order",
                         "reference": f"Our {cat} policy answer {i}: please check your order in your account.",
                         "needs_context": False, "type": "answerable", "pair_id": None, "author": "test"})
    rows.append({"id": "hard-0", "category": "refund", "query": "Why was I charged twice? Explain step by step.",
                 "reference": "We will investigate.", "needs_context": False, "type": "hard"})
    p = tmp_path / "evalset.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


def test_gate_router_dry_run_end_to_end_on_mock(tmp_path, monkeypatch):
    from eval import gate_router
    monkeypatch.setenv("COSTGUARD_BACKEND", "mock")
    monkeypatch.delenv("COSTGUARD_CASSETTE", raising=False)
    monkeypatch.setattr(gate_router, "GATE_PATH", tmp_path / "router_gate.json")
    full = write_gate(tmp_path / "router_gate.json", {"returns": True})          # an existing *full* gate
    res = gate_router.run(gate_router.parse_args(
        ["--dry-run", "--data", str(_gate_data(tmp_path)), "--results-out", str(tmp_path / "res.json"),
         "--min-n", "3", "--bootstrap", "200", "--quiet"]))
    assert res["dry_run"] and res["allowed"] == [] and res["n_scored"] == 13
    assert res["judge"] == "heuristic-mock"
    cats = res["categories"]
    assert cats["returns"]["n"] == 4 and cats["returns"]["reason"] == "dry-run" and len(cats["returns"]["ci"]) == 2
    assert cats["refund"]["n_items"] == 5 and cats["refund"]["n"] == 4            # the hard item is not routable
    assert cats["order"]["n"] == 0 and "pairwise" in cats["returns"]
    assert res["hard_signal_rate"]["hard"]["flagged"] == 1
    assert res["price_gap"]["saving_per_downshifted_request"] > 0
    assert json.loads(full.read_text())["categories"]["returns"]["allow"] is True   # full gate NOT overwritten
    saved = json.loads((tmp_path / "res.json").read_text())
    assert saved["written"]["gate"] is None and (tmp_path / "res_items.json").exists()

    # an explicit --gate-out is written, marked dry_run, and allows nothing
    out = tmp_path / "dry_gate.json"
    gate_router.run(gate_router.parse_args(["--data", str(_gate_data(tmp_path)), "--gate-out", str(out),
                                            "--results-out", str(tmp_path / "res2.json"), "--quiet"]))
    g = json.loads(out.read_text())
    assert g["dry_run"] is True and not any(v["allow"] for v in g["categories"].values())
    assert set(g["categories"]) == set(CATEGORIES) and g["backend"] == "mock"


def test_gate_router_paid_backend_requires_yes(tmp_path, monkeypatch):
    from eval import gate_router
    monkeypatch.delenv("COSTGUARD_CASSETTE", raising=False)
    monkeypatch.setattr(gate_router, "GATE_PATH", tmp_path / "router_gate.json")

    def boom(*a, **k):
        raise AssertionError("must not build an engine without --yes")
    monkeypatch.setattr(gate_router, "make_engine", boom)
    res = gate_router.run(gate_router.parse_args(["--backend", "anthropic", "--data", str(_gate_data(tmp_path)),
                                                  "--cassette", str(tmp_path / "c.jsonl"), "--quiet"]))
    assert res["status"] == "needs_confirmation"
    est = res["estimate"]
    assert est["generations_new"] == {"strong": 13, "cheap": 13} and est["usd_total_upper_bound"] > 0
    assert not (tmp_path / "router_gate.json").exists()
