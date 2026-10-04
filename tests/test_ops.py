"""Ops workstream tests: Prometheus /metrics, never-raising hooks (incl. broken Langfuse), PSI drift maths, and the
dashboard module importing without running Streamlit. Mock backend only; no network."""
from __future__ import annotations

import importlib.util
import math
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from costguard.config import Settings, load_policy
from costguard.obs import drift as drift_mod
from costguard.obs.drift import DriftHook, DriftMonitor, psi, psi_band, psi_term, quantile_edges, bin_numeric
from costguard.obs.hooks import SafeHook, build_hooks
from costguard.obs.langfuse_hook import LangfuseHook, build_langfuse_hook, otel_attributes, send_trace
from costguard.obs.logger import RequestLogger
from costguard.obs.metrics import CostGuardMetrics, MetricsHook
from costguard.pipeline import CostGuard
from costguard.pricing import PriceBook
from costguard.providers.mock import MockProvider
from costguard.schemas import TraceRecord
from costguard.server import create_app

ROOT = Path(__file__).resolve().parent.parent


def _engine(tmp_path: Path, hooks: list) -> CostGuard:
    s = Settings(backend="mock", db_path=tmp_path / "t.sqlite")
    pol = load_policy(s.policy_path)
    return CostGuard(pol, s, MockProvider(), PriceBook(s.prices_path, pol.billing), hooks=hooks)


def _metric_sum(text: str, name: str, **labels) -> float:
    total = 0.0
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()):
                total += s.value
    return total


def _rec(**kw) -> TraceRecord:
    base = dict(request_id="r1", ts=time.time(), mode="balanced", category="order", provider="mock",
                cache_status="miss", model_used="strong", model_id="mock-strong", input_tokens_original=120,
                input_tokens_sent=100, output_tokens=40, cost_usd=0.0002, baseline_cost_usd=0.0003, saved_usd=0.0001,
                latency_ms=310.0, upstream_latency_ms=300.0, overhead_ms=10.0, stage_ms={"router": 0.2, "upstream": 300.0},
                config_hash="abc123", query="where is my order", response_text="soon")
    base.update(kw)
    return TraceRecord(**base)


# ---------------------------------------------------------------- /metrics
def test_metrics_endpoint_counts_requests(tmp_path):
    m = CostGuardMetrics()
    eng = _engine(tmp_path, [RequestLogger(tmp_path / "log.sqlite"), MetricsHook(m)])
    client = TestClient(create_app(eng))
    for q in ("Where is my order?", "How do I return a jacket?", "Where is my order?"):
        body = {"messages": [{"role": "user", "content": q}], "costguard": {"mode": "balanced", "category": "order"}}
        assert client.post("/v1/chat/completions", json=body).status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    assert _metric_sum(text, "costguard_requests_total") == 3
    assert _metric_sum(text, "costguard_requests_total", mode="balanced", category="order") == 3
    assert _metric_sum(text, "costguard_latency_ms_count") == 3
    assert _metric_sum(text, "costguard_overhead_ms_count") == 3
    assert _metric_sum(text, "costguard_input_tokens_count", kind="original") == 3
    assert _metric_sum(text, "costguard_baseline_cost_usd_total") > 0
    assert _metric_sum(text, "costguard_build_info", backend="mock") == 1
    assert client.get("/v1/drift").json()["status"] == "disabled"   # no drift hook on this engine


def test_metrics_hook_labels_are_clamped_and_negative_savings_counted():
    m = CostGuardMetrics()
    h = MetricsHook(m)
    h(_rec(category="Some Unknown Thing", mode="weird", saved_usd=-0.001, cache_similarity=0.93,
           cache_status="semantic", model_used="", stage_errors={"semantic_cache": "boom"}, compression_ratio=2.0))
    text = m.render().decode()
    assert _metric_sum(text, "costguard_requests_total", category="other", mode="other", model_used="cache") == 1
    assert _metric_sum(text, "costguard_negative_savings_total") == 1
    assert _metric_sum(text, "costguard_stage_errors_total", stage="semantic_cache") == 1
    assert _metric_sum(text, "costguard_cache_similarity_count", outcome="hit") == 1
    assert _metric_sum(text, "costguard_compression_ratio_count") == 1


# ---------------------------------------------------------------- hooks never raise
def test_build_hooks_default_has_metrics_and_drift_only(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    hooks = build_hooks(Settings(), None)
    assert [h.name for h in hooks] == ["metrics", "drift"]
    assert all(isinstance(h, SafeHook) for h in hooks)


def test_hooks_never_raise_with_broken_langfuse_config(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-broken")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-broken")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")     # nothing listens there
    monkeypatch.setenv("COSTGUARD_DRIFT_BASELINE", "/nonexistent/baseline.json")
    hooks = build_hooks(Settings(), None)                           # must not raise, langfuse installed or not
    assert {"metrics", "drift"} <= {h.name for h in hooks}
    for h in hooks:
        h(_rec())
        h(None)                 # garbage in: still no exception
        h(object())
    for h in hooks:             # stop any background worker promptly
        close = getattr(h.inner, "close", None)
        if close:
            close(timeout=1)


def test_langfuse_missing_package_degrades_to_none(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setitem(sys.modules, "langfuse", None)              # makes `import langfuse` raise ImportError
    assert build_langfuse_hook() is None


def test_langfuse_hook_is_async_and_survives_failing_client():
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("langfuse is down")

    hook = LangfuseHook(client=Exploding(), max_queue=5)
    t0 = time.perf_counter()
    for _ in range(50):
        hook(_rec())                                                # never blocks, never raises
    assert (time.perf_counter() - t0) < 0.5
    hook.close(timeout=2)
    assert hook.enqueued + hook.dropped == 50 and hook.sent == 0

    bad_factory = LangfuseHook(client_factory=lambda: (_ for _ in ()).throw(ValueError("bad host")))
    bad_factory(_rec())
    bad_factory.close(timeout=2)
    assert bad_factory.disabled_reason and "bad host" in bad_factory.disabled_reason


def test_send_trace_uses_v3_api_with_otel_genai_names():
    calls = []

    class Obs:
        def __init__(self, kind, kw):
            calls.append((kind, kw))

        def start_observation(self, **kw):
            return Obs(kw.get("as_type"), kw)

        def update_trace(self, **kw):
            calls.append(("trace", kw))

        def end(self):
            calls.append(("end", {}))

    class Client:
        def start_observation(self, **kw):
            return Obs("root", kw)

    send_trace(Client(), _rec(cached_input_tokens=30))
    kinds = [k for k, _ in calls]
    assert kinds.count("generation") == 1 and kinds.count("end") == 2 and "trace" in kinds
    gen = next(kw for k, kw in calls if k == "generation")
    assert gen["usage_details"] == {"input": 70, "output": 40, "cache_read_input_tokens": 30}
    attrs = otel_attributes(_rec(cached_input_tokens=30))
    assert attrs["gen_ai.usage.input_tokens"] == 100 and attrs["gen_ai.usage.cache_read.input_tokens"] == 30
    assert attrs["costguard.cost_usd"] == pytest.approx(0.0002) and attrs["costguard.cache.status"] == "miss"
    assert None not in attrs.values()


def test_engine_keeps_serving_when_every_hook_explodes(tmp_path):
    def boom(rec):
        raise RuntimeError("hook exploded")

    eng = _engine(tmp_path, [SafeHook(boom, "boom"), boom])
    client = TestClient(create_app(eng))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


# ---------------------------------------------------------------- drift / PSI
def test_psi_course_example_one_bin_20_to_30_percent():
    assert psi_term(0.20, 0.30) == pytest.approx(0.1 * math.log(1.5)) == pytest.approx(0.041, abs=5e-4)
    # one bin 20% -> 30% (and another 30% -> 20% to keep shares summing to 1): two equal contributions
    assert psi([0.2, 0.3, 0.5], [0.3, 0.2, 0.5]) == pytest.approx(2 * 0.0405465, rel=1e-4)
    assert psi([1, 1, 1], [1, 1, 1]) == 0.0
    assert psi([20, 30, 50], [0.2, 0.3, 0.5]) == pytest.approx(0.0)          # counts or shares


def test_psi_bands_match_course():
    assert psi_band(0.05) == "stable"
    assert psi_band(0.10) == "investigate" and psi_band(0.25) == "investigate"
    assert psi_band(0.2501) == "alert"
    assert psi_band(None) == "insufficient_data"


def test_numeric_bins_are_open_ended():
    edges = quantile_edges(list(range(100)), 10)
    assert len(edges) == 9
    counts = bin_numeric([-1e9, 1e9, 50], edges)
    assert sum(counts) == 3 and counts[0] == 1 and counts[-1] == 1     # out-of-range values are not dropped


def test_drift_monitor_flags_input_shift_but_not_stable_mix():
    mon = DriftMonitor(window=200, baseline_size=200, min_samples=50)
    cats = ["order", "returns", "refund", "shipping"]
    for i in range(200):
        mon.observe(_rec(category=cats[i % 4], input_tokens_original=80 + (i % 20)))
    assert mon.report()["status"] == "ok"
    for i in range(200):   # same category mix, inputs ~10x longer (users pasting documents)
        mon.observe(_rec(category=cats[i % 4], input_tokens_original=900 + (i % 50)))
    rep = mon.report()
    assert rep["features"]["input_tokens"]["band"] == "alert" and rep["features"]["input_tokens"]["psi"] > 0.25
    assert rep["features"]["category"]["band"] == "stable"
    assert rep["features"]["cache_similarity"]["band"] == "insufficient_data"
    assert rep["overall"] == "alert"


def test_drift_baseline_snapshot_roundtrip_and_offline_compare():
    rows = [{"input_tokens_original": 50 + i % 10, "category": "order", "cache_status": "miss", "model_used": "strong",
             "output_tokens": 30, "cache_similarity": 0.8} for i in range(100)]
    snap = DriftMonitor().fit_rows(rows)
    mon = DriftMonitor(baseline=snap, min_samples=10)
    for r in rows:
        mon.observe(r)
    assert all(f["band"] == "stable" for f in mon.report()["features"].values())
    shifted = [dict(r, category="refund", cache_status="exact") for r in rows]
    rep = drift_mod.compare_rows(rows, shifted, min_samples=10)
    assert rep["features"]["category"]["band"] == "alert" and rep["features"]["route"]["band"] == "alert"


def test_drift_endpoint_and_gauges(tmp_path):
    m = CostGuardMetrics()
    mon = DriftMonitor(window=50, baseline_size=10, min_samples=5)
    eng = _engine(tmp_path, [MetricsHook(m), DriftHook(mon, metrics=m, recompute_every=5)])
    client = TestClient(create_app(eng))
    for i in range(15):
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": f"order {i % 3}"}],
                                                  "costguard": {"category": "order"}})
    rep = client.get("/v1/drift").json()
    assert rep["status"] == "ok" and set(rep["features"]) >= {"input_tokens", "category", "route"}
    assert rep["features"]["category"]["band"] == "stable"
    text = client.get("/metrics").text
    assert _metric_sum(text, "costguard_drift_status", feature="category") == 0
    assert _metric_sum(text, "costguard_drift_window_requests") == 15
    assert client.post("/v1/drift/baseline").json()["baseline_source"] == "window"


def test_drift_rebaseline_auth(tmp_path, monkeypatch):
    """POST /v1/drift/baseline: open in keyless dev mode, token-checked when a token is set, closed when caller keys
    are configured without an admin token (a shared deployment must not let any caller mask a drift alert)."""
    eng = _engine(tmp_path, [DriftHook(DriftMonitor(window=50, baseline_size=10, min_samples=5))])
    client = TestClient(create_app(eng))
    monkeypatch.delenv("COSTGUARD_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("COSTGUARD_API_KEYS", raising=False)
    assert client.post("/v1/drift/baseline").status_code == 200
    monkeypatch.setenv("COSTGUARD_API_KEYS", "k1:shopnest-support")
    assert client.post("/v1/drift/baseline").status_code == 403
    monkeypatch.setenv("COSTGUARD_ADMIN_TOKEN", "s3cret")
    assert client.post("/v1/drift/baseline").status_code == 403
    assert client.post("/v1/drift/baseline", headers={"x-costguard-admin-token": "wrong"}).status_code == 403
    assert client.post("/v1/drift/baseline", headers={"x-costguard-admin-token": "s3cret"}).status_code == 200


# ---------------------------------------------------------------- dashboard
def _load_dashboard():
    spec = importlib.util.spec_from_file_location("costguard_dashboard_app", ROOT / "dashboard" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dashboard_imports_without_running_streamlit(tmp_path):
    had_streamlit = "streamlit" in sys.modules
    dash = _load_dashboard()
    assert callable(dash.main)
    if not had_streamlit:
        assert "streamlit" not in sys.modules          # UI library is only imported inside main()
    # pure data helpers work on a real log and on missing files
    db = tmp_path / "log.sqlite"
    log = RequestLogger(db)
    eng = _engine(tmp_path, [log])
    from costguard.schemas import ChatMessage, ChatRequest, CostGuardOptions
    for q in ("Where is my order?", "Refund status?"):
        eng.handle(ChatRequest(messages=[ChatMessage(role="user", content=q)], costguard=CostGuardOptions(mode="off")))
    log.flush()                                         # logger writes off the request path
    df = dash.load_log(db)
    assert len(df) == 2
    s = dash.summarize(df)
    assert s["requests"] == 2 and s["saved_pct"] == pytest.approx(0, abs=1e-6)
    assert dash.load_json(tmp_path / "missing.json") is None
    assert dash.load_log(tmp_path / "missing.sqlite") is None
