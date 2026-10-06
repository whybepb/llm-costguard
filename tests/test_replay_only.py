"""Replay-only eval runs (E2E verification 2): reproducing a committed local-model run must need no key, no MLX and
never generate, and a substituted tier must never be scored as the requested one. Fake providers only, no keys."""
from __future__ import annotations

import json

import pytest

from costguard import factory
from costguard.config import ROOT, Settings, load_policy
from costguard.factory import build_engine
from costguard.pipeline import build_messages
from costguard.providers.cassette import CassetteMiss, call_key
from costguard.schemas import ChatMessage
from costguard.tokens import count_messages
from eval import gate_router, run_ab

POLICY = load_policy(ROOT / "configs" / "policy.yaml")
STRONG, CHEAP = POLICY.model_id("mlx", "strong"), POLICY.model_id("mlx", "cheap")
MAX = POLICY.default_max_tokens


def _entry(msgs, model, text, n_in, n_out=7):
    return {"key": call_key(msgs, model, MAX, 0.0), "model": model,
            "completion": {"text": text, "model": model, "latency_ms": 5.0, "finish_reason": "stop", "raw": {},
                           "usage": {"input_tokens": n_in, "output_tokens": n_out, "cached_input_tokens": 0,
                                     "cache_write_tokens": 0}}}


def _write(path, entries):
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


@pytest.fixture
def no_real_upstream(monkeypatch):
    """Fail the test if anything builds the real upstream provider (MLXProvider would load model weights)."""
    def boom(settings):
        raise AssertionError(f"real {settings.backend} provider built during a replay-only run")
    monkeypatch.setattr(factory, "make_provider", boom)
    monkeypatch.setattr("costguard.providers.make_provider", boom)


def test_keyless_replay_counts_tokens_from_the_cassette_on_local_backends(tmp_path):
    msgs = build_messages(POLICY.system_prompt, [], "Where is my order?", "")
    cas = _write(tmp_path / "c.jsonl", [_entry(msgs, STRONG, "strong answer", 321)])
    prov = run_ab.keyless_provider("mlx", cas, "replay", MAX)
    assert prov.count_tokens(msgs, STRONG) == 321                      # recorded usage, not an o200k estimate
    other = build_messages(POLICY.system_prompt, [], "Something never recorded", "")
    assert prov.count_tokens(other, STRONG) == count_messages(other)   # fallback: estimate
    assert prov.complete(msgs, STRONG, MAX, 0.0).text == "strong answer"
    with pytest.raises(CassetteMiss):                                  # a miss never generates
        prov.complete(other, STRONG, MAX, 0.0)
    # paid backends keep the adapter's estimate (their live pre-call count is an estimate too)
    assert run_ab.keyless_provider("anthropic", cas, "replay", MAX).count_tokens(msgs, STRONG) == round(
        count_messages(msgs) * 1.15)


def test_gate_router_replay_scores_the_real_cheap_answer_without_the_real_upstream(tmp_path, no_real_upstream):
    it = {"id": "x1", "category": "order", "query": "Where is my order?", "context": []}
    msgs = gate_router.serving_messages(POLICY, it)
    cas = _write(tmp_path / "gate.jsonl", [_entry(msgs, STRONG, "strong answer", 321),
                                           _entry(msgs, CHEAP, "cheap answer", 330)])
    settings = Settings(backend="mlx", cassette=cas, cassette_mode="replay")
    eng = gate_router.make_engine(settings, replay=True)
    st, ch = gate_router.generate(eng, it, "strong"), gate_router.generate(eng, it, "cheap")
    assert (st["text"], ch["text"]) == ("strong answer", "cheap answer")
    assert st["input_tokens_original"] == ch["input_tokens_original"] == 321     # exact, from the strong call
    assert ch["model"] == CHEAP


def test_gate_router_refuses_a_cheap_answer_that_the_pipeline_served_on_strong(tmp_path):
    """Before the fix a replay without mlx-lm failed the token count, the pipeline failed safe to strong, and the
    gate compared strong with strong: every category showed diff 0.0 [0.0, 0.0]."""
    eng = build_engine(Settings(backend="mock", db_path=tmp_path / "x.sqlite"), with_logger=False,
                       skip=gate_router.SKIP)

    class CannotCount:
        name = "mock"

        def __init__(self, inner):
            self.inner = inner

        def count_tokens(self, messages, model):
            raise ImportError("No module named 'mlx_lm'")

        def complete(self, *a):
            return self.inner.complete(*a)

    eng.provider = CannotCount(eng.provider)
    it = {"id": "x1", "category": "order", "query": "Where is my order?", "context": []}
    with pytest.raises(RuntimeError, match="asked for the cheap tier but the pipeline served 'strong'"):
        gate_router.generate(eng, it, "cheap")


def test_run_ab_replay_never_builds_the_upstream_and_reproduces_counts(tmp_path, monkeypatch, no_real_upstream):
    monkeypatch.setenv("COSTGUARD_JUDGE_CASSETTE_MODE", "auto")          # --replay sets it; restored after
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    rows = [{"pos": i, "item_id": f"i{i}", "cluster_id": f"c{i % 3}", "query": q, "category": "order", "context": [],
             "reference": "r", "source": "bitext", "dup_of": None}
            for i, q in enumerate(["where is order 1", "where is order 2", "cancel order 3", "where is order 1"])]
    trace = tmp_path / "t.jsonl"
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    recorded = {r["query"]: 400 + i for i, r in enumerate(rows[:3])}
    cas = _write(tmp_path / "mlx_ab.jsonl", [
        _entry(build_messages(POLICY.system_prompt, [], q, ""), STRONG, f"answer to {q}", n) for q, n in recorded.items()])
    out = tmp_path / "s.json"
    argv = ["--backend", "mlx", "--replay", "--trace", str(trace), "--arms", "A0,A1", "--judge", "heuristic",
            "--cassette", str(cas), "--out", str(out), "--out-dir", str(tmp_path)]
    assert run_ab.main(argv) == 0
    s = json.loads(out.read_text())
    a0, a1 = s["arms"]["A0"], s["arms"]["A1"]
    assert s["meta"]["replay_only"] is True and a0["complete"] and a1["complete"]
    assert "stage_errors" not in a0 and a0["route_reasons"] == {"router-off": 4}
    assert a0["tokens"]["input_original"] == a0["tokens"]["input_sent"] == 400 + 401 + 402 + 400
    assert a1["hits"] == {"exact": 1} and a1["savings_pct"] > 0
    # a request missing from the cassette fails (arm incomplete) instead of generating
    cas.write_text("".join(cas.read_text().splitlines(keepends=True)[:2]))
    assert run_ab.main(argv) == 0
    s = json.loads(out.read_text())
    assert s["arms"]["A0"]["n_errors"] == 1 and s["arms"]["A0"]["complete"] is False


def test_run_ab_replay_needs_a_cassette(tmp_path):
    with pytest.raises(SystemExit):
        run_ab.main(["--backend", "mock", "--replay", "--out-dir", str(tmp_path)])


def test_ci_gate_record_estimate_needs_no_key_and_calls_nothing(tmp_path, monkeypatch, no_real_upstream, capsys):
    from eval import build_trace as bt
    from eval import ci_gate
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    monkeypatch.setattr(ci_gate, "CASSETTES", tmp_path)                 # empty: every call is new
    est = ci_gate.estimate_record(bt.CI_SUBSET, "anthropic")
    assert est["new_generations"] > 0 and est["est_usd"] > 0
    assert set(est["by_model"]) <= {"claude-sonnet-5-5", "claude-haiku-4-5-20251001"}
    assert ci_gate.main(["--record", "--backend", "anthropic", "--estimate-only"]) == 0
    assert "NEW generations" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())                                 # nothing recorded
    with pytest.raises(SystemExit):                                     # an estimate is for a real backend
        ci_gate.main(["--estimate-only"])
