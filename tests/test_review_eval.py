"""Regression tests for the Codex review findings on the eval harness (#1-#5). Mock / fake providers only, no keys."""
from __future__ import annotations

import json
import os
import threading
import types

import pytest

from costguard.config import Settings
from costguard.factory import build_engine
from costguard.providers.cassette import CassetteMiss, CassetteProvider, call_key
from costguard.schemas import ChatMessage, Completion, RouteDecision, Usage
from eval import build_trace as bt
from eval import ci_gate, gate_router, report, run_ab
from eval.judge import GRADE_PROMPT, JUDGE_ERROR, SYSTEM, _REF_BLOCK, Judge, human_agreement
from eval.run_ab import PROMPT_CACHE_ENV


class RaisingProvider:
    name = "fake"

    def count_tokens(self, messages, model):
        return 1

    def complete(self, messages, model, max_tokens, temperature):
        raise RuntimeError("upstream down")


class ScriptedProvider:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def count_tokens(self, messages, model):
        return 1

    def complete(self, messages, model, max_tokens, temperature):
        return Completion(text=self.replies.pop(0), model=model, usage=Usage(input_tokens=100, output_tokens=1),
                          latency_ms=1.0)


# ------------------------------------------------------------------------------------------- #2 judge failures
def test_failed_pairwise_judgment_is_an_error_not_a_tie(tmp_path):
    j = Judge(RaisingProvider(), model="m")
    d = j.pairwise_detail("q", "x", "y", "r")
    assert d["verdict"] == JUDGE_ERROR == "error" and d["order1"] is None and d["order2"] is None
    assert j.stats["pairwise_errors"] == 1 and j.stats["errors"] == 2
    # one order fine, the swapped order unparseable twice: still missing evidence, not a position-swap tie
    assert Judge(ScriptedProvider(["A", "??", "??"]), model="m").pairwise("q", "x", "y") == "error"
    # genuine position-swap disagreement is still a tie
    assert Judge(ScriptedProvider(["A", "A"]), model="m").pairwise("q", "x", "y") == "tie"
    # hand-label agreement skips failed judgments instead of scoring them as disagreements
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"question": "q", "answer_a": "x", "answer_b": "y", "human": "A"}) + "\n")
    assert human_agreement(path, Judge(RaisingProvider(), model="m"))["n_pairwise"] == 0


class _GateEngine:
    provider = None

    def handle(self, req):
        cheap = req.model == "cheap"
        comp = Completion(text=f"{req.model} answer", model=req.model, usage=Usage(), latency_ms=1.0)
        rec = types.SimpleNamespace(model_id=req.model, cost_usd=0.001 if cheap else 0.01, input_tokens_sent=50,
                                    input_tokens_original=50, output_tokens=10, cached_input_tokens=0)
        return comp, rec


class _StubJudge:
    """Equal grades and ties everywhere, except the listed queries, whose judgments fail."""
    label = "stub-judge"

    def __init__(self, failing=()):
        self.failing = set(failing)

    def grade(self, question, answer, reference=None):
        return None if question in self.failing else 0.75

    def pairwise(self, question, a, b, reference=None):
        return JUDGE_ERROR if question in self.failing else "tie"

    def describe(self):
        return {"judge": self.label}


def _run_gate_router(tmp_path, monkeypatch, judge, extra=(), backend="mlx", make_engine=None):
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps({"id": f"o{i}", "category": "order", "query": f"Where is my parcel number {i}",
                                        "reference": "It ships in 2 days."}) + "\n" for i in range(32)))
    monkeypatch.setattr(gate_router, "make_engine", make_engine or (lambda settings, replay: _GateEngine()))
    monkeypatch.setattr(gate_router, "make_judge", lambda kind, settings, engine: (judge, judge.label))
    return gate_router.run(gate_router.parse_args(
        ["--backend", backend, "--data", str(data), "--quiet", "--cassette", str(tmp_path / "gate.jsonl"),
         "--gate-out", str(tmp_path / "gate.json"), "--results-out", str(tmp_path / "rg.json"), *extra]))


def test_gate_router_never_allows_on_failed_judgments(tmp_path, monkeypatch):
    # every judgment failed: before the fix, 32 "ties" (diff 0, CI [0, 0]) allowed the category
    res = _run_gate_router(tmp_path, monkeypatch, Judge(RaisingProvider(), model="m"), ["--scorer", "pairwise"])
    o = res["categories"]["order"]
    assert not o["allow"] and o["n"] == 0 and o["reason"] == "judge-coverage 0.0% < 95%"
    assert o["judge_coverage"] == 0.0 and o["pairwise"]["errors"] == 32 and o["pairwise"]["non_inferior_rate"] is None
    assert res["errors"]["pairwise_missing"] == 32
    items = json.loads((tmp_path / "rg_items.json").read_text())["items"]
    assert {r["pairwise"] for r in items} == {"error"} and {r["diff_pairwise"] for r in items} == {None}


@pytest.mark.parametrize("n_failing, allowed", [(2, False), (1, True)])
def test_gate_router_requires_95pct_judge_coverage(tmp_path, monkeypatch, n_failing, allowed):
    failing = [f"Where is my parcel number {i}" for i in range(n_failing)]
    res = _run_gate_router(tmp_path, monkeypatch, _StubJudge(failing))      # scorer both, decision on grade
    o = res["categories"]["order"]
    assert o["n_routable"] == 32 and o["n"] == 32 - n_failing and o["n_unjudged"] == n_failing
    assert o["judge_coverage"] == round((32 - n_failing) / 32, 4)
    assert o["allow"] is allowed
    if not allowed:                                   # 30/32 = 93.75% < 95%, although n = 30 >= min_n
        assert o["reason"] == "judge-coverage 93.8% < 95%"
        assert any("judge-coverage" in w for w in res["warnings"])
    assert o["pairwise"]["errors"] == n_failing and o["pairwise"]["ties"] == 32 - n_failing
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert gate["min_judge_coverage"] == gate_router.MIN_JUDGE_COVERAGE == 0.95
    assert gate["categories"]["order"]["judge_coverage"] == o["judge_coverage"]


def test_run_ab_pairwise_errors_are_not_ties():
    rows = [{"pos": p, "query": f"q{p}", "reference": "r", "answer": f"base {p}"} for p in range(3)]
    arm = [{**r, "answer": f"arm {r['pos']}"} for r in rows]
    q = run_ab.judge_results(Judge(RaisingProvider(), model="m"), {"A0": rows, "A1": arm}, "pairwise", 10, 1)
    assert set(q["pairwise"]["A1"].values()) == {"error"} and q["mode"] == "pairwise"


# ------------------------------------------------------------------------------------------- #3 A/B coverage
_BASE = dict(cluster_id="c", intent="i", category="order", source="bitext", dup_kind=None, pair_id=None, query="q",
             reference="r", cost=0.01, est_baseline=0.01, in_orig=10, in_sent=10, out_tokens=5, cached_in=0,
             compression_ratio=None, docs_in=0, docs_kept=0, latency_ms=1.0, overhead_ms=1.0, upstream_ms=0.0,
             stage_errors={}, model_used="strong", route_reason="router-off", answer="a", false_hit=False,
             false_hit_intent=False, cache_guard=None, cache_status="miss")


def _row(pos, **kw):
    return {**_BASE, "pos": pos, "cluster_id": f"c{pos}", **kw}


def _failed(pos):
    return {"pos": pos, "cluster_id": f"c{pos}", "query": "q", "error": "RuntimeError: boom", "cost": 0.0,
            "cache_status": "error"}


def test_request_rates_divide_by_attempted_requests():
    hit = _row(0, cache_status="exact", cost=0.0, false_hit=True, false_hit_intent=True)
    s = run_ab.summarize_arm("A1", [hit, _failed(1)], None, None, n_boot=50)
    assert s["n"] == 1 and s["n_attempted"] == 2 and s["n_errors"] == 1
    assert s["hit_rate"]["total"] == 50.0 and s["false_hit_rate"] == 50.0 and s["false_hit_rate_intent"] == 50.0
    assert s["complete"] is False and s["coverage"] == {"attempted": 2, "failed": 1, "ungraded": None,
                                                        "pairwise_errors": 0}


def test_missing_grades_and_pairwise_errors_mark_the_arm_incomplete():
    a0 = [_row(p) for p in range(3)]
    arm = [_row(p, cost=0.005, answer=f"other {p}") for p in range(3)]
    q = {"grades": {"A0": {0: 1.0, 1: 1.0, 2: 1.0}, "A1": {0: 1.0, 1: None, 2: 1.0}},
         "pairwise": {"A1": {0: "win", 1: "error", 2: "tie"}}, "identical": {"A1": 0}, "mode": "both"}
    s = run_ab.summarize_arm("A1", arm, a0, q, n_boot=50)
    assert s["complete"] is False and s["coverage"]["ungraded"] == 1 and s["coverage"]["pairwise_errors"] == 1
    pw = s["quality"]["pairwise"]
    assert pw["judged"] == 2 and pw["errors"] == 1 and pw["tie"] == 1 and pw["win"] == 1
    assert pw["non_inferior_rate_judged"] == 100.0                      # 2 of 2 valid judgments, error excluded
    full = {**q, "grades": {"A0": q["grades"]["A0"], "A1": {0: 1.0, 1: 1.0, 2: 1.0}},
            "pairwise": {"A1": {0: "win", 1: "loss", 2: "tie"}}}
    assert run_ab.summarize_arm("A1", arm, a0, full, n_boot=50)["complete"] is True
    # pairwise-only judging: no grades were requested, so none are "missing"
    pw_only = {**full, "grades": {"A0": {}, "A1": {}}, "mode": "pairwise"}
    assert run_ab.summarize_arm("A1", arm, a0, pw_only, n_boot=50)["complete"] is True


def test_waterfall_compares_the_same_items_in_every_arm():
    results = {"A0": [_row(p, cost=0.01) for p in range(3)],
               "A1": [_row(0, cost=0.0, cache_status="exact"), _failed(1), _row(2, cost=0.01)],
               "A2": [_row(p, cost=0.0 if p == 1 else 0.004) for p in range(3)]}
    per_arm = {a: run_ab.summarize_arm(a, r, results["A0"], None, n_boot=50) for a, r in results.items()}
    w = run_ab.waterfall(per_arm, results)
    # items 0 and 2 succeeded everywhere: A0 0.02 -> A1 0.01 -> A2 0.008; item 1 (failed in A1) is excluded
    assert [(s["n_items"], s["n_excluded"]) for s in w] == [(2, 1), (2, 1)]
    assert [s["saved_usd"] for s in w] == [pytest.approx(0.01), pytest.approx(0.002)]
    assert [s["saved_pct_of_a0"] for s in w] == [50.0, 10.0]
    # without failures it equals the legacy whole-arm computation
    clean = {a: [_row(p, cost=c) for p in range(3)] for a, c in (("A0", 0.01), ("A1", 0.006))}
    summ = {a: run_ab.summarize_arm(a, r, clean["A0"], None, n_boot=50) for a, r in clean.items()}
    legacy = {k: v for k, v in run_ab.waterfall(summ)[0].items()}
    new = run_ab.waterfall(summ, clean)[0]
    assert {k: new[k] for k in legacy} == legacy and new["n_excluded"] == 0


def _summary(arms: dict) -> dict:
    meta = {"backend": "mlx", "limit": None, "judge": {"judge": "j"}, "trace": "eval/data/trace_v1.jsonl",
            "trace_sha256": "0" * 64, "rows_used": 3, "trace_rows_total": 3, "config_hash": "abc", "models": {},
            "billing": {}, "prices_checked_on": "x", "overrides": {}, "components": {}, "judge_mode": "both",
            "pairwise_max": 150, "correct_threshold": 0.75, "git_commit": None, "generated_at": "now"}
    return {"meta": meta, "arms": arms, "waterfall": run_ab.waterfall(arms)}


def test_report_flags_incomplete_arms_and_never_headlines_them(tmp_path, monkeypatch):
    a0 = [_row(p) for p in range(3)]
    q = {"grades": {a: {p: 1.0 for p in range(3)} for a in ("A0", "A4", "A5")}, "pairwise": {}, "identical": {},
         "mode": "grade"}
    q["grades"]["A5"] = {0: 1.0, 1: None, 2: None}
    arms = {"A0": run_ab.summarize_arm("A0", a0, a0, q, n_boot=50),
            "A4": run_ab.summarize_arm("A4", [_row(p, cost=0.006) for p in range(3)], a0, q, n_boot=50),
            "A5": run_ab.summarize_arm("A5", [_row(0, cost=0.001), _failed(1), _row(2, cost=0.001)], a0, q,
                                       n_boot=50)}
    assert arms["A5"]["complete"] is False and arms["A4"]["complete"] is True
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    (tmp_path / "ab_summary.json").write_text(json.dumps(_summary(arms)))
    md = report.build()
    assert "| **A5** **INCOMPLETE: 1 failed / 1 ungraded** |" in md
    assert "**Headline (A4," in md and "**Headline (A5," not in md
    assert "Not used as the headline" in md
    assert "**A5** **INCOMPLETE: 1 failed / 1 ungraded**" in report.readme_block()
    # an incomplete A0 (the paired baseline) leaves no headline at all
    arms["A0"] = run_ab.summarize_arm("A0", [_failed(0), *a0[1:]], a0, q, n_boot=50)
    (tmp_path / "ab_summary.json").write_text(json.dumps(_summary(arms)))
    md = report.build()
    assert "**Headline" not in md and "No headline" in md


# ------------------------------------------------------------------------------------------- #4 CI gate
class _CheapRouter:
    def route(self, rin, policy):
        return RouteDecision(alias="cheap", reason="stub-cheap")


class _CheapMisses:
    """A replay cassette holding every strong completion but no cheap one."""

    def __init__(self, inner, miss: bool):
        self.inner, self.miss, self.name = inner, miss, "mock+cassette"

    def count_tokens(self, messages, model):
        return self.inner.count_tokens(messages, model)

    def complete(self, messages, model, max_tokens, temperature):
        if self.miss and model == "mock-cheap":
            raise CassetteMiss(f"no cassette entry for model={model}")
        return self.inner.complete(messages, model, max_tokens, temperature)


def _ci_engine(tmp_path, miss: bool):
    eng = build_engine(Settings(backend="mock", db_path=tmp_path / "ci.sqlite"), with_logger=False,
                       skip=("semantic_cache", "context_optimizer", "compressor"))
    eng.router, eng.provider = _CheapRouter(), _CheapMisses(eng.provider, miss)
    return eng


def test_ci_gate_fails_on_a_cheap_tier_cassette_miss_hidden_by_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    kw = dict(subset=bt.CI_SUBSET, baseline=tmp_path / "baseline.json", out=tmp_path / "gate.json", backend="mock")
    code, _ = ci_gate.run_gate(**kw, engine=_ci_engine(tmp_path, miss=False), update_baseline=True)
    assert code == 0
    code, res = ci_gate.run_gate(**kw, engine=_ci_engine(tmp_path, miss=False))
    assert code == 0 and res["metrics"]["stage_cassette_misses"] == 0
    code, res = ci_gate.run_gate(**kw, engine=_ci_engine(tmp_path, miss=True))
    m = res["metrics"]
    assert m["errors"] == 0 and m["stage_cassette_misses"] > 0      # every request "succeeded" via fallback
    chk = next(c for c in res["checks"] if c["check"].startswith("replay complete"))
    assert code == 1 and chk["status"] == "FAIL" and chk["value"] == m["stage_cassette_misses"]
    assert "upstream_cheap: CassetteMiss" in chk["note"]


# ------------------------------------------------------------------------------------------- #5 parallel judging
def test_parallel_judging_counts_new_and_replayed_calls_per_call(tmp_path):
    started, release = threading.Event(), threading.Event()

    class SlowInner:
        name = "fake"

        def count_tokens(self, messages, model):
            return 1

        def complete(self, messages, model, max_tokens, temperature):
            started.set()
            release.wait(5)
            return Completion(text="4", model=model, usage=Usage(input_tokens=100, output_tokens=1), latency_ms=1.0)

    path = tmp_path / "judge.jsonl"
    cas = CassetteProvider(SlowInner(), path, "auto")
    j = Judge(cas, model="m")
    j._price = lambda comp: 1.0                                  # $1 per call: makes the split easy to read
    prompt = GRADE_PROMPT.format(question="q", ref_block=_REF_BLOCK.format(reference="r"), answer="old",
                                 agrees="agrees with the reference")
    key = call_key([ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=prompt)], "m", 5, 0.0)
    cas._store[key] = Completion(text="5", model="m", usage=Usage(input_tokens=100, output_tokens=1),
                                 latency_ms=1.0).model_dump()
    t = threading.Thread(target=lambda: j.grade("q", "new", "r"))   # new call, blocked inside the provider ...
    t.start()
    assert started.wait(5)
    assert j.grade("q", "old", "r") == 1.0                          # ... while a replay completes
    release.set()
    t.join(5)
    assert dict(j.stats) == {"grades": 2, "calls": 2, "new_calls": 1}
    assert j.cost_new_usd == 1.0 and j.cost_all_usd == 2.0
    assert (cas.hits, cas.misses) == (1, 1)
    # the per-call flag never leaks into the cassette file
    assert all("cassette" not in json.loads(x)["completion"]["raw"] for x in path.read_text().splitlines())
    assert cas.complete([ChatMessage(role="user", content="x")], "m", 5, 0.0).raw["cassette"] == "new"
    assert cas.complete([ChatMessage(role="user", content="x")], "m", 5, 0.0).raw["cassette"] == "replay"


# ------------------------------------------------------------------------------------------- #1 prompt caching
class _FakeAnthropicClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        usage = types.SimpleNamespace(input_tokens=10, output_tokens=2, cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0)
        return types.SimpleNamespace(id="m1", stop_reason="end_turn", usage=usage,
                                     content=[types.SimpleNamespace(type="text", text="hi")])


@pytest.mark.parametrize("enabled", [False, True])
def test_prompt_cache_switch_controls_the_anthropic_cache_breakpoint(monkeypatch, enabled):
    from costguard.providers.anthropic_provider import AnthropicProvider
    monkeypatch.setenv(PROMPT_CACHE_ENV, "unset")                # restored after the test
    assert run_ab.set_provider_prompt_cache(enabled) is enabled
    client = _FakeAnthropicClient()
    AnthropicProvider(client=client).complete([ChatMessage(role="system", content="S"),
                                               ChatMessage(role="user", content="q")], "claude", 5, 0.0)
    system = client.calls[0]["system"]
    assert (isinstance(system, list) and "cache_control" in system[0]) is enabled
    if not enabled:
        assert system == "S"


@pytest.mark.parametrize("flag", [False, True])
def test_run_ab_disables_provider_prompt_cache_on_anthropic_unless_flagged(tmp_path, monkeypatch, flag):
    monkeypatch.setenv(PROMPT_CACHE_ENV, "unset")
    monkeypatch.delenv("COSTGUARD_JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    monkeypatch.setattr(run_ab, "CASSETTES", tmp_path)
    trace = tmp_path / "t.jsonl"
    rows = [{"pos": i, "item_id": f"i{i}", "cluster_id": f"c{i}", "query": f"where is order {i}", "category": "order",
             "context": [], "reference": "r", "source": "bitext", "dup_of": None} for i in range(4)]
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))
    seen, real = [], run_ab.make_engine

    def fake_make_engine(settings, provider=None, with_logger=False):
        if provider is not None:                                 # pre-flight dry run: no real provider
            return real(settings, provider=provider, with_logger=with_logger)
        seen.append(os.environ.get(PROMPT_CACHE_ENV))           # the real provider would be built now
        return build_engine(Settings(backend="mock", db_path=settings.db_path), with_logger=with_logger,
                            skip=("semantic_cache", "context_optimizer", "compressor"))

    monkeypatch.setattr(run_ab, "make_engine", fake_make_engine)
    out = tmp_path / "s.json"
    argv = ["--backend", "anthropic", "--trace", str(trace), "--arms", "A0,A1", "--judge", "heuristic", "--yes",
            "--no-prefetch", "--out", str(out), "--out-dir", str(tmp_path)]
    argv += ["--provider-prompt-cache"] if flag else []
    assert run_ab.main(argv) == 0
    assert seen == ["1" if flag else "0"]
    s = json.loads(out.read_text())
    assert s["meta"]["provider_prompt_cache"] is flag and s["arms"]["A1"]["complete"] is True


def test_ci_gate_and_router_gate_record_the_prompt_cache_setting(tmp_path, monkeypatch):
    monkeypatch.setenv(PROMPT_CACHE_ENV, "unset")
    monkeypatch.setenv("COSTGUARD_JUDGE_CASSETTE_MODE", "auto")     # gate_router --replay sets it; restored after
    monkeypatch.delenv("COSTGUARD_JUDGE_BACKEND", raising=False)
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    seen = []

    def fake_make_engine(settings, provider=None, with_logger=False):
        seen.append(os.environ.get(PROMPT_CACHE_ENV))
        return _ci_engine(tmp_path, miss=False)

    monkeypatch.setattr(ci_gate, "make_engine", fake_make_engine)
    monkeypatch.setattr(ci_gate, "CASSETTES", tmp_path)
    code, res = ci_gate.run_gate(subset=bt.CI_SUBSET, baseline=tmp_path / "b.json", out=None, backend="anthropic",
                                 judge_kind="heuristic", record=True, update_baseline=True)
    assert code == 0 and seen == ["0"] and res["meta"]["provider_prompt_cache"] is False
    # router gate on anthropic (--replay: $0, the fake engine calls nothing)
    def gate_engine(settings, replay):
        seen.append(os.environ.get(PROMPT_CACHE_ENV))
        return _GateEngine()

    for flag, want in (([], False), (["--provider-prompt-cache"], True)):
        seen.clear()
        res = _run_gate_router(tmp_path, monkeypatch, _StubJudge(), ["--replay", *flag], backend="anthropic",
                               make_engine=gate_engine)
        assert seen == ["1" if want else "0"] and res["provider_prompt_cache"] is want
        assert json.loads((tmp_path / "gate.json").read_text())["provider_prompt_cache"] is want


def test_local_backends_never_touch_the_prompt_cache_switch(tmp_path, monkeypatch):
    """mlx / mock report no cached tokens; their runs leave the switch alone and record None."""
    monkeypatch.setenv(PROMPT_CACHE_ENV, "sentinel")
    res = _run_gate_router(tmp_path, monkeypatch, _StubJudge())
    assert res["provider_prompt_cache"] is None and os.environ[PROMPT_CACHE_ENV] == "sentinel"
