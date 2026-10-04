"""Evaluation machinery: stats, judge, trace builder, A/B runner, CI gate, report. Mock backend only, no keys."""
from __future__ import annotations

import json
import re
import sqlite3

import numpy as np
import pytest

from costguard.config import Settings
from costguard.pipeline import normalize_query
from costguard.schemas import Completion, SemanticHit, Usage
from eval import build_trace as bt
from eval import ci_gate, report, run_ab
from eval.judge import HeuristicJudge, Judge, human_agreement, parse_grade, parse_verdict
from eval.stats import cohen_kappa, mcnemar, paired_bootstrap, proportion_ci, ratio_bootstrap


# ------------------------------------------------------------------------------------------- stats
def test_paired_bootstrap_ci_sanity():
    rng = np.random.default_rng(1)
    d = rng.normal(1.0, 1.0, 400).tolist()
    m, lo, hi = paired_bootstrap(d, n=2000, seed=0)
    assert lo < m < hi and lo < 1.0 < hi
    assert 0.1 < hi - lo < 0.35                       # ~ 2 * 1.96 / sqrt(400) = 0.196
    assert paired_bootstrap(d, seed=0) == (m, lo, hi)  # seeded
    assert paired_bootstrap([0.5] * 10) == (0.5, 0.5, 0.5)
    assert all(np.isnan(x) for x in paired_bootstrap([]))
    # duplicates are correlated: resampling clusters must give a wider interval than resampling requests
    base = rng.normal(0, 1, 40)
    diffs, clusters = np.repeat(base, 10).tolist(), np.repeat(np.arange(40), 10).tolist()
    _, l_c, h_c = paired_bootstrap(diffs, clusters=clusters)
    _, l_i, h_i = paired_bootstrap(diffs)
    assert (h_c - l_c) > 2 * (h_i - l_i)
    r, rlo, rhi = ratio_bootstrap([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert r == pytest.approx(0.5) and rlo == pytest.approx(0.5) and rhi == pytest.approx(0.5)


def test_wilson_and_friends():
    p, lo, hi = proportion_ci(5, 10)
    assert p == 0.5 and lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    p, lo, hi = proportion_ci(0, 10)
    assert p == 0 and lo == 0 and hi == pytest.approx(0.2775, abs=1e-3)
    assert proportion_ci(10, 10)[2] == 1.0 and proportion_ci(10, 10)[1] == pytest.approx(0.7225, abs=1e-3)
    assert proportion_ci(0, 0)[1:] == (0.0, 1.0)
    assert mcnemar(0, 0) == (0.0, 1.0) and mcnemar(10, 0)[1] < 0.01 and mcnemar(30, 28)[1] > 0.5
    assert cohen_kappa(list("ABAB"), list("ABAB")) == 1.0 and cohen_kappa(list("AABB"), list("ABAB")) == 0.0


# ------------------------------------------------------------------------------------------- judge
REF = "Apparel can be returned within 30 days of delivery, unused and with tags."
GOOD = "You can return apparel within 30 days of delivery if it is unused with tags."
BAD = "Please contact support about shipping options for your parcel."


def test_heuristic_judge_position_swap():
    j = HeuristicJudge()
    assert j.label == "heuristic-mock"
    assert j.pairwise("return window?", GOOD, BAD, REF) == "A"
    assert j.pairwise("return window?", BAD, GOOD, REF) == "B"
    assert j.pairwise("q", GOOD, GOOD, REF) == "tie"                      # identical: not judged
    # two different answers with the same score: the first-position bonus wins both orders -> inconsistent -> tie
    d = j.pairwise_detail("q", "apparel returned", "returned apparel", REF)
    assert d["order1"] == "A" and d["order2"] == "B" and d["verdict"] == "tie"
    assert j.stats["position_inconsistent"] >= 1
    assert 0 <= j.grade("q", BAD, REF) < j.grade("q", GOOD, REF) <= 1


def test_disagreement_between_orders_is_a_tie():
    class FirstAlways(HeuristicJudge):           # pure position bias: always prefers whatever is shown first
        def _verdict(self, question, first, second, reference):
            return "A"
    assert FirstAlways().pairwise("q", GOOD, BAD, REF) == "tie"


class ScriptedProvider:
    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def count_tokens(self, messages, model):
        return 10

    def complete(self, messages, model, max_tokens, temperature):
        self.calls.append((messages, model, max_tokens, temperature))
        return Completion(text=self.replies.pop(0), model=model, usage=Usage(input_tokens=100, output_tokens=1),
                          latency_ms=1.0)


def test_model_judge_parsing_retry_and_swap():
    assert parse_verdict("**B**") == "B" and parse_verdict("TIE.") == "TIE" and parse_verdict("Answer A") == "A"
    assert parse_verdict("neither") is None and parse_grade("Score: 4/5") == 4 and parse_grade("great") is None
    p = ScriptedProvider(["4"])
    assert Judge(p, model="m").grade("q", "a", "r") == 0.75 and p.calls[0][3] == 0.0   # temperature 0
    p = ScriptedProvider(["no idea", "5"])                       # parse failure -> one retry
    assert Judge(p, model="m").grade("q", "a2", "r") == 1.0 and len(p.calls) == 2
    p = ScriptedProvider(["no idea", "still no"])                # two failures -> None
    j = Judge(p, model="m")
    assert j.grade("q", "a3", "r") is None and j.stats["parse_failures"] == 2
    assert Judge(ScriptedProvider(["A", "B"]), model="m").pairwise("q", "x", "y") == "A"    # consistent
    assert Judge(ScriptedProvider(["A", "A"]), model="m").pairwise("q", "x", "y") == "tie"  # position bias
    assert Judge(ScriptedProvider(["B", "A"]), model="m").pairwise("q", "x", "y") == "B"


def test_human_agreement(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [{"question": "q", "answer_a": GOOD, "answer_b": BAD, "reference": REF, "human": "A"},
            {"question": "q", "answer_a": BAD, "answer_b": GOOD, "reference": REF, "human": "B"},
            {"question": "q", "answer_a": "apparel returned", "answer_b": "returned apparel", "reference": REF,
             "human": "tie"},
            {"question": "q", "answer": GOOD, "reference": REF, "human_score": 5}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = human_agreement(path, HeuristicJudge())
    assert res["n_pairwise"] == 3 and res["agreement"] == 1.0 and res["n_grade"] == 1
    assert human_agreement(tmp_path / "missing.jsonl") is None


# ------------------------------------------------------------------------------------------- eval set + trace
def test_seed_evalset_is_valid():
    rows = bt.load_evalset()
    seed = [r for r in rows if r["author"] == "seed"]
    assert len(seed) >= 40 and not bt.validate_evalset(rows)
    assert {r["category"] for r in seed} == set(bt.CATEGORIES)
    assert sum(r["type"] == "trap_pair" for r in seed) >= 12 and any(r["type"] == "hard" for r in seed)
    assert bt.validate_evalset([{"id": "x", "category": "nope", "query": "q", "reference": "r", "type": "trap_pair",
                                 "author": "a"}])


def _synthetic_bitext() -> list[dict]:
    pre = ["", "please", "i need to", "help me", "can you", "how do i"]
    post = ["", "now", "asap", "today", "thanks"]
    rows = []
    for intent, verb in (("cancel_order", "cancel"), ("track_order", "track"), ("change_order", "change")):
        for i, (a, b) in enumerate((a, b) for a in pre for b in post):
            rows.append({"instruction": f"{a} {verb} order {{{{Order Number}}}} {b}".strip(), "intent": intent,
                         "category": "ORDER", "response": f"To {verb} order {{{{Order Number}}}} open "
                                                          f"{{{{Online Order Interaction}}}}. ({i})"})
    for intent, words in (("check_refund_policy", "refund policy"), ("delivery_period", "delivery time"),
                          ("complaint", "file a complaint"), ("place_order", "buy an item")):
        for a in pre:
            for b in post:
                rows.append({"instruction": f"{a} {words} {b}".strip(), "intent": intent, "category": "X",
                             "response": f"Here is how: {words}."})
    for intent in ("create_account", "delete_account"):
        for a in pre:
            for b in post:
                rows.append({"instruction": f"{a} {intent.split('_')[0]} {{{{Account Type}}}} account {b}".strip(),
                             "intent": intent, "category": "ACCOUNT", "response": "Use {{Account Type}} settings."})
    return rows


def _fake_kb():
    qs = [{"id": f"kbq-{i}", "query": f"What is the policy number {i} about {w}?", "reference": f"Policy {i}.",
           "category": "returns", "paraphrases": [f"Tell me policy {i} on {w}"]}
          for i, w in enumerate(["returns", "refunds", "shipping", "warranty", "payments", "accounts"])]
    return qs, (lambda q, k=8: [f"doc about {q[:20]}", "generic boilerplate doc"]), "fake kb"


def test_trace_builder_deterministic_and_duplicate_rate():
    bx, ev, kb = _synthetic_bitext(), bt.load_evalset(), _fake_kb()
    rows, st = bt.build_trace(200, 0.3, 7, bitext=bx, evalset=ev, kb=kb)
    again, _ = bt.build_trace(200, 0.3, 7, bitext=bx, evalset=ev, kb=kb)
    other, _ = bt.build_trace(200, 0.3, 8, bitext=bx, evalset=ev, kb=kb)
    assert rows == again and rows != other
    assert len(rows) == 200 and [r["pos"] for r in rows] == list(range(200))
    assert abs(st["dup_rate"] - 0.3) <= 0.02
    assert {"bitext", "kb", "trap"} <= set(st["sources"]) and abs(st["sources"]["trap"] / 200 - 0.10) <= 0.02
    for r in rows:
        if r["dup_of"] is not None:
            src = rows[r["dup_of"]]
            assert src["pos"] < r["pos"] and src["cluster_id"] == r["cluster_id"]
            if r["dup_kind"] in ("exact", "surface"):
                assert normalize_query(src["query"]) == normalize_query(r["query"]) and src["context"] == r["context"]
    pairs: dict = {}
    for r in rows:
        if r["source"] == "trap":
            pairs.setdefault(r["pair_id"], []).append(r)
    assert pairs and all(len(p) == 2 and p[0]["cluster_id"] != p[1]["cluster_id"] and abs(p[0]["pos"] - p[1]["pos"]) <= 8
                         for p in pairs.values())
    assert all(r["category"] in bt.CATEGORIES for r in rows)
    zero, st0 = bt.build_trace(200, 0.0, 7, bitext=bx, evalset=ev, kb=kb)
    assert st0["dup_rate"] == 0 and len({r["cluster_id"] for r in zero}) == 200
    assert "{{" not in json.dumps(rows)


@pytest.mark.skipif(not (bt.RAW / "bitext.parquet").exists(), reason="Bitext not cached locally (eval/data/raw)")
def test_trace_builder_on_real_bitext_small():
    rows, st = bt.build_trace(120, 0.3, 7, kb=([], None, "skipped in test"))
    assert st["n"] == 120 and abs(st["dup_rate"] - 0.3) <= 0.02 and st["sources"].get("kb", 0) == 0


def test_committed_trace_and_ci_subset_match_their_hashes():
    if not bt.TRACE.exists():
        pytest.skip("trace not built")
    sha = bt.TRACE.with_suffix(".sha256").read_text().split()[0]
    assert bt.file_sha256(bt.TRACE) == sha
    sub = [json.loads(x) for x in bt.CI_SUBSET.read_text().splitlines() if x.strip()]
    pos: dict = {}
    for r in sub:
        if r.get("type") == "trap_pair":
            pos.setdefault(r["pair_id"], []).append(r["pos"])
    assert len(pos) >= 15 and all(len(p) == 2 and p[1] - p[0] == 1 for p in pos.values())   # pair members adjacent


# ------------------------------------------------------------------------------------------- A/B + report
def test_run_ab_mock_produces_all_arms(tmp_path, monkeypatch):
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    trace = tmp_path / "trace_small.jsonl"
    rows, _ = bt.build_trace(60, 0.3, 7, bitext=_synthetic_bitext(), evalset=bt.load_evalset(), kb=_fake_kb())
    bt.write_jsonl(rows, trace)
    assert run_ab.main(["--backend", "mock", "--limit", "30", "--trace", str(trace), "--out-dir", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "ab_summary_trace_small_mock_limit30.json").read_text())
    assert list(s["arms"]) == run_ab.CUMULATIVE and len(s["waterfall"]) == 5
    for arm, a in s["arms"].items():
        assert a["n"] == 30 and a["n_errors"] == 0 and a["quality"]["n_graded"] == 30
        assert a["savings_pct"] is not None and a["latency_ms"]["e2e"]["p50"] is not None
    assert s["arms"]["A0"]["savings_pct"] == 0 and s["arms"]["A0"]["hit_rate"]["total"] == 0
    assert s["meta"]["judge"]["judge"] == "heuristic-mock" and s["meta"]["trace_sha256"]
    with sqlite3.connect(tmp_path / "ab_mock_trace_small_limit30.sqlite") as c:
        assert c.execute("SELECT COUNT(*), COUNT(DISTINCT arm) FROM requests").fetchone() == (180, 6)
    monkeypatch.setattr(report, "RESULTS", tmp_path)
    md = report.build()
    assert "| **A5** |" in md and "MOCK BACKEND" in md


def test_run_ab_refuses_paid_backend_without_yes(tmp_path, monkeypatch):
    trace = tmp_path / "t.jsonl"
    rows, _ = bt.build_trace(40, 0.3, 7, bitext=_synthetic_bitext(), evalset=bt.load_evalset(), kb=_fake_kb())
    bt.write_jsonl(rows, trace)
    monkeypatch.delenv("COSTGUARD_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(run_ab, "CASSETTES", tmp_path)
    args = ["--backend", "anthropic", "--trace", str(trace), "--limit", "5", "--arms", "A0,A1", "--out-dir", str(tmp_path)]
    assert run_ab.main(args + ["--estimate-only"]) == 0       # pre-flight works without a key
    assert run_ab.main(args) == 2                              # paid + new generations + no --yes -> refuse


# ------------------------------------------------------------------------------------------- CI gate
class JaccardSemanticCache:
    """Deterministic stand-in for the semantic cache: token Jaccard similarity, no guards."""

    def __init__(self):
        self.entries: list[tuple[str, str, object]] = []

    @staticmethod
    def _sim(a: str, b: str) -> float:
        x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
        return len(x & y) / max(1, len(x | y))

    def lookup(self, query, partition, threshold):
        cands = [(self._sim(query, q), q, e) for p, q, e in self.entries if p == partition]
        if not cands:
            return SemanticHit()
        s, q, e = max(cands, key=lambda c: c[0])
        return SemanticHit(entry=e if s >= threshold else None, similarity=round(s, 4), neighbor_query=q)

    def insert(self, query, partition, entry):
        self.entries.append((partition, query, entry))

    def clear(self):
        self.entries.clear()


def _gate_engine(tmp_path, fake_semantic: bool):
    from costguard.factory import build_engine
    settings = Settings(backend="mock", db_path=tmp_path / "ci.sqlite")
    if not fake_semantic:      # integration: every real stage component
        return build_engine(settings, with_logger=False)
    # hermetic: deterministic Jaccard cache, slow stages that don't affect cache hits skipped
    eng = build_engine(settings, with_logger=False, skip=("semantic_cache", "context_optimizer", "compressor"))
    eng.semantic = JaccardSemanticCache()
    return eng


@pytest.mark.parametrize("fake_semantic", [True, False], ids=["jaccard-cache", "real-components"])
def test_ci_gate_passes_on_baseline_and_fails_at_low_tau(tmp_path, monkeypatch, fake_semantic):
    if not fake_semantic:
        status = _gate_engine(tmp_path, False).component_status.get("semantic_cache", "")
        if any(w in status for w in ("not available", "failed", "skipped")):
            pytest.skip(f"semantic cache not built: {status}")
    monkeypatch.delenv("COSTGUARD_TAU_OVERRIDE", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    base = tmp_path / "baseline.json"
    kw = dict(subset=bt.CI_SUBSET, baseline=base, out=tmp_path / "gate.json", backend="mock")
    code, _ = ci_gate.run_gate(**kw, engine=_gate_engine(tmp_path, fake_semantic), update_baseline=True)
    assert code == 0 and base.exists()
    summary = tmp_path / "summary.md"
    code, res = ci_gate.run_gate(**kw, engine=_gate_engine(tmp_path, fake_semantic), step_summary=str(summary))
    assert code == 0 and res["passed"] and res["metrics"]["trap_false_hits"] == 0
    assert "PASS" in summary.read_text()
    monkeypatch.setenv("COSTGUARD_TAU_OVERRIDE", "0.5")
    code, res = ci_gate.run_gate(**kw, engine=_gate_engine(tmp_path, fake_semantic))
    assert code == 1 and not res["passed"] and res["metrics"]["tau"] == 0.5
    failed = {c["check"] for c in res["checks"] if c["status"] == "FAIL"}
    assert "(a) trap false hits" in failed or "(a') false hits, all rows" in failed
