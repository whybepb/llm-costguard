"""LLM-as-judge for CostGuard's paired evaluation (contract: docs/CONTRACT.md "Shared eval APIs").

    judge = get_judge()                      # backend strong tier, temperature 0, cassette-backed
    judge.pairwise(q, a, b, reference)       # "A" | "B" | "tie"   (both orders; disagreement -> "tie")
                                             # or "error" if either order failed / was unparseable (never a tie)
    judge.grade(q, answer, reference)        # 0..1 (1-5 rubric scaled), or None if unparseable twice

Judge model: COSTGUARD_JUDGE_BACKEND / COSTGUARD_JUDGE_MODEL if set, else the engine backend's strong tier.
Every judge call goes through the cassette eval/cassettes/judge.jsonl, so re-judging is free and replayable
(CI replays it with no key). On the mock backend answers are meaningless, so a deterministic heuristic judge
(token overlap with the reference) is used instead; everything it produces is labelled "heuristic-mock".

Known judge biases (MT-Bench, arXiv 2306.05685) and what we do about them:
  position bias  -> every pairwise comparison runs in both orders; inconsistent verdicts count as a tie
  verbosity      -> the prompts tell the judge to ignore length and style
  self-preference-> a same-family judge (e.g. Sonnet grading Sonnet vs Haiku) can favour its own family;
                    mitigated by the position swap, reference-guided grading and the hand-label agreement check
                    (`python -m eval.judge agreement`), and removable by setting COSTGUARD_JUDGE_BACKEND/MODEL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
from collections import Counter
from pathlib import Path
from typing import Optional

from costguard.config import ROOT, Settings, load_policy
from costguard.pricing import PriceBook
from costguard.providers.cassette import CassetteProvider
from costguard.schemas import ChatMessage, Completion
from costguard.tokens import count_messages

from .stats import cohen_kappa, proportion_ci

log = logging.getLogger("costguard.eval.judge")

JUDGE_CASSETTE = ROOT / "eval" / "cassettes" / "judge.jsonl"
HUMAN_LABELS = ROOT / "eval" / "data" / "human_labels.jsonl"
HEURISTIC_LABEL = "heuristic-mock"
LOCAL_BACKENDS = ("mock", "mlx")          # everything else is a paid API
JUDGE_ERROR = "error"   # pairwise outcome when a judgment is missing: never a tie, never evidence of non-inferiority

SYSTEM = ("You are an impartial expert evaluator of customer-support answers for ShopNest, an online store selling "
          "electronics, home goods and apparel. Follow the instructions exactly.")

_REF_BLOCK = ("\n\n[Reference answer] (use it to check facts; a good answer may be worded differently or shorter)\n"
              "{reference}")

PAIRWISE_PROMPT = """Compare two answers to the same customer question.

[Customer question]
{question}{ref_block}

[Answer A]
{a}

[Answer B]
{b}

Which answer better and more correctly answers the customer's question{given_ref}? Judge factual correctness first \
(a claim that contradicts the reference is a serious error), then whether the answer actually resolves the question. \
Ignore length, tone and formatting, and do not favour an answer because of its position.
Reply with exactly one token: A, B or TIE."""

GRADE_PROMPT = """Grade the answer to the customer's question.

[Customer question]
{question}{ref_block}

[Answer to grade]
{answer}

Rubric:
5 = correct and complete: every key fact {agrees} and the question is fully resolved
4 = correct, with a minor omission or imprecision
3 = partly correct: misses a key fact, or too vague to act on
2 = mostly wrong or off-topic, or gets a key fact wrong
1 = wrong, unsafe, or does not address the question
Ignore length and style. Reply with a single digit from 1 to 5."""

RETRY_SUFFIX = {"pairwise": "\n\nYour previous reply could not be parsed. Reply with exactly one of: A, B, TIE.",
                "grade": "\n\nYour previous reply could not be parsed. Reply with only one digit: 1, 2, 3, 4 or 5."}

_VERDICT_RE = re.compile(r"\b(TIE|A|B)\b")
_DIGIT_RE = re.compile(r"\b([1-5])\b")


def parse_verdict(text: str) -> Optional[str]:
    m = _VERDICT_RE.search((text or "").upper().replace("*", " "))
    return m.group(1) if m else None


def parse_grade(text: str) -> Optional[int]:
    m = _DIGIT_RE.search(text or "")
    return int(m.group(1)) if m else None


def _flip(v: Optional[str]) -> Optional[str]:
    return {"A": "B", "B": "A"}.get(v, v) if v else v


class ReplayOnlyProvider:
    """Stand-in upstream when the real one can't be built (no key, no MLX). Cassette hits still work;
    a miss raises, and the judge turns that into None (grade) / "error" (pairwise) and counts it as an error."""

    def __init__(self, name: str, reason: str = "", ratio: float = 1.0):
        self.name, self.reason, self.ratio = name, reason, ratio

    def count_tokens(self, messages, model):
        return int(round(count_messages(messages) * self.ratio))

    def complete(self, messages, model, max_tokens, temperature) -> Completion:
        raise RuntimeError(f"{self.name} upstream unavailable ({self.reason or 'replay only'})")


class Judge:
    """Model-backed judge. Subclasses override `_verdict` (one presentation order) and `_grade_once`."""

    def __init__(self, provider=None, model: str = "", backend: str = "mock", max_tokens: int = 5,
                 prices: Optional[PriceBook] = None, price_alias: Optional[str] = None):
        self.provider, self.model, self.backend, self.max_tokens = provider, model, backend, max_tokens
        self.prices, self.price_alias = prices, price_alias
        self.stats = Counter()
        self.cost_new_usd = 0.0       # spend on calls that were not already in the cassette
        self.cost_all_usd = 0.0       # list-price value of every judge call, replayed or not
        self._memo: dict[str, object] = {}
        self._lock = threading.Lock()   # judging runs in threads (run_ab --workers): keep the accounting exact

    def _count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] += n

    @property
    def label(self) -> str:
        return self.model

    # ---------------------------------------------------------------- one model call
    def _call(self, prompt: str) -> Optional[str]:
        msgs = [ChatMessage(role="system", content=SYSTEM), ChatMessage(role="user", content=prompt)]
        try:
            comp = self.provider.complete(msgs, self.model, self.max_tokens, 0.0)
        except Exception as e:  # cassette miss in replay mode, API error, ...
            self._count("errors")
            log.warning("judge call failed: %s: %s", type(e).__name__, str(e)[:200])
            return None
        # per-call status from the cassette; a shared hit counter races when judging runs in threads
        replayed = (comp.raw or {}).get("cassette") == "replay"
        cost = self._price(comp)
        with self._lock:
            self.stats["calls"] += 1
            self.cost_all_usd += cost
            if not replayed:
                self.stats["new_calls"] += 1
                self.cost_new_usd += cost
        return comp.text

    def _price(self, comp: Completion) -> float:
        if not self.prices:
            return 0.0
        u = comp.usage
        try:
            return self.prices.cost(self.price_alias or self.model, u.input_tokens, u.output_tokens,
                                    u.cached_input_tokens, getattr(u, "cache_write_tokens", 0))
        except KeyError:
            return 0.0

    # ---------------------------------------------------------------- pairwise
    def _verdict(self, question: str, first: str, second: str, reference: Optional[str]) -> Optional[str]:
        """Verdict for one presentation order: 'A' (= first), 'B' (= second), 'TIE', or None if unparseable."""
        ref_block = _REF_BLOCK.format(reference=reference.strip()) if reference else ""
        prompt = PAIRWISE_PROMPT.format(question=question.strip(), ref_block=ref_block, a=first.strip(),
                                        b=second.strip(), given_ref=", given the reference" if reference else "")
        for attempt in range(2):
            text = self._call(prompt if attempt == 0 else prompt + RETRY_SUFFIX["pairwise"])
            if text is None:
                return None
            v = parse_verdict(text)
            if v:
                return v
            self._count("parse_failures")
        return None

    def pairwise_detail(self, question: str, answer_a: str, answer_b: str, reference: Optional[str] = None) -> dict:
        if (answer_a or "").strip() == (answer_b or "").strip():
            return {"verdict": "tie", "order1": "TIE", "order2": "TIE", "consistent": True, "judge": self.label,
                    "skipped": "identical"}
        key = "p:" + hashlib.sha256(json.dumps([question, answer_a, answer_b, reference]).encode()).hexdigest()
        if key in self._memo:
            return dict(self._memo[key])  # type: ignore[arg-type]
        v1 = self._verdict(question, answer_a, answer_b, reference)            # A = answer_a
        v2 = _flip(self._verdict(question, answer_b, answer_a, reference))     # mapped back: A = answer_a
        consistent = v1 is not None and v1 == v2
        if v1 is None or v2 is None:          # a missing order is a failed judgment, not a position-swap tie
            verdict = JUDGE_ERROR
            self._count("pairwise_errors")
        else:
            verdict = v1 if consistent and v1 in ("A", "B") else "tie"
        self._count("pairwise")
        if v1 is not None and v2 is not None and v1 != v2:
            self._count("position_inconsistent")
        out = {"verdict": verdict, "order1": v1, "order2": v2, "consistent": consistent, "judge": self.label}
        self._memo[key] = out
        return dict(out)

    def pairwise(self, question: str, answer_a: str, answer_b: str, reference: Optional[str] = None) -> str:
        return self.pairwise_detail(question, answer_a, answer_b, reference)["verdict"]

    # ---------------------------------------------------------------- absolute grade
    def _grade_once(self, question: str, answer: str, reference: Optional[str]) -> Optional[float]:
        ref_block = _REF_BLOCK.format(reference=reference.strip()) if reference else ""
        prompt = GRADE_PROMPT.format(question=question.strip(), ref_block=ref_block, answer=(answer or "").strip(),
                                     agrees="agrees with the reference" if reference else "is plausible and specific")
        for attempt in range(2):                  # parse failure -> retry once -> None
            text = self._call(prompt if attempt == 0 else prompt + RETRY_SUFFIX["grade"])
            if text is None:
                return None
            g = parse_grade(text)
            if g is not None:
                return (g - 1) / 4.0
            self._count("parse_failures")
        return None

    def grade(self, question: str, answer: str, reference: Optional[str] = None) -> Optional[float]:
        key = "g:" + hashlib.sha256(json.dumps([question, answer, reference]).encode()).hexdigest()
        if key not in self._memo:
            self._count("grades")
            self._memo[key] = self._grade_once(question, answer, reference)
        return self._memo[key]  # type: ignore[return-value]

    def describe(self) -> dict:
        return {"judge": self.label, "backend": self.backend, "stats": dict(self.stats),
                "cost_new_usd": round(self.cost_new_usd, 6), "cost_all_usd": round(self.cost_all_usd, 6)}


# ---------------------------------------------------------------------------------------- heuristic (mock)
_STOP = set("""a an the and or but if of to in on at for from by with about as is are was were be been being it its
this that these those i you your we our us me my he she they them their there here what which who whom when where why
how can could would should will shall may might do does did done have has had not no yes so than then too very just
also please thank thanks hi hello any some all each more most other into out up down over again once only own same
s t don can't won't i'm you're it's let help need want get""".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9₹]+", (text or "").lower()) if t not in _STOP and len(t) > 1]


def overlap_f1(answer: str, target: str) -> float:
    a, b = Counter(_tokens(answer)), Counter(_tokens(target))
    if not a or not b:
        return 0.0
    common = sum((a & b).values())
    if common == 0:
        return 0.0
    p, r = common / sum(a.values()), common / sum(b.values())
    return 2 * p * r / (p + r)


class HeuristicJudge(Judge):
    """Deterministic stand-in for tests and the mock backend: token-overlap F1 with the reference (or with the
    question when there is no reference). It carries a small first-position bonus, like real LLM judges, so the
    position-swap logic is exercised: near-ties come out inconsistent and are scored as "tie"."""

    def __init__(self, position_bias: float = 0.01):
        super().__init__(provider=None, model=HEURISTIC_LABEL, backend="mock")
        self.position_bias = position_bias

    def _score(self, question: str, answer: str, reference: Optional[str]) -> float:
        return overlap_f1(answer, reference if reference else question)

    def _verdict(self, question, first, second, reference):
        s1 = self._score(question, first, reference) + self.position_bias
        s2 = self._score(question, second, reference)
        return "A" if s1 > s2 else ("B" if s2 > s1 else "TIE")

    def _grade_once(self, question, answer, reference):
        return round(min(1.0, max(0.0, self._score(question, answer, reference))), 4)


# ---------------------------------------------------------------------------------------- construction
def _inner_provider(backend: str):
    from costguard.providers import make_provider
    try:
        return make_provider(Settings(backend=backend))
    except Exception as e:  # no key / no MLX: cassette replay still works
        log.warning("judge backend %s unavailable (%s); replay-only", backend, e)
        return ReplayOnlyProvider(backend, str(e)[:120], ratio=1.15 if backend == "anthropic" else 1.0)


def get_judge(settings: Optional[Settings] = None, provider=None) -> Judge:
    """Judge for `settings.backend` (strong tier) unless COSTGUARD_JUDGE_BACKEND / COSTGUARD_JUDGE_MODEL override it.

    `provider` (optional) is an already-built upstream provider for the same backend (e.g. the A/B engine's), so a
    local model is not loaded twice. The judge always wraps it in its own cassette (eval/cassettes/judge.jsonl)."""
    settings = settings or Settings.from_env()
    backend = os.environ.get("COSTGUARD_JUDGE_BACKEND") or settings.backend
    if backend == "mock":
        return HeuristicJudge()
    policy = load_policy(settings.policy_path)
    model = os.environ.get("COSTGUARD_JUDGE_MODEL") or policy.model_id(backend, "strong")
    if isinstance(provider, CassetteProvider):
        provider = provider.inner
    if provider is None or getattr(provider, "name", "").split("+")[0] != backend:
        provider = _inner_provider(backend)
    mode = os.environ.get("COSTGUARD_JUDGE_CASSETTE_MODE") or settings.cassette_mode or "auto"
    path = Path(os.environ.get("COSTGUARD_JUDGE_CASSETTE") or JUDGE_CASSETTE)
    alias = next((a for a in ("strong", "cheap") if policy.backends.get(backend, {}).get(a) == model), None)
    prices = PriceBook(settings.prices_path, policy.billing_for(backend))
    return Judge(CassetteProvider(provider, path, mode), model, backend=backend, prices=prices, price_alias=alias)


# ---------------------------------------------------------------------------------------- hand labels
def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def human_agreement(path: Path = HUMAN_LABELS, judge: Optional[Judge] = None) -> Optional[dict]:
    """Judge-vs-human agreement on hand labels (eval/data/human_labels.jsonl). Returns None if there are no labels.

    Pairwise rows: {"question","reference","answer_a","answer_b","human": "A"|"B"|"tie", "judge"?: ...}
    Grade rows:    {"question","reference","answer","human_score": 1..5, "judge_score"?: 0..1}
    A stored "judge"/"judge_score" is used when no judge is passed; otherwise the judge is re-run (free on replay)."""
    path = Path(path)
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    pw = [r for r in rows if r.get("human") in ("A", "B", "tie", "TIE") and "answer_a" in r]
    gr = [r for r in rows if r.get("human_score") is not None and "answer" in r]
    out: dict = {"labels_file": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                 "judge": judge.label if judge else "stored", "n_pairwise": 0, "n_grade": 0}
    hum, jud = [], []
    for r in pw:
        j = (judge.pairwise(r["question"], r["answer_a"], r["answer_b"], r.get("reference")) if judge
             else r.get("judge"))
        if j is None or j == JUDGE_ERROR:
            continue
        hum.append(str(r["human"]).lower() if str(r["human"]).upper() == "TIE" else str(r["human"]))
        jud.append(str(j).lower() if str(j).upper() == "TIE" else str(j))
    if hum:
        agree = sum(h == j for h, j in zip(hum, jud))
        decisive = [(h, j) for h, j in zip(hum, jud) if h != "tie" and j != "tie"]
        p, lo, hi = proportion_ci(agree, len(hum))
        out.update(n_pairwise=len(hum), agreement=round(p, 4), agreement_ci=[round(lo, 4), round(hi, 4)],
                   agreement_excl_ties=(round(sum(h == j for h, j in decisive) / len(decisive), 4) if decisive else None),
                   n_decisive=len(decisive), kappa=round(cohen_kappa(hum, jud), 4))
    hs, js = [], []
    for r in gr:
        j = judge.grade(r["question"], r["answer"], r.get("reference")) if judge else r.get("judge_score")
        if j is None:
            continue
        hs.append((float(r["human_score"]) - 1) / 4.0)
        js.append(float(j))
    if hs:
        out.update(n_grade=len(hs), grade_mae=round(sum(abs(a - b) for a, b in zip(hs, js)) / len(hs), 4),
                   grade_pass_kappa=round(cohen_kappa([h >= 0.75 for h in hs], [j >= 0.75 for j in js]), 4))
    return out


def export_label_pairs(db_path: Path, arm: str, n: int, out_path: Path, trace_path: Optional[Path] = None,
                       seed: int = 0) -> int:
    """Write n blinded (A0 vs arm) answer pairs for hand labelling. Order inside each pair is randomised and the
    arm names are kept in underscore fields, so labellers see only answer_a / answer_b."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    get = lambda a: {r["trace_pos"]: dict(r) for r in con.execute(  # noqa: E731
        "SELECT trace_pos, item_id, query, response_text FROM requests WHERE arm = ? AND error IS NULL", (a,))}
    base, other = get("A0"), get(arm)
    refs = {}
    if trace_path and Path(trace_path).exists():
        refs = {r["pos"]: r.get("reference") for r in _read_jsonl(Path(trace_path))}
    cands = [p for p in sorted(base) if p in other and base[p]["response_text"] != other[p]["response_text"]]
    rng = random.Random(seed)
    rng.shuffle(cands)
    with Path(out_path).open("w") as f:
        for p in cands[:n]:
            pair = [("A0", base[p]["response_text"]), (arm, other[p]["response_text"])]
            rng.shuffle(pair)
            f.write(json.dumps({"trace_pos": p, "item_id": base[p]["item_id"], "question": base[p]["query"],
                                "reference": refs.get(p), "answer_a": pair[0][1], "answer_b": pair[1][1],
                                "human": None, "labeler": None, "_a_arm": pair[0][0], "_b_arm": pair[1][0]},
                               ensure_ascii=False) + "\n")
    return min(n, len(cands))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Judge utilities: hand-label agreement and label export.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agreement", help="judge-vs-human agreement on eval/data/human_labels.jsonl")
    a.add_argument("--labels", default=str(HUMAN_LABELS))
    a.add_argument("--rejudge", action="store_true", help="re-run the judge instead of using stored verdicts")
    e = sub.add_parser("export", help="export blinded A0-vs-arm pairs for hand labelling")
    e.add_argument("--db", required=True)
    e.add_argument("--arm", default="A5")
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--trace", default=str(ROOT / "eval" / "data" / "trace_v1.jsonl"))
    e.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "agreement":
        res = human_agreement(Path(args.labels), get_judge() if args.rejudge else None)
        print(json.dumps(res, indent=2) if res else f"no labels at {args.labels}")
        return 0
    k = export_label_pairs(Path(args.db), args.arm, args.n, Path(args.out), Path(args.trace))
    print(f"wrote {k} pairs to {args.out}; fill in 'human' with A, B or tie")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
