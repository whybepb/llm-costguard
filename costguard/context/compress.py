"""Stage 4: compress the retrieved-context block. Never the system prompt, never the question.

The engine calls `compress(block, rate, query)` only when the block is at least `compression_min_tokens` long. `rate`
is the fraction of tokens to KEEP (0.5 -> about 2x). Two real compressors and a no-op, chosen with the env var
`COSTGUARD_COMPRESSOR`:

- `heuristic` (default): query-aware, sentence-level extractive compression in pure Python (plus the tokenizer).
  No model and no extra RAM, so it fits a free 512 MB host.
  1. Split each `[n]` doc into units: sentences, list items (tied to their "...:" lead-in line) and table rows (tied
     to their header row). Markdown table separator rows are dropped as formatting.
  2. Drop exact and near-duplicate units (word Jaccard >= 0.8 *and* the same numbers, so "1-3 days" never
     dedups against "3-5 days") and boilerplate ("For more information...", "We value your business...").
  3. Score the rest by similarity to the query: IDF-weighted term overlap (`lexical`, default), bge-small cosine
     (`embed`) or both (`hybrid`); set `COSTGUARD_HEURISTIC_SCORER`.
  4. Protect any unit that contains a number and matches the query's terms (or repeats a number from the query). It
     is never dropped, even if that overshoots the target.
  5. Keep the best units, in original order, until `rate x tokens_before` is reached. Doc markers `[n]` and titles
     are kept for every surviving doc; docs with nothing left are dropped whole.
- `llmlingua2`: Microsoft LLMLingua-2 token classification (`microsoft/llmlingua-2-bert-base-multilingual-cased-
  meetingbank`; `COSTGUARD_LLMLINGUA_MODEL=large` selects the xlm-roberta-large one). Lazy-loaded on first use.
  Each doc is compressed separately and its `[n]` marker re-attached, because the model drops the digits inside
  `[1]`. It is task-agnostic: it ignores the query.
- `none`: passthrough with honest token counts (for A/B arms).

Docs that look like code or JSON pass through verbatim in both compressors. Token-dropping corrupts them, and the
research brief flags them as the main failure mode.

Why only the volatile block: providers bill a cached prompt prefix at about 0.1x the input price. A 3,000-token
cached system prompt costs the same as about 300 fresh tokens. Compressing it 2x gives 1,500 *uncached* tokens,
about 5x the cost, and may drop a "not" from the instructions. So the system prompt stays byte-identical and first,
the compressed context comes after it, and the question comes last, verbatim.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..schemas import CompressResult
from ..tokens import count_text
from .optimizer import content_terms, lexical_scores

log = logging.getLogger("costguard.context")

DOC_MARKER = re.compile(r"(?m)^\[(\d+)\] ")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"'(\[])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_DIGIT = re.compile(r"\d")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_WORDS = re.compile(r"[a-z0-9₹%]+")

BOILERPLATE = [re.compile(p, re.I) for p in (
    r"^for more information\b",
    r"\bwe value your (business|trust)\b",
    r"\bthank you for (shopping|choosing)\b",
    r"\breserves the right to modify\b",
    r"^last updated\b",
    r"\b(is|are) (always )?happy to help\b",
    r"^happy shopping\b",
    r"^we'?re sorry for the inconvenience\b",
    r"^#+\s*need more help\??\s*$",
)]


def split_docs(block: str) -> list[tuple[str, str]]:
    """'[1] a\n\n[2] b' -> [('[1] ', 'a'), ('[2] ', 'b')]. Text without markers is one unmarked doc."""
    ms = list(DOC_MARKER.finditer(block or ""))
    if not ms:
        return [("", block or "")]
    out = []
    if block[: ms[0].start()].strip():
        out.append(("", block[: ms[0].start()].strip("\n")))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(block)
        out.append((m.group(0), block[m.end(): end].strip("\n")))
    return out


def structured_kind(text: str) -> Optional[str]:
    """'code' / 'json' for content that token- or sentence-dropping would corrupt; None for prose."""
    s = (text or "").strip()
    if "```" in s:
        return "code"
    if s[:1] in "{[":
        try:
            json.loads(s)
            return "json"
        except ValueError:
            pass
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 3:
        codey = sum(bool(re.search(r"[;{}]\s*$|^\s*(def|class|import|from|function|const|let|var|return)\b", ln))
                    for ln in lines)
        if codey / len(lines) > 0.3:
            return "code"
    return None


def _numbers(text: str) -> frozenset:
    return frozenset(n.replace(",", "") for n in _NUM.findall(text))


class PassthroughCompressor:
    method = "none"

    def __init__(self):
        self.last_ms = 0.0

    def compress(self, text: str, rate: float, query: Optional[str] = None) -> CompressResult:
        n = count_text(text)
        return CompressResult(text=text, tokens_before=n, tokens_after=n, method=self.method)


# ----------------------------------------------------------------------------------------------- heuristic

@dataclass
class _Unit:
    idx: int
    doc: int
    line: int
    kind: str                  # title | heading | sentence | bullet | header_row | row | verbatim
    text: str
    score_text: str
    deps: list = field(default_factory=list)
    tokens: int = 0
    score: float = 0.0
    protected: bool = False
    dropped: Optional[str] = None


class HeuristicCompressor:
    """Sentence-level extractive compression; see the module docstring."""

    def __init__(self, scorer: str = "lexical", near_dup: float = 0.8, carry: float = 0.5,
                 protect_first: bool = False):
        self.scorer = scorer if scorer in ("lexical", "embed", "hybrid") else "lexical"
        self.near_dup = near_dup
        self.carry = carry                  # share of a sentence's score passed to the next sentence of its paragraph
        self.protect_first = protect_first  # True: protected units jump the queue (old behaviour)
        self.method = f"heuristic-{self.scorer}"
        self.last_ms = 0.0
        self.last_stats: dict = {}

    # -- parsing
    def _units(self, docs: list[tuple[str, str]]) -> list[_Unit]:
        units: list[_Unit] = []

        def add(**kw) -> _Unit:
            u = _Unit(idx=len(units), **kw)
            u.tokens = count_text(u.text) + 1
            units.append(u)
            return u

        for d, (_, body) in enumerate(docs):
            if structured_kind(body):
                add(doc=d, line=0, kind="verbatim", text=body, score_text=body)
                continue
            title, lead, header = None, None, None
            for ln_no, line in enumerate(body.splitlines()):
                s = line.strip()
                if not s:          # a blank line ends a table, but a "...:" lead-in still governs the list after it
                    header = None
                    continue
                base = [title.idx] if title else []
                if ln_no == 0 and s.startswith("# "):
                    title = add(doc=d, line=ln_no, kind="title", text=s, score_text=s.lstrip("# "))
                elif s.startswith("#"):
                    add(doc=d, line=ln_no, kind="heading", text=s, score_text=s.lstrip("# "), deps=base)
                    lead, header = None, None
                elif s.startswith("|"):
                    if _TABLE_SEP.match(s):
                        continue
                    if header is None:
                        header = add(doc=d, line=ln_no, kind="header_row", text=s, score_text=s, deps=base)
                    else:
                        add(doc=d, line=ln_no, kind="row", text=s, score_text=f"{header.text} {s}",
                            deps=base + [header.idx])
                elif _BULLET.match(s):
                    deps = base + ([lead.idx] if lead else [])
                    add(doc=d, line=ln_no, kind="bullet", text=s,
                        score_text=f"{lead.text} {s}" if lead else s, deps=deps)
                else:
                    header = None
                    last = None
                    for sent in _SENT_SPLIT.split(s):
                        if sent.strip():
                            last = add(doc=d, line=ln_no, kind="sentence", text=sent.strip(),
                                       score_text=sent.strip(), deps=base)
                    lead = last if last is not None and last.text.endswith(":") else None
        return units

    # -- scoring
    def _score(self, query: str, units: list[_Unit]) -> None:
        texts = [u.score_text for u in units]
        lex = np.asarray(lexical_scores(query, texts), dtype=np.float32)
        if self.scorer == "lexical":
            s = lex
        else:
            from .optimizer import embed_passages, embed_queries
            cos = embed_passages(texts) @ embed_queries([query])[0]
            if self.scorer == "embed":
                s = cos
            else:
                z = lambda v: (v - v.mean()) / (v.std() + 1e-6)  # noqa: E731
                s = z(lex) + z(cos)
        raw = [float(v) for v in s]
        for i, u in enumerate(units):
            # answers often sit in the sentence right after the one that matches the question
            # ("Once an order has shipped, it cannot be cancelled. You can either refuse the delivery ...")
            prev = units[i - 1] if i else None
            bonus = raw[i - 1] if prev is not None and prev.doc == u.doc and prev.line == u.line else 0.0
            u.score = raw[i] + self.carry * bonus

    def _protect(self, query: str, units: list[_Unit]) -> None:
        q = set(content_terms(query))
        need = 1 if len(q) <= 2 else 2
        qnums = _numbers(query)
        for u in units:
            if u.kind == "verbatim":
                u.protected = True
            elif _DIGIT.search(u.text):
                overlap = len(q & set(content_terms(u.score_text)))
                u.protected = overlap >= need or bool(qnums & _numbers(u.text))

    def _dedup_and_strip(self, units: list[_Unit]) -> None:
        seen_exact: set = set()
        kept_sets: list[tuple[frozenset, frozenset]] = []
        for u in units:
            if u.kind in ("title", "verbatim"):
                continue
            if not u.protected and any(p.search(u.text) for p in BOILERPLATE):
                u.dropped = "boilerplate"
                continue
            key = " ".join(_WORDS.findall(u.text.lower()))
            if key in seen_exact:
                u.dropped = "duplicate"
                continue
            seen_exact.add(key)
            words, nums = frozenset(key.split()), _numbers(u.text)
            if len(words) >= 5 and any(n == nums and len(words & w) / len(words | w) >= self.near_dup
                                       for w, n in kept_sets):
                u.dropped = "near-duplicate"
                continue
            kept_sets.append((words, nums))

    # -- main
    def compress(self, text: str, rate: float, query: Optional[str] = None) -> CompressResult:
        t0 = time.perf_counter()
        try:
            return self._compress(text, rate, query or "")
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000

    def _compress(self, text: str, rate: float, query: str) -> CompressResult:
        before = count_text(text)
        if rate >= 1.0 or not text.strip():
            return CompressResult(text=text, tokens_before=before, tokens_after=before, method=self.method)
        docs = split_docs(text)
        units = self._units(docs)
        self._score(query, units)
        self._protect(query, units)
        self._dedup_and_strip(units)

        target = max(1, int(round(rate * before)))
        selected: set = set()
        used = 0

        def cost(u: _Unit) -> int:
            extra = sum(units[j].tokens + (3 if units[j].kind == "title" else 0)
                        for j in u.deps if j not in selected and not units[j].dropped)
            return u.tokens + extra

        pri = (lambda u: (not u.protected, -u.score, u.idx)) if self.protect_first else (lambda u: (-u.score, u.idx))
        order = sorted((u for u in units if not u.dropped and u.kind != "title"), key=pri)

        def take(u: _Unit):
            nonlocal used
            used += cost(u)
            selected.add(u.idx)
            selected.update(j for j in u.deps if not units[j].dropped)

        for u in order:                      # best-first until the target is reached (skip what doesn't fit)
            if not selected or used + cost(u) <= target:
                take(u)
        for u in order:                      # then protected numeric facts, even past the target
            if u.protected and u.idx not in selected:
                take(u)

        out_docs = []
        for d, (marker, _) in enumerate(docs):
            kept = [u for u in units if u.doc == d and u.idx in selected]
            if not kept:
                continue
            lines: dict[int, list[str]] = {}
            for u in kept:
                lines.setdefault(u.line, []).append(u.text)
            out_docs.append(marker + "\n".join(" ".join(v) for _, v in sorted(lines.items())))
        out = "\n\n".join(out_docs)
        self.last_stats = {
            "units": len(units), "kept": len(selected), "protected": sum(u.protected for u in units),
            "boilerplate": sum(u.dropped == "boilerplate" for u in units),
            "duplicate": sum(u.dropped in ("duplicate", "near-duplicate") for u in units),
            "docs_in": len(docs), "docs_out": len(out_docs), "target": target,
        }
        return CompressResult(text=out, tokens_before=before, tokens_after=count_text(out), method=self.method)


# ----------------------------------------------------------------------------------------------- LLMLingua-2

LLMLINGUA2_MODELS = {
    "bert": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",     # ~180M params, fast
    "large": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",               # ~560M params, ~2.2 GB
}
FORCE_TOKENS = ["\n", "?", ".", "[", "]"]


def _default_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LLMLingua2Compressor:
    """Task-agnostic token-level compression (offline / beefy hosts). Lazy-loads on the first call."""

    _lock = threading.Lock()

    def __init__(self, model: str = "bert", device: Optional[str] = None, force_reserve_digit: bool = False):
        self.model_name = LLMLINGUA2_MODELS.get(model, model)
        self.device = device
        self.force_reserve_digit = force_reserve_digit
        self.method = "llmlingua2" + ("-large" if "large" in self.model_name else "") + \
                      ("+digits" if force_reserve_digit else "")
        self._pc = None
        self._fallback: Optional[HeuristicCompressor] = None
        self.load_ms: Optional[float] = None
        self.last_ms = 0.0
        self.fallback_reason: Optional[str] = None

    def load(self):
        with self._lock:
            if self._pc is None:
                t = time.perf_counter()
                from llmlingua import PromptCompressor
                self.device = self.device or _default_device()
                self._pc = PromptCompressor(model_name=self.model_name, use_llmlingua2=True, device_map=self.device)
                self.load_ms = (time.perf_counter() - t) * 1000
        return self._pc

    def compress(self, text: str, rate: float, query: Optional[str] = None) -> CompressResult:
        t0 = time.perf_counter()
        try:
            return self._compress(text, rate, query)
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000

    def _compress(self, text: str, rate: float, query: Optional[str]) -> CompressResult:
        before = count_text(text)
        if rate >= 1.0 or not text.strip():
            return CompressResult(text=text, tokens_before=before, tokens_after=before, method=self.method)
        if self._fallback is None:
            try:
                pc = self.load()
            except Exception as e:  # llmlingua/torch missing or model download failed: degrade, and say so
                self.fallback_reason = f"{type(e).__name__}: {e}"[:200]
                log.warning("LLMLingua-2 unavailable, using the heuristic compressor: %s", e)
                self._fallback = HeuristicCompressor()
        if self._fallback is not None:
            r = self._fallback.compress(text, rate, query)
            return r.model_copy(update={"method": f"{r.method} (llmlingua2 unavailable)"})

        docs = split_docs(text)
        todo = [i for i, (_, body) in enumerate(docs) if body.strip() and not structured_kind(body)]
        outs = {i: body for i, (_, body) in enumerate(docs)}
        if todo:
            with self._lock:
                res = pc.compress_prompt([docs[i][1] for i in todo], rate=rate, force_tokens=FORCE_TOKENS,
                                         drop_consecutive=True, use_context_level_filter=False,
                                         force_reserve_digit=self.force_reserve_digit)
            comp = res.get("compressed_prompt_list") or []
            if len(comp) == len(todo):
                outs.update({i: c.strip() for i, c in zip(todo, comp)})
            else:  # defensive: never misalign markers and docs
                raise RuntimeError(f"llmlingua returned {len(comp)} contexts for {len(todo)} inputs")
        out = "\n\n".join(m + outs[i] for i, (m, _) in enumerate(docs) if outs[i].strip())
        return CompressResult(text=out, tokens_before=before, tokens_after=count_text(out), method=self.method)


# ----------------------------------------------------------------------------------------------- factory

def build_compressor(settings, policy):
    """Factory hook. COSTGUARD_COMPRESSOR = heuristic (default) | llmlingua2 | none.
    COSTGUARD_HEURISTIC_SCORER = lexical (default) | embed | hybrid. COSTGUARD_LLMLINGUA_MODEL = bert | large."""
    kind = os.environ.get("COSTGUARD_COMPRESSOR", "heuristic").strip().lower()
    if kind in ("none", "off", "passthrough"):
        return PassthroughCompressor()
    if kind in ("llmlingua2", "llmlingua-2", "llmlingua"):
        return LLMLingua2Compressor(model=os.environ.get("COSTGUARD_LLMLINGUA_MODEL", "bert"))
    return HeuristicCompressor(scorer=os.environ.get("COSTGUARD_HEURISTIC_SCORER", "lexical"))
