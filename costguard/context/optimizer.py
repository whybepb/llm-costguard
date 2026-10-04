"""Stage 3: context optimiser (rerank -> dynamic-k -> whole-doc budget fit -> order).

Retrieved RAG context is usually "top-8 by embedding similarity": generous, partly redundant and partly off-topic.
This stage re-scores every (query, doc) pair with a local cross-encoder and keeps only what the question needs:

1. **Score.** `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed (22M params, ONNX on CPU, a few ms per doc).
   If it cannot load, fall back to bge-small bi-encoder cosine, then to lexical overlap, and say so in `note`.
2. **Dynamic-k.** If the policy gives `min_score`, keep docs scoring >= it (in the active scorer's units). Otherwise
   apply a relative rule: keep docs within `gap` of the best score. The top-1 doc is always kept.
3. **Near-duplicate docs** (word-shingle Jaccard >= 0.8 with a better-scored kept doc) are dropped.
4. **Budget.** Fit into `budget_tokens` (o200k count of the formatted `[n] doc` block) by dropping *whole* docs,
   lowest score first. Never truncates mid-sentence. The top-1 doc stays even if it alone exceeds the budget.
5. **Order.** Best doc first, second-best last, the rest in between ("lost in the middle": models use the start and
   the end of the context best). Set `COSTGUARD_CONTEXT_ORDER=score|original` to change it.

`last_ms` is the latency of the most recent `optimize` call; `load_ms` the one-off model load.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np

from ..config import ROOT
from ..pipeline import format_docs
from ..schemas import ContextResult
from ..tokens import count_text

log = logging.getLogger("costguard.context")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Relative dynamic-k gap per scorer, in that scorer's units (calibrated with eval/compression_eval.py; see
# docs/components/context_and_compression.md). Cross-encoder scores are raw ms-marco logits.
DEFAULT_GAP = {"cross-encoder": 6.0, "bi-encoder": 0.08, "lexical": 0.35}

_load_lock = threading.Lock()


# ----------------------------------------------------------------------------------------------- shared models

def _cache_dir() -> str:
    # same location the semantic cache and the Docker image use (fastembed's own default, the OS temp dir, gets wiped)
    return os.environ.get("FASTEMBED_CACHE_PATH") or str(ROOT / "models" / "fastembed")


@lru_cache(maxsize=2)
def _local_embedder(model: str):
    from fastembed import TextEmbedding
    return TextEmbedding(model, cache_dir=_cache_dir())


@lru_cache(maxsize=2)
def _cross_encoder(model: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder
    return TextCrossEncoder(model, cache_dir=_cache_dir())


def get_cross_encoder(model: str = CROSS_ENCODER_MODEL):
    with _load_lock:
        return _cross_encoder(model)


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def embed_passages(texts: Sequence[str], model: str = EMBED_MODEL) -> np.ndarray:
    """L2-normalised passage embeddings (n, d). Re-uses the semantic cache's process-wide bge-small instance when it
    is available, so the model is loaded (and held in RAM) once."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    try:
        from ..cache.embedder import get_embedder as shared_embedder
    except ImportError:
        shared_embedder = None
    if shared_embedder is not None:
        return _norm(shared_embedder(model).embed(list(texts)))
    with _load_lock:
        emb = _local_embedder(model)
    return _norm(np.stack(list(emb.embed(list(texts)))))


def embed_queries(texts: Sequence[str], model: str = EMBED_MODEL) -> np.ndarray:
    """L2-normalised query embeddings, with the BGE retrieval instruction prepended."""
    prefix = BGE_QUERY_PREFIX if "bge" in model.lower() else ""
    return embed_passages([prefix + t for t in texts], model)


# ----------------------------------------------------------------------------------------------- scorers

_WORD = re.compile(r"[a-z0-9₹%]+")
_STOP = frozenset("""a an the and or but if of to in on at by for with from as is are was were be been being am do does did
have has had i me my we our you your it its this that these those there here what which who whom when where why how can
could would should will shall may might must not no yes so than then too very just also any all some about into over
after before up down out get got please want need know tell much many""".split())


def content_terms(text: str) -> list[str]:
    """Lowercased, stop-word-free, lightly stemmed terms (shared with the heuristic compressor)."""
    out = []
    for w in _WORD.findall((text or "").lower()):
        if w in _STOP or (len(w) < 2 and not w.isdigit()):
            continue
        out.append(stem(w))
    return out


def stem(w: str) -> str:
    """Tiny suffix stripper, enough to match cancel/cancelled, ship/shipped/shipping, charge/charges, box/boxes."""
    if w.isdigit() or len(w) <= 3 or any(c.isdigit() for c in w):
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "ches", "shes", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def lexical_scores(query: str, texts: Sequence[str]) -> list[float]:
    """IDF-weighted share of query terms present in each text, in [0, 1]. No model needed."""
    q = set(content_terms(query))
    if not q:
        return [0.0] * len(texts)
    sets = [set(content_terms(t)) for t in texts]
    n = len(sets)
    idf = {t: float(np.log((n + 1) / (sum(t in s for s in sets) + 0.5))) + 1.0 for t in q}
    tot = sum(idf.values())
    return [sum(idf[t] for t in q & s) / tot for s in sets]


class Scorer:
    kind = "base"

    def __init__(self, name: str):
        self.name = name

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        raise NotImplementedError


class CrossEncoderScorer(Scorer):
    kind = "cross-encoder"

    def __init__(self, model: str = CROSS_ENCODER_MODEL):
        super().__init__(f"cross-encoder:{model}")
        self.model_name = model

    def load(self):
        return get_cross_encoder(self.model_name)

    def score(self, query, docs):
        return [float(s) for s in self.load().rerank(query, list(docs))]


class BiEncoderScorer(Scorer):
    kind = "bi-encoder"

    def __init__(self, model: str = EMBED_MODEL):
        super().__init__(f"bi-encoder:{model}")
        self.model_name = model

    def load(self):
        return embed_passages(["warm up"], self.model_name)

    def score(self, query, docs):
        q = embed_queries([query], self.model_name)[0]
        return [float(x) for x in embed_passages(docs, self.model_name) @ q]


class LexicalScorer(Scorer):
    kind = "lexical"

    def __init__(self):
        super().__init__("lexical-overlap")

    def load(self):
        return None

    def score(self, query, docs):
        return lexical_scores(query, docs)


# ----------------------------------------------------------------------------------------------- helpers

def _shingles(text: str, n: int = 3) -> set:
    w = _WORD.findall(text.lower())
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def edge_order(ranked: list[int]) -> list[int]:
    """[best, 3rd, 4th, ..., 2nd]: strongest evidence at both ends of the context, weakest in the middle."""
    if len(ranked) <= 2:
        return list(ranked)
    return [ranked[0], *ranked[2:], ranked[1]]


# ----------------------------------------------------------------------------------------------- optimiser

class RerankContextOptimizer:
    """`optimize(query, docs, budget_tokens, min_score) -> ContextResult` (see module docstring)."""

    def __init__(self, scorer: Optional[Scorer] = None, gap: Optional[float] = None, order: str = "edges",
                 dedup_threshold: float = 0.8, max_docs: Optional[int] = None,
                 fallbacks: Optional[list[Callable[[], Scorer]]] = None):
        self.scorer = scorer or CrossEncoderScorer()
        fb = fallbacks if fallbacks is not None else [BiEncoderScorer, LexicalScorer]
        self._fallbacks = [f for f in fb if f is not type(self.scorer)]
        self._gap = gap
        self.order = order
        self.dedup_threshold = dedup_threshold
        self.max_docs = max_docs
        self.last_ms: float = 0.0
        self.load_ms: Optional[float] = None
        self.fallback_reason: Optional[str] = None
        self.last_scores: list[float] = []

    @property
    def gap(self) -> float:
        return self._gap if self._gap is not None else DEFAULT_GAP.get(self.scorer.kind, 0.0)

    def warmup(self) -> "RerankContextOptimizer":
        """Load the scoring model now (instead of on the first request), falling back if it fails."""
        self._score("warmup", ["warmup document"])
        return self

    def _score(self, query: str, docs: list[str]) -> list[float]:
        while True:
            try:
                if self.load_ms is None:
                    t = time.perf_counter()
                    load = getattr(self.scorer, "load", None)
                    if load:
                        load()
                    self.load_ms = (time.perf_counter() - t) * 1000
                return self.scorer.score(query, docs)
            except Exception as e:  # model missing / download failed / onnx error -> degrade, and say so
                if not self._fallbacks:
                    raise
                nxt = self._fallbacks.pop(0)()
                self.fallback_reason = f"{self.scorer.name} failed ({type(e).__name__}: {e}"[:200] + ")"
                log.warning("context scorer %s unavailable, falling back to %s: %s", self.scorer.name, nxt.name, e)
                self.scorer, self.load_ms = nxt, None

    def optimize(self, query: str, docs: list[str], budget_tokens: Optional[int] = None,
                 min_score: Optional[float] = None) -> ContextResult:
        t0 = time.perf_counter()
        try:
            return self._optimize(query, list(docs), budget_tokens, min_score)
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000

    def _optimize(self, query, docs, budget_tokens, min_score) -> ContextResult:
        tokens_before = count_text(format_docs(docs)) if docs else 0
        if not docs:
            return ContextResult(docs=[], note="no docs")
        scores = self._score(query, docs)
        self.last_scores = scores
        ranked = sorted(range(len(docs)), key=lambda i: -scores[i])
        top = ranked[0]

        # 1. dynamic-k: absolute floor if configured, else relative gap to the best score; top-1 always survives
        if min_score is not None:
            cand, rule = [i for i in ranked if scores[i] >= min_score], f"min_score>={min_score}"
        else:
            cand, rule = [i for i in ranked if scores[i] >= scores[top] - self.gap], f"gap<={self.gap:g}"
        if top not in cand:
            cand.insert(0, top)
        if self.max_docs:
            cand = cand[: self.max_docs]
        n_rule = len(cand)

        # 2. near-duplicate docs: keep the better-scored copy
        kept, shingles, n_dup = [], [], 0
        for i in cand:
            sh = _shingles(docs[i])
            if any(jaccard(sh, s) >= self.dedup_threshold for s in shingles):
                n_dup += 1
                continue
            kept.append(i)
            shingles.append(sh)

        # 3. budget: add whole docs in score order while the formatted block fits; skip ones that don't
        n_budget = 0
        if budget_tokens and budget_tokens > 0:
            fit = [kept[0]]
            for i in kept[1:]:
                if count_text(format_docs([docs[j] for j in fit + [i]])) <= budget_tokens:
                    fit.append(i)
                else:
                    n_budget += 1
            kept = fit

        # 4. order
        if self.order == "edges":
            final = edge_order(kept)
        elif self.order == "original":
            final = sorted(kept)
        else:
            final = kept
        out = [docs[i] for i in final]
        note = (f"{self.scorer.name}; {rule}; kept {len(final)}/{len(docs)} "
                f"(rule dropped {len(docs) - n_rule}, dup {n_dup}, budget {n_budget}); order={self.order}")
        if self.fallback_reason:
            note += f"; FALLBACK: {self.fallback_reason}"
        return ContextResult(docs=out, kept_indices=final, scores=[round(scores[i], 4) for i in final],
                             tokens_before=tokens_before, tokens_after=count_text(format_docs(out)), note=note)


def build_context_optimizer(settings, policy) -> RerankContextOptimizer:
    """Factory hook. Env: COSTGUARD_RERANKER (cross-encoder|bi-encoder|lexical, default cross-encoder),
    COSTGUARD_RERANK_GAP (float), COSTGUARD_CONTEXT_ORDER (edges|score|original)."""
    kind = os.environ.get("COSTGUARD_RERANKER", "cross-encoder").lower()
    scorer = {"bi-encoder": BiEncoderScorer, "lexical": LexicalScorer}.get(kind, CrossEncoderScorer)()
    gap = os.environ.get("COSTGUARD_RERANK_GAP")
    return RerankContextOptimizer(scorer=scorer, gap=float(gap) if gap else None,
                                  order=os.environ.get("COSTGUARD_CONTEXT_ORDER", "edges"))
