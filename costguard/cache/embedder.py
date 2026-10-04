"""Local sentence embeddings for the semantic cache (fastembed / ONNX, CPU, $0 per call).

Thresholds are only meaningful for the model they were calibrated on: every cosine scale is different
(bge-small and MiniLM put the same paraphrase pair at different similarities). So the model name is
recorded on every embedder, written into the calibration JSON, and used to name the Qdrant collection.

    from costguard.cache.embedder import get_embedder
    emb = get_embedder()                 # process-wide singleton, model loaded on first use
    v = emb.embed_one("where is my order")          # (384,) float32, L2-normalised
    m = emb.embed(["a", "b"])                        # (2, 384)
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..config import ROOT

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# fastembed names of models we have calibrated or compared (dimension, licence)
KNOWN_MODELS = {
    "BAAI/bge-small-en-v1.5": (384, "MIT"),
    "sentence-transformers/all-MiniLM-L6-v2": (384, "Apache-2.0"),
}


def _cache_dir() -> str:
    # fastembed's own default is the OS temp dir, which macOS clears; keep weights under the (git-ignored) models/
    return os.environ.get("FASTEMBED_CACHE_PATH") or str(ROOT / "models" / "fastembed")


def _l2(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


class Embedder:
    """Thin, thread-safe wrapper over fastembed.TextEmbedding. Lazy: the ONNX model loads on first call."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Optional[str] = None, batch_size: int = 64,
                 threads: Optional[int] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or _cache_dir()
        self.batch_size = batch_size
        self.threads = threads
        self._model = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()  # onnxruntime sessions are thread-safe, but serialising keeps latency stable
        self.dim: Optional[int] = KNOWN_MODELS.get(model_name, (None, None))[0]
        self.last_ms: float = 0.0          # wall time of the most recent embed()/embed_one() call
        self.load_ms: float = 0.0

    # ------------------------------------------------------------------ loading
    def _ensure(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from fastembed import TextEmbedding  # heavy import, keep it lazy
                    t0 = time.perf_counter()
                    Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                    kw = {"cache_dir": self.cache_dir}
                    if self.threads:
                        kw["threads"] = self.threads
                    self._model = TextEmbedding(self.model_name, **kw)
                    self.load_ms = (time.perf_counter() - t0) * 1000
        return self._model

    def warmup(self) -> "Embedder":
        self.embed_one("warm up")
        return self

    # ------------------------------------------------------------------ embedding
    def embed(self, texts: Sequence[str] | Iterable[str], batch_size: Optional[int] = None) -> np.ndarray:
        texts = [t if isinstance(t, str) else str(t) for t in texts]
        model = self._ensure()
        t0 = time.perf_counter()
        if not texts:
            out = np.zeros((0, self.dim or 0), dtype=np.float32)
        else:
            with self._run_lock:
                vecs = list(model.embed(texts, batch_size=batch_size or self.batch_size))
            out = _l2(np.vstack(vecs))
            self.dim = out.shape[1]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return out

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def __repr__(self) -> str:
        return f"Embedder({self.model_name!r}, loaded={self._model is not None})"


_SINGLETONS: dict[str, Embedder] = {}
_SINGLETON_LOCK = threading.Lock()


def get_embedder(model_name: Optional[str] = None) -> Embedder:
    """Process-wide embedder per model name (loading an ONNX model costs ~0.3-1 s and ~130 MB, do it once)."""
    name = model_name or os.environ.get("COSTGUARD_EMBED_MODEL") or DEFAULT_MODEL
    with _SINGLETON_LOCK:
        if name not in _SINGLETONS:
            _SINGLETONS[name] = Embedder(name)
        return _SINGLETONS[name]


def embed(texts: Sequence[str]) -> np.ndarray:
    return get_embedder().embed(texts)


def embed_one(text: str) -> np.ndarray:
    return get_embedder().embed_one(text)
