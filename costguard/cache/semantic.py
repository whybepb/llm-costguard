"""Tier 2: semantic response cache (embed -> nearest neighbour in the same partition -> threshold -> guards).

    lookup(query, partition, tau)
      1. embed the query locally (bge-small, ~2-5 ms on CPU, $0)
      2. top-k cosine neighbours *inside this partition only* (tenant | system-prompt hash | kb_version | ctx)
      3. walk candidates with similarity >= tau, best first; serve the first one every guard accepts
      4. otherwise a miss; `similarity` / `neighbor_query` always describe the best neighbour (also on a miss)
         and `guard_rejected` says why a near-hit was refused

    insert(query, partition, entry): skip near-exact duplicates (cosine >= 0.98 and guards agree), except that a
    strong-tier answer replaces a cheap-tier one (quality mode never serves cheap answers).

Two storage backends behind one small interface:
  memory  per-partition float32 matrix + brute-force dot product (exact search; ~1 ms at 10k entries)
  qdrant  qdrant-client; server at settings.qdrant_url, else embedded local mode under data/runtime/qdrant
Entries expire after policy.cache.ttl_seconds. kb_version is part of the partition, so bumping it in
policy.yaml invalidates every semantic entry at once without touching storage.
"""
from __future__ import annotations

import atexit
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

from ..config import ROOT
from ..schemas import CacheEntry, SemanticHit
from . import guards as G

DEFAULT_TOP_K = 5
DEFAULT_DEDUP = 0.98
DEFAULT_MAX_PER_PARTITION = 50_000
DEFAULT_MAX_TOTAL = 200_000          # across all partitions (memory backend)
PURGE_EVERY = 1024                   # inserts between expiry sweeps (both backends)


@dataclass
class _Rec:
    id: str
    query: str
    entry: CacheEntry
    expires_at: float
    created_at: float
    last_used: float = 0.0


class _Store(Protocol):
    def search(self, partition: str, vec: np.ndarray, k: int, now: float) -> list[tuple[float, _Rec]]: ...
    def add(self, partition: str, vec: np.ndarray, rec: _Rec) -> None: ...
    def replace(self, partition: str, old: _Rec, vec: np.ndarray, rec: _Rec) -> None: ...
    def touch(self, partition: str, rec: _Rec, now: float) -> None: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...


# ============================================================================================ memory backend


class _Partition:
    __slots__ = ("vecs", "recs", "expires", "alive", "n", "dead")

    def __init__(self, dim: int, cap: int = 256):
        self.vecs = np.zeros((cap, dim), dtype=np.float32)
        self.expires = np.zeros(cap, dtype=np.float64)
        self.alive = np.zeros(cap, dtype=bool)
        self.recs: list[Optional[_Rec]] = []
        self.n = 0
        self.dead = 0


class MemoryStore:
    """Exact brute-force cosine search over a per-partition numpy matrix (vectors are L2-normalised)."""

    name = "memory"

    def __init__(self, max_entries_per_partition: int = DEFAULT_MAX_PER_PARTITION,
                 max_entries_total: int = DEFAULT_MAX_TOTAL):
        self.max_per_partition = int(max_entries_per_partition)
        self.max_total = int(max_entries_total)
        self._parts: "OrderedDict[str, _Partition]" = OrderedDict()   # least recently written/hit first
        self._adds = 0
        self.evictions = 0

    def search(self, partition, vec, k, now):
        p = self._parts.get(partition)
        if p is None or p.n == 0:
            return []
        sims = p.vecs[: p.n] @ vec
        valid = p.alive[: p.n] & (p.expires[: p.n] > now)
        if not valid.any():
            return []
        sims = np.where(valid, sims, -np.inf)
        k = min(k, int(valid.sum()))
        idx = np.argpartition(-sims, k - 1)[:k] if k < p.n else np.arange(p.n)
        idx = idx[np.argsort(-sims[idx])][:k]
        return [(float(sims[i]), p.recs[i]) for i in idx if np.isfinite(sims[i])]

    def _index_of(self, p: _Partition, rec: _Rec) -> Optional[int]:
        for i in range(p.n - 1, -1, -1):
            if p.recs[i] is rec:
                return i
        return None

    def add(self, partition, vec, rec):
        p = self._parts.get(partition)
        if p is None:
            p = self._parts[partition] = _Partition(vec.shape[0])
        self._parts.move_to_end(partition)
        if p.n >= self.max_per_partition:
            self._compact(p, now=time.time(), make_room=True)
        if p.n == p.vecs.shape[0]:
            cap = p.vecs.shape[0] * 2
            p.vecs = np.resize(p.vecs, (cap, p.vecs.shape[1]))
            p.expires = np.resize(p.expires, cap)
            p.alive = np.resize(p.alive, cap)
            p.alive[p.n:] = False
        i = p.n
        p.vecs[i] = vec
        p.expires[i] = rec.expires_at
        p.alive[i] = True
        p.recs.append(rec)
        p.n += 1
        self._adds += 1
        if self._adds % PURGE_EVERY == 0:
            self.purge_expired()

    def replace(self, partition, old, vec, rec):
        p = self._parts.get(partition)
        i = self._index_of(p, old) if p else None
        if i is None:
            self.add(partition, vec, rec)
            return
        p.vecs[i], p.expires[i], p.recs[i] = vec, rec.expires_at, rec

    def touch(self, partition, rec, now):
        rec.last_used = now
        if partition in self._parts:
            self._parts.move_to_end(partition)

    def _compact(self, p: _Partition, now: float, make_room: bool = False):
        keep = [i for i in range(p.n) if p.alive[i] and p.expires[i] > now]
        if make_room and len(keep) >= self.max_per_partition:
            # drop the least recently used 10% (last_used falls back to created_at)
            keep.sort(key=lambda i: max(p.recs[i].last_used, p.recs[i].created_at))
            drop = max(1, len(keep) // 10)
            self.evictions += drop
            keep = sorted(keep[drop:])
        m = len(keep)
        p.vecs[:m] = p.vecs[keep]
        p.expires[:m] = p.expires[keep]
        p.alive[:m] = True
        p.alive[m:] = False
        p.recs = [p.recs[i] for i in keep]
        p.n = m

    def purge_expired(self, now: Optional[float] = None) -> None:
        """Drop expired entries and empty partitions, then enforce the global bound by evicting whole partitions,
        least recently used first (many distinct system prompts must not grow memory without limit)."""
        now = time.time() if now is None else now
        for key in list(self._parts):
            p = self._parts[key]
            self._compact(p, now)
            if p.n == 0:
                del self._parts[key]
        total = self.count()
        while total > self.max_total and len(self._parts) > 1:
            _, p = self._parts.popitem(last=False)
            total -= p.n
            self.evictions += p.n

    def clear(self):
        self._parts.clear()

    def count(self) -> int:
        return sum(p.n for p in self._parts.values())

    def partitions(self) -> int:
        return len(self._parts)


# ============================================================================================ qdrant backend


class QdrantStore:
    """One collection for all partitions; `partition` is a payload filter (keyword index)."""

    name = "qdrant"

    def __init__(self, dim: int, collection: str, url: Optional[str] = None, path: Optional[str | Path] = None):
        from qdrant_client import QdrantClient, models
        self._m = models
        if url:
            self.client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))  # Qdrant Cloud needs a key
            self.location = url
        else:
            path = Path(path or ROOT / "data" / "runtime" / "qdrant")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.location = str(path)
            atexit.register(self.close)   # release the embedded store's file lock before interpreter teardown
        self.collection, self.dim, self.remote = collection, dim, bool(url)
        self._ensure()

    def _ensure(self):
        m = self._m
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection,
                                          vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.COSINE))
            if self.remote:  # embedded local mode has no payload indexes (filters still work, by scan)
                self.client.create_payload_index(self.collection, "partition", m.PayloadSchemaType.KEYWORD)

    @staticmethod
    def _to_payload(partition: str, rec: _Rec) -> dict:
        return {"partition": partition, "query": rec.query, "entry": rec.entry.model_dump(mode="json"),
                "expires_at": rec.expires_at, "created_at": rec.created_at}

    @staticmethod
    def _from_point(pt) -> _Rec:
        pl = pt.payload
        return _Rec(id=str(pt.id), query=pl["query"], entry=CacheEntry.model_validate(pl["entry"]),
                    expires_at=float(pl["expires_at"]), created_at=float(pl["created_at"]))

    def search(self, partition, vec, k, now):
        m = self._m
        flt = m.Filter(must=[m.FieldCondition(key="partition", match=m.MatchValue(value=partition)),
                             m.FieldCondition(key="expires_at", range=m.Range(gt=now))])
        res = self.client.query_points(self.collection, query=vec.astype(np.float32).tolist(), query_filter=flt,
                                       limit=k, with_payload=True)
        return [(float(pt.score), self._from_point(pt)) for pt in res.points]

    def add(self, partition, vec, rec):
        self.client.upsert(self.collection, points=[
            self._m.PointStruct(id=rec.id, vector=vec.astype(np.float32).tolist(), payload=self._to_payload(partition, rec))])
        self._adds = getattr(self, "_adds", 0) + 1
        if self._adds % PURGE_EVERY == 0:
            self.purge_expired()

    def purge_expired(self, now: Optional[float] = None) -> None:
        """Search filters expired points out; this deletes them so storage does not grow without limit."""
        m, now = self._m, time.time() if now is None else now
        try:
            self.client.delete(self.collection, points_selector=m.FilterSelector(filter=m.Filter(
                must=[m.FieldCondition(key="expires_at", range=m.Range(lt=now))])))
        except Exception:  # cleanup is best-effort; never fail an insert over it
            pass

    def replace(self, partition, old, vec, rec):
        rec.id = old.id
        self.add(partition, vec, rec)

    def touch(self, partition, rec, now):
        pass

    def clear(self):
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._ensure()

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


# ============================================================================================ the cache


class SemanticCacheImpl:
    """Satisfies costguard.interfaces.SemanticCache."""

    def __init__(self, embedder, store: Optional[_Store] = None, ttl_seconds: Optional[float] = 7 * 24 * 3600,
                 guards: G.GuardConfig = G.ALL_ON, top_k: int = DEFAULT_TOP_K, dedup_threshold: float = DEFAULT_DEDUP,
                 clock: Callable[[], float] = time.time):
        self.embedder = embedder
        self.store = store if store is not None else MemoryStore()
        self.ttl = float(ttl_seconds) if ttl_seconds else None
        self.guards = guards
        self.top_k = int(top_k)
        self.dedup_threshold = float(dedup_threshold)
        self._clock = clock
        self._lock = threading.RLock()
        self._vec_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()  # lookup() then insert() embeds once
        # observability
        self.last_embed_ms = 0.0
        self.last_search_ms = 0.0
        self.last_guard_ms = 0.0
        self.last_lookup: dict[str, Any] = {}
        self.embed_ms_history: deque = deque(maxlen=10_000)
        self.search_ms_history: deque = deque(maxlen=10_000)
        self.counters = {"lookups": 0, "hits": 0, "guard_rejections": 0, "inserts": 0, "dedup_skips": 0,
                         "tier_upgrades": 0}

    # ------------------------------------------------------------------ helpers
    @property
    def model_name(self) -> str:
        return getattr(self.embedder, "model_name", "unknown")

    def _vec(self, text: str) -> np.ndarray:
        with self._lock:
            v = self._vec_cache.get(text)
            if v is not None:
                self._vec_cache.move_to_end(text)
                self.last_embed_ms = 0.0
                return v
        t0 = time.perf_counter()
        v = np.asarray(self.embedder.embed_one(text), dtype=np.float32)
        self.last_embed_ms = (time.perf_counter() - t0) * 1000
        self.embed_ms_history.append(self.last_embed_ms)
        with self._lock:
            self._vec_cache[text] = v
            while len(self._vec_cache) > 512:
                self._vec_cache.popitem(last=False)
        return v

    # ------------------------------------------------------------------ interface
    def lookup(self, query: str, partition: str, threshold: float) -> SemanticHit:
        q = self._vec(query)
        now = self._clock()
        with self._lock:
            self.counters["lookups"] += 1
            t0 = time.perf_counter()
            cands = self.store.search(partition, q, self.top_k, now)
            self.last_search_ms = (time.perf_counter() - t0) * 1000
            self.search_ms_history.append(self.last_search_ms)
            self.last_lookup = {"candidates": [(round(s, 4), r.query) for s, r in cands], "rejected": [],
                                "served": None}
            if not cands:
                return SemanticHit()
            best_sim, best = cands[0]
            first_reason: Optional[str] = None
            tg = time.perf_counter()
            for sim, rec in cands:
                if sim < threshold:
                    break
                reason = G.check(query, rec.query, self.guards) if self.guards.any else None
                if reason is None:
                    self.store.touch(partition, rec, now)
                    self.counters["hits"] += 1
                    self.last_guard_ms = (time.perf_counter() - tg) * 1000
                    self.last_lookup["served"] = rec
                    return SemanticHit(entry=rec.entry, similarity=round(sim, 4), neighbor_query=rec.query)
                self.last_lookup["rejected"].append((reason, rec))
                first_reason = first_reason or reason
            self.last_guard_ms = (time.perf_counter() - tg) * 1000
            if first_reason:
                self.counters["guard_rejections"] += 1
            return SemanticHit(similarity=round(best_sim, 4), neighbor_query=best.query, guard_rejected=first_reason)

    def insert(self, query: str, partition: str, entry: CacheEntry) -> None:
        q = self._vec(query)
        now = self._clock()
        rec = _Rec(id=str(uuid.uuid4()), query=query, entry=entry, created_at=now,
                   expires_at=now + self.ttl if self.ttl else float("inf"))
        with self._lock:
            for sim, old in self.store.search(partition, q, 3, now):
                if sim < self.dedup_threshold:
                    break
                if G.check(query, old.query, G.ALL_ON) is not None:   # "order 4821" vs "order 4822": not a dup
                    continue
                if entry.model_alias == "strong" and old.entry.model_alias != "strong":
                    self.store.replace(partition, old, q, rec)
                    self.counters["tier_upgrades"] += 1
                else:
                    self.counters["dedup_skips"] += 1
                return
            self.store.add(partition, q, rec)
            self.counters["inserts"] += 1

    def clear(self) -> None:
        with self._lock:
            self.store.clear()
            self._vec_cache.clear()

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        def pct(xs, p):
            return round(float(np.percentile(list(xs), p)), 3) if xs else None
        return {"model": self.model_name, "backend": getattr(self.store, "name", "?"), "entries": self.store.count(),
                "guards": {k: getattr(self.guards, k) for k in ("numbers", "negation", "entities", "content")},
                "embed_ms_p50": pct(self.embed_ms_history, 50), "embed_ms_p99": pct(self.embed_ms_history, 99),
                "search_ms_p50": pct(self.search_ms_history, 50), "search_ms_p99": pct(self.search_ms_history, 99),
                **self.counters}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def build_semantic_cache(settings, policy) -> SemanticCacheImpl:
    from .embedder import get_embedder
    c = getattr(policy, "cache", {}) or {}
    emb = get_embedder(c.get("embedding_model"))
    emb.warmup()  # load the ONNX model at startup so a broken install fails the build (-> no-op stage), not a request
    ttl = float(c.get("ttl_seconds", 7 * 24 * 3600))
    guards = G.GuardConfig.from_mapping(c.get("guards"))
    backend = (getattr(settings, "semantic_backend", "memory") or "memory").lower()
    if backend == "qdrant":
        store = QdrantStore(dim=emb.dim, collection=f"costguard_semcache__{_slug(emb.model_name)}",
                            url=getattr(settings, "qdrant_url", None))
    elif backend == "memory":
        store = MemoryStore(int(c.get("semantic_max_entries", DEFAULT_MAX_PER_PARTITION)),
                            int(c.get("semantic_max_entries_total", DEFAULT_MAX_TOTAL)))
    else:
        raise ValueError(f"unknown semantic_backend {backend!r} (memory | qdrant)")
    return SemanticCacheImpl(emb, store, ttl_seconds=ttl, guards=guards,
                             top_k=int(c.get("semantic_top_k", DEFAULT_TOP_K)),
                             dedup_threshold=float(c.get("semantic_dedup_threshold", DEFAULT_DEDUP)))
