"""Tier 1: exact-match response cache.

The engine builds the key (pipeline.py): sha256 over partition (tenant | system-prompt sha256 | kb_version |
max_tokens | context digest), the normalised query (lower-cased, whitespace collapsed, trailing punctuation dropped)
and a hash of the retrieved context. So a direct hit is the same question up to that normalisation (case-only
differences share a key); semantic hits promoted here carry their similarity and the engine re-checks it against the
current mode's tau. It costs one dict lookup (~1 µs) and runs before the semantic tier, which costs an embedding
(~2-5 ms) and can be wrong.

In-process, thread-safe, TTL + LRU bounded. A multi-replica deployment would swap this for Redis behind the
same three-method interface (`ExactCache`).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

from ..schemas import CacheEntry

DEFAULT_MAX_ENTRIES = 50_000


class InMemoryExactCache:
    def __init__(self, ttl_seconds: float = 7 * 24 * 3600, max_entries: int = DEFAULT_MAX_ENTRIES,
                 clock: Callable[[], float] = time.time):
        self.ttl = float(ttl_seconds) if ttl_seconds else None
        self.max_entries = int(max_entries)
        self._clock = clock
        self._d: "OrderedDict[str, tuple[float, CacheEntry]]" = OrderedDict()  # key -> (expires_at, entry)
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.expired = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        now = self._clock()
        with self._lock:
            item = self._d.get(key)
            if item is None:
                self.misses += 1
                return None
            expires, entry = item
            if expires <= now:
                del self._d[key]
                self.expired += 1
                self.misses += 1
                return None
            self._d.move_to_end(key)  # LRU: most recently used at the end
            self.hits += 1
            return entry

    def put(self, key: str, entry: CacheEntry) -> None:
        expires = self._clock() + self.ttl if self.ttl else float("inf")
        with self._lock:
            old = self._d.get(key)
            # never overwrite a strong-tier answer with a cheap-tier one (quality mode only serves strong answers)
            if old is not None and old[0] > self._clock() and old[1].model_alias == "strong" \
                    and entry.model_alias != "strong":
                return
            self._d[key] = (expires, entry)
            self._d.move_to_end(key)
            while len(self._d) > self.max_entries:
                self._d.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def size(self) -> int:
        # deliberately not __len__: an empty cache must stay truthy (the engine does `exact_cache or NoExactCache()`)
        return len(self._d)

    def stats(self) -> dict:
        return {"entries": len(self._d), "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "expired": self.expired, "ttl_seconds": self.ttl, "max_entries": self.max_entries}


def build_exact_cache(settings, policy) -> InMemoryExactCache:
    c = getattr(policy, "cache", {}) or {}
    return InMemoryExactCache(ttl_seconds=float(c.get("ttl_seconds", 7 * 24 * 3600)),
                              max_entries=int(c.get("exact_max_entries", DEFAULT_MAX_ENTRIES)))
