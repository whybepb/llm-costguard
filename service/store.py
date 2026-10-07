"""Redis-backed exact and semantic caches, scoped per tenant.

Exact:    cg:exact:<tenant>:<sha256 key>  -> JSON entry (plain string key, with TTL)
Semantic: cg:sem:<tenant>:<sha256 key>    -> HASH with the query embedding, indexed by
          RediSearch (idx:cg_sem, FLAT, cosine). A KNN search filtered by tenant
          returns the nearest cached queries; the app then runs the context check,
          the numbers/negation guards and the LLM judge on them, best first, the
          same order as cache/semantic.py.
"""

import json
import re
import time

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from config import CFG

INDEX = "idx:cg_sem"
DIM = 384  # bge-small-en-v1.5
TTL_SECONDS = CFG["cache"]["ttl_seconds"]
KNN = CFG["cache"]["semantic_knn"]


def tenant_slug(tenant):
    """Tenant ids become key parts and TAG values: keep them to a safe alphabet."""
    slug = re.sub(r"[^a-z0-9_]", "_", (tenant or "default").lower())[:64]
    return slug or "default"


class CacheStore:
    def __init__(self, url="redis://localhost:6379/0"):
        self.r = redis.Redis.from_url(url)
        self._ensure_index()

    def _ensure_index(self):
        try:
            self.r.ft(INDEX).info()
        except redis.ResponseError:
            self.r.ft(INDEX).create_index(
                [
                    TagField("tenant"),
                    TextField("query"),
                    NumericField("created"),
                    VectorField("vec", "FLAT", {"TYPE": "FLOAT32", "DIM": DIM, "DISTANCE_METRIC": "COSINE"}),
                ],
                definition=IndexDefinition(prefix=["cg:sem:"], index_type=IndexType.HASH),
            )

    def ping(self):
        return self.r.ping()

    # --- exact ---------------------------------------------------------------
    def exact_get(self, tenant, key):
        raw = self.r.get(f"cg:exact:{tenant}:{key}")
        return json.loads(raw) if raw else None

    def exact_put(self, tenant, key, entry):
        self.r.set(f"cg:exact:{tenant}:{key}", json.dumps(entry), ex=TTL_SECONDS)

    # --- semantic ------------------------------------------------------------
    def semantic_candidates(self, tenant, vec, k=KNN):
        """Nearest cached entries for this tenant: list of (cosine, entry), best first."""
        q = (Query(f"(@tenant:{{{tenant}}})=>[KNN {k} @vec $v AS dist]")
             .sort_by("dist").return_fields("dist", "entry").paging(0, k).dialect(2))
        params = {"v": np.asarray(vec, np.float32).tobytes()}
        try:
            res = self.r.ft(INDEX).search(q, query_params=params)
        except redis.ResponseError as e:
            if "no such index" not in str(e).lower():
                raise
            self._ensure_index()  # e.g. after a FLUSHDB/FLUSHALL; creation re-indexes existing keys
            res = self.r.ft(INDEX).search(q, query_params=params)
        return [(1.0 - float(d.dist), json.loads(d.entry)) for d in res.docs]

    def semantic_put(self, tenant, key, vec, query, entry):
        k = f"cg:sem:{tenant}:{key}"
        self.r.hset(k, mapping={
            "tenant": tenant, "query": query, "created": int(time.time()),
            "vec": np.asarray(vec, np.float32).tobytes(), "entry": json.dumps(entry),
        })
        self.r.expire(k, TTL_SECONDS)

    # --- admin ---------------------------------------------------------------
    def counts(self, tenant=None):
        pat = tenant or "*"
        return {"exact": sum(1 for _ in self.r.scan_iter(f"cg:exact:{pat}:*", count=500)),
                "semantic": sum(1 for _ in self.r.scan_iter(f"cg:sem:{pat}:*", count=500))}

    def clear(self, tenant=None):
        pat = tenant or "*"
        n = 0
        for prefix in ("cg:exact", "cg:sem"):
            for k in self.r.scan_iter(f"{prefix}:{pat}:*", count=500):
                n += self.r.delete(k)
        return n

    def log_request(self, record, keep=200):
        self.r.lpush("cg:requests", json.dumps(record))
        self.r.ltrim("cg:requests", 0, keep - 1)

    def recent(self, n=50):
        return [json.loads(x) for x in self.r.lrange("cg:requests", 0, n - 1)]
