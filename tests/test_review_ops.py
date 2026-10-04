"""Regression tests for review findings on the request log (#18) and semantic-cache memory bounds (#17)."""
from __future__ import annotations

import time

import numpy as np

from costguard.cache.semantic import MemoryStore, _Rec
from costguard.obs.logger import RequestLogger
from costguard.schemas import CacheEntry, TraceRecord


def _rec(i: int, expires: float = float("inf")) -> _Rec:
    e = CacheEntry(entry_id=str(i), query_text=f"q{i}", response_text="a", model_alias="strong",
                   input_tokens=1, output_tokens=1)
    return _Rec(id=str(i), query=f"q{i}", entry=e, expires_at=expires, created_at=0.0)


def _vec(dim: int = 4) -> np.ndarray:
    v = np.random.default_rng(0).random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_logger_full_queue_drops_and_counts_instead_of_growing(tmp_path):
    log = RequestLogger(tmp_path / "log.sqlite", max_queue=2, flush_interval_s=5.0)
    log._q.put_nowait(None)        # occupy the queue while the writer is blocked waiting for a batch to fill
    log._q.put_nowait(None)
    for _ in range(3):
        log.log(TraceRecord(request_id="r", ts=0.0))
    assert log.dropped >= 1
    assert log.flush(timeout=10.0)


def test_logger_failed_insert_is_counted_and_flush_returns(tmp_path):
    log = RequestLogger(tmp_path / "log.sqlite")
    log._sql = "INSERT INTO no_such_table VALUES (1)"     # every insert now fails
    log.log(TraceRecord(request_id="r1", ts=0.0))
    assert log.flush(timeout=5.0)
    h = log.health()
    assert h["write_errors"] == 1 and h["last_error"] and h["writer_alive"]


def test_logger_flush_has_a_deadline(tmp_path):
    log = RequestLogger(tmp_path / "log.sqlite")
    with log._q.all_tasks_done:
        log._q.unfinished_tasks += 1                     # a task that will never complete
    t0 = time.monotonic()
    assert log.flush(timeout=0.3) is False
    assert time.monotonic() - t0 < 2.0
    with log._q.all_tasks_done:
        log._q.unfinished_tasks -= 1


def test_memory_store_drops_expired_and_empty_partitions():
    s = MemoryStore()
    for i in range(100):
        s.add(f"p{i}", _vec(), _rec(i, expires=10.0))
    assert s.partitions() == 100
    s.purge_expired(now=20.0)
    assert s.partitions() == 0 and s.count() == 0


def test_memory_store_global_bound_evicts_least_recent_partitions():
    s = MemoryStore(max_entries_total=50)
    for i in range(120):
        s.add(f"p{i}", _vec(), _rec(i))
    s.touch("p0", s._parts["p0"].recs[0], now=1.0)     # p0 was just hit: it should survive
    s.purge_expired(now=1.0)
    assert s.count() <= 50 and "p0" in s._parts and "p1" not in s._parts
    assert s.evictions >= 70
