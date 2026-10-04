"""One SQLite row per request (TraceRecord). Feeds the dashboard, /v1/stats and the README numbers.

Writes are off the request path: `log()` enqueues; a daemon thread batches inserts (one transaction per batch).
`rows()` / `flush()` wait for pending writes, so readers always see everything logged before the call.
The queue is bounded: if SQLite stalls, new rows are dropped and counted (`dropped`) instead of growing memory, and
failed inserts are counted (`write_errors`), never raised into serving. `flush()` has a deadline, so a stuck
writer cannot hang shutdown or /v1/stats.
"""
from __future__ import annotations

import atexit
import json
import queue
import sqlite3
import threading
import time
from pathlib import Path

from ..schemas import TraceRecord

_JSON_COLS = {"stage_ms", "stage_errors", "route_signals"}


class RequestLogger:
    def __init__(self, path: Path, batch_size: int = 200, flush_interval_s: float = 0.25, max_queue: int = 100_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cols = list(TraceRecord.model_fields)
        self._names = ", ".join(f'"{k}"' for k in self._cols)
        self._sql = f"INSERT INTO requests ({self._names}) VALUES ({', '.join('?' * len(self._cols))})"
        self._q: "queue.Queue[list | None]" = queue.Queue(maxsize=max_queue)
        self._batch_size, self._interval = batch_size, flush_interval_s
        self.dropped = 0            # rows refused because the queue was full (SQLite stalled)
        self.write_errors = 0       # rows lost to a failed insert
        self.last_error: str | None = None
        with self._conn() as c:
            c.execute(f"CREATE TABLE IF NOT EXISTS requests ({self._names})")
            # schema migration: add columns introduced after the table was created (TraceRecord grows additively)
            existing = {r[1] for r in c.execute("PRAGMA table_info(requests)")}
            for k in self._cols:
                if k not in existing:
                    c.execute(f'ALTER TABLE requests ADD COLUMN "{k}"')
            c.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_requests_arm ON requests(arm)")
        self._thread = threading.Thread(target=self._writer, name="costguard-logger", daemon=True)
        self._thread.start()
        atexit.register(self.flush)

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    # ------------------------------------------------------------------ write path
    def __call__(self, rec: TraceRecord) -> None:
        self.log(rec)

    def log(self, rec: TraceRecord) -> None:
        d = rec.model_dump()
        try:
            self._q.put_nowait([json.dumps(d[k]) if k in _JSON_COLS else d[k] for k in self._cols])
        except queue.Full:
            self.dropped += 1

    def _writer(self) -> None:
        conn = None
        while True:
            first = self._q.get()
            batch, done = [first], 1
            try:
                while len(batch) < self._batch_size:
                    batch.append(self._q.get(timeout=self._interval))
                    done += 1
            except queue.Empty:
                pass
            rows = [b for b in batch if b is not None]
            try:
                if rows:
                    conn = conn or self._conn()
                    with conn:
                        conn.executemany(self._sql, rows)
            except Exception as e:  # logging must never take the process down; count the loss instead
                self.write_errors += len(rows)
                self.last_error = f"{type(e).__name__}: {e}"[:300]
                conn = None
            finally:
                for _ in range(done):
                    self._q.task_done()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every queued row is written (or failed). False if the deadline passed or the writer died."""
        deadline = time.monotonic() + timeout
        with self._q.all_tasks_done:
            while self._q.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._thread.is_alive():
                    return False
                self._q.all_tasks_done.wait(min(remaining, 0.5))
        return True

    def health(self) -> dict:
        return {"queued": self._q.qsize(), "dropped": self.dropped, "write_errors": self.write_errors,
                "last_error": self.last_error, "writer_alive": self._thread.is_alive()}

    # ------------------------------------------------------------------ read path
    def rows(self, where: str = "", params: tuple = ()) -> list[dict]:
        self.flush()
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            out = [dict(r) for r in c.execute(f"SELECT * FROM requests {where}", params)]
        for r in out:
            for k in _JSON_COLS:
                if r.get(k):
                    r[k] = json.loads(r[k])
        return out
