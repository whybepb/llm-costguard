"""One SQLite row per request (TraceRecord). Feeds the dashboard, /v1/stats and the README numbers.

Writes are off the request path: `log()` enqueues; a daemon thread batches inserts (one transaction per batch).
`rows()` / `flush()` wait for pending writes, so readers always see everything logged before the call.
"""
from __future__ import annotations

import atexit
import json
import queue
import sqlite3
import threading
from pathlib import Path

from ..schemas import TraceRecord

_JSON_COLS = {"stage_ms", "stage_errors", "route_signals"}


class RequestLogger:
    def __init__(self, path: Path, batch_size: int = 200, flush_interval_s: float = 0.25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cols = list(TraceRecord.model_fields)
        self._names = ", ".join(f'"{k}"' for k in self._cols)
        self._sql = f"INSERT INTO requests ({self._names}) VALUES ({', '.join('?' * len(self._cols))})"
        self._q: "queue.Queue[list | None]" = queue.Queue()
        self._batch_size, self._interval = batch_size, flush_interval_s
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
        self._q.put([json.dumps(d[k]) if k in _JSON_COLS else d[k] for k in self._cols])

    def _writer(self) -> None:
        conn = self._conn()
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
                    with conn:
                        conn.executemany(self._sql, rows)
            except Exception:  # logging must never take the process down
                pass
            finally:
                for _ in range(done):
                    self._q.task_done()

    def flush(self) -> None:
        self._q.join()

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
