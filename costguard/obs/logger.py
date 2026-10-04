"""One SQLite row per request (TraceRecord). Feeds the dashboard, /v1/stats and the README numbers."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..schemas import TraceRecord

_JSON_COLS = {"stage_ms", "stage_errors", "route_signals"}


class RequestLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cols = list(TraceRecord.model_fields)
        with self._conn() as c:
            cols = ", ".join(f'"{k}"' for k in self._cols)
            c.execute(f"CREATE TABLE IF NOT EXISTS requests ({cols})")
            # schema migration: add columns introduced after the table was created (TraceRecord grows additively)
            existing = {r[1] for r in c.execute("PRAGMA table_info(requests)")}
            for k in self._cols:
                if k not in existing:
                    c.execute(f'ALTER TABLE requests ADD COLUMN "{k}"')
            c.execute("CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_requests_arm ON requests(arm)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=30)

    def __call__(self, rec: TraceRecord) -> None:
        self.log(rec)

    def log(self, rec: TraceRecord) -> None:
        d = rec.model_dump()
        vals = [json.dumps(d[k]) if k in _JSON_COLS else d[k] for k in self._cols]
        with self._lock, self._conn() as c:
            names = ", ".join(f'"{k}"' for k in self._cols)
            c.execute(f"INSERT INTO requests ({names}) VALUES ({', '.join('?' * len(self._cols))})", vals)

    def rows(self, where: str = "", params: tuple = ()) -> list[dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            out = [dict(r) for r in c.execute(f"SELECT * FROM requests {where}", params)]
        for r in out:
            for k in _JSON_COLS:
                if r.get(k):
                    r[k] = json.loads(r[k])
        return out
