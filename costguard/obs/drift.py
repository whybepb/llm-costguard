"""Online drift monitoring with the population stability index (course W3S2), on features already in the TraceRecord.

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)
    bands: < 0.10 stable | 0.10-0.25 investigate | > 0.25 alert            (W3S2 notebook bands)
    e.g. one bin moving 20% -> 30% contributes 0.10 * ln(1.5) = 0.0405

No embeddings in the hot path: the hook appends one small tuple per request (O(1)); PSI is recomputed lazily on
`/v1/drift`, on `/metrics` scrapes and every `recompute_every` requests (O(window), ~1 ms for 500 rows).

Features (each maps to a course monitoring category):
  input_tokens      numeric      input     usage-pattern drift (W7S2: input-token percentiles)
  category          categorical  input     topic/intent mix (W7S2 topic drift, using the request's category hint)
  cache_similarity  numeric      drift     top-1 cosine distribution (embedding-model change, new paraphrase mix)
  output_tokens     numeric      output    answer length on upstream calls (short generic answers after a downshift)
  route             categorical  drift     exact / semantic / strong / cheap / bypass mix (hit-rate drop, router shift)

Numeric bins are 10 baseline-quantile bins with OPEN outer edges, so values outside the baseline range land in the
first/last bin instead of being dropped (the W3S2 notebook's np.histogram undercounts them).

Baseline: loaded from COSTGUARD_DRIFT_BASELINE (a JSON snapshot, e.g. built from the A0 replay log with
`python -m costguard.obs.drift --db eval/results/ab_A0.sqlite --out configs/drift_baseline.json`), otherwise frozen
from the first `baseline_size` requests after start. `POST /v1/drift/baseline` re-freezes it after an intended change.
"""
from __future__ import annotations

import argparse
import bisect
import json
import logging
import math
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

log = logging.getLogger("costguard.obs")

PSI_STABLE = 0.10
PSI_ALERT = 0.25
EPS = 1e-4
CATEGORIES = ("order", "shipping", "returns", "refund", "payment", "account", "product", "other", "none")
ROUTES = ("exact", "semantic", "strong", "cheap", "bypass", "other")


# ---------------------------------------------------------------- PSI maths (pure functions)
def _normalise(xs: Sequence[float]) -> list[float]:
    s = float(sum(xs))
    return [float(x) / s for x in xs] if s > 0 else [0.0 for _ in xs]


def psi_term(expected: float, actual: float, eps: float = EPS) -> float:
    """One bin's contribution: (a - e) * ln(a / e), with both shares floored at eps (empty bins)."""
    e, a = max(float(expected), eps), max(float(actual), eps)
    return (a - e) * math.log(a / e)


def psi(expected: Sequence[float], actual: Sequence[float], eps: float = EPS) -> float:
    """PSI between two binned distributions (counts or shares; each is normalised to sum to 1)."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have the same number of bins")
    e, a = _normalise(expected), _normalise(actual)
    return float(sum(psi_term(x, y, eps) for x, y in zip(e, a)))


def psi_band(value: Optional[float]) -> str:
    if value is None:
        return "insufficient_data"
    if value < PSI_STABLE:
        return "stable"
    if value <= PSI_ALERT:
        return "investigate"
    return "alert"


def quantile_edges(values: Sequence[float], bins: int = 10) -> list[float]:
    """Interior bin edges at the baseline's 1/bins .. (bins-1)/bins quantiles (deduplicated for tied data)."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return []
    edges = []
    for i in range(1, bins):
        q = i / bins * (len(xs) - 1)
        lo, hi = int(math.floor(q)), int(math.ceil(q))
        edges.append(xs[lo] + (xs[hi] - xs[lo]) * (q - lo))
    out: list[float] = []
    for e in edges:
        if not out or e > out[-1]:
            out.append(round(e, 6))
    return out


def bin_numeric(values: Iterable[float], edges: Sequence[float]) -> list[int]:
    """Counts over (-inf, e1], (e1, e2], ..., (ek, +inf)."""
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[bisect.bisect_left(edges, float(v))] += 1
    return counts


def bin_categorical(values: Iterable[str], categories: Sequence[str]) -> list[int]:
    idx = {c: i for i, c in enumerate(categories)}
    other = idx.get("other", len(categories) - 1)
    counts = [0] * len(categories)
    for v in values:
        counts[idx.get(v, other)] += 1
    return counts


# ---------------------------------------------------------------- features
def _get(r: Any, key: str, default: Any = None) -> Any:
    v = r.get(key, default) if isinstance(r, dict) else getattr(r, key, default)
    return default if isinstance(v, float) and math.isnan(v) else v     # pandas rows carry NaN for NULL


def _route(r: Any) -> str:
    cs = _get(r, "cache_status")
    if cs in ("exact", "semantic"):
        return cs
    if cs in ("bypass",):
        return "bypass"
    m = _get(r, "model_used")
    return m if m in ("strong", "cheap") else "other"


def _category(r: Any) -> str:
    c = _get(r, "category")
    if not c:
        return "none"
    c = str(c).lower()
    return c if c in CATEGORIES else "other"


def _pos(key: str) -> Callable[[Any], Optional[float]]:
    def f(r: Any) -> Optional[float]:
        v = _get(r, key)
        return float(v) if v else None
    return f


def _output_tokens(r: Any) -> Optional[float]:
    return float(_get(r, "output_tokens") or 0) if _get(r, "model_used") else None   # upstream calls only


def _similarity(r: Any) -> Optional[float]:
    v = _get(r, "cache_similarity")
    return float(v) if v is not None else None


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str                                   # "numeric" | "categorical"
    extract: Callable[[Any], Any]               # TraceRecord or log-row dict -> value (None = not applicable)
    category: str                               # course monitoring category
    categories: tuple[str, ...] = ()
    bins: int = 10


DEFAULT_FEATURES: tuple[Feature, ...] = (
    Feature("input_tokens", "numeric", _pos("input_tokens_original"), "input"),
    Feature("category", "categorical", _category, "input", categories=CATEGORIES),
    Feature("cache_similarity", "numeric", _similarity, "drift"),
    Feature("output_tokens", "numeric", _output_tokens, "output"),
    Feature("route", "categorical", _route, "drift", categories=ROUTES),
)


@dataclass
class _Baseline:
    edges: dict[str, list[float]] = field(default_factory=dict)
    shares: dict[str, list[float]] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    n: int = 0
    created_at: float = 0.0
    source: str = "warmup"


class DriftMonitor:
    def __init__(self, features: Sequence[Feature] = DEFAULT_FEATURES, *, window: int = 500,
                 baseline_size: int = 500, min_samples: int = 50, baseline: Optional[dict] = None):
        self.features = tuple(features)
        self.window_size, self.baseline_size, self.min_samples = int(window), int(baseline_size), int(min_samples)
        self._lock = threading.Lock()
        self._window: deque = deque(maxlen=self.window_size)
        self._warmup: list[tuple] = []
        self._baseline: Optional[_Baseline] = None
        self.n_seen = 0
        if baseline:
            self.load_baseline(baseline)

    @classmethod
    def from_env(cls) -> "DriftMonitor":
        e = os.environ.get
        mon = cls(window=int(e("COSTGUARD_DRIFT_WINDOW", 500)), baseline_size=int(e("COSTGUARD_DRIFT_BASELINE_N", 500)),
                  min_samples=int(e("COSTGUARD_DRIFT_MIN_SAMPLES", 50)))
        path = e("COSTGUARD_DRIFT_BASELINE")
        if path:
            try:
                mon.load_baseline(json.loads(Path(path).read_text()))
            except Exception as ex:  # a bad baseline file must not stop serving
                log.warning("could not load drift baseline %s: %s (falling back to warm-up)", path, ex)
        return mon

    # ------------------------------------------------------------ hot path
    def _row(self, rec: Any) -> tuple:
        out = []
        for f in self.features:
            try:
                out.append(f.extract(rec))
            except Exception:
                out.append(None)
        return tuple(out)

    def observe(self, rec: Any) -> None:
        row = self._row(rec)
        with self._lock:
            self.n_seen += 1
            self._window.append(row)
            if self._baseline is None:
                self._warmup.append(row)
                if len(self._warmup) >= self.baseline_size:
                    self._baseline = self._fit(self._warmup, "warmup")
                    self._warmup = []

    # ------------------------------------------------------------ baseline
    def _fit(self, rows: Sequence[tuple], source: str) -> _Baseline:
        b = _Baseline(n=len(rows), created_at=time.time(), source=source)
        for i, f in enumerate(self.features):
            vals = [r[i] for r in rows if r[i] is not None]
            b.counts[f.name] = len(vals)
            if f.kind == "numeric":
                edges = quantile_edges(vals, f.bins)
                b.edges[f.name] = edges
                b.shares[f.name] = _normalise(bin_numeric(vals, edges)) if vals else []
            else:
                b.shares[f.name] = _normalise(bin_categorical(vals, f.categories)) if vals else []
        return b

    def fit_rows(self, rows: Iterable[Any], source: str = "rows") -> dict:
        """Build a baseline from TraceRecords or SQLite log rows (dicts); returns the snapshot."""
        with self._lock:
            self._baseline = self._fit([self._row(r) for r in rows], source)
            self._warmup = []
        return self.snapshot()

    def freeze_baseline(self, from_window: bool = True) -> None:
        with self._lock:
            rows = list(self._window) if from_window else list(self._warmup)
            if rows:
                self._baseline = self._fit(rows, "window" if from_window else "warmup")
                self._warmup = []

    def snapshot(self) -> dict:
        b = self._baseline
        if b is None:
            return {}
        return {"version": 1, "n": b.n, "created_at": b.created_at, "source": b.source, "edges": b.edges,
                "shares": b.shares, "counts": b.counts}

    def load_baseline(self, snap: dict) -> None:
        b = _Baseline(edges={k: list(v) for k, v in snap.get("edges", {}).items()},
                      shares={k: list(v) for k, v in snap.get("shares", {}).items()},
                      counts=dict(snap.get("counts", {})), n=int(snap.get("n", 0)),
                      created_at=float(snap.get("created_at", time.time())), source=snap.get("source", "file"))
        with self._lock:
            self._baseline = b
            self._warmup = []

    # ------------------------------------------------------------ report
    def report(self) -> dict:
        with self._lock:
            rows = list(self._window)
            b = self._baseline
            warm = len(self._warmup)
        out: dict[str, Any] = {
            "status": "warming_up" if b is None else "ok", "seen": self.n_seen, "window_requests": len(rows),
            "baseline_requests": b.n if b else warm, "baseline_source": b.source if b else None,
            "bands": {"stable": f"< {PSI_STABLE}", "investigate": f"{PSI_STABLE}-{PSI_ALERT}", "alert": f"> {PSI_ALERT}"},
            "features": {}, "computed_at": time.time(),
        }
        worst = "stable"
        order = ["insufficient_data", "stable", "investigate", "alert"]
        for i, f in enumerate(self.features):
            vals = [r[i] for r in rows if r[i] is not None]
            fr: dict[str, Any] = {"kind": f.kind, "monitoring_category": f.category, "n_current": len(vals),
                                  "psi": None, "band": "insufficient_data"}
            if b is not None and f.name in b.shares:
                exp = b.shares[f.name]
                fr["n_baseline"] = b.counts.get(f.name, 0)
                if f.kind == "numeric":
                    edges = b.edges.get(f.name, [])
                    act = _normalise(bin_numeric(vals, edges)) if vals else []
                    fr["edges"] = edges
                else:
                    act = _normalise(bin_categorical(vals, f.categories)) if vals else []
                    fr["bins"] = list(f.categories)
                fr["expected"], fr["actual"] = [round(x, 4) for x in exp], [round(x, 4) for x in act]
                if len(vals) >= self.min_samples and fr["n_baseline"] >= self.min_samples and exp and \
                        len(exp) == len(act):
                    fr["psi"] = round(psi(exp, act), 4)
                    fr["band"] = psi_band(fr["psi"])
            out["features"][f.name] = fr
            if b is not None and order.index(fr["band"]) > order.index(worst):
                worst = fr["band"]
        if b is not None:
            out["overall"] = worst
        return out


class DriftHook:
    """Engine hook: feeds the monitor and refreshes the Prometheus drift gauges every `recompute_every` requests."""

    name = "drift"

    def __init__(self, monitor: Optional[DriftMonitor] = None, metrics: Any = None, recompute_every: int = 50):
        self.monitor = monitor or DriftMonitor.from_env()
        self.metrics = metrics
        self.recompute_every = max(1, int(recompute_every))

    def __call__(self, rec: Any) -> None:
        try:
            self.monitor.observe(rec)
            if self.metrics is not None and self.monitor.n_seen % self.recompute_every == 0:
                self.metrics.set_drift(self.monitor.report())
        except Exception as e:  # never raise into serving
            log.debug("drift hook failed: %s", e)


# ---------------------------------------------------------------- offline helpers (dashboard, CLI)
def load_log_rows(db: Path, where: str = "", params: tuple = ()) -> list[dict]:
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(f"SELECT * FROM requests {where} ORDER BY ts", params)]


def compare_rows(baseline_rows: Sequence[Any], current_rows: Sequence[Any], min_samples: int = 30) -> dict:
    """PSI report for two batches of log rows (e.g. first vs last N requests, or arm A0 vs live traffic)."""
    mon = DriftMonitor(window=max(1, len(current_rows)), baseline_size=10 ** 9, min_samples=min_samples)
    mon.fit_rows(baseline_rows, source="offline")
    for r in current_rows:
        mon._window.append(mon._row(r))
    mon.n_seen = len(current_rows)
    return mon.report()


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build a drift baseline snapshot from a CostGuard SQLite log")
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--arm", default=None, help="only rows with this arm label (e.g. A0)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = load_log_rows(a.db, "WHERE arm = ?", (a.arm,)) if a.arm else load_log_rows(a.db)
    snap = DriftMonitor().fit_rows(rows, source=f"{a.db.name}{':' + a.arm if a.arm else ''}")
    text = json.dumps(snap, indent=2)
    if a.out:
        a.out.write_text(text)
        print(f"wrote baseline over {snap.get('n', 0)} rows -> {a.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
