"""CostGuard savings dashboard (Streamlit). Reads only local files, so it works offline and on Streamlit Cloud:

  * the SQLite request log (one TraceRecord per row): $COSTGUARD_DB, data/runtime/costguard.sqlite, or a replay log
    eval/results/ab_*.sqlite chosen in the sidebar;
  * eval/results/*.json written by the eval / cache / context / router workstreams, and configs/router_gate.json.

Every source is optional: a missing file shows the command that produces it.

    ./.venv/bin/python -m streamlit run dashboard/app.py          (or: make dashboard)

Importing this module does not import Streamlit; the UI lives in main(). The data helpers are plain pandas and are
unit-tested in tests/test_ops.py.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run dashboard/app.py` puts dashboard/ on sys.path, not the repo root
    sys.path.insert(0, str(ROOT))

RESULTS = Path(os.environ.get("COSTGUARD_RESULTS_DIR") or ROOT / "eval" / "results")
CONFIGS = Path(os.environ.get("COSTGUARD_CONFIG_DIR") or ROOT / "configs")
DEFAULT_DB = ROOT / "data" / "runtime" / "costguard.sqlite"

# Fixed categorical order (validated reference palette, light steps); colour follows the route, never its rank.
ROUTE_ORDER = ["exact", "semantic", "strong", "cheap", "bypass", "other"]
ROUTE_COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#e87ba4", "#9a9893"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#9a9893"
LEVERS = ["exact cache", "semantic cache", "context", "compression", "router"]

HOW_TO = {
    "log": "Start the proxy (`make serve`) and send traffic, run the load test (`make loadtest`), or run the A/B "
           "replay (`make ab`, writes eval/results/ab_*.sqlite).",
    "ab_summary.json": "Run the cumulative ablation A/B: `make ab` (writes eval/results/ab_summary.json).",
    "threshold_sweep.json": "Run the semantic-cache threshold sweep: `make sweep`.",
    "compression_eval.json": "Run the context/compression eval: `python -m eval.compression_eval`.",
    "router_gate.json": "Run the router eval gate: `make gate` (writes configs/router_gate.json).",
    "router_classifier.json": "`python -m costguard.router.classifier train`.",
    "loadtest.json": "Run the load test: `make loadtest` (or `bash loadtest/run.sh --quick`).",
}


# ====================================================================== data helpers (no Streamlit)
def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def list_dbs() -> list[Path]:
    seen, out = set(), []
    env = os.environ.get("COSTGUARD_DB")
    for p in [Path(env)] if env else []:
        out.append(p)
    out.append(DEFAULT_DB)
    out += sorted(RESULTS.glob("ab_*.sqlite")) + sorted(RESULTS.glob("*.sqlite"))
    out += sorted((ROOT / "data" / "runtime").glob("*.sqlite"))
    res = []
    for p in out:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            seen.add(k)
            res.append(p)
    return res


def list_ab_summaries() -> list[Path]:
    """eval/results/ab_summary.json first (the headline run), then any ab_summary_*.json variants, newest first."""
    main_ = RESULTS / "ab_summary.json"
    others = sorted((p for p in RESULTS.glob("ab_summary*.json") if p != main_), key=lambda p: p.stat().st_mtime,
                    reverse=True)
    return ([main_] if main_.exists() else []) + others


def _json_col(v: Any) -> dict:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v:
        try:
            out = json.loads(v)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def route_of(cache_status: Any, model_used: Any) -> str:
    if cache_status in ("exact", "semantic"):
        return cache_status
    if cache_status == "bypass" and not model_used:
        return "bypass"
    return model_used if model_used in ("strong", "cheap") else "other"


def load_log(db: Path, limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """The request log as a DataFrame (+ derived columns), or None if the file/table is missing or empty."""
    db = Path(db)
    if not db.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
            q = "SELECT * FROM requests ORDER BY ts" + (f" DESC LIMIT {int(limit)}" if limit else "")
            df = pd.read_sql_query(q, c)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.sort_values("ts").reset_index(drop=True)
    for col in ("stage_ms", "stage_errors"):
        df[col] = df[col].map(_json_col) if col in df else [{} for _ in range(len(df))]
    for col, default in (("cost_usd", 0.0), ("baseline_cost_usd", 0.0), ("saved_usd", 0.0), ("latency_ms", 0.0),
                         ("overhead_ms", 0.0), ("upstream_latency_ms", 0.0), ("input_tokens_original", 0),
                         ("input_tokens_sent", 0), ("output_tokens", 0)):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default) if col in df else default
    for col in ("cache_status", "model_used", "mode", "category", "arm", "error", "config_hash", "cache_similarity",
                "compression_ratio", "request_id", "query", "response_text", "route_reason"):
        if col not in df:
            df[col] = None
    df["route"] = [route_of(c, m) for c, m in zip(df["cache_status"], df["model_used"])]
    df["hit"] = df["cache_status"].isin(["exact", "semantic"])
    df["time"] = pd.to_datetime(df["ts"], unit="s")
    df["n"] = np.arange(len(df))
    df["has_error"] = df["error"].notna() & (df["error"].astype(str) != "") | df["stage_errors"].map(bool)
    return df


def pct(xs: Iterable[float], q: float) -> Optional[float]:
    arr = np.asarray([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    return float(np.percentile(arr, q)) if arr.size else None


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    cost, base = float(df["cost_usd"].sum()), float(df["baseline_cost_usd"].sum())
    hits = df[df["hit"]]
    miss = df[~df["hit"]]
    return {
        "requests": n, "cost_usd": cost, "baseline_cost_usd": base,
        "saved_usd": base - cost, "saved_pct": (1 - cost / base) * 100 if base else 0.0,
        "cost_per_request": cost / n if n else 0.0, "baseline_per_request": base / n if n else 0.0,
        "hit_rate": len(hits) / n if n else 0.0,
        "exact_rate": float((df["cache_status"] == "exact").mean()) if n else 0.0,
        "semantic_rate": float((df["cache_status"] == "semantic").mean()) if n else 0.0,
        "cheap_share": float((df["route"] == "cheap").mean()) if n else 0.0,
        "route_mix": df["route"].value_counts().reindex(ROUTE_ORDER).dropna().astype(int).to_dict(),
        "latency_p50": pct(df["latency_ms"], 50), "latency_p95": pct(df["latency_ms"], 95),
        "latency_p99": pct(df["latency_ms"], 99),
        "hit_latency_p50": pct(hits["latency_ms"], 50), "miss_latency_p50": pct(miss["latency_ms"], 50),
        "overhead_p50": pct(df["overhead_ms"], 50), "overhead_p99": pct(df["overhead_ms"], 99),
        "hit_overhead_p99": pct(hits["overhead_ms"], 99), "miss_overhead_p99": pct(miss["overhead_ms"], 99),
        "error_rate": float(df["error"].notna().mean()) if n else 0.0,
        "stage_error_rate": float(df["stage_errors"].map(bool).mean()) if n else 0.0,
        "config_hashes": sorted(x for x in df["config_hash"].dropna().unique()),
    }


def by_arm(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df["arm"].dropna().nunique() < 1:
        return None
    rows = []
    for arm, g in df.groupby(df["arm"].fillna("(none)")):
        s = summarize(g)
        rows.append({"arm": arm, "requests": s["requests"], "cost_usd": s["cost_usd"], "saved_pct": s["saved_pct"],
                     "hit_rate": s["hit_rate"], "cheap_share": s["cheap_share"], "p50_ms": s["latency_p50"],
                     "p99_ms": s["latency_p99"], "overhead_p99_ms": s["overhead_p99"]})
    return pd.DataFrame(rows)


def stage_frame(df: pd.DataFrame) -> pd.DataFrame:
    st = pd.DataFrame(list(df["stage_ms"]), index=df.index).add_prefix("ms_")
    return st.fillna(0.0)


# ---------------------------------------------------------------- tolerant JSON -> table helpers
def find_records(obj: Any, want: tuple[str, ...] = ()) -> Optional[list[dict]]:
    """First list of dicts in `obj` (breadth-first), preferring one whose rows contain any key in `want`."""
    queue, fallback = [obj], None
    while queue:
        o = queue.pop(0)
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            if not want or any(k in o[0] for k in want):
                return o
            fallback = fallback or o
        if isinstance(o, dict):
            queue.extend(o.values())
        elif isinstance(o, list):
            queue.extend(x for x in o if isinstance(x, (dict, list)))
    return fallback


def to_frame(obj: Any, index_name: str = "key") -> Optional[pd.DataFrame]:
    """List of dicts -> rows; dict of dicts -> one row per key; dict of scalars -> key/value table."""
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        return pd.DataFrame(obj)
    if isinstance(obj, dict) and obj:
        if all(isinstance(v, dict) for v in obj.values()):
            df = pd.DataFrame.from_dict(obj, orient="index")
            df.index.name = index_name
            return df.reset_index()
        scalars = {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}
        if scalars:
            return pd.DataFrame({"key": list(scalars), "value": [str(v) for v in scalars.values()]})
    return None


def pick(df: pd.DataFrame, *aliases: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in cols:
            return cols[a.lower()]
    return None


def fmt_ci(v: Any) -> Any:
    if isinstance(v, (list, tuple)) and len(v) in (2, 3) and all(isinstance(x, (int, float)) for x in v):
        lo, hi = (v[1], v[2]) if len(v) == 3 else (v[0], v[1])
        return f"[{lo:.4g}, {hi:.4g}]"
    return v


def displayable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (list, tuple, dict))).any():
            out[c] = out[c].map(lambda v: fmt_ci(v) if isinstance(v, (list, tuple)) else
                                (json.dumps(v)[:120] if isinstance(v, dict) else v))
    return out


def ab_frame(ab: Any) -> Optional[pd.DataFrame]:
    """Arms table from ab_summary.json (list of arm dicts, or {"arms": {name: {...}}})."""
    if ab is None:
        return None
    arms = ab.get("arms") if isinstance(ab, dict) and "arms" in ab else ab
    if isinstance(arms, dict):
        rows = [dict({"arm": k}, **v) for k, v in arms.items() if isinstance(v, dict)]
    else:
        rows = find_records(arms, ("arm", "name")) or []
    if not rows:
        return None
    df = pd.json_normalize(rows, max_level=1)        # e.g. quality.retained, tokens.input_sent
    name = pick(df, "arm", "name", "id", "label")
    if name and name != "arm":
        df = df.rename(columns={name: "arm"})
    return df


AB_KEY_COLS = ["arm", "description", "n", "cost_usd", "savings_pct", "savings_ci", "hit_rate.exact",
               "hit_rate.semantic", "false_hit_rate", "false_hit_ci", "trap_false_hits", "quality.retained",
               "quality.retained_ci", "quality.cost_per_correct_usd", "latency_ms.overhead", "latency_ms.e2e"]


def _p50_p99(v: Any) -> Any:
    if isinstance(v, dict) and ("p50" in v or "p99" in v):
        return f"{v.get('p50', 'n/a')} / {v.get('p99', 'n/a')}"
    return v


def ab_key_table(abf: pd.DataFrame) -> pd.DataFrame:
    """The columns a reviewer needs (savings + CI, hits, false hits, quality retained + CI, latency)."""
    cols = [c for c in AB_KEY_COLS if c in abf.columns] or list(abf.columns)
    out = abf[cols].copy()
    for c in cols:
        out[c] = out[c].map(_p50_p99)
    return out


def ci_frame(abf: pd.DataFrame, value: str, ci: str) -> Optional[pd.DataFrame]:
    """arm | value | lo | hi, for point + error-bar charts (CI given as [lo, hi] or [mean, lo, hi])."""
    if value not in abf.columns:
        return None
    rows = []
    for _, r in abf.iterrows():
        v, c = r.get(value), r.get(ci) if ci in abf.columns else None
        if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
            continue
        lo = hi = None
        if isinstance(c, (list, tuple)) and len(c) in (2, 3):
            lo, hi = (c[1], c[2]) if len(c) == 3 else (c[0], c[1])
        rows.append({"arm": str(r["arm"]), "value": float(v), "lo": lo if lo is not None else float(v),
                     "hi": hi if hi is not None else float(v)})
    return pd.DataFrame(rows) if rows else None


def waterfall_frame(ab: Any) -> Optional[pd.DataFrame]:
    """Savings waterfall: start (baseline) -> one step per lever -> end (CostGuard). Uses an explicit
    `waterfall`/`levers` list when present, else consecutive differences of cumulative arms (A0 -> A5)."""
    if ab is None:
        return None
    steps: list[tuple[str, float]] = []
    explicit = ab.get("waterfall") or ab.get("levers") if isinstance(ab, dict) else None
    df = ab_frame(ab)
    cost_col = pick(df, "cost_usd", "total_cost_usd", "cost", "total_cost", "mean_cost_usd") if df is not None else None
    if explicit:
        ex = to_frame(explicit, "lever")
        if ex is None:
            return None
        lev = pick(ex, "lever", "name", "stage", "key", "arm")
        val = pick(ex, "saved_usd", "delta_usd", "savings_usd", "saved", "delta", "value")
        if not (lev and val):
            return None
        steps = [(str(a), float(b)) for a, b in zip(ex[lev], ex[val])]
        if ab.get("baseline_cost_usd"):
            start = float(ab["baseline_cost_usd"])
        elif df is not None and cost_col:
            a0 = df[df["arm"].astype(str) == "A0"]
            start = float((a0 if len(a0) else df)[cost_col].iloc[0])
        else:
            start = sum(abs(v) for _, v in steps)
    elif df is not None and cost_col:
        d = df.sort_values("arm") if df["arm"].astype(str).str.match(r"^A\d").all() else df
        costs = d[cost_col].astype(float).tolist()
        labels = d.get("lever", d.get("label", d["arm"])).astype(str).tolist()
        if len(costs) < 2:
            return None
        start = costs[0]
        names = [labels[i] if labels[i] != str(d["arm"].iloc[i]) else
                 (LEVERS[i - 1] if 0 < i <= len(LEVERS) else labels[i]) for i in range(len(costs))]
        steps = [(names[i], costs[i - 1] - costs[i]) for i in range(1, len(costs))]
    else:
        return None
    rows, level = [{"step": "baseline", "start": 0.0, "end": start, "kind": "total", "delta": start}], start
    for name, saved in steps:
        rows.append({"step": name, "start": level, "end": level - saved, "kind": "lever", "delta": -saved})
        level -= saved
    rows.append({"step": "CostGuard", "start": 0.0, "end": level, "kind": "total", "delta": level})
    out = pd.DataFrame(rows)
    out["pct_of_baseline"] = out["delta"] / start * 100 if start else 0.0
    out["order"] = range(len(out))
    return out


def _columnar_curves(obj: Any, path: tuple = ()) -> Iterable[tuple[tuple, dict]]:
    """Yield (path, dict) for every dict holding parallel lists `tau` and `hit_rate` (eval/sweep_threshold.py)."""
    if isinstance(obj, dict):
        tau, hr = obj.get("tau"), obj.get("hit_rate")
        if isinstance(tau, list) and isinstance(hr, list) and len(tau) == len(hr) and tau:
            yield path, obj
        for k, v in obj.items():
            yield from _columnar_curves(v, path + (str(k),))


def sweep_frame(sweep: Any) -> Optional[pd.DataFrame]:
    """Long table tau | hit_rate | false_hit_rate | series. Reads the columnar curves of eval/sweep_threshold.py
    (models -> simulation -> stream -> guards_on/off), or a plain list of {tau, hit_rate, ...} rows."""
    if sweep is None:
        return None
    frames = []
    primary = sweep.get("embedding_model") if isinstance(sweep, dict) else None
    for path, d in _columnar_curves(sweep):
        if primary and "models" in path and primary not in path:
            continue                                   # comparison embedding model: shown via the selector only
        n = len(d["tau"])
        f = pd.DataFrame({"tau": d["tau"], "hit_rate": d["hit_rate"]})
        if isinstance(d.get("false_hit_rate"), list) and len(d["false_hit_rate"]) == n:
            f["false_hit_rate"] = d["false_hit_rate"]
        if isinstance(d.get("precision_per_hit"), list) and len(d["precision_per_hit"]) == n:
            f["precision_per_hit"] = d["precision_per_hit"]
        keep = [x for x in path if x not in ("models", "simulation", primary or "")]
        f["series"] = " / ".join(keep).replace("guards_on", "guards on").replace("guards_off", "guards off") or "curve"
        frames.append(f)
    if not frames:
        rows = find_records(sweep, ("tau", "threshold"))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        tau = pick(df, "tau", "threshold")
        hr = pick(df, "hit_rate", "hits_rate", "hit")
        fhr = pick(df, "false_hit_rate", "false_hit_rate_per_request", "fhr", "false_hits_rate")
        if not (tau and hr):
            return None
        f = pd.DataFrame({"tau": df[tau], "hit_rate": df[hr]})
        if fhr:
            f["false_hit_rate"] = df[fhr]
        f["series"] = "curve"
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    for c in ("tau", "hit_rate", "false_hit_rate", "precision_per_hit"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["series", "tau"]).reset_index(drop=True)


def recommended_taus(sweep: Any) -> dict[str, float]:
    """{mode: tau} from `recommended` ({mode: {tau, budget_false_hit_rate}}) or a single recommended value."""
    if not isinstance(sweep, dict):
        return {}
    out: dict[str, float] = {}
    for k in ("recommended", "recommended_tau", "tau_recommended", "recommendation", "chosen_tau"):
        v = sweep.get(k)
        if isinstance(v, (int, float)):
            out["recommended"] = float(v)
        elif isinstance(v, dict):
            if isinstance(v.get("tau"), (int, float)):
                out["recommended"] = float(v["tau"])
            for mode, r in v.items():
                if isinstance(r, dict) and isinstance(r.get("tau"), (int, float)):
                    out[str(mode)] = float(r["tau"])
        if out:
            break
    return out


def recommended_tau(sweep: Any, mode: str = "balanced") -> Optional[float]:
    r = recommended_taus(sweep)
    return r.get(mode, r.get("recommended"))


def compression_frames(comp: Any) -> dict[str, pd.DataFrame]:
    """Tables from eval/compression_eval.py: methods (per configuration), grid (budget x rate) and recommendation."""
    out: dict[str, pd.DataFrame] = {}
    if not isinstance(comp, dict):
        f = to_frame(comp)
        return {"results": f} if f is not None else {}
    for key in ("methods", "results", "grid", "recommendation", "gap_sweep"):
        v = comp.get(key)
        f = to_frame(v, "method") if v is not None else None
        if f is not None:
            out[key] = f
    if not out:
        f = to_frame(find_records(comp) or comp)
        if f is not None:
            out["results"] = f
    return out


def policy_tau(mode: str = "balanced") -> Optional[float]:
    try:
        import yaml
        return float(yaml.safe_load((CONFIGS / "policy.yaml").read_text())["modes"][mode]["tau"])
    except Exception:
        return None


def timeseries(df: pd.DataFrame, max_points: int = 60) -> tuple[pd.DataFrame, str]:
    """Bucket the log over wall time when it spans >10 min, else over request order (replays are bursty)."""
    d = df.sort_values("ts")
    span = float(d["ts"].max() - d["ts"].min()) if len(d) else 0.0
    if span >= 600:
        for freq, sec in (("1min", 60), ("5min", 300), ("15min", 900), ("1h", 3600), ("6h", 21600), ("1D", 86400)):
            if span / sec <= max_points:
                break
        key, xlabel = d["time"].dt.floor(freq), f"time ({freq} buckets)"
    else:
        size = max(1, math.ceil(len(d) / max_points))
        key, xlabel = (d["n"] // size) * size, f"request # ({size}/bucket)"
    g = d.groupby(key)
    out = pd.DataFrame({
        "requests": g.size(), "cost_per_request": g["cost_usd"].mean(),
        "baseline_per_request": g["baseline_cost_usd"].mean(), "hit_rate": g["hit"].mean(),
        "p50_ms": g["latency_ms"].median(), "p99_ms": g["latency_ms"].quantile(0.99),
        "overhead_p99_ms": g["overhead_ms"].quantile(0.99), "errors": g["has_error"].sum(),
    })
    out.index.name = "x"
    return out.reset_index(), xlabel


def drift_offline(df: pd.DataFrame, window: int = 500) -> Optional[dict]:
    """PSI of the latest window vs the earliest window of this log (same maths as the live /v1/drift)."""
    if len(df) < 60:
        return None
    from costguard.obs.drift import compare_rows
    k = min(window, len(df) // 2)
    recs = df.to_dict("records")
    rep = compare_rows(recs[:k], recs[-k:], min_samples=min(30, k))
    rep["baseline_desc"], rep["current_desc"] = f"first {k} requests", f"last {k} requests"
    return rep


def drift_table(rep: dict) -> pd.DataFrame:
    rows = []
    for name, f in (rep.get("features") or {}).items():
        rows.append({"feature": name, "category": f.get("monitoring_category"), "psi": f.get("psi"),
                     "band": f.get("band"), "n_baseline": f.get("n_baseline"), "n_current": f.get("n_current")})
    return pd.DataFrame(rows)


def alert_checks(s: dict, df: pd.DataFrame, drift: Optional[dict]) -> pd.DataFrame:
    """The RUNBOOK alert rules evaluated on this log (design-time thresholds; see docs/RUNBOOK.md)."""
    med = float(df.loc[df["cost_usd"] > 0, "cost_usd"].median()) if (df["cost_usd"] > 0).any() else 0.0
    spikes = int((df["cost_usd"] > 5 * med).sum()) if med else 0
    rows = [
        ("Operational", "error rate", f"{s['error_rate'] * 100:.2f}%", "> 0.5%", s["error_rate"] > 0.005),
        ("Operational", "hit-path overhead p99", _ms(s["hit_overhead_p99"]), "> 50 ms",
         (s["hit_overhead_p99"] or 0) > 50),
        ("Operational", "requests > 5x median cost", str(spikes), "any", spikes > 0),
        ("Operational", "stage fail-open rate", f"{s['stage_error_rate'] * 100:.2f}%", "> 1%",
         s["stage_error_rate"] > 0.01),
        ("Output/Quality", "savings vs baseline", f"{s['saved_pct']:.1f}%", "< 30% target", s["saved_pct"] < 30),
    ]
    if drift and drift.get("features"):
        for name, f in drift["features"].items():
            rows.append(("Drift" if f.get("monitoring_category") == "drift" else f.get("monitoring_category", "").title(),
                         f"PSI {name}", "n/a" if f.get("psi") is None else f"{f['psi']:.3f}", "> 0.25 alert",
                         f.get("band") == "alert"))
    return pd.DataFrame(rows, columns=["category", "check", "value", "threshold", "firing"])


def num_label(v: Any) -> str:
    """Category label for a numeric-ish value: None/NaN -> "none", 1200.0 -> "1200"."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "none"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def num_sort_key(v: Any) -> tuple:
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, 0.0)


def _ms(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1f} ms"


def _usd(v: Optional[float], digits: int = 4) -> str:
    return "n/a" if v is None else f"${v:,.{digits}f}"


# ====================================================================== UI
def main() -> None:  # pragma: no cover - exercised by `streamlit run`, not by pytest
    import altair as alt
    import streamlit as st

    st.set_page_config(page_title="CostGuard savings", page_icon=":material/savings:", layout="wide")
    route_scale = alt.Scale(domain=ROUTE_ORDER, range=ROUTE_COLORS)

    def chart(c: Any) -> None:
        st.altair_chart(c, width="stretch")

    def table(d: pd.DataFrame, **kw: Any) -> None:
        st.dataframe(displayable(d), width="stretch", hide_index=True, **kw)

    def missing(key: str, path: Optional[Path] = None) -> None:
        shown = path.relative_to(ROOT) if path and path.is_relative_to(ROOT) else path
        where = f"`{shown}` not found. " if path else ""
        st.info(f"{where}{HOW_TO[key]}", icon=":material/info:")

    def raw(obj: Any, label: str = "Raw JSON") -> None:
        with st.expander(label):
            st.json(obj, expanded=False)

    # ---------------------------------------------------------------- sidebar
    st.sidebar.title("CostGuard")
    dbs = list_dbs()
    labels = [f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}{'' if p.exists() else '  (missing)'}"
              for p in dbs]
    first_existing = next((i for i, p in enumerate(dbs) if p.exists()), 0)
    choice = st.sidebar.selectbox("Request log", range(len(dbs)), index=first_existing, format_func=lambda i: labels[i])
    db = dbs[choice]
    df_all = st.cache_data(ttl=10, show_spinner=False)(load_log)(db)
    df = df_all
    if df_all is not None:
        arms = sorted(df_all["arm"].dropna().unique())
        if arms:
            default_arm = ["A5"] if "A5" in arms else arms[-1:]
            sel = st.sidebar.multiselect("Arms", arms, default=default_arm if len(arms) > 1 else arms,
                                         help="A/B replay logs carry an arm label per row")
            df = df_all[df_all["arm"].isin(sel)] if sel else df_all
        modes = sorted(df["mode"].dropna().unique())
        if len(modes) > 1:
            msel = st.sidebar.multiselect("Modes", modes, default=modes)
            df = df[df["mode"].isin(msel)] if msel else df
        if df.empty:
            df = None
    proxy_url = st.sidebar.text_input("Live proxy URL (optional)", os.environ.get("COSTGUARD_URL", ""),
                                      help="e.g. http://localhost:8000 - used for the live /v1/drift report")
    if st.sidebar.button("Reload data", icon=":material/refresh:"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption("Costs are list-price-equivalent USD from configs/prices.yaml. Savings = 1 - cost / baseline "
                       "(baseline = strong tier, full prompt, no cache).")

    ab_files = list_ab_summaries()
    ab_path = ab_files[0] if ab_files else None
    if len(ab_files) > 1:
        ab_path = st.sidebar.selectbox("A/B summary", ab_files, format_func=lambda p: p.name)
    ab = load_json(ab_path) if ab_path else None
    sweep = load_json(RESULTS / "threshold_sweep.json")
    comp = load_json(RESULTS / "compression_eval.json")
    gate = load_json(CONFIGS / "router_gate.json")
    clf = load_json(RESULTS / "router_classifier.json")
    lt = load_json(RESULTS / "loadtest.json")

    st.title("LLM CostGuard - savings at equal quality")
    tabs = st.tabs(["Overview", "Savings waterfall", "A/B", "Cache", "Context & compression", "Router",
                    "Monitoring", "Request explorer"])

    # ---------------------------------------------------------------- Overview
    with tabs[0]:
        if df is None:
            missing("log", db)
        else:
            s = summarize(df)
            st.caption(f"{s['requests']:,} requests from `{db.name}` | config hash(es): "
                       f"{', '.join(s['config_hashes']) or 'n/a'}")
            c = st.columns(4)
            c[0].metric("Cost (CostGuard)", _usd(s["cost_usd"]), f"{-s['saved_pct']:.1f}% vs baseline",
                        delta_color="inverse", border=True)
            c[1].metric("Baseline cost", _usd(s["baseline_cost_usd"]), border=True,
                        help="Same requests on the strong tier, full prompt, no cache")
            c[2].metric("Saved", f"{s['saved_pct']:.1f}%", _usd(s["saved_usd"]), border=True)
            c[3].metric("Cost / 1k requests", _usd(s["cost_per_request"] * 1000, 3),
                        f"baseline {_usd(s['baseline_per_request'] * 1000, 3)}", delta_color="off", border=True)
            c = st.columns(4)
            c[0].metric("Cache hit rate", f"{s['hit_rate'] * 100:.1f}%",
                        f"exact {s['exact_rate'] * 100:.1f}% / semantic {s['semantic_rate'] * 100:.1f}%",
                        delta_color="off", border=True)
            c[1].metric("Latency p50 / p99", f"{_ms(s['latency_p50'])} / {_ms(s['latency_p99'])}", border=True,
                        help="End-to-end inside CostGuard, from the log")
            c[2].metric("Overhead p50 / p99", f"{_ms(s['overhead_p50'])} / {_ms(s['overhead_p99'])}", border=True,
                        help="Latency CostGuard adds (total - upstream)")
            c[3].metric("Error rate", f"{s['error_rate'] * 100:.2f}%",
                        f"stage fail-open {s['stage_error_rate'] * 100:.1f}%", delta_color="off", border=True)

            left, right = st.columns([1, 2])
            with left:
                st.subheader("Route mix")
                mix = pd.DataFrame({"route": list(s["route_mix"]), "requests": list(s["route_mix"].values())})
                mix["share"] = mix["requests"] / max(1, s["requests"])
                chart(alt.Chart(mix).mark_bar(cornerRadiusEnd=4, height=18).encode(
                    y=alt.Y("route:N", sort=ROUTE_ORDER, title=None),
                    x=alt.X("share:Q", axis=alt.Axis(format="%"), title="share of requests"),
                    color=alt.Color("route:N", scale=route_scale, legend=None),
                    tooltip=["route", "requests", alt.Tooltip("share:Q", format=".1%")]).properties(height=200))
            with right:
                st.subheader("Cumulative spend vs baseline")
                cum = pd.DataFrame({"request": df["n"].values - df["n"].values.min(),
                                    "CostGuard": df["cost_usd"].cumsum().values,
                                    "baseline": df["baseline_cost_usd"].cumsum().values})
                long = cum.melt("request", var_name="series", value_name="usd")
                chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
                    x=alt.X("request:Q", title="request #"), y=alt.Y("usd:Q", title="cumulative USD"),
                    color=alt.Color("series:N", scale=alt.Scale(domain=["baseline", "CostGuard"],
                                                               range=[NEUTRAL, SERIES[0]]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["request", "series", alt.Tooltip("usd:Q", format="$.5f")]).properties(height=220))
            arms_tbl = by_arm(df_all)
            if arms_tbl is not None and len(arms_tbl) > 1:
                st.subheader("Per arm (from the log)")
                table(arms_tbl.round(4))
            st.caption("Hit-path vs miss-path p50: "
                       f"{_ms(s['hit_latency_p50'])} vs {_ms(s['miss_latency_p50'])}. A blended p50 hides that hits "
                       "never touch the provider.")

    # ---------------------------------------------------------------- Waterfall
    with tabs[1]:
        st.subheader("Savings waterfall by lever")
        wf = waterfall_frame(ab)
        if ab is None:
            missing("ab_summary.json", RESULTS / "ab_summary.json")
        elif wf is None:
            st.warning("ab_summary.json has no arm costs or `waterfall` list this view understands.")
            raw(ab)
        else:
            st.caption(f"Source: `{ab_path.name}`. Cumulative ablation on the frozen trace: each step is the cost "
                       "removed by adding one lever "
                       "(exact cache -> semantic -> context -> compression -> router). Bars in list-price USD.")
            wf["lo"], wf["hi"] = wf[["start", "end"]].min(axis=1), wf[["start", "end"]].max(axis=1)
            chart(alt.Chart(wf).mark_bar(cornerRadius=4, size=36).encode(
                x=alt.X("step:N", sort=list(wf["step"]), title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("lo:Q", title="USD"), y2="hi:Q",
                color=alt.Color("kind:N", scale=alt.Scale(domain=["total", "lever"], range=[NEUTRAL, SERIES[0]]),
                                legend=None),
                tooltip=["step", alt.Tooltip("delta:Q", format="$.5f", title="change"),
                         alt.Tooltip("pct_of_baseline:Q", format=".1f", title="% of baseline")])
                  .properties(height=320))
            table(wf[["step", "delta", "pct_of_baseline"]].rename(
                columns={"delta": "USD (total or change)", "pct_of_baseline": "% of baseline"}).round(6))
            st.caption("Expect compression to be large in tokens but small in dollars: output tokens cost 5x input "
                       "and only cache hits and downshifts remove output tokens.")
            raw(ab)

    # ---------------------------------------------------------------- A/B
    with tabs[2]:
        st.subheader("A/B arms with confidence intervals")
        abf = ab_frame(ab)
        if ab is None:
            missing("ab_summary.json", RESULTS / "ab_summary.json")
        elif abf is None:
            raw(ab)
        else:
            st.caption(f"Source: `{ab_path.name}`. Savings = 1 - arm cost / A0 cost on the same items (paired). "
                       "Quality retained = arm judge "
                       "score / A0 score (paired bootstrap 95% CI); target >= 95%. False-hit rate is per request.")
            table(ab_key_table(abf))
            a, b = st.columns(2)
            for col, (val, ci, title, ref) in zip((a, b), (("savings_pct", "savings_ci", "Savings vs A0 (%)", None),
                                                           ("quality.retained", "quality.retained_ci",
                                                            "Quality retained (%)", 95.0))):
                cf = ci_frame(abf, val, ci)
                if cf is None:
                    continue
                base = alt.Chart(cf).encode(x=alt.X("arm:N", title=None, sort=list(cf["arm"]),
                                                    axis=alt.Axis(labelAngle=0)))
                layers = [base.mark_rule(strokeWidth=2, color=SERIES[0]).encode(y=alt.Y("lo:Q", title=title,
                                                                                    scale=alt.Scale(zero=False)),
                                                                                   y2="hi:Q"),
                          base.mark_point(size=80, filled=True, color=SERIES[0]).encode(
                              y="value:Q", tooltip=["arm", alt.Tooltip("value:Q", format=".1f"),
                                                    alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")])]
                if ref is not None:
                    layers.append(alt.Chart(pd.DataFrame({"y": [ref]})).mark_rule(strokeDash=[4, 4], color=NEUTRAL)
                                  .encode(y="y:Q"))
                with col:
                    chart(alt.layer(*layers).properties(height=240, title=title))
            with st.expander("All arm columns"):
                table(abf)
            meta = {k: v for k, v in ab.items() if k != "arms" and not isinstance(v, (list, dict))} \
                if isinstance(ab, dict) else {}
            if meta:
                st.json(meta, expanded=False)
        if df is not None and by_arm(df_all) is not None:
            st.subheader("Arms in the selected log")
            table(by_arm(df_all).round(4))

    # ---------------------------------------------------------------- Cache
    with tabs[3]:
        st.subheader("Semantic-cache threshold curve")
        sw = sweep_frame(sweep)
        if sweep is None:
            missing("threshold_sweep.json", RESULTS / "threshold_sweep.json")
        elif sw is None:
            raw(sweep)
        else:
            series = list(dict.fromkeys(sw["series"]))
            default = next((i for i, x in enumerate(series) if "guards on" in x), 0)
            pick_series = st.selectbox("Curve", series, index=default,
                                       help="replay stream rendering / guards on-off (eval/sweep_threshold.py)") \
                if len(series) > 1 else series[0]
            cur = sw[sw["series"] == pick_series]
            recs = recommended_taus(sweep)
            cur_tau = policy_tau("balanced")
            ys = [c for c in ("hit_rate", "false_hit_rate") if c in cur]
            long = cur.melt("tau", value_vars=ys, var_name="metric", value_name="rate")
            base = alt.Chart(long).encode(
                x=alt.X("tau:Q", title="similarity threshold (tau)", scale=alt.Scale(zero=False)),
                y=alt.Y("rate:Q", title="share of ALL requests", axis=alt.Axis(format="%")),
                color=alt.Color("metric:N", scale=alt.Scale(domain=["hit_rate", "false_hit_rate"],
                                                           range=[SERIES[0], SERIES[1]]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["tau", "metric", alt.Tooltip("rate:Q", format=".2%")])
            layers = [base.mark_line(strokeWidth=2), base.mark_point(size=50, filled=True)]
            rules = [{"tau": t, "label": f"recommended: {m}"} for m, t in recs.items()]
            if cur_tau is not None:
                rules.append({"tau": cur_tau, "label": "policy.yaml (balanced)"})
            if rules:
                rdf = pd.DataFrame(rules)
                rdf["row"] = range(len(rdf))
                r = alt.Chart(rdf).encode(x="tau:Q")
                layers += [r.mark_rule(strokeDash=[4, 4], color=NEUTRAL),
                           r.mark_text(align="left", dx=4, color=NEUTRAL).encode(
                               y=alt.Y("row:Q", axis=None, scale=alt.Scale(domain=[-1, len(rdf) * 3], reverse=True)),
                               text="label:N")]
            chart(alt.layer(*layers).resolve_scale(y="independent").properties(height=320))
            st.caption("False-hit rate is wrong hits / ALL requests (not per hit), so an aggressive tau cannot hide "
                       "behind high per-hit precision. Recommended tau per mode = highest hit rate whose false-hit "
                       "rate stays inside that mode's budget.")
            if recs and isinstance(sweep, dict) and isinstance(sweep.get("recommended"), dict):
                table(to_frame(sweep["recommended"], "mode"))
            with st.expander("Curve data"):
                table(cur)
        if df is not None:
            st.subheader("Similarity of the nearest cached query (from the log)")
            sim = df[df["cache_similarity"].notna()].copy()
            if sim.empty:
                st.info("No semantic lookups in this log yet (semantic cache off, or no cacheable requests).")
            else:
                guarded = sim["cache_guard"].notna() if "cache_guard" in sim else pd.Series(False, index=sim.index)
                sim["outcome"] = np.where(sim["cache_status"] == "semantic", "served (hit)",
                                          np.where(guarded, "refused by guard", "below tau"))
                chart(alt.Chart(sim).mark_bar(cornerRadiusEnd=2).encode(
                    x=alt.X("cache_similarity:Q", bin=alt.Bin(step=0.02), title="cosine similarity"),
                    y=alt.Y("count():Q", title="requests", stack=True),
                    color=alt.Color("outcome:N", scale=alt.Scale(domain=["served (hit)", "below tau",
                                                                        "refused by guard"],
                                                                range=[SERIES[2], NEUTRAL, SERIES[1]]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["outcome", "count()"]).properties(height=240))
                worst = sim[sim["cache_status"] == "semantic"].nsmallest(10, "cache_similarity")
                if not worst.empty:
                    st.markdown("**Least similar accepted hits** (where false hits live; review these by hand)")
                    table(worst[["cache_similarity", "query", "cache_neighbor"] if "cache_neighbor" in worst else
                                ["cache_similarity", "query"]])

    # ---------------------------------------------------------------- Context & compression
    with tabs[4]:
        st.subheader("Context trimming and compression")
        if comp is None:
            missing("compression_eval.json", RESULTS / "compression_eval.json")
        else:
            frames = compression_frames(comp)
            if not frames:
                raw(comp)
            else:
                if isinstance(comp, dict) and isinstance(comp.get("setup"), dict):
                    st.caption(comp["setup"].get("metric", ""))
                for name in ("recommendation", "methods", "results"):
                    if name in frames:
                        st.markdown(f"**{name}**")
                        table(frames[name])
                g = frames.get("grid")
                if g is not None and {"rate", "evidence_retention"} <= set(g.columns):
                    st.markdown("**Budget x rate grid**: evidence retained vs tokens kept")
                    g = g.copy()
                    g["budget"] = g["budget"].map(num_label) if "budget" in g else "all"
                    xcol = "kept_share" if "kept_share" in g else "rate"
                    if "mode_min" in g and g["mode_min"].nunique() > 1:
                        mm = st.selectbox("compression_min_tokens", sorted(g["mode_min"].unique()))
                        g = g[g["mode_min"] == mm]
                    budgets = sorted(g["budget"].unique(), key=num_sort_key)
                    chart(alt.Chart(g).mark_line(point=alt.OverlayMarkDef(size=50, filled=True), strokeWidth=2)
                          .encode(x=alt.X(f"{xcol}:Q", title="share of context tokens kept", axis=alt.Axis(format="%")),
                                  y=alt.Y("evidence_retention:Q", title="evidence retained", axis=alt.Axis(format="%")),
                                  color=alt.Color("budget:N", sort=budgets, title="token budget",
                                                  scale=alt.Scale(range=SERIES + ["#e87ba4", "#008300", "#4a3aa7"])),
                                  tooltip=[c for c in g.columns if c != "retention_ci95"][:8])
                          .properties(height=300))
                    with st.expander("Grid data"):
                        table(g)
                raw(comp)
        if df is not None:
            ctx = df[df["input_tokens_original"] > 0]
            withc = df[pd.to_numeric(df.get("context_docs_in", 0), errors="coerce").fillna(0) > 0] \
                if "context_docs_in" in df else df.iloc[0:0]
            c = st.columns(3)
            c[0].metric("Input tokens sent / original", f"{ctx['input_tokens_sent'].sum() / max(1, ctx['input_tokens_original'].sum()) * 100:.1f}%",
                        help="Includes cache hits (0 sent)", border=True)
            if len(withc):
                c[1].metric("Context docs kept / in", f"{withc['context_docs_kept'].sum() / max(1, withc['context_docs_in'].sum()) * 100:.1f}%",
                            f"{len(withc)} requests with context", delta_color="off", border=True)
            cr = pd.to_numeric(df["compression_ratio"], errors="coerce").dropna()
            if len(cr):
                c[2].metric("Compression ratio (median)", f"{cr.median():.2f}x", f"{len(cr)} compressed",
                            delta_color="off", border=True)
                chart(alt.Chart(pd.DataFrame({"ratio": cr})).mark_bar(cornerRadiusEnd=2, color=SERIES[0]).encode(
                    x=alt.X("ratio:Q", bin=alt.Bin(maxbins=30), title="tokens before / after"),
                    y=alt.Y("count():Q", title="requests"), tooltip=["count()"]).properties(height=220))

    # ---------------------------------------------------------------- Router
    with tabs[5]:
        st.subheader("Downshift eval gate (per category)")
        if gate is None:
            missing("router_gate.json", CONFIGS / "router_gate.json")
        else:
            gf = to_frame(gate.get("categories", gate) if isinstance(gate, dict) else gate, "category")
            if gf is None:
                raw(gate)
            else:
                st.caption("A category is downshifted to the cheap tier only if the lower bound of its paired 95% CI "
                           "on quality clears the margin.")
                table(gf)
                raw(gate)
        if clf is not None:
            st.subheader("Category classifier (held-out accuracy)")
            rows = []
            for split in ("heldout_bitext", "heldout_seed_frames", "probes"):
                for method, m in (clf.get(split) or {}).items():
                    if isinstance(m, dict) and "accuracy" in m:
                        rows.append({"split": split, "method": method, "n": m.get("n"),
                                     "accuracy": m.get("accuracy"), "coverage": m.get("coverage"),
                                     "accuracy_when_covered": m.get("accuracy_when_covered")})
            if rows:
                table(pd.DataFrame(rows))
            raw(clf, "router_classifier.json")
        else:
            missing("router_classifier.json")
        if df is not None:
            st.subheader("Route mix by category (from the log)")
            rc = df.assign(category=df["category"].fillna("none")).groupby(["category", "route"]).size() \
                .rename("requests").reset_index()
            chart(alt.Chart(rc).mark_bar().encode(
                y=alt.Y("category:N", title=None),
                x=alt.X("requests:Q", stack="normalize", axis=alt.Axis(format="%"), title="share"),
                color=alt.Color("route:N", scale=route_scale, legend=alt.Legend(orient="top", title=None)),
                order=alt.Order("route:N"),
                tooltip=["category", "route", "requests"]).properties(height=260))

    # ---------------------------------------------------------------- Monitoring
    with tabs[6]:
        if df is None:
            missing("log", db)
        else:
            s = summarize(df)
            ts, xlabel = timeseries(df)
            xt = "T" if pd.api.types.is_datetime64_any_dtype(ts["x"]) else "Q"
            st.subheader("Operational")
            a, b = st.columns(2)
            with a:
                long = ts.melt("x", value_vars=["cost_per_request", "baseline_per_request"], var_name="series",
                               value_name="usd")
                chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
                    x=alt.X(f"x:{xt}", title=xlabel), y=alt.Y("usd:Q", title="USD per request"),
                    color=alt.Color("series:N", scale=alt.Scale(domain=["baseline_per_request", "cost_per_request"],
                                                               range=[NEUTRAL, SERIES[0]]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip(f"x:{xt}", title="bucket"), "series", alt.Tooltip("usd:Q", format="$.6f")])
                      .properties(height=220, title="Cost per request"))
            with b:
                long = ts.melt("x", value_vars=["p50_ms", "p99_ms"], var_name="series", value_name="ms")
                chart(alt.Chart(long).mark_line(strokeWidth=2).encode(
                    x=alt.X(f"x:{xt}", title=xlabel), y=alt.Y("ms:Q", title="ms"),
                    color=alt.Color("series:N", scale=alt.Scale(domain=["p50_ms", "p99_ms"],
                                                               range=[SERIES[0], SERIES[1]]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip(f"x:{xt}", title="bucket"), "series", alt.Tooltip("ms:Q", format=".1f")])
                      .properties(height=220, title="Latency"))
            a, b = st.columns(2)
            with a:
                chart(alt.Chart(ts).mark_line(strokeWidth=2, color=SERIES[2]).encode(
                    x=alt.X(f"x:{xt}", title=xlabel), y=alt.Y("hit_rate:Q", title="hit rate",
                                                              axis=alt.Axis(format="%")),
                    tooltip=[alt.Tooltip(f"x:{xt}", title="bucket"), alt.Tooltip("hit_rate:Q", format=".1%"),
                             "requests"]).properties(height=200, title="Cache hit rate"))
            with b:
                chart(alt.Chart(ts).mark_bar(cornerRadiusEnd=2, color=SERIES[1]).encode(
                    x=alt.X(f"x:{xt}", title=xlabel), y=alt.Y("errors:Q", title="requests with errors"),
                    tooltip=[alt.Tooltip(f"x:{xt}", title="bucket"), "errors", "requests"])
                      .properties(height=200, title="Errors (upstream + fail-open stages)"))

            st.subheader("Drift (PSI, course W3S2 bands: <0.10 stable, 0.10-0.25 investigate, >0.25 alert)")
            live = None
            if proxy_url:
                try:
                    import httpx
                    live = httpx.get(proxy_url.rstrip("/") + "/v1/drift", timeout=2.0).json()
                except Exception as e:
                    st.warning(f"Could not reach {proxy_url}/v1/drift: {type(e).__name__}")
            drift = live if (live and live.get("features")) else drift_offline(df)
            if drift is None:
                st.info("Need at least 60 requests in the log (or a live proxy URL) to compute PSI.")
            else:
                st.caption("Live /v1/drift from the proxy" if drift is live else
                           f"Offline: {drift['current_desc']} vs {drift['baseline_desc']} of this log")
                dt = drift_table(drift)
                ok = dt[dt["psi"].notna()]
                if not ok.empty:
                    bands = pd.DataFrame({"y": [0.10, 0.25], "label": ["investigate", "alert"]})
                    bars = alt.Chart(ok).mark_bar(cornerRadiusEnd=4, height=16, color=SERIES[0]).encode(
                        y=alt.Y("feature:N", title=None), x=alt.X("psi:Q", title="PSI"),
                        tooltip=["feature", alt.Tooltip("psi:Q", format=".3f"), "band", "n_baseline", "n_current"])
                    rules = alt.Chart(bands).mark_rule(strokeDash=[4, 4], color=NEUTRAL).encode(x="y:Q")
                    chart(alt.layer(bars, rules).properties(height=40 * len(ok) + 30))
                table(dt)
            st.subheader("Alert checks (thresholds from docs/RUNBOOK.md)")
            table(alert_checks(s, df, drift))

            st.subheader("Recent errors")
            errs = df[df["has_error"]].tail(25)
            if errs.empty:
                st.success("No upstream errors or fail-open stage errors in this log.", icon=":material/check:")
            else:
                table(errs[["time", "request_id", "mode", "error", "stage_errors"]].iloc[::-1])

            st.subheader("Load test (mock upstream)")
            if lt is None:
                missing("loadtest.json", RESULTS / "loadtest.json")
            else:
                c = st.columns(4)
                th = lt.get("throughput_rps")
                e2e = lt.get("end_to_end_ms", {})
                ov = lt.get("overhead_ms_header", {}) or lt.get("overhead_ms", {})
                c[0].metric("Throughput", f"{th:.1f} rps" if th else "n/a", border=True)
                c[1].metric("E2E p50 / p99", f"{_ms(e2e.get('p50'))} / {_ms(e2e.get('p99'))}", border=True)
                c[2].metric("Overhead p50 / p99", f"{_ms(ov.get('p50'))} / {_ms(ov.get('p99'))}", border=True)
                c[3].metric("Failures", f"{lt.get('failure_rate', 0) * 100:.2f}%", border=True)
                st.caption(lt.get("note", "Mock upstream: measures CostGuard's own overhead, not provider latency."))
                raw(lt, "loadtest.json")

    # ---------------------------------------------------------------- Explorer
    with tabs[7]:
        if df is None:
            missing("log", db)
        else:
            n = st.slider("Last N requests", 10, min(1000, max(10, len(df))), min(50, len(df)))
            last = df.tail(n).iloc[::-1]
            stages = stage_frame(last)
            cols = ["time", "request_id", "mode", "category", "cache_status", "route", "cache_similarity",
                    "input_tokens_original", "input_tokens_sent", "output_tokens", "cost_usd", "saved_usd",
                    "latency_ms", "overhead_ms"]
            view = pd.concat([last[[c for c in cols if c in last]], stages.round(2)], axis=1)
            table(view, height=360)
            rid = st.selectbox("Inspect request", list(last["request_id"]))
            r = last[last["request_id"] == rid].iloc[0]
            a, b = st.columns([3, 2])
            with a:
                st.markdown(f"**Query**  \n{r['query']}")
                st.markdown(f"**Response**  \n{str(r['response_text'])[:2000]}")
                st.caption(f"route: {r['route']} ({r.get('route_reason') or '-'}) | cache: {r['cache_status']} "
                           f"| config {r['config_hash']}")
                if r["stage_errors"]:
                    st.warning(f"Stage errors (failed open): {r['stage_errors']}")
            with b:
                sm = pd.DataFrame({"stage": list(r["stage_ms"]), "ms": list(r["stage_ms"].values())})
                if not sm.empty:
                    chart(alt.Chart(sm).mark_bar(cornerRadiusEnd=4, height=14, color=SERIES[0]).encode(
                        y=alt.Y("stage:N", sort=list(sm["stage"]), title=None), x=alt.X("ms:Q", title="ms"),
                        tooltip=["stage", alt.Tooltip("ms:Q", format=".2f")]).properties(
                        height=28 * len(sm) + 30, title="Stage timings"))


if __name__ == "__main__":
    main()
