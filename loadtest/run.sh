#!/usr/bin/env bash
# Load test CostGuard against a MOCK upstream (fixed sleep, $0, no keys): measures CostGuard's own overhead and
# throughput, not provider latency. See loadtest/README.md.
#
#   bash loadtest/run.sh             # 50 users, 60 s
#   bash loadtest/run.sh --quick     # 20 users, 20 s
#   bash loadtest/run.sh --users 100 --duration 120s --latency-ms 800 --mode economy --out eval/results/loadtest_economy.json
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

USERS=50; SPAWN=10; DURATION=60s; LATENCY_MS=300; MODE=balanced; OUT="eval/results/loadtest.json"; PROFILE=full
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) USERS=20; SPAWN=10; DURATION=20s; PROFILE=quick ;;
    --users) USERS="$2"; shift ;;
    --spawn-rate) SPAWN="$2"; shift ;;
    --duration) DURATION="$2"; shift ;;
    --latency-ms) LATENCY_MS="$2"; shift ;;
    --mode) MODE="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PORT="$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/costguard-loadtest.XXXXXX")"
DB="$RUN_DIR/requests.sqlite"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "[loadtest] proxy on 127.0.0.1:$PORT  backend=mock  mock_latency=${LATENCY_MS}ms  mode=$MODE  run_dir=$RUN_DIR"
# Explicit env wins over any .env (load_dotenv(override=False)); Langfuse off so the test never leaves the machine.
COSTGUARD_BACKEND=mock COSTGUARD_MOCK_LATENCY_MS="$LATENCY_MS" COSTGUARD_DB="$DB" COSTGUARD_CASSETTE="" \
LANGFUSE_PUBLIC_KEY="" LANGFUSE_SECRET_KEY="" COSTGUARD_API_KEYS="" \
  "$PY" -m uvicorn costguard.server:get_app --factory --host 127.0.0.1 --port "$PORT" --log-level warning \
  >"$RUN_DIR/proxy.log" 2>&1 &
PROXY_PID=$!
cleanup() { kill "$PROXY_PID" 2>/dev/null || true; wait "$PROXY_PID" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 240); do   # up to 120 s: first start may load the ONNX embedding models
  if curl -fsS "http://127.0.0.1:$PORT/health" -o "$RUN_DIR/health.json" 2>/dev/null; then break; fi
  if ! kill -0 "$PROXY_PID" 2>/dev/null; then echo "[loadtest] proxy died:"; cat "$RUN_DIR/proxy.log"; exit 1; fi
  sleep 0.5
done
[ -s "$RUN_DIR/health.json" ] || { echo "[loadtest] proxy did not become healthy"; cat "$RUN_DIR/proxy.log"; exit 1; }
echo "[loadtest] health: $(cat "$RUN_DIR/health.json")"

# Warm-up on a separate tenant (own cache partition): loads lazy models without pre-filling the test's caches.
for q in "warm up one" "warm up two please" "warm up three thanks"; do
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"costguard\":{\"tenant\":\"warmup\",\"mode\":\"$MODE\",\"context\":[\"Returns are accepted within 30 days of delivery for a full refund.\"]}}" \
    -o /dev/null || true
done

echo "[loadtest] locust: users=$USERS spawn=$SPAWN duration=$DURATION"
COSTGUARD_LOADTEST_SAMPLES="$RUN_DIR/samples.json" COSTGUARD_LOADTEST_MODE="$MODE" \
  "$PY" -m locust -f loadtest/locustfile.py --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" \
  --host "http://127.0.0.1:$PORT" --csv "$RUN_DIR/locust" --only-summary --stop-timeout 5 --loglevel WARNING \
  --exit-code-on-error 0 2>&1 | tail -n 30

cleanup   # stop the proxy before reading its SQLite log
trap - EXIT

mkdir -p "$(dirname "$OUT")"
RUN_DIR="$RUN_DIR" OUT="$OUT" USERS="$USERS" SPAWN="$SPAWN" DURATION="$DURATION" LATENCY_MS="$LATENCY_MS" \
MODE="$MODE" PROFILE="$PROFILE" "$PY" - <<'PYEOF'
import csv, json, os, platform, sqlite3, subprocess, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

run = Path(os.environ["RUN_DIR"]); out = Path(os.environ["OUT"]); lat = float(os.environ["LATENCY_MS"])


def pcts(xs, qs=(50, 95, 99)):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float)
    d = {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs}
    d.update(n=len(xs), mean=round(float(a.mean()), 2), max=round(float(a.max()), 2))
    return d


rows = list(csv.DictReader(open(run / "locust_stats.csv")))
agg = next(r for r in rows if r["Name"] == "Aggregated")
n_req, n_fail = int(agg["Request Count"]), int(agg["Failure Count"])
samples = json.load(open(run / "samples.json"))["samples"] if (run / "samples.json").exists() else []
ok = [s for s in samples if s[4] == 200]
hit = [s for s in ok if s[1] in ("exact", "semantic")]
miss = [s for s in ok if s[1] not in ("exact", "semantic")]

by_kind = {}
for kind in sorted({s[0] for s in ok}):
    ks = [s for s in ok if s[0] == kind]
    by_kind[kind] = {"requests": len(ks), "hit_rate": round(sum(s[1] in ("exact", "semantic") for s in ks) / len(ks), 4),
                     "client_ms": pcts([s[3] for s in ks]), "overhead_ms": pcts([s[2] for s in ks])}

log = {}
db = run / "requests.sqlite"
if db.exists():
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        lr = [dict(r) for r in c.execute("SELECT * FROM requests WHERE tenant != 'warmup'")]
    stage = defaultdict(list)
    for r in lr:
        for k, v in json.loads(r["stage_ms"] or "{}").items():
            stage[k].append(v)
    by_status = defaultdict(list)
    for r in lr:
        by_status[r["cache_status"]].append(r["overhead_ms"])
    cost, base = sum(r["cost_usd"] for r in lr), sum(r["baseline_cost_usd"] for r in lr)
    log = {"requests": len(lr), "latency_ms": pcts([r["latency_ms"] for r in lr]),
           "overhead_ms": pcts([r["overhead_ms"] for r in lr]),
           "overhead_ms_by_cache_status": {k: pcts(v) for k, v in sorted(by_status.items())},
           "stage_ms": {k: pcts(v) for k, v in sorted(stage.items())},
           "stage_errors": dict(Counter(k for r in lr for k in json.loads(r["stage_errors"] or "{}"))),
           "saved_pct": round(100 * (1 - cost / base), 2) if base else None}

health = json.load(open(run / "health.json")) if (run / "health.json").exists() else {}
try:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    sha = ""
dur = os.environ["DURATION"]
res = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_sha": sha, "profile": os.environ["PROFILE"],
    "config_hash": health.get("config_hash"), "components": health.get("components"),
    "setup": {"backend": "mock", "mock_latency_ms": lat, "users": int(os.environ["USERS"]),
              "spawn_rate": int(os.environ["SPAWN"]), "duration": dur, "mode": os.environ["MODE"],
              "server": "1 uvicorn worker, sync endpoint (anyio threadpool, 40 threads)", "client": "locust, same host",
              "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}"},
    "requests": n_req, "failures": n_fail, "failure_rate": round(n_fail / n_req, 5) if n_req else None,
    "throughput_rps": round(float(agg["Requests/s"]), 2),
    "end_to_end_ms": pcts([s[3] for s in ok]),
    "end_to_end_ms_locust": {"p50": float(agg["50%"]), "p95": float(agg["95%"]), "p99": float(agg["99%"])},
    "hit_path": {"requests": len(hit), "client_ms": pcts([s[3] for s in hit]), "overhead_ms": pcts([s[2] for s in hit])},
    "miss_path": {"requests": len(miss), "client_ms": pcts([s[3] for s in miss]),
                  "overhead_ms": pcts([s[2] for s in miss]),
                  "client_added_ms": pcts([s[3] - lat for s in miss])},
    "overhead_ms_header": pcts([s[2] for s in ok]),
    "cache_status_mix": dict(Counter(s[1] for s in ok)),
    "by_kind": by_kind,
    "server_log": log,
    "note": ("Mock upstream (fixed %d ms sleep): measures CostGuard's own overhead and throughput, not provider "
             "latency. overhead_ms = CostGuard stage time from the x-costguard-overhead-ms header (total - upstream). "
             "miss_path.client_added_ms = client-observed latency minus the mock sleep, so it also includes HTTP, "
             "JSON, threadpool queueing and the synchronous request-log write." % lat),
}
out.write_text(json.dumps(res, indent=2))
print(json.dumps({k: res[k] for k in ("requests", "failure_rate", "throughput_rps", "end_to_end_ms",
                                     "overhead_ms_header", "cache_status_mix")}, indent=2))
print(f"hit path p50/p99 overhead: {res['hit_path']['overhead_ms']}")
print(f"miss path client-added p50/p99: {res['miss_path']['client_added_ms']}")
print(f"[loadtest] wrote {out}")
PYEOF
