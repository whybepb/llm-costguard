"""Load test the CostGuard service with the LLM mocked: measures the service itself
(caches, embedding, windowing, routing, Redis), not Anthropic's latency.

Usage:
  # separate instance (not scraped by Prometheus), mock-only; the "loadtest" tenant keeps its
  # cache keys apart. Don't use another Redis DB: RediSearch indexes are global, and
  # flushing that DB drops the shared vector index.
  COSTGUARD_LLM_MODES=mock HF_HUB_OFFLINE=1 \
    .venv/bin/uvicorn service.app:app --port 8100
  .venv/bin/python -m loadtest.run --url http://localhost:8100 --requests 600 --concurrency 16

Traffic: requests drawn at random (seeded) from the eval trace, so the mix has cold
misses, exact repeats and paraphrases (with only 40 distinct queries, mostly exact hits).
--unique tags every query with a request number, so every request is a cold miss: the
worst case (embedding, semantic lookup, windowing, routing, mocked LLM, cache writes). Reports p50/p95/p99
latency overall and per layer, throughput and errors; writes results/loadtest__*.csv.
"""

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


async def run(args):
    rows = [json.loads(line) for line in (ROOT / args.data / "queries.jsonl").open()]
    cases = json.loads((ROOT / args.data / "cases.json").read_text())
    rng = random.Random(args.seed)
    plan = [rng.choice(rows) for _ in range(args.requests)]
    if args.unique:
        plan = [{**r, "query": f"{r['query']} (ticket {i})"} for i, r in enumerate(plan)]
    results, sem = [], asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:
        (await client.delete("/api/cache", params={"tenant": args.tenant})).raise_for_status()

        async def one(r):
            body = {"query": r["query"], "tenant": args.tenant, "attachments": r["attachments"],
                    "case": cases[r["case_id"]] if r["case_id"] else None, "window_budget": args.budget,
                    "shadow": False, "llm_mode": "mock"}
            async with sem:
                t = time.perf_counter()
                try:
                    res = await client.post("/v1/answer", json=body)
                    ok = res.status_code == 200
                    layer = res.json().get("layer", "") if ok else f"http_{res.status_code}"
                except httpx.HTTPError as e:
                    ok, layer = False, type(e).__name__
                results.append({"layer": layer, "ok": int(ok), "latency_s": time.perf_counter() - t})

        start = time.perf_counter()
        await asyncio.gather(*(one(r) for r in plan))
        wall = time.perf_counter() - start

    ok = [x for x in results if x["ok"]]
    lat = [x["latency_s"] for x in ok]
    summary = {"traffic": "all_unique" if args.unique else "trace_mix", "concurrency": args.concurrency, "requests": len(results), "errors": len(results) - len(ok),
               "wall_s": round(wall, 2), "throughput_rps": round(len(ok) / wall, 1),
               "p50_ms": round(1000 * pct(lat, 50), 1), "p95_ms": round(1000 * pct(lat, 95), 1),
               "p99_ms": round(1000 * pct(lat, 99), 1), "mean_ms": round(1000 * statistics.mean(lat), 1),
               "mock_llm_latency_s": args.mock_latency}
    by_layer = []
    for layer in sorted({x["layer"] for x in ok}):
        ls = [x["latency_s"] for x in ok if x["layer"] == layer]
        by_layer.append({"traffic": summary["traffic"], "concurrency": args.concurrency, "layer": layer, "requests": len(ls),
                         "p50_ms": round(1000 * pct(ls, 50), 1), "p95_ms": round(1000 * pct(ls, 95), 1),
                         "p99_ms": round(1000 * pct(ls, 99), 1)})
    return summary, by_layer


def append_csv(path, rows):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8100")
    ap.add_argument("--data", default="eval/data/evalset")
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--tenant", default="loadtest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unique", action="store_true", help="make every request a cold miss")
    ap.add_argument("--mock-latency", type=float, default=0.0, help="label only: the server's COSTGUARD_MOCK_LATENCY_S")
    args = ap.parse_args()
    summary, by_layer = asyncio.run(run(args))
    out = ROOT / "results"
    append_csv(out / "loadtest__summary.csv", [summary])
    append_csv(out / "loadtest__latency_by_layer.csv", [{**x, "mock_llm_latency_s": args.mock_latency} for x in by_layer])
    print(json.dumps(summary))
    for x in by_layer:
        print(f"  {x['layer']:15s} n={x['requests']:4d}  p50 {x['p50_ms']}ms  p95 {x['p95_ms']}ms  p99 {x['p99_ms']}ms")


if __name__ == "__main__":
    main()
