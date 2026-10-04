# Load test: CostGuard overhead and throughput

```bash
make loadtest                      # = bash loadtest/run.sh        (50 users, 60 s)
bash loadtest/run.sh --quick       # 20 users, 20 s  (writes eval/results/loadtest_quick.json)
bash loadtest/run.sh --users 100 --duration 120s --latency-ms 800 --out eval/results/loadtest_800ms.json
```

`run.sh` does the following:

1. Starts the proxy on a free port with `COSTGUARD_BACKEND=mock COSTGUARD_MOCK_LATENCY_MS=300`, a throwaway SQLite log, Langfuse off and open dev auth.
2. Waits for `/health`, then warms up on a separate tenant. This loads the ONNX models without pre-filling the test's caches.
3. Runs Locust headless.
4. Stops the proxy.
5. Writes `eval/results/loadtest.json`, which the dashboard's Monitoring tab and the README read.

## What it measures, and what it does not

**The upstream is a mock: a fixed 300 ms sleep that returns canned tokens.** So the numbers are **CostGuard's own overhead and throughput**, not provider latency or provider rate limits. That is deliberate:

- It costs $0.
- It needs no key.
- It never trips a provider 429.
- It isolates the one latency the team controls.

Never point this at Render. The free host is a single instance and can be suspended for unusual traffic.

| Field in `loadtest.json` | Meaning |
|---|---|
| `throughput_rps` | Locust aggregate requests/s, with 0 failures required. This is offered load, not a saturation point: 50 users with a 0.1–0.5 s think time. |
| `end_to_end_ms` | Client-observed latency, exact percentiles over every request. |
| `overhead_ms_header` | `x-costguard-overhead-ms`: time spent in CostGuard stages, i.e. total minus upstream, as measured inside the engine. |
| `hit_path` / `miss_path` | The same numbers split by `x-costguard-cache`. A blended p50 hides that hits never touch the provider. |
| `miss_path.client_added_ms` | Client latency minus the 300 ms mock sleep. Adds HTTP, JSON, threadpool queueing and the hook fan-out (metrics, drift, request-log enqueue) on top of `overhead_ms`. |
| `server_log.stage_ms` | Per-stage p50/p95/p99 from the request log. This is the measured column of the latency budget in `docs/ARCHITECTURE.md`. |
| `components` | `/health` component status at run time, so every number traces to the stages that were live. |

## Traffic mix (`locustfile.py`)

| Kind | Share | What it exercises |
|---|---|---|
| `repeat` | 40% | Canonical wording of 10 Zipf-weighted ShopNest questions. Should hit the exact cache after the first request. |
| `paraphrase` | 30% | Reworded variants. These are semantic-cache candidates. |
| `context` | 20% | KB questions with 4–6 policy documents attached. This is the context-rerank and compression path. Half of them send `no_cache: true`, so rerank and compression run under load instead of being cached. |
| `novel` | 10% | A fresh order number each time. Always a miss, and it probes the number guard. |

The mix is a stress profile, not the workload trace. Its 90% repeat/paraphrase share makes `server_log.saved_pct` meaningless as a savings claim. **Savings numbers come only from the frozen-trace A/B (`make ab`).**

## Latest results (`eval/results/loadtest.json`, 50 users, 60 s, Apple-silicon laptop)

All five stages were live; `components` in the JSON records them.

| Metric | Value |
|---|---|
| Requests / failures | 7,667 / 0 |
| Throughput | 127.6 req/s, offered-load bound |
| End-to-end p50 / p95 / p99 | 6.3 / 342.0 / 380.3 ms |
| CostGuard overhead, all requests, p50 / p99 | 0.3 / 57.2 ms |
| **Hit path** (exact + semantic, n = 6,116) overhead p50 / p99 | **0.3 / 3.6 ms**, within the 50 ms SLO |
| **Miss path** (n = 1,552) overhead p50 / p99 | **11.8 / 94.6 ms**, within the 100 ms budget |
| RAG requests that bypass the cache (n = 745) overhead p50 / p99 | 25.9 / **105.8 ms**, over budget |
| Miss path client-added latency p50 / p99 | 25.1 / 124.1 ms |
| Heaviest stage | `context` cross-encoder rerank, p50 25.1 / p99 104.4 ms (n = 753) |

`eval/results/loadtest_quick.json` holds the `--quick` run: 20 users, 20 s, 987 requests, 49.4 req/s, 0 failures.

**The finding:** under 50 concurrent users, the CPU-bound cross-encoder pushes RAG misses past the 100 ms miss-path budget. Hits are about 30× inside their SLO. See DESIGN_DECISIONS §3 for the fix order.

## Reading the results honestly

- **Throughput vs threads.** The chat endpoint is a sync `def`, so it runs in AnyIO's threadpool (40 threads) on one uvicorn worker. With a 300 ms upstream, pure-miss traffic tops out near 40 / 0.3 s ≈ 133 req/s per worker, whatever the CPU. Cache hits release the thread in a few ms. To scale out, add workers (`--workers N`) or replicas behind a shared Redis/Qdrant cache.
- **CPU-bound stages.** On the miss path, the stages that burn CPU are the ONNX embedder and the cross-encoder. They are the first to degrade under concurrency: watch `costguard_stage_ms{stage="context"}`.
- **What sits outside `overhead_ms`.** The hooks run after the record is complete: Prometheus, drift, and the core `RequestLogger`, which only enqueues the row (a background thread batches the SQLite inserts). That time sits outside `overhead_ms`, but it is included in `client_added_ms`. Rows reach SQLite within ~0.25 s of a pause in traffic; see docs/VERIFICATION.md for the shutdown caveat.
