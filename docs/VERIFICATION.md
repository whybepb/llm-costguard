# End-to-end verification

**Date:** 2026-10-04.
**Code under test:** `main` at `db5e5b2`. The run started on `6c3b3bf`; the history rewrite since then changed only
docs, `.gitignore` and one results file, so the code is identical.
**Environment:** macOS (Apple silicon), Python 3.12.13, `COSTGUARD_BACKEND=mock` throughout.

**What was not touched:**
- Nothing ran on the MLX backend. The GPU job (`eval.gate_router` → `eval.run_ab`, mlx) kept running.
- No files under `eval/cassettes/*`, `eval/results/ab_*`, `eval/results/router_gate*` or `eval/results/logs/*`
  were modified. `configs/router_gate.json` checksum was the same before and after.
- Every eval script that writes tracked files was pointed at a scratch directory with `--out`.

**Scope:** an examiner's path through the repo:
- fresh install;
- a live proxy driven by the official OpenAI SDK;
- tenancy, fail-open, the CI gate, the dashboard and the load test;
- Docker, the Makefile, and security.

## Summary

| | |
|---|---|
| Checks | 10 |
| PASS as found | 8 |
| FAIL as found, then fixed | 2 (#7 load test, #9 Makefile) |
| Small bugs fixed | 7 (listed below) |
| Core bugs reported, not fixed | 3, plus 2 minor (repro and patch below) |
| Test suite | `159 passed, 1 skipped` (was 158 + 1; one new test) |

## Checklist

| # | Check | Command | Result | Evidence |
|---|---|---|---|---|
| 1 | Fresh-clone install | `git clone` → new venv (cpython 3.12.13) → `pip install ".[dev,dashboard]"` → `pytest -q -rs` | **PASS** | The install resolved with no mlx/llmlingua. `157 passed, 2 skipped in 57.67s`. The two skips: `test_context.py:183` (LLMLingua-2, `RUN_SLOW=1`) and `test_eval.py:188` (Bitext not cached; `eval/data/raw` is gitignored). The fastembed models downloaded on first use because `models/` is gitignored. In the clean clone, `python -m eval.ci_gate` also returned PASS. Temp dir deleted. |
| 2 | Live server + official OpenAI SDK | `uvicorn costguard.server:get_app --factory` (free port, mock, temp `COSTGUARD_DB`), then `openai.OpenAI(base_url=…/v1)` with `with_raw_response` and `extra_body={"costguard":…}` | **PASS** (16/16) | Shape: `object=chat.completion`, `id=chatcmpl-…`, `usage` 125/73/198. All 8 `x-costguard-*` headers present. Repeat → `exact`, cost 0, saved $0.00042. Paraphrase "…usually take?" → `semantic` (sim 0.9877). `SN-48213`→`SN-48231` → miss, guard `number_mismatch` (sim 0.9724). "I want / I don't want to cancel…" → miss, guard `negation_mismatch` (sim 0.9381). 8 docs from `eval.kb.retrieve` → docs 8→4, tokens 1927→577, compression ×2.08. `mode:"off"` → `disabled`, saved 0.0, docs kept 8. Multi-turn and temperature 0.9 → `bypass`. Last message not `user` → 400. `/health`: all five stages real (`InMemoryExactCache`, `SemanticCacheImpl`, `RerankContextOptimizer`, `HeuristicCompressor`, `GatedRouter`); none is a no-op. The router never downshifts today, because `configs/router_gate.json` is a dry-run gate (`route_reason=gated:dry-run-gate`). Langfuse hook absent (no keys). `/v1/stats` OK. `/metrics`: 57 `costguard_*` series, `costguard_build_info{config_hash="b9bbbd4ee512"}`. `/v1/drift`: `warming_up`. **Found:** `stream=True` silently yields 0 chunks (core issue C3). |
| 3 | Service-side tenancy | Same server with `COSTGUARD_API_KEYS="k1:shopnest-support,k2:shopnest-internal"` | **PASS** (10/10) | No key or a wrong key → 401; the SDK raises `AuthenticationError`. With k1, body `mode:"economy"` and `tenant:"shopnest-internal"` still served `balanced` and logged tenant `shopnest-support`. k1 body `mode:"off"` was ignored. k2 defaults to `quality` and can override to `economy`. Isolation: the same question under k1 and k2 gave `miss,exact` for each tenant. A k2 paraphrase of a k1-only question → miss; best similarity in k2's own partition was 0.92. **Fixed:** `POST /v1/drift/baseline` returned 200 with no key and no token while caller keys were configured; it now returns 403. |
| 4 | Fail-open | (a) Live server with `FASTEMBED_CACHE_PATH=<empty> HF_HUB_OFFLINE=1`. (b) In process, monkeypatch every optional stage to raise. | **PASS** | (a) `/health` shows `semantic_cache: "build failed: Could not load model…"`. The reranker logs fallback cross-encoder → bi-encoder → lexical. Requests return 200. (b) The response is 200 with the full context, strong tier, and `stage_errors` = exact_cache, semantic_cache, context, compression, router, exact_write, semantic_write. The same keys are in the SQLite row, and `costguard_stage_errors_total` has 7 series. A raising raw hook is swallowed. A cheap-tier failure → `fallback-after-cheap-error` with `upstream_cheap` in `stage_errors`. A strong failure → 502 with an error row and `costguard_request_errors_total`. **Found:** if cheap fails and the strong retry also fails, there is no row and no error metric (core bug C2). |
| 5 | CI gate red/green | `python -m eval.ci_gate`, then `COSTGUARD_TAU_OVERRIDE=0.6 python -m eval.ci_gate` | **PASS** | Green: exit 0. Red: exit 1, with 11 trap false hits and 22 false hits overall. Full output below. Baseline not updated. The tracked `ci_gate.json` was restored after the run. |
| 6 | Dashboard | `streamlit run dashboard/app.py --server.headless true` (free port), then `curl /_stcore/health`. Also every data helper called directly, and the whole script run under `streamlit.testing.AppTest`. | **PASS** | Health `ok`, `GET /` 200, process killed. Helpers on the temp DB: `load_log` (14×48), `summarize`, `stage_frame`, `timeseries`, `alert_checks`. Helpers on every `eval/results/*.json` (A/B, waterfall, sweep, compression, gate, load test) all succeed. Missing file, empty SQLite and `None` inputs return `None` or `{}` and don't crash. AppTest with real data: 0 exceptions, 15 metrics, 8 tabs. With an empty results dir and a missing DB: 0 exceptions and 9 "how to produce this" infos. |
| 7 | Load test `--quick` | `bash loadtest/run.sh --quick` | **FAIL → fixed** | As found, `--quick` defaulted to `--out eval/results/loadtest.json`, so it would have **overwritten the headline 50-user result**. The `loadtest_quick.json` that the README cites can't have come from this flag. Caught by reading the script, before running it. After the fix it writes `loadtest_quick.json`, and `loadtest.json`'s sha1 `5e218dfe…` is unchanged. Numbers: 1,032 requests, 0 failures, 51.5 req/s. Hit-path overhead p50/p99 0.1/2.7 ms. Miss-path overhead 4.5/33.8 ms. Miss client-added p99 46.6 ms. E2E p50/p99 1.7/327 ms. No stage errors. **Second fix:** the server log was missing 15–16 rows per quick run (1,025 responses vs 1,010 rows) because of core bug C1. `run.sh` now drains the log through `/v1/stats` before the kill, and rows equal responses (1,032 = 1,032). |
| 8 | Docker | `docker info` | **PASS** (static review; daemon not running, none started) | Reviewed `Dockerfile`:<br>- it is multi-stage, and the default target is `proxy`;<br>- the fastembed models are baked in at build time;<br>- it runs as a non-root user;<br>- `exec uvicorn` makes it PID 1;<br>- the HEALTHCHECK hits `/health`;<br>- serving reads only `costguard/` and `configs/`, both copied in;<br>- `centroids.npz` is in the package dir.<br><br>`docker-compose.yml`: `qdrant/qdrant:v1.19.1` is wired to the proxy (`COSTGUARD_SEMANTIC_BACKEND=qdrant`, `QDRANT_URL=http://qdrant:6333`, `depends_on`), and the dashboard mounts the proxy volume read-only.<br><br>`render.yaml` has no `dockerCommand`, so Render runs the Dockerfile `CMD`; `healthCheckPath: /health` matches. Risks are R7 and R12. |
| 9 | Makefile reproducibility | Each target, with tracked outputs redirected to scratch | **FAIL → fixed** | As found, `COSTGUARD_BACKEND=anthropic make serve` (RUNBOOK §1) still ran **mock**, because the recipe hard-codes `COSTGUARD_BACKEND=mock` (`make -n` showed it). Fixed, and `PORT` is now overridable. `make serve PORT=<free>` → health ok, `x-costguard-cache: miss`, killed. `make test` → `159 passed, 1 skipped`. `make ci-gate` → PASS. Trace: `eval.build_trace --out <scratch>` is **byte-identical** to `trace_v1.jsonl` (sha `19c0834c…`), and `--ci-subset` is byte-identical too. Report: `eval.report --out <scratch>` matches `docs/RESULTS.md` with no diff. `sweep_threshold --quick` (CPU, 9 s) recommends τ 0.95/0.93/0.89, which equals `policy.yaml`. `run_ab --backend mock --limit 80 --out-dir <scratch>`: 6 arms; savings A1 8.5% → A5 48.8%. `gate_router --backend mock --dry-run --gate-out/--results-out <scratch>` OK; `configs/router_gate.json` unchanged. Not run: `serve-mlx` (GPU), `make gate`/`make ab`/full `make sweep` in place (they write files I may not touch), `make dashboard` (opens a browser; the headless equivalent passed in #6), full `make loadtest` (writes `loadtest.json`; same script as #7). |
| 10 | Security and privacy | grep over tracked + untracked-unignored files | **PASS** | **Secrets:** no key-shaped strings (`sk-ant-`, `sk-…`, `AKIA`, `ghp_`, `hf_`, `AIza`, `xox`, private keys, Langfuse `pk-lf`/`sk-lf`) and no literal secret assignments. **Env files:** `.env` is ignored and only `.env.example` (placeholder) is tracked; `.env.*` is now ignored too. **Personal data:** no real people's names, IDs or institutional emails. The only person names are the fictional ShopNest names in `build_trace.PERSON_NAMES` and public celebrity names in the QQP pairs. **Emails:** all are `@shopnest.example`, except `mail@live.com` in the public QQP data and `support@shopnest.com` in a model-generated cassette answer. **Local paths:** no `/Users/…` paths in any code. At `6c3b3bf` four docs/results files had them; after the coordinator's rewrite, no commit on any ref contains `/Users/`. **Fixed:** `gate_router` would write absolute paths into the next gate results file. **Size:** no file over 5 MB in the tree or the history; the largest blob is `trace_v1.jsonl` at 1.76 MB, and `.git` is 4 MB. |

### CI gate output (screenshot material)

Green, on the current code (`python -m eval.ci_gate`, exit 0):

```text
CostGuard CI eval gate: PASS  (backend=mock, judge=heuristic-mock, rows=78, tau=0.93, config=b9bbbd4ee512)
check                                           value                         baseline                      limit
replay complete (no upstream/cassette errors)   0                             0                             == 0          PASS
(a) trap false hits                             0                             0                             == 0          PASS
baseline matches backend/judge/subset           mock/heuristic-mock/2f5d554b  mock/heuristic-mock/2f5d554b  equal         PASS
(a') false hits, all rows                       0                             0                             <= 0          PASS
(b) quality, mean grade (0-1)                   0.0483                        0.0485                        >= 0.0185     PASS
(c) savings vs A0 (%)                           61.15                         60.49                         >= 58.49      PASS
hit rate 10.26% (exact 8, semantic 0), savings 61.15% [54.3, 66.38], quality retained 109.89%
```

Red (`COSTGUARD_TAU_OVERRIDE=0.6 python -m eval.ci_gate`, exit 1). This is after the listing fix (F6), so all 11 trap
items are named:

```text
CostGuard CI eval gate: FAIL  (backend=mock, judge=heuristic-mock, rows=78, tau=0.6, config=a91c7722635b)
(a) trap false hits                             11                            0                             == 0          FAIL
(a') false hits, all rows                       22                            0                             <= 0          FAIL
(b) quality, mean grade (0-1)                   0.0419                        0.0485                        >= 0.0185     PASS
(c) savings vs A0 (%)                           73.27                         60.49                         >= 58.49      PASS
hit rate 35.9% (exact 8, semantic 20), savings 73.27% [66.17, 78.96], quality retained 95.44%
  TRAP false hit: 'edit info on Elite accunt' <- served answer of 'update information on Pro account' (semantic, sim 0.7494)
  TRAP false hit: 'using Elite account' <- served answer of 'update information on Pro account' (semantic, sim 0.6867)
  TRAP false hit: 'Can I change the size of a shirt in an order I placed this morning?' <- served answer of 'How much does shipping cost for an order within India?' (semantic, sim 0.6062)
  TRAP false hit: 'My order already shows Shipped. Can I still cancel it?' <- served answer of 'How much does shipping cost for an order within India?' (semantic, sim 0.6155)
  TRAP false hit: 'find information about the deletion of a Platinum account' <- served answer of 'find information about opening a Platinum account' (semantic, sim 0.8631)
  TRAP false hit: 'delieries to Port Blair' <- served answer of 'delieries to Bengaluru' (semantic, sim 0.7694)
  TRAP false hit: 'use Plus account' <- served answer of 'edit Plus accont' (semantic, sim 0.7312)
  TRAP false hit: 'create nbew Seller account' <- served answer of 'i want information about creatign a Freemium account' (semantic, sim 0.6755)
  TRAP false hit: 'change to Platinum acount' <- served answer of 'find information about opening a Platinum account' (semantic, sim 0.7569)
  TRAP false hit: "I want a refund for the jeans I bought last week, they don't fit." <- served answer of "Money was debited from my account but my order wasn't confirmed. What now?" (semantic, sim 0.6236)
  TRAP false hit: 'How long is the warranty on Nestra headphones?' <- served answer of 'Can I return earbuds if I have not opened the hygiene seal?' (semantic, sim 0.635)
  false hit: … (11 non-trap false hits follow, including 2 served by the exact tier after semantic-hit promotion; see R10)
```

## Bugs found and fixed

| # | File | What | Why |
|---|---|---|---|
| F1 | `loadtest/run.sh`, `loadtest/README.md` | `--quick` now writes `eval/results/loadtest_quick.json`; an explicit `--out` still wins. | The default was `loadtest.json` for every profile. A quick smoke run silently overwrote the 50-user headline numbers that ARCHITECTURE §3, the README and the dashboard cite. |
| F2 | `loadtest/run.sh` | Before stopping the proxy, call `GET /v1/stats`, whose `rows()` flushes the logger. | Works around core bug C1. Without it, `server_log` (the stage-latency table the docs quote) dropped the last ~1.5% of requests. |
| F3 | `loadtest/README.md`, `loadtest/run.sh` (note string) | Say the request log *enqueues* rather than "writes one SQLite row synchronously". | `costguard/obs/logger.py` is an async batched writer since the Ops commit. |
| F4 | `Makefile` | `serve` uses `COSTGUARD_BACKEND=$${COSTGUARD_BACKEND:-mock}`; new `PORT ?= 8000` for `serve`/`serve-mlx`; the `report` comment names `docs/RESULTS.md`. | `COSTGUARD_BACKEND=anthropic make serve` (RUNBOOK §1, "real API") ran the mock. |
| F5 | `costguard/obs/metrics.py`, `tests/test_ops.py` | `POST /v1/drift/baseline`:<br>- uses `hmac.compare_digest`;<br>- **fails closed (403)** when `COSTGUARD_API_KEYS` is set but `COSTGUARD_ADMIN_TOKEN` is not;<br>- keyless dev mode is unchanged.<br><br>New test `test_drift_rebaseline_auth`. | On a keyed deployment with no token (the `docker-compose.yml` default), any caller could re-freeze the drift baseline and mask a drift alert. |
| F6 | `eval/ci_gate.py` | The gate log and `ci_gate.json` list every trap false hit first, capped at 25 (was the first 10 in trace order). | At τ = 0.6, 4 of the 11 failing trap items were not named. RUNBOOK §8's screenshot relies on "gate log lines naming the failed trap items". |
| F7 | `eval/gate_router.py` | `results["written"]` paths go through the existing `_rel()`. | They were absolute (`/Users/<name>/…`), which is how the scrubbed path had entered `router_gate_dryrun.json`. The running mlx job loaded the old code, so its output still needs a scrub (R3). |
| — | `.gitignore` | Added `.env.*`, `!.env.example`, `build/`, `*.egg-info/`. Already committed by the coordinator in `db5e5b2`. | `pip install .` leaves `build/` and `llm_costguard.egg-info/` untracked, and a `.env.local` would have been committable. |

## Core bugs found by the verifier

> **Status: all fixed after this report**, each with a regression test in `tests/test_engine.py` / `tests/test_router.py`:
> - **C1:** a FastAPI lifespan drains the request log on shutdown.
> - **C2:** a failed strong retry now records `error`, emits the row and counts the error.
> - **C3:** `stream=true` returns 400.
> - **R1:** a gate measured on another model pair now routes everything to strong (warn-only on `mock`).
> - **R14:** `openai` is in the `dev` extra.
>
> The analysis below is kept as found.

### C1. Request-log rows are lost on every SIGTERM (redeploy, Render sleep, `docker stop`, end of the load test)

Files: `costguard/obs/logger.py`, `costguard/server.py`.

**What goes wrong:**
- `RequestLogger` relies on `atexit.register(self.flush)`.
- Recent uvicorn (0.54 here) re-raises SIGTERM with the default handler after its graceful shutdown, so the
  process dies and **atexit never runs**. A bare FastAPI app with an atexit hook confirmed this: exit 143, hook not
  run.
- The writer only commits a batch when 200 rows have accumulated or no new row has arrived for 0.25 s. Under steady
  traffic a pending batch can therefore hold up to 199 rows.

Repro (mock, 30 requests about 15/s, then SIGTERM right after the last response):

```python
# start: COSTGUARD_BACKEND=mock COSTGUARD_DB=/tmp/x.sqlite uvicorn costguard.server:get_app --factory --port $P
for i in range(30): httpx.post(f"{url}/v1/chat/completions", json={"messages": [{"role": "user", "content": f"Where is my order SN-{20000+i}?"}]}); time.sleep(0.05)
os.kill(pid, signal.SIGTERM)   # then count rows
# -> 200 responses: 30; rows after exit: 0; lost: 30      (3/3 runs)
```

Suggested patch, validated: the same repro with a wrapper app that adds this lifespan lost **0** rows.

```python
# costguard/server.py, create_app(): flush the log during uvicorn's graceful shutdown (before it re-raises SIGTERM)
from contextlib import asynccontextmanager
...
    req_logger = next((h for h in engine.hooks if isinstance(h, RequestLogger)), None)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if req_logger is not None:
            req_logger.flush()

    app = FastAPI(title="LLM CostGuard", version="0.1.0", lifespan=lifespan)
```

It is also worth bounding the batch by a deadline, not a per-item timeout, in `logger.py` `_writer`. Then rows are
durable within 0.25 s even under constant traffic, and the dashboard (which reads SQLite directly) lags by at most
0.25 s, instead of up to 200 × 0.25 s at low steady traffic:

```python
deadline = time.monotonic() + self._interval
while len(batch) < self._batch_size:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        break
    batch.append(self._q.get(timeout=remaining)); done += 1
```

### C2. If the cheap tier fails and its strong retry also fails, the request leaves no trace

File: `costguard/pipeline.py`, stage 6.

**What goes wrong:**
- The fallback `self.provider.complete(sent, strong_id, …)` is outside any `try`.
- If it raises, `handle()` exits without setting `rec.error` and without `_emit(rec)`.
- So there is no SQLite row, no `costguard_request_errors_total`, and the client gets a 502.
- This is exactly a full-provider outage, which is what the RUNBOOK P1 error-rate alert exists for, and it
  undercounts.

Repro: in-process `TestClient`, with `eng.router.route` forced to return `cheap` and `eng.provider.complete` raising.
The result is a 502 with 0 rows logged and the metric unchanged. With the router forced to `strong`, the same failure
gives 1 row and the metric goes up by 1.

Patch:

```python
            rec.model_used, rec.route_reason, rec.model_id = "strong", "fallback-after-cheap-error", strong_id
            try:
                comp = self.provider.complete(sent, strong_id, max_tokens, req.temperature)
            except Exception as e2:
                rec.error = f"{type(e2).__name__}: {e2}"[:300]
                rec.latency_ms = (time.perf_counter() - t0) * 1000
                self._emit(rec)
                raise
```

### C3. `stream=True` from the OpenAI SDK silently returns an empty stream

Files: `costguard/schemas.py`, `costguard/server.py`.

**What goes wrong:**
- `ChatRequest` ignores the unknown `stream` field, so the server returns plain JSON.
- `client.chat.completions.create(..., stream=True)` then iterates **0 chunks** with no error.
- The request is still billed and logged.
- Streaming is out of scope (ARCHITECTURE §5), but a drop-in client that streams gets empty answers instead of a clear
  error.

Patch:
- add `stream: Optional[bool] = None` to `ChatRequest`;
- at the top of `chat()`:
  `if req.stream: raise HTTPException(400, "stream=true is not supported by CostGuard; use stream=false")`.

### Minor (core)

- **`temperature: null` is rejected with 422.** Accept `Optional[float]` and treat `None` as 0.
- **Provider-raised `ValueError` subclasses become 400s.** `server.chat` maps every `ValueError` to 400, and that
  includes provider errors such as a pydantic `ValidationError`, which should be 502. Raise a dedicated
  `RequestError` for client mistakes in `pipeline.handle` and catch only that.
- **`mode:"off"` reports non-zero savings on Anthropic.** The baseline uses the ratio-scaled *estimate*
  `input_tokens_original`, while cost uses actual usage. Mock is exact. When nothing changed the prompt and the route
  is strong, use the actual cost as the baseline. Or scale the estimate by `actual / estimate(sent)`.

## Open risks (for the coordinator)

- **R1. The gate tier mismatch is only a warning.**
  - The running mlx gate run will write `configs/router_gate.json` with `backend: mlx` (Qwen 7B vs 1.5B).
  - `GatedRouter._check_gate_matches` only logs a warning, so Render (`anthropic`, Sonnet vs Haiku) would downshift
    on Qwen evidence.
  - Consider treating a backend/model mismatch as `dry_run`.
- **R2. The CI gate needs re-checking after the gate lands.**
  - A new `router_gate.json` changes `config_hash`.
  - The CI gate replays on **mock**, so allowed categories will route to `mock-cheap`, and savings and quality move.
  - Re-run `make ci-gate`; `--update-baseline` may be needed in the same PR.
  - The committed `ci_baseline.json` is already from config `b4291e45d86e`; HEAD is `b9bbbd4ee512` (savings
    60.49 → 61.15, within margin).
- **R3. Scrub the paths the mlx run writes.** The mlx `gate_router` started before F7, so the
  `eval/results/router_gate*.json` it writes will contain absolute `/Users/…` paths in `"written"`. Scrub them before
  committing.
- **R4. A non-editable install works only from the repo root.**
  - `costguard.config.ROOT` resolves to `site-packages/..`, so a non-editable install can't find `configs/`.
  - `centroids.npz` is not package data.
  - It works from the repo root (cwd shadows the install), with `pip install -e`, and in Docker.
  - Document `-e`, or package `configs/` and the npz.
- **R5. Unauthenticated ops endpoints.**
  - These need no key even when keys are set: `/v1/stats` (aggregate cost across all tenants, plus a full 24 h table
    scan on every call), `/metrics`, `/v1/drift` and `/health`.
  - That is fine on a private network, but they are exposed on the public Render URL.
- **R6. `/health` hides degraded components.** It shows class names only. A reranker that silently fell back to
  lexical (models missing) still reads `RerankContextOptimizer`.
- **R7. Compose start-up race.**
  - `depends_on: qdrant` has no health condition.
  - If Qdrant isn't accepting connections when the proxy builds the semantic cache, the stage becomes a no-op for the
    life of the process; there is no retry, and `restart:` won't help because nothing crashes.
  - Add a Qdrant healthcheck with `condition: service_healthy`, or retry in `QdrantStore`.
- **R8. There is no top-level `README.md`.**
  - CONTRACT, RUNBOOK §8, the Makefile and `eval/report.py` all refer to "the README".
  - The deliverable needs one with the numbers.
- **R9. Docs still say the request-log write is synchronous.** See ARCHITECTURE §1 (line ~80), the §3 latency table
  row "Request-log write" and DESIGN_DECISIONS #12. I left them alone because docs were being edited concurrently;
  `loadtest/README.md` is fixed. RUNBOOK's `COSTGUARD_ADMIN_TOKEN` row should also mention F5's fail-closed behaviour.
- **R10. Promotion can spread a wrong semantic hit.**
  - Promoting semantic hits into the exact tier means one wrong semantic hit is then served by the "zero-risk" exact
    tier on every verbatim repeat, for the TTL.
  - The τ = 0.6 run shows 2 exact false hits created this way.
  - The guards are not re-run on promoted entries.
- **R11. The calibrated CI gate barely exercises the semantic cache.** At τ = 0.93 the CI subset has 0 semantic hits
  on mock (8 exact). Check (a) therefore tests the guards weakly at the calibrated τ, although the τ = 0.6 run shows it
  does catch regressions.
- **R12. Stale comments.**
  - `render.yaml` says the semantic store doesn't pass a Qdrant Cloud key; `QDRANT_API_KEY` is supported now.
  - The Dockerfile and `.github/workflows/ci.yml` say `anthropic` and `python-dotenv` are "not yet declared" in
    pyproject; they are. The duplicates are harmless.
  - The CI `eval-gate` job still has a "ci_gate.py not found" fallback.
- **R13. Cosmetic load-test counts.** Locust's CSV "requests" can differ by 1–4% from the per-request samples used
  for the percentiles (1,028 vs 1,032).
- **R14. The OpenAI SDK is undeclared.** It is used by the docs and this verification, and is present only through
  `litellm`. Add `openai` to the `dev` extra.

## How the checks were run

The throwaway scripts lived in the session scratchpad and are not committed. To reproduce:

1. Start `uvicorn costguard.server:get_app --factory` on a free port, with `COSTGUARD_BACKEND=mock`,
   `COSTGUARD_DB=<tmp>` and `COSTGUARD_CASSETTE=""`, with and without `COSTGUARD_API_KEYS`.
2. Drive it with the OpenAI SDK as in the table.
3. Kill the server.

The mock A/B, router gate, sweep, trace and report runs all used `--out`/`--out-dir`/`--gate-out`/`--results-out`
into the scratch directory, so tracked files stayed untouched.

Every process I started (uvicorn, Streamlit, Locust) was stopped. The only Python processes left running are the
coordinator's mlx experiment.
