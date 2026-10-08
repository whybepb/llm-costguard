# LLM CostGuard

A drop-in layer in front of an LLM that cuts spend without hurting answer quality: an exact cache, a guarded semantic cache, context-window management, optional prompt compression and model routing. Every stage is measured against a naive baseline (Sonnet 5.5 with the full context), and a dashboard shows the savings per request.

Domain: customer support for **Northstar Market**, a fictional store with 23 policy documents (returns, refunds, cancellations, delivery, marketplace sellers, gift cards, privacy, and more). Each request carries a query, a customer case, and a bundle of attached policies (about 5,500 tokens when large) of which usually one matters.

> Portfolio project 12 (Semantic cache + prompt-compression cost-optimizer), Scaler LLMOps.

## Results

**Held-out eval set**: 40 queries, never used for tuning. Every answer graded blind by Sonnet 5.5 against a reference answer, its required facts and a must-not-claim list. Source: `results/e2e_evalset__cost_vs_quality_summary.csv`.

| Pipeline | Cost / query | vs baseline | Correct | Partial | Wrong |
|---|---|---|---|---|---|
| Baseline: Sonnet 5.5, full context, no cache | $0.0139 | 100% | 97.5% | 2.5% | 0% |
| **CostGuard, window 800** (default) | **$0.0034** | **24.5%** | 87.5% | 12.5% | 0% |
| CostGuard, window 1,200 | $0.0042 | 30.2% | 90.0% | 10.0% | 0% |
| CostGuard, window 800 + LLMLingua-2 0.7 | $0.0033 | 23.7% | 95.0% | 5.0% | 0% |
| CostGuard, window 1,200 + LLMLingua-2 0.7 | $0.0042 | 29.8% | 95.0% | 2.5% | 2.5% |

**Cost cut by 75.5% with no wrong answers** (default pipeline). Correct answers fell 10 points: answers that are right but miss a detail the reference includes. On 40 queries one answer is 2.5 points, so the gaps between the variants are within noise; the cost order is not.

On the **150-query tuning set** the default costs **17.4%** of baseline with 97.3% correct vs 98.0% (`results/e2e__cost_vs_quality_summary.csv`). That set has more repeats (68 of 150 served from cache vs 11 of 40), so its savings are higher. Savings depend on how often traffic repeats. On cache misses alone, windowing cuts input tokens by 81% and cost by 67% (output tokens don't shrink).

### Per stage

| Stage | Result | Evidence |
|---|---|---|
| Exact cache | 36/36 repeats hit, 0 false hits, 21.7% of context tokens avoided | `results/exact__summary.json` |
| Semantic cache, bge-small, cosine ≥ 0.90 | Wrong hits on 49 traps: cosine only **43** → + context check **21** → + numbers/negation guards **10** → + Haiku 4.5 judge **1**. Paraphrases caught: 33 → 33 → 31 → 31 of 36 | `results/semantic_mode_comparison__bge-small_t0.90.csv`, threshold sweeps for two embedding models |
| Context window, hybrid @ 800 | 96.1% of queries keep every required fact at ~823 tokens (85.5% fewer), on held-out clusters across 5 seeds | `results/window_split_eval__*.csv` |
| Prompt compression | Sentence pruning: no gain over picking the window budget. LLMLingua-2 at 0.7: drops facts in string checks (51.6% exact), yet end-to-end answers held up; only 3% cheaper | `results/compression_*.csv`, e2e tables above |
| Model routing (Haiku 5.5 / Sonnet 5.5) | Logistic regression on query embedding + features, threshold chosen so no complex query goes to the small model (out-of-fold). Too cautious: routes 0 of 29 eval-set misses to Haiku | `results/routing__*.csv` |

### Latency and throughput

The service with the LLM mocked: caches, embedding, windowing, routing and Redis, on one laptop process (8-core CPU). `results/loadtest__summary.csv`.

| Traffic | Concurrency | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| Eval trace mix (mostly cache hits) | 8 | 821 req/s | 8.3 ms | 19.5 ms | 34.5 ms | 0 |
| Every request a cold miss | 8 | 332 req/s | 23.9 ms | 31.7 ms | 36.9 ms | 0 |
| Every request a cold miss | 32 | 340 req/s | 93.2 ms | 127.1 ms | 136.5 ms | 0 |

CostGuard adds about 25 ms to a cache miss and answers a cache hit in under 10 ms. End-to-end latency of a live miss is dominated by the model call. Per-stage live latency (p50/p95) is on the Grafana dashboard.

**Total API spend** for every experiment and evaluation so far: about $5.50 (1,046 unique calls, recorded in `results/e2e__api_memo.json`).

## Architecture

```
request (tenant, query, case, policy docs)
   │
   ├─ 1. exact cache ── Redis key: sha256(config version, query, case, sorted doc hashes) ─── hit → answer
   │
   ├─ 2. semantic cache ── Redis vector KNN (bge-small, cosine ≥ 0.90, tenant filter), then per candidate:
   │        context check   the cached answer's docs and case are all present in this request
   │        guards          same numbers/codes; same negation count in the words that differ
   │        LLM judge       Haiku 4.5: "would one correct answer serve both?"            ─── hit → answer
   │
   ├─ 3. context window ── hybrid: top-scoring doc whole minus repeated boilerplate, then the best
   │                        sections of other docs up to the token budget (800)
   ├─ 4. router ───────── P(needs large model) from embedding + features → Haiku 5.5 or Sonnet 5.5
   ├─ 5. LLM call ─────── answer stored in both caches with depends_on = docs the window kept + case
   │
   └─ metrics → Prometheus → Grafana        shadow mode: also runs the baseline and has Sonnet 5.5
                                            compare the two answers (live quality delta)
```

| Component | Where |
|---|---|
| FastAPI service, demo page, `/metrics` | `service/` |
| Redis Stack (exact keys + RediSearch vector index), Prometheus, Grafana (provisioned dashboard) | `docker-compose.yml`, `infra/` |
| Cache, guards, judge | `cache/` |
| Window management | `window/` |
| Compression experiments | `prompt_compression/` |
| Router | `routing/`, trained at startup by `service/router_model.py` |
| End-to-end evaluation | `e2e/` |
| Quality gates | `gates/`, `tests/`, `.github/workflows/ci.yml` |
| Trace replay, load test | `replay/`, `loadtest/` |
| Config and prompts | `config/costguard.toml`, `config/prompts/*.txt` |

### Design decisions

- **The cache key is content, not ids.** Production requests don't have stable ids, and an edited policy must not serve a stale answer.
- **No model in the cache key.** The cache runs before routing, so a cached answer is reused whichever model wrote it. The **config version** (answer prompt + model set) is in the key, so changing the prompt starts a fresh cache.
- **The semantic cache needs more than cosine.** Traps like "delivered 10 days ago" vs "40 days ago" score 0.98. Each added check removes one class of wrong hit (table above). The judge runs only after the free checks pass, so it is called on few requests.
- **Window before compression.** Picking the right sections removed 85% of tokens with almost no fact loss. Compression on top bought 3% more at measurable fact loss in string checks, so it is off by default.
- **The router errs toward the large model.** A complex query on the small model risks a wrong answer; the reverse only costs money.

## Evaluation

| Set | Size | Purpose |
|---|---|---|
| `demo_data/` | 150 queries, 44 cases | Tuning: thresholds, budgets, router. Labels: expected cache outcome, gold docs, gold facts, must-not-claim, difficulty |
| `eval_data/` | 40 queries, 25 cases | **Held out**, used only for final measurement. 20 cold (13 on topics the tuning set barely covers), 5 exact repeats, 6 paraphrases, 9 traps (number, negation, intent, state, case, date boundary, source swap, duties vs tax) |

`eval_data/build_eval.py` builds the set and checks it: every gold fact appears verbatim in its gold policy, every gold doc is attached, gold docs are not always first in the bundle, and no query copies the tuning set.

**Scoring:**
1. **LLM-as-judge.** Sonnet 5.5 grades each answer blind against the reference answer, gold facts and must-not-claim list: correct / partial / wrong + a 1–5 score (`config/prompts/eval_grader_system.txt`).
2. **Programmatic checks.** Cache outcomes against labels (wrong hits, missed repeats); required facts present in the windowed context.
3. **Cost** from each response's real token usage at list prices.

Cache hits return the real answer generated earlier for the query they were served from, and are graded like any other answer.

## Quality gates and CI

`.github/workflows/ci.yml` runs on every pull request.

| Job | What it checks | API calls |
|---|---|---|
| `gates` | Unit tests; exact-cache replay (0 false hits, 36/36 repeats); full semantic chain on 150 queries with memoized judge verdicts (≤ 1 wrong hit, ≥ 31 paraphrases); window keeps all facts in ≥ 95% of queries | None |
| `llm-gate` | e2e on the held-out set: default pipeline wrong ≤ 2.5%, correct within 12.5 points of baseline, cost ≤ 35% of baseline | None if prompts are unchanged (memo replay). Changed prompts run live with the `ANTHROPIC_API_KEY` secret under a $1.50 cap |

Bounds live in `config/costguard.toml` (`[gates.*]`). For example, cutting the window budget to 400 drops fact coverage to 85.6% and fails the `gates` job.

## Observability

Grafana at http://localhost:3000 (dashboard *LLM CostGuard*), filterable by tenant:

- **Last request:** layer, model, cost, baseline cost, % saved, tokens sent vs baseline, latency, quality preserved
- **Totals:** requests, actual vs baseline $, % saved, share served from cache, share of shadow-checked answers rated equivalent or better
- **Over time:** cumulative actual vs baseline cost and tokens, requests by layer, p50/p95 latency per stage
- **Breakdown:** layers, models, quality verdicts, semantic-cache rejection reasons, overhead costs
- **Offline evaluation** results

**Before/after on a fixed trace.** `python -m replay.trace` sends the held-out set through the running service in replay mode: answers, baseline and grades come from the memoized evaluation, so it costs nothing while the service does the real cache, window and routing work. Result: 40/40 requests take the same path as the offline run, **$0.135 vs $0.558 baseline (75.8% saved)**, quality equivalent on 36, worse on 4. Shown under tenant `trace_eval_data`.

## Setup

Requires Python 3.14, Docker, and an Anthropic API key for live mode.

```bash
python3.14 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
echo "ANTHROPIC_API_KEY=..." > .env && chmod 600 .env

./run_demo.sh            # Redis Stack, Prometheus, Grafana in Docker; the service on the host
# demo page   http://localhost:8000      Grafana http://localhost:3000
# Prometheus  http://localhost:9090      RedisInsight http://localhost:8001
```

The demo page has three LLM modes: **live** (real calls; shadow mode adds a baseline + judge, about 2× cost), **replay** (saved evaluation answers, free; only for dataset queries with their exact attachments), and **mock** (no API).

| Task | Command | Cost |
|---|---|---|
| Unit tests + deterministic gates | `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q tests` | free |
| Gate report | `.venv/bin/python -m gates.deterministic` | free |
| End-to-end eval (memoized) | `.venv/bin/python -m e2e.run --data eval_data` | free if memoized, else ~$1.50 |
| LLM gate | `.venv/bin/python -m gates.llm results/e2e_evalset__cost_vs_quality_summary.csv` | free |
| Replay the trace into Grafana | `.venv/bin/python -m replay.trace` | free |
| Load test | see `loadtest/run.py` | free (mock) |

`COSTGUARD_OFFLINE=1` makes any unmemoized call an error; `COSTGUARD_MAX_NEW_COST_USD` caps new spend per run.

## Known limitations

- **One semantic-cache false hit on the held-out set.** "Cheaper **at another store**" was served the answer for "cheaper **on Northstar**": cosine 0.947, both guards passed, and Haiku 4.5 judged them the same.
- **Multi-doc questions lose the second policy.** "Pickup + return" keeps the returns doc but drops the pickup rule that changes when the window starts. A cross-encoder reranker, or keeping the top two docs when their scores are close, is the next step.
- **The router barely downshifts.** All savings come from caching and windowing.
- **The eval set was written by the project team**, so it is likely easier for the pipeline than real customer messages. Each figure is one grading run; on 40 queries one answer is 2.5 points.
- **Not deployed publicly yet.** The service needs the embedding model (~130 MB) and Redis Stack; a public demo also needs a spend cap.

## Resume line

Built an LLM cost-optimization layer (exact + guarded semantic cache, context-window management, prompt compression, model routing) with eval-gated CI and a Prometheus/Grafana savings dashboard; cut LLM spend by 75% on a held-out eval with no wrong answers versus a full-context Sonnet baseline.
