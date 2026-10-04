# Design decisions

The trade-offs behind CostGuard, written in the rubric's form: **"We chose X over Y because Z. The cost is …"**. Each one names the constraint that forced it. Numbers come from `eval/results/*.json` and `configs/prices.yaml`, checked 2026-10-03.

Related documents:

- [ARCHITECTURE.md](ARCHITECTURE.md): system diagram, latency budget, requirements.
- [RUNBOOK.md](RUNBOOK.md): how the decisions are operated.
- Each stage's own write-up: [semantic_cache.md](components/semantic_cache.md), [context_and_compression.md](components/context_and_compression.md), [router_and_gate.md](components/router_and_gate.md).

**Key insight.** Cost is set by *which tokens are never sent*, and quality risk sits in false cache hits and downshifts. So the levers run in order of quality risk: exact cache → semantic cache → trim context → compress → downshift last.

---

## 1. Trade-offs

### Caching

**1. Exact cache in front of the semantic cache.** We chose an exact-match tier before the semantic tier over a semantic-only cache.

- **Why:** an exact hit is the same normalised question in the same partition, so it carries **zero false-hit risk**. It costs a ~1 µs dict lookup instead of a 2–5 ms embedding.
- **Cost:** one extra lookup per miss, and duplicate storage of popular answers.

**2. τ chosen from a per-request false-hit budget.** We chose the threshold from a curve of false hits over **all requests**, with a budget per mode (0.5% / 1% / 3% for quality / balanced / economy, see `eval/results/threshold_sweep.json`). We rejected a single eyeballed or hit-rate-maximising τ.

- **Why:** a false hit is a confidently wrong answer with no error signal. Per-hit precision flatters aggressive thresholds. In AWS's ElastiCache benchmark (Titan Text Embeddings V2 on SemBenchmarkLmArena), τ = 0.80 has 91.8% per-hit "accuracy", but with an 87.6% hit rate that is a wrong answer for about 7.2% of all requests, 1 in 14.
- **Cost:** a lower hit rate, and τ must be recalibrated whenever the embedding model changes.

**3. Deterministic guards on near-hits.** We chose number/ID, negation and entity guards that veto a near-hit over raising τ for everyone.

- **Why:** the dangerous pairs ("cancel order #4821" vs "don't cancel #4822") score about 0.9, as high as true paraphrases. A guard that runs only on candidates above τ costs microseconds and keeps the paraphrase hits.
- **Cost:** a hand-built lexicon. It is a project heuristic with no published benchmark, so it is evaluated on our own trap pairs.

**4. Local embeddings.** We chose fastembed `bge-small-en-v1.5` (ONNX, CPU) over an embedding API.

- **Why:** $0 per call, no vendor dependency or network hop on the hit path, and τ is calibrated on exactly this model.
- **Cost:** about 70 MB baked into the image, CPU load that grows with traffic, and a full re-calibration plus re-embedding (blue-green index) if the model changes.

### Context and compression

**5. Compress only volatile retrieved context.** We chose to compress only the retrieved-context block, never the system prompt or the question, over compressing the whole prompt.

- **Why:** provider prefix caching needs a byte-identical prefix. The Anthropic adapter puts `cache_control` on the system prompt, so cache reads bill at 10% of the input price. Rewriting the prefix would forfeit that discount on every call.
- **Cost:** short prompts gain little. Compression only pays on long RAG context (`compression_min_tokens`).

**6. Rerank and drop whole documents.** We chose a cross-encoder rerank, then dynamic-k, then dropping whole documents to a token budget, over mid-chunk truncation or token-level dropping.

- **Why:** half-chunks strand claims from their qualifiers ("returns within 30 days *unless* opened"). Published comparisons favour selecting whole passages over token pruning (Jha et al. 2024: extractive selection "often outperforms" it, up to 10× compression).
- **Cost:** the cross-encoder is our slowest stage, p50 25 / p99 104 ms under 50-user load (n = 753, `eval/results/loadtest.json`). It pushes RAG misses just past the 100 ms miss-path budget.

**7. Heuristic compressor live, LLMLingua-2 offline.** We chose a query-aware extractive compressor in the serving path over LLMLingua-2 in the serving path.

- **Why:** LLMLingua-2 needs torch plus a 0.7–2.2 GB model. Peak RSS measured 1.6 GB (mBERT) and 4.1 GB (xlm-roberta-large) (`compression_eval.json`), which cannot run on Render free's 512 MB. Our heuristic protects numbers and runs at about 1 ms p50 / 7.5 ms p99 in the pipeline (`loadtest.json`), 4 ms p50 in the offline eval.
- **Cost:** a lower compression ratio at equal retention. LLMLingua-2 results are reported from the offline eval only (`eval/results/compression_eval.json`).

### Routing and models

**8. Downshift behind a per-category eval gate.** We chose to send a category to the cheap tier only when the **lower bound** of its paired 95% CI on quality clears the margin (`configs/router_gate.json`). We rejected a cheap model everywhere and an opaque learned router.

- **Why:** downshifting is the highest quality-risk lever. A gate per category is auditable, and it rolls back by editing one file, which the router re-reads on mtime change.
- **Cost:** a small category (n ≈ 30–50) can only be gated coarsely, so some safe traffic stays on the strong tier.

**9. Sonnet 5.5 as strong, Haiku 4.5 as cheap.** We chose this Anthropic pair over a wider-gap pair such as GPT-5.4-mini → GPT-5-nano (about 12×).

- **Why:** one provider and SDK for both tiers, the same prompt-caching semantics, and we expect Haiku 4.5 to pass the gate on simple categories. That is an expectation, not a measurement: the gate has not been run on the Anthropic pair.
- **Limitation:** **the price gap is only 2×**: $2 / $10 vs $1 / $5 per 1M input / output tokens. A downshift saves at most 50% on a routed request. A cache hit saves 100%, including output tokens, which cost 5× input.
- **Consequence:** this is the honest reason **caching matters more than routing here**. The waterfall should show routing as a small lever, and we say so before an examiner does.
- **Measured runs:** the headline A/B and the router gate run on the local MLX stand-in (Qwen2.5-7B → 1.5B), billed at GPT-5.4-mini → GPT-5-nano list prices, a ~12× gap. Routing will look bigger in that waterfall than it would on the Anthropic pair, so quote it with that caveat.

**10. Native Anthropic SDK.** We chose the native `anthropic` SDK adapter over LiteLLM for the real backend.

- **Exact usage fields:** the SDK returns `cache_read_input_tokens` and `cache_creation_input_tokens` separately. PriceBook bills them at the cached price and the cache-write price. LiteLLM re-maps them into the OpenAI usage shape (`prompt_tokens_details.cached_tokens` plus an extra `cache_creation_input_tokens`), one more translation layer between the provider's numbers and the bill.
- **Fewer layers:** one fewer dependency between the request and the bill, and failures surface as the provider's own errors.
- **Explicit endpoint:** the adapter always calls `https://api.anthropic.com` unless `COSTGUARD_ANTHROPIC_BASE_URL` is set. It deliberately ignores an inherited `ANTHROPIC_BASE_URL`, so the key can't be silently sent to some other tool's proxy.
- **Cost:** a second code path to maintain. LiteLLM is kept for the other backends and as an off-the-shelf comparison arm.

### Reliability and operations

**11. Fail open.** We chose fail-open passthrough over failing the request when a stage errors.

- **Why:** a cost layer must never lower availability. Each stage's exception goes into `stage_errors` and `costguard_stage_errors_total`, and the request continues un-optimised. A cheap-tier error retries strong once.
- **Cost:** a broken stage silently costs money instead of paging. That is why the stage fail-open rate has its own alert (RUNBOOK §6).

**12. Asynchronous telemetry.** We chose async telemetry over inline scoring and tracing:

- Prometheus is pulled.
- Langfuse goes through a bounded queue and a background thread.
- The LLM judge runs offline on samples.

- **Why:** monitoring must add no latency. The in-process hooks cost 19 µs (metrics) and 3 µs (drift) per request in a micro-benchmark (not yet saved under `eval/results`). Langfuse init and network I/O happen off-thread and drop events rather than block.
- **Cost:** quality signals arrive minutes or hours late, not per request.

**13. Our own SQLite log as the source of truth.** We chose a local SQLite TraceRecord log over Langfuse or a SaaS as the system of record.

- **Why:** no quota (Langfuse Hobby is 50k units/month, about 8k requests at ~6 units per request), it is reproducible, and the dashboard, `/v1/stats` and README tables all read the same rows.
- **Cost:** it is single-host. The core logger writes synchronously (~0.5 ms), and a multi-replica deployment would need Postgres or ClickHouse.

**14. Cassette replay in CI.** We chose to replay recorded upstream responses (cassettes) in CI over calling the live API.

- **Why:** no secrets in CI, $0, and deterministic: the same PR always gets the same verdict. The τ = 0.6 PR fails for the right reason, not because of judge noise.
- **Cost:** a prompt change that misses the cassette needs a recorded re-run before CI can judge it.

**15. Drift from TraceRecord features.** We chose PSI over features already in the TraceRecord over embedding-level drift in the hot path. The features are input-token length, category mix, top-1 cache similarity, output length and route mix.

- **Why:** an O(1) append per request and lazy recomputation, using the course's own W3S2 maths and bands (< 0.10 / 0.10–0.25 / > 0.25).
- **Cost:** a new topic hidden behind an existing category hint is caught only indirectly, through the similarity distribution and the hit rate. Weekly embedding clusters with JS divergence (W7S2) remain an offline job.

**16. Render free with models baked in.** We chose a Docker image on Render free, with models downloaded at build time, over Fly.io or Hugging Face Docker Spaces.

- **Why:** Fly.io and HF Docker Spaces are no longer free as of October 2026. Baking the ONNX models means a wake-up needs no Hugging Face download.
- **Cost:** a 15-minute idle sleep and about 1 minute to wake (we pre-warm before demos), no persistent disk, a single instance. Load tests therefore run locally.

**17. Cost per correct answer as the headline.** We chose cost per correct answer, reported next to savings % with a CI, over cost per call or tokens saved.

- **Why:** a downshift that produces a wrong answer, or triggers a retry, can lower cost per call while raising cost per resolved request.
- **Cost:** it needs the judged eval set for every arm.

### Alternatives considered and rejected

- **Self-hosting a quantised model as the cheap tier.** At about $2.3k/month baseline spend (ARCHITECTURE §4), we are well below the ~$10k/month break-even, and it adds hidden ops cost.
- **A global semantic cache without tenant namespaces.** It would leak one tenant's answers to another. The tenant is part of every cache partition, and the tenant comes from the caller key, not the body.
- **TTL-only invalidation.** It serves stale answers after the knowledge base changes. A `kb_version` bump creates a new partition instantly; TTL is the backstop.
- **Building entirely on the LiteLLM proxy.** It hides the four optimisations the project has to demonstrate and measure. We use its ideas, such as the `x-…-cost` header convention, not its pipeline.

---

## 2. Failure modes

| Failure | Detection | Mitigation / recovery |
|---|---|---|
| **False semantic hit** (negation, swapped order number or product) | Offline: trap pairs in the CI gate, false-hit rate per request. Online: sampled judge on semantic hits; dashboard "least similar accepted hits" | Guards veto near-hits. Raise τ for the mode. Evict the entry. Add the pair to the trap set |
| **Stale cached answer** after a policy or KB change | Judge failures on hits; age of cached answers; user reports | Bump `kb_version` (new partition) or restart (in-memory tier). TTL 7 days as backstop |
| **Embedding model changes silently** | PSI on `cache_similarity` > 0.25; hit-rate drop | Version the cache by model. Re-calibrate τ. Blue-green re-embed |
| **Stage crashes** (reranker OOM, Qdrant down) | `costguard_stage_errors_total{stage}` > 1% of requests | Fail open: the request continues un-optimised. Fix or disable the lever in `policy.yaml` |
| **Provider outage or 529 overload** | `costguard_request_errors_total` / requests > 0.5%; `upstream_latency` p99 +20% | The SDK retries 4× with backoff. A cheap-tier failure retries strong. Cached answers keep serving |
| **Downshift regresses a category** | Per-category CI gate; canary judge score per category | Set the category to `allow: false` in `router_gate.json` (hot-reloaded), or route policy `gated` → strong |
| **Compression drops a key fact** (number, negation) | `evidence_retention` in `compression_eval.json`; CI gate slice | Number-bearing units are protected. Raise `compression_rate` or `compression_min_tokens` |
| **Prefix caching broken** by a volatile token at the prompt head | `costguard_provider_cache_tokens_total{kind="read"}` / sent tokens drops | Keep the system prompt static. Only context is rewritten |
| **Cost spike** (long pasted documents, a runaway caller) | Any request > 5× median cost; daily spend +10–20% vs baseline; PSI on `input_tokens` | The context budget caps RAG tokens. Find the caller by tenant in the log |
| **Cross-tenant leak** | Unit test on partitions; tenant in every key | The tenant comes from the caller key. Body overrides are ignored when keys are configured |
| **Observability backend down** (Langfuse, Prometheus) | `costguard_hook_dropped_total`, `costguard_hook_errors_total` | A bounded queue drops events and never blocks; the SQLite log is unaffected |
| **Cold cache after deploy or wake** | Hit rate low for the first N requests; latency p50 high | The canary ramp absorbs it. Pre-warm by replaying top queries. Models are baked in |
| **Bad policy config** (e.g. τ = 0.6) | CI eval gate (required check); `config_hash` on every row | Merge blocked. If deployed: revert `policy.yaml` (RUNBOOK §4) |

---

## 3. What breaks first at 10× scale

At 10× the design point, that is 200k requests/day and ~12 req/s at peak, the order is as follows.

1. **CPU-bound miss-path stages.** The cross-encoder rerank, already p99 104 ms under 50 concurrent users, and the ONNX query embedder. They degrade p50 and p95, not just p99, because every miss pays them.
   - **Fixes:**
     - batch embedding and rerank calls;
     - cap the rerank depth;
     - cache query embeddings, since repeats already skip them;
     - move rerank to a sidecar, or switch it off in `economy`;
     - scale replicas horizontally.
2. **One worker's threadpool.** The sync endpoint uses 40 threads, so a 300 ms upstream caps pure-miss traffic near 133 req/s per worker. With real 2–5 s provider latency, that cap drops to about 8–20 req/s.
   - **Fix:** more uvicorn workers or replicas, or an async provider path.
3. **In-process caches stop being shared.** Each replica has its own exact and semantic cache, so the hit rate falls as replicas are added.
   - **Fix:** Redis for the exact tier and Qdrant (already supported) for the semantic tier. Tenant-namespaced keys carry over unchanged.
4. **Provider rate limits** (Anthropic tokens/minute). They show up as 429/529s and a rising error rate.
   - **Fixes:**
     - per-tenant rate limiting;
     - request queueing with backoff;
     - cached answers as graceful degradation;
     - higher caching in `economy` mode during incidents.
5. **The SQLite log.** A single writer that commits per row becomes a contention point.
   - **Fix:** a batched async writer, then Postgres or ClickHouse.
6. **Langfuse free quota.** 50k units/month is about 8k requests.
   - **Fix:** `COSTGUARD_LANGFUSE_SAMPLE` at 5–10%, or self-hosted Langfuse or Phoenix.

The course's five scaling levers apply in that order: autoscaling, caching, batching, rate limiting, fallbacks (W7S2).
