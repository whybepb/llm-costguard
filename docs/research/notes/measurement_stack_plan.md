# Measuring, observing and demonstrating "LLM CostGuard": A/B harness, savings dashboard, free-tier stack and 5-day plan (checked 2026-10-03)

All free-tier and pricing facts below were fetched on **2026-10-03** unless noted. Free tiers change often, so re-check the pricing page before quoting a number in the README. Model names and prices are reported exactly as the provider pages showed them on that date.

---

## Q1. Architecture: what does a "drop-in" cost layer look like, and is building on the LiteLLM proxy a smart shortcut?

### Takeaway
The most convincing "drop-in" demo is an **OpenAI-compatible HTTP proxy**: the client only changes `base_url`, and the proxy runs an explicit pipeline (cache, then compress/rerank/truncate, then route/downshift, then upstream call, then log). LiteLLM already provides most of the plumbing: per-request cost headers, spend logs, Redis/Qdrant semantic caches, per-request cache controls and Prometheus metrics. Building *entirely* on the LiteLLM proxy would therefore hide the four optimisations the examiner wants to see. The best trade-off is to **own a thin FastAPI proxy whose pipeline stages are your code**, and to use the **LiteLLM Python SDK** only for provider abstraction and its price map. LiteLLM's built-in semantic cache can then serve as a comparison baseline rather than as the product.

### Cited Findings
- The LiteLLM proxy returns a per-request cost header, `x-litellm-response-cost`. It persists every transaction to a `LiteLLM_SpendLogs` table, which needs a database (typically PostgreSQL). Logged fields include key, user, team, tags, end-user, model, spend, tokens and metadata. — [LiteLLM cost tracking docs](https://docs.litellm.ai/docs/proxy/cost_tracking)
- LiteLLM can split spend by API key, user, team, custom tags (`metadata.tags`) and model. Some features are Enterprise-only: spend reports via `/global/spend/report`, custom spend-log metadata and RBAC on `/spend` endpoints. — [LiteLLM cost tracking docs](https://docs.litellm.ai/docs/proxy/cost_tracking)
- The LiteLLM proxy supports these cache backends: `redis`, `redis-semantic`, `valkey-semantic`, `qdrant-semantic`, `s3`, `gcs`, `local` (in-memory, dev only, single worker) and `disk`. Cached responses carry an `x-litellm-cache-key` header. The docs state: "Redis is the right default for anything past a single worker." — [LiteLLM proxy caching](https://docs.litellm.ai/docs/proxy/caching)
- Semantic-cache configuration:
  - `similarity_threshold` (0–1), `redis_semantic_cache_embedding_model` (default `text-embedding-ada-002`) and `ttl`.
  - The Qdrant variant adds `qdrant_collection_name`, `qdrant_quantization_config` (`binary`/`product`/`scalar`) and `qdrant_semantic_cache_vector_size`.
  - Per-request controls are passed as `cache={...}`: `no-cache`, `no-store`, `ttl` and `s-maxage`.
  - Cache hits appear to callbacks as `kwargs["cache_hit"]`.
  - Source: [LiteLLM all caches](https://docs.litellm.ai/docs/caching/all_caches)
- LiteLLM's Prometheus integration is open-source. Enable it with `litellm_settings: callbacks: ["prometheus"]`; metrics are served at `/metrics` (API-key auth by default). Metric names: `litellm_spend_metric`, `litellm_input_tokens_metric`, `litellm_output_tokens_metric`, `litellm_request_total_latency_metric`, `litellm_llm_api_latency_metric`, `litellm_cache_hits_metric` and `litellm_cache_misses_metric`. — [LiteLLM Prometheus docs](https://docs.litellm.ai/docs/proxy/prometheus)
- Existing routing research shows downshift results are reported as **"cost reduction at X% of strong-model quality"**:
  - RouteLLM cut cost by over 85% on MT Bench, 45% on MMLU and 35% on GSM8K versus GPT-4-only, while reaching 95% of GPT-4 performance.
  - The strong model was GPT-4 Turbo and the weak model was Mixtral 8x7B.
  - Four routers were tested: similarity-weighted ranking, matrix factorization, a BERT classifier and a causal-LLM classifier.
  - Sources: [LMSYS RouteLLM blog](https://lmsys.org/blog/2024-07-01-routellm/); [RouteLLM paper arXiv 2406.18665](https://arxiv.org/abs/2406.18665)
- LLMLingua-2 is the standard compression component:
  - It is "3x-6x faster than existing prompt compression methods" and accelerates end-to-end latency by 1.6x–2.9x at 2x–5x compression. — [LLMLingua-2 arXiv 2403.12968](https://arxiv.org/abs/2403.12968)
  - The default checkpoint is XLM-RoBERTa-large (0.6B params, F32). A smaller `llmlingua-2-bert-base-multilingual-cased-meetingbank` variant also exists.
  - Usage: `compress_prompt_llmlingua2(prompt, rate=0.6, force_tokens=[...])`. — [HF model card](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank)

### Inferences
- **Options compared:**
  - **(a) OpenAI-compatible proxy (FastAPI).** Language-agnostic. The "drop-in" claim is provable live by changing one line (`base_url`) in an unmodified OpenAI-SDK client. Easiest to load-test with k6/Locust, and gives a public URL. **Recommended.**
  - **(b) SDK wrapper/decorator.** Fastest to code (no network hop, no deployment). Cannot be load-tested as a service and is Python-only, which makes the "drop-in" and "live URL" deliverables weaker.
  - **(c) Middleware inside an app.** Ties the work to one app, so it is the weakest "drop-in" story.
  - **Practical hybrid:** build the core as a library (`costguard.pipeline.run(request) -> response, log_record`). Expose it through FastAPI (`POST /v1/chat/completions`). Have the A/B harness call the library directly for speed. One codebase serves all three uses.
- **LiteLLM as a shortcut.**
  - Use the **LiteLLM SDK** (`litellm.acompletion`, built-in model price map / `completion_cost`) to avoid writing provider adapters and price tables.
  - Do **not** make LiteLLM proxy's `redis-semantic` cache *the* product. Examiners want to see your threshold logic, similarity scores, false-hit handling, compression and routing decisions in your own code and logs.
  - A strong move is to include "LiteLLM's built-in redis-semantic cache at threshold 0.8" as a **comparison arm**. That shows you know the off-the-shelf option and measured against it.
  - Copy LiteLLM's header convention for your own transparency headers:
    - `x-costguard-cache: hit|miss|bypass`
    - `x-costguard-similarity`
    - `x-costguard-route: <model>`
    - `x-costguard-cost-usd`
    - `x-costguard-baseline-cost-usd`
    - `x-costguard-saved-usd`
    - `x-costguard-overhead-ms`
- **Pipeline order** (each stage behind a config flag so ablations are one CLI switch):
  1. Exact-match cache.
  2. Semantic cache. Embed, do a top-1 lookup, then accept only if similarity ≥ τ. Optionally add a cheap verifier for borderline scores.
  3. Context reranking/truncation for RAG-style requests: keep the top-k chunks within a token budget.
  4. LLMLingua-2 compression above N input tokens. Only compress long prompts; short prompts gain nothing and add latency.
  5. Router/downshift: a heuristic or classifier sends "easy" requests to the cheap model.
  6. Upstream call.
  7. Write to the cache (`no-store` for errors and low-confidence answers).
  8. Emit one structured log record plus an OTel span.
- **Per-request log record:** this is the single source of truth for both the dashboard and the README. Write it to SQLite/Parquet locally and mirror it to the observability tool. Fields:
  - IDs and arm: `request_id`, `trace_pos`, `arm`, `ts`, `user_id`
  - Routing: `model_requested`, `model_served`, `route_reason`
  - Cache: `cache_status`, `cache_sim`, `cache_hit_source_id`
  - Tokens: `in_tokens_original`, `in_tokens_sent`, `cached_in_tokens`, `out_tokens`
  - Cost: `cost_usd`, `baseline_cost_usd`
  - Latency: `latency_total_ms`, `latency_upstream_ms`, `t_cache_ms`, `t_compress_ms`, `t_route_ms`
  - Outcome: `error`, `quality_score`
- **Counterfactual "baseline cost" for the live dashboard** (where there is no paired baseline call): estimate it as `in_tokens_original × baseline_in_price + out_tokens × baseline_out_price`, and label it as an estimate. The A/B on the fixed trace gives the *measured* savings.

### Gaps
- I did not verify whether LiteLLM's open-source proxy UI shows cache-hit and latency percentile dashboards without an Enterprise licence. Only the Prometheus metrics were confirmed open-source.
- I found no published comparison of proxy overhead (ms) for LiteLLM versus a hand-written FastAPI proxy. Measure it yourselves with a mock upstream (see Q5).

---

## Q2. Observability tools and free tiers: what is quickest for per-request cost, latency, cache-hit/route logs and a savings dashboard?

### Takeaway
The quickest credible stack has three parts:
1. Your own structured per-request log (SQLite/Parquet), which is the source of truth for the A/B numbers.
2. **Langfuse Cloud Hobby**, which provides traces, cost per generation and a polished UI for the demo.
3. A **Streamlit dashboard** reading the log, for the savings story: cumulative $ saved, hit rate, cost/quality Pareto and p50/p99.

Prometheus with Grafana Cloud Free is optional extra credit. Helicone's free tier is small (10k requests, 7-day retention). The binding constraint is Langfuse's 50k units/month: threshold sweeps must be logged locally, not to Langfuse.

### Cited Findings
- **Langfuse Cloud Hobby (free):**
  - 50k units/month, 30 days data access, 2 users, community support.
  - Ingestion 1,000 req/min, general API 30 req/min, Metrics API v2 100 req/day, 1 annotation queue, 2 alerts.
  - Core costs $29/month for 100k units, 90 days and unlimited users; overage is $8 per 100k units.
  - Source: [Langfuse pricing](https://langfuse.com/pricing)
- A Langfuse unit = count of traces + count of observations + count of scores. Self-hosted Langfuse is MIT-licensed and free. — [Langfuse billable units](https://langfuse.com/docs/administration/billable-units)
- How Langfuse computes cost:
  - Cost is either ingested or inferred, and ingested values take priority.
  - Inference matches the generation's `model` against regex-based model definitions; OpenAI, Anthropic and Google models ship by default. Custom prices can be added via the UI or `POST /api/public/models`.
  - Usage keys are mutually exclusive buckets: `input` (excluding cached), `output`, `cache_read_input_tokens`, etc.
  - Dashboards show cost by model, cost over time, and top users/use cases by cost.
  - Source: [Langfuse token & cost tracking](https://langfuse.com/docs/observability/features/token-and-cost-tracking)
- **Arize:**
  - Phoenix (OSS) is under Elastic License 2.0. "Self-hosting on your own infrastructure or in your cloud account is free and fully permitted", with "no feature gates". — [Phoenix license docs](https://arize.com/docs/phoenix/self-hosting/license)
  - Arize AX Free: 25k spans/month, 1 GB/month, 15-day retention, unlimited users. AX Pro costs $50/month. — [Arize pricing](https://arize.com/pricing/)
- **Helicone Hobby (free):** 10,000 requests/month, 7-day retention, 1 seat, 1 GB storage, API limits of 10 calls/min. Caching and Gateway appear in the feature table, but per-tier availability isn't spelled out. Pro costs $79/month. — [Helicone pricing](https://www.helicone.ai/pricing)
- **Grafana Cloud Free:**
  - 10k active metric series, 50 GB logs, 50 GB traces and 50 GB profiles per month, all with 14-day retention.
  - 3 active visualization users.
  - k6 performance testing: 500 virtual-user hours/month.
  - "Agent Observability" limited to 30k generations/month.
  - Source: [Grafana pricing](https://grafana.com/pricing/)
- **OpenTelemetry GenAI semantic conventions:**
  - They have moved to a dedicated repository. — [OTel spec page](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/); [semantic-conventions-genai repo](https://github.com/open-telemetry/semantic-conventions-genai)
  - `gen_ai.usage.cache_read.input_tokens` is defined as "number of input tokens served from a provider-managed cache". It SHOULD also be counted in `gen_ai.usage.input_tokens`, which should include all input token types. — [OTel Gen AI attribute registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/) (via search snippet)
- **Streamlit Community Cloud:** 0.078–2 CPU cores, 690 MB–2.7 GB memory, up to 50 GB storage. "All apps without traffic for 12 hours go to sleep"; any viewer can wake one. — [Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- **LiteLLM Prometheus metrics** (spend, tokens, total and upstream latency, cache hits/misses) are open-source. — [LiteLLM Prometheus](https://docs.litellm.ai/docs/proxy/prometheus)

### Inferences
- **Langfuse unit budget arithmetic:**
  - Suppose one proxied request = 1 trace + 4 observations (cache lookup, compress, route, generation) + 1 score ≈ **6 units**.
  - Then 50k units ≈ **8k requests/month**.
  - A 2,000-request trace × 2 arms ≈ 24k units, so one full A/B fits in Langfuse. A threshold sweep (say 5 thresholds × 2,000) does not.
  - Rule: send only the *headline* A/B and live demo traffic to Langfuse. Keep sweeps, ablations and CI runs in local SQLite/Parquet.
- **Quickest path to each requirement:**
  - **Per-request cost/latency/cache/route log:** your own JSONL/SQLite record (Q1 schema). It takes about 1 hour and cannot hit quota limits.
  - **Trace UI for the demo:** Langfuse Python SDK `@observe` decorators on each pipeline stage. Set cache status, similarity and route as metadata, and pass usage/cost explicitly so Langfuse doesn't have to infer prices for new model names.
  - **Savings dashboard:** Streamlit + pandas/Plotly reading the log, with these tiles and charts:
    - Total spend: baseline vs CostGuard, plus % saved with a 95% CI.
    - Cost per request and cost per correct answer.
    - Cache hit rate and false-hit rate.
    - Route mix.
    - p50/p95/p99 latency per arm.
    - Cumulative savings over trace position.
    - Cost–quality Pareto scatter across configs.
    - Table of the "worst" cache hits (lowest similarity accepted), which shows honest error analysis.
  - **Prometheus + Grafana Cloud** is nice-to-have. It needs a scrape agent (Grafana Alloy) or remote-write from the host, which costs time. Do it only if one person has spare capacity on Day 3–4.
- **Phoenix (self-hosted)** is a good no-quota alternative if Langfuse limits bite. It needs a host that stays up (it is another service to deploy), so Langfuse Cloud is simpler in a 5-day window.
- **OTel is a cheap credibility win:** name span attributes after the GenAI conventions (`gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.cache_read.input_tokens`). Put custom attributes under your own namespace (`costguard.cache.status`, `costguard.cost_usd`, `costguard.saved_usd`).

### Gaps
- I could not confirm whether the OTel GenAI conventions define a standard **cost** attribute. None was found, so treat cost as a custom attribute.
- I could not confirm the current stability level (experimental vs stable) of the GenAI conventions: the repo page fetch did not expose it.
- I did not confirm whether Langfuse Cloud dashboards show latency percentiles (p50/p99) out of the box. Compute those in your own dashboard.
- I did not verify the effort or resources needed to self-host Langfuse v3 (believed to need several services). Avoid it in 5 days.

---

## Q3. Infrastructure free tiers (vector store/cache, hosting, CI): limits that matter

### Takeaway
- **Cache / vector store:** Qdrant Cloud free (1 GB RAM) or Upstash Vector/Redis are the most generous. Redis Cloud free (30 MB) is tiny.
- **Hosting:** there is no great always-on free host for a Python proxy in 2026.
  - HF Spaces **now requires PRO ($9/month) for Docker/Gradio Spaces**.
  - Render free sleeps after 15 min.
  - Fly.io has no free tier for new orgs.
  - Railway gives a $5 one-time trial.
  - Modal gives $30/month of credits.
- **Plan:** host the proxy on Render free (pre-warm before the demo) or on Railway trial / HF PRO. Do the **heavy compression model offline** (in the A/B harness on laptops or Modal), not on a 512 MB host.
- **CI:** use a **public repo**, where GitHub Actions standard runners are free.

### Cited Findings
**Vector store / cache**
- **Qdrant Cloud free:** "0.5 vCPU / 1GB RAM / 4 GB Disk", single node, described as for "testing, and prototypes". The page does not state vector capacity or an inactivity policy. — [Qdrant pricing](https://qdrant.tech/pricing/)
- **Upstash Redis free:** 256 MB data, 500K commands/month, 10 GB bandwidth/month, 1 database, 10 MB max request, 10,000 max commands/sec. — [Upstash Redis pricing](https://upstash.com/pricing/redis)
- **Upstash Vector free:**
  - Max 200M vectors×dimensions, max 1,536 dimensions, 100 namespaces.
  - **10K queries/updates per day**, 1 GB max data, 48 KB metadata per vector.
  - Source: [Upstash Vector pricing](https://upstash.com/pricing/vector)
- **Redis Cloud free:** 30 MB, 1 database, shared, best-effort SLA. Essentials starts at $0.007/hour (min $5/month). — [Redis pricing](https://redis.io/pricing/)
  - Secondary sources (unverified on redis.io; one is a competitor's blog) add: 30 connections, 100 ops/sec, 5 GB bandwidth, and deletion after 30 days of inactivity. — [Upstash blog](https://upstash.com/blog/redis-cloud-pricing-in-2026-plans-costs-and-real-examples); [layerbase](https://layerbase.com/blog/redis-free-tier-comparison)
- **Supabase Free:** 500 MB database, 2 active projects, "paused after 1 week of inactivity", 5 GB egress, 500 MB RAM shared CPU. Pro is $25/month. — [Supabase pricing](https://supabase.com/pricing)
- **Neon Free:** 1 GB/project (20 GB total), 100 CU-hours/project/month, 100 projects, scale-to-zero after 5 min (cannot disable), up to 2 CU (8 GB RAM). pgvector is included on all plans. The paid Launch plan is $0.106/CU-hour. — [Neon pricing](https://neon.com/pricing)

**Hosting**
- **Hugging Face Spaces:**
  - "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts…"
  - Free accounts can host up to 2 Gradio Spaces on ZeroGPU.
  - CPU Basic is 2 vCPU / 16 GB RAM / 50 GB non-persistent disk at no hourly cost. Free hardware "will go to sleep … after a period of time if unused".
  - Allowed outbound ports: 80, 443, 8080.
  - Source: [HF Spaces overview](https://huggingface.co/docs/hub/spaces-overview)
  - HF PRO costs $9/month and includes "Host ZeroGPU, Gradio & Docker Spaces". — [HF pricing](https://huggingface.co/pricing)
- **Render free web service:**
  - Spins down after "15 minutes without receiving any inbound traffic"; spin-up takes "about one minute".
  - 750 free instance hours/month, single instance, no persistent disk.
  - Free Postgres expires 30 days after creation (1 GB). Free Key Value is in-memory only and loses data on restart.
  - Render may suspend for "uncommonly high volume of traffic".
  - The docs page does not state RAM/CPU.
  - Source: [Render free docs](https://render.com/docs/free)
- **Fly.io:** "New organizations don't have a free tier or a monthly free usage allowance". The trial is up to 2 hours of Machine runtime or 7 days, no card. The cheapest always-on machine is shared-cpu-1x 256 MB at $2.19/month (512 MB at $3.69/month). — [Fly.io pricing](https://docs.fly.io/about/pricing)
- **Railway:** $5 one-time trial credit for 30 days (2 vCPU / 1 GB per service, no card). The Free plan gives $1/month with 1 vCPU / 0.5 GB per service. Hobby is $5/month including $5 of usage. — [Railway pricing](https://railway.com/pricing)
- **Modal Starter:** $30/month free credits, 100 containers + 10 GPU concurrency, 3 seats. CPU costs $0.0000131/core/sec, memory $0.00000222/GiB/sec, and a T4 about $0.59/hour. — [Modal pricing](https://modal.com/pricing)

**CI**
- **GitHub Actions:**
  - Free plan: 2,000 minutes/month, 500 MB artifacts, 10 GB cache. Pro/Team: 3,000 minutes.
  - "The use of standard GitHub-hosted runners is free" for public repositories.
  - Linux 2-core costs $0.006/min.
  - Source: [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)

**Memory-relevant model facts**
- The LLMLingua-2 default checkpoint is 0.6B params in F32. — [HF model card](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank)
- Streamlit Community Cloud caps memory at 2.7 GB. — [Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

### Inferences
- **Memory caps decide where compression runs.**
  - 0.6B params × 4 bytes ≈ **2.4 GB** for the weights alone. That will not fit Render free, Railway Free (0.5 GB) or Railway trial (1 GB), and is borderline on Streamlit (2.7 GB).
  - Options:
    - (i) Run LLMLingua-2 in the offline A/B harness on a laptop or Colab and precompute compressed prompts.
    - (ii) Use the smaller mBERT-base variant. My estimate is about 0.7 GB in F32; I did not verify the parameter count.
    - (iii) Serve compression from a Modal function, billed per second against the $30 credit.
    - (iv) Pay $9 for HF PRO (16 GB CPU Basic).
  - For the live proxy on a small host, ship a **cheap fallback** (rule-based truncation / extractive top-k) and show LLMLingua results from the offline A/B.
- **Embeddings on small hosts:** use a hosted embedding API rather than a local model. text-embedding-3-small costs $0.02/1M tokens ([OpenAI pricing](https://developers.openai.com/api/docs/pricing)), so a 2,000-query trace costs well under $0.01.
- **Cache store choice:**
  - **Qdrant Cloud free (1 GB)** is the most "vector-DB-like" option and is directly supported by LiteLLM's `qdrant-semantic`, so a comparison arm is easy.
  - **Upstash Vector** is simplest (REST), but its **10K queries+updates/day** can be exhausted by threshold sweeps. Each replayed request does about 1 query plus up to 1 upsert, so 5 sweeps × 2,000 requests ≈ 10–20K ops. Run sweeps against a local in-process index (numpy/FAISS) and use the cloud store only for the live demo.
  - **Redis Cloud 30 MB:** a 1,536-dim float32 vector is about 6 KB before the cached response text, so expect only a few thousand entries. Fine for a demo, too small for sweeps.
  - **Neon/Supabase pgvector** works but adds cold starts: Neon suspends after 5 min, and Supabase pauses projects after 1 week idle.
- **Hosting recommendation for the live public URL:**
  - Proxy and Streamlit dashboard on **Render free** ($0). Hit the URL 2–3 min before any demo, since spin-up takes about 1 minute. Keep a recorded backup video.
  - Or **Railway trial** (no sleep, $5 credit).
  - Or **HF PRO** at $9, if the team wants one URL with 16 GB RAM that can run LLMLingua live. That fits the $5–20 budget but uses about half of it.
  - The dashboard can go on Streamlit Community Cloud (free, sleeps after 12 h idle).
- **CI:** make the repo public (free Actions minutes, and examiners can see the blocked PR). Never commit secrets: use Actions secrets, and on HF/Render use their secret managers.

### Gaps
- Render's free web-service RAM/CPU is not stated on the docs page fetched. It is commonly reported as 512 MB, but that is unverified here.
- I could not confirm the exact sleep timeout for free HF Spaces (the docs only say "after a period of time if unused").
- I could not confirm Qdrant free-cluster inactivity suspension or deletion rules, or whether a credit card is needed (the pricing page is silent).
- I could not confirm whether Upstash Vector's free tier includes built-in embedding models.
- I could not confirm Supabase pgvector availability on the free plan from the pricing page (it was not mentioned there).

---

## Q4. Request trace for the A/B: datasets, realistic duplicate rates, deterministic replay

### Takeaway
Build a **fixed 1,000–2,000-request JSONL trace** with a *controlled, documented* near-duplicate rate:
- **Base:** a domain set with natural paraphrase clusters (Bitext customer support).
- **Ground-truth duplicate labels:** Quora Question Pairs, to measure false-hit precision/recall.
- **Realism layer:** a WildChat sample (ODC-BY, timestamps, hashed IPs).

Target **about 20–35% near-duplicates** for the headline number, citing MeanCache's 31% per-user figure. Also show a sensitivity sweep (0/15/30/50%), so savings aren't an artefact of the chosen rate. Do not commit LMSYS-Chat-1M-derived data to a public repo: its licence forbids redistribution.

### Cited Findings
- **MeanCache** (arXiv 2403.02694):
  - Headline: "approximately 31% of such queries have a high similarity with a previously submitted query".
  - Method: 20 ChatGPT users (professors and grad students), over 27,000 queries, average 332 days of usage. A local script flagged queries "highly similar to at least one of the previously submitted queries **by the same participant**". This is a **per-user** rate, not a global one. Queries over 256 tokens were excluded.
  - GPTCache's suggested threshold of 0.7 produced 233 false hits on 700 unique queries, versus 89 for MeanCache.
  - Optimal thresholds: MPNet 0.83, ALBERT 0.78.
  - Sources: [MeanCache HTML](https://arxiv.org/html/2403.02694v3); [arXiv abs](https://arxiv.org/abs/2403.02694)
- **GPT Semantic Cache** (arXiv 2411.05276): cache hit rates of 61.6%–68.8% "across various query categories", API calls reduced by up to 68.8%, and "positive hit rates exceeding 97%". It uses query embeddings in Redis. The abstract does not give the dataset or threshold. — [arXiv 2411.05276](https://arxiv.org/abs/2411.05276)
- **LMSYS-Chat-1M:**
  - 1M conversations from 25 LLMs, collected April–August 2023, from 210,479 unique IPs in 154 languages.
  - Gated behind the "LMSYS-Chat-1M Dataset License Agreement". Names are redacted (e.g., "NAME_1"). It may contain unsafe content and has no benchmark decontamination.
  - Licence: non-exclusive and limited, with "strict prohibitions on redistribution".
  - Source: [HF dataset card](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- **WildChat-1M:**
  - 838k conversations with GPT-3.5-turbo/GPT-4, licensed **ODC-BY**.
  - Fields include timestamps, language, country/state, hashed IP, and OpenAI moderation and Detoxify toxicity flags.
  - The preview shows repeated prompts from the same user.
  - Source: [HF dataset card](https://huggingface.co/datasets/allenai/WildChat-1M)
- **Bitext customer support:**
  - 26,872 Q/A pairs across 27 intents in 10 categories, about 1,000 pairs per intent. Fields: `flags`, `instruction`, `category`, `intent`, `response`.
  - Licence: **CDLA-Sharing 1.0**.
  - Language-variation tags include colloquial, politeness, offensive, abbreviations and "spelling issues, wrong punctuation".
  - Generated with a hybrid NLG pipeline curated by linguists, so it is **synthetic-ish**.
  - Source: [HF dataset card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Quora Question Pairs:** 404,290 pairs, of which 36.92% (149,263) are labelled duplicates. — [Quora QP analysis (Medium)](https://medium.com/@princebari01/quora-question-pair-similarity-8955e3d2664) (secondary source; the figure is widely repeated, e.g. [arXiv 1907.01041](https://arxiv.org/pdf/1907.01041))
- **Determinism with the OpenAI seed parameter:**
  - It is best-effort: "Determinism is not guaranteed." Use a fixed seed and identical parameters, and log `system_fingerprint`, which changes when backend changes "might impact determinism".
  - Source: [OpenAI cookbook: seed](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter)

### Inferences
- **Recommended trace recipe:** about 1,500 requests, small enough for the $5–20 budget.
  1. **Domain core (60%):** sample 30–50 seed questions per Bitext intent across ~10 intents. Within each intent, keep Bitext's natural paraphrases. This gives "true" semantic duplicates (same intent, same answer) and "near-miss" non-duplicates (different intent, similar wording, e.g. cancel order vs track order). Near-misses are the key **false-hit test**.
  2. **Labelled-pair probe (20%):** inject QQP pairs (duplicate and non-duplicate) as consecutive-ish requests. Their labels give ground-truth cache precision/recall at each threshold τ.
  3. **Open-domain realism (20%):** sample WildChat English single-turn first user messages. Filter out toxic rows, and keep short prompts (≤256 tokens, mirroring MeanCache's filter). Long ones serve as compression/truncation targets.
  4. **Add a RAG slice** if reranking/truncation is to be demonstrated: requests with retrieved context (e.g., Bitext responses or a small FAQ corpus as "documents"), so the reranker has something to cut.
- **Controlling the duplicate rate:**
  - Assign each request to a cluster. Generate the arrival order by drawing from a Zipf-like popularity distribution over clusters, so popular intents recur.
  - Report the achieved rate: the share of requests whose cluster appeared earlier in the trace.
  - Present the headline at about 30%, explicitly justified by MeanCache's per-user 31%, and caveat that it is per-user, not service-wide.
  - Present a sensitivity table at 0%, 15%, 30% and 50%. At 0%, savings should come only from compression and routing; that is a useful honesty check.
  - Note that GPT Semantic Cache's 61–69% hit rates are a category-specific best case, not a realistic default.
- **Deterministic replay design:**
  - Use a frozen `trace.jsonl` with `{request_id, pos, t_offset_ms, user_id, cluster_id, messages, gold_answer?, gold_intent?, dup_of?}`. Commit it with its SHA-256 and the generator script with its RNG seed.
  - Replay **sequentially in trace order**, because cache state depends on order. Flush the cache before each arm.
  - Use temperature 0 and a fixed `seed`, and log `system_fingerprint`.
  - **Record/replay the upstream ("VCR cassette"):** store every upstream response keyed by `hash(model, messages, params)`. Re-running the analysis then costs $0 and is exactly reproducible, which also enables CI.
  - Run **baseline once** (all requests to the "expensive" model, no optimisations) and **each optimised configuration once**.
  - Use a separate concurrent replay only for load testing (Q5), not for the cost/quality numbers.
- **Licensing:** LMSYS-Chat-1M bars redistribution, so a public repo/trace must not contain it. Use WildChat (ODC-BY, attribution) and Bitext (CDLA-Sharing), and credit both in the README.

### Gaps
- I found no rigorous *service-wide* (cross-user) repetition statistic for public LLM chat logs. MeanCache's 31% is per-user, from 20 users. The "~30% of queries are similar" folklore traces back to it.
- I could not verify the GPT Semantic Cache paper's dataset or threshold from the abstract.
- I did not compute duplicate rates within WildChat or LMSYS directly. If time permits, a quick embedding pass over a 10k WildChat sample would give the team its own number.

---

## Q5. Evaluation methodology for "savings at equal quality": paired comparison, LLM-as-judge, metrics, CIs, Pareto, load testing

### Takeaway
Treat the A/B as a **paired experiment on the same requests**. The baseline is the expensive model with no optimisations; each optimised config runs on the same trace. Report three things:
1. **Cost:** total $, $/request and $/correct answer, with paired bootstrap 95% CIs.
2. **Quality:** a hand-written domain eval set scored by reference-guided rubric, plus pairwise LLM-judge win/tie/loss with **position swapping**, plus a human-labelled judge-agreement check.
3. **Latency:** p50/p99 overall, split into cache hits and misses, plus proxy overhead.

Frame the headline like RouteLLM: "X% cost reduction while retaining ≥95% of baseline quality". Show a cost–quality Pareto curve over threshold, compression-rate and router settings.

### Cited Findings
- **MT-Bench / "Judging LLM-as-a-judge"** (arXiv 2306.05685):
  - GPT-4 judge agreement with humans: 66% including ties, 85% excluding ties. Human–human agreement: 63% / 81%.
  - Position-consistency: GPT-4 65.0%, GPT-3.5 46.2%, Claude-v1 23.8%. Claude-v1 favoured the first answer 75% of the time.
  - Mitigation: "call a judge twice by swapping the order of two answers and only declare a win when an answer is preferred in both orders. If the results are inconsistent after swapping, we can call it a tie."
  - Verbosity "repetitive list" attack failure rates: Claude-v1 91.3%, GPT-3.5 91.3%, GPT-4 8.7%.
  - Self-enhancement: GPT-4 about +10% win rate for itself, Claude-v1 about +25%, flagged as inconclusive.
  - Reference-guided grading cut math failure rate from 70% to 15%.
  - Sources: [arXiv HTML](https://arxiv.org/html/2306.05685v4); [abs](https://arxiv.org/abs/2306.05685)
- **"Adding Error Bars to Evals"** (Evan Miller, arXiv 2411.00640) recommends:
  - Treat eval questions as samples, so standard errors follow from the CLT.
  - Use clustered standard errors when questions are grouped.
  - Use **paired-difference tests when comparing two models on the same questions**.
  - Reduce variance by resampling answers.
  - Use power analysis for sample size.
  - Source: [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- **RouteLLM reporting template:** % cost reduction at 95% of strong-model performance (85% MT Bench, 45% MMLU, 35% GSM8K). — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
- **Cache correctness is measurable:** GPT Semantic Cache reports ">97%" positive-hit accuracy ([arXiv 2411.05276](https://arxiv.org/abs/2411.05276)). At τ=0.7, GPTCache produced 233 false hits on 700 unique queries ([MeanCache](https://arxiv.org/html/2403.02694v3)).
- **k6 thresholds:**
  - Percentile thresholds look like `http_req_duration: ['p(95)<200', 'p(99)<300']` and support decimals (`p(99.9)`).
  - `abortOnFail` with `delayAbortEval` is supported.
  - Custom `Trend` metrics support `p(N)` thresholds.
  - Failing thresholds make k6 exit non-zero.
  - Source: [k6 thresholds](https://grafana.com/docs/k6/latest/using-k6/thresholds/)
  - Grafana Cloud Free includes 500 k6 VU-hours/month. — [Grafana pricing](https://grafana.com/pricing/)
- **Locust:** headless mode is `--headless --users N --spawn-rate R`, and the web UI charts RPS, response times and users. — [Locust quickstart](https://docs.locust.io/en/stable/quickstart.html)
- **Prices fetched 2026-10-03 for budgeting** (standard tier, per 1M tokens, input / cached input / output):
  - `gpt-5-nano` $0.05 / $0.005 / $0.40
  - `gpt-6-luna` $0.10 / $0.01 / $0.50
  - `gpt-5.4-mini` $0.75 / $0.075 / $4.50
  - `gpt-6.1-sol` $2.00 / $0.10 / $10.00
  - `gpt-6-astra` $10.00 / $1.00 / $50.00
  - `text-embedding-3-small` $0.02
  - Batch is about 50% off.
  - Source: [OpenAI API pricing](https://developers.openai.com/api/docs/pricing). These were extracted by an automated page summariser; re-check the exact names and prices on the page before putting them in the README.

### Inferences
- **Arms (ablation table):**
  - A0: baseline (expensive model, passthrough)
  - A1: +exact cache
  - A2: +semantic cache
  - A3: +compression/truncation
  - A4: +router/downshift
  - A5: all (CostGuard)
  - B: LiteLLM `redis-semantic` at 0.8 (off-the-shelf comparator)
  
  Each arm uses the same trace and cassette. This turns "it saves money" into an attribution table showing how much each technique contributed.
- **Quality measurement, layered:**
  1. **Hand-written domain eval set** (course requirement): 60–120 items written by the team, for example 10 per teammate per day over 2 days. Each item has a rubric and gold facts. Include near-miss pairs designed to *trap* the semantic cache, long-context items for compression, and hard items that should *not* be downshifted. Score with **reference-guided** LLM grading (pass/fail per rubric item), which MT-Bench showed is far more reliable than unguided grading on reasoning items.
  2. **Pairwise judge on the trace:** compare baseline vs optimised answers on a 200–300-request subset. Run both orders and count a win only if it is consistent, otherwise a tie (MT-Bench protocol). Report win/tie/loss; "equal quality" means loss rate ≤ a pre-registered margin (e.g., ≤5 pp), i.e. a non-inferiority framing.
     - Use a judge from a **different model family** than the served models where possible, to limit self-enhancement bias.
     - Cap answer length or instruct the judge to ignore length, to limit verbosity bias.
  3. **Judge calibration:** two teammates hand-label 40–50 pairs. Report judge–human agreement (% and Cohen's κ). This is cheap and pre-empts the "LLM judge is biased" critique.
  4. **Reference-based cheap metrics** as secondary signals: intent accuracy (for Bitext), embedding cosine to the gold response, ROUGE-L. Don't headline these, since they correlate weakly with helpfulness.
  5. **Cache-specific metrics:** false-hit rate (a hit whose cluster/QQP label says "not duplicate") and precision/recall vs τ. This is the most important honesty metric for a semantic cache.
- **Statistics:**
  - Use a **paired bootstrap** over request IDs. Draw 10,000 resamples of the trace indices, and for each compute Δ$ (A5 − A0), Δquality and $/correct; report the 2.5/97.5 percentiles.
  - Cluster-resample by `cluster_id` or `user_id` if requests are grouped (Miller's clustered-SE point). Duplicates are correlated by construction, so resampling clusters avoids overstated confidence.
  - Report the McNemar test or a bootstrap CI on the pass-rate difference for the eval set.
- **Headline metrics for the README:**
  - Cost/request (mean and median) for baseline vs CostGuard.
  - % saved with a 95% CI.
  - Cost per correct answer = total $ / #items passing.
  - Quality retained = CostGuard pass rate / baseline pass rate.
  - Win/tie/loss.
  - Cache hit rate and false-hit rate.
  - Mean compression ratio.
  - Route mix.
  - Latency p50/p99 for hits vs misses, and proxy overhead p50/p99 (total − upstream).
  - Load-test throughput (RPS at a p99 SLO).
- **Pareto curve:**
  - Sweep τ ∈ {0.75, 0.80, 0.85, 0.90, 0.95}, compression rate ∈ {0.33, 0.5, 0.7} and router threshold over a few values.
  - Plot $/request (x) against quality (y) and highlight the non-dominated frontier and the chosen operating point.
  - Run the sweeps from cassettes and a local vector index to keep them free.
- **Load testing:**
  - Test the proxy against a **mock upstream**: a FastAPI stub that sleeps for a sampled realistic latency (e.g., 300–1,500 ms) and returns canned tokens. This measures CostGuard's own overhead and throughput without spending money or hitting provider 429s.
  - Then run a short, low-concurrency real-upstream test (e.g., 2–5 min at 2–5 VUs) to show end-to-end p50/p99.
  - k6 with `p(99)` thresholds doubles as a CI-able SLO. Locust is fine if the team prefers Python.
  - Report hit-path vs miss-path latency separately; a blended p50 hides that cache hits are about 10–50 ms while misses take seconds.
  - Note the host limits: Render free is a single instance and can be suspended for unusual traffic, so load-test locally or on Railway/Modal, not against Render free.
- **Budget arithmetic** (using the fetched prices; assume about 300 input + 300 output tokens per request):
  - **Baseline on `gpt-5.4-mini`, 1,500 requests:** 0.45M in × $0.75 + 0.45M out × $4.50 ≈ **$2.36**.
  - **Same on `gpt-5-nano`:** about $0.20.
  - **Pairwise judging, 300 items × 2 orders:** about 600 calls × ~1.2k tokens on a mini-class judge ≈ **$0.5–1**.
  - **Embeddings:** under $0.01.
  - **Total:** the whole experiment fits in about **$5**. The rest of the budget covers re-runs, a judge upgrade or HF PRO ($9).
  - Avoid a flagship-tier baseline (`gpt-6-astra` at $50/1M output would cost about $22 in output alone for 1,500 requests). Choose "mini → nano" as the downshift pair, or present cost in **list-price dollars computed from tokens**, clearly labelled.
  - Set a hard spend cap in the provider dashboard.

### Gaps
- I did not verify Locust's exact CSV percentile columns or the `--csv` flag on the fetched page.
- I found no peer-reviewed standard for "equal quality" margins in LLM cost-optimisation work. The non-inferiority margin must be justified by the team.
- I could not fetch LLMLingua-2's per-task quality-drop numbers to set expectations for compression-induced quality loss.

---

## Q6. CI evaluation gate: "eval regression fails the PR" cheaply

### Takeaway
Run a **small, cached eval subset** (30–60 items from the hand-written set plus about 20 cache-trap pairs) on every PR touching `costguard/**`. Use promptfoo (`--fail-on-error` plus a pass-rate gate on its JSON output) or DeepEval (`deepeval test run`, where failing metrics fail pytest). Mark the job as a **required status check**. Stage the demo PR so it lowers the semantic-cache threshold (e.g., 0.85 → 0.6): cache-trap items then return wrong cached answers and the PR shows a red ✗. With LLM-call caching and a public repo, CI costs about $0.

### Cited Findings
- **promptfoo GitHub Action:**
  - Use `promptfoo/promptfoo-action@v1`, with inputs `github-token`, `prompts`, `config`, `openai-api-key` and `cache-path`.
  - It posts a PR comment linking to a before/after web viewer.
  - "The cache stores LLM requests and outputs, which can be reused in future runs to save cost."
  - The example triggers on `pull_request` with `paths: ['prompts/**']`.
  - Source: [promptfoo GitHub Action docs](https://www.promptfoo.dev/docs/integrations/github-action/)
- **promptfoo CI/CD:**
  - `npx promptfoo@latest eval --fail-on-error` fails on test failures.
  - Quality-gate pattern: compute `PASS_RATE` from `results.json` (`.results.stats.successes / (successes + failures)`), then `exit 1` if below 95.
  - Caching via `PROMPTFOO_CACHE_PATH` and `PROMPTFOO_CACHE_TTL`; outputs `-o results.json` and JUnit XML.
  - Source: [promptfoo CI/CD docs](https://www.promptfoo.dev/docs/integrations/ci-cd/)
- **DeepEval:**
  - `deepeval test run test_llm_app.py` wraps pytest. `assert_test()` fails when metrics fall below threshold, and "failing metrics fail the pytest run".
  - `CONFIDENT_API_KEY` is optional; only a judge key (e.g., `OPENAI_API_KEY`) is needed.
  - Metrics can be marked `flaky=True` so they don't fail the build.
  - Source: [DeepEval CI/CD docs](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd)
- **GitHub Actions:** standard runners are free on public repos; private repos on Free get 2,000 min/month and a 10 GB cache. — [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- **Required status checks:** they "must have a `successful`, `skipped`, or `neutral` status before collaborators can make changes to a protected branch". Strict mode requires the branch to be up to date. — [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- **k6** exits non-zero when thresholds fail, so a latency-SLO gate can also run in CI against the mock upstream. — [k6 thresholds](https://grafana.com/docs/k6/latest/using-k6/thresholds/)

### Inferences
- **Cheapest robust design: pure-Python, no promptfoo dependency.**
  - `pytest tests/eval/test_regression.py` loads `eval/ci_subset.jsonl`, runs it through the CostGuard pipeline, and scores it.
  - Two ways to make it cheap:
    - (a) Deterministic checks (intent match, required-fact substring/regex, cache-trap must be a MISS).
    - (b) An LLM judge whose calls are cached in `actions/cache` keyed on `hash(prompt+model)`.
  - Compare against a committed `eval/baseline_scores.json` and fail if pass rate drops more than 2 items (or 5 pp) **or** the cache false-hit rate rises above 0.
  - Upload the results JSON as an artifact and post a short PR comment with a before/after table (`gh pr comment` or `actions/github-script`).
- **Why promptfoo is still attractive:** the before/after PR comment viewer is exactly the "CI evidence" screenshot examiners like. Its built-in cache keeps re-runs cheap. Use its `--fail-on-error` plus the jq pass-rate gate.
- **Use DeepEval** if the team wants pytest-native tests with GEval-style rubric metrics. Avoid `flaky=True` on the gating metric.
- **Flakiness control:**
  - Temperature 0 with a fixed seed. Gate mostly on deterministic assertions.
  - Use the LLM judge only for a handful of items, with majority-of-3 or position-swap.
  - Cache judge outputs so an unchanged PR re-runs identically.
  - Set the gate margin above observed run-to-run noise: run the subset 3× on `main` to measure it.
- **Demo choreography:**
  - PR #1 "tune cache threshold to 0.6 for more savings" shows the savings number going up but the eval check red, with merge blocked.
  - PR #2 reverts to 0.85 and adds a verifier, and goes green.
  - Screenshot both for the README. This tells the "savings at equal quality" story in one image.
- **Keep secrets safe:** `pull_request` workflows from forks don't receive secrets. For the team's own branches this is fine, and cassette-replay mode lets CI run with **no API key at all**, which is safest.

### Gaps
- I could not confirm from the fetched page which GitHub plans allow protected branches / rulesets on **private** repos. Making the repo public avoids the question, but verify this.
- I did not verify the exact numeric exit code promptfoo uses on assertion failures (docs only say non-zero / `--fail-on-error`).
- I did not verify DeepEval's caching (`-c`) or parallelism (`-n`) flags on the fetched page.

---

## Q7. A strong 5-day plan for a 6-person team, and common failure modes

### Takeaway
Freeze the **interfaces and the trace on Day 1**, build in **six parallel workstreams** on Days 1–3, run **integration and the full A/B on Day 4** with a code freeze that evening, and use **Day 5 for README numbers, rehearsal and submission**. Most student projects of this kind fail on measurement, not on features:
- unreproducible numbers;
- unmeasured cache false hits;
- load tests that hit the real API;
- demos that die on a sleeping free host;
- last-day integration.

### Cited Findings
- Free hosts sleep:
  - Render spins down after 15 min idle, with about 1 min to spin up. — [Render free docs](https://render.com/docs/free)
  - Streamlit Community Cloud sleeps after 12 h with no traffic. — [Streamlit docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
  - Supabase pauses after 1 week of inactivity. — [Supabase pricing](https://supabase.com/pricing)
  - Neon suspends after 5 min. — [Neon pricing](https://neon.com/pricing)
- Quotas that sweeps can exhaust:
  - Upstash Vector: 10K queries/updates per day. — [Upstash Vector pricing](https://upstash.com/pricing/vector)
  - Langfuse Hobby: 50k units/month. — [Langfuse pricing](https://langfuse.com/pricing)
  - Helicone: 10k requests/month. — [Helicone pricing](https://www.helicone.ai/pricing)
- LMSYS-Chat-1M forbids redistribution. — [HF card](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- LLM judges have position, verbosity and self-enhancement biases; mitigate with swap-and-tie. — [arXiv 2306.05685](https://arxiv.org/html/2306.05685v4)
- Seeded determinism is best-effort only. — [OpenAI cookbook](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter)

### Inferences
**Team roles (six workstreams, each owning a directory and a test):**

| # | Owner focus | Deliverables |
|---|---|---|
| W1 | **Proxy core & contracts** | FastAPI `/v1/chat/completions` (OpenAI-compatible, streaming optional, so skip streaming if time is short); pipeline skeleton with stage flags; `x-costguard-*` headers; pricing table via LiteLLM SDK; log schema (SQLite/JSONL); mock-upstream server; Dockerfile |
| W2 | **Semantic cache** | Exact + semantic cache; embeddings (API or small local model); store adapter (local numpy/FAISS for sweeps, Qdrant/Upstash for live); τ sweep; false-hit analysis with QQP/Bitext near-miss labels; optional LiteLLM `redis-semantic` comparator arm |
| W3 | **Compression + rerank/truncation** | LLMLingua-2 (offline/Modal) with a cheap fallback truncation for the live host; reranker / top-k-within-budget for RAG slice; compression-ratio vs quality sweep |
| W4 | **Router/downshift** | Heuristic router first (length, keywords, intent class); optional small classifier; per-route quality check on the hard-items subset; cost accounting correctness tests (tokens × price = logged $) |
| W5 | **Trace, eval & stats** | Trace generator (seeded, documented dup rate); cassette record/replay; hand-written eval set coordination (everyone writes 10–20 items); judge prompts (pairwise + swap, reference-guided); judge-human agreement; bootstrap CIs; ablation and Pareto tables (`make ab` → `results/*.json`) |
| W6 | **Observability, dashboard, CI, deploy, load** | Langfuse integration; Streamlit savings dashboard; GitHub Actions eval gate + branch protection + staged red/green PRs; deploy (Render/Railway/HF PRO); k6/Locust scripts against the mock upstream; README skeleton with auto-filled numbers |

**Day-by-day.** Submission is Thu 8 Oct 2026; treat Sat 3 Oct as D0 if the team can meet today.
- **D0/D1 (Sat 3 – Sun 4 Oct): contracts and skeleton.**
  - Morning (1 h all-hands): agree on the API, the log schema, the config flags, the arm definitions, the baseline/cheap model pair, the non-inferiority margin and the trace recipe. Create the repo, CI skeleton and secrets. Set a provider spend cap.
  - W1 ships a passthrough proxy plus logging by end of day.
  - W5 ships `trace_v1.jsonl` (frozen) plus the cassette recorder.
  - Everyone writes 10 eval items.
  - W6 deploys the "hello" proxy to the chosen host, which de-risks hosting on Day 1.
- **D2 (Mon 5 Oct): components.**
  - W2, W3 and W4 each build their stage behind a flag, with unit tests against cassette data.
  - W5 records the **baseline arm** (A0) cassette on the full trace; this is the main real-API spend.
  - W6 builds a dashboard on fake/early logs and adds the CI eval job, initially non-blocking.
- **D3 (Tue 6 Oct): sweeps and first numbers.**
  - Each component owner runs their sweep from cassettes: τ for the cache, rate for compression, threshold for the router. Pick the operating points.
  - W5 builds the judge pipeline and the bootstrap.
  - W6 adds the Langfuse traces and the k6 script against the mock upstream.
  - **End-of-day checkpoint:** a single command produces a first ablation table, even if rough.
- **D4 (Wed 7 Oct): integration and the full A/B. Code freeze at 20:00.**
  - Run all arms on the frozen trace. Run the pairwise judge, the judge calibration (40–50 human labels) and the load tests (mock and short real).
  - Make the CI gate required. Stage the red PR and the green PR.
  - Deploy the final build and verify the public URL from a phone.
  - Record a **backup demo video**.
- **D5 (Thu 8 Oct): README, rehearsal, submit.**
  - Auto-generate the README tables from `results/*.json` (no hand-typed numbers).
  - Do two timed rehearsals: live drop-in demo (change `base_url`, show headers, dashboard, Langfuse trace, red/green PR).
  - Pre-warm hosts 5 min before.
  - Submit with a buffer of several hours.

**Common failure modes (and the guard for each):**
1. **Numbers that cannot be reproduced.** The trace changed, the cache wasn't flushed, or the order differed. Guard: a frozen trace with a hash, cassette replay, `make ab` and committed `results/*.json`.
2. **Savings from wrong answers.** An aggressive τ gives big hit rates with silent false hits. Guard: near-miss traps, false-hit-rate reporting, a quality gate in CI, and showing the worst accepted hits.
3. **The headline duplicate rate looks rigged.** Guard: cite MeanCache's 31% per-user figure, show the 0/15/30/50% sensitivity, and report savings excluding the cache.
4. **Unchecked LLM-judge bias.** Guard: swap positions, use a different-family judge, publish the human agreement check, and report ties.
5. **Single-run point estimates.** Guard: paired bootstrap CIs and clustered resampling.
6. **Load test measures OpenAI rather than CostGuard, burns the budget, or trips 429s.** Guard: a mock upstream for throughput and overhead, and only a tiny real run.
7. **The free host sleeps or OOMs during the demo** (Render 15 min sleep with about 1 min wake; LLMLingua at about 2.4 GB weights on a ≤1 GB host). Guard: pre-warm, a lightweight live path, heavy compression offline, and a backup video.
8. **Quota exhaustion mid-sweep** (Upstash Vector 10K/day, Langfuse 50k units). Guard: run sweeps locally and use cloud services for the demo only.
9. **Licence and secret leaks.** LMSYS data in a public repo, or API keys committed. Guard: WildChat/Bitext only, secret scanning, and cassette-mode CI without keys.
10. **Big-bang integration on the last day.** Guard: a passthrough proxy deployed on D1, stages behind flags, and the integration checkpoint on D3 evening.
11. **Cost accounting bugs.** Wrong price table, cached-token pricing ignored, or embedding/judge costs omitted. Guard: a unit test that recomputes $ from tokens. Include CostGuard's *own* overhead (embedding calls, compression compute) in the optimised arm's cost, because examiners will ask.
12. **Scope creep** (streaming, auth, multi-provider, fancy frontend). Guard: an explicit "won't do" list on D1.

### Gaps
- I found no published post-mortems or rubrics specific to student LLM cost-optimisation projects. The failure modes above are inferred from the tool limits and evaluation literature cited, not from a survey of such projects.
- I did not verify the team's course rubric weighting, e.g. whether the live URL or the CI evidence is graded as heavily as the A/B. Prioritise accordingly.
