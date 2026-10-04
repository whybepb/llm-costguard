# Semantic Caching for LLM Applications: State of the Art, Tools, Published Numbers and Evaluation (as of Oct 2026)

Scope note: research done 2026-10-03. Primary sources are arXiv PDFs (read in full text where numbers are quoted), vendor docs and GitHub metadata pulled through the GitHub API. Anything dated before 2024 is flagged **[pre-2024]**. Where a number comes from a synthetic or paraphrase-generated benchmark rather than real traffic, that is stated, because it changes how far the number can be trusted.

---

## 1. How does semantic caching work end to end?

### Takeaway
A semantic cache embeds the incoming prompt, runs a nearest-neighbour (ANN) search over embeddings of earlier prompts, and returns the stored response if the similarity is at or above a threshold τ. Otherwise it calls the LLM and writes (embedding, prompt, response, metadata) back to the cache. TTL and eviction keep the cache fresh and bounded. Production designs put cheaper deterministic tiers (exact-hash, curated static answers) before the fuzzy semantic tier. The whole correctness question comes down to the single hit/miss decision at τ.

### Cited Findings
**Core loop (canonical description)**
- vCache (ICLR 2026) describes the standard loop: embed the request x into E(x), retrieve the nearest neighbour nn(x) and its response from a vector DB, and compute s(x)=sim(E(x),E(y)) ∈ [0,1]. If s(x) ≥ t the cache returns the cached response ("exploitation"). Otherwise it calls the LLM, adds E(x) to the vector DB and stores r(x) in metadata ("exploration"). — [vCache, arXiv 2502.03771v5](https://arxiv.org/abs/2502.03771)
- On thresholds, vCache says existing systems "use a predefined threshold (e.g., 0.8) or determine one by testing multiple values upfront". If t is too low, unrelated prompts produce false hits. If it is too high, valid reuse is lost. — [vCache](https://arxiv.org/abs/2502.03771)
- GPTCache **[pre-2024, Dec 2023]** is built from these modules: LLM adapter, embedding generator, cache manager (cache storage + vector store + eviction), similarity evaluator and post-processor. Its paper reports a cache hit costing ~0.3 s on a local Mac (paraphrase-albert-small-v2 in ONNX) against ~3 s average for ChatGPT. That is roughly 1/10 of the latency, and no tokens are consumed. — [GPTCache paper, NLP-OSS 2023](https://aclanthology.org/2023.nlposs-1.24.pdf)
- GPTCache's pluggable parts:
  - Embeddings: OpenAI, ONNX, Hugging Face, Cohere, fastText, SentenceTransformers, Timm.
  - Vector stores: Milvus, Zilliz, FAISS, Hnswlib, PGVector, Chroma, Qdrant, Weaviate and others.
  - Scalar stores: SQLite, Postgres, MySQL, Redis, MongoDB and others.
  - Similarity evaluators: distance, ONNX model, exact match, BM25.
  - Eviction: LRU, FIFO, LFU, RR. — [GPTCache GitHub](https://github.com/zilliztech/GPTCache)

**Write-back and identity**
- RedisVL `SemanticCache.store()` writes the prompt, response and optional metadata/filters. The entry id is "a deterministic hash of the prompt + filters", so an identical prompt with identical filters overwrites the old entry, while different filters create separate entries. — [RedisVL LLM cache guide](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html)

**TTL / eviction**
- RedisVL: `ttl=None` by default, meaning entries persist. When a TTL is set, a successful `check()` refreshes the TTL on the matched entry, which gives a sliding window. — [RedisVL](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html)
- AWS ElastiCache guidance:
  - Recommended TTLs: static facts 24 h; product info 12–24 h; general assistant answers 1–4 h; real-time data (prices, inventory) 5–15 min; conversation context 30 min.
  - Add random jitter to TTLs and use `maxmemory-policy allkeys-lru` for eviction.
  - Plan ~4–6 KB per entry (embedding dims × 4 bytes + query + response), i.e. ~170,000 entries per 1 GB.
  - Invalidate stale entries by text-searching on a topic. — [AWS ElastiCache best practices](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)
- Category-aware caching (IBM Research/Tencent/Red Hat, Oct 2025) argues that one global TTL "either wastes space (code patterns stable for months) or serves stale data (stock prices changing per second)". It proposes per-category thresholds, TTLs and quotas. — [Category-Aware Semantic Caching, arXiv 2510.26835](https://arxiv.org/abs/2510.26835)

**Lookup cost / ANN**
- Category-aware paper:
  - Remote vector-DB search costs ~30 ms, so a category needs a 15–20% hit rate to break even.
  - An in-memory HNSW index with external document storage cuts miss cost to ~2 ms, and break-even drops to 3–5%.
  - ~2 KB per entry in their design. — [arXiv 2510.26835](https://arxiv.org/abs/2510.26835)
- A practitioner report (VentureBeat op-ed by a lead engineer at an unnamed company, Jan 2026) gives these overheads:
  - Query embedding 12 ms p50 / 28 ms p99.
  - Vector search 8 ms p50 / 19 ms p99.
  - Total lookup 20 ms p50 / 47 ms p99. — [VentureBeat, Jan 2026](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)
- AWS measured individual hits at 0.11–0.13 s against misses of 1.64–6.51 s, i.e. 12×–59× faster per query (ElastiCache r7g.large, Titan Text Embeddings V2, Claude 3 Haiku). — [AWS ElastiCache benchmarks](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-benchmarks.html)

**Exact-match tier first / tiered caches**
- Apple's Krites paper (Feb/Mar 2026) says production deployments "typically use a tiered static-dynamic design":
  - A read-only **static** cache of curated, offline-vetted answers mined from logs, checked first with τ_static.
  - Then a **dynamic** cache populated online, checked with τ_dynamic.
  - Then the LLM backend, with write-back into the dynamic tier only. — [Krites, arXiv 2602.13165](https://arxiv.org/abs/2602.13165)
- The agent-caching paper (Mar 2026) proposes a five-tier cascade: fingerprint (exact) → BERT → SetFit intent classifier → cheap LLM → deep agent. It reports that this handles 85% of interactions locally. — [arXiv 2602.18922](https://arxiv.org/abs/2602.18922)
- Gateways that do *only* exact-match caching (a hash of the request) include Helicone, Cloudflare AI Gateway and Portkey "simple cache" (details in §2). — [Cloudflare docs](https://developers.cloudflare.com/ai-gateway/features/caching/); [Helicone docs](https://docs.helicone.ai/features/advanced-usage/caching)
- VentureBeat case: of 100,000 production queries, 18% were exact duplicates, 47% were semantically similar and 35% were novel. In that workload an exact tier alone catches about 18%. — [VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)

**Reranking / verification stage (optional step between ANN and return)**
- Redis/NYU (EMNLP 2026 Industry): a cross-encoder reranker can jointly encode each query–candidate pair and replace the retriever score for the threshold decision. They evaluate top-K=50 ANN candidates. — [Closing the Operational Gap, arXiv 2606.19719](https://arxiv.org/abs/2606.19719)
- vCache labels hits during exploration with string match (classification) or LLM-as-judge (open-ended). It says the judge call can run asynchronously off the critical path, and the threshold update costs ≤ ~1.5 ms. — [vCache](https://arxiv.org/abs/2502.03771)

### Inferences
- A defensible CostGuard pipeline is:
  1. Normalise and hash the full request (model + system prompt + messages + sampling params) as an exact tier.
  2. Embed the last user message, plus a context digest for multi-turn.
  3. Run ANN top-k with metadata filters (tenant, model, system-prompt hash).
  4. Apply the threshold; optionally rerank or verify.
  5. Return or call the LLM.
  6. Write back with a TTL.
- Every stage maps to a cited system.
- With a local embedding model and in-process index (FAISS/hnswlib), lookup overhead should be in the tens of ms. That is negligible next to multi-second LLM latency. The real overhead is paid on misses, which is why low-hit-rate workloads can come out net negative on latency (the category-aware break-even argument).

### Gaps
- No primary source gives a canonical "normalisation" step (lower-casing, whitespace stripping) before the exact tier. It is common practice but uncited here.

---

## 2. Main implementations and how they differ

### Takeaway
The open-source options students can run for free are GPTCache (MIT, modular, but no release since Aug 2024), RedisVL `SemanticCache` (MIT, actively released, has tenant filters and TTL), LiteLLM's proxy cache (redis-semantic / qdrant-semantic) and research code such as vCache (non-commercial licence). Managed gateways split into two groups. Helicone and Cloudflare AI Gateway offer *exact-match only*. Portkey offers exact match on all plans, but semantic caching only on select Enterprise plans. Redis LangCache is a managed semantic-cache service. For a 5-day student build, RedisVL or a hand-rolled FAISS + sentence-transformers cache is the most controllable choice. GPTCache is the classic baseline to compare against.

### Cited Findings

**Comparison table (verified via docs and the GitHub API, 2026-10-03)**

| Tool | Type | Semantic? | Licence / status | Key knobs |
|---|---|---|---|---|
| **GPTCache** (Zilliz) | Python library | Yes | MIT; 8.2k stars; latest release **0.1.44 on 2024-08-01**; repo last pushed 2026-09-22; README warns "API may be subject to change at any time" | embedding / vector-store / evaluator / eviction modules; paper used threshold 0.7 with ALBERT |
| **RedisVL SemanticCache** | Python library on Redis (Stack/Cloud) | Yes | MIT; v0.27.2 released 2026-09-10 | `distance_threshold` (Redis COSINE *distance*, 0–2; **default 0.1**), `ttl`, `filterable_fields`, vectorizer (guide uses `redis/langcache-embed-v2`) |
| **Redis LangCache** | Managed service (REST) on Redis Cloud | Yes | Proprietary, managed | uses Redis's LangCache-Embed models |
| **LiteLLM** proxy/SDK | Gateway cache | Yes (`redis-semantic`, `valkey-semantic`, `qdrant-semantic`) + exact (`redis`, `local`, `disk`, `s3`, `gcs`) | GitHub reports "NOASSERTION" (mixed licence); 60k stars; v1.103.2 (2026-10-01) | `similarity_threshold` (docs examples 0.8 Redis / 0.7 Qdrant); embedding defaults to `text-embedding-ada-002` |
| **Portkey** | Gateway (OSS gateway MIT; hosted SaaS) | Simple (exact) on all plans; semantic only on "select Enterprise plans" | gateway MIT, 13.1k stars, v1.15.2 (Jan 2026) | default threshold **0.95**; semantic only for requests <8,191 tokens and ≤4 messages; **system prompt ignored**; max_age default 7 days (min 60 s, max 90 days); `cache_namespace`; force refresh |
| **Helicone** | Observability gateway | **No (exact only)** | Apache-2.0 | `Helicone-Cache-Enabled`; `Cache-Control: max-age` default 7 days, max 365 days; `Helicone-Cache-Bucket-Max-Size` (default 1); `Helicone-Cache-Seed` namespace; stored in Cloudflare Workers KV |
| **Cloudflare AI Gateway** | Edge gateway | **No (exact only)**; "We plan on adding semantic search for caching in the future" | Managed | key = provider + endpoint + model + auth header + full body; TTL min 60 s, max 1 month; headers `cf-aig-cache-ttl`, `cf-aig-skip-cache`, `cf-aig-cache-key` |
| **vCache** (Berkeley) | Research library | Yes, with learned per-entry thresholds | **CC BY-NC-ND 3.0** (non-commercial, no derivatives); v1.0; `pip install -e .` | `VerifiedDecisionPolicy(delta=0.01)` error-rate bound; OpenAI by default |
| **Upstash semantic-cache** | TS library on Upstash Vector | Yes | MIT; last release v1.0.5 (2024-11-21) | — |

Sources for the table:
- GPTCache: [GitHub](https://github.com/zilliztech/GPTCache)
- RedisVL: [docs](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html), [repo](https://github.com/redis/redis-vl-python)
- LangCache: [Redis docs](https://redis.io/docs/latest/develop/ai/langcache/); the page returned 404 when fetched directly, so the description comes from its search-index summary.
- LiteLLM: [caching overview](https://docs.litellm.ai/docs/proxy/caching), [all caches](https://docs.litellm.ai/docs/caching/all_caches)
- Portkey: [cache docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic), [gateway repo](https://github.com/Portkey-AI/gateway)
- Helicone: [docs](https://docs.helicone.ai/features/advanced-usage/caching), [repo](https://github.com/Helicone/helicone)
- Cloudflare: [docs](https://developers.cloudflare.com/ai-gateway/features/caching/)
- vCache: [repo](https://github.com/vcache-project/vCache)
- Upstash: [repo](https://github.com/upstash/semantic-cache)

**Additional details**
- LiteLLM's cache key combines model name, messages, temperature and logit_bias. Per-request controls are `no-cache`, `no-store`, `ttl` and `s-maxage`. `qdrant-semantic` supports binary/product/scalar quantisation. — [LiteLLM all caches](https://docs.litellm.ai/docs/caching/all_caches)
- Portkey's semantic-cache embeddings support only OpenAI, Azure OpenAI, Google and Vertex AI as embedding providers. The threshold is customisable via `SEMANTIC_CACHE_SIMILARITY_THRESHOLD`. — [Portkey docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic)
- Redis LangCache is described as "a fully-managed semantic caching service" that checks whether a similar response is already stored before calling the LLM. — [Redis LangCache docs](https://redis.io/docs/latest/develop/ai/langcache/)
- Redis's caching-specific embedding models:
  - `redis/langcache-embed-v2`: ModernBERT-based, 149M params, 768-dim.
  - `redis/langcache-embed-v3-small`: initialised from all-MiniLM-L6-v2, 22.6M params, 384-dim, Apache-2.0, trained on 8M paraphrase pairs with contrastive + ArcFace loss, updated Dec 2025. — [HF langcache-embed-v3-small](https://huggingface.co/redis/langcache-embed-v3-small); [arXiv 2606.19719](https://arxiv.org/abs/2606.19719)
- LangChain ships a `RedisSemanticCache` in `langchain-redis`. — [LangChain reference](https://reference.langchain.com/python/langchain-redis/cache/RedisSemanticCache)
- Cloud platforms that now ship semantic caching, per the 2026 security papers, include AWS (Bedrock/ElastiCache), Microsoft Azure and Alibaba Higress. — [Poisoning defense, arXiv 2609.35908](https://arxiv.org/abs/2609.35908); [Key Collision Attack, arXiv 2601.23088](https://arxiv.org/abs/2601.23088)
- AWS documents semantic caching on ElastiCache (Valkey) with tag/numeric filters to decide which queries are eligible and to scope hits. — [AWS best practices](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)

**2025–2026 newcomers (research)**
- vCache: verified, per-embedding learned thresholds. — [arXiv 2502.03771](https://arxiv.org/abs/2502.03771)
- Krites (Apple): an asynchronous LLM judge promotes "grey-zone" static matches into the dynamic tier. — [arXiv 2602.13165](https://arxiv.org/abs/2602.13165)
- Category-aware caching: per-category threshold, TTL and quota. — [arXiv 2510.26835](https://arxiv.org/abs/2510.26835)
- LaCache: checks the first k speculatively decoded tokens to resist collisions. — [arXiv 2608.01718](https://arxiv.org/abs/2608.01718)
- Cisco "generative caching": synthesises answers from multiple cached responses and claims to be "considerably faster than GPTcache". — [arXiv 2503.17603](https://arxiv.org/abs/2503.17603)
- W5H2 structured intent canonicalisation for agent caches. — [arXiv 2602.18922](https://arxiv.org/abs/2602.18922)
- The vLLM Semantic Router project is active: Apache-2.0, v0.4.0 released 2026-09-27. — [GitHub](https://github.com/vllm-project/semantic-router)

### Inferences
- **Pick RedisVL or a minimal custom cache, and benchmark against GPTCache.** GPTCache is the baseline every paper compares against (MeanCache, SCALM and vCache all use it). Its last PyPI release is from August 2024, so treat it as "classic baseline", not current state of the art.
- RedisVL's default `distance_threshold=0.1` is cosine *distance*. Assuming Redis COSINE distance = 1 − cosine similarity, which matches its stated 0 = identical / 2 = opposite range, that is equivalent to cosine similarity ≥ 0.9. Students must convert units when reporting thresholds, or the examiner will catch the inconsistency.
- vCache's licence (CC BY-NC-ND) permits academic use and comparison. Redistributing a modified copy inside a product would be a problem, so reimplement the idea from the paper rather than vendoring the code.
- Helicone and Cloudflare are useful as "exact-tier" reference points. They do not demonstrate a semantic threshold trade-off.

### Gaps
- LlamaIndex's current semantic-cache integration was not verified (no primary doc fetched).
- Redis LangCache pricing and free-tier limits were not retrieved (the docs page returned 404).
- Whether Maxim's Bifrost gateway (Apache-2.0, active) ships semantic caching was not verified.
- Redis Cloud free-tier memory size was not verified.

---

## 3. Published hit rates, false-hit rates, latency and cost savings

### Takeaway
Published numbers vary enormously with the dataset. Benchmarks built by *generating paraphrases* of seed prompts report 50–90% hit rates. Analyses of *real* chat logs (LMSYS, MOSS) find only ~4–8% of queries are answerable from a semantically similar earlier query. Vendor and production reports cluster around ~20% (Portkey) up to ~65% for narrow support/FAQ workloads. Even at very strict thresholds (0.99), static-threshold caches show non-zero error that grows with traffic. That finding motivates verified and adaptive methods (vCache, Krites).

### Cited Findings

**Academic papers (with dataset / model / threshold)**

- **GPTCache paper [pre-2024, Dec 2023]**
  - Setup: dataset of ChatGPT-generated sentence pairs (similar / opposite), 30,000 positive pairs cached; embedding paraphrase-albert-small-v2 (ONNX); threshold 0.7 (chosen as the balance point); hit correctness judged by ChatGPT similarity ≥ 0.7.
  - Experiment 1 (1,000 paraphrased queries): 876 hits / 124 misses; 837 positive, **39 negative (false) hits**; hit latency 0.20 s.
  - Experiment 2 (1,160 queries, 50% positive / 50% unrelated): 570 hits, 590 misses, 549 positive, **21 negative hits**; 0.17 s.
  - The authors state "even the best cache hit rates do not exceed 90%" and that positive-hit rates "are unlikely to reach production requirements, such as 99%, without decreasing cache hits". — [GPTCache paper](https://aclanthology.org/2023.nlposs-1.24.pdf)
- **MeanCache (IPDPS 2025; arXiv v1 Mar 2024)**
  - Design: user-side cache, federated-learned embedding model, context chains for follow-ups.
  - Evaluated on the GPTCache dataset against GPTCache (ALBERT, τ=0.7).

    | Metric | GPTCache | MeanCache MPNet | MeanCache Albert |
    |---|---|---|---|
    | F-score | 0.56 | 0.73 | 0.68 |
    | Precision | 0.52 | 0.72 | 0.66 |
    | Recall | 0.85 | 0.78 | 0.77 |

  - Contextual queries (450-query synthetic GPT-4 set): GPTCache F 0.67 / precision 0.66 against MeanCache F 0.93 / precision 0.98.
  - Headline: ~17% higher F-score, ~20% higher precision, 83% less storage (PCA compression), 11% faster matching.
  - Learned optimal thresholds: **MPNet τ=0.83** (F1 0.89, precision 0.92, accuracy 0.90) and **Albert τ=0.78** (F1 0.88). The authors conclude GPTCache's suggested 0.7 is suboptimal and that "the optimal threshold τ values varies with the embedding model".
  - User study: 20 ChatGPT users, 27K queries; ~31% of queries were similar to previous ones (academic setting). — [MeanCache, arXiv 2403.02694](https://arxiv.org/abs/2403.02694)
- **SCALM (arXiv May 2024)**
  - Real human–LLM logs: MOSS (1M conversations, 6.7 rounds avg) and LMSYS (300k conversations, 1.8 rounds avg); embedding text-embedding-3-small; threshold **0.90**, chosen via GPT-4 judgement as the balance between correctness and reuse.
  - Only **4.5% (MOSS) and 7.5% (LMSYS)** of queries could be answered with responses to similar queries.
  - Baseline GPTCache hit ratios were **3.8% and 6.4%** (1,000 sampled conversations, 100-entry cache).
  - SCALM's semantic-cluster-aware eviction gives on average a **63% relative increase in hit ratio and 77% relative increase in token savings** over GPTCache.
  - Example in absolute terms: at 5,000 conversations, 11.7% / 11.6% hit ratio against LFU baselines >4.3 points lower; token-saving ratio 8.4% / 8.2%. — [SCALM, arXiv 2406.00025](https://arxiv.org/abs/2406.00025)
- **GPT Semantic Cache (arXiv Nov 2024)**
  - Setup: 8,000 constructed Q&A pairs in 4 categories; 2,000 test queries (500/category); all-MiniLM-L6-v2 + Redis + ANN; threshold **0.8**; hit validity judged by GPT-4o-mini.
  - Hit rates 61.6–68.8% (e.g., order/shipping 68.8%, Python basics 67%, network support 67%, shopping QA 61.6%).
  - "Positive hit rate" above 97% (reported range 92.5% upward across categories).
  - Caveat: the test queries were generated for the experiment, not taken from real traffic. — [arXiv 2411.05276](https://arxiv.org/abs/2411.05276)
- **vCache (ICLR 2026)**
  - Setup: 3 embedding models (GTE-large-en-v1.5, E5-large-v2, text-embedding-3-small), 2 LLMs (Llama-3-8B/3.1-8B, GPT-4o-mini; GPT-4.1-nano also used), 5 datasets / 4 benchmarks:
    - SemCacheLMArena: 60,000 prompts.
    - SemCacheClassification: 45,000.
    - SemCacheSearchQueries: 150,000 from MS MARCO.
    - SemCacheCombo: 27,500.
  - Results:
    - Up to **12.5× higher hit rate and 26× lower error rate** than static-threshold GPTCache and fine-tuned-embedding baselines (on SemCacheLMArena), while meeting a user-set error bound δ (e.g., 1.5–3%).
    - Static thresholds show error *rising with sample size*. On SearchQueries, **a static threshold of 0.99 still yields 1.7% error after 150k samples**, and GPTCache at 0.83 on Combo shows growing error.
    - On SemCacheClassification, vCache beats all baselines for error bounds above 1.5%.
  - Error rate is defined as FP/n over *all* prompts. — [vCache](https://arxiv.org/abs/2502.03771)
- **Krites (Apple, 2026)**
  - Trace-driven simulation on SemCacheLMArena and SemCacheSearchQueries.
  - Increases the share of requests served with curated static answers by up to **136% (conversational) and 290% (search)**, i.e. up to 3.9×, at fixed error rate.
  - No increase in critical-path latency, since the judge runs asynchronously. — [arXiv 2602.13165](https://arxiv.org/abs/2602.13165)
- **Redis LangCache-Embed (Apr 2025)**
  - Fine-tuning ModernBERT for one epoch:
    - Quora: average precision 76% → 92%, precision 64% → 84%.
    - Medical: AP 92% → 97%, precision 78% → 92%.
  - Medical, synthetic-only fine-tune:
    - LangCache-Embed-Synthetic: precision 0.87, recall 0.90, F1 0.89.
    - text-embedding-3-small: 0.83 / 0.89 / 0.86.
    - text-embedding-3-large: 0.85 / 0.87 / 0.86. — [arXiv 2504.02268](https://arxiv.org/abs/2504.02268)
- **Category-aware (Oct 2025)**
  - States that high-repetition categories (code, docs) achieve 40–60% hit rates and account for 60–70% of traffic, while low-repetition or volatile categories get 5–15%.
  - Load-adaptive policies reduce traffic to overloaded models by 9–17% "in theoretical projections".
  - Caveat: these are characterisations without a named public dataset. — [arXiv 2510.26835](https://arxiv.org/abs/2510.26835)
- **Agent workloads (Mar 2026)**: GPTCache got only **3.3% hit rate** on synthetic personal-agent tasks and 37.9% accuracy as clustering on MASSIVE; Agentic Plan Caching got 0–12%. — [arXiv 2602.18922](https://arxiv.org/abs/2602.18922)

**Vendor / production numbers**
- **AWS ElastiCache benchmark (2025–26)**
  - Setup: 63,796 chatbot queries "and their paraphrased variants" from **SemBenchmarkLmArena** (vCache's benchmark); Titan Text Embeddings V2; Claude 3 Haiku; cache started empty; queries streamed randomly. Baseline cost $49.50/day, 4.35 s average latency.

    | Threshold | Hit ratio | Accuracy of cached responses | Cost savings | Latency reduction |
    |---|---|---|---|---|
    | 0.99 | 23.5% | 92.1% | 15.8% | 17.1% |
    | 0.95 | 56.0% | 92.6% | 51.9% | 57.7% |
    | 0.90 | 74.5% | 92.3% | 72.5% | 72.2% |
    | 0.80 | 87.6% | 91.8% | 84.6% | 86.1% |
    | 0.75 | 90.3% | 91.2% | 86.3% | 88.3% |
    | 0.50 | 94.3% | 87.5% | 88.0% | 89.3% |

  - Per-query hit latency 0.11–0.13 s against 1.64–6.51 s misses (12–59×). — [AWS benchmarks](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-benchmarks.html)
  - Portkey's April 2026 blog reproduces this table as "AWS study". — [Portkey thresholds blog](https://portkey.ai/blog/semantic-caching-thresholds/)
- **Portkey**
  - ~20% cache hit rate at 99% accuracy for Q&A/RAG; in RAG use cases, 18% to as high as 60% hit rate with accurate results 99% of the time (search-result summary of Portkey's blog). — [Portkey blog](https://portkey.ai/blog/reducing-llm-costs-and-latency-semantic-cache/)
  - Recommends starting around 0.95 and backtesting on ~5,000 queries until accuracy stays above 99%, "based on learnings from over 250 million cache requests".
  - Puts the practical false-positive ceiling at ~3–5% before the embedding model becomes the bottleneck. — [Portkey thresholds blog, 2026-04-18](https://portkey.ai/blog/semantic-caching-thresholds/)
- **VentureBeat op-ed (Jan 2026, unnamed company, self-reported)**
  - API cost $47,000 → $12,700/month (−73%); hit rate 18% → 67%; latency 850 → 300 ms.
  - False-positive rate 0.8%; customer complaints +0.3%.
  - Per-type thresholds: FAQ 0.94, search 0.88, support 0.92, transactional 0.97, default 0.92. — [VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)
- **vCache README marketing**: "Reduce LLM API Costs by up to 10x. Decrease latency by up to 100x." No dataset attached. — [vCache repo](https://github.com/vcache-project/vCache)

### Inferences
- **Explain the gap between benchmark and real traffic.** Paraphrase-generated benchmarks (GPT Semantic Cache, AWS on SemBenchmarkLmArena, the GPTCache paper) guarantee that a near neighbour exists for most queries, which inflates hit rates. SCALM's real LMSYS/MOSS analysis (4.5–7.5% reusable) and Portkey's ~20% are better priors for general chat. CostGuard should report both a synthetic benchmark and a realistic one.
- **AWS accuracy barely moves (91–93%) between thresholds 0.75 and 0.99.** That suggests the "accuracy" metric has a floor set by judge noise and LLM non-determinism rather than by the threshold. Converting their per-hit accuracy to per-request error:
  - At 0.99: 0.235 × 7.9% ≈ 1.9% of all requests wrong.
  - At 0.80: 0.876 × 8.2% ≈ 7.2%.

  This is the vCache-style metric. Students should report both per-hit precision and per-request error.
- Static-threshold error grows as the cache fills, because more neighbours means more chances of a wrong nearest neighbour (vCache Fig. 4). Any CostGuard evaluation should therefore stream the full dataset and plot error against number of requests, not just a single final number.

### Gaps
- No public, independently audited production hit-rate dataset was found. Production numbers are vendor- or self-reported (Portkey, VentureBeat).
- Krites' absolute hit/error numbers were not extracted beyond the relative gains in the abstract and introduction.

---

## 4. How to choose and evaluate the threshold

### Takeaway
Treat the threshold as a binary classifier operating point. Label (query, nearest-cached-query) pairs as "same answer acceptable" or not, sweep τ, and plot cache-hit rate against false-hit rate (or precision). Report the operating point that meets a target error budget. Cosine thresholds are *embedding-model-specific*: published optima range from 0.7 (ALBERT, GPTCache) to 0.78–0.83 (MPNet/Albert, MeanCache), 0.80 (MiniLM, GPT Semantic Cache) and 0.90 (text-embedding-3-small, SCALM), up to the 0.95 defaults of Portkey and AWS's strict tier. So "0.8–0.95" is realistic only once calibrated per model and domain. The 2026 literature shows that a single global threshold can never guarantee an error rate, and that ranking metrics (PR-AUC) mislead model selection.

### Cited Findings

**Metric definitions**
- MeanCache defines true hit / false hit / true miss / false miss, then:
  - Precision = TP/(TP+FP).
  - Recall = TP/(TP+FN).
  - Fβ.
  - Accuracy = (TP+TN)/all.

  It notes that "traditional hit/miss metrics are potentially misleading in semantic caches". — [MeanCache](https://arxiv.org/abs/2403.02694)
- vCache error rate = FP/n over all prompts; cache-hit rate is reported alongside. Its evaluation plots hit rate against error rate across thresholds (ROC-like Pareto curves) and against average latency. — [vCache](https://arxiv.org/abs/2502.03771)
- SCALM adds a "total token saving ratio" metric because hit ratio alone ignores that long queries are worth more. — [SCALM](https://arxiv.org/abs/2406.00025)
- Redis/NYU (EMNLP 2026) introduces **P-CHR AUC**, precision integrated across cache-hit-ratio levels as τ varies, and the **Operational Retention Rate** (ORR = P-CHR AUC / PR-AUC).
  - They show "models with the highest PR-AUC are often the worst in operation".
  - There is an irreducible structural gap Δstr = 1 − p(1 − ln p) set by the dataset's positive rate p.
  - Retriever results on LangCache SentencePairs v3 test (74,265 pairs, 45% positive), PR-AUC → P-CHR AUC:
    - LangCache-Embed-v3: 0.833 → 0.445.
    - BGE-base-en-v1.5: 0.660 → 0.372.
    - E5-base-v2: 0.632 → 0.358.
    - Nomic-embed-v1.5: 0.633 → 0.373.
  - Their conclusion: "model selection for semantic caching is a threshold-utility problem, not a ranking one." — [arXiv 2606.19719](https://arxiv.org/abs/2606.19719)

**Why one static threshold fails**
- vCache: correct and incorrect hits have "highly overlapping similarity distributions". Optimal per-embedding thresholds "vary substantially" (Fig. 3 on SemCacheClassification), so no single threshold suffices. Maintaining a bounded error with a static threshold requires continually raising it, and "no static threshold below 1.0 may suffice". — [vCache](https://arxiv.org/abs/2502.03771)
- vCache method:
  - For each cached embedding, fit a sigmoid P(correct | similarity) to observed (similarity, correct?) labels by MLE.
  - Take a confidence bound, and exploit with probability that keeps the overall error ≤ δ.
  - No offline training; the authors call it model-agnostic, with ≤1.5 ms overhead.
  - It assumes i.i.d. data and a sigmoid family. — [vCache](https://arxiv.org/abs/2502.03771)
- Krites keeps the static threshold on the serving path and sends grey-zone near-misses to an *async* LLM judge. Example pairs below conservative thresholds that are still interchangeable: "What's the word on my dog having honey?" ↔ "Can my dog have honey?". — [Krites](https://arxiv.org/abs/2602.13165)

**Per-domain / per-category thresholds**
- Category-aware paper: at 0.80, dense code embeddings produce 15% false matches (e.g., sort_ascending vs sort_descending), and 0.90 cuts that to 3%. Sparse conversational embeddings at 0.80 miss paraphrases, and 0.75 captures them without more false positives. (Illustrative; dataset not named.) — [arXiv 2510.26835](https://arxiv.org/abs/2510.26835)
- AWS guidance table:

  | Threshold | Hit rate | Use for |
  |---|---|---|
  | 0.95 | ~25% | medical / legal / finance |
  | 0.90 | ~55% | general chatbots |
  | 0.80 | ~75% | FAQ / IT support |
  | 0.75 | ~90% | high-volume repetitive |

  AWS advises starting at 0.90–0.95, lowering gradually while monitoring accuracy, and A/B testing. — [AWS best practices](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)
- Portkey: start at 0.90–0.95 (Portkey uses ~0.95); domain systems (medical, legal, technical) need stricter thresholds; backtest on ~5,000 queries. — [Portkey thresholds blog](https://portkey.ai/blog/semantic-caching-thresholds/)
- VentureBeat production thresholds by query type: 0.88–0.97. — [VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)

**Labelled datasets usable for threshold sweeps**
- **Quora Question Pairs**: 404,290 labelled pairs (`pair-class` config; 149,263 positive pairs in `pair`). — [HF sentence-transformers/quora-duplicates](https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
  - The Redis paper used a Quora split of 323,491 train / 53,486 eval and a medical duplicate set (2,438 / 610). — [arXiv 2504.02268](https://arxiv.org/abs/2504.02268)
- **vCache SemBenchmark suite** (on HF): SemBenchmarkLmArena, SemBenchmarkSearchQueries, SemBenchmarkClassification, SemBenchmarkCombo.
  - SemBenchmarkLmArena (Apache-2.0): 3,500 distinct LM-Arena prompts sampled from 100,000 real user queries, each with 1–23 GPT-4.1-nano paraphrases (60,000 prompts total).
  - Each prompt carries a **class ID**, and a hit is correct if the retrieved entry is in the same class.
  - **Pre-generated responses from GPT-4.1-nano and GPT-4o-mini are included.** — [HF vCache/SemBenchmarkLmArena](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena)
- **Redis LangCache SentencePairs v3**: 40,004,529 train / 10,789 validation / 74,265 test pairs.
  - Sub-configs include `qqp`, `paws`, `mrpc`, `stsb`, `sick`, `parade`, `tapaco`, `apt`, `chatgpt-paraphrases`, `llm-paraphrases` and others. — [HF redis/langcache-sentencepairs-v3](https://huggingface.co/datasets/redis/langcache-sentencepairs-v3)
- **PAWS** (adversarial paraphrase pairs with high lexical overlap): useful for word-swap and entity-swap hard negatives. — [HF google-research-datasets/paws](https://huggingface.co/datasets/google-research-datasets/paws)
- **LMSYS / MOSS logs**: SCALM used LMSYS (300k conversations) and MOSS (1M) with cosine similarity plus GPT-4 judgement, not a pre-labelled duplicate set. — [SCALM](https://arxiv.org/abs/2406.00025)

**Is 0.8–0.95 realistic?**
- Published optima depend on the embedding model:

  | Embedding model | Threshold | Source |
  |---|---|---|
  | paraphrase-albert-small-v2 | 0.7 | [GPTCache paper](https://aclanthology.org/2023.nlposs-1.24.pdf) |
  | Albert | 0.78 | [MeanCache](https://arxiv.org/abs/2403.02694) |
  | MPNet | 0.83 | [MeanCache](https://arxiv.org/abs/2403.02694) |
  | all-MiniLM-L6-v2 | 0.8 | [GPT Semantic Cache](https://arxiv.org/abs/2411.05276) |
  | text-embedding-3-small | 0.90 | [SCALM](https://arxiv.org/abs/2406.00025) |
  | Portkey default | 0.95 | [Portkey docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic) |
  | RedisVL default | distance 0.1 ≈ sim 0.9 | [RedisVL](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html) |
  | LiteLLM doc examples | 0.8 / 0.7 | [LiteLLM](https://docs.litellm.ai/docs/caching/all_caches) |

- MeanCache: "The optimal threshold τ values varies with the embedding model." — [MeanCache](https://arxiv.org/abs/2403.02694)

### Inferences
- **Recommended CostGuard evaluation protocol (budget-friendly).**
  1. Use SemBenchmarkLmArena, a subset such as 5–10k prompts. Correctness comes free via class IDs, and responses are pre-generated, so the LLM spend for the sweep is ~$0.
  2. Use QQP and PAWS pairs for a pairwise ROC/PR analysis of the embedding model (precision and recall at each τ).
  3. Stream prompts into an empty cache in random order, as AWS and vCache did. For τ in 0.70–0.99 (step 0.01), record hit rate, per-hit precision, per-request error (FP/n) and estimated $ saved. Plot the hit-rate vs false-hit-rate curve and P-CHR.
  4. Repeat for 2–3 embedding models (MiniLM, bge-small, text-embedding-3-small).
  5. Optionally fit per-category thresholds.
  6. Optionally implement a simplified vCache-style per-entry sigmoid, so there is a "static vs adaptive" comparison an examiner will recognise.
- Report thresholds with the embedding model named, and state whether the number is similarity or distance.
- A small LLM-judge pass is affordable for a few hundred sampled hits on real or open-ended data, matching Portkey's "sample and evaluate" advice. Use class IDs wherever possible to avoid judge noise.

### Gaps
- No publicly labelled duplicate-pair set derived *directly* from LMSYS-Chat-1M was found. SemBenchmarkLmArena (LM-Arena prompts plus synthetic paraphrases) is the closest.
- No source gives a universal mapping between embedding models' cosine-similarity scales. Calibration has to be empirical.

---

## 5. Embedding models for cache keys (cost, latency, dimension)

### Takeaway
Small local bi-encoders are good enough for a student build and cost $0:
- all-MiniLM-L6-v2: 22.7M params, 384-d, Apache-2.0.
- bge-small-en-v1.5: 33.4M params, 384-d, MIT.
- Redis's caching-tuned langcache-embed-v3-small: 22.6M params, 384-d, Apache-2.0, built from MiniLM.

Hosted embeddings are also cheap: text-embedding-3-small at ~$0.02 per 1M tokens, and gemini-embedding-001 at $0.15 per 1M with a free tier. The literature shows small *domain-tuned* models can match or beat large general models on cache precision. What matters most is calibrating the threshold per model.

### Cited Findings

**Local and caching-tuned models**
- all-MiniLM-L6-v2: 22,713,728 params, hidden size 384, Apache-2.0, ~240M downloads. — [HF model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- bge-small-en-v1.5: 33,360,512 params, hidden size 384, MIT. — [HF model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- redis/langcache-embed-v3-small: 22,565,376 params, 384-d, Apache-2.0, fine-tuned from all-MiniLM-L6-v2 on 8M paraphrase pairs (Dec 2025).
  - On the LangCache SentencePairs v3 test set it gets PR-AUC 0.833 / P-CHR AUC 0.445, against BGE-base-en-v1.5 at 0.660 / 0.372. — [HF](https://huggingface.co/redis/langcache-embed-v3-small); [arXiv 2606.19719](https://arxiv.org/abs/2606.19719)
- redis/langcache-embed-v2: 149M params (ModernBERT), 768-d. — [HF langcache-embed-v2](https://huggingface.co/redis/langcache-embed-v2)

**Hosted models**
- OpenAI text-embedding-3-small: $0.02 per 1M tokens ($0.01 via Batch) per a 2026 pricing guide (secondary source). — [Layer3Labs pricing guide](https://www.layer3labs.io/guides/openai-embedding-models-pricing)
- gemini-embedding-001: $0.15 per 1M input tokens, with free and paid tiers in the Gemini API (per Google's GA announcement, as summarised in search results). — [Google Developers Blog](https://developers.googleblog.com/gemini-embedding-available-gemini-api/)

**Head-to-head numbers**
- Redis medical-duplicate benchmark (precision / recall / F1):

  | Model | Precision | Recall | F1 |
  |---|---|---|---|
  | text-embedding-3-small | 0.83 | 0.89 | 0.86 |
  | text-embedding-3-large | 0.85 | 0.87 | 0.86 |
  | ada-002 | 0.77 | 0.90 | 0.83 |
  | Cohere embed-english-v3 | 0.78 | 0.83 | 0.81 |
  | gte-modernbert-base | 0.78 | 0.89 | 0.84 |
  | multilingual-e5-large-instruct | 0.87 | 0.82 | 0.84 |
  | Linq-Embed-Mistral (7B) | 0.84 | 0.93 | 0.88 |
  | Fine-tuned ModernBERT (synthetic data only) | 0.87 | 0.90 | 0.89 |

  Over-training (6 epochs) gave +22% precision in-domain but −8% precision out-of-domain. — [arXiv 2504.02268](https://arxiv.org/abs/2504.02268)
- Latency per query embedding (MeanCache): MPNet ~0.009 s, Albert ~0.005 s, Llama-2 much slower.
  - Storage: Llama-2 embeddings ~32 KB per query against ~6 KB for MPNet/Albert. PCA compression cut storage by 83%. — [MeanCache](https://arxiv.org/abs/2403.02694)
- vCache experiments used GTE-large-en-v1.5, E5-large-v2 and text-embedding-3-small. vCache's gains held across all three. — [vCache](https://arxiv.org/abs/2502.03771)
- GPT Semantic Cache used all-MiniLM-L6-v2 with threshold 0.8. — [arXiv 2411.05276](https://arxiv.org/abs/2411.05276)
- Production report: query embedding 12 ms p50 (model unspecified). — [VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)

### Inferences
- **Budget math.** 60,000 benchmark prompts × ~30 tokens ≈ 1.8M tokens. That costs ≈ $0.04 with text-embedding-3-small and $0 with local MiniLM/bge-small. Embedding cost is not a constraint, so spend the $5–20 on LLM calls for live demos and judge sampling.
- **Recommended default.** Use a local 384-d model (MiniLM or langcache-embed-v3-small) for the cache key: zero cost, CPU-friendly, ~1.5 KB per vector. Keep text-embedding-3-small as a comparison arm. The examiner-friendly claim is "we calibrated τ per model and show the curves", not "model X is best".
- Use one embedding model per cache index and include its name and version in the cache metadata. Cosine scales differ across models, so mixing them silently invalidates the threshold.

### Gaps
- No primary OpenAI pricing page was fetched (only a 2026 secondary guide).
- Gemini free-tier rate limits for embeddings were not retrieved.
- No published latency benchmark of bge-small vs MiniLM on CPU *for caching* specifically was found.

---

## 6. Pitfalls and mitigations

### Takeaway
The dominant failure is a *confident wrong answer* on a near-duplicate that differs in negation, entity, number, constraint or context. Similarity is not validity. Other failure modes are:
- Multi-turn context leaks.
- Stale entries.
- Deliberate poisoning or key collision (86% hijack success reported in 2026).
- Cross-tenant leakage.
- Non-deterministic outputs being frozen.
- Cache keys that ignore the system prompt or model.

Real systems mitigate these with metadata/namespace scoping, exact keys on model + system prompt + params, per-category thresholds and TTLs, exclusion rules, LLM-judge or reranker verification (sync or async), and response-side integrity checks.

### Cited Findings

**Semantically close but different questions (negation / entities / numbers)**
- GPTCache authors: results "with semantics opposite to the input text are acceptable in search … but this is unacceptable in caching scenarios". — [GPTCache paper](https://aclanthology.org/2023.nlposs-1.24.pdf)
- 2026 poisoning paper: "Highly similar queries can require different answers when they differ in constraints, instructions…". It names the needed property "cache-hit validity" and argues that query embeddings lose the information needed to separate valid from invalid hits (an information-bottleneck view). — [arXiv 2609.35908](https://arxiv.org/abs/2609.35908)
- Code-domain example: sort_ascending vs sort_descending match at 0.80. — [arXiv 2510.26835](https://arxiv.org/abs/2510.26835)
- Mitigations:
  - Rerank with a cross-encoder before thresholding. — [arXiv 2606.19719](https://arxiv.org/abs/2606.19719)
  - Learn per-entry thresholds. — [vCache](https://arxiv.org/abs/2502.03771)
  - Canonicalise structured intent (W5H2 slots) so equivalent queries map to the same key and hits are "safe to execute". — [arXiv 2602.18922](https://arxiv.org/abs/2602.18922)
  - Use domain-tuned embeddings trained with hard negatives. — [arXiv 2504.02268](https://arxiv.org/abs/2504.02268)

**Personalised / context-dependent prompts and multi-turn**
- MeanCache stores a context chain with each cached query and checks the conversation history before returning a hit. On 450 synthetic contextual queries, precision was 0.98 against GPTCache's 0.66. — [MeanCache](https://arxiv.org/abs/2403.02694)
- AWS: for multi-turn, retrieve the key facts and recent messages first, then cache on "the combination of the current user message and the retrieved context, instead of embedding the entire raw dialogue". — [AWS best practices](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)
- Portkey limits semantic cache to ≤4 messages. — [Portkey docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic)
- Portkey's blog says multi-turn conversations need context-aware embeddings. — [Portkey thresholds blog](https://portkey.ai/blog/semantic-caching-thresholds/)
- vCache notes semantic caches are effective mainly for "single-turn interactions with short to medium context". — [vCache](https://arxiv.org/abs/2502.03771)
- The production report excludes personalised, time-sensitive and transactional responses from caching. — [VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)

**Stale data**
- TTL by data type and topic-based invalidation (AWS, see §1). — [AWS best practices](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)
- Production practice:
  - TTLs: pricing 4 h, product info 1 day, policies 7 days, FAQs 14 days.
  - Event-based invalidation on content updates.
  - Periodic staleness checks. — [VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)
- Per-category TTLs. — [arXiv 2510.26835](https://arxiv.org/abs/2510.26835)

**Cache poisoning / key collisions (2026 security literature)**
- "CacheAttack" treats semantic keys as fuzzy hashes. It argues that the locality needed for hit rate conflicts with collision resistance, and reaches an **86% hit rate in LLM response hijacking**, transferring across embedding models. A financial-agent case study is included. — [arXiv 2601.23088](https://arxiv.org/abs/2601.23088)
- Attacks have been demonstrated against *commercial* semantic caches in black-box settings (Wu et al. 2026, cited by the poisoning-defense paper). The proposed defence uses "Deletion Gain" plus an "Answer Check" on the raw cached-query text. It blocks **82.0–98.2% of poisoned entries at a 5% false-positive rate**, with negligible serving overhead. — [arXiv 2609.35908](https://arxiv.org/abs/2609.35908)
- LaCache moves the integrity check from query to response: it also checks the first k speculatively decoded tokens. It reduces attack hit rate "to near zero while preserving over 90% of benign cache utility". It argues that query-side filters (perplexity, consistency, LLM classifiers) can be evaded by GCG-optimised fluent queries. — [arXiv 2608.01718](https://arxiv.org/abs/2608.01718)

**Multi-tenant isolation and privacy**
- RedisVL `filterable_fields`: store with `filters={"user_id": "abc"}` and check with the same filter "prevents cross-tenant leakage". — [RedisVL](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html)
- Gateway equivalents: Portkey `cache_namespace` for per-user partitioning; Helicone `Helicone-Cache-Seed`; Cloudflare includes the provider auth header in the key. — [Portkey](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic); [Helicone](https://docs.helicone.ai/features/advanced-usage/caching); [Cloudflare](https://developers.cloudflare.com/ai-gateway/features/caching/)
- The key-collision paper notes that the fuzzy match "improves hit rates across tenants", which is exactly what creates the cross-user attack surface. — [arXiv 2601.23088](https://arxiv.org/abs/2601.23088)
- MeanCache's privacy answer is a per-user local cache with federated-learned embeddings, so queries never leave the device. — [MeanCache](https://arxiv.org/abs/2403.02694)
- Provider prompt caches: Anthropic never shares caches between organisations and isolates them per workspace on the Claude API. OpenAI caches "are not shared across organizations". — [Anthropic docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching); [OpenAI docs](https://developers.openai.com/api/docs/guides/prompt-caching)

**Temperature / non-determinism**
- Helicone's `Helicone-Cache-Bucket-Max-Size` stores multiple responses for non-deterministic prompts (default 1). — [Helicone](https://docs.helicone.ai/features/advanced-usage/caching)
- LiteLLM's key includes temperature, and Portkey's key includes temperature and all body parameters. A request with different sampling params therefore misses. — [LiteLLM](https://docs.litellm.ai/docs/caching/all_caches); [Portkey](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic)

**System prompt and model in the key**
- **Portkey's semantic cache ignores the system prompt**: "changing it won't affect cache hits". Model and other body params must match exactly. — [Portkey docs](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic)
- Cloudflare's key includes model, endpoint and full body (and so the system prompt). LiteLLM's key includes model and messages. — [Cloudflare](https://developers.cloudflare.com/ai-gateway/features/caching/); [LiteLLM](https://docs.litellm.ai/docs/caching/all_caches)

**Verification of hits**
- Synchronous LLM judge: GPT Semantic Cache used GPT-4o-mini to validate hits (offline evaluation). — [arXiv 2411.05276](https://arxiv.org/abs/2411.05276)
- vCache uses an LLM judge to label exploration samples, asynchronously. — [vCache](https://arxiv.org/abs/2502.03771)
- Krites runs an async judge only on grey-zone near-misses, with no critical-path latency. It notes that putting an LLM judge directly on the serving path "adds latency and compute to borderline requests, eroding the main benefit of caching". — [Krites](https://arxiv.org/abs/2602.13165)
- Portkey: "failures are invisible by default". A bad hit returns a confident wrong answer with 200 OK, so false positives must be estimated by sampling with humans or LLM judges. — [Portkey thresholds blog](https://portkey.ai/blog/semantic-caching-thresholds/)

### Inferences
- **Minimal mitigation set CostGuard can build in 5 days.**
  - Exact-key components: hash of (model, system prompt, tools, sampling params).
  - Metadata filters: tenant_id and namespace.
  - Do-not-cache rules: personalised, time-sensitive, or requests with temperature above a cutoff unless they opt in.
  - Per-category TTL.
  - A cheap "entity/number guard": refuse a hit if the sets of numbers or named entities (regex or spaCy) differ between query and cached query. **This guard is our own suggestion, not a published method**, but it directly targets the negation and entity failures above.
  - Optional async LLM-judge sampling of a few % of hits to estimate the live false-hit rate.
- For the demo, build a small "adversarial near-duplicates" test set: negation pairs, swapped entities, changed numbers, plus PAWS. Show the false-hit rate before and after the guard. That is a strong, examiner-friendly result.
- Do not copy Portkey's "ignore system prompt" behaviour. In a drop-in multi-app layer, two apps with different system prompts would otherwise share answers.

### Gaps
- No published benchmark isolating negation-specific false-hit rates was found.
- No published, evaluated "entity check" component for semantic caches was found (hence the guard above is labelled our own suggestion).

---

## 7. Semantic caching vs provider-native prompt caching, and combining them

### Takeaway
Provider prompt caching (Anthropic, OpenAI, Gemini) is *exact-prefix* and *lossless*. It discounts repeated input-prefix tokens (reads ≈ 0.1× input price), but you still pay for the uncached suffix and all output tokens, and it needs a long prefix (≥512–4,096 tokens depending on model). Semantic caching skips the LLM call entirely (100% saving on that request, including output), but it is lossy: false hits are possible. The two are complementary. Order the pipeline as exact-match cache, then semantic cache, then an LLM call whose prompt is structured with a stable prefix to maximise provider prompt-cache hits.

### Cited Findings

**Anthropic**
- Prefix-based caching in the order tools → system → messages.
- Explicit `cache_control` breakpoints (max 4) or automatic caching.
- Default 5-minute TTL; 1-hour option.
- Pricing: writes 1.25× base input (5 min) or 2× (1 h); reads **0.1×** base input, with some newer models lower (0.05× on Opus 5.5).
- Minimum cacheable length 512–4,096 tokens depending on model.
- Hits require 100% identical prompt segments. Changing tool definitions invalidates everything; changing tool_choice or images invalidates the messages level.
- Usage fields: `cache_creation_input_tokens`, `cache_read_input_tokens`. — [Anthropic prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

**OpenAI**
- Automatic prompt caching; minimum 1,024 tokens for GPT-5.6+.
- Cached input at 0.1× (0.05× on GPT-6.1 Sol); writes 1.25× on GPT-5.6+.
- In-memory retention "typically 5 to 10 minutes of inactivity, up to one hour"; extended 24 h on some models.
- Routing by hash of the initial tokens plus optional `prompt_cache_key`.
- Usage field `usage.input_tokens_details.cached_tokens`.
- Advice: put stable instructions first. — [OpenAI prompt caching docs](https://developers.openai.com/api/docs/guides/prompt-caching)

**Gemini**
- Implicit caching is on by default for Gemini 2.5+.
- Minimum 2,048 tokens (2.5 Flash/Pro) and 4,096 tokens (newer 3.x models listed).
- Savings are passed on automatically.
- Advice: put large common content at the start and send similar-prefix requests close together.
- Explicit context caching exists, but its TTL and storage pricing were not on the fetched page. — [Gemini caching docs](https://ai.google.dev/gemini-api/docs/caching)

**Production example of prompt caching**
- ProjectDiscovery raised its prompt-cache hit rate from **7% to 84%** by moving dynamic working memory out of the system prompt into a trailing user message. The change cut LLM cost 59% overall (66–70% after further optimisation), with 9.8B tokens served from cache. — [ProjectDiscovery blog](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)

**Mechanism contrast**
- vCache contrasts semantic caching with exact string-match caching. The key-collision and poisoning papers contrast it with KV/prefix caching: "Unlike KV caching, which reuses intermediate states for matching token prefixes, semantic caching can reuse responses across queries". — [arXiv 2609.35908](https://arxiv.org/abs/2609.35908); [arXiv 2601.23088](https://arxiv.org/abs/2601.23088)

### Inferences
- **Per-request savings (illustrative).**
  - Semantic hit: saves 100% of input and output cost minus the embedding cost (≈$0).
  - Provider prompt-cache hit on a 90%-cached prefix with 0.1× read price: saves ≈81% of *input* cost and 0% of output.
  - So for chat-style workloads with short prompts and long answers, semantic caching dominates when it hits. For RAG or agent workloads with long, stable system prompts and tool definitions, prompt caching is the safer, lossless lever.
- **Combined design for CostGuard.**
  1. Exact-hash tier.
  2. Semantic tier, with filters and guards.
  3. On a miss, forward to the provider with prompt-caching-friendly ordering (static system prompt and tools first, dynamic content last; `cache_control` for Anthropic; `prompt_cache_key` for OpenAI).
  4. Log `cached_tokens` / `cache_read_input_tokens` so the dashboard shows both savings sources separately.

  This also gives the examiner a clean ablation: no cache / exact only / +semantic / +prompt-cache structuring.
- Prompt caching needs ≥512–4,096-token prefixes, so many short student test prompts will never trigger it. The demo should include a long-system-prompt scenario to show it.

### Gaps
- The Gemini explicit context-caching storage price and discount percentage were not retrieved from the fetched page.
- Model names and multipliers above are taken verbatim from the docs as fetched on 2026-10-03 and may change.
