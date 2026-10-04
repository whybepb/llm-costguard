# Prove CostGuard's savings, not its features

**LLM CostGuard should compete on evidence, not on how many features it has.** Every lever in its brief already has mature, free tooling: LLMLingua-2 for compression, RedisVL or a hand-built index for the semantic cache, small cross-encoders for reranking, RouteLLM-style routers, and LiteLLM or native SDKs for provider adapters. The problem is that the published savings behind these tools shrink sharply on realistic traffic. Cache benchmarks built from generated paraphrases report 50–90% hit rates, yet only **4.5–7.5% of real LMSYS/MOSS chat queries** can be answered from an earlier similar query ([SCALM](https://arxiv.org/abs/2406.00025)). RouteLLM's **85%** saving holds on MT-Bench but falls to **35–45%** on GSM8K and MMLU ([LMSYS](https://lmsys.org/blog/2024-07-01-routellm/)). LLMLingua-2 loses almost nothing on QA at 3x compression but drops summary BLEU by **22%** ([LLMLingua-2](https://arxiv.org/html/2403.12968)). So the real deliverable is the team's own measured, quality-bounded savings frontier, built from four things. The first is a threshold calibrated on labelled pairs, with false hits reported per request. The second is a paired evaluation with bootstrap confidence intervals for every lever. The third is dollar costs priced from each provider's `usage` object, with input and output priced separately. The fourth is one frozen request trace replayed through ablation arms, so savings are credited to the right lever. At October 2026 list prices, a GPT-5.4-mini → GPT-5-nano model pair gives a ~12x price gap and keeps the whole experiment near **$5**. A free-tier stack covers the rest: a thin FastAPI proxy, local embeddings and FAISS for sweeps, Qdrant Cloud for the live cache, LLMLingua-2 run offline, Langfuse, Streamlit, GitHub Actions on a public repo and a pre-warmed Render host. It holds as long as sweeps stay off metered quotas. The presentation rubric gives **30% to architecture** and **50% to framing, trade-offs and delivery**, none of which needs new code. The plan: freeze the trace today, freeze the code Wednesday night, and spend Friday and Saturday on the deck and Q&A.

## Published savings shrink by half or more on realistic traffic

Each of the brief's five LLMOps bullets rests on real research. Nearly every headline number, though, was produced under conditions CostGuard will not reproduce. The table pairs each lever's best-known claim with the most realistic anchor in the literature.

| Lever | Headline claim | Realistic anchor | Why the gap exists |
|---|---|---|---|
| Semantic cache | 61.6–68.8% hit rate with >97% valid hits ([GPT Semantic Cache](https://arxiv.org/abs/2411.05276)); 87.6% hits at τ=0.80 ([AWS](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-benchmarks.html)) | 4.5% (MOSS) and 7.5% (LMSYS) of real queries reusable ([SCALM](https://arxiv.org/abs/2406.00025)); ~20% hit rate at 99% accuracy on Q&A/RAG ([Portkey](https://portkey.ai/blog/reducing-llm-costs-and-latency-semantic-cache/)) | Benchmarks are built by paraphrasing seed prompts, so nearly every query has a close neighbour |
| Model routing | >85% cost cut at 95% of GPT-4 quality ([RouteLLM](https://lmsys.org/blog/2024-07-01-routellm/)); "up to 98%" ([FrugalGPT](https://arxiv.org/abs/2305.05176)) | 45% on MMLU and 35% on GSM8K (RouteLLM); ~35% at <2% accuracy loss for the best open routers ([RouterArena](https://arxiv.org/html/2510.00202v1)) | MT-Bench is forgiving to weak models; FrugalGPT used 2023 API price gaps |
| Prompt compression | "Up to 20x compression with minimal performance loss" ([LLMLingua](https://github.com/microsoft/LLMLingua)) | At 3.1x, QA exact match drops 0.8 points but summary BLEU falls 22.34 → 17.37; LongBench falls 44.0 → 39.1 under a 2,000-token budget ([LLMLingua-2](https://arxiv.org/html/2403.12968)) | The 20x figure came from GSM8K few-shot demonstrations, which are highly redundant |
| Production cache | API bill −73%, hit rate 18% → 67% ([VentureBeat op-ed](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)) | No independently audited production dataset exists | Self-reported by an unnamed company |

Two of these gaps hide metric traps that are worth naming in the presentation. AWS reports cached-answer "accuracy" of about 92% at every threshold from 0.75 to 0.99, which looks flat. Measured as errors over *all* requests, the picture changes. A 0.99 threshold serves a wrong answer to about **1.9%** of traffic (23.5% hits × 7.9% wrong), while 0.80 does so for about **7.2%**, roughly one request in fourteen ([AWS](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-benchmarks.html)). Per-hit precision flatters aggressive thresholds. Per-request error, which vCache defines as false positives divided by all prompts, does not ([vCache](https://arxiv.org/abs/2502.03771)). The second trap is RouteLLM's "3.66x". It is the ratio of strong-model calls a random router needs to recover half the weak-to-strong quality gap on MT-Bench versus what its matrix-factorisation router needs (49.03% vs 13.40%). It is not a 3.66x saving over sending everything to GPT-4; the ">85%" figure is that comparison ([RouteLLM paper](https://arxiv.org/html/2406.18665)). Quote both numbers and name each one's baseline.

The gaps also show where CostGuard can produce new evidence. Every LLMLingua-family result was measured against GPT-3.5-Turbo or Mistral-7B, and there are no published numbers against 2025–2026 frontier API models ([LLMLingua-2](https://arxiv.org/html/2403.12968)). The research found no benchmark that isolates negation-driven false hits in semantic caches, and no published curve linking score-threshold dynamic-k to answer quality. The widely repeated "~30% of queries repeat" also needs care: it traces back to MeanCache's per-user study of 20 academic ChatGPT users, not to a service-wide measurement ([MeanCache](https://arxiv.org/abs/2403.02694)). A team that measures these things on its own trace, and includes a 0%-duplicate arm where every saving must come from compression and routing, will have evidence no other team in the session has.

## Semantic caching: calibrate the threshold, then guard it

### Own the threshold logic and benchmark against GPTCache and LiteLLM

A semantic cache works like this. It embeds the incoming prompt and finds the nearest stored neighbour. If cosine similarity clears a threshold τ, it returns the stored response. Otherwise it calls the LLM and writes the new embedding and response back ([vCache](https://arxiv.org/abs/2502.03771)). Production designs put deterministic tiers first. Apple's Krites paper describes the typical deployment as a curated static cache, then a dynamic online cache, then the LLM ([Krites](https://arxiv.org/abs/2602.13165)). In one production workload, 18% of 100,000 queries were exact duplicates, 47% were semantically similar and 35% were novel ([VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)).

The 2025–2026 research targets the single weak point: the static τ.

- **vCache** (ICLR 2026) learns, for each cached entry, a sigmoid estimate of P(correct | similarity) and holds a user-set error bound. It reports up to **12.5x higher hit rate and 26x lower error** than static-threshold GPTCache ([vCache](https://arxiv.org/abs/2502.03771)).
- **Krites** sends grey-zone near-misses to an asynchronous LLM judge. It serves up to 3.9x more curated answers at a fixed error rate, with no added latency on the serving path ([Krites](https://arxiv.org/abs/2602.13165)).
- **Category-aware caching** sets τ, TTL and quota separately for each request category ([arXiv 2510.26835](https://arxiv.org/abs/2510.26835)).
- **A Redis/NYU EMNLP 2026 paper** shows that "models with the highest PR-AUC are often the worst in operation". It proposes P-CHR AUC instead: precision integrated over the cache-hit ratio as τ varies ([arXiv 2606.19719](https://arxiv.org/abs/2606.19719)).

| Tool | Semantic? | Status (checked 3 Oct 2026) | Role in CostGuard |
|---|---|---|---|
| GPTCache | Yes | MIT; last release 0.1.44 on 1 Aug 2024 | The classic baseline every paper compares against |
| RedisVL SemanticCache | Yes | MIT; v0.27.2 (Sep 2026); `distance_threshold` default 0.1 (cosine *distance*, about similarity 0.9); per-tenant `filterable_fields`; TTL | Reference design for filters and TTL |
| LiteLLM proxy cache | Yes (`redis-semantic`, `qdrant-semantic`), plus exact | v1.103.2 (1 Oct 2026); embedding model defaults to `text-embedding-ada-002` | Off-the-shelf comparison arm |
| Portkey | Exact on all plans; semantic only on select Enterprise plans | Default 0.95; at most 4 messages; ignores the system prompt | Design choice to avoid copying |
| Helicone, Cloudflare AI Gateway | Exact only | Managed | Reference points for the exact tier |
| vCache | Yes, learned per-entry thresholds | CC BY-NC-ND 3.0 (non-commercial, no derivatives) | Reimplement the idea; don't vendor the code |

Sources: [GPTCache](https://github.com/zilliztech/GPTCache), [RedisVL](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html), [LiteLLM](https://docs.litellm.ai/docs/caching/all_caches), [Portkey](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic), [Helicone](https://docs.helicone.ai/features/advanced-usage/caching), [Cloudflare](https://developers.cloudflare.com/ai-gateway/features/caching/), [vCache repo](https://github.com/vcache-project/vCache).

For a five-day build, either hand-roll the cache or wrap RedisVL. A hand-rolled cache would use sentence-transformers with an in-process FAISS or numpy index for sweeps, and Qdrant for the live demo. Either way, the threshold decision, similarity scores and false-hit handling stay visible in the team's own code and logs. Run GPTCache's classic configuration (ALBERT embeddings, τ=0.7) or LiteLLM's `redis-semantic` at 0.8 as the comparison arm. Always report thresholds with the embedding model named and the unit stated. RedisVL's default *distance* of 0.1 equals a *similarity* of 0.9 under Redis's 1 − cosine convention ([RedisVL](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html)), and an examiner will catch mixed units.

### Thresholds are model-specific, and static ones leak more errors as the cache fills

Published optimal thresholds vary widely because each embedding model has its own cosine scale:

| Embedding model | Optimal τ | Source |
|---|---|---|
| paraphrase-albert-small-v2 | 0.70 | [GPTCache paper](https://aclanthology.org/2023.nlposs-1.24.pdf) |
| ALBERT | 0.78 | [MeanCache](https://arxiv.org/abs/2403.02694) |
| MPNet | 0.83 | [MeanCache](https://arxiv.org/abs/2403.02694) |
| all-MiniLM-L6-v2 | 0.80 | [GPT Semantic Cache](https://arxiv.org/abs/2411.05276) |
| text-embedding-3-small | 0.90 | [SCALM](https://arxiv.org/abs/2406.00025) |
| Portkey default | 0.95 | [Portkey](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic) |

MeanCache found that GPTCache's suggested 0.7 produced **233 false hits on 700 unique queries**, against 89 with its own tuned setting ([MeanCache](https://arxiv.org/html/2403.02694v3)).

Within a single domain, the right τ also changes by category. Dense code embeddings at 0.80 produce about 15% false matches (sort ascending vs sort descending), and 0.90 cuts that to 3% ([arXiv 2510.26835](https://arxiv.org/abs/2510.26835)). One production team runs thresholds from 0.88 for search up to 0.97 for transactional queries ([VentureBeat](https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73)).

More worrying, vCache shows that a static threshold's error rate *grows with traffic*: the more entries the cache holds, the more chances a wrong neighbour lands closest. On 150,000 search queries, even τ=0.99 still produced **1.7% errors** ([vCache](https://arxiv.org/abs/2502.03771)). Practitioner guidance therefore starts strict. AWS suggests 0.90–0.95 and lowering gradually ([AWS](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)). Portkey suggests 0.95, backtested on about 5,000 queries until accuracy stays above 99% ([Portkey](https://portkey.ai/blog/semantic-caching-thresholds/)).

### The full threshold evaluation costs almost nothing

The cheapest rigorous protocol uses data that already has correctness labels. vCache's SemBenchmarkLmArena (Apache-2.0) has 60,000 prompts: 3,500 distinct LM-Arena prompts plus GPT-4.1-nano paraphrases. Each prompt carries a class ID, and a hit is correct exactly when the retrieved entry has the same class. It also ships pre-generated responses from GPT-4.1-nano and GPT-4o-mini, so a full threshold sweep needs **zero LLM calls** ([SemBenchmarkLmArena](https://huggingface.co/datasets/vCache/SemBenchmarkLmArena)).

Run the sweep the way AWS and vCache did:

1. Stream a 5,000–10,000-prompt subset, in random order, into an empty cache.
2. For τ from 0.70 to 0.99 in steps of 0.01, record hit rate, per-hit precision, per-request error and dollars saved.
3. Plot error against the number of requests served, not just the final value.
4. Add a pairwise ROC analysis on Quora Question Pairs (404,290 labelled pairs) and on PAWS, whose adversarial pairs share most words but differ in meaning ([QQP](https://huggingface.co/datasets/sentence-transformers/quora-duplicates), [PAWS](https://huggingface.co/datasets/google-research-datasets/paws)).
5. Because the deliverables require a hand-written eval set, add a team-written set of near-miss traps: negations, swapped entities, changed numbers, and "cancel order" versus "track order".
6. Repeat with two or three embedding models.

All three candidate models run free on CPU:

- all-MiniLM-L6-v2: 22.7M parameters, 384 dimensions, Apache-2.0.
- bge-small-en-v1.5: 33.4M parameters, MIT.
- Redis's cache-tuned langcache-embed-v3-small: 22.6M parameters, built from MiniLM. On LangCache sentence pairs it scores PR-AUC 0.833, against 0.660 for BGE-base.

Sources: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5), [langcache-embed-v3-small](https://huggingface.co/redis/langcache-embed-v3-small), [arXiv 2606.19719](https://arxiv.org/abs/2606.19719).

Choose the operating point by weighting false hits far more heavily than misses. Optionally, add a simplified vCache-style per-entry sigmoid so the examiner sees a static-versus-adaptive comparison.

One operational constraint matters: whichever embedding model τ was calibrated on is the one the live proxy must use. If the deployed host cannot hold a local model, calibrate on text-embedding-3-small instead. At $0.02 per million tokens, the entire 60,000-prompt benchmark costs about $0.04 to embed ([OpenAI pricing](https://developers.openai.com/api/docs/pricing)).

### Confident wrong answers, leaks and poisoning are the failure modes

The main failure is a near-duplicate that differs in a negation, entity, number or constraint and gets a confidently wrong cached answer, delivered with a 200 OK. Portkey's operators note that such failures "are invisible by default" and can only be estimated by sampling ([Portkey](https://portkey.ai/blog/semantic-caching-thresholds/)). A 2026 security paper argues that query embeddings structurally lose the information needed to tell valid hits from invalid ones ([arXiv 2609.35908](https://arxiv.org/abs/2609.35908)). "CacheAttack" reaches an **86% response-hijack rate** by crafting colliding queries that transfer across embedding models ([arXiv 2601.23088](https://arxiv.org/abs/2601.23088)). Two defences exist. One checks the cached query's raw text and blocks 82–98% of poisoned entries at a 5% false-positive rate ([arXiv 2609.35908](https://arxiv.org/abs/2609.35908)). LaCache instead checks the first few speculatively decoded response tokens ([arXiv 2608.01718](https://arxiv.org/abs/2608.01718)).

Multi-turn prompts are the second trap. MeanCache's context chains raised precision on contextual queries from 0.66 to 0.98 ([MeanCache](https://arxiv.org/abs/2403.02694)). AWS advises caching on the current message plus retrieved key facts rather than on the raw dialogue ([AWS](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)). vCache says semantic caches work mainly for single-turn prompts of short to medium length ([vCache](https://arxiv.org/abs/2502.03771)).

Key design is the third trap. Portkey's semantic cache ignores the system prompt ([Portkey](https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic)). In a drop-in layer shared by several apps, that would let two applications with different system prompts serve each other's answers.

CostGuard's minimal set of mitigations:

- **Exact key** over model, system prompt, tools and sampling parameters.
- **Tenant and namespace filters**, which RedisVL documents as preventing cross-tenant leakage ([RedisVL](https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html)).
- **Do-not-cache rules** for personalised, time-sensitive and high-temperature requests.
- **Per-category TTLs**, following AWS's guidance of 24 hours for static facts, 1–4 hours for general answers and 5–15 minutes for real-time data ([AWS](https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html)).
- **Asynchronous LLM-judge sampling** of a few percent of hits.
- **An entity-and-number guard** that refuses a hit when the numbers or named entities differ between the new query and the cached one. This guard is the research notes' own proposal, not a published method, so the before/after false-hit rate on the trap set would be a new result.

### Semantic caching and provider prompt caching stack in a fixed order

The two caches behave differently. Provider prompt caching only matches an exact prefix, but it never changes the answer. Semantic caching skips the whole call, but can return a wrong answer. Provider caching details:

- **Anthropic:** cache reads cost 0.1x base input (0.05x on Opus 5.5) after a 1.25x write. The minimum cacheable prefix is 512–4,096 tokens depending on model ([Anthropic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)).
- **OpenAI:** caches automatically from 1,024 tokens on GPT-5.6 and later, at 0.1x ([OpenAI caching](https://developers.openai.com/api/docs/guides/prompt-caching)).
- **Gemini:** implicit caching starts at 2,048–4,096 tokens, and cache hits bill at 10% of the input price ([Gemini caching](https://ai.google.dev/gemini-api/docs/caching), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)).

Prefix caching discounts only the repeated input, never the output. A semantic hit saves both. Prompt layout matters more than any setting: ProjectDiscovery raised its prompt-cache hit rate from **7% to 84%** just by moving dynamic working memory out of the system prompt, and cut its LLM cost by 59% ([ProjectDiscovery](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)).

CostGuard's order should therefore be:

1. Exact-hash cache.
2. Semantic cache, with its guards.
3. On a miss, a provider call whose prompt puts the static system text and tools first and volatile content last.

Log `cached_tokens` and `cache_read_input_tokens` so the dashboard shows the two savings sources separately. Many short test prompts never reach the minimum cacheable length, so the demo needs one long-system-prompt scenario to show this lever at all.

## Compression and context trimming pay only on long, volatile input

### LLMLingua-2 is the only practical drop-in token compressor for closed APIs

A layer that sits in front of OpenAI, Anthropic or Gemini can only use compressors that take text in and return text. Gist tokens and AutoCompressors compress prompts into learned vectors and need access to the model's weights ([Gisting](https://arxiv.org/abs/2304.08467), [AutoCompressors](https://arxiv.org/abs/2305.14788)). They belong on a related-work slide, not in the build.

The original LLMLingua and LongLLMLingua drop tokens using a small language model's perplexity, and need a 2.7–7B model to do it. LLMLingua-2 instead treats compression as a keep/drop decision for each token, made by an XLM-RoBERTa-large or multilingual-BERT encoder. That encoder was trained on GPT-4 compressions of MeetingBank meeting transcripts. LLMLingua-2 installs with `pip install llmlingua` and peaks at **2.1 GB of GPU memory**, against 16.6 GB for LLMLingua. It runs 3–6x faster than earlier compressors, taking **0.4–0.5 s per prompt** on a V100 ([LLMLingua-2](https://arxiv.org/html/2403.12968), [README](https://github.com/microsoft/LLMLingua)).

The notes disagree on its size. The paper's memory figure implies a small model, but the Hugging Face model card lists about 0.6B parameters in F32, which is roughly 2.4 GB of weights. That is too large for a 512 MB–1 GB free host ([HF card](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank)). Measure it on a laptop on day one. Until then, plan to run it offline in the A/B harness and serve a lightweight fallback on the live host.

Provence (ICLR 2025) combines reranking and sentence-level pruning in one 430M-parameter DeBERTa cross-encoder, so pruning costs almost nothing extra. It is English-only, limited to 512 tokens, and licensed CC BY-NC-ND 4.0, which is acceptable for a student demo but not for a product ([Provence](https://huggingface.co/naver/provence-reranker-debertav3-v1)).

### Reranking with dynamic-k beats token dropping on retrieved context

Independent comparisons favour *selecting* whole passages over dropping individual tokens:

- Jha et al. found that selecting whole passages reaches up to **10x compression with minimal accuracy loss** and "often outperforms" token pruning ([Jha et al.](https://arxiv.org/abs/2407.08892)).
- RECOMP keeps 5–10% of tokens with under 10% relative QA drop ([RECOMP](https://arxiv.org/html/2310.04408)).
- LongLLMLingua's question-aware reordering improved NaturalQuestions by up to **21.4% with about 4x fewer tokens** ([LongLLMLingua](https://arxiv.org/abs/2310.06839)).
- A 2025 study across 13 datasets found that moderate compression can even *improve* LongBench results ([Zhang et al.](https://arxiv.org/abs/2505.00019)).

The cheapest strong implementation is `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters) running on CPU to rerank 20–50 retrieved chunks. Community benchmarks put it about 14x faster on CPU than bge-reranker-v2-m3 for 40 candidates, though those numbers come from small projects on unknown hardware ([GitHub PR](https://github.com/rusty-chris/lets-talk-climate-emergency/pull/383)). If the team wants a hosted arm, Cohere Rerank 4 costs $2.00–2.50 per 1,000 searches ([OpenRouter](https://openrouter.ai/cohere/rerank-4-pro)).

After reranking, choose how many chunks to keep per request (dynamic-k):

- Keep chunks above a score τ, tuned so that recall of the gold evidence stays at or above 95% on a dev set.
- Stop at a relative cutoff or at the largest score gap.
- Always keep at least one or two chunks.
- Stop when a hard token budget is reached.
- Send no context at all when even the top score is below a floor, as RECOMP does.

Before filling the budget, reserve room for the system prompt and the expected completion. Drop whole chunks rather than cutting mid-sentence. Recall@k of the gold evidence is a free proxy metric. Confirm it with end-to-end exact match/F1 or a judge on 200–300 items.

### Never compress the cached prefix

Compression interacts badly with provider prefix caching. A 3,000-token system prompt cached at 0.1x costs the equivalent of 300 input tokens. Compress it 2x and it becomes 1,500 *uncached* tokens, which costs **five times as much** and risks dropping a "not" or an "only" from the instructions. The rule follows directly:

- **Static system prompt, tools and few-shot examples:** first, uncompressed, cached. LLMLingua's `<llmlingua compress=False>` tags and `force_tokens` exist for exactly this ([README](https://github.com/microsoft/LLMLingua)).
- **Retrieved context and history:** reranked or compressed.
- **User question:** last, verbatim.

Content type matters too. LLMLingua-2 was trained only on English meeting transcripts, so code, JSON, tables, IDs and number-dense text are all at risk. Detect them (code fences, whether `json.loads` succeeds, how dense the numbers are) and skip compression for them. Back this with a unit test that asserts every number and JSON value from the original survives.

Compress history in append-only blocks that are frozen once written. Re-summarising earlier turns on every request rewrites the prefix and breaks the cache from that point on.

Even when compression works, the effect on the bill is modest:

> saving ≈ input share of cost × compressible share of input × (1 − 1/r), where r is the compression ratio

For a RAG call where input is 80% of the cost and retrieved context is 70% of the input, 3x compression saves about **37%** of the bill. LLMLingua-2 also adds about 0.4–0.7 s per call, so only run it on prompts with more than 1,000–2,000 compressible tokens.

### Choose the compression rate with a paired non-inferiority test

Run every item twice, once with the baseline prompt and once compressed, against the same model, and score the difference. Use three kinds of scoring:

- **Task metrics where gold answers exist:** exact match or F1 on NaturalQuestions or HotpotQA slices, exact match on GSM8K, ROUGE or a judge on MeetingBank.
- **RAGAS Faithfulness for RAG**, because compression can strip out the evidence an answer cites ([RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)).
- **A judge that grades against a reference answer.** In the MT-Bench study, this cut maths grading failures from 70% to 15% ([MT-Bench](https://arxiv.org/html/2306.05685v4)).

Sweep the `rate` setting over {0.7, 0.5, 0.33, 0.2}, roughly 1.4x to 5x compression. Choose the most aggressive rate whose 95% confidence-interval lower bound stays above a margin fixed in advance, for example −3 points. Compute the interval with McNemar's test or a paired bootstrap ([Miller](https://arxiv.org/abs/2411.00640)).

Pairing matters for the budget. By standard McNemar power arithmetic, detecting a 5-point drop at 80% power takes about **250 paired items**, and a 3-point drop about **500**. An unpaired design needs thousands per arm. A hand-written set of 60–120 items can therefore only support coarse gates (about ±10 points at n=100). Use it for the graded artefact and the CI gate, and use public dataset slices for the statistically powered curves. Add a hostile set of JSON, code and number-heavy inputs. The notes found no published study of LLMLingua-2 on these, so the team's result there is new.

### Output-side levers often save more dollars than compression

At current prices, output tokens cost 5–8x as much as input tokens: GPT-5-nano is $0.05 in vs $0.40 out, GPT-5.4-mini $0.75 vs $4.50, Claude Sonnet 5.5 $2 vs $10 ([OpenAI pricing](https://developers.openai.com/api/docs/pricing), [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)). Compression only touches input. Several output-side levers cost nothing to build:

- `max_tokens` caps set per route, with `finish_reason == "length"` logged as a quality-risk signal.
- Stop sequences.
- Terse structured output.
- Trimming template text that is duplicated across messages.

Report these as their own step in the savings waterfall so compression is not credited for them.

One caution for reasoning-capable models: hidden reasoning tokens are billed at the output rate, which Gemini states explicitly. A low `max_tokens` can then cut off the visible answer entirely. Set reasoning effort low where the API offers it, and log reasoning tokens ([Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)).

## Downshift routing: simple routers behind a strict eval gate

### Independent benchmarks find simple routers competitive

Routers come in three families:

- **Cascades** try the cheap model first and escalate when a check fails. FrugalGPT is one. AutoMix is another; it uses self-verification and reports over 50% cost cuts at comparable performance ([FrugalGPT](https://arxiv.org/abs/2305.05176), [AutoMix](https://arxiv.org/html/2310.12963v5)).
- **Predictive routers** choose a model before generating anything. Examples:
  - RouteLLM, whose matrix-factorisation router was trained on about 55,000 Chatbot Arena preferences ([RouteLLM](https://github.com/lm-sys/RouteLLM)).
  - Hybrid LLM, which reports up to 40% fewer large-model calls with no quality drop ([Hybrid LLM](https://arxiv.org/abs/2404.14618)).
  - Avengers-Pro, which matched the strongest single model at 27% lower cost ([Avengers-Pro](https://arxiv.org/abs/2508.12631)).
- **Heuristic or market routers**, such as OpenRouter's Auto Router. It sorts each prompt into one of about 30 task types and ranks models by trailing seven-day spend on that type ([OpenRouter](https://openrouter.ai/docs/guides/routing/auto-model-selection)).

Two independent 2025–2026 benchmarks deflate the sophisticated end of this list. RouterArena (ICLR 2026) found that no router tops every metric. It ranks Not Diamond **#12** because it "frequently selects expensive models", and finds the best open routers reach about 35% lower cost at under 2% accuracy loss ([RouterArena](https://arxiv.org/html/2510.00202v1)). LLMRouterBench covers 33 models and 21 datasets. It finds that several recent methods, "including commercial routers, fail to reliably outperform a simple baseline" ([LLMRouterBench](https://arxiv.org/abs/2601.07206)). Not Diamond's own "at least 20–40%" savings claim is vendor-reported ([Not Diamond](https://www.notdiamond.ai/pricing)).

For CostGuard, this justifies a simple router. It can use rules on token count, task keywords, the presence of code or maths, and conversation depth, or a k-nearest-neighbour classifier over embeddings of a small labelled set. Add an optional cascade only where the cheap model's answer is cheap to verify, for example with a JSON schema or unit tests.

RouteLLM's pretrained router works out of the box through its OpenAI-compatible server. But it was trained on 2024 GPT-4/Mixtral preference data, and its authors warn that real traffic can differ from benchmarks. Its threshold must therefore be recalibrated on the team's own traffic ([RouteLLM paper](https://arxiv.org/html/2406.18665)).

### The price gap decides whether downshifting pays

Approximate savings:

- **Router:** r × (1 − p_cheap / p_strong), where r is the share of traffic routed to the cheap model.
- **Cascade:** about 1 − p_cheap / p_strong − escalation rate − verifier cost.

Two worked examples, using a request of 1,500 input and 400 output tokens and Claude Sonnet 5.5 as the strong model ($0.0070 per request) ([Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI pricing](https://developers.openai.com/api/docs/pricing)):

| Cheap model (cost per request) | Route half the traffic | Cascade | Ceiling at 100% downshift |
|---|---|---|---|
| GPT-6-luna ($0.00035) | 47.5% saved | 75% saved at 20% escalation | ~95% |
| Claude Haiku 4.5 ($0.0035) | 25% saved | 10% saved at 40% escalation, before verifier cost | 50% |

Moving down one tier within the same vendor saves little. Routing pays when the target is a "nano" or "lite" model several tiers down.

Downshifting can also break caching. A 3,000-token shared system prompt can be cached on Sonnet 5.5, whose minimum is 512 tokens, but not on Haiku 4.5, whose minimum is 4,096 ([Anthropic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)). On that prefix, the "cheaper" model therefore costs about five times more. Add Anthropic's newer tokenizer, which produces about 30% more tokens for Claude 4.7 and later models (assuming Sonnet 5.5 uses it), and a nominal 50% saving shrinks to roughly **25%** ([Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)). Report routing savings against a counterfactual in which the strong model also gets its cache hits; otherwise the router takes credit for a provider discount.

### Gate each category on the lower bound of a paired confidence interval

The eval gate should treat downshifting as a promotion decision made separately for each category. Steps:

1. **Freeze an eval set** of 300–500 prompts, stratified by category: chit-chat, extraction/JSON, summarisation, code, maths, long-context.
2. **Score** with programmatic checks where possible and a pairwise judge elsewhere.
3. **Summarise with RouteLLM's metrics** ([RouteLLM paper](https://arxiv.org/html/2406.18665)). PGR is the share of the weak-to-strong quality gap the router recovers. CPT(x%) is the share of calls that must go to the strong model to recover x% of that gap.
4. **Judge each pair in both orders**, and count order-inconsistent verdicts as ties. In the MT-Bench study, GPT-4 gave the same verdict regardless of order only 65% of the time. Its agreement with humans (66% counting ties, 85% excluding them) matched human-human agreement (63% and 81%) ([MT-Bench](https://arxiv.org/html/2306.05685v4)).
5. **Use a judge from a different model family** than the models being judged, to avoid self-preference bias.
6. **Hand-label 40–50 pairs** and report judge-human agreement and Cohen's κ.
7. **Ship a category only if** the 95% lower bound of its non-inferior rate clears a threshold fixed in advance, such as 0.90. The non-inferior rate is the share of prompts where the cheap answer wins or ties.

Sample sizes limit how fine these gates can be. For a proportion near 0.5, the confidence-interval half-width is ±13.9 points at n=50, ±9.8 at 100 and ±5.7 at 300. A 50-item category therefore gets only a coarse gate; pool categories or oversample the high-traffic ones.

Batch APIs halve the cost of running the gate at OpenAI, Anthropic and Gemini ([OpenAI Batch](https://developers.openai.com/api/docs/guides/batch), [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)). The routing notes estimate a 300-prompt gate at about $6 at standard prices, or $3 through Batch, with Sonnet 5.5 as both the strong model and the judge. With mini-tier models it costs far less.

Online, the same decision rolls out in stages: shadow mode, then canary, then full rollout. Watch escalation rate, format failures and sampled judge scores, and roll back automatically if any of them breaches its threshold.

## October 2026 prices put nano-tier models at about 3% of frontier cost

Prices below come from the official pages fetched on 3 October 2026 unless marked otherwise.

| Tier | Model | Input / output per 1M tokens | Cost per 1k requests (1,500 in / 400 out) | Note |
|---|---|---|---|---|
| Frontier | Claude Opus 5.5 | $4 / $20 | $14.00 | |
| Frontier | Gemini 3.1 Pro Preview | $2 / $12 | $7.80 | No free tier |
| Frontier | GPT-6.1-sol; Claude Sonnet 5.5 | $2 / $10 | $7.00 | |
| Mid | Claude Haiku 4.5 | $1 / $5 | $3.50 | Cheapest current Claude; 4,096-token cache minimum |
| Mid | GPT-5.4-mini | $0.75 / $4.50 | $2.93 | Suggested CostGuard baseline |
| Mid | Gemini 2.5 Flash | $0.30 / $2.50 | $1.45 | Free tier; output price includes thinking tokens |
| Cheap | DeepSeek V4.1-Flash (peak hours) | $0.30 / $1.20 | $0.93 | Off-peak is half price |
| Cheap | GPT-6-luna | $0.10 / $0.50 | $0.35 | |
| Cheap | Gemini 2.5 Flash-Lite | $0.10 / $0.40 | $0.31 | Free tier |
| Cheap | GPT-5-nano | $0.05 / $0.40 | $0.24 | Suggested downshift target |
| Cheap | Groq gpt-oss-20b | $0.075 / $0.30 | $0.23 | Price from the LiteLLM map; free plan available |
| Embedding | text-embedding-3-small | $0.02 | — | Local MiniLM or bge-small cost $0 |

Sources: [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI](https://developers.openai.com/api/docs/pricing), [Gemini](https://ai.google.dev/gemini-api/docs/pricing), [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing), [LiteLLM price map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

Provider-side discounts often beat routing, and a cost meter has to model them correctly:

| Lever | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Cache read | 0.1x input (0.05x Opus 5.5) | 0.1x on GPT-5.6+ (0.05x GPT-6.1-sol); 75% off on GPT-4.1/o3; 50% off on GPT-4o | 10% of input; explicit caches add $1–4.50 per MTok per hour of storage |
| Cache write | 1.25x (5-min TTL), 2x (1-hour TTL) | 1.25x on GPT-5.6+ | — |
| Minimum cacheable prefix | 512–4,096 tokens by model; below it, no caching and no error | 1,024 tokens | 2,048 (2.5) / 4,096 (3.x) |
| Batch | 50% off; stacks with caching | 50% off, 24-hour window, separate rate limits | 50% off |
| Flex | — | Batch price; may return 429 (not charged) | Batch price |
| Priority / Fast | 2x (Opus 5.5) | 2x (renamed Fast on 30 Jul 2026) | 1.8x |

Sources: [Anthropic caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI caching](https://developers.openai.com/api/docs/guides/prompt-caching), [OpenAI Flex](https://developers.openai.com/api/docs/guides/flex-processing), [Gemini caching](https://ai.google.dev/gemini-api/docs/caching), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing). DeepSeek prices automatic cache hits about 50x below misses, and its off-peak hours are half price ([DeepSeek](https://api-docs.deepseek.com/quick_start/pricing)).

**Free tiers are real but come with strings:**

- **Gemini** offers a free tier on most Flash, Flash-Lite and embedding models, but not on 3.1 Pro. Free-tier content is "used to improve our products". The official rate-limit page publishes no free-tier numbers and says limits are "not guaranteed" ([Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)).
- **Groq's** free plan allows 30 requests per minute, 1,000 per day, 8,000 tokens per minute and 200,000 tokens per day on its gpt-oss models ([Groq](https://console.groq.com/docs/rate-limits)).
- **Anthropic** offers only small starter credits ([Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)).

Free-tier calls cost $0, so report them at their list-price equivalent.

**Several prices change on a schedule**, so the price table must be versioned with a `checked_on` date:

- Gemini 3.6–3.8 Flash costs $0.75/$3.75 until 31 December 2026, then doubles.
- GPT-5.6-sol is on promotional pricing until at least 21 November 2026.
- DeepSeek has peak and off-peak rates.

Sources: [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [OpenAI pricing](https://developers.openai.com/api/docs/pricing).

**Cost accounting needs one formula and one source of truth.** The formula:

> cost = input price × uncached input + cache-write price × written tokens + cache-read price × read tokens + output price × (output + reasoning tokens)

Then multiply by the service tier (0.5x for batch or flex, 1.8–2x for priority) and any region surcharge (1.1x).

The source of truth is the provider's returned `usage` object. Token counters are only for routing decisions made *before* the request: Anthropic's `count_tokens` is free but explicitly an estimate, and Gemini has `countTokens` ([Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting), [Gemini tokens](https://ai.google.dev/gemini-api/docs/tokens)). Other details to get right:

- Anthropic's tool-use system prompt silently adds 286–675 input tokens per call ([Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)).
- LiteLLM's price map matched the official pages for OpenAI, Anthropic and Gemini, but was stale for DeepSeek ([LiteLLM map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)).
- Attribute savings to routing, caching and service tier separately.
- Include CostGuard's own overhead (embedding calls, compression compute, judge calls) in the optimised arm's cost, because examiners will ask.

**Recommended model pairing:**

- **Baseline:** GPT-5.4-mini.
- **Downshift target:** GPT-5-nano, or GPT-6-luna.
- **Judge:** Gemini 2.5 Flash. It is a different model family from the judged models, and its free tier is fine for public data.

Any of them works as a judge, but GPT-4o-mini as the *baseline* would leave little room to downshift: GPT-5-nano is only about 2x cheaper per request than GPT-4o-mini, but about 12x cheaper than GPT-5.4-mini. Avoid a flagship baseline as well. GPT-6-astra, at $50 per million output tokens, would cost about $22 in output alone for a 1,500-request trace. If the team wants a frontier story, have the dashboard reprice logged tokens at frontier rates and label that view "list-price equivalent". Set a hard spend cap in every provider dashboard on day one.

## A free-tier stack that keeps sweeps off metered quotas

### A thin OpenAI-compatible proxy is the most convincing drop-in

The strongest "drop-in" demonstration is an unmodified OpenAI-SDK client that changes only `base_url` to point at a FastAPI `POST /v1/chat/completions` endpoint.

The LiteLLM proxy already does a lot of this. It returns an `x-litellm-response-cost` header, writes spend logs to Postgres, offers Redis and Qdrant semantic caches, and exports open-source Prometheus metrics for spend, tokens, latency and cache hits ([LiteLLM cost tracking](https://docs.litellm.ai/docs/proxy/cost_tracking), [proxy caching](https://docs.litellm.ai/docs/proxy/caching), [Prometheus](https://docs.litellm.ai/docs/proxy/prometheus)). Building entirely on it would hide the four optimisations the examiner wants to see.

The better split:

- **Own the pipeline stages in your code**, each behind a config flag.
- **Use the LiteLLM SDK** (`litellm.acompletion` and `completion_cost`) only for provider adapters and its price map.
- **Run LiteLLM's own semantic cache** as a comparison arm.
- **Copy its header convention:** `x-costguard-cache`, `x-costguard-similarity`, `x-costguard-route`, `x-costguard-cost-usd`, `x-costguard-baseline-cost-usd`, `x-costguard-saved-usd` and `x-costguard-overhead-ms`.
- **Build the core as a library**, so the A/B harness calls it directly without an HTTP hop.

The pipeline order:

1. Exact-match cache.
2. Semantic cache.
3. Rerank and truncate retrieved context.
4. Compress, only above a token threshold.
5. Route.
6. Upstream call, with a circuit breaker that fails open to passthrough.
7. Write to the cache, with no-store for errors.
8. Emit one structured log record and one OpenTelemetry span.

The log record is the single source of truth for both the dashboard and the README. It holds request ID, trace position, arm, requested and served model, route reason, cache status and similarity, original and sent input tokens, cached tokens, output tokens, cost, baseline cost, and a latency breakdown by stage.

Name span attributes after the OpenTelemetry GenAI conventions, such as `gen_ai.usage.input_tokens` and `gen_ai.usage.cache_read.input_tokens`. The research found no standard cost attribute, so put cost under a `costguard.*` namespace ([OTel GenAI](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)).

### The stack, with the limits that bite

| Layer | Pick | Limit that matters | Fallback |
|---|---|---|---|
| Proxy | FastAPI + LiteLLM SDK | — | — |
| Embeddings | Local MiniLM, bge-small or langcache-embed-v3-small | $0; must be the model τ was calibrated on | text-embedding-3-small at $0.02/M tokens |
| Vectors for sweeps | In-process numpy or FAISS | None | — |
| Vectors for the live demo | Qdrant Cloud free | 0.5 vCPU, 1 GB RAM, 4 GB disk | Upstash Vector (10K queries + updates per day); Redis Cloud (30 MB) |
| Compression | LLMLingua-2 offline (laptop or Colab) | About 2.4 GB of F32 weights | Modal ($30/month credits); HF PRO ($9/month) |
| Reranker | ms-marco-MiniLM-L-6-v2 on CPU | — | bge-reranker-v2-m3; Cohere at $2–2.50 per 1k |
| Tracing | Langfuse Cloud Hobby | 50k units/month; 30-day data access | Self-hosted Arize Phoenix (ELv2, no feature gates) |
| Dashboard | Streamlit Community Cloud | Sleeps after 12 h idle; at most 2.7 GB RAM | Same app run locally |
| Hosting | Render free | Sleeps after 15 min idle; ~1 min to wake; no persistent disk | Railway ($5 one-time trial); HF PRO ($9/month) |
| CI | GitHub Actions on a public repo | Standard runners free | 2,000 min/month if private |
| Load test | k6 or Locust against a mock upstream | Grafana Cloud k6: 500 VU-hours/month | Run locally |

Sources: [Qdrant](https://qdrant.tech/pricing/), [Upstash Vector](https://upstash.com/pricing/vector), [Redis](https://redis.io/pricing/), [Modal](https://modal.com/pricing), [Langfuse](https://langfuse.com/pricing), [Phoenix licence](https://arize.com/docs/phoenix/self-hosting/license), [Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [Render](https://render.com/docs/free), [Railway](https://railway.com/pricing), [GitHub Actions](https://docs.github.com/en/billing/concepts/product-billing/github-actions), [Grafana](https://grafana.com/pricing/).

The quotas decide where sweeps run. A Langfuse unit is one trace, one observation or one score ([Langfuse units](https://langfuse.com/docs/administration/billable-units)). A proxied request with four pipeline spans and one score uses about six units, so the free 50,000 units cover only about 8,000 requests a month. One 2,000-request A/B across two arms (about 24,000 units) fits. A five-threshold sweep does not. Upstash Vector's 10,000 daily operations would likewise run out after one or two full sweeps. The rule: send only the headline A/B and live-demo traffic to cloud services, and keep every sweep, ablation and CI run in local SQLite or Parquet. Helicone's free tier is smaller still, at 10,000 requests a month with 7-day retention ([Helicone](https://www.helicone.ai/pricing)).

### Commonly suggested free hosts are partly out of date

As of 3 October 2026, two of the three are no longer free:

- **Fly.io:** "New organizations don't have a free tier"; the trial covers only up to 2 hours of machine time or 7 days ([Fly.io](https://docs.fly.io/about/pricing)).
- **Hugging Face Spaces:** Gradio and Docker Spaces "require a paid plan to create" (PRO, $9/month), and only static Spaces are free ([HF Spaces](https://huggingface.co/docs/hub/spaces-overview), [HF pricing](https://huggingface.co/pricing)).
- **Render free:** still available, but it sleeps after 15 idle minutes, takes about a minute to wake, and can be suspended for unusually high traffic ([Render](https://render.com/docs/free)). Its RAM is not stated on the docs page; it is commonly reported as 512 MB, but that is unverified.

The pragmatic choice is Render free with a lightweight live path: no LLMLingua, and the reranker optional. Pre-warm it minutes before the demo, keep a recorded backup video, and run load tests locally against the mock upstream, never against Render. If the team wants LLMLingua running live, $9 for HF PRO buys a 2-vCPU, 16 GB CPU Space, which is about half the budget.

### One frozen trace, replayed from cassettes

**What goes into the trace.** Build a trace of about 1,500 requests from licence-safe public data:

- **About 60% from Bitext customer support:** 26,872 Q/A pairs across 27 intents, CDLA-Sharing 1.0. It contains natural paraphrase clusters and near-miss intents such as cancel versus track order ([Bitext](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)).
- **About 20% QQP pairs**, injected close together in the trace, which give ground-truth cache precision and recall.
- **About 20% from WildChat:** English single-turn first messages (ODC-BY), with toxic rows removed and prompts kept to 256 tokens or fewer ([WildChat](https://huggingface.co/datasets/allenai/WildChat-1M)).
- **A RAG slice** with retrieved documents attached, so the reranker and compressor have something to cut.

Keep LMSYS-Chat-1M out of the public repo: its licence carries "strict prohibitions on redistribution" ([LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)). The trace is workload, not the graded eval set.

**Controlling the duplicate rate.** Assign every request to a cluster, and draw the arrival order from a Zipf-like popularity distribution over clusters. Present a headline rate of about 30%, justified by MeanCache's 31% and explicitly caveated as a per-user figure. Add a sensitivity table at 0%, 15%, 30% and 50%.

**Replaying it reproducibly:**

1. Replay requests sequentially in trace order, and flush the caches before each arm.
2. Use temperature 0 and a fixed seed, and log `system_fingerprint`. OpenAI's seed is best-effort: "Determinism is not guaranteed" ([OpenAI seed](https://developers.openai.com/cookbook/examples/reproducible_outputs_with_the_seed_parameter)).
3. Record every upstream response in a cassette keyed by a hash of model, messages and parameters. Re-analysis then costs $0 and is exactly reproducible, and CI can run with no API key at all.
4. Commit `trace.jsonl` together with its SHA-256 hash and the generator's random seed.

## Five days: freeze the trace Sunday and the code Wednesday

Counting today, Sunday 4 October, that leaves five working days. The presentation is Sunday 11 October, so Friday and Saturday belong to the deck and Q&A rehearsal, not to code. The six-person team should split into six workstreams, each owning one directory and one test.

| Workstream | Owns | Key deliverables |
|---|---|---|
| W1: Proxy core | FastAPI endpoint, pipeline skeleton with flags, `x-costguard-*` headers, LiteLLM pricing, log schema, mock upstream, Dockerfile | Passthrough proxy with logging, deployed on day 1 |
| W2: Semantic cache | Exact and semantic tiers, embedding adapter, local and Qdrant stores, guards | τ sweep, false-hit analysis, comparator arm |
| W3: Context | Reranker with dynamic-k, LLMLingua-2 offline, fallback truncation | Compression-rate and dynamic-k quality curves |
| W4: Router | Heuristic or kNN router, per-category gate, cost-accounting tests (tokens × price = logged $) | Gate result per category |
| W5: Eval and stats | Trace generator, cassette recorder, judge prompts, human labels, bootstrap | `make ab` → `results/*.json` |
| W6: Ops | Langfuse, Streamlit dashboard, CI gate and branch protection, deploy, k6 or Locust | Red and green PRs, live URL, README auto-filled from results |

| Day | Work | Exit criterion |
|---|---|---|
| Sun 4 Oct | Agree the API, log schema, config flags, arm definitions, model pair, quality margin and trace recipe in one hour. Set spend caps. Create repo and CI skeleton. Freeze `trace_v1`. Build the cassette recorder. Each member writes 10 eval items | Passthrough proxy live on the chosen host |
| Mon 5 Oct | Each lever behind a flag, with unit tests on cassette data. Record the baseline arm on the full trace (the main real spend). Dashboard on early logs. CI eval job running but not yet blocking | Baseline cassette committed |
| Tue 6 Oct | Sweeps from cassettes: τ, compression rate, dynamic-k, router threshold. Pick operating points. Judge pipeline and bootstrap. Langfuse traces. k6 against the mock upstream | One command prints a rough ablation table |
| Wed 7 Oct | All arms on the frozen trace. Pairwise judge plus 40–50 human labels. Load tests. Make the CI gate required and stage red and green PRs. Final deploy checked from a phone. Record a backup video | Code freeze at 20:00 |
| Thu 8 Oct | README tables generated from `results/*.json`, never hand-typed. Resume line. Submit hours early | Submitted |
| Fri 9 – Sat 10 Oct | Deck weighted by rubric. Two timed rehearsals. Q&A drill on likely reviewer questions. Assign who writes the review sheets | Rehearsed under 15 minutes |

|---|---|---|
| A0 | Baseline: strong model, passthrough | "Before" column of the A/B |
| A1 | + exact cache | Safe-first lever |
| A2 | + semantic cache at the chosen τ, with guards | Hit rate vs false-hit rate |
| A3 | + rerank and dynamic-k truncation | Context-window management, measured |
| A4 | + LLMLingua-2 on volatile context only | Compression with a quality-delta eval |
| A5 | + router behind the eval gate | Downshift with an eval gate |
| B | LiteLLM `redis-semantic` at 0.8 | Off-the-shelf comparator |

**Headline metrics:**

- Cost per request.
- Percent saved with a 95% confidence interval. Use a paired bootstrap that resamples by cluster or user, because duplicate requests are correlated by construction ([Miller](https://arxiv.org/abs/2411.00640)).
- Cost per correct answer.
- Quality retained, and judge win/tie/loss.
- Hit rate and false-hit rate.
- Mean compression ratio.
- Route mix.
- p50/p99 latency, split by hit and miss.
- Proxy overhead.
- Throughput at a p99 latency target.

The handout's resume template ends in an "X% token-spend cut at <Y% quality loss" claim.

**Budget:**

| Item | Estimated cost |
|---|---|
| Baseline arm: 1,500 requests at ~300 in / 300 out on GPT-5.4-mini | ~$2.36 |
| The same trace on GPT-5-nano | ~$0.20 |
| 600 pairwise judge calls on a mini-class judge | $0.50–1.00 |
| Embeddings | under $0.01 |
| **Whole experiment** | **about $5** |

That leaves room for re-runs or HF PRO.

The cheapest version is a pytest job that runs 30–60 hand-written items plus about 20 cache traps from cassettes. It fails if the pass rate drops by more than two items or five points, or if any trap produces a false hit. Mark it a required status check ([GitHub](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)). promptfoo's GitHub Action adds a before/after PR comment and caches LLM calls ([promptfoo](https://www.promptfoo.dev/docs/integrations/github-action/)). DeepEval is the pytest-native alternative ([DeepEval](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd)). Run the gate three times on `main` to measure run-to-run noise, and set the margin above it. Stage two pull requests for the screenshots:

1. A PR that lowers τ from 0.85 to 0.6 "for more savings". The savings number rises, the check goes red and the merge is blocked.
2. A PR that restores τ and adds the entity guard. The check goes green.

**Common failure modes, each with its guard:**

| Failure | Guard |
|---|---|
| Numbers that cannot be reproduced | Frozen trace hash, cassette replay, committed results |
| Savings that come from wrong answers | Trap set and false-hit reporting |
| A duplicate rate that looks rigged | Sensitivity table and a savings-excluding-cache view |
| LLM-judge bias | Order swapping, a different-family judge, published human agreement |
| Single-run point estimates | Bootstrap intervals |
| Load tests that measure OpenAI instead of CostGuard | A mock upstream |
| A sleeping or out-of-memory free host | Pre-warming, a light live path, a backup video |
| Quota exhaustion mid-sweep | Local sweeps |
| Licence or secret leaks | No LMSYS data; cassette-mode CI with no keys |
| Last-day integration | A day-one deploy and a Tuesday-evening checkpoint |
| Cost bugs | A unit test that recomputes dollars from tokens, including CostGuard's own overhead |
| Scope creep (streaming, auth, fancy frontends) | A written "won't do" list on day one |

## Conclusion

CostGuard's most defensible novelty is not a technique but filling gaps the literature leaves open: compression measured against current frontier models, false hits driven by negations and changed entities, quality curves for dynamic-k, and savings at a 0% duplicate rate. Every published number the team could quote comes from a different model, dataset or era, so the team's own measured frontier is the contribution. Treating the work as an experiment also turns the riskiest parts into the strongest slides. A visibly rejected τ=0.6 pull request, a trap set the cache refuses, and a confidence interval that admits a 50-item category can only be gated coarsely all signal the operational rigour the handout says separates a portfolio project from a demo.

The dollar arithmetic also says something the brief's wording hides. Output tokens cost 5–8x as much as input tokens, and only cache hits and downshifts touch output. So the savings waterfall will almost certainly show compression as the *largest* lever in tokens saved and the *smallest* in dollars, and routing's contribution will depend more on the chosen price gap than on how clever the router is. Saying this before an examiner does, and leading with cost per correct answer rather than tokens saved, turns a likely Q&A attack into evidence that the team understood its own system.
