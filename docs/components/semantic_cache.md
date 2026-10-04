# Response caching: exact tier, semantic tier and guards

**Question this component answers:** can this request be answered from an earlier answer without an LLM call, and how often would that answer be wrong?

Caching is stages 1, 2 and 7 of `costguard/pipeline.py`. A hit skips the upstream call entirely, so it saves 100% of input and output cost. A wrong hit is a confident wrong answer returned with HTTP 200, so the whole design centres on one number: the **false-hit rate per request**.

Code:

| File | What it holds |
|---|---|
| `costguard/cache/exact.py` | `build_exact_cache` and `InMemoryExactCache`: thread-safe dict with TTL and an LRU cap. |
| `costguard/cache/embedder.py` | Local fastembed model (`BAAI/bge-small-en-v1.5`, 384-d, MIT). Process-wide singleton, L2-normalised float32. |
| `costguard/cache/semantic.py` | `build_semantic_cache`, `SemanticCacheImpl`, and two stores: `MemoryStore` (numpy) and `QdrantStore`. |
| `costguard/cache/guards.py` | Deterministic lookalike guards: numbers, negation, entities, content. |
| `eval/cache_pairs.py` | Builds the labelled pairs file `eval/data/cache_pairs/pairs_v1.jsonl`. |
| `eval/sweep_threshold.py` | Threshold sweep and cache simulation. Writes `eval/results/threshold_sweep.json`. |
| `tests/test_cache.py` | 36 tests. Runtime is about 1 s after the model loads. |

## 1. Design

### Request flow

```
request ──► cacheable?  (mode has caching on, temperature ≤ 0.3, single turn, no_cache = false)
              │ no ─► bypass (cache_status = "bypass")
              ▼ yes
       1. exact tier    key = sha256(partition, normalised query, hash(retrieved context))       ~1 µs
              │ miss
              ▼
       2. semantic tier embed query (bge-small, local) ─► top-5 cosine neighbours in the SAME partition
                        ─► walk candidates with similarity ≥ τ(mode), best first
                        ─► serve the first one that every guard accepts                         ~2–5 ms
              │ miss (no neighbour ≥ τ, or every candidate vetoed ─► cache_guard = reason)
              ▼
       3–6. context, compression, router, upstream call
              ▼
       7. write-back    exact.put(key) + semantic.insert(query); skip near-duplicates (cos ≥ 0.98)
```

Every lookup fills `SemanticHit.similarity` and `neighbor_query`, even on a miss, so the request log shows how close each miss came. The pipeline copies these into `TraceRecord.cache_similarity`, `cache_neighbor` and `cache_guard`. `SemanticCacheImpl.stats()` adds embed and search latency (p50/p99), hit and guard-rejection counters, and the model name.

### Why the exact tier comes first

- **It is lossless.** The key covers the partition, the normalised query and a hash of the retrieved context, so a hit can never be a different question.
- **It costs almost nothing.** One dict lookup takes about 1 µs; the semantic tier pays for an embedding (about 2–5 ms).
- **It catches a real share of traffic.** In the Bitext cache simulation, 10–24% of requests (depending on τ and rendering) were served from a cached query with identical text. Like every Bitext hit rate, read this as an upper bound (section 3).

The research brief's references put a deterministic tier first for the same reasons (Krites: static then dynamic cache; VentureBeat: 18% exact duplicates).

### Partitioning

The pipeline builds the partition as `tenant | sha256(system prompt)[:8] | kb_version | ctx-or-noctx`. Both tiers only match inside one partition: the memory store keeps one matrix per partition, and Qdrant filters on a `partition` payload.

| Part | Why it is in the key |
|---|---|
| `tenant` | Without it, a cache shared across tenants leaks answers between them. A fuzzy key also creates a cross-tenant attack surface (CacheAttack, arXiv 2601.23088). |
| system-prompt hash | Two apps behind one proxy can share answers only if they share instructions. We deliberately don't copy Portkey, whose semantic cache ignores the system prompt. |
| `kb_version` | Answers grounded in the store-policy knowledge base go stale when it changes. Bumping `kb_version` in `policy.yaml` starts a fresh partition, which invalidates every old entry at once without touching storage. |
| `ctx` / `noctx` | A RAG answer and a context-free answer to the same words are different products. |

The retrieved documents themselves are in the exact key but not the semantic key. Retrieval is a deterministic function of the query and `kb_version`, so near-identical queries retrieve near-identical context.

### What bypasses the cache, and why

These rules are enforced in `pipeline.py`:

- **Multi-turn requests** (`single_turn_only: true`). An answer to "and what about the blue one?" depends on the conversation history, which the key doesn't see. MeanCache measured precision on contextual queries at 0.66 without context chains; vCache says semantic caches work mainly for single-turn prompts.
- **`temperature > 0.3`.** The caller asked for variety, and serving a frozen sample defeats that.
- **`no_cache: true`** forces a fresh answer, and mode `off` disables caching.

### Tier-mismatch rule

Quality mode never serves an answer that the cheap tier produced. `_entry_ok` in the pipeline refuses the hit and logs `cache_guard = "tier_mismatch"`.

Both caches cooperate with this rule:

- The exact cache never overwrites a live strong-tier entry with a cheap one.
- On write-back, the semantic cache replaces a cheap-tier near-duplicate with a strong-tier answer instead of skipping it as a duplicate (`counters["tier_upgrades"]`).

Without the upgrade, an economy-mode answer cached first would block quality mode from ever getting a semantic hit on that question.

### Guards against lookalike false hits

Bi-encoder embeddings put "I want to cancel my order #4821" and "I don't want to cancel my order #4821" at cosine 0.92. The guards are cheap lexical checks (p50 0.07 ms, cold) that run only on candidates that already cleared τ. Each one returns a reason or `None`, and each can be switched off: `GuardConfig`, or `policy.cache.guards` (see "Requested policy keys").

| Guard | Rejects when | Example | Reason string |
|---|---|---|---|
| numbers | Digits, IDs (`SN-48213`), amounts, dates or number words differ, or only one side has them | order #4821 vs #4822; 30 vs 60 days | `number_mismatch` |
| negation | A word negated on one side appears un-negated on the other. Hedges such as "I don't know how to" and "I can't" are stripped first. An `un-` antonym also counts. | "cancel" vs "don't want to cancel"; "with" vs "without the receipt"; subscribe vs unsubscribe | `negation_mismatch` |
| entities | Both queries name something from the same ShopNest lexicon group, and the two sets are disjoint. Groups: product, action, payment, tier, shipping, time, timing, place, object. | laptops vs phones; refund vs exchange; PayPal vs UPI; express vs standard | `entity_mismatch:<group>` |
| content | Otherwise near-identical queries each carry a different uncommon word that the lexicon doesn't know | "ship to Nagpur" vs "ship to Indore" | `content_mismatch` |

**These guards are our own engineering heuristic, not a published method.** The research notes found no benchmark for entity or number guards. The rules were developed by reading Bitext false hits and false rejections, plus the 28 seed traps in `eval/cache_pairs.py` (`author: "seed"`, AI-written scaffolding, not team-written). Treat the trap pass rate as optimistic. The 6 trap pairs in `eval/data/evalset/seed.jsonl`, written separately by the eval workstream (also AI-written seed rows), are the cleaner held-out check, and all 6 are caught. The team's hand-written trap pairs are the real test once they land.

If the best candidate is vetoed, the cache tries the next-best candidate above τ. A query about order #4822 can therefore still hit a cached #4822 answer when a cached #4821 answer scores slightly higher.

### TTL and invalidation

- **TTL.** Entries expire after `policy.cache.ttl_seconds` (7 days). Expired entries never match and are compacted lazily.
- **Bulk invalidation** goes through `kb_version` (see "Partitioning").
- **Bounded size.** Each partition is capped at 50k entries, evicting the least-recently-used 10% when full. The exact tier has a 50k LRU cap.

Per-category TTLs, such as the AWS guidance of minutes for prices and 24 h for policies, would be a small extension. They need the request category passed to `insert`.

### Storage backends

| | `memory` (default) | `qdrant` (`COSTGUARD_SEMANTIC_BACKEND=qdrant`) |
|---|---|---|
| Search | Exact brute-force dot product over a per-partition float32 matrix | qdrant-client cosine, `partition` and `expires_at` as payload filters |
| Where | In-process | `QDRANT_URL` if set, else embedded local mode in `data/runtime/qdrant` |
| Scale | p50 0.02 ms at 1.1k entries; fine to about 50k per partition | Survives restarts; shared by replicas only in server mode |
| Collection | – | `costguard_semcache__baai_bge_small_en_v1_5`: the model is in the name, because τ is only valid for one model |

Embedded Qdrant takes a file lock, so only one process can open `data/runtime/qdrant`. Run a Qdrant server for multi-worker deployments.

### Poisoning and leakage risks, and mitigations

The semantic key is a fuzzy hash, so collisions can be engineered. CacheAttack reports an 86% response-hijack rate. The 2026 defence paper (arXiv 2609.35908) adds that query embeddings can lose the information needed to separate valid hits from invalid ones. Our exposure and mitigations:

| Risk | Mitigation in CostGuard | Gap |
|---|---|---|
| Tenant A plants an entry that tenant B hits | Tenant is part of the partition, so cross-tenant hits are impossible by construction | – |
| A user poisons their own tenant's cache with a crafted query that collides with popular questions | Guards compare the raw cached-query text, which is the "check the cached query" idea from arXiv 2609.35908; entries expire through TTL; `clear()`; `kb_version` bump | Not adversarially robust. A crafted query with matching numbers and entities passes. Production would add per-user write quotas, write-back only for authenticated or verified sessions, and asynchronous LLM-judge sampling of hits (Krites style). |
| Stale answers after a policy change | `kb_version` partition and TTL | Per-category TTL not yet implemented |
| A cheap-tier answer served in quality mode | Tier-mismatch rule plus tier upgrade on write-back | – |
| Personalised answers ("your order #4821 is delayed") served to others | Numbers guard (IDs must match) and tenant partition | Personal data without numbers, such as names, relies on the content guard. A `no_cache` rule for account-specific intents would be stricter. |

## 2. Calibration method

### Data

All of it is stored under `eval/data/cache_pairs/`; build it with `python -m eval.cache_pairs`.

| Source | Rows | Licence | Use |
|---|---|---|---|
| Bitext customer-support (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`, 26,872 queries, 27 intents) | 3,000 pairs; the replay stream | **CDLA-Sharing-1.0** | Domain pairs and cache simulation. We keep a compact parquet with instruction, category and intent (450 kB). |
| Quora Question Pairs (`nyu-mll/glue`, qqp validation) | 2,000 pairs, 1,000 positive and 1,000 negative | Quora's original release terms; the GLUE card lists "other" | General-domain check only, for non-commercial evaluation; only the sample is stored |
| Trap pairs | 34: 28 seed (`author: "seed"`, AI-written) and 6 from `eval/data/evalset/seed.jsonl` (also AI-written seed rows) | project | Lookalikes that need different answers |
| Seed paraphrases | 14 (`author: "seed"`, AI-written) | project | Pairs that *should* hit, to measure what the guards wrongly block |

**Labels.** A hit is correct when the cached query and the new query have the same intent *and* the same specifics.

- Bitext responses are written per intent, but they echo the customer's specifics. In our analysis of the dataset (not saved under `eval/results`), `{{Order Number}}` is echoed 100% of the time, `{{Account Type}}` 98%, and literal tiers 70%. So "cancel order #A" and "cancel order #B" share an intent but are labelled a wrong hit.
- One relabel: Bitext's `newsletter_subscription` mixes subscribe and unsubscribe requests whose responses differ, so it is split in two, giving 28 labels.
- `pairs_v1.jsonl` keeps `label_intent_only` for anyone who wants the plain intent label.

**Pair mix.**

| Kind | Pairs | Label |
|---|---|---|
| Same intent, same specifics | 1,200 | 1 |
| Same intent, different specifics | 300 | 0 |
| Near-miss intents in one category (cancel_order vs track_order, get_refund vs track_refund) | 600 | 0 |
| Embedder-mined nearest neighbour with a different intent | 600 | 0 |
| Random | 300 | 0 |

Seed 13 throughout.

### Two views

1. **Pairwise curve.** Precision, recall and false-positive rate on the pairs, for τ from 0.70 to 0.99, with and without guards, per source. It is good for comparing models, but it is *not* the operating metric. A cache never sees a balanced set of pairs; it sees each new query's nearest neighbour among everything cached so far.
2. **Cache simulation (the one that sets τ).**
   - **Stream.** 5,000 Bitext queries with Zipf (s = 1) popularity over the 28 labels and 29.6% repeats of earlier queries.
   - **Run.** The stream goes, in arrival order, through the real `SemanticCacheImpl` (same code as serving; only the embeddings are precomputed), for every τ, with guards on and off. There are no LLM calls: correctness comes from the labels.
   - **Two renderings of the same stream.** *Templated* keeps Bitext's `{{Order Number}}` placeholders. *Filled* substitutes sampled order numbers (`SN-48213`), tiers, cities, countries and names, the way real customers type them.
   - **Metrics** follow `docs/CONTRACT.md`:
     - hit rate = hits ÷ requests;
     - **false-hit rate = wrong hits ÷ requests**;
     - precision per hit, reported alongside.
   - **Recommendation rule.** For each mode, pick the τ with the highest hit rate whose false-hit rate stays within budget (quality 0.5%, balanced 1%, economy 3%) with guards on. It must hold in both renderings, so we take the stricter τ.

## 3. Results

These numbers come from `eval/results/threshold_sweep.json` for `BAAI/bge-small-en-v1.5`; similarities are cosine *similarity*, not distance. Re-run `make sweep` to refresh them; the README tables are generated from the JSON.

### Recommended τ (guards on)

| Mode | Budget | **τ** | Templated: hit / false-hit [95% CI] / precision per hit | Filled: hit / false-hit [95% CI] / precision per hit |
|---|---|---|---|---|
| quality | 0.5% | **0.94** | 73.7% / 0.48% [0.32, 0.71] / 99.35% | 54.7% / 0.18% [0.09, 0.34] / 99.67% |
| balanced | 1% | **0.93** | 77.3% / 0.84% [0.62, 1.13] / 98.91% | 57.7% / 0.34% [0.21, 0.54] / 99.41% |
| economy | 3% | **0.89** | 87.2% / 2.54% [2.14, 3.01] / 97.09% | 67.3% / 1.56% [1.25, 1.94] / 97.68% |

`policy.yaml` has since moved from its starting points (0.95, 0.90, 0.85) to 0.95, 0.93 and 0.89: balanced and economy follow the sweep, and quality keeps the stricter 0.95 (next paragraph).

Quality's point estimate is just inside budget, but its CI upper bound (0.71%) is not. If the team wants the CI upper bound inside budget, use **τ = 0.95**: 69.5% hit and 0.24% false-hit templated, 51.0% hit and 0.14% false-hit filled.

**Read the hit rates as an upper bound.** Bitext is 27 intents with roughly 1,000 template paraphrases each, so nearly every query has a close neighbour. The research brief warns about exactly this: paraphrase-generated benchmarks report 50–90% hit rates, while real chat logs show 4.5% (MOSS) and 7.5% (LMSYS) reusable queries (SCALM), and Portkey reports about 20% (18–60% on RAG) at 99% accuracy on Q&A/RAG traffic.

The calibrated quantity is the false-hit rate at a given τ. Realised savings come from the frozen-trace A/B (`eval/run_ab.py`). Of the hits above, 10–24% of all requests are served from a cached query with identical text, which the exact tier would serve first.

### What the guards buy

The same stream was run with guards on and off. "False hits removed" is the number of wrong hits avoided. "Correct-hit change" can be positive: a vetoed near-hit is answered fresh and cached, so later paraphrases find a correct neighbour.

| τ | Rendering | False-hit off → on | False hits removed | Correct-hit change | Hit rate off → on |
|---|---|---|---|---|---|
| 0.85 | templated | 10.02% → 4.90% | 256 | +222 | 92.7% → 92.1% |
| 0.85 | filled | 32.98% → 3.54% | 1,472 | +586 | 89.8% → 72.1% |
| 0.90 | templated | 2.52% → 2.12% | 20 | +10 | 85.5% → 85.3% |
| 0.90 | filled | 21.78% → 1.22% | 1,028 | +369 | 78.7% → 65.5% |
| 0.93 | templated | 1.02% → 0.84% | 9 | +2 | 77.5% → 77.3% |
| 0.93 | filled | 9.88% → 0.34% | 477 | +138 | 64.4% → 57.7% |

**Without guards, real order numbers make semantic caching unusable below τ ≈ 0.96–0.97.** "Please cancel order SN-48213" and "Please cancel order SN-48231" embed at 0.976 (`trap_detail` in the JSON). Without guards, the τ that meets each budget in both renderings would be 0.97 / 0.96 / 0.95, giving filled hit rates of 41.6% / 47.0% / 52.7%. Guards allow 0.94 / 0.93 / 0.89 instead, giving 54.7% / 57.7% / 67.3%. That is **+11 to +15 points of hit rate at the same error budget**.

Per-candidate verdicts at τ = 0.90, counting each vetoed candidate as either "would have been wrong" or "would have been correct":

| Guard | Templated: would be wrong / would be correct | Filled: would be wrong / would be correct |
|---|---|---|
| numbers | 18 / 0 | 3,114 / 0 |
| entities | 3 / 1 | 11 / 1 |
| negation | 1 / 2 | 1 / 1 |
| content | – | – |

On the labelled pairs above 0.80 (Bitext), the numbers guard rejects 48 wrong pairs against 1 correct one. Entities reject 95 against 2, negation 4 against 2, content 0 against 0.

On **traps**, all 34 are caught. 16 of them are above 0.90, where the embedding alone would serve the wrong answer. On the 14 seed paraphrases, the guards block 0.

**The guards are domain-specific.** On QQP (general domain) above 0.80, they reject 81 of 363 non-duplicates but also 149 of 938 true duplicates, mostly through the numbers and content guards. Turn the content guard off (`guards: {content: false}`) for non-support traffic.

**The residual false hits at the recommended τ** are mostly near-synonymous Bitext intents that a lexical rule can't separate:

- check_invoice ↔ get_invoice;
- track_refund ↔ get_refund;
- edit_account ↔ switch_account.

Many of these would be acceptable answers in practice, so the per-request false-hit rate is pessimistic.

### False-hit rate does not grow with traffic here

vCache warns that a static τ leaks more errors as the cache fills. At τ = 0.93 (templated), the cumulative false-hit rate after 500, 2,000, 3,500 and 5,000 requests is 0.60%, 0.85%, 0.94% and 0.84%. It rises over the first 2,000 requests and then stays flat, because the cache saturates: 1,133 entries for 28 labels. Watch this on the real trace, which has a longer tail.

### Thresholds are model-specific

The same sweep on `sentence-transformers/all-MiniLM-L6-v2` recommends 0.88 / 0.85 / 0.76, against bge-small's 0.94 / 0.93 / 0.89. At τ = 0.85, bge-small's false-hit rate is 4.9% templated and 3.5% filled, while MiniLM's is 0.7% and 0.5%. The same number means different things on different models.

Pairwise ROC AUC is similar for the two models: Bitext 0.69 vs 0.70, QQP 0.87 vs 0.87. So the model choice matters less than calibrating τ for the model actually deployed. That's why the embedder records its model name, the JSON records it, and the Qdrant collection is named after it.

### Latency and cost

- **Embedding:** `embed_one` p50 2–3.6 ms, p99 3–8 ms, measured on CPU across runs.
- **Full lookup:** embed, search and guards on a 2k-entry cache take p50 2.5 ms and p99 5.2 ms.
- **Search alone** (memory store, 1.1k entries): p50 0.02 ms.
- **Model load:** 58 ms in the saved run (`latency_ms.model_load_ms`), up to about 250 ms in other runs, when cached on disk; the first-ever download took about 15 s (not saved).
- **Cost:** $0 per lookup. `overhead_cost_usd` stays 0.
- **Memory:** the quantised ONNX weights are 66 MB on disk (`models/fastembed`). Process RSS was not measured for the cache alone; with the KB index also loaded, `compression_eval.json` records 302 MB (`setup.peak_rss_mb.after_retrieval_index`).

For comparison, the research brief cites a production report of 20 ms p50 for embedding plus vector search, and the upstream call is 1–6 s.

## 4. Trade-offs

- **We chose local bge-small over OpenAI text-embedding-3-small** because the cache must work offline with no keys (this project has none), and because τ must be calibrated on the model actually serving. A hosted model would add a network hop (our estimate: about 50–200 ms, not measured) to every lookup, hits and misses alike, plus a vendor dependency on the hot path. That would erode the latency win of a hit. Embedding cost is not the reason: about $0.02 per 1M tokens is negligible.

  We picked bge-small over MiniLM on licence parity (MIT vs Apache) and on behaviour: it is the more conservative cosine scale, its pairwise AUC is the same, and it is fastembed's default. Redis's cache-tuned `langcache-embed-v3-small` is the natural next comparison.

- **We chose an in-memory numpy matrix, with Qdrant as the persistent option, over pgvector or RedisVL** because at our scale (thousands to about 50k entries per partition) exact brute-force search takes microseconds, has no recall loss from approximate indexes, and needs no service to run CI. Qdrant gives persistence and a server mode with payload filters for partitions behind the same three-method interface. pgvector would add a Postgres dependency for no gain at this size. RedisVL is the reference design we borrowed from (tenant filters, TTL), not a runtime dependency.

- **We chose per-request false-hit rate over per-hit precision as the calibration metric** because per-hit precision flatters loose thresholds. At τ = 0.85 (templated), per-hit precision is 94.7%, which sounds fine, but 4.9% of *all* requests get a wrong answer. Users experience the per-request rate, and it is what vCache and the project contract use.

- **We chose a static τ per mode plus deterministic guards over learned per-entry thresholds (vCache) or an LLM judge on the serving path** because guards are explainable (each refusal is logged with a reason), cost 0.07 ms and $0, and attack the dominant failure directly: lookalikes that differ in a number, negation or entity. A synchronous judge would add an LLM call to borderline requests, which defeats the purpose of caching. Krites makes the same argument and runs its judge asynchronously. A vCache-style adaptive τ is the obvious next step once real traffic gives per-entry labels.

- **We chose a strict label ("same intent *and* same specifics") over Bitext's plain intent label** because a cached "your order SN-48213 has been cancelled" served to someone asking about SN-90211 is the canonical false hit, and Bitext's own responses echo those specifics. The cost is that part of the numbers guard's measured gain is defined by the label. The trap set and the filled rendering are where that gain is real.

## 5. Limitations

- **Hit rates are inflated** by Bitext's template-heavy paraphrases (see section 3). Use the frozen trace for savings claims.
- **Guard rules were tuned on Bitext and seed-trap examples**, so they are dev-set-tuned. They don't handle constraint pairs the lexicon doesn't name, and they are not robust to adversarial queries.
- **Constraint traps need lexicon entries.** Pairs such as before vs after shipping, or early vs late, are caught only because the `timing` group lists those words.
- **The top-5 candidate walk.** When the best neighbour is vetoed, a lower-similarity candidate that is still ≥ τ can be served if it passes every guard. This is included in the measured numbers.
- **One seed.** The quality-mode recommendation sits at the budget edge; re-run with `--seed` to check stability.

## 6. Run it

```bash
./.venv/bin/python -m eval.cache_pairs          # build pairs_v1.jsonl (downloads Bitext + QQP once; ~45 s)
./.venv/bin/python -m eval.sweep_threshold      # sweep + simulation, both models (~1–2 min) -> eval/results/threshold_sweep.json
./.venv/bin/python -m eval.sweep_threshold --quick   # 1,500-request smoke run, bge only
./.venv/bin/python -m pytest -q tests/test_cache.py
```

### Requested policy keys

These are for the coordinator. All are optional, and the code falls back to the defaults shown.

```yaml
cache:
  embedding_model: BAAI/bge-small-en-v1.5     # τ below is only valid for this model
  guards: {numbers: true, negation: true, entities: true, content: true}
  semantic_max_entries: 50000                 # per partition
modes:
  quality:  {tau: 0.94}    # 0.95 if the CI upper bound must be inside budget
  balanced: {tau: 0.93}
  economy:  {tau: 0.89}
```
