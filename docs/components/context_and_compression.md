# Context optimisation and compression

**Owner:** context workstream. **Code:** `costguard/context/optimizer.py` (stage 3), `costguard/context/compress.py`
(stage 4), `eval/kb.py` (knowledge base and retrieval), `eval/compression_eval.py` (measurement).
**Results:** `eval/results/compression_eval.json`. The tables below are printed by
`python -m eval.compression_eval --markdown` from that file; they are not typed by hand.

## What these stages do

A RAG request reaches CostGuard with `costguard.context = [8 retrieved docs]`. The engine sends
`system prompt (cached, never touched) -> "Store policy context: [1] ... [8] ..." -> "Customer question: ..."`.
Only the middle block is volatile, so only the middle block is optimised:

```
retrieved top-8 --> 3. rerank + dynamic-k + whole-doc budget + order --> 4. compress (if >= compression_min_tokens) --> upstream
                    (cross-encoder, ~65 ms CPU)                         (heuristic ~4 ms | LLMLingua-2 ~210 ms MPS)
```

Both stages fail open. If either raises, the pipeline logs `stage_errors[...]` and sends the un-optimised block.

## Design

### Knowledge base and retrieval (`eval/kb.py`)

- `eval/data/kb/`: 22 ShopNest policy docs, 360-620 words each. One fact sheet keeps the numbers consistent across
  docs: 30-day returns, 10 days for electronics, UPI 1-3 / net banking 3-5 / cards 5-7 business days, ₹79
  reverse-pickup fee, and so on.
  - The docs are deliberately messy, like a real help-centre export. Every doc ends with the same
    "For more information... We value your business..." footer, refund timelines are restated in four docs, the
    security warning in three, and there are five tables.
  - The company is fictional; the folder README says so.
- `load_kb()` chunks the docs into 61 passages of 99-315 tokens (median 219). It packs whole paragraphs, lists and
  tables, prefixes each chunk with the doc title, and never splits a sentence.
- `retrieve(query, k=8)` is deliberately generous naive RAG. It takes the bge-small bi-encoder top-8, with no
  reranking and no score floor. The index is cached in `data/runtime/kb_index.npz` and keyed by a hash of the KB.
  A median request carries 1,756 context tokens, and 100% of the seed questions' key facts are somewhere in the
  top-8. Recall is not the problem; there is simply far too much context.
- `kb_questions()` gives 64 seed workload questions (`"author": "seed"`, not the graded eval set): 44 single-fact,
  12 multi-hop across two docs and 8 unanswerable. Each answerable question lists `key_facts`, the numbers and short
  phrases its reference answer depends on, and a test checks that every key fact occurs in its gold docs.

### Stage 3: `RerankContextOptimizer` (`build_context_optimizer`)

1. **Score** every (query, doc) pair with the local cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` (fastembed/ONNX,
   22M parameters, about 65 ms for 8 docs on a laptop CPU).
   - If the model cannot load, it falls back to bge-small cosine, then to IDF-weighted term overlap.
   - The fallback is written into `ContextResult.note` (`FALLBACK: ...`) and logged. It never degrades silently.
2. **Dynamic-k.**
   - If the mode sets `context_min_score`, keep docs with score >= it, in the active scorer's units (raw ms-marco
     logits for the cross-encoder).
   - Otherwise keep docs within `gap` of the best score (default 6.0 logits; `COSTGUARD_RERANK_GAP`).
   - The top-1 doc is always kept.
3. **Near-duplicate docs** (word-trigram Jaccard >= 0.8 with a better-scored kept doc) are dropped. Overlapping
   chunkers produce these all the time.
4. **Budget.** Whole docs are added in score order while `count_text(format_docs(kept))` (o200k) stays
   <= `context_budget_tokens`. A doc that does not fit is skipped and the next smaller one is tried. Nothing is cut
   mid-sentence. The top-1 doc stays even if it alone exceeds the budget.
5. **Order: best first, second-best last** ("lost in the middle", Liu et al. 2023). Models use the start and the end
   of a long context best, so the two strongest docs take the edges and weaker docs sit in between.
   - `COSTGUARD_CONTEXT_ORDER=score|original` switches the order.
   - Evidence retention cannot see ordering, so this choice rests on the literature and must be confirmed with the
     judge.
6. The result is a `ContextResult` with the kept docs, `kept_indices` (into the input list, in final order), their
   `scores`, `tokens_before`/`tokens_after` and a `note`. `optimizer.last_ms` and `optimizer.load_ms` expose
   latency.

### Stage 4: compressors (`build_compressor`, env `COSTGUARD_COMPRESSOR`)

| `COSTGUARD_COMPRESSOR` | Class | What it does | Extra RAM |
|---|---|---|---|
| `heuristic` (default) | `HeuristicCompressor` | query-aware, sentence-level extractive | none (pure Python plus the tokenizer) |
| `llmlingua2` | `LLMLingua2Compressor` | LLMLingua-2 token classification, task-agnostic | torch plus a 0.7-2.2 GB model |
| `none` | `PassthroughCompressor` | no-op with honest token counts (A/B arms) | none |

**Heuristic** (`rate` = fraction of tokens to keep):

1. Split each `[n]` doc into units: sentences, list items (tied to their "...:" lead-in line) and table rows (tied
   to the header row). Markdown separator rows are dropped as formatting.
2. Drop exact and near-duplicate units (word Jaccard >= 0.8 **and the same numbers**, so "UPI 1-3 days" is never
   merged with "UPI 3-5 days") and boilerplate patterns ("For more information...", "We value your business...",
   "reserves the right to modify...").
3. Score each unit by IDF-weighted overlap with the query's stemmed content terms. A sentence passes half its score
   to the next sentence of the same paragraph, because answers often follow the sentence that matches the question:
   "Once an order has shipped, it cannot be cancelled. **You can either refuse the delivery...**"
4. Keep the best units, in original order, until `rate x tokens_before` is reached. Doc markers and titles survive
   for every doc that keeps something; docs with nothing left are dropped whole.
5. **Protected units** contain a number and share at least 2 content terms with the query (1 for very short
   queries), or repeat a number from the query. They are always kept, even past the target.
6. Docs that look like **code or JSON** pass through verbatim.

`COSTGUARD_HEURISTIC_SCORER=embed|hybrid` swaps in bge-small cosine. It reuses the semantic cache's embedder.
Hybrid gained 2 of 56 questions at rate 0.33, but at 450-600 ms per call instead of 4 ms (table below), so `lexical`
is the default.

**LLMLingua-2:**

- Model: `PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
  use_llmlingua2=True, device_map=mps|cuda|cpu)`. `COSTGUARD_LLMLINGUA_MODEL=large` selects the xlm-roberta-large
  model.
- Call: `compress_prompt(docs, rate=rate, force_tokens=['\n','?','.','[',']'], drop_consecutive=True,
  use_context_level_filter=False)`.
- Each doc is compressed separately and its `[n]` marker re-attached, because the model strips the digits inside
  `[1]` (measured: `[1]` becomes `[ ]`).
- It is lazy-loaded; `load_ms` and `last_ms` are recorded.
- If llmlingua or torch is missing, it degrades to the heuristic and says so in `method`.

### Only the volatile block, and only above `compression_min_tokens`

The pipeline compresses the formatted context block only, and only when it holds at least `compression_min_tokens`
(400 in balanced, 250 in economy). The system prompt is never compressed. The reason is the pitfall below.

## The cache-prefix pitfall

Provider prompt caching bills a byte-identical prompt prefix at about **0.1x the input price** on cache reads:
Anthropic cache reads, and OpenAI's newer models (`cached_input` in `configs/prices.yaml`, e.g. gpt-5.4-mini
$0.075 vs $0.75).

- A 3,000-token cached system prompt costs the same as 3,000 x 0.1 = **300 fresh tokens**.
- "Compress" it 2x and you send 1,500 tokens that no longer match the cached prefix, all billed at full price:
  **1,500 token-equivalents, 5x more expensive**.
- You also risk dropping a "not" or an "only" from the instructions.

The rule follows:

- **System prompt, tools and few-shot examples:** first, byte-identical and cached. CostGuard never edits them.
- **Retrieved context:** reranked and compressed.
- **Question:** last and verbatim.

Compression also changes per query, so it must sit after the cache breakpoint. Anything compressed per request can
never be a cache hit.

Even done right, the saving is bounded by `input share x compressible share x (1 - 1/r)`.

- A median ShopNest RAG request has ~1,870 input tokens, 1,756 of them context, and ~100 output tokens, priced at
  gpt-5.4-mini rates ($0.75 in, $4.50 out). So input is ~76% of the bill and context ~94% of the input.
- At the balanced setting (r ≈ 5.4) that is 0.76 x 0.94 x 0.81 ≈ **58% of the bill**.
- Most of that comes from reranking, the stage that cannot break a sentence.

## Results

Measured on 64 seed questions (56 answerable, 12 of them multi-hop), each with its naive top-8 context (median 1,756
tokens). Hardware: Apple-silicon laptop. The cross-encoder and the heuristic run on CPU; LLMLingua-2 runs on MPS
unless noted.

**Evidence retention** is the quality proxy. It is the share of answerable questions whose key facts (the numbers and
short phrases the reference answer depends on) all still appear in the optimised context, among questions whose
facts were in the full context. All 56 were, because retrieval recall at k = 8 is 100%. If the fact is cut, the
answer must degrade, so retention is a cheap upper bound on quality kept. The *Lenient* column ignores filler words,
so "refuse delivery" counts for "refuse the delivery". The 95% CI is Wilson, n = 56.

| Method | Tokens kept | Ratio | Evidence retention (95% CI) | Lenient | p50 ms | p99 ms |
|---|---:|---:|---|---:|---:|---:|
| passthrough | 100% | 1.00x | 100.0% (94%-100%) | 100.0% | 0.0 | 0.0 |
| truncate@400 (no rerank) | 14% | 6.96x | 76.8% (64%-86%) | 76.8% | 0.1 | 0.1 |
| truncate@800 (no rerank) | 40% | 2.53x | 91.1% (81%-96%) | 91.1% | 0.2 | 0.3 |
| truncate@1200 (no rerank) | 61% | 1.63x | 94.6% (85%-98%) | 94.6% | 0.5 | 0.6 |
| rerank gap=6 | 44% | 2.30x | 98.2% (91%-100%) | 98.2% | 65.0 | 106.4 |
| rerank+budget@400 | 15% | 6.62x | 83.9% (72%-91%) | 83.9% | 67.5 | 110.0 |
| rerank+budget@800 | 31% | 3.23x | 94.6% (85%-98%) | 94.6% | 62.1 | 82.0 |
| rerank+budget@1200 | 38% | 2.61x | 96.4% (88%-99%) | 96.4% | 87.2 | 150.1 |
| heuristic-lexical@0.33 | 32% | 3.13x | 96.4% (88%-99%) | 96.4% | 4.0 | 9.4 |
| heuristic-lexical@0.5 | 48% | 2.10x | 100.0% (94%-100%) | 100.0% | 4.0 | 5.5 |
| heuristic-lexical@0.7 | 67% | 1.50x | 100.0% (94%-100%) | 100.0% | 4.2 | 5.6 |
| heuristic-hybrid@0.33 | 32% | 3.13x | 100.0% (94%-100%) | 100.0% | 594.4 | 1105.3 |
| heuristic-hybrid@0.5 | 48% | 2.10x | 100.0% (94%-100%) | 100.0% | 447.1 | 1042.8 |
| rerank@1200+heuristic@0.5 | 18% | 5.45x | 96.4% (88%-99%) | 96.4% | 68.0 | 136.5 |
| rerank@800+heuristic@0.33 | 11% | 9.21x | 82.1% (70%-90%) | 82.1% | 71.9 | 92.0 |
| llmlingua2@0.33 | 34% | 2.95x | 30.4% (20%-43%) | 41.1% | 214.8 | 270.4 |
| llmlingua2@0.5 | 51% | 1.96x | 51.8% (39%-64%) | 66.1% | 206.6 | 234.8 |
| llmlingua2+digits@0.5 | 51% | 1.95x | 51.8% (39%-64%) | 64.3% | 207.9 | 215.0 |
| rerank@1200+llmlingua2@0.5 | 20% | 5.10x | 51.8% (39%-64%) | 64.3% | 154.3 | 237.7 |
| llmlingua2-large@0.33 | 36% | 2.74x | 39.3% (28%-52%) | 51.8% | 657.9 | 1095.2 |
| llmlingua2-large@0.5 | 52% | 1.91x | 53.6% (41%-66%) | 67.9% | 789.9 | 863.4 |

What the table says:

- **Reranking is the big, safe lever.**
  - The cross-encoder with the relative gap alone sends 2.3x fewer tokens and keeps 98.2% of the evidence.
  - Naive truncation to the same budget (no reranker) keeps less: at 800 tokens, 91.1% vs 94.6%, and the reranked
    context is smaller (31% vs 40% of tokens), because dynamic-k stops early when only 1-2 chunks matter.
  - Every reranking miss is a multi-hop question. Single-fact retention is 100% at every budget of 800 or more.
- **The heuristic compressor is nearly free.**
  - 4 ms per call, no model, and 100% retention at rate 0.5 (2.1x).
  - Stacked on reranking (rerank@1200 + heuristic@0.5), it sends **5.45x fewer context tokens at 96.4% retention**,
    and adds only ~4 ms on top of the rerank.
  - Rows in this table compress unconditionally. The pipeline skips compression below `compression_min_tokens`,
    which the grid below simulates: today's balanced policy measures 4.99x at 96.4% there.
  - The hybrid scorer (lexical + bge-small) kept 2 more questions at rate 0.33 (100% vs 96.4%; within noise at
    n = 56). It costs 450-600 ms per call instead of 4 ms.
- **LLMLingua-2 is the wrong tool for policy text.**
  - At rate 0.5 it keeps only 51.8% of the key facts (66% lenient); the xlm-roberta-large model keeps 53.6%.
  - It costs ~207 ms per call on MPS, and 385 ms p50 / 1.29 s p99 on the laptop CPU.
  - It drops units and range structure ("UPI 1–3, net banking 3–5 business days" -> "UPI refunds 1 3 net banking
    3") and negations (see below).
  - Forcing digits (`force_reserve_digit`) does not help: the digits were never the problem, the words around them
    were.
  - Memory: peak RSS reached 1.6 GB after loading the mBERT model and 4.1 GB with the large one, against 443 MB for
    bge-small + cross-encoder. That rules it out on a free 512 MB host.
- **Unanswerable questions still carry context.** The relative gap keeps ~1,260 tokens on average for the 8
  unanswerable questions, because when nothing is relevant everything is "close to the best". An absolute floor
  (`context_min_score`, RECOMP-style) is the knob for this. It is not enabled, because an empty context changes how
  the model answers and needs judge validation first.

### Recommended per-mode settings

From a grid of `context_budget_tokens ∈ {none, 2000, 1600, 1200, 1000, 800, 600, 400}` × heuristic
`compression_rate ∈ {off, 0.7, 0.5, 0.33}`, each mode gets the most aggressive point (fewest tokens sent) whose
retention clears its bar. The grid simulates the pipeline exactly: rerank, then compress only if the block is
>= `compression_min_tokens`.

| Mode | Bar | context_budget_tokens | compression_rate | Tokens kept | Retention (95% CI) |
|---|---:|---:|---|---:|---|
| quality | 98% | 1600 | 0.7 | 29% | 98.2% (91%-100%) |
| balanced | 95% | 1000 | 0.5 | 19% | 96.4% (88%-99%) |
| economy | 90% | 800 | 0.5 | 16% | 92.9% (83%-97%) |

How this compares with `configs/policy.yaml` today:

- **quality:** today budget 2000 with compression off: 43.5% of tokens kept, 98.2%. The grid's pick is budget
  1600 with compression on at 0.7 (29% kept, same retention). The policy currently disables compression in quality
  mode; the conservative alternative is budget 1600 with no compression (42.5% kept, 98.2%).
- **balanced:** today budget 1200 at rate 0.5: 20% kept, 96.4%, which already passes. The pick, budget 1000, trims to
  19% at the same retention. Rerank-only alternative: budget 1000 (35% kept, 96.4%).
- **economy:** today budget 800 at rate 0.33 measures **85.7%, below the 90% bar**. The pick keeps budget 800 and
  raises the rate to 0.5 (16% kept, 92.9%). Rerank-only alternative: budget 600 (25% kept, 91.1%).

The coordinator owns `policy.yaml`. The numbers are in `eval/results/compression_eval.json` under `recommendation`.
With n = 56, the 98% bar means at most one miss: treat these as starting points for the paired judge test, not as
guarantees.

### Dynamic-k gap (cross-encoder logits, no budget)

| Gap | Retention | Ratio | Docs kept (mean of 8) |
|---:|---:|---:|---:|
| 2 | 83.9% | 4.29x | 1.86 |
| 3 | 87.5% | 3.70x | 2.17 |
| 4 | 91.1% | 3.15x | 2.53 |
| 5 | 96.4% | 2.66x | 3.02 |
| 6 | 98.2% | 2.30x | 3.48 |
| 8 | 98.2% | 1.75x | 4.61 |
| 10 | 98.2% | 1.44x | 5.55 |

Retention plateaus at gap 6. Gaps 8 and 10 only add tokens; the one remaining miss is the multi-topic chunk below.
Default: 6.0.

## When compression hurts

- **Numbers and ranges.** Token droppers keep digits but lose units and range structure.
- **Negation and qualifiers.** LLMLingua-2 dropped "never" from "ShopNest Plus members **never** pay the
  reverse-pickup fee", producing "ShopNest Plus pay reverse pickup fee", which reverses the policy. The heuristic
  keeps or drops whole sentences, so it cannot flip a sentence. It can still drop the *next* sentence that carries
  the exception, which is why it passes score on to the next sentence and protects numeric facts.
- **Code and JSON.** Dropping tokens breaks syntax, and dropping lines breaks semantics. Both compressors detect
  code fences, code-like lines and `json.loads`-able docs and pass them through verbatim. A unit test checks that
  a JSON doc survives byte-for-byte.
- **Instructions** (system prompts, output formats, tool schemas). They are short, cached and safety-relevant, so
  compression never touches them. Removing one "do not" costs far more than the tokens it saves.
- **Tables and lists.** Row-by-row dropping orphans values from their column names. The heuristic ties each row to
  its header and drops only the `|---|` separator. On the international-shipping chunk at rate 0.5, LLMLingua-2:
  - kept "Nepal ₹599 5" and "UAE Singapore ₹1, 499";
  - dropped the entire "UK, USA, Canada, Australia | ₹2,499" row and the "Shipping fee" column name;
  - deleted "the United States" from the list of countries ShopNest ships to.
- **Multi-topic chunks.** Reranking is chunk-level. In the one question that every reranking setting misses
  (kbq-053, NestCoins earned during the sale), the fact sits in a short paragraph at the end of a chunk that is
  mostly about sale returns and delivery. The cross-encoder scores that chunk -7, far below the top. Smaller or
  section-aligned chunks would fix it; a looser gap does not (the gap sweep plateaus at 98.2%).
- **Short contexts.** Below a few hundred tokens the absolute saving is tiny. A long-lived LLMLingua-2 model still
  costs about 200 ms per call (MPS) and much more on CPU, and every cut risks a fact. Hence `compression_min_tokens`.

## Trade-offs

- **We chose whole-chunk dropping (rerank + dynamic-k + budget) over token dropping for retrieved context, because**
  retrieved chunks are the natural unit of relevance. A chunk is either about the question or not, so dropping it
  whole loses no grammar, no negation and no table structure. On our KB, rerank+budget@800 keeps 31% of the
  context tokens at 94.6% evidence retention, while LLMLingua-2@0.5 keeps 51% at 51.8%. This matches Jha et al.
  (2024), where extractive selection often beats token pruning at up to 10x.
- **We chose the heuristic compressor for the live proxy and LLMLingua-2 offline only, because** the free host has
  about 512 MB of RAM. Torch plus the smallest LLMLingua-2 model peaked at 1.6 GB RSS (large: 4.1 GB), against 443 MB
  for bge-small plus the cross-encoder.
  The heuristic needs no model, runs in about 4 ms, and measured better evidence retention than LLMLingua-2 at
  every rate on this content. LLMLingua-2 stays as a measured, selectable arm (`COSTGUARD_COMPRESSOR=llmlingua2`)
  for beefy hosts and for the A/B harness.
- **We chose a local cross-encoder (ms-marco-MiniLM-L-6-v2) over Cohere Rerank, because**:
  - it costs $0 per call against $2.00-2.50 per 1,000 searches;
  - it adds about 65 ms with no network round-trip and no third-party data processing;
  - it is already baked into the Docker image, so a cold start needs no download.
  The quality gap is not decisive at k = 8. The local reranker reaches 98.2% evidence retention with the relative
  gap alone, and its one miss is the chunking problem above, which a better reranker would not fix either.
- **We chose a relative score gap over a fixed top-k, because** the number of relevant chunks varies per question
  (1 for "What's the COD limit?", 2-3 for multi-hop questions). On this KB, fixed truncation to 800 tokens without
  reranking keeps 91.1% of the evidence; reranking with the same budget keeps 94.6% at a smaller size.
- **We chose lexical scoring over embeddings inside the heuristic, because** the hybrid scorer gained 2 questions at
  rate 0.33, within noise at n = 56, and none at 0.5. It costs 450-600 ms per call instead of 4 ms, which is more
  than the reranker itself. It is one env var away (`COSTGUARD_HEURISTIC_SCORER=hybrid`) if the hand-written eval
  set shows the paraphrase gap matters.
- **We chose an LLM-free evidence-retention proxy as the gate, with the judge as confirmation, because** it is
  deterministic, free and runs in CI. It is an upper bound: if the fact is gone, the answer must degrade. It misses
  paraphrase and reasoning failures, so `--with-llm` (and the A/B harness) confirm the chosen setting with the
  judge.

## Limitations

- **Small n.** Each retention number covers 56 answerable questions, so a 95% CI spans about ±7-8 points, and the
  98% "quality" bar means at most one miss. Read the recommendation as a starting point for the paired judge test,
  not as a calibrated guarantee.
- **Vocabulary leakage.** The seed questions were written with the KB open, so they share wording with it. That
  flatters the lexical heuristic: real customers paraphrase ("money back" for "refund"). Re-measure on the
  hand-written eval set before trusting the heuristic's numbers.
- **Strict matching.** The strict metric needs the exact phrase ("1-3 business days"). The `lenient` column ignores
  filler words, which is fairer to token-level compressors. The ranking of methods does not change.
- **Ordering** ("lost in the middle") is not measurable by this proxy at all.

## Run it

```bash
./.venv/bin/python -m eval.compression_eval                    # ~4 min; writes eval/results/compression_eval.json
./.venv/bin/python -m eval.compression_eval --fast             # skip LLMLingua-2
./.venv/bin/python -m eval.compression_eval --llmlingua-large  # also the xlm-roberta-large model (2.2 GB download)
./.venv/bin/python -m eval.compression_eval --with-llm         # + generate/grade a subset (needs eval/judge.py + backend)
./.venv/bin/python -m eval.compression_eval --markdown         # print the tables in this doc from the JSON
./.venv/bin/python -m pytest -q tests/test_context.py          # RUN_SLOW=1 adds the LLMLingua-2 test
```

| Env var | Default | Meaning |
|---|---|---|
| `COSTGUARD_COMPRESSOR` | `heuristic` | `heuristic` \| `llmlingua2` \| `none` |
| `COSTGUARD_HEURISTIC_SCORER` | `lexical` | `lexical` \| `embed` \| `hybrid` |
| `COSTGUARD_LLMLINGUA_MODEL` | `bert` | `bert` (mBERT-base) \| `large` (xlm-roberta-large) |
| `COSTGUARD_RERANKER` | `cross-encoder` | `cross-encoder` \| `bi-encoder` \| `lexical` |
| `COSTGUARD_RERANK_GAP` | 6.0 (cross-encoder logits) | relative dynamic-k gap |
| `COSTGUARD_CONTEXT_ORDER` | `edges` | `edges` (best first, second-best last) \| `score` \| `original` |
| `FASTEMBED_CACHE_PATH` | `models/fastembed` | where the reranker and embedder weights live (shared with the semantic cache) |
