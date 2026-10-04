# CostGuard: 15-minute presentation outline

**Slot:** Team 8, Sunday 11 October, 4:00–4:20 PM. 15 minutes of talk, then Q&A.

**Reviewers:**

- **Team 10, Guardrails & Safety Layer.** Their core metric, catch rate vs false positives, mirrors our hit rate vs false hits.
- **Team 6, RAG Q&A with Automated Evaluation in CI.** They will probe eval-set provenance, the CI gate and judge reliability.

**Assume both have already seen a semantic-cache pitch that day.** Spend about 10 seconds defining caching and the rest on measured evidence.

**Rubric weights → time:**

| Criterion | Weight | Time |
|---|---|---|
| Problem framing | 20% | 3:00 |
| Architecture | 30% | 4:30 |
| LLMOps depth | 20% | 3:00 |
| Trade-offs | 20% | 3:00 |
| Presentation quality (delivery; results slide) | 10% | 1:30 |

Six speakers × 2:30, so time is shared equally.

**Rule for every slide: no numbers, no credit.** Each number on a slide names its source file in small print. Fill numbers from `eval/results/*.json` (via `make report`); never type them by hand.

---

## Speaker split

| Speaker | Owns (workstream) | Time | Covers |
|---|---|---|---|
| S1 | Framing / product | 0:00–2:30 | Slides 1–3 |
| S2 | Gateway + caching | 2:30–5:00 | Slide 4 (scope, 0:30) + slides 5–6 |
| S3 | Context, compression, router | 5:00–7:30 | Slides 7–9 |
| S4 | Eval + CI | 7:30–10:00 | Slides 10–12 |
| S5 | Ops / monitoring | 10:00–12:30 | Slide 13 + slide 14 (first half) |
| S6 | Results / close | 12:30–15:00 | Slide 14 (second half) + slides 15–16 |

Each speaker also owns the Q&A questions about their slides (see the end of this document).

---

## Slides

### Framing (3:00)

**1. Title + the one-line result** (S1, 0:30)

- "CostGuard: an OpenAI-compatible gateway that cut ShopNest's LLM bill by **X%** at **Y%** quality retained, with no change to the calling code."
- X = the A5 arm's `savings_pct` with `savings_ci`; Y = its `quality.retained` with `quality.retained_ci`. Take both from the real-run summary: `eval/results/ab_summary_mlx.json` for the local MLX run (dollars are list-price equivalents at GPT-5.4-mini / GPT-5-nano prices, so say "list-price equivalent"), or `ab_summary.json` for a paid-backend run. Never `ab_summary_mock.json`.

**2. Problem and business objective** (S1, 1:00)

- The objective in one sentence (ARCHITECTURE §0).
- Business metric: **$ per correct answer**, not $ per call.
- The chain: higher true hit rate + fewer input tokens per miss + a safe cheap-tier share → lower $ per request.
- The guardrails on that chain: quality ≥ 95%, false hits ≤ the budget, hit-path overhead p99 < 50 ms.
- Say Goodhart out loud: maximising hit rate alone lowers τ and buys false hits.

**3. The ML problem: three prediction targets** (S1, 1:00)

- (a) Is query q cache-equivalent to a cached q′? A binary decision, cosine ≥ τ plus guards.
- (b) Which context tokens can be dropped?
- (c) Is downshifting safe for this request class?
- Inputs: an OpenAI-format request plus the caller key. Outputs: an answer plus route, tokens, $, latency and `config_hash`.
- The cost of being wrong: a false hit is a confidently wrong answer with no error signal.

### Architecture (4:30)

**4. Requirements with values + scope** (S2, 0:30)

- Show the NFR table (ARCHITECTURE §4). Each value sits next to the design decision it forces.
- One line on scope: no streaming, no guardrails, no billing.

**5. End-to-end diagram** (S2, 1:00)

- The mermaid diagram from ARCHITECTURE §1. Every edge is labelled sync/async, protocol and format.
- Two paths:
  - the **synchronous request path**: gateway → exact → semantic → context → compress → route → Anthropic;
  - the **asynchronous telemetry path**: Prometheus pull, a Langfuse background thread, the SQLite log feeding the dashboard and eval.
- Call out service-side tenancy: the caller key decides tenant and mode.

**6. Caching: exact → semantic → guards** (S2, 1:00)

- The threshold curve from `threshold_sweep.json`: hit rate and **per-request** false-hit rate against τ, with the recommended τ per mode. Label the hit rates as an upper bound: Bitext is template paraphrases, so nearly every query has a close neighbour.
- The guard effect: false hits removed vs correct hits lost.
- One trap example: "cancel #4821" vs "don't cancel #4822" at about 0.9 cosine.
- Why the exact cache sits in front: zero risk, ~1 µs.

**7. Context and compression** (S3, 1:00)

- Cross-encoder rerank → dynamic-k → whole-doc budget → "lost in the middle" ordering.
- The evidence-retention vs tokens-kept curve from `compression_eval.json`.
- We never touch the system prompt, so Anthropic prefix caching survives (cache reads cost 10% of the input price).

**8. Router + eval gate** (S3, 0:45)

- The per-category gate table from `configs/router_gate.json`: allow/block with the lower bound of the paired CI.
- Anything uncertain goes to strong.
- **Be upfront:** Sonnet → Haiku is only a **2× price gap**, so routing is a small lever and caching dominates. The measured MLX runs are billed at GPT-5.4-mini → GPT-5-nano (~12×), so their routing step overstates what the Anthropic pair would save. Say which pair each number uses.

**9. Latency budget** (S3, 0:45)

- The table from ARCHITECTURE §3, budget vs measured, from `loadtest.json`. Hit-path overhead p99 is **3.6 ms** against a 50 ms budget.
- Miss-path overhead p99 is 94.6 ms against a 100 ms budget.
- RAG misses reach p99 105.8 ms because of the reranker (p99 104 ms). It is the stage over budget, and we say so.

### LLMOps depth (3:00)

**10. Offline + online evaluation** (S4, 1:00)

- Offline:
  - a frozen trace with SHA-256 + seed;
  - cassettes make re-analysis $0 and deterministic;
  - a cumulative ablation (A0–A5);
  - paired bootstrap CIs;
  - a hand-written domain eval set (until the team's rows land, only AI-written seed rows exist; say so);
  - a judge with position swap, plus a human-agreement check.
- Online:
  - offline shadow replay of logged queries;
  - a caller/session-level canary;
  - a sampled judge on semantic hits and downshifts.

**11. The CI gate blocks a bad PR** (S4, 1:00)

- **The screenshot:** "Lower τ to 0.6 for more savings". Savings go up, `eval-gate` goes red, merging is blocked. Then the revert goes green.
- `tests` and `eval-gate` are required checks; there are no keys in CI (RUNBOOK §8).

**12. Rollout and rollback** (S4, 0:30)

- Offline gate → shadow → canary 0.5 → 1 → 5 → 10 → 50% → full, with numeric promote and rollback rules (RUNBOOK §3).
- Rollback = revert `policy.yaml`; `config_hash` on every log row proves the cutover.
- `router_gate.json` hot-reloads.
- Kill switch: `mode: "off"`, with quotes (the YAML gotcha).

**13. Monitoring, all 5 categories** (S5, 1:00)

- The table from RUNBOOK §7.
- Dashboard screenshot: Monitoring tab with cost/latency/hit-rate series, PSI bars with the 0.10 / 0.25 bands, and the alert-check table.
- The drift is course W3S2 PSI on TraceRecord features. It needs no embeddings in the hot path, and the same maths runs online (`/v1/drift`, Prometheus gauges) and in the dashboard.
- Tools named and justified: Prometheus, SQLite + Streamlit, Langfuse (async, sampled to the 50k-unit tier), GitHub Actions.

### Trade-offs (3:00)

**14. Five trade-offs in the rubric's form** (S5 2:00 → S6 1:00)

The rest are in DESIGN_DECISIONS §1; keep them as backup slides.

1. Exact cache in front of the semantic cache: zero false-hit risk.
2. τ from a per-request false-hit budget, not the hit-rate maximum.
3. Compress volatile context only, which keeps prefix caching.
4. Downshift only behind a per-category CI gate, Sonnet → Haiku: with a 2× gap, caching matters more.
5. Fail open plus async telemetry: a cost layer must never lower availability or add latency.

Plus one line on rejected alternatives: self-hosting is below the ~$10k/month break-even; a global cache would leak across tenants; TTL-only invalidation serves stale answers.

**15. Failure modes + what breaks at 10×** (S6, 0:30)

- Four rows from the failure table (DESIGN_DECISIONS §2): false hit, stale answer, embedding change, provider outage.
- What breaks first at 10×: the CPU-bound reranker and embedder on the miss path, then one worker's threadpool, then the per-replica caches, then provider rate limits.

### Results and delivery (1:30)

**16. Results** (S6, 1:30)

- The savings waterfall by lever (dashboard tab, from the real-run summary, e.g. `ab_summary_mlx.json`; never the mock file).
- The A/B table with CIs.
- Cost per correct answer, baseline vs CostGuard.
- Load test: **128 req/s, 0 failures**; hit-path overhead p99 3.6 ms; miss-path p99 94.6 ms; mock upstream.
- Close with the honest headline. Compression is big in tokens but small in dollars (output tokens cost 5× input on Anthropic, 6–8× at the GPT prices the MLX run is billed at), and on Anthropic routing is capped by the 2× price gap.
- The resume line.

---

## Likely Q&A

Each question has an owner and a short answer, with the file that proves it.

### From Team 10 (guardrails: catch rate vs false positives, the same trade-off as ours)

1. **"How did you pick τ? Wasn't it eyeballed?"** (S2)
   - No. It is read off a curve of per-request false-hit rate against hit rate on labelled pairs plus a replay stream, with a budget per mode (0.5/1/3%).
   - The recommended τ is the highest hit rate inside the budget (`threshold_sweep.json`).
   - Per-hit precision would flatter low τ values.
2. **"What do your hard negatives look like?"** (S2)
   - Hand-written trap pairs: negations, different order numbers, different products, refund vs exchange.
   - The guards are evaluated on them with on/off curves (`guard_effect`). We show one example.
3. **"Your guards are regex heuristics. How do you know they don't block good hits?"** (S2)
   - We report "correct hits lost" next to "false hits removed" for each τ.
   - They are a project heuristic. We say so, and there is no published benchmark for them.
4. **"What happens when a cached answer is wrong in production?"** (S5)
   - Detected by the sampled judge on semantic hits and the least-similar accepted hits view.
   - Recovered by evicting the entry or bumping `kb_version`, adding the pair to the traps, and raising τ for that mode.
   - The fail-safe is `mode: "off"` per tenant.
5. **"Could a cache leak one customer's answer to another?"** (S2)
   - The partition is tenant | system-prompt hash | kb_version | ctx.
   - The tenant comes from the caller's API key, never the body.
   - Single-turn only: anything with history bypasses the cache.
6. **"Shadow mode?"** (S4)
   - Offline shadow: we replay logged and frozen traffic through the candidate policy.
   - Live shadow-serving (compute the cached answer, serve the model's) is designed but out of scope; it is in ARCHITECTURE §5.

### From Team 6 (RAG with eval in CI)

7. **"Where does your eval set come from? Is it contaminated by the trace?"** (S4)
   - Hand-written by the team, with authors recorded per row; AI seeds are labelled `"author": "seed"`.
   - Be honest about the current state: until the team's files land, the only rows are the 44 AI-written seed rows, and the router gate falls back to a Bitext + KB sample.
   - The workload trace (Bitext + KB questions) is separate from the graded set.
8. **"How reliable is your judge?"** (S4)
   - Position-swapped pairwise judging; disagreement counts as a tie. Reference-guided grading.
   - A human-agreement check on a labelled subset.
   - Temperature 0, cached in cassettes so CI is deterministic.
9. **"How does the CI gate avoid flakiness?"** (S4)
   - Cassette replay: the same PR gets the same verdict.
   - The gate is mostly deterministic assertions (traps must miss).
   - Margins are set above the run-to-run noise measured on `main`.
10. **"Does compression hurt faithfulness?"** (S3)
    - Measured as evidence retention: do the key facts survive in the optimised context, with a CI (`compression_eval.json`).
    - Number-bearing sentences are protected. We never compress the system prompt or the question.
11. **"Why rerank rather than retrieve better?"** (S3)
    - We are a drop-in layer and don't own the caller's retriever.
    - Retrieval is deliberately generous (top-8), so trimming it is where the tokens are.

### General and examiner questions

12. **"How is this different from a gateway like LiteLLM or Portkey?"** (S6)
    - Gateways ship caching and routing.
    - We ship the **evidence**: per-request false-hit rate, quality deltas with CIs per lever, an eval-gated downshift, and a fixed-trace A/B. We also block a bad config in CI.
13. **"Why Anthropic's SDK, not LiteLLM?"** (S3)
    - Exact cache read and write usage, so costs are billed correctly.
    - Fewer layers, and an explicit endpoint that ignores `ANTHROPIC_BASE_URL` (DESIGN_DECISIONS #10).
14. **"Isn't 2× too small a gap for routing to matter?"** (S3)
    - Yes, and we say so. Routing saves at most 50% of a routed request, while a hit saves 100%, including output tokens at 5× the input price.
   - The MLX run is billed at a ~12× pair, so its routing share is an upper bound for the Anthropic deployment.
    - That is why the levers are ordered as they are.
15. **"What is your p99, and does CostGuard slow things down?"** (S5)
    - Hit-path overhead p99 3.6 ms; miss path 94.6 ms; 128 req/s with 0 failures (`loadtest.json`, mock upstream).
    - The reranker is the stage to scale.
16. **"What breaks first at 10×?"** (S6)
    - The CPU-bound reranker and embedder on the miss path, then one worker's threadpool, then per-replica caches (move to Redis/Qdrant), then provider rate limits (DESIGN_DECISIONS §3).
17. **"Is the 30% duplicate rate realistic?"** (S6)
    - It follows MeanCache's per-user figure, labelled as such.
    - The trace builder takes `--dup-rate`, so we re-run at 15% and 0% duplicates as sensitivity arms. At 0%, all savings must come from context, compression and routing.
