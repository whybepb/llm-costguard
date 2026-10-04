# Model downshift: router and eval gate

**Question this component answers:** for this request, can a cheaper model give an acceptable answer, and can we prove it?

The router (stage 5 of `costguard/pipeline.py`) picks the `strong` or `cheap` tier before the upstream call. It is deliberately simple. The safety comes from an **offline eval gate**, which decides per category whether downshifting is allowed, using a confidence interval fixed in advance. The router only enforces that decision.

| Mode (`configs/policy.yaml`) | Router policy | Behaviour |
|---|---|---|
| `off`, `quality` | none | Always the requested tier, strong by default. |
| `balanced` | `gated` | Downshift only categories the gate allowed, and only requests with no hardness signal. |
| `economy` | `aggressive` | Downshift every request with no hardness signal. No gate. |

Code:

- `costguard/router/router.py` holds `build_router` and `GatedRouter`.
- `costguard/router/features.py` holds the hardness signals.
- `costguard/router/classifier.py` holds category inference, plus `centroids.npz`.
- `eval/gate_router.py` is the gate.
- `configs/router_gate.json` is the gate's output, which the router reads live.
- `tests/test_router.py` has the tests.

## 1. Design

### Decision order

The first rule that fires wins, and every decision is logged as `TraceRecord.route_reason`:

| # | Condition | Tier | `route_reason` |
|---|---|---|---|
| 1 | The client asked for `model: "cheap"` | cheap | `requested-cheap` |
| 2 | Any hardness signal | strong | `hard:<signal>` |
| 3 | Policy `aggressive` | cheap | `aggressive:easy` |
| 4 | Gate file missing, unreadable or dry run | strong | `gated:no-gate-file`, `gated:bad-gate-file`, `gated:dry-run-gate` |
| 5 | Category sent but not one of the 8 | strong | `gated:unknown-category` |
| 6 | No category sent, and inference is unsure | strong | `gated:unclassified` |
| 7 | Category not in the gate file | strong | `gated:<cat>-not-evaluated` |
| 8 | Gate allows the category | cheap | `gated:<cat>-allowed` (or the `shadow`/`canary` stages in section 4) |
| 9 | Gate blocks the category | strong | `gated:<cat>-blocked` |

Every unexpected state ends on strong. A failure inside the router can't break serving either: the pipeline's fail-open `timed()` wrapper keeps the requested tier.

### Hardness signals (`features.py`)

These are regexes and token counts, deterministic and with no I/O. They run in all policies.

| Signal | Fires on |
|---|---|
| `escalation` | Anger, legal or fraud words (angry, lawyer, fraud, chargeback, dispute, unauthorised, complaint…), profanity, `!!`, ALL CAPS. |
| `non-english` | Over 20% non-Latin letters, or foreign function words outnumbering English ones. Covers es, fr, de, pt, it and Hinglish (`mera refund kab aayega`). |
| `code` | Code fences, stack traces, `TypeError`, JSON or HTML, `curl`, `HTTP error 500`. |
| `arithmetic` | Arithmetic expressions, two or more money amounts, a percentage with numbers, or a price question about a stated quantity ("7 kg to Canada, what will it cost?"). |
| `reasoning-keywords` | why, compare / vs, difference between, troubleshoot / won't connect, calculate, how much would, step by step, what happens if, is it better to … or …, "if I …, what …". |
| `multi-question` | Two or more questions, enumerated lists, two fused asks ("how much, and when?"), or "also" across sentences. |
| `long-query`, `long-input`, `deep-history`, `many-context-docs` | Query over 120 tokens, prompt over 4,000 tokens, more than 4 prior messages, more than 8 retrieved docs. These are configurable under `router.hardness` in policy.yaml. |

Measured rates (`eval/results/router_gate_dryrun.json`, plus `kb_questions()`):

| Set | Items flagged |
|---|---|
| Seed eval set, type `hard` | 6/7 |
| Seed eval set, type `answerable` | 3/25 (all genuine two-part questions) |
| Seed eval set, type `trap_pair` | 0/12 |
| Bitext | 5.6%, almost all profanity → escalation |
| KB `multi_hop` questions | 2/12 |
| KB single-fact questions | 1/44 |

The rules were extended after reading the seed set's 7 hard items, so 6/7 is optimistic. The KB questions were never used for tuning, and their 2/12 is the honest recall. **Low recall is acceptable by design:** the gate measures cheap-tier quality on every item the router would downshift, so missed hard items are already inside the measured quality difference. The signals cut risk; they are not what makes routing safe.

### Category inference (`classifier.py`)

The gate decides per category, but clients don't always send `costguard.category`. When it's missing, the router infers it with a nearest-centroid classifier over `BAAI/bge-small-en-v1.5` embeddings. This is the same local fastembed model the caches and KB retrieval use, so it costs $0.

- **Centroids:** one per source intent, 43 in total, saved in `costguard/router/centroids.npz` (63 KB). Serving needs no dataset.
- **Bitext training data:** 10,400 Bitext rows, mapped onto our 8 categories:

  | Bitext intents | Our category |
  |---|---|
  | ORDER | order |
  | DELIVERY, SHIPPING | shipping |
  | REFUND | refund |
  | PAYMENT, INVOICE | payment |
  | ACCOUNT, SUBSCRIPTION | account |
  | CONTACT, FEEDBACK | other |

  CANCEL/cancellation-fee rows are excluded: they're contract terms with no store equivalent.
- **Seed data:** 555 templated, hand-written seeds. Bitext has no returns or product intents, and none of ShopNest's UPI/EMI/COD, warranty or installation topics.
- **Confidence floor:** if top similarity is below `min_similarity` (0.704), the category is `None`. The floor is the 95th percentile of top similarity over 26 off-topic queries, and rejects 100% of a held-out off-topic set.
- **Margin rule (gated only):** an inferred category is used only if it beats the best other category by at least 0.03 cosine. A pure keyword guess is never trusted for a downshift.
- **Fallback:** keyword rules if the embedder can't load (no model files and no network). The gated policy then downshifts only requests that carry an explicit category.

Accuracy, from `python -m costguard.router.classifier train`, written to `eval/results/router_classifier.json`. "Gated" means what the router acts on after the margin rule:

| Test set | n | Accuracy | Gated coverage | Gated misrouted |
|---|---|---|---|---|
| Bitext held-out (in-distribution) | 2,600 | 99.4% | 98.2% | 0.1% |
| Seed frames held out by template | 169 | 89.9% | 71.6% | 0.6% |
| Team seed eval set (`evalset/seed.jsonl`) | 44 | 84.1% | 72.7% | 2.3% |
| KB questions (`eval.kb.kb_questions`) | 64 | 67.2% | 53.1% | 6.2% |

Two caveats on these numbers:

- **Earlier, blind results.** On the two out-of-distribution sets, the first run scored 70% and 58%. The `*_kb` seed frames (warranty, installation, sizing, damaged items) were then written from the KB's section headings. They were not written from those questions, but the numbers are no longer fully blind. Re-run `train` when the team's hand-written eval rows land.
- **Why the gap.** Most remaining errors are either genuinely two-topic questions ("return it, and how long will the refund take?") or ShopNest-only concepts (NestCoins, Plus, sales). The two eval sets label those concepts inconsistently, so they are left to fall below the floor and stay on strong.

Router overhead: 0.03 ms p50 with an explicit category, 1.9 ms p50 when it has to embed the query.

## 2. Why simple features plus a statistical gate

- **Independent benchmarks don't reward router sophistication.**
  - RouterArena (ICLR 2026) found that no router tops every metric. It ranks Not Diamond #12 because it "frequently selects expensive models". The best open routers reach about 35% lower cost at under 2% accuracy loss ([arXiv 2510.00202](https://arxiv.org/html/2510.00202v1)).
  - LLMRouterBench (33 models, 21 datasets) finds that several recent methods, "including commercial routers, fail to reliably outperform a simple baseline", and that "backbone embedding models have limited impact" ([arXiv 2601.07206](https://arxiv.org/abs/2601.07206)).
- **Pretrained routers don't transfer for free.** RouteLLM's `mf` router was trained on 2024 Chatbot Arena GPT-4/Mixtral preferences. Its authors warn that real traffic can differ from benchmarks, so its threshold would need recalibrating on our traffic, and we have no labelled preference data for our domain ([arXiv 2406.18665](https://arxiv.org/html/2406.18665)).
- **What a simple router gives instead:** explainable reasons on every request, no training data, under 2 ms, and a gate that measures **our** model pair on **our** questions. The gate is where the evidence lives.

## 3. Economics: the price gap decides whether downshifting pays

The gate prints this on every run (`price_gap` in the results JSON). Prices come from `configs/prices.yaml` via `PriceBook(…, policy.billing_for(backend))`. The examples assume a request of 1,500 input and 400 output tokens.

| Backend pair | Strong / cheap per request | Cheap as % of strong | Saving per downshifted request | Routing 50% of traffic saves |
|---|---|---|---|---|
| **anthropic**: Sonnet 5.5 → Haiku 4.5 | $0.0070 / $0.0035 | 50% | **50%** | **25%** |
| openai: GPT-5.4-mini → GPT-5-nano | $0.00293 / $0.00024 | 8% | 92% | 46% |
| gemini: 2.5 Flash → 2.5 Flash-Lite | $0.00145 / $0.00031 | 21% | 79% | 39% |

**The real experiments run on Anthropic, which is only a 2× gap. Be honest about what that means:**

- **Router savings are capped.** Savings ≈ r × (1 − p_cheap/p_strong) = r × 50%. Even with every category allowed and 85% of requests easy (r ≈ 0.85), routing alone saves about 42% of upstream spend. If half the traffic is downshifted, it saves at most about 25% on that slice. That ceiling is before any category is blocked.
- **A cheap-first cascade barely pays.** Cascade cost ≈ p_cheap + e × p_strong (+ verifier). It breaks even at e = 50% escalation and saves only 30% at 20% escalation, before paying a verifier. We therefore route before generating instead of cascading (see trade-offs).
- **Prompt caching can erase the gap.** Sonnet 5.5 caches prefixes from 512 tokens; Haiku 4.5 only from 4,096 (Anthropic prompt-caching docs, via the research notes). Take a 3,000-token shared prefix plus 500 new input and 400 output tokens. Sonnet reads the prefix at $0.20/M and costs $0.0056; Haiku can't cache it and costs $0.0055. **The downshift then saves 1.8%, not 50%.**
  - **Today:** the system prompt is about 110 tokens, below both minimums, and retrieved context sits after the cache breakpoint. So neither tier caches, and the 2× gap holds.
  - **The risk:** if the team grows the shared prefix into the 512–4,095-token band (for example, by moving policy text into the system prompt), the router could lose money while appearing to save it.
  - **How the gate catches it:** it bills each tier from its returned `usage` (cache reads and writes included). So `savings_if_allowed` is measured, not assumed, and a category is blocked unless that measured saving exceeds `--min-saving`.
  - **Tokenizers too:** Sonnet 5.5's newer tokenizer (about 30% more tokens than older Claude models) is measured the same way rather than assumed.

**A design input, not a failure.** The 2× gap doesn't make routing wrong; it changes the trade. Each downshift buys at most 50% on that request, while the quality risk is the same as on a 12× pair. The gate makes that trade explicit per category:

- quality must be non-inferior within the margin (5 points by default);
- the measured saving must exceed `--min-saving`.

On a 2× pair the team may reasonably tighten `--margin` (accept less quality risk for a smaller prize) or raise `--min-saving`. On the OpenAI pair the same margin buys 92% per request. Whatever the gate decides, it is written down with its evidence. Caching and the other stages remain the bigger, risk-free levers on this pair.

## 4. The gate (`python -m eval.gate_router`)

### Data

- **Default:** the hand-written eval set `eval/data/evalset/*.jsonl`. Items with `needs_context` get context from `eval.kb.retrieve`.
- **If that set is missing:** a fixed-seed Bitext sample (40 per category, seed 0) plus the KB seed questions, which cover returns and product. The script says so in its output. Bitext references are generic chatbot replies, so treat that gate as weaker evidence.

### Generation and scoring

- **Generation:** each item is answered by both tiers with the serving prompt. The engine runs in mode `off` with all stages skipped, at temperature 0. Calls go through the cassette `eval/cassettes/gate.jsonl`, so re-runs are free and resumable. Calls are sequential, with one progress line per item.
- **Scoring:** the shared judge (`eval.judge`) grades both answers 1–5 against the reference, scaled to 0..1, and gives a position-swapped pairwise verdict. On mock, a labelled heuristic judge is used. If `eval.judge` were missing, a proxy (embedding similarity to the reference, labelled `proxy:*`) would be used.

### Maths

For category c, keep the **routable** items, R_c: those with no hardness signal, which are exactly what the gated router would send cheap. Hard items stay strong in production, so including them would measure a policy we don't run. `all_items_diff` is still reported for comparison.

- Per item: d_i = 100 · (grade_cheap,i − grade_strong,i), in quality points from −100 to +100.
- d̄ = mean(d_i). Its 95% CI is a paired percentile bootstrap (`eval.stats.paired_bootstrap`, 2,000 resamples) that resamples whole `pair_id` clusters, so the two halves of a trap pair are not counted as independent evidence (Miller, [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)).
- **ALLOW** iff n_c ≥ `min_n` (30) **and** CI_lo ≥ −margin (−5 points) **and** measured saving > `min_saving` (0). This is a non-inferiority test, decided before any data is seen. Otherwise the category stays on strong.
- Also reported:
  - pairwise wins, ties and losses, with the non-inferior rate (cheap wins or ties) and its Wilson CI;
  - `n_needed` = ⌈(1.96 · sd / (d̄ + margin))²⌉, the number of items at which the observed spread would clear the margin;
  - measured `savings_if_allowed` and traffic share;
  - expected overall routing savings if deployed.

**Why grade difference rather than pairwise for the decision:** the pairwise per-item score is ±100 or 0, so its standard deviation is about 75 points. Proving a 5-point margin with it would take about 900 items per category. The rubric grade moves in 25-point steps and is mostly 0 between two decent answers, so its standard deviation is roughly 15–25 points.

### Sample size: why n < 30 warns, and small categories stay strong

The CI half-width is about 1.96 · sd / √n points:

| sd \ n | 30 | 40 | 60 | 100 | 200 |
|---|---|---|---|---|---|
| 15 points | ±5.4 | ±4.6 | ±3.8 | ±2.9 | ±2.1 |
| 20 points | ±7.2 | ±6.2 | ±5.1 | ±3.9 | ±2.8 |
| 25 points | ±9.0 | ±7.8 | ±6.3 | ±4.9 | ±3.5 |

Even if cheap and strong were truly equal (d̄ = 0), the lower bound only clears −5 once n ≥ 35 at sd 15, n ≥ 62 at sd 20, or n ≥ 97 at sd 25.

- **Under 30 items:** the interval is too wide to allow anything. The gate prints a `WARN` and the category stays on strong, which is the intended behaviour, not a bug.
- **To unlock a category:** add items to it. `n_needed` in the results says roughly how many.
- **Today:** the 44-item seed set has only 2–8 routable items per category, so a real run on it will allow nothing until the team's hand-written rows land.

### Judge caveats

- **Same-family judging.** On Anthropic, the default judge is Sonnet grading Sonnet against Haiku. Self-preference bias probably favours the strong answer, which makes the gate conservative, but it is a bias. Mitigations: the position swap, reference-guided grading, the hand-label agreement check (`python -m eval.judge agreement`), or `COSTGUARD_JUDGE_BACKEND`/`COSTGUARD_JUDGE_MODEL` for a cross-family judge (MT-Bench, [arXiv 2306.05685](https://arxiv.org/html/2306.05685v4)).
- **Evaluated in isolation.** The gate measures the router alone, with full context. The cumulative A/B (`eval/run_ab`) measures it combined with trimming and compression.

### Commands

```bash
python -m costguard.router.classifier train        # rebuild centroids + eval/results/router_classifier.json
python -m eval.gate_router --dry-run               # mock: whole pipeline, writes a dry_run gate, allows nothing
COSTGUARD_BACKEND=anthropic python -m eval.gate_router            # plan + estimated $, then stops (exit 2)
COSTGUARD_BACKEND=anthropic python -m eval.gate_router --yes      # run it (sequential, resumable via cassettes)
COSTGUARD_BACKEND=anthropic python -m eval.gate_router --replay   # re-score from cassettes only: $0, no key needed
python -m eval.gate_router --limit 8 --yes         # smoke run (round-robin across categories, marked partial)
```

Spend guard: on any backend other than mock or MLX, the script first prints the number of new generations (it checks the cassette for each call key), the number of judge calls, and an upper-bound dollar estimate. It calls nothing without `--yes`. On the seed set, the Anthropic estimate is at most $0.64; on the Bitext+KB fallback (304 items) it is at most $3.30.

### Outputs

| Run | Gate file | Results |
|---|---|---|
| Full | `configs/router_gate.json` | `eval/results/router_gate.json`, plus `router_gate_items.json` with every question, both answers and both grades, ready for hand-labelling |
| Dry | `configs/router_gate.json` only if no full gate exists there | `router_gate_dryrun.json` |
| Partial (`--limit`) | `configs/router_gate.json` only if no full gate exists there | `router_gate_partial.json` |

Dry and partial runs never overwrite a full gate unless `--force` is passed. The gate file is written atomically.

## 5. Rollout and rollback for a new routing policy

The gate is offline evidence. Production still earns trust in stages, and every stage is set by one field in `configs/router_gate.json`. The router re-reads that file whenever its mtime changes, so no restart or deploy is needed.

| Stage | Gate entry for the category | What happens | Move on when |
|---|---|---|---|
| 0. Offline gate | written by `eval.gate_router` | `allow` is decided on the eval set | CI_lo ≥ −margin and n ≥ 30 |
| 1. Shadow | `"allow": true, "rollout": "shadow"` | Users get strong; logs show `gated:<cat>-shadow` (would have downshifted). A daily sample is answered by cheap offline and judged against the served strong answer. | About a week, with a sampled-judge CI inside the margin and route mix as expected |
| 2. Canary | `"rollout": "canary:0.05"` | A stable 5% of queries (hashed, so the same question always lands in the same arm, which keeps caches coherent) get cheap (`-canary`); the rest are `-holdout` | The monitors below hold for the canary arm versus holdout; then try 25% |
| 3. Full | `"rollout"` removed (or `"full"`) | `gated:<cat>-allowed` | Keep monitoring |

**Rollback** is a file edit, live within one request:

- block one category: set `"allow": false`;
- send everything to strong: delete or rename the file (`gated:no-gate-file`);
- return to the last good gate: `git checkout <rev> -- configs/router_gate.json`.

A broader kill switch is the `quality` mode, which turns the router off, or setting `router_policy` for `balanced`.

**Rollback triggers** (thresholds fixed in advance; the monitoring job should flip `allow` automatically, which isn't wired up yet):

- the sampled-judge CI lower bound drops below −margin over the last 200 cheap answers in a category;
- escalation rate exceeds twice its shadow-week baseline;
- cheap-tier error or format-failure rate exceeds 2%.

On any trigger, set that category's `allow` to false.

**Re-run the gate** whenever the model pair, prices, system prompt or KB version changes. The gate file records `backend`, `models`, `billing` and `policy_hash`, and the router logs a warning if the gate was computed for a different model pair than the one being served.

## 6. Monitoring

| Signal | Source | Why |
|---|---|---|
| Route mix: share by `route_reason` and by category; cheap share; `hard:*` share; `gated:unclassified` share | `TraceRecord.route_reason`, request log, `/metrics` | Shows drift in traffic or classifier. A jump in `unclassified` means new topics; a jump in `hard:escalation` means an incident. |
| Cheap-tier quality | `eval.judge` on a daily sample of cheap-served requests, regenerated on strong offline (shadow pair); rolling paired CI per category | The production version of the gate statistic. Breaching −margin triggers rollback. |
| Escalation rate | Requests ending on strong after a cheap attempt (`fallback-after-cheap-error`), plus re-asks of a near-duplicate question by the same tenant within 10 minutes | The user-visible proxy for "the cheap answer didn't help". |
| Realised savings versus the gate's `savings_if_allowed` | `cost_usd` versus `baseline_cost_usd` per category | Catches caching or tokenizer effects (section 3) that make a "cheap" route expensive. |
| Router latency | `TraceRecord.stage_ms["router"]`, `GatedRouter.last_ms` | Should stay at about 2 ms p50; embedder trouble shows here first. |

## 7. Trade-offs: "We chose X over Y because Z"

1. **Rules plus a nearest-centroid classifier over a learned router (RouteLLM, Not Diamond)**, because independent benchmarks find simple routers competitive (RouterArena, LLMRouterBench), we have no preference data for our domain, and every decision must be explainable in one log string. Cost: we may miss some downshift opportunities a trained router would find.
2. **A per-category gate on the CI lower bound over a single global quality average**, because an average hides one bad category behind good ones, and a lower bound is automatically conservative when data is thin. Cost: small categories never get downshifted until someone writes more eval items.
3. **Pre-generation routing over a cheap-first cascade**, because on the Anthropic pair (cheap = 50% of strong) a cascade breaks even at 50% escalation and saves only 30% at 20% escalation before verifier cost. Every request also pays for the cheap call and its latency. Cost: no second chance when a cheap answer is bad; the gate and the monitors carry that risk instead.
4. **Leaving uncertain requests on strong over trusting the classifier's top-1**, via the margin rule. On the team's eval set, misroutes drop from 11.4% to 2.3% at 73% coverage. A misroute risks quality; non-coverage only costs savings.
5. **One centroid per source intent over one per category**, because heterogeneous categories ("other" covers contact, complaints, reviews and store info) average into a blurry centroid: Bitext held-out accuracy is 99.4% versus 96.7%.
6. **Deciding on the routable subset over all items**, because hard items never reach the cheap tier in production. The gate should measure the policy we actually run, and missed hard items stay inside the measurement. `all_items_diff` is still reported.
7. **A hot-reloaded JSON gate file over thresholds in `policy.yaml`**, because measured evidence (who allowed what, on which data and judge) should be separate from hand-set config, and rollback should not need a deploy. Cost: `config_hash` doesn't cover the gate, so `route_reason` and the gate's `created` stamp carry the traceability (see requested core changes).

## 8. Known limitations and requested core changes

- **`RouteInput` has no count of context documents**, so `many-context-docs` can't fire from the pipeline yet. `features.from_route_input` already reads `context_docs` if it appears. Request: add `context_docs: int` to `RouteInput`, set from `len(docs)`.
- **The inferred category and hardness reasons are not logged.** They live in `GatedRouter.last`, but `TraceRecord.category` is only the client's category. Request: a `route_detail: dict` (or `category_inferred`) field on `TraceRecord`, filled from `router.last`, so the dashboard can show route mix per category for unlabelled traffic.
- **Flipping the gate doesn't change `config_hash`.** Request: include the gate file's hash (or `created`) in the logged config hash, or add a `gate_version` field.
- **Product queries naming specific products often score just under the similarity floor**, so they stay strong unless the client sends `category`. Recommended: KB and product-page clients should always send it.
