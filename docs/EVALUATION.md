# Evaluation: how CostGuard's savings are measured

**The question:** given an LLM request, what is the cheapest way to produce an acceptable-quality answer, and
can we prove it?

The evaluation answers this with one frozen workload. Every optimisation lever is switched on one at a time over that
workload, and each step's saving is compared, on the same requests, with a baseline that has every lever off. Quality
is judged on those same requests. Every number in `docs/RESULTS.md` is generated from `eval/results/*.json` by
`python -m eval.report`, and none of it is typed by hand.

| Piece | File | Command |
|---|---|---|
| Frozen replay trace | `eval/data/trace_v1.jsonl` + `.sha256` | `python -m eval.build_trace` |
| Hand-written eval set | `eval/data/evalset/*.jsonl` | (written by the team; today only `seed.jsonl`, AI-written scaffolding, see §9) |
| CI subset | `eval/data/ci_subset.jsonl` | `python -m eval.build_trace --ci-subset` |
| Cumulative ablation (A/B) | `eval/results/ab_summary.json` | `python -m eval.run_ab` |
| CI eval gate | `eval/results/ci_gate.json`, `ci_baseline.json` | `python -m eval.ci_gate` |
| Judge | `eval/judge.py`, `eval/cassettes/judge.jsonl` | `python -m eval.judge agreement` |
| Statistics | `eval/stats.py` | — |
| Report | `docs/RESULTS.md` | `python -m eval.report` |

## 1. The frozen trace

**What's in it.** `trace_v1.jsonl` has 600 requests (seed 7). It is the *workload*, not the graded eval set.

| Slice | Share | Source | Reference answer |
|---|---|---|---|
| Customer-support queries | ~62% | [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (CDLA-Sharing 1.0); `{{slots}}` filled with fictional ShopNest values (order numbers `SN-xxxxx`, ₹ amounts, Indian cities) | the dataset's own response, slot-filled |
| KB questions with retrieved context | ~28% | `eval.kb.kb_questions()`; context = `eval.kb.retrieve(query, k=8)`, stored in the row | the KB question's reference |
| Trap pairs | 10% | the eval set's `trap_pair` rows, topped up with Bitext near-miss pairs | as above |

Exact repeats and near-duplicates (30% of requests in the headline trace) are spread over the first two slices; see duplicates below. The
default size is 600 so that a full A/B fits the budget (and local MLX generation stays feasible).

**Clusters: what counts as the "same question".** A cache may serve one request's answer for another only if both
are in the same cluster.

- **Bitext:** one cluster per intent + slot values. "cancel order SN-48213" and "cancel order SN-48231" are
  different clusters. Phrasings without slots form one generic cluster per intent. Generic phrasings that contain
  a literal number or an account type are dropped, because they hide an unlabelled entity.
  - Two Bitext intent pairs are merged because ShopNest gives the same answer to both: `check_invoice`/`get_invoice`
    and `contact_customer_service`/`contact_human_agent`.
- **KB:** one cluster per KB question.
- **Eval-set rows:** one cluster per row, unless the row sets `same_as` to the KB question or eval row it duplicates.

These labels are **strict**. A generic "how do I edit my account" served for "edit my Seller account" counts as a
false hit, so at low thresholds the false-hit rate is slightly pessimistic. `false_hit_rate_intent` in the summary is
the lenient same-intent view.

The labels were audited by replaying the trace through the real semantic cache at τ = 0.8 and reading every
cross-cluster hit. Three sources of label noise were found and fixed: near-synonym intents, literal entities in
generic phrasings, and an eval trap row identical to a KB question. The hits that remain are genuine near-misses,
such as a refund request vs a refund-status question, Nepal vs Canada, or one customer's invoice vs another's.

**Duplicates.** Exactly `round(dup_rate × N)` requests have an earlier semantically equivalent request. The trace
records this in three fields:
- `dup_of`: the position of that earlier request;
- `dup_kind`: one of the kinds below;
- `cluster_id`: the shared cluster.

| `dup_kind` | Share of duplicates | What it is | Lever it exercises |
|---|---|---|---|
| `exact` | 1/6 | verbatim repeat (same context) | exact cache |
| `surface` | 1/6 | case, punctuation or spacing change that normalises to the same exact-cache key | exact cache |
| `paraphrase` | ≈2/3 | a different phrasing of the same Bitext intent and slots, taken from the dataset itself (no LLM) | semantic cache |
| `wrapped` | (KB paraphrase slots) | the KB has no paraphrases, so its question is wrapped in a greeting or sign-off ("Hi, …", "… Thanks!") | semantic cache |
| `cross_source` | rare | an eval trap row that is `same_as` a KB question seen earlier | either |

Which cluster a duplicate repeats follows a **Zipf (s = 1.1)** popularity law:
- **head:** FAQ-like clusters (generic Bitext intents, KB questions);
- **tail:** customer-specific clusters (one order number).

KB duplicates fill the KB slice's shortfall, because only 64 KB questions exist. They are capped at 60% of all
duplicates.

**Trap pairs.** Each trap pair's two members sit 1–5 positions apart, so the first is in the cache when the second
arrives. Bitext near-miss pairs come in two kinds:
- **Cross-intent:** two intents that share a slot, given the same value, e.g. cancel vs track the same order, or
  create vs delete the same account type.
- **Entity:** the same template with a different order number (transposed digits), amount, account type, name or
  country.

Near-synonym intent pairs are excluded because they don't need different answers.

**Reproducibility.**
- The generator is deterministic for a seed, and `trace_v1.sha256` is committed beside the trace. The A/B summary
  records the trace hash it used.
- The Bitext CSV is cached at `eval/data/raw/bitext.parquet`.
- The dup-rate variants (`trace_v1_dup00/15/50.jsonl`) draw their unique items from one seeded stream, so they are
  nested: the 0.30 trace's unique items are a subset of the 0.15 trace's. They share most cassette entries.

## 2. Cassettes: every number is replayable

Every upstream call goes through `CassetteProvider`, keyed on `sha256(model, messages, max_tokens, temperature)`.

| Cassette | Written by | Purpose |
|---|---|---|
| `eval/cassettes/<backend>_ab.jsonl` | `run_ab` (auto mode) | identical calls across arms are made once; re-runs and re-analysis cost $0 |
| `eval/cassettes/judge.jsonl` | the judge (all backends) | re-judging is free and replayable |
| `eval/cassettes/<backend>_ci.jsonl` | `ci_gate --record` | CI replays with no API key |

Replayed completions keep their recorded usage and generation latency, so cost and latency figures are identical on
replay.

**Provider prompt caching is off for eval runs.** When `run_ab`, `gate_router` or `ci_gate` record or replay on
`anthropic`, they set `COSTGUARD_ANTHROPIC_PROMPT_CACHE=0` before the provider is built, unless
`--provider-prompt-cache` is passed. Why:
- **A cassette stores one usage per call key.** With caching on, the first call of a prefix records a cache *write*
  (1.25×), and every identical call would replay that write instead of the cache *read* (0.1×) it would really get.
- **Cache warmth depends on call order.** Which call warms the prefix depends on parallel prefetch, on which arm runs
  first and on how long ago the prefix was last used (the cache expires). A0's cost, and so every paired saving,
  would depend on recording conditions.
- **So prompt-cache savings are reported separately**, as a provider-side and order-dependent effect, and are not
  mixed into the paired A/B.
- **Today it changes nothing anyway.** The system prompt is about 110 tokens, below Anthropic's minimum cacheable
  prefix (512 tokens on Sonnet 5.5 and 4,096 on Haiku 4.5; see `docs/components/router_and_gate.md` §3).

The setting is recorded as `provider_prompt_cache` in the A/B summary `meta`, in the CI gate `meta` and in the router
gate files. It is `null` on mlx and mock, which never report cached tokens. If a run with caching off still replays
cache-read or cache-write tokens, the cassette was recorded with caching on: `run_ab` prints a warning and records
`meta.prompt_cache_warning`, and the cassette must be re-recorded before quoting headline numbers.

## 3. The A/B: cumulative ablation

The arms are ordered by quality risk; each arm adds one lever to the previous arm.

| Arm | Enabled | Answers |
|---|---|---|
| A0 | everything off, strong tier | the "before" column (real usage on every item) |
| A1 | + exact cache | the safe-first lever |
| A2 | + semantic cache at the policy τ, with guards | hit rate vs false-hit rate |
| A3 | + context rerank / dynamic-k | context-window management |
| A4 | + compression (retrieved context only) | compression with a quality delta |
| A5 | + router (gated) | the full **balanced** mode |
| B, Q | economy / quality mode as configured (optional) | the other operating points |

**How each arm runs.**
- One engine is built with `costguard.factory.build_engine`. Each arm gets a deep copy of the policy with
  `modes["balanced"]` overridden; `configs/policy.yaml` is never edited. `COSTGUARD_TAU_OVERRIDE` is honoured.
- Caches are flushed before each arm, and the trace is replayed **sequentially in trace order**. The costguard
  options `arm`, `trace_pos`, `item_id`, `category` and `context` come from the trace row.
- Every `TraceRecord` goes to `eval/results/ab_<backend>.sqlite`.

**Pre-flight.**
- Before any non-mock run, `run_ab` dry-runs the whole ablation. The real pipeline runs with a stand-in upstream
  that replays the cassette and records every other call. Cache, context, compression and routing decisions depend on
  queries and context, not answers, so this gives the exact set of new generations and their estimated cost, plus an
  upper bound on judge calls.
- Paid backends need `--yes` to spend.
- With `--workers N`, the known new calls are generated in parallel first, so the sequential replay is all cassette
  hits.
- Estimate for the headline trace (anthropic, empty cassettes; console output of `--estimate-only`, not saved under
  `eval/results`): **684 new Sonnet generations ≈ $2.01, ≤ 2,189 judge
  calls ≤ $2.48, total ≤ $4.49**. Router downshifts, once the router gate is computed for anthropic, add a few cents
  of Haiku calls.

## 4. Definitions

These are used exactly as written. The contract definitions are in `docs/CONTRACT.md`.

**Savings (primary, paired).** Savings = 1 − Σ arm cost ÷ Σ **A0 actual cost**, over the same items.
- A0 has real provider usage on every item, so this is a paired comparison. The 95% CI comes from a **cluster**
  bootstrap over `cluster_id`.
- `saved_per_request_usd` is the mean paired difference with its own CI.
- `savings_on_misses_pct` excludes requests that were served from the cache: the "savings without the cache" view.

**Estimated savings (secondary).** Uses each record's `baseline_cost_usd`: the strong tier, the full prompt, the
pre-call token estimate and this arm's output tokens.

**Hit rate.** Cache hits ÷ all **attempted** requests (failed requests included), split into exact and semantic.

**False-hit rate.** Wrong cache hits ÷ **all attempted requests**, not ÷ hits; per-hit precision flatters loose
thresholds.
- A hit is wrong when the served entry was created by a request in a different trace cluster. It is attributed
  through the entry id, falling back to the neighbour text. The Wilson 95% CI is reported.
- `trap_false_hits` counts the false hits that involve a trap row.

**Quality score.** The judge's 1–5 rubric grade against the item reference, scaled to 0–1.

**Quality retained.** Arm mean score ÷ A0 mean score on the same items, with a paired cluster-bootstrap 95% CI. The
paired score difference is reported too.

**Win/tie/loss.** `judge.pairwise(A0 answer, arm answer)` on up to `--pairwise-max` differing items per arm (default
150), sampled deterministically.
- Identical answers are not judged and count as ties.
- A failed judgment (see §5) is an `error`: it is counted in `pairwise.errors` and left out of win/tie/loss, never
  counted as a tie.
- The non-inferior rate (win + tie, over valid judgments) has a Wilson CI.

**Completeness.** Every arm summary carries `complete` and `coverage` (`attempted`, `failed`, `ungraded`,
`pairwise_errors`).
- An arm is incomplete if any request failed or, when grading was requested, any answer has no grade or any pairwise
  judgment failed. Missing grades silently drop possibly bad answers from quality retained and its CI, so they are
  never ignored.
- `eval.report` prints **INCOMPLETE: n failed / n ungraded** next to that arm, in RESULTS.md and in the README block.
- An incomplete arm is never the headline, and nothing is when A0 is incomplete, because every paired number is
  measured against A0.

**Correct answer and cost per correct answer.**
- An answer is correct when its grade is ≥ 4 out of 5 (≥ 0.75).
- Cost per correct answer = arm cost on graded items ÷ number of correct answers.

**Latency.**
- End to end = CostGuard overhead + the recorded upstream generation time, so cassette replays don't look
  instantaneous. Wall-clock time is reported too.
- Split into cache hits vs misses, with p50 and p99.
- Overhead = `overhead_ms` from the record.

**Waterfall.** Each arm's increment over the previous arm, in dollars, % of A0 cost, and input and output tokens
saved. Every step uses the **same items**: those that succeeded in every arm (`n_items`, `n_excluded` on each step;
the report says so), so the increments add up. Tokens overstate compression's dollar value because output tokens cost
5× input tokens on the Anthropic pair (6× for gpt-5.4-mini and 8× for gpt-5-nano, the prices mlx and mock are billed
at), so both are reported.

## 5. The judge

`eval/judge.py` implements the contract's `Judge.pairwise` / `Judge.grade` / `get_judge`.

**Model.**
- **Default:** the engine backend's strong tier, at temperature 0 with ≤ 5 output tokens. On `anthropic` that is
  `claude-sonnet-5-5`; on `mlx` it is `mlx-community/Qwen2.5-7B-Instruct-4bit`.
- **Override:** `COSTGUARD_JUDGE_BACKEND` / `COSTGUARD_JUDGE_MODEL`.
- **Cassette:** every call is cassette-backed (`COSTGUARD_JUDGE_CASSETTE`, `COSTGUARD_JUDGE_CASSETTE_MODE`).
- **Without a key or MLX:** the judge runs replay-only.

**`pairwise`.** One-token verdict: A, B or TIE.
- The prompt asks which answer better and more correctly answers the customer's question, given the reference.
  Correctness against the reference comes first; length, tone and position are to be ignored.
- It runs in **both orders**, and the swapped verdict is mapped back.
- If the two orders disagree, the result is a tie (the MT-Bench protocol).
- If either order fails (cassette miss, API error) or stays unparseable after one retry, the result is `"error"`:
  a missing judgment, never a tie. The router gate treats it as missing evidence
  (`docs/components/router_and_gate.md` §4), and the A/B marks the arm incomplete.

**`grade`.**
- A 1–5 rubric:
  - 5: correct and complete;
  - 4: minor omission;
  - 3: partly correct or vague;
  - 2: mostly wrong, or a key fact wrong;
  - 1: wrong, unsafe, or doesn't answer.
- The grade is scaled to 0–1.
- A parse failure gets one retry with a stricter suffix, which is a different cassette key; a second failure returns
  `None`.

**Mock backend.**
- Mock answers are meaningless, so a deterministic **heuristic judge** is used: token-overlap F1 with the reference.
  It is labelled `"judge": "heuristic-mock"` everywhere.
- It carries a small first-position bonus, so the swap logic is exercised: near-ties come out position-inconsistent,
  then tie.

**Known biases and mitigations.**
- **Position bias.** In the MT-Bench study, GPT-4 was position-consistent in only 65% of cases. The swap converts
  inconsistent verdicts to ties, and `stats.position_inconsistent` reports how often that happened.
  - In an MLX smoke run (output not saved under `eval/results`), Qwen-7B answered "A" in both orders: pure position
    bias, correctly scored as a tie.
- **Verbosity bias.** The prompts say to ignore length.
- **Self-preference risk.** A **Sonnet judge grading Sonnet vs Haiku** answers (arm A5) can favour its own family. On
  mlx the default judge is the strong model itself (Qwen2.5-7B grading Qwen2.5-7B vs Qwen2.5-1.5B), with the same risk.
  - Mitigations: the position swap, reference-guided grading (an absolute grade against a written reference, not
    taste), and the hand-label agreement check below.
  - If keys allow, set `COSTGUARD_JUDGE_BACKEND`/`COSTGUARD_JUDGE_MODEL` to a different family and report both.

**Hand-label agreement.**
1. Export blinded A0-vs-arm pairs:
   `python -m eval.judge export --db eval/results/ab_anthropic.sqlite --arm A5 --n 50 --out eval/data/human_labels_todo.jsonl`.
   The pair order is randomised, and the arm names are hidden in `_a_arm` / `_b_arm`.
2. Two teammates fill in `"human": "A" | "B" | "tie"`.
3. Save the result as `eval/data/human_labels.jsonl`.
4. `python -m eval.judge agreement --rejudge` (or `run_ab --agreement`) reports agreement including ties, agreement
   excluding ties, Cohen's κ, and, for `human_score` rows, grade MAE and pass/fail κ.
   - For reference, in MT-Bench GPT-4 agreed with humans 66% / 85% (including / excluding ties), and humans agreed
     with each other 63% / 81%.

## 6. Statistics (`eval/stats.py`, numpy only)

**Functions.**
- `paired_bootstrap(diffs, n=2000, seed=0, clusters=None)`: the mean difference and a percentile 95% CI.
  - With `clusters`, whole clusters are resampled. Duplicates of one question are correlated by construction, so an
    i.i.d. bootstrap would overstate confidence. The tests check that the clustered interval is wider.
- `ratio_bootstrap(num, den, …)`: the same resampling for Σnum / Σden. Used for savings and quality retained.
- `proportion_ci(k, n)`: Wilson; exact at 0 and n.
- `mcnemar(b, c)`: exact binomial when b + c < 25, else χ² with continuity correction.
- `cohen_kappa`.

**Sample-size caveat.** For a proportion near 0.5, the 95% half-width is ±13.9 points at n = 50, ±9.8 at n = 100
and ±5.7 at n = 300. The 600-row trace supports the savings and false-hit claims. The hand-written eval set (10–15
rows per author, about 60–90 in total; today only the 44 AI-written seed rows) supports only coarse gates.

## 7. Sensitivity to the duplicate rate

The headline uses a 30% duplicate rate.
- **Source:** MeanCache's 31%, which is a **per-user** figure from 20 ChatGPT users, not a service-wide one.
- **Contrast:** SCALM found 4.5% (MOSS) and 7.5% (LMSYS) of real chat queries answerable from a similar earlier query
  (text-embedding-3-small, τ = 0.90). Customer support is FAQ-heavy, so the true value is
  probably in between.

So the A/B is repeated on nested variants:

| Variant | Dup rate | Composition (Bitext / KB / trap) | Unique clusters |
|---|---|---|---|
| `trace_v1_dup00.jsonl` | 0% (+1 cross-source) | 79% / 11% / 10% | 599 |
| `trace_v1_dup15.jsonl` | 15% | 70% / 20% / 10% | 509 |
| `trace_v1.jsonl` | 30% | 62% / 28% / 10% | 419 |
| `trace_v1_dup50.jsonl` | 50% | 62% / 28% / 10% | 299 |

**The 0% variant.** With no duplicates, every saving must come from context trimming, compression and routing.
This is the honesty check against "the duplicate rate is rigged". The KB slice is smaller there because the KB has
only 64 questions.

**Running it.**
`python -m eval.run_ab --trace eval/data/trace_v1_dup00.jsonl --arms A0,A5 --backend anthropic --yes`, and the same
for dup15 and dup50. The outputs are `ab_summary_dupXX.json`, and `eval/report` renders the sensitivity table.

## 8. The CI eval gate

`python -m eval.ci_gate` runs the subset in `ci_subset.jsonl`: 30 eval-set rows, 20 trap pairs (adjacent; 6 from the
eval set, the rest Bitext near-misses) and 8 legitimate repeats.

**What it replays.** Two passes: A0, and balanced mode exactly as configured.
- Upstream: `eval/cassettes/<backend>_ci.jsonl` in **replay** mode, with no key. The backend is chosen by
  `COSTGUARD_CI_BACKEND` and defaults to anthropic.
- If that cassette is missing, the gate uses the deterministic mock backend.
- Quality uses the deterministic heuristic judge by default; `--judge model` replays the judge cassette instead.

**Checks.** Exit 1 on any failure.

| Check | Rule |
|---|---|
| replay complete | no cassette misses (a miss means the cassette is stale: re-record), including a cheap-tier miss that the pipeline papered over by falling back to strong (`CassetteMiss` in any `stage_errors` entry) |
| (a) trap false hits | must be 0 |
| (a′) false hits, all rows | ≤ baseline |
| (b) mean quality | ≥ baseline − 0.03 |
| (c) savings vs A0 | ≥ baseline − 2 points |
| baseline matches | backend, judge and subset hash equal (otherwise `--update-baseline` in the same PR) |

**Outputs.** A plain table on stdout, a markdown table in `$GITHUB_STEP_SUMMARY`, and `eval/results/ci_gate.json`.
False hits are listed as "request ← served the cached answer of".

**Maintenance.**
- Changing the subset or the eval set: rebuild the subset, then run `python -m eval.ci_gate --update-baseline`.
- Real-model CI: `python -m eval.ci_gate --record --backend anthropic --yes`, roughly $0.2–0.5. It records A0, the
  balanced pass and the no-cache miss path of every row, so a PR that *raises* τ still replays.

**The demo PR.** `COSTGUARD_TAU_OVERRIDE=0.6 python -m eval.ci_gate`, or a PR that lowers `tau` in `policy.yaml`:
- savings rise from 60.5% to 72.7%;
- 11 trap false hits and 22 false hits overall appear;
- the gate fails (exit 1).

These figures come from the mock backend with every real stage component, on 2026-10-04. Only the 60.5% baseline is
saved (`ci_gate.json` → `metrics.savings_pct`). The τ = 0.6 run's output was not saved, so re-run it and keep its
output before quoting 72.7% / 11 / 22.

At the calibrated τ it passes with zero false hits. The mock backend is enough for this, because the cache decision
depends only on the queries.

## 9. What's honest about the numbers

- **Prices.** Dollars are list-price costs computed from token usage, using `configs/prices.yaml` (`checked_on` is
  recorded in every summary).
  - On `anthropic`, they are the real Sonnet 5.5 / Haiku 4.5 prices. Prompt-cache reads (0.1×) and writes (1.25×)
    are billed when present, but eval runs turn provider prompt caching off unless `--provider-prompt-cache` is
    passed (§2).
  - **Local models (mlx) and mock are billed at the list price of the API model each tier stands in for**
    (`policy.billing`: strong = gpt-5.4-mini, cheap = gpt-5-nano). Their dollar figures are what the same token
    counts *would* cost, not money spent. Local token counts come from the local model's tokenizer, which differs
    from the API model's.
- **Mock numbers are placeholders.** Savings on mock reflect real cache, trimming and compression decisions on
  real queries and contexts, but answers are synthetic and the quality columns mean nothing. The report banners
  them.
- **The trace is semi-synthetic.**
  - Bitext was generated by an NLG pipeline and curated by linguists, and its references are generic templates.
    The judge therefore grades Bitext items against a "good generic answer". That bias is constant across arms and
    cancels in quality *retained*.
  - The KB "wrapped" paraphrases are template wrappers, not real user paraphrases.
- **The duplicate rate is a parameter, not a measurement**, hence the sensitivity table.
- **`eval/data/evalset/seed.jsonl` is AI-written scaffolding** (`"author": "seed"`). The graded, hand-written set is
  the team's `<name>.jsonl` files. Results computed on the seed rows must say so.
- **The judge is an LLM from the same family as the served models** (see §5). Its agreement with hand labels must be
  reported next to any quality claim.
- **Determinism.** Temperature 0 plus cassettes makes every reported number exactly reproducible from the committed
  files. Re-generating a cassette from scratch would not reproduce the same text bit for bit.

## 10. How to run everything

```bash
PY=./.venv/bin/python
$PY -m eval.build_trace --all-variants                 # trace_v1 (+ dup00/15/50) and sha256 files
$PY -m eval.build_trace --ci-subset                    # eval/data/ci_subset.jsonl
$PY -m eval.run_ab --backend mock --limit 50           # smoke test: seconds, $0
$PY -m eval.run_ab --backend anthropic --estimate-only # pre-flight: new generations + estimated $
$PY -m eval.run_ab --backend anthropic --yes --workers 4            # headline A/B (A0..A5)
$PY -m eval.run_ab --backend anthropic --yes --arms A0,A5 --trace eval/data/trace_v1_dup00.jsonl   # sensitivity
$PY -m eval.judge export --db eval/results/ab_anthropic.sqlite --arm A5 --n 50 --out eval/data/human_labels_todo.jsonl
$PY -m eval.judge agreement --rejudge                  # after hand-labelling -> eval/data/human_labels.jsonl
$PY -m eval.ci_gate                                    # CI gate (exit 1 on regression)
$PY -m eval.ci_gate --update-baseline                  # after changing the subset / eval set
$PY -m eval.report                                     # docs/RESULTS.md
```

**Output naming.**
- Summaries are named `ab_summary[_<trace tag>][_<local backend>][_limitN].json`. Paid-backend full runs write the
  canonical `ab_summary.json`, mock writes `ab_summary_mock.json`, and `--limit` runs never overwrite a full run.
- `--out` overrides the path.

**Trace row format** (contract fields first):
- contract fields: `pos, item_id, cluster_id, query, category, context, reference, source (bitext|kb|trap), dup_of`;
- extra fields: `dup_kind`, `intent`, `pair_id`, `trap_kind`, `trap_origin`, `author`, `kb_type`, `key_facts`.
