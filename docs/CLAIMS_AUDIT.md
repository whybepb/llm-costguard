# Claims audit

**Date:** 2026-10-04. **Scope:** `docs/ARCHITECTURE.md`, `DESIGN_DECISIONS.md`, `RUNBOOK.md`, `PRESENTATION_OUTLINE.md`, `EVALUATION.md`, `RESULTS.md`, `CONTRACT.md`, `docs/components/*.md`, `eval/data/kb/README.md`, `eval/data/evalset/README.md`, the comments in `configs/policy.yaml` and `configs/prices.yaml`, and a spot-check of the research report's key numbers.

**Rule applied:** every factual claim is backed by either our own results (file → key) or a cited external source (URL, paper or course file). Line numbers refer to the files *after* the fixes below.

## Summary

**241 claims checked** (137 our measurements, 85 external facts, 15 course facts, 4 design opinions). About 38 external sources were opened live on 2026-10-04.

| Status | Count | Meaning |
|---|---:|---|
| verified-results | 89 | matches a results JSON key, a committed data file or a recomputation |
| verified-live | 70 | the cited page was fetched and shows the number |
| verified-course | 13 | matches the course notes or PDF text |
| matches-research-only | 3 | cited in the research report, source not fetched |
| ok-opinion | 4 | a design choice, phrased as one |
| **fixed** | **55** | doc edited (59 listed fixes): 21 were wrong (contradicted, stale or arithmetic), 17 were unsupported and are now re-sourced or labelled "not saved", 17 needed an honesty caveat or their source context |
| contradicted (left) | 5 | all in files this audit may not edit: `policy.yaml:2`, `policy.yaml:65`, generated `RESULTS.md`, and two lines of the research record |
| unsupported (left) | 2 | `RESULTS.md` (generated) has no upper-bound caveat; one rough cost estimate in `EVALUATION.md:307` |

**What this means.**

- **Prices are clean.** All nine rows in `configs/prices.yaml` match the live provider pages, checked again on 2026-10-04. So do the Anthropic cache multipliers (reads 0.1×, writes 1.25× / 2×) and minimums (Sonnet 5.5 512 tokens, Haiku 4.5 4,096 tokens).
- **Measured numbers are clean.** Every load-test, threshold-sweep, compression, classifier and gate number quoted in the docs matches its JSON key. The exceptions were a handful of slips, all now fixed: 96.7% for 96.6%, "2–8" for 2–7 routable items, "130 MB" for 66 MB, "under 1 ms" for 1–4 ms, and "10–20 req/s" for 8–20. About a dozen quoted measurements exist in no results file; they are now labelled "not saved" (open item 4).
- **The biggest problems were framing, not arithmetic:**
  1. Several docs said the "real experiments run on Anthropic". The measured runs actually use the local MLX stand-in (Qwen2.5-7B / 1.5B), billed at GPT-5.4-mini / GPT-5-nano prices. That pair has a ~12× price gap, not 2×, so the MLX waterfall will overstate what routing saves on the Anthropic pair.
  2. Several places called the eval set or the seed traps "hand-written". All 44 eval rows and all 34 trap pairs are AI-written seed scaffolding (`author: "seed"`).
  3. Some text still described the old `policy.yaml` values (τ 0.90, budget 1200, "today's policy").
- **Types:** a = our measurement, b = external fact, c = course fact, d = design opinion.
- **Statuses:**
  - **verified-live:** the cited page was fetched on 2026-10-04 and shows the number.
  - **verified-results:** the number matches a results JSON key, a committed data file or a recomputation from repo code.
  - **verified-course:** the number matches the course notes or PDF text, as summarised in `docs/research/notes/course_rubric_mapping.md`. That file is course-derived, so it stays local and is gitignored; public readers will not see it.
  - **matches-research-only:** the number is in the research report with a source, but the source was not fetched.
  - **fixed:** the doc was edited. The *Action* column says what was wrong.
  - **contradicted / unsupported:** still wrong, or still has no backing. All but one sit in files this audit may not edit, and each is listed under the open items. The exception is `EVALUATION.md:307`, which is already worded as a rough estimate.

---

## Claims table

### `docs/ARCHITECTURE.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| A-01 | Objective: ≥ 30% cost cut at ≥ 95% quality retained | ARCHITECTURE.md:3 | d | Stated as the objective and target, not as a result | ok-opinion | none |
| A-02 | Fly.io and HF Docker Spaces are no longer free; Render free still is | ARCHITECTURE.md:104 | b | https://docs.fly.io/about/pricing ("New organizations don't have a free tier"); https://huggingface.co/docs/hub/spaces-overview (Gradio/Docker "require a paid plan to create"); research § "suggested free hosts" | verified-live | none |
| A-03 | Render free host has 512 MB RAM | ARCHITECTURE.md:94 | b | https://render.com/docs/compute-plans (Free: "0.1 CPU", "512 MB") | verified-live | Rewrote cell to cite it. The research report still calls this "unverified" (open item 7) |
| A-04 | "LLMLingua-2 needs ~2 GB of torch" | ARCHITECTURE.md:94 | a | `compression_eval.json` → `llmlingua2.peak_rss_mb_after_load` 1584.2, `llmlingua2.large.peak_rss_mb_after_load` 4098.2 | fixed | Was imprecise. Now gives the measured 1.6 GB / 4.1 GB peak RSS |
| A-05 | Load test: 50 users, 60 s, 300 ms mock, Apple silicon, five stages | ARCHITECTURE.md:133 | a | `loadtest.json` → `setup` (users 50, duration 60s, mock_latency_ms 300, machine Darwin arm64), `components` | verified-results | none |
| A-06 | Token count p50 0.13 / p99 2.1 ms | ARCHITECTURE.md:137 | a | `loadtest.json` → `server_log.stage_ms.count` (0.13 / 2.1) | verified-results | none |
| A-07 | Exact cache 0.00 / 0.01 ms, n = 6,923 | ARCHITECTURE.md:138 | a | `server_log.stage_ms.exact_cache` | verified-results | none |
| A-08 | Semantic stage p50 0.17 / p99 17.0 ms, n = 2,981 | ARCHITECTURE.md:139 | a | `server_log.stage_ms.semantic_cache` (0.17 / 16.95, n 2981) | verified-results | none |
| A-09 | Rerank p50 25.1 / p99 104.4 ms, n = 753 | ARCHITECTURE.md:140; DESIGN_DECISIONS.md:49; PRESENTATION_OUTLINE.md:106 | a | `server_log.stage_ms.context` (25.07 / 104.41, n 753) | verified-results | none |
| A-10 | Compression p50 1.0 / p99 7.5 ms, n = 52 | ARCHITECTURE.md:141 | a | `server_log.stage_ms.compression` (1.03 / 7.45, n 52) | verified-results | none |
| A-11 | Router p50 0.14 / p99 2.1 ms | ARCHITECTURE.md:142 | a | `server_log.stage_ms.router` (0.14 / 2.11) | verified-results | none |
| A-12 | Upstream 305 / 313 ms; "real Sonnet/Haiku TTFT is seconds" | ARCHITECTURE.md:143 | a | `server_log.stage_ms.upstream` (305.17 / 313.17); the TTFT part had no source | fixed | Reworded to "a real completion takes seconds (not measured here)" |
| A-13 | Write-back p50 0.14 / p99 0.8 ms | ARCHITECTURE.md:144 | a | `server_log.stage_ms.semantic_write` (0.14 / 0.77) + `exact_write` (0.01 / 0.02) | verified-results | none |
| A-14 | Hooks cost 19 µs + 3 µs per request | ARCHITECTURE.md:145; DESIGN_DECISIONS.md:90 | a | No results file | fixed | Labelled "micro-benchmark, not saved under `eval/results`" (open item 4) |
| A-15 | SQLite log write ≈ 0.5 ms | ARCHITECTURE.md:62, 146; DESIGN_DECISIONS.md:96 | a | Only `loadtest/README.md:73`; no results key | fixed | Labelled "not saved" in the table (open item 4) |
| A-16 | Hit-path overhead p50 0.3 / p99 3.6 ms, n = 6,116 | ARCHITECTURE.md:147, 173; PRESENTATION_OUTLINE.md:104, 169, 235 | a | `loadtest.json` → `hit_path.overhead_ms` | verified-results | none |
| A-17 | Miss-path overhead p50 11.8 / p99 94.6 ms, n = 1,552 | ARCHITECTURE.md:148, 174; PRESENTATION_OUTLINE.md:105 | a | `miss_path.overhead_ms` (11.8 / 94.59) | verified-results | none |
| A-18 | RAG (bypass) requests p50 25.9 / p99 105.8 ms, n = 745 | ARCHITECTURE.md:149; PRESENTATION_OUTLINE.md:106 | a | `server_log.overhead_ms_by_cache_status.bypass` (25.88 / 105.83) | verified-results | none |
| A-19 | Client-observed added latency p50 25.1 / p99 124.1 ms | ARCHITECTURE.md:150 | a | `miss_path.client_added_ms` (25.08 / 124.08) | verified-results | none |
| A-20 | 20,000 req/day ≈ 0.23 req/s; ~1.2 req/s at 5× peak | ARCHITECTURE.md:188 | d | Design point; arithmetic checked (20,000 / 86,400 = 0.231) | ok-opinion | none |
| A-21 | Baseline ≈ $0.0039/request, $78/day, $2.3k/month at Sonnet 5.5 $2/$10 | ARCHITECTURE.md:193; DESIGN_DECISIONS.md:120 | a | 1,200 × $2/M + 150 × $10/M = $0.0039; prices verified-live (PR-08) | verified-results | none |
| A-22 | Self-hosting break-even ≈ $10k/month (W3S1) | ARCHITECTURE.md:194; DESIGN_DECISIONS.md:120; PRESENTATION_OUTLINE.md:155 | c | course_rubric_mapping.md:246, 266, 422 (W3S1 notes §5.3) | verified-course | none |
| A-23 | 128 req/s, 0 failures, about 100× the assumed peak | ARCHITECTURE.md:195; PRESENTATION_OUTLINE.md:169, 235 | a | `loadtest.json` → `throughput_rps` 127.59, `failures` 0; 127.6 / 1.16 ≈ 110× | verified-results | none |
| A-24 | "Anthropic Sonnet 5.5 / Haiku 4.5 for real runs" | ARCHITECTURE.md:205 | a | Contradicted by the runs actually happening: `eval/results/logs/gate_mlx.log` (judge `mlx-community/Qwen2.5-7B-Instruct-4bit`), CONTRACT.md:19 (no API keys), `policy.yaml:13, 18-20` | fixed | Now says Anthropic is the real API backend and MLX is the stand-in used for the measured runs, billed at GPT-5.4-mini / GPT-5-nano |
| A-25 | Experiment budget $5–20 | ARCHITECTURE.md:181 | c | deliverables.pdf Notes, via course_rubric_mapping.md:35 | verified-course | none |
| A-26 | Langfuse free tier = 50k units/month | ARCHITECTURE.md:101; RUNBOOK.md:180; PRESENTATION_OUTLINE.md:141 | b | https://langfuse.com/pricing ("50k units / month included", "30 days data access") | verified-live | none |
| A-27 | Prices checked 2026-10-03 | ARCHITECTURE.md:97 | b | `prices.yaml` `checked_on`; re-checked live 2026-10-04, unchanged (PR-01…PR-09) | verified-live | none |
| A-28 | Exact cache lookup ≈ 1 µs | ARCHITECTURE.md:90; DESIGN_DECISIONS.md:21; PRESENTATION_OUTLINE.md:88 | a | `server_log.stage_ms.exact_cache` p50 0.00 / p99 0.01 ms (10 µs resolution) | verified-results | none |
| A-29 | Anthropic adapter: 60 s timeout, 4 retries | ARCHITECTURE.md:57; DESIGN_DECISIONS.md:135 (failure table) | a | `costguard/providers/anthropic_provider.py:36` (`timeout_s=60.0, max_retries=4`) | verified-results | none |

### `docs/DESIGN_DECISIONS.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| D-01 | False-hit budgets 0.5% / 1% / 3% | DESIGN_DECISIONS.md:24; PRESENTATION_OUTLINE.md:182; RUNBOOK.md:94 | a | `threshold_sweep.json` → `budgets_false_hit_rate` | verified-results | none |
| D-02 | AWS ~92% "accuracy" at τ = 0.80 is 1 wrong answer in 14 | DESIGN_DECISIONS.md:26 | b | https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-benchmarks.html (0.80: 87.6% hit ratio, 91.8% accuracy → 7.2% of requests); research §1 | fixed | Number verified live. Added the source context: Titan embeddings, SemBenchmarkLmArena, 87.6% hits |
| D-03 | Lookalike pairs score ≈ 0.9 cosine | DESIGN_DECISIONS.md:31; PRESENTATION_OUTLINE.md:87; ARCHITECTURE.md:92 | a | `threshold_sweep.json` → `models["BAAI/bge-small-en-v1.5"].pairwise.trap_detail`: #4821/#4822 0.970, negation 0.924, median of 34 traps 0.874 | verified-results | none |
| D-04 | bge-small adds ≈ 70 MB to the image | DESIGN_DECISIONS.md:37 | a | `models/fastembed/.../bge-small-en-v1.5-onnx-Q/model_optimized.onnx` = 66,465,124 bytes | verified-results | none |
| D-05 | Anthropic cache reads bill at 10% of input | DESIGN_DECISIONS.md:43; PRESENTATION_OUTLINE.md:94; context_and_compression.md:114 | b | https://platform.claude.com/docs/en/about-claude/pricing ("Cache read (hit) 0.1x base input price"; Sonnet 5.5 $0.20 vs $2) | verified-live | none |
| D-06 | "Rerank + dynamic-k is the published winner on retrieved context" | DESIGN_DECISIONS.md:48 | b | https://arxiv.org/abs/2407.08892 says extractive selection "often outperforms" token pruning, up to 10× | fixed | Was an overclaim. Reworded to what the paper says |
| D-07 | LLMLingua-2 needs "about 2 GB of weights"; heuristic "under 1 ms" | DESIGN_DECISIONS.md:53 | a | `compression_eval.json` → `llmlingua2.peak_rss_mb_after_load` 1584 (mBERT, ~0.7 GB model), `large` 4098; `loadtest.json` → `server_log.stage_ms.compression` p50 1.03 / p99 7.45; `methods["heuristic-lexical@0.5"].latency_ms_p50` 4.03 | fixed | Was contradicted. Now gives the measured RSS and latency |
| D-08 | GPT-5.4-mini → GPT-5-nano ≈ 12× | DESIGN_DECISIONS.md:63 | a | `router_gate_dryrun.json` → `price_gap.cost_ratio_cheap_over_strong` 0.0803 (12.4×); prices verified-live | verified-results | none |
| D-09 | "Haiku 4.5 is strong enough to pass the gate on simple categories" | DESIGN_DECISIONS.md:65 | a | No Anthropic gate run exists (the only gate result is a mock dry run; the MLX gate is running now) | fixed | Was unsupported. Now stated as an expectation |
| D-10 | Anthropic gap is 2×: $2/$10 vs $1/$5; output costs 5× input | DESIGN_DECISIONS.md:66 | b | Anthropic pricing page (Sonnet 5.5 $2/$10, Haiku 4.5 $1/$5); `router_gate_dryrun.json` → `price_gap_all_backends.anthropic.cost_ratio` 0.5 | verified-live | none |
| D-11 | Measured runs are MLX, billed at a ~12× pair | DESIGN_DECISIONS.md:68 | a | `policy.yaml:13, 18-20`; `logs/gate_mlx.log` | fixed | New bullet. Headline routing savings must carry this caveat |
| D-12 | The SDK returns `cache_read_input_tokens` and `cache_creation_input_tokens` separately | DESIGN_DECISIONS.md:72 | b | https://platform.claude.com/docs/en/build-with-claude/prompt-caching (usage fields) | verified-live | none |
| D-13 | "LiteLLM's normalised usage blurs them" | DESIGN_DECISIONS.md:72 | b | https://docs.litellm.ai/docs/completion/prompt_caching returns `prompt_tokens_details.cached_tokens` plus `cache_creation_input_tokens` | fixed | Was contradicted. Now says LiteLLM *re-maps* the fields, which is one more translation layer |
| D-14 | Langfuse 50k units ≈ 8k requests | DESIGN_DECISIONS.md:95, 170 | b | Langfuse pricing (50k, verified live); ~6 units/request is the research's assumption (research §"stack", line 379) | fixed | Added the "~6 units per request" assumption at :95 |
| D-15 | PSI bands < 0.10 / 0.10–0.25 / > 0.25 (W3S2) | DESIGN_DECISIONS.md:105; RUNBOOK.md:174; PRESENTATION_OUTLINE.md:139 | c | Course W3S2 lecture notes, PSI section ("below 0.10 is stable, 0.10–0.25 means investigate, above 0.25 means retrain"); not in the research report | verified-course | none |
| D-16 | Render: 15-minute idle sleep, ~1 min wake, no persistent disk | DESIGN_DECISIONS.md:111; RUNBOOK.md:54 | b | https://render.com/docs/free | verified-live | none |
| D-17 | 10× scale = 200k req/day, ~12 req/s peak | DESIGN_DECISIONS.md:149 | d | Arithmetic from A-20 | ok-opinion | none |
| D-18 | 40 threads cap a 300 ms upstream at ~133 req/s | DESIGN_DECISIONS.md:158 | a | `loadtest.json` → `setup.server` ("40 threads"); 40 / 0.3 s | verified-results | none |
| D-19 | At 2–5 s upstream the cap is "about 10–20 req/s" | DESIGN_DECISIONS.md:158 | a | 40 / 5 s = 8 | fixed | Arithmetic error. Now "8–20 req/s" |
| D-20 | Course scaling levers: autoscaling, caching, batching, rate limiting, fallbacks (W7S2) | DESIGN_DECISIONS.md:173 | c | course_rubric_mapping.md:303, 465 | verified-course | none |

### `docs/RUNBOOK.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| R-01 | Render free may be suspended for unusual traffic | RUNBOOK.md:55 | b | https://render.com/docs/free ("may suspend a Free web service that initiates an uncommonly high volume of traffic") | verified-live | none |
| R-02 | Canary 0.5% → 1% → 5% → 10% → 50% | RUNBOOK.md:95 | c | course_rubric_mapping.md:326 (W2S2/W3S2 ramp starting at 0.5%) | verified-course | none (see open item 12 on the router doc's different ramp) |
| R-03 | PyYAML reads a bare `off` as `False` | RUNBOOK.md:131 | a | `yaml.safe_load('default_mode: off')` → `{'default_mode': False}` (run in repo venv) | verified-results | none |
| R-04 | Course alert rules: +10–20%, > 0.5% errors, > 5× median cost | RUNBOOK.md:139 | c | course_rubric_mapping.md:286-287, 354, 474 (W7S2) | verified-course | none |
| R-05 | The rubric needs 3 of the 5 monitoring categories | RUNBOOK.md:166 | c | rubric.pdf p.1 via course_rubric_mapping.md:42 | verified-course | none |
| R-06 | Required checks need a public repo, or GitHub Pro/Team for a private one | RUNBOOK.md:208 | b | docs.github.com "About protected branches" (public repos on GitHub Free; public and private on Pro, Team, Enterprise; seen in the docs' search snippet, the fetched page omitted the plan box) | verified-live | none |
| R-07 | Demo PR changes balanced `tau: 0.90 -> 0.60` | RUNBOOK.md:232 | a | `configs/policy.yaml:60` is `tau: 0.93` | fixed | Was stale. Now `0.93 -> 0.60` |
| R-08 | `/v1/drift` reports `warming_up` until 500 requests | RUNBOOK.md:64, 183 | a | `costguard/obs/drift.py:183-184` (`baseline_size=500`) | verified-results | none |

### `docs/EVALUATION.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| V-01 | Eval set "written by the team" | EVALUATION.md:14 | a | `eval/data/evalset/` holds only `seed.jsonl` (44 rows, all `author: "seed"`) | fixed | Added "today only `seed.jsonl`, AI-written scaffolding" |
| V-02 | `trace_v1.jsonl` has 600 requests, seed 7 | EVALUATION.md:24 | a | 600 rows; `eval/build_trace.py:487` (`seed: int = 7`); `ab_summary_mock.json` → `meta.trace_stats.n` | verified-results | none |
| V-03 | Slices ~62% / ~28% / 10% | EVALUATION.md:28-30, 269 | a | `ab_summary_mock.json` → `meta.trace_stats.source_share` (0.618 / 0.282 / 0.1) | verified-results | none |
| V-04 | "The brief's '10% exact repeats and near-duplicates'" | EVALUATION.md:32 | c | Not in the handout, allocation sheet, deliverables PDF or research (pdftotext grep) | fixed | Was unsupported. Now describes the trace's own 30% duplicate rate |
| V-05 | Duplicate kinds: exact 1/6, surface 1/6, paraphrase ≈ 2/3 | EVALUATION.md:63-66 | a | `meta.trace_stats.dup_kinds`: exact 30, surface 30, paraphrase 54 + wrapped 66, of 181 duplicates | verified-results | none |
| V-06 | Zipf s = 1.1 | EVALUATION.md:69 | a | `eval/build_trace.py:56` (`ZIPF_S = 1.1`) | verified-results | none |
| V-07 | Only 64 KB questions | EVALUATION.md:73 | a | `eval.kb.kb_questions()` → 64; `compression_eval.json` → `setup.questions` 64 | verified-results | none |
| V-08 | Anthropic estimate: 684 generations ≈ $2.01, ≤ 2,189 judge calls ≤ $2.48, total ≤ $4.49 | EVALUATION.md:135-136 | a | Console output of `--estimate-only`; no file. The sum checks (2.01 + 2.48 = 4.49) | fixed | Labelled "not saved under `eval/results`" (open item 4) |
| V-09 | "Output tokens cost 5× input tokens" | EVALUATION.md:181 | b | True for Sonnet/Haiku only. gpt-5.4-mini is 6× and gpt-5-nano 8× (OpenAI pricing, verified live), and mlx/mock bill at those | fixed | Added the 6× / 8× case |
| V-10 | Default judge = strong tier (`claude-sonnet-5-5` on anthropic) | EVALUATION.md:189 | a | `policy.yaml:11`; `logs/gate_mlx.log` ("judge: mlx-community/Qwen2.5-7B-Instruct-4bit") | fixed | Added the mlx judge model |
| V-11 | MT-Bench: GPT-4 position-consistent in 65% | EVALUATION.md:218 | b | https://arxiv.org/html/2306.05685v4 Table 2 ("65.0%") | verified-live | none |
| V-12 | MLX smoke run: Qwen-7B answered "A" in both orders | EVALUATION.md:220 | a | No results file | fixed | Labelled "output not saved" (open item 4) |
| V-13 | Self-preference risk (Sonnet judging Sonnet vs Haiku) | EVALUATION.md:223-224 | d | n/a | fixed | Added that on mlx, Qwen2.5-7B judges itself against 1.5B |
| V-14 | MT-Bench agreement: GPT-4–human 66% / 85%, human–human 63% / 81% | EVALUATION.md:237; research:245 | b | arXiv 2306.05685v4 Table 5, first turn: G4-Pair vs Human 66% (S1) / 85% (S2); Human 63% / 81% | verified-live | none |
| V-15 | 95% half-widths ±13.9 / ±9.8 / ±5.7 at n = 50 / 100 / 300 | EVALUATION.md:251; research:250 | a | 1.96 × √(0.25 / n) = 0.139 / 0.098 / 0.057 | verified-results | none |
| V-16 | Eval set "~40 rows per author group" | EVALUATION.md:252-253 | a | evalset README: 10–15 rows per author, 6 authors; today 44 seed rows | fixed | Was contradicted. Corrected |
| V-17 | MeanCache 31% is a per-user figure from 20 ChatGPT users | EVALUATION.md:258; PRESENTATION_OUTLINE.md:242 | b | https://arxiv.org/html/2403.02694v3 ("We recruited 20 participants … 31% of a user's queries are similar") | verified-live | none |
| V-18 | SCALM measured 4.5–7.5% on real chat logs | EVALUATION.md:259; semantic_cache.md:188 | b | https://arxiv.org/html/2406.00025 ("4.5% … MOSS and 7.5% … LMSYS"; text-embedding-3-small, 0.90) | fixed | Verified live. Added the dataset and model context |
| V-19 | Dup-rate variants: composition and unique clusters (599 / 509 / 419 / 299) | EVALUATION.md:267-270 | a | Recomputed from `trace_v1*.jsonl`: dup00 79.3 / 10.7 / 10, 599; dup15 70.3 / 19.7 / 10, 509; dup30 61.8 / 28.2 / 10, 419; dup50 61.8 / 28.2 / 10, 299 | verified-results | none |
| V-20 | CI subset = 30 eval rows + 20 trap pairs + 8 repeats (78 rows) | EVALUATION.md:282; RESULTS.md:73 | a | `ci_subset.jsonl` (40 trap + 38 eval rows); `ci_gate.json` → `metrics.n` 78, `hits.exact` 8 | verified-results | none |
| V-21 | τ = 0.6 demo: savings 60.5% → 72.7%; 11 trap false hits, 22 overall | EVALUATION.md:311-317 | a | 60.5% = `ci_gate.json` → `metrics.savings_pct` 60.49. The τ = 0.6 figures are in no file | fixed | Labelled which part is saved (open item 4) |
| V-22 | Real-model CI re-record costs roughly $0.2–0.5 | EVALUATION.md:307 | a | No file; worded as a rough estimate | unsupported | Left as is; it is clearly an estimate |
| V-23 | Anthropic billing: cache reads 0.1×, writes 1.25× | EVALUATION.md:326 | b | Anthropic pricing page | verified-live | none |
| V-24 | mlx and mock are billed as gpt-5.4-mini / gpt-5-nano | EVALUATION.md:329 | a | `policy.yaml:18-20`; `ab_summary_mock.json` → `meta.billing` | verified-results | none |
| V-25 | Bitext was generated by NLG and curated by linguists | EVALUATION.md:336 | b | https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset ("all steps … curated by computational linguists") | verified-live | none |
| V-26 | Trace hash committed and recorded in the summary | EVALUATION.md:86-87; RESULTS.md:65 | a | `trace_v1.sha256` = `shasum` of the file = `19c0834c…7e37` (1,841,156 bytes) = `ab_summary_mock.json` → `meta.trace_sha256` | verified-results | none |

### `docs/PRESENTATION_OUTLINE.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| P-01 | Team 8 slot: Sunday 11 Oct, 4:00–4:20 PM | PRESENTATION_OUTLINE.md:3 | c | schedule.pdf via course_rubric_mapping.md:95, 111 | verified-course | none |
| P-02 | Reviewers are Team 10 (Guardrails) and Team 6 (RAG + CI) | PRESENTATION_OUTLINE.md:7-8 | c | peer_review.pdf via course_rubric_mapping.md:98 | verified-course | none |
| P-03 | Rubric weights 20 / 30 / 20 / 20 / 10 | PRESENTATION_OUTLINE.md:16-20 | c | deliverables.pdf text ("architecture design (30%) … presentation quality (10%)"); course_rubric_mapping.md:40-44 | verified-course | none |
| P-04 | The 10% criterion is "Results and delivery" | PRESENTATION_OUTLINE.md:20 | c | The rubric criterion is "Presentation Quality" (course_rubric_mapping.md:44) | fixed | Relabelled "Presentation quality (delivery; results slide)" |
| P-05 | 15 minutes plus Q&A; six speakers × 2:30 | PRESENTATION_OUTLINE.md:3, 22 | c | deliverables.pdf §5 (15 min + 5 min Q&A) via course_rubric_mapping.md:30 | verified-course | none |
| P-06 | Headline from `ab_summary.json` keys `saved_pct` / `quality_retained` | PRESENTATION_OUTLINE.md:50 | a | `eval/run_ab.py:532` writes `ab_summary_mlx.json` for mlx. The keys are `savings_pct` / `savings_ci` and `quality.retained` / `retained_ci` (`ab_summary_mock.json`) | fixed | Was contradicted. File and keys corrected, and "list-price equivalent" wording added |
| P-07 | Threshold-curve slide shows hit rates with no caveat | PRESENTATION_OUTLINE.md:85 | a | `threshold_sweep.json` hit rates come from Bitext template paraphrases | fixed | Added the upper-bound caveat |
| P-08 | Sonnet → Haiku is a 2× gap, so routing is a small lever | PRESENTATION_OUTLINE.md:100, 152, 170, 233-234 | b | True for Anthropic (D-10). The measured MLX runs bill at ~12× (D-11) | fixed | Added MLX caveats at :100, :170, :234 |
| P-09 | "A hand-written domain eval set" | PRESENTATION_OUTLINE.md:117, 206-207 | a | Only 44 AI-written seed rows exist; `logs/gate_mlx.log` shows the gate falling back to Bitext + KB | fixed | Added honesty caveats |
| P-10 | Retrieval is deliberately generous (top-8) | PRESENTATION_OUTLINE.md:222 | a | `compression_eval.json` → `setup.k` 8 | verified-results | none |

### `docs/CONTRACT.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| K-01 | "Eval set: hand-written by the team" | CONTRACT.md:11 | a | evalset holds only AI-written `seed.jsonl` | fixed | Added the current state |
| K-02 | Savings = 1 − Σ cost ÷ Σ baseline_cost | CONTRACT.md:100-103 | a | EVALUATION §4: the headline is paired against A0 actual cost; `baseline_cost_usd` gives "est. savings" | fixed | Added a note reconciling the two definitions |
| K-03 | Trace `source` enum includes `qqp` | CONTRACT.md:89 | a | No trace row has `source: qqp` (bitext / kb / trap only) | ok-opinion | Schema allows it; harmless (open item 9) |

### `docs/components/semantic_cache.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| S-01 | `tests/test_cache.py` has 36 tests | semantic_cache.md:17 | a | `pytest --collect-only` → "36 tests collected" | verified-results | none |
| S-02 | 10–24% of requests served from identical text | semantic_cache.md:46, 190 | a | `threshold_sweep.json` → `recommended.*.{templated,filled}.exact_text_hit_rate` 0.102–0.238 | fixed | Verified. Added the Bitext upper-bound caveat at :46 |
| S-03 | Krites: static then dynamic tier; VentureBeat: 18% exact duplicates | semantic_cache.md:48 | b | https://arxiv.org/abs/2602.13165; https://venturebeat.com/orchestration/why-your-llm-bill-is-exploding-and-how-semantic-caching-can-cut-it-by-73 ("Only 18% were exact duplicates") | verified-live | none |
| S-04 | CacheAttack: 86% response-hijack rate | semantic_cache.md:56, 118 | b | https://arxiv.org/abs/2601.23088 ("hit rate of 86% in LLM response hijacking") | verified-live | none |
| S-05 | MeanCache 0.66 precision on contextual queries; vCache says single-turn | semantic_cache.md:67 | b | arXiv 2403.02694v3 Table 1 (GPTCache 0.66 vs MeanCache 0.98); arXiv html 2502.03771 ("effective in single-turn interactions with short to medium context") | verified-live | none |
| S-06 | "cancel #4821" vs "don't cancel #4821" at cosine 0.92 | semantic_cache.md:84 | a | `pairwise.trap_detail` sim 0.9236 | verified-results | none |
| S-07 | Guards cost p50 0.07 ms (cold) | semantic_cache.md:84, 261 | a | `latency_ms.guard_check_ms_cold.p50` 0.074 | verified-results | none |
| S-08 | Rules tuned on "the hand-written seed traps" | semantic_cache.md:93 | a | `eval/cache_pairs.py:170-171` labels them `author: "seed"`, the project's AI-written convention (evalset README:11) | fixed | Was contradicted. Now "AI-written scaffolding, not team-written" |
| S-09 | AWS TTL guidance: minutes for prices, 24 h for policies | semantic_cache.md:103 | b | https://docs.aws.amazon.com/AmazonElastiCache/latest/dg/semantic-caching-best-practices.html (24 h static facts/policies; 5–15 min prices) | verified-live | none |
| S-10 | Memory-store search p50 0.02 ms at 1.1k entries | semantic_cache.md:111, 244 | a | `simulation.templated.guards_on.search_ms_p50` 0.012–0.023; `entries` 1,133 | verified-results | none |
| S-11 | Query embeddings "structurally lose" validity information | semantic_cache.md:118 | b | https://arxiv.org/abs/2609.35908 abstract: "can lose information" | fixed | Was contradicted (overstated). Now "can lose", with the arXiv id |
| S-12 | Checking the cached query text is the 2609.35908 defence | semantic_cache.md:123 | b | Same abstract: "blocks 82.0% to 98.2% of poisoned entries at a 5% false-positive rate" | verified-live | none |
| S-13 | Bitext: 26,872 queries, 27 intents, CDLA-Sharing-1.0 | semantic_cache.md:136 | b | HF dataset card; `threshold_sweep.json` → `datasets.bitext_rows` 26872 | verified-live | none |
| S-14 | Compact Bitext parquet is 450 kB | semantic_cache.md:136 | a | `eval/data/cache_pairs/bitext_queries.parquet` = 450,162 bytes | verified-results | none |
| S-15 | Trap pairs and seed paraphrases listed as "project" with no authorship | semantic_cache.md:138-139 | a | `cache_pairs.py:170-174` (`author: "seed"`) | fixed | Marked them AI-written |
| S-16 | Bitext echoes `{{Order Number}}` 100%, `{{Account Type}}` 98%, tiers 70% | semantic_cache.md:143 | a | No results file | fixed | Labelled "not saved" (open item 4) |
| S-17 | Pair mix 1,200 / 300 / 600 / 600 / 300; seed 13 | semantic_cache.md:151-157 | a | `pairs_v1.manifest.json` → `by_kind`, `seed` | verified-results | none |
| S-18 | Stream: 5,000 queries, Zipf s = 1, 29.6% repeats, 28 labels | semantic_cache.md:163 | a | `threshold_sweep.json` → `datasets.stream` | verified-results | none |
| S-19 | Recommended τ 0.94 / 0.93 / 0.89 with hit, false-hit (CI) and precision | semantic_cache.md:180-182; RESULTS.md:92-97 | a | `threshold_sweep.json` → `recommended.*` | verified-results | none |
| S-20 | "Current `policy.yaml` starting points are 0.95, 0.90, 0.85" | semantic_cache.md:184 | a | `policy.yaml:51, 60, 72` = 0.95 / 0.93 / 0.89 | fixed | Was stale. Updated |
| S-21 | τ = 0.95: 69.5% / 0.24% templated; 51.0% / 0.14% filled | semantic_cache.md:186 | a | `models[bge].simulation.*.guards_on` at τ = 0.95 (0.695 / 0.0024; 0.5096 / 0.0014) | verified-results | none |
| S-22 | Benchmarks 50–90% vs real 4.5–7.5% and "support/FAQ workloads about 20–60%" | semantic_cache.md:188 | b | 50–90%: GPT Semantic Cache 61.6–68.8% (arXiv 2411.05276) and AWS 87.6%, both verified live. "Support/FAQ 20–60%" had no source; Portkey blog says "~20% … at 99% accuracy for Q&A (or RAG)" and "18% to as high as 60%" on RAG | fixed | Re-attributed to Portkey with its context |
| S-23 | Guard-effect table (τ 0.85 / 0.90 / 0.93, both renderings) | semantic_cache.md:196-203 | a | `models[bge].guard_effect.{templated,filled}` (all 36 cells match) | verified-results | none |
| S-24 | "SN-48213" vs "SN-90211" embed at about 0.97 | semantic_cache.md:205 | a | That pair is in no result. The measured pair SN-48213 / SN-48231 = 0.9757 (`trap_detail`) | fixed | Swapped in the measured pair |
| S-25 | Without guards τ = 0.97 / 0.96 / 0.95; filled hit 41.6 / 47.0 / 52.7%; +11 to +15 points | semantic_cache.md:205 | a | `recommended_without_guards.*`; 54.7 − 41.6 … 67.3 − 52.7 = +10.7 to +14.6 | verified-results | none |
| S-26 | Per-candidate guard verdicts at τ = 0.90 | semantic_cache.md:211-213 | a | `simulation.*.guards_on.guard_rejections[τ=0.90]` | verified-results | none |
| S-27 | Labelled pairs > 0.80: numbers 48/1, entities 95/2, negation 4/2 | semantic_cache.md:216 | a | `pairwise.bitext.per_guard_at_0.80` | verified-results | none |
| S-28 | All 34 traps caught, 16 above 0.90; 0 of 14 paraphrases blocked | semantic_cache.md:218 | a | `pairwise.trap_detail` (34 with a guard, 16 ≥ 0.90); `pairwise.seed.per_guard_at_0.80` all 0 | verified-results | none |
| S-29 | QQP > 0.80: guards reject 81 of 363 non-duplicates, 149 of 938 duplicates | semantic_cache.md:220 | a | `pairwise.qqp.curve` at τ = 0.80: fpr 0.363 → 0.282, recall 0.938 → 0.789 | verified-results | none |
| S-30 | Cumulative false-hit rate 0.60 / 0.85 / 0.94 / 0.84%; 1,133 entries | semantic_cache.md:232 | a | `simulation.templated.guards_on.cumulative` and `entries` at τ = 0.93 | verified-results | none |
| S-31 | MiniLM τ 0.88 / 0.85 / 0.76; false hits at 0.85: 4.9 / 3.5% vs 0.7 / 0.5% | semantic_cache.md:236 | a | `models[MiniLM].recommended.guards_on`; `simulation` at τ = 0.85 | verified-results | none |
| S-32 | ROC AUC: Bitext 0.69 vs 0.70, QQP 0.87 vs 0.87 | semantic_cache.md:238 | a | `pairwise.{bitext,qqp}.roc_auc` (0.6937 / 0.7006; 0.8732 / 0.8696) | verified-results | none |
| S-33 | `embed_one` p50 2–3.6 ms, p99 3–8 ms "across runs" | semantic_cache.md:242 | a | Saved run: `latency_ms.embed_one_ms` 3.603 / 7.814; other runs not saved | verified-results | none (in range) |
| S-34 | Full lookup on 2k entries p50 2.5 / p99 5.2 ms | semantic_cache.md:243 | a | `latency_ms.lookup_end_to_end_ms_2k_entries` (2.45 / 5.226) | verified-results | none |
| S-35 | Model load 60–250 ms; first download about 15 s | semantic_cache.md:245 | a | `latency_ms.model_load_ms` 57.7; 250 ms and 15 s are in no file | fixed | Cites the saved 58 ms and labels the rest "not saved" |
| S-36 | "About 130 MB of ONNX weights per process" | semantic_cache.md:247 | a | The ONNX file is 66 MB; `compression_eval.json` → `setup.peak_rss_mb.after_retrieval_index` 301.7 | fixed | Was contradicted. Now gives the measured values |
| S-37 | Production report: 20 ms p50 for embedding + vector search | semantic_cache.md:249 | b | VentureBeat article (embedding 12 ms + search 8 ms = 20 ms p50; 47 ms p99); upstream 1–6 s is consistent with AWS's example misses (1.64 s, 6.51 s) | verified-live | none |
| S-38 | A hosted embedder adds a network hop of about 50–200 ms | semantic_cache.md:253 | a | Not measured, no source | fixed | Labelled "our estimate, not measured" |
| S-39 | text-embedding-3-small costs $0.02 / 1M tokens | semantic_cache.md:253 | b | OpenAI pricing page | verified-live | none |
| S-40 | bge-small is MIT, MiniLM is Apache; bge-small is fastembed's default | semantic_cache.md:255 | b | HF cards (MIT, 33.4M, 384-d; Apache-2.0, 22.7M, 384-d); installed `fastembed.TextEmbedding` default `BAAI/bge-small-en-v1.5` | verified-live | none |
| S-41 | τ = 0.85 templated: per-hit precision 94.7%, but 4.9% of requests wrong | semantic_cache.md:259 | a | `simulation.templated.guards_on` at 0.85 (0.94677 / 0.049) | verified-results | none |

### `docs/components/context_and_compression.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| X-01 | 22 KB docs of 360–620 words | context_and_compression.md:25 | a | 22 `.md` files besides README; `wc -w` 360–621 | verified-results | none |
| X-02 | 61 passages of 99–315 tokens (median 219) | context_and_compression.md:32 | a | Recomputed: `eval.kb.load_kb()` + o200k → 61, 99, 315, 219. Not in a results file | verified-results | none (open item 4) |
| X-03 | Median 1,756 context tokens; key facts 100% in top-8 | context_and_compression.md:36 | a | `compression_eval.json` → `setup.context_tokens_median` 1756; `eligible` 56 / `answerable` 56 | verified-results | none |
| X-04 | 64 questions: 44 single-fact, 12 multi-hop, 8 unanswerable | context_and_compression.md:38 | a | `kb_questions()` kb_type counts | verified-results | none |
| X-05 | Cross-encoder has 22M parameters and takes ~65 ms for 8 docs | context_and_compression.md:45 | b/a | 22M: research:146 (not fetched); 65 ms: `methods["rerank gap=6"].latency_ms_p50` 65.05 | matches-research-only | none |
| X-06 | LLMLingua-2 model is 0.7–2.2 GB | context_and_compression.md:72 | b | HF card for the large model: 0.6B params F32 (verified live); research measurement_stack_plan.md:203 | verified-live | none |
| X-07 | Hybrid scorer gained 2 of 56 at rate 0.33 | context_and_compression.md:92, 188 | a | `methods["heuristic-hybrid@0.33"].retained` 56 vs `lexical` 54 | verified-results | none |
| X-08 | OpenAI cached input is 0.1×: gpt-5.4-mini $0.075 vs $0.75 | context_and_compression.md:114-116 | b | OpenAI pricing page | verified-live | none |
| X-09 | 3,000 cached tokens cost the same as 300; compressing 2× costs 5× more | context_and_compression.md:118-120 | b | Arithmetic; research:160 | verified-results | none |
| X-10 | Saving bound 0.76 × 0.94 × 0.81 ≈ 58% (~100 output tokens) | context_and_compression.md:134-137 | a | 1,870 × 0.75 / (1,870 × 0.75 + 100 × 4.5) = 0.757; 1,756 / 1,870 = 0.939; 1 − 1/5.36 = 0.813. The 100 output tokens is an assumption | fixed | Labelled "(assumed)" |
| X-11 | Full compression results table (21 methods) | context_and_compression.md:151-173; RESULTS.md:101-123 | a | `compression_eval.json` → `methods.*` (ratio, kept_share, retention, CI, lenient, p50 / p99) | verified-results | none |
| X-12 | rerank@1200 + heuristic@0.5 = 5.45× at 96.4% | context_and_compression.md:184 | a | `methods["rerank@1200+heuristic@0.5"]` 5.448 / 0.9643 | verified-results | none |
| X-13 | "Today's balanced policy measures 4.99×" | context_and_compression.md:187 | a | `grid` (1200, 0.5) = 4.987 was the old policy; the current (1000, 0.5) = 5.363 | fixed | Was stale. Both values given |
| X-14 | LLMLingua-2 keeps 51.8% (66% lenient), large 53.6%; ~207 ms MPS; 385 ms / 1.29 s CPU | context_and_compression.md:192-193 | a | `methods["llmlingua2@0.5"]`, `["llmlingua2-large@0.5"]`; `llmlingua2.cpu` (384.8 / 1290.3) | verified-results | none |
| X-15 | Peak RSS 1.6 GB (mBERT), 4.1 GB (large), 443 MB (bge-small + cross-encoder) | context_and_compression.md:198, 280-281 | a | `llmlingua2.peak_rss_mb_after_load` 1584.2; `.large` 4098.2; `setup.peak_rss_mb.after_cross_encoder` 443.0 | verified-results | none |
| X-16 | Unanswerable questions still keep ~1,260 tokens | context_and_compression.md:200 | a | `methods["rerank gap=6"].unanswerable_tokens_after_mean` 1264.2 | verified-results | none |
| X-17 | Recommended per-mode settings table | context_and_compression.md:213-216; RESULTS.md:129-131 | a | `recommendation.*` (29% / 19% / 16%; 98.2 / 96.4 / 92.9%; CIs) | verified-results | none |
| X-18 | "How this compares with `policy.yaml` today" (quality 2000, balanced 1200, economy 800 @ 0.33) | context_and_compression.md:218-226 | a | `policy.yaml:53, 55, 62, 65, 74` now 1600 / off, 1000 / 0.5, 800 / 0.5 | fixed | Was stale. Rewritten in the past tense with the current values |
| X-19 | Gap-sweep table | context_and_compression.md:233-241 | a | `compression_eval.json` → `gap_sweep` | verified-results | none |
| X-20 | Jha et al.: extractive selection often beats token pruning, up to 10× | context_and_compression.md:277 | b | https://arxiv.org/abs/2407.08892 | verified-live | none |
| X-21 | Free host has about 512 MB RAM | context_and_compression.md:280 | b | render.com/docs/compute-plans | verified-live | none |
| X-22 | Cohere Rerank costs $2.00–2.50 per 1,000 searches | context_and_compression.md:286 | b | Search results for OpenRouter's Rerank 4 Pro ($2.50 / 1k) and Fast ($2.00 / 1k). The research's cited URL https://openrouter.ai/cohere/rerank-4-pro returned 404 to the fetcher | verified-live | none (open item 7) |
| X-23 | Lost in the middle (Liu et al. 2023) | context_and_compression.md:58 | b | research:156 (arXiv 2307.03172), not fetched | matches-research-only | none |
| X-24 | n = 56 gives a CI of about ±7–8 points | context_and_compression.md:305 | a | Wilson CIs in `retention_ci95`, e.g. 96.4% → [87.9, 99.0]: asymmetric, roughly −8 / +3 | verified-results | none |

### `docs/components/router_and_gate.md`

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| G-01 | Hardness flags on seed rows: hard 6/7, answerable 3/25, trap 0/12 | router_and_gate.md:60-62 | a | `router_gate_dryrun.json` → `hard_signal_rate` | verified-results | none |
| G-02 | Hardness flags on Bitext 5.6%, KB multi-hop 2/12, single-fact 1/44 | router_and_gate.md:56, 63-65 | a | No results file | fixed | The source line now says these rows are not saved (open item 4) |
| G-03 | 43 centroids (63 KB), 10,400 Bitext rows, 555 seeds, floor 0.704 from 26 off-topic queries, 100% held-out rejection | router_and_gate.md:73-87 | a | `router_classifier.json` → `train.*`, `centroids_bytes` 63429, `min_similarity` 0.7039, `min_similarity_rule`, `off_topic_rejected.heldout` 1.0 | verified-results | none |
| G-04 | Classifier table: 99.4 / 98.2 / 0.1; 89.9 / 71.6 / 0.6; 84.1 / 72.7 / 2.3; 67.2 / 53.1 / 6.2 | router_and_gate.md:95-98 | a | `router_classifier.json` → `{heldout_bitext, heldout_seed_frames, evalset, kb_questions}.shipped.accuracy` and `.gated_view` | verified-results | none |
| G-05 | Row label "Team seed eval set" | router_and_gate.md:97 | a | The seed set is AI-written | fixed | Relabelled |
| G-06 | Earlier blind runs scored 70% and 58% | router_and_gate.md:102 | a | `router_classifier.json` → `notes[4]` ("evalset 0.70, kb_questions 0.58") | verified-results | none |
| G-07 | Router overhead 0.03 ms (explicit category) / 1.9 ms (embedding) | router_and_gate.md:105 | a | No results file; under load `server_log.stage_ms.router` was 0.14 / 2.11 | fixed | Labelled the unsaved part and added the load-test numbers |
| G-08 | RouterArena: Not Diamond #12, ~35% cheaper at < 2% loss, no router tops every metric; ICLR 2026 | router_and_gate.md:110 | b | https://arxiv.org/html/2510.00202v1 (all three quotes). The venue was not on the fetched page (research only) | verified-live | none |
| G-09 | LLMRouterBench: 33 models, 21 datasets, plus two quotes | router_and_gate.md:111 | b | https://arxiv.org/abs/2601.07206 | verified-live | none |
| G-10 | RouteLLM `mf` trained on Arena GPT-4/Mixtral data; authors warn traffic may differ | router_and_gate.md:112 | b | https://arxiv.org/html/2406.18665 ("real-world applications may have distributions that differ substantially") | verified-live | none |
| G-11 | Economics table for the anthropic / openai / gemini pairs | router_and_gate.md:121-123 | a | Prices verified live; `router_gate_dryrun.json` → `price_gap`, `price_gap_all_backends` (0.5 / 0.0803 / 0.2138) | verified-results | none |
| G-12 | "The real experiments run on Anthropic" | router_and_gate.md:125 | a | Measured runs are MLX (A-24) | fixed | Was contradicted. Now "production backend is Anthropic", with the MLX caveat |
| G-13 | r ≈ 0.85 → 42%; "half downshifted saves at most about 25% on that slice" | router_and_gate.md:127 | a | 0.85 × 0.5 = 0.425. Half downshifted gives 25% *overall* (50% on that half) | fixed | Wording error fixed |
| G-14 | Cascade breaks even at 50% escalation and saves 30% at 20% | router_and_gate.md:128 | a | 0.5 + e = 1; 0.5 + 0.2 = 0.7 | verified-results | none |
| G-15 | Sonnet 5.5 caches from 512 tokens, Haiku 4.5 only from 4,096 | router_and_gate.md:129 | b | https://platform.claude.com/docs/en/build-with-claude/prompt-caching (512: Sonnet 5.5; 4,096: Haiku 4.5; below the minimum, "no error is returned") | verified-live | none |
| G-16 | 3,000-token prefix: Sonnet $0.0056 vs Haiku $0.0055, a 1.8% saving | router_and_gate.md:129 | a | 3,000 × 0.2 + 500 × 2 + 400 × 10 = 5,600; 3,500 × 1 + 400 × 5 = 5,500 (µ$) | verified-results | none |
| G-17 | The system prompt is about 110 tokens | router_and_gate.md:130 | a | o200k count of `configs/prompts/system_v1.txt` = 110 | verified-results | none |
| G-18 | Sonnet 5.5 tokenizer gives about 30% more tokens | router_and_gate.md:133 | b | Anthropic pricing page: "Claude 4.7 and later models … approximately 30% more tokens". Haiku 4.5 uses the old one | verified-live | none |
| G-19 | Pairwise sd ≈ 75 means ~900 items | router_and_gate.md:167 | a | (1.96 × 75 / 5)² ≈ 864 | verified-results | none |
| G-20 | Half-width table; n ≥ 35 / 62 / 97 | router_and_gate.md:175-179 | a | 1.96 · sd / √n, all 15 cells recomputed | verified-results | none |
| G-21 | Seed set has "2–8 routable items per category" | router_and_gate.md:183 | a | `router_gate_dryrun.json` → `categories.*.n` = 2–7 (8 is `n_items` including hard items) | fixed | Was contradicted. Now 2–7 |
| G-22 | Anthropic gate estimates ≤ $0.64 (seed), ≤ $3.30 (304-item fallback) | router_and_gate.md:201 | a | No file (the only saved estimate is mock: `estimate.usd_total_upper_bound` 0.114) | fixed | Labelled "console estimates, not saved" |
| G-23 | Margin rule: misroutes 11.4% → 2.3% at 73% coverage, "on the team's eval set" | router_and_gate.md:257 | a | `router_classifier.json` → `evalset.shipped.misrouted_share` 0.1136, `gated_view` 0.0227 / 0.7273. The set is seed, not team-written | fixed | Numbers verified. Label corrected |
| G-24 | One centroid per intent vs per category: 99.4% vs 96.7% | router_and_gate.md:258 | a | `heldout_bitext.one_centroid_per_category.accuracy` 0.9658 = 96.6% | fixed | Rounding error. Now 96.6% |
| G-25 | Paired bootstrap resampling by cluster (Miller, arXiv 2411.00640) | router_and_gate.md:159 | b | research:184, 448; not fetched | matches-research-only | none |

### Data READMEs

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| KB-01 | KB chunks are "~150-250-token passages" | eval/data/kb/README.md:9 | a | `load_kb()`: 61 passages, 99–315 tokens, median 219 | fixed | Was contradicted. Corrected |
| EV-01 | `seed.jsonl` is AI-written scaffolding; don't present it as hand-written | eval/data/evalset/README.md:11-14 | a | All 44 rows `author: "seed"` | verified-results | none |
| EV-02 | The course requires a hand-written, domain-specific set with no copied benchmarks | eval/data/evalset/README.md:3-4 | c | deliverables.pdf §3 via course_rubric_mapping.md:28 | verified-course | none |

### `docs/RESULTS.md` (generated; not edited)

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| RS-01 | A/B is labelled MOCK and its quality numbers meaningless | RESULTS.md:7 | a | `ab_summary_mock.json` → `meta.backend` "mock", judge heuristic-mock | verified-results | none |
| RS-02 | Headline 58.6% [47.7, 67.2], hit 22.5% (mock) | RESULTS.md:9 | a | `arms[A5].savings_pct` 58.62, `savings_ci`, `hit_rate.total` 22.5 | verified-results | none (to be replaced by MLX) |
| RS-03 | Threshold-calibration table has no upper-bound caveat on hit rates | RESULTS.md:86-97 | a | Same numbers as S-19; caveat missing | unsupported | Open item 6 (generator) |
| RS-04 | Recommended settings show quality at rate 0.70; policy runs quality with compression off | RESULTS.md:129; policy.yaml:55 | a | `compression_eval.json` → `recommendation.quality` vs `policy.yaml` | contradicted | Open item 6 |
| RS-05 | Load test, CI gate and router dry-run tables | RESULTS.md:73-84, 135-157 | a | `loadtest.json`, `ci_gate.json`, `router_gate_dryrun.json` | verified-results | none |

### `configs/policy.yaml` and `configs/prices.yaml` comments (code; not edited)

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| Y-01 | Model mapping anthropic `claude-sonnet-5-5` / `claude-haiku-4-5-20251001`; mlx Qwen2.5-7B / 1.5B 4-bit | policy.yaml:11, 13 | a | Matches the brief and EVALUATION.md:189 | verified-results | none |
| Y-02 | Local and mock tiers billed as gpt-5.4-mini / gpt-5-nano | policy.yaml:17-20 | a | `ab_summary_mock.json` → `meta.billing` | verified-results | none |
| Y-03 | "Thresholds below are starting points; the sweep scripts produce the calibrated values" | policy.yaml:2 | a | The values are now the calibrated ones (S-19, X-17) | contradicted | Open item 5 |
| Y-04 | τ comments (0.94 meets 0.5%, 0.95 keeps the upper bound inside; 0.34–0.84%; 1.56–2.54%) | policy.yaml:51, 60, 72 | a | `threshold_sweep.json` → `recommended.*` | verified-results | none |
| Y-05 | Budget comments: 98.2% / 96.4% / 92.9%, 0.33 → 85.7%; "rate 0.7 gives the same retention" | policy.yaml:53, 55, 62, 74 | a | `compression_eval.json` → `grid`, `recommendation` | verified-results | none |
| Y-06 | "keep ~50% of context tokens … LLMLingua-2 kept only 51.8%" | policy.yaml:65 | a | 51.8% is evidence *retention*; LLMLingua-2@0.5 kept 51% of *tokens* (`methods["llmlingua2@0.5"]`) | contradicted | Ambiguous wording, open item 5 |
| Y-07 | TTL 604,800 s = 7 days; max_temperature 0.3 | policy.yaml:35, 37 | a | 7 × 86,400; docs agree (ARCHITECTURE.md:112, semantic_cache.md:99) | verified-results | none |
| PR-01 | gpt-5.4-mini $0.75 / $4.50 / cached $0.075 | prices.yaml:5 | b | https://developers.openai.com/api/docs/pricing | verified-live | none |
| PR-02 | gpt-5-nano $0.05 / $0.40 / $0.005 | prices.yaml:6 | b | same page | verified-live | none |
| PR-03 | gpt-4o-mini $0.15 / $0.60 / $0.075 | prices.yaml:7 | b | same page | verified-live | none |
| PR-04 | gpt-6-luna $0.10 / $0.50 / $0.01 | prices.yaml:8 | b | same page | verified-live | none |
| PR-05 | gemini-2.5-flash $0.30 / $2.50 / $0.03 | prices.yaml:9 | b | https://ai.google.dev/gemini-api/docs/pricing | verified-live | none |
| PR-06 | gemini-2.5-flash-lite $0.10 / $0.40 / $0.01 | prices.yaml:10 | b | same page | verified-live | none |
| PR-07 | claude-haiku-4.5 $1 / $5 / read $0.10 / write $1.25 | prices.yaml:11 | b | https://platform.claude.com/docs/en/about-claude/pricing | verified-live | none |
| PR-08 | claude-sonnet-5.5 $2 / $10 / read $0.20 / write $2.50 | prices.yaml:12 | b | same page | verified-live | none |
| PR-09 | text-embedding-3-small $0.02 | prices.yaml:13 | b | OpenAI pricing page | verified-live | none |

### Research report spot-check (`docs/research/LLM CostGuard project research.md`; not edited)

| ID | Claim (short) | Location | Type | Backing | Status | Action |
|---|---|---|---|---|---|---|
| RR-01 | RouteLLM saves > 85% on MT-Bench, 45% on MMLU, 35% on GSM8K, at 95% of GPT-4 quality | research:3, 12 | b | https://lmsys.org/blog/2024-07-01-routellm/ | verified-live | none |
| RR-02 | "3.66×" = CPT(50%) 49.03% (random) vs 13.40% (MF) on MT-Bench | research:16 | b | https://arxiv.org/html/2406.18665 Table 1 (49.03 / 13.40 = 3.66) | verified-live | none |
| RR-03 | FrugalGPT: up to 98% cost reduction | research:12 | b | https://arxiv.org/abs/2305.05176 | verified-live | none |
| RR-04 | LLMLingua-2: QA −0.8 points, BLEU 22.34 → 17.37, LongBench 44.0 → 39.1, 2.1 vs 16.6 GB, GPT-3.5 / Mistral-7B | research:3, 13, 18, 131 | b | https://arxiv.org/html/2403.12968 (QA 87.75 → 86.92) | verified-live | none |
| RR-05 | SCALM 4.5% (MOSS) / 7.5% (LMSYS) | research:3, 11 | b | https://arxiv.org/html/2406.00025 | verified-live | none |
| RR-06 | GPT Semantic Cache 61.6–68.8% hits, > 97% valid | research:11 | b | https://arxiv.org/abs/2411.05276 | verified-live | none |
| RR-07 | AWS 87.6% hits at 0.80; per-request error 1.9% at 0.99 and 7.2% at 0.80 | research:11, 16 | b | AWS benchmark table (0.99: 23.5% × 7.9% = 1.86%; 0.80: 87.6% × 8.2% = 7.2%) | verified-live | none |
| RR-08 | vCache: 12.5× hits, 26× lower error, ICLR 2026, error = FP / n, 1.7% at 0.99 after 150k | research:16, 28, 63 | b | https://arxiv.org/abs/2502.03771 and its HTML version | verified-live | none |
| RR-09 | MeanCache: 233 vs 89 false hits on 700 queries; τ 0.78 (ALBERT) / 0.83 (MPNet); 0.66 → 0.98 | research:53-54, 59, 94 | b | https://arxiv.org/html/2403.02694v3 | verified-live | none |
| RR-10 | Krites up to 3.9×, no added latency on the serving path | research:29 | b | https://arxiv.org/abs/2602.13165 | verified-live | none |
| RR-11 | Query embeddings "structurally lose" validity information | research:92 | b | arXiv 2609.35908 abstract says "can lose" | contradicted | Wording overstated; record not edited (open item 7) |
| RR-12 | Portkey: default 0.95, ≤ 4 messages, ignores the system prompt, semantic cache only on select Enterprise plans | research:38, 57, 96 | b | https://portkey.ai/docs/product/ai-gateway-streamline-llm-integrations/cache-simple-and-semantic | verified-live | none |
| RR-13 | RedisVL default `distance_threshold` 0.1 (cosine distance) | research:36, 44 | b | https://docs.redisvl.com/en/latest/user_guide/03_llmcache.html | verified-live | none |
| RR-14 | Anthropic multipliers 0.1× (0.05× Opus 5.5), 1.25× / 2×; minimums 512–4,096; tool-use prompt 286–675 tokens; tokenizer +30% | research:111, 236, 282-284, 315 | b | Anthropic pricing and prompt-caching pages (Opus 5.5 286 tokens, Opus 4.7 675) | verified-live | none |
| RR-15 | OpenAI table prices; Fast (formerly Priority) 2×; GPT-5.6-sol promo until at least 21 Nov 2026; GPT-6-astra $50 output | research:262-274, 287, 302, 328 | b | OpenAI pricing page | verified-live | none |
| RR-16 | Gemini prices, free tiers (none on 3.1 Pro), free-tier data used to improve products, 3.x Flash promo to 31 Dec 2026 | research:263, 267, 271, 293, 301 | b | Gemini pricing page | verified-live | none |
| RR-17 | Groq free plan: 30 RPM / 1K RPD / 8K TPM / 200K TPD | research:294 | b | https://console.groq.com/docs/rate-limits | verified-live | none |
| RR-18 | Gemini publishes no free-tier limit numbers; limits "not guaranteed" | research:293 | b | https://ai.google.dev/gemini-api/docs/rate-limits | verified-live | none |
| RR-19 | Qdrant Cloud free tier: 0.5 vCPU, 1 GB RAM, 4 GB disk | research:368 | b | https://qdrant.tech/pricing/ | verified-live | none |
| RR-20 | Langfuse Hobby: 50k units, 30-day data access | research:371 | b | https://langfuse.com/pricing | verified-live | none |
| RR-21 | Render RAM "commonly reported as 512 MB, but unverified" | research:387 | b | render.com/docs/compute-plans now confirms 512 MB | contradicted | The research is outdated, not wrong in substance (open item 7) |
| RR-22 | Fly.io trial: 2 h of machine time or 7 days; HF Docker/Gradio need PRO | research:385-386 | b | Fly.io and HF pages. HF also allows free accounts up to 2 Gradio Spaces on ZeroGPU, which the research omits; Docker still needs a paid plan | verified-live | none |
| RR-23 | Bitext: 26,872 pairs, 27 intents, CDLA-Sharing-1.0 | research:395 | b | HF dataset card | verified-live | none |
| RR-24 | RouterArena and LLMRouterBench findings | research:12, 214 | b | arXiv 2510.00202v1, 2601.07206 | verified-live | none |
| RR-25 | MT-Bench: 65% consistency; 66 / 85% and 63 / 81% agreement; reference-guided maths failures 70% → 15% | research:182, 245 | b | arXiv 2306.05685v4 | verified-live | none |
| RR-26 | VentureBeat: 18 / 47 / 35% split, hits 18% → 67%, bill −73%, thresholds 0.88 → 0.97 | research:14, 24, 61 | b | VentureBeat article | verified-live | none |
| RR-27 | AWS: start at 0.90–0.95; TTLs 24 h / 1–4 h / 5–15 min | research:63, 103 | b | AWS best-practices page | verified-live | none |
| RR-28 | Budget and per-1k-request arithmetic ($2.36, $0.20, $14.00 … $0.23; 47.5% / 75% / 25% / 10% routing examples) | research:227-232, 260-273, 464-465 | a | Recomputed from the verified-live prices; all match | verified-results | none |
| RR-29 | Course facts: 20 marks, due 8 Oct, 10% per day late, rubric weights, slot and reviewers | research:3, 413, 498-506, 568 | c | course_rubric_mapping.md:19-20, 40-44, 95-98 | verified-course | none |
| RR-30 | LLMLingua-2 large is 0.6B params in F32; bge-small 33.4M MIT; MiniLM 22.7M Apache | research:80-81, 133 | b | HF model cards | verified-live | none |
| RR-31 | Cohere Rerank 4 costs $2.00–2.50 per 1k via OpenRouter | research:146 | b | Cited URL now 404; the price is confirmed by search results | verified-live | Open item 7 |

---

## Fixes applied

All edits are to Markdown docs in scope. No code, config, results, research or cassette file was touched.

**`docs/ARCHITECTURE.md`**

1. :94. Replaced "Fits a 512 MB free host; LLMLingua-2 needs ~2 GB of torch" with the verified Render RAM and the measured LLMLingua-2 peak RSS (1.6 GB / 4.1 GB).
2. :143. Replaced "real Sonnet/Haiku TTFT is seconds" with "a real Sonnet/Haiku completion takes seconds (not measured here)".
3. :145-146. Marked the hook and SQLite micro-benchmarks as "not saved under `eval/results`".
4. :205. Scope now says Anthropic is the real API backend, and MLX (Qwen2.5-7B / 1.5B, billed as GPT-5.4-mini / GPT-5-nano) is the stand-in used for the measured runs.

**`docs/DESIGN_DECISIONS.md`**

5. :26. AWS example now names its benchmark context (Titan V2, SemBenchmarkLmArena) and the 91.8% / 87.6% / 7.2% derivation.
6. :48. "Published winner" replaced by what Jha et al. actually say.
7. :53. LLMLingua-2 "about 2 GB" and heuristic "under 1 ms" replaced by measured values.
8. :65. Haiku passing the gate is now an expectation, not a fact.
9. :68. New bullet: measured runs use MLX billed at a ~12× pair, so caveat the routing step.
10. :72. LiteLLM "blurs them" corrected to "re-maps them into the OpenAI usage shape".
11. :90. Hook timings labelled as an unsaved micro-benchmark.
12. :95. Langfuse "about 8k requests" now states the ~6 units/request assumption.
13. :158. "About 10–20 req/s" corrected to "8–20 req/s" (40 threads / 5 s = 8).

**`docs/RUNBOOK.md`**

14. :232. Demo PR baseline τ corrected from 0.90 to the deployed 0.93.

**`docs/EVALUATION.md`**

15. :14. Eval-set row says only AI-written `seed.jsonl` exists today.
16. :32. Removed the unsourced "brief's 10% exact repeats" quote.
17. :135. Anthropic pre-flight estimate labelled as unsaved console output.
18. :181. "Output costs 5× input" now adds 6× / 8× for the GPT prices that mlx and mock bill at.
19. :189. Names the mlx judge (Qwen2.5-7B-Instruct-4bit).
20. :220. Qwen smoke-run observation labelled as unsaved.
21. :223-224. Self-preference caveat extended to the mlx judge grading itself.
22. :252-253. "~40 rows per author group" corrected to 10–15 per author (60–90 total), with today's 44 seed rows.
23. :259. SCALM figures now carry their dataset and model context.
24. :315-317. τ = 0.6 demo figures: says which number is saved (60.5%) and which are not.

**`docs/PRESENTATION_OUTLINE.md`**

25. :20. Rubric row renamed to "Presentation quality (delivery; results slide)".
26. :50. Headline source corrected to `ab_summary_mlx.json` (or `ab_summary.json` for paid runs) with the real key names (`savings_pct` / `savings_ci`, `quality.retained` / `retained_ci`), "list-price equivalent" wording, and never the mock file.
27. :85. Upper-bound caveat on Bitext hit rates.
28. :100. MLX ~12× caveat next to the 2× routing message.
29. :117. Hand-written eval set caveat.
30. :166. Waterfall source corrected.
31. :170. Output-price ratio context; routing cap scoped to Anthropic.
32. :207. Q&A answer on eval-set provenance now states the current seed-only state and the gate's Bitext + KB fallback.
33. :234. MLX upper-bound note in the "2× gap" answer.

**`docs/CONTRACT.md`** (the coordinator has since committed both edits in `db5e5b2`)

34. :11. Eval-set line states the AI-written seed reality.
35. :103. Reconciles the contract's savings definition with EVALUATION's paired A0 headline.

**`docs/components/semantic_cache.md`**

36. :46. Bitext upper-bound caveat on the 10–24% exact-text share.
37. :93. Seed traps described as AI-written scaffolding, not "hand-written".
38. :118. "Structurally lose" changed to the paper's "can lose", with the arXiv id.
39. :138-139. Trap and paraphrase rows marked AI-written.
40. :143. Bitext echo statistics labelled as not saved.
41. :184. Stale "current policy.yaml starting points 0.95 / 0.90 / 0.85" updated to 0.95 / 0.93 / 0.89.
42. :188. SCALM context added; the unsourced "support/FAQ 20–60%" replaced by Portkey's verified figure and its context.
43. :205. Unmeasured SN-90211 example replaced by the measured SN-48213 / SN-48231 pair (0.976).
44. :245. Model-load time now cites the saved 58 ms and labels the unsaved values.
45. :247. "130 MB of ONNX weights" corrected to 66 MB on disk plus the measured 302 MB RSS context.
46. :253. Network-hop latency labelled as an unmeasured estimate.

**`docs/components/context_and_compression.md`**

47. :134. "~100 output tokens" labelled as an assumption.
48. :187. "Today's balanced policy 4.99×" corrected (old 1200 / 0.5 = 4.99×; current 1000 / 0.5 = 5.36×).
49. :218-226. The "compared with policy.yaml today" block rewritten in the past tense with the current policy values.

**`docs/components/router_and_gate.md`**

50. :56. Source line says the Bitext and KB hardness rows are not saved.
51. :97. "Team seed eval set" relabelled "Seed eval set (AI-written)".
52. :105. Router micro-benchmark labelled as unsaved, and the load-test router timings added.
53. :125. "The real experiments run on Anthropic" corrected, with the MLX ~12× caveat.
54. :127. "25% on that slice" corrected to "25% overall (50% on that half)".
55. :183. "2–8 routable items" corrected to 2–7.
56. :201. Anthropic gate estimates labelled as unsaved console output.
57. :257. "Team's eval set" relabelled as the 44 AI-written seed rows.
58. :258. 96.7% corrected to 96.6%.

**`eval/data/kb/README.md`**

59. :9. "~150-250-token passages" corrected to "61 passages of 99-315 tokens (median 219)".

---

## Open items for the coordinator

1. **Replace the mock A/B with the real MLX run.**
   - `docs/RESULTS.md` and every slide number must come from `eval/results/ab_summary_mlx.json`, regenerated with `python -m eval.report`. Its dollars are list-price equivalents at GPT-5.4-mini / GPT-5-nano prices.
   - The report's headline sentence should say "list-price equivalent" and name the judge: Qwen2.5-7B, the same model as the strong tier.
   - Also needed from the real run: the cost per correct answer, the hand-label agreement (κ), and the dup00/15/50 sensitivity summaries.
2. **The router gate now running on MLX is weaker evidence than the docs imply.**
   - `eval/results/logs/gate_mlx.log` shows it found no hand-written eval set. It fell back to 304 Bitext + KB items, and Qwen2.5-7B judges its own answers against Qwen2.5-1.5B.
   - It will write `configs/router_gate.json`. Label its verdicts "Bitext + KB fallback, same-family judge" in the router doc, the deck and the README.
   - DESIGN_DECISIONS #9 still has no Anthropic gate evidence.
3. **Hand-written eval set.**
   - All 44 eval rows and all 34 trap pairs are AI-written (`author: "seed"`).
   - These numbers are on seed data and must be re-run once the team's `<name>.jsonl` files land: the classifier "OOD" accuracies, the trap pass rate, the CI subset, and the router gate.
   - Until then, any slide that says "hand-written" is not yet true.
4. **Measurements quoted in docs but not saved under `eval/results`.** Save each one to a JSON file, or drop it from the docs:
   - hook timings (19 µs / 3 µs) and the SQLite write (~0.5 ms);
   - router overhead (0.03 / 1.9 ms);
   - hardness rates on Bitext (5.6%) and the KB (2/12, 1/44);
   - Bitext slot-echo rates (100 / 98 / 70%);
   - the τ = 0.6 CI demo (72.7%, 11 trap and 22 total false hits);
   - the Anthropic pre-flight estimate (684 generations, ≤ $4.49) and the gate estimates (≤ $0.64 / $3.30);
   - the Qwen position-bias smoke run;
   - the first-download time (15 s);
   - KB chunk statistics (reproducible from `eval.kb.load_kb()`, but not in a results file).
5. **Code comments and data labels this audit may not edit.**
   - `configs/policy.yaml:2` still calls the thresholds "starting points". They are now the calibrated values.
   - `configs/policy.yaml:65`: "LLMLingua-2 kept only 51.8%" conflates evidence retention with tokens kept. Suggested wording: "LLMLingua-2@0.5 retained only 51.8% of key facts".
   - These all describe the `author: "seed"` traps and paraphrases as "hand-written (this repo)", which contradicts the evalset README convention: `eval/cache_pairs.py:18, 21, 55-56`, `eval/data/cache_pairs/pairs_v1.manifest.json` → `licences`, and `threshold_sweep.json` → `datasets.licences`.
6. **Report generator (`eval/report.py` → `RESULTS.md`).**
   - Add the Bitext upper-bound caveat to the threshold-calibration table.
   - The "Recommended settings" table shows quality at compression rate 0.70, while `policy.yaml` deliberately runs quality with compression off. Show the deployed policy too.
   - Consider leaving mock quality numbers out of the mock headline sentence.
7. **Source pages that disagree with the research record** (the research file was left as is):
   - Render free RAM is now confirmed at 512 MB (https://render.com/docs/compute-plans); the research says "unverified".
   - arXiv 2609.35908 says embeddings "can lose" information, not "structurally lose".
   - The cited OpenRouter Cohere URL returns 404; the prices still hold via search.
   - HF Spaces now allows free accounts up to 2 Gradio Spaces on ZeroGPU. Docker is still paid, so the conclusion stands.
8. **Decide which price gap the deck headlines.** The Anthropic production pair is 2×; the measured MLX runs bill at ~12×. The docs now caveat this, but the deck should show routing savings for both pairs, or reprice the MLX token counts at Sonnet / Haiku prices. That is a PriceBook change, coordinator-owned.
9. **Minor consistency points.**
   - `CONTRACT.md:89` lists `qqp` as a trace source, but the trace has none.
   - `EVALUATION.md:307` gives "$0.2–0.5" for real-model CI with no source.
10. **"Fits a 512 MB host" is not measured for the server.** The eval process peaked at 443 MB with bge-small and the cross-encoder loaded. Measure the server's RSS (uvicorn, FastAPI, both models, an empty cache) before claiming it fits Render free.
11. **Concurrent changes and git provenance.**
    - During the audit the branch history was rewritten. Commit `db5e5b2` already includes this audit's two `CONTRACT.md` fixes (items 34-35).
    - Files this audit did not touch also changed in the working tree: `Makefile`, `costguard/obs/metrics.py`, `eval/gate_router.py`, `tests/test_ops.py`, `loadtest/*` and `eval/results/loadtest_quick.json`. No doc cites `loadtest_quick.json`, and `loadtest.json` (the file the docs cite) was unchanged.
    - After the rewrite, the provenance SHAs recorded in the results are on no branch: `ab_summary_mock.json` → `meta.git_commit` `fca1374` (also printed in `RESULTS.md:69`), and `loadtest.json` → `git_sha` `1710507`. Re-run, or note the mapping, before quoting provenance.
12. **Two rollout ramps.**
    - RUNBOOK §3 ramps a policy 0.5 → 1 → 5 → 10 → 50%, the course ramp.
    - router_and_gate.md §5 ramps a category canary 5% → 25%.
    - Both are defensible, but say in one line why they differ, or align them.
