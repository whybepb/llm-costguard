# LLM Model Routing, Cascades, "Model Downshift" and LLM API Pricing / Provider Cost Levers (as of 2026-10-03)

All pricing pages were fetched on **2026-10-03** unless noted otherwise. Prices are USD per 1M tokens (MTok). "Strong" means the frontier or expensive model and "weak" or "cheap" means the downshift target.

---

## Q1. What are the main routing / cascade approaches, and how do they work?

### Takeaway
There are three families:
- **Cascades** try the cheap model first and escalate when a scorer or self-check says the answer is bad. Examples are FrugalGPT and AutoMix.
- **Predictive routers** pick one model before generating, using a learned classifier, embedding similarity or matrix factorisation. Examples are RouteLLM, Hybrid LLM, Avengers-Pro and Not Diamond.
- **Heuristic or market routers** use rules, task-type classifiers or aggregate spend data. An example is the OpenRouter Auto Router.

Independent 2025–2026 benchmarks (RouterArena, LLMRouterBench) find that many sophisticated and commercial routers do not reliably beat simple baselines. That supports a simple, well-evaluated router for a 5-day student build.

### Cited Findings

**FrugalGPT (Chen, Zaharia, Zou; arXiv 2305.05176, May 2023)**
- The paper proposes three cost strategies: **prompt adaptation, LLM approximation, and LLM cascade**. FrugalGPT is the cascade. It "learns which combinations of LLMs to use for different queries" to optimise the cost/accuracy trade-off — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- Mechanism: queries go sequentially through a chain of LLM APIs, cheapest first. A learned scoring function judges each answer, and the query stops at the first answer whose score clears a learned threshold — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176). The abstract states the cascade concept; scorer details are in the paper body and were not re-verified this session (see Gaps).

**RouteLLM (Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica; LMSYS/Anyscale; arXiv 2406.18665, v1 26 Jun 2024, v4 23 Feb 2025; ICLR 2025)**
- Routers "dynamically select between a stronger and a weaker LLM during inference". They are trained on **human preference data plus data augmentation** — [arXiv 2406.18665](https://arxiv.org/abs/2406.18665); ICLR 2025 per the [arXiv PDF header](https://arxiv.org/pdf/2406.18665)
- Five routers are in the repo: **`mf` (matrix factorisation, "recommended")**, `sw_ranking` (similarity-weighted Elo), `bert` (BERT classifier), `causal_llm` (LLM-based classifier) and `random` — [lm-sys/RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
- Training data was about **55k Chatbot Arena preference samples**. Augmentation used **LLM-judge labels** and **golden labels from the MMLU validation split** (about 1,500 samples, "less than 2% of the overall training data") — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
- The router outputs a win-probability for the strong model. You **calibrate a threshold** to a target share of strong-model calls. The repo's example is that calibrating for 50% strong calls gives threshold 0.11593, used as the model name `router-mf-0.11593`. It ships as an OpenAI-compatible server (`pip install "routellm[serve,eval]"`) — [lm-sys/RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
- Routers were trained on the pair **gpt-4-1106-preview (strong) / Mixtral-8x7B-Instruct (weak)**. The authors say they "generalize well to other strong and weak model pairs" without retraining — [GitHub](https://github.com/lm-sys/RouteLLM). The blog shows transfer to **Claude 3 Opus / Llama 3 8B** — [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
- Stated limitations:
  - real-world distributions can differ substantially from benchmarks;
  - routing is **binary only** (two models);
  - "Performance between different routers trained on the same dataset can vary widely on the same benchmark without clear explanation."
  - Source: [arXiv HTML 2406.18665](https://arxiv.org/html/2406.18665)

**Hybrid LLM (Ding, Mallick, Wang, Sim, Mukherjee, Ruhle, Lakshmanan, Awadallah; Microsoft/UBC; ICLR 2024; arXiv 2404.14618)**
- A router picks between a small (edge) model and a large model. It uses **predicted query difficulty** and a **desired quality level that is tunable at test time**, so the cost/quality trade-off is a runtime knob — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)

**AutoMix (Aggarwal, Madaan et al.; arXiv 2310.12963)**
- Three steps: (1) the small model generates; (2) the small model **self-verifies**, framed as an **entailment** check of the answer against the context; (3) a **POMDP-based router** uses the noisy self-verification confidence to decide whether to escalate. For N models the process repeats up the chain — [AutoMix arXiv HTML v5](https://arxiv.org/html/2310.12963v5); code at [automix-llm/automix](https://github.com/automix-llm/automix)

**Embedding/cluster ("mixture-of-experts-style") routers**
- **Avengers-Pro** (arXiv 2508.12631, Aug 2025, rev. Oct 2025) embeds queries, **clusters** them, and routes each cluster to the model with the best **performance-efficiency score**. Its pool is 8 models, including GPT-5-medium, Gemini-2.5-pro and Claude-opus-4.1 — [arXiv 2508.12631](https://arxiv.org/abs/2508.12631)
- A 2026 survey, "Dynamic Model Routing and Cascading for Efficient LLM Inference" (arXiv 2603.04445), covers the field. It describes **RouterBench** as more than 405k precomputed outputs from 11 LLMs across 7 tasks (MMLU, MT-Bench, MBPP, HellaSwag, WinoGrande, GSM8K, ARC) with cost metadata — [Survey arXiv 2603.04445](https://arxiv.org/html/2603.04445v2)
- Other 2025–2026 router papers surfaced but not read: R2-Router ([arXiv 2602.02823](https://arxiv.org/html/2602.02823v1)), OrcaRouter, which uses hybrid offline–online learning ([arXiv 2605.30736](https://arxiv.org/html/2605.30736v1)), and LLM Routing with Dueling Feedback ([arXiv 2510.00841](https://arxiv.org/pdf/2510.00841))

**Independent router benchmarks (important for "should we build or buy")**
- **RouterArena** (arXiv 2510.00202; ICLR 2026). Metrics are accuracy, cost, routing optimality ("cheapest correct selection"), robustness to query perturbation, and router latency. Findings:
  - "no router ranks at the top across all metrics";
  - **GPT-5 (as a router) ranks #7 "due to its restricted model pool"**;
  - **NotDiamond "ranks #12 because it frequently selects expensive models"**;
  - "commercial routers tend to achieve higher accuracy at a much greater expense, while open-source routers often present more cost-efficient solutions";
  - **vLLM-SR and CARROT reach "roughly 35% lower cost with under 2% accuracy degradation"**;
  - all routers fall short of the oracle "because they are inefficient at recognizing when smaller, cheaper models are sufficient."
  - Sources: [RouterArena arXiv HTML](https://arxiv.org/html/2510.00202v1); leaderboard at [routeworks.github.io](https://routeworks.github.io/), code at [RouteWorks/RouterArena](https://github.com/RouteWorks/RouterArena)
- **LLMRouterBench** (arXiv 2601.07206, 12 Jan 2026) has over 400K instances, 21 datasets, 33 models and 10 baselines. Findings:
  - "many routing methods exhibit similar performance under unified evaluation, and several recent approaches, **including commercial routers, fail to reliably outperform a simple baseline**";
  - "backbone embedding models have limited impact";
  - "larger ensembles exhibit diminishing returns compared to careful model curation";
  - a substantial oracle gap remains, "driven primarily by persistent model-recall failures."
  - Source: [arXiv 2601.07206](https://arxiv.org/abs/2601.07206)

**Commercial routers (state as of Oct 2026)**
- **OpenRouter Auto Router (`openrouter/auto`)**:
  - It classifies each prompt into about **30 task types** (e.g. "code:debugging", "customer_support").
  - It ranks models by **aggregate OpenRouter spend for that task type over a trailing 7-day window**, then applies a cost tier (low / medium / high / xhigh / max).
  - "You pay the standard rate for whichever model is selected. There is **no additional fee**."
  - Models can be restricted with `allowed_models` / `excluded_models` wildcards such as `anthropic/*`, and the response `model` field shows which model ran.
  - Source: [OpenRouter docs](https://openrouter.ai/docs/guides/routing/auto-model-selection). The current docs describe this spend-based mechanism, not a Not Diamond-powered one.
- **Not Diamond** is a model *selector* (it recommends; your app executes the call) — [Maxim AI roundup](https://www.getmaxim.ai/articles/top-5-auto-routing-tools-for-llm-apps-in-2026/):
  - Pay-as-you-go is **$0.05 per million tokens routed**; enterprise pricing is custom.
  - Vendor claims are "at least 20-40% cost savings, and often more, without any degradation in quality", "10x ROI", and a modelled example of $4.8M/yr → $3.6M/yr (25% lower).
  - Source: [Not Diamond pricing](https://www.notdiamond.ai/pricing). These claims are vendor-reported, and RouterArena's #12 ranking contradicts them (above).
- **Martian** was the first commercial LLM router ("model mapping" interpretability) and raised a $9M seed — [Yahoo Finance press release](https://finance.yahoo.com/news/martian-invents-model-router-beats-190000381.html). A Medium post claims it cuts "costs by 20% to 97%" and reports a ~$1.3B valuation in April 2026. This is a **low-quality, unverified source** — [Medium](https://medium.com/@sarawgiapoorvwork347/martian-the-san-francisco-based-startup-that-invented-the-first-llm-router-is-reportedly-nearing-4211dd768296)
- **Unify**: RouteLLM's 2024 blog benchmarked against Martian and Unify (see Q2). No current (2026) Unify product information was found this session.
- **LiteLLM** (self-hosted proxy, used as a gateway or router) is described in a third-party roundup as "8ms P95 at 1,000 RPS, zero per-request cost, and 100+ provider integrations". This is vendor/roundup data, not independently verified — [toolchew](https://toolchew.com/en/best-llm-router/) via search snippet

**Heuristic / signal-based escalation**
- **Token counting is explicitly documented as a routing input.** Anthropic's token-counting page lists "Make smart model routing decisions" as a use case for prompt-length-based routing — [Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
- **Self-verification-based escalation** is the AutoMix mechanism (above) — [AutoMix](https://arxiv.org/html/2310.12963v5)
- A runtime quality-threshold knob is the Hybrid LLM mechanism — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)

### Inferences
- For a 5-day build, the most defensible design has three parts:
  - **(a) a pre-routing tier**: rules plus a cheap classifier (prompt length via token count, task-type keywords or regex, presence of code or maths, conversation depth), or a **kNN/embedding router** over a small labelled set;
  - **(b) an optional cascade tier** for categories where the cheap model's answer can be cheaply verified (format or JSON validity, unit tests, self-verification);
  - **(c) an offline eval gate** that approves the policy per category.
  - LLMRouterBench and RouterArena both suggest simple baselines are competitive, so the eval gate matters more than router sophistication.
- RouteLLM's pre-trained `mf` router is usable out of the box via its OpenAI-compatible server. However, it was trained on GPT-4-Turbo/Mixtral preference data from 2024. Its threshold must be **re-calibrated on the team's own traffic sample** and judged by the team's own eval, because the paper itself warns of distribution mismatch.
- Cascades vs routers is a cost-structure question (see Q2 inferences). A cascade always pays for the cheap call, so it only pays off when the cheap model costs a small fraction (roughly ≤10–15%) of the strong model and escalation rates are low.

### Gaps
- FrugalGPT's scorer architecture (the paper body describes a small fine-tuned model scoring query–answer pairs) and its per-dataset numbers were not re-verified this session. Only the abstract was fetched.
- AutoMix's venue and exact per-dataset cost/quality numbers were not fetched beyond the ">50% cost reduction for comparable performance" summary.
- No current 2026 information on Unify, Azure AI Foundry Model Router or AWS Bedrock Intelligent Prompt Routing was gathered. Martian's current pricing and product claims could not be verified from a primary source.
- RouterArena's exact numeric leaderboard table (per-router accuracy and cost) was not extracted, only the rank statements quoted above.

---

## Q2. What published savings numbers exist (exact figures, benchmarks, models, caveats)?

### Takeaway
Headline numbers are real but narrow:
- **RouteLLM**: ">85% cost reduction at 95% of GPT-4 quality" holds **only on MT-Bench** with GPT-4-Turbo vs Mixtral-8x7B and LLM-judge-augmented training. It drops to 45% on MMLU and 35% on GSM8K.
- **FrugalGPT**: "up to 98%" is a best-case figure on 2023 APIs and specific datasets.
- **Recent multi-model routers and independent benchmarks** report more modest **~25–35% savings at near-matched quality**.

### Cited Findings

**RouteLLM**

| Claim | Benchmark / setup | Source |
|---|---|---|
| Cost reduced by **over 85%** while keeping **95% of GPT-4 performance** | MT-Bench; GPT-4 Turbo vs Mixtral 8x7B | [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/) |
| **45%** reduction at 95% GPT-4 quality | MMLU | [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/) |
| **35%** reduction at 95% GPT-4 quality | GSM8K | [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/) |
| MT-Bench at 95% quality needs **26% GPT-4 calls** (MF, Arena data only) and **14% GPT-4 calls** (MF, Arena + LLM-judge augmentation) | MT-Bench | [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/) |
| MMLU needs **54% GPT-4 calls** (causal-LLM router, Arena + golden-label augmentation) | MMLU | [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/) |
| "**Over 40% cheaper**" than the commercial routers **Martian and Unify AI** at the same performance (2024) | MT-Bench | [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/); [GitHub](https://github.com/lm-sys/RouteLLM) |
| Abstract wording: "reduces costs—**by over 2 times in certain cases**—without compromising the quality" | multiple | [arXiv abstract](https://arxiv.org/abs/2406.18665) |
| "**Up to 3.66x** cost savings" on MT-Bench at the 95% GPT-4 quality level; **1.41x** on MMLU (92% quality); **1.49x** on GSM8K (87% quality) | paper tables | [arXiv HTML](https://arxiv.org/html/2406.18665) |

- Paper metrics:
  - **PGR** = (router perf − weak perf) / (strong perf − weak perf).
  - **APGR** = the integral of PGR over 0–100% strong-call budgets, approximated over 10 cost levels.
  - **CPT(x%)** = the minimum % of strong-model calls needed to reach x% PGR.
  - Source: [arXiv HTML](https://arxiv.org/html/2406.18665)
- MT-Bench table:
  - Random: CPT(50%) 49.03%, CPT(80%) 78.08%, APGR 0.500.
  - **MF (Arena + Judge): CPT(50%) 13.40%, CPT(80%) 31.31%, APGR 0.802.**
  - GSM8K: Random CPT(50%) 50.00% vs causal-LLM (Arena + Judge) 33.64%; APGR 0.497 vs 0.622.
  - MMLU: Random CPT(50%) 50.07% vs SW-ranking (Arena + Gold) 35.40%; APGR 0.603.
  - Source: [arXiv HTML](https://arxiv.org/html/2406.18665)
- **Router overhead**: SW-ranking adds **$39.26 per million requests**, GPU-based routers about **$3–5 per million requests**, and the costliest router adds "no more than **0.4%** overhead compared to GPT-4 generation" — [arXiv HTML](https://arxiv.org/html/2406.18665)

**FrugalGPT**
- Can "match the performance of the best individual LLM (e.g. GPT-4) with **up to 98% cost reduction**", or "improve the accuracy over GPT-4 by **4%** with the same cost" — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)

**Hybrid LLM**
- "Up to **40% fewer calls to the large model**, with **no drop in response quality**" — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)

**AutoMix**
- Across five LMs and five datasets, it reduces "computational cost by **over 50%** for comparable performance" — [AutoMix](https://arxiv.org/html/2310.12963v5) (via search summary)

**Avengers-Pro (2025, GPT-5-era pool)**
- **+7%** average accuracy over the strongest single model.
- **Matches the strongest single model at 27% lower cost.**
- About **90% of peak performance at 63% lower cost**.
- Source: [arXiv 2508.12631](https://arxiv.org/abs/2508.12631)

**RouterArena (independent)**
- The best open-source routers (vLLM-SR, CARROT) achieve "roughly **35% lower cost with under 2% accuracy degradation**" — [RouterArena](https://arxiv.org/html/2510.00202v1)

**Vendor claims (unverified)**
- Not Diamond: "at least **20-40%** cost savings" — [Not Diamond pricing](https://www.notdiamond.ai/pricing)
- Martian: "20% to 97%" — [Medium (low quality)](https://medium.com/@sarawgiapoorvwork347/martian-the-san-francisco-based-startup-that-invented-the-first-llm-router-is-reportedly-nearing-4211dd768296)

### Inferences
- **The 3.66x figure is the CPT(50%) ratio vs the random router: 49.03% / 13.40% ≈ 3.66.** On MT-Bench, 50% PGR roughly equals 95% of GPT-4's score. It is therefore "fewer strong-model calls than random routing", not "3.66x cheaper than all-GPT-4". The ">85% vs all-GPT-4" figure follows from needing only 14% GPT-4 calls when the weak model is far cheaper than GPT-4. Teams should quote both numbers and name the baseline each one uses.
- **Generic savings formulas** for CostGuard, with p = price-weighted cost per request:
  - **Router**: savings ≈ r × (1 − p_cheap/p_strong), where r is the fraction routed to the cheap model, plus router overhead.
  - **Cascade**: savings ≈ 1 − p_cheap/p_strong − e − (verifier cost / p_strong), where e is the escalation rate.
- **Worked example** (my arithmetic, using Oct 2026 list prices from Q4, at 1,500 input / 400 output tokens per request and ignoring tokenizer differences). Strong = **Claude Sonnet 5.5 ($2/$10) = $0.0070/request**.
  - **Downshift to GPT-6-luna** ($0.10/$0.50 = $0.00035/request, a 0.05 price ratio):
    - routing 50% saves **47.5%**; 70% saves **66.5%**; 86% (RouteLLM-like) saves **81.7%**;
    - a cascade with 20% escalation saves **75%**, or about 72% including a cheap self-verify call.
  - **Downshift to Claude Haiku 4.5** ($1/$5 = $0.0035/request, a 0.5 price ratio):
    - savings cap at **50%** even at 100% downshift; routing 50% saves only **25%**;
    - a cascade with 40% escalation saves only 10%, and is **break-even or negative** once a verifier call is added or escalation exceeds 50%.
  - **Lesson**: cascades need a big price gap. Same-vendor "one tier down" moves (Sonnet→Haiku) give much smaller savings than cross-vendor "nano/lite" targets.
- Expect published savings to shrink on real traffic. RouteLLM's own MMLU and GSM8K numbers (45%, 35%) and RouterArena's ~35% are more realistic anchors than 85% or 98%.

### Gaps
- FrugalGPT's dataset-level breakdown (which dataset gave 98%, and which 12 2023-era APIs were in the pool) was not re-verified this session.
- The LMSYS blog does not define CPT/APGR; definitions come from the paper HTML. The paper's exact price assumptions for GPT-4 Turbo and Mixtral in the cost calculation were not extracted.
- No independent replication of Not Diamond or Martian savings claims on a public benchmark was found other than RouterArena's ranking.

---

## Q3. How do you build an "eval gate" for a downshift policy?

### Takeaway
Gate the policy offline on a fixed, stratified eval set:
1. Run both the strong model and the policy.
2. Score with a cross-checked LLM judge (pairwise, both orderings) plus exact-match or programmatic checks where possible.
3. Compute **paired** per-category quality deltas with confidence intervals.
4. Ship only categories whose **CI lower bound** clears a pre-registered threshold.
5. Then run shadow mode → canary → full rollout, with escalation-rate and quality monitors.

A 300–500-prompt gate costs about **$3–10** at Oct 2026 prices, so it fits the budget.

### Cited Findings
- **Quality metric for routers**: use RouteLLM's **PGR / APGR / CPT(x%)**:
  - PGR = share of the weak→strong quality gap recovered;
  - CPT(x%) = strong-call % needed to recover x% of the gap.
  - These turn a router into a cost/quality curve, from which you pick an operating point by threshold — [arXiv HTML 2406.18665](https://arxiv.org/html/2406.18665)
- **Threshold calibration** to a target strong-call share (e.g. 50% → 0.11593 for `mf`) — [RouteLLM GitHub](https://github.com/lm-sys/RouteLLM)
- Hybrid LLM's quality knob is "tunable at test time", so the gate can choose the operating point after training — [arXiv 2404.14618](https://arxiv.org/abs/2404.14618)
- **LLM-as-judge validity** (Zheng et al., MT-Bench / Chatbot Arena paper):
  - Strong judges like GPT-4 achieve "**over 80% agreement**, the same level of agreement between humans".
  - The paper studies **position bias, verbosity bias, self-enhancement bias, and limited reasoning ability**, and proposes mitigations.
  - It released 3K expert votes and 30K human-preference conversations.
  - Source: [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)
- **Statistics** (Evan Miller, Anthropic, "Adding Error Bars to Evals", arXiv 2411.00640):
  - Treat evals as experiments drawn from a super-population.
  - Report standard errors (CLT).
  - Use **clustered standard errors** when questions are grouped.
  - Use **paired-difference tests** when comparing two models on the same questions.
  - Reduce variance by **resampling answers**.
  - Do **power analysis** to choose the sample size.
  - Source: [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- **Router-specific eval dimensions** from RouterArena:
  - accuracy;
  - cost;
  - **routing optimality** (did it pick the cheapest model that was correct?);
  - **robustness** (consistency under query perturbation);
  - **router overhead latency**.
  - Source: [RouterArena](https://arxiv.org/html/2510.00202v1)
- **Distribution shift risk**: RouteLLM notes "real-world applications may have distributions that differ substantially from benchmarks" — [arXiv HTML 2406.18665](https://arxiv.org/html/2406.18665)
- **Cheap infrastructure for the gate**:
  - The OpenAI Batch API gives a "**50% cost discount** compared to synchronous APIs", has a 24h window, takes up to 50,000 requests or 200 MB per batch, and uses a **separate rate-limit pool** — [OpenAI Batch guide](https://developers.openai.com/api/docs/guides/batch)
  - Anthropic and Gemini Batch are also 50% off — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing); [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

### Inferences
- **Recommended gate recipe** (synthesis):
  1. **Eval set**: 300–500 prompts sampled from target traffic (or a public proxy such as MT-Bench, Arena-style prompts or task-specific sets), **stratified by category** (e.g. chit-chat, summarisation, extraction/JSON, code, maths, long-context). Freeze it and version it.
  2. **Ground truth where possible**: exact match or numeric tolerance for maths, JSON-schema validation for extraction, unit tests for code. Use an LLM judge only for open-ended categories.
  3. **Pairwise judge**: compare cheap vs strong answers and run **both orderings** (to cancel position bias). Treat order-inconsistent verdicts as ties. Instruct the judge to ignore length, or length-normalise (verbosity bias). **Don't let a model judge its own family's answers alone** (self-enhancement bias). Use a judge from a third vendor, or two judges and require agreement. Hand-label 30–50 items to measure judge–human agreement, since the MT-Bench paper's ~80% is the reference point.
  4. **Gate metric per category**: "non-inferior rate" = P(cheap answer is a win or tie vs strong). Ship a category only if the **lower bound of the 95% CI ≥ threshold**, e.g. ≥ 0.90 for low-risk categories and higher (or never downshift) for high-risk ones. Use a **paired bootstrap** (resample prompts, recompute the metric, take the 2.5/97.5 percentiles) or the Miller paired-difference SE.
  5. **Sample-size reality check**: the 95% CI half-width for a proportion near 0.5 is ±13.9pp at n=50, ±9.8pp at n=100, ±5.7pp at n=300 and ±4.4pp at n=500. Per-category gates with about 50 items each are only coarse, so pool where you can or put more items into the highest-traffic categories.
  6. **Pick the operating point** on the cost/quality curve by calibrating the router threshold (RouteLLM-style) to the largest downshift share that still passes every category gate.
  7. **Shadow mode**: in production, keep serving the strong model, but also compute the routing decision and (for a sample) the cheap answer offline. Log the would-be savings and judge agreement. No user impact.
  8. **Canary**: route a small slice (e.g. 1–5%) of real traffic. Watch the escalation rate, user-feedback/regeneration rate, error/format-failure rate, and a sampled judge score. Auto-roll back if any monitor breaches its threshold.
  9. **Production monitors**: escalation rate per category (a sudden rise signals drift); a weekly re-run of the frozen eval set when provider models change (model IDs and prices change often, see Q4); and cost per request computed from API `usage` fields.
- **Budget estimate** (my arithmetic; assumes 1,500 input / 400 output tokens per prompt, Sonnet 5.5 as strong model and judge, GPT-6-luna as cheap model, pairwise judging in both orders with a ~300-token rubric and ~150-token verdict):
  - n=300: about **$6.2 at standard prices, about $3.1 via Batch**;
  - n=500: about **$10.4 standard, about $5.2 Batch**.
  - Free tiers (Gemini, Groq) can supply the cheap-model generations at $0.

### Gaps
- No primary source was fetched specifically on shadow/canary practices for LLM routing. The recipe above is standard deployment practice, not a cited finding.
- The MT-Bench paper's quantitative bias measurements (e.g. position-consistency rates per judge) and specific mitigation results were not extracted, only the abstract-level claims.
- No source was found giving recommended non-inferiority thresholds for downshift policies. Thresholds must be chosen and justified by the team.

---

## Q4. Current per-million-token prices (budget, frontier, embeddings) and free tiers, as of Oct 2026

### Takeaway
Cheapest credible downshift targets in Oct 2026:
- **GPT-5-nano** ($0.05/$0.40)
- **GPT-6-luna** ($0.10/$0.50)
- **Gemini 2.5 Flash-Lite** ($0.10/$0.40, also on the free tier)
- **Groq gpt-oss-20b** ($0.075/$0.30, also on the free tier)
- **DeepSeek V4.1-Flash** ($0.15–0.30 / $0.60–1.20 depending on peak hours)
- **Mistral Small 4** ($0.15/$0.60)

Frontier "strong" models cluster at **$2/$10** (Claude Sonnet 5.5, GPT-6.1-sol) to **$10/$50** (GPT-6-astra, Claude Fable 5.1). Several prices are time-limited (Gemini 3.x Flash doubles on 1 Jan 2027; GPT-5.6-sol promo runs through 21 Nov 2026), so CostGuard's price table must be versioned and dated.

### Cited Findings

**Anthropic (official pricing page, fetched 2026-10-03)** — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)

| Model | Input | 5m cache write | 1h cache write | Cache hit | Output | Batch in/out |
|---|---|---|---|---|---|---|
| Claude Fable 5.1 | $10 | $12.50 | $20 | $0.25 (0.025x) | $50 | $5 / $25 |
| Claude Opus 5.5 | $4 | $5 | $8 | $0.20 (0.05x) | $20 | $2 / $10 |
| Claude Opus 5 / 4.8 / 4.7 / 4.6 / 4.5 | $5 | $6.25 | $10 | $0.50 | $25 | $2.50 / $12.50 |
| Claude Sonnet 5.5 | $2 | $2.50 | $4 | $0.20 | $10 | $1 / $5 |
| Claude Sonnet 5 | $2 | $2.50 | $4 | $0.20 | $10 | $1 / $5 |
| Claude Sonnet 4.6 / 4.5 | $3 | $3.75 | $6 | $0.30 | $15 | $1.50 / $7.50 |
| **Claude Haiku 4.5** (cheapest current Claude) | **$1** | $1.25 | $2 | $0.10 | **$5** | $0.50 / $2.50 |

- Sonnet 5: "The $2/$10 … introductory pricing through August 31, 2026, is now the standard price. The previously scheduled increase to $3/$15 … will not occur" — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- "Claude 4.7 and later models … use a newer tokenizer… This tokenizer produces **approximately 30% more tokens for the same text**." Claude Sonnet 4.6 and earlier use the previous tokenizer — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Claude 4.6+ models get the full **1M context at standard pricing**, with no long-context surcharge — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- There is no free tier: "New users receive a **small amount of free credits** to test the API" — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)

**OpenAI (official pricing page, fetched 2026-10-03; no page date shown)** — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)

Flagship table (short context; "long context" means >272K input tokens):

| Model | Input | Cached input | Cache write | Output | Long-ctx in / out | Batch = Flex in / out |
|---|---|---|---|---|---|---|
| gpt-6-astra | $10.00 | $1.00 | $12.50 | $50.00 | $20 / $75 | $5 / $25 |
| gpt-6.1-sol | $2.00 | $0.10 | $2.50 | $10.00 | $4 / $15 | $1 / $5 |
| **gpt-6-luna** | **$0.10** | $0.01 | $0.125 | **$0.50** | $0.20 / $0.75 | $0.05 / $0.25 |

Other current models (Standard tier):

| Model | Input | Cached | Output |
|---|---|---|---|
| GPT-5.6-sol | $4.00 | $0.40 (cache write $5) | $20.00 |
| GPT-5.6-terra | $2.00 | $0.20 (write $2.50) | $12.00 |
| GPT-5.6-luna | $0.20 | $0.02 (write $0.25) | $1.20 |
| GPT-5.5 | $5.00 | $0.50 | $30.00 |
| GPT-5.4 | $2.50 | $0.25 | $15.00 |
| GPT-5.4-mini | $0.75 | $0.075 | $4.50 |
| GPT-5.4-nano | $0.20 | $0.02 | $1.25 |
| GPT-5 | $1.25 | $0.125 | $10.00 |
| GPT-5-mini | $0.25 | $0.025 | $2.00 |
| **GPT-5-nano** | **$0.05** | $0.005 | **$0.40** |
| GPT-4.1 | $2.00 | $0.50 | $8.00 |
| GPT-4.1-mini | $0.40 | $0.10 | $1.60 |
| GPT-4o | $2.50 | $1.25 | $10.00 |
| GPT-4o-mini | $0.15 | $0.075 | $0.60 |
| o3 | $2.00 | $0.50 | $8.00 |
| o4-mini | $1.10 | $0.275 | $4.40 |

- Sources: [OpenAI pricing](https://developers.openai.com/api/docs/pricing) (WebFetch extraction plus direct HTML parse). GPT-4.1-nano ($0.10/$0.40, cached $0.025) is from the [LiteLLM price map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), not confirmed on the official page.
- "GPT-5.6 Sol's **promotional pricing** is available at least through **November 21, 2026**." "**Priority processing was renamed Fast mode on July 30, 2026**." Fast for gpt-6.1-sol is $4/$20, i.e. 2x; Ultrafast for gpt-6-astra is $60/$300, i.e. 6x — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)

**Google Gemini (official pricing page, "Last updated October 1, 2026")** — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

| Model | Free tier | Input | Output (incl. thinking) | Context cache | Batch / Flex in / out | Priority in / out |
|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview | **Not available** | $2.00 (≤200k) / $4.00 (>200k) | $12 / $18 | $0.20 / $0.40 + $4.50/MTok/hr storage | $1 / $6 (≤200k) | — |
| Gemini 3.8 / 3.7 / 3.6 Flash | Yes | **$0.75 through 12/31/26 → $1.50 from 1/1/27** | **$3.75 → $7.50** | $0.075 → $0.15 | $0.375 / $1.875 | $1.35 / $6.75 |
| Gemini 3.5 Flash | Yes | $1.50 | $9.00 | — | $0.75 / $4.50 | $2.70 / $16.20 |
| Gemini 3.5 Flash-Lite | Yes | $0.30 | $2.50 | — | $0.15 / $1.25 | — |
| Gemini 3.1 Flash-Lite | Yes | $0.25 (text/img/video), $0.50 audio | $1.50 | — | $0.125 / $0.75 | — |
| Gemini 2.5 Pro | Yes | $1.25 (≤200k) / $2.50 | $10 / $15 | $0.125 / $0.25 + $4.50/hr | $0.625 / $5.00 | $2.25 / $18 |
| Gemini 2.5 Flash | Yes | $0.30 | $2.50 | — | $0.15 / $1.25 | $0.54 / $4.50 |
| **Gemini 2.5 Flash-Lite** | **Yes ("Free of charge")** | **$0.10** | **$0.40** | $0.01 + $1.00/MTok/hr | $0.05 / $0.20 | $0.18 / $0.72 |

- Gemini 3.1 Pro row comes from a direct HTML parse of the same page.
- Free tier data use: "Content **used to improve our products**". On the paid tier, content is "**not** used" — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

**DeepSeek (official, no page date; © 2026)** — [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
- **deepseek-flash (DeepSeek-V4.1-Flash)**: input cache-miss **$0.15–$0.30**, cache-hit $0.003–$0.006, output **$0.60–$1.20**.
- **deepseek-v4-pro (V4-Pro-0813)**: input $0.66–$1.32, cache-hit $0.022–$0.044, output $1.98–$3.96.
- Both have 1M context and 384K max output.
- "**Off-peak rates are half of the peak rates.** Peak hours are 01:00–04:00 and 06:00–10:00 UTC, Monday through Friday."

**Mistral**
- **Mistral Small 4 (`mistral-small-2603`)**: $0.15 input / $0.60 output, cache read $0.015 — [OpenRouter listing](https://openrouter.ai/mistralai/mistral-small-2603); matches [LiteLLM map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
- LiteLLM also lists Mistral Medium 3.5 at $1.50/$7.50 and Mistral Large 3 at $0.50/$1.50 — [LiteLLM map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json). Not confirmed on Mistral's own page.

**Groq / Together (open-weight models)**
- Groq's official pricing page **could not be extracted** (the fetch returned only marketing text) — [groq.com/pricing](https://groq.com/pricing)
- LiteLLM lists Groq **gpt-oss-20b at $0.075/$0.30** (cached $0.0375), **gpt-oss-120b at $0.15/$0.60**, and qwen3.8-27b at $0.80/$4.00 — [LiteLLM map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
- Groq's free-plan model list (official) has **no Llama chat models**. It lists gpt-oss-120b/20b, gpt-oss-safeguard-20b, qwen3.8-27b, Llama Prompt Guard and Whisper — [Groq rate limits](https://console.groq.com/docs/rate-limits). A third-party headline also says "Llama Is Gone" — [klymentiev.com](https://klymentiev.com/blog/groq-pricing)
- Together AI (from LiteLLM, not verified on Together's page) — [LiteLLM map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json):
  - Llama-3.3-70B-Instruct-Turbo $1.04/$1.04;
  - Meta-Llama-3.1-8B-Instruct-Turbo $0.18/$0.18;
  - Llama-4-Scout-17B-16E $0.18/$0.59.

**Embeddings**
- OpenAI **text-embedding-3-small $0.02**, **text-embedding-3-large $0.13** — [OpenAI pricing](https://developers.openai.com/api/docs/pricing). Batch prices are $0.01 / $0.065 per [LiteLLM](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).
- **Gemini Embedding 2**: text **$0.20** (Batch $0.10). A free tier is available — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- From LiteLLM, not verified on vendor pages — [LiteLLM map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json):
  - gemini-embedding-001 $0.15;
  - Voyage 3.5 $0.06 and 3.5-lite $0.02;
  - mistral-embed $0.10.

**Free tiers usable by students**
- **Gemini API**:
  - Free tier exists for most Flash/Flash-Lite/2.5 models and Embedding 2, but **not for 3.1 Pro Preview**.
  - Google AI Studio is "free of charge" — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
  - The official rate-limits page (dated 2 Sep 2026) **does not publish free-tier numbers**. It directs users to the AI Studio dashboard and notes limits are "per project, not per API key", RPD resets at midnight Pacific, and "Specified rate limits are not guaranteed" — [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
  - Grounding on 2.5 Flash-Lite: "Free of charge, up to 500 RPD" — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- **Groq free plan** (official): gpt-oss-120b / gpt-oss-20b / qwen3.8-27b at **30 RPM, 1K RPD, 8K TPM, 200K TPD**. Limits apply "at the organization level" — [Groq rate limits](https://console.groq.com/docs/rate-limits)
- **Anthropic**: small free credits only — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **OpenRouter Auto Router**: no extra fee over the selected model's price — [OpenRouter docs](https://openrouter.ai/docs/guides/routing/auto-model-selection)

### Inferences
- **Per-request cost** at 1,500 input / 400 output tokens (my arithmetic from list prices, ignoring tokenizer differences):

| Model | $ per request | $ per 1k requests |
|---|---|---|
| Claude Opus 5.5 | $0.0140 | $14.0 |
| Gemini 3.1 Pro | $0.0078 | $7.80 |
| Claude Sonnet 5.5 / GPT-6.1-sol | $0.0070 | $7.00 |
| Claude Haiku 4.5 | $0.0035 | $3.50 |
| Gemini 3.1 Flash-Lite | $0.00098 | $0.98 |
| DeepSeek flash (peak) | $0.00093 | $0.93 |
| GPT-5.4-nano | $0.00080 | $0.80 |
| GPT-6-luna | $0.00035 | $0.35 |
| Gemini 2.5 Flash-Lite | $0.00031 | $0.31 |
| GPT-5-nano | $0.00024 | $0.24 |
| Groq gpt-oss-20b | $0.00023 | $0.23 |

- A good demo pairing for students within the budget:
  - **strong = Claude Sonnet 5.5 or GPT-6.1-sol**;
  - **cheap = GPT-6-luna / GPT-5-nano / Gemini 2.5 Flash-Lite (free tier) / Groq gpt-oss-20b (free tier)**;
  - **judge = a third vendor's frontier model**.
- Free-tier traffic is **not representative for cost reporting**, since it is $0 and the data may be used for training. Compute "list-price-equivalent" cost for reporting, and don't send sensitive data to free tiers.

### Gaps
- Groq's official per-token prices were not retrievable, so the LiteLLM figures are used. Together AI and Mistral prices were not checked on vendor pages.
- Exact Gemini free-tier RPM/RPD per model is not published on the official docs page. Third-party guides give inconsistent figures (roughly 500–1,500 RPD for Flash-Lite) and were not relied on.
- No newer Claude Haiku (e.g. a Haiku 5) appears on Anthropic's pricing page, so Haiku 4.5 is the cheapest current Claude model as of 2026-10-03.
- OpenAI's full Batch/Flex/Fast tables for the non-flagship (GPT-5.x/4.x) models were not fully extracted. LiteLLM shows Batch at 50% of Standard for these.

---

## Q5. Provider-side cost levers (caching, batch, flex/priority) and how a cost layer should account for them

### Takeaway
Provider levers often beat routing on savings:
- **Prompt caching** cuts repeated-prefix input cost by **90%** (Anthropic 0.1x read; OpenAI GPT-5.x/6.x 0.1x, and 0.05x on GPT-6.1-sol; Gemini cache price = 10% of input).
- **Batch** is **50% off** on all three major providers.
- **Flex** is priced at batch rates on OpenAI and Gemini.
- **Priority/Fast** costs about 1.8–2x more.

CostGuard must price each request from the actual `usage` breakdown (uncached, cache-write, cache-read, output), at the tier actually used. It must also attribute savings separately to routing vs caching vs batch. Downshifting can **break cache warmth or fall below a model's minimum cacheable length**, which erodes the nominal savings.

### Cited Findings

**Anthropic prompt caching**
- Multipliers: **5-minute cache write 1.25x**, **1-hour cache write 2x**, **cache read 0.1x** (0.05x on Opus 5.5, 0.025x on Fable 5.1 / Mythos 5.1). Caching "pays off after one cache read for the 5-minute duration … or after two cache reads for the 1-hour duration". "These multipliers **stack** with other pricing modifiers, including the Batch API discount and data residency" — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Minimum cacheable length** by model — [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
  - **512 tokens**: Opus 5.5, Opus 5, Sonnet 5.5, Fable 5/5.1;
  - **1,024**: Opus 4.8, Sonnet 5, Sonnet 4.6/4.5;
  - **2,048**: Opus 4.7, Haiku 3.5;
  - **4,096**: Opus 4.6/4.5 and **Haiku 4.5**.
  - "Any requests to cache fewer than this number of tokens will be processed without caching, and **no error is returned**."
- **TTL**: 5 min by default, or 1h at 2x. It is "refreshed for no additional cost each time the cached content is used". Lifetime is measured from the start of the request, so long streaming responses eat into it — [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- Structure and isolation — [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
  - **Up to 4 breakpoints** per request.
  - Cache hierarchy is tools → system → messages; changing tool definitions invalidates everything.
  - Caches are isolated per organisation and **per workspace**.
  - Hits require "100% identical prompt segments".
- **Automatic caching**: a top-level `cache_control` makes the cache point move forward as conversations grow — [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Usage accounting**: `total_input_tokens = cache_read_input_tokens + cache_creation_input_tokens + input_tokens`, where "`input_tokens` … represents only the tokens that come **after the last cache breakpoint**" — [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **Other Anthropic modifiers** — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing):
  - Batch is **50% off** input and output.
  - **Fast mode** (Opus 5.5 at $8/$40, i.e. 2x) is not available with Batch.
  - `inference_geo: "us"` costs **1.1x** on Claude 4.6+.
  - Regional/multi-region endpoints on Bedrock and Google Cloud carry a **10% premium**.

**OpenAI prompt caching / tiers**
- Caching is "**enabled by default**" — [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching):
  - **GPT-5.6 and later** need **1,024 visible input tokens**;
  - for earlier models, cached-token reporting "rounds down to a multiple of 128";
  - GPT-5.6+ cached reads are **0.1x** uncached input, or **0.05x for GPT-6.1 Sol**;
  - "Cache writes cost **1.25×** the standard, uncached input-token rate" on GPT-5.6+.
- Retention is in-memory "around 5 to 10 minutes", with an extended-retention option. `prompt_cache_key` gives per-customer cache accounting. Usage field: `usage.input_tokens_details.cached_tokens` — [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- Cached-input discounts on older models (from the price table) — [OpenAI pricing](https://developers.openai.com/api/docs/pricing):
  - **90%** off on the GPT-5 family;
  - **75%** off on GPT-4.1, o3 and o4-mini;
  - **50%** off on GPT-4o and 4o-mini.
- **Batch**: "50% cost discount", 24h window, separate rate limits — [OpenAI Batch](https://developers.openai.com/api/docs/guides/batch)
- **Flex** (`service_tier="flex"`) is priced "at Batch API rates". It may return "**429 Resource Unavailable** … You will not be charged when this occurs". Its default SDK timeout is 10 min and it is in beta with limited models — [OpenAI Flex](https://developers.openai.com/api/docs/guides/flex-processing)
- **Fast mode** (formerly Priority, renamed 30 Jul 2026) is 2x Standard on gpt-6.1-sol. **Data-residency endpoints carry a +10% uplift** for models released on or after 5 Mar 2026 — [OpenAI pricing](https://developers.openai.com/api/docs/pricing)

**Gemini caching / tiers**
- **Implicit caching** is "enabled by default for all Gemini 2.5 and newer models" — [Gemini caching](https://ai.google.dev/gemini-api/docs/caching) (page dated 2026-09-02):
  - minimum is **4,096 tokens** on Gemini 3.5–3.8 Flash and 3.1 Pro Preview, and **2,048** on 2.5 Flash/Pro;
  - "We automatically pass on cost savings if your request hits caches";
  - tip: "put large and common contents at the beginning of your prompt".
- Cache price is **10% of input**: e.g. 2.5 Flash-Lite $0.01 vs $0.10, 3.8 Flash $0.075 vs $0.75. **Explicit caching** adds **hourly storage**: $1.00/MTok/hr on 2.5 Flash-Lite, $4.50/MTok/hr on 2.5 Pro and 3.1 Pro — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- The Interactions API "only supports implicit caching" — [Gemini caching](https://ai.google.dev/gemini-api/docs/caching)
- **Batch is 50% off**, **Flex is priced the same as Batch**, and **Priority is 1.8x** (e.g. 2.5 Flash-Lite $0.18 vs $0.10) — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

**DeepSeek**
- Automatic cache-hit pricing is about 50x cheaper than a miss ($0.003–0.006 vs $0.15–0.30 on flash), and **off-peak is 50% off** — [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)

### Inferences
- **Cost-per-request formula CostGuard should implement** (per provider, per tier):

  `cost = p_in·uncached_in + p_cw·cache_write_in + p_cr·cache_read_in + p_out·(output + reasoning/thinking) + per-call tool fees`

  Then apply the tier multiplier (batch/flex 0.5x, priority/fast ~1.8–2x), region multiplier (1.1x) and long-context tier (Gemini Pro >200k, OpenAI >272k).
- **Savings attribution** should be reported separately so the router isn't credited for provider discounts:
  - **(1) routing savings** = cost(strong, same caching/tier) − cost(actual model);
  - **(2) caching savings** = cost at uncached prices − actual;
  - **(3) tier savings** (batch/flex).
  - The counterfactual for routing must assume the strong model would also have had its cache hits.
- **Downshift can break caching, a concrete gotcha**:
  - Take a 3,000-token shared system prompt. It is cacheable on Sonnet 5.5 (min 512) but **not on Haiku 4.5 (min 4,096)**.
  - Per request, the prefix costs about $0.0006 on Sonnet 5.5 (cache read at $0.20/MTok) vs $0.003 on Haiku 4.5 (uncached at $1/MTok), so it is **5x more expensive on the "cheaper" model**.
  - Add a 500-token user turn and a 400-token output, and also account for Sonnet 5.5's newer tokenizer (~30% more tokens, assuming Sonnet 5.5 counts as a "Claude 4.7 and later" model):
    - Sonnet 5.5: about $0.0073/request (3,900 × $0.2 + 650 × $2 + 520 × $10, per MTok);
    - Haiku 4.5: about $0.0055/request (3,500 × $1 + 400 × $5, per MTok).
  - That is a real saving of only about **25%**, vs the nominal 50% from list prices.
  - Switching models per request also means each model keeps its own cache. Low-traffic routes on a model may never stay warm within a 5-minute TTL.
- **Caching example**: a 3k-token system prompt over 100 requests within the TTL on Sonnet 5.5 costs about $0.60 uncached vs about $0.067 cached, i.e. **about 89% saved on that prefix**. This is often a bigger and safer win than downshifting, and needs no quality gate.
- **Batch and Flex** are ideal for CostGuard's own offline eval runs, and for any client workload that is not latency-sensitive.

### Gaps
- Whether OpenAI caching discounts apply on top of Batch was not stated in the Batch guide. The Flex guide says Flex gets "Batch API rates with additional prompt caching discounts". The Anthropic docs explicitly say caching stacks with Batch.
- OpenAI's exact extended-retention (24h) eligibility and pricing per model were not clearly extracted (the summary was garbled).
- Gemini explicit-cache minimum token count and default TTL were not on the fetched caching page (that page now focuses on the Interactions API).
- Whether Anthropic caches are model-specific, so that switching models forfeits the cache, is implied by the "100% identical" requirement and standard behaviour but was not explicitly quoted from the docs this session.

---

## Q6. How do you compute cost per request correctly and report savings honestly?

### Takeaway
Use the provider's **returned `usage` object** as ground truth, priced from a **dated, versioned price table**. Use token-counting tools (tiktoken, Anthropic `count_tokens`, Gemini `countTokens`) only for *pre-request* estimates and routing decisions.

Honest reporting requires:
- the same baseline conditions (caching, tier, region);
- actual output lengths per model, including thinking tokens;
- tokenizer differences;
- all overheads (router, verifier, judge, escalations, retries);
- a stated quality level with confidence intervals.

### Cited Findings
- **Anthropic `count_tokens`**:
  - It is "**free to use**", rate-limited per tier (Start 5,000 / Build 10,000 / Scale 20,000 RPM, separate from message limits).
  - It accepts system, tools, images and PDFs (base64).
  - It errors on server tools, the MCP connector and URL/file sources.
  - "The token count is an **estimate**… might differ by a small amount." It may include system-added tokens that are **not billed**.
  - It does **not** use caching logic.
  - Source: [Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
- **Tokenizer changes within one vendor**: Claude Opus 4.7+, Fable 5/5.1 and Mythos use a tokenizer producing "roughly 30 percent higher" counts. "Don't reuse token counts measured on the older model to estimate costs." Recount per target model — [Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting); [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Hidden input overhead from tools on Anthropic**: the tool-use system prompt adds **286–675 tokens** depending on model (e.g. Opus 5.5 286; Opus 4.7 675; Haiku 4.5 496 for `auto`/`none`). Tool definitions, `tool_use` and `tool_result` blocks are all billed as input — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Anthropic usage fields**: `cache_creation_input_tokens`, `cache_read_input_tokens`, `input_tokens`. Here `input_tokens` excludes cached portions — [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
- **OpenAI usage**: `usage.input_tokens_details.cached_tokens` — [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- **Gemini `countTokens`** returns "the total number of tokens in the input only". System instructions and tools count as input. Usage reports `total_input_tokens`, `total_output_tokens`, **`total_thought_tokens`**, `total_cached_tokens`, `total_tool_use_tokens` and `total_tokens`. The rule of thumb is "a token is equivalent to about **4 characters**. 100 tokens is equal to about 60-80 English words" — [Gemini tokens](https://ai.google.dev/gemini-api/docs/tokens)
- Gemini output prices are listed as "**Output price (including thinking tokens)**", so reasoning is billed at the output rate — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Anthropic's rule of thumb: "1 token is approximately 4 characters or 0.75 words in English" — [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- **Machine-readable price table**: LiteLLM's `model_prices_and_context_window.json` (about 3 MB, downloaded 2026-10-03). It has fields such as `input_cost_per_token`, `output_cost_per_token`, `cache_read_input_token_cost`, `cache_creation_input_token_cost`, `input_cost_per_token_batches`, `*_flex`, `*_priority` and `*_above_200k_tokens` / `*_above_272k_tokens` — [LiteLLM JSON](https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json)
  - **It can be stale**: it still lists `deepseek/deepseek-chat` at $0.28/$0.42, whereas DeepSeek's official page now lists V4.1-Flash and V4-Pro with peak/off-peak pricing — [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
  - Its OpenAI, Anthropic and Gemini rows checked this session matched the official pages.
- **Time-varying prices to version** — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing); [OpenAI pricing](https://developers.openai.com/api/docs/pricing); [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing); [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing):
  - Gemini 3.6–3.8 Flash doubles on 2027-01-01;
  - GPT-5.6-sol promo runs "at least through November 21, 2026";
  - Sonnet 5's intro price became permanent;
  - DeepSeek has peak/off-peak pricing.
- **tiktoken** is OpenAI's open-source BPE tokenizer library for OpenAI models — [openai/tiktoken](https://github.com/openai/tiktoken) (repo not fetched this session; general reference).

### Inferences
- **Implementation checklist for CostGuard's cost meter**:
  1. Log raw `usage` per call (all cache and thinking fields), plus model ID, service tier, region and timestamp.
  2. Price with a pinned price-table version (LiteLLM JSON snapshot plus manual overrides from official pages, each with a `checked_on` date).
  3. Compute cost per *logical request* by summing all calls it triggered: router/classifier call, cheap attempt, verifier, escalated strong call, retries, and judge calls if sampled online.
  4. Store the counterfactual strong-model cost. For the strong model's token counts, either (a) measure them in shadow mode, or (b) estimate with that vendor's counter and the *observed* strong-model output-length distribution. Do not assume equal output lengths across models.
- **Honest-reporting rules for the demo/report**:
  - (a) state baseline, model pair, workload mix and quality level with its CI;
  - (b) separate routing vs caching vs batch savings;
  - (c) include all overheads and escalation waste;
  - (d) report the escalation rate and per-category savings, not just the aggregate;
  - (e) flag free-tier or promo pricing;
  - (f) note tokenizer differences (e.g. ~30% within Anthropic generations, plus cross-vendor differences);
  - (g) show sensitivity to price changes (e.g. Gemini's Jan 2027 doubling);
  - (h) avoid cherry-picking the best benchmark. RouteLLM's own spread (85% / 45% / 35%) is a good example to cite.
- For **pre-request routing on prompt length**, use local estimators: tiktoken for OpenAI models, and char/4 as a cross-vendor approximation. Use Anthropic's free `count_tokens` or Gemini `countTokens` only when exactness matters, since each adds a network round-trip.

### Gaps
- Which tiktoken encoding applies to GPT-5.x/GPT-6.x models, and whether tiktoken supports them, was not verified this session. The safe default is API `usage` for billing truth.
- Whether Gemini `countTokens` is free or rate-limited was not stated on the fetched page.
- OpenAI's reasoning-token field name and billing (reasoning tokens billed as output) was not re-verified this session for GPT-6.x.
- Cross-vendor tokenizer ratios (e.g. the same text on OpenAI vs Anthropic vs Gemini) were not measured. The team can measure them cheaply on its eval set using each provider's usage fields.
