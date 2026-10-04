# Prompt/Context Compression + Reranking & Truncation Policies for LLM Cost Reduction (state as of Oct 2026)

Scope: notes for "LLM CostGuard" components (1) LLMLingua-style compression with a measured quality delta and (2) reranking + truncation policies. Constraints: 5 days, $5–20 API budget, free tiers, CPU or free GPU preferred.
Source-quality legend: [paper] = primary arXiv/venue paper fetched this session; [repo/docs] = official repo or docs; [snippet] = search-result summary not opened in full (treat as lower confidence); [prior-knowledge] = well-known fact not re-fetched this session (verify before quoting in a final deliverable). Anything published before 2024 is marked (pre-2024).

---

## Q1. Main compression methods, how they work, drop-in usability, hardware needs

### Takeaway
For a closed-API "drop-in" layer, the realistic choices are text-in/text-out compressors: **LLMLingua-2** (355M XLM-RoBERTa or 110M mBERT token classifier; `pip install llmlingua`; fits on CPU or a 2 GB GPU) for task-agnostic compression, and **query-aware extractive pruning** (rerankers, LongLLMLingua's question-conditioned mode, or Provence) for RAG context. Soft-prompt/gist/AutoCompressor methods need white-box model weights and cannot be used with OpenAI/Anthropic/Gemini APIs.

### Cited Findings
**LLMLingua family (Microsoft) — the drop-in library**
- Install with `pip install llmlingua`. The repo supports LLMLingua (EMNLP 2023), LongLLMLingua (ACL 2024), LLMLingua-2 (ACL 2024 Findings) and SecurityLingua (a jailbreak defence via security-aware compression, CoLM 2025) — [microsoft/LLMLingua README, repo/docs](https://github.com/microsoft/LLMLingua)
- Default/listed compressor models are `microsoft/phi-2` for LLMLingua/LongLLMLingua, `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` and `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` for LLMLingua-2, `TheBloke/Llama-2-7b-Chat-GPTQ` as a quantized option, and `SecurityLingua/securitylingua-xlm-s2s` — [LLMLingua README](https://github.com/microsoft/LLMLingua)
- Key `compress_prompt()` parameters: `rate` (e.g. 0.33, 0.55), `target_token` (e.g. 200), `force_tokens` (e.g. `['\n', '?']`), `use_llmlingua2`, and the LongLLMLingua options `condition_in_question`, `reorder_context`, `dynamic_context_compression_ratio` and `rank_method` — [LLMLingua README](https://github.com/microsoft/LLMLingua)
- Structured compression: `<llmlingua>` tags with optional `compress` and `rate` attributes give section-level control, for example leaving the instructions uncompressed while compressing the documents — [LLMLingua README](https://github.com/microsoft/LLMLingua)
- Integrations: LangChain, LlamaIndex and Prompt Flow — [LLMLingua README](https://github.com/microsoft/LLMLingua)
- Microsoft hosts a public Hugging Face Space for LLMLingua-2 (`spaces/microsoft/llmlingua-2`), which shows it runs on HF Spaces infrastructure — [HF Space app.py, snippet](https://huggingface.co/spaces/microsoft/llmlingua-2/blob/main/app.py)
- An early GitHub issue is titled "LLMLingua doesn't work on CPU as device_map" (#79). It refers to the original 7B-LM variant; CPU support was a known rough edge — [GitHub issue #79, snippet/title only](https://github.com/microsoft/LLMLingua/issues/79)

**LLMLingua (original; arXiv 2310.05736, EMNLP 2023; pre-2024)**
- Coarse-to-fine compression with three modules: a **Budget Controller** that allocates the compression ratio across instruction, demonstrations and question; **Iterative Token-level Compression**, which drops low-information tokens using a small LM's perplexity; and **Distribution Alignment**, which instruction-tunes the small LM toward the target LLM. Small LMs tested: Alpaca-7B, GPT-2-Alpaca and LLaMA-7B. Target LLMs: GPT-3.5-Turbo-0301 and Claude-v1.3 — [LLMLingua paper](https://arxiv.org/html/2310.05736)

**LongLLMLingua (arXiv 2310.06839, v1 Oct 2023, ACL 2024)**
- Query-aware compression for long contexts. It targets three problems: cost, performance loss and position bias ("lost in the middle"). The README exposes question-conditioning, context reordering and dynamic per-document compression ratios — [LongLLMLingua abstract](https://arxiv.org/abs/2310.06839); [README params](https://github.com/microsoft/LLMLingua)

**LLMLingua-2 (arXiv 2403.12968, ACL 2024 Findings)**
- Compression is framed as **token classification** (keep/drop per token) by a bidirectional encoder. The encoder is trained on an extractive-compression dataset distilled from GPT-4 over MeetingBank (5,169 chunks; 10 epochs; Adam, LR 1e-5). Models are XLM-RoBERTa-large (355M) and mBERT (110M). Target LLMs evaluated: GPT-3.5-Turbo and Mistral-7B. It is **task-agnostic**: it does not condition on the question — [LLMLingua-2 paper](https://arxiv.org/html/2403.12968)
- Peak GPU memory is **2.1 GB** for LLMLingua-2, against 16.6 GB for LLMLingua and 26.5 GB for Selective-Context (V100-32G) — [LLMLingua-2 paper](https://arxiv.org/html/2403.12968)
- On CPU, a figure of about **689 ms per prompt** is reported. The source is a search summary, probably a 2026 arXiv paper on edge-device compression, and was not opened, so it is low-confidence — [snippet: arXiv 2606.20571](https://arxiv.org/pdf/2606.20571); `device_map="cpu"` is the documented way to run it on CPU — [snippet: Medium hands-on guide](https://deveshsurve.medium.com/prompt-compression-hands-on-guide-for-llm-lingua-2-2baac53800d6)

**Provence (Naver Labs Europe, arXiv 2501.16214, ICLR 2025) — reranker and pruner in one model**
- Treats context pruning as sequence labeling and **unifies pruning with reranking** in one DeBERTa-v3 cross-encoder, so pruning is "almost zero cost" in a RAG pipeline that already reranks. It decides dynamically how much to prune — [Provence HF blog](https://huggingface.co/blog/nadiinchi/provence); [arXiv 2501.16214](https://arxiv.org/pdf/2501.16214); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5e956fef0946dc1e39760f94b78045fe-Abstract-Conference.html)
- The model `naver/provence-reranker-debertav3-v1` has **430M params** and a **512-token context**, is **English only**, and is licensed **CC BY-NC-ND 4.0** (non-commercial). It is used via a `process(question, context)` call with `threshold` (0.1 is conservative, 0.5 gives more compression) and `always_select_title=True`. It was trained only on MS MARCO and Natural Questions. A multilingual variant, **XProvence**, is built on bge-reranker-v2-m3 — [HF model card](https://huggingface.co/naver/provence-reranker-debertav3-v1)

**RECOMP (arXiv 2310.04408, ICLR 2024; pre-2024 preprint)**
- Two trained compressors for retrieved documents. The **extractive** one selects the relevant sentences; the **abstractive** one writes a summary across documents. Both can return an empty string when the retrieved documents do not help ("selective augmentation") — [RECOMP](https://arxiv.org/html/2310.04408)

**Selective Context, gist tokens, AutoCompressors (all pre-2024) [prior-knowledge, not re-fetched]**
- Selective Context (Li et al. 2023, arXiv 2304.12102) prunes tokens, phrases or sentences with low **self-information** under a small causal LM. LLMLingua-2 measured it at 26.5 GB peak GPU memory — [Selective Context](https://arxiv.org/abs/2304.12102); [memory figure from LLMLingua-2](https://arxiv.org/html/2403.12968)
- Gist tokens (Mu et al. 2023, arXiv 2304.08467) fine-tune the LM, through attention masking, to compress a prompt into a few learned "gist" tokens. This needs **model-weight access** — [Gisting](https://arxiv.org/abs/2304.08467)
- AutoCompressors (Chevalier et al. 2023, arXiv 2305.14788) fine-tune OPT/Llama-2 to compress long context into summary vectors (soft prompts). This also needs white-box weights — [AutoCompressors](https://arxiv.org/abs/2305.14788)

**2024–2026 successors and signals (mostly title-level evidence)**
- Selection-p, a self-supervised task-agnostic compressor (arXiv 2410.11786) — [snippet](https://arxiv.org/pdf/2410.11786); PIS, importance sampling plus attention (arXiv 2504.16574) — [snippet](https://arxiv.org/pdf/2504.16574); a 2026 evaluation of LLMLingua-2 on diffusion LLMs (LLaDA, arXiv 2605.17932) — [snippet](https://arxiv.org/pdf/2605.17932). Taken together, these show LLMLingua-2 is still the standard reference baseline in 2026.
- 2026 "optical"/visual compression of retrieved text (RAGOCR, arXiv 2608.00765) — [snippet/title](https://arxiv.org/pdf/2608.00765)
- Kong's API gateway lists an "AI Prompt Compressor" plugin, a sign that gateway-level compression is now a product category — [Kong docs, title only](https://developer.konghq.com/plugins/ai-prompt-compressor/)

### Inferences
- **The best fit for CostGuard is LLMLingua-2 with `xlm-roberta-large` on CPU**, falling back to mBERT for speed. The original LLMLingua and LongLLMLingua need a 2.7B–7B causal LM (phi-2 or Llama-2-7B), so they effectively need a GPU (16.6 GB peak in the paper). That rules them out on a free CPU box, though they may fit a free Colab T4 with phi-2. Treat this as plausible but untested.
- For RAG specifically, **query-aware extractive selection** (a cross-encoder reranker plus sentence-level pruning) is easier to defend than task-agnostic token dropping. See the Jha et al. findings in Q2.
- Provence's CC BY-NC-ND licence is acceptable for a student or academic demo but not for a commercial product. Say this explicitly if the team pitches CostGuard as a product.
- Soft-prompt methods (gist tokens, AutoCompressors, xRAG and 500x-style compressors) should be listed as "related work, incompatible with closed APIs" and not built.

### Gaps
- No measured LLMLingua-2 CPU latency per 1k tokens on a known CPU was found from a primary source; the 689 ms figure is unverified. The team should measure it on their own box (half a day).
- I did not verify the exact free-tier specs (HF Spaces CPU basic, Colab T4) this session.
- I found no primary numbers for 2025–2026 successors such as Selection-p or PIS beyond their titles.

---

## Q2. Published numbers: compression ratios, quality retained, compressor latency, end-to-end savings

### Takeaway
The published sweet spot is **2x–5x for task-agnostic token compression**, with small losses on QA and reasoning and noticeable losses on summarization and LongBench. **Query-aware or extractive methods reach 4x–10x and sometimes improve accuracy** by removing distractors. The "20x" headline comes from GSM8K few-shot prompts, where demonstrations are highly redundant. It does not transfer to general context.

### Cited Findings
**LLMLingua (GPT-3.5-Turbo-0301 target; pre-2024)**
- GSM8K exact match (EM): **79.08 at 5x** (1-shot budget), against **78.85** for the full prompt; **77.41 at 14x**; **77.33 at 20x** — [LLMLingua paper](https://arxiv.org/html/2310.05736)
- BBH EM: **70.11 at 3x** vs 70.07 full; **61.60 at 5x**; **56.85 at 7x**. BBH degrades much faster than GSM8K — [LLMLingua paper](https://arxiv.org/html/2310.05736)
- ShareGPT: 27.36 BLEU and 89.52 BERTScore-F1 at 1.9x. Arxiv-March23: 23.15 BLEU and 90.33 BERTScore-F1 at 4x — [LLMLingua paper](https://arxiv.org/html/2310.05736)
- End-to-end latency is **1.7x–5.7x** faster at 2x–10x compression (V100-32G). At 20x on GSM8K it scores 33.10 EM points higher than Selective-Context — [LLMLingua paper](https://arxiv.org/html/2310.05736)
- The README headline is "up to 20x compression with minimal performance loss" — [README](https://github.com/microsoft/LLMLingua)

**LongLLMLingua (GPT-3.5-Turbo)**
- On NaturalQuestions, performance improves by **up to 21.4% with ~4x fewer tokens**. It gives a **94.0% cost reduction on LooGLE** and **1.4x–2.6x** end-to-end speedup when compressing ~10k-token prompts at 2x–6x — [LongLLMLingua abstract](https://arxiv.org/abs/2310.06839)

**LLMLingua-2 (GPT-3.5-Turbo target unless noted)**
- MeetingBank, in-domain, at **3.1x** (970 vs 3,003 tokens): QA EM **86.92 vs 87.75** for the original (−0.8 pts); summary BLEU **17.37 vs 22.34** (−22% relative) — [LLMLingua-2 paper](https://arxiv.org/html/2403.12968)
- Out of domain with a 2,000-token budget: LongBench average **39.1 vs 44.0** for the original at ~5x (−11% relative); ZeroSCROLLS **33.4 vs 34.7** (−3.7% relative) — [LLMLingua-2 paper](https://arxiv.org/html/2403.12968)
- The paper reports GSM8K (1-shot budget) at **79.08 EM at 5x** vs 78.85 full, and BBH at **70.02 EM at 3x** vs 70.07 full. Note: the GSM8K figure is identical to the original LLMLingua's, so the fetch tool may have mixed up two table rows. Check Table 3 of the PDF before quoting it — [LLMLingua-2 paper](https://arxiv.org/html/2403.12968)
- Compressor latency is **0.4–0.5 s** for LLMLingua-2 vs 1.5–2.9 s for LLMLingua on a V100. It is "3x–6x faster than existing prompt compression methods", with **1.6x–2.9x end-to-end speedup** at 2x–5x compression — [LLMLingua-2 paper](https://arxiv.org/html/2403.12968)
- A secondary summary claims "95–98% accuracy retention". This is an aggregator figure, not a number from the paper — [snippet: Medium, Kuldeep Paul](https://medium.com/@kuldeep.paul08/prompt-compression-techniques-reducing-context-window-costs-while-improving-llm-performance-afec1e8f1003)

**Independent comparisons (2024–2025)**
- Jha et al., "Characterizing Prompt Compression Methods for Long Context Inference" (arXiv 2407.08892, ES-FoMo @ ICML 2024): **extractive compression reaches up to 10x with minimal accuracy degradation and "often outperforms all the other approaches"**. Token-pruning methods (the LLMLingua family) "often lag behind extractive compression"; summarization-based compression gave only marginal gains — [Jha et al. 2024](https://arxiv.org/abs/2407.08892)
- Reranker-based extractive compression reportedly gave **+7.89 F1 on 2WikiMultihopQA at 4.5x**. This comes from a search summary attributing it to a 2024 study, very likely Jha et al.; verify in the PDF — [snippet](https://medium.com/@kuldeep.paul08/prompt-compression-techniques-reducing-context-window-costs-while-improving-llm-performance-afec1e8f1003)
- Zhang et al., "An Empirical Study on Prompt Compression for LLMs" (arXiv 2505.00019, ICLR 2025 Building Trust workshop): six methods across 13 datasets (news, science, commonsense QA, math, long-context QA, VQA). Compression has **a bigger impact in long-context settings**, and **"moderate compression even enhances LLM performance" on LongBench**. The study also analysed hallucination and word omission, and released code — [Zhang et al. 2025](https://arxiv.org/abs/2505.00019)
- RECOMP (pre-2024): on QA, the best compressor keeps **5–10% of the tokens with <10% relative performance drop**. On language modelling it reaches 25% of tokens with minimal drop; oracle compressors reach 6% — [RECOMP](https://arxiv.org/html/2310.04408)
- Provence: "little-to-no drop in performance" across 7 datasets and sits on the efficiency Pareto front. The model card gives no exact pruning percentages — [HF model card](https://huggingface.co/naver/provence-reranker-debertav3-v1)

### Inferences
- **Savings formula for the pitch.** Let `input_share` be the fraction of the bill from input tokens, `compressible_share` the fraction of input tokens that are compressible context, and `r` the compression ratio. Then `bill_saving ≈ input_share × compressible_share × (1 − 1/r)`. For example, an input-heavy RAG call (input = 80% of cost, context = 70% of input) at 3x saves ≈ 0.8 × 0.7 × 0.667 ≈ **37%** of the bill. The 94% LooGLE figure is an upper bound for very long-document workloads.
- The compressor costs about 0.4–0.7 s, so on short prompts (<1k tokens) it can add more latency than it saves. Gate it on prompt length, for example compress only when compressible context exceeds 1–2k tokens.
- A defensible claim: "token-level compression at ≤3x keeps QA/reasoning within ~1–4% of baseline in the published results; summarization and generative tasks lose much more (−22% BLEU at 3.1x)."

### Gaps
- I found no published 2025–2026 numbers for LLMLingua-2 against current frontier API models (GPT-5-class, Claude 4-class, Gemini 2.5/3). All headline numbers use GPT-3.5-Turbo or Mistral-7B. Stronger models may be more robust to compression, or more sensitive to dropped tokens; this is unknown, so the team's own eval is the evidence.
- The "−20%+ in the middle" figure from Lost-in-the-Middle could not be verified from the abstract (see Q5).

---

## Q3. How to evaluate the "quality delta" rigorously and choose a compression rate

### Takeaway
Run a **paired** evaluation: the same items and the same target model, with baseline vs compressed prompts. Measure a **task metric** (EM/F1/accuracy) where gold answers exist, plus an **LLM-judge** (pairwise or reference-based correctness) and **faithfulness** for RAG. Report the delta with a bootstrap/McNemar CI. Choose the largest rate whose **CI lower bound** stays within the allowed margin (2–5%). Plan for about 200–500 paired items per condition, which fits the $5–20 budget with a mini-tier model.

### Cited Findings
- RAGAS (current docs) metric families: **Faithfulness, Response Relevancy, Context Precision, Context Recall, Context Entities Recall, Noise Sensitivity** (RAG); **Factual Correctness, Semantic Similarity, BLEU, ROUGE, CHRF, Exact Match, String Presence** (comparison against a reference); NVIDIA **Answer Accuracy, Context Relevance, Response Groundedness**; and general **Aspect Critic / Rubrics-based scoring**. LLM-based metrics "might use one or more LLM calls" — [RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- The published evals use these task metrics: EM on GSM8K, BBH and MeetingBank-QA; BLEU and BERTScore for summaries; LongBench and ZeroSCROLLS averages — [LLMLingua](https://arxiv.org/html/2310.05736); [LLMLingua-2](https://arxiv.org/html/2403.12968)
- Zhang et al. 2025 measured hallucination and word omission in addition to accuracy. Compression changes *how* models fail, not only how often — [arXiv 2505.00019](https://arxiv.org/abs/2505.00019)
- LLM-as-judge (Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench", arXiv 2306.05685, pre-2024) [prior-knowledge]: strong judges reach **>80% agreement with human preferences**, about the same as human–human agreement. Known biases are **position, verbosity and self-enhancement**, so swap A/B order and use a different model family as the judge where possible — [Zheng et al. 2023](https://arxiv.org/abs/2306.05685)

### Inferences (recommended protocol for CostGuard)
1. **Datasets (free).** Use a slice of **NaturalQuestions/HotpotQA or a LongBench subset** for RAG QA (gold answers → EM/F1), **GSM8K** for few-shot reasoning (gold answers → EM), and **MeetingBank** for summarization, the domain LLMLingua-2 was trained on. Add one "hostile" set (JSON, code, number-heavy tables) to show where compression fails.
2. **Conditions.** Baseline; LLMLingua-2 at rate ∈ {0.7, 0.5, 0.33, 0.2} (≈1.4x, 2x, 3x, 5x); reranker top-k / threshold variants; and the combination. Log input tokens, output tokens, $ cost, compressor latency and end-to-end latency for each.
3. **Metrics.** (a) Task metric against gold. (b) An LLM judge on *answer correctness vs reference*: binary or 1–5, which is cheaper and more stable than open pairwise judging. (c) Pairwise "baseline vs compressed: which is better or tie", run in both orders to cancel position bias. (d) RAGAS Faithfulness for RAG, since compression can strip the evidence an answer cites. Report **quality delta = metric(compressed) − metric(baseline)**, both absolute and relative.
4. **Statistics.** Use a paired design: each item is answered under both conditions. With binary correctness, use McNemar or a paired bootstrap (10k resamples) for the 95% CI of the delta. Approximate sizes for 80% power at α = 0.05, paired, assuming discordant-pair rates of 6–8%:
   - To detect a **5-point drop**: about **250 items**.
   - To detect a **3-point drop**: about **500 items**.
   - With **100 items**, the CI half-width on the delta is about **±5 points**. That is enough for a 5% tolerance but not for 2%.
   - An unpaired design needs about 3,000 items per arm for a 3-point gap, so pairing matters.
5. **Choosing the rate (non-inferiority framing).** For each rate, compute the delta and its 95% CI. Pick the most aggressive rate whose **lower CI bound ≥ −margin** (e.g. −3 pts). Plot the curve of tokens saved (x) against quality delta (y) with CIs; this is the headline chart. You can also choose the rate per task type: QA tolerates more, summarization less.
6. **Budget sanity.** Cost per condition ≈ items × (prompt tokens × input price + output tokens × output price), plus judge calls. With a small or "mini" model as the target and judge, 500 items × ~6 conditions × ~2–3k tokens is roughly 6–9M input tokens. The team must check current mini-model prices; at sub-$0.50/M input this fits in $5–20. Cap max_tokens on the judge and use short rubric outputs (a score only).
7. **Hygiene.** Use temperature 0 (or several seeds), a fixed model snapshot, a cache of baseline answers, and blinding (the judge does not know which answer is compressed). Spot-check 30–50 items by hand to validate the judge.

### Gaps
- I did not retrieve RAGAS's exact algorithms (claim decomposition for Faithfulness, F1 over claims for Factual Correctness) or its ground-truth requirements this session. Check the per-metric doc pages.
- I found no published guidance specific to compression on minimum eval-set size; the numbers above are my power calculations, not cited results.

---

## Q4. When compression hurts (code, numbers, JSON, instructions, multilingual, prompt caching)

### Takeaway
Token-dropping compressors are trained on natural-language prose (LLMLingua-2: MeetingBank transcripts). Expect damage on **exact-token content**: code, numbers, IDs, JSON/tables and instructions. Protect these with `<llmlingua compress=False>` sections or `force_tokens`, or route them around the compressor. The cost interaction to design around is **provider prefix caching**: it requires byte-identical prefixes and already gives 50–90% off cached input, so compress only the **variable suffix** (retrieved docs and history), never the static system prompt.

### Cited Findings
- LLMLingua-2's training data is GPT-4-distilled compressions of **MeetingBank** meeting transcripts only (5,169 chunks) — [LLMLingua-2](https://arxiv.org/html/2403.12968)
- Generative or summary output degrades more than QA under compression: MeetingBank summary BLEU **22.34 → 17.37** at 3.1x, while QA EM went 87.75 → 86.92 — [LLMLingua-2](https://arxiv.org/html/2403.12968)
- Reasoning benchmarks diverge: BBH falls from 70.07 to **56.85 at 7x**, while GSM8K holds 77.33 at 20x. Tolerance is highly task-dependent — [LLMLingua](https://arxiv.org/html/2310.05736)
- The library provides **`force_tokens`** (e.g. `['\n', '?']`) to keep specific tokens, and **`<llmlingua compress=False>`** tags to exempt sections, the intended mechanism for system prompts and instructions — [LLMLingua README](https://github.com/microsoft/LLMLingua)
- Microsoft shipped **SecurityLingua** (CoLM 2025) for security-aware compression. This shows that compression interacts with safety-relevant prompt content — [README](https://github.com/microsoft/LLMLingua)
- Multilingual: LLMLingua-2 ships an XLM-RoBERTa and a multilingual-BERT checkpoint, but both are trained on English MeetingBank — [README](https://github.com/microsoft/LLMLingua); [paper](https://arxiv.org/html/2403.12968). Provence is **English only**; XProvence (based on bge-reranker-v2-m3) is the multilingual variant — [Provence model card](https://huggingface.co/naver/provence-reranker-debertav3-v1)
- Prompt caching:
  - **OpenAI** caches automatically for prompts **≥1,024 tokens** on an **exact prefix match**: "any inputs that vary by a single character will not" hit. The cached-input discount is 50%, and up to 90% on some newer models — [snippet: Humanloop](https://humanloop.com/blog/prompt-caching); [DEV: "90% off cached input tokens"](https://dev.to/rikuq/openai-prompt-caching-explained-automatic-free-to-enable-90-off-cached-input-tokens-7bn); [OpenAI Cookbook: Prompt Caching 201](https://developers.openai.com/cookbook/examples/prompt_caching_201)
  - **Anthropic** needs explicit `cache_control` breakpoints, has a minimum of ~1,024 tokens for most models, and charges about **10% of base input price** for cache reads — [snippet: tokonomics](https://tokonomics.ca/blog/prompt-caching-guide-openai-anthropic); [ngrok explainer](https://ngrok.com/blog/prompt-caching)
  - Keep repeated blocks **byte-identical** to get cached pricing — [snippet: glukhov.org](https://www.glukhov.org/llm-performance/cost-effective-llm-applications/)

### Inferences
- **Compressing a cached prefix can raise cost.** A 3k-token system prompt cached at 10% of the input price costs the equivalent of 300 tokens. Compressing it 2x to 1.5k uncached tokens costs 1,500 token-equivalents, **5x more**, and risks breaking instructions. Rule: [static system prompt + tools + few-shot examples] go first and stay **uncompressed and cached**; [retrieved context + history] are compressed or reranked; [user question] comes last and is uncompressed.
- **Determinism.** LLMLingua-2 is a deterministic classifier given the same input and settings, so the same document compresses the same way. Question-aware methods (LongLLMLingua with `condition_in_question`, reranker selection) give different output per query by design. Place their output *after* the cache breakpoint.
- **Content-type router (cheap and defensible).** Before compressing, detect code fences, JSON (does `json.loads` succeed?), markdown tables, number density (e.g. >15% numeric tokens), and non-Latin scripts. Bypass or protect these. Unit test: values in JSON, code identifiers and numbers in the original must still appear in the compressed text, or the item is flagged.
- Instructions, output-format specs, schemas, tool definitions and safety text should always be `compress=False`. Losing one "not" or "only" changes behaviour, and these are usually the cached prefix anyway.
- Multi-turn chat: re-compressing history every turn changes earlier turns and breaks the cache from that point on. Compress or summarize history in **append-only blocks** (freeze a summary and stop rewriting it) so the prefix stays stable.

### Gaps
- I found no quantitative published study (fetched this session) of LLMLingua-2 on code, JSON or tabular inputs. LongBench includes code tasks (LCC, RepoBench-P), but per-task breakdowns were not retrieved. The team's "hostile set" eval would add something new here.
- I did not verify exact current cache minimums and discounts for every model from official OpenAI/Anthropic/Gemini docs; the figures above come from secondary blogs plus the OpenAI cookbook link. Confirm against official pricing pages before the demo.

---

## Q5. Reranking and truncation for RAG context (rerankers, top-k/dynamic-k, token budgets, history, ordering)

### Takeaway
A small CPU cross-encoder (**ms-marco-MiniLM-L-6-v2**, 22M params) reranking about 20–50 retrieved chunks, followed by **dynamic-k** (score threshold plus a token budget) and **edge-placed ordering**, is the cheapest robust lever. It is free, runs in about 0.1–2 s on CPU, and published extractive-selection results show 4x–10x context reduction with flat or improved accuracy. bge-reranker-v2-m3 is stronger and multilingual but about 15x slower on CPU. Cohere Rerank costs about $2–2.50 per 1k queries if a hosted option is wanted.

### Cited Findings
**Rerankers: cost and latency**
- **Cohere Rerank 4 Pro is $2.50 per 1k searches; Rerank 4 Fast is $2.00 per 1k.** A "search unit" is one query with up to 100 documents. Rerank 3.5 is about $2.00 per 1k on AWS Bedrock — [snippet: eesel](https://www.eesel.ai/blog/cohere-ai-pricing); [OpenRouter Rerank 4 Pro](https://openrouter.ai/cohere/rerank-4-pro); [OpenRouter Rerank 4 Fast](https://openrouter.ai/cohere/rerank-4-fast)
- On CPU, in a community benchmark (hardware unspecified): **bge-reranker-v2-m3 (560M) took 32.6 s for 40 candidates; ms-marco-MiniLM-L-6-v2 (22M) took 2.24 s**, ~14.5x faster with "comparable" retrieval quality on that project's data. Another report puts MiniLM at **~100 ms per 30 candidates vs ~255 ms for bge-reranker-base**. One project's PR is titled "Swap reranker to ms-marco-MiniLM-L-6-v2 (~160s→~11s/query, quality held)" — [snippet: GitHub PR #383](https://github.com/rusty-chris/lets-talk-climate-emergency/pull/383); [snippet: rag-with-receipts PR #5](https://github.com/lofeodo/rag-with-receipts/pull/5). These are anecdotal, small-project numbers.
- **Provence** merges reranking with sentence-level pruning, so pruning adds almost no cost on top of reranking. Its `threshold` (0.1 vs 0.5) is a ready-made dynamic pruning knob; 430M params and 512-token context — [HF model card](https://huggingface.co/naver/provence-reranker-debertav3-v1); [HF blog](https://huggingface.co/blog/nadiinchi/provence)

**Selection and compression results relevant to top-k**
- Extractive (reranker-style) compression reaches **up to 10x with minimal accuracy loss** and often beats token pruning (Jha et al. 2024) — [arXiv 2407.08892](https://arxiv.org/abs/2407.08892)
- RECOMP's selective augmentation returns *nothing* when the retrieved documents do not help, keeping **5–10% of tokens with <10% relative drop** on QA (pre-2024) — [RECOMP](https://arxiv.org/html/2310.04408)
- LongLLMLingua's question-aware reorder plus dynamic per-document ratios gave **+21.4% on NQ at ~4x fewer tokens** — [arXiv 2310.06839](https://arxiv.org/abs/2310.06839)

**Ordering ("lost in the middle", pre-2024)**
- Liu et al., "Lost in the Middle" (arXiv 2307.03172, TACL 2023): performance "is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades" when it is in the middle. Tasks: multi-document QA and key-value retrieval — [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)
- LongLLMLingua explicitly targets this position bias through `reorder_context` — [README](https://github.com/microsoft/LLMLingua); [abstract](https://arxiv.org/abs/2310.06839)

### Inferences (truncation policy design for CostGuard)
- **Pipeline:** retrieve top-N (20–50) → rerank with a cross-encoder → **dynamic k** → order → optional sentence pruning or LLMLingua-2 inside the kept chunks → assemble within a **hard token budget**.
- **Dynamic-k rules,** to combine and tune on a dev set:
  - (i) keep chunks with score ≥ τ, with τ chosen so recall of the gold-supporting chunk stays ≥95% on dev;
  - (ii) relative cutoff: keep while score ≥ α × top_score, or stop at the largest score gap ("elbow");
  - (iii) always keep a minimum of 1–2 chunks and a maximum of k_max;
  - (iv) stop when the cumulative-token budget B is reached, e.g. B = 1,500 tokens of context;
  - (v) if the top score is below τ_min, send no context at all (RECOMP-style) or answer "not found".
- **Ordering:** put the highest-scoring chunk first and the second-highest last (closest to the question), with weaker chunks in the middle. LangChain's `LongContextReorder` implements this pattern [prior-knowledge, not fetched].
- **Token-budget truncation:** truncate at sentence or chunk boundaries, never mid-sentence. Prefer dropping whole low-score chunks over shortening every chunk.
- **Conversation history:**
  - Keep the last N turns verbatim, for example within a token budget of the last 2–4k tokens.
  - Fold older turns into a running summary, generated once by a cheap model and then frozen so the cache prefix stays stable.
  - Pin facts the user explicitly stated, such as IDs and preferences, into a small "memory" block.
  - Measure the effect with the same paired eval on multi-turn transcripts.
- **Metric for this component:** tokens sent vs answer quality, with recall@k of gold evidence as a cheap proxy that needs no LLM calls. Then run end-to-end EM/F1 or an LLM judge on a 200–300 item subset.
- **Recommended choice for 5 days on free tiers:** use `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers on CPU as the default; offer `BAAI/bge-reranker-v2-m3` as a "quality/multilingual" option on a free GPU; keep Cohere Rerank as an optional hosted adapter (about $0.25–$1 for a 100–500-query eval at $2–2.50 per 1k).

### Gaps
- Jina Reranker pricing and latency (v2/v3) was not retrieved this session.
- Lost-in-the-Middle's exact degradation figures (models, 20-document setting, % drop, comparison to closed-book) were not in the abstract and were not verified. Do not quote a specific percentage without checking the PDF.
- I found no primary BAAI or official latency benchmark for bge-reranker-v2-m3; the CPU figures are community anecdotes with unknown hardware.
- I found no published "tokens saved vs quality" curve specifically for score-threshold dynamic-k. It needs to be produced by the team's own eval.

---

## Q6. Simple non-ML cost levers worth including

### Takeaway
The cheapest wins need no ML. Output tokens cost several times more than input, so **max_tokens caps, stop sequences, explicit length instructions and structured or terse output formats** often save more than prompt compression. **Prompt caching plus trimming redundant template text and few-shot examples** handles the input side. CostGuard should apply these first and report their savings separately from ML compression.

### Cited Findings
- Output tokens typically cost **2–5x more than input tokens**; about 5x for Claude 3 models. Output is generated sequentially (autoregressively), so it is slower as well as costlier — [snippet: DeepInfra Pricing 101](https://deepinfra.com/blog/pricing-101-token-math-cost-per-completion); [snippet: glukhov.org](https://www.glukhov.org/llm-performance/cost-effective-llm-applications/); [snippet: Medium, "Wasting LLM tokens"](https://medium.com/@leotonezi/what-i-learned-about-wasting-llm-tokens-bb4f2e332ba4)
- Practitioner guidance: set **max_tokens** sensibly and use **stop sequences** on every request; keep repeated blocks byte-identical to get cached input pricing — [snippet: glukhov.org](https://www.glukhov.org/llm-performance/cost-effective-llm-applications/); [snippet: Statsig on max tokens](https://www.statsig.com/perspectives/sure-please-provide-the-title-or-main-topic-of-the-blog)
- Few-shot redundancy is real. In LLMLingua's GSM8K setup, a prompt compressed to a **1-shot budget (5x)** scored 79.08 EM, against 78.85 for the full multi-shot prompt. Most demonstration tokens were not needed — [LLMLingua](https://arxiv.org/html/2310.05736)
- Caching: OpenAI applies automatic exact-prefix caching from 1,024 tokens at a 50–90% discount; Anthropic cache reads cost about 10% of the base price — [OpenAI Cookbook](https://developers.openai.com/cookbook/examples/prompt_caching_201); [snippet: Humanloop](https://humanloop.com/blog/prompt-caching); [snippet: tokonomics](https://tokonomics.ca/blog/prompt-caching-guide-openai-anthropic)
- Academic framing of cost-aware LLM usage (quality/cost trade-off optimization) exists: "Towards Optimizing the Costs of LLM Usage" (arXiv 2402.01742) and LLMBridge (arXiv 2410.11857) — [snippet](https://arxiv.org/pdf/2402.01742); [snippet](https://arxiv.org/pdf/2410.11857)

### Inferences
Each lever below is cheap to implement and easy to measure with the same eval harness.
- **max_tokens per route.** Set it per route (e.g. 256 for QA, 800 for summaries). Log `finish_reason == "length"` as a quality-risk signal.
- **Reasoning models.** On o-series and GPT-5 "thinking"-style models, hidden reasoning tokens are billed as output. A low max-output setting can truncate the answer entirely, so set the reasoning-effort setting low where offered rather than only capping max_tokens. This is not verified this session; check the provider docs.
- **Stop sequences.** Use them for list or record outputs, for example stopping at "\n\n" or a sentinel.
- **Structured output.** JSON-schema or function-calling output removes preamble and pleasantries ("Sure! Here's…") and makes outputs parseable, so there are fewer retries. Keep schemas short, because schemas are input tokens; they are cacheable.
- **Template trimming.**
  - Deduplicate instructions repeated across system and user messages.
  - Remove verbose politeness and boilerplate.
  - Shorten field names in in-context JSON.
  - Strip whitespace and HTML from retrieved docs (a deterministic pre-pass, often 10–30% of web-scraped text; the team should measure this rather than quote it).
- **Few-shot pruning.** Choose 1–2 examples by embedding similarity to the query instead of a fixed 8, or drop examples entirely for strong models. Note that dynamic example selection breaks prefix caching. Choose either *static and cached* or *dynamic and short*, and measure both.
- **Ordering for caching.** Put static content first and variable content last, and keep a stable tool order. This costs nothing to implement.
- **Reporting.** Show a savings waterfall: baseline → caching → output caps → template trim → reranker dynamic-k → LLMLingua-2. Present each step's quality delta with a CI. Reviewers will attribute savings to the right component, and the team can defend that compression is used only where it pays.

### Gaps
- I did not fetch official current input/output prices for specific 2026 models (OpenAI, Anthropic, Google mini tiers). The team must pull them from official pricing pages on the day; do not quote the blog ratios as exact.
- I found no rigorous published measurement of savings from structured output or template trimming; these are practitioner claims. CostGuard's own measurements would fill this gap.
