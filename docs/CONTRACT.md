# CostGuard build contract (read before touching code)

**Story:** "Given an LLM request, what is the cheapest way to produce an acceptable-quality answer, and can we prove it?"
**Deployment model, service-side:** CostGuard is a gateway the *operator* runs between its own backend services and the LLM provider.
- End users never call it, and the savings land on the operator's LLM bill.
- Tenant and mode come from the caller's API key (`COSTGUARD_API_KEYS`, per-tenant policy in `configs/policy.yaml`).
- Body fields are hints from trusted internal callers. Mode overrides are honoured only for tenants that allow them.

**Domain:** customer support for *ShopNest*, a fictional online store selling electronics, home goods and apparel.
- Workload trace: public customer-support data (Bitext) plus questions about the store-policy knowledge base, which carry retrieved context.
- Eval set: hand-written by the team. Today only `eval/data/evalset/seed.jsonl` exists, and it is AI-written scaffolding (`"author": "seed"`).

**Research brief:** `docs/research/LLM CostGuard project research.md` (detailed notes in `../research/notes/`). Read the sections for your component; it has the published numbers, pitfalls and evaluation method.

## Environment rules
- **Python:** always run tools as `./.venv/bin/python -m <tool>` (3.12), e.g. `-m pip install …` or `-m pytest`.
  - The venv was moved, so its console-script shebangs are stale; `-m` always works.
  - **Don't edit `pyproject.toml`.** List new dependencies in your final report instead.
- **API keys:** none are available. Tests must run with `COSTGUARD_BACKEND=mock` (the default).
  - Never call MLX models from tests; they're slow, and the coordinator runs the real experiments.
- **Audio and windows:** never play audio or open GUI windows or browsers.
- **Core files are read-only for builders:** `costguard/{schemas,interfaces,pipeline,config,factory,server,pricing,tokens}.py` and `costguard/providers/*`. If you need a change, describe it in your final report and work around it meanwhile.
- **Git:** don't run `git commit`. The coordinator commits.
- **Tests:** run the whole suite before finishing: `./.venv/bin/python -m pytest -q`. It must stay green.

## How the engine calls you (costguard/pipeline.py)
Requests flow through these stages in order:
1. exact cache
2. semantic cache
3. context optimiser
4. compressor (context block only)
5. router
6. upstream provider
7. cache write-back
8. one `TraceRecord`, sent to every hook

Each stage module exposes a builder that `costguard/factory.py` imports by name, with signature `builder(settings: Settings, policy: Policy) -> component`.

| Component | Module | Builder | Must satisfy (`costguard/interfaces.py`) |
|---|---|---|---|
| Exact cache | `costguard/cache/exact.py` | `build_exact_cache` | `ExactCache`: `get(key)`, `put(key, entry)`, `clear()` |
| Semantic cache | `costguard/cache/semantic.py` | `build_semantic_cache` | `SemanticCache`: `lookup(query, partition, threshold) -> SemanticHit`, `insert(query, partition, entry)`, `clear()` |
| Context optimiser | `costguard/context/optimizer.py` | `build_context_optimizer` | `optimize(query, docs, budget_tokens, min_score) -> ContextResult` |
| Compressor | `costguard/context/compress.py` | `build_compressor` | `compress(text, rate, query) -> CompressResult` (`rate` = fraction of tokens to keep) |
| Router | `costguard/router/router.py` | `build_router` | `route(RouteInput, policy: "gated"\|"aggressive") -> RouteDecision` (alias `"strong"`/`"cheap"`) |
| Observability hooks | `costguard/obs/hooks.py` | `build_hooks(settings, policy) -> list[callable(TraceRecord)]` | Must never raise into serving |
| Prometheus | `costguard/obs/metrics.py` | `mount(app)` adds `GET /metrics` | |

**Partitions:** `partition` strings look like `tenant|syshash|kb_version|ctx-or-noctx`. Never return an entry from a different partition.

**Mode thresholds:** these live in `configs/policy.yaml` (`tau`, `context_budget_tokens`, `compression_rate`, `router_policy`). Calibration scripts should *write their recommendation* to `eval/results/*.json` and print it. The coordinator updates `policy.yaml`.

## Shared eval APIs (owned by the eval workstream; others code against these signatures)

**`eval/judge.py`**
```python
class Judge:  # wraps a provider + model; default = the engine's backend, strong tier, temperature 0
    def pairwise(self, question: str, answer_a: str, answer_b: str, reference: str | None = None) -> str: ...
        # returns "A" | "B" | "tie"; internally runs both orders (position swap); disagreement -> "tie"
    def grade(self, question: str, answer: str, reference: str | None = None) -> float: ...
        # 0..1 absolute correctness/helpfulness against the reference (rubric in the prompt)
def get_judge(settings=None) -> Judge: ...
```

**`eval/stats.py`**
```python
def paired_bootstrap(diffs: list[float], n: int = 2000, seed: int = 0, clusters: list | None = None) -> tuple[float, float, float]: ...  # mean, lo95, hi95
def proportion_ci(k: int, n: int) -> tuple[float, float, float]: ...  # Wilson
```

**`eval/kb.py`** (owned by the context workstream)
```python
def load_kb() -> list[dict]: ...             # {"id","title","text"} chunks from eval/data/kb/*.md
def retrieve(query: str, k: int = 8) -> list[str]: ...   # deliberately generous naive retrieval (bi-encoder top-k)
```

## Data formats

**`eval/data/evalset/*.jsonl`:** one file per author. This is the graded, hand-written set; AI-written seed rows say `"author": "seed"`.
```json
{"id": "ret-001", "category": "returns", "query": "...", "reference": "...", "needs_context": true,
 "type": "answerable" | "trap_pair" | "hard", "pair_id": null, "should_cache_hit": null, "author": "prathmesh"}
```
- `trap_pair` rows come in twos sharing a `pair_id`. They look alike but need different answers (negation, different order number, different product), so a cache hit between them is a **false hit**.

**`eval/data/trace_v1.jsonl`:** the frozen replay workload. `trace_v1.sha256` sits next to it.
```json
{"pos": 0, "item_id": "...", "cluster_id": "...", "query": "...", "category": "...", "context": ["..."],
 "reference": "...", "source": "bitext|kb|qqp|trap", "dup_of": null}
```

**Results:** everything numeric goes to `eval/results/*.json`. The README tables are generated from these files and never hand-typed.

## Categories (router + gate + dashboard)
`order`, `shipping`, `returns`, `refund`, `payment`, `account`, `product`, `other`.

## Definitions (use exactly these)
- **Hit rate:** cache hits ÷ all requests.
- **False-hit rate:** wrong cache hits ÷ all requests. It's per request, not per hit (see the research brief on why per-hit precision flatters thresholds).
- **Savings:** 1 − Σ cost ÷ Σ baseline_cost.
  - The baseline is the strong tier with the full prompt and no cache.
  - Costs are list-price equivalents from `configs/prices.yaml`.
  - In the A/B, the headline (paired) savings divide by A0's actual cost on the same items; the `baseline_cost_usd` version is reported as "est. savings" (`docs/EVALUATION.md` §4).
- **Quality retained:** the optimised arm's judge score ÷ the baseline score on the same items, with a paired bootstrap 95% CI.
