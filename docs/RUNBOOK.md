# CostGuard runbook

How to deploy, change, roll back, switch off and watch CostGuard. Architecture: [ARCHITECTURE.md](ARCHITECTURE.md). Rationale: [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).

**Every request row and every `/metrics` scrape carries the policy `config_hash`.** So every number, alert and incident traces back to one version of `configs/policy.yaml` (plus its system-prompt file).

---

## 1. Deploy

### Secrets and configuration

| Variable | Purpose | Where it comes from |
|---|---|---|
| `COSTGUARD_BACKEND` | `mock` (default, no key), `anthropic` (real), `mlx` (local, Apple), others via LiteLLM | env |
| `COSTGUARD_ANTHROPIC_API_KEY` | Anthropic key, **secret** | local `.env` (gitignored; copy `.env.example`) or the host's secret store. Never in git, never in the image (`.dockerignore` excludes `.env`) |
| `COSTGUARD_ANTHROPIC_BASE_URL` | Override the endpoint (default `https://api.anthropic.com`). An inherited `ANTHROPIC_BASE_URL` is deliberately ignored | env; leave unset |
| `COSTGUARD_API_KEYS` | Caller keys → tenants, `"key1:shopnest-support,key2:shopnest-internal"`. Empty = open dev mode. **Required on any public URL** | secret |
| `COSTGUARD_POLICY` | Policy file (default `configs/policy.yaml`); used for canaries | env |
| `COSTGUARD_DB` | SQLite request log (default `data/runtime/costguard.sqlite`) | env |
| `COSTGUARD_SEMANTIC_BACKEND`, `QDRANT_URL` | `memory` (default) or `qdrant` | env |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `COSTGUARD_LANGFUSE_SAMPLE` | Optional async tracing. It also needs `pip install langfuse`, or the image built with `--build-arg EXTRA_PIP=langfuse`. Without the package it is silently disabled | secret / env |
| `COSTGUARD_ADMIN_TOKEN` | Guards `POST /v1/drift/baseline` | secret |
| `COSTGUARD_DRIFT_BASELINE` | Path to a frozen drift baseline JSON (§7) | env |
| `COSTGUARD_METRICS=0`, `COSTGUARD_DRIFT=0` | Disable those hooks | env |

### Local

```bash
make serve                                   # mock backend on :8000
COSTGUARD_BACKEND=anthropic make serve       # real API, key from .env
make dashboard                               # Streamlit on :8501
```

### Docker / Compose

```bash
docker build -t costguard .                                  # proxy image; fastembed models baked in
docker run --rm -p 8000:8000 costguard                       # mock backend
docker run --rm -p 8000:8000 -e COSTGUARD_BACKEND=anthropic -e COSTGUARD_ANTHROPIC_API_KEY costguard
docker compose up --build                                    # proxy + Qdrant
docker compose --profile dashboard up --build                # + dashboard on :8501
```

### Render (free web service)

1. Push the repo, then in Render choose **New → Blueprint** and select it. `render.yaml` defines one free Docker web service with `healthCheckPath: /health`.
2. When prompted, enter the secrets:
   - `COSTGUARD_ANTHROPIC_API_KEY`;
   - `COSTGUARD_API_KEYS`, which is **mandatory**: without it, anyone with the URL spends our budget.
   - The Langfuse keys are optional.
3. Also set a monthly spend limit in the Anthropic console.
4. `autoDeploy: true` redeploys on every push to `main`. Because `main` only accepts PRs with green `tests` and `eval-gate` checks (§8), every deploy has passed the eval gate.
5. **Pre-warm 2–3 minutes before any demo:** `curl -fsS https://<service>.onrender.com/health`. A free instance sleeps after 15 idle minutes and takes about 1 minute to wake. Keep a recorded backup video.
6. Never load-test Render. It is a single instance and can be suspended for unusual traffic. Use `make loadtest` locally.

### Post-deploy check (any target)

```bash
curl -fsS $URL/health          # status ok; components show class names, not "not available"; config_hash = expected
curl -fsS $URL/v1/chat/completions -H "Authorization: Bearer $CALLER_KEY" -H 'content-type: application/json' \
     -d '{"messages":[{"role":"user","content":"Where is my order?"}]}' -D - | grep x-costguard
curl -fsS $URL/metrics | grep costguard_build_info      # config_hash label matches
curl -fsS $URL/v1/drift | head -c 300                   # "warming_up" until 500 requests, then "ok"
```

---

## 2. What a "policy version" is

The deployable unit of behaviour is the pair:

- **`configs/policy.yaml`:** modes, τ, budgets, compression rates, router policy, tenants, `kb_version`, and the system-prompt file.
- **`configs/router_gate.json`:** per-category downshift verdicts, written by `make gate`.

Both are versioned in git. `config_hash` is a sha256 over the policy and its system prompt. It appears:

- on every log row;
- in the `x-costguard-config-hash` header;
- in `/health`;
- in `costguard_build_info` and `costguard_requests_by_config_total{config_hash}`.

`router_gate.json` is re-read whenever its modification time changes, so a gate change needs no restart. A policy change needs a restart or redeploy, because the policy is loaded at start-up.

---

## 3. Rollout of a new policy version: shadow → canary → full

**The experiment unit is the caller/session, not the request.** Semantic hits, compression and downshifts all change outputs, so per-request randomisation is invalid (W8S2).

| Stage | How | Gate to the next stage |
|---|---|---|
| **0. Offline gate** | Open a PR that changes `policy.yaml` or `router_gate.json`. CI runs the tests and `python -m eval.ci_gate` on cassettes | Both required checks green: no quality regression beyond margin, trap false-hits within budget |
| **1. Shadow** (offline replay, zero user impact) | Replay the frozen trace with the candidate: `COSTGUARD_POLICY=configs/policy.candidate.yaml make ab`. Optionally also replay yesterday's logged queries (`query` + context in the SQLite log) through the candidate | Savings ≥ the current policy's. Quality retained, lower 95% CI ≥ **0.95**. False-hit rate ≤ the mode budget (0.5 / 1 / 3%). Miss-path overhead p99 < 100 ms (`make loadtest`) |
| **2. Canary** | Run the candidate as a second instance (`COSTGUARD_POLICY=…candidate.yaml`, second Render service or container). Move callers by key or session: internal tenant first, then **0.5% → 1% → 5% → 10% → 50%** of support sessions, assigned by hashing the session id in the calling service | Each step lasts ≥ 24 h **and** ≥ 500 requests. Canary vs control: error rate < 0.5%; p99 latency ≤ +20%; hit-path overhead p99 < 50 ms; stage fail-open < 1%; sampled judge score on semantic hits and downshifts not below control (lower CI ≥ 0.95 retained); cost per request ≤ control. **Any breach → rollback (§4)** |
| **3. Full** | Merge the candidate into `configs/policy.yaml` and redeploy. Retire the control instance after 24 h | Watch the §6 alerts for 24 h. `costguard_requests_by_config_total` should show 100% on the new hash |

Two special cases:

- **Embedding-model change:** use **blue-green** only. Build the new index and re-calibrate τ (`make sweep`), then switch, because the old τ is meaningless for the new model.
- **KB change:** bump `kb_version`. That creates a new cache partition, so the cache starts cold. Expect the hit rate to dip and use the canary ramp to absorb it.

---

## 4. Rollback

1. **Revert the policy:** `git revert <commit>` on `configs/policy.yaml` (and/or `configs/router_gate.json`), then push. Render redeploys in a few minutes. Or point `COSTGUARD_POLICY` at the previous file and restart.
2. **Router-only rollback, no restart:** restore the previous `configs/router_gate.json`, or set the category to `"allow": false`. The router picks up the new mtime on the next request.
3. **During a canary:** move callers back to the control instance first. That is instant and caller-side. Fix forward afterwards.
4. **Verify:**
   - `/health` and `costguard_build_info` show the old `config_hash`;
   - `costguard_requests_by_config_total` stops increasing on the bad hash;
   - in the dashboard, filter the log by `config_hash` to compare before and after.
5. **Stale answers:** if the bad version wrote wrong answers into the cache, bump `kb_version`, which orphans every semantic entry at once, or restart to drop the in-memory tiers.

---

## 5. Kill switches

From least to most drastic:

| Scope | Switch | Takes effect |
|---|---|---|
| Downshift only | `router_gate.json`: every category `"allow": false` (gated mode then always uses strong) | Next request (hot reload) |
| One lever | `policy.yaml → modes.<mode>.semantic_cache / compression / context / router: false` | Restart |
| One tenant | `policy.yaml → tenants.<tenant>.mode: "off"` | Restart |
| One request | Body `{"costguard": {"mode": "off"}}`, for tenants with `allow_mode_override: true` | Immediately |
| Everything | `default_mode: "off"` and every tenant `mode: "off"`: pure passthrough, cost = baseline, still logged | Restart |
| Tracing | Unset `LANGFUSE_*` or set `COSTGUARD_LANGFUSE_SAMPLE=0` | Restart |

> **Quote `"off"` in YAML.** PyYAML reads a bare `off` as boolean `False`. `policy.mode(False)` then falls back to the default mode, so the kill switch silently does nothing. `modes:` already quotes its key. Do the same in `tenants:` and `default_mode:`.

Fail-open still applies in every mode. A crashing stage never needs a kill switch to keep serving; it needs one only to stop wasting money or serving bad hits.

---

## 6. Alerts and thresholds

Thresholds follow the course rules: alert on deviation from a design-time baseline (+10–20%), error rate > 0.5%, any request > 5× median cost (W7S2). The baseline numbers come from the frozen-trace A0 replay and `eval/results/loadtest.json`.

| Alert | Category | Severity | Condition (PromQL or log check) | Response |
|---|---|---|---|---|
| Error rate | Operational | P1 page | `sum(rate(costguard_request_errors_total[5m])) / sum(rate(costguard_requests_total[5m])) > 0.005` for 5m | Check provider status. Cached answers keep serving. Roll back the last deploy if it correlates |
| Hit-path overhead | Operational | P2 | `histogram_quantile(0.99, sum by (le) (rate(costguard_overhead_ms_bucket{cache_status=~"exact\|semantic"}[5m]))) > 50` | Profile the semantic stage. Check CPU saturation and embedder threads |
| Miss-path overhead | Operational | P2 | same with `cache_status="miss"` `> 100` | Usually the reranker (`costguard_stage_ms{stage="context"}`). Scale out, or lower the rerank depth |
| Latency regression | Operational | P2 | p99 `costguard_latency_ms` > 1.2 × the same window 7 days ago | Compare by `config_hash`. Check provider latency (`costguard_upstream_latency_ms`) |
| Stage fail-open | Operational | P2 | `sum by (stage) (rate(costguard_stage_errors_total[5m])) / scalar(sum(rate(costguard_requests_total[5m]))) > 0.01` | The lever is silently off and costing money. Fix it, or disable it explicitly |
| Daily spend | Operational (cost) | P2 | `sum(increase(costguard_cost_usd_total[1d])) > 1.2 * sum(increase(costguard_cost_usd_total[1d] offset 7d))` | Find the tenant/caller in the log. Check the hit-rate and route-mix panels |
| Cost spike per request | Operational (cost) | P3 | Log check (dashboard "Alert checks"): any request > 5× the median cost | Inspect it in the Request explorer: long pasted context, runaway history |
| Savings below target | Output/business | P3 | `1 - sum(rate(costguard_cost_usd_total[1d])) / sum(rate(costguard_baseline_cost_usd_total[1d])) < 0.30` | Compare against the A/B. Usually the hit rate dropped (see drift) |
| Hit-rate drop | Drift | P2 | hit rate over 1h < 0.8 × the same hour yesterday | KB or `kb_version` change? New topic? Embedding model changed? |
| Input drift | Input | P3 | `costguard_drift_status{feature=~"input_tokens\|category"} == 2` for 30m (PSI > 0.25) | Stratified sample of traces. New intent mix → re-run the τ sweep and router gate |
| Similarity drift | Drift | P2 | `costguard_drift_status{feature="cache_similarity"} == 2` for 30m | Embedding model or tokenizer change, or a paraphrase-mix shift. Re-calibrate τ |
| Output drift | Output | P3 | `costguard_drift_status{feature="output_tokens"} == 2` | Short generic answers after a downshift? Check the judge sample by route |
| Route-mix drift | Drift | P3 | `costguard_drift_status{feature="route"} == 2` | Router gate changed? A category flipped? |
| Prefix-cache ratio drop | Operational (cost) | P3 | `sum(rate(costguard_provider_cache_tokens_total{kind="read"}[1h])) / sum(rate(costguard_sent_input_tokens_total[1h]))` < 0.8 × last week | Something volatile entered the system prompt. Keep the prefix static |
| Negative savings | Output | info | `increase(costguard_negative_savings_total[1h]) > 0` | A request cost more than its baseline (cache write + retry). Inspect it |
| Telemetry dropping | Operational | info | `rate(costguard_hook_dropped_total{reason="queue_full"}[5m]) > 0` | Langfuse is slow or down. Serving is unaffected. Lower the sample rate |

Without a Prometheus server (for example on Render free), the dashboard's **Monitoring → Alert checks** table evaluates the same rules on the SQLite log.

---

## 7. Monitoring plan: the course's 5 categories

All five are covered; the rubric needs three.

| Category | Signals | Source / tool | Threshold | Why this tool |
|---|---|---|---|---|
| **Operational** | Latency p50/p95/p99 per path (hit/miss) and per stage; CostGuard overhead; error rate; stage fail-open rate; cost per request; $ saved; requests per `config_hash` | Prometheus `/metrics` (`costguard_latency_ms`, `_overhead_ms`, `_stage_ms`, `_request_errors_total`, `_cost_usd_total`, `_saved_usd_total`); dashboard Overview and Monitoring | §6 rows | Prometheus is the standard pull model; histograms give per-path percentiles cheaply |
| **Input** | Prompt-token distribution (original vs sent); category mix; context docs in/kept | `costguard_input_tokens{kind}`; PSI on `input_tokens` and `category` (`/v1/drift`) | p95 tokens +20% vs baseline; PSI > 0.25 | PSI is the course's W3S2 drift metric and needs no embeddings in the hot path |
| **Output** | Output-token distribution by tier; refusal / "I don't know" share (log query); negative-savings count; cached-answer age | `costguard_output_tokens`; PSI on `output_tokens`; SQLite `response_text` | PSI > 0.25; refusal share > 1.5× baseline | Shortening answers is the first symptom of a bad downshift or over-compression |
| **Quality** | Quality retained vs baseline (judge, paired bootstrap CI); false-hit rate; per-category gate verdicts | Offline: `eval/` judge on the frozen trace + CI gate. Online procedure (cron, **not yet automated**): sample 5–10% of semantic hits and downshifts from the log daily and grade them with `eval.judge` against a fresh strong-tier answer. Langfuse holds the traces and scores | Lower 95% CI of retained quality < 0.95 → roll back | A judge from the eval workstream with position swap; Langfuse is the course-recommended trace and score store |
| **Drift** | PSI on `cache_similarity` (embedding/model drift) and `route` (hit-rate / router shift); hit-rate trend; `config_hash` on every trace | `costguard_drift_psi{feature}`, `costguard_drift_status`, `/v1/drift`; dashboard Monitoring | PSI < 0.10 stable, 0.10–0.25 investigate, > 0.25 alert | Same maths online (`costguard/obs/drift.py`) and offline (dashboard), so a dashboard number reproduces an alert |

**Tools, and why each was chosen:**

- **Prometheus client:** standard; alert rules live outside the app.
- **SQLite log + Streamlit dashboard:** the source of truth, with no quota. It reads the same rows as the README.
- **Langfuse:** optional and async, for trace inspection in the demo. It is sampled to fit the 50k-unit free tier.
- **GitHub Actions:** the eval gate as a required check.

**Drift baseline.** By default the monitor freezes its baseline from the first 500 requests after start; `/v1/drift` reports `warming_up` until then. For a design-time baseline, build it from the A0 replay:

```bash
python -m costguard.obs.drift --db eval/results/ab_<run>.sqlite --arm A0 --out configs/drift_baseline.json
COSTGUARD_DRIFT_BASELINE=configs/drift_baseline.json make serve
```

After an *intended* change, such as a new KB or a new tenant, re-freeze it:

```bash
curl -X POST $URL/v1/drift/baseline -H "x-costguard-admin-token: $TOKEN"
```

---

## 8. CI as a required check, and the "τ = 0.6 gets blocked" screenshot

`.github/workflows/ci.yml` runs on every push and PR:

- **`tests`:** `pytest -q` with `COSTGUARD_BACKEND=mock`.
- **`eval-gate`:** `python -m eval.ci_gate` on cassettes, with no keys.
- **`docker-build`:** builds the image and smoke-tests it. Informational only.

Pip, fastembed and Hugging Face model directories are cached.

**Make the checks required.** This needs a public repo, or GitHub Pro/Team for a private one.

1. Let the workflow run once, so the check names exist.
2. Go to *Settings → Branches → Add branch protection rule* (or *Rules → Rulesets*) for `main`:
   - Require a pull request before merging;
   - **Require status checks to pass**, selecting `tests` and `eval-gate`;
   - Require branches to be up to date;
   - Do not allow bypassing.
3. Or use the CLI:

   ```bash
   gh api -X PUT repos/OWNER/REPO/branches/main/protection --input - <<'EOF'
   {"required_status_checks": {"strict": true, "contexts": ["tests", "eval-gate"]},
    "enforce_admins": true, "required_pull_request_reviews": null, "restrictions": null}
   EOF
   ```

**Stage the screenshot.** It is the single most persuasive artefact for reviewers.

1. Check that `main` is green, and record the gate's numbers on `main`.
2. Open a branch with the bad change:

   ```bash
   git checkout -b demo/tau-0.6
   # in configs/policy.yaml, modes.balanced:  tau: 0.90  ->  tau: 0.60
   git commit -am "Lower semantic-cache tau to 0.6 for more savings"
   git push -u origin demo/tau-0.6
   gh pr create --title "Lower semantic-cache tau to 0.6 for more savings" \
     --body "Raises the hit rate, so savings go up."
   ```

3. Wait for `eval-gate` to go red. The trap pairs, such as negations and different order numbers, now return cached wrong answers, and the false-hit rate exceeds its budget.
4. Screenshot three things:
   - the PR's checks box: **"eval-gate — failing"** and **"Merging is blocked"**;
   - the gate log lines naming the failed trap items and the false-hit rate against its budget;
   - the uploaded `eval-gate-results` artifact.
5. Push a second commit that reverts to the calibrated τ. Screenshot the green check. Then close the PR without merging.
6. Put both screenshots in the README and the deck, next to the threshold curve from `eval/results/threshold_sweep.json`.
