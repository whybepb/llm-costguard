"""Prometheus metrics for the CostGuard service (exposed at /metrics).

Counters accumulate across requests; the costguard_last_request_* series describe
only the most recent request, so a single demo request is visible on the dashboard.
"""

import csv
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, Info

REQUESTS = Counter("costguard_requests_total", "Requests answered", ["tenant", "layer", "model"])
COST = Counter("costguard_cost_usd_total", "Answer cost in USD: actual (pipeline) vs baseline (Sonnet 5.5, full context)",
               ["tenant", "kind"])
OVERHEAD = Counter("costguard_overhead_usd_total", "Cost outside the answer: cache judge, shadow baseline, quality judge",
                   ["tenant", "kind"])
TOKENS = Counter("costguard_tokens_total", "Input tokens: baseline (full context) vs sent (after the pipeline)",
                 ["tenant", "kind"])
SEMANTIC_REJECTS = Counter("costguard_semantic_rejections_total", "Semantic cache candidates rejected, by reason",
                           ["tenant", "reason"])
QUALITY = Counter("costguard_quality_verdicts_total", "Shadow-mode verdicts: optimized answer vs baseline answer",
                  ["tenant", "verdict", "layer"])
ERRORS = Counter("costguard_errors_total", "Failed requests", ["stage"])
STAGE_LATENCY = Histogram("costguard_stage_seconds", "Latency per pipeline stage", ["stage"],
                          buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80))
REQUEST_LATENCY = Histogram("costguard_request_seconds", "End-to-end latency of the optimized answer", ["layer"],
                            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80))

LAST = Gauge("costguard_last_request_value", "Numbers for the most recent request", ["field"])
LAST_INFO = Info("costguard_last_request", "Labels for the most recent request")

EVAL = Gauge("costguard_offline_eval", "Offline evaluation results (150 demo queries, graded by Sonnet 5.5)",
             ["pipeline", "field"])


def load_offline_eval(path):
    path = Path(path)
    if not path.exists():
        return
    for row in csv.DictReader(path.open()):
        for field in ("correct_pct", "partial_pct", "wrong_pct", "mean_score", "cost_vs_baseline_pct",
                      "cost_per_1k_queries_usd"):
            EVAL.labels(row["pipeline"], field).set(float(row[field]))


def record(r):
    t = r["tenant"]
    REQUESTS.labels(t, r["layer"], r["model"] or "none").inc()
    COST.labels(t, "actual").inc(r["cost_usd"])
    COST.labels(t, "baseline").inc(r["baseline_cost_usd"])
    TOKENS.labels(t, "baseline").inc(r["baseline_input_tokens"])
    TOKENS.labels(t, "sent").inc(r["input_tokens"])
    for kind in ("cache_judge", "shadow_baseline", "quality_judge"):
        if r["overhead_usd"].get(kind):
            OVERHEAD.labels(t, kind).inc(r["overhead_usd"][kind])
    for reason in r["semantic_rejections"]:
        SEMANTIC_REJECTS.labels(t, reason).inc()
    if r["quality"]:
        QUALITY.labels(t, r["quality"]["verdict"], r["layer"]).inc()
    for stage, sec in r["timings"].items():
        STAGE_LATENCY.labels(stage).observe(sec)
    REQUEST_LATENCY.labels(r["layer"]).observe(r["latency_s"])

    for field, value in {
        "cost_usd": r["cost_usd"], "baseline_cost_usd": r["baseline_cost_usd"],
        "savings_usd": r["savings_usd"], "savings_pct": r["savings_pct"],
        "input_tokens": r["input_tokens"], "baseline_input_tokens": r["baseline_input_tokens"],
        "output_tokens": r["output_tokens"], "latency_s": r["latency_s"],
        "router_p_large": r["router"]["p_large"] if r["router"] else -1,
        "quality_preserved": (1 if r["quality"]["verdict"] in ("equivalent", "better") else 0) if r["quality"] else -1,
    }.items():
        LAST.labels(field).set(value)
    LAST_INFO.info({
        "request_id": r["request_id"], "tenant": t, "layer": r["layer"], "model": r["model"] or "none",
        "quality": r["quality"]["verdict"] if r["quality"] else "not_checked",
        "query": r["query"][:120],
    })
