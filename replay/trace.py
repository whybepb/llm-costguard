"""Replay a fixed request trace through the running service, so Grafana shows baseline vs
optimized for exactly that trace.

Usage: .venv/bin/python -m replay.trace [--data eval_data] [--mode replay] [--url http://localhost:8000]

Clears the trace tenant's cache, then sends every query in order with its case and exact
attachments, shadow mode on. In replay mode (default) answers, the baseline and quality all
come from the memoized e2e run, so it costs nothing; the service still does the real cache,
window and routing work and records every metric. Cross-checks each request's layer
against the offline e2e run and writes results/trace_replay_<data>__per_query.csv.
"""

import argparse
import csv
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval_data")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--mode", default="replay", choices=["replay", "live", "mock"])
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--tenant", default=None)
    args = ap.parse_args()
    tenant = args.tenant or f"trace_{args.data}"
    rows = [json.loads(line) for line in (ROOT / args.data / "queries.jsonl").open()]
    cases = json.loads((ROOT / args.data / "cases.json").read_text())
    prefix = "e2e_evalset" if args.data != "demo_data" else "e2e"
    offline = {r["id"]: r for r in csv.DictReader(open(ROOT / "results" / f"{prefix}__per_query_all_pipelines.csv"))
               if r["pipeline"] == f"optimized_w{args.budget}"}

    client = httpx.Client(base_url=args.url, timeout=300)
    client.delete("/api/cache", params={"tenant": tenant}).raise_for_status()
    out = []
    for r in rows:
        body = {"query": r["query"], "tenant": tenant, "attachments": r["attachments"],
                "case": cases[r["case_id"]] if r["case_id"] else None, "window_budget": args.budget,
                "shadow": True, "use_cache": True, "llm_mode": args.mode}
        res = client.post("/v1/answer", json=body)
        if res.status_code != 200:
            raise SystemExit(f"{r['id']}: HTTP {res.status_code}: {res.text[:300]}")
        x = res.json()
        ref = offline.get(r["id"], {})
        out.append({
            "id": r["id"], "variant": r["variant"], "layer": x["layer"], "model": x["model"],
            "offline_layer": ref.get("layer", ""), "layer_matches_offline": int(x["layer"] == ref.get("layer")),
            "cost_usd": x["cost_usd"], "baseline_cost_usd": x["baseline_cost_usd"],
            "baseline_source": x["baseline_source"], "savings_pct": x["savings_pct"],
            "input_tokens": x["input_tokens"], "baseline_input_tokens": x["baseline_input_tokens"],
            "quality": x["quality"]["verdict"] if x["quality"] else "", "latency_s": x["latency_s"],
            "query": r["query"],
        })
        print(f"{r['id']} {x['layer']:15s} {x['model'] or '-':18s} ${x['cost_usd']:.5f} vs ${x['baseline_cost_usd']:.5f}"
              f"  {out[-1]['quality'] or '-'}")

    path = ROOT / "results" / f"trace_replay_{args.data}__per_query.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    cost, base = sum(o["cost_usd"] for o in out), sum(o["baseline_cost_usd"] for o in out)
    print(json.dumps({
        "requests": len(out), "cost_usd": round(cost, 4), "baseline_cost_usd": round(base, 4),
        "savings_pct": round(1 - cost / base, 4),
        "layers_matching_offline_run": f"{sum(o['layer_matches_offline'] for o in out)}/{len(out)}",
        "quality": {q: sum(1 for o in out if o["quality"] == q) for q in sorted({o["quality"] for o in out})},
        "written": str(path.relative_to(ROOT)),
    }, indent=1))


if __name__ == "__main__":
    main()
