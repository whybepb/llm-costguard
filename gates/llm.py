"""LLM quality gate on the held-out eval set.

Usage: .venv/bin/python -m gates.llm results/<prefix>__cost_vs_quality_summary.csv

Reads the summary written by `python -m e2e.run --data eval_data` and checks the configured
pipeline against [gates.llm] in config/costguard.toml: wrong answers, the drop in correct
answers vs the baseline, and cost vs the baseline. Exit code 1 on any failure.
"""

import csv
import sys

from config import CFG


def check(summary_path):
    g = CFG["gates"]["llm"]
    rows = {r["pipeline"]: r for r in csv.DictReader(open(summary_path))}
    base, pipe = rows["baseline"], rows[g["pipeline"]]
    drop = float(base["correct_pct"]) - float(pipe["correct_pct"])
    return [
        ("wrong_pct", float(pipe["wrong_pct"]), f"<= {g['max_wrong_pct']}", float(pipe["wrong_pct"]) <= g["max_wrong_pct"]),
        ("correct_drop_vs_baseline", round(drop, 4), f"<= {g['max_correct_drop_vs_baseline']}",
         drop <= g["max_correct_drop_vs_baseline"] + 1e-9),
        ("cost_vs_baseline", float(pipe["cost_vs_baseline_pct"]), f"<= {g['max_cost_vs_baseline']}",
         float(pipe["cost_vs_baseline_pct"]) <= g["max_cost_vs_baseline"]),
    ]


def main():
    rows = check(sys.argv[1])
    print(f"pipeline {CFG['gates']['llm']['pipeline']} vs baseline")
    for name, value, bound, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:26s} {value!s:>8}  (bound {bound})")
    raise SystemExit(0 if all(ok for *_, ok in rows) else 1)


if __name__ == "__main__":
    main()
