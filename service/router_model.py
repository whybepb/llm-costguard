"""Train the live router on all 150 demo queries (same method as routing/eval_split.py).

Threshold: out-of-fold P(large) with group k-fold by cluster, then the highest
threshold that sends no complex query to the small model. The fitted router is
saved to service_data/router.joblib so later starts skip training.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from demo_data.build_dataset import render_case
from routing.router import Router, text_features
from window.manager import MIN_CONTEXT_TOKENS, approx_tokens, chunk_docs, select_hybrid

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]
TRAIN_BUDGET = 800


def train(docs, cases, embed, chunk_vec, boilerplate, budget=TRAIN_BUDGET):
    rows = [json.loads(line) for line in (ROOT / "demo_data" / "queries.jsonl").open()]
    qvecs = embed(r["query"] for r in rows)
    feats, ys, groups = [], [], []
    for r, qv in zip(rows, qvecs):
        case_text = render_case(cases[r["case_id"]]) if r["case_id"] else ""
        chunks = chunk_docs(docs, r["attachments"])
        if approx_tokens(case_text) + sum(c["tokens"] for c in chunks) >= MIN_CONTEXT_TOKENS:
            chunks, _ = select_hybrid(chunks, qv, [chunk_vec(c) for c in chunks], boilerplate, budget=budget)
        ctx = approx_tokens(case_text) + sum(c["tokens"] for c in chunks)
        feats.append(text_features(r["query"], len({c["doc_id"] for c in chunks}), ctx))
        ys.append(int(not r["small_model_ok"]))
        groups.append(r["cluster_id"])
    Xe, Xf, y = np.stack(qvecs), np.array(feats, float), np.array(ys)
    oof = np.zeros(len(y))
    for a, v in GroupKFold(n_splits=5).split(Xe, y, groups):
        oof[v] = Router().fit(Xe[a], Xf[a], y[a]).p_large(Xe[v], Xf[v])
    ok = [t for t in THRESHOLDS if not any(yy and s < t for yy, s in zip(y, oof))]
    return Router(max(ok) if ok else THRESHOLDS[0]).fit(Xe, Xf, y)


def load_or_train(path, *args):
    path = Path(path)
    if path.exists():
        return joblib.load(path)
    router = train(*args)
    path.parent.mkdir(exist_ok=True)
    joblib.dump(router, path)
    return router
