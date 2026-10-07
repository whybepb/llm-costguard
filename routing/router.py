"""Model router: small model (Haiku 5.5) or large model (Sonnet 5.5) for a cache miss.

A logistic regression over the query's bge embedding plus a few text and context
features predicts P(needs large). Routing runs after window management, so it can
use what windowing kept. Above the threshold the query goes large; the threshold
leans toward large, since a complex query on the small model risks a wrong answer
while the reverse only costs money.
"""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import CFG, price

SMALL, LARGE = CFG["models"]["small"], CFG["models"]["large"]
PRICES = {SMALL: price(SMALL), LARGE: price(LARGE)}  # $ per 1M input, output tokens

_CONNECTIVES = re.compile(r"\b(but|or|and|instead|unless|if|while|although|already|still)\b", re.I)


def text_features(query, docs_kept, context_tokens):
    words = query.split()
    return [
        len(words),
        len(re.findall(r"[.!?]+", query)),
        query.count("?"),
        len(re.findall(r"\d+", query)),
        len(_CONNECTIVES.findall(query)),
        int(query.rstrip().lower().endswith("right?")),
        docs_kept,
        np.log1p(context_tokens),
    ]


class Router:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced")

    def _x(self, embeddings, feats, fit=False):
        f = self.scaler.fit_transform(feats) if fit else self.scaler.transform(feats)
        return np.hstack([embeddings, f])

    def fit(self, embeddings, feats, needs_large):
        self.model.fit(self._x(embeddings, feats, fit=True), needs_large)
        return self

    def p_large(self, embeddings, feats):
        return self.model.predict_proba(self._x(embeddings, feats))[:, 1]

    def route(self, embeddings, feats):
        return [LARGE if p >= self.threshold else SMALL for p in self.p_large(embeddings, feats)]


def cost(model, input_tokens, output_tokens):
    pin, pout = PRICES[model]
    return (input_tokens * pin + output_tokens * pout) / 1e6
