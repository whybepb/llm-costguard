"""Semantic response cache: nearest cached query by cosine similarity.

Embeddings are L2-normalized, so cosine similarity is a dot product.

Each entry can carry depends_on: the hashes of the context pieces its answer used
(policy docs and the case). With check_context on, an entry is only served when
every one of those hashes is present in the new request's context, so unrelated
attachments don't block a hit but a changed fact does.

guards is a list of (name, fn(new_query, cached_query) -> bool) run after the
context check; any False rejects that candidate.
"""

import numpy as np


class SemanticCache:
    def __init__(self, threshold, check_context=True, guards=()):
        self.threshold = threshold
        self.check_context = check_context
        self.guards = list(guards)
        self._vecs = []
        self._entries = []
        self.hits = 0
        self.misses = 0

    def _reject_reason(self, entry, query, context_hashes):
        if self.check_context and not entry["depends_on"] <= context_hashes:
            return "context_mismatch"
        for name, ok in self.guards:
            if not ok(query, entry["query"]):
                return f"{name}_mismatch"
        return None

    def lookup(self, vec, query="", context_hashes=frozenset()):
        """Return (entry, score, reason).

        entry is None on a miss. score is the cosine of the served entry on a hit,
        else the best cosine in the cache (None when empty). reason is "hit",
        "empty", "low_cosine", or why the closest candidate above threshold was
        rejected ("context_mismatch", "<guard>_mismatch").
        """
        if not self._vecs:
            self.misses += 1
            return None, None, "empty"
        scores = np.stack(self._vecs) @ vec
        best = float(scores.max())
        first_reason = None
        for i in np.argsort(-scores):
            score = float(scores[i])
            if score < self.threshold:
                break
            reason = self._reject_reason(self._entries[i], query, context_hashes)
            if reason is None:
                self.hits += 1
                return self._entries[i], score, "hit"
            first_reason = first_reason or reason
        self.misses += 1
        return None, best, first_reason or "low_cosine"

    def put(self, vec, value, query="", depends_on=frozenset()):
        self._vecs.append(vec)
        self._entries.append({**value, "query": query, "depends_on": frozenset(depends_on)})

    def __len__(self):
        return len(self._entries)
