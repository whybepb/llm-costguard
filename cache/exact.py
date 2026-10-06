"""Exact-match response cache.

A hit requires a byte-identical query, case text and set of document texts.
The key is built from content, not ids, so an edited policy or case gets a new key.
The model is left out: the cache runs before routing picks one.
No normalization: paraphrases belong to the semantic cache.
"""

import hashlib
import json


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(query, case_text, doc_texts):
    payload = json.dumps(
        {
            "query": query,
            "case": case_text or "",
            # Hash each document and sort, so attachment order doesn't change the key.
            "docs": sorted(_sha(t) for t in doc_texts),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return _sha(payload)


class ExactCache:
    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def put(self, key, value):
        self._store[key] = value

    def __len__(self):
        return len(self._store)
