"""Query-aware sentence pruning of the windowed context.

Runs after window management. Every sentence of the kept policy sections is
scored by cosine similarity to the query; the best keep_ratio of them survive,
with at least one per section so no kept section vanishes. With neighbors=1 the
sentences right before and after a kept one are kept too, so a rule doesn't lose
the sentence that qualifies it. Surviving sentences are unchanged and stay in order.
The case block is never compressed.
"""

import math
import re

from window.manager import approx_tokens, sentences

_SPACES = re.compile(r"\s+")


def clean(text):
    """Formatting cleanup: collapse runs of whitespace."""
    return _SPACES.sub(" ", text).strip()


def compress(chunks, query_vec, sentence_vec, keep_ratio=0.7, neighbors=0):
    """Return chunks with pruned bodies. sentence_vec(s) gives a normalized embedding."""
    split = []
    for c in chunks:
        header, body = c["text"].split("\n", 1)
        ss = [clean(s) for s in sentences(body)]
        split.append((c, header, ss, [float(sentence_vec(s) @ query_vec) for s in ss]))

    flat = [(ci, si, score) for ci, (_, _, _, scores) in enumerate(split) for si, score in enumerate(scores)]
    n_keep = max(1, math.ceil(keep_ratio * len(flat)))
    keep = {(ci, si) for ci, si, _ in sorted(flat, key=lambda x: -x[2])[:n_keep]}
    for ci, (_, _, ss, scores) in enumerate(split):
        if ss and not any((ci, si) in keep for si in range(len(ss))):
            keep.add((ci, max(range(len(ss)), key=lambda i: scores[i])))
    if neighbors:
        keep |= {(ci, si + d) for ci, si in list(keep) for d in range(-neighbors, neighbors + 1)
                 if 0 <= si + d < len(split[ci][2])}

    out = []
    for ci, (c, header, ss, _) in enumerate(split):
        body = " ".join(s for si, s in enumerate(ss) if (ci, si) in keep)
        out.append({**c, "text": header + "\n" + body, "tokens": approx_tokens(body)})
    return out
