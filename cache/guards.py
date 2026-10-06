"""Query-text guards for the semantic cache.

Each guard compares the new query with a cached query and returns True when they
are safe to treat as the same question. A guard only rejects on concrete textual
evidence that the answers may differ.
"""

import re
from collections import Counter

# Any token containing a digit: 10, $62, 3.5, NS-22011, ABCD-1234, 2026-10-16.
_NUMBER = re.compile(r"[$€£]?[A-Za-z0-9]*\d[A-Za-z0-9.,:/-]*")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_NEGATIONS = {
    "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither", "nor",
    "cannot", "without",
}


def numbers(text):
    return Counter(m.rstrip(".,:/-") for m in _NUMBER.findall(text))


def numbers_match(a, b):
    """Same numbers, amounts and codes, with the same counts."""
    return numbers(a) == numbers(b)


def _is_negation(word):
    return word in _NEGATIONS or word.endswith("n't")


def negation_diff(a, b):
    """Negation words left in each query after removing the words both share."""
    wa = Counter(_WORD.findall(a.lower().replace("’", "'")))
    wb = Counter(_WORD.findall(b.lower().replace("’", "'")))
    only_a, only_b = wa - wb, wb - wa
    neg_a = sum(n for w, n in only_a.items() if _is_negation(w))
    neg_b = sum(n for w, n in only_b.items() if _is_negation(w))
    return neg_a, neg_b


def negation_match(a, b):
    """Same number of negations among the words that differ.

    Shared background ("the status has not changed") cancels out, so only a
    negation added or removed in the reworded part counts.
    """
    neg_a, neg_b = negation_diff(a, b)
    return neg_a == neg_b


GUARDS = [("numbers", numbers_match), ("negation", negation_match)]
