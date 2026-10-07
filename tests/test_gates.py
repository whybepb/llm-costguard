"""Deterministic quality gates (see gates/deterministic.py). No API calls."""

import pytest

from gates import deterministic as d


@pytest.fixture(scope="module")
def metrics():
    return {**d.exact_metrics(), **d.semantic_metrics(), **d.window_metrics()}


@pytest.mark.parametrize("name", ["exact_false_hits", "exact_hits", "semantic_wrong_hits",
                                  "semantic_paraphrase_hits", "window_all_facts_kept"])
def test_gate(metrics, name):
    value, bound, ok = next((v, b, ok) for n, v, b, ok in d.check(metrics) if n == name)
    assert ok, f"{name} = {value}, bound {bound}"
