"""Small, dependency-free (numpy only) statistics for paired A/B evaluation.

- paired_bootstrap: mean of per-item differences with a percentile 95% CI. Pass `clusters` to resample whole
  clusters (e.g. trace cluster_id) instead of single requests: duplicates of one question are correlated by
  construction, so an i.i.d. bootstrap would overstate confidence (Miller, "Adding Error Bars to Evals", 2024).
- ratio_bootstrap: sum(num) / sum(den) with the same (optionally clustered) resampling. Used for savings
  (1 - cost/baseline) and quality retained (arm score / baseline score).
- proportion_ci: Wilson score interval (well behaved at 0 and n).
- mcnemar: paired pass/fail test on the discordant counts (exact binomial when small).
- cohen_kappa: chance-corrected agreement, for judge-vs-human labels.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence

import numpy as np

NAN = float("nan")


def _cluster_index(clusters: Sequence) -> tuple[np.ndarray, int]:
    labels = [str(c) for c in clusters]
    order: dict[str, int] = {}
    inv = np.fromiter((order.setdefault(lab, len(order)) for lab in labels), dtype=np.int64, count=len(labels))
    return inv, len(order)


def _resampled_sums(values: list[np.ndarray], n: int, seed: int, clusters: Optional[Sequence]) -> list[np.ndarray]:
    """For each bootstrap replicate, the resampled sum of every array (same resample for all arrays) plus counts."""
    rng = np.random.default_rng(seed)
    m = len(values[0])
    if clusters is None:
        idx = rng.integers(0, m, size=(n, m))
        return [v[idx].sum(axis=1) for v in values] + [np.full(n, float(m))]
    if len(clusters) != m:
        raise ValueError("clusters must have the same length as the data")
    inv, k = _cluster_index(clusters)
    per = [np.bincount(inv, weights=v, minlength=k) for v in values]
    cnt = np.bincount(inv, minlength=k).astype(float)
    idx = rng.integers(0, k, size=(n, k))
    return [p[idx].sum(axis=1) for p in per] + [cnt[idx].sum(axis=1)]


def paired_bootstrap(diffs: list[float], n: int = 2000, seed: int = 0,
                     clusters: Optional[list] = None) -> tuple[float, float, float]:
    """Mean of paired differences and its 95% percentile-bootstrap CI: (mean, lo95, hi95).

    With `clusters`, whole clusters are resampled with replacement and the statistic is the mean over all
    requests in the resampled clusters (cluster bootstrap)."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        return (NAN, NAN, NAN)
    mean = float(d.mean())
    if d.size == 1:
        return (mean, mean, mean)
    sums, counts = _resampled_sums([d], n, seed, clusters)
    boots = sums / counts
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (mean, float(lo), float(hi))


def ratio_bootstrap(num: list[float], den: list[float], n: int = 2000, seed: int = 0,
                    clusters: Optional[list] = None) -> tuple[float, float, float]:
    """sum(num) / sum(den) on paired items, with a (cluster) bootstrap 95% CI: (ratio, lo95, hi95)."""
    a, b = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if a.size == 0 or a.size != b.size or b.sum() == 0:
        return (NAN, NAN, NAN)
    ratio = float(a.sum() / b.sum())
    if a.size == 1:
        return (ratio, ratio, ratio)
    sa, sb, _ = _resampled_sums([a, b], n, seed, clusters)
    ok = sb != 0
    boots = sa[ok] / sb[ok]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (ratio, float(lo), float(hi))


def proportion_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson score interval: (p, lo95, hi95). Returns (nan, 0, 1) when n == 0."""
    if n <= 0:
        return (NAN, 0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)       # exact endpoints at the boundaries
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (p, lo, hi)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar test on discordant pairs (b: baseline pass / arm fail, c: baseline fail / arm pass).

    Returns (statistic, two-sided p). Exact binomial when b + c < 25, else chi-square with continuity correction."""
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    if n < 25:
        k = min(b, c)
        p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        return (float(k), min(1.0, p))
    stat = (abs(b - c) - 1) ** 2 / n
    return (stat, math.erfc(math.sqrt(stat / 2)))   # chi2(1) survival function


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's kappa for two raters over the same items (labels compared as strings)."""
    if len(a) != len(b) or not a:
        return NAN
    a, b = [str(x) for x in a], [str(x) for x in b]
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else NAN
    return (po - pe) / (1 - pe)


def percentile(xs: Sequence[float], q: float) -> float:
    v = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.percentile(v, q)) if v else NAN
