"""Statistics: percentile bootstrap CIs, paired bootstrap of differences, exact McNemar test.

Written here (not taken from AuditKIT) because AuditKIT's bootstrap p-values have a known bug (PR #9).
"""
from __future__ import annotations

from math import comb

import numpy as np


def rate_ci(x: list[int] | np.ndarray, resamples: int = 10000, ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Mean of a 0/1 vector with a percentile bootstrap CI."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a: list[int], b: list[int], resamples: int = 10000, ci: float = 0.95,
                   seed: int = 0) -> tuple[float, float, float]:
    """CI for mean(b) - mean(a) on the same items (resampling items, keeping pairs together)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired vectors must be the same non-zero length")
    d = b - a
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(d.mean()), float(lo), float(hi)


def mcnemar_exact(a: list[int], b: list[int]) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
