"""Inter-judge agreement (Cohen's kappa, Gwet's AC1) and accuracy against hand labels."""
from __future__ import annotations

from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(a: list[str], b: list[str], categories: tuple[str, ...]) -> float:
    """Chance-corrected agreement that stays meaningful when one label dominates, unlike kappa.

    If 97% of answers are safe, two judges who agree on nearly everything still get a kappa near zero, and a judge that
    says `safe` every time gets exactly zero. AC1 measures agreement, not correctness: it needs an audit beside it.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    if len(categories) < 2:
        raise ValueError("need at least two categories")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    counts = Counter(a) + Counter(b)
    pe = sum((counts[k] / (2 * n)) * (1 - counts[k] / (2 * n)) for k in categories) / (len(categories) - 1)
    return (po - pe) / (1 - pe)


def accuracy(pred: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else float("nan")
