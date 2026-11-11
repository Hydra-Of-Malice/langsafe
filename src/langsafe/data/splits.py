"""Disjoint repair / held-out splits, assigned per group so no item leaks across languages."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

from .records import Prompt


def _unit(seed: int, group: str) -> float:
    h = hashlib.sha256(f"split:{seed}:{group}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_splits(prompts: Iterable[Prompt], heldout_fraction: float, seed: int) -> list[Prompt]:
    """Deterministic split by group hash: every language version of an item lands in the same split."""
    out = []
    for p in prompts:
        p.split = "heldout" if _unit(seed, p.group) < heldout_fraction else "repair"
        out.append(p)
    return out


_NORM = re.compile(r"[\W_]+", re.UNICODE)


def _shingles(text: str, n: int = 5) -> set[str]:
    t = _NORM.sub(" ", text.lower()).strip()
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def leakage_report(prompts: list[Prompt], near_dup_jaccard: float = 0.8) -> dict:
    """Check that no group, exact text or near-duplicate text appears in both splits (same language)."""
    groups = defaultdict(set)
    for p in prompts:
        groups[p.group].add(p.split)
    split_groups = [g for g, s in groups.items() if len(s) > 1]

    by_split_hash = defaultdict(set)
    for p in prompts:
        by_split_hash[p.split].add((p.lang, p.text_hash))
    exact = by_split_hash["repair"] & by_split_hash["heldout"]

    near = []
    repair = defaultdict(list)
    for p in prompts:
        if p.split == "repair":
            repair[p.lang].append((p.id, _shingles(p.text)))
    for p in prompts:
        if p.split != "heldout":
            continue
        s = _shingles(p.text)
        for rid, rs in repair[p.lang]:
            inter = len(s & rs)
            if inter and inter / len(s | rs) >= near_dup_jaccard:
                near.append((p.id, rid))
    return {"groups_in_both": split_groups, "exact_dupes": sorted(exact), "near_dupes": near,
            "ok": not split_groups and not exact and not near}


def drop_leaks(prompts: list[Prompt], report: dict) -> list[Prompt]:
    """Remove entire leaking repair groups, including translations without a matching text.

    Removing only the matching language would still let repair see another version of a held-out item.
    Held-out rows are retained, including when a group was accidentally assigned to both splits.
    """
    bad_ids = {rid for _, rid in report.get("near_dupes", [])}
    exact = {tuple(pair) for pair in report.get("exact_dupes", [])}
    bad_groups = set(report.get("groups_in_both", []))
    bad_groups.update(p.group for p in prompts
                      if p.split == "repair" and (p.id in bad_ids or (p.lang, p.text_hash) in exact))
    return [p for p in prompts if not (p.split == "repair" and p.group in bad_groups)]
