"""CuratorKIT stage: clean, schema-check and near-duplicate-filter each language's prompts, with Lexsi provenance.

CuratorKIT does ingest, cleaning, MinHash dedup and the provenance chain. It has no language ID, no group-aware split
and its exporters drop metadata, so grouping, splitting and export stay in langsafe (see docs/UPSTREAM.md).
Dedup runs per language: a multilingual run would treat parallel translations as duplicates.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from ..config import ROOT
from .records import Prompt, write_jsonl

WORK = Path(os.environ.get('LANGSAFE_RUNS_DIR', ROOT / 'runs')) / 'curatorkit'


def _to_sample(row: dict):
    from curatorkit.schema import DataSample

    return DataSample(id=row["id"], source_uri=row["source"], instruction=row["text"], task_type="prompt_only",
                      metadata={k: row[k] for k in ("group", "lang", "kind", "category", "origin", "source_id")})


def curate(prompts: list[Prompt], minhash_threshold: float = 0.85) -> tuple[list[Prompt], dict]:
    """Return the prompts that pass CuratorKIT, plus per-language stage counts and provenance file paths."""
    from curatorkit import Curator, CuratorConfig

    by_lang: dict[str, list[Prompt]] = defaultdict(list)
    for p in prompts:
        by_lang[p.lang].append(p)

    kept_ids: set[str] = set()
    report: dict[str, dict] = {}
    for lang, ps in sorted(by_lang.items()):
        src = WORK / lang / "input.jsonl"
        write_jsonl(src, [asdict(p) for p in ps])
        cfg = CuratorConfig(
            dataset=[str(src)], format="prompt_only", preprocessing_fn=_to_sample,
            min_tokens=1,                               # SchemaGate counts words; short Indic prompts are valid
            dedup="minhash", minhash_threshold=minhash_threshold,
            clean_transforms={"strip_html": False},     # jailbreak prompts can contain tags we must keep
            output_dir=str(WORK / lang / "out"),
        )
        result = Curator(cfg).run()
        ids = {s.id for s in result.passed}
        kept_ids |= ids
        report[lang] = {"in": len(ps), "kept": len(ids), "stage_counts": _counts(result.stage_counts),
                        "provenance": str((WORK / lang / "out" / "lexsi_provenance.json").relative_to(ROOT))}
    kept = [p for p in prompts if p.id in kept_ids]
    (WORK / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return kept, report


def _counts(stage_counts) -> dict:
    try:
        return {k: (v if isinstance(v, (int, float, dict, list, str)) else str(v)) for k, v in dict(stage_counts).items()}
    except (TypeError, ValueError):
        return {"raw": str(stage_counts)}
