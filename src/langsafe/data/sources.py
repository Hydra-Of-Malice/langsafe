"""Loaders for the public datasets, normalised to `Prompt`.

Only prompts and labels are loaded. No model completions are ever stored from these sources.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from ..languages import AYA_FILES, MULTIJAIL_COLUMNS
from .records import Prompt, make_id

# XSTest types: safe prompts have plain type names, unsafe look-alikes are prefixed "contrast_".
XSTEST_UNSAFE_PREFIX = "contrast_"

# Verified against each dataset's primary Hugging Face metadata on 2026-11-03.
# Custom repositories resolve their current commit once; canonical sources use these immutable revisions.
SOURCE_REVISIONS = {
    "CohereLabs/aya_redteaming": "5a16fee03c19d681b4f3527b6a750e1ed4d434c7",
    "DAMO-NLP-SG/MultiJail": "93d65c778925d55973b9ff30c7bccc8545494eb6",
    "natolambert/xstest-v2-copy": "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
}


def _first_category(raw: str | list | None) -> str:
    if not raw:
        return ""
    if isinstance(raw, list):
        return raw[0] if raw else ""
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return str(raw)
    return val[0] if isinstance(val, list) and val else str(val)


def _discard(discarded: dict | None, reason: str) -> None:
    if discarded is not None:
        discarded[reason] = discarded.get(reason, 0) + 1


def _text(raw, discarded: dict | None) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        _discard(discarded, "empty_text")
        return None
    return raw.strip()


def _has_id(row: dict, discarded: dict | None) -> bool:
    if row.get("id") is None or not str(row["id"]).strip():
        _discard(discarded, "missing_source_id")
        return False
    return True


def aya_redteaming_rows(lang: str, rows: Iterator[dict], *, discarded: dict | None = None) -> Iterator[Prompt]:
    """Rows from one aya_<lang>.jsonl file. Items are written natively per language, so groups are per language."""
    for i, r in enumerate(rows):
        text = _text(r.get("prompt"), discarded)
        if text is None:
            continue
        group = f"aya-{lang}-{i:05d}"
        yield Prompt(
            id=make_id(group, lang), group=group, lang=lang, kind="harmful", text=text,
            source="CohereLabs/aya_redteaming", source_id=f"{AYA_FILES[lang]}#{i}",
            category=_first_category(r.get("harm_category")), origin="native",
            meta={"global_or_local": r.get("global_or_local", "")},
        )


def multijail_rows(rows: Iterator[dict], langs: list[str], *, discarded: dict | None = None) -> Iterator[Prompt]:
    """MultiJail rows are parallel: one group per row, one Prompt per requested language."""
    for r in rows:
        if not _has_id(r, discarded):
            continue
        group = f"multijail-{r['id']}"
        for lang in langs:
            text = _text(r.get(MULTIJAIL_COLUMNS[lang]), discarded)
            if text is None:
                continue
            yield Prompt(
                id=make_id(group, lang), group=group, lang=lang, kind="harmful", text=text,
                source="DAMO-NLP-SG/MultiJail", source_id=str(r["id"]),
                category=_first_category(r.get("tags")), origin="native" if lang == "en" else "translated",
                meta={"upstream_source": r.get("source", "")},
            )


def xstest_rows(rows: Iterator[dict], *, discarded: dict | None = None) -> Iterator[Prompt]:
    """XSTest v2 prompts (English). Safe prompts become `benign`, contrast prompts become `harmful`."""
    for r in rows:
        if not _has_id(r, discarded):
            continue
        ptype = r.get("type")
        if not isinstance(ptype, str) or not ptype.strip():
            _discard(discarded, "missing_source_label")
            continue
        text = _text(r.get("prompt"), discarded)
        if text is None:
            continue
        kind = "harmful" if ptype.startswith(XSTEST_UNSAFE_PREFIX) else "benign"
        group = f"xstest-{r['id']}"
        yield Prompt(
            id=make_id(group, "en"), group=group, lang="en", kind=kind, text=text,
            source="natolambert/xstest-v2-copy", source_id=str(r["id"]), category=ptype, origin="native",
        )


# ---- network loaders (thin wrappers so the normalisers above stay testable offline) ----------------------------

def _revision(repo: str, revision: str | None) -> str:
    selected = revision or SOURCE_REVISIONS.get(repo)
    if selected and len(selected) == 40 and all(c in "0123456789abcdef" for c in selected):
        return selected
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo, revision=selected or "main").sha


def _download(repo: str, filename: str, revision: str, provenance: dict | None) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    if provenance is not None:
        provenance.update({"repo": repo, "revision": revision})
        provenance.setdefault("files", {})[filename] = {"sha256": sha256, "bytes": path.stat().st_size}
    return path, sha256


def _with_provenance(rows: Iterator[Prompt], repo: str, revision: str, sha256: str) -> Iterator[Prompt]:
    for p in rows:
        yield replace(p, source=repo, meta={**p.meta, "source_revision": revision, "source_sha256": sha256})


def load_aya_redteaming(lang: str, repo: str = "CohereLabs/aya_redteaming", *, revision: str | None = None,
                       provenance: dict | None = None) -> Iterator[Prompt]:
    revision = _revision(repo, revision)
    path, sha256 = _download(repo, AYA_FILES[lang], revision, provenance)
    with path.open(encoding="utf-8") as f:
        discarded = provenance.setdefault("discarded", {}) if provenance is not None else None
        rows = aya_redteaming_rows(lang, (json.loads(line) for line in f if line.strip()), discarded=discarded)
        yield from _with_provenance(rows, repo, revision, sha256)


def load_multijail(langs: list[str], repo: str = "DAMO-NLP-SG/MultiJail", *, revision: str | None = None,
                   provenance: dict | None = None) -> Iterator[Prompt]:
    revision = _revision(repo, revision)
    path, sha256 = _download(repo, "MultiJail.csv", revision, provenance)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        discarded = provenance.setdefault("discarded", {}) if provenance is not None else None
        rows = multijail_rows(csv.DictReader(stream), [lang for lang in langs if lang in MULTIJAIL_COLUMNS],
                              discarded=discarded)
        yield from _with_provenance(rows, repo, revision, sha256)


def load_xstest(repo: str = "natolambert/xstest-v2-copy", *, revision: str | None = None,
                provenance: dict | None = None) -> Iterator[Prompt]:
    from datasets import load_dataset

    revision = _revision(repo, revision)
    # Only the prompt split is needed; the other five splits contain model completions.
    path, sha256 = _download(repo, "data/prompts-00000-of-00001.parquet", revision, provenance)
    ds = load_dataset("parquet", data_files={"prompts": str(path)}, split="prompts")
    discarded = provenance.setdefault("discarded", {}) if provenance is not None else None
    yield from _with_provenance(xstest_rows(iter(ds), discarded=discarded), repo, revision, sha256)
