"""Build bounded, group-isolated repair and held-out prompt sets from public sources.

Translation uses the explicit spec ``translation`` configuration. Its client supplies a persistent
cache; accepted translations remain unreviewed until a separate human quality check exists.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, load_spec, spec_digest
from ..languages import AYA_FILES, LANGS, MULTIJAIL_COLUMNS
from ..provenance import Ledger
from .records import Prompt, make_id, read_prompts, write_jsonl
from .sources import load_aya_redteaming, load_multijail, load_xstest
from .splits import assign_splits, drop_leaks, leakage_report

OUT = ROOT / "data" / "processed"


def _out_dir() -> Path:
    """Resolve per call so different profiles can use the same Python process."""
    return Path(os.environ.get("LANGSAFE_DATA_DIR") or OUT).expanduser()


def _cap(prompts: list[Prompt], per_lang: int, seed: int) -> list[Prompt]:
    """Deterministic per-language cap with a salt independent of split assignment."""
    if per_lang < 0:
        raise ValueError("Per-language caps must be nonnegative")
    by_lang = defaultdict(list)
    for p in prompts:
        by_lang[p.lang].append(p)
    out = []
    for _, ps in sorted(by_lang.items()):
        ps.sort(key=lambda p: (hashlib.sha256(f"cap:{seed}:{p.group}".encode()).hexdigest(), p.id))
        out.extend(ps[:per_lang])
    return out


def _validate(prompts: list[Prompt], discarded: Counter, seen: dict[str, Prompt] | None = None) -> list[Prompt]:
    seen = seen if seen is not None else {}
    valid = []
    for p in prompts:
        if not isinstance(p.text, str) or not p.text.strip():
            discarded["empty_text"] += 1
            continue
        if not p.group or not p.lang or p.id != make_id(p.group, p.lang):
            raise ValueError(f"Invalid prompt identity: {p.id}")
        if p.kind not in {"harmful", "benign"}:
            raise ValueError(f"Invalid prompt kind for {p.id}")
        p = replace(p, text=p.text.strip())
        previous = seen.get(p.id)
        if previous is not None:
            identity = lambda row: (row.group, row.lang, row.kind, row.text, row.source, row.source_id)
            if identity(previous) != identity(p):
                raise ValueError(f"Conflicting duplicate prompt ID: {p.id}")
            discarded["duplicate_id"] += 1
            continue
        seen[p.id] = p
        valid.append(p)
    return valid


def _source_config(data: dict, key: str, default: str) -> tuple[str, str | None]:
    configured = data.get("sources", {}).get(key, default)
    if isinstance(configured, str):
        return configured, data.get("source_revisions", {}).get(key)
    if isinstance(configured, dict) and configured.get("repo"):
        return configured["repo"], configured.get("revision")
    raise ValueError(f"Invalid data source configuration: {key}")


def _source_report(report: dict, rows: list[Prompt]) -> None:
    """Also hash normalized records, so parsing changes are distinguishable from source changes."""
    digest = hashlib.sha256()
    for p in sorted(rows, key=lambda row: (row.id, row.text_hash)):
        digest.update(json.dumps(asdict(p), sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    report.update({"normalized_rows": len(rows), "normalized_sha256": digest.hexdigest()})


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _translation_config(spec: dict) -> dict:
    config = spec.get("translation")
    if not isinstance(config, dict) or not config.get("model") or not config.get("base_url"):
        raise ValueError("An explicit translation configuration with model and base_url is required")
    config = {"batch_size": 8, "max_tokens": 600, **config}
    if int(config["batch_size"]) < 1 or int(config["max_tokens"]) < 1:
        raise ValueError("translation batch_size and max_tokens must be positive")
    return config


def _coverage(prompts: list[Prompt], langs: list[str], limits: dict[str, int]) -> dict:
    counts = Counter((p.lang, p.kind) for p in prompts)
    split_counts = Counter((p.split, p.kind, p.lang) for p in prompts)
    missing = [f"{lang}/{kind}" for lang in langs for kind in limits if not counts[lang, kind]]
    missing_splits = [f"{split}/{kind}/{lang}" for split in ("heldout", "repair")
                      for lang in langs for kind in limits if not split_counts[split, kind, lang]]
    shortfalls = {f"{lang}/{kind}": limit - counts[lang, kind]
                  for lang in langs for kind, limit in limits.items() if counts[lang, kind] < limit}
    return {"complete": not missing and not missing_splits, "missing": missing,
            "missing_split_cells": missing_splits, "shortfalls": shortfalls,
            "at_configured_caps": not shortfalls,
            "counts": {f"{lang}/{kind}": counts[lang, kind] for lang in langs for kind in limits}}


def build(translate: bool = False, use_curatorkit: bool = True, translator: str = "router",
          family: str | None = None) -> dict:
    spec = load_spec()
    data = spec["data"]
    families = spec["languages"]
    if family is not None and family not in families:
        raise ValueError(f"Unknown language family: {family}")
    langs = sorted(set(families[family]) if family else {lang for values in families.values() for lang in values})
    if not langs or any(lang not in LANGS for lang in langs):
        raise ValueError("Selected family must contain supported languages")
    seed = int(data["seed"])
    limits = {"harmful": int(data["max_harmful_per_lang"]), "benign": int(data["max_benign_per_lang"])}
    if min(limits.values()) < 0 or not 0 < data["heldout_fraction"] < 1:
        raise ValueError("Invalid data caps or heldout_fraction")
    config = _translation_config(spec) if translate else None
    sources, discarded, seen = {}, Counter(), {}

    def read_source(key: str, loader, *args) -> list[Prompt]:
        report = sources.setdefault(key, {})
        rows = list(loader(*args, provenance=report))
        _source_report(report, rows)
        discarded.update(report.get("discarded", {}))
        return _validate(rows, discarded, seen)

    aya_repo, aya_revision = _source_config(data, "aya_redteaming", "CohereLabs/aya_redteaming")
    native = []
    for lang in sorted(set(langs) & set(AYA_FILES)):
        # Bind keyword settings here while keeping loaders' existing positional APIs intact.
        rows = read_source(f"aya_redteaming/{lang}",
                           lambda language, **kw: load_aya_redteaming(language, repo=aya_repo,
                                                                     revision=aya_revision, **kw), lang)
        native.extend(rows)
    mj_repo, mj_revision = _source_config(data, "multijail", "DAMO-NLP-SG/MultiJail")
    mj_langs = sorted((set(langs) & set(MULTIJAIL_COLUMNS)) | ({"en"} if translate else set()))
    multijail = read_source("multijail", lambda requested, **kw: load_multijail(
        requested, repo=mj_repo, revision=mj_revision, **kw), mj_langs) if mj_langs else []
    native.extend(multijail)
    xs_repo, xs_revision = _source_config(data, "xstest", "natolambert/xstest-v2-copy")
    xstest = read_source("xstest", lambda **kw: load_xstest(repo=xs_repo, revision=xs_revision, **kw)) \
        if "en" in langs or translate else []
    native.extend(xstest)
    relevant = [p for p in native if p.lang in langs]
    discarded["unselected_language"] += len(native) - len(relevant)
    pool = [p for kind, limit in limits.items() for p in _cap([p for p in relevant if p.kind == kind], limit, seed)]
    discarded["cap"] += len(relevant) - len(pool)
    selected_counts = Counter((p.lang, p.kind) for p in pool)

    translation_report = {"enabled": translate, "status": "disabled", "requested": 0,
                          "completed": 0, "failed": 0, "quality_status": "not_applicable"}
    failed, diagnostics = [], []
    if translate:
        from ..llm_client import make_client
        from .translate import translate_all

        translation_report.update({"model": config["model"], "base_url": config["base_url"],
                                   "batch_size": int(config["batch_size"]), "max_tokens": int(config["max_tokens"]),
                                   "quant": config.get("quant", "auto") if config["base_url"] == "local" else None,
                                   "quality_status": "unreviewed", "status": "not_needed"})
        existing = {p.id for p in pool}
        seeds = {"harmful": [p for p in multijail if p.lang == "en"],
                 "benign": [p for p in xstest if p.kind == "benign" and p.lang == "en"]}
        jobs = {}
        for lang in langs:
            if lang == "en":
                continue
            selected = []
            for kind, limit in limits.items():
                available = [p for p in seeds[kind] if make_id(p.group, lang) not in existing]
                deficit = max(0, limit - selected_counts[lang, kind])
                selected.extend(_cap(available, deficit, seed))
            if selected:
                jobs[lang] = selected
        translation_report["requested"] = sum(len(rows) for rows in jobs.values())
        if jobs:
            client = None
            try:
                kwargs = {}
                if "cache_path" in config:
                    kwargs["cache_path"] = Path(config["cache_path"]).expanduser()
                if config["base_url"] == "local":
                    kwargs.update({"batch_size": int(config["batch_size"]), "quant": config.get("quant", "auto")})
                client = make_client(config["base_url"], config["model"], **kwargs)
                cache_identity = getattr(getattr(client, "_cache", None), "model_key", None)
                if cache_identity is not None:
                    translation_report["cache_identity"] = cache_identity
                for lang, rows in jobs.items():
                    translated, failures = translate_all(client, rows, [lang], batch_size=int(config["batch_size"]),
                                                        max_tokens=int(config["max_tokens"]), diagnostics=diagnostics)
                    pool.extend(translated)
                    failed.extend(failures)
            except Exception as exc:
                if client is not None:
                    raise
                for lang, rows in jobs.items():
                    for p in rows:
                        target_id = make_id(p.group, lang)
                        failed.append(target_id)
                        diagnostics.append({"id": target_id, "source_id": p.id, "target": lang,
                                            "reason": "client_initialization_error", "error_type": type(exc).__name__})
            finally:
                if client is not None:
                    client.close()
            translation_report["failed"] = len(failed)
            translation_report["completed"] = translation_report["requested"] - len(failed)
            translation_report["status"] = "complete" if not failed else (
                "failed" if not translation_report["completed"] else "partial_failure")
        translation_report["failures"] = diagnostics

    # This check runs before CuratorKIT: it must never receive duplicate identities or an unbounded pool.
    pool = _validate(pool, discarded)
    curation = None
    before_curation = len(pool)
    if use_curatorkit and pool:
        from .curate import curate

        pool, curation = curate(pool)
    discarded["curation"] += before_curation - len(pool)
    pool = _validate(pool, discarded)
    all_prompts = assign_splits(pool, data["heldout_fraction"], seed)
    initial = leakage_report(all_prompts)
    before_leaks = list(all_prompts)
    if not initial["ok"]:
        all_prompts = drop_leaks(all_prompts, initial)
    final = leakage_report(all_prompts)
    if not final["ok"]:
        raise ValueError("Split leakage remains after repair-group removal")
    discarded["leakage"] += len(before_leaks) - len(all_prompts)
    leaked_groups = {p.group for p in before_leaks} - {p.group for p in all_prompts}

    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("heldout", "repair"):
        rows = sorted((p for p in all_prompts if p.split == split), key=lambda p: p.id)
        path = out / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[split] = {"path": _path_label(path), "rows": len(rows),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    counts = Counter((p.split, p.kind, p.lang) for p in all_prompts)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(), "spec_digest": spec_digest(spec), "seed": seed,
        "family": family, "languages": langs, "translated": translate,
        "translator": ("local" if config["base_url"] == "local" else "router") if config else None,
        "translation": translation_report, "translation_failures": failed,
        "curatorkit": curation, "files": files,
        "counts": {f"{split}/{kind}/{lang}": count for (split, kind, lang), count in sorted(counts.items())},
        "coverage": _coverage(all_prompts, langs, limits), "discarded": dict(sorted(discarded.items())),
        "validation": {"unique_ids": len({p.id for p in all_prompts}) == len(all_prompts),
                       "nonempty_text": all(bool(p.text.strip()) for p in all_prompts)},
        "leakage_check": {"ok": final["ok"], "near_dup_jaccard": 0.8,
                          "initial_groups_in_both": len(initial["groups_in_both"]),
                          "initial_exact_dupes": len(initial["exact_dupes"]),
                          "initial_near_dupes": len(initial["near_dupes"]),
                          "removed_rows": len(before_leaks) - len(all_prompts),
                          "removed_groups": len(leaked_groups)},
        "sources": data.get("sources", {}), "source_provenance": sources,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    Ledger.default().append("build-data", {"manifest": manifest})
    return manifest


def load_split(split: str) -> list[Prompt]:
    if split not in {"heldout", "repair"}:
        raise ValueError("split must be heldout or repair")
    return list(read_prompts(_out_dir() / f"{split}.jsonl"))
