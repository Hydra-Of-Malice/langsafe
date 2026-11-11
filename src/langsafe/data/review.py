"""Private translation-review packets and fail-closed import checks.

Translation outputs are evidence for a benchmark only after a person has reviewed
each source/translation pair.  This module prepares a CSV that contains the text a
reviewer needs, but it never fills a label, changes the data manifest, or treats a
machine-produced field as a human decision.  Imported summaries are safe to feed to
an orchestration layer; the raw packet remains private.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


REVIEW_VERSION = "translation-review-v1"
REVIEW_LABELS = ("accepted", "rejected", "unclear")
_LABEL_ALIASES = {"accept": "accepted", "accepted": "accepted",
                  "reject": "rejected", "rejected": "rejected",
                  "unclear": "unclear"}
_MACHINE_REVIEWER_IDS = {"auto", "automated", "machine", "model", "llm", "system", "unknown"}
_DATA_FILES = ("repair", "heldout")
_TRANSLATED_ORIGINS = {"translated", "code-mixed"}
_LABEL_COLUMNS = ("review_label", "translation_label", "label")
_REVIEWER_COLUMNS = ("reviewer_id", "reviewer")


class TranslationReviewError(ValueError):
    """Raised when a packet is stale, incomplete, or cannot be tied to the dataset."""


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_manifest(manifest_path: Path) -> tuple[dict, str]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise TranslationReviewError(f"manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationReviewError(f"could not read manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise TranslationReviewError("manifest must contain a files mapping")
    return manifest, _sha256_bytes(manifest_path)


def _data_path(data_dir: Path, manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    candidates = [data_dir / path.name, manifest_path.parent / path.name, Path.cwd() / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TranslationReviewError(f"manifest data file does not exist: {value}")


def _load_dataset(data_dir: Path, manifest_path: Path, manifest: dict) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    file_hashes: dict[str, str] = {}
    for split in _DATA_FILES:
        info = manifest["files"].get(split)
        if not isinstance(info, dict) or not info.get("path") or not info.get("sha256"):
            raise TranslationReviewError(f"manifest has no hashed {split} data file")
        path = _data_path(data_dir, manifest_path, str(info["path"]))
        actual = _sha256_bytes(path)
        if actual != info["sha256"]:
            raise TranslationReviewError(f"{split} data file hash does not match manifest")
        file_hashes[split] = actual
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TranslationReviewError(f"invalid {split} JSONL row {line_number}") from exc
                    if not isinstance(row, dict) or not row.get("id"):
                        raise TranslationReviewError(f"invalid {split} record at row {line_number}")
                    row = {**row, "split": row.get("split") or split}
                    if row["split"] != split:
                        raise TranslationReviewError(f"{split} record has mismatched split at row {line_number}")
                    rows.append(row)
        except OSError as exc:
            raise TranslationReviewError(f"could not read data file: {path}") from exc
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            raise TranslationReviewError(f"duplicate dataset prompt id: {row['id']}")
        by_id[row["id"]] = row
    return rows, file_hashes


def _load_source_reference(path: Path | None) -> tuple[dict[str, dict], str, int]:
    """Load an optional private JSONL source index for rows excluded by dataset caps."""
    if path is None:
        return {}, "", 0
    path = Path(path)
    if not path.is_file():
        raise TranslationReviewError(f"source reference does not exist: {path}")
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("id") or not isinstance(row.get("text"), str):
                    raise TranslationReviewError(f"invalid source reference row {line_number}")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise TranslationReviewError(f"could not read source reference: {path}") from exc
    by_id = {}
    for row in rows:
        if row["id"] in by_id:
            raise TranslationReviewError(f"duplicate source reference id: {row['id']}")
        by_id[row["id"]] = row
    return by_id, _sha256_bytes(path), len(rows)


def _is_translation(row: dict) -> bool:
    return row.get("origin") in _TRANSLATED_ORIGINS or bool(row.get("meta", {}).get("translated_from"))


def _source_id(row: dict) -> str:
    meta = row.get("meta") or {}
    explicit = meta.get("translated_from")
    if explicit:
        return str(explicit)
    return f"{row.get('group', '')}::en"


def _review_key(manifest_sha256: str, row: dict, source: dict | None,
                source_reference_sha256: str = "") -> str:
    source_text = str((source or {}).get("text") or "")
    target_text = str(row.get("text") or "")
    return _digest({"version": REVIEW_VERSION, "manifest_sha256": manifest_sha256,
                    "source_reference_sha256": source_reference_sha256,
                    "target_id": row.get("id"), "source_id": _source_id(row),
                    "source_sha256": _sha256_text(source_text) if source else "",
                    "target_sha256": _sha256_text(target_text)})


def _row_for_review(manifest_sha256: str, row: dict, by_id: dict[str, dict], file_hashes: dict[str, str],
                    source_reference_sha256: str = "") -> dict:
    source_id = _source_id(row)
    source = by_id.get(source_id)
    source_text = str((source or {}).get("text") or "")
    target_text = str(row.get("text") or "")
    source_hash = _sha256_text(source_text) if source else ""
    target_hash = _sha256_text(target_text)
    return {
        "review_version": REVIEW_VERSION,
        "manifest_sha256": manifest_sha256,
        "repair_sha256": file_hashes.get("repair", ""),
        "heldout_sha256": file_hashes.get("heldout", ""),
        "review_key": _review_key(manifest_sha256, row, source, source_reference_sha256),
        "source_reference_sha256": source_reference_sha256,
        "target_id": row.get("id", ""),
        "source_id": source_id,
        "source_lang": (source or {}).get("lang", "en" if source else ""),
        "target_lang": row.get("lang", ""),
        "kind": row.get("kind", ""),
        "split": row.get("split", ""),
        "source_present": "yes" if source else "no",
        "source_sha256": source_hash,
        "target_sha256": target_hash,
        "source_text": source_text,
        "translation_text": target_text,
        "target_origin": row.get("origin", ""),
        "translator": (row.get("meta") or {}).get("translator", ""),
        "translation_quality": (row.get("meta") or {}).get("translation_quality", "unreviewed"),
        "review_label": "",
        "translation_label": "",
        "reviewer_id": "",
        "reviewer": "",
        "reviewed_at": "",
        "review_notes": "",
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_translation_review(data_dir: Path, manifest_path: Path | None = None,
                             out_path: Path | None = None,
                             source_reference: Path | None = None) -> dict:
    """Write a private CSV packet for every translated/code-mixed dataset record.

    No label or reviewer field is populated.  A missing English source is retained
    as an explicit blocked row so a packet cannot silently omit hard cases.
    """
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path) if manifest_path is not None else data_dir / "manifest.json"
    manifest, manifest_sha256 = _read_manifest(manifest_path)
    rows, file_hashes = _load_dataset(data_dir, manifest_path, manifest)
    references, source_reference_sha256, source_reference_rows = _load_source_reference(source_reference)
    by_id = {**references, **{row["id"]: row for row in rows}}
    review_rows = [_row_for_review(manifest_sha256, row, by_id, file_hashes, source_reference_sha256)
                   for row in rows if _is_translation(row)]
    review_rows.sort(key=lambda row: row["review_key"])
    if len({row["review_key"] for row in review_rows}) != len(review_rows):
        raise TranslationReviewError("duplicate translation review identities")
    packet_id = _digest({"version": REVIEW_VERSION, "manifest_sha256": manifest_sha256,
                         "rows": [row["review_key"] for row in review_rows]})
    summary = {
        "review_version": REVIEW_VERSION,
        "packet_id": packet_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "data_files": file_hashes,
        "source_reference_path": str(source_reference) if source_reference else None,
        "source_reference_sha256": source_reference_sha256,
        "source_reference_rows": source_reference_rows,
        "rows": len(review_rows),
        "source_available": sum(row["source_present"] == "yes" for row in review_rows),
        "missing_source": sum(row["source_present"] != "yes" for row in review_rows),
        "by_lang": dict(sorted(Counter(row["target_lang"] for row in review_rows).items())),
        "labels_supplied": 0,
        "quality_status": "unreviewed" if review_rows else "not_applicable",
        "complete": not review_rows,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fields = list(review_rows[0]) if review_rows else [
            "review_version", "manifest_sha256", "review_key", "source_reference_sha256", "target_id", "source_id",
            "source_lang", "target_lang", "kind", "split", "source_present", "source_sha256",
            "target_sha256", "source_text", "translation_text", "target_origin", "translator",
            "translation_quality", "review_label", "translation_label", "reviewer_id", "reviewer",
            "reviewed_at", "review_notes",
        ]
        with out_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(review_rows)
        summary["packet_path"] = str(out_path)
        summary["summary_path"] = str(out_path.with_suffix(".summary.json"))
        _write_json(out_path.with_suffix(".summary.json"), summary)
    return summary


def _normalise_label(value: str) -> str:
    value = str(value or "").strip().casefold()
    return _LABEL_ALIASES.get(value, value)


def _valid_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return parsed.tzinfo is not None


def _read_packet(packet_path: Path) -> list[dict]:
    try:
        with Path(packet_path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise TranslationReviewError("translation review packet has no header")
            required = {"review_version", "manifest_sha256", "repair_sha256", "heldout_sha256", "review_key",
                        "source_reference_sha256", "target_id", "source_id",
                        "source_present", "source_sha256", "target_sha256", "source_text",
                        "translation_text", "reviewed_at"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise TranslationReviewError(f"translation review packet is missing columns: {sorted(missing)}")
            if not any(column in reader.fieldnames for column in _LABEL_COLUMNS):
                raise TranslationReviewError("translation review packet is missing a review label column")
            if not any(column in reader.fieldnames for column in _REVIEWER_COLUMNS):
                raise TranslationReviewError("translation review packet is missing a reviewer column")
            return list(reader)
    except OSError as exc:
        raise TranslationReviewError(f"could not read translation review packet: {packet_path}") from exc


def import_translation_review(packet_path: Path, manifest_path: Path | None = None,
                              data_dir: Path | None = None, out_path: Path | None = None,
                              source_reference: Path | None = None) -> dict:
    """Validate reviewer edits against the exact manifest and return a quality summary.

    This function never writes labels into dataset records.  ``complete`` only means
    all source-available rows have a valid label and all source-missing rows remain
    visibly blocked; ``quality_status == human_reviewed`` additionally requires every
    row to be accepted.  Reviewer identity/timestamp fields are required for every
    non-blank label and are evidence supplied by the reviewer, not generated here.
    """
    packet_path = Path(packet_path)
    packet = _read_packet(packet_path)
    if manifest_path is None:
        raw = packet[0].get("manifest_path") if packet else None
        manifest_path = Path(raw) if raw else None
    if manifest_path is None:
        raise TranslationReviewError("manifest_path is required to import a translation review")
    manifest_path = Path(manifest_path)
    data_dir = Path(data_dir) if data_dir is not None else manifest_path.parent
    manifest, manifest_sha256 = _read_manifest(manifest_path)
    rows, file_hashes = _load_dataset(data_dir, manifest_path, manifest)
    references, source_reference_sha256, source_reference_rows = _load_source_reference(source_reference)
    by_id = {**references, **{row["id"]: row for row in rows}}
    expected = {}
    for dataset_row in rows:
        if _is_translation(dataset_row):
            expected_row = _row_for_review(manifest_sha256, dataset_row, by_id, file_hashes,
                                           source_reference_sha256)
            expected[expected_row["review_key"]] = expected_row
    if not packet and expected:
        raise TranslationReviewError("translation review packet contains no rows")
    supplied = {}
    for line, row in enumerate(packet, start=2):
        if row.get("review_version") != REVIEW_VERSION:
            raise TranslationReviewError(f"packet row {line} has unsupported review_version")
        if row.get("manifest_sha256") != manifest_sha256:
            raise TranslationReviewError(f"packet row {line} belongs to a different manifest")
        key = row.get("review_key", "")
        if key in supplied:
            raise TranslationReviewError(f"duplicate review_key at packet row {line}")
        target = expected.get(key)
        if target is None:
            raise TranslationReviewError(f"packet row {line} is absent from the current dataset")
        for field in ("target_id", "source_id", "source_lang", "target_lang", "kind", "split",
                      "source_present", "source_sha256", "target_sha256", "target_origin",
                      "source_reference_sha256", "repair_sha256", "heldout_sha256", "translator",
                      "translation_quality"):
            if str(row.get(field, "")) != str(target.get(field, "")):
                raise TranslationReviewError(f"packet row {line} has stale {field}")
        if row.get("source_text", "") != target.get("source_text", ""):
            raise TranslationReviewError(f"packet row {line} has stale source_text")
        if row.get("translation_text", "") != target.get("translation_text", ""):
            raise TranslationReviewError(f"packet row {line} has stale translation_text")
        raw_labels = {str(row.get(column, "")).strip() for column in _LABEL_COLUMNS if column in row and str(row.get(column, "")).strip()}
        if len({_normalise_label(value) for value in raw_labels}) > 1:
            raise TranslationReviewError(f"packet row {line} has conflicting review labels")
        label = _normalise_label(next(iter(raw_labels), ""))
        if label and label not in REVIEW_LABELS:
            raise TranslationReviewError(f"packet row {line} has invalid review_label")
        raw_reviewers = {str(row.get(column, "")).strip() for column in _REVIEWER_COLUMNS if column in row and str(row.get(column, "")).strip()}
        if len(raw_reviewers) > 1:
            raise TranslationReviewError(f"packet row {line} has conflicting reviewer identities")
        reviewer = next(iter(raw_reviewers), "")
        timestamp = str(row.get("reviewed_at", "")).strip()
        if label:
            if not reviewer or reviewer.casefold() in _MACHINE_REVIEWER_IDS:
                raise TranslationReviewError(f"packet row {line} needs a human reviewer_id")
            if not _valid_timestamp(timestamp):
                raise TranslationReviewError(f"packet row {line} needs an ISO-8601 reviewed_at")
            if target["source_present"] != "yes":
                raise TranslationReviewError(f"packet row {line} cannot be labelled without its source text")
        elif reviewer or timestamp:
            raise TranslationReviewError(f"packet row {line} has reviewer metadata without a review_label")
        supplied[key] = {**target, "review_label": label, "reviewer_id": reviewer, "reviewed_at": timestamp}
    missing = set(expected) - set(supplied)
    source_missing = sum(row["source_present"] != "yes" for row in expected.values())
    labelled = [row for row in supplied.values() if row["review_label"]]
    labels = Counter(row["review_label"] for row in labelled)
    reviewer_ids = sorted({row["reviewer_id"] for row in labelled})
    complete = not expected or (not missing and source_missing == 0 and len(labelled) == len(expected))
    quality_status = "not_applicable" if not expected else "unreviewed"
    if complete:
        quality_status = "human_reviewed" if labels["accepted"] == len(expected) else "human_reviewed_with_rejections"
    summary = {
        "review_version": REVIEW_VERSION,
        "packet_path": str(packet_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "data_files": file_hashes,
        "source_reference_path": str(source_reference) if source_reference else None,
        "source_reference_sha256": source_reference_sha256,
        "source_reference_rows": source_reference_rows,
        "packet_id": _digest({"version": REVIEW_VERSION, "manifest_sha256": manifest_sha256,
                              "rows": sorted(expected)}),
        "rows": len(expected), "packet_rows": len(packet), "labels_supplied": len(labelled),
        "missing_rows": len(missing), "source_available": len(expected) - source_missing,
        "source_missing": source_missing,
        "label_counts": {label: labels[label] for label in REVIEW_LABELS},
        "reviewer_ids": reviewer_ids,
        "complete": complete,
        "quality_status": quality_status,
        "evidence": {"reviewer_metadata_present": bool(labelled and reviewer_ids),
                     "all_labelled_rows_have_reviewer_metadata": all(bool(row["reviewer_id"] and row["reviewed_at"])
                                                                       for row in labelled)},
    }
    if out_path is not None:
        out_path = Path(out_path)
        summary["summary_path"] = str(out_path)
        _write_json(out_path, summary)
    return summary


# Explicit aliases make the packet workflow discoverable without coupling callers to
# the CLI (which intentionally remains unchanged for this offline preparation step).
build_translation_review_packet = export_translation_review
check_translation_review = import_translation_review
validate_translation_review = import_translation_review
