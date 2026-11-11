"""Offline evidence checks for translation-review packet preparation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from langsafe.data.review import (TranslationReviewError, export_translation_review,
                                  import_translation_review)
from langsafe.eval.judge_run import compare_judge_sets, evaluation_digest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(tmp_path: Path, *, with_missing: bool = True) -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    source = {"id": "g::en", "group": "g", "lang": "en", "kind": "harmful",
              "text": "source wording", "origin": "native", "split": "repair", "meta": {}}
    target = {"id": "g::hi", "group": "g", "lang": "hi", "kind": "harmful",
              "text": "translated wording", "origin": "translated", "split": "repair",
              "meta": {"translated_from": "g::en", "translator": "offline-fixture",
                       "translation_quality": "unreviewed"}}
    rows = [source, target]
    if with_missing:
        rows.append({"id": "missing::hi", "group": "missing", "lang": "hi", "kind": "harmful",
                     "text": "orphan translation", "origin": "code-mixed", "split": "heldout",
                     "meta": {"translated_from": "missing::en", "translation_quality": "unreviewed"}})
    by_split = {"repair": [row for row in rows if row["split"] == "repair"],
                "heldout": [row for row in rows if row["split"] == "heldout"]}
    files = {}
    for split, split_rows in by_split.items():
        path = data / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows), encoding="utf-8")
        files[split] = {"path": str(path), "rows": len(split_rows), "sha256": _sha(path)}
    manifest = data / "manifest.json"
    manifest.write_text(json.dumps({"files": files, "translation": {"quality_status": "unreviewed"}}, indent=2),
                        encoding="utf-8")
    return data, manifest


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_export_includes_orphan_translations_without_claiming_review(tmp_path):
    data, manifest = _dataset(tmp_path)
    packet = tmp_path / "translation_review.csv"
    summary = export_translation_review(data, manifest, packet)
    assert summary["rows"] == 2
    assert summary["source_available"] == 1
    assert summary["missing_source"] == 1
    assert summary["quality_status"] == "unreviewed"
    rows = _read(packet)
    assert {row["review_label"] for row in rows} == {""}
    assert {row["source_present"] for row in rows} == {"yes", "no"}


def test_import_never_unlocks_when_source_is_missing_or_label_is_blank(tmp_path):
    data, manifest = _dataset(tmp_path)
    packet = tmp_path / "translation_review.csv"
    export_translation_review(data, manifest, packet)
    summary = import_translation_review(packet, manifest, data)
    assert summary["labels_supplied"] == 0
    assert summary["missing_rows"] == 0
    assert summary["source_missing"] == 1
    assert summary["complete"] is False
    assert summary["quality_status"] == "unreviewed"


def test_import_requires_reviewer_identity_and_reports_human_acceptance(tmp_path):
    data, manifest = _dataset(tmp_path, with_missing=False)
    packet = tmp_path / "translation_review.csv"
    export_translation_review(data, manifest, packet)
    rows = _read(packet)
    rows[0]["review_label"] = "accept"
    with packet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(TranslationReviewError, match="human reviewer_id"):
        import_translation_review(packet, manifest, data)
    rows[0]["reviewer_id"] = "reviewer-1"
    rows[0]["reviewed_at"] = "2026-11-04T10:00:00+00:00"
    with packet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = import_translation_review(packet, manifest, data)
    assert summary["labels_supplied"] == 1
    assert summary["complete"] is True
    assert summary["quality_status"] == "human_reviewed"
    assert summary["reviewer_ids"] == ["reviewer-1"]


def test_import_rejects_edited_translation_even_when_review_key_is_preserved(tmp_path):
    data, manifest = _dataset(tmp_path, with_missing=False)
    packet = tmp_path / "translation_review.csv"
    export_translation_review(data, manifest, packet)
    rows = _read(packet)
    rows[0]["translation_text"] = "edited after export"
    rows[0]["review_label"] = "accepted"
    rows[0]["reviewer_id"] = "reviewer-1"
    rows[0]["reviewed_at"] = "2026-11-04T10:00:00+00:00"
    with packet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(TranslationReviewError, match="translation_text"):
        import_translation_review(packet, manifest, data)


def test_import_rejects_machine_identity_and_manifest_drift(tmp_path):
    data, manifest = _dataset(tmp_path, with_missing=False)
    packet = tmp_path / "translation_review.csv"
    export_translation_review(data, manifest, packet)
    rows = _read(packet)
    rows[0].update({"review_label": "accepted", "reviewer_id": "model",
                    "reviewed_at": "2026-11-04T10:00:00+00:00"})
    with packet.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(TranslationReviewError, match="human reviewer_id"):
        import_translation_review(packet, manifest, data)
    # The data file is evidence-bound; changing it without rebuilding the manifest is stale.
    repair = data / "repair.jsonl"
    repair.write_text(repair.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(TranslationReviewError, match="hash does not match"):
        import_translation_review(packet, manifest, data)


def test_calibration_digest_changes_when_prompt_changes_under_same_generation_key():
    row = {"key": "backend-key", "prompt_sha256": "prompt-a", "response_sha256": "response-a",
           "judges": {"one": "full_refusal", "two": "full_refusal"}, "label": "full_refusal",
           "rubric_hash": "rubric"}
    changed = {**row, "prompt_sha256": "prompt-b"}
    assert evaluation_digest([row]) != evaluation_digest([changed])


def test_router_alignment_does_not_match_same_key_with_changed_completion():
    local = [{"key": "backend-key", "lang": "en", "label": "full_refusal", "judge_set": "local",
              "judges": {"one": "full_refusal", "two": "full_refusal"}, "rubric_hash": "rubric",
              "prompt_sha256": "prompt", "response_sha256": "response-a", "parse_error": False}]
    router = [{**local[0], "judge_set": "router", "response_sha256": "response-b"}]
    result = compare_judge_sets(local, router)
    assert result["verified"] is False
    assert result["shared_rows"] == 0


def test_calibration_digest_binds_the_disagreement_cap():
    row = {"key": "backend-key", "prompt_sha256": "prompt-a", "response_sha256": "response-a",
           "judges": {"one": "full_refusal", "two": "full_refusal"}, "label": "full_refusal",
           "rubric_hash": "rubric"}
    assert evaluation_digest([{**row, "disagreement_cap": 10}]) != evaluation_digest([{**row, "disagreement_cap": 20}])


def test_rows_judged_before_the_cap_keep_their_stamped_digest():
    # This value comes from the code before the cap existed. Summaries already on disk carry it, so a calibration
    # imported today has to reproduce it or the gate would call that calibration a different evaluation.
    row = {"key": "backend-key", "prompt_sha256": "prompt-a", "response_sha256": "response-a",
           "judges": {"one": "full_refusal", "two": "full_refusal"}, "label": "full_refusal",
           "rubric_hash": "rubric"}
    assert evaluation_digest([row]) == "45f8a8ca80f9132daf770d5e27ec736b3cd0c5fe788e45bd44821bbcc8830866"
