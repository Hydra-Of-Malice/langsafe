#!/usr/bin/env python3
"""Prepare or validate private translation-review packets.

The command prints aggregate status only; packet CSVs contain prompt text and must
remain in the private run directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langsafe.data.review import export_translation_review, import_translation_review


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="directory containing manifest.json, repair.jsonl and heldout.jsonl")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="manifest path (defaults to DATA_DIR/manifest.json)")
    parser.add_argument("--source-reference", type=Path, default=None,
                        help="optional private JSONL source index for capped-out English rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare/validate private translation review evidence")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write a blank translation-review CSV")
    _common(export)
    export.add_argument("--out", type=Path, required=True, help="private CSV output path")
    check = sub.add_parser("check", help="validate reviewer edits and summarize quality status")
    _common(check)
    check.add_argument("--packet", type=Path, required=True, help="review CSV path")
    check.add_argument("--out", type=Path, default=None, help="optional JSON summary path")
    args = parser.parse_args()
    if args.command == "export":
        result = export_translation_review(args.data_dir, args.manifest, args.out, args.source_reference)
    else:
        manifest = args.manifest or args.data_dir / "manifest.json"
        result = import_translation_review(args.packet, manifest, args.data_dir, args.out, args.source_reference)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
