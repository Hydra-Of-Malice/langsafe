"""Load the pre-registered spec (configs/spec.yaml)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "configs" / "spec.yaml"


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or os.environ.get("LANGSAFE_SPEC", SPEC_PATH)).expanduser()
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"Experiment spec must be a mapping: {path}")
    return spec


def capability_languages(spec: dict, family: str | None = None) -> list[str]:
    """The family's benchmark languages, plus stand-ins for languages the benchmark doesn't have."""
    languages = [lang for lang in spec["capability_languages"] if family is None or lang in spec["languages"][family]]
    return list(dict.fromkeys(languages + spec.get("capability_proxies", {}).get(family, [])))


def spec_digest(spec: dict[str, Any]) -> str:
    """Stable hash of the spec, stamped on every run so results can be traced to the exact settings used."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
