"""Tamper-evident run ledger (append-only, sha256-chained JSONL), plus Lexsi provenance export.

Every step that produces a result (data build, generation, judging, circuit discovery, repair, quantisation) appends
an entry. `verify()` fails at the first entry that was edited, deleted or reordered, so a reviewer can trust that the
published numbers came from the logged runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "Ledger":
        return cls(Path(os.environ.get("LANGSAFE_RUNS_DIR", ROOT / "runs")) / "ledger.jsonl")

    def _last(self) -> tuple[int, str]:
        if not self.path.exists():
            return -1, "0" * 64
        last = None
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
        return (last["seq"], last["hash"]) if last else (-1, "0" * 64)

    def append(self, step: str, payload: dict) -> dict:
        seq, prev = self._last()
        body = {"seq": seq + 1, "prev_hash": prev, "step": step, "at": datetime.now(timezone.utc).isoformat(),
                "git": _git_rev(), "host": platform.node(), "payload": payload}
        body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
        return body

    def verify(self) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq)."""
        prev = "0" * 64
        if not self.path.exists():
            return True, None
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(l for l in f if l.strip()):
                e = json.loads(line)
                h = e.pop("hash")
                if e["seq"] != i or e["prev_hash"] != prev:
                    return False, i
                if hashlib.sha256(json.dumps(e, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != h:
                    return False, i
                prev = h
        return True, None


def write_lexsi_provenance(out_dir: Path, tool: str, inputs: list[dict], outputs: dict, params: dict) -> Path:
    """Write a `lexsi_provenance.json` next to an artifact so the Lexsi kits (and reviewers) can chain it."""
    doc = {"schema": "lexsi.provenance/1", "tool": f"langsafe:{tool}", "created_at": datetime.now(timezone.utc).isoformat(),
           "git": _git_rev(), "inputs": inputs, "outputs": outputs, "params": params}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "lexsi_provenance.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
