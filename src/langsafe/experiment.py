"""Experiment inputs and resumable, content-verified stage state.

The CLI owns model jobs. This module keeps state independent of an accelerator so
recovery and split isolation can be verified on a CPU.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .data.records import Prompt


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def capability_path(runs: Path, tag: str, preamble: str) -> Path:
    if preamble not in {'default', 'none'}:
        raise ValueError(f'Unsupported capability preamble: {preamble}')
    suffix = '' if preamble == 'default' else '.none'
    return runs / 'capability' / f'{tag}{suffix}.json'


def checkpoint_identity(model: str) -> dict:
    """Pin Hub commits; hash complete local checkpoint inputs."""
    from .gen.model import resolve_model
    resolved = resolve_model(model)
    path = Path(resolved)
    if path.is_dir():
        files = {p.name: file_sha256(p) for p in sorted(path.iterdir()) if p.is_file()
                 and p.suffix in {'.json', '.safetensors', '.model', '.jinja'}}
        if 'config.json' not in files or not any(name.endswith('.safetensors') for name in files):
            raise ValueError(f'Incomplete local checkpoint: {path}')
        return {'model': model, 'files': files}
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(resolved, 'config.json')
    if not isinstance(cached, str):
        raise ValueError(f'Model must be downloaded before evaluation: {resolved}')
    return {'model': resolved, 'revision': Path(cached).parent.name}


def select_prompts(prompts: list[Prompt], languages: list[str], split: str,
                   limit: int | None = None) -> list[Prompt]:
    if split not in {"repair", "heldout"}:
        raise ValueError("split must be repair or heldout")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if limit is not None:     # hash by group, so a limited sample mixes sources and stays parallel across languages
        prompts = sorted(prompts, key=lambda p: (hashlib.sha256(f'sample:{p.group}'.encode()).hexdigest(), p.id))
    seen: dict[tuple[str, str], int] = {}
    selected = []
    for prompt in prompts:
        if prompt.split != split or prompt.lang not in languages:
            continue
        key = (prompt.lang, prompt.kind)
        if limit is not None and seen.get(key, 0) >= limit:
            continue
        seen[key] = seen.get(key, 0) + 1
        selected.append(prompt)
    return selected


class ExperimentState:
    """Only skip stages whose settings and output-file hashes still match."""

    def __init__(self, path: Path, identity: dict):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("identity") != identity:
                raise ValueError("Experiment identity changed; use a separate experiment directory")
        else:
            self.data = {"identity": identity, "stages": {}}

    def completed(self, stage: str, settings: dict) -> bool:
        record = self.data["stages"].get(stage)
        if not record or record["settings"] != settings or not record.get("artifacts"):
            return False
        for directory, names in record.get('directories', {}).items():
            path = Path(directory)
            if not path.is_dir() or sorted(str(p.resolve()) for p in path.rglob('*') if p.is_file()) != names:
                return False
        for item in record["artifacts"]:
            path = Path(item["path"])
            if not path.is_file() or file_sha256(path) != item["sha256"]:
                return False
        return True

    def record(self, stage: str, settings: dict, artifacts: list[Path], result: dict | None = None) -> None:
        if not artifacts:
            raise ValueError("A completed stage must have verifiable artifacts")
        files = []
        directories = {}
        for path in artifacts:
            path = Path(path)
            if path.is_dir():
                members = sorted(p for p in path.rglob('*') if p.is_file())
                directories[str(path.resolve())] = sorted(str(p.resolve()) for p in members)
                files.extend(members)
            else:
                files.append(path)
        if not files:
            raise ValueError("Completed artifact directories cannot be empty")
        self.data["stages"][stage] = {
            "settings": settings,
            "artifacts": [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in files],
            "directories": directories,
            "result": result or {},
        }
        atomic_json(self.path, self.data)

    def retire(self, stage: str, paths: list[Path], reason: str, consumer: dict | None = None) -> None:
        """Retire only verified generated outputs, retaining their full historical recipe and hashes.

        A retired stage is no longer complete. Its saved descendants can remain
        usable through separately verified evidence receipts; needing the weights
        again requires reconstructing them from the recorded recipe.
        """
        record = self.data['stages'].get(stage)
        if not record or not reason.strip():
            raise ValueError('Retirement requires a recorded stage and a reason')
        root = self.path.parent.resolve()
        paths = [Path(path) for path in paths]
        for path in paths:
            if path.is_symlink() or path.resolve() == root or not path.resolve().is_relative_to(root):
                raise ValueError('Can retire only generated artifacts inside the run directory')
            if path.is_dir() and any(p.is_symlink() for p in path.rglob('*')):
                raise ValueError('Cannot retire generated artifacts containing symlinks')
        files = {str(p.resolve()) for path in paths for p in
                 (path.rglob('*') if path.is_dir() else [path]) if p.is_file()}
        if files != {item['path'] for item in record['artifacts']} or not self.completed(stage, record['settings']):
            raise ValueError('Generated artifacts changed or do not match the verified stage')
        receipt = {'at': datetime.now(timezone.utc).isoformat(), 'reason': reason,
                   'paths': [str(path.resolve()) for path in paths], 'record': record,
                   'consumer': consumer, 'status': 'retiring'}
        self.data.setdefault('retired_stages', {}).setdefault(stage, []).append(receipt)
        atomic_json(self.path, self.data)  # Preserve evidence before deleting regenerable weights.
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        receipt['status'] = 'retired'
        del self.data['stages'][stage]
        atomic_json(self.path, self.data)
