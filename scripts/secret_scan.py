"""Scan a git repo's whole history and its tracked files, or any folder, for secrets. Prints masked hits only.

    .venv/Scripts/python.exe scripts/secret_scan.py               # this repo: every blob in history, plus tracked files
    .venv/Scripts/python.exe scripts/secret_scan.py --tree <dir>  # every file under a folder, such as a public copy

Exits 1 when anything is found. A hit is printed as its kind, its first six characters, its length and where it is, so the
output itself never leaks the secret.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8_000_000

PATTERNS = {
    'huggingface token': re.compile(rb'\bhf_[A-Za-z0-9]{30,}'),
    'anthropic key': re.compile(rb'sk-ant-[A-Za-z0-9_\-]{20,}'),
    'openrouter key': re.compile(rb'sk-or-v1-[A-Za-z0-9]{32,}'),
    'openai-style key': re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9]{32,}'),
    'github token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})'),
    'aws access key': re.compile(rb'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b'),
    'google api key': re.compile(rb'\bAIza[0-9A-Za-z_\-]{35}'),
    'slack token': re.compile(rb'\bxox[baprs]-[A-Za-z0-9\-]{10,}'),
    'private key': re.compile(rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY'),
    'discord webhook': re.compile(rb'discord(?:app)?\.com/api/webhooks/\d+/[A-Za-z0-9_\-]{20,}'),
    'discord bot token': re.compile(rb'\b[MN][A-Za-z\d]{23,25}\.[\w\-]{6}\.[\w\-]{27,}'),
    # A quoted value at least 16 characters long with both letters and digits, assigned to a secret-sounding name.
    'assignment of a secret': re.compile(
        rb'(?i)\b(?:api[_\-]?key|secret|access[_\-]?token|auth[_\-]?token|password|passwd|hf_token)\b\s*[:=]\s*'
        rb'[\'"](?=[^\'"\s]*[A-Za-z])(?=[^\'"\s]*\d)([^\'"\s]{16,})[\'"]'),
    'bearer header': re.compile(rb'(?i)authorization[\'"]?\s*[:=]\s*[\'"]bearer\s+[A-Za-z0-9_\-\.]{20,}'),
}
# Placeholders that look like secrets but are not.
BENIGN = re.compile(rb'(?i)(x{6,}|\*{4,}|<[^>]*>|your[_\-]|example|placeholder|dummy|fake|\$\{|\{\{|REDACTED)')


def scan_bytes(data: bytes) -> list[tuple[str, bytes, int]]:
    """(kind, matched text, line number) for every secret-like match."""
    hits = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(data):
            value = match.group(1) if pattern.groups else match.group(0)
            if kind != 'private key' and BENIGN.search(value):
                continue
            hits.append((kind, match.group(0), data.count(b'\n', 0, match.start()) + 1))
    return hits


def mask(value: bytes) -> str:
    return f"{value[:6].decode('utf-8', 'replace')}… ({len(value)} characters)"


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(['git', '-C', str(repo), *args], input=stdin, capture_output=True, check=True).stdout


def scan_repo(repo: Path) -> dict[tuple[str, bytes], set[str]]:
    """Every blob reachable from any ref, and every tracked file as it is on disk now."""
    paths: dict[str, str] = {}
    for line in _git(repo, 'rev-list', '--objects', '--all').decode('utf-8', 'replace').splitlines():
        sha, _, path = line.partition(' ')
        if path:
            paths.setdefault(sha, path)
    hits: dict[tuple[str, bytes], set[str]] = {}
    checked = _git(repo, 'cat-file', '--batch-check', stdin='\n'.join(paths).encode()).decode().splitlines()
    for line in checked:
        sha, kind, size = line.split()
        if kind == 'blob' and int(size) <= MAX_BYTES:
            for name, value, line_no in scan_bytes(_git(repo, 'cat-file', 'blob', sha)):
                hits.setdefault((name, value), set()).add(f'history:{paths[sha]}:{line_no}')
    for rel in _git(repo, 'ls-files').decode('utf-8', 'replace').splitlines():
        path = repo / rel
        if path.is_file() and path.stat().st_size <= MAX_BYTES:
            for name, value, line_no in scan_bytes(path.read_bytes()):
                hits.setdefault((name, value), set()).add(f'tracked:{rel}:{line_no}')
    return hits


def scan_tree(root: Path) -> dict[tuple[str, bytes], set[str]]:
    """Every file under a folder, skipping `.git`."""
    hits: dict[tuple[str, bytes], set[str]] = {}
    for path in sorted(root.rglob('*')):
        if '.git' in path.relative_to(root).parts or not path.is_file() or path.stat().st_size > MAX_BYTES:
            continue
        for name, value, line_no in scan_bytes(path.read_bytes()):
            hits.setdefault((name, value), set()).add(f'{path.relative_to(root).as_posix()}:{line_no}')
    return hits


def report(hits: dict[tuple[str, bytes], set[str]]) -> list[str]:
    lines = []
    for (name, value), where in sorted(hits.items()):
        places = sorted(where)
        lines.append(f"{name}: {mask(value)} at {', '.join(places[:4])}{' and more' if len(places) > 4 else ''}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('repo', nargs='?', type=Path, default=ROOT, help='git repo to scan, history included')
    parser.add_argument('--tree', type=Path, help='scan every file under this folder instead')
    args = parser.parse_args()
    hits = scan_tree(args.tree) if args.tree else scan_repo(args.repo)
    for line in report(hits):
        print(line)
    print(f'{len(hits)} possible secrets found')
    return 1 if hits else 0


if __name__ == '__main__':
    sys.exit(main())
