#!/usr/bin/env python3
"""Bring a finished remote NF4 run's results into the full run, so one report covers FP16, Q4 and NF4.

    python scripts/import_nf4.py runs/experiments/full/earth-nf4-hf-a10g --report

The NF4 check ran on a rented GPU (scripts/remote_nf4.py) into its own directory. Before copying anything this checks
that the NF4 run is complete, that it used this run's spec, and that every source file its plan hashed still has the
same SHA-256 here, so its NF4 numbers belong to this run's evidence. It also checks that the full run's chain is not
writing: the run's ledger is a hash chain with one writer at a time.

It copies the -nf4 tags' generations, judgements, judge summaries, capability results and AuditKIT records, refuses to
replace a different file, and records every copied file, with its SHA-256 and the NF4 hardware, in nf4-import.json and
in the run's ledger. --report then rebuilds the run's report over its own tags plus the NF4 tags, with the project's
report command, and verifies the ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f'Missing {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def chain_idle(status_path: Path) -> None:
    if not status_path.is_file():
        return
    status = read_json(status_path)
    if status.get('running'):
        raise SystemExit(f"The chain is still running {status['running']}; import once it has stopped")


def nf4_files(nf4: Path, tags: list[str]) -> list[str]:
    files = []
    for tag in tags:
        for name in (f'gens/{tag}.jsonl', f'judged/{tag}.local.jsonl', f'judged/{tag}.local.summary.json',
                     f'capability/{tag}.json', f'capability/{tag}.none.json', f'hand_check/{tag}.local.csv'):
            if (nf4 / name).is_file():
                files.append(name)
        audit = nf4 / 'auditkit' / f'{tag}.local'
        if audit.is_dir():
            files.extend(path.relative_to(nf4).as_posix() for path in sorted(audit.rglob('*')) if path.is_file())
    return files


def import_nf4(nf4: Path, run: Path, status_path: Path) -> dict:
    from langsafe.provenance import Ledger

    chain_idle(status_path)
    environment = read_json(nf4 / 'nf4-environment.json')
    status = read_json(nf4 / 'nf4-status.json')
    if environment.get('status') != 'complete' or not status.get('software_run_complete') or status.get('nf4_cuda') != 'executed':
        raise SystemExit('The NF4 run is not complete')
    experiment = read_json(run / 'experiment.json')
    run_digest = (experiment.get('identity') or {}).get('spec')
    if environment.get('spec_digest') != run_digest:
        raise SystemExit(f"The NF4 run used spec {environment.get('spec_digest')}, this run {run_digest}")
    evidence = environment['source_evidence']['files']
    changed = [name for name, identity in evidence.items()
               if not (run / name).is_file() or sha256(run / name) != identity['sha256']]
    if changed:
        raise SystemExit(f'Source evidence differs from what the NF4 run was given: {changed[:5]}')
    tags = [tag for tag in environment['report_tags'] if tag.endswith('-nf4')]
    copied = {}
    for name in nf4_files(nf4, tags):
        source, target = nf4 / name, run / name
        digest = sha256(source)
        if target.exists():
            if sha256(target) != digest:
                raise SystemExit(f'{target} exists with different content; refusing to replace it')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied[name] = digest
    hardware = environment.get('hardware', {})
    record = {'imported_at': datetime.now(timezone.utc).isoformat(), 'source': str(nf4), 'tags': tags,
              'files': copied, 'nf4_environment_sha256': sha256(nf4 / 'nf4-environment.json'),
              'device': hardware.get('device'), 'nvidia_smi': hardware.get('nvidia_smi'),
              'started_at': environment.get('started_at'), 'finished_at': environment.get('finished_at'),
              'code': (environment.get('remote_code') or environment.get('code') or {}).get('content_sha256'),
              'evidence_checked': len(evidence)}
    (run / 'nf4-import.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    Ledger(run / 'ledger.jsonl').append('import-nf4', {'source': str(nf4), 'tags': tags, 'files': len(copied),
                                                       'record_sha256': sha256(run / 'nf4-import.json'),
                                                       'device': hardware.get('device')})
    return record


def rebuild_report(run: Path, nf4_tags: list[str]) -> list[str]:
    """The run's own report tags plus the NF4 tags, through the project's report command, then a ledger check."""
    tags = list(read_json(run / 'report.json')['models'])
    tags += [tag for tag in nf4_tags if tag not in tags]
    env = dict(os.environ, LANGSAFE_RUNS_DIR=str(run), PYTHONUTF8='1')
    base = [sys.executable, '-m', 'langsafe.cli', '--runs-dir', str(run)]
    subprocess.run([*base, 'report', '--tags', ','.join(tags), '--family', 'earth', '--judges', 'local'],
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([*base, 'ledger-verify'], cwd=ROOT, env=env, check=True)
    return tags


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('nf4', type=Path, help='a finished NF4 output directory, e.g. runs/experiments/full/earth-nf4-hf-a10g')
    parser.add_argument('--run', type=Path, default=ROOT / 'runs' / 'experiments' / 'full' / 'earth')
    parser.add_argument('--chain-status', type=Path, default=ROOT / 'runs' / 'chain-status.json')
    parser.add_argument('--report', action='store_true', help='then rebuild the report over all tags and verify the ledger')
    args = parser.parse_args(argv)
    record = import_nf4(args.nf4, args.run, args.chain_status)
    print(f"imported {len(record['files'])} files for {', '.join(record['tags'])} "
          f"(evidence checked: {record['evidence_checked']} files, device {record['device']})")
    if args.report:
        tags = rebuild_report(args.run, record['tags'])
        print(f'report rebuilt over {len(tags)} tags')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
