"""scripts/import_nf4.py: a remote NF4 run's results come into the full run only if they belong to its evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('import_nf4', ROOT / 'scripts' / 'import_nf4.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
TAGS = ['global-nf4', 'earth-nf4', 'earth-circuit_merge-0p5-nf4']


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    run, nf4 = tmp_path / 'earth', tmp_path / 'earth-nf4-remote'
    for name, text in (('gens/global.jsonl', '{"id": 1}\n'), ('selection.json', '{"split": "repair"}'),
                       ('data/manifest.json', '{"ok": true}')):
        for root in (run, nf4):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text, encoding='utf-8')
    (run / 'experiment.json').write_text(json.dumps({'identity': {'spec': 'abc123'}}), encoding='utf-8')
    evidence = {name: {'sha256': digest(run / name), 'bytes': (run / name).stat().st_size}
                for name in ('gens/global.jsonl', 'selection.json', 'data/manifest.json')}
    for tag in TAGS:
        for name in (f'gens/{tag}.jsonl', f'judged/{tag}.local.jsonl', f'judged/{tag}.local.summary.json',
                     f'capability/{tag}.json', f'capability/{tag}.none.json', f'auditkit/{tag}.local/{tag}.local.json'):
            (nf4 / name).parent.mkdir(parents=True, exist_ok=True)
            (nf4 / name).write_text(f'{tag} {name}', encoding='utf-8')
    environment = {'status': 'complete', 'spec_digest': 'abc123', 'source_evidence': {'files': evidence},
                   'report_tags': ['global', 'earth', *TAGS], 'hardware': {'device': 'NVIDIA A10G'},
                   'started_at': 'then', 'finished_at': 'later', 'code': {'content_sha256': 'code'}}
    (nf4 / 'nf4-environment.json').write_text(json.dumps(environment), encoding='utf-8')
    (nf4 / 'nf4-status.json').write_text(json.dumps({'software_run_complete': True, 'nf4_cuda': 'executed'}),
                                         encoding='utf-8')
    status = tmp_path / 'chain-status.json'
    status.write_text(json.dumps({'running': None, 'done': ['earth-full']}), encoding='utf-8')
    return run, nf4, status


def test_a_matching_run_is_imported_with_its_provenance(setup):
    from langsafe.provenance import Ledger

    run, nf4, status = setup
    record = module.import_nf4(nf4, run, status)
    assert record['tags'] == TAGS and record['device'] == 'NVIDIA A10G' and record['evidence_checked'] == 3
    assert len(record['files']) == 18
    for name, sha in record['files'].items():
        assert digest(run / name) == sha == digest(nf4 / name)
    assert json.loads((run / 'nf4-import.json').read_text(encoding='utf-8'))['files'] == record['files']
    ledger = Ledger(run / 'ledger.jsonl')
    assert ledger.verify()[0]
    assert json.loads((run / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()[-1])['step'] == 'import-nf4'
    # Importing again is harmless: the copies are identical.
    module.import_nf4(nf4, run, status)


@pytest.mark.parametrize('mutation,message', [
    ('running', 'still running'), ('evidence', 'Source evidence differs'), ('spec', 'used spec'),
    ('incomplete', 'not complete'), ('clash', 'different content')])
def test_an_import_that_does_not_belong_is_refused(setup, mutation, message):
    run, nf4, status = setup
    if mutation == 'running':
        status.write_text(json.dumps({'running': 'earth-full'}), encoding='utf-8')
    elif mutation == 'evidence':
        (run / 'selection.json').write_text('{"split": "heldout"}', encoding='utf-8')
    elif mutation == 'spec':
        (run / 'experiment.json').write_text(json.dumps({'identity': {'spec': 'other'}}), encoding='utf-8')
    elif mutation == 'incomplete':
        (nf4 / 'nf4-status.json').write_text(json.dumps({'software_run_complete': False}), encoding='utf-8')
    elif mutation == 'clash':
        (run / 'judged').mkdir(parents=True, exist_ok=True)
        (run / 'judged' / 'earth-nf4.local.jsonl').write_text('something else', encoding='utf-8')
    with pytest.raises(SystemExit, match=message):
        module.import_nf4(nf4, run, status)
    if mutation != 'clash':
        assert not (run / 'nf4-import.json').exists() and not (run / 'gens' / 'earth-nf4.jsonl').exists()
