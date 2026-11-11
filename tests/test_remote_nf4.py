"""Offline checks for the rented-GPU NF4 runner: no network, GPU or Hub call is made."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'remote_nf4.py'


@pytest.fixture
def remote():
    loader = importlib.util.spec_from_file_location('remote_nf4', SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def test_public_repo_is_refused(remote, monkeypatch):
    monkeypatch.setattr(remote, 'api', lambda: SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(private=False)))
    with pytest.raises(SystemExit, match='private'):
        remote.private_repo('someone/public')
    monkeypatch.setattr(remote, 'api', lambda: SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(private=True)))
    remote.private_repo('someone/private')


def test_bundle_will_not_replace_one_that_runs_started_from(remote, monkeypatch):
    hub = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(private=True),
                          list_repo_files=lambda *a, **k: ['code.tar', 'results/hf-a10g/nf4-environment.json'])
    monkeypatch.setattr(remote, 'api', lambda: hub)
    monkeypatch.setattr(remote, 'git', lambda *args: '')
    with pytest.raises(SystemExit, match='hf-a10g'):
        remote.bundle(SimpleNamespace(repo='r', no_upload=True, replace_with_runs=False))


def test_kaggle_start_needs_no_project_imports():
    text = (ROOT / 'scripts' / 'kaggle_start.py').read_text(encoding='utf-8')
    assert "get_secret('HF_TOKEN')" in text and 'DEADLINE_MINUTES' in text and 'kaggle-t4' in text
    compile(text, 'kaggle_start.py', 'exec')


def test_half_written_generation_line_is_dropped(remote, tmp_path):
    path = tmp_path / 'tag-fp.jsonl'
    path.write_text('{"key": "a"}\n{"key": "b"}\n{"key": "c", "respo', encoding='utf-8')
    assert remote.trim_partial_lines(path) == 1
    assert [json.loads(line)['key'] for line in path.read_text(encoding='utf-8').splitlines()] == ['a', 'b']
    assert remote.trim_partial_lines(path) == 0


def test_unfinished_q4_evidence_is_left_out_but_finished_q4_stays(remote):
    names = ['gens\\global.jsonl', 'capability\\global.json',
             'gens\\global-q4.jsonl', 'capability\\global-q4.json', 'capability\\global-q4.none.json',
             'gens\\earth-q4.jsonl', 'capability\\earth-q4.json', 'judged\\earth-q4.local.jsonl',
             'judged\\earth-q4.local.summary.json', 'calibration\\earth-circuit_merge-0p5-q4.local.json']
    assert remote.unfinished_q4(names) == {'gens/global-q4.jsonl', 'capability/global-q4.json',
                                           'capability/global-q4.none.json',
                                           'calibration/earth-circuit_merge-0p5-q4.local.json'}
    assert remote.unfinished_q4(['gens/global.jsonl', 'judged/earth.local.jsonl']) == set()


def test_build_cache_round_trip(remote, monkeypatch, tmp_path):
    stored = {}

    class Hub:
        def repo_info(self, *a, **k): return SimpleNamespace(private=True)
        def list_repo_files(self, *a, **k): return list(stored)
        def upload_file(self, path_or_fileobj, path_in_repo, **k): stored[path_in_repo] = Path(path_or_fileobj).read_bytes()

    monkeypatch.setattr(remote, 'api', lambda: Hub())
    import huggingface_hub

    def download(repo, name, repo_type=None, local_dir=None):
        target = Path(local_dir) / 'blob'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(stored[name])
        return str(target)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    args = SimpleNamespace(repo='r', work=str(tmp_path), name='cache/llama-sm86.tar.gz', to=str(tmp_path / 'out'))
    with pytest.raises(SystemExit):
        remote.cache_get(args)
    (tmp_path / 'bin').mkdir()
    (tmp_path / 'bin' / 'llama-server').write_bytes(b'binary')
    remote.cache_put(SimpleNamespace(repo='r', work=str(tmp_path), name=args.name, source=str(tmp_path / 'bin')))
    remote.cache_get(args)
    assert (tmp_path / 'out' / 'llama-server').read_bytes() == b'binary'


def test_judge_cache_snapshot_is_a_readable_copy(remote, tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    db = sqlite3.connect(output / 'llm_cache.sqlite')
    db.execute('create table cache (k text primary key, v text)')
    db.executemany('insert into cache values (?, ?)', [(str(i), 'verdict') for i in range(50)])
    db.commit()
    db.execute('insert into cache values (?, ?)', ('pending', 'uncommitted'))  # an open write transaction
    remote.snapshot_databases(output, tmp_path / 'snap')
    copy = sqlite3.connect(tmp_path / 'snap' / 'llm_cache.sqlite')
    assert copy.execute('select count(*) from cache').fetchone()[0] == 50
    copy.close(); db.rollback(); db.close()


def test_constraints_pin_the_laptop_but_not_torch_or_the_kits(remote):
    torch, lines, kits = remote.pins()
    names = {line.split('==')[0] for line in lines}
    assert 'torch' not in names and not names & set(remote.KITS) and 'langsafe' not in names
    assert {'transformers', 'bitsandbytes', 'accelerate', 'datasets', 'numpy'} <= names
    assert all(kit['url'].startswith('https://github.com/Lexsi-Labs/') and len(kit['commit']) == 40 for kit in kits.values())
    assert '+cu' in torch


def _manifest_and_plan():
    files = {'model.safetensors': {'sha256': 'a' * 64, 'bytes': 3}}
    manifest = {'spec_digest': 's', 'spec_sha256': 'f', 'code_files_sha256': {'src/langsafe/cli.py': 'c'},
                'evidence': {'gens/global.jsonl': {'sha256': 'e', 'bytes': 1}},
                'base_models': {'global': {'files': files}, 'earth': {'files': files}},
                'selected': {'tag': 'earth-circuit_merge-0p5', 'files': files},
                'judges': {'qwen3-8b': {'filename': 'Qwen3-8B-Q4_K_M.gguf', 'sha256': 'q' * 64}},
                'report_tags': ['global', 'earth-nf4'], 'stages': ['global-nf4-heldout', 'report']}
    plan = {'spec_digest': 's', 'spec_sha256': 'f', 'code': {'files_sha256': {'src\\langsafe\\cli.py': 'c'}},
            'source_evidence': {'files': {'gens\\global.jsonl': {'sha256': 'e', 'bytes': 1}}},
            'checkpoints': {'global': {'files': files}, 'earth': {'files': files},
                            'earth-circuit_merge-0p5': {'files': files},
                            'judge:qwen3-8b': {'files': {'Qwen3-8B-Q4_K_M.gguf': {'sha256': 'q' * 64}}}},
            'report_tags': ['global', 'earth-nf4'], 'stages': [{'name': 'global-nf4-heldout'}, {'name': 'report'}]}
    return manifest, plan


@pytest.mark.parametrize('mutation,label', [
    (None, None), ('code', 'code'), ('evidence', 'source evidence'), ('earth', 'earth checkpoint'),
    ('repair', 'repair checkpoint'), ('judge', 'judge qwen3-8b'), ('stages', 'stages'), ('spec', 'spec digest')])
def test_remote_plan_must_match_the_laptop_field_by_field(remote, monkeypatch, tmp_path, mutation, label):
    manifest, plan = _manifest_and_plan()
    if mutation == 'code':
        plan['code']['files_sha256']['src\\langsafe\\cli.py'] = 'other'
    elif mutation == 'evidence':
        plan['source_evidence']['files']['gens\\extra.jsonl'] = {'sha256': 'x', 'bytes': 1}
    elif mutation == 'earth':
        plan['checkpoints']['earth'] = {'files': {'model.safetensors': {'sha256': 'b' * 64, 'bytes': 3}}}
    elif mutation == 'repair':
        plan['checkpoints']['earth-circuit_merge-0p5'] = {'files': {}}
    elif mutation == 'judge':
        plan['checkpoints']['judge:qwen3-8b']['files']['Qwen3-8B-Q4_K_M.gguf']['sha256'] = 'z' * 64
    elif mutation == 'stages':
        plan['stages'] = plan['stages'][:1]
    elif mutation == 'spec':
        plan['spec_digest'] = 'other'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(remote, 'dry_run_plan', lambda *args, **kwargs: plan)
    args = SimpleNamespace(work=str(tmp_path), repo='unused')
    if mutation is None:
        remote.verify(args)
    else:
        with pytest.raises(SystemExit, match=label):
            remote.verify(args)


def test_launch_dry_run_names_no_secret(remote, monkeypatch, capsys):
    monkeypatch.setattr(remote, 'private_repo', lambda repo: None)
    common = dict(repo='HydraOfMalice/lexsi-nf4-private', platform='hf-a10g', flavor='a10g-small', timeout='11h',
                  image='nvidia/cuda:12.8.1-devel-ubuntu24.04', dry_run=True)
    remote.launch_hf(SimpleNamespace(**common, job='nf4'))
    job = json.loads(capsys.readouterr().out)
    assert set(job['env']) == {'BUNDLE_REPO', 'PLATFORM', 'WORK'}
    assert 'HF_TOKEN' not in json.dumps(job['env']) and 'hf_' not in remote.BOOTSTRAP
    remote.launch_hf(SimpleNamespace(**common, job='fair'))
    fair = json.loads(capsys.readouterr().out)
    assert fair['env']['CODE_TAR'] == 'fair/code.tar' and fair['env']['ENTRY'] == 'remote_fair_baseline.sh'
    assert (ROOT / 'scripts' / fair['env']['ENTRY']).is_file()


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
def test_remote_shell_script_parses():
    # Through stdin, so a bash that cannot see Windows paths (WSL's) still checks it.
    for text in ((ROOT / 'scripts' / 'remote_nf4.sh').read_text(encoding='utf-8'),
                 (ROOT / 'scripts' / 'remote_fair_baseline.sh').read_text(encoding='utf-8'), _bootstrap()):
        assert '\r' not in text
        # Bytes, not text: a Windows text pipe would turn every newline into CRLF.
        result = subprocess.run(['bash', '-n'], input=text.encode(), capture_output=True)
        assert result.returncode == 0, result.stderr.decode(errors='replace')


def _bootstrap():
    loader = importlib.util.spec_from_file_location('remote_nf4_bootstrap', SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module.BOOTSTRAP
