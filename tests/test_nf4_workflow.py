"""Offline contracts for the CUDA handoff; no model, GPU, or network is used."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'check_nf4_environment.py'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def handoff(tmp_path):
    source = tmp_path / 'source'
    models = tmp_path / 'models'
    for name in ('global', 'fire'):
        checkpoint = models / name
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{"model_type":"cohere2"}')
        (checkpoint / 'model.safetensors').write_bytes(b'fixture weights, never loaded')
        (checkpoint / 'tokenizer.json').write_text('{}')
        (checkpoint / 'tokenizer_config.json').write_text('{}')
    checkpoint = source / 'checkpoints' / 'fire-resta-a0.1'
    checkpoint.mkdir(parents=True)
    for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
        (checkpoint / name).write_bytes((models / 'fire' / name).read_bytes())
    spec = {
        'spec_version': 1, 'frozen': False,
        'models': {'reference': 'Fixture/global', 'drifted': {'fire': 'Fixture/fire'}},
        'languages': {'fire': ['en']}, 'capability_languages': ['en'],
        'preamble_conditions': ['default', 'none'],
        'decoding': {'dtype': 'float16', 'do_sample': False, 'max_new_tokens': 256, 'batch_size': 2},
        'data': {'seed': 1}, 'experiment': {'local_batch_size': 2},
        'judging': {'local_judges': [
            {'name': 'first', 'model': 'Fixture/global', 'base_url': 'local'},
            {'name': 'second', 'model': 'Fixture/fire', 'base_url': 'local'}]},
        'quantization': ['nf4', 'gguf_q4_k_m'], 'thresholds': {'quantized_must_hold': True},
    }
    spec_path = tmp_path / 'spec.yaml'
    spec_path.write_text(yaml.safe_dump(spec))
    data = source / 'data'
    heldout = [dict(id=f'heldout-{kind}::en', group=f'heldout-{kind}', lang='en', kind=kind,
                    text=f'{kind} held-out fixture', source='fixture', source_id=kind, split='heldout')
               for kind in ('harmful', 'benign')]
    repair = [dict(row, id=row['id'].replace('heldout', 'repair'),
                   group=row['group'].replace('heldout', 'repair'), split='repair') for row in heldout]
    _write_rows(data / 'heldout.jsonl', heldout)
    _write_rows(data / 'repair.jsonl', repair)
    manifest = {'spec_digest': _digest(spec), 'family': 'fire', 'languages': ['en'],
                'coverage': {'complete': True}, 'leakage_check': {'ok': True},
                'files': {split: {'sha256': hashlib.sha256((data / f'{split}.jsonl').read_bytes()).hexdigest(),
                                 'rows': 2} for split in ('heldout', 'repair')}}
    (data / 'manifest.json').write_text(json.dumps(manifest))
    selected = {'tag': 'fire-resta-0p1', 'arm': 'resta', 'knob': 0.1,
                'model': '/original-machine/run/checkpoints/fire-resta-a0.1'}
    selection = {'split': 'repair', 'preamble': 'default', 'worst_language': 'en',
                 'arms': {'resta': {'selected': selected}}, 'selected_for_quantization': selected}
    (source / 'selection.json').write_text(json.dumps(selection))
    for tag in ('global', 'fire', 'fire-resta-0p1'):
        gens = [dict(row, preamble=pre, model_tag=tag, fingerprint=tag, response='fixture',
                     rendered_prompt_sha256=f"{row['id']}-{pre}") for row in heldout for pre in ('default', 'none')]
        _write_rows(source / 'gens' / f'{tag}.jsonl', gens)
        _write_rows(source / 'judged' / f'{tag}.local.jsonl', [dict(row, label='unclear') for row in gens])
        (source / 'judged' / f'{tag}.local.summary.json').write_text(json.dumps({'judges': ['first', 'second']}))
        for pre in ('default', 'none'):
            suffix = '' if pre == 'default' else '.none'
            path = source / 'capability' / f'{tag}{suffix}.json'
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'complete': True, 'metadata': {'quant': None, 'preamble': pre},
                                       'languages': {'en': {'data_sha256': 'same-items', 'n': 2, 'per_item': [1, 0]}}}))
    return {'source': source, 'models': models, 'spec': spec_path, 'output': tmp_path / 'remote-nf4'}


def run_handoff(handoff, *args):
    env = dict(os.environ, LANGSAFE_MODELS=str(handoff['models']), HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return subprocess.run([sys.executable, str(SCRIPT), '--source-run', str(handoff['source']),
                           '--spec', str(handoff['spec']), '--output', str(handoff['output']), *args],
                          cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)


def test_offline_plan_preserves_full_prompts_and_precision_matched_trio(handoff):
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['status'] == 'planned_not_executed'
    assert plan['profile'] == 'full'
    assert plan['hardware_checked'] is False
    commands = [stage['args'] for stage in plan['stages']]
    generations = [args for args in commands if args[0] == 'generate']
    assert [args[args.index('--tag') + 1] for args in generations] == ['global-nf4', 'fire-nf4', 'fire-resta-0p1-nf4']
    assert all(args[args.index('--quant') + 1] == 'nf4' and '--limit' not in args and '--preamble' not in args
               for args in generations)
    capabilities = [args for args in commands if args[0] == 'capability']
    assert len(capabilities) == 6
    assert {args[args.index('--preamble') + 1] for args in capabilities} == {'default', 'none'}
    assert all('--quant' in args and '--limit' not in args for args in capabilities)
    assert len([args for args in commands if args[0] == 'judge' and '--judges' in args]) == 3
    assert len([args for args in commands if args[0] == 'audit']) == 3
    report = next(args for args in commands if args[0] == 'report')
    assert report[report.index('--tags') + 1] == 'global,fire,fire-resta-0p1,global-nf4,fire-nf4,fire-resta-0p1-nf4'
    assert not handoff['output'].exists()
    assert plan['resolved_models']['fire-resta-0p1'] == str(handoff['source'] / 'checkpoints' / 'fire-resta-a0.1')
    assert plan['model_refs']['global'] == 'Fixture/global'
    assert plan['source_evidence']['selected_checkpoint']['tag'] == 'fire-resta-0p1'
    assert plan['source_evidence']['selected_checkpoint']['files']
    selected_generation = next(args for args in generations if args[args.index('--tag') + 1] == 'fire-resta-0p1-nf4')
    assert selected_generation[selected_generation.index('--model') + 1] == str(handoff['source'] / 'checkpoints' / 'fire-resta-a0.1')
    assert plan['code']['content_sha256'] and plan['source_evidence']['files']
    assert Path(plan['selected_checkpoint_target']).parts[-2:] == ('checkpoints', 'fire-resta-0p1')


@pytest.mark.parametrize('mutation,expected', [
    ('data', 'hash'), ('spec', 'spec'), ('selection', 'repair'), ('preambles', 'preamble'),
    ('precision', 'precision'), ('prompts', 'prompt'), ('occupied', 'empty'), ('nested', 'overlap')])
def test_handoff_fails_closed_before_gpu_work(handoff, mutation, expected):
    if mutation == 'data':
        with (handoff['source'] / 'data' / 'heldout.jsonl').open('a') as stream:
            stream.write('\n')
    elif mutation in {'spec', 'preambles'}:
        spec = yaml.safe_load(handoff['spec'].read_text())
        if mutation == 'spec':
            spec['data']['seed'] = 9
        else:
            spec['preamble_conditions'] = ['default']
        handoff['spec'].write_text(yaml.safe_dump(spec))
    elif mutation == 'selection':
        path = handoff['source'] / 'selection.json'
        value = json.loads(path.read_text()); value['split'] = 'heldout'; path.write_text(json.dumps(value))
    elif mutation == 'precision':
        path = handoff['source'] / 'capability' / 'global.json'
        value = json.loads(path.read_text()); value['metadata']['quant'] = 'nf4'; path.write_text(json.dumps(value))
    elif mutation == 'prompts':
        path = handoff['source'] / 'gens' / 'fire.jsonl'
        path.write_text(path.read_text().splitlines()[0] + '\n')
    elif mutation == 'occupied':
        handoff['output'].mkdir(); (handoff['output'] / 'unrelated').write_text('keep me')
    elif mutation == 'nested':
        handoff['output'] = handoff['source'] / 'nf4'
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode != 0
    assert expected in result.stderr.lower()
    assert not (handoff['output'] / 'gens').exists()


def test_source_generation_quantization_metadata_cannot_be_mixed(handoff):
    path = handoff['source'] / 'gens' / 'global.jsonl'
    value = json.loads(path.read_text().splitlines()[0]); value['variant'] = {'quant': 'nf4'}
    path.write_text(json.dumps(value) + '\n' + '\n'.join(path.read_text().splitlines()[1:]) + '\n')
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode != 0 and 'full precision' in result.stderr.lower()


def test_nf4_outputs_must_say_nf4_on_every_answer(handoff, tmp_path):
    """An answer without the mark could be full precision, because fp16 and NF4 share a weight fingerprint."""
    loader = importlib.util.spec_from_file_location('nf4_check', SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    output = tmp_path / 'nf4-output'
    (output / 'gens').mkdir(parents=True)
    for tag in ('global', 'fire', 'fire-resta-0p1'):
        rows = [json.loads(line) for line in (handoff['source'] / 'gens' / f'{tag}.jsonl').read_text().splitlines()]
        _write_rows(output / 'gens' / f'{tag}-nf4.jsonl', [dict(row, model_tag=f'{tag}-nf4') for row in rows])
    with pytest.raises(ValueError, match='generation identity'):
        module.validate_nf4_outputs({'output': str(output), 'source_run': str(handoff['source']), 'family': 'fire'})


def test_development_inputs_require_explicit_label(handoff):
    spec = yaml.safe_load(handoff['spec'].read_text()); spec['profile'] = 'development'
    handoff['spec'].write_text(yaml.safe_dump(spec))
    path = handoff['source'] / 'data' / 'manifest.json'
    manifest = json.loads(path.read_text()); manifest['spec_digest'] = _digest(spec); path.write_text(json.dumps(manifest))
    denied = run_handoff(handoff, '--dry-run')
    assert denied.returncode != 0 and 'development' in denied.stderr.lower()
    allowed = run_handoff(handoff, '--dry-run', '--allow-development')
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout)['profile'] == 'development'


def test_missing_cached_checkpoint_is_an_explicit_offline_blocker(handoff):
    (handoff['models'] / 'global' / 'model.safetensors').unlink()
    result = run_handoff(handoff, '--dry-run')
    assert result.returncode != 0
    assert 'checkpoint' in result.stderr.lower()


def test_nf4_model_validation_rejects_cpu_or_disk_offload():
    from langsafe.gen.model import validate_nf4_model

    class Offloaded:
        is_loaded_in_4bit = True
        hf_device_map = {'': 'cuda', 'lm_head': 'cpu'}

    with pytest.raises(RuntimeError, match='CUDA'):
        validate_nf4_model(Offloaded())


def test_nf4_validation_rejects_unquantized_or_unplaced_models():
    from types import SimpleNamespace
    from langsafe.gen.model import validate_nf4_model

    with pytest.raises(RuntimeError, match='4-bit'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=False, hf_device_map={'': 'cuda'}))
    with pytest.raises(RuntimeError, match='CUDA'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True, hf_device_map={'lm_head': 'cpu'}))
    with pytest.raises(RuntimeError, match='device map'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True, hf_device_map=None))
    on = lambda device: lambda: [('layer.weight', SimpleNamespace(device=device))]
    validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True, named_parameters=on('cuda:0')))
    with pytest.raises(RuntimeError, match='CUDA'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True, named_parameters=on('cpu')))
    quantizer = SimpleNamespace(quantization_config=SimpleNamespace(_load_in_4bit=True, bnb_4bit_quant_type='nf4'))
    validate_nf4_model(SimpleNamespace(hf_quantizer=quantizer, hf_device_map={'': 0}))
    with pytest.raises(RuntimeError, match='quantizer type'):
        validate_nf4_model(SimpleNamespace(is_loaded_in_4bit=True,
                                           hf_quantizer=SimpleNamespace(quantization_config=SimpleNamespace(
                                               _load_in_4bit=True, bnb_4bit_quant_type='fp4')),
                                           hf_device_map={'': 0}))


def test_nf4_load_requires_cuda_before_tokenizer_or_checkpoint_loading(monkeypatch):
    from types import SimpleNamespace
    from langsafe.gen import model

    def forbidden(*args, **kwargs):
        pytest.fail('A CUDA preflight failure must precede all checkpoint/tokenizer loading')

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=forbidden),
        AutoTokenizer=SimpleNamespace(from_pretrained=forbidden)))
    monkeypatch.setenv('LANGSAFE_DEVICE', 'cuda')
    with pytest.raises(RuntimeError, match='CUDA'):
        model.load('must-not-load', quant='nf4')


def test_nf4_cuda_map_and_quantization_settings_are_explicit(monkeypatch):
    from types import SimpleNamespace
    from langsafe.gen import model

    loaded = SimpleNamespace(is_loaded_in_4bit=True, hf_device_map={'': 'cuda:0'},
                             named_parameters=lambda: iter([('weight', SimpleNamespace(device=SimpleNamespace(type='cuda')))]),
                             eval=lambda: None)
    capture = {}
    def from_pretrained(path, **kwargs):
        capture.update(kwargs)
        return loaded
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        float16='fake-fp16', cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=from_pretrained),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args: 'fake-tokenizer'),
        BitsAndBytesConfig=lambda **kwargs: kwargs))
    monkeypatch.setenv('LANGSAFE_DEVICE', 'cuda')
    actual, tokenizer = model.load('offline-fixture', quant='nf4')
    assert actual is loaded and tokenizer == 'fake-tokenizer'
    assert capture['device_map'] == {'': 0}
    assert capture['quantization_config'] == {
        'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4', 'bnb_4bit_compute_dtype': 'fake-fp16',
        'bnb_4bit_use_double_quant': False}
    with pytest.raises(RuntimeError, match='strict single-CUDA'):
        model.load('offline-fixture', quant='nf4', device_map='auto')


def test_plan_commands_are_accepted_by_existing_cli(handoff, monkeypatch):
    from langsafe import cli

    result = run_handoff(handoff, '--dry-run')
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    called = []
    for name in ('generate', 'capability', 'judge', 'audit', 'report', 'ledger_verify'):
        monkeypatch.setattr(cli, 'cmd_' + name, lambda args: called.append(args))
    monkeypatch.setenv('LANGSAFE_SPEC', str(handoff['spec']))
    for stage in plan['stages']:
        cli.main(['--spec', str(handoff['spec']), *stage['args']])
    assert len(called) == len(plan['stages'])


def _module():
    loader = importlib.util.spec_from_file_location('nf4_check_resume', SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def _plan(module, handoff, monkeypatch, resume=False):
    from types import SimpleNamespace

    monkeypatch.setenv('LANGSAFE_MODELS', str(handoff['models']))
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    args = SimpleNamespace(source_run=handoff['source'], output=handoff['output'], spec=handoff['spec'], family='fire',
                           judges='local', allow_development=False, resume=resume)
    return module.build_plan(args)


def _fake_stages(module, plan, monkeypatch, fail_at=None):
    """Stand in for the CLI: record each stage, write the report when asked, and fail at one stage if told to."""
    ran = []
    real_run = subprocess.run

    def run(command, cwd=None, env=None, stdout=None, stderr=None, check=False, **kwargs):
        if 'langsafe.cli' not in command:
            return real_run(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=check, **kwargs)
        # The runner puts eight items before the stage's own arguments and swaps in its copy of the checkpoint.
        args = [plan['selected_checkpoint_source'] if part == plan['selected_checkpoint_target'] else part
                for part in command[8:]]
        name = next(stage['name'] for stage in plan['stages'] if stage['args'] == args)
        if name == fail_at:
            raise subprocess.CalledProcessError(1, command)
        ran.append(name)
        if name == 'report':
            Path(env['LANGSAFE_RUNS_DIR'], 'report.json').write_text(json.dumps(
                {'empirical_repair_success': False, 'validation_gaps': []}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module, 'validate_nf4_outputs', lambda plan: None)
    return ran


@pytest.mark.parametrize('model', ['D:\\lexsi\\run\\checkpoints\\fire-resta-a0.1',
                                   '/original-machine/run/checkpoints/fire-resta-a0.1',
                                   'D:\\lexsi\\run\\checkpoints\\fire-resta-a0.1\\'])
def test_selected_checkpoint_name_survives_another_os_path(model):
    assert _module().checkpoint_name(model) == 'fire-resta-a0.1'


def test_windows_selection_path_relocates_into_the_copied_source(handoff, monkeypatch):
    path = handoff['source'] / 'selection.json'
    selection = json.loads(path.read_text())
    windows = 'D:\\lexsi hackathon\\run\\checkpoints\\fire-resta-a0.1'
    selection['selected_for_quantization']['model'] = windows
    selection['arms']['resta']['selected']['model'] = windows
    path.write_text(json.dumps(selection))
    plan = _plan(_module(), handoff, monkeypatch)
    assert plan['resolved_models']['fire-resta-0p1'] == str((handoff['source'] / 'checkpoints' / 'fire-resta-a0.1').resolve())


def test_interrupted_run_resumes_without_repeating_finished_stages(handoff, monkeypatch):
    module = _module()
    plan = _plan(module, handoff, monkeypatch)
    names = [stage['name'] for stage in plan['stages']]
    ran = _fake_stages(module, plan, monkeypatch, fail_at='fire-nf4-heldout')
    with pytest.raises(subprocess.CalledProcessError):
        module.execute(plan, {'device': 'first'})
    first = names[:names.index('fire-nf4-heldout')]
    assert ran == first
    saved = json.loads((handoff['output'] / 'nf4-environment.json').read_text())
    assert saved['status'] == 'failed' and saved['completed_stages'] == first
    # A fresh execute still refuses the occupied directory; only --resume may continue it.
    with pytest.raises(ValueError, match='occupied'):
        module.execute(plan, {'device': 'second'})
    with pytest.raises(ValueError, match='empty'):
        _plan(module, handoff, monkeypatch)
    plan = _plan(module, handoff, monkeypatch, resume=True)
    ran = _fake_stages(module, plan, monkeypatch)
    module.execute(plan, {'device': 'second'}, resume=True)
    assert ran == names[len(first):]
    saved = json.loads((handoff['output'] / 'nf4-environment.json').read_text())
    assert saved['status'] == 'complete' and saved['completed_stages'] == names
    assert len(saved['resumed_at']) == 1 and saved['earlier_hardware'] == [{'device': 'first'}]
    assert saved['hardware'] == {'device': 'second'}
    with pytest.raises(ValueError, match='already complete'):
        module.execute(plan, {'device': 'third'}, resume=True)


def test_resume_refuses_an_output_from_a_different_plan(handoff, monkeypatch):
    module = _module()
    plan = _plan(module, handoff, monkeypatch)
    _fake_stages(module, plan, monkeypatch, fail_at='global-nf4-capability-none')
    with pytest.raises(subprocess.CalledProcessError):
        module.execute(plan, {})
    changed = dict(plan, stages=[*plan['stages'], {'name': 'extra', 'args': ['status']}])
    with pytest.raises(ValueError, match='different plan'):
        module.execute(changed, {}, resume=True)


def test_resume_flag_is_only_for_execute(handoff):
    result = run_handoff(handoff, '--dry-run', '--resume')
    assert result.returncode != 0 and '--resume' in result.stderr


def test_router_execution_requires_explicit_network_authorization_before_work(handoff):
    result = run_handoff(handoff, '--execute', '--judges', 'router')
    assert result.returncode != 0
    assert '--allow-router' in result.stderr
    assert not handoff['output'].exists()
