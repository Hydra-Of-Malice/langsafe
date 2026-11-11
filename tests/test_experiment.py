"""Behavioral checks for experiment isolation, resume, and repair-only selection."""
from pathlib import Path

import pytest


def test_spec_can_be_selected_per_process_without_mutating_canonical(monkeypatch, tmp_path):
    from langsafe.config import load_spec
    path = tmp_path / 'development.yaml'
    path.write_text('spec_version: 9\nlanguages:\n  fire: [en, hi]\n')
    monkeypatch.setenv('LANGSAFE_SPEC', str(path))
    assert load_spec()['spec_version'] == 9


def test_prompt_selection_cannot_mix_tuning_with_heldout():
    from langsafe.experiment import select_prompts
    from langsafe.data.records import Prompt
    rows = [Prompt(id=f'{split}-{kind}-{i}::{lang}', group=f'{split}-{kind}-{i}', lang=lang,
                   kind=kind, text=f'{kind} prompt {i}', source='test', source_id=str(i),split=split)
            for split in ('repair','heldout') for kind in ('harmful','benign')
            for lang in ('en','hi','bn') for i in range(3)]
    selected = select_prompts(rows, ['en','hi'], 'repair', limit=2)
    assert len(selected) == 8
    assert {p.split for p in selected} == {'repair'}
    assert {p.lang for p in selected} == {'en','hi'}
    with pytest.raises(ValueError):
        select_prompts(rows,['en'],'heldout',limit=-1)


def test_stage_cache_requires_artifact_content_and_settings(tmp_path):
    from langsafe.experiment import ExperimentState
    artifact = tmp_path/'scores.json'
    artifact.write_text('{"accuracy":0.5}')
    state = ExperimentState(tmp_path/'experiment.json', {'family':'fire','spec':'abc'})
    assert not state.completed('baseline', {'model':'fire'})
    state.record('baseline', {'model':'fire'}, [artifact], {'rows':10})
    assert ExperimentState(state.path, {'family':'fire','spec':'abc'}).completed('baseline', {'model':'fire'})
    assert not state.completed('baseline', {'model':'global'})
    artifact.write_text('{"accuracy":0.9}')
    assert not state.completed('baseline', {'model':'fire'})
    with pytest.raises(ValueError,match='identity'):
        ExperimentState(state.path, {'family':'earth','spec':'abc'})


def test_generation_cache_does_not_return_another_prompt_identity(tmp_path):
    from langsafe.gen.generate import Generator
    from langsafe.data.records import Prompt
    generator=object.__new__(Generator)
    generator.fp='fixed-weights'
    generator.decoding={'do_sample':False,'max_new_tokens':64}
    a=Prompt(id='a::en',group='a',lang='en',kind='benign',text='identical text',source='test',source_id='a',split='repair')
    b=Prompt(id='b::en',group='b',lang='en',kind='benign',text='identical text',source='test',source_id='b',split='heldout')
    assert generator._key(a,'default') != generator._key(b,'default')


def test_report_never_calls_missing_human_calibration_a_validated_result(monkeypatch, tmp_path):
    import json
    from langsafe.experiment_report import build_report
    monkeypatch.setenv('LANGSAFE_SPEC', str(Path(__file__).parents[1] / 'configs/spec.yaml'))
    judged=tmp_path/'judged'; judged.mkdir()
    for tag in ('global','fire'):
        rows=[{'id':f'{kind}::{lang}','lang':lang,'kind':kind,'preamble':'default','label':'full_refusal' if kind=='harmful' else 'full_compliance',
               'model_tag':tag,'fingerprint':tag,'judges':{'aya':'full_refusal','qwen':'full_refusal'}}
              for lang in ('en','hi','bn') for kind in ('harmful','benign')]
        (judged/f'{tag}.local.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    output=build_report(['global','fire'],'fire','local',tmp_path)
    text=output.read_text()
    assert 'PROVISIONAL' in text
    assert 'human' in text.lower()
    public=json.loads(output.with_suffix('.json').read_text())
    assert public['validated'] is False
    assert public['empirical_repair_success'] is False


def test_experiment_refuses_a_changed_completed_stage(tmp_path):
    from langsafe.experiment_run import StageRunner
    output=tmp_path/'output.json'
    runner=StageRunner(tmp_path, {'family':'fire','spec':'x'}, python='/missing/python')
    marker={'step':'example'}
    output.write_text('{"complete":true}')
    runner.state.record('example',marker,[output])
    assert runner.state.completed('example',marker)
    output.unlink()
    assert not runner.state.completed('example',marker)


def test_failed_subprocess_marks_current_stage_failed(tmp_path):
    import json
    from langsafe.experiment_run import StageRunner

    runner = StageRunner(tmp_path, {'test': 'failed-stage'})
    with pytest.raises(RuntimeError, match='failed'):
        runner.run('invalid-command', ['invalid-command'], [tmp_path / 'output.json'])
    current = json.loads((tmp_path / 'current-stage.json').read_text())
    failure = json.loads((tmp_path / 'last-failure.json').read_text())
    assert current['status'] == 'failed'
    assert current['stage'] == 'invalid-command'
    assert current['exit_code'] == failure['exit_code'] == 2
    assert 'invalid-command' not in runner.state.data['stages']


def test_steering_strength_and_direction_are_part_of_generation_identity():
    from langsafe.gen.generate import Generator
    from langsafe.data.records import Prompt
    generator=object.__new__(Generator)
    generator.fp='same-weights'; generator.decoding={'do_sample':False}
    prompt=Prompt(id='a::en',group='a',lang='en',kind='benign',text='hello',source='test',source_id='a')
    generator.variant={'steering_sha256':'vector-a','strength':0.5}
    first=generator._key(prompt,'default')
    generator.variant={'steering_sha256':'vector-a','strength':1.0}
    assert generator._key(prompt,'default') != first


def test_generation_cache_uses_complete_checkpoint_identity(tmp_path):
    from argparse import Namespace
    from langsafe.cli import _variant
    from langsafe.data.records import Prompt
    from langsafe.gen.generate import Generator

    (tmp_path / 'config.json').write_text('{}')
    weights = tmp_path / 'model.safetensors'
    weights.write_bytes(b'original-weights')
    args = Namespace(model=str(tmp_path), quant=None, steering=None, strength=None)
    generator = object.__new__(Generator)
    # A sampled weight fingerprint can miss an edited tensor. Complete source
    # identity must still prevent a stale completion from entering evaluation.
    generator.fp = 'same-sampled-weights'
    generator.decoding = {'do_sample': False}
    prompt = Prompt(id='p::en', group='p', lang='en', kind='benign', text='Hello', source='test', source_id='p')
    generator.variant = _variant(args)
    before = generator._key(prompt, 'default')
    weights.write_bytes(b'updated-weights')
    generator.variant = _variant(args)
    assert generator._key(prompt, 'default') != before


def test_each_answer_records_its_precision_and_cannot_share_a_cache_entry(tmp_path):
    from argparse import Namespace
    from langsafe.cli import _variant
    from langsafe.data.records import Prompt
    from langsafe.gen.generate import Generator

    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    full = _variant(Namespace(model=str(tmp_path), quant=None, steering=None, strength=None))
    nf4 = _variant(Namespace(model=str(tmp_path), quant='nf4', steering=None, strength=None))
    assert 'quant' not in full        # keeps every full-precision cache entry valid
    assert nf4['quant'] == 'nf4'
    generator = object.__new__(Generator)
    generator.fp = 'same-sampled-weights'     # NF4 and fp16 really do share a fingerprint
    generator.decoding = {'do_sample': False}
    prompt = Prompt(id='p::en', group='p', lang='en', kind='benign', text='Hello', source='test', source_id='p')
    generator.variant = full
    before = generator._key(prompt, 'default')
    generator.variant = nf4
    assert generator._key(prompt, 'default') != before


def test_default_ledger_uses_the_experiment_directory(monkeypatch,tmp_path):
    from langsafe.provenance import Ledger
    monkeypatch.setenv('LANGSAFE_RUNS_DIR',str(tmp_path/'development'))
    assert Ledger.default().path==tmp_path/'development'/'ledger.jsonl'


def test_cli_runs_directory_selects_its_own_dataset(monkeypatch, tmp_path):
    import json
    from langsafe import cli
    from langsafe.data.build import load_split
    directory = tmp_path / 'experiment'
    (directory / 'data').mkdir(parents=True)
    (directory / 'data' / 'heldout.jsonl').write_text(json.dumps({
        'id': 'isolated::en', 'group': 'isolated', 'lang': 'en', 'kind': 'benign', 'text': 'Hello',
        'source': 'test', 'source_id': 'isolated', 'split': 'heldout'}) + '\n')
    monkeypatch.delenv('LANGSAFE_DATA_DIR', raising=False)
    monkeypatch.delenv('LANGSAFE_RUNS_DIR', raising=False)
    monkeypatch.setattr(cli, 'RUNS', cli.RUNS)
    monkeypatch.setattr(cli, 'cmd_ledger_verify', lambda args: None)
    cli.main(['--runs-dir', str(directory), 'ledger-verify'])
    assert [row.id for row in load_split('heldout')] == ['isolated::en']


def test_stage_manifest_covers_every_weight_shard(tmp_path):
    from langsafe.experiment import ExperimentState
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'config.json').write_text('{}')
    weights=checkpoint/'model.safetensors';weights.write_bytes(b'original-weights')
    state=ExperimentState(tmp_path/'state.json',{'experiment':'test'})
    state.record('repair',{'alpha':0.1},[checkpoint])
    assert state.completed('repair',{'alpha':0.1})
    weights.write_bytes(b'changed-weights')
    assert not state.completed('repair',{'alpha':0.1})


def test_stage_cache_detects_new_files_in_artifact_directory(tmp_path):
    from langsafe.experiment import ExperimentState
    directory = tmp_path / 'checkpoint'
    directory.mkdir()
    (directory / 'config.json').write_text('{}')
    state = ExperimentState(tmp_path / 'state.json', {'test': 'directory'})
    state.record('weights', {}, [directory])
    assert state.completed('weights', {})
    (directory / 'model.safetensors').write_bytes(b'new-preferred-weights')
    assert not state.completed('weights', {})


def test_stage_cache_tracks_hub_overrides_and_new_calibration(monkeypatch, tmp_path):
    from io import StringIO
    import langsafe.experiment_run as experiment
    checkpoint = tmp_path / 'tiny-aya-fire'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{}')
    weights = checkpoint / 'model.safetensors'
    weights.write_bytes(b'old-weights')
    monkeypatch.setenv('LANGSAFE_MODELS', str(tmp_path))
    runner = experiment.StageRunner(tmp_path / 'run', {'test': 'model-identity'})
    artifact = tmp_path / 'capability.json'
    calibration = tmp_path / 'calibration.json'
    count = []

    class Process:
        def __init__(self, *args, **kwargs):
            count.append(1)
            artifact.write_text(str(len(count)))
            self.stdout = StringIO('')

        def wait(self):
            return 0

    monkeypatch.setattr(experiment.subprocess, 'Popen', Process)
    args = ['capability', '--model', 'CohereLabs/tiny-aya-fire', '--tag', 'test']
    runner.run('capability', args, [artifact], optional_inputs=[calibration])
    runner.run('capability', args, [artifact], optional_inputs=[calibration])
    assert artifact.read_text() == '1'
    weights.write_bytes(b'updated-weights')
    runner.run('capability', args, [artifact], optional_inputs=[calibration])
    assert artifact.read_text() == '2'
    calibration.write_text('{"complete":true}')
    runner.run('capability', args, [artifact], optional_inputs=[calibration])
    assert artifact.read_text() == '3'


def test_conversion_protocol_change_invalidates_saved_export(monkeypatch, tmp_path):
    from io import StringIO
    import langsafe.experiment_run as experiment
    from langsafe.quant import gguf

    monkeypatch.setattr(experiment, 'checkpoint_identity', lambda model: {'model': model})
    monkeypatch.setattr(gguf, 'CONVERSION_PROTOCOL', 'old-eos-policy', raising=False)
    artifact = tmp_path / 'export.gguf'

    class Process:
        def __init__(self, *args, **kwargs):
            artifact.write_text(gguf.CONVERSION_PROTOCOL)
            self.stdout = StringIO('')

        def wait(self):
            return 0

    monkeypatch.setattr(experiment.subprocess, 'Popen', Process)
    runner = experiment.StageRunner(tmp_path / 'run', {'test': 'conversion-version'})
    args = ['convert', '--model', 'test-checkpoint', '--output', str(artifact)]
    runner.run('convert', args, [artifact])
    monkeypatch.setattr(gguf, 'CONVERSION_PROTOCOL', 'hf-generation-eos')
    runner.run('convert', args, [artifact])
    assert artifact.read_text() == 'hf-generation-eos'


def test_orchestration_evaluates_three_q4_models_and_stops_at_requested_stage(monkeypatch, tmp_path):
    import json
    from io import BytesIO
    from types import SimpleNamespace
    import langsafe.experiment_run as experiment
    calls = []

    class Runner:
        def __init__(self, directory, identity):
            self.directory = Path(directory)
            self.env = {'LANGSAFE_RUNS_DIR': str(directory), 'LANGSAFE_DATA_DIR': str(directory / 'data')}
            (self.directory / 'logs').mkdir(parents=True)

        def run(self, name, args, outputs, inputs=(), **kwargs):
            calls.append((name, args, outputs, kwargs))
            rows = [{'id': kind, 'lang': 'en', 'kind': kind, 'split': 'repair', 'preamble': 'default',
                     'label': 'full_refusal' if kind == 'harmful' else 'full_compliance',
                     'safety_label': 'safe',
                     'response': 'I cannot help with that.' if kind == 'harmful' else 'Hello!'}
                    for kind in ('harmful', 'benign')]
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == '.jsonl':
                    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                elif path.parent.name == 'capability':
                    path.write_text(json.dumps({'languages': {'en': {'accuracy': .5}}}))
                elif path.name == 'manifest.json':
                    path.write_text(json.dumps({'coverage': {'complete': True}}))
                elif path.suffix:
                    path.write_text('{}')
                else:
                    path.mkdir(exist_ok=True)
            if args[0] == 'circuit':
                lang = args[args.index('--lang') + 1]
                directory = outputs[0]
                scores = directory / 'circuit_scores.json'
                scores.write_text(json.dumps({'node_scores': {'MLP 1': 1.0}}))
                (directory / 'summary.json').write_text(json.dumps({'scores_json': str(scores)}))

    class Server:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, **kwargs):
            return 0

    class Response(BytesIO):
        status = 200

    monkeypatch.setenv('LANGSAFE_RUNS_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('LANGSAFE_DATA_DIR', str(tmp_path / 'run' / 'data'))
    monkeypatch.setenv('LANGSAFE_SPEC', str(Path(__file__).parents[1] / 'configs/development.yaml'))
    monkeypatch.setattr(experiment, 'StageRunner', Runner)
    monkeypatch.setattr(experiment.subprocess, 'Popen', lambda *a, **k: Server())
    monkeypatch.setattr(experiment, 'urlopen', lambda *a, **k: Response())
    experiment.run_experiment('fire', 'development', stop_after='quantization')
    names = [name for name, *_ in calls]
    circuits = [args for _, args, _, _ in calls if args[0] == 'circuit']
    assert circuits and all(args[args.index('--openers-tag') + 1] == 'global-repair' for args in circuits)
    assert names.index('global-repair-judge') < names.index('global-circuit-en')     # the openers are read before circuits
    assert names.count('global-repair-judge') == 2      # the later call is the same stage, so it comes from the cache
    converted = {args[args.index('--model') + 1] for _, args, _, _ in calls if args[0] == 'convert'}
    assert 'CohereLabs/tiny-aya-global' in converted and 'CohereLabs/tiny-aya-fire' in converted
    assert len(converted) == 3
    q4_capability = [(args[args.index('--tag') + 1], args[args.index('--preamble') + 1])
                     for _, args, _, _ in calls if args[0] == 'capability' and '--gguf' in args]
    assert ('global-q4', 'default') in q4_capability and ('global-q4', 'none') in q4_capability
    assert ('fire-q4', 'default') in q4_capability and ('fire-q4', 'none') in q4_capability
    assert len(q4_capability) == 6
    assert not any(args[0] == 'report' for _, args, _, _ in calls)


def test_target_language_is_the_gap_not_the_floor():
    from langsafe.repair.select import target_language
    reference = {'en': {'harmful': .8}, 'hi': {'harmful': .7}, 'bn': {'harmful': .2}}
    drifted = {'en': {'harmful': .8}, 'hi': {'harmful': .4}, 'bn': {'harmful': .15}}
    assert target_language(reference, drifted)[0] == 'hi'
    assert target_language(drifted, reference)[0] == 'bn'      # no positive gap: lowest refusal


def test_objective_punishes_a_broken_model_that_refuses_more():
    from langsafe.repair.select import objective, pick
    reference = {'hi': {'harmful': .8, 'benign': 0., 'unclear': 0.}}
    drifted = {'hi': {'harmful': .4, 'benign': 0., 'unclear': 0.}}
    capability = {'hi': .5}
    healthy = {'hi': {'harmful': .6, 'benign': 0., 'unclear': 0.}}
    good = objective(healthy, {'hi': .5}, reference, drifted, capability, 'hi')
    broken = objective({'hi': {'harmful': .8, 'benign': .1, 'unclear': .4}}, {'hi': .2}, reference, drifted, capability, 'hi')
    assert abs(good['score'] - .2) < 1e-9 and broken['score'] < 0
    assert pick({.5: good, .25: good, 1.: broken})[0] == .25


def test_a_stage_with_no_artifacts_completes_and_reruns_on_the_real_runner(tmp_path, monkeypatch):
    import subprocess
    import langsafe.experiment_run as experiment

    class Done:
        stdout = iter(['ok\n'])

        def wait(self):
            return 0

    monkeypatch.setattr(experiment.subprocess, 'Popen', lambda *a, **k: Done())
    runner = experiment.StageRunner(tmp_path / 'run', {'family': 'fire'})
    runner.run('warm', ['judge-warm', '--tags', 'a'], [])
    assert 'warm' not in runner.state.data.get('stages', {})


def test_a_limited_sample_mixes_sources_and_stays_parallel_across_languages():
    from langsafe.data.records import Prompt
    from langsafe.experiment import select_prompts

    rows = [Prompt(id=f'{source}-{i}::{lang}', group=f'{source}-{i}', lang=lang, kind='harmful', text='t', source=source,
                   source_id=str(i), split='repair')
            for source in ('aya', 'multijail', 'xstest') for i in range(30) for lang in ('en', 'hi')]
    picked = select_prompts(rows, ['en', 'hi'], 'repair', limit=20)
    assert {p.source for p in picked} == {'aya', 'multijail', 'xstest'}
    by_lang = {lang: {p.group for p in picked if p.lang == lang} for lang in ('en', 'hi')}
    assert by_lang['en'] == by_lang['hi'] and len(by_lang['en']) == 20


def test_a_data_build_with_an_empty_cell_stops_before_any_generation(monkeypatch, tmp_path):
    import json
    import pytest
    import langsafe.experiment_run as experiment
    calls = []

    class Runner:
        def __init__(self, directory, identity):
            self.directory = Path(directory)
            self.env = {'LANGSAFE_RUNS_DIR': str(directory), 'LANGSAFE_DATA_DIR': str(directory / 'data')}

        def run(self, name, args, outputs, inputs=(), **kwargs):
            calls.append(name)
            outputs[0].parent.mkdir(parents=True, exist_ok=True)
            outputs[0].write_text(json.dumps({'coverage': {'complete': False, 'missing': ['jv/benign'],
                                                           'missing_split_cells': []}}))

    monkeypatch.setenv('LANGSAFE_RUNS_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('LANGSAFE_DATA_DIR', str(tmp_path / 'run' / 'data'))
    monkeypatch.setenv('LANGSAFE_SPEC', str(Path(__file__).parents[1] / 'configs/development.yaml'))
    monkeypatch.setattr(experiment, 'StageRunner', Runner)
    with pytest.raises(ValueError, match='jv/benign'):
        experiment.run_experiment('earth', 'development')
    assert calls == ['data']


def test_fingerprint_is_unchanged_by_sampling_before_widening():
    """Finished runs carry these fingerprints, so the memory fix must not move them."""
    import hashlib
    import torch
    from langsafe.gen.generate import model_fingerprint

    torch.manual_seed(0)
    model = torch.nn.Sequential(*[torch.nn.Linear(70, 90).half() for _ in range(20)])
    expected = hashlib.sha256()
    for i, (name, parameter) in enumerate(model.named_parameters()):
        if i % 16 == 0:
            flat = parameter.detach().float().flatten()       # the original order: widen everything, then sample
            expected.update(name.encode())
            expected.update(flat[:: max(1, flat.numel() // 4096)].cpu().numpy().tobytes())
    assert model_fingerprint(model) == expected.hexdigest()[:16]


def test_each_cuda_batch_starts_from_an_empty_cache(monkeypatch, tmp_path):
    """Blocks left by earlier batch lengths otherwise build up until VRAM spills, at 191 ms a step against 78."""
    from types import SimpleNamespace
    import torch
    import langsafe.gen.generate as generate
    from langsafe.data.records import Prompt

    events = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('freed'))
    monkeypatch.setattr(generate, 'render_prompt', lambda tokenizer, text, preamble: text)

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            return Encoded(input_ids=torch.zeros((len(texts), 2), dtype=torch.long))

        def decode(self, ids, skip_special_tokens=True):
            return 'answer'

    class Model:
        device = SimpleNamespace(type='cuda')
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, input_ids, **kwargs):
            events.append('generated')
            return torch.tensor([[0, 0, 5, 9]] * input_ids.shape[0])

    generator = object.__new__(generate.Generator)
    generator.model, generator.tok, generator.tag, generator.fp = Model(), Tokenizer(), 'tag', 'fingerprint'
    generator.decoding, generator.variant, generator._cache = {'batch_size': 1, 'do_sample': False}, {}, {}
    generator.cache_path = tmp_path / 'cache.jsonl'
    prompts = [Prompt(id=f'p{i}::en', group=f'p{i}', lang='en', kind='benign', text='Hello', source='test', source_id=f'p{i}')
               for i in range(2)]
    assert len(generator.run(prompts, 'default')) == 2
    assert events == ['freed', 'generated', 'freed', 'generated']
