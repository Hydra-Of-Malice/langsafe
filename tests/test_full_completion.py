"""Offline checks for safe full-run resource use and isolation."""
from pathlib import Path
import json

import pytest


def test_local_chat_cache_follows_active_experiment_directory(monkeypatch, tmp_path):
    from langsafe.llm_client import LocalChatClient
    monkeypatch.setattr('langsafe.gen.model.best_device', lambda: 'cpu')
    monkeypatch.setattr('langsafe.gen.model.resolve_model', lambda model: model)
    one = tmp_path / 'one'
    two = tmp_path / 'two'
    monkeypatch.setenv('LANGSAFE_RUNS_DIR', str(one))
    first = LocalChatClient('fixture/model', quant=None)
    first._cache.put('same-key', 'first-experiment')
    first.close()
    monkeypatch.setenv('LANGSAFE_RUNS_DIR', str(two))
    second = LocalChatClient('fixture/model', quant=None)
    try:
        assert second._cache.get('same-key') is None
        assert (one / 'llm_cache.sqlite').is_file()
        assert (two / 'llm_cache.sqlite').is_file()
    finally:
        second.close()


def test_full_resources_retire_only_generated_artifacts_after_preserving_receipt(tmp_path):
    from langsafe.experiment import ExperimentState
    state = ExperimentState(tmp_path / 'experiment.json', {'family': 'test'})
    output = tmp_path / 'checkpoints' / 'candidate'
    output.mkdir(parents=True)
    (output / 'config.json').write_text('{}')
    (output / 'model.safetensors').write_bytes(b'generated-weights')
    state.record('candidate-weights', {'recipe': 1}, [output])
    recorded = json.loads(state.path.read_text())['stages']['candidate-weights']
    state.retire('candidate-weights', [output], 'unselected candidate after saved repair evidence')
    assert not output.exists()
    current = json.loads(state.path.read_text())
    assert 'candidate-weights' not in current['stages']
    assert current['retired_stages']['candidate-weights'][-1]['record'] == recorded
    assert not state.completed('candidate-weights', {'recipe': 1})


def test_retirement_refuses_original_weights_and_changed_generated_files(tmp_path):
    from langsafe.experiment import ExperimentState
    run = tmp_path / 'run'
    run.mkdir()
    source = tmp_path / 'original.safetensors'
    source.write_bytes(b'original')
    state = ExperimentState(run / 'experiment.json', {'family': 'test'})
    state.record('original', {}, [source])
    with pytest.raises(ValueError, match='outside|generated|run directory'):
        state.retire('original', [source], 'do not delete originals')
    assert source.exists()
    output = run / 'checkpoint.gguf'
    output.write_bytes(b'generated')
    state.record('generated', {}, [output])
    output.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed|match|verified'):
        state.retire('generated', [output], 'do not delete changed artifacts')
    assert output.read_bytes() == b'changed'


def test_retired_candidate_reuses_only_exact_verified_repair_evidence(monkeypatch, tmp_path):
    from langsafe.experiment_run import StageRunner
    from langsafe.experiment import checkpoint_identity
    donor = tmp_path / 'donor'
    donor.mkdir()
    (donor / 'config.json').write_text('{}')
    (donor / 'model.safetensors').write_bytes(b'donor')
    spec = {'models': {'drifted': {'fire': str(donor)}, 'reference': str(donor), 'base': str(donor)}}
    monkeypatch.setattr('langsafe.experiment_run.load_spec', lambda: spec)
    runner = StageRunner(tmp_path / 'run', {'test': 'compact'})
    checkpoint = runner.directory / 'checkpoints' / 'candidate'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'config.json').write_text('{}')
    (checkpoint / 'model.safetensors').write_bytes(b'candidate')
    data = runner.directory / 'data.json'
    data.write_text('{"version":1}')
    producer_args = ['repair', '--family', 'fire', '--arm', 'resta', '--knob', '0.1']
    consumer_args = ['generate', '--model', str(checkpoint), '--tag', 'candidate-repair', '--split', 'repair']
    evidence = runner.directory / 'repair.jsonl'
    evidence.write_text('{"rate":0.5}\n')
    producer_settings = runner.settings(producer_args)
    runner.state.record('candidate-weights', producer_settings, [checkpoint])
    consumer_settings = runner.settings(consumer_args, [data])
    runner.state.record('candidate-repair', consumer_settings, [evidence])
    runner.retire_checkpoint('candidate-weights', checkpoint, 'candidate-repair')
    assert not checkpoint.exists()
    assert runner.reuse_retired_checkpoint('candidate-weights', producer_args, checkpoint,
                                           'candidate-repair', consumer_args, [data])
    evidence.write_text('{"rate":1.0}\n')
    assert not runner.reuse_retired_checkpoint('candidate-weights', producer_args, checkpoint,
                                               'candidate-repair', consumer_args, [data])
    evidence.write_text('{"rate":0.5}\n')
    data.write_text('{"version":2}')
    assert not runner.reuse_retired_checkpoint('candidate-weights', producer_args, checkpoint,
                                               'candidate-repair', consumer_args, [data])
    data.write_text('{"version":1}')
    (donor / 'model.safetensors').write_bytes(b'changed-source')
    assert not runner.reuse_retired_checkpoint('candidate-weights', producer_args, checkpoint,
                                               'candidate-repair', consumer_args, [data])
