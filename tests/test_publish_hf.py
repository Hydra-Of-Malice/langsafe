"""scripts/publish_hf.py: the release plan, its rule and its refusals. No network call is made."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('publish_hf', ROOT / 'scripts' / 'publish_hf.py')
publish = importlib.util.module_from_spec(SPEC)
sys.modules['publish_hf'] = publish   # dataclasses look their module up here
SPEC.loader.exec_module(publish)


def safety(delta: float, low: float, high: float) -> dict:
    return {'content_safe': {'delta': delta, 'ci': [low, high]}}


@pytest.fixture
def setup(tmp_path):
    run, export, cards = tmp_path / 'run', tmp_path / 'export', tmp_path / 'cards'
    arms = {'resta': ('earth-resta-0p1', 0.1, 'earth-resta-a0.1'), 'merge_back': ('earth-merge_back-0p5', 0.5, 'earth-merge-l0.5'),
            'circuit': ('earth-circuit-0p1', 0.1, 'earth-circuit-s0.1-top0.2')}
    selection = {'arms': {}, 'controls': [{'tag': 'earth-merge_back-0p5-rand1'}],
                 'selected_for_quantization': {'tag': 'earth-merge_back-0p5'}}
    for arm, (tag, knob, folder) in arms.items():
        checkpoint = run / 'checkpoints' / folder
        checkpoint.mkdir(parents=True)
        for name in ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'):
            (checkpoint / name).write_text(name, encoding='utf-8')
        selection['arms'][arm] = {'selected': {'tag': tag, 'knob': knob, 'model': f'D:\\run\\checkpoints\\{folder}'}}
    (run / 'checkpoints' / 'earth-resta-a0.1' / 'optimizer.pt').write_text('not for release', encoding='utf-8')
    steer = run / 'checkpoints' / 'earth-steer-default'
    steer.mkdir()
    (steer / 'refusal_direction.pt').write_text('direction', encoding='utf-8')
    (steer / 'meta.json').write_text('{}', encoding='utf-8')
    selection['arms']['steer'] = {'selected': {'tag': 'earth-steer-1p0', 'knob': 1.0, 'model': 'CohereLabs/tiny-aya-earth',
                                               'steering': str(steer)}}
    (run / 'quantized').mkdir()
    (run / 'quantized' / 'earth-merge_back-0p5-q4_k_m.gguf').write_text('gguf', encoding='utf-8')
    comparisons = {f'{tag}/{condition}': {'vs_drifted': {'safety': {'th': safety(-0.01, -0.05, 0.03)}}}
                   for tag, _, _ in arms.values() for condition in ('default', 'none')}
    comparisons['earth-steer-1p0/default'] = {'vs_drifted': {'safety': {'th': safety(0.0, -0.02, 0.02)}}}
    # Circuit is significantly less safe than Earth in Swahili without the preamble: it must stay private.
    comparisons['earth-circuit-0p1/none']['vs_drifted']['safety']['sw'] = safety(-0.04, -0.074, -0.011)
    (run / 'report.json').write_text(json.dumps({'paired_comparisons': comparisons}), encoding='utf-8')
    (run / 'selection.json').write_text(json.dumps(selection), encoding='utf-8')
    for name in ('data/heldout.jsonl', 'reports/judged/earth.local.summary.json', 'verdicts/earth.jsonl'):
        (export / name).parent.mkdir(parents=True, exist_ok=True)
        (export / name).write_text('{}', encoding='utf-8')
    (export / 'EXPORT_MANIFEST.json').write_text(json.dumps({'scan': {'result': 'passed'}}), encoding='utf-8')
    cards.mkdir()
    return run, export, cards


def test_the_plan_applies_the_release_rule_and_the_file_allowlist(setup):
    run, export, cards = setup
    repos, private = publish.plan('team', run=run, export=export, cards=cards)
    names = [repo.repo_id for repo in repos]
    assert names == ['team/tiny-aya-earth-langsafe-eval', 'team/tiny-aya-earth-langsafe-resta-0p1',
                     'team/tiny-aya-earth-langsafe-merge-back-0p5', 'team/tiny-aya-earth-langsafe-merge-back-0p5-gguf',
                     'team/tiny-aya-earth-langsafe-steer-1p0']
    assert any('earth-circuit-0p1' in item and 'sw none' in item for item in private)
    assert any('rand1' in item for item in private)
    resta = repos[1]
    assert 'optimizer.pt' not in resta.files and any('optimizer.pt' in p for p in resta.problems)
    assert set(repos[-1].files) == {'refusal_direction.pt', 'meta.json'}
    assert all(any('card missing' in p for p in repo.problems) for repo in repos)
    assert not any('not part of the export' in p for p in repos[0].problems)   # reports/judged summaries are the export's own


def test_a_stray_folder_in_the_export_or_an_unscanned_export_is_flagged(setup):
    run, export, cards = setup
    (export / 'gens').mkdir()
    (export / 'gens' / 'earth.jsonl').write_text('{}', encoding='utf-8')
    (export / 'EXPORT_MANIFEST.json').write_text(json.dumps({'scan': {'result': 'failed'}}), encoding='utf-8')
    dataset = publish.plan('team', run=run, export=export, cards=cards)[0][0]
    assert any('not part of the export: gens/earth.jsonl' in p for p in dataset.problems)
    assert any('no passed final scan' in p for p in dataset.problems)


def test_execute_needs_the_reviewed_digest_and_every_card(setup, monkeypatch):
    run, export, cards = setup
    import huggingface_hub

    class Forbidden:
        def __init__(self, *args, **kwargs):
            pytest.fail('no upload may start')

    monkeypatch.setattr(huggingface_hub, 'HfApi', Forbidden)
    repos, _ = publish.plan('team', run=run, export=export, cards=cards)
    with pytest.raises(SystemExit, match='changed since it was reviewed'):
        publish.execute(repos, 'not-the-digest')
    with pytest.raises(SystemExit, match='card missing'):
        publish.execute(repos, publish.digest(repos))


def test_the_digest_follows_the_files(setup):
    run, export, cards = setup
    first = publish.digest(publish.plan('team', run=run, export=export, cards=cards)[0])
    assert first == publish.digest(publish.plan('team', run=run, export=export, cards=cards)[0])
    (run / 'quantized' / 'earth-merge_back-0p5-q4_k_m.gguf').write_text('a different gguf', encoding='utf-8')
    assert first != publish.digest(publish.plan('team', run=run, export=export, cards=cards)[0])
