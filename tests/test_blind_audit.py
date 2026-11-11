"""scripts/blind_audit.py: readers never see the judges' labels, and each sample is scored as what it is."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('blind_audit', ROOT / 'scripts' / 'blind_audit.py')
blind_audit = importlib.util.module_from_spec(SPEC)
sys.modules['blind_audit'] = blind_audit
SPEC.loader.exec_module(blind_audit)

TAGS = ['global', 'earth']
ALL_TAGS = TAGS + ['fire']
IDS = [f'p{i}' for i in range(10)]


def judged_row(item: str, preamble: str, label: str, lang: str = 'jv') -> dict:
    return {'id': item, 'preamble': preamble, 'lang': lang, 'kind': 'harmful', 'safety_label': label,
            'safety_votes': {'a': 'safe', 'b': 'unsafe'}, 'degenerate': False}


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def run(tmp_path: Path) -> Path:
    write(tmp_path / 'data' / 'heldout.jsonl', [{'id': item, 'text': f'prompt {item}'} for item in IDS])
    for tag in ALL_TAGS:
        rows, answers = [], []
        for preamble in ('default', 'none'):
            for number, item in enumerate(IDS):
                split = (tag == 'earth' and number < 3) or (tag == 'global' and number == 0)
                rows.append(judged_row(item, preamble, 'disputed' if split else 'safe'))
                answers.append({'id': item, 'preamble': preamble, 'response': f'{tag} answer {item}'})
            rows.append(judged_row('other', preamble, 'disputed', lang='bn'))
        write(tmp_path / 'judged' / f'{tag}.local.jsonl', rows)
        write(tmp_path / 'gens' / f'{tag}.jsonl', answers)
    return tmp_path


def label_everything(folder: Path, label: str = 'safe') -> None:
    key = blind_audit.read_rows(folder / 'key.jsonl')
    write(folder / 'labels_a.jsonl', [{'audit_id': row['audit_id'], 'label': label, 'confidence': 'high', 'reason': 'x'}
                                      for row in key])


def test_the_random_sample_pairs_models_and_hides_the_labels(run):
    assert blind_audit.sample(run, TAGS, 'jv', 4, 1)['items'] == 16
    blind = blind_audit.read_rows(run / 'audit' / 'blind.jsonl')
    assert all(set(row) == {'audit_id', 'language', 'prompt', 'response'} for row in blind)
    key = blind_audit.read_rows(run / 'audit' / 'key.jsonl')
    for preamble in ('default', 'none'):
        chosen = [{row['id'] for row in key if row['tag'] == tag and row['preamble'] == preamble} for tag in TAGS]
        assert chosen[0] == chosen[1] and len(chosen[0]) == 4


def test_the_disputed_sample_holds_every_split_answer_and_nothing_else(run):
    assert blind_audit.sample(run, TAGS, 'jv', 4, 1, disputed=True)['items'] == 8
    key = blind_audit.read_rows(run / 'audit' / 'disputed' / 'key.jsonl')
    assert {row['safety_label'] for row in key} == {'disputed'}
    assert sum(row['tag'] == 'earth' for row in key) == 6 and sum(row['tag'] == 'global' for row in key) == 2
    assert not (run / 'audit' / 'key.jsonl').exists()


def test_the_disputed_sample_gives_no_gap_because_its_rows_are_not_paired(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1, disputed=True)
    label_everything(run / 'audit' / 'disputed', 'unsafe')
    result = blind_audit.score(run, 'global', resamples=100, disputed=True)
    assert result['sample'] == 'disputed rows only'
    assert result['cells']['earth/default']['unsafe'] == 1.0
    assert all('gap_to_reference' not in cell for cell in result['cells'].values())


def test_the_random_sample_reports_a_gap_to_the_reference(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    label_everything(run / 'audit')
    result = blind_audit.score(run, 'global', resamples=100)
    assert result['sample'] == 'random'
    assert result['cells']['earth/default']['gap_to_reference']['n'] == 4


def test_scoring_refuses_a_partly_labelled_sample(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    write(run / 'audit' / 'labels_a.jsonl', [{'audit_id': 0, 'label': 'safe'}])
    with pytest.raises(ValueError, match='cover the sample'):
        blind_audit.score(run, 'global', resamples=100)


def chosen_ids(folder: Path, preamble: str) -> set[str]:
    return {row['id'] for row in blind_audit.read_rows(folder / 'key.jsonl') if row['preamble'] == preamble}


def test_a_named_read_of_one_condition_takes_the_same_prompts_as_the_first_read(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    assert blind_audit.sample(run, ['fire'], 'jv', 4, 1, name='regions', preambles=['none'])['items'] == 4
    folder = run / 'audit' / 'regions'
    assert chosen_ids(folder, 'none') == chosen_ids(run / 'audit', 'none')
    assert chosen_ids(folder, 'default') == set()


def test_scoring_can_compare_models_read_at_different_times(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    blind_audit.sample(run, ['fire'], 'jv', 4, 1, name='regions')
    label_everything(run / 'audit')
    label_everything(run / 'audit' / 'regions', 'invalid')
    result = blind_audit.score(run, 'global', resamples=100, name='regions', include=('main',))
    gap = result['cells']['fire/default']['gap_to_reference']
    assert gap['n'] == 4 and abs(gap['delta']) == 1.0


def test_the_same_answers_read_twice_cannot_be_merged(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    blind_audit.sample(run, TAGS, 'jv', 4, 1, name='second')
    label_everything(run / 'audit')
    label_everything(run / 'audit' / 'second')
    with pytest.raises(ValueError, match='read twice'):
        blind_audit.score(run, 'global', resamples=100, name='second', include=('main',))


def test_two_reads_are_compared_answer_by_answer(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    blind_audit.sample(run, TAGS, 'jv', 4, 1, name='second')
    label_everything(run / 'audit')
    label_everything(run / 'audit' / 'second')
    assert blind_audit.agree(run, 'main', 'second')['agreement'] == 1.0
    label_everything(run / 'audit' / 'second', 'invalid')
    result = blind_audit.agree(run, 'main', 'second')
    assert result['agreement'] == 0.0 and result['n'] == 16 and result['table'] == {'safe/invalid': 16}


def test_a_partial_read_says_how_many_answers_went_unlabelled(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    key = blind_audit.read_rows(run / 'audit' / 'key.jsonl')
    write(run / 'audit' / 'labels_a.jsonl', [{'audit_id': row['audit_id'], 'label': 'safe'} for row in key[:10]])
    result = blind_audit.score(run, 'global', resamples=100, partial=True)
    assert result['unlabelled'] == 6 and result['labels'] == 10
    assert sum(cell['n'] for cell in result['cells'].values()) == 10


def test_a_partial_read_still_refuses_a_label_outside_the_rubric(run):
    blind_audit.sample(run, TAGS, 'jv', 4, 1)
    write(run / 'audit' / 'labels_a.jsonl', [{'audit_id': 0, 'label': 'maybe'}])
    with pytest.raises(ValueError, match='cover the sample'):
        blind_audit.score(run, 'global', resamples=100, partial=True)
