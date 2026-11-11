"""scripts/token_fertility.py: reads benign prompts only, and divides closed tokens by Tiny Aya tokens per language."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('token_fertility', ROOT / 'scripts' / 'token_fertility.py')
token_fertility = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(token_fertility)


def write_data(run: Path) -> Path:
    (run / 'data').mkdir(parents=True)
    rows = {'heldout': [('jv', 'benign', 'aa bb'), ('jv', 'harmful', 'never read'), ('th', 'benign', 'cc')],
            'repair': [('jv', 'benign', 'dd ee ff')]}
    for split, items in rows.items():
        lines = [json.dumps({'lang': lang, 'kind': kind, 'text': text}) for lang, kind, text in items]
        (run / 'data' / f'{split}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return run


def test_only_benign_prompts_from_both_splits_are_read(tmp_path):
    texts = token_fertility.benign_texts(write_data(tmp_path))
    assert texts == {'jv': ['aa bb', 'dd ee ff'], 'th': ['cc']}


def test_fertility_is_closed_tokens_over_tiny_aya_tokens(tmp_path):
    texts = token_fertility.benign_texts(write_data(tmp_path))
    words = lambda text: len(text.split())
    letters = lambda text: len(text.replace(' ', ''))
    result = token_fertility.fertility(texts, words, letters)
    assert result['jv'] == {'prompts': 2, 'tiny_aya_tokens': 5, 'closed_tokens': 10, 'closed_per_tiny_aya': 2.0}
    assert result['th']['closed_per_tiny_aya'] == 2.0


def test_a_language_without_tokens_is_refused():
    with pytest.raises(ValueError, match='jv'):
        token_fertility.fertility({'jv': []}, len, len)
