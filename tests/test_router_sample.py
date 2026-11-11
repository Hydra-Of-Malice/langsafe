"""Offline checks for the router-judge sample: no network or billed call is made."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'router_sample.py'


def _module():
    loader = importlib.util.spec_from_file_location('router_sample', SCRIPT)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def _rows():
    return [{'id': f'p{i}::{lang}', 'lang': lang, 'kind': kind, 'preamble': pre, 'response': f'answer {i}'}
            for lang in ('th', 'jv') for kind in ('harmful', 'benign') for pre in ('default', 'none') for i in range(30)]


def test_sample_fills_every_cell_and_is_repeatable():
    module = _module()
    key = lambda row: f"{row['id']}|{row['preamble']}"
    first = module.stratified_sample(_rows(), 5, 7, key)
    assert len(first) == 2 * 2 * 2 * 5
    cells = {(r['lang'], r['kind'], r['preamble']) for r in first}
    assert len(cells) == 8
    assert first == module.stratified_sample(list(reversed(_rows())), 5, 7, key)
    assert first != module.stratified_sample(_rows(), 5, 8, key)


def test_small_cells_give_what_they_have():
    module = _module()
    rows = [r for r in _rows() if not (r['lang'] == 'jv' and r['kind'] == 'harmful')][:-25]
    sample = module.stratified_sample(rows, 10, 1, lambda row: row['id'] + row['preamble'])
    assert sum(1 for r in sample if (r['lang'], r['kind'], r['preamble']) == ('jv', 'benign', 'none')) == 5


def test_usage_counts_each_billed_call_once_even_from_many_threads(monkeypatch, tmp_path):
    """The judge sends requests from 8 threads; the first full sample's counter stacked wrappers and counted calls
    about five times over. Every billed call must count exactly once, and cache hits not at all."""
    import time

    module = _module()
    from langsafe import llm_client
    import openai

    billed = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            def create(**request):
                time.sleep(0.002)  # let threads overlap
                billed.append(request['messages'][0]['content'])
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"label": "safe"}'))],
                                       usage=SimpleNamespace(prompt_tokens=600, completion_tokens=12))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr(openai, 'OpenAI', FakeOpenAI)
    monkeypatch.setenv('HF_TOKEN', 'not-a-real-token')
    monkeypatch.setattr(llm_client.ChatClient, '__init__', llm_client.ChatClient.__init__)
    module.record_usage()
    module.record_usage()  # main() calls it once per tag; it must not wrap twice
    client = llm_client.ChatClient('https://router.example/v1', 'judge-32b', cache_path=tmp_path / 'cache.sqlite')
    batch = [[{'role': 'user', 'content': f'item {i % 40}'}] for i in range(120)]  # 40 distinct, 80 repeats
    client.chat_many(batch)
    client.chat_many(batch)  # all cached now
    client.close()
    assert dict(module.USAGE['judge-32b']) == {'calls': len(billed), 'prompt_tokens': 600 * len(billed),
                                               'completion_tokens': 12 * len(billed)}
    assert 40 <= len(billed) < 120


def _judged(judge_set, n, label=lambda i: 'full_refusal' if i % 3 else 'full_compliance'):
    return [{'id': f'p{i}', 'lang': 'th' if i % 2 else 'jv', 'kind': 'harmful', 'preamble': 'default',
             'prompt': f'prompt {i}', 'response': f'answer {i}', 'judge_set': judge_set,
             'judges': {'a': {}, 'b': {}}, 'rubric_hash': 'rubric', 'label': label(i),
             'safety_label': 'safe' if i % 4 else 'unsafe'} for i in range(n)]


def test_calibration_file_covers_the_whole_local_evaluation(tmp_path):
    import json as _json
    from langsafe.eval.judge_run import evaluation_digest
    from langsafe.provenance import Ledger

    module = _module()
    run, out = tmp_path / 'earth', tmp_path / 'earth-router-sample'
    local = _judged('local', 40)
    (run / 'judged').mkdir(parents=True)
    (out / 'judged').mkdir(parents=True)
    (run / 'judged' / 'earth.local.jsonl').write_text(''.join(_json.dumps(r) + '\n' for r in local))
    summary = run / 'judged' / 'earth.local.summary.json'
    summary.write_text(_json.dumps({'evaluation_digest': evaluation_digest(local)}))
    router = _judged('router', 40)[::4]  # the sample: every fourth answer
    (out / 'judged' / 'earth.router-sample.jsonl').write_text(''.join(_json.dumps(r) + '\n' for r in router))
    path = Path(module.write_calibration(run, out, 'earth', 20, replace=False))
    result = _json.loads(path.read_text())
    assert path == run / 'calibration' / 'earth.router-comparison.json'
    assert result['verified'] and result['shared_rows'] == 10 and result['sample']['answers'] == 10
    assert result['local_evaluation_digest'] == evaluation_digest(local)
    assert Ledger(out / 'ledger.jsonl').verify()[0]
    with pytest.raises(SystemExit, match='replace'):
        module.write_calibration(run, out, 'earth', 20, replace=False)
    summary.write_text(_json.dumps({'evaluation_digest': 'another evaluation'}))
    with pytest.raises(SystemExit, match='no longer match'):
        module.write_calibration(run, out, 'earth', 20, replace=True)


def test_content_agreement_skips_disputed_rows():
    module = _module()
    base = {'prompt': 'q', 'kind': 'harmful', 'preamble': 'default'}
    local = [dict(base, id='1', lang='th', response='r1', safety_label='safe'),
             dict(base, id='2', lang='th', response='r2', safety_label='unsafe'),
             dict(base, id='3', lang='th', response='r3', safety_label='disputed')]
    router = [dict(row, safety_label=label) for row, label in zip(local, ('safe', 'safe', 'safe'))]
    result = module.content_agreement(local, router)['by_lang']['th']
    assert result['shared'] == 3 and result['decided_by_both'] == 2 and result['agreement'] == 0.5
    assert result['local_safe'] == 1 and result['router_safe'] == 2
