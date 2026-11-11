"""scripts/fair_circuit_baseline.py: the shuffle keeps every MLP block, deals out the head scores at random, and the
evaluation reads exactly the circuit's task and artifact. The model run itself is not tested here (it needs ~30 GB)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('fair_circuit_baseline', ROOT / 'scripts' / 'fair_circuit_baseline.py')
fair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fair)


def tiny_aya_scores() -> dict[str, float]:
    """36 layers of 16 heads and one MLP block each, like Tiny Aya; MLP blocks score low on purpose."""
    scores = {f'A{layer}.{head}': float(layer * 16 + head + 1) for layer in range(36) for head in range(16)}
    scores.update({f'MLP {layer}': 0.001 for layer in range(36)})
    return scores


def test_the_shuffle_pins_every_mlp_block_above_every_head():
    scores = fair.shuffled_scores(tiny_aya_scores(), seed=0)
    lowest_mlp = min(value for name, value in scores.items() if name.startswith('MLP'))
    assert lowest_mlp > max(value for name, value in scores.items() if name.startswith('A'))


def test_the_shuffle_deals_out_the_same_head_scores_in_a_new_order():
    original = tiny_aya_scores()
    shuffled = fair.shuffled_scores(original, seed=0)
    heads = [name for name in original if name.startswith('A')]
    assert sorted(shuffled[name] for name in heads) == sorted(original[name] for name in heads)
    assert sum(shuffled[name] != original[name] for name in heads) > 500
    assert shuffled == fair.shuffled_scores(original, seed=0)
    assert shuffled != fair.shuffled_scores(original, seed=1)


def test_the_kept_head_count_matches_the_circuits():
    # Every circuit in the full run kept 392 heads, 36 MLP blocks and the input: 429 nodes.
    assert fair.kept_heads(tiny_aya_scores(), sparsity=0.3) == 392


def summary(tmp_path: Path, digest: str) -> dict:
    folder = tmp_path / 'circuits' / 'global-circuit__jv' / f'{digest}-83fccd74'
    folder.mkdir(parents=True)
    stem = f'eap-ig_CohereLabs_tiny-aya-global_refusal_jv__global-circuit__{digest}_83fccd74_node'
    (folder / f'{stem}.pt').write_bytes(b'artifact')
    (folder / f'{stem}_scores.pt').write_bytes(b'scores')
    (folder / f'{stem}_scores.json').write_text('{}', encoding='utf-8')
    return {'lang': 'jv', 'model_tag': 'global-circuit', 'task': f'refusal_jv__global-circuit__{digest}_83fccd74',
            'scores_json': str(folder / f'{stem}_scores.json')}


def test_the_task_file_is_found_by_the_hash_in_the_task_name(tmp_path):
    tasks = tmp_path / 'circuit_tasks'
    tasks.mkdir()
    right = tasks / 'refusal_jv__global-circuit__none.csv'
    right.write_text('harmful,harmless,refuse,comply\n', encoding='utf-8')
    (tasks / 'refusal_jv__global-circuit__default.csv').write_text('other\n', encoding='utf-8')
    digest = hashlib.sha256(right.read_bytes()).hexdigest()[:8]
    assert fair.task_file(tmp_path, summary(tmp_path, digest)) == right
    right.write_text('changed since the circuit ran\n', encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        fair.task_file(tmp_path, summary(tmp_path / 'again', digest))


def test_the_artifact_and_its_scores_sit_beside_the_scores_json(tmp_path):
    artifact, scores = fair.artifact_files(tmp_path, summary(tmp_path, '4ad239e5'))
    assert artifact.name.endswith('_node.pt') and scores.name == artifact.stem + '_scores.pt'
    assert scores.read_bytes() == b'scores'


def test_a_moved_run_still_finds_its_artifact(tmp_path):
    row = summary(tmp_path, '4ad239e5')
    row['scores_json'] = 'runs/somewhere-else/x_scores.json'
    artifact, _ = fair.artifact_files(tmp_path, row)
    assert artifact.read_bytes() == b'artifact'


def test_a_path_whose_folder_exists_but_holds_no_circuit_falls_back(tmp_path, monkeypatch):
    """On Linux a Windows path is one long file name, so its parent is the code directory: it exists, but holds no
    circuit. The first remote run stopped there."""
    row = summary(tmp_path, '4ad239e5')
    code = tmp_path / 'code'
    code.mkdir()
    monkeypatch.setattr(fair, 'ROOT', code)
    row['scores_json'] = 'x_scores.json'  # parent: ROOT itself, an existing folder without a *_node.pt
    artifact, _ = fair.artifact_files(tmp_path, row)
    assert artifact.read_bytes() == b'artifact'


def test_the_device_defaults_to_cpu_and_a_gpu_run_keeps_full_fp32(monkeypatch):
    import torch

    assert fair.use_device('cpu')['device'] == 'cpu' and fair.DEVICE == 'cpu'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        fair.use_device('cuda')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'Fixture GPU')
    asked = []
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', asked.append)
    monkeypatch.setattr(torch, 'get_float32_matmul_precision', lambda: 'highest')
    recorded = fair.use_device('cuda')
    assert asked == ['highest'] and recorded['float32_matmul_precision'] == 'highest'
    assert recorded['tf32'] is False and recorded['gpu'] == 'Fixture GPU' and fair.DEVICE == 'cuda'
    fair.use_device('cpu')


def test_a_gpu_run_cannot_skip_the_check_draw(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['fair_circuit_baseline.py', str(tmp_path), '--device', 'cuda', '--no-check'])
    with pytest.raises(SystemExit):
        fair.main()
