"""A fair random baseline for each refusal circuit: random heads, but every MLP block kept, as the circuit keeps them.

CircuitKIT's own baseline draws the circuit's node count from heads and MLP blocks alike, so a random set usually loses
about a third of the MLP blocks. Every one of our circuits kept all 36 on merit, so that baseline mostly measures what
losing MLP blocks costs. This script asks the sharper question: with all MLP blocks and the input kept, do the circuit's
heads beat the same number of random heads?

It writes a shuffled copy of the circuit's scores (every MLP block pinned to the top, head scores permuted) and runs
CircuitKIT's patching pillar on it, with the same task, model and settings as the circuit. A first check draw uses the
real scores and must reproduce the circuit's logged patching ratio, which shows the harness matches the original run.

Each evaluation takes about 12 minutes on the CPU and the model needs about 30 GB of RAM, so run it when the chain is
idle:

    python scripts/fair_circuit_baseline.py runs/experiments/full/earth --draws 3
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINNED_MLP_SCORE = 1e6
# The circuits were found on the CPU. `--device cuda` runs the same fp32 evaluation on a GPU with TF32 off; the check
# draw must still reproduce each circuit's logged ratio within 1e-3, so a GPU that disagrees writes nothing.
DEVICE = 'cpu'


def use_device(device: str) -> dict:
    """Select the evaluation device and return what the result should record about it."""
    global DEVICE
    import torch

    DEVICE = device
    if device == 'cpu':
        return {'device': 'cpu', 'torch': torch.__version__}
    if not torch.cuda.is_available():
        raise RuntimeError('--device cuda needs a CUDA GPU')
    torch.set_float32_matmul_precision('highest')  # no TF32 in float32 matrix multiplies
    for backend in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(backend, 'fp32_precision'):  # torch >= 2.9 names it this way
            backend.fp32_precision = 'ieee'
    return {'device': 'cuda', 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'tf32': False, 'float32_matmul_precision': torch.get_float32_matmul_precision()}


def is_head(name: str) -> bool:
    return name.startswith('A')


def is_mlp(name: str) -> bool:
    return name.startswith('MLP')


def shuffled_scores(node_scores: dict[str, float], seed: int) -> dict[str, float]:
    """Every MLP block pinned above any head; the head scores dealt out to the heads in random order."""
    heads = sorted(name for name in node_scores if is_head(name))
    values = np.array([node_scores[name] for name in heads], dtype=float)
    np.random.RandomState(seed).shuffle(values)
    result = {name: float(value) for name, value in zip(heads, values)}
    for name, value in node_scores.items():
        if not is_head(name):
            result[name] = PINNED_MLP_SCORE if is_mlp(name) else value
    return result


def kept_heads(node_scores: dict[str, float], sparsity: float) -> int:
    """Heads in a circuit that keeps every MLP block, at CircuitKIT's node count for scope "both"."""
    heads = sum(is_head(name) for name in node_scores)
    mlps = sum(is_mlp(name) for name in node_scores)
    return int((heads + mlps) * (1 - sparsity)) - mlps


def task_file(run: Path, summary: dict) -> Path:
    """The task CSV whose hash is in the circuit's task name, so the evaluation reads exactly the circuit's pairs."""
    lang, tag = summary['lang'], summary['model_tag']
    digest = summary['task'].rsplit('__', 1)[1].split('_')[0]
    for path in sorted((run / 'circuit_tasks').glob(f'refusal_{lang}__{tag}__*.csv')):
        if hashlib.sha256(path.read_bytes()).hexdigest()[:8] == digest:
            return path
    raise FileNotFoundError(f'no task file for {summary["task"]} matches hash {digest}')


def artifact_files(run: Path, summary: dict) -> tuple[Path, Path]:
    """The circuit's pruning artifact and its scores file."""
    scores_json = Path(summary['scores_json'])
    scores_json = scores_json if scores_json.is_absolute() else ROOT / scores_json
    folder = scores_json.parent
    # The run moved since the summary was written, or another OS wrote the path: a Windows path read on Linux is one
    # long file name, so its "folder" is the code directory, which exists but holds no circuit.
    if not folder.is_dir() or not any(folder.glob('*_node.pt')):
        folder = next((run / 'circuits' / f"{summary['model_tag']}__{summary['lang']}").glob('*-*/'))
    artifact = next(folder.glob('*_node.pt'))
    return artifact, artifact.with_name(artifact.stem + '_scores.pt')


def patching_ratio(summary: dict, task_csv: Path, artifact: Path, scores: dict, workdir: Path, pipe=None):
    """CircuitKIT's unclipped patching ratio for a circuit rebuilt from `scores`, set up as discovery set it up."""
    import circuitkit  # noqa: F401  (must be imported before transformer_lens)
    import torch
    from circuitkit.api import prepare_custom_task
    from circuitkit.circuit import Circuit
    from circuitkit.pipeline import Pipeline

    settings = summary['settings']
    workdir.mkdir(parents=True, exist_ok=True)
    fake = workdir / artifact.name
    shutil.copyfile(artifact, fake)
    torch.save({'algo': 'eap-ig', 'level': 'node', 'node_scores': scores}, fake.with_name(fake.stem + '_scores.pt'))
    if pipe is None:
        pipe = Pipeline.from_custom_data(
            summary['resolved_model'], str(task_csv), clean_prompt='{harmful}', clean_answer='{refuse}',
            corrupt_prompt='{harmless}', corrupt_answer='{comply}', task_name=summary['task'],
            precision=settings['precision'], output_dir=str(workdir), device=DEVICE)
        pipe._custom_data_cfg['max_records'] = settings['data_max_records']
    config = {'model': pipe._model_cfg(),
              'discovery': {'algorithm': 'eap-ig', 'task': summary['task'], 'level': 'node', 'batch_size': 1,
                            'data_params': {'num_examples': settings['n_examples'], 'batch_size': 1,
                                            'seed': settings['seed']},
                            'seed': settings['seed'], 'ig_steps': settings['ig_steps'],
                            'model_name': summary['resolved_model']},
              'pruning': {'target_sparsity': settings['sparsity'], 'scope': 'both'},
              'output_path': str(fake)}
    if pipe._custom_data_cfg is not None:
        config['data'] = pipe._custom_data_cfg
        pipe.task = prepare_custom_task(config, pipe._ensure_model(), task_name=summary['task'])
        pipe._custom_data_cfg = None   # registered once; later draws reuse the task
    config['discovery']['task'] = pipe.task
    pipe._discovery_cfg = config
    pipe._artifact_path = str(fake)
    pipe._circuit = Circuit.from_artifact(str(fake))
    pipe.evaluate(pillars=['patching'], n_examples=settings['n_examples'])
    report = pipe.report
    raw = (getattr(report, 'metadata', None) or {}).get('patching_raw_ratio')
    return (report.patching_score if raw is None else raw), pipe


def run_circuit(run: Path, folder: Path, draws: int, check: bool) -> dict:
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    task_csv = task_file(run, summary)
    artifact, scores_pt = artifact_files(run, summary)
    import torch
    node_scores = {name: float(value) for name, value in
                   torch.load(scores_pt, map_location='cpu', weights_only=True)['node_scores'].items()}
    faithfulness = summary['faithfulness']
    circuit_ratio = (faithfulness.get('metadata') or {}).get('patching_raw_ratio', faithfulness.get('patching_score'))
    workdir = folder / 'fair-baseline'
    pipe, reproduced = None, None
    if check:
        reproduced, pipe = patching_ratio(summary, task_csv, artifact, node_scores, workdir / 'check')
        if abs(reproduced - circuit_ratio) > 1e-3:
            raise RuntimeError(f'{folder.name}: the harness gives {reproduced:.4f}, the circuit logged '
                               f'{circuit_ratio:.4f}; the setup differs, so no baseline is written')
    ratios = []
    for seed in range(draws):
        ratio, pipe = patching_ratio(summary, task_csv, artifact, shuffled_scores(node_scores, seed),
                                     workdir / f'seed{seed}', pipe)
        ratios.append(ratio)
        print(f'{folder.name} seed {seed}: {ratio:.3f}', flush=True)
    if reproduced is not None and all(ratio == reproduced for ratio in ratios):
        raise RuntimeError(f'{folder.name}: every shuffle scored exactly like the circuit, so the scores were ignored')
    del pipe
    gc.collect()
    return {'method': 'random heads, every MLP block and the input kept; CircuitKIT patching pillar',
            'kept_heads': kept_heads(node_scores, summary['settings']['sparsity']),
            'kept_mlps': sum(is_mlp(name) for name in node_scores), 'seeds': list(range(draws)),
            'patching_ratios': ratios, 'mean': float(np.mean(ratios)),
            'circuit_patching_ratio': circuit_ratio, 'check_ratio': reproduced,
            'n_examples': summary['settings']['n_examples'], 'circuitkit_commit': summary.get('circuitkit_commit')}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--only', help='one circuit folder, e.g. global-circuit__en')
    parser.add_argument('--no-check', action='store_true', help='skip the draw that reproduces the circuit')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='cuda runs the same fp32 evaluation on a GPU, TF32 off; the check draw still applies')
    args = parser.parse_args()
    if args.device == 'cuda' and args.no_check:
        parser.error('--device cuda needs the check draw: it is what shows the GPU reproduces the CPU circuit')
    hardware = use_device(args.device)
    folders = sorted(path.parent for path in (args.run / 'circuits').glob('*__*/summary.json'))
    if args.only:
        folders = [folder for folder in folders if folder.name == args.only]
    if not folders:
        print('no circuits found', file=sys.stderr)
        return 1
    for folder in folders:
        out = folder / 'fair_baseline.json'
        if out.exists():
            print(f'{folder.name}: done already')
            continue
        result = {**run_circuit(args.run, folder, args.draws, not args.no_check), 'hardware': hardware}
        out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f"{folder.name}: random heads {result['mean']:.3f} against the circuit's "
              f"{result['circuit_patching_ratio']:.3f}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
