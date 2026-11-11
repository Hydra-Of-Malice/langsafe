#!/usr/bin/env python3
"""Offline CUDA/NF4 handoff planner and fail-closed executor.

--dry-run only reads files/package metadata; it never imports torch, loads a model,
contacts a service, creates the destination, or claims that NF4 ran.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
GIB = 1 << 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path: Path):
    if not path.is_file():
        raise ValueError(f'Required artifact is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f'Required prompt/evaluation artifact is missing: {path}')
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def safe_tag(tag: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', tag) or '..' in tag:
        raise ValueError(f'Invalid selected tag: {tag!r}')
    return tag


def checkpoint_name(model: str) -> str:
    """The checkpoint folder's name, whichever OS wrote the path (selection.json keeps the source machine's path)."""
    return re.split(r'[\\/]', str(model).rstrip('\\/'))[-1]


# What must be identical for a run to continue in an existing output directory. Paths to cached models and the
# software versions may differ between sessions; the inputs, the code, the stages and the output location may not.
RESUME_KEYS = ('profile', 'family', 'judge_set', 'source_run', 'output', 'spec_digest', 'spec_sha256', 'source_evidence',
               'model_refs', 'judge_models', 'stages', 'report_tags', 'selected_checkpoint_source',
               'selected_checkpoint_target', 'preamble_conditions')


def resume_identity(plan: dict) -> str:
    fields = {key: plan.get(key) for key in RESUME_KEYS}
    fields['checkpoints'] = {tag: checkpoint_content(identity) for tag, identity in plan.get('checkpoints', {}).items()}
    fields['code'] = plan.get('code', {}).get('content_sha256')
    return digest(fields)


def capability_file(tag: str, preamble: str) -> Path:
    return Path('capability') / (tag + ('.none' if preamble == 'none' else '') + '.json')


def judge_ref(judge: dict) -> str:
    """The GGUF a local judge really runs from, when it is on this machine."""
    return judge['gguf'] if judge.get('gguf') and (ROOT / judge['gguf']).is_file() else judge['model']


def checkpoint(model: str) -> tuple[Path, dict]:
    """Resolve only local directories/Hub cache; hash every checkpoint/tokenizer file."""
    from langsafe.gen.model import resolve_model

    if str(model).endswith('.gguf'):
        path = (ROOT / model).resolve()
        files = {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size}}
        return path, {'resolved_path': str(path), 'snapshot_revision': None, 'files': files,
                      'content_sha256': digest(files), 'bytes': path.stat().st_size}
    path = Path(resolve_model(model)).expanduser()
    if not path.is_dir():
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model, 'config.json')
        if not isinstance(config, str):
            raise ValueError(f'Checkpoint is not cached; provision it before this offline run: {model}')
        path = Path(config).parent
    path = path.resolve()
    if not (path / 'config.json').is_file() or not (path / 'tokenizer_config.json').is_file():
        raise ValueError(f'Incomplete checkpoint config/tokenizer: {path}')
    if not any((path / name).is_file() for name in ('tokenizer.json', 'tokenizer.model', 'spiece.model')):
        raise ValueError(f'Incomplete checkpoint tokenizer: {path}')
    indexes = list(path.glob('*.safetensors.index.json'))
    weights = sorted(path.glob('*.safetensors'))
    if not weights:
        raise ValueError(f'Checkpoint requires safetensors weights: {path}')
    for index in indexes:
        for name in set(read_json(index).get('weight_map', {}).values()):
            if Path(name).name != name or not (path / name).is_file():
                raise ValueError(f'Incomplete checkpoint shard: {path / name}')
    files = {str(file.relative_to(path)): {'sha256': sha256(file), 'bytes': file.stat().st_size}
             for file in sorted(path.rglob('*')) if file.is_file() and '.cache' not in file.relative_to(path).parts}
    return path, {'resolved_path': str(path), 'snapshot_revision': path.name if path.parent.name == 'snapshots' else None,
                  'files': files, 'content_sha256': digest(files),
                  'bytes': sum(file['bytes'] for file in files.values())}


def checkpoint_content(identity: dict) -> dict:
    """Drop machine-specific paths before comparing a transferred snapshot."""
    return {key: value for key, value in identity.items() if key != 'resolved_path'}


def software() -> dict:
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'machine': platform.machine()}
    for package in ('langsafe', 'torch', 'transformers', 'accelerate', 'bitsandbytes', 'datasets',
                    'huggingface_hub', 'auditkit', 'numpy', 'pyyaml'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def code_identity() -> dict:
    paths = [*ROOT.glob('src/langsafe/**/*.py'), ROOT / 'pyproject.toml',
             ROOT / 'scripts/run_nf4_evaluation.sh', Path(__file__).resolve()]
    files = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths) if path.is_file()}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=False)
    return {'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
            'files_sha256': files, 'content_sha256': digest(files)}


def check_source(source: Path, spec: dict, family: str, judge_set: str) -> tuple[dict, list[dict], dict]:
    """Validate source evidence before any accelerator or destination activity."""
    from langsafe.config import spec_digest

    manifest = read_json(source / 'data/manifest.json')
    if manifest.get('spec_digest') != spec_digest(spec):
        raise ValueError('Source data spec digest does not match --spec; do not mix experiment profiles')
    if manifest.get('family') != family or not manifest.get('leakage_check', {}).get('ok'):
        raise ValueError('Source family/leakage evidence is missing or mismatched')
    if not manifest.get('coverage', {}).get('complete'):
        raise ValueError('Source data coverage is incomplete; finish the source experiment data first')
    split_rows = {}
    for split in ('heldout', 'repair'):
        path = source / 'data' / f'{split}.jsonl'
        info = manifest.get('files', {}).get(split, {})
        if not path.is_file() or sha256(path) != info.get('sha256'):
            raise ValueError(f'Source {split} data hash does not match its manifest')
        split_rows[split] = rows(path)
        if len(split_rows[split]) != info.get('rows') or not split_rows[split]:
            raise ValueError(f'Source {split} prompt count does not match manifest')
        if any(row.get('split') != split for row in split_rows[split]):
            raise ValueError(f'Source {split} prompt split is incorrect')
        ids = [row['id'] for row in split_rows[split]]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Source {split} prompt identities are duplicated')
    if {row['group'] for row in split_rows['heldout']} & {row['group'] for row in split_rows['repair']}:
        raise ValueError('Source repair/heldout prompt groups overlap')
    selection = read_json(source / 'selection.json')
    selected = selection.get('selected_for_quantization', {})
    if selection.get('split') != 'repair' or selection.get('worst_language') not in spec['languages'][family]:
        raise ValueError('Selection must be fixed on the repair split with a target language')
    if not selected or selected.get('steering') or selected.get('arm') == 'steer':
        raise ValueError('NF4 requires the selected repair weight checkpoint, not a steering-only selection')
    tag = safe_tag(selected.get('tag', ''))
    if tag in ('global', family) or tag.endswith(('-nf4', '-q4')):
        raise ValueError('Selected repair tag collides with a baseline or precision suffix')
    if selected not in [arm.get('selected') for arm in selection.get('arms', {}).values()]:
        raise ValueError('Selected quantization candidate is absent from repair selection evidence')
    languages = set(spec['languages'][family])
    prompts = [row for row in split_rows['heldout'] if row['lang'] in languages]
    if not prompts:
        raise ValueError('Source held-out prompts are missing for the selected family')
    expected = {(row['id'], row['lang'], row['kind'], pre) for row in prompts for pre in ('default', 'none')}
    for lang in languages:
        for kind in ('harmful', 'benign'):
            if not any(row['lang'] == lang and row['kind'] == kind for row in prompts):
                raise ValueError(f'Source held-out prompts are missing {lang}/{kind}')
    evidence = {'data/manifest.json', 'data/heldout.jsonl', 'data/repair.jsonl', 'selection.json'}
    tags = ['global', family, tag]
    capabilities = {}
    rendered = None
    for source_tag in tags:
        generated = rows(source / 'gens' / f'{source_tag}.jsonl')
        if any((row.get('variant') or {}).get('quant') not in (None, '') for row in generated):
            raise ValueError(f'{source_tag}: source generation evidence is not full precision')
        keys = [(row['id'], row['lang'], row['kind'], row['preamble']) for row in generated]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError(f'{source_tag}: source prompt set differs from complete heldout/both preambles')
        if any(row.get('split') != 'heldout' for row in generated):
            raise ValueError(f'{source_tag}: source prompt split is not heldout')
        rendered_hashes = {(row['id'], row['preamble']): row.get('rendered_prompt_sha256') for row in generated}
        if not all(rendered_hashes.values()) or (rendered is not None and rendered != rendered_hashes):
            raise ValueError(f'{source_tag}: source rendered prompt hashes do not match baselines')
        rendered = rendered_hashes
        judged_name = f'{source_tag}.jsonl' if judge_set == 'router' else f'{source_tag}.local.jsonl'
        judged_path = Path('judged') / judged_name
        judged = rows(source / judged_path)
        if {(row['id'], row['lang'], row['kind'], row['preamble']) for row in judged} != expected or len(judged) != len(expected):
            raise ValueError(f'{source_tag}: source judged prompt set is incomplete')
        summary = judged_path.with_suffix('.summary.json')
        if len(read_json(source / summary).get('judges', [])) != 2:
            raise ValueError(f'{source_tag}: source evidence requires two judges')
        evidence.update((str(Path('gens') / f'{source_tag}.jsonl'), str(judged_path), str(summary)))
        for pre in ('default', 'none'):
            cap_path = capability_file(source_tag, pre)
            cap = read_json(source / cap_path)
            if cap.get('metadata', {}).get('quant') is not None or cap.get('metadata', {}).get('gguf'):
                raise ValueError(f'{source_tag}: source FP16 capability precision is mismatched')
            if not cap.get('complete'):
                raise ValueError(f'{source_tag}/{pre}: source capability is incomplete')
            cells = cap.get('languages', {})
            from langsafe.config import capability_languages
            langs = set(capability_languages(spec, family))
            identity = {lang: {'n': cells.get(lang, {}).get('n'), 'data_sha256': cells.get(lang, {}).get('data_sha256')}
                        for lang in sorted(langs)}
            if any(not cell['n'] or not cell['data_sha256'] or len(cells[lang].get('per_item', [])) != cell['n']
                   for lang, cell in identity.items()):
                raise ValueError(f'{source_tag}/{pre}: source capability item evidence is missing')
            if pre in capabilities and capabilities[pre] != identity:
                raise ValueError(f'{source_tag}/{pre}: source capability inputs differ between checkpoints')
            capabilities[pre] = identity
            evidence.add(str(cap_path))
    # Preserve available selected-arm/Q4 report evidence and real calibration; never copy GGUF weights.
    all_tags = [*tags, *[safe_tag(arm['selected']['tag']) for arm in selection.get('arms', {}).values() if arm.get('selected')]]
    for source_tag in dict.fromkeys(all_tags):
        for variant in (source_tag, source_tag + '-q4'):
            suffix = '' if judge_set == 'router' else '.local'
            names = [Path('gens') / f'{variant}.jsonl', Path('judged') / f'{variant}{suffix}.jsonl',
                     Path('judged') / f'{variant}{suffix}.summary.json',
                     *(capability_file(variant, pre) for pre in ('default', 'none')),
                     Path('calibration') / f'{variant}.{judge_set}.json',
                     Path('calibration') / f'{variant}.router-comparison.json']
            evidence.update(str(path) for path in names if (source / path).is_file())
    identities = {name: {'sha256': sha256(source / name), 'bytes': (source / name).stat().st_size}
                  for name in sorted(evidence)}
    return selection, prompts, {'files': identities, 'tags': list(dict.fromkeys(all_tags)),
                                'required_tags': tags,
                                'capability': capabilities, 'rendered_prompts_sha256': digest(sorted(rendered.items()))}


def build_plan(args) -> dict:
    from langsafe.config import load_spec, spec_digest

    source, output, spec_path = args.source_run.resolve(), args.output.resolve(), args.spec.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output directories overlap; use a separate CUDA artifact directory')
    resuming = getattr(args, 'resume', False) and (output / 'nf4-environment.json').is_file()
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not resuming:
        raise ValueError('NF4 output must be a new or empty directory; source artifacts are never overwritten')
    spec = load_spec(spec_path)
    if set(spec.get('preamble_conditions', [])) != {'default', 'none'} or len(spec['preamble_conditions']) != 2:
        raise ValueError('NF4 comparison requires both default and none preamble conditions')
    profile = spec.get('profile', 'full')
    if profile != 'full' and not args.allow_development:
        raise ValueError('Development inputs require --allow-development and remain development-only evidence')
    if spec.get('decoding', {}).get('dtype') != 'float16' or spec['decoding'].get('do_sample') is not False:
        raise ValueError('This NF4 workflow requires the experiment float16 compute dtype and deterministic decoding')
    if profile == 'full' and spec.get('experiment', {}).get('capability_limit'):
        raise ValueError('Full experiment cannot use a reduced capability_limit')
    judges = spec.get('judging', {}).get('local_judges' if args.judges == 'local' else 'judges', [])
    if len(judges) != 2 or len({(judge.get('base_url'), judge.get('model')) for judge in judges}) != 2:
        raise ValueError('NF4 evaluation requires exactly two independent configured judges')
    if args.judges == 'local' and any(judge.get('base_url') != 'local' or judge.get('quant', 'auto') not in ('auto', 'nf4')
                                      or judge.get('dtype', 'float16') != 'float16' for judge in judges):
        raise ValueError('Local CUDA judge precision must be automatic NF4 or explicit NF4 with float16 compute')
    selection, prompts, source_identity = check_source(source, spec, args.family, args.judges)
    selected = selection['selected_for_quantization']
    selected_path = Path(selected['model']).expanduser()
    # Relocate a copied source-run without rewriting its repair-only selection evidence.
    if not selected_path.is_dir():
        selected_path = source / 'checkpoints' / checkpoint_name(selected['model'])
    if not selected_path.is_dir():
        raise ValueError(f'Selected repair checkpoint missing; copy source checkpoints/{selected_path.name}')
    # Keep baseline references portable in the CLI plan. On the CUDA host the
    # same Hub IDs resolve through its offline cache/LANGSAFE_MODELS; only the
    # selected repair checkpoint is copied into this isolated output.
    model_refs = {'global': spec['models']['reference'], args.family: spec['models']['drifted'][args.family],
                  selected['tag']: str(selected_path)}
    models = dict(model_refs)
    resolved, identities = {}, {}
    for tag, model in models.items():
        path, identity = checkpoint(model)
        resolved[tag] = str(path); identities[tag] = identity
    # Store a portable source-run path rather than the original machine's
    # absolute checkpoint path; the source bundle is transferred as a unit.
    selected_checkpoint_source = str((source / 'checkpoints' / selected_path.name).resolve())
    resolved[selected['tag']] = selected_checkpoint_source
    model_refs[selected['tag']] = selected_checkpoint_source
    judge_models = {judge['name']: judge_ref(judge) for judge in judges} if args.judges == 'local' else {}
    source_identity['selected_checkpoint'] = {
        'tag': selected['tag'], 'source': selected_checkpoint_source,
        'files': identities[selected['tag']]['files'],
        'bytes': identities[selected['tag']]['bytes'],
    }
    if args.judges == 'local':
        for judge in judges:
            _, identity = checkpoint(judge_ref(judge))
            identities['judge:' + judge['name']] = identity
    stages = []
    for tag in models:
        nf4_tag = tag + '-nf4'
        stages.append({'name': nf4_tag + '-heldout', 'args': ['generate', '--family', args.family,
                       '--split', 'heldout', '--tag', nf4_tag, '--model', model_refs[tag], '--quant', 'nf4']})
        for pre in spec['preamble_conditions']:
            command = ['capability', '--family', args.family, '--tag', nf4_tag, '--model', model_refs[tag],
                       '--quant', 'nf4', '--preamble', pre]
            if spec.get('experiment', {}).get('capability_limit'):
                command += ['--limit', str(spec['experiment']['capability_limit'])]
            stages.append({'name': nf4_tag + '-capability-' + pre, 'args': command})
        for command in ('judge', 'audit'):
            stages.append({'name': nf4_tag + '-' + command, 'args': [command, '--tag', nf4_tag, '--judges', args.judges]})
    report_tags = list(source_identity['required_tags'])
    for tag in list(report_tags):
        if str(Path('judged') / (tag + '-q4' + ('' if args.judges == 'router' else '.local') + '.jsonl')) in source_identity['files']:
            report_tags.append(tag + '-q4')
    report_tags += [tag + '-nf4' for tag in models]
    stages.append({'name': 'report', 'args': ['report', '--tags', ','.join(report_tags), '--family', args.family, '--judges', args.judges]})
    stages.append({'name': 'ledger-verify', 'args': ['ledger-verify']})
    return {'schema_version': 1, 'status': 'planned_not_executed', 'precision': 'nf4', 'compute_dtype': 'float16',
            'profile': profile, 'family': args.family, 'judge_set': args.judges, 'hardware_checked': False,
            'source_run': str(source), 'output': str(output), 'spec': str(spec_path), 'spec_digest': spec_digest(spec),
            'spec_sha256': sha256(spec_path), 'source_evidence': source_identity, 'model_refs': model_refs,
            'resolved_models': resolved,
            'selected_checkpoint_source': selected_checkpoint_source,
            'selected_checkpoint_target': str(output / 'checkpoints' / selected['tag']),
            'checkpoints': identities, 'software': software(), 'code': code_identity(),
            'judge_models': judge_models,
            'heldout_prompts': len(prompts), 'heldout_evaluations_per_model': len(prompts) * 2,
            'preamble_conditions': spec['preamble_conditions'], 'report_tags': report_tags, 'stages': stages,
            'network': 'offline model/data; router requests only with explicit --judges router --allow-router',
            'copy_bytes': (sum(file['bytes'] for file in source_identity['files'].values())
                           + identities[selected['tag']]['bytes'] + spec_path.stat().st_size)}


def check_runtime(plan: dict, min_free_gib: float, min_vram_gib: float = 12) -> dict:
    """Called only by explicit remote --check-hardware/--execute, never --dry-run."""
    if platform.system() not in {'Linux', 'Windows'}:
        raise ValueError('Genuine CUDA hardware is required; NF4 is not executed on a Mac')
    remote_software = software()
    remote_code = code_identity()
    if remote_code.get('content_sha256') != plan.get('code', {}).get('content_sha256'):
        raise ValueError('Remote code hash differs from the offline plan; transfer the exact working tree')
    missing = [name for name, version in remote_software.items()
               if name in {'torch', 'transformers', 'accelerate', 'bitsandbytes', 'datasets', 'auditkit', 'huggingface_hub'} and not version]
    if missing:
        raise ValueError('Missing runtime packages: ' + ', '.join(missing))
    for tag, model in plan['model_refs'].items():
        path, identity = checkpoint(model)
        expected = plan['checkpoints'][tag]
        if checkpoint_content(identity) != checkpoint_content(expected):
            raise ValueError(f'Checkpoint content differs from the offline plan for {tag}: {path}')
    for name, model in plan.get('judge_models', {}).items():
        path, identity = checkpoint(model)
        expected = plan['checkpoints']['judge:' + name]
        if checkpoint_content(identity) != checkpoint_content(expected):
            raise ValueError(f'Judge checkpoint content differs from the offline plan for {name}: {path}')
    import torch
    import bitsandbytes as bnb
    from bitsandbytes.cextension import lib

    if not torch.version.cuda or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Expose exactly one genuine CUDA device (CUDA_VISIBLE_DEVICES=0); no CPU/MPS fallback')
    if not getattr(lib, 'compiled_with_cuda', False):
        raise ValueError('bitsandbytes CUDA native library is unavailable')
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if props.major < 7 or total < min_vram_gib * GIB or free < min(8, min_vram_gib * .75) * GIB:
        raise ValueError(f'CUDA preflight requires compute capability >=7.0, >={min_vram_gib} GiB VRAM and enough free; '
                         'prefer 24 GiB')
    # Exercise actual NF4 quantization/dequantization and a CUDA kernel without loading a checkpoint.
    test = torch.arange(128, device='cuda', dtype=torch.float16).reshape(2, 64) / 128
    packed, state = bnb.functional.quantize_4bit(test, blocksize=64, quant_type='nf4')
    restored = bnb.functional.dequantize_4bit(packed, state)
    if restored.device.type != 'cuda' or not torch.isfinite(restored).all().item():
        raise ValueError('bitsandbytes NF4 CUDA smoke check failed')
    torch.cuda.synchronize()
    del test, packed, state, restored
    torch.cuda.empty_cache()
    parent = Path(plan['output'])
    while not parent.exists():
        parent = parent.parent
    available = shutil.disk_usage(parent).free
    required = plan['copy_bytes'] + max(min_free_gib, 2) * GIB
    if available < required:
        raise ValueError(f'Insufficient output storage: {available / GIB:.1f} GiB free; {required / GIB:.1f} GiB required')
    from langsafe.eval.capability import load_items
    from langsafe.config import load_spec
    spec = load_spec(plan['spec'])
    capabilities = {}
    for lang, expected in plan['source_evidence']['capability']['default'].items():
        items = load_items(lang, spec.get('experiment', {}).get('capability_limit'))
        actual = {'n': len(items), 'data_sha256': digest(items)}
        if actual != expected or actual != plan['source_evidence']['capability']['none'][lang]:
            raise ValueError(f'Cached capability inputs differ from source experiment: {lang}; copy the same HF dataset cache')
        capabilities[lang] = actual
    import auditkit
    if not all(hasattr(auditkit, name) for name in ('Sample', 'RunConfig', 'evaluate', 'scorer')):
        raise ValueError('AuditKIT runtime lacks the APIs required by the existing audit bridge')
    query = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'],
                           capture_output=True, text=True, check=False) if shutil.which('nvidia-smi') else None
    return {'device': props.name, 'compute_capability': [props.major, props.minor], 'vram_bytes': total,
            'free_vram_bytes': free, 'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'nvidia_smi': query.stdout.strip() if query and query.returncode == 0 else None,
            'bitsandbytes_nf4_kernel': 'passed', 'capability_inputs': capabilities,
            'output_free_bytes': available, 'output_required_bytes': required,
            'software': remote_software, 'code': remote_code}


def validate_nf4_outputs(plan: dict) -> None:
    """Verify actual output identity/coverage before software_run_complete may be written."""
    output = Path(plan['output'])
    source = Path(plan['source_run'])
    selected = read_json(source / 'selection.json')['selected_for_quantization']['tag']
    expected_keys = None
    for tag in ('global', plan['family'], selected):
        baseline = rows(source / 'gens' / f'{tag}.jsonl')
        expected = {(row['id'], row['lang'], row['kind'], row['preamble']): row['rendered_prompt_sha256'] for row in baseline}
        generated = rows(output / 'gens' / f'{tag}-nf4.jsonl')
        actual = {(row['id'], row['lang'], row['kind'], row['preamble']): row.get('rendered_prompt_sha256') for row in generated}
        if actual != expected or len(generated) != len(expected):
            raise ValueError(f'{tag}-nf4: prompt set/rendering differs from source baseline')
        if any(row.get('model_tag') != tag + '-nf4' or row.get('split') != 'heldout'
               or (row.get('variant') or {}).get('quant') != 'nf4' for row in generated):
            raise ValueError(f'{tag}-nf4: generation identity differs from NF4 heldout run')
        for pre in plan['preamble_conditions']:
            cap = read_json(output / capability_file(tag + '-nf4', pre))
            if not cap.get('complete') or cap.get('metadata', {}).get('quant') != 'nf4':
                raise ValueError(f'{tag}-nf4/{pre}: capability precision/completion missing')
            for lang, expected_cell in plan['source_evidence']['capability'][pre].items():
                cell = cap.get('languages', {}).get(lang, {})
                if any(cell.get(key) != value for key, value in expected_cell.items()):
                    raise ValueError(f'{tag}-nf4/{pre}: capability inputs changed for {lang}')
        judged_name = f'{tag}-nf4' + ('' if plan['judge_set'] == 'router' else '.local')
        judged = rows(output / 'judged' / (judged_name + '.jsonl'))
        keys = [(row['id'], row['lang'], row['kind'], row['preamble']) for row in judged]
        if set(keys) != set(expected) or len(keys) != len(expected):
            raise ValueError(f'{tag}-nf4: incomplete judged prompt coverage')
        if len(read_json(output / 'judged' / (judged_name + '.summary.json')).get('judges', [])) != 2:
            raise ValueError(f'{tag}-nf4: two-judge evidence missing')
        variant = f'{tag}-nf4.{plan["judge_set"]}'
        read_json(output / 'auditkit' / variant / (variant + '.json'))
    report = read_json(output / 'report.json')
    if not set(plan['report_tags']) <= set(report.get('models', {})):
        raise ValueError('Report omitted requested NF4/source tags')
    comparisons = report.get('thresholds', {})
    for pre in plan['preamble_conditions']:
        if f'{selected}-nf4/{pre}' not in comparisons:
            raise ValueError('Report omitted same-precision NF4 repair comparison')


def execute(plan: dict, hardware: dict, resume: bool = False) -> None:
    from langsafe.experiment import atomic_json

    output, source = Path(plan['output']), Path(plan['source_run'])
    previous = None
    now = datetime.now(timezone.utc).isoformat()
    if resume and (output / 'nf4-environment.json').is_file():
        previous = read_json(output / 'nf4-environment.json')
        if previous.get('status') == 'complete':
            raise ValueError('NF4 output is already complete; there is nothing to resume')
        if previous.get('resume_identity') != resume_identity(plan):
            raise ValueError('NF4 output belongs to a different plan; refusing to resume into it')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('NF4 output became occupied after preflight; refusing to overwrite it')
    output.mkdir(parents=True, exist_ok=True)
    completed = list(previous.get('completed_stages', [])) if previous else []
    metadata = {**plan, 'status': 'running', 'hardware_checked': True, 'hardware': hardware,
                'software': hardware.get('software', plan.get('software')),
                'remote_code': hardware.get('code'),
                'started_at': previous.get('started_at', now) if previous else now,
                'resumed_at': [*previous.get('resumed_at', []), now] if previous else [],
                'earlier_hardware': [*previous.get('earlier_hardware', []), previous.get('hardware')] if previous else [],
                'resume_identity': resume_identity(plan), 'completed_stages': completed,
                'software_run_complete': False, 'nf4_cuda': 'not_complete'}
    atomic_json(output / 'nf4-environment.json', metadata)
    try:
        selected_source = Path(plan['selected_checkpoint_source'])
        selected_target = Path(plan['selected_checkpoint_target'])
        if not selected_source.is_dir():
            raise ValueError(f'Selected repair checkpoint disappeared before execution: {selected_source}')
        selected_identity = plan['source_evidence']['selected_checkpoint']['files']
        for relative, identity in selected_identity.items():
            path = selected_source / relative
            if not path.is_file() or sha256(path) != identity['sha256']:
                raise ValueError(f'Selected repair checkpoint changed since preflight: {relative}')
        selected_target.parent.mkdir(parents=True, exist_ok=True)
        if previous and selected_target.is_dir():
            # A resumed run keeps its copy only if every file still matches the source checkpoint.
            if any(not (selected_target / relative).is_file() or sha256(selected_target / relative) != identity['sha256']
                   for relative, identity in selected_identity.items()):
                shutil.rmtree(selected_target)
        if not selected_target.is_dir():
            shutil.copytree(selected_source, selected_target, dirs_exist_ok=False)
        for name, identity in plan['source_evidence']['files'].items():
            if sha256(source / name) != identity['sha256']:
                raise ValueError(f'Source changed since preflight: {name}')
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, destination)
        shutil.copy2(plan['spec'], output / 'source-spec.yaml')
        env = dict(os.environ, LANGSAFE_DEVICE='cuda', LANGSAFE_RUNS_DIR=str(output),
                   LANGSAFE_DATA_DIR=str(output / 'data'), LANGSAFE_SPEC=str(output / 'source-spec.yaml'),
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        for stage in plan['stages']:
            if stage['name'] in completed:
                print(f'[NF4] {stage["name"]} (already complete)', flush=True)
                continue
            print(f'[NF4] {stage["name"]}', flush=True)
            atomic_json(output / 'current-stage.json', {'stage': stage['name'], 'status': 'running'})
            log = output / 'logs' / (stage['name'] + '.log')
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable, '-u', '-m', 'langsafe.cli', '--spec', str(output / 'source-spec.yaml'),
                       '--runs-dir', str(output), *stage['args']]
            command = [str(selected_target) if value == plan['selected_checkpoint_source'] else value
                       for value in command]
            # Every stage continues from its own caches, so an interrupted one simply runs again.
            with log.open('a' if previous else 'w', encoding='utf-8') as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            atomic_json(output / 'current-stage.json', {'stage': stage['name'], 'status': 'complete'})
            completed.append(stage['name'])
            metadata['completed_stages'] = completed
            atomic_json(output / 'nf4-environment.json', metadata)
        validate_nf4_outputs(plan)
        metadata.update(status='complete', software_run_complete=True, nf4_cuda='executed',
                        finished_at=datetime.now(timezone.utc).isoformat())
        metadata['artifacts_sha256'] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob('*'))
                                       if path.is_file() and path.name not in ('nf4-environment.json', 'nf4-status.json')}
        report = read_json(output / 'report.json')
        atomic_json(output / 'nf4-status.json', {'software_run_complete': True, 'nf4_cuda': 'executed',
                    'profile': plan['profile'], 'family': plan['family'], 'precision': 'nf4',
                    'empirical_repair_success': report['empirical_repair_success'],
                    'validation_gaps': report['validation_gaps'], 'report': str(output / 'report.md')})
    except BaseException as error:
        metadata.update(status='failed', failure_type=type(error).__name__, nf4_cuda='not_complete')
        atomic_json(output / 'current-stage.json', {'stage': locals().get('stage', {}).get('name'), 'status': 'failed'})
        raise
    finally:
        atomic_json(output / 'nf4-environment.json', metadata)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True, help='Completed source experiment/evidence bundle')
    parser.add_argument('--spec', type=Path, required=True, help='Exact same source spec')
    parser.add_argument('--output', type=Path, required=True, help='New isolated CUDA/NF4 artifact directory')
    parser.add_argument('--family', choices=('fire', 'earth'), default='fire')
    parser.add_argument('--judges', choices=('local', 'router'), default='local')
    parser.add_argument('--allow-router', action='store_true', help='Explicitly authorize configured billed router requests during execution')
    parser.add_argument('--allow-development', action='store_true', help='Permit clearly marked development-only evidence')
    parser.add_argument('--min-free-gib', type=float, default=10, help='Output headroom in addition to copied evidence (default 10 GiB)')
    parser.add_argument('--min-vram-gib', type=float, default=12, help='VRAM floor (default 12; a 2 GB NF4 model fits on 8 GB cards)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true', help='Hash inputs/print plan only; never load torch or touch GPU/network')
    mode.add_argument('--check-hardware', action='store_true', help='On CUDA host: check NF4 kernel, storage, offline capability cache; no checkpoint loads')
    mode.add_argument('--execute', action='store_true', help='On authorized CUDA host: preflight then run all stages serially')
    parser.add_argument('--resume', action='store_true',
                        help='With --execute: continue an interrupted run of the same plan in --output, skipping finished stages')
    args = parser.parse_args(argv)
    # Model and dataset code cannot silently download replacements during the handoff.
    os.environ.update(HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', LANGSAFE_DEVICE='cuda')
    try:
        if args.min_free_gib < 0:
            raise ValueError('--min-free-gib must be nonnegative')
        if args.resume and not args.execute:
            raise ValueError('--resume only applies to --execute')
        if args.execute and args.judges == 'router' and not args.allow_router:
            raise ValueError('Router execution requires explicit --allow-router authorization and HF_TOKEN')
        if args.execute and args.judges == 'router' and not os.environ.get('HF_TOKEN'):
            raise ValueError('Router execution requires HF_TOKEN; its value is never logged')
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        hardware = check_runtime(plan, args.min_free_gib, args.min_vram_gib)
        if args.check_hardware:
            print(json.dumps({**plan, 'status': 'preflight_passed_not_executed', 'hardware_checked': True, 'hardware': hardware}, indent=2))
        else:
            execute(plan, hardware, resume=args.resume)
        return 0
    except (ValueError, OSError, ImportError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        # Never log exception payloads from remote APIs or credentials.
        if isinstance(error, subprocess.CalledProcessError):
            print('NF4 blocked: stage failed; inspect isolated output logs and current-stage.json', file=sys.stderr)
        else:
            print(f'NF4 blocked: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
