#!/usr/bin/env python3
"""Run the NF4 check on a rented Linux GPU (Hugging Face Jobs or Kaggle) instead of the laptop.

On the laptop, with the code committed:

    python scripts/remote_nf4.py bundle                 # private bundle -> HF dataset repo
    python scripts/remote_nf4.py launch-hf --flavor a10g-small --timeout 11h
    python scripts/remote_nf4.py fetch --platform hf-a10g

On the GPU host, scripts/remote_nf4.sh calls `prepare`, `verify`, `restore` and `sync`. See docs/REMOTE_NF4.md.

The bundle holds the held-out prompts, harmful ones included, and the models' answers to them, so it lives only in a
private repo: every command that touches the repo refuses a public one. The remote run must match the laptop's: the
same committed code, spec, source evidence, checkpoints (by SHA-256), judge GGUF files and llama.cpp commit. `verify`
compares the remote plan with the laptop's and stops on any difference.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = 'HydraOfMalice/lexsi-nf4-private'
SOURCE = ROOT / 'runs' / 'experiments' / 'full' / 'earth'
STAGING = ROOT / 'runs' / 'experiments' / 'remote-nf4-bundle'
FETCHED = ROOT / 'runs' / 'experiments' / 'full'
# Where the laptop's two judge GGUF files came from (scripts/fetch_models.py); `bundle` checks the Hub copy's SHA-256.
GGUF_SOURCES = {'aya-expanse-8b-Q4_K_M.gguf': 'bartowski/aya-expanse-8b-GGUF', 'Qwen3-8B-Q4_K_M.gguf': 'Qwen/Qwen3-8B-GGUF'}
KITS = ('auditkit', 'curatorkit', 'safetune')
# Not pinned on Linux: torch comes from its CUDA index, the kits from git, and these are local or Windows-only.
UNPINNED = {'torch', 'triton', 'langsafe', 'circuitkit', 'pywin32', 'pywin32-ctypes', *KITS}
EVIDENCE_PATTERNS = ['source/data/**', 'source/gens/**', 'source/judged/**', 'source/capability/**',
                     'source/calibration/**', 'source/selection.json', 'code.tar', 'manifest.json', 'laptop-plan.json',
                     'requirements/**', 'gguf/**', 'kaggle_start.py']


def nf4_module():
    loader = importlib.util.spec_from_file_location('check_nf4_environment', ROOT / 'scripts' / 'check_nf4_environment.py')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def norm(path) -> str:
    return str(path).replace('\\', '/')


def normalised(files: dict) -> dict:
    return {norm(name): value for name, value in files.items()}


def log(message: str) -> None:
    print(f'[remote-nf4 {datetime.now(timezone.utc):%H:%M:%S}] {message}', flush=True)


def api():
    from huggingface_hub import HfApi
    return HfApi()


def private_repo(repo: str) -> None:
    info = api().repo_info(repo, repo_type='dataset')
    if not info.private:
        raise SystemExit(f'{repo} is public; the NF4 bundle may only live in a private repo')


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dry_run_plan(source: Path, spec: Path, output: Path, env: dict | None = None, code: Path = ROOT) -> dict:
    result = subprocess.run([sys.executable, str(code / 'scripts' / 'check_nf4_environment.py'), '--source-run', str(source),
                             '--spec', str(spec), '--output', str(output), '--family', 'earth', '--judges', 'local',
                             '--dry-run'], cwd=code, capture_output=True, text=True, env=env)
    if result.returncode:
        raise SystemExit(f'NF4 dry run failed: {result.stderr.strip()[-2000:]}')
    return json.loads(result.stdout)


def pins() -> tuple[str, list[str], dict]:
    torch = importlib.metadata.version('torch')
    lines, kits = [], {}
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: (d.metadata['Name'] or '').lower()):
        name = (dist.metadata['Name'] or '').lower()
        if name in KITS:
            url = json.loads(dist.read_text('direct_url.json') or '{}')
            kits[name] = {'url': url.get('url'), 'commit': url.get('vcs_info', {}).get('commit_id')}
        if not name or name in UNPINNED or name.startswith('nvidia-cu') or name.startswith('nvidia-nccl'):
            continue
        lines.append(f'{name}=={dist.version}')
    if set(kits) != set(KITS) or not all(kit['url'] and kit['commit'] for kit in kits.values()):
        raise SystemExit(f'Kit provenance is incomplete: {kits}')
    return torch, lines, kits


def unfinished_q4(names: list[str]) -> set[str]:
    """Q4 evidence the chain is still writing: any file of a `<tag>-q4` variant whose judging summary is not there yet.
    The NF4 run does not need Q4 files, and one that changes mid-bundle would fail the remote hash check."""
    names = [norm(name) for name in names]
    variants = {match.group(1) for name in names if (match := re.search(r'([A-Za-z0-9_.-]+?-q4)(?=[./])', name.split('/')[-1]))}
    finished = {variant for variant in variants if f'judged/{variant}.local.summary.json' in names}
    return {name for name in names for variant in variants - finished
            if re.match(re.escape(variant) + r'[./]', name.split('/')[-1])}


def bundle(args) -> None:
    """Stage the code (git HEAD), source evidence, manifest and pins, check the Hub copies, and upload privately.

    The evidence is copied first and the laptop plan is made from those copies, so the manifest describes exactly the
    files uploaded even while the chain keeps writing its run directory."""
    nf4 = nf4_module()
    dirty = git('status', '--porcelain', '--', 'src', 'scripts', 'configs', 'pyproject.toml')
    if dirty:
        raise SystemExit(f'Commit the code first; the bundle ships git HEAD and these differ:\n{dirty}')
    private_repo(args.repo)
    # A run resumes only into the bundle it started from, so the bundle stays frozen while any run has results.
    started = sorted({name.split('/')[1] for name in api().list_repo_files(args.repo, repo_type='dataset')
                      if name.startswith('results/') and name.endswith('/nf4-environment.json')})
    if started and not args.replace_with_runs:
        raise SystemExit(f'Runs {started} started from the current bundle; replacing it would stop them resuming. '
                         'Pass --replace-with-runs only once they are finished or abandoned.')
    commit = git('rev-parse', 'HEAD')
    shutil.rmtree(STAGING, ignore_errors=True)
    (STAGING / 'requirements').mkdir(parents=True)
    log('listing the source evidence')
    live = dry_run_plan(SOURCE, ROOT / 'configs' / 'spec.yaml', STAGING / 'unused-output')
    names = list(normalised(live['source_evidence']['files']))
    skipped = unfinished_q4(names)
    if skipped:
        log(f'leaving out {len(skipped)} Q4 files still being written: {sorted(skipped)}')
    for name in names:
        if name in skipped:
            continue
        target = STAGING / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    log('laptop dry run on the copies (hashes them, both base models, the repair checkpoint and both judges)')
    plan = dry_run_plan(STAGING / 'source', ROOT / 'configs' / 'spec.yaml', STAGING / 'unused-output')
    if plan['code']['git_commit'] != commit:
        raise SystemExit('Dry-run code identity does not come from HEAD')
    if set(normalised(plan['source_evidence']['files'])) != set(names) - skipped:
        raise SystemExit('The plan made from the copies lists different evidence from what was copied')
    (STAGING / 'laptop-plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    subprocess.run(['git', 'archive', '--format=tar', '-o', str(STAGING / 'code.tar'), 'HEAD'], cwd=ROOT, check=True)
    shutil.copy2(ROOT / 'scripts' / 'kaggle_start.py', STAGING / 'kaggle_start.py')
    torch, lines, kits = pins()
    local = torch.split('+')[1] if '+' in torch else None
    if not local or not local.startswith('cu'):
        raise SystemExit(f'Laptop torch {torch} is not a CUDA build; cannot choose the matching Linux wheel')
    (STAGING / 'requirements' / 'torch.txt').write_text(f'torch=={torch}\n')
    (STAGING / 'requirements' / 'constraints.txt').write_text('\n'.join(lines) + '\n')
    selected = plan['source_evidence']['selected_checkpoint']
    folder = nf4.checkpoint_name(selected['source'])
    hub = api()
    judges = {}
    for name, ref in plan['judge_models'].items():
        filename = nf4.checkpoint_name(ref)
        identity = plan['checkpoints']['judge:' + name]['files'][filename]
        upstream = list(hub.get_paths_info(GGUF_SOURCES[filename], [filename], expand=True))
        lfs = getattr(upstream[0], 'lfs', None) if upstream else None
        same = bool(lfs) and lfs.sha256 == identity['sha256']
        if not same:
            log(f'{filename}: the Hub copy differs from the laptop file, so it goes in the bundle')
            (STAGING / 'gguf').mkdir(exist_ok=True)
            shutil.copy2(ROOT / ref, STAGING / 'gguf' / filename)
        judges[name] = {'filename': filename, 'repo': GGUF_SOURCES[filename], 'source': 'hub' if same else 'bundle',
                        **identity}
    base = {}
    for tag in ('global', 'earth'):
        identity = plan['checkpoints'][tag]
        if not identity.get('snapshot_revision'):
            raise SystemExit(f'{tag}: the laptop copy is not a Hub snapshot, so its revision is unknown')
        base[tag] = {'repo': plan['model_refs'][tag], 'revision': identity['snapshot_revision'],
                     'files': normalised(identity['files'])}
    paths = [f'source/checkpoints/{folder}/{name}' for name in normalised(selected['files'])]
    present = {item.path: item for item in hub.get_paths_info(args.repo, paths, repo_type='dataset', expand=True)}
    for name, identity in normalised(selected['files']).items():
        item = present.get(f'source/checkpoints/{folder}/{name}')
        lfs = getattr(item, 'lfs', None) if item else None
        if item is None or (lfs and lfs.sha256 != identity['sha256']) or item.size != identity['bytes']:
            raise SystemExit(f'Repair checkpoint file missing or different in {args.repo}: {name}; upload it first')
    manifest = {
        'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
        'spec_digest': plan['spec_digest'], 'spec_sha256': plan['spec_sha256'],
        'code_files_sha256': normalised(plan['code']['files_sha256']),
        'evidence': normalised(plan['source_evidence']['files']),
        'selected': {'tag': selected['tag'], 'folder': folder, 'files': normalised(selected['files'])},
        'base_models': base, 'judges': judges, 'capability': plan['source_evidence']['capability'],
        'report_tags': plan['report_tags'], 'stages': [stage['name'] for stage in plan['stages']],
        'llama_cpp': {'url': git('-C', 'cache/llama.cpp', 'remote', 'get-url', 'origin'),
                      'commit': git('-C', 'cache/llama.cpp', 'rev-parse', 'HEAD')},
        'kits': kits, 'torch': {'version': torch, 'index': f'https://download.pytorch.org/whl/{local}'},
        'python': '.'.join(sys.version.split()[0].split('.')[:2]),
    }
    (STAGING / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if args.no_upload:
        log(f'staged in {STAGING}; not uploaded')
        return
    log(f'uploading the bundle to {args.repo} (private)')
    hub.upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=str(STAGING), path_in_repo='',
                      ignore_patterns=['unused-output/**', 'source/checkpoints/**'], delete_patterns=EVIDENCE_PATTERNS,
                      commit_message=f'NF4 bundle for {commit[:7]}')
    log(f'bundle uploaded: {len(manifest["evidence"])} evidence files, code {commit[:7]}')


def verify_sha(path: Path, identity: dict, what: str) -> None:
    nf4 = nf4_module()
    if not path.is_file() or path.stat().st_size != identity['bytes'] or nf4.sha256(path) != identity['sha256']:
        raise SystemExit(f'{what} differs from the laptop copy: {path}')


def prepare(args) -> None:
    """On the GPU host: fetch the source evidence, base models, judges and capability data, all checked by SHA-256."""
    from huggingface_hub import hf_hub_download, snapshot_download

    work, code = Path(args.work), Path(args.work) / 'code'
    private_repo(args.repo)
    manifest = json.loads((work / 'manifest.json').read_text(encoding='utf-8'))
    log('source evidence and repair checkpoint')
    snapshot_download(args.repo, repo_type='dataset', local_dir=str(work / 'bundle'),
                      allow_patterns=['source/**', 'laptop-plan.json', 'gguf/**'])
    source = work / 'bundle' / 'source'
    for name, identity in manifest['evidence'].items():
        verify_sha(source / name, identity, 'Evidence')
    for name, identity in manifest['selected']['files'].items():
        verify_sha(source / 'checkpoints' / manifest['selected']['folder'] / name, identity, 'Repair checkpoint')
    for tag, model in manifest['base_models'].items():
        target = work / 'models' / model['repo'].split('/')[-1]
        log(f'{tag}: {model["repo"]} at {model["revision"][:10]}')
        snapshot_download(model['repo'], revision=model['revision'], local_dir=str(target), allow_patterns=list(model['files']))
        extra = {norm(path.relative_to(target)) for path in target.rglob('*')
                 if path.is_file() and '.cache' not in path.relative_to(target).parts} - set(model['files'])
        if extra:
            raise SystemExit(f'{tag}: unexpected files beside the laptop set: {sorted(extra)}')
        for name, identity in model['files'].items():
            verify_sha(target / name, identity, f'{tag} checkpoint')
    gguf = code / 'cache' / 'gguf'
    gguf.mkdir(parents=True, exist_ok=True)
    for name, judge in manifest['judges'].items():
        log(f'judge {name}: {judge["filename"]} from {judge["source"]}')
        if judge['source'] == 'hub':
            hf_hub_download(judge['repo'], judge['filename'], local_dir=str(gguf))
        else:
            shutil.copy2(work / 'bundle' / 'gguf' / judge['filename'], gguf / judge['filename'])
        verify_sha(gguf / judge['filename'], judge, f'Judge {name}')
    sys.path.insert(0, str(code / 'src'))
    from langsafe.eval.capability import load_items
    nf4 = nf4_module()
    for lang, expected in manifest['capability']['default'].items():
        items = load_items(lang)
        if {'n': len(items), 'data_sha256': nf4.digest(items)} != expected:
            raise SystemExit(f'Global-MMLU-Lite {lang} differs from the laptop run')
    log('all inputs match the laptop by SHA-256')


def verify(args) -> None:
    """On the GPU host: build the plan from the fetched inputs and compare it with the laptop's, field by field."""
    work = Path(args.work)
    manifest = json.loads((work / 'manifest.json').read_text(encoding='utf-8'))
    code = work / 'code'
    plan = dry_run_plan(work / 'bundle' / 'source', code / 'configs' / 'spec.yaml', work / 'preflight-unused',
                        env=dict(os.environ), code=code)
    problems = []
    def check(label, actual, expected):
        if actual != expected:
            problems.append(label)
    check('spec digest', plan['spec_digest'], manifest['spec_digest'])
    check('spec file', plan['spec_sha256'], manifest['spec_sha256'])
    check('code', normalised(plan['code']['files_sha256']), manifest['code_files_sha256'])
    check('source evidence', normalised(plan['source_evidence']['files']), manifest['evidence'])
    for tag, model in manifest['base_models'].items():
        check(f'{tag} checkpoint', normalised(plan['checkpoints'][tag]['files']), model['files'])
    check('repair checkpoint', normalised(plan['checkpoints'][manifest['selected']['tag']]['files']),
          manifest['selected']['files'])
    for name, judge in manifest['judges'].items():
        files = plan['checkpoints']['judge:' + name]['files']
        check(f'judge {name}', files.get(judge['filename'], {}).get('sha256'), judge['sha256'])
    check('report tags', plan['report_tags'], manifest['report_tags'])
    check('stages', [stage['name'] for stage in plan['stages']], manifest['stages'])
    if problems:
        raise SystemExit('Remote plan differs from the laptop plan: ' + ', '.join(problems))
    log('remote plan matches the laptop plan: code, spec, evidence, checkpoints, judges, stages')


def restore(args) -> None:
    """Print `fresh`, `resume` or `complete`, restoring an earlier session's synced output for a resume."""
    from huggingface_hub import snapshot_download

    private_repo(args.repo)
    output, work = Path(args.output), Path(args.work)
    prefix = f'results/{args.platform}'
    files = [name for name in api().list_repo_files(args.repo, repo_type='dataset') if name.startswith(prefix + '/')]
    if f'{prefix}/nf4-environment.json' not in files:
        print('fresh')
        return
    snapshot_download(args.repo, repo_type='dataset', local_dir=str(work / 'restore'), allow_patterns=[prefix + '/**'])
    shutil.rmtree(output, ignore_errors=True)
    shutil.copytree(work / 'restore' / prefix, output)
    (output / 'job-status.json').unlink(missing_ok=True)
    for path in (output / 'generations').glob('*.jsonl'):
        trim_partial_lines(path)
    status = json.loads((output / 'nf4-environment.json').read_text(encoding='utf-8')).get('status')
    print('complete' if status == 'complete' else 'resume')


def trim_partial_lines(path: Path) -> int:
    """The generation cache is appended a batch at a time, so a sync can catch its last line half-written. Keep the
    complete lines; the generator redoes the rest. Returns how many lines were dropped."""
    kept, dropped = [], 0
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            json.loads(line)
            kept.append(line)
        except json.JSONDecodeError:
            dropped += 1
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(''.join(line + '\n' for line in kept))
    return dropped


def snapshot_databases(output: Path, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    for path in output.rglob('*.sqlite'):
        destination = target / path.relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source, copy = sqlite3.connect(path), sqlite3.connect(destination)
        with copy:
            source.backup(copy)  # consistent even while the judge is writing
        source.close(); copy.close()


def sync_once(args) -> None:
    output, work = Path(args.output), Path(args.work)
    if not output.is_dir():
        return
    hub, prefix = api(), f'results/{args.platform}'
    snapshots = work / 'sync-snapshots'
    snapshot_databases(output, snapshots)
    status = work / 'job-status.json'
    if status.is_file():
        snapshots.mkdir(parents=True, exist_ok=True)
        shutil.copy2(status, snapshots / 'job-status.json')
    hub.upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=str(output), path_in_repo=prefix,
                      ignore_patterns=['checkpoints/**', '*.tmp', '**/*.tmp', '*.sqlite*', '**/*.sqlite*'],
                      commit_message=f'NF4 progress from {args.platform}')
    if snapshots.is_dir() and any(snapshots.rglob('*')):
        hub.upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=str(snapshots), path_in_repo=prefix,
                          commit_message=f'NF4 caches from {args.platform}')


def sync(args) -> None:
    """Upload the output every --interval seconds until the stop file appears, then once more."""
    private_repo(args.repo)
    stop = Path(args.work) / 'sync.stop'
    while True:
        stopping = stop.exists()
        try:
            sync_once(args)
            log('synced')
        except Exception as error:  # a network blip must not end the run; the next pass retries
            log(f'sync failed ({type(error).__name__}); retrying next pass')
        if stopping:
            return
        for _ in range(args.interval):
            if stop.exists():
                break
            time.sleep(1)


BOOTSTRAP = r'''set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! (command -v curl && command -v git && command -v cmake && command -v g++) >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq curl git cmake build-essential ca-certificates >/dev/null
fi
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$WORK" && cd "$WORK"
CODE_TAR=${CODE_TAR:-code.tar}
uv run -q --no-project --python 3.12 --with huggingface_hub==1.33.0 python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['BUNDLE_REPO'], repo_type='dataset', local_dir='.', allow_patterns=[os.environ.get('CODE_TAR', 'code.tar'), 'manifest.json', 'requirements/*', 'fair/manifest.json'])"
rm -rf code && mkdir code && tar -xf "$CODE_TAR" -C code
exec bash "code/scripts/${ENTRY:-remote_nf4.sh}"
'''
FAIR = ROOT / 'runs' / 'experiments' / 'remote-fair-bundle'


def fair_bundle(args) -> None:
    """Stage the fair circuit baseline's inputs (the circuits and their task files), the code at HEAD and a manifest,
    and upload them to fair/ in the private repo. The task files hold harmful prompts, so this too stays private."""
    dirty = git('status', '--porcelain', '--', 'src', 'scripts', 'configs', 'pyproject.toml')
    if dirty:
        raise SystemExit(f'Commit the code first; the job runs git HEAD and these differ:\n{dirty}')
    private_repo(args.repo)
    nf4_manifest = json.loads((STAGING / 'manifest.json').read_text(encoding='utf-8'))
    shutil.rmtree(FAIR, ignore_errors=True)
    files = {}
    for folder in ('circuits', 'circuit_tasks'):
        for path in sorted((SOURCE / folder).rglob('*')):
            relative = norm(path.relative_to(SOURCE))
            if not path.is_file() or 'fair-baseline' in relative.split('/') or path.name == 'fair_baseline.json':
                continue
            target = FAIR / 'run' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files[relative] = {'sha256': nf4_module().sha256(target), 'bytes': target.stat().st_size}
    models = {json.loads((SOURCE / name).read_text(encoding='utf-8'))['resolved_model']
              for name in files if name.endswith('/summary.json')}
    by_repo = {model['repo']: model for model in nf4_manifest['base_models'].values()}
    if not models <= set(by_repo):
        raise SystemExit(f'Circuits use models outside the pinned set: {sorted(models - set(by_repo))}')
    circuitkit = json.loads(importlib.metadata.distribution('circuitkit').read_text('direct_url.json'))
    manifest = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(), 'git_commit': git('rev-parse', 'HEAD'),
                'files': files, 'models': [by_repo[model] for model in sorted(models)],
                'circuitkit': {'url': circuitkit['url'], 'commit': circuitkit['vcs_info']['commit_id']}}
    (FAIR / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    subprocess.run(['git', 'archive', '--format=tar', '-o', str(FAIR / 'code.tar'), 'HEAD'], cwd=ROOT, check=True)
    api().upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=str(FAIR), path_in_repo='fair',
                        delete_patterns=['run/**', 'code.tar', 'manifest.json'],
                        commit_message=f'Fair baseline inputs for {manifest["git_commit"][:7]}')
    log(f'fair bundle uploaded: {len(files)} files, models {sorted(models)}')


def fair_prepare(args) -> None:
    """On the GPU host: fetch the circuits and task files, and the models at the laptop's revisions, all by SHA-256."""
    from huggingface_hub import snapshot_download

    private_repo(args.repo)
    work = Path(args.work)
    manifest = json.loads((work / 'fair' / 'manifest.json').read_text(encoding='utf-8'))
    snapshot_download(args.repo, repo_type='dataset', local_dir=str(work), allow_patterns=['fair/run/**'])
    for name, identity in manifest['files'].items():
        verify_sha(work / 'fair' / 'run' / name, identity, 'Fair baseline input')
    hub = Path(os.environ['HF_HOME']) / 'hub'
    for model in manifest['models']:
        log(f'{model["repo"]} at {model["revision"][:10]}')
        folder = Path(snapshot_download(model['repo'], revision=model['revision'], allow_patterns=list(model['files'])))
        for name, identity in model['files'].items():
            verify_sha(folder / name, identity, f'{model["repo"]} checkpoint')
        # CircuitKIT loads the model by name, so point the cache's main ref at the laptop's revision.
        refs = hub / ('models--' + model['repo'].replace('/', '--')) / 'refs'
        refs.mkdir(parents=True, exist_ok=True)
        (refs / 'main').write_text(model['revision'])
    log('fair baseline inputs match the laptop by SHA-256')


def fair_upload(args) -> None:
    """On the GPU host: put each circuit's fair_baseline.json, the log and the exit status in fair/results/."""
    private_repo(args.repo)
    work = Path(args.work)
    results = work / 'fair-results'
    shutil.rmtree(results, ignore_errors=True)
    results.mkdir(parents=True)
    for path in (work / 'fair' / 'run' / 'circuits').glob('*/fair_baseline.json'):
        (results / path.parent.name).mkdir()
        shutil.copy2(path, results / path.parent.name / path.name)
    if (work / 'fair-baseline.log').is_file():
        shutil.copy2(work / 'fair-baseline.log', results / 'fair-baseline.log')
    (results / 'job-status.json').write_text(json.dumps({'exit_code': int(args.status),
                                                         'finished_at': datetime.now(timezone.utc).isoformat()}))
    api().upload_folder(repo_id=args.repo, repo_type='dataset', folder_path=str(results), path_in_repo='fair/results',
                        commit_message='Fair baseline results')


def fair_fetch(args) -> None:
    """On the laptop: copy each circuit's fair_baseline.json from fair/results/ into the run's circuit folder."""
    from huggingface_hub import snapshot_download

    private_repo(args.repo)
    staging = STAGING.parent / 'remote-fair-fetch'
    snapshot_download(args.repo, repo_type='dataset', local_dir=str(staging), allow_patterns=['fair/results/**'])
    copied = {}
    for path in sorted((staging / 'fair' / 'results').glob('*/fair_baseline.json')):
        target = SOURCE / 'circuits' / path.parent.name / 'fair_baseline.json'
        if not target.parent.is_dir():
            raise SystemExit(f'No circuit folder for {path.parent.name}')
        if target.exists() and not args.replace:
            raise SystemExit(f'{target} exists; pass --replace to overwrite it')
        shutil.copy2(path, target)
        result = json.loads(path.read_text(encoding='utf-8'))
        copied[path.parent.name] = {'random_mean': round(result['mean'], 4), 'circuit': result['circuit_patching_ratio'],
                                    'check': result['check_ratio'], 'device': result.get('hardware', {}).get('gpu')}
    print(json.dumps(copied, indent=2))


def launch_hf(args) -> None:
    from huggingface_hub import get_token, run_job

    private_repo(args.repo)
    env = {'BUNDLE_REPO': args.repo, 'PLATFORM': args.platform, 'WORK': '/tmp/lexsi'}
    if args.job == 'fair':
        env.update(CODE_TAR='fair/code.tar', ENTRY='remote_fair_baseline.sh')
    if args.dry_run:
        print(json.dumps({'image': args.image, 'flavor': args.flavor, 'timeout': args.timeout, 'env': env,
                          'command': ['bash', '-c', BOOTSTRAP]}, indent=2))
        return
    token = get_token()
    if not token:
        raise SystemExit('No Hugging Face login on this machine; run `hf auth login` yourself first')
    job = run_job(image=args.image, command=['bash', '-c', BOOTSTRAP], env=env, secrets={'HF_TOKEN': token},
                  flavor=args.flavor, timeout=args.timeout, name=f'lexsi-{args.job}', labels={'project': 'lexsi'})
    print(json.dumps({'id': job.id, 'url': getattr(job, 'url', None), 'flavor': args.flavor, 'timeout': args.timeout}))


def cache_get(args) -> None:
    """Unpack a cached build from the private repo into --to; exit 1 if there is none."""
    from huggingface_hub import hf_hub_download

    private_repo(args.repo)
    if args.name not in api().list_repo_files(args.repo, repo_type='dataset'):
        raise SystemExit(1)
    path = hf_hub_download(args.repo, args.name, repo_type='dataset', local_dir=str(Path(args.work) / 'cache-download'))
    target = Path(args.to)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        archive.extractall(target, filter='data')


def cache_put(args) -> None:
    """Pack --from into a tarball and keep it in the private repo, so a retry on the same GPU type skips the build."""
    private_repo(args.repo)
    archive = Path(args.work) / 'cache-upload.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(args.source, arcname='.')
    api().upload_file(path_or_fileobj=str(archive), path_in_repo=args.name, repo_id=args.repo, repo_type='dataset',
                      commit_message=f'Build cache {args.name}')


def fetch(args) -> None:
    """Download a platform's synced output into runs/experiments/full/earth-nf4-<platform> (git-ignored)."""
    from huggingface_hub import snapshot_download

    private_repo(args.repo)
    prefix = f'results/{args.platform}'
    staging = STAGING.parent / 'remote-nf4-fetch'
    snapshot_download(args.repo, repo_type='dataset', local_dir=str(staging), allow_patterns=[prefix + '/**'])
    target = FETCHED / f'earth-nf4-{args.platform}'
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(staging / prefix, target)
    status = target / 'nf4-environment.json'
    state = json.loads(status.read_text(encoding='utf-8')) if status.is_file() else {}
    print(json.dumps({'output': str(target), 'status': state.get('status'),
                      'completed_stages': len(state.get('completed_stages', [])), 'stages': len(state.get('stages', []))}))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    command = sub.add_parser('bundle'); command.add_argument('--no-upload', action='store_true')
    command.add_argument('--replace-with-runs', action='store_true',
                         help='replace the bundle even though runs started from it (they can no longer resume)')
    command.set_defaults(fn=bundle)
    for name, fn in (('prepare', prepare), ('verify', verify), ('restore', restore), ('sync', sync)):
        command = sub.add_parser(name); command.set_defaults(fn=fn)
        command.add_argument('--work', required=True)
        if name in ('restore', 'sync'):
            command.add_argument('--output', required=True)
            command.add_argument('--platform', required=True)
        if name == 'sync':
            command.add_argument('--interval', type=int, default=900)
    command = sub.add_parser('launch-hf'); command.set_defaults(fn=launch_hf)
    command.add_argument('--flavor', default='a10g-small')
    command.add_argument('--timeout', default='11h')
    command.add_argument('--platform', default='hf-a10g')
    command.add_argument('--image', default='nvidia/cuda:12.8.1-devel-ubuntu24.04')
    command.add_argument('--dry-run', action='store_true')
    command.add_argument('--job', choices=('nf4', 'fair'), default='nf4',
                         help='nf4: the NF4 check; fair: the fair circuit baseline (inputs from fair-bundle)')
    command = sub.add_parser('fair-bundle'); command.set_defaults(fn=fair_bundle)
    command = sub.add_parser('fair-prepare'); command.set_defaults(fn=fair_prepare)
    command.add_argument('--work', required=True)
    command = sub.add_parser('fair-upload'); command.set_defaults(fn=fair_upload)
    command.add_argument('--work', required=True); command.add_argument('--status', required=True)
    command = sub.add_parser('fair-fetch'); command.set_defaults(fn=fair_fetch)
    command.add_argument('--replace', action='store_true')
    for name, fn in (('cache-get', cache_get), ('cache-put', cache_put)):
        command = sub.add_parser(name); command.set_defaults(fn=fn)
        command.add_argument('--work', required=True); command.add_argument('--name', required=True)
        command.add_argument('--to' if name == 'cache-get' else '--from', dest='to' if name == 'cache-get' else 'source',
                             required=True)
    command = sub.add_parser('fetch'); command.set_defaults(fn=fetch)
    command.add_argument('--platform', required=True)
    for command in sub.choices.values():
        command.add_argument('--repo', default=REPO)
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
