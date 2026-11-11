#!/usr/bin/env python3
"""Plan the public Hugging Face release, and carry it out only on an explicit go-ahead.

    python scripts/publish_hf.py                          # print the plan: repos, files, sizes, what stays private
    python scripts/publish_hf.py --execute <plan digest>  # create the public repos and upload exactly that plan

Publishing is the team's decision (docs/submission/decisions.md, D3 and D4), so nothing is uploaded without --execute,
and --execute needs the digest the plan printed. Whatever runs is the plan the team reviewed.

What the plan holds:
- **Dataset.** The sanitised export (scripts/export_public.py), only if its final scan passed.
- **Models.** Every selected repair arm that meets the release rule. The rule: no language is significantly less safe
  than Earth, by paired 95% interval of the content-safe rate, in either preamble condition. Each arm's repo gets only
  the files a checkpoint needs (config, weights, tokenizer, chat template, provenance). The steering arm publishes its
  direction instead of weights.
- **GGUF.** The selected arm's Q4_K_M file.
- **Kept private.** The random-layer controls, plus any arm that fails the rule.

Each repo needs its README card in --cards (the filled templates). A missing card blocks --execute. The plan also
refuses any path from a private folder (model answers, judgements, audit records), whatever the allowlist says.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'experiments' / 'full' / 'earth'
EXPORT = ROOT / 'runs' / 'experiments' / 'public' / 'earth-full'
CARDS = ROOT / 'runs' / 'experiments' / 'public' / 'cards'
PREFIX = 'tiny-aya-earth-langsafe'   # LSAL section 2: no kit names in repo names
CHECKPOINT_FILES = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'chat_template.jinja', 'special_tokens_map.json', 'lexsi_provenance.json', 'model.safetensors',
                    'model.safetensors.index.json'}
STEERING_FILES = {'refusal_direction.pt', 'meta.json'}
EXPORT_FOLDERS = {'data', 'provenance', 'reports', 'verdicts'}
PRIVATE_PARTS = {'gens', 'generations', 'judged', 'auditkit', 'hand_check', 'calibration', 'audit', 'review',
                 'llm_cache.sqlite', 'remote-nf4-bundle', 'remote-fair-bundle', 'earth-router-sample'}


@dataclass
class Repo:
    repo_id: str
    repo_type: str
    files: dict[str, Path]          # path in the repo -> local file
    card: Path
    note: str = ''
    problems: list[str] = field(default_factory=list)

    def size(self) -> int:
        return sum(path.stat().st_size for path in self.files.values() if path.is_file())


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def is_private(path: Path) -> bool:
    return any(part in PRIVATE_PARTS for part in path.parts)


def release_rule(report: dict, tag: str) -> list[str]:
    """Languages where the arm is significantly less safe than Earth (paired 95% CI wholly below zero)."""
    worse = []
    for condition in ('default', 'none'):
        safety = (report['paired_comparisons'].get(f'{tag}/{condition}', {}).get('vs_drifted') or {}).get('safety') or {}
        for lang, cell in sorted(safety.items()):
            interval = (cell.get('content_safe') or {}).get('ci')
            if interval and interval[1] < 0:
                worse.append(f'{lang} {condition} ({100 * cell["content_safe"]["delta"]:+.1f} points)')
    return worse


def checkpoint_folder(model: str) -> str:
    return model.replace('\\', '/').rstrip('/').split('/')[-1]


def plan(namespace: str, run: Path = RUN, export: Path = EXPORT, cards: Path = CARDS) -> tuple[list[Repo], list[str]]:
    report, selection = read_json(run / 'report.json'), read_json(run / 'selection.json')
    repos, private = [], []
    manifest = export / 'EXPORT_MANIFEST.json'
    dataset = Repo(f'{namespace}/{PREFIX}-eval', 'dataset',
                   {path.relative_to(export).as_posix(): path for path in sorted(export.rglob('*')) if path.is_file()},
                   cards / 'dataset_card.md', 'sanitised export: prompts, labels, reports, provenance; no model text')
    if not manifest.is_file() or (read_json(manifest).get('scan') or {}).get('result') != 'passed':
        dataset.problems.append('the export has no passed final scan; rerun scripts/export_public.py')
    # The export's own allowlist decides what is inside its folders (its reports/judged holds aggregate summaries only).
    # Anything outside the folders it writes did not come from it.
    dataset.problems += [f'not part of the export: {name}' for name in dataset.files
                         if name != 'EXPORT_MANIFEST.json' and name.split('/')[0] not in EXPORT_FOLDERS]
    repos.append(dataset)
    selected = selection['selected_for_quantization']['tag']
    for arm, entry in selection['arms'].items():
        chosen = entry.get('selected')
        if not chosen:
            continue
        tag, knob = chosen['tag'], str(chosen['knob']).replace('.', 'p')
        worse = release_rule(report, tag)
        if worse:
            private.append(f'{tag}: fails the release rule in {", ".join(worse)}')
            continue
        name = f'{namespace}/{PREFIX}-{arm.replace("_", "-")}-{knob}'
        if chosen.get('steering'):
            folder = Path(chosen['steering'])
            folder = folder if folder.is_dir() else run / 'checkpoints' / checkpoint_folder(chosen['steering'])
            allowed = STEERING_FILES
        else:
            folder = run / 'checkpoints' / checkpoint_folder(chosen['model'])
            allowed = CHECKPOINT_FILES
        files, problems = {}, []
        for path in sorted(folder.rglob('*')) if folder.is_dir() else []:
            if not path.is_file():
                continue
            relative = path.relative_to(folder).as_posix()
            if relative in allowed:
                files[relative] = path
            else:
                problems.append(f'unexpected file left out: {relative}')
        if not files:
            problems.append(f'no files found in {folder}')
        repo = Repo(name, 'model', files, cards / f'model_card-{tag}.md',
                    'steering direction, applied at run time' if chosen.get('steering') else 'merged weights, FP16',
                    problems)
        repos.append(repo)
        if tag == selected:
            gguf = run / 'quantized' / f'{tag}-q4_k_m.gguf'
            repos.append(Repo(f'{name}-gguf', 'model', {gguf.name: gguf}, cards / f'model_card-{tag}-gguf.md',
                              'Q4_K_M GGUF of the selected arm', [] if gguf.is_file() else [f'missing {gguf}']))
    for control in selection.get('controls') or []:
        private.append(f"{control['tag']}: random-layer control, kept private by design")
    for repo in repos:
        checked = [repo.card] if repo.repo_type == 'dataset' else [*repo.files.values(), repo.card]
        for path in checked:
            if is_private(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                repo.problems.append(f'refused: {path} is under a private folder')
        if not repo.card.is_file():
            repo.problems.append(f'card missing: {repo.card}')
    return repos, private


def digest(repos: list[Repo]) -> str:
    shape = [[repo.repo_id, repo.repo_type, sorted((name, path.stat().st_size if path.is_file() else -1)
                                                   for name, path in repo.files.items())] for repo in repos]
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()[:16]


def describe(repos: list[Repo], private: list[str]) -> str:
    lines = []
    for repo in repos:
        lines.append(f'{repo.repo_type:7s} {repo.repo_id}  ({len(repo.files)} files, {repo.size() / 1e9:.2f} GB) - {repo.note}')
        lines += [f'          ! {problem}' for problem in repo.problems]
    lines.append(f'total upload {sum(repo.size() for repo in repos) / 1e9:.1f} GB')
    lines += [f'private  {item}' for item in private]
    lines.append(f'plan digest {digest(repos)}')
    return '\n'.join(lines)


def execute(repos: list[Repo], expected: str) -> None:
    from huggingface_hub import HfApi

    if digest(repos) != expected:
        raise SystemExit('The plan changed since it was reviewed; print it again and pass the new digest')
    blocking = [f'{repo.repo_id}: {problem}' for repo in repos for problem in repo.problems
                if not problem.startswith('unexpected file left out')]
    if blocking:
        raise SystemExit('Not publishing:\n' + '\n'.join(blocking))
    api = HfApi()
    for repo in repos:
        api.create_repo(repo.repo_id, repo_type=repo.repo_type, private=False, exist_ok=True)
        for name, path in repo.files.items():
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo.repo_id, repo_type=repo.repo_type,
                            commit_message=f'Add {name}')
        api.upload_file(path_or_fileobj=str(repo.card), path_in_repo='README.md', repo_id=repo.repo_id,
                        repo_type=repo.repo_type, commit_message='Add card')
        print(f'published https://huggingface.co/{"datasets/" if repo.repo_type == "dataset" else ""}{repo.repo_id}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--namespace', help='HF user or org; default: the logged-in user')
    parser.add_argument('--cards', type=Path, default=CARDS)
    parser.add_argument('--execute', metavar='DIGEST', help='publish the plan with this digest')
    args = parser.parse_args(argv)
    namespace = args.namespace
    if not namespace:
        from huggingface_hub import whoami
        namespace = whoami()['name']
    repos, private = plan(namespace, cards=args.cards)
    print(describe(repos, private))
    if args.execute:
        execute(repos, args.execute)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
