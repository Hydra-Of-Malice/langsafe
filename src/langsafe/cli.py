"""Executable multilingual safety-repair pipeline; all raw completions stay local."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from .config import ROOT, capability_languages, load_spec, spec_digest
from .experiment import atomic_json, capability_path, checkpoint_identity, file_sha256, select_prompts
from .provenance import Ledger

RUNS = Path(os.environ.get('LANGSAFE_RUNS_DIR', ROOT / 'runs'))
JUDGE_SETS = ('router', 'local', 'rules')


def _judged_path(tag: str, judges: str) -> Path:
    return RUNS / 'judged' / (f'{tag}.jsonl' if judges == 'router' else f'{tag}.{judges}.jsonl')


def _judges_cfg(spec: dict, judges: str) -> list[dict]:
    return {'router': spec['judging']['judges'], 'local': spec['judging']['local_judges'], 'rules': []}[judges]


def _hand_check_settings(spec: dict) -> dict:
    """What a sheet selects. A spec without a cap keeps every disagreement, which asks for more human work, not less."""
    return {'audit_size': spec['judging']['hand_check_sample'], 'audit_seed': spec['data']['seed'],
            'disagreement_cap': spec['judging'].get('hand_check_disagreement_cap')}


def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _ledger(step: str, payload: dict) -> None:
    Ledger(RUNS / 'ledger.jsonl').append(step, {**payload, 'spec': spec_digest(load_spec())})


def _heldout(family: str, limit: int | None):
    return _prompts(family, 'heldout', limit)


def _prompts(family: str, split: str, limit: int | None):
    from .data.build import load_split
    return select_prompts(load_split(split), load_spec()['languages'][family], split, limit)


def cmd_build_data(a):
    from .data.build import build
    result = build(translate=a.translate, use_curatorkit=not a.no_curatorkit,
                   translator=a.translator, family=a.family)
    print(json.dumps({k: result.get(k) for k in ('files', 'counts', 'coverage', 'translation_failures')},
                     indent=2, ensure_ascii=False))


def _decoding(a):
    decoding = dict(load_spec()['decoding'])
    decoding['batch_size'] = a.batch_size or load_spec().get('experiment', {}).get('local_batch_size', 2)
    if a.max_new_tokens:
        decoding['max_new_tokens'] = a.max_new_tokens
    return decoding


def _steering_context(a, model):
    if getattr(a, 'steering', None):
        from .repair.arms import steered
        if a.strength is None:
            raise ValueError('--steering requires --strength')
        return steered(model, Path(a.steering), a.strength)
    return nullcontext()


def _variant(a):
    variant = {'checkpoint_identity': checkpoint_identity(a.model)}
    if a.quant:      # the weight fingerprint can't tell precisions apart, so each answer records its own
        variant['quant'] = a.quant
    if a.steering:
        variant.update(steering_sha256=file_sha256(Path(a.steering) / 'refusal_direction.pt'),
                       strength=a.strength)
    return variant


def cmd_generate(a):
    from .gen.generate import Generator
    from .gen.model import load
    from .data.build import load_split
    spec = load_spec()
    prompts = select_prompts(load_split(a.split), spec['languages'][a.family], a.split, a.limit)
    if not prompts:
        raise ValueError(f'No {a.split} prompts; build the selected family data first')
    decoding = _decoding(a)
    model = None
    if a.gguf:
        from transformers import AutoTokenizer
        from .gen.model import resolve_model
        from .quant.gguf import GGUFGenerator
        tokenizer = AutoTokenizer.from_pretrained(resolve_model(a.model))
        generator = GGUFGenerator(tokenizer, decoding, a.tag, a.gguf, server=a.server,
                                  cache_dir=RUNS / 'generations')
    else:
        model, tokenizer = load(a.model, dtype=decoding['dtype'], quant=a.quant, adapter=a.adapter)
        generator = Generator(model, tokenizer, decoding, a.tag, cache_dir=RUNS / 'generations', variant=_variant(a))
    outputs = []
    conditions = [a.preamble] if a.preamble else spec['preamble_conditions']
    with _steering_context(a, model):
        for preamble in conditions:
            outputs.extend(generator.run(prompts, preamble))
    path = RUNS / 'gens' / f'{a.tag}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in outputs), encoding='utf-8')
    temporary.replace(path)
    _ledger('generate', {'tag': a.tag, 'model': a.model, 'quant': a.quant, 'adapter': a.adapter,
                         'gguf': a.gguf, 'steering': a.steering, 'strength': a.strength,
                         'fingerprint': generator.fp, 'split': a.split, 'n': len(outputs),
                         'artifact_sha256': file_sha256(path), 'decoding': decoding})
    print(f'{len(outputs)} generations -> {path} (fingerprint {generator.fp})', flush=True)


def cmd_judge(a):
    from .data.build import load_split
    from .eval.judge_run import judge_generations, write_hand_check_sheet
    spec = load_spec()
    prompts = {p.id: p for p in load_split(a.split)}
    generations = _read_rows(RUNS / 'gens' / f'{a.tag}.jsonl')
    if any(g['id'] not in prompts for g in generations):
        raise ValueError('Generation IDs do not belong to the requested split/data')
    calibration = json.loads(Path(a.calibration).read_text()) if a.calibration else None
    result = judge_generations(generations, prompts, _judges_cfg(spec, a.judges), _judged_path(a.tag, a.judges),
                               **_hand_check_settings(spec), min_kappa=spec['judging']['min_judge_kappa'],
                               min_ac1=spec['judging']['min_judge_ac1'],
                               calibration=calibration,
                               primary_safety_axis=spec.get('experiment', {}).get('primary_safety_axis', 'refusal'))
    queue = result.pop('queue')
    sample = result.pop('audit_sample', [])
    sheet = RUNS / 'hand_check' / f'{a.tag}.{a.judges}.csv'
    write_hand_check_sheet(queue, sample, sheet)
    _ledger('judge', {'tag': a.tag, 'split': a.split, 'judge_set': a.judges, **result})
    print(json.dumps({**result, 'hand_check_sheet': str(sheet)}, indent=2, ensure_ascii=False, allow_nan=False))


def cmd_calibrate(a):
    from .eval.judge_run import import_calibration, compare_judge_sets
    if a.compare_router:
        result = compare_judge_sets(_read_rows(_judged_path(a.tag, 'local')),
                                    _read_rows(_judged_path(a.tag, 'router')))
        output = RUNS / 'calibration' / f'{a.tag}.router-comparison.json'
        atomic_json(output, result)
    else:
        if not a.sheet:
            raise ValueError('calibrate requires --sheet or --compare-router')
        output = RUNS / 'calibration' / f'{a.tag}.{a.judges}.json'
        result = import_calibration(Path(a.sheet), output,
                                    judged_rows=_read_rows(_judged_path(a.tag, a.judges)),
                                    **_hand_check_settings(load_spec()))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    _ledger('calibrate', {'tag': a.tag, 'artifact': str(output), 'sha256': file_sha256(output)})


def cmd_audit(a):
    from .data.build import load_split
    from .eval.auditkit_bridge import record_run, to_samples
    prompts = {p.id: p for p in load_split(a.split)}
    generations = _read_rows(RUNS / 'gens' / f'{a.tag}.jsonl')
    judged = _read_rows(_judged_path(a.tag, a.judges))
    by_key = {(r['id'], r['preamble']): r for r in generations}
    if len(by_key) != len(generations) or any((r['id'], r['preamble']) not in by_key for r in judged):
        raise ValueError('AuditKIT inputs have duplicate or unmatched generation identities')
    variant = f'{a.tag}.{a.judges}'
    output = RUNS / 'auditkit' / variant
    record_run(to_samples(prompts, judged, by_key), variant, output,
               metadata={'model_fingerprints': sorted({r['fingerprint'] for r in generations}),
                         'judge_set': a.judges, 'split': a.split, 'spec': spec_digest(load_spec())})
    path = output / f'{variant}.json'
    _ledger('auditkit', {'tag': a.tag, 'judge_set': a.judges, 'artifact': str(path),
                         'sha256': file_sha256(path)})
    print(f'AUDITKIT_ARTIFACT={path}', flush=True)


def cmd_capability(a):
    from .eval.capability import accuracy, load_items
    from .gen.model import load
    from huggingface_hub import try_to_load_from_cache
    spec = load_spec()
    languages = a.languages.split(',') if a.languages else capability_languages(spec, a.family)
    if not languages:
        raise ValueError('No capability languages selected')
    path = capability_path(RUNS, a.tag, a.preamble or 'default')
    settings = {'model': a.model, 'quant': a.quant, 'adapter': a.adapter,
                'steering': a.steering, 'strength': a.strength, 'gguf': a.gguf,
                'languages': languages, 'limit': a.limit, 'preamble': a.preamble or 'default',
                'mmlu_split': a.mmlu_split,
                'spec': spec_digest(spec), 'checkpoint_identity': checkpoint_identity(a.model),
                'variant': _variant(a), 'gguf_sha256': file_sha256(a.gguf) if a.gguf else None}
    prepared = {lang: load_items(lang, a.limit, split=a.mmlu_split) for lang in languages}
    hashes = {lang: hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
              for lang, items in prepared.items()}
    settings['data_sha256'] = hashes
    result = json.loads(path.read_text()) if path.exists() else {'metadata': settings, 'languages': {}, 'complete': False}
    if any(result['metadata'].get(k) != v for k, v in settings.items()):
        raise ValueError('Capability tag already has different settings; use a new tag')
    if result.get('complete') and all(result['languages'].get(l, {}).get('data_sha256') == hashes[l]
                                    and result['languages'][l].get('n') == len(prepared[l]) for l in languages):
        print(f'Capability already complete: {path}')
        return
    model = None
    if a.gguf:
        from transformers import AutoTokenizer
        from .gen.model import resolve_model
        from .quant.gguf import gguf_mmlu_accuracy
        tokenizer = AutoTokenizer.from_pretrained(resolve_model(a.model))
    else:
        model, tokenizer = load(a.model, dtype=spec['decoding']['dtype'], quant=a.quant, adapter=a.adapter)
    config_path = try_to_load_from_cache(a.model, 'config.json') if not Path(a.model).exists() else str(Path(a.model) / 'config.json')
    result['metadata']['revision'] = str(Path(config_path).parent.name) if isinstance(config_path, str) else None
    with _steering_context(a, model):
        for language in languages:
            items = prepared[language]
            data_hash = hashes[language]
            record = result['languages'].setdefault(language, {'data_sha256': data_hash, 'per_item': []})
            if record['data_sha256'] != data_hash:
                raise ValueError(f'Capability data changed for {language}')
            for start in range(len(record['per_item']), len(items), 20):
                chunk = items[start:start + 20]
                if a.gguf:
                    scored = gguf_mmlu_accuracy(tokenizer, chunk, server=a.server, preamble=a.preamble or 'default')
                    if scored.get('n_unscorable') or not scored.get('complete', True):
                        atomic_json(RUNS / 'capability' / f'{a.tag}.unscorable.json', scored)
                        reasons = sorted({row['reason'] for row in scored.get('outcomes', []) if row.get('reason')})
                        raise RuntimeError(f'GGUF capability is unscorable: {reasons[:3]}')
                else:
                    scored = accuracy(model, tokenizer, chunk,
                                      batch_size=a.batch_size or spec.get('experiment', {}).get('capability_batch_size', 1),
                                      preamble=a.preamble or 'default')
                record['per_item'].extend(scored['per_item'])
                record.update(n=len(record['per_item']), correct=sum(record['per_item']))
                record['accuracy'] = record['correct'] / record['n']
                atomic_json(path, result)
                print(f'{a.tag}/{language}: {record["n"]}/{len(items)}; {record["correct"]} correct', flush=True)
    result['complete'] = all(result['languages'][l]['n'] == len(prepared[l]) for l in languages)
    atomic_json(path, result)
    _ledger('capability', {'tag': a.tag, 'model': a.model, 'artifact': str(path), 'sha256': file_sha256(path)})


def cmd_judge_warm(a):
    """Fill the verdict cache for many tags with one load of each judge; the per-tag `judge` stages then hit the cache."""
    from .data.build import load_split
    from .judge.llm_judge import LLMJudge
    prompts = {p.id: p for p in load_split(a.split)}
    rows = [row for tag in a.tags.split(',') for row in _read_rows(RUNS / 'gens' / f'{tag}.jsonl')]
    items = [(prompts[row['id']].text, row['response'], row['lang']) for row in rows]
    harmful = [item for item, row in zip(items, rows) if row['kind'] == 'harmful']
    for cfg in _judges_cfg(load_spec(), a.judges):
        kwargs = {k: cfg[k] for k in ('quant', 'dtype', 'batch_size', 'checkpoint_identity', 'cache_path', 'gguf')
                  if k in cfg}
        judge = LLMJudge(cfg['name'], cfg['base_url'], cfg['model'], **kwargs)
        try:
            judge.judge_many(items)
            judge.judge_safety_many(harmful)
        finally:
            judge.close()
    print(f'{len(items)} responses warmed for {a.judges} judges', flush=True)


def cmd_circuit(a):
    from transformers import AutoTokenizer
    from .circuits.discover import discover
    from .circuits.task import DEFAULT_OPENERS, english_pairs, openers_from_judged, translate_pairs, write_task_csv
    from .gen.model import resolve_model
    spec = load_spec()
    openers = None
    if a.task_csv:
        task = Path(a.task_csv)
    else:
        pairs = english_pairs(a.pairs)
        if a.lang != 'en':
            from .llm_client import make_client
            translation = spec['translation']
            client = make_client(translation['base_url'], translation['model'],
                                 batch_size=translation.get('batch_size', 2), quant=translation.get('quant'))
            try:
                pairs = translate_pairs(client, pairs, a.lang)
            finally:
                client.close()
        tokenizer = AutoTokenizer.from_pretrained(resolve_model(a.model))
        openers = DEFAULT_OPENERS[a.lang]
        if a.openers_tag:
            answers = {(row['id'], row['preamble']): row for row in _read_rows(RUNS / 'gens' / f'{a.openers_tag}.jsonl')}
            openers = openers_from_judged(_read_rows(_judged_path(a.openers_tag, a.judges)), answers, a.lang, tokenizer)
        print(f'{a.lang}: refusal opener {openers[0]!r}, compliance opener {openers[1]!r}', flush=True)
        task = write_task_csv(tokenizer, pairs, a.lang, a.tag, openers, preamble=a.preamble, limit=a.examples)
    with open(task, encoding='utf-8', newline='') as stream:
        examples = sum(1 for _ in csv.DictReader(stream))
    result = discover(a.model, a.tag, a.lang, task, n_examples=a.examples, ig_steps=a.ig_steps,
                      verify_parity=not a.no_parity_check)
    result = {**result, 'task_examples': examples, 'openers': openers}
    atomic_json(RUNS / 'circuits' / f'{a.tag}__{a.lang}' / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


def cmd_repair(a):
    from .repair import arms
    spec = load_spec()
    model_id = a.model or spec['models']['drifted'][a.family]
    if a.arm == 'resta':
        output = arms.resta(model_id, a.knob, a.family)
    elif a.arm == 'circuit':
        if not a.scores:
            raise ValueError('circuit repair requires --scores')
        output = arms.circuit(model_id, a.scores, a.knob, a.family, top_fraction=a.top_fraction,
                              random_seed=a.random_seed)
    elif a.arm == 'circuit_merge':
        if not a.scores:
            raise ValueError('circuit repair requires --scores')
        output = arms.circuit(model_id, a.scores, a.knob, a.family, top_fraction=a.top_fraction, vector='drift',
                              random_seed=a.random_seed)
    elif a.arm == 'merge_back':
        output = arms.merge_back(model_id, a.knob, a.family)
    else:
        from .gen.model import load
        prompts = _prompts(a.family, 'repair', a.limit)
        model, tokenizer = load(model_id, dtype=spec['decoding']['dtype'])
        output = arms.extract_direction(model, tokenizer,
                                        [p.text for p in prompts if p.kind == 'harmful'],
                                        [p.text for p in prompts if p.kind == 'benign'], a.family,
                                        pick_layer=a.layer, preamble=a.preamble)
    print(f'REPAIR_ARTIFACT={Path(output).resolve()}', flush=True)


def _local_checkpoint(model_id: str) -> Path:
    from .gen.model import resolve_model
    from huggingface_hub import snapshot_download
    resolved = resolve_model(model_id)
    if Path(resolved).is_dir():
        return Path(resolved)
    # Hub 2.x checks snapshot completeness even offline. Conversion needs the
    # checkpoint/tokenizer, without optional model-card and image assets.
    return Path(snapshot_download(resolved, local_files_only=True,
                                  allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja']))


def cmd_convert(a):
    from .quant.gguf import convert_hf_to_gguf
    result = convert_hf_to_gguf(_local_checkpoint(a.model), a.output, a.llama_cpp, overwrite=a.overwrite)
    _ledger('convert', {'model': a.model, 'checkpoint_identity': checkpoint_identity(a.model),
                        'output': str(result), 'sha256': file_sha256(result), 'precision': 'f16'})
    print(f'CONVERTED_ARTIFACT={result}', flush=True)


def cmd_quantize(a):
    from .quant.gguf import convert_hf_to_gguf, quantize_gguf
    output = Path(a.output)
    source = Path(a.input) if a.input else output.with_name(output.stem + '-f16.gguf')
    if not a.input:
        if not a.model:
            raise ValueError('quantize requires --model or --input')
        source = convert_hf_to_gguf(_local_checkpoint(a.model), source, a.llama_cpp, overwrite=a.overwrite)
    result = quantize_gguf(source, output, a.llama_cpp, quantization='Q4_K_M', overwrite=a.overwrite)
    _ledger('quantize', {'input': str(source), 'output': str(result), 'sha256': file_sha256(result), 'quant': 'Q4_K_M'})
    print(f'QUANTIZED_ARTIFACT={result}')


def cmd_demo(a):
    from .data.records import Prompt
    from .gen.generate import Generator
    from .gen.model import load
    model, tokenizer = load(a.model, dtype='float16')
    generator = Generator(model, tokenizer, {'batch_size': 1, 'do_sample': False,
                          'max_new_tokens': a.max_new_tokens, 'repetition_penalty': 1.0},
                          'interactive-demo', cache_dir=RUNS / 'demo-cache', variant=_variant(a))
    with _steering_context(a, model):
        while True:
            text = a.prompt or input('You (empty to exit): ').strip()
            if not text:
                break
            group = hashlib.sha256(text.encode()).hexdigest()[:16]
            prompt = Prompt(id=f'demo-{group}::{a.lang}', group=group, lang=a.lang,
                            kind='benign', text=text, source='interactive', source_id=group)
            row = generator.run([prompt], a.preamble)[0]
            print(row['response'], flush=True)
            if a.prompt:
                break


def cmd_report(a):
    from .experiment_report import build_report
    result = build_report(a.tags.split(','), a.family, a.judges, RUNS, output=Path(a.output) if a.output else None)
    print(result.read_text(encoding='utf-8'))


def cmd_experiment(a):
    from .experiment_run import run_experiment
    run_experiment(a.family, a.profile, stop_after=a.stop_after, judges=a.judges,
                   skip_circuit=a.skip_circuit, skip_quantization=a.skip_quantization)


def cmd_ledger_verify(a):
    ok, bad = Ledger(RUNS / 'ledger.jsonl').verify()
    print('ledger OK' if ok else f'ledger BROKEN at entry {bad}')
    if not ok:
        raise SystemExit(1)


def cmd_status(a):
    directory = RUNS if a.directory is None else Path(a.directory)
    manifest = directory / 'experiment.json'
    if not manifest.is_file():
        raise ValueError(f'No experiment manifest in {directory}')
    experiment = json.loads(manifest.read_text())
    result = {'directory': str(directory.resolve()), 'identity': experiment['identity'],
              'recorded_completed_stages': list(experiment['stages']),
              'stage_elapsed_seconds': {name: stage.get('result', {}).get('elapsed_seconds')
                                        for name, stage in experiment['stages'].items()}}
    for name in ('current-stage', 'last-failure', 'status'):
        path = directory / f'{name}.json'
        if path.is_file():
            result[name.replace('-', '_')] = json.loads(path.read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog='langsafe')
    parser.add_argument('--spec', type=Path, help='Experiment spec; canonical spec is not modified')
    parser.add_argument('--runs-dir', type=Path, help='Isolated experiment artifacts')
    sub = parser.add_subparsers(required=True)
    command = sub.add_parser('build-data'); command.set_defaults(fn=cmd_build_data)
    command.add_argument('--translate', action='store_true')
    command.add_argument('--no-curatorkit', action='store_true')
    command.add_argument('--translator', choices=['router','local'], default='local')
    command.add_argument('--family', choices=['fire','earth'])
    for name, function in [('generate', cmd_generate), ('capability', cmd_capability)]:
        command = sub.add_parser(name); command.set_defaults(fn=function)
        command.add_argument('--model', required=True); command.add_argument('--tag', required=True)
        command.add_argument('--family', choices=['fire','earth'], required=name == 'generate')
        command.add_argument('--quant', choices=['nf4']); command.add_argument('--adapter')
        command.add_argument('--steering'); command.add_argument('--strength', type=float)
        command.add_argument('--limit', type=int); command.add_argument('--batch-size', type=int)
        if name == 'capability':
            command.add_argument('--mmlu-split', choices=['test','dev'], default='test')
        command.add_argument('--preamble', choices=['default','none'])
        command.add_argument('--gguf'); command.add_argument('--server', default='http://127.0.0.1:8080')
        if name == 'generate':
            command.add_argument('--split', choices=['repair','heldout'], default='heldout')
            command.add_argument('--max-new-tokens', type=int)
        else:
            command.add_argument('--languages', help='Comma-separated capability language codes')
    command = sub.add_parser('judge'); command.set_defaults(fn=cmd_judge)
    command.add_argument('--tag', required=True); command.add_argument('--judges', choices=JUDGE_SETS, default='local')
    command.add_argument('--split', choices=['repair','heldout'], default='heldout'); command.add_argument('--calibration')
    command = sub.add_parser('judge-warm'); command.set_defaults(fn=cmd_judge_warm)
    command.add_argument('--tags', required=True); command.add_argument('--judges', choices=JUDGE_SETS, default='local')
    command.add_argument('--split', choices=['repair','heldout'], default='heldout')
    command = sub.add_parser('calibrate'); command.set_defaults(fn=cmd_calibrate)
    command.add_argument('--tag', required=True); command.add_argument('--judges', choices=['local','router'], default='local')
    command.add_argument('--sheet'); command.add_argument('--compare-router', action='store_true')
    command = sub.add_parser('audit'); command.set_defaults(fn=cmd_audit)
    command.add_argument('--tag', required=True); command.add_argument('--judges', choices=JUDGE_SETS, default='local')
    command.add_argument('--split', choices=['repair','heldout'], default='heldout')
    command = sub.add_parser('circuit'); command.set_defaults(fn=cmd_circuit)
    command.add_argument('--preamble', choices=['default','none'], default='default')
    command.add_argument('--model', required=True); command.add_argument('--tag', required=True)
    command.add_argument('--lang', default='en'); command.add_argument('--pairs', type=int, default=32)
    command.add_argument('--examples', type=int, default=16); command.add_argument('--ig-steps', type=int, default=3)
    command.add_argument('--task-csv'); command.add_argument('--no-parity-check', action='store_true')
    command.add_argument('--openers-tag', help='read the answer tokens from these judged answers, not the stock phrases')
    command.add_argument('--judges', choices=['router','local','rules'], default='local')
    command = sub.add_parser('repair'); command.set_defaults(fn=cmd_repair)
    command.add_argument('--family', choices=['fire','earth'], required=True); command.add_argument('--model')
    command.add_argument('--arm', choices=['resta','circuit','circuit_merge','merge_back','steer'], required=True)
    command.add_argument('--random-seed', type=int)
    command.add_argument('--knob', type=float, default=0.1); command.add_argument('--scores')
    command.add_argument('--top-fraction', type=float, default=0.2); command.add_argument('--layer', type=int, default=24)
    command.add_argument('--limit', type=int, default=20)
    command.add_argument('--preamble', choices=['default','none'], default='default')
    command = sub.add_parser('convert'); command.set_defaults(fn=cmd_convert)
    command.add_argument('--model', required=True); command.add_argument('--output', required=True)
    command.add_argument('--llama-cpp', default=os.environ.get('LANGSAFE_LLAMA_CPP', 'cache/llama.cpp'))
    command.add_argument('--overwrite', action='store_true')
    command = sub.add_parser('quantize'); command.set_defaults(fn=cmd_quantize)
    command.add_argument('--model'); command.add_argument('--input'); command.add_argument('--output', required=True)
    command.add_argument('--llama-cpp', default=os.environ.get('LANGSAFE_LLAMA_CPP', 'cache/llama.cpp'))
    command.add_argument('--overwrite', action='store_true')
    command = sub.add_parser('demo'); command.set_defaults(fn=cmd_demo)
    command.add_argument('--model', default='CohereLabs/tiny-aya-fire'); command.add_argument('--prompt')
    command.add_argument('--lang', default='en'); command.add_argument('--max-new-tokens', type=int, default=256)
    command.add_argument('--preamble', choices=['default','none'], default='default')
    command.add_argument('--steering'); command.add_argument('--strength', type=float)
    command = sub.add_parser('report'); command.set_defaults(fn=cmd_report)
    command.add_argument('--tags', required=True); command.add_argument('--family', choices=['fire','earth'], required=True)
    command.add_argument('--judges', choices=JUDGE_SETS, default='local'); command.add_argument('--output')
    command = sub.add_parser('experiment'); command.set_defaults(fn=cmd_experiment)
    command.add_argument('--family', choices=['fire','earth'], default='earth')
    command.add_argument('--profile', choices=['development','full'], default='development')
    command.add_argument('--judges', choices=JUDGE_SETS, default='local')
    command.add_argument('--stop-after', choices=['data','baseline','circuit','repairs','quantization'])
    command.add_argument('--skip-circuit', action='store_true'); command.add_argument('--skip-quantization', action='store_true')
    command = sub.add_parser('ledger-verify'); command.set_defaults(fn=cmd_ledger_verify)
    command = sub.add_parser('status'); command.set_defaults(fn=cmd_status)
    command.add_argument('--directory', type=Path)
    args = parser.parse_args(argv)
    if args.spec:
        os.environ['LANGSAFE_SPEC'] = str(args.spec.resolve())
    if args.runs_dir:
        os.environ['LANGSAFE_RUNS_DIR'] = str(args.runs_dir.resolve())
        os.environ['LANGSAFE_DATA_DIR'] = str(args.runs_dir.resolve() / 'data')
        global RUNS
        RUNS = args.runs_dir.resolve()
    for field in ('tag',):
        value = getattr(args, field, None)
        if value and (not value.replace('-', '').replace('_', '').replace('.', '').isalnum() or '..' in value):
            parser.error('tag must contain letters, numbers, hyphens, underscores, or single dots')
    for field in ('limit', 'batch_size', 'max_new_tokens', 'pairs', 'examples', 'ig_steps'):
        value = getattr(args, field, None)
        if value is not None and value <= 0:
            parser.error(f'{field} must be positive')
    args.fn(args)


if __name__ == '__main__':
    main()
