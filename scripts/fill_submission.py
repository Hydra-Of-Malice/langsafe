"""Fills the measured values in the submission drafts from a finished run, so nobody copies numbers by hand.

A value goes in only if the run's own files hold it. Everything else stays as its {{NAME}}: links, team decisions, and
the sentences that need someone to read the result first. `remaining.md` lists what is left in each draft. Which
[ALT-*] block survives follows the rule at the top of the article, and a case that rule can't settle keeps its markers.

    python scripts/fill_submission.py runs/experiments/full/earth docs/submission/filled
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ('article.md', 'model_card_template.md', 'dataset_card_template.md', 'roi.md')
SENTENCES = ROOT / 'docs' / 'submission' / 'sentences.json'
TEAM_SECTION = re.compile(r'^## (Appendix for the team|Placeholder list)', re.M)
PLACEHOLDER = re.compile(r'\{\{([A-Z0-9_]+)\}\}')
LIST_ROW = re.compile(r'^\| `\{\{([A-Z0-9_]+)\}\}` \| (.+?) \|', re.M)

LANGUAGES = {'AR': 'ar', 'SW': 'sw', 'JV': 'jv', 'BN': 'bn', 'TH': 'th', 'KO': 'ko', 'EN': 'en'}
LANGUAGE_NAMES = {'ar': 'Arabic', 'sw': 'Swahili', 'jv': 'Javanese', 'bn': 'Bengali', 'th': 'Thai', 'ko': 'Korean',
                  'en': 'English'}
CAPABILITY_LANGUAGE = {'AR': 'ar', 'SW': 'sw', 'JV': 'id', 'BN': 'bn', 'KO': 'ko', 'EN': 'en'}   # no Javanese or Thai set
CONDITIONS = {'default': 'with the preamble', 'none': 'without the preamble'}
ARMS = {'RESTA': 'resta', 'CIRCUIT': 'circuit', 'MERGE': 'merge_back', 'CMERGE': 'circuit_merge', 'STEER': 'steer'}
ARM_DETAILS = {   # name, formula, SafeTune function, donors; these match the table in the model card
    'resta': ('ReSta', 'Earth + α(Global − Base)', 'apply_resta', 'Tiny Aya Global, Tiny Aya Base'),
    'circuit': ('Circuit', 'Earth + s(Global − Base)', 'apply_ctheta', 'Tiny Aya Global, Tiny Aya Base'),
    'merge_back': ('Merge-back', '(1 − λ)Earth + λGlobal', 'apply_wise_ft', 'Tiny Aya Global'),
    'circuit_merge': ('Circuit merge', 'Earth + λ(Global − Earth)', 'apply_ctheta', 'Tiny Aya Global'),
    'steer': ('Steering', 'Refusal direction added at layer 24 at run time', None, None)}
PRECISIONS = {'FP16': '', 'Q4': '-q4', 'NF4': '-nf4'}
JUDGE_NAMES = {'local': ('local 8B', 'Aya Expanse 8B and Qwen3-8B')}


def dig(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def percent(value: float | None) -> str | None:
    return None if value is None else f'{100 * value:.1f}'


def signed(value: float | None) -> str | None:
    return None if value is None else f'{100 * value:+.1f}'


def difference(cell: dict | None) -> str | None:
    """A paired change in points, with its 95% CI."""
    if not cell or cell.get('delta') is None:
        return None
    return f"{signed(cell['delta'])} ({signed(cell['ci'][0])} to {signed(cell['ci'][1])})"


def rate_with_ci(rate: float | None, ci: list | None) -> str | None:
    if rate is None or not ci:
        return None
    return f'{percent(rate)}% ({percent(ci[0])}–{percent(ci[1])})'


def p_value(value: float | None) -> str | None:
    if value is None:
        return None
    return '<0.0001' if value < 0.0001 else f'{value:.4f}'


def unclear_share(cell: dict | None) -> float | None:
    """Unclear labels over harmful and benign rows together, the share the knob score in repair/select.py uses."""
    harmful, benign = dig(cell, 'harmful_refusal'), dig(cell, 'over_refusal')
    if not harmful or not benign or harmful.get('unclear') is None or benign.get('unclear') is None:
        return None
    return (harmful['unclear'] * harmful['n'] + benign['unclear'] * benign['n']) / (harmful['n'] + benign['n'])


def numbers_pass(gate: dict | None) -> bool:
    return bool(gate) and all(dig(gate, name, 'pass') for name in ('harmful_gap_closed', 'over_refusal_increase',
                                                                   'capability_drop', 'coverage'))


def gate_word(gate: dict | None) -> str | None:
    """`pass` also needs human calibration, so a bare 'no' would hide an arm that met every number."""
    if not gate:
        return None
    if gate.get('pass'):
        return 'yes'
    return 'numbers pass, not validated' if numbers_pass(gate) else 'no'


def largest(gate: dict | None, name: str) -> str | None:
    values = dig(gate, name, 'by_lang')
    return signed(max(values.values())) if values else None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


@dataclass
class Results:
    run: Path
    report: dict
    selection: dict
    localisation: dict
    manifest: dict

    @property
    def drifted(self) -> str:
        return self.report.get('family', 'earth')

    @property
    def target(self) -> str | None:
        return self.selection.get('worst_language')

    @property
    def condition(self) -> str:
        """The confirmed condition for the target language; with none, the condition the knobs were chosen in."""
        return (self.report.get('confirmed_conditions') or ['default'])[0]

    @property
    def best(self) -> dict:
        return self.selection.get('selected_for_quantization') or {}

    def gaps(self, condition: str) -> dict:
        return dig(self.report, 'gap_confirmation', 'conditions', condition) or {}

    def cell(self, tag: str, lang: str | None, condition: str | None = None) -> dict:
        return dig(self.report, 'models', tag, 'conditions', condition or self.condition, lang) or {}

    def capability(self, tag: str, lang: str) -> dict:
        return dig(self.report, 'models', tag, 'capability_by_condition', self.condition, lang) or {}

    def gate(self, tag: str | None) -> dict | None:
        return dig(self.report, 'thresholds', f'{tag}/{self.condition}')


def _safe_rate(results: Results, tag: str, lang: str | None, condition: str | None = None) -> float | None:
    return dig(results.cell(tag, lang, condition), 'content_safe', 'rate')


def gap_values(results: Results) -> dict:
    values = {}
    for suffix, condition in (('', 'default'), ('_NONE', 'none')):
        for code, lang in LANGUAGES.items():
            gap = results.gaps(condition).get(lang, {})
            confirmed = 'not measurable' if gap.get('measurable') is False else 'yes' if gap.get('confirmed') else 'no'
            values.update({f'GAP_{code}_GLOBAL{suffix}': percent(_safe_rate(results, 'global', lang, condition)),
                           f'GAP_{code}_EARTH{suffix}': percent(_safe_rate(results, results.drifted, lang, condition)),
                           f'GAP_{code}_FIRE{suffix}': percent(dig(results.report, 'comparison_models', 'fire', 'conditions',
                                                                   condition, lang, 'rate')),
                           f'GAP_{code}_DIFF{suffix}': difference(gap), f'GAP_{code}_P{suffix}': p_value(gap.get('p_mcnemar')),
                           f'GAP_{code}_OK{suffix}': confirmed if gap else None})
        for model, tag in (('GLOBAL', 'global'), ('EARTH', results.drifted)):
            rates = [_safe_rate(results, tag, lang, condition) for lang in LANGUAGES.values()]
            if all(rate is not None for rate in rates):
                name = 'DEFAULT' if condition == 'default' else 'NONE'
                values[f'HELDOUT_MEAN_{model}_{name}'] = percent(sum(rates) / len(rates))
                values[f'HELDOUT_WORST_{model}_{name}'] = percent(min(rates))
    tests = dig(results.report, 'gap_confirmation', 'tests')
    cells = [(lang, condition, cell) for condition in CONDITIONS for lang, cell in results.gaps(condition).items()]
    if tests:
        values.update({'N_GAP_TESTS': str(tests), 'GAP_P_THRESHOLD': f"{results.report['gap_confirmation']['alpha']:.4f}"})
    if cells:
        lang, condition, top = max(cells, key=lambda item: item[2]['delta'])
        values.update({'LARGEST_GAP_POINTS': f"{100 * top['delta']:.1f}",
                       'LARGEST_GAP_LANGUAGE': f'{LANGUAGE_NAMES[lang]}, {CONDITIONS[condition]}',
                       'LARGEST_GAP_CI': f"95% CI {signed(top['ci'][0])} to {signed(top['ci'][1])}",
                       'GAP_UPPER_BOUND': f"{100 * max(cell['ci'][1] for _, _, cell in cells):.1f}",
                       'SCALE_JUDGE_BEFORE': f"{100 * top['delta']:.1f} points"})
        confirmed = [f'{LANGUAGE_NAMES[lang]} {CONDITIONS[condition]}' for lang, condition, cell in cells if cell.get('confirmed')]
        values['CONFIRMED_CELLS'] = ', '.join(confirmed) if confirmed else None
    for model, tag in (('GLOBAL', 'global'), ('EARTH', results.drifted)):
        rates = [_safe_rate(results, tag, lang, condition) for condition in CONDITIONS for lang in LANGUAGES.values()]
        rates = [rate for rate in rates if rate is not None]
        if rates:
            values.update({f'{model}_RANGE_LOW': percent(min(rates)), f'{model}_RANGE_HIGH': percent(max(rates))})
    return values


def target_values(results: Results) -> dict:
    target = results.target
    if not target:
        return {}
    gap = results.gaps(results.condition).get(target, {})
    values = {'TARGET_LANGUAGE': LANGUAGE_NAMES[target], 'TARGET_CONDITION': CONDITIONS[results.condition],
              'TARGET_SAFE_EARTH': percent(_safe_rate(results, results.drifted, target)),
              'TARGET_SAFE_GLOBAL': percent(_safe_rate(results, 'global', target)),
              'TARGET_GAP_POINTS': None if not gap else f"{100 * gap['delta']:.1f}",
              'TARGET_GAP_CI': None if not gap else f"95% CI {signed(gap['ci'][0])} to {signed(gap['ci'][1])}",
              'TARGET_GAP_P': p_value(gap.get('p_mcnemar')), 'TARGET_GAP_N': None if not gap else str(gap['n'])}
    for letter, word, tag in (('G', 'GLOBAL', 'global'), ('E', 'EARTH', results.drifted), ('R', None, results.best.get('tag'))):
        split = dig(results.cell(tag, target), 'content_safe') or {}
        for name, key in (('SAFE', 'rate'), ('UNSAFE', 'unsafe'), ('INVALID', 'invalid'), ('DISPUTED', 'disputed')):
            share = percent(split.get(key))
            values[f'TARGET_{letter}_{name}'] = None if share is None else f'{share}%'
            if word:
                values[f'R_{word}_{name}'] = share
        if word:
            values[f'R_{word}_REFUSAL'] = percent(dig(results.cell(tag, target), 'harmful_refusal', 'rate'))
    for letter, tag in (('G', 'global'), ('E', results.drifted)):
        for word, condition in (('ON', 'default'), ('OFF', 'none')):
            cell = results.cell(tag, target, condition)
            split = cell.get('content_safe', {})
            values.update({f'FAIL_{letter}_{word}_SAFE': percent(split.get('rate')),
                           f'FAIL_{letter}_{word}_UNSAFE': percent(split.get('unsafe')),
                           f'FAIL_{letter}_{word}_INVALID': percent(split.get('invalid')),
                           f'FAIL_{letter}_{word}_DISPUTED': percent(split.get('disputed')),
                           f'FAIL_{letter}_{word}_DEGENERATE': percent(dig(cell, 'degenerate', 'rate'))})
    return values


def target_confirmation_values(results: Results) -> dict:
    """Whether the gap the repairs aim at passed the held-out test. The target comes from the repair split and confirmed
    gaps from held-out items, so a run can confirm a gap in one language and aim the repairs at another. Both values sit
    inside ALT-A, so they only need to cover a run where some gap was confirmed."""
    target, alpha = results.target, dig(results.report, 'gap_confirmation', 'alpha')
    cells = {condition: results.gaps(condition).get(target) for condition in CONDITIONS} if target else {}
    if alpha is None or not cells or not all(cells.values()):
        return {}
    name = LANGUAGE_NAMES[target]
    passed = [CONDITIONS[condition] for condition, cell in cells.items() if cell.get('confirmed')]
    if passed:
        where = ' and '.join(passed)
        return {'TARGET_CONFIRMATION_SENTENCE': f'That gap passed our test {where}.',
                'TARGET_CONFIRMATION_PARAGRAPH': f'Held-out items confirmed the gap in {name} {where}, so the repair '
                                                 'results there are claims. In any other condition they are exploratory.'}
    gap = cells[results.condition]
    sentence = (f"That gap did not pass our test (p = {p_value(gap.get('p_mcnemar'))} against a bar of {alpha:.4f}), "
                'so every repair result here is exploratory.')
    parts = []
    split = repair_split_rates(results.selection, target)
    if split and split[0] > split[1]:
        parts.append(f'The repair split picked {name}, where Global led Earth by {100 * (split[0] - split[1]):.1f} points.')
    parts.append(f'Held-out items did not confirm the gap in {name} in either preamble condition, so we claim no repair, '
                 'and every arm below is exploratory.')
    others = {}
    for condition in CONDITIONS:
        for lang, cell in results.gaps(condition).items():
            if lang != target and cell.get('confirmed'):
                others.setdefault(lang, []).append(CONDITIONS[condition])
    for lang, conditions in others.items():
        rates = repair_split_rates(results.selection, lang)
        if not rates:
            continue
        intro = (f"{LANGUAGE_NAMES[lang]}, where held-out items did confirm a gap ({' and '.join(conditions)}), "
                 'was not the target:')
        if rates[1] >= rates[0]:
            parts.append(f'{intro} on the repair split Earth was not below Global there ({percent(rates[1])}% against '
                         f'{percent(rates[0])}%).')
        else:
            parts.append(f'{intro} on the repair split Global led Earth there by only '
                         f'{100 * (rates[0] - rates[1]):.1f} points.')
    return {'TARGET_CONFIRMATION_SENTENCE': sentence, 'TARGET_CONFIRMATION_PARAGRAPH': ' '.join(parts)}


def repair_split_rates(selection: dict, lang: str) -> tuple[float, float] | None:
    """Global's and the drifted model's primary safety rate for one language on the repair split, from selection.json."""
    reference, drifted = (dig(selection, key, lang, 'harmful') for key in ('reference_rates', 'drifted_rates'))
    return None if reference is None or drifted is None else (reference, drifted)


def repair_values(results: Results) -> dict:
    values = {}
    target = results.target
    earth_unclear = {lang: unclear_share(results.cell(results.drifted, lang)) for lang in LANGUAGES.values()}
    for key, arm in ARMS.items():
        selected = dig(results.selection, 'arms', arm, 'selected')
        if not selected:
            continue
        tag = selected['tag']
        gate = results.gate(tag)
        split = dig(results.cell(tag, target), 'content_safe') or {}
        rises = [unclear_share(results.cell(tag, lang)) - base for lang, base in earth_unclear.items()
                 if base is not None and unclear_share(results.cell(tag, lang)) is not None]
        l2 = dig(results.localisation, 'edits', arm, 'relative_edit_l2')
        values.update({f'R_{key}_KNOB': str(selected['knob']), f'R_{key}_SAFE': percent(split.get('rate')),
                       f'R_{key}_UNSAFE': percent(split.get('unsafe')), f'R_{key}_INVALID': percent(split.get('invalid')),
                       f'R_{key}_DISPUTED': percent(split.get('disputed')),
                       f'R_{key}_REFUSAL': percent(dig(results.cell(tag, target), 'harmful_refusal', 'rate')),
                       f'R_{key}_OVER': largest(gate, 'over_refusal_increase'), f'R_{key}_CAP': largest(gate, 'capability_drop'),
                       f'R_{key}_UNCLEAR': signed(sum(rises) / len(rises)) if rises else None,
                       f'R_{key}_PASS': gate_word(gate)})
        if arm != 'steer':
            values[f'R_{key}_L2'] = None if l2 is None else f'{l2:.5f}'
    return values


def selected_arm_values(results: Results) -> dict:
    best = results.best
    if not best:
        return {}
    arm, tag = best['arm'], best['tag']
    name, formula, function, donors = ARM_DETAILS[arm]
    spec = yaml.safe_load((ROOT / 'configs' / 'spec.yaml').read_text(encoding='utf-8'))
    knobs = next((row for row in spec['repair_arms'] if row['id'] == arm), {})
    edit = dig(results.localisation, 'edits', arm) or {}
    layers = edit.get('edited_layers')
    gate = results.gate(tag)
    closed = dig(gate, 'harmful_gap_closed', 'value')
    change = dig(results.report, 'paired_comparisons', f'{tag}/{results.condition}', 'vs_drifted', 'safety', results.target,
                 'content_safe')
    values = {'BEST_ARM': name, 'ARM_NAME': name, 'ARM_FORMULA': formula, 'SAFETUNE_FUNCTION': function, 'DONOR_MODELS': donors,
              'KNOB_NAME': knobs.get('knob'), 'KNOB_VALUE': str(best['knob']),
              'KNOB_GRID': ', '.join(str(value) for value in knobs['values']) if knobs else None,
              'PRECISION': 'FP16', 'EDITED_TENSORS': None if edit.get('edited_tensors') is None else str(edit['edited_tensors']),
              'EDITED_LAYERS': None if not layers else ', '.join(str(layer) for layer in layers),
              'RELATIVE_EDIT_L2': None if edit.get('relative_edit_l2') is None else f"{edit['relative_edit_l2']:.5f}",
              'CIRCUIT_SOURCE': None if arm in ('circuit', 'circuit_merge') else 'not used',
              'GAP_CLOSED_PCT': None if closed is None else f'{100 * closed:.1f}',
              'MAX_OVER_REFUSAL_RISE': largest(gate, 'over_refusal_increase'),
              'MAX_CAPABILITY_DROP': largest(gate, 'capability_drop'), 'SAFE_CHANGE': difference(change),
              'PASS_GAP': _yes_no(dig(gate, 'harmful_gap_closed', 'pass')),
              'PASS_OVER': _yes_no(dig(gate, 'over_refusal_increase', 'pass')),
              'PASS_CAP': _yes_no(dig(gate, 'capability_drop', 'pass'))}
    for word, suffix in PRECISIONS.items():
        gate = results.gate(tag + suffix)
        earth = percent(_safe_rate(results, results.drifted + suffix, results.target))
        repaired = percent(_safe_rate(results, tag + suffix, results.target))
        over, cap = largest(gate, 'over_refusal_increase'), largest(gate, 'capability_drop')
        values.update({f'Q_EARTH_{word}_SAFE': earth, f'Q_REP_{word}_SAFE': repaired, f'Q_REP_{word}_OVER': over,
                       f'Q_REP_{word}_CAP': cap, f'{word}_E_SAFE': earth, f'{word}_R_SAFE': repaired, f'{word}_OVER': over,
                       f'{word}_CAP': cap})
        if suffix:
            verdict = gate_word(gate)
            values[f'PASS_{word}'] = verdict
            values[f'{word}_RESULT'] = {'yes': 'held', 'no': 'did not hold'}.get(verdict, verdict)
    return values


def _yes_no(flag: bool | None) -> str | None:
    return None if flag is None else 'yes' if flag else 'no'


def language_values(results: Results) -> dict:
    values = {}
    tag = results.best.get('tag')
    models = (('G', 'global'), ('E', results.drifted), ('R', tag))
    for code, lang in LANGUAGES.items():
        capability_lang = CAPABILITY_LANGUAGE.get(code)
        for letter, model in models:
            cell = results.cell(model, lang)
            refusal, over = cell.get('harmful_refusal', {}), cell.get('over_refusal', {})
            capability = results.capability(model, capability_lang) if capability_lang else {}
            unclear = percent(unclear_share(cell))
            values.update({f'{code}_{letter}_SAFE': rate_with_ci(dig(cell, 'content_safe', 'rate'), dig(cell, 'content_safe', 'ci')),
                           f'{code}_{letter}_REF': rate_with_ci(refusal.get('rate'), refusal.get('ci')),
                           f'{code}_{letter}_OVER': rate_with_ci(over.get('rate'), over.get('ci')),
                           f'{code}_{letter}_UNC': None if unclear is None else f'{unclear}%'})
            if capability_lang:
                values[f'{code}_{letter}_CAP'] = rate_with_ci(capability.get('accuracy'), capability.get('ci95'))
            if letter != 'G':
                values.update({f'L_{code}_SAFE_{letter}': percent(dig(cell, 'content_safe', 'rate')),
                               f'L_{code}_REF_{letter}': percent(refusal.get('rate')),
                               f'L_{code}_OVER_{letter}': percent(over.get('rate'))})
                if capability_lang:
                    values[f'L_{code}_CAP_{letter}'] = percent(capability.get('accuracy'))
        reference = results.cell('global', lang)
        for name, key in (('HARM', 'content_safe'), ('BEN', 'over_refusal')):
            n = dig(reference, key, 'n')
            values[f'{code}_N_{name}'] = None if n is None else str(n)
    return values


def circuit_values(results: Results) -> dict:
    values = {}
    examples, summaries, fair = [], [], []
    for letter, model in (('G', 'global'), ('E', results.drifted)):
        for code in ('EN', 'JV', 'BN'):
            folder = results.run / 'circuits' / f'{model}-circuit__{code.lower()}'
            summary = _read(folder / 'summary.json')
            summaries.append(summary)
            fair.append(_read(folder / 'fair_baseline.json'))
            faithfulness = summary.get('faithfulness', {})
            # The headline ratios are clipped to [0, 1]. The unclipped ones show a circuit that pushes further than the
            # full model (Bengali's ablation ratio is 1.06) instead of a bland 1.00.
            raw = faithfulness.get('metadata') or {}
            for name, key in (('PATCH', 'patching'), ('ABL', 'ablation')):
                score = raw.get(f'{key}_raw_ratio', faithfulness.get(f'{key}_score'))
                values[f'CIRC_{name}_{letter}_{code}'] = None if score is None else f'{score:.2f}'
            baselines = faithfulness.get('baseline_comparison') or {}
            score, best = baselines.get('circuit_score'), baselines.get('best_baseline_score')
            ends = clean_and_corrupt(results.run / 'logs' / f'{model}-circuit-{code.lower()}.log', score)
            # A ratio of raw scores means nothing once they go negative, as they do in Javanese, so the baseline is put
            # on the patching scale: 0 is the harmless twin, 1 the full model.
            values[f'CIRC_BASE_{letter}_{code}'] = (None if ends is None or best is None
                                                   else f'{(best - ends[1]) / (ends[0] - ends[1]):.2f}')
            values[f'CIRC_SIGNAL_{letter}_{code}'] = None if ends is None else f'{ends[0] - ends[1]:.1f}'
            if summary.get('task_examples') is not None:
                examples.append(summary['task_examples'])
    values['CIRC_STABILITY_SENTENCE'] = stability_sentence(summaries)
    values['CIRC_FAIR_BASELINE_SENTENCE'] = fair_baseline_sentence(fair)
    if examples:
        low, high = min(examples), max(examples)
        values['CIRCUIT_N_PAIRS'] = str(low) if low == high else f'{low} to {high}'
    overlap = dig(_read(results.run / 'circuits' / 'overlap.json'), 'jaccard') or {}
    drifted = results.drifted
    pairs = [('en', 'jv'), ('en', 'bn'), ('jv', 'bn')]
    across = [(a, b, dig(overlap, f'global/{a}', f'global/{b}')) for a, b in pairs]
    between = [(lang, dig(overlap, f'global/{lang}', f'{drifted}/{lang}')) for lang in ('en', 'jv', 'bn')]
    if all(score is not None for *_, score in across):
        values['CIRC_JACCARD_LANGS'] = ', '.join(f'{a}–{b} {score:.2f}' for a, b, score in across)
    if all(score is not None for _, score in between):
        values['CIRC_JACCARD_MODELS'] = ', '.join(f'{lang} {score:.2f}' for lang, score in between)
    layers = dig(results.localisation, 'edits', 'circuit', 'layer_subset')
    values['CIRCUIT_LAYERS'] = None if not layers else ', '.join(str(layer) for layer in layers)
    return values


PILLAR_ONE = re.compile(r'Pillar 1 Score .*?raw metric: (-?[\d.]+), clean: (-?[\d.]+), corrupt: (-?[\d.]+)')


def clean_and_corrupt(log: Path, circuit_score: float | None) -> tuple[float, float] | None:
    """The full model's and the harmless twin's mean logit difference, read from the circuit's log.

    CircuitKIT logs them but leaves them out of summary.json. A rerun appends to the log, so the last pillar-1 line wins,
    and its raw metric must match the summary's circuit score, or the line belongs to another run."""
    if circuit_score is None or not log.is_file():
        return None
    found = PILLAR_ONE.findall(log.read_text(encoding='utf-8', errors='replace'))
    if not found:
        return None
    raw, clean, corrupt = map(float, found[-1])
    if abs(raw - circuit_score) > 1e-3 or clean <= corrupt:
        return None
    return clean, corrupt


def fair_baseline_sentence(cells: list[dict]) -> str | None:
    """Random heads with every MLP block kept, from scripts/fair_circuit_baseline.py, once all six circuits have it."""
    if not cells or any(cell.get('mean') is None or cell.get('circuit_patching_ratio') is None for cell in cells):
        return None
    heads = {cell.get('kept_heads') for cell in cells}
    draws = {len(cell.get('patching_ratios') or []) for cell in cells}
    means = [cell['mean'] for cell in cells]
    circuits = [cell['circuit_patching_ratio'] for cell in cells]
    size = f'{heads.pop()} random heads' if len(heads) == 1 else 'random heads of the same number'
    each = f'the mean of {draws.pop()} draws for each circuit' if len(draws) == 1 else 'averaged over several draws'
    sentence = (f'A fairer baseline keeps every MLP block and the input, as the circuits do, and adds {size}. It '
                f'reaches {min(means):.2f} to {max(means):.2f} on the patching scale ({each}), against '
                f'{min(circuits):.2f} to {max(circuits):.2f} for the circuits.')
    gpus = {(cell.get('hardware') or {}).get('gpu') for cell in cells if (cell.get('hardware') or {}).get('device') == 'cuda'}
    if gpus:
        # The circuits were found on a CPU; say where the baseline ran and how closely it reproduced them.
        gaps = [abs(cell['check_ratio'] - cell['circuit_patching_ratio']) for cell in cells
                if cell.get('check_ratio') is not None]
        where = gpus.pop() if len(gpus) == 1 else 'GPUs'
        match = f', and each circuit’s check draw matched the ratio logged on the CPU within {max(gaps):.0e}' if gaps else ''
        sentence += f' The baseline ran in full fp32 on an {where} rather than the CPU{match}.'
    return sentence


def stability_sentence(summaries: list[dict]) -> str | None:
    """The stability pillar reruns discovery with new seeds. When every seed reads the same pairs, a perfect score shows
    only that discovery is deterministic, so the sentence says that rather than calling the circuits robust."""
    cells = [(dig(summary, 'faithfulness', 'stability') or {}, summary.get('settings') or {}) for summary in summaries]
    if not cells or any(stability.get('mean_jaccard') is None for stability, _ in cells):
        return None
    overlaps = [stability['mean_jaccard'] for stability, _ in cells]
    seeds = {stability.get('n_runs') for stability, _ in cells}
    same_pairs = all(settings.get('data_max_records') == settings.get('n_examples') is not None for _, settings in cells)
    runs = f'{seeds.pop()} seeds' if len(seeds) == 1 else 'several seeds'
    if min(overlaps) == 1.0 and same_pairs:
        return (f'Stability, the fourth pillar, rediscovered each circuit with {runs} and got the same nodes every time. '
                'Each seed reads the same pairs, so this shows that discovery is deterministic, not that the circuit '
                'holds on other pairs.')
    caveat = ' Each seed reads the same pairs, so this does not test other pairs.' if same_pairs else ''
    return (f'Stability, the fourth pillar, rediscovered each circuit with {runs}: mean Jaccard overlap between runs '
            f'ranged from {min(overlaps):.2f} to {max(overlaps):.2f}.{caveat}')


def localisation_values(results: Results) -> dict:
    values = {}
    rows = [('C_TWIN', dig(results.localisation, 'whole_model_twin', 'circuit')),
            ('CM_TWIN', dig(results.localisation, 'whole_model_twin', 'circuit_merge'))]
    controls = (results.localisation.get('random_control') or {}).values()
    rows += [(f"RAND{control['seed']}", control) for control in controls]
    for name, block in rows:
        cells = dig(block, 'conditions', results.condition) or {}
        values.update({f'LOC_{name}_SAFE': difference(cells.get('target_content_safe')),
                       f'LOC_{name}_OVER': difference(cells.get('over_refusal')),
                       f'LOC_{name}_CAP': difference(cells.get('capability'))})
    return values


def data_values(results: Results) -> dict:
    manifest = results.manifest
    if not manifest:
        return {}
    counts = manifest.get('counts', {})
    files = manifest.get('files', {})
    translation = manifest.get('translation', {})
    leakage = manifest.get('leakage_check', {})
    values = {f'C_{code}_{word}_{split.upper()}': str(counts[f'{split}/{kind}/{lang}'])
              for code, lang in LANGUAGES.items() for word, kind in (('HARM', 'harmful'), ('BEN', 'benign'))
              for split in ('heldout', 'repair') if f'{split}/{kind}/{lang}' in counts}
    rows = {name: dig(files, name, 'rows') for name in ('heldout', 'repair')}
    if all(rows.values()):
        values.update({'N_PROMPTS_TOTAL': f'{sum(rows.values()):,}', 'N_PROMPTS_HELDOUT': f"{rows['heldout']:,}",
                       'N_PROMPTS_REPAIR': f"{rows['repair']:,}"})
    harmful = [counts[f'heldout/harmful/{lang}'] for lang in LANGUAGES.values() if f'heldout/harmful/{lang}' in counts]
    if harmful:
        values['GAP_PROMPTS_NOW'] = f'{min(harmful)} to {max(harmful)}'
    minhash = [dig(cell, 'stage_counts', 'MinHashDeduplicator') or {} for cell in manifest.get('curatorkit', {}).values()]
    if minhash and all('output_count' in stage for stage in minhash):
        values['MINHASH_REMOVED'] = str(sum(stage['input_count'] - stage['output_count'] for stage in minhash))
    values.update({'TRANSLATION_REQUESTED': _text(translation.get('requested')), 'TRANSLATION_FAILED': _text(translation.get('failed')),
                   'TRANSLATION_QUALITY_STATUS': translation.get('quality_status'),
                   'LEAKAGE_REMOVED_ROWS': _text(leakage.get('removed_rows')),
                   'LEAKAGE_REMOVED_GROUPS': _text(leakage.get('removed_groups')),
                   'CARD_DATE': manifest.get('built_at', '')[:10] or None,
                   'DATA_MANIFEST_SHA256': hashlib.sha256((results.run / 'data' / 'manifest.json').read_bytes()).hexdigest()})
    if translation.get('quality_status') == 'unreviewed':
        values.update({'TRANSLATION_REVIEWED_ROWS': '0', 'TRANSLATION_REJECTED_ROWS': '0', 'TRANSLATION_REVIEWERS': 'none yet'})
    # The packet `scripts/review_data.py export` wrote. One built from other data counts other rows, so it must match.
    packet = _read(results.run / 'review' / 'translation_review.with-source.summary.json')
    if packet.get('rows') and packet.get('manifest_sha256') == values['DATA_MANIFEST_SHA256']:
        values['TRANSLATION_TOTAL_ROWS'] = f"{packet['rows']:,}"
    return values


def fertility_values(results: Results) -> dict:
    """Closed-tokenizer tokens per Tiny Aya token, from `scripts/token_fertility.py`."""
    languages = _read(results.run / 'fertility.json').get('languages', {})
    return {f'FERTILITY_{code}': f"{languages[lang]['closed_per_tiny_aya']:.2f}"
            for code, lang in LANGUAGES.items() if code != 'EN' and lang in languages}


def _text(value) -> str | None:
    return None if value is None else str(value)


def run_values(results: Results) -> dict:
    report = results.report
    judges = JUDGE_NAMES.get(report.get('judge_set'), (None, None))
    gaps = [results.gaps(condition) for condition in CONDITIONS]
    missing = sorted({lang for cells in gaps for lang, cell in cells.items() if cell.get('measurable') is False})
    tested = [LANGUAGE_NAMES[lang] for lang in LANGUAGES.values() if lang not in missing]
    agreement = dig(report, 'models', results.drifted, 'eligibility', 'by_lang') or {}
    scores = [(lang, agreement.get(lang, {}).get('agreement')) for lang in LANGUAGES.values()]
    confirmed = report.get('confirmed_conditions') or []
    frozen = re.search(r'frozen on (\d{1,2} \w+ \d{4})', (ROOT / 'SPEC.md').read_text(encoding='utf-8'))
    values = {'SPEC_DIGEST': report.get('spec_digest'), 'JUDGE_SET_USED': judges[0], 'JUDGE_PAIR': judges[1],
              'SPEC_FREEZE_DATE': frozen.group(1) if frozen and report.get('spec_frozen') else None,
              'LANGUAGES_TESTED': ', '.join(tested) if any(gaps) else None,
              'LANGUAGES_NOT_MEASURABLE': (', '.join(LANGUAGE_NAMES[lang] for lang in missing) or 'none') if any(gaps) else None}
    if all(score is not None for _, score in scores):
        values['AC1_BY_LANGUAGE'] = ', '.join(f'{lang} {score:.2f}' for lang, score in scores)
    if 'empirical_repair_success' in report:
        values['ACCEPTANCE_STATUS'] = ('passed acceptance' if report['empirical_repair_success']
                                       else 'failed acceptance' if confirmed else 'no repair claim')
    return values


def measured(run: Path) -> dict[str, str]:
    """Every value the run's files settle. A value they don't hold is left out, so its placeholder stays."""
    results = Results(run, _read(run / 'report.json'), _read(run / 'selection.json'), _read(run / 'localisation.json'),
                      _read(run / 'data' / 'manifest.json'))
    values = {}
    for part in (gap_values, target_values, target_confirmation_values, repair_values, selected_arm_values,
                 language_values, circuit_values, localisation_values, data_values, fertility_values, run_values):
        values.update(part(results))
    return {name: value for name, value in values.items() if value is not None}


def written(report: dict, path: Path = SENTENCES) -> dict[str, str]:
    """Sentences a person or reader wrote for one run. Another run's numbers differ, so they apply only to that run's spec."""
    data = _read(path)
    return dict(data.get('sentences', {})) if data.get('spec_digest') == report.get('spec_digest') else {}


def kept_blocks(report: dict, selection: dict) -> dict[str, bool]:
    """Which [ALT-*] blocks stay. A block missing here keeps its markers, because the rule can't settle it from the files."""
    conditions = dig(report, 'gap_confirmation', 'conditions')
    if not conditions:
        return {}
    if not any(cell.get('confirmed') for cells in conditions.values() for cell in cells.values()):
        return {'ALT-A': False, 'ALT-A1': False, 'ALT-A2': False, 'ALT-B': True}
    blocks = {'ALT-A': True, 'ALT-B': False}
    if report.get('empirical_repair_success'):
        return {**blocks, 'ALT-A1': True, 'ALT-A2': False}
    tag = dig(selection, 'selected_for_quantization', 'tag')
    gates = [dig(report, 'thresholds', f'{tag}/{condition}') for condition in report.get('confirmed_conditions') or []]
    if any(numbers_pass(gate) for gate in gates):
        return blocks     # every number passed but nothing is validated: a person decides how to word that
    return {**blocks, 'ALT-A1': False, 'ALT-A2': True}


def apply_blocks(text: str, blocks: dict[str, bool]) -> str:
    for name in sorted(blocks, key=len, reverse=True):     # the nested ALT-A1 and ALT-A2 go first
        pattern = rf'\*\*\[{name}\]\*\*\n(.*?)\*\*\[END {name}\]\*\*\n?'
        text = re.sub(pattern, lambda match: match.group(1) if blocks[name] else '', text, flags=re.S)
    return text


def fill(text: str, values: dict[str, str], blocks: dict[str, bool]) -> tuple[str, dict[str, str]]:
    """The filled draft, without its team section, and what is still open, with each item's description from that section."""
    section = TEAM_SECTION.search(text)
    body, team = (text[:section.start()].rstrip() + '\n', text[section.start():]) if section else (text, '')
    described = dict(LIST_ROW.findall(team))
    body = PLACEHOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), apply_blocks(body, blocks))
    return body, {name: described.get(name, '') for name in dict.fromkeys(PLACEHOLDER.findall(body))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    report, selection = _read(args.run / 'report.json'), _read(args.run / 'selection.json')
    values = {**written(report), **measured(args.run)}
    blocks = kept_blocks(report, selection)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = ['# What is still open', '',
             f"Run `{args.run.as_posix()}`, spec `{report.get('spec_digest')}`, frozen: {report.get('spec_frozen')}, "
             f"validated: {report.get('validated')}. Blocks kept: "
             f"{', '.join(name for name, keep in sorted(blocks.items()) if keep) or 'none settled'}.", '']
    for name in DRAFTS:
        text = (ROOT / 'docs' / 'submission' / name).read_text(encoding='utf-8')
        filled, left = fill(text, values, blocks)
        (args.out / name.replace('_template', '')).write_text(filled, encoding='utf-8', newline='\n')
        lines += [f'## {name} ({len(left)} open)', ''] + [f'- `{key}`: {about}'.rstrip(': ') for key, about in left.items()] + ['']
        print(f'{name}: {len(left)} open')
    (args.out / 'values.json').write_text(json.dumps(values, indent=1, ensure_ascii=False, sort_keys=True) + '\n',
                                          encoding='utf-8', newline='\n')
    (args.out / 'remaining.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print(f'{len(values)} values filled into {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
