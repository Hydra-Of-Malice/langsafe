#!/usr/bin/env python3
"""Write an allowlisted public copy of a run folder, then prove it holds no model text.

Usage: .venv/Scripts/python.exe scripts/export_public.py runs/experiments/full/fire exports/fire-public

Model outputs on harmful prompts must never be published (Cohere acceptable use policy; README "Responsible release";
docs/submission/checklist.md). A run folder cannot be uploaded as it stands: AuditKIT saves `raw_output`, judged rows
keep each judge's `reason` and `raw` text, and reports hold local paths and per-item vectors.

Two independent defences:

1. Allowlist. Only the files in ROUTES are read for export, and only the fields their schemas name are copied. Every
   other file, field and row is left out and listed in EXPORT_MANIFEST.json with the reason.
2. Final scan. The written output is re-read from disk and compared with the private generations, whatever the first
   defence did. Any match, forbidden key or local path blocks the export: the staging folder is removed, OUT_DIR is
   never created, and the exit code is non-zero.

The scan compares text. It cannot recognise a paraphrase or an encoded copy of an answer; keeping free text out in the
first place is the allowlist's job. Answers under MIN_SUBSTRING letters and digits match only as a whole field, since
"No." and "Sure." occur everywhere. Published prompts get a longer fragment length than other fields, because models
echo prompts and harmful prompts share stock phrases. Nothing here prints model text, and the source run is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

from langsafe.judge.llm_judge import LABELS, SAFETY_LABELS
from langsafe.provenance import Ledger

MANIFEST_NAME = 'EXPORT_MANIFEST.json'
MANIFEST_SCHEMA = 'langsafe.public-export/1'

# Scan sizes, counted in normalised characters (letters and digits only).
MIN_SUBSTRING = 12      # below this a response only matches a whole field
WINDOW = 24             # fragment fingerprint length
STRIDE = 4              # a shared run of WINDOW + STRIDE - 1 = 27 characters is always caught
PROMPT_WINDOW = 48      # the same for verified prompt text, where 51 characters are always caught
MAX_LOCATED = 100       # matches traced back to a location; the rest are still counted as violations
MAX_NESTED_JSON = 6
MAX_SMALL_LIST = 4      # a confidence interval fits; a per-item vector does not

# What docs/submission/dataset_card_template.md promises.
CARD_LICENCES = {'CohereLabs/aya_redteaming': 'Apache-2.0', 'DAMO-NLP-SG/MultiJail': 'MIT',
                 'natolambert/xstest-v2-copy': 'CC-BY-4.0'}
CARD_COMPILATION_LICENCE = 'CC-BY-NC-4.0'
CARD_LANGUAGES = frozenset({'en', 'bn', 'ar', 'sw', 'jv', 'th', 'ko', 'hi', 'ta', 'hi-Latn'})    # Earth's, then Fire's
CARD_ORIGINS = frozenset({'native', 'translated', 'code-mixed'})
KINDS = frozenset({'harmful', 'benign'})
SPLITS = frozenset({'heldout', 'repair'})

_NOT_LETTER_OR_DIGIT = re.compile(r'[\W_]+')
_HEX = re.compile(r'[0-9a-f]{8,128}')
_NAME = re.compile(r'[A-Za-z0-9][\w.:+\-/#@]{0,199}')
_ARG = re.compile(r'[\w.:+\-/#@,=<>]{1,300}')
_LABEL = re.compile(r'[A-Za-z0-9_.:+\-/#@\[\]<>*]{1,160}')
_TIMESTAMP = re.compile(r'\d{4}-\d{2}-\d{2}T[\d:.]+(?:Z|[+-]\d{2}:\d{2})?')
_ABSOLUTE = re.compile(r'[A-Za-z]:/|/|~/|file:')
_RELATIVE_PATH = re.compile(r'[\w.+\-=#@:]+(?:/[\w.+\-=#@:]+)*')
_LOCAL_PATH = re.compile(
    r'(?<![\w/])[A-Za-z]:[\\/]'
    r'|\\\\[\w.$-]+\\'
    r'|(?<![\w.:/-])/(?:home|Users|root|mnt|media|tmp|var|opt|srv|data|workspace|scratch|private|Volumes|content)/'
    r'|file:/|(?<!\w)~[\\/]|(?i:appdata)[\\/]')
_FORBIDDEN_IN_TEXT = re.compile(
    r'''\\*["'](?:response|raw|raw_output|completion|reason|actual_output|model_output)\\*["']\s*:''', re.I)
_FORBIDDEN_WORDS = frozenset({'response', 'responses', 'raw', 'completion', 'completions', 'reason', 'reasons',
                              'rationale', 'explanation', 'answer', 'answers', 'prediction', 'predictions'})
_FORBIDDEN_KEYS = frozenset({'output', 'actual_output', 'model_output', 'generated_text', 'generation', 'generations',
                             'verdicts', 'per_item', 'host', 'hostname'})
_PROMPT_KEYS = frozenset({'prompt', 'prompts', 'text', 'source_text', 'translation_text'})
_HASH_SUFFIXES = frozenset({'sha256', 'hash', 'digest'})

_DROP = object()
ANY, PATH, ARG, NUMBERS, SCALARS = object(), object(), object(), object(), object()


class ExportError(Exception):
    """The request itself must be refused before any export work starts."""


def file_sha256(path: Path) -> str:
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def normalise(text: str) -> str:
    """Fold away case, compatibility forms, spacing, punctuation, zero-width characters and combining marks."""
    return _NOT_LETTER_OR_DIGIT.sub('', unicodedata.normalize('NFKC', text).casefold())


def safe_label(text: str) -> str:
    """A printable name for `text`, or a short hash when it could itself be private."""
    if _LABEL.fullmatch(text):
        return text
    return '<sha256:' + hashlib.sha256(text.encode('utf-8', 'surrogatepass')).hexdigest()[:12] + '>'


def is_forbidden_key(key: str, value=None) -> bool:
    """True for keys that carry model text. `response_sha256` is allowed only while it really holds a hash."""
    words = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', key).casefold()
    tokens = [token for token in re.split(r'[^a-z0-9]+', words) if token]
    if '_'.join(tokens) not in _FORBIDDEN_KEYS and not _FORBIDDEN_WORDS.intersection(tokens):
        return False
    return not (tokens[-1] in _HASH_SUFFIXES and isinstance(value, str) and bool(_HEX.fullmatch(value)))


def public_path(value, run_root: str) -> str | None:
    """A run-relative path. Runs built elsewhere (Mac, CUDA box) keep only what follows the run folder."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace('\\', '/').rstrip('/')
    if _ABSOLUTE.match(text):
        root = run_root.rstrip('/')
        if text.casefold().startswith(root.casefold() + '/'):
            text = text[len(root) + 1:]
        else:
            parts = [part for part in text.split('/') if part]
            lowered = [part.casefold() for part in parts]
            tail: list[str] = []
            if 'experiments' in lowered:       # runs/experiments/<profile>/<family>/...
                tail = parts[len(lowered) - lowered[::-1].index('experiments') + 2:]
            elif 'runs' in lowered:
                tail = parts[len(lowered) - lowered[::-1].index('runs'):]
            text = '/'.join(tail) or f'external/{parts[-1] if parts else "unnamed"}'
    return text if _RELATIVE_PATH.fullmatch(text) and '..' not in text.split('/') else None


# ---- leaf rules: each returns the clean value, or _DROP ---------------------------------------------------------

def _number(value):
    if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)):
        return value
    return _DROP


def _flag(value):
    return value if value is None or isinstance(value, bool) else _DROP


def _number_or_flag(value):
    return value if isinstance(value, bool) else _number(value)


def _hash(value):
    return value if value in (None, '') or (isinstance(value, str) and _HEX.fullmatch(value)) else _DROP


def _is_name(value) -> bool:
    return isinstance(value, str) and bool(_NAME.fullmatch(value)) and not _LOCAL_PATH.search(value)


def _name(value):
    return value if value is None or _is_name(value) else _DROP


def _scalar(value):
    clean = _number_or_flag(value)
    return _name(value) if clean is _DROP else clean


def _timestamp(value):
    return value if value is None or (isinstance(value, str) and _TIMESTAMP.fullmatch(value)) else _DROP


def _source_text(value):
    """A short one-line field copied from a source dataset, such as a harm category."""
    if value is None or (isinstance(value, str) and len(value) <= 200 and value.isprintable()
                         and not _LOCAL_PATH.search(value)):
        return value
    return _DROP


def _base_url(value):
    public = isinstance(value, str) and re.fullmatch(r'https://[a-z0-9.\-]+\.[a-z]+(?:/[\w.\-]*)*', value)
    return value if value in (None, 'local') or public else _DROP


def _never(value):
    """A field that is known and deliberately not published, so the manifest can name it."""
    return _DROP


def _one_of(*allowed) -> Callable:
    choices = frozenset(allowed)
    return lambda value: value if value is None or (isinstance(value, str) and value in choices) else _DROP


def _template(pattern: str) -> Callable:
    compiled = re.compile(pattern)
    return lambda value: value if isinstance(value, str) and compiled.fullmatch(value) else _DROP


def _repo_reference(value):
    """`data.sources` entries are a repo id or {repo, revision}."""
    if isinstance(value, dict):
        return {key: value[key] for key in ('repo', 'revision') if _is_name(value.get(key))} or _DROP
    return _name(value)


@dataclass(frozen=True)
class Keyed:
    """A mapping whose keys are local paths: keys are made public and values follow `values`."""
    values: object


# ---- schemas: what each file may contain ------------------------------------------------------------------------

# Fixed wording from langsafe.experiment_report and eval.report. If it changes, the text is dropped and counted.
_LIMITS = ('Human calibration and language-quality review require actual reviewed labels.',
           'Local judges additionally require shared-sample router calibration.',
           'The router comparison covers refusal labels only, so it does not check content-safe numbers.',
           'Rules are development signals, not safety ground truth.',
           'NF4 requires a CUDA run; Q4_K_M is the locally available Mac condition.')
_ELIGIBILITY_GAPS = frozenset({
    'rule-only or unknown judge set is not reportable', 'exactly two independent judges are required',
    'missing or invalid inter-judge agreement', 'inter-judge agreement below the required minimum',
    'missing or incomplete verified human calibration',
    'missing or incomplete verified human calibration of content labels',
    'human calibration belongs to a different evaluation',
    'missing or insufficient shared-sample router calibration',
    'router calibration belongs to a different local evaluation'})
_validation_gap = _template(
    r'[\w.\-/]+: (?:missing judgments|duplicate evaluation identities|missing [\w\-]+/[\w\-]+/\w+'
    r'|capability run is incomplete|incomplete capability evidence for [\w\-]+'
    r'|evaluated prompt/condition set differs from reference|capability inputs differ from reference for [\w\-]+'
    r'|missing same-precision reference/drifted baselines'
    r'|required reference, drifted, and repaired precision runs are missing)'
    r'|No repair-split selection of the final weight candidate was recorded')

CHECKPOINT = {'model': PATH, 'revision': _name, 'files': {ANY: _hash}}
CAPABILITY_CELL = {'data_sha256': _hash, ANY: NUMBERS}
CAPABILITY_BY_LANG = {ANY: CAPABILITY_CELL}
ELIGIBILITY = {'status': _name, 'judge_set': _name, 'statistic': _name, 'by_lang': {ANY: {'status': _name, ANY: NUMBERS}}, ANY: NUMBERS}
CALIBRATION = {'source': _never, 'source_sha256': _hash, 'evaluation_digest': _hash, ANY: NUMBERS}
ROUTER_CALIBRATION = {'local_evaluation_digest': _hash, 'router_evaluation_digest': _hash, ANY: NUMBERS}
COVERAGE = {'missing': [_name], 'missing_split_cells': [_name], ANY: NUMBERS}
DESCRIPTOR = {'arm': _name, 'knob': _number, 'tag': _name, 'model': PATH, 'steering': PATH}
SELECTION = {
    'split': _name, 'preamble': _name, 'scorer': _one_of('two-judge consensus'), 'judges': _name,
    'capability_split': _name, 'worst_language': _name, 'reference_rates': NUMBERS, 'drifted_rates': NUMBERS,
    'drifted_capability': NUMBERS, 'omitted': [{'arm': _name}], 'selected_for_quantization': DESCRIPTOR,
    'arms': {ANY: {'knob': _number, 'candidates': NUMBERS, 'selected': DESCRIPTOR}},
    'controls': [{**DESCRIPTOR, 'random_seed': _number}]}
GATE = {'status': _name, 'harmful_gap_closed': {'lang': _name, 'status': _name, ANY: NUMBERS},
        'coverage': {'missing': [_name], 'required_languages': [_name], 'required_capability_languages': [_name],
                     ANY: NUMBERS},
        'eligibility': {'detail': {'eligible': _flag}, ANY: NUMBERS}, ANY: NUMBERS}
MODEL = {'status': _name, 'eligibility': ELIGIBILITY, 'conditions': NUMBERS, 'capability': CAPABILITY_BY_LANG,
         'capability_by_condition': {ANY: CAPABILITY_BY_LANG}, 'quality_by_lang': NUMBERS,
         'hand_check_sampling_by_lang': NUMBERS, 'generation_validity': NUMBERS, 'human_calibration': CALIBRATION,
         'router_calibration': ROUTER_CALIBRATION}
REPORT = {
    'family': _name, 'profile': _name, 'spec_digest': _hash, 'spec_frozen': _flag, 'judge_set': _name,
    'validated': _flag, 'empirical_repair_success': _flag, 'models': {ANY: MODEL}, 'thresholds': {ANY: GATE},
    'gap_confirmation': {'metric': _name, 'tests': _number, 'alpha': _number, 'conditions': {ANY: NUMBERS}},
    'comparison_models': {ANY: {'agreement_by_lang': NUMBERS, 'conditions': NUMBERS}},
    'paired_comparisons': {ANY: {ANY: {'baseline': _name, 'safety': NUMBERS, 'capability': NUMBERS,
                                       'pairing_complete': _flag}}},
    'acceptance_candidate': _name, 'confirmed_conditions': [_name], 'required_acceptance_comparisons': [_name],
    'selection': SELECTION,
    'validation_gaps': [_validation_gap], 'data_coverage': COVERAGE, 'translation_quality_status': _name,
    'artifacts_sha256': Keyed(_hash), 'limits': [_one_of(*_LIMITS)]}
STATUS = {
    'software_run_complete': _flag, 'profile': _name, 'family': _name, 'evaluated_tags': [_name],
    'scientific_validation': _one_of('provisional pending real human/router calibration'),
    'all_repair_arms_evaluated': _flag, 'empirical_repair_success': _flag, 'validation_gaps': [_validation_gap],
    'spec_frozen': _flag, 'nf4_cuda': _name, 'q4_evaluated': _flag}
CAPABILITY = {
    'languages': CAPABILITY_BY_LANG, 'complete': _flag,
    'metadata': {'model': PATH, 'quant': _name, 'adapter': PATH, 'steering': PATH, 'strength': _number, 'gguf': PATH,
                 'languages': [_name], 'limit': _number, 'preamble': _name, 'mmlu_split': _name, 'spec': _hash,
                 'checkpoint_identity': CHECKPOINT, 'gguf_sha256': _hash, 'data_sha256': {ANY: _hash},
                 'revision': _name,
                 'variant': {'checkpoint_identity': CHECKPOINT, 'steering_sha256': _hash, 'strength': _number}}}
JUDGE_SUMMARY = {
    'judges': [lambda value: value if value == 'rules (provisional)' else _name(value)], 'judge_set': _name,
    'judge_models': {ANY: {'model': PATH, 'base_url': _base_url, 'quant': _name, 'dtype': _name, 'gguf': PATH,
                           'checkpoint_identity': _name}},
    'rows': _number, 'hand_check_queue': _number, 'hand_check_sampled': _number,
    'hand_check_sampling_by_lang': NUMBERS, 'audit_sample_size': _number, 'audit_size': _number,
    'audit_seed': _number, 'disagreement_cap': _number, 'kappa_by_lang': NUMBERS, 'safety_kappa_by_lang': NUMBERS,
    'safety_ac1_by_lang': NUMBERS, 'quality_by_lang': NUMBERS, 'evaluation_digest': _hash, 'eligibility': ELIGIBILITY}
DATA_MANIFEST = {
    'built_at': _timestamp, 'spec_digest': _hash, 'seed': _number, 'family': _name, 'languages': [_name],
    'translated': _flag, 'translator': _name,
    'translation': {'status': _name, 'quality_status': _name, 'model': _name, 'quant': _name, 'failure_counts': NUMBERS,
                    ANY: NUMBERS},
    'translation_failure_count': _number,
    'curatorkit': {ANY: {'in': _number, 'kept': _number, 'stage_counts': NUMBERS}},
    'files': {ANY: {'path': PATH, 'rows': _number, 'sha256': _hash}}, 'counts': NUMBERS, 'coverage': COVERAGE,
    'discarded': NUMBERS, 'validation': NUMBERS, 'leakage_check': NUMBERS, 'sources': {ANY: _repo_reference},
    'source_provenance': {ANY: {'repo': _name, 'revision': _name, 'files': {ANY: {'sha256': _hash, 'bytes': _number}},
                                'discarded': NUMBERS, 'normalized_rows': _number, 'normalized_sha256': _hash}}}
REVIEW_SUMMARY = {
    'review_version': _name, 'manifest_sha256': _hash, 'data_files': {ANY: _hash}, 'source_reference_sha256': _hash,
    'packet_id': _hash, 'quality_status': _name, ANY: NUMBERS}
EXPERIMENT = {
    'identity': SCALARS,
    'stages': {ANY: {'settings': {'args': [ARG], 'inputs_sha256': Keyed(_hash), 'optional_inputs_sha256': Keyed(_hash),
                                  'models': Keyed(CHECKPOINT), 'spec': _hash, 'conversion_protocol': _name},
                     'artifacts': [{'path': PATH, 'sha256': _hash}], 'result': {'elapsed_seconds': _number}}}}
LEXSI_PROVENANCE = {
    'schema': _name, 'library': _name, 'version': _name, 'git_sha': _name, 'created_at': _timestamp,
    'base_model': PATH, 'method': _name, 'inputs': [{'kind': _name, 'ref': PATH, 'config': _name}], 'params': SCALARS}
# `category` is left out: it is upstream metadata anyone can get from the source datasets, and in the full Earth run
# some category labels share fragments with answers the models gave, which the final scan rightly blocks.
PROMPT_ROW = {
    'id': _name, 'group': _name, 'lang': _name, 'kind': _name, 'text': lambda value: value, 'source': _name,
    'source_id': _source_text, 'origin': _name, 'split': _name,
    'licence': _source_text, 'license': _source_text,
    'meta': {'global_or_local': _source_text, 'upstream_source': _source_text, 'source_revision': _hash,
             'source_sha256': _hash, 'translated_from': _name, 'translator': _name, 'translation_quality': _name,
             'licence': _source_text, 'license': _source_text}}
_verdict = _one_of(*LABELS)
VERDICT_ROW = {
    'key': _hash, 'id': _name, 'group': _name, 'lang': _name, 'kind': _name, 'preamble': _name, 'model_tag': _name,
    'fingerprint': _name, 'split': _name, 'response_sha256': _hash, 'prompt_sha256': _hash, 'rubric_hash': _hash,
    'label': _verdict, 'judges': {ANY: _verdict}, 'judge_set': _name, 'safety_votes': {ANY: _one_of(*SAFETY_LABELS)},
    'safety_label': _one_of(*SAFETY_LABELS, 'disputed'), 'safety_parse_error': _flag, 'rule_refusal': _flag, 'degenerate': _flag,
    'hand_label': _one_of(*LABELS, ''), 'hand_safety_label': _one_of(*SAFETY_LABELS, ''),
    'needs_hand_check': _flag, 'disagreement': _flag, 'parse_error': _flag, 'audit_size': _number,
    'audit_seed': _number, 'disagreement_cap': _number}
LEDGER_ENTRY = {'seq': _number, 'prev_hash': _hash, 'step': _name, 'at': _timestamp, 'git': _name, 'hash': _hash}
_ARTIFACT_STEP = {'tag': _name, 'spec': _hash, 'artifact': PATH, 'sha256': _hash}
LEDGER_PAYLOADS = {
    'build-data': {'manifest': DATA_MANIFEST},
    'generate': {**_ARTIFACT_STEP, 'model': PATH, 'quant': _name, 'adapter': PATH, 'gguf': PATH, 'steering': PATH,
                 'strength': _number, 'fingerprint': _name, 'split': _name, 'n': _number, 'artifact_sha256': _hash,
                 'decoding': SCALARS},
    'judge': {**JUDGE_SUMMARY, 'tag': _name, 'split': _name, 'spec': _hash},
    'calibrate': _ARTIFACT_STEP,
    'auditkit': {**_ARTIFACT_STEP, 'judge_set': _name},
    'capability': {**_ARTIFACT_STEP, 'model': PATH},
    'convert': {'model': PATH, 'checkpoint_identity': CHECKPOINT, 'sha256': _hash, 'precision': _name, 'spec': _hash},
    'quantize': {'sha256': _hash, 'quant': _name, 'spec': _hash},
    'circuit-discover': {'model': PATH, 'lang': _name},
    'repair': {'method': _name, 'out': PATH, 'inputs': [PATH], 'params': SCALARS},
    'steer-extract': {'family': _name, 'layer': _number, 'n_harmful': _number, 'n_harmless': _number,
                      'preamble': _name, 'model': PATH, 'safetune_commit': _name, 'prompt_sha256': _hash}}

# report.md is free text, so each line must match what experiment_report.build_report writes.
_MD_LINES = frozenset({
    '', '**PROVISIONAL: human/router calibration or other scientific validation is incomplete.**',
    '**Judging calibration verified for the included evaluations.**',
    'A completed software run does not imply a successful safety repair. Missing language cells, unclear answers, '
    'judge disagreement, and capability changes are retained.',
    '## Is the gap confirmed?', '## Why answers to harmful prompts were not safe',
    '## Comparison models (context only)',
    'Other regional models on the same held-out items. They were never repaired or selected. These gaps sit outside the '
    'tests above, so none of them counts as confirmed.',
    '| Model | Condition | Language | Safety rate | Degenerate | Reference − model (95% CI) | McNemar p | n | Judge agreement |',
    '| Condition | Language | Reference − drifted (95% CI) | McNemar p | n | Confirmed |', '|---|---|---|---|---|---|',
    'Unsafe content and a model breaking down are different failures. Invalid answers are garbled, off-topic, or '
    'caught by the degeneracy rule; disputed means the judges disagreed.',
    '| Model | Language | Unsafe | Invalid | Disputed |', '|---|---|---|---|---|',
    '## Degenerate answers, harmful and benign prompts (lower is better)',
    'Empty answers, repetition loops and bare echoes of the prompt.',
    '## Content-safe rate on harmful prompts (higher is safer)', '## Harmful full-refusal rate',
    '## Harmful full-compliance rate (lower is safer)', '## Benign over-refusal rate', '## Capability accuracy',
    '## Paired changes versus the original regional model', '## Measurement quality', '## Human hand check',
    'People label a seeded audit plus a capped sample of disagreements, so these rows are a sample of the queue. '
    'Every disagreement, sampled or not, stays `unclear` or `disputed` and never counts as safe. Accuracy comes from '
    'the random audit, because sampled disagreements over-represent hard rows.',
    '## Repair acceptance',
    'The target language and repair parameters are selected on the repair split. Held-out outcomes cannot be used to '
    'retune them. The JSON companion contains coverage, per-language partial/unclear rates, verdict quality, '
    'validation gates, and provenance hashes.',
    '## Remaining validation', *(f'- {limit}' for limit in _LIMITS)})
_MD_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r'\d+ tests \(languages x preamble conditions\), so a gap counts as confirmed at McNemar p < [\d.]+ with a paired 95% CI '
    r'above zero\. Repairs in a condition without a confirmed gap are exploratory\.',
    r'\| [\w\-]+ \| [\w\-]+ \| [+\-][\d.]+ pp \([+\-][\d.]+–[+\-][\d.]+\) \| [\d.]+ \| \d+ \| (?:yes|no) \|',
    r'\| [\w\-]+ \[\w+\] \| [\w\-]+ \| [\d.]+% \| [\d.]+% \| [\d.]+% \|',
    r'# Tiny Aya [\w\-]+ repair experiment',
    r'Profile: `[\w.\-]+`\. Judges: `[\w.\-]+`\. Spec: `[0-9a-f]+`\. Spec frozen: `(?:True|False)`\.',
    r'\|(?: ?[\w .%+\-–()=/\[\]]{0,80} ?\|)+',
    r'- `[\w.\-]+`: \w+; eligible=(?:True|False)\.',
    r'Empirical repair success: \*\*(?:True|False)\*\*\.'))
_MD_LANGUAGE_LINE = re.compile(r'  - [\w\-]+: \w+; (?:kappa|gwet_ac1)=[\w.\-]+; ?(?P<gaps>.*)')


def _md_line_allowed(line: str) -> bool:
    if line in _MD_LINES or any(pattern.fullmatch(line) for pattern in _MD_PATTERNS):
        return True
    match = _MD_LANGUAGE_LINE.fullmatch(line)
    return bool(match) and all(gap in _ELIGIBILITY_GAPS for gap in filter(None, match['gaps'].split('; ')))


def _reduce_data_manifest(document: dict) -> dict:
    """Per-row translation failures become the counts the card asks for."""
    document = dict(document)
    translation = dict(document.get('translation') or {})
    failures = translation.pop('failures', None) or []
    translation['failure_counts'] = dict(Counter(str(item.get('reason')) for item in failures
                                                 if isinstance(item, dict)))
    document['translation'] = translation
    document['translation_failure_count'] = len(document.pop('translation_failures', None) or [])
    return document


def _reduce_review_summary(document: dict) -> dict:
    """The card publishes how many reviewers there were, never who they were."""
    reviewers = document.get('reviewer_ids') or []
    public = {key: value for key, value in document.items() if key != 'reviewer_ids'}
    return {**public, 'reviewer_count': len(reviewers)}


_TAG = r'[A-Za-z0-9][A-Za-z0-9._\-]*'
ROUTES = tuple((re.compile(pattern), target, kind, schema, prepare) for pattern, target, kind, schema, prepare in (
    (r'report\.json', 'reports/report.json', 'json', REPORT, None),
    (r'report\.md', 'reports/report.md', 'markdown', None, None),
    (r'selection\.json', 'reports/selection.json', 'json', SELECTION, None),
    (r'status\.json', 'reports/status.json', 'json', STATUS, None),
    (rf'capability/({_TAG})(?<!\.unscorable)\.json', r'reports/capability/\1.json', 'json', CAPABILITY, None),
    (rf'judged/({_TAG})\.summary\.json', r'reports/judged/\1.summary.json', 'json', JUDGE_SUMMARY, None),
    (rf'judged/({_TAG})\.jsonl', r'verdicts/\1.jsonl', 'verdicts', None, None),
    (r'data/manifest\.json', 'data/manifest.json', 'json', DATA_MANIFEST, _reduce_data_manifest),
    (r'data/(heldout|repair)\.jsonl', r'data/\1.jsonl', 'prompts', None, None),
    (r'review/translation_review\.checked\.json', 'data/translation_review.summary.json', 'json', REVIEW_SUMMARY,
     _reduce_review_summary),
    (r'ledger\.jsonl', 'provenance/ledger.jsonl', 'ledger', None, None),
    (r'experiment\.json', 'provenance/experiment.json', 'json', EXPERIMENT, None),
    (rf'auditkit/({_TAG})/lexsi_provenance\.json', r'provenance/auditkit/\1.lexsi_provenance.json', 'json',
     LEXSI_PROVENANCE, None)))

REFUSALS = tuple((re.compile(pattern), why) for pattern, why in (
    (r'gens/|generations/', 'raw model outputs, including answers to harmful prompts'),
    (r'auditkit/', 'AuditKIT run files save every prompt and raw_output; the checklist says never upload them'),
    (r'hand_check/|review/', 'human review sheets contain model text'),
    (r'calibration/', 'kept private by the checklist; its aggregates are in reports/report.json'),
    (r'capability/.*\.unscorable\.json$', 'diagnostic dump from a failed capability run'),
    (r'logs/|.*\.log$', 'free-text logs can quote prompts or outputs; provenance/ledger.jsonl is the public log'),
    (r'circuits/|circuit_tasks/', 'circuit detail is a dual-use decision the team has not made yet'),
    (r'checkpoints/|quantized/|.*\.(?:safetensors|gguf|pt)$', 'weights are released separately, after re-evaluation'),
    (r'curatorkit/', 'intermediate prompt copies; data/ holds the released rows'),
    (r'.*\.sqlite', 'the LLM cache holds model and judge text'),
    (r'.*\.csv$', 'review sheets and task tables can hold model text')))
UNLISTED = 'not on the allowlist'


def _why_refused(relative: str) -> str:
    return next((why for pattern, why in REFUSALS if pattern.match(relative)), UNLISTED)


def _walk(run: Path) -> Iterator[tuple[str, bool]]:
    """Every (relative path, is_link) in the run. Links are never followed."""
    for folder, directories, files in os.walk(run):
        for name in sorted(directories):
            if os.path.islink(os.path.join(folder, name)):
                yield Path(folder, name).relative_to(run).as_posix(), True
        directories[:] = sorted(name for name in directories if not os.path.islink(os.path.join(folder, name)))
        for name in sorted(files):
            yield Path(folder, name).relative_to(run).as_posix(), os.path.islink(os.path.join(folder, name))


@dataclass
class PromptIndex:
    """Verified prompt text. Models echo prompts, so fragment matching must not fire on the prompts themselves."""
    hashes: set[int] = field(default_factory=set)
    by_group: dict[str, list[str]] = field(default_factory=dict)

    def add(self, row: dict) -> None:
        text = normalise(row['text'])
        self.hashes.add(hash(text))
        self.by_group.setdefault(row['group'], []).append(text)

    def is_echo(self, row: dict, text: str) -> bool:
        """True when `text` sits inside a prompt from the group that produced `row`."""
        group = row.get('group') or str(row.get('id', '')).split('::')[0]
        return any(text in prompt for prompt in self.by_group.get(group, ()))


class _Export:
    """One export: copies allowlisted content into `staging` and remembers everything it left out."""

    def __init__(self, run: Path, staging: Path):
        self.run, self.staging = run, staging
        self.run_root = str(run).replace('\\', '/')
        self.exported: list[dict] = []
        self.refused: list[dict] = []
        self.failures: list[dict] = []
        self.dropped: Counter = Counter()
        self.refused_rows: Counter = Counter()
        self.integrity: dict = {}
        self.prompts = PromptIndex()
        self.current = ''
        self._data_manifest: dict | None = None

    # -- schema application --

    def _drop(self, where: str):
        self.dropped[self.current, where] += 1
        return _DROP

    def _tree(self, kind, value, where: str):
        if isinstance(value, dict):
            return self._apply({ANY: kind}, value, where)
        if isinstance(value, list):
            if len(value) > MAX_SMALL_LIST:
                return self._drop(where)
            items = (self._tree(kind, item, where + '[]') for item in value)
            return [item for item in items if item is not _DROP]
        clean = (_number_or_flag if kind is NUMBERS else _scalar)(value)
        return self._drop(where) if clean is _DROP else clean

    def _apply(self, schema, value, where: str):
        if value is None and isinstance(schema, (dict, list, Keyed)):
            return None     # an absent object, such as a model with no calibration yet
        if schema is NUMBERS or schema is SCALARS:
            return self._tree(schema, value, where)
        if schema is PATH:
            clean = None if value is None else public_path(value, self.run_root)
            return self._drop(where) if clean is None and value is not None else clean
        if schema is ARG:
            if not isinstance(value, str):
                return self._drop(where)
            if _ABSOLUTE.match(value.replace('\\', '/')):
                return public_path(value, self.run_root) or '<redacted>'
            return value if _ARG.fullmatch(value) else '<redacted>'
        if isinstance(schema, Keyed):
            if not isinstance(value, dict):
                return self._drop(where)
            clean = {}
            for key, item in value.items():
                name = public_path(key, self.run_root)
                if name is None:
                    self._drop(where + '.<path>')
                    continue
                result = self._apply(schema.values, item, where + '.*')
                if result is not _DROP:
                    clean[name] = result
            return clean
        if isinstance(schema, dict):
            if not isinstance(value, dict):
                return self._drop(where)
            clean = {}
            for key, item in value.items():
                rule, label = schema.get(key), key
                if rule is None and ANY in schema and _is_name(key) and not is_forbidden_key(key):
                    rule, label = schema[ANY], '*'
                if rule is None:
                    self._drop(f'{where}.{safe_label(str(key))}')
                    continue
                result = self._apply(rule, item, f'{where}.{label}')
                if result is not _DROP:
                    clean[key] = result
            return clean
        if isinstance(schema, list):
            if not isinstance(value, list):
                return self._drop(where)
            items = (self._apply(schema[0], item, where + '[]') for item in value)
            return [item for item in items if item is not _DROP]
        clean = schema(value)
        return self._drop(where) if clean is _DROP else clean

    # -- bookkeeping and writing --

    def refuse(self, relative: str, why: str) -> None:
        self.refused.append({'file': safe_label(relative), 'why': why})

    def _fail(self, check: str, relative: str, detail: str) -> None:
        self.failures.append({'check': check, 'file': safe_label(relative), 'detail': detail})
        self.refuse(relative, detail)

    def _output(self, target: str) -> Path:
        path = self.staging / target
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _record(self, relative: str, target: str, rows: int | None = None) -> None:
        path = self.staging / target
        entry = {'file': target, 'from': relative, 'sha256': file_sha256(path),
                 'source_sha256': file_sha256(self.run / relative), 'bytes': path.stat().st_size}
        self.exported.append(entry if rows is None else {**entry, 'rows': rows})

    def _export_json(self, relative: str, target: str, schema, prepare) -> None:
        document = json.loads((self.run / relative).read_text(encoding='utf-8-sig'))
        if prepare and isinstance(document, dict):
            document = prepare(document)
        clean = self._apply(schema, document, 'root')
        if clean is _DROP or clean is None:
            self.refuse(relative, 'not the expected document shape')
            return
        with self._output(target).open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(clean, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        self._record(relative, target)

    def _export_jsonl(self, relative: str, target: str, clean_row: Callable[[object, str], dict | None]) -> None:
        rows = 0
        with (self.run / relative).open(encoding='utf-8-sig') as source, \
                self._output(target).open('w', encoding='utf-8', newline='\n') as sink:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    self.refused_rows[relative, 'line is not valid JSON'] += 1
                    continue
                clean = clean_row(row, relative)
                if clean is not None:
                    sink.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + '\n')
                    rows += 1
        self._record(relative, target, rows)

    def _export_markdown(self, relative: str, target: str) -> None:
        lines = (self.run / relative).read_text(encoding='utf-8-sig').splitlines()
        kept = [line for line in lines if _md_line_allowed(line)]
        if len(kept) != len(lines):
            self.dropped[relative, 'unrecognised lines'] += len(lines) - len(kept)
        with self._output(target).open('w', encoding='utf-8', newline='\n') as stream:
            stream.write('\n'.join(kept) + '\n')
        self._record(relative, target)

    # -- row cleaners --

    def _clean_verdict(self, row, relative: str) -> dict | None:
        """Labels only. `verdicts` holds each judge's reason and raw text, so it is never in the schema."""
        if not isinstance(row, dict) or row.get('label') not in LABELS or row.get('kind') not in KINDS \
                or not _is_name(row.get('id')):
            self.refused_rows[relative, 'not a recognised verdict row'] += 1
            return None
        return self._apply(VERDICT_ROW, row, 'row')

    def _clean_prompt(self, row, relative: str, split: str) -> dict | None:
        why = _prompt_problem(row, split)
        if why:
            self.refused_rows[relative, why] += 1
            return None
        clean = self._apply(PROMPT_ROW, row, 'row')
        self.prompts.add(clean)
        return clean

    def _clean_ledger_entry(self, entry, relative: str) -> dict | None:
        if not isinstance(entry, dict):
            self.refused_rows[relative, 'not a ledger entry'] += 1
            return None
        payload = entry.get('payload') if isinstance(entry.get('payload'), dict) else {}
        step = entry.get('step')
        public = payload
        if step == 'build-data' and isinstance(payload.get('manifest'), dict):
            public = {**payload, 'manifest': _reduce_data_manifest(payload['manifest'])}
        clean = self._apply(LEDGER_ENTRY, {key: value for key, value in entry.items() if key != 'payload'}, 'entry')
        clean['payload_sha256'] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8', 'surrogatepass')).hexdigest()
        schema = LEDGER_PAYLOADS.get(str(step), {})
        clean['payload'] = self._apply(schema, public, f'payload[{safe_label(str(step))}]')
        return clean

    # -- routed files --

    def _manifest_hash(self, split: str) -> str | None:
        if self._data_manifest is None:
            path = self.run / 'data' / 'manifest.json'
            self._data_manifest = json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}
        entry = (self._data_manifest.get('files') or {}).get(split) or {}
        return entry.get('sha256')

    def _export_prompts(self, relative: str, target: str, split: str) -> None:
        expected = self._manifest_hash(split)
        if not expected:
            self.refuse(relative, 'data/manifest.json records no hash to verify this prompt file against')
            return
        if file_sha256(self.run / relative) != expected:
            self._fail('data_integrity', relative, 'prompt file does not match the hash in data/manifest.json')
            return
        self.integrity[f'data/{split}.jsonl'] = 'matches data/manifest.json'
        self._export_jsonl(relative, target, lambda row, name: self._clean_prompt(row, name, split))

    def _export_ledger(self, relative: str, target: str) -> None:
        try:
            ok, bad = Ledger(self.run / relative).verify()
        except (ValueError, KeyError, TypeError, AttributeError):
            ok, bad = False, None
        if not ok:
            self._fail('ledger_chain', relative, f'ledger chain does not verify (first bad entry: {bad})')
            return
        self.integrity['ledger'] = ('chain verified before redaction; published entries omit the host and most of the '
                                    'payload, so `hash` links entries but cannot be recomputed from this file')
        self._export_jsonl(relative, target, self._clean_ledger_entry)

    def _handle(self, match: re.Match, target_template: str, kind: str, schema, prepare, relative: str) -> None:
        target = match.expand(target_template)
        self.current = relative
        if kind == 'json':
            self._export_json(relative, target, schema, prepare)
        elif kind == 'markdown':
            self._export_markdown(relative, target)
        elif kind == 'verdicts':
            self._export_jsonl(relative, target, self._clean_verdict)
        elif kind == 'prompts':
            self._export_prompts(relative, target, match[1])
        else:
            self._export_ledger(relative, target)

    def run_all(self) -> None:
        for relative, is_link in _walk(self.run):
            if is_link:
                self.refuse(relative, 'symbolic link')
                continue
            route = next(((pattern.fullmatch(relative), *rest) for pattern, *rest in ROUTES
                          if pattern.fullmatch(relative)), None)
            if route is None:
                self.refuse(relative, _why_refused(relative))
                continue
            try:
                self._handle(*route, relative)
            except ValueError:      # includes JSON and Unicode decoding errors
                self.refuse(relative, 'could not be parsed, so it cannot be checked')
                for entry in [e for e in self.exported if e['from'] == relative]:
                    self.exported.remove(entry)
                target = self.staging / route[0].expand(route[1])
                target.unlink(missing_ok=True)

    def manifest(self, run_name: str) -> dict:
        return {
            'schema': MANIFEST_SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
            'run': safe_label(run_name), 'exporter_sha256': file_sha256(Path(__file__)),
            'policy': 'Allowlist. Only files and fields named in scripts/export_public.py are copied. '
                      'Everything else is left out and listed here.',
            'licence': {'compilation': CARD_COMPILATION_LICENCE, 'by_source': CARD_LICENCES},
            'exported': self.exported, 'refused': self.refused,
            'refused_rows': [{'file': safe_label(name), 'why': why, 'count': count}
                             for (name, why), count in sorted(self.refused_rows.items())],
            'dropped_fields': [{'file': safe_label(name), 'field': where, 'count': count}
                               for (name, where), count in sorted(self.dropped.items())],
            'integrity': self.integrity}


def _prompt_problem(row, split: str) -> str | None:
    """Why a row cannot be released under the card, or None. The card's licence table is keyed by `source`."""
    if not isinstance(row, dict):
        return 'not a prompt row'
    if not isinstance(row.get('text'), str) or not row['text'].strip():
        return 'empty prompt text'
    if row.get('source') not in CARD_LICENCES:
        return 'source is not in the dataset card, so its licence is unknown'
    if row.get('lang') not in CARD_LANGUAGES:
        return 'language is not in the dataset card'
    if row.get('kind') not in KINDS or row.get('origin') not in CARD_ORIGINS or row.get('split') != split:
        return 'kind, origin or split is not one the card defines'
    if not _is_name(row.get('group')) or row.get('id') != f"{row['group']}::{row['lang']}":
        return 'id is not <group>::<lang>'
    return None


# ---- final scan -------------------------------------------------------------------------------------------------

class _Pairs(list):
    """A JSON object kept as (key, value) pairs, so a repeated key cannot hide a value from the scan."""


@dataclass(frozen=True)
class _Leaf:
    file: str
    where: str
    text: str
    is_key: bool = False
    value: object = None


def _value_leaves(relative: str, value, where: str, depth: int = 0) -> Iterator[_Leaf]:
    if isinstance(value, _Pairs):
        for key, item in value:
            label = f'{where}.{safe_label(key)}'
            yield _Leaf(relative, f'{where}.<key>', key, True, item)
            yield from _value_leaves(relative, item, label, depth)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _value_leaves(relative, item, f'{where}[{index}]', depth)
    elif isinstance(value, str):
        yield _Leaf(relative, where, value)
        if depth < MAX_NESTED_JSON and value.lstrip()[:1] in ('{', '[', '"'):
            try:
                inner = json.loads(value, object_pairs_hook=_Pairs)
            except ValueError:
                return
            yield from _value_leaves(relative, inner, where + '<json>', depth + 1)


def _unescape(text: str) -> str:
    """Undo JSON string escaping in text that is not parsed as JSON, so an escaped copy still matches."""
    text = re.sub(r'\\u([0-9a-fA-F]{4})', lambda match: chr(int(match[1], 16)), text)
    return re.sub(r'\\[nrt"/\\]', ' ', text)


def _file_leaves(path: Path, relative: str, violations: list[dict]) -> Iterator[_Leaf]:
    suffix = path.suffix.lower()
    if suffix not in ('.json', '.jsonl', '.md'):
        violations.append({'check': 'unexpected_file_type', 'file': safe_label(relative)})
    elif suffix != '.md':
        try:
            if suffix == '.json':
                document = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_Pairs)
                yield from _value_leaves(relative, document, 'root')
            else:
                with path.open(encoding='utf-8') as stream:
                    for number, line in enumerate(stream, start=1):
                        if line.strip():
                            row = json.loads(line, object_pairs_hook=_Pairs)
                            yield from _value_leaves(relative, row, f'line {number}')
            return
        except ValueError:
            violations.append({'check': 'unreadable', 'file': safe_label(relative)})
    yield _Leaf(relative, '<text>', _unescape(path.read_text(encoding='utf-8', errors='replace')))


def _output_leaves(out_dir: Path, violations: list[dict]) -> Iterator[_Leaf]:
    for path in sorted(item for item in out_dir.rglob('*') if item.is_file()):
        relative = path.relative_to(out_dir).as_posix()
        yield _Leaf(relative, '<file name>', relative)
        yield from _file_leaves(path, relative, violations)


def _is_prompt_field(leaf: _Leaf) -> bool:
    return leaf.file.startswith('data/') and leaf.file.endswith('.jsonl') and leaf.where.endswith('.text')


def _check_key(leaf: _Leaf, violations: list[dict]) -> None:
    key = leaf.text
    forbidden = is_forbidden_key(key, leaf.value)
    if key.casefold() in _PROMPT_KEYS and not (key == 'text' and leaf.file.startswith('data/')):
        forbidden = True
    if forbidden:
        violations.append({'check': 'forbidden_key', 'file': safe_label(leaf.file), 'where': leaf.where,
                           'key': safe_label(key)})


@dataclass
class _Hit:
    fragment: str
    text: str
    normalised: str
    source: str
    line: int
    row_id: str
    in_prompt: bool


@dataclass
class ScanReport:
    violations: list[dict] = field(default_factory=list)
    private_files: list[str] = field(default_factory=list)
    rows_compared: int = 0
    distinct_texts: int = 0
    empty_rows: int = 0
    short_rows: int = 0
    prompt_echoes_allowed: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> dict:
        return {'result': 'passed' if self.ok else 'failed', 'private_files': self.private_files,
                'rows_compared': self.rows_compared, 'distinct_texts': self.distinct_texts,
                'empty_rows': self.empty_rows, 'rows_too_short_for_substring_match': self.short_rows,
                'prompt_echoes_allowed': self.prompt_echoes_allowed, 'minimum_substring_characters': MIN_SUBSTRING,
                'fragment_characters_always_caught': WINDOW + STRIDE - 1,
                'prompt_fragment_characters_always_caught': PROMPT_WINDOW + STRIDE - 1}


class _UnreadablePrivateRow(Exception):
    """A private row that cannot be read cannot be compared, so the output cannot be proved clean."""

    def __init__(self, source: str, line: int):
        super().__init__(source, line)
        self.source, self.line = source, line


def _private_rows(paths: Iterable[Path]) -> Iterator[tuple[str, int, dict]]:
    """Stream private rows one at a time, so memory stays flat however large the generations are."""
    for path in paths:
        label, number = f'{path.parent.name}/{path.name}', 0
        try:
            with path.open(encoding='utf-8-sig') as stream:
                for number, line in enumerate(stream, start=1):
                    if line.strip():
                        row = json.loads(line)
                        if not isinstance(row, dict) or not isinstance(row.get('response'), str):
                            raise _UnreadablePrivateRow(label, number)
                        yield label, number, row
        except ValueError:      # bad JSON, or bytes that are not UTF-8
            raise _UnreadablePrivateRow(label, number + 1) from None


def _window_at(text: str, windows: set[int]) -> int | None:
    for start in range(0, len(text) - WINDOW + 1, STRIDE):
        if hash(text[start:start + WINDOW]) in windows:
            return start
    return None


def _foreign_prompt_window(text: str, row: dict, prompts: PromptIndex, windows: set[int]) -> str | None:
    """A passage of a response found in a prompt it was not answering. Its own prompts are an echo, not a leak."""
    for start in range(0, len(text) - PROMPT_WINDOW + 1, STRIDE):
        window = text[start:start + PROMPT_WINDOW]
        if hash(window) in windows and not prompts.is_echo(row, window):
            return window
    return None


def _locate(hits: list[_Hit], out_dir: Path, prompts: PromptIndex) -> list[dict]:
    """Turn matches into locations. Only paths and row ids leave this function, never the matched text."""
    found: list[dict] = []
    if not hits:
        return found
    remaining = {id(hit): 5 for hit in hits}
    for leaf in _output_leaves(out_dir, []):
        text = normalise(leaf.text)
        trusted = _is_prompt_field(leaf) and hash(text) in prompts.hashes
        for hit in hits:
            matched = hit.fragment == text if len(hit.fragment) < MIN_SUBSTRING else hit.fragment in text
            if matched and remaining[id(hit)] and hit.in_prompt == trusted:
                remaining[id(hit)] -= 1
                match = 'exact' if hit.text in leaf.text else 'normalised' if hit.normalised in text else 'fragment'
                found.append({'check': 'response_text', 'match': match, 'file': safe_label(leaf.file),
                              'where': leaf.where, 'private_source': safe_label(hit.source),
                              'private_line': hit.line, 'row_id': safe_label(hit.row_id)})
    return found


def scan_output(out_dir: Path, private_files: list[Path], prompts: PromptIndex | None = None, *,
                allow_prompt_echo: bool = False) -> ScanReport:
    """Re-read everything under `out_dir` and fail on response text, forbidden keys or local paths.

    Response text is matched whole (exact and normalised) in every string, key and file name. Long responses are also
    matched by fragment, so a quoted passage is caught too.

    `prompts` must come from the verified source data, never from the output being scanned. Models echo prompts, so
    verified prompt text is held to a narrower test: a passage of a response is fine in the prompts of its own group,
    and fails from PROMPT_WINDOW characters in any other prompt. A whole response inside a prompt always fails, unless
    `allow_prompt_echo` is set and the prompt belongs to the response's own group.
    """
    prompts = prompts or PromptIndex()
    report = ScanReport(private_files=[f'{path.parent.name}/{path.name}' for path in private_files])
    strict: set[str] = set()
    whole_fields: set[str] = set()
    prompt_texts: set[str] = set()
    seen: set[str] = set()
    for leaf in _output_leaves(out_dir, report.violations):
        if leaf.is_key:
            _check_key(leaf, report.violations)
        if leaf.text in seen or _HEX.fullmatch(leaf.text):
            continue
        seen.add(leaf.text)
        text = normalise(leaf.text)
        if _is_prompt_field(leaf) and hash(text) in prompts.hashes:
            prompt_texts.add(text)
            continue
        if _LOCAL_PATH.search(leaf.text) or _FORBIDDEN_IN_TEXT.search(leaf.text):
            check = 'local_path' if _LOCAL_PATH.search(leaf.text) else 'forbidden_key'
            report.violations.append({'check': check, 'file': safe_label(leaf.file), 'where': leaf.where})
        strict.add(text)
        if ' ' in leaf.text or not leaf.text.isascii():     # identifiers like `complete` are not text
            whole_fields.add(text)
    del seen
    windows = {hash(text[start:start + WINDOW]) for text in strict for start in range(len(text) - WINDOW + 1)}
    strict_blob, prompt_blob = '\0'.join(strict), '\0'.join(prompt_texts)
    longest_prompt = max(map(len, prompt_texts), default=0)
    prompt_windows = {hash(text[start:start + PROMPT_WINDOW]) for text in prompt_texts
                      for start in range(len(text) - PROMPT_WINDOW + 1)}
    del strict, prompt_texts

    hits: list[_Hit] = []
    total_hits = 0
    compared: set[int] = set()
    try:
        for source, line, row in _private_rows(private_files):
            report.rows_compared += 1
            text = normalise(row['response'])
            if not text:
                report.empty_rows += 1
                continue
            key = hash(text)
            if key in compared:
                continue
            compared.add(key)
            fragment = None
            if len(text) < MIN_SUBSTRING:
                report.short_rows += 1
                fragment = text if text in whole_fields else None
            elif len(text) < WINDOW:
                fragment = text if text in strict_blob else None
            else:
                start = _window_at(text, windows)
                fragment = None if start is None else text[start:start + WINDOW]
            in_prompt = False
            if fragment is None and len(text) >= MIN_SUBSTRING:
                if len(text) <= longest_prompt and text in prompt_blob:
                    if allow_prompt_echo and prompts.is_echo(row, text):
                        report.prompt_echoes_allowed += 1
                    else:
                        fragment = text
                else:
                    fragment = _foreign_prompt_window(text, row, prompts, prompt_windows)
                in_prompt = fragment is not None
            if fragment is not None:
                total_hits += 1
                if len(hits) < MAX_LOCATED:
                    hits.append(_Hit(fragment, row['response'], text, source, line, str(row.get('id', '')),
                                     in_prompt))
    except _UnreadablePrivateRow as exc:
        report.violations.append({'check': 'private_row_unreadable', 'private_source': safe_label(exc.source),
                                  'private_line': exc.line})
    report.distinct_texts = len(compared)
    located = _locate(hits, out_dir, prompts)
    report.violations.extend(located)
    if total_hits > len(hits):
        report.violations.append({'check': 'response_text', 'match': 'not located',
                                  'count': total_hits - len(hits)})
    return report


# ---- entry point ------------------------------------------------------------------------------------------------

def private_generation_files(run: Path, gens_dir: Path | None = None) -> list[Path]:
    folders = [run / 'gens', run / 'generations', *([gens_dir] if gens_dir else [])]
    return sorted(path for folder in folders if folder.is_dir() for path in folder.glob('*.jsonl'))


def _check_request(run: Path, out: Path) -> None:
    if not run.is_dir():
        raise ExportError('RUN_DIR is not a folder')
    if out.exists():
        raise ExportError('OUT_DIR already exists; choose a new folder')
    if out.is_relative_to(run) or run.is_relative_to(out):
        raise ExportError('OUT_DIR must be outside RUN_DIR, because the source run is never modified')


def _report_failure(out: Path, failures: list[dict], violations: list[dict]) -> None:
    problems = [*failures, *violations]
    details = out.with_name(out.name + '.scan-failed.json')
    details.write_text(json.dumps({'status': 'failed', 'problem_count': len(problems), 'problems': problems},
                                  ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'EXPORT BLOCKED: {len(problems)} problem(s). Nothing was published and the staging folder was removed. '
          'The source run was not changed.', file=sys.stderr)
    for problem in problems[:20]:
        print('  ' + ' '.join(f'{key}={value}' for key, value in problem.items()), file=sys.stderr)
    print(f'Locations only, no text: {details}', file=sys.stderr)


def export_run(run_dir: Path, out_dir: Path, *, gens_dir: Path | None = None, allow_prompt_echo: bool = False) -> int:
    """Export `run_dir` to `out_dir`. OUT_DIR appears only after the scan passes. Returns the process exit code."""
    run, out = Path(run_dir).resolve(), Path(out_dir).resolve()
    _check_request(run, out)
    private = private_generation_files(run, Path(gens_dir).resolve() if gens_dir else None)
    if not private:
        raise ExportError('no private generations found (gens/*.jsonl), so the output cannot be proved clean; '
                          'pass --gens-dir if they live elsewhere')
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(f'.{out.name}.staging-{os.getpid()}')
    staging.mkdir()
    try:
        job = _Export(run, staging)
        job.run_all()
        manifest = job.manifest(run.name)
        draft = staging / MANIFEST_NAME
        draft.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        report = scan_output(staging, private, job.prompts, allow_prompt_echo=allow_prompt_echo)
        if job.failures or not report.ok:
            _report_failure(out, job.failures, report.violations)
            return 1
        # The draft manifest was scanned above; this adds only the scan's own counts.
        draft.write_text(json.dumps({**manifest, 'scan': report.summary()}, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
        os.replace(staging, out)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    out.with_name(out.name + '.scan-failed.json').unlink(missing_ok=True)
    print(f'Exported {len(job.exported)} files; refused {len(job.refused)} files; '
          f'{sum(job.refused_rows.values())} rows and {sum(job.dropped.values())} fields left out.')
    print(f'Scan passed: {report.rows_compared} private rows from {len(private)} files. See {out / MANIFEST_NAME}')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Write a sanitised, allowlisted public copy of a run folder.')
    parser.add_argument('run_dir', type=Path, help='run folder, for example runs/experiments/full/fire (only read)')
    parser.add_argument('out_dir', type=Path, help='new folder to create, outside the run folder')
    parser.add_argument('--gens-dir', type=Path, help='extra folder of private generation JSONL files to scan against')
    parser.add_argument('--allow-prompt-echo', action='store_true',
                        help='let a response that only repeats a prompt from its own group pass; the count is recorded')
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):      # a Windows console may not encode every character in a path
            stream.reconfigure(errors='backslashreplace')
    try:
        return export_run(args.run_dir, args.out_dir, gens_dir=args.gens_dir, allow_prompt_echo=args.allow_prompt_echo)
    except ExportError as exc:
        print(f'EXPORT REFUSED: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        # Only the type: an exception's message can quote the data that caused it.
        print(f'EXPORT FAILED: {type(exc).__name__}. Nothing was published.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
