# Upstream patches for the Lexsi kits

Six patches for the starred items in `docs/UPSTREAM.md`. They are prepared for review only. No
pull request, fork, push or GitHub comment was made; opening them is the team's decision.

| # | Patch | Target repo | Base commit | New tests |
|---|---|---|---|---|
| 1 | `safetune-layer-ranking.patch` | Lexsi-Labs/SafeTune | `3901a9f` | 17 |
| 2 | `auditkit-multilingual-refusal.patch` | Lexsi-Labs/AuditKIT | `71316da` | 67 |
| 3 | `auditkit-benchmark-loaders.patch` | Lexsi-Labs/AuditKIT | `71316da` | 34 |
| 4 | `safetune-multilingual-benchmarks.patch` | Lexsi-Labs/SafeTune | `3901a9f` | 18 |
| 5 | `curatorkit-group-split.patch` | Lexsi-Labs/CuratorKIT | `dd9e835` | 29 |
| 6 | `circuitkit-multilingual-refusal-task.patch` | Lexsi-Labs/CircuitKIT | `e71d5e1` | 43 |

**Nothing was skipped.** On 2026-11-05 all four pinned commits were still the tip of `main`, and no
open pull request covers any item. The nearest, AuditKIT #18 and CircuitKIT #25, touch other code.

Each patch is one commit on `main`. Patches 1 and 4, and patches 2 and 3, apply in either order.
Each was re-applied with `git am` to a clean `main` and matched the commit that was tested.

## Before you submit

- **Set a real author.** The commits use the placeholder `Upstream Contributor
  <contributor@example.invalid>`. After `git am`, run `git commit --amend --reset-author --no-edit`.
- **Have the copyright holders submit.** All four licences say a contribution grants Lexsi Labs a
  perpetual, royalty-free right to use and relicense it under any terms, including commercial ones,
  and that you have the right to make it. These patches come from `langsafe`, which is MIT and
  copyright Aditya Arnav and Pallab Das, so both of them should agree to the grant. It is simplest
  if they submit it themselves, because MIT asks that its notice stays with substantial portions
  of the code. This is a reading of the licence texts, not legal advice.
- **Get the language strings checked.** The Hindi, Hinglish, Bengali, Tamil, Arabic and Swahili
  test sentences in patch 2 were written for this patch and have had no native-speaker review.

### What each repo asks for

| Repo | Licence and where the terms are | Conventions that apply |
|---|---|---|
| SafeTune | LSAL v1.2, section 6. `CONTRIBUTING.md` restates the grant and requires agreeing to it. | Fill in the PR template (pillar, type of change, audit badge). `black`, `isort`, line length 100. Entry under `[Unreleased]`, done. |
| AuditKIT | LSAL v1.2, section **5**. There is no root `CONTRIBUTING.md`; `docs/community/contributing.md` doesn't mention the licence. | All tests must pass (see below). Metrics should subclass `Metric`, so patch 2's `@ak.scorer` may draw a comment, though you asked for that form. |
| CircuitKIT | LSAL v1.2, section 6. `CONTRIBUTING.md` restates it. | One feature per PR. Subject line of 50 characters or fewer, done. New tests go in `tests/unit`: `tests/tasks` is deselected by default, so CI would never run them there. |
| CuratorKIT | LSAL **v1.1**, not v1.2, section 6. `CONTRIBUTING.md` has no licence section. | Conventional Commit titles. `ruff check .` and `pytest tests/unit -q` must pass. Update `configuration.md` and `cli.md`, done. British spelling in docs. |

The patches ship no dataset rows, only loaders. The sources are MIT (MultiJail), CC-BY-4.0 (XSTest)
and Apache-2.0 (Global-MMLU, Aya Red-teaming). The XSTest default in patch 3 is a third-party copy,
`natolambert/xstest-v2-copy`, while SafeTune uses `walledai/XSTest`. A maintainer may prefer one.

## The patches

### 1. `safetune-layer-ranking.patch`

Adds `layer_fraction=` to `circuit_info_from_node_scores`, `load_circuit_info_from_file` and
`get_circuit_info`, plus `layer_mass` and `top_layers`. It ranks layers by total absolute node score.
It is opt-in, so existing results don't change.

The fix had to reach further than the function named in the issue. CircuitKIT 0.2 writes
`layer_suggestions` beside `node_scores`, and those keys win when the file is read, so a real
`*_scores.json` would have kept the unranked selection. With `layer_fraction` set, they are now
re-derived.

**Tests.** `tests/safety/test_circuit_kit.py`, 19 passed (16 of them fail on `main`). The circuit
tests in `test_interop.py`, plus `test_recover_ctheta.py` and `test_safety_lora.py`: 30 passed in all.

**PR title.** `fix(circuit-kit): rank layers by total node score with layer_fraction`

**PR description.**

> `circuit_info_from_node_scores` keeps every layer holding any top-fraction node. Top nodes are
> spread through a model, so a top-20% node set selected 30 of 36 layers on Tiny Aya, and a
> circuit-restricted edit restricted little.
>
> `layer_fraction=` selects the top fraction of layers by total absolute node score instead, and
> `top_fraction` then picks among the nodes in those layers. It is opt-in, so existing results don't
> change. On a file with `node_scores` it replaces the stored `safety_units` and
> `layer_suggestions`, because CircuitKIT 0.2 writes those unranked.
>
> Left for you to decide: whether this should become the default, and whether CircuitKIT's
> `interop_fields` should rank layers too.

### 2. `auditkit-multilingual-refusal.patch`

Adds `src/auditkit/metrics/refusal.py`, with `ak.multilingual_refusal` (an `@ak.scorer`) and
`ak.refusal_scorer` for another name or direction, and registers the `refusal_multilingual` red-team
detector. It covers en, hi, Hinglish, bn, ta, ar and sw. `RefusalDetector` is unchanged.

**Differences from `src/langsafe/judge/rules.py`.** The upstream version fixes real gaps: see
"Found along the way". A language with no patterns is reported as `not_tested` and left out of the
aggregate, because a 0.0 there would read as a failed refusal and inflate attack success.

**Tests.** `tests/test_refusal_multilingual.py`, 67 passed, and `tests/test_redteam.py` passed. No new
failures in the full suite (see "Test failures").

**PR title.** `feat(metrics): add a multilingual refusal scorer and detector`

**PR description.**

> `RefusalDetector` matches eight English regexes, and `can'?t` misses the curly apostrophe. A
> refusal in another language scores as "did not refuse", so refusal rates on multilingual sets are
> wrong.
>
> This adds refusal phrases for English, Hindi, Hinglish, Bengali, Tamil, Arabic and Swahili, as
> `ak.multilingual_refusal` and the `refusal_multilingual` detector. Text is NFC-normalised, curly
> apostrophes fold to straight ones and zero-width joiners are dropped. English is always checked,
> because models often switch to it to refuse. A language with no patterns is `not_tested`, not 0.0.
>
> This is a phrase list: it misses paraphrases and can fire on a reply that only discusses legality.

### 3. `auditkit-benchmark-loaders.patch`

Adds `src/auditkit/benchmark_loaders.py` with `ak.load_multijail`, `ak.load_xstest` and
`ak.load_global_mmlu`. Each returns `Sample(task=lang)`, so `per_task_deltas()` reports per language.
Translations share `metadata["group"]`. Revisions are pinned and recorded on each sample.

**Tests.** `tests/test_benchmark_loaders.py`, 34 passed, offline. No new failures in the full suite.
Also run against the real repos at their pinned revisions: MultiJail 3,150 samples, XSTest 450 (250
safe, 200 unsafe), Global-MMLU-Lite 400 per language, Global-MMLU `sw` dev 285.

**PR title.** `feat(loaders): add MultiJail, XSTest and Global-MMLU loaders`

**PR description.**

> Evaluating safety per language needs samples tagged by language, and there is no loader for the
> multilingual sets people use for it. These three return samples with `task` set to the language,
> so `per_task_deltas()` reports each metric per language. Translations of one item share
> `metadata["group"]`, for splits and paired tests.
>
> Each defaults to a pinned revision and records it, as `load_bfcl` does. `load_xstest` reads only
> the prompt file, because the other five splits of that repo hold model completions. `datasets`
> stays a lazy import under the `interop` extra.

### 4. `safetune-multilingual-benchmarks.patch`

Adds `multijail` and `aya_redteaming` to the dataset table, the `evaluate()` registry and
`load_prompts()`. Each row has a `language` column, and `languages=` picks a subset. This adds data
only: the English refusal lists and the WildGuard judge are unchanged, and the docs warn about that.

**Tests.** `tests/runtime/test_multilingual_benchmarks.py`, 18 passed. With
`test_eval_reliability.py` and `test_runtime_config.py`, 156 passed, 1 skipped and 3 failed; those
3 also fail on `main` here (see "Test failures"). Also run against the real repos: Aya 7,419 rows
across 8 languages, MultiJail 3,150 rows.

**PR title.** `feat(eval): add MultiJail and Aya Red-teaming benchmarks`

**PR description.**

> Every built-in safety benchmark is English, so there is no way to measure whether a repair holds
> in other languages. This adds `multijail` (315 prompts in each of 10 languages, translated) and
> `aya_redteaming` (7,419 prompts written by native speakers of 8 languages).
>
> Aya Red-teaming has one split per language and no `train` split, so its table entry names no
> split and the loader asks for each one.
>
> This adds data only. The refusal lists are English and the default judge has not been validated
> on these languages. A refusal a judge can't read counts as an attack success, so the docs and
> the registry descriptions warn about it. No audit badge was added to `feature-map.md`; that
> verdict is yours.

### 5. `curatorkit-group-split.patch`

Adds `output_split_group` (`CuratorConfig` and YAML) and `curatorkit/splits.py`, with `split_samples`
and `leakage_report`. `Curator` and the CLI each had their own split code, and both now use this one.
Without the option, splitting is unchanged, and a test pins it to the old shuffle.

This covers the split and leakage check only. Language ID, also in that `UPSTREAM.md` entry, wasn't
in this task's list and is not included.

**Tests.** `tests/unit/test_splits.py`, 29 passed. `pytest tests/unit`: 413 passed, 13 skipped, against
384 and 13 on `main`. `ruff check .` and `scripts/check_version_consistency.py` pass.

**PR title.** `feat: add group-aware splits and a leakage check`

**PR description.**

> `output_split` shuffles sample by sample, so the translations of one item can land in both train
> and test, and a test score then measures memorisation.
>
> `output_split_group` names a metadata key whose value keeps a group in one split. A group's split
> is a hash of its id and `output_split_seed`, so an item keeps its split as the corpus grows, at the
> cost of split sizes that only approximate the fractions. `leakage_report` finds groups, exact
> prompts and near-duplicate prompts in more than one split, and a grouped run warns if it finds any.
> It compares prompts, not prompt plus answer, because one prompt with two answers still leaks.
>
> Splitting without the option is unchanged. `Curator` and the CLI now share one implementation.

### 6. `circuitkit-multilingual-refusal-task.patch`

Adds the `multilingual_refusal` task, the first use of `binary_align/jailbreak_binary.csv` by any
task. English pairs ship with the package; other languages take a `pairs_path` CSV, and no
translations are included. The default `chat_template_mode` is `on`, so a base model is an error.

**Tests.** `tests/unit/test_multilingual_refusal.py`, 43 passed. `tests/unit` and
`tests/test_custom_data.py`: 1595 passed against 1552 on `main`, with the same 12 failed and 17 errors.
Also run with Tiny Aya's real tokenizer (tokenizer files only, no weights): 239 of the 334 English
pairs survive token alignment, each prompt has exactly one BOS, and the default openers start on
different tokens in all seven languages.

**PR title.** `Add a multilingual refusal minimal-pair task`

**PR description.**

> There is no cross-language task and no refusal task for chat models. This adds
> `multilingual_refusal`: a harmful and a harmless request that differ in one or two words, scored on
> the logit difference between the first token of a refusal and of a compliance.
>
> English pairs come from `jailbreak_binary.csv`, which no task used. Other languages take a CSV,
> because a pair has to be translated together and reviewed. Prompts go through the chat template,
> with `prepend_bos` handled as the other chat tasks do. Pairs that tokenize to two lengths are
> dropped, and openers sharing a first token are an error. EAP family only.
>
> The template check accepts a dict as well as a string, because `_has_chat_template` accepts only
> a string and Tiny Aya's template is a dict. That helper is unchanged here, since fixing it changes
> `"auto"` for every other task.

## Test failures

No patch adds a failure. These fail the same way on unmodified `main` in this environment:

| Repo | Failures on `main` | Cause |
|---|---|---|
| AuditKIT | 67 failed, 1 collection error, 2682 passed | Mostly optional extras that aren't installed (`requests`, `transformers`) in a core-plus-`dev` venv. Run with the extras on Linux before submitting, since the guide says all tests must pass. |
| AuditKIT | `test_agent_endpoint.py::TestContract::test_default_round_trip` | Flaky: failed 1 of 3 runs on `main`. It is why the failure count moves between 65 and 67. |
| SafeTune | 3 in `test_runtime_config.py` | They reach huggingface.co for a model config, and these runs were offline on purpose. |
| CircuitKIT | 12 failed, 17 errors in `test_graph`, `test_security_hardening`, `test_steering`, `test_structural_pruner` | Present on `main`. Not investigated. |

## Not verified

- **No model was loaded.** `multilingual_refusal` has not been run through EAP-IG. Its data path was
  checked with a real tokenizer, but no circuit has been discovered with it.
- **Non-English pairs** were only tested with fakes, because no translated pairs exist on this machine.
- **The "30 of 36 layers" figure** comes from `UPSTREAM.md`. There is no `*_scores.json` here to
  reproduce it, so the test builds synthetic scores of that shape. Check it against a real file.
- **Refusal phrases** have no measured precision or recall. `rules.py` patterns were carried over
  apart from the fixes noted here.
- **Partial test runs.** SafeTune ran the related files only, not the full suite. CircuitKIT ran
  `tests/unit` and `test_custom_data.py`, and its slow tier was not run.
- **Docs were not built** with MkDocs.

## Found along the way

These are outside the six patches.

1. **`src/langsafe/judge/rules.py` has the gaps patch 2 fixes.** It misses curly apostrophes (`I can’t
   help`), `I'm unable to help`, `I am not able to provide`, and the one-character nukta form of
   `माफ़`. So the rule signal undercounts those refusals. How much that moved any reported number
   has not been measured. The file is unchanged.
2. **CircuitKIT treats Tiny Aya as a base model.** `_chat._has_chat_template` accepts only a string,
   and the `chat_template` of `tiny-aya-global` and `tiny-aya-fire` is a dict, under transformers
   5.22.1, the project's version. So `chat_template_mode="auto"` resolves to False, and `auto` tasks
   such as `mmlu` and `boolq` run without the template. This is probably worth its own patch, but it
   changes results for every `auto` task, so that is the team's call.
3. **CircuitKIT's `interop_fields`** writes the same unranked `layer_subset` that patch 1 works around.
4. **`AuditKIT.RefusalDetector`** still misses curly apostrophes. Patch 2 leaves it alone.
5. **Patch 1 picks `target_modules` differently from `repair/arms.py`.** It uses the top nodes inside
   the selected layers; `arms.py` uses the global top nodes. Adopting it could change a repair run.
