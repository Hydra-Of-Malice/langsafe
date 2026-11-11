# Same model, different language, different safety

**Find, repair and verify per-language safety drift in Tiny Aya Earth, using the Lexsi alignment and safety stack.**
Lexsi Labs AI Safety & Alignment Hackathon, Track 2b (*Repairing safety drift without retraining*).

Cohere's Tiny Aya Earth (Africa, West Asia) and Fire (South Asia) sit close to Tiny Aya Global on MultiJail **on
average**: 3.3 and 1.1 points below it. In their **weakest language** the gap is 9 to 10 points: 87.0% safe for Global,
78.1% for Fire and 77.5% for Earth (Tiny Aya tech report, Table 7). The report doesn't name that language.

Earth is the main drifted model and Global is the reference. Fire is a smaller comparison. Water is a control. So far it has
only a 4-bit pilot, so no Water result is stated here. We test Arabic and Swahili (Earth's region), Javanese, Bengali, Thai and Korean
(where our pilot found Earth weakest against Global), and English as the control. This project:

1. checks the gap per language on held-out prompts at full precision, with content safety (safe, unsafe or invalid, the
   Tiny Aya report's rubric) as the primary axis and refusal reported alongside it;
2. finds where refusal lives with CircuitKIT;
3. repairs it with SafeTune, comparing circuit-restricted edits against whole-model edits and a random-layer control;
4. checks that the repair costs nothing elsewhere and survives 4-bit quantisation.

> **The gap is not confirmed yet.** A pilot pointed to Javanese, but it used 4-bit weights, repair-split items and local
> 8B judges, so it is a planning signal, not a result. There are no held-out results yet. A gap counts as confirmed only
> with full-precision weights on held-out items, after correcting for every language and preamble condition tested
> ([SPEC.md](SPEC.md), Amendments 13 and 17). Without that there is no repair claim, and the repairs are exploratory.

> **Thesis, if the gap is confirmed.** A repair located with CircuitKIT and applied with SafeTune restores per-language
> safety without raising benign over-refusal or hurting capability, and it still holds at 4-bit.

## Start here (team)

| Read | For |
|---|---|
| [docs/PROGRESS.md](docs/PROGRESS.md) | What is running now, who owns what, what needs a person |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Which machine runs what, setup, long jobs on Windows, the day-1 command |
| [docs/submission/checklist.md](docs/submission/checklist.md) | Every submission rule and its status, with the drafts beside it |
| [docs/brief/mentor_brief.pdf](docs/brief/mentor_brief.pdf) | One-page summary we share with mentors |
| [docs/proposal/](docs/proposal/Lexsi_Hackathon_Proposal_Multilingual_Safety_Repair.pdf) | Full 10-page proposal with references, written for the first plan (Fire) |
| [SPEC.md](SPEC.md) | Pre-registered metrics, thresholds and rules (don't change after freeze without an amendment). Its amendments carry the move to Earth |
| [docs/research/](docs/research/README.md) | Hackathon rules, judging, models, kit deep-dives, idea ranking |
| [docs/UPSTREAM.md](docs/UPSTREAM.md) | Gaps in the Lexsi kits and the PRs we plan |

## Pipeline

```
CuratorKIT ─► langsafe data ─► generate ─► 2 LLM judges ─► AuditKIT run ─► CircuitKIT (EAP-IG) ─► SafeTune arms ─► re-test ─► NF4 / GGUF re-test
 clean+dedup   group splits     Global/      Aya Expanse 32B   fingerprint    refusal circuit      ReSta, C-ΔΘ,       all axes   same axes
 provenance    translations     Earth        + Qwen3-32B                      en, jv, bn           merge-back,
                                                                                                   circuit merge,
                                                                                                   steer
```

| Stage | Lexsi kit | langsafe module |
|---|---|---|
| Clean, near-duplicate filter, provenance | CuratorKIT | `data/curate.py` |
| Sources, group-aware split, leakage check, translation | — (missing upstream) | `data/` |
| Generation (preamble on/off, cached, fingerprinted) | — | `gen/` |
| Content and refusal judging, agreement, degenerate answers, hand-check queue | — (AuditKIT's detector is English) | `judge/`, `eval/judge_run.py` |
| Run records, fingerprints | AuditKIT | `eval/auditkit_bridge.py` |
| Statistics (bootstrap CIs, exact McNemar) | — (AuditKIT PR #9 open) | `eval/stats.py`, `eval/axes.py` |
| Refusal circuits, faithfulness, cross-language overlap | CircuitKIT | `circuits/` |
| Repair arms | SafeTune | `repair/arms.py`, `repair/select.py` |
| 4-bit verification | bitsandbytes, llama.cpp | `gen/model.py`, `quant/gguf.py` |
| Tamper-evident run ledger + `lexsi_provenance.json` | — | `provenance.py` |

The full design is in [SPEC.md](SPEC.md) (pre-registered metrics, thresholds and rules). Gaps found in the kits are in
[docs/UPSTREAM.md](docs/UPSTREAM.md).

## Status

The full Earth run has not started, so there are no held-out results. [docs/PROGRESS.md](docs/PROGRESS.md) says what is
running.

The executable experiment connects data preparation, original-model baselines, circuit discovery, all five repair arms
(`resta`, `circuit`, `merge_back`, `circuit_merge`, `steer`) and the random-layer control, selection on repair data,
held-out evaluation, the two local judges, AuditKIT, Q4 export, and reports. Generation and capability support both
preamble conditions. Stages resume only when their inputs, settings, and saved artifacts match. The development profile
uses small real samples and is separate from the canonical spec.

A first development run, on Fire before the move to Earth, completed four arms and nine FP16/Q4 model variants: 702
held-out responses, 1,404 individual judge verdicts, and 1,080 capability decisions. **No repair passed acceptance:**
improvements were mixed, capability drops remained, and judging was uncalibrated. See
[development results](docs/DEVELOPMENT_RESULTS.md) for scores, limits, and local artifacts. Human/translation review,
and mentor comments remain pending.

The Mac inventory and review-sheet paths from that run are in [Mac status](docs/MAC_STATUS.md). The portable CUDA plan is
in [CUDA evaluation](docs/CUDA_EVALUATION.md). Both were written for Fire.

Scientific acceptance is separate from software completion. An unconfirmed gap, unreviewed translations, missing
human/router calibration, missing precision conditions, and incomplete evaluation cells cannot produce a successful
repair claim. The spec was frozen on 7 November, without mentor comments on the thresholds; nobody's labels or mentor approval are
generated automatically.

## Quick start

```bash
uv venv --python 3.12 .venv
LANGSAFE_PYTHON=.venv/bin/python bash scripts/install_kits.sh --repair
.venv/bin/python -m pytest -q
```

**Who runs what, where** (laptop, M1 Max, Kaggle, HF router), plus setup: [docs/RUNBOOK.md](docs/RUNBOOK.md).
Day-1 smoke test on any machine: `JUDGES=local bash scripts/day1.sh`. On Kaggle/Colab, open
[`notebooks/00_day1_smoke_test.ipynb`](notebooks/00_day1_smoke_test.ipynb). Both are wiring checks, and they still run Fire.

Run the complete local integration experiment, then the larger configured experiment:

```bash
bash scripts/run_experiment.sh --family earth --profile development
.venv/bin/python -m langsafe.cli status --directory runs/experiments/development/earth

bash scripts/run_experiment.sh --family earth --profile full
```

Pass `--family earth` every time: the command defaults to Fire. On Windows, run long jobs through `scripts/run_chain.py`
instead ([runbook](docs/RUNBOOK.md)).

Rerun the same command after an interruption. Results, checkpoints, logs, audit sheets, and the hash-verified stage
manifest are isolated under `runs/experiments/<profile>/<family>/`. `report.md` and `report.json` contain aggregate
results and explicit validation gaps. See the [runbook](docs/RUNBOOK.md) for CircuitKIT's separate environment,
llama.cpp, calibration, and the interactive demo. Only one model job should use the accelerator at a time.

This is the hackathon's repair-without-retraining track: the weight arms apply compatible checkpoint deltas or
interpolation, and steering adds an activation direction. No gradient-training claim is made for these methods.

## Responsible release
- Only prompts, labels, metrics, adapters/checkpoints of **repaired** models and logs are published.
- Model outputs on harmful prompts (`runs/gens/`, `runs/generations/`) are git-ignored and never uploaded.
- Tiny Aya is CC-BY-NC 4.0 under the Cohere Labs acceptable-use policy, so all work here is non-commercial.

## Team
Aditya Arnav ([@Hydra-Of-Malice](https://github.com/Hydra-Of-Malice)) · Pallab Das ([@PDGamerSG](https://github.com/PDGamerSG))

## Licence
Code: MIT. Models and data keep their own licences: Tiny Aya CC-BY-NC 4.0, Aya red-teaming Apache-2.0, MultiJail MIT,
XSTest CC-BY-4.0, Global-MMLU-Lite Apache-2.0. The Lexsi kits are under LSAL (research use).
