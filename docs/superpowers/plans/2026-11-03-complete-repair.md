# End-to-end Tiny Aya repair implementation plan

> **For agentic workers:** Use subagent-driven development for independent data, judging, and toolkit integrations. The controller owns CLI orchestration and all accelerator runs. Never run model jobs concurrently.

**Goal:** Make the existing Tiny Aya Global/Fire safety-repair experiment executable, exercise it with real checkpoints, and deliver reproducible results with explicit validation gates.

**Architecture:** Retain `langsafe`'s data, generation, judging, circuit, repair, statistics, and provenance modules. Add the missing CLI and orchestration seams rather than a new application. A development profile exercises the full path before the full configured run. Keep local-judge results provisional until required calibration and human review exist.

**Tech stack:** Python 3.12, PyTorch MPS FP16 on the 64 GiB M1 Max, Hugging Face originals, the four pinned Lexsi toolkits, llama.cpp Q4_K_M on Metal. CUDA NF4 is available as a portable command, with its execution status stated accurately on this Mac.

**Spec:** `SPEC.md`, `configs/spec.yaml`, and the user's approved choice of Tiny Aya Global as reference and Fire as repair target. Existing benchmark evidence is in the workspace's `research/` directory.

## Global constraints

- Work on `feat/end-to-end-repair`; preserve the existing user checkout and cached models.
- Stay within the approved Global/Fire experiment. Earth remains runnable via its existing family setting.
- Keep source group IDs intact across translations and splits. Never tune on held-out data.
- Output validity, formatting, capability, harmful refusal, and benign over-refusal are separate measures.
- An unrelated or unintelligible answer is not a successful refusal. Judge parsing must fail closed.
- Missing language cells, missing capability, low judge agreement, or absent calibration cannot pass success thresholds.
- Do not fabricate human labels, mentor approval, CUDA/NF4 results, or a successful repair.
- Harmful completions, judge raw text, calibration sheets, caches, and weight files remain ignored and local.
- Local free inference only. No cloud spend, external publication, messaging, or remote pushes.
- Preserve original FP16 results; compressed results have distinct model identities and provenance.

### Task 1: Data preparation and translation

**Owner/files:** data worker; `src/langsafe/data/{build,sources,translate,splits}.py`, `tests/test_data_pipeline.py`.

**Consumes:** `load_spec()` and `Prompt`; `make_client()` implementing `chat`, `chat_many`, and `close`.
**Produces:** backward-compatible `build(...) -> dict` extended with `family: str | None = None`; `load_split(split) -> list[Prompt]`; reproducible manifests with source revisions/hashes, coverage, translations, and leakage counts. Honor `LANGSAFE_DATA_DIR` when supplied.

- [x] Demonstrate regressions for data limiting before expensive translation, group leakage removal across translations, and rejected/failed translation handling with offline fakes.
- [x] Build only selected-family source languages when `family` is given. Cap candidate data before translation and curation; batch/cache translations without changing the split-group identity.
- [x] Read an explicit `translation` config with the Cohere Tiny Aya Global local translator, avoiding unsupported Expanse Bengali/Tamil as an unexamined default. Record translator identity and flag unreviewed translation quality.
- [x] Pin or record actual dataset revisions. Validate unique IDs and nonempty text, and record discarded rows.
- [x] Run `python -m pytest -q tests/test_data_pipeline.py tests/test_core.py` and report exact changes and remaining external dependencies.

### Task 2: Trustworthy judging and gates

**Owner/files:** judging worker; `src/langsafe/{llm_client.py,judge/llm_judge.py,eval/judge_run.py,eval/report.py}`, `tests/test_judging_pipeline.py`.

**Consumes:** generation rows, `Prompt` lookup, judge configs, and capability tables.
**Produces:** backward-compatible judging/report interfaces; per-judge raw verdicts and counts in private artifacts; random human audit plus disagreement queues; explicit report eligibility and fail-closed threshold decisions.

- [x] Demonstrate that unrelated output is not defined as refusal, ambiguous bare labels do not parse as the first match, and empty/missing evaluation cells cannot pass thresholds.
- [x] Release local MPS model/cache resources on close; close SQLite/client resources; keep cache hits model-free. Cache identity must include backend/precision and checkpoint identity where available.
- [x] Export the configured random hand-check sample as well as disagreements, with stable unique keys and no duplicate rows.
- [x] Preserve `unclear`, partial-refusal, parse-error and judge disagreement rates. Retain per-language kappa and mark uncalibrated results as provisional.
- [x] Implement a calibration-file import/summary path callable from CLI; only actual supplied human labels count as human evidence.
- [x] Run focused judging regression tests and original CPU tests. Do not load models or change CLI/config files.

### Task 3: Model-tool and quantization integrations

**Owner/files:** toolkit worker; `scripts/install_kits.sh`, `pyproject.toml`, `src/langsafe/{circuits/discover.py,circuits/task.py,repair/arms.py,quant/gguf.py}`, `tests/test_toolkit_integration.py`.

**Consumes:** pinned toolkits; existing model loaders and runtime paths.
**Produces:** working Mac install path, complete model-stage adapters, saved repair provenance, strict GGUF identity/scoring, and explicit unsupported-device errors.

- [x] Verify pinned upstream function signatures and dependencies against fetched source before changing adapters.
- [x] Install model dependencies without requiring CUDA-only bitsandbytes on Darwin. Preserve the current functioning PyTorch/Transformers runtime where supported.
- [x] Ensure circuit outputs resolve paths correctly and contain usable selected layers; validate shape compatibility and use explicit repair knobs.
- [x] Stream GGUF hashing instead of reading multi-gigabyte files into memory. Exact A/B/C/D candidate scoring must not silently treat missing top-40 letters as wrong; expose unscorable items explicitly.
- [x] Add conversion/quantization helpers using subprocess argument lists, without starting browser servers. Inference server lifecycle belongs to the controller.
- [x] Run focused mocked integration tests; do not run GPUs, install packages, download model weights, edit CLI/config, or commit shared files.

### Task 4: CLI, experiment lifecycle, and real runs

**Owner/files:** controller; `src/langsafe/{cli.py,config.py,gen/generate.py,eval/capability.py,experiment.py}`, `scripts/run_experiment.sh`, `tests/test_experiment.py`, experiment configs/docs.

**Consumes:** task 1–3 interfaces, original local checkpoints, optional Base access.
**Produces:** commands for development/held-out generation, capability, circuit discovery, repair creation/selection, calibration import, quantization checks, status/resume, demo, and final report.

- [x] Add `--spec PATH` and a small separately identified development profile. Do not silently modify the canonical thresholds or freeze the spec as mentor-approved.
- [x] Add split-aware generation with batch-size and output-limit controls, native preamble conditions, resumable cache writes, progress, generation-token/time metadata, and error-safe cleanup.
- [x] Add selected-language, resumable capability scoring with final-token projection, per-item outcomes, intervals, and input hashes. Retain the existing benchmark scoring definition.
- [x] Connect circuit task creation, discovery, all four repair arms, repair-split knob selection, and selected-model held-out evaluation. Never select a knob or target language from held-out outcomes.
- [x] Add a resumable experiment manifest that records each stage's inputs/settings/artifact paths and skips only verified completed work.
- [x] Run the real limited-data end-to-end experiment, reload repaired checkpoints, and exercise Q4_K_M conversion/evaluation.
- [ ] Run the full configured experiment when storage permits. Its checkpoints/exports need roughly 75 GiB alone; the completed development run leaves less than that available. Human/router calibration, translation review, mentor freeze, and CUDA NF4 are separate outstanding validation work.
- [x] Deliver an interactive local inference/demo command for reference, original Fire, and selected repaired model. Keep raw harmful responses out of public reports.

### Task 5: Review, proof, and handoff

**Owner/files:** controller and final reviewer; README/runbook, machine-readable status, aggregate results, visual report.

- [x] Run `python -m pytest -q`, `python -m langsafe.cli --help`, applicable CLI integration commands, and `python -m langsafe.cli ledger-verify` against final code.
- [x] Review the diff and fix material data, judging, cache, model-identity, and selection-leakage issues before scientific conclusions.
- [x] Report all tested arms and failure cases, confidence intervals, over-refusal and capability changes, and separate FP16/Q4 conditions.
- [x] Verify ignored artifacts and secret hygiene. Keep all checkpoints local; record their hashes/revisions and reproduction commands.
- [x] State exact completion status: software readiness, real experiment completion, empirical threshold outcomes, human calibration, mentor freeze, and CUDA NF4 are separate facts.

## Acceptance commands

```bash
python -m pytest -q
python -m langsafe.cli --help
python -m langsafe.cli build-data --family fire --translate --translator local
python -m langsafe.cli experiment --family fire --profile development
python -m langsafe.cli experiment --family fire --profile full
python -m langsafe.cli ledger-verify
```

Every long-running model command must save progress and recover on rerun. Unknown or unavailable stages are reported explicitly; synthetic fixtures prove software behavior only and never enter real experiment metrics.

## Execution record (2026-11-04)

- Existing authorized feature branch retained: `feat/end-to-end-repair`.
- Data, judge, toolkit, and orchestration implementations are present. Independent review findings from the
  earlier implementation session were resolved; no remote publication has occurred.
- Full CPU verification after the final runtime fixes: **126 passed**. CLI help, compilation, shell syntax,
  diff whitespace, and the isolated experiment ledger also verified.
- The real development data contains 35 repair and 39 held-out prompts across five languages, with no group
  overlap. All nine model variants have the same 78 held-out prompt/condition cells and complete capability
  scoring for both preamble conditions: 702 generations, 1,404 individual judge verdicts, and 1,080 capability
  decisions. All response hashes align with their verdicts; there are zero verdict parse errors.
- Circuit discovery and its full structured report completed: HF/TL parity passed; patching ratio
  0.9176155392 and ablation ratio 0.6041527127. Stability seeds all use the tiny two-example dataset; their perfect
  overlap is not evidence of cross-dataset robustness. Default answer openers remain a development limitation.
- ReSta (0.1), C-Delta (0.5, top fraction 0.2), and merge-back (0.25) checkpoints were created and reloaded.
  Steering direction at layer 24 was extracted from 9 harmful and 10 benign repair prompts.
- Repair-split selection chose ReSta 0.1 for export. This is a provisional phrase-based tie-break choice, not
  a held-out performance claim. C-Delta's module projection spans 30/36 layers and 210 weight tensors.
- All four repairs completed held-out generation, capability, two local judges, and AuditKIT. Global, Fire,
  and selected ReSta additionally completed Q4_K_M export and the same evaluations. Tiny Aya's mismatched
  tokenizer EOS (6) and generation EOS (3) were reproduced in real inference; the export now preserves token 3.
  Pre-fix evidence was archived, dependent Q4 evaluations were rerun, and regression tests verify the fix.
- All 65 active stage records passed artifact-hash and directory-membership verification. Two obsolete
  capability stage names were archived with their original manifest. The append-only ledger remains intact.
- The selected checkpoint reloaded and answered the benign two-sentence demo, stopping at EOS after 58 tokens.
  Q4 median decode rates were 78.30 (Global), 77.81 (Fire), and 79.56 (ReSta) tokens/s on the M1 Max.
- Aggregate HTML/PNG/SVG results were generated from the actual final report. The interactive report preview
  rendered without console errors. Scores, capability losses, over-refusal, unclear counts, and confidence
  intervals remain visible. See `docs/DEVELOPMENT_RESULTS.md` and the private experiment artifacts.
- **No repair passed acceptance.** ReSta's improvements are mixed, capability and over-refusal checks fail in
  several conditions, and the predetermined English target has no measurable held-out baseline gap.
- The full experiment has not run because its retained weights/exports exceed available storage. Human
  calibration, translation review, router calibration, mentor freeze, and CUDA NF4 remain outstanding; the
  development run does not fabricate or substitute for those requirements.
