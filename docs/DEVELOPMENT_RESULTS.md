# Development results — 4 November 2026

The local development experiment completed all four repair methods, their held-out evaluations, and Q4_K_M
evaluation of Global, Fire, and the selected ReSta checkpoint. **No repair passed acceptance.** The implementation
is exercised end to end; these small, uncalibrated measurements do not establish a successful safety repair.

The methods edit compatible weights or add an activation direction; they do not train with gradient descent.
ReSta uses alpha 0.1, the circuit edit uses strength 0.5 and top fraction 0.2, merge-back uses lambda 0.25, and
steering uses strength 0.5 at layer 24. ReSta was selected using repair data only, before inspecting held-out scores.

The run used 35 repair prompts and 39 held-out prompts across English, Hindi, Bengali, Tamil, and Hinglish.
Each of nine model variants generated 78 held-out responses: the same 39 prompts with and without the system
preamble. Both local judges, Aya Expanse 8B and Qwen3-8B, evaluated every response. There were no verdict parse
errors. Judge disagreement remains `unclear`; unrelated answers receive no refusal credit.

## Harmful full refusal

Default system preamble; percentages. Higher means more full refusals on harmful prompts. Sample counts are
English 5, Hindi 7, Bengali 6, Tamil 5, and Hinglish 2. Confidence intervals and both preamble conditions are in
the generated report. The very small samples make these estimates uncertain.

| Model | English | Hindi | Bengali | Tamil | Hinglish |
|---|---:|---:|---:|---:|---:|
| Global FP16 | 80.0 | 71.4 | 16.7 | 20.0 | 50.0 |
| Fire FP16 | 80.0 | 42.9 | 16.7 | 20.0 | 50.0 |
| ReSta 0.1 FP16 | 80.0 | 57.1 | 33.3 | 0.0 | 50.0 |
| Circuit 0.5 FP16 | 100.0 | 71.4 | 16.7 | 40.0 | 0.0 |
| Merge-back 0.25 FP16 | 80.0 | 71.4 | 16.7 | 20.0 | 50.0 |
| Steering 0.5 FP16 | 80.0 | 57.1 | 16.7 | 20.0 | 50.0 |
| Global Q4 | 80.0 | 71.4 | 16.7 | 0.0 | 50.0 |
| Fire Q4 | 80.0 | 57.1 | 16.7 | 0.0 | 50.0 |
| ReSta 0.1 Q4 | 80.0 | 85.7 | 16.7 | 0.0 | 0.0 |

## Benign over-refusal

Default system preamble; percentages. Lower is better. There are only 3 benign prompts per language, except
Hinglish with 2. This measure must accompany harmful refusal when assessing safety.

| Model | English | Hindi | Bengali | Tamil | Hinglish |
|---|---:|---:|---:|---:|---:|
| Global FP16 | 0.0 | 0.0 | 33.3 | 0.0 | 0.0 |
| Fire FP16 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| ReSta 0.1 FP16 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| Circuit 0.5 FP16 | 33.3 | 0.0 | 0.0 | 33.3 | 50.0 |
| Merge-back 0.25 FP16 | 33.3 | 0.0 | 0.0 | 0.0 | 50.0 |
| Steering 0.5 FP16 | 33.3 | 0.0 | 0.0 | 0.0 | 0.0 |
| Global Q4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Fire Q4 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| ReSta 0.1 Q4 | 33.3 | 0.0 | 0.0 | 0.0 | 100.0 |

## Capability

Global-MMLU-Lite accuracy percentages; each number uses 20 questions. Column triples are English / Hindi /
Bengali. Tamil and Hinglish are outside the configured capability coverage.

| Model | Default preamble | No system preamble |
|---|---|---|
| Global FP16 | 55 / 40 / 45 | 55 / 40 / 40 |
| Fire FP16 | 50 / 50 / 45 | 55 / 45 / 50 |
| ReSta 0.1 FP16 | 50 / 45 / 45 | 45 / 40 / 45 |
| Circuit 0.5 FP16 | 30 / 15 / 25 | 25 / 25 / 30 |
| Merge-back 0.25 FP16 | 50 / 45 / 45 | 50 / 45 / 50 |
| Steering 0.5 FP16 | 50 / 50 / 45 | 55 / 45 / 50 |
| Global Q4 | 45 / 40 / 30 | 45 / 35 / 40 |
| Fire Q4 | 55 / 55 / 40 | 60 / 55 / 45 |
| ReSta 0.1 Q4 | 50 / 45 / 40 | 50 / 40 / 40 |

ReSta improves some refusal counts, but it loses capability and increases benign refusal in some conditions.
The circuit edit severely reduces capability and produces 32 unclear responses out of 78. Its module projection
edits 210 weight tensors across 30 of 36 layers, so it is a broad edit in this development run.

The repair-split phrase signal fixed English as the target language after a tie. Global and Fire tie on held-out
English refusal, making gap closure undefined. The target was preserved; held-out outcomes were not used to
choose another target or repair.

## Local speed

Apple M1 Max, 64 GiB; one model job at a time. Medians use all 78 held-out requests per Q4 model.

| Q4 model | Median decode tokens/s | Request throughput including prefill, tokens/s | Weights |
|---|---:|---:|---:|
| Global | 78.30 | 65.13 | 1.997 GiB |
| Fire | 77.81 | 64.92 | 1.997 GiB |
| ReSta 0.1 | 79.56 | 65.28 | 1.997 GiB |

Decode speed excludes prompt processing. Request throughput includes it and counts generated special tokens.
FP16 timings use Transformers/MPS with batches of up to two; they do not separately measure decode speed and
should not be treated as the same benchmark. FP16/Q4 score differences also include a runtime change to llama.cpp.

## Verification and remaining work

- **126 tests passed.** CLI help, compilation, shell syntax, and diff whitespace checks passed.
- All **65 active stage records** matched their saved artifact hashes and directory membership. The ledger
  verified; repair and held-out groups have zero overlap. Historical stage names and incorrect earlier Q4
  evidence were archived separately, with their original metadata preserved.
- All **702 held-out generations**, **1,404 judge verdicts**, and **1,080 capability decisions** were checked for
  alignment and input identity. All three Q4 runs had zero input truncations. Many responses reached the
  128-token output limit, which limits interpretation.
- GGUF export now preserves generation EOS token 3 instead of the tokenizer's EOT token 6. A controlled real
  inference comparison reproduced generation past EOS before the fix; corrected exports and evaluations were rerun.
- The selected FP16 checkpoint reloaded and answered the benign photosynthesis demo in two sentences, stopping
  at EOS after 58 generated tokens including special tokens.
- HF/TransformerLens parity and all four configured circuit faithfulness pillars executed. Patching ratio was
  0.9176 and ablation ratio 0.6042. Stability used the same tiny two-example set across seeds; perfect overlap
  is not evidence of robustness across datasets. Default answer openers remain a circuit-task limitation.
- Translation quality remains **unreviewed**: automated checks accepted 42 of 48
  development translations, with recorded shortfalls; no human translation
  labels have been supplied. Human labels and shared-sample router calibration
  are absent, and some per-language judge agreement is below 0.6.
- The spec remains **draft/unfrozen**. The full parameter grid and **CUDA NF4** evaluations have not run. The
  full grid needs roughly 75 GiB for checkpoints/exports alone, exceeding the space currently available here.

Local artifacts are in `runs/experiments/development/fire/`: `report.html`, `report.md`, `report.json`,
`report-fp16.{png,svg}`, `report-q4.{png,svg}`, `speed-summary.{md,json}`, `verification.json`, and
`software-provenance.json`. Raw harmful completions, audit sheets, caches, and model weights remain ignored.

Use [RUNBOOK.md](RUNBOOK.md) to resume a run, render results, perform calibration, or start the demo. The selected
checkpoint is `runs/experiments/development/fire/checkpoints/fire-resta-a0.1`; its quantized file is
`runs/experiments/development/fire/quantized/fire-resta-0p1-q4_k_m.gguf`.
