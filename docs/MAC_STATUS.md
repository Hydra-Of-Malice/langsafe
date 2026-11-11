# Mac evaluation status

The source machine is the MacBook Pro `MacBookPro18,2` with an Apple M1 Max,
10 CPU cores, 64 GB unified memory, Darwin `27.0.0` on `arm64`. The local
environment reports Python `3.12.14`, PyTorch `2.15.0`, Transformers `5.22.1`,
MPS available, CUDA unavailable, and no installed `bitsandbytes`. `nvidia-smi`
and `nvcc` are not present. No SSH, Kaggle, Colab, or other CUDA session is
configured in this workspace, so there is no host/session identifier to hand
over and no credential is recorded here.

The Mac run uses MPS/FP16 for Transformers and Metal through llama.cpp for the
Q4 comparison. The completed development run is documented in
`docs/DEVELOPMENT_RESULTS.md`; its ledger and stage hashes verify, all four
repair arms were exercised, and the report did not establish an accepted
repair. NF4 is intentionally unexecuted on this machine.

The full Fire data stage has completed under
`runs/experiments/full/fire/`. It contains 2,613 group-isolated prompts:
1,580 held out and 1,033 for repair. Coverage is present in English, Hindi,
Bengali, Tamil, and Hinglish for both harmful and benign cells. Automated
translation validation excluded 134 requests; the manifest records each
failure and keeps the resulting dataset eligible for review. The subsequent
full model stages have not started because this volume has about 52 GiB free,
while the retained full parameter grid needs roughly 75 GiB before temporary
conversion space. Resume the same command on an external volume with enough
space:

```bash
LANGSAFE_RUNS_DIR=/Volumes/EXTERNAL/lexsi/full/fire \
  bash scripts/run_experiment.sh --family fire --profile full
```

## Human review ownership

Only your team’s human reviewers can supply the safety labels and translation
quality decisions. Agents, local judges, and router models are not human
reviewers. Use reviewers who can read the target language and its safety
meaning: Hindi and Hinglish, Bengali, and Tamil reviewers should each review
their own language; a second reviewer can adjudicate `unclear` or rejected
items. The reviewer records a stable human ID and an ISO-8601 timestamp in the
sheet.

The prepared full-data packet is
`runs/experiments/full/fire/review/translation_review.with-source.csv`.
It has 1,766 translated/code-mixed rows, all 1,766 English sources available,
and zero labels. The private source index used to restore capped-out English
sources is
`runs/experiments/full/fire/review/multijail-source-reference.jsonl`; keep it
with the packet. Its validation summary is
`runs/experiments/full/fire/review/translation_review.checked.json`, currently
`complete: false` and `quality_status: unreviewed` because no human labels have
been entered.

The development packet remains available at
`runs/experiments/development/fire/review/translation_review.with-source.csv`
(50 rows, zero labels). The local judge hand-check sheets are under
`runs/experiments/development/fire/hand_check/`; every existing sheet currently
has zero human labels. Follow [REVIEW_WORKFLOW.md](REVIEW_WORKFLOW.md) and edit
only the documented label, reviewer, timestamp, and notes columns.

## CUDA handoff

Use [CUDA_EVALUATION.md](CUDA_EVALUATION.md) for the transfer manifest,
offline dry-run, Linux preflight, serial NF4 execution, and final validation.
The reviewable code/document patch is
`docs/CUDA_HANDOFF.patch`. The recommended target is a fresh Linux NVIDIA
host such as a Kaggle T4 or Colab T4; the runner records the actual GPU model,
UUID, driver, VRAM, CUDA runtime, and package versions during preflight.
