# Portable CUDA/NF4 evaluation

NF4 is an in-process `bitsandbytes` format. It requires a CUDA host, Linux or Windows, and
the same float16 experiment inputs used by the source run. The Mac workflow
does not execute NF4: `scripts/run_nf4_evaluation.sh --dry-run` only validates
the handoff and prints a plan.

The main run uses the laptop's RTX 4060 through the `earth-nf4` step of
`scripts/run_chain.py`, with `--min-vram-gib 7`. A Kaggle or Colab T4 also works;
its non-secret hardware and software identity are captured by `--check-hardware`.

The handoff is deliberately isolated. Point `--source-run` at a completed
experiment directory and `--output` at a new directory on the CUDA machine.
The source run must contain its manifest, repair/held-out JSONL files,
`selection.json`, Global/Earth/selected-repair generations, two-judge judgments,
AuditKIT records, and both capability conditions. The planner hashes these
files, the source spec, all checkpoint files, and the code/runtime identity.
It rejects mismatched specs or data hashes, overlapping repair/held-out groups,
incomplete judge/capability coverage, steering-only selections, and an output
directory that is occupied or overlaps the source. It never copies or evaluates
GGUF files as NF4.

## Offline plan on the source machine

```bash
LANGSAFE_PYTHON=.venv/bin/python bash scripts/run_nf4_evaluation.sh \
  --source-run runs/experiments/full/earth \
  --spec configs/spec.yaml \
  --output /mnt/lexsi/nf4-earth \
  --family earth --judges local --dry-run > nf4-plan.json
```

The plan must say `status: planned_not_executed`, `precision: nf4`, and
`hardware_checked: false`. It lists exactly three NF4 model tags:
`global-nf4`, `earth-nf4`, and the repair tag recorded in
`selection.json` with `-nf4` appended. Every tag is generated on the complete
held-out prompt set; capability is run with both `default` and `none`
preambles. Each tag then receives the configured two judges, AuditKIT, and a
report containing same-precision Global/Earth/repair comparisons. No `--limit`
is added to a full-profile plan. Development specs are refused unless
`--allow-development` is supplied and remain development-only evidence.

The plan is a portable contract. Copy it and the source checkpoint/evidence
bundle to the CUDA host through the project's approved transfer mechanism. Do
not copy harmful raw generations to a public location. On the remote host,
provision the exact Python environment, model snapshots, and dataset cache
offline; the runner never installs packages or silently downloads replacements.

The repository's current local source tree is intentionally uncommitted. The
handoff therefore includes the source commit (when one exists) plus the
`code.files_sha256` map in `nf4-plan.json`; copy the exact working tree or an
archive made from that map rather than checking out an unrelated branch. Keep
the source `data/manifest.json`, `selection.json`, both split JSONL files, and
the selected `checkpoints/<tag>/` directory together. The planner's
`source_evidence.files` plus `source_evidence.selected_checkpoint.files` maps
are the transfer manifest. They record every copied path, byte count, and
SHA-256; validate them after transfer before running
preflight. Do not add generated generations, judged responses, tokens, or
checkpoint files to a public commit.

Before transfer, preserve the working-tree record alongside the private bundle:

```bash
git status --short > /mnt/lexsi/source-git-status.txt
git diff --binary -- src configs scripts docs pyproject.toml > /mnt/lexsi/source-working-tree.diff
```

The diff is a review aid for the intentional local edits; the runner itself
uses the code/spec/data/checkpoint SHA-256 maps as the execution identity.
Keep that diff private because the source tree and generated evidence may
contain experiment details that are not for publication.

## CUDA preflight

On Linux, run:

```bash
LANGSAFE_PYTHON=.venv/bin/python bash scripts/run_nf4_evaluation.sh \
  --source-run /mnt/lexsi/full-earth \
  --spec /mnt/lexsi/full-earth/source-spec.yaml \
  --output /mnt/lexsi/nf4-earth \
  --family earth --judges local --check-hardware
```

Preflight requires one visible CUDA device, compute capability 7.0 or newer,
at least 12 GiB total and 8 GiB free VRAM, a CUDA-enabled `bitsandbytes`, and
enough output storage. It performs a tiny NF4 quantize/dequantize CUDA kernel
smoke test and verifies that cached Global-MMLU-Lite items hash exactly like
the source evidence. It does not load any checkpoint or start evaluation.

The prepared environment must use Python 3.12+, `torch>=2.7`,
`transformers>=5.15,<6`, `accelerate>=1.0,<2`, `bitsandbytes>=0.45`,
`datasets>=3.0`, `huggingface_hub>=0.30`, `numpy>=1.26`, `pyyaml>=6`,
`auditkit` commit `71316da`, `curatorkit` commit `dd9e835`, and SafeTune
commit `3901a9f`. The existing `scripts/install_kits.sh --repair --cuda`
workflow prepares these packages; the handoff runner never installs them.
Record the actual versions printed by that installer in the remote run's
`nf4-environment.json`; package metadata is part of the provenance.
Preflight also recomputes the repository code hash and rejects a remote tree
that differs from the offline plan.

For reference, this Mac's source environment currently reports Python 3.12,
torch 2.15.0, transformers 5.22.1, accelerate 1.15.0, datasets 5.1.0,
hugging-hub 2.4.0, NumPy 2.5.4, PyYAML 6.0.3, AuditKIT 1.1.3, CuratorKIT
0.1.21, SafeTune 0.1.12, and PEFT 0.21.2. These are provenance facts for the
source tree; they are not a claim that this Darwin environment can run NF4.
The CUDA host's recorded versions, not this list, are the versions used for
the remote result. `scripts/install_kits.sh --repair --cuda` preserves the
host's existing torch/transformers pins and installs the project's declared
compatibility ranges; do not silently upgrade them between preflight and run.

The local judge pair is also loaded with NF4 on the same CUDA device. Router
judges are optional and require both `--judges router` and explicit
`--allow-router` plus `HF_TOKEN`; router requests are never made by a dry run
or local-jury execution.

## Execute serially

```bash
LANGSAFE_PYTHON=.venv/bin/python bash scripts/run_nf4_evaluation.sh \
  --source-run /mnt/lexsi/full-earth \
  --spec /mnt/lexsi/full-earth/source-spec.yaml \
  --output /mnt/lexsi/nf4-earth \
  --family earth --judges local --execute
```

The runner copies the selected repair checkpoint into the isolated output,
sets `LANGSAFE_DEVICE=cuda`, enables Hub/datasets offline mode, and calls the
existing CLI one stage at a time. `current-stage.json`, per-stage logs,
`nf4-environment.json`, and `nf4-status.json` are written atomically. A failed
stage leaves `nf4_cuda: not_complete`; no failed or partial directory can be
reported as an NF4 result. Final validation checks generation identities and
rendered-prompt hashes against the source, explicit `quant: nf4` capability
records for both preambles, two-judge coverage, AuditKIT artifacts, same-
precision report comparisons, and artifact hashes.

Useful post-run checks, all against the isolated output, are:

```bash
.venv/bin/python -m langsafe.cli --spec /mnt/lexsi/nf4-earth/source-spec.yaml \
  --runs-dir /mnt/lexsi/nf4-earth ledger-verify
.venv/bin/python -m langsafe.cli --spec /mnt/lexsi/nf4-earth/source-spec.yaml \
  --runs-dir /mnt/lexsi/nf4-earth status --directory /mnt/lexsi/nf4-earth
python - <<'PY'
import json
from pathlib import Path
p = Path('/mnt/lexsi/nf4-earth')
status = json.loads((p / 'nf4-status.json').read_text())
assert status['software_run_complete'] and status['nf4_cuda'] == 'executed'
assert json.loads((p / 'report.json').read_text())['validation_gaps'] == []
print(status['report'])
PY
```

The report retains refusal rates, benign over-refusal, capability accuracy,
paired confidence intervals, McNemar tests, judge kappa, human calibration,
and AuditKIT scorer artifacts. Run `scripts/render_results.py` on
`report.json` only after the status and ledger checks pass; a report with
missing calibration or an absent NF4 same-precision comparison remains
provisional.

`src/langsafe/gen/model.py` refuses NF4 on CPU/MPS, uses a strict CUDA device
map (no CPU/disk offload), and rejects a loaded model that is not actually
4-bit or has any non-CUDA placement. This prevents a silently mixed-precision
run.

The Mac cannot satisfy these preflight conditions: it is Darwin arm64 with no
CUDA device and no installed `bitsandbytes`. That is an environment fact, not
an NF4 execution result.
