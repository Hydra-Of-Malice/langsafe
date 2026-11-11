#!/usr/bin/env bash
# The fair circuit baseline (scripts/fair_circuit_baseline.py) on a rented GPU, in fp32 with TF32 off. Its inputs, the
# six circuits and their task files, come from the private repo's fair/ folder (`remote_nf4.py fair-bundle`), and the
# results go back to fair/results/. Each circuit's check draw must reproduce the ratio the laptop logged within 1e-3,
# or the script writes nothing for it. Needs HF_TOKEN and BUNDLE_REPO; see docs/REMOTE_NF4.md.
set -euo pipefail
: "${HF_TOKEN:?}" "${BUNDLE_REPO:?}"
WORK=${WORK:-/tmp/lexsi}
CODE="$WORK/code"
export PATH="$HOME/.local/bin:$PATH" PYTHONUTF8=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME="$WORK/hf"
log() { printf '[remote-fair %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
field() { uv run -q --no-project --python 3.12 python -c "import json; m=json.load(open('$WORK/$1')); print($2)"; }

log "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

log "Python environment, pinned to the laptop's versions"
uv venv -q --python "$(field manifest.json "m['python']")" "$WORK/venv"
PY="$WORK/venv/bin/python"
uv pip install -q --python "$PY" --index-url "$(field manifest.json "m['torch']['index']")" -r "$WORK/requirements/torch.txt"
uv pip install -q --python "$PY" -c "$WORK/requirements/constraints.txt" -e "$CODE[models]" hf_xet \
  "circuitkit @ git+$(field fair/manifest.json "m['circuitkit']['url']")@$(field fair/manifest.json "m['circuitkit']['commit']")"
"$PY" -c "import torch, circuitkit, transformer_lens; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'transformer_lens', transformer_lens.__version__ if hasattr(transformer_lens, '__version__') else '?')"

cd "$CODE"
log "inputs and models, checked against the laptop by SHA-256"
"$PY" scripts/remote_nf4.py fair-prepare --work "$WORK" --repo "$BUNDLE_REPO"

# One circuit at a time, uploading after each, so a timeout loses at most the circuit in progress. Loading the fp32 model
# for CircuitKIT peaks near 30 GB of host RAM: use a flavour with more than that (l40sx1: 62 GB; a10g-small has 15).
STATUS=0
for SUMMARY in "$WORK"/fair/run/circuits/*__*/summary.json; do
  CIRCUIT=$(basename "$(dirname "$SUMMARY")")
  log "circuit $CIRCUIT"
  set +e
  # Offline for the evaluation only, so CircuitKIT loads the pinned snapshots; the upload needs the Hub.
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$PY" -u scripts/fair_circuit_baseline.py "$WORK/fair/run" --draws 3 \
    --device cuda --only "$CIRCUIT" 2>&1 | tee -a "$WORK/fair-baseline.log"
  CODE_EXIT=${PIPESTATUS[0]}
  set -e
  [ "$CODE_EXIT" = 0 ] || STATUS=$CODE_EXIT
  log "circuit $CIRCUIT ended with exit code $CODE_EXIT"
  "$PY" scripts/remote_nf4.py fair-upload --work "$WORK" --repo "$BUNDLE_REPO" --status "$STATUS"
done
log "fair baseline ended with exit code $STATUS"
exit "$STATUS"
