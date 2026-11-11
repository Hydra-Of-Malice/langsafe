#!/usr/bin/env bash
# One NF4 run on a rented Linux GPU (Hugging Face Jobs or Kaggle), from the private bundle that
# `scripts/remote_nf4.py bundle` uploads. The bootstrap in remote_nf4.py fetches code.tar, manifest.json and
# requirements/ into $WORK and runs this script from the unpacked code. See docs/REMOTE_NF4.md.
#
# Needs HF_TOKEN, BUNDLE_REPO and PLATFORM. Optional: WORK (default /tmp/lexsi), DEADLINE_MINUTES (stop cleanly,
# sync and exit before a platform's session cap; Kaggle's is 12 h), CUDA_ARCH (default: read from nvidia-smi).
set -euo pipefail
: "${HF_TOKEN:?}" "${BUNDLE_REPO:?}" "${PLATFORM:?}"
WORK=${WORK:-/tmp/lexsi}
CODE="$WORK/code"
OUT="$WORK/earth-nf4"
STARTED=$(date +%s)
export PATH="$HOME/.local/bin:$PATH" PYTHONUTF8=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME="$WORK/hf"
log() { printf '[remote-nf4 %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
field() { uv run -q --no-project --python 3.12 python -c "import json,sys; m=json.load(open('$WORK/manifest.json')); print($1)"; }

log "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '. ')}

log "Python environment, pinned to the laptop's versions"
uv venv -q --python "$(field "m['python']")" "$WORK/venv"
PY="$WORK/venv/bin/python"
uv pip install -q --python "$PY" --index-url "$(field "m['torch']['index']")" -r "$WORK/requirements/torch.txt"
uv pip install -q --python "$PY" -c "$WORK/requirements/constraints.txt" -e "$CODE[cuda,repair]" hf_xet
uv pip install -q --python "$PY" --no-deps "safetune @ git+$(field "m['kits']['safetune']['url']")@$(field "m['kits']['safetune']['commit']")"
uv pip install -q --python "$PY" -c "$WORK/requirements/constraints.txt" \
  "auditkit @ git+$(field "m['kits']['auditkit']['url']")@$(field "m['kits']['auditkit']['commit']")" \
  "curatorkit @ git+$(field "m['kits']['curatorkit']['url']")@$(field "m['kits']['curatorkit']['commit']")"
"$PY" -c "import torch, bitsandbytes, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0)); print('bitsandbytes', bitsandbytes.__version__, 'transformers', transformers.__version__)"

LLAMA_COMMIT=$(field "m['llama_cpp']['commit']")
# The CUDA build takes about half an hour on 4 vCPUs, so each GPU type's build is kept in the private repo.
CUDA_RELEASE=$({ nvcc --version 2>/dev/null || true; } | grep -o 'release [0-9.]*' | head -n 1 | cut -d ' ' -f 2 || true)
LLAMA_CACHE="cache/llama-server-${LLAMA_COMMIT:0:12}-sm${ARCH}-cuda${CUDA_RELEASE:-unknown}.tar.gz"
cd "$CODE"
if "$PY" scripts/remote_nf4.py cache-get --work "$WORK" --repo "$BUNDLE_REPO" --name "$LLAMA_CACHE" --to "$WORK/llama-bin"; then
  log "llama.cpp ${LLAMA_COMMIT:0:9} for sm_$ARCH from the build cache"
else
  log "llama.cpp ${LLAMA_COMMIT:0:9} with CUDA for sm_$ARCH (building)"
  mkdir -p "$WORK/llama.cpp" && cd "$WORK/llama.cpp"
  [ -d .git ] || git init -q
  git fetch -q --depth 1 "$(field "m['llama_cpp']['url']")" "$LLAMA_COMMIT" && git checkout -q FETCH_HEAD
  cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF > "$WORK/llama-cmake.log" 2>&1
  cmake --build build --config Release -j "$(nproc)" --target llama-server > "$WORK/llama-build.log" 2>&1
  mkdir -p "$WORK/llama-bin" && cp build/bin/llama-server "$WORK/llama-bin/"
  cd "$CODE"
  "$PY" scripts/remote_nf4.py cache-put --work "$WORK" --repo "$BUNDLE_REPO" --name "$LLAMA_CACHE" --from "$WORK/llama-bin" \
    || log "build cache upload failed; continuing"
fi
export LANGSAFE_LLAMA_SERVER="$WORK/llama-bin/llama-server"
"$LANGSAFE_LLAMA_SERVER" --version 2>&1 | grep -i -m 2 'version\|built'

log "inputs, checked against the laptop by SHA-256"
"$PY" scripts/remote_nf4.py prepare --work "$WORK" --repo "$BUNDLE_REPO"
export LANGSAFE_MODELS="$WORK/models"
"$PY" scripts/remote_nf4.py verify --work "$WORK" --repo "$BUNDLE_REPO"

STATE=$("$PY" scripts/remote_nf4.py restore --work "$WORK" --repo "$BUNDLE_REPO" --platform "$PLATFORM" --output "$OUT" | tail -n 1)
log "earlier output: $STATE"
if [ "$STATE" = complete ]; then exit 0; fi
RESUME=(); [ "$STATE" = resume ] && RESUME=(--resume)

rm -f "$WORK/sync.stop" "$WORK/job-status.json"
"$PY" scripts/remote_nf4.py sync --work "$WORK" --repo "$BUNDLE_REPO" --platform "$PLATFORM" --output "$OUT" &
SYNC=$!
RUN=("$PY" -u scripts/check_nf4_environment.py --source-run "$WORK/bundle/source" --spec "$CODE/configs/spec.yaml"
     --output "$OUT" --family earth --judges local --execute "${RESUME[@]}")
log "NF4 run (${STATE})"
set +e
if [ "${DEADLINE_MINUTES:-0}" -gt 0 ]; then
  LEFT=$(( DEADLINE_MINUTES * 60 - ($(date +%s) - STARTED) ))
  timeout --signal=INT --kill-after=120 "$LEFT" "${RUN[@]}"
else
  "${RUN[@]}"
fi
CODE_EXIT=$?
set -e
log "NF4 run ended with exit code $CODE_EXIT"
printf '{"platform": "%s", "exit_code": %s, "deadline_hit": %s, "finished_at": "%s"}\n' "$PLATFORM" "$CODE_EXIT" \
  "$([ "$CODE_EXIT" = 124 ] && echo true || echo false)" "$(date -u +%FT%TZ)" > "$WORK/job-status.json"
touch "$WORK/sync.stop"
wait "$SYNC"
log "final sync done"
exit "$CODE_EXIT"
