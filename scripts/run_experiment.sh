#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec .venv/bin/python -u -m langsafe.cli experiment "$@"
