#!/usr/bin/env bash
# Portable CUDA entry point. Planning is always explicit; execution needs --execute.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON="${LANGSAFE_PYTHON:-$TASK_ROOT/.venv/bin/python}"
if [[ ! -x "$TASK_PYTHON" ]]; then
  printf '%s\n' 'Set LANGSAFE_PYTHON to the prepared Python 3.12+ environment; this runner never installs dependencies.' >&2
  exit 1
fi
exec "$TASK_PYTHON" -u "$TASK_ROOT/scripts/check_nf4_environment.py" "$@"
