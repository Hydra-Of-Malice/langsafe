#!/usr/bin/env bash
# Day-1 smoke test on any machine with the venv active (Apple Silicon, CUDA laptop, Kaggle terminal).
#   JUDGES=local bash scripts/day1.sh        # default: free local judge pair
#   JUDGES=router bash scripts/day1.sh       # pre-registered 32B judges on the HF router (needs inference credits)
#   SKIP_CIRCUIT=1 bash scripts/day1.sh      # skip the CircuitKIT step
# Writes runs/day1_report.md (metrics only, no model outputs: safe to commit and share).
set -euo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
JUDGES="${JUDGES:-local}"
LIMIT="${LIMIT:-20}"
OUT=runs/day1_report.md
mkdir -p runs
lx() { python -m langsafe.cli "$@"; }

echo "== 1. data + tests"
lx build-data --no-curatorkit
python -m pytest -q

echo "== 2. generate ($LIMIT per language and kind, both preamble conditions)"
lx generate --model CohereLabs/tiny-aya-global --tag smoke-global --family fire --limit "$LIMIT"
lx generate --model CohereLabs/tiny-aya-fire   --tag smoke-fire   --family fire --limit "$LIMIT"

echo "== 3. judge: rules (provisional), then $JUDGES"
{
  echo "# Day-1 smoke report ($(date -u +%Y-%m-%dT%H:%MZ), $(python -c 'import platform;print(platform.platform())'))"
  echo
  for J in rules "$JUDGES"; do
    echo "## Judges: $J"
    echo
    for T in smoke-global smoke-fire; do
      echo "Judge summary for $T (kappa per language; hand-check queue size):"
      echo '```'; lx judge --tag "$T" --judges "$J"; echo '```'
    done
    lx report --tags smoke-global,smoke-fire --family fire --judges "$J"
  done
} | tee "$OUT"

if [ -z "${SKIP_CIRCUIT:-}" ]; then
  echo "== 4. CircuitKIT tiny EAP-IG + SafeTune reads the scores"
  python scripts/circuit_smoke.py 2>&1 | tee -a "$OUT"
fi

lx ledger-verify | tee -a "$OUT"
echo "Done. Share $OUT and runs/ledger.jsonl; never runs/gens/ (raw model outputs)."
