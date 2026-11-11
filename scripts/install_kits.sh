#!/usr/bin/env bash
# Pinned APIs verified from upstream source. Run in the intended virtualenv.
# --repair keeps an existing generation runtime (including huggingface-hub 2.x).
# --all / --circuits require hub>=1.5,<2: use a separate environment if needed.
# LANGSAFE_PYTHON=/path/to/python bash scripts/install_kits.sh --repair
set -euo pipefail

profile="${1:---all}"
case "$profile" in
  --repair|--circuits|--all) ;;
  *) echo "Usage: $0 [--repair|--circuits|--all] [--cuda]" >&2; exit 2 ;;
esac
cuda="${2:-}"
if [[ -n "$cuda" && "$cuda" != "--cuda" ]]; then
  echo "Only --cuda may follow the installation profile" >&2
  exit 2
fi

kit_python="${LANGSAFE_PYTHON:-python}"
kit_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$kit_root"

# Refuse an implicit hub downgrade before changing anything. CircuitKIT's
# patched TransformerLens 3.8.0 requires its declared hub range. Pin the working
# torch/transformers versions below so dependency resolution cannot replace them.
"$kit_python" - "$profile" "$cuda" <<'PY'
import importlib.metadata as metadata
import platform
import sys
from packaging.version import Version

profile, cuda = sys.argv[1:]
if cuda and platform.system() == "Darwin":
    raise SystemExit("CUDA/NF4 requires a CUDA host; use the portable profile and GGUF on Darwin.")
if profile in {"--all", "--circuits"}:
    try:
        hub = metadata.version("huggingface-hub")
    except metadata.PackageNotFoundError:
        hub = None
    if hub is not None and not Version("1.5") <= Version(hub) < Version("2"):
        raise SystemExit(
            f"CircuitKIT e71d5e1 requires huggingface-hub>=1.5,<2; installed {hub}. "
            "Create a separate CircuitKIT virtualenv, or use --repair in this runtime. "
            "No packages were changed.")
PY

kit_constraints="$(mktemp -t langsafe-kit-constraints.XXXXXX)"
trap 'rm -f "$kit_constraints"' EXIT
"$kit_python" - "$kit_constraints" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path

pins = []
for package in ("torch", "transformers"):
    try:
        pins.append(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        pass
Path(sys.argv[1]).write_text("\n".join(pins) + "\n")
PY

if "$kit_python" -m pip --version >/dev/null 2>&1; then
  kit_install=("$kit_python" -m pip install --constraint "$kit_constraints")
elif command -v uv >/dev/null 2>&1; then
  kit_install=(uv pip install --python "$kit_python" --constraint "$kit_constraints")
else
  echo "Install pip in the selected environment or make uv available first." >&2
  exit 2
fi

if [[ -n "$cuda" ]]; then
  "${kit_install[@]}" -e '.[repair,cuda,dev]'
else
  "${kit_install[@]}" -e '.[repair,dev]'
fi
# SafeTune's unconditional bitsandbytes/training dependencies are unnecessary
# for our CPU weight arithmetic and steering APIs. Smoke-test those exact APIs.
if [[ "$profile" != "--circuits" ]]; then
  "${kit_install[@]}" --no-deps 'safetune @ git+https://github.com/Lexsi-Labs/SafeTune@3901a9f'
  "${kit_install[@]}" 'auditkit @ git+https://github.com/Lexsi-Labs/AuditKIT@71316da'
  "${kit_install[@]}" 'curatorkit @ git+https://github.com/Lexsi-Labs/CuratorKIT@dd9e835'
fi
if [[ "$profile" != "--repair" ]]; then
  "${kit_install[@]}" 'huggingface-hub>=1.5,<2' 'transformer-lens==3.8.0' \
    'circuitkit @ git+https://github.com/Lexsi-Labs/CircuitKIT@e71d5e1'
fi

"$kit_python" - "$profile" <<'PY'
import importlib.metadata as metadata
import sys
import torch
import transformers

profile = sys.argv[1]
if profile != "--repair":
    import circuitkit  # applies the pinned TransformerLens patch series
    from circuitkit.pipeline import Pipeline
if profile != "--circuits":
    import auditkit
    import curatorkit
    from safetune.recover import apply_resta, apply_ctheta, apply_wise_ft
    from safetune.steer import RefusalDirectionConfig, RefusalDirectionModel, extract_refusal_direction
packages = ["torch", "transformers", "huggingface-hub"]
packages += ["circuitkit", "transformer-lens"] if profile == "--circuits" else ["safetune", "auditkit", "curatorkit"]
if profile == "--all":
    packages += ["circuitkit", "transformer-lens"]
for package in packages:
    print(f"{package}: {metadata.version(package)}")
print(f"cuda: {torch.cuda.is_available()} | mps: {torch.backends.mps.is_available()}")
print("SafeTune is intentionally installed with only the repair-stage dependencies; its optional training/eval metadata may be unmet.")
PY
