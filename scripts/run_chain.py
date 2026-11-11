"""Run long jobs one after another, in a process that outlives the session that started it.

    python scripts/run_chain.py e2e-fire water e2e-earth

Each step is resumable, so rerunning a chain continues where it stopped. Progress is in runs/chain-status.json, and each
step logs to runs/chain-<step>.log. To stop a chain, kill this process and then its child; there is no retry loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "runs" / "chain-status.json"
CLI = [sys.executable, "-u", "-m", "langsafe.cli"]
SWEEP = [sys.executable, "-u", "scripts/pilot_multijail.py"]
NF4 = [sys.executable, "-u", "scripts/check_nf4_environment.py", "--family", "earth", "--min-vram-gib", "7", "--execute"]
STEPS = {
    "e2e-fire": CLI + ["experiment", "--family", "fire", "--profile", "development"],
    "water": SWEEP + ["--models", "global,fire,earth,water", "--preambles", "default,none"],
    "e2e-earth": CLI + ["experiment", "--family", "earth", "--profile", "development"],
    "e2e-baseline": CLI + ["experiment", "--family", "earth", "--profile", "development", "--stop-after", "baseline"],
    "earth-baseline": CLI + ["experiment", "--family", "earth", "--profile", "full", "--stop-after", "baseline"],
    "earth-circuit": CLI + ["experiment", "--family", "earth", "--profile", "full", "--stop-after", "circuit"],
    "earth-repairs": CLI + ["experiment", "--family", "earth", "--profile", "full", "--stop-after", "repairs"],
    "earth-full": CLI + ["experiment", "--family", "earth", "--profile", "full"],
    "e2e-nf4": NF4 + ["--source-run", "runs/experiments/development/earth", "--spec", "configs/development.yaml",
                      "--output", "runs/experiments/development/earth-nf4", "--allow-development"],
    "earth-nf4": NF4 + ["--source-run", "runs/experiments/full/earth", "--spec", "configs/spec.yaml",
                        "--output", "runs/experiments/full/earth-nf4"],
}
NEEDS_EARTH = {"e2e-earth", "e2e-baseline", "earth-baseline", "earth-circuit", "earth-repairs", "earth-full", "e2e-nf4", "earth-nf4"}
# The comparison-model check needs a directory of its own: its spec differs from the finished development runs.
STEP_ENV = {"e2e-baseline": {"LANGSAFE_RUNS_DIR": str(ROOT / "runs" / "experiments" / "development" / "earth-compare")}}


def status(**fields) -> None:
    current = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    current.update(fields, updated=datetime.now().isoformat(timespec="seconds"))
    STATUS.write_text(json.dumps(current, indent=2), encoding="utf-8")


def earth_ready() -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download("CohereLabs/tiny-aya-earth", local_files_only=True,
                          allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"])
        return True
    except Exception:
        return False


def main() -> None:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", LANGSAFE_CIRCUIT_DEVICE="cpu")
    status(pid=os.getpid(), chain=sys.argv[1:], done=[], failed=None, running=None)
    done = []
    for step in sys.argv[1:]:
        if step in NEEDS_EARTH:
            status(running=f"waiting for Earth weights before {step}")
            while not earth_ready():
                time.sleep(60)
        status(running=step)
        with open(ROOT / "runs" / f"chain-{step}.log", "ab") as log:
            code = subprocess.run(STEPS[step], cwd=ROOT, env={**env, **STEP_ENV.get(step, {})}, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            status(running=None, failed={"step": step, "exit_code": code})
            sys.exit(code)
        done.append(step)
        status(done=done, running=None)
    status(finished=True)


if __name__ == "__main__":
    main()
