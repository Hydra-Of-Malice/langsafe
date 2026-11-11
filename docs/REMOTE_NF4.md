# NF4 on a rented GPU

The NF4 check (`scripts/check_nf4_environment.py`, the chain's `earth-nf4` step) takes 12 to 15 hours on the laptop's
RTX 4060. While the laptop finishes the Q4 checks, the same check runs on rented Linux GPUs instead, through
`scripts/remote_nf4.py` and `scripts/remote_nf4.sh`:

- **Hugging Face Jobs, `a10g-small`** (A10G 24 GB, about $1 an hour, paid from the team's Hugging Face credit). This is
  the main run.
- **Kaggle, a T4** (free, at most 12 hours a session). This is a second run on different hardware: a backup if the first
  fails, and a check that the NF4 results do not depend on the GPU.

Everything else in `docs/CUDA_EVALUATION.md` still holds; this adds the transfer and the comparison with the laptop.

## What leaves the laptop, and where it goes

One **private** Hugging Face dataset repo, `HydraOfMalice/lexsi-nf4-private`, holds the bundle and the results:

| Path | What |
|---|---|
| `code.tar` | `git archive HEAD`: the committed code, nothing uncommitted |
| `manifest.json`, `laptop-plan.json` | The laptop's NF4 plan and the SHA-256 of every input |
| `requirements/` | The laptop's package versions, as pins |
| `source/` | The source evidence the plan lists: the held-out and repair prompts, the full-precision answers and judgements, capability results, `selection.json`, and the selected repair checkpoint (6.3 GB) |
| `results/<platform>/` | Each run's NF4 output, synced every 15 minutes |

The prompts include harmful ones, and the answers are the models' replies to them. So the repo must stay private.
Every command that touches it checks that it is, and refuses a public one. Nothing from it goes into git, the public
copy or the submission. Delete the repo once the results are in.

The base models (`CohereLabs/tiny-aya-global`, `-earth`), the judge GGUF files and Global-MMLU-Lite are downloaded on
the GPU host from their own Hub repos. The base models come at the laptop's exact snapshot revision. Every file is
checked against the laptop's SHA-256.

## Same run, different machine

`scripts/remote_nf4.sh` rebuilds the laptop's setup and stops on the first difference:

- **Python and packages:** Python 3.12, the laptop's torch build (from PyTorch's CUDA index), and every other package at
  the laptop's version. The Lexsi kits come at their pinned commits.
- **Judges:** the same two GGUF files (`aya-expanse-8b`, `Qwen3-8B`, Q4_K_M), checked by SHA-256. They are served by
  llama.cpp built from the laptop's commit, compiled for the host's GPU. So the remote judges are the same models and
  the same server code as the laptop's, not a transformers fallback.
- **Plan check:** `remote_nf4.py verify` builds the NF4 plan on the host and compares it with the laptop's, field by
  field: code, spec, source evidence, all four checkpoints, both judges, the stages and the report tags.

What does differ is the GPU, and the record says so: `nf4-environment.json` names the device, driver and CUDA version.
Kernels differ between GPUs, so a few greedy answers or judge labels can differ by machine. All three NF4 models of a run
are judged on the same GPU, so the NF4 comparisons in a run's report are like with like. Comparing the A10G run with the
T4 run measures how much the GPU matters.

## Interruptions

`check_nf4_environment.py --execute --resume` continues a run in its own output directory. It skips the stages already
done and reruns the one that was interrupted, from that stage's own caches. It refuses an output that belongs to a
different plan, or one that is already complete.

`remote_nf4.sh` syncs the output to `results/<platform>/` every 15 minutes, using a consistent copy of the judge cache.
At start it restores any earlier output from there. With `DEADLINE_MINUTES` set, it stops the run cleanly before the
platform's cap, syncs, and exits, so the next session resumes.

## Commands

On the laptop, with the code committed:

```bash
.venv/Scripts/python.exe scripts/remote_nf4.py bundle
.venv/Scripts/python.exe scripts/remote_nf4.py launch-hf --flavor a10g-small --timeout 11h --dry-run
.venv/Scripts/python.exe scripts/remote_nf4.py launch-hf --flavor a10g-small --timeout 11h
.venv/Scripts/hf.exe jobs logs <job id>
.venv/Scripts/python.exe scripts/remote_nf4.py fetch --platform hf-a10g
```

- `launch-hf` passes the laptop's Hugging Face login to the job as its `HF_TOKEN` secret, the way `hf jobs run -s
  HF_TOKEN` does. The token is never printed.
- `--timeout` caps the cost: 11 hours on `a10g-small` is at most $11.
- `fetch` writes `runs/experiments/full/earth-nf4-<platform>/`, which git ignores.

On Kaggle, in a notebook with a T4 accelerator and internet on:

1. Add a secret named `HF_TOKEN`. Use a fine-grained token with read and write access to the private repo only. You add
   it yourself.
2. Run this cell. *Save Version → Save & Run All* runs it in the background for up to 12 hours.

```python
import os, subprocess, sys, tarfile
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub==1.33.0'], check=True)
from kaggle_secrets import UserSecretsClient
from huggingface_hub import hf_hub_download
os.environ.update(HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN'), BUNDLE_REPO='HydraOfMalice/lexsi-nf4-private',
                  PLATFORM='kaggle-t4', WORK='/tmp/lexsi', DEADLINE_MINUTES='690', CUDA_VISIBLE_DEVICES='0')
tarfile.open(hf_hub_download(os.environ['BUNDLE_REPO'], 'code.tar', repo_type='dataset',
                             local_dir='/tmp/lexsi')).extractall('/tmp/lexsi/code')
sys.path.insert(0, '/tmp/lexsi/code/scripts')
import remote_nf4
subprocess.run(['bash', '-c', remote_nf4.BOOTSTRAP], check=True)
```

A T4 run may need a second session. Run the same notebook again and it resumes from `results/kaggle-t4/`.

The cell above is the long form. `scripts/kaggle_start.py`, which `bundle` puts at the repo root, does the same thing,
so the notebook can instead run the one line in its docstring. A one-line cell is easier to paste into Kaggle's editor.

**Keep the bundle frozen while runs use it.** A run resumes only into the plan it started from. Rebuilding the bundle
changes that plan, and the run then refuses to resume. `bundle` refuses while any run has results. Pass
`--replace-with-runs` only once those runs are finished or given up.

## Stopping the laptop's own NF4 step

The chain runs `earth-nf4` on the laptop right after Q4. Once a remote run is under way, put any file in
`runs/experiments/full/earth-nf4/`. The laptop's NF4 preflight then refuses the occupied directory, and the chain stops
after Q4 without touching anything. That frees the laptop for the fair circuit baseline. If the remote runs fail, delete
the file and run the step on the laptop:

```bash
.venv/Scripts/python.exe scripts/run_chain.py earth-nf4
```
