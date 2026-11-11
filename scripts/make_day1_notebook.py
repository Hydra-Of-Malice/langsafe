"""Write notebooks/00_day1_smoke_test.ipynb."""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Day 1 smoke test: can the whole chain run on Tiny Aya?

Runs on a free **Kaggle T4** (preferred: 30 GPU h/week, 29 GB RAM) or **Colab T4** (16 GB). Nothing here needs HF credits. Each check is small; together they tell us whether to freeze the
spec or re-plan. Expected time: about 60–90 minutes, mostly model downloads.

| # | Check | Pass condition |
|---|---|---|
| 1 | Install the four Lexsi kits | all import, CUDA visible |
| 2 | Build the data sets | held-out/repair written, leakage check passes |
| 3 | Generate with Global and Fire (20 prompts per language and kind, both preamble conditions) | no errors; preamble strip works |
| 4 | Judge: rule detector (instant), then the local judge pair (Aya Expanse 8B + Qwen3-8B, NF4, free) | labels parse; kappa per language printed |
| 5 | Reproduce the gap | Fire refuses less than Global in at least one Fire language |
| 6 | CircuitKIT parity + tiny EAP-IG on Global (English pairs) | scores.json written |
| 7 | SafeTune: load the scores into CircuitInfo | layer subset printed |

**Kaggle settings:** Accelerator → GPU T4 x2 (or P100); Internet → On (needs a phone-verified account).

**Secrets needed** (Kaggle: Add-ons → Secrets; Colab: key icon): `HF_TOKEN` (read access, with the Tiny Aya and
Aya Expanse 8B licences accepted; add "Inference Providers" permission only when we switch to the router judges) and
`GH_TOKEN` (a fine-grained token with read-only Contents access to this repo).
""")
code("""
import os
try:
    from google.colab import userdata          # Colab
    for k in ("HF_TOKEN", "GH_TOKEN"):
        os.environ[k] = userdata.get(k)
except ImportError:
    from kaggle_secrets import UserSecretsClient   # Kaggle
    s = UserSecretsClient()
    for k in ("HF_TOKEN", "GH_TOKEN"):
        os.environ[k] = s.get_secret(k)
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")
code("""
!git clone -q https://$GH_TOKEN@github.com/Hydra-Of-Malice/LEXSI-Hackathon-Project.git project
%cd project
!bash scripts/install_kits.sh
""")
md("## 2. Data")
code("""
from huggingface_hub import login; login(os.environ["HF_TOKEN"])
!python -m langsafe.cli build-data --no-curatorkit
!python -m pytest -q
""")
md("## 3. Generation (small) with Global and Fire")
code("""
!python -m langsafe.cli generate --model CohereLabs/tiny-aya-global --tag smoke-global --family fire --limit 20
!python -m langsafe.cli generate --model CohereLabs/tiny-aya-fire   --tag smoke-fire   --family fire --limit 20
""")
md("## 4–5. Judge and compare")
code("""
# JUDGES = "router" uses the pre-registered 32B judges on the HF router (needs inference credits);
# "local" runs the free 8B pair on this GPU; "rules" is the instant, provisional phrase detector.
JUDGES = "local"
!python -m langsafe.cli judge --tag smoke-global --judges rules
!python -m langsafe.cli judge --tag smoke-fire   --judges rules
!python -m langsafe.cli report --tags smoke-global,smoke-fire --family fire --judges rules
!python -m langsafe.cli judge --tag smoke-global --judges $JUDGES
!python -m langsafe.cli judge --tag smoke-fire   --judges $JUDGES
!python -m langsafe.cli report --tags smoke-global,smoke-fire --family fire --judges $JUDGES
""")
md("## 6. CircuitKIT on Tiny Aya Global (English minimal pairs, tiny run)")
code("""
import os, torch
os.environ["CIRCUITKIT_SKIP_MEM_GUARD"] = "1"
from transformers import AutoTokenizer
from langsafe.circuits.task import english_pairs, write_task_csv, DEFAULT_OPENERS
from langsafe.circuits.discover import discover

tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
csv_path = write_task_csv(tok, english_pairs(limit=32), "en", "smoke-global", DEFAULT_OPENERS["en"])
res = discover("CohereLabs/tiny-aya-global", "smoke-global", "en", csv_path, n_examples=16, ig_steps=3)
print(res["scores_json"]); print(res["faithfulness"])
print("peak GPU GB:", torch.cuda.max_memory_allocated() / 1e9)
""")
md("## 7. SafeTune reads the CircuitKIT scores")
code("""
from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
info = load_circuit_info_from_file(res["scores_json"], top_fraction=0.2)
print("layers:", info.layer_suggestions.layer_subset)
print("modules:", info.layer_suggestions.target_modules)
!python -m langsafe.cli ledger-verify
""")
md("""
## What to send back
Paste the outputs of cells 4–7 (the report tables, kappa, peak GPU memory and the layer list) into the team chat, or
commit `runs/ledger.jsonl` and `runs/judged/` (never `runs/gens/`: it holds raw model outputs).
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
