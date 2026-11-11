"""Tiny CircuitKIT run: EAP-IG on English refusal minimal pairs, then SafeTune reads the scores.

    python scripts/circuit_smoke.py [--model CohereLabs/tiny-aya-global] [--pairs 32] [--examples 16]

Runs on CUDA or Apple Silicon (MPS); set LANGSAFE_DEVICE=cpu to force CPU. Prints peak memory and the layer subset
SafeTune would edit. Same steps as cells 10-12 of notebooks/00_day1_smoke_test.ipynb.
"""
import argparse
import os
import time

os.environ.setdefault("CIRCUITKIT_SKIP_MEM_GUARD", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from transformers import AutoTokenizer

from langsafe.circuits.discover import discover
from langsafe.circuits.task import DEFAULT_OPENERS, english_pairs, write_task_csv
from langsafe.gen.model import best_device, resolve_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
ap.add_argument("--pairs", type=int, default=32)
ap.add_argument("--examples", type=int, default=16)
ap.add_argument("--preamble", choices=["default", "none"], default="default")
a = ap.parse_args()

dev = best_device()
print("device:", dev)
tag = "smoke-" + a.model.split("/")[-1]
tok = AutoTokenizer.from_pretrained(resolve_model(a.model))
csv_path = write_task_csv(tok, english_pairs(limit=a.pairs), "en", tag, DEFAULT_OPENERS["en"], preamble=a.preamble,
                          limit=a.examples)
t0 = time.time()
res = discover(a.model, tag, "en", csv_path, n_examples=a.examples, ig_steps=3)
print("scores:", res["scores_json"], f"({time.time() - t0:.0f}s)")
print("faithfulness:", res["faithfulness"])
if dev == "cuda":
    print("peak GPU GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))
elif dev == "mps":
    print("MPS driver GB:", round(torch.mps.driver_allocated_memory() / 1e9, 2))

from safetune.core.circuit_kit.adapter import load_circuit_info_from_file

info = load_circuit_info_from_file(res["scores_json"], top_fraction=0.2)
print("SafeTune layer subset:", info.layer_suggestions.layer_subset)
print("SafeTune target modules:", info.layer_suggestions.target_modules)
