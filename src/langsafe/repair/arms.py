"""SafeTune repair arms. Weight-space arms run on CPU (allow 40-50 GB of commit at peak); steering runs on GPU.

Arms (configs/spec.yaml):
  resta          Fire + a * (Global - Base)                          whole model          SafeTune apply_resta
  circuit        Fire + s * (Global - Base), top CircuitKIT layers   circuit-restricted   SafeTune apply_ctheta
  merge_back     (1 - l) * Fire + l * Global                         whole model          SafeTune apply_wise_ft
  circuit_merge  Fire + l * (Global - Fire), top CircuitKIT layers   circuit-restricted   SafeTune apply_ctheta
  steer          add a refusal direction at one layer at run time    inference-time       SafeTune RefusalDirectionModel

Each circuit-restricted arm has a whole-model twin with the same vector, and a control that applies it to the same
number of randomly chosen non-circuit layers. Twin and control together isolate what CircuitKIT's localisation adds.
Knobs (alpha, strength, lambda, steering strength) are chosen on the *repair* split only (see select.py), never on
held-out data. SafeTune's defaults (ReSta alpha=1, steering alpha=20) break Tiny Aya, so every knob is explicit.
"""
from __future__ import annotations

import functools
import gc
import json
import importlib.metadata
import math
import os
import random
from pathlib import Path

from ..config import ROOT, load_spec
from ..provenance import Ledger

CKPT = Path(os.environ.get("LANGSAFE_RUNS_DIR", ROOT / "runs")) / "checkpoints"
SAFETUNE_COMMIT = "3901a9f"


def _finite_knob(name: str, value: float, lower: float = 0, upper: float | None = None):
    if not math.isfinite(value) or value < lower or (upper is not None and value > upper):
        raise ValueError(f"invalid {name}: {value}")


def _compatible(target, *donors, keys: list[str] | None = None) -> list[str]:
    """Fail before mutation instead of letting SafeTune skip mismatched weights."""
    target_state = target.state_dict()
    selected = list(target_state) if keys is None else keys
    if not selected:
        raise ValueError("no parameters selected for repair")
    for donor in donors:
        state = donor.state_dict()
        if keys is None and set(state) != set(target_state):
            raise ValueError("repair checkpoints have different state-dict keys")
        for key in selected:
            if key not in state:
                raise ValueError(f"repair donor is missing parameter {key}")
            if target_state[key].shape != state[key].shape:
                raise ValueError(f"repair tensor shape mismatch for {key}: "
                                 f"{tuple(target_state[key].shape)} != {tuple(state[key].shape)}")
    return selected


def _circuit_keys(model, layers: list[int], modules: list[str]) -> list[str]:
    keys = []
    matched_layers, matched_modules = set(), set()
    for key in model.state_dict():
        if not key.endswith(".weight"):
            continue
        layer = next((int(part) for part in key.split(".") if part.isdigit()), None)
        if layer in layers and any(module in key for module in modules):
            keys.append(key)
            matched_layers.add(layer)
            matched_modules.update(module for module in modules if module in key)
    if not keys or matched_layers != set(layers) or matched_modules != set(modules):
        raise ValueError(f"no parameters match circuit layers {layers} and modules {modules}")
    return keys


def _load_cpu(name: str):
    import torch
    from safetune.runner.utils.model_utils import load_model_cpu

    from ..gen.model import resolve_model

    return load_model_cpu(resolve_model(name), dtype=torch.float16)


def edit_stats(model, original) -> dict:
    """How much was edited: comparing arms is only fair if the size and location of each edit are on record."""
    before = original.state_dict()
    total = weights = 0.0
    tensors, layers, seen = 0, set(), set()
    for key, value in model.state_dict().items():
        if value.data_ptr() in seen:        # tied weights share storage: count lm_head and the embedding once
            continue
        seen.add(value.data_ptr())
        old = before[key].float()
        change = float((value.float() - old).pow(2).sum())
        weights += float(old.pow(2).sum())
        if change > 0:
            total += change
            tensors += 1
            layers.update(int(part) for part in key.split(".") if part.isdigit())
    return {"edit_l2": math.sqrt(total), "relative_edit_l2": math.sqrt(total / weights) if weights else 0.0,
            "edited_tensors": tensors, "edited_layers": sorted(layers)}


def _save(model, tok_name: str, out: Path, method: str, inputs: list[str], params: dict) -> Path:
    from safetune.provenance import input_entry, write_provenance
    from transformers import AutoTokenizer

    params = {**params, **edit_stats(model, _load_cpu(inputs[0]))}
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    from ..gen.model import resolve_model

    AutoTokenizer.from_pretrained(resolve_model(tok_name)).save_pretrained(out)
    runtime = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetune"):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = None
    params = {**params, "safetune_commit": SAFETUNE_COMMIT, "runtime": runtime}
    write_provenance(out, f"langsafe.{method}", inputs=[input_entry("model", resolve_model(r)) for r in inputs],
                     base_model=inputs[0], params=params)
    try:
        out_ref = str(out.relative_to(ROOT))
    except ValueError:
        out_ref = str(out)
    Ledger.default().append("repair", {"method": method, "out": out_ref, "inputs": inputs,
                                       "params": params})
    return out


def resta(drifted: str, alpha: float, family: str) -> Path:
    from safetune.recover import apply_resta

    _finite_knob("alpha", alpha)
    m = load_spec()["models"]
    model = _load_cpu(drifted)
    base, aligned = _load_cpu(m["base"]), _load_cpu(m["reference"])
    _compatible(model, base, aligned)
    apply_resta(model, base=base, aligned=aligned, alpha=alpha, dare=False, device="cpu")
    del base, aligned
    gc.collect()
    return _save(model, drifted, CKPT / f"{family}-resta-a{alpha}", "resta",
                 [drifted, m["base"], m["reference"]], {"alpha": alpha, "dare": False, "device": "cpu"})


def circuit(drifted: str, scores_json: str, strength: float, family: str, top_fraction: float = 0.2,
            vector: str = "safety", random_seed: int | None = None) -> Path:
    """C-ΔΘ restricted to the layers CircuitKIT ranked highest (its *_scores.json, read by SafeTune's adapter).

    vector: `safety` is Global - Base (the ReSta vector); `drift` is Global - Fire (the merge-back vector).
    random_seed: the control. Same vector and strength on as many randomly chosen layers outside the circuit.
    """
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    from safetune.recover import apply_ctheta

    _finite_knob("strength", strength)
    if not math.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if vector not in ("safety", "drift"):
        raise ValueError(f"unknown repair vector: {vector}")
    m = load_spec()["models"]
    scores_path = Path(scores_json).expanduser()
    if not scores_path.is_absolute():
        scores_path = ROOT / scores_path
    scores_path = scores_path.resolve()
    scores_data = json.loads(scores_path.read_text(encoding="utf-8"))
    ranked_layers = None
    if scores_data.get("node_scores"):
        from safetune.core.circuit_kit.adapter import circuit_info_from_node_scores
        from ..circuits.discover import layer_mass, node_scores, top_layers

        # CircuitKIT interop fields are already selected at its default 20%.
        # SafeTune otherwise ignores top_fraction when those fields exist.
        values = node_scores(scores_path)
        info = circuit_info_from_node_scores(values, top_fraction=top_fraction, raw_output_path=str(scores_path))
        ranked_layers = top_layers(values, top_fraction)
    else:
        info = load_circuit_info_from_file(str(scores_path), top_fraction=top_fraction)
    if info is None:
        raise ValueError(f"no usable circuit info in {scores_json}")
    suggestions = getattr(info, "layer_suggestions", None)
    layers = ranked_layers or list(getattr(suggestions, "layer_subset", None) or [])
    modules = list(getattr(suggestions, "target_modules", None) or [])
    if not layers or any(type(layer) is not int or layer < 0 for layer in layers):
        raise ValueError("circuit must specify a nonempty valid layer subset")
    if not modules or any(not isinstance(module, str) or not module.strip() for module in modules):
        raise ValueError("circuit must specify target modules")
    circuit_layers = layers
    if random_seed is not None:
        if ranked_layers is None:
            raise ValueError("the random-layer control needs CircuitKIT node scores")
        outside = sorted(set(layer_mass(values)) - set(layers))
        if len(outside) < len(layers):
            raise ValueError("too few non-circuit layers for a matched random control")
        layers = sorted(random.Random(random_seed).sample(outside, len(layers)))
    model = _load_cpu(drifted)
    keys = _circuit_keys(model, layers, modules)
    negative_name = m["base"] if vector == "safety" else drifted
    aligned = _load_cpu(m["reference"])
    negative = _load_cpu(negative_name) if vector == "safety" else model     # apply_ctheta writes only at the end
    _compatible(model, aligned, negative, keys=keys)
    apply_ctheta(model, aligned, negative, info, strength=strength, layer_subset=layers, target_modules=modules)
    del aligned, negative
    gc.collect()
    from ..quant.gguf import file_sha256

    params = {"strength": strength, "top_fraction": top_fraction, "scores": str(scores_path),
              "scores_sha256": file_sha256(scores_path), "matched_parameters": len(keys), "vector": vector,
              "layer_subset": layers, "circuit_layers": circuit_layers, "random_seed": random_seed,
              "target_modules": modules}
    return _save(model, drifted, circuit_checkpoint(family, strength, top_fraction, vector, random_seed), "ctheta",
                 [drifted, m["reference"], negative_name], params)


def circuit_checkpoint(family: str, strength: float, top_fraction: float, vector: str = "safety",
                       random_seed: int | None = None) -> Path:
    name = "circuit" if vector == "safety" else "circuit_merge"
    control = "" if random_seed is None else f"-rand{random_seed}"
    return CKPT / f"{family}-{name}-s{strength}-top{top_fraction}{control}"


def merge_back(drifted: str, lam: float, family: str) -> Path:
    from safetune.recover import apply_wise_ft

    _finite_knob("lambda", lam, upper=1)
    m = load_spec()["models"]
    model = _load_cpu(drifted)
    aligned = _load_cpu(m["reference"])
    _compatible(model, aligned)
    apply_wise_ft(model, aligned, alpha=lam)
    return _save(model, drifted, CKPT / f"{family}-merge-l{lam}", "wise_ft", [drifted, m["reference"]], {"lambda": lam})


# ---- steering (inference time) ------------------------------------------------------------------------------------

class _PreambleTokenizer:
    """Keep SafeTune's internal chat rendering on the requested experimental condition."""

    def __init__(self, tokenizer, preamble: str):
        self.tokenizer, self.preamble = tokenizer, preamble

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    @property
    def padding_side(self):
        return self.tokenizer.padding_side

    @padding_side.setter
    def padding_side(self, value):
        self.tokenizer.padding_side = value

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        from ..gen.generate import render_prompt

        if len(messages) != 1 or messages[0]["role"] != "user" or not add_generation_prompt:
            raise ValueError("steering extraction expects one user prompt and a generation prefix")
        rendered = render_prompt(self.tokenizer, messages[0]["content"], self.preamble)
        return self.tokenizer.encode(rendered, add_special_tokens=False) if tokenize else rendered


def _model_dimensions(model) -> tuple[int, int]:
    config = model.config
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return int(config.num_hidden_layers), int(config.hidden_size)


def _validate_direction(model, direction, layer: int):
    import torch

    n_layers, hidden = _model_dimensions(model)
    if type(layer) is not int or not 0 <= layer < n_layers:
        raise ValueError(f"steering layer {layer} is outside the model's {n_layers} layers")
    if direction.ndim != 1 or direction.numel() != hidden:
        raise ValueError(f"steering direction must match model hidden width {hidden}")
    if not torch.isfinite(direction).all() or float(direction.norm()) <= 1e-8:
        raise ValueError("steering direction must be finite and nonzero")


def extract_direction(model, tok, harmful: list[str], harmless: list[str], family: str, pick_layer: int = 24,
                      preamble: str = "default") -> Path:
    """Refusal direction from repair-split prompts. pick_layer=24 of 36 follows SafeTune's Tiny Aya note."""
    import torch
    from safetune.steer import RefusalDirectionConfig, extract_refusal_direction

    if preamble not in {"default", "none"}:
        raise ValueError(f"unsupported preamble condition: {preamble}")
    if not harmful or not harmless or any(not text.strip() for text in harmful + harmless):
        raise ValueError("steering extraction needs nonempty harmful and harmless repair prompts")
    n_layers, _ = _model_dimensions(model)
    if type(pick_layer) is not int or not 0 <= pick_layer < n_layers:
        raise ValueError(f"pick_layer must be in [0, {n_layers})")
    forward = getattr(model, "forward", None)
    if forward is not None:       # the extractor reads hidden states; full-sequence logits don't fit beside 8 GB of weights
        model.forward = functools.partial(forward, logits_to_keep=1)
    try:
        direction, layer, _ = extract_refusal_direction(model, _PreambleTokenizer(tok, preamble), harmful, harmless,
            RefusalDirectionConfig(pick_layer=pick_layer, target_layers=[pick_layer], pool_method="last_token",
                                   normalize=True, select_directions=False))
    finally:
        if forward is not None:
            del model.forward
    if layer != pick_layer:
        raise ValueError(f"SafeTune returned layer {layer} instead of requested layer {pick_layer}")
    _validate_direction(model, direction, int(layer))
    out = CKPT / f"{family}-steer-{preamble}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction.detach().cpu(), "layer": int(layer)}, out / "refusal_direction.pt")
    import hashlib
    meta = {"layer": int(layer), "n_harmful": len(harmful), "n_harmless": len(harmless), "preamble": preamble,
            "model": getattr(model.config, "_name_or_path", None), "safetune_commit": SAFETUNE_COMMIT,
            "prompt_sha256": hashlib.sha256(json.dumps([harmful, harmless], ensure_ascii=False).encode()).hexdigest()}
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    Ledger.default().append("steer-extract", {"family": family, **meta})
    return out


def steered(model, steer_dir: Path, strength: float):
    """Context manager: inside it, model.generate runs with the refusal direction added at the chosen layer."""
    import torch
    from safetune.steer import RefusalDirectionModel

    _finite_knob("strength", strength)
    blob = torch.load(Path(steer_dir) / "refusal_direction.pt", weights_only=True, map_location="cpu")
    _validate_direction(model, blob["direction"], blob["layer"])
    return RefusalDirectionModel(model, blob["direction"].to(model.device), mode="steer", strength=strength,
                                 layers=[blob["layer"]])
