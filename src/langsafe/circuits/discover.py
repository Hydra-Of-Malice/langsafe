"""CircuitKIT stage: EAP-IG refusal-circuit discovery per (model, language), faithfulness, and cross-language overlap.

Needs ~16 GB of accelerator memory: a Kaggle/Colab T4, or Apple Silicon with enough unified memory (MPS).
An 8 GB card does not fit Tiny Aya under TransformerLens. `LANGSAFE_DEVICE=cpu` is the slow fallback.
CircuitKIT notes that matter here:
- `discover` returns the nodes to PRUNE; the circuit is the top-scoring nodes in the *_scores.json it writes;
- reload the model between runs (discovery mutates model.cfg flags);
- the custom-task cache is keyed by task name, so the name includes model, language and pair-file hash;
- only the Tiny Aya base checkpoint is parity-tested upstream, so we run the parity check on Global and Fire first.
"""
from __future__ import annotations

import hashlib
import gc
import json
import math
import os
import re
from pathlib import Path

from ..config import ROOT
from ..provenance import Ledger

OUT = Path(os.environ.get("LANGSAFE_RUNS_DIR", ROOT / "runs")) / "circuits"
PILLARS = ["patching", "ablation", "baselines", "stability"]   # robustness pillar is not meaningful for signed metrics


def discover(model_id: str, model_tag: str, lang: str, task_csv: Path, n_examples: int = 64, ig_steps: int = 3,
             precision: str = "float16", sparsity: float = 0.3, seed: int = 0,
             verify_parity: bool = True, parity_result: dict | None = None,
             pillars: list[str] | None = None, n_stability_runs: int = 3) -> dict:
    import circuitkit  # noqa: F401  (must be imported before transformer_lens)
    from circuitkit.pipeline import Pipeline

    if n_examples < 1 or ig_steps < 1 or not 0 <= sparsity <= 1:
        raise ValueError("discovery requires positive example/IG counts and sparsity in [0, 1]")
    from ..gen.model import best_device, resolve_model

    # An 8 GB card does not fit Tiny Aya under TransformerLens, so discovery can run on the CPU by itself.
    device = os.environ.get("LANGSAFE_CIRCUIT_DEVICE") or best_device()
    if device == "cpu" and precision == "float16":
        precision = "float32"
    task_csv = Path(task_csv).resolve()
    digest = hashlib.sha256(task_csv.read_bytes()).hexdigest()[:8]
    settings = {"precision": precision, "n_examples": n_examples, "ig_steps": ig_steps,
                "sparsity": sparsity, "seed": seed, "data_max_records": n_examples,
                "pillars": PILLARS if pillars is None else pillars, "n_stability_runs": n_stability_runs}
    setting_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]
    task_name = f"refusal_{lang}__{model_tag}__{digest}_{setting_hash}"
    out_dir = OUT / f"{model_tag}__{lang}" / f"{digest}-{setting_hash}"

    parity = parity_result
    if parity is not None and parity.get("model") != model_id:
        raise ValueError("parity result belongs to a different model")
    if parity is None and verify_parity:
        parity = parity_check(model_id)
    if verify_parity and not (parity or {}).get("passed"):
        raise RuntimeError(f"CircuitKIT/HF next-token parity failed: {parity}")

    # Every invocation constructs a fresh pipeline/model. The pinned toolkit
    # mutates cfg hook/GQA flags during discovery; never share it across jobs.
    pipe = Pipeline.from_custom_data(
        resolve_model(model_id), str(task_csv), clean_prompt="{harmful}", clean_answer="{refuse}",
        corrupt_prompt="{harmless}", corrupt_answer="{comply}", task_name=task_name,
        precision=precision, output_dir=str(out_dir), device=device)
    try:
        # Pinned CircuitKIT's NormalizedTaskSpec ignores data_params.num_examples
        # for EAP. Its template loader does honor data.max_records, before
        # token-length alignment. Pin both budgets; settings also enter the task
        # cache key so a different cap cannot reuse an older normalized CSV.
        if not isinstance(getattr(pipe, '_custom_data_cfg', None), dict):
            raise RuntimeError('Pinned CircuitKIT custom-data configuration is unavailable')
        pipe._custom_data_cfg['max_records'] = n_examples
        pipe.discover(algorithm="eap-ig", level="node", sparsity=sparsity, n_examples=n_examples, batch_size=1,
                      ig_steps=ig_steps, seed=seed)
        artifact = Path(pipe.artifact_path)
        scores = artifact.with_name(f"{artifact.stem}_scores.json")
        values = node_scores(scores)
        if not any(value != 0 for value in values.values()):
            raise ValueError("CircuitKIT produced only zero node scores")
        if not any(re.fullmatch(r"A\d+\.\d+|MLP \d+|L\d+\.\d+", key) for key in values):
            raise ValueError("CircuitKIT scores contain no usable layers")
        pipe.evaluate(pillars=PILLARS if pillars is None else pillars, n_examples=n_examples,
                      n_stability_runs=n_stability_runs)
        report = pipe.report  # evaluate() returns self in CircuitKIT e71d5e1.
        if report is None:
            raise ValueError("CircuitKIT returned no faithfulness report")
        try:
            scores_ref = str(scores.relative_to(ROOT))
        except ValueError:
            scores_ref = str(scores)
        res = {"model": model_id, "resolved_model": resolve_model(model_id), "model_tag": model_tag,
               "lang": lang, "task": task_name, "scores_json": scores_ref,
               "faithfulness": _plain(report), "parity": parity or {"passed": False, "status": "not_run"},
               "settings": settings, "circuitkit_commit": "e71d5e1"}
        (out_dir / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False),
                                              encoding="utf-8")
    finally:
        del pipe
        _release_memory()
    Ledger.default().append("circuit-discover", {k: res[k] for k in ("model", "lang", "task", "scores_json")})
    return res


def _release_memory():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def parity_check(model_id: str, prompts: list[str] | None = None, device: str = "cpu",
                 kl_tolerance: float = 1e-4, max_logprob_tolerance: float = 5e-3) -> dict:
    """Apply upstream's float32 next-token distribution gate to the actual checkpoint.

    CircuitKIT e71d5e1 exposes no parity helper. Its real-weight test compares
    log_softmax (rather than fold-dependent raw logits); Global/Fire need this
    check in addition to the upstream Base-only test. HF weights are reused
    during TL conversion, then both copies are released before discovery.
    """
    import circuitkit  # noqa: F401  patches must precede TransformerLens.
    import torch
    import torch.nn.functional as functional
    from circuitkit.quick import load_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..gen.model import resolve_model

    prompts = prompts or ["The capital of France is", "2 + 2 =", "Water boils at a temperature of"]
    resolved = resolve_model(model_id)
    hf = tl = None
    try:
        hf = AutoModelForCausalLM.from_pretrained(resolved, dtype=torch.float32, device_map="cpu",
                                                use_safetensors=True)
        hf.eval()
        tokenizer = AutoTokenizer.from_pretrained(resolved)
        tl = load_model(resolved, dtype="float32", device=device, hf_model=hf,
                        fold_ln=False, center_writing_weights=False, center_unembed=False,
                        fold_value_biases=False)
        tl.eval()
        rows = []
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                hf_logits = hf(ids.to(next(hf.parameters()).device), logits_to_keep=1).logits[0, -1].float().cpu()
                tl_logits = tl(ids.to(next(tl.parameters()).device))[0, -1].float().cpu()
            if hf_logits.shape != tl_logits.shape:
                raise ValueError("HF/TL parity vocab shapes differ")
            hf_lp = functional.log_softmax(hf_logits, dim=-1)
            tl_lp = functional.log_softmax(tl_logits, dim=-1)
            kl = float(functional.kl_div(tl_lp, hf_lp, log_target=True, reduction="sum"))
            deviation = float((hf_lp - tl_lp).abs().max())
            rows.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "kl": kl,
                         "max_logprob_deviation": deviation,
                         "argmax_agrees": int(hf_lp.argmax()) == int(tl_lp.argmax())})
        passed = all(math.isfinite(row["kl"]) and math.isfinite(row["max_logprob_deviation"])
                     and row["kl"] < kl_tolerance and row["max_logprob_deviation"] < max_logprob_tolerance
                     and row["argmax_agrees"] for row in rows)
        return {"model": model_id, "resolved_model": resolved, "passed": passed, "dtype": "float32",
                "device": device, "kl_tolerance": kl_tolerance, "max_logprob_tolerance": max_logprob_tolerance,
                "per_prompt": rows, "circuitkit_commit": "e71d5e1"}
    finally:
        del hf, tl
        _release_memory()


def _plain(x):
    if hasattr(x, "to_dict"):
        return _plain(x.to_dict())
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "tolist"):
        return _plain(x.tolist())
    if hasattr(x, "__dict__"):
        return {k: _plain(v) for k, v in vars(x).items() if not k.startswith("_")}
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if hasattr(x, "item"):
        return _plain(x.item())
    return x


# ---- overlap across languages and models -------------------------------------------------------------------------

def node_scores(scores_json: str | Path) -> dict[str, float]:
    path = Path(scores_json).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    scores = json.loads(path.read_text(encoding="utf-8"))["node_scores"]
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"no node scores in {path}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in scores.values()):
        raise ValueError(f"non-finite or invalid node scores in {path}")
    return scores


_NODE_LAYER = re.compile(r"(?:A|MLP |L)(\d+)")


def layer_mass(scores: dict[str, float]) -> dict[int, float]:
    """Total absolute circuit score per layer. Node names: A<layer>.<head>, MLP <layer>, L<layer>.<neuron>."""
    mass: dict[int, float] = {}
    for name, value in scores.items():
        match = _NODE_LAYER.match(name)
        if match:
            layer = int(match.group(1))
            mass[layer] = mass.get(layer, 0.0) + abs(value)
    if not mass:
        raise ValueError("no node name identifies a layer")
    return mass


def top_layers(scores: dict[str, float], fraction: float) -> list[int]:
    """The top `fraction` of layers by circuit mass.

    SafeTune's node-level mask keeps every layer holding any top-20% node. Those nodes are spread through the model,
    so on Tiny Aya it selected 30 of 36 layers, which is not a localised edit.
    """
    mass = layer_mass(scores)
    count = max(1, round((max(mass) + 1) * fraction))
    return sorted(sorted(mass, key=lambda layer: (-mass[layer], layer))[:count])


def top_k(scores: dict[str, float], k: int) -> set[str]:
    return set(sorted(scores, key=scores.get, reverse=True)[:k])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    import numpy as np

    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return float("nan")
    ra = np.argsort(np.argsort([a[k] for k in keys]))
    rb = np.argsort(np.argsort([b[k] for k in keys]))
    return float(np.corrcoef(ra, rb)[0, 1])


def overlap_matrix(circuits: dict[str, str | Path], k: int = 50) -> dict:
    """circuits: {"global/hi": scores_json, "fire/hi": ...} -> pairwise top-k Jaccard and full-ranking Spearman."""
    s = {name: node_scores(p) for name, p in circuits.items()}
    names = sorted(s)
    def rank(a, b):
        value = spearman(s[a], s[b])
        return value if math.isfinite(value) else None

    return {"k": k,
            "jaccard": {a: {b: jaccard(top_k(s[a], k), top_k(s[b], k)) for b in names} for a in names},
            "spearman": {a: {b: rank(a, b) for b in names} for a in names}}


def merge_scores(paths: list[str | Path], out: Path, model: str) -> Path:
    """Average node scores across languages into one scores.json SafeTune can read (node_scores-only form)."""
    if not paths:
        raise ValueError("at least one circuit score file is required")
    acc: dict[str, float] = {}
    for p in paths:
        for n, v in node_scores(p).items():
            acc[n] = acc.get(n, 0.0) + v / len(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model, "algorithm": "eap-ig", "level": "node", "node_scores": acc,
                               "metadata": {"merged_from": [str(p) for p in paths]}}, indent=2), encoding="utf-8")
    return out
