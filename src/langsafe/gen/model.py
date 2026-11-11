"""Load Tiny Aya (full precision, NF4, or with a LoRA/steering adapter) the same way for every arm.

Runs on CUDA (Kaggle/Colab T4, RTX laptops), Apple Silicon (MPS) or CPU. NF4 needs CUDA (bitsandbytes);
on a Mac the 4-bit check uses GGUF through llama.cpp instead (see quant/gguf.py).
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_model(model_id: str) -> str:
    """Use a local copy when `LANGSAFE_MODELS` points at a folder holding e.g. `tiny-aya-fire/` (safetensors, as
    downloaded with `hf download CohereLabs/tiny-aya-fire --local-dir ...`). Otherwise the Hub id (and HF cache)."""
    root = os.environ.get("LANGSAFE_MODELS")
    if root:
        cand = Path(root).expanduser() / model_id.split("/")[-1]
        if (cand / "config.json").is_file():
            return str(cand)
    return model_id


def best_device() -> str:
    """`LANGSAFE_DEVICE` wins; otherwise cuda, then mps, then cpu."""
    import torch

    if os.environ.get("LANGSAFE_DEVICE"):
        return os.environ["LANGSAFE_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def validate_nf4_model(model) -> None:
    """Reject a 4-bit load that silently fell back to CPU/disk or another quantizer."""
    loaded_in_4bit = bool(getattr(model, "is_loaded_in_4bit", False))
    quantizer = getattr(model, "hf_quantizer", None)
    config = (getattr(quantizer, "quantization_config", None)
              or getattr(model, "quantization_config", None))
    if config is not None:
        quant_type = str(getattr(config, "bnb_4bit_quant_type", "")).lower()
        if quant_type and quant_type != "nf4":
            raise RuntimeError(f"NF4 requested but loaded quantizer type is {quant_type!r}")
        loaded_in_4bit = loaded_in_4bit or (
            bool(getattr(config, "_load_in_4bit", getattr(config, "load_in_4bit", False)))
            and quant_type == "nf4"
        )
    if not loaded_in_4bit:
        raise RuntimeError("NF4 requested but Transformers did not load a 4-bit model")
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        # Transformers 5 keeps no map for a single-device load, so check where each tensor actually is.
        named = getattr(model, "named_parameters", None)
        device_map = {name: tensor.device for name, tensor in named()} if named else {}
    if not device_map:
        raise RuntimeError("NF4 model has no verifiable CUDA device map")
    offloaded = []
    for name, device in device_map.items():
        if isinstance(device, int):
            on_cuda = device == 0
        else:
            on_cuda = str(device).lower() in {"cuda", "cuda:0"}
        if not on_cuda:
            offloaded.append(f"{name or '<root>'}={device}")
    if offloaded:
        raise RuntimeError("NF4 model has non-CUDA placement (CPU/disk offload is forbidden): " + ", ".join(offloaded))


def load(model_id: str, dtype: str = "float16", quant: str | None = None, adapter: str | None = None,
         device_map: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = best_device()
    if quant == "nf4" and (dev != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("NF4 (bitsandbytes) needs a CUDA GPU; refusing a CPU/MPS fallback")
    if quant == "nf4" and device_map is not None and device_map != {"": 0}:
        raise RuntimeError("NF4 requires the strict single-CUDA device map {'': 0}; CPU/disk offload is forbidden")
    if device_map is None:
        # Auto placement can silently spill 4-bit layers onto host memory.  A
        # strict CUDA map makes insufficient VRAM fail at load time.
        device_map = {"": 0} if quant == "nf4" else ("auto" if dev == "cuda" else {"": dev})
    if dev == "cpu" and dtype == "float16":
        dtype = "float32"          # fp16 matmuls are slow or unsupported on CPU
    kwargs = {"device_map": device_map, "dtype": getattr(torch, dtype)}
    if quant == "nf4":
        if dev != "cuda":
            raise RuntimeError("NF4 (bitsandbytes) needs a CUDA GPU; on a Mac use the GGUF check (quant/gguf.py)")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=getattr(torch, dtype),
                                                           bnb_4bit_use_double_quant=False)
    elif quant:
        raise ValueError(f"unsupported in-process quantisation {quant!r} (GGUF runs through llama.cpp, see quant/)")
    path = resolve_model(model_id)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if quant == "nf4":
        validate_nf4_model(model)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok
