"""Capability axis: Global-MMLU-Lite accuracy by next-token letter scoring (no generation, deterministic).

The question and four options go through the same chat template (default preamble) as the safety prompts; the
model's next-token logits over the letters A-D decide the answer.
"""
from __future__ import annotations

from ..gen.generate import render_prompt
from ..languages import GMMLU_CONFIGS

TEMPLATE = "{q}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer with only the letter (A, B, C or D)."


def load_items(lang: str, limit: int | None = None, repo: str = "CohereLabs/Global-MMLU-Lite",
               split: str = "test") -> list[dict]:
    """`test` measures capability; `dev` is reserved for knob selection, so tuning never sees the test items."""
    from datasets import load_dataset

    ds = load_dataset(repo, GMMLU_CONFIGS[lang], split=split)
    items = [dict(r) for r in ds]
    return items[:limit] if limit else items


def _letter_ids(tok) -> list[int]:
    ids = []
    for L in "ABCD":
        toks = tok.encode(L, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"letter {L} is not a single token for this tokenizer: {toks}")
        ids.append(toks[0])
    return ids


def accuracy(model, tok, items: list[dict], batch_size: int = 8, preamble: str = "default") -> dict:
    import torch
    import inspect

    if not items or batch_size <= 0:
        raise ValueError("Capability evaluation requires items and positive batch size")
    forward = getattr(model, "forward", model.__call__)
    projection = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(forward).parameters else {}

    letters = _letter_ids(tok)
    tok.padding_side = "left"
    correct = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        texts = [render_prompt(tok, TEMPLATE.format(q=r["question"], a=r["option_a"], b=r["option_b"],
                                                    c=r["option_c"], d=r["option_d"]), preamble) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**enc, **projection).logits[:, -1, :]
        picks = logits[:, letters].argmax(dim=-1).tolist()
        correct += [int("ABCD"[k] == r["answer"].strip().upper()) for k, r in zip(picks, batch)]
    return {"accuracy": sum(correct) / len(correct), "n": len(correct), "per_item": correct}
