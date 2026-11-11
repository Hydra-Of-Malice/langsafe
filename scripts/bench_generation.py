"""How fast can this GPU generate at full precision? Times Transformers and llama-server on the same prompts.

    python scripts/bench_generation.py --gguf cache/gguf/tiny-aya-earth-f16.gguf

Uses repair-split MultiJail prompts with the default preamble and prints speed and how often the two backends produce
the same greedy tokens. Only numbers are written, to runs/pilot/bench-generation.json; no model output is kept.
Stop other GPU jobs first, or the timings mean nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("LANGSAFE_RUNS_DIR", str(ROOT / "runs" / "pilot"))
os.environ.setdefault("LANGSAFE_DATA_DIR", str(ROOT / "runs" / "pilot" / "data"))

from langsafe.gen.generate import render_prompt                                    # noqa: E402
from langsafe.llm_client import llama_server_binary                                # noqa: E402
from pilot_multijail import items                                                  # noqa: E402

OUT = ROOT / "runs" / "pilot" / "bench-generation.json"
PORT = 8123


def prompts_for(tokenizer, count: int) -> list[list[int]]:
    picked = items()
    step = max(1, len(picked) // count)
    texts = [render_prompt(tokenizer, p["text"], "default") for p in picked[::step][:count]]
    return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


def transformers_run(model_id: str, prompts: list[list[int]], sizes: list[int], max_new: int, results: dict) -> dict:
    import torch
    from langsafe.gen.model import load

    model, tokenizer = load(model_id, dtype="float16")
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    tokenizer.padding_side = "left"
    answers = {}
    for size in sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        generated, started, row = 0, time.perf_counter(), {"batch": size}
        try:
            for start in range(0, len(prompts), size):
                batch = prompts[start:start + size]
                enc = tokenizer.pad({"input_ids": batch}, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                         pad_token_id=tokenizer.pad_token_id)
                for index, seq in enumerate(out[:, enc["input_ids"].shape[1]:].tolist()):
                    stop = next((j + 1 for j, token in enumerate(seq) if token in eos), len(seq))
                    generated += stop
                    answers.setdefault(size, {})[start + index] = seq[:stop]
            seconds = time.perf_counter() - started
            row.update(seconds=round(seconds, 1), tokens=generated, tokens_per_second=round(generated / seconds, 1),
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        except torch.OutOfMemoryError:
            row["failed"] = "out of memory"
        print("transformers", row, flush=True)
        results["transformers"].append(row)
        OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    good = [size for size in sizes if len(answers.get(size, {})) == len(prompts)]
    return answers[good[-1]] if good else {}


def post(body: dict) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def server_run(gguf: Path, prompts: list[list[int]], slots: int, slot_tokens: int, max_new: int, results: dict) -> dict:
    command = [llama_server_binary(), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(PORT), "-ngl", "99",
               "-np", str(slots), "-c", str(slots * slot_tokens)]
    log = open(ROOT / "runs" / "pilot" / "bench-server.log", "ab")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    row, answers = {"slots": slots, "slot_tokens": slot_tokens}, {}
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError("server exited while loading")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, ValueError):
                time.sleep(1)

        def one(index: int) -> int:
            body = {"prompt": prompts[index], "n_predict": max_new, "temperature": 0, "top_k": 1, "seed": 0,
                    "cache_prompt": True, "return_tokens": True}
            reply = post(body)
            answers[index] = reply.get("tokens") or []
            return reply["tokens_predicted"]

        started = time.perf_counter()
        with ThreadPoolExecutor(slots) as pool:
            generated = sum(pool.map(one, range(len(prompts))))
        seconds = time.perf_counter() - started
        row.update(seconds=round(seconds, 1), tokens=generated, tokens_per_second=round(generated / seconds, 1))
    except Exception as error:                     # a slot count that doesn't fit is a result, not a crash
        row["failed"] = repr(error)[:200]
    finally:
        server.terminate()
        server.wait()
        log.close()
    print("llama-server", row, flush=True)
    results["llama_server"].append(row)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return answers


def agreement(reference: dict, other: dict, eos: int) -> dict:
    """How long the two greedy answers stay identical, as a share of the reference answer."""
    pairs = [(reference[i], [t for t in other[i] if t != eos]) for i in reference if other.get(i)]
    pairs = [([t for t in a if t != eos], b) for a, b in pairs]
    if not pairs:
        return {"n": 0}
    shared = [next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) for a, b in pairs]
    return {"n": len(pairs), "identical": sum(a == b for a, b in pairs) / len(pairs),
            "mean_shared_prefix": sum(shared) / len(shared), "mean_length": sum(len(a) for a, _ in pairs) / len(pairs)}


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CohereLabs/tiny-aya-earth")
    parser.add_argument("--gguf", default="cache/gguf/tiny-aya-earth-f16.gguf")
    parser.add_argument("--prompts", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--batches", default="2,8,16")
    parser.add_argument("--slots", default="4,8,12,16")
    parser.add_argument("--slot-tokens", type=int, default=768)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = prompts_for(tokenizer, args.prompts)
    results = {"model": args.model, "gguf": args.gguf, "prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
               "mean_prompt_tokens": sum(map(len, prompts)) / len(prompts), "transformers": [], "llama_server": []}
    reference = {}
    if args.batches:
        reference = transformers_run(args.model, prompts, [int(b) for b in args.batches.split(",")], args.max_new_tokens, results)
    answers = {}
    for slots in [int(s) for s in args.slots.split(",")]:
        answers = server_run(ROOT / args.gguf, prompts, slots, args.slot_tokens, args.max_new_tokens, results) or answers
    if reference and answers:
        results["agreement"] = agreement(reference, answers, tokenizer.eos_token_id)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
