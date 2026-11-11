"""4-bit GGUF (Q4_K_M) verification through llama.cpp. NF4 runs in-process (gen/model.py --quant nf4).

    python llama.cpp/convert_hf_to_gguf.py runs/checkpoints/fire-circuit-... --outtype f16 --outfile x-f16.gguf
    llama.cpp/build/bin/llama-quantize x-f16.gguf x-q4_k_m.gguf Q4_K_M
    llama.cpp/build/bin/llama-server -m x-q4_k_m.gguf --port 8080 -c 4096

Prompts use the same Hugging Face chat template, token IDs, preamble condition, and generation EOS as the
full-precision runs. The llama.cpp backend and quantized weights can both introduce numerical differences.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import urllib.error
import urllib.request

from ..config import ROOT
from ..data.records import Prompt
from ..gen.generate import render_prompt


CONVERSION_PROTOCOL = 'hf_generation_eos_v1'


def _match_generation_eos(checkpoint: Path, output: Path) -> None:
    """Keep the exported stopping token consistent with HF model.generate.

    Tiny Aya's tokenizer names EOT (6) as EOS, while its generation config
    stops at EOS (3). The converter follows the tokenizer; allowing that
    mismatch makes llama.cpp generate past the answer into repeated text.
    Change only the temporary export, never the source checkpoint.
    """
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    generation_path = checkpoint / 'generation_config.json'
    generation = json.loads(generation_path.read_text(encoding='utf-8')) if generation_path.is_file() else {}
    eos = generation.get('eos_token_id')
    if eos is None:
        eos = config.get('eos_token_id')
    if eos is None:
        return
    if isinstance(eos, list) and len(eos) == 1:
        eos = eos[0]
    if type(eos) is not int or eos < 0:
        raise ValueError('GGUF export currently requires one nonnegative HF generation EOS token ID')
    from gguf import GGUFReader, GGUFValueType
    reader = GGUFReader(output, 'r+')
    field = reader.get_field('tokenizer.ggml.eos_token_id')
    tokens = reader.get_field('tokenizer.ggml.tokens')
    if field is None or field.types != [GGUFValueType.UINT32] or tokens is None or eos >= len(tokens.data):
        raise ValueError('GGUF metadata cannot represent the checkpoint generation EOS token')
    previous = int(field.contents())
    if previous != eos:
        field.parts[field.data[0]][0] = eos
        reader.data.flush()
        print(f'GGUF generation EOS matched to HF checkpoint: {previous} -> {eos}', flush=True)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Identify the complete weight file with bounded memory use."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_hf_to_gguf(checkpoint: str | Path, out: str | Path, llama_cpp_dir: str | Path,
                       outtype: str = "f16", python: str | None = None, overwrite: bool = False) -> Path:
    """Convert a local HF checkpoint using a caller-provided llama.cpp checkout."""
    checkpoint, out, kit = Path(checkpoint), Path(out), Path(llama_cpp_dir)
    script = kit / "convert_hf_to_gguf.py"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"not a local Hugging Face checkpoint: {checkpoint}")
    if not script.is_file():
        raise FileNotFoundError(f"llama.cpp converter is missing: {script}")
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + '.partial')
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run([python or sys.executable, str(script), str(checkpoint), "--outtype", outtype,
                        "--outfile", str(temporary)], check=True)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(f"llama.cpp conversion produced no GGUF: {out}")
        _match_generation_eos(checkpoint, temporary)
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def quantize_gguf(source: str | Path, out: str | Path, llama_cpp_dir: str | Path,
                  quantization: str = "Q4_K_M", overwrite: bool = False) -> Path:
    """Quantize through llama-quantize; no inference server is started here."""
    source, out, kit = Path(source), Path(out), Path(llama_cpp_dir)
    executable = kit / "build" / "bin" / "llama-quantize"
    if not executable.exists() and executable.with_suffix(".exe").exists():
        executable = executable.with_suffix(".exe")
    if not executable.is_file():
        raise FileNotFoundError(f"build llama.cpp's llama-quantize first: {executable}")
    if not source.is_file():
        raise FileNotFoundError(source)
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + '.partial')
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run([str(executable), str(source), str(temporary), quantization], check=True)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(f"llama.cpp quantization produced no GGUF: {out}")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return out


def _post(url: str, body: dict, attempts: int = 5) -> dict:
    """A local connection can drop while the server is healthy. Every call here is safe to repeat; a 4xx is a real answer."""
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts - 1:
                raise
            print(f"llama-server {error.code}, retrying", flush=True)
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            print(f"llama-server connection failed ({type(error).__name__}), retrying", flush=True)
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class GGUFGenerator:
    """Same interface and cache format as gen.generate.Generator, backed by a running llama-server."""

    def __init__(self, tokenizer, decoding: dict, tag: str, gguf_path: str, server: str = "http://127.0.0.1:8080",
                 cache_dir: Path = ROOT / "runs" / "generations"):
        self.tok, self.decoding, self.tag, self.server = tokenizer, decoding, tag, server.rstrip("/")
        self.fp = file_sha256(gguf_path)
        self.cache_path = cache_dir / f"{tag}-{self.fp}.jsonl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as stream:
                for line in stream:
                    r = json.loads(line)
                    self._cache[r["key"]] = r

    def _key(self, p: Prompt, preamble: str) -> str:
        d = json.dumps({"fp": self.fp, "pre": preamble, "dec": self.decoding, "text": p.text,
                        "id": p.id, "split": p.split, "lang": p.lang, "kind": p.kind,
                        "input_ids": self.tok.encode(render_prompt(self.tok, p.text, preamble),
                                                     add_special_tokens=False),
                        "prompt_transport": "exact_hf_ids_v1"}, sort_keys=True)
        return hashlib.sha256(d.encode()).hexdigest()[:24]

    def run(self, prompts: Iterable[Prompt], preamble: str) -> list[dict]:
        prompts = list(prompts)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                k = self._key(p, preamble)
                if k in self._cache:
                    continue
                rendered = render_prompt(self.tok, p.text, preamble)
                started = time.perf_counter()
                res = _post(f"{self.server}/completion", {
                    "prompt": self.tok.encode(rendered, add_special_tokens=False),
                    "n_predict": self.decoding["max_new_tokens"],
                    "temperature": 0, "top_k": 1, "repeat_penalty": self.decoding.get("repetition_penalty", 1.0),
                    "cache_prompt": True})
                r = {"key": k, "id": p.id, "lang": p.lang, "kind": p.kind, "preamble": preamble,
                     "split": p.split, "group": p.group, "model_tag": self.tag, "fingerprint": self.fp,
                     "response": res["content"].strip(), "batch_size": 1,
                     "batch_generation_seconds": time.perf_counter() - started,
                     "generated_tokens_including_special": res.get('tokens_predicted', res.get('timings', {}).get('predicted_n')),
                     "server_timings": res.get('timings', {}), "input_truncated": res.get('truncated', False),
                     "finish_reason": 'length' if res.get('stopped_limit') or res.get('stop_type') == 'limit' else
                                      'eos' if res.get('stopped_eos') or res.get('stop_type') == 'eos' else 'other',
                     "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest()}
                self._cache[k] = r
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if (i + 1) % 10 == 0 or i + 1 == len(prompts):
                    print(f'{self.tag} [{preamble}]: {i + 1}/{len(prompts)} complete', flush=True)
        return [self._cache[self._key(p, preamble)] for p in prompts]


def _candidate_setup(tokenizer, server: str) -> tuple[list[int], list[list]]:
    from ..eval.capability import _letter_ids

    letters = _letter_ids(tokenizer)
    if len(set(letters)) != 4:
        raise ValueError("A/B/C/D do not have four distinct token IDs")
    vocab_size = len(tokenizer)
    if not all(0 <= token < vocab_size for token in letters):
        raise ValueError("letter token IDs are outside the tokenizer vocabulary")
    for letter, token in zip("ABCD", letters):
        got = _post(f"{server}/tokenize", {"content": letter, "add_special": False, "parse_special": False})
        if got.get("tokens") != [token]:
            raise ValueError(f"llama.cpp tokenizer disagrees on letter {letter}")
    # llama.cpp's logit-bias sampler always runs even with an empty sampling
    # chain. Disabling every other token preserves the relative A-D logits.
    # Unlike grammar, this also excludes alternative tokenizations of 'A'.
    return letters, [[token, False] for token in range(vocab_size) if token not in letters]


def _candidate_scores(response: dict, letters: list[int]) -> dict[int, float]:
    steps = response.get("completion_probabilities") or response.get("probs") or []
    if not steps:
        raise ValueError("server returned no next-token probabilities")
    candidates = steps[0].get("top_probs") or steps[0].get("top_logprobs") or steps[0].get("probs") or []
    scores = {}
    for candidate in candidates:
        token = candidate.get("id", candidate.get("token_id"))
        if token not in letters:
            continue
        value = candidate.get("prob")
        if value is None and candidate.get("logprob") is not None:
            value = math.exp(float(candidate["logprob"]))
        if value is not None and math.isfinite(float(value)) and 0 <= float(value) <= 1:
            scores[token] = float(value)
    if set(scores) != set(letters) or sum(scores.values()) <= 0:
        raise ValueError("server did not return valid scores for all four exact letter tokens")
    return scores


def gguf_mmlu_accuracy(tokenizer, items: list[dict], server: str = "http://127.0.0.1:8080",
                       preamble: str = "default") -> dict:
    """Score the same four next-token logits as HF; incomplete scoring fails closed.

    Uses llama.cpp /completion logit_bias + post_sampling_probs + samplers=[]
    (verified against tools/server and common/sampling.cpp). Temperature or
    top-k sampling would destroy the letter ranking. A missing candidate, an
    unsupported server response or a tokenizer mismatch is explicitly unscorable.
    """
    from ..eval.capability import TEMPLATE

    if preamble not in {"default", "none"}:
        raise ValueError(f"unsupported preamble condition: {preamble}")
    server = server.rstrip("/")
    outcomes, correct = [], []
    try:
        letters, bias = _candidate_setup(tokenizer, server)
        setup_error = None
    except (OSError, ValueError, KeyError, TypeError) as error:
        letters, bias, setup_error = [], [], str(error)
    for i, row in enumerate(items):
        outcome = {"id": row.get("id", str(i)), "answer": row["answer"].strip().upper(), "preamble": preamble}
        try:
            if setup_error:
                raise ValueError(setup_error)
            prompt = render_prompt(tokenizer, TEMPLATE.format(q=row["question"], a=row["option_a"],
                                    b=row["option_b"], c=row["option_c"], d=row["option_d"]), preamble)
            response = _post(f"{server}/completion", {
                "prompt": tokenizer.encode(prompt, add_special_tokens=False), "n_predict": 1, "n_probs": 4,
                "temperature": 1.0, "samplers": [], "post_sampling_probs": True,
                "logit_bias": bias, "repeat_penalty": 1.0, "seed": 0, "cache_prompt": True})
            if response.get('truncated'):
                raise ValueError('server truncated the capability input')
            scores = _candidate_scores(response, letters)
            pick = max(range(4), key=lambda index: scores[letters[index]])
            prediction = "ABCD"[pick]
            value = int(prediction == outcome["answer"])
            correct.append(value)
            outcome.update(status="scored", prediction=prediction, correct=value,
                           scores={letter: scores[token] for letter, token in zip("ABCD", letters)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            correct.append(None)
            outcome.update(status="unscorable", prediction=None, correct=None, reason=str(error))
        outcomes.append(outcome)
    scored = [value for value in correct if value is not None]
    complete = bool(items) and len(scored) == len(items)
    scored_accuracy = sum(scored) / len(scored) if scored else None
    return {"accuracy": scored_accuracy if complete else None, "scored_accuracy": scored_accuracy,
            "n": len(items), "n_scored": len(scored), "n_unscorable": len(items) - len(scored),
            "complete": complete, "per_item": correct, "outcomes": outcomes,
            "preamble": preamble, "scoring": "next_token_exact_abcd"}
