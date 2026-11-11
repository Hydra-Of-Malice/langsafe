"""Chat clients for judging and translation, with an on-disk cache.

Three backends share one interface (`chat`, `chat_many`, `close`):
- `ChatClient`: any OpenAI-compatible endpoint (the Hugging Face router by default; billed to inference credits).
- `LocalChatClient`: the same model family run locally in NF4 on CUDA or FP16 on MPS (no credits).
- `ServerChatClient`: a local GGUF behind llama-server (no credits, and a third of the download on a slow link).

Calls are cached by backend, checkpoint identity, precision, messages and parameters.
`make_client(base_url="local", ...)` picks the local backend.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import ROOT

CACHE_PATH = ROOT / "runs" / "llm_cache.sqlite"


def _cache_path(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get('LANGSAFE_RUNS_DIR', ROOT / 'runs')) / 'llm_cache.sqlite'


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def _checkpoint_identity(model: str) -> dict:
    """Record cached revision or local file identity without loading weights or accessing the network.

    Weight metadata detects replaced local files without hashing multi-gigabyte checkpoints on every cache hit.
    This is cache invalidation evidence, not a substitute for the experiment's full weight provenance hash.
    """
    path = Path(model).expanduser()
    if not path.is_dir():
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model, "config.json")
            if isinstance(cached, str):
                path = Path(cached).parent
        except (ImportError, OSError, ValueError):
            pass
    if not path.is_dir():
        return {"model": model, "revision": "unknown"}
    identity = {"path": str(path.resolve()), "files": []}
    for item in sorted(path.iterdir()):
        if item.is_file() and (item.suffix in {".json", ".safetensors", ".bin", ".model", ".jinja"}):
            stat = item.stat()
            entry = {"name": item.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if item.suffix in {".json", ".jinja"}:
                entry["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
            identity["files"].append(entry)
    if path.parent.name == "snapshots":
        identity["revision"] = path.name
    return identity


class _Cache:
    def __init__(self, path: Path, model_key: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self.model_key = model_key
        self._closed = False

    def key(self, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"m": self.model_key, "msgs": messages, "p": params}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str, attempts: int = 5) -> None:
        """A transient failure to open the journal must not throw away a verdict that took a model call to get."""
        with self._lock:
            for attempt in range(attempts):
                try:
                    self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, v))
                    self._db.commit()
                    return
                except sqlite3.OperationalError:
                    if attempt == attempts - 1:
                        raise
                    time.sleep(2 ** attempt)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str = "HF_TOKEN", cache_path: Path | None = None,
                 max_retries: int = 5, workers: int = 8, checkpoint_identity: str | None = None):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Set {api_key_env} to call {model} at {base_url}")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.max_retries = max_retries
        self.workers = workers
        if max_retries < 1 or workers < 1:
            self.client.close()
            raise ValueError("max_retries and workers must be positive")
        identity = {"backend": "openai-compatible", "base_url": base_url.rstrip("/"), "model": model,
                    "checkpoint": checkpoint_identity, "precision": "provider-controlled"}
        self._cache = _Cache(_cache_path(cache_path), json.dumps(identity, sort_keys=True))
        self._closed = False

    def chat(self, messages: list[dict], **params) -> str:
        params = {"temperature": 0, "max_tokens": 512, **params}
        k = self._cache.key(messages, params)
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages, **params)
                text = resp.choices[0].message.content or ""
                break
            except Exception:  # rate limits and transient provider errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        self._cache.put(k, text)
        return text

    def chat_many(self, batch: list[list[dict]], **params) -> list[str]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda m: self.chat(m, **params), batch))

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.client.close()
            finally:
                self._cache.close()


class LocalChatClient:
    """Greedy, batched generation on the local GPU. The model loads on the first cache miss, so a fully cached
    re-run needs no model. Backend, device, precision, runtime versions and checkpoint identity keep local
    verdicts separate from hosted verdicts and changed checkpoints."""

    def __init__(self, model: str, quant: str | None = "auto", batch_size: int = 8, cache_path: Path | None = None,
                 dtype: str = "float16", checkpoint_identity: str | None = None):
        from .gen.model import best_device, resolve_model

        try:
            device = best_device()
        except ImportError:
            device = "unavailable"
        if quant == "auto":            # NF4 on CUDA; fp16 elsewhere (an 8B judge in fp16 is ~16 GB, fine on a 64 GB Mac)
            quant = "nf4" if device == "cuda" else None
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model
        self._resolved_model = resolve_model(model)
        self.quant = quant
        self.dtype = dtype
        self.batch_size = batch_size
        identity = {"backend": "local-transformers", "model": model, "quant": quant, "dtype": dtype,
                    "effective_dtype": "float32" if device == "cpu" and dtype == "float16" else dtype,
                    "device": device, "checkpoint": _checkpoint_identity(self._resolved_model),
                    "declared_checkpoint": checkpoint_identity, "torch": _version("torch"),
                    "transformers": _version("transformers")}
        self._cache = _Cache(_cache_path(cache_path), json.dumps(identity, sort_keys=True))
        self._m = self._tok = None
        self._closed = False

    def _load(self):
        if self._m is None:
            from .gen.model import load

            self._m, self._tok = load(self._resolved_model, dtype=self.dtype, quant=self.quant)
            self._tok.padding_side = "left"
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token

    def _render(self, messages: list[dict]) -> str:
        # enable_thinking is read by Qwen3's template and ignored by others.
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)

    def chat(self, messages: list[dict], **params) -> str:
        return self.chat_many([messages], **params)[0]

    def chat_many(self, batch: list[list[dict]], **params) -> list[str]:
        params = {"temperature": 0, "max_tokens": 512, **params}
        keys = [self._cache.key(m, params) for m in batch]
        out = [self._cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        if not todo:
            return out
        import torch
        from tqdm import tqdm

        self._load()
        # Longest first keeps padding low within a batch.
        todo.sort(key=lambda i: -sum(len(x["content"]) for x in batch[i]))
        for s in tqdm(range(0, len(todo), self.batch_size), desc=f"local {self.model}", leave=False):
            idx = todo[s:s + self.batch_size]
            enc = self._tok([self._render(batch[i]) for i in idx], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(self._m.device)
            with torch.no_grad():
                gen = self._m.generate(**enc, max_new_tokens=params["max_tokens"], do_sample=False,
                                       pad_token_id=self._tok.pad_token_id)
            texts = self._tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t
                self._cache.put(keys[i], t)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._m is not None or self._tok is not None:
                import gc

                import torch

                self._m = self._tok = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        finally:
            self._cache.close()


def llama_server_binary() -> str:
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    return (os.environ.get("LANGSAFE_LLAMA_SERVER") or shutil.which("llama-server")
            or str(ROOT / "cache" / "llama.cpp" / "build" / "bin" / name))


class ServerChatClient:
    """A GGUF model behind a private llama-server. It starts on the first cache miss and stops on close(), so a fully
    cached re-run starts nothing. Requests run in parallel slots; each slot holds `slot_context` tokens."""

    def __init__(self, model: str, gguf: str | Path, batch_size: int = 4, cache_path: Path | None = None,
                 slot_context: int = 3072, **_ignored):
        self.model, self.gguf = model, Path(gguf)
        self.batch_size, self.slot_context = batch_size, slot_context
        identity = {"backend": "llama-server-gguf", "model": model, "gguf": self.gguf.name,
                    "gguf_bytes": self.gguf.stat().st_size}
        self._cache = _Cache(_cache_path(cache_path), json.dumps(identity, sort_keys=True))
        self._log = _cache_path(cache_path).with_name("llama-server.log")
        self._server = self._url = None
        self._restart_lock = threading.Lock()

    def _revive(self) -> None:
        """Bring back a server that died mid-run. Parallel requests all land here, so only the first one restarts it."""
        with self._restart_lock:
            if self._server is not None and self._server.poll() is not None:
                print(f"llama-server exited with {self._server.returncode}; restarting", flush=True)
                self._server = None
                self._start()

    def _start(self) -> None:
        if self._server is not None:
            return
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self._url = f"http://127.0.0.1:{port}"
        with open(self._log, "ab") as log:
            self._server = subprocess.Popen(
                [llama_server_binary(), "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99",
                 "-np", str(self.batch_size), "-c", str(self.slot_context * self.batch_size)],
                stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if self._server.poll() is not None:
                raise RuntimeError(f"llama-server exited while loading {self.gguf}")
            try:
                with urllib.request.urlopen(self._url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, ValueError):
                time.sleep(1)
        raise RuntimeError("llama-server did not become ready")

    def _post(self, path: str, body: bytes, attempts: int = 5) -> dict:
        """llama-server can fail a request transiently under load; a persistent failure raises with its message."""
        for attempt in range(attempts):
            request = urllib.request.Request(self._url + path, data=body, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=900) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                detail = error.read()[:300].decode("utf-8", "replace")
                if error.code < 500 or attempt == attempts - 1:
                    raise RuntimeError(f"llama-server {error.code}: {detail}") from error
                print(f"llama-server {error.code}, retrying: {detail}", flush=True)
                time.sleep(2 ** attempt)
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as error:
                if attempt == attempts - 1:
                    raise
                print(f"llama-server connection failed ({type(error).__name__}), retrying", flush=True)
                time.sleep(2 ** attempt)
                self._revive()
        raise RuntimeError("unreachable")

    def _complete(self, messages: list[dict], params: dict) -> str:
        body = json.dumps({"messages": messages, "temperature": 0, "seed": 0, "max_tokens": params["max_tokens"],
                           "chat_template_kwargs": {"enable_thinking": False}}).encode()
        return self._post("/v1/chat/completions", body)["choices"][0]["message"].get("content") or ""

    def _raw(self, ids: list[int], params: dict) -> str:
        body = json.dumps({"prompt": ids, "n_predict": params["max_tokens"], "temperature": 0, "top_k": 1, "seed": 0,
                           "cache_prompt": True}).encode()
        return self._post("/completion", body)["content"].strip()

    def _many(self, batch: list, call, params: dict) -> list[str]:
        from tqdm import tqdm

        keys = [self._cache.key(item, params) for item in batch]
        out = [self._cache.get(k) for k in keys]
        todo = [i for i, v in enumerate(out) if v is None]
        if not todo:
            return out
        self._start()

        def run(i: int) -> None:
            out[i] = call(batch[i], params)
            self._cache.put(keys[i], out[i])

        with ThreadPoolExecutor(self.batch_size) as pool:
            list(tqdm(pool.map(run, todo), total=len(todo), desc=f"gguf {self.model}", leave=False))
        return out

    def chat(self, messages: list[dict], **params) -> str:
        return self.chat_many([messages], **params)[0]

    def chat_many(self, batch: list[list[dict]], **params) -> list[str]:
        return self._many(batch, self._complete, {"temperature": 0, "max_tokens": 512, **params})

    def complete_many(self, batch: list[list[int]], **params) -> list[str]:
        """Raw token prompts, for exact control of the chat template (such as stripping the preamble)."""
        return self._many(batch, self._raw, {"raw": True, "max_tokens": 512, **params})

    def close(self) -> None:
        if self._server is not None:
            self._server.terminate()
            try:
                self._server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._server.kill()
                self._server.wait()
            self._server = None
        self._cache.close()


def make_client(base_url: str, model: str, *, gguf: str | None = None, **kwargs):
    """`gguf` is used when that file is on this machine; otherwise the transformers backend loads the same model."""
    if base_url == "local":
        if gguf and (ROOT / gguf).is_file():
            return ServerChatClient(model, ROOT / gguf, **kwargs)
        return LocalChatClient(model, **kwargs)
    return ChatClient(base_url, model, **kwargs)
