"""Fill the Hugging Face cache over a connection that drops, keeping every byte already received.

    python scripts/fetch_models.py CohereLabs/tiny-aya-global CohereLabs/tiny-aya-fire CohereLabs/tiny-aya-base
    python scripts/fetch_models.py Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf       # one file, into cache/gguf

`hf download` names each partial file per process, so every retry after a dropped connection starts that file from zero.
This writes straight into the cache's snapshot folder and continues a partial file with an HTTP range request. There is
no retry shell around it, so stopping it stops the download.
"""
from __future__ import annotations

import fnmatch
import os
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE

PATTERNS = ("*.json", "*.safetensors", "*.model", "*.jinja")
CHUNK = 1 << 20
GGUF_DIR = Path(__file__).resolve().parents[1] / "cache" / "gguf"


def wanted(path: str) -> bool:
    return "/" not in path and any(fnmatch.fnmatch(path, pattern) for pattern in PATTERNS)


def fetch(url: str, target: Path, size: int, headers: dict) -> None:
    part = target.with_name(target.name + ".part")
    while not target.exists():
        have = part.stat().st_size if part.exists() else 0
        if have >= size:
            part.replace(target)
            break
        try:
            with requests.get(url, headers={**headers, "Range": f"bytes={have}-"}, stream=True, timeout=(30, 120)) as reply:
                if reply.status_code == 416:        # asked past the end: the file is already whole
                    continue
                if reply.status_code in (401, 403, 404):
                    sys.exit(f"{target.name}: HTTP {reply.status_code}. Check the licence on the model page and your login.")
                if reply.status_code not in (200, 206):
                    raise OSError(f"HTTP {reply.status_code}")
                mode = "ab" if reply.status_code == 206 else "wb"     # 200 means the server ignored the range
                with open(part, mode) as stream:
                    for chunk in reply.iter_content(CHUNK):
                        stream.write(chunk)
        except (OSError, MemoryError, requests.RequestException) as error:
            print(f"retrying {target.name} at {have / 2**30:.2f} GiB: {error}", flush=True)
            time.sleep(20)


def repo_info(api: HfApi, repo: str, token: str | None):
    while True:
        try:
            return api.model_info(repo, files_metadata=True, token=token)
        except (OSError, requests.RequestException) as error:
            print(f"waiting for the network: {type(error).__name__}", flush=True)
            time.sleep(30)


def main() -> None:
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api = HfApi()
    for item in sys.argv[1:]:
        if item.endswith(".gguf"):
            repo, name = item.rsplit("/", 1)
            info = repo_info(api, repo, token)
            size = next(f.size for f in info.siblings if f.rfilename == name)
            GGUF_DIR.mkdir(parents=True, exist_ok=True)
            fetch(hf_hub_url(repo, name, revision=info.sha), GGUF_DIR / name, size, headers)
            print(f"complete {name}", flush=True)
            continue
        repo = item
        info = repo_info(api, repo, token)
        folder = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        snapshot = folder / "snapshots" / info.sha
        snapshot.mkdir(parents=True, exist_ok=True)
        (folder / "refs").mkdir(exist_ok=True)
        (folder / "refs" / "main").write_text(info.sha, encoding="utf-8")
        for file in sorted((f for f in info.siblings if wanted(f.rfilename)), key=lambda f: f.size or 0):
            target = snapshot / file.rfilename
            if target.exists() and target.stat().st_size == file.size:
                continue
            target.unlink(missing_ok=True)
            fetch(hf_hub_url(repo, file.rfilename, revision=info.sha), target, file.size, headers)
            print(f"{repo}: {file.rfilename} ({file.size / 2**30:.2f} GiB)", flush=True)
        print(f"complete {repo}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
