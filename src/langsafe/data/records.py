"""The one record schema every prompt is normalised to."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

Kind = Literal["harmful", "benign"]


@dataclass
class Prompt:
    id: str                  # unique per (group, lang)
    group: str               # same group = same underlying item in another language; never split across sets
    lang: str
    kind: Kind
    text: str
    source: str              # dataset id
    source_id: str
    category: str = ""       # harm category or XSTest prompt type
    origin: str = "native"   # native | translated | code-mixed
    split: str = ""          # repair | heldout
    meta: dict = field(default_factory=dict)

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.strip().encode()).hexdigest()[:16]


def make_id(group: str, lang: str) -> str:
    return f"{group}::{lang}"


def write_jsonl(path: str | Path, rows: Iterable[Prompt | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r) if isinstance(r, Prompt) else r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_prompts(path: str | Path) -> Iterator[Prompt]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Prompt(**json.loads(line))
