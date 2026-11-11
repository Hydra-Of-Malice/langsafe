"""Counts how many tokens a closed model's tokenizer needs for each Tiny Aya token, per language, for the ROI's f.

The sample is the run's benign prompts (XSTest and its translations, held-out and repair splits), so no harmful text is
read. Only counts are written. `o200k_base` is the tokenizer of OpenAI's current models; Anthropic and Google publish
none, so their f stays an assumption.

    python scripts/token_fertility.py runs/experiments/full/earth
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from pathlib import Path

DEFAULT_MODEL = 'CohereLabs/tiny-aya-global'
DEFAULT_ENCODING = 'o200k_base'


def benign_texts(run: Path) -> dict[str, list[str]]:
    """Benign prompt text by language, from both splits."""
    texts: dict[str, list[str]] = {}
    for split in ('heldout', 'repair'):
        path = run / 'data' / f'{split}.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('kind') == 'benign' and row.get('text'):
                texts.setdefault(row['lang'], []).append(row['text'])
    return texts


def fertility(texts: dict[str, Iterable[str]], ours: Callable[[str], int], closed: Callable[[str], int]) -> dict:
    """Per language: prompts, token totals for both tokenizers, and closed tokens per Tiny Aya token."""
    result = {}
    for lang in sorted(texts):
        rows = list(texts[lang])
        mine = sum(ours(text) for text in rows)
        theirs = sum(closed(text) for text in rows)
        if not rows or not mine:
            raise ValueError(f'No tokens to compare for {lang}')
        result[lang] = {'prompts': len(rows), 'tiny_aya_tokens': mine, 'closed_tokens': theirs,
                        'closed_per_tiny_aya': theirs / mine}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', type=Path)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--encoding', default=DEFAULT_ENCODING)
    parser.add_argument('--out', type=Path, help='default: <run>/fertility.json')
    args = parser.parse_args()
    import tiktoken
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoding = tiktoken.get_encoding(args.encoding)
    languages = fertility(benign_texts(args.run),
                          lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
                          lambda text: len(encoding.encode(text, disallowed_special=())))
    out = args.out or args.run / 'fertility.json'
    result = {'model': args.model, 'encoding': args.encoding, 'sample': 'benign prompts, held-out and repair splits',
              'languages': languages}
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for lang, cell in languages.items():
        print(f"{lang}: {cell['closed_per_tiny_aya']:.2f} ({cell['prompts']} prompts)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
