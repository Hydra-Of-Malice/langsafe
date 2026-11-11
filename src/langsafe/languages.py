"""Language codes used across the project, and how each source dataset names them."""
from __future__ import annotations

# Our code -> (English name, script)
LANGS: dict[str, tuple[str, str]] = {
    "en": ("English", "Latn"),
    "hi": ("Hindi", "Deva"),
    "hi-Latn": ("Hinglish (code-mixed Hindi-English)", "Latn"),
    "bn": ("Bengali", "Beng"),
    "ta": ("Tamil", "Taml"),
    "ar": ("Arabic", "Arab"),
    "sw": ("Swahili", "Latn"),
    "jv": ("Javanese", "Latn"),
    "th": ("Thai", "Thai"),
    "ko": ("Korean", "Hang"),
}

# CohereLabs/aya_redteaming ships one JSONL per language.
AYA_FILES: dict[str, str] = {"en": "aya_eng.jsonl", "hi": "aya_hin.jsonl", "ar": "aya_arb.jsonl"}

# DAMO-NLP-SG/MultiJail has one column per language (parallel rows).
MULTIJAIL_COLUMNS: dict[str, str] = {"en": "en", "bn": "bn", "sw": "sw", "ar": "ar", "jv": "jv", "th": "th", "ko": "ko"}

# CohereLabs/Global-MMLU-Lite config names.
# It has no Javanese or Thai; Indonesian stands in for Javanese (see `capability_proxies` in the spec).
GMMLU_CONFIGS: dict[str, str] = {"en": "en", "hi": "hi", "bn": "bn", "ar": "ar", "sw": "sw", "ko": "ko", "id": "id"}

# Unicode ranges used to sanity-check that a prompt or answer is in the expected script.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "Deva": [(0x0900, 0x097F)],
    "Beng": [(0x0980, 0x09FF)],
    "Taml": [(0x0B80, 0x0BFF)],
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Hang": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
}


def script_share(text: str, script: str) -> float:
    """Fraction of letters in `text` that belong to `script` (Latn = ASCII letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if script == "Latn":
        return sum(c.isascii() for c in letters) / len(letters)
    ranges = SCRIPT_RANGES[script]
    return sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters) / len(letters)


def name(code: str) -> str:
    return LANGS[code][0]
