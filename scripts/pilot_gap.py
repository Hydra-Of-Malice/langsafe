"""Pilot: how large is the safety gap between Global and a regional model at a real sample size?

    python scripts/pilot_gap.py --regional fire

Uses Cohere's official 4-bit GGUFs (ungated, about 2 GB each, in cache/gguf) and runs on the repair split only, so it
needs no login and never touches held-out data. It covers the languages with native prompts (no translations) and the
default preamble. Its labels come from whichever local judges are in cache/gguf. It is a planning check, not a result:
4-bit weights, uncalibrated judges, no human review. Writes aggregates only to runs/pilot/pilot-<regional>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LANGSAFE_RUNS_DIR", str(ROOT / "runs" / "pilot"))
os.environ.setdefault("LANGSAFE_DATA_DIR", str(ROOT / "runs" / "pilot" / "data"))
sys.path.insert(0, str(ROOT / "src"))

from langsafe.config import load_spec                                              # noqa: E402
from langsafe.data.build import build, load_split                                  # noqa: E402
from langsafe.eval.stats import mcnemar_exact, rate_ci                             # noqa: E402
from langsafe.judge.llm_judge import LLMJudge, consensus                           # noqa: E402
from langsafe.judge.rules import refusal_signal                                    # noqa: E402
from langsafe.llm_client import ServerChatClient                                   # noqa: E402

GGUF = ROOT / "cache" / "gguf"


def generate(name: str, prompts: list) -> list[str]:
    client = ServerChatClient(f"CohereLabs/tiny-aya-{name}", GGUF / f"tiny-aya-{name}-q4_k_m.gguf", batch_size=4)
    try:
        return client.chat_many([[{"role": "user", "content": p.text}] for p in prompts], max_tokens=256)
    finally:
        client.close()


def label(judges: list[dict], prompts: list, responses: list[str]) -> list[str]:
    items = [(p.text, r, p.lang) for p, r in zip(prompts, responses)]
    votes = []
    for cfg in judges:
        judge = LLMJudge(cfg["name"], "local", cfg["model"], gguf=cfg["gguf"], batch_size=4)
        try:
            votes.append([v.label for v in judge.judge_many(items)])
        finally:
            judge.close()
    if len(votes) == 1:
        return votes[0]
    return [consensus(list(pair))[0] for pair in zip(*votes)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regional", choices=["fire", "earth"], default="fire")
    parser.add_argument("--limit", type=int, help="prompts per language and kind")
    args = parser.parse_args()
    spec = load_spec()
    judges = [j for j in spec["judging"]["local_judges"] if (ROOT / j["gguf"]).is_file()]
    if not judges:
        raise SystemExit("No judge GGUF in cache/gguf")
    if not (Path(os.environ["LANGSAFE_DATA_DIR"]) / "manifest.json").is_file():
        build(translate=False, use_curatorkit=False)
    languages = spec["languages"][args.regional]
    counts: dict = defaultdict(int)
    prompts = []
    for p in load_split("repair"):
        if p.lang in languages and (not args.limit or counts[p.lang, p.kind] < args.limit):
            counts[p.lang, p.kind] += 1
            prompts.append(p)

    labels, rules = {}, {}
    for name in ("global", args.regional):
        responses = generate(name, prompts)
        labels[name] = label(judges, prompts, responses)
        rules[name] = [refusal_signal(r, p.lang) for r, p in zip(responses, prompts)]

    table = {}
    for lang in sorted({p.lang for p in prompts}):
        for kind in ("harmful", "benign"):
            index = [i for i, p in enumerate(prompts) if p.lang == lang and p.kind == kind]
            if not index:
                continue
            cell = {"n": len(index)}
            for name in labels:
                refused = [labels[name][i] == "full_refusal" for i in index]
                mean, low, high = rate_ci(refused, 2000)
                cell[name] = {"refusal": mean, "ci": [low, high],
                              "compliance": sum(labels[name][i] == "full_compliance" for i in index) / len(index),
                              "unclear": sum(labels[name][i] == "unclear" for i in index) / len(index),
                              "rule_refusal": sum(rules[name][i] for i in index) / len(index)}
            a = [int(labels["global"][i] == "full_refusal") for i in index]
            b = [int(labels[args.regional][i] == "full_refusal") for i in index]
            cell["gap"] = cell["global"]["refusal"] - cell[args.regional]["refusal"]
            cell["p_mcnemar"] = mcnemar_exact(a, b)
            table[f"{lang}/{kind}"] = cell
            print(f"{lang:8}{kind:8} n={len(index):4} global={cell['global']['refusal']:.3f} "
                  f"{args.regional}={cell[args.regional]['refusal']:.3f} gap={cell['gap']:+.3f} "
                  f"p={cell['p_mcnemar']:.4f} compliance {cell['global']['compliance']:.3f}/"
                  f"{cell[args.regional]['compliance']:.3f}", flush=True)
    out = Path(os.environ["LANGSAFE_RUNS_DIR"]) / f"pilot-{args.regional}.json"
    out.write_text(json.dumps({"status": "pilot: 4-bit, repair split, uncalibrated, not a result",
                               "judges": [j["name"] for j in judges], "cells": table}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
