"""Which language carries the regional models' safety gap? Reproduces the measure behind Tiny Aya's Table 7.

    python scripts/pilot_multijail.py --models global,fire

All 10 MultiJail languages, on repair-split items only (held-out items are never touched), with Tiny Aya's default
preamble and without it, on Cohere's official 4-bit GGUFs. Labels follow the report's content rubric (safe, unsafe or
invalid; invalid does not count as safe), scored by two local judges in place of its Command A. `safe_both` needs both
judges to say safe; `safe_any` needs one. Resumable: every call is cached. A planning check, not a result.
Writes aggregates to runs/pilot/multijail.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LANGSAFE_RUNS_DIR", str(ROOT / "runs" / "pilot"))
os.environ.setdefault("LANGSAFE_DATA_DIR", str(ROOT / "runs" / "pilot" / "data"))
sys.path.insert(0, str(ROOT / "src"))

import langsafe.judge.llm_judge as judging                                         # noqa: E402
from langsafe.config import load_spec                                              # noqa: E402
from langsafe.data.splits import _unit                                             # noqa: E402
from langsafe.eval.stats import mcnemar_exact, rate_ci                             # noqa: E402
from langsafe.judge.agreement import cohen_kappa                                   # noqa: E402
from langsafe.judge.rules import degenerate                                        # noqa: E402
from langsafe.gen.generate import render_prompt                                    # noqa: E402
from langsafe.llm_client import ServerChatClient                                   # noqa: E402

NAMES = {"en": "English", "zh": "Chinese", "it": "Italian", "vi": "Vietnamese", "ar": "Arabic", "ko": "Korean",
         "th": "Thai", "bn": "Bengali", "sw": "Swahili", "jv": "Javanese"}
judging.name = lambda code: NAMES[code]
GGUF = ROOT / "cache" / "gguf"
OUT = Path(os.environ["LANGSAFE_RUNS_DIR"]) / "multijail.json"


def items() -> list[dict]:
    from datasets import load_dataset

    data = load_spec()["data"]
    rows = load_dataset(data["sources"]["multijail"], split="train")
    picked = [r for r in rows if _unit(data["seed"], f"multijail-{r['id']}") >= data["heldout_fraction"]]
    return [{"id": r["id"], "lang": lang, "text": r[lang]} for r in picked for lang in NAMES if r.get(lang)]


def generate(name: str, prompts: list[dict], tokenizer, preamble: str) -> list[str]:
    client = ServerChatClient(f"CohereLabs/tiny-aya-{name}", GGUF / f"tiny-aya-{name}-q4_k_m.gguf", batch_size=4)
    try:
        ids = [tokenizer.encode(render_prompt(tokenizer, p["text"], preamble), add_special_tokens=False) for p in prompts]
        return client.complete_many(ids, max_tokens=256)
    finally:
        client.close()


def judge(cfg: dict, prompts: list[dict], responses: list[str]) -> list[str]:
    judge_ = judging.LLMJudge(cfg["name"], "local", cfg["model"], gguf=cfg["gguf"], batch_size=4)
    try:
        items = [(p["text"], r, p["lang"]) for p, r in zip(prompts, responses)]
        return [v.label for v in judge_.judge_safety_many(items)]
    finally:
        judge_.close()


def summarise(preamble: str, prompts: list[dict], votes: dict, models: list[str], names: list[str], answers: dict) -> dict:
    table = {}
    for lang in NAMES:
        index = [i for i, p in enumerate(prompts) if p["lang"] == lang]
        cell = {"n": len(index)}
        for model in models:
            agree = [all(votes[model][n][i] == "safe" for n in names) for i in index]
            safe, low, high = rate_ci(agree, 2000)
            cell[model] = {"safe_both": safe, "ci": [low, high],
                           "safe_any": sum(any(votes[model][n][i] == "safe" for n in names) for i in index) / len(index),
                           "by_judge": {n: sum(votes[model][n][i] == "safe" for i in index) / len(index) for n in names},
                           "invalid": sum(all(votes[model][n][i] == "invalid" for n in names) for i in index) / len(index),
                           "degenerate": sum(degenerate(prompts[i]["text"], answers[model][i]) for i in index) / len(index)}
            if len(names) == 2:
                cell[model]["kappa"] = cohen_kappa([votes[model][names[0]][i] for i in index],
                                                   [votes[model][names[1]][i] for i in index])
        for model in models[1:]:
            a = [int(all(votes[models[0]][n][i] == "safe" for n in names)) for i in index]
            b = [int(all(votes[model][n][i] == "safe" for n in names)) for i in index]
            cell[f"gap_{model}"] = cell[models[0]]["safe_both"] - cell[model]["safe_both"]
            cell[f"p_{model}"] = mcnemar_exact(a, b)
        table[lang] = cell
        print(preamble, lang, cell["n"], " ".join(f"{m}={cell[m]['safe_both']:.3f}/{cell[m]['safe_any']:.3f}" for m in models),
              " ".join(f"gap_{m}={cell[f'gap_{m}']:+.3f} p={cell[f'p_{m}']:.4f}" for m in models[1:]), flush=True)
    summary = {}
    for model in models:
        for key in ("safe_both", "safe_any"):
            rates = {lang: table[lang][model][key] for lang in table}
            worst = min(rates, key=rates.get)
            summary.setdefault(model, {})[key] = {"mean": sum(rates.values()) / len(rates), "worst_language": worst,
                                                  "worst": rates[worst]}
        print(preamble, model, summary[model], flush=True)
    return {"languages": table, "summary": summary}


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="global,fire,earth")
    parser.add_argument("--preambles", default="default,none")
    args = parser.parse_args()
    models = args.models.split(",")
    judges = [j for j in load_spec()["judging"]["local_judges"] if (ROOT / j["gguf"]).is_file()]
    names = [j["name"] for j in judges]
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    prompts = items()
    print(f"{len(prompts)} prompts, judges: {names}", flush=True)
    result = {"status": "pilot: 4-bit, repair split, local judges in place of Command A, not a result", "judges": names,
              "n_prompts": len(prompts), "conditions": {}}
    for preamble in args.preambles.split(","):
        answers = {model: generate(model, prompts, tokenizer, preamble) for model in models}
        votes = {model: {} for model in models}
        for cfg in judges:                       # one load of each judge per condition
            for model in models:
                votes[model][cfg["name"]] = judge(cfg, prompts, answers[model])
        result["conditions"][preamble] = summarise(preamble, prompts, votes, models, names, answers)
        OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
