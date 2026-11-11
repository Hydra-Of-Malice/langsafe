"""Does the circuit map help? Compares each circuit arm with its whole-model twin and with its random-layer controls.

    python scripts/localisation_analysis.py runs/experiments/full/fire [--judges local]

The standard report compares every arm with the original model. The project's main question needs two other contrasts,
paired on the same prompts: circuit vs the same vector on the whole model, and circuit vs the same vector and knob on
random layers. Reads judged labels, capability records and edit statistics; writes aggregates only, to
localisation.json and localisation.md in the run directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                     # noqa: E402

from langsafe.eval.stats import mcnemar_exact, paired_diff_ci          # noqa: E402

TWINS = {"circuit": "resta", "circuit_merge": "merge_back"}


def judged(run: Path, tag: str, judges: str) -> dict:
    name = f"{tag}.jsonl" if judges == "router" else f"{tag}.{judges}.jsonl"
    with open(run / "judged" / name, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {(r["id"], r["preamble"]): r for r in rows}


def capability(run: Path, tag: str, preamble: str = "default") -> dict:
    path = run / "capability" / (f"{tag}.json" if preamble == "default" else f"{tag}.none.json")
    return json.loads(path.read_text(encoding="utf-8")).get("languages", {}) if path.exists() else {}


def edit(descriptor: dict) -> dict:
    """The size and location of an edit, from the provenance saved with its checkpoint."""
    path = Path(descriptor.get("model", "")) / "lexsi_provenance.json"
    if not path.is_file():
        return {}
    params = json.loads(path.read_text(encoding="utf-8")).get("params", {})
    return {key: params.get(key) for key in ("relative_edit_l2", "edited_tensors", "edited_layers", "layer_subset")}


def contrast(a: dict, b: dict, lang: str, kind: str, label: str, resamples: int, field: str = "label") -> dict | None:
    """Paired change from a to b in the rate of `label` in one language, over prompts both runs answered."""
    keys = sorted(k for k in set(a) & set(b) if a[k]["kind"] == kind and a[k]["lang"] == lang
                  and field in a[k] and field in b[k])
    if not keys:
        return None
    x = [int(a[k][field] == label) for k in keys]
    y = [int(b[k][field] == label) for k in keys]
    delta, low, high = paired_diff_ci(x, y, resamples)
    return {"delta": delta, "ci": [low, high], "p_mcnemar": mcnemar_exact(x, y), "n": len(keys)}


def cluster_ci(differences: list[float], resamples: int) -> dict:
    """Bootstrap over items. Each item's languages form one cluster, because translations aren't independent."""
    values = np.array(differences, dtype=float)
    draws = np.random.default_rng(20261103).integers(0, len(values), (resamples, len(values)))
    means = values[draws].mean(axis=1)
    return {"delta": float(values.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p_mcnemar": None, "n": len(values)}


def clustered(a: dict, b: dict, kind: str, label: str, resamples: int, field: str = "label") -> dict | None:
    """Paired change across all languages, with every language version of an item as one cluster."""
    groups: dict = {}
    for k in sorted(set(a) & set(b)):
        if a[k]["kind"] == kind and field in a[k] and field in b[k]:
            change = int(b[k][field] == label) - int(a[k][field] == label)
            groups.setdefault(a[k].get("group", k[0]), []).append(change)
    return cluster_ci([sum(v) / len(v) for v in groups.values()], resamples) if groups else None


def capability_contrast(a: dict, b: dict, resamples: int) -> dict | None:
    """Global-MMLU items are parallel across languages, so each item index is one cluster."""
    languages = [lang for lang in sorted(set(a) & set(b)) if a[lang].get("per_item")
                 and len(a[lang]["per_item"]) == len(b[lang].get("per_item", []))]
    if not languages:
        return None
    size = min(len(a[lang]["per_item"]) for lang in languages)
    return cluster_ci([sum(b[lang]["per_item"][i] - a[lang]["per_item"][i] for lang in languages) / len(languages)
                       for i in range(size)], resamples)


def compare(run: Path, baseline: str, candidate: str, target: str, judges: str, resamples: int) -> dict:
    """What moving from `baseline` to `candidate` does on the three axes, for each preamble condition separately."""
    first, second = judged(run, baseline, judges), judged(run, candidate, judges)
    conditions = {}
    for preamble in sorted({key[1] for key in first}):
        a = {k: v for k, v in first.items() if k[1] == preamble}
        b = {k: v for k, v in second.items() if k[1] == preamble}
        conditions[preamble] = {
            "target_content_safe": contrast(a, b, target, "harmful", "safe", resamples, field="safety_label"),
            "all_content_safe": clustered(a, b, "harmful", "safe", resamples, field="safety_label"),
            "target_harmful_refusal": contrast(a, b, target, "harmful", "full_refusal", resamples),
            "all_harmful_refusal": clustered(a, b, "harmful", "full_refusal", resamples),
            "over_refusal": clustered(a, b, "benign", "full_refusal", resamples),
            "unclear": clustered(a, b, "harmful", "unclear", resamples),
            "capability": capability_contrast(capability(run, baseline, preamble), capability(run, candidate, preamble),
                                              resamples)}
    return {"baseline": baseline, "candidate": candidate, "conditions": conditions}


def analyse(run: Path, judges: str = "local", resamples: int = 10000) -> dict:
    selection = json.loads((run / "selection.json").read_text(encoding="utf-8"))
    target = selection["worst_language"]
    arms = {name: arm["selected"] for name, arm in selection.get("arms", {}).items()}
    result = {"target_language": target, "target_reason": selection.get("target_reason"), "judges": judges,
              "status": "provisional until human and router calibration are complete",
              "edits": {name: {"knob": d["knob"], **edit(d)} for name, d in arms.items()},
              "whole_model_twin": {}, "random_control": {}}
    for circuit, twin in TWINS.items():
        if circuit in arms and twin in arms:
            result["whole_model_twin"][circuit] = compare(run, arms[twin]["tag"], arms[circuit]["tag"], target, judges,
                                                          resamples)
    for control in selection.get("controls", []):
        circuit = arms[control["arm"]]
        result["random_control"][control["tag"]] = {"seed": control["random_seed"], "edit": edit(control),
                                                    **compare(run, control["tag"], circuit["tag"], target, judges, resamples)}
    return result


def _pp(cell: dict | None) -> str:
    if not cell:
        return "n/a"
    p = "" if cell["p_mcnemar"] is None else f"p={cell['p_mcnemar']:.3f}, "
    return f"{100 * cell['delta']:+.1f} pp ({100 * cell['ci'][0]:+.1f} to {100 * cell['ci'][1]:+.1f}), {p}n={cell['n']}"


def render(result: dict) -> str:
    lines = ["# Does the circuit map help?", "", f"**{result['status'].capitalize()}.** Judges: `{result['judges']}`. "
             f"Target language: `{result['target_language']}` ({result['target_reason']}).", "",
             "Each row is the change from the baseline to the circuit arm, paired on the same prompts. Positive safety, "
             "refusal and capability changes favour the circuit arm; positive over-refusal and unclear changes count "
             "against it. All-language figures treat an item's translations as one cluster, so they carry no McNemar test.",
             "", "## Edit sizes", "", "| Arm | Knob | Relative L2 | Tensors | Layers |", "|---|---|---|---|---|"]
    for name, cell in result["edits"].items():
        l2 = cell.get("relative_edit_l2")
        layers = cell.get("edited_layers")
        lines.append(f"| {name} | {cell['knob']} | {f'{l2:.5f}' if l2 is not None else 'n/a'} | "
                     f"{cell.get('edited_tensors', 'n/a')} | {len(layers) if layers else 'n/a'} |")
    for title, section in (("Circuit vs whole-model twin", result["whole_model_twin"]),
                           ("Circuit vs random layers", result["random_control"])):
        lines += ["", f"## {title}", "",
                  "| Baseline → circuit arm | Preamble | Target content-safe | All content-safe | Target refusal "
                  "| All harmful refusal | Over-refusal | Unclear | Capability |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for entry in section.values():
            for preamble, cell in entry["conditions"].items():
                lines.append(f"| {entry['baseline']} → {entry['candidate']} | {preamble} | "
                             f"{_pp(cell['target_content_safe'])} | {_pp(cell['all_content_safe'])} | "
                             f"{_pp(cell['target_harmful_refusal'])} | {_pp(cell['all_harmful_refusal'])} | "
                             f"{_pp(cell['over_refusal'])} | {_pp(cell['unclear'])} | {_pp(cell['capability'])} |")
        if not section:
            lines.append("| none available | | | | | | | | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--judges", default="local")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):      # a Windows console can't encode the arrows in the report
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = analyse(args.run, args.judges)
    (args.run / "localisation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (args.run / "localisation.md").write_text(render(result), encoding="utf-8")
    print(render(result))


if __name__ == "__main__":
    main()
