# Gaps found in the Lexsi kits (candidate upstream contributions)

These come from reading the kits' source at the pinned commits. Each was worked around in `langsafe`. The ones marked ⭐ are
the pull requests we plan to open.

## AuditKIT (1.1.3, 71316da)
- ⭐ **The refusal detector is English-only** (8 English regexes; `can'?t` misses the curly apostrophe). PR: a multilingual
  refusal scorer (en, hi, Hinglish, bn, ta, ar, sw) as a `Detector` and an `@ak.scorer` (from `src/langsafe/judge/rules.py`).
- ⭐ **No XSTest, MultiJail or Global-MMLU loaders.** PR: loaders that return `Sample(task=lang)`.
- **The paired-bootstrap p-value is invalid** (differences are not centred). PR #9 fixes it but is still open.
  We use our own McNemar and bootstrap tests.
- **HF model identity ignores quantisation**, so base and NF4 runs of the same repo share a cache fingerprint.
  Workaround: a distinct `variant` in `RunConfig.extra`.
- **No judge-agreement statistics, and no GGUF/llama.cpp backend.**
- **CSV save has no `encoding=`**, so non-ASCII text can fail on Windows.

## SafeTune (0.1.12, 3901a9f)
- ⭐ **No MultiJail or Aya red-teaming benchmark loaders.** The built-in refusal checks and the WildGuard judge are English.
- **`RefusalDirectionTrainer` can't take a config.** Its default alpha=20 breaks Tiny Aya, and ReSta's default alpha=1.0 also breaks it.
- **`build_safety_lora_config` ignores `layer_subset`**, so a circuit-guided LoRA still trains every layer.
- ⭐ **`circuit_info_from_node_scores` turns a top-20% node set into nearly the whole model.** It keeps every layer holding
  any selected node, which on Tiny Aya was 30 of 36 layers, so "circuit-restricted" edits are not restricted. PR: rank
  layers by total node score (`top_layers` in `src/langsafe/circuits/discover.py`).
- **`QReSafeTrainer` never passes `pre_quant_model`.** `evaluate()` without `tokenizer=` fails, and `examples/demos/full_pipeline.py` has this bug.

## CircuitKIT (e71d5e1)
- ⭐ **No cross-language task, and no refusal task for chat models.** PR: a multilingual refusal minimal-pair task
  built from `jailbreak_binary.csv`, with chat-template rendering.
- **The custom-data path never applies a chat template** (the docs say it does). That causes a double BOS unless prompts are pre-rendered.
- **Only `tiny-aya-base` is parity-tested.** Global, Fire and Earth are registered but have not been run.
- **`discover` returns the nodes to prune**, which is easy to misread as "the circuit".
- **The scores export takes absolute values**, so sign is lost, and head detail collapses to the module level for SafeTune.

## CuratorKIT (0.1.21, dd9e835)
- ⭐ **No language ID and no group-aware split or leakage check.** Translations of one prompt can land in both splits.
- **The multilingual Detoxify model covers none of hi, bn, ta, sw or ar**, and the toxicity gate rejects rather than scores.
- **No translation task, and exporters drop `metadata`** (language and labels are lost).
- **`SchemaGate` counts words**, so short Indic prompts are rejected unless `min_tokens` is lowered.
