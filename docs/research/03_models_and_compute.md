# Allowed Cohere Models: Specs, Safety, Tooling, Compute

Researched 2026-11-02. **[V]** = verified (HF API, model cards, Tiny Aya tech report arXiv 2603.11510, GitHub source). **[I]** = inferred or estimated.

The official `CohereLabs/*/config.json` files are gated, so the values below come from public mirrors that match the paper.

## 1. Specs

| | tiny-aya-base / global / fire / water / earth | aya-expanse-8b | aya-expanse-32b |
|---|---|---|---|
| Params | 3.35B (~0.5B of it embeddings) [V] | 8.03B [V] | 32.3B [V] |
| HF class | `Cohere2ForCausalLM` [V] | `CohereForCausalLM` [V] | `CohereForCausalLM` [V] |
| Layers / hidden / FFN | 36 / 2048 / 11008 [V] | 32 / 4096 / 14336 [V] | 40 / 8192 / 24576 [V] |
| Heads (Q/KV) | 16/4 GQA, head_dim 128 [V] | 32/8 [V] | 64/8 [V] |
| Vocab | **262,144** [V] | 256,000 [V] | 256,000 [V] |
| Attention | 3:1 sliding window (4096, RoPE θ=50k), with every 4th layer **full attention with NoPE** [V] | full, RoPE 10k [V] | full, RoPE 4M [V] |
| Block | parallel attn+MLP, bias-free LN, SwiGLU, **tied embeddings** [V] | same [V] | same [V] |
| logit_scale | 1.0 [V] | 0.125 [V] | 0.0625 [V] |
| Context | 8K in + 8K out [V] | 8K [V] | card says 128K, config says 8192 [V] |
| Tuning | base is pretrained only; global/fire/water/earth are SFT + merge. **Only Global got preference tuning** (per the paper) [V] | instruct [V] | instruct [V] |
| Languages | 70+ [V] | 23 [V] | 23 [V] |

**Regional variants:**
- **Global:** balanced across all languages.
- **Fire:** South Asia. Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu, Nepali, Urdu.
- **Earth:** West Asia and Africa. Arabic, Persian, Urdu, Turkish, Maltese, Hebrew, Amharic, Hausa, Igbo, Malagasy, Shona, Swahili, Wolof, Xhosa, Yoruba, Zulu.
- **Water:** Europe (31 languages) and Asia-Pacific (12 languages).
- Every variant includes English and code.

**Training:**
- 6T tokens of pretraining, then SFT on translated and multi-teacher "FusioNN" data.
- Each regional model is its regional SFT merged with Global (linear, Slerp or TIES), with a check that safety does not regress.

## 2. Licence and acceptable-use policy: this affects the project
- **CC-BY-NC 4.0** plus the Cohere Labs AUP. Access is gated with automatic approval. [V]
- The AUP bans "generating synthetic data for commercial purposes", including training or benchmarking derivatives. **There is no explicit carve-out for red-teaming.** [V]
- What to do about it [I]:
  - Present the ROI comparison as hypothetical, since commercial use needs a licence from Cohere.
  - **Don't publish unredacted harmful generations.** Release aggregate metrics, and gate or redact any harmful samples.
  - Releasing jailbroken weights is a grey area. Avoid it, or gate the repo.

## 3. Chat template gotchas
- **Tiny Aya always injects a default system preamble of about 366 tokens**, even with `use_default_system_prompt: false`. It contains a "contextual safety mode", the knowledge cutoff, and identity rules. A user system prompt goes under "# Developer Preamble". [V]
  - **Treat the preamble as an experimental variable** and report safety with and without it.
  - SafeTune notes that you need `max_len ≥ 512`.
- Turns must alternate user/assistant, or the template raises an error. [V]
- **Tiny Aya's template has no tool calling.** The tokens `<|START_ACTION|>`, `<|START_TOOL_RESULT|>` and `<|START_THINKING|>` are in the vocabulary but unused. [V]
  - A community model, `RamAnanth1/tiny-aya-hermes-tool-calling`, adds Hermes-style tool calls.
- Aya Expanse has Command-R `tool_use` and `rag` templates. [V]

## 4. Known safety posture (Tiny Aya report, Table 7) [V]

**MultiJail (10 languages), safe-rate, worst language / mean across languages:**

| Model | Min | Mean |
|---|---|---|
| Tiny Aya Global | **87.0** | 91.1 |
| Tiny Aya Water | 82.9 | 89.7 |
| Tiny Aya Fire | 78.1 | 90.0 |
| Tiny Aya Earth | **77.5** | 87.8 |
| Gemma3-4B | 79.9 | 88.7 |
| Qwen3-4B | 1.6 | 85.8 |

**XSTest (English), over-refusal / under-refusal:**
- Tiny Aya Global: **10.4** / 15.5
- Gemma3-4B: 3.6 / 44.0

**Takeaways:**
- Global is the safest model in its size class, but it **over-refuses the most**.
- **Regional merges lose 4–10 points of worst-language safety.** This gives us a ready-made drift → repair story, for example Earth on African languages or Fire on Indic languages.
- Aya Expanse publishes no safety numbers.

**Safety datasets:**

| Dataset | Notes |
|---|---|
| `CohereLabs/aya_redteaming` | Apache-2.0. About 7.4k human-written harmful prompts in 8 languages (en, hi, fr, es, ru, ar, sr, fil), 9 harm categories, global vs local harm labels |
| MultiJail (`DAMO-NLP-SG/MultiJail`) | 10-language jailbreak set used in the Tiny Aya report |
| XSTest | Over-refusal |
| PolyGuard, RTP-LX, M-ALERT | Multilingual safety and toxicity sets |
| Indic sets used in Lexsi's Gemma-4 repair | IndicSafetyBench, Sarvam Indic-Safety, IndicGuard, BharatBBQ |

**Capability benchmarks:** `Global-MMLU(-Lite)`, `include-lite-44`, `global-mgsm`, `m-ArenaHard`, `aya_evaluation_suite`.

**Existing community work (a crowded space to avoid):**
- `crosslingual-em/*`: about 50 emergent-misalignment fine-tunes of Tiny Aya.
- `AfriGuard-tiny-aya-global`.
- No abliterated Tiny Aya exists yet.

## 5. Capability baseline (Tiny Aya Global) [V]

| Benchmark | Score |
|---|---|
| Global-MMLU | 44.9 |
| INCLUDE | 45.1 |
| Global-PIQA | 68.3 |
| mDolly | 86.9 |
| m-ArenaHard | 67.4 |
| GlobalMGSM | 52.8 |
| GPQA-D | 28.3 |

- **Strong at generation and translation, weak at math and reasoning.**
- Aya Expanse MMLU-Pro: 33.7 (8B), 45.4 (32B).

## 6. Tooling compatibility

| Tool | Status |
|---|---|
| transformers | Cohere2 supported. **Bug #49176: in transformers v5, the Tiny Aya tokenizer splits numbers digit by digit.** Pin `transformers==4.57.x` or load with `PreTrainedTokenizerFast(tokenizer_file=…)`. This matters for math, GRPO rewards and number circuits [V] |
| **CircuitKIT** | **Validated on real weights for all 5 Tiny Aya variants and Aya Expanse 8B** (EAP, EAP-IG, EAP-GP, IBCircuit, CD-T, faithfulness, prune, quant, steer). **32B: discovery does not run** (needs ~64 GiB; multi-GPU is broken). ACDC is too slow at 3–8B. With the 262k vocabulary, use single-token tasks such as `greater_than` instead of IOI. Licence: LSAL v1.2, free for academic use [V] |
| TransformerLens | Legacy 3.x has no Cohere support. TransformerBridge v4.2+ has Cohere and Cohere2 adapters [V] |
| **SafeTune** | README includes Tiny Aya notes. Refusal-direction layer is about 24 of 36. **ReSta recovery works on Tiny Aya at α≈0.25; α=1 breaks the model.** Run `pip uninstall torchao` on Colab [V] |
| AlignTune | Supports TRL / Unsloth / ES backends [V] |
| vLLM | Cohere and Cohere2 supported, including LoRA. **Needs compute capability ≥7.5: T4 works, P100 does not** [V] |
| Unsloth | Cohere2 uses the generic FastModel path. The vLLM fast-inference GRPO path probably does not work [I] |
| GGUF | Official `CohereLabs/tiny-aya-*-GGUF` repos exist. Q4_K_M is 2.14 GB [V] |
| GPTQ/AWQ | No Tiny Aya quantised repos exist. Use llm-compressor [I] |

**Analysis gotchas:**
- Tied embeddings mean `W_U = W_E`.
- Fold `logit_scale` before doing logit lens on Aya Expanse.
- The NoPE global layers break positional analyses.
- Logits take about 1 MB per token in fp32.

## 7. Compute (estimates)

| | Tiny Aya (3.35B) | Aya Expanse 8B | Aya Expanse 32B |
|---|---|---|---|
| fp16 weights | ~6.7 GB, **fits a T4** | ~16 GB, does not fit a T4 | ~65 GB |
| 4-bit weights | ~2.2–3 GB | ~5.5–6 GB | ~20 GB |
| LoRA, fp16 base, T4 | ~10–13 GB at seq 2048, batch 1, with gradient checkpointing | no | no |
| QLoRA, T4 | ~6–9 GB | ~10–14 GB at seq ≤2048 | no |
| Full fine-tune | needs an A100-80GB (credits) | no | no |
| LoRA throughput, T4 | ~3–5M tokens/hour | ~200–400 tok/s | n/a |
| Decode, T4 | ~20–35 tok/s at batch 1; hundreds of tok/s batched in vLLM | ~8–30 tok/s | n/a |

**Pitfalls:**
- The vocabulary dominates memory. Use chunked or fused cross-entropy (Liger or Unsloth) and small GRPO groups.
- T4 has no bf16, and the model was trained in bf16. Watch for fp16 overflow.

**Where to run what:**
- **GRPO:** Kaggle 2×T4 is the realistic free setup, with `trl vllm-serve` on GPU1 and training on GPU0. Kaggle gives about 30 GPU-hours a week. Colab with a single T4 works but is slow.
- **ZeroGPU:** for Space demos only.
- **HF credits ($20, then $200 if shortlisted):** spend them on the final A100 runs.
- **Aya Expanse 8B GRPO:** not feasible on free tiers.

## 8. Recommendation

| Goal | Model | Notes |
|---|---|---|
| Multilingual safety / red-teaming | **tiny-aya-global**, plus **fire** (Indic) and/or **earth** (Africa / W. Asia) | Regional variants are the drift-and-repair story. Datasets: aya_redteaming, MultiJail, XSTest |
| CircuitKIT interpretability | **tiny-aya-global** (or **base** for clean circuits) | Avoid the 32B |
| RL agents | **tiny-aya-global** | GRPO on Kaggle 2×T4, then a credits run. You must define your own tool-call format |
| Comparison point | Aya Expanse 8B, 4-bit | |
