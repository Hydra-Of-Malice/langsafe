# Lexsi Toolkit Capability Map (2026-11-02)

Method: shallow clones of all six repos, then a read of their READMEs, pyproject files, docs and source, plus the `gh api` (issues, PRs, releases) and PyPI. **[seen]** = verified in code, docs or API. **[inferred]** = my reading.

## 0. Org inventory (github.com/Lexsi-Labs, 17 repos)

| Repo | Stars | Last push | Notes |
|---|---|---|---|
| TabTune | 141 | 2026-10-15 | Tabular and time-series FMs. Not a hackathon repo |
| Orion-MSP / Orion-BiX | 51 / 16 | 2025 | Tabular ICL models |
| **aligntune** | 48 | 2026-11-02 | |
| **CuratorKIT** | 34 | 2026-10-13 | |
| **CircuitKIT** | 29 | 2026-10-08 | |
| DLBacktrace | 28 | 2026-05 | Relevance-propagation XAI. No Cohere support |
| **SafeTune** | 20 | 2026-10-21 | |
| xai_evals | 18 | 2025-11 | |
| **AuditKIT** | 18 | 2026-10-07 | |
| **AgentTune** | 12 | 2026-10-07 | |
| alignbeam | 0 | 2026-10-04 | README says "Code coming soon" |
| c-theta-delta-demo | 0 | 2026-03 | Static demo page |

**Licences [seen]:**
- AlignTune is MIT, but the Unsloth backend pulls in AGPL.
- CuratorKIT is LSAL v1.1. SafeTune, CircuitKIT and AuditKIT are LSAL v1.2: source-available, free for research, commercial use needs a licence. SafeTune also forbids deploying unrepaired drifted checkpoints.
- AgentTune is "NOASSERTION".

**The organisers are actively fixing these repos for this hackathon [seen]:**
- SafeTune and AgentTune changelogs have "Hackathon fixes (Cohere models)" sections.
- SafeTune has `tests/runtime/test_hackathon_*.py`.
- AlignTune has a `fix/hackathon-pins` branch.

**Shared glue [seen]:**
- `lexsi_provenance.json` (`lexsi.provenance/1`) is written by CuratorKIT, AlignTune, AgentTune, SafeTune and CircuitKIT. Each tool nests the provenance of its inputs.
- A CuratorKIT output folder loads directly with `load_dataset(dir, "sft_messages")`.
- All six kits pin `transformers>=5.15,<6`. ⚠ This conflicts with the Tiny Aya tokenizer bug (#49176, numbers split digit by digit in v5). See `03_models_and_compute.md`.

---

## 1. CuratorKIT, v0.1.22
- **Pipeline:** ingest → clean/dedup → hygiene → generate → quality gates → adaptive recovery → export.
- **Connectors:** jsonl, json, csv, parquet, HF, PDF. PDF goes through MinerU, which is AGPL.
- **Hygiene:**
  - `SecretsGate`.
  - `PIIPseudonymizer` (Presidio).
  - `ToxicityGate`: Detoxify first, then an LLM judge on borderline cases.
- **Generators:** qa, preference, grpo_rollout, multiturn, evol_instruct, cot, adversarial_preference, adversarial_qa.
- **Gates:** hallucination (grounding against the source chunk), reward (multi-dimension LLM judge), diversity, schema.
- **Diagnostics:** a `FailureMode` taxonomy for rejected samples (source_ambiguous, generator_parametric, threshold_marginal, near_duplicate…). Recoverable samples are repaired by `reward_refine` and `pass2`.
- **Exporters:** alpaca, sharegpt, messages, dpo, grpo, ppo, corpus.
- **LLM access:** LiteLLM and Ollama.
- **Usage:** `Curator(CuratorConfig(...)).run()` or `curatorkit run pipeline.yaml`.
- **Outputs:** manifest, rejected.jsonl with reasons, dataset card, HF README, provenance, checksums, diagnostic summary.
- **Model use:** Aya can be the generator or the judge (`cohere_chat/c4ai-aya-expanse-32b`, `ollama/aya-expanse`, or a vLLM endpoint). Tiny Aya needs local serving.
- **⚠ English defaults:**
  - Dedup and diversity use `all-MiniLM-L6-v2`.
  - Detoxify uses `unbiased`.
  - Presidio uses `language="en"`.
  - Multilingual Detoxify and stanza PII are available but must be configured.
- **Maturity:**
  - 9 Colab notebooks, arXiv papers 2606.21631 and 2606.11127. No open issues.
  - Six open fix PRs. **GRPO export is the least reliable path:**
    - gold answer leaks into the prompt (#36) and is dropped from `grpo.jsonl` (#34);
    - failed scoring defaults to 0.5 (#35);
    - plus fixes for non-finite judge scores, overlapping PII spans and empty batches.
- **Gaps:**
  - Multilingual dedup, diversity and hygiene, e.g. with LaBSE or multilingual-e5.
  - Small judges such as Tiny Aya often fail to return JSON.
  - Aya Vision only gets text.

## 2. AlignTune, v0.1.28
- **Backends:** TRL, Unsloth or gradient-free ES.
- **Algorithms:** SFT, DPO, Online-DPO, PPO, GRPO, GSPO, DAPO, Dr.GRPO, GBMPO (4 divergences), Counterfactual GRPO, PACE, ORPO, SPIN, RAFT, and distillation (standard and SDFT).
- **PEFT:** MoA, Text2LoRA, Doc2LoRA.
- **Long context:** RoPE linear/NTK/YaRN, S2-Attn, SWA, packing.
- **Merging:** vendored mergekit plus LoRA merge.
- **Verifiable rewards** (`aligntune.rewards.verifiable`): math, code execution, SQL, **JSON schema**, regex.
- **Indic:** a tokenizer extension for 6 scripts, and `aligntune indic-eval` (MILU, IndicXTREME, IndicGenBench).
- **Safety:**
  - `AlignmentAuditor` / `AlignmentAuditCallback` with **BFSI, legal and healthcare probes**.
  - `RefusalRate`, plus `SafetyReward`, `ToxicityReward` and `HarmlessnessReward`, which default to English-only `unitary/toxic-bert`.
- **Usage:**
  - Python: `create_sft_trainer`, `create_rl_trainer(algorithm=...)`.
  - CLI `aligntune` / `at`: train, recipes, validate, diagnose, advise, merge, aligner, export, verify-export, adapters, compose, indic-eval.
  - YAML recipes.
- **Cohere support [seen]:**
  - `hf_compat.py` adds `Cohere(2)ForSequenceClassification`, so reward and PPO value models can be built from Aya.
  - The loader handles Tiny Aya Base having no chat template.
  - Notebooks: `test_algos_trl_aya.ipynb` and `test_algos_unsloth_aya.ipynb`.
- **Integration:**
  - Consumes CuratorKIT folders and writes provenance.
  - The code comment says "AuditKIT does not read this file yet", yet AuditKIT has AlignTune importers. The provenance join is thin.
- **Open bugs:**
  - #36: SFT runs at least 10 steps when `max_steps` is unset.
  - #37: the core install is heavy.
  - #38: PEFTMerger configs are misread.
  - #44: Counterfactual GRPO uses the GSM8K parser on MATH.
  - Open PRs: an SGLang rollout backend and math-grading fixes.
- **Gaps:**
  - **`recipes/configs/compositions/full_stack.yaml` references 4 config files that don't exist.**
  - Its "audit" stage is only an SFT pass.
  - `docs/novelty_frontiers.md` lists unimplemented ideas: R1 pipeline, MoA-RLHF, PKE, curriculum RLHF, MTP, diffusion refinement.
  - **No Aya recipe and no safety-aware recipe.**

## 3. AgentTune, v1.1.3
- **Training:** `create_agentic_trainer(algorithm=GRPO|PPO|DPO|RLOO|BCO)` runs multi-turn tool-calling rollouts. Notebooks also cover GSPO and KTO. There are also `create_rag_trainer` and `create_distill_trainer`.
- **Agent spine:**
  - `Project`, `Harness` (`DictToolHarness`, OpenEnv sandbox), `AgentStrategy` (ReAct, Plan-and-Solve, Reflexion, MemoryReAct, ToT).
  - Memory: in-context, vector or graph.
  - `EventLog` has a light tier and a full tier (token spans plus logprobs).
- **DECIDE:** turns YAML business rules into a LangGraph StateGraph, with an audit log and destinations such as Postgres and webhooks.
- **Self-healing loop** (`decide.closed_loop`): FailureDetector → Classifier → TrainingExampleGenerator → RetrainingTrigger → DeploymentGate.
- **Agentic RAG:** SQLite-FTS or Chroma, plus a 6-stage docs-to-multi-hop-QA synthesis pipeline.
- **Multi-agent:** `AgentTuneGraph`.
- **Rewards:** 23 builtin rewards, e.g. correctness, structure, search_grounding, tool_chain, hybrid_prm, distilled_judge, finqa, sql.
- **Judges:** `LLMJudge` via API, transformers or vLLM.
- **Tools:** sql, code, file, web search, GitHub, Slack, Playwright, finqa, openenv.
- **Cohere support [seen]:**
  - Aya templates don't render tools, so AgentTune puts tool schemas into the system prompt and passes results back as user turns.
  - It parses Cohere's native `<|START_ACTION|>`.
  - It handles strict turn alternation.
  - **The README has a Tiny Aya GRPO example that runs on CPU or Mac without vLLM.** Python 3.12 and torch 2.11.0 are pinned.
- **Integration:**
  - With tools, GRPO and RLOO write `trajectories.jsonl`, which `auditkit.agent_eval.episodes_from_agenttune` reads directly.
  - `episode_from_eventlog(log)` converts an EventLog.
- **Safety content: almost none.** Notebook 14 covers reward defences (clamping, judge rotation, groundedness). **No refusal or prompt-injection rewards, no safety benchmarks.**
- **Maturity:** only 5 commits public, the fewest stars. `known-issues.md` lists:
  - a missing `langchain_community` can crash tool fetching;
  - `DistilledJudge` silently returns 0;
  - ROUGE and BLEU silently return 0.0 when their dependency is missing;
  - `StageWiseRunner` ignores routing;
  - **`AuditReader.extract_dpo_pairs` always returns `[]`** (use `CollectRunner` instead);
  - no `--tools` CLI flag (PR #5 adds one).
- **Gaps:** agent-safety training, closing the loop from the DECIDE audit log to DPO pairs, multilingual tool-use evaluation.

## 4. SafeTune, v0.1.10 (EMNLP 2026 demo)
- A "library, not a pipeline". Every method has a **faithfulness-audit badge** (Faithful / Simplified / Variant / Wrong / Stub) with `file:line` evidence in `docs/reference/feature-map.md`.
- **Harden, 27 methods:** SafeGrad, Lisa, Vaccine/TVaccine, Booster, RepNoise, TAR, SaLoRA, DeRTa, MART, DeepRefusal, AsFT, SAP, DOOR, LookAhead, Antibody, Surgery, SEAM, CTRAP, SEAL, ConstrainedSFT, LoXHarden, CST/EMA, and others.
- **Recover, 26 methods:** C-ΔΘ, **ReSta**, SafeMerge, LoX, Safe-LoRA, Safe-Delta, Antidote/v2, MSCP, NLSR, PKE, SafeReAct, QReSafe, AAQ, LSSF, WiSE-FT, safety-vector restore, one-shot patch, TracIn, and others.
- **Unlearn:** RMU, NPO, GradAscent, GradDiff, flat, SimDPO, CRISP (Variant).
- **Steer:** refusal direction (Arditi), CAA, linear-probe guard, SafeDecoding, AlphaSteer, RepBend, SafeSteer, SafeSwitch, SCANS, STA, Circuit Breaker / RR, Nudging, AdaSteer, CAST. The vLLM backend is only partially faithful for SCANS and AlphaSteer (code TODO).
- **Interpret:** `identify_safety_neurons`, `safety_circuit_info`, `CircuitInfo`.
- **Evaluate:**
  - Judges: StringMatch, HarmBench classifier, StrongREJECT (OpenAI), and **WildGuard 7B by default (gated, ~14.5 GB, needs a GPU)**.
  - Also `SpectralEntropyMonitor`, Pareto utilities, `TamperBenchEvaluator`, and red-team attacks (Abliteration, BoN).
- **Benchmarks:**

  | Type | Benchmarks |
  |---|---|
  | Jailbreak | harmbench, wildjailbreak, sorrybench_v1, advbench, hexphi, ailuminate, orbench_toxic, beavertails, jailbreakbench, safedialbench |
  | Over-refusal | orbench, orbench_hard, xstest |
  | Capability | gsm8k, humaneval, medmcqa, mmlu |
  | Domain | star1, muse, rwku |
  | Also loaded | agentharm, saladbench, airbench, cares |

  - TamperBench attacks include `multilingual_finetune`, backdoor, gcg and refusal_ablation.
  - **There is no multilingual safety benchmark: no MultiJail, XSafety, PolyGuard or Aya red-teaming loader.**
- **Usage:** CLI `safetune train|patch|eval|unlearn|list` with YAML. New methods plug in through `register_harden(...)`.
- **Cohere notes:**
  - The ~366-token preamble sizes `max_len`.
  - **ReSta works on Tiny Aya at α≈0.25; α=1 breaks the model.**
  - Refusal-direction layer: set it by hand (≈24 of 36) if needed.
  - On Colab, run `pip uninstall torchao`.
- **Integration:**
  - **`safetune/core/circuit_kit/adapter.py` reads CircuitKIT `*_scores.json` (`safety_units`, `layer_suggestions`, `node_scores`) into a `CircuitInfo`, which drives SafetyLoRA targets and neuron-level code.**
  - The CLI accepts CuratorKIT folders, and AuditKIT has SafeTune importers.
- **Maturity:** 10 Colab notebooks, including `full_pipeline`. No open issues. The core runs on CPU, but evaluation needs a GPU.

## 5. CircuitKIT, v0.2.0
- **Pipeline:** discover → faithfulness (6 pillars) → prune / quantize / edit / steer / fine-tune → export an HF checkpoint → lm-eval.
- **Algorithms:**
  - 6 stable: eap, eap-ig, eap-gp, acdc, ibcircuit, cdt. CD-T is approximate on RoPE models.
  - 7 research: eap-ig-activations, eap-clean-corrupted, eap-exact, atp-gd, relp, peap, eap-ifr.
- **Editing:** ROME, MEMIT, CAKE, MCircKE (MLP targets only).
- **Fine-tuning and quantisation:** circuit-restricted LoRA, selective fine-tuning, circuit-aware mixed-precision quantisation.
- **Tasks:**
  - ioi, greater_than, sva, double_io, capital_country, **gender_bias**, hypernymy, boolq, glue, gsm8k, ifeval, mmlu, truthfulqa, winogrande, **wmdp**.
  - **Safety tasks** (advbench, harmbench, sorrybench, jailbreakbench, truthfulqa, stereoset) are paired by `SafetyPromptAdapter` as refusal-vs-compliance or harmful-vs-benign.
- **Corruptions:** entity / role / voice / token / colour swap, negation, paraphrase, distractor, position shift.
- **Usage:**
  - Python: `Pipeline(model, task).discover().evaluate().prune().export()`.
  - CLI: discover, discover-yaml, discover-smart, evaluate, transfer-matrix, heal, steer, benchmark, hparams, …
- **Model constraints:**
  - Pinned to `transformer-lens==3.8.0`, monkeypatched. **Import circuitkit before transformer_lens.** No nnsight; non-TL discovery is "Not Planned".

  | Model | Status |
  |---|---|
  | Tiny Aya (all 5), Command R7B, Aya Expanse 8B | Experimental. Validated on real weights |
  | Aya Expanse 32B | Registration and parity checks only; discovery needs ~64 GiB |

  - On 7–8B, eap, eap-ig, eap-gp, acdc and ibcircuit exceed a 47 GB GPU and fall back to CPU. cdt does not. ACDC took more than 2.5 hours on CPU at 7B.
  - **Single-token metrics break on the 262k vocabulary.** Use greater_than or a custom single-token task.
- **Integration:**
  - `*_scores.json` carries `safety_units` and `layer_suggestions` for SafeTune.
  - It loads SafeTune and AlignTune outputs.
  - AuditKIT has a `circuitkit_faithfulness_report` importer. It notes that CircuitKIT emits bare NaN.
- **Maturity:** 8 notebooks, all GPT-2 on CPU. Open PRs: graph export, a `get_scores_exact` TypeError, a sign bug.
- **Roadmap gaps:**
  - v1.1: head-level ROME, IBCircuit OOM above 3B, CD-T RoPE validation, HF Hub push for circuits, interactive HTML graphs.
  - v1.2: multi-model circuit comparison, circuit-drift CI, and **"Cross-language support — extend to non-English tasks with multilingual evaluation"**, which does not exist yet.

## 6. AuditKIT, v1.1.3
- A zero-dependency evaluation spine: Dataset → Adapter → Model → Metrics → RunResult, with a sha256 fingerprint per run.
- **API:** `ak.evaluate`, `evaluate_many`, `compare` / `compare_models` (paired bootstrap), `ExperimentDB`, MLflow, an lm-eval bridge, BFCL, HF publish.
- **CLI:** eval, init, list, **redteam**, compare, agent.
- **Metrics:**

  | Category | Metrics |
  |---|---|
  | Text | EM, F1, BLEU, ROUGE, ChrF |
  | LLM judges | GEval, RubricItem |
  | Guard judges | **`GuardJudge`** profiles: llama_guard, wildguard, harmbench, shield_gemma, granite_guardian |
  | Encoder judges | NLI factuality, sentiment |
  | Structured output | code, JSON, regex checks |
  | RAG | groundedness, overlap, RAG-stress (includes a multilingual perturbation) |
  | Hallucination and bias | FactualConsistency, `ToxicityScore` (toxic-bert), HateSpeech, RepresentationSkew |
  | Other | latency and throughput, agent and tool metrics |

- **Red-team:**
  - 4 static probe classes (prompt_injection, jailbreak, encoding, refusal). **The probe file is only 85 lines, about 40 English prompts.**
  - Detectors: keyword, refusal, system_prompt_leak, injection_success.
  - **`RefusalDetector` is English-only regex.**
- **Scenarios:** only arc, gsm8k, hellaswag, humaneval, mmlu, truthfulqa.
- **Backends:** echo, openai, anthropic, hf, lexsi, vllm, litellm, api, groq, openrouter, agent, or any callable.
- **Cohere:**
  - Tiny Aya and Expanse run on hf, vLLM and SGLang.
  - `examples/cohere/` has notebooks for compare_models, metrics_generation, run_lmeval, sglang and tgi_endpoint.
- **Integration:** `agent_eval/lexsi.py` has read-only importers for every sibling kit, plus `join_evidence`.
- **Maturity:** about 3,000 tests, Beta status.
  - Open issue #17: F1 below 1.0 on identical text when a token repeats.
  - **Open PR #9: paired-bootstrap p-values in `compare` are not centred on the null, so they are currently meaningless.** Patch or work around this before reporting significance.
- **Gaps:** multilingual probes and refusal detection, a tiny probe library, few safety scenarios, a thin provenance join, the bootstrap bug.

---

## 7. Natural pipelines (documented)
1. CuratorKIT → AlignTune → AuditKIT (`ak.compare` base vs tuned).
2. CuratorKIT → SafeTune Harden (SafeGrad / Lisa) → SafeTune evaluate / AuditKIT GuardJudge.
3. AlignTune fine-tune → SafeTune Recover (ReSta, α≈0.25 on Tiny Aya) → AuditKIT.
4. **CircuitKIT safety task → `scores.json` → SafeTune adapter → SafetyLoRA / neuron targeting.** This is the only coded bridge between interpretability and safety.
5. AgentTune GRPO with tools → `trajectories.jsonl` → `auditkit.agent_eval`.

## 8. Underexplored combinations that would stand out
- **Multilingual safety on Aya. This is the biggest gap.** Every safety component defaults to English:
  - AuditKIT: the refusal regex and toxic-bert.
  - AlignTune: `SafetyReward`.
  - SafeTune: its benchmarks and the WildGuard judge.
  - CuratorKIT: Detoxify and MiniLM.
  - CircuitKIT: no cross-language tasks (only on the v1.2 roadmap).

  The sanctioned multilingual threat model already exists: TamperBench `multilingual_finetune`.
- **Circuit-guided Harden vs all-linear LoRA.** The adapter exists, but nobody has published the comparison.
- **Agent safety loop.** AgentTune has no safety rewards, but AuditKIT has injection and leak detectors and SafeTune has an agentharm loader.
- **An end-to-end provenance and evidence chain** across all kits, joined by AuditKIT `join_evidence`.
- **Low-risk upstream contributions:**
  - AlignTune's missing `full_stack` configs.
  - AuditKIT bootstrap #9 and F1 #17.
  - CuratorKIT GRPO leak #34–36.
  - AgentTune `--tools`.
  - SafeTune SCANS on vLLM.

**Hardware reality:**
- Tiny Aya is the practical target in all six kits.
- Aya Expanse 8B pushes CircuitKIT's EAP family onto CPU.
- Use Aya Expanse 32B only as an API generator or judge.
- Budget about 14.5 GB of GPU for the WildGuard judge.
