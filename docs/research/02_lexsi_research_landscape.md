# Lexsi Labs: What They Actually Work On

Researched 2026-11-02. Tags: **[V]** = checked against a primary source (lexsi.ai, arXiv API, HF API, GitHub API, Screenloop job posts). **[I]** = inference.

## 0. Identity

- Lexsi Labs was formerly "AryaXAI Alignment Labs". The name stands for "Le eXplainable Safe Superintelligence". aryaxai.com redirects to lexsi.ai. [V]
- Founder and CEO: **Vinay Kumar Sankarapu** (Arya.ai co-founder, 2013; IIT Bombay). He is an author on every Lexsi paper. [V]
- **Pratinav Seth** [I] is the de facto lead for alignment, safety and interpretability. He is senior author on SafeTune, CircuitKIT, C-ΔΘ, MANSU, ALIGNBEAM, CuratorKIT, AuditKIT and AgentTune. His background includes uncertainty and calibration, medical imaging and multilingual NLP.
- **Ram Mohan Rao Kadiyala** [V] hosted the kickoff and co-authored AuditKIT and AgentTune. He works on multilingual and Indic NLP and is active in the Cohere community (Global PIQA, Kaleidoscope, Mantra-14B Hindi-English). [I] He is likely the hackathon lead and will care about multilingual and Indic work.
- The safety and tooling team [I]: Saisab Sadhu, Aadit Sengupta, Aditya Kasliwal, Hem Gosalia, Chirag Chawla, Soham Bhattacharjee, Utsav Avaiya, Zera Lyngkhoi.
- The TFM team, probably Paris-based: Mohamed Bouadi, Nassim Bouarour, Aditya Tanna, Shivam Dubey.
- RL theory: Rui Yuan, Mykola Khandoga.

## 1. Their core research bets, ranked

### #1 Weight-level safety: audit → correct → verify instead of perimeter guardrails [V]
Tagline: *"Safety that reaches the model, not just the perimeter."*

**Stack open-sourced, Sept 2026.** Components: CuratorKIT, AlignTune, SafeTune, CircuitKIT, AuditKIT and the MANSU unlearning method. Source: https://lexsi.ai/resources/articles/open-sourcing-the-lexsi-alignment-and-safety-stack-frontier-safety-research-for-auditing-correcting-and-verifying-model-weights

**Three failure modes they documented:**
- Benign domain fine-tuning erodes safety. One clinical model's HarmBench refusal fell from 87.0% to 57.5%.
- **4-bit NF4 quantisation reverses unlearning**, restoring up to 83% of "forgotten" knowledge.
- Compliance systems ignore the rule they cite.

**Gemma-4-31B repair, Oct 2026.** Selected MLP weights were edited offline for multilingual and cyber safety. Source: https://lexsi.ai/resources/articles/the-lexsi-alignment-and-safety-stack-correcting-gemma-4-31b-for-multilingual-and-cybersecurity-safety-in-the-weights-copy

Results:
- Five-turn jailbreak refusal rose from 48.6% to 80.0%.
- FORTRESS attacker-useful output fell from 35% to 8%.

Costs they **reported openly**:
- Over-refusal rose.
- Fairness shifted.
- SWE-Bench Lite fell from 109 to 102.

Evaluation suites: IndicSafetyBench, Sarvam Indic-Safety, IndicGuard, BharatBBQ, MultiJail and others.

**Their stated future work. These are the best hackathon hooks:**
- Extend evaluation to **agentic and prompt-injection settings**.
- Investigate the **fairness shift caused by the multilingual edit**.
- Test **per-language benign refusal and native-script fidelity**.
- Run the loop on more open-weight models.
- Compare against a fine-tuning baseline.

**Papers:**

| Paper | arXiv |
|---|---|
| C-ΔΘ, circuit-restricted weight arithmetic for selective refusal | 2602.04521 |
| Forgetting That Sticks / MANSU | 2605.15138 |
| ALIGNBEAM, inference-time alignment transfer by cross-vocabulary logit mixing | 2606.12342 |
| SafeTune (EMNLP 2026 demos) | 2609.22153 |
| Position: *Behavioural Assurance Cannot Verify the Safety Claims Governance Now Demands* | 2605.15164 |

**Their evaluation style, which we should copy:**
- Every result is scored on **three axes at once: harmful refusal, over-refusal and utility**.
- They write a **spec before the edit**.
- They check judge validity with a 3-judge panel.
- They state residual failures and negative results explicitly.

### #2 Mechanistic interpretability as the locator and the evidence for interventions [V]

| Work | Details |
|---|---|
| **CircuitKIT** (ICML 2026 MI workshop) | arXiv 2607.19317. 13 discovery algorithms, 6-pillar faithfulness suite, prune / edit / circuit-LoRA / circuit-aware quant / steer |
| *Faithfulness to Refusal: A Causal Audit of Neuron Selectors* | arXiv 2607.05355 |
| *Every Ablation Is a Dose: Counterweights and the Semblance of Self-Repair* | arXiv 2610.02173 |
| *Interpretability as Alignment* (position paper) | arXiv 2509.08592 |
| HF dataset `Lexsi/circuitkit-capitals-contrastive` | |

Slide line: *"CircuitKIT finds the circuit; SafeTune measures and repairs."*

### #3 Tabular foundation models: the most papers and the commercial differentiator [V]

- Models: Orion-MSP (2511.02818), Orion-BiX (2512.00181), OrionMoE-v1.0 (2610.20418, Oct 2026).
- TabTune (2511.02802) is their most-starred repo.
- About 8 TFM papers in 2026.
- **Not a hackathon repo.** Out of scope for us, except as background.

### #4 Reproducible open post-training infrastructure with data provenance [V]

| Work | Key points |
|---|---|
| **AlignTune** (ICLR 2026 SPOT workshop, arXiv 2602.09621) | 13+ RL algorithms, 27+ rewards, an alignment audit (reward hacking, sycophancy, refusal collapse), a VRAM/cost/carbon advisor |
| **CuratorKIT** (EMNLP 2026 Findings, arXiv 2606.21631) | 8 generation tasks, 3 quality gates including provenance-grounded hallucination checks, an append-only per-sample provenance chain |
| Provenance-grounded gating paper (2606.11127) | Targeted regeneration recovered 72.7% of failed samples, against 34.6% for naive retry. HF: `Lexsi/provenance-grounded-synthetic-qa` |
| **AuditKIT** (arXiv 2610.22511) | "Evaluation spine": run fingerprints, cached reruns, run comparison. Missing evidence is marked as unscored |

Commercial "specialists beat generalists" posts:
- A 4B SFT model beat GPT-5 (2-shot) on a retail-banking template task.
- A wealth-management alignment case study.

### #5 Counterfactual audits of compliance and legal reasoning [V]

| Paper | arXiv | Finding |
|---|---|---|
| *What Do Compliance Detectors Read?* | 2608.16852 | Probes and guard models largely ignore the stated rule |
| *Verdict Without the Rule* | 2610.12313 | Verdicts don't change when the rule is removed, swapped or negated |
| *Cited but Not Consulted* | 2610.12361 | Legal CoT faithfulness |

HF artifacts: `Lexsi/obl-ex-4b-raw` / `-curated` (obligation extraction LoRAs) and `obl-ex-4b-logs`.

[I] This is the bridge to their banking, insurance and regulated-industry customers.

### #6 RL for reasoning [V]
- *Beyond KL: Bregman divergences for group policy optimization* (2602.04380)
- *Causal credit assignment for policy optimization* (2602.09331)

### #7 Agents: the fastest-growing commercial priority [V]

**AgentTune** (arXiv 2610.13335):
- Lifecycle: build → collect → evaluate → train → distill → **heal**.
- EventLog schema.
- Self-healing loop.
- CPU-first spine.

**Hiring signal:** 7 of the 18 open roles are about agents. The job posts stress:
- Coding, data-science and AI-engineering agents **inside customer networks, often air-gapped**.
- Small models under real cost and latency budgets.
- Rigorous evals: pass@k, variance, failure taxonomies.
- Turning production failures into test cases.

The launch post names an "autonomous AI engineer" as a mission goal.

### #8 Classical XAI: the AryaXAI heritage [V]
- DLBacktrace (2411.12643), xai_evals (2502.03014), XAI-for-compliance metrics (2502.04695), interpretability-aware pruning for medical imaging (2507.08330).
- The legacy console offers ML observability: Backtrace/SHAP, drift, bias, risk policies, and a loan-underwriting example.

## 2. Hugging Face org (huggingface.co/Lexsi) [V]

47 models, 3 datasets, no public Spaces.

| Artifact | Details |
|---|---|
| Tabular | Orion-MSP, Orion-BiX |
| SafeTune drift/recover grid (May 2026), about 35 gated checkpoints | Llama-3.1-8B, Llama-3.2-3B, Gemma-3-4B, Qwen3-4B × gsm8k / code / dolly / medical / legal. Each comes as `*-sft-drift`, `*-dpo-recover` or `*-hh-rlhf-aligned` |
| **Hackathon-ready (Oct 5–6, 2026)** | `tiny-aya-{global,fire,water,earth}-hh-rlhf-aligned` and `aya-expanse-{8b,32b}-hh-rlhf-aligned`: DPO + LoRA on 20k hh-rlhf harmless-base samples, built with SafeTune. **[I] These are ready-made "aligned base" checkpoints for drift → repair experiments** |
| Other | Gemma-4-31B repairs; obl-ex-4b raw/curated |
| Datasets | `provenance-grounded-synthetic-qa`, `obl-ex-4b-logs`, `circuitkit-capitals-contrastive` |

Lexsi's model modifications use the "Lexsi Labs Source Available License v1.2", which is non-commercial. Check this before reusing them.

## 3. "Coming soon" tools (no public info; guesses)

The DevRel job post already names **InterpSuite** and **PruneKIT**.

| Tool | Best guess |
|---|---|
| PruneKIT [I] | Pruning, linked to their "MI-Pruning" pillar |
| InterpSuite [I] | DLBacktrace + xai_evals + interpretable telemetry |
| Reducto [I] | Compression and distillation |
| DL Tune [I] | Fine-tuning for non-LLM deep learning |
| IOE Suite | Unknown |

Research pillars from the launch post [V]: efficient RL, MI-Pruning, safety-aware fine-tuning and unlearning, **interpretable telemetry (activations as live signals)**, the "AI engineer loop", specialised foundation models.

## 4. What the job descriptions emphasise [V]

There are 18 posts on Screenloop under "AryaXAI".

- **AI Research Scientist:** reward modelling, safety fine-tuning, **pruning, unlearning**, circuit discovery, mechanistic interpretability, **information-theoretic and causal analysis**, open tooling, top-venue publications.
- **AI Research Intern** (6 months, conversion possible). Any one of: library development, XAI, mechanistic interpretability (activation patching), **uncertainty quantification**, quantisation and pruning, RLHF and reward modelling, TFMs, LoRA and distillation. Risk-sensitive finance or healthcare experience is a plus.
- **DevRel intern:** cookbooks, tutorials and demos for the tools, plus running the Discord community.
- **Agent roles:** harness and sandboxing, trace schemas as audit records, verifiers and grader calibration, seeds/variance/pass@k, air-gapped on-prem deployment with small models.

## 5. Commercial angle

- **Current pitch:** audit and correct open-weight checkpoints for a named deployment, with risk tolerance that depends on the sector (defence, banking, healthcare, cyber, consumer).
- **Case studies:** retail banking and wealth management with small specialist models, credit risk with TFMs, health data.
- **[I] Thesis:** *small, specialised, auditable open models running on-prem in regulated, Indian-multilingual settings, with model-level safety evidence that governance accepts.*

## 6. Themes that would resonate most

In every case, use their vocabulary: drift / repair / verify, three-axis scoring, provenance. Pre-register a spec. Report costs, residual failures and negative results. Publish HF collections and an ROI table.

1. **Safety that survives deployment.** Drift → locate → repair → **verify after quantisation and after a further benign fine-tune** on Tiny Aya, starting from `Lexsi/tiny-aya-*-hh-rlhf-aligned`.
2. **Indic / code-mixed multilingual safety audit and repair, including the fairness-shift question** they left open.
3. **A rule-sensitive compliance assistant.** Use their *Verdict Without the Rule* counterfactual audit as the before/after evaluation.
4. **Agentic safety drift and prompt injection**, which is their explicitly stated future work.
5. **Unlearning that survives quantisation and tampering.**
6. **Activation-probe hallucination abstention** with provenance-grounded data, tying into uncertainty quantification and interpretable telemetry.

Lower fit: pure TFM work, classical SHAP/LIME, generic chatbot fine-tunes with no safety measurement.

## Sources
https://lexsi.ai · https://lexsi.ai/resources/research-papers · https://lexsi.ai/resources/articles · https://huggingface.co/Lexsi · https://github.com/Lexsi-Labs · https://app.screenloop.com/careers/aryaxai/ · arXiv author feed https://export.arxiv.org/api/query?search_query=au:Sankarapu · https://luma.com/66jkhulx
