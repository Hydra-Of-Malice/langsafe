# Idea Bank and Ranking

This is the draft as of 2026-11-02, before the Discord scan. Re-rank after we see what other teams are building.

## Constraints every idea must satisfy
- Use one of the 7 Aya models. **Tiny Aya (3.35B)** is the practical choice: it runs in all six kits and fits a T4.
- Chain **at least 2, ideally 3+ Lexsi repos** into one pipeline, with `lexsi_provenance.json` passed between them.
- Report before/after results on **more than one metric**. Use Lexsi's **three-axis** framing: target metric, over-refusal or false positives, and capability retention.
- Make everything public on HF (collections of models and data, plus logs), with an HF article that follows the 9-item checklist.
- Build window: **Nov 2–14 (about 12 days)**. Compute: Colab/Kaggle T4, plus $20 of HF credits, plus $200 if shortlisted.
- Cohere's acceptable-use policy: don't publish raw harmful generations, and present the ROI section as hypothetical.

## Gaps we found that we can exploit

| Gap | Evidence | Why it matters |
|---|---|---|
| **Every Lexsi safety component defaults to English.** AuditKIT's RefusalDetector is English regex; its toxicity metric is toxic-bert; SafeTune has no MultiJail or Aya red-team loader and uses the English-centric WildGuard as judge; CircuitKIT has no cross-language tasks (only on the v1.2 roadmap); CuratorKIT uses MiniLM and English Detoxify | `04_toolkit_capability_map.md` | The tooling can't measure safety correctly on a **multilingual** model, yet the hackathon mandates one. Fixing this is useful to them and can be upstreamed |
| **Tiny Aya's regional merges lose 4–10 points of worst-language safety** (MultiJail: Global 87.0, Fire 78.1, Earth 77.5) | Tiny Aya tech report, Table 7 | A ready-made, *real* drift → repair story. We don't have to manufacture the drift |
| **Tiny Aya Global over-refuses the most** in its class (XSTest 10.4% vs 3.6% for Gemma3-4B) | Same | Repairs must not make over-refusal worse. Reducing it would be a bonus |
| **Lexsi's stated future work** after the Gemma-4 repair: agentic and prompt-injection settings, the fairness shift from multilingual edits, per-language benign refusal, native-script fidelity, running on more open models | Gemma-4 article | Working on their own to-do list maximises research fit |
| **4-bit quantisation reverses safety and unlearning** (Lexsi finding: up to 83% recovery) | Stack launch article | Checking that a repair survives quantisation is a test that almost nobody runs |
| **The CircuitKIT → SafeTune adapter exists, but no published comparison uses it** | Toolkit map | "Circuit-targeted vs all-linear" is an open, cheap and clean ablation |
| **AgentTune has no safety rewards and no injection evaluation** | Toolkit map | Leaves an opening for an agent-safety project |
| Already done in Expedition Tiny Aya: code-mixed safety *audit*, emergent misalignment, language-aware quantisation, an SAE analysis | `05_past_winners_and_judging.md` | Don't repeat an audit on its own; add **repair and verification** |

## The idea bank (15)

How to read the scores (1–5):
- **Fit**: how close the idea is to Lexsi's research.
- **Novelty**: how much ground it covers that isn't already crowded.
- **Feasibility**: how doable it is in 12 days on free compute.
- **Use**: practical usefulness plus how measurable the results are.

| # | Idea | Track | Repos | Fit | Novelty | Feasibility | Use | Σ |
|---|---|---|---|---|---|---|---|---|
| 1 | **Regional-merge safety drift → circuit-located repair → survives 4-bit** (Tiny Aya Fire/Earth vs Global) | 2b | Curator, Circuit, SafeTune, Audit | 5 | 5 | 4 | 4 | **18** |
| 2 | **Indic BFSI assistant with safety-preserving fine-tuning** (plain SFT vs SafeTune Harden vs circuit-guided SafetyLoRA) + ROI | 1 / 2b | Curator, AlignTune, SafeTune, (Circuit), Audit | 5 | 4 | 4 | 5 | **18** |
| 3 | **Rule-reading compliance checker**: train so verdicts flip when the cited rule is removed, swapped or negated (extends *Verdict Without the Rule*) | 1 | Curator, AlignTune, Audit | 5 | 5 | 3 | 4 | **17** |
| 4 | **Agentic prompt-injection drift and repair**: a KYC/claims tool agent trained with GRPO, injections hidden in tool outputs, an injection-resistance reward, a self-heal loop | 3 (+2) | AgentTune, Audit, (SafeTune) | 5 | 5 | 2 | 4 | **16** |
| 5 | **Cross-lingual unlearning that sticks**: forget a hazardous or PII slice; test recovery via translation into other languages, NF4/GGUF and a small relearn | 2b | SafeTune, Circuit, Audit | 5 | 4 | 3 | 3 | **15** |
| 6 | **Circuit-aware mixed-precision quantisation that protects refusal per language** vs uniform 4-bit | 2b | Circuit, SafeTune, Audit | 4 | 4 | 3 | 4 | **15** |
| 7 | **Over-refusal repair for Tiny Aya Global**: lower benign refusal per language with steering or ReSta, without losing harmful refusal | 2b | SafeTune, Audit, (Circuit) | 4 | 3 | 5 | 3 | **15** |
| 8 | **Activation-probe hallucination abstention** on provenance-grounded answerable and unanswerable QA (multilingual) | 2b | Curator, Circuit, Audit | 4 | 3 | 4 | 4 | **15** |
| 9 | **Indic support-ticket triage / structured extraction that replaces a closed API**: GRPO with a JSON-schema reward, plus ROI | 1 | Curator, AlignTune, Audit | 3 | 2 | 5 | 5 | **15** |
| 10 | **Multilingual multi-turn red-team audit with turn-of-failure transcripts**, then a SafeTune fix | 2a | Curator, SafeTune, Audit | 4 | 2 | 4 | 3 | **13** |
| 11 | **Multilingual tool-calling distillation** from Aya Expanse 32B → Tiny Aya, which has no tool template | 3 | AgentTune, Audit | 3 | 3 | 3 | 4 | **13** |
| 12 | **Self-healing SQL/finance data agent** with verified rewards and a DECIDE audit trail | 3 | AgentTune, Audit | 4 | 3 | 2 | 4 | **13** |
| 13 | **Tamper-resistant safety** against the `multilingual_finetune` attack (TAR / Booster / Vaccine) | 2b | SafeTune, Audit | 4 | 3 | 2 | 3 | **12** |
| 14 | **Reward-hacking-resistant multilingual judge** for GRPO, using AlignTune's alignment audit | 3 / 1 | AlignTune, Audit | 3 | 3 | 3 | 3 | **12** |
| 15 | **"Safety passport"**: one evidence report joining provenance across all kits via AuditKIT `join_evidence` | any | all | 4 | 3 | 4 | 3 | **14**, best used as a *layer on top of* another idea |

## Top 3 in depth

### 🥇 Candidate A: "Same model, different language, different safety" (ideas 1 + 7 + 15)

**Thesis, one falsifiable sentence:**
> Tiny Aya's regional specialisation silently erodes refusal in exactly the languages it targets. A repair located with CircuitKIT and applied with SafeTune restores per-language safety **without raising benign over-refusal or hurting multilingual capability**, and the repair **still holds after 4-bit quantisation**.

**Pipeline:**
1. **CuratorKIT.** Build a native-script plus code-mixed evaluation and repair set from `aya_redteaming`, MultiJail and XSTest-style benign look-alikes, covering Hindi, Hinglish, Tamil, Bengali, Swahili and others.
   - Configure the multilingual Detoxify model and a multilingual embedder for dedup.
   - Keep repair data and held-out data strictly disjoint.
   - Publish only labels and prompts, never harmful completions.
2. **AuditKIT, baseline.** Score Global, Fire and Earth on three axes, per language: harmful refusal, benign over-refusal, and capability (Global-MMLU-Lite, mDolly or include-lite).
   - Use a **multilingual refusal detector** plus a second judge (Aya Expanse 32B via API, or a guard model), with a hand spot-check.
   - Test with and without the 366-token default preamble.
3. **CircuitKIT.** Run EAP-IG on a refusal-vs-compliance task for each language.
   - Measure how much the refusal circuits overlap across languages and between Global and Fire.
   - Locate where the regional merge damaged them.
   - Check faithfulness with the 6 pillars.
4. **SafeTune.** Compare four repair arms:
   - (a) ReSta at α≈0.25
   - (b) C-ΔΘ / circuit-restricted edit, fed by CircuitKIT's `scores.json`
   - (c) refusal-direction steering, the baseline Lexsi showed causes over-refusal
   - (d) merge-back interpolation toward Global, the naive baseline
5. **Verify.**
   - Re-run AuditKIT on all three axes.
   - Then **quantise to GGUF Q4_K_M and NF4 and re-verify.**
   - Optionally run a small benign fine-tune and re-verify.
6. **Deliver.**
   - HF collections: repaired adapters and checkpoints, eval sets, all logs.
   - An HF Space demo: type a prompt in any language and compare base vs repaired, with a refusal verdict.
   - The HF article.
   - **Upstream PRs:** a multilingual RefusalDetector for AuditKIT, MultiJail and aya-redteaming loaders for SafeTune, and a cross-language refusal task for CircuitKIT.
   - Name a deployment for usefulness: a public-service or banking helpline in India or Africa running a 4-bit model on-prem.

**Why it wins:**
- It follows Lexsi's Gemma-4 work almost exactly, on a hackathon model, and tackles their *stated future work*: per-language benign refusal, native-script fidelity, the fairness shift, more open models.
- It uses 4 repos plus the only coded bridge between them (CircuitKIT → SafeTune).
- The drift is real and documented, not manufactured.
- It moves past what Expedition Tiny Aya did (audit only) into repair plus a quantisation check.

**Risks:**
- Refusal circuits may not be cleanly language-specific. That would still be a finding, and we would report it honestly.
- Building the multilingual judge takes time.
- The `transformers` v5 pin vs the tokenizer bug matters less here because there is no math.

**Compute:** everything fits a T4. Discovery runs on Tiny Aya only. Use the $20 credits for the judge and the final evaluation sweep.

### 🥈 Candidate B: "Hinglish banking assistant that stays safe" (idea 2 + an ROI layer)

**Thesis:**
> Fine-tuning Tiny Aya Fire on Indian banking regulations (RBI/SEBI FAQs) gives closed-API-level accuracy at about 1/20th the cost. Plain SFT erodes safety; SafeTune hardening (and circuit-targeted LoRA) keeps it.

**Pipeline:**
1. CuratorKIT turns public regulator FAQs and circulars into grounded QA, with hallucination gates, in Hindi, English and Hinglish.
2. Train three arms:
   - AlignTune SFT (plain)
   - SafeTune Harden (SafeGrad or Lisa)
   - circuit-guided SafetyLoRA
3. AuditKIT measures task accuracy and groundedness, harmful refusal, over-refusal, and AlignTune's BFSI alignment probes.
4. Add an ROI table against GPT-class API pricing.

**Why it's strong:**
- This is Lexsi's commercial sweet spot: regulated, Indic, small on-prem specialists.
- It has the highest "practical usefulness" score, and the structure is clean.

**Weakness:**
- It's the most predictable idea. Many teams will build "domain assistant + safety". Novelty depends on the circuit-guided arm and the multilingual safety evaluation.

### 🥉 Candidate C: "Injection-proof KYC agent" (idea 4)

**Thesis:**
> Tool-use RL on Tiny Aya makes agents *more* likely to follow instructions injected into tool outputs. An injection-resistance reward plus AgentTune's self-heal loop fixes this without hurting task success.

**Pipeline:**
1. AgentTune GRPO on a KYC/claims tool environment.
2. AuditKIT: `injection_success` / `system_prompt_leak` detectors on the agent's episodes.
3. A safety reward we write ourselves.
4. Self-heal loop: detect → classify → retrain → gate.

**Why it's attractive:**
- It is Lexsi's *explicitly stated* next step.
- 7 of their 18 open roles are about agents.
- AgentTune has a gap here.

**Weakness:**
- AgentTune is the least mature kit (known silent-zero bugs, `extract_dpo_pairs` broken).
- GRPO on a T4 is slow, and Tiny Aya has no native tool template.
- This is the highest execution risk of the three.

## Current recommendation
**Lead with A.**
- It has the strongest research fit with Lexsi's own work and their open questions.
- It is the most novel relative to Expedition Tiny Aya, feasible on free compute, and naturally uses 4 repos.
- It has an obvious upstream-contribution story.

To raise A's "practical usefulness" score, borrow **B's deployment framing**: a named Indic helpline or banking use case, plus an ROI and on-prem 4-bit cost table.

Keep C as the fallback if the Discord scan shows multilingual repair is crowded.

## Before committing (next steps)
1. **Discord scan:** what are teams building in #hackathon-lexsi, #find-teammates and #report-issues? Is multilingual safety crowded? What do the organisers say in office hours?
2. **Open a Discord ticket** to validate A. Ask the open rule questions in `01_official_brief.md` §11, especially whether we can use Aya Expanse 32B as a judge and cross-track repos.
3. **Smoke test, day 1:**
   - Load Tiny Aya Fire in SafeTune + AuditKIT on Colab.
   - Reproduce the MultiJail worst-language gap on about 200 prompts.
   - Run CircuitKIT EAP-IG on a small refusal task.

   If any of these breaks, re-rank.
4. Draft the HF article skeleton early, using the 9 checklist headings.

## Rough 12-day plan for A
| Days | Work |
|---|---|
| Nov 2–3 | Discord scan, ticket, environment and smoke tests, freeze the spec ("pre-registered" metrics and thresholds) |
| Nov 4–5 | CuratorKIT eval/repair sets; multilingual refusal judge; baseline AuditKIT sweep (Global / Fire / Earth × languages × preamble on/off) |
| Nov 6–7 | CircuitKIT discovery per language plus faithfulness; cross-language overlap analysis |
| Nov 8–10 | SafeTune repair arms a–d; three-axis evaluation; pick the winner |
| Nov 11 | 4-bit (GGUF + NF4) re-verification; optional benign fine-tune re-verification |
| Nov 12 | HF collections, Space demo, upstream PRs |
| Nov 13 | HF article (9 headings, cost table, limitations and dual-use note), reproducibility pass |
| Nov 14 | Buffer and submission |
