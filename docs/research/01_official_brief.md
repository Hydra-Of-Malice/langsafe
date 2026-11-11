# Lexsi Labs – AI Safety and Alignment Hackathon: Official Brief

Scraped 2026-11-02 from https://hackathon.lexsi.ai/ (all sections), https://hackathon.lexsi.ai/kickoff-slides/ (all 34 slides), and the Tally registration form. Raw text is in `raw/`.

## 1. At a glance

| Item | Detail |
|---|---|
| Organiser | Lexsi Labs, with Cohere Labs and Hugging Face |
| Outreach partners | AISafety.com, AI Safety India (AISIN) |
| Format | Fully online. Kickoff (Oct 31) and Demo Day (Nov 22) are in person at Polaris School of Technology, Bengaluru, with an online option |
| Teams | 1–3 people; one team per person; the team lead registers the whole team |
| Cost | Free. Travel not reimbursed |
| Scale (live counter, 2026-11-02 09:21 UTC) | **1,296 teams, 1,603 participants** (1,369 registrations) |
| Shortlist | **10–20 teams** go to Demo Day, so roughly the top 1.5% |
| Stated goal | "chain the stack into a real pipeline and show, with before-and-after metrics, that a small open-weights model becomes meaningfully better, safer or more reliable for a real use case" |
| Organiser's hidden goal | Make their open-source stack public and **gather real feedback and feature requests** on it. Feedback, bug reports and upstream PRs to their repos are likely to count in our favour |

## 2. Timeline (all deadlines AoE)

| Date | Event |
|---|---|
| Oct 12 | Registration opens |
| Oct 31, 4–7 PM IST | Kickoff (slides public; the recording is still marked unavailable on the site) |
| Nov 1 | Build window opens. $20 HF credit coupons emailed at 5:30 PM IST (first 800 to redeem) |
| **Nov 5** | **Registration closes.** Re-registering before this date replaces the team list |
| **Nov 14, end of day AoE** | **Submission deadline.** Submission form is emailed a couple of days before |
| Nov 15–16 | Judging (double-blind, multiple judges per submission, normalised averages) |
| Nov 16 | Up to 20 Demo Day teams announced; they get +$200 HF credits |
| Nov 16–21 | "Prove your thesis" window for shortlisted teams |
| Nov 22 | Demo Day: live demo, then final judging |

## 3. The open-source stack (hackathon repos)

| Repo | What it does | Docs |
|---|---|---|
| **CuratorKIT** | Data curation + synthetic data. Ingest (PDF/JSONL/HF) → clean + dedup (exact, MinHash, embedding) → hygiene (secrets, PII, toxicity) → generate (QA, CoT, DPO pairs, GRPO) → quality gates (grounding, reward, diversity) → export (Alpaca, ShareGPT, DPO, GRPO) | curatorkit.lexsi.ai/latest |
| **AlignTune** | Post-training. Backends TRL / Unsloth / ES; 13+ RL algorithms selectable by one flag; 27+ reward functions (verifiable math, code, SQL); SFT/DPO/PPO/GRPO unified API; multi-stage pipelines; adapters MoA, Text2LoRA, Doc2LoRA; long context (RoPE, S2-Attention, packing); distillation (SDFT) and merging (mergekit); tokenizer adaptation; export GGUF/Ollama/Hub; **alignment audit (reward hacking, sycophancy, refusal collapse)**; utility advisor (VRAM, time, cost, carbon) | aligntune.lexsi.ai |
| **AgentTune** | Agent post-training. TRL + vLLM; 5 RL algorithms, 5 agent strategies; tool-calling rollouts; lifecycle build → collect → evaluate → train → distill → heal; DECIDE (decision workflows as YAML); EventLog (one trajectory schema); distillation teacher → <3B student; agentic RAG; grounded difficulty-tagged QA synthesis; **self-healing loop (detect → classify → retrain → gate)**; runs on CPU first | agenttune.lexsi.ai |
| **SafeTune** | Measure and repair safety drift. Pipeline Drift → Measure → Choose → Repair → Verify → Deploy. Four interventions: **Harden** (train time), **Recover** (weight space, no retraining), **Unlearn** (weight space, forget/retain sets), **Steer** (inference time). Has built-in tools to locate safety-relevant circuits and neurons | safetune.lexsi.ai |
| **CircuitKIT** | Mechanistic interpretability. Model → task (built-in/custom/YAML) → **13 discovery algorithms** (6 stable: EAP, EAP-IG, EAP-GP, ACDC, IBCircuit, CD-T; 7 research: EAP-IG-activations, EAP-clean-corrupted, EAP-exact, AtP-GD, RelP, PEAP, EAP-IFR) → **6-pillar faithfulness** (causal patching, ablation, stability, robustness, baseline comparison, transfer) → intervene (8 prune: node/neuron/weight × random/magnitude/Taylor/Wanda/multi-granular; 8 edit: ROME, MEMIT, CaKE, MCircKE, circuit-guided…; 6 fine-tune: Circuit LoRA, adapter, prefix, BitFit, soft healing, selective; 3 quantise: GPTQ, AWQ, TACQ; 3 steer: activation, circuit weight, composer) → export (HF checkpoint, lineage, lm-eval). "Finds the circuit; does not measure safety or bias itself" | circuitkit.lexsi.ai |
| **AuditKIT** | Evaluation. Dataset (benchmarks, HF, custom) → prompt format (generation, chat, instruction) → model (API, HF, vLLM) → metrics (accuracy, LLM judge, safety) → results (fingerprint, cached reruns, compare runs) | auditkit.lexsi.ai |

**Shared across all tracks:** CuratorKIT and AuditKIT.
**Not part of the hackathon, listed as "coming soon":** TabTune (tabular, already released), DL Tune, IOE Suite, Reducto, PruneKIT, InterpSuite.

**Their own "which tool" guide (slide 20):**
- Fine-tuned model got less safe → SafeTune
- Fine-tune without losing safety → SafeTune
- Remove a capability → SafeTune
- Which components drive a behaviour → CircuitKIT
- Prune/edit only that circuit → CircuitKIT
- Safe enough to ship? → SafeTune + AuditKIT
- **Repair that targets where safety lives → CircuitKIT + SafeTune** (presented as the flagship combination)

**Their reference safety pipeline:** aligned base → fine-tune (AlignTune/AgentTune) → SafeTune (measure drift, repair) → AuditKIT (evaluate before release) → deploy with runtime guardrails, with CircuitKIT "finding where it lives." The "where safety leaks" story: aligned model → fine-tune on benign data → quantise to 4-bit → deploy with I/O-only guardrail.

## 4. Tracks and problem statements

You may submit to more than one track, but **only your final submission before the deadline counts**. Organisers tag each submission to the closest track or sub-track. Ideas outside the listed statements are allowed if they stay within the rules; they recommend validating them through a Discord ticket in #report-issues.

### Track 1: Fine-tuning and Data Curation
Repos: CuratorKIT (data), AlignTune (train), AuditKIT (eval)
1. **Domain assistants for regulated industries**: banking, insurance, legal or healthcare; answers must follow policy and stay accurate
2. **Replacing a closed-model API**: fine-tune a small open model to match a large closed model on a high-volume task (classification, extraction, summarisation, drafting) at a fraction of the cost
3. **Grounded answers, fewer hallucinations**: train from an organisation's own documents; answer only from them; say when the answer isn't there; measurably fewer false claims
4. **Low-resource and multilingual languages**: underserved languages, dialects or code-mixed text in education, public services, agriculture or healthcare
5. **Structured output and extraction**: forms, invoices, reports, clinical notes or tickets → schema-valid structured data
6. **Reasoning with verifiable answers**: maths, finance calculations, code or logic, with reward-based training
7. **Preference and style alignment**: house style or quality bar, using preference data
8. **Data-scarce domains**: generate, verify and filter synthetic data; show that curation beats raw generated data
9. **Long documents**: multi-stage fine-tuning for summarising and reasoning over filings, contracts, transcripts or papers

### Track 2a: AI Safety and Red Teaming
Repos: CuratorKIT, CircuitKIT, SafeTune, AuditKIT
1. **Red-teaming and robustness evaluation**: jailbreaks, prompt injection, encoding tricks, best-of-N, weight-space attacks (abliteration); report robustness **alongside over-refusal**
2. **Adversarial and safety evaluation sets**: domain-, language- or risk-specific benchmark with labels and scoring; the deliverable can be the dataset plus a report
3. **Multi-turn audits**: long conversations, injection hidden in tools or documents, gradual jailbreaks, requests for cyber-attack help; report what broke and in which turn, with reproducible transcripts
4. **Multilingual and code-mixed safety**: test whether safety holds across languages, then close the gaps

### Track 2b: AI Safety and Correction
1. **Safety-preserving fine-tuning**: harden during domain fine-tuning versus plain fine-tuning
2. **Repairing safety drift without retraining**: weight-space methods (safety patching, merging, projection) that keep task performance
3. **Unlearning**: forget/retain sets; prove removal under paraphrased and adversarial prompts; general capability survives
4. **Inference-time safety steering**: refusal directions, contrastive or conditional steering, safety-aware decoding; low over-refusal
5. **Locating where safety lives**: circuits, neurons or directions for refusal or a harm, then a targeted fix
6. **Circuit-level interventions**: circuit steering, circuit-restricted FT or knowledge editing; verify nothing else broke
7. **Safety-aware compression**: circuit-aware mixed-precision quantisation or pruning; safety survives
8. **Tamper resistance**: safety survives later harmful fine-tuning; demonstrate with attack fine-tunes
9. **Hallucination detection and abstention**: activation probes trigger abstention or retrieval; fewer wrong answers while still answering most questions

### Track 3: Agentic Fine-tuning
Repos: AgentTune, AuditKIT
1. **Reliable tool calling**: RL on real tool-calling rollouts (right tool, well-formed args, right order)
2. **Efficient trajectories**: fewer redundant, looping or failed calls and steps, same success rate
3. **Agentic distillation**: teacher trajectories → small local student
4. **Fine-tuning for RAG**: when and what to retrieve; grounded; ignores distractors
5. **Synthetic multi-hop training data**: corpus → grounded, difficulty-graded multi-hop Qs → train a search agent
6. **Data and database agents**: business questions over DBs/APIs, rewarded on verified results
7. **Business workflow and decision agents**: claims, KYC, ticket triage, compliance review with a full audit trail
8. **Self-healing agents**: detect failures → classify → corrective data → retrain → gate redeploy on accuracy
9. **Reward design and reward-hacking defences**: process rewards, LLM judges, groundedness checks that can't be gamed

## 5. Rules (merged from slides and site)

1. **Teams:** 1–3 members, one team per person, at least one member on Discord.
2. **Built during the hackathon** (Nov 1–14).
3. **Repos:** chain repos into **one connected pipeline**. Minimum 2 hackathon repos; 3+ preferred.
   - ⚠ Discrepancy: the slide says "at least 2 repos as one connected system" (any of the 6). The site says "at least 2 repos *from the track* … (except Track 3)." Track 3 only lists AgentTune + AuditKIT. CuratorKIT and AuditKIT are "common, usable across all tracks." **Ask on Discord whether, for example, Track 2 can use CuratorKIT + AlignTune, and what "except Track 3" means.**
4. **Model:** one of 7 dense Cohere Labs models: Tiny Aya Base / Global / Fire / Water / Earth, Aya Expanse 8B, Aya Expanse 32B.
5. **Public on Hugging Face:** models, adapters, logs, datasets, benchmarks.
6. **Public deployment** by the deadline (public demo or repo link). An HF Space if relevant.
7. **Before-and-after metrics: more than one metric.**
8. **Write-up** (HF article): which metrics and why; the reasoning behind datasets, tasks and model; how the result is meaningfully better in use.
9. **Optional but preferred: ROI comparison** of a closed model vs your small model: self-hosting cost vs token billing at a realistic volume, GPU hours, $/GPU-hour, tokens/sec.
10. **Originality:** your own work; OSS libraries and public datasets allowed where licences permit.
11. **Code of conduct:** harassment, plagiarism or disruptive behaviour can lead to disqualification.
12. **IP:** teams keep ownership; organisers may showcase projects.
13. Organisers may change the rules; their decisions are final. Late or incomplete submissions may be excluded.

## 6. Submission checklist (HF article sections, "complete details will be shared soon")

1. Problem statement and approach (brief)
2. Hackathon repos used: which ones, for what, which features
3. Models used + licences
4. Datasets and benchmarks used + licences
5. Model artifacts (weights, adapters…) public as an **HF collection**
6. Data artifacts (datasets, logs, benchmarks…) public as an **HF collection**
7. Experiments run, findings, what else was tested
8. **"With more credits": your approach, before and after results** (a scale-up plan)
9. Teammates tagged

Submission form: emailed a couple of days before the deadline; open until Nov 14 EOD AoE.

## 7. Judging

- **Labelling:** internal team checks validity (public artifacts, write-up, checklist, no duplicates) and tags the closest track or sub-track.
- **Scoring:** startup founders, academics, industry research labs and the organisers' research teams score **double-blind**. Each team gets several scores; normalised averages pick the 10–20 Demo Day teams.
- **Rubric:** Write-up clarity /10 · Practical usefulness /10 · Code/notebook quality /10 · Overall review /100.
- **Demo Day:** live demo, then judges deliberate on final winners.

Implications: the write-up is roughly a third of the explicit sub-scores, and **double-blind means the HF article and artifacts must make the case on their own.** "Practical usefulness" rewards a real use case with an ROI story.

## 8. Prizes and perks ($30k+)

| Place | Lexsi credits | Cohere credits | Other |
|---|---|---|---|
| 1st | $1,750 | $1,750 | 6 mo HF Pro per teammate + hiring |
| 2nd | $1,250 | $1,250 | same |
| 3rd | $900 | $900 | same |
| 4th | $600 | $600 | same |
| 5th | $500 | $500 | same |
| Demo Day finalists (≤20) | | | $200 HF credits + interview invites |
| Valid submissions | | | Certificate; a few strong ones may get interview invites |
| Registered + on Discord (first 800) | | | $20 HF credits (ZeroGPU/GPU), expire 30 days after redemption |

Note that the prizes are **credits**, not cash. The bigger prize is the **hiring pipeline** (Research Scientist/Engineer, Research Intern, DevRel Intern, AI Engineer, SWE).

Credits fine print: emails go to the team lead; HF usernames must match the registration form; HF may ask for a payment method; the $20 coupon works for the first 800 teams; credits expire 30 days after redemption.

Free compute: Google Colab and Kaggle quotas "can be enough for the Tiny Aya models."

Referral rewards (teams that register via your link and make a valid submission): 2 → +$5, 5 → +$10, 10 → +$20, 20 → +$45, 35 → +$80, 50 → +$90 (cumulative).

## 9. Registration form (Tally) notes

- Personal email (not academic or work). Discord, HF, LinkedIn, X and GitHub usernames for every teammate.
- Preferred track; post-training experience; red-teaming experience; idea pitch (changeable later).
- Hiring section: open to interview, domains of interest (top 3), role type, CV of the **team lead only** (teammates get a separate form later).
- The form still says kickoff "Oct 17–18, venue TBD", which is stale; the site says Oct 31 at Polaris.

## 10. Channels

- Discord: https://discord.gg/dtEDQ2Z3eg. Get the role for **#hackathon-lexsi**; also #find-teammates, #report-issues (tickets for idea validation), and office hours several times a week (Events tab)
- Emails come from the hackathon address shown on the site; check spam
- GitHub https://github.com/Lexsi-Labs · HF https://huggingface.co/Lexsi · X @Lexsi_labs · LinkedIn lexsi-labs
- Papers linked from the site: arXiv 2602.09621, arXiv 2606.21631
- Careers: https://app.screenloop.com/careers/aryaxai/

## 11. Open questions for Discord

1. Does "at least 2 repos from the track" allow cross-track repos (for example AlignTune in a Track 2 project)? What does "except Track 3" mean?
2. Can several Aya models be used (for example Aya Expanse 8B as teacher or judge, Tiny Aya as student)? Can non-Aya models be used as judges or attackers (for example Llama Guard, a closed API for ROI comparison)?
3. Will the final submission checklist add anything beyond the 9 items?
4. Do PRs or bug reports to Lexsi repos count in judging?
5. When will the kickoff recording be posted?
