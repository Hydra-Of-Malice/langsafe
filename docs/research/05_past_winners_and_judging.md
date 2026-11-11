# Past Winners and What the Judges Value

Researched 2026-11-02. **[V]** = verified with a URL; **[I]** = inferred.

## Bottom line

**Lexsi Labs / AryaXAI has never run a public, judged hackathon with published winners.** This is their first one. [V, by exhaustive search]

The only earlier event was a one-day **"SSI Club" hackathon** that AryaXAI and FutureGPT co-organised on 28 Sept 2024. Its theme was a "production-ready ML setup for enterprises" (observability, explainability, risk). No results were ever published. Source: https://www.indiablooms.com/finance/aryaai-founder-to-create-global-open-source-community-to-boost-safe-super-intelligence-ecosystem/details. Vinay Kumar described the goal as an "open-source research community that can crowdsource ideas."

There are no Lexsi hackathons on Kaggle, Unstop or Devfolio. Their Luma calendar has no past events. The Arya College "Hack Arya Verse" events are unrelated.

So "what won before" has to be pieced together from three sources:
1. The **published rubric** (kickoff slides).
2. **How Lexsi writes up its own work.** The judges include the organisers' research team.
3. **Proxy events judged by the co-hosts**: Cohere's Expedition Aya and Expedition Tiny Aya, and HF's Build Small.

## 1. The rubric (kickoff slides) [V]

- **Validity labelling comes first.** The internal team checks for public artifacts, the write-up, the checklist and duplicates, then tags the track. **An incomplete submission is out before any judge sees it.**
- **Judging is double-blind** by founders, academics, industry labs and organiser research teams. Each team gets several scores, which are normalised and averaged.
- **Scores:** Write-up clarity /10 · Practical usefulness /10 · Code/notebook quality /10 · Overall /100.
- Wording in the problem statements shows what judges will check:
  - "report robustness **alongside over-refusal**"
  - "prove it is gone under **paraphrased and adversarial** prompts, and show general capability survives"
  - "**verify nothing else broke**"
  - "show that curation **measurably beats** raw generated data"
  - "**without lowering** the success rate"
  - "**reproducible transcripts**"

## 2. How Lexsi evaluates its own work [V]

### Gemma-4-31B repair article (Oct 2026)
Source: https://lexsi.ai/resources/articles/the-lexsi-alignment-and-safety-stack-correcting-gemma-4-31b-for-multilingual-and-cybersecurity-safety-in-the-weights-copy

- **Three axes every time:** harmful refusal, benign over-refusal, utility. Costs are shown next to gains, never netted against them.
- **They publish their own regressions.**
  - Legitimate-security refusal went from 21.2% to 49.3%.
  - SWE-Bench Lite went from 109 to 102.
  - One attack showed no improvement (five-turn phishing, 95.0% to 96.1%).
- **They compare against a baseline.** Activation steering matched the refusal gain but refused 68% of benign prompts. Their weight edit refused 2.8%.
- **Method hygiene:**
  - Spec written before the edit.
  - Held-out data kept disjoint from the correction data.
  - Identical prompts, decoding and judges for base and repaired models.
  - A second independent judge.
  - Every held-out row published.
- Their mantras:
  - "A model that refuses more can simply be a model that refuses everything."
  - "Benchmarks locate the cost; they do not set the tolerance."
  - "Rejection is a valid result."

### "Specialization Dividend" wealth-management article (Feb 2026)
- Compares base 0-shot → SFT → SFT+DPO on four metrics, against GPT-4o and GPT-5 2-shot.
- Includes qualitative examples and Colab notebooks.
- Results are scoped "for this benchmark and evaluation setup only."
- **It has no ROI numbers. Filling that gap is an easy way to stand out.**

### "Debugging TabFM" article (Jul 2026)
- "that number is the start of due diligence, not the end." Read past the leaderboard: look at stability, fairness after fine-tuning, shift and serving cost.

### Their HF releases use matched controls
- `*-sft-drift` vs `*-dpo-recover`.
- `obl-ex-4b-raw` vs `-curated`, plus logs.
- **[I] Judges will reward the same design: a controlled ablation with every arm public on HF.**

## 3. Proxy events judged by the co-hosts [V]

### Expedition Aya 2024 (Cohere)
Source: https://sites.google.com/cohere.com/expedition-aya/project-showcase

- Winners:
  - Gold: Multilingual Climate Chatbot
  - Silver: Evaluating Reward Models in Multilingual Settings
  - Bronze: Sanskriti Aya (a quality index)
  - Most Innovative: The Language Effect
  - Most Promising: Maya (Multimodal Aya)
- **[I] Two of the five winners were evaluation or benchmark work.** Measurement projects win with Cohere.

### Expedition Tiny Aya (Cohere, Jul 2026)
Source: https://cohere.com/blog/tiny-aya-in-the-wild

This used the **same model family as our hackathon, so it is the closest proxy.** Projects highlighted:
- **Mix, Fine-Tune, Break.** Code-mixed safety on Tiny Aya Global. A binary benchmark rated it "96% safe", but its refusal rate was only 66%. They added a "confused" judge category. https://labscommunity.cohere.com/blog/2026/mix-fine-tune-break/
- **Cross-Lingual Emergent Misalignment.** About 50 Tiny Aya fine-tunes on HF, under `crosslingual-em/*`.
- **Cultural Context Robustness.**
- **Inside Tiny Aya.** An SAE analysis.
- **Language-Aware Quantization.** Compression hurts some languages more than others.
- **Tiny Facade.** On-device multilingual tool calling at 4-bit. https://huggingface.co/blog/Bronsn/tiny-facade
- **Kids Companion.** Released a benchmark dataset on HF.

Cohere praised:
- Evaluations that reveal **"blind spots … missed by traditional multilingual benchmarks"**.
- Open datasets.
- Local deployment.
- Underserved languages.

⚠ **Code-mixed safety, emergent misalignment and language-aware quantisation on Tiny Aya are already done.** We must go one step further, for example **repair plus verification with the Lexsi stack**, rather than repeat the audit.

### HF Build Small hackathon (Jun–Jul 2026; Cohere sponsored)
Source: https://huggingface.co/spaces/build-small-hackathon/prize-ledger

- Judges rewarded a "specific, real problem", builders who used their own tool, an honest fit to small models, and polish. A fine-tuned published model and a write-up earned bonus points.
- Winners solved concrete offline or private problems, such as an offline clinic-protocol assistant and a private blood-test explainer.

## 4. Comparable safety hackathons [V]

- **Apart Research sprints** score 1–5 on three things: impact and innovation, execution quality, and presentation and clarity.
  - Submissions need a **"Limitations and Dual-Use / Ethical Considerations" appendix**.
  - Prior work must be declared.
- **AI Safety India** is an outreach partner but has run no judged hackathons.

## 5. What gets emphasised most (synthesis)

Ranked by how strongly the evidence points to each:

1. **Rigorous before/after evidence on several axes.** The target metric, an over-refusal or false-positive metric, and capability retention, all with identical evaluation settings. This shows up in the rules, the problem statements and every Lexsi article.
2. **A real, connected pipeline across their repos, with public artifacts.** The hackathon exists to get their stack adopted and to collect feedback. Use 3 or more repos with provenance chained, and publish HF collections.
3. **Honesty.** Show regressions, failed variants and "what else was tested". It is a checklist item and a Lexsi value.
4. **Practical usefulness.** A named user or workflow, a regulated domain, Indic or underserved languages, and an ROI table against a closed API.
5. **Reproducibility and code quality.** One-click notebooks, pinned versions, seeds, and AuditKIT run fingerprints.
6. **Multilingual and Indic focus** pleases both co-hosts. Cohere's mission is multilingual, Lexsi's Gemma-4 work centred on Indic safety, and the kickoff host has a multilingual background.

## Winning patterns to copy
1. Write the HF article with the **9 checklist headings in order**.
2. Use **three-axis scoring** with identical evaluation settings before and after.
3. Publish **regressions and failed variants**.
4. Run a **controlled comparison**, with every arm on HF and logs included.
5. Name a **specific user**, and add the **ROI table**.
6. Ship a **one-click reproducible notebook** chaining 3+ repos.
7. Use a **second independent judge** and hand spot-checks.
8. Put **one falsifiable thesis** in the first paragraph. Demo Day is literally "prove your thesis".
9. Add a **limitations and dual-use note**, especially for Track 2a.
10. **Upstream fixes and features** to the Lexsi repos and cite the PRs. [I] This directly serves their "gather feedback" goal.

## Pitfalls
- A refusal gain with no over-refusal number.
- A single headline metric.
- Overlap between training and evaluation data, or different decoding settings before and after.
- Repos bolted together rather than chained.
- Using a model outside the 7 allowed.
- Gated or private artifacts.
- Missing licences.
- Untagged teammates.
- Overclaiming.
- App polish without measurement.
- Assuming the final submission form matches the slides. **Re-check the email and Discord before Nov 14.**
- **Repeating an Expedition Tiny Aya project without going further.**
