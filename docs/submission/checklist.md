# Submission checklist

Updated on 6 November 2026 for the move to Tiny Aya Earth, against commit `63da69e`. Items the move did not touch were last checked on 5 November at commit `f09a5a1`. Sources: hackathon site and kickoff slides (`docs/research/01_official_brief.md`), the organisers' answers on Discord, and a read of the code and repo state.

**Deadline: 14 November, midnight AoE. That is 15 November, 17:30 IST.** Late or incomplete submissions may be excluded. An internal team checks validity before any judge reads the work: public artifacts, write-up, checklist, no duplicates.

Status means one of three things:

- **done**: nothing is left, or nothing is needed.
- **waiting on results**: the structure exists, and it needs numbers from a run.
- **needs a person**: only the team can do it, or the team must decide.

## Do these first

| # | Action | Why now | Status |
|---|---|---|---|
| 1 | Confirm the team is registered, with both HF usernames matching the form | **Registration closes today, 5 November.** We have no record here that this was checked | done: Aditya confirmed on 5 November that both are registered with matching HF usernames |
| 2 | Decide the rule for choosing ALT-A or ALT-B, and log it in `SPEC.md` | It had to be fixed before held-out results exist, or the choice could be accused of following the data | done: `SPEC.md` Amendments 13 and 17 fix it, and `report.json` applies it under `gap_confirmation` |
| 3 | Calibrate the primary axis | Content safety is primary in `configs/spec.yaml` and `SPEC.md` Amendment 10, so that choice is settled. Agreement and hand sheets now cover content labels, but nobody has labelled a row, and the router comparison covers refusal labels only. See "Risks", item 2 | needs a person |
| 4 | Decide what happens to the Earth grid if there is no gap | It had to be settled before the grid runs. See "Risks", item 3, for what the selection score still does | done: `SPEC.md` Amendment 13 says the write-up reports that the gap was not confirmed, and the repairs are exploratory |
| 5 | Break the freeze deadlock | The spec was to freeze after mentors comment. A later freeze would restart a four-day run, so it froze without their comments (`SPEC.md` Amendment 26). The brief is still worth sending: a mentor's change would be a dated amendment | done: frozen on 7 November. Aditya was told first and said to continue |
| 6 | Find native readers for Javanese first, then Bengali, Arabic, Swahili, Thai and Korean | The main result depends on Javanese. The review packet counts every row with origin `translated`, MultiJail's own included. It was exported on 9 November: 2,994 rows in six languages, 538 of them Javanese, every one with its English source (`docs/REVIEW_WORKFLOW.md`). The hand-check sample needs people too. Our report blocks any success claim without them | needs a person |
| 7 | Redeem HF credits and check the balance | The 32B judges need them, and the local judges cannot be used for headlines without a shared-sample check against the 32B pair | needs a person |
| 8 | Give the go-ahead (or not) to make the repo public and publish to HF | Required by the rules. Sections below list what must happen first | needs a person |

## Rules

| # | Rule | Status | Where we stand | What is left |
|---|---|---|---|---|
| 1 | Teams of 1 to 3, one team per person, at least one member on Discord | needs a person | Team of two. Discord has been read by a team member | Confirm the registration entry (action 1) |
| 2 | Built during the hackathon, 1 to 14 November | needs a person | First commit is 3 November | Confirm that no Kavach code was reused. If any was, disclose it in `{{PRIOR_WORK_STATEMENT}}` |
| 3 | At least 2 repos chained as one pipeline, 3+ preferred | waiting on results | Four repos in code: CuratorKIT, AuditKIT, CircuitKIT, SafeTune. The development run exercised the chain | Confirm the final manifest's `curatorkit` field is filled. Pilots skip it, so both local manifests show `null`. AuditKIT records do not carry the content axis yet |
| 4 | Core model is one of the 7 listed | done | Tiny Aya Earth is the core. Global, Base, Fire and Water are also listed. Qwen3 is judge only, which the organisers allow | Nothing |
| 5 | Models, adapters, logs, datasets and benchmarks public on HF | needs a person | Nothing is published | Team go-ahead, then see "Before anything goes on Hugging Face" |
| 6 | Public deployment by the deadline: a public demo or repo link | needs a person | **The GitHub repo is private as of 5 November.** A draft Gradio demo is in `space/` | **The repo must be public by the deadline. Publishing needs the team's go-ahead.** See "Before the repo goes public" |
| 7 | Before-and-after metrics, more than one | waiting on results | Three axes plus a secondary metric are designed and tested on the development run | Full run |
| 8 | HF article: metrics and why, reasoning for data, tasks and model, how the result is better in use | waiting on results | Draft in `article.md` | Fill placeholders, choose the story, publish |
| 9 | ROI comparison (optional, preferred) | waiting on results | `roi.md` has the formula, prices and arithmetic | Measure batched tokens per second. Re-check prices on the day |
| 10 | Originality; licences respected | needs a person | Dataset and model licences are checked against HF metadata. Kit licences read in full | Kavach question (rule 2). Questions for organisers below |
| 11 | Code of conduct | done | No action needed | Nothing |
| 12 | Teams keep IP; organisers may showcase | done | No action needed | Note that LSAL section 6 gives Lexsi broad rights over any PR we open |
| 13 | Organisers may change the rules | needs a person | The 9-item list was marked "complete details will be shared soon" | Watch email and Discord. The form arrives a couple of days before the deadline and may ask for more |

## The 9 article items

| # | Item | Status | Where we stand | What is left |
|---|---|---|---|---|
| 1 | Problem statement and approach | waiting on results | Written, with ALT passages | Choose the story |
| 2 | Hackathon repos used, for what, which features | waiting on results | Written from the code. Claims match what the pipeline calls | Six patches are drafted in `docs/upstream_patches/`. None is opened. Their author line is a placeholder. Fill PR links, or mark them "planned" |
| 3 | Models used and licences | waiting on results | Written with Earth as the core model. Full Earth weights are in (`docs/PROGRESS.md`, 6 November) | Name the judge pair used. State no Water result unless the control finishes |
| 4 | Datasets and benchmarks, with licences | waiting on results | Written and checked | Final counts and translation review status |
| 5 | Model artifacts public as an HF collection | waiting on results | Card template in `model_card_template.md` | Selected checkpoint, then go-ahead, then publish |
| 6 | Data artifacts public as an HF collection | waiting on results | Card template in `dataset_card_template.md` | Final data, a sanitised log export, go-ahead, publish |
| 7 | Experiments, findings, what else was tested | waiting on results | Development run and planning checks written. Result tables ready | Fill section 7 |
| 8 | "With more credits": approach and before-and-after results | waiting on results | Plan written | Fill "before" values and the cost estimate |
| 9 | Teammates tagged | needs a person | Names are in. HF usernames are unknown to this draft | Add both usernames. GitHub handles may differ, so do not assume |

## Judged surfaces

| Item | Status | Where we stand | What is left |
|---|---|---|---|
| Write-up clarity (/10) | waiting on results | Follows the 9 items in order and stands alone | Read it cold once it is filled in. Delete every team note |
| Practical usefulness (/10) | waiting on results | A named kind of user, an audit recipe, ROI | Make sure section 7.9 matches the final story |
| Code and notebook quality (/10) | waiting on results | `notebooks/01_reproduce_results.ipynb` is a guided reproduction, but it has not been run on a GPU end to end | Run it once, and pin versions and seeds |
| Model cards, which judges see in their own UI | waiting on results | Template is complete | One card per published checkpoint |
| Track tag | needs a person | We target 2b, problem 2 | Under ALT-B, organisers may tag it 2a, problem 4. Say so in the article, or ask |
| Demo Day readiness (if shortlisted) | needs a person | Live demo, ideally two presenters. A draft Space exists in `space/` | Plan only if time allows |

## Our own science gates

The report refuses to claim success until these pass. They are separate from the rules, and a submission can be valid without them, provided we say so plainly.

| Gate | Status | What is left |
|---|---|---|
| 10-language sweep on the content rubric, with Earth, both judges, both preamble conditions | done | Finished on 5 November (`runs/pilot/multijail.json`). It is a planning signal, not a result: 4-bit weights, repair-split items, local 8B judges |
| Held-out confirmation of any gap | waiting on results | `SPEC.md` Amendments 13 and 17: full precision, held-out items, every language in both preamble conditions, 14 tests. `report.json` computes it under `gap_confirmation`. `earth-baseline` in `scripts/run_chain.py` only generates, so the baselines need judging by hand to see it before the repairs (`docs/RUNBOOK.md`). The full Earth run has not started |
| Spec frozen | done | 7 November, without mentor comments. See action 5 |
| Two judges on every cell | waiting on results | Local pair available. 32B pair needs credits |
| Judge agreement per language: AC1 at or above 0.6 on content labels, kappa at or above 0.6 on refusal labels | waiting on results | `SPEC.md` Amendment 16 moved the content gate to AC1. On the pilot's rows, a planning signal, Earth's Javanese AC1 is 0.77 with the preamble and 0.75 without, the lowest cells but above the floor |
| Content rubric calibrated | needs a person | The code now computes content agreement, and hand sheets take `hand_safety_label`. No hand labels exist yet, and the router comparison covers refusal labels only |
| Blind audit of held-out answers in the target language | waiting on results | Required by `SPEC.md` Amendment 16, and reported next to the result. `scripts/blind_audit.py` is ready. It needs held-out answers first |
| Hand labels: a 200-row audit per model, plus at most 10 flagged rows per language and model | needs a person | Zero labels so far |
| Translation review: about 3,000 rows for Earth's languages | needs a person | Zero labels so far. Status `unreviewed` |
| Full Earth grid: five arms, random-layer control | waiting on results | Not started. Steps are in `scripts/run_chain.py`, from `earth-baseline` to `earth-full` |
| Circuit runs on Global and Earth in en, jv, bn, without the preamble | waiting on results | CPU on the laptop, since it does not fit in 8 GB |
| Q4_K_M and NF4 re-tests | waiting on results | Both are required by `quantized_must_hold` |
| `scripts/localisation_analysis.py` on the full run | waiting on results | Script is in the repo, and it reads content-safe labels as well as refusal |
| Fire and Water on Earth's held-out items | waiting on results | `experiment.comparison_models` generates both inside `earth-baseline` (Amendment 21). Water's 4-bit pilot is done; its held-out result isn't, so state only the pilot until then |

## Before the repo goes public

The repo is private today, and all history goes public with it. No generation, judged-output or hand-check path has ever been committed, which is good. These items need a decision first.

- [x] Two private research notes and the organisers' raw material are left out of the public repo, which starts with fresh history.
- [x] `SPEC.md`'s Thesis, Models, Languages, Metrics, Judging, Repair arms and Success thresholds sections still describe the Fire plan and refusal. Only its Target section and amendments carry the move. 9 November: the body stays as written, because it is the pre-registration. A table under "Reading order" now points each section to the amendments that change it, and says what is different. It changes no rule and gives no result. `README.md` and the mentor brief no longer state the gap as fact.
- [x] `README.md` listed "compare" under AuditKIT, and nothing calls `compare_runs`. The claim is dropped in the working tree, for review.
- [x] `README.md` said "all four repair arms" and quoted test counts from the development run. It now says five arms and a control, in the working tree, for review.
- [ ] `runs/ledger.jsonl` is tracked, and ledger entries store the machine's hostname. Checked 9 November: the committed rows hold the two machines' hostnames and a Hugging Face cache path under a generic Mac user name. No token or file of ours beyond that. Low risk, but it's the team's call.
- [x] `docs/CUDA_EVALUATION.md` says some source-tree details should stay private. Checked 9 November: that applies to the working-tree diff made at transfer time, which was never committed. `docs/CUDA_HANDOFF.patch` holds code only: no paths, hostnames, tokens or outputs. It is an old copy of changes already in the tree, which have been edited since, so it no longer applies. Dropping it from the public version loses nothing.
- [ ] `space/` and `docs/upstream_patches/` are in the repo. Decide whether they stay in the public version. `space/sample_report.json` is a fake fixture.
- [x] Run a secret scan over the tree and history. Done 9 November over all 77 commits (461 files) and the working tree, for Hugging Face, OpenAI, Anthropic, OpenRouter, GitHub, AWS, Google, Slack and Discord tokens, private keys, and secret assignments. Nothing real: the one hit is a placeholder string in `space/test_app.py`. Run it again just before the repo goes public.
- [x] The working repo's history carried the authors' email addresses. The public repo starts with fresh history under a GitHub noreply address.
- [x] `.gitignore` still blocks every output path. Checked 9 November: generations, judged files, hand-check sheets, calibration, AuditKIT files, pilots, every run folder (blind reads and review packets included), caches, checkpoints, GGUF files and `.env` are all ignored. Nothing under `runs/` except `runs/ledger.jsonl` has ever been committed. `exports/` is not ignored, because the sanitised export is meant to be published. Check again before the final push.

## Before anything goes on Hugging Face

- [ ] **Never upload `runs/auditkit/`.** AuditKIT's JSON saves full predictions, including `prompt` and `raw_output`. These are harmful completions.
- [ ] **Run the sanitised log export on the full run.** The rules ask for public logs. `scripts/export_public.py` keeps hashes, settings, scores and aggregates, and drops all model text. It passed on the Earth development run on 7 Nov. It needs `--allow-prompt-echo`, because a broken answer can repeat a public prompt exactly. Check that count in `EXPORT_MANIFEST.json`.
- [ ] Keep `runs/gens/`, `runs/judged/`, `runs/hand_check/`, `runs/calibration/` and all review CSVs private.
- [ ] Decide which losing arms to publish. The proposal said every arm. It also said no weights less safe than the originals. Those conflict if an arm lowers refusal in any language. Set one rule, and use it in all cards.
- [ ] Publish a checkpoint only after held-out re-evaluation. LSAL section 4 bars derivatives that were not repaired and re-evaluated from user-facing systems.
- [ ] **Demo.** The draft in `space/` meets our release rule by design: no text box, a fixed list of benign prompts, and no API. Before it goes public:
  - Replace the FAKE DATA fixture with a real `report.json`, and confirm the fixture cannot load on the public Space.
  - Add the content-safe rate. The Results explorer shows refusal and compliance only.
  - Review `report.json` first. Its README warns that it holds local file paths and per-item vectors.
  - Decide what "repaired" means on the page if ALT-B is the story. Side-by-side needs a real re-evaluated checkpoint.
  - Check it against LSAL section 4, since it loads original Fire in a user-facing page.
  - Move it to Earth's languages. It uses the Fire pair for English, Hindi, Bengali and Tamil, and the Earth pair only for Arabic.
  - Get its prompts reviewed. A repo link alone also meets the rule if time runs out.
- [ ] Keep "SafeTune", "CircuitKIT" and "Lexsi" out of repo names. LSAL section 2 limits those names to unmodified copies.
- [ ] Put CC-BY-NC 4.0, the Cohere policy, attribution and a "modified" notice on every model repo.
- [ ] Consider including the LSAL notices with released checkpoints. Section 1 asks for them on copies and derivatives of the Licensed Work. Whether our checkpoints count is unclear, and including them costs nothing.
- [ ] Do not gate the repos unless organisers agree. Validity needs public artifacts.
- [ ] Decide how much circuit detail to publish. The proposal flagged this as a dual-use question.
- [ ] Add steering vectors to the collection. They are optional but preferred.

## Risks of failing a rule, most serious first

1. **Repo or artifacts not public by the deadline.** This fails validity before any judge reads it. It depends on a team decision, plus the clean-up above.
2. **The primary axis is not validated.** Selection, the acceptance gates, the agreement gate and the hand sheets now use content labels. But no hand label exists, the router comparison covers refusal labels only, and AC1 shows agreement, not correctness. The blind audit found that Aya misses unsafe content and Qwen over-flags. Both judges must say `safe`, so disagreement also pushes the rate down, and in the pilot, a planning signal, Earth's Javanese had the lowest agreement of any cell. Local judges stand in for Command A, and the pipeline has no check of them on this rubric. Fix this before any headline number.
3. **The grid cannot fully answer the question if there is no gap.** Amendments 13 and 17 settle the reporting: without a confirmed gap the repairs are exploratory, and `empirical_repair_success` is false. The selection score has not changed. It uses the repair-split gap, so with no positive gap there it gives no reward for safety gains and favours the least damaging knob. With a positive gap there that fails on held-out items, knobs are tuned to a gap we cannot confirm.
4. **Earth's languages are now in every tool.** Checked 9 November: `CARD_LANGUAGES` in `scripts/export_public.py` includes jv, th and ko, `src/langsafe/judge/rules.py` has Javanese, Thai and Korean refusal phrases, and `docs/REVIEW_WORKFLOW.md` assigns reviewers for Earth's languages, Javanese first. The full-run translation packet is exported: 2,994 rows, all with their English source.
5. **Picking the largest gap favours noise.** Choosing the target as the largest repair-split gap selects for chance, and the pilot chose the added languages the same way. A gap may shrink or vanish on held-out data. Report that openly.
6. **Harmful text leaking through logs.** See the AuditKIT item above. This would break our release rule, and possibly the Cohere policy.
7. **The reproduction notebook hasn't been run on a GPU.** Code and notebook quality is a third of the sub-scores.
8. **Time.** Eight days remain from 6 November. The full Earth run looks like about 3 to 4 days (`docs/PROGRESS.md`) and has not started, and human review has not started. If it cannot finish, the honest fallback is the measurement story with a smaller grid, and the article can carry that.
9. **Machine-translated data from a model under test.** The proposal said "a stronger model". The config uses Tiny Aya Global. The article and card say so. In Earth's set this affects the benign prompts and circuit pairs only, and the translator choice remains a real weakness.
10. **Thresholds tighter than the data can resolve.** One point of 400 questions is 4 questions. Report intervals and do not call a point estimate a pass without them.

## Questions for organisers

Discord advises using a ticket. We have no record that one was opened.

1. Is a public safety evaluation set (prompts and labels, including machine-translated harmful prompts made with Tiny Aya) acceptable under the Cohere Labs acceptable use policy?
2. Is Tiny Aya Global acceptable as the reference for drift in Earth?
3. If the gap is not confirmed, does a measurement-led submission stay in Track 2b, or should it move to 2a?
4. May Command A be used as a judge to replicate the report's method?
5. Does the checklist add anything beyond the 9 items?
6. Does publishing the checkpoints require including the LSAL notice?
7. What is the right way to meet the "public logs" rule when raw logs contain harmful completions?
