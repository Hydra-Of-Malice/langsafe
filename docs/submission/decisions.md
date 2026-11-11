# What we still have to decide before submitting

For Aditya and Pallab. As of Wed 11 Nov, 03:40 IST, at commit `ee0a957`. Deadline: Sat 14 Nov, anywhere on Earth
(Sun 15 Nov, 17:30 IST).

The experiments are finished. Full precision, Q4_K_M and NF4 are in one report with no validation gaps, the public
results export has passed its safety scan, and a clean public copy of the repo is built. Everything left is a decision
for the two of us. Each item has a recommendation and the date it has to be settled by. `checklist.md` keeps the full
background.

## Suggested order

| Wed 11 (today) | Thu 12 (publish) | Fri 13 (finish) | Sat 14 (buffer) |
|---|---|---|---|
| Decide D1 to D4. Ask native readers (D6). Organiser ticket (D16). Send the one-pager (D15) | Push the public repo. Upload data and models. Fill the URLs (D5). Open the kit PRs (D11) | Fold in reader labels. Final read of the article. Submit the form (D17) | Fixes only. Delete the private repo (D12) |

## Needed for a valid submission

The rules require public code and public artifacts on Hugging Face. Nothing is public yet.

**D1. How do we publish the code?** Recommend: publish the clean public copy as a new public GitHub repo, and keep the
working repo private.
- **Why not the working repo:** its history carries a personal Gmail address on most commits, our notes on other
  teams, the Kavach notes and the organisers' raw text.
- **The public copy:** one commit under the GitHub noreply address, with those notes left out. Its 855 tests pass twice
  on its own source.
- **What it takes:** create an empty repo, add it as the copy's remote, push. Aditya creates the repo; 10 minutes; by
  Thursday.

**D2. What may stay in the public copy?** Recommend: keep everything as built: `space/`, `docs/upstream_patches/`, the
ledger and the Hugging Face username. The builder flags two details:
- **Ledger rows** carry two generic machine names, `HydraOfMalice` and `Admins-MacBook-Pro.local`. The hash chain is
  part of the provenance story, and removing them would break it.
- **The remote-run scripts,** their doc and tests name the private bundle repo `HydraOfMalice/lexsi-nf4-private`. The
  username goes public with the Hugging Face artifacts anyway, and the repo itself stays private, then gets deleted
  (D12).

Both of us; by Thursday.

**D3. Do we publish the results and data on Hugging Face?** Recommend: yes. Publish the export in
`runs/experiments/public/earth-full/` as one public dataset repo, with the licences from the dataset card.
- **What it holds:** 152 files (94 MB): prompts, verdict labels, reports and provenance, with no model text. The final
  scan compared every exported field against 191,984 private answers and found no match.
- **One change:** prompts' `category` labels are left out, because some share wording with model answers. The card
  says how to recover them.

I upload; 20 minutes; by Thursday.

**D4. Which model weights do we publish?** Recommend: all five repair arms at full precision, the selected arm's Q4_K_M
GGUF and the steering vector. Keep the random-layer controls private.
- **Proposed rule:** publish an arm only if no language is significantly less safe than Earth, by paired 95% interval,
  in either preamble condition.
- **Check on the final report:** all five arms pass. The lowest point estimate is Circuit merge in Korean, -4.5 points,
  with an interval that spans zero. One control fails: random layers, seed 2, in Swahili without the preamble, -4.0
  points (-7.4 to -1.1).
- **Each repo gets:** CC-BY-NC 4.0, the Cohere policy, a "modified" notice and the LSAL notice. Names avoid "Lexsi",
  "SafeTune" and "CircuitKIT", and nothing is gated.
- **Upload:** about 25 GB, roughly an hour and a half. I upload; by Thursday.

**D5. Fill the article's links and usernames.** Recommend: I fill all 22 once D1, D3 and D4 exist. I need both
Hugging Face usernames. By Thursday.

## Choices that shape what the article claims

**D6. Can we get native readers before Friday?** Recommend: ask for Javanese and Thai readers today, for the
hand-check samples only. If nobody can do it by Friday noon, submit with the numbers labelled provisional. The article
is already written that way.
- **What waits on them:** three placeholders (judge accuracy, items labelled, queue size). The report counts every
  language as provisional until they exist.
- **What's already covered:** the router check (kappa 0.79 to 1.00; content labels the same in 94% to 100%).
- **If more time appears:** the full translation packet (2,994 rows) is ready.

**D7. How much circuit detail do we publish?** Recommend: publish the layer lists, summary metrics and editing mask.
Keep the node-level score files private and offer them to researchers on request. Draft for
`CIRCUIT_RELEASE_DECISION_SENTENCE`:

> We publish the circuits' layers, summary metrics and the editing mask, which are enough to reproduce every repair. We
> keep the node-level scores private and share them with researchers on request.

**D8. Do we mention Kavach as prior work?** Recommend: yes, in one line, if the hash-chained run log came from it. No
Kavach code is in this repo. Draft for `PRIOR_WORK_STATEMENT`:

> The hash-chained run log follows a design from our earlier project Kavach (HackIndia AgentVerse '26, MIT licence).
> No code from it is reused.

If nothing came from it, I delete the line.

**D9. What do we do with the remaining Hugging Face credit?** Recommend: spend about $7 of the roughly $10 left.
- **Spent so far:** about $9.65 (NF4 about $8.10, router sample about $1.00, fair baseline about $0.55).
- **Command A on the 1,680-answer sample:** about $6, if the router serves it. That turns the "report's judge" row of
  section 8 into a result.
- **The 32B sample on the three 4-bit models:** about $1.

Each spend gets an OK first.

**D10. Do we deploy the demo Space?** Recommend: skip it unless D1 to D6 are done by Thursday evening. A repo link
meets the rule. `space/` still uses the Fire language pairs and a placeholder report, and needs two to three hours of
work and a prompt review.

**D11. Do we open the six PRs to the Lexsi kits?** Recommend: yes. Open them Thursday from Aditya's GitHub account,
with descriptions I draft, and link them in the article.
- **The patches** (`docs/upstream_patches/`): AuditKIT benchmark loaders and multilingual refusal scorer; CircuitKIT
  multilingual refusal task; CuratorKIT group split; SafeTune layer ranking and multilingual benchmarks.
- **Otherwise,** the article marks them "planned".

## Housekeeping

**D12. When do we delete the private bundle repo?** Recommend: delete `HydraOfMalice/lexsi-nf4-private` now. Nothing
needs it any more.
- **Why delete it:** it holds the harmful prompts, the models' answers and the repair checkpoint.
- **Why it's safe now:** the Kaggle T4 cross-check finished at 11:18 and matched the A10G run, and both runs' results
  are on the laptop.
- **Deleting is permanent,** so it waits for Aditya's word.

**D13. The ROI page's open inputs.** Recommend: give me the cost assumptions (product name, licence and running costs
per month, target GPU use). Decide whether the token-count row is worth one small download (the public `o200k_base`
tokenizer file, a few MB); without it the row says "not measured". The measured speed is in: 159 output tokens a
second on the laptop RTX 4060.

**D14. Confirm the spend** on the Hugging Face billing page. The article says "about $10".

**D15. What goes to the mentors?** Recommend: send `docs/brief/what_we_are_building.pdf` today. The longer brief
predates the final results and should not go out as it stands.

**D16. Open a ticket with the organisers.** Recommend: open it today, leading with question 1 of `checklist.md` (may we
publish a safety evaluation set that includes harmful prompts?), and publish in parallel unless they say no. Our
harmful prompts are MultiJail's and Aya's own public items. Our translations are of benign prompts and circuit pairs
only.

**D17. Who submits the form, and when?** Recommend: Aditya, on Friday 13 Nov in the evening (IST), leaving Saturday as
a buffer.

## Already settled this week

- **Full run:** the Earth run is done at full precision, Q4_K_M and NF4, and the final report has no validation gaps.
- **Repairs:** none met the pre-set conditions. The Thai gap was not confirmed, so all the repairs are exploratory.
- **Judges:** the 32B router judges agree with the local judges (kappa 0.79 to 1.00), and the calibration files are
  written.
- **Fair circuit baseline:** circuits 0.90 to 1.01, against 0.21 to 0.69 for random heads.
- **Public export:** the sanitised export of the full run passed its scan.
- **Article:** 554 values are filled. Only the links, the reader labels and D7 and D8 remain.
- **Public copy:** rebuilt, and tests pass twice in both repos.
- **Mentor summary:** the one-pager is written.
- **NF4 cross-check:** repeated on a Kaggle T4. Every rate is within 1.7 points of the A10G's, and every
  pass-or-fail decision is the same.
- **Release plan:** ready in `scripts/publish_hf.py`: 7 repos and 29.1 GB, a dry run until you approve.
