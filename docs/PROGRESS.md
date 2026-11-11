# Progress log

Newest entry first. Updated with every push, so anyone on the team can see what is running and who owns what. Pull
`main` before starting a run.

## Who is doing what

| Person / machine | Now | Next |
|---|---|---|
| Aditya's laptop (RTX 4060, 1.2 TB free) | Full Earth run finished (FP16, Q4, NF4); final report built. Kaggle T4 NF4 cross-check running | The report with Q4 and NF4, the last sentences and tests, about Wed 05:00 to 07:00 |
| Pallab's M1 Max | Free | Please pull `main`, and don't start another full run: this one owns the frozen spec |
| Aditya | Send the brief to mentors (hold it until the note below is read) | Native readers (HF credits now pay for the remote NF4 run and the router check) |

## Still needs a person
- Native readers for Javanese first, then Bengali, Arabic, Swahili, Thai and Korean, to review translations and hand-label
  a sample. Javanese matters most, because the main result depends on it.
- HF credits, for the 32B judges that verify the free local judges.
- A go-ahead before anything is published to Hugging Face, and the final submission form.
- The rules ask for a public demo or repo by 14 Nov, so this repo must go public before the deadline. That's the
  team's call. Raw model outputs stay out of it either way.

## 11 November 2026
- **11:26** **The Kaggle T4 repeated the NF4 run and got the same results: the hardware does not change the NF4
  findings.**
  - **The run:** two free Kaggle sessions. The first stopped itself at the 11.5-hour cap after 8 of 17 stages. The
    second resumed from the synced output at 05:44 and finished all 17 stages at 11:18.
  - **Against the A10G run:**
    - Answers: 83% to 84% are word for word identical.
    - Content labels: the same for 98.1% to 98.7% of harmful-prompt answers.
    - Content-safe rates: all 42 (3 models, 7 languages, 2 conditions) are within 1.7 points; the median gap is 0.6.
      Earth's Javanese NF4 rate is 70.5% on both.
    - Gates: every pass-or-fail decision is the same, and the repair fails on both.
  - **Written:** `FULL_RUN_HARDWARE_SENTENCE` and `COMPUTE_SUMMARY` now say this.
  - **D12 of the decision list is now unblocked.** The private bundle repo is no longer needed and can be deleted on
    Aditya's word.
  - **Also new:** `scripts/publish_hf.py`, the release plan for D3 and D4.
    - **Dry run by default.** It lists 7 public repos and 29.1 GB: the dataset, four weight arms, the selected arm's
      GGUF and the steering direction. The controls stay private.
    - **Publishing** needs the reviewed plan's digest, and it refuses while a card is missing. The six kit PRs already
      have full descriptions in `docs/upstream_patches/README.md`.
- **03:13** **The full run is finished, and the final report covers full precision, Q4 and NF4 with no validation
  gaps. No repair held at 4-bit, and 4-bit itself breaks Earth's Javanese.**
  - **The chain:** it ended at 02:59. Q4 was evaluated for all three models. The NF4 step then stopped at the file
    that switched it off, as planned.
  - **NF4 brought in:** `scripts/import_nf4.py` checked that the HF run was complete, used this spec and was given this
    run's evidence (all 39 source files matched). It then copied 24 NF4 files with their SHA-256 into
    `nf4-import.json` and the ledger, and rebuilt `report.json` over all 16 tags. The chain's own report is kept as
    `report.chain.json`. The ledger verifies.
  - **Content-safe rate with the preamble, FP16 / Q4_K_M / NF4.** Local judges, so provisional:

    | Model | Thai (target) | Javanese | Mean of 7 |
    |---|---|---|---|
    | Global | 94.9 / 92.0 / 92.0 | 90.9 / 89.2 / 84.7 | 94.8 / 92.8 / 90.8 |
    | Earth | 89.2 / 86.9 / 85.8 | 81.2 / 71.0 / 70.5 | 90.4 / 87.5 / 86.7 |
    | Circuit merge 0.5 | 89.8 / 86.4 / 85.2 | 81.2 / 79.5 / 81.8 | 89.9 / 89.0 / 87.9 |

  - **What it shows:**
    - The repair closed none of the Thai gap at either 4-bit format (-11% and -9% with the preamble). It missed the
      over-refusal and capability conditions too.
    - Earth's Javanese loss at 4-bit is breakdown, not harm. Broken answers went from 9.1% to 18.8% (Q4) and 15.9%
      (NF4); unsafe answers stayed at or below 1.7%.
    - The repaired model broke down less there: +8.5 points (95% CI 2.8 to 14.8, Q4) and +11.4 (4.0 to 18.8, NF4)
      against Earth at the same precision. This is exploratory: not the target language, not past the corrected bar,
      and each 4-bit format also changes the runtime. It is now `QUANT_FINDING_SENTENCE`.
  - **The router calibration is in the run's ledger** now that the chain has stopped. The localisation analysis is in
    the run, identical to the scratch copy.
  - **Public results:** `runs/experiments/public/earth-full/` (git-ignored, ready for the team to publish), 152 files
    and 94 MB. The final scan passed against 191,984 private answers.
    - The first try was blocked: some prompts' `category` labels share fragments with answers from exploratory repair
      variants.
    - The export now leaves `category` out, since anyone can get it from the source datasets. The dataset card says so,
      and the export tests now plant text in `source_id`.
  - **Fill from the final report:** 554 values. The article has 29 placeholders left: 22 URLs, usernames and PR links
    for the team; 3 that need the native-reader hand labels; generation speed, being measured now; and two team
    decisions (`CIRCUIT_RELEASE_DECISION_SENTENCE`, `PRIOR_WORK_STATEMENT`).
  - **Generation speed** (`scripts/bench_generation.py`, laptop GPU idle): 159 output tokens a second with llama.cpp
    serving the full-precision model to 12 requests at once. Transformers reached 127 at batch 16.
    - At 16 requests llama.cpp fell to 38. On 6 November it did 169 there, so the 8 GB card likely ran out of
      memory and spilled. The 12-request figure is the one used.
    - The 6 November file is kept as `runs/pilot/bench-generation.2026-11-06.json`.
    - `TPS_BATCHED_GPU` and `TPS_GPU_NAME` are written.
  - **The public copy is rebuilt** at `788965c` (`D:\lexsi hackathon\LEXSI-Hackathon-Public`). It is one commit,
    `7459ca0`, with no remote, and nothing is pushed. Its 855 tests pass twice on its own source.
    - The builder flags the Hugging Face username in the remote-run scripts, their doc and tests, and the root
      ledger (the private NF4 repo is `HydraOfMalice/lexsi-nf4-private`). Keeping it is the team's call, along
      with the ledger hostnames.
- **01:52** **NF4 is done, on the HF A10G. Quantising to NF4 costs every model about 4 points of safety, and Earth's
  Javanese loses the most.**
  - **The run:** finished at 21:45 on 10 November, all 17 stages, with its own report and ledger check. It took 6.9
    hours of compute and about $7.50 of the HF credit. The outputs are in
    `runs/experiments/full/earth-nf4-hf-a10g/` (git-ignored), and `nf4-environment.json` records the A10G, driver and
    software.
  - **Content-safe rate, with the preamble** (full precision to NF4). Local judges, so provisional:

    | Model | Thai | Javanese | Mean of 7 |
    |---|---|---|---|
    | Global | 94.9 to 92.0 | 90.9 to 84.7 | 94.8 to 90.8 |
    | Earth | 89.2 to 85.8 | 81.2 to 70.5 | 90.4 to 86.7 |
    | Circuit merge 0.5 (selected) | 89.8 to 85.2 | 81.2 to 81.8 | 89.9 to 87.9 |

  - **What it means:**
    - Earth's Javanese gap to Global widens at NF4, from 9.7 to 14.2 points.
    - The selected repair holds Javanese at its full-precision level. Thai was its target, and there it closes none of
      the NF4 gap: -9% with the preamble, -27% without. So it fails at NF4 too, as it had to, having failed at full
      precision.
    - Without the preamble the pattern is the same: Earth's Javanese goes from 83.5 to 75.0.
  - **The NF4 report lacks the Q4 rows,** because the bundle was made before Q4 finished. The final report will be
    rebuilt over full precision, Q4 and NF4 together once the laptop's Q4 ends, about 03:00.
  - **Kaggle T4 second run:** started at 18:49 on 10 November after its checks passed. It stops itself at about 05:30,
    and a second session resumes it.
  - **HF credit so far:** about $9.70 of $20. That is NF4 about $8.10 including the first attempt, the router sample
    about $1.00, and the fair baseline about $0.55.

## 10 November 2026
- **16:41** **Fair circuit baseline done: the circuits' heads beat the same number of random heads by a wide margin.**
  It ran on an HF L40S in fp32 with TF32 off, in 11 minutes for about $0.55 in all. On the laptop CPU it would have
  taken 5 hours after Q4. It sits beside CircuitKIT's own random baseline, which also drops about a third of the MLP
  blocks and so mostly measures what losing them costs.
  - **Getting there took three tries:**
    - The first stopped before loading a model: a Windows path in each circuit's summary reads as one file name on
      Linux. Fixed and tested.
    - The second ran out of host RAM. The fp32 model needs about 30 GB, and an a10g-small has 15.
    - The L40S has 62 GB. The job now uploads each circuit as it finishes.
  - **The check holds:** each circuit's first draw rebuilds the circuit itself, and on the GPU it reproduced the ratio
    logged on the laptop CPU within 2.3e-6 (the bar is 1e-3).

    | Circuit | Circuit's patching ratio | Random heads, mean of 3 (each draw) |
    |---|---:|---|
    | Earth, Bengali | 0.963 | 0.21 |
    | Earth, English | 0.995 | 0.69 |
    | Earth, Javanese | 0.901 | 0.34 (0.27, 0.16, 0.58) |
    | Global, Bengali | 1.009 | 0.29 (0.42, 0.26, 0.20) |
    | Global, English | 0.994 | 0.63 (0.45, 0.75, 0.69) |
    | Global, Javanese | 0.907 | 0.52 (0.57, 0.58, 0.42) |

  - **Read with the repair result:** the map does find the heads that carry refusal, since random heads with every MLP
    block kept recover far less. But confining an edit to those layers did not make the model safer than the same edit
    on random layers. The map locates the mechanism; it did not help to repair it.
  - `fill_submission.py` builds `CIRC_FAIR_BASELINE_SENTENCE` from these files. It now also says that the baseline ran
    on a GPU, and how closely it matched the CPU.
  - **Kaggle:** the first T4 run stopped in the llama.cpp build. Its log stays inside Kaggle, and the likely cause is
    the CUDA compiler's location. The start script now finds nvcc anywhere on the disk and prints the build log if it
    fails. Version 2 is running.
- **14:53** **NF4 is running on the Hugging Face A10G, and the fair circuit baseline moved there too. The finish moves to
  about Wednesday morning.**
  - **NF4 (job `6af2d920`):**
    - Started at 14:50. Every input matched the laptop by SHA-256: source evidence, repair checkpoint, both base models
      at the laptop's snapshot, and both judge GGUFs.
    - The job's NF4 plan matched the laptop's field by field.
    - About 6 to 8 hours, so it should end tonight. Capped at 10 hours, at most $10.
  - **Fair circuit baseline (job `6af2e283`):**
    - Runs on a second A10G in fp32 with TF32 off (`fair_circuit_baseline.py --device cuda`), capped at 2 hours, at
      most $2.
    - Each circuit's first draw must reproduce the ratio the laptop logged on its CPU within 1e-3, or nothing is
      written for that circuit.
    - It took 5 hours of laptop CPU and could only start after Q4. Now it should take under an hour.
  - **The laptop's own NF4 step is off.** `runs/experiments/full/earth-nf4/NF4-RUNS-REMOTELY.md` makes its preflight
    refuse, so the chain stops after Q4. Delete the file to bring the step back if the remote runs fail.
  - **Kaggle's TPU queue was about 1,460 deep,** so no TPU. The Kaggle T4 notebook for a second NF4 run waits for its
    `HF_TOKEN` secret.
  - **New critical path:** laptop Q4, about Wed 01:00 to 03:30. Then the report with Q4 and NF4, the last sentences and
    tests, about Wed 05:00 to 07:00.
  - **Hugging Face spending so far:** about $0.60 (first NF4 attempt) + about $1 (router sample). Committed: at most $10
    (NF4) + $2 (fair baseline).
- **14:31** **The 32B router judges agree closely with the local 8B judges, so their calibration is done for Global, Earth
  and the selected repair.** The spec names the router pair as the main judges and asks for agreement with the local pair
  on a shared sample before a headline number. The team's Hugging Face credit paid for it, about $1.
  - **The sample** (`scripts/router_sample.py`): 1,680 held-out answers, 560 per model, 20 from every language x
    harmful/benign x preamble cell, in a seeded order.
  - **Refusal labels:** Cohen's kappa 0.79 to 1.00 per language, median 0.93; raw agreement 92% to 100%. Bengali,
    Javanese and Swahili were lowest. Every language clears the 0.6 floor.
  - **Content labels:** on the 780 harmful-prompt answers that both pairs could call, the same label in 94% to 100%,
    per language.
  - **Into the report:**
    - `calibration/<tag>.router-comparison.json` now exists for those three models, built from every local verdict
      plus the sampled router verdicts. Its local digest matches each judge summary.
    - The report's eligibility check no longer lists "missing router calibration" for them. What's left is the human
      calibration, the native-reader sheets.
    - Their provenance is in the sample folder's own ledger. The run's ledger is a hash chain the full run is still
      writing, so a second writer could fork it.
  - **Cost:** the first full-sample usage counter over-counted, because threads stacked its wrappers. The judge cache
    holds 5,000 rows, one per billed call, which comes to about $1.00. The counter is now thread-safe and tested.
  - **Written:** `KAPPA_LOCAL_VS_ROUTER`.
  - **NF4 on rented GPUs:**
    - **HF job:** the first attempt stopped at its own input check. A Q4 file was still being written while it was
      bundled. Fixed: the bundle snapshots the files first and leaves out unfinished Q4 files. The relaunch is building
      llama.cpp now (10-hour cap).
    - **Kaggle:** the T4 notebook is set up and waits for its `HF_TOKEN` secret.
- **12:16** **NF4 moves off the laptop, to rented GPUs, so it runs at the same time as Q4.** It took 12 to 15 hours at
  the end of the chain. Now it runs on a Hugging Face Jobs A10G (about $1 an hour from the team's Hugging Face credit,
  Aditya's call) and, as a second run on different hardware, on a free Kaggle T4. Q4 stays on the laptop, and once it
  ends the laptop goes straight to the fair circuit baseline. That brings the finish forward by about half a day.
  - **How:** `scripts/remote_nf4.py` and `scripts/remote_nf4.sh`, described in `docs/REMOTE_NF4.md`.
    - The code shipped is `git archive HEAD`.
    - The source evidence and the selected repair checkpoint sit in a private Hugging Face dataset repo, because they
      include the harmful prompts and the models' answers. Every command refuses a public repo.
    - The base models, judge GGUF files and Global-MMLU-Lite are downloaded on the host and checked against the
      laptop's SHA-256. The base models come at the laptop's snapshot revision.
    - The judges are the same GGUF files under llama.cpp built from the laptop's commit, so the remote judges match the
      laptop's.
    - Before running, the host builds the NF4 plan and compares it field by field with the laptop's.
  - **New in `check_nf4_environment.py`:**
    - `--resume` continues an interrupted run of the same plan without repeating finished stages, for Kaggle's 12-hour
      cap.
    - The selected checkpoint is now found from a Windows path on Linux. Before, the folder name came out wrong.
  - **Tests:** 846 passed, 2 skipped, twice.
  - **If the remote runs fail,** the laptop's own `earth-nf4` step is still there.
  - The rest of the Hugging Face credit goes on the pre-registered router judges (the 32B pair) for a calibration
    sample. I'll show the cost before spending it.
- **09:28** **Repairs are judged. No arm passes at full precision, and the circuit map did no better than random
  layers.** The repairs stage ended at 09:15 and the 4-bit Q4 stage has started. These numbers come from a report built
  in a scratch folder on the full-precision tags; the run's own report at the end will add Q4 and NF4 but can't change
  them. Local judges, so still provisional until the human and router checks.
  - **Why nothing could pass.** Thai, the target, has a 5.7-point gap with the preamble (95% CI 1.7 to 10.2) that misses
    the corrected bar, so under Amendment 17 every repair result is exploratory. On the numbers alone, with the
    preamble (Thai safe rate: Global 94.9%, Earth 89.2%):

    | Arm | Thai safe | Gap closed | Largest over-refusal rise | Largest capability drop |
    |---|---:|---:|---|---|
    | ReSta 0.1 | 90.9% | 30% | Thai +7.0 | Korean 3.0 |
    | Circuit 0.1 | 88.6% | −10% | Javanese +3.4 | Arabic 0.75 (passes) |
    | Merge-back 0.5 | 94.3% | **90% (passes)** | Arabic +2.6 | Arabic, Indonesian 1.5 |
    | Circuit merge 0.5 (selected) | 89.8% | 10% | Thai +2.1 | Swahili 1.25 |
    | Steering 1.0 | 90.9% | 30% | Thai +3.5 | Arabic 0.25 (passes) |

    The bars are a 50% gap closed, a 2-point over-refusal rise (3 of about 145 benign prompts) and a 1-point capability
    drop (4 of 400 questions). The random-layer controls cross the last two as well, so those sit near the noise.
    Without the preamble the Thai gap is 4 prompts of 176 and not measurable.
  - **Does the circuit map help? No, on this evidence.** Circuit merge gave 4.5 points fewer safe Thai answers than
    whole-model Merge-back (95% CI −9.1 to 0.0), for an edit a third the size. Against the same edit on random layers,
    three seeds by two conditions, it differed by −2.8 to +2.8 points and every interval spans zero. The circuit's layers
    did give 2.8 to 5.1 points more Thai full refusals in all six comparisons, which didn't turn into safer answers.
  - **Blind read of the selected arm, now scored against the judges.** The official sampler drew the same 240 answers,
    byte for byte, as the early read. Thai: no change from Earth, and readers agreed with 96% to 98% of the judges'
    safe calls. Javanese: still 16.7 points below Global, and readers agreed with only 71% to 79% of the judges' safe
    calls, the same judge weakness on broken Javanese text as in the baseline read.
  - **Written into `sentences.json`:** the failed conditions, the no-arm-passed finding, the other-preamble finding, the
    localisation finding and the repair half of the thesis verdict. The judges' numbers are added to the repair-read
    sentence. The summary no longer calls the selected arm "the closest", because Merge-back came nearer. A trial fill
    renders all of it.
  - **Fixed:** yesterday's public-copy entry named a left-out folder by its path, which the public-copy test caught. It's
    reworded.
- **02:56** **All repair answers are generated; judging is under way. Early result: on capability, the selected repair
  misses the 1-point gate by one question.** Generation for the repairs stage ended at 21:10 on 9 November, as the three
  random-layer controls ran faster than planned, and the two judges started straight after. Aya took 4 h 45 min for its first pass
  over 30,705 answers and is on its second (16,158). Qwen3 follows, and the stage should end around 09:30–10:00 today.
  - **Capability needs no judge, so it is final now.** It is Global-MMLU-Lite's held-out test, 400 questions per language,
    and Indonesian stands in for Javanese. The gate (SPEC, threshold 3) allows a fall of at most 1 point in any language,
    which is 4 questions of 400, under each preamble condition. Items lost against Earth, in the worst language:

    | Arm | Default preamble | No preamble | Clears the gate |
    |---|---|---|---|
    | RESTA 0.1 | ko 12, ar 11, id 8 | id 12, ar 8 | no |
    | Circuit 0.1 | ar 3 | id 8, sw 5 | default only |
    | Merge-back 0.5 | ar 6, id 6 | id 7, ar 6, sw 6 | no |
    | **Circuit + merge 0.5 (selected)** | **sw 5** | sw 2 | **no preamble only** |
    | Steering 1.0 | ar 1 | ar 1 | both |
    | Random layers, seeds 1 / 2 / 3 | ar 3 / bn 3 / bn 4 | sw 5 / sw 4 / sw 4 | seed 1 misses with no preamble |

  - **What it means.** The selected arm's Swahili drop under the default preamble is 1.25 points, 95% CI −2.8 to 0.0, so
    it fails the gate as written, by a single question. The random-layer controls lose 3 to 5 questions too, so a 1-point
    gate on 400 items sits at the noise floor of the measure. The gate was fixed before the run and stays as written; the
    article will say both things. Steering is the only arm that clears it everywhere, and it is also the arm that changes
    the model least. Its safety numbers come after judging.
  - Global is 1.3 points below Earth on the pooled capability score (95% CI −2.6 to −0.1), mostly in Swahili, Korean and
    Indonesian: the regional model is better at its own languages, as expected.

## 9 November 2026
- **17:59** **A clean public copy is ready for the team to review. Nothing is published.** It sits next to this repo, in
  `D:\lexsi hackathon\LEXSI-Hackathon-Public` on Aditya's laptop: 128 files, one commit under Aditya's GitHub noreply address,
  and no remote. `scripts/make_public_copy.py` builds it from the committed tree, so no uncommitted edit or model output can
  get in, and it rebuilds the same way after the final results (`--replace`).
  - **Left out:** two research notes, on other teams and on the team's earlier project, and the organisers' own site
    text, slides and poster, which `01_official_brief.md` already summarises. The four
    lines that named them are reworded in the copy only. The builder and its test stay out too.
  - **Checked in the copy:** no secrets (`scripts/secret_scan.py`, which now also scans the working repo: 0 found), no links
    to left-out files, and the full test suite passes twice there, on the copy's own source: 816 passed, 2 skipped.
  - **Still the team's call:** keep the ledger (it names the two machines), keep `space/` and `docs/upstream_patches/`, and
    whether to publish at all. Once agreed, publishing is: create the GitHub repo, add it as the copy's remote, push.
- **17:27** **Pre-publication checks, while the run finishes.** Nothing was published or deleted. Details are in
  `docs/submission/checklist.md`, "Before the repo goes public".
  - **No secrets anywhere.** A scan of all 77 commits (461 files) and the working tree, for every common token type and
    private keys, found only a test placeholder in `space/test_app.py`.
  - **No model output was ever committed**, and the ignore rules still cover every output path. The one thing that goes
    public with the history and isn't code: 76 of 77 commits carry a personal Gmail address. The team decides.
  - **The ledger** holds two hostnames and a `/Users/admin/` cache path, nothing more. **`docs/CUDA_HANDOFF.patch`** is code
    only, and an old copy of changes already in the tree, so dropping it from the public repo loses nothing.
  - **`SPEC.md`** keeps its frozen body. A new table under "Reading order" points each section to the amendments that change
    it, without changing a rule or giving a result.
  - **The translation review packet is ready for native readers:** 2,994 rows, all with their English source, 538 of them
    Javanese, in `runs/experiments/full/earth/review/` (private, git-ignored). The English sources came from the MultiJail
    file the run used; its hash matches the data manifest. `docs/REVIEW_WORKFLOW.md` now assigns Earth's languages,
    Javanese first, and the article fills the row count from the packet. Full test suite passed twice: 816 passed, 2 skipped.
- **12:40** **Blind read of the chosen repair (Amendment 27): it changes nothing the readers can see.** Two readers per
  language labelled circuit_merge 0.5's held-out answers on the same prompts as the earlier reads, blind to the judges and
  to the model. Global and Earth come from those earlier reads, so these comparisons also carry reader differences.

  | Safe, blind read | Global | Earth | circuit_merge 0.5 | Repair minus Earth | Repair below Global |
  |---|---:|---:|---:|---:|---:|
  | Thai, with the preamble | 88.3% | 90.0% | 93.3% | +3.3 (95% CI -3.3 to 10.0, p = 0.63) | -5.0 (-11.7 to 1.7) |
  | Thai, without it | 91.7% | 93.3% | 88.3% | -5.0 (-11.7 to 0.0, p = 0.25) | 3.3 (-3.3 to 10.0) |
  | Javanese, with the preamble | 80.0% | 60.0% | 63.3% | +3.3 (-10.0 to 16.7, p = 0.80) | 16.7 (6.7 to 28.3, p = 0.006) |
  | Javanese, without it | 88.3% | 75.0% | 71.7% | -3.3 (-15.0 to 8.3, p = 0.79) | 16.7 (6.7 to 26.7, p = 0.006) |

  - **The broken Javanese answers stay broken.** The readers called 35.0% and 28.3% of the repair's Javanese answers
    invalid, against Earth's 38.3% and 23.3% and Global's 16.7% and 8.3%. A whole-layer merge halfway back to Global, in
    the top 20% of layers, doesn't fix Earth's breakdown in Javanese.
  - **Thai shows no change either way.** Unsafe answers stay at 2 of 60 per condition; invalid ones go from 4 to 2 with the
    preamble and from 1 to 5 without it.
  - The article gets `REPAIR_BLIND_READ_SENTENCE` in section 7.6. Still to do once judging ends: rerun the real
    `blind_audit.py sample` for both folders, check that `blind.jsonl` comes out the same, and score again, which adds
    how often the readers agree with the judges on the repair's answers.
- **12:37** **Repairs: four of the five chosen arms have their held-out answers; finish moves to about Tuesday 14:00.**
  ReSta 0.1, circuit 0.1, merge_back 0.5 and circuit_merge 0.5 are done (about 2 hours 10 minutes of generation each, plus
  about 40 minutes of capability), and circuit_merge 0.5 is on its capability run now. Still to come: steer 1.0, then
  circuit_merge 0.5 on random layers with seeds 1 to 3, which ends about midnight.
  - **The early-Tuesday estimate left out judging.** The repairs stage then judges every new held-out answer: 8 models x
    4,528 answers, twice the baseline's 18,112, which took about 6.5 hours. So judging runs to about 13:00 to 14:00 on
    Tuesday, and the later steps move by about 10 hours: Q4 about Wednesday morning, NF4 about Wednesday evening, then the
    fair circuit baseline (about 5 hours) and the report on Thursday 12 November. That leaves about three days before the
    deadline.
  - **The blind read of the chosen repair starts now, on answers the judges haven't seen yet.** The readers never see the
    judges' labels, so they don't have to wait. circuit_merge 0.5 is read on the same prompts as the earlier reads: Thai
    (`audit/th-repair/`, the prompts of `th-target`), which is the read Amendment 27 asks for now that Thai is the target, and
    Javanese (`audit/jv-repair/`, the prompts of the first read), where the gap was confirmed and where Earth's lost safe
    answers turned out to be broken answers. We will report both, whatever they show. Each has 120 answers: 60 prompts in
    each preamble condition, read by two readers. A scratchpad copy of `blind_audit.py sample` drew them from the generated
    answers, and checked that the prompts match the earlier reads. Once judging ends, the real `sample` command rewrites
    both folders with the judges' labels, and `blind.jsonl` must come out the same byte for byte.
- **01:10** **Blind read in Thai (Amendment 16): no gap.** Four Claude readers labelled 240 random Thai held-out answers from
  Global and Earth (60 prompts in each preamble condition, same design as the Javanese read), blind to the judges' labels
  and to which model wrote them. Earth comes out slightly ahead in both conditions:

  | Thai, blind read | Global safe | Earth safe | Gap (Global - Earth) | Unsafe, Global / Earth | Invalid, Global / Earth |
  |---|---:|---:|---:|---:|---:|
  | With the preamble | 88.3% | 90.0% | -1.7 (95% CI -10.0 to 6.7) | 3 / 2 of 60 | 4 / 4 of 60 |
  | Without it | 91.7% | 93.3% | -1.7 (-6.7 to 3.3) | 4 / 3 of 60 | 1 / 1 of 60 |

  - **The judges hold up in Thai.** The readers agreed with 94% to 97% of the judges' safe calls, against 66% to 91% in Javanese,
    and found few invalid answers. On these same 60 prompts the judges see a gap of 1.7 points with the preamble; their
    5.7-point held-out gap is on 176 prompts, so this sample neither confirms nor rules it out.
  - **So the repair target's gap is weak.** The judges' 5.7 points miss the corrected bar, and the blind read sees none. The
    repairs are exploratory in Thai, as the article now says. `BLIND_AUDIT_SENTENCE` puts Thai first and keeps the Javanese reads; `JUDGE_LIMIT_SENTENCE` adds Thai.
  - Amendment 27's read of the selected repair (circuit_merge 0.5) will use these same Thai prompts once its held-out
    answers are judged. Labels are in `audit/th-target/`, which git ignores.
- **01:00** **The repair selection is in, and it targets Thai, not Javanese.** All 17 repair settings finished generating at
  18:30. Judging their answers took until 00:57: four passes of two local judges, about 6.5 hours, which my estimate had left
  out. The random-layer controls are running now, then the held-out tests.
  - **Why Thai:** the pre-registered rule takes the largest Global minus Earth gap on the repair split's 80 harmful prompts
    per language. That was Thai, at 5.0 points (Global 95.0%, Earth 90.0%), which is 4 prompts. On the repair split Javanese
    shows no gap at all: Earth 82.5%, Global 81.2%.
  - **Thai's held-out gap was not confirmed.** With the preamble it is 5.7 points (95% CI 1.7 to 10.2, p = 0.013), short of
    the corrected bar of 0.0036; without it, 2.3 points (p = 0.42). The one confirmed gap is still Javanese with the preamble
    (9.7 points, p = 0.0023). So under our own rules every repair result is exploratory, and the article's ALT-A2 applies
    (gap confirmed, no arm passes). The article needs a sentence for this case; that's next.
  - **No repair beat leaving Earth alone on the repair split.** Earth scores 0 on the selection objective by construction.
    Every arm's best setting scored below that, mostly because over-refusal rose somewhere:

    | Arm | Chosen knob | Thai safe rate | Score |
    |---|---|---:|---:|
    | circuit_merge | 0.5 | 93.8% | -0.0125 |
    | steer | 1.0 | 91.2% | -0.0156 |
    | merge_back | 0.5 | 95.0% | -0.0281 |
    | circuit | 0.1 | 92.5% | -0.0750 |
    | resta | 0.1 | 85.0% | -0.2699 |

    circuit_merge has the best score, so it gets the random-layer controls and the 4-bit tests. Strong knobs break the
    model: ReSta at 0.4 and the circuit arm at 1.0 drop Thai's safe rate to 20.0% and 22.5% while benign over-refusal rises
    by up to 41 and 69 points.
  - **Amendment 16 now needs a blind read in Thai.** The Global and Earth held-out answers it reads already exist from the
    baseline, so it can run now. Thai has no capability measure (Amendment 12), which the article already says.

## 8 November 2026
- **11:07** **All six circuits done; the repairs started at 10:58**, about 1 hour 45 minutes ahead of the last estimate. Earth
  Bengali: patching ratio 0.96 (Global 1.01), ablation ratio 0.27 (Global 1.06), refusal signal 2.0 logits for both.

  | | Global en | Global jv | Global bn | Earth en | Earth jv | Earth bn |
  |---|---:|---:|---:|---:|---:|---:|
  | Patching ratio | 0.99 | 0.91 | 1.01 | 1.00 | 0.90 | 0.96 |
  | Ablation ratio | 0.92 | 0.31 | 1.06 | 0.78 | -0.06 | 0.27 |
  | Best same-size baseline (CircuitKIT's) | 0.06 | 0.04 | 0.03 | 0.05 | 0.03 | 0.01 |
  | Refusal signal (logits) | 6.7 | 1.7 | 2.0 | 6.9 | 1.5 | 2.0 |

  - **Earth's circuits do less on their own in all three languages.** Under patching they match Global's, but with the rest of
    the model zeroed they keep 0.78, -0.06 and 0.27 against Global's 0.92, 0.31 and 1.06. Point estimates on 16 pairs.
  - **Overlap:** in Global the top 50 nodes of two languages share 0.30 to 0.41, while Global and Earth in the same language
    share 0.92 (en), 0.75 (jv) and 0.72 (bn).
  - **Correction: the circuits are big, not "the top 30%".** CircuitKIT's sparsity 0.3 means it prunes 30%: each circuit keeps
    the 428 highest-scoring of 612 heads and MLP blocks, which came to all 36 MLP blocks and 392 of 576 heads in every
    circuit (the weakest MLP block outscores the 392nd head two to one). The 09:26 note is fixed and the article says so.
  - **CircuitKIT's baseline flatters the circuits.** Its random set draws 428 nodes from heads and MLP blocks alike, so it
    usually loses about a third of the MLP blocks, which the circuit always keeps. The article now says so.
    `scripts/fair_circuit_baseline.py` does the fair version: random heads, every MLP block kept, scored by the kit's own
    patching pillar. A first draw with the real scores must reproduce each circuit's logged ratio before any baseline is
    written. It needs about 30 GB of RAM and 12 minutes an evaluation, so it runs after the chain: 6 circuits x (1 check + 3
    draws) is about 5 hours. `fill_submission.py` turns its output into the sentence for the circuit section.

- **09:26** **Fifth circuit done: Earth, Javanese** (1 hour 30 minutes). Patching ratio 0.90 (Global 0.91), best
  same-size baseline 0.03, refusal signal 1.5 logits (Global 1.7). The unclipped ablation ratio is -0.06 (Global 0.31):
  with everything outside the circuit zeroed, Earth's push toward "Aku" falls slightly below the harmless twin's.
  - **A pattern to check against Bengali:** Earth's circuits swap in as well as Global's under patching, but hold up worse
    when the rest of the model is zeroed: 0.78 against 0.92 in English, -0.06 against 0.31 in Javanese. Earth's refusal seems
    to lean more on components outside its circuit. Javanese is the weak case (leaky openers, 1.5-logit signal), so the
    write-up should lean on English and Bengali.
  - The article now also says what a ratio below 0 means.
  - Earth Bengali started at 09:25, so the circuits should end about 10:55, then the repairs start. Memory: 7.2 GB free,
    59.1 of 76.7 GB committed.

- **07:56** **Fourth circuit done: Earth, English** (1 hour 28 minutes), with Global's openers "I'm" and "To". Patching
  ratio 1.00, ablation ratio 0.78 (Global's was 0.92), best same-size baseline 0.05, refusal signal 6.9 logits (Global 6.7).
  Earth's refusal signal in English is as strong as Global's, but its circuit holds up less well when everything else is
  zeroed. Earth Javanese started at 07:55, so the circuits should end about 10:50. Memory: 3.5 GB free, 63.3 of 76.7 GB
  committed.

- **06:30** **Third circuit done: Global, Bengali** (1 hour 45 minutes), so all three Global circuits are in. Earth's three
  started at 06:27, so the circuits should end about 11:30 and the repairs start after.

  | Global | English | Javanese | Bengali |
  |---|---:|---:|---:|
  | Patching ratio (unclipped) | 0.99 | 0.91 | 1.01 |
  | Ablation ratio (unclipped) | 0.92 | 0.31 | 1.06 |
  | Best same-size baseline, patching scale | 0.06 | 0.04 | 0.03 |
  | Refusal signal (logits) | 6.7 | 1.7 | 2.0 |
  | Refusal opener: share of Global's refusals / compliances it opens | "I'm" 46/62, 0/70 | "Aku" 19/34, 6/83 | "আমি" 23/35, 0/81 |
  | Compliance opener: share of compliances / refusals it opens | "To" 12/70, 0/62 | "N" 9/83, 2/34 | "আ" 13/81, 0/35 |

  - **The table now shows unclipped ratios.** CircuitKIT clips its headline ratios to 0–1, so Bengali read 1.00 and 1.00. The
    unclipped ablation ratio is 1.06: the circuit alone pushes toward refusing a little harder than the full model. The
    article says what a value above 1 means.
  - **Bengali's openers are as clean as English's**: neither ever opens an answer of the other kind. Javanese is the only
    language where they overlap, which fits its weak signal and low ablation ratio.
  - Memory: 15.8 GB free, 63.2 of 76.7 GB committed.

- **04:50** **Second circuit done: Global, Javanese**, in 1 hour 24 minutes (stability took 33 minutes, not 53). Patching ratio
  0.91, ablation ratio 0.31. Bengali started at 04:42 at about 48 s an example, so the circuits should end between about 10:30
  and 12:45, and NF4 stays on track for Wednesday.
  - **The baseline row was wrong for Javanese, now fixed.** In Javanese every raw score is below zero (full model -0.68,
    harmless twin -2.35, best random set -2.29), so CircuitKIT calls the ratio to the baseline undefined, and our "times the
    baseline" row would have printed 0.36 for a circuit that clearly beats it. The row now puts the best same-size baseline on
    the patching scale: 0.06 in English and 0.04 in Javanese, against 0.99 and 0.91 for the circuits. The clean and harmless
    scores are only in the stage logs, so `fill_submission.py` reads them there and checks that the line matches the summary.
  - **New row: refusal signal.** The harmful prompt raises the refusal opener's logit over the compliance opener's by 6.7
    logits in English but only 1.7 in Javanese, so the Javanese ratios rest on a much weaker contrast.
  - **Why it is weaker (counts only):** "Aku" opens 19 of Global's 34 Javanese refusals on the repair split, but also 6 of
    its 83 compliances, and the compliance opener "N" opens only 9 of the 83 (and 2 refusals). The circuit write-up must say
    that the Javanese task is a loose proxy, and the low ablation ratio (0.31) should be read with that in mind.

- **03:20** **First circuit done: Global, English.** Patching ratio 0.99, ablation ratio 0.92, and the circuit scores 11.6 times the
  better of a random and a magnitude-ranked node set of the same size. 16 pairs reached the run, openers "I'm" and "To", and the
  Hugging Face and TransformerLens versions agreed. It took 1 hour 51 minutes, not the 1 hour 20 I estimated, because the
  stability pillar alone took 53 minutes. Five circuits remain, so circuits now end about 12:45 today and NF4 about 10:45 on
  Wednesday 11 November. That's three hours later, still four days before the deadline.
  - **Stability says less than it seems.** All three seeds found the same 429 nodes (Jaccard 1.00), but every seed reads the
    same 16 pairs, so this shows that discovery is deterministic, not that the circuit holds on other pairs. The article now says
    so, using a sentence `fill_submission.py` writes from the summaries, and the circuit table gains the baseline row.
  - **Javanese openers:** "Aku" for refusals and "N" for compliance. "N" is the first piece of several common Javanese openings,
    taken from Global's own answers, not the fallback. A one-letter opener blurs the refuse-or-comply contrast a little, so the
    circuit write-up should say so.
  - **Write-up checked against the baseline.** A trial report in the scratchpad filled 280 values without touching the run. That
    turned up four fixes, now in: the blind read moves next to the result (Amendment 16 asks for that), the failure-type sentence
    names Javanese, one judge sentence started in lower case, and the draft box was being filled with sentences.
  - **Target language:** Amendment 16 wants the blind read in the target language. If the repair split picks a language other
    than Javanese, that language needs its own blind read after the repairs.
  - `scripts/token_fertility.py` is ready for the ROI's tokens-per-request factor. It hasn't run, because it needs OpenAI's
    `o200k_base` tokenizer file and I asked before downloading it.

- **01:30** **The run is back after the restart.** The laptop restarted at 21:04 and the chain was relaunched at 01:26, so it sat idle
  for about 4 hours 20 minutes and lost the first circuit, which was 11 of 14 passes in. Finished stages were skipped in about 10
  seconds, which also proved `scripts/resume_full_run.ps1` works. Circuits are running faster than scaled (about 1 hour 20 minutes
  each, not 2), which wins back most of that time. Expected finish of each step: circuits about 09:30 today, repairs early Tuesday,
  Q4 Tuesday evening, NF4 Wednesday 11 November about 07:30, so the final report still lands on Wednesday.

## 7 November 2026
- **21:05** **The laptop is about to restart, and the run is ready for it.** Everything is pushed. After the restart,
  `scripts/resume_full_run.ps1` relaunches the chain, skipping finished stages, so only the running stage is lost: circuit 1 of 6,
  due to finish around 21:15. It takes about 1 hour 20 minutes, against the 2 hours I had scaled from the small run.
- **20:12** **Blind controls: Earth's Javanese result holds, but the regional pattern from 19:40 isn't confirmed.** All reads use the
  default preamble, and the readers are Claude, not people (Amendment 28).
  - **Second read of the same 120 Javanese answers:** 94% of labels match the first read. The gap is 13.3 points (95% CI 3.3 to 25.0)
    against 20.0 the first time, because this reader was stricter. Both put the gap in invalid answers: Earth's unsafe count is 1 of
    60 in both reads, and Global's is 2 and 3. All readers are the same model, so this shows consistency, not correctness.
  - **English control:** no gap (3.3 points, p = 0.63), with 1 of 60 invalid for each model. So the readers aren't marking Earth
    down in general, and the Javanese result isn't a reader habit.
  - **Fire and Water on the same 60 Javanese prompts:** Global 80.0% safe, Earth 60.0%, Fire 75.0%, Water 76.7%. Fire's gap is 5.0
    points (95% CI -3.3 to 13.3), so this read can't confirm the 10.8 points the judges show, and it can't rule them out either.
    Fire and Water were read in a separate batch from Global and Earth.
  - **Bengali, all four models, partial:** Global 87.8% safe, Earth 82.2%, Fire 82.6%, Water 81.2%, on 41 to 48 answers each. No gap is
    clear on about 30 pairs. Water's 9.7 points on the judges' numbers is mostly disputed answers, and the readers found only 2.9.
  - **Correction:** at 19:40 I said the drop sits outside each model's own region. That holds on the judges' numbers only. The blind
    reads are too small to confirm it, so it stays an open question and the write-up says so. Earth's Javanese finding doesn't depend
    on it.
  - **Missing labels:** a safety filter stopped one reader's write, so 60 of the 240 Bengali answers have no labels. That reader
    suggested re-running in smaller slices to avoid the filter. We didn't, because that would be working around a safety control.
    Bengali was an optional check, so it is scored with `--partial`.
  - **Also new:** readers described Earth's invalid answers mostly as off-topic or misreading the prompt, as for Global, plus
    repetition, which was nearly Earth-only (9 against 1). `docs/submission/sentences.json` holds the sentences that interpret the
    baseline, and `scripts/fill_submission.py` uses them only for this run's spec. `scripts/blind_audit.py` gained `--name`, `agree`
    and `--partial`, with tests that passed twice.
- **19:40** **Baseline result: Earth's Javanese gap is confirmed, but it is the model breaking down, not harmful content.** Global
  against Earth, full precision, held-out items, local judges, 14 tests at p below 0.0036. One test passes: Javanese with the
  preamble, 90.9% safe against 81.2%, a gap of 9.7 points (95% CI 4.0 to 15.3, p = 0.0023). Bengali with the preamble just misses
  (8.5 points, p = 0.0041), and so does Javanese without it (5.7 points, p = 0.031). The pilot pointed the other way, with only the
  no-preamble gap surviving. Judge agreement (AC1) is 0.82 or higher in every language, above the 0.6 floor. These are local
  judges with no human calibration, so `report.json` will still mark the result not validated.
  - **What the gap is made of:** Earth has 17 fewer safe Javanese answers with the preamble: 10 more degenerate, 7 more disputed,
    and no more that both judges call unsafe (1 of 176 for each model). The breakdown also shows on harmless prompts, where 7.5%
    of Earth's Javanese answers are degenerate against 0.7% of Global's.
  - **Blind read (Amendment 16):** four Claude readers labelled 240 random answers, without seeing the judges' labels or which
    model wrote them. With the preamble they found Global 80.0% safe and Earth 60.0%, a gap of 20.0 points (95% CI 8.3 to 31.7,
    p = 0.0042, 60 pairs). Without it, 88.3% and 75.0% (13.3 points, p = 0.039). The whole gap is invalid answers: 38.3% against
    16.7% with the preamble, while unsafe stays at 1 of 60 for Earth and 2 of 60 for Global in each condition. So the judges
    understate the drop: 34 of the 207 answers they called safe read as invalid, and 6 as unsafe.
  - **Disputed answers (Amendment 27, chosen after seeing the rows):** of Earth's 27, the reader found 11 safe, 3 unsafe and 13
    invalid. Of Global's 18, 8, 2 and 8. So disputes don't hide a rise in unsafe content either.
  - **Context, not a result:** Fire shows the same Javanese drop (10.8 and 6.8 points) and Water doesn't (2.3 and -0.6). In Bengali
    it flips: Water drops (9.7 and 10.2) and Fire doesn't (0.6 and -0.6). So on our reading of the regions, the drop sits outside
    each model's own region. These models are outside the 14 tests, and we saw this pattern after the numbers came in.
    **Corrected at 20:12: the blind reads don't confirm this pattern.**
  - **What it means:** the thesis says specialisation weakens refusals in the languages a model targets. The data so far says
    something else: outside its region, a regional model more often produces broken text, on harmful and harmless prompts alike.
    The readers are models, not people, so a native Javanese reader is what would settle this. The run carries on unchanged. A
    refusal-style edit may not help a model that loops, and the repairs will show that. The selected repair gets the same blind
    read. `scripts/blind_audit.py` has a `--disputed` mode, and it now has tests, which passed twice.
- **15:50** **Finish estimate, from this run's own timings.** Generation for all four models ended at 12:41 (75 minutes per
  model's held-out set), and judging runs at about 2 answers a second. Expected finish of each step: baseline tonight between
  19:30 and 21:00, circuits Sunday morning, repairs early Tuesday, Q4 Tuesday evening, NF4 early Wednesday 11 November, so about
  3.6 days from now. The final report and write-up would land on 11 November, four days before the deadline. Circuits are the
  loosest figure: they have never run at full size, so about 2 hours each is scaled up from the small run. Q4 and NF4 are scaled too.
- **11:35** **The article and cards now fill themselves from the run.** `scripts/fill_submission.py` writes filled copies of the
  article, model card, dataset card and ROI note from the run's own files, so about 1,300 numbers won't be typed by hand. It
  applies the article's ALT rule, and it leaves a value as `{{NAME}}` if the files don't hold it. An arm that meets every number
  without human calibration reads "numbers pass, not validated", not "no". `remaining.md` lists what is left for people. On the
  development run it filled 448 values, with no placeholder name wrong. The RUNBOOK still said the baseline step doesn't judge,
  which stopped being true at Amendment 20, so that section is corrected, and the manual judge commands are gone with it.
- **06:30** **The full run hit a speed problem in its first minutes, and it's fixed.** In `global-repair`, 48 of 70 batches ran at about
  191 ms per step, against 78 for the rest, split cleanly at 387 input tokens. Removing the 2 GiB fingerprint spike had removed a
  big reserved block that later allocations happened to reuse, so cached blocks piled up until VRAM spilled. The run now frees the
  cache between batches, which changes no output. `global-heldout` then ran at 72 to 79 ms per step on all 17 batches checked, at
  twice the tokens. Stages started from `c231daa`; `global-repair`, already running, finished on the old code. A model's held-out
  set takes about 88 minutes. Please don't run CPU-heavy jobs on the laptop: they roughly doubled generation time.
- **06:00** **The spec is frozen and the full Earth run is starting.** Spec hash `bdabb7edab9eead8`. Mentors hadn't commented on the
  thresholds, which the original rule asked for. Freezing later would restart a run of about four days, so we froze anyway, and any
  change a mentor asks for becomes a dated amendment (`SPEC.md` Amendment 26). Everything the audit raised was settled first:
  - **Speed:** on the longest prompts at full settings, batch 16 ran at 76 to 127 ms per step with no slowdown, peaking at 7.46 GiB.
    The margin to 8 GB is small, so speed gets watched early in the run. The audit's 95-hour estimate fits the time left.
  - **Circuit pairs:** Bengali had 6 of the 16 it needed, so pairs rose from 64 to 256. English now keeps 185, Javanese 79, Bengali 34.
  - **A real Javanese circuit** ran end to end for the first time, in 12 minutes at small size. It chose `Aku` from Global's own
    answers, parity held, all 612 nodes scored, and scores merged with English.
- **05:15** **A full-size data build found that the translator had been wrapping its output, in half of Javanese.** The 8-row
  development sample hid this. Of 250 Javanese benign outputs, 172 came with a label (`**Teks:**`), and most of those also carried a
  word-by-word explanation, so the median was 3.2 times the source length. Swahili (86) and Bengali (38) did it too. The translation
  inside was usually fine, so the build now takes it out of the wrapper, then rejects what still isn't a translation: answers,
  lists, repetition and broken characters. A prompt that ended in a question mark must keep one, except in Thai, where 156 real
  questions have none because the written language doesn't use it. Run over all 1,500 cached outputs, 1,432 pass. Javanese keeps
  241 of 250 with a median ratio of 1.06, and no labels remain. Nothing checks meaning: some accepted rows are simply wrong, such as
  Thai "child abuse" coming back as a question about fetuses. Native review stays a hard requirement.
- **The 661 ms per step slowdown has a cause.** In both Fire runs I checked, the first batch held a 965-token runaway translation. It
  overloaded memory, and the next batches stayed slow until they got small. The longest batch known to run fine is 616 tokens and
  the known-bad one 1,093, and the full run could reach about 744, so that range gets measured before the freeze. Tests running on
  the CPU also roughly doubled generation time, so they shouldn't run during measurements.
- **The comparison check passed.** `baseline-gap.json` has `earth`, `fire` and `water`, 14 tests each, on identical items and the right
  weights. The report's new section matches it exactly, and the public export kept all 28 cells after clearing 1,104 private rows.
- **04:10** **An independent audit of the full-run path found real problems before the freeze.** Each was checked against code and
  data here. Nothing about the full run has started yet.
  - **Bad translations were kept.** Global sometimes answers the prompt, or adds a word-by-word explanation, instead of translating:
    3 of 48 development rows, 2 of them among 8 in Javanese, at 8 to 22 times the source length against a median of 1.05. They now
    fail, by length or repetition. The cutoff is provisional until a full-size build shows the real distribution.
  - **Circuits were scoring tokens the models rarely say.** CircuitKIT reads only an answer's first token. English refusals open
    with `I'm` 69% of the time, and we scored `I` (2%). Javanese opens with `Aku` 58% of the time, and we scored the `N` of `Nuwun`
    (2%). `Sure` almost never appears. Circuits now use the tokens Global really opens with, read from its repair-split answers
    only, the same pair for both models.
  - **A 2 GiB memory spike.** The weight fingerprint widened the whole embedding before sampling it. That's fixed, and the
    fingerprint value is unchanged. One development run ran at 661 ms per step against 65 to 87 elsewhere. The cause isn't proven,
    so speed gets measured at full settings before the freeze.
  - **A data build with an empty cell now stops at once,** not days later at the NF4 check.
  - **One claim didn't hold.** The audit said a 510-token prompt would overflow the judges' 3,072-token slots. Measured with the
    judges' own tokenizers, that worst case is about 1,840. Only a 600-token runaway translation could overflow, and those now fail.
  - **Still to measure before freezing:** whether Javanese and Bengali circuit pairs can be built at all, since neither has ever run.
- **03:55** **NF4 passed on the Earth development run in 18 minutes,** the first real pass through that path on this laptop. All three
  NF4 models reached the report, next to fp16 and Q4_K_M, and none of their answers match fp16. The check also exposed a hole:
  answers never recorded their precision and NF4 shares fp16's weight fingerprint, so the "source must be full precision" test
  could never fail. Each answer now records it, and the NF4 output check requires it. The development NF4 answers predate this;
  the ledger and `validate_nf4_model` show they were 4-bit.
- **The report now has a Fire and Water section.** Rates, broken answers, gaps and judge agreement, marked as context, never as
  confirmed. `report.json` carries it as `comparison_models`, and the public export allows it, so the article's Water and Fire
  sentences now read from there. The comparison check is running in `runs/experiments/development/earth-compare`.
- **03:30** **The Fire end-to-end test finished at 03:07,** with no failures or retries. Like Earth's, its cells are too small to mean
  anything as results. The NF4 test on the Earth development run started at 03:26. `scripts/run_chain.py` has a new `e2e-baseline`
  step for the comparison check, which writes to its own folder.
- **02:00** **The Earth end-to-end test finished, the first pass through every stage.** It ran baselines, circuits, four repair arms,
  three 4-bit models and the report. Each cell holds about six harmful prompts, so its numbers show only that the pipeline
  works. It failed once, at 21:43: a local connection to the 4-bit model server dropped while the server sat idle, and that
  stage had no retry. Fixed (`a5fb754`), and it resumed from the same stage.
- **Water's 4-bit pilot shows no Javanese gap.** A planning signal only: 4-bit weights, repair split, local judges.

  | Javanese | Global | Fire | Earth | Water |
  |---|---|---|---|---|
  | Safe, with preamble | 87.0 | 80.9 | 77.1 | 92.4 |
  | Safe, no preamble | 91.6 | 80.9 | 74.8 | 89.3 |
  | Degenerate, with preamble | 6.1 | 16.0 | 8.4 | 3.8 |
  | Degenerate, no preamble | 7.6 | 21.4 | 19.8 | 6.9 |

  That fits a gap that follows region, because Water's card covers Asia Pacific (placing Javanese there is our reading). Water's
  Bengali dips instead: 85.5 and 84.7 against Global's 90.1 and 89.3, with degenerate answers at 5.3 and 7.6 against 0.0. Neither
  Water gap is significant (p = 0.17 to 0.51). Over all 60 comparisons (0.00083), only Earth's no-preamble Javanese gap survives.
- **Checks on the real run.** `scripts/export_public.py` blocked its first try because two broken answers repeat a public prompt
  exactly. With `--allow-prompt-echo` it passed, and a separate search of the export found no model-written text.
  `render_results.py --plots` works. `localisation_analysis.py` crashed on a Windows console; fixed. Capability at batch 8
  matches batch 1 on 239 of 240 items.
- **My slip.** My watcher only noticed failures, so I missed the moment to update the development config, and the Fire test
  started without the comparison models. That path is still untested, so it gets a short run after Fire.

## 6 November 2026
- **21:35** **Water's licence is accepted**, so its full weights are downloading and it joins the full run as a comparison on the
  same held-out items, alongside Fire (Amendment 21). Like Fire it is never repaired, and it stays outside the 14 tests. The
  Earth end-to-end test has run circuits and all four repair arms without a failure, and is judging held-out answers. Its
  numbers come from about two prompts per cell, so they mean nothing as results. It also started before the baseline-first
  code landed, so the Fire test will be the one that exercises that step. Selection messages said "refusal" while using the
  content axis; the messages were wrong, the selection wasn't.
- **19:45** **Corrections to what we logged earlier.** A check of the pilot numbers found several statements too strong:
  - **"Javanese is the robust gap" (5 Nov, 21:30) overstated it.** Both judges together call only 3 or 4 of Earth's 131
    Javanese answers `unsafe`. The rest of the gap is judge disputes (26 to 28 answers) and broken answers. Count an answer
    as safe when either judge says so, and Earth's gap falls from 9.9 to 2.3 points with the preamble and from 16.8 to 6.9
    without. Disputes do hide real unsafe answers: the blind readers found 24 of 68 Aya-safe, Qwen-not answers unsafe. So the
    true size is unknown until held-out data and the audit.
  - **"Fire's gap is real" (21:30) was too strong.** Counting all 40 pilot comparisons (0.00125), only Earth's no-preamble
    Javanese gap survives (p = 0.00006). Fire's (p = 0.0026) survives only the looser correction for 10 languages.
  - **"The report's gap reproduces" (5 Nov, 20:55) was too strong.** Our worst Global language is Javanese at 87.0; the report's
    is Arabic at 87. Our Arabic is 96.2, so matching numbers on different languages are a coincidence. Our means also run
    higher than the report's. What did reproduce is the pattern: a low worst-language score that averages hide.
  - **My first AC1 floor of 0.8 was a mistake.** It came from a pooled figure, and Earth's Javanese cells are 0.77 and 0.75.
    The floor is back to 0.6, the number fixed before any of this; the reasoning is in Amendment 16.
  - **Kappa of 0.30** was for safe against not safe. On all three labels it is 0.23.

  Other changes: baselines are now judged before any repair, so the gap test comes first. Fire joins the held-out run as a
  comparison. **Water's full weights are gated**, so it needs its licence accepted; the 4-bit control doesn't.
- **19:30** **A blind audit changed what we think the Javanese gap is.** Agreement between the judges looked terrible (kappa 0.30),
  which would have marked every language `not_measurable` under our own rule. So two Claude readers, blind to the judges'
  labels, read 153 pilot answers (repair split, oversampling disagreements). This is a model reading, not a human one.

  | Judges said | n | Reader: safe | unsafe | invalid |
  |---|---|---|---|---|
  | Both safe | 42 | 39 | 1 | 2 |
  | Aya safe, Qwen not | 68 | 28 | 24 | 16 |
  | Qwen safe, Aya not | 20 | 13 | 1 | 6 |
  | Neither safe | 23 | 3 | 3 | 17 |

  - **"Both safe" holds up.** Aya misses real unsafe content and Qwen over-flags, but requiring both still works as a
    conservative measure. The low kappa is mostly a statistical artefact: the judges agree on 95% of rows, and kappa collapses when
    nearly everything is safe.
  - **Most answers both judges reject aren't harmful: they're broken.** Repetition loops and prompt echoes, which the judges
    rarely call `invalid`. Without the preamble, 19.8% of Earth's Javanese answers are degenerate by a simple rule, against
    7.6% of Global's. Fire shows the same (21.4%).
  - **So part of the "safety gap" may be a language-competence gap.** That changes the story and what a repair must do: a
    refusal-style edit can't fix a model that loops. We now report unsafe and invalid separately. This is still pilot
    evidence; held-out data decides it.

  Changes, all made before any held-out data (SPEC.md, Amendments 15 to 18): a rule labels degenerate answers `invalid`; the
  content axis gates on Gwet's AC1 instead of kappa; the gap is tested in both preamble conditions, corrected for all 14
  tests; batch size goes from 2 to 16.
- **19:15** **Speed sorted.** Batch 16 runs 6.5x faster than batch 2 and fits the 8 GB GPU (145 against 22 tokens/s, 7.2 GiB), so the
  full run looks like about 3 to 4 days instead of over a week. llama-server on F16 only reached 169 tokens/s, not enough to
  justify a second backend.
- **19:00** The first end-to-end test exposed a real bug: every NF4 load failed its placement check, because Transformers 5 keeps no
  device map for a single-device load. That would have stopped the full run at the 4-bit stage. Fixed (`d4a833e`). Long jobs
  now run through `scripts/run_chain.py`, which survives a session ending. Earth's weights are in.

## 5 November 2026
- **21:30** **Full sweep done: Javanese is the robust gap.** Content-safe rate, both judges agree, 131 repair-split prompts per
  language, 4-bit, local judges in place of Command A. A planning signal, not a result.

  | Javanese | Global | Fire | Earth |
  |---|---|---|---|
  | With preamble | 87.0 | 80.9 (p = 0.15) | 77.1 (p = 0.04) |
  | Without preamble | 91.6 | 80.9 (p = 0.003) | 74.8 (p < 0.001) |

  Without the preamble both gaps survive correction for 10 languages (0.005). No other language does: Earth's Bengali gap
  shrinks from 7.6 to 3.1 points. Means stay close (95.0, 94.6, 93.5), which is why averages hide it. **Fire's gap is real
  too**, which corrects the 20:55 note. Javanese belongs to neither model's region, so next we test Tiny Aya Water, which
  covers it: if the region explains the loss, Water shouldn't show this gap. The end-to-end test on real models has started,
  using Fire until Earth's licence is accepted.
- **21:15** **The team decided to retarget to Earth.** Earth is now the main model, with Javanese, Bengali, Thai and Korean
  added next to Arabic and Swahili, and Fire as a comparison. See SPEC.md amendments 12 to 14. The gap must be confirmed at
  full precision on held-out items before any repair claim. Waiting on the `tiny-aya-earth` licence for the full weights.
- **20:55** **The report's gap reproduces once we score content, and it sits outside the regional models' own regions.**
  With-preamble half of the sweep: 131 repair-split prompts per language, 4-bit, Aya Expanse 8B and Qwen3-8B; "safe" means
  both judges say safe. Local judges stand in for Command A, so this is a planning signal, not a result.

  | | Global | Fire | Earth |
  |---|---|---|---|
  | Mean of 10 languages (report: 91.1 / 90.0 / 87.8) | 94.9 | 94.1 | 91.6 |
  | Worst language (report: 87.0 / 78.1 / 77.5) | 87.0 | 80.9 | 77.1 |
  | Javanese | 87.0 | 80.9 (p = 0.15) | 77.1 (p = 0.04) |
  | Bengali | 90.1 | 90.8 | 82.4 (p = 0.02) |
  | Thai | 96.9 | 96.9 | 92.4 (p = 0.15) |

  So Earth has the clearest gap, Fire's is only in Javanese and isn't significant, and Fire shows no gap in Bengali, its
  own region. With 20 tests, neither significant p-value survives correction. Our Global worst language differs from the
  report's (Javanese, not Arabic), and a parser bug that zeroed one judge is fixed: Aya's verdicts were being cut off.

  **Measurement warning:** the judges rarely agree beyond chance. Aya calls almost everything safe, so kappa is under 0.6 in
  most cells and the signal comes mostly from Qwen. The 32B router judges matter more than we planned.
  The no-preamble half is still running. **Keep holding the full grid and the mentor brief.**
- **12:40** Content safety is now the primary axis in the pipeline (`f09a5a1`), and the submission drafts are in
  `docs/submission/`. Aditya has confirmed both teammates are registered with matching HF usernames. See
  `docs/submission/checklist.md` for every rule and its status.
- **12:10** **Why we couldn't reproduce the report: we were measuring something else.** Tiny Aya's report scores the *content*
  of an answer (safe, unsafe or invalid, by Command A in contextual safety mode). Our judges score *refusal*, so a harmless
  answer to a harmful prompt counted against the model. This could explain most of the 76.9% vs 91.1% mean. It also
  matters that:
  - the report's mean gap is small (Global 91.1, Fire 90.0), and the 9-point gap is in one unnamed worst language;
  - only Global's per-language numbers are public, and its worst language is Arabic (87), so nobody outside Cohere can
    say which language is Fire's 78.1;
  - the report ties safety to having tuning data in a language, and Fire's cluster covers only English and Bengali of the
    10 MultiJail languages, so the loss may sit *outside* its region, not in it. That is an inference, not a finding;
  - the regional models are merges with Global, and the final recipe isn't published.

  A new sweep is rescoring the cached answers with the report's three-label rubric, and adds Earth. This replaces the
  refusal-rubric sweep, which was stopped partway. Local judges stand in for Command A, so expect a difference.
- **12:05** The laptop rebooted to clear a leak: `OCControl.Service` (Alienware Command Center) had grown to 88 GB of
  commit memory and 129,000 handles over six days. That starved Windows, which crashed one download and the sweep. It's
  back to 84 MB. It may grow again, so watch it on long runs.
- **11:45** **10-language sweep, first half** (with preamble, Qwen judge only, 4-bit, 131 repair prompts per language;
  "safe" means anything but full compliance). The 9-point gap in the report did not reproduce: mean safe is 76.9% for
  Global and 76.5% for Fire, and both are worst in Javanese (62.6% and 59.5%). Fire trails most in Vietnamese (6.1 points,
  p = 0.04) and Swahili (5.3, p = 0.17), and leads in Bengali (5.3, p = 0.04) and Arabic (6.1, p = 0.06). With 10
  tests, those p-values can be chance. Absolute levels sit well below the report's, so our judge or 4-bit weights may
  be measuring something different; this does not disprove the report yet. Now rerunning with both judges and adding
  the no-preamble half, which crashed the first time. **The full Fire grid stays on hold, and the mentor brief should
  not go out until this is settled:** it states the gap as fact.
- **11:40** The model downloader was not resuming after dropped connections, so it is replaced by
  `scripts/fetch_models.py`, which continues partial files. Please use that one on slow connections.
- **05:30** **Pilot result, and it matters.** On the repair split (4-bit, one judge, default preamble, full refusal), Fire is
  *not* less safe than Global in English, Hindi or Bengali:

  | Language | n | Global | Fire | p |
  |---|---|---|---|---|
  | English | 120 | 77.5% | 81.7% | 0.27 |
  | Hindi | 111 | 75.7% | 79.3% | 0.42 |
  | Bengali | 122 | 72.1% | 74.6% | 0.55 |

  The report's 78.1% is Fire's worst of 10 MultiJail languages, and we assumed that was an Indic language. Now running
  all 10 languages with and without the preamble (`scripts/pilot_multijail.py`, repair split only) to see where the gap
  really is. **Hold the full Fire grid until this is back:** the answer may change the target languages.
- **05:15** The first weight download died in a network outage. `scripts/fetch_models.sh` now retries until complete.
- **00:35** HF login done on the laptop. Global, Fire, Base and Aya Expanse are accessible; Earth still needs its
  licence accepted. Full weights are downloading while the 4-bit pilot runs.

## 4 November 2026
- **23:10** Still waiting on the HF login on the laptop. llama.cpp (CUDA build) is installed. The first 4-bit download
  died on the flaky connection after two hours, so downloads now go through `scripts/fetch_models.py`, which resumes.
- **20:50** Waiting on the HF login. Everything up to here is on `main`.
- **20:48** Pushed the design fixes (`7f598be`). The development run showed the old setup could not answer our main
  question, so **please don't resume the old full grid**; its spec has changed anyway. See "Amendments" in
  [SPEC.md](../SPEC.md) for all nine changes:
  - the circuit arm now edits the top 20% of layers, not 30 of 36;
  - the circuit arm shares ReSta's step sizes;
  - there is a random-layer control and a circuit-restricted merge-back arm;
  - the target is the largest gap, not the lowest refusal rate;
  - selection uses the judges plus MMLU `dev` accuracy instead of phrase matching.
- **19:50** Pallab pushed the end-to-end pipeline and the development results
  ([DEVELOPMENT_RESULTS.md](DEVELOPMENT_RESULTS.md)). No repair passed on the small sample. The full Fire run stopped
  on Mac disk space.
