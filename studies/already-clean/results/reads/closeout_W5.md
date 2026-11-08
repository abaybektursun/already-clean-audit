# Closeout check after the sixth unblind run (SPEC_D1 §9.6)

- Commit checked: `39260448bcccb7e708cdb373f41256f6f81b0fb8` (`drafts/already-clean`), read in the detached worktree
  `interp_research-check-closeout`. Data read in place: explain worktree `neurons/explain` 06135e88 (clean), harvest worktree
  `neurons/harvest` bca491a9 (clean), and S3 `s3://pg-lineage-769063704696/results/`.
- Run: 2026-10-24T02:54:34Z to 2026-10-24T03:09:56Z (UTC, `date -u`). Model: Opus 5.5 (`claude-opus-5-5`).
- Read-only: the only file written is this report. Copies of S3 objects were fetched into the session scratchpad
  `checker_closeout/`. No edit, no S3 write, no git state change, no subagent.
- Verdict scale. VERIFIED means I opened the content and it holds. "In substance only" means the spec's sentence is not met as
  written. REFUTED means the row's claim does not hold for the named content. UNCLEAR means I could not settle it from the record.

## 1. Acceptance rows (handoff/S6.md, "Acceptance against §9.4 and §10")

| row | verdict | evidence |
|---|---|---|
| §10 samples, one file per readout × arm | VERIFIED, in substance only | `bank/w5/verdict/samples/` holds 8 files: `q1_{G4,O7,Q4B}_poles.jsonl` (600 lines each), `q1_{G4,O7,Q4B}.jsonl` (10 each, the gallery), `q2_clusters.jsonl` (2,465), `q2_parent.jsonl` (4). The spec's form is "`out/samples/<readout>_<arm>.jsonl`" (SPEC_D1.md:619). The names and the directory differ. |
| §10 metrics.json | VERIFIED | `bank/w5/verdict/metrics.json` sha256 `002108c3ea15da9f7727105c54a3b1d21c7101ac74b9aff5c7c9a5965d85bcae`, created_utc 2026-10-24T02:28:17.941016Z. I checked figure numbers against it: F1 ONE 0.5633/0.5183/0.5967 (`q1.<cell>.fractions`); F2 pos 0.6430/0.5990/0.6897 against shuffled 0.0029/0.0026/0.0054 (`q1.<cell>.strong_rates`); F4 medians T_11 and T_12, recomputed from `q2.births` counts; F5 p_u 0.2654 [0.2109, 0.3208] (`q2.p_u`); F3 dead share 0.0 at T_0 and T_15. The S3 copy `results/neurons/w5/` has 30 objects; its metrics.json ETag equals the local MD5 (87dba120…). |
| §10 controls.json, manifest.json per harvest run | VERIFIED | The row names one path. I checked every banked run. Each stage directory has controls.json and manifest.json (w3/grid/1 8+8, 1b 8+8, 1c 1+1, w1/q4b/1 2+2, w0/verify/3 3+3, every W2 probe 1+1). The center probe's banked copy has no controls.json. S3 has 16 controls.json under `results/neurons/x/center_probe/1/out/`. |
| §10 registry | VERIFIED (present); content in substance only, see F30 | `bank/w5/REGISTRY.md` carries metrics sha 002108c3…. Its judgments hash 800fb2f774254ad5 and clusters hash afa666e2b0bf2650 match `shasum` of the files. The figure rows carry no input hashes, and several number rows give "same", "clusters, grid" or "judgments" without a hash. |
| §10 figures F1–F6 as SVG | VERIFIED | Six SVGs. The `<text>` titles match S6. Every footer reads "sha256 002108c3ea15da9f". F5's title is "p_u = 0.265 [0.211, 0.321]: H_SP; 1238 late-born clusters on 600 poles". |
| §10 figures as PDF | VERIFIED | Six PDFs (ReportLab). I decoded the content streams (ASCII85+Flate): every footer carries 002108c3ea15da9f, and F5's title is the run-6 one. Run 5's F5 PDF carries b020ea29bd01b190 and 0.262. Each MediaBox is the SVG size × 0.75 (F5: 690 × 360 for 920 × 480). |
| §10 figures as PNG | REFUTED (files exist; the content is cropped) | All six PNGs are 1840 × 1840 (`sips`), rendered by QuickLook (REGISTRY.md). They crop the figure: **F3 and F4 stop at T_13**. F4 drops the T_14/T_15 bins (main 47+7, non-main 239+29 births). **F5** loses the "H_RG if upper < 0.10" and "H_SP if lower > 0.20" labels (SVG x = 782; the crop is at x ≈ 769). **F1** and **F2** lose most of the reference and null legend text ("dashed: the ess…", "dashed: 1%, θ_…"). **F6** loses its bottom: the third pole's last exemplar line, its time axis, the legend and the footer with the sha. I viewed every PNG. See F24. |
| §10 code: infra/pg/neurons_*.py | VERIFIED | `neurons_explain.py`, `neurons_figures.py`, `neurons_llm.py`, `neurons_verdict.py` and `neurons_prompts/`. Every prompt sha256 in `bank/w4/o7_s_v2/explain/manifest.json` and in the three W2-C manifests equals the file's sha256 at HEAD. The one exception is the W2-C manifests' clusterer v1 (557a806d…), which is not a file at HEAD; W2-C does not use the clusterer. |
| §10 code: infra/aws_pg/jobs/neurons_*.py, corpus manifest | VERIFIED, in substance only | `neurons_corpus.py` and `neurons_harvest.py` are in `infra/aws_pg/jobs/`. The corpus manifest is not at the named path. It sits in the harvest bank and S3 (`w1/q4b/1/out/corpus/corpus/manifest.json`), with sha256 0c99a8e8…, equal to `corpus_manifest_sha256` in every harvest manifest I opened. |
| §10 study files | VERIFIED, in substance only (F32) | The study directory holds SPEC_D1.md, JOURNAL.md, handoff/, checks/ and INDEX.md. RESULTS.md and figures/ are not in it. They live in the explain worktree's `bank/w5/` (not under git) and on S3. |
| §10 RESULTS.md | VERIFIED (present, run 6) | The header carries sha 002108c3…. The finding line gives p_u = 0.265 [0.211, 0.321] over 1,238 clusters. One "____" occurs only in the header's own description. Line 131 states a stale count (F25 b). |
| §10 samples' content, Q1 per pole | REFUTED for "contexts"; counts VERIFIED | Across all 1,800 rows, `q1_<cell>_poles.jsonl` has no context field (the union of keys has no `explainer_saw`; rows are about 5.6 KB). The 30 contexts as the explainer saw them appear only in the gallery files `q1_<cell>.jsonl` (`explainer_saw`: Q4B 8, G4 6, O7 6 poles) and in the explain directories' `requests/c1-explainer.jsonl`. The passing/failing counts in the row match exactly (e.g. Q4B UNK 11/180, O7 UNK 14/234). Declined poles (passes null) are Q4B 5, G4 4, O7 1. See F23. |
| §10 samples' content, Q2 | VERIFIED as described; spec in substance only (F29) | `q2_clusters.jsonl`: 2,465 rows with `judgments` (answer, correct), the birth table (med, present_pct, birth_pct, …) and `ranks` (exemplar indices, not texts). `q2_parent.jsonl`: 4 rows with `judge_saw`, answer and truth. |
| §10 PENDING readouts listed | VERIFIED | No PENDING in grid.md or RESULTS.md. The gallery lists absent class/flag pairs as "no such pole in the data" and does not omit them. |
| §10 manifest.json per harvest run (content) | VERIFIED | `w3/grid/1/out/t15_step1413814/manifest.json` has corpus_manifest_sha256 0c99a8e8…, tokens_sha256 2aca80ca…, revision 373bad25…, transformers 5.9.0, torch 2.12.0+cu130, timing_s and tokens_per_second 3334.3. The same fields are present in w1/g4/1, w1/q4b/1 and w2c/q4b_probe/1. |
| §10 events.jsonl, run.log, compute per stage | VERIFIED as described, except the banked count; spec met in substance only (section 3, F28) | All 26 S3 runs listed have both files. For all 26, run.log has the same ETag and size as its job.log. Banked runs with the two files: **20, not 21** (the 21st directory at that depth, `w2d/q4b_probe/out`, is empty and is not a run). |
| §10 model id of every agent call | VERIFIED, in substance only (F31) | Every `calls.jsonl` under the explain bank (20 files, 0 rows without `model`). `w4/o7_s_v2`: 12,902 rows (judge manual 11,497; clusterer manual 1,088, API 317). `models_by_role` is in each manifest. The spec places it in `out/manifest.json`. |
| §10 index saying what is live | VERIFIED for the results row | INDEX.md's results row names the sixth run (002108c3…). Its code row says `neurons/explain` ef81cd87, but HEAD is 06135e88 (F25 c). |
| §9.4 handoff parts | VERIFIED, in substance only (F26) | S6.md has "Record state", "Open jobs", "Exact next action" and "Resume steps from a cold start". None of S2-S4.md, S5.md or S6.md labels any record item NEW or REUSED. Open jobs = none, checked: `vastai show instances` reports 0 instances, and `aws ec2 describe-instances` (us-east-1, pending/running/stopping) returns nothing. |
| §9.4 roster as first line of the status report | VERIFIED, in substance only (F27) | The Roster is line 3 of S6.md, after the title. It gives run/total per wave, with no unrun or parked counts. The GPU run x/center_probe/1 is not counted as a cell. |
| §13 item 6, companion paper | VERIFIED | No companion paper exists as of 2026-10-24T03:05Z (section 4). The closure reading is faithful. |
| §14 closeout paragraph and §15 open items | UNCLEAR (§14 VERIFIED; §15 "nothing open" not shown by §15's text, F35) | The §14 run-6 addendum (SPEC_D1.md:788) matches metrics.json: a_ctrl 0.9242, A 0.8116, unmatched 53 (share 0.0473), hard control 0.7983, flags 125 (0.1010). B4 m-within-1 moved from 0.668 to 0.678 and is not restated (F25 f). For §15, see F35. |

Rows: 19 VERIFIED (of which 10 only in substance or with a noted exception), 2 REFUTED, 1 UNCLEAR.

## 2. §11 against the sixth run

- `python3 -I runbook/fill_spec11.py SPEC_D1.md …/bank/w5/verdict/grid.md` (no `--write`) printed "0 cell(s) differ across 0
  row(s)". In my own comparison, all 10 table lines of §11 (header, separator and the 8 data rows) are byte-identical to grid.md.
- `diff` of run 5's and run 6's grid.md: exactly four rows differ (p_u, H_MIG, r(c), hard-decoy). This matches the note's "Four
  cells moved".
- §11 note for the sixth run against metrics.json (run 5 values from `bank/w5_run5_20261024T0103Z/verdict/metrics.json`):

| note says | metrics.json | verdict |
|---|---|---|
| sha256 002108c3ea15da9f… | `shasum` 002108c3ea15da9f7727…; runs 1–5 hashes in the note match their kept directories (9129a195, 56886c88, 4b2f3c77, cf696d9e, b020ea29) | VERIFIED |
| p_u 0.262 [0.207, 0.316] → 0.265 [0.211, 0.321] | run 5 0.26172 [0.20730, 0.31641]; run 6 0.26540 [0.21089, 0.32083] | VERIFIED |
| A 0.812 over 1,238 clusters; a_ctrl 0.924 | A 0.81159; n_clusters 1238; a_ctrl 0.92417; degenerate draws 0 of 10,000 | VERIFIED |
| H_MIG 16 (0.027) → 17 (0.028) | run 5 16, 0.02667; run 6 17, 0.02833 | VERIFIED |
| median r at birth 0.2908 → 0.2913 | 0.29083 → 0.29128 | VERIFIED |
| hard on flagged 0.351 → 0.356 (125 flagged) | 0.35081 → 0.356; n_flagged 124 → 125 | VERIFIED |
| F2 sensitivity 0.272 [0.216, 0.328] → 0.275 [0.220, 0.332] | 0.27190 [0.21614, 0.32795] → 0.27540 [0.21969, 0.33198]; 1,177 clusters | VERIFIED |
| primary set 1,239 → 1,238; 13 in, 14 out; 84 lines set aside; w4j2 78 lines, seed 4113; 205 clusterer rows changed | `status_counts.late_primary` 1239 → 1238; plans `w4j-judge.left_primary_F15_…` 84 lines, `w4j2-judge.jsonl` 78; manifest `seeds["judge:w4j2"]` 4113, `f15_left_primary` 14 clusters; clusters.jsonl against `clusters.before_recollect_20261024T021426Z.jsonl`: 205 of 1,200 rows differ, and 205 ids are in the two `*_removed` files | VERIFIED |
| one judge line re-answered with the same verdict | `judge:w4j:991` answered "A" in the removed row and "A" in the new row (utc 2026-10-24T01:51:53Z); judgments.jsonl correct = true | VERIFIED |

- §11's heading says "(filled once, at W5, whatever the result)" (SPEC_D1.md:641). The grid was filled six times, and the
  refusal was passed by moving `bank/w5` aside five times (runs 1–5 kept). This is the W4–W5 checker's F20 (accepted). It is
  disclosed in RESULTS.md:131, but with a wrong count (F25 b).

## 3. §10 derived files (run.log, events.jsonl)

| run | lane | run.log sha256 = job.log sha256 | S3 = bank copy | event lines traceable | first line |
|---|---|---|---|---|---|
| w1/q4b/1 | AWS | 2dfd7c02… = `job.log` 2dfd7c02… | run.log and events.jsonl identical | 6 of 6 (STATUS "exit=0"; pipeline_status equal to the file; 2 stage starts and 2 finishes equal to `out/{q4b,c0}/manifest.json` started_utc, finished_utc, timing_s, load_s, tokens_per_second, n_tokens_scored, device_name) | `derived: true`, written 2026-10-24T02:44:12Z, sources STATUS, out/pipeline_status.json, two manifests |
| w3/grid/1b | vast | 00c22721… = `vast/job.log` 00c22721… | identical | 27 of 27 (9 of 9 timings.tsv rows with epoch → ISO equal; rental_complete fields equal; 8 starts and 8 finishes equal to the 8 stage manifests) | `derived: true`, 02:44:12Z, 11 sources |
| x/center_probe/1 | vast | 24c113df… = `results/mdl/mendeleev_neurons_x_center_probe/job.log` 24c113df… | identical | 43 of 43 (9 timings, rental_complete, pipeline_status, 16 starts and 16 finishes) | `derived: true`, 02:51:34Z, 19 sources naming the `mdl/…` paths |

- All 26 runs: `aws s3api head-object` gives equal ETag and ContentLength for each run.log and its job.log. Event counts per run
  (all 26 fetched): AWS lane 2–34 and W3 grid 1/1b 27 each, 1c 13, center probe 43. These match the JOURNAL 02:55Z entry. Every
  first line has `derived: true` and a sources list.
- What the files do not cover. AWS-lane `status` and `pipeline_status` events have no time (2 per AWS run, 1 per vast run). The
  corpus stage's compute (`read_seconds` 146.4 in `w1/q4b/1/out/corpus/corpus/manifest.json`) is not carried, because the
  script reads manifests at depth `out/<stage>/manifest.json` only. Failed attempts w0/verify/1 and /2 carry 0 and 1 stage
  events. The center probe's event lines name `vast/timings.tsv` and `vast/_rental_complete.json` as source, and no such path
  exists under `results/neurons/x/center_probe/1/` (the header names the real `mdl/…` paths). No explain, clusterer, judge or
  verdict run has run.log or events.jsonl; per-call `utc` is in `calls.jsonl`. One rental's log serves several run prefixes
  (w2d/q4b_probe/4 = /5, and /6 = /7 = /9 by ETag), as the job.log was per rental.
- Whether §10 is met. "`events.jsonl`, `run.log`, compute per stage" (SPEC_D1.md:624–626) is **met in substance, not as
  written**. The runs never wrote these files. They were derived at closeout (2026-10-24T02:44–02:51Z), after the unblind,
  from the runs' own files, verbatim for run.log and traceable line by line for events.jsonl. They cover the GPU harvest runs
  only. §15 item 15 records the name defect.

## 4. §13 item 6, the companion paper

- My searches, 2026-10-24 about 03:00–03:06Z:
  - WebSearch (extended) `"Neural Networks Are Already Clean" Cammarata Goodfire paper`. Results were coverage only, and one
    summary says "The technical companion paper is still forthcoming."
  - WebSearch `Goodfire "already clean" neurons companion paper technical report`. No companion was found.
  - WebSearch `Cammarata Tigges language model neurons paper arXiv 2026`. No arXiv paper was found.
  - WebSearch restricted to arxiv.org (`Cammarata neurons "one main job" Qwen3-32B neuron poles`). No match.
  - The essay page https://already-clean.goodfire.com/ (whole page, two reads). It says "A technical companion paper will
    offer a more detailed account of language model neuron experiments" and "brings in experiments from our forthcoming
    technical paper studying language neurons in depth" (the section done with Curt Tigges). It has no link to any paper.
  - https://www.goodfire.com/research (redirected from goodfire.ai). The newest posts are "Neural Networks Are Already Clean"
    (October 15, 2026), "Training and Deploying Production Cyber Monitors on Kimi K3" (October 8) and "Better biosecurity
    monitors…" (October 1). None is a companion paper.
  - arXiv API: `au:"Nick Cammarata"` returned 1 entry, 2606.25234 (2026-06-23, block-sparse featurizers).
    `au:"Curt Tigges"` returned 9 entries; the newest is 2610.20767 (2026-10-14, robot-model featurizers), not a neuron
    paper. `au:Cammarata_N OR au:Tigges_C` returned 0.
- Result: **no companion paper exists as of 2026-10-24.** The essay itself calls it forthcoming. The freeze is
  2026-10-20T16:50:32Z (SPEC_D1.md §0). A paper absent today did not appear before the freeze.
- The spec says: "Before freeze, read at source …: … the essay's companion paper if it appears before freeze (its tests
  replace ours where they are the same test)" (SPEC_D1.md:737–744). §15 item 16 says: "closed 2026-10-24 as not applicable:
  the item was conditional on the paper appearing before the freeze; it did not (searched 2026-10-23 and 2026-10-24, general
  and arxiv.org-restricted). Should it appear later, comparing its same-test rows with this study's is a dated amendment after
  closeout and changes no frozen rule (§6.7)."
- Judgment: **a faithful reading.** The only obligation in the sentence is conditional on appearance before the freeze, and
  the condition was not met. The spec asked for the check "before freeze", and the study made it after the unblind (see F34).
  The outcome is the same, because the paper does not exist even now. Item 16's last sentence adds a procedure §13 does not
  state; it does not contradict §13. The study's search did not look at the essay page, Goodfire's research index or the
  co-author named in the essay. I looked at all three and found nothing (F36, low).

## 5. §9.4 and §9.6

- §9.4 (SPEC_D1.md:603–607): "Each session ends at a wave boundary after writing `handoff/S<n>.md` with the NEW/REUSED record
  state, open jobs, exact next action and resume steps. The first line of every status report is the run/unrun/parked count
  over the cell roster (W0: 4 cells; W1: 3; W2: 4 scoring jobs; W3: 2; W4–W5: 0 GPU)."
  - S6.md exists for S6 = W5 and closeout. The four parts exist by heading. Open jobs = none, which I checked on vast.ai and
    on AWS us-east-1 EC2.
  - The Exact next action is "None in the spec's waves…". The resume steps name explain ≥ 06135e88 and harvest ≥ bca491a9, and
    both worktrees are at those commits and clean. The 14 runbook files the handoff lists exist. The incident log has 44
    lines, as stated.
  - Not as written: no record item is labelled NEW or REUSED (F26). The roster is the third line, not the first, and has no
    unrun or parked counts (F27).
  - The Record state's "Spec" line still names run 1's metrics hash (F25 d).
- §9.6 (SPEC_D1.md:612–615): "Read-only, non-blocking sanity checks after every wave, run as Opus agents only, never more than
  two at once, each in its own worktree." This check against that form:
  - Opus: yes (`claude-opus-5-5`).
  - Read-only: yes, apart from this report and scratchpad copies of S3 objects.
  - Own worktree: yes for the study files (the detached `interp_research-check-closeout` at 39260448). The data banks
    (`bank/`, not under git) were read in place in the lead's explain and harvest worktrees. This report is written, as the
    brief directs, into the drafts worktree `interp_research-drafts`.
  - Non-blocking: nothing in the spec's waves waits on it (S6 "Exact next action: None").
  - "Never more than two at once": UNCLEAR; it cannot be seen from inside this agent.
  - The brief came from the lead's message, not from `checks/CHECKERS.md`. CHECKERS.md lists no closeout checker.

## 6. Flags (F23 onward)

**F23 (high). The S6 acceptance row for Q1 samples names contexts that the named file does not contain.**
- Spec: "for Q1, per pole the 30 contexts as the explainer saw them, the explanation, every Agent 2 text with its score and the
  strong flag, the shuffled-pole scores, the class" (SPEC_D1.md:619–621).
- Evidence: S6.md row "§10 samples' content: Q1 per pole (contexts, …) | `verdict/samples/q1_<cell>_poles.jsonl`, 600 poles
  per cell". The union of keys over all 1,800 rows of those files has no context field. Contexts exist only for the gallery
  poles (`q1_<cell>.jsonl` `explainer_saw`: 8, 6 and 6 poles) and in `bank/w2c/<cell>/explain/requests/c1-explainer.jsonl`.
- What would satisfy it: per-pole explainer input in the sample files, or a per-pole pointer (request id and sha256) to
  `requests/c1-explainer.jsonl`, with the row corrected to say which.

**F24 (high). The PNG renders are cropped, and the S6 figure review describes them as complete.**
- Spec: "Figures (PNG for review, PDF for the paper; title states the finding; null drawn on the panel)" (SPEC_D1.md:628).
- Evidence: every PNG is 1840 × 1840 (QuickLook) and cuts the figure at about SVG x = 769.
  - F3 and F4 show T_0–T_13 only. F4 omits 322 of 2,454 births (T_14 and T_15).
  - F5 loses the H_RG/H_SP rule labels.
  - F1 and F2 lose the reference and null legend text.
  - F6 loses its bottom, including the footer.
  - S6.md:92 says that for the sixth run "F3, F4, F5 and F6 opened again from the PNG renders (footers carrying metrics sha256
    002108c3…)". The F6 PNG has no footer, and the F3 and F4 PNGs do not show T_15, which their titles and §11 cite.
  - The SVG and PDF files are complete.
- What would satisfy it: PNGs rendered at each SVG's own aspect ratio that contain the whole figure, re-reviewed.

**F25 (high by the brief's definition: stated facts are wrong; none changes a §11 number or label). Stale or wrong statements in the record.**
- Spec: §10 "an index saying what is live" and §9.4 "record state", plus RESULTS as the readout.
- Evidence:
  - (a) S6.md:84 and JOURNAL.md:1377 say "21 banked runs". There are 20 (`find … -name run.log` gives 20).
  - (b) RESULTS.md:131 says the "filled once" refusal "was bypassed three times by moving `bank/w5` aside". Five earlier runs
    are kept (`bank/w5_run1…run5`). The sentence comes unchanged from `runbook/results_hand_edits.py:59` since run 4.
  - (c) INDEX.md:19 code row: `neurons/explain` ef81cd87. HEAD, S6 and REGISTRY say 06135e88.
  - (d) S6.md:14 "Spec: … §11 filled 2026-10-23 … (metrics.json sha256 9129a195…)". That is run 1. §11 now carries run 6.
  - (e) S6.md:92's per-figure list gives F5's title as "p_u = 0.262 [0.207, 0.316]: H_SP; 1239 late-born clusters". That is
    run 5's title, and the same paragraph says it is now 0.265.
  - (f) §14's 2026-10-23 paragraph says "B4 m-within-1 is 0.668". Run 6's metrics give 0.678, and the run-6 addendum
    (SPEC_D1.md:788) restates the unmatched count but not B4.
- What would satisfy it: a dated correction of each item.

**F26 (medium). No handoff carries a NEW/REUSED record state.**
- Spec: "`handoff/S<n>.md` with the NEW/REUSED record state" (SPEC_D1.md:605–606).
- Evidence: `grep` over handoff/*.md finds no NEW or REUSED label. S6's Record state lists files without that marking. The
  S6 acceptance row says "yes".
- What would satisfy it: each Record state item marked NEW (written in S6) or REUSED (from an earlier session).

**F27 (low). The roster line is not the first line and omits unrun and parked counts.**
- Spec: "The first line of every status report is the run/unrun/parked count over the cell roster" (SPEC_D1.md:606–607).
- Evidence: S6.md line 1 is the title, and line 3 starts a roster paragraph of about 9 lines with n/n counts only. The center-probe GPU run
  (x/center_probe/1, $4.67) is not counted, while the spec's roster says W4–W5: 0 GPU.
- What would satisfy it: line 1 as run/unrun/parked per wave, with the extra GPU run counted or named.

**F28 (medium). §10's run logs exist only as files derived at closeout, for harvest runs only.**
- Spec: "`events.jsonl`, `run.log`, compute per stage" (SPEC_D1.md:624–626).
- Evidence: section 3. The files were written 2026-10-24T02:44–02:51Z, after the unblind. No explain, judge or verdict run has
  them. Status events carry no time. The corpus stage's compute is not carried. The center probe's event lines name
  `vast/…` source paths that do not exist under its prefix.
- What would satisfy it: §15 item 15 stating this scope; the corpus-stage timing added; the center probe's event `source`
  fields naming the `mdl/…` paths, as its header does.

**F29 (medium). Q2 samples carry the judge's view for 4 tests, not per cluster.**
- Spec: "for Q2, per cluster the exemplars, the two records as the judge saw them, the judge's answer and the truth, the birth
  table" (SPEC_D1.md:621–622).
- Evidence: `q2_parent.jsonl` has 4 rows with `judge_saw`. `q2_clusters.jsonl` gives exemplars as rank indices (`ranks`) and
  answers with truth, with no record text. There are 11,028 judge lines over 1,238 test clusters and 600 controls.
- What would satisfy it: per-cluster pointers (request ids and sha256) to the judge requests in `bank/w4/o7_s_v2/explain/`, or
  the texts themselves.

**F30 (medium). The registry lacks input hashes for figures and for several numbers.**
- Spec: "Registry: every figure and number → generating script, input hashes, DISCOVERY or CONFIRMATORY" (SPEC_D1.md:627).
- Evidence: REGISTRY.md's figure table gives metrics fields and no input hash. Number rows give "same", "clusters, grid",
  "judgments" or "the two p-values". Several RESULTS numbers (b's drift share, flag rate, B4, the center-curve parts) are not
  listed.
- What would satisfy it: an input sha256 on every row, and a row for every number RESULTS cites.

**F31 (medium). For 12,585 manual-backend calls, the recorded model id is the requested id, not the answering model.**
- Spec: "model id of every agent call" (SPEC_D1.md:625).
- Evidence: `bank/w4/o7_s_v2/explain/calls.jsonl` judge 11,497 and clusterer 1,088 rows have `backend: manual` and model
  `anthropic/claude-sonnet-5-5`. `runbook/manual_import.py:6–7, 56–57` requires a row's model to equal the request's model.
  §12's backend amendment says the serving is "Claude Code's `sonnet` alias". The model that answered is not recorded.
- What would satisfy it: the model id read from each producing transcript (the F22 audit already maps chunks to transcripts),
  tallied per chunk.

**F32 (medium). RESULTS.md and figures/ are not among the study files.**
- Spec: "Study files: `SPEC_D1.md` …, `JOURNAL.md` (live), `RESULTS.md`, `figures/`, `handoff/`, `checks/`, an index saying
  what is live" (SPEC_D1.md:638–639).
- Evidence: neither is in `studies/already-clean/`. They live in the explain worktree's `bank/w5/`, which is not under git
  (S6 "`bank/` is not under git"), and in S3.
- What would satisfy it: copies in the study directory at the checked commit, or a dated amendment naming their location.

**F33 (medium). Three figures carry no null.**
- Spec: "null drawn on the panel" (SPEC_D1.md:628).
- Evidence: F3, F4 and F6 SVGs have no null line; S6.md:92 states this. F4's birth-order question has a defined null (the §6.4
  permutation band), which is reported in §11 but not drawn.
- What would satisfy it: the null drawn where one is defined (F4), and a dated note for F3 and F6.

**F34 (medium). The §13 read-at-source list was done after the unblind, not before the freeze.**
- Spec: "Before freeze, read at source for the specific claim, with the quote and its location recorded" (SPEC_D1.md:738–739).
- Evidence: §13's 2026-10-23 status says "The read-at-source list was not completed before the freeze …". §15 item 2 says
  "Closed at closeout 2026-10-23, after the freeze". This is disclosed and is not a requirement met as written.
- What would satisfy it: nothing can now. The disclosure is the record.

**F35 (medium). §15's text does not show that nothing is open.**
- Spec: §15 status line, "items 1, 2, 4 and 5 stay open; items 9–14 are new" (SPEC_D1.md:792–793).
- Evidence: items 4, 5, 9, 10, 11, 12 and 14 carry no "Closed" marker in §15, and the status line is unchanged. The S6 row says
  "items 1, 4, 5 closed earlier … nothing open". Item 5 is met in substance (`checks/CHECKERS.md` exists), and item 1 is closed
  by item 13. I found no record closing item 4 in SPEC, JOURNAL, handoffs or checks.
- What would satisfy it: a dated closure line on each of items 4, 5, 9–12 and 14, or the S6 row corrected.

**F36 (low). The companion-paper search scope and wording.**
- Spec: §13 item 6 (SPEC_D1.md:743–744).
- Evidence: `checks/literature_W5.md` §6 did not fetch the essay page or Goodfire's research index and did not query the
  co-author named in the essay's language-model section. Its line 259 says the rule "does not yet apply", while §15 item 16
  says "not applicable". My searches of those sources found no companion paper (section 4).
- What would satisfy it: the added sources in the §13 status record, and one wording.

## Verdict

The closeout acceptance against §9.4 and §10 does not hold as written. Two acceptance rows are refuted: the Q1 samples lack the
per-pole contexts the row names, and the PNG figures are cropped. One row is unclear (§15 "nothing open"). Several requirements
are met only in substance: the run logs were derived at closeout, NEW/REUSED is not labelled, the roster is not the first line,
model ids for manual calls are the requested ids, and RESULTS and figures sit outside the study files. The §11 grid matches the
sixth run's instrument output cell for cell, and every number in its note matches metrics.json. Flags raised: F23, F24, F25,
F26, F27, F28, F29, F30, F31, F32, F33, F34, F35, F36.
