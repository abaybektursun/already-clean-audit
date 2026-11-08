# Closeout re-check of the answers to F23–F36 (SPEC_D1 §9.6)

- Commits checked: drafts `c86e308c` (read in the detached worktree `interp_research-check-closeout2` at c86e308c; the drafts
  worktree HEAD is also c86e308c), explain `5a313c46` (HEAD, clean), harvest `bca491a9` (HEAD, clean). Data read in place:
  explain `bank/` (not under git: `bank/.gitignore:1` is `*`), harvest `bank/neurons/`, and
  `s3://pg-lineage-769063704696/results/`.
- Run: 2026-10-24T03:27:00Z to 2026-10-24T03:48Z (UTC, `date -u`). Model: Opus 5.5 (`claude-opus-5-5`).
- Read-only. The only file written outside the scratchpad is this report. Scratchpad `checker_closeout2/s3/`: six
  `events.jsonl` copies from S3 and one S3 key listing (`logs_listing.txt`). No edit, no S3 write, no git state change, no
  subagent. The drafts worktree has uncommitted edits to `handoff/judge_incident_log.txt` (+1 line) and
  `walkthrough/index.html` that are not in c86e308c. I read the committed versions.
- Verdict scale. RESOLVED AS WRITTEN: the spec sentence is met, and so is the flag's remedy. RESOLVED IN SUBSTANCE: the
  flag's remedy is in the record and I verified it, but the spec sentence is still not met word for word. OPEN: the remedy is
  missing or does not hold.

## Flags F23–F36

| flag | verdict | evidence |
|---|---|---|
| F23 Q1 samples lack contexts | RESOLVED IN SUBSTANCE | `bank/w5/verdict/samples/q1_{Q4B,G4,O7}_poles.requests_index.jsonl` exist. Each has 601 lines: a header (`derived: true`, written 2026-10-24T03:16:06Z, sources `bank/w2c/<cell>/explain/requests/c1-{explainer,second_reader}.jsonl`) and then 600 poles, in the same order as `q1_<cell>_poles.jsonl`. Each pole has 2 pointers. I checked all 3,600 pointers: the sha256 of the named line (newline stripped, as `write_sample_pointers.py` hashes it) equals the pointer's sha256 every time (0 mismatches). For a random draw (seed 20261024) I followed each pointer by id to its line: O7 `L00N2968-` (`c1-explainer.jsonl` line 8), Q4B `L21N5637-` (line 362), O7 `L14N9086+` (line 287), and the same lines in `c1-second_reader.jsonl`. Every sha256 matches. Each `user` field opens with the pole's header and holds 30 contexts (`#0`…`#29`). S6.md:91 says which file holds what. Spec: "Samples, one file per readout × arm, `out/samples/<readout>_<arm>.jsonl`: for Q1, per pole the 30 contexts as the explainer saw them". The sample file still holds no contexts. It reaches them through a derived pointer file. |
| F24 PNGs cropped | RESOLVED AS WRITTEN | `sips` gives F1 1840×1056, F2 1840×1056, F3 1840×1472, F4 1840×1145, F5 1840×960, F6 1840×2124. These match the SVG and PDF aspect ratios (820×470, 820×470, 900×720, 900×560, 920×480, 1000×1154; MediaBox = SVG × 0.75); F1/F2 come out 1.4 px taller than the exact ratio, with no visible effect. I viewed all six PNGs. F3: T_0–T_15 and the footer. F4: T_14 and T_15 bars in both panels, the null line and the footer. F5: the "H_SP if lower > 0.20" and "H_RG if upper < 0.10" labels. F6: complete down to its legend line and the footer "sha256 002108c3ea15da9f". F1/F2: legend text whole. All six SVG footers and all six decoded PDF content streams (ASCII85+Flate) carry `002108c3ea15da9f`; `shasum` of `verdict/metrics.json` is `002108c3ea15da9f7727…`. S6.md:86 and :103 and REGISTRY.md:5 name the sips-from-PDF method. |
| F25 stale statements | RESOLVED AS WRITTEN | (a) S6.md:95 says "the 20 banked runs", and `find bank/neurons` gives 20 `run.log` and 20 `events.jsonl`. JOURNAL.md:1377 still says 21; the journal is append-only, and JOURNAL.md:1401 corrects it. (b) RESULTS.md:131 and `runbook/results_hand_edits.py:59`: "bypassed five times, once per re-run (runs 2–6 …)". (c) INDEX.md:20 code row: `neurons/explain` 5a313c46, which is the explain HEAD. (d) S6.md:16 Spec line: run 6, metrics sha256 002108c3…. (e) S6.md:103 F5 title: "p_u = 0.265 [0.211, 0.321]: H_SP; 1238 late-born clusters on 600 poles", which matches the PNG title. (f) SPEC_D1.md:788: "B4 m-within-1 0.678 (closeout: 0.668)"; `metrics.json q2.B4_m_within_1` = 0.67833. |
| F26 NEW/REUSED | RESOLVED IN SUBSTANCE | S6.md:16–54: seven Record state items marked "NEW (S6)" and one marked "REUSED (from S2–S5)". JOURNAL.md:1405 says "REUSED (from S4/S5)", a small difference in wording. `handoff/S2-S4.md` and `S5.md` are unchanged and carry no NEW/REUSED label. Spec (SPEC_D1.md:605–606): "Each session ends at a wave boundary after writing `handoff/S<n>.md` with the NEW/REUSED record state". This is met for S6 only. Labelling of the judge data, see F40. |
| F27 roster line | RESOLVED AS WRITTEN | S6.md:1: "Roster, run/unrun/parked over the §9.4 cell roster: W0 4/0/0 of 4; W1 3/0/0 of 3; W2 4/0/0 of 4 scoring jobs; W3 2/0/0 of 2; W4–W5 0 GPU cells; 13 run, 0 unrun, 0 parked. One GPU run outside the roster: … `x/center_probe/1` (2026-10-23, $4.67)." The counts add up (4+3+4+2 = 13). |
| F28 run logs | RESOLVED IN SUBSTANCE | SPEC_D1.md:842–845 (§15 item 15) states the scope: GPU harvest and probe runs, 26 prefixes; explain, judge and verdict steps keep `calls.jsonl` and `metrics.json`. `calls.jsonl` rows carry `utc, model, backend, response_sha256`. All 26 S3 `events.jsonl` have LastModified 2026-10-24T03:17:13–03:21:27Z; all 20 banked copies have header `written_utc` 2026-10-24T03:16:50Z. S3 = bank (sha256) for w1/q4b/1, w1/g4/1, w3/grid/1b and x/center_probe/1. AWS lane `w1/q4b/1` now has 7 events, one of them `corpus_stage` (`read_seconds` 146.4, source `out/corpus/corpus/manifest.json`). `w0/verify/2` and `/3` also have one each (40.8 s, 38.6 s). Center probe: every event `source` names `mdl/mendeleev_neurons_x_center_probe/timings.tsv` (9) or `…/_rental_complete.json` (1), or an `out/…` manifest. Both `mdl/…` keys exist under `results/mdl/mendeleev_neurons_x_center_probe/`. run.log was not rewritten: the 26 S3 run.log LastModified are 2026-10-24T02:44:32–02:52:17Z. The ETag and size are equal to the job log for the three runs I checked: center probe `3bed95bc…` 1,112,847 = `results/mdl/…/job.log`; w1/q4b/1 `d9a86abe…` 6,543,743 = `job.log`; w3/grid/1b `f30943cb…` 1,163,033 = `vast/job.log`. Spec (SPEC_D1.md:625–626): "`events.jsonl`, `run.log`, compute per stage". These are still files derived at closeout, for harvest runs only. Residual defects: F37 (item 15's regeneration time), F38 (corpus events have no time). |
| F29 Q2 judge view per cluster | RESOLVED IN SUBSTANCE | `q2_clusters.requests_index.jsonl`: header plus 2,465 rows, in the same order as `q2_clusters.jsonl`. Every cluster has 2 clusterer pointers. There are 11,028 judge pointers, equal to the 11,028 judgments in `q2_clusters.jsonl`, over the 1,615 clusters that have judgments: w4j-judge-s0 5,475, s1 5,475, w4j2-judge-s0 39, s1 39. I checked all 15,958 pointers: 0 sha256 mismatches. For a random draw I followed every pointer of `L02N5291+:c3`, `L10N3963-:c5` and `L25N8888+:c0` to its line (2 clusterer and 6 judge each). Every sha256 matches. Each clusterer `user` field holds the pole and 50 contexts. Each judge `user` field has the sections "Exemplars of the job at the end of training:", "Record A:" and "Record B:". Spec (SPEC_D1.md:621–622): "for Q2, per cluster the exemplars, the two records as the judge saw them". This is reached through the pointer file, not held in the sample file. Where the pointed-to files are kept, see F39. |
| F30 registry hashes | RESOLVED IN SUBSTANCE | REGISTRY.md (bank/w5 = results/). Every figure row (lines 9–14) gives `metrics.json 002108c3ea15da9f`. Number rows give hashes, and `shasum` agrees: judgments 800fb2f774254ad5, clusters afa666e2b0bf2650, positions 965c88e55ad5f2d9, metrics 002108c3ea15da9f. The rows the journal names as added are present: b's interval and drift share (:27), v (:28), flag rate (:30), B4 and status counts (:32), the center-curve parts (:33) and Holm (:35). Residuals. Row :34 (G2) gives only the path `bank/w4/o7_t0_g2/explain/explanations.jsonl`, with no hash. "16 grid dirs (`q2.inputs.grid`)" are paths; `metrics.json q2.inputs.grid` is a list of 16 directory paths with no hashes. The F4 row's "numbers from" says `q2.births`, while the F4 footer reads `q2.births, q2.birth_order`. JOURNAL.md:1411's "every number row names judgments, clusters and positions hashes" is not literally true: the θ_s, gate, Q1, G2 and Holm rows use other inputs, and the p_u row names no positions hash. Spec (SPEC_D1.md:627): "every figure and number → generating script, input hashes". |
| F31 answering model | RESOLVED IN SUBSTANCE | `bank/w4/o7_s_v2/explain/manual_answering_models.jsonl` has 12,586 lines. Its header says: manual_calls 12,585, unattributed 5, attributed_by write 12,189 / brief+command 376 / sole brief 15 (sum 12,580), by role clusterer 1,083 + judge 11,497, all `claude-sonnet-5-5`. The (id, utc) pairs equal the 12,585 `backend: manual` rows of `calls.jsonl` (judge 11,497, clusterer 1,088). For 5 random rows (seed 424242: judge:w4j:6861, 9908, 1495, 10295, 10314) the named transcript exists under `~/.claude/projects/-Users-abaybektursun-projects-FuelOS/964d1f2d-…/subagents/`, and the named chunk file under the session scratchpad `manual/` holds that id with that utc. The assistant messages' `model` field counts equal the recorded ones (75, 76, 82, 82, 83, all `claude-sonnet-5-5`); I read no other field. One row each for "brief+command" (clusterer:w4c:933) and "sole brief" (judge:w4j:6977) also holds. The 5 unattributed rows are `old_trial/w4c_s1/chunk_000` at 2026-10-21T21:42:46Z. Each of the 5 ids has a later attributed row (`r6b/w4c_s1/chunk_000`, 2026-10-24T02:13:14Z). Spec (SPEC_D1.md:624–625): "`out/manifest.json` (… model id of every agent call)". The answering model is in a derived file beside `calls.jsonl`, not in `manifest.json`. The chunk files and transcripts it is read from are not named in the record's header and are not banked (F39). |
| F32 RESULTS and figures in study files | RESOLVED AS WRITTEN | `studies/already-clean/results/` holds `RESULTS.md`, `REGISTRY.md`, 6 svg, 6 pdf, 6 png and `README.md`. All 20 compared files are byte-identical to `bank/w5` (sha256). The pdf and png files are committed as git-lfs pointers (e.g. F4 png oid sha256 c851b827… = the file). INDEX.md:18 lists `results/`. INDEX.md:17 keeps `bank/w5` and S3 as the live results. S3 `results/neurons/w5/` has 34 objects, and every ETag equals the local MD5 (0 multipart). |
| F33 nulls on figures | RESOLVED IN SUBSTANCE | Explain 5a313c46 changes only `neurons_figures.py` f4 (+5/−1). The F4 SVG and PDF text has "null (§6.4 permutation band on b, main oldest): mean 0.414, 97.5th 0.449; observed b = 0.476 [0.436, 0.517] over 574 poles, above the band". `metrics.json q2.birth_order`: null_mean 0.41406, null_97_5 0.44948, b 0.47561, ci95 [0.43554, 0.51742], n_poles 574, supports_main_oldest true. `bank/w5_figures_run6_before_F33/F4_births.svg` lacks that line. The other five SVGs are byte-identical between the two sets. F3 and F6 have no null text; S6.md:103 says "no null defined for this descriptive panel" and "no null defined for a gallery". Spec (SPEC_D1.md:628): "null drawn on the panel". F4's null is a line of text under the histogram, not a mark on it. The births histogram does not plot b, so no mark on the panel could show the band. |
| F34 read-at-source after the freeze | RESOLVED IN SUBSTANCE (cannot be met as written) | SPEC_D1.md §13 "Status 2026-10-23 (closeout)": "The read-at-source list was not completed before the freeze". SPEC_D1.md:818 item 2: "Closed at closeout 2026-10-23, after the freeze". JOURNAL.md:1423 records it as disclosed. The spec's "Before freeze, read at source" is not met. The disclosure is the remedy the flag allowed. |
| F35 §15 nothing open | RESOLVED IN SUBSTANCE | SPEC_D1.md:794: "Status 2026-10-24 (closeout checker F35): every item is closed with a dated line below; nothing is open." Closure lines now exist on items 4 (:821, 2026-10-24), 5 (:825), 9 (:805), 10 (:807), 11 (:809), 12 (:811) and 14 (:816), plus 2, 3, 6, 7, 8, 13, 15 and 16. The cited evidence exists. Harvest `504fd4f5` (2026-10-19T16:54:21−10:00, "dedup, window position 0, drawn poles, second G0 failing input"). Explain `d281852c` (2026-10-19T17:02:41−10:00; its parent `cee153ed` is the commit titled "prompt v2, 256-token contexts, UNK-N/UNK-F, X3, X7", and JOURNAL.md:102 names d281852c as the explain side at W1). Drafts `f87f1b50` (2026-10-21T07:18:22−10:00; the first commit of `checks/CHECKERS.md`). JOURNAL 2026-10-20T03:15Z and 2026-10-21T17:37:11Z both exist. Every W1 harvest has `out/<cell>/poles/` (w1/q4b/1 q4b and c0, w1/g4/1, w1/o7_t15/1). Not literal: item 1 has no closure line of its own (item 13 closes it); items 10, 13 and 14 carry a wave label ("at W1", "at W4") and no date. Some closure lines say things the record contradicts (F37). |
| F36 companion-paper search | RESOLVED AS WRITTEN | SPEC_D1.md:759 (§13 "Status 2026-10-24") names the essay page, Goodfire's research index (newest post 2026-10-15) and the arXiv API by both authors, citing `checks/closeout_W5.md §4`. It says "The reading 'closed as not applicable' is the checker's too." §15 item 16 (:846) says "closed 2026-10-24 as not applicable". `checks/literature_W5.md:259` still says "does not yet apply". That file is a checker report and was not edited. |

Counts: 5 RESOLVED AS WRITTEN (F24, F25, F27, F32, F36), 9 RESOLVED IN SUBSTANCE (F23, F26, F28, F29, F30, F31, F33,
F34, F35), 0 OPEN.

## The two rows the first check refuted, run again

- **§10 samples' content, Q1 per pole** (S6.md:91, now "yes, through the pointer files"). Holds as now stated. All 1,800
  poles have explainer and second-reader pointers, and all 3,600 hashes equal their request lines. The three poles I drew
  hold 30 contexts in the explainer's `user` field. The class and flag counts in the row were verified by the first check and
  are unchanged: the sample files are dated 2026-10-23 16:29 local (run 6) and were not rewritten.
- **§10 figures as PNG** (S6.md:86). Holds. All six PNGs keep their figure's aspect ratio and show the whole figure: T_14
  and T_15 in F3/F4, the rule labels in F5, F6's footer, and F1/F2's legends. Every footer carries 002108c3ea15da9f.

## New flags (F37 onward)

**F37 (low). Four §15 closure lines say things the record contradicts.**
- Spec: §15 status (SPEC_D1.md:794), "every item is closed with a dated line below".
- Evidence:
  - Item 5 (:825), "Closed at W1: `checks/CHECKERS.md` (drafts f87f1b50, 2026-10-21)". W1 ran 2026-10-20T03:15–07:35Z
    (JOURNAL.md:100 launch, :153 runs B and C complete). `git log` for `checks/CHECKERS.md` shows its first commit as f87f1b50, 2026-10-21T17:18:22Z,
    after "W3 complete" (JOURNAL 2026-10-21T16:20Z).
  - Item 12 (:811) cites "`verify/VERIFY.md` §2" for the u×1.1 failing input. VERIFY.md (160 lines) has no "1.1", and its
    G0 row lists only the silu↔gelu_tanh swap. The u×1.1 result is in JOURNAL.md:103 and :109 (2026-10-20T03:15Z entry).
  - Item 14 (:816–817) says the judge lines were "all answered on the manual backend in one session (S5)". S6.md:3 is
    "Handoff after W4 on S and W5 (session S6)". S6.md:33 marks the judge data "NEW (S6)". `calls.jsonl` judge rows by utc
    date: 2026-10-22 9,166; 2026-10-23 1,868; 2026-10-24 463. S5.md's last commit (93fd5143, 2026-10-22T00:18:59Z) says
    "judge in progress".
  - Item 15 (:845) says "the corpus stage's compute is in the events since the 02:5xZ regeneration". The regeneration's
    `written_utc` is 2026-10-24T03:16:50Z, and the S3 objects are dated 03:17–03:21Z. S6.md:95 says "regenerated 03:1xZ".
- What would satisfy it: a dated correction of the four lines.

**F38 (low). Every `corpus_stage` event has no time, though its source has one.**
- Spec: SPEC_D1.md:625–626, "`events.jsonl`, `run.log`, compute per stage", together with item 15's "every event line … read
  from the named source".
- Evidence: in `w1/q4b/1`, `w0/verify/2` and `w0/verify/3`, the `corpus_stage` event has `"utc": null`, `"n_tokens": null`
  and `"sha256": null`. The source `out/corpus/corpus/manifest.json` has `created_utc` 2026-10-20T03:04:46Z (w1/q4b/1) and no
  keys named `n_tokens` or `sha256`. `write_run_logs.py` looks for `finished_utc`, `built_utc` or `utc` only. The event sorts
  first by an empty time.
- What would satisfy it: the event reading `created_utc`, with the two absent fields dropped, or a dated note.

**F39 (medium). What the new pointers and the model attribution point to has no copy outside the laptop.**
- Spec: no sentence requires an off-laptop copy. §10's samples requirement (SPEC_D1.md:619–622) now depends, through the
  pointer files, on the files below. The F31 attribution depends on the chunk files and the transcripts.
- Evidence. `bank/w2c/<cell>/explain/requests/` and `bank/w4/o7_s_v2/explain/`, which hold the requests, responses,
  `judgments.jsonl`, `clusters.jsonl`, `calls.jsonl` and `manual_answering_models.jsonl`, are untracked (`bank/.gitignore`
  `*`; `git ls-files bank` returns 0). A full listing of `s3://pg-lineage-769063704696/` (597,574 objects) has no key matching
  `c1-explainer`, `w4j-judge`, `o7_s_v2`, `/judgments.jsonl` or `manual_answering`. S3 holds no `results/neurons/w4/`.
  The chunk files are under `/private/tmp/claude-501/…/scratchpad/manual/`. The transcripts are under
  `~/.claude/projects/…/subagents/` (2,362 files). Neither location is named in `manual_answering_models.jsonl`, in the
  journal's F31 entry or in S6. `checks/W4-W5.md:15–17` names them.
- What would satisfy it: copies under the bucket's `results/neurons/`, or wherever the user chooses, named in INDEX.md, or a
  dated note that these inputs live only on the laptop.

**F40 (low). Statements left stale by the closeout.**
- Spec: §10 "an index saying what is live"; §9.4 record state.
- Evidence:
  - INDEX.md:13 says "one file per session, `S1.md` … `S6.md`". `handoff/` holds `S2-S4.md`, `S5.md`, `S6.md` and no
    `S1.md`.
  - INDEX.md:7 says "§11 filled 2026-10-23; §12 and §14 closeout records 2026-10-23". It does not mention the 2026-10-24
    refill from run 6 or the §13 and §15 addenda.
  - S6.md:12–13 says "the entries after it record the re-runs, the center-probe rental, the checker's answers and the fourth
    run". The entries now run to the sixth run and the closeout checker.
  - REGISTRY.md:12, the F4 row's "numbers from", is `q2.births`; the figure also reads `q2.birth_order` (see F30).
  - S6.md:33 marks the judge data "NEW (S6)", while 9,166 judge calls are dated 2026-10-22 and S5 was last committed
    2026-10-22T00:18:59Z with the judge in progress (see F37).
- What would satisfy it: a dated correction of each item.

## Verdict

Every flag F23–F36 has a remedy in the record that I opened and verified: 5 now meet the spec as written, 9 only in
substance (spec sentences that a derived file or a disclosure answers but cannot meet word for word), and none remain open.
Both rows refuted at 39260448 now hold. Four new flags are raised (F37–F40: one medium, three low).
