# NEURONRW — independent result review, 2026-10-26

Every headline number reproduces exactly from the raw judge answers and banks (Q4B Arm 0 D_w 0.175 [0.105, 0.250];
Q4B Arm 2 on-label rate 0.531 [0.425, 0.637]). Both headline labels exist only under post-freeze amendments. Both
effects are carried mostly by the poles whose label names one of their own top-20 output tokens. The missed-prediction
list leaves out every §11 miss, including the H_suff miss.

- Reviewer: Claude Opus 5.5, a subagent that wrote none of the study. Read order: handoff → SPEC (§13, §14.2) →
  JOURNAL → results → banks → code.
- Snapshot: worktree `interp_research-neuronrw`, branch `neurons/rw` at ca4fe721 (2026-10-26T18:13:49Z). Freeze tag
  `neuronrw-freeze-1` = 967a8db3 (05:06:42Z). Bank manifest `bank/rw/BANK_SHA256.txt` (sha256 6f232bea…): I re-hashed
  all 579 local files, 0 mismatches.
- Method: answers were parsed from `judge/responses/*.jsonl` and joined by id to `judge/keys/*.jsonl`. Samples came from
  `bank/rw/<cell>/out/`. Bootstraps use 10,000 resamples, `default_rng(20261025)` fresh per interval, items in
  `roles_<M>.json` order (the code's order). The scripts are reviewer scratch, not committed. Verdict files were opened
  only to compare.
- Status key: PASS = recomputed equal and the claim is worded right; FLAG = the number holds but the wording, a stamp
  or a code detail is wrong; FAIL = the claim is wrong as stated.

## 1. Claims, one line each

| # | claim | reported | recomputed | status | artifact and discrepancy |
|---|---|---|---|---|---|
| 1a | Arm 0 Q4B (re-explained labels, 100 ONE items) | A 0.180 [0.110, 0.260], Sh 0.005 [0, 0.015], Fx 0.020 [0, 0.050], D_w 0.175 [0.105, 0.250], D_f 0.160 [0.090, 0.240], δ_w 0.03 | identical (1 shuffled "yes" in 200) | PASS | — |
| 1b | Arm 0 G4 | A 0.200 [0.120, 0.280], Sh 0.015, Fx 0.010, D_w 0.185 [0.110, 0.265], D_f 0.190, δ_w 0.05 | identical (3 shuffled "yes" in 200) | PASS | — |
| 1c | Arm 0 O7 | A 0.190 [0.120, 0.270], Sh 0.065 [0.030, 0.105], Fx 0.050, D_w 0.125 [0.035, 0.215], D_f 0.140, δ_w 0.10 | identical | PASS | δ_w depends on item order at this seed: the same data in sorted pole order gives 0.090. No label moves. |
| 1d | label against δ_w | Q4B above 0.03; G4 above 0.05; O7 crosses 0.10 | same | PASS | — |
| 1e | blinding of Arm 0 and Arm 2 requests | no pole id or true-label marker | 5,572 requests scanned for `L\d+N\d+`, class names, model names, study name, dose words. Every hit is content: the token "UNK" in a token list, "Dani Olmo", "Samsung G4", a postcode "G4". Each yes/no call shows one label, so a call cannot compare candidates. Ids are opaque. fc3 truth letters are 40/40/40 per model. Candidate texts appear in A/B/C order: 2,000 of 2,000. | PASS | — |
| 1f | shuffle construction | shuf1, shuf2, fixed list: distinct ONE poles of the same tercile, never the pole or its neuron | 450 items: 0 defects. A label is reused at most 5 times. No shuffled label text equals the true one. | PASS | — |
| 1g | judged inputs equal banked ones | — | token lists equal the banked attested top-20: 1,650 of 1,650. Labels equal the stated label set: 1,650 re-explained, 300 with-vocabulary, 300 original. | PASS | — |
| 2a | B1 pooled | 0.013 [−0.010, 0.037], EQUIVALENT within ±0.05 | 0.0133 [−0.010, 0.0367], n 300 | PASS | 14 discordant pairs of 300 (9 up, 5 down). The A22 statistic (yn_true difference) replaces the frozen s-difference; recorded before any judge call. |
| 2b | H_ONE pooled | 0.050 [−0.033, 0.133], NOT SHOWN | 0.050 [−0.0333, 0.1333], 150 pairs | PASS | — |
| 3a | Arm 2 Q4B H_suff | o(32) 0.531 [0.425, 0.637]; o(0) 0.344 [0.262, 0.431]; paired 0.188 [0.056, 0.325] | 0.53125 [0.425, 0.6375]; 0.3438 [0.2623, 0.4313]; 0.1875 [0.05625, 0.325] | FLAG (code, minor) | `infra/pg/neuron_rw_analysis.py` `cmd_unblind2` passes `n_prompts0 = 5`, but only prompts 0 and 1 have α = 0 answers (A22). In 722 of 10,000 replicates every drawn α = 0 prompt is unjudged, and the code drops them (`verdict_arm2_Q4B.json` paired `n_finite` 9,278). Resampling the two judged prompts gives the same interval. No number moves. |
| 3b | frozen coherent-only reading | 0.487 [0.380, 0.600], paired 0.140 [0.007, 0.280], 75 poles | 0.4867 [0.380, 0.600], 0.140 [0.0067, 0.280], 75 of 80 | PASS | — |
| 3c | G4 Arm 2 (exploratory) | 0.525 [0.419, 0.631], paired 0.281, α 0 0.244 | 0.525 [0.4188, 0.6312], 0.2812 [0.1375, 0.4188], 0.2437 | PASS | — |
| 3d | chance level and decoys | chance 1/3 | Truth letters balanced. Decoys: 2 ONE poles of the same tercile, never the pole, its neuron or its matched partner; 360 triples, 0 defects. The triple and letter are the same at every dose and at α 0. Judge letter shares at α 0 (Q4B): A 0.400, B 0.242, C 0.358; B is inside 1/3 ± 0.1 by 0.009. | PASS | Draw seeds were not re-derived (make_plan not rerun). |
| 3e | G-judge2 | Q4B 1.00 / 0.396 / failing 0.3125; G4 1.00 / 0.375 / 0.225 | identical | PASS | The label-prompted continuations mostly copy the label text, so this positive control is weak. It does fire, and its failing input fails. |
| 3f | G-retest κ | Q4B 0.94 / 0.86; G4 0.81 / 0.88 | 0.940 (25 pairs, 24 agree) / 0.864 (71, 65); G4 0.810 / 0.883. Derangements 0.034, −0.089, 0.048, −0.008. Retest texts identical to the originals: 282 of 282 (A0 and A2). | PASS | — |
| 3g | judge saw the banked texts; instrument fired | — | Shown continuations equal the cell's banked completions: 2,000 of 2,000. Smallest per-cell median `realized_cos`: 0.9975 (α 2), 0.9994 (α 4), 1.0 (α 32). | PASS | — |
| 3h | Q4B Arm 2 G-coh fails only on label-prompted text | 0.880 < 0.90; natural 0.934; α 0 0.987 | band [0.3697, 4.9974]; identical shares | PASS | All 48 out-of-band label-prompted samples sit below the floor: they copy the label from the prompt. |
| 4a | every Arm 1 label CANCELLED under P8 | — | G-coh repeated-token input NLL 1.339 (Q4B) and 0.298 (G4) lies inside the freeze1 band, so the gate fails in both cells | PASS | — |
| 4b | React useEffect all-sample presence | 0.81 at 0, 0.53 / 0.52 at ∓8, 0.00 at ±32; control −16 0.12, 0 0.81, +16 0.37 | identical (control +16 0.365) | PASS | The matched control falls the same way (±32: 0.00 / 0.04), so the drop is generic degradation, not label-specific. |
| 4c | three Q4B swings | SQL 0.096 [0.020, 0.193]; Angular 0.397 [0.228, 0.577]; HTML 0.390 [0.193, 0.604] | identical. S_t +0.086 / +0.124 / +0.315; S_c −0.010 / −0.273 / −0.075; d* 64 | PASS | — |
| 4d | Arm 1 band quoted in JOURNAL 16:23Z and SPEC A27 | L 0.077 (Q4B), 0.064 (G4) | `results/thresholds_arm1_{Q4B,G4}.json`: L 0.0681, 0.0728 | FLAG (minor) | The quoted values are the jobs' preview bands, not freeze1's. The conclusion (repetition not flagged) is unchanged. |
| 5a | Q4B α_c 64; Liu lands | collapse ≤ 0.085 up to ±32 | collapse max 0.085 (−32), 0.03 (+32). α* 65.95, recomputed from `arm0/rms.json` (ratio 48.138). | PASS | 64 is the frozen value for "no dose up to 32 reaches 0.5", not a measured crossing. Liu's hit is one-sided (α_c > 32). |
| 5b | G4, O7 frozen 64, A24 fallback 32 | — | G4 collapse ≤ 0.425 (−16); O7 ≤ 0.05. Fallback = the one banked G-pc dose (G4 12.32 nats, O7 0.199 nats). α* G4 14.85. | PASS | The fallback measures no dose scale. On O7, 0.199 nats is 1.3% of the 15-nat ceiling. |
| 5c | Liu G4 "missed" | missed (64 against 16 or 32) | the frozen value is 64 | FLAG | `plan/dose_scale_G4.json`, `results/missed_predictions.md`. The frozen α_c comes from a sweep band whose G-coh failed on G4: the repeated-token input (NLL 0.555 ≥ L 0.2155) was not flagged, and natural spans scored 0.887. JOURNAL 15:46Z reads G4 at ±32 as "mostly not text" and ±16 as degraded, which is what Liu predicts. The miss should be printed with the stamp "collapse instrument failed G-coh on G4". |
| 6 | gates and amendments | — | see §2 and §3 | FLAG | A24 (1) and A27 frozen readings are missing from some files. A27 changed the study-level H_suff rule. |
| 7a | RE-EXPLAIN answers and model | 2,674, `anthropic/claude-sonnet-5-5` | 2,674 responses, 2,674 usage rows, 2,674 `calls.jsonl` rows, every one that model; attempts 1; finish stop; 0 declined | PASS | — |
| 7b | ONE retention | 329/358, 306/338, 292/311 | identical | PASS | — |
| 7c | hashes against manifests | — | Six request files: sha256 and id lists equal `plan/reexplain_<M>.json`. The rx batches contain 0 vocabulary lines and 0 system clauses; the rxv batches contain 1,007 of each. First call 05:10:16Z, after the freeze. | PASS | — |
| 8a | judge spend | $22.50, 5,574 calls | 8,811,505 in + 487,214 out = $22.50. J0 $0.23, A0 $3.90, J2 $0.86, A2 $17.50. 5,574 usage rows = 5,572 requests + 2 re-asks of the content-filtered guess. | PASS | The handoff prints A0 as $3.91 (rounding). |
| 8b | RX spend | $93.04 | 41,909,707 in + 921,711 out = $93.04 | PASS | — |
| 8c | instance-hours | 5.99 over 13 instances | Roster launch→STATUS + 0.08 h each sums to 5.994. `metrics.json` start and update times sit inside every launch→STATUS window. | PASS | Not checked against AWS records. The 0.08 h shutdown is a modelled constant, not a measurement. |
| 9 | sample check | JOURNAL 16:23Z: "α 32 mostly fluent" | §4: about half of 10 random judged dose-32 cells are repetition loops | FLAG | The JOURNAL entry read one sample (prompt 0, k 0). |
| 10 | not a relabel of known work | SPEC §1 names Arad 2025 as closest | §5 | FLAG | The novelty claim must be narrowed. |
| 11 | banned words | — | 0 hits in SPEC.md, JOURNAL.md, the handoff, `results/*.md` | PASS | — |
| E1 | "11 of 18 predictions missed" | `results/missed_predictions.md` | the file covers §5.6 and the two Liu rows only | FAIL | SPEC §5.6 requires "every prediction in this section and in §11". Missing §11 misses: H_suff (predicted NOT SHOWN at 0.33–0.42; observed SUPPORTED 0.531); D1 X2 ("predicts H_suff is not SUPPORTED"); B1 (predicted SUPPORTED, L 0.02–0.10; observed EQUIVALENT 0.013); Arm 0 absolute rates (predicted A 0.20–0.50, Sh and Fx 0.05–0.25; observed Q4B A 0.18, Sh 0.005, Fx 0.02, G4 Sh 0.015, Fx 0.01, O7 A 0.19); H_depth Arm 0 (predicted SUPPORTED; observed MIXED); gates pass in every job (G-known failed in `arm0_o7/1` and `arm1_q4b/1`); α 0 Arm 2 rate 0.25–0.42 (G4 ONE poles 0.244). Not scorable: fc3 accuracy (not measured), H_repair and H_repair_ONE (CANCELLED), the TeX row (slot dropped). The handoff repeats "11 of 18". |
| E2 | stamps (§7.7) | — | — | FLAG | `results/grid.md` and the G4 `rows.H_write.label` in `verdict_arm0_G4.json` lack the frozen G-proj CANCELLED reading that A24 (1) promises; it sits only in the `g_proj.frozen` block and the handoff. `results/study_arm2.json` prints "SUPPORTED (decided on Q4B alone, A27)" without the coherence stamp and without the frozen study reading. |

## 2. Amendments A20–A27 (commit times from `git log`, UTC)

The first judge call was at 13:44:54Z (J0). The first Arm 2 judge call was at 16:54:19Z (J2).

| amendment | recorded | seen before it | judge outcomes on disk before it | frozen reading printed beside | moved a threshold or rule? |
|---|---|---|---|---|---|
| A20, A21 | d1512023, 09:46:31Z | `pre_q4b/1` gates and crash | none | yes (handoff: every Arm 0 job raises; every −d* Arm 1 cell reads "not shown to fire") | no (instrument fixes) |
| A22 | 20f2be95, 10:12:40Z | preflight only | none | yes (handoff) | n cut from 24,393 to 6,908 calls; B1 statistic redefined; G-n line 300 → 140. Recorded before any judge call; no margin moved. |
| A23 | d1b2fd68, 11:12:21Z | `pre_q4b/2` | none | yes (`grid.md` repair rows) | no; self-repair is CANCELLED either way |
| A24 | ccefadd3, 13:18:19Z | Arm 0 GPU cells (G4 G-proj 0.880, O7 G-known room 0.114, G4 G-coh) | none | partly: dose scale and the O7 layer-0 reading yes; G4 G-proj only in the `g_proj` block and the handoff (E2) | A24 (1) moves G4 Arm 0 from CANCELLED to counted; A24 (2) lets O7 into Arm 0. Both are instrument fixes recorded before any judge call. |
| A25 | ddab18fd, 14:14:50Z | freeze0 and the room-check failure | yes: all A0 answers, true-label ones included, committed at 1632fa5d (13:53:37Z) and 8b816035 (14:08:57Z). unblind0 followed at 9094d023 (14:30:16Z). | yes (every label carries the constant-plant reading) | It changes the readability rule, not δ_w or the H_write rule. Its cause (shuffled yes-rates 0.005 / 0.015 / 0.065) is visible from control data alone. That no true-label answer was read rests on the JOURNAL's word; no artifact can show it. |
| A26 | 9c329a52, 15:23:05Z | Arm 0 unblinded; `arm1_q4b/1` G-known 0.00336 | Arm 0 only | yes (truncated-window reading in `controls.json`) | no |
| A27 | 3e1a65d4, 16:25:52Z | Q4B Arm 2 job preview: label-prompted G-coh 0.745 (70a4fe94, 16:23:10Z); Arm 1 G-coh failures; G4 and O7 ±32 sample checks; Q4B α 32 samples | none for Arm 2 (J2 at 16:54:19Z, A2 at 16:55:58Z); no Arm 1 regex counts (freeze1 at 16:52:31Z) | partly: `verdict_arm2_Q4B.json → frozen_coherent_only` and the handoff yes; `study_arm2.json` and the `grid.md` study row no | **Yes.** (2) replaces the frozen §7.1 study rule. Under that rule one counted model reads "MIXED (1 counted models)" (`study_label`); A27 has Q4B alone decide. (3) moves Q4B H_suff from the frozen CANCELLED (P8) to a headline. (4)'s "too loose" test names the natural spans and the α 0 half, and leaves out the label-prompted input that the preview had already shown failing. Mitigation: that input fails because the texts copy the label (all 48 failures are below the floor), and the frozen coherent-only reading also gives SUPPORTED, 0.487 [0.380, 0.600]. |

Under the unamended rules neither headline exists. H_write: G4 is CANCELLED by G-proj, O7 leaves after two G-known
aborts, Q4B reads NOT READABLE by the constant-plant room check, so the study label would be MIXED (1 counted model).
H_suff: CANCELLED by P8. No amendment was written after a judge outcome of the arm it changed, and no margin, δ_w or
interval rule moved. The amendment that most changed a headline is A27.

## 3. Gates, every cell

Read from `bank/rw/<cell>/out/controls.json` and `metrics.json`. In all 13 cells G-det, G-ident, G-place, G-gen,
G-floor-dir, G-floor-pos and G-pc passed, and every registered failing input failed. G-known failed twice: in
`arm0_o7/1` its failing input lacked room (shift 0.114), which made the gate fail as registered; in `arm1_q4b/1` the
ΔNLL gap was 0.00336. In both the cell aborted as specified. G-coh passed in `pre_q4b/2`, `pre_q4b/3` and
`arm0_q4b/1`; `pre_q4b/1`, `arm0_o7/1` and `arm1_q4b/1` stopped before it. It failed in the other seven cells: G4 and
O7 Arm 0, both Arm 1 cells, and all three Arm 2 previews. In six of those seven the repeated-token input was not
flagged; Q4B Arm 2 was the exception. Analysis gates:
- G-proj: 453/454 (Q4B), 411/411 (O7). G4 reads 0.880 under the frozen rule and passes under A24.
- G-judge0: 0.94 / 1.00 / 0.88; shuffle 0.007 / 0.013 / 0.057; failing input 0.02 / 0.04 / 0.04. Recomputed.
- G-judge2, G-retest: rows 3e and 3f.
- G-n: 150 ≥ 140 and 80 ≥ 56.

Two failing inputs always fail by construction, so they test nothing: G-n's "need − 1" input and G-pc's α = 0 input
(exactly 0). Both are as specified. The A24 G-proj failing input on the Q4B and G4 banks used "the next audited pole of another neuron"
instead of the registered neighbouring column (JOURNAL 13:45Z), because those banks predate A24.

Each verdict reads its gates. unblind0 reads the model gates, G-proj, G-judge0, G-n and the retest stamp. unblind2
reads the model gates, G-judge2, G-coh (stamp), G-n, `realized_cos` and the retest stamp. study2 reads A27 (4).
Self-repair was not recomputed.

## 4. Samples opened

Q4B Arm 2, dose 32, ten fc3 cells drawn at random (seed 7): 6 judged right, 4 wrong.
- Right, label visible: L26N3166- ("anyone"): "Imagine anyone could make anyone do anything. Anyone. Anyone, said
  anyone." This is a repetition loop of the labelled token.
- Right, label visible: L31N9536- ("a" in idioms): "A week ago a a storm a big a a a a a a". Same pattern.
- Right, weakly visible: L19N5468- (category noun after a proper noun): "Python module is a subset of JavaScript …
  Python: Python:".
- Right, label not visible: L05N8095- (Estonia "ston"): "the results of the Imerso survey", "In the October
  elections".
- Right, label not visible: L07N1025+ ("To" purpose clause).
- Right, label not visible: L03N1314+ (repeated identifier). Its α 0 call was also right.
- Wrong: L20N9552+ ("the, The, T, whom, who"; the judge chose a Sudoku-digits decoy).
- Wrong: L19N9685+ ("Revision: 3.44.3794; 1.8.3301").
- Wrong: L25N5080+ (Indian PIN digits; text "ISLINGTON-1201902: 12-13 February 2010"; borderline).
- Wrong: L07N7370+ ("This is a test message."; a correct miss).

So in 3 of the 6 right cells the text shows the label, and all 3 are repetition of a token the label names. Over all
Q4B ONE cells at α 32, 29% have a mean distinct-bigram ratio below 0.5 (0% at α 2). The coherence band marks 4.4% of
α 32 samples collapsed. This bigram metric and its 0.5 cut are the reviewer's, chosen after seeing the data.

Arm 0, three items:
- Q4B L31N9536-, judged yes. Tokens `" a" "-a" "a" "_a" "*a"`; label "the word 'a' … in fixed idioms". It agrees by
  token identity.
- Q4B L10N8041+, judged no. Tokens `"arma" "惜" "煮" " distinct"`; label about head nouns of proper names. The "no" is
  right.
- O7 L02N91+, judged yes. Tokens `"esser" "ulp" "iges" "peon"`; label "a middle word-piece inside a longer proper name".
  This is a generic label: its `yn_fixed` (another pole's list) was also judged yes.

## 5. Literature

No web search was run; the brief allowed this. Claims were checked only as recorded in `studies/neuronoff/LIT_steering.md`
and SPEC §1 and §5.3, not against the papers.
- Arm 0 is not a relabel, but its carrier is known. Arad 2025's input score counts top-activating tokens that appear
  among the top logit-lens tokens. Here the agreement concentrates in exactly that token-identity overlap (§6). The new
  part is the object (MLP neuron poles in three base-model families) and the read side (natural-language labels written
  without the output tokens).
- Arm 2 has no prior large-sample test of single-neuron steering by input label in base models (LIT §D). Gur-Arieh
  2025's 36.9% against 33.3% used a different protocol: the judge picks the steered text set for a description, at KL
  0.25–0.5, with 25-token outputs. LIT §C2 and §C3 predicted the pattern seen here: on-label effects from poles whose
  top projected tokens overlap the label, and visible change first as repetition of those tokens. The late-layer part
  of that prediction is not supported (H_depth Arm 2 NOT SHOWN).
- Liu 2026's α* reproduces from the banked RMS: 65.95 (Q4B) and 14.85 (G4).

## 6. Findings not in the report (exploratory, reviewer-made splits)

1. **H_write sits in the literal items.** These are items where the re-explained label names a token of the pole's own
   judged top-20, by the frozen `literal_overlap` rule; recomputed from the request texts.
   - D_w on literal items: Q4B 0.553 [0.342, 0.763] (19 items), G4 0.618 [0.353, 0.853] (17), O7 0.556 [0.333, 0.778]
     (18).
   - D_w on the rest: Q4B 0.086 [0.037, 0.148], G4 0.096 [0.036, 0.163], O7 0.030 [−0.055, 0.116].
   - These rows are in the verdict files (`secondary.literal_overlap`) and absent from the handoff.
2. **Q4B H_suff sits in the same subset.**
   - 17 literal poles: o(32) 0.765 against o(0) 0.265; pole-level paired gain 0.50 [0.29, 0.71].
   - 63 non-literal poles: 0.468 against 0.365; gain 0.103 [−0.032, 0.238], which crosses 0.
   - G4 (exploratory dose): literal gain 0.50, non-literal 0.227 [0.086, 0.359].
   - The literal split predicts steering success more sharply than the frozen agree/disagree split (R 1.52 [1.00, 2.22]).
   - The gain is present in both repetitive and non-repetitive dose-32 cells: +0.22 and +0.17 against the same poles'
     α 0 rate.
3. Repetition at d_p 32 is about 29% of Q4B cells (§4). The coherence band does not see it.

Not mined: the by-sign and by-explainer rows, the free-text guesses (`judge_a2-guess-*`), the O7 Arm 2 samples, and the
self-repair banks.

## 7. Steps not done

- No web search.
- Self-repair numbers not recomputed.
- make_plan not rerun: the subsample, decoy and draw seeds were not re-derived.
- §7.5 room-check simulations not re-run (R-pass and R-null were read from the thresholds files).
- G-proj and G-repair-fit not recomputed from projections.
- AWS instance records not queried.

## 8. Reviewer's own errors

- My first Arm 0 pass bootstrapped items in sorted pole order and gave intervals off by one grid step (O7 δ_w 0.090).
  Re-run in the code's order, everything matched. The size of the shift is recorded at 1c.
- My first bank-hash check mis-split the manifest lines and reported 579 missing files. Corrected: 0 missing, 0
  mismatches.
- One decoy sub-check iterated the wrong structure. I redid it on `sweep.poles`: 0 drawn poles in the sweep or among
  the Arm 1 targets.

No other reviewer covered this corpus.

## 9. Verdict

- **Numbers that survive, all recomputed exactly:**
  - Arm 0 A, Sh, Fx, D_w, D_f and δ_w for three models; B1 pooled EQUIVALENT; H_ONE pooled NOT SHOWN.
  - Arm 2 Q4B o(32), o(0), the paired difference and the coherent-only reading; G4 exploratory rows.
  - G-judge0, G-judge2, the G-retest kappas, Arm 1 curves and swings, the dose scales.
  - RE-EXPLAIN counts, models and hashes; spend; bank hashes; blinding.
- **Need a wording change:**
  - H_write SUPPORTED: say it holds only under A24 and A25, and print the frozen result (no readable study label).
    Report the absolute rate (0.18–0.20). Say the effect is carried by the 17–19% of poles whose label names a top-20
    output token; on the rest D_w is 0.03–0.10, and O7 crosses 0.
  - H_suff SUPPORTED: say "Q4B only, under A27; frozen: CANCELLED (P8) and MIXED (1 counted model) under §7.1". Say it
    is carried by the 17 literal poles; non-literal gain 0.10 [−0.03, 0.24]. Say about 29% of dose-32 cells are
    repetition loops. The §1 decision sentence "the next neuron study can steer by label" does not follow beyond
    token-copy poles.
  - Liu G4 "missed": add the collapse instrument's G-coh failure stamp.
  - SPEC A27's claim "no hypothesis threshold moves" is literally true, but A27 changed the study aggregation rule.
  - Stamps (E2); the 16:23Z sample-check sentence; the A0 cost rounding; the Arm 1 band values (4d); the α 0 prompt
    slots in the paired bootstrap (3a).
- **Wrong:** "11 of 18 missed" (`results/missed_predictions.md`, handoff). It is a count over a partial list. The §11
  misses must be added, H_suff's first among them.

## 10. Cheapest decisive next check

The O7 Arm 2 samples are banked and not judged (about $9.55). Before judging, register the prediction that the H_suff
gain is confined to literal poles, with this refuting observation: a non-literal pole-level paired gain whose interval
lies above 0. Then judge O7 once. A literal gain at least 0.30 above the non-literal gain would confirm the token-copy
reading on a third model.

---

## Addendum 2026-10-26 (second pass, `neurons/rw` at dbebba9c)

Verdict: the closeout can stand once two FAIL items and the wording FLAGs below are fixed. Every number I recomputed in
this pass matches its artifact. The two FAILs are statements of the frozen reading:
- the frozen B1 label, which is NOT SHOWN, not EQUIVALENT;
- the frozen missed count, which is 16, not 15: the H_write row is missing.

- Snapshot: commits 550cba04 (A28, 18:40:05Z) to dbebba9c (19:40:58Z).
- Same method as the first pass: answers parsed from `judge/responses/`, joined to `judge/keys/`; samples from
  `bank/rw/`; scripts in reviewer scratch.
- A28 timing: A28 was recorded at 18:40:05Z. The first O7 Arm 2 judge call was at 19:14:19Z
  (`judge/responses/a2-fc3-O7.usage.jsonl`). The O7 literal set computed from the banked projections and re-explained
  labels equals A28's registered 16 poles exactly.

### A. Recomputed items

| # | item | reported | recomputed | status | discrepancy and fix |
|---|---|---|---|---|---|
| A1 | O7 Arm 2 H_suff | o(16) 0.450 [0.350, 0.550]; paired 0.156 [0.044, 0.263] | 0.450 [0.350, 0.5502]; 0.15625 [0.04375, 0.2625] with the code's 5 α 0 slots | PASS | Also checked: 1,000/1,000 shown texts equal the banked cells; candidates in A/B/C order; min cell median `realized_cos` 0.9977; 0 pole ids in requests. Per dose: α 0 0.294, α 2 0.325, α 4 0.331. |
| A2 | O7 G-judge2, κ | 1.00 / 0.325 / failing 0.275; κ 0.92 / 0.80 | identical; κ 0.921 (22 pairs) / 0.796 (74); derangements −0.10 / 0.021 | PASS | — |
| A3 | P-lit on O7 | literal 0.250 [0.031, 0.500] two-way, [0.094, 0.438] pole-only; non-literal 0.133 [0.016, 0.250] / [0.039, 0.227]; "not confirmed, not refuted" | identical | PASS (numbers, reading) | See C3 on how much room the test had. |
| A4 | literal split reproduces the first pass on Q4B and G4 | `results/literal_split_{Q4B,G4}.json` | Arm 0 and pole-only Arm 2 identical. Two-way equal in the code's sorted pole order: Q4B literal 0.500 [0.294, 0.706], non-literal 0.103 [−0.048, 0.262]; G4 0.500 [0.188, 0.750], 0.227 [0.063, 0.391]. | PASS | — |
| A5 | repetition at d_p | Q4B 29.3% vs band 4.1%; G4 48.3% vs 28.4%; O7 7.0% vs 2.3% | 0.2925 / 0.0408, 0.4825 / 0.2838, 0.07 / 0.0228 (ONE cells; ONE samples for the band) | PASS | My first pass's 4.4% was the band share over all poles (0.0438); 4.1% is over ONE poles. A28 (3) quotes 4.4%; state the population beside it. |
| A6 | missed-prediction arithmetic | 15 of 25 frozen, 17 of 25 amended; seven §11 rows | Counted from `results/missed_predictions.md`: 15 frozen MISSED, 2 not scorable (Liu G4, H_suff), 17 amended. All seven §11 rows from the first pass are present. | PASS | See C1 for what the list still leaves out. |
| A7 | G4 Arm 0 verdict re-render | no number changed | Diffing `verdict_arm0_G4_preA28.json` against `verdict_arm0_G4.json`: only five `rows.*.label` strings (frozen G-proj stamp appended) and `created_utc` differ | PASS | Minor: the labels under `secondary` (literal_overlap, by_sign, by_explainer, …) carry no frozen G-proj stamp (§7.7). Fix: stamp them, or state at the top of the file that every G4 Arm 0 row is CANCELLED under the frozen G-proj. |
| A8 | judge spend | $31.19 over 6,910 calls | 12,086,650 in + 701,593 out = $31.19. 6,910 usage rows = 6,908 requests + 2 re-asks. O7 A2 $8.69 (1,336 calls), under the $48 stop. | PASS | — |
| A9 | sample quotes in RESULTS §8 | L26N3166- prompt 1 k 0; L07N7370+ prompt 0 k 0; Arm 0 token lists and labels | All four match the banks. L26N3166- is right at prompts 0 and 1; L07N7370+ is wrong at both. | PASS | Optional: L26N3166-'s α 0 call at prompt 1 was also right (B), so that cell is not a gain over α 0. Say so beside the quote. |

### B. FAIL items

| # | where | discrepancy | exact fix |
|---|---|---|---|
| B1 | `RESULTS.md` §1 table, rows "H_ONE pooled" and "B1 pooled" (column "frozen rule (decides)"); §7; `CLOSEOUT.md` §1 item 5; `results/study_arm0.json → frozen_rule` (no pooled rows); `results/missed_predictions.md` B1 row, frozen column ("the frozen room check changes no label here") | Under the frozen rules, G4's Arm 0 labels are CANCELLED (exact-order G-proj) and O7 has no Arm 0 data (two G-known aborts). `cmd_study0` pools counted models only, so the frozen pooled rows are Q4B alone. Frozen B1 = 0.020 [−0.020, 0.060] over 100 items: NOT SHOWN, because the interval is not inside ±0.05. Frozen H_ONE = 0.020 [−0.110, 0.150] over 50 pairs: NOT SHOWN. EQUIVALENT 0.013 and 0.050 over 150 pairs are amended readings (A24). | (1) RESULTS §1: frozen column "B1: NOT SHOWN, Q4B alone, L 0.020 [−0.020, 0.060] (G4 and O7 not counted)" with amended "EQUIVALENT, L 0.013 [−0.010, 0.037], 3 models (A24)"; same for H_ONE (frozen 0.020 [−0.110, 0.150], 50 pairs; amended 0.050 [−0.033, 0.133], 150 pairs). (2) RESULTS §7 and CLOSEOUT §1.5: prefix "Under the amendments (A24)" and give the frozen reading. (3) Add frozen pooled B1 and H_ONE to `study_arm0.json → frozen_rule`. (4) missed_predictions.md B1 frozen column: "NOT SHOWN, Q4B alone, L 0.020 [−0.020, 0.060]"; the result is still MISSED. |
| B2 | `results/missed_predictions.md` header; `RESULTS.md` §1 and §9; `CLOSEOUT.md` §2.7; `README.md` row "every §5.6 and §11 prediction against its outcome" | The §11 row "H_write, study label: SUPPORTED" is absent. Under the frozen rules it reads MIXED (1 counted models), so it is a frozen miss and an amended hit. The frozen count is 16 missed, not 15. Also absent, though their frozen or amended outcome is not a hit: H_pred_agree, H_pred_conf and H_depth Arm 2 (predicted NOT SHOWN; frozen CANCELLED under P8; amended Q4B hit), H_repair and H_repair_ONE (CANCELLED), fc3 accuracy (not measured), the TeX target and its control (slot dropped). "25" counts listed rows, not predictions. | Add the H_write row (frozen MISSED, amended hit) and the not-scorable rows above. Print the counts as "frozen: 16 missed, 10 not scorable, N hit, of T predictions; amended: 17 missed, 6 not scorable, …", with T = every row of §5.6 and §11. Change README's "every … prediction" only once the file lists every prediction. |

### C. FLAG items

| # | where | discrepancy | exact fix |
|---|---|---|---|
| C1 | `RESULTS.md` opening paragraph: "so the effect is mostly token copying but not only" | Three problems. (a) "Token copying" names a mechanism; what was measured is a label that names one of the pole's own top-20 output tokens. (b) The literal poles carry most of the summed gain only in Arm 0 (60%, 57%, 80% of summed D_w) and in Q4B's Arm 2 (57%). On G4 and O7 the non-literal poles carry 64% and 68% of the summed Arm 2 gain. (c) "Not only" is shown on G4 and O7 (non-literal 0.227 [0.063, 0.391], 0.133 [0.016, 0.250]) and not on Q4B, the one counted Arm 2 model (0.103 [−0.048, 0.262]). | Replace with: "Arm 0's agreement sits mostly (57–80% of summed D_w) in the 17–19% of items whose label names one of the pole's own top-20 output tokens. In Arm 2 those poles gain more per pole on every model (0.50, 0.50, 0.25 against 0.10, 0.23, 0.13). They carry most of the summed gain on Q4B (57%), and not on G4 (36%) or O7 (32%), where the other poles' gain is above 0." |
| C2 | `RESULTS.md` §3, last paragraph: "P-lit is not confirmed. What it means: the effect is carried mostly, but not only, by poles whose label names their own output token." | This inverts the held-out result. P-lit failed because the literal gain on O7 (0.250) fell short of 0.30, and O7's literal advantage is 0.12 (0.250 − 0.133). The held-out model does not show the literal concentration. | Replace with: "P-lit is not confirmed: on the held-out model the literal poles gained 0.250 against 0.133 for the rest, and the rest carry 68% of the summed gain. The literal concentration seen in sample on Q4B is not replicated on O7." |
| C3 | `RESULTS.md` §3 ("the refuting observation did not occur"); `results/literal_split_{Q4B,G4}.json → arm2.p_lit.reading` | The refuting branch (non-literal two-way lower end > 0.15) needed a non-literal gain of about 0.27 at O7's half-width (about 0.12). G4's non-literal 0.227 would not have refuted it, and the code reads G4 as "P-lit confirmed". The Q4B and G4 files print "P-lit confirmed" on the data P-lit was built from. | RESULTS §3: add "the refuting observation had little room: it needed a non-literal gain near 0.27; G4's 0.227 would not have fired it." In the Q4B and G4 split files, print "in sample (P-lit was built from these data)" instead of a confirm reading. |
| C4 | `CLOSEOUT.md` §2 item 3 | It cites P-lit's non-confirmation as support for "not established beyond token-copy poles". It is evidence against the token-copy concentration on O7. G4 and O7 show non-literal gains above 0. | Replace with: "That steering by label works on poles whose label names no output token: not shown on Q4B, the counted model (0.103 [−0.048, 0.262]). On G4 and O7, whose dose scale is not validated, the non-literal gain is above 0 (0.227 [0.063, 0.391], 0.133 [0.016, 0.250]), and P-lit was not confirmed on O7." Use "literal poles" for "token-copy poles". |
| C5 | `RESULTS.md` §4: "in 3 of 6 correct Q4B dose-32 cells … a repetition loop of a token the label names (section 9)" | Two errors. (a) This is my own first-pass error: L19N5468- repeats "Python:" and "Python module". That shows the label's pattern (a category noun after a proper noun), not a token the label names. (b) The samples are section 8, not 9. | Replace with: "in 2 of 6 correct cells (L26N3166- 'anyone', L31N9536- 'a') the text shows the label as a repetition loop of a token the label names; in a third (L19N5468-) the label's pattern appears inside a repetition loop (section 8)." |
| C6 | `RESULTS.md` §2 table, column "label (amended)"; §2 text "D_w is close to A on every model"; §7 rows H_depth and H_pred_*; §4 label cells for G4 and O7 | (a) The §2 column prints the amended label first. (b) On O7, D_w is 0.125 against A 0.190. (c) The §7 per-model rows are amended readings without a frozen label: H_depth frozen is Q4B SUPPORTED NOT READABLE and G4/O7 CANCELLED; the Arm 2 rows are CANCELLED under P8. (d) The G4 and O7 H_suff frozen reading (CANCELLED, P8) is not printed. | (a) Column "label: frozen; amended", frozen first. (b) "on Q4B and G4 (0.175 against 0.180, 0.185 against 0.200); on O7 Sh is 0.065, so D_w is 0.125 against A 0.190". (c) Prefix §7's per-model rows "amended:" and give the frozen label. (d) Add "frozen CANCELLED (P8)" to the G4 and O7 cells. |
| C7 | `RESULTS.md` §9 Novel: "Arm 2 is a larger base-model test … than Gur-Arieh et al. 2025's near-chance result" | Gur-Arieh used Llama-3.1-8B-Instruct, not a base model. Their number of neurons is not recorded in `LIT_steering.md`, so "larger" is unverified. The Arm 0 sentence against Arad 2025 is correct. | Replace with: "Arm 2 tests steering single neurons by input label in base models (80 poles per model), which `LIT_steering.md` §D lists as not done before. Gur-Arieh et al. 2025, on an instruct model with a different protocol, found input-based labels near chance (36.9% against 33.3%). LIT §C2–C3 predicted the overlap pattern seen here." |
| C8 | `results/grid.md` | The file was appended, not regenerated. Superseded rows still stand above the new frozen-first block: line 18 study H_write "H_WRITE", line 40 study H_suff "SUPPORTED (decided on Q4B alone, A27)" without the coherence stamp, line 8 the G4 H_write row without the frozen G-proj stamp, line 42 "O7 … not judged" directly above O7's judged rows. Line 64, the study H_depth Arm 2 row, has no frozen reading (CANCELLED, P8). `study_arm2.json → H_depth` likewise. | Regenerate `grid.md` with current rows only, keeping the old file as `grid_preA28.md` (as was done for the study files), or mark lines 8, 18–19 and 40–42 "superseded by lines 47–64". Add "frozen: CANCELLED (P8)" to the study H_depth Arm 2 row in `grid.md` and `study_arm2.json`. |
| C9 | `RESULTS.md` §10 (amendments table) | The table gives what each amendment changed. It does not give when each was recorded relative to what had been seen: A25 came after all A0 answers were committed; A27 came after the Q4B preview G-coh reading and before any Arm 2 judge call; A28 came after this review and before any O7 call. | Add a column "recorded (commit, UTC) / visible before it", from §2 of this review plus A28 (550cba04, 18:40:05Z; no O7 Arm 2 answer existed). |
| C10 | `RESULTS.md` §11 row "3a, 4d … no number moves"; `results/study_arm2.json → Q4B_H_suff.frozen_coherent_only.paired_minus_alpha0.n_finite` 9,227 | The 5-slot α 0 resampling is still used in every unblind2 interval. On O7, resampling only the two judged prompts gives paired [0.050, 0.2625] against the printed [0.044, 0.263]. No label moves. | Change "no number moves" to "no label moves (O7 paired lower end 0.050 with the two judged prompts against 0.044 printed)". |
| C11 | root `README.md` bullet; study `README.md` status line: "the amended readings sit mostly in poles whose label names their own output token" | True for Arm 0 and for Q4B's Arm 2 (57%). The held-out O7 test did not confirm it. | Append: "; the held-out test on O7 did not confirm this". |

### D. Dispositions of the first pass

| first-pass item | disposition at dbebba9c | status |
|---|---|---|
| E1 missed list | regenerated with the §11 rows | partly: see B2 |
| E2 stamps | added to verdicts and the study files; `grid.md` appended | partly: see C8 and A7 |
| 6 amendments | frozen-first study files; RESULTS states both readings | partly: see B1, C6, C9 |
| 5c Liu G4 | "not decidable with this coherence instrument" in RESULTS §6, missed list, CLOSEOUT §2.4 | PASS |
| 9 dose-32 repetition | measured by the study's code (A5) | PASS (see C5 for my own error) |
| 10 novelty | Arm 0 narrowed to Arad 2025 | PASS for Arm 0; C7 for Arm 2 |
| §10 next check | P-lit registered before O7's judge, then judged once | PASS |
| 3a, 4d | recorded | see C10 |
| H_write and H_suff wording | frozen first in RESULTS §1, CLOSEOUT status line, README | PASS, apart from C1 and C2 |

- Banned words: 0 hits in `RESULTS.md`, `CLOSEOUT.md`, the study `README.md`, the root `README.md` bullet, the handoff,
  `SPEC.md`, `JOURNAL.md` and `results/*.md`. PASS.
- Reviewer's own errors this pass: C5 (a), from my first pass. A5's 4.4% against 4.1% was a population difference, now
  stated.
- Not done this pass:
  - the O7 frozen coherent-only reading, beyond reading it from the verdict (0.473 [0.360, 0.580]);
  - O7 H_pred_conf (ρ −0.365 [−0.539, −0.129], REFUTED NOT READABLE) is printed in `grid.md` but not discussed in
    RESULTS;
  - the handoff rewrite, beyond its spend lines;
  - no web search.

---

## Addendum 2026-10-26 (third pass, `neurons/rw` at c87ea404)

Verdict: the study can close once N1 is fixed. N1 is a counting-rule fix in the missed-prediction totals, and no label
or headline depends on it. All 13 second-pass items and the A7 stamps are RESOLVED.

What was recomputed, from the raw answers and banks:
- Frozen pooled rows over Q4B alone: B1 0.020 [−0.020, 0.060] NOT SHOWN; H_ONE 0.020 [−0.110, 0.150] NOT SHOWN.
- O7, resampling the two judged α 0 prompts: o(16) 0.450 [0.350, 0.556]; paired 0.156 [0.050, 0.263].
- Q4B, the same way:
  - coherent-only reading: o 0.487 [0.380, 0.593], paired 0.140 [0.000, 0.280], so it now reads ABOVE CHANCE, NOT
    ABOVE α = 0;
  - H_pred_agree: R 1.518 [1.003, 2.204], so the amended label is now AGREE ABOVE DISAGREE, BELOW 2X.
  - Both label moves are journaled (JOURNAL 20:18:10Z) and stated in RESULTS §4, §7 and §11.
- Missed totals, recounted from the 39 rows of `results/missed_predictions.md`, with the three α_c rows counted once
  (37 predictions): frozen 17 missed, 10 not scorable, 10 landed; amended 19, 6, 12. These equal the printed totals.
- A7 re-render:
  - `verdict_arm0_G4.json` against `*_preA7` and against `*_preA28`: 0 differences in any non-label field.
  - 22 of 23 labels carry the frozen G-proj stamp; all 17 `secondary` labels are stamped. The one unstamped label is
    the G-judge0 gate label, which is not an Arm 0 row.
  - `study_arm0.json` differs from `*_preA7` only in one G4 label string.
- No thresholds file and no judge answer changed since fe248f2b.
- Banned words: 0 hits.

| item | status | evidence |
|---|---|---|
| B1 frozen pooled rows | RESOLVED | `study_arm0.json → frozen_rule.{B1,H_ONE}_pooled`, `grid.md` rows 23–24, RESULTS §1 and §7, CLOSEOUT §1.5, missed list B1 row |
| B2 missed list | RESOLVED, but see N1 | H_write row and the not-scorable rows added; 37 predictions |
| C1 literal-share wording | RESOLVED | RESULTS first paragraph and §3 give 60/57/80% and 57/36/32% |
| C2 P-lit direction | RESOLVED | RESULTS §3 |
| C3 P-lit room; in-sample readings | RESOLVED | RESULTS §3; Q4B and G4 split files read "in sample …, not a test" |
| C4 CLOSEOUT §2.3 | RESOLVED | — |
| C5 sample count | RESOLVED | RESULTS §4: 2 of 6, plus L19N5468- as the label's pattern; section 8 cited |
| C6 frozen-first labels | RESOLVED | RESULTS §2, §4, §7 |
| C7 Gur-Arieh wording | RESOLVED | RESULTS §9 |
| C8 `grid.md` | RESOLVED | regenerated (55 lines, no superseded rows); frozen H_depth Arm 2 row added; old file `grid_preC10.md` |
| C9 amendment timing | RESOLVED | RESULTS §10 "recorded" column; the times match the commits |
| C10 α 0 slots | RESOLVED | numbers above |
| C11 README lines | RESOLVED | study README and root README bullet |
| A7 secondary stamps | RESOLVED | above |

The first paragraph of RESULTS.md no longer overstates. Every claim in it carries its rule set (frozen, then amended)
and its measured share. It says P-lit did not confirm the literal concentration, and it says repetition is part of the
Q4B gain. Optional, not a FLAG: the paragraph could add that the Q4B gain filtered to coherent cells is 0.140 [0.000,
0.280], not above α 0.

### Remaining items

**N1 (FAIL; my own miss in the first two passes).** RESULTS §1 defines "not scorable" as "the label is CANCELLED (P8, or
underpowered) …". CLOSEOUT §2.7 says the same. But `missed_compute` scores the six Arm 1 target rows by
`head(label) != predicted label`, so a P8-CANCELLED label counts as **missed**. It also scores "at most 2 of 7 SWING" as
landed on a count of cancelled labels. H_suff's CANCELLED label, by contrast, is "not scorable". The printed totals
(17/10/10 frozen, 19/6/12 amended) therefore contradict the printed definition. Fix: apply one rule and recount. Two
options:
- If the definition covers labels, as written: the six target rows and the SWINGS-count row become not scorable.
  Frozen: 11 missed, 17 not scorable, 9 landed. Amended: 13, 13, 11.
- If it covers every coherent-only readout P8 cancelled: also the six control rows and the "every control" row.
  Frozen: 6, 24, 7. Amended: 8, 20, 9.
- Then update `missed_predictions.md`, RESULTS §1, §9 and §11, CLOSEOUT §2.7, and the handoff.

**N2 (FLAG).** `handoff/2026-10-26_run.md` line 183 still reads "Q4B's would-be label is SUPPORTED on coherent cells
(0.487 [0.380, 0.600])". RESULTS §10 cites the handoff for the frozen readings. Fix: replace with "Q4B's coherent-only
reading is ABOVE CHANCE, NOT ABOVE α = 0 (o 0.487 [0.380, 0.593], paired 0.140 [0.000, 0.280])".

Correction to my first pass: §2, row A27, gave as mitigation that "the frozen coherent-only reading also gives
SUPPORTED, 0.487 [0.380, 0.600]". With the two judged α 0 prompts resampled (C10) it reads ABOVE CHANCE, NOT ABOVE
α = 0. The mitigation is weaker than stated there.
