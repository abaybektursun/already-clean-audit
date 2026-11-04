# NEURONRW — results

Under the frozen rules the study has no headline: the H_write study label is MIXED with 1 counted model, and the H_suff
study label is CANCELLED because the coherence gate failed on every model (`results/study_arm0.json →
frozen_rule.H_write`, `results/study_arm2.json → frozen_rule.H_suff`). Under the post-freeze amendments, which are
exploratory, a blind judge accepted a pole's input label for its own top-20 output tokens on 0.180–0.200 of poles
against 0.005–0.065 for shuffled labels (D_w 0.175, 0.185 and 0.125 on Q4B, G4 and O7), and steering Q4B poles at dose
32 raised the three-way on-label rate from 0.344 [0.262, 0.431] to 0.531 [0.425, 0.637]. Arm 0's agreement sits mostly
(57–80% of summed D_w) in the 17–19% of items whose label names one of the pole's own top-20 output tokens. In Arm 2
those literal poles gain more per pole on every model (0.50, 0.50, 0.25 against 0.10, 0.23, 0.13), but they carry most
of the summed gain only on Q4B (57%; G4 36%, O7 32%). The held-out test of the literal concentration on O7 (P-lit) did
not confirm it, and part of the dose-32 gain on Q4B is the label's token repeated.

Paths are relative to `studies/neuronrw/`. Banks: `bank/rw/<cell>/<attempt>/` (untracked; S3
`s3://<s3-bucket>/results/neurons/rw/`, per-file sha256 in `bank/rw/BANK_SHA256.txt`). Freeze tag
`neuronrw-freeze-1` = 967a8db3. Post-freeze amendments A20–A28 are in `SPEC.md` §13 and §14.2. An independent review
recomputed every headline number from the raw judge answers and banks (`checks/2026-10-26_result_review.md`). Models:
Q4B `Qwen/Qwen3-4B-Base`, G4 `google/gemma-3-4b-pt`, O7 `allenai/Olmo-3-1025-7B`.

## 1. Study verdicts

Frozen rules first; they decide. Amended readings second; they are exploratory.

| readout | frozen rule (decides) | amended (exploratory) | artifact |
|---|---|---|---|
| H_write (Arm 0), study | MIXED (1 counted model): G4 CANCELLED by the exact-order G-proj (0.880 < 0.99); O7 CANCELLED by the layer-0 G-known input (0.114 nats < 0.5); Q4B's supported label NOT READABLE (constant-plant room check cannot fail) | H_WRITE on 3 counted models (A24, A25) | `results/study_arm0.json → frozen_rule, H_write` |
| H_depth (Arm 0), study | MIXED (1 counted model) | MIXED (3 counted models) | `results/study_arm0.json → frozen_rule.H_depth, H_depth` |
| H_suff (Arm 2), study | CANCELLED (P8: coherence gate not shown to work on every model); the two-model §7.1 rule on Q4B alone: MIXED (1 counted model) | SUPPORTED, decided on Q4B alone (A27), stamped "coherence gate not shown to work" | `results/study_arm2.json → frozen_rule, H_suff` |
| H_depth (Arm 2), study | CANCELLED (P8) | NOT SHOWN, decided on Q4B alone (A27) | `results/study_arm2.json → frozen_rule.H_depth, H_depth` |
| Arm 1, every target | CANCELLED (P8) | CANCELLED; curves exploratory | `results/verdict_arm1_{Q4B,G4}.json` |
| self-repair | CANCELLED (underpowered) | CANCELLED (underpowered) | `results/repair_{Q4B,G4,pooled}.json` |
| H_ONE pooled | NOT SHOWN, Q4B alone, 0.020 [−0.110, 0.150] over 50 pairs (G4 and O7 not counted) | NOT SHOWN, 0.050 [−0.033, 0.133] over 150 pairs, 3 models (A24) | `results/study_arm0.json → frozen_rule.H_ONE_pooled, H_ONE_pooled` |
| B1 pooled | NOT SHOWN, Q4B alone, L 0.020 [−0.020, 0.060] over 100 items (G4 and O7 not counted) | EQUIVALENT, L 0.013 [−0.010, 0.037] over 300 items, 3 models (A24) | `results/study_arm0.json → frozen_rule.B1_pooled, B1_pooled` |
| predictions (37) | 6 missed, 24 not scorable, 7 landed | 8 missed, 20 not scorable, 9 landed | `results/missed_predictions.md` |
| P-lit (held out on O7, A28) | not confirmed: O7 does not show the literal concentration | — | `results/literal_split_O7.json → arm2` |

"Not scorable" means the readout a prediction names does not exist under that rule set: a gate cancelled it (a CANCELLED
label, P8 or underpowered, or a number read from a cancelled cell, such as a cancelled Arm 1 pair's S), the readout was
not measured (Arm 0 fc3, A22), the slot was dropped (TeX), or the verdict is not decidable (Liu's G4 α_c). One rule
applies to every row (review N1). A prediction with several rows counts once. 24 of 37 predictions could not be scored under the frozen rules: 20 because a gate cancelled their readout, and 4 because the readout was not measured, dropped or not decidable.

## 2. Arm 0 — read against write, per model

A22 subsample: 150 items per model, 100 of them ONE poles; re-explained labels written without the output tokens (SPEC
§5.0). A = true-label "yes" rate; Sh = shuffled-label "yes" rate; Fx = "yes" rate of the true label against another
pole's list; D_w = mean over ONE items of y_true − (y_s1 + y_s2) / 2. Intervals are 95% bootstrap intervals.

| model | A | Sh | Fx | D_w | D_f | δ_w | label: frozen; amended | artifact |
|---|---|---|---|---|---|---|---|---|
| Q4B | 0.180 [0.110, 0.260] | 0.005 [0.000, 0.015] | 0.020 [0.000, 0.050] | 0.175 [0.105, 0.250] | 0.160 [0.090, 0.240] | 0.03 | frozen: H_WRITE NOT READABLE (room check cannot fail); amended: H_WRITE, above the margin | `results/verdict_arm0_Q4B.json` |
| G4 | 0.200 [0.120, 0.280] | 0.015 [0.000, 0.035] | 0.010 [0.000, 0.030] | 0.185 [0.110, 0.265] | 0.190 [0.110, 0.270] | 0.05 | frozen: CANCELLED (exact-order G-proj 0.880); amended: H_WRITE, above the margin | `results/verdict_arm0_G4.json` |
| O7 | 0.190 [0.120, 0.270] | 0.065 [0.030, 0.105] | 0.050 [0.010, 0.100] | 0.125 [0.035, 0.215] | 0.140 [0.060, 0.220] | 0.10 | frozen: CANCELLED (no Arm 0 data after two G-known aborts); amended: H_WRITE, interval crosses the margin | `results/verdict_arm0_O7.json` |

The judge rarely accepts a shuffled label. On Q4B and G4, D_w is close to A (0.175 against 0.180; 0.185 against
0.200); on O7, Sh is 0.065, so D_w is 0.125 against A 0.190. An item is accepted mostly when
its label names a token of its own list (section 3). Parent-ONE poles re-explained as ONE: Q4B 329/358, G4 306/338, O7
292/311 (`results/reexplain_<M>.jsonl`). The three-way token-list task (fc3) was not run (A22).

## 3. The literal split and the held-out test P-lit

A pole is literal when its re-explained label names one of its own top-20 promoted output tokens, by the frozen
`literal_overlap` rule (SPEC §5.1). The split was made after the review and is exploratory, except on O7's Arm 2,
where it was fixed before any O7 Arm 2 answer existed (SPEC §14.2 A28 (2)).

| model | Arm 0 D_w literal | Arm 0 D_w non-literal | Arm 2 paired gain literal | Arm 2 paired gain non-literal | artifact |
|---|---|---|---|---|---|
| Q4B | 0.553 [0.342, 0.763] (19 items) | 0.086 [0.037, 0.148] (81) | 0.500 [0.294, 0.706] (17 poles) | 0.103 [−0.048, 0.262] (63) | `results/literal_split_Q4B.json` |
| G4 | 0.618 [0.353, 0.853] (17) | 0.096 [0.036, 0.163] (83) | 0.500 [0.188, 0.750] (16) | 0.227 [0.063, 0.391] (64) | `results/literal_split_G4.json` |
| O7 | 0.556 [0.333, 0.778] (18) | 0.030 [−0.055, 0.116] (82) | 0.250 [0.031, 0.500] (16) | 0.133 [0.016, 0.250] (64) | `results/literal_split_O7.json` |

Arm 2 gains are o(d_p) − o(0) with the two-way bootstrap of SPEC §5.7; d_p is 32 on Q4B and 16 on G4 and O7.

P-lit predicted, on O7 at d_p 16, a literal gain of at least 0.30 and a non-literal interval that includes 0.10 or lies
below it; it named a non-literal interval entirely above 0.15 as the refuting observation. O7 read a literal gain of
0.250 [0.031, 0.500] and a non-literal gain of 0.133 [0.016, 0.250] (`results/literal_split_O7.json →
arm2.literal.paired_gain_two_way`, `arm2.non_literal`). P-lit is not confirmed: on the held-out model the literal
poles gained 0.250 against 0.133 for the rest, and the non-literal poles carry 68% of the summed gain. O7 does not show
the literal concentration seen in sample on Q4B. The refuting observation did not occur, but it had little room: at
O7's interval half-width (about 0.12) it needed a non-literal gain near 0.27, and G4's 0.227 would not have fired it.
The Q4B and G4 rows are in sample (P-lit was built from them), not tests (`results/literal_split_{Q4B,G4}.json →
arm2.p_lit`).

Shares of the summed effect carried by the literal items or poles: Arm 0 D_w 60% (Q4B), 57% (G4), 80% (O7); Arm 2 gain
57% (Q4B), 36% (G4), 32% (O7). The concentration is strong in Arm 0 on every model and in Arm 2 on Q4B only. On G4 and
O7 the non-literal Arm 2 gain is above 0; on Q4B, the one counted Arm 2 model, it is not shown (0.103 [−0.048, 0.262]).

## 4. Arm 2 — steering by label, per model and dose

80 ONE poles per model; the judge picks the pole's label out of three; chance is 1/3; two judged prompts per cell (A22).

| model | α 0 | α 2 | α 4 | α d_p | paired o(d_p) − o(0) | ONE cells repetitive at d_p | band-collapsed samples at d_p | label | artifact |
|---|---|---|---|---|---|---|---|---|---|
| Q4B (d_p 32) | 0.344 [0.262, 0.431] | 0.338 [0.250, 0.425] | 0.394 [0.306, 0.475] | 0.531 [0.425, 0.637] | 0.188 [0.056, 0.325] | 29.3% | 4.1% | frozen CANCELLED (P8); amended SUPPORTED [coherence gate not shown to work] | `results/verdict_arm2_Q4B.json → rows.H_suff, secondary.other_doses` |
| G4 (d_p 16) | 0.244 [0.175, 0.319] | 0.463 [0.375, 0.550] | 0.500 [0.406, 0.594] | 0.525 [0.419, 0.631] | 0.281 [0.138, 0.419] | 48.3% | 28.4% | frozen CANCELLED (P8); amended exploratory (dose scale not validated; coherence gate not shown to work) | `results/verdict_arm2_G4.json` |
| O7 (d_p 16) | 0.294 [0.219, 0.369] | 0.325 [0.244, 0.412] | 0.331 [0.256, 0.412] | 0.450 [0.350, 0.556] | 0.156 [0.050, 0.263] | 7.0% | 2.3% | frozen CANCELLED (P8); amended exploratory (same stamps) | `results/verdict_arm2_O7.json` |

Repetition is the review's post-hoc metric: a cell is repetitive when its continuations have a mean distinct-bigram
ratio below 0.5 (`results/literal_split_<M>.json → repetition`); the band's share is over ONE-pole samples. The
coherence band flags far fewer samples. Part of the on-label gain at d_p is the label's token repeated: in 2 of the 6
correct Q4B dose-32 cells the review opened (L26N3166- "anyone", L31N9536- "a"), the text shows the label as a
repetition loop of a token the label names; in a third (L19N5468-), the label's pattern appears inside a repetition
loop (section 8). The review also found the gain in
non-repetitive cells (+0.17 against the same poles' α 0 rate; `checks/2026-10-26_result_review.md` §6).

Q4B coherent-only reading (cells whose shown samples all lie in the band, 75 poles), beside the amended one: o(d_p)
0.487 [0.380, 0.593], paired 0.140 [0.000, 0.280], which reads ABOVE CHANCE, NOT ABOVE α = 0; its frozen label is
CANCELLED (P8) (`results/verdict_arm2_Q4B.json → frozen_coherent_only`). G-judge2 passed on every model (label-prompted
1.00).

The Arm 2 verdicts, literal splits, study files, missed list and grid were re-rendered after unblinding with the
merged review fixes (`neurons/rw-code` eabc64cd, merged at bf38fb0f) and the same frozen thresholds: the paired
bootstrap now resamples the two judged α = 0 prompts (C10), the frozen pooled rows pool Q4B alone (B1), and the
prediction list is complete (B2). The pre-fix files are kept as `*_preC10` and `results/study_arm0_preB1.json`
(d611393c); every moved number is listed in `JOURNAL.md` 2026-10-26T20:18:10Z.

## 5. Arm 1 — the volume knob

Every Arm 1 label is CANCELLED under P8: the coherence gate's repeated-token input lies inside the band in both cells
(`results/thresholds_arm1_{Q4B,G4}.json`). The curves below are exploratory and never a headline. The TeX slot was
dropped: no candidate fires on 15 of 20 prompts (SPEC §5.5).

React useEffect (G4 L07N10202-), all-sample construct presence over 200 samples per point: −32 0.00, −16 0.26, −8 0.53,
−4 0.71, 0 0.81, +4 0.66, +8 0.52, +16 0.02, +32 0.00; the matched control L07N5834- reads −16 0.12, 0 0.81, +16 0.37
(`results/verdict_arm1_G4.json`). Presence falls in both directions for target and control, which is generic
degradation, not a label-specific push.

Three Q4B swings, paired S_t − S_c at d* 64 (`results/verdict_arm1_Q4B.json → targets`):

| target | control, d | paired S_t − S_c | label if not cancelled |
|---|---|---|---|
| SQL FOREIGN KEY L11N5238+ | L11N4144+, 0.61 | 0.096 [0.020, 0.193] | UNRESOLVED |
| Angular L27N2924- | L27N7674+, 0.06 | 0.397 [0.228, 0.577] | UNRESOLVED |
| HTML title L08N3398- | L08N9674-, 1.43 (poor match) | 0.390 [0.193, 0.604] | SWINGS (underpowered) |

## 6. Dose scale and Liu 2026

| model | α_c frozen | α_c running | Liu's held-out α_c | verdict | artifact |
|---|---|---|---|---|---|
| Q4B | 64 | 64 | 64 | lands (α* recomputed 65.9; one-sided: no sweep dose up to 32 collapses half the samples) | `plan/dose_scale_Q4B.json` |
| G4 | 64 | 32 (A24 fallback) | 16 or 32 | not decidable with this coherence instrument: the sweep band failed G-coh on G4, and at ±32 the text is mostly not text | `plan/dose_scale_G4.json` |
| O7 | 64 | 32 (A24 fallback) | none | — (G-coh failed; ±32 sweep text mostly repetition) | `plan/dose_scale_O7.json` |

The A24 fallback was refuted by its own check on G4 and O7 (A27), so neither has a validated dose scale.

## 7. Other rows

Frozen reading first, amended second.

- B1 pooled: frozen NOT SHOWN, Q4B alone, L 0.020 [−0.020, 0.060] over 100 items; under the amendments (A24) EQUIVALENT,
  L 0.013 [−0.010, 0.037] over 300 ONE items: labels written with the vocabulary lines agree with the output tokens
  about as often as labels written without them (`results/study_arm0.json → frozen_rule.B1_pooled, B1_pooled`).
- H_ONE pooled: frozen NOT SHOWN, Q4B alone, 0.020 [−0.110, 0.150] over 50 pairs; amended NOT SHOWN, 0.050 [−0.033,
  0.133] over 150 pairs (`→ frozen_rule.H_ONE_pooled, H_ONE_pooled`).
- H_depth (Arm 0), per model: frozen Q4B SUPPORTED NOT READABLE (room check cannot fail), G4 and O7 CANCELLED; amended Q4B
  0.344 [0.173, 0.522] SUPPORTED (underpowered), G4 0.076 [−0.114, 0.266] NOT SHOWN, O7 0.217 [−0.003, 0.442] NOT SHOWN;
  study MIXED under both (`results/verdict_arm0_<M>.json → rows.H_depth`).
- Self-repair CANCELLED, underpowered, under both: 16 of 156 Q4B poles and 15 of 201 G4 poles qualify against a minimum
  of 20; M_r 3.94 [0.265, ∞) and 0.725 [0.196, 3.04] (`results/repair_{Q4B,G4}.json`).
- Arm 2 rows on Q4B, frozen CANCELLED (P8) for each; amended: H_pred_agree AGREE ABOVE DISAGREE, BELOW 2X, R 1.518 [1.003,
  2.204]; H_pred_conf NOT SHOWN, ρ −0.033 [−0.248, 0.186]; H_depth (Arm 2) NOT SHOWN, 0.069 [−0.190, 0.326]
  (`results/verdict_arm2_Q4B.json → rows`). The G4 and O7 rows are exploratory; O7's H_pred_conf reads ρ −0.365
  [−0.540, −0.129] with "rule fires on the null", so it cannot be read (`results/verdict_arm2_O7.json → rows`).

## 8. Banked samples

Arm 0, judged yes. Q4B L31N9536-: top tokens `" a"`, `"个"`, `"\ta"`, `"-a"`, `"a"`; label "Fires on the word "a" (or
the "-a" piece) in fixed idioms where the next word is almost certainly "lifetime" …". Agreement by token identity.
(`bank/rw/arm0_q4b/1/out/arm0/projections.jsonl`; `results/reexplain_Q4B.jsonl`; `results/judge_a0-yn-Q4B.jsonl`.)

Arm 0, judged no. Q4B L10N8041+: top tokens `"arma"`, `"惜"`, `"煮"`, `" distinct"`, `"的就是"`; label "Fires on the head
noun that completes a multi-word proper name or title (Highway in "U.S. Highway 6" …". The "no" is right. (Same files.)

Arm 2 Q4B, α 32, on-label (judge picked the true label at prompts 0 and 1). L26N3166-, label about "anyone"/"anybody":
"Imagine anyone could make anyone do anything. Anyone. Anyone, said anyone. Anyone, said anyone, anyone, …" — a
repetition loop of the labelled token (`bank/rw/arm2_q4b/1/out/arm2/samples.jsonl`, prompt 1, k 0;
`results/judge_a2-fc3-Q4B.jsonl`). The pole's α 0 call at prompt 1 was also right, so this cell is not a gain over α 0.

Arm 2 Q4B, α 32, off-label (judge picked a decoy at both prompts). L07N7370+, label about the "alon" piece of
"Thessalonians": "This is a test message.\n\nThis is a test message with a lot of text in it. …" (same files, prompt 0,
k 0). The text does not show the label, so the miss is correct.

## 9. Three dimensions

- **Predictive.** Liu's held-out α_c for Q4B landed (64), though only one-sidedly, and the registered held-out P-lit on O7
  was not confirmed. Under the frozen rules 6 of 37 pre-registered predictions missed and 7 landed; 24 of 37 predictions could not be scored under the frozen rules: 20 because a gate cancelled their readout, and 4 because the readout was not measured, dropped or not decidable.
- **Novel.** Arm 0's agreement sits in the token-identity overlap that Arad et al. 2025's input score already counts; what
  is new is the object (MLP neuron poles in three base-model families) and the read side (labels written without the
  output tokens). Arm 2 tests steering single neurons by input label in base models (80 poles per model), which
  `studies/neuronoff/LIT_steering.md` §D lists as not done before; Gur-Arieh et al. 2025, on an instruct model with a
  different protocol, found input-based labels near chance (36.9% against 33.3%), their sample size is not recorded in
  our notes, and AxBench already ranks direction steering below prompting.
- **Practical.** No win against a baseline was measured: there is no comparison to prompting, a difference-of-means
  direction or an SAE latent. The Q4B on-label gain of 0.188 includes repetition of the label's token, so it is not a
  usable steering method as measured.

## 10. Amendments A20–A28

| id | recorded (JOURNAL, UTC) | what failed | what changed | what the frozen rule reads |
|---|---|---|---|---|
| A20 | 2026-10-26T09:46:31Z | the G-judge0 firing-token step asserted 20 distinct tokens; the plan kept 300 positions | top 1,000 positions, top 20 by frequency, floor 10, frozen replacement | every Arm 0 job raises; no Arm 0 data |
| A21 | 09:46:31Z | `realized_cos` was measured against +w for every dose | measured against the signed intended vector | every negative-dose Arm 1 cell reads "instrument not shown to fire" |
| A22 | 10:12:40Z | the judge budget ($30) was far below the frozen scale (24,393 calls) | 6,908 calls, stop at $28, underpowered rows marked | the judge steps stop at the provider limit with A0 and A2 incomplete |
| A23 | 11:12:21Z | G-repair-fit compared fp32 with bf16; its failing input had no room | model-dtype manual path; failing input at layer n // 2 | both models' repair labels CANCELLED (128/156 and 128/201 poles out) |
| A24 | 13:18:19Z | exact-order G-proj on near-tied tokens; G-known failing input without room on O7; G-coh failed on G4 | 9-of-10 overlap; layers 0..n // 4 zeroed; fallback α_c | G4 Arm 0 CANCELLED; O7 leaves the study; G4 Arms 1 and 2 at α_c 64 |
| A25 | 14:14:50Z, after all A0 answers were committed and before unblind0 | the room check planted a constant on nulls that are almost all 0 | Bernoulli plant on the true-label answers | H_write's supported label NOT READABLE on every model |
| A26 | 15:23:05Z | Arms 1 and 2 cut the G-known windows short | G-known on the Arm 0 windows | Q4B fails G-known twice and leaves Arms 1 and 2; the run stops |
| A27 | 16:25:52Z, after the Q4B Arm 2 preview G-coh reading and before any Arm 2 judge call | the A24 fallback dose gave mostly broken text; G-coh had no room | no coherence change; G4 and O7 Arm 2 exploratory; H_suff on Q4B alone | every H_suff label CANCELLED |
| A28 | 18:40:05Z, after the review and before any O7 Arm 2 judge call | the judge budget could not cover O7's Arm 2 | stop at $48; P-lit registered on O7; review findings accepted | O7's Arm 2 stays unjudged |

The frozen readings come from `handoff/2026-10-26_run.md` and `SPEC.md` §14.2.

## 11. Review findings and what was done

| item | finding | done |
|---|---|---|
| E1, B2 | "11 of 18 missed" covered §5.6 only; the second pass found the H_write row and the not-scorable rows missing | regenerated over all 37 predictions; after N1 (below): frozen 6 missed, 24 not scorable, 7 landed; amended 8, 20, 9 (`results/missed_predictions.md`) |
| N1 | the printed "not scorable" definition covered cancelled labels, but cancelled Arm 1 rows were scored as missed or landed | one rule for every row (a readout a gate cancelled is not scorable); `missed` re-rendered; 17 / 10 / 10 → 6 / 24 / 7 frozen, 19 / 6 / 12 → 8 / 20 / 9 amended; old file `results/missed_predictions_preN1.md` |
| N2 | the handoff still called Q4B's coherent-cell reading SUPPORTED | corrected to ABOVE CHANCE, NOT ABOVE α = 0, 0.487 [0.380, 0.593], paired 0.140 [0.000, 0.280] |
| B1 | frozen pooled B1 and H_ONE printed with the amended (three-model) values | frozen rows now pool Q4B alone: B1 NOT SHOWN 0.020 [−0.020, 0.060], H_ONE NOT SHOWN 0.020 [−0.110, 0.150] |
| C1–C5, C7, C11 | literal-share wording; P-lit stated the wrong way round; refuting branch with little room; sample count; Gur-Arieh comparison; README wording | rewritten with the measured shares, P-lit as not confirmed on O7, the 2-of-6 sample count, the instruct-model note, and the README line |
| C6, C9 | frozen labels missing from some rows; amendment timing | frozen first in every table and row; "recorded" column in section 10 |
| C8 | `grid.md` appended rather than regenerated | regenerated from the results; old file kept as `grid_preC10.md` |
| E2 | stamps missing in `grid.md`, `verdict_arm0_G4.json`, `study_arm2.json` | added; earlier files kept as `*_preA28` |
| 6 | frozen readings missing from some files; A27 changed the study rule | study files regenerated frozen-first; this document states both readings in that order |
| 5c | Liu G4 "missed" printed without the G-coh failure | reworded to "not decidable with this coherence instrument" |
| 9 | "α 32 mostly fluent" read one sample | repetition measured: 29.3% of Q4B cells at α 32 |
| 10 | novelty must be narrowed to Arad 2025 | adopted (section 9) |
| §10 | judge O7 with a registered literal-pole prediction | done: P-lit registered and O7 judged once; not confirmed |
| 3a, 4d, C10 | five α = 0 prompt slots in a bootstrap with two judged; preview band floors quoted | bootstrap fixed and re-rendered; no H_suff label moves; O7 paired lower end 0.044 → 0.050; two secondary readings changed: the Q4B coherent-only paired lower end 0.007 → 0.000 (reading now ABOVE CHANCE, NOT ABOVE α = 0) and Q4B H_pred_agree (amended, now AGREE ABOVE DISAGREE, BELOW 2X) |

## 12. Cost

5.99 instance-hours over 13 instances; judge steps $31.19 over 6,910 calls; RE-EXPLAIN $93.04 over 2,674 requests
(`handoff/2026-10-26_run.md`, roster and spend tables).

## 13. Not done

- The coherence instrument was not redesigned. It fails to flag repetition, and P8 cancels every Arm 1 label.
- O7 entered only exploratory rows in Arm 2; G4 and O7 have no validated dose scale.
- Arm 1 is cancelled on every target.
- Self-repair is cancelled as underpowered; the Arm 0 three-way task was not run.
