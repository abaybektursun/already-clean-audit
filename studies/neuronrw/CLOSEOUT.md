# CLOSEOUT — NEURONRW

Status: **CLOSED**, 2026-10-26. Freeze tag `neuronrw-freeze-1` = 967a8db3. Post-freeze amendments A20–A28 in `SPEC.md`
§13 and §14.2. Under the frozen rules there is no headline: H_write MIXED (1 counted model), H_suff CANCELLED (P8: the
coherence gate failed on every model) (`results/study_arm0.json → frozen_rule`, `results/study_arm2.json →
frozen_rule`). The amended readings are
exploratory. Result document: `RESULTS.md`. Independent review: `checks/2026-10-26_result_review.md`.

## 1. Established

1. Every model gate (G-det, G-ident, G-place, G-gen, G-floor-dir, G-floor-pos, G-pc) passed in all 13 cells, and every
   registered failing input failed (review §3; `bank/rw/<cell>/out/controls.json`).
2. The known number reproduced exactly on Q4B in every Arm 0 forward and, after A26, in Arms 1 and 2 (own clean NLL
   1.3216137, own ΔNLL 0.3082006; JOURNAL 2026-10-26T11:40:00Z, 16:23:00Z).
3. The steering hook adds the intended vector after the post-MLP norm, at prefill and at decode steps, for both signs
   (G-place and G-gen cosines ≥ 0.9937; A21 wrong-sign inputs −0.996).
4. Under the amendments, a blind judge accepts a pole's re-explained input label for its own top-20 output tokens on
   0.180–0.200 of poles, against 0.005–0.065 for shuffled labels (`results/verdict_arm0_<M>.json`). The literal items
   (17–19%), whose label names one of the pole's own output tokens, carry 60%, 57% and 80% of the summed D_w: D_w
   0.553–0.618 there, 0.030–0.096 elsewhere (`results/literal_split_<M>.json → arm0`).
5. B1: under the frozen rules NOT SHOWN, Q4B alone, L 0.020 [−0.020, 0.060]. Under the amendments (A24), labels written
   with the vocabulary lines agree with the output tokens about as often as labels written without them: EQUIVALENT,
   L 0.013 [−0.010, 0.037] over three models (`results/study_arm0.json → frozen_rule.B1_pooled, B1_pooled`).
6. Under the amendments, steering Q4B poles at dose 32 raises the three-way on-label rate from 0.344 [0.262, 0.431] to
   0.531 [0.425, 0.637] (`results/verdict_arm2_Q4B.json → rows.H_suff`). About 29% of those cells are repetition loops
   by the review's post-hoc bigram metric, against 4.1% flagged by the coherence band
   (`results/literal_split_Q4B.json → repetition`).
7. Liu 2026's held-out collapse dose for Q4B (64) landed, one-sidedly (`plan/dose_scale_Q4B.json`).

## 2. Not established

1. A frozen headline for H_write or H_suff (status line above).
2. That input labels predict write directions beyond token identity: the non-literal D_w is 0.030–0.096 and crosses 0
   on O7.
3. That steering by label works on poles whose label names no output token: not shown on Q4B, the counted model
   (0.103 [−0.048, 0.262]). On G4 and O7, whose dose scale is not validated, the non-literal gain is above 0 (0.227
   [0.063, 0.391], 0.133 [0.016, 0.250]). The registered held-out P-lit was not confirmed: on O7 the literal poles
   gained 0.250 [0.031, 0.500] against 0.133 for the rest, so O7 does not show the literal concentration.
4. A validated dose scale on G4 and O7; Liu's G4 prediction is not decidable with this coherence instrument.
5. Any Arm 1 result (CANCELLED under P8), self-repair (CANCELLED, underpowered), H_ONE (NOT SHOWN), H_depth (MIXED).
6. Any win against a baseline (no comparison to prompting, a difference-of-means direction or an SAE latent).
7. The study's own predictions: of 37, under the frozen rules 6 missed and only 7 landed; 24 of 37 predictions could not be scored under the frozen rules: 20 because a gate cancelled their readout, and 4 because the readout was not measured, dropped or not decidable.
   Amended: 8 missed, 20 not scorable, 9 landed (`results/missed_predictions.md`).

## 3. Bugs not to repeat

1. Gates without room. Several registered failing inputs could not fail on real data: G-repair-fit's layer n − 2 read
   (A23), G-known's layer-0 zero on OLMo 3 (A24), G-coh's repeated-token string against a floor set by code text (A27).
   Show each failing input failing on real data of every model before the freeze, not only on tiny test models.
2. Exact equality on float lists. G-proj compared top-10 token lists in exact order; near-tied scores swap order under
   a 0.03% precision change (A24). Compare sets with a stated overlap, and keep a numeric gate primary.
3. bf16 against fp32. G-repair-fit compared an fp32 manual path with the model's bf16 NLL and measured rounding, not the
   formula (A23). Compare paths in the same dtype.
4. Window truncation between stages. Arms 1 and 2 forwarded the known windows shorter than Arm 0 and broke an exact
   reproduction (A26). A known-number gate must reuse the exact inputs that produced the banked number.
5. A plant with no variance. The room check added a constant to nulls that were almost all 0, so its failing input
   could not fail (A25). Plant effects on the quantity that carries the statistic's variance.
6. API spend outside the budget rule. RE-EXPLAIN ($93.04) ran with no cost rule, and the judge budget then forced A22's
   cut from 24,393 to 6,908 calls. Every API step gets a projected cost and a stop line before it runs.
7. An amendment written after a preview reading. A27 changed the study aggregation rule after the Q4B Arm 2 job's preview
   band had been read (review §2). Fix aggregation rules for every failure branch before any reading of that arm.

## 4. Next steps

1. Search over write directions, not input labels: contrastive directions and gradient attribution against
   behaviour-level labels, so a label is tested against what the unit writes rather than what it reads.
2. A coherence detector that flags repetition (distinct n-gram rate, repeated-span length), frozen on natural text and on
   steered samples labelled before the study, with registered repetition and gibberish inputs shown to fail on every
   model.
3. A pre-registered literal / non-literal design: draw literal and non-literal poles as separate strata with their own
   power, and make the non-literal steering gain the primary readout.

## 5. Artifacts

`README.md` indexes the study's files. Banks: `bank/rw/` = `s3://<s3-bucket>/results/neurons/rw/<cell>/<attempt>/`,
sha256 per file in `bank/rw/BANK_SHA256.txt`. Infrastructure note: `infra/aws_pg/operations/2026-10-26_neuronrw_deepinfra_limit.md`.
Cost: 5.99 instance-hours over 13 instances; judge $31.19 over 6,910 calls; RE-EXPLAIN $93.04 (`handoff/2026-10-26_run.md`).
