# N2V: the head's read of the verdict and of the independent read

2026-11-07 19:40 CST (Fable). Read: `verdict_n2v.json`, `thresholds.json`, `banks/score-02/n2v/metrics.json`, the executor's
`JOURNAL.md`, `READ_n2v.md` and the reader's outputs under `read_n2v/` (`r1_recount.json`, `r2_verdict.json`, `r4_compare.json`,
`r5_reread.json`, `r6_checks.json`, `r7_examples.json`, `r8_gates.json`); `infra/aws_pg/jobs/n2v_score.py` for the defect in §8 h1.

**The frozen labels stand and are quotable without a margin clause: on the test half N12's banked bf16 first-order prediction
ranks the fp32 zero-ablation loss change of 284 pole ablations with Spearman 0.960 [0.915, 0.990] against a shuffle bar of 0.118;
the finite-difference, own-position and fp32-prediction readouts pass at 0.987, 0.986 and 0.967; branch A.** The reader's recount
from the banked per-position losses reproduces all eight readout fields of all 612 pairs bitwise, the freeze-half thresholds to
3e-16 and the registered verdict's 81 numbers to 2.8e-14; 42 gates pass with their 42 failing inputs failing; the reader committed
its own labels before running the registered code.

## What holds, and how firmly

1. The label does not depend on the interval method: seeds 0, 1 and 2, 10,000 resamples and the basic interval give lower bounds
   between 0.909 and 0.930 on the window readout, against a shuffle floor near 0.12 and the bar 0.7 (`r2_verdict.json`). Leaving
   any one unit out keeps ρ within 0.959–0.979.
2. The rank agreement is within units, which is the question N12 asks. The pole with the largest predicted change is the pole
   with the largest measured change in 65 of 71 test units (0.915 [0.845, 0.972]; chance level 0.25) and 72 of 82 freeze units;
   Kendall τ over a unit's four poles averages 0.915; ρ after per-unit centering is 0.938. Permuting the measurement within
   units, which keeps all between-unit spread, leaves ρ at a median 0.286 (`r6_checks.json` a).
3. A failure was reachable. The same pairs read with N12's bf16 measurement give ρ 0.043 (test) and 0.100 (freeze); a shuffled
   measurement gives about 0.01. G-ROOM's failing input reads 0.076 and 0.064 against the bar 0.9. The ceiling worked as a copy
   detector and had no room to fire on a genuine window measurement (a measurement equal to the fp32 prediction gives 0.995).
4. The zero ablation is first order at the median and not at the extremes. Median ΔL / prediction lies between 0.94 and 1.06 in
   every decile of |prediction|; the 10 % finite difference has slope 0.988 (test) and 1.008 (freeze) through the origin; the
   second-order share, median |b| / |ΔL| from a quadratic through the zero and 10 % rows, is 0.18–0.20. The window slopes 0.837
   (test) and 1.110 (freeze) come from a few large pairs: the top five test pairs carry 48 % of Σ pred², and one of them,
   L06N8966− in L06N7759+:c2, reverses sign under zero ablation (prediction −0.650, ΔL +0.230) while its 10 % step keeps the sign.
   Without the top three the test slope is 1.025 (`r6_checks.json` c).
5. The two halves agree: freeze − test in ρ is +0.011 [−0.036, +0.063] on window and inside its interval on every readout.

## What the result does and does not say

- It validates N12's instrument, not N12's hypotheses. On branch A, N12's gradient labels move from DOES NOT COUNT to counted at
  the values N12's exploratory read printed on the 6th: H_signal REFUTED (mean u_S 0.527 [0.451, 0.600] against the bar 0.4225:
  at T_(k−1) the carrier's pull on its contexts is not above its competitors' beyond the matched floor), H_sel_signal_pb and
  H_sel_signal_dt AT_SELECTION (−0.039 [−0.134, 0.057]; −0.020 [−0.124, 0.084]: the carrier's pull advantage is no larger than a
  twin pole's), H_bridge UNDECIDED (−0.013 [−0.072, 0.046]: more pull at T_(k−1) does not foretell who keeps rank by T_k). These
  four labels read N12's gradient records, not the measured column, so N2V could change their status and not their values. The
  reader recomputed them with its own code to 1.1e-16 (`r5_reread.json`). Status: a dated re-reading.
- It retags one sentence of N12's head read. "Activation-times-gradient does not predict single-neuron ablation at a neuron's
  strongest positions in this model" (`studies/next/n12/READ_head.md`) was a statement about the instrument; the cause was the bf16
  resolution of the summed loss (N12's median measured change was 0.65 nats against predictions of 0.02; in fp32 the medians are
  0.0224 and 0.0219). The sentence is retagged today with a pointer here. The spec's five hypotheses: H_res stands; H_nonlin,
  H_grad, H_later and H_floor are refuted.
- The bf16 gradient's precision matters for some units. Where the bf16 and the fp32 forward-mode predictions disagree most (the
  top 10 % of pairs by their difference), the measurement sides with the fp32 one in 26 of 28 test pairs; in those pairs the bf16
  prediction's median relative error is 0.64 against 0.085 elsewhere. One test unit, L02N4649+:c3 (layer 2, T_14), has the wrong
  bf16 prediction on all four poles and names the wrong top pole (`r6_checks.json` e). Rule for later instruments: compute the
  first-order prediction in fp32, by forward mode where the backward does not fit, and report the finite-difference check beside
  the zero ablation.
- The practical dimension (spec §11) is now measured in this program's terms: one fp32 forward-mode pass per window and pole
  (2.6 s per row) names the same largest-effect pole as zero ablation in 0.915 of units (the full signed order of the four poles in 0.873, 62 of 71; corrected 2026-11-08 after the abstract check, which found the earlier text conflating the two, and its re-check, which found 0.845 — the bf16 window readout's share — written here in place of the fp32 forward-mode pass's); per-pole ablation of the same pairs
  cost 1.0 s per row and the full card 5.93 GPU-h. The saving is in the instrument's reach, not in this run: a gradient pass ranks
  every neuron of a layer at once where ablation needs one row per pole.

## Reading

N12's finding stands on its own measurement and now has its gradient part. At the point before a real birth the training signal
on the cluster's contexts does not single out the carrier from its competitors (H_signal REFUTED under a validated instrument),
and the pull at that point does not foretell who keeps rank (H_bridge UNDECIDED). With N12's realized labels (the carrier wins by
its own rise; its leaders keep their rank) and N1 (the carrier's rows change no more than a twin's), the picture is consistent: the
winner is decided between the checkpoints, by something a single gradient at T_(k−1) does not show and the carrier's own weight
norms do not carry. The next instrument stays the one N1's decision table names: a projection of the row change onto the cluster's
direction, read with finer checkpoints where they exist.

## Status of the claims

| id | claim | status |
|---|---|---|
| N2V-1 | G-VAL window PASS on the test half: ρ 0.960 [0.915, 0.990], shuffle 97.5th 0.118, 284 pairs, 71 units; freeze 0.971 [0.939, 0.993]; branch A | stands under the frozen rule |
| N2V-2 | G-FD 0.987 [0.971, 0.995], G-VAL own 0.986 [0.969, 0.996], window_jvp 0.967 [0.923, 0.994], all PASS on both halves; G-ROOM 0.9999, 0.9957, 1.0000 with its failing input 0.076 | stands |
| N2V-3 | within-unit rank agreement: top pole matched in 65 of 71 test units (0.915 [0.845, 0.972]); within-unit permutation leaves ρ 0.286 | stands, exploratory (not a registered readout) |
| N2V-4 | zero ablation is first order at the median (ΔL / pred 0.94–1.06 by decile; fd slopes 0.988, 1.008) with a second-order share of 0.18–0.20 and one sign reversal among the five largest pairs | reported, exploratory |
| N2V-5 | N12's gradient labels counted at their printed values: H_signal REFUTED 0.527 [0.451, 0.600] against 0.4225; H_sel_signal_pb AT −0.039 [−0.134, 0.057]; H_sel_signal_dt AT −0.020 [−0.124, 0.084]; H_bridge UNDECIDED −0.013 [−0.072, 0.046] | counted; a dated re-reading, not a prediction |
| N2V-6 | N12's sentence that the gradient does not predict ablation is retagged: the cause was bf16 loss resolution; the bf16 prediction itself misses in about 10 % of pairs, where the fp32 prediction is right in 26 of 28 | retag recorded today |
| N2V-7 | defect in a reported field: `sum_abs_h` is keyed by (window, layer) in the job's capture, so L01N369+:c0's four pairs sum c4's h at c4's indices in three shared windows at T_11; no rule or readout reads it; the forward-mode tangent reads h inside the forward and is not affected (`n2v_score.py` jvp_rows) | limitation recorded; the field is not quotable for units sharing a window and layer |

Not verified by anyone: K-A's bf16 rows, I-FWD's and I-JVP's second passes, h and the probe's v (not banked; the reader checked
their failing inputs and two further identities from the per-position rows); pred_bf16 from N12's pull arrays (N12's reader did,
to 6.7e-8); u_matched (taken from N12's records); the decoded token text (the job's). The dry run is where the fixed failing inputs
were shown to stop the job; nobody reran it today.
