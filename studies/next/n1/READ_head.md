# N1: the head's read of the verdict and of the independent read

2026-11-07 15:00 CST (Fable). Read: `verdict_n1.json`, `thresholds.json`, `banks/weights-03/n1/metrics.json`, `READ_n1.md` and the reader's
outputs under `read_n1/` (`baseline.json`, `fragility.json`, `verdict_recount.json`, `matching.json`, `compare.json`).

**The frozen label stands and is quotable with its margin: on the test half the carrier's input rows change less than its
distinct-token twin's over the rise window, real − dt −0.432 [−0.817, −0.059] over 67 pairs; the output column is at selection,
−0.258 [−0.654, 0.130]; neither side exceeds the matched floor. The label is NEITHER with the input side BELOW_SELECTION.**
The reader's own recount from `stats.npz` reproduces every record to 2.5e-14 and the registered code's 180 verdict numbers to 1.1e-16;
the freeze-half thresholds to 7.8e-16; all 49 gates pass with their failing inputs failing.

## What holds, and how firmly

1. The input-side BELOW label does not depend on the interval method: percentile under three seeds, 100,000 resamples, basic, BCa,
   Student t and a bootstrap clustered by carrier neuron give upper bounds between −0.045 and −0.062 (`fragility.json`,
   `verdict_recount.json`); sign test p 0.014, Wilcoxon p 0.023.
2. The margin is two births. No single dropped birth flips the label (`drop_one_in`); dropping the two most negative births reads AT
   (+0.041 upper bound). The freeze half alone read AT (−0.217 [−0.638, 0.204]); the two halves do not differ beyond their
   intervals (z −0.74); pooled over 140 pairs the gap is −0.320 [−0.604, −0.035]. Quote the label with this sentence beside it.
3. In natural units the gap is small: the carriers' input rows change 1.6 % [0.03 % more, 3.2 % less] less of their relative change
   per window than the twins' (`scale.json`). The claim is "not more, and if anything less"; it is not a large effect.
4. The matching is clean on layer, birth bracket, group size and distinct tokens (67 of 67) and on the threshold ratio, but not on
   activation size: carriers' max |activation| is 1.39 times the twins' (log difference 0.329 [0.202, 0.468]). Adjusting for it
   widens the interval to AT (−0.20 [−0.59, 0.21]); but e has no slope on activation size within either kind (p ≥ 0.27) and the 34
   pairs closest in activation show the same gap (−0.436). Recorded as a limitation of the twin construction, not as a retag.
5. The birth bracket k is not the cause: e_in has no trend with k within real (p 0.93) or within dt (p 0.22), and every twin
   carries its birth's k.

## What the exploratory layer comparison does and does not say

The reader's sentence "the gap comes from twins changing more, not carriers changing less" rests on the layer mean as the reference.
The layer-wide distribution of e_in at the same k is skewed: mean −0.41, median +0.02 (`baseline.json`). Against the mean, carriers
sit at the layer (+0.11 [−0.15, 0.36]) and twins above it (+0.52 [0.26, 0.78]). Against the median, carriers sit below the layer
(−0.32 [−0.57, −0.08]) and twins at it (+0.09 [−0.17, 0.35]). The two references disagree on which side of the pair is unusual, so
the layer comparison does not settle it and is reported as exploratory with both numbers. What it does say either way: the carriers'
input rows do not change more than a typical neuron's in their layer, and the twins' change at least as much as typical. The
registered twin comparison, which is matched on the dating rule, is the claim.

## Reading

A birth is not a whole-vector weight event on the carrier. Over the rise window the carrier's gate rows, up rows and down column
change no more than a dated twin's, than its own earlier brackets, or than its layer; the input rows change somewhat less than the
twin's, and the carrier's input-row change keeps falling after the birth (profile k−2 −0.18, k−1 −0.30, k −0.41, k+1 −0.59). The
dating rule, applied to a random group of a pole's own future contexts, selects a neuron whose activation is rising and therefore
whose input rows are moving at least as much as typical; applied to a real birth it finds a neuron whose rows move like any other.
Together with N12 (the carrier wins by its own rise, which the dating rule nearly forces) and M2 (random groups are "predicted"
better than real births), the picture is consistent: what is special about a real birth is not in the carrier's weight norms. The
spec's decision table names the next instrument for NEITHER: a job-direction projection of the row change, not a norm, and the
readers of the column. Hypothesis to carry, post hoc, with its refutation: the real cluster's rise is carried by a change in the
residual stream the carrier reads (its inputs), not by the carrier's rows; refuted if the projection of the row change onto the
cluster's direction is larger for carriers than for twins.

## Status of the claims

| id | claim | status |
|---|---|---|
| N1-1 | input rows: real − dt −0.432 [−0.817, −0.059], BELOW_SELECTION, 67 pairs; margin two births; freeze half AT; pooled −0.320 [−0.604, −0.035] | stands under the frozen rule; quoted with its margin |
| N1-2 | output column: real − dt −0.258 [−0.654, 0.130], AT_SELECTION | stands |
| N1-3 | neither side above the matched floor (in −0.297 [−0.550, −0.059] against −0.053; out −0.039 [−0.276, 0.202] against 0.147); H_rows, H_col, H_together, H_sel refuted | stands |
| N1-4 | in natural units the input-row gap is −1.6 % [−3.2 %, +0.03 %] of r per window | reported, exploratory |
| N1-5 | layer reference: carriers at the layer mean and below its median; twins above the mean and at the median | exploratory, both references reported, not decided |
| N1-6 | activation-size mismatch of twins (1.39 times); adjustment widens to AT; within-kind slopes flat; close pairs show the gap | limitation recorded |

Not verified by anyone: `stats.npz` against the checkpoints themselves (no weights downloaded; K-B covers the down column at 11
points against two independent banks, 1.49e-7 against 1.3e-4; the input rows have no bank). The dry run is where the fixed
failing inputs of G_SAME, INPUTS, G_REV and G_SHAPE were shown to fail; neither the reader nor the head reran it today.
