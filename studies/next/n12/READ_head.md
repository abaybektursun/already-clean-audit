# N12, the head's read (Fable, the different-model read, 2026-11-06 13:50Z)

**Answer.** The verdict stands as written: `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal DOES NOT COUNT; H_bridge
DOES NOT COUNT; H_dt SUPPORTED`. My own code (`read_head/recount.py`, numpy and scipy only, reading the job's unit records
`banks/score-02/n12/n12_test.jsonl` and nothing from `infra/pg/n12.py`) gives every realized label and its numbers to the
printed digits with my own bootstrap seed (mean F 0.9507 [0.731, 1.179] against the verdict's [0.738, 1.180]; the competitors'
drops 0.938 real against 1.164 for M2's twins, paired −0.226 [−0.401, −0.051]; hits 42 of 71 real, 53 of 70 M2 twins, 41 of 67
distinct-token twins; real − dt −0.030 [−0.149, +0.090]; dt − pb −0.134 [−0.254, −0.015]) and the failed validation (Spearman
0.0425 on 284 (unit, pole) pairs; output in `read_head/recount_output.txt`). The decomposition's identity holds on every record
(F = log10 r_o − log10 r_c to 1e-9). The independent reader's recount is pending; this is the different-model read.

## What the numbers say, in plain words

The question: one checkpoint before a cluster's birth the carrier is often not the leader on the cluster's contexts (M2).
By the birth checkpoint it leads. Does it win by its own rise, or because the leaders fall away? And did the training signal
one checkpoint before already favour it?

| quantity, test half | value |
|---|---|
| F = log10 r_o − log10 r_c, the rank the carrier's own move buys beyond the others' move, mean over 71 births | +0.95 decades [0.73, 1.18]; positive in 51 of 71 |
| the same for M2's twins; for the distinct-token twins | +0.77 [0.56, 0.99]; +0.94 [0.68, 1.22] |
| the leaders' loss of rank by the birth checkpoint, real births; M2's twins; distinct-token twins | 0.94 decades; 1.16; 1.06 |
| hits (top 10 under B one checkpoint before): real; M2's twins; distinct-token twins | 42 of 71; 53 of 70; 41 of 67 |
| validation of the gradient instrument: Spearman between the first-order prediction and the measured ablation | 0.04 (freeze half 0.10; bar 0.7); sign agreement 0.50; the measured change is 27 times the prediction at the median |

- The carrier wins by its own rise. Holding the others at their earlier values, the carrier's own move takes it about a
  decade further up the layer's ranking than the others' moves do with the carrier held. Twins show the same (+0.77 and
  +0.94), so this is what the dating rule gives any pole whose median on a group of its contexts crosses its own threshold:
  a real birth is not special here, and the spec reports the twins beside it for that reason.
- The leaders of real births fall less than the leaders of M2's twins (0.94 against 1.16 decades, paired −0.23 [−0.41,
  −0.05]) and about as much as the leaders of the distinct-token twins (−0.11 [−0.29, +0.06]). So the "competition" at a
  real birth is not a release: the poles that led the cluster's contexts keep more of their rank than a selected top 20
  regresses, and the difference from M2's twins is again the repeated-token difference. The decision table's reading: the
  rise decides the rank; the leaders stay.
- The gradient instrument failed its validation, so the two gradient hypotheses do not count. Zeroing a neuron's
  down-projection input at the positions where it is among its own strongest activations changes the window loss by 0.64
  nats at the median, 27 times what the local derivative predicts, with the sign of the change agreeing with the prediction
  in half the cases, chance level. The ten read examples show the measured change up to 8,000 times the prediction. The
  first-order approximation does not hold for removing a neuron at its strongest positions; whether the gradient itself is
  right at small perturbations was not checked on the instance (the dry run checked it on a tiny model; G-ABL-ID checks
  only the clean row). As exploratory numbers, the carrier's specific pull sits at the median of its competitors (u_S 0.53)
  and the pull does not foretell who gains rank (Spearman −0.01); they carry no weight.
- M2's follow-up landed: twins matched on the number of distinct tokens are named one checkpoint before as rarely as the
  real carriers (41 of 67 against 42 of 71), and less often than M2's twins (−0.13 [−0.25, −0.01]). Token repetition
  explains M2's gap, as pre-registered with prior 0.50. M2's BELOW_SELECTION now reads: repeated-token contexts are
  contested, and a cluster of distinct tokens is named about as well as a random group of the pole's own contexts.

## The three dimensions

- Predictive: pre-registered, frozen on the freeze half, unblinded once. H_cf landed on the branch the author gave 0.75
  (OWN_RISE), H_dt on the branch given 0.50 (SUPPORTED), H_away_sel_pb on the branch given 0.20 (BELOW). G-VAL was given
  0.6 to pass and failed. The instrument's failure is recorded as such, not as a result about the signal.
- Novel: the counterfactual decomposition of a rank change into the pole's own move and the field's move, and the
  distinct-token twin, were not in the literature the spec searched; the finding that a late-born cluster's carrier wins by
  its own rise while its leaders keep their rank is new at 7B. The gradient part adds a negative about the instrument:
  activation-times-gradient does not predict single-neuron ablation at a neuron's strongest positions in this model.
- Practical: none measured; the practical answer is that the gradient-per-unit-activation score cannot stand in for
  ablation here.

## Gates and limitations

- Every one of the 73 gates passed and every failing input failed; the forward equals M2's banked vectors at every point;
  the two identity pairs are exact under the deterministic backward; the cross-device residual at the Blackwell-banked
  points is reported as in M2 (over the tolerance at T_13 and T_7–T_11), with the poles' own top-50 values agreeing
  everywhere.
- The executor added one setting the head had not ordered: bf16 reductions inside the math attention kernel, to fit the
  L40S's memory in the gradient pass; it changes the gradient by about 1 % in relative norm on its CPU check and is stated
  in the spec as the executor's. It does not bear on the validation failure, which is a factor of 27.
- The gradient labels' failure means the question "did the signal at T_(k−1) already favour the carrier" is unanswered,
  not answered no. The cheap next instrument is the spec's own: finite differences at small scale (a 10 % reduction of the
  neuron's input at the same positions, about 0.3 GPU-h), which separates "the gradient is wrong" from "the perturbation is
  too large"; and ablation per pole for the competitor comparison (about 1 GPU-h). To be decided with the rest of J12.
- Two N12 attempts, about 2.5 GPU-h in all; the first stopped on the nondeterministic GPU backward, a gap the dry run's
  exact CPU backward could not show.

## Decision

Closed as the verdict states, pending the reader's recount. Under the spec's decision table: the weight change behind the
carrier's rise is the next object (N1, with the onsets located first), and the gradient readout needs the small-scale
check before any further use. The abstract carries: the carrier wins by its own rise (about a decade of rank beyond the
field's move), the leaders keep their rank, token repetition explains the earlier twin gap, and the local gradient does not
predict single-neuron ablation at a neuron's strongest positions.

## Addendum after the independent read (head, 2026-11-06 14:10Z)

The reader (`READ_n12.md`, branch n12-read, merged) reproduces every label and all but two of 90 fields to 4e-16 (the two
differ by a tie-break and a bootstrap seed), matches the job's activations to M2's banked vectors bit for bit on 40 million
values, and recomputes B with its own percentile code. Two findings sharpen the reading above.

1. OWN_RISE is close to forced by the dating rule. In all 442 units of every kind B rises from T_(k−1) to T_k, so r_c ≤ r0
   follows from the rule, and F > 0 then needs only r_o ≥ r0, which holds in 77–91 % of units by kind; F's size follows
   the starting rank (Spearman 0.89–0.94), and at a matched starting rank real − dt is +0.017 [−0.098, +0.129]. The spec's
   §1.3 sentence "its sign is not forced by the dating rule" is contradicted by the records. So the label OWN_RISE says
   what the rule makes any dated pole do, and the informative realized findings are the other two: the leaders of real
   births keep their rank (the drop against M2's twins is mostly token repetition, −0.387 on repeated-token births against
   −0.047 on varied ones, pooled; a remainder against the distinct-token twins survives pooled, −0.179 [−0.306, −0.056],
   exploratory, while the test half alone is AT_SELECTION), and token repetition explains M2's gap (H_dt).
2. G-VAL measured the ablation's resolution, not the gradient's validity. The measured loss change behaves like the bf16
   forward's rounding noise: it is unrelated to the predicted size (Spearman 0.01) and to the activation removed (−0.02),
   grows with the number of loss terms after the ablated positions (0.24), and its ratio to the prediction falls as the
   activation grows (−0.29), the opposite of a large-perturbation failure. With the measured noise, ρ could not reach 0.7
   even for effects ten times larger. So the sentence above, "the first-order approximation does not hold for removing a
   neuron at its strongest positions", is withdrawn: the window-summed loss in bf16 cannot resolve a 0.02-nat effect, and
   a small-scale finite-difference check in the same forward would fail for the same reason. The next instrument, before
   any gradient readout counts: the ablation's loss change measured in fp32 (and on the exemplar's own next-token loss as a
   second readout), about 0.3–0.5 GPU-h, registered as a J12 follow-up. Read as exploratory, the gradient readouts would be
   H_signal REFUTED (u_S 0.527 at or above the bar), the selection contrasts AT, and H_bridge UNDECIDED.

The blind protocol was broken before the reader computed anything (a commit subject on the branch names every label),
disclosed. The reader also lists six code–spec differences, none changing a label; the §1.3 sentence is corrected in the
record by this addendum.

---

**Postscript, 2026-11-07 (Fable), a retag.** The sentence above, "activation-times-gradient does not predict single-neuron
ablation at a neuron's strongest positions in this model", is retagged. N2V (`studies/next/n2v/verdict_n2v.json`, `READ_n2v.md`,
`READ_head.md`) read the same 612 pole ablations in fp32: N12's banked bf16 prediction ranks the fp32 zero-ablation loss change with
Spearman 0.960 [0.915, 0.990] on the test half (0.971 on the freeze half), and the pole with the largest predicted change is the
pole with the largest measured change in 65 of 71 test units. The cause of the 0.04 above was the bf16 resolution of the summed
loss (median measured change 0.65 nats in bf16 against 0.0224 in fp32, predictions 0.02). The gradient labels now count at their
printed values: H_signal REFUTED, H_sel_signal_pb AT, H_sel_signal_dt AT, H_bridge UNDECIDED (N2V-5). What stands of the sentence:
the bf16 prediction itself misses in about 10 % of pairs, where the fp32 forward-mode prediction is right in 26 of 28 (N2V-6).
