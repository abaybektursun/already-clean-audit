# M2, the head's read (Fable, the different-model read, 2026-11-06 08:15Z)

**Answer.** The verdict stands as written: `H_B SUPPORTED; H_sel_B BELOW_SELECTION; H_A SUPPORTED; H_AB LOSES`. My own code
(`read_head/recount.py`, numpy only, reading the job's per-record ranks `banks/score-06/m2/ranks_test.jsonl` and
`ranks_freeze.jsonl` and nothing from `infra/pg/m2.py`) gives every count, precision and Wilson bound of `verdict_m2.json` to
the printed digits, and the two paired differences with the same bounds under my own bootstrap seed (output in
`read_head/recount_output.txt`). The freeze half matches the DISCOVERY block (B 39 of 82, A 7 of 82, twins under B 58 of 79).

## What the numbers say, in plain words

The question: one checkpoint before a cluster of a neuron's strongest contexts first appears (its birth), can the eventual
carrier pole be named among the 22,016 poles of its layer? The score B is the pole's own percentile of its median activation on
the cluster's contexts at that earlier checkpoint; a hit is a rank in the top 10.

| quantity, test half (71 births) | value |
|---|---|
| carrier in the top 10 under B, one checkpoint before birth | 42 of 71 = 0.59 [0.48, 0.70]; 18 at rank 1; median rank 6 |
| chance; matched-pole floor; the bar | 0.00045; 0.012; 0.012 |
| the same at the birth checkpoint (the ceiling) | 65 of 71 = 0.92 |
| twins (a random same-size group of another pole's own future top contexts, dated to the same point) | 53 of 70 = 0.76 |
| carrier minus twin, paired | −0.17 [−0.29, −0.06] |
| the free banked-array score A (exemplar positions already in the pole's top-50) | 13 of 71 = 0.18; 57 of 71 have no overlap at all |
| A minus B, paired | −0.41 [−0.52, −0.30]; A never hits where B misses |

- The carrier can be named before birth. From its current usage of the cluster's contexts, the eventual carrier is already
  among the ten most responsive poles of its layer in six births of ten, and the single most responsive in one of four.
  Against chance (0.00045) and the activation-matched floor (0.012), this is not close.
- But that is not a property of coherent late-born jobs. A random group of another pole's own future top-50 contexts, dated to
  the same checkpoint by the same rule, names its pole more often (0.76 against 0.59). The selection of a pole's future top
  contexts carries the pre-birth signal on its own; real clusters carry less of it.
- Where the gap comes from (`read_head` extra check, test half): the carriers' own percentiles at the pre point are as high as
  the twins' (median 0.99836 against 0.99783), so the carrier is not weaker on its own contexts; the difference is in the
  competitors. On real clusters' contexts, more other poles of the layer stand above the carrier (mean 5.2 poles of the top
  10 above it, against 3.5 for twins). The two sampled births in `raw_examples` show it concretely: other poles already hold
  the cluster's positions in their own top-50 one checkpoint before the birth (NPR "Copyright": L10N3510+ holds all 5; SQL
  "CHARSET": two poles hold 4 and 3, the carrier none). So the plain reading of BELOW_SELECTION: a coherent cluster's contexts
  are ones that other poles also respond to strongly, and the eventual carrier does not yet lead that competition one
  checkpoint before; a random grab-bag of one pole's contexts has no competitor. This is the decision table's third row
  (§1.2): other poles held the job's contexts first; the competition instrument (N12) moves up.
- The twin's confound, as the executor recorded: twins are typical groups of their pole while real births were selected for
  being later than chance for theirs. The dating rule fixes both at "below the pole's own threshold at T_(k−1), above it at
  T_k", and the own-percentile comparison above shows the carriers are not behind on their own contexts, so lateness does not
  explain the gap through the carrier's side. Whether lateness changes the competitor side (late clusters landing on contested
  contexts) is the same question as the competition reading; this run cannot separate "coherent" from "late". It is a stated
  limitation.
- The free score A names the carrier in 18 % of births and never where B does not; 57 of 71 carriers hold none of the
  cluster's positions in their top-50 one checkpoint before. The banked arrays cannot replace the forward for J12's competitor
  sets (the plan's practical question): a negative, practical answer.
- By birth point (real hits / twin hits of n): T_11 9/11 of 12, T_12 9/10 of 13, T_13 10/11 of 12, T_14 3/5 of 13, T_15 4/5
  of 7. The late points are harder for both.

## The three dimensions

- Predictive: pre-registered, thresholds frozen on the other half, unblinded once. H_B landed (prior 0.97, at 0.59 inside
  the prior's 0.3–0.7). H_sel_B landed on the branch the author gave 0.25 (BELOW), against AT 0.45 and ABOVE 0.30: the
  selection control reversed the hoped-for reading, which is the control working.
- Novel: the ranking of the eventual carrier among all poles of a 7B layer before a dated birth, and the selection control,
  were open as far as the spec's search went; the finding that the selection baseline exceeds the real births is the novel
  part, and it is a negative for "a pre-birth signal specific to jobs".
- Practical: the banked-array score does not substitute for the forward (H_AB LOSES); J12 budgets one forward per point.

## Gates and limitations

- Every gate passed on the counting run and every failing input failed; K0 and the identity pair bit-identical; the
  construction control 71 of 71 under both scores with the opposite pole at 0; the birth-point ceiling above the pre-birth
  value, so no ABOVE_CEILING.
- The cross-device residual between the Blackwell-harvested bank (T_7–T_14) and the L40S forward exceeded the A2 tolerance
  at T_13, T_11, T_10, T_9 and T_7 on 1–2 % of the pre-birth exemplar entries (concentrated on a few poles with large
  activations), while the poles' own top-50 values agreed everywhere (G-TOPK p99 ≤ 0.016). B ranks poles under one forward,
  so the residual enters all poles alike; stated as a limitation, as A3 requires.
- Six attempts were needed; the first five stopped on defects that the banked data could have shown before launch
  (cross-device agreement form, degenerate poles, a failing input with no power at early checkpoints). The rule for the next
  specs: the offline gate reads every banked control and evaluates both sides of every gate on it before launch (recorded in
  the program journal, 2026-11-06 00:50 CST).
- The independent reader's recount (`READ_m2.md`, branch m2-read) is pending at the time of this read; this read is the
  different-model read.

## Decision

Closed as the verdict states, pending the reader's recount. Next under the spec's decision table: N12 (competition) moves up
in J12; J12 budgets one forward per scored point. The abstract carries: the carrier is nameable before birth from current
usage (0.59 against 0.0005 chance), but a selection baseline names its pole more often (0.76), so the pre-birth signal is not
specific to coherent jobs and points to competition among poles for the same contexts.

## Addendum after the independent read (head, 2026-11-06 08:40Z)

The reader (`READ_m2.md`, branch m2-read, merged) reproduces every count, rate, bound and label, recomputed A for all 757
records from the banked top-50 arrays and B for 155 records from the job's vectors and the banked histograms (largest
difference 1.1e-16), and reproduced the births, the split, the twin seeds and the raw examples' tokens. Blindness was partly
broken (commit subjects on the branch name the labels), disclosed.

Its exploratory finding sharpens the competition reading above. About half the real births are clusters whose exemplars
repeat one or a few tokens (" CHARSET", " Copyright"); the twin construction almost never produces such a group (16 of 149
twins against 78 of 149 births). On varied-token births carriers and twins hit alike (57 against 58 of 71, difference
−0.014 [−0.113, +0.085]); on repeated-token births the carrier loses (22 against 53 of 78, −0.397 [−0.513, −0.282]). Giving
a carrier its twin's competitors raises its hits from 41 to 49, and lateness does not explain the gap (twins as late as the
median real birth still hit 29 of 36). So the contested contexts are the repeated-token ones: many poles respond to a
repeated token, and the eventual carrier does not yet lead among them one checkpoint before. The split was chosen after
looking at the data, so it is not a registered result; the frozen label BELOW_SELECTION stands, and the registered follow-up
is a twin matched on the number of distinct tokens (one forward per point, about 0.7 GPU-h, to be specified with J12). One
measure of coherence (the W4 judge's accuracy on the cluster's label) points the other way, unexplained; stated as open.
