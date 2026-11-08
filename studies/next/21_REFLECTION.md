# 21. Reflection on the already-clean program

Written 2026-11-08 by the head (Fable), after the sixth study closed and the abstract passed its fourth check. This is
not a result. Every number below names its file; most are copied from the evidence table of `20_ABSTRACT.md`, which the
fresh reader verified (`20_ABSTRACT_CHECK.md`). The map of the program is `00_INDEX.md`; the current state is `HANDOFF.md`.

## 1. What was asked and what was delivered

The brief: run the highest-leverage questions left by the already-clean neuron study (OLMo-3 7B, 16 training
checkpoints) with Opus workers doing the work and the head orchestrating and reviewing, every claim pre-registered with
the observation that refutes it, every result read blind by a different model, and end with an abstract of at most 300
words plus an evidence table.

Delivered: six studies closed (M1, M2, N12 with N2V, N1, N13, N6), two pilots decided (J8 passed, J10 NO-GO), the
abstract at 300 words with five results and one study that does not count, and a check file in which every number in the
abstract was traced to its record by a fresh reader in four passes (175, 25, 9 and 31 items). Nothing is running.

## 2. Predictions against outcomes

Each spec logged the author's prior before any data (§1.4 of each spec in `07_PICKS/`). The table puts the registered
probability beside what happened. This is the predictive dimension of the program, scored honestly.

| study | registered prior (file, §1.4) | outcome (file) | landed? |
|---|---|---|---|
| M1 | H_lead SUPPORTED 0.65; H_order SUPPORTED 0.4; H_specific SUPPORTED 0.6 | H_lead SUPPORTED; H_order UNDECIDED; H_specific SUPPORTED (`m1/verdict_m1.json`) | two of three on the likelier branch |
| M2 | H_B SUPPORTED 0.97; H_A 0.9; H_AB LOSES 0.85; H_sel_B AT 0.45, ABOVE 0.30, BELOW 0.25 | H_B, H_A SUPPORTED; H_AB LOSES; H_sel_B BELOW_SELECTION (`m2/verdict_m2.json`) | three of four; the twin baseline beating the carrier (76 % against 59 %) was the 0.25 branch |
| N12 | H_cf OWN_RISE 0.75; H_away_sel_pb BELOW 0.20; H_signal SUPPORTED 0.35 / UNDECIDED 0.35 / REFUTED 0.30; G-VAL passes 0.6; H_dt SUPPORTED 0.5 | OWN_RISE; BELOW_SELECTION; G-VAL failed in bf16 (Spearman 0.100 and 0.043); H_signal REFUTED after the fp32 re-read; H_dt SUPPORTED (`n12/verdict_n12.json`, `n2v/verdict_n2v.json` n12_reread) | the instrument failing (0.4) and REFUTED (0.30) were the less likely branches |
| N2V | G-VAL window PASS 0.6; above 0.999 0.1; G-FD PASS 0.8; G-VAL own 0.7; G-CAUSAL bitwise 0.9 | all three PASS (0.960, 0.987, 0.986); not above 0.999; G-CAUSAL not bitwise across two paths, within tolerance (`n2v/verdict_n2v.json`, `n2v/READ_n2v.md`) | the mainline landed; bitwise equality across two computation paths failed again, as it had in A30 |
| N1 | NEITHER 0.45; a BELOW label on either side 0.1 | NEITHER, with the input side BELOW_SELECTION (`n1/verdict_n1.json`) | the headline landed; the BELOW label was the 0.1 branch and is fragile (two births, freeze half at selection level) |
| N13 | G 15–35; L 80–92; top_3 % reaches H 0.6; top_1 % 0.35; dare_0.1 % DAMAGED 0.7; rndT within δ of top 0.5; H_artifact refuted 0.6 | G 34.8; L 92.0; no top set reaches H 74.6 (63.8 at best); dare_0.1 % 85.2, not damaged; rndT equal to top at every k; H_artifact INDETERMINATE (`n13/verdict_n13.json`) | the two bands landed at their edges; the three central priors on top-k and DARE were wrong, and the wrong one about DARE is the program's most useful result |
| N6 | UNDECIDED including no power 0.30; H_age SUPPORTED 0.25 | DOES NOT COUNT: no power (ratio 1.186, bar 0.5) and the positive control inverted (2 of 8) (`n6/thresholds.json`, `n6/banks/L2a-01/metrics.json`) | the no-power branch landed; the control inversion had no prior at all |
| J10 / N22 | at least 100 of 1,200 answered 0.80 | 7 of 1,200; NO-GO (`n22/verdict_n22_t15.json`) | the 0.20 branch |

Reading the table: the mainline predictions landed in every study that counted. The surprises were all on the control
side: a baseline that beat the thing it was a baseline for (M2), an instrument that did not work in the precision it was
run in (N12), a sparse random estimate that worked where the prior said it would be destroyed (N13), and a positive
control that ran backwards (N6). That is where the information was.

## 3. What the method bought

Gates that fired before a number was quoted: N6's L1-01 at G-POLE; N12's G-VAL in bf16; N6's G-PC and G-POWER; J10's
go/no-go. Each stopped a result that would otherwise have been written up.

What the blind readers caught, none of which the executor or the head had seen (paths in each READ file):

- N2V (`n2v/READ_n2v.md`): the regression slope of 0.837 against the finite difference's 0.988 came from a few pairs with
  a large second-order term; in the tenth of test pairs where the bf16 and fp32 predictions disagree most, fp32 is closer
  to the measured change in 26 of 28; a defect in a banked field that no rule reads (`sum_abs_h`).
- N13 (`n13/READ_n13.md`): answer format does not make the gap (boxed answers 92.6–94.4 against 65.5–82.4); the DARE
  estimate carries 33 to 1,000 times the energy of the whole change and still lands within 1.2 of the whole MLP family;
  one GSM8K contrast (top against rndT) flips under 18 of 100 seeds, and both small contrasts are marked fragile.
- N1 (`n1/READ_n1.md`): the headline's margin is two births; the freeze half alone sits at selection level.
- The abstract checker (`20_ABSTRACT_CHECK.md`): a slot-shift bound that was quoted one way and measured another (0.36
  of a slot); a GSM8K number attributed to the wrong arm; the top-pole share (0.915) conflated with the full signed order
  (0.873 for the fp32 pass); "per neuron" where the record says per pole; a closing sentence that claimed more than the
  tested subsets.

Without the readers, four wrong numbers would be in the abstract now.

## 4. What broke, and the rule each break left

| what broke | where | the rule now written |
|---|---|---|
| first-order predictions in bf16 ranked zero ablations at Spearman 0.10 | N12 (`n12/READ_n12.md` G-VAL) | compute first-order predictions in fp32, forward mode where the backward does not fit; report the finite-difference check beside the zero ablation (`n2v/READ_head.md`) |
| bit-for-bit equality across two computation paths failed | N13's A30 on summation order (`HANDOFF_2026-11-07.md`), N2V G-CAUSAL (`n2v/READ_n2v.md`) | freeze a tolerance on a control pair; never gate on bitwise equality between paths |
| the positive control inverted: the random edit moved the residual 1.3–2.5 times more than the neuron's own write | N6 (`n6/STATE.md`, "What L2a measured") | match controls on realized residual change, not on weight energy; audit realized effects, not intended ones |
| the power preview said 0.435 and the run said 1.186 | N6 (`n6/STATE.md`) | preview power at the registered window count, or scale both the noise and the effect with the count; a preview that holds one side fixed is not a preview |
| the job's remaining-time field was one layer stale | N6 L2a (`n6/STATE.md`) | a projection is read from the rate and the count, not from a field written by the job |
| journal stamps written ahead of the clock by up to 1 h 27 min | head's journal, 2026-11-08 | every stamp is read from the clock at the moment of writing; the correction is dated in the journal |
| a commit landed after a failed word-count assert | abstract draft 3 (`JOURNAL.md` 2026-11-08) | chain the check and the commit with `&&`; the check runs before any write |

## 5. Where the head misjudged

- **N6 ran its positive control inside the main run.** The 24 control edits are about a fifth of the run (24 of 113
  edits in 18.9 GPU-h; `n6/banks/L2a-01/metrics.json`, `edits.jsonl`), about four GPU-hours. Run first, on their own,
  they would have cancelled the study before the other fifteen (`infra/aws_pg/operations/2026-11-07_next_n6.md`). The
  rule from now on: the positive control is its own first job, and the main run waits for it.
- **N12 was run in bf16 with the prior at 0.6 that the instrument would validate.** The fp32 instrument then had to be
  built and run as N2V. A small fp32 validation on a few units before the main run would have cost a fraction of that.
- **A practical card was proposed to the user before the deciding measurement.** The top-1 % transplant idea went out on
  the Friday; N13's result withdrew it two days later (top 0.1–3 % reads 57.6–63.8 against the base's 57.2). Practical
  proposals wait for the measurement that decides them.
- **Cost concentrated in one study.** N13 took 102.6 of the program's 147.8–148.1 GPU-h, about $249 of $352 (`00_INDEX.md`, cost table). In
  hindsight the allocation was right: it produced the one result that changes what a practitioner would do. But the
  decision to spend was made on the top-k hypothesis, which turned out wrong; the value came from the control arm.

## 6. What stands out as worth building on

- **The RL math gain is diffuse and cheaply estimable.** A random 0.1 % of MLP entries, scaled by 1,000, reads 85.2 of
  the RL model's 92.0 on MATH-500 (`n13/verdict_n13.json`); attention with norms alone reads 83.0. No 0.1–3 % neuron
  subset does any of this. This is the practical result, and the rank test of the delta is the next measurement.
- **A gradient instrument that works, with its failure mode measured.** In fp32 it ranks 284 zero ablations at Spearman
  0.960 (`n2v/verdict_n2v.json`); in bf16 it does not work at all. Both numbers are now banked side by side.
- **A twin baseline that beats the prediction it was meant to floor.** A dated random group of another pole's future
  contexts names its pole more often than the carrier's own contexts do (`m2/verdict_m2.json`). Any future "predict the
  carrier" claim here is measured against that floor.
- **A design constraint for edit-cost studies.** A control matched on weight energy is not matched on effect; N6's banks
  show the gap in every control layer. The next edit-cost design matches on realized residual change first.

## 7. What is next

The abstract names it: a rank test of the RL delta per MLP matrix (rank 16, 64, 256) against random subspaces of equal
rank and energy, under N13's harness, 12–16 GPU-h on MATH-500 and 15–20 with GSM8K on the rank-64 pair
(`20_ABSTRACT.md`, "The highest-leverage next step"). It decides whether the diffuse gain is low-rank, which is the
question a practitioner needs answered before compressing or transplanting it. The decision is the user's.

## 8. Housekeeping done with this reflection

Merged branches and their worktrees removed, the branch record kept, the old programs' result directories ignored rather
than deleted, the map and the handoff written: `11_HOUSEKEEPING.md` (2026-11-08 section), `00_INDEX.md`, `HANDOFF.md`.
