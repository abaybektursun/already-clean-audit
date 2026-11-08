# score-01 stopped at I-GRAD (T_13), 2026-11-06T10:32:22Z

**The job stopped on the second clause of I-GRAD: 4 units' u_S changed between two gradient passes over the same
windows and weights; the rule allows 0.** The first clause passed: Spearman of the two passes' G 0.999642 against the
bar 0.999. Source: `metrics.json` → `gates.I_GRAD` (this folder) and `job.log` line 164.

| quantity | value | rule |
|---|---|---|
| ρ (Spearman of G, pass 1 against the reversed pass 2) | 0.999642 over 7,419,392 entries | ≥ 0.999: pass |
| exact share of G entries | 0.2129 | reported |
| units whose u_S changed | 4 of 98 pre-role units at T_13 | 0: **fail** |
| failing input (axis rolled by one) | ρ 0.000401 | fails, as registered |

What it shows. The batch-1 backward on the L40S is not bit-deterministic: 79 % of the captured gradient entries differ
between two passes. The forward is: I-ID at the same point was exact (8,035,840 activations, exact share 1.000, 0
flips of top-10 membership). The difference in G is small in rank terms (ρ 0.99964), but it moves at least one
competitor across the carrier's S in 4 units, so u_S (steps of 1/20) changes there.

Could the cause have been checked from banked data? **No.** No gradient of this model was ever banked. The dry run's
backward ran on CPU, where it is exact. The spec stated the risk (§1.4: "the sdpa backward on the L40S is not shown
deterministic anywhere in the repo", prior 0.6 for bit-identity, 0.95 within the rule). The dry run's failing input
for I-GRAD was a noisy second pass (Spearman 0.696, 29 flips); no planted input had a near-exact pass (ρ ≈ 0.9996), so
the "0 flips" clause was never shown to have room against GPU nondeterminism. It could have been checked before the
full run only on a GPU, with two backward passes of a few windows.

Not recorded by the job: which 4 units flipped, their half, and the size of the change (the gate writes only the
count); the second pass's gradients are not banked.

State on S3 (`results/next_n12/score-01/out/score/n12/`): `arrays/t14`, `t15`, `t13` (B, a, pull), partial
`ranks_freeze.jsonl` and `ranks_test.jsonl`. No `n12_*.jsonl` (written after the last point). Nothing of the job's
records was opened; `ranks_test.jsonl` was not downloaded.

Before this gate every gate at T_14, T_15 and T_13 passed with its failing input failing (K0 and G-M2V exact, G-ABL-ID
0.0 nats). The run DOES NOT COUNT. No relaunch without the head.
