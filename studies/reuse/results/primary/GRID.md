Jobs over 150: run 150, not run 0, cancelled at birth 2, live at birth 148, live at both checkpoints 147.

# REUSE primary verdict: SPEC §11 and §13.2 filled

Every number is in `studies/reuse/results/primary/verdict.json` with the artifact it was read from. Live = read at the load and not cancelled. Intervals: 95% percentile bootstrap over neurons, 2,000 draws, seed 0; the planted median's over pairs within tercile. m 50, direct ranking, mean replacement.

| prediction | predicted | observed | verdict |
|---|---|---|---|
| R1 H_reuse | no point registered | median I at birth 0.226 [0.170, 0.308] on 148 live jobs (102 neurons) against planted median 0.174 [0.117, 0.221] | refuted: the interval overlaps the planted median's |
| R2 H_generic | no point registered | median cross − foreign at birth 0.084 [0.060, 0.122] | specific: the interval excludes 0 and is positive |
| R3 H_consolidate | no point registered | median I_T15 − I_birth 0.098 [0.068, 0.128] on 147 jobs live at both | consolidation: the interval is above 0 |
| Positive control floor | ≥ 0.25 at frozen m | median own fall at birth 0.528 on 150 jobs; failing input median random 0.007; identity 0.000 | pass |
| Room | ≥ 0.20 | 0.329 on 65 setting splits (0.313 on the 50 not shared with a primary ceiling), line 0.200 (m_r 0.102); failing input, planted labels shuffled: -0.202; identity null | pass |
| R4 H_related-share | no point registered; coordinator's expectation at freeze: near 0.1, interval may include 0 | median I related − unrelated 0.227 [0.133, 0.297] (56 related, 92 unrelated live; medians 0.382, 0.155) | holds: the interval is above 0 |
| Cancel share at birth | under 40% of 150 | 2 of 150 = 0.013; failing input, own and random swapped: 1.000; identity null; sets of 7 contexts from no job not run | pass |
| Identity | `\|Δ\| ≤ 1e-3` on every context at every load | max \|Δ\| 0.0 over 3869 contexts at 9 loads; failing input (own drivers, m 50) over 1e-3 on 3869 of 3869 contexts | pass |
| Known number | ≤ 1e-2 at each of 11 loads; the failing input fails at each | 9 loads (t15, t06, t07, t08, t09, t10, t11, t12, t13; a checkpoint where no job is born is not loaded); max ratio 1.33e-03; min failing-input ratio 2.96e-02; completeness max 1.28e-03; loads that count 9 of 9 | pass |

> Coordinator's expectation at freeze, after S1–S3 and before any primary forward, no rule attached: median I at birth near 0.45 against planted 0.174 and ceiling 0.788, so R1 not refuted and the position near 0.45; cross − foreign near 0.2, so R2 positive; I at T_15 near 0.6, so R3 positive; related minus unrelated near 0.1 with an interval that may include 0, so R4 is the one most likely to fail. The six sniper jobs gave I 0.66 at T_15 and 0.46 at T_10 at m 50, and that is what these guesses rest on.

(SPEC §11, the coordinator's expectation at freeze, quoted unchanged; no rule attached.)

