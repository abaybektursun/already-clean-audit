# INSTRUMENTS: claims register (one line per load-bearing claim; status tags: HOLDS, REFUTED, NOT SEPARATED, CANCELLED, EXPLORATORY, PENDING; a claim contradicted by its artifact is retagged the day it is found)

| id | claim | status | artifact |
|---|---|---|---|
| I1 | E0's same-token positive control fails on its draw (20 of 30); R0 cancelled | CANCELLED (gate) | `results/e0/verdict_e0.json` |
| I2 | Low-layer clusterer sets read coherent more often than deep ones (0.486 vs 0.289 per pole) | EXPLORATORY | `results/e0/verdict_e0.json`, recount in the journal 15:50 CST |
| I3 | The reader is stable (27 of 30 on SPEC3's sets; 28 labels identical); the same-token-same-document control passes (29 of 30) | HOLDS | `results/e0b/verdict_e0b.json` |
| I4 | E1's gates pass on the real model; known numbers 0.163, 0.067, 0.294 reproduce | HOLDS | `results/e1/verdict_e1.json` |
| I5 | Resampling shrinks the exclusion gap from 0.096 to 0.055 (D1 0.041 [0.005, 0.084]) | NOT SEPARATED at 0.049 | `results/e1/verdict_e1.json` R1 |
| I6 | The median value leaves the gap unchanged (H_value refuted for the median) | REFUTED (H_value) | `results/e1/verdict_e1.json` R1m |
| I7 | With the global exclusion the null is 0.147 (resample), 0.067 (mean), 0.049 (median), 0.007 (zero); own falls unchanged | HOLDS | `results/e1/verdict_e1.json` A1; read item 2 |
| I8 | The real cases' ratio is 0.293 under zero and 0.300 under the median (0.294 mean), 0.362 under resample | HOLDS | R3; `cases_I` |
| I9 | Exclusion and resampling do not agree as routes (R2 0.128 [0.079, 0.177]) | HOLDS (disagree) | R2 |
| I10 | 91 of 150 cases clear their rule's line under zero with exclusion, 57 of 149 under the mean | HOLDS | `results/e1/read/READ_e1.md` item 14 |
| I11 | The members' push is unchanged by resampling (1.06 [0.78, 1.21]) and not predicted by the direct marked-token term (median 0.0004 vs 0.066; r 0.29) | HOLDS | `members_ratio`, R4 members; read items 10, 17 |
| I12 | On the own arm the direct term predicts the fall with r 0.77 and over-predicts by a fifth | HOLDS | R4 own; read item 11 |
| I13 | "Zero removes a constant push" as a mechanism | HYPOTHESIS (untested; r −0.003) | read item 12 |
| I14 | E3's gates pass; the clusterer's self-agreement ceiling 0.645 (0.763 low-layer) | HOLDS | `results/e3/verdict_e3.json` |
| I15 | The driver partition sees the clusterer's split above chance (0.253 vs 0.061; 79 percent of poles) | HOLDS (H_boring refuted) | `results/e3/verdict_e3.json` |
| I16 | …but no more than the token partition (R1 −0.004 [−0.040, +0.024]) | NOT SEPARATED | R1 |
| I17 | The token predicts held-out labels better than the drivers (R2 −0.117 [−0.142, −0.093]) | REFUTED (H_predict) | R2 |
| I18 | Ward linkage on the same vectors sees the split beyond the token (D +0.155 [0.118, 0.191]) | EXPLORATORY (post hoc rule) | `results/e3/read/READ_e3.md` item 10 |
| I19 | The detector's silhouette clause has no room; the flag is the ARI clause (0.81 and 0.74) | HOLDS (instrument) | `results/e3/verdict_e3.json` |
| I20 | Driver-defined sets read as one role less often than the clusterer's (R3 −0.154 [−0.193, −0.115]) | HOLDS | `results/e3r/verdict_e3r.json` |
| I21 | On multi-token sets alone the difference is −0.124 [−0.162, −0.086] | NOT SEPARATED at −0.10 | read item 13 |
| I22 | E1b: the instrument works on gemma-3-4b-pt and Qwen3-4B-Base (own 0.647, 0.570; random ≈ 0) | HOLDS | `results/e1b/verdict_e1b.json` |
| I23 | Neither model has a generic term (G(mean) 0.005 and 0.002; null ≈ 0.004 under every value); H_model refuted | REFUTED (H_model) | `results/e1b/verdict_e1b.json` R5 |
| I24 | E2: the product ranking is stable across halves (overlap 106 of 128) and recovers 72 of the editor's 128 (null 8.7) | HOLDS | `results/e2/verdict_e2.json` reported |
| I25 | E2: the frozen room gate fails on its cut clause (0.96); every rule cancelled under it | CANCELLED (frozen gate) | `results/e2/verdict_e2.json` |
| I31 | E2 (am. 39): the editor's edit lowers the repetition share by 0.317 [0.247, 0.383]; the direction-only 64 by 0.307; R_E − R_D +0.010 [−0.057, +0.073] | NOT SEPARATED (R4) | `results/e2/read_fable/recount_e2_fable.txt` |
| I32 | E2 (am. 39): Q = R_P64/R_E 0.80 [0.61, 1.04]; R_P64 − R_D −0.053 [−0.127, +0.023] | NOT SEPARATED (R2, R3) | same |
| I33 | E2 (am. 39): the dose is monotone (0.090, 0.253, 0.377) and P256 exceeds the editor's effect | HOLDS (R5) | same |
| I34 | E2 (am. 39): R_KL − R_norm −0.013 [−0.083, +0.050]; random columns give 0.08 to 0.09 at either realized change | NOT SEPARATED (R6) | same |
| I35 | E2: the activation-only ranking gives 0.237 at Δloss 0.0010 (E 0.0084) | HOLDS (reported) | `results/e2/verdict_e2.json` per_arm |
| I26 | E4 (exploratory, Ward clusters, zero with the global exclusion): the shared-driver fraction forecasts the collateral ratio, r(I, s_proxy) 0.808 [0.759, 0.849], r(I, s_true) 0.868 | EXPLORATORY (holds by its rule) | `results/e4/verdict_e4.json` R1 |
| I27 | E4: the fit I = 0.040 + 1.553 s_true against the program's 0.08 + 1.45 (slope interval [1.453, 1.671]) | EXPLORATORY ("does not generalize" by 0.003 on the slope) | `results/e4/verdict_e4.json` R2 |
| I28 | E4: the |ā| variant of the proxy predicts at 0.345; the side's own mean activations are what the proxy needs | EXPLORATORY | `results/e4/verdict_e4.json` reported |
| I29 | E1b (read): gemma's outlier members carry a tail term (p95 0.235 → 0.060 with the exclusion) and a zero-value term (p95 0.711 unexcluded) | HOLDS | `results/e1b/read/READ_e1b.md` items 11, 12 |
| I30 | E1: three unexcluded 95th percentiles (0.661, 0.449, 0.472) sit on a floating-point tie; exact 0.575, 0.445, 0.447 | HOLDS (correction) | `results/e1b/read/READ_e1b.md` item 18 |
| I40 | E1b code audit: no bug (six hypotheses); the cross push on gemma is bidirectional (per-context 0.037, signed mean 0.003) and the term is indirect on every model (Olmo 47× its first-order value); Olmo keeps a BOS, so BOS is not the difference | HOLDS | `results/e1b/read/BUGCHECK_e1b.md` |
| I41 | E5 T2: writing only at the marked token leaves a cross fall of 0.005 [0.002, 0.010] (free 0.074): the term travels through the other window positions | HOLDS | `results/e5/verdict_e5.json` |
| I42 | E5 T1: holding the target layer's reader inputs removes 0.031 [0.019, 0.045] of the 0.074 term; cross-specific beyond the own fall's shrinkage 0.014 [0.005, 0.022] (19 percent) | NOT SEPARATED (about a fifth) | same; `results/e5/read/READ_e5.md` items 8, 9 |
| I43 | E5 T4: E1b's construction on Olmo-3 gives N 0.128 [0.086, 0.220]; the construction is not the difference (Olmo-3 is one of three models at a stage-1 checkpoint; T3 untested) | HOLDS | same |
| I44 | E5 T5: per member, Olmo's always-on units write 8 times more of the residual's length than gemma's; with matched counts 1.1 times; per set 2.0 times | REPORTED (not separated) | same; READ_e5.md items 12 to 14 |
| I45 | E6a: every registered write statistic has no power (planted 2.0 reads [0.80, 3.30] or wider); per pair Olmo's largest member writes 2.8 [1.47, 3.5] times gemma's, registered under the same lines and cancelled with them; E6a was not blind to E5's read | CANCELLED | `results/e6/e6a_writes.json`; READ_e6ac.md D20 |
| I46 | E6c: I_marked 0.231 [0.187, 0.286] against the marked null's plain p95 0.098 (its own interval [0.047, 0.206]; tercile-weighted 0.113); like for like 69 percent of cases clear 0.098 against 5 percent of pairs; the reuse result survives the marked-token instrument | HOLDS | `results/e6/verdict_e6c.json`; READ_e6ac.md D9 |
| I47 | E6c: I_marked − I_free median paired −0.026 [−0.043, −0.012]; the median case keeps 0.89 of its free ratio (19 of 149 lose more than 0.15). The routing sentence ("on real pairs the effect is mostly at the marked token, on unrelated pairs elsewhere") is withdrawn: the removed part is the same size on cases and pairs (0.019 against 0.021) | HOLDS (reported); interpretation withdrawn | same; READ_e6ac.md D10, D11 |
| I48 | E6c: the marked-token write is not shown cleaner than zero-at-every-position with the exclusion: null 95th percentiles 0.098 [0.047, 0.206] against 0.205 [0.103, 0.376] (overlapping); both leave the mean's 0.067 / 0.384 behind | NOT SEPARATED | `results/e6/verdict_e6c.json`, `results/e1/verdict_e1.json`; READ_e6ac.md D12 |
