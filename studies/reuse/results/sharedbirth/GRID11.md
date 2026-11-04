Cases 47 primary (47 live on 28 neurons) and 36 fresh (36 live on 17 neurons) at their birth; K reads 101 (1-2 per case); K dropped (born later) 34, SPEC10 cases dropped 24; births 3, 4, 4, 10, 8, 24, 30 over t07, t08, t09, t10, t11, t12, t13; sets scanned 552; forwards 12538, tokens 4380768.

# SPEC11 SHAREDBIRTH verdict: §10 grid and §7 predictions

Every number is in `studies/reuse/results/sharedbirth/verdict_sharedbirth.json` with the artifact it was read from. Intervals: 95% percentile bootstrap over neurons (the pole without its sign), 2,000 draws, seed 0, pooled over the loads. Falls at the case's birth checkpoint t_b(J): the mean over contexts with a_P > 0 of 1 - a_P(ablated)/a_P(clean), mean replacement, free scale, that checkpoint's B_t excluded. I_O = Δ(D'(O*); J)/Δ(D'(J); J); I_K = the mean over the case's K that exist at t_b(J) of Δ(D'(K); J)/Δ(D'(J); J); d = I_O - I_K; live = SPEC4's live flags at birth and not cancelled at t_b(J).

| rule | reading | outcome |
|---|---|---|
| Y1 | median d -0.005 [-0.051, 0.028] over 47 live cases on 28 neurons; I_O 0.171 [0.084, 0.284], I_K 0.215 [0.089, 0.276]; read on K with own fall >= 0.25 only: -0.005 [-0.051, 0.028] | H_old-at-birth refuted: the interval is below 0.05 (at the birth the dependence is shared) |
| Y1f | median d -0.014 [-0.076, 0.025] over 36 live cases on 17 neurons; I_O 0.218 [0.146, 0.376], I_K 0.247 [0.081, 0.398]; read on K with own fall >= 0.25 only: -0.014 [-0.078, 0.042] | H_old-at-birth refuted: the interval is below 0.05 (at the birth the dependence is shared) |
| gates | 28 of 28 pass | all pass |
| GPU hours | 0.371 h job wall time over the loads | landed |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| I_O primary; fresh (SPEC4 read 0.207 and 0.257 on all cases) | 0.21 [0.15, 0.28]; 0.25 [0.17, 0.34] | outside [0.05, 0.50] | 0.171 [0.084, 0.284]; 0.218 [0.146, 0.376] | landed |
| I_K primary; fresh | 0.19 [0.12, 0.28]; 0.22 [0.13, 0.33] | outside [0.03, 0.50] | 0.215 [0.089, 0.276]; 0.247 [0.081, 0.398] | landed |
| d primary (Y1); fresh (Y1f) | 0.02 [−0.05, 0.09]; 0.02 [−0.06, 0.10] | outside [−0.15, 0.25] | -0.005 [-0.051, 0.028]; -0.014 [-0.076, 0.025] | landed |
| K's own fall at t_b(J) | 0.45 [0.35, 0.55] | under 0.25 | 0.544 [0.482, 0.595] (both sets pooled) | landed |
| GPU hours (seven loads) | 0.7 [0.4, 1.2] | over 2.0 | 0.371 h of job wall time over the loads | landed |

Reported, no rule (live cases):

| quantity | primary | fresh |
|---|---|---|
| listed; live; cancelled at the birth load; neurons | 47; 47; 0; 28 | 36; 36; 0; 17 |
| I_O | 0.171 [0.084, 0.284] | 0.218 [0.146, 0.376] |
| SPEC4's per-case I at birth, same cases (SPEC4's R1c: 0.207); median per-case abs difference to I_O | 0.163 [0.105, 0.284]; 0.004 | 0.219 [0.137, 0.380]; 0.004 |
| I_K | 0.215 [0.089, 0.276] | 0.247 [0.081, 0.398] |
| share with I_O > I_K | 0.468 [0.318, 0.625] | 0.417 [0.303, 0.529] |
| K's own fall Δ(D'(K); K), per case; per K; K under 0.25 | 0.534 [0.465, 0.607]; 0.531; 0 | 0.555 [0.414, 0.643]; 0.554; 1 |
| d read on K with own fall >= 0.25 only (cases; K) | -0.005 [-0.051, 0.028] (47; 56) | -0.014 [-0.078, 0.042] (35; 44) |
| reverse fall Δ(D'(J); K); its ratio to K's own fall | 0.090 [0.038, 0.142]; 0.206 [0.064, 0.265] | 0.104 [0.055, 0.176]; 0.242 [0.101, 0.489] |
| d over sibling K only (cases; K) | -0.011 [-0.072, 0.088] (32; 35) | -0.014 [-0.078, 0.007] (33; 38) |
| d over non-sibling K only (cases; K) | -0.005 [-0.030, 0.026] (15; 21) | 0.085 [-0.118, 0.456] (5; 7) |
| overlap D'(K) with D'(O*): sibling K; non-sibling K | 0.080; 0.080 | 0.090; 0.100 |
| overlap D'(K) with D'(J): sibling K; non-sibling K | 0.140; 0.060 | 0.120; 0.180 |
| overlap D'(J) with D'(O*) | 0.080 | 0.090 |
| pole's mean clean a_P on K's contexts; on J's | 5.998; 3.715 | 4.632; 4.195 |
| cross_K, median over every K read | 0.095 (56 reads) | 0.085 (45 reads) |
| d over every listed case (no cancel rule) | -0.005 [-0.051, 0.028] | -0.014 [-0.076, 0.025] |
| median falls own', cross_O, random', identity | 0.545, 0.089, 0.003, 0.000 | 0.541, 0.106, -0.000, 0.000 |

Pooled over both sets: d -0.013 [-0.048, 0.023], K's own fall 0.544 [0.482, 0.595] (83 live cases).

| load | primary listed (live) | fresh listed (live) | K reads |
|---|---|---|---|
| T_7 | 0 (0) | 3 (3) | 3 |
| T_8 | 4 (4) | 0 (0) | 7 |
| T_9 | 4 (4) | 0 (0) | 4 |
| T_10 | 4 (4) | 6 (6) | 10 |
| T_11 | 5 (5) | 3 (3) | 11 |
| T_12 | 18 (18) | 6 (6) | 26 |
| T_13 | 12 (12) | 18 (18) | 40 |

K dropped as born after t_b(J): 34; SPEC10 cases left with no K: 24.
Largest |a_before| difference between two arms read on the same contexts: 0.

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t07 | 0.000819961463809643 | a_wrong at T_7: the previous grid checkpoint's exemplar_a: 0.07868939426605578 | 0.0 | pass |
| completeness t07 | 0.0012617202010005713 | the column from the wrong layer: 0.32522740960121155 | None | pass |
| capture_identity t07 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 49.25235150080782 | 0.0 | pass |
| known_number t08 | 0.0008851542610905565 | a_wrong at T_8: the previous grid checkpoint's exemplar_a: 0.05364612084112383 | 0.0 | pass |
| completeness t08 | 0.0011829049326479435 | the column from the wrong layer: 0.24265962094068527 | None | pass |
| capture_identity t08 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 20.36868023221645 | 0.0 | pass |
| known_number t09 | 0.00024701980075715423 | a_wrong at T_9: the previous grid checkpoint's exemplar_a: 0.029578937538463485 | 0.0 | pass |
| completeness t09 | 0.0008651390089653432 | the column from the wrong layer: 0.23180870711803436 | None | pass |
| capture_identity t09 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 6.469266346242089 | 0.0 | pass |
| known_number t10 | 0.0007734560594757688 | a_wrong at T_10: the previous grid checkpoint's exemplar_a: 0.1468453072399958 | 0.0 | pass |
| completeness t10 | 0.0012794611975550652 | the column from the wrong layer: 0.24074269086122513 | None | pass |
| capture_identity t10 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 3.02584609097348 | 0.0 | pass |
| known_number t11 | 0.0003021315977600342 | a_wrong at T_11: the previous grid checkpoint's exemplar_a: 0.12102014709698802 | 0.0 | pass |
| completeness t11 | 0.0009985077776946127 | the column from the wrong layer: 0.2558227553963661 | None | pass |
| capture_identity t11 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 1.8653475996204523 | 0.0 | pass |
| known_number t12 | 0.0008169767474135661 | a_wrong at T_12: the previous grid checkpoint's exemplar_a: 0.12603903111757375 | 0.0 | pass |
| completeness t12 | 0.0009582680650055408 | the column from the wrong layer: 0.2692252993583679 | None | pass |
| capture_identity t12 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 1.68084524513291 | 0.0 | pass |
| known_number t13 | 0.0013254752973527384 | a_wrong at T_13: the previous grid checkpoint's exemplar_a: 0.1288775479989188 | 0.0 | pass |
| completeness t13 | 0.001132183417212218 | the column from the wrong layer: 0.2754121795296669 | None | pass |
| capture_identity t13 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 1.5802226088846887 | 0.0 | pass |
| known_spec4 | 0.0014183072852218537 | the next case's SPEC4 falls (list order over the loads, cyclic): 0.08053990812040672 | 0.0 | pass |
| exclusion | {"n_in_B": 0, "n_flagged": 0, "n_not_first_outside": 0, "n_bank_short": 0, "B_count": {"t07": 3412, "t08": 3412, "t09": 3412, "t10": 3412, "t11": 3412, "t12": 3412, "t13": 3412}} | the unexcluded rankings (the first 50 of every top-200 bank) against B_t: members found: 1493 | 23884 | pass |
| positive_control primary | 0.5450477919247341 | random' in place of own': median fall under 0.25: 0.003357886422931901 | 0.0 | pass |
| cancel_share primary | 0.0 | own' and random' swapped: 1.0 | None | pass |
| positive_control fresh | 0.5408390247353139 | random' in place of own': median fall under 0.25: -0.0002004793753130204 | 0.0 | pass |
| cancel_share fresh | 0.0 | own' and random' swapped: 1.0 | None | pass |
| k_validity | 0 | every K of SPEC10's list put back (the K born after t_b included): 30 | 0 | pass |
