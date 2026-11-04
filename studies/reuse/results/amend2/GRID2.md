Jobs over 150: live at birth 148 (by load {'t06': 2, 't07': 3, 't08': 9, 't09': 17, 't10': 13, 't11': 28, 't12': 41, 't13': 35}), live at T_15 149, live at both 147; ceiling reads at birth 138, splits at T_15 129; planted pairs 56.

# REUSE amendment 2 verdict: SPEC2 §5 rows the GPU job answers

Every number is in `studies/reuse/results/amend2/verdict2.json` with the artifact it was read from. Intervals: 95% percentile bootstrap over neurons, 2,000 draws, seed 0; the planted median's over pairs within tercile. m 50, direct ranking, mean replacement.

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| planted median at birth, pooled | 0.14 [0.09, 0.19] | outside [0.05, 0.25] | 0.031 [0.021, 0.051] | refuted |
| room-b | 0.25 | under 0.20 | 0.366 | landed |
| R1b | refuted (overlap) | not refuted | not refuted: no overlap and real I is higher | refuted |
| R_J fall median | 0.02 | over 0.06 | -0.005 at birth over 148 live jobs | landed |
| R2b | specific beyond activity | interval contains 0 | 0.126 [0.100, 0.141] | landed |
| ceiling rise over 138, paired median | 0.10 [0.05, 0.16] | under 0.03 or over 0.20 | 0.081 [0.052, 0.119] over 138 reads | landed |
| R3b | generic | reuse grows beyond | 0.017 [-0.031, 0.053]: generic | landed |

| rule | observed | verdict |
|---|---|---|
| R1b | median I at birth 0.226 [0.170, 0.308] against planted at birth 0.031 [0.021, 0.051]; T_15: I 0.375 [0.303, 0.446] against 0.174 [0.117, 0.221] | not refuted: no overlap and real I is higher |
| R2b | median cross − R_J at birth 0.126 [0.100, 0.141] on 148 live jobs | specific beyond activity: the interval is above 0 |
| R3b | R3 rise 0.098 − ceiling rise 0.081 = 0.017 [-0.031, 0.053] | generic: the interval contains 0 |

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t15 | 0.0 | a_wrong at T_15: the previous grid checkpoint's exemplar_a: 0.19487728677276725 | 0.0 | pass |
| completeness t15 | 0.0010306638723704964 | the column from the wrong layer: 0.3238200545310974 | None | pass |
| identity t15 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 889 | None | pass |
| known_number t06 | 0.0 | a_wrong at T_6: the previous grid checkpoint's exemplar_a: 0.06333445371846437 | 0.0 | pass |
| completeness t06 | 0.001118676271289587 | the column from the wrong layer: 0.22408752143383026 | None | pass |
| identity t06 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1343 | None | pass |
| known_number t07 | 0.000819961463809643 | a_wrong at T_7: the previous grid checkpoint's exemplar_a: 0.07868939426605578 | 0.0 | pass |
| completeness t07 | 0.0012617202010005713 | the column from the wrong layer: 0.32522740960121155 | None | pass |
| identity t07 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1343 | None | pass |
| known_number t08 | 0.0008851542610905565 | a_wrong at T_8: the previous grid checkpoint's exemplar_a: 0.05364612084112383 | 0.0 | pass |
| completeness t08 | 0.0011829049326479435 | the column from the wrong layer: 0.24265962094068527 | None | pass |
| identity t08 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1342 | None | pass |
| known_number t09 | 0.00024701980075715423 | a_wrong at T_9: the previous grid checkpoint's exemplar_a: 0.029578937538463485 | 0.0 | pass |
| completeness t09 | 0.0008651390089653432 | the column from the wrong layer: 0.23180870711803436 | None | pass |
| identity t09 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1338 | None | pass |
| known_number t10 | 0.0007734560594757688 | a_wrong at T_10: the previous grid checkpoint's exemplar_a: 0.1468453072399958 | 0.0 | pass |
| completeness t10 | 0.0012794611975550652 | the column from the wrong layer: 0.24074269086122513 | None | pass |
| identity t10 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1343 | None | pass |
| known_number t11 | 0.0003021315977600342 | a_wrong at T_11: the previous grid checkpoint's exemplar_a: 0.12102014709698802 | 0.0 | pass |
| completeness t11 | 0.0009985077776946127 | the column from the wrong layer: 0.2558227553963661 | None | pass |
| identity t11 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1344 | None | pass |
| known_number t12 | 0.0008169767474135661 | a_wrong at T_12: the previous grid checkpoint's exemplar_a: 0.12603903111757375 | 0.0 | pass |
| completeness t12 | 0.0009582680650055408 | the column from the wrong layer: 0.2692252993583679 | None | pass |
| identity t12 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1342 | None | pass |
| known_number t13 | 0.0013254752973527384 | a_wrong at T_13: the previous grid checkpoint's exemplar_a: 0.1288775479989188 | 0.0 | pass |
| completeness t13 | 0.001132183417212218 | the column from the wrong layer: 0.2754121795296669 | None | pass |
| identity t13 | 0.0 | own drivers, m 50 direct (contexts over 1e-3): 1343 | None | pass |
| b2_a_before_reproduces | 0.0 | the same job's a_before at the other checkpoint (birth <-> T_15) in the primary bank: 0.34690967071926043 | 0.0 | pass |
| b2_drivers_reproduce | 0 | the primary bank's own and cross sets at the other checkpoint: 300 | 0 | pass |
| b3_s3_splits_agree | 0.0 | the same split's side-2 a_before at its birth load (primary bank): 0.2765525626668989 | 0.0 | pass |
| room_b | 0.3655742848813728 | planted side labels shuffled within pair at the same loads (own and cross): -0.3755484159211073 | None | pass |
| positive_control_rj | 0.5275728539285014 | R_J in place of own: median fall at birth must be under 0.25: -0.004605536816870823 | 0.0 | pass |
