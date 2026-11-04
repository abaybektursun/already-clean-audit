Primary 150: read at birth 150, live 149 (by load {'t06': 2, 't07': 3, 't08': 9, 't09': 18, 't10': 13, 't11': 28, 't12': 41, 't13': 35}), read at T_15 150, live 150; fresh 93: read 93, live 91; ceiling reads at birth 138 primary, 31 fresh; splits at T_15 129; planted pairs 56; loads t15, t05, t06, t07, t08, t09, t10, t11, t12, t13.

# SPEC4 BIRTH verdict: §15 grid and §5 predictions

Every number is in `studies/reuse/results/birth/verdict_birth.json` with the artifact it was read from. Primes: B_t excluded from every set. The 150 primary cases are a re-analysis under the changed instrument; the 93 fresh cases and the pre-birth captures are confirmatory. Intervals: 95% percentile bootstrap over neurons, 2,000 draws, seed 0; the nulls over pairs within tercile, weighted as SPEC2 B1.

| row | predicted (§5) | observed | verdict |
|---|---|---|---|
| gate L | 1.0 [0.8, 1.2] | primary: ρ_a(O*) 1.293 [1.203, 1.458], ρ_ω(O*) 0.942 [0.921, 1.003]; failing input ρ_P 0.954; fresh: ρ_a(O*) 1.232 [1.159, 1.379], ρ_ω(O*) 0.922 [0.867, 0.986]; failing input ρ_P 0.862 | primary FAIL, fresh FAIL |
| L1, primary | H_read | not read | not read (gates failed: gate_L primary) |
| L1, fresh | H_read | not read | not read (gates failed: gate_L fresh); decision: not read |
| room-c, room-c15 | 0.45; ≥ 0.20 | 0.476 (ceiling 0.649 − null p95 0.172); 0.296 (ceiling 0.672 − null p95 0.376); shuffled -0.330, -0.305 | pass, pass |
| R1c, R1c15 | not refuted, not refuted | I′ 0.207 [0.156, 0.291] against null′ 0.012 [0.002, 0.024]; I′ 0.294 [0.229, 0.345] against null′ 0.065 [0.034, 0.088] | not refuted: no overlap and real I' is higher; not refuted: no overlap and real I' is higher |
| M1 | 0.02, inside the band | 0.011 [0.004, 0.033] | neither: the interval crosses an edge of the band |
| M2 | one null | T_15 0.065 [0.034, 0.088] against birth 0.012 [0.002, 0.024] | the T_15 null has a term the high-background neurons do not explain |
| R2c | specific | cross′ − foreign′ 0.088 [0.066, 0.123]; cross′ − R_J′ 0.102 [0.080, 0.139] | specific: both intervals above 0 |
| R3c | generic | -0.004 [-0.039, 0.030] | generic: the interval contains 0 |
| S1 | not refuted, 0.11 against 0.05 | s′ 0.120 [0.080, 0.160] against null′ p95 0.060; position 0.316 | H_share not refuted: the interval is above the null's 95th percentile |
| positive control, cancel share, identity, known number, completeness, exclusion, memory | as §7 | own′ at birth 0.544 / 0.536; cancel share 0.007 / 0.022; identity max 0; exclusion members in B_t 0 of 293350 (failing input 2665); memory {"t15": 20.97023488, "t05": 20.52742656, "t06": 20.52742656, "t07": 20.531293184, "t08": 20.646338048, "t09": 20.97023488, "t10": 20.97023488, "t11": 20.97023488, "t12": 20.97023488, "t13": 20.97023488} | FAIL: gate_L primary, gate_L fresh |
| billed hours, dollars, offer rented | 0.75 h, $0.6 | not in the bank | the rental record |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| null′ at T_15 (M2 input) | 0.05 [0.02, 0.10] | outside [0.00, 0.12] | 0.065 [0.034, 0.088] | landed |
| null′ at birth | 0.01 [0.00, 0.03] | outside [−0.02, 0.05] | 0.012 [0.002, 0.024] | landed |
| room-c at birth | 0.45 | under 0.20 | 0.476 | landed |
| I′ at birth, primary re-read | 0.17 [0.12, 0.23] | outside [0.08, 0.30] | 0.207 [0.156, 0.291] | landed |
| I′ at birth, fresh set | 0.17 [0.10, 0.25] | outside [0.05, 0.35] | 0.257 [0.199, 0.348] | landed |
| I′ at T_15, primary | 0.24 [0.18, 0.31] | outside [0.12, 0.40] | 0.294 [0.229, 0.345] | landed |
| M1, foreign′ − random′ at T_15 | 0.02 [−0.01, 0.04] | above 0.05 | 0.011 [0.004, 0.033] | landed |
| R2c | specific against both | either interval contains 0 | 0.088 [0.066, 0.123]; 0.102 [0.080, 0.139] | landed |
| R3c | generic, difference 0.00 [−0.05, 0.05] | interval above 0.05 or below −0.05 | -0.004 [-0.039, 0.030] | landed |
| S1, fresh s′ | 0.11 [0.07, 0.15] against a null′ p95 of 0.05 and a ceiling′ of 0.38 | interval touches the p95 | 0.120 [0.080, 0.160] against p95 0.060; ceiling′ 0.380 | landed |
| L1 | H_read: r_a 0.75 [0.55, 0.95], r_ω 0.35 [0.15, 0.55], ρ_P 0.15 [0.05, 0.30] | r_a under 0.5 or r_ω over 0.5 | primary: not read; fresh: not read | primary not read; fresh not read |
| gate L baseline, ρ_a(O*), ρ_ω(O*) | 1.0 [0.8, 1.2] each | outside [0.5, 2.0] | primary: ρ_a(O*) 1.293 [1.203, 1.458], ρ_ω(O*) 0.942 [0.921, 1.003]; fresh: ρ_a(O*) 1.232 [1.159, 1.379], ρ_ω(O*) 0.922 [0.867, 0.986] | primary landed; fresh landed |
| peak GPU memory, deepest poles | 23 GB [19, 27] | outside | max 20.970 GB over 10 loads | landed |
| billed time on the rental | 0.75 h [0.5, 1.3]; cost $0.6 [0.4, 1.2] of credit | over 2 h or $2 | not in the bank (the rental record) | not read here |

Per tercile (reported, no rule):

| tercile | M1 | M2 | R1c: I′ against null′ |
|---|---|---|---|
| 0 | 0.048 [0.001, 0.104]: neither: the interval crosses an edge of the band | 0.038 vs 0.002: one null at both checkpoints | 0.285 [0.156, 0.398] against 0.002 [-0.010, 0.034]: real T higher |
| 1 | 0.011 [0.001, 0.035]: neither: the interval crosses an edge of the band | 0.089 vs 0.026: neither: its own outcome | 0.201 [0.151, 0.426] against 0.026 [-0.001, 0.039]: real T higher |
| 2 | 0.004 [-0.009, 0.019]: inside the band: the high-background neurons were the generic term | 0.067 vs 0.007: neither: its own outcome | 0.147 [0.099, 0.255] against 0.007 [0.003, 0.023]: real T higher |

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t15 | 0.0031067298450119176 | a_wrong at T_15: the previous grid checkpoint's exemplar_a: 0.1959097573097563 | 0.0 | pass |
| completeness t15 | 0.0008556043612770736 | the column from the wrong layer: 0.32718442380428314 | None | pass |
| identity t15 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 3696 | None | pass |
| known_number t05 | 0.0006347999899440761 | a_wrong at T_5: the previous grid checkpoint's exemplar_a: 0.049414852234093115 | 0.0 | pass |
| completeness t05 | 0.001098610955523327 | the column from the wrong layer: 0.23023514449596405 | None | pass |
| identity t05 | not read: no arm is read at a pre-birth load; the identity gate has nothing to read | | | reported |
| known_number t06 | 0.0010430887959292081 | a_wrong at T_6: the previous grid checkpoint's exemplar_a: 0.06570649178445889 | 0.0 | pass |
| completeness t06 | 0.0012146575609222054 | the column from the wrong layer: 0.22653408348560333 | None | pass |
| identity t06 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 1397 | None | pass |
| known_number t07 | 0.0 | a_wrong at T_7: the previous grid checkpoint's exemplar_a: 0.07684167890406367 | 0.0 | pass |
| completeness t07 | 0.0011196023551747203 | the column from the wrong layer: 0.32495424151420593 | None | pass |
| identity t07 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 1424 | None | pass |
| known_number t08 | 0.0 | a_wrong at T_8: the previous grid checkpoint's exemplar_a: 0.05266428403910494 | 0.0 | pass |
| completeness t08 | 0.0010166307329200208 | the column from the wrong layer: 0.24135342240333557 | None | pass |
| identity t08 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 1520 | None | pass |
| known_number t09 | 0.0 | a_wrong at T_9: the previous grid checkpoint's exemplar_a: 0.02960960103208915 | 0.0 | pass |
| completeness t09 | 0.0013297644327394664 | the column from the wrong layer: 0.23069265484809875 | None | pass |
| identity t09 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 1696 | None | pass |
| known_number t10 | 0.0 | a_wrong at T_10: the previous grid checkpoint's exemplar_a: 0.14684500886684276 | 0.0 | pass |
| completeness t10 | 0.0009752684272825718 | the column from the wrong layer: 0.24425402283668518 | None | pass |
| identity t10 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 1757 | None | pass |
| known_number t11 | 0.0 | a_wrong at T_11: the previous grid checkpoint's exemplar_a: 0.12237208964167065 | 0.0 | pass |
| completeness t11 | 0.0011171804799232632 | the column from the wrong layer: 0.2551591396331787 | None | pass |
| identity t11 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 1910 | None | pass |
| known_number t12 | 0.0 | a_wrong at T_12: the previous grid checkpoint's exemplar_a: 0.12647526588569538 | 0.0 | pass |
| completeness t12 | 0.0009450121433474123 | the column from the wrong layer: 0.27013564109802246 | None | pass |
| identity t12 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 2331 | None | pass |
| known_number t13 | 0.0 | a_wrong at T_13: the previous grid checkpoint's exemplar_a: 0.1288775479989188 | 0.0 | pass |
| completeness t13 | 0.0012340612011030316 | the column from the wrong layer: 0.2729680836200714 | None | pass |
| identity t13 | 0.0 | own' drivers, m 50 (contexts over 1e-3): 2290 | None | pass |
| exclusion | 0 | the primary bank's own and cross sets (no exclusion) at 9 loads against this run's B_t: members found: 2665 | 0 | pass |
| positive_control primary | 0.5436362449371597 | random' in place of own': median fall at birth under 0.25: 0.0015876556583371464 | 0.0 | pass |
| cancel_share primary | 0.006666666666666667 | own' and random' swapped in the bank: 1.0 | None | pass |
| positive_control fresh | 0.5358899337826667 | random' in place of own': median fall at birth under 0.25: 0.0004488987654441164 | 0.0 | pass |
| cancel_share fresh | 0.021505376344086023 | own' and random' swapped in the bank: 1.0 | None | pass |
| room_c | 0.4764578737346178 | planted side labels shuffled within pair (own' and cross'): -0.33044302821299354 | None | pass |
| room_c15 | 0.29588702213844076 | planted side labels shuffled within pair (own' and cross'): -0.30533656647964635 | None | pass |
| gate_L primary | {"rho_a_O": 1.2932355639800348, "rho_w_O": 0.9423826540881721} | rho_P (P's own mean activation on J at t_b - 1 over t_b) as the baseline: must read under 0.5: 0.953584382632158 | None | FAIL |
| gate_L fresh | {"rho_a_O": 1.232346303849211, "rho_w_O": 0.9219228716381175} | rho_P (P's own mean activation on J at t_b - 1 over t_b) as the baseline: must read under 0.5: 0.862248409956167 | None | FAIL |
| memory | {"t15": 20.97023488, "t05": 20.52742656, "t06": 20.52742656, "t07": 20.531293184, "t08": 20.646338048, "t09": 20.97023488, "t10": 20.97023488, "t11": 20.97023488, "t12": 20.97023488, "t13": 20.97023488} | | | reported |
