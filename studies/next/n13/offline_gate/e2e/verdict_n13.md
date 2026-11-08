# N13 verdict: COMPLETE

| gate | pass | failing input failed | value |
|---|---|---|---|
| G-GRADER | True | True | 500 |
| G-TEMPLATE | True | True |  |
| G-INIT | True | True | 0.6824408637152778 |
| G-PARTITION | True | True |  |
| PC-RANK | True | True | 1.0 |
| G-KNOWN | True | True | 40.0 |
| G-HASH | True | True |  |
| G-MASK | True | True |  |
| G-GATE | True | True | 58.60000000000001 |
| G-ID | True | True | [-1.4000000000000057, -1.0] |
| S-BAND | True | True | 89.60000000000001 |
| G-GATE-GSM | True | True | 53.33333333333333 |
| G-COMPLETE | True | True | {"base/math500": true, "base/gsm8k": true, "rl/math500": tru |

| arm | status | acc (%) | Wilson 95% | label | cap-hit share | answer_line share | mean tokens |
|---|---|---|---|---|---|---|---|
| base | done | 29.6 | [25.8, 33.7] | - | 0.00 | 0.90 | 7 |
| rl | done | 88.2 | [85.1, 90.7] | - | 0.00 | 0.91 | 9 |
| zero | done | 30.6 | [26.7, 34.8] | - | 0.00 | 0.89 | 7 |
| id | done | 89.6 | [86.6, 92.0] | - | 0.00 | 0.87 | 9 |
| top_k1 | done | 74.8 | [70.8, 78.4] | REC | 0.00 | 0.91 | 8 |
| dare_k1 | done | 34.2 | [30.2, 38.5] | NONE | 0.00 | 0.90 | 7 |
| mlp | done | 80.4 | [76.7, 83.6] | REC | 0.00 | 0.91 | 9 |
| top_k3 | done | 83.4 | [79.9, 86.4] | REC | 0.00 | 0.91 | 9 |
| top_k0.3 | done | 43.4 | [39.1, 47.8] | NONE | 0.00 | 0.91 | 7 |
| top_k0.1 | done | 33.2 | [29.2, 37.4] | NONE | 0.00 | 0.90 | 7 |
| dare_k3 | done | 47.0 | [42.7, 51.4] | PART | 0.00 | 0.91 | 7 |
| dare_k0.3 | done | 29.6 | [25.8, 33.7] | NONE | 0.00 | 0.90 | 7 |
| dare_k0.1 | screened | 0.0 | [0.0, 16.1] | DAMAGED | 0.80 | 0.20 | 205 |
| comp_k1 | done | 39.4 | [35.2, 43.7] | NONE | 0.00 | 0.90 | 7 |
| comp_k3 | done | 36.8 | [32.7, 41.1] | NONE | 0.00 | 0.90 | 7 |
| rnd0_k1 | done | 30.8 | [26.9, 35.0] | NONE | 0.00 | 0.90 | 7 |
| top_k1_B | done | 77.2 | [73.3, 80.7] | REC | 0.00 | 0.88 | 9 |
| dare_k1_d2 | done | 34.2 | [30.2, 38.5] | NONE | 0.00 | 0.90 | 7 |
| rndE_k1 | done | 32.2 | [28.3, 36.4] | NONE | 0.00 | 0.90 | 7 |
| rndT_k1 | done | 33.2 | [29.2, 37.4] | NONE | 0.00 | 0.90 | 7 |
| attn | done | 30.8 | [26.9, 35.0] | NONE | 0.00 | 0.90 | 7 |
| rndE_k3 | done | 33.2 | [29.2, 37.4] | NONE | 0.00 | 0.90 | 7 |
| rndT_k3 | done | 33.4 | [29.4, 37.6] | NONE | 0.00 | 0.90 | 7 |
| rndE_k0.3 | done | 32.2 | [28.3, 36.4] | NONE | 0.00 | 0.90 | 7 |
| rndT_k0.3 | done | 33.2 | [29.2, 37.4] | NONE | 0.00 | 0.90 | 7 |
| rndE_k0.1 | done | 32.2 | [28.3, 36.4] | NONE | 0.00 | 0.90 | 7 |
| rndT_k0.1 | done | 33.2 | [29.2, 37.4] | NONE | 0.00 | 0.90 | 7 |
| comp_k0.3 | done | 39.4 | [35.2, 43.7] | NONE | 0.00 | 0.90 | 7 |
| comp_k0.1 | done | 39.4 | [35.2, 43.7] | NONE | 0.00 | 0.90 | 7 |
| rndT_k1_d2 | done | 33.2 | [29.2, 37.4] | NONE | 0.00 | 0.90 | 7 |
| topF_k1 | done | 73.4 | [69.4, 77.1] | REC | 0.00 | 0.91 | 8 |

| hypothesis | verdict |
|---|---|
| k_rec | k1 |
| H_local | SUPPORTED |
| H_diffuse | REFUTED |
| H_attention | REFUTED |
| H_artifact | REFUTED |
| H_artifact_values | {'J_floor_k1': 1.0, 'J_round_k1': 1.0, 'J_round_synthetic_k1': 0.23446886446886445} |
| necessity | {'k0.1': 'NECESSARY', 'k0.3': 'NECESSARY', 'k1': 'NECESSARY', 'k3': 'NECESSARY'} |
