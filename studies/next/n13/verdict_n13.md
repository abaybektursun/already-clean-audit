# N13 verdict: COMPLETE

| gate | pass | failing input failed | value |
|---|---|---|---|
| G-GRADER | True | True | 500 |
| G-TEMPLATE | True | True |  |
| G-INIT | True | True | 0.5571487372697785 |
| G-PARTITION | True | True |  |
| PC-RANK | True | True | 1.0 |
| G-KNOWN | True | True | 39.963934564541645 |
| G-HASH | True | True |  |
| G-MASK | True | True |  |
| G-GATE | True | True | 34.800000000000004 |
| G-ID | True | True | [0.7999999999999972, 1.79999999999999] |
| S-BAND | True | True | 91.2 |
| G-GATE-GSM | True | True | 28.73388931008339 |
| G-COMPLETE | True | True | {"base/math500": true, "base/gsm8k": true, "rl/math500": tru |

| arm | status | acc (%) | Wilson 95% | label | cap-hit share | answer_line share | mean tokens |
|---|---|---|---|---|---|---|---|
| base | done | 57.2 | [52.8, 61.5] | - | 0.14 | 0.44 | 3350 |
| rl | done | 92.0 | [89.3, 94.1] | - | 0.03 | 0.00 | 4499 |
| zero | done | 55.4 | [51.0, 59.7] | - | 0.13 | 0.39 | 3280 |
| id | done | 91.2 | [88.4, 93.4] | - | 0.03 | 0.00 | 4333 |
| top_k1 | done | 61.8 | [57.5, 66.0] | NONE | 0.16 | 0.43 | 3715 |
| mlp | done | 86.4 | [83.1, 89.1] | REC | 0.08 | 0.01 | 4469 |
| top_k0.3 | done | 61.4 | [57.1, 65.6] | NONE | 0.13 | 0.45 | 3347 |
| dare_k3 | done | 86.2 | [82.9, 88.9] | REC | 0.09 | 0.01 | 4456 |
| dare_k0.1 | done | 85.2 | [81.8, 88.0] | REC | 0.10 | 0.02 | 4607 |
| comp_k3 | done | 90.6 | [87.7, 92.9] | REC | 0.05 | 0.00 | 4499 |
| top_k1_B | done | 60.0 | [55.6, 64.2] | NONE | 0.15 | 0.41 | 3653 |
| dare_k1 | done | 85.8 | [82.5, 88.6] | REC | 0.10 | 0.01 | 4667 |
| top_k3 | done | 63.8 | [59.5, 67.9] | NONE | 0.17 | 0.44 | 4004 |
| top_k0.1 | done | 57.6 | [53.2, 61.9] | NONE | 0.15 | 0.43 | 3545 |
| dare_k0.3 | done | 86.8 | [83.6, 89.5] | REC | 0.10 | 0.02 | 4569 |
| comp_k1 | done | 91.6 | [88.8, 93.7] | REC | 0.05 | 0.00 | 4523 |
| rnd0_k1 | done | 59.6 | [55.2, 63.8] | NONE | 0.15 | 0.43 | 3482 |
| dare_k1_d2 | done | 87.8 | [84.6, 90.4] | REC | 0.09 | 0.01 | 4478 |
| rndE_k1 | done | 59.2 | [54.8, 63.4] | NONE | 0.13 | 0.42 | 3324 |
| attn | done | 83.0 | [79.5, 86.0] | REC | 0.12 | 0.04 | 4428 |
| rndT_k3 | done | 62.8 | [58.5, 66.9] | NONE | 0.13 | 0.45 | 3346 |
| rndT_k0.3 | done | 58.0 | [53.6, 62.2] | NONE | 0.14 | 0.43 | 3366 |
| rndT_k0.1 | done | 59.4 | [55.0, 63.6] | NONE | 0.12 | 0.46 | 3133 |
| comp_k0.1 | done | 90.4 | [87.5, 92.7] | REC | 0.06 | 0.00 | 4671 |
| topF_k1 | done | 64.0 | [59.7, 68.1] | NONE | 0.13 | 0.44 | 3435 |
| rndT_k1 | done | 60.0 | [55.6, 64.2] | NONE | 0.14 | 0.45 | 3438 |
| rndE_k3 | done | 63.8 | [59.5, 67.9] | NONE | 0.13 | 0.43 | 3363 |
| rndE_k0.3 | done | 59.8 | [55.4, 64.0] | NONE | 0.13 | 0.46 | 3287 |
| rndE_k0.1 | done | 58.8 | [54.4, 63.0] | NONE | 0.12 | 0.46 | 3148 |
| comp_k0.3 | done | 90.8 | [87.9, 93.0] | REC | 0.05 | 0.00 | 4570 |
| rndT_k1_d2 | done | 59.6 | [55.2, 63.8] | NONE | 0.14 | 0.44 | 3399 |

| hypothesis | verdict |
|---|---|
| k_rec | None |
| H_local | REFUTED (k_rec NONE) |
| H_diffuse | SUPPORTED (smallest k k0.1) |
| H_attention | REFUTED (BOTH) |
| H_artifact | INDETERMINATE |
| H_artifact_values | {'J_floor_k1': 0.6420414821719879, 'J_round_k1': 0.5341453089295043, 'J_round_P_plus_k1': 0.8438896212798138, 'J_sp_k1': 0.007867257903018166, 'J_sp_P_plus_k1': 0.7540453074433657, 'rule_P_plus': 'SUPPORTED', 'rule_P_minus': 'REFUTED'} |
| necessity | {'k0.1': 'NOT NECESSARY', 'k0.3': 'NOT NECESSARY', 'k1': 'NOT NECESSARY', 'k3': 'NOT NECESSARY'} |
