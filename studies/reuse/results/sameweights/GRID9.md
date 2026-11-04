Cases 150 primary (149 live on 103 neurons) and 93 fresh (91 live on 58 neurons); ceiling splits 129; sets scanned 1473; captures 14458, tokens 3484567.

# SPEC9 SAMEWEIGHTS verdict: §10 grid and §7 predictions

Every number is in `studies/reuse/results/sameweights/verdict_sameweights.json` with the artifact it was read from. Intervals: 95% percentile bootstrap over neurons (the pole without its sign), 2,000 draws, seed 0. s_O = |D'(J) ∩ D'(O*)| / 50; s_swap = the mean over three swapped sets of |D'(J) ∩ D'(swap)| / 50; d = s_O - s_swap; live = SPEC4's live flags at birth.

| rule | reading | outcome |
|---|---|---|
| W2 | median d 0.127 [0.093, 0.153] over 149 live cases on 103 neurons; s_O 0.140 [0.110, 0.180], s_swap 0.013 [0.013, 0.020] | H_weights refuted: the interval is above 0.03 (the overlap is specific to the two clusters) |
| W2f | median d 0.133 [0.093, 0.167] over 91 live cases on 58 neurons; s_O 0.160 [0.120, 0.180], s_swap 0.013 [0.007, 0.013] | H_weights refuted: the interval is above 0.03 (the overlap is specific to the two clusters) |
| gates | 7 of 7 pass | all pass |
| GPU hours | 0.140 h job wall time | landed |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| s_O primary at T_15; fresh | 0.12 [0.10, 0.14]; 0.12 [0.08, 0.16] | outside [0.04, 0.25] | 0.140 [0.110, 0.180]; 0.160 [0.120, 0.180] | landed |
| s_swap primary; fresh | 0.02 [0.01, 0.04]; 0.02 [0.01, 0.04] | outside [0.00, 0.10] | 0.013 [0.013, 0.020]; 0.013 [0.007, 0.013] | landed |
| d primary (W2); fresh (W2f) | 0.09 [0.06, 0.12]; 0.09 [0.05, 0.13] | outside [0.00, 0.20] | 0.127 [0.093, 0.153]; 0.133 [0.093, 0.167] | landed |
| share of cases with s_O > s_swap, fresh | 0.80 [0.70, 0.88] | outside [0.5, 0.95] | 0.934 [0.875, 0.979] | landed |
| known number (median overlap with SPEC6's sets) | 0.96 [0.92, 1.00] | under 0.90 | J 1.000, O* 1.000 | landed |
| GPU hours (one load, captures only) | 0.25 [0.15, 0.45] | over 0.8 | 0.140 h of job wall time | landed |

Reported, no rule (live cases):

| quantity | primary | fresh |
|---|---|---|
| s_O at T_15 (SPEC4's s' at birth) | 0.140 [0.110, 0.180] (0.120) | 0.160 [0.120, 0.180] (0.120) |
| s_swap against 0.060 | 0.013 [0.013, 0.020]: not above | 0.013 [0.007, 0.013]: not above |
| share with s_O > s_swap | 0.866 [0.806, 0.921] | 0.934 [0.875, 0.979] |
| per-swap medians swap_1, swap_2, swap_3 | 0.020, 0.000, 0.000 | 0.000, 0.000, 0.000 |
| per-case spread of the three (median; max) | 0.020; 0.280 | 0.020; 0.220 |
| D'(O*) against the swaps | 0.013 [0.007, 0.013] | 0.007 [0.007, 0.013] |
| d on swaps with P's mean a_P >= 0.1 of J's (§8) | 0.310 [0.160, 0.460] (2 cases, 147 dropped; 2 of 447 swaps): above 0.03 | -0.060 [-0.060, -0.060] (1 cases, 90 dropped; 1 of 273 swaps): below 0.03 |
| J and O* share a document; d without | 0 of 149; 0.127 [0.093, 0.153] (149 cases): above 0.03 | 0 of 91; 0.133 [0.093, 0.167] (91 cases): above 0.03 |

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t15 | 0.0 | a_wrong at T_15: the previous grid checkpoint's exemplar_a: 0.19487728677276725 | 0.0 | pass |
| completeness t15 | 0.0010306638723704964 | the column from the wrong layer: 0.3238200545310974 | None | pass |
| capture_identity t15 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 0.35796789287815806 | 0.0 | pass |
| known_spec6 | {"J": 1.0, "O": 1.0} | the next case's SPEC6 set (item order, cyclic): {"J": 0.0, "O": 0.0} | {"J": 1.0, "O": 1.0} | pass |
| exclusion | {"n_in_B": 0, "n_flagged": 0, "n_not_first_outside": 0, "B_count": 3412} | the unexcluded rankings (the first 50 of every top-200 bank) against B_t: members found: 3786 | 3412 | pass |
| swap_validity | 0 | per case swap_1 replaced by a case on P's own neuron (the first other case on it in item order, else the case): 243 | 0 | pass |
| room | 0.32 | the side-2 sets shuffled across splits (default_rng(0).permutation, splits in sorted order): -0.06 | 0.94 | pass |
