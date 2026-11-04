Cases 64 primary (64 live on 28 neurons) and 43 fresh (43 live on 17 neurons); K reads 135 (1-2 per case); sets scanned 726; forwards 16312, tokens 4855595.

# SPEC10 SHAREDINPUTS verdict: §10 grid and §7 predictions

Every number is in `studies/reuse/results/sharedinputs/verdict_sharedinputs.json` with the artifact it was read from. Intervals: 95% percentile bootstrap over neurons (the pole without its sign), 2,000 draws, seed 0. Falls: the mean over contexts with a_P > 0 of 1 - a_P(ablated)/a_P(clean), mean replacement, free scale, B_t excluded. I_O = Δ(D'(O*); J)/Δ(D'(J); J); I_K = the mean over the case's K of Δ(D'(K); J)/Δ(D'(J); J); d = I_O - I_K; live = SPEC4's live flags at birth and not cancelled at this load.

| rule | reading | outcome |
|---|---|---|
| X1 | median d 0.028 [0.002, 0.104] over 64 live cases on 28 neurons; I_O 0.242 [0.198, 0.333], I_K 0.242 [0.149, 0.353] | not separated: the interval contains 0.05 |
| X1f | median d -0.046 [-0.091, 0.030] over 43 live cases on 17 neurons; I_O 0.255 [0.156, 0.363], I_K 0.332 [0.168, 0.433] | H_old refuted: the interval is below 0.05 (the dependence is shared) |
| gates | 10 of 10 pass | all pass |
| GPU hours | 0.159 h job wall time | landed |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| I_O primary; fresh | 0.29 [0.23, 0.35]; 0.26 [0.18, 0.34] | outside [0.10, 0.50] | 0.242 [0.198, 0.333]; 0.255 [0.156, 0.363] | landed |
| I_K primary; fresh | 0.26 [0.18, 0.33]; 0.24 [0.15, 0.33] | outside [0.05, 0.50] | 0.242 [0.149, 0.353]; 0.332 [0.168, 0.433] | landed |
| d primary (X1); fresh (X1f) | 0.03 [−0.03, 0.09]; 0.02 [−0.05, 0.09] | outside [−0.15, 0.25] | 0.028 [0.002, 0.104]; -0.046 [-0.091, 0.030] | landed |
| share with I_O > I_K, fresh | 0.55 [0.42, 0.68] | outside [0.3, 0.9] | 0.442 [0.326, 0.561] | landed |
| K's own fall (positive control for K) | 0.50 [0.42, 0.58] | under 0.25 | 0.514 [0.478, 0.599]; 0.465 [0.368, 0.582] | landed |
| GPU hours (one load) | 0.4 [0.25, 0.7] | over 1.2 | 0.159 h of job wall time | landed |

Reported, no rule (live cases):

| quantity | primary | fresh |
|---|---|---|
| listed; live; cancelled at this load; neurons | 64; 64; 0; 28 | 43; 43; 0; 17 |
| I_O (SPEC6's re-read 0.302 inside?) | 0.242 [0.198, 0.333] (yes) | 0.255 [0.156, 0.363] (yes) |
| I_K | 0.242 [0.149, 0.353] | 0.332 [0.168, 0.433] |
| share with I_O > I_K | 0.609 [0.500, 0.719] | 0.442 [0.326, 0.561] |
| K's own fall Δ(D'(K); K) | 0.514 [0.478, 0.599] | 0.465 [0.368, 0.582] |
| reverse fall Δ(D'(J); K); its ratio to K's own fall | 0.109 [0.080, 0.161]; 0.234 [0.167, 0.337] | 0.149 [0.079, 0.205]; 0.332 [0.144, 0.475] |
| d over sibling K only (cases; K) | 0.017 [-0.012, 0.107] (41; 44) | -0.063 [-0.173, 0.028] (38; 44) |
| d over non-sibling K only (cases; K) | 0.045 [-0.024, 0.128] (27; 33) | 0.014 [-0.543, 0.030] (11; 14) |
| cross_K, median over every K read | 0.108 (77 reads) | 0.144 (58 reads) |
| d over every listed case (no cancel rule) | 0.028 [0.002, 0.104] | -0.046 [-0.091, 0.030] |
| median falls own', cross_O, random', identity | 0.508, 0.121, 0.009, 0.000 | 0.480, 0.096, 0.016, 0.000 |

Largest |a_before| difference between two arms read on the same contexts: 0.

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t15 | 0.0 | a_wrong at T_15: the previous grid checkpoint's exemplar_a: 0.19487728677276725 | 0.0 | pass |
| completeness t15 | 0.0010306638723704964 | the column from the wrong layer: 0.3238200545310974 | None | pass |
| capture_identity t15 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 0.35796789287815806 | 0.0 | pass |
| known_spec6 | 0.0 | the next case's SPEC6 falls (case-list order, cyclic): 0.08048750311214992 | 0.0 | pass |
| exclusion | {"n_in_B": 0, "n_flagged": 0, "n_not_first_outside": 0, "n_bank_short": 0, "B_count": 3412} | the unexcluded rankings (the first 50 of every top-200 bank) against B_t: members found: 1211 | 3412 | pass |
| positive_control primary | 0.5081641244035354 | random' in place of own': median fall under 0.25: 0.009334546601042725 | 0.0 | pass |
| cancel_share primary | 0.0 | own' and random' swapped: 1.0 | None | pass |
| positive_control fresh | 0.4803936329539666 | random' in place of own': median fall under 0.25: 0.016177930935822954 | 0.0 | pass |
| cancel_share fresh | 0.0 | own' and random' swapped: 1.0 | None | pass |
| k_validity | 0 | every K replaced by J (J's cluster index, J's contexts): 107 | 0 | pass |
