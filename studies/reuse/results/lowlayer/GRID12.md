# SPEC12 LOWLAYER verdict: §10 grid and §7 predictions

Every number is in `studies/reuse/results/lowlayer/verdict_lowlayer.json` with the artifact it was read from. Intervals: 95% percentile bootstrap over neurons, 2,000 draws, seed 0; the E1 null 2,000 permutations of t_b among this set's live cases.

Rules. E1 reads three ways, as SPEC8's verdict: the interval wholly above the within-set null's p95 (lo > p95) -> holds; wholly below (hi <= p95) -> refuted; containing p95 -> not separated. The gain: wholly above 1 (lo > 1) -> holds; wholly below 1 (hi < 1) -> refuted; containing 1 -> not separated. H_token-event: not refuted only when E1 holds and the gain's interval lies wholly above 1; refuted when E1 is refuted or the gain's interval lies wholly below 1; cannot be read when E1 or the gain cannot be read and neither refutes; otherwise not separated.

Readings (§1): {"H_token-event": "not separated", "H_context-event": "not refuted", "H_same": "refuted", "H_artifact": "not refuted"}

| rule | reading | outcome |
|---|---|---|
| E1 | 0.440 [0.329, 0.545] (37 of 84); within-set null p95 0.405 | not separated (interval contains the null's p95) |
| gain | 1.372 [1.094, 1.747] (70 cases; excluded {"t_b or t_b - 2 not entering": 2, "J/O*(t_b - 2) <= 0": 12}) | holds: H_token-event's second clause (interval wholly above 1) |
| E2 | 0.073 [0.005, 0.143]; J 0.525 [0.456, 0.593], mixed 0.451 [0.396, 0.517] (84 cases) | H_context-event not refuted |
| gates | 72 of 72 pass | all pass |
| yield | 84 cases on 41 of 64 neurons; per layer {"0": 3, "1": 19, "2": 39, "3": 23}; 0 poles declined | landed |
| GPU hours | 1.258 billed | landed |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| qualifying cases; neurons | 85 [60, 115]; 45 [30, 58] | under 40 cases (the floor) | 84; 41 | landed |
| births at T_12 or T_13, share | 0.55 [0.40, 0.70] | outside [0.2, 0.9] | 0.548 (live cases) | landed |
| E1 share; within-set null 95th percentile | 0.40 [0.28, 0.52]; 0.36 | share outside [0.1, 0.7] | 0.440 [0.329, 0.545]; 0.405 | landed |
| gain into t_b over elsewhere (sign rule) | 1.10 [0.90, 1.40] | outside [0.6, 2.5] | 1.372 [1.094, 1.747] | landed |
| E2 seed-free difference | 0.07 [0.00, 0.14] | outside [−0.1, 0.3] | 0.073 [0.005, 0.143] | landed |
| per-context locking share | 0.22 [0.14, 0.32] | outside [0.05, 0.6] | 0.191 [0.097, 0.283] | landed |
| clusterer re-asks per request | 0.17 [0.05, 0.35] | over 0.6 | 0.008 | landed |
| GPU hours (16 loads) | 2.5 [1.5, 4.0] | over 6 | 1.258 billed | landed |

Reported, no rule:

- E3 under g_l: r_J^g 3.247 [2.687, 3.921], r_O^g 1.832 [1.578, 2.571] (H_old-decline-global refuted, 82 cases).
- E4: 0.131 [0.050, 0.222] at the pair from T_12; null p95 0.214.
- Per-context locking (J within one step of t_b minus O*): 0.191 [0.097, 0.283]; J 0.336 [0.281, 0.392], O* 0.145 [0.078, 0.212].
- Calendar-matched percentile of the change into t_b: 0.655 [0.598, 0.712] (0.5 under no birth locking).
- SPEC5's 240 live cases (layers 4 to 31; 9 without g_l leave E2): E1 0.300 [0.240, 0.364] against its within-set p95 0.275; gain 1.092 [1.019, 1.204]; E2 seed-free 0.022 [-0.004, 0.049]; locking 0.143 [0.112, 0.174]; calendar 0.593 [0.554, 0.633].
- Births, new live cases: {"T_5": 2, "T_6": 1, "T_8": 4, "T_9": 5, "T_10": 11, "T_11": 15, "T_12": 17, "T_13": 29}; third set layers 0 to 3: {"T_4": 2, "T_6": 1, "T_7": 1, "T_8": 3, "T_9": 3, "T_10": 6, "T_11": 11, "T_12": 20, "T_13": 21}; share at T_12 or T_13 0.548 against 0.603.
- J/O* aligned to t_b (median, n): -8 0.122 (56), -7 0.142 (50), -6 0.165 (53), -5 0.136 (62), -4 0.159 (72), -3 0.177 (76), -2 0.225 (82), -1 0.325 (84), +0 0.583 (84), +1 0.791 (84), +2 0.914 (84), +3 0.896 (55).
- Clusterer calls: {"path": "studies/reuse/results/lowlayer/clusters/explain/calls.jsonl", "n_calls": 256, "n_requests": 256, "note": "calls.jsonl lists the accepted answers; the \u00a77 re-ask row reads clusterer_reasks"}.
- Clusterer re-asks per request (§7 row): 0.008 = (rejected rows {"s0": 0, "s1": 0} + withheld 2) / 256 requests.
- Yield per neuron (cases: neurons): {"0": 23, "1": 22, "2": 9, "3": 4, "4": 2, "5": 1, "6": 2, "7": 1}; dropped by filter: {"J_under_6": 111, "O_under_6": 43, "not_primary": 106, "shares_document": 47}.

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_harvest t00 | 0.0 | exemplar_a at T_1: 1.0143050319427145 | 0.0 | pass |
| known_instrument t00 | 0.0 | the same files at T_1 (the next load's bank): 0.9981177026438501 | 0.0 | pass |
| capture_identity t00 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 4.60123685002327, "a_P": 3.4325040227676773} | 0.0 | pass |
| completeness t00 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t01 | 0.0 | exemplar_a at T_0: 2.2948186684655094 | 0.0 | pass |
| known_instrument t01 | 0.0 | the same files at T_2 (the next load's bank): 0.9952223179611499 | 0.0 | pass |
| capture_identity t01 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 3.161673992872238, "a_P": 2.9121201077618553} | 0.0 | pass |
| completeness t01 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t02 | 0.0 | exemplar_a at T_1: 1.0150962115784212 | 0.0 | pass |
| known_instrument t02 | 0.0 | the same files at T_3 (the next load's bank): 0.7179965469464417 | 0.0 | pass |
| capture_identity t02 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 5.718022391200066, "a_P": 6.027216134973813} | 0.0 | pass |
| completeness t02 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t03 | 0.0 | exemplar_a at T_2: 0.8530281710882226 | 0.0 | pass |
| known_instrument t03 | 0.0 | the same files at T_4 (the next load's bank): 0.6336323248413726 | 0.0 | pass |
| capture_identity t03 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 13.466415509581566, "a_P": 11.529055428148865} | 0.0 | pass |
| completeness t03 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t04 | 0.0 | exemplar_a at T_3: 0.8863826345978523 | 0.0 | pass |
| known_instrument t04 | 0.0 | the same files at T_5 (the next load's bank): 0.7874641640734066 | 0.0 | pass |
| capture_identity t04 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 14.455907016992569, "a_P": 12.83730070910911} | 0.0 | pass |
| completeness t04 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t05 | 0.0 | exemplar_a at T_4: 0.8180204730094427 | 0.0 | pass |
| known_instrument t05 | 0.0 | the same files at T_6 (the next load's bank): 1.0004735117248453 | 0.0 | pass |
| capture_identity t05 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 9.100064434111118, "a_P": 7.605258519966559} | 0.0 | pass |
| completeness t05 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t06 | 0.0 | exemplar_a at T_5: 0.8942436053499871 | 0.0 | pass |
| known_instrument t06 | 0.0 | the same files at T_7 (the next load's bank): 0.28005592487237063 | 0.0 | pass |
| capture_identity t06 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 8.016834419220686, "a_P": 7.850349385726888} | 0.0 | pass |
| completeness t06 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t07 | 0.004081690782512926 | exemplar_a at T_6: 0.6471368423675639 | 0.0 | pass |
| known_instrument t07 | 0.0 | the same files at T_8 (the next load's bank): 2.1335168901933823 | 0.0 | pass |
| capture_identity t07 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 6.592189457267523, "a_P": 5.851347035846772} | 0.0 | pass |
| completeness t07 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t08 | 0.00336482894205516 | exemplar_a at T_7: 0.8288335060040479 | 0.0 | pass |
| known_instrument t08 | 0.0 | the same files at T_9 (the next load's bank): 1.8956701072260755 | 0.0 | pass |
| capture_identity t08 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.8277622312307358, "a_P": 0.8457560197175913} | 0.0 | pass |
| completeness t08 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t09 | 0.0 | exemplar_a at T_8: 0.7963961021020434 | 0.0 | pass |
| known_instrument t09 | 0.0 | the same files at T_10 (the next load's bank): 1.5315741475024567 | 0.0 | pass |
| capture_identity t09 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.35416791774332523, "a_P": 0.33887168285870817} | 0.0 | pass |
| completeness t09 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t10 | 0.0 | exemplar_a at T_9: 0.9052319427663196 | 0.0 | pass |
| known_instrument t10 | 0.0 | the same files at T_11 (the next load's bank): 0.832511509943656 | 0.0 | pass |
| capture_identity t10 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.07184542901813984, "a_P": 0.07005167185604595} | 0.0 | pass |
| completeness t10 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t11 | 0.0 | exemplar_a at T_10: 0.9147600281733306 | 0.0 | pass |
| known_instrument t11 | 0.0 | the same files at T_12 (the next load's bank): 0.6374016355135474 | 0.0 | pass |
| capture_identity t11 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.10546072386205196, "a_P": 0.11003654056701455} | 0.0 | pass |
| completeness t11 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t12 | 0.0 | exemplar_a at T_11: 1.0520183870205733 | 0.0 | pass |
| known_instrument t12 | 0.0 | the same files at T_13 (the next load's bank): 0.5950166547146059 | 0.0 | pass |
| capture_identity t12 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.23826169967651367, "a_P": 0.2606342975550888} | 0.0 | pass |
| completeness t12 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t13 | 0.0 | exemplar_a at T_12: 0.5291062425138613 | 0.0 | pass |
| known_instrument t13 | 0.0 | the same files at T_14 (the next load's bank): 0.5921603597555873 | 0.0 | pass |
| capture_identity t13 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.9904696084558964, "a_P": 0.9500951162328014} | 0.0 | pass |
| completeness t13 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t14 | 0.0 | exemplar_a at T_13: 0.4018438260813162 | 0.0 | pass |
| known_instrument t14 | 0.0 | the same files at T_15 (the next load's bank): 1.217840332092039 | 0.0 | pass |
| capture_identity t14 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.42797212954610586, "a_P": 0.42991049430071937} | 0.0 | pass |
| completeness t14 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| known_harvest t15 | 0.0 | exemplar_a at T_14: 0.4270159784274976 | 0.0 | pass |
| known_harvest_new t15 | 0.0 | topk.a in reversed rank order: 0.15367709312039515 | 0.0 | pass |
| known_instrument t15 | 0.0 | the same files at T_14 (the next load's bank): 0.5495729584808574 | 0.0 | pass |
| capture_identity t15 | {"a_h": 0.0, "a_P": 0.0} | neuron 7551 read at layer 1 from the second call (the hook on L + 1): {"a_h": 0.22647430980578065, "a_P": 0.2262338622489878} | 0.0 | pass |
| completeness t15 | {"poles_expected": 136, "poles_bad_or_missing": 0} | the first new pole's npz left out: {"poles_bad_or_missing": 1} | 0 | pass |
| routes | 3.8960411461277356e-08 | aP scaled by 1.1 on those positions (smallest over loads): 0.09999999734405496 | 0.0 | pass |
| clusterer | {"violations": 0, "poles_declined": 0} | L00N7551+'s sample 1 left out: {"violations": 1} | 0 | pass |
| dating_known | 0 | q999 of the next checkpoint (T_15: T_14): 147 | 0 | pass |
| global_scale | {"sha_mismatch": [], "t_index_mismatch": [], "g_nonpositive": []} | T_0's background.json replaced by T_1's: {"sha_mismatch": ["T_0"], "t_index_mismatch": ["T_0"], "g_nonpositive": []} | {"sha_mismatch": [], "t_index_mismatch": [], "g_nonpositive": []} | pass |
| scale_validity | {"n_nonpositive_T15": 0, "min_valid_share_from_T9": 0.9642857142857143} | the first case's s_P(T_15) set to 0: {"n_nonpositive_T15": 1} | None | pass |
| null_power | {"E1_null_p95": 0.40476190476190477} | the observed births shuffled, which must land inside the null: {"E1_shuffled_lo": 0.28735326485693324, "inside": true} | {"E1_share_break_at_every_tb": 1.0} | pass |
| case_count | {"live": 84, "neurons": 41} | the first 39 live cases: {"live": 39} | None | pass |
