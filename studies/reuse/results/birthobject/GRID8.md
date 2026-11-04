Cases 127 on 65 neurons (80 poles); live 127; E1 127, E2 127 (0 live cases under 6 contexts with a jump), E4 127; loads 16 (t00..t15); contexts per load 2645.

# SPEC8 BIRTHOBJECT verdict: §10 grid and §7 predictions

Every number is in `studies/reuse/results/birthobject/verdict_birthobject.json` with the artifact it was read from. Intervals: 95% percentile bootstrap over neurons, 2,000 draws, seed 0; nulls 2,000 draws. s_P(t) is the scale stage's median at each load; g_l(t) the layer's median |ā| at each load; the log-rate x-axis is log steps.

| rule | reading | outcome |
|---|---|---|
| E1 | 0.307 [0.225, 0.395] (39 of 127); null p95 0.283 | not separated |
| E2 | difference 0.038 [-0.020, 0.097]; J 0.540 [0.494, 0.585], mixed 0.501 [0.450, 0.553] (127 cases) | H_context-event refuted |
| E3 | r_J^g 2.427 [2.050, 3.434]; r_O^g 1.867 [1.704, 2.119] (122 cases) | H_old-decline-global refuted |
| E4 | 0.110 [0.060, 0.171] at the pair (T_8, T_9); null p95 0.189 (127 cases) | H_drift not refuted |
| gates | 100 of 100 pass | all pass |
| GPU hours | 1.106 h job wall time | landed |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| E1 share at the birth; shuffled null 95th percentile | 0.30 [0.22, 0.38]; 0.30 | share outside [0.05, 0.55] | 0.307 [0.225, 0.395]; null p95 0.283 | landed |
| E2 concentration of J; of the same pole's mixed set | 0.44 [0.38, 0.50]; 0.43 | outside [0.25, 0.65] | 0.540 [0.494, 0.585]; 0.501 [0.450, 0.553] | J landed; mixed landed |
| E3 J ratio; O* ratio under the global scale | 1.15 [1.0, 1.35]; 0.85 [0.7, 1.0] | outside [0.7, 2.0]; outside [0.4, 1.3] | 2.427 [2.050, 3.434]; 1.867 [1.704, 2.119] | J refuted; O* refuted |
| E4 share at the modal checkpoint; null 95th percentile | 0.13 [0.07, 0.20]; 0.18 | outside [0.05, 0.5] | 0.110 [0.060, 0.171]; null p95 0.189 | landed |
| two-step gain into t_b over the case's gain elsewhere | 1.0 [0.9, 1.1] | outside [0.7, 1.5] | 1.052 [0.985, 1.266] | landed |
| GPU hours (16 loads, two stages and the ā pass) | 1.8 [1.3, 2.6] | over 3.5 | 1.106 h of job wall time | landed |

Reported, no rule:

- O*'s own concentration 0.585 [0.526, 0.647]; contexts from six different cases of the same tercile, null p95 0.377.
- Two-step gain into t_b over the case's gain elsewhere 1.052 [0.985, 1.266] (SPEC5's fresh set: 0.97 [0.85, 1.06]).
- J/O* aligned to t_b (median, n): -8 0.189 (76), -7 0.261 (91), -6 0.296 (93), -5 0.349 (99), -4 0.404 (111), -3 0.359 (113), -2 0.431 (121), -1 0.418 (127), +0 0.574 (127), +1 0.755 (127), +2 0.871 (127), +3 0.890 (94).

| SPEC5 rule under scale | D1 share with a doubling | D2 r_J; r_O (form) | D3 share within one step |
|---|---|---|---|
| median | 0.528 [0.426, 0.624] | 1.042 [0.937, 1.192]; 0.901 [0.816, 0.959] (H_old-decline refuted) | 0.094 [0.046, 0.154] |
| mean | 0.567 [0.477, 0.653] | 1.178 [1.053, 1.412]; 0.921 [0.871, 0.972] (H_old-decline refuted) | 0.150 [0.076, 0.230] |
| global | 0.969 [0.933, 0.993] | 2.393 [1.990, 3.053]; 1.853 [1.674, 2.119] (H_old-decline refuted) | 0.693 [0.591, 0.792] |

| checkpoint | valid share |
|---|---|
| T_0 | 0.307 |
| T_1 | 0.677 |
| T_2 | 0.685 |
| T_3 | 0.732 |
| T_4 | 0.811 |
| T_5 | 0.827 |
| T_6 | 0.866 |
| T_7 | 0.874 |
| T_8 | 0.937 |
| T_9 | 0.945 |
| T_10 | 0.937 |
| T_11 | 0.976 |
| T_12 | 1.000 |
| T_13 | 1.000 |
| T_14 | 1.000 |
| T_15 | 1.000 |

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t00 | 0.0 | a_wrong at T_0: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.312225493873111 | 0.0 | pass |
| completeness t00 | 0.002849033335223794 | the column from the wrong layer: 0.3608144372701645 | None | pass |
| capture_identity t00 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 8.18805296777021 | 0.0 | pass |
| known_number t01 | 0.0 | a_wrong at T_1: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 1.2749011732901587 | 0.0 | pass |
| completeness t01 | 0.0015712773893028498 | the column from the wrong layer: 0.330911248922348 | None | pass |
| capture_identity t01 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 21.317945692179805 | 0.0 | pass |
| known_number t02 | 0.0 | a_wrong at T_2: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.08997557734586176 | 0.0 | pass |
| completeness t02 | 0.001593678374774754 | the column from the wrong layer: 0.3030632585287094 | None | pass |
| capture_identity t02 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 61.39465717466005 | 0.0 | pass |
| known_number t03 | 0.0 | a_wrong at T_3: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.08787699735994971 | 0.0 | pass |
| completeness t03 | 0.0013878460158593953 | the column from the wrong layer: 0.37077200412750244 | None | pass |
| capture_identity t03 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 67.63589046497061 | 0.0 | pass |
| known_number t04 | 0.0 | a_wrong at T_4: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12735407148425432 | 0.0 | pass |
| completeness t04 | 0.0008974310185294598 | the column from the wrong layer: 0.2756931185722351 | None | pass |
| capture_identity t04 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 66.36422008834478 | 0.0 | pass |
| known_number t05 | 0.0 | a_wrong at T_5: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.05933579160573861 | 0.0 | pass |
| completeness t05 | 0.001219068537466228 | the column from the wrong layer: 0.2321346029639244 | None | pass |
| capture_identity t05 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 60.34481226202834 | 0.0 | pass |
| known_number t06 | 0.0 | a_wrong at T_6: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.06333445371846437 | 0.0 | pass |
| completeness t06 | 0.001118676271289587 | the column from the wrong layer: 0.22408752143383026 | None | pass |
| capture_identity t06 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 35.73474033313848 | 0.0 | pass |
| known_number t07 | 0.000819961463809643 | a_wrong at T_7: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.07868939426605578 | 0.0 | pass |
| completeness t07 | 0.0012617202010005713 | the column from the wrong layer: 0.32522740960121155 | None | pass |
| capture_identity t07 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 15.93074297166709 | 0.0 | pass |
| known_number t08 | 0.0008851542610905565 | a_wrong at T_8: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.05364612084112383 | 0.0 | pass |
| completeness t08 | 0.0011829049326479435 | the column from the wrong layer: 0.24265962094068527 | None | pass |
| capture_identity t08 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 6.4002025665283835 | 0.0 | pass |
| known_number t09 | 0.00024701980075715423 | a_wrong at T_9: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.029578937538463485 | 0.0 | pass |
| completeness t09 | 0.0008651390089653432 | the column from the wrong layer: 0.23180870711803436 | None | pass |
| capture_identity t09 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 2.273025476314498 | 0.0 | pass |
| known_number t10 | 0.0007734560594757688 | a_wrong at T_10: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.1468453072399958 | 0.0 | pass |
| completeness t10 | 0.0012794611975550652 | the column from the wrong layer: 0.24074269086122513 | None | pass |
| capture_identity t10 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 1.113207856146224 | 0.0 | pass |
| known_number t11 | 0.0003021315977600342 | a_wrong at T_11: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12102014709698802 | 0.0 | pass |
| completeness t11 | 0.0009985077776946127 | the column from the wrong layer: 0.2558227553963661 | None | pass |
| capture_identity t11 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.9898364827415431 | 0.0 | pass |
| known_number t12 | 0.0008169767474135661 | a_wrong at T_12: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12603903111757375 | 0.0 | pass |
| completeness t12 | 0.0009582680650055408 | the column from the wrong layer: 0.2692252993583679 | None | pass |
| capture_identity t12 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.980696578327226 | 0.0 | pass |
| known_number t13 | 0.0013254752973527384 | a_wrong at T_13: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.1288775479989188 | 0.0 | pass |
| completeness t13 | 0.001132183417212218 | the column from the wrong layer: 0.2754121795296669 | None | pass |
| capture_identity t13 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 1.222937115951739 | 0.0 | pass |
| known_number t14 | 0.0017313122416193999 | a_wrong at T_14: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.13860545094057528 | 0.0 | pass |
| completeness t14 | 0.0010764695471152663 | the column from the wrong layer: 0.2910688892006874 | None | pass |
| capture_identity t14 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.5137373548591616 | 0.0 | pass |
| known_number t15 | 0.0 | a_wrong at T_15: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.19487728677276725 | 0.0 | pass |
| completeness t15 | 0.0010306638723704964 | the column from the wrong layer: 0.3238200545310974 | None | pass |
| capture_identity t15 | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.2634161894302181 | 0.0 | pass |
| known_number t00/scale_stage | 0.0 | a_wrong at T_0: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.312225493873111 | 0.0 | pass |
| completeness t00/scale_stage | 0.002849033335223794 | the column from the wrong layer: 0.3608144372701645 | None | pass |
| capture_identity t00/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 8.18805296777021 | 0.0 | pass |
| known_number t01/scale_stage | 0.0 | a_wrong at T_1: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 1.2749011732901587 | 0.0 | pass |
| completeness t01/scale_stage | 0.0015712773893028498 | the column from the wrong layer: 0.330911248922348 | None | pass |
| capture_identity t01/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 21.317945692179805 | 0.0 | pass |
| known_number t02/scale_stage | 0.0 | a_wrong at T_2: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.08997557734586176 | 0.0 | pass |
| completeness t02/scale_stage | 0.001593678374774754 | the column from the wrong layer: 0.3030632585287094 | None | pass |
| capture_identity t02/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 61.39465717466005 | 0.0 | pass |
| known_number t03/scale_stage | 0.0 | a_wrong at T_3: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.08787699735994971 | 0.0 | pass |
| completeness t03/scale_stage | 0.0013878460158593953 | the column from the wrong layer: 0.37077200412750244 | None | pass |
| capture_identity t03/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 67.63589046497061 | 0.0 | pass |
| known_number t04/scale_stage | 0.0 | a_wrong at T_4: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12735407148425432 | 0.0 | pass |
| completeness t04/scale_stage | 0.0008974310185294598 | the column from the wrong layer: 0.2756931185722351 | None | pass |
| capture_identity t04/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 66.36422008834478 | 0.0 | pass |
| known_number t05/scale_stage | 0.0 | a_wrong at T_5: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.05933579160573861 | 0.0 | pass |
| completeness t05/scale_stage | 0.001219068537466228 | the column from the wrong layer: 0.2321346029639244 | None | pass |
| capture_identity t05/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 60.34481226202834 | 0.0 | pass |
| known_number t06/scale_stage | 0.0 | a_wrong at T_6: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.06333445371846437 | 0.0 | pass |
| completeness t06/scale_stage | 0.001118676271289587 | the column from the wrong layer: 0.22408752143383026 | None | pass |
| capture_identity t06/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 35.73474033313848 | 0.0 | pass |
| known_number t07/scale_stage | 0.000819961463809643 | a_wrong at T_7: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.07868939426605578 | 0.0 | pass |
| completeness t07/scale_stage | 0.0012617202010005713 | the column from the wrong layer: 0.32522740960121155 | None | pass |
| capture_identity t07/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 15.93074297166709 | 0.0 | pass |
| known_number t08/scale_stage | 0.0008851542610905565 | a_wrong at T_8: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.05364612084112383 | 0.0 | pass |
| completeness t08/scale_stage | 0.0011829049326479435 | the column from the wrong layer: 0.24265962094068527 | None | pass |
| capture_identity t08/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 6.4002025665283835 | 0.0 | pass |
| known_number t09/scale_stage | 0.00024701980075715423 | a_wrong at T_9: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.029578937538463485 | 0.0 | pass |
| completeness t09/scale_stage | 0.0008651390089653432 | the column from the wrong layer: 0.23180870711803436 | None | pass |
| capture_identity t09/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 2.273025476314498 | 0.0 | pass |
| known_number t10/scale_stage | 0.0007734560594757688 | a_wrong at T_10: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.1468453072399958 | 0.0 | pass |
| completeness t10/scale_stage | 0.0012794611975550652 | the column from the wrong layer: 0.24074269086122513 | None | pass |
| capture_identity t10/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 1.113207856146224 | 0.0 | pass |
| known_number t11/scale_stage | 0.0003021315977600342 | a_wrong at T_11: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12102014709698802 | 0.0 | pass |
| completeness t11/scale_stage | 0.0009985077776946127 | the column from the wrong layer: 0.2558227553963661 | None | pass |
| capture_identity t11/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.9898364827415431 | 0.0 | pass |
| known_number t12/scale_stage | 0.0008169767474135661 | a_wrong at T_12: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12603903111757375 | 0.0 | pass |
| completeness t12/scale_stage | 0.0009582680650055408 | the column from the wrong layer: 0.2692252993583679 | None | pass |
| capture_identity t12/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.980696578327226 | 0.0 | pass |
| known_number t13/scale_stage | 0.0013254752973527384 | a_wrong at T_13: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.1288775479989188 | 0.0 | pass |
| completeness t13/scale_stage | 0.001132183417212218 | the column from the wrong layer: 0.2754121795296669 | None | pass |
| capture_identity t13/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 1.222937115951739 | 0.0 | pass |
| known_number t14/scale_stage | 0.0017313122416193999 | a_wrong at T_14: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.13860545094057528 | 0.0 | pass |
| completeness t14/scale_stage | 0.0010764695471152663 | the column from the wrong layer: 0.2910688892006874 | None | pass |
| capture_identity t14/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.5137373548591616 | 0.0 | pass |
| known_number t15/scale_stage | 0.0 | a_wrong at T_15: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.19487728677276725 | 0.0 | pass |
| completeness t15/scale_stage | 0.0010306638723704964 | the column from the wrong layer: 0.3238200545310974 | None | pass |
| capture_identity t15/scale_stage | 0.0 | P's neuron read at layer 0 (the hook on L-1): 0.2634161894302181 | 0.0 | pass |
| known_number scale T_15 | 0.013077018536252542 | the harvest bank's s_P(T_14) in place of s_P(T_15): 0.3142947644780021 | 0.0 | pass |
| scale | {"n_nonpositive_T15": 0, "min_valid_share_from_T9": 0.937007874015748, "max_abs_diff_T15": 0.0} | each pole's scale-stage T_15 median against the dating stage's T_15 median of the next pole: 18.189331720280094 | 0.0 | pass |
| global_scale | {"median_rel_diff_T15": 0.0008782669465173612, "g_min_all_loads": 0.00026425463147461414} | SPEC4's T_13 per-layer median |ā| in place of its T_15: 0.5313984609674522 | 0.0 | pass |
| null_power | {"E1_null_p95": 0.28346456692913385, "E4_null_p95": 0.1889763779527559} | the observed births shuffled (E1) and one random pair per case (E4), which must land inside their nulls: {"E1_shuffled_interval_lo": 0.1450381679389313, "E1_shuffled_inside": true, "E4_shuffled_share": 0.15748031496062992, "E4_shuffled_inside": true} | {"E1_share_break_set_to_tb": 1.0, "E4_share_all_at_one_pair": 1.0} | pass |
