Cases 243 (150 primary, 93 fresh) on 183 poles; live 149 primary, 91 fresh; loads 16 (t00..t15); contexts per load 4979.

# SPEC5 DATING verdict: §10 grid and §7 predictions

Every number is in `studies/reuse/results/dating/verdict_dating.json` with the artifact it was read from. r_J(t) = mean a_P over J at T_t / s_P(t); intervals: 95% percentile bootstrap over neurons, 2,000 draws, seed 0. The 150 primary cases are discovery, the 93 fresh cases confirmation; live as SPEC4 at birth.

| row | predicted | observed | verdict |
|---|---|---|---|
| D1 fresh | 0.35 | not read | not read (gates failed: scale, anchor) |
| D2 | 1.4; 0.7 | not read | not read (gates failed: scale, anchor) |
| D3 | 0.5 | not read | not read (gates failed: scale, anchor) |
| gates | pass | 48 of 50 pass | FAIL: scale, anchor |
| GPU hours | 1.8 | 1.038 h job wall time | landed |

| quantity | predicted | refuted if | observed | outcome |
|---|---|---|---|---|
| D1 share with a doubling pair, fresh | 0.35 [0.2, 0.5] | outside [0.1, 0.7] | not read | not read: gates failed scale, anchor |
| D2 r_J(t_b)/r_J(t_b−2); r_O | 1.4 [1.1, 1.8]; 0.7 [0.5, 0.9] | outside [0.8, 2.5]; outside [0.3, 1.2] | not read | not read: gates failed scale, anchor |
| D3 share within one step | 0.5 [0.35, 0.65] | outside [0.2, 0.8] | not read | not read: gates failed scale, anchor |
| largest adjacent ratio, median | 1.6 [1.3, 2.0] | outside [1.1, 3] | 2.246 [1.829, 2.911] | landed (gates failed: scale, anchor) |
| GPU hours | 1.8 [1.2, 2.6] | over 3.0 | 1.038 h of job wall time | landed |

| rule | primary (discovery) | fresh (confirmation) |
|---|---|---|
| D1 | not read (gates failed: scale, anchor) | not read (gates failed: scale, anchor) |
| D2 | not read (gates failed: scale, anchor) | not read (gates failed: scale, anchor) |
| D3 | not read (gates failed: scale, anchor) | not read (gates failed: scale, anchor) |

Curves (reported, no rule): median over live cases per checkpoint, valid pairs only (s_P(t) > 0.01 s_P(T_15)); valid share over all cases of the set.

| checkpoint | r_J primary | r_O primary | r_J fresh | r_O fresh | s_P median (fresh) | valid primary | valid fresh |
|---|---|---|---|---|---|---|---|
| T_0 | 15.802 | 19.392 | 2.510 | 5.348 | 0.000 | 0.200 | 0.290 |
| T_1 | 0.926 | 2.001 | 1.371 | 1.790 | 0.112 | 0.520 | 0.559 |
| T_2 | 1.008 | 2.232 | 1.755 | 3.069 | 0.457 | 0.613 | 0.613 |
| T_3 | 0.977 | 2.846 | 1.195 | 1.956 | 1.126 | 0.733 | 0.645 |
| T_4 | 1.139 | 2.587 | 1.170 | 1.999 | 8.672 | 0.820 | 0.763 |
| T_5 | 1.034 | 2.233 | 1.141 | 1.847 | 28.058 | 0.793 | 0.753 |
| T_6 | 0.990 | 1.902 | 1.029 | 1.670 | 41.399 | 0.893 | 0.796 |
| T_7 | 0.952 | 1.510 | 0.915 | 1.713 | 39.732 | 0.887 | 0.849 |
| T_8 | 0.909 | 1.606 | 0.848 | 1.607 | 25.467 | 0.940 | 0.903 |
| T_9 | 0.923 | 1.534 | 0.897 | 1.543 | 13.816 | 0.967 | 0.935 |
| T_10 | 0.900 | 1.447 | 0.928 | 1.431 | 6.361 | 0.987 | 1.000 |
| T_11 | 0.964 | 1.382 | 0.872 | 1.311 | 4.443 | 1.000 | 1.000 |
| T_12 | 0.956 | 1.275 | 0.902 | 1.264 | 4.825 | 1.000 | 1.000 |
| T_13 | 0.969 | 1.156 | 0.946 | 1.107 | 4.623 | 1.000 | 1.000 |
| T_14 | 0.976 | 1.099 | 0.983 | 1.068 | 3.669 | 1.000 | 1.000 |
| T_15 | 1.013 | 1.055 | 1.027 | 1.088 | 3.011 | 1.000 | 1.000 |

Doubling pairs (reported): primary 117 pairs, 12 at t_b; fresh 76 pairs, 9 at t_b.
Global scale: not available (the layer's median |ā| is banked by this run at T_15 only (t15/background.json) and by SPEC4 at T_5..T_13 and T_15 (results/birth/1/t*/background.json); T_0..T_4 and T_14 have none, so the 16-checkpoint statistics with s_P replaced by it cannot be formed).

B_15 stability and token coverage (reported, no rule): Jaccard of B_15 and the 1M-position top 1% 0.832; B_15 members' median share of positions with a > 0 0.551, with |a| > 0.5 |ā_200| 0.838; 1022000 positions.

| gate | value | failing input | identity | ok |
|---|---|---|---|---|
| known_number t00 | 0.0 | a_wrong at T_0: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.312225493873111 | 0.0 | pass |
| completeness t00 | 0.002849033335223794 | the column from the wrong layer: 0.3608144372701645 | None | pass |
| capture_identity t00 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 22.957395137263255 | 0.0 | pass |
| known_number t01 | 0.0 | a_wrong at T_1: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 1.2749011732901587 | 0.0 | pass |
| completeness t01 | 0.0015712773893028498 | the column from the wrong layer: 0.330911248922348 | None | pass |
| capture_identity t01 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 40.17219822275075 | 0.0 | pass |
| known_number t02 | 0.0 | a_wrong at T_2: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.08997557734586176 | 0.0 | pass |
| completeness t02 | 0.001593678374774754 | the column from the wrong layer: 0.3030632585287094 | None | pass |
| capture_identity t02 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 101.07223549978872 | 0.0 | pass |
| known_number t03 | 0.0 | a_wrong at T_3: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.08787699735994971 | 0.0 | pass |
| completeness t03 | 0.0013878460158593953 | the column from the wrong layer: 0.37077200412750244 | None | pass |
| capture_identity t03 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 111.21573591122953 | 0.0 | pass |
| known_number t04 | 0.0 | a_wrong at T_4: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12735407148425432 | 0.0 | pass |
| completeness t04 | 0.0008974310185294598 | the column from the wrong layer: 0.2756931185722351 | None | pass |
| capture_identity t04 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 135.99074532306835 | 0.0 | pass |
| known_number t05 | 0.0 | a_wrong at T_5: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.05933579160573861 | 0.0 | pass |
| completeness t05 | 0.001219068537466228 | the column from the wrong layer: 0.2321346029639244 | None | pass |
| capture_identity t05 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 135.499532384797 | 0.0 | pass |
| known_number t06 | 0.0 | a_wrong at T_6: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.06333445371846437 | 0.0 | pass |
| completeness t06 | 0.001118676271289587 | the column from the wrong layer: 0.22408752143383026 | None | pass |
| capture_identity t06 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 68.6832524422879 | 0.0 | pass |
| known_number t07 | 0.000819961463809643 | a_wrong at T_7: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.07868939426605578 | 0.0 | pass |
| completeness t07 | 0.0012617202010005713 | the column from the wrong layer: 0.32522740960121155 | None | pass |
| capture_identity t07 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 49.25235150080782 | 0.0 | pass |
| known_number t08 | 0.0008851542610905565 | a_wrong at T_8: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.05364612084112383 | 0.0 | pass |
| completeness t08 | 0.0011829049326479435 | the column from the wrong layer: 0.24265962094068527 | None | pass |
| capture_identity t08 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 20.36868023221645 | 0.0 | pass |
| known_number t09 | 0.00024701980075715423 | a_wrong at T_9: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.029578937538463485 | 0.0 | pass |
| completeness t09 | 0.0008651390089653432 | the column from the wrong layer: 0.23180870711803436 | None | pass |
| capture_identity t09 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 6.469266346242089 | 0.0 | pass |
| known_number t10 | 0.0007734560594757688 | a_wrong at T_10: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.1468453072399958 | 0.0 | pass |
| completeness t10 | 0.0012794611975550652 | the column from the wrong layer: 0.24074269086122513 | None | pass |
| capture_identity t10 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 3.02584609097348 | 0.0 | pass |
| known_number t11 | 0.0003021315977600342 | a_wrong at T_11: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12102014709698802 | 0.0 | pass |
| completeness t11 | 0.0009985077776946127 | the column from the wrong layer: 0.2558227553963661 | None | pass |
| capture_identity t11 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 1.8653475996204523 | 0.0 | pass |
| known_number t12 | 0.0008169767474135661 | a_wrong at T_12: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.12603903111757375 | 0.0 | pass |
| completeness t12 | 0.0009582680650055408 | the column from the wrong layer: 0.2692252993583679 | None | pass |
| capture_identity t12 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 1.68084524513291 | 0.0 | pass |
| known_number t13 | 0.0013254752973527384 | a_wrong at T_13: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.1288775479989188 | 0.0 | pass |
| completeness t13 | 0.001132183417212218 | the column from the wrong layer: 0.2754121795296669 | None | pass |
| capture_identity t13 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 1.5802226088846887 | 0.0 | pass |
| known_number t14 | 0.0017313122416193999 | a_wrong at T_14: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.13860545094057528 | 0.0 | pass |
| completeness t14 | 0.0010764695471152663 | the column from the wrong layer: 0.2910688892006874 | None | pass |
| capture_identity t14 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 0.8246320650097548 | 0.0 | pass |
| known_number t15 | 0.0 | a_wrong at T_15: the previous grid checkpoint's exemplar_a; at T_0 the next one's (T_1): 0.19487728677276725 | 0.0 | pass |
| completeness t15 | 0.0010306638723704964 | the column from the wrong layer: 0.3238200545310974 | None | pass |
| capture_identity t15 | 0.0 | P's neuron read at layer 3 (the hook on L-1): 0.35796789287815806 | 0.0 | pass |
| scale | {"n_nonpositive_T15": 0, "min_valid_share_from_T6": 0.9547325102880658, "max_rel_diff_T15": 0.5724742040093167} | each pole's captured T_15 median against the bank median of the next pole: 120.3071548956629 | 0.0 | FAIL |
| anchor | 0.02757504681234968 | SPEC4's rho_P_J of the next case: 3.211690531179989 | 0.0 | FAIL |
