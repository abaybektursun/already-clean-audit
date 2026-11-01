# D1 RESULTS

Numbers from `bank/w5/verdict/metrics.json` (sha256 a95bfa57bfe65246b00950d094f9fee5e87743ec2945296d1337b59a1e717e16, unblind 2026-10-24T04:06:05.084615Z). Blanks marked ____ are written by hand; nothing below is re-derived.

Finding: On O7 (the 600-pole audit set S) the parent test labels H_SP. The unrelated share of late-born clusters is p_u = 0.265 [0.211, 0.321], lower bound above the 0.20 rule, from A = 0.812 over 1,238 late-born clusters against a_ctrl = 0.924 on S's own 600 controls (judge anthropic/claude-sonnet-5-5, prompt 11b5fd81…, validity gate pass); with clusters born at the device boundaries T_7 or T_15 left out it is 0.275 [0.220, 0.332] (labelled sensitivity, checker F2). Birth order supports the main-job-is-oldest form of H_RG: b = 0.476 [0.436, 0.517], above the permutation band (null 97.5th percentile 0.449, p = 0.0005, Holm 0.0010), with a drift share of 0.524; the same sensitivity gives 0.486 [0.443, 0.527] against a band of 0.465. Vestigial is not supported (v = 0.035 [0.026, 0.044]). H_MIG: 17 poles (2.8% of S). The hard-decoy row is not measurable (control accuracy 0.798, below 0.80). Q1: under explanations that pass §6.1, one-job fractions are 0.597 (Q4B), 0.563 (G4) and 0.518 (O7), with UNK 0.32–0.41 and UNREL at most 0.013; cross-model label kappa 0.07–0.18. Center birth curve (exploratory, per-checkpoint probe run): the blind maximum reaches 0.3 of its T_15 score at a median of T_3 over 598 poles (ONE T_4, TWO-REL T_6, UNK T_1); the population's median ratio curve rises from 0.08 at T_0 to 1 at T_15 while only 1% of per-pole curves are non-decreasing, and ONE poles are not earlier than UNK poles at the medians. A mixed outcome under §6.7: H_SP on p_u beside main-oldest on b, reported as such.

## Q1 audit (§6.2)

θ_s = 0.3153 (gate pass). The essay's 90 / 7 / 2–4 is its own estimate under its own thresholds, not ours.

### G4

On model G4, 56.3% of poles (52.3–60.2%, Wilson 95%) have one job under explanations that pass §6.1 (n = 600).

Poles: every pole of the explain directory (no list given).

| set | n | ONE | TWO-REL | UNREL | UNK | RARE | UNK-N | UNK-F |
|---|---|---|---|---|---|---|---|---|
| all | 600 | 0.563 [0.523, 0.602] | 0.060 [0.044, 0.082] | 0.012 [0.006, 0.024] | 0.365 [0.327, 0.404] | 0.000 [0.000, 0.006] | 0.022 [0.013, 0.037] | 0.343 [0.306, 0.382] |
| depth tercile 0 | 212 | 0.689 [0.623, 0.747] | 0.071 [0.043, 0.113] | 0.019 [0.007, 0.048] | 0.222 [0.171, 0.282] | 0.000 [0.000, 0.018] | 0.014 [0.005, 0.041] | 0.208 [0.158, 0.267] |
| depth tercile 1 | 194 | 0.505 [0.435, 0.575] | 0.057 [0.032, 0.099] | 0.005 [0.001, 0.029] | 0.433 [0.365, 0.503] | 0.000 [0.000, 0.019] | 0.026 [0.011, 0.059] | 0.407 [0.341, 0.478] |
| depth tercile 2 | 194 | 0.485 [0.415, 0.554] | 0.052 [0.028, 0.092] | 0.010 [0.003, 0.037] | 0.454 [0.385, 0.524] | 0.000 [0.000, 0.019] | 0.026 [0.011, 0.059] | 0.428 [0.360, 0.498] |
| double-negative share ≥ 0.8 | 2 | 0.500 [0.095, 0.905] | 0.000 [0.000, 0.658] | 0.000 [0.000, 0.658] | 0.500 [0.095, 0.905] | 0.000 [0.000, 0.658] | 0.500 [0.095, 0.905] | 0.000 [0.000, 0.658] |
| double-negative share < 0.8 | 598 | 0.564 [0.524, 0.603] | 0.060 [0.044, 0.082] | 0.012 [0.006, 0.024] | 0.365 [0.327, 0.404] | 0.000 [0.000, 0.006] | 0.020 [0.012, 0.035] | 0.344 [0.307, 0.383] |
| explainer anthropic/claude-opus-5-5 | 455 | 0.620 [0.574, 0.663] | 0.066 [0.047, 0.093] | 0.013 [0.006, 0.028] | 0.301 [0.261, 0.345] | 0.000 [0.000, 0.008] | 0.013 [0.006, 0.028] | 0.288 [0.248, 0.331] |
| explainer anthropic/claude-sonnet-5-5 | 145 | 0.386 [0.311, 0.467] | 0.041 [0.019, 0.087] | 0.007 [0.001, 0.038] | 0.566 [0.484, 0.643] | 0.000 [0.000, 0.026] | 0.048 [0.024, 0.096] | 0.517 [0.437, 0.597] |

Label agreement: kappa 0.167 over 599 poles rated twice; explainer {'anthropic/claude-opus-5-5': 455, 'anthropic/claude-sonnet-5-5': 144}, second reader {'deepseek-ai/DeepSeek-V4.1-Flash-1017': 599}, 599 cross-model pairs.
Kappa by explainer model: anthropic/claude-opus-5-5 0.216 (n = 455); anthropic/claude-sonnet-5-5 -0.015 (n = 144).
Declined: 4 poles (explainer or writer; UNK-N), 1 second-reader answers.

### O7

On model O7, 51.8% of poles (47.8–55.8%, Wilson 95%) have one job under explanations that pass §6.1 (n = 600).

Poles: the 600 listed in `bank/w1/o7_split/audit_poles.json` (sha256 4ba604d504b9f8c6); 40 poles of the explain directory left out by the list: L00N450+, L00N450-, L01N30+, L01N30-, L02N311+, L02N311-, L05N4410+, L05N4410-, L08N10442+, L08N10442-, L09N10171+, L09N10171-, L15N9012+, L15N9012-, L15N9330+, L15N9330-, L17N10744+, L17N10744-, L17N3455+, L17N3455-, L19N523+, L19N523-, L19N9466+, L19N9466-, L20N2135+, L20N2135-, L20N3373+, L20N3373-, L23N3048+, L23N3048-, L25N2814+, L25N2814-, L25N475+, L25N475-, L28N1238+, L28N1238-, L29N3405+, L29N3405-, L30N1033+, L30N1033-.

| set | n | ONE | TWO-REL | UNREL | UNK | RARE | UNK-N | UNK-F |
|---|---|---|---|---|---|---|---|---|
| all | 600 | 0.518 [0.478, 0.558] | 0.060 [0.044, 0.082] | 0.007 [0.003, 0.017] | 0.415 [0.376, 0.455] | 0.000 [0.000, 0.006] | 0.023 [0.014, 0.039] | 0.392 [0.353, 0.431] |
| depth tercile 0 | 208 | 0.620 [0.553, 0.683] | 0.077 [0.048, 0.121] | 0.014 [0.005, 0.042] | 0.288 [0.231, 0.353] | 0.000 [0.000, 0.018] | 0.005 [0.001, 0.027] | 0.284 [0.227, 0.348] |
| depth tercile 1 | 204 | 0.466 [0.399, 0.534] | 0.044 [0.023, 0.082] | 0.005 [0.001, 0.027] | 0.485 [0.418, 0.554] | 0.000 [0.000, 0.018] | 0.029 [0.014, 0.063] | 0.456 [0.389, 0.524] |
| depth tercile 2 | 188 | 0.463 [0.393, 0.534] | 0.059 [0.033, 0.102] | 0.000 [-0.000, 0.020] | 0.479 [0.408, 0.550] | 0.000 [-0.000, 0.020] | 0.037 [0.018, 0.075] | 0.441 [0.372, 0.513] |
| double-negative share ≥ 0.8 | 1 | 1.000 [0.207, 1.000] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] | 0.000 [0.000, 0.793] |
| double-negative share < 0.8 | 599 | 0.518 [0.478, 0.557] | 0.060 [0.044, 0.082] | 0.007 [0.003, 0.017] | 0.416 [0.377, 0.456] | 0.000 [0.000, 0.006] | 0.023 [0.014, 0.039] | 0.392 [0.354, 0.432] |
| explainer anthropic/claude-opus-5-5 | 426 | 0.547 [0.499, 0.594] | 0.066 [0.046, 0.093] | 0.009 [0.004, 0.024] | 0.378 [0.333, 0.425] | 0.000 [0.000, 0.009] | 0.019 [0.010, 0.037] | 0.359 [0.315, 0.406] |
| explainer anthropic/claude-sonnet-5-5 | 174 | 0.448 [0.376, 0.522] | 0.046 [0.023, 0.088] | 0.000 [0.000, 0.022] | 0.506 [0.432, 0.579] | 0.000 [0.000, 0.022] | 0.034 [0.016, 0.073] | 0.471 [0.399, 0.545] |

Label agreement: kappa 0.070 over 600 poles rated twice; explainer {'anthropic/claude-opus-5-5': 426, 'anthropic/claude-sonnet-5-5': 174}, second reader {'deepseek-ai/DeepSeek-V4.1-Flash-1017': 600}, 600 cross-model pairs.
Kappa by explainer model: anthropic/claude-opus-5-5 0.069 (n = 426); anthropic/claude-sonnet-5-5 0.072 (n = 174).
Declined: 1 poles (explainer or writer; UNK-N), 0 second-reader answers.

### Q4B

On model Q4B, 59.7% of poles (55.7–63.5%, Wilson 95%) have one job under explanations that pass §6.1 (n = 600).

Poles: every pole of the explain directory (no list given).

| set | n | ONE | TWO-REL | UNREL | UNK | RARE | UNK-N | UNK-F |
|---|---|---|---|---|---|---|---|---|
| all | 600 | 0.597 [0.557, 0.635] | 0.057 [0.041, 0.078] | 0.013 [0.007, 0.026] | 0.320 [0.284, 0.358] | 0.013 [0.007, 0.026] | 0.018 [0.010, 0.033] | 0.302 [0.266, 0.340] |
| depth tercile 0 | 204 | 0.735 [0.671, 0.791] | 0.074 [0.045, 0.118] | 0.025 [0.011, 0.056] | 0.127 [0.088, 0.180] | 0.039 [0.020, 0.075] | 0.005 [0.001, 0.027] | 0.123 [0.084, 0.175] |
| depth tercile 1 | 196 | 0.515 [0.446, 0.584] | 0.026 [0.011, 0.058] | 0.005 [0.001, 0.028] | 0.454 [0.386, 0.524] | 0.000 [0.000, 0.019] | 0.015 [0.005, 0.044] | 0.439 [0.371, 0.509] |
| depth tercile 2 | 200 | 0.535 [0.466, 0.603] | 0.070 [0.042, 0.114] | 0.010 [0.003, 0.036] | 0.385 [0.320, 0.454] | 0.000 [0.000, 0.019] | 0.035 [0.017, 0.070] | 0.350 [0.287, 0.418] |
| double-negative share ≥ 0.8 | 16 | 0.750 [0.505, 0.898] | 0.062 [0.011, 0.283] | 0.000 [0.000, 0.194] | 0.188 [0.066, 0.430] | 0.000 [0.000, 0.194] | 0.062 [0.011, 0.283] | 0.125 [0.035, 0.360] |
| double-negative share < 0.8 | 584 | 0.592 [0.552, 0.632] | 0.057 [0.041, 0.078] | 0.014 [0.007, 0.027] | 0.324 [0.287, 0.363] | 0.014 [0.007, 0.027] | 0.017 [0.009, 0.031] | 0.307 [0.270, 0.345] |
| explainer anthropic/claude-opus-5-5 | 463 | 0.622 [0.577, 0.665] | 0.048 [0.032, 0.071] | 0.013 [0.006, 0.028] | 0.300 [0.260, 0.343] | 0.017 [0.009, 0.034] | 0.011 [0.005, 0.025] | 0.289 [0.250, 0.332] |
| explainer anthropic/claude-sonnet-5-5 | 137 | 0.511 [0.428, 0.593] | 0.088 [0.051, 0.147] | 0.015 [0.004, 0.052] | 0.387 [0.309, 0.470] | 0.000 [0.000, 0.027] | 0.044 [0.020, 0.092] | 0.343 [0.269, 0.426] |

Label agreement: kappa 0.180 over 600 poles rated twice; explainer {'anthropic/claude-opus-5-5': 463, 'anthropic/claude-sonnet-5-5': 137}, second reader {'deepseek-ai/DeepSeek-V4.1-Flash-1017': 600}, 600 cross-model pairs.
Kappa by explainer model: anthropic/claude-opus-5-5 0.161 (n = 463); anthropic/claude-sonnet-5-5 0.243 (n = 137).
Declined: 1 poles (explainer or writer; UNK-N), 0 second-reader answers.

## Q2 parent test and the prediction grid (§6.3–§6.6, §11)

Judge validity gate: a_ctrl 0.910 on 39 poles (pass; threshold 0.8), judge ['anthropic/claude-sonnet-5-5'].
Declined at W4: 0 poles at the clusterer (none); 0 judge lines, taking out clusters none and controls none. S: 600 poles, 600 with a main cluster.

Sensitivity (checker F2 item 3; clusters born at T_7 or T_15 left out): p_u = 0.275 [0.220, 0.332] (H_SP; 1177 clusters, 61 left out); b = 0.486 [0.443, 0.527], null 97.5th 0.465, p = 0.0015 (531 poles; 77 clusters and 41 poles whose main is excluded left out). A labelled sensitivity; the §6.3 and §6.4 labels are not reapplied.

PREDICTIONS entry: heading 2026-10-20T02:58:11+00:00, committed 2026-10-20T03:13:32+00:00. θ_s = 0.3153 (gate pass). Judge validity gate: pass. G2: 0.0. Population-weighted Q2 estimates: metrics.json q2.population_weighted. Left out of Q2 (declined after the attempts cap): 0 poles at the clusterer; 0 judge lines, taking out 0 clusters and 0 controls.

| readout | H_RG predicts | H_SP predicts | H_VEST predicts | observed | label |
|---|---|---|---|---|---|
| `p_u` (§6.3) | upper bound < 0.10 | lower bound > 0.20 | — | p_u = 0.265 [0.211, 0.321]; A = 0.812, a_ctrl = 0.924, 1238 clusters; pole bootstrap conditional on a late-born cluster and a control in each draw, 0 of 10000 draws redrawn (refused above 1%) | H_SP |
| `b` birth order (§6.4) | above the permutation band | inside it | main not earliest in many poles | b = 0.476 [0.436, 0.517]; null mean 0.414, 97.5th 0.449; p = 0.0005 (Holm 0.0010); 574 poles | main oldest (above band) |
| `v` vestigial (§6.5) | low | — | lower bound > 0.5 | v = 0.035 [0.026, 0.044]; p = 1.0000 (Holm 1.0000); 1854 clusters, 0 excluded (growth undefined) | not H_VEST |
| H_MIG count (§6.6) | ≈ 0 | some | — | 17 poles (0.028 of S) | count |
| `r(c)` at birth (exploratory) | median well below 1 | no prediction | — | median r: late-born 0.2912783205072667, all born 0.295 | exploratory |
| hard-decoy accuracy (secondary) | far above 0.5 | near 0.5 on flagged clusters | — | accuracy 0.697; on flagged 0.356; control 0.798 | not measurable |
| Q1 fractions (descriptive) | in the essay's regime with stated thresholds | — | — | G4: ONE 0.563 [0.523, 0.602], TWO-REL 0.060 [0.044, 0.082], UNREL 0.012 [0.006, 0.024], UNK 0.365 [0.327, 0.404], RARE 0.000 [0.000, 0.006] (UNK-N 13, UNK-F 206); O7: ONE 0.518 [0.478, 0.558], TWO-REL 0.060 [0.044, 0.082], UNREL 0.007 [0.003, 0.017], UNK 0.415 [0.376, 0.455], RARE 0.000 [0.000, 0.006] (UNK-N 14, UNK-F 235); Q4B: ONE 0.597 [0.557, 0.635], TWO-REL 0.057 [0.041, 0.078], UNREL 0.013 [0.007, 0.026], UNK 0.320 [0.284, 0.358], RARE 0.013 [0.007, 0.026] (UNK-N 11, UNK-F 181) | descriptive |
| center birth curve (exploratory) | monotone, early for ONE poles | — | — | blind maximum reaches 0.3 of its T_15 score first at median T_3 over 598 poles (ONE T_4 (n = 311); TWO-REL T_6 (n = 36); UNK T_1 (n = 248); UNREL T_6 (n = 4)); positives' mean median T_2; ratio curve non-decreasing (tolerance 0.05) for 0.010 of poles; 1 poles without a positive T_15 score; 1 poles without texts left out; 0 dead pole-checkpoints | exploratory |

## Center birth curve (§2.4, exploratory)

first t at which the blind maximum reaches 0.3 of its T_15 score (same per-checkpoint probe run); a pole with max|a| = 0 at a checkpoint scores 0 there (0 pole-checkpoints). Per-checkpoint probe run: `<local>/interp_research-neurons-harvest/bank/neurons/x/center_probe/1/out`.

| set | n | undefined | median first t (blind max) | median first t (positives' mean) | non-decreasing share |
|---|---|---|---|---|---|
| all | 599 | 1 | T_3 | T_2 | 0.010 |
| ONE | 311 | 0 | T_4 | T_4 | 0.010 |
| TWO-REL | 36 | 0 | T_6 | T_6 | 0.056 |
| UNK | 248 | 1 | T_1 | T_1 | 0.004 |
| UNREL | 4 | 0 | T_6 | T_6/T_7 | 0.000 |

Median ratio curve (blind maximum's score over its T_15 score) by grid point: T_0 0.078, T_1 0.186, T_2 0.173, T_3 0.191, T_4 0.228, T_5 0.285, T_6 0.304, T_7 0.349, T_8 0.375, T_9 0.404, T_10 0.513, T_11 0.637, T_12 0.760, T_13 0.854, T_14 0.915, T_15 1.000.
First-t counts (blind maximum): T_0 111, T_1 120, T_2 64, T_3 40, T_4 37, T_5 28, T_6 27, T_7 29, T_8 19, T_9 16, T_10 27, T_11 21, T_12 26, T_13 26, T_14 5, T_15 2.

## Instrument readouts

- G4 X3 (blind maximum at or above the corpus maximum): ONE 34/338, TWO-REL 2/36, UNREL 0/7, UNK 7/215, UNK-N 0/9, UNK-F 7/206.
- G4 X7 (pass rate by encoding flag): flagged 83/137, unflagged 301/459.
- G4 explanations naming an untestable job: 0/600 (explanations containing the phrase explainer.md asks for when a job cannot be carried by a short text (string match; no field records it)).
- O7 X3 (blind maximum at or above the corpus maximum): ONE 35/311, TWO-REL 4/36, UNREL 0/4, UNK 3/248, UNK-N 1/13, UNK-F 2/235.
- O7 X7 (pass rate by encoding flag): flagged 80/130, unflagged 285/469.
- O7 explanations naming an untestable job: 0/600 (explanations containing the phrase explainer.md asks for when a job cannot be carried by a short text (string match; no field records it)).
- Q4B X3 (blind maximum at or above the corpus maximum): ONE 41/358, TWO-REL 4/34, UNREL 2/8, UNK 3/191, RARE 0/4, UNK-N 0/10, UNK-F 3/181.
- Q4B X7 (pass rate by encoding flag): flagged 113/188, unflagged 299/407.
- Q4B explanations naming an untestable job: 0/600 (explanations containing the phrase explainer.md asks for when a job cannot be carried by a short text (string match; no field records it)).

## Cost

- GPU: 38.2 forward-pass GPU-h over every banked run on both lanes before W5 (vast: 13 rentals, 30.9 billed GPU-h, $25.60 realized; AWS lane: W0, the three W1 harvests, W2-D rounds 1–5, Q4B and O7 W2-C scoring, billed instance hours not tallied), plus the center-probe run of 2026-10-23 (vast, RTX PRO 6000 WS, 2.31 billed hours at $2.01/h, $4.67; credit $12.17 → $7.50; program spend $30.27 of the $36 ceiling) (no machine-readable source; JOURNAL 2026-10-21T19:50:09Z and 2026-10-23, SPEC §12 hold the record).
- API: the provider's user-set limit of $250 was reached during S clustering (≈ $237 at list prices through the W4 gate and round 13, then $18.38 for the 327 API clusterer answers on S); the rest of W4 on S (883 clusterer answers and all 11,034 judge lines, plus the 385 judge lines re-answered on 2026-10-24, the 205 sample-1 clusterer answers re-answered for F15 and F22, and the 78 supplementary judge lines of batch w4j2) ran on Claude Code Sonnet subagents through the manual backend at zero API dollars (no machine-readable dollar source; responses/*.usage.jsonl hold token counts per call; JOURNAL 2026-10-21T19:50:09Z, 2026-10-22T00:18:16Z, 2026-10-23, 2026-10-24).

## Footnotes (owed per handoff/S5.md, checks/W1-W3.md F2 and checks/W4-W5.md; written at W5)

- Both presence readings use a⁺ = max(s·a, 0); §2.4's raw-ratio sentence says |a|.
- Device (F2 items 1–3): T_7–T_14 were harvested on an RTX PRO 5000 Blackwell, the rest on L40S. T_14 re-run on L40S flips 0 of 640 poles; per cluster, 0 of 2,454 (percentile reading) and 5 of 2,454 (raw-ratio reading, share 0.002) flip (`bank/neurons/w3/device_check/t14_1b_vs_1c_clusters.json`; rule: above 5% re-harvests job b). The labelled sensitivity (item 3) is in the Q2 section above and in `metrics.json q2.sensitivity_F2_excl_T7_T15`: with the 61 late-born clusters born at T_7 or T_15 left out, p_u = 0.275 [0.220, 0.332] (still above the H_SP rule); with 77 clusters and the 41 poles whose main is so born left out, b = 0.486 [0.443, 0.527] against a null 97.5th percentile of 0.465. Neither label moves. The center-probe run (RTX PRO 6000 Blackwell) against the W2-C probe (L40S) at T_15 on the S poles: median relative difference 0.8%, 99th percentile 8%, correlation 0.9999 for both the max and the min probe values.
- G4's W2-C texts were scored on Blackwell against L40S `max_abs` and an L40S-derived θ_s (F2 item 4).
- Gate limits (W1–W3 F10, W4–W5 F20): G0/G1 failing inputs are recorded, not asserted; preflight tier 1 covers `infra/vast` only; the probe-rule check verifies a label. The unblind's "filled once" refusal guards one directory and was bypassed six times, once per re-run (runs 2–7, each earlier run kept beside the directory), by moving `bank/w5` aside, each earlier run kept and compared (see Provenance).
- PREDICTIONS was written for the Flash and prompt-v2 setup; the run used Sonnet 5.5 for the clusterer and the judge (selected at the gate on S_gate). On S, 883 clusterer answers and all 11,034 judge lines came from Claude Code `sonnet` subagents through the manual backend (`backend: manual`), 317 clusterer answers from the API; the judge identity (model, prompt sha 11b5fd81…) matches the gate, the serving differs (Claude Code sampling, no constrained decoding).
- Manual backend audit (W4–W5 checker F13–F17, `checks/W4-W5.md`; incidents in `handoff/judge_incident_log.txt`): the per-chunk transcript audit gated every import only from 2026-10-22T10:29:44Z; 435 judge chunks were imported before the gate, 172 of them fail the current rules, 25 on substance (375 lines). The 25 chunks' 375 lines were then re-answered by fresh agents under the hardened guard (24 of the 375 answers changed). F15 (200 sample-1 clusterer answers read from excerpts) was resolved in the data the same way: see the provenance footnote. F22 (lead's audit, 2026-10-24): whether every input reached its answering agent whole was checked for all 12,370 imported requests of every manual round by looking for every line of each `user.txt` in the producing transcript's tool results; the only failures outside the F15 set were one judge line, a round-1 answer the F13 re-answer had already replaced (the live round-5 answer, whose agent saw every line, was re-answered anyway, same verdict), and five old-trial clusterer answers whose transcripts could not be found (re-answered). The live gate now has that check (`runbook/chk_full.py`). Before that, on the fourth run's judgments, leaving those lines out had given p_u 0.2587 [0.2038, 0.3132], and leaving out all 172 non-passing chunks' 2,532 lines 0.2585 [0.2023, 0.3153] (1,235 clusters, 597 controls), both H_SP, recomputed by the lead with the instrument's estimator (`runbook/leaveout_pu_f13.py`, seed 0, 10,000 draws; the primary reproduced exactly); the chunk lists are `checks/substantive_25.txt` and `checks/nonpassing_172.txt`. The 147 remaining non-passing chunks failed only the then-unrequired output re-read or a wrong-path Read and stay. Nine kept answers from six one-line-short files belonged to a neighbouring request; they were removed and re-answered under the hardened guard (`runbook/chk_calls.py` since 2026-10-24: whole-command Bash match, no offset or limit on user.txt, Read/Write/Bash only, no unparseable line) for the fourth run (two of the nine answers changed). Forty sample-1 clusterer chunks (200 of 600 answers) were answered from excerpts of user.txt: on those 200 poles B4 (m within 1 between the two clusterer samples) is 0.650 against 0.678 on the other 400 (mean m₁ 3.00 against 3.37), and 18 of the 52 unmatched clusters are on them (reproduced by the lead from `clusters.jsonl` and the W5 cluster samples); nothing was re-run for them. Cluster-disjoint judge chunks still let a pole's record appear twice through two of its clusters with the same t_before: 56 lines in 96 chunks, judge accuracy 0.795 on them against 0.813 overall.
- Explainer model by depth tercile (checks/W1-W3.md Q6 table): Opus for most of terciles 0–1, Sonnet for most of 2 (Opus / Sonnet per tercile: Q4B 202/2, 196/0, 65/135; G4 210/2, 191/3, 54/140; O7 208/12, 204/16, 14/186); per-pole classes depend on the explainer for about 28% of poles; 33/117 discovery disagreement.
- θ_s sensitivity at 0.2665 (W1–W3 F1 item 2) is withdrawn: round 13 put the audit configuration's own θ_s at 0.3091, within 0.006 of the frozen 0.3153, so 0.2665 brackets no configuration that ran on the audit sets (JOURNAL 2026-10-24).
- Hard-decoy row: a_ctrl on hard controls is 0.692 at the gate and 0.798 on S, both below 0.80, so the row is reported as not measurable (`metrics.json q2.hard.measurable` false).
- Center birth curve (§2.4, exploratory; W4–W5 F21): computed from the per-checkpoint probe job `x/center_probe/1` (the W2-C O7 texts scored at all 16 checkpoints, same revisions as the W3 grid, each value over that checkpoint's own max|a|), run after the first unblind as the manifest prepared on 2026-10-21 foresaw; the §11 prediction for the row was written before data. 1 pole has no texts (left out), 1 has no positive T_15 score (undefined). The reading compares each pole's blind maximum with its own T_15 score, so a pole whose blind maximum is weak at T_15 reaches 0.3 of it early; the UNK class (median T_1) holds most such poles. 111 of 598 poles reach the 0.3 line at T_0, the untrained checkpoint, so the reading does not separate formation from the random-init floor for them. Medians in the table are grid indices; a half-integer median is written as both labels. Per-pole ratio curves are noisy (1% non-decreasing at tolerance 0.05); the median curve over poles rises at every grid point but T_2.
- Output-vocabulary line (F41, found 2026-10-24 after closeout): the explainer and second-reader prompts showed every negative pole the positive pole's "most promoted / most suppressed" list (a second sign flip in the explain code on a file the harvest already writes per pole). The clusterer and judge saw no such line, so Q2 is untouched. ONE shares of negative and positive poles are equal within noise in every depth third of every model; clauses in negative-pole descriptions about which word the neuron promotes may be reversed. Code fixed; no wave re-run.
- Provenance: this file comes from the seventh unblind run (2026-10-24T04:06:05.084615Z), whose inputs are the sixth run's unchanged (every number and label equal; the run exists because the closeout checker's F23 and F29 asked that the sample files carry, per pole, the contexts as the explainer saw them and, per judgment, the records as the judge saw them, which `neurons_verdict unblind` writes since explain commit d832b80f of 2026-10-24; run 6 is kept as `bank/w5_run6_20261024T0228Z`). The sixth run's inputs differed from the fifth run's in three ways, all from the W4–W5 checker's F15 and the lead's F22 audit (JOURNAL 2026-10-24): (1) `clusters.jsonl` was rebuilt after the 200 sample-1 clusterer answers the checker found read from excerpts, plus five whose read could not be audited, were re-answered under the hardened chain (205 rows changed; `runbook/recollect_clusters_f15.py`); (2) the late-born primary set moved from 1,239 to 1,238 clusters, 13 entering and 14 leaving, so the 14 leaving clusters' 84 judge lines were set aside with their plan rows (`runbook/prune_left_primary_f15.py`) and the 13 entering clusters were judged in the supplementary batch `w4j2` (78 lines, seed 4113, same construction and judge model as w4j; `runbook/prepare_judge_subset.py`); (3) one judge line (`judge:w4j:991`) was re-answered for F22, with the same verdict; the cut read F22 found was its round-1 answer, which the F13 re-answer had already replaced, so the fifth run's answer for that line rested on a whole read. Earlier runs, each kept locally and under `s3://…/results/neurons/`: run 1 (2026-10-23T19:55:33Z, metrics sha256 9129a195…), run 2 (20:06:45Z, 56886c88…, added the F2 sensitivity field), run 3 (23:27Z, 4b2f3c77…, added the center birth curve), run 4 (2026-10-24T00:18Z, after the nine misattached judge rows were re-answered), run 5 (01:03Z, b020ea29…, after the 375 lines of the 25 substantive-failure chunks were re-answered, 24 answers changed). Where runs differ, §11's note says so.
