# N13 claims record

One line per load-bearing claim (spec §10.2). Started 2026-11-08 by the twelfth N13 executor, after the verdict ran once.
Status tags: **VERDICT** = decided by the frozen rules of `n13-freeze-2` and the code of `n13-freeze-2a`, read from
`studies/next/n13/verdict_n13.json` (sha256 `cae55c04ebf8…`); **BANKED** = read from the S4 arm records; **EXPLORATORY** = §6.3 last
paragraph. Every VERDICT line was recounted by the independent reader on 2026-11-08 (`READ_n13.md`: `RECOUNT_OK`, 465 of 465 verdict fields equal, every answer re-graded to its banked score) and read by the head (`READ_head.md`, claims N13-1 to N13-9). Accuracy is MATH-500 percent
correct, n = 500, one sample per problem, seed set A unless named; differences are percentage points, first arm minus second; p values
are McNemar, BH-adjusted over the family of 14 (§7). Before: `allenai/Olmo-3-1025-7B` `a81bae42`; after:
`allenai/Olmo-3.1-7B-RL-Zero-Math` `364820b0`.

| ID | claim | artifact (field) | status |
|---|---|---|---|
| N13-C1 | The verdict is COMPLETE: all 13 gates pass and every gate's failing input fails; no stamp; no arm parked; all 14 family comparisons available. | `verdict_n13.json` (`verdict`, `gates`, `stamps`, `parked`, `comparisons._family_size_available`) | VERDICT |
| N13-C2 | Endpoints: base 57.2 [52.8, 61.5], rl 92.0 [89.3, 94.1], zero (seed B) 55.4, id (seed B) 91.2. G = 34.8 points against the gate 5√2σ = 13.60 (σ 1.924); δ 5.44; H 74.6; Q 65.9. | `arms.base/rl/zero/id`, `thresholds`, `three_numbers` | VERDICT |
| N13-C3 | Three numbers: G-KNOWN 39.96 (within 1.6 of 40.0; failing input 1.12 fails); \|L − L'\| 0.8 ≤ 8.16; B − B' 1.8; S-BAND L' 91.2 inside [81.5, 96.7]; PC-RANK recovery 1.0; id hash equals rl. | `three_numbers`, `gates` | VERDICT |
| N13-C4 | No top-k transplant recovers the gain: top_k is NONE at every k (57.6, 61.4, 61.8, 63.8 for k = 0.1, 0.3, 1, 3 %; all at or under Q 65.9); so k_rec is NONE. | `arms.top_k*.acc/label`, `hypotheses.k_rec` | VERDICT |
| N13-C5 | **H_local: REFUTED (k_rec NONE).** At every k the energy-matched random arms are within δ of top_k: top − rndE −1.2, +1.6, +2.6, 0.0 and top − rndT −1.8, +3.4, +1.8, +1.0 (k 0.1, 0.3, 1, 3); smallest BH p 0.107. | `hypotheses.H_local`, `comparisons.top_k*\|rndE_k*`, `top_k*\|rndT_k*` | VERDICT |
| N13-C6 | **H_diffuse: SUPPORTED (smallest k 0.1 %).** dare_k is REC at every k (85.2, 86.8, 85.8, 86.2) and beats top_k: top − dare −27.6, −25.4, −24.0, −22.4; BH p ≤ 2.6e-21. | `hypotheses.H_diffuse`, `arms.dare_k*`, `comparisons.top_k*\|dare_k*` | VERDICT |
| N13-C7 | **H_attention: REFUTED (BOTH).** mlp (all MLP from RL, non-MLP before) 86.4 REC; attn (attention and norms from RL) 83.0 REC; mlp − attn +3.4, BH p 0.107. | `hypotheses.H_attention`, `arms.mlp/attn`, `comparisons.mlp\|attn` | VERDICT |
| N13-C8 | **Necessity: NOT NECESSARY at all four k.** comp_k (everything except S_k, non-MLP from RL) 90.4, 90.8, 91.6, 90.6, each within δ of rl 92.0; top_1 − comp_1 −29.8, BH p 2.2e-37. | `hypotheses.necessity`, `arms.comp_k*`, `comparisons.top_k1\|comp_k1` | VERDICT |
| N13-C9 | **H_artifact: INDETERMINATE.** J_floor(1 %) 0.642 and J_round(1 %) 0.534, between the rule's 0.4 and 0.7; PC-ART valid (rule(P+) SUPPORTED with J_round 0.844; rule(P−) REFUTED). | `hypotheses.H_artifact`, `H_artifact_values` | VERDICT |
| N13-C10 | **GSM8K at 1 %** (G-GATE-GSM 28.7, pass): top − dare −23.7 [−26.5, −21.1], top − rndE +2.1 [+0.2, +3.9], top − rndT +1.9 [+0.1, +3.6]; all three labeled REPLICATES by the frozen rule. On MATH-500 the rndE and rndT differences at 1 % are not significant (BH p 0.364, 0.530). | `gsm8k`, `comparisons` | VERDICT |
| N13-C11 | Spread arms agree with their first draws: top_1 seed B 60.0 (seed A 61.8), rndT_1 second draw 59.6 (60.0), dare_1 second draw 87.8 (85.8), rnd0_1 (count-matched, natural scale) 59.6; topF_1 (above-floor score) 64.0, NONE. | `arms.top_k1_B/rndT_k1_d2/dare_k1_d2/rnd0_k1/topF_k1` | VERDICT |
| N13-C12 | Every S4 arm was built as registered: 355 tensors equal R1 stage 6 for every arm but rndE (96 MLP tensors changed by A30's stochastic rounding, as expected); A31 passes on all four rndE arms (relative differences 1.2e-16 to 1.0e-15, 0 tensors differ); rndT energies 1.0000 to 1.0011 of E(S_k), none short; every engine block equals S3's; every G-COMPLETE passes with its failing input failing; no screen fired. | `studies/next/n13/s4/*/*/arm_record.json`, `engine_check.json`; `s4/gates_table.txt` | BANKED |
| N13-C13 | Answer format follows accuracy: the 16 non-endpoint arms labeled NONE write an `Answer:` line in 0.41 to 0.46 of answers (base 0.44, zero 0.39); the 11 labeled REC in 0.00 to 0.04 (rl and id 0.00). Mean output tokens 3,133 to 4,004 for the former, 4,428 to 4,671 for the latter. | `arms.*.answer_line_share`, `mean_tokens` | EXPLORATORY |

## Open for the independent reader (not claims)

- dare_k rescales each kept entry by 1/q_k (33 to 1,000; spec §3.2), so its change energy is not matched to top_k's; rndE is the
  energy-matched control. Whether N13-C6 measures "spread" or "scale" is not decided by this verdict.
- With the same seeds, many answers are token-identical across arms near base (JOURNAL.md 2026-11-07 20:40Z), so many McNemar pairs
  are concordant.
