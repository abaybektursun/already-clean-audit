# Check of every number in 20_ABSTRACT.md (draft 3) against its artifacts

2026-11-08. Reader: a fresh Opus 5.5 agent, worktree `interp_research-abscheck`, branch `abscheck` (from `clean-neurons` at
`c25b51156`, the draft-3 commit). CPU only, no model run. Script: `studies/next/abscheck/check_abstract.py`, output
`studies/next/abscheck/check_abstract_output.txt` (202 lines; its comparator and its banned-word scan each fire on a planted input).

**Answer: 175 numbers checked; 158 equal, 10 fairly rounded, 4 mismatched, 3 not found (all three are design values of the
next step, with no artifact).** The four mismatches:

1. Abstract (1), "about one top-50 activation slot in fifty". The artifact gives 0.56 to 0.87 slots of 50 per pole by band
   (G − start −0.0113, −0.0170, −0.0173, −0.0161; mean 0.77 slots, one slot in 65). No band reaches one slot.
2. Abstract (3), "orders a unit's poles as zero ablation does in 65 of 71 units". 65 of 71 is the count of units whose top pole
   matches. The four poles are in the same signed order in 60 of 71 units (absolute order: 50 of 71).
3. Evidence table, N13 energy-matched row, "top − control −1.2 to +3.4". The low end is −1.8 (top_k0.1 − rndT_k0.1).
4. Next step, "DARE's 0.1 % sample … 89.8 of 92.3 on GSM8K". GSM8K was run on DARE at 1 % only (dare_k1, 89.765). The 0.1 %
   arm has no GSM8K reading. The same attribution is in `n13/READ_head.md` (practical dimension).

## Conventions

- EQUAL: the artifact value, rounded to the digits printed, gives the printed number. A range whose two ends are such copies is
  EQUAL.
- ROUNDED fairly: the printed number is derived from artifact values (a share from a count, a percentage from a ratio, a bound, a
  min–max over many values), or the abstract prints fewer digits than the evidence table; the derivation checks.
- MISMATCH: the printed number or the quantity it names differs from the artifact. NOT FOUND: no artifact holds it.
- Counting unit: one estimate with its interval is one number; a list of four band values is four numbers. Grouped rows below
  share one status for every value in them.
- Paths are under `studies/next/` unless they start with `infra/`.

## 1. The abstract paragraph (27 numbers: 20 EQUAL, 5 ROUNDED, 2 MISMATCH)

| # | text as written | artifact and field | value found | status |
|---|---|---|---|---|
| A1 | OLMo-3 7B | `m1/READ_head.md`; `n22/JOURNAL.md` 18:01Z | OLMo-3 7B | EQUAL |
| A2 | 100B tokens (G) | `m1/READ_head.md` table, row G; `m1/EXPLORATORY_turnover.md` | "G, 100B tokens with 0 % code+math" | EQUAL |
| A3 | about one top-50 activation slot in fifty | `m1/verdict_m1.json` exploratory.shift_from_start.G[b][0] | −0.01129, −0.01703, −0.01732, −0.01611: 0.56, 0.85, 0.87, 0.81 slots of 50; one slot in 58 to 89 | MISMATCH: 0.56–0.87 slots of 50 (1.1–1.7 % of top-50 slots; mean 0.77, one in 65) |
| A4 | a 65 % code-and-math mix | `m1/READ_head.md` table, row M | 65 % | EQUAL |
| A5 | a 40 % mix | `m1/READ_head.md` table, row R | 40 % | EQUAL |
| A6 | 66–76 % of the way between | λ = (C_R − C_G)/(C_M − C_G) from `verdict_m1.json` exploratory.C; `m1/READ_m1.md` | 0.6587, 0.6826, 0.7043, 0.7634 | ROUNDED fairly |
| A7 | named one checkpoint early | `m2/READ_head.md` (score B at T_(k−1)) | pre role, one checkpoint before birth | EQUAL |
| A8 | top 10 | `m2/READ_head.md` "a hit is a rank in the top 10" | 10 | EQUAL |
| A9 | 22,016 poles | `m2/READ_head.md`; `n12/READ_n12.md` (442 units × 22,016 poles) | 22,016 | EQUAL |
| A10 | 59 % of held-out births | `m2/verdict_m2.json` labels.H_B (test half) | 42 of 71 = 59.15 % | ROUNDED fairly |
| A11 | chance 0.05 % | 10 / 22,016; `m2/READ_head.md` "chance 0.00045" | 0.0454 % | ROUNDED fairly |
| A12 | names its pole more often (76 %) | `m2/verdict_m2.json` test.pb_pre_B | 53 of 70 = 75.71 % | ROUNDED fairly |
| A13 | orders a unit's poles as zero ablation does in 65 of 71 units | `n2v/read_n2v/r6_checks.json` a_within_unit.test.window | top_abs_pole_match 0.9155 = 65 of 71; kendall_signed.share_tau_1 0.8451 = 60 of 71; kendall_abs.share_tau_1 0.7042 = 50 of 71 | MISMATCH: same order in 60 of 71; 65 of 71 is "picks the same top pole" |
| A14 | Spearman 0.960 [0.915, 0.990] | `n2v/verdict_n2v.json` test.window.validation | 0.95996 [0.91547, 0.99022] | EQUAL |
| A15 | −0.43 [−0.82, −0.06] | `n1/verdict_n1.json` test.pair_in_real_dt | −0.43169 [−0.81670, −0.05897] | ROUNDED fairly |
| A16 | 67 pairs | same, n | 67 | EQUAL |
| A17 | a two-birth margin | `n1/read_n1/fragility.json` test.drop_one_in, drop_extremes_in | no single drop flips; dropping the two most negative gives upper bound +0.041 (AT) | EQUAL |
| A18 | OLMo-3.1-7B-RL-Zero-Math | `n13/CLAIMS.md`; `n13/s3/s3gate-02/setup/setup.json` | `allenai/Olmo-3.1-7B-RL-Zero-Math` 364820b0 | EQUAL (the Hugging Face id writes "Olmo") |
| A19 | MATH-500 92.0 | `n13/verdict_n13.json` arms.rl.acc | 92.0 | EQUAL |
| A20 | against 57.2 | arms.base.acc | 57.2 | EQUAL |
| A21 | the top 0.1–3 % | arms.top_k0.1 … top_k3 | k 0.1, 0.3, 1, 3 % | EQUAL |
| A22 | read 57.6–63.8 | arms.top_k*.acc | 57.6, 61.4, 61.8, 63.8 | EQUAL |
| A23 | as do random neurons matched on change energy | arms.rndE_k*.acc (rndT_k*) | 58.8–63.8 (58.0–62.8) | EQUAL |
| A24 | a random 0.1 % of MLP entries … reads 85.2 | arms.dare_k0.1.acc | 85.2 | EQUAL |
| A25 | all MLP alone 86.4 | arms.mlp.acc | 86.4 | EQUAL |
| A26 | attention with norms alone 83.0 | arms.attn.acc; `n13/CLAIMS.md` N13-C7 "attention and norms from RL" | 83.0 | EQUAL |
| A27 | the RL model without its top set 91.6 | arms.comp_k1.acc | 91.6 (k 1 %; comp_k0.1, k0.3, k3 read 90.4, 90.8, 90.6) | EQUAL (k not named; the other three k read lower) |

## 2. The evidence table (134 numbers: 131 EQUAL, 2 ROUNDED, 1 MISMATCH)

Every artifact path the table names exists in the worktree. Two N1 numbers are not in the files their row names (marked "cited
file does not hold it"); they are equal where they are found.

| row (claim) | text as written | artifact and field | value found | status |
|---|---|---|---|---|
| M1 G − start | −0.0113, −0.0170, −0.0173, −0.0161 | `m1/verdict_m1.json` exploratory.shift_from_start.G[b][0] | −0.011290, −0.017028, −0.017323, −0.016109 | EQUAL ×4 |
| M1 M − start | −0.0004, −0.0070, +0.0001, +0.0035 | same, .M | −0.000389, −0.007039, +0.000068, +0.003510 | EQUAL ×4 |
| M1 M − G | +0.0109, +0.0100, +0.0174, +0.0196 | `verdict_m1.json` lead[b].mean | 0.010901, 0.009989, 0.017392, 0.019619 | EQUAL ×4 |
| M1 drift bar | 0.0082, 0.0066, 0.0023, 0.0035 | `verdict_m1.json` bar | 0.008199, 0.006556, 0.002253, 0.003538 | EQUAL ×4 |
| M1 λ | 0.70, 0.68, 0.66, 0.76 | `m1/READ_m1.md` line 163; recomputed from exploratory.C | 0.7043, 0.6826, 0.6587, 0.7634 | EQUAL ×4 |
| M1 identity | 0 in every band | `verdict_m1.json` G_ID.I | [0, 0, 0, 0] | EQUAL |
| M1 identity | counts.npz sha256 8c332d7d… twice | `m1/banks/start_a/m1/counts.npz`, `start_b/…` (shasum run here) | 8c332d7d8f260efd… for both | EQUAL |
| M1 turnover | arrivals under G 0.085–0.165 | `m1/EXPLORATORY_turnover.md` Answer | 0.165, 0.165, 0.085, 0.090 | EQUAL |
| M1 turnover | the lowest of nine pairs | same | "the lowest of the nine pairs that span training" | EQUAL |
| M2 top 10 | 42 of 71 | `m2/verdict_m2.json` labels.H_B hits, n | 42, 71 | EQUAL |
| M2 top 10 | 0.592 [0.475, 0.698] | same, p, lo, hi | 0.59155 [0.47539, 0.69831] | EQUAL |
| M2 top 10 | bar 0.0122 | same, bar | 0.012195 | EQUAL |
| M2 twins | 53 of 70 | test.pb_pre_B[3], test.n_pb | 53, 70 | EQUAL |
| M2 twins | 0.757 | test.pb_pre_B[0] | 0.75714 | EQUAL |
| M2 twins | −0.171 [−0.286, −0.057] | labels.H_sel_B | −0.17143 [−0.28571, −0.05714] | EQUAL |
| M2 repeated tokens | varied −0.014 [−0.113, +0.085] | `m2/read/r5b_token_share.json` both.real_share_ge_0.67 | −0.01408 [−0.11268, +0.08451] (both halves pooled, n 71) | EQUAL |
| M2 repeated tokens | repeated −0.397 [−0.513, −0.282] | same, both.real_share_lt_0.67 | −0.39744 [−0.51282, −0.28205] (both halves pooled, n 78) | EQUAL |
| N12 own rise | +0.951 [0.738, 1.180] | `n12/verdict_n12.json` labels.H_cf (OWN_RISE) | 0.95071 [0.73752, 1.18009] | EQUAL |
| N12 own rise | twins +0.77 | test.F_pb.mean | 0.76510 | EQUAL |
| N12 own rise | twins +0.94 | test.F_dt.mean | 0.94219 | EQUAL |
| N12 leaders | 0.939 | test.drop_real.mean | 0.93860 | EQUAL |
| N12 leaders | 1.164 | test.drop_pb.mean | 1.16387 | EQUAL |
| N12 leaders | −0.226 [−0.408, −0.056] | labels.H_away_sel_pb | −0.22565 [−0.40774, −0.05613] | EQUAL |
| N12 dt twins | real 42 of 71 | test.hits.real.pre[0] × 71 | 42 | EQUAL |
| N12 dt twins | dt 41 of 67 | test.hits.dt.pre[0] × 67 | 41 | EQUAL |
| N12 dt twins | pb 53 of 70 | test.hits.pb.pre[0] × 70 | 53 | EQUAL |
| N12 dt twins | real − dt −0.030 [−0.149, +0.090] | labels.H_sel_dt | −0.02985 [−0.14925, +0.08955] | EQUAL |
| N12 G-VAL bf16 | Spearman 0.100 freeze | labels.H_signal.reason; `n12/READ_n12.md` §6 | 0.0996 | EQUAL |
| N12 G-VAL bf16 | 0.043 test | same; test.validation.rho | 0.04254 | EQUAL |
| N12 G-VAL bf16 | ΔL 27 times the prediction at the median | `n12/READ_n12.md` §6 table | 27.3 (test; freeze 31.5) | EQUAL |
| N2V window | 0.960 [0.915, 0.990] | `n2v/verdict_n2v.json` test.window.validation | 0.95996 [0.91547, 0.99022] | EQUAL |
| N2V window | shuffle 97.5th 0.118 | same, shuffle_q975 | 0.11762 | EQUAL |
| N2V window | 284 pairs | same, n_pairs | 284 | EQUAL |
| N2V window | 71 units | same, n_units | 71 | EQUAL |
| N2V window | freeze 0.971 [0.939, 0.993] | `n2v/thresholds.json` freeze.window.validation | 0.97059 [0.93903, 0.99302] | EQUAL |
| N2V other readouts | 0.987 [0.971, 0.995] | test.fd.validation | 0.98747 [0.97117, 0.99475] | EQUAL |
| N2V other readouts | 0.986 [0.969, 0.996] | test.own.validation | 0.98636 [0.96928, 0.99640] | EQUAL |
| N2V other readouts | 0.967 [0.923, 0.994] | test.window_jvp.validation | 0.96661 [0.92319, 0.99415] | EQUAL |
| N2V top pole | 65 of 71 | `n2v/read_n2v/r6_checks.json` a_within_unit.test.window.top_abs_pole_match | 0.91549 × 71 = 65 | EQUAL |
| N2V top pole | 0.915 [0.845, 0.972] | same | 0.91549 [0.84507, 0.97183] | EQUAL |
| N2V top pole | chance level 0.25 | same, chance | 0.25 | EQUAL |
| N2V top pole | within-unit permutation ρ 0.286 | same window, within_unit_perm.rho_pairs_median | 0.28590 | EQUAL |
| N2V first order | ΔL / prediction 0.94–1.06 by decile | r6 c_linearity dL_over_pred_bf16_by_abs_pred_decile medians | test 0.936–1.041; freeze 0.948–1.056 | ROUNDED fairly (the range spans both halves) |
| N2V first order | fd slope 0.988 | c_linearity.test.slope_fd_over_0.1 | 0.98842 | EQUAL |
| N2V first order | fd slope 1.008 | c_linearity.freeze.slope_fd_over_0.1 | 1.00760 | EQUAL |
| N2V first order | second-order share 0.18–0.20 | c_linearity.*.second_order_share_median_abs_b_over_abs_dL | 0.17538 (test), 0.20311 (freeze) | EQUAL |
| N12 labels after N2V | H_signal 0.527 [0.451, 0.600] | `verdict_n2v.json` n12_reread.labels.H_signal; `r5_reread.json` | 0.52676 [0.45141, 0.60000] | EQUAL |
| N12 labels after N2V | against 0.4225 | same, bar | 0.42252 | EQUAL |
| N12 labels after N2V | sel pb −0.039 [−0.134, 0.057] | n12_reread.labels.H_sel_signal_pb | −0.03857 [−0.13357, 0.05714] | EQUAL |
| N12 labels after N2V | sel dt −0.020 [−0.124, 0.084] | n12_reread.labels.H_sel_signal_dt | −0.02015 [−0.12388, 0.08360] | EQUAL |
| N12 labels after N2V | H_bridge −0.013 [−0.072, 0.046] | n12_reread.labels.H_bridge | −0.01259 [−0.07180, 0.04590] | EQUAL |
| N1 input rows | −0.432 [−0.817, −0.059] | `n1/verdict_n1.json` test.pair_in_real_dt | −0.43169 [−0.81670, −0.05897] | EQUAL |
| N1 input rows | 67 pairs | same, n | 67 | EQUAL |
| N1 input rows | 23 above | same, x_only | 23 | EQUAL |
| N1 input rows | 44 below | same, y_only | 44 | EQUAL |
| N1 input rows | margin two births | `n1/read_n1/fragility.json` test.drop_one_in, drop_extremes_in.drop_2_most_negative | 0 flips on one drop; upper bound +0.0412 on two | EQUAL |
| N1 input rows | freeze half −0.217 [−0.638, 0.204] | cited file does not hold it; `n1/thresholds.json` discovery.pair_in_real_dt | −0.21660 [−0.63788, 0.20417] | EQUAL |
| N1 input rows | pooled −0.320 [−0.604, −0.035] | cited file does not hold it; `n1/read_n1/verdict_recount.json` confounds.c_halves.in.pooled_both_halves | −0.31953 [−0.60403, −0.03535] | EQUAL |
| N1 output column | −0.258 [−0.654, 0.130] | `verdict_n1.json` test.pair_out_real_dt | −0.25812 [−0.65450, 0.13015] | EQUAL |
| N1 floor | in −0.297 [−0.550, −0.059] | labels.floor_in | −0.29702 [−0.54980, −0.05872] | EQUAL |
| N1 floor | against −0.053 | labels.floor_in.bar; `n1/thresholds.json` floor_in_freeze.q975 | −0.05340 | EQUAL |
| N1 floor | out −0.039 [−0.276, 0.202] | labels.floor_out | −0.03891 [−0.27622, 0.20229] | EQUAL |
| N1 floor | against 0.147 | labels.floor_out.bar | 0.14679 | EQUAL |
| N1 natural units | −1.6 % [−3.2 %, +0.03 %] | `n1/read_n1/scale.json` test_pair_mean_log_r_window_in.as_relative | −0.01567 [−0.03216, +0.00035] | EQUAL |
| J8 | S − W +0.0079 | `pilot0/GPU_CHECK2.md` table, step 61,001 | +0.0078880 | EQUAL |
| J8 | S − W +0.0094 | same, step 100,001 | +0.0093759 | EQUAL |
| J8 | ≤ 0.0124 | same, the frozen band | 0.0124 | EQUAL |
| J8 | neighbours farther | same | \|N − W\| 0.0823 (61,001); 0.0483 (100,001, rerun b2) | EQUAL |
| J10 | 7 answered | `n22/verdict_n22_t15.json` n_answered; `n22/banks/l1-01/n22/metrics.json` points.15.answered | freeze 2 + test 5 = 7 | EQUAL |
| J10 | of 1,200 items | `n22/items.json`; `n22/JOURNAL.md` 17:53Z | 1,200 | EQUAL |
| J10 | K0 exact | `n22/banks/l1-01/n22/metrics.json` gates.t15_K0 | diff 0.0 (2.675372037760417 both) | EQUAL |
| N13 known number | 39.96 | `n13/r0/s2setup-07/r0_known/metrics.json` G-KNOWN.value | 39.96393 | EQUAL |
| N13 known number | 40.0 ± 1.6 | same, target, tol | 40.0, 1.6 | EQUAL |
| N13 gate | base 57.2, rl 92.0, zero 55.4, id 91.2 | `n13/thresholds.json` B, L, B_prime, L_prime | 57.2, 92.0, 55.4, 91.2 | EQUAL ×4 |
| N13 gate | G 34.8 against 13.60 | thresholds G, room.bound_2.5delta | 34.8, 13.6015 | EQUAL ×2 |
| N13 gate | σ 1.924 | thresholds sigma | 1.92354 | EQUAL |
| N13 top-k | n 500 | thresholds n; arms.*.n | 500 | EQUAL |
| N13 top-k | 57.6, 61.4, 61.8, 63.8, all NONE | `verdict_n13.json` arms.top_k*.acc, label | 57.6, 61.4, 61.8, 63.8; NONE ×4 | EQUAL ×4 |
| N13 top-k | H 74.6 | thresholds H | 74.6 | EQUAL |
| N13 energy-matched | rndE 58.8, 59.8, 59.2, 63.8 | arms.rndE_k0.1, k0.3, k1, k3 | 58.8, 59.8, 59.2, 63.8 | EQUAL ×4 |
| N13 energy-matched | rndT 59.4, 58.0, 60.0, 62.8 | arms.rndT_k* | 59.4, 58.0, 60.0, 62.8 | EQUAL ×4 |
| N13 energy-matched | top − control −1.2 to +3.4 | comparisons top_k*\|rndE_k*, top_k*\|rndT_k* | rndE −1.2, +1.6, +2.6, 0.0; rndT −1.8, +3.4, +1.8, +1.0 | MISMATCH: −1.8 to +3.4 (`n13/CLAIMS.md` N13-C5 has −1.8) |
| N13 energy-matched | smallest BH p 0.107 | same eight comparisons, bh_p | 0.10658 (top_k0.3\|rndT_k0.3) | EQUAL |
| N13 DARE | 85.2, 86.8, 85.8, 86.2 | arms.dare_k* | 85.2, 86.8, 85.8, 86.2 | EQUAL ×4 |
| N13 DARE | top − dare −27.6 to −22.4 | comparisons top_k*\|dare_k* | −27.6, −25.4, −24.0, −22.4 | EQUAL |
| N13 DARE | BH p ≤ 2.6e-21 | same, bh_p | largest 2.537e-21 | ROUNDED fairly (a true bound; 2.5e-21 is nearer) |
| N13 DARE | dare − mlp within 1.2 | `read_n13/checks_read.json` j_other.dare_vs_mlp | −1.2, +0.4, −0.6, −0.2 | EQUAL (the second draw dare_k1_d2 is +1.4 from mlp) |
| N13 families | mlp 86.4, attn 83.0 | arms.mlp, arms.attn (REC, REC) | 86.4, 83.0 | EQUAL ×2 |
| N13 families | +3.4, BH p 0.107 | comparisons.mlp\|attn | +3.4, 0.10658 | EQUAL ×2 |
| N13 families | comp 90.4, 90.8, 91.6, 90.6 | arms.comp_k* | 90.4, 90.8, 91.6, 90.6 | EQUAL ×4 |
| N13 GSM8K | n 1,319 | thresholds gsm8k.n | 1319 | EQUAL |
| N13 GSM8K | top − dare −23.7 [−26.5, −21.1] | `verdict_n13.json` gsm8k | −23.730 [−26.535, −21.077] | EQUAL |
| N13 GSM8K | top − rndE +2.1 [+0.2, +3.9] | same | +2.123 [+0.227, +3.942] | EQUAL |
| N13 GSM8K | top − rndT +1.9 [+0.1, +3.6] | same | +1.895 [+0.076, +3.639] | EQUAL |
| N13 hypotheses | J 0.642, 0.534 | hypotheses.H_artifact_values J_floor_k1, J_round_k1 | 0.64204, 0.53415 | EQUAL ×2 |
| N13 cost | 102.555 GPU-h | `infra/aws_pg/operations/2026-11-06_next_n13_s4.md` last line | 102.555 | EQUAL |
| N13 cost | S3 and S4 $230.32 | same | $230.32 ($42.51 + $187.81) | EQUAL |
| N6 | (pending) | none | none | not counted |

### The "read by" column

READ files that exist: `m1/READ_m1.md`, `m1/READ_head.md`; `m2/READ_m2.md`, `m2/READ_head.md`; `n12/READ_n12.md`,
`n12/READ_head.md`; `n2v/READ_n2v.md`, `n2v/READ_head.md`; `n1/READ_n1.md`, `n1/READ_head.md`; `n13/READ_n13.md`,
`n13/READ_head.md`. Every "reader" and "head" entry for M1, M2, N12, N2V, N1 and N13 has its file.

- M1 turnover (head only): the head's read of it is `m1/READ_head.md` line 76 and the program `JOURNAL.md` 2026-11-05 18:25.
- J8 (executor, head): `pilot0/` has no READ file. The head's verdict is the paragraph at `pilot0/GPU_CHECK2.md` line 8.
- J10 (executor, head): `n22/` has no READ file. The head's decision is the program `JOURNAL.md` entry 2026-11-06 12:05 CST.
- The reader claims check: "bitwise recount of 612 pairs" (`n2v/READ_head.md`: all 612 pairs bitwise); "recount to 2.5e-14"
  (`n1/READ_head.md`); "every answer re-graded" (`n13/READ_head.md`: 30,009 answers, 0 disagreements).

### Cited-field defects (values equal, location wrong or incomplete)

- N1 freeze half and pooled values: not in `verdict_n1.json` or `fragility.json`; they are in `n1/thresholds.json` discovery
  and `n1/read_n1/verdict_recount.json` confounds.c_halves.
- J10 NO-GO: the label is not a field of `verdict_n22_t15.json` (that file holds readout NOT_PRESENT and slot NOT_RUN). It is in
  `n22/JOURNAL.md` 18:00Z and the program `JOURNAL.md`.
- N13 "all in cap-hit problems": not in the verdict arms fields. It is `n13/READ_n13.md` §(b) "Pairwise drop" (mlp − attn 0.0 on
  423 uncapped problems) and `n13/READ_head.md` item 5.
- M2 r5b values are pooled over both halves (n 149). The row does not say so.

## 3. The highest-leverage next step (14 numbers: 7 EQUAL, 3 ROUNDED, 1 MISMATCH, 3 NOT FOUND)

| # | text as written | artifact and field | value found | status |
|---|---|---|---|---|
| S1 | no 0.1–3 % set of whole neurons | arms.top_k*, rndE_k*, rndT_k* | k 0.1, 0.3, 1, 3 % | EQUAL |
| S2 | DARE 33–1,000 times the change's energy | `read_n13/checks_read.json` d_matching.k*.dare.over_full | 1000.1, 333.2, 99.8, 33.1 | ROUNDED fairly |
| S3 | r = 16 | none | a design value | NOT FOUND (design) |
| S4 | r = 64 | `n13/READ_head.md` Reading ("the rank-r arm at r = 64") | 64 | EQUAL |
| S5 | r = 256 | none | a design value | NOT FOUND (design) |
| S6 | the rank-64 arm within δ | `n13/thresholds.json` delta; `n13/READ_head.md` | δ 5.44 | EQUAL |
| S7 | about 2.1–2.6 GPU-h per arm on one L40S | S4 arm records `n13/s4/*/*/arm_record.json` wall_min (`s4/gates_table.txt`); operations note | MATH-500-only step wall 2.00–2.62 h (n 20, median 2.33); with GSM8K 3.43–4.83 h (n 7) | ROUNDED fairly, for arms read on MATH-500 only |
| S8 | eight arms | none | the section names six (three ranks, each against a random-subspace control) | NOT FOUND |
| S9 | 17–21 GPU-h | 8 × 2.1 = 16.8; 8 × 2.6 = 20.8 | 16.8–20.8 | ROUNDED fairly, under S7's and S8's premises |
| S10 | DARE's 0.1 % sample | arms.dare_k0.1 | 0.1 % | EQUAL |
| S11 | 85.2 … on MATH-500 | arms.dare_k0.1.acc | 85.2 | EQUAL |
| S12 | of 92.0 | arms.rl.acc | 92.0 | EQUAL |
| S13 | 89.8 … on GSM8K (for the 0.1 % sample) | `read_n13/checks_read.json` i_gsm8k | GSM8K arms: base, rl, zero, id, top_k1, mlp, dare_k1, comp_k1, rndE_k1, attn, rndT_k1; dare_k1 89.765 | MISMATCH: 89.8 is DARE at 1 %; the 0.1 % arm has no GSM8K reading |
| S14 | of 92.3 | i_gsm8k.rl.acc | 92.343 | EQUAL |

### The per-arm rate, recomputed from the banked S4 arm wall times

- Step wall, MATH-500 only: 20 arms, 119.7 to 156.9 min (2.00 to 2.62 h), median 2.33 h. Arms that recover the gain (dare,
  comp) write longer answers and run 2.40 to 2.62 h; base-level arms run 2.00 to 2.62 h.
- Step wall, MATH-500 plus GSM8K: 7 arms, 205.9 to 289.6 min (3.43 to 4.83 h).
- Launch overhead beyond the step walls: 0.16 to 0.18 h per completed launch, about 0.03 h per arm over 5 to 7 arms.
- Launch to STATUS per arm: B2-02 (5 MATH-500-only arms) 2.24 h; A1, A2, B1-02 (7 arms each, 2 with GSM8K) 2.95, 3.07, 2.74 h.
  All of S4: 76.511 GPU-h over 27 banked arms, 2.83 h per arm.

The estimate holds for arms read on MATH-500 alone: 2.0–2.65 h each with overhead, so eight arms take about 16–21 GPU-h. It does
not hold if "with GSM8K held out" means GSM8K is read on every arm: then each arm takes 3.4–4.8 h and eight arms 27–39 GPU-h. The
cost of building a rank-r truncation (one SVD per MLP matrix) has no measured counterpart in N13.

## 4. Labels, word count, banned words

Labels. The abstract paragraph names no label word. Each label it implies, and each label the table prints, equals its verdict:

| claim | label in the text | verdict field | label found |
|---|---|---|---|
| abstract (2), table M2 | carrier named before birth; twins name theirs more often | `m2/verdict_m2.json` labels.H_B, H_sel_B | SUPPORTED; BELOW_SELECTION |
| abstract (2), table N12 | distinct-token twins close the gap | `n12/verdict_n12.json` labels.H_sel_dt, H_dt_pb, H_dt | AT_SELECTION; LOWER; SUPPORTED |
| abstract (3), table N12 | wins by its own rise | labels.H_cf | OWN_RISE |
| abstract (3), table N12 after N2V | REFUTED, AT, AT, UNDECIDED | `n2v/verdict_n2v.json` n12_reread.labels | H_signal REFUTED; H_sel_signal_pb AT_SELECTION; H_sel_signal_dt AT_SELECTION; H_bridge UNDECIDED |
| table N2V | instrument passes | labels_test, labels_freeze | PASS ×4 on both halves; decision branch A |
| abstract (4), table N1 | BELOW_SELECTION (input), AT_SELECTION (output), neither above floor | `n1/verdict_n1.json` labels; hypothesis | sel_in BELOW_SELECTION; sel_out AT_SELECTION; floor_in, floor_out NOT_ABOVE_FLOOR; NEITHER |
| abstract (5), table N13 | top sets NONE; H_local REFUTED (k_rec NONE) | `n13/verdict_n13.json` arms.top_k*.label; hypotheses | NONE ×4; "REFUTED (k_rec NONE)" |
| abstract (5), table N13 | the gain is diffuse; H_diffuse SUPPORTED (0.1 %) | hypotheses.H_diffuse | "SUPPORTED (smallest k k0.1)" |
| table N13 | H_attention REFUTED (BOTH); mlp, attn recover | hypotheses.H_attention; arms.mlp, attn | "REFUTED (BOTH)"; REC, REC |
| table N13 | H_artifact INDETERMINATE; necessity NOT NECESSARY ×4 | hypotheses | INDETERMINATE; NOT NECESSARY ×4 |
| table N13 | GSM8K labels (the last two fragile) | gsm8k.*.label | REPLICATES ×3 |
| table M1 | (H_lead behind abstract (1)) | `m1/verdict_m1.json` H_lead | SUPPORTED |
| table J8 | pass under the frozen rule | `pilot0/GPU_CHECK2.md` lines 3, 8 | pass, both steps |
| table J10 | NO-GO | `n22/JOURNAL.md` 18:00Z (not in the verdict file) | NO-GO |

Word count: the text under "## Abstract" up to "## Evidence table" is **297 words** by `len(text.split())`, under the 300 limit.
The draft-3 commit subject says 296; the difference is the counting method, not the text.

Banned words: none of the three banned words appears anywhere in `20_ABSTRACT.md` (case-insensitive scan of every line; the scan
fires on a planted line).

## 5. What the abstract says that its artifacts do not support

1. **"about one top-50 activation slot in fifty"** (A3). The shift is 0.56 to 0.87 slots of 50 per pole; no band reaches one.
   `m1/READ_head.md` uses the same phrase, so the error comes from the head read. A fair wording: "about three-quarters of a slot
   in fifty (1.1–1.7 % of a pole's top-50 slots)".
2. **"a 65 % code-and-math mix held them."** True in three bands. In layers 8–15, M − start is −0.0070, beyond that band's bar
   0.0066, and 41 % of G's loss there (`verdict_m1.json` exploratory.shift_from_start.M; `m1/READ_head.md` says "0.007 below it in
   layers 8–15").
3. **"by subtraction"** rests on a dated exploratory analysis (`m1/EXPLORATORY_turnover.md`). By its frozen rule band 0 reads
   "within drift", and its "loss at fixed identity" label also fired on a pure stage-1 pair (planted input P1); the reading
   rests on the arrival fractions, which the head read and no one recounted. The opening clause "every claim tested against a
   control fixed before the data were read and recounted independently" does not cover this claim, the post hoc "the dating
   rule nearly guarantees" (`n12/READ_head.md` addendum, from the reader's records), or the exploratory 65 of 71 (N2V-3, "not a
   registered readout").
4. **"orders a unit's poles as zero ablation does in 65 of 71 units"** (A13). 65 of 71 is the top-pole match. The full order
   agrees in 60 of 71. `n2v/READ_head.md` writes "ranks a unit's poles with the same order as zero ablation in 0.915 of units";
   0.915 there is the top-pole share (and, by coincidence, also the mean Kendall τ), not the share of units in the same order.
5. **"a gradient instrument that, with the loss in fp32"**. The prediction N2V validated is N12's banked bf16 first-order
   prediction (`verdict_n2v.json` test.window.validation.prediction "pred_bf16"); what was read in fp32 is the zero-ablation loss
   change it was compared with. The table row "the same instrument in fp32" has the same ambiguity. A faithful wording: "N12's
   bf16 gradient prediction, read against zero ablation with the loss in fp32".
6. **"At the birth the carrier's weights are not the event."** N1 measured how much the input rows and the output column
   changed (e, a z-scored log of the relative change r per bracket), not the direction of the change. `n1/READ_head.md` says "what is special about a real birth is not in the carrier's
   weight norms" and names a projection onto the cluster direction as the next instrument. The abstract's "weights" is broader
   than the measurement. The head also asked that the BELOW label be quoted with its margin sentence (freeze half AT, −0.217
   [−0.638, 0.204]; pooled −0.320 [−0.604, −0.035]) and called the effect small (−1.6 % of r per window); the abstract keeps only
   "a two-birth margin".
7. **"the RL model without its top set 91.6"** is the k 1 % complement only. The other three read 90.4, 90.8, 90.6; the head writes
   "90.4 to 91.6".
8. **"The gain is diffuse."** `n13/READ_head.md` limits "diffuse" to neuron subsets: DARE carries the gain at 33 to 1,000 times the
   whole change's energy, so the result does not say the gain is spread at matched energy. The abstract states it without that
   limit.
9. **Next step, "89.8 of 92.3 on GSM8K"** for the 0.1 % sample (S13): it is the 1 % DARE arm. `n13/READ_head.md` (practical
   dimension) carries the same attribution.
10. **Next step, "eight arms"**: the design described has six arms. If N13's identity pairs and known number are rerun rather than
    reused from N13's banks, the arm count and the cost rise further. If GSM8K is read on every arm, the cost is 27–39 GPU-h,
    not 17–21.
11. **Next step, "carries it as well as the whole change does"**: DARE equals the whole MLP change (86.4; dare − mlp −1.2 to +0.4),
    not the whole RL change (92.0). The sentence's earlier clause names the MLP change, so the reading is recoverable, but "the
    whole change" alone reads as the RL model.

## Files

- `studies/next/abscheck/check_abstract.py`: recomputes every number above from the artifacts (json only).
- `studies/next/abscheck/check_abstract_output.txt`: its output on this worktree; 4 MISMATCH lines, the four listed at the top.

## Re-check of draft 4 (2026-11-08)

Draft 4 is clean-neurons `95df5fa58`, merged into abscheck as `42e38c5b9` (the merge also brought `508e28f5f`, the journal entry,
and the N13 figure and REGISTRY update). The script's draft-4 block is at the end of `abscheck/check_abstract.py`; its output is
the "DRAFT4" part of `abscheck/check_abstract_output.txt`. The lines above that marker check draft 3's numbers and are kept as
the record of draft 3.

**Answer: 25 items re-checked; 18 EQUAL, 4 ROUNDED fairly, 3 MISMATCH.** The four draft-3 mismatches are corrected. Three
remain, none of them a wrong artifact value:

1. "per neuron" in result (1): the measure is per pole (each pole has its own top-50; a neuron has two poles).
2. The closing sentence, "No neuron subset carries the gain": the tested subsets are 0.1–3 % of neurons. The complements of those
   sets carry the gain (comp_k REC, 90.4–91.6), and so does all MLP (REC, 86.4).
3. "13–16 GPU-h": six arms at the stated 2.0–2.6 GPU-h per arm cost 12.0–15.6 GPU-h.

Word count: **295 words** by `len(text.split())` (the draft's header says 293), under 300. Banned words: none of the three
appears anywhere in `20_ABSTRACT.md`.

| # | item as written in draft 4 | artifact and field | value found | status |
|---|---|---|---|---|
| D1 | (1) "moved 0.6–0.9 of a top-50 activation slot" (G) | `m1/verdict_m1.json` exploratory.shift_from_start.G × 50 | 0.5645, 0.8514, 0.8662, 0.8054 | ROUNDED fairly |
| D2 | (1) "a 65 % … mix moved at most 0.35 of a slot" (M) | same, .M, \|value\| × 50 | 0.0194, 0.3519, 0.0034, 0.1755 | ROUNDED fairly (0.352 at most; as a strict bound 0.35 is exceeded by 0.002) |
| D3 | (1) "per neuron" (the object of D1 and D2) | `m1/READ_head.md` "The measure, per neuron pole"; `m1/EXPLORATORY_turnover.md` cm_share(p) per pole | the fraction is of one pole's 50 slots; per neuron (two poles, 100 slots) the shift is 1.1–1.7 slots | MISMATCH: say "per pole" |
| D4 | (3) "ranks 284 fp32 zero ablations at Spearman 0.960 [0.915, 0.990]" | `n2v/verdict_n2v.json` test.window.validation | n_pairs 284; measurement dL (fp32); prediction pred_bf16; 0.95996 [0.91547, 0.99022] | EQUAL (fp32 now sits on the ablations, where it belongs) |
| D5 | (4) "the size of the carrier's weight change is not the event" | `n1/READ_n1.md` (e, a z-scored log of the relative change r); `n1/READ_head.md` "not in the carrier's weight norms" | a size measure, not a direction | EQUAL |
| D6 | (4) "the freeze half alone at selection level" | `n1/thresholds.json` discovery.pair_in_real_dt; `n1/READ_head.md` "The freeze half alone read AT" | −0.217 [−0.638, 0.204], interval spans 0: AT | EQUAL |
| D7 | (5) "the RL model without its top sets 90.4–91.6" | `n13/verdict_n13.json` arms.comp_k* | 90.4, 90.8, 91.6, 90.6 | EQUAL |
| D8 | (5) "No neuron subset carries the gain; a sparse unbiased estimate of the whole change does." | arms.top_k*, rndE_k*, rndT_k*, rnd0_k1, topF_k1 (NONE); comp_k* (REC); mlp (REC); dare_k* (REC, MLP entries only); `n13/READ_head.md` "no set of 0.1 to 3 % of whole neurons" | the 0.1–3 % subsets are NONE; the complements (97–99.9 % of neurons) and all MLP are REC; DARE estimates the whole MLP change | MISMATCH (scope): "No 0.1–3 % neuron subset carries the gain; a sparse unbiased estimate of the whole MLP change does." |
| D9 | table N13: "top − control −1.8 to +3.4" | comparisons top_k*\|rndE_k*, rndT_k* | −1.8 to +3.4 | EQUAL |
| D10 | table N1: freeze half at `n1/thresholds.json` discovery | discovery.pair_in_real_dt | −0.21660 [−0.63788, 0.20417] | EQUAL |
| D11 | table N1: pooled at `n1/read_n1/verdict_recount.json` confounds.c_halves | confounds.c_halves.in.pooled_both_halves | −0.31953 [−0.60403, −0.03535] | EQUAL |
| D12 | table N2V: "full signed order 60 of 71" | `n2v/read_n2v/r6_checks.json` a_within_unit.test.window.kendall_signed.share_tau_1 | 0.84507 × 71 = 60 | EQUAL |
| D13 | table J10: "the NO-GO in `studies/next/JOURNAL.md` 2026-11-06 12:05"; read by "head (journal)" | `studies/next/JOURNAL.md` entry 2026-11-06 12:05 CST | "J10 … ends at T_15 on its registered go/no-go: NO-GO" | EQUAL |
| D14 | table M1 turnover: "(exploratory, not recounted)" | `m1/EXPLORATORY_turnover.md` (dated exploratory); read by the head only | as written | EQUAL |
| D15 | next step: "six arms" | the design as described: r 16, 64, 256, each against a random subspace | 3 × 2 = 6 | EQUAL |
| D16 | next step: "2.0–2.6 GPU-h per MATH-500 arm" | `n13/s4/*/*/arm_record.json` wall_min, MATH-500-only arms (n 20) | 1.995–2.615 h | EQUAL |
| D17 | next step: "3.4–4.8 with GSM8K" | same, arms with GSM8K (n 7) | 3.432–4.827 h | EQUAL |
| D18 | next step: "six arms, 13–16 GPU-h" | 6 × the stated 2.0–2.6 | 12.0–15.6 (banked walls 12.0–15.7) | MISMATCH: 12–16. The low end 13 follows only if every arm costs B2-02's launch average, 2.24 h × 6 = 13.4 |
| D19 | next step: "about 19 with GSM8K on the rank-64 pair" | 4 × (2.0–2.6) + 2 × (3.4–4.8) | 14.8–20.0 (banked walls 14.8–20.1) | ROUNDED fairly (inside the range, in its upper half; "15–20" states it) |
| D20 | next step: "the 1 % sample reads 89.8 of 92.3 on GSM8K" | `n13/read_n13/checks_read.json` i_gsm8k.dare_k1, rl | 89.765, 92.343 | EQUAL |
| D21 | next step: "r = 16, 64, 256, design values" | none needed; r = 64 in `n13/READ_head.md` | labelled as design values | EQUAL |
| D22 | next step: "the rank-r builds (one SVD per matrix) are not costed" | no SVD build in N13's arms | as written | EQUAL |
| D23 | `m1/READ_head.md`: "by 0.6–0.9 of one slot … the shift is −0.0113 to −0.0173 of 50 slots" | `verdict_m1.json` shift_from_start.G; the file defines the measure per neuron pole (line 12) | 0.56–0.87 of a pole's slot | EQUAL |
| D24 | `m1/READ_head.md`: "moves them by at most 0.35 of a slot … −0.0070 against the bar 0.0066" | shift_from_start.M[1]; bar[1] | 0.352 slots; −0.00704 against 0.00656 | ROUNDED fairly |
| D25 | `n13/READ_head.md`: "the 1 % sample reads 89.8 against 92.3 on GSM8K" | `read_n13/checks_read.json` i_gsm8k | 89.765, 92.343 | EQUAL |

Still open, outside this list:

- `n2v/READ_head.md` (practical dimension) still says the forward-mode pass "ranks a unit's poles with the same order as zero
  ablation in 0.915 of units". 0.915 is the top-pole share. The same signed order holds in 60 of 71 units (0.845). The abstract
  and the table no longer use the sentence.
- The next step still says DARE "carries it as well as the whole change does". The artifacts support "as well as the whole MLP
  change" (dare − mlp −1.2 to +0.4); the RL model reads 92.0 against DARE's 85.2–86.8.

## Re-check of draft 5 (2026-11-08)

Draft 5 is clean-neurons `2497fbb71`, merged into abscheck as `73360ab24`. The script's draft-5 block is at the end of
`abscheck/check_abstract.py`; its output is the "DRAFT5" part of `abscheck/check_abstract_output.txt`.

**Answer: 9 items; 6 EQUAL, 2 ROUNDED fairly, 1 MISMATCH.** Everything in `20_ABSTRACT.md` and `m1/READ_head.md` is EQUAL or
ROUNDED fairly. The remaining mismatch is in `n2v/READ_head.md`, and it came from this check. My draft-4 note quoted 0.845 for the
"same order" share. That is the value for the bf16 prediction's readout (window). The head's sentence is about the fp32
forward-mode pass, whose readout is window_jvp, and there the share is 0.873.

Word count: **298 words** by `len(text.split())`, equal to the header's "298 words", under 300. Banned words: none of the three
appears anywhere in `20_ABSTRACT.md`.

| # | line as written in draft 5 | artifact and field | value found | status |
|---|---|---|---|---|
| E1 | abstract (1): "0.6–0.9 of a top-50 activation slot per pole" | `m1/verdict_m1.json` shift_from_start.G × 50; `m1/READ_head.md` line 12 "per neuron pole" | 0.5645–0.8662 per pole | EQUAL |
| E2 | abstract (1): "moved at most 0.36 of a slot" | shift_from_start.M, \|value\| × 50 | largest 0.3519 | ROUNDED fairly (a true bound; 0.35 is nearer) |
| E3 | `m1/READ_head.md`: "moves them by at most 0.36 of a slot" | same | 0.3519 | ROUNDED fairly |
| E4 | abstract closing: "No 0.1–3 % neuron subset carries the gain; a sparse unbiased estimate of the whole MLP change does." | `n13/verdict_n13.json` arms top_k*, rndE_k*, rndT_k*, rnd0_k1, topF_k1 (all NONE); dare_k* (REC, MLP entries only) | as written | EQUAL |
| E5 | next step: "six arms, 12–16 GPU-h" | 6 × 2.0–2.6 GPU-h (banked MATH-500 arms 1.995–2.615 h) | 12.0–15.6 | EQUAL |
| E6 | next step: "15–20 with GSM8K on the rank-64 pair" | 4 × (2.0–2.6) + 2 × (3.4–4.8) | 14.8–20.0 | EQUAL |
| E7 | next step: "carries it as well as the whole MLP change does" | `n13/read_n13/checks_read.json` j_other.dare_vs_mlp | dare − mlp −1.2, +0.4, −0.6, −0.2 | EQUAL |
| E8 | `n2v/READ_head.md`: the fp32 forward-mode pass "names the same largest-effect pole as zero ablation in 0.915 of units" | `n2v/read_n2v/r6_checks.json` a_within_unit.test.window_jvp.top_abs_pole_match | 0.9155 (65 of 71; the window readout gives the same) | EQUAL |
| E9 | `n2v/READ_head.md`: "(the full signed order of the four poles in 0.845)" | same, window_jvp.kendall_signed.share_tau_1 | 0.8732 (62 of 71); 0.845 (60 of 71) is the window readout, N12's bf16 prediction | MISMATCH: 0.873 for the forward-mode pass |

The evidence table's N2V row ("full signed order 60 of 71") is about the window readout and stays EQUAL.

## Check of draft 6 (2026-11-08)

Draft 6 is clean-neurons `a72695234` (N6 merged as `aadbf97a8`), merged into abscheck with `13501d3b9` (journal). The N6 rows are
recounted by `abscheck/check_n6.py` (output `abscheck/check_n6_output.txt`), which reads N6's files from outside their directory.
Fields used from `n6/banks/L2a-01/edits.jsonl` (113 records): `role`, `kind`, `layer`, `half`, `dL`, `R`. The names "dL" and "R"
are the file's own.

**Answer: 31 items; 31 EQUAL, 0 ROUNDED, 0 MISMATCH.** Every number in the two N6 rows equals its file. Both recounts from
`edits.jsonl` reproduce the rows. The reworded abstract changes no number from draft 5. The draft-5 mismatch in
`n2v/READ_head.md` is corrected (0.873, 62 of 71).

Word count: **300 words** by `len(text.split())`, equal to the header and at the limit. Banned words: none of the three appears
anywhere in `20_ABSTRACT.md`.

### (A) The two N6 rows against their files

| # | text as written | artifact and field | value found | status |
|---|---|---|---|---|
| F1 | headline `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT` | `n6/verdict_n6.json` headline | the same string | EQUAL |
| F2 | DOES NOT COUNT | verdict H_age, H_adj, H_eff .label (realized UNDECIDED_NO_POWER) | DOES NOT COUNT ×3 | EQUAL |
| F3 | G-PC col > rand in 2 of 8 PC layers | `n6/banks/L2a-01/metrics.json` gates.G_PC n_col_gt_rand, n_layers | 2, 8 (layers 0, 4, …, 28) | EQUAL |
| F4 | (need 6) | gates.G_PC.need | 6 | EQUAL |
| F5 | swapped labels 6 of 8 | gates.G_PC.failing_input_swapped | 6 | EQUAL |
| F6 | (failing input passed) | gates.G_PC.failing_input.pass; verdict failed_gates | true; ["G_PC", "G_PC:failing_input_passed"] | EQUAL |
| F7 | G-POWER τ_s 1.5096e-6 | `n6/thresholds.json` G_POWER.value (= verdict G_POWER) | 1.50962e-6 | EQUAL |
| F8 | against 0.5 × median \|D_col\| 6.363e-7 | G_POWER.bound; median_abs_D_col | 6.36293e-7; 1.27259e-6 | EQUAL |
| F9 | ratio 1.186 | G_POWER.ratio | 1.18626 | EQUAL |
| F10 | bar 0.5 | G_POWER.rule | "tau_s <= 0.5 x median \|D_col\|" | EQUAL |
| F11 | no test pair read | verdict n_test_read | 0 | EQUAL |
| F12 | p_col 0.762 | thresholds discovery.p_col | 0.76190 | EQUAL |
| F13 | [0.549, 0.894] | discovery.p_col_ci | [0.54909, 0.89372] | EQUAL |
| F14 | 16 of 21 | discovery.p_col × discovery.n | 16.00 of 21 | EQUAL |
| F15 | p_rand 0.381 | discovery.p_rand | 0.38095 | EQUAL |
| F16 | R 0.095 | discovery.R | 0.09524 | EQUAL |
| F17 | L2a 18.93–18.96 GPU-h | `infra/aws_pg/operations/2026-11-07_next_n6.md` line 79 | 18.927 to 18.961 h | EQUAL |
| F18 | N6 19.58–19.63 | same, lines 80 and 88 | 19.581–19.626 GPU-h | EQUAL |

### (B) Recounts from `edits.jsonl`

| # | text as written | recount | value found | status |
|---|---|---|---|---|
| F19 | col > rand in 2 of 8 PC layers, layers 24 and 28 | PC records (`role` "pc"): `dL`(col) − `dL`(rand) > tau_eps 3.3651e-8 (`thresholds.json` tau_eps = G_PC.tau) | 2 of 8: layer 24 (+3.591e-2 against +8.03e-4), layer 28 (+1.479e-3 against +1.334e-3) | EQUAL |
| F20 | swapped labels 6 of 8 | `dL`(rand) − `dL`(col) > tau_eps | 6 of 8 (layers 0, 4, 8, 12, 16, 20) | EQUAL |
| F21 | the random edit's realized residual change 1.3–2.5 × the column edit's in all 8 PC layers | `R`(rand) / `R`(col) per PC layer | 2.507, 1.828, 1.815, 1.913, 2.056, 1.746, 1.335, 1.623 (all > 1) | EQUAL |
| F22 | freeze-pair edits matched, medians 60.8 vs 60.7 | median `R` over `role` "pair", `half` "freeze", `kind` "col" and "rand" (42 records each; no test-half records in the file) | 60.817, 60.678 | EQUAL |

### (C) The reworded sentences

A comparison of every number token in the draft-5 and draft-6 abstracts finds no changed number. Draft 6 drops one repeat of
"0.1–3 %" and adds "(6)".

| # | sentence | check | status |
|---|---|---|---|
| F23 | the opening ("every claim was tested against a control fixed in advance and recounted independently") | no number; same meaning as draft 5 | EQUAL |
| F24 | result (1) (verb "moved" now carried from the first clause; "activation" dropped) | 0.6–0.9, 0.36, 65 %, 40 %, 66–76 % unchanged | EQUAL |
| F25 | result (2) | top 10, 22,016, 59 %, 76 % unchanged | EQUAL |
| F26 | result (3) | 284, 0.960 [0.915, 0.990] unchanged; fp32 still on the ablations | EQUAL |
| F27 | result (4) | −0.43 [−0.82, −0.06], 67 pairs, two-birth margin unchanged | EQUAL |
| F28 | result (5) numbers | 92.0, 57.2, 0.1–3 %, 57.6–63.8, 0.1 %, 85.2, 86.4, 83.0, 90.4–91.6 unchanged | EQUAL |
| F29 | result (5) lead: "is carried by no 0.1–3 % neuron subset but by a sparse unbiased estimate of the whole MLP change" | the same two claims as the draft-5 closing sentence, with the same scope (0.1–3 % subsets NONE; DARE REC on MLP entries); nothing added | EQUAL |
| F30 | result (6): "does not count: positive control inverted, noise above the effect" | G-PC 2 of 8 with swapped labels 6 of 8; τ_s 1.186 × median \|D_col\|; headline DOES NOT COUNT | EQUAL |
| F31 | `n2v/READ_head.md`, the draft-5 mismatch | "the full signed order of the four poles in 0.873, 62 of 71" against `r6_checks.json` a_within_unit.test.window_jvp.kendall_signed.share_tau_1 0.87324 | EQUAL |

Two notes, not mismatches:

- **The "read by" column.** N6 has no `READ_n6.md` and no `READ_head.md`. The head's read is the program `JOURNAL.md` entry for
  2026-11-08 09:15 CST and the head's decision in `n6/JOURNAL.md` at 15:15Z. The rows say "executor, head", which matches.
- **The opening clause.** It says every claim was "recounted independently". For result (6), the independent recounts are the
  head's analytic recount of the G-POWER ratio (1.181, against the verdict's 1.186, in that journal entry) and this check's recount
  of G-PC, the R ratios and the medians. No independent reader recounted N6's full verdict.
