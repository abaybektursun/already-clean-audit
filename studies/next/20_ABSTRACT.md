# What this program learned, and the highest-leverage next step

FINAL, 2026-11-08 09:30 CST (Fable): draft 6 after the fresh reader's fourth pass (`20_ABSTRACT_CHECK.md`, abscheck 763c85fd3: 31 items,
all equal; the passes on drafts 3–5 checked 175, 25 and 9 numbers, every mismatch corrected). Abstract 300 words. Six studies are
closed; nothing is running. Every number below names its artifact; the read-by column says who recounted it.

## Abstract

We asked how training shapes single MLP neurons in OLMo-3 7B across checkpoints; every claim was tested against a control fixed in
advance and recounted independently. (1) The stage-2 data mix moves where a neuron's strongest activations land: 100B tokens without
code or math moved 0.6–0.9 of a top-50 slot per pole off code and math documents; a 65 % code-and-math mix at most 0.36 of a slot; a
40 % mix landed 66–76 % of the way between. (2) A new activation cluster's carrier can be named one checkpoint early (top 10 of
22,016 poles in 59 % of held-out births), but a random group of another pole's future contexts, dated alike, names its pole more
often (76 %); twins matched on distinct-token count close the gap. (3) The carrier wins by its own rise, and the training signal one
checkpoint earlier does not single it out (a gradient instrument ranking 284 fp32 zero ablations at Spearman 0.960 [0.915, 0.990]).
(4) At the birth the size of the carrier's weight change is not the event: its input rows change less than a dated twin's (−0.43
[−0.82, −0.06], 67 pairs; two-birth margin; the freeze half alone at selection level). (5) The RL math gain of
OLMo-3.1-7B-RL-Zero-Math (MATH-500 92.0 against 57.2) is carried by no 0.1–3 % neuron subset but by a sparse unbiased estimate of
the whole MLP change: its most-changed neurons transplanted into the base read 57.6–63.8, as do random neurons matched on change
energy; a random 0.1 % of MLP entries scaled to estimate the whole change reads 85.2, all MLP alone 86.4, attention with norms alone
83.0, and the RL model without its top sets 90.4–91.6. (6) A sixth study, whether early-born poles cost more to edit, does not
count: positive control inverted, noise above the effect.

## Evidence table (one row per load-bearing number)

| claim | number | artifact | read by |
|---|---|---|---|
| M1: G − start by band (top-50 slot fraction) | −0.0113, −0.0170, −0.0173, −0.0161 | `studies/next/m1/verdict_m1.json` exploratory.shift_from_start.G | executor, Opus reader, head |
| M1: M − start by band | −0.0004, −0.0070, +0.0001, +0.0035 | same, .M | same |
| M1: M − G by band and the drift bar | +0.0109, +0.0100, +0.0174, +0.0196 against 0.0082, 0.0066, 0.0023, 0.0035 | `verdict_m1.json` lead, bar | same |
| M1: λ = Δ_RG / Δ_MG | 0.70, 0.68, 0.66, 0.76 | `m1/READ_m1.md` | reader, head |
| M1: identity pair | 0 in every band; counts.npz sha256 8c332d7d… twice | `verdict_m1.json` G_ID; banks | same |
| M1: loss at fixed identity, not a move between poles (exploratory, not recounted) | arrivals under G 0.085–0.165, the lowest of nine pairs | `m1/EXPLORATORY_turnover.md` | head |
| M2: carrier in top 10 one checkpoint before birth | 42 of 71 = 0.592 [0.475, 0.698]; bar 0.0122 | `studies/next/m2/verdict_m2.json` labels.H_B | executor, reader, head |
| M2: twins | 53 of 70 = 0.757; carrier − twin −0.171 [−0.286, −0.057] | same, pb_pre_B, H_sel_B | same |
| M2: repeated-token births carry the gap | varied −0.014 [−0.113, +0.085]; repeated −0.397 [−0.513, −0.282] (post hoc) | `m2/read/r5b_token_share.json` | reader |
| N12: own rise F | +0.951 [0.738, 1.180]; twins +0.77 and +0.94 | `studies/next/n12/verdict_n12.json` labels.H_cf; `n12/READ_n12.md` | executor, reader, head |
| N12: leaders' drop, real against M2 twins | 0.939 against 1.164 decades; −0.226 [−0.408, −0.056] | same, H_away_sel_pb | same |
| N12: distinct-token twins | real 42 of 71, dt 41 of 67, pb 53 of 70; real − dt −0.030 [−0.149, +0.090] | same, H_sel_dt, H_dt_pb | same |
| N12: G-VAL in bf16 | Spearman 0.100 freeze, 0.043 test; measured ΔL 27 times the prediction at the median | same, gates; `n12/READ_n12.md` G-VAL | same |
| N2V: the same instrument in fp32, test half | Spearman 0.960 [0.915, 0.990], shuffle 97.5th 0.118, 284 pairs, 71 units; freeze 0.971 [0.939, 0.993] | `studies/next/n2v/verdict_n2v.json` test.window; `n2v/thresholds.json` | executor, reader (bitwise recount of 612 pairs), head |
| N2V: finite difference, own position, fp32 prediction | 0.987 [0.971, 0.995]; 0.986 [0.969, 0.996]; 0.967 [0.923, 0.994] | same, test.fd, test.own, test.window_jvp | same |
| N2V: within-unit top pole matched (exploratory) | 65 of 71 = 0.915 [0.845, 0.972]; full signed order 60 of 71; chance level 0.25; within-unit permutation leaves ρ 0.286 | `n2v/read_n2v/r6_checks.json` a | reader, head (exploratory) |
| N2V: zero ablation against first order | ΔL / prediction 0.94–1.06 by decile; fd slopes 0.988, 1.008; second-order share 0.18–0.20 | same, c | reader, head (exploratory) |
| N12 gradient labels, counted after N2V | H_signal REFUTED 0.527 [0.451, 0.600] against 0.4225; sel pb AT −0.039 [−0.134, 0.057]; sel dt AT −0.020 [−0.124, 0.084]; H_bridge UNDECIDED −0.013 [−0.072, 0.046] | `verdict_n2v.json` n12_reread; `n2v/read_n2v/r5_reread.json` | executor, reader, head (a dated re-reading) |
| N1: input rows at the birth, real − dt on e | −0.432 [−0.817, −0.059], 67 pairs, 23 above and 44 below; BELOW_SELECTION; margin two births; freeze half −0.217 [−0.638, 0.204]; pooled −0.320 [−0.604, −0.035] | `studies/next/n1/verdict_n1.json`; `n1/read_n1/fragility.json`; freeze half `n1/thresholds.json` discovery; pooled `n1/read_n1/verdict_recount.json` confounds.c_halves | executor, reader (recount to 2.5e-14), head |
| N1: output column | −0.258 [−0.654, 0.130], AT_SELECTION | `verdict_n1.json` | same |
| N1: neither side above the matched floor | in −0.297 [−0.550, −0.059] against −0.053; out −0.039 [−0.276, 0.202] against 0.147 | `verdict_n1.json`; `n1/thresholds.json` | same |
| N1: natural units | input rows −1.6 % [−3.2 %, +0.03 %] of r per window | `n1/read_n1/scale.json` | reader, head (exploratory) |
| J8: step 61,001 and 100,001 pass under the frozen rule | S − W +0.0079 and +0.0094 ≤ 0.0124; neighbours farther | `studies/next/pilot0/GPU_CHECK2.md` | executor, head |
| J10: NO-GO | 7 of 1,200 items answered; K0 exact | `studies/next/n22/verdict_n22_t15.json`; banks/l1-01; the NO-GO in `studies/next/JOURNAL.md` 2026-11-06 12:05 | executor, head (journal) |
| N13: known number | 39.96 against 40.0 ± 1.6 | `studies/next/n13/r0/s2setup-07/r0_known/metrics.json` | executor, head |
| N13: the gate | base 57.2, rl 92.0, zero 55.4, id 91.2; G 34.8 against 13.60; σ 1.924 | `studies/next/n13/thresholds.json` | executor, head |
| N13: top-k transplants, MATH-500 (n 500) | 57.6, 61.4, 61.8, 63.8 at k 0.1, 0.3, 1, 3 %, all NONE against H 74.6 | `studies/next/n13/verdict_n13.json` arms.top_k*, thresholds | executor, reader (every answer re-graded), head |
| N13: energy-matched random neurons | rndE 58.8, 59.8, 59.2, 63.8; rndT 59.4, 58.0, 60.0, 62.8; top − control −1.8 to +3.4, smallest BH p 0.107 | same, arms.rndE_k*, rndT_k*, comparisons | same |
| N13: DARE (random entries, scaled 1/q) | 85.2, 86.8, 85.8, 86.2; top − dare −27.6 to −22.4, BH p ≤ 2.6e-21; dare − mlp within 1.2 | same, arms.dare_k*; `n13/read_n13/checks_read.json` j | same |
| N13: whole families and complements | mlp 86.4, attn 83.0 (+3.4, BH p 0.107, all in cap-hit problems); comp 90.4, 90.8, 91.6, 90.6 | same, arms.mlp, attn, comp_k* | same |
| N13: GSM8K at 1 % (n 1,319) | top − dare −23.7 [−26.5, −21.1]; top − rndE +2.1 [+0.2, +3.9]; top − rndT +1.9 [+0.1, +3.6] (the last two fragile) | same, gsm8k; `read_n13/checks_read.json` g | same |
| N13: hypotheses | H_local REFUTED (k_rec NONE); H_diffuse SUPPORTED (0.1 %); H_attention REFUTED (BOTH); H_artifact INDETERMINATE (J 0.642, 0.534); necessity NOT NECESSARY ×4 | same, hypotheses | same |
| N13: cost | 102.555 GPU-h; S3 and S4 $230.32 | `infra/aws_pg/operations/2026-11-06_next_n13_s4.md` | executor, head |
| N6: edit the young, gates | DOES NOT COUNT: G-PC col > rand in 2 of 8 PC layers (need 6), swapped labels 6 of 8 (failing input passed); G-POWER τ_s 1.5096e-6 against 0.5 × median \|D_col\| 6.363e-7 (ratio 1.186, bar 0.5); no test pair read; headline `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT` | `studies/next/n6/verdict_n6.json`; `n6/thresholds.json` G_POWER; `n6/banks/L2a-01/metrics.json` gates.G_PC | executor, head; reader (G-PC recounted from `edits.jsonl`; G-POWER checked against `thresholds.json`, τ_s itself not recounted) |
| N6: freeze half (exploratory) and cost | p_col 0.762 [0.549, 0.894], 16 of 21; p_rand 0.381; R 0.095; the random edit's realized residual change 1.3–2.5 × the column edit's in all 8 PC layers (freeze-pair edits matched, medians 60.8 vs 60.7); L2a 18.93–18.96 GPU-h, N6 19.58–19.63 | `n6/thresholds.json` discovery; `n6/banks/L2a-01/edits.jsonl` dL, R; `infra/aws_pg/operations/2026-11-07_next_n6.md` | executor, head (exploratory); reader (R ratios and medians recounted from `edits.jsonl`) |

## The highest-leverage next step

N13 removed the localized account: no 0.1–3 % set of whole neurons carries the RL gain at its own or at the top set's energy,
while an unbiased sparse estimate of the whole MLP change (DARE, 1/q-scaled, 33–1,000 times the change's energy) carries it as
well as the whole MLP change does. What that leaves open is the shape of the change that the network tolerates such noise on. The
instrument that separates the two readings is a rank test: for each MLP matrix of the RL delta, a rank-r truncation (r = 16, 64,
256, design values) transplanted into the base, against a random subspace of equal rank and equal energy, on MATH-500 with GSM8K
held out on the rank-64 pair, under N13's frozen harness, gates and controls (the identity pairs, the known number, the seed noise). Hypothesis and its refutation:
the gain sits in a low-rank component of each matrix's change (the rank-64 arm within δ of the RL model); refuted if it reads at
the random-subspace control's level. Cost at N13's measured rate (2.0–2.6 GPU-h per MATH-500 arm on one L40S, 3.4–4.8 with GSM8K): six arms, 12–16 GPU-h,
15–20 with GSM8K on the rank-64 pair; the rank-r builds (one SVD per matrix) are not costed. Practical reading if it holds: the RL delta compresses to r × (d + n) numbers per matrix for transplant and merging; if it
fails, the diffuse account stands at the entry level and DARE's 0.1 % sample is already the cheapest carrier measured here
(85.2 of 92.0 on MATH-500; the 1 % sample reads 89.8 of 92.3 on GSM8K).
