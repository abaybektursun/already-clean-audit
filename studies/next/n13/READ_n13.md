# N13 independent read

Written 2026-11-08 by the independent reader (Claude Opus 5.5), in worktree `../interp_research-n13read`, branch `n13read`,
made from tag `n13-s4-complete` (`2d81963d9`). CPU only; no model was run.

## The answer

**The registered verdict stands under my recount, number for number.** The number that settles it: the largest top-k
transplant, top_3 %, reads **63.8 % [59.5, 67.9]** on MATH-500, below the halfway line H = 74.6 %, so k_rec is NONE and H_local is
REFUTED. DARE at the same parameter count reads 85.2 % to 86.8 % at every k (top_0.1 % − dare_0.1 % = −27.6 points, BH p 2.7e-30).

My headline, in the spec's words (§1.3, §6.3):

- **H_local REFUTED (k_rec NONE)**: top_3 % stays below H (63.8 against 74.6). At every k, rndE_k and rndT_k are within δ of top_k.
- **H_diffuse SUPPORTED (smallest k 0.1 %)**: dare_k is REC at every k, and WITHIN(dare_k, top_k) holds at every k: dare_k is
  22.4 to 27.6 points above top_k.
- **H_attention REFUTED (BOTH)**: `mlp` 86.4 REC and `attn` 83.0 REC.
- **H_artifact INDETERMINATE**: J_floor(1 %) 0.642 and J_round(1 %) 0.534, both between 0.4 and 0.7; PC-ART valid.
- **Necessity**: NOT NECESSARY at every k (comp_k 90.4 to 91.6, each within δ of rl 92.0).
- **GSM8K at 1 %**: top − dare, top − rndE, top − rndT all REPLICATES by the frozen rule. The top − rndT label is fragile (check g).

The registered verdict (`verdict_n13.json`, sha256 `cae55c04ebf8db2e7e08817b5f604bc9324a7061cf755d703de12ef041250831`, from
`results/next_n13/s5-01/`) gives the same six lines, `verdict` COMPLETE, no stamp, nothing parked. Its figure title reads "Top 0.1-3 %
MLP neurons recover little of the MATH-500 gain; DARE on as many entries recovers most". `recount_n13.py` prints `RECOUNT_OK`. Of 465
numbers and labels in the verdict, 465 equal mine (`read_n13/compare_verdict.json`).

What the verdict does not say, from my checks: DARE is matched on entry count, not on energy. Its realized MLP change energy is 33 to
1,000 times the full RL change's (check d), and it reads the same as the `mlp` arm (all of the MLP change) within 1.2 points at every k
(check j). The energy-matched random arms read the same as the top sets (58.0 to 63.8 against 57.6 to 63.8), all under Q.
So "H_diffuse SUPPORTED" here means: the DARE estimate of the whole MLP change does what the whole MLP change does, and no set of
0.1 % to 3 % of whole neurons does, at its own or at the top set's energy.

## Order of work and provenance

1. Worktree at the tag; the banks downloaded from `s3://pg-lineage-769063704696/results/next_n13/s5-01/banks/` to
   `<SP>/n13read/s3/banks/`. `manifest_sha256.json` sha256 `56c87b3c06c47dff17302f352d0436063add2f5398c57a4abb58f781bed790de`
   (as given). All 165 files, 411,082,754 bytes, equal the manifest's size and sha256 (`read_n13/manifest_check.json`, script
   `verify_manifest.py`). `_inputs/thresholds.json` sha256 `60bf6d4e…` equals the worktree's frozen file.
2. Gates, accuracies, verdict quantities and checks computed by my own scripts and committed before the verdict was downloaded:
   `fd42f7e08` (steps 1 to 3) and `bd2f6d7d6` (step 4). My scripts import no n13 code, except the planted grader pairs (data) in
   `gates_read.py`.
3. Then the verdict was downloaded, `recount_n13.py` was run, and every number compared (`0f85c5242`).
4. The claims record, registry, samples coverage and figure were read by exact path from branch `n13` (no `git log`).

The interpreter is the n13 venv (Python 3.12.12, numpy 2.5.4, math-verify 0.9.0), every script run with `-I`.

## Step 1. Gates (§6.1), from the banks and the R0 and R1 records

Script `read_n13/gates_read.py`, output `gates_read.json`. Every gate passes and every registered failing input fails.

| gate | value | pass rule | registered failing input: value | fails? |
|---|---|---|---|---|
| G-GRADER | 24/24 equivalent true, 16/16 non-equivalent false, 500/500 golds self-equal, derangement rate 0.006 (recomputed on the banked golds, rule `dollar_both`) | all, and rate ≤ 0.006 + 0.01 = 0.016 | derangement scored as identity: 3 of 500 (needs 500) | yes |
| G-TEMPLATE | setup 500/500 rendered as §2.2, first id not BOS, chat template sha256 `4380e622…`, ids sha256 `ff88a606…`; every arm's `metrics.json` template pass (n 500 and 1,319); my check: every prompt in all 42 banks is the §2.2 prefix + problem + suffix, and prompt text and token metadata are identical across arms | §2.2 assertions | `add_generation_prompt=False` render: differs | yes |
| G-INIT | (a) shard hashes pass; (b) unchanged MLP share 0.5626 (main) − 0.0055 (stage 2) = 0.557 | (a) and (b) ≥ 0.10 | (a) stage-2 oids: fails; (b) swapped: −0.557 | yes |
| G-PARTITION | 355 names; MLP 4,328,521,728 + attention 2,147,483,648 + norms 528,384 + embeddings 821,477,376 = 7,298,011,136 (equals §0.2) | cover and sum | index without layer-0 `down_proj`: fails | yes |
| PC-RANK | recovery 1.0 | ≥ 0.99 | rows of `down_proj`: 0.0 (chance 0.013) | yes (≤ 0.05) |
| G-KNOWN (R0) | 39.96 (`primary_score` × 100) | \|x − 40.0\| ≤ 1.6 | deranged golds: 1.12 | yes (< 10.0) |
| PC-ART (A2) | P+: J_floor 0.006, J_round 0.844 → SUPPORTED; P−: J_floor 1.000, J_round 1.000 → REFUTED | SUPPORTED on P+ and REFUTED on P− | P− must give REFUTED: it does | yes |
| G-HASH | `zero` = base and `id` = RL on all 355 tensors (R1 and both builds); only 1 of 355 tensors is equal between base and RL | every tensor | `id` with neuron (0, 0) reverted, `zero` with it added: mismatch | yes |
| G-MASK | 20 mask arms: realized changed gate rows, up rows and down columns equal S ∩ {nonzero change} (banked flag, and my comparison of the realized lists with the expected lists per layer); realized ⊆ S; set sizes equal `sets.json` | equality | `top_1%` with `down_proj` rows: fails (R1) | yes |
| G-COMPLETE (A25) | 42 arm × benchmark banks: 500 or 1,319 records, every index once, seeds 1,000,000 + i (A) or 2,000,000 + i (B: `zero`, `id`, `top_k1_B`), n_generated ≥ 1, a finish_reason | all | one record removed: fails | yes |
| G-GATE | G = 92.0 − 57.2 = 34.8; G' = 91.2 − 55.4 = 35.8; McNemar p 1.1e-45 | G, G' ≥ 5√2σ = 13.60; p < 0.001 | (rl-A, id-B) as the pair: G 0.8, p 0.54 | yes |
| G-ID | \|L − L'\| 0.8; \|B − B'\| 1.8 | ≤ 3√2σ = 8.16 | (base-A, rl-A) as identity: −34.8 | yes |
| S-BAND | L' 91.2 | 81.5 ≤ L' ≤ 96.7 | id-B against deranged golds: 0.8 | yes (< 81.5) |
| G-GATE-GSM | G = 92.34 − 63.61 = 28.73; G' 27.82; p 2.9e-82 (σ_gsm 1.234) | ≥ 5√2σ_gsm = 8.73 | (rl-A, id-B): G 0.30 | yes |

`n_empty` (the stop token alone, reported, not gated): base 7 / 71 (MATH-500 / GSM8K), zero 11 / 83, top_k1 9 / 72, rndE_k1 8 / 71,
rndT_k1 9 / 73, other base-like arms 6 to 11 on MATH-500; rl, id, dare, comp, mlp 0; attn 0 / 3.

**G-HASH per arm, from the arm records and from my own comparison.** The 27 S4 arm records give 355 of 355 tensors equal to R1 stage 6
for 23 arms, and 259 of 355 for the four rndE arms. The 96 that differ are the MLP tensors, which A30's stochastic rounding changed
after R1 ran (each record carries that note). My comparison of each arm's 355 recorded sha256 against `zero` (base) and `id` (RL): in
every arm, every tensor outside the family the arm changes equals the expected endpoint. Non-MLP-before arms: all 259 attention, norm
and embedding tensors equal base. comp arms: all of them equal RL. `attn`: the 128 attention and 129 norm tensors equal RL, the 96 MLP
and 2 embedding tensors equal base. `mlp`: the 96 MLP tensors equal RL, the rest equal base.

**A31 on the four rndE arms** (`build.json` → `a31`, relative difference recomputed): rndE_0.1 % 1.43e-16, rndE_0.3 % 1.18e-16,
rndE_1 % 1.02e-15, rndE_3 % 3.83e-16, each ≤ 1e-12; 0 of 355 tensors differ; realized / intended MLP energy 1.0044, 1.0004, 0.9951,
0.9919 (A30 band 5 %); c built 1.4276, 1.2780, 1.1752, 1.1134.

**Engine blocks.** All 42 arm × benchmark `metrics.json` engine blocks equal the S3 reference (`base` MATH-500, `s3gate-03`): vLLM
0.32.0, max_num_seqs 128, max_num_batched_tokens 2,048, num_gpu_blocks 12,817, FLASH_ATTN, `VLLM_USE_FLASHINFER_SAMPLER=0`, and every
`engine_args` key except `model`. The 27 S4 `engine_check.json` files in the worktree say equal, with libraries equal to S3's (torch
2.13.0, triton 3.7.1, transformers 5.19.0, tokenizers 0.23.3). No damage screen fired (at most 6 cap hits among problems 0 to 19).

**The three numbers (§5).** Fit on both sides: G-KNOWN 39.96 against 40.0 ± 1.6; \|L − L'\| = 0.8 ≤ 8.16 with the `id` hash equal to
RL's; S-BAND L' 91.2 in [81.5, 96.7]. Positive control that fired: G = 34.8 ≥ 13.60; PC-RANK recovery 1.0. Identity pairs: L − L' =
0.8, B − B' = 1.8. None is missing.

## Step 2. Accuracies, MATH-500 (n = 500, seed set A unless named)

Script `read_n13/recount_read.py`, output `recount_read.json`. Three readings per arm: the banked scores; my re-grade of the banked
extracted string against the banked gold (A1 wrap, `math_verify.verify(parse(wrap(gold)), parse(wrap(x)))`); and my own
re-extraction of the generated text by §2.4 item 2 as amended by A10 and A26, then graded. All three agree on every answer of all 42
arm × benchmark banks: 0 score disagreements and 0 extraction differences (mode and string) in 30,009 answers. Golds and prompts are
identical across arms for every problem. Wilson intervals at z = 1.959964.

| arm | right / 500 | acc, banked scores (%) | Wilson 95 % | acc, re-graded (%) | acc, re-extracted (%) | verdict acc (%) | largest banked-vs-regrade disagreement | label (reader) | label (verdict) | cap hits | mean new tokens | modes line / box / none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 286 | 57.2 | [52.8, 61.5] | 57.2 | 57.2 | 57.2 | 0 answers | (endpoint) | (endpoint) | 69 | 3350 | 220 / 143 / 137 |
| rl | 460 | 92.0 | [89.3, 94.1] | 92.0 | 92.0 | 92.0 | 0 answers | (endpoint) | (endpoint) | 16 | 4499 | 0 / 494 / 6 |
| zero | 277 | 55.4 | [51.0, 59.7] | 55.4 | 55.4 | 55.4 | 0 answers | (endpoint) | (endpoint) | 65 | 3280 | 193 / 161 / 146 |
| id | 456 | 91.2 | [88.4, 93.4] | 91.2 | 91.2 | 91.2 | 0 answers | (endpoint) | (endpoint) | 17 | 4333 | 0 / 491 / 9 |
| top_k0.1 | 288 | 57.6 | [53.2, 61.9] | 57.6 | 57.6 | 57.6 | 0 answers | NONE | NONE | 75 | 3545 | 215 / 144 / 141 |
| top_k0.3 | 307 | 61.4 | [57.1, 65.6] | 61.4 | 61.4 | 61.4 | 0 answers | NONE | NONE | 67 | 3347 | 225 / 152 / 123 |
| top_k1 | 309 | 61.8 | [57.5, 66.0] | 61.8 | 61.8 | 61.8 | 0 answers | NONE | NONE | 79 | 3715 | 213 / 151 / 136 |
| top_k3 | 319 | 63.8 | [59.5, 67.9] | 63.8 | 63.8 | 63.8 | 0 answers | NONE | NONE | 83 | 4004 | 222 / 142 / 136 |
| dare_k0.1 | 426 | 85.2 | [81.8, 88.0] | 85.2 | 85.2 | 85.2 | 0 answers | REC | REC | 48 | 4607 | 9 / 445 / 46 |
| dare_k0.3 | 434 | 86.8 | [83.6, 89.5] | 86.8 | 86.8 | 86.8 | 0 answers | REC | REC | 49 | 4569 | 9 / 451 / 40 |
| dare_k1 | 429 | 85.8 | [82.5, 88.6] | 85.8 | 85.8 | 85.8 | 0 answers | REC | REC | 52 | 4667 | 6 / 453 / 41 |
| dare_k3 | 431 | 86.2 | [82.9, 88.9] | 86.2 | 86.2 | 86.2 | 0 answers | REC | REC | 43 | 4456 | 5 / 454 / 41 |
| rndE_k0.1 | 294 | 58.8 | [54.4, 63.0] | 58.8 | 58.8 | 58.8 | 0 answers | NONE | NONE | 62 | 3148 | 228 / 148 / 124 |
| rndE_k0.3 | 299 | 59.8 | [55.4, 64.0] | 59.8 | 59.8 | 59.8 | 0 answers | NONE | NONE | 67 | 3287 | 228 / 151 / 121 |
| rndE_k1 | 296 | 59.2 | [54.8, 63.4] | 59.2 | 59.2 | 59.2 | 0 answers | NONE | NONE | 67 | 3324 | 212 / 164 / 124 |
| rndE_k3 | 319 | 63.8 | [59.5, 67.9] | 63.8 | 63.8 | 63.8 | 0 answers | NONE | NONE | 64 | 3363 | 213 / 167 / 120 |
| rndT_k0.1 | 297 | 59.4 | [55.0, 63.6] | 59.4 | 59.4 | 59.4 | 0 answers | NONE | NONE | 61 | 3133 | 229 / 146 / 125 |
| rndT_k0.3 | 290 | 58.0 | [53.6, 62.2] | 58.0 | 58.0 | 58.0 | 0 answers | NONE | NONE | 68 | 3366 | 213 / 159 / 128 |
| rndT_k1 | 300 | 60.0 | [55.6, 64.2] | 60.0 | 60.0 | 60.0 | 0 answers | NONE | NONE | 71 | 3438 | 225 / 143 / 132 |
| rndT_k3 | 314 | 62.8 | [58.5, 66.9] | 62.8 | 62.8 | 62.8 | 0 answers | NONE | NONE | 63 | 3346 | 225 / 159 / 116 |
| comp_k0.1 | 452 | 90.4 | [87.5, 92.7] | 90.4 | 90.4 | 90.4 | 0 answers | REC | REC | 28 | 4671 | 0 / 487 / 13 |
| comp_k0.3 | 454 | 90.8 | [87.9, 93.0] | 90.8 | 90.8 | 90.8 | 0 answers | REC | REC | 24 | 4570 | 1 / 488 / 11 |
| comp_k1 | 458 | 91.6 | [88.8, 93.7] | 91.6 | 91.6 | 91.6 | 0 answers | REC | REC | 24 | 4523 | 0 / 489 / 11 |
| comp_k3 | 453 | 90.6 | [87.7, 92.9] | 90.6 | 90.6 | 90.6 | 0 answers | REC | REC | 27 | 4499 | 0 / 483 / 17 |
| mlp | 432 | 86.4 | [83.1, 89.1] | 86.4 | 86.4 | 86.4 | 0 answers | REC | REC | 42 | 4469 | 7 / 459 / 34 |
| attn | 415 | 83.0 | [79.5, 86.0] | 83.0 | 83.0 | 83.0 | 0 answers | REC | REC | 60 | 4428 | 22 / 421 / 57 |
| rnd0_k1 | 298 | 59.6 | [55.2, 63.8] | 59.6 | 59.6 | 59.6 | 0 answers | NONE | NONE | 74 | 3482 | 213 / 152 / 135 |
| top_k1_B | 300 | 60.0 | [55.6, 64.2] | 60.0 | 60.0 | 60.0 | 0 answers | NONE | NONE | 76 | 3653 | 207 / 146 / 147 |
| dare_k1_d2 | 439 | 87.8 | [84.6, 90.4] | 87.8 | 87.8 | 87.8 | 0 answers | REC | REC | 45 | 4478 | 4 / 461 / 35 |
| rndT_k1_d2 | 298 | 59.6 | [55.2, 63.8] | 59.6 | 59.6 | 59.6 | 0 answers | NONE | NONE | 69 | 3399 | 221 / 147 / 132 |
| topF_k1 | 320 | 64.0 | [59.7, 68.1] | 64.0 | 64.0 | 64.0 | 0 answers | NONE | NONE | 67 | 3435 | 221 / 157 / 122 |

GSM8K accuracies are in check (i). On GSM8K too, banked scores, re-grade and re-extraction agree on every answer.


## Step 3. The verdict quantities, by my own code

Script `read_n13/verdict_read.py` (function `run`), output `verdict_read.json`.

- **σ** from the seed pairs (§0.5): σ_base = 1.924 (base-A against zero-B), σ_rl = 0.693 (rl-A against id-B); σ = 1.924; **δ = 2√2σ =
  5.441**. B 57.2, L 92.0, G 34.8, B' 55.4, L' 91.2, H = 74.6 (373 / 500), Q = 65.9 (329.5 / 500). All equal `thresholds.json` to 1e-9.
  DAMAGED below B − δ = 51.76; ABOVE above L + δ = 97.44. Labels are computed on exact counts; no arm sits on a threshold.
- **Labels** (table above): every top_k, rndE_k, rndT_k, rnd0_1 %, top_1 %_B, rndT_1 %_d2 and topF_1 % is NONE (57.6 to 64.0, all
  ≤ Q). Every dare_k, dare_1 %_d2, comp_k, `mlp` and `attn` is REC (83.0 to 91.6). No DAMAGED, no ABOVE, no PART.
- **k_rec = NONE.**
- **The primary family** (exact two-sided McNemar on discordant problems; BH over the 14 tests; paired bootstrap, 2,000 resamples of
  problems, `default_rng(0)` drawing integer indices, percentile interval). The last column is the verdict's value; every one equals
  mine.

| comparison (first minus second) | diff (points) | discordant (b / c) | McNemar p | BH p (14 tests) | bootstrap 95 % | identical text | verdict: diff, p, BH p, interval |
|---|---|---|---|---|---|---|---|
| top_k0.1 − dare_k0.1 | -27.6 | 164 (13 / 151) | 5.72e-31 | 2.67e-30 | [-32.0, -23.4] | 0.000 | -27.6, 5.72e-31, 2.67e-30, [-32.0, -23.4] |
| top_k0.1 − rndE_k0.1 | -1.2 | 66 (30 / 36) | 0.539 | 0.628 | [-4.4, +2.2] | 0.294 | -1.2, 0.539, 0.628, [-4.4, +2.2] |
| top_k0.1 − rndT_k0.1 | -1.8 | 65 (28 / 37) | 0.321 | 0.49 | [-5.0, +1.2] | 0.286 | -1.8, 0.321, 0.49, [-5.0, +1.2] |
| top_k0.3 − dare_k0.3 | -25.4 | 141 (7 / 134) | 1.43e-31 | 1e-30 | [-29.4, -21.0] | 0.000 | -25.4, 1.43e-31, 1e-30, [-29.4, -21.0] |
| top_k0.3 − rndE_k0.3 | +1.6 | 56 (32 / 24) | 0.35 | 0.49 | [-1.2, +4.6] | 0.248 | +1.6, 0.35, 0.49, [-1.2, +4.6] |
| top_k0.3 − rndT_k0.3 | +3.4 | 67 (42 / 25) | 0.0498 | 0.107 | [+0.4, +6.6] | 0.242 | +3.4, 0.0498, 0.107, [+0.4, +6.6] |
| top_k1 − dare_k1 | -24.0 | 148 (14 / 134) | 9.19e-26 | 3.22e-25 | [-28.0, -19.6] | 0.000 | -24.0, 9.19e-26, 3.22e-25, [-28.0, -19.6] |
| top_k1 − rndE_k1 | +2.6 | 91 (52 / 39) | 0.208 | 0.364 | [-1.0, +6.4] | 0.158 | +2.6, 0.208, 0.364, [-1.0, +6.4] |
| top_k1 − rndT_k1 | +1.8 | 97 (53 / 44) | 0.417 | 0.53 | [-2.0, +5.8] | 0.150 | +1.8, 0.417, 0.53, [-2.0, +5.8] |
| top_k3 − dare_k3 | -22.4 | 150 (19 / 131) | 9.06e-22 | 2.54e-21 | [-26.6, -17.8] | 0.000 | -22.4, 9.06e-22, 2.54e-21, [-26.6, -17.8] |
| top_k3 − rndE_k3 | +0.0 | 100 (50 / 50) | 1 | 1 | [-3.8, +3.6] | 0.070 | +0.0, 1, 1, [-3.8, +3.6] |
| top_k3 − rndT_k3 | +1.0 | 97 (51 / 46) | 0.685 | 0.738 | [-2.8, +4.8] | 0.066 | +1.0, 0.685, 0.738, [-2.8, +4.8] |
| mlp − attn | +3.4 | 69 (43 / 26) | 0.0533 | 0.107 | [+0.0, +6.6] | 0.000 | +3.4, 0.0533, 0.107, [+0.0, +6.6] |
| top_k1 − comp_k1 | -29.8 | 161 (6 / 155) | 1.57e-38 | 2.19e-37 | [-33.8, -25.4] | 0.000 | -29.8, 1.57e-38, 2.19e-37, [-33.8, -25.4] |

- **Hypothesis verdicts.** H_local REFUTED (k_rec NONE). H_diffuse SUPPORTED (smallest k 0.1 %): DW_k holds at all four k, DF_k at
  none. H_attention REFUTED (BOTH). H_artifact INDETERMINATE (J_floor(1 %) 0.642, J_round(1 %) 0.534; at every k the rule gives
  INDETERMINATE: J_floor 0.656, 0.632, 0.642, 0.639 and J_round 0.540, 0.566, 0.534, 0.525). J_sp(1 %) of the real change 0.008, of P+
  0.754. PC-ART valid.
- **Necessity line.** NOT NECESSARY at k 0.1, 0.3, 1 and 3 %: comp_k 90.4, 90.8, 91.6, 90.6 are REC and ≥ rl − δ = 86.56.
- **GSM8K line at 1 %** (G-GATE-GSM passed): top − dare −23.73 points [−26.54, −21.08], MATH-500 −24.0: REPLICATES. top − rndE +2.12
  [+0.23, +3.94], MATH-500 +2.6: REPLICATES. top − rndT +1.90 [+0.08, +3.64], MATH-500 +1.8: REPLICATES. On MATH-500 the rndE and rndT
  differences at 1 % are not significant (BH p 0.364 and 0.530); the frozen GSM8K rule reads only the sign and the interval.
- **Against the best control** (§7): at every k the best control is dare_k; top − best = −27.6, −25.4, −24.0, −22.4.

## Step 4. Checks

Script `read_n13/checks_read.py`, output `checks_read.json`. Each verdict variant below is `verdict_read.run` on modified score
vectors, with thresholds recomputed from those vectors.

### (a) Concordance under shared seeds

Discordant counts and identical-text shares for the 14 primary comparisons are in the family table above. No McNemar p rests on fewer
than 20 discordant problems (the smallest count is 56, top_0.3 % against rndE_0.3 %). Arms near base share many token-identical
answers: top_k against rndE_k or rndT_k 6.6 % (k 3 %) to 29.4 % (k 0.1 %) of problems; base against top_0.1 % 29.8 %, base against
rndE_1 % 28.6 %, rndT_1 % against its second draw 30.0 %. Arms far from base share none: top against dare, top_1 % against comp_1 %,
`mlp` against `attn`, dare_1 % against `mlp`: 0 identical answers. Same weights under the two seed sets share almost none (base-A
against zero-B 2 of 500; top_1 % against top_1 %_B 1; rl against id 0), so the identical texts come from the shared seeds. An identical
text always has an identical score (0 exceptions over all compared pairs). Effect on the verdict: the concordant problems make the
paired McNemar test more sensitive than an unpaired comparison, while δ is the band for two independent seed readings, so BEATS's δ
condition is conservative for shared-seed pairs. No decisive comparison depends on this: the dare comparisons have 141 to 164
discordant problems, 7 to 19 of them in top's favour.

### (b) Cap hits and lengths

Cap C = 16,384 tokens. Cap-hit shares: base-like arms (top, rndE, rndT, rnd0, topF, base, zero) 12.2 % to 16.6 %; dare 8.6 % to 10.4 %;
`mlp` 8.4 %; `attn` 12.0 %; comp 4.8 % to 5.6 %; rl 3.2 %, id 3.4 %. Mean new tokens: base-like arms 3,133 to 4,004 (medians 716 to
1,174); RL-like arms (rl, id, dare, comp, `mlp`, `attn`) 4,333 to 4,671 (medians 1,951 to 2,956).
Accuracy among non-capped answers (MATH-500): base 66.4, top_1 % 73.4, top_3 % 76.3, rndE_1 % 68.4, rndT_1 % 69.7, dare_1 % 94.2,
`mlp` 93.9, `attn` 93.6, comp_1 % 95.0, rl 94.0 (all arms in the format table below and in `checks_read.json` → `b_caps`). A cap-hit answer is almost always scored wrong (0 to 7 right per arm, mode `answer_line`
or `boxed` from an earlier line).

- **Cap-hit answers scored wrong in every arm** (thresholds recomputed: B 57.2, L 91.0, H 74.1, Q 65.65): no label, k_rec, hypothesis,
  necessity or GSM8K label changes.
- **Cap-hit problems dropped** (the union over the 31 MATH-500 arms leaves 241 problems; over the 11 GSM8K arms, 1,105): thresholds
  B 73.0, L 100.0, H 86.5, Q 79.8, δ 7.92. top_3 % moves from NONE to PART (80.1); k_rec stays NONE; every hypothesis verdict and the
  necessity line are unchanged. The GSM8K labels top − rndE and top − rndT move from REPLICATES to CONSISTENT.
- **Pairwise drop** (problems capped in either arm): top − dare −23.9, −22.9, −19.5, −19.2; top − rndE +1.5, +2.2, +4.3, +1.0; top −
  rndT −0.2, +4.0, +3.6, +3.5; top_1 % − comp_1 % −22.3. `mlp` − `attn` goes from +3.4 to 0.0 (n 423): that difference lies entirely in
  cap-hit problems.

### (c) Format

Extraction modes per arm are in the accuracy table. Base-like arms (base, zero, top, rndE, rndT, rnd0, topF) write an `Answer:`
label line in 38.6 % to 45.8 % of answers, a box only in 28.4 % to 33.4 %, and neither in 23.2 % to 29.4 %. RL-like arms box: rl 494
boxed and 0 label lines; comp 0 or 1 label line; dare 4 to 9; `mlp` 7; `attn` 22. The prompt asks for an `Answer:` line, so RL's
boxing habit is itself part of what RL changed, and the arms that recover the gain also adopt it.

| arm | acc (%) | acc among non-capped (%) | acc among extracted (%) | acc in mode answer_line (%) | acc in mode boxed (%) | non-capped answers with no extraction | wrong answers holding the gold elsewhere: label anywhere / any box / last line | lenient acc (%) |
|---|---|---|---|---|---|---|---|---|
| base | 57.2 | 66.4 | 78.8 | 84.5 | 69.9 | 69 | 22 / 0 / 24 | 61.6 |
| rl | 92.0 | 94.0 | 93.1 | — | 93.1 | 0 | 0 / 2 / 1 | 92.4 |
| zero | 55.4 | 63.2 | 78.2 | 84.5 | 70.8 | 83 | 26 / 1 / 28 | 60.6 |
| id | 91.2 | 93.6 | 92.9 | — | 92.9 | 0 | 0 / 2 / 1 | 91.6 |
| top_k0.1 | 57.6 | 67.8 | 80.2 | 83.7 | 75.0 | 67 | 25 / 1 / 25 | 62.6 |
| top_k0.3 | 61.4 | 70.7 | 81.4 | 86.7 | 73.7 | 58 | 21 / 1 / 23 | 65.6 |
| top_k1 | 61.8 | 73.4 | 84.9 | 88.3 | 80.1 | 59 | 28 / 0 / 23 | 67.4 |
| top_k3 | 63.8 | 76.3 | 87.6 | 91.0 | 82.4 | 55 | 26 / 0 / 27 | 69.0 |
| dare_k0.1 | 85.2 | 93.8 | 93.8 | 100.0 | 93.7 | 1 | 6 / 1 / 3 | 86.6 |
| dare_k0.3 | 86.8 | 95.3 | 94.3 | 100.0 | 94.2 | 0 | 4 / 1 / 0 | 87.8 |
| dare_k1 | 85.8 | 94.2 | 93.5 | 83.3 | 93.6 | 0 | 3 / 1 / 2 | 86.6 |
| dare_k3 | 86.2 | 94.1 | 93.9 | 100.0 | 93.8 | 0 | 4 / 1 / 1 | 87.0 |
| rndE_k0.1 | 58.8 | 67.1 | 78.2 | 86.4 | 65.5 | 62 | 17 / 0 / 25 | 62.2 |
| rndE_k0.3 | 59.8 | 68.8 | 78.9 | 85.1 | 69.5 | 57 | 17 / 0 / 23 | 63.2 |
| rndE_k1 | 59.2 | 68.4 | 78.7 | 85.4 | 70.1 | 59 | 22 / 2 / 28 | 63.8 |
| rndE_k3 | 63.8 | 72.9 | 83.9 | 86.9 | 80.2 | 57 | 23 / 0 / 28 | 68.4 |
| rndT_k0.1 | 59.4 | 67.7 | 79.2 | 86.5 | 67.8 | 65 | 24 / 1 / 22 | 64.4 |
| rndT_k0.3 | 58.0 | 67.1 | 78.0 | 84.0 | 69.8 | 60 | 21 / 1 / 22 | 62.4 |
| rndT_k1 | 60.0 | 69.7 | 81.5 | 87.1 | 72.7 | 62 | 22 / 2 / 24 | 64.6 |
| rndT_k3 | 62.8 | 71.6 | 81.8 | 85.8 | 76.1 | 55 | 22 / 2 / 25 | 67.4 |
| comp_k0.1 | 90.4 | 94.3 | 92.8 | — | 92.8 | 0 | 0 / 1 / 0 | 90.6 |
| comp_k0.3 | 90.8 | 94.1 | 92.8 | 100.0 | 92.8 | 0 | 0 / 2 / 0 | 91.2 |
| comp_k1 | 91.6 | 95.0 | 93.7 | — | 93.7 | 0 | 2 / 2 / 1 | 92.4 |
| comp_k3 | 90.6 | 94.5 | 93.8 | — | 93.8 | 0 | 3 / 0 / 0 | 91.2 |
| mlp | 86.4 | 93.9 | 92.7 | 100.0 | 92.6 | 0 | 8 / 0 / 4 | 88.0 |
| attn | 83.0 | 93.6 | 93.7 | 86.4 | 94.1 | 2 | 8 / 2 / 3 | 84.8 |
| rnd0_k1 | 59.6 | 70.0 | 81.6 | 87.3 | 73.7 | 61 | 18 / 0 / 19 | 63.2 |
| top_k1_B | 60.0 | 70.5 | 85.0 | 87.4 | 81.5 | 72 | 30 / 2 / 32 | 66.2 |
| dare_k1_d2 | 87.8 | 95.8 | 94.4 | 100.0 | 94.4 | 0 | 6 / 1 / 3 | 89.2 |
| rndT_k1_d2 | 59.6 | 69.1 | 81.0 | 85.5 | 74.1 | 63 | 24 / 1 / 23 | 64.4 |
| topF_k1 | 64.0 | 73.7 | 84.7 | 88.7 | 79.0 | 60 | 26 / 1 / 30 | 69.2 |

"Wrong answers holding the gold elsewhere" counts wrong answers where the gold-equivalent string (by `math_verify`) sits after an
"answer:" or "answer is" anywhere in a line (for example `**Answer**: 21`, `The final answer is 83.`, `…).Answer: \((a + 5)(b + 2)\)`),
in any box, or on the last non-empty line. The rule (§2.4 as amended) reads none of these.

- **Format does not set the accuracy gap.** Within the same extraction mode, RL-like arms are more accurate: boxed answers 92.6 % to
  94.4 % for rl, id, comp, dare, `mlp`, `attn`, against 65.5 % to 82.4 % for base-like arms. Among answers with any extracted string:
  base 78.8, top_1 % 84.9, rndE_1 % 78.7, rndT_1 % 81.5, rl 93.1, dare_1 % 93.5.
- **The rule costs base-like arms more answers.** Base-like arms finish 55 to 83 non-capped answers with nothing the rule reads
  (RL-like arms 0 to 2). A lenient re-score (label anywhere or any box) turns 17 to 31 wrong answers right per base-like arm (12 to 23
  of them non-capped), against 1 to 9 per RL-like arm (0 to 3 non-capped).
- **The verdict under the lenient re-score** (thresholds recomputed: B 61.6, L 92.4, H 77.0, Q 69.3, δ 5.29): no label or hypothesis
  changes. top_3 % 69.0 and topF_1 % 69.2 sit just under Q. Adding the last-line candidate (H 77.6, Q 70.1) moves topF_1 % to PART
  (71.0); no hypothesis changes. The GSM8K labels are unchanged under the lenient re-score.

### (d) The controls' matching (realized MLP change energy, `build.json` and `sets.json`)

Full MLP change energy E_total = 11.182 (`id` build; `sets.json` 11.182).

| k | top_k: E(S_k), share of E_total | rndE_k: E / E(S_k) (c₀ → c built) | rndT_k: m_k, E / E(S_k) | dare_k: kept / expected (sd) | dare_k: 1/q | dare_k: E / E_total | dare_k: E / E(S_k) | dare_k: realized / intended |
|---|---|---|---|---|---|---|---|---|
| 0.1 % | 0.02414, 0.22 % | 1.0044 (1.4735 → 1.4276) | 765, 1.0011 | 4,325,381 / 4,325,376 (+0.00) | 1,000.7 | 1,000.1 | 463,347 | 0.9998 |
| 0.3 % | 0.05888, 0.53 % | 1.0004 (1.3253 → 1.2780) | 1,861, 1.0003 | 12,986,283 / 12,988,416 (−0.59) | 333.3 | 333.2 | 63,271 | 1.0002 |
| 1 % | 0.16472, 1.47 % | 0.9951 (1.2175 → 1.1752) | 5,221, 1.0001 | 43,288,867 / 43,290,624 (−0.27) | 100.0 | 99.8 | 6,774 | 0.9974 |
| 3 % | 0.43806, 3.92 % | 0.9919 (1.1478 → 1.1134) | 13,910, 1.0000 | 129,851,265 / 129,859,584 (−0.74) | 33.3 | 33.1 | 845 | 0.9935 |

rndE is within 5 % of the top set's energy at every k (0.8 % at most). rndT is never short. The DARE kept counts lie within 0.74 binomial
standard deviations (tolerance 4, A4). Second draws: dare_1 %_d2 kept 43,290,732 (+0.02 sd), energy 99.7 × E_total; rndT_1 %_d2 5,203
neurons, 1.0001 × E(S_1%). rnd0_1 % (natural scale) 0.677 × E(S_1%); topF_1 % 0.987 × E(S_1%). DARE's energy is not matched to the top
set's or to the full change's; that follows from its 1/q rescale and is as registered (§3.2).

### (e) Second draws and the seed-B top arm, against δ = 5.44

top_1 % − top_1 %_B = +1.8; rndT_1 %_d2 − rndT_1 % = −0.4; dare_1 %_d2 − dare_1 % = +2.0. Each is within δ. σ_mid (top_1 % A against B)
= 1.892 = 0.98 σ, under the 1.5 σ flag.

### (f) The base's own answers

| arm | base right, arm wrong | base wrong, arm right |
|---|---|---|
| top_1 % | 38 | 61 |
| rndE_1 % | 32 | 42 |
| rndT_1 % | 29 | 43 |
| dare_1 % | 16 | 159 |
| `attn` | 18 | 147 |
| `mlp` | 13 | 159 |
| comp_1 % | 6 | 178 |
| rl | 6 | 180 |
| top_3 % | 42 | 75 |

### (g) Leverage

- **Bootstrap seed.** Over seeds 1 to 100, no hypothesis verdict, label or necessity line changes. Only the GSM8K label top − rndT moves,
  to CONSISTENT, under 18 of 100 seeds (its lower bound ranges from −0.002 to +0.152 over seeds 0 to 100). top − rndE stays REPLICATES
  under all 101 seeds (lower bound +0.15 to +0.38). top − dare stays REPLICATES.
- **BH at q 0.10.** No change. The rejected tests are the four top − dare and top_1 % − comp_1 %; top_0.3 % − rndT_0.3 % and `mlp` −
  `attn` have BH p 0.107.
- **Newcombe interval from the two Wilson intervals, in place of the bootstrap.** The hypothesis verdicts do not use the interval and
  do not change. The GSM8K labels top − rndE and top − rndT move to CONSISTENT ([−1.52, +5.75] and [−1.74, +5.53]); top − dare stays
  REPLICATES. The exploratory GSM8K McNemar p values are 0.030 (rndE) and 0.050 (rndT).

### (h) H_local at every k, against each control and the best control

| k | top_k | dare_k (top − dare) | rndE_k (top − rndE) | rndT_k (top − rndT) | a control within δ? | top beats all three? | if this k were k_rec |
|---|---|---|---|---|---|---|---|
| 0.1 % | 57.6 NONE | 85.2 REC (−27.6) | 58.8 NONE (−1.2) | 59.4 NONE (−1.8) | all three | no | REFUTED (a control within δ) |
| 0.3 % | 61.4 NONE | 86.8 REC (−25.4) | 59.8 NONE (+1.6) | 58.0 NONE (+3.4, BH p 0.107) | all three | no | REFUTED (a control within δ) |
| 1 % | 61.8 NONE | 85.8 REC (−24.0) | 59.2 NONE (+2.6) | 60.0 NONE (+1.8) | all three | no | REFUTED (a control within δ) |
| 3 % | 63.8 NONE | 86.2 REC (−22.4) | 63.8 NONE (0.0) | 62.8 NONE (+1.0) | all three | no | REFUTED (a control within δ) |

The best control is dare_k at every k. dare_k beats top_k by more than δ with BH p ≤ 2.6e-21 at every k (an exploratory reading of the
reverse comparison). So H_local would be REFUTED at every k even if top_k had reached H.

### (i) GSM8K per-arm accuracies (n = 1,319)

| arm | GSM8K right / 1,319 | acc (%) | Wilson 95 % | acc, re-graded (%) | cap hits | mean new tokens |
|---|---|---|---|---|---|---|
| attn | 1169 | 88.63 | [86.80, 90.23] | 88.63 | 64 | 1571 |
| base | 839 | 63.61 | [60.98, 66.16] | 63.61 | 33 | 702 |
| comp_k1 | 1219 | 92.42 | [90.86, 93.73] | 92.42 | 25 | 2002 |
| dare_k1 | 1184 | 89.76 | [88.01, 91.29] | 89.76 | 60 | 1746 |
| id | 1214 | 92.04 | [90.45, 93.38] | 92.04 | 21 | 2009 |
| mlp | 1182 | 89.61 | [87.85, 91.15] | 89.61 | 73 | 1883 |
| rl | 1218 | 92.34 | [90.78, 93.66] | 92.34 | 24 | 1986 |
| rndE_k1 | 843 | 63.91 | [61.28, 66.46] | 63.91 | 43 | 837 |
| rndT_k1 | 846 | 64.14 | [61.51, 66.68] | 64.14 | 45 | 861 |
| top_k1 | 871 | 66.03 | [63.44, 68.54] | 66.03 | 43 | 857 |
| zero | 847 | 64.22 | [61.59, 66.76] | 64.22 | 23 | 571 |

top_1 % is 2.4 points above base on GSM8K (66.03 against 63.61); rndE_1 % and rndT_1 % are within 0.6 points of base.

### (j) Other findings

1. **DARE equals `mlp`.** dare_k − `mlp` = −1.2, +0.4, −0.6, −0.2 (k 0.1 to 3 %; McNemar p 0.50, 0.89, 0.79, 1.00); dare_1 % against
   `mlp` has 55 discordant problems and 0 identical texts. The DARE arms reproduce the effect of the whole MLP change while carrying
   1/q times its energy. That is the published DARE property (the rescaled sparse sample has the full change as its expectation); I did not measure
   activations, so the mechanism is not shown here.
2. **The top set adds little beyond random at matched energy, but never less.** top − rndE and top − rndT are ≥ 0 at k ≥ 0.3 % (+1.6,
   +3.4, +2.6, +1.8, 0.0, +1.0), none significant after BH; among extracted answers top_1 % reads 84.9 against rndE_1 % 78.7. The size of
   any such effect (a few points) is far below δ and far below the 17.4 points needed to reach H. Exploratory.
3. **comp_k at RL's level.** Removing the top 0.1 % to 3 % of neurons' change from RL leaves 90.4 to 91.6 (rl 92.0, id 91.2).
4. **`mlp` and `attn` each recover most of the gain** (86.4 and 83.0 against L 92.0). `mlp` is 5.6 points under rl, just more than δ.
5. **Every generated text starts with "."** and blank lines: the 3.1 template is a plain prompt with no role tokens, and the model's
   first token closes the prompt's last sentence. This is the registered template (§2.2) in every arm.
6. **Figure.** §10.3 asks that each rndT point be annotated with m_k; `fig_n13.png` carries no m_k labels. At k 3 % the direct labels
   "rndT", "top", "rndE" overlap. The plotted values match the verdict.

## Step 5. The registered verdict against my recount

- `recount_n13.py --banks <my downloaded banks> --verdict <the downloaded verdict>`: 31 lines "ok", then `RECOUNT_OK`
  (`read_n13/recount_n13_output.txt`, run under `-I` through `run_recount_n13.py`). It recounts MATH-500 accuracies only.
- `read_n13/compare_verdict.py`: 465 fields compared (every arm's accuracy, n, label, Wilson interval, cap-hit share, mean tokens,
  answer-line share, status; every comparison's difference, McNemar p, BH p, bootstrap interval, n_paired; k_rec and every hypothesis;
  the H_artifact values; the three GSM8K differences, intervals and labels; the thresholds; the three numbers; every gate's pass, value,
  failing-input value and `n_empty`; verdict, stamps, parked). 465 agree to 1e-9 relative.
- Script hashes equal the registry's: `verdict_n13.py` `f4ca90174df2`, `recount_n13.py` `c680fd2ce5d8`, `figure_n13.py`
  `a7092c877735`, `samples_coverage.py` `bb57a434bf90`. `infra/pg/n13_*.py` and `studies/next/n13/scripts/` do not differ between
  `n13-freeze-2a` and `n13-s4-complete`.
- **CLAIMS.md** (branch `n13`): every number in N13-C1 to C13 equals my recount. Checked: C2 Wilson intervals (base [52.8, 61.5], rl
  [89.3, 94.1]); C5 the eight top − rndE and top − rndT differences and the smallest BH p 0.107; C6 dare 85.2, 86.8, 85.8, 86.2,
  differences −27.6 to −22.4, the largest BH p 2.54e-21 (written "≤ 2.6e-21"); C7, C8 (BH p 2.19e-37); C9; C10 intervals; C11; C12
  (A31 1.2e-16 to 1.0e-15; rndT 1.0000 to 1.0011); C13 (answer-line shares 0.41 to 0.46 for the 16 NONE arms, 0.00 to 0.044 for the 11
  REC arms; mean tokens 3,133 to 4,004 against 4,428 to 4,671). Its two open questions are answered by checks (d), (j) and (a).
- **REGISTRY.md**: the input and output hashes it lists equal what I downloaded or read (manifest `56c87b3c06c4`, verdict
  `cae55c04ebf8`, thresholds `60bf6d4eea9c`, controls `f98c30e4c3d2`, R0 `ee359fc6c5f3`, grader `151384d2b8eb`, template
  `ad003c9c9c1d`, sets `49a1065c5705`, figure PNG `6bf9a4ff67ad`).
- **SAMPLES_COVERAGE.md**: 42 of 42 arm × benchmark samples present, 3 to 5 rows each; matches the banks.
- **fig_n13.png**: `git show` returns a Git LFS pointer (oid `6bf9a4ff67ad…`, 99,025 bytes). I resolved it with `git lfs smudge`
  into `<SP>/n13read/fig/fig_n13_real.png`, whose sha256 equals the oid, and opened it. Lines and points match the verdict (top 57.6 to
  63.8; comp 90.4 to 91.6; dare 85.2 to 86.8; rndE, rndT 58.0 to 63.8; `mlp` 86.4; `attn` 83.0; base and rl bands; seed-B lines; H
  dotted at 74.6). See (j) 6 for the two figure defects.

## Step 6. Raw examples (MATH-500 `samples.jsonl`)

Script `read_n13/samples_read.py`, output `samples_read.json`. Every sample row equals its `generations.jsonl` record on every shared
field, and every banked score equals my re-grade. Each file holds problems 0 to 2, the lowest right and lowest wrong, and a cap hit.

Problem first lines: 0 "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates. Enter your answer in the form
$(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$"; 1 "Define" (then the sums p and q); 2 "If $f(x) = \frac{3x-2}{x-2}$, what is
the value of $f(-2) +f(-1)+f(0)$? Express your answer as a common fraction."; 4 "The results of a cross-country team's training run are
graphed below. Which student has the greatest average speed?"; 9 "The expression $2\cdot 3 \cdot 4\cdot 5+1$ is equal to 121, since
multiplication is carried out before addition. …"; 10 "What is the least positive integer multiple of 30 that can be written with only
the digits 0 and 2?"; 11 "Let $p(x)$ be a polynomial of degree 5 such that".

| arm | problem (role) | mode | extracted | gold | score | my re-grade |
|---|---|---|---|---|---|---|
| base | 0 (fixed, lowest right) | answer_line | `(3, π/2)` | `\left( 3, \frac{\pi}{2} \right)` | 1 | 1 |
| base | 1 (fixed) | boxed | `p - q` | `p - q` | 1 | 1 |
| base | 2 (fixed, lowest wrong) | boxed | `-1` | `\frac{14}{3}` | 0 | 0 |
| base | 9 (cap hit) | none | — | `4` | 0 | 0 |
| rl | 0 (fixed, lowest right) | boxed | `(3, \frac{\pi}{2})` | as above | 1 | 1 |
| rl | 1 (fixed) | boxed | `p - q` | `p - q` | 1 | 1 |
| rl | 2 (fixed) | boxed | `\frac{14}{3}` | `\frac{14}{3}` | 1 | 1 |
| rl | 9 (lowest wrong, cap hit) | none | — | `4` | 0 | 0 |
| top_1 % | 0 (fixed, lowest right) | answer_line | `$\boxed{\left(3, \dfrac{\pi}{2}\right)}$` | as above | 1 | 1 |
| top_1 % | 1 (fixed) | boxed | `p - q` | `p - q` | 1 | 1 |
| top_1 % | 2 (fixed, lowest wrong) | boxed | `-1` | `\frac{14}{3}` | 0 | 0 |
| top_1 % | 10 (cap hit) | none | — | `2220` | 0 | 0 |
| dare_1 % | 0 (fixed, lowest right) | boxed | `\left(3, \frac{\pi}{2}\right)` | as above | 1 | 1 |
| dare_1 % | 1 (fixed) | boxed | `p - q` | `p - q` | 1 | 1 |
| dare_1 % | 2 (fixed) | boxed | `\dfrac{14}{3}` | `\frac{14}{3}` | 1 | 1 |
| dare_1 % | 4 (lowest wrong, cap hit) | none | — | `\text{Evelyn}` | 0 | 0 |
| rndE_1 % | 0 (fixed, lowest right) | answer_line | `(3, π/2)` | as above | 1 | 1 |
| rndE_1 % | 1 (fixed) | boxed | `p - q` | `p - q` | 1 | 1 |
| rndE_1 % | 2 (fixed, lowest wrong) | boxed | `-1` | `\frac{14}{3}` | 0 | 0 |
| rndE_1 % | 11 (cap hit) | none | — | `\frac{3}{56}` | 0 | 0 |
| `attn` | 0 (fixed, lowest right) | boxed | `\left(3, \frac{\pi}{2}\right)` | as above | 1 | 1 |
| `attn` | 1 (fixed) | boxed | `p - q` | `p - q` | 1 | 1 |
| `attn` | 2 (fixed) | boxed | `\dfrac{14}{3}` | `\frac{14}{3}` | 1 | 1 |
| `attn` | 9 (lowest wrong, cap hit) | none | — | `4` | 0 | 0 |

Under the shared seeds, base, top_1 % and rndE_1 % write the identical 617-token text on problem 1, and top_1 % and rndE_1 % write
the identical 246-token text on problem 2 (answer probabilities 0.9957 and 0.9952; base writes a different wrong answer there). The
RL-like arms solve problem 2.

**One right answer in full**: base, problem 1, seed 1,000,001, 617 new tokens, finish `stop`, mode `boxed`, extracted `p - q`, gold
`p - q`, score 1, answer probability 0.989.

~~~~text
.

To solve the problem, we need to find a way to express the double sum
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\]
in terms of the given sums \(p\) and \(q\), where
\[p = \sum_{k = 1}^\infty \frac{1}{k^2}\]
and
\[q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]

First, let's understand the structure of the double sum:
\[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}.\]
This sum is over all pairs \((j, k)\) where \(j\) and \(k\) are both positive integers.

We can try to re-index the sum by introducing a new variable \(n = j + k\). For each fixed \(n\), \(j\) can range from 1 to \(n - 1\) (since \(k\) must also be at least 1). Thus, the double sum can be rewritten as:
\[\sum_{n = 2}^\infty \sum_{j = 1}^{n - 1} \frac{1}{n^3}.\]
This simplifies to:
\[\sum_{n = 2}^\infty \frac{n - 1}{n^3}.\]
We can split this sum into two parts:
\[\sum_{n = 2}^\infty \left( \frac{1}{n^2} - \frac{1}{n^3} \right).\]
This can be further split into two separate sums:
\[\sum_{n = 2}^\infty \frac{1}{n^2} - \sum_{n = 2}^\infty \frac{1}{n^3}.\]
Recognizing that these sums are related to the given sums \(p\) and \(q\), we have:
\[\sum_{n = 2}^\infty \frac{1}{n^2} = p - \frac{1}{1^2} = p - 1,\]
and
\[\sum_{n = 2}^\infty \frac{1}{n^3} = q - \frac{1}{1^3} = q - 1.\]
Thus, the double sum can be written as:
\[\sum_{n = 2}^\infty \left( \frac{1}{n^2} - \frac{1}{n^3} \right) = (p - 1) - (q - 1) = p - q.\]
Therefore, the answer is:
\[\boxed{p - q}.\]
~~~~

**One wrong answer in full**: top_1 %, problem 2, seed 1,000,002, 246 new tokens, finish `stop`, mode `boxed`, extracted `-1`, gold
`\frac{14}{3}`, score 0, answer probability 0.996. The model writes Python and then an output block it never ran; the true sum is
14/3.

~~~~text
.

To find the value of \(f(-2) + f(-1) + f(0)\) for the function \(f(x) = \frac{3x-2}{x-2}\), we need to evaluate each term individually and then sum the results.

Let's break down the problem:

1. Evaluate \(f(-2)\).
2. Evaluate \(f(-1)\).
3. Evaluate \(f(0)\).
4. Sum the results of the evaluations.

We'll use Python to ensure the accuracy of the calculations.
```python
# Define the function f(x)
def f(x):
    return (3 * x - 2) / (x - 2)

# Evaluate f(-2), f(-1), and f(0)
f_minus_2 = f(-2)
f_minus_1 = f(-1)
f_0 = f(0)

# Sum the results
result = f_minus_2 + f_minus_1 + f_0
print(result)
```
```output
-1.0
```
The value of \(f(-2) + f(-1) + f(0)\) is \(\boxed{-1}\).
~~~~

## What I could not verify

- **Weights.** No weights are local. G-HASH and G-MASK rest on the sha256 values and mask lists that the builder recorded in
  `build.json`; I checked them for consistency (each family against base and RL, realized against expected neurons) but did not
  re-hash any tensor. The R1 numbers (G-INIT shares, G-PARTITION, PC-RANK, PC-ART, J_floor, J_round, J_sp, E_total, the sets) are read
  from `controls.json` and `sets.json`, not recomputed.
- **G-KNOWN.** 39.96 and the failing input's 1.12 are read from the R0 record; the olmes predictions are not in the banks.
- **Text.** The generated text was decoded on the instance; the generated token ids are not banked (only the prompt's first and last 20
  ids), so I could not re-decode.
- **Golds.** The MATH-500 and GSM8K golds were not compared with the Hugging Face files (no download); they are identical across all
  arms and the MATH-500 golds pass G-GRADER's self-equality.
- **Engine libraries.** torch, triton, transformers and tokenizers versions come from the executor's `engine_check.json` files; the
  banks' `metrics.json` carry only the vLLM version. The pip-freeze differences outside those libraries (click, jiter, typer, regex,
  cbor2, cachetools, sse-starlette; `STATE.md`) were not re-checked.
- **Mechanism.** Why DARE at 33 to 1,000 times the change energy reproduces the full MLP change is not measured here (no activations).

## Files

Under `studies/next/n13/read_n13/`: `verify_manifest.py` → `manifest_check.json`; `recount_read.py` → `recount_read.json`;
`gates_read.py` → `gates_read.json`; `verdict_read.py` → `verdict_read.json`; `checks_read.py` → `checks_read.json`;
`run_recount_n13.py` → `recount_n13_output.txt`; `compare_verdict.py` → `compare_verdict.json`; `samples_read.py` → `samples_read.json`.
No generations file is committed. Downloads in `<SP>/n13read/s3/` (`<SP>` = this session's scratchpad).
