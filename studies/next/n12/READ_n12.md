# N12: the independent read

Reader: an Opus agent, worktree `interp_research-n12read`, branch `n12-read` from `bio` at `a865eddc5`,
2026-11-06. No GPU, no model run. S3 read-only (profile `fuelos`, `us-east-1`). Paths below are relative to
`studies/next/n12/` unless they start with `infra/` or `studies/`.

## 1. Answer

**The headline reproduces exactly**: `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal DOES NOT COUNT;
H_bridge DOES NOT COUNT; H_dt SUPPORTED` (`read/r3_summary.json` → `labels.headline`, equal to
`verdict_n12.json` → `headline`). The number that settles H_cf is the mean F over the 71 real test units:
0.9507 [0.736, 1.183] (reader, seed 7) against 0.9507 [0.738, 1.180] (job, seed 0). Of 90 compared fields (45
per half), every point value agrees within 4.4e-16 except two: the Spearman(S, gain) means (difference 1.7e-4) and the
matched floor's 2.5th percentile (difference 0.0014, a random-replicate quantity) (`read/r8_compare.json`).

Three findings change how the labels read:

1. **OWN_RISE is what every dated pole shows, real or twin.** The distinct-token twin's mean F is 0.942
   [0.676, 1.216]; paired real − dt is +0.067 [−0.210, +0.333]. In 442 of 442 units of all kinds B rises
   between T_(k−1) and T_k, so r_c ≤ r0 in every unit. That follows from the dating rule. F > 0 then needs only
   that the others' move alone does not lift the pole (r_o ≥ r0), true in 77–91 % of units by kind
   (`read/r4_unanticipated.json`).
2. **G-VAL could not pass with this ablation.** Its ρ is 0.0996 (freeze) and 0.0425 (test). The measured ΔL has a
   median |ΔL| of 0.65 nats against a median |ΔL̂| of 0.022 nats. |ΔL| does not grow with the size of the
   perturbation: Spearman with |ΔL̂| is 0.010, with Σ|h| −0.018. It grows with the number of loss terms after
   the ablated positions (Spearman 0.241). If ΔL were exactly ΔL̂ plus noise of the measured size, ρ would be
   0.054 [−0.023, 0.137]. The bar of 0.7 was out of reach (`read/r2_gval.json`).
3. **The smaller competitor drop at real births sits in the repeated-token births**, as M2's hit gap does.
   Real − pb drop, both halves: −0.387 [−0.553, −0.228] on repeated-token births and −0.047 [−0.217, +0.117]
   on varied-token births. Matching distinct tokens does not remove it all: real − dt, both halves, is −0.179
   [−0.306, −0.056] (the test half alone, which decides the label, is AT_SELECTION) (`read/r4_unanticipated.json`).

## 2. Blind order and what broke it

Phase 1 is commit `884902cf1`: scripts and JSON under `read/`. It was made before opening `verdict_n12.json`,
`thresholds.json` (any block), `infra/pg/n12.py`, `infra/aws_pg/jobs/n12_score.py`, the N12 `JOURNAL.md` or
spec §12. Phase 2 (the comparison) is `e92f40eed`. Phase 3 (the code) came after both.

What I saw before phase 1:

- the subject of commit `3b8f72a2f`, printed by my first `git log`. It names all five labels and the G-VAL ρ
  values (0.100 freeze, 0.043 test);
- the spec's §0 status table. It states the same headline and "G-VAL failed on the freeze half (ρ 0.0996)";
- the task brief. It names OWN_RISE, BELOW_SELECTION against pb and AT against dt.

So I knew the labels and the G-VAL ρ values before I computed anything. My code computes them from the arrays
and does not read them from anywhere.

Read before phase 1: the spec's §0–§11 (the brief allowed §0.5, §1.3, §2, §3, §5–§7; §0, §1, §4 and §8–§11
are rules and the runbook) and §14. The brief forbade "§14" but allowed "the amendments", and §14 is the
amendments section. It holds only A1, written before score-02, and contains no score-02 number. I also read the
reused M2 helpers in `infra/pg/m2.py` (`percentile`, `median_a_plus`, `ranks`, `matched`, `paired`, `wilson`,
`gates_ok`) for file formats and definitions. So my B check (`read/r6_bcheck.py`) is my own code, but written
after reading `m2.percentile`. The matched-floor rule for a matched pole inside C (§9 item 2) was inferred in
phase 1 from the records, which the brief allows. I did not read the N12 `JOURNAL.md` or spec §12 at any point.
I did not use any other agent's scratchpad directory.

## 3. What was recomputed, and from what

| quantity | source | reader code | agreement with the job |
|---|---|---|---|
| r0, r1, r_c, r_o, F, move, B_pre, B_birth, bar, C (ordered), C's B and ranks at both points, drop, u_gain, the identity readout | job arrays `arrays/tNN/<unit>_{pre,birth}.npz` → `B` (S3 score-02, synced) | `read/r1_recount.py` | 442 of 442 units equal, max diff 8.9e-16 (`read/r1_units.json` → `compare_with_records`) |
| S, u_S, u_P, u_P′, ρ(S, B_pre) | `<unit>_pull.npz` → `P`, `Pp` | same | 442 of 442 equal |
| P and P′ from G, Glm and q999 | `<unit>_pull.npz` → `G`, `Glm`, `valid`; grid `stats/layer_LL.npz` → `q999` (byte-range reads, `read/fetch_stats.py`) | same | max relative error 2.2e-16 over 442 units × 22,016 poles |
| ΔL̂ = Σ −h·G | `<unit>_pull.npz` → `H`, `G` | same | the record's `abl[i][1]` within 6.7e-8 nats, all 612 pairs (`read/r1_abl.json`) |
| matched poles and their u | grid `stats` → `max_abs` at T_(k−1) | same | 15,291 of 15,300 values equal; 9 differ by the rule in §9 item 2 |
| ρ(S, gain) | as above | same | 433 of 442 equal; 9 differ by ≤ 0.0135 (§9 item 6) |
| B from activations | job `a` + grid `stats` → `hist`, `hist_edges`, my own percentile code | `read/r6_bcheck.py` | 66 (unit, role) arrays over 6 (point, layer) pairs, all 22,016 poles: max diff 4.4e-16 |
| G-M2V and K0, every unit M2 banked | job `a` against S3 `results/next_m2/score-06/.../vectors/`; M2's `ranks_*.jsonl` | `read/r5_m2v_k0.py` | 455 files, 40,135,168 values, 0 unequal; B at the pole equal to M2's `value.B` (max diff 0); my r0/r1 equal M2's `rank.B` 455 of 455; 20 pb births at T_15 equal M2's pb end vectors bit for bit |
| the job's M2-format records | `ranks_{freeze,test}.jsonl` → `rank.B` | same | 1,024 of 1,024 equal to my r0, r1 and dt end rank |
| summaries, intervals, labels | the above | `read/r3_summary.py` | §4, §5 |
| G-VAL and its analysis | measured ΔL from the record (`abl[i][2]`), my ΔL̂ | `read/r2_gval.py` | §6 |
| gates | `banks/score-02/n12/metrics.json` | `read/r7_gates.py` | §5 |

The banked files are the S3 files: `n12_test.jsonl` has sha256 `877926da…` in both places, and `n12_freeze.jsonl`
has `30f31cb1…`, which is the frozen value. `metrics.json` has `53178f8d…` (frozen), `thresholds.json`
`85726798…` and `units.json` `c223dfee…` (registered).

## 4. Field by field against `verdict_n12.json` (test half)

The job's bootstrap seed is 0 and mine is 7, so the interval bounds differ in the third decimal. The freeze half
against `thresholds.json` → `discovery` gives the same picture: every point value agrees within 4.4e-16 except
the same two fields (`read/r8_compare.json`).

| field | job (`verdict_n12.json` → `test`) | reader (`read/r3_summary.json` → `test`) |
|---|---|---|
| n real / pb / dt | 71 / 70 / 67 | 71 / 70 / 67 |
| F real | 0.9507 [0.7375, 1.1801] | 0.9507 [0.7357, 1.1827] |
| move real (movement control) | 0.8535 [0.6450, 1.0737] | 0.8535 [0.6462, 1.0800] |
| log r0 − log r_c (own move alone) | 0.8788 [0.6666, 1.1064] | 0.8788 [0.6678, 1.1095] |
| log r0 − log r_o (others' move alone) | −0.0719 [−0.1167, −0.0289] | −0.0719 [−0.1160, −0.0310] |
| F pb / F dt | 0.7651 / 0.9422 | 0.7651 / 0.9422 |
| paired F real − pb | +0.1992 [+0.0037, +0.4056] | +0.1992 [−0.0016, +0.4087] |
| paired F real − dt | +0.0669 [−0.2015, +0.3370] | +0.0669 [−0.2100, +0.3327] |
| drop real / pb / dt | 0.9386 / 1.1639 / 1.0617 | 0.9386 / 1.1639 / 1.0617 |
| paired drop real − pb | −0.2257 [−0.4077, −0.0561] (28 / 42) | −0.2257 [−0.4020, −0.0553] (28 / 42) |
| paired drop real − dt | −0.1129 [−0.2947, +0.0653] (29 / 38) | −0.1129 [−0.2957, +0.0621] (29 / 38) |
| hits at T_(k−1), real / pb / dt | 42/71 0.592; 53/70 0.757; 41/67 0.612 | same counts, same Wilson bounds |
| hits at T_k; own move only; others' move only (real) | 0.915; 0.944; 0.549 | 0.915; 0.944; 0.549 |
| paired hit real − dt | −0.0299 [−0.1493, +0.0896] (7 / 9) | −0.0299 [−0.1493, +0.0896] (7 / 9) |
| paired hit dt − pb | −0.1343 [−0.2537, −0.0149] (4 / 13) | −0.1343 [−0.2537, −0.0149] (4 / 13) |
| paired hit real − pb (M2's number) | −0.1714 [−0.2857, −0.0571] | −0.1714 [−0.2857, −0.0571] |
| median bar real / pb / dt | 0.99606 / 0.98362 / 0.98970 | 0.99606 / 0.98362 / 0.98970 |
| dt control at T_15: dt pole; opposite pole | 66/67 0.985 [0.920, 0.997]; 0/67 | same |
| u_S, u_P, u_P′, u_gain (real) | 0.5268, 0.4965, 0.4563, 0.0627 | 0.5268, 0.4965, 0.4563, 0.0627 |
| matched floor q025 (test) | 0.4225 | 0.4239 (job's rule; seed 7) |
| ρ(S, gain) real | −0.01259 [−0.0718, +0.0459] | −0.01242 [−0.0706, +0.0483] |
| ρ(S, B_pre) real | 0.00894 | 0.00894 |
| paired u_S pb − real; dt − real | −0.0386; −0.0201 | −0.0386; −0.0201 |
| G-VAL ρ; lower bound; shuffle q975; slope | 0.0425; −0.0594; 0.1159; 0.3264 | 0.0425; −0.0601; 0.1274; 0.3264 |
| u_S resolution, freeze / test | 0 of 82 / 0 of 71; smallest gap 2.89e-6 | 0 / 0; smallest gap 2.89e-6 |

One reported quantity with no label sits on the edge of zero: the paired F real − pb has a lower bound of
+0.0037 under the job's seed and −0.0016 under mine.

## 5. Labels and gates

| label | spec rule (§5) | job | reader |
|---|---|---|---|
| movement control (both halves) | lower bound > 0 | passes | freeze 1.056 [0.830, 1.287], test 0.854 [0.646, 1.080] |
| H_cf | lower bound of mean F > 0 | OWN_RISE | OWN_RISE |
| H_away_sel_pb | paired drop real − pb | BELOW_SELECTION | BELOW_SELECTION |
| H_away_sel_dt | paired drop real − dt | AT_SELECTION | AT_SELECTION |
| G-VAL, both halves | ρ ≥ 0.7 and lower bound > shuffle q975 | fails | fails |
| H_signal, H_sel_signal_pb/dt, H_bridge | DOES NOT COUNT if G-VAL fails | DOES NOT COUNT | DOES NOT COUNT |
| the same, exploratory | the §5 rule applied anyway | not written (§9 item 3) | H_signal REFUTED (u_S 0.527 ≥ bar: 0.4225 by the job's floor, 0.4239 by mine); H_sel_signal_pb AT; H_sel_signal_dt AT; H_bridge UNDECIDED |
| H_sel_dt | paired hit real − dt | AT_SELECTION | AT_SELECTION |
| H_dt_pb | paired hit dt − pb | LOWER | LOWER |
| dt construction control | dt ≥ 0.90 and opposite < 0.90 | passes | 66/67 and 0/67 (test); 73/73 and 0/73 (freeze) |
| H_dt | AT or ABOVE, and LOWER | SUPPORTED | SUPPORTED |

Gates (`read/r7_gates.json`). All 73 instance gates pass, and every one of the 73 registered failing inputs fails.
The job is complete with nothing parked.

- K0: max |ΔB| 0 against a tolerance of 1e-6. Its failing input's smallest |ΔB| is 9.0e-4.
- G-M2V: p99 0, exact share 1.0 at all 11 points.
- I-ID: exact over 98 units, 0 top-10 flips. Failing input p99 3.38.
- I-GRAD: Spearman 0.9999999999999998, exact share 1.0, max |ΔG| 0, max |ΔS| 0, changed units 0 (freeze) and
  0 (test). Failing input ρ 0.0004.
- G-ABL-ID: max |clean − forward| 0.0 at 9 points (153 clean rows). Failing input's smallest gap 2.17 nats.
- G-TOPK: passes at every point.
- G-EX is gated at T_5, T_6 and T_15 and is exact there. It is reported at T_7–T_14 and exceeds the cross-device
  tolerance (p99 ≤ 0.0387) at T_13 (0.042), T_11 (0.046), T_10 (0.078), T_9 (0.093), T_8 (0.046) and T_7 (0.078).
  M2 listed T_13, T_11, T_10, T_9 and T_7; T_8 is new here, on a larger set of windows.

## 6. G-VAL: why it fails

The data are the 612 (real unit, pole) pairs. The measured ΔL comes from the job's records; ΔL̂ was recomputed
from H and G. All numbers are from `read/r2_gval.json`.

| | freeze | test | both |
|---|---|---|---|
| Spearman(ΔL̂, ΔL) | 0.0996 | 0.0425 | 0.0739 |
| bootstrap lower bound over units (10,000) | −0.020 | −0.060 | −0.005 |
| shuffle q975 (1,000) | 0.107 | 0.127 | 0.073 |
| slope of ΔL on ΔL̂ through the origin | 0.489 | 0.326 | 0.398 |
| sign agreement | 0.558 | 0.496 | 0.529 |
| median \|ΔL\| / median \|ΔL̂\|, nats | 0.677 / 0.021 | 0.643 / 0.022 | 0.650 / 0.022 |
| median \|ΔL\| / \|ΔL̂\| | 31.5 | 27.3 | 30.0 |
| share of pairs with \|ΔL\| / \|ΔL̂\| > 10 | 0.71 | 0.70 | 0.71 |
| Spearman(\|ΔL\|, \|ΔL̂\|) | 0.011 | 0.012 | 0.010 |
| Spearman(\|ΔL\|, Σ\|h\| at the ablated positions) | 0.024 | −0.031 | −0.018 |
| Spearman(\|ΔL\|, loss terms after the ablated positions) | 0.232 | 0.254 | 0.241 |
| Spearman(\|ΔL\| / \|ΔL̂\|, Σ\|h\|) | −0.288 | −0.281 | −0.287 |

What the records show:

- **|ΔL| does not depend on how much activation was removed.** Both halves, by quartile of Σ|h| at the ablated
  positions: Σ|h| of 0.04–1.8 gives a median |ΔL| of 0.73 nats; 1.8–6.3 gives 0.63; 6.3–37 gives 0.64;
  37–14,524 gives 0.65. Over the same quartiles the median |ΔL̂| rises from 0.010 to 0.037 nats. A causal effect
  of zeroing an activation that sums to 0.04 cannot be 0.7 nats.
- **|ΔL| grows with the number of later positions whose loss the change can reach.** Median |ΔL| is 0.22 nats with
  fewer than 200 later loss terms (4 pairs), 0.36 with 200–1,000, 0.41 with 1,000–3,000, 0.67 with 3,000–10,000
  and 0.91 above 10,000. Over the same bins the median |ΔL̂| stays between 0.004 and 0.029. Several poles
  ablated at the same positions of the same unit scatter by a median within-unit SD of 0.60 nats. This is what a
  bf16 forward gives when any change at a position alters the rounding of every later position's computation, and
  the window's summed loss adds up those rounding changes. The clean row equals the forward bit for bit
  (G-ABL-ID 0.0), so the batch layout is not the cause.
- **The ratio |ΔL| / |ΔL̂| falls as |h| grows** (Spearman −0.29). If the first-order approximation broke down
  for large perturbations, the ratio would grow with |h|, and small perturbations would agree in sign. Here the
  smallest quartile of Σ|h| has a sign agreement of 0.54 and a median ratio of 78; the largest has 0.45 and 16.
- **The observed values are what a correct ΔL̂ plus this noise gives.** The model: take noise from the 132 pairs
  with |ΔL̂| < 0.005 nats, scaled by the square root of the number of later loss terms, and set
  ΔL = β·ΔL̂ + noise. With β = 1 it gives ρ 0.054 [−0.023, 0.137], an overall sign agreement of 0.52 and a
  top-decile sign agreement of 0.58 [0.45, 0.69]. With β = 0 it gives ρ −0.003 [−0.088, 0.080] and a top-decile
  agreement of 0.50 [0.37, 0.63]. The observed values are ρ 0.074, 0.53 overall and 0.645 in the top decile. In
  no replicate does ρ reach 0.7, even with β = 10 (median 0.39). The records therefore cannot tell a correct
  gradient from one unrelated to the ablation. ρ lies inside the β = 0 range; the top decile's sign agreement lies
  just above it.
- **A finite-difference check at small scale would not have passed in the same forward.** Scaling h by (1 ± ε)
  changes the residual at those positions, which changes later rounding just as zeroing does. The difference
  would stay near the same 0.2–0.9-nat level and the quotient would be divided by a small ε. It would pass in an
  fp32 or fp64 forward. The dry run's fp64 tiny model passed (largest relative error 1.3e-3 against central
  differences; Spearman 0.990 against ablation, spec §3.5), but that model cannot show a bf16 effect. It would
  also pass on a loss restricted to the terms the first-order effect dominates, such as the exemplar's own
  next-token loss. Neither is in the records, so this point is reasoning, not a measurement.

The gate's registered failing input (the sign-flipped prediction) failed, but the pass side was never shown on a
bf16 7B forward. The spec registered no measurement of the ablation's own resolution, for example a no-effect
ablation of a neuron with h ≈ 0. G-VAL as built measures this resolution and says nothing about whether G is
right. What does show G's quality: I-GRAD is bit-identical, P = f(G, q999) holds to 2e-16, and A1's CPU check
puts bf16 gradients within about 1 % (relative L2) of the fused kernel's.

## 7. The twins

### 7.1 F and the dating rule

The dating rule puts the pole's median a⁺ below its own q999 at T_(k−1) and at or above it at T_k. On the job's
arrays: B_birth ≥ 0.999 in 100 % of units of every kind and half, and B_birth > B_pre in 442 of 442. With the
others fixed at T_(k−1), a higher own value cannot rank lower, so r_c ≤ r0 in every unit. Write F as
(log r0 − log r_c) − (log r0 − log r_o). The first term is ≥ 0 by construction, so F ≥ 0 whenever r_o ≥ r0.
That holds in 77 % (real), 91 % (pb) and 81 % (dt) of test units. F ≥ 0 in 70/71, 69/70 and 67/67. The others'
move alone costs the pole rank on average in every kind: −0.072 (real), −0.103 (pb), −0.043 (dt) decades, test.
All from `read/r4_unanticipated.json` → `a_F_and_the_dating_rule`.

When the competitors sit near B 0.9999 (repeated-token positions), the dating rule's 0.999 alone does not pass
them, and the spec's §1.3 says this is why F's sign is not forced. The size of r_c is indeed not forced. On
repeated-token test real births the competitors' median B at T_(k−1) runs from 0.99885 (the 20th) to 0.99995
(the 1st). The pole's T_k value passes all 20 in 51 % of units, against 81 % on varied-token births. But F needs
only r_c < r_o, and r_c ≤ r0 ≤ r_o already gives F ≥ 0 in most units wherever the competitors sit. Where they sit
sets F's size, through the starting rank: Spearman(F, log10 r0) is 0.89 (real), 0.91 (pb) and 0.94 (dt), test.
Repeated-token real births start at a median r0 of 33 and have F 1.29 [0.99, 1.60]; varied ones start at 2 and
have F 0.54 [0.29, 0.84] (`→ a2_where_the_competitors_sit`).

The twins are the spec's named reference for F:

| test half | real | pb | dt |
|---|---|---|---|
| mean F | 0.951 [0.736, 1.183] | 0.765 [0.559, 0.989] | 0.942 [0.676, 1.216] |
| F, r0 = 1 | 0.033 (18) | 0.058 (32) | 0.046 (28) |
| F, r0 2–10 | 0.596 (24) | 0.720 (21) | 0.497 (13) |
| F, r0 11–100 | 1.091 (14) | 1.583 (8) | 1.146 (8) |
| F, r0 > 100 | 2.488 (15) | 2.657 (9) | 2.566 (18) |
| repeated-token units: F (n) | 1.289 (39) | 0.896 (11) | 1.380 (36) |

The paired real − twin difference after a common fit of F on log10 r0, test half: against pb −0.161
[−0.270, −0.059], against dt +0.017 [−0.098, +0.129]. At a matched starting rank, a carrier's own move buys no
more rank than a dt twin's, and less than a pb twin's. H_cf OWN_RISE is a correct computation. It describes
dated poles in general, not the competition at real births.

### 7.2 The competitors' drop and token repetition

From `read/r4_unanticipated.json` → `b_drop_and_token_repetition`; "repeated" means a distinct-token share
d / size < 0.67 (the reader's split from `READ_m2.md` §5, chosen after M2's data).

| | test | both halves |
|---|---|---|
| drop, real repeated / real varied | 0.710 / 1.218 | 0.792 / 1.218 |
| drop, dt repeated / dt varied | 0.933 / 1.211 | 1.026 / 1.353 |
| real − pb, repeated real births | −0.449 [−0.668, −0.247] (39) | −0.387 [−0.553, −0.228] (78) |
| real − pb, varied real births | +0.055 [−0.211, +0.310] (31) | −0.047 [−0.217, +0.117] (71) |
| real − dt, all (registered label on the test half) | −0.113 [−0.296, +0.062] (67) | −0.179 [−0.306, −0.056] (140) |
| real − dt, repeated / varied | −0.223 [−0.445, −0.016] / +0.015 [−0.278, +0.309] | −0.232 [−0.407, −0.053] / −0.127 [−0.299, +0.044] |
| OLS of drop on kind: real − pb, without / with the distinct-token share | −0.225 / −0.086 | −0.232 / −0.070 (share +0.62 per unit) |

The BELOW_SELECTION against pb is mostly a token-repetition effect: it sits in the repeated-token births, and
adding the distinct-token share to a pooled fit shrinks it from −0.23 to −0.07 decades. On repeated tokens the
top 20 at T_(k−1) start higher (median 20th-place B 0.99885 for real, 0.99729 for dt, 0.98054 for pb, test) and
fall less by T_k.

Matching d per birth removes about half of the test-half gap (−0.226 to −0.113, AT_SELECTION). On both halves
pooled, real − dt is still −0.179 [−0.306, −0.056]: the freeze half alone gives −0.240 [−0.408, −0.069]. The
pooled number is exploratory (both halves; the split was chosen after the data). It says the competitors of real
births fall less than those of d-matched twins by about 0.18 decades. Something other than the distinct-token
count, such as lateness, which the spec's §3.3 names, may carry that remainder.

### 7.3 Part B (H_dt)

Hits under B at T_(k−1), test (`read/r3_summary.json`): real 42/71 (0.592), dt 41/67 (0.612), pb 53/70 (0.757).
The paired real − dt difference is −0.030 [−0.149, +0.090] with 7 against 9 discordant births. Pooled over both
halves it is −0.014 [−0.093, +0.064] over 140 births. M2's real − pb gap reproduces: −0.171 [−0.286, −0.057].
So does the M2 reader's split, real − pb hits on both halves: repeated −0.397 [−0.513, −0.282], varied −0.014
[−0.113, +0.085]. The d-matched twin hits as rarely as the carrier, and its bar (0.98970) sits between the
carrier's (0.99606) and pb's (0.98362).

The construction control at T_15 passes (66/67 and 0/67 test; 73/73 and 0/73 freeze). For the 19 dt units of the
example births, d recounted from the decoded exemplars equals both the record's d and the birth's d
(`read/r9_examples.json`). The other 121 dt units were not checked against text.

H_dt SUPPORTED holds. The repeated-token pattern explains the hit gap completely and the drop gap mostly.

## 8. Raw examples (test half; `banks/score-02/n12/samples/raw_examples_test.jsonl`)

All 58 example records (20 births) equal the unit records and my recount (`read/r9_examples.json`).

**Scored F > 0: L02N4649+:c3, "SQL dump CHARSET after DEFAULT"** (k 15, size 7, d 1). r0 597 → r1 3, r_c 1,
r_o 662, F +2.82, drop 1.04.

- All 7 exemplars are ` CHARSET` after `DEFAULT` in `CREATE TABLE … ENGINE=InnoDB DEFAULT CHARSET=utf8` dumps.
  They read as the job.
- The carrier's own top-3 contexts at T_14 do not: ` Lopez`, ` quil`(ombo), `vr` (a+ 0.35).
- Competitor 1, L02N9310+ (B 0.999996, rank 1 → 745): ` Gle`(ason), ` Millions`, ` Gle`(ize). Not the job.
- Competitor 2, L02N5687− (rank 2 → 7): two of its three own top contexts are `DEFAULT CHARSET=utf16` and
  `DEFAULT CHARSET=utf8`, the third `isPress[Esc]`.
- Competitor 3, L02N6933− (rank 3 → 1,050): one `DEFAULT CHARSET=utf8` among its three.
- So the job was contested at T_14 by poles that already had it among their own best contexts; the carrier passes
  them by T_15.
- The first matched random pole, L02N9894−: ` Objective`, `onia`, ` nem` (unrelated).

**Scored F = 0: L03N9631+:c3, "mid-word fragments"** (k 13, size 10, d 8). r0 2 → r1 1, r_c 1, r_o 1, F 0.

- The exemplars are mid-word pieces of long words in older prose: An`ecd`otes, ` ump`ire, ins`cribed` (twice),
  ` blas`phemous, Depos`itory` (twice), ` pamph`lets, ` abbrev`iations, E`ffic`acy. They read as the job.
- The carrier's own top-3 at T_12 are close to it: ` epith`(ets), `quoted`, `quoted`, in classical-citation
  prose.
- The single leader, L03N4464− (B 0.996 → 0.0, rank 1 → 22,016), has own contexts Maca`ú`ba, Accom`pany` and
  ` short`. Two of the three are mid-word fragments.
- Competitor L03N50− (rank 3 → 873) reads as the job: ` conject`ure, ` observ`ance, ` indign`ation in
  biblical prose.
- L03N8769+ (rank 4 → 639) has ` Bry`, UN`I`-CUB twice: capitalized fragments, not the job.
- The matched random pole, L03N1043−: MJ`PEG`, NC`IS` twice (unrelated).
- F is 0 because each move alone gives rank 1. The carrier's rise (B 0.9957 → 0.9995) passes the leader's 0.9960,
  and the leader's fall to B 0 also leaves the carrier first. F cannot tell these apart.

## 9. Code–spec differences (phase 3: `infra/pg/n12.py`, `infra/aws_pg/jobs/n12_score.py`)

1. **G-VAL's bootstrap over units uses 1,000 resamples** (`n12.validation`). The spec's §2.1 sets 10,000 for its
   intervals. With 10,000 the lower bound is −0.060 against the job's −0.059. No label changes.
2. **u of a matched pole that is itself in C** is taken against the other 19 competitors (`u_share` drops the
   scored pole from C). The spec's §3.4 says "against the unit's C". This affects 9 of 15,300 values. The floor
   q025 does not move when the rule alone changes: 0.4239 (test) and 0.4439 (freeze) under both rules, seed 7
   (`read/r3_summary.json`, `floor_matched_all20` against `floor_matched_job_rule`).
3. **The exploratory gradient labels are not written.** The spec's §3.5 says that when G-VAL fails the gradient
   labels "DOES NOT COUNT, … reported as exploratory". The verdict writes DOES NOT COUNT and the inputs (u_S,
   floor, ρ(S, gain)), but not the branch each would take. By my reading of §5 they are H_signal REFUTED,
   H_sel_signal_pb AT, H_sel_signal_dt AT and H_bridge UNDECIDED (§5 above).
4. **The ablation zeroes only the valid positions** (input index ≤ 2,047), not index 2,048. The spec says "the
   unit's positions". There is no effect, since index 2,048 has no later loss term. One position is
   affected: the first of L22N7483−:c5's 7.
5. **G-ABL-ID checks one clean row per unit**, the unit's first window (`n12_score.run`). That matches "each unit's
   clean row" literally. A unit's other windows rely on the forward's loss without a clean row of their own.
6. **Ties in ρ(S, gain).** The job computes gain from scalar `np.log10` of integer ranks. Mathematically equal
   rank ratios (for example 1/7 and 17/119) can differ in the last bit between scalar and array log10. A tie can
   therefore become an order either way. In 9 of 442 units my value differs by up to 0.0135; the means differ by
   1.7e-4. The spec does not define ties of gain.
7. **The spec's §1.3 says F's sign "is not forced by the dating rule".** That is a spec statement, not code, and
   §7.1 shows the dating rule forces r_c ≤ r0 in every unit. The spec chose the twins' F as the reference but put
   no label on F real − twin. The reported paired difference shows no gap against dt.
8. **No difference found** in: the competitor rule (top 20 by B, stable order, B > 0, the pole's neuron excluded;
   no unit had fewer than 20 candidates or a tie at place 20); `cf_ranks`; `bar_pre` (10th-highest of the other
   22,015); the identity readout; `pull` (q999 NaN → 0); the loss-match rule (j ∈ [1, 2,047], ±8 of any unit
   position in the window, ties by smaller j); the gradient leaf at the lowest needed layer; ΔL̂ = Σ −h·G over
   valid positions; ΔL summed over windows against the forward's loss; the labels, bar, ceiling and dt control in
   `verdict`; `gates_ok`.

## 10. What I could not check

- **The measured ΔL and the gradient G themselves.** They need the GPU. I used the job's ΔL and banked G. The
  only checks on G are I-GRAD and P = f(G, q999).
- **The loss-matched positions.** The per-position losses ℓ and the chosen indices are not banked (only Glm,
  their mean G), so `dell` (median 0.0168 nats, test) and the match are not recomputed.
- **B from activations for every unit.** Only 66 arrays on 6 (point, layer) pairs were checked against my own
  percentile code. The rest of the real and pb B values are bit-identical to M2's banked ones, which come from the
  same `m2.percentile`. The dt units (new positions) have no second run, and their B was checked only on those 6
  pairs.
- **The dating of the dt twins.** It is done on the grid's banked arrays (Blackwell at T_7–T_14), and I did not
  redate. In the sampled arrays, the L40S median a⁺/q999 is below 1 at T_(k−1) and above 1 at T_k for the dt
  poles. One pb pole (L28N4613+:c0:pb at T_9) is at 1.003 at T_(k−1), which fits the cross-device residual.
- **d for 121 of 140 dt units.** Only the 19 in the examples were checked against decoded text.
- **The twins' T_k forward against a second run.** Only the 20 pb births at T_15 have one (M2's pb end vectors,
  exact).
- I did not read the N12 `JOURNAL.md` or spec §12, so their text was not checked against the artifacts.

## 11. Files

- `read/fetch_stats.py`: byte-range reads of q999, max_abs and histograms (output in the scratchpad, not
  committed).
- `read/r1_recount.py` → `read/r1_units.json`, `read/r1_abl.json`: per-unit recount, the comparison with the
  records, and the ablation pairs.
- `read/r2_gval.py` → `read/r2_gval.json`: G-VAL, its diagnostics, and the power and sign simulation.
- `read/r3_summary.py` → `read/r3_summary.json`: summaries and labels by the spec's §5 rules.
- `read/r4_unanticipated.py` → `read/r4_unanticipated.json`: F against the dating rule; the drop against token
  repetition.
- `read/r5_m2v_k0.py` → `read/r5_m2v_k0.json`: G-M2V and K0 over every banked unit.
- `read/r6_bcheck.py` → `read/r6_bcheck.json`: B from activations and histograms.
- `read/r7_gates.py` → `read/r7_gates.json`: the gates.
- `read/r8_compare.py` → `read/r8_compare.json`: phase 2, field by field.
- `read/r9_examples.py` → `read/r9_examples.json`: the raw examples' records and d.

Inputs not committed: the job's arrays (S3 `results/next_n12/score-02/out/score/n12/arrays/`, 800 MB), M2's
vectors (S3 `results/next_m2/score-06/out/score/m2/vectors/`) and the grid stats members, all synced to the
scratchpad.
