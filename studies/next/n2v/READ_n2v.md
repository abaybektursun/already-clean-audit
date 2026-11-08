# N2V: the independent read

Reader: an Opus agent, worktree `interp_research-n2vread`, branch `n2vread`, made detached at tag `n2v-freeze-2`
(`0a3742e43`), 2026-11-07 (the run ended 2026-11-08 UTC). CPU only, no model run. S3 reads only (profile `fuelos`,
`us-east-1`). Scripts and their JSON outputs: `studies/next/n2v/read_n2v/`. Downloaded files:
`$SP/n2vread/s3/score-02/n2v/` (`$SP` = this session's scratchpad). The registered verdict was written to
`$SP/n2vread/verdict_n2v.json`, not to the repo.

## 1. Answer

**The verdict stands under my recount: G-VAL window PASS, G-FD PASS, G-VAL own PASS, window_jvp PASS on both halves;
branch A.** The number that settles it is the test half's window Spearman: pred_bf16 against the fp32 zero-ablation
ΔL, ρ 0.9600, bootstrap interval over units [0.9155, 0.9902], shuffle 97.5th percentile 0.1176, 284 pairs, 71 units
(`read_n2v/r2_verdict.json` → `halves.test.window`). The freeze half's was ρ 0.9706 [0.9390, 0.9930]
(`thresholds.json` → `freeze.window.validation`, reproduced by my code to 3e-16).

- Every pair's eight readouts recount bitwise from `losses/tNN.npz`: 612 of 612 pairs, 0.0 difference in every field
  (`read_n2v/r1_recount.json` → `compare_with_records`).
- Every number of the registered verdict equals mine within 2.8e-14; labels and branch are identical
  (`read_n2v/r4_compare.json`: 81 fields, 62 bitwise, the rest within 1 to 2 units of the last place).
- The re-reading (branch A) gives N12's printed exploratory labels and values exactly: H_signal REFUTED (mean u_S
  0.5268 [0.4514, 0.6000] against the bar 0.4225), H_sel_signal_pb AT, H_sel_signal_dt AT, H_bridge UNDECIDED. This is
  a re-reading of known values, not a test (§5).

Three findings change how the result reads:

1. **The rank agreement is not carried by between-unit spread.** Within a unit, the four poles' predicted and
   measured ΔL agree in order: on the test half the largest-|prediction| pole is the largest-|measurement| pole in 65
   of 71 units (0.915 [0.845, 0.972]; chance level 0.25), and ρ after per-unit centering is 0.938. Permuting the
   measurement within units leaves ρ at a median 0.286 (§6 a).
2. **The slope through the origin (0.837 on the test half, 1.110 on the freeze half) is set by a few large pairs
   whose zero ablation is far from first order.** The top 5 of 284 test pairs carry 48 % of Σ pred²; without the top 3
   the slope is 1.025. The 10 % finite difference has slope 0.988 (test) and 1.008 (freeze) (§6 c).
3. **Where the bf16 prediction and the fp32 forward-mode prediction disagree, the measurement sides with the fp32
   one.** In the 10 % of test pairs where they disagree most, pred_jvp is closer to ΔL in 26 of 28. One test unit,
   L02N4649+:c3 (layer 2, T_14, a K-A unit), has the wrong bf16 prediction on all four poles (§6 e).

## 2. Blind order

- I read the spec, the code and the freeze-half files listed in the brief, at the tag. I ran no `git log` outside my
  worktree, read no other worktree, branch, journal or scratchpad.
- The freeze half's values were known to me before any recount (`thresholds.json`, `JOURNAL.md`), as the brief allows.
- The test records (`n2v_test.jsonl`, `samples/raw_examples_test.jsonl`) and `losses/` were opened only for the recount.
- Phase 1, commit `ee40dba7c`: the recount of every pair (`r1_recount.*`) and the verdict quantities with my own code
  (`r2_verdict.*`), including my labels and branch. It was committed before the registered verdict was run.
- Phase 2: the registered verdict (`r3_run_registered.py`), the comparison (`r4_compare.*`), the re-reading
  (`r5_reread.*`), the checks (`r6_checks.*`), the raw examples (`r7_examples.*`), the gates (`r8_gates.*`).

## 3. The records and their sha256

Bucket `pg-lineage-769063704696`, prefix `results/next_n2v/score-02/out/score/n2v/`, downloaded 2026-11-07.

| file | bytes | sha256 |
|---|---|---|
| `losses/t05.npz` | 464,987 | `7a913c9b4705e55f1d50e39c471c639b3a69d4caf562acc94205907485c56ed3` |
| `losses/t07.npz` | 5,182,057 | `0457974cf7bee89fd8e532a1d994cfe2e90901cd1f9cf1e15d17be16875da800` |
| `losses/t08.npz` | 4,852,515 | `b82b2bb70f7f4ec3272540a2d7738941022e3bd689526ed62d4171a4745ba7c8` |
| `losses/t09.npz` | 5,607,894 | `fa68ebaa34c41250f9a51a1e2954ef552b845b882e6e635b755438ac4b85fffe` |
| `losses/t10.npz` | 11,419,186 | `44b1261b76b36398bfa8623d996087c04b49feb942bca2f86c3ff19bf61ec341` |
| `losses/t11.npz` | 18,735,950 | `465fd3c7bb48a1bf6c8415767344b9647b97ef0c583dfe0008726804f6713320` |
| `losses/t12.npz` | 19,196,761 | `863f1043617aae7762e1b3a7a02dd9803f553d4c821c1d72f53b073f1bc63a40` |
| `losses/t13.npz` | 17,861,685 | `c2f07be260390457d8cc138488c3c8b4ef1915b035614831a41c496fde51f23f` |
| `losses/t14.npz` | 10,161,114 | `1c976e9871d31c2379cad8dd384c02f9853222fac5a2a1fbd3bee58fe12a5612` |
| `n2v_test.jsonl` | 140,070 | `7bcee720568b5b97c38beb622fc3819cbc8f63e425f797136efe464f0dcc238f` |
| `samples/raw_examples_test.jsonl` | 62,481 | `2501ec020f8964212f497e0855c5c518fd5e76a2e83315f34bb16f5eee12c0b7` |
| `metrics.json` | 39,229 | `28019dab37d80b178d38f2ed3221cc50800a720c99b37a7117e8b5f3f410b7aa` |
| `n2v_freeze.jsonl` | 162,278 | `58e85a6403bb6101ae358e709d285b5d0f7e48d00c4bb845a72aca469ec06747` |
| `samples/raw_examples_freeze.jsonl` | 57,516 | `c4340212d3984dbed972998430bdfa2d5c8d65ab6e47503079a0ef294c0c075a` |

The npz files total 93,482,149 bytes. `metrics.json` and `n2v_freeze.jsonl` equal `thresholds.json` →
`sha256_job_metrics` and `sha256_n2v_freeze`. All three freeze-side files equal the banked copies in
`banks/score-02/n2v/` byte for byte. N12's inputs in the worktree equal the registered sha256 (`c223dfee…`,
`30f31cb1…`, `877926da…`). The registered verdict output has sha256 `78b57f78…cd5fb2`; it read thresholds sha256
`cdd2ed26…`, the value the journal records for freeze 2.

## 4. Gates (`read_n2v/r8_gates.json`)

`metrics.json`: complete true, nothing parked, parts core, fd, jvp. 42 gates, 42 pass. 42 registered failing inputs,
42 fail.

| gate | pass rule | value | failing input and its value |
|---|---|---|---|
| G_SAME | torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S | exactly those | transformers "0.0.0": fails |
| INPUTS | units.json and N12's two files at registered sha256 | no mismatch | a zero sha256: fails |
| G_CORPUS | tokens sha256 `2aca80ca…` | equal | a zero sha256: fails |
| tNN_G_REV (9 points) | downloaded commit = registered | equal at all 9; I checked each against `m2.POINTS` | a zero revision: fails |
| tNN_G_CFG (9) | no differing config key | `diff_keys` [] | model_type renamed: fails (`diff_keys` ["model_type"]) |
| K_A | 40 pairs' bf16 ΔL within 1e-6 of N12's banked ΔL | max 0.0, 40 of 40 exact | poles rolled: smallest gap 0.0166 nats, fails. I recomputed 0.0166015625 from N12's records; the 10 units are the first 10 real T_14 units by id |
| tNN_G_FP32 (9) | all parameters fp32, "highest", TF32 off | fp32 | dtypes before the cast: bf16, fails |
| tNN_G_CAUSAL (9) | every changed row bitwise equal to its clean row at j < i0 | 8,883 of 8,883 rows, largest 0.0 | zero rows through i0 must differ: 52 of 3,948 equal (10, 20, 6, 8, 5, 0, 0, 3, 0 at T_14 … T_5), so it fails |
| I_FWD | T_13 clean rows again, reversed: bitwise | 187 of 187, largest 0.0 | clean row against its first zero row: fails |
| I_JVP | 16 windows × 2 poles again, reversed: bitwise | 32 of 32, largest 0.0 | pole 1's tangent against pole 0's: fails |

G-CAUSAL from the raw rows (`r1_recount.json` → `G_CAUSAL_from_npz`, `r8_gates.json` → `cross_checks`):

- All 8,883 changed rows (3,948 zero, 3,948 fd, 987 probe) at all nine points equal their clean row bitwise before i0.
- 8,859 differ at or after i0. 8,414 differ at i0 itself.
- 24 rows equal their clean row at every position: 18 probe rows (the probe was exactly 0) and 6 zero or fd rows of
  L24N3498+:c3 at T_7 (poles 2 and 3 in windows 2914 and 6726). No pair's ΔL is exactly 0.
- Rows per point from the npz equal `n_rows` at every point. The zero rows equal through i0 equal the failing input's
  `n_equal` at every point.
- 72 example rows across all nine points are written with their first differing j. In each, the first difference is
  at i0 (`examples`).

I-FWD and I-JVP: the second passes are not banked, so the identities themselves cannot be recomputed. What the npz
allows:

- I-FWD's failing input: 187 of 187 T_13 clean rows differ from their zero-0 row.
- I-JVP's failing input: 16 of 16 pole-1 tangents differ from pole 0's.
- A further identity with other batch companions: the same window's clean row in two different units at the same
  point is bitwise equal in 10 of 10 cases (largest 0.0).
- A causal identity of the tangents: all 3,948 forward-mode rows are exactly 0 before i0, and all are nonzero at or
  after it.

`slot_control` (read, not recomputed): view offsets 0, 8, 0, 8 bytes; logits and the copy's loss 0.0 at every slot;
the row view's loss 9.54e-7 at slots 1 and 3. Forward-mode peak memory 36.62 GB at every point.

## 5. The readouts recounted (`read_n2v/r1_recount.json`)

My code rebuilds the plan from `units.json` and N12's `abl` order. It does not import `infra.pg.n2v`.

- The windows per unit equal the npz keys for 153 of 153 units. All 153 key sets are complete.
- ΔL, ΔL_own, ΔL_fd, ΔL_fd_own, pred_jvp, pred_jvp_own, probe and probe_own are summed as §0.5 defines them (fp64 over
  fp32 per-position differences, the unit's windows, the own indices).

| field | largest abs difference vs records (job's order) | largest rel difference | bitwise | exact sum (math.fsum) vs records |
|---|---|---|---|---|
| dL | 0.0 | 0.0 | 612 / 612 | 0.0 |
| dL_own | 0.0 | 0.0 | 612 | 0.0 |
| dL_fd | 0.0 | 0.0 | 612 | 0.0 |
| dL_fd_own | 0.0 | 0.0 | 612 | 0.0 |
| pred_jvp | 0.0 | 0.0 | 612 | 2.8e-17 (rel 2.0e-16) |
| pred_jvp_own | 0.0 | 0.0 | 612 | 1.7e-18 (rel 2.2e-16) |
| probe | 0.0 | 0.0 | 612 | 0.0 |
| probe_own | 0.0 | 0.0 | 612 | 0.0 |

- pred_bf16 and meas_bf16 in the N2V records equal N12's banked `abl[i][1]` and `abl[i][2]`: 612 of 612 bitwise.
- t, half, rank, n_windows and n_later are equal for 612 of 612.

## 6. The verdict quantities, mine (`read_n2v/r2_verdict.json`) and the registered code's

Procedure, as `n12.validation` registers it:

- One `default_rng(seed)` stream draws 1,000 shuffles of the prediction, then 1,000 bootstrap resamples of units.
- The interval is the 2.5th and 97.5th percentiles. Pairs are ordered by unit id, then rank.
- G-ROOM: 1,000 replicates, seed 0, measurement = prediction + each pair's probe (own: probe_own) × a random sign.
  Its failing input uses N12's bf16 carrier ΔL as each unit's floor.
- The registered code walks G-ROOM's pairs in file order (T_14 first). My G-ROOM in that order equals the registered
  medians to 3e-16. In unit order the medians move by at most 3e-5, and the failing input's by at most 0.005.

**Test half** (284 pairs, 71 units). The "registered" column is `verdict_n2v.json` → `test`.

| readout | ρ | interval seed 0 (registered) | seed 1 | seed 2 | basic interval, seed 0 | 10,000 resamples | shuffle q975 (seeds 0/1/2) | sign agreement | slope | G-ROOM median | registered values |
|---|---|---|---|---|---|---|---|---|---|---|---|
| window | 0.95996 | [0.9155, 0.9902] | [0.9104, 0.9905] | [0.9093, 0.9901] | [0.9297, 1.0045] | [0.9132, 0.9902] | 0.118 / 0.125 / 0.113 | 0.954 | 0.837 | 0.99990 | ρ 0.9599593, [0.9154659, 0.9902215], q975 0.1176193, room 0.9999020 |
| fd | 0.98747 | [0.9712, 0.9947] | [0.9697, 0.9948] | [0.9703, 0.9948] | [0.9802, 1.0038] | [0.9712, 0.9948] | 0.116 / 0.121 / 0.112 | 0.954 | 0.988 | 0.99571 | ρ 0.9874693, [0.9711686, 0.9947461], q975 0.1162966, room 0.9957126 |
| own | 0.98636 | [0.9693, 0.9964] | [0.9690, 0.9962] | [0.9684, 0.9964] | [0.9763, 1.0034] | [0.9679, 0.9963] | 0.118 / 0.116 / 0.113 | 0.982 | 0.976 | 0.999997 | ρ 0.9863571, [0.9692836, 0.9963950], q975 0.1178929, room 0.9999974 |
| window_jvp | 0.96661 | [0.9232, 0.9941] | [0.9177, 0.9942] | [0.9170, 0.9943] | [0.9391, 1.0100] | [0.9213, 0.9940] | 0.119 / 0.124 / 0.111 | 0.972 | 0.854 | 0.99992 | ρ 0.9666058, [0.9231852, 0.9941457], q975 0.1188621, room 0.9999167 |

- Failing inputs (the sign flip): ρ −0.960, −0.987, −0.986, −0.967. All fail.
- G-ROOM's failing input on the test half: median ρ 0.0762 (file order; registered 0.07619). It fails, as it must for
  the verdict to proceed.
- Ceiling: no readout has ρ > 0.999. The registered code ran no inspection (its `inspection` field is empty). My inspection
  anyway: no pair's measurement equals its prediction on any readout. The median |meas − pred| / |pred| is 0.105
  (window), 0.131 (fd), 0.069 (own) and 0.085 (window_jvp). Each would clear the 1e-3 bar.
- fd copy test: largest |ΔL_fd − 0.1 pred| / |0.1 pred| is 80.1 (test) and 29.7 (freeze). Not a copy.
- bf16 against fp32 prediction: Spearman 0.99505, median relative difference 0.0263 (registered: 0.9950541, 0.0263477).
- Probe medians over carriers: window 1.47e-4, own 2.66e-6 nats (equal to the registered values).

**Freeze half**, my code against `thresholds.json`: ρ, interval, shuffle q975, slope and sign agreement equal to
within 3e-16 on all four readouts. G-ROOM in file order is equal to 3e-16; its failing input is 0.06407.

**Labels, by §5 and `g_label`'s rules applied with my code:**

| readout | freeze (thresholds.json) | freeze (mine) | test (mine) | both halves (mine) | registered (both) |
|---|---|---|---|---|---|
| window | PASS | PASS | PASS | PASS | PASS |
| fd | PASS | PASS | PASS | PASS | PASS |
| own | PASS | PASS | PASS | PASS | PASS |
| window_jvp | PASS | PASS | PASS | PASS | PASS |

The test half has 71 units, at least 50. G-ROOM's failing input fails on the test half. Window PASS gives **branch A**.
In the spec's words (§1.2): "N12's instrument tracks zero ablation once the loss is resolved". Next:
`n2v.reread`, "a dated re-reading, not a confirmation".

Against the hypotheses of §1.3:

- H_res predicted window PASS on both halves; it was not refuted.
- H_nonlin is refuted by window PASS.
- H_grad is refuted by G-FD PASS.
- H_later is refuted by window PASS.
- H_floor is refuted by G-ROOM PASS on both halves.

The registered headline, reproduced: `G-VAL window PASS; G-FD PASS; G-VAL own PASS; branch A; N12 re-read: H_cf
OWN_RISE; H_away_sel_pb BELOW_SELECTION; H_signal REFUTED; H_bridge UNDECIDED; H_dt SUPPORTED`. Field by field:
`read_n2v/r4_compare.json`; the largest numeric difference is 2.8e-14, on fd `max_rel_dev` (80.12, so relative 3.5e-16).

## 7. The re-reading (branch A; `read_n2v/r5_reread.json`)

This is a re-reading of known values. N12's test half was read on 2026-11-06, and these labels were printed then as
exploratory (`studies/next/n12/READ_n12.md` §4–§5, `N12_SPEC.md` §12).

- The four gradient labels do not read the measured column. They read u_S, u_gain, u_matched and ρ(S, gain) from
  N12's records. So the fp32 measurement changes their status, from DOES NOT COUNT to counted, and cannot change their
  values. Only N12's G-VAL reads the measurement.
- I recomputed u_S from each record's `comp[*].S` and `pull.S` (442 of 442 equal to the records). I recomputed
  ρ(S, gain) from `comp[*].S` and `gain` (largest difference 1.1e-16).
- I recomputed the bootstrap and the test floor with my own code under N12's rules. `u_matched` is taken from the
  records, because the matched poles' S is not banked.

| label | mine | registered re-read | N12's printed exploratory value |
|---|---|---|---|
| G-VAL with fp32 ΔL, freeze (N12 file order) | ρ 0.97059 [0.94049, 0.99227], q975 0.1091, PASS | identical (≤ 1.1e-16) | ρ 0.0996 with the bf16 ΔL (fails) |
| G-VAL with fp32 ΔL, test | ρ 0.95996 [0.91547, 0.99022], q975 0.1176, PASS | identical (≤ 2.2e-16) | ρ 0.0425 with the bf16 ΔL (fails) |
| H_signal | REFUTED: mean u_S 0.52676 [0.45141, 0.60000]; bar min(0.5, 0.44573, 0.42252) = 0.42252; u_gain 0.0627 [0.0408, 0.0880] | REFUTED, identical values | REFUTED, u_S 0.5268 [0.451, 0.600], bar 0.4225 |
| H_sel_signal_pb | AT_SELECTION: −0.03857 [−0.13357, +0.05714], n 70 | identical | AT, −0.0386 (READ_n12 §4); −0.039 [−0.134, +0.057] (N12_SPEC §12) |
| H_sel_signal_dt | AT_SELECTION: −0.02015 [−0.12388, +0.08360], n 67 | identical | AT, −0.0201; −0.020 [−0.124, +0.084] |
| H_bridge | UNDECIDED: −0.01259 [−0.07180, +0.04590], n 71 | identical | UNDECIDED, −0.01259 [−0.0718, +0.0459] |

The freeze interval differs from N2V's own freeze window interval ([0.9390, 0.9930]). The re-reading passes N12's
freeze records in N12's file order, so the shuffles and resamples are drawn on another pair order; ρ is the same.

## 8. Checks (`read_n2v/r6_checks.json`)

### (a) Within-unit ranking

N12 asks which of a unit's four poles moves the loss most. Per unit (four poles), window readout:

| | freeze (82 units) | test (71 units) |
|---|---|---|
| largest-\|pred\| pole = largest-\|ΔL\| pole | 72 of 82, 0.878 [0.805, 0.939] | 65 of 71, 0.915 [0.845, 0.972] |
| same, under a within-unit permutation of ΔL (97.5th percentile) | 0.341 (chance level 0.25) | 0.352 |
| Kendall τ over the 4 poles, signed values: mean; share τ = 1 | 0.902 [0.854, 0.943]; 0.78 | 0.915 [0.854, 0.962]; 0.85 |
| Kendall τ over \|values\| (which pole moves the loss most): mean; share τ = 1 | 0.858 [0.801, 0.911]; 0.71 | 0.859 [0.784, 0.920]; 0.70 |
| ρ over pairs | 0.971 | 0.960 |
| ρ after per-unit centering | 0.960 | 0.938 |
| ρ of within-unit ranks (1–4) | 0.932 | 0.927 |
| ρ after dividing both by the unit's max \|pred\| | 0.969 | 0.946 |
| ρ under within-unit permutation of ΔL (keeps all between-unit spread), median [2.5 %, 97.5 %] | 0.272 [0.160, 0.394] | 0.286 [0.173, 0.405] |
| ρ after centering, under the same permutation: median; 97.5 % | 0.058; 0.207 | 0.049; 0.222 |

- Between-unit spread alone gives ρ near 0.28. The observed 0.96 needs the within-unit order. So the across-pair ρ
  does not rest on between-unit spread.
- The fd, own and window_jvp readouts give the same picture. On the test half the top-pole match is 0.915, 0.930 and
  0.915, and the centered ρ is 0.991, 0.975 and 0.943.

### (b) The smallest predictions and the measured floor

The measured floor is the probe, per unit, summed over the windows:

- freeze: median |probe| 1.67e-4, largest 7.9e-4 nats; own 4.3e-6.
- test: median 1.47e-4, largest 7.6e-4; own 2.7e-6.

Pairs with small predictions:

- **|pred_bf16| < 1e-3**: 20 freeze and 14 test pairs. Median |ΔL| is 5.1e-4 on both halves, against median |probe|
  1.8e-4 and 1.1e-4. The sign agrees with pred_bf16 in 0.85 and 0.79 of them, and with pred_jvp in 0.85 and 0.93.
  |ΔL| exceeds 10 × |probe| in 0.10 and 0.43. These pairs sit within a few probe floors.
- **|pred_bf16| < 1e-4**: 3 pairs.
  - L16N864+ (freeze carrier): pred −3.9e-5, ΔL +6.5e-5, probe 7.8e-5. At the floor.
  - L19N7231+ (freeze): pred_bf16 8.9e-5, ΔL 9.4e-5, probe −1.2e-4, pred_jvp 2.2e-3. Its own-loss terms are 0.0508
    measured and 0.0524 predicted, and the later positions cancel them; the window sum is a small difference of large
    terms.
  - L01N7376+ (test): pred_bf16 −5.7e-5, but pred_jvp −5.5e-4 and ΔL −5.4e-4. The bf16 prediction is 10 times too small.
- Over all pairs, the residual |ΔL − pred_jvp| has median 1.3e-3 (freeze) and 1.5e-3 (test). It exceeds 10 × the
  unit's probe in 51 % and 53 % of pairs. The gap between prediction and measurement is not rounding (see c).

### (c) Linearity

- Median ΔL / pred_bf16 lies between 0.94 and 1.06 in every |pred| decile of both halves.
- Median ΔL_fd / (0.1 ΔL) lies between 0.89 and 1.13 in deciles 3–10. In the bottom decile it spreads widely (10th to
  90th percentile −4.6 to 14.6 at freeze), where values near the probe floor.
- ΔL_fd / (0.1 pred_jvp) tightens with |pred|. In the top decile it is 0.97 / 1.01 / 1.05 (freeze) and 0.98 / 1.01 /
  1.04 (test) at the 10th, 50th and 90th percentiles.
- A quadratic through 0 fitted to the two rows (ΔL = a + b, ΔL_fd = 0.1a + 0.01b) gives a first-order coefficient a.
  It tracks pred_jvp with Spearman 0.990 and 0.993 and slope 1.000 and 1.008. The second-order share, median
  |b| / |ΔL|, is 0.20 (freeze) and 0.18 (test).

**Why the test slope is 0.837.** The slope Σ pred·ΔL / Σ pred² is set by the largest predictions: the top 5 test pairs
carry 48 % of Σ pred². Among the top 3:

- L03N235- (in L03N9245-:c3, T_13): pred −0.770, ΔL −0.497, ΔL / pred 0.65. Its 10 % step agrees (ΔL_fd / 0.1 pred = 1.0004).
- L06N8966- (in L06N7759+:c2, T_13): pred −0.650, pred_jvp −0.564. Zero ablation gives ΔL +0.230, with the opposite
  sign. The 10 % step keeps the sign (ΔL_fd −0.0486, ratio 0.75). Its own-loss part stays negative (ΔL_own −0.283
  against pred_jvp_own −0.488), so the reversal comes from the later positions.

Without the top 3 pairs the test slope is 1.025. Without the top 5 it is 1.052. The freeze half's 1.110 is the mirror
case: L07N4530- (pred 0.664, ΔL 0.972) and L29N8509- (ratio 1.57), with fd ratios 1.06 and 1.04. Without its top pair
it is 1.053. Both slopes come from second-order response of the zero ablation on a few large pairs. They do not come
from the gradient: fd slopes 0.988 and 1.008.

### (d) Leverage

| | freeze window | test window | test fd | test own | test window_jvp |
|---|---|---|---|---|---|
| Spearman | 0.971 | 0.960 | 0.987 | 0.986 | 0.967 |
| Pearson | 0.971 | 0.880 | 0.995 | 0.975 | 0.897 |
| leave one unit out, range | 0.9695–0.9800 | 0.9587–0.9790 | 0.9869–0.9927 | 0.9858–0.9930 | 0.9656–0.9859 |
| carriers (rank 0) | 0.985 (82) | 0.988 (71) | 0.973 | 0.993 | 0.995 |
| competitors (ranks 1–3) | 0.965 (246) | 0.951 (213) | 0.991 | 0.985 | 0.958 |

- The test window Pearson (0.880) is pulled down by the L06N8966- pair. Removing L06N7759+:c2 raises the test window
  Spearman to 0.979.
- By point (test window): T_13 0.861 (13 units), T_14 0.934 (7), all other points 0.974–1.000 (1–13 units each).
- By layer group (test window): L00–07 0.893 (27 units), the others 0.984–0.993. On the freeze half: T_14 0.883
  (16 units), L08–15 0.868 (11 units).
- The weakest cells are early layers and T_13 / T_14, where the large second-order pairs and the bf16 misses of (e) sit.

### (e) bf16 prediction against the fp32 forward-mode prediction

- Spearman 0.987 (freeze) and 0.995 (test). Median relative difference 0.038 and 0.026.
- The two differ by more than 10 % in 22.6 % and 19.7 % of pairs, and by more than 50 % in 6.1 % and 4.6 %.
- pred_jvp is closer to ΔL in 59 % and 61 % of all pairs. In the 10 % of pairs where the two predictions disagree
  most, it is closer in 24 of 32 (freeze) and 26 of 28 (test).
- In those pairs the bf16 prediction's median relative error against ΔL is 0.51 and 0.64, against 0.087 and 0.085 in
  the rest. Spearman of |pred_bf16 − pred_jvp| with |ΔL − pred_bf16| is 0.66 and 0.59.
- With the 10 % step as the measurement, 0.1 × pred_jvp gives ρ 0.992 and 0.994, against 0.978 and 0.987 for
  0.1 × pred_bf16.

So the pairs where the bf16 gradient departs from the fp32 one are mostly where pred_bf16 misses the measurement.

The clearest case is L02N4649+:c3 (test, layer 2, T_14, a K-A unit and a raw example). All four bf16 predictions are
wrong:

| pole | pred_bf16 | pred_jvp | ΔL |
|---|---|---|---|
| carrier | −0.0118 | +0.00057 | +0.00112 |
| L02N9310+ | −0.0039 | +0.0031 | +0.0040 |
| L02N5687- | −0.0239 | +0.00044 | +0.0009 |
| L02N6933- | +0.00023 | −0.0275 | −0.0267 |

Within this unit the bf16 prediction names L02N5687- as the pole that moves the loss most; the fp32 measurement names
L02N6933-.

Two counter-cases:

- L15N10364+ (freeze): pred_bf16 +0.035 against pred_jvp −0.265. The 10 % step sides with pred_jvp (ΔL_fd −0.0258
  against −0.0265). The zero ablation saturates (ΔL −0.069).
- L19N7231+: pred_jvp is off on the window sum, through cancellation (b).

### (f) Freeze half against test half

Bootstrap of the difference in ρ, freeze − test, 2,000 resamples per half:

- window +0.011 [−0.036, +0.063]
- fd −0.009 [−0.036, +0.014]
- own −0.002 [−0.020, +0.020]
- window_jvp +0.012 [−0.032, +0.061]

No validation differs beyond its interval. The slopes differ (1.110 against 0.837 on window), for the reason in (c).

### (g) Room

**Could a readout have failed on this data?** Yes. With N12's bf16 measurement in place of ΔL, the window readout
gives ρ 0.0996 (freeze) and 0.0425 (test). With ΔL shuffled over pairs it gives median 0.002 and 0.009. The sign flip
gives −0.97 and −0.96. The bar of 0.7 and the shuffle floor (0.10–0.12) were reachable on both sides.

**G-ROOM:**

- Medians: 0.99989 (window), 0.9943 / 0.9957 (fd), 0.999996 (own), 0.99988 / 0.99992 (window_jvp).
- Its failing input: 0.064 (freeze) and 0.076 (test).
- The measured floor (1.5e-4) is about a hundredth of the 0.017-nat bound the offline gate computed for the window
  readout. For the fd readout it is a tenth of the 0.0017-nat bound.

**Could the ceiling have fired?** Not on window from any fp32 measurement short of a copy of pred_bf16 itself. A
measurement exactly equal to the fp32 first-order prediction would give ρ = Spearman(pred_bf16, pred_jvp) = 0.987 and
0.995, below 0.999. On own and window_jvp a copy gives 1.0 > 0.999, and the inspection would not clear it, because
the pairs would be equal. So the ceiling worked as a copy detector, as §3.4 registers it. It had no room to fire on a
genuine near-linear window measurement.

### (h) Other findings

1. **A defect in a reported field, `sum_abs_h`.** The job's h capture is keyed by (window, layer).
   - Two units share three windows at one point and layer: L01N369+:c0 and L01N369+:c4 at T_11, layer 1, windows
     5909, 5949 and 7957, at different indices (`r6_checks.json` → `h_shared_window_layer_indices`).
   - The later unit's capture overwrites the earlier one's. c0's four pairs' `sum_abs_h` therefore sum c4's h at c4's
     indices for those windows.
   - No rule or readout reads `sum_abs_h`. I could not check the size, because h is not banked.
   - L01N369+:c0 is a raw example of the test half.
2. **Zero rows that change nothing.** Six zero or fd rows of L24N3498+:c3 (T_7, poles 2 and 3) equal their clean row
   at every position: the ablation was an identity in fp32 at those indices. 18 probe rows are exactly 0. No pair's
   ΔL is exactly 0, so the sign agreement has no zero entries.
3. **The fd scale.** The registered fd prediction is scaled by 1 − 0.9 = 0.09999999999999998, not 0.1. My code uses
   0.1. Every fd number agrees to ≤ 1.1e-16 except `max_rel_dev` (2.8e-14 on 80.1).
4. **The bootstrap count.** The bootstrap over units runs 1,000 resamples, as §2.1 states for N12's code. With 10,000
   the test window interval is [0.9132, 0.9902]. Seeds 1 and 2 move the lower bound by at most 0.006. The basic
   interval's lower bound is 0.930. Every interval stays above 0.9; the shuffle floor is near 0.12.

## 9. Raw examples (`samples/raw_examples_test.jsonl`; `read_n2v/r7_examples.json`)

For all 10 test examples:

- every pair's ten numbers in the record equal my npz recount bitwise (largest difference 0.0);
- the record's `carrier_zero_top5` (window, j, ΔL at j, clean loss) equals the top 5 recomputed from the npz;
- all five positions sit at or after i0.

The token text is the job's decode. The corpus was not opened here.

**Small |ΔL − pred_bf16|, among the carriers: L03N9631+:c3, pole L03N9631+ (carrier), T_12.**

- pred_bf16 0.007634, ΔL 0.007518 (|difference| 1.2e-4), pred_jvp 0.007414, ΔL_fd 0.000473 (0.1 × pred 0.000763).
- ΔL_own 0.01093 against pred_jvp_own 0.01091. N12's bf16 measurement was −1.298.
- Exemplar: "…their names are ins [[cribed]] on the wall in the VC Corner Cemetery Memorial…" (10 exemplars).
- Five largest changes in the first window (1960, i0 1888), with the token predicted:

| j | change (nats) | token |
|---|---|---|
| 1895 | −0.000548 | " Cemetery" |
| 1890 | −0.000357 | " wall" |
| 1896 | −0.000258 | " Memorial" |
| 1893 | +0.000188 | " VC" |
| 1892 | +0.000177 | " the" |

- Over all its windows, the largest change is +0.0076 at j 345 of window 4163 (i0 345).
- Over all 40 pairs, the smallest gap is L01N2347+ in L01N369+:c0: pred_bf16 −0.011226, ΔL −0.011320, pred_jvp
  −0.011535, ΔL_fd −0.00136, N12's bf16 3.025. Its own pole's top change is −0.0045 at window 5949, j 863 (i0 861).
  No token text is banked for a non-carrier pole.

**Large |ΔL − pred_bf16|, among the carriers: L12N5641+:c0, pole L12N5641+ (carrier), T_12.**

- pred_bf16 0.35449, ΔL 0.29680 (|difference| 0.0577), pred_jvp 0.34226, ΔL_fd 0.03425 (0.1 × pred 0.03545).
- ΔL_own 0.1547 against pred_jvp_own 0.2288. N12's bf16 measurement was −0.190.
- Exemplar: "…https://meetings.webex [[.com]] /collabs/files/playRecording…" (10 exemplars).
- Five largest changes in window 1626 (i0 1160):

| j | change (nats) | token |
|---|---|---|
| 1162 | +0.1333 | "abs" |
| 1160 | +0.1053 | "/c" |
| 1163 | +0.0662 | "/files" |
| 1194 | −0.0259 | "E" |
| 1164 | −0.0240 | "/play" |

These are the URL tokens after the exemplar. The 10 % step agrees with a tenth of the prediction to 3 %; the zero
ablation falls 16 % short.

Over all 40 pairs, the largest gap is L10N8869+ in L10N10529-:c3 (T_12, rank 1):

- pred_bf16 −0.39909, ΔL −0.28345, pred_jvp −0.40831, ΔL_fd −0.04004 (0.1 × pred −0.0399).
- ΔL_own +0.0725 against pred_jvp_own +0.0728. N12's bf16 measurement was −0.143.
- Exemplar: "…That they had the Rongor [[ongo]] script…".
- Its own top changes are in window 3117 at j 810, 811, 814, 804 and 808 (i0 802), largest +0.184.
- The own loss and the 10 % step match the prediction. The zero ablation's window sum is 29 % short, the second-order
  pattern of (c).

## 10. What I could not verify

- **K-A's bf16 rows** are not banked. I read the gate (40 of 40 exact) and recomputed its failing input's 0.0166 from
  N12's records.
- **I-FWD's and I-JVP's second passes** are not banked. I checked their failing inputs from the npz, plus two other
  identities (same-window clean rows across units bitwise; tangents exactly 0 before i0).
- **h, the probe's v and the forward itself** are not banked or not rerun. `sum_abs_h` is not recomputed; its defect
  (§8 h1) is shown from the code and the index sets only.
- **pred_bf16 comes from N12's banked records.** N12's reader recomputed it from H and G within 6.7e-8. I did not
  repeat that.
- **u_matched (H_signal's test floor)** is taken from N12's records. The matched poles' S is not banked.
- **The decoded token text** in the raw examples is the job's. I did not open the corpus. There is no token text for
  non-carrier poles.
- **slot_control, the timings and the forward-mode memory** are read from `metrics.json`, not measured.
- **The spec's §12 prediction grid** is left for the result document.

## 11. Files

`studies/next/n2v/read_n2v/`:

| file | role |
|---|---|
| `r1_recount.py`, `r1_recount.json` | every pair from the npz; G-CAUSAL from the raw rows; identities; per-pair recount (612 pairs) |
| `r2_verdict.py`, `r2_verdict.json` | the four validations, G-ROOM, ceiling, inspection, fd copy test, labels, branch (own code) |
| `r3_run_registered.py` | runs `infra.pg.n2v verdict` under `python -I`; output in the scratchpad |
| `r4_compare.py`, `r4_compare.json` | the registered verdict against mine, 81 fields |
| `r5_reread.py`, `r5_reread.json` | N12's gradient statistics with the fp32 ΔL (own code), against the registered re-read |
| `r6_checks.py`, `r6_checks.json` | checks (a)–(h) |
| `r7_examples.py`, `r7_examples.json` | the test raw examples against the npz |
| `r8_gates.py`, `r8_gates.json` | every gate, its failing input, cross-checks against the npz and the registered constants |

Run: `$SP/m1_venv/bin/python -I <script> …` with paths as arguments (each script's docstring has its command line);
numpy 1.26.4, Python 3.12.12.
