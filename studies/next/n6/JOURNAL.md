# N6 journal

One line per event, UTC. Spec: `studies/next/07_PICKS/N6_SPEC.md`. Offline gate: `studies/next/07_PICKS/N6_OFFLINE_GATE.md`.
Infrastructure and cost: `infra/aws_pg/operations/2026-11-06_next_n6_edit.md`.

- 2026-11-06T15:10Z The head's review: the four design calls accepted as written (the sign-restricted functional edit that
  equals E1's zero replacement at κ = 1; the fp32 forward throughout; the two-stage pairing with L1 banking the column
  norms and energies; the weak ceiling, stated). Nothing in the draft changes. L1 authorized as §9.5 writes it.
- 2026-11-06T15:11Z–15:13Z Dry run rerun on `743d22324`: `N6_DRYRUN_OK`, 78 checks, 0 failed, 1 min 41 s; every CHECK
  line equal to the draft's run and `dryrun_summary.json` equal (`offline_gate/dryrun_freeze1.log`). `gate_sides.py`:
  `GATE_SIDES_OK`, output equal to the committed `gate_sides.json`.
- 2026-11-06T15:13:49Z Annotated tag `n6-freeze-1` (tag object `9f9c3b040`) on `743d22324`. Commit `e05b11ccc` records it
  in spec §0 and adds the rerun log; `git diff n6-freeze-1 HEAD -- infra/` is empty.
- 2026-11-06T15:11Z–15:14Z Pre-launch checks of §9.1: `pg ps` lists two instances not from N6 (g6e.2xlarge `pipeline`,
  g6e.4xlarge `bytegraft_round1`); `pg audit` 0 reaper-less. `results/next_n6/` held no object version and no delete
  marker. Inputs uploaded 15:14:17Z–15:14:20Z and downloaded back: `pairs.json` `adeca764…` and `e1_known.json`
  `79da230c…` equal to `n6_edit.INPUT_SHA["L1"]` before and after (2 of 2); `L1-01/steps.json` byte-equal to
  `infra/aws_pg/steps/next_n6_L1.json` (job stop rule `--budget-min 47`).
- 2026-11-06T15:18:53Z L1 `L1-01` launched on `n6-freeze-1`, try 2 (try 1 at 15:14:50Z failed on `VcpuLimitExceeded`):
  `i-075cff985f3d39f3e`, g6e.2xlarge (one L40S), us-east-1a; hard kill `--budget-min 53`, reaper deadline 16:21:53Z.
  Watch every 120 s on `metrics.json`, `pipeline_status.json` and `STATUS`, never `job.log`.
- 2026-11-06T15:26:27Z G_SAME (torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S) and G_REV (`373bad25…`) pass. **K1
  PASS**: job_fall 0.6742177606 against E1's 0.6742178218 (difference 6.1e-8); the 12 activations before and after
  equal E1's banked values (max difference 0.0). Failing input (the cross arm −0.0023431) fails. **K2 PASS, equal to
  the last digit**: 2.675372037760417; failing input (T_14's 2.714762) fails. Download 14.61 GB at 58.0 MB/s.
- 2026-11-06T15:29:08Z **G-POLE FAILED; the job stopped (STATUS exit=1 at 15:29:25Z).** Worst pole L08N3846+ 0.01506
  of its largest banked activation, tolerance 0.01; 6 of 197 poles above it. The values fall in two groups: 156 poles at
  ≤ 1e-6 (g and u equal to the bank's), 41 at 1e-4 to 0.015 (bf16 level). Failing input (neuron n + 1) smallest 0.923,
  fails. K2 on full harvest-shaped windows is exact; G-POLE forwards right-padded prefixes cut at each exemplar
  (`reuse_ablate.readout`); the likely cause is that these shapes select GPU kernels that round bf16 differently
  (hypothesis; g, u and a per exemplar are not recorded). The 1e-2 tolerance assumed bit-identity. **L1-01 DOES NOT
  COUNT.** Not checkable from banked data: the offline pass side was bank against bank (0.0 by construction), and the
  banked prefix-path known numbers (E1, E4) report a median. Nothing after G-POLE ran: no norm, energy, fp32 number or
  edit. Cause in `diag/L1-01/cause.md`; `job.log` copied once to `diag/L1-01/`. No relaunch without the head.
- 2026-11-06T15:29:36Z EC2 `shutting-down` (instance-initiated). Uptime 0.176–0.179 h, $0.39–0.40 on g6e.2xlarge.
- 2026-11-06T15:35:31Z EC2 `terminated`. `pg ps` 15:35:37Z: 2 running, none from N6; `pg audit` 0 reaper-less.
- 2026-11-06T15:40Z The head's decision on L1-01 (second executor from here): amendment A1, G-POLE's tolerance 0.05 on the
  same statistic (3.3 times room over L1-01's worst pass-side value 0.01506, 18 times below the failing input's 0.9234; the bar
  set after seeing one run's pass side, said so); the job records g, u and a per pole and exemplar beside the bank's a
  (`g_pole.npz`) and the peak GPU memory per phase; the stage-2 file is `pairs_L2.json`; the rule for gates that compare two
  computation paths goes into `N6_OFFLINE_GATE.md` §6; then freeze 1a and L1-02. L2 only on the head's word.
- 2026-11-06T15:41Z–15:47Z A1 written: `n6_edit.POLE_TOL` 5e-2; `g_pole.npz` and `measured.peak_mem_gib` (nine phases) in the
  job; five dry-run checks added. All 197 L1-01 values copied to `diag/L1-01/A1_g_pole_197.tsv` (111 exactly 0, 45 in (0, 1e-6],
  1 in (1e-4, 1e-3], 15 in (1e-3, 5e-3], 19 in (5e-3, 1e-2], 6 in (1e-2, 0.0151], none above 0.05). Dry run 15:44:51Z–15:46:39Z:
  `N6_DRYRUN_OK`, 83 checks, 0 failed; the 78 earlier CHECK lines equal to freeze 1's apart from the tolerance printed in
  G_POLE's 0.7 × a line; planted G-POLE 0.030000 passes, 0.060000 stops the job (`offline_gate/dryrun_freeze1a.log`).
  `gate_sides.py`: `GATE_SIDES_OK`, `gate_sides.json` differs from freeze 1's only in G_POLE's `tol` (0.05). Spec §4, §9.4,
  §9.5, §14; offline gate §2, §3, §6. The amendment also names K3, G-ZN and G-CACHE as gates that compare two paths without a
  control pair (unchanged by A1; L1-02 is G-ZN's and G-CACHE's first GPU measurement).
- 2026-11-06T15:48:33Z Annotated tag `n6-freeze-1a` (tag object `ba2674cc5`) on `e5c5f21f2`. Recorded in spec §0.
- 2026-11-06T15:48Z Pre-launch checks: `pg ps` 2 running, none from N6 (`i-024d55ad5aef03233` g6e.2xlarge pipeline,
  `i-02e60cfd0669e2669` g6e.4xlarge bytegraft_round1); `pg audit` 7 reapers, 5 orphans, 0 reaper-less.
  `results/next_n6/L1-02/` held no object version and no delete marker. The L1 inputs on S3 re-downloaded: sha256 equal to
  `INPUT_SHA["L1"]` (2 of 2). `L1-02/steps.json` uploaded 15:48:54Z and downloaded back, byte-equal to `next_n6_L1.json`.
- 2026-11-06T15:50:41Z L1 `L1-02` launched on `n6-freeze-1a` (HEAD `4de9b40eb`), try 1 (15:49:28Z–15:50:42Z):
  `i-04e9ceb7f4c8973f2`, g6e.2xlarge (one L40S), us-east-1d; hard kill `--budget-min 53`, reaper deadline 16:53:41Z. Watch every
  120 s on `metrics.json`, `pipeline_status.json` and `STATUS`, never `job.log`.
- 2026-11-06T16:00Z Head note received during L1-02 (the user's rule of 2026-11-06 10:00 CST): if the fp32 forward with the kept
  residuals does not fit the L40S's 48 GB at the registered batch, or any step's peak memory exceeds it, precision and the
  registered settings are not changed to fit; the shortfall is recorded with its numbers and reported to the head as a lane
  decision (such a step can run on a Vast H100 with more memory). This replaces spec §0.3 item 3's "rerun with `--batch 2`".
- 2026-11-06T15:58:49Z L1-02 bf16 phase: G_SAME, G_REV pass; K1 0.6742177605628967 (as L1-01; a_before, a_after equal to E1's,
  max difference 0.0); K2 2.675372037760417 (exact); **G_POLE PASS at 0.05**: value 0.015059128128922768, 6 poles above 0.01,
  failing input smallest 0.9234; all 197 per-pole values equal to L1-01's bit for bit (a second instance, us-east-1d). Download
  14.61 GB at 95.9 MB/s, load 209.0 s. Peak memory so far (GiB): load_K1 13.94, K2 18.195, G_POLE 15.384.
- 2026-11-06T16:01:57Z–16:05:28Z L1-02 fp32 phase: G_FP32 pass; **K3 PASS** 2.675162242635091 against bf16 2.675372037760417
  (difference −2.1e-4; failing input, targets one position off, 10.093); base pass 1 113.4 s; **G_ID PASS** (0.0); base mean
  loss on the 96 band-3 windows 2.0958845506. Peak memory (GiB): bf16_probes 16.664, cast 27.96, K3 31.023, pass1 31.024,
  pass2 31.024. The bf16 probes moved the loss by 3.886e-5 and 5.781e-6 nats per token.
- 2026-11-06T16:08Z–16:18:57Z The progression: 13 of 13 edits, nothing parked; 3.20 s per edit-layer unit; peak memory 36.201
  GiB (no step above the 48 GB of the L40S; the head's memory note did not apply). **G_EPS PASS**: fp32 probes 1.221e-8 and
  5.653e-9 nats per token (bound 1e-6; bf16 side above it, fails as required). **G_NORM PASS** (largest E⁺ relative error
  1.9e-8). **G_SAMPLES PASS** (10). **G_ZN FAILED**: the zero edit's ΔL −3.015e-11, not bitwise equal to base. **G_CACHE
  FAILED**: largest per-token difference 9.537e-7 nats. Both are post-edit gates (the job does not stop on them); `STATUS` exit=0
  at 16:19:20Z, `N6_JOB_OK`. Likely cause: the progression applies the head to one window, the base pass to a batch of 4
  (hypothesis; `diag/L1-02/record.md`). Not checkable from banked data before L1-02 (no GPU fp32 forward banked; CPU bitwise).
- 2026-11-06T16:19:51Z EC2 `shutting-down` (instance-initiated). Uptime 0.478–0.486 h, $1.07–1.09 on g6e.2xlarge.
- 2026-11-06T16:20Z–16:30Z Closing offline. Banks pulled to `banks/L1-02/`; `job.log` copied once with the run's records to
  `diag/L1-02/`. G-POLE join (`diag/g_pole_join.py` → `diag/L1-02/g_pole_join.json`): the 41 poles above 1e-6 are exactly the
  poles whose recomputed g or u differs from the harvest's bf16 value, by 1–2 bf16 steps; the 156 others have g and u equal.
  Raw examples filled (`samples/raw_examples_L1.{json,md}`): ΔL of the ten column edits −4.87e-6 to +9.76e-6 nats per token.
  **Stage 2** (`n6 final` → `pairs_L2.json`, `stage2_check.py` → `stage2_check.json`): 45 pairs (28 freeze, 17 test, 25
  layers), calipers ln max|a| 0.10, ln density 0.50, ln ‖w‖ 0.10, ln E⁺ 0.25; ρ_pair +0.022 / −0.067 (ln max|a|, ln density),
  SMD +0.002 / +0.015 / +0.061 / −0.034 (with ln ‖w‖, ln E⁺); balanced, at least 40. Failing inputs: the 176 unmatched pairs
  fail; the planted age-dependent column norm stops with "L2 does not launch". 27 of the 74 stage-1 pairs kept. Not gated:
  ρ_pair(age, ln ‖w‖) +0.111. The L1-edited poles in `pairs_L2.json` are all in freeze pairs. `projection_l1.py` now plans L2 on
  `pairs_L2.json` (it read the stage-1 `pairs.json`): measured rate 29.0 TFLOP/s; L2 281 edits, center 4.466 GPU-h, high 5.582,
  hard kill 436 min, job `--budget-min` 392 (`projection_L1.json`). Room at 17 test pairs: FLAT unreachable (Wilson at 0.5
  [0.285, 0.715]); SUPPORTED needs 13 wins. G-RES preview on the ten L1 edits: bound 1.075e-8 against F_eps 1.221e-8 (fails by
  1.14). L2 not launched; the head decides G-ZN/G-CACHE, room and G-RES before freeze 1b. L2 commands in `STATE.md`.
- 2026-11-06T16:25:32Z EC2 `terminated` first seen (still `shutting-down` at 16:24:57Z). `pg ps` lists 2 running, none from N6;
  `pg audit` 0 reaper-less.
- 2026-11-06T16:30Z The head's decisions (third executor from here): A2, G-ZN and G-CACHE get tolerances frozen from L1-02's values
  under the two-path rule (G-ZN \|ΔL\| ≤ τ_eps 1.22e-7; G-CACHE mean ≤ τ_eps and largest ≤ 1e-5), failing inputs kept, the job to
  record both quantities and the two paths' batch shapes; A3, the test half becomes the 28-pair half before any outcome exists, the
  pairs L1's edits touched kept in the freeze half; A4, read G-RES, compute its preview, state the cause, fix it under the two-path
  rule or another stated reason, or stop before freeze 1b and report. L2 authorized if A2–A4 resolve as described and the dry run passes.
- 2026-11-06T16:35Z–16:47Z A3 written (`n6.half_of`, `final_pairs(..., l1_edited)`). Four stage-2 pairs are L1 pairs with the same two
  poles and c, so their D_col is banked: P003, P008, P025, P039 (stage-1 P002, P020, P039, P069); all have even early neurons, so the
  swap alone would put them in the test half. Kept in freeze: **24 test, 21 freeze**, not 28 / 17. `pairs_L2.json` regenerated (sha256
  `2d206ffc…`; was `46627827…`), pairs and balance unchanged; `stage2_check.json`, `projection_L1.json` rerun (projection unchanged).
  Room at 24: FLAT only at exactly 12 wins, SUPPORTED 17, REVERSED at most 7; H_adj 16 concordant test pairs. A2 written (`zn_gate`,
  `cache_gate`, `TAU_EPS_L1` 1.2210622e-7, `CACHE_MAX_TOL` 1e-5; G-CACHE's failing input scores each window from the next window's
  final residual; a pre-hook on `lm_head` records the input shapes per phase). `INPUT_SHA["L2"]` registered; `next_n6_L2.json`
  `--budget-min 392`.
- 2026-11-06T16:38Z–16:46Z Dry run: first two runs failed one check each (A16 compared floats bitwise across numpy 2.4.3, which wrote
  `pairs_L2.json`, and the venv's 1.26.4, 8th digit; a planted 2e-7 offset was rounded away in fp32); both checks fixed in the dry run,
  not in the job. Third run `N6_DRYRUN_OK`, 99 checks, 0 failed, 1 min 55 s (`offline_gate/dryrun_A2A3.log`); the 83 earlier CHECK
  lines identical to freeze 1a's. Planted: zero edit 5e-8 passes, 5e-7 fails; largest difference 5e-6 passes, 5e-5 fails; a uniform
  2e-7 offset fails on the mean. `gate_sides.py` `GATE_SIDES_OK`, output equal. pyflakes clean.
- 2026-11-06T16:40Z–16:50Z G-RES (A4) read and previewed (`diag/g_res_preview.py` → `diag/L1-02/g_res_preview.json`): q10 of the ten
  L1 \|ΔL_col\| 1.0753e-6, bound 1.0753e-8, F_eps 1.2211e-8, fails by 1.136 (q10 / F_eps 88.1, the rule needs 100). Cause: small edits
  (median \|ΔL_col\| 2.2e-6 against the spec's predicted 1e-5 to 1e-3), not two paths (the zero edit runs the probe's path; its ΔL
  −3.0e-11 is 0.25 % of F_eps). No defect, so no fix under the two-path rule; changing the bar would set it after seeing it fail.
  Eval-set noise on the five L1 pairs: window-bootstrap sd of D_col 3.5e-6 to 1.16e-5 (median 6.6e-6) against \|D_col\| 2.1e-6 to
  6.0e-6; none resolved at 2 sd; 9 of 10 single edits within one sd of zero. **Stopped before freeze 1b**; A2 and A3 committed without
  the tag; the definition, numbers, options and proposal (report G-RES rather than gate it, the power preview stated) in `STATE.md`.
  L2 not launched; nothing uploaded to S3.
- 2026-11-06T17:00Z The head's decision on A4 (fourth executor from here). Part 1: G-RES's factor becomes 50 (q10 / F_eps was 88 where
  the rule asked 100; the cause is small edits, median \|ΔL_col\| 2.2e-6, not a path effect), said plainly as set after the preview; the
  planted failing input stays. Part 2, power: (a) 1,024 eval windows of the registered order; (b) G-POWER at freeze 2, τ_s ≤ 0.5 × the
  median \|D_col\| over the freeze pairs, else UNDECIDED_NO_POWER before the test half is read; (c) the positive control in 8 layers (0, 4,
  …, 28) with its gate at 6 of 8, the random-direction and raw-example edits kept. The L2 projection from L1-02's rate, the memory per
  phase, the hard kill at 1.3 times, $2.24 per hour; dry run, tag `n6-freeze-1b`, §0, the manifest, the launch commands in STATE. No
  launch: the head launches L2 after N13's S4 is on the GPUs.
- 2026-11-06T17:00Z–17:14Z A4 written (spec §14 A4). Eval set: the registered rule at 1,024 (`n6.eval_windows(1024)`, 10,045 to 24,663,
  list sha256 `5822191…`; 32 of the 96 L1 windows among them). Not taken: the first 1,024 consecutive band-3 windows (about 3.5 of band 3's
  50 directories). `pairs_L2.json` regenerated by `n6 final`: only `eval_windows`, `w_eval` and the new `inst_windows` differ from A3's file
  (sha256 `7e3a9948…`, was `2d206ffc…`); `stage2_check.json` rerun (only that sha256 changed); `INPUT_SHA["L2"]` updated. Two consequences
  in the job: G-INST reads E⁺ in its own pass over L1's 96 windows (over 1,024 windows E⁺ differs and the 1e-5 comparison would fail
  without a defect); the kept residual of 1,024 windows (34.4 GB) is kept in host memory and goes to the GPU in chunks of 96 windows (on the
  GPU together h_l and h_(l+1) would take 64 GiB beside the 27.2 GiB model); the job asserts MemAvailable ≥ 40.0 GiB at start. Preview
  (`diag/a4_preview.py` → `diag/L1-02/a4_preview.json`): G-RES at the factor 50 passes (F_eps / bound 0.568); G-POWER on the five L1 pairs
  1.42 at 96 windows (fails), 0.435 with τ_s scaled to 1,024 independent windows and \|D_col\| held (passes; optimistic, said so).
  `g_res_preview.py` pinned to freeze 1's factor; its rerun writes the committed JSON byte for byte.
- 2026-11-06T17:10Z–17:14Z L2 projection (`projection_l1.py` → `projection_L1.json`): 209 edits (24 positive control, 2 + 2 probes, 1 zero,
  180 pair edits); edits 121,345 s at 29.01 TFLOP/s; L1's window-dependent seconds 402 s × 10.67; G-INST pass 113 s; host-device copies
  384 s (6 GB/s assumed). Center 35.20 GPU-h, high 44.01; $78.86 / $98.57 at $2.24 per hour; hard kill 3,433 min; job stop rule 3,089 min
  (`next_n6_L2.json`). A3's: 4.466 / 5.582 GPU-h. The trim pays 1,296 of 5,287 edit-layer units (24.5 %); the 1,024 windows multiply each
  edit by 10.67; net 7.9 times A3's center; §9.6's 12-GPU-h rule fires (A4 is the head's decision under it). GPU peak per phase projected at
  L1-02's measured values (progression at most 36.201 GiB, 38.9 GB, under 48 GB); host 36.3 GiB of 64 GiB on g6e.2xlarge. No lane decision
  on memory.
- 2026-11-06T17:12Z–17:18:11Z Dry run. First run 115 of 117: the dry run's planted G-PC records had col above rand in every layer (the
  losing layers were scaled, not swapped); fixed in the dry run, not in the job. Second run 17:15:42Z–17:18:11Z: `N6_DRYRUN_OK`, 117 checks,
  0 failed, 2 min 29 s (`offline_gate/dryrun_A4.log`). Planted: G-POWER 0.4 passes and the test half is read, 0.6 fails with
  UNDECIDED_NO_POWER and no test pair read; G-PC 6 of 8 passes, 5 fails; G-RES 0.015 × q10 passes, 0.05 × q10 fails; one chunk and two
  chunks give equal records; G-INST read on L2's eval set stops the job. 92 of the 99 earlier CHECK lines identical, 7 changed (listed in
  spec §14 A4). `gate_sides.py` `GATE_SIDES_OK`, output equal. pyflakes clean.
- 2026-11-06T17:24Z–17:28Z Dry run rerun on the committed tree (`eba7035cb`): `N6_DRYRUN_OK`, 117 checks; every CHECK line and
  `dryrun_summary.json` equal to the 17:15Z run. 2026-11-06T17:28:17Z annotated tag `n6-freeze-1b` (tag object `4cc2251e0`) on
  `eba7035cb`; recorded in spec §0. The L2 upload and launch commands (hard kill `--budget-min 3433`; job stop rule 3,089 min in
  `next_n6_L2.json`) are in `STATE.md`. Nothing uploaded to S3; L2 not launched (the head's, after N13's S4 is on the GPUs).
- 2026-11-06T17:35Z The head's decision A5 (fifth executor from here). L2 at 1,024 windows projects to 35.20–44.01 GPU-h and G-POWER may
  still fail (the preview's 0.435 holds \|D_col\| at its 96-window size). G-POWER is a check on the freeze half, so it is paid for first:
  L2a (the freeze half, the positive control, the probes, the raw-example edits, every gate; its records decide freeze 2) and L2b (the
  test half, the gates that do not depend on the half re-run, L2a's eval set and pairs file asserted by sha256), launched only after
  freeze 2 is tagged with G-POWER passed. Smallest change; record formats kept; both manifests and projections; dry run with the split;
  tag `n6-freeze-1c`. No launch: the head launches L2a when a GPU slot is free after N13's S4.
- 2026-11-06T17:36Z–17:52Z A5 written (spec §14 A5). One argument, one pairs file: `n6_edit --half freeze|test` at stage L2,
  `n6.plan(half=)`; `pairs_L2.json` and `INPUT_SHA["L2"]` unchanged. L2a 113 edits (PC 24, PC probes 2, freeze pairs 84, pair probes 2,
  zero 1), L2b 99 (test pairs 96 and the same 2 pair probes and zero edit, which L2b's G-EPS and G-ZN read; the head listed both gates
  for L2b). L2b reads L2a's `metrics.json` (`--against`) before its first gate and stops unless it is a complete freeze-half launch with
  nothing parked and the same eval windows, eval tokens and pairs content (`pairs_sha256`, new in `metrics.json`). Banks
  `banks/L2a-01/`, `banks/L2b-01/`; `n6 freeze --bank`, `n6 verdict --bank --bank` through `n6.read_banks` (L2b's gates under `L2b:`).
  Projection (`projection_L1.json` → `L2a`, `L2b`, `A5_sum`; A4's `L2` block unchanged by the rerun): L2a 19.054 / 23.818 GPU-h ($42.68 /
  $53.35), hard kill 1,858 min, job stop rule 1,672 min; L2b 18.239 / 22.799 ($40.86 / $51.07), 1,779 / 1,601 min; sum 37.293 / 46.617 =
  A4's 35.204 + one setup 1.498 + the 3 edits in both 0.591 (asserted in the script). Manifests `next_n6_L2a.json`, `next_n6_L2b.json`;
  `next_n6_L2.json` removed (kept at `n6-freeze-1b`).
- 2026-11-06T17:38:44Z–17:48:47Z Dry run. First run `N6_DRYRUN_OK`, 135 checks, 0 failed. The offline-gate record then said the two
  manifests parse in the dry run, which was only a manual check; a check was added (`n6_edit.check_args`, the manifests' `--budget-min`
  equal to the projection's job stop rules). Second run 17:46:00Z–17:48:47Z: `N6_DRYRUN_OK`, 136 checks, 0 failed, 2 min 48 s
  (`offline_gate/dryrun_A5.log`); the 117 A4 CHECK lines identical, 19 added. Planted, each stops: an L2b with another eval set (before
  G-SAME, nothing written), the verdict given such a bank, an L2b against an L2a that parked, `n6 freeze` given L2b's bank, the verdict
  with G-POWER passed and L2a alone, stage L2 without `--half`, `--half test` without `--against`. The tiny L2a and L2b records equal the
  one-launch records field by field; freeze on L2a and the verdict on both equal the one launch's. `gate_sides.py` `GATE_SIDES_OK`,
  output equal. pyflakes clean.
- 2026-11-06T17:50:54Z–17:54:17Z Dry run rerun on the committed tree (`b94dbd8e1`): `N6_DRYRUN_OK`, 136 checks; every CHECK line and
  `dryrun_summary.json` equal to the 17:46Z run; `gate_sides.py` output equal to the committed `gate_sides.json`. 2026-11-06T17:54:17Z
  annotated tag `n6-freeze-1c` (tag object `843f66877`) on `b94dbd8e1`; recorded in spec §0. The L2a upload and launch commands (hard
  kill `--budget-min 1858`; job stop rule 1,672 min in `next_n6_L2a.json`) and L2b's (after `n6-freeze-2` with G-POWER passed; hard kill
  1,779 min, job stop rule 1,601 min) are in `STATE.md`. Nothing uploaded to S3; nothing launched (the head's, when a GPU slot is free
  after N13's S4).
- 2026-11-07T20:00Z The head authorized the L2a launch (sixth executor from here): one g6e.2xlarge slot free, `--type g6e.2xlarge` only.
  Tree clean at `20945a7ec`; `git diff n6-freeze-1c HEAD -- infra/ studies/next/n6/pairs_L2.json studies/next/n6/inputs/` empty.
- 2026-11-07T20:02:54Z–20:05:39Z Dry run on `20945a7ec`: `N6_DRYRUN_OK`, 136 checks, 0 failed; the 136 CHECK lines and
  `dryrun_summary.json` equal to `dryrun_A5.log`'s run (`offline_gate/dryrun_L2a.log`, committed as `d5554e2ad`).
- 2026-11-07T20:04Z–20:06:19Z `pg ps` 3 running, none from N6; `pg audit` 0 reaper-less. `inputs_L2/`, `L2a-01/`, `L2b-01/` held no object
  version and no delete marker. Uploaded and downloaded back: `pairs_L2.json` `7e3a9948…`, `e1_known.json` `79da230c…` (2 of 2 equal to
  `INPUT_SHA["L2"]`); `L2a-01/steps.json` byte-equal to `next_n6_L2a.json` (job stop rule `--budget-min 1672`).
- 2026-11-07T20:06:42Z **L2a-01 launched** on `d5554e2ad` (try 1): `i-00e9ff97327126cd4`, g6e.2xlarge (one L40S), us-east-1d; hard kill
  `--budget-min 1858`, reaper 2026-11-09T03:14:42Z. `PG_GIT_SHA` `d5554e2ad921` = HEAD; the code tarball `4f431c1c67b1` holds HEAD's
  tracked files unchanged (2,079 equal, 0 differ). Watch every 120 s on `metrics.json`, `pipeline_status.json` and `STATUS`, never
  `job.log`; gate records banked at first sight in `banks/L2a-01/live/`.
- 2026-11-07T20:11:01Z–20:14:17Z L2a-01 bf16 phase (`banks/L2a-01/live/`): `half` freeze, `w_eval` 1,024, `eval_windows_sha256`
  `5822191728d4…` (registered), `pairs_sha256` `c837fe89…` (recomputed locally from `pairs_L2.json`: equal), host MemAvailable 59.67 GiB
  (needs 40.02). G_SAME, G_REV pass; **K1 PASS** 0.6742177605628967 (as L1-02; a_before, a_after equal to E1's, max difference 0.0);
  **K2 PASS** 2.675372037760417 (exact); **G_POLE PASS** 0.010999 (L25N10599−; 3 of 115 poles above 0.01; failing input smallest 0.9234);
  all 115 per-pole values bit-equal to L1-02's (a third instance). Download 112.9 MB/s, load 188.7 s.
- 2026-11-07T20:27:06Z–21:07:43Z L2a-01 fp32 phase: bf16 probes −3.054e-6 and +6.891e-6 nats per token (G-EPS's failing side above 1e-6);
  G_FP32 pass; **K3 PASS** 2.675162242635091 (as L1-02; off by one 10.093 fails); base pass 1 over 1,024 windows 1,213.5 s; **G_INST
  PASS** value 0.0 (failing input smallest 0.00547); the positive-control poles at 1,024 windows L00N4051−, L04N4791+, L08N3661−, L12N5682+,
  L16N6351+, L20N9155−, L24N7030−, L28N223+ (layer 8's differs from L1's 96-window choice L08N1033−); **G_ID PASS** (0.0); base mean loss
  2.0989741734. Peak memory at most 31.031 GiB.
- 2026-11-07T22:15:53Z L2a-01 layer 0's 4 edits written: 30.849 s per edit-layer unit (brief: about 34 s; `projection_L1.json` L2a edit
  seconds per unit 30.40). Projected end about 2026-11-08T14:55Z, about 18.8 GPU-h from launch; the job's stop rule (elapsed + projected
  > 90,288 s) has 22,856 s of room. Nothing parked.
- 2026-11-08T12:05Z The head checked L2a-01 (n_done 69; `projection_remaining_s` 13,619 → end about 15:47Z). The poller was live (pid
  73090). The job's `projection_remaining_s` is set at the start of a layer and written after it, so it is one layer stale: at n_done 69
  it still held layer 20's 98 units. Corrected then: 343 units × 30.898 s → end about 14:57Z plus G-CACHE and the records. Realized:
  `finished_utc` 15:02:13Z (the rate rose to 31.002 s per unit over the last layers).
- 2026-11-08T15:02:13Z **L2a-01 complete** (`banks/L2a-01/metrics.json`): 113 of 113 edits, nothing parked, `complete` true; `STATUS`
  exit=0 at 15:02:21Z. Gates after the edits: **G_CACHE PASS** (mean −2.710e-10, largest 1.907e-6; tolerances 1.221e-7 and 1e-5; not
  bitwise; failing input, another window's final residual, mean 12.58, largest 42.93, fails); **G_EPS PASS** F_eps 3.365e-9 (four fp32
  probes: PC00 −3.365e-9, P005_late +1.857e-9, P015_late −9.45e-10, PC31 −1.95e-10; bound 1e-6; bf16 side −3.054e-6, +6.891e-6 fails);
  **G_ZN PASS** (zero edit ΔL 2.710e-10, tolerance 1.221e-7; failing input PC00_col 3.968e-4 fails); **G_ID** failing side fails
  (PC00_col not bitwise equal); **G_NORM PASS** (largest E⁺ relative error 1.07e-8); **G_DOSE PASS** (col > col_half in 8 of 8; swapped
  0); **G_SAMPLES PASS** (10). **G_PC FAILED: col > rand in 2 of 8 positive-control layers (need 6), and its failing input passed: rand
  > col in 6 of 8** (τ 3.365e-8). Under spec §4 and §5 item 6 every label of the verdict reads DOES NOT COUNT, whatever L2b would show.
  Progression peak memory 34.859 GiB. `colnorms.npy` sha256 equal to L1-02's.
- 2026-11-08T15:05Z The PC rows (`edits.jsonl`, ΔL in nats per token, col / rand / col_half): L0 L00N4051− 3.968e-4 / 5.188e-3 /
  1.013e-4; L4 L04N4791+ 1.521e-3 / 3.145e-3 / 3.612e-4; L8 L08N3661− 8.666e-4 / 2.667e-3 / 1.263e-4; L12 L12N5682+ 7.069e-4 / 1.096e-2 /
  1.341e-4; L16 L16N6351+ 2.503e-2 / 5.837e-2 / 2.241e-4; L20 L20N9155− 4.979e-4 / 1.400e-3 / 1.059e-4; L24 L24N7030− 3.591e-2 / 8.027e-4 /
  1.993e-2; L28 L28N223+ 1.479e-3 / 1.334e-3 / 2.572e-4. Every edit moves the loss far above F_eps (smallest 1.0e-4, 30,000 F_eps): the
  instrument reads these edits; the ordering is the reverse of the registered expectation in layers 0–20. The realized residual change
  after layer l, R, is larger for rand than for col in all 8 layers (R_rand / R_col 2.51, 1.83, 1.82, 1.91, 2.06, 1.75, 1.34, 1.62); on the
  42 freeze-pair col and 42 rand edits the medians are equal (60.82 against 60.68). One hypothesis, not tested: the post-feedforward
  RMSNorm rescales the MLP output, so removing the dominant pole's own write is partly undone by the norm, while a random direction of the
  same size is not. Row scored right (L24): `PC24_col` ΔL 3.591e-2 against rand 8.027e-4; its largest token is window 364, target
  position 2011, base loss 0.000383 → 20.79 after. Row scored wrong (L16): `PC16_col` ΔL 2.503e-2 against rand 5.837e-2; col's largest
  token window 458, position 1073, base loss 0.00111 → 18.58; rand's window 235, position 477, base 2.75e-5 → 19.21. The PC rows' contexts
  are not decoded (the job decodes only the raw-example rows; decoding needs the corpus and tokenizer, which are not on the laptop).
- 2026-11-08T15:05Z The freeze pairs' raw rows (`banks/L2a-01/samples/raw_examples.jsonl`). Scored right for H_age (the early edit costs
  more): P008, layer 22, L22N7990− (T_9) against L22N10231+ (T_12), c 0.9841: ΔL_col early +1.927e-6, late −3.319e-7 (D_col +2.258e-6);
  rand twins +3.022e-7, +2.362e-6. The early edit's largest token: window 735, position 1268, "…you can understand why they’re playing it
  this way: if Hindley and Car" → "ap", base loss 1.5528 → 1.9883. Scored wrong: P005, layer 1, L01N4337− (T_3) against L01N8975− (T_14),
  c 0.9625: early +1.580e-7, late +1.431e-6 (D_col −1.272e-6); rand twins +4.587e-7, −1.047e-7. The late edit's largest token: window 750,
  position 600, "…Duck's Garage is minutes from everything. We are driving distance from" → " you", base 4.6064 → 4.9402. Input as
  specified (band-3 windows with BOS), output the next-token loss of the model's own forward, the scored quantity that loss's change.
- 2026-11-08T15:06:12Z **Freeze 2** (`PYTHONPATH=. python3 -m infra.pg.n6 freeze --bank studies/next/n6/banks/L2a-01 --pairs
  studies/next/n6/pairs_L2.json --out studies/next/n6/thresholds.json`; sha256 `7dbdd22881089bc29f92a062c68c86cb5e86bcb68ae0d71ef1b6d1b9b59a8c95`):
  F_eps 3.365e-9, τ_eps 3.365e-8, τ_s 1.5096e-6, q10 of \|ΔL_col\| over the freeze half 2.438e-7. **G_RES PASS** (F_eps 3.365e-9 ≤ bound
  4.876e-9 = 0.02 × q10). **G_POWER FAILED**: τ_s 1.5096e-6 against the bound 6.363e-7 = 0.5 × median \|D_col\| 1.2726e-6; ratio 1.186
  (the bar is 0.5). Its registered failing input (the dry run's τ_s at 0.6 × median \|D_col\|) fails and 0.4 passes
  (`offline_gate/dryrun_L2a.log`). Recount with other code (the analytic sd of the per-window mean, not the bootstrap): median sd
  1.5027e-6, ratio 1.181. A4's preview projected 0.435: τ_s came out lower than its scaled 1.946e-6, but the median \|D_col\| is 1.27e-6
  against the 96-window 4.47e-6 the preview held fixed. The four L1 pairs at 1,024 windows: P003 +3.696e-6, P008 +2.258e-6, P025 −6.63e-7,
  P039 +1.37e-7 (at 96 windows all five L1 pairs read −2.1e-6 to −6.0e-6, each within 2 sd of zero). To pass at this \|D_col\|, τ_s must
  shrink by 2.37 times, about 5,760 windows (1,024 × 2.373²) if windows were independent (not a proposal; the head's call).
- 2026-11-08T15:08Z **DISCOVERY numbers** (freeze half, 21 pairs, `thresholds.json` → `discovery`): p_col 0.762 (16 wins, 5 losses at
  τ_eps), Wilson [0.549, 0.894]; p_rand 0.381 (8 wins), [0.208, 0.591]; R 0.095 (2 of 21 pairs with \|D_col\| > 2 τ_s); median D_col
  +9.82e-7 nats per token (mean +1.035e-6). Room at 24 test pairs: Wilson at 0.5 [0.314, 0.686]; FLAT reachable; SUPPORTED needs 17 wins.
  **Prediction for the test half.** Under the registered rule no test label is issued: G-POWER failed, so the verdict on L2a alone
  writes UNDECIDED_NO_POWER for H_age, H_adj and H_eff and reads no test pair; and because G-PC failed with its failing input passing,
  every label reads DOES NOT COUNT with UNDECIDED_NO_POWER kept as the realized label (`n6.no_power`). If the test half were read anyway:
  p_col between 0.55 and 0.80 (13–19 wins of 24), its lower Wilson bound within 0.1 of 0.5, H_age UNDECIDED or SUPPORTED, R below 0.2,
  p_rand within [0.3, 0.6]. The verdict is not run; it and L2b are the head's call.
- 2026-11-08T15:08:50Z Annotated tag `n6-freeze-2` (tag object `adb92679a`) on `037948fac` (banks, `thresholds.json`, journal, spec §0).
  The four entries above carried clock labels written ahead of time (15:10Z, 15:12Z) in the tagged commit; corrected here from the file
  times (`thresholds.json` written 15:06:12Z; the entries were written before the tag).
- 2026-11-08T15:04:22Z EC2 `shutting-down` first seen (running at 15:02:18Z; `STATUS` 15:02:21Z); `terminated` first seen 15:09:00Z.
  Uptime 18.927–18.961 h, $42.40–42.47 on g6e.2xlarge ($2.24 per hour), under the 19.054 GPU-h center projection. `pg ps` 15:07Z: no
  pg-lineage instance running; `pg audit` 0 reaper-less. N6 so far (L1-01, L1-02, L2a-01): 19.581–19.626 GPU-h, $43.86–43.96. L2b is
  not launched; the verdict is not run. Both wait for the head.
- 2026-11-08T15:15Z The head's decision: L2b is not launched (the frozen rule, spec §5 and §3.4, gives UNDECIDED_NO_POWER before any test
  pair is read, and G-PC's failing input passing makes every label DOES NOT COUNT). N6 closes. The registered next step for no power
  (§1.2: a larger eval set or more pairs) is not taken now: lower leverage than the rank test of the RL delta.
- 2026-11-08T15:12Z The registered verdict on L2a alone, as the script runs it with no L2b records (`PYTHONPATH=. python3 -m infra.pg.n6
  verdict --bank studies/next/n6/banks/L2a-01 --pairs studies/next/n6/pairs_L2.json --thresholds studies/next/n6/thresholds.json --out
  studies/next/n6/verdict_n6.json`): sha256 `b774450623aac2caab55a8fb482dbce9fc04a09e3ea11a9931d2586a41c33599`; headline verbatim
  `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT`; each realized UNDECIDED_NO_POWER; `failed_gates` `G_PC`,
  `G_PC:failing_input_passed`; `n_test_read` 0. **CLOSED 2026-11-08: DOES NOT COUNT (G-PC failed and its failing input passed; G-POWER
  failed, no test label).** STATE.md holds "What L2a measured" and the claims N6-1 (stands), N6-2 (stands), N6-3 (exploratory); spec §0 and
  §1.2 updated.
