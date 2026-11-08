# N12 journal

One line per event, UTC. Spec: `studies/next/07_PICKS/N12_SPEC.md`. Offline gate: `studies/next/07_PICKS/N12_OFFLINE_GATE.md`.
Infrastructure and cost: `infra/aws_pg/operations/2026-11-06_next_n12_score.md`.

- 2026-11-06T09:41Z Head's decision before freeze 1 (stated 2026-11-06 09:50Z) applied: G-VAL's registered bar rises from ρ ≥ 0.5 to ρ ≥ 0.7
  (`n12.VAL_RHO_MIN`). Unchanged: the bootstrap lower bound above the shuffle's 97.5th percentile, the ceiling 0.999,
  the sign-flip failing input. Below the bar the gradient labels do not count and are reported as exploratory. Marked
  in spec §3.5, §4, §6 and §12, and in `N12_OFFLINE_GATE.md` item 3. Everything else in the draft is accepted as
  written.
- 2026-11-06T09:42Z–09:49Z Dry run again on the changed code (`$SP/n12/dry_f1`): `N12_DRYRUN_OK`, 94 checks, 0
  failed, 6 min 37 s. Every CHECK line equals the draft's run; `dryrun_summary.json` equal as JSON
  (`offline_gate/dryrun.log`).
- 2026-11-06T09:49:01Z Commit `f60249949` (decision, dry-run log). 09:49:08Z annotated tag `n12-freeze-1` (tag object
  `8a7dc51a5`) on it. The next commit records the tag in spec §0.
- 2026-11-06T09:49Z Pre-launch checks of §9.5 S1: `pg ps` listed three instances not from N12 (two g6e.2xlarge
  pipelines, one r7i.2xlarge); `pg audit` 0 reaper-less. `results/next_n12/` was empty. Inputs uploaded and read back:
  `units.json` and M2's two ranks files 3 of 3 equal to the registry; M2's 16 exemplar files 16 of 16; `steps.json`
  byte-equal.
- 2026-11-06T09:53:02Z L1 `score-01` launched on `n12-freeze-1`, try 2 (try 1 at 09:49:56Z failed on
  `InsufficientInstanceCapacity`): `i-01be391d168b7c4a0`, g6e.2xlarge (one L40S), us-east-1d; hard kill
  `--budget-min 182`, reaper deadline 13:05:02Z. Watch every 120 s on `metrics.json`, never `job.log`.
- 2026-11-06T09:54:30Z G_SAME (torch 2.12.0+cu130, transformers 5.9.0), INPUTS (no mismatch), G_CORPUS (`2aca80ca…`)
  pass; each failing input failed (job started 09:54:12Z).
- 2026-11-06T09:57:11Z **K0 PASS, bit-identical**: the 10 carriers' B recomputed at T_14 equal M2's banked pre-role B
  (max |ΔB| 0.0, tolerance 1e-6); their 39 activations equal M2's vectors (p99 0.0, exact share 1.000). Failing input
  (M2's birth-role B of the same carriers): smallest |ΔB| 8.995e-04, fails. Prior for bit-identity was 0.9. T_14 G-REV
  `cb286dcd…`, G-CFG no differing key, G0 max ratio 0.00478 (u × 1.1: 0.0954–0.1049, fails; activation swap 0.572,
  reported). Download 14.61 GB at 302.9 MB/s; fixed cost 108.9 s.
- 2026-11-06T10:01:04Z T_14: **G-M2V pass, exact**: 5,470,976 activations of 76 (unit, role) pairs equal M2's vectors
  (p99 0, exact share 1.000); failing input (axis rolled) p99 7.262, fails. G-TOPK pass: 2,038 entries, p99 0.01412
  (tolerance 0.04427), failing input p99 1.172, fails. G-EX reported (cross-device): 921 entries, p99 0.02559 against
  0.03872 on |a_bank| ≥ 0.1, 5 entries above (the largest L29N38− 0.385); failing input p99 1.139. Forward 175.3 s for
  737 windows.
- 2026-11-06T10:04:39Z T_14 done. **G-ABL-ID pass**: 23 clean rows, max |clean − forward| 0.0 nats (tolerance 1e-3);
  failing input (another window's loss) smallest difference 40.54 nats, fails. **Gradient rate measured 0.3707 s per
  window** (86.0 s for 232 windows, batch 1), below the spec's 0.54 s center and 1.0 s high. Ablation 107.6 s for 451
  rows; forward and ablation 0.2367 s per row (the spec's 0.2156 center, 0.25 high); fixed cost recorded for the stop
  rule 138.9 s (108.9 measured + 30). Records written at T_14: 97 freeze, 58 test (not opened).
- 2026-11-06T10:06:36Z Projection at T_15 (the job's stop rule): 627 s elapsed, 4,957 s remaining, budget 10,080 s
  (stop at 0.9 × 168 min = 9,072 s). Nothing to park.
- 2026-11-06T10:11:22Z T_15 done: G-REV `373bad25…`, G-CFG, G0 (0.00531; u × 1.1 0.0972–0.1058 fails) pass; G-M2V exact
  (2,763,008 activations; failing input p99 6.927); G-EX gated, exact (1,285 entries; failing input p99 7.499); G-TOPK
  exact (2,937; failing input p99 1.099).
- 2026-11-06T10:26:04Z T_13: G-REV `db8c654b…`, G-CFG, G0 (0.00497; u × 1.1 0.0967–0.1041 fails) pass; G-M2V exact
  (8,035,840; failing input p99 8.670); G-TOPK pass (2,004 entries, p99 0.01381; failing input 1.212); G-EX reported
  over its cross-device tolerance (1,345 entries, p99 0.04222 against 0.03872), as M2's A3 records for T_13; G-ABL-ID
  pass (33 clean rows, 0.0 nats; failing input 2.90 nats). Gradient pass 207.7 s for 539 windows (0.385 s per window).
- 2026-11-06T10:28:50Z **I-ID PASS, exact**: the 98 pre-role units' 8,035,840 activations equal across the reversed
  second forward (exact share 1.000), 0 top-10 flips; failing input (M2's birth-role vectors at T_14) p99 3.381, fails.
- 2026-11-06T10:32:22Z **I-GRAD FAILED; the job stopped (STATUS exit=1 at 10:32:34Z).** Spearman of the two gradient
  passes 0.999642 (bar 0.999, passes), exact share 0.2129 of 7,419,392 entries, but **4 of 98 units' u_S changed**
  (the rule allows 0). Failing input (axis rolled) ρ 0.000401, fails. The batch-1 backward on the L40S is not
  bit-deterministic; the forward is. Prior: bit-identical 0.6, within the rule 0.95. **score-01 DOES NOT COUNT.**
  Not checkable from banked data: no gradient of this model was ever banked, the dry run's backward is exact on CPU,
  and no planted input tested a near-exact second pass. Which 4 units flipped and by how much is not recorded (the gate
  writes the count only). Cause banked in `diag/score-01/cause.md`; `job.log` copied once to `diag/score-01/`. No
  record of either half was opened; G-VAL was never computed. No relaunch without the head.
- 2026-11-06T10:34:28Z EC2 `shutting-down` (instance-initiated); `terminated` first seen 10:38:52Z. Uptime 0.659–0.691 h,
  $1.48–1.55 on g6e.2xlarge.
- 2026-11-06T10:45Z **Head's decision, amendment A1**: the gradient pass and its I-GRAD re-run with
  `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA starts, and sdpa on the
  math kernel, for that pass only; I-GRAD's rule unchanged; flip records; the u_S resolution reported; re-freeze as
  `n12-freeze-1a`; then score-02. A second executor took the card from the first one's record.
- 2026-11-06T10:50Z–11:05Z Sources for A1, torch 2.12.0: the `use_deterministic_algorithms` docstring (CPU wheel) and
  the source at tag `v2.12.0` (flash and memory-efficient backwards nondeterministic by default; math kernel
  composite, fp32 upcast of bf16 by default; cuBLAS workspace 8.125 MiB by default, 32 MiB with `:4096:8`; the
  deterministic-mode check of the variable present at `v2.5.0`, absent at `v2.12.0`). transformers 5.9 calls sdpa with
  no mask and `is_causal=True` on these windows, so score-01's backward was most likely flash's (not verified on the
  GPU). Spec §14.
- 2026-11-06T11:13Z **Memory measured on the CPU** (`offline_gate/a1_memory.json`, one-eighth width, scaled by 8):
  saved per layer in the graph 0.388 GB (default), 0.959 GB (math, fp32 upcast), 0.640 GB (math, bf16 reductions).
  Estimated peak for a window whose lowest needed layer is 0 (64 of 2,801 gradient windows, at T_14 and T_13): 28.7,
  **46.4** and 36.5 GB against the L40S's 48.3 GB. The fp32 upcast would very likely stop the job out of memory at
  T_14. **Executor's addition, stated in §14**: the math kernel runs with bf16 reductions
  (`allow_fp16_bf16_reduction_math_sdp(True)` inside `grad_mode`). On the CPU its gradient differs from the fused
  kernel's by a relative L2 of 0.010–0.012, against 0.009–0.011 for the fp32 upcast.
- 2026-11-06T11:15Z Projection A1 (`projection_a1.json`, score-01's measured 0.2367 s per row and 116.5 s fixed cost,
  gradient 0.65 s center and 1.1 s high, estimated): 1.83–2.42 GPU-h; hard kill 189 min; job budget 175 min.
- 2026-11-06T11:11Z–11:19Z Dry run on the A1 code: `N12_DRYRUN_OK`, 100 checks, 0 failed (6 new). The planted
  near-exact second pass (one entry moved by 1e6): Spearman 0.999995 over 9,984 entries, 1 unit's u_S changed (the one
  reading the entry), I-GRAD fails. `N12_OFFLINE_GATE.md` "Amendment A1".
- 2026-11-06T11:20:59Z–11:28:42Z Dry run on commit `1791c3cfa` (the A1 code, clean tree): `N12_DRYRUN_OK`, 100 checks,
  0 failed; CHECK lines equal to the run before the commit (`offline_gate/dryrun_a1.log`). Two values moved against
  the freeze-1 run with the gradient pass's kernel (bf16 tiny world): the noisy plant's Spearman 0.6960 → 0.6957 and
  the tiny job's freeze-half mean u_S 0.531 → 0.559.
- 2026-11-06T11:29:07Z Commit `8bde64594` (dry-run record); annotated tag `n12-freeze-1a` (tag object `936ce18ed`) on
  it. Its code equals `1791c3cfa` (`git diff` of `infra/` between them: the operations note only). Recorded in spec §0.
- 2026-11-06T11:36:05Z L1 `score-02` launched on `n12-freeze-1a`, try 3 (tries 1 and 2 at 11:29:33Z and 11:32:33Z
  failed on `VcpuLimitExceeded`): `i-0057dc2e305a57a3b`, g6e.2xlarge, us-east-1a; hard kill `--budget-min 189`,
  reaper deadline 14:55:05Z. Inputs on S3 checked again (3 of 3 sha256 equal); prefix fresh; steps byte-equal. Watch
  every 120 s on `metrics.json`, never `job.log`.
- 2026-11-06T11:38Z G_SAME (torch 2.12.0+cu130, transformers 5.9.0), INPUTS, G_CORPUS pass; each failing input
  failed. `grad_pass_settings` recorded (`:4096:8`, deterministic, MATH, bf16 reductions).
- 2026-11-06T11:41:02Z **K0 PASS, bit-identical with the 32 MiB cuBLAS workspace**: max |ΔB| 0.0 over the 10 carriers
  (tolerance 1e-6); their 39 activations exact (p99 0.0, exact share 1.000); failing input (M2's birth-role B) smallest
  |ΔB| 8.995e-04, fails. So `CUBLAS_WORKSPACE_CONFIG=:4096:8` did not change the forward's bits on these windows (A1's
  stated risk; prior 0.85). T_14 G-REV `cb286dcd…`, G-CFG, G0 (0.00478; u × 1.1 0.0954–0.1049 fails; activation swap
  0.572, reported) pass. Download 14.61 GB at 217.8 MB/s; fixed cost 128.6 s.
- 2026-11-06T11:44:49Z T_14 forward 169.8 s (737 windows): **G-M2V pass, exact** (5,470,976 activations of 76 pairs,
  p99 0, exact share 1.000; failing input p99 7.262); G-TOPK pass (2,038 entries, p99 0.01412 against 0.04427; failing
  input 1.172); G-EX reported (921 entries, p99 0.02559 against 0.03872, 5 above, the largest L29N38− 0.385; failing
  input 1.139). Every value equals score-01's at T_14.
- 2026-11-06T11:49:10Z T_14 done. **The deterministic gradient pass ran without a raised operation: 135.3 s for 232
  windows, 0.5832 s per window** (score-01 0.3707; A1's estimate 0.65 center, 1.1 high), **peak memory 36.0 GB**
  (A1's CPU-based estimate for a lowest layer of 0: 36.5 GB). G-ABL-ID pass (23 clean rows, 0.0 nats; failing input
  40.54 nats). Ablation 104.2 s for 451 rows; measured 0.2293 s per row, fixed cost for the stop rule 158.6 s.
- 2026-11-06T11:51Z Projection at T_15 (the job's stop rule): 694 s elapsed, 5,738 s remaining, budget 10,500 s (stop
  at 0.9 × 175 min = 9,450 s). Nothing to park.
- 2026-11-06T11:55:56Z T_15 done: G-REV `373bad25…`, G-CFG, G0 (0.00531; u × 1.1 0.0972–0.1058 fails) pass; G-M2V exact
  (2,763,008; failing input p99 6.927); G-EX gated, exact (1,285; failing input p99 7.499); G-TOPK exact (2,937;
  failing input p99 1.099). Forward 240.1 s for 1,056 windows; download 313.6 MB/s.
- 2026-11-06T12:03:22Z T_13: G-REV `db8c654b…`, G-CFG, G0 (0.00497; u × 1.1 0.0967–0.1041 fails) pass; G-M2V exact
  (8,035,840; failing input p99 8.670); G-TOPK pass (2,004 entries, p99 0.01381; failing input 1.212); G-EX reported
  over its cross-device tolerance (1,345 entries, p99 0.04222 against 0.03872, 17 above), as M2's A3 records for T_13
  and as score-01. Forward 249.8 s for 1,081 windows.
- 2026-11-06T12:13:50Z T_13 gradient pass 334.7 s for 539 windows (0.621 s per window), peak memory 36.04 GB; ablation
  180.1 s for 781 rows; G-ABL-ID pass (33 clean rows, 0.0 nats; failing input 2.90 nats).
- 2026-11-06T12:15:10Z I-ID pass, exact: 8,035,840 activations of the 98 pre-role units equal across the reversed
  forward (exact share 1.000), 0 top-10 flips; failing input (M2's birth-role vectors at T_14) p99 3.381, fails.
- 2026-11-06T12:20:47Z **I-GRAD PASS, bit-identical**: the reversed second gradient pass equals the first on every one
  of 7,419,392 entries (exact share 1.000, Spearman 1.000000, largest |ΔG| 0.0 against largest |G| 2.047); 0 of 98
  units changed u_S; 0 units changed S order in either half; largest |ΔS| 0.0. Failing input (axis rolled) ρ 0.000393,
  fails. A1's deterministic backward works on the L40S (prior 0.9). The noise level for the u_S resolution report is
  therefore 0.
- 2026-11-06T12:21Z–13:25:11Z T_12 … T_5 done, every gate passing with its failing input failing (73 gates in all, 0
  problems): G-M2V exact at all 11 points; G-EX gated and exact at T_15, T_6, T_5, reported at T_7–T_14 (over its
  cross-device tolerance at T_13, T_11, T_10, T_9, T_8, T_7, as M2's A3 records); G-TOPK and G-ABL-ID (0.0 nats) at
  every point with entries. Gradient rate per point 0.48–0.71 s per window, 2,801 windows in 1,580.6 s (0.564 s per
  window), peak memory 34.23–36.04 GB. The T_11 download ran at 39.6 MB/s (368.5 s). Nothing parked.
- 2026-11-06T13:25:34Z **STATUS `exit=0`, the job complete** (job 11:37:36Z–13:25:11Z, 6,454.5 s); unit records
  234 freeze, 208 test. EC2 shutting down (instance-initiated) by 13:25:52Z.
- 2026-11-06T13:26Z Banked (freeze half only): `banks/score-02/` (`STATUS`, `job.log` copied once after the end,
  `heartbeat.jsonl`, `dmon.log`, `pipeline_status.json`) and `banks/score-02/n12/` (`metrics.json`,
  `ranks_freeze.jsonl`, `n12_freeze.jsonl`, `igrad_changed_freeze.jsonl` (empty), `samples/raw_examples_freeze.jsonl`).
  No test-half file was downloaded.
- 2026-11-06T13:27Z **Freeze-half raw examples, the sample questions of §9.5 S3** (`samples/read_examples.py`). Of
  the ten example births, four score right under H_signal (u_S ≤ 0.25), three wrong (u_S ≥ 0.75), three between.
  - Scored right: **L03N7746+ c0 (C corp)**, u_S 0.25 (S +0.00089; u_P 0.55, u_P′ 0.75). The five exemplars decode
    to " corp" or "orpor" at the end of "taxed as a C corp" or "form as a C-c[orpor]ation": the cluster's name. The
    top three competitors (B at T_(k−1) 1.00000, 0.99999, 0.99998) have their own top contexts elsewhere: L03N5771− in
    legal enactment text, L03N8235+ in URLs and unrelated prose, L03N1326− in LLC registration text (the job's topic,
    not its token). High B on E_c says E_c sits in each pole's upper tail, not that these are their best contexts, as
    in the zero-GPU examples. The loss-matched positions are other tokens of the same documents (" taxes", "ate",
    " You", " tax", " clearly", " and"), none the job's token; mean |Δℓ| 0.0070 nats. The carrier's ablation ΔL
    +0.288 nats has the sign of ΔL̂ +0.0040, about 70 times smaller.
  - Scored wrong: **L20N5782+ c1 (responsibility and consequences words)**, u_S 1.0 (S −0.0077; every competitor's
    S is higher; u_P′ 0.2). The 15 exemplars decode to " consequences", " accountable", " responsibility",
    " responsible", " actions": the name. The competitors are rivals on the same job: L20N1814+ has one of the
    job's exemplars among its own top 3, L20N4687− has "feared the negative consequences" at its top. Loss-matched
    tokens (" Federal", " broken", " ethical", " letter", "34", " flags") are not the job's; mean |Δℓ| 0.044. The
    carrier's ablation ΔL −1.141 has the sign of ΔL̂ −0.187.
  - Over the ten carriers the ablation and its first-order prediction agree in sign in 7 (L06N9844+, L16N1206+,
    L07N8340+ disagree; L07N8340+ measured −2.67 against +0.031), and the measured |ΔL| is larger than |ΔL̂| in 9 of
    10 (ratio 0.6 to 8,000; NPR −1.128 against −0.00014). Among the six competitor rows of the two units above, two
    have opposite signs (L03N5771− ΔL̂ −0.110, ΔL +0.827; L20N1814+ −0.219, +0.247). G-VAL is a rank agreement over
    the half, computed next by the freeze.
- 2026-11-06T13:27:46Z **Freeze 2** (`n12 freeze --banks studies/next/n12/banks/score-02`, freeze half only):
  `N12_FREEZE_OK`, `thresholds.json` sha256 `85726798…`. Gates 73 of 73, the freeze half's 82 real units, the
  movement control 1.056 [0.840, 1.286] decades, the identity on every record, room: all pass. Frozen: the matched
  floor of u, q025 **0.4457** (mean 0.4993); `sha256_n12_freeze` `30f31cb1…`; `sha256_job_metrics` `53178f8d…`.
  **G-VAL on the freeze half FAILS**: Spearman of ΔL̂ against the measured ablation ΔL 0.0996 [−0.0276, 0.2146] over
  328 (unit, pole) pairs of 82 units, against the bar 0.7 and the shuffle's 97.5th percentile 0.1103; slope through
  the origin 0.489; failing input (sign flipped) −0.0996, fails. By §3.5 every gradient label (H_signal,
  H_sel_signal_*, H_bridge) will read DOES NOT COUNT at the verdict and is reported as exploratory; the realized labels
  and part B stand. Prior for G-VAL passing on both halves was 0.6 (logged for the draft's 0.5 bar).
- 2026-11-06T13:27:46Z Freeze half, DISCOVERY (not a test; `thresholds.json` → `discovery`): mean F 1.167 [0.949, 1.397]
  (own rise); the competitors' drop real 1.039, pb 1.278, dt 1.307 decades; paired real − pb −0.225 [−0.394, −0.060],
  real − dt −0.240 [−0.411, −0.065] (the leaders of real births fall less than a selected top 20 of the twins); mean u_S
  0.509 [0.443, 0.576], u_gain 0.052; Spearman(S, gain) 0.004 [−0.041, 0.048]; part B: hit real − dt 0.000
  [−0.110, 0.110], dt − pb −0.205 [−0.329, −0.082]; the dt construction control 73 of 73 at T_15, opposite pole 0.
  u_S resolution at noise 0 (I-GRAD's |ΔS|): 0 of 82 real units within it; the smallest carrier-to-nearest-competitor
  gap in S is 4.5e-06, the median 4.9e-04.
- 2026-11-06T13:28:55Z Commit `709464e3a` (the run, the freeze half's banks, `thresholds.json`, the sample answers);
  annotated tag `n12-freeze-2` (tag object `328078daf`) on it. Correction in the next commit: three journal lines of
  that commit carried estimated times (13:35Z, 13:40Z, 13:27Z); the file times are 13:26Z (bank), 13:27:46Z
  (`thresholds.json`).
- 2026-11-06T13:29Z Test half pulled after the tag (`ranks_test.jsonl`, `n12_test.jsonl`, `igrad_changed_test.jsonl`
  (empty), `samples/raw_examples_test.jsonl`); `n12_freeze.jsonl` sha256 still `30f31cb1…`.
- 2026-11-06T13:29:28Z **The verdict, run once** (`verdict_n12.json`): `H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION;
  H_signal DOES NOT COUNT; H_bridge DOES NOT COUNT; H_dt SUPPORTED`. Gates 0 problems, complete.
  - H_cf OWN_RISE: mean F 0.951 [0.738, 1.180] over 71 real test units (own move 0.879, others' move −0.072 decades).
  - H_away_sel_pb BELOW_SELECTION: the competitors' drop real 0.939, pb 1.164 decades; paired −0.226 [−0.408, −0.056]
    over 70 pairs. H_away_sel_dt AT_SELECTION: −0.113 [−0.295, +0.065] over 67.
  - G-VAL test ρ 0.0425 [−0.0594, 0.1411] over 284 pairs (shuffle q975 0.1159, slope 0.326); with the freeze half's
    0.0996 it fails on both halves, so H_signal, H_sel_signal_pb, H_sel_signal_dt and H_bridge read DOES NOT COUNT.
    Exploratory values: mean u_S 0.527 [0.451, 0.600] against a bar of 0.4225 (min of 0.5, 0.4457, 0.4225); u_gain
    0.063; u_pb − u_real −0.039 [−0.134, +0.057]; Spearman(S, gain) −0.013 [−0.072, +0.046].
  - Part B, H_dt SUPPORTED: hits at T_(k−1) real 42 of 71, dt 41 of 67, pb 53 of 70; real − dt −0.030 [−0.149,
    +0.090] (AT_SELECTION); dt − pb −0.134 [−0.254, −0.015] (LOWER); real − pb −0.171 [−0.286, −0.057] (M2's gap
    again). dt construction control 66 of 67, opposite pole 0.
  - H_0 refuted for the pb twin (drop BELOW, hits −0.171), not for the dt twin (AT, AT).
  - u_S resolution at noise 0: 0 of 82 freeze and 0 of 71 test real units; smallest gaps 4.5e-06 and 2.9e-06.
  Priors (§1.4): H_cf OWN_RISE 0.75 (landed); H_away_sel_pb BELOW 0.20 (landed); H_dt SUPPORTED 0.50 (landed); G-VAL
  passing 0.6 (did not). §12 filled from the verdict file and its two inputs. No result document; not merged; the
  different-model read (§7 step 5) is open.
