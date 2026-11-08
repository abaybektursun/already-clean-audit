# N2V journal

One line per event, UTC. Spec: `studies/next/07_PICKS/N2V_SPEC.md`. Offline gate: `studies/next/07_PICKS/N2V_OFFLINE_GATE.md`.
State and launch commands: `studies/next/n2v/STATE.md`.

- 2026-11-06T19:25Z Head's decisions before freeze 1 (spec §0.6): forward-mode differentiation in fp32; parts core and
  the 10 % finite difference; the own-loss readout if it rides on the same rows; the 0.999-ceiling inspection as
  written; K-A at 1e-6, bitwise expected; G-ROOM as written. The first J12 executor started applying them (an uncommitted
  dry-run check for `--parts core,fd`, `projection_core_fd.json`) and was cut off by an account limit at about 19:28Z.
- 2026-11-07T15:30Z Applied by the second executor. The own readout does not ride on the core and fd rows: ΔL_own and
  ΔL_fd_own do (written per pair, no label), but its prediction pred_jvp_own exists only from the forward-mode rows, and
  N12 banked no own-loss gradient. G-VAL own and window_jvp read NOT_RUN. Steps file `--parts core,fd --budget-min 360`;
  projection 3.90–5.10 GPU-h, hard kill 398 min. Dry run `N2V_DRYRUN_OK`, 75 checks, 0 failed (three new checks).
- 2026-11-07T15:30:31Z Freeze 1 tagged `n2v-freeze-1` on `f977f93a3` (tag object `76840b83d`). Inputs uploaded to
  `results/next_n2v/inputs/` and read back, 3 of 3 sha256 equal to the registry. Not launched: the GPU quota is held by
  N13's S4; the head gives the word. The launch commands are in `STATE.md`.
- 2026-11-07T16:06Z The coordinator's decision (spec §0.6 item 8): add the forward-mode part so the own readout has its
  prediction; the full 6.53–9.78 GPU-h approved; launch now under the capacity rule. Manifest `--parts core,fd,jvp
  --budget-min 700`, hard kill 763 min. Dry run `N2V_DRYRUN_OK`, 76 checks, 0 failed (one new: the manifest's parts and
  budget).
- 2026-11-07T16:06:06Z Freeze 1a tagged `n2v-freeze-1a` on `cd628c8a8` (tag object `baa19e63f`).
- 2026-11-07T16:06:49Z score-01 launched on `n2v-freeze-1a` (`i-03754787362f997e8`, g6e.2xlarge, first try). Operations: `infra/aws_pg/operations/2026-11-07_next_n2v.md`.
- 2026-11-07T16:44:52Z score-01 stopped at `t14_G_CAUSAL` (STATUS `exit=1`, job elapsed 2,148.9 s). 428 of 963 changed
  rows (zero, fd, probe) equal their unit window's clean row bitwise before i0; 535 differ somewhere before i0. The
  failing input failed as registered: 2 zero rows equal through i0. Passed before it: G_SAME, INPUTS, G_CORPUS,
  t14_G_REV, t14_G_CFG, K_A (40 of 40 pairs exact), t14_G_FP32. Forward mode was not reached. No freeze or test record
  was written. G-CAUSAL is a stop gate (spec §3.2, §6), so score-01 does not count and nothing is unblinded. The job
  records only the count, not which rows differ or by how much, so the cause is not identified. The next attempt is the
  coordinator's decision. Diagnostics: `studies/next/n2v/diag/score-01/`.
- 2026-11-07T18:40Z (third J12 executor) The cause of score-01's G-CAUSAL failure, from the code: the 428 bitwise rows
  are exactly the changed rows at an even slot distance from their clean row (10 rows per unit window in batches of 4;
  zero 1, zero 3, fd 1, fd 3 always even; 4 x 107 = 428 at T_14; `diag/score-01/slot_count.json`). The fp32 loss read
  each row's logits in place (`.float()` is a no-op in fp32); 100,278 fp32 logits are 8 bytes past a multiple of 16, so
  position j of slot b starts 16-byte aligned when b * 2049 + j is even and 8 bytes off when odd, and the CUDA
  log-softmax sums a misaligned row in another order. In bf16 `.float()` copied the row, which is why N12 and K-A were
  bitwise. Not run on a GPU; poles 1 and 3 are the same rows as the even slot distances in this design, and the
  instance's slot control separates the two readings.
- 2026-11-07T18:47Z A1 (spec §14): `ce_pos` copies each row's logits before the loss; G-CAUSAL records per-row
  differences; report `slot_control` at the first point (predicted: logits and copy 0 at every slot, view 0 at slot 2
  and above 0 at slots 1 and 3); I-FWD and I-JVP record their largest difference. Pass rules unchanged. Dry run
  `N2V_DRYRUN_OK`, 83 checks, 0 failed (`offline_gate/dryrun_a1.log`); the planted early hook still stops the job and
  its record shows 126 of 288 rows over, largest 0.0812. Manifest `next_n2v_score-02.json`, projection unchanged
  (6.53-9.78 GPU-h, hard kill 763 min).
- 2026-11-07T18:48:14Z Freeze 1b tagged `n2v-freeze-1b` on `d0745eb81` (tag object `4accda02c`).
- 2026-11-07T18:49:33Z score-02 launched on `n2v-freeze-1b` (`i-0537218438154119c`, g6e.2xlarge, first try). Operations:
  `infra/aws_pg/operations/2026-11-07_next_n2v.md`.
- 2026-11-07T18:55Z score-02 at T_14 (elapsed 358.5 s): G_SAME, INPUTS, G_CORPUS, t14_G_REV, t14_G_CFG, K_A (40 of 40
  pairs exact, max abs diff 0.0) and t14_G_FP32 pass; T_14 download 230.4 s at 63.4 MB/s. The report `slot_control`
  (window 4441, fp32, vocabulary 100,278) landed as A1 predicted: the view's start offsets 0, 8, 0, 8 bytes mod 16; the
  logits 0.0 at every slot against slot 0; the copy's loss 0.0 at every slot; the row view's loss 0.0 at slot 2 and
  9.54e-7 nats (one fp32 step at a loss near 8-16) at slots 1 and 3. The alignment diagnosis holds on the instance; the
  pole-parity reading is out (no ablation in this control). Gates as landed: `banks/score-02/live/`.
- 2026-11-07T19:16Z score-02 `t14_G_CAUSAL` passes: 963 of 963 changed rows bitwise equal to their clean row before i0,
  largest difference 0.0, 0 over the tolerance; the failing input fails (10 of 428 zero rows equal through i0; in
  score-01, 2, when only the 214 zero rows at an even slot distance could be equal before i0 at all; a zero row equal at
  i0 is one whose ablation does not change the loss at i0 in fp32, not read further). Plain rows 1,070.
- 2026-11-07T19:35Z score-02 T_14 done (elapsed 2,570.4 s): forward-mode memory `points.14.jvp_max_mem_gb` 36.62 GB on
  the 48 GB L40S at batch 2 (registered prior: at most 45 GiB, 0.85); `measured` 1.0166 s per plain row (projection 1.18
  center, 1.40 high), 2.6241 s per forward-mode row (2.6 times a plain row; projection 2-3 times), fixed 326.1 s per
  point. `banks/score-02/live/metrics_after_T14.json`.
- 2026-11-07T20:49Z score-02 T_13 done (elapsed 7,021.2 s): `t13_G_CAUSAL` 1,683 of 1,683 changed rows bitwise before i0
  (largest 0.0; failing input fails, 20 zero rows equal through i0); `I_FWD` 187 of 187 clean rows bitwise in reversed
  order with other companions (largest 0.0; failing input fails); `I_JVP` 32 of 32 forward-mode rows bitwise reversed
  (largest 0.0; failing input fails). Forward-mode memory 36.62 GB again. The job's projection at T_13: 21,150 s
  remaining after 2,570 s, under the 0.9 x 700 min park line (37,800 s). `banks/score-02/live/`.
- 2026-11-07T23:08Z score-02 T_12 and T_11 done (elapsed 15,324 s at T_10's start): `t12_G_CAUSAL` 1,827 of 1,827 and
  `t11_G_CAUSAL` 1,782 of 1,782 bitwise before i0 (largest 0.0; failing inputs fail, 6 and 8 zero rows equal through
  i0). The job's projection at T_11: 12,081 s remaining after 11,191 s; end expected near 01:20Z.
- 2026-11-08T00:44:44Z score-02 ended: STATUS `exit=0` (S3 00:44:45Z), `complete` true, nothing parked, elapsed 21,160.5 s,
  42 gates, all pass, no failing input passed: G_SAME, INPUTS, G_CORPUS; G_REV, G_CFG, G_FP32 and G_CAUSAL at all 9
  points (G-CAUSAL bitwise before i0 at every one of 8,883 changed rows, largest difference 0.0); K_A 40 of 40 exact;
  I_FWD 187 of 187; I_JVP 32 of 32. Forward-mode memory 36.62 GB at every point. 328 freeze and 284 test pairs. Downloads
  45-230 s per point (47.7-322 MB/s). Reports: `slot_control` (above); the bf16 probe at the K-A units, median |window
  sum| 0.299 nats.
- 2026-11-08T00:50Z Spec §7 step 3 (freeze half only; `n2v_test.jsonl`, `raw_examples_test.jsonl` and `losses/` not
  opened). Banked `banks/score-02/n2v/{metrics.json, n2v_freeze.jsonl, samples/raw_examples_freeze.jsonl}`. Read with
  `read/freeze_read.py` (new; first run checked the exemplar at N12's list index and found 3 of 10, because the job
  lists exemplars in window order and N12 in its own; corrected to a verbatim search before any value was used) ->
  `read/freeze_read.json`. Answers:
  1. The decoded exemplars are N12's: N12's exemplar text is found verbatim among the job's decoded exemplars for 10 of
     10 births (at a different list position in 7).
  2. The rows' losses before i0 equal the clean row's: yes, bitwise at every changed row of every point (G-CAUSAL).
  3. The carrier's five largest per-position changes sit at or after the zeroed indices: 10 of 10 births (e.g.
     L20N5782+:c1, i0 1480, j = 1480, 1484, 1486, 1485, 1481).
  4. The probe is at the fp32 floor: over the 82 freeze carriers, median |probe| 1.67e-4 nats (largest 7.9e-4) against
     median |dL| 0.0132 nats, ratio median 0.011; the own-position probe median 4.3e-6. The ratio reaches 1.39 for the
     carrier with the smallest |dL|, so a few pairs sit at the floor; G-ROOM at freeze decides whether the floor leaves
     room. The bf16 floor at the K-A units was 0.299 nats.
  5. Small |dL - pred_bf16|: L10N9052+:c2 (the NPR " Copyright" carrier's birth), pole L10N3950-: pred_bf16 -0.00836,
     fp32 dL -0.00844, pred_jvp -0.00777; N12's bf16 measurement was -2.215. Large: L03N7746+:c0 (" corp" in LLC
     taxation), pole L03N1326-: pred_bf16 -0.1418, dL -0.3520, pred_jvp -0.1407, dL_fd -0.0151 (0.1 x pred -0.0142);
     the carrier's largest change predicts " comply" (clean loss 7.89, change -0.0052).
- 2026-11-08T00:53Z `n2v freeze` with the freeze files only stopped with `FileNotFoundError` on `n2v_test.jsonl`
  (`read_bank` loaded both halves; `freeze` used only the freeze half). The test file was not pulled. A2 (spec §14):
  freeze reads the freeze half only; dry run `N2V_DRYRUN_OK`, 84 checks (one new: the same thresholds with or without
  the test file in the bank).
- 2026-11-08T00:52:42Z Freeze 2 (`n2v freeze`, freeze half only): `N2V_FREEZE_OK`, `thresholds.json` sha256 `cdd2ed26…`;
  freeze records sha256 `58e85a64…`; gates pass. Freeze-half validations (DISCOVERY, 328 pairs, 82 units; bar rho 0.7,
  ceiling 0.999): window (pred_bf16 against fp32 dL) rho 0.971 [0.939, 0.993], shuffle q975 0.104, slope 1.11, sign
  agreement 0.970; fd (0.1 x pred_bf16 against dL_fd) rho 0.978 [0.952, 0.993], slope 1.008; own (pred_jvp_own against
  dL_own) rho 0.985 [0.971, 0.995]; window_jvp rho 0.979 [0.948, 0.996]. G-ROOM medians 0.99989 (window), 0.99435 (fd),
  1.0 (own), 0.99988 (window_jvp); its failing input (N12's bf16 measurement as floor) 0.064, fails. No readout above the
  ceiling. bf16 against fp32 forward-mode prediction: Spearman 0.987, median relative difference 0.038. Median |dL|
  0.0218 against median |pred| 0.0210 nats (N12's bf16 measurement: 0.65).
- 2026-11-08T00:53:08Z Tagged `n2v-freeze-2` (tag object `94de754c6`) on `0a3742e43`, whose subject names no label.
  Then `n2v_test.jsonl` and `raw_examples_test.jsonl` pulled.
- 2026-11-08T00:53:22Z Spec §7 step 4, the verdict once (`verdict_n2v.json`, thresholds sha256 `cdd2ed26…`, gates pass):
  **G-VAL window PASS; G-FD PASS; G-VAL own PASS; branch A; N12 re-read: H_cf OWN_RISE; H_away_sel_pb BELOW_SELECTION;
  H_signal REFUTED; H_bridge UNDECIDED; H_dt SUPPORTED**. Test half, 284 pairs, 71 units: window (N12's banked bf16
  prediction against the fp32 zero ablation) rho 0.960 [0.915, 0.990], shuffle q975 0.118, slope 0.837, sign agreement
  0.954; fd rho 0.987 [0.971, 0.995], slope 0.988; own rho 0.986 [0.969, 0.996]; window_jvp rho 0.967 [0.923, 0.994].
  G-ROOM 0.9999 (window), 0.996 (fd), 1.0 (own); its failing input 0.076, fails. No readout above the 0.999 ceiling.
  Both halves pass every readout, so the labels hold. Median |dL| 0.0224 against median |pred| 0.0219 nats; N12's bf16
  measurement of the same pairs failed G-VAL with rho 0.043. The bf16 gradient against the fp32 forward-mode
  prediction: Spearman 0.995, median relative difference 0.026. The re-reading (N12's code, records and rules, the
  fp32 window measurement in place of the bf16 one; N12's ceiling not lifted): H_signal REFUTED (0.527 [0.451, 0.600]
  against the bar 0.423), H_sel_signal_pb AT_SELECTION (-0.039 [-0.134, 0.057]), H_sel_signal_dt AT_SELECTION (-0.020
  [-0.124, 0.084]), H_bridge UNDECIDED (-0.013 [-0.072, 0.046]): N12's exploratory gradient labels, now counted; status
  a dated re-reading (N12's test half was read on 2026-11-06), not a new prediction. In N12's own verdict these four
  read DOES NOT COUNT.
  Raw examples (`banks/score-02/n2v/samples/raw_examples_test.jsonl`): small error, L01N369+:c0 pole L01N2347+:
  pred_bf16 -0.01123, fp32 dL -0.01132, pred_jvp -0.01154 (N12's bf16 measurement +3.025); exemplar "... rumors and
  internet trolls about her [[ breakup]] with Addison Ray at Bryce Hall"; the carrier's largest change predicts
  " Bryce" (clean loss 7.50, change -0.0077). Large error, L10N10529-:c3 pole L10N8869+: pred_bf16 -0.399, dL -0.283,
  pred_jvp -0.408, dL_fd -0.0400 (0.1 x pred -0.0399): the finite difference follows the prediction while the zero
  ablation falls short of it, as a second-order effect would; exemplar "... the Rongor[[ongo]] script before the
  1860s".
  Step 5 (the fresh Opus reader in a worktree made from `n2v-freeze-2`, recomputing every pair's dL from
  `losses/tNN.npz` on S3, the four validations, G-ROOM and the re-reading) is not run by this executor (never
  delegate); it is the head's.
- 2026-11-08T01:50Z Spec §7 step 5 done (the head). The independent reader (Opus, worktree at `n2v-freeze-2`, branch `n2vread`,
  `READ_n2v.md`, `read_n2v/`): the labels and branch stand; every pair's eight readouts recount bitwise from `losses/tNN.npz`
  (612 of 612); the registered verdict's 81 numbers to 2.8e-14; its own labels were committed (`ee40dba7c`) before the registered
  code ran. Findings: within-unit rank agreement 65 of 71 test units (chance level 0.25), permutation within units leaves ρ 0.286;
  the slopes 0.837 and 1.110 come from a few large second-order pairs (fd slopes 0.988, 1.008); where bf16 and fp32 predictions
  disagree the measurement sides with fp32 (26 of 28); a defect in the reported field `sum_abs_h` for units sharing a window and
  layer (L01N369+:c0 and :c4 at T_11), read by no rule. The head's read: `READ_head.md`, claims N2V-1 to N2V-7, one retag of N12's
  head read (its postscript). Merged into bio2. N2V is closed.
