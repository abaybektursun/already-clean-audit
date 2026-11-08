# N22 journal

One line per event, UTC. Spec: `studies/next/07_PICKS/N22_SPEC.md`. Offline gate: `studies/next/07_PICKS/N22_OFFLINE_GATE.md`.
Infrastructure and cost: `infra/aws_pg/operations/2026-11-06_next_n22_score.md`.

## The author's prior (spec §1.4, logged 2026-11-06 before any model output existed), copied here before the first sample check

| quantity | prior |
|---|---|
| ≥ 100 of 1,200 answered at T_15 | 0.80 |
| the interchange control fires (GO) | 0.75 given ≥ 100 |
| H_slot (LENS_M) SLOT_DEPENDENT | 0.20 |
| LENS_U (the paper's protocol) SLOT_DEPENDENT | 0.25 |
| DOM_M SLOT_DEPENDENT (and counts) | 0.35 |
| readout PRESENT for s1 at T_15; for s2 | 0.50; 0.35 |
| READOUT_SINK | 0.30 |
| H_order | SUPPORTED 0.10, CENSORED 0.25, REFUTED_SAME 0.10, REFUTED_CONTRARY 0.05, NOT_RUN 0.50 |
| N25 H_arch: LEARNED on the CKA block statistic | 0.70 |
| I-GRAD bit-identical | 0.90 |
| K0 exact at T_15 | 0.95 |

## Events

- 2026-11-06T15:20Z The head's review: the draft accepted as written (the coarse 0.01 K0 gate at the Blackwell-banked
  points with the fine difference reported against 1.23e-4 and the neighbouring point's bank as failing input; the
  lens-check cosine gate; fp32 readouts with the lens fitted in bf16; the query-position swap; the split third
  launch). Nothing in the draft changes.
- 2026-11-06T15:17:53Z–15:22:46Z Dry run again on HEAD `76fde0751` (`$SP/n22/dry_f1`): `N22_DRYRUN_OK`, 100 checks,
  0 failed (`offline_gate/dryrun_freeze1.log`). `dryrun_summary.json` equal as JSON to the draft's. Four CHECK lines
  differ in a printed value, all PASS: the CKA of unrelated matrices 0.128 → 0.15 and the Laplace kurtosis 2.917 →
  2.989 (draws without a seed in the SCORE checks), the planted noisy I-GRAD repeat's max_rel 0.00731 → 0.00728, the
  stop rule's elapsed 24 → 25 s.
- 2026-11-06T15:23:29Z Annotated tag `n22-freeze-1` (tag object `a877b3390`) on `76fde0751`. Recorded in spec §0.
- 2026-11-06T15:24Z Pre-launch checks of §9.5 S1: `pg ps` 15:18:00Z listed two instances not from N22, 15:24:08Z three
  (32 of 32 G vCPU); `pg audit` 0 reaper-less. `results/next_n22/` empty (no version, no delete marker). `items.json`
  sha256 `dbf0b225…` before the upload and after downloading it back; `l1-01/steps.json` byte-equal.
- 2026-11-06T17:19:46Z L1 `l1-01` launched on `n22-freeze-1`, try 16 (tries 1–15 from 15:24:17Z refused:
  `VcpuLimitExceeded` while N13, N6 and bytegraft held the quota, then g6e.2xlarge capacity): `i-09988ba374ab9fbfb`,
  g6e.2xlarge (one L40S), us-east-1b; hard kill `--budget-min 138`, reaper deadline 19:47:46Z. Watch every 120 s on
  `metrics.json`, never `job.log`.
- 2026-11-06T17:20Z Memory against 48 GB (the head's note): no L1 step projects above it (fp32 phase about 33–35 GB,
  the bf16 lens fit at or below N12's measured 36.0 GB); the job records no GPU peak. Table in the operations note.
- 2026-11-06T17:21:31Z G_SAME (torch 2.12.0+cu130, transformers 5.9.0), INPUTS (`dbf0b225…`), G_CORPUS (`2aca80ca…`)
  pass; each failing input failed (job started 17:21:27Z).
- 2026-11-06T17:25:17Z T_15 download 14.61 GB in 226.2 s (64.6 MB/s; the registered high was 192 s); G-REV
  `373bad25…`, G-CFG (no differing key) pass, failing inputs failed.
- 2026-11-06T17:26:15Z T_15 G-TOK pass (20 items; failing input item 0 against item 1 fails). **K0 PASS, exact**:
  2.675372037760417 against the bank 2.675372037760417, difference 0.0 (tier 1e-6, same device); failing input (T_14's
  bank 2.714761865234375) difference −0.03939, fails. Prior for exact was 0.95. Load 5.7 s.
- 2026-11-06T17:26:55Z T_15 massive scan written (`t15/massive.json`, reported): max |h_i| / median |h| over the 24
  lens prompts (BOS included) 120.5 at block 0, lowest 56.2 at block 28, 156.5 at block 30, largest 204.8 at block 31;
  top-3 coordinates 2553, 856, 3885 in blocks 0–3; block 30's top-3 2196, 20, 2174. The lens fit started after it.
- 2026-11-06T17:52:17Z **I-GRAD PASS, bit-identical**: prompt 0's Jacobian recomputed after the fit, max relative
  Frobenius difference 0.0 over source blocks 0–29 (tolerance 1e-6); failing input (prompt 1's Jacobian) smallest
  relative difference 0.2685, fails. Prior for bit-identity 0.9. Lens fit 1,553.0 s for 25 passes (24 prompts and the
  repeat): **61.9 s per prompt** (registered 78 s center, 120 s high).
- 2026-11-06T17:53:13Z Pool forward in fp32: 39.3 s for 1,200 prompts (**0.0328 s per prompt**; registered 0.018
  center, 0.040 high). **Answered at T_15: 2 of 600 freeze items, 5 of 600 test items, 7 of 1,200** (`metrics.json` →
  `points.15.answered`). The go/no-go needs 100; the edits need 50 per half. Prior for ≥ 100 was 0.80.
- 2026-11-06T17:53:13Z **LENS_FIDELITY PASS**: mean cos(J_29 h_29, h_30) 0.9441 against 0.0053 with the rotated
  residual; paired difference 0.9388 [0.9379, 0.9398] over 1,200 items; the untransported cosine 0.9309. Failing input
  (rotated against itself) fails.
- 2026-11-06T17:56:39Z T_15 done: readouts through the five lenses, class means (5,056 prompts, 165.5 s, 0.0327 s per
  prompt), probes and N25's statistics written; **edits NOT_RUN** (fewer than 50 answered in a half). So the job did not
  run the identity pairs I-FWD, I-EDIT and G-IC-ID, the interchange control at the operand positions, the DoM sweep or
  the raw-example edits: in `n22_score.run_point` all of them sit after the N_MIN check. Wall 2,107.0 s for the point.
- 2026-11-06T17:57:03Z **STATUS `exit=0`, the job complete** (job 17:21:27Z–17:56:39Z, 2,111.5 s), 9 gates, every one
  passing with its failing input failing, nothing parked. EC2 shutting down (instance-initiated) by 17:57:24Z.
- 2026-11-06T17:58Z Banked: `banks/l1-01/` (`STATUS`, `job.log` copied once after the end, `heartbeat.jsonl`,
  `dmon.log`, `pipeline_status.json`) and `banks/l1-01/n22/` (`metrics.json`; `t15/` without `lens_AB.pt`, which stays
  on S3 at `results/next_n22/l1-01/out/score/n22/t15/lens_AB.pt`, 2,013,282,415 bytes). No `edits_*.jsonl`,
  `ic_*.jsonl` or `samples/` exist. `accuracy.json`, `clean_lp_numbers.npy` and `readout.npz` hold both halves; only
  their freeze-half rows (even item index) were read before freeze 2.
- 2026-11-06T17:58Z **The first sample check (§9.5 S3), freeze half only.** The job wrote no raw-example file, so the
  ten were filled from the arrays (`samples/raw_examples_l1.py` → `samples/raw_examples_t15.{md,json}`).
  - The prompt is what §0.5 says: item 0 decodes to the three solved lines and "(2 + 20) * 4 + 62 = ".
  - The model's top-1 at position 62 is a number token "0"–"999" for 600 of 600 freeze items; the 1,000 number tokens
    hold median 0.998 (minimum 0.996) of the probability. So the harness reads the answer position and the model
    reads the task as "write a number".
  - The answers are diffuse and wrong: p(top-1) median 0.028; p(y) median 0.0036 (uniform over numbers 0.001), largest
    0.093; the rank of y among the 1,000 numbers median 73, top 10 in 9 % of items; the top-1 is within 10 of y in
    10.5 %. The top-1 equals s2 in 26 items and d in 19, y in 2, s1, a, b, c in 0. 564 of 600 answers have three digits;
    498 top-1 tokens do.
  - The item-specific floor: lp(y) exceeds lp(the partner's y) in 80.3 % of freeze items, by 2.42 nats on average; the
    partner's y has median rank 247.5. The model holds item-specific information about the answer's region, not the
    answer.
  - One unanswered item, 0: "(2 + 20) * 4 + 62 =" (y 150), top-1 "124" at 0.055, p(y) 0.0144, y rank 23 among numbers;
    under the full lens s1 = 22 reaches rank 29 (block 25) and s2 = 88 rank 17 (block 25); under the logit lens 12
    (block 21) and 8 (block 24). One answered item, 714: "(5 + 45) * 2 + 36 =" (y 136), top-1 "136" at 0.064; s1 = 50
    best rank 86 (block 26), s2 = 100 rank 40 (block 29); y rank 1 at block 29. The lens top-10 at the bands and the
    outputs after the edits were not produced.
  - Bug check, said plainly: these facts are what a correct harness gives for a model that cannot do the arithmetic in
    one token; nothing found points to a format or position error. K0 exact in bf16 and LENS_FIDELITY 0.944 show the
    weights and the fp32 residuals are sound. No fp32 identity pair ran, so the fp32 forward's noise floor is not
    measured; it cannot move an argmax over 1,000 numbers for 593 of 600 freeze items by itself, but that is an
    argument, not a measurement.
- 2026-11-06T17:59:25Z **Freeze 2** (`n22 freeze --banks studies/next/n22/banks/l1-01`, freeze half; `thresholds.json`,
  `metrics_sha256` `95262d57…`): bands from the 2 answered freeze items ℓ1 = 0, ℓ2 = 25, ℓm = 0, B_m = {0, 1}, matched
  net readout 0.0; delta_read 0.5 (lens A against B on two items: one item's hit is a share of 0.5). **No matched layer**
  (0.0 ≤ 0.5). DoM guard: answer net readout at ℓm 0.0, not readable. PC-ANS and the interchange control: not
  computed (no edit records). **Go/no-go: NO-GO** — 7 answered of 1,200 (needs 100) and the interchange control did not
  run (needs 50 answered per half). L2 and L3 are not launched (spec §3.3).
- 2026-11-06T18:00:44Z Commit `a083b357f` (banks, `thresholds.json`, raw examples, this journal, the operations note);
  annotated tag `n22-freeze-2` (tag object `8da8b5881`) on it. The test-half rows are opened only after this.
- 2026-11-06T18:00:50Z **The T_15 verdict, run once** (`n22 verdict` → `verdict_n22_t15.json`; the brief's file name,
  where spec §9.5 S3 wrote `verdict_t15.json`): on the 5 answered test items, readout s1 at block 0 hit 0.0 against
  the partner's 0.0 (net 0.0), s2 at block 25 hit 0.2 against 0.0 (net 0.2), both **NOT_PRESENT** under delta_read 0.5;
  the projected lens gives the same nets, so both **READOUT_SURVIVES**. **Slot labels: "NOT_RUN: no edits at this
  point"**: H_slot (LENS_M), the depth reading (LENS_U), DOM_M, H_answer, H_inject, H_outside, PC-ANS and the ceiling
  have no records to score. H_order NOT_RUN. N25's labels need T_0 (L3) and are not computed. Spec §12's T_15 rows
  filled from the verdict, naming `thresholds.json` where the verdict holds no value.
- 2026-11-06T18:01Z Reading, plain: OLMo-3 7B at the last stage-1 checkpoint answers 7 of 1,200 three-step expressions
  "(a + b) * c + d =" without chain of thought (0.58 %), against a prior of 0.80 for at least 100. The registered
  NO-GO branch fires: the slot test had no answered items to edit. The study ends at T_15 as §3.3 says; L2 and L3 are
  the head's decision only if the card is changed (for example a simpler expression or chain of thought), which this
  run does not decide.
- Not verified, said plainly: no fp32 identity pair ran (I-FWD, I-EDIT, G-IC-ID all sit after the N_MIN check), so
  the fp32 forward's noise floor at T_15 is not measured; the interchange control never ran, so whether this harness
  moves the answer was not shown; the lens top-10 at the bands for the raw examples needs residuals the job does not
  bank; the GPU memory peak was not recorded.
