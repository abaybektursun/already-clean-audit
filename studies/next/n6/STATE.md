# N6 state, 2026-11-08T15:30Z

**CLOSED 2026-11-08: DOES NOT COUNT (G-PC failed and its failing input passed; G-POWER failed, no test label).** The registered verdict
on L2a alone (`n6 verdict --bank studies/next/n6/banks/L2a-01`, no L2b records) wrote `studies/next/n6/verdict_n6.json` (sha256
`b774450623aac2caab55a8fb482dbce9fc04a09e3ea11a9931d2586a41c33599`); its headline, verbatim: `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT` (realized label
UNDECIDED_NO_POWER for each, `failed_gates` `G_PC`, `G_PC:failing_input_passed`, `n_test_read` 0). The head's decision (2026-11-08
15:15Z): L2b is not launched. The registered next step for no power (spec §1.2, a larger eval set or more pairs) is not taken now: it
has lower leverage than the rank test of the RL delta.

## What L2a measured

- **The positive control is inverted.** In the 8 positive-control layers (0, 4, …, 28) each layer's highest-energy pole was edited
  along its own column (col) and along a random direction of the same size (rand). The column edit cost more corpus loss than the random
  edit only in layers 24 and 28: `metrics.json` → `gates.G_PC.n_col_gt_rand` 2 (need 6), `gates.G_PC.failing_input_swapped` 6. The
  per-layer losses are the `dL` field of the records `PC00_col` … `PC28_rand` in `studies/next/n6/banks/L2a-01/edits.jsonl` (for example
  layer 12: col 7.07e-4, rand 1.10e-2 nats per token). Every one of these edits moves the loss far above the rounding floor (smallest
  1.0e-4 against F_eps 3.4e-9), so the instrument resolves them; the order is reversed. The realized change of the residual stream after
  the edited layer (field `R` of the same records) is 1.3–2.5 times larger for rand than for col in all 8 PC layers (2.51, 1.83, 1.82,
  1.91, 2.06, 1.75, 1.34, 1.62), while on the freeze-pair edits (records with `role` pair, `kind` col and rand, 42 each) the medians are
  matched, 60.82 against 60.68.
- **There was not enough power.** At 1,024 eval windows the eval-set noise of one pair's contrast is τ_s 1.5096e-6 nats per token
  against a median |D_col| of 1.2726e-6 over the 21 freeze pairs (`thresholds.json` → `G_POWER`; ratio 1.186, bar 0.5). The 96-window
  preview of amendment A4 held |D_col| at 4.47e-6; at 1,024 windows it is 3.5 times smaller. To pass at this |D_col| the noise would need
  to be 2.37 times smaller, about 5,760 windows if windows were independent.
- **Freeze-half readout, exploratory and not a claim** (`thresholds.json` → `discovery`): the early-born pole's column edit cost more in
  16 of 21 freeze pairs, p_col 0.762, Wilson [0.549, 0.894]; R 0.095, so only 2 pairs are above twice the noise; p_rand 0.381 [0.208,
  0.591]. The test half was never read.
- **Hypothesis, untested:** OLMo-3's post-feedforward RMSNorm rescales the MLP output, so removing a dominant pole's own write is partly
  undone by the norm, while a random direction of the same size is not. Nothing in N6 tests this.
- **Not verified:** the exact second the instance began shutting down (between 2026-11-08T15:02:18Z and 15:04:22Z); the token contexts
  of the PC rows (the job decodes only the raw-example rows; decoding needs the corpus and tokenizer, which are not on the laptop).
- **Cost:** L2a 18.93–18.96 GPU-h ($42.40–42.47 at $2.24 per hour); N6 in all (L1-01, L1-02, L2a-01) 19.58–19.63 GPU-h, $43.86–43.96
  (`infra/aws_pg/operations/2026-11-06_next_n6_edit.md`, `2026-11-07_next_n6.md`).

## Claims (N6 has no claims table; one line each)

- **N6-1** The positive control is inverted: col beat rand in 2 of 8 PC layers, rand beat col in 6 of 8. STANDS
  (`banks/L2a-01/metrics.json` → `gates.G_PC`).
- **N6-2** No power at 1,024 windows: τ_s 1.5096e-6 against 0.5 × median |D_col| 6.363e-7 (ratio 1.186). STANDS (`thresholds.json` →
  `G_POWER`).
- **N6-3** Freeze-half readout p_col 0.762 (16 of 21), R 0.095, p_rand 0.381. EXPLORATORY, not a claim (`thresholds.json` → `discovery`).

## L2a-01 result (2026-11-08; `banks/L2a-01/`)

| item | value |
|---|---|
| run | `i-00e9ff97327126cd4`, `finished_utc` 2026-11-08T15:02:13Z, 113 of 113 edits, nothing parked, `complete` true, `STATUS` exit=0 |
| gates passed | G-SAME, G-REV, K1 0.6742177606, K2 2.675372037760417, G-POLE 0.010999 (115 poles, bit-equal to L1-02), G-FP32, K3 2.675162242635091, G-INST 0.0, G-ID 0.0, G-CACHE (mean −2.71e-10, largest 1.91e-6), G-EPS (F_eps 3.365e-9), G-ZN (2.71e-10), G-NORM (1.07e-8), G-DOSE 8 of 8, G-SAMPLES 10; each failing input fails |
| gate failed | **G-PC**: col > rand in 2 of 8 PC layers (need 6); rand > col in 6 of 8 (the failing input passed). The PC rows and R in the journal (2026-11-08T15:05Z) |
| freeze 2 | F_eps 3.365e-9, τ_eps 3.365e-8, τ_s 1.5096e-6, q10 2.438e-7; G-RES pass (3.365e-9 ≤ 4.876e-9); **G-POWER fail** (ratio 1.186) |
| DISCOVERY (21 freeze pairs) | p_col 0.762 [0.549, 0.894] (16 wins); p_rand 0.381 [0.208, 0.591]; R 0.095; median D_col +9.82e-7 nats per token |
| verdict | run 2026-11-08 on L2a alone: `verdict_n6.json` (sha256 `b774450623aa…`), `H_age DOES NOT COUNT; H_adj DOES NOT COUNT; H_eff DOES NOT COUNT`; L2b not launched (the head's decision) |

## L2a-01 launch record (2026-11-07)

| field | value |
|---|---|
| tree, dry run | `git status --short` clean at `20945a7ec`; `git diff n6-freeze-1c HEAD -- infra/ pairs_L2.json inputs/` empty; dry run on `20945a7ec` `N6_DRYRUN_OK` 136 checks, 0 failed, CHECK lines and `dryrun_summary.json` equal to `dryrun_A5.log`'s (`offline_gate/dryrun_L2a.log`), committed as `d5554e2ad` |
| S3 before | `inputs_L2/`, `L2a-01/`, `L2b-01/` held no object version and no delete marker (20:04:02Z) |
| inputs | uploaded 20:06:16Z–20:06:19Z, downloaded back: `pairs_L2.json` `7e3a9948…`, `e1_known.json` `79da230c…` (= `INPUT_SHA["L2"]`); `L2a-01/steps.json` byte-equal to `next_n6_L2a.json` |
| instance | `i-00e9ff97327126cd4`, g6e.2xlarge, us-east-1d, try 1 (20:06:36Z–20:06:44Z), `LaunchTime` 2026-11-07T20:06:42Z |
| PG_GIT_SHA | `d5554e2ad921` = HEAD at launch (not dirty) |
| code tarball | `code/4f431c1c67b1.tgz`: 2,079 files equal to HEAD's blobs, 0 differ, 0 missing, 21 `__pycache__` files |
| allowance | hard kill 1,858 min (reaper `pg-reap-i-00e9ff97327126cd4` at 2026-11-09T03:14:42Z); job stop rule `--budget-min 1672` (parks when elapsed + projected > 1,504.8 min) |
| watch | every 120 s, `metrics.json`, `pipeline_status.json`, `STATUS`; gate records banked at first sight in `banks/L2a-01/live/` |
| cost and lifecycle | `infra/aws_pg/operations/2026-11-07_next_n6.md` |

| item | state |
|---|---|
| freeze 1, 1a, 1b | `n6-freeze-1` on `743d22324`; `n6-freeze-1a` on `e5c5f21f2`; `n6-freeze-1b` on `eba7035cb` |
| freeze 1c | `n6-freeze-1c` (tag object `843f66877`) on `b94dbd8e1` (`b94dbd8e1d405af3c5e8760696774e4ba9cf3989`), 2026-11-06T17:54:17Z; dry run rerun on that commit before the tag, 136 checks, CHECK lines and summary equal |
| L1-01, L1-02 | L1-01 DOES NOT COUNT (G-POLE at the old bar); L1-02 complete, banks in `banks/L1-02/` |
| A2, A3, A4 | G-ZN, G-CACHE tolerances; the split 24 test / 21 freeze; G-RES at the factor 50, 1,024 eval windows, G-POWER at freeze 2, the positive control in 8 layers (spec §14) |
| A5 | L2 as two launches: `--half freeze` (L2a) and `--half test` (L2b) on one pairs file; L2b checks L2a's `metrics.json` (`--against`) before its first gate: stage L2, half freeze, complete with nothing parked, same eval windows, eval tokens and pairs content (sha256) |
| banks | `banks/L2a-01/`, `banks/L2b-01/`, each as the job writes it; `n6 freeze --bank L2a`; `n6 verdict --bank L2a --bank L2b` (`n6.read_banks`; L2b's gates under `L2b:`) |
| dry run | `N6_DRYRUN_OK` 136 checks, 0 failed (`offline_gate/dryrun_A5.log`); the 117 A4 CHECK lines identical; 19 added |
| L2 inputs | unchanged: `INPUT_SHA["L2"]`, `pairs_L2.json` `7e3a9948…`, `e1_known.json` `79da230c…` |
| L2a manifest | `infra/aws_pg/steps/next_n6_L2a.json`, job stop rule `--budget-min 1672`; hard kill `--budget-min 1858` |
| L2b manifest | `infra/aws_pg/steps/next_n6_L2b.json`, `--against` L2a-01's `metrics.json`, job stop rule `--budget-min 1601`; hard kill `--budget-min 1779` |
| A4's one-launch manifest | `next_n6_L2.json` removed (kept at `n6-freeze-1b`) |
| S3 | `inputs_L2/{pairs_L2.json, e1_known.json}` and `L2a-01/steps.json` uploaded 2026-11-07T20:06Z; `L2b-01/` not written |

## The two launches (`projection_L1.json`, A4's model at L1-02's 29.01 TFLOP/s)

| launch | edits | center GPU-h | high GPU-h | $ at $2.24 (center / high) | high at $3.00424 (g6e.4xlarge) | hard kill | job stop rule |
|---|---|---|---|---|---|---|---|
| L2a (freeze) | 113: PC 24, PC probes 2, freeze pairs 84, pair probes 2, zero 1 | 19.054 | 23.818 | 42.68 / 53.35 | 71.56 | 1,858 min | 1,672 min |
| L2b (test) | 99: test pairs 96, the same pair probes 2 and zero 1 | 18.239 | 22.799 | 40.86 / 51.07 | 68.49 | 1,779 min | 1,601 min |
| sum | 212 | 37.293 | 46.617 | 83.54 / 104.42 | — | — | — |

- The sum is A4's one launch (35.204 / 44.005) plus one more setup (1.498 GPU-h center: fixed 604 s, window part 4,290 s, G-INST pass
  113 s, residual copies 384 s) plus the three edits that run in both launches (0.591 GPU-h center: `P005_late_probe`, `P005_late_zero`
  at layer 1, `P015_late_probe` at layer 30). They run in L2b because the head listed G-EPS and G-ZN among L2b's gates, and those gates
  read the fp32 pair probes and the zero edit. Dropping them from L2b saves 0.591 / 0.739 GPU-h and drops those two gates from L2b; the
  head's call, not taken.
- GPU and host memory per phase as A4 projected (each launch runs the same phases; L2b edits fewer poles, so its exemplar log-probabilities
  take less host memory).

## Not verified (said plainly)

- From A4: the host-device copy rate (6 GB/s assumed); MemAvailable at start (the job asserts ≥ 40.0 GiB); the chunked progression's GPU
  peak (projected ≤ 36.201 GiB); base pass 2's time and the bf16 probe passes (both scaled by 10.67); G-POWER at 1,024 windows (the 0.435
  preview holds \|D_col\| at its 96-window value).
- New with A5: whether two L40S instances give bitwise-equal records (the three edits in both launches show it on the GPU; K1 and K2 were
  exact across instances before). L2b's budget is projected from L1-02's rate; L2a measures the rate at 1,024 windows first. L2b's
  launch condition (freeze 2 tagged with G-POWER passed) is a runbook step, not a check in the job.

## L2a launch (the head's; run 2026-11-07T20:06Z with `--type g6e.2xlarge` only, record above)

```
B=s3://pg-lineage-769063704696
P="--profile fuelos --region us-east-1"
# 1. the three prefixes hold no object version and no delete marker
aws s3api list-object-versions --bucket pg-lineage-769063704696 --prefix results/next_n6/inputs_L2/ $P
aws s3api list-object-versions --bucket pg-lineage-769063704696 --prefix results/next_n6/L2a-01/ $P
aws s3api list-object-versions --bucket pg-lineage-769063704696 --prefix results/next_n6/L2b-01/ $P
# 2. upload (from the worktree at n6-freeze-1c or a later commit with the same files)
aws s3 cp studies/next/n6/pairs_L2.json $B/results/next_n6/inputs_L2/pairs_L2.json $P
aws s3 cp studies/next/n6/inputs/e1_known.json $B/results/next_n6/inputs_L2/e1_known.json $P
aws s3 cp infra/aws_pg/steps/next_n6_L2a.json $B/results/next_n6/L2a-01/steps.json $P
# 3. download back and compare
aws s3 cp $B/results/next_n6/inputs_L2/pairs_L2.json - $P | shasum -a 256    # 7e3a99480b3a2b939ca6eb0a8ae6fe43ee0e8bad22bee302c1a37ca1e4e1755b
aws s3 cp $B/results/next_n6/inputs_L2/e1_known.json - $P | shasum -a 256    # 79da230cc099e6981dad1df00bee7c46dfd3b6e890cb9ca7e9a174937e27db61
aws s3 cp $B/results/next_n6/L2a-01/steps.json - $P | cmp - infra/aws_pg/steps/next_n6_L2a.json
# 4. launch: hard kill 1,858 min; the job's stop rule (in steps.json) 1,672 min
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n6/L2a-01/steps.json" \
    --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n6/L2a-01 --budget-min 1858 --json --detach
```
The capacity rule (every 180 s for six tries, then every 600 s); watch `metrics.json`, `pipeline_status.json`, `STATUS` every 120 s.
First reads: `half` freeze, G-SAME, K1, K2, G-POLE, `host_mem`, `w_eval` 1,024, `eval_windows_sha256` = `5822191…`, `pairs_sha256`, G-INST,
`peak_mem_gib`, then `measured.sec_per_edit_layer_unit` (projected about 34 s at 1,024 windows).

After L2a: pull `metrics.json`, `edits.jsonl`, `colnorms.npy`, `energies.npy`, `g_pole.npz`, `samples/` from
`$B/results/next_n6/L2a-01/out/edit/n6/` to `studies/next/n6/banks/L2a-01/`; open the freeze pairs' raw rows and the PC rows; then
```
PYTHONPATH=. python3 -m infra.pg.n6 freeze --bank studies/next/n6/banks/L2a-01 --pairs studies/next/n6/pairs_L2.json \
    --out studies/next/n6/thresholds.json
```
The DISCOVERY numbers and the prediction for the test half go in the journal; commit; tag `n6-freeze-2`. If `G_POWER.pass` is false, L2b
is not launched and the verdict runs on L2a alone (`n6 verdict --bank studies/next/n6/banks/L2a-01 ...`: UNDECIDED_NO_POWER).

## L2b launch after `n6-freeze-2` with G-POWER passed (the head's; not run)

```
aws s3api list-object-versions --bucket pg-lineage-769063704696 --prefix results/next_n6/L2b-01/ $P    # empty
aws s3 cp infra/aws_pg/steps/next_n6_L2b.json $B/results/next_n6/L2b-01/steps.json $P
aws s3 cp $B/results/next_n6/L2b-01/steps.json - $P | cmp - infra/aws_pg/steps/next_n6_L2b.json
# L2a's record must be at $B/results/next_n6/L2a-01/out/edit/n6/metrics.json with complete true; the job reads it (--against)
./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/results/next_n6/L2b-01/steps.json" \
    --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n6/L2b-01 --budget-min 1779 --json --detach
```
First reads: `against` (L2a's four values, equal to L2b's), `half` test, then the gates as for L2a. After L2b: pull to
`studies/next/n6/banks/L2b-01/`, then
```
PYTHONPATH=. python3 -m infra.pg.n6 verdict --bank studies/next/n6/banks/L2a-01 --bank studies/next/n6/banks/L2b-01 \
    --pairs studies/next/n6/pairs_L2.json --thresholds studies/next/n6/thresholds.json --out studies/next/n6/verdict_n6.json
```
