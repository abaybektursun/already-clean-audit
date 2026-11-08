# 2026-11-05 (UTC 2026-11-06) — M2 score, attempt `results/next_m2/score-01`

Study: `studies/next/07_PICKS/M2_SPEC.md` §9.5 S1–S2. Worktree `interp_research-m2`, branch `m2`, HEAD `24c836932`
(the freeze-1 record; tag `m2-freeze-1` on `35ff9d7ce`, whose `infra/` is identical), working tree clean at launch, so
the code tarball `pg` ships equals that commit.

## Launch

| field | value |
|---|---|
| attempt | `score-01` (`results/next_m2/` was empty before the inputs were uploaded) |
| inputs | `results/next_m2/inputs/`: `births.json` and `exemplar_a_t00.npz` … `exemplar_a_t15.npz`; sha256 of all 17 equal to the registry before upload (laptop) and after (downloaded back) |
| instance | `i-0666dc626f797d452` |
| type, AZ, AMI | g6e.4xlarge (1x L40S), us-east-1c, `ami-0cee78ee4b1c24614` (`pg-env-v1`), on-demand |
| tries | 1. `--type g6e.2xlarge,g6e.4xlarge`; the launcher's JSON lists `tried: [g6e.2xlarge, g6e.4xlarge]` and took g6e.4xlarge |
| steps file | `s3://pg-lineage-769063704696/results/next_m2/score-01/steps.json` = `infra/aws_pg/steps/next_m2_score-01.json` (step `score`, job budget 150 min) |
| launch command | `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_m2/score-01/steps.json" --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_m2/score-01 --budget-min 165 --json --detach` |
| launch UTC (EC2 `LaunchTime`) | 2026-11-06T01:46:12Z |
| reaper deadline | 2026-11-06T04:41:12Z (`pg-reap-i-0666dc626f797d452`) |
| other instance running at launch | `i-05bc3bb26bd3188fd` (g6e.2xlarge, `results/next_n13/s2smoke-09`, launched 01:30:23Z, with a reaper; not M2). `pg audit` before launch: 0 reaper-less, 2 orphan reapers (`pg-reap-i-05e4d5bdfadce9f86`, `pg-reap-i-06445d4b72697db2d`), not from M2 |

## Watch

A laptop loop copied `out/score/m2/metrics.json`, read `STATUS`, the S3 time of `heartbeat.jsonl` and the EC2 state
every 120 s (`scratchpad/m2/poll/poll.log`). It never read `job.log` while the job ran.

| UTC | event (artifact) |
|---|---|
| 01:46:12 | `LaunchTime` |
| 01:48:35 | job started (`metrics.json` → `started_utc`) |
| 01:48:51 | G_SAME, INPUTS, G_CORPUS pass |
| 01:49–01:51 | T_15 download, 14.61 GB in 42.1 s (346.8 MB/s); load 7.8 s |
| 01:51:58 | K0 pass |
| 01:58:42 | T_15 done (elapsed 607 s); measured fixed cost 154.4 s per point, 0.2174 s per window, 9,446 tokens per second |
| 02:00:12 | T_14 G0 pass; download 333.8 MB/s; projection at T_14: 2,282 s remaining, budget 9,000 s (no parking) |
| 02:01:34 | `t14_G_EX` failed; the step exited 1 |
| 02:02:04 | `STATUS` `exit=1` (S3 `LastModified`); `out/pipeline_status.json` → `failed` |
| 02:03:25 | EC2 `shutting-down` (first poll after 02:01:22, when it was `running`) |
| 02:12:21 | EC2 `terminated` (first seen; polled every 15 s from about 02:09) |

## End

- Result: the job stopped on its own gate at T_14 (`t14_G_EX`, p99 relative difference 0.2563 against 0.0444). The
  science record and the cause are in `studies/next/m2/JOURNAL.md` and `studies/next/m2/diag/`. No relaunch: the
  head decides.
- `job.log` copied once after the end, with `STATUS`, `pipeline_status.json`, `heartbeat.jsonl`, `dmon.log` and
  `metrics.json`, to `studies/next/m2/banks/score-01/`. The rank files written at T_15 (`ranks_freeze.jsonl`,
  `ranks_test.jsonl`) and `vectors/t15/` stay on S3 unopened.
- Uptime billed: 01:46:12Z to `shutting-down`, which began between 02:02:04Z and 02:03:25Z: 0.264 to 0.287 h, about
  0.28 GPU-hours (EC2 stops billing at `shutting-down`). Cost at $3.00424 per hour (g6e.4xlarge, `09_PLAN.md`): $0.79
  to $0.86. M2's ceiling is 3.0 GPU-h; 2.7 remain.
- `pg audit` at 02:05:50Z: 1 running (the N13 instance), 4 reapers, 3 orphans, 0 reaper-less. The new orphan is
  `pg-reap-i-0666dc626f797d452`, a one-shot for this terminated instance, due 04:41:12Z. Not removed.
- The job is not acknowledged in the `pg` registry (`pg ack` not run), so `pg reconcile` still shows it as failed.
- Rates measured on this lane: download 333.8–346.8 MB/s (the projection assumed 143–234); forward 9,446 tokens per
  second on the L40S, 2.8 times the banked harvest rate the projection used (3,215–3,334), because this job does no
  all-pole top-K bookkeeping. A relaunch of the same plan would take about 607 s + 2,282 s of job time plus boot
  (`metrics.json` → `projection`), about 0.85–0.9 GPU-h.

# 2026-11-06 — M2 score, attempt `results/next_m2/score-02`: not launched (capacity)

Second executor. Worktree `interp_research-m2`, branch `m2`, HEAD `a7832a82e` (the freeze-1b record; tag
`m2-freeze-1b` on `1584962e0`, whose `infra/` is identical), working tree clean, so the code tarball `pg` packages
equals that commit.

## Before the launch

| field | value |
|---|---|
| `pg ps` / `pg audit` (02:26Z) | one instance running, `i-05bc3bb26bd3188fd` (g6e.2xlarge, N13, not M2, with a reaper); 4 reapers, 3 orphans (including `pg-reap-i-0666dc626f797d452` of score-01), 0 reaper-less |
| prefix | `results/next_m2/` listed `inputs/` and `score-01/` only; `score-02/` was empty |
| inputs | the 17 files of `results/next_m2/inputs/` downloaded to the scratchpad (`m2b/inputs_check02/`): 17 of 17 sha256 equal to `m2_score.BIRTHS_SHA` and `m2.EXEMPLAR_SHA` |
| steps file | `infra/aws_pg/steps/next_m2_score-01.json` (names no attempt) uploaded to `results/next_m2/score-02/steps.json` at 02:27:28Z; downloaded back, byte-equal |
| launch command | `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_m2/score-02/steps.json" --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_m2/score-02 --budget-min 165 --json --detach` |

## The six tries (the user's capacity rule of 2026-11-05)

Each try is one `pg run`. The launcher tries g6e.2xlarge, then g6e.4xlarge, each in the five subnets and then in an
EC2-chosen AZ (12 `run_instances` calls), and skips capacity errors. Starts were 180 s apart (scratchpad
`m2b/launch/tries.sh`, log `m2b/launch/tries.log`, outputs `m2b/launch/try1.out` … `try6.out`).

| try | start (UTC) | end (UTC) | exit | last error |
|---|---|---|---|---|
| 1 | 02:27:34 | 02:29:37 | 1 | `none of ['g6e.2xlarge', 'g6e.4xlarge'] launchable in any AZ`; last `('g6e.4xlarge', 'InsufficientInstanceCapacity')` |
| 2 | 02:30:34 | 02:32:41 | 1 | the same |
| 3 | 02:33:34 | 02:35:47 | 1 | the same |
| 4 | 02:36:34 | 02:38:58 | 1 | the same |
| 5 | 02:39:34 | 02:41:40 | 1 | the same |
| 6 | 02:42:34 | 02:44:39 | 1 | the same |

All six failed on capacity, so the executor stopped and reported to the head, as the rule says.

## After

- `pg ps` and `pg audit` at 02:45Z: the same N13 instance only; 4 reapers, 3 orphans, 0 reaper-less. No reaper was
  created for M2.
- No EC2 instance carries the tag `s3prefix = results/next_m2/score-02`.
- `results/next_m2/score-02/` holds only `steps.json`. No job wrote there: no STATUS, no metrics.json.
- GPU-hours 0, cost $0. M2 total: about 0.28 GPU-h (score-01) against the 3.0 ceiling; 2.7 remain.
- The error is `InsufficientInstanceCapacity` (AWS capacity), not a vCPU quota: the one running G instance holds 8
  of the 32 G-family vCPUs.

# 2026-11-06 — M2 score, attempt `results/next_m2/score-02` on `m2-freeze-1c`: launched

Third executor. Worktree `interp_research-m2`, branch `m2`, HEAD `589909cc8` (the freeze-1c record; tag `m2-freeze-1c`
on `6fa5c46dd`, whose `infra/` is identical; `git diff m2-freeze-1c HEAD -- infra/` is empty). The only uncommitted
change at launch was one journal line, outside what `pg` packages (`infra/`, `configs/`, `scripts/`), so the code
tarball equals the tag's.

## Before the launch

| field | value |
|---|---|
| `pg ps` / `pg audit` (02:56Z) | one instance running, `i-05bc3bb26bd3188fd` (g6e.2xlarge, N13, not M2, with a reaper); 4 reapers, 3 orphans, 0 reaper-less |
| prefix | reused with the head's permission: `results/next_m2/score-02/` held one object version, `steps.json` of 02:27:28Z, no delete marker; no job had written there |
| steps file | `score-02/steps.json` downloaded back, byte-equal to `infra/aws_pg/steps/next_m2_score-01.json` |
| inputs | the 17 files of `results/next_m2/inputs/` downloaded to the scratchpad (`m2c/inputs_check03/`): 17 of 17 sha256 equal to the registry |
| launch command | as for the first score-02 attempt (`--type g6e.2xlarge,g6e.4xlarge --budget-min 165 --json --detach`) |

## Tries (the head's capacity rule: 180 s apart for 12 tries, then 600 s, until 2 h after the first)

Scratchpad `m2c/launch/tries.sh`, log `m2c/launch/tries.log`, outputs `m2c/launch/try1.out` … `try4.out`.

| try | start (UTC) | end (UTC) | exit | result |
|---|---|---|---|---|
| 1 | 02:57:52 | 03:00:09 | 1 | `InsufficientInstanceCapacity` in every AZ (last g6e.4xlarge) |
| 2 | 03:00:52 | 03:02:57 | 1 | the same |
| 3 | 03:03:52 | 03:06:21 | 1 | the same |
| 4 | 03:06:52 | 03:09:06 | 0 | launched |

## Launch

| field | value |
|---|---|
| instance | `i-0a72805bc626c095d` |
| type, AZ, AMI | g6e.4xlarge (1x L40S), us-east-1a, `ami-0cee78ee4b1c24614` (`pg-env-v1`), on-demand |
| tries | the launcher's JSON lists `tried: [g6e.2xlarge, g6e.4xlarge]` and took g6e.4xlarge |
| launch UTC (EC2 `LaunchTime`) | 2026-11-06T03:09:05Z |
| reaper deadline | 2026-11-06T06:04:05Z (`pg-reap-i-0a72805bc626c095d`) |
| `pg audit` after launch (03:09Z) | 3 running: this one, the N13 instance `i-05bc3bb26bd3188fd` and `i-01d42a998e607b1d3` (m7g.medium, no job tag, not from M2); 6 reapers, 3 orphans, 0 reaper-less |

## Watch

A laptop loop (`scratchpad/m2c/poll.sh`) copies `out/score/m2/metrics.json`, reads `STATUS`, the S3 time of
`heartbeat.jsonl` and the EC2 state every 120 s (`m2c/poll/poll.log`, `events.log`). It never reads `job.log` while the
job runs.

| UTC | event (artifact) |
|---|---|
| 03:09:05 | `LaunchTime` |
| 03:11:58 | job started (`metrics.json` → `started_utc`) |
| 03:12:15 | G_SAME (torch 2.12.0+cu130, transformers 5.9.0), INPUTS, G_CORPUS pass; each failing input failed |
| 03:12–03:16 | T_15 download, 14.61 GB in 109.6 s (133.3 MB/s; score-01 had 346.8); load 9.8 s |
| 03:16:34 | K0 pass |
| 03:23:13 | T_15 done (elapsed 675 s); measured fixed cost 227.3 s per point (the slower download), 0.2149 s per window, 9,555 tokens per second |
| 03:26:27 | T_14 G-REV, G-CFG, G0 pass; download 100.2 MB/s; projection at T_14: 3,003 s remaining, budget 9,000 s (no parking) |
| 03:27:55 | T_14 done: `t14_G_EX` passes (p99 0.04034 against 0.044273), G-TOPK passes |
| 03:32:12 | T_13 G-REV, G-CFG, G0 pass |
| 03:34:15 | `t13_G_EX` failed (p99 0.05433 against 0.044273); the step exited 1 |
| 03:34:37 | `STATUS` `exit=1` (S3 `LastModified`); `out/pipeline_status.json` → `failed` |
| 03:36:08 | EC2 `shutting-down` (first poll after 03:34:05, when it was `running`); reason `Client.InstanceInitiatedShutdown` |
| 03:41:12 | EC2 `terminated` (first seen; polled every 20 s from 03:37) |

## End

- Result: the job stopped on its own gate at T_13 (`t13_G_EX`). The run DOES NOT COUNT: gates. The science record and
  the cause are in `studies/next/m2/JOURNAL.md` and `studies/next/m2/diag/score02_t13_gex.*`. No relaunch on freeze 1c;
  the head decided amendment A2 and one more attempt (`score-03`).
- `job.log` copied once after the end, with `STATUS`, `pipeline_status.json`, `heartbeat.jsonl`, `dmon.log` and
  `metrics.json`, to `studies/next/m2/banks/score-02/`. The rank files written at T_15 and T_14, the raw-example files
  and `vectors/t15/`, `vectors/t14/` stay on S3 unopened.
- Uptime billed: 03:09:05Z to `shutting-down`, which began between 03:34:37Z and 03:36:08Z: 0.426 to 0.451 h. Cost at
  $3.00424 per hour: $1.28 to $1.35.
- **M2 total so far: 0.690 to 0.738 GPU-hours** (score-01 0.264–0.287, score-02 0.426–0.451), $2.07 to $2.22, against
  the 3.0 ceiling: 2.26 to 2.31 remain. The bounds come from the 120 s poll; EC2 gives no time for an instance-initiated
  shutdown.
- Rates on this lane: download 133.3 MB/s at T_15 and 100.2 MB/s at T_14 (score-01 had 333.8–346.8), so the fixed cost
  rose to 227 s per point; forward 9,555 tokens per second. A full run of the plan would take about 675 s + 3,003 s of
  job time plus boot, about 1.05–1.1 GPU-h at these download rates.
- The orphan reaper `pg-reap-i-0a72805bc626c095d` (deadline 06:04:05Z) remains, as for score-01. Not removed.

# 2026-11-06 — M2 GPU-hour ceiling lifted

**2026-11-05 21:42 CST (2026-11-06 03:42Z).** The user removed the per-study GPU-hour ceilings: "Remove the GPU limits"
(the user's words, relayed by the head). M2's 3.0 GPU-h ceiling (`M2_SPEC.md` §9.6) no longer applies. The per-launch
hard kill stays as runaway protection: `--budget-min 165` (reaper at budget + 10 min) and the job's 150-minute stop
rule. Spent at that time: 0.690–0.738 GPU-h, $2.07–2.22.

# 2026-11-06 — M2 score, attempt `results/next_m2/score-03` on `m2-freeze-1d`: launched

Third executor. Worktree `interp_research-m2`, branch `m2`, HEAD `24ade6c69` (the freeze-1d record; tag `m2-freeze-1d` on
`b37c0b9df`; `git diff m2-freeze-1d HEAD -- infra/` is empty). The only uncommitted change at launch was one journal
line, outside what `pg` packages.

## Before the launch

| field | value |
|---|---|
| `pg ps` / `pg audit` (03:49Z) | no pg-lineage instance running; 4 reapers, 4 orphans, 0 reaper-less |
| prefix | `results/next_m2/score-03/`: no object version, no delete marker (fresh) |
| inputs | the 17 files of `results/next_m2/inputs/` downloaded to the scratchpad (`m2c/inputs_check04/`): 17 of 17 sha256 equal to the registry |
| steps file | `infra/aws_pg/steps/next_m2_score-01.json` uploaded to `score-03/steps.json`; downloaded back, byte-equal |
| launch command | `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_m2/score-03/steps.json" --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_m2/score-03 --budget-min 165 --json --detach` |
| capacity rule | the head's: 180 s apart for 12 tries, then 600 s, until 2 h after the first (scratchpad `m2d/launch/tries.sh`, log `m2d/launch/tries.log`) |

## Launch

| field | value |
|---|---|
| tries | 1 (03:49:39Z to 03:49:45Z, exit 0); the launcher's JSON lists `tried: [g6e.2xlarge, g6e.4xlarge]` and took g6e.2xlarge |
| instance | `i-0c403790a9756a0b7` |
| type, AZ, AMI | g6e.2xlarge (1x L40S), us-east-1d, `ami-0cee78ee4b1c24614` (`pg-env-v1`), on-demand, $2.24 per hour (`09_PLAN.md`) |
| launch UTC (EC2 `LaunchTime`) | 2026-11-06T03:49:45Z |
| reaper deadline | 2026-11-06T06:44:45Z (`pg-reap-i-0c403790a9756a0b7`) |
| `pg audit` after launch | 1 running, 5 reapers, 4 orphans, 0 reaper-less |

## Watch

The same loop as score-02 (`scratchpad/m2d/poll.sh`, `m2d/poll/poll.log`, `events.log`), every 120 s; never `job.log`.

| UTC | event (artifact) |
|---|---|
| 03:49:45 | `LaunchTime` |
| 03:52:54 | job started (`metrics.json` → `started_utc`) |
| 03:53:10 | G_SAME, INPUTS, G_CORPUS pass; each failing input failed |
| 03:53–03:57 | T_15 download, 14.61 GB in 77.8 s (187.8 MB/s); load 10.1 s |
| 03:57:14 | K0 pass |
| 04:03:57 | T_15 done (elapsed 664 s); measured fixed cost 195.5 s per point, 0.2173 s per window, 9,452.9 tokens per second |
| 04:05:37 | T_14 G-REV, G-CFG, G0 pass; download 279.8 MB/s; projection at T_14: 664 s elapsed, 2693 s remaining, budget 9,000 s |
| 04:07:06 | T_14 done: `t14_G_EX` (A2) passes (p99 0.02886 against 0.038718), G-TOPK passes |
| 04:08:54 | T_13 G-REV, G-CFG, G0 pass |
| 04:10:58 | `t13_G_EX` (A2) failed (591 read entries, p99 0.05191 against 0.038718); the step exited 1 |
| 04:11:34 | `STATUS` `exit=1` (S3 `LastModified`); `out/pipeline_status.json` → `failed` |
| 04:12:26 | EC2 `shutting-down` (first poll after 04:10:23, when it was `running`) |
| 04:17:44 | EC2 `terminated` (first seen; polled every 20 s) |

## End

- Result: the job stopped on its own gate at T_13 (`t13_G_EX` under A2). The run DOES NOT COUNT: gates. Science record
  and cause: `studies/next/m2/JOURNAL.md`, `studies/next/m2/diag/score03_t13_gex.*`. The head decided amendment A3 and a
  last attempt (`score-04`).
- `job.log` copied once after the end, with `STATUS`, `pipeline_status.json`, `heartbeat.jsonl`, `dmon.log` and
  `metrics.json`, to `studies/next/m2/banks/score-03/`. Rank files, raw-example files and vectors stay on S3 unopened.
- Uptime billed: 03:49:45Z to `shutting-down`, which began between 04:11:34Z and 04:12:26Z: 0.364 to 0.378 h. Cost at
  $2.24 per hour (g6e.2xlarge): $0.81 to $0.85.
- **M2 total over three attempts: 1.054 to 1.116 GPU-hours, $2.89 to $3.06** (score-01 0.264–0.287 on g6e.4xlarge,
  score-02 0.426–0.451 on g6e.4xlarge, score-03 0.364–0.378 on g6e.2xlarge). The ceiling is lifted (note above).
- Rates: download 187.8 MB/s at T_15 and 279.8 MB/s at T_14; forward 9,453 tokens per second. Projection at T_14: 664 s
  elapsed, 2,693 s remaining.
- The orphan reaper `pg-reap-i-0c403790a9756a0b7` (deadline 06:44:45Z) remains. Not removed.

# 2026-11-06 — M2 score, attempt `results/next_m2/score-04` on `m2-freeze-1e` (the last): launched

Third executor. Worktree `interp_research-m2`, branch `m2`, HEAD `7592eb2f6` (the freeze-1e record; tag `m2-freeze-1e` on
`ae2a6a43f`; `git diff m2-freeze-1e HEAD -- infra/` is empty). Uncommitted at launch: journal lines only.

## Before the launch

| field | value |
|---|---|
| `pg ps` / `pg audit` (04:20Z) | two N13 instances running (`i-02a1d57859ff38ea5` on `results/next_n13/s3gate-02`, `i-0f87b6f6fad1ff3aa` on `s3gate-01`, g6e.2xlarge, launched 04:07:5xZ); 0 reaper-less |
| prefix | `results/next_m2/score-04/`: no object version, no delete marker (fresh) |
| inputs | 17 of 17 sha256 equal to the registry (downloaded back to the scratchpad `m2e/inputs_check/`) |
| steps file | `infra/aws_pg/steps/next_m2_score-01.json` uploaded to `score-04/steps.json` at 04:20:54Z; downloaded back, byte-equal |
| launch command | as score-03 with `--s3-prefix results/next_m2/score-04` and the steps of `score-04` |

## Tries

| try | start (UTC) | end (UTC) | exit | result |
|---|---|---|---|---|
| 1 | 04:21:15 | 04:23:26 | 1 | `InsufficientInstanceCapacity` in every AZ (last g6e.4xlarge) |
| 2 | 04:24:15 | 04:25:52 | 1 | last error `VcpuLimitExceeded` on g6e.4xlarge: three other g6e.2xlarge held 24 of the 32 G-family vCPUs (the two N13 instances and `i-07d92ab4fda3a9812`, `results/bytegraft/bake/3`, launched 04:25:39Z). The first loop (`m2e/launch/tries.sh`) stopped on it by its own rule. |
| 3 | 04:27:15 | 04:27:52 | 0 | launched (resumed loop `m2e/launch/tries2.sh`, same two-hour window from 04:21:15Z, retrying on both errors) |

## Launch

| field | value |
|---|---|
| instance | `i-0cfccda25b48ab347` |
| type, AZ, AMI | g6e.2xlarge (1x L40S), us-east-1c, `ami-0cee78ee4b1c24614` (`pg-env-v1`), on-demand, $2.24 per hour |
| launch UTC (EC2 `LaunchTime`) | 2026-11-06T04:27:51Z |
| reaper deadline | 2026-11-06T07:22:51Z (`pg-reap-i-0cfccda25b48ab347`) |
| `pg audit` after launch | 4 running (this one, the two N13 instances, the bytegraft bake), 10 reapers, 6 orphans, 0 reaper-less |

## Watch

`scratchpad/m2e/poll.sh` (every 120 s; `m2e/poll/poll.log`, `events.log`); never `job.log`.

| UTC | event (artifact) |
|---|---|
| 04:27:51 | `LaunchTime` |
| 04:29:26 | job started (`metrics.json` → `started_utc`) |
| 04:29:43 | G_SAME, INPUTS, G_CORPUS pass; each failing input failed |
| 04:29–04:32 | T_15 download, 14.61 GB in 71.1 s (205.5 MB/s); load 7.2 s |
| 04:32:47 | K0 pass |
| 04:39:30 | T_15 done (elapsed 604 s); measured fixed cost 180.3 s per point, 0.2171 s per window, 9,458.2 tokens per second |
| 04:41:44 | T_14 G-REV, G-CFG, G0 pass; download 166.8 MB/s; projection at T_14: 604 s elapsed, 2541 s remaining, budget 9,000 s |
| 04:43:13 | T_14 done: G-EX reported (p99 0.02886, 4 above), G-TOPK passes |
| 04:48:25 | T_13 G-EX reported (p99 0.05191, 11 above); the job continues |
| 04:50:04 | T_13 done: G-TOPK and I-ID pass |
| 05:01:32 | T_12 done (download 29.8 MB/s): G-EX reported (p99 0.03246, 3 above); G-TOPK passes |
| 05:14:01 | T_11 done (download 24.9 MB/s): G-EX reported (p99 0.06398, 8 above); G-TOPK passes |
| 05:20:40 | T_10 G-REV, G-CFG, G0 pass; G-EX reported (p99 0.04298, 4 above) |
| 05:21:50 | `t10_G_TOPK` passes; then `m2.percentile` raises `AssertionError: a pole with no token of its sign` while scoring; the step exited 1 |
| 05:22:02 | `STATUS` `exit=1` (S3 `LastModified`); `out/pipeline_status.json` → `failed` |
| 05:23:26 | EC2 `shutting-down` (first poll after 05:21:23, when it was `running`) |
| 05:28:16 | EC2 `terminated` (first seen) |

## End

- Result: the job crashed in the scoring code at T_10 (not a gate). Every gate passed. The run DOES NOT COUNT
  (incomplete). Cause: two banked poles with no token of their own sign (L24N9826+ at T_10, L07N640− at T_8;
  `studies/next/m2/diag/score04_empty_poles.json`). No relaunch: the head's rule ends M2 on any stop of score-04.
- Weight downloads were slow on this attempt: 205.5 MB/s at T_15, then 166.8, 111.3, 29.8 (T_12), 24.9 (T_11) MB/s and
  about 6 min at T_10 (`job.log`). The stop rule's projection never came near its limit (at T_11: 1,926 s elapsed,
  1,534 s remaining, against 8,100 s).
- `job.log` copied once after the end, with `STATUS`, `pipeline_status.json`, `heartbeat.jsonl`, `dmon.log` and
  `metrics.json`, to `studies/next/m2/banks/score-04/`. The rank files (records of T_15–T_11), raw-example files and
  vectors stay on S3 unopened.
- Uptime billed: 04:27:51Z to `shutting-down`, which began between 05:22:02Z and 05:23:26Z: 0.903 to 0.926 h. Cost at
  $2.24 per hour: $2.02 to $2.08.
- **M2 total over four attempts: 1.957 to 2.042 GPU-hours, $4.91 to $5.14** (score-01 0.264–0.287 and score-02
  0.426–0.451 on g6e.4xlarge at $3.00424; score-03 0.364–0.378 and score-04 0.903–0.926 on g6e.2xlarge at $2.24).
- The orphan reaper `pg-reap-i-0cfccda25b48ab347` (deadline 07:22:51Z) remains. Not removed.

# 2026-11-06 — M2 score, attempt `results/next_m2/score-05` on `m2-freeze-1f` (amendment A4, the last relaunch)

Fourth executor. Worktree `interp_research-m2`, branch `m2`, HEAD `8efeba809` (the freeze-1f record; tag `m2-freeze-1f` on
`8599e3261`; `git diff m2-freeze-1f HEAD -- infra/ configs/ scripts/` is empty), working tree clean at launch.

## Before the launch

| field | value |
|---|---|
| `pg ps` / `pg audit` (05:45Z) | 4 running, none from M2: `i-0410dc8e9176b354d` (g6e.4xlarge, bytegraft_look), `i-08af333d381f921c4` (r7i.xlarge, bytegraft_ids), `i-02a1d57859ff38ea5` (g6e.2xlarge, pipeline, N13 `s3gate-02`), `i-024d99f0a61db28fb` (g6e.xlarge, bytegraft_env); 14 reapers, 10 orphans, 0 reaper-less. The three G instances hold 28 of the 32 G-family vCPUs |
| prefix | `results/next_m2/score-05/`: no object version, no delete marker (fresh) |
| inputs | the 17 files of `results/next_m2/inputs/` downloaded to the scratchpad (`m2f/inputs_check/`): 17 of 17 sha256 equal to the registry |
| steps file | `infra/aws_pg/steps/next_m2_score-01.json` uploaded to `score-05/steps.json` (05:46Z); downloaded back, byte-equal |
| launch command | `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_m2/score-05/steps.json" --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_m2/score-05 --budget-min 165 --json --detach` |
| capacity rule | every 180 s for six tries, then every 600 s until 2 h after the first; `InsufficientInstanceCapacity` and `VcpuLimitExceeded` retried; any other error stops (scratchpad `m2f/launch/tries.sh`, log `m2f/launch/tries.log`) |

## Tries

| try | start (UTC) | end (UTC) | exit | result |
|---|---|---|---|---|
| 1 | 05:46:37 | 05:46:56 | 1 | last error `VcpuLimitExceeded` on g6e.4xlarge (other jobs' G instances held 28 of the 32 G-family vCPUs) |
| 2 | 05:49:37 | 05:49:57 | 1 | the same |
| 3 | 05:52:37 | 05:52:56 | 1 | the same |
| 4 | 05:55:37 | 05:57:25 | 0 | launched (by 05:53:56Z two G instances, 12 vCPUs, were left) |

## Launch

| field | value |
|---|---|
| instance | `i-03bf9bcd82f6f7005` |
| type, AZ, AMI | g6e.4xlarge (1x L40S), us-east-1a, `ami-0cee78ee4b1c24614` (`pg-env-v1`), on-demand, $3.00424 per hour |
| tries | the launcher's JSON lists `tried: [g6e.2xlarge, g6e.4xlarge]` and took g6e.4xlarge |
| launch UTC (EC2 `LaunchTime`) | 2026-11-06T05:57:24Z |
| reaper deadline | 2026-11-06T08:52:24Z (`pg-reap-i-03bf9bcd82f6f7005`) |
| `pg audit` after launch | 3 running, 14 reapers, 11 orphans, 0 reaper-less |

## Watch

`scratchpad/m2f/poll.sh` (every 120 s: `metrics.json`, `STATUS`, heartbeat time, EC2 state; `m2f/poll/poll.log`,
`events.log`); never `job.log`.

| UTC | event (artifact) |
|---|---|
| 05:57:24 | `LaunchTime` |
| 05:58:59 | job started (`metrics.json` → `started_utc`) |
| 05:59:15 | G_SAME, INPUTS, G_CORPUS pass; each failing input failed |
| 05:59–06:05 | T_15 download, 14.61 GB in 233.5 s (62.5 MB/s) |
| 06:05:13 | K0 pass |
| 06:12:02 | T_15 done (elapsed 783 s); measured fixed cost 343.8 s per point, 0.2209 s per window, 9,296.3 tokens per second |
| 06:15:43 | T_14 done (download 180.3 MB/s) |
| 06:21:40 | T_13 done, with I-ID (download 202.6 MB/s) |
| 06:27–06:39 | T_12, T_11, T_10, T_9, T_8, T_7 done (downloads 155.8–311.3 MB/s); projection at T_6: 2,429 s elapsed, 691 s remaining, budget 9,000 s |
| 06:40:27 | `t06_G0`: the gate passed, its failing input passed too; `Run.gate` asserted; the step exited 1 |
| 06:40:58 | `STATUS` `exit=1` (S3 `LastModified`); `out/pipeline_status.json` → `failed` |
| 06:42:46 | EC2 `shutting-down` (first poll after 06:40:43, when it was `running`); reason `Client.InstanceInitiatedShutdown` |
| 06:48:06 | EC2 `terminated` (first seen; polled every 15 s from 06:45); `pg audit` then: 2 running, 10 reapers, 8 orphans, 0 reaper-less |

## End

- Result: the job stopped on its own gate at T_6: `t06_G0`'s failing input (the activation swap) passed, max ratio
  0.00970 under G0_TOL 0.01. Every other gate passed through T_7. The run DOES NOT COUNT: gates. The science record and
  the cause are in `studies/next/m2/JOURNAL.md` and `studies/next/m2/diag/score05_g0_fail.*`. No relaunch: the head's
  rule for the last relaunch.
- Weight downloads were fast after T_15 (155.8–311.3 MB/s; T_15 62.5 MB/s). The stop rule's projection never came near
  its limit.
- `job.log` copied once after the end, with `STATUS`, `pipeline_status.json`, `heartbeat.jsonl` and `metrics.json`, to
  `studies/next/m2/banks/score-05/` (no `dmon.log` was written on S3). The rank files (records of T_15–T_7), the
  raw-example files and `vectors/` stay on S3 unopened.
- Uptime billed: 05:57:24Z to `shutting-down`, which began between 06:40:58Z and 06:42:46Z: **0.726 to 0.756 h**. Cost
  at $3.00424 per hour (g6e.4xlarge): **$2.18 to $2.27**.
- **M2 total over five attempts: 2.683 to 2.798 GPU-hours, $7.09 to $7.41** (score-01 0.264–0.287, score-02
  0.426–0.451 and score-05 0.726–0.756 on g6e.4xlarge at $3.00424; score-03 0.364–0.378 and score-04 0.903–0.926 on
  g6e.2xlarge at $2.24). The ceiling is lifted (§9.6 note).
- The orphan reaper `pg-reap-i-03bf9bcd82f6f7005` (deadline 08:52:24Z) will remain after termination. Not removed.

# 2026-11-06 — M2 score, attempt `results/next_m2/score-06` on `m2-freeze-1g` (amendment A5, the last attempt)

Fifth executor. Worktree `interp_research-m2`, branch `m2`, HEAD `ac71ca76b` (the freeze-1g record; tag `m2-freeze-1g` on
`9a8e1a6a1`; `git diff m2-freeze-1g HEAD -- infra/ configs/ scripts/` is empty), working tree clean at launch.

## Before the launch

| field | value |
|---|---|
| `pg ps` / `pg audit` (07:08Z) | 2 running, none from M2: `i-02a1d57859ff38ea5` (g6e.2xlarge, pipeline, N13 `s3gate-02`), `i-020022d38d614bca0` (g6e.4xlarge, bytegraft sandbox, launched 07:06:39Z); 10 reapers, 8 orphans, 0 reaper-less |
| prefix | `results/next_m2/score-06/`: no object version, no delete marker (fresh) |
| inputs | the 17 files of `results/next_m2/inputs/` downloaded to the scratchpad (`m2g/inputs_check/`): 17 of 17 sha256 equal to the registry |
| steps file | `infra/aws_pg/steps/next_m2_score-01.json` uploaded to `score-06/steps.json` (07:08:10Z); downloaded back, byte-equal |
| launch command | `./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_m2/score-06/steps.json" --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_m2/score-06 --budget-min 165 --json --detach` |
| capacity rule | every 180 s for six tries, then every 600 s until 2 h after the first; `InsufficientInstanceCapacity` and `VcpuLimitExceeded` retried; any other error stops (scratchpad `m2g/launch/tries.sh`, log `m2g/launch/tries.log`) |

## Tries

| try | start (UTC) | end (UTC) | exit | result |
|---|---|---|---|---|
| 1 | 07:08:21 | 07:08:38 | 1 | last error `VcpuLimitExceeded` on g6e.4xlarge (the two other G instances held 24 of the 32 G-family vCPUs) |
| 2 | 07:11:21 | 07:12:18 | 0 | launched |

## Launch

| field | value |
|---|---|
| instance | `i-00e24f621cc4be70c` |
| type, AZ, AMI | g6e.2xlarge (1x L40S), us-east-1d, `ami-0cee78ee4b1c24614` (`pg-env-v1`), on-demand, $2.24 per hour |
| tries | the launcher's JSON lists `tried: [g6e.2xlarge, g6e.4xlarge]` and took g6e.2xlarge |
| launch UTC (EC2 `LaunchTime`) | 2026-11-06T07:12:17Z |
| reaper deadline | 2026-11-06T10:07:17Z (`pg-reap-i-00e24f621cc4be70c`) |
| `pg audit` after launch (07:13Z) | 2 running (this one and the N13 instance), 11 reapers, 9 orphans, 0 reaper-less |

## Watch

`scratchpad/m2g/poll.sh` (every 120 s: `metrics.json`, `STATUS`, heartbeat time, EC2 state; `m2g/poll/poll.log`,
`events.log`); never `job.log`.

| UTC | event (artifact) |
|---|---|
| 07:12:17 | `LaunchTime` |
| 07:13:31 | job started (`metrics.json` → `started_utc`) |
| 07:13:47 | G_SAME, INPUTS, G_CORPUS pass; each failing input failed |
| 07:13–07:16 | T_15 download, 14.61 GB in 46.3 s (315.7 MB/s) |
| 07:16:37 | K0 pass |
| 07:23:18 | T_15 done; measured fixed cost 159.8 s per point, 0.2156 s per window, 9,525.9 tokens per second |
| 07:26:24 | T_14 done |
| 07:33:41 | T_13 done, with I-ID |
| 07:36–07:48 | T_12, T_11, T_10, T_9, T_8, T_7 done (downloads 178.1–321.4 MB/s) |
| 07:50:12 | T_6 done (G-EX and G-TOPK gated, pass) |
| 07:50:25 | T_5 done; `complete` true; projection at T_5: 2,137 s elapsed, 162 s remaining, budget 9,000 s |
| 07:50:44 | `STATUS` `exit=0` (S3 `LastModified`); `out/pipeline_status.json` → `complete` |
| 07:52:26 | EC2 `shutting-down` (first poll after 07:50:23, when it was `running`); reason `Client.InstanceInitiatedShutdown` |
| 07:57:30 | EC2 `terminated` (first seen; not polled between); `pg audit` then: 1 running, 9 reapers, 8 orphans, 0 reaper-less |

## End

- Result: the job completed (`M2_JOB_OK`, `PIPELINE_OK`): 52 gates passed, every failing input failed, nothing parked.
  Freeze 2 and the verdict were run on the laptop (`studies/next/m2/JOURNAL.md`; `M2_SPEC.md` §0, §12). Not part of
  this note.
- Weight downloads were fast at every point: 178.1–321.4 MB/s (T_11 the slowest). The stop rule's projection never came
  near its limit.
- Banked once after the end to `studies/next/m2/banks/score-06/`: `STATUS`, `job.log`, `heartbeat.jsonl`, `dmon.log`
  (from `out/dmon.log`), `pipeline_status.json`, and under `m2/` the job's `metrics.json`, `ranks_freeze.jsonl`,
  `samples/raw_examples_freeze.jsonl`, then (after the tag `m2-freeze-2`) `ranks_test.jsonl` and
  `samples/raw_examples_test.jsonl`. `vectors/` stays on S3 for the independent recount.
- Uptime billed: 07:12:17Z to `shutting-down`, which began between 07:50:44Z and 07:52:26Z: **0.641 to 0.669 h**. Cost at
  $2.24 per hour (g6e.2xlarge): **$1.44 to $1.50**.
- **M2 total over six attempts: 3.324 to 3.467 GPU-hours, $8.53 to $8.91** (score-01 0.264–0.287, score-02 0.426–0.451
  and score-05 0.726–0.756 on g6e.4xlarge at $3.00424; score-03 0.364–0.378, score-04 0.903–0.926 and score-06
  0.641–0.669 on g6e.2xlarge at $2.24). The ceiling is lifted (§9.6 note).
- The orphan reaper `pg-reap-i-00e24f621cc4be70c` (deadline 10:07:17Z) remains, as for the earlier attempts. Not removed.
