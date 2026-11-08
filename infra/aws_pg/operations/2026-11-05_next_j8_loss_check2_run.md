# J8 second data-order loss check on the AWS lane, 2026-11-05

One `pg run infra.aws_pg.jobs.next_pilot0_loss` launch from the worktree `interp_research-clean-neurons` (branch
`clean-neurons`, HEAD `5070a539d`, uncommitted changes to the job module and `studies/next/pilot0/`; the job's
`PG_GIT_SHA` reads `5070a539d248-dirty`; code tarball `s3://pg-lineage-769063704696/code/85df697618f4.tgz`).
Fresh prefix `results/next_pilot0/2026-11-05_b1` (empty before the input upload). Polled `out/metrics.json`,
`STATUS` and the instance state every 5 min; `job.log` not polled. The science record is
`studies/next/pilot0/GPU_CHECK2.md`.

Launch command (from the worktree root):

```
./infra/aws_pg/pg run infra.aws_pg.jobs.next_pilot0_loss --no-weights --type g6e.2xlarge \
    --s3-prefix results/next_pilot0/2026-11-05_b1 --budget-min 49 --detach --json
```

Instance `i-03e36825423328376`, g6e.2xlarge, us-east-1b, launched 15:50:08Z. Reaper `pg-reap-i-03e36825423328376`,
`at(2026-11-05T16:49:09)`. In-instance budget 2,940 s.

## Deviation, 2026-11-05 (16:20Z): extension decided, not applied

The head decided to extend the self-termination by 15 minutes (ceiling 1.0 → 1.25 GPU-hours) if the launcher can
change the reaper or budget of a running job without a relaunch. It cannot. The user-data passes budget `2940` to
`run_job.sh`, which runs the job under `timeout --signal=KILL 2940s`, and the exit trap then shuts the instance down.
`pg watchdog`, `reconcile` and `register` do not change a running job (they classify, or write the S3 registry
record). The instance has no key pair and no SSM registration. The reaper schedule could be moved with the
scheduler API, but that would not outlast the in-instance timeout, so it was left at 16:49:09Z. Nothing was
changed; the job ends at about 16:40Z.

Cause of the shortfall: the first checkpoint download (`stage1-step61000`, 14.6 GB from the HF hub) took 435.7 s
(`out/metrics.json` → `downloads`), against 75.2–102.4 s in pilot 0. The 49-minute budget had been sized on the
pilot-0 download time.

## Timeline (UTC)

| event | time | source |
|---|---|---|
| `pg run` issued | 15:49:03Z | laptop clock |
| instance launch | 15:50:08Z | `describe-instances` → `LaunchTime` |
| job process running | by 15:51:29Z | `heartbeat.jsonl` (first sample with a python pid) |
| download `stage1-step61000` done (blocking) | about 15:59:49Z, 435.7 s | `out/metrics.json` → `downloads`; `job.log` |
| run 1 done (`batch_61001`, 512 rows) | about 16:12:00Z, 730.2 s | `out/metrics.json` → `runs` |
| download `stage1-step100000` (background) | 351.8 s | `job.log` |
| run 2 done (`batch_61000`, 512 rows) | about 16:22:40Z, 640.2 s | `out/metrics.json` |
| download `stage1-step60000` (background) | 448.0 s | `job.log` |
| run 3 done (`batch_100001`, 512 rows) | about 16:33:20Z, 640.2 s | `out/metrics.json` |
| run 4 killed (`batch_100000`, 320 rows) | 16:40:10Z (last `metrics.json` write) | `job.log` line 140: `timeout --signal=KILL "${BUDGET}s"` … `Killed` |
| STATUS `exit=137` | 16:40:19Z | S3 object time |
| state shutting-down | by 16:41:11Z | watcher poll |
| state terminated | by 16:49:46Z (`Client.InstanceInitiatedShutdown`) | `describe-instances` |
| reaper `pg-reap-i-03e36825423328376` | fired and deleted itself at 16:49:09Z; gone at 16:49:46Z | `scheduler get-schedule` |

Run 5 (the known number) never started.

## Cost

At most 0.851 h of instance time (launch to the first shutting-down reading), about $1.91 at the plan's $2.24 per
hour for g6e.2xlarge. Inside the J8 ceiling of 1.0 GPU-hours.

## Teardown and records

- `pg ps` afterwards: `no pg-lineage instances running -> $0 GPU spend`.
- `pg audit` afterwards: `0 running, 2 reapers, 2 orphans, 0 reaper-less`. The two orphans,
  `pg-reap-i-02f773eedd54233d5` and `pg-reap-i-047efd8fa80c4254c`, are not this run's. Untouched.
- `pg watchdog results/next_pilot0/2026-11-05_b1`: `FAILED (STATUS exit=137)`. The job was then acknowledged
  (`pg ack`) after the outputs were pulled, as the pilot-0 attempts were.
- Pulled: `out/batch_{61001,61000,100001,100000}/loss_check_sdpa.json` to `studies/next/pilot0/batch_<s>/`;
  `out/metrics.json`, `out/dmon.log`, `heartbeat.jsonl`, `job.log` and `STATUS` to the session scratchpad
  (`.../scratchpad/j8/b1/`). `job.log` was read once, after the job ended.

## Unusual

1. HF hub downloads ran at 33–42 MB/s per checkpoint today, against about 140–195 MB/s in pilot 0 on the same
   instance type and AMI. The 49-minute budget was sized on the pilot-0 rate. A rerun should size the budget on
   today's rate, or use a launch where the checkpoint is already in the S3 weight cache.
2. The AMI is still missing `tiktoken`, `protobuf` and `sentencepiece`. `run_job.sh` installed them at start, as
   designed.
3. A running job's budget cannot be changed: the limit is the `timeout` inside `run_job.sh`, and instances have no
   key pair and no SSM registration.
