# J8 rerun b2 on the AWS lane, 2026-11-05: no instance (g6e.2xlarge capacity)

**Update, 2026-11-05T22:10Z: b2 ran in a second launch round.** Instance `i-0bc4d228306510dd5`, g6e.4xlarge,
launched 21:28:14Z on try 5 of that round, STATUS `exit=0` at 21:45:18Z. At most 0.313 GPU-hours, about $0.94. Section
"Second launch round and the run" at the end. The text below it up to that section is the first round, kept as written.

The rerun of runs 4 and 5 of `infra.aws_pg.jobs.next_pilot0_loss` was meant to run `batch_100000/` at 512 rows and
the 32 known-number rows of `batch_60001/`. It did not launch. All nine attempts, 18:38–19:13Z, ended in
`RuntimeError: none of ['g6e.2xlarge'] launchable in any AZ ... last=('g6e.2xlarge', 'InsufficientInstanceCapacity')`.
Cost 0 GPU-hours. The science record is `studies/next/pilot0/GPU_CHECK2.md`, section "Rerun b2".

Worktree `interp_research-clean-neurons`, branch `clean-neurons`, HEAD `841432df7`. Command, from the worktree root:

```
./infra/aws_pg/pg run infra.aws_pg.jobs.next_pilot0_loss --args " --runs 4,5" --no-weights --type g6e.2xlarge \
    --s3-prefix results/next_pilot0/2026-11-05_b2 --budget-min 40 --detach --json
```

Only g6e.2xlarge was tried, as the brief allowed. Attempt 1 ran by hand. Attempts 2–9 were run by a script,
180 s after the previous attempt ended (`$SP/j8/b2/retry_launch.sh`, log `$SP/j8/b2/launch_attempts.log`; `$SP` is
this session's scratchpad). Each `pg run` tries the five configured subnets in order and then an EC2-chosen AZ. The
error names only the last code.

| attempt | start (UTC) | end (UTC) | rc |
|---|---|---|---|
| 1 | 18:38:23 | 18:39:32 | 1 |
| 2 | 18:42:49 | 18:44:06 | 1 |
| 3 | 18:47:06 | 18:48:14 | 1 |
| 4 | 18:51:14 | 18:52:26 | 1 |
| 5 | 18:55:26 | 18:56:27 | 1 |
| 6 | 18:59:27 | 19:00:44 | 1 |
| 7 | 19:03:44 | 19:04:49 | 1 |
| 8 | 19:07:49 | 19:09:02 | 1 |
| 9 | 19:12:02 | 19:13:08 | 1 |

## State afterwards (19:13:15Z)

- `describe-instances` with tag `s3prefix=results/next_pilot0/2026-11-05_b2`: no instance.
- Registry: no record names the prefix. No reaper was armed, because `_arm_deadline` runs only after a launch.
- `pg ps`: two g6e.2xlarge `pipeline` instances from other work, `i-0bcf8476da3757298` and `i-01ca98670df35bb1c`.
  At 18:37:12Z a third was running, `i-012b11b08014eaac0`. None is this run's. Untouched.
- `pg audit`: `2 running, 2 reapers, 0 orphans, 0 reaper-less`. Clean.
- `s3://pg-lineage-769063704696/results/next_pilot0/2026-11-05_b2/` holds only the five inputs uploaded at
  18:38:01–07Z: `loss_check.py` and `batch_100000/` and `batch_60001/` (`tokens.npy`, `meta.json`). No STATUS and no
  `out/`.

## Unusual

1. The error was capacity, not quota. Two or three of our own g6e.2xlarge instances were running throughout,
   8–12 vCPU of the 32-vCPU G quota. That quota would have returned `VcpuLimitExceeded`, which was not the last
   code.
2. Every attempt uploaded a code tarball before `run_instances` was called (`package_code` runs first): nine objects
   of 41,392,048 B under `code/`, 18:38:28Z to 19:12:07Z. Each has a different hash, although the worktree did not
   change between attempts. The cause was not checked. Timestamps in the tar or gzip headers would explain it.
   No instance used them. They stay, because the bucket denies deletes.

## Second launch round and the run (2026-11-05, 21:06–21:55Z)

Command, from the worktree root. g6e.4xlarge was added as the second type. It carries the same single NVIDIA L40S
as g6e.2xlarge, at $3.00424 per hour.

```
./infra/aws_pg/pg run infra.aws_pg.jobs.next_pilot0_loss --args " --runs 4,5" --no-weights --type g6e.2xlarge,g6e.4xlarge \
    --s3-prefix results/next_pilot0/2026-11-05_b2 --budget-min 40 --detach --json
```

HEAD at the try that launched was `316ef6d25` (committed 21:16:33Z). `infra/aws_pg/jobs/next_pilot0_loss.py` and
`studies/next/pilot0/loss_check.py` were last changed in `26d0cd9aa`. The `loss_check.py` input on S3 has ETag
`02d2ebdd…`, equal to the md5 of the repo file.

Prefix. The head reused `results/next_pilot0/2026-11-05_b2` instead of a fresh prefix. The nine tries of the first
round and the four failed tries of this round produced no instance, no STATUS and no output under it. Its only
objects were the five inputs the job reads.

Try 1 ran by hand. Tries 2–5 were run by `$SP/j8/b2/retry_launch2.sh`, 180 s after the previous try ended. Log:
`$SP/j8/b2/launch_attempts2.log`. Each try first checked the state of the killed N13 instance `i-0332664b342a35308`.

| try | start (UTC) | end (UTC) | rc | last error, or result |
|---|---|---|---|---|
| 1 | 21:06:49 | — | 1 | `('g6e.4xlarge', 'VcpuLimitExceeded')`; the killed N13 instance was still shutting down |
| 2 | 21:10:37 | 21:13:07 | 1 | `('g6e.4xlarge', 'InsufficientInstanceCapacity')`; N13 instance terminated |
| 3 | 21:16:08 | 21:18:18 | 1 | `('g6e.4xlarge', 'InsufficientInstanceCapacity')` |
| 4 | 21:21:18 | 21:23:40 | 1 | `('g6e.4xlarge', 'InsufficientInstanceCapacity')` |
| 5 | 21:26:41 | 21:28:15 | 0 | `i-0bc4d228306510dd5`, g6e.4xlarge, us-east-1b; `tried` g6e.2xlarge then g6e.4xlarge; reaper `at(2026-11-05T22:18:14)` |

The error of tries 2–4 reads `none of ['g6e.2xlarge', 'g6e.4xlarge'] launchable in any AZ`. It names only the last
code, so the g6e.2xlarge codes of this round are not in the log. Every try uploaded a code tarball before
`run_instances`: five objects of 41,393,544 B under `code/`, at 21:06:55, 21:10:43, 21:16:13, 21:21:24 and
21:26:46Z, each with a different name. They stay, because the bucket denies deletes.

### The run

| time (UTC) | event | source |
|---|---|---|
| 21:28:14 | launch | `describe-instances` → `LaunchTime` |
| 21:28:51 | first heartbeat, no job process yet | `heartbeat.jsonl` |
| 21:29:24 | job process (pid 2013) in the heartbeat | `heartbeat.jsonl` |
| — | `stage1-step100000`: 14,605,764,269 B in 165.6 s (88.2 MB/s), blocking | `out/metrics.json` → `downloads` |
| — | run 4, `batch_100000/`, 512 rows: 700.2 s wall, 626.8 s forward, 8.1 s load | `out/metrics.json` → `runs`; the run file |
| — | `stage1-step60000`: 14,605,764,269 B in 196.5 s (74.3 MB/s), in the background during run 4 | `out/metrics.json` |
| 21:44:01 | `out/batch_100000/loss_check_sdpa.json` last written | S3 `LastModified` |
| — | run 5, `batch_60001/`, 32 rows: 50.0 s wall, 38.8 s forward, 2.6 s load, download wait 0.0 s | `out/metrics.json` |
| 21:45:14 | `metrics.json` `updated_utc`, stage `complete`; heartbeat shows no job process | `out/metrics.json`, `heartbeat.jsonl` |
| 21:45:18 | `STATUS` = `exit=0` | S3 `LastModified` |
| after 21:45:17 | OS power-down at kernel uptime 1,022.9 s | `aws ec2 get-console-output --latest` |
| 21:47 | `shutting-down` | the head's poll |
| 21:51:00 | `shutting-down`, `StateReason` `Client.InstanceInitiatedShutdown` | `describe-instances`, this session |
| 21:55:45 | `terminated` | `describe-instances`, this session |

Peak GPU memory 21.19 GB in both runs (`max_memory_allocated_gb`). `out/dmon.log` shows the GPU at 100 % SM during
run 4. At boot the AMI lacked `tiktoken`, `google.protobuf` and `sentencepiece`, and the setup installed them
(`job.log`, `[deps]` lines). Both downloads ran at 74–88 MB/s, faster than the 25–42 MB/s the budget assumed.
The job ended about 16 minutes after it started, far inside the 40-minute `timeout`.

Objects at the prefix afterwards: `STATUS`, `heartbeat.jsonl`, `job.log`, the five inputs, and under `out/`:
`metrics.json`, `dmon.log`, `batch_100000/loss_check_sdpa.json`, `batch_60001/loss_check_sdpa_rows32.json`. The
four `out/` files were downloaded and their sha256 matched the scratchpad copies (`$SP/j8/b2/result/`).

The reaper schedule `pg-reap-i-0bc4d228306510dd5` was still `ENABLED` at 21:55Z (`at(2026-11-05T22:18:14)`,
`ActionAfterCompletion` `DELETE`). It will find the instance already terminated.

### GPU-hours and cost

EC2 stops billing when the instance enters `shutting-down`. The exact time of that transition is not in the
instance record.

| end point | seconds from 21:28:14Z | GPU-hours | cost at $3.00424/h |
|---|---|---|---|
| `STATUS` written, 21:45:18Z (the OS power-down, at kernel uptime 1,022.9 s, is at or after 21:45:17Z) | 1,024 | 0.284 | $0.85 |
| `shutting-down` seen by 21:47:00Z (the ceiling used here) | 1,126 | 0.313 | $0.94 |

b2 used at most 0.313 GPU-hours, inside its 0.85 ceiling. J8 as a whole: b1 0.851 (about $1.91 at $2.24/h) plus b2 at
most 0.313 gives at most 1.164 GPU-hours, about $2.85. That is above the registered 1.0 and inside the 1.70
deviation ceiling the head set on 2026-11-05 at 10:55 CST.
