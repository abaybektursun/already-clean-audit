# MECHANICS screen run on the AWS lane, 2026-11-04

One `pg run infra.aws_pg.jobs.mechanics_screen` launch from the worktree `interp_research-mechanics` (branch `research/mechanics`, HEAD `05083f686`, tag `mechanics-freeze-1`; the job's `pg_git_sha` reads `05083f6867b2`). Fresh prefix `results/mechanics/screen/1` (checked empty before launch). Watched with `pg watch` and by polling `out/metrics.json`; `job.log` not read. Nothing scored by the runner.

Launch command (from the worktree root, after `ln -s ../interp_research-reuse/.pgvenv .pgvenv` as BUILD_NOTES_mechanics.md states):

```
./infra/aws_pg/pg run infra.aws_pg.jobs.mechanics_screen --args " --t T_15 --revision stage1-step1413814 --corpus s3://pg-lineage-769063704696/results/mdl/neurons/w1/q4b/1/out/corpus/corpus --bank s3://pg-lineage-769063704696/results/neurons/w3/grid/1/out/t15_step1413814" --no-weights --budget-min 120 --type g6e.2xlarge,g6e.4xlarge,g6e.xlarge --s3-prefix results/mechanics/screen/1 --detach --json
```

| prefix | instance | type | command issued (UTC) | instance launch | first metrics.json | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|---|---|
| `results/mechanics/screen/1` | `i-0314130d9dcf79292` | g6e.4xlarge (us-east-1c) | 19:42:35Z | 19:44:38Z | 19:50:51Z | 19:55:28Z, `exit=0` | 0.181 h (650 s) | metrics `status: done`; job wall 486.4 s (download, corpus and bank 140.5 s, load 101.8 s, run 243.9 s); 144 forwards, 1.12 M tokens; peak GPU memory 27.5 GB on 1x L40S |

Cost at the on-demand price of $3.00424 per hour for a g6e.4xlarge (AWS Price List API, us-east-1, Linux, shared tenancy, read 2026-11-04): 0.181 h, about $0.54 (billing stops at shutting-down, which followed STATUS; the state read `terminated` by 20:02:14Z, 0.293 h after launch, $0.88 as an upper bound).

Teardown: the instance self-terminated (`Client.InstanceInitiatedShutdown`). The launcher's reaper schedule `pg-reap-i-0314130d9dcf79292` (`at(2026-11-04T21:54:38)`) was still enabled after termination and was deleted. `pg audit` afterwards: `audit: 0 running, 0 reapers, 0 orphans, 0 reaper-less / clean.`

Bank: S3 `s3://pg-lineage-769063704696/results/mechanics/screen/1/out/` (3,503 objects, 440.8 MB). Pulled to `studies/mechanics/results/screen/1/` (every object except `dmon.log`; 3,502 files, sizes checked equal to the S3 listing). Committed: `metrics.json`, `samples/mechanics.jsonl`, `samples/known_activation.json`, and in a separate commit the 3,494 `contexts/<pole>.jsonl` files (335.0 MB). Not committed: the npz files (`signatures.npz` 71.3 MB, `flagged.npz` 33.0 MB, `contexts/flags.npz` 1.3 MB, `heads.npz` 25 KB; 105.6 MB total, over the 50 MB line) and `families.json`; their copies of record are at the S3 prefix above and on local disk.

Unusual:
1. The first type, g6e.2xlarge, did not launch; the launcher fell back to g6e.4xlarge. `pg run` took about 2 min (19:42:35Z to 19:44:39Z) before returning the instance.
2. The contexts directory is much larger than the dry run's (3,494 pole files against 118). It is committed on its own so it can be dropped with one reset before a merge if it should not enter git history.
3. The scheduler also lists `shackleton-reaper-5min`, which is not this run's; untouched. Instance `i-0762252be3d0877cc` (another session) was not touched; `pg ps` before launch listed no pg-lineage instance.
