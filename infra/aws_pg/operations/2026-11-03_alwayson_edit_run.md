# ALWAYSON SPEC2 edit run on the AWS lane, 2026-11-03

One `pg run infra.aws_pg.jobs.alwayson_edit` launch from the `interp_research-reuse` worktree (the alwayson worktree has no `.pgvenv`; `research/alwayson` was merged into `research/reuse` at `f31bb7a54` first and `PG_GIT_SHA` records that merge), `--type g6e.2xlarge,g6e.4xlarge --no-weights --budget-min 150`, fresh prefix `results/alwayson/edit/1`, watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/alwayson/edit/1` | `i-015ebcbfa83094e50` | g6e.2xlarge (us-east-1c) | 08:49:44Z | 10:17:30Z | 1.463 h (5,266 s) | job started 08:52:27Z, finished 10:17:08Z (wall 5,156 s); transformers 5.20.2 installed in-job over the AMI's 5.9.0; the Gated DeltaNet blocks on the torch reference path; pilot 8 problems × 5 arms at 227 to 235 tokens/s (batch 8), then 200 problems × 5 arms at 868 to 966 tokens/s (batch 50, 4,096 steps per batch, 212 to 231 s per batch, 873 to 875 s per arm); peak memory allocated 7.9 GB of the 47.7 GB |

Cost at the on-demand price of about $2.24 per hour for a g6e.2xlarge (the price is quoted here, not banked): about $3.28, an estimate. The orphan reaper schedule (`pg-reap-i-015ebcbfa83094e50`) deleted after the run; `pg audit` clean (0 running, 0 reapers). API $0. The run was kept after the pilot showed the frozen loop flag saturated (every pilot answer hit the cap inside its thinking block); the decision and the exploratory amendment are in the science documents, not here.

Science: `studies/alwayson/results/RESULTS.md` (the SPEC2 section), `studies/alwayson/results/REVIEW_edit.md`, REUSE journal entries 2026-11-03 02:50 CST (08:50Z), 03:12 CST (09:12Z) and the close-out entry.
