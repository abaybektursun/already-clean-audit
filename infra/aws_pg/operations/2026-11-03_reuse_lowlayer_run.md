# REUSE SPEC12 lowlayer capture on the AWS lane, 2026-11-03

One `pg run infra.aws_pg.jobs.pipeline` launch from the `interp_research-reuse` worktree (`research/lowlayer` merged into `research/reuse` at `78d0ff611` first; `PG_GIT_SHA` records that merge), `--no-weights --budget-min 180`, fresh prefix `results/reuse/lowlayer/1`, 16 loads (T_15 first, then T_0 to T_14; the forward stopped at each pole's own layer, at most layer 3), watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/reuse/lowlayer/1` | `i-0a5e57fd16e16feda` | g6e.4xlarge (us-east-1c; the first attempt found no capacity for g6e.2xlarge or g6e.4xlarge in any zone, the second with four L40S types landed the 4xlarge) | 11:40:06Z | 12:55:35Z | 1.258 h (4,529 s) | per load: checkpoint download 14.6 GB in 119 to 156 s (93 to 122 MB/s), load 62.9 s at the first load and 4.7 to 4.9 s at the others, then 122 to 129 s of captures (6,876 captures per load: 136 poles × 50 positions in two windows plus the identity's 2 × 20 and the known cases' 36; 5,454 harvest-condition windows of 2,048 tokens, 12.8 M tokens, about 105,000 tokens/s on the new poles); peak memory allocated 16.5 GB of 47.7 |

Cost at the on-demand price of about $3.0 per hour for a g6e.4xlarge (the price is quoted here, not banked): about $3.8, an estimate. The orphan reaper schedule (`pg-reap-i-0a5e57fd16e16feda`) deleted after the run; `pg audit` clean (0 running, 0 reapers). API $0 (the clusterer answered by Claude Code Sonnet subagents through the manual backend). The pipeline completed 16 of 16 steps with every step's return code 0; the bank (2,226 files, 9.5 MB, `lowlayer/<pole>.npz` per pole per load with `metrics.json` and `samples/` per load) pulled to `studies/reuse/results/lowlayer/1/`.

Science: `studies/reuse/SPEC12_lowlayer.md`, `studies/reuse/results/RESULTS.md` (the SPEC12 section), `studies/reuse/results/REVIEW_lowlayer.md`, journal entries 2026-11-03 05:40 CST (11:40Z), 06:15 CST (12:15Z) and the close-out entry.
