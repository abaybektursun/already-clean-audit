# REUSE SPEC9 sameweights run on the AWS lane, 2026-11-03

One `pg run infra.aws_pg.jobs.pipeline` launch, `--type g6e.2xlarge,g6e.4xlarge --no-weights`, fresh prefix `results/reuse/sameweights/1`, one T_15 load of captures (no ablation), watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/reuse/sameweights/1` | `i-0ea30f53f6fa5905e` | g6e.2xlarge | 05:10:37Z | 05:21:01Z | 0.170 h (611 s) | download and corpus 126 s, model load 72 s, units 215 s; 14,458 captures, 3.5 M tokens; peak memory 20.6 GB |

Cost at the on-demand price of $2.24 per hour: about $0.38. The orphan reaper schedule deleted after the run; `pg audit` clean. Cap: the REUSE study's frozen hard cap is 5.0 GPU-h; it stood at 9.64 before this run and at 9.81 after; the overrun is the coordinator's decision under the standing instruction, recorded in the science journal (02:45 CST) and here.

Science: `studies/reuse/results/RESULTS.md` Results 50 to 52, `studies/reuse/results/REVIEW_sameweights.md`, journal entries 2026-11-03 02:00 to 03:05 CST.
