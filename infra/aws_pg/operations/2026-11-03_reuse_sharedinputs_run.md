# REUSE SPEC10 sharedinputs run on the AWS lane, 2026-11-03

One `pg run infra.aws_pg.jobs.pipeline` launch, `--type g6e.2xlarge,g6e.4xlarge --no-weights`, fresh prefix `results/reuse/sharedinputs/1`, one T_15 load (ablations), watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/reuse/sharedinputs/1` | `i-0a732cfa5a76f2ecb` | g6e.2xlarge | 06:11:35Z | 06:22:46Z | 0.189 h (671 s) | download and corpus 120 s, model load 71 s, units 292 s; 16,312 readouts, 3,722 captures, 4.9 M tokens; peak memory 20.7 GB |

Cost at the on-demand price of $2.24 per hour: about $0.42. The orphan reaper schedule deleted after the run; `pg audit` clean. Cap: the REUSE study's frozen hard cap is 5.0 GPU-h; it stood at 9.81 before this run and at 10.00 after; the overrun is the coordinator's decision under the standing instruction, recorded in the science journal (04:20 CST) and here.

Science: `studies/reuse/results/RESULTS.md` Results 53 to 55, `studies/reuse/results/REVIEW_sharedinputs.md`, journal entries 2026-11-03 03:55 to 04:45 CST.
