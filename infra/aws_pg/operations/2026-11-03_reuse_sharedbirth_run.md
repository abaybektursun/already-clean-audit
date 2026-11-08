# REUSE SPEC11 sharedbirth run on the AWS lane, 2026-11-03

One `pg run infra.aws_pg.jobs.pipeline` launch, `--type g6e.2xlarge,g6e.4xlarge --no-weights`, fresh prefix `results/reuse/sharedbirth/1`, seven loads (T_7 to T_13, ablations at each case's birth checkpoint), watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/reuse/sharedbirth/1` | `i-0c2b6abec971c570b` | g6e.4xlarge (the 2xlarge had no capacity at launch) | 07:10:17Z | 07:34:20Z | 0.401 h (1,443 s) | job wall 0.371 h over the seven loads (0.040 to 0.080 h each, downloads included); 12,538 forwards, 4,664 captures, 4.4 M tokens; peak memory 20.7 GB |

Cost at the on-demand price of about $3.0 per hour for a g6e.4xlarge: about $1.20. The orphan reaper schedule (`pg-reap-i-0c2b6abec971c570b`) deleted after the run; `pg audit` clean. Cap: the REUSE study's frozen hard cap is 5.0 GPU-h; it stood at 10.00 before this run and at 10.40 after; the overrun is the coordinator's decision under the standing instruction, recorded in the science journal (06:15 CST entry, written before the launch) and here. The journal's hand-written CST stamps up to the launch run about five hours ahead of the clock (the launch was 01:10 CST); the UTC times here are the launcher's.

Science: `studies/reuse/results/RESULTS.md` Results 56 to 58, `studies/reuse/results/REVIEW_sharedbirth.md`, journal entries 2026-11-03 06:15 CST and 01:45 CST (07:45Z).
