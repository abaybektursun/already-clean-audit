# REUSE SPEC8 birthobject run on the AWS lane, 2026-11-03

One `pg run infra.aws_pg.jobs.pipeline` launch, `--type g6e.2xlarge,g6e.4xlarge --no-weights`, fresh prefix `results/reuse/birthobject/1`, 16 loads with the T_15 load first, watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/reuse/birthobject/1` | `i-08207078f1275763a` | g6e.2xlarge | 03:00:53Z | 04:09:07Z | 1.135 h (4,087 s) | per load 214 to 244 s (download about 120 s, the dating stage 34 s, the scale stage with the ā pass 56 s); the T_15 load 634 s with the million-token pass; 2,645 case contexts and 4,000 scale contexts per load |

Times from the launcher's record: the reaper was set at 06:30:53Z for a 200-minute budget plus 10, so the launch was at 03:00:53Z; the watcher saw STATUS 4,087 s later, at about 04:09:00Z (its own clock printed 04:09:07Z). Cost at the on-demand price of $2.24 per hour: about $2.54. Both orphan reaper schedules (this instance's and SPEC6's) were deleted after the runs; `pg audit` clean. Cap: the REUSE study's frozen hard cap is 5.0 GPU-h; it stood at 8.50 before this run and at 9.64 after; the overrun is the coordinator's decision under the standing instruction, recorded in the science journal (22:25 CST) and here.

Science: `studies/reuse/results/RESULTS.md` Results 44 to 49, `studies/reuse/results/REVIEW_birthobject.md`, journal entries 2026-11-02 22:25 CST and 2026-11-03 00:15 to 00:40 CST.
