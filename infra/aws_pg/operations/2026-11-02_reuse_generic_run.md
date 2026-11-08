# REUSE SPEC6 generic run on the AWS lane, 2026-11-02

One `pg run infra.aws_pg.jobs.pipeline` launch, `--type g6e.2xlarge,g6e.4xlarge --no-weights`, fresh prefix `results/reuse/generic/1`, watched with `pg watch` under a background shell; nothing scored before `STATUS`.

| prefix | instance | type | launched (UTC) | STATUS | launch to STATUS | notes |
|---|---|---|---|---|---|---|
| `results/reuse/generic/1` | `i-0b92279f06409538e` | g6e.4xlarge (g6e.2xlarge had no capacity) | 02:45:40Z (11-03) | 03:04:09Z, exit 0 | 0.305 h (1,099 s) | one T_15 load: download and corpus 136 s, model load 88 s, gates 24 s, units 652 s; 37,331 forwards, 11.0 M tokens; peak memory 20.9 GB |

Cost at the on-demand price of about $3.0 per hour for a g6e.4xlarge: about $0.92. Reaper schedule cleared with the instance's self-termination (`pg audit` after the run: no orphan). Cap: the REUSE study's frozen hard cap is 5.0 GPU-h; it stood at 8.195 before this run and at 8.50 after; the overrun is the coordinator's decision under the standing instruction to run the queue autonomously, recorded in the science journal (22:05 CST) and here. SPEC8 (`results/reuse/birthobject/1`, `i-08207078f1275763a`, g6e.2xlarge, budget 200 min) launched at 03:00:53Z and is recorded in `2026-11-03_reuse_birthobject_run.md`.

Science: `studies/reuse/results/RESULTS.md` Results 38 to 43, `studies/reuse/results/REVIEW_generic.md`, journal entries 2026-11-02 22:05 to 23:20 CST.
