# 2026-10-25 — NEURONOFF cells: two reaper schedules left after self-termination

Scope: infrastructure only (CLAUDE.md Process 6). Science readings are in `studies/neuronoff/JOURNAL.md`.

## What happened
- Cells `results/neurons/off/q4b/1` (i-04cc88b41f0f87bc1, g6e.2xlarge, RunInstances 18:41:52Z) and
  `results/neurons/off/g4/1` (i-08ee3febe82659221, g6e.2xlarge, RunInstances 18:43:03Z) were launched with
  `pg run --detach`. Each launch armed a one-shot EventBridge reaper: `pg-reap-i-04cc88b41f0f87bc1` at
  `at(2026-10-26T00:37:52)` and `pg-reap-i-08ee3febe82659221` at `at(2026-10-26T02:12:03)`.
- Both jobs finished and the instances shut themselves down (`STATUS` written 19:53:13Z and 20:05:36Z; EC2
  state `terminated` checked at 20:12:07Z). Self-termination through `shutdown -h now` does not delete the reaper,
  so both schedules stayed ENABLED. `pg audit` at 20:08Z reported "0 running, 2 reapers, 2 orphans, 0 reaper-less".
- The preflight cell (`pre_q4b/1`, i-0064aa1ae3fa09108) had the same pattern; its reaper fired at its deadline
  (CloudTrail `TerminateInstances` at 19:31:54Z) on an already-terminated instance, and the one-shot schedule
  was gone by the 20:08Z audit.

## Cleanup
- 20:27:40Z: both instances confirmed `terminated`; both schedules read back ENABLED with the deadlines above.
- 20:27:48Z: `./infra/aws_pg/pg kill i-04cc88b41f0f87bc1 i-08ee3febe82659221` (profile fuelos, region us-east-1 from
  `infra/aws_pg/config.py`). `pg audit` has no cleanup subcommand; `pg kill` is the launcher's own path that deletes
  `pg-reap-<id>` after the terminate call. The terminate call on already-terminated instances changes nothing.
  Output: "terminated: i-04cc88b41f0f87bc1 i-08ee3febe82659221".
- 20:27:51Z: `./infra/aws_pg/pg audit` → "audit: 0 running, 0 reapers, 0 orphans, 0 reaper-less / clean."
  `./infra/aws_pg/pg ps` → "no pg-lineage instances running".

## Effect
- None on cost or results: an orphan reaper only calls TerminateInstances on an instance that no longer exists.
- Open: `run_job.sh`'s exit path does not delete its own reaper schedule (it only names it in comments); the
  schedule is created with `ActionAfterCompletion="DELETE"` (`pg.py:175`), so it removes itself after it fires. Every
  self-terminating job therefore leaves one orphan until its deadline. Not changed here.
