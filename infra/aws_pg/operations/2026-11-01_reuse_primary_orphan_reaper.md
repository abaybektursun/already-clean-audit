# 2026-11-01 — REUSE primary run: one reaper schedule left after self-termination

Scope: infrastructure only (CLAUDE.md Process 6). Science readings are in `studies/reuse/JOURNAL.md` and `studies/reuse/results/`.

## What happened
- `results/reuse/primary/1` (i-01aea00e9de38c08b, g6e.2xlarge, EC2 LaunchTime 2026-11-01T15:58:32Z) was launched with `pg run --detach`,
  budget 150 min, which armed the one-shot EventBridge reaper `pg-reap-i-01aea00e9de38c08b` at `at(2026-11-01T18:38:33)`.
- The pipeline finished (`STATUS exit=0` at 16:29:18Z, 9 of 9 steps ok) and the instance shut itself down (terminated 16:35:26Z per the
  executor's check). Self-termination does not delete the reaper, so `pg audit` after the run read "0 running, 1 reapers, 1 orphans,
  0 reaper-less". Same pattern as 2026-10-25 (`2026-10-25_neuronoff_orphan_reapers.md`) and as the two S1/S3 schedules of 2026-11-01
  noted in the REUSE journal (those two had fired at their deadlines and were gone by 09:30 CST).
- Cost: 1,846 s launch to STATUS, 0.513 GPU-h; REUSE study total 1.037 GPU-h.

## Cleanup
- 2026-11-01T16:44:55Z: instance state read back `terminated`.
- `./infra/aws_pg/pg kill i-01aea00e9de38c08b` (profile fuelos, region us-east-1): "terminated: i-01aea00e9de38c08b". The terminate call on a terminated instance
  changes nothing; the launcher deletes `pg-reap-<id>` after it.
- 2026-11-01T16:44:58Z: `./infra/aws_pg/pg audit` → "audit: 0 running, 0 reapers, 0 orphans, 0 reaper-less   clean. ".
