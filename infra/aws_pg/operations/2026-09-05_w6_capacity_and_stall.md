# 2026-09-05 — w6 launch: single-GPU capacity dry in us-east-1; one g5.4xlarge cell stalled

Scope: infrastructure only (CLAUDE.md Process 6). Science readings are in `studies/stagediff/JOURNAL.md`.

## Capacity
- 14:34Z first `pg run` for the w6 queue: `InsufficientInstanceCapacity` for g6e.2xlarge and g6e.4xlarge in every AZ (the
  config's two types). 14:38Z relaunched with `--type g6e.2xlarge,g6e.4xlarge,g5.2xlarge`; 14:41Z g5.2xlarge dry too.
  14:42Z relaunched with eight single-GPU types (g6e.2xlarge, g6e.4xlarge, g6e.xlarge, g5.2xlarge, g5.4xlarge, g5.xlarge,
  g6.2xlarge, g6.4xlarge). First placement 14:43Z (g6e.2xlarge); the next at 14:49Z landed on g5.4xlarge; from 15:11Z
  g6e capacity returned (g6e.4xlarge, then g6e.2xlarge ×2). Each failed round costs about 2–3 min (8 types × AZs).
- The launcher (`stagediff_launch.sh w6`) retries every 60 s for 120 rounds; the 32-vCPU quota is the second limiter once two
  4xlarge instances are up (16 + 16), so the E1 cells waited on the rot-floor cell to finish.

## Stall
- `i2m-H5p/1` on g5.4xlarge (i-0ffa30d31fd65215b, launched 14:49Z): job started 14:52:52Z, model loaded 14:53:50Z, then the
  run_job heartbeat and `out/` sync (30 s cadence) stopped at 14:54:04Z. The last job.log lines show the input fetch from S3
  running at 1.5–6 MiB/s. At 15:25Z (31 min without a heartbeat; the sibling `i2m-H5/1` on g6e.2xlarge finished its whole
  run in 16 min) the instance was terminated with `pg kill <id> --force` and the cell requeued as `i2m-H5p/2` with the same
  arguments under the eight-type fallback list.
- Not diagnosed: whether the g5.4xlarge lost network (the heartbeat's own `aws s3 cp` would then fail silently) or the job
  hung. The wrapper's `>/dev/null 2>&1` on the sync loop hides which. Follow-up for the launcher, not this study: a
  heartbeat-age watchdog on the control plane (kill + requeue after N minutes without a heartbeat).

## Cost note
- Budgets: i2m 100 min, rot-floor 35, rank 35 (45 for C1/C2) — sized for the slowest fallback GPU (L4); actual use on g6e
  is a third of that. The stalled cell burned ~0.6 h of g5.4xlarge.

## Second stall (same signature, different instance type)
- `i2mb-H5p/1` on g6e.2xlarge (i-0fbc8ea60f76732f0, launched 16:03Z): job start 16:07Z, model loaded by 16:08:09Z, heartbeat
  and `out/` sync stopped at 16:08:19Z; at 16:14Z (6 min silent, sibling `i2mb-H5/1` on g5.4xlarge heartbeating normally)
  killed with `pg kill <id> --force` and requeued as `i2mb-H5p/2`. Both stalls hit a C6 (MolmoAct2-DROID) cell right after
  the model load, during the input fetch; `i2m-H5p/2` (C6, g6e.4xlarge) ran clean, so it is not deterministic. Two of five E3
  cells so far. Watchdog rule applied by hand: no heartbeat for > 5 min after a 30 s cadence → kill and requeue.
- The E3b family with a 3-PC control span could not match δ̄ ⊥ μ_C's energy on most band layers at H5 (the job stamped
  itself VOID and kept running its arms); both E3b cells were requeued as `/2` with `--n-pcs 8` at 16:15Z.
- `i2mb-H5p/2` on g5.xlarge (i-05558063f8a2abb77, launched 16:19Z): after 22 min the S3 prefix held nothing at all — no
  job.log, no heartbeat — while its twin `i2mb-H5/2` on the same type heartbeated normally; killed 16:42Z and requeued as
  `i2mb-H5p/3` restricted to g6e types (the two C6 cells that ran clean were both on g6e). Three C6 launches out of five
  have now failed to produce a job; the two failures with heartbeats stopped right after the model load, this one never
  reported. Suspect the C6 weight sync / load path on some hosts; not diagnosed from the control plane.
- 2026-09-06 01:19–01:46Z: the two E3c cells (`i2mc-H5p/1`, `i2mc-H5/1`) placed on g6e.2xlarge on the first attempt and ran clean
  (heartbeats continuous; 25 min each). The C6 stall of the afternoon did not recur; still undiagnosed.
