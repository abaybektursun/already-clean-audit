# 2026-11-05 — M1 mix harvests (`results/next_m1/mix-a-01`, `results/next_m1/mix-b-01`)

Study: `studies/next/07_PICKS/M1_SPEC.md` §9.5 S4, §9.7. Worktree `interp_research-m1`, branch `m1`, launched from
commit `e76b318aa502` (tag `m1-freeze-2`), tracked tree clean (`git status --porcelain --untracked-files=no` empty;
no untracked file under `infra/`), so the code tarball `pg` ships equals that commit. Lane: AWS g6e.2xlarge (1x
L40S) only, by the head's decision; no fallback type. Every AWS call: `--profile fuelos --region us-east-1`.

## Pre-launch (2026-11-05T16:48:54Z)

- `./infra/aws_pg/pg ps`: no pg-lineage instance running. `pg audit`: 0 running, 3 orphan reapers (for the
  terminated `i-02f773eedd54233d5`, `i-03e36825423328376`, `i-047efd8fa80c4254c`), 0 reaper-less.
- `results/next_m1/` held only `inputs/` (the five files of the K0 record) and `k0-01/`.
- Step files uploaded 16:49:06Z and 16:49:08Z to `results/next_m1/{mix-a-01,mix-b-01}/steps.json`; each
  downloaded back and byte-equal (`cmp`) to `infra/aws_pg/steps/next_m1_{mix-a-01,mix-b-01}.json`.

## Launch attempts

| UTC | attempt | command result |
|---|---|---|
| 16:49:14 to 16:50:30 | `mix-a-01` | `RuntimeError: none of ['g6e.2xlarge'] launchable in any AZ`, last error `InsufficientInstanceCapacity`, over the five subnets and the EC2-chosen AZ. No instance was created; `results/next_m1/mix-a-01/` holds only `steps.json`. |
| 16:51:30 to 16:52:39 | `mix-a-01`, retry 1 of a bounded loop (`scratchpad/m1_run/launch_retry.sh mix-a-01 390 24 240`: the same `pg run` command, g6e.2xlarge only, up to 24 tries 240 s apart; log `scratchpad/m1_run/launch_mix-a-01.log`) | `InsufficientInstanceCapacity` in every AZ; no instance |
| 16:56:39 to 16:57:41 | `mix-a-01`, retry 2 of the same loop | **launched** `i-01ca98670df35bb1c`, g6e.2xlarge, us-east-1b, AMI `ami-0cee78ee4b1c24614`, EC2 `LaunchTime` 2026-11-05T16:57:40Z, reaper `at(2026-11-05T23:37:40)` (390 min + 10). The loop exited 0. |

### The head's fallback rule (relayed by the coordinator, received about 16:56Z)

Retry g6e.2xlarge in every zone for up to 20 more minutes. If there is still no capacity, launch the first harvest on
g6e.4xlarge (the same single L40S, so the one-GPU-model rule holds; $3.00 per hour instead of $2.24; the 12 GPU-h
ceiling is unchanged) and record it as a dated deviation. vCPU quota 32: one more g6e.4xlarge fits while N13 holds a
g6e.4xlarge; keep retrying g6e.2xlarge for the second harvest meanwhile, and allow a second g6e.4xlarge once N13's
instance is gone (checked with describe-instances). Never another GPU model.

- Deadline taken as 2026-11-05T17:16:16Z (20 minutes after the 16:56:16Z read below).
- `describe-instances` at 16:56:16Z (states pending, running, shutting-down, stopping): **no instance at all in the
  account and region**, so no N13 instance at that time.
- `mix-a-01` launched on g6e.2xlarge before the deadline, so the g6e.4xlarge fallback was not used for it.
- `mix-b-01` launcher: `scratchpad/m1_run/launch_rule.sh 2026-11-05T17:16:16Z 2026-11-05T18:30:00Z 60 g6e.2xlarge`
  (log `scratchpad/m1_run/launch_mix-b-01.log`), started 16:58Z. Until the deadline: g6e.2xlarge only, every 60 s
  after each try. After it: `g6e.2xlarge,g6e.4xlarge` (g6e.2xlarge first, to keep the identity pair on A's instance
  type), but only when `describe-instances` shows no pending or running instance tagged `results/next_n13/*`;
  otherwise g6e.2xlarge only. It stops at 18:30Z if B has not launched.

## mix-a-01 progress (from artifacts)

| UTC | event (artifact) |
|---|---|
| 16:59:01 | `out/pipeline_status.json` present, step `start_a` running (boot under 81 s) |
| 16:59:23 | `start_a` phase `download` (`out/start_a/m1/metrics.json`) |
| 17:09:06 | download done: 14,605,764,269 B in 576.7 s = **25.3 MB/s** (`out/start_a/m1/download.json`); G-SHARD pass, its failing input fails |
| 17:09:51 | phase `harvest`: load 41.2 s; G-CFG pass (`diff_keys` [], `sliding_window` 4096); G-CAST pass, its failing input fails; versions torch 2.12.0+cu130, transformers 5.9.0 |
| 17:12:06 | `pg liveness`: COMPUTING (cpu_seconds +32 per 32 s heartbeat) |

The download rate is a third of K0's 76.0 MB/s and a ninth of T_15's 234 MB/s. The §9.7 projection at 3,337.75 tokens/s
and 25.3 MB/s (`scratchpad/k0_checks/project.py 3337.75 25.3 0.27`): bf16 harvest 1.996 h, fp32 harvest 2.173 h;
mix-a 6.39 h, mix-b 4.22 h; L1 + L2 = 10.87 GPU-h, below the stop rule's 11.7, so nothing is parked. **Risk:** at
this rate `mix-a-01` ends at about 23:21Z, against its 390-min kill at about 23:27Z (job start plus 390 min) and the
reaper at 23:37:40Z. A slower fp32 download would kill R near its end. R alone relaunched would then cross the 12 GPU-h
ceiling (about 0.27 + 6.5 + 4.22 + 2.2 = 13.2), so a timeout of R amounts to parking R (H_order NOT_RUN).

## mix-b-01 launch

`scratchpad/m1_run/launch_rule.sh` made 8 tries from 16:58:08Z; the first 7 hit `InsufficientInstanceCapacity` in
every AZ. Try 8, 17:13:10Z to 17:14:19Z, **launched** `i-0bcf8476da3757298`, g6e.2xlarge, us-east-1c, AMI
`ami-0cee78ee4b1c24614`, EC2 `LaunchTime` 2026-11-05T17:14:18Z, reaper `at(2026-11-05T21:54:19)` (270 min + 10).
That was before the 17:16:16Z deadline, so the g6e.4xlarge fallback was never used. **No deviation:** both
instances are g6e.2xlarge (1x L40S), the type of K0 and of the registered lane, and the identity pair (start_a on
A, start_b on B) shares one instance type as `M1_SPEC.md` §3.1 requires. The tracked tree was clean at both
launches (an M1 journal edit was held back until after B's launch for that reason). The untracked file under
`infra/` at both launches was this note; it is not code.

`pg audit` at about 17:15Z: 2 running, 3 reapers, 1 orphan reaper (`pg-reap-i-0b9bdaf961cc7a0ac`, not an M1
instance), 0 reaper-less. vCPU in use: 16 of 32.

## Liveness at 17:18Z (artifacts)

- `mix-a-01`: `out/start_a/metrics.json` batch 50 of 1,256 at 3,271.0 tokens/s, `G0_pass` true, `updated_utc`
  17:14:16Z.
- `mix-b-01`: `out/pipeline_status.json` step `start_b` running; `out/start_b/m1/metrics.json` phase `download` since
  17:16:32Z.
- Watch loops (laptop, bounded, never read `job.log`): `scratchpad/m1_run/watch.sh <attempt> <instance> <launch UTC>
  <polls>` every 300 s, logs `scratchpad/m1_run/watch_{mix-a-01,mix-b-01}.log`. Each exits when a step changes state,
  on STATUS, on the instance disappearing, or when the newest `metrics.json` is older than 1,800 s.

## start_a done (18:58:21Z)

Harvest loop 6,155.0 s at 3,340.4 tokens/s; snapshot write 160.7 s; final write 154.0 s (`out/start_a/manifest.json`
→ `timing_s`). Uptime at the end of start_a: 2.01 h (projection for a bf16 harvest at 25.3 MB/s plus boot: 2.04 h).
M began downloading at 18:58:54Z. `mix-b-01`'s start_b was at batch 1,100 of 1,256 at 18:56:28Z.

## start_b done (about 19:11Z)

Download 14,605,764,269 B in 230.1 s = 63.5 MB/s; harvest loop 6,127.1 s at 3,355.6 tokens/s; snapshot write 160.5 s;
final write 151.5 s (`out/start_b/manifest.json`, `m1/download.json`). G began downloading after it.

**Watch-loop bug (laptop tool, not the job).** At 19:06:38Z `watch.sh` for `mix-b-01` exited with "NO metrics.json
1800 s after launch" although `out/start_b/metrics.json` had been written at 19:06:34Z and `pg liveness` read
COMPUTING. One S3 listing came back empty, and the rule treated that as "never written". Fixed in
`scratchpad/m1_run/watch2.sh`: the rule fires only if no `metrics.json` has ever been seen; an empty listing after one
was seen is logged as a failed read. Both loops were restarted on `watch2.sh` at 19:07Z. No instance was touched.

## M done (20:56:09Z), G done (21:11:28Z), mix-b-01 ended

- M (`mix-a-01`): download 29,201,786,046 B in 511.8 s = 57.1 MB/s; sha256 18.4 s; load 5.6 s; harvest loop 6,155.1 s.
  R began downloading at 20:56:18Z.
- G (`mix-b-01`): download 29,201,786,097 B in 794.1 s = 36.8 MB/s; harvest loop 6,132.0 s. STATUS `exit=0`;
  `pipeline_status.json` complete, 2 of 2. Instance `i-0bcf8476da3757298` `shutting-down` at 21:14:57Z. Uptime
  17:14:18Z to about 21:14Z, about 4.0 h (projection at 25.3 MB/s: 4.22 h).

## R done (22:53:30Z), mix-a-01 ended

- R: download 29,201,786,046 B in 508.1 s = 57.5 MB/s; harvest loop 6,157.1 s at 3,339.3 tokens/s.
- STATUS `exit=0`; 3 of 3 steps. Instance `i-01ca98670df35bb1c` `shutting-down` at 22:57:45Z. Uptime 16:57:40Z to
  about 22:57Z, about 6.0 h (projection at 25.3 MB/s: 6.39 h; kill limit 6.5 h). `pg ack` on both prefixes.
- GPU-h for M1 in all: L1 about 0.27 + `mix-a-01` about 6.0 + `mix-b-01` about 4.0 = **about 10.3 GPU-h**, against the
  12 GPU-h ceiling. At $2.24 per hour: about $23.
