# 2026-10-26 — vast.ai path for NEURONSHOW: probe station (and the batch cell it replaced)

Scope: infrastructure and cost only (CLAUDE.md Process 6). Worktree `../interp_research-vast`, branch
`infra/vast-neuronshow` from `25fd077e`. Times are UTC on 2026-10-27 unless marked. AWS calls: `--profile fuelos
--region us-east-1`.

**Answer.** The existing SHACKLETON/MENDELEEV harness works unchanged for a 40–48 GB single-GPU probe station. I
added one cell (`cells/probe_station.py`), a laptop client (`probe.py`), an echo stand-in server (`probe_echo.py`)
and a manifest (`manifests/neuronshow_probe.json`). The image is `shackleton:mdl-v1`, already in ECR, so no rebuild.
Weights come from the Hub with `HF_TOKEN`. The station smoke on an RTX 6000 Ada worked: warm round trip
0.58–0.63 s, and it destroyed itself 30.0 min after its last request (idle exit 05:05:44Z, destroy trap 05:06:00Z, `vastai show instances` empty at 05:06:14Z). All rentals today cost $0.381 for three instances; the vast credit fell from $32.4396 to $32.0586, the same amount.

## 1. What I found

| item | finding | evidence |
|---|---|---|
| image | `shackleton:mdl-v1` in ECR, pushed 2026-09-03, 3.33 GB, digest `sha256:f2975abb…`. Deps only: torch 2.12.0+cu130, transformers 5.9.0, awscli, vastai; no weights. The neurons program ran Q4B, G4 and O7 on it on 2026-10-21..23. | `aws ecr describe-images --repository-name shackleton`; `mendeleev_spend.json` rows |
| rebuild | Not needed. If ever needed: `launch_builder.sh` on a self-terminating m7i.2xlarge; the first mdl-v1 build took about 10 min of builder time; a cached rebuild is shorter (2026-09-03 note §2.3, §2.7). | `operations/2026-09-03_mendeleev_infra.md` |
| plain `vastai/pytorch` base | More expensive for this path. `lifecycle.sh` needs `aws` before it starts (the onstart command fetches the code with `aws s3 cp`) and `vastai` for the destroy trap. A plain base would pip-install both plus transformers 5.9.0 during billed time on every rental, and the destroy guarantee would depend on that install succeeding. The mdl-v1 pull happens in unbilled `loading`. | `controller.py create_instance`, `lifecycle.sh finish()` |
| weights `download` | With `weights_hf_repo` set, `lifecycle.sh` downloads one repo with `HF_TOKEN`. With it empty (the convention used here), the job or server downloads itself under `HF_ONLINE=1`. Needs `HF_TOKEN` only; the offer filter adds `inet_down>=800`. | `lifecycle.sh`, `controller.search_offers` |
| weights `s3` | Syncs named models from `s3://…/weights/hf/` and needs S3 read on `weights/hf/*` on the box. The cache holds Qwen3-4B-Base and gemma-3-4b-pt but **not** Olmo-3-1025-7B. Not used. | `aws s3 ls …/weights/hf/hub/` |
| AWS keys on the box | Required by this harness: the onstart command fetches code from S3, a streamer uploads every 60 s, and the net probe, banking and cell syncs all use S3. The earlier program's answer is reused: the scoped IAM user `mendeleev-box` (`program.json` `box_profile`; policy `iam/mendeleev_box_policy.json`) can put/get `results/mdl/*` and read `weights/hf/*`, nothing else. Checked today: that profile writes `results/mdl/neurons/show/_localcheck/` and gets AccessDenied on `results/neurons/show/`. | §3 |
| HF token | `controller.creds_env` reads SSM `/pg-lineage/hf-token` at rent time and ships it as `HF_TOKEN`; the box has no SSM access. Checked with that token: `google/gemma-3-4b-pt` config 200 (401 without a token); Qwen3-4B-Base and Olmo `stage1-step1413814` resolve (307). On the box the echo server reports `hf_token_set: true`. The token's role is **write** (`whoami-v2`), and the host can read container env. | §4 |
| L40S | **No L40S passes the README base filter** at 04:30Z. All L40S offers either charge for bandwidth, run a CUDA 12.8/12.9 driver (mdl-v1 needs 13.0+), are under 800 Mb/s, or sit in Japan, Taiwan or South Africa. The closest card is the RTX 6000 Ada (same AD102 chip and SM count as the L40S). It is also excluded only by its bandwidth charge ($0.0026–0.01/GB), so it is rented by explicit offer id; `controller.py rent` does not re-apply the filter. | `vastai search offers "… gpu_name in [L40S,RTX_6000Ada,L40] cuda_vers>=13.0 …"` |
| `gpu_ram` units | Vast reads `gpu_ram` in GB of 1000 MB. `gpu_ram<=48` drops the 48 GB cards (48935–49140 MB), and `gpu_ram>=24` keeps the 4090 (24564 MB). | plan outputs |
| net probe | `lifecycle.sh` times one 50 MB single-stream GET from S3 us-east-1, including aws CLI start-up, and aborts below 10 MB/s. 6 of the 18 finished instances in `mendeleev_spend.json` ended `slow_net` (Ireland, Ohio, Australia among those still listed). Today the Ireland 4090 advertising 13 Gb/s measured 6.0 MB/s. It measures the S3 path, not the Hub path that download-mode jobs use. Each abort costs cents; the real cost is wall clock. | `attempts/1/job.log`; spend-file rows |
| invoices | `vastai show invoices` (deprecated, what `controller.py ledger` reads) returned 2 items for this account today. `vastai show invoices-v1 --charges` returns per-instance GPU, disk and bandwidth lines. | §5 |
| budget | `budget.json` ceiling $36.0. The spend file totalled $30.27 before today, so the watchdog prints BUDGET-SOFT at start and stops everything at $36. Vast credit read at start: $32.44. Raising the ceiling is the owner's call. | `vastai show user` |

## 2. What I changed (branch `infra/vast-neuronshow`)

- `9acbf7fe` `cells/module_job.py`: batch cell, `python -m <job> <args> --out <out_dir>`, then `aws s3 sync out_dir
  s3_out` (also on a nonzero exit, then raise). Built for the first target; still valid for batch jobs (§7).
- `be2a3077` `manifests/neuronshow_smoke.json`: the batch smoke (24–50 GB, smoke_stats + module_job).
- `8ce2c0a6` the probe station: `cells/probe_station.py`, `probe.py`, `probe_echo.py`, `manifests/neuronshow_probe.json`.
- This note and the new rows in `studies/lineage/cross-lineage/research/mendeleev_spend.json`.
- No change to `controller.py`, `lifecycle.sh`, `run_cells.py`, `policy.py`, `watchdog.py`, the IAM policy or
  `budget.json`.

## 3. Offline gates (laptop, weight-free)

- `TIER=1 bash preflight.sh` → `OK: 26/26 scenarios + invariants`, `T1 PASS`, stamps `be2a3077` and then `8ce2c0a6`.
  T2 builds a docker image (a container build, which CLAUDE.md bans on the laptop) and T3 needs a GPU. Both are
  covered by the rental smokes below.
- module_job with the real `neuron_rw_dryrun --arch gemma3 --stage arm0` and the `mendeleev-box` keys: exit 0 and 20
  files under `results/mdl/neurons/show/_localcheck/ok/`. Failing inputs: an unknown module gives the sync, then
  `AssertionError: no_such_module_xyz exited 1`; an `s3_out` under `results/neurons/` gives AccessDenied, so the
  sync raises.
- probe_station with probe_echo under CPU torch (9 s idle): round trip 0.07 s; a server that exits writes
  `server_exit.json` and `probe.py box` exits 3 with the log tail; a redeploy answers again, including the request
  left pending; requests 5 s apart keep the station alive for 15 s; the station returns 9 s after the last request
  and the server process is gone.
- `bash deploy_reaper.sh invoke` → `checked 0 harness instances, reaped 0`; schedule `shackleton-reaper-5min`
  ENABLED; its prefix table maps `mendeleev_` → `results/mdl/`.

## 4. Rentals (watchdog `python3 -u watchdog.py` running during each rental)

| rental | instance | GPU, host | $/h listed | created → running | billed (invoice) | $ | outcome |
|---|---|---|---|---|---|---|---|
| neuronshow_smoke #1 | 57915420 | RTX 4090, m15489 Ireland | 0.546 | 04:13:15 → 04:17:49 | GPU 0, disk 0.149 h | 0.002 | net probe 6.0 MB/s → exit 9, self-destroyed in 12 s |
| neuronshow_smoke #2 | 57916157 | RTX PRO 5000 48 GB, m152030 Nebraska (clean twice before) | 0.750 | 04:20:03 → 04:20:43 (image cached) | GPU 0.034 h at $0.733, disk 0.034 h | 0.026 | probe 36.6 MB/s; smoke_stats Qwen3-4B-Base from the Hub (3 files in 30 s, load 35.8 s, argmax " Paris", torch 2.12.0+cu130); module_job dry run, all three stages, gates true; 111.7 s of cells; self-destroyed exit 0 |
| neuronshow_probe | 57917200 | RTX 6000 Ada, m117959 California | 0.646 | 04:30:52 → 04:34:33 | GPU 0.541 h at $0.587, disk 0.593 h at $0.059, 0.1 GB at $0.015/GB | 0.353 | probe 25.0 MB/s; station up 31.1 min; 7 requests, 7 responses; idle exit; 17 files banked; self-destroyed exit 0 |

After attempt 1 the watchdog would have respawned on the cheapest offer, a Taiwan 3090 likely to fail the same
probe. I stopped it before its next poll and stamped the row as it would have (`slow_net`, `killed_infra`). I moved
the attempt's markers to `attempts/1/` (`watchdog.archive_attempt`) and rented a host with two clean rentals behind
it.

Station smoke (rental `neuronshow_probe`):
- Cold `probe.py send`: vast lookup 2.02 s plus round trip 2.42 s. The echo response names `NVIDIA RTX 6000 Ada
  Generation` with `hf_token_set: true`.
- Warm, with `PROBE_SSH` exported (SSH control connection reused): 0.61, 0.63, 0.62, 0.58, 0.60 s.
- `probe.py deploy --ref HEAD --server "python3 -m infra.vast.probe_echo"`: 7.5 MB shipped in 26 s, then the server
  restarted with `PG_GIT_SHA=8ce2c0a6`. The first request after the restart took 2.37 s (torch import).
- Last request 04:35:43Z. The idle timer fired at 05:05:44Z (`idle 30.0 min > idle_min 30`). The station banked 17 files: 7 requests, 7 responses, `server.log`, `server_cmd`, `server_sha` and `_done.json`. The lifecycle trap destroyed the instance at 05:06:00Z, and `vastai show instances` was empty at 05:06:14Z. `bash deploy_reaper.sh invoke` afterwards gave `checked 0 harness instances, reaped 0`; the only schedule left is the standing `shackleton-reaper-5min`. The watchdog closed the row (`instance gone (complete marker present)`) and was stopped.

Batch smoke output was copied to `results/neurons/show/0/dryrun_gemma3/` (38 files). Station files are banked
under `results/mdl/mendeleev_neuronshow_probe/` and were then moved to `results/mdl/neuronshow_probe_sessions/20261027T0430/` (42 objects; §6 step 6), so the rental prefix is empty for the next session.

## 5. Cost

| instance | GPU | disk | bandwidth | total |
|---|---|---|---|---|
| 57915420 (probe abort) | $0.000 | $0.002 | $0 | $0.002 |
| 57916157 (batch smoke) | $0.025 (2.0 min) | $0.001 | $0 | $0.026 |
| 57917200 (station, 31 min idle-out) | $0.317 (32.5 min) | $0.035 | $0.001 | $0.353 |
| **total** | | | | **$0.381** |

Source: `vastai show invoices-v1 --charges -s 2026-10-26 -e 2026-10-28`. The credit drop over the session matches ($32.4396 → $32.0586). The three rows are recorded in `mendeleev_spend.json` with `billing_source` naming that command; I did not run `controller.py ledger`, because it would overwrite `mendeleev_spend_invoices_raw.json` with the 2 items the deprecated endpoint returns. A station on this host bills about $0.65/h while up (GPU $0.587 + disk $0.059). It also bills bandwidth at the invoiced $0.015/GB, not the $0.01 listed: about $0.47 for the real server's 31 GB of weights per start. The spend file now totals $30.65 against the $36 ceiling, which leaves about 8 h of station time.

## 6. Runbook (from `infra/vast/` of a checkout that has this branch merged)

```bash
cd infra/vast
# 1. ship gate: commit, then stamp HEAD (rent refuses an unstamped infra/ tree)
TIER=1 bash preflight.sh
# 2. detection layer, in its own terminal or a session Monitor (-u is required)
python3 -u watchdog.py
# 3. pick an offer. plan lists offers that pass the base filter; L40S / RTX 6000 Ada currently fail its bandwidth
#    term, so list them separately and rent by id
python3 controller.py plan manifests/neuronshow_probe.json
vastai search offers "verified=true rentable=true reliability>0.98 num_gpus=1 gpu_name in [L40S,RTX_6000Ada,L40] cuda_vers>=13.0 inet_down>=800 cpu_ram>=30 disk_space>=80" --storage 80 -o dph
# 4. rent (the instance starts the echo server; HF_TOKEN and the scoped S3 keys are shipped automatically)
python3 controller.py rent manifests/neuronshow_probe.json <OFFER_ID>
#    ready when the station cell has started:
aws --profile fuelos --region us-east-1 s3 cp s3://pg-lineage-769063704696/results/mdl/mendeleev_neuronshow_probe/live_out/progress.json -
# 5. probe: one command per request; response JSON on stdout, timings on stderr
export PROBE_SSH=$(python3 probe.py url)          # optional: skips 2 s of vast lookups per call
python3 probe.py send request.json                # exit 3 = server process exited (log tail shown), 4 = timeout
python3 probe.py log -n 50                        # server stdout/stderr
# start the real server once infra/pg/probe_server.py is committed (any branch of this repo; worktrees share
# objects): ships `git archive <ref> infra configs scripts` into /app and restarts the server
python3 probe.py deploy --ref <branch-or-commit> --server "python3 -m infra.pg.probe_server"
# 6. end: idle 30 min after the last request ends it by itself; to end now (banks everything first):
python3 probe.py stop
#    hard kill (no banking; the lifecycle trap is skipped):
python3 controller.py kill <INSTANCE_ID>
#    confirm gone, then read the charges
vastai show instances
vastai show invoices-v1 --charges -s <YYYY-MM-DD> --latest-first
#    pull the session (every request, response and server.log; streamed to live_out/station/ every 60 s while up)
aws --profile fuelos --region us-east-1 s3 sync s3://pg-lineage-769063704696/results/mdl/mendeleev_neuronshow_probe/cells/station/ ./probe_session/
#    REQUIRED before the next rental of the same manifest: move the session aside. Otherwise the new instance finds
#    cells/station/_done.json, skips the station and destroys itself (cents, no station)
aws --profile fuelos --region us-east-1 s3 mv --recursive s3://pg-lineage-769063704696/results/mdl/mendeleev_neuronshow_probe/ s3://pg-lineage-769063704696/results/mdl/neuronshow_probe_sessions/<YYYYMMDDTHHMM>/
```

Alternative start for the real server: set the cell's `"server"` in `neuronshow_probe.json` to its command on a
branch that contains `infra/pg/probe_server.py`, commit, run preflight, and rent. The code then ships at rent time
and no deploy is needed.

**Protocol for `infra/pg/probe_server.py`** (the station supervises it; details in `cells/probe_station.py`):
- Read `$PROBE_DIR/requests/*.json` (`PROBE_DIR=/workspace/probe`). For each request without
  `$PROBE_DIR/responses/<same name>`, write the response to a temporary name that does not end in `.json`, then
  rename it. Never delete requests.
- stdout and stderr go to `server.log`. If the process exits, the station records `server_exit.json` and stays up
  until idle; `probe.py send` then exits 3.
- Environment: `HF_TOKEN`, `HF_HOME=/workspace/hf`, `HF_HUB_CACHE=/workspace/hf/hub`, `HF_HUB_DISABLE_XET=1`,
  `HF_HUB_OFFLINE` unset, `PYTHONPATH=/app`, and `PG_GIT_SHA` after a deploy.
- Load the three models once at start, about 31 GB of downloads. The disk is 80 GB.
- Idle counts from the newest request, so one request computing longer than 30 min would be cut off.

## 7. Batch path (first target, kept)

For a one-shot job: the manifest cell is `{"id": "q4b", "module": "infra.vast.cells.module_job", "job":
"infra.aws_pg.jobs.neuronshow", "args": "--model-key Q4B --corpus s3://pg-lineage-769063704696/results/mdl/neurons/show/inputs/q4b_corpus",
"s3_out": "s3://pg-lineage-769063704696/results/mdl/neurons/show/<n>/q4b/out/", "pg_git_sha": "<commit with the job>"}`.
The cell appends `--out /workspace/out/q4b`. The Q4B corpus (`results/neurons/w1/q4b/1/out/corpus/corpus`) must
first be copied server-side under `results/mdl/` (`aws s3 sync … …/results/mdl/neurons/show/inputs/q4b_corpus/`),
because the box user reads only `results/mdl/*`. Afterwards, copy the outputs to `results/neurons/show/<n>/q4b/out/`
with `aws s3 sync` under the fuelos profile.

## 8. Follow-ups (not changed here)

1. `controller.py ledger` reads the deprecated `vastai show invoices`, which returned 2 items today. Port it to
   `invoices-v1 --charges` so `realized_usd` stops reading null.
2. The net probe gates on single-stream S3 throughput even when weights come from the Hub. It aborted 6 of 18
   recorded instances. Options: a per-manifest threshold, or a parallel `aws s3 cp`.
3. The HF token shipped to third-party hosts has the write role. A read-only token in its own SSM parameter would
   narrow the exposure. `VAST_API_KEY` (full account) is also in the container env; this is existing design.
4. `budget.json` ($36) leaves about $5 of headroom over the spend file. The owner decides whether the new credit
   raises it.
5. `probe.py` turns off host-key checking (vast host keys change per instance), and its SSH control connection
   persists 10 min after the last call.
