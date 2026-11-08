# SHACKLETON vast.ai infra — zero GPU-billed waste

## The three billing facts this design is built on (docs.vast.ai, checked 2026-08-16)

1. **GPU $/hr is billed only in the `running` state.** The `loading` state (docker image
   pull + extraction) is NOT billed as GPU time. Storage is billed from creation
   ($/GB/month, cents), bandwidth per GB at the host's rate (most hosts: $0).
2. **Stopped instances keep billing storage.** We DESTROY, never stop.
3. **Interruptible instances**: paused when outbid; disk kept; the onstart script re-runs
   automatically on restart — which is exactly our resume path.

## Consequence: bake everything into the image

Model weights, pinned deps, repo code, and corpora are ALL baked into per-rental-group
docker images pushed to ECR (us-east-1, same account as the pg bucket). The host pulls
the image during unbilled `loading`; when billing starts the instance goes straight to
measurement. Billed non-compute time = seconds of asserts + small artifact uploads.

- Base image: the current `vastai/pytorch` template tag (host-cached → fast load),
  pinned by digest at build time into the image manifest.
- Weights ride as one docker layer per shard file (parallel pull, under registry layer
  caps). Layer order: pip deps → weights (stable) → repo code (volatile, tiny) — a code
  fix rebuilds/pushes only the last small layer.
- Images are built on an ephemeral self-terminating EC2 builder in us-east-1
  (HF → EC2 → ECR is fast and costs us nothing from the $50).

## Files

| file | role |
|---|---|
| `policy.py` | EVERY kill/warn/respawn/budget decision (pure, table-tested) |
| `test_policy.py` | the scenarios that must never be mis-handled; run by preflight |
| `watchdog.py` | operator-side daemon: observes, applies policy, respawns, ledgers |
| `reaper_lambda.py` + `deploy_reaper.sh` | server-side backstop (EventBridge, 5 min) |
| `budget.json` | program spend ceiling + soft fraction (watchdog budget guard) |
| `gen_dockerfile.py` | emits the Dockerfile (grouped ≤40 GB COPY layers, deps, code) |
| `build_image.sh` | EC2-side: HF download → cached build → push ECR → manifest to S3 |
| `launch_builder.sh` | local: launches the self-terminating EC2 builder |
| `lifecycle.sh` | in-image onstart: trap-destroy, asserts, weights download mode, cells |
| `run_cells.py` | resume-aware cell dispatcher; per-cell S3 bank + progress signal |
| `cells/` | cell modules (`run(cell, out_dir)` contract); `smoke_stats` runs first |
| `controller.py` | local: plan / rent (`--race N`) / respawn / ledger / kill |
| `preflight.sh` | the single ship gate (3 tiers, real base image, real GPU) — replaced the old slim-base `rehearse.sh` |
| `manifests/` | one JSON per rental: gpu query, image, disk, cells, MANDATORY `watch` bounds |

## Offer filter (why each term exists)

`verified=true reliability>0.98 inet_down_cost<=0.0001 inet_up_cost<=0.0001` —
bandwidth-charging hosts would bill the (large) image pull bytes; we only rent hosts
where bytes are free. `inet_down>=200` keeps the unbilled loading phase reasonable
(wall clock only); `weights_mode: "download"` and `"s3"` manifests raise it to `>=800`
because either way the weight pull rides BILLED time. No geolocation filter: geography costs wall clock,
not dollars, and dropping it halved $/hr in the 2026-08-16 pool. `storage_cost<=0.30`
caps the cents. Hosts whose machine_id ever wedged on us are excluded (ledger
`host_verdict` — vast's reliability score did not predict our wedges). GPU RAM /
num_gpus / cuda_vers / disk come from the rental manifest.

## Weights: bake vs download vs s3 (chosen per manifest)

Default `weights_mode: "baked"`: weights ride the image through the unbilled
`loading` phase — proven cheapest for ≤40 GB models. For ≥100 GB models the giant
image was the wedge-prone path (3 host wedges, $1.41 of the program's $2.92), so
`"weights_mode": "download", "weights_hf_repo": ..., "weights_dir": ...` instead
downloads in-run via hf_transfer on a fast host: minutes of billed time (cents)
against zero 150 GB-pull wedge exposure. `timings.tsv` brackets the download so the
ledger audits the trade per rental.

`"weights_mode": "s3"` (MENDELEEV default since 2026-09-03) pulls from OUR OWN paid-for
HF cache `s3://pg-lineage-769063704696/weights/hf/hub/` instead of the hub — the same
cache the AWS lane syncs in ~2 min. `"weights_s3_models"` is a comma list of HF cache
dir names (`models--google--gemma-2-2b`); each is synced with `--exclude '*' --include
'hub/<m>/*'` into `$HF_HOME` (`/workspace/hf`), exactly as `infra/aws_pg/run_job.sh`
does it, so a 3-4B cell never drags the whole cache down. A model the cache does not
hold is simply not listed and falls through to the hub in-run — the fallback is
per-model, not all-or-nothing. `weights_source.json` is banked per rental naming which
model came from which source. `scripts/matrix_vast.py weights_plan` fills both lists by
LISTING the bucket at plan time (the `s3_cached` flag in `matrix_models.ROWS` is
cross-checked against it, never trusted alone). This exists because the hub path killed
two rentals on 2026-09-03 — one net-probed at 7.4 MB/s, one burned 34 billed minutes on
an H100 at max 0% GPU utilisation for $1.45 and produced zero cells, stuck in
`snapshot_download`'s retry loop. Read access for the scoped box user is
`infra/vast/iam/mendeleev_box_policy.json` (GetObject on `weights/hf/*` + ListBucket
conditioned on that prefix; no new write anywhere).

## Runbook (per rental)

```bash
# once ever: server-side backstop reaper (laptop-independent money guarantee)
bash deploy_reaper.sh && bash deploy_reaper.sh invoke

# once per model group (off-meter, EC2; registry cache makes rebuilds ~8 min):
bash launch_builder.sh r_v1_14b            # builds+pushes image, manifest → S3

# per rental (local):
python3 controller.py plan  manifests/r_v1_14b.json          # pick an offer
python3 controller.py rent  manifests/r_v1_14b.json OFFER_ID [--bid 0.40] [--race 2]
python3 -u watchdog.py                                        # THE detection layer
python3 controller.py ledger                                  # realized $ → spend ledger
```

`rent` refuses manifests without declared `watch` bounds and unstamped preflight
SHAs. The watchdog destroys + classifies + auto-respawns (infra failures only, ≤3
attempts, wedged hosts excluded); `--race 2` hedges a wedge-prone big image with a
sibling — the loser costs storage cents. Everything on the instance is autonomous:
smoke cell first (a failed smoke self-destroys for cents), per-cell banking plus a
60 s `out/` stream (a crash loses ≤60 s of rows), verify-then-destroy.

## MENDELEEV additions (2026-09-03)

- `program.json` now also carries `s3_results` (`results/mdl`), `ledger`, and `box_profile`:
  the keys shipped to the box come from the local profile `mendeleev-box` (IAM user of the same
  name, inline policy `infra/vast/iam/mendeleev_box_policy.json`: put/get under `results/mdl/*`
  + ListBucket on that prefix only — D23). `HF_TOKEN` is read from SSM `/pg-lineage/hf-token`
  at rent time (the only live token); the box itself has no SSM access.
- Net probe (D21) in `lifecycle.sh`: 50 MB ranged GET of `results/mdl/_netprobe/probe.bin`
  before any cell; < 10 MB/s → exit 9 + `_rental_complete.json{reason: net_probe}`; policy
  classifies exit 9 as infra → watchdog respawns on another host, verdict `slow_net`.
- Deps-only image `shackleton:mdl-v1` (base `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime`,
  pins = the AWS `pg-env` AMI freeze; NO weights): `MODELS="" PIN_ASSERT="name==ver ..."
  WORKTREE=1 bash launch_builder.sh` — the in-image gate is the pin check. Jobs pull their own
  models through the hub in-run (`cells/mdl_job.py`, `HF_ONLINE=1` in the manifest env).
- Watchdog for this program (do not start it from an infra agent — the lead's session owns it):
  `cd infra/vast && python3 -u watchdog.py` under a session Monitor (`-u` is required, D14).
  Reaper redeployed 2026-09-03 with the `mendeleev_` → `results/mdl/` prefix table.

## Field notes after one full program (2026-08-17, 20 rentals, $2.92 realized)

**Experience and best use.** The design held: bake weights into ECR images (the pull
rides vast's unbilled `loading` state — verified from invoices, not docs), ship code
and manifests at RENT time (a fix never costs an image rebuild), preflight every SHA on
the local box against the real base image and a real GPU before anything launches, and
let the watchdog daemon (`python3 -u watchdog.py` under a session monitor) be the only
thing you trust about liveness. Best flow per job: write the manifest → preflight →
`controller.py plan` (rank by $/hr for measurement jobs, by $/dlperf-hr for training
jobs) → `rent` → walk away; the instrument's own smoke cell buys any remaining bug for
cents. Billed time on a healthy rental is almost pure measurement (a 14B full-depth
sweep bills ~68s; the whole V2b 7200-step training arm billed ~20 min).

**Room for improvement.** (1) Big images (≥100GB) are the weak point: three host
wedges happened during 72B loads — for that tier a billed in-run download (~$0.25) on
a fast host is probably the better trade than a 150GB image. (2) Builders are
cache-less fresh EC2 — a persistent builder volume or `--cache-from` would cut rebuild
cycles from ~50 to ~8 min. (3) The rehearsal harness still uses a slim base; the
preflight gate covers the real base, but folding them together would be cleaner.
(4) Cell-level banking means a long single-cell job that crashes loses its partial
rows — split long arms into resumable cells or stream rows to S3 mid-cell.

**What the $2.92 bought (per job type, realized invoices).** Measurement sweeps are
nearly free: 7B $0.02, 32B $0.08, gemma anchor $0.002, 14B $0.18 (two attempts).
Training arms cost dimes: V2a replication $0.21, V2b 10x continuation $0.38, V3 batch
dial $0.25, V4 substrate ladder $0.01. The ablation (V1a) cost $0.40 — half of it one
killed attempt doing CPU SVDs. The outlier is 72B at $1.41 across five attempts:
almost all of it storage-hours on wedged loads ($0.81 from one 14h silent wedge),
not GPU. Lesson in one line: at this scale, failures cost more than science, and
detection latency is what prices a failure.

**Warnings (each one cost us something).**
- `vastai destroy` prompts `[y/N]` interactively and silently no-ops without
  `echo y |` — this defeated BOTH auto-cleanup layers once (D9).
- Vast base images run an init that swallows `docker run` commands — always
  `--entrypoint` for in-image checks (D6).
- Some hosts run storage drivers with a ~42-layer cap: group weight COPYs into
  ≤40GB layers or pulls fail with "max depth exceeded" (D15).
- The hub freezes `HF_HUB_OFFLINE` at import time: call `hf_setup()` (or unset the
  ENV) BEFORE importing transformers/huggingface_hub (D17).
- device_map=auto sees the PREVIOUS cell's unfreed model and silently CPU-offloads —
  meta-tensor crashes downstream; free GPU memory between cells (D16).
- A watcher that only fires on change is silent when things wedge — every watcher
  needs heartbeats and progress-based stall detection (D14: 14 hours blind).
- Storage bills from CREATION and for stopped instances: destroy, never stop, and
  reconcile invoices per category (`controller.py ledger`) after every rental.
- Some hosts reach `running` but never execute onstart / never reach S3 (2 of 7
  in the 2026-08-18 drills — zero bytes uploaded after 9 min "running"). The
  liveness layer kills them in ≤9 min, they get a `died` verdict, and plans
  exclude them thereafter. Vast's reliability score predicted neither of them.
