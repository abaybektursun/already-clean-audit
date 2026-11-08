# MENDELEEV infrastructure — 2026-09-03 (resumed run)

Worktree `ntm-frozen-gemma-mendeleev`, branch `science/mendeleev`, HEAD `8c8e3831` at the time of
the builder launch (the lead committed science files concurrently; none of the files below are
committed by this agent). Evidence only; no science here. Every AWS call: `--profile fuelos
--region us-east-1`.

## 0. Audit of the interrupted run (verified from `git diff`, not from the prior report)

The prior agent believed the lifecycle/watchdog edits had not applied (heredoc collision). They
HAD applied and are complete:

| file | state found | verdict |
|---|---|---|
| `infra/vast/lifecycle.sh` | net-probe block present (50 MB ranged `s3api get-object`, exit 9, marker `reason: net_probe`); download block gated on `WEIGHTS_HF_REPO` non-empty | complete; `bash -n` OK |
| `infra/vast/watchdog.py` | reads `program.json` (`s3_results`, `ledger`), `s3_json()`, `archive_attempt()` before respawn, exit-9 → `slow_net` in both the live and the instance-gone branch | complete |
| `infra/vast/policy.py` | `NET_PROBE_EXIT = 9`; `has_complete` + `exit_code == 9` → kill/infra | complete |
| `infra/vast/test_policy.py` | two exit-9 cases added | `OK: 26/26 scenarios + invariants` |
| `infra/vast/controller.py` | `CFG` from program.json, `BOX_PROFILE`, `slow_net` in wedged set, `session` in ledger rows | complete, but `creds_env` read the HF token from a memory file that contains no token (see §2.5) — fixed |
| `infra/vast/program.json` | `s3_results`, `ledger`, `box_profile` | complete |
| `infra/vast/reaper_lambda.py` | `BASES` table with `mendeleev_ → results/mdl/` | complete (deployed Lambda was stale — redeployed, §2.6) |
| `infra/vast/build_image.sh`, `launch_builder.sh` | `MODELS` optional, `PIN_ASSERT` gate, `WORKTREE=1` | complete; launcher's HF-token grep would abort under `set -e` when the memory file has no token — fixed |
| `infra/vast/cells/mdl_job.py` | shim complete; `python3 infra/vast/cells/mdl_job.py` dry-run → `mdl_job dry-run OK: ['--row','2','--recipe','full','--arm','rec','--smoke','--out',<tmp>]` | complete |
| `infra/vast/iam/mendeleev_box_policy.json` | put/get/abort-multipart on `results/mdl/*`; ListBucket with `s3:prefix` = `results/mdl/*` | complete; applied §2.5 |
| four `infra/vast/manifests/mdl_*.json` | mandatory `watch`, `weights_mode: download`, `HF_ONLINE=1`, `HF_HUB_DISABLE_XET=1`, image `shackleton:mdl-v1` | complete (gpu queries checked in §3) |
| `studies/lineage/cross-lineage/research/mendeleev_spend.json` | `[]` | exists (ledger schema = controller.py `create_instance` row) |

Not done by the prior run (no diff): `infra/aws_pg/{config.py,pg.py,run_job.sh}`, `scripts/matrix_check.py` — done now (§1).

Note for the lead: `infra/aws_pg/jobs/` has NO `matrix_recovery.py` / `matrix_behaviour.py` yet — the
manifests and the shim target modules the MODELS agent still has to write. Nothing here can be
rented until they exist (the smoke cell would fail at import, costing cents).

## 1. AWS lane

1. `infra/aws_pg/config.py`: `INSTANCE_TYPES = ["g6e.2xlarge", "g6e.4xlarge"]` (spec §11.1 / D26).
   Check: `python3 -c "from infra.aws_pg import config as C; print(C.INSTANCE_TYPES)"` →
   `['g6e.2xlarge', 'g6e.4xlarge']`.
2. `pg run --weights <models--org--name,...>` (also on `map`): `run(..., weights=)` → `_launch` →
   `render_user_data` writes `export PG_WEIGHTS_INCLUDE="…"` before `run_job.sh`; `run_job.sh`
   loops the list with `aws s3 sync $WEIGHTS_S3 /opt/hf-cache/hf/ --exclude '*' --include "hub/<m>/*"`.
   No flag = the full sync (unchanged); `--no-weights` unchanged (empty `WEIGHTS_S3`).
   S3 layout verified: `aws s3 ls s3://pg-lineage-769063704696/weights/hf/hub/` →
   `models--google--gemma-2-2b/ gemma-3-4b-it/ gemma-3-4b-pt/ gemma-scope-2-4b-pt/
   paligemma2-3b-mix-224/ paligemma2-3b-pt-224/ siglip-so400m-patch14-224/` (+ `CACHEDIR.TAG`).
   Rendered user-data (from `.pgvenv/bin/python`):
   `export PG_WEIGHTS_INCLUDE="models--google--gemma-2-2b,models--google--paligemma2-3b-pt-224"`.
3. `run_job.sh`: `nvidia-smi dmon -s u -d 30 > "$OUT/dmon.log" 2>&1 &` right after the streamer
   (so the 30 s `out/` sync streams it); `kill "$DMON"` in `finish()` before the final sync.
   CPU-only box: the background job just exits, nothing can fail the run.
4. N16: `pg.classify(s3_prefix, launched_utc, instance_id)` returns `("EXTERNAL", "external (vast
   watchdog) vast:<id>")` for a `vast:` handle before any S3/EC2 lookup; `reconcile` and `watchdog`
   pass the registry's `instance_id`; `EXTERNAL` maps to exit 2 (in flight), never alarm.
   `pg register <prefix> --handle vast:<id>` accepted (`--help` shows it).
   Check: `classify('results/mdl/x', None, 'vast:12345')` → `('EXTERNAL', 'external (vast watchdog) vast:12345')`.
5. `scripts/matrix_check.py`: `vast_spend()` reads the ledger; new column `Vast $ realised` per
   session, a totals line (`… over N rental(s) (M not yet reconciled)`) and the literal
   `AWS spend: see registry`. `python scripts/matrix_check.py` regenerates `MENDELEEV_STATUS.md`
   (first line = the roster count; the roster itself is the lead's and moved during this run).

## 2. Vast lane

### 2.1 Shim — verified (see table). Env contract on the box: `HF_HUB_DISABLE_XET=1`, `HF_HOME=/workspace/hf`,
`HF_TOKEN` shipped by the controller, `hf_online_setup()` before the job imports transformers.

### 2.2 Manifests and `controller.py plan` (2026-09-03 ~06:40 UTC; base filter = README "Offer filter" + `inet_down>=800` for download mode)

| manifest | gpu_query | disk | watch | top offers |
|---|---|---|---|---|
| `mdl_rec_full_row2` | `gpu_ram>=79 num_gpus=1 compute_cap>=800 cuda_vers>=13.0` (compute-bound → ranked by $/dlperf) | 120 | 5 h / 30 / 30 / 180 | 1) `47059688` 1×RTX PRO 6000 WS $1.549/h, 189 GB RAM/64c, 4012 Mb/s, m34225 KR · 2) `24548919` 1×H100 NVL $2.644/h, 181 GB/28c, 915 Mb/s, m29527 BG — 4 offers |
| `mdl_rec_full_row3b` | same | 120 | same | identical list (same pool) |
| `mdl_row6_behaviour` | `gpu_name in [H100_SXM,H100_PCIE,H100_NVL] gpu_ram>=79 num_gpus=1 cuda_vers>=13.0` | 150 | 3 h / 30 / 20 / 90 | 1) `24548931` H100 NVL $2.656/h, 181 GB/28c, 915 Mb/s, m29527 BG · 2) `36346574` H100 SXM $4.154/h, 221 GB/26c, 13623 Mb/s, m100511 US — 3 offers |
| `mdl_s9pre_row4` | `gpu_ram>=180 compute_cap>=1000 num_gpus=1 cuda_vers>=13.0` | 800 | 2 h / 60 / 20 / 60 | **0 offers** |

S9-pre pool, read-only probe with the cost terms removed (`vastai search offers "gpu_ram>=180
compute_cap>=1000 num_gpus=1"`): exactly 2 single-B200 offers exist —
`48796642` $9.30/h, 2076 GB disk avail, bandwidth free, storage **$0.62/GB/mo**, reliability
**0.968**, m148764 Virginia; `49410057` $9.99/h, only **230 GB** disk avail, bandwidth
**$0.049/GB**, storage $0.94/GB/mo, m56359 Maryland. Both fail `storage_cost<=0.30`; the second
also fails the 800 GB disk and would bill ≈ $17 for a 340 GB pull. Renting S9-pre needs a
deliberate filter relaxation by the lead; at $9.30/h the $12 hard cap is ≈ 1.3 h, i.e.
`max_billed_h` must drop from 2 to ~1.25 for that offer, and 800 GB at $0.62/GB/mo is ≈ $0.68/day.

### 2.3 Image build (deps-only, no weights)
Base: `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime` — the exact torch of the AMI freeze
(torch==2.12.0, PyPI default cu130); the harness's last real builds used `pytorch/pytorch:2.11.0-…`
(v1-72b_ok.log), not a `vastai/pytorch` tag, and vastai/pytorch has no 2.12.0 tag (2.12.1 only).
Pins = the AMI freeze `s3://…/bake/pip-freeze.txt`: torch 2.12.0, transformers 5.9.0, peft 0.19.1,
accelerate 1.13.0, datasets 4.8.5, huggingface_hub 1.16.1, hf_transfer, scikit-learn 1.8.0,
scipy 1.17.1, einops 0.8.2, numpy 2.4.6, pillow 12.2.0, safetensors 0.7.0, tokenizers 0.22.2,
sentencepiece, awscli, vastai, boto3. In-image gate = `PIN_ASSERT` on the first 11 + import smoke.
Preflight: the Ubuntu box (192.168.1.95) is unreachable (ssh timeout), so only `TIER=1` (static)
ran locally: `OK: 26/26 …`, `T1 PASS: layers<=40, manifests sane, no bare destroys`,
`PREFLIGHT PASS (T1) 8c8e3831` → `.preflight_pass` = HEAD. T2/T3 are covered by the in-image pin
gate and the rent-time smoke cell; recorded as a gap.
Launch: `WORKTREE=1 MODELS="" TAG=mdl-v1 BASE=… DISK_GB=100 ITYPE=m7i.2xlarge PIP_PKGS=… PIN_ASSERT=…
bash launch_builder.sh` → shipped `code/shackleton/repo_8c8e3831-dirty.tar.gz` (7.5 MB), builder
**`i-0526300f3840cc8a5`** (m7i.2xlarge, `shackleton-builder-mdl-v1`, self-terminating).
Markers to poll: `s3://pg-lineage-769063704696/code/shackleton/build_logs/mdl-v1_{progress,ok,failed}.log`,
image manifest `code/shackleton/images/image_mdl-v1.json`, ECR tag `shackleton:mdl-v1`.
Self-termination: `InstanceInitiatedShutdownBehavior=terminate` (verified via
`describe-instance-attribute` → `terminate`) + user-data `shutdown -h now` on both outcomes. The
harness arms no deadline for builders, so one was added the `pg._arm_deadline` way:
EventBridge one-shot `pg-reap-i-0526300f3840cc8a5` → `ec2:terminateInstances` at 2026-09-03T08:15Z
(`ActionAfterCompletion=DELETE`; harmless if the builder is already gone).
Polling: `aws s3 ls …/build_logs/` at a 5-min cadence for `mdl-v1_ok.log` / `mdl-v1_failed.log`
(never the log body). At 06:46Z: `mdl-v1_progress.log` 220814 bytes, builder `running`.
Attempt 1 (`i-0526300f3840cc8a5`): `mdl-v1_failed.log` at 06:50Z (archived as
`mdl-v1_failed_attempt1.log`). The BUILD succeeded — log line `PIN GATE OK torch 2.12.0+cu130 cuda
13.0 pins 11` — and the failure is the final `docker push`: 9 layers `Pushed`, then `failed to copy:
rpc error: code = Canceled … grpc: the client connection is closing` on the last layer; ECR shows no
`mdl-v1` tag (manifest never landed). The `#7 ERROR: failed to configure registry cache importer …
not found` line is the expected first-build cache miss, not the cause.
**Root cause (corrected):** my first reading — "a transient in the daemon's push path" — was wrong.
The lead reports calling `pg._arm_deadline` on the builder while my CLI-armed
`pg-reap-i-0526300f3840cc8a5` already existed; `CreateSchedule` raised `ConflictException`, and the
helper's blanket `except Exception: terminate_instances` killed the instance mid-push. The
"client connection is closing" is the daemon dying under termination. The builder did not
"self-terminate as designed" — it was terminated externally; the lead deleted the orphan schedule.
Fixes: (a) `pg._arm_deadline` now catches `ConflictException`, returns the existing schedule's
expression via `get_schedule`, and terminates only on OTHER failures (module docstring updated);
proven on a dummy schedule (`pg-reap-i-00000000000000000`, then deleted) and by arming the
attempt-2 builder twice; (b) `build_image.sh` keeps a bounded push retry (3 tries, 15 s, fail
loud) as hardening — ECR keeps landed layers, so a retry ships only the remainder.
Attempt 2 (`i-0208aea19bd5a9a77`, the lead's authorised restart; HEAD had moved to `27aa3548`, so
`TIER=1` preflight re-ran → `PREFLIGHT PASS (T1) 27aa3548`, tar `repo_27aa3548-dirty.tar.gz`):
deadline `pg-reap-i-0208aea19bd5a9a77` armed through the fixed helper; attempt-1 log archived as
`mdl-v1_failed_attempt1.log` so the poll cannot read a stale marker. Result: see §2.7.

### 2.4 Net probe object
`results/mdl/_netprobe/probe.bin` created server-side (multipart `upload-part-copy`, bytes 0–52428799
of a gemma-2-2b blob; no local bytes moved): `head-object` → `52428800 2026-09-03T06:39:15Z`.
Under the box profile a 1 MB ranged read of that key succeeds (§2.5), so the probe path is exercised
with the exact credentials the box will hold. `S3_BASE` shipped = `s3://…/results/mdl` → the lifecycle
derives key `results/mdl/_netprobe/probe.bin`. Policy classification checked by `test_policy.py`.

### 2.5 IAM user `mendeleev-box`
`iam create-user` → `arn:aws:iam::769063704696:user/mendeleev-box`; inline policy `mendeleev-box-s3`
from `infra/vast/iam/mendeleev_box_policy.json`; one access key written straight into the local
profile `mendeleev-box` via `aws configure set` (secret never printed; the scratch file deleted).
Scope check with that profile: `s3 ls results/mdl/` OK · put+get `results/mdl/_iamcheck/t.txt` OK
(10-byte object left in place: versioned bucket) · 1 MB ranged get of the probe OK ·
`s3 ls weights/` **AccessDenied** · `s3 ls results/` **AccessDenied** · `s3 cp bake/DONE` **403** ·
`ssm get-parameter /pg-lineage/hf-token` **AccessDeniedException**.
`controller.creds_env` ships THIS profile's keys (`program.json.box_profile`) and now reads
`HF_TOKEN` from SSM `/pg-lineage/hf-token` with the fuelos profile (memory
`reference_infrastructure.md` line 45: "the ONLY live token is SSM"); the previous code asserted on a
regex over that memory file, which contains no token — rent would have failed at the first call.
`launch_builder.sh` got the same fix (SSM, only when `MODELS` is non-empty).

### 2.6 Reaper
`bash deploy_reaper.sh invoke` on the deployed code (LastModified 2026-08-19) → `checked 0 harness
instances, reaped 0` — but that code only knew `shackleton_`/`amundsen_`, so `mendeleev_` rentals
were invisible to layer 4. Redeployed (`bash deploy_reaper.sh` → `[deploy_reaper] code updated`,
LastModified 2026-09-03T06:42:38Z) and invoked again → `{"checked": 0, "reaped": []}`.
Schedule `shackleton-reaper-5min`: ENABLED, `rate(5 minutes)`.

### 2.8 Watchdog (documented, NOT started)
`cd infra/vast && python3 -u watchdog.py` under the lead's session Monitor (`-u` required, D14).
It filters instances by label `mendeleev_*`, reads/writes `studies/lineage/cross-lineage/research/
mendeleev_spend.json`, and enforces `budget.json` (`program_ceiling_usd: 30`; Vast credit at run
time $37.82, `vastai show user`). `vastai show instances` → 0 instances at every point of this run.

## 3. Waste risks in the launch path (for the lead)
- `controller.py rent` → `ship()` requires `.preflight_pass`'s SHA to cover HEAD's `infra/` TREE;
  committing the infra changes moves that tree → re-run `cd infra/vast && TIER=1 bash preflight.sh`
  (or the box T3 when reachable) after the commit, before the first rent.
- The manifests' job modules (`infra.aws_pg.jobs.matrix_recovery`, `matrix_behaviour`) do not exist
  yet; the smoke cell fails at import (cents, but a wasted respawn cycle would not happen: a
  traceback is `science` → no respawn).
- Top offer for the two full-FT manifests is a Blackwell RTX PRO 6000 WS (96 GB) at $1.55/h ranked
  by $/dlperf; `expected_gpu` is empty so nothing asserts the class — fine for bf16/fp32 Adam, but
  the H100 NVL at $2.64/h is the conventional choice if the lead wants Hopper numerics.
- S9-pre: 0 offers under the base filter (see §2.2); any rent there is a manual filter relaxation
  with `max_billed_h` re-derived from the $12 cap at the offer's $/h.
- `MENDELEEV_STATUS.md` totals are estimates until `controller.py ledger` reconciles invoices.

### 2.7 Build result (attempt 2, `i-0208aea19bd5a9a77`)
`mdl-v1_ok.log`: `PIN GATE OK torch 2.12.0+cu130 cuda 13.0 pins 11` → `[build] pushed
769063704696.dkr.ecr.us-east-1.amazonaws.com/shackleton@sha256:f2975abb9a712bb367039c0aded95bffe21e89527cd1b515b964ce6d12c73035`
(no push retry was needed). ECR: tag `mdl-v1`, that digest, pushed 06:51:49Z, 3.33 GB.
`code/shackleton/images/image_mdl-v1.json`: base `pytorch/pytorch@sha256:c263514402c8…`, models "",
code tar `repo_27aa3548-dirty.tar.gz`, versions torch 2.12.0+cu130 / transformers 5.9.0, built
06:51:51Z. Builder `terminated` by itself; its deadline schedule deleted; EC2 fleet and Vast both empty.
Manifests already carried `shackleton:mdl-v1`; the digest above is the pin of record for §11.9.

### 2.9 Hopper pin + re-stamp + plans (post-build, HEAD `295d82a2`)
`mdl_rec_full_row2` / `mdl_rec_full_row3b`: `gpu_query` → `gpu_ram>=79 num_gpus=1 compute_cap>=800
compute_cap<1000 cuda_vers>=13.0`, `expected_gpu: "H100"` (lifecycle asserts the substring at boot;
spec confound 27 — one hardware class on the Vast lane). `TIER=1 bash preflight.sh` →
`PREFLIGHT PASS (T1) 295d82a2`; `git rev-parse <stamp>:infra == HEAD:infra` → rent gate green.
Plans: row2 — `24548919` H100 NVL $2.644/h 181 GB/28c 915 Mb/s m29527 BG; `36551369` H100 SXM
$4.340/h 221 GB/26c 10.5 Gb/s m108977 US (3 offers). row3b — `24548921` H100 NVL $2.644/h (same
host m29527); `36551369` (3 offers). row6 — `24548919` H100 NVL $2.656/h; `36346574` H100 SXM
$4.154/h 13.6 Gb/s m100511 US (3 offers). S9-pre — 0 offers (unchanged, §2.2). No rental started.
