#!/usr/bin/env bash
# SHACKLETON §9.2: one rental = one scripted lifecycle. Runs as the vast onstart.
# Every exit path — success, crash, assert failure — uploads the log to S3 and
# DESTROYS this instance (§9.3; stopped instances keep billing storage).
# Env contract (set at `controller.py rent` time):
#   RENTAL       rental id, names the manifest and the S3 prefix
#   S3_BASE      s3://bucket/results
#   VAST_API_KEY for self-destroy          EXPECTED_GPU  substring assert (optional)
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY   for S3 banking
#   VAST_STUB=1  rehearsal mode: no GPU asserts, destroy is a no-op
set -euo pipefail
: "${RENTAL:?}" "${S3_BASE:?}"
# PROGRAM is set at rent time from program.json (controller.creds_env)
S3="$S3_BASE/${PROGRAM:-shackleton}_${RENTAL}"
LOG=/workspace/job.log
TIMINGS=/workspace/timings.tsv
mkdir -p /workspace
stamp() { echo -e "$(date -u +%s.%3N)\t$1" >> "$TIMINGS"; }
export TIMINGS
exec > >(tee -a "$LOG") 2>&1
stamp lifecycle_start

finish() {
  code=$?
  echo "[lifecycle] exit code=$code — uploading log, destroying instance"
  stamp "finish_enter_code_$code"
  aws s3 cp "$TIMINGS" "$S3/timings.tsv" || echo "[lifecycle] WARN: timings upload failed"
  aws s3 cp "$LOG" "$S3/job.log" || echo "[lifecycle] WARN: log upload failed"
  if [[ "${VAST_STUB:-0}" == "1" ]]; then
    echo "[lifecycle] STUB destroy (rehearsal)"
    exit "$code"
  fi
  iid="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-}}"
  iid="${iid#C.}"
  if [[ -z "$iid" ]]; then
    # container env absent: find ourselves by rental label via the API (D9)
    iid=$(vastai show instances --raw --api-key "${VAST_API_KEY:?}" \
      | python3 -c "import json,sys; d=json.load(sys.stdin); m=[i['id'] for i in d if i.get('label')=='${PROGRAM:-shackleton}_${RENTAL}']; print(m[0] if m else '')")
  fi
  # the CLI interactively prompts [y/N] — pipe the confirmation (D9 root cause)
  echo y | vastai destroy instance "$iid" --api-key "${VAST_API_KEY:?}" --retry 6
}
trap finish EXIT

echo "[lifecycle] rental=$RENTAL start=$(date -u +%FT%TZ)"

# ---- qualification (seconds; a lying host costs cents) ----
stamp qual_start
if [[ "${VAST_STUB:-0}" != "1" ]]; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  if [[ -n "${EXPECTED_GPU:-}" ]] && ! grep -qi "$EXPECTED_GPU" <<<"$gpu"; then
    echo "[lifecycle] GPU mismatch: got '$gpu' expected '$EXPECTED_GPU'"
    exit 12
  fi
  python3 -c "import torch; assert torch.cuda.is_available(), 'no CUDA'"
fi

stamp qual_ok
# ---- net probe (D21): before any billed work, measure the pipe we will pay for ----
# A host advertising 800 Mb/s that delivers 5 MB/s turns a $0.10 download into
# dollars of billed idle. 50 MB ranged GET from the program's S3 base, timed;
# < 10 MB/s aborts with exit 9 and a completion marker naming the reason —
# policy classifies exit 9 as infra (respawn elsewhere, host verdict slow_net).
if [[ "${VAST_STUB:-0}" != "1" ]]; then
  stamp net_probe_start
  NET_MBPS=$(python3 - <<'PY'
import os, subprocess, time
bucket, key_base = os.environ["S3_BASE"][len("s3://"):].split("/", 1)
t0 = time.time()
subprocess.run(["aws", "s3api", "get-object", "--bucket", bucket,
                "--key", f"{key_base}/_netprobe/probe.bin",
                "--range", "bytes=0-52428799", "/dev/null"],
               check=True, capture_output=True)
print(f"{50 / (time.time() - t0):.1f}")
PY
  )
  echo "[lifecycle] net probe: ${NET_MBPS} MB/s (50 MB ranged GET)"
  stamp "net_probe_${NET_MBPS}MBps"
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < 10 else 1)" "$NET_MBPS"; then
    echo "[lifecycle] net probe FAILED (< 10 MB/s) — aborting before any billed cell (D21)"
    printf '{"rental": "%s", "reason": "net_probe", "exit_code": 9, "mb_per_s": %s}\n' \
      "$RENTAL" "$NET_MBPS" > /workspace/_rental_complete.json
    aws s3 cp /workspace/_rental_complete.json "$S3/_rental_complete.json"
    exit 9
  fi
fi
# ---- weights (s3 mode) — pull named models from OUR paid-for HF cache, not the hub ----
# The hub path killed two rentals on 2026-09-03: one net-probed at 7.4 MB/s, and one burned 34
# billed minutes on an H100 at max 0% GPU utilisation for $1.45 and produced ZERO cells, because it
# never got out of `snapshot_download`'s retry loop. The same weights already sit in
# s3://pg-lineage-769063704696/weights/hf/hub/, which the AWS lane syncs in ~2 min, and the net
# probe above has just measured THIS EXACT PIPE at >= 10 MB/s — so on this path a slow host is
# caught before any weight byte is billed, which is precisely what the hub path could not do.
# Per-model include, exactly as infra/aws_pg/run_job.sh does it: a 3-4B cell must not drag the whole
# cache down (MATRIX_CONTRACT §9). Models the cache does NOT hold are simply not listed and fall
# through to the hub in-run (cells/mdl_job.py + hf_files), so the fallback is per-model, not
# all-or-nothing. weights_source.json banks which model came from which source, per rental.
if [[ "${WEIGHTS_MODE:-baked}" == "s3" ]]; then
  : "${WEIGHTS_S3_BASE:?}" "${WEIGHTS_S3_MODELS:?}"
  # mdl_job.py does os.environ.setdefault("HF_HOME", "/workspace/hf") — sync into the SAME cache the
  # job will read, on the rental's sized disk rather than the container overlay.
  export HF_HOME="${HF_HOME:-/workspace/hf}"
  mkdir -p "$HF_HOME/hub"
  stamp weights_s3_start
  for m in ${WEIGHTS_S3_MODELS//,/ }; do
    echo "[lifecycle] weights: hub/$m <- $WEIGHTS_S3_BASE (s3 cache)"
    # --exclude '*blobs/*' after the include (last match wins) halves the pull: the cache was seeded
    # with HF symlinks DEREFERENCED, so blobs/ and snapshots/ hold the same bytes twice (gemma-2b:
    # 9.4 GiB in S3, 4.7 GiB pulled without blobs). The library reads the SNAPSHOT path, and
    # snapshot_download(local_files_only=True) resolves against a blob-less cache with every shard,
    # the index and the tokenizer files intact (proved locally 2026-09-03). The seeder still uploads
    # both copies on purpose — re-seeding costs more than it saves.
    time aws s3 sync "$WEIGHTS_S3_BASE" "$HF_HOME/" --exclude '*' --include "hub/$m/*" --exclude '*blobs/*'
    # fail loud: a manifest that names a model the cache does not hold would silently fall back to
    # the very hub path this block exists to avoid, and the first symptom would be a billed stall.
    if [[ -z "$(ls -A "$HF_HOME/hub/$m" 2>/dev/null)" ]]; then
      echo "[lifecycle] weights: hub/$m matched ZERO keys under $WEIGHTS_S3_BASE"
      exit 13
    fi
  done
  stamp weights_s3_done
  printf '{"rental": "%s", "mode": "s3", "s3_base": "%s", "hf_home": "%s", "from_s3": "%s", "from_hub": "%s"}\n' \
    "$RENTAL" "$WEIGHTS_S3_BASE" "$HF_HOME" "$WEIGHTS_S3_MODELS" "${WEIGHTS_HUB_MODELS:-}" \
    > /workspace/weights_source.json
  cat /workspace/weights_source.json
  aws s3 cp /workspace/weights_source.json "$S3/weights_source.json"
fi

# ---- weights (download mode) ----
# For >=100GB models the 150GB baked image was the wedge-prone path (3 host
# wedges, $1.41); on a fast pipe (offer filter adds inet_down>=800) a BILLED
# in-run download is minutes and cents. TIMINGS brackets it so the ledger can
# audit the trade per rental. MENDELEEV manifests declare download mode for the
# fast-pipe filter but leave WEIGHTS_HF_REPO empty: each job pulls its own
# models through the hub in-run (cells/mdl_job.py), so this block is skipped.
if [[ "${WEIGHTS_MODE:-baked}" == "download" && -n "${WEIGHTS_HF_REPO:-}" ]]; then
  : "${WEIGHTS_HF_REPO:?}" "${WEIGHTS_DIR:?}" "${HF_TOKEN:?}"
  stamp weights_download_start
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # image bakes offline=1 (D17)
  python3 -m pip install -q hf_transfer
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY'
import os, time
from huggingface_hub import snapshot_download
# many repos ship the SAME weights twice (safetensors + original/*.pth or *.bin).
# Downloading both doubles BILLED minutes for nothing, so the duplicates are
# excluded by default exactly as the image builder excludes them.
ignore = [p for p in os.environ.get("WEIGHTS_IGNORE",
                                    "*.pth,original/*,*.gguf,*.bin").split(",") if p]
t0 = time.time()
snapshot_download(os.environ["WEIGHTS_HF_REPO"],
                  local_dir=os.environ["WEIGHTS_DIR"],
                  ignore_patterns=ignore,
                  token=os.environ["HF_TOKEN"])
print(f"[lifecycle] weights downloaded in {time.time()-t0:.0f}s (ignored {ignore})")
PY
  stamp weights_download_done
fi

# ---- cells: smoke first (manifest order), per-cell bank+verify, resume-aware ----
export PYTHONPATH=/app
if [[ "${HF_ONLINE:-0}" == "1" ]]; then
  # the image bakes offline=1 as docker ENV; hub freezes it at import (D17)
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
python3 /app/infra/vast/run_cells.py \
  --manifest "/app/infra/vast/manifests/${RENTAL}.json" --s3 "$S3"

stamp all_cells_complete
echo "[lifecycle] all cells complete and verified"
