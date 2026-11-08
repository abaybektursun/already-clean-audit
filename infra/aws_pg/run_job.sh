#!/usr/bin/env bash
# In-instance job runner. Invoked by per-launch user-data AFTER the repo tarball is
# fetched+extracted to /opt/pg/repo. Runs exactly ONE job fail-loud, streams the log and
# output to S3 every 30s (near-continuous durability + live `pg logs`), then on ANY exit
# uploads the tail and shuts down.
#
# Teardown model: `shutdown -h now` + InstanceInitiatedShutdownBehavior=terminate is the
# FAST PATH (clean jobs die in seconds). The per-instance EventBridge schedule armed by the
# launcher is the OS-INDEPENDENT backstop that kills a wedged box regardless of this script.
# The `|| true` on the teardown uploads is deliberate: a failed final upload must never block
# shutdown (idle burn is worse); per-artifact syncs during the run are the real R3 guarantee.
set -euo pipefail
# -e: fail-fast on any step (weight sync, env exports, etc.); the EXIT trap below still runs.
# `|| true` is intentionally retained on teardown uploads so a failed final upload never blocks
# `shutdown -h now` (idle burn is worse).
JOB_MODULE="$1"; JOB_ARGS="$2"; S3PREFIX="$3"; BUDGET="${4:-43200}"; WEIGHTS_S3="${5:-}"
B=pg-lineage-769063704696
S3="s3://$B/$S3PREFIX"
OUT=/work/out; mkdir -p "$OUT"
LOG=/var/log/pg-job.log
: > "$LOG"
# route ALL output to the streamed log so weight-sync + everything is visible via `pg logs`.
exec > >(tee -a "$LOG") 2>&1

# ---------------------------------------------------------------------------------------------
# HEARTBEAT — liveness measured by the SUPERVISOR, not inferred from what the job prints.
#
# The failure this closes: for the whole life of this harness the only signal reaching S3 was the
# job's own stdout, so "is it alive" could only be answered when the science code happened to
# speak. It does not speak inside a `snapshot_download` of a 5 GB shard or a 16384-wide
# eigendecomposition — precisely the operations you most want to watch. A healthy job mid-download
# and a dead one produce byte-identical silence, and `run_job.sh` re-uploads job.log every 30 s
# regardless, so even mtime says nothing.
#
# That gap cost two wrong calls. In M-D97 a census was killed 11/16 through its base because a
# frozen log read as a stall; five hours of work destroyed. In the recensus7 case a job that was
# merely slow (one repo's shards, >2.5 h) was diagnosed as wedged and a redundant hedge launched —
# the job then finished on its own. Both times the response was to write a comment saying you
# cannot tell from outside. A comment is not a sensor.
#
# These four counters make the question DECIDABLE rather than a judgement call:
#   cpu_seconds   advancing -> the process is computing
#   net_rx_bytes  advancing -> the process is downloading (the recensus7 case exactly)
#   read/write_bytes advancing -> the process is doing disk I/O
#   none advancing across several samples -> genuinely stuck, and now that is EVIDENCE
#
# Read from /proc via the pid rather than reported by the job, so a wedged interpreter cannot
# fake it and a silent-but-working one cannot be mistaken for dead. Every field defaults to 0 and
# the whole thing is `|| true` — a heartbeat must never be able to kill the run it observes.
HB=/var/log/pg-heartbeat.jsonl
: > "$HB"
heartbeat() {
  local pid cpu rss rb wb net rx tx p comm t best
  # PICK THE INTERPRETER, NOT ITS WRAPPER. The job runs as
  #   timeout --signal=KILL <n>s /opt/pg/venv/bin/python -m <module>
  # so `pgrep -f /opt/pg/venv/bin/python` matches BOTH the python process and the `timeout` parent,
  # whose command line contains that same path — and `head -1` takes the lowest pid, which is the
  # parent. The first version of this heartbeat did exactly that and reported the wrapper's
  # counters: cpu_seconds pinned at 0 and rss at 1900 KB while the job was mid-computation.
  #
  # That is a fail-OPEN bug in the guard, not a cosmetic one. With cpu_seconds permanently 0, a
  # compute-bound job (the M-D97 case — a long eigendecomposition with no network traffic) would
  # classify NO-PROGRESS, and `pg kill` would have happily terminated the exact job the guard exists
  # to protect. Matching on /proc/<pid>/comm is what separates them: the wrapper's comm is
  # `timeout`, the interpreter's is `python3`.
  #
  # AND PICK THE WORKING CHILD, NOT THE SUPERVISING PARENT (2026-09-03). A pack job runs
  # `infra.aws_pg.jobs.pipeline`, which spawns each step with `subprocess.run([sys.executable, ...])`.
  # BOTH are `/opt/pg/venv/bin/python` with comm `python3`, so the comm filter passes on the parent
  # too — and the parent is the LOWER pid, blocked in `subprocess.run`, accumulating no CPU. Every
  # pack heartbeat therefore reported `cpu_seconds: 0` while the GPU sat at 92-100 % (measured on
  # i-0f0bf7e02bed28de5 and i-09ce294a93bb91d74: cpu_seconds 0 and rss ~14 MB in every sample).
  # That is the SAME fail-open the comm filter was written to close, one level down: a compute-bound
  # pack with little network traffic classifies NO-PROGRESS and `pg kill`'s guard would green-light
  # killing it. Taking the LARGEST utime+stime picks the process actually doing the work — the
  # supervising parent can never out-accumulate the child it is blocked on, and on a single-step
  # (non-pack) job there is only one python to choose from, so the old behaviour is unchanged.
  pid=""; best=-1
  for p in $(pgrep -f '/opt/pg/venv/bin/python' 2>/dev/null); do
    comm=$(cat "/proc/$p/comm" 2>/dev/null || echo "")
    case "$comm" in python*) ;; *) continue ;; esac
    t=$(awk '{print $14+$15}' "/proc/$p/stat" 2>/dev/null || echo 0)
    if [ "${t:-0}" -gt "$best" ]; then best="$t"; pid="$p"; fi
  done
  cpu=0; rss=0; rb=0; wb=0
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    # /proc/<pid>/stat fields 14,15 = utime,stime in clock ticks (100/s on Linux)
    cpu=$(awk '{printf "%d", ($14+$15)/100}' "/proc/$pid/stat" 2>/dev/null || echo 0)
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    rb=$(awk '/^read_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null || echo 0)
    wb=$(awk '/^write_bytes:/{print $2}' "/proc/$pid/io" 2>/dev/null || echo 0)
  fi
  net=$(awk 'NR>2 {gsub(/:/," "); if ($1!="lo") {r+=$2; t+=$10}} END{printf "%d %d", r+0, t+0}' \
        /proc/net/dev 2>/dev/null || echo "0 0")
  rx=${net%% *}; tx=${net##* }
  printf '{"utc":"%s","pid":"%s","cpu_seconds":%s,"rss_kb":%s,"read_bytes":%s,"write_bytes":%s,"net_rx_bytes":%s,"net_tx_bytes":%s,"log_bytes":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${pid:-none}" "${cpu:-0}" "${rss:-0}" "${rb:-0}" "${wb:-0}" \
    "${rx:-0}" "${tx:-0}" "$(stat -c%s "$LOG" 2>/dev/null || echo 0)" >> "$HB"
}

( while true; do
    heartbeat || true
    aws s3 cp "$LOG" "$S3/job.log" --region us-east-1 >/dev/null 2>&1
    aws s3 cp "$HB" "$S3/heartbeat.jsonl" --region us-east-1 >/dev/null 2>&1
    aws s3 sync "$OUT" "$S3/out/" --region us-east-1 >/dev/null 2>&1
    sleep 30
  done ) &
STREAMER=$!
# N13 (spec §11.5): GPU utilisation sampled every 30 s for the whole job (SM-active %, mem %) — the
# 60 % rule is judged from this file. It lives in out/ so the 30 s sync streams it. On a CPU-only
# box nvidia-smi is absent and the background job simply exits; nothing here can fail the run.
nvidia-smi dmon -s u -d 30 > "$OUT/dmon.log" 2>&1 &
DMON=$!

finish() {
  code=$?
  kill "$STREAMER" 2>/dev/null || true
  kill "$DMON" 2>/dev/null || true
  heartbeat || true
  aws s3 cp "$HB" "$S3/heartbeat.jsonl" --region us-east-1 || true
  aws s3 cp "$LOG" "$S3/job.log" --region us-east-1 || true
  aws s3 sync "$OUT" "$S3/out/" --region us-east-1 || true
  printf 'exit=%s\n' "$code" > /tmp/STATUS && aws s3 cp /tmp/STATUS "$S3/STATUS" --region us-east-1 || true
  shutdown -h now
}
trap finish EXIT

# sync weights from S3 to local disk — sequential, network-speed, no EBS lazy-load. GPU jobs
# pass a weights URI; CPU-only jobs (e.g. finalize) pass empty to skip.
# PG_WEIGHTS_INCLUDE (pg run --weights a,b): sync ONLY those HF-cache model dirs (hub/<m>/*) — a
# 3B cell should not pull the whole 7-model set (MATRIX_CONTRACT §9); models not in S3 download from
# the hub in-job. Empty = today's full sync.
# --exclude '*blobs/*' HALVES the transfer and loses nothing. The seeder uploaded the HF cache with
# symlinks DEREFERENCED, so hub/<m>/blobs/<hash> and hub/<m>/snapshots/<sha>/<file> are two full copies
# of the same bytes (verified: identical ContentLength and identical leading bytes; gemma-2b is 9.4 GiB
# in S3 for 4.7 GiB of model). transformers/huggingface_hub read the SNAPSHOT path, and
# snapshot_download(..., local_files_only=True) resolves against a blob-less cache with every
# safetensors shard, the index and the tokenizer files present and non-empty (proved locally
# 2026-09-03). Filter order matters: the exclusion must come AFTER --include or the include wins.
# The SEEDER is deliberately unchanged — re-seeding costs more than it saves and the blobs are paid for.
mkdir -p /opt/hf-cache/hf
if [ -n "$WEIGHTS_S3" ]; then
  if [ -n "${PG_WEIGHTS_INCLUDE:-}" ]; then
    for m in ${PG_WEIGHTS_INCLUDE//,/ }; do
      echo "syncing weights hub/$m from $WEIGHTS_S3 ..."
      time aws s3 sync "$WEIGHTS_S3" /opt/hf-cache/hf/ --exclude '*' --include "hub/$m/*" --exclude '*blobs/*' --region us-east-1
    done
  else
    echo "syncing weights from $WEIGHTS_S3 ..."
    time aws s3 sync "$WEIGHTS_S3" /opt/hf-cache/hf/ --exclude '*blobs/*' --region us-east-1
  fi
fi

# THE SAME CACHE, FOR DATASETS (2026-09-03). Weight caching worked; dataset caching did not exist,
# and the asymmetry dominated the wall-clock: on i-09ce294a93bb91d74 the weight restore moved
# 16.94 GB in 99 s (~220 MB/s) and the SAME job then spent 1.8 h pulling ~15.8 GB of evaluation
# datasets from the hub at 1-5 MB/s before its first GPU stage (jobs flip HF_HUB_OFFLINE to 0 via
# `diff_harvest.hf_online_setup()` and run with HF_HUB_DISABLE_XET=1, so the hub path is the slow
# one). Image-bearing rows were spending more wall-clock before the first stage than on the science.
# `infra/aws_pg/seed_datasets_cpu.sh` seeds an HF-cache-shaped copy of the evaluation sets
# (MMLU, wikitext, WMDP, TriviaQA, RefCOCO/VQAv2/TextVQA + the VQAv2 REH train shards) to
# s3://<bucket>/datasets/hf/ — both `hub/datasets--*` (raw) and `datasets/*` (built arrow), so a
# cache HIT needs no hub round-trip at all. Restored into the SAME HF_HOME as the weights, where
# `hub/datasets--*` and `hub/models--*` sit side by side without collision.
# NOT a gate: this is a cache, so a MISS is legal — anything absent still downloads from the hub
# in-job exactly as before, and an empty/absent prefix makes the sync a fast no-op.
# TEXT BY DEFAULT, IMAGES ON REQUEST — and the reason is a measurement, not a preference.
# The text sets (MMLU + wikitext + WMDP + TriviaQA) are ~0.4 GB and every row reads at least one, so
# they always restore: 434 MB, seconds, always worth it. The IMAGE sets are 182 GB in S3. Not because
# images are large in themselves — because `load_dataset(repo, split=...)` materialises EVERY split
# of the repo, and the pipeline calls it that way to read 200 rows (`matrix_items._select_rows`).
# Seeded and measured, i-0d8f0f3593f47ee97: VQAv2 155.4 GB (124.1 arrow + 31.3 parquet), TextVQA
# 21.0, RefCOCO 5.4 -> 181.8 GB of images against 434 MB of text (MMLU 220.6 MB, TriviaQA 189.0,
# wikitext 21.3, WMDP 3.2). ROOT_GB is 200, so the image half DOES NOT FIT beside any row's weights
# at all — it is seeded, and it is not restorable onto this instance shape until either the science
# call is narrowed or this sync learns to take one split's arrow files. Restore rate measured at
# ~110 MB/s (25 GB in 3m45s, i-0e0e7d8859a6e3101).
# THAT REAL FIX LANDED, 2026-09-03 (SUBSETS agent). The eval-split half of the image cache is GONE
# from S3 — the eval subsets below replace it — so PG_DATASETS_VISION=1 no longer restores the
# RefCOCO testA / TextVQA validation / VQAv2 validation splits: they are not there. What survives
# under the image includes is the 3.71 GB the REH recovery arm actually needs and can hold, the six
# VQAv2 `train-*-of-00136` shards plus their arrow build (`matrix_items.vqav2_train`, 19,578 rows) —
# already narrowed by `data_files=`, which is exactly the pattern `_select_rows` now follows.
# --exclude '*blobs/*' is belt-and-braces: the seeder already excludes blobs at UPLOAD (symlinks are
# dereferenced by `aws s3 sync`, so `snapshots/` alone is complete). Nothing here should match; if a
# future seed uploads blobs, this keeps the job from paying for them twice. NOTE: the seeder's own
# offline-load probe of that blob-less copy did NOT run — 182 GB cannot be restored beside the
# 175 GB it was built from on one 300 GB volume — so the standing evidence that the blob-less layout
# loads is the live jobs reading it, not that probe.
DATASETS_S3="${PG_DATASETS_S3:-s3://$B/datasets/hf/}"
echo "syncing text evaluation datasets from $DATASETS_S3 ..."
time aws s3 sync "$DATASETS_S3" /opt/hf-cache/hf/ --exclude '*' \
  --include '*mmlu*' --include '*wikitext*' --include '*wmdp*' --include '*trivia_qa*' \
  --exclude '*blobs/*' --exclude 'xet/*' --region us-east-1
# THE 200 SCORED ROWS, NOT THE SPLIT THEY CAME FROM (2026-09-03). The image sync below is what does
# not fit; this is its replacement. `seed_eval_subsets_cpu.sh` writes one `save_to_disk` directory per
# vision task holding exactly the 200 rows the frozen index lists name, images included — tens of MB
# per task against tens of GB for the split. `matrix_items.vision_rows` reads them from here (default
# PG_EVAL_SUBSETS=/opt/hf-cache/eval_subsets) and re-verifies every row's content hash against the
# git-tracked `infra/pg/vision_eval_index.json` before scoring, so a corrupted or stale subset raises
# instead of scoring a different item set. Absent, the O column falls back to materialising the split.
# Gated on the seeder's DONE marker, which it writes LAST — after publishing and after its own
# offline restore probe. A half-published prefix restored mid-seed would be a subset the job cannot
# verify, and the verifier's job is to raise, so the sync must not start one.
SUBSETS_S3="${PG_EVAL_SUBSETS_S3:-s3://$B/datasets/eval_subsets/}"
if aws s3 ls "${SUBSETS_S3}DONE" --region us-east-1 >/dev/null 2>&1; then
  echo "syncing the vision eval subsets (the 200 scored rows) ..."
  time aws s3 sync "$SUBSETS_S3" /opt/hf-cache/eval_subsets/ \
    --exclude 'seed.log' --exclude 'DONE' --region us-east-1
  du -sh /opt/hf-cache/eval_subsets || true
else
  echo "no published vision eval subsets at $SUBSETS_S3 (no DONE marker) — O falls back to the split"
fi
if [ "${PG_DATASETS_VISION:-0}" = "1" ]; then
  echo "PG_DATASETS_VISION=1: syncing what is left of the image cache — 3.71 GB, the VQAv2 REH train"
  echo "shards and their arrow build. The 178.08 GB of eval splits were deleted on 2026-09-03; the"
  echo "eval subsets above replace them. Set this for a row whose recovery arm is REH or --im."
  time aws s3 sync "$DATASETS_S3" /opt/hf-cache/hf/ --exclude '*' \
    --include '*RefCOCO*' --include '*ref_coco*' --include '*textvqa*' \
    --include '*VQAv2*' --include '*vq_av2*' \
    --exclude '*blobs/*' --exclude 'xet/*' --region us-east-1
fi
df -h /
du -sh /opt/hf-cache/hf/datasets /opt/hf-cache/hf/hub 2>/dev/null || true

# PG_S3_PREFIX lets a long job push `out/` to S3 as it goes. Without it the only upload is the
# EXIT trap, so a job killed by the EventBridge reaper (which terminates the instance outright,
# trap or no trap) loses every row it produced. A job that runs for hours per cell has to be able
# to be crash-safe, and it cannot be without knowing where to write.
export HF_HOME=/opt/hf-cache/hf HF_HUB_CACHE=/opt/hf-cache/hf/hub HF_HUB_OFFLINE=1 \
       PYTHONUNBUFFERED=1 PYTHONPATH=/opt/pg/repo TORCHINDUCTOR_CACHE_DIR=/work/jit \
       PG_S3_PREFIX="$S3PREFIX"
cd /opt/pg/repo
# Tokenizer-conversion dependencies the AMI does not carry. `bake_env.sh` pins none of these, and
# transformers imports them only when a repo ships a sentencepiece or tiktoken vocabulary — so they are
# invisible until a row with such a tokenizer runs. Row 7a (zephyr-7b-beta -> Zephyr_RMU) died on
# `ModuleNotFoundError: tiktoken` after 8 MB of log and took 19 cells; row 3c-Act (MolmoAct2) died on the
# protobuf equivalent and took 13. Installed here rather than baked because a rebake costs an AMI cycle
# mid-run; add them to `bake_env.sh` at the next bake and this becomes a no-op.
# Idempotent and offline-safe: the import check skips the install entirely once the AMI carries them.
# NO `|| true` here. It was written that way and it was wrong: a swallowed pip failure is
# indistinguishable from "the deps were already present", and the whole point of this block is to make a
# missing dependency loud BEFORE the job spends an hour loading weights. Fail-loud is the house rule
# (CLAUDE.md: "Fail and trace beats fail and hide"), and a box that cannot install them should not run.
/opt/pg/venv/bin/python - <<'PYDEPS' 
import importlib, importlib.util, subprocess, sys


def have(mod):
    """`find_spec` RAISES ModuleNotFoundError for a dotted name whose PARENT is absent — it imports the
    parent to find the child. So `find_spec("google.protobuf")` on a box without protobuf does not
    return None, it raises "No module named 'google'". With the `|| true` removed (correctly), that
    crash became fatal and killed the 3b recovery packs on 2026-09-04. Catching it here is not hiding an
    error: the exception IS the answer to "is this importable", which is the only question asked."""
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


missing = [m for m in ("tiktoken", "google.protobuf", "sentencepiece") if not have(m)]
if missing:
    pkgs = ["tiktoken" if m == "tiktoken" else "protobuf" if m == "google.protobuf" else "sentencepiece"
            for m in missing]
    print(f"[deps] AMI is missing {missing}; installing {pkgs}", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    importlib.invalidate_caches()
    still = [m for m in missing if not have(m)]
    assert not still, f"[deps] pip reported success but {still} are STILL not importable"
    print(f"[deps] installed and importable: {pkgs}", flush=True)
else:
    print("[deps] tokenizer deps present in the AMI", flush=True)
PYDEPS
# fail-loud: no try/except around the science; a crash exits non-zero and the trap tears down.
timeout --signal=KILL "${BUDGET}s" /opt/pg/venv/bin/python -m "$JOB_MODULE" $JOB_ARGS --out "$OUT"
