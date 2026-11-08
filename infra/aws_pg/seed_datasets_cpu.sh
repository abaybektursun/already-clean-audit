#!/usr/bin/env bash
# CPU DATASET seed (EC2 user-data, runs as root on Amazon Linux 2023 arm64).
#
# WHY THIS EXISTS (2026-09-03). Weight caching works; dataset caching did not exist. Measured on a
# live pack job: the S3 weight restore moved 16.94 GB in 99 s (~220 MB/s), and the SAME job then
# spent 1.8 h pulling ~15.8 GB of EVALUATION DATASETS from the hub at 1-5 MB/s before its first GPU
# stage (jobs call `diff_harvest.hf_online_setup()`, which flips HF_HUB_OFFLINE to 0, and run with
# HF_HUB_DISABLE_XET=1). Image-bearing rows were spending more wall-clock before the first stage
# than on the science. This is the weight cache's twin for datasets: seed once here, restore from
# S3 in `run_job.sh`, fall back to the hub only for what is missing.
#
# WHY CPU / WHY m7g.medium: identical reasoning to seed_weights_cpu.sh — the work is
# download-then-upload, needs no GPU, and must not touch the 32-vCPU G quota the science fills. The
# account's Standard on-demand quota is 1 vCPU (verified 2026-09-03), and m7g.medium is the only
# shape that fits it with a real network allocation.
#
# LAYOUT: HF_HOME here is /opt/pg/hfds, and the whole tree is synced to s3://$B/datasets/hf/, so S3
# mirrors an HF cache exactly — `hub/datasets--<org>--<name>/…` (raw downloads) and `datasets/…`
# (the built arrow). `run_job.sh` restores it INTO /opt/hf-cache/hf/, where it merges with the
# weight cache's `hub/models--…` without collision.
#
# BLOBS: `aws s3 sync` DEREFERENCES the HF cache's symlinks, so `blobs/<hash>` and
# `snapshots/<sha>/<file>` would be two full copies of the same bytes (the lesson that halved the
# weight sync; run_job.sh:118). Excluded HERE, at upload, so it is never paid for at all — and then
# PROVED to still load, offline, from a blob-less restore before this script reports SEED_OK.
#
# Syncs after EVERY set so a mid-run failure banks everything already finished. Self-terminates via
# the EXIT trap + InstanceInitiatedShutdownBehavior=terminate; an EventBridge reaper is the backstop.
exec > /var/log/pg-seed-ds.log 2>&1
B=pg-lineage-769063704696
up() { aws s3 cp /var/log/pg-seed-ds.log s3://$B/datasets/seed.log --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

# stream the log to S3 every 30s so progress is visible mid-run.
( while true; do aws s3 cp /var/log/pg-seed-ds.log s3://$B/datasets/seed.log --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

df -h /
dnf install -y python3.11 python3.11-pip
python3.11 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
# PINNED TO THE AMI (bake_env.sh): datasets 4.8.5 + huggingface_hub 1.16.1. The arrow cache key is a
# function of the library's own builder-config fingerprint, so a version mismatch here would seed a
# cache the job cannot hit — it would silently re-download and this whole exercise would buy nothing.
/opt/pg/venv/bin/pip install 'datasets==4.8.5' 'huggingface_hub==1.16.1'

mkdir -p /opt/pg/hfds
export HF_HOME=/opt/pg/hfds HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN=$(aws ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
                  --region us-east-1 --query Parameter.Value --output text)

# The CALLS ARE THE CONTRACT. A cache entry is keyed by the exact (repo, config, split, data_files,
# verification_mode) tuple, so each line below is byte-identical to the call the job makes; the
# source of each is named. Anything else would land in the cache under a key nothing looks up.
cat > /opt/pg/fetch_ds.py <<'PY'
import json
import sys
from datasets import load_dataset, load_dataset_builder

# (repo, config, split, kwargs) — verbatim the pipeline's calls:
#   text_eval.py:83,182,278,664,906  load_dataset("cais/mmlu", "all")              [all splits]
#   matrix_items._rows               ld(repo, config, split=split)
#   text_eval.py:372,458             wikitext-2-raw-v1 / test        (the ppl column)
#   text_eval.py:533                 cais/wmdp bio+cyber / test
#   matrix_recovery.py:580           trivia_qa rc.nocontext / train  (the recovery trainer)
#   matrix_behaviour.py:265,
#   vlm_skill_eval.load_rows:215     ld(vision_repo, split=split)    (the O column)
#   matrix_items.vqav2_train         ld(VQAv2, data_files={"train": 6 shards}, split="train",
#                                       verification_mode="no_checks")   (REH)
SETS = [
    ("cais/mmlu", "all", None, {}),
    ("Salesforce/wikitext", "wikitext-2-raw-v1", "test", {}),
    ("cais/wmdp", "wmdp-bio", "test", {}),
    ("cais/wmdp", "wmdp-cyber", "test", {}),
    ("mandarjoshi/trivia_qa", "rc.nocontext", "validation", {}),
    ("mandarjoshi/trivia_qa", "rc.nocontext", "train", {}),
    ("lmms-lab-encoder/RefCOCO", None, "testA", {}),
    ("lmms-lab-encoder/textvqa", None, "validation", {}),
    ("lmms-lab-encoder/VQAv2", None, "validation", {}),
    ("lmms-lab-encoder/VQAv2", None, "train",
     {"data_files": {"train": [f"data/train-{i:05d}-of-00136.parquet" for i in range(6)]},
      "verification_mode": "no_checks"}),
]

if sys.argv[1] == "count":
    print(len(SETS)); raise SystemExit
repo, config, split, kw = SETS[int(sys.argv[1])]
if split is not None:
    kw["split"] = split
args = (repo,) if config is None else (repo, config)
# `_hf_split_rows` (matrix_items.py:399) reads split sizes from the builder metadata for every
# VISION repo — cheap, and it puts the repo's metadata in the same cache.
print("BUILDER", repo, json.dumps({k: v.num_examples for k, v in (load_dataset_builder(*args).info.splits or {}).items()}), flush=True)
ds = load_dataset(*args, **kw)
n = ds.num_rows if hasattr(ds, "num_rows") else {k: v.num_rows for k, v in ds.items()}
print("FETCHED", repo, config, split, n, flush=True)
PY

N=$(/opt/pg/venv/bin/python /opt/pg/fetch_ds.py count)
echo "seeding $N dataset entries"

for i in $(seq 0 $((N - 1))); do
  /opt/pg/venv/bin/python /opt/pg/fetch_ds.py "$i"
  du -sh /opt/pg/hfds
  df -h /
  # `xet/` is huggingface_hub's local chunk cache — a download accelerator's scratch, not data.
  # It is useless to a restoring job and it grows with the bytes pulled, so it never goes to S3.
  # `*.incomplete` is a partial download/build that the library ignores and then deletes locally;
  # synced mid-flight it would be orphaned garbage in S3 forever (sync never deletes).
  aws s3 sync /opt/pg/hfds/ s3://$B/datasets/hf/ --exclude '*blobs/*' --exclude 'xet/*' \
    --exclude '*.incomplete*' --region us-east-1
  echo "SEEDED entry $i"
done
echo "SEED_OK"

# ---- restore probe: the blob-less S3 copy must actually load, OFFLINE ------------------------
# The same shape of proof seed_weights_cpu.sh runs for weights. If this fails, the seed is not
# usable and the log says so loudly rather than a job discovering it at 3 a.m.
# Free the source first: everything in it is already in S3, and the image splits are big enough
# that keeping both copies on one 300 GB volume is how this step would die (VQAv2 validation is
# 214,354 rows, each carrying its image).
du -sh /opt/pg/hfds; rm -rf /opt/pg/hfds; df -h /
mkdir -p /opt/pg/dsprobe
aws s3 sync s3://$B/datasets/hf/ /opt/pg/dsprobe/ --region us-east-1
du -sh /opt/pg/dsprobe
HF_HOME=/opt/pg/dsprobe HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /opt/pg/venv/bin/python - <<'PY'
from datasets import load_dataset
mm = load_dataset("cais/mmlu", "all", split="test")
print("PROBE mmlu/test rows:", mm.num_rows, "| first subject:", mm[0]["subject"])
wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
print("PROBE wikitext/test rows:", wt.num_rows)
vq = load_dataset("lmms-lab-encoder/VQAv2", split="validation")
print("PROBE vqav2/validation rows:", vq.num_rows, "| cols:", vq.column_names)
print("DATASET_RESTORE_OK")
PY
# ---------------------------------------------------------------------------------------------

aws s3 ls s3://$B/datasets/hf/ --recursive --summarize | tail -3
date -u > /tmp/DONE
aws s3 cp /tmp/DONE s3://$B/datasets/DONE --region us-east-1
up
