#!/usr/bin/env bash
# Weights seed (runs as root via EC2 user-data on a DLAMI builder with a fresh 300G scratch
# volume). Downloads the 3B model set onto the scratch volume (the DLAMI root is too small for
# 25GB + temp), then `aws s3 sync`s the HF cache to S3. Job instances sync it back to local
# disk at launch (run_job.sh) — so there is no persistent EBS snapshot/volume for weights.
exec > /var/log/pg-seed.log 2>&1
B=pg-lineage-769063704696
up() { aws s3 cp /var/log/pg-seed.log s3://$B/seed/seed.log --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

# stream the log to S3 every 30s so progress is visible mid-run.
( while true; do aws s3 cp /var/log/pg-seed.log s3://$B/seed/seed.log --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.12-venv
python3 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
/opt/pg/venv/bin/pip install huggingface_hub==1.16.1 hf_transfer

# fresh 300G scratch disk (no filesystem) -> xfs -> mount; ample room for 25GB + downloader temp.
DEV=$(lsblk -dprno NAME,SIZE,FSTYPE | awk '$2=="300G" && $3=="" {print $1; exit}')
test -n "$DEV"
mkfs.xfs "$DEV"
mkdir -p /opt/pg/hf
mount "$DEV" /opt/pg/hf

export HF_HOME=/opt/pg/hf HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=$(aws ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
                  --region us-east-1 --query Parameter.Value --output text)

/opt/pg/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
repos = [
    "google/paligemma2-3b-pt-224",
    "google/paligemma2-3b-mix-224",
    "google/gemma-2-2b",
    "google/siglip-so400m-patch14-224",
]
for r in repos:
    print("downloading", r, "->", snapshot_download(r))
print("SEED_OK")
PY

du -sh /opt/pg/hf
echo "syncing HF cache to s3://$B/weights/hf/ ..."
aws s3 sync /opt/pg/hf/ s3://$B/weights/hf/ --region us-east-1
date -u > /tmp/DONE
aws s3 cp /tmp/DONE s3://$B/seed/DONE --region us-east-1
up
