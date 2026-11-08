#!/usr/bin/env bash
# CPU weights seed (EC2 user-data, runs as root on Amazon Linux 2023 arm64).
#
# WHY CPU: seeding is download-then-upload; it needs no GPU. The GPU path (seed_weights.sh,
# g5.xlarge) eats the 32-vCPU G quota that the science jobs fill completely, so it kept getting
# skipped. This runs on the Standard/on-demand quota instead and costs the science lane nothing.
#
# WHY m7g.medium: the account's Standard on-demand quota is 1 vCPU (verified 2026-09-03, NOT the
# 16 that was assumed; an increase request is a support case, not auto-approved). m7g.medium is
# the only shape that fits: 1 vCPU, 4 GiB, "up to 12.5 Gigabit" network, 1250 MB/s EBS. The
# alternatives at 1 vCPU are t2.* (burstable, "Low to Moderate" network) which would be far slower.
#
# Root volume is sized at launch (300G gp3) and cloud-init growpart expands it, so there is no
# separate scratch BDM + mkfs step. Syncs to S3 AFTER EACH REPO so a mid-run failure banks
# everything already finished. Self-terminates via the EXIT trap +
# InstanceInitiatedShutdownBehavior=terminate; an EventBridge reaper is the AWS-side backstop.
exec > /var/log/pg-seed.log 2>&1
B=pg-lineage-769063704696
up() { aws s3 cp /var/log/pg-seed.log s3://$B/__LOG_KEY__ --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

# stream the log to S3 every 30s so progress is visible mid-run.
( while true; do aws s3 cp /var/log/pg-seed.log s3://$B/__LOG_KEY__ --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

df -h /

# 4 GiB of RAM is not enough headroom for a multi-shard fetch, and the OOM killer is silent in the
# log except for a bare "Killed" (it took out allenai/Molmo2-4B on 2026-09-04 at 06:16). Back the
# box with swap on the 300G root before anything downloads; the volume is provisioned at 500 MB/s.
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
free -h

dnf install -y python3.11 python3.11-pip
python3.11 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
/opt/pg/venv/bin/pip install huggingface_hub

mkdir -p /opt/pg/hf
# huggingface_hub >=1.x ignores HF_HUB_ENABLE_HF_TRANSFER (deprecated, warns) and transfers via
# Xet; HF_XET_HIGH_PERFORMANCE is the live knob. Measured 2026-09-03 on this 1-vCPU box: ~500
# MiB/s down and up, 8 repos (72 GB) in ~50 min.
export HF_HOME=/opt/pg/hf HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN=$(aws ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
                  --region us-east-1 --query Parameter.Value --output text)

cat > /opt/pg/fetch.py <<'PY'
import sys
from huggingface_hub import snapshot_download
# Format duplicates only -- these patterns drop alternate encodings of weights that are also
# present as safetensors (gguf, pytorch .bin, flax/onnx), never anything transformers needs.
# Verified per-repo against the HF API before each seed wave that every repo in REPOS has a
# COMPLETE safetensors set + index. Checked 2026-09-04 for the Molmo2 family: Molmo2-4B and
# Molmo2-ER are 4 shards / 19.42 GB each, MolmoAct2 is 5 shards / 21.78 GB, all safetensors-only,
# none gated.
IGNORE = ["*.gguf", "*.bin", "*.pth", "*.h5", "*.msgpack", "*.onnx", "*.onnx_data"]
# max_workers=1: the default (8) pulls every shard at once, and Molmo2-4B is 4 x 4.85 GB -- more
# concurrent buffer than a 4 GiB box has, which is exactly how the first attempt died. This is a
# 1-vCPU instance whose bottleneck is the network, so serialising costs almost nothing.
print("downloading", sys.argv[1], "->",
      snapshot_download(sys.argv[1], ignore_patterns=IGNORE, max_workers=1))
PY

# Wave 2 (2026-09-04): the Molmo2 family. Row 3b (Qwen3-4B-Instruct -> Molmo2-4B) is the only
# non-Google modern VLM row and its Qwen half is already cached (verified in S3), so Molmo2-4B was
# the single repo forcing a ~19 GB hub re-download on every attempt -- one of which died mid-stream
# at 00:49 with ChunkedEncodingError after 4.75 GB. Molmo2-ER + MolmoAct2 are row 3c-Act's pair and
# are seeded in the same run: the instance-hour is the whole cost, so the marginal price of the
# other two repos is minutes, not dollars.
# STAGEDIFF (2026-09-04): the launcher (`seed_launch_cpu.sh`) substitutes the repo list and the
# two S3 keys (DONE marker, log) into the double-underscore placeholders, so one script seeds any roster.
REPOS="__REPOS__"

for r in $REPOS; do
  /opt/pg/venv/bin/python /opt/pg/fetch.py "$r"
  du -sh /opt/pg/hf
  df -h /
  aws s3 sync /opt/pg/hf/ s3://$B/weights/hf/ --region us-east-1
  echo "SEEDED $r"
done
echo "SEED_OK"


du -sh /opt/pg/hf
date -u > /tmp/DONE
aws s3 cp /tmp/DONE s3://$B/__DONE_KEY__ --region us-east-1
up
