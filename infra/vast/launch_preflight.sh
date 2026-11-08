#!/usr/bin/env bash
# Run preflight.sh on an ephemeral self-terminating EC2 GPU box and pull the
# stamp back. Replaces run_on_ubuntu.sh when the house Ubuntu box (192.168.1.95)
# is off-network: the ship gate must run against a REAL base image and a REAL
# GPU, so it moves to EC2 rather than being downgraded to a tier-1 static pass.
# AWS side is unlimited by directive; only vast dollars are budgeted.
#
# Usage: bash launch_preflight.sh            # launch, wait, pull the stamp
#        WAIT=0 bash launch_preflight.sh     # launch only
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
AWS=(aws --profile fuelos --region us-east-1)
S3_CODE="s3://pg-lineage-769063704696/code/shackleton"
ITYPE="${ITYPE:-g6e.2xlarge}"      # L40S 48GB — the house GPU class
DISK_GB="${DISK_GB:-200}"
# gp3 at 1,000 MB/s: at the 125 MB/s default the 42 GB n13 image took about 20 min to pull and unpack before T2 could start
BASE="${BASE:-pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime}"   # our images' real base
TIER="${TIER:-3}"

SHA=$(git -C "$ROOT" rev-parse --short HEAD)
DIRTY=$(git -C "$ROOT" status --porcelain -- infra | head -1)
[ -z "$DIRTY" ] || { echo "BLOCKED: infra/ has uncommitted changes — preflight stamps a SHA, so commit first"; exit 3; }
git -C "$ROOT" archive --format=tar.gz HEAD infra ${CODE_PATHS:-} > "/tmp/repo_${SHA}.tar.gz"
"${AWS[@]}" s3 cp "/tmp/repo_${SHA}.tar.gz" "$S3_CODE/repo_${SHA}.tar.gz" >/dev/null

AKID=$(aws configure get aws_access_key_id --profile fuelos)
ASEC=$(aws configure get aws_secret_access_key --profile fuelos)
AMI=$("${AWS[@]}" ssm get-parameter --name \
  /aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.11-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text)

UD=$(cat <<EOF
#!/bin/bash
set -euo pipefail
exec > /var/log/amundsen_preflight.log 2>&1
# credentials are exported with tracing OFF: this log is uploaded to S3, and
# 'set -x' would print the secret key into it (it did, in the first attempts —
# and in every inherited launch_builder.sh log)
export AWS_ACCESS_KEY_ID=$AKID AWS_SECRET_ACCESS_KEY=$ASEC AWS_DEFAULT_REGION=us-east-1
set -x
# VISIBILITY FIRST (D14): the streamer starts before anything that can block,
# so "no news" can never mean "nobody looked".
( while true; do aws s3 cp /var/log/amundsen_preflight.log "$S3_CODE/preflight_logs/${SHA}_progress.log" >/dev/null 2>&1 || true; sleep 30; done ) &
# preflight T2 wants a python with torch+transformers at /tmp/gpuvenv/bin/python3.
# The DLAMI already HAS one; building a venv there needs apt (ensurepip is
# absent) and apt on first boot contends with unattended-upgrades for the dpkg
# lock — 25 silent minutes on the first attempt. Point at the AMI's python and
# add transformers to it instead: no apt, no 3 GB torch re-download.
PY_TORCH=""
for c in "\$(command -v python3)" /opt/pytorch/bin/python3 /opt/conda/bin/python3 /usr/bin/python3; do
  [ -x "\$c" ] || continue
  if "\$c" -c "import torch" 2>/dev/null; then PY_TORCH="\$c"; break; fi
done
[ -n "\$PY_TORCH" ] || { echo "no python with torch on this AMI"; exit 9; }
echo "[preflight] using \$PY_TORCH (\$("\$PY_TORCH" -c 'import torch;print(torch.__version__)'))"
"\$PY_TORCH" -m pip install --quiet --break-system-packages transformers accelerate numpy \
  || "\$PY_TORCH" -m pip install --quiet transformers accelerate numpy
# a WRAPPER, not a symlink: a venv python resolves sys.prefix from its own
# executable path, so calling it through a symlink outside the venv silently
# drops the venv's site-packages (that is why transformers "vanished")
mkdir -p /tmp/gpuvenv/bin
printf '#!/bin/sh\nexec %s "\$@"\n' "\$PY_TORCH" > /tmp/gpuvenv/bin/python3
chmod +x /tmp/gpuvenv/bin/python3
/tmp/gpuvenv/bin/python3 -c "import torch, transformers, numpy; print('[preflight] host python OK', torch.__version__, transformers.__version__)"
mkdir -p /pf && cd /pf
aws s3 cp "$S3_CODE/repo_${SHA}.tar.gz" repo.tar.gz && tar xzf repo.tar.gz
cd infra/vast
# the real image may be one of ours (N13: shackleton:n13-v1 in ECR): log in before preflight pulls BASE
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 769063704696.dkr.ecr.us-east-1.amazonaws.com
PF_SHA=$SHA BASE="$BASE" TIER=$TIER bash preflight.sh && STATUS=pass || STATUS=fail
# N13 (N13_GATE_S3 set, BASE = the n13 image): the image gate (infra/vast/n13_gate.sh) on THIS commit's code, mounted over the
# image's code layer: the venv's freeze against the AWS freeze, Python.h, the baked revisions, the N13 offline gate.
if [ -n "${N13_GATE_S3:-}" ] && [ "\$STATUS" = pass ]; then
  mkdir -p /gate_in && aws s3 sync "${N13_GATE_S3:-}" /gate_in --only-show-errors
  cp /pf/${N13_FREEZE:-} /gate_in/expected_freeze.txt
  docker run --rm -v /pf:/code -v /gate_in:/gate_in:ro -e CODE=/code -e N13_MODELS="${N13_MODELS:-}" --entrypoint bash "$BASE" \
    /code/infra/vast/n13_gate.sh > /var/log/n13_gate.log 2>&1 || STATUS=fail
  grep -q '^N13_IMAGE_GATE_OK' /var/log/n13_gate.log || STATUS=fail
  aws s3 cp /var/log/n13_gate.log "$S3_CODE/preflight_logs/${SHA}_n13_gate.log"
fi
if [ "\$STATUS" = pass ]; then aws s3 cp .preflight_pass "$S3_CODE/preflight_logs/${SHA}.pass"; fi
aws s3 cp /var/log/amundsen_preflight.log "$S3_CODE/preflight_logs/${SHA}_\$STATUS.log"
shutdown -h now
EOF
)

# GPU capacity comes and goes; the gate must not be blocked by one instance
# type being full in one AZ. Try the house type first, then any GPU box that can
# run the same container — preflight needs A gpu, not THE gpu.
IID=""
for T in $ITYPE ${FALLBACK_TYPES:-g6.2xlarge g5.2xlarge g4dn.2xlarge}; do
  IID=$("${AWS[@]}" ec2 run-instances \
    --image-id "$AMI" --instance-type "$T" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"Throughput\":1000,\"Iops\":6000,\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=amundsen-preflight-$SHA}]" \
    --user-data "$UD" --query 'Instances[0].InstanceId' --output text 2>/dev/null) && \
    { ITYPE="$T"; break; } || { echo "[preflight] $T unavailable, trying the next type"; IID=""; }
done
[ -n "$IID" ] || { echo "[preflight] no GPU capacity in any candidate type"; exit 7; }
echo "[preflight] $IID launched at $SHA ($ITYPE, tier $TIER) — it self-terminates"
echo "[preflight] log: ${AWS[*]} s3 cp $S3_CODE/preflight_logs/${SHA}_progress.log -"

[ "${WAIT:-1}" = "1" ] || exit 0
for _ in $(seq 1 120); do   # <= 60 min
  if "${AWS[@]}" s3 cp "$S3_CODE/preflight_logs/${SHA}.pass" "$HERE/.preflight_pass" >/dev/null 2>&1; then
    echo "[preflight] PASS stamp $(cat "$HERE/.preflight_pass") pulled to the local gate"; exit 0
  fi
  if "${AWS[@]}" s3 ls "$S3_CODE/preflight_logs/${SHA}_fail.log" >/dev/null 2>&1; then
    echo "[preflight] FAILED — tail of the log:"
    "${AWS[@]}" s3 cp "$S3_CODE/preflight_logs/${SHA}_fail.log" - | tail -40; exit 5
  fi
  sleep 30
done
echo "[preflight] TIMEOUT after 60 min — check the instance and the progress log"; exit 6
