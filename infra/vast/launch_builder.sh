#!/usr/bin/env bash
# Launch the ephemeral self-terminating EC2 image builder (us-east-1, fuelos).
# AWS side is unlimited by directive; only vast dollars are budgeted.
#
# Usage: MODELS="Qwen/Qwen2.5-14B" TAG=v1-14b BASE=vastai/pytorch:<tag> \
#        [DISK_GB=800] [ITYPE=m7i.4xlarge] [PIP_PKGS=...] [PIN_ASSERT=...] [WORKTREE=1] bash launch_builder.sh
# N13 image (one command): bash n13_build.sh
# MODELS may be empty (deps-only image: MENDELEEV jobs download their own weights in-run);
# PIN_ASSERT="name==ver ..." makes the in-image gate a pin check instead of dryrun_v1.py.
set -euo pipefail
: "${TAG:?}" "${BASE:?}"
MODELS="${MODELS:-}"
DISK_GB="${DISK_GB:-800}"
ITYPE="${ITYPE:-m7i.4xlarge}"
AWS=(aws --profile fuelos --region us-east-1)
S3_CODE="s3://pg-lineage-769063704696/code/shackleton"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# ship the frozen code state (committed tree only — commit before building)
SHA=$(git -C "$ROOT" rev-parse --short HEAD)
# preflight gate (E18): refuse to ship an unstamped SHA
[ "$(cat "$HERE/.preflight_pass" 2>/dev/null)" = "$SHA" ] || {
  echo "BLOCKED: preflight has not passed at $SHA — run: bash run_on_ubuntu.sh preflight.sh"; exit 4; }
# CODE_PATHS: tracked paths outside infra/ that the image's code layer and its gate need (N13: the scoring module under
# studies/reproductions/alwayson/code, the venv freeze, the offline gate's scripts and files)
if [ "${WORKTREE:-0}" = 1 ]; then
  # ship the WORKING TREE (the AWS lane's package_code convention) — for a deps-only image whose
  # code layer is overwritten by the rent-time code tar anyway; the tar name records it as -dirty
  SHA="${SHA}-dirty"
  tar czf "/tmp/repo_${SHA}.tar.gz" -C "$ROOT" infra ${CODE_PATHS:-}
else
  git -C "$ROOT" archive --format=tar.gz HEAD infra ${CODE_PATHS:-} > "/tmp/repo_${SHA}.tar.gz"
fi
"${AWS[@]}" s3 cp "/tmp/repo_${SHA}.tar.gz" "$S3_CODE/repo_${SHA}.tar.gz"

AKID=$(aws configure get aws_access_key_id --profile fuelos)
ASEC=$(aws configure get aws_secret_access_key --profile fuelos)
# HF token (only used when MODELS is non-empty): the ONLY live token is SSM /pg-lineage/hf-token
# (memory: reference_infrastructure) — the same source controller.py ships at rent time.
HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$MODELS" ] && [ -z "$HF_TOKEN" ]; then
  HF_TOKEN=$("${AWS[@]}" ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
    --query Parameter.Value --output text)
fi
AMI=$("${AWS[@]}" ssm get-parameter --name \
  /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)

UD=$(cat <<EOF
#!/bin/bash
set -euo pipefail
exec > /var/log/shackleton_build.log 2>&1
# tracing OFF for the credential lines: this log is uploaded to S3 (the
# pre-AMUNDSEN build logs contain the secret key and the HF token because it
# was not)
export AWS_ACCESS_KEY_ID=$AKID AWS_SECRET_ACCESS_KEY=$ASEC AWS_DEFAULT_REGION=us-east-1 HF_TOKEN=$HF_TOKEN
set -x
apt-get update && apt-get install -y docker.io docker-buildx python3-pip python3-venv unzip
curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip
unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install
mkdir -p /build && cd /build
aws s3 cp "$S3_CODE/repo_${SHA}.tar.gz" repo.tar.gz && tar xzf repo.tar.gz
cd infra/vast
# incremental observability (D5): stream the build log to S3 every 2 minutes
( while true; do aws s3 cp /var/log/shackleton_build.log "$S3_CODE/build_logs/${TAG}_progress.log" >/dev/null 2>&1 || true; sleep 120; done ) &
MODELS="$MODELS" TAG="$TAG" BASE="$BASE" BAKE_MODE="${BAKE_MODE:-dir}" PIP_PKGS="${PIP_PKGS:-}" PIN_ASSERT="${PIN_ASSERT:-}" \
  APT_PKGS="${APT_PKGS:-}" VENV_FREEZE="${VENV_FREEZE:-}" HF_CACHE_DEST="${HF_CACHE_DEST:-}" IMAGE_ENV="${IMAGE_ENV:-}" \
  GATE="${GATE:-}" N13_GATE_S3="${N13_GATE_S3:-}" \
  CODE_TAR="$S3_CODE/repo_${SHA}.tar.gz" bash build_image.sh \
  && STATUS=ok || STATUS=failed
aws s3 cp /var/log/shackleton_build.log "$S3_CODE/build_logs/${TAG}_\$STATUS.log"
shutdown -h now
EOF
)

"${AWS[@]}" ec2 run-instances \
  --image-id "$AMI" --instance-type "$ITYPE" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\",\"Throughput\":700,\"Iops\":6000,\"DeleteOnTermination\":true}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=shackleton-builder-$TAG}]" \
  --user-data "$UD" \
  --query 'Instances[0].InstanceId' --output text
echo "[builder] launched; it builds, pushes, uploads its log, and terminates itself."
echo "[builder] watch: aws --profile fuelos s3 ls $S3_CODE/build_logs/"
