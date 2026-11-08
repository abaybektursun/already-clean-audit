#!/usr/bin/env bash
# Run a shackleton script on the Ubuntu box (abay@192.168.1.95) — the designated
# build/rehearsal machine. Never the Mac (tiny), never EC2 (quota shared with
# another lane). Docker layer cache and downloaded weights persist on the box,
# so rebuilds are cheap.
#
# Usage:
#   bash run_on_ubuntu.sh preflight.sh
#   MODELS="Qwen/Qwen2.5-14B" TAG=v1-14b BASE=vastai/pytorch:cuda-12.8.1-auto \
#     bash run_on_ubuntu.sh build_image.sh
set -euo pipefail
BOX=abay@192.168.1.95
SCRIPT="${1:?script name, e.g. rehearse.sh or build_image.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
S3_CODE="s3://pg-lineage-769063704696/code/shackleton"

SHA=$(git -C "$ROOT" rev-parse --short HEAD)
git -C "$ROOT" archive --format=tar.gz HEAD infra > "/tmp/repo_${SHA}.tar.gz"
aws --profile fuelos s3 cp "/tmp/repo_${SHA}.tar.gz" "$S3_CODE/repo_${SHA}.tar.gz" > /dev/null
scp -q "/tmp/repo_${SHA}.tar.gz" "$BOX:/tmp/repo_${SHA}.tar.gz"

AKID=$(aws configure get aws_access_key_id --profile fuelos)
ASEC=$(aws configure get aws_secret_access_key --profile fuelos)

# persistent per-purpose workdir: code re-untarred fresh every run, while
# infra/vast/build/ (downloaded weights, docker context) survives across runs —
# so a post-fix rebuild never re-downloads a model
PURPOSE="${TAG:-rehearsal}"
ssh "$BOX" "set -euo pipefail
  mkdir -p ~/shackleton/w_${PURPOSE}
  cd ~/shackleton/w_${PURPOSE}
  tar xzf /tmp/repo_${SHA}.tar.gz
  cd infra/vast
  env AWS_ACCESS_KEY_ID='$AKID' AWS_SECRET_ACCESS_KEY='$ASEC' \
      AWS_DEFAULT_REGION=us-east-1 \
      MODELS='${MODELS:-}' TAG='${TAG:-}' BASE='${BASE:-}' PF_SHA='$SHA' \
      CODE_TAR='$S3_CODE/repo_${SHA}.tar.gz' \
      bash $SCRIPT"

# preflight stamps prove box-validation to the local launch gates (E18)
if [ "$SCRIPT" = "preflight.sh" ]; then
  scp -q "$BOX:~/shackleton/w_${PURPOSE}/infra/vast/.preflight_pass" "$HERE/.preflight_pass"
  echo "[preflight] stamp $(cat "$HERE/.preflight_pass") pulled to local gate"
fi
