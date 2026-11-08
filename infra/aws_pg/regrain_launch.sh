#!/usr/bin/env bash
# REGRAIN launch helper — the ONE way this program starts a pg job.
#
# Why it exists: `./pg run ... | tail -4` swallows the launcher's exit code (a bash pipeline's rc
# is the LAST command's rc), so a RuntimeError like `VcpuLimitExceeded` / `InsufficientInstance
# Capacity` reads as success. That happened once in this program (the Arm-C sealed pass silently
# never launched). This wrapper never pipes, always clears the stale STATUS marker first, and
# fails loud.
#
#   ./infra/aws_pg/regrain_launch.sh <prefix> <module> <budget-min> [type-list] [-- args...]
set -euo pipefail

PREFIX="$1"; MODULE="$2"; BUDGET="$3"
TYPES="${4:-g6e.8xlarge,p5.4xlarge}"
shift 4 || shift 3
[ "${1:-}" = "--" ] && shift || true
ARGS="${*:-}"

BUCKET="s3://pg-lineage-769063704696"
DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "[regrain] clearing stale STATUS for results/${PREFIX}"
aws s3 rm "${BUCKET}/results/${PREFIX}/STATUS" --profile fuelos --region us-east-1 >/dev/null 2>&1 || true

echo "[regrain] launching ${MODULE} -> results/${PREFIX} (${TYPES}, ${BUDGET}min)"
if [ -n "$ARGS" ]; then
  "${DIR}/infra/aws_pg/pg" run "$MODULE" --type "$TYPES" --no-weights \
      --s3-prefix "results/${PREFIX}" --budget-min "$BUDGET" --args "$ARGS" --detach
else
  "${DIR}/infra/aws_pg/pg" run "$MODULE" --type "$TYPES" --no-weights \
      --s3-prefix "results/${PREFIX}" --budget-min "$BUDGET" --detach
fi
echo "[regrain] launch OK (rc=0) — watch with: ./infra/aws_pg/pg watch results/${PREFIX}"
