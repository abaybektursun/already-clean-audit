#!/usr/bin/env bash
# DEPTHOPEN — launch arms with the step count and batch the smoke MEASURED, not a guess.
#
#   bash infra/aws_pg/depthopen_launch_arms.sh <smoke-prefix> <target-minutes> <arm> [<arm> ...]
#   bash infra/aws_pg/depthopen_launch_arms.sh results/depthopen/smoke/3 70 A0 A3
#
# Reads `stages.g_smoke.seconds_per_step` and `device_batch_size` from the smoke cell's banked metrics,
# solves for the step count that fills the target training time, and launches one job per arm with the
# same steps, batch and seed — §2's "same steps, same seed" is enforced here rather than remembered.
# Every arm gets its own S3 prefix and its own capacity retry loop, because g6e capacity comes and goes.
# NO `set -e` around the launches: `pg run` exits non-zero when g6e capacity is short, and with -e the
# assignment `OUT=$(pg run ...)` killed this script silently on the first shortage.
set -uo pipefail
export AWS_PROFILE=fuelos
BUCKET=pg-lineage-769063704696
PREFIX=${1:?smoke prefix, e.g. results/depthopen/smoke/3}
TARGET_MIN=${2:?target training minutes per arm}
shift 2
ARMS=("$@")
SEED=20260909

read -r SPS BATCH <<EOF
$(aws s3 cp "s3://$BUCKET/$PREFIX/out/metrics.json" - --region us-east-1 | python3 -c '
import json, sys
d = json.load(sys.stdin)
g = (d.get("verdict") or {}).get("g_smoke") or (d.get("stages") or {}).get("g_smoke") or {}
sps = g.get("seconds_per_step")
batch = g.get("device_batch_size")
if not sps or not batch:
    raise SystemExit("the smoke cell has no measured seconds_per_step / device_batch_size")
print(sps, batch)')
EOF
STEPS=$(python3 -c "print(int(round($TARGET_MIN * 60 / $SPS / 50.0) * 50))")
SAMPLES=$(python3 -c "print($STEPS * $BATCH)")
echo "measured ${SPS}s/step at device_batch=${BATCH} -> ${STEPS} steps (~${TARGET_MIN} min, ${SAMPLES} samples per arm)"

for ARM in "${ARMS[@]}"; do
  for attempt in $(seq 1 30); do
    OUT=$(./infra/aws_pg/pg run infra.aws_pg.jobs.depthopen_arm \
        --type g6e.2xlarge,g6e.4xlarge \
        --weights models--allenai--MolmoAct2-Think,models--allenai--MolmoAct-7B-D-0812 \
        --budget-min 300 --s3-prefix "results/depthopen/arm/$ARM/1" \
        --args " --arm $ARM --steps $STEPS --device-batch $BATCH --seed $SEED" --detach 2>&1 | tail -2 || true)
    if printf '%s' "$OUT" | grep -q launched; then
      echo "$(date -u +%H:%M) $ARM $OUT"
      break
    fi
    echo "$(date -u +%H:%M) $ARM: no capacity (attempt $attempt)"
    sleep 120
  done
done
