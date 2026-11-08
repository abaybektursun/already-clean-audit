#!/usr/bin/env bash
# PLENUM Arm 3 — the fleet, launched once the bf16 AdamW feasibility gate passes.
#
# Three runs, all bf16 AdamW on an L40S (the fp32 formulation needed 80GB and AWS has never supplied
# it; bf16 holds params 6 + grads 6 + two moments 12 ~= 24GB and fits a 44GB card):
#
#   U1  --calibrate   the frozen budget procedure (smallest 2^k with pc_success >= 0.5 on a 3-task
#                     probe, cap 16k -> VOID-budget). Dual purpose: it also writes pi0's film
#                     checkpoint grid, and its earliest checkpoint is the feature-adapted step-0
#                     policy that pi0's control cell needs (the published config cannot be paired
#                     with the LIBERO env -- plenum_ctl_a proved that).
#   U3  --steps N     U3's film checkpoints + its control clock.
#   U3c --steps N     the causal recipe intervention. Publishes the u3c unit to S3, which unblocks
#                     24 of u3c's 25 map cells (the battery jobs read it from there).
#
# ON THE BUDGET. Every Arm-3 savings number is stated at {B*/4, B*}, and B* comes from the
# calibration -- which takes the better part of a day. Serializing U3/U3c behind it would leave the
# 24 u3c cells unmeasured for no scientific gain, because §2.3 already pre-registers the answer:
# recipe-causality is read "at matched control competence (primary) and matched budget (fallback,
# disclosed)". Running U3 and U3c at the SAME step count IS the matched-budget arm. N=4096 is on the
# calibration ladder (2^12), so if B* lands there the two coincide; otherwise the matched-budget
# reading is the one we report, labelled.
#
#   ./infra/aws_pg/plenum_arm3_fleet.sh [N_STEPS]
set -eu
cd "$(dirname "$0")/../.."
N="${1:-4096}"
D="--dtype bf16 --optim adamw --batch-size 1 --vl-batch-size 1"

launch () {  # launch <s3-suffix> <args> <budget-min>
  local suffix="$1"; shift
  local jargs="$1"; shift
  local budget="$1"; shift
  for i in $(seq 1 240); do
    if ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_a3 --type g6e.2xlarge --no-weights \
        --args "$jargs" --s3-prefix "results/$suffix" --budget-min "$budget" --detach --json \
        >> "/tmp/arm3_$suffix.log" 2>&1; then
      echo "  launched $suffix"; return 0
    fi
    sleep 180        # G-family quota is 32 vCPU and lags terminations by minutes
  done
  echo "  GAVE UP on $suffix after the retry window — quota never freed" >&2
  return 1
}

echo "[arm3] U3c co-training (the causal intervention; publishes the u3c unit)"
launch plenum_a3_u3c "--unit U3c --steps $N --film-points 8 $D" 900 &
sleep 20
echo "[arm3] U3 control fine-tune (film checkpoints + control clock)"
launch plenum_a3_u3 "--unit U3 --steps $N --film-points 8 $D" 900 &
sleep 20
echo "[arm3] U1 budget calibration (B*, pi0 film grid, pi0 step-0 control policy)"
launch plenum_a3_u1calib "--unit U1 --calibrate $D" 1200 &
wait
echo "[arm3] all three queued; poll out/metrics.json (never job.log — it is ~4MB of weight-sync spam)"
