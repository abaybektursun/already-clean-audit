#!/usr/bin/env bash
# DEPTHOPEN — the whole run, in the order SPEC §9 sets, as commands rather than prose.
#
# Nothing here loops or waits: each block is launched, watched through `metrics.json`, and the next block
# reads the previous one's banked numbers. The order is not cosmetic — §5 freezes the decision rules after
# G-smoke, G-grad and the reference R5 row, and before any arm's R2.
#
# Usage: read it, run one block at a time. `pg run --detach` returns the instance id; poll
#   aws s3 cp s3://pg-lineage-769063704696/results/depthopen/<prefix>/out/metrics.json - | jq .stages
# and never `job.log` (skill gotcha: the log carries megabytes of transfer progress).
set -euo pipefail
export AWS_PROFILE=fuelos
PG=./infra/aws_pg/pg
BASE=models--allenai--MolmoAct2-Think
TUNED=models--allenai--MolmoAct2-Think-LIBERO
DEPTH=models--allenai--MolmoAct-7B-D-0812
ARMS_S3=s3://pg-lineage-769063704696/depthopen/arms/v1
SAMPLE_S3=s3://pg-lineage-769063704696/depthopen/r5_sample/v1
SEED=20260909

# ---- 1. substrate: the subset and its depth companions (done: results/depthopen/data/1) --------
# 379 files enumerated, 373 file_index values off by one, 91 files / 395 episodes selected.
: <<'DONE'
$PG run infra.aws_pg.jobs.depthopen_data --type g5.xlarge --weights $DEPTH \
   --budget-min 180 --s3-prefix results/depthopen/data/1 --args " --per-suite-episodes 100" --detach
DONE

# ---- 2. G-smoke, G-grad and D-6 on A3 ----------------------------------------------------------
# Sets the arms' device_batch_size from measured memory, and D-6 checks the companions against the
# released fine-tune's own predicted grid before any arm trains.
: <<'NEXT'
$PG run infra.aws_pg.jobs.depthopen_smoke --type g6e.2xlarge --weights $BASE,$TUNED,$DEPTH \
   --budget-min 240 --s3-prefix results/depthopen/smoke/1 \
   --args " --probe-batches 1,2,4 --smoke-steps 50" --detach
NEXT

# ---- 3. the reference R5 row, before the freeze ------------------------------------------------
# The released checkpoint on the 200-task camera sample TWICE (§5's run-to-run flip rate) and on the
# layout sample once. The first run draws and uploads the sample; the others read it.
: <<'NEXT'
$PG run infra.aws_pg.jobs.depthopen_r5 --type g6e.2xlarge --weights $TUNED,$DEPTH --budget-min 300 \
   --s3-prefix results/depthopen/r5/released/camera/1 \
   --args " --model-name released --dimension camera" --detach
# then, with the sample banked:
$PG run infra.aws_pg.jobs.depthopen_r5 --type g6e.2xlarge --weights $TUNED,$DEPTH --budget-min 300 \
   --s3-prefix results/depthopen/r5/released/camera/2 \
   --args " --model-name released --dimension camera --sample-s3 $SAMPLE_S3" --detach
$PG run infra.aws_pg.jobs.depthopen_r5 --type g6e.2xlarge --weights $TUNED,$DEPTH --budget-min 300 \
   --s3-prefix results/depthopen/r5/released/layout/1 \
   --args " --model-name released --dimension layout --sample-s3 $SAMPLE_S3" --detach
NEXT

# ---- 4. FREEZE --------------------------------------------------------------------------------
# git tag depthopen-freeze-1 after §6's rules are in the tree and the three blocks above have landed.
# The rules are already code (infra/pg/depthopen_rules.py) and already gated on planted numbers.

# ---- 5. the arms: A0 and A3 first (§9 step 6) --------------------------------------------------
# --steps comes from the smoke's seconds_per_step, sized for about 1.2 h of training per arm.
: <<'NEXT'
for ARM in A0 A3; do
  $PG run infra.aws_pg.jobs.depthopen_arm --type g6e.2xlarge --weights $BASE,$DEPTH \
     --budget-min 300 --s3-prefix results/depthopen/arm/$ARM/1 \
     --args " --arm $ARM --steps <STEPS> --device-batch <BATCH> --seed $SEED" --detach
done
NEXT

# ---- 6. R4 per arm, then R5 on A0 and A3 ------------------------------------------------------
: <<'NEXT'
for ARM in A0 A3; do
  $PG run infra.aws_pg.jobs.depthopen_rollout --type g6e.2xlarge --weights $DEPTH --budget-min 360 \
     --s3-prefix results/depthopen/r4/$ARM/1 \
     --args " --arm $ARM --hf-s3 $ARMS_S3/$ARM/seed$SEED/hf" --detach
  for DIM in camera layout; do
    $PG run infra.aws_pg.jobs.depthopen_r5 --type g6e.2xlarge --weights $DEPTH --budget-min 300 \
       --s3-prefix results/depthopen/r5/$ARM/$DIM/1 \
       --args " --model-name $ARM --hf-s3 $ARMS_S3/$ARM/seed$SEED/hf --dimension $DIM --sample-s3 $SAMPLE_S3" --detach
  done
done
NEXT

# ---- 7. WHAT ACTUALLY RAN, and what did not --------------------------------------------------
# A4 ran. A1 and A2 were cut in amendment A4's pre-registered order once g6e capacity became the binding
# constraint: they separate the two ingredients only if A0 and A3 differ, and they do not. R5 on the arms
# was cancelled with H_better when G-known returned 0 of 10 — see CLOSEOUT.md. The released checkpoint's
# camera row ran once rather than twice, so the run-to-run flip rate was never banked; rule 3, its only
# consumer, never fired.
#
# ---- the original plan, kept for the record ---------------------------------------------------
: <<'NEXT'
for ARM in A4 A1 A2; do
  $PG run infra.aws_pg.jobs.depthopen_arm --type g6e.2xlarge --weights $BASE,$DEPTH \
     --budget-min 300 --s3-prefix results/depthopen/arm/$ARM/1 \
     --args " --arm $ARM --steps <STEPS> --device-batch <BATCH> --seed $SEED" --detach
  $PG run infra.aws_pg.jobs.depthopen_rollout --type g6e.2xlarge --weights $DEPTH --budget-min 360 \
     --s3-prefix results/depthopen/r4/$ARM/1 \
     --args " --arm $ARM --hf-s3 $ARMS_S3/$ARM/seed$SEED/hf" --detach
done
NEXT

# ---- 8. fill the grid, draw the figures, then a different model reads the artifacts ------------
: <<'NEXT'
aws s3 sync s3://pg-lineage-769063704696/results/depthopen/ studies/depthopen/results/ \
   --exclude '*job.log' --exclude '*heartbeat*'
.pgvenv/bin/python scripts/depthopen_report.py --results studies/depthopen/results
NEXT
