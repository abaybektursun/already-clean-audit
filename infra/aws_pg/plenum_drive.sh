#!/usr/bin/env bash
# PLENUM launch driver — shards the Arm-1 grid and the remaining arms across the G quota.
#
# The G bucket is 32 vCPU total and EVERY g6e size carries the same single L40S
# (reference_pg_gpu_quota_shape), so throughput scales with instance COUNT, not size: four
# g6e.2xlarge = four GPUs, where two g6e.4xlarge = two. The grid is therefore sharded into four.
# Each shard retries until capacity frees rather than failing — "never wait on a dry pool, never
# idle" means keep trying, not keep a box warm.
#
#   ./infra/aws_pg/plenum_drive.sh grid_a     # band rows (FLOOR+CEILING) + FT quarter 1
#   ./infra/aws_pg/plenum_drive.sh grid_b|grid_c|grid_d
#   ./infra/aws_pg/plenum_drive.sh policies   # pi0/pi05/u3/u4 + stitch0/rand_dec
#   ./infra/aws_pg/plenum_drive.sh arm4 | arm2
set -u
cd "$(dirname "$0")/../.."

# The 24-FT roster split into four, in roster (alphabetical) order — no cherry-picking.
FT_A="ft-ai2d,ft-aokvqa-da,ft-aokvqa-mc,ft-coco35l,ft-cococap,ft-docvqa"
FT_B="ft-gqa,ft-infovqa,ft-nlvr2,ft-ocrvqa,ft-okvqa,ft-refcoco-seg"
FT_C="ft-rsvqa-hr,ft-rsvqa-lr,ft-scicap,ft-science-qa,ft-screen2words,ft-stvqa"
FT_D="ft-tallyqa,ft-textcaps,ft-textvqa,ft-vizwizvqa,ft-vqav2,ft-widgetcap"

launch () {  # launch <module> <types> <args> <prefix> <budget>
  for i in $(seq 1 120); do
    if ./infra/aws_pg/pg run "$1" --type "$2" --no-weights --args "$3" \
         --s3-prefix "$4" --budget-min "$5" --detach --json 2>/dev/null; then
      echo "LAUNCHED $4"; return 0
    fi
    sleep 120
  done
  echo "GAVE UP launching $4 after 4h of retries"; return 1
}

case "${1:?usage: plenum_drive.sh <stage>}" in
  # Band rows travel with shard A so FLOOR/CEILING are measured on the same items as the units
  # scored against them; every shard shares the frozen item sample and choice seed, so the halves
  # are poolable by construction.
  grid_a)
    launch infra.aws_pg.jobs.plenum_fc g6e.2xlarge,g6e.4xlarge \
      "--units pg1,matched_dw,$FT_A" results/plenum_grid_a 1400 ;;
  grid_b)
    launch infra.aws_pg.jobs.plenum_fc g6e.2xlarge,g6e.4xlarge "--units $FT_B" results/plenum_grid_b 1400 ;;
  grid_c)
    launch infra.aws_pg.jobs.plenum_fc g6e.2xlarge,g6e.4xlarge "--units $FT_C" results/plenum_grid_c 1400 ;;
  grid_d)
    launch infra.aws_pg.jobs.plenum_fc g6e.2xlarge,g6e.4xlarge "--units $FT_D" results/plenum_grid_d 1400 ;;
  policies)
    launch infra.aws_pg.jobs.plenum_fc g6e.2xlarge,g6e.4xlarge \
      "--units stitch0,rand_dec,pi0,pi05,u3,u4" results/plenum_grid_pol 1000 ;;
  arm4)
    launch infra.aws_pg.jobs.plenum_patch g6e.2xlarge,g6e.4xlarge \
      "--n 250 --arms p5,locus,embed_swap,proj_swap" results/plenum_patch 900 ;;
  arm2)
    launch infra.aws_pg.jobs.plenum_sae g6e.2xlarge,g6e.4xlarge \
      "--streams pg1_mm,pg1_text,pi05_mm,pi0_mm --sites 9,14" results/plenum_sae 700 ;;
  *) echo "unknown stage $1"; exit 2 ;;
esac
