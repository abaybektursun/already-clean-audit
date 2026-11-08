#!/usr/bin/env bash
# PLENUM execution queue — launches the remaining program in dependency order, self-pacing.
#
# Each entry retries until the G bucket has room AND capacity exists; a launch that cannot happen
# yet is not an error, it is backpressure. Nothing is skipped and nothing waits idle on a warm box.
# Stages are ordered so the law's inputs land first (§9: grid before Arm 4, Arms 2/3 in parallel).
set -u
cd "$(dirname "$0")/../.."
LOG=/tmp/plenum_queue.log
: > "$LOG"

try () {  # try <prefix> <module> <types> <args> <budget>
  local prefix="$1" mod="$2" types="$3" args="$4" budget="$5"
  # already launched or finished? never relaunch a prefix that has results
  if aws s3 ls "s3://pg-lineage-769063704696/$prefix/" --profile fuelos >/dev/null 2>&1; then
    echo "SKIP $prefix (already has results)" | tee -a "$LOG"; return 0
  fi
  for i in $(seq 1 200); do
    if ./infra/aws_pg/pg run "$mod" --type "$types" --no-weights --args "$args" \
         --s3-prefix "$prefix" --budget-min "$budget" --detach --json >>"$LOG" 2>&1; then
      echo "LAUNCHED $prefix" | tee -a "$LOG"; return 0
    fi
    sleep 150
  done
  echo "GAVE UP $prefix" | tee -a "$LOG"; return 1
}

T="g6e.2xlarge,g6e.4xlarge,g6e.xlarge,g6.2xlarge"
FC=infra.aws_pg.jobs.plenum_fc
A="ft-ai2d,ft-aokvqa-da,ft-aokvqa-mc,ft-coco35l,ft-cococap,ft-docvqa"
B="ft-gqa,ft-infovqa,ft-nlvr2,ft-ocrvqa,ft-okvqa,ft-refcoco-seg"
C="ft-rsvqa-hr,ft-rsvqa-lr,ft-scicap,ft-science-qa,ft-screen2words,ft-stvqa"
D="ft-tallyqa,ft-textcaps,ft-textvqa,ft-vizwizvqa,ft-vqav2,ft-widgetcap"
BAND="pg1,matched_dw"

# 1. the MDE on GENERATION (grounding's forced-choice column VOIDed on the N1 power gate)
try results/plenum_mde_gen "$FC" "$T" \
  "--units $BAND,ft-vqav2,ft-refcoco-seg --tasks refcoco,vqav2 --n 1500 --readout generation" 420

# 2. the full grid, GENERATION readout — fills behavior/*/generation for every unit and supplies
#    the law's outcome on any faculty whose forced-choice cell is VOID
for s in a:"$BAND,$A" b:"$B" c:"$C" d:"$D"; do
  k=${s%%:*}; u=${s#*:}
  try "results/plenum_gen_$k" "$FC" "$T" "--units $u --readout generation" 1400
done

# 3. the full grid, FORCED CHOICE — the primary family on the faculties where it has power
for s in a:"$BAND,$A" b:"$B" c:"$C" d:"$D"; do
  k=${s%%:*}; u=${s#*:}
  try "results/plenum_fcg_$k" "$FC" "$T" "--units $u" 1400
done

# 4. policies + pre-points, both readouts
try results/plenum_gen_pol "$FC" "$T" \
  "--units stitch0,rand_dec,pi0,pi05,u3,u4 --readout generation" 1000
try results/plenum_fcg_pol "$FC" "$T" "--units stitch0,rand_dec,pi0,pi05,u3,u4" 1000

# 5. Arm 4 (rides the Arm-1 rows) and Arm 2 (independent queue)
try results/plenum_patch infra.aws_pg.jobs.plenum_patch "$T" \
  "--n 250 --arms p5,locus,embed_swap,proj_swap" 900
try results/plenum_sae infra.aws_pg.jobs.plenum_sae "$T" \
  "--streams pg1_mm,pg1_text,pi05_mm,pi0_mm --sites 9,14" 700

echo "QUEUE COMPLETE" | tee -a "$LOG"
