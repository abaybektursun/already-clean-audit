#!/usr/bin/env bash
# REGRAIN closeout — renders the FULL §14 figure suite against every landed artifact, with the
# closeout commit's SHA stamped into every provenance footer. Run AFTER all arms are green.
# Post-critic rule: if any verdict moves, re-run this (figures keep the pre-critic verdict).
set -uo pipefail
cd "$(dirname "$0")/../.."
PG=./infra/aws_pg/pg
B=s3://pg-lineage-769063704696/results
SHA=$(git rev-parse HEAD)

echo "[closeout] rendering figure suite at SHA ${SHA}"
aws s3 rm "${B}/regrain_figs/STATUS" --profile fuelos --region us-east-1 >/dev/null 2>&1 || true

ARGS="--sha ${SHA}"
add() { # add <flag> <s3-prefix> <sentinel-file>
  if aws s3 ls "$2/$3" --profile fuelos --region us-east-1 >/dev/null 2>&1; then
    ARGS="${ARGS} $1 $2"; echo "[closeout]   + $1 $2"
  else
    echo "[closeout]   - $1 ABSENT (panel will render an explicit SKIPPED placeholder)"
  fi
}
add --cdisc-s3    "${B}/regrain_c_disc/out"              c_disc.json
add --csealed-s3  "${B}/regrain_c_sealed/out"            metrics.json
add --ce-s3       "${B}/regrain_c_energy/out"            metrics.json
add --factorial-s3 "${B}/regrain_i_factorial/out"        gates.json
add --frontier-s3 "${B}/regrain_pipeA/out/i_frontier"    frontier.json
add --d-s3        "${B}/regrain_d/out"                   metrics.json
add --e-s3        "${B}/regrain_e/out"                   metrics.json
add --h-s3        "${B}/regrain_pipeB/out/h_stream"      metrics.json
add --b-s3        "${B}/regrain_pipeB/out/b_blocks"      metrics.json
add --f-s3        "${B}/regrain_pipeB/out/f_steer"       metrics.json
add --g-s3        "${B}/regrain_g/out"                   metrics.json
add --gallery-s3  "${B}/regrain_pipeB/out/gallery"       metrics.json

for i in $(seq 1 20); do
  if $PG run infra.aws_pg.jobs.regrain_figures --type r7i.8xlarge,r7i.4xlarge --no-weights \
       --s3-prefix results/regrain_figs --budget-min 60 --args "${ARGS}" --detach; then
    $PG watch results/regrain_figs || { echo "[closeout] figures RED"; exit 1; }
    break
  fi
  echo "[closeout] quota busy (attempt $i)"; sleep 60
done

echo "[closeout] pulling figures into the repo"
mkdir -p studies/lineage/paligemma1-pi0/research/figs
aws s3 sync "${B}/regrain_figs/out/" studies/lineage/paligemma1-pi0/research/figs/ \
    --exclude "*" --include "regrain_F_*.png" --include "regrain_F_*.pdf" \
    --profile fuelos --region us-east-1
aws s3 cp "${B}/regrain_figs/out/figure_registry.json" \
    studies/lineage/paligemma1-pi0/research/figs/regrain_figure_registry.json \
    --profile fuelos --region us-east-1

echo "[closeout] fleet check (must be empty)"
$PG ps
echo "[closeout] DONE — commit the figures + registry, then tag regrain-closeout-<date>"
