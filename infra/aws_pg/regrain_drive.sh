#!/usr/bin/env bash
# REGRAIN GPU driver — launches the remaining GPU work in dependency order, retrying past the
# G-quota drain window (a self-terminating g6e.8xlarge holds its 32 vCPU for 3-7 minutes) and
# watching each job to completion before starting the next. Never pipes a `pg run` (its exit code
# is the launch gate). Fail-loud: a red job stops the chain.
set -uo pipefail
cd "$(dirname "$0")/../.."
PG=./infra/aws_pg/pg
B=s3://pg-lineage-769063704696

launch_retry() {   # launch_retry <prefix> <args...>
  local prefix="$1"; shift
  aws s3 rm "${B}/results/${prefix}/STATUS" --profile fuelos --region us-east-1 >/dev/null 2>&1 || true
  for i in $(seq 1 25); do
    if $PG run "$@" --detach; then echo "[drive] ${prefix}: launched (attempt $i)"; return 0; fi
    echo "[drive] ${prefix}: quota/capacity busy (attempt $i), sleeping 60s"; sleep 60
  done
  echo "[drive] ${prefix}: FAILED to launch after 25 attempts"; return 1
}

watch_job() {      # watch_job <prefix>
  echo "[drive] watching results/$1"
  $PG watch "results/$1"
  local rc=$?
  echo "[drive] results/$1 watch rc=$rc"
  return $rc
}

step() {           # step <prefix> <pg-run-args...>
  local prefix="$1"
  launch_retry "$@" || return 1
  watch_job "$prefix" || return 1
  echo "[drive] === $prefix COMPLETE ==="
}

# 1) the energy ladder (decisive follow-up to the sealed MDE)
step regrain_c_energy infra.aws_pg.jobs.regrain_c_energy \
  --type g6e.8xlarge,p5.4xlarge --no-weights --s3-prefix results/regrain_c_energy \
  --budget-min 150 --args "--cdisc-s3 ${B}/results/regrain_c_disc/out" || exit 1

# 2) pipeline A: harvest -> factorial -> frontier (sequenced on EC2)
step regrain_pipeA infra.aws_pg.jobs.pipeline \
  --type g6e.8xlarge,p5.4xlarge --no-weights --s3-prefix results/regrain_pipeA \
  --budget-min 480 --args "--steps-s3 ${B}/results/regrain_pipeA/steps.json" || exit 1

# 3) pipeline B: h_stream -> b_blocks -> f_steer
step regrain_pipeB infra.aws_pg.jobs.pipeline \
  --type g6e.8xlarge,p5.4xlarge --no-weights --s3-prefix results/regrain_pipeB \
  --budget-min 480 --args "--steps-s3 ${B}/results/regrain_pipeB/steps.json" || exit 1

echo "[drive] ALL GPU WORK COMPLETE"
