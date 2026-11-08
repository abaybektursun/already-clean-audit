#!/usr/bin/env bash
# REGRAIN CPU driver — waits for the GPU pipelines' artifacts, then runs the CPU-only arms and the
# final figure suite. Runs on the Standard (r-family) quota, so it never contends with the GPU
# driver for the G bucket. Fail-loud per job; a red arm does not stop the others (each is
# independent), but every failure is printed and the driver exits non-zero.
set -uo pipefail
cd "$(dirname "$0")/../.."
PG=./infra/aws_pg/pg
B=s3://pg-lineage-769063704696
RC=0

wait_for() {   # wait_for <s3-key> <label>
  echo "[cpu] waiting for $2 ($1)"
  until aws s3 ls "$1" --profile fuelos --region us-east-1 >/dev/null 2>&1; do sleep 120; done
  echo "[cpu] $2 present"
}

# A job's JSON artifact exists from its FIRST write, long before it is finished (every REGRAIN job
# checkpoints incrementally). Waiting on file existence therefore starts the consumer against a
# half-written producer. Wait on the producer's own `green` flag instead.
wait_for_green() {   # wait_for_green <s3-json> <label>
  # Producers write results incrementally, so file-existence is NOT completion: `green` is.
  # But `green` alone is not health either -- a job can write green and then die on a later
  # statement. So poll BOTH: green => done; a nonzero STATUS => fail loud, never proceed.
  local base="${1%%/out/*}"
  echo "[cpu] waiting for $2 to go GREEN ($1)"
  while true; do
    if aws s3 cp "$base/STATUS" /tmp/_status.txt --profile fuelos --region us-east-1 >/dev/null 2>&1 \
       && ! grep -q 'exit=0' /tmp/_status.txt; then
      echo "[cpu] FATAL: $2 exited nonzero ($(cat /tmp/_status.txt)) -- halting driver" >&2
      exit 1
    fi
    if aws s3 cp "$1" /tmp/_green.json --profile fuelos --region us-east-1 >/dev/null 2>&1 \
       && python3 -c "import json,sys; sys.exit(0 if json.load(open('/tmp/_green.json')).get('green') else 1)" 2>/dev/null; then
      echo "[cpu] $2 GREEN"; return 0
    fi
    sleep 120
  done
}

run_job() {    # run_job <prefix> <module> <budget> <args>
  local prefix="$1" module="$2" budget="$3" args="$4"
  aws s3 rm "${B}/results/${prefix}/STATUS" --profile fuelos --region us-east-1 >/dev/null 2>&1 || true
  for i in $(seq 1 20); do
    if $PG run "$module" --type r7i.8xlarge,r7i.4xlarge --no-weights \
         --s3-prefix "results/${prefix}" --budget-min "$budget" --args "$args" --detach; then
      $PG watch "results/${prefix}" || { echo "[cpu] $prefix RED"; RC=1; }
      echo "[cpu] === $prefix done ==="; return 0
    fi
    echo "[cpu] $prefix: quota busy (attempt $i), sleeping 90s"; sleep 90
  done
  echo "[cpu] $prefix: FAILED to launch"; RC=1
}

HARVEST="${B}/results/regrain_pipeA/out/harvest"
FACT="${B}/results/regrain_pipeA/out/i_factorial"

# Arm D needs only the 5 S_vl draws from the harvest step (skip if already green)
if ! (aws s3 cp "${B}/results/regrain_d/out/metrics.json" /tmp/_d.json --profile fuelos --region us-east-1 >/dev/null 2>&1 \
      && python3 -c "import json,sys; sys.exit(0 if json.load(open('/tmp/_d.json')).get('green') else 1)" 2>/dev/null); then
  wait_for "${HARVEST}/svl_draws5.pt" "harvest svl_draws5"
  run_job regrain_d infra.aws_pg.jobs.regrain_d_draws 120 "--harvest-s3 ${HARVEST}"
else
  echo "[cpu] regrain_d already GREEN — skipping"
fi

# Arm E (atlas + C3 geometry) needs the factorial's banks AND its gates to be final
wait_for_green "${FACT}/gates.json" "i_factorial"
run_job regrain_e infra.aws_pg.jobs.regrain_e_atlas 180 \
  "--harvest-s3 ${HARVEST} --cdisc-s3 ${B}/results/regrain_c_disc/out --factorial-s3 ${FACT}"

echo "[cpu] CPU ARMS COMPLETE (rc=$RC)"
exit $RC
