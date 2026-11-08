#!/usr/bin/env bash
# REGRAIN driver, phase 2. pipeA is complete (harvest + frontier valid; the factorial was re-run
# standalone after the Grassmann inference-threshold fix). This driver:
#   1. waits for the corrected i_factorial to go GREEN
#   2. runs pipeB (h_stream -> b_blocks -> f_steer -> gallery) on GPU
#   3. runs Arm E (atlas + C3 geometry) on CPU, against the CORRECTED factorial
#   4. renders the full figure suite
# Never pipes a `pg run` (its exit code is the launch gate). Waits on producers' `green` flags,
# never on file existence (every job checkpoints its JSON incrementally).
set -uo pipefail
cd "$(dirname "$0")/../.."
PG=./infra/aws_pg/pg
B=s3://pg-lineage-769063704696
RC=0

wait_green() {  # wait_green <s3-json> <label>
  # `green` => the producer finished its science. A nonzero STATUS => it died; fail loud.
  local base="${1%%/out/*}"

  echo "[d2] waiting for $2 to go GREEN"
  while true; do
    if aws s3 cp "$base/STATUS" /tmp/_s2.txt --profile fuelos --region us-east-1 >/dev/null 2>&1 \
       && ! grep -q 'exit=0' /tmp/_s2.txt; then
      echo "FATAL: $2 exited nonzero ($(cat /tmp/_s2.txt)) -- halting driver" >&2; exit 1
    fi
    if aws s3 cp "$1" /tmp/_g2.json --profile fuelos --region us-east-1 >/dev/null 2>&1 \
       && python3 -c "import json,sys;sys.exit(0 if json.load(open('/tmp/_g2.json')).get('green') else 1)" 2>/dev/null; then
      echo "[d2] $2 GREEN"; return 0
    fi
    sleep 120
  done
}

launch() {      # launch <prefix> <pg run args...>
  local prefix="$1"; shift
  aws s3 rm "${B}/results/${prefix}/STATUS" --profile fuelos --region us-east-1 >/dev/null 2>&1 || true
  for i in $(seq 1 30); do
    if $PG run "$@" --detach; then echo "[d2] ${prefix} launched (attempt $i)"; return 0; fi
    echo "[d2] ${prefix}: capacity/quota busy (attempt $i)"; sleep 60
  done
  echo "[d2] ${prefix}: FAILED to launch"; return 1
}

FACT="${B}/results/regrain_i_factorial/out"
HARVEST="${B}/results/regrain_pipeA/out/harvest"

wait_green "${FACT}/gates.json" "i_factorial (corrected)"

# ---- pipeB on GPU, pointing at the CORRECTED factorial ----
python3 - <<EOF
import json, pathlib
B = "s3://pg-lineage-769063704696"
steps = [
 {"name":"h_stream","module":"infra.aws_pg.jobs.regrain_h_stream",
  "args":f"--harvest-s3 {B}/results/regrain_pipeA/out/harvest --factorial-s3 {B}/results/regrain_i_factorial/out"},
 {"name":"b_blocks","module":"infra.aws_pg.jobs.regrain_b_blocks",
  "args":f"--factorial-s3 {B}/results/regrain_i_factorial/out --cdisc-s3 {B}/results/regrain_c_disc/out --harvest-s3 {B}/results/regrain_pipeA/out/harvest"},
 {"name":"f_steer","module":"infra.aws_pg.jobs.regrain_f_steer",
  "args":f"--cdisc-s3 {B}/results/regrain_c_disc/out --csealed-s3 {B}/results/regrain_c_sealed/out --harvest-s3 {B}/results/regrain_pipeA/out/harvest"},
 {"name":"gallery","module":"infra.aws_pg.jobs.regrain_gallery",
  "args":f"--cdisc-s3 {B}/results/regrain_c_disc/out"},
]
pathlib.Path("infra/aws_pg/steps/regrain_pipeB.json").write_text(json.dumps(steps, indent=1))
EOF
aws s3 cp infra/aws_pg/steps/regrain_pipeB.json "${B}/results/regrain_pipeB/steps.json" --profile fuelos --region us-east-1 >/dev/null

launch regrain_pipeB infra.aws_pg.jobs.pipeline --type g6e.8xlarge,p5.4xlarge --no-weights \
  --s3-prefix results/regrain_pipeB --budget-min 480 \
  --args "--steps-s3 ${B}/results/regrain_pipeB/steps.json" || RC=1
$PG watch results/regrain_pipeB || { echo "[d2] pipeB RED"; RC=1; }

# ---- Arm E on CPU against the corrected factorial (can overlap pipeB, different quota bucket) ----
launch regrain_e infra.aws_pg.jobs.regrain_e_atlas --type r7i.8xlarge,r7i.4xlarge --no-weights \
  --s3-prefix results/regrain_e --budget-min 180 \
  --args "--harvest-s3 ${HARVEST} --cdisc-s3 ${B}/results/regrain_c_disc/out --factorial-s3 ${FACT}" || RC=1
$PG watch results/regrain_e || { echo "[d2] Arm E RED"; RC=1; }

echo "[d2] ALL ARMS DONE (rc=$RC)"
exit $RC
