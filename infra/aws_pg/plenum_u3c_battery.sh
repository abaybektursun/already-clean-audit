#!/usr/bin/env bash
# The u3c row: every battery the other 31 units were measured with, run on the published U3c unit.
#
# U3c's 25 map cells are not a new measurement — they are the EXISTING battery pointed at one more
# unit, which is the whole reason `plenum_units.install_unit` is the single entry point. Nothing in
# here is u3c-specific except the unit id.
#
# Waits for the unit to appear at weights/lacuna_units/U3c (published by plenum_a3 at seed 0's final
# checkpoint) rather than assuming it is there, because launching the battery against a missing unit
# would fail 15GB into a download.
#
#   ./infra/aws_pg/plenum_u3c_battery.sh
set -eu
cd "$(dirname "$0")/../.."
S3=s3://pg-lineage-769063704696
UNIT="$S3/weights/lacuna_units/U3c"

echo "[u3c] waiting for the unit to publish at $UNIT"
for i in $(seq 1 720); do
  if aws s3 ls "$UNIT/" --profile fuelos 2>/dev/null | grep -q safetensors; then
    echo "[u3c] unit present"; break
  fi
  sleep 120
done
aws s3 ls "$UNIT/" --profile fuelos 2>/dev/null | grep -q safetensors || {
  echo "[u3c] unit never published — NOT launching a battery against a missing unit" >&2; exit 1; }

launch () {  # launch <suffix> <module> <args> <budget> <type>
  local suffix="$1" mod="$2" jargs="$3" budget="$4" itype="$5"
  for i in $(seq 1 240); do
    if ./infra/aws_pg/pg run "$mod" --type "$itype" --no-weights \
         --args "$jargs" --s3-prefix "results/$suffix" --budget-min "$budget" --detach --json \
         >> "/tmp/u3cb_$suffix.log" 2>&1; then
      echo "  launched $suffix"; return 0
    fi
    sleep 180
  done
  echo "  GAVE UP on $suffix — quota never freed" >&2; return 1
}

# Forced choice is the PRIMARY readout family (§2.1); generation fills behavior/*/generation and is
# the readout for any faculty whose forced-choice cell VOIDs on N1 (grounding did).
launch plenum_fc_u3c   infra.aws_pg.jobs.plenum_fc \
  "--units u3c --readout forced_choice" 700 g6e.2xlarge &
sleep 15
launch plenum_gen_u3c  infra.aws_pg.jobs.plenum_fc \
  "--units u3c --readout generation" 700 g6e.2xlarge &
sleep 15
launch plenum_text_u3c infra.aws_pg.jobs.plenum_text \
  "--units u3c" 700 g6e.2xlarge &
sleep 15
# census is CPU-only (weights, no forward) so it does not compete for the G quota
launch plenum_census_u3c infra.aws_pg.jobs.plenum_census \
  "--units u3c" 300 r7i.4xlarge,r7i.2xlarge &
sleep 15
launch plenum_ctl_u3c  infra.aws_pg.jobs.plenum_control \
  "--units u3c" 300 g6e.2xlarge &
wait
echo "[u3c] battery queued; poll out/metrics.json (never job.log)"
