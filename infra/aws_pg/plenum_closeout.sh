#!/usr/bin/env bash
# PLENUM closeout — pull every banked shard, aggregate, render the map, draw the figures.
#
# Runs on the control plane (no GPU): everything here is a pure function of banked JSON, which is
# the point of banking per-item scores in the first place. Re-runnable at any time — it reports the
# CURRENT pending count rather than assuming the run is finished, so it is also the progress meter.
#
#   ./infra/aws_pg/plenum_closeout.sh [half]        # half: confirmation (default) | discovery | full
set -eu
cd "$(dirname "$0")/../.."
HALF="${1:-confirmation}"
B=s3://pg-lineage-769063704696/results
W=/tmp/plenum_closeout
mkdir -p "$W/banks"

fetch () {  # fetch <prefix> <local-name> — returns NON-ZERO if the prefix has not landed yet.
  # The return code is load-bearing: the caller only adds a bank to the aggregate if it exists.
  # An `if` wrapper here would always succeed and hand the aggregate a path to a missing file.
  if aws s3 cp "$B/$1/out/fc_banks.json" "$W/banks/$2.json" --profile fuelos >/dev/null 2>&1; then
    echo "  + $1"; return 0
  fi
  return 1
}

echo "[closeout] fetching banks"
FC=""; GEN=""
for k in a b c d pol; do
  if fetch "plenum_fcg_$k" "fc_$k"; then FC="$FC $W/banks/fc_$k.json"; fi
  if fetch "plenum_gen_$k" "gen_$k"; then GEN="$GEN $W/banks/gen_$k.json"; fi
done
# caption-only re-runs (the original shards' caption banks are VOID under bank misalignment)
for k in a b c d; do
  if fetch "plenum_fcg_cap_$k" "fc_cap_$k"; then FC="$FC $W/banks/fc_cap_$k.json"; fi
done
# shard D's units are launched ONE UNIT PER JOB (a 6-unit forced-choice shard at frozen density
# gets reaped mid-run with exit=0 -- see the journal entry on completion signals)
for u in ft-tallyqa ft-textcaps ft-textvqa ft-vizwizvqa ft-vqav2 ft-widgetcap; do
  if fetch "plenum_fcgu_$u" "fc_u_$u"; then FC="$FC $W/banks/fc_u_$u.json"; fi
done
# policies are launched ONE UNIT PER JOB: four 13-16GB state dicts in one process is an OOM kill
for u in stitch0 rand_dec pi0 pi05 u3 u4; do
  if fetch "plenum_gen_pol_$u" "gen_pol_$u"; then GEN="$GEN $W/banks/gen_pol_$u.json"; fi
  if fetch "plenum_fcg_pol_$u" "fc_pol_$u"; then FC="$FC $W/banks/fc_pol_$u.json"; fi
done
if [ -z "$FC$GEN" ]; then echo "[closeout] no grid banks have landed yet"; exit 0; fi

# The MDE is NOT pooled with the grid. It runs at n=1500 while the grid runs at the frozen
# per-task density (RefCOCO 1975), so the two drew DIFFERENT item samples — the aggregate's
# poolability assertion catches this, and the right response is to respect it rather than to
# relax the guard. The MDE is its own pre-registered analysis (§3.1) with its own band rows, so
# it is aggregated separately and reported separately.
MFC=""; MGEN=""
if fetch plenum_mde4 fc_mde; then MFC="--fc $W/banks/fc_mde.json"; fi
if fetch plenum_mde_gen2 gen_mde; then MGEN="--gen $W/banks/gen_mde.json"; fi
if [ -n "$MFC$MGEN" ]; then
  echo "[closeout] aggregating the MDE separately (n=1500 sample)"
  python3 -m infra.aws_pg.jobs.plenum_aggregate --out "$W/mde" --half "$HALF" $MFC $MGEN \
    --pointer "s3://pg-lineage-769063704696/results/plenum_mde4" || true
fi

# EVERY census shard, not one: the aggregator merges them (C6 -- passing only the FT census
# silently dropped pi0/pi05). `plenum_census_exp` carries the action-expert absolute census and
# PG1's own self-reference row, which no earlier shard had.
CEN=""
for c in plenum_census plenum_census2 plenum_census3 plenum_census_syn plenum_census_exp; do
  if aws s3 cp "$B/$c/out/metrics.json" "$W/cen_$c.json" --profile fuelos >/dev/null 2>&1; then
    CEN="$CEN $W/cen_$c.json"; echo "  + census $c"
  fi
done

# CONTROL shards (§2.3 step-0 cells).
CTL=""
for c in plenum_ctl_a plenum_ctl_b plenum_ctl_b2 plenum_ctl_u3c plenum_ctl_film; do
  if aws s3 cp "$B/$c/out/metrics.json" "$W/ctl_$c.json" --profile fuelos >/dev/null 2>&1; then
    CTL="$CTL $W/ctl_$c.json"; echo "  + control $c"
  fi
done

# SAE shards (Arm 2 feature cells, incl. the gemma-2b text stream).
SAE=""
for c in plenum_sae plenum_sae2 plenum_sae_g2b2; do
  if aws s3 cp "$B/$c/out/metrics.json" "$W/sae_$c.json" --profile fuelos >/dev/null 2>&1; then
    SAE="$SAE $W/sae_$c.json"; echo "  + sae $c"
  fi
done
aws s3 cp "$B/plenum_freeze2/out/plenum_predictions.json" "$W/predictions.json" \
  --profile fuelos >/dev/null 2>&1 || true

echo "[closeout] aggregating (half=$HALF)"
ARGS=""
[ -n "$FC" ] && ARGS="$ARGS --fc$FC"
[ -n "$GEN" ] && ARGS="$ARGS --gen$GEN"
[ -n "$CEN" ] && ARGS="$ARGS --census$CEN"
[ -n "$CTL" ] && ARGS="$ARGS --control$CTL"
[ -n "$SAE" ] && ARGS="$ARGS --sae$SAE"
[ -f "$W/predictions.json" ] && ARGS="$ARGS --predictions $W/predictions.json"
python3 -m infra.aws_pg.jobs.plenum_aggregate --out "$W/agg" --half "$HALF" $ARGS

echo "[closeout] rendering MAP"
python3 scripts/plenum_map_render.py --records "$W/agg/cells.json" \
  --out studies/lineage/MAP.md

echo "[closeout] figures"
PATCH=""
aws s3 cp "$B/plenum_patch/out/metrics.json" "$W/patch.json" --profile fuelos >/dev/null 2>&1 \
  && PATCH="--patch $W/patch.json"
python3 -m infra.aws_pg.jobs.plenum_figures --out studies/lineage/paligemma1-pi0/research/plenum/figs \
  --agg "$W/agg/metrics.json" --cells "$W/agg/cells.json" \
  ${PATCH} $([ -f "$W/predictions.json" ] && echo "--predictions $W/predictions.json")

echo "[closeout] pending check (non-zero exit = the run is NOT done)"
python3 scripts/plenum_map_render.py --records "$W/agg/cells.json" --require-complete
