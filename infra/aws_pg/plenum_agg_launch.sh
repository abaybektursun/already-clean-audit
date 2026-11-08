#!/usr/bin/env bash
# Launch a full PLENUM aggregation on a CPU box, reading every shard straight from S3.
#
# `plenum_aggregate._load` accepts s3:// URIs, so nothing is staged locally — which matters because
# the bank set is now ~30 shards and the control plane is a laptop that has already hit ENOSPC once.
#
# SHARD DISCOVERY, not a hardcoded list. My first version named the prefixes from memory and missed
# three census shards (`plenum_census_pol{,2,3}`) and all six text shards (`plenum_text_1..6`) —
# which would have pushed those cells back to PENDING and produced a wrong count to report. The
# forced-choice/generation banks stay explicit because their item samples must be poolable and a new
# prefix with a different sample would trip the poolability assertion far from its cause; the
# census/text/sae/control families are per-unit and safely enumerable.
#
# Known-bad prefixes are excluded BY NAME with the reason, never silently.
#
#   ./infra/aws_pg/plenum_agg_launch.sh <s3-suffix> [half]
set -eu
cd "$(dirname "$0")/../.."
SUF="${1:?usage: plenum_agg_launch.sh <s3-suffix> [half]}"
HALF="${2:-full}"
B=s3://pg-lineage-769063704696/results

FC=""
# ALL forced-choice shards, including the ones superseded by per-unit re-runs (plenum_fcg_d and
# plenum_fcg_pol). I briefly excluded those two to dodge the duplicate-bank guard; the better fix was
# to teach the guard the difference between "the same measurement twice" and "two different values
# for one cell". It now keeps the positionally aligned bank, and records a bit-identical aligned
# duplicate as a VERIFIED REPLICATION instead of failing.
#
# Including them is therefore worth more than excluding them: every doubly-measured cell becomes a
# systematic reproducibility audit (`verified_replications` in the artifact) rather than the two
# comparisons I happened to run by hand. Where they differ — coco_captions, where the pre-fix shards
# have len=1972/nan=0 and the re-runs have len=2000/nan=28 — alignment against the frozen item count
# picks the re-run, which is the one whose positional pairing with FLOOR and CEILING is intact.
for k in plenum_fcg_a plenum_fcg_b plenum_fcg_c plenum_fcg_d plenum_fcg_pol \
         plenum_fcg_cap_a plenum_fcg_cap_b plenum_fcg_cap_c \
         plenum_fcgu_ft-tallyqa plenum_fcgu_ft-textcaps plenum_fcgu_ft-textvqa \
         plenum_fcgu_ft-vizwizvqa plenum_fcgu_ft-vqav2 plenum_fcgu_ft-widgetcap \
         plenum_fcpol_pi0 plenum_fcpol_pi05 plenum_fcpol_u3 plenum_fcpol_u4 \
         plenum_fcsyn_rand_dec plenum_fcsyn_stitch0 \
         plenum_fc_u3c3; do
  FC="$FC $B/$k/out/fc_banks.json"
done

GEN=""
for k in plenum_gen_a plenum_gen_b plenum_gen_c plenum_gen_d plenum_gen_pol \
         plenum_gen_pol_pi0 plenum_gen_pol_pi05 plenum_gen_pol_rand_dec \
         plenum_gen_pol_stitch0 plenum_gen_pol_u3 plenum_gen_pol_u4 \
         plenum_gen_u3c3; do
  GEN="$GEN $B/$k/out/fc_banks.json"
done

# Census: EVERY shard. plenum_census_exp carries the action-expert absolute census and PG1's own
# self-reference row, neither of which any earlier shard had.
# EXCLUDE: plenum_sae_g2b (exit=1, zero cells — the row-floor failure, superseded by _g2b2).
EXCLUDE="plenum_sae_g2b"

discover () {  # discover <prefix-regex> <basename> -> prints s3 uris that actually exist
  aws s3 ls "$B/" --profile fuelos 2>/dev/null | awk '{print $2}' | tr -d '/' \
    | grep -E "$1" | grep -vxF "$EXCLUDE" | sort | while read -r k; do
        if aws s3 ls "$B/$k/out/$2" --profile fuelos >/dev/null 2>&1; then echo "$B/$k/out/$2"; fi
      done
}

# The film jobs are named `plenum_film_<UNIT>_<STEP>_{fc,ctl}` — they do NOT start with
# plenum_ctl / plenum_fcg, so the family patterns below would have dropped EVERY ONE of them.
# All 27 would have run, landed, and been invisible to the aggregation, and the film onset cells
# would have reported undefined for a bookkeeping reason rather than a scientific one. The film
# control evals are control shards and the film batteries are forced-choice banks, so each family
# pattern has to admit them explicitly.
CEN=$(discover '^plenum_census' metrics.json | tr '\n' ' ')

# SAE: the gemma-2b text stream is its own run, so a single-file flag would drop a stream.
SAE=$(discover '^plenum_sae' metrics.json | tr '\n' ' ')

CTL=$(discover '^plenum_ctl|^plenum_film_.*_ctl' metrics.json | tr '\n' ' ')

TEXT=$(discover '^plenum_text' text_banks.json | tr '\n' ' ')

# Film forced-choice banks: same battery, different weights, so they pool with the grid's FC banks.
# `install_unit` gave them ckpt: unit ids, and the aggregator routes ckpt: rows to the film clock
# instead of minting map cells — so adding them here is safe and is what makes the onset computable.
for k in $(aws s3 ls "$B/" --profile fuelos 2>/dev/null | awk '{print $2}' | tr -d '/' \
           | grep -E '^plenum_film_.*_fc' | sort); do
  if aws s3 ls "$B/$k/out/fc_banks.json" --profile fuelos >/dev/null 2>&1; then
    FC="$FC $B/$k/out/fc_banks.json"; echo "  + film fc $k"
  fi
done

ARGS="--half $HALF --fc$FC --gen$GEN --census $CEN --sae $SAE"
[ -n "$CTL" ]  && ARGS="$ARGS --control $CTL"
[ -n "$TEXT" ] && ARGS="$ARGS --text $TEXT"
ARGS="$ARGS --patch $B/plenum_patch/out/metrics.json"
ARGS="$ARGS --predictions $B/plenum_freeze2/out/plenum_predictions.json"

echo "[agg] launching $SUF (half=$HALF)"
echo "[agg] census : $(echo $CEN | wc -w) shards"
echo "[agg] text   : $(echo $TEXT | wc -w) shards"
echo "[agg] sae    : $(echo $SAE | wc -w) shards"
echo "[agg] control: $(echo $CTL | wc -w) shards"
# 2xlarge FIRST, deliberately. The Standard (A,C,D,H,I,M,R,T,Z) on-demand quota is **16 vCPU**, not
# 32 — the 32 figure is the G family. So one r7i.4xlarge saturates the entire standard family and
# locks out every other CPU job: it blocked the backup aggregation and the VL-supervision check at
# the same time. An 8-vCPU 2xlarge leaves room for a second CPU job, and the aggregation is bounded
# by bank JSON in memory (64 GB is ample) rather than by cores.
exec ./infra/aws_pg/pg run infra.aws_pg.jobs.plenum_aggregate \
  --type r7i.2xlarge,r6i.2xlarge,r7i.4xlarge,m7i.2xlarge --no-weights \
  --args "$ARGS" --s3-prefix "results/$SUF" --budget-min "${BUDGET:-600}" --detach --json
