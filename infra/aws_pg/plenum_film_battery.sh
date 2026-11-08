#!/usr/bin/env bash
# THE FILM (§2.3): the Tier-1 battery at every checkpoint of U1, U3 and U3c, plus the control eval
# at the same checkpoints, so `onset(faculty)` and `acquisition(control)` are read on ONE clock.
#
# Checkpoints enter through `install_unit` as `ckpt:<s3-uri>` — the same loader every roster unit
# uses — so a film point and a grid row are the same measurement on different weights. That is why
# there is no `plenum_film.py`: a second battery could drift from the first, and then a film point
# would not be comparable to the grid row it is supposed to be compared against.
#
# ON DENSITY. §12's degradation ladder permits an items floor of 1,000/task, which would roughly
# halve the cost of 24 checkpoint batteries. NOT taken, and the reason is not thoroughness for its
# own sake: S is defined against FLOOR and CEILING rows measured at the FROZEN per-task counts, and
# the aggregator asserts shard poolability on the item hashes. A film at n=1,000 would draw a
# different sample, fail that assertion (correctly), and need its OWN pg1 and matched_dw band rows
# at n=1,000 plus a separate aggregation — the MDE precedent. Cheaper per run, more moving parts, and
# a second set of bands to keep straight. The frozen density pools with the bands that already exist.
#
#   ./infra/aws_pg/plenum_film_battery.sh [U3c U3 U1]
set -eu
cd "$(dirname "$0")/../.."
S3=s3://pg-lineage-769063704696
UNITS="${*:-U3c U3 U1}"

launch () {  # launch <suffix> <module> <args> <budget> <type>
  local suffix="$1" mod="$2" jargs="$3" budget="$4" itype="$5"
  for i in $(seq 1 480); do
    if ./infra/aws_pg/pg run "$mod" --type "$itype" --no-weights \
         --args "$jargs" --s3-prefix "results/$suffix" --budget-min "$budget" --detach --json \
         >> "/tmp/film_$suffix.log" 2>&1; then
      echo "  launched $suffix"; return 0
    fi
    sleep 180        # the training fleet holds the G quota; the film waits it out
  done
  echo "  GAVE UP on $suffix — quota never freed" >&2; return 1
}

for U in $UNITS; do
  # Wait for that unit's training job to reach a terminal state, so its checkpoint grid is complete.
  # Filming a partial grid would make the onset statistic undefined on a grid that is about to change,
  # and §2.3 forbids interpolating an undefined onset rather than re-reading it later.
  # The job prefix per film unit. These are LAUNCH-SPECIFIC and must be updated when a unit is
  # relaunched: an earlier version of this script waited on prefixes that had since failed, so their
  # exit=1 STATUS files satisfied the wait instantly and the film would have skipped every unit for
  # "no checkpoints". Waiting on a STATUS file means waiting for a TERMINAL state, not a good one —
  # so the checkpoint listing below is what actually decides, and an empty listing is recorded as
  # incomplete rather than filmed.
  case "$U" in
    U3c) J=plenum_a3_u3run2 ;;     # the co-training run (prefix-LM leak fixed)
    U3)  J=plenum_a3_u3c2 ;;       # unit U3 -- the suffix is unfortunate, the UNIT is U3
    U1)  J=plenum_a3_u1c3 ;;
    *)   echo "unknown film unit $U" >&2; exit 1 ;;
  esac
  echo "[film] $U: waiting for $J to finish"
  for i in $(seq 1 720); do
    aws s3 ls "$S3/results/$J/STATUS" --profile fuelos >/dev/null 2>&1 && break
    sleep 120
  done

  CKS=$(aws s3 ls "$S3/weights/plenum_film/$U/seed0/" --profile fuelos 2>/dev/null \
        | awk '{print $2}' | tr -d '/' | sort)

  # SUBSAMPLE TO THE §2.3 GRID. The spec films "step 0 then every B*/8" — eight or nine points, not
  # every checkpoint that happens to exist. U3/U3c write exactly 8 (film_points=8), but the U1
  # calibration writes every 512 steps to the 16k cap, i.e. 32 checkpoints. Filming all of them would
  # be 64 jobs for one unit at frozen battery density, for a statistic defined on 8 points.
  #
  # Evenly spaced over what exists, keeping the LAST checkpoint (the endpoint the onset is measured
  # relative to). Dropped points are LOGGED, never silently truncated — a film that quietly covered a
  # quarter of the grid would still look like a film.
  N_WANT=8
  N_HAVE=$(echo "$CKS" | grep -c .)
  if [ "$N_HAVE" -gt "$N_WANT" ]; then
    CKS_ALL="$CKS"
    CKS=$(echo "$CKS" | awk -v n="$N_HAVE" -v k="$N_WANT" 'NR % int((n+k-1)/k) == 0 || NR == n')
    echo "[film] $U: subsampled $N_HAVE checkpoints -> $(echo "$CKS" | grep -c .) per §2.3 (grid is B*/8, not every save)"
    echo "[film] $U: dropped $(comm -23 <(echo "$CKS_ALL") <(echo "$CKS") | tr '\n' ' ')"
  fi
  if [ -z "$CKS" ]; then echo "[film] $U: no checkpoints — skipping (recorded as incomplete)"; continue; fi
  echo "[film] $U checkpoints: $(echo $CKS | tr '\n' ' ')"

  # ONE JOB PER CHECKPOINT. A multi-checkpoint job at frozen density gets reaped at its budget
  # deadline with exit=0 and a partial bank — the completion-signal lesson this program already paid
  # for once.
  for C in $CKS; do
    URI="ckpt:$S3/weights/plenum_film/$U/seed0/$C/pretrained_model"
    launch "plenum_film_${U}_${C}_fc"  infra.aws_pg.jobs.plenum_fc \
      "--units $URI --readout forced_choice" 700 g6e.2xlarge
    launch "plenum_film_${U}_${C}_ctl" infra.aws_pg.jobs.plenum_control \
      "--units $URI" 300 g6e.2xlarge
  done
done
echo "[film] all checkpoint batteries queued"
