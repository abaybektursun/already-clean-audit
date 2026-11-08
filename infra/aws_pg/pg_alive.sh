#!/usr/bin/env bash
# "Is anything actually running?" — answered from AWS and S3, never from a belief or a watcher.
#
# WHY THIS EXISTS. Three times in one program, work stalled and the stall was invisible because the
# signal being read had no failure branch:
#
#   1. `plenum_close_watch.sh` watched an S3 prefix nobody was going to write. Its silence meant
#      "no job exists"; it was read as "job still running". 25.6 hours, zero jobs, zero writes.
#   2. `plenum_agg_final2` already held a two-day-old STATUS. A fresh launch into that prefix read
#      the stale `exit=0` within seconds and reported success for a run that had just started.
#   3. The assistant answered "it's going well" twice without checking either.
#
# All three share one shape: NOTHING HAPPENING and WORKING FINE produce the identical observation.
# `memory/feedback_signals_that_cannot_fail.md` is the standing rule — name the observation that
# would make the signal report FAILURE. Silence cannot. So this script makes idleness LOUD:
#
#   * it exits 1 when nothing is running, so "no news" cannot be reported as "fine";
#   * it prints the AGE of the newest artifact per prefix, so a running instance that has stopped
#     producing is distinguishable from one that is working;
#   * it flags a STATUS older than its own job.log as STALE — the occupied-prefix trap.
#
# Run it before making ANY claim about progress.
#
#   ./infra/aws_pg/pg_alive.sh                 # account-wide
#   ./infra/aws_pg/pg_alive.sh plenum_agg      # only prefixes matching a substring
set -u
FILTER="${1:-}"
BUCKET=s3://pg-lineage-769063704696/results
PROFILE=fuelos
now=$(date -u +%s)

echo "=== live pg instances ==="
inst=$(aws ec2 describe-instances --profile "$PROFILE" --region us-east-1 \
        --filters "Name=instance-state-name,Values=pending,running" "Name=tag:Name,Values=pg-*" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]' \
        --output text 2>/dev/null)
if [ -z "$inst" ]; then
  echo "  (none)"
else
  echo "$inst" | while read -r id typ launch; do
    ls_s=$(date -u -j -f "%Y-%m-%dT%H:%M:%S" "${launch%%+*}" +%s 2>/dev/null || echo "$now")
    echo "  $id  $typ  up $(( (now - ls_s) / 60 )) min"
  done
fi

echo
echo "=== prefixes written in the last 6h ==="
recent=$(aws s3 ls "$BUCKET/" --recursive --profile "$PROFILE" 2>/dev/null \
         | awk -v f="$FILTER" '($0 ~ f)' \
         | awk '{d=$1" "$2; p=$4; sub(/^results\//,"",p); sub(/\/.*$/,"",p);
                 if (d > seen[p]) seen[p]=d}
                END {for (p in seen) print seen[p], p}' | sort -r | head -20)
[ -z "$recent" ] && echo "  (nothing)" || echo "$recent" | sed 's/^/  /'

echo
echo "=== STALE-STATUS check (a STATUS older than its own job.log = occupied prefix) ==="
stale=0
for p in $(aws s3 ls "$BUCKET/" --profile "$PROFILE" 2>/dev/null | awk '{print $2}' | tr -d '/' \
           | grep -- "$FILTER" | tail -40); do
  s=$(aws s3 ls "$BUCKET/$p/STATUS" --profile "$PROFILE" 2>/dev/null | awk '{print $1" "$2}')
  l=$(aws s3 ls "$BUCKET/$p/job.log" --profile "$PROFILE" 2>/dev/null | awk '{print $1" "$2}')
  if [ -n "$s" ] && [ -n "$l" ] && [ "$s" \< "$l" ]; then
    echo "  STALE: $p  STATUS=$s  job.log=$l  <- STATUS predates this run's log"
    stale=1
  fi
done
[ "$stale" = "0" ] && echo "  (none)"

echo
if [ -z "$inst" ]; then
  echo "VERDICT: IDLE — no pg instance is running. Nothing is in progress."
  echo "         Do NOT report this as 'running' or 'going well'. If work should be"
  echo "         happening, it is not: relaunch it."
  exit 1
fi
echo "VERDICT: $(echo "$inst" | wc -l | tr -d ' ') instance(s) running."
exit 0
