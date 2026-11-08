#!/usr/bin/env bash
# Autonomous final closeout: wait for all work to drain, then aggregate → render → figures → gate.
#
# This exists so the program finishes without a human in the loop. It waits on two conditions that are
# both necessary — an empty queue means nothing is left to LAUNCH, and no live pg instances means
# nothing is left RUNNING. Either alone is insufficient: the queue empties the moment the last job is
# launched, hours before it finishes.
#
# It then runs the aggregation with EVERY shard discovered (the launcher globs the prefix namespace, so
# the u3c re-measurements and the census fix are picked up without being named here), renders the MAP,
# regenerates the figures, and runs the completeness gate. The gate's non-zero exit is preserved in the
# log rather than swallowed: `pending > 0` at the end is a fact about the program, not a script error.
#
#   nohup ./infra/aws_pg/plenum_finish.sh /tmp/q2.txt > /tmp/finish.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
Q="${1:?usage: plenum_finish.sh <queue-file>}"
L=/tmp/plenum_finish.log
# A FRESH prefix, and cleared before use. `plenum_agg_final` already existed from a previous session
# (STATUS exit=0, dated a day earlier, predating the argv and incremental-save fixes). The wait loop
# below breaks as soon as the target's STATUS is non-empty — so launching into an occupied prefix would
# have read the STALE exit=0 immediately and rendered the FINAL map and figures from yesterday's cells.
# Clearing the prefix first makes "STATUS exists" mean "this run finished" rather than "some run did".
SUF="plenum_agg_close"

say () { echo "$(date -u +%H:%M:%S) $*" | tee -a "$L"; }

# THE GATE IS THE MAP, NOT THE QUEUE. Once every cell is decided, the remaining queue is exploratory
# content (U1/U3/U3c faculty batteries whose onset cells are already terminal), and holding the closeout
# for ~30 GPU-h of work that cannot change a single cell would be the wrong trade. So: close out as soon
# as the CELL-DECIDING jobs are done, and let the exploratory tail keep running behind it.
#
# `DECIDERS` lists the prefixes a cell depends on. Empty or all-terminal ⇒ proceed.
DECIDERS="${DECIDERS:-plenum_film_U3_00004096_ctl3}"
say "waiting on the cell-deciding jobs: $DECIDERS"
for i in $(seq 1 300); do
  pend=""
  for d in $DECIDERS; do
    aws s3 ls "s3://pg-lineage-769063704696/results/$d/STATUS" --profile fuelos >/dev/null 2>&1 || pend="$pend $d"
  done
  [ -z "$pend" ] && { say "all deciders terminal"; break; }
  sleep 60
done

say "waiting for the queue to drain and all instances to exit"
for i in $(seq 1 2000); do
  left=$(grep -cvE '^\s*(#|$)' "$Q" 2>/dev/null || echo 1)
  # Scoped to pg jobs by tag. Counting EVERY instance in the account would let one unrelated box
  # block the closeout forever, which would silently defeat the whole point of this script.
  live=$(aws ec2 describe-instances --profile fuelos --region us-east-1 \
           --filters "Name=instance-state-name,Values=pending,running" "Name=tag:Name,Values=pg-*" \
           --query 'length(Reservations[].Instances[])' --output text 2>/dev/null || echo 1)
  [ "$left" = "0" ] && [ "$live" = "0" ] && { say "drained (queue 0, instances 0)"; break; }
  # Do not wait out the exploratory tail: if every decider is terminal, the map is decided and the
  # closeout can render while those jobs continue. They are recorded in the outcome audit either way.
  alldone=1
  for d in $DECIDERS; do
    aws s3 ls "s3://pg-lineage-769063704696/results/$d/STATUS" --profile fuelos >/dev/null 2>&1 || alldone=0
  done
  if [ "$alldone" = "1" ] && [ "$i" -ge 3 ]; then
    say "deciders terminal; closing out with $left exploratory job(s) still queued"
    break
  fi
  sleep 120
done

left=$(grep -cvE '^\s*(#|$)' "$Q" 2>/dev/null || echo 1)
if [ "$left" != "0" ]; then
  say "NOTE: $left exploratory job(s) still queued — they cannot change a map cell, so the aggregate is"
  say "      complete with respect to the deliverable. They appear in the outcome audit below."
fi

# OUTCOME AUDIT BEFORE AGGREGATING. The drainer removes a line once the job LAUNCHES; it never checks
# whether the job SUCCEEDED. Without this, a battery that exits 1 leaves its cells unmeasured, the
# queue still drains, the closeout still fires, and the final `pending` count reports a number with no
# indication that jobs failed. Enumerate every prefix this run launched and say which ones did not
# exit 0 — a closeout that cannot distinguish "not measured" from "measured and failed" is the same
# error class as a census cell that cannot distinguish "no change" from "no comparison".
say "auditing launched-job outcomes"
: > /tmp/plenum_failed_jobs.txt
for suf in $(grep -oE '^[a-z0-9_.-]+' "$Q.attempts" 2>/dev/null | sort -u) \
           $(grep -oE 'LAUNCHED [a-z0-9_.-]+' "${Q%.txt}.log" 2>/dev/null | awk '{print $2}' | sort -u); do
  st=$(aws s3 cp "s3://pg-lineage-769063704696/results/$suf/STATUS" - --profile fuelos 2>/dev/null || echo "NO-STATUS")
  case "$st" in
    exit=0) ;;
    *) echo "$suf $st" >> /tmp/plenum_failed_jobs.txt ;;
  esac
done
nf=$(wc -l < /tmp/plenum_failed_jobs.txt | tr -d ' ')
if [ "$nf" != "0" ]; then
  say "$nf launched job(s) did NOT exit 0 — their cells will be missing from the aggregate:"
  sed 's/^/    /' /tmp/plenum_failed_jobs.txt | tee -a "$L"
else
  say "every launched job exited 0"
fi
if [ -s "$Q.failed" ]; then
  say "queue entries SET ASIDE as unlaunchable:"; sed 's/^/    /' "$Q.failed" | cut -d'|' -f1 | tee -a "$L"
fi

say "clearing any prior artifacts at $SUF so its STATUS means THIS run"
aws s3 rm "s3://pg-lineage-769063704696/results/$SUF" --recursive --profile fuelos >/dev/null 2>&1 || true
if aws s3 ls "s3://pg-lineage-769063704696/results/$SUF/STATUS" --profile fuelos >/dev/null 2>&1; then
  say "REFUSING: $SUF/STATUS still present after clear — a stale STATUS would be read as this run's"
  exit 1
fi

say "final aggregation (all shards discovered)"
BUDGET=600 ./infra/aws_pg/plenum_agg_launch.sh "$SUF" full >> "$L" 2>&1 || { say "launch failed"; exit 1; }

say "waiting for $SUF"
for i in $(seq 1 400); do
  st=$(aws s3 cp "s3://pg-lineage-769063704696/results/$SUF/STATUS" - --profile fuelos 2>/dev/null || true)
  [ -n "$st" ] && { say "$SUF STATUS=$st"; break; }
  sleep 120
done

st=$(aws s3 cp "s3://pg-lineage-769063704696/results/$SUF/STATUS" - --profile fuelos 2>/dev/null || echo missing)
if [ "$st" != "exit=0" ]; then say "aggregation did not exit clean ($st) — refusing to render"; exit 1; fi

say "render + figures + completeness gate"
./infra/aws_pg/plenum_post_agg.sh "$SUF" >> "$L" 2>&1
gate=$?
say "post_agg exit=$gate (non-zero = pending > 0, which is a fact about the run)"

say "committing artifacts"
git add studies/lineage/MAP.md studies/lineage/paligemma1-pi0/research/plenum/figs 2>/dev/null
git commit -q -m "PLENUM: final MAP + figures from $SUF

Autonomous closeout: queue drained, all instances exited, full aggregation over every discovered
shard (including the census fix for the 10 false vision verdicts and the clean u3c re-measurements),
MAP re-rendered, figures regenerated, completeness gate run." 2>&1 | tee -a "$L"
say "done"
