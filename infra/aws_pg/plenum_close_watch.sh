#!/usr/bin/env bash
# Complete the closeout when a SPECIFIC aggregation lands. Narrower than plenum_finish.sh: that script
# launches its own aggregation, which would have made two writers on one prefix — the collision that cost
# this run duplicate instances twice. This one only ever CONSUMES a prefix someone else produced.
set -u
cd "$(dirname "$0")/../.."
SUF="${1:?usage: plenum_close_watch.sh <agg-prefix>}"
L=/tmp/plenum_close_watch.log
say () { echo "$(date -u +%H:%M:%S) $*" | tee -a "$L"; }

say "waiting for $SUF"
for i in $(seq 1 400); do
  st=$(aws s3 cp "s3://pg-lineage-769063704696/results/$SUF/STATUS" - --profile fuelos 2>/dev/null || true)
  [ -n "$st" ] && { say "$SUF STATUS=$st"; break; }
  sleep 120
done
st=$(aws s3 cp "s3://pg-lineage-769063704696/results/$SUF/STATUS" - --profile fuelos 2>/dev/null || echo missing)
if [ "$st" != "exit=0" ]; then
  say "aggregation did not exit clean ($st) — refusing to render. Its stages_completed says how far it got."
  exit 1
fi

say "outcome audit: which launched jobs did NOT exit 0"
: > /tmp/plenum_failed_jobs.txt
for s in $(grep -oE 'LAUNCHED [a-z0-9_.-]+' /tmp/q2.log 2>/dev/null | awk '{print $2}' | sort -u); do
  js=$(aws s3 cp "s3://pg-lineage-769063704696/results/$s/STATUS" - --profile fuelos 2>/dev/null || echo NO-STATUS)
  [ "$js" != "exit=0" ] && echo "$s $js" >> /tmp/plenum_failed_jobs.txt
done
n=$(wc -l < /tmp/plenum_failed_jobs.txt | tr -d ' ')
if [ "$n" != "0" ]; then say "$n launched job(s) not at exit=0:"; sed 's/^/    /' /tmp/plenum_failed_jobs.txt | tee -a "$L"
else say "every launched job exited 0"; fi

say "render MAP + figures + completeness gate"
./infra/aws_pg/plenum_post_agg.sh "$SUF" >> "$L" 2>&1
say "post_agg exit=$? (non-zero = pending > 0, a fact about the run)"
grep -n "pending = " studies/lineage/MAP.md | head -1 | tee -a "$L"

git add studies/lineage/MAP.md studies/lineage/paligemma1-pi0/research/plenum/figs 2>/dev/null
git commit -q -m "PLENUM: final MAP + figures from $SUF

Closeout over every discovered shard: 12 census (incl. the corrected vision cells), 24 control, 10 text,
2 sae, and the film forced-choice banks. Every cell decided by a banked measurement." 2>&1 | tee -a "$L"
say "done"
