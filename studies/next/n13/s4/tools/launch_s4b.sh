#!/usr/bin/env bash
# S4 capacity rule (the head's brief to the tenth N13 executor, 2026-11-06), as amended by the head at about 18:19Z: each launch tries
# --type g6e.2xlarge,g6e.4xlarge every 180 s for six tries, then every 600 s with no time limit; VcpuLimitExceeded and
# RequestLimitExceeded are retried the same way; any other launch error stops that launch. The loops' tries are serialized by a shared
# lock, and a try starts at least 60 s after the previous try of any loop ended. Every try is logged.
# usage: launch_s4b.sh <prefix name> <budget min> <log file> <number of the first try>
set -u
NAME=$1; BUDGET=$2; LOG=$3; TRY=$(( $4 - 1 ))
S=/private/tmp/claude-501/-Users-abaybektursun-projects-interp-research/5fa97882-3e2c-4d6c-9dab-49b9a1604fdb/scratchpad/s4launch
LOCK=$S/trylock; LAST=$S/last_try_end
cd /Users/abaybektursun/projects/interp_research-n13
while :; do
  TRY=$((TRY+1))
  until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
  if [ -f "$LAST" ]; then G=$(( $(cat "$LAST") + 60 - $(date +%s) )); [ $G -gt 0 ] && sleep $G; fi
  echo "TRY $TRY $NAME start $(date -u +%Y-%m-%dT%H:%M:%SZ) commit $(git rev-parse --short=12 HEAD) dirty_tracked=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')" >> "$LOG"
  OUT=$(./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 s3://pg-lineage-769063704696/results/next_n13/$NAME/steps.json" \
        --no-weights --type g6e.2xlarge,g6e.4xlarge --s3-prefix results/next_n13/$NAME --budget-min "$BUDGET" --json --detach 2>&1)
  RC=$?
  date +%s > "$LAST"; rmdir "$LOCK"
  echo "TRY $TRY $NAME end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$RC" >> "$LOG"
  echo "$OUT" | tail -5 >> "$LOG"
  if [ $RC -eq 0 ] && echo "$OUT" | grep -q '"instance_id"'; then echo "LAUNCHED $NAME" >> "$LOG"; exit 0; fi
  if ! echo "$OUT" | grep -q -E 'launchable in any AZ|RequestLimitExceeded'; then echo "STOP $NAME: not a capacity, quota or request-rate failure" >> "$LOG"; exit 2; fi
  if [ $TRY -lt 6 ]; then W=180; else W=600; fi
  echo "WAIT $W s" >> "$LOG"
  sleep $W
done
