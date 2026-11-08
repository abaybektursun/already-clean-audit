#!/bin/bash
# pg reconcile tick — the root-cause fix for the silent-idle / missed-completion failure mode.
# Run by a macOS launchd agent every N minutes (com.pglineage.reconcile). Because it runs on the
# laptop it HAS the fuelos creds + .pgvenv + the worktree; because it's launchd it survives the
# terminal/session ending (unlike /loop or background monitors). Each tick:
#   1. `pg reconcile` over the durable active-jobs registry.
#   2. exit 0 → all jobs in flight (or none): log a heartbeat, done.
#   3. exit 1/2 → a job FAILED/STALLED or COMPLETED: macOS-notify AND re-invoke a fresh headless
#      Claude agent to HANDLE it (pull results + ack on complete; surface job.log + record on
#      failure). One handler at a time (lockfile). So the agent is re-invoked on a fixed cadence
#      whenever — and only when — there is AWS work needing attention. Idle bounded to the tick.
#
# Disable anytime:  launchctl unload ~/Library/LaunchAgents/com.pglineage.reconcile.plist
# Logs:             <worktree>/.pg_reconcile.log   (+ .pg_failures.log for surfaced failures)
set -uo pipefail

WT=/Users/abaybektursun/projects/interp_research
LOG="$WT/.pg_reconcile.log"
LOCK=/tmp/pg_reconcile_handler.lock
export PATH="/Users/abaybektursun/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
# AUTONOMOUS=1 → re-invoke a headless Claude agent to handle terminal jobs (full autonomy, runs
# bash on this machine). AUTONOMOUS=0 → notify + record only (no agent spawn). Default 1.
AUTONOMOUS="${PG_RECONCILE_AUTONOMOUS:-1}"

cd "$WT" || exit 0
ts=$(date -u +%FT%TZ)
out=$(./infra/aws_pg/pg reconcile 2>&1); rc=$?
printf '[%s] reconcile rc=%s\n%s\n' "$ts" "$rc" "$out" >> "$LOG"
[ "$rc" -eq 0 ] && exit 0

osascript -e "display notification \"pg jobs need attention (reconcile rc=$rc)\" with title \"AWS pg\"" >/dev/null 2>&1 || true

if [ "$AUTONOMOUS" != "1" ]; then
  printf '[%s] actionable (rc=%s) — notify-only (AUTONOMOUS=0)\n' "$ts" "$rc" >> "$LOG"
  exit 0
fi

# one headless handler at a time (mkdir is atomic). Stale lock >30min is reclaimed.
if [ -d "$LOCK" ] && [ "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then rmdir "$LOCK" 2>/dev/null || true; fi
if mkdir "$LOCK" 2>/dev/null; then
  trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT
  printf '[%s] spawning headless claude handler (rc=%s)...\n' "$ts" "$rc" >> "$LOG"
  claude -p "You are the AWS pg job reconciler (a SAFETY NET — capture state, do not do science). \
Working dir: $WT. Run ./infra/aws_pg/pg reconcile. For EACH prefix it reports COMPLETE: \
aws s3 sync s3://pg-lineage-769063704696/<prefix>/out/ into .pg_pulls/<prefix>/ (a gitignored scratch \
dir; --profile fuelos) — do NOT write into the repo's tracked results/ tree, the interactive agent \
places canonical results — then report what completed + the headline numbers from any metrics.json, \
append a dated COMPLETE line to .pg_failures.log, and ./infra/aws_pg/pg ack <prefix>. For EACH prefix \
it reports FAILED or STALLED: aws s3 cp s3://pg-lineage-769063704696/<prefix>/job.log - --profile fuelos \
| tail -40, append a dated error summary to .pg_failures.log, report loudly, then ./infra/aws_pg/pg ack \
<prefix> so it does not loop. Do NOT re-launch/retry jobs, do NOT commit, do NOT edit tracked files. \
Keep it tight, then exit." \
    --dangerously-skip-permissions >> "$LOG" 2>&1 || true
  printf '[%s] handler done\n' "$(date -u +%FT%TZ)" >> "$LOG"
fi
