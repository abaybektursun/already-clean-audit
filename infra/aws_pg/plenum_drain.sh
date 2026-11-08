#!/usr/bin/env bash
# ONE serialized drainer for the remaining PLENUM work.
#
# Competing retry loops caused two duplicate-instance incidents in this run (two jobs writing one S3
# prefix, overwriting each other's metrics). This is the single launcher: it drains a pipe-delimited
# queue file in order, retries only the head of the line, and removes a line only after a launch
# succeeds.
#
# The removal uses `|| true` before `mv`: `grep -v` exits 1 when it selects NO lines, so removing the
# LAST queue entry would leave an exit-code-gated `&& mv` unrun and the entry would relaunch forever.
# That bug cost three duplicate instances earlier.
#
#   queue line format:  <s3-suffix>|<module>|<args>|<budget-min>|<instance-types>[|<wait-for-prefix>]
#
# The optional 6th field holds a job prefix that must reach a TERMINAL state first. Needed because
# re-measuring a re-trained unit must not start against the OLD published weights -- the u3c
# safetensors gate cannot tell one U3c from another.
#   ./infra/aws_pg/plenum_drain.sh /tmp/q2.txt
set -u
cd "$(dirname "$0")/../.."
Q="${1:?usage: plenum_drain.sh <queue-file>}"
L="${Q%.txt}.log"
while :; do
  line=$(grep -vE '^\s*(#|$)' "$Q" 2>/dev/null | head -1)
  if [ -z "$line" ]; then echo "$(date -u +%H:%M:%S) queue empty — done" >> "$L"; exit 0; fi
  suf=$(echo "$line" | cut -d'|' -f1)
  mod=$(echo "$line" | cut -d'|' -f2)
  args=$(echo "$line" | cut -d'|' -f3)
  bud=$(echo "$line" | cut -d'|' -f4)
  typ=$(echo "$line" | cut -d'|' -f5)
  dep=$(echo "$line" | cut -d'|' -f6)
  if [ -n "$dep" ]; then
    if ! aws s3 ls "s3://pg-lineage-769063704696/results/$dep/STATUS" --profile fuelos >/dev/null 2>&1; then
      echo "$(date -u +%H:%M:%S) WAIT $suf (needs $dep)" >> "$L"; sleep 180; continue
    fi
    # AND the ARTIFACT must be newer than the producing job. Gating on STATUS alone is not enough:
    # u3run3 reached terminal state at 08:56 with its in-job publish FAILED, the gate fired at 08:59,
    # and the unit was not republished until 09:05 — so the battery synced the PREVIOUS run's weights
    # and returned results bit-identical to the contaminated run. A job being "done" does not mean the
    # artifact its consumers read has been updated.
    case "$args" in
      *u3c*|*U3c*)
        js=$(aws s3api head-object --bucket pg-lineage-769063704696 \
               --key "results/$dep/STATUS" --profile fuelos --query LastModified --output text 2>/dev/null)
        us=$(aws s3api head-object --bucket pg-lineage-769063704696 \
               --key weights/lacuna_units/U3c/model.safetensors --profile fuelos \
               --query LastModified --output text 2>/dev/null)
        if [ -z "$us" ] || [ "$us" \< "$js" ]; then
          echo "$(date -u +%H:%M:%S) WAIT $suf (u3c unit older than $dep: unit=$us job=$js)" >> "$L"
          sleep 180; continue
        fi ;;
    esac
  fi
  # Skip anything already complete, so a re-drain is idempotent.
  if aws s3 cp "s3://pg-lineage-769063704696/results/$suf/STATUS" - --profile fuelos 2>/dev/null | grep -q "exit=0"; then
    echo "$(date -u +%H:%M:%S) SKIP $suf (already exit=0)" >> "$L"
    grep -vF "$line" "$Q" > "$Q.tmp" || true; mv "$Q.tmp" "$Q"; continue
  fi
  # DEPENDENCY GATE: anything that installs the u3c UNIT must wait for it to publish. Without this
  # the drainer would launch the u3c battery against an empty path and fail 15GB into a download.
  case "$args" in
    *u3c*|*U3c*)
      if ! aws s3 ls "s3://pg-lineage-769063704696/weights/lacuna_units/U3c/" --profile fuelos 2>/dev/null | grep -q safetensors; then
        echo "$(date -u +%H:%M:%S) HOLD $suf (u3c not published yet)" >> "$L"
        sleep 180; continue
      fi ;;
  esac
  echo "$(date -u +%H:%M:%S) trying $suf" >> "$L"
  if ./infra/aws_pg/pg run "$mod" --type "$typ" --no-weights --args "$args" \
       --s3-prefix "results/$suf" --budget-min "$bud" --detach --json >> "$L" 2>&1; then
    echo "$(date -u +%H:%M:%S) LAUNCHED $suf" >> "$L"
    grep -vF "$line" "$Q" > "$Q.tmp" || true; mv "$Q.tmp" "$Q"
    sleep 90
  else
    # A job that can NEVER launch (malformed args, retired instance family) would otherwise be retried
    # at the head of the line forever — the queue would never drain and the autonomous closeout would
    # never fire. Count attempts per line and set it aside after enough of them, so a single bad entry
    # cannot hold the whole program hostage. Set-aside entries are recorded, never silently dropped.
    n=$(grep -cF "$line" "$Q.attempts" 2>/dev/null || echo 0)
    echo "$line" >> "$Q.attempts"
    if [ "$n" -ge 20 ]; then
      echo "$(date -u +%H:%M:%S) SET ASIDE after $n attempts: $suf" >> "$L"
      echo "$line" >> "$Q.failed"
      grep -vF "$line" "$Q" > "$Q.tmp" || true; mv "$Q.tmp" "$Q"
    fi
    sleep 150
  fi
done
