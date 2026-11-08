#!/bin/bash
# W4 (SPEC_D1 §2.4, §4.4, §6.3; runbook EXPLAIN.md §10.19–10.21): clusters and judge on S_gate, then on S.
# usage: w4_chain.sh <preflight|gate-clusters|gate-judge|s-clusters|s-judge>   (NEURONS_API_KEY_1 in the environment)
set -euo pipefail
MODE=$1
SC=/private/tmp/claude-501/-Users-abaybektursun-projects-FuelOS/964d1f2d-6631-419a-90f2-7fe212a804e5/scratchpad
NE=$(cd "$(dirname "$0")" && pwd)/ne   # the ne wrapper beside this script
HB=/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons
G=$HB/w3/grid; CORPUS=$HB/w1/q4b/1/out/corpus/corpus; T15=$G/1/out/t15_step1413814
GRID=$(ls -d $G/1/out/t0[0-6]_step* | paste -sd, -),$(ls -d $G/1b/out/t*_step* | paste -sd, -),$T15
FLASH=deepseek-ai/DeepSeek-V4.1-Flash-1017
cd /Users/abaybektursun/projects/interp_research-neurons-explain
# env overrides for the clusterer comparison and the v2 prompt runs: SET, CLUSTER_MODEL, JUDGE_MODEL, GATE, S, SEED_GC, SEED_GJ, SEED_SC, SEED_SJ, CAP
SET=${SET:-infra/pg/settings/deepinfra_w2c_mixed.json}; CLUSTER_MODEL=${CLUSTER_MODEL:-$FLASH}; JUDGE_MODEL=${JUDGE_MODEL:-$FLASH}
GATE=${GATE:-bank/w4/o7_gate}; S=${S:-bank/w4/o7_s}; CAP=${CAP:-8}
SEED_GC=${SEED_GC:-4101}; SEED_GJ=${SEED_GJ:-4102}; SEED_SC=${SEED_SC:-4103}; SEED_SJ=${SEED_SJ:-4104}
# cost guard: measured list prices per request (Sonnet, usage logs 2026-10-21) and ceilings per role; a round whose
# pending requests project above the ceiling is refused before any call is sent
PRICE_CLUSTERER=${PRICE_CLUSTERER:-0.053}; PRICE_JUDGE=${PRICE_JUDGE:-0.0162}
CLUSTER_MAX_USD=${CLUSTER_MAX_USD:-70}; JUDGE_MAX_USD=${JUDGE_MAX_USD:-200}
pending() {  # pending <explain dir> <batches csv>: requests without a kept answer or a decline
  python3 -I -c '
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); n = 0
for b in sys.argv[2].split(","):
    done = set()
    for suf in (".jsonl", ".declined.jsonl"):
        p = d / "responses" / (b + suf)
        if p.exists(): done |= {json.loads(l)["id"] for l in open(p)}
    n += sum(1 for l in open(d / "requests" / (b + ".jsonl")) if json.loads(l)["id"] not in done)
print(n)' "$1" "$2"
}
guard() {  # guard <explain dir> <batches csv> <role> <round>: print the projected cost; refuse above the ceiling
  local price max; case $3 in clusterer) price=$PRICE_CLUSTERER; max=$CLUSTER_MAX_USD ;; judge) price=$PRICE_JUDGE; max=$JUDGE_MAX_USD ;; *) echo "no price for role $3" >&2; return 5 ;; esac
  local n usd; n=$(pending "$1" "$2"); usd=$(python3 -c "print(round($n*$price, 2))")
  echo "round $4: $n pending $3 requests, projected \$$usd at \$$price each (ceiling \$$max)"
  python3 -c "import sys; sys.exit(0 if $usd <= $max else 5)" || { echo "refused: projected \$$usd exceeds the $3 ceiling \$$max" >&2; return 5; }
}
q() { "$@" 2> >(grep -v 'PyTorch was not found' >&2); }
loop() {  # loop <explain root> <batches csv> <roles> <collect subcommand> <batch>
  local D=$1 batches=$2 roles=$3 sub=$4 batch=$5
  local maxr=$((CAP+1))
  for round in $(seq 1 $maxr); do
    guard $D/explain "$batches" $roles $round
    q $NE llm run-many --explain $D/explain --batches "$batches" --settings $SET --concurrency 8
    if q $NE explain $sub --out $D --batch $batch --backend openai_compat; then echo "$sub ok (round $round)"; return 0; fi
    [ $round -lt $maxr ] || { echo "$sub still refused after $maxr rounds" >&2; return 1; }
    q $NE explain reject --out $D --batch $batch --roles $roles --backend openai_compat --max-attempts $CAP
  done
}
case $MODE in
  preflight)
    echo "grid dirs: $(echo $GRID | tr ',' '\n' | wc -l | tr -d ' ')"; echo "$GRID" | tr ',' '\n' | sed 's#.*/out/##'
    uv run --no-project --with transformers --with tokenizers --with anthropic --with numpy python -c "from infra.pg.neurons_verdict import check_grid; check_grid('$GRID'.split(',')); print('check_grid ok')" 2> >(grep -v 'PyTorch was not found' >&2)
    q $NE explain render --harvest $T15 --corpus $CORPUS --pole L00N181+ --n 1 --before 16 | head -20 ;;
  gate-clusters)
    if [ ! -f $GATE/explain/requests/w4c-clusterer-s0.jsonl ]; then
      q $NE explain positions --harvest $T15 --out $GATE --poles bank/w3/o7/draw/S_gate.json
      q $NE explain prepare-clusters --harvest $T15 --corpus $CORPUS --out $GATE --batch w4c --model $CLUSTER_MODEL --seed $SEED_GC
    fi
    loop $GATE w4c-clusterer-s0,w4c-clusterer-s1 clusterer collect-clusters w4c ;;
  gate-judge)
    q $NE explain prepare-judge --grid "$GRID" --corpus $CORPUS --out $GATE --batch w4j --model $JUDGE_MODEL --seed $SEED_GJ --gate $GATE
    loop $GATE w4j-judge-s0,w4j-judge-s1 judge collect-judge w4j
    q $NE verdict judge-gate --explain $GATE --out $GATE ;;
  s-clusters)
    if [ ! -f $S/explain/requests/w4c-clusterer-s0.jsonl ]; then
      q $NE explain positions --harvest $T15 --out $S --poles bank/w3/o7/draw/S.json
      q $NE explain prepare-clusters --harvest $T15 --corpus $CORPUS --out $S --batch w4c --model $CLUSTER_MODEL --seed $SEED_SC
    fi
    loop $S w4c-clusterer-s0,w4c-clusterer-s1 clusterer collect-clusters w4c ;;
  s-judge)
    [ -f $GATE/verdict/judge_gate.json ] || { echo "no judge_gate.json yet" >&2; exit 2; }
    q $NE explain prepare-judge --grid "$GRID" --corpus $CORPUS --out $S --batch w4j --model $JUDGE_MODEL --seed $SEED_SJ --gate $GATE
    loop $S w4j-judge-s0,w4j-judge-s1 judge collect-judge w4j ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "W4 $MODE DONE $(date -u)"
