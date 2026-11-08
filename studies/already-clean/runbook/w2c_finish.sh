#!/bin/bash
# W2-C under the §2.2 model policy amendment: finish one cell's writer chain (Sonnet) or second-reader chain (Flash).
# usage: w2c_finish.sh <q4b|g4|o7> <writer|reader>   (NEURONS_API_KEY_1 in the environment)
set -euo pipefail
CELL=$1; WHAT=$2
SC=/private/tmp/claude-501/-Users-abaybektursun-projects-FuelOS/964d1f2d-6631-419a-90f2-7fe212a804e5/scratchpad
NE=$SC/bin/ne; SET=$SC/settings/deepinfra_w2c_mixed.json
HB=/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1; C=$HB/q4b/1/out/corpus/corpus
case $CELL in q4b) H=$HB/q4b/1/out/q4b ;; g4) H=$HB/g4/1/out/g4 ;; o7) H=$HB/o7_t15/1/out/o7_t15 ;; *) echo "bad cell" >&2; exit 2 ;; esac
cd /Users/abaybektursun/projects/interp_research-neurons-explain
OUT=bank/w2c/$CELL
q() { "$@" 2> >(grep -v 'PyTorch was not found' >&2); }
if [ "$WHAT" = writer ]; then
  q $NE explain prepare-writer --out $OUT --batch c1 --model anthropic/claude-sonnet-5-5
  for round in 1 2 3 4; do
    q $NE llm run-many --explain $OUT/explain --batches c1-writer,c1-writer_job --settings $SET --concurrency 8
    if q $NE explain collect-writer --out $OUT --batch c1 --backend openai_compat --harvest $H --corpus $C --seed 0; then echo "collect-writer ok (round $round)"; break; fi
    [ $round -lt 4 ] || { echo "collect-writer still refused after 4 rounds" >&2; exit 1; }
    q $NE explain reject --out $OUT --batch c1 --roles writer,writer_job --backend openai_compat --harvest $H --corpus $C --max-attempts 3
  done
  echo "texts $(wc -l < $OUT/explain/texts.jsonl | tr -d ' ') probe poles $(python3 -I -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' $OUT/explain/probe_poles.json) declined $(cat $OUT/explain/responses/*.declined.jsonl 2>/dev/null | wc -l | tr -d ' ')"
elif [ "$WHAT" = reader ]; then
  for round in 1 2 3 4; do
    q $NE llm run-many --explain $OUT/explain --batches c1-second_reader --settings $SET --concurrency 8
    if q $NE explain collect --out $OUT --batch c1 --roles second_reader --backend openai_compat; then echo "collect second_reader ok (round $round)"; break; fi
    [ $round -lt 4 ] || { echo "collect second_reader still refused after 4 rounds" >&2; exit 1; }
    q $NE explain reject --out $OUT --batch c1 --roles second_reader --backend openai_compat --max-attempts 3
  done
  echo "second reader answers $(wc -l < $OUT/explain/responses/c1-second_reader.usage.jsonl | tr -d ' ')"
else echo "bad mode" >&2; exit 2; fi
echo "W2C_FINISH $CELL $WHAT DONE $(date -u)"
