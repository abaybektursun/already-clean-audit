#!/bin/bash
# Q1 explain wave through the DeepInfra backend (SPEC_D1 §2.2, §9.2 W2-D / W2-C):
#   prepare -> run explainer + second reader -> collect (reject/rerun loop) -> prepare-writer -> run writer (+ job) -> collect-writer (same loop)
# usage: q1_explain_wave.sh <harvest out dir> <corpus dir> <explain root (holds explain/)> <batch> <poles.json> <shuffle seed> [skip-prepare]
# Requires NEURONS_API_KEY_1 in the environment (source ~/.config/interp_research/keys.env first; never print it).
set -euo pipefail
H=$1; C=$2; OUT=$3; B=$4; POLES=$5; SEED=$6; SKIP=${7:-}
NE=/private/tmp/claude-501/-Users-abaybektursun-projects-FuelOS/964d1f2d-6631-419a-90f2-7fe212a804e5/scratchpad/bin/ne
SETTINGS_DIR=/Users/abaybektursun/projects/interp_research-neurons-explain/infra/pg/settings
EXPL_SETTINGS=${EXPL_SETTINGS:-$SETTINGS_DIR/deepinfra.json}; EXPL_MODEL=${EXPL_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
WRITER_SETTINGS=${WRITER_SETTINGS:-$SETTINGS_DIR/deepinfra.json}; WRITER_MODEL=${WRITER_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
MAX_ROUNDS=4
cd /Users/abaybektursun/projects/interp_research-neurons-explain
[ -n "${NEURONS_API_KEY_1:-}" ] || { echo "NEURONS_API_KEY_1 not in environment" >&2; exit 2; }
q() { "$@" 2> >(grep -v 'PyTorch was not found' >&2); }

# run the batches, then collect; on a collect refusal move the rejected lines aside and run again
loop() {  # loop <settings> <roles csv> <batches csv> <reject extra args (quoted, may be empty)> <collect subcommand> [collect extra args...]
  local settings=$1 roles=$2 batches=$3 rextra=$4 sub=$5; shift 5
  local round=1
  while :; do
    q $NE llm run-many --explain "$OUT/explain" --batches "$batches" --settings "$settings" --concurrency 8
    if q $NE explain "$sub" --out "$OUT" --batch "$B" --backend openai_compat "$@"; then return 0; fi
    [ $round -lt $MAX_ROUNDS ] || { echo "collect still refused after $MAX_ROUNDS rounds" >&2; return 1; }
    q $NE explain reject --out "$OUT" --batch "$B" --roles "$roles" --backend openai_compat --max-attempts 3 $rextra
    round=$((round+1))
  done
}

if [ -z "$SKIP" ]; then
  q $NE explain prepare --harvest "$H" --corpus "$C" --out "$OUT" --batch "$B" --roles explainer,second_reader --poles "$POLES" --model "$EXPL_MODEL"
fi
loop "$EXPL_SETTINGS" explainer,second_reader "$B-explainer,$B-second_reader" "" collect --roles explainer,second_reader
q $NE explain prepare-writer --out "$OUT" --batch "$B" --model "$WRITER_MODEL"
if [ -f "$OUT/explain/requests/$B-writer_job.jsonl" ]; then
  loop "$WRITER_SETTINGS" writer,writer_job "$B-writer,$B-writer_job" "--harvest $H --corpus $C" collect-writer --harvest "$H" --corpus "$C" --seed "$SEED"
else
  loop "$WRITER_SETTINGS" writer "$B-writer" "--harvest $H --corpus $C" collect-writer --harvest "$H" --corpus "$C" --seed "$SEED"
fi
wc -l "$OUT/explain/explanations.jsonl" "$OUT/explain/texts.jsonl"
echo "probe poles: $(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$OUT/explain/probe_poles.json")"
echo "WAVE DONE $(date -u)"
