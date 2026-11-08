#!/bin/bash
# W2-C audit explain waves with the frozen prompt and models (SPEC_D1 §9.2 W2-C): one cell per call.
# usage: w2c_explain.sh <cell: q4b|g4|o7>   (requires NEURONS_API_KEY_1 in the environment)
set -euo pipefail
CELL=$1
HB=/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/w1
EB=/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w2c
SETTINGS_DIR=/Users/abaybektursun/projects/interp_research-neurons-explain/infra/pg/settings
WAVE=/private/tmp/claude-501/-Users-abaybektursun-projects-FuelOS/964d1f2d-6631-419a-90f2-7fe212a804e5/scratchpad/bin/q1_explain_wave.sh
CORPUS=$HB/q4b/1/out/corpus/corpus
case $CELL in
  q4b) H=$HB/q4b/1/out/q4b; POLES=/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w1/q4b_split/audit_poles.json ;;
  g4)  H=$HB/g4/1/out/g4;   POLES=/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w1/g4_split/audit_poles.json ;;
  o7)  H=$HB/o7_t15/1/out/o7_t15; POLES=$EB/o7_all_poles.json
       python3 -I -c "import json,sys; a=json.load(open(sys.argv[1])); e=json.load(open(sys.argv[2])); json.dump(a+e, open(sys.argv[3],'w'), indent=1); print('o7 poles', len(a)+len(e))" \
         /Users/abaybektursun/projects/interp_research-neurons-explain/bank/w1/o7_split/audit_poles.json \
         /Users/abaybektursun/projects/interp_research-neurons-explain/bank/w1/o7_split/extra_poles.json "$POLES" ;;
  *) echo "unknown cell $CELL" >&2; exit 2 ;;
esac
mkdir -p "$EB"
# frozen models: explainer, second reader, writer, job writer on anthropic/claude-opus-5-5 (JOURNAL 2026-10-20 08:20Z, 08:38Z)
EXPL_SETTINGS=$SETTINGS_DIR/deepinfra_opus_explainer_writer.json EXPL_MODEL=anthropic/claude-opus-5-5 \
WRITER_SETTINGS=$SETTINGS_DIR/deepinfra_opus_explainer_writer.json WRITER_MODEL=anthropic/claude-opus-5-5 \
  bash "$WAVE" "$H" "$CORPUS" "$EB/$CELL" c1 "$POLES" 0
