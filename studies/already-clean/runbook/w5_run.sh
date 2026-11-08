#!/bin/bash
# W5 after the S judge is complete (handoff/S5.md step 4; same commands as runbook/w4_w5_resume.sh's W5 block, no API step).
# usage: w5_run.sh collect | unblind | figures | results | all      (run from anywhere; cds into the explain worktree)
#        CENTER=<per-checkpoint probe out dir, e.g. $HW/bank/neurons/x/center_probe/1/out> adds --center-probe (§2.4 center birth curve)
#        all = collect, unblind, figures, results. collect refuses once $S is collected ("ids already in .../calls.jsonl"), and
#        set -e stops all there. unblind, figures and results each refuse an existing output in bank/w5. To re-derive bank/w5,
#        move it aside and run unblind, figures, results (handoff/S6.md, Resume steps).
set -euo pipefail
E=/Users/abaybektursun/projects/interp_research-neurons-explain
HW=/Users/abaybektursun/projects/interp_research-neurons-harvest
H=$HW/bank/neurons; G=$H/w3/grid
J=/Users/abaybektursun/projects/interp_research-drafts/studies/already-clean/JOURNAL.md
S=bank/w4/o7_s_v2; GATE=bank/w4/o7_gate_v2_sonnet_js
PYV="/Users/abaybektursun/.local/bin/uv run --no-project --with transformers --with tokenizers --with anthropic --with numpy python -m"
q() { "$@" 2> >(grep -v 'PyTorch was not found' >&2); }
cd "$E"
step=${1:-all}
if [ "$step" = collect ] || [ "$step" = all ]; then
  q $PYV infra.pg.neurons_explain collect-judge --out "$S" --batch w4j --backend mixed
  echo "COLLECT DONE $(date -u)"
fi
if [ "$step" = unblind ] || [ "$step" = all ]; then
  GRID=$(ls -d "$G"/1/out/t0[0-6]_step* | paste -sd, -),$(ls -d "$G"/1b/out/t*_step* | paste -sd, -),$G/1/out/t15_step1413814
  [ "$(echo "$GRID" | tr ',' '\n' | wc -l | tr -d ' ')" = 16 ] || { echo "GRID is not 16 dirs" >&2; exit 2; }
  q $PYV infra.pg.neurons_verdict unblind --journal "$J" --theta bank/w1/q4b_disc_v3/verdict/theta.json \
    --q1 "Q4B=$H/w1/q4b/1/out/q4b,$H/w2c/q4b_probe/1/out/probe,bank/w2c/q4b" \
    --q1 "G4=$H/w1/g4/1/out/g4,$H/w2c/g4_probe/1/out/probe,bank/w2c/g4" \
    --q1 "O7=$H/w1/o7_t15/1/out/o7_t15,$H/w2c/o7_probe/1/out/probe,bank/w2c/o7,poles=bank/w1/o7_split/audit_poles.json" \
    --q2-strata O7 --judge-gate "$GATE/verdict/judge_gate.json" --q2-grid "$GRID" --q2-explain "$S" --b2 \
    --t0-explanations bank/w4/o7_t0_g2 --seed 0 --out bank/w5 ${CENTER:+--center-probe "$CENTER"}
  echo "UNBLIND DONE $(date -u)"
fi
if [ "$step" = figures ] || [ "$step" = all ]; then
  q $PYV infra.pg.neurons_figures --unblind bank/w5 --out bank/w5/figures
  echo "FIGURES DONE $(date -u)"
fi
if [ "$step" = results ] || [ "$step" = all ]; then
  q $PYV infra.pg.neurons_verdict results-template --unblind bank/w5 --out bank/w5/RESULTS.md
  echo "RESULTS DONE $(date -u)"
fi
