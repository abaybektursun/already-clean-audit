#!/bin/bash
# Resume W4 on S and run W5 once the provider accepts calls again (SPEC_D1 §9.2 W4–W5; handoff/S5.md).
# While the provider refuses with HTTP 402 (the user-set limit) an attempt costs nothing; the chain retries every INTERVAL
# seconds, at most MAX_TRIES times. Any other failure stops the script. It ends after RESULTS.md; the §11 grid and the
# closeout entry are written by hand from bank/w5.
# usage: NEURONS_API_KEY_1 in the environment (source ~/.config/interp_research/keys.env; never print it); w4_w5_resume.sh
set -euo pipefail
R=$(cd "$(dirname "$0")" && pwd)
E=/Users/abaybektursun/projects/interp_research-neurons-explain
HW=/Users/abaybektursun/projects/interp_research-neurons-harvest
H=$HW/bank/neurons; G=$H/w3/grid; J=$R/../JOURNAL.md
INTERVAL=${INTERVAL:-600}; MAX_TRIES=${MAX_TRIES:-144}
export SET=$R/settings/deepinfra_w4_sonnet_judge.json CLUSTER_MODEL=anthropic/claude-sonnet-5-5 JUDGE_MODEL=anthropic/claude-sonnet-5-5
export S=bank/w4/o7_s_v2 GATE=bank/w4/o7_gate_v2_sonnet_js SEED_SC=4111 SEED_SJ=4112 CAP=8
[ -n "${NEURONS_API_KEY_1:-}" ] || { echo "NEURONS_API_KEY_1 not in environment" >&2; exit 2; }
PYV="/Users/abaybektursun/.local/bin/uv run --no-project --with transformers --with tokenizers --with anthropic --with numpy python -m"
TMP=$(mktemp)
until_allowed() {  # until_allowed <w4_chain.sh mode>: retry only on the provider's HTTP 402
  local try=0
  while :; do
    if "$R/w4_chain.sh" "$1" > "$TMP" 2>&1; then cat "$TMP"; return 0; fi
    if ! grep -q 'HTTP 402' "$TMP"; then cat "$TMP" >&2; echo "w4_chain.sh $1 failed for a reason other than the limit" >&2; return 1; fi
    try=$((try+1)); [ "$try" -lt "$MAX_TRIES" ] || { echo "provider limit still in force after $try tries" >&2; return 3; }
    echo "$1: provider limit in force (try $try, $(date -u)); next attempt in ${INTERVAL}s"; sleep "$INTERVAL"
  done
}
until_allowed s-clusters
# per-cluster device check; rule: more than 5% of clusters flipping presence re-harvests job b before unblinding
cd "$HW"
$PYV infra.pg.neurons_device_check --a bank/neurons/w3/grid/1b/out/t14_step1000000 --b bank/neurons/w3/grid/1c/out/t14_step1000000 \
  --poles "$E/bank/w3/o7/draw/grid_poles.json" --clusters "$E/$S/explain/clusters.jsonl" --out bank/neurons/w3/device_check/t14_1b_vs_1c_clusters.json
python3 -I -c "import json,sys; c=json.load(open(sys.argv[1]))['clusters']; s={r:c[r]['flip_share'] for r in ('percentile','raw_ratio')}; print('cluster flip shares', s); sys.exit(4 if max(s.values())>0.05 else 0)" bank/neurons/w3/device_check/t14_1b_vs_1c_clusters.json
until_allowed s-judge
# W5
cd "$E"
GRID=$(ls -d "$G"/1/out/t0[0-6]_step* | paste -sd, -),$(ls -d "$G"/1b/out/t*_step* | paste -sd, -),$G/1/out/t15_step1413814
$PYV infra.pg.neurons_verdict unblind --journal "$J" --theta bank/w1/q4b_disc_v3/verdict/theta.json \
  --q1 "Q4B=$H/w1/q4b/1/out/q4b,$H/w2c/q4b_probe/1/out/probe,bank/w2c/q4b" \
  --q1 "G4=$H/w1/g4/1/out/g4,$H/w2c/g4_probe/1/out/probe,bank/w2c/g4" \
  --q1 "O7=$H/w1/o7_t15/1/out/o7_t15,$H/w2c/o7_probe/1/out/probe,bank/w2c/o7,poles=bank/w1/o7_split/audit_poles.json" \
  --q2-strata O7 --judge-gate "$GATE/verdict/judge_gate.json" --q2-grid "$GRID" --q2-explain "$S" --b2 \
  --t0-explanations bank/w4/o7_t0_g2 --seed 0 --out bank/w5
$PYV infra.pg.neurons_figures --unblind bank/w5 --out bank/w5/figures
$PYV infra.pg.neurons_verdict results-template --unblind bank/w5 --out bank/w5/RESULTS.md
AWS_PROFILE=fuelos AWS_REGION=us-east-1 aws s3 sync bank/w5 s3://pg-lineage-769063704696/results/neurons/w5/ --only-show-errors
echo "W4 S AND W5 DONE $(date -u)"
