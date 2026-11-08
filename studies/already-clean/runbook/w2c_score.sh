#!/bin/bash
# W2-C scoring job for one cell (SPEC_D1 §9.2 W2-C): upload the wave's texts and probe poles, then launch
# the probe on AWS (committed manifest) or, when AWS has no capacity, on vast.ai (committed vast manifest).
# usage: w2c_score.sh <q4b|g4|o7> <aws|vast> [vast offer id]
set -euo pipefail
CELL=$1; LANE=$2; OFFER=${3:-}
export AWS_PROFILE=fuelos AWS_REGION=us-east-1
B=s3://pg-lineage-769063704696
E=/Users/abaybektursun/projects/interp_research-neurons-explain/bank/w2c/$CELL/explain
HW=/Users/abaybektursun/projects/interp_research-neurons-harvest
P=results/neurons/w2c/${CELL}_probe/1
M=results/mdl/neurons/w2c/${CELL}_probe/1
[ -s "$E/texts.jsonl" ] && [ -s "$E/probe_poles.json" ] || { echo "no texts/probe_poles for $CELL yet" >&2; exit 2; }
echo "texts $(wc -l < "$E/texts.jsonl" | tr -d ' ')  probe poles $(python3 -I -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$E/probe_poles.json")"
for f in texts.jsonl probe_poles.json; do
  aws s3 cp "$E/$f" "$B/$P/$f" --only-show-errors
  aws s3 cp "$E/$f" "$B/$M/$f" --only-show-errors
done
aws s3 ls "$B/$P/steps.json" >/dev/null && aws s3 ls "$B/$M/steps.json" >/dev/null
cd "$HW"; [ -z "$(git status --short)" ] || { echo "harvest worktree not clean" >&2; exit 3; }
case $LANE in
  aws)
    case $CELL in q4b) W="--weights models--Qwen--Qwen3-4B-Base" ;; g4) W="--weights models--google--gemma-3-4b-pt" ;; o7) W="--no-weights" ;; esac
    ./infra/aws_pg/pg run infra.aws_pg.jobs.pipeline --args " --steps-s3 $B/$P/steps.json" --s3-prefix "$P" \
      --type g6e.2xlarge,g6e.4xlarge --budget-min 90 $W --detach --json | tail -1 ;;
  vast)
    [ -n "$OFFER" ] || { echo "vast lane needs an offer id" >&2; exit 2; }
    cd infra/vast && python3 controller.py rent "manifests/neurons_w2c_${CELL}_probe.json" "$OFFER" 2>&1 \
      | sed -E 's/(hf_|AWS_SECRET_ACCESS_KEY=|VAST_API_KEY=)[A-Za-z0-9+\/_-]+/\1<redacted>/g' | tail -3 ;;
  *) echo "lane must be aws or vast" >&2; exit 2 ;;
esac
echo "LAUNCHED $CELL on $LANE $(date -u)"
