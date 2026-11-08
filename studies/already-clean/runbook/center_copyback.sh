#!/bin/bash
# Copy the center-probe rental's banked outputs back to the results prefix and pull the probe files locally.
# usage: center_copyback.sh        (after the rental's _rental_complete marker; the instance self-destroys)
set -euo pipefail
export AWS_PROFILE=fuelos AWS_REGION=us-east-1
B=s3://pg-lineage-769063704696
RENTAL=$B/results/mdl/mendeleev_neurons_x_center_probe
DST=$B/results/neurons/x/center_probe/1/out
LOCAL=/Users/abaybektursun/projects/interp_research-neurons-harvest/bank/neurons/x/center_probe/1/out
aws s3 cp $RENTAL/_rental_complete.json - | cut -c1-400; echo
aws s3 sync $RENTAL/cells/center_probe/ $DST/ --only-show-errors
echo "copied back: $(aws s3 ls --recursive $DST/ | wc -l | tr -d ' ') objects"
mkdir -p "$LOCAL"
aws s3 cp $DST/pipeline_status.json "$LOCAL/pipeline_status.json" --only-show-errors
aws s3 cp $DST/steps.json "$LOCAL/steps.json" --only-show-errors
for step in $(aws s3 ls $DST/ | awk '/PRE t/{print $2}' | tr -d /); do
  aws s3 cp $DST/$step/manifest.json "$LOCAL/$step/manifest.json" --only-show-errors
  aws s3 cp $DST/$step/metrics.json "$LOCAL/$step/metrics.json" --only-show-errors
  aws s3 cp $DST/$step/poles/poles.json "$LOCAL/$step/poles/poles.json" --only-show-errors
  aws s3 cp $DST/$step/poles/probe/scores.npz "$LOCAL/$step/poles/probe/scores.npz" --only-show-errors
  echo "$step: $(stat -f %z "$LOCAL/$step/poles/probe/scores.npz") bytes"
done
python3 -I -c "import json,sys; s=json.load(open(sys.argv[1])); print('pipeline:', s.get('state'), s.get('completed'), 'of', s.get('n_steps'))" "$LOCAL/pipeline_status.json"
echo "COPYBACK DONE $(date -u)"
