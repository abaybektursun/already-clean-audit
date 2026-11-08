#!/usr/bin/env bash
# Deploy the server-side backstop reaper: SSM key, IAM, Lambda, 5-min schedule.
# Idempotent — re-run after editing reaper_lambda.py or policy.py.
#   bash deploy_reaper.sh          deploy/update everything
#   bash deploy_reaper.sh invoke   run it once now and print its log
set -euo pipefail
cd "$(dirname "$0")"
REGION=us-east-1 PROFILE=fuelos FN=shackleton-reaper
A() { aws --profile "$PROFILE" --region "$REGION" "$@"; }
ACCT=$(A sts get-caller-identity --query Account --output text)

if [[ "${1:-}" == "invoke" ]]; then
  A lambda invoke --function-name $FN --log-type Tail \
    --query LogResult --output text /tmp/reaper_out.json | base64 -d
  echo "--- return value ---"; cat /tmp/reaper_out.json; echo
  exit 0
fi

A ssm put-parameter --name /shackleton/vast-api-key --type SecureString \
  --value "$(cat ~/.config/vastai/vast_api_key)" --overwrite >/dev/null

NEW_ROLE=0
if ! A iam get-role --role-name $FN-role >/dev/null 2>&1; then
  A iam create-role --role-name $FN-role --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  NEW_ROLE=1
fi
A iam put-role-policy --role-name $FN-role --policy-name $FN-policy --policy-document '{
  "Version":"2012-10-17","Statement":[
   {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"},
   {"Effect":"Allow","Action":["s3:GetObject","s3:ListBucket"],
    "Resource":["arn:aws:s3:::pg-lineage-769063704696","arn:aws:s3:::pg-lineage-769063704696/*"]},
   {"Effect":"Allow","Action":"ssm:GetParameter",
    "Resource":"arn:aws:ssm:us-east-1:'$ACCT':parameter/shackleton/vast-api-key"}]}'

Z=/tmp/reaper_bundle.zip; rm -f "$Z"
zip -j -q "$Z" reaper_lambda.py policy.py

if A lambda get-function --function-name $FN >/dev/null 2>&1; then
  A lambda update-function-code --function-name $FN --zip-file fileb://"$Z" >/dev/null
  echo "[deploy_reaper] code updated"
else
  if [[ "$NEW_ROLE" == 1 ]]; then sleep 12; fi   # IAM propagation
  A lambda create-function --function-name $FN --runtime python3.12 \
    --handler reaper_lambda.handler --timeout 90 --memory-size 256 \
    --role "arn:aws:iam::$ACCT:role/$FN-role" --zip-file fileb://"$Z" >/dev/null
  echo "[deploy_reaper] function created"
fi
FN_ARN=$(A lambda get-function --function-name $FN \
  --query Configuration.FunctionArn --output text)

NEW_SCHED_ROLE=0
if ! A iam get-role --role-name $FN-sched-role >/dev/null 2>&1; then
  A iam create-role --role-name $FN-sched-role --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"scheduler.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  NEW_SCHED_ROLE=1
fi
A iam put-role-policy --role-name $FN-sched-role --policy-name $FN-sched-policy \
  --policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
   "Action":"lambda:InvokeFunction","Resource":"'"$FN_ARN"'"}]}'
if [[ "$NEW_SCHED_ROLE" == 1 ]]; then sleep 12; fi

SCHED='{"Name":"'$FN'-5min","ScheduleExpression":"rate(5 minutes)",
 "FlexibleTimeWindow":{"Mode":"OFF"},
 "Target":{"Arn":"'"$FN_ARN"'","RoleArn":"arn:aws:iam::'$ACCT':role/'$FN'-sched-role"}}'
if A scheduler get-schedule --name $FN-5min >/dev/null 2>&1; then
  A scheduler update-schedule --cli-input-json "$SCHED" >/dev/null
else
  A scheduler create-schedule --cli-input-json "$SCHED" >/dev/null
fi
echo "[deploy_reaper] $FN live on a 5-min schedule. Smoke: bash deploy_reaper.sh invoke"
