#!/usr/bin/env bash
# Launch the DEPTHOPEN recon seeder (see seed_depthopen_recon2.sh for what it reads and why CPU).
# Ephemeral + self-terminating: shutdown-behavior=terminate plus an EventBridge one-shot reaper.
# Run from the worktree root:  bash infra/aws_pg/seed_launch_depthopen_recon.sh
set -euo pipefail
P="fuelos"; R="us-east-1"
B=pg-lineage-769063704696
PFX=depthopen
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --profile $P --region $R --query Parameter.Value --output text)
BUDGET_S=2400   # 40 min; a meta-only dataset read and two HF listings.

# A stale marker makes any poll false-positive immediately (skill gotcha 7).
aws s3 rm s3://$B/$PFX/seed/DONE2 --profile $P --region $R || true

IID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type m7g.medium --count 1 \
  --iam-instance-profile Name=pg-instance-profile \
  --security-group-ids sg-0ff9d83db750a5a32 \
  --subnet-id subnet-01b382fad3f60144b \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":120,"VolumeType":"gp3","Throughput":500,"Iops":6000,"DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=app,Value=pg-lineage},{Key=Name,Value=pg-seed-depthopen2}]' \
  --user-data file://infra/aws_pg/seed_depthopen_recon2.sh \
  --profile $P --region $R --query 'Instances[0].InstanceId' --output text)
echo "launched $IID (m7g.medium, 1 vCPU, Standard quota)"

DEADLINE=$(python3 -c "import datetime as d;print((d.datetime.now(d.timezone.utc)+d.timedelta(seconds=$BUDGET_S+600)).strftime('%Y-%m-%dT%H:%M:%S'))")
aws scheduler create-schedule --name "pg-reap-$IID" \
  --schedule-expression "at($DEADLINE)" --schedule-expression-timezone UTC \
  --flexible-time-window Mode=OFF --action-after-completion DELETE \
  --target "{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:terminateInstances\",\"RoleArn\":\"arn:aws:iam::769063704696:role/pg-scheduler-role\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"$IID\\\"]}\"}" \
  --profile $P --region $R >/dev/null
echo "reaper armed: pg-reap-$IID at $DEADLINE UTC"
echo "progress:  aws s3 cp s3://$B/$PFX/seed/seed2.log - --profile $P --region $R | tail -40"
echo "recon ready: aws s3 ls s3://$B/$PFX/recon/v1/ --profile $P --region $R"
echo "done when: aws s3 ls s3://$B/$PFX/seed/DONE2 --profile $P --region $R"
