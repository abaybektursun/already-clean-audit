#!/usr/bin/env bash
# Launch the CPU DATASET seeder (see seed_datasets_cpu.sh for why CPU / why m7g.medium / what it
# seeds). Same shape as seed_launch_cpu.sh: ephemeral + self-terminating, shutdown-behavior=terminate
# plus an EventBridge one-shot reaper, on the Standard quota so the G quota stays for the science.
# Run from the repo root:  bash infra/aws_pg/seed_launch_datasets_cpu.sh
set -euo pipefail
P="fuelos"; R="us-east-1"
B=pg-lineage-769063704696
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --profile $P --region $R --query Parameter.Value --output text)
# 8 h. The download is fast (Xet, ~500 MB/s measured on this shape); the long pole is the arrow
# build of the image splits on ONE vCPU, which is why this is not the weights seeder's 6 h.
BUDGET_S=28800

# A stale marker makes any poll false-positive immediately (skill gotcha 7).
aws s3 rm s3://$B/datasets/DONE --profile $P --region $R || true

IID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type m7g.medium --count 1 \
  --iam-instance-profile Name=pg-instance-profile \
  --security-group-ids sg-0ff9d83db750a5a32 \
  --subnet-id subnet-01b382fad3f60144b \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Throughput":500,"Iops":6000,"DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=app,Value=pg-lineage},{Key=Name,Value=pg-seed-datasets}]' \
  --user-data file://infra/aws_pg/seed_datasets_cpu.sh \
  --profile $P --region $R --query 'Instances[0].InstanceId' --output text)
echo "launched $IID (m7g.medium, 1 vCPU, Standard quota)"

# AWS-side backstop: one-shot self-deleting schedule -> TerminateInstances. The pg-scheduler-role
# is conditioned to app=pg-lineage, which this instance carries.
DEADLINE=$(python3 -c "import datetime as d;print((d.datetime.now(d.timezone.utc)+d.timedelta(seconds=$BUDGET_S+600)).strftime('%Y-%m-%dT%H:%M:%S'))")
aws scheduler create-schedule --name "pg-reap-$IID" \
  --schedule-expression "at($DEADLINE)" --schedule-expression-timezone UTC \
  --flexible-time-window Mode=OFF --action-after-completion DELETE \
  --target "{\"Arn\":\"arn:aws:scheduler:::aws-sdk:ec2:terminateInstances\",\"RoleArn\":\"arn:aws:iam::769063704696:role/pg-scheduler-role\",\"Input\":\"{\\\"InstanceIds\\\":[\\\"$IID\\\"]}\"}" \
  --profile $P --region $R >/dev/null
echo "reaper armed: pg-reap-$IID at $DEADLINE UTC"
echo "progress:  aws s3 cp s3://$B/datasets/seed.log - --profile $P --region $R | tail -40"
echo "done when: aws s3 ls s3://$B/datasets/DONE --profile $P --region $R"
