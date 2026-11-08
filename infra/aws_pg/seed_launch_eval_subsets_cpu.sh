#!/usr/bin/env bash
# Launch the CPU EVAL-SUBSET seeder (see seed_eval_subsets_cpu.sh for what it builds and why).
# Same shape as seed_launch_datasets_cpu.sh: ephemeral + self-terminating, shutdown-behavior=terminate
# plus an EventBridge one-shot reaper, on the Standard quota (1 vCPU) so the G quota stays for the
# science. This launcher does not touch any running job.
#   bash infra/aws_pg/seed_launch_eval_subsets_cpu.sh
set -euo pipefail
P="fuelos"; R="us-east-1"
B=pg-lineage-769063704696
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --profile $P --region $R --query Parameter.Value --output text)
# 4 h. The long pole is restoring the 181.8 GB of seeded image splits once; the select + upload is
# ~100 MB and the probe is seconds.
BUDGET_S=14400

# The science code the seeder imports (matrix_items needs numpy only). Working tree, not HEAD — the
# same convention pg.py uses, so what runs is what is on disk here.
tar czf /tmp/pg_subsets_code.tgz -C "$REPO" infra
aws s3 cp /tmp/pg_subsets_code.tgz s3://$B/seed/eval_subsets_code.tgz --profile $P --region $R

# A stale marker makes any poll false-positive immediately (skill gotcha 7).
aws s3 rm s3://$B/datasets/eval_subsets/DONE --profile $P --region $R || true

IID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type m7g.medium --count 1 \
  --iam-instance-profile Name=pg-instance-profile \
  --security-group-ids sg-0ff9d83db750a5a32 \
  --subnet-id subnet-01b382fad3f60144b \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Throughput":500,"Iops":6000,"DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=app,Value=pg-lineage},{Key=Name,Value=pg-seed-eval-subsets}]' \
  --user-data file://infra/aws_pg/seed_eval_subsets_cpu.sh \
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
echo "progress:  aws s3 cp s3://$B/datasets/eval_subsets/seed.log - --profile $P --region $R | tail -40"
echo "done when: aws s3 ls s3://$B/datasets/eval_subsets/DONE --profile $P --region $R"
