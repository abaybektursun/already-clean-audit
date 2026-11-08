#!/usr/bin/env bash
# Launch the CPU weights seeder (see seed_weights_cpu.sh for why CPU / why m7g.medium).
# Ephemeral + self-terminating: shutdown-behavior=terminate plus an EventBridge one-shot reaper.
# Run from the repo root:  bash infra/aws_pg/seed_launch_cpu.sh
set -euo pipefail
P="fuelos"; R="us-east-1"
B=pg-lineage-769063704696
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --profile $P --region $R --query Parameter.Value --output text)
BUDGET_S=21600   # 6 h hard cap; the seed is expected to take ~1-2 h on 1 vCPU.

# A stale marker makes any poll false-positive immediately (skill gotcha 7).
# STAGEDIFF (2026-09-04): repo list + S3 keys are launcher-side. SEED_REPOS is a comma list;
# SEED_TAG names the prefix (seed/<tag>/{DONE,seed.log}); the defaults reproduce the original run.
SEED_REPOS="${SEED_REPOS:-allenai/Molmo2-4B,allenai/Molmo2-ER,allenai/MolmoAct2}"
SEED_TAG="${SEED_TAG:-}"
KEYDIR="seed${SEED_TAG:+/$SEED_TAG}"
UD=$(mktemp /tmp/pg_seed_ud.XXXXXX)
python3 - "$SEED_REPOS" "$KEYDIR" > "$UD" <<'PY'
import sys, pathlib
repos, keydir = sys.argv[1].split(","), sys.argv[2]
s = pathlib.Path("infra/aws_pg/seed_weights_cpu.sh").read_text()
assert (s.count("__REPOS__"), s.count("__DONE_KEY__"), s.count("__LOG_KEY__")) == (1, 1, 2)
s = s.replace("__REPOS__", "\n".join(r.strip() for r in repos if r.strip()))
s = s.replace("__DONE_KEY__", f"{keydir}/DONE").replace("__LOG_KEY__", f"{keydir}/seed.log")
sys.stdout.write(s)
PY
echo "seeding: $SEED_REPOS  -> s3://$B/$KEYDIR/{DONE,seed.log}"
aws s3 rm s3://$B/$KEYDIR/DONE --profile $P --region $R || true

IID=$(aws ec2 run-instances \
  --image-id "$AMI" --instance-type m7g.medium --count 1 \
  --iam-instance-profile Name=pg-instance-profile \
  --security-group-ids sg-0ff9d83db750a5a32 \
  --subnet-id subnet-01b382fad3f60144b \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":300,"VolumeType":"gp3","Throughput":500,"Iops":6000,"DeleteOnTermination":true}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=app,Value=pg-lineage},{Key=Name,Value=pg-seed-cpu}]' \
  --user-data "file://$UD" \
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
echo "progress:  aws s3 cp s3://$B/$KEYDIR/seed.log - --profile $P --region $R | tail -40"
