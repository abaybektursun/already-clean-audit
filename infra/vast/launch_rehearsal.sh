#!/usr/bin/env bash
# Run the §9.1 rehearsal on an ephemeral self-terminating EC2 box (never the
# local machine — house rule). Uploads its log to S3 and terminates itself.
# Usage: bash launch_rehearsal.sh   (watch: s3 ls .../code/shackleton/rehearsal_logs/)
set -euo pipefail
ITYPE="${ITYPE:-c7i.2xlarge}"
AWS=(aws --profile fuelos --region us-east-1)
S3_CODE="s3://pg-lineage-769063704696/code/shackleton"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

SHA=$(git -C "$ROOT" rev-parse --short HEAD)
git -C "$ROOT" archive --format=tar.gz HEAD infra > "/tmp/repo_${SHA}.tar.gz"
"${AWS[@]}" s3 cp "/tmp/repo_${SHA}.tar.gz" "$S3_CODE/repo_${SHA}.tar.gz" > /dev/null

AKID=$(aws configure get aws_access_key_id --profile fuelos)
ASEC=$(aws configure get aws_secret_access_key --profile fuelos)
AMI=$("${AWS[@]}" ssm get-parameter --name \
  /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)

UD=$(cat <<EOF
#!/bin/bash
set -euxo pipefail
exec > /var/log/shackleton_rehearsal.log 2>&1
export AWS_ACCESS_KEY_ID=$AKID AWS_SECRET_ACCESS_KEY=$ASEC AWS_DEFAULT_REGION=us-east-1
apt-get update && apt-get install -y docker.io python3-pip python3-venv unzip
curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/a.zip
unzip -q /tmp/a.zip -d /tmp && /tmp/aws/install
mkdir -p /run_rehearsal && cd /run_rehearsal
aws s3 cp "$S3_CODE/repo_${SHA}.tar.gz" repo.tar.gz && tar xzf repo.tar.gz
bash infra/vast/rehearse.sh && STATUS=pass || STATUS=fail
aws s3 cp /var/log/shackleton_rehearsal.log "$S3_CODE/rehearsal_logs/${SHA}_\$STATUS.log"
shutdown -h now
EOF
)

"${AWS[@]}" ec2 run-instances \
  --image-id "$AMI" --instance-type "$ITYPE" \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=shackleton-rehearsal-$SHA}]" \
  --user-data "$UD" \
  --query 'Instances[0].InstanceId' --output text
echo "[rehearsal] launched at code $SHA; result lands in $S3_CODE/rehearsal_logs/${SHA}_{pass|fail}.log"
