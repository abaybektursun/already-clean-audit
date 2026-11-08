"""Static config for the pg AWS launcher.

fuelos / us-east-1 are hardcoded by design: the fuelos profile has no default region,
so an accidental omission must fail loud rather than hit the wrong account/region.
"""

PROFILE = "fuelos"
REGION = "us-east-1"
ACCOUNT = "769063704696"

BUCKET = "pg-lineage-769063704696"
INSTANCE_PROFILE = "pg-instance-profile"
SCHEDULER_ROLE_ARN = "arn:aws:iam::769063704696:role/pg-scheduler-role"
SECURITY_GROUP = "sg-0ff9d83db750a5a32"

# default-VPC subnets by AZ; the launcher tries them in order on InsufficientInstanceCapacity.
SUBNETS = [
    "subnet-01b382fad3f60144b",  # us-east-1d
    "subnet-05cce85fb54f95bee",  # us-east-1b
    "subnet-05facf28bddb82218",  # us-east-1a
    "subnet-0dfd68be6bc2702a0",  # us-east-1c
    "subnet-04561d65e5e200ae5",  # us-east-1e
    "subnet-0f8f8a2f0d08df4b5",  # us-east-1f
]

# Instance-type PREFERENCE LIST (the launcher tries them in order, taking the first with
# capacity — see pg._launch). Each entry must have ENOUGH VRAM for the job; the launcher never
# silently drops to a smaller GPU. MENDELEEV (spec §11.1, D26): g6e.2xlarge (1×L40S 48 GB, 8 vCPU,
# 64 GB host RAM) so up to 4 run at once under the 32-vCPU G quota; g6e.4xlarge as the capacity
# fallback. g6e.xlarge is excluded (32 GB host RAM cannot hold two 8B copies); the old default
# g6e.8xlarge took the whole quota and p5.4xlarge hit a banned type. For ≥80GB jobs pass
# --type p5.4xlarge,p4de.24xlarge explicitly.
INSTANCE_TYPES = ["g6e.2xlarge", "g6e.4xlarge"]
INSTANCE_TYPE = INSTANCE_TYPES[0]   # back-compat single default
# Live-cost estimate for `pg ps`. None => show uptime only (verify the real rate, then set).
ONDEMAND_USD_PER_HR = None

# Filled in after the bake / seed steps (see bake_env.sh / seed_weights.sh):
AMI_ID = "ami-0cee78ee4b1c24614"  # pg-env-v1: py3.12 venv + torch 2.12+cu130 + pinned stack
# Weights live in S3 (3B model set in HF-cache layout) and are synced to local disk at launch.
# Chosen over an EBS snapshot: snapshot volumes lazy-load blocks from S3 on first read, and
# safetensors' random reads make that brutally slow (~60 min cold). A sequential `aws s3 sync`
# to local disk runs at network speed (~3 min) on ANY instance/AZ, and is fanout-safe.
WEIGHTS_S3_URI = "s3://pg-lineage-769063704696/weights/hf/"

# The env AMI's root is only 30GB (DLAMI default) — too small for the weights sync + scratch.
# Resize the root at launch; Ubuntu cloud-init growpart auto-expands the filesystem on boot.
ROOT_DEVICE = "/dev/sda1"
ROOT_GB = 200
