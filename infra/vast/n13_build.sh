#!/usr/bin/env bash
# Build the N13 image (shackleton:n13-v1) on the lane's ephemeral EC2 builder (launch_builder.sh -> build_image.sh).
# Layers on top of shackleton:mdl-v1 (pytorch/pytorch 2.12.0 cu130, Ubuntu python 3.12, the AWS pg-env AMI pins):
#   apt gcc + python3.12-dev + python3.12-venv (Triton builds its CUDA launcher with gcc against Python.h);
#   /opt/n13 = exactly the venv that ran on AWS (pip install --no-deps of its freeze, then pip check);
#   the three revisions n13_setup.MODELS names for base, stage2 and rl, baked in HF-cache layout at /opt/hf-cache/hf
#   (the AWS lane's HF_HOME), one layer per repo;
#   VLLM_USE_FLASHINFER_SAMPLER=0 (A21) as image ENV.
# The in-image gate (n13_gate.sh) runs before the real tag is pushed. Revisions are read from n13_setup.py, not typed here.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
MODELS=$(python3 - "$ROOT/infra/aws_pg/jobs/n13_setup.py" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
m = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "MODELS")
print(" ".join(f"{m[k][0]}@{m[k][1]}" for k in ("base", "stage2", "rl")))
PY
)
echo "[n13_build] models: $MODELS"
TAG="${TAG:-n13-v1}" \
BASE=769063704696.dkr.ecr.us-east-1.amazonaws.com/shackleton@sha256:f2975abb9a712bb367039c0aded95bffe21e89527cd1b515b964ce6d12c73035 \
BAKE_MODE=cache MODELS="$MODELS" PIP_PKGS=- \
APT_PKGS="gcc python3.12-dev python3.12-venv" \
VENV_FREEZE="/opt/n13=studies/next/n13/r0/s2setup-07/setup/pip_freeze_n13.txt" \
HF_CACHE_DEST=/opt/hf-cache/hf \
IMAGE_ENV="VLLM_USE_FLASHINFER_SAMPLER=0" \
GATE=n13 N13_GATE_S3=s3://pg-lineage-769063704696/code/shackleton/n13_gate_inputs/20261105 \
CODE_PATHS="studies/reproductions/alwayson/code studies/next/n13/scripts studies/next/n13/r0/s2setup-06/path_check_tokens.json studies/next/n13/r0/s2setup-07/setup/pip_freeze_n13.txt studies/next/07_PICKS/N13_MATH_TASK.md" \
DISK_GB="${DISK_GB:-400}" \
bash launch_builder.sh
