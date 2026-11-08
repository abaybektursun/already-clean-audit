#!/usr/bin/env bash
# Env-AMI bake provisioning (runs as root via EC2 user-data on a DLAMI builder).
# Produces a clean venv at /opt/pg/venv with the EXACT pinned stack from the Modal
# image (infra/image_base.py + infra/modal_pg.py). Weights are NOT baked here — they
# live in a separate EBS snapshot (see seed_weights.sh). Reports to S3; on success
# writes the DONE marker and stops (instance-initiated-shutdown-behavior=stop) so the
# launcher can create-image from the stopped builder.
exec > /var/log/pg-bake.log 2>&1
B=pg-lineage-769063704696
up() { aws s3 cp /var/log/pg-bake.log s3://$B/bake/env_bake.log --region us-east-1; }
# always upload the log AND stop the builder on ANY exit (success or failure), so a failed
# bake can never sit running and bill (shutdown-behavior=stop -> stopped, imageable, ~$0).
trap 'up; shutdown -h now' EXIT
set -ex

python3 --version
# DLAMI base python lacks ensurepip; install venv support before creating the clean venv.
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.12-venv
python3 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip

# torch/torchvision: PyPI default == cu130 (Blackwell-ready, fine on H100 sm_90 / A10G sm_86) — matches Modal.
/opt/pg/venv/bin/pip install torch==2.12.0 torchvision==0.27.0

# research stack — exact pins from infra/image_base.py + infra/modal_pg.py
/opt/pg/venv/bin/pip install \
  "transformers>=4.47,<6" "accelerate>=1.10,<2" "safetensors>=0.5,<1" numpy Pillow \
  huggingface_hub==1.16.1 nnsight==0.7.0 datasets==4.8.5 \
  scikit-learn==1.8.0 scipy==1.17.1 einops==0.8.2 peft==0.19.1 matplotlib wandb \
  tiktoken protobuf sentencepiece

# tiktoken / protobuf / sentencepiece: transformers imports these ONLY when a repo ships a tiktoken or
# sentencepiece vocabulary, so their absence is invisible until such a row runs. On 2026-09-04 that cost
# 32 cells across two rows — 7a (zephyr-7b-beta -> Zephyr_RMU) on `ModuleNotFoundError: tiktoken` and
# 3c-Act (MolmoAct2) on the protobuf equivalent — each after the job had already loaded weights.
# `run_job.sh` installs them at job start for AMIs baked before this line; once this AMI ships, that
# check finds them present and does nothing.

# BatchTopK crosscoder fork — --no-deps (its nnsight pin conflicts with 0.7), same as Modal.
/opt/pg/venv/bin/pip install --no-deps git+https://github.com/jkminder/dictionary_learning.git

# fail-loud import + real-GPU smoke
/opt/pg/venv/bin/python - <<'PY'
import torch, transformers, peft, nnsight, datasets, sklearn, scipy, einops
assert torch.cuda.is_available(), "CUDA not available on builder GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "peft", peft.__version__, "nnsight", nnsight.__version__, "datasets", datasets.__version__)
import dictionary_learning
print("dictionary_learning OK")
print("IMPORTS_OK")
PY

/opt/pg/venv/bin/pip freeze > /opt/pg/pip-freeze.txt
aws s3 cp /opt/pg/pip-freeze.txt s3://$B/bake/pip-freeze.txt --region us-east-1
date -u > /tmp/DONE
aws s3 cp /tmp/DONE s3://$B/bake/DONE --region us-east-1
up
# shutdown handled by the EXIT trap
