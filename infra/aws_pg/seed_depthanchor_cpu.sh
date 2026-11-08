#!/usr/bin/env bash
# DEPTHANCHOR seed (EC2 user-data, runs as root on Amazon Linux 2023 arm64, m7g.medium).
#
# WHY A NEW SCRIPT and not `seed_weights_cpu.sh` with a repo list: this study needs four different
# kinds of object seeded, three of which that script cannot express —
#   1. a full model snapshot            -> weights/hf/            (allenai/MolmoAct2-Think-LIBERO, 21.8 GB)
#   2. TWO FILES out of a 33 GB repo    -> weights/hf/            (vae-final.pt 31 MB +
#      depth_anything_v2_vitb.pth 390 MB from allenai/MolmoAct-7B-D-0812; its seven safetensors
#      shards are NOT wanted and are 32 GB)
#   3. a 23.8 GB LeRobot-v3 dataset     -> depthanchor/liberods/v1/  (plain path tree, not an HF cache:
#      the job reads the parquet files by path, so nothing depends on cache-key semantics)
#   4. the checkpoint's REMOTE CODE     -> depthanchor/code/v1/      (the six .py/.json text files, so the
#      science can be written against the real `predict_action`, read on the control plane, not from
#      memory; uploaded FIRST, in the first minute, because everything else waits on it)
#
# Same operational shape as the other CPU seeders and for the same reasons (see seed_weights_cpu.sh):
# download-then-upload needs no GPU and must not eat the 32-vCPU G quota the science fills; the
# Standard on-demand quota is 1 vCPU, so m7g.medium is the only shape that fits; 8 GB of swap because
# the OOM killer took out a multi-shard fetch on this shape once and said only "Killed"; sync to S3
# after every stage so a mid-run failure banks what is already finished; self-terminates via the EXIT
# trap plus InstanceInitiatedShutdownBehavior=terminate, with an EventBridge reaper as the backstop.
exec > /var/log/pg-seed-da.log 2>&1
B=pg-lineage-769063704696
PFX=depthanchor
up() { aws s3 cp /var/log/pg-seed-da.log s3://$B/$PFX/seed/seed.log --region us-east-1; }
trap 'up; shutdown -h now' EXIT
set -ex

( while true; do aws s3 cp /var/log/pg-seed-da.log s3://$B/$PFX/seed/seed.log --region us-east-1 >/dev/null 2>&1; sleep 30; done ) &

df -h /
fallocate -l 8G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
free -h

dnf install -y python3.11 python3.11-pip
python3.11 -m venv /opt/pg/venv
/opt/pg/venv/bin/pip install --upgrade pip
/opt/pg/venv/bin/pip install huggingface_hub

mkdir -p /opt/pg/hf /opt/pg/code /opt/pg/lds
export HF_HOME=/opt/pg/hf HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN=$(aws ssm get-parameter --name /pg-lineage/hf-token --with-decryption \
                  --region us-east-1 --query Parameter.Value --output text)

MODEL=allenai/MolmoAct2-Think-LIBERO
DEPTHREPO=allenai/MolmoAct-7B-D-0812
DATASET=allenai/MolmoAct2-LIBERO-Dataset

# ---- stage 1: the remote code, as a plain tree, uploaded immediately -------------------------
# local_dir= gives real files, not a symlinked cache, so `aws s3 cp --recursive` uploads a readable
# tree. These are the files §1 of the spec was written from; the control plane reads them from S3.
/opt/pg/venv/bin/python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(
    sys.argv[1],
    local_dir="/opt/pg/code/molmoact2_think_libero",
    allow_patterns=["*.py", "*.json", "*.jinja", "*.md", "assets/*"],
)
print("CODE_AT", p)
PY
aws s3 cp /opt/pg/code/ s3://$B/$PFX/code/v1/ --recursive --exclude '.cache/*' --exclude '*/.cache/*' --region us-east-1
echo "STAGE1_OK code uploaded"

# ---- stage 2: reference scripts from the two public repos ------------------------------------
# The VQ-VAE constructor (`reconstruct_from_tokens.py`, MolmoAct v1) and the annotation script that
# loads vae-final.pt + Depth-Anything-V2 (`generate_depth_annotation.py`, MolmoAct2) — G1 and G2 need
# both. Status is printed per URL rather than swallowed: a 404 must be visible in the log, and it is
# not fatal to the weight seed, which is what the science cannot proceed without.
mkdir -p /opt/pg/code/ref
for U in \
  "https://raw.githubusercontent.com/allenai/molmoact2/main/experiments/scripts/generate_depth_annotation.py" \
  "https://raw.githubusercontent.com/allenai/molmoact/main/scripts/reconstruct_from_tokens.py" \
  "https://raw.githubusercontent.com/allenai/molmoact2/main/experiments/README.md" \
  "https://raw.githubusercontent.com/allenai/molmoact2/main/launch_scripts/data_mixtures.py" ; do
  N=$(basename "$U")
  if curl -fsSL "$U" -o "/opt/pg/code/ref/$N"; then echo "REF_OK   $U"; else echo "REF_MISS $U"; fi
done
ls -la /opt/pg/code/ref
aws s3 cp /opt/pg/code/ref/ s3://$B/$PFX/code/v1/ref/ --recursive --region us-east-1
echo "STAGE2_OK refs uploaded"

# ---- stage 3: the checkpoint ------------------------------------------------------------------
# max_workers=1: 4 GB of RAM cannot hold eight concurrent 4.9 GB shard buffers (the lesson from the
# Molmo2-4B OOM). The bottleneck here is the network, so serialising costs almost nothing.
/opt/pg/venv/bin/python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("MODEL_AT", snapshot_download(sys.argv[1], max_workers=1))
PY
du -sh /opt/pg/hf; df -h /
aws s3 sync /opt/pg/hf/ s3://$B/weights/hf/ --region us-east-1
echo "STAGE3_OK model seeded: $MODEL"

# ---- stage 4: the two depth references out of the 7B repo -------------------------------------
# allow_patterns keeps the seven model-*.safetensors shards (32 GB) out. Nothing in this study loads
# the MolmoAct v1 policy; only its VQ-VAE decoder and the Depth-Anything-V2 checkpoint it shipped.
/opt/pg/venv/bin/python - "$DEPTHREPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("DEPTHREF_AT", snapshot_download(
    sys.argv[1], max_workers=1,
    allow_patterns=["vae-final.pt", "depth_anything_v2_vitb.pth", "*.json", "*.py", "*.yaml"],
))
PY
du -sh /opt/pg/hf; df -h /
aws s3 sync /opt/pg/hf/ s3://$B/weights/hf/ --region us-east-1
echo "STAGE4_OK depth references seeded: $DEPTHREPO"

# ---- stage 5: the LIBERO dataset as a plain path tree ----------------------------------------
# 23.8 GB, 378 data parquet files + meta. local_dir= so the tree in S3 mirrors the repo's own paths:
# the job syncs meta/ first (1 MB), reads which file-index holds each episode on its frozen list, and
# then syncs only those data files. An HF cache layout would buy nothing and cost the blob duplicate.
/opt/pg/venv/bin/python - "$DATASET" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("DATASET_AT", snapshot_download(
    sys.argv[1], repo_type="dataset", local_dir="/opt/pg/lds", max_workers=1))
PY
du -sh /opt/pg/lds; df -h /
aws s3 sync /opt/pg/lds/ s3://$B/$PFX/liberods/v1/ --exclude '.cache/*' --region us-east-1
echo "STAGE5_OK dataset seeded: $DATASET"

# ---- manifest + DONE -------------------------------------------------------------------------
# Provenance: what was actually seeded, with sizes, so a later reader never has to trust this log.
/opt/pg/venv/bin/python - <<'PY' > /tmp/manifest.json
import hashlib, json, os, pathlib, datetime
out = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "trees": {}}
for label, root in (("weights_hf", "/opt/pg/hf"), ("liberods", "/opt/pg/lds"), ("code", "/opt/pg/code")):
    files = {}
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            files[str(p.relative_to(root))] = p.stat().st_size
    out["trees"][label] = {"n_files": len(files), "bytes": sum(files.values()), "files": files}
for label, root in (("code_sha256", "/opt/pg/code"),):
    h = {}
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        h[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    out[label] = h
print(json.dumps(out, indent=1))
PY
aws s3 cp /tmp/manifest.json s3://$B/$PFX/seed/manifest.json --region us-east-1
echo "SEED_OK"
date -u > /tmp/DONE
aws s3 cp /tmp/DONE s3://$B/$PFX/seed/DONE --region us-east-1
up
