#!/usr/bin/env bash
# Build + push one rental-group image. Runs on the EC2 builder (amd64) — or
# anywhere with docker and AWS creds. All of this is off-meter (§9.10: the
# image is built and pushed BEFORE any rental).
#
# Usage:
#   MODELS="Qwen/Qwen2.5-14B" TAG=v1-14b BASE=vastai/pytorch:<tag> \
#   CODE_TAR=s3://.../code/shackleton/repo_<sha>.tar.gz bash build_image.sh
set -euo pipefail
: "${TAG:?}" "${BASE:?}" "${CODE_TAR:?}"
MODELS="${MODELS:-}"   # empty = deps-only image (MENDELEEV: every job downloads its own weights in-run)
REGION=us-east-1
ACCT=$(aws sts get-caller-identity --query Account --output text)
ECR="$ACCT.dkr.ecr.$REGION.amazonaws.com"
REPO=shackleton
S3_CODE_PREFIX="s3://pg-lineage-769063704696/code/shackleton"

aws ecr describe-repositories --repository-names $REPO --region $REGION \
  >/dev/null 2>&1 || aws ecr create-repository --repository-name $REPO --region $REGION

# login before the pull: a base may itself be one of our ECR images (N13: shackleton:mdl-v1, the AMI-freeze pins)
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin "$ECR"
# pin the base by digest so the image manifest records exactly what shipped
docker pull "$BASE"
BASE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$BASE")

export HF_HUB_DISABLE_XET=1
mkdir -p build/models
if [ -n "$MODELS" ]; then
# venv: Ubuntu 24.04 PEP-668 blocks system pip, and --break-system-packages
# collides with Debian-owned packages (the v1-14b_failed.log lesson).
# huggingface_hub 1.x dropped the hf_transfer extra; plain HTTP is fine off-meter.
python3 -m venv /tmp/hfvenv
/tmp/hfvenv/bin/pip install -q -U huggingface_hub datasets transformers numpy
for m in $MODELS; do
  if [ "${BAKE_MODE:-dir}" = "cache" ] && [[ "$m" == *@* ]]; then
    # repo@commit (N13): the exact revision in HF-cache layout, so snapshot_download(repo, revision=commit) on the box
    # resolves from the image; the include set is n13_setup's allow_patterns
    HF_HOME=$PWD/build/hf_home /tmp/hfvenv/bin/hf download "${m%@*}" --revision "${m#*@}" \
      --include "*.json" --include "*.safetensors" --include "*.jinja" --include "*.txt"
  elif [ "${BAKE_MODE:-dir}" = "cache" ]; then
    # HF-cache layout: from_pretrained(repo_id) resolves offline-fast (V2 path)
    HF_HOME=$PWD/build/hf_home /tmp/hfvenv/bin/hf download "$m" \
      --exclude "*.bin" --exclude "*.pth" --exclude "*.gguf" --exclude "original/*"
  else
    /tmp/hfvenv/bin/hf download "$m" --local-dir "build/models/${m//\//_}" \
      --exclude "*.bin" --exclude "*.pth" --exclude "*.gguf" --exclude "original/*"
    # freeze + tokenize the V1 corpus into the model dir (baked into the image)
    /tmp/hfvenv/bin/python3 prepare_corpus.py "build/models/${m//\//_}"
  fi
done
fi
aws s3 cp "$CODE_TAR" build/repo.tar.gz
# D13: /app is a tar extract, not a git checkout — stamp the house provenance
# file with the tar's TRUE frozen SHA (the mercator_launch.sh pattern), so
# mercator_core.provenance() reads it instead of shelling out to git
CODE_SHA=$(basename "$CODE_TAR" | sed 's/^repo_//; s/\.tar\.gz$//')
mkdir -p build/stamp/infra/pg
printf '{"program": "mercator", "commit_sha": "%s", "dirty": false, "source": "shackleton image build (D13 stamp)"}' \
  "$CODE_SHA" > build/stamp/infra/pg/mercator_provenance.json
(cd build/stamp && tar czf ../provenance.tar.gz infra)

# generate from inside the context so walked paths == emitted COPY paths
PIP_ARGS=()
# PIP_PKGS="-": no system pip layer (the base already carries its pins; N13 builds on mdl-v1)
[ "${PIP_PKGS:-}" = "-" ] && PIP_ARGS=(--pip "")
[ -n "${PIP_PKGS:-}" ] && [ "${PIP_PKGS:-}" != "-" ] && PIP_ARGS=(--pip "$PIP_PKGS")
[ -n "${APT_PKGS:-}" ] && PIP_ARGS+=(--apt "$APT_PKGS")
if [ -n "${VENV_FREEZE:-}" ]; then
  # VENV_FREEZE="DIR=repo/relative/freeze.txt" (the freeze rides the code tar)
  cp "../../${VENV_FREEZE#*=}" build/venv_freeze.txt
  PIP_ARGS+=(--venv "${VENV_FREEZE%%=*}=venv_freeze.txt")
fi
[ -n "${HF_CACHE_DEST:-}" ] && PIP_ARGS+=(--hf-cache-dest "$HF_CACHE_DEST")
for kv in ${IMAGE_ENV:-}; do PIP_ARGS+=(--env "$kv"); done
(cd build && python3 ../gen_dockerfile.py --base "$BASE_DIGEST" \
  --models-dir models --code-tar repo.tar.gz "${PIP_ARGS[@]}" > Dockerfile)
cat build/Dockerfile
# registry cache: builders are fresh EC2 with empty docker caches — the prior
# pushed image (inline cache metadata) lets unchanged dep/weight layers hit
# cache by CONTENT hash, cutting a rebuild from ~50 to ~8 min. Login happens
# before build so BuildKit can resolve the remote cache ref; a missing tag is
# a warning, not an error (first build of a tag).
(cd build && DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from "$ECR/$REPO:$TAG" --cache-from "$ECR/$REPO:$TAG-candidate" -t "$ECR/$REPO:$TAG" .)
if [ "${GATE:-}" = "n13" ]; then
  # a 70 GB image: push the candidate BEFORE the gate, so a gate failure costs a code-layer rebuild (cache-from), not a
  # second weights download and push. Only a gated image gets the real tag (below); manifests name that tag's digest.
  docker tag "$ECR/$REPO:$TAG" "$ECR/$REPO:$TAG-candidate"
  for attempt in 1 2 3; do
    docker push "$ECR/$REPO:$TAG-candidate" && break
    [ "$attempt" = 3 ] && { echo "[build] candidate push failed 3 times"; exit 1; }
    sleep 15
  done
fi

# §9.1 gate: the instrument dry-run must pass INSIDE the image before push
# --entrypoint: the vast base image's init would swallow the command and hang (D6)
if [ "${GATE:-}" = "n13" ]; then
  # N13 (infra/vast/n13_gate.sh): the venv's freeze equals the AWS freeze, gcc and Python.h, the vLLM import, the baked
  # revisions, and the N13 offline gate (N13_DRYRUN_OK) under the venv's own interpreter. Inputs: N13_GATE_S3 (small files).
  mkdir -p /tmp/n13_gate_in && aws s3 sync "${N13_GATE_S3:?}" /tmp/n13_gate_in --only-show-errors
  cp "../../${VENV_FREEZE#*=}" /tmp/n13_gate_in/expected_freeze.txt
  docker run --rm -v /tmp/n13_gate_in:/gate_in:ro -e N13_MODELS="$MODELS" --entrypoint bash "$ECR/$REPO:$TAG" \
    /app/infra/vast/n13_gate.sh | tee /tmp/n13_gate.log
  grep -q "^N13_IMAGE_GATE_OK" /tmp/n13_gate.log
elif [ -n "${PIN_ASSERT:-}" ]; then
  # deps-only image (MENDELEEV): the gate IS the pin set — every "name==version" in PIN_ASSERT
  # must be what the image resolves (local tags like +cu130 ignored) and the science stack must import
  docker run --rm -i -e PIN_ASSERT="$PIN_ASSERT" --entrypoint python3 "$ECR/$REPO:$TAG" - <<'EOF'
import importlib, importlib.metadata as md, os
bad = []
for spec in os.environ["PIN_ASSERT"].split():
    name, want = spec.split("==")
    got = md.version(name)
    if got.split("+")[0] != want:
        bad.append(f"{name}: want {want} got {got}")
assert not bad, bad
for m in ["torch", "transformers", "peft", "accelerate", "datasets", "sklearn", "scipy", "einops", "PIL", "boto3", "huggingface_hub"]:
    importlib.import_module(m)
import torch
print("PIN GATE OK torch", torch.__version__, "cuda", torch.version.cuda, "pins", len(os.environ["PIN_ASSERT"].split()))
EOF
else
  docker run --rm --entrypoint python3 "$ECR/$REPO:$TAG" /app/infra/vast/dryrun_v1.py
fi

# push retry (mdl-v1 build 2026-09-03: the image built and passed the gate, then the LAST layer's
# push died with "grpc: the client connection is closing" — a transient in the daemon's push
# path). ECR keeps every layer that landed ("Layer already exists"), so a retry only ships the
# remainder; three tries, then fail loud.
for attempt in 1 2 3; do
  docker push "$ECR/$REPO:$TAG" && break
  [ "$attempt" = 3 ] && { echo "[build] push failed 3 times"; exit 1; }
  echo "[build] push attempt $attempt failed — retrying in 15 s"; sleep 15
done
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' "$ECR/$REPO:$TAG")

VERSIONS=$(docker run --rm -i --entrypoint python3 "$ECR/$REPO:$TAG" - <<'EOF'
import json, torch, transformers
print(json.dumps({"torch": torch.__version__,
                  "transformers": transformers.__version__}))
EOF
)
SIZE=$(docker image inspect --format='{{.Size}}' "$ECR/$REPO:$TAG")
cat > "image_${TAG}.json" <<EOF
{"tag": "$TAG", "image": "$ECR/$REPO:$TAG", "digest": "$DIGEST",
 "base_digest": "$BASE_DIGEST", "models": "$MODELS",
 "code_tar": "$CODE_TAR", "versions": $VERSIONS, "size_bytes": $SIZE,
 "venv_freeze": "${VENV_FREEZE:-}", "gate": "${GATE:-}",
 "built_utc": "$(date -u +%FT%TZ)"}
EOF
aws s3 cp "image_${TAG}.json" "$S3_CODE_PREFIX/images/image_${TAG}.json"
if [ "${GATE:-}" = "n13" ]; then aws s3 cp /tmp/n13_gate.log "$S3_CODE_PREFIX/images/image_${TAG}_gate.log"; fi
echo "[build] pushed $DIGEST and banked its manifest"
