#!/usr/bin/env bash
set -Eeuo pipefail

umask 027
export PARALLAX_ROOT="${PARALLAX_ROOT:-/workspace/parallax}"
export HF_HOME="${HF_HOME:-/workspace/cache/huggingface}"
mkdir -p "$PARALLAX_ROOT"/{artifacts,events,manifests,state,staging} "$HF_HOME"

# Startup is intentionally installation-free.  A different freeze means the
# image is wrong and the instance must stop before it touches scientific data.
python -m pip check
test -s /opt/parallax/pip-freeze.txt
sha256sum /opt/parallax/pip-freeze.txt > "$PARALLAX_ROOT/state/pip-freeze.sha256"
python -m infra.vast.parallax.image_smoke \
  --lock /opt/parallax/infra/vast/parallax/requirements.lock \
  --output "$PARALLAX_ROOT/state/device_env.json" \
  --require-h100

touch "$PARALLAX_ROOT/state/ONSTART_GREEN"
