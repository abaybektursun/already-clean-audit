# Vast's SSH-capable CUDA base, pinned to the live amd64 manifest digest.
# The tag documents the human version; only the digest controls the bytes.
FROM vastai/base-image:cuda-13.1.2-auto@sha256:4dc9d17083bc5e09dabdcf142db10637823cbf38c4e4253949f508d30d9f1749

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git git-lfs jq openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/parallax
COPY infra/vast/parallax/requirements.lock /opt/parallax/requirements.lock
RUN python -m pip install --no-cache-dir --upgrade pip==26.1.1 \
    && python -m pip install --no-cache-dir -r requirements.lock \
    && python -m pip check \
    && python -m pip freeze --all | sort > /opt/parallax/pip-freeze.txt

COPY infra /opt/parallax/infra
COPY pyproject.toml /opt/parallax/pyproject.toml
COPY studies/lineage/cross-lineage/config /opt/parallax/studies/lineage/cross-lineage/config
ENV PYTHONPATH=/opt/parallax \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/cache/huggingface \
    PARALLAX_ROOT=/workspace/parallax

RUN python -m infra.vast.parallax.cli dag-status \
      --matrix studies/lineage/cross-lineage/config/parallax_matrix.yaml \
      --state /tmp/parallax-dag.json >/dev/null

RUN python -m infra.vast.parallax.image_smoke \
      --lock infra/vast/parallax/requirements.lock \
      --output /opt/parallax/image-smoke-cpu.json

CMD ["bash", "-lc", "sleep infinity"]
