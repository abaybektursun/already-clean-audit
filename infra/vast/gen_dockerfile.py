#!/usr/bin/env python3
"""Emit a Dockerfile baking deps, model weights, and repo code.

Weights ride docker layers because vast bills GPU time only in `running`;
the image pull happens in unbilled `loading`. One COPY per large file keeps
every layer under registry caps and lets the host pull layers in parallel.
Layer order: deps -> weights (stable) -> code (volatile) so a code fix only
rebuilds the last, tiny layer.
"""
import argparse
import os

BIG = 1 << 30  # files >= 1 GiB get their own layer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="base image (digest-pinned)")
    p.add_argument("--models-dir", required=True,
                   help="dir INSIDE the build context holding models/<name>/...")
    p.add_argument("--code-tar", required=True,
                   help="repo tar.gz INSIDE the build context (ADD auto-extracts)")
    p.add_argument("--pip", default="transformers accelerate safetensors awscli "
                                    "vastai boto3 numpy sentencepiece hf_transfer",
                   help="pip packages baked into the image")
    p.add_argument("--pip-extra", default="",
                   help="extra pip args line, e.g. CPU torch for rehearsal")
    p.add_argument("--apt", default="",
                   help="apt packages baked into the image (N13: gcc and the Python headers Triton compiles against)")
    p.add_argument("--venv", default="",
                   help="DIR=FREEZE: a venv at DIR holding exactly the pins of FREEZE (a pip freeze INSIDE the build context)")
    p.add_argument("--hf-cache-dest", default="",
                   help="bake build/hf_home/hub/<repo dir> to DEST/hub/<repo dir>, one layer per repo (HF cache layout, symlinks kept)")
    p.add_argument("--env", action="append", default=[], help="K=V baked as docker ENV (repeatable)")
    args = p.parse_args()

    lines = [f"FROM {args.base}",
             # PEP-668: official pytorch images mark their env externally
             # managed; inside the image there is no distro ownership to
             # protect, so the override is safe (D10)
             "ENV PIP_BREAK_SYSTEM_PACKAGES=1"]
    if args.pip_extra:
        lines.append(f"RUN pip install --no-cache-dir {args.pip_extra}")
    if args.pip:
        lines.append(f"RUN pip install --no-cache-dir {args.pip}")
    if args.apt:
        lines.append("RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "
                     f"{args.apt} && rm -rf /var/lib/apt/lists/*")
    if args.venv:
        # --no-deps installs exactly the frozen versions (no re-resolution, so no drift); pip check then proves the
        # frozen set is self-consistent, so a later `pip install <pinned pkg>` inside the venv is a no-op
        vdir, freeze = args.venv.split("=", 1)
        lines.append(f"COPY {freeze} /tmp/venv_freeze.txt")
        lines.append(f"RUN python3 -m venv {vdir} && {vdir}/bin/pip install --no-cache-dir --no-deps -r /tmp/venv_freeze.txt "
                     f"&& {vdir}/bin/pip check")

    root = args.models_dir
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir):
            continue
        # D15: per-file layers exceeded old storage drivers' depth limit
        # (aufs caps at 42; the 72B image hit "max depth exceeded" on pull).
        # Group files into layers of <= GROUP_GB so total layer count stays
        # small on every host while each layer stays under registry caps.
        GROUP = 40 << 30
        entries = []
        for dirpath, dirs, files in os.walk(mdir):
            # grouped COPY flattens into /models/<model>/, which is only valid
            # for flat snapshot dirs — prune dot-dirs (HF .cache metadata)
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in sorted(files):
                fp = os.path.join(dirpath, f)
                entries.append((os.path.relpath(fp, root), os.path.getsize(fp)))
        group, gsize = [], 0
        for rel, size in sorted(entries):
            if group and gsize + size > GROUP:
                srcs = " ".join(f"{root}/{r}" for r in group)
                lines.append(f"COPY {srcs} /models/{model}/")
                group, gsize = [], 0
            group.append(rel)
            gsize += size
        if group:
            srcs = " ".join(f"{root}/{r}" for r in group)
            lines.append(f"COPY {srcs} /models/{model}/")

    # HF-cache bake (V2 path): one layer, internal symlinks preserved;
    # blobs for PG1+gemma total ~17GB, under registry layer caps
    hf_home = os.path.join(os.path.dirname(args.models_dir) or ".", "hf_home")
    if args.hf_cache_dest:
        # one layer per repo dir: N13 bakes 3 revisions of 2 repos (29.2 GB + 14.6 GB), each layer under the 40 GB group cap
        for repo in sorted(r for r in os.listdir(os.path.join(hf_home, "hub")) if r.startswith("models--")):
            lines.append(f"COPY hf_home/hub/{repo} {args.hf_cache_dest}/hub/{repo}")
        lines.append(f"ENV HF_HOME={args.hf_cache_dest} HF_HUB_CACHE={args.hf_cache_dest}/hub")
    elif os.path.isdir(hf_home):
        lines.append("COPY hf_home /hf_home")
        lines.append("ENV HF_HOME=/hf_home")
    lines.append(f"ADD {args.code_tar} /app/")
    # provenance stamp lands AFTER the code tar so it wins the layer overlay
    if os.path.exists(os.path.join(os.path.dirname(args.models_dir) or ".",
                                   "provenance.tar.gz")):
        lines.append("ADD provenance.tar.gz /app/")
    lines.append("RUN pip freeze > /app/pip_freeze.txt")
    lines.append("ENV PYTHONPATH=/app HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                 "HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1")
    for kv in args.env:
        assert "=" in kv and " " not in kv, kv
        lines.append(f"ENV {kv}")
    lines.append("WORKDIR /app")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
