"""W10 vision harvest job (prereg §9 step 2, control-data development).

Harvests the four §2 encoders × two representation points (vision_harvest.py) and writes one
npz per encoder + a manifest.json recording every dim / patch-count / held-count / SHA. Same job
serves the SMOKE (small --n-per-probe → validate structure cheaply, the foundation gate before
saev/USAE) and the real harvest (--n-per-probe 384). No real cross-encoder readout here — this
only produces activations.

    python -m infra.aws_pg.jobs.w10_harvest --out <dir> --n-per-probe 8           # smoke
    python -m infra.aws_pg.jobs.w10_harvest --out <dir> --n-per-probe 384         # full
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-probe", type=int, default=384)
    ap.add_argument("--encoders", default=",".join(("dinov2", "siglip-base", "pg2-vision", "random")))
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # hf_online_setup MUST precede any hf/transformers import (offline constant is cached at import).
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import numpy as np
    import torch
    from infra.pg import vision_harvest as V

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoders = [e.strip() for e in args.encoders.split(",") if e.strip()]
    print(f"[W10-HARV] device={device} encoders={encoders} n_per_probe={args.n_per_probe}")

    probe = V.load_probe_images(n_per_probe=args.n_per_probe)
    images, held = probe["images"], probe["held"]
    print(f"[W10-HARV] probe images={len(images)} held={int(held.sum())} "
          f"({float(held.mean()):.3f}) tasks={sorted(set(probe['tasks']))}")

    manifest = {"device": device, "probe": probe["manifest"], "encoders": {}}
    for name in encoders:
        h = V.harvest_encoder(name, images, held, device, batch_size=args.batch_size)
        post = h["post"]
        np.savez_compressed(
            out / f"acts_{name}.npz",
            pre=h["pre"].numpy(),
            post=(post.numpy() if post is not None else np.zeros(0, dtype=np.float32)),
            held_patch=h["held_patch"], image_idx=h["image_idx"])
        rec = {"info": h["info"], "d_pre": h["d_pre"], "d_post": h["d_post"],
               "n_per_image": h["n_per_image"], "n_patch": int(h["pre"].shape[0]),
               "n_patch_held": int(h["held_patch"].sum()), "has_post": post is not None}
        manifest["encoders"][name] = rec
        print(f"[W10-HARV] {name:12s} pre={tuple(h['pre'].shape)} "
              f"post={(tuple(post.shape) if post is not None else None)} "
              f"n_per_image={h['n_per_image']} held_patch={rec['n_patch_held']}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[W10-HARV] done → {out}/manifest.json + {len(encoders)} npz")


if __name__ == "__main__":
    main()
