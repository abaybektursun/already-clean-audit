"""LATMECH carrier-overlap diagnostic — is the content-gradient carrier the principal (top-σ) subspace?

Loads two carrier.pt files and computes the per-matrix subspace overlap between their selected
left/right subspaces (mean cos² principal angle = ‖A Bᵀ‖²_F / k for orthonormal row-bases). Overlap≈1
⇒ the content carrier IS the top-σ principal subspace (the H_store confound); overlap≈k/q ⇒ distinct
subspaces (content-gradient finds critical directions beyond σ). Pure tensor, NO model — run on pg CPU.

    python -m infra.aws_pg.jobs.latmech_overlap --out <dir> \
        --carrier-a s3://.../latmech-calib3/out/calib/carrier.pt \
        --carrier-b s3://.../latmech-topsig/out/topsig/carrier.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib


def _fetch(uri, dest):
    if uri.startswith("s3://"):
        import subprocess
        subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", "us-east-1"], check=True)
        return str(dest)
    return uri


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--carrier-a", required=True, help="carrier.pt A (e.g. content)")
    ap.add_argument("--carrier-b", required=True, help="carrier.pt B (e.g. top-σ)")
    ap.add_argument("--label", default="content_vs_topsig")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import numpy as np
    from infra.pg import savb

    a = torch.load(_fetch(args.carrier_a, out / "a.pt"), map_location="cpu")
    b = torch.load(_fetch(args.carrier_b, out / "b.pt"), map_location="cpu")
    keys = [k for k in a if k in b]
    ovU, ovV = [], []
    for k in keys:
        Ua, _, Va = a[k]["carrier"]
        Ub, _, Vb = b[k]["carrier"]
        # subspace_overlap(rows_a (ka,d), rows_b (kb,d)) = ‖A Bᵀ‖²_F / ka ; cols orthonormal → transpose.
        ovU.append(savb.subspace_overlap(Ua.t().numpy(), Ub.t().numpy()))
        ovV.append(savb.subspace_overlap(Va.t().numpy(), Vb.t().numpy()))
    ka = int(a[keys[0]]["carrier"][0].shape[1])
    kb = int(b[keys[0]]["carrier"][0].shape[1])
    # chance overlap ≈ k / dim (tiny); report it for context per matrix isn't constant, so just k's.
    res = {"label": args.label, "n_matrices": len(keys), "ka": ka, "kb": kb,
           "mean_overlap_U": float(np.mean(ovU)), "mean_overlap_V": float(np.mean(ovV)),
           "median_overlap_U": float(np.median(ovU)), "median_overlap_V": float(np.median(ovV))}
    (out / "overlap.json").write_text(json.dumps(res, indent=1))
    print("[latmech-overlap]", json.dumps(res))


if __name__ == "__main__":
    main()
