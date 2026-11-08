"""PRAGNANZ post-critic structured comparator (critic #10): is the ERLANGEN L9 q_proj core
largely inside the layer's TOP-ACTIVATION subspace? If yes, the M2 "same-object" overlap
(DOME nuisance vs core, both ≫ isotropic null) is explained by shared activation structure
rather than a core-specific address. CPU-only, from the banked PG1 site-9 harvest shards.

Disclosure: the harvest is the site-9 OUTPUT residual; L9 q_proj reads the pre-attention
stream — adjacent-stream offset, stated not hidden.

    python -m infra.aws_pg.jobs.pragnanz_actpca --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BASIS_S3 = "s3://pg-lineage-769063704696/results/erlangen_m_textvqa/out/erlangen_core_basis.npz"
HARVEST_S3 = "s3://pg-lineage-769063704696/results/gestalt_arm_s8/out/harvest_pg1/site9"
K_PCA = 32
N_NULL = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    sdir = out / "h"
    sdir.mkdir(exist_ok=True)
    subprocess.run(["aws", "s3", "sync", HARVEST_S3, str(sdir), "--region", "us-east-1",
                    "--only-show-errors", "--exclude", "item_idx*"], check=True)
    shards = sorted(sdir.glob("acts_*.npy"))
    assert shards, "no harvest shards"
    X = np.concatenate([np.load(p) for p in shards], 0).astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    C = (Xc.T @ Xc) / Xc.shape[0]
    w, V = np.linalg.eigh(C)
    P = V[:, ::-1][:, :K_PCA]                                  # top-K_PCA activation PCs
    ev_frac = float(w[::-1][:K_PCA].sum() / w.sum())

    local = out / "core_basis.npz"
    subprocess.run(["aws", "s3", "cp", BASIS_S3, str(local), "--region", "us-east-1"],
                   check=True)
    z = np.load(local)
    key = next(k for k in z.files if "q_proj" in k and ("__9__" in k or "layers__9" in k))
    Q = z[key].astype(np.float64)                              # (2048, 32) IN-axis core slice
    d = Q.shape[0]
    assert P.shape[0] == d, f"dim mismatch {P.shape} vs {Q.shape}"

    def frac_in(A, B):
        """Mean squared projection of A's columns into span(B)."""
        return float(((B.T @ A) ** 2).sum(axis=0).mean())

    obs_core_in_pca = frac_in(Q, P)
    obs_pca_in_core = frac_in(P, Q)
    rng = np.random.default_rng(20260806)
    null = []
    for _ in range(N_NULL):
        R = rng.standard_normal((d, K_PCA))
        Qr, _ = np.linalg.qr(R)
        null.append(frac_in(Qr, P))
    null = np.sort(np.asarray(null))
    p_floor = (1 + int((null >= obs_core_in_pca).sum())) / (N_NULL + 1)

    rep = {"key": key, "k_pca": K_PCA, "n_tokens": int(X.shape[0]),
           "pca_ev_frac_top32": ev_frac,
           "core_in_actpca_frac": obs_core_in_pca,
           "actpca_in_core_frac": obs_pca_in_core,
           "null_core_in_actpca": {"lo": float(null[0]), "hi": float(null[-1]),
                                   "p2_5": float(np.percentile(null, 2.5)),
                                   "p97_5": float(np.percentile(null, 97.5)), "n": N_NULL},
           "p_vs_null": p_floor,
           "reading": "HIGH core_in_actpca (vs ~32/2048 null) => the M2 overlap is "
                      "shared-activation-structure-consistent; LOW => the core is NOT the "
                      "top-activation subspace and M2's overlap needs another explanation",
           "stream_offset_disclosure": "harvest = site-9 output residual; q_proj reads the "
                                       "pre-attention stream (adjacent)"}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1))
    print("PRAGNANZ_ACTPCA_OK", json.dumps({k: rep[k] for k in
          ("core_in_actpca_frac", "actpca_in_core_frac", "pca_ev_frac_top32")}), flush=True)


if __name__ == "__main__":
    main()
