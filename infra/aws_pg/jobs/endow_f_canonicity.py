"""ENDOW-F atlas — canonicity axis (spec §5.2) for the conserved cores. CPU, cached ΔWs. A core's
canonicity c = mean pairwise principal-subspace overlap across independent estimation draws, at
k_can, normalized against the chance floor k/d. The draws for a consensus core are subsample-consensus
(bottom-k of a random 12-of-24 FT subset — the §5.2 recipe). High c = a canonical, sampling-stable
subspace; c≈0 = draw-to-draw noise.

Feeds the two-kinds law (endow_atlas.two_kinds_law): primitives predicted canonical (high c), reads
predicted family-valued (low c). Reads' canonicity is banked (INHERIT: S_action top-16 reproducibility
0.36 → normalized low); assembled at synthesis.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_f_canonicity --type r7i.4xlarge --no-weights \
      --s3-prefix results/endow_f_canonicity --budget-min 120 --args "--k-can 16 --n-draws 3"
"""
import argparse
import json
import pathlib
import subprocess

from infra.aws_pg.jobs._inherit_shared import INVCORE_SCRATCH, N_FT

SITES = ("q_proj", "v_proj", "o_proj", "gate_proj", "up_proj")


def _prefetch_site(site, workdir):
    subprocess.run(["aws", "s3", "cp", f"{INVCORE_SCRATCH}/", str(workdir), "--recursive",
                    "--exclude", "*", "--include", f"{site}__*", "--region", "us-east-1",
                    "--only-show-errors"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-can", type=int, default=16)
    ap.add_argument("--n-draws", type=int, default=3)
    ap.add_argument("--layers", default="0,3,6,9,12,15")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]

    import numpy as np
    import torch
    from infra.pg.inherit_core import bottom_eigvecs, consensus_gram
    from infra.pg import endow_atlas as EA

    work = pathlib.Path("/tmp/endow_can")
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    # fixed subsample memberships (12-of-24) per draw, shared across sites/layers
    draws_idx = [sorted(rng.choice(N_FT, size=N_FT // 2, replace=False).tolist())
                 for _ in range(a.n_draws)]
    rep = {"k_can": a.k_can, "n_draws": a.n_draws, "layers": layers, "draws_idx": draws_idx,
           "per_site": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    for site in SITES:
        _prefetch_site(site, work)
        d = None
        c_by_layer = []
        for L in layers:
            deltas = [torch.as_tensor(np.load(f"{work}/{site}__L{L}__{i}.npy").astype(np.float64))
                      for i in range(N_FT)]
            d = int(deltas[0].shape[1])
            bases = []
            for idx in draws_idx:
                G = consensus_gram([deltas[i] for i in idx])
                U, _ = bottom_eigvecs(torch.as_tensor(G), a.k_can)
                bases.append(U)
            c_by_layer.append(EA.canonicity(bases, d))
        c_mean = float(np.mean(c_by_layer))
        rep["per_site"][site] = {"d": d, "canonicity": c_mean,
                                 "c_by_layer": {str(L): c_by_layer[i] for i, L in enumerate(layers)}}
        print(f"[CAN] {site}: canonicity={c_mean:.4f} (k_can={a.k_can}, {a.n_draws} draws)", flush=True)
        for f in work.glob(f"{site}__*"):
            f.unlink()
        _w()

    rep["green"] = True
    _w()
    print("CANONICITY_OK", json.dumps({s: rep["per_site"][s]["canonicity"] for s in SITES}, default=str))


if __name__ == "__main__":
    main()
