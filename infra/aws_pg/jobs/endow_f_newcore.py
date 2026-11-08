"""ENDOW-F new-core estimation (endow spec §5.3, Finding-2) — does every PG1 fine-tune refuse to
write into the v/o and MLP-input subspaces the way it avoids q/k? Runs the invariant_core consensus
recipe + the rotation-invariant matched-random existence null on the CACHED 24-FT ΔWs (CPU, no model
download, no GPU). One projection TYPE per invocation (all 18 layers); fan out with `pg map`.

Reuses invariant_core's null + verdict machinery VERBATIM (Finding-2: `--types` already covers every
site; no new consensus code). `q_proj` is the reproduce-control — it must land BEYOND_NULL_below,
matching the banked U_unt-qk, or the pipeline is mis-wired (halt). Saves the bottom-k consensus
eigvecs per (type, layer) = the candidate U_unt-<site> basis, for the ENDOW-F atlas and ENDOW-J
ledger to reuse.

  ./infra/aws_pg/pg map infra.aws_pg.jobs.endow_f_newcore \
      --over '["--types q_proj","--types v_proj","--types o_proj","--types gate_proj","--types up_proj"]' \
      --type r7i.4xlarge --no-weights --s3-prefix results/endow_f_newcore --budget-min 240
"""
import argparse
import json
import os
import pathlib
import subprocess

from infra.aws_pg.jobs._inherit_shared import INVCORE_SCRATCH, N_FT
from infra.aws_pg.jobs.invariant_core import (_bottom_topk_eigs, _consensus_gram, _null_band,
                                              _spectrum_normsq, _verdict)


def _prefetch_type(typ, workdir):
    """Bulk-parallel download ALL of one type's cached ΔWs (24 FTs × 18 layers) in ONE aws s3 cp
    --recursive (the CLI parallelizes) — NOT 432 sequential single-file cp (the q_proj slowness)."""
    subprocess.run(["aws", "s3", "cp", f"{INVCORE_SCRATCH}/", str(workdir), "--recursive",
                    "--exclude", "*", "--include", f"{typ}__*", "--region", "us-east-1",
                    "--only-show-errors"], check=True)


def _load_deltas(typ, L, workdir):
    """The 24 cached fp16 ΔW matrices for (type, layer) from LOCAL disk (pre-fetched) → fp64 tensors."""
    import numpy as np
    import torch
    return [torch.as_tensor(np.load(f"{workdir}/{typ}__L{L}__{i}.npy").astype(np.float64))
            for i in range(N_FT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--types", default="q_proj", help="ONE type per job (pg map fans out)")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--k-grid", default="8,32,128")
    ap.add_argument("--n-null", type=int, default=64)
    ap.add_argument("--k-existence", type=int, default=8, help="the k at which the existence verdict is read")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path("/tmp/endow_f")
    work.mkdir(parents=True, exist_ok=True)

    ncpu = str(os.cpu_count() or 8)                       # let numpy/LAPACK use every core (QR/eigh)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, ncpu)
    import numpy as np
    types = [t for t in a.types.split(",") if t]
    layers = list(range(18)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    k_grid = [int(x) for x in a.k_grid.split(",")]
    rep = {"types": types, "layers": layers, "k_grid": k_grid, "n_null": a.n_null,
           "k_existence": a.k_existence, "seed": a.seed, "per_type": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    ke = str(a.k_existence)
    for typ in types:
        _prefetch_type(typ, work)                        # ONE bulk parallel download for all 18 layers
        print(f"[F-NEWCORE] {typ}: prefetched ΔWs", flush=True)
        per_layer = {}
        for L in layers:
            deltas = _load_deltas(typ, L, str(work))
            n = int(deltas[0].shape[1])
            G = _consensus_gram(deltas)
            spectra = [_spectrum_normsq(dW) for dW in deltas]
            eigvals, real_bot, real_top = _bottom_topk_eigs(G, k_grid)
            null_bot, null_top, null_samp = _null_band(spectra, n, k_grid, a.n_null, a.seed + L)
            verdict = _verdict(real_bot, null_bot, null_samp, k_grid)
            # the candidate U_unt-<site> basis = bottom-k_max consensus eigvecs (ascending eigh)
            w_full, V_full = np.linalg.eigh(G)
            np.save(out / f"uunt_{typ}_L{L}.npy", V_full[:, :max(k_grid)].astype(np.float32))
            per_layer[str(L)] = {"n": n, "real_bot": real_bot, "verdict": verdict,
                                 "bottom_eig_min": float(np.clip(w_full, 0.0, None)[:max(k_grid)].mean())}
            print(f"[F-NEWCORE] {typ} L{L}: k{a.k_existence} {verdict[ke]['verdict']} "
                  f"z={verdict[ke]['z']:.1f} p={verdict[ke]['p_lower']:.4f}", flush=True)
            rep["per_type"][typ] = {"per_layer": per_layer}
            _w()
        # existence: a core EXISTS iff a strict majority of layers land BEYOND_NULL_below at k_existence
        below = [L for L in layers if per_layer[str(L)]["verdict"][ke]["verdict"] == "BEYOND_NULL_below"]
        exists = len(below) > len(layers) / 2
        rep["per_type"][typ]["existence"] = {
            "k": a.k_existence, "n_layers_below_null": len(below), "n_layers": len(layers),
            "layers_below": below, "exists": bool(exists),
            "median_z": float(np.median([per_layer[str(L)]["verdict"][ke]["z"] for L in layers]))}
        _w()
        for f in work.glob(f"{typ}__*"):                 # free disk before the next type's prefetch
            f.unlink()

    if "q_proj" in types:                              # reproduce-control gate
        qex = rep["per_type"]["q_proj"]["existence"]
        assert qex["exists"], f"q_proj reproduce-control did NOT reproduce U_unt-qk (BEYOND_NULL): {qex}"
    rep["green"] = True
    _w()
    print("F_NEWCORE_OK", json.dumps({t: rep["per_type"][t]["existence"] for t in types}, default=str))


if __name__ == "__main__":
    main()
