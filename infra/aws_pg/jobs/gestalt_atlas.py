"""GESTALT Arm D atlas fits — one unit, both sites, at the Arm-S winning K (FR-5).
Requires the unit's harvest bank (gestalt_harvest --unit X) on S3. Per (unit, site) it banks:
  * full-fit atlases (EM seeds 0,1) on the frozen fit subsample of the unit's REAL stream,
  * split-half atlases (disjoint token halves A/B, seed 0) — the P4 band inputs,
  * twin atlases (fit on the unit's moment-matched twin stream, seeds 0,1) — the FR-6 floor
    inputs (PG1-atlas-vs-twin-of-target survival = what moments alone explain).
All correspondence math happens downstream in gestalt_diff (CPU) from these banked npz files.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_atlas \
      --args "--unit pi0 --harvest-prefix results/gestalt_harvest_pi0 --k-d 16 --rank 2" \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/gestalt_atlas_pi0 --budget-min 180

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import time

SITES = (9, 14)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--harvest-prefix", default="",
                    help="FULL S3 path under the bucket to the harvest OUT dir (no /out appended)")
    ap.add_argument("--harvest-local", default="", help="local harvest out-dir (pipeline mode)")
    ap.add_argument("--k-d", type=int, required=True, help="the Arm-S winning K (FR-5, frozen)")
    ap.add_argument("--rank", type=int, required=True, help="r_frozen from the offline gate")
    ap.add_argument("--sites", default="9,14")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    from infra.aws_pg import config as C
    from infra.pg import gestalt_splat as GS

    assert torch.cuda.is_available(), "gestalt_atlas needs CUDA"
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    sites = [int(s) for s in a.sites.split(",")]
    assert bool(a.harvest_prefix) != bool(a.harvest_local), \
        "exactly one of --harvest-prefix / --harvest-local"
    rep = {"object": "GESTALT Arm D atlas fits", "unit": a.unit, "k_d": a.k_d, "rank": a.rank,
           "sites": sites, "harvest_prefix": a.harvest_prefix or None,
           "harvest_local": a.harvest_local or None,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    cells = []
    rep["cells"] = cells

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    if a.harvest_local:
        hdir = pathlib.Path(a.harvest_local)
        assert (hdir / "item_hashes.json").exists(), f"no harvest bank at {hdir}"
    else:
        hdir = pathlib.Path("/tmp/gestalt_bank")
        subprocess.run(["aws", "s3", "sync", f"s3://{C.BUCKET}/{a.harvest_prefix}", str(hdir),
                       "--region", "us-east-1", "--only-show-errors"], check=True)

    for L in sites:
        sdir = hdir / f"site{L}"
        shards = sorted(sdir.glob("acts_*.npy"))
        assert shards, f"no shards under {sdir}"
        X = torch.cat([torch.from_numpy(np.load(p)).to(torch.float32) for p in shards], 0)
        mz = np.load(sdir / "moments.npz")
        mean = torch.from_numpy(mz["mean"])
        cov = torch.from_numpy(mz["cov"])
        n, d = X.shape
        # X stays on CPU; only fit subsamples go to device (24GB-pool safe, mirrors the MDE)
        Lc = GS.cov_sym_sqrt(cov)
        fits_dir = out / f"site{L}"
        fits_dir.mkdir(exist_ok=True)

        def bank_fit(tag, Xsrc, seed):
            Xs = GS.fit_subsample(Xsrc, seed=seed).to(device)
            atlas = GS.fit_mppca(Xs, K=a.k_d, r=a.rank, seed=seed)
            key = f"{a.unit}|site{L}|{tag}|s{seed}"
            if atlas["void"]:
                cells.append({"key": key, "void": True, "void_reason": atlas["void_reason"]})
                _w()
                print(f"[GA] {key} VOID: {atlas['void_reason']}", flush=True)
                return
            cpu = GS.atlas_to_cpu(atlas)
            np.savez(fits_dir / f"{tag}_s{seed}.npz",
                     **{k: cpu[k].numpy() for k in ("log_pi", "mu", "U", "lam", "sigma2")})
            cells.append({"key": key, "void": False, "sha": GS.atlas_sha(cpu),
                          "n_iters": atlas["n_iters"], "collapsed": atlas["collapsed"]})
            _w()
            print(f"[GA] {key} sha={cells[-1]['sha']}", flush=True)

        for s in GS.EM_SEEDS:
            bank_fit("full", X, s)
        # split halves: disjoint deterministic permutation halves of the REAL tokens (P4 inputs)
        g = torch.Generator(device="cpu").manual_seed(7)
        perm = torch.randperm(n, generator=g)
        bank_fit("halfA", X[perm[: n // 2]], 0)
        bank_fit("halfB", X[perm[n // 2:]], 0)
        # twin atlases (the FR-6 floor inputs); n_fit direct draw — an iid subsample of an iid
        # draw IS an iid draw (the mde lean-path note)
        Xt = GS.twin_sample(mean, Lc, min(GS.N_FIT, n), seed=0, device=device)
        for s in GS.EM_SEEDS:
            bank_fit("twin", Xt, s)
        del X, Xt
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        rep[f"site{L}"] = {"n_tokens": int(n), "d": int(d),
                           "mean_sha": hashlib.sha256(mz["mean"].tobytes()).hexdigest()[:16]}
        _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print(f"GESTALT_ATLAS_OK unit={a.unit}", flush=True)


if __name__ == "__main__":
    main()
    # D7: datasets/cuda extension threads abort the interpreter during finalization
    # (PyGILState_Release fatal, rc=-6) AFTER all artifacts are banked; every write is
    # explicit and already streamed, so skip finalization outright.
    import os as _os
    _os._exit(0)
