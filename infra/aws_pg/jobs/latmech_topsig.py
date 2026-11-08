"""LATMECH top-σ de-confound carrier — the critical control for the H_store unblind.

Removes the top-k directions of each decoder matrix BY SINGULAR VALUE (energy_indices = [0..k-1]),
NOT by the content gradient — to test whether the content-ablation MDE (RF_content≈floor) is
genuinely content-gradient-specific or just **principal-subspace removal** (the W12 energy≠function
trap: the energy-matched `emrand` removes the low-σ tail, so it never tested removing the principal
directions). No gradient needed. Saves carrier_topsig.pt in the carrier.pt format so a normal
`latmech_cell --condition content --carrier carrier_topsig.pt` ablates the top-σ subspace.

If a top-σ-ablated cell ALSO tanks re-entry → the effect is principal-subspace removal (content
selection incidental). If top-σ leaves re-entry intact while the content carrier tanks it →
content-gradient-specific (H_store stands sharply).

    python -m infra.aws_pg.jobs.latmech_topsig --out <dir> --carrier-k 4 --svd-q 512 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="topk", choices=["topk", "randspan"],
                    help="topk = remove top-k BY σ; randspan = remove a RANDOM σ-matched subspace within "
                         "the top-M σ band at the content carrier's per-matrix energy (the gold-standard "
                         "content-specificity null: matched energy AND σ-profile, scrambled selection)")
    ap.add_argument("--carrier-k", type=int, default=4)
    ap.add_argument("--within-m", type=int, default=64, help="randspan: top-M σ band to draw from")
    ap.add_argument("--content-carrier", default="", help="randspan: content carrier.pt (local or s3) for "
                    "per-matrix target energy")
    ap.add_argument("--svd-q", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from infra.pg import diff_harvest as H
    H.hf_online_setup()

    import torch
    from infra.pg import w14_localize as W14
    from infra.pg import w15_carrier as C
    from infra.pg import latmech as Lm
    from infra.aws_pg.jobs.e4_capability import SEED

    cfac = None
    if args.mode == "randspan":
        assert args.content_carrier, "randspan needs --content-carrier (for per-matrix target energy)"
        cpath = args.content_carrier
        if cpath.startswith("s3://"):
            import subprocess
            cpath = str(out / "content_carrier.pt")
            subprocess.run(["aws", "s3", "cp", args.content_carrier, cpath, "--region", "us-east-1"],
                           check=True)
        cfac = torch.load(cpath, map_location="cpu")

    assert torch.cuda.is_available(), "latmech_topsig needs GPU."
    device = "cuda"
    dkeys = W14.decoder_keys()
    pg_after, pg_before, proc, paths, _it, _gate = H.build_before_after(device)
    del pg_before
    torch.cuda.empty_cache()
    lm = pg_after.get_submodule(paths["language_model"])
    torch.manual_seed(SEED)                                   # same basis seeding as latmech_carrier_factors
    factors = {}
    for k in dkeys:
        W = lm.get_parameter(k).detach().float().to(device)
        qq = min(args.svd_q, min(W.shape))
        U, S, V = torch.svd_lowrank(W, q=qq, niter=8)
        if args.mode == "topk":
            idx = C.energy_indices(args.carrier_k, S.numel())  # TOP-k BY σ (the principal directions)
        else:                                                  # randspan: random σ-matched within top-M
            n = S.numel()
            M = min(args.within_m, n)
            outside = torch.arange(M, n)                        # exclude the low-σ tail → draw within top-M
            idx, _ = Lm.energy_matched_random_idx(S, float(cfac[k]["removed_E"]), seed=args.seed,
                                                  exclude=outside)
        removed_E = Lm.removed_energy(S, idx)
        em_idx, em_E = Lm.energy_matched_random_idx(S, removed_E, seed=args.seed + 1, exclude=idx)
        factors[k] = {
            "carrier": (U[:, idx].cpu(), S[idx].cpu(), V[:, idx].cpu()),
            "emrand": (U[:, em_idx].cpu(), S[em_idx].cpu(), V[:, em_idx].cpu()),
            "removed_E": float(removed_E), "emrand_E": float(em_E),
            "n_dir": int(S.numel()), "k": int(idx.numel()),
        }
        del W, U, S, V
        if device == "cuda":
            torch.cuda.empty_cache()
    torch.save(factors, out / "carrier.pt")
    tot = sum(factors[k]["removed_E"] for k in dkeys)
    (out / "topsig.json").write_text(json.dumps(
        {"mode": args.mode, "carrier_k": args.carrier_k, "within_m": args.within_m,
         "total_removed_E": tot, "n_matrices": len(factors)}, indent=1))
    print(f"[latmech-topsig] mode={args.mode} saved carrier, total removed_E={tot:.1f}")


if __name__ == "__main__":
    main()
