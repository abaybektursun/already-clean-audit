"""GESTALT P1-at-scale standalone — the d=2048 planted-shape gate WITHOUT the harvest/battery.
Exists because d=2048 fitter behavior is invisible to the d=32 offline gate (D10/D11/D14/D15
were each found by burning a full pipeline cycle); this job answers the same question in one
cheap GPU run. Pure synthetic; no weights, no HF.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_p1gpu --args "--rank 2" \
      --type g6e.2xlarge,p5.4xlarge --s3-prefix results/gestalt_p1gpu --budget-min 60

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import time


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--diag", action="store_true",
                    help="instrumented two-cluster K=2 replay with per-iter trajectory bank")
    ap.add_argument("--diag-manifold", action="store_true",
                    help="instrumented curved-manifold K=16 replay: per-component placement")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch

    from infra.aws_pg.jobs.gestalt_mde import p1_at_scale
    from infra.pg import gestalt_splat as GS

    assert torch.cuda.is_available(), "gestalt_p1gpu needs CUDA"
    torch.backends.cuda.matmul.allow_tf32 = True
    rep = {"object": "GESTALT P1-at-scale standalone", "rank": a.rank, "d": a.d,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}

    if a.diag:
        diag_two_cluster(GS, a.rank, a.d, rep, out)
        rep["elapsed_s"] = time.time() - t0
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        print("GESTALT_P1GPU_DIAG_OK", flush=True)
        return

    if a.diag_manifold:
        diag_manifold(GS, a.rank, a.d, rep, out)
        rep["elapsed_s"] = time.time() - t0
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
        print("GESTALT_P1GPU_DIAGM_OK", flush=True)
        return

    p1_at_scale(GS, a.rank, a.d, "cuda", rep)     # asserts on failure (fail-loud)
    rep["elapsed_s"] = time.time() - t0
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("GESTALT_P1GPU_OK", json.dumps(rep["P1_at_scale"], default=str), flush=True)


def diag_two_cluster(GS, r, d, rep, out):
    """Instrumented replay of the failing two-cluster K=2 fit: banks the seeding choices and
    the per-iteration trajectory (component sep-axis coords, weights, sigma2, shifts) so the
    stuck mechanism is READ, not theorized."""
    import json as _json

    import torch
    ns = (32.0 / d) ** 0.5
    X, lab = GS.make_two_cluster(60_000, d, seed=2, noise=1.0 * ns)
    X = X.to("cuda")
    # the true separation axis in R^d (re-derive the embedding frame exactly as the plant does)
    g = torch.Generator(device="cpu").manual_seed(2 + 77)
    A = torch.randn(d, 2, generator=g)
    Qf, _ = torch.linalg.qr(A)
    axis = Qf[:, 0].to("cuda")

    # --- replicate fit_mppca's exact init path, instrumented -------------------------------
    n = X.shape[0]
    g2 = torch.Generator(device="cpu").manual_seed(0)
    sub = X[torch.randperm(n, generator=g2)[:20_000].to(X.device)].float()
    Xc = sub - sub.mean(0)
    torch.manual_seed(0)
    _, S, V = torch.svd_lowrank(Xc, q=r + GS.SVD_OVERSAMPLE, niter=GS.SVD_NITER)
    Z = (Xc @ V).to(torch.float64)
    idx0 = int(torch.randint(sub.shape[0], (1,), generator=g2).item())
    d2 = ((Z - Z[idx0]) ** 2).sum(1)
    p = (d2 / d2.sum()).cpu()
    idx1 = int(torch.multinomial(p, 1, generator=g2).item())
    seeds_ax = [float((sub[i] @ axis).item()) for i in (idx0, idx1)]
    rep["seeding"] = {"idx": [idx0, idx1], "sep_axis_coords": seeds_ax,
                      "cross_cluster": bool(seeds_ax[0] * seeds_ax[1] < 0),
                      "top_sv": [float(x) for x in S[:4]]}
    print(f"[diag] seeds at axis coords {seeds_ax} cross={rep['seeding']['cross_cluster']}",
          flush=True)

    # --- run EM manually with per-iter logging (same code path calls) ----------------------
    atlas = GS._init_atlas(X, 2, r, 0)
    floor = GS.SIGMA2_MIN_FRAC * float((X * X).mean().item())
    traj = []
    for it in range(GS.EM_MAX_ITERS):
        log_gamma, total_ll = GS._log_resp(X, atlas)
        prev_mu = atlas["mu"]
        atlas = GS._m_step(X, log_gamma, r, 0, floor)
        gam = log_gamma.exp()
        traj.append({
            "it": it, "ll_pp": total_ll / n,
            "mu_axis": [float((atlas["mu"][c] @ axis).item()) for c in (0, 1)],
            "w": [float(x) for x in atlas["log_pi"].exp()],
            "s2": [float(x) for x in atlas["sigma2"]],
            "lam_max": [float(atlas["lam"][c].max()) for c in (0, 1)],
            "hard_frac": float((gam.max(1).values > 0.99).float().mean().item()),
            "mu_shift": float((atlas["mu"] - prev_mu).norm(dim=1).max().item())})
        if it % 10 == 0:
            print(f"[diag] it={it} ll/pt={traj[-1]['ll_pp']:.4f} mu_ax={traj[-1]['mu_axis']} "
                  f"w={traj[-1]['w']} shift={traj[-1]['mu_shift']:.5f}", flush=True)
    (out / "diag_traj.json").write_text(_json.dumps(traj, indent=1))
    rep["final"] = traj[-1]
    rep["true_cluster_axis_means"] = [float((X[lab == 0] @ axis).mean()),
                                      float((X[lab == 1] @ axis).mean())]


def diag_manifold(GS, r, d, rep, out):
    """Per-component placement of the K=16 curved-manifold fit: where each mean sits in the
    planted 3-frame (s, t, z), its z-residual vs curv*(s^2+t^2), off-frame norm, weight, and
    member count — enough to read whether the 4.5 floor is fit failure, curvature bias, or a
    statistic artifact."""
    import json as _json

    import torch
    ns = (32.0 / d) ** 0.5
    X = GS.make_curved_manifold(60_000, d, seed=3, noise=0.15 * ns).to("cuda")
    atlas = GS.fit_mppca(X, K=16, r=r, seed=0)
    assert not atlas["void"]
    g = torch.Generator(device="cpu").manual_seed(3 + 77)
    A = torch.randn(d, 3, generator=g)
    Q, _ = torch.linalg.qr(A)
    mu = atlas["mu"].cpu()
    w = atlas["log_pi"].exp().cpu()
    mu3 = mu @ Q[:, :3]
    off = (mu - mu3 @ Q[:, :3].T).norm(dim=1)
    resid = (mu3[:, 2] - 0.35 * (mu3[:, 0] ** 2 + mu3[:, 1] ** 2)).abs()
    lg, _ = GS._log_resp(X, atlas)
    counts = lg.argmax(1).bincount(minlength=16).cpu()
    comps = [{"c": c, "w": float(w[c]), "n_members": int(counts[c]),
              "s": float(mu3[c, 0]), "t": float(mu3[c, 1]), "z": float(mu3[c, 2]),
              "z_resid": float(resid[c]), "off_frame": float(off[c])}
             for c in range(16)]
    rep["manifold_components"] = comps
    rep["stat_value"] = float((resid + off).mean())
    rep["stat_live_only"] = float((resid + off)[w >= GS.MIN_WEIGHT].mean())
    (out / "diag_manifold.json").write_text(_json.dumps(comps, indent=1))
    for e in comps:
        print(f"[diagM] c={e['c']:2d} w={e['w']:.4f} n={e['n_members']:5d} "
              f"(s,t,z)=({e['s']:+.2f},{e['t']:+.2f},{e['z']:+.2f}) "
              f"zresid={e['z_resid']:.3f} off={e['off_frame']:.3f}", flush=True)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # D7: extension-thread finalization abort
