"""PRAGNANZ offline gate — weight-free dry-run of every NEW instrument (runbook step 2).
Each check replays a KNOWN event and RAISES on failure. N1 (the five-program lesson) prints the
width of every null/band used.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.pragnanz_offline --type r7i.2xlarge --no-weights \
      --s3-prefix results/pragnanz_offline --budget-min 60

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib

D_SYN = 32
N_SYN = 4000


def check_ladder(rep):
    import torch

    from infra.pg import gestalt_splat as GS
    from infra.pg import pragnanz_core as PC

    X, lab = GS.make_two_cluster(N_SYN, D_SYN, seed=2, sep=8.0, noise=1.0)
    atlas = GS.fit_mppca(X, K=2, r=2, seed=0)
    assert not atlas["void"]
    assign = PC.hard_assignments(X, atlas)

    # P2: identity permutation is BIT-IDENTICAL to r1
    r1 = PC.resample_from_assignments(atlas, assign, ladder_seed=7)
    r1b = PC.resample_from_assignments(atlas, assign.clone(), ladder_seed=7)
    assert torch.equal(r1, r1b), "P2 FAIL: identical assignments differ bit-level"
    perm_id = torch.arange(assign.shape[0])
    r2_id = PC.resample_from_assignments(atlas, assign[perm_id], ladder_seed=7)
    assert torch.equal(r1, r2_id), "P2 FAIL: identity permutation != r1"

    # marginal preservation is EXACT under permutation
    a2 = PC.permute_assignments(assign, perm_seed=11)
    assert torch.equal(assign.bincount(minlength=2).sort()[0],
                       a2.bincount(minlength=2).sort()[0]), "marginals not preserved"

    # the ladder's KNOWN EVENT: nearest-cluster-mean readout — r1 keeps it, r2/r3 kill it
    m0, m1 = X[lab == 0].mean(0), X[lab == 1].mean(0)

    def acc(Y):
        d0 = ((Y - m0) ** 2).sum(1)
        d1 = ((Y - m1) ** 2).sum(1)
        return float(((d1 < d0).float() == lab).float().mean())

    r2 = PC.resample_from_assignments(atlas, a2, ladder_seed=7)
    m, C = GS.stream_moments(X)
    r3 = GS.twin_sample(m, GS.cov_sym_sqrt(C), X.shape[0], seed=5)
    accs = {"r0": acc(X), "r1": acc(r1), "r2": acc(r2), "r3": acc(r3)}
    assert accs["r1"] > 0.95, f"r1 lost the readout: {accs}"
    assert accs["r2"] < 0.62 and accs["r3"] < 0.62, f"r2/r3 did not destroy arrangement: {accs}"
    rep["ladder"] = {"accs": accs, "note": "r2-r1 contrast fires on the plant; P2 bit-exact"}
    print(f"[PZ] ladder PASS {accs}", flush=True)
    return atlas, assign, X


def check_transport(rep, atlas, assign, X):
    import torch

    from infra.pg import pragnanz_core as PC

    # identity correspondence on the SAME atlas => zero shift
    ident = list(range(atlas["mu"].shape[0]))
    z = PC.transport_shift(atlas, atlas, assign, ident)
    assert float(z.abs().max()) == 0.0, "identity transport not a no-op"
    # planted offset: dst = src means + delta => transported tokens shift by exactly delta
    delta = torch.randn(atlas["mu"].shape[1], generator=torch.Generator().manual_seed(3))
    dst = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in atlas.items()}
    dst["mu"] = atlas["mu"] + delta
    sh = PC.transport_shift(atlas, dst, assign, ident)
    assert torch.allclose(sh, delta.expand_as(sh), atol=1e-6), "planted-delta transport wrong"
    # matched-random arm: per-token norms EQUAL, directions uncorrelated
    rnd = PC.random_transport_shift(sh, seed=4)
    assert torch.allclose(rnd.norm(dim=1), sh.norm(dim=1), atol=1e-4), "random arm norm mismatch"
    cos = (rnd * sh).sum(1) / (rnd.norm(dim=1) * sh.norm(dim=1)).clamp(min=1e-30)
    assert abs(float(cos.mean())) < 0.1, f"random arm correlated with real: {float(cos.mean())}"
    rep["transport"] = {"identity_noop": True, "delta_exact": True,
                        "rand_cos_mean": float(cos.mean())}
    print("[PZ] transport PASS", flush=True)


def check_projector(rep):
    import torch

    from infra.pg import pragnanz_core as PC

    g = torch.Generator().manual_seed(0)
    Q = PC.haar_basis(24, 4, seed=1)
    G = torch.randn(24, 16, generator=g)
    R = PC.project_out_grad(G, Q)
    assert float((Q.T @ R).abs().max()) < 1e-5, "in-span not removed (out-axis)"
    P = torch.eye(24) - Q @ Q.T
    assert torch.allclose(R, P @ G, atol=1e-5), "projector wrong (out-axis)"
    G2 = torch.randn(16, 24, generator=g)
    R2 = PC.project_out_grad(G2, Q)
    assert float((R2 @ Q).abs().max()) < 1e-5, "in-span not removed (in-axis)"

    lin = torch.nn.Linear(24, 8, bias=False)
    lin.weight.grad = torch.randn(8, 24, generator=g)
    pj = PC.GradProjector({"weight": PC.haar_basis(24, 3, seed=2)})
    n = pj.apply(lin)
    assert n == 1
    rb = PC.random_bases_like({"a": Q, "b": None}, seed0=100)
    assert rb["a"].shape == Q.shape and rb["b"] is None
    assert not torch.allclose(rb["a"], Q), "random basis equals core basis"
    rep["projector"] = "pass"
    print("[PZ] projector PASS", flush=True)


def check_dome(rep):
    """P6: the estimator recovers a planted C-class scatter structure; the filter preserves a
    mean direction orthogonal to it; nuisance fractions separate from the rotation null (N1:
    the null's width is printed)."""
    import torch

    from infra.pg import pragnanz_core as PC

    d, C, B, T = 64, 4, 32, 120
    g = torch.Generator().manual_seed(0)
    Qc = PC.haar_basis(d, C, seed=5)              # planted scatter subspace
    m0 = PC.haar_basis(d, C + 1, seed=6)[:, C]    # mean direction, ~orthogonal to plants
    m0 = m0 - Qc @ (Qc.T @ m0)
    m0 = m0 / m0.norm()
    est = PC.DomeEstimator(d, k=C, seed=7)
    for _ in range(T):
        cls = torch.randint(0, C, (B,), generator=g)
        s = Qc[:, cls].T * 3.0                                    # (B, d) class scatter
        G = m0.unsqueeze(0) + s + 0.1 * torch.randn(B, d, generator=g)
        est.update(G)
    # PZ-D1: the CENTERED scatter of C balanced classes has rank C−1 (the within-span mean
    # direction has zero variance) — the recovery criterion targets that C−1-dim TRUE eigenspace,
    # never the full C-dim plant. Estimator keeps k=C (one over-parameterized junk dim, as
    # DOME's own k=C² over-parameterizes).
    one = torch.full((C,), C ** -0.5)
    Pc = torch.eye(C) - torch.outer(one, one)
    w, V = torch.linalg.eigh(Pc)
    Qtrue = Qc @ V[:, w > 0.5]                                    # (d, C−1)
    top = est.U[:, torch.argsort(-est.Lam)[: C - 1]]
    ov = torch.linalg.svdvals(top.T @ Qtrue).min()
    null_ovs = [float(torch.linalg.svdvals(PC.haar_basis(d, C - 1, seed=100 + i).T
                                           @ Qtrue).min()) for i in range(24)]
    null_hi = max(null_ovs)
    print(f"[PZ][N1] DOME rotation-null min-cos width: [{min(null_ovs):.3f}, {null_hi:.3f}]",
          flush=True)
    assert null_hi < 0.9, "rotation null saturated — cannot certify recovery"
    assert float(ov) > 0.98 > null_hi, f"plant not recovered: min-cos {float(ov)} vs null {null_hi}"
    # filter preserves the orthogonal mean direction (PZ-D2: the scatter part of the test
    # vector is drawn in the REMOVABLE Qtrue space — a Qc·1 component lies in the zero-variance
    # direction the estimator correctly cannot see and would survive filtering by design).
    # Threshold 0.90: the k-th (junk) estimator column removes an O(1/sqrt(d)) sliver of m0
    # (mirror-measured 0.954); a REAL filter failure (removing a true-space mean component)
    # lands ~0.85.
    gm = m0 * 2.0 + Qtrue @ torch.randn(C - 1, generator=g)
    filt = est.filter(gm)
    cos = float((filt / filt.norm()) @ m0)
    assert cos > 0.90, f"filter destroyed the mean direction: cos {cos}"
    # nuisance fraction separates in-plant vs random (in-plant drawn in the TRUE C−1 space)
    frac_in = est.nuisance_fraction(Qtrue @ torch.randn(C - 1, generator=g))
    frac_rand = est.nuisance_fraction(PC.haar_basis(d, 1, seed=200)[:, 0])
    assert frac_in > 0.9 and frac_rand < 0.6, f"fractions do not separate: {frac_in} {frac_rand}"
    rep["dome"] = {"plant_min_cos": float(ov), "null_hi": null_hi, "filter_cos": cos,
                   "frac_in": frac_in, "frac_rand": frac_rand}
    print(f"[PZ] DOME PASS plant={float(ov):.3f} null_hi={null_hi:.3f} filter_cos={cos:.3f}",
          flush=True)


def check_mean_cancel(rep):
    """M1's known event: a planted tug-of-war (alternating-sign in-core component) yields
    ratio << the drifting control's."""
    import torch

    from infra.pg import pragnanz_core as PC

    g = torch.Generator().manual_seed(0)
    T, k = 200, 8
    u = torch.randn(k, generator=g)
    signs = torch.tensor([1.0 if t % 2 == 0 else -1.0 for t in range(T)])
    V_tug = signs.unsqueeze(1) * u.unsqueeze(0) + 0.2 * torch.randn(T, k, generator=g)
    V_drift = u.unsqueeze(0) + 0.2 * torch.randn(T, k, generator=g)
    r_tug = PC.mean_cancel_ratio(V_tug)
    r_drift = PC.mean_cancel_ratio(V_drift)
    assert r_tug["ratio"] < 0.05 < 5.0 < r_drift["ratio"], \
        f"mean-cancel does not separate: tug {r_tug['ratio']} drift {r_drift['ratio']}"
    rep["mean_cancel"] = {"tug": r_tug["ratio"], "drift": r_drift["ratio"]}
    print(f"[PZ] mean-cancel PASS tug={r_tug['ratio']:.4f} drift={r_drift['ratio']:.1f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "PRAGNANZ offline gate"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    _w()
    atlas, assign, X = check_ladder(rep)
    _w()
    check_transport(rep, atlas, assign, X)
    _w()
    check_projector(rep)
    _w()
    check_dome(rep)
    _w()
    check_mean_cancel(rep)
    _w()
    print("PRAGNANZ_OFFLINE_OK")


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)     # D7 (GESTALT): extension-thread finalization abort
