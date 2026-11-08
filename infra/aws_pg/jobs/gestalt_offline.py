"""GESTALT offline gate — weight-free dry-run of every NEW instrument (spec §9 step 2).
No model, no weights, no network. Each check replays a KNOWN event and RAISES on failure.

Checks (P-numbers per spec §5):
  P1-logic   planted circle / two-cluster / curved 2-manifold recovered by the fitter, and the
             SAME plants fitted on their moment-matched twins must NOT appear (both directions).
             Also FREEZES the rank r (smallest candidate passing all three plants) — the spec's
             "r frozen at dry-run BEFORE any real fit".
  P2-logic   twin generator determinism + moment reproduction + twin-of-twin fit consistency.
  P3-algebra r=d reconstruction == identity; SplatSwapHook == pure reconstruct; tuple rebuild;
             dim-mismatch RAISES; EM divergence VOID path fires on poisoned data.
  P4-logic   correspondence: planted survive/move/die/merge recovered; split-half band has
             NONZERO width; chance band sits above it; Bures^2 analytic 1-D check.
  F-logic    probe determinism + endpoint algebra; ridge recovers a planted linear field;
             the injected-field integrator matches the closed-form ODE; field_diff zero/scale.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_offline --type r7i.2xlarge --no-weights \
      --s3-prefix results/gestalt_offline --budget-min 60

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib

R_CANDIDATES = (2, 4, 8)
D_SYN = 32
N_SYN = 4000


def check_p1_plants(rep):
    import torch

    from infra.pg import gestalt_splat as GS

    verdicts = {}
    for r_cand in R_CANDIDATES:
        ok = {}
        # ---- circle: K=8 fit; real means arranged on a circle, twin means NOT ----
        X = GS.make_circle(N_SYN, D_SYN, seed=1)
        m, C = GS.stream_moments(X)
        L = GS.cov_sym_sqrt(C)
        stats = {"real": [], "twin": []}
        for s in GS.EM_SEEDS:
            a_real = GS.fit_mppca(X, K=8, r=r_cand, seed=s)
            assert not a_real["void"], f"circle real fit VOID at r={r_cand} s={s}"
            stats["real"].append(GS.circle_stat(a_real, X))
            tw = GS.twin_sample(m, L, X.shape[0], seed=100 + s)
            a_twin = GS.fit_mppca(tw, K=8, r=r_cand, seed=s)
            assert not a_twin["void"], f"circle twin fit VOID at r={r_cand} s={s}"
            stats["twin"].append(GS.circle_stat(a_twin, X))
        real_worst = max(stats["real"])
        twin_best = min(stats["twin"])
        ok["circle"] = {"real_cv": stats["real"], "twin_cv": stats["twin"],
                        "detected": bool(real_worst < 0.5 * twin_best)}
        # ---- two-cluster: the DISCRIMINATOR is the K=1->2 log-likelihood GAIN (large only
        #      for genuine bimodality; a K=2 fit on the elongated-Gaussian twin splits along
        #      the separation axis and would score high PURITY spuriously — purity is only a
        #      real-side sanity check here, never the twin discriminator) ----
        X2, lab = GS.make_two_cluster(N_SYN, D_SYN, seed=2)
        m2, C2 = GS.stream_moments(X2)
        L2 = GS.cov_sym_sqrt(C2)
        gain_r, gain_t, pur_r = [], [], []
        for s in GS.EM_SEEDS:
            a1 = GS.fit_mppca(X2, K=1, r=r_cand, seed=s)
            a2 = GS.fit_mppca(X2, K=2, r=r_cand, seed=s)
            assert not a1["void"] and not a2["void"]
            gain_r.append(GS.ll_per_point(a2, X2) - GS.ll_per_point(a1, X2))
            pur_r.append(GS.cluster_purity(a2, X2, lab))
            tw = GS.twin_sample(m2, L2, X2.shape[0], seed=200 + s)
            t1 = GS.fit_mppca(tw, K=1, r=r_cand, seed=s)
            t2 = GS.fit_mppca(tw, K=2, r=r_cand, seed=s)
            assert not t1["void"] and not t2["void"]
            gain_t.append(GS.ll_per_point(t2, tw) - GS.ll_per_point(t1, tw))
        ok["two_cluster"] = {"ll_gain_real": gain_r, "ll_gain_twin": gain_t,
                             "real_purity": pur_r,
                             "detected": bool(min(gain_r) > 2.0 * max(gain_t)
                                              and min(pur_r) >= 0.95)}
        # ---- curved 2-manifold: K=16 means on the sheet for real, off it for twin ----
        X3 = GS.make_curved_manifold(N_SYN, D_SYN, seed=3)
        m3, C3 = GS.stream_moments(X3)
        L3 = GS.cov_sym_sqrt(C3)
        d_r, d_t = [], []
        for s in GS.EM_SEEDS:
            a = GS.fit_mppca(X3, K=16, r=r_cand, seed=s)
            assert not a["void"]
            d_r.append(GS.manifold_dist_stat(a, curv=0.35, seed=3, d=D_SYN))
            tw = GS.twin_sample(m3, L3, X3.shape[0], seed=300 + s)
            at = GS.fit_mppca(tw, K=16, r=r_cand, seed=s)
            assert not at["void"]
            d_t.append(GS.manifold_dist_stat(at, curv=0.35, seed=3, d=D_SYN))
        ok["curved_manifold"] = {"real_dist": d_r, "twin_dist": d_t,
                                 "detected": bool(max(d_r) < 0.5 * min(d_t))}
        verdicts[r_cand] = ok
        del X, X2, X3, tw
    passing = [r for r in R_CANDIDATES
               if all(verdicts[r][p]["detected"] for p in ("circle", "two_cluster",
                                                           "curved_manifold"))]
    assert passing, f"P1 FAILED at every candidate rank: {json.dumps(verdicts, default=str)[:800]}"
    r_frozen = min(passing)
    rep["P1"] = {"verdicts": verdicts, "passing_ranks": passing, "r_frozen": r_frozen}
    print(f"[gestalt-offline] P1 PASS — r_frozen={r_frozen} (candidates passing: {passing})")
    return r_frozen


def check_p2_twin(rep, r_frozen):
    import torch

    from infra.pg import gestalt_splat as GS

    X = GS.make_two_cluster(N_SYN, D_SYN, seed=5)[0]
    m, C = GS.stream_moments(X)
    L = GS.cov_sym_sqrt(C)
    a = GS.twin_sample(m, L, 2000, seed=42)
    b = GS.twin_sample(m, L, 2000, seed=42)
    c = GS.twin_sample(m, L, 2000, seed=43)
    assert torch.equal(a, b), "twin generator NOT deterministic at fixed seed"
    assert not torch.equal(a, c), "twin generator ignores the seed"
    big = GS.twin_sample(m, L, 200_000, seed=44)
    mb, Cb = GS.stream_moments(big)
    # Mean error is normalized by the cloud's RMS radius sqrt(tr C), NOT by ||m||: a centered
    # plant has ||m|| ~ 0 and the ||m||-relative form explodes on a CORRECT twin (defect D1,
    # caught by this gate's own first run — journalled).
    scale = float(torch.sqrt(torch.trace(C)))
    mean_err = float((mb - m).norm()) / scale
    cov_err = float((Cb - C).norm() / C.norm().clamp(min=1e-9))
    assert mean_err < 0.05 and cov_err < 0.05, f"twin moments off: mean {mean_err} cov {cov_err}"
    # twin-of-twin: fits agree within the seed spread (P2's fit-consistency leg)
    lls = {}
    for tag, data_seed in (("twin", 46), ("twin_of_twin", 47)):
        src = GS.twin_sample(m, L, N_SYN, seed=data_seed)
        if tag == "twin_of_twin":
            m2, C2 = GS.stream_moments(GS.twin_sample(m, L, N_SYN, seed=48))
            src = GS.twin_sample(m2, GS.cov_sym_sqrt(C2), N_SYN, seed=data_seed)
        lls[tag] = [GS.fit_mppca(src, K=4, r=r_frozen, seed=s)["ll_trace"][-1] / N_SYN
                    for s in GS.EM_SEEDS]
    gap = abs(sum(lls["twin"]) / 2 - sum(lls["twin_of_twin"]) / 2)
    seed_spread = abs(lls["twin"][0] - lls["twin"][1]) + abs(
        lls["twin_of_twin"][0] - lls["twin_of_twin"][1])
    assert gap < 10 * (seed_spread + 0.05), \
        f"twin-of-twin per-point LL gap {gap:.4f} outside seed spread {seed_spread:.4f}"
    rep["P2"] = {"mean_rel_err": mean_err, "cov_rel_err": cov_err,
                 "ll_per_point": lls, "gap": gap, "seed_spread": seed_spread}
    print(f"[gestalt-offline] P2 PASS — twin deterministic; moments {mean_err:.3f}/{cov_err:.3f}; "
          f"twin-of-twin gap {gap:.4f}")


def check_p3_swap(rep, r_frozen):
    import torch

    from infra.pg import gestalt_splat as GS

    d = D_SYN
    # r=d identity: hand-built full-rank atlas => reconstruct == identity for ANY gamma
    g = torch.Generator().manual_seed(0)
    X = torch.randn(500, d, generator=g)
    eye = torch.eye(d)
    atlas_full = {"log_pi": torch.log(torch.tensor([0.3, 0.7])),
                  "mu": torch.randn(2, d, generator=g),
                  "U": torch.stack([eye, eye]),
                  "lam": torch.ones(2, d), "sigma2": torch.tensor([0.5, 0.5]),
                  "void": False}
    err = (GS.reconstruct(X, atlas_full) - X).abs().max()
    assert float(err) < 1e-4, f"r=d reconstruction is not the identity (max err {float(err)})"

    # hook == pure core, tuple rebuild, dim mismatch RAISES
    X2 = GS.make_two_cluster(2000, d, seed=6)[0]
    atlas = GS.fit_mppca(X2, K=2, r=r_frozen, seed=0)
    assert not atlas["void"]

    class _Blk(torch.nn.Module):
        def forward(self, x):
            return (x, "aux")

    blk = _Blk()
    h = X2[:64].reshape(4, 16, d)
    with GS.SplatSwapHook(blk, atlas):
        out = blk(h)
    assert isinstance(out, tuple) and out[1] == "aux", "hook broke the output tuple"
    pure = GS.reconstruct(h.reshape(-1, d), atlas).reshape(h.shape)
    assert torch.allclose(out[0], pure, atol=1e-4), "SplatSwapHook != pure reconstruct"
    bad = torch.randn(4, 16, d + 1)
    raised = False
    with GS.SplatSwapHook(blk, atlas):
        try:
            blk(bad)
        except RuntimeError:
            raised = True     # assert-raises probe (the house _assert_raises pattern), not hiding
    assert raised, "dim-mismatch did not RAISE"

    # reconstruction improves with K where local tangents ROTATE (the splat idea's own known
    # event): a circle at r=1 — K=1 captures one diameter, K=8 captures local arcs. (Defect D2:
    # the first version used two-cluster MSE, where K=1's rank-2 plane already contains the
    # separation axis and the improvement is structurally ~zero — the event could not fire.)
    X8 = GS.make_circle(4000, d, seed=7)
    mses = {}
    for K in (1, 8):
        a = GS.fit_mppca(X8, K=K, r=1, seed=0)
        assert not a["void"]
        mses[K] = float(((GS.reconstruct(X8, a) - X8) ** 2).mean())
    assert mses[8] < 0.5 * mses[1], f"K=8 did not beat K=1 on circle data at r=1: {mses}"

    # non-finite INPUT must RAISE (contract), never VOID (D3)
    Xn = X2.clone()
    Xn[0, 0] = float("nan")
    raised_input = False
    try:
        GS.fit_mppca(Xn, K=2, r=r_frozen, seed=0)
    except AssertionError:
        raised_input = True   # assert-raises probe, not hiding
    assert raised_input, "non-finite input did not RAISE"
    # divergence VOID path on VALID (finite) data whose fp32 E-step row-sums overflow: retry
    # then VOID. Scale 1e19: per-element squares (~1e38) stay finite through the init SVD, the
    # d-dim sums do not — the divergence is genuinely EM-side, not an input-contract violation.
    Xp = X2.clone() * 1e19
    v = GS.fit_mppca(Xp, K=2, r=r_frozen, seed=0)
    assert v["void"] and v["retry_used"], f"overflow fit did not VOID through the retry path: {v}"
    rep["P3"] = {"identity_max_err": float(err), "mse_by_K": mses,
                 "void_path": v["void_reason"]}
    print(f"[gestalt-offline] P3 PASS — identity {float(err):.2e}; K-improvement {mses}; VOID path fires")


def check_p4_corr(rep):
    import numpy as np

    from infra.pg import gestalt_corr as GC

    # Bures^2 analytic: 1-D variances a,b -> (sqrt(a)-sqrt(b))^2, embedded in d
    d = 24
    e = np.zeros((d, 1))
    e[0, 0] = 1.0
    for a_var, b_var in ((4.0, 1.0), (2.25, 2.25)):
        got = GC.bures2_in_span(e * np.sqrt(a_var), e * np.sqrt(b_var))
        want = (np.sqrt(a_var) - np.sqrt(b_var)) ** 2
        assert abs(got - want) < 1e-9, f"Bures 1-D analytic failed: {got} vs {want}"

    na = GC.synth_atlas(K=12, d=d, r=3, seed=0)
    jit = 0.05
    # split-half band: two jittered copies of the SAME atlas
    h1, _ = GC.perturb_atlas(na, [], [], jitter=jit, move_dist=0.0, seed=1)
    h2, _ = GC.perturb_atlas(na, [], [], jitter=jit, move_dist=0.0, seed=2)
    Cself = GC.cost_matrix(h1, h2)
    _, _, mc_self = GC.hungarian(Cself)
    band = GC.selfhalf_band(mc_self)
    assert not band["zero_width"], "P4 split-half band has ZERO width (identity-band failure)"
    # planted diff: move 2, kill 2 (+ a merge: two components moved ONTO component 0's mean)
    nb, truth = GC.perturb_atlas(na, move_idx=[3, 5], die_idx=[7, 9], jitter=jit,
                                 move_dist=3.0, seed=3)
    C = GC.cost_matrix(na, nb)
    _, ci, _ = GC.hungarian(C)
    chance = GC.shuffle_chance_costs(C, assign_cols=ci)
    tau_chance = float(np.percentile(chance, GC.TAU_CHANCE_PCTL))
    lab = GC.label_pair(C, band["tau_move"], tau_chance)
    assert not lab["void_banded"], f"chance band {tau_chance} not above tau_move {band['tau_move']}"
    got = lab["labels"]
    hits = sum(g == t for g, t in zip(got, truth))
    assert hits == len(truth), f"planted labels missed: got {got} want {truth}"
    # merge detection (plant per the detector's OWN definition — two SOURCE components close
    # together whose target hosts ONE component there, the freed slot elsewhere; D6: the first
    # plant moved a target component onto another, which is a MOVE, not a merge):
    na2 = {"mu": na["mu"].copy(), "S": na["S"].copy(), "weight": na["weight"].copy()}
    na2["mu"][4] = na2["mu"][0] + 0.01
    na2["S"][4] = na2["S"][0]
    nm, _ = GC.perturb_atlas(na2, [], [], jitter=jit, move_dist=0.0, seed=4)
    rng4 = np.random.default_rng(44)
    v = rng4.standard_normal(na["mu"].shape[1])
    nm["mu"][4] = 40.0 * v / np.linalg.norm(v)          # the freed slot leaves the site
    Cm = GC.cost_matrix(na2, nm)
    _, ci_m, _ = GC.hungarian(Cm)
    tau_chance_m = float(np.percentile(GC.shuffle_chance_costs(Cm, assign_cols=ci_m),
                                       GC.TAU_CHANCE_PCTL))
    labm = GC.label_pair(Cm, band["tau_move"], tau_chance_m)
    assert labm["frac_merged"] > 0, f"planted merge not flagged: {labm['labels']}"
    rep["P4"] = {"band": band, "tau_chance": tau_chance,
                 "planted_labels_recovered": hits, "merge_flagged": labm["frac_merged"]}
    print(f"[gestalt-offline] P4 PASS — band width {band['band_width']:.3g} > 0; "
          f"labels {hits}/{len(truth)}; merge flagged")


def check_field_logic(rep):
    import numpy as np
    import torch

    from infra.pg import gestalt_field as GF

    # probe determinism + endpoint algebra
    z1 = GF.probe_noise(4, 2, 5, 3)
    z2 = GF.probe_noise(4, 2, 5, 3)
    assert torch.equal(z1, z2), "probe noise not deterministic"
    a_ref = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(1))
    x1 = GF.probe_points(a_ref, z1, 1.0)
    x0 = GF.probe_points(a_ref, z1, 0.0)
    assert torch.allclose(x1, z1) and torch.allclose(x0, a_ref.unsqueeze(1).expand_as(z1)), \
        "probe endpoint algebra wrong"

    # integrator vs closed form: constant field v=c => x_final = z - c (dt=-1/N, N steps)
    c = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(2))
    z = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(3))
    out = GF.integrate(None, None, None, None, z, num_steps=10, field_fn=lambda x, t: c)
    assert torch.allclose(out, z - c, atol=1e-6), "integrator != closed form for constant field"
    # linear field v = x  =>  x' = -x  => x_final = z*(1-1/N)^N
    outl = GF.integrate(None, None, None, None, z, num_steps=10, field_fn=lambda x, t: x)
    assert torch.allclose(outl, z * (1 - 0.1) ** 10, atol=1e-5), "integrator wrong on linear field"

    # ridge recovers a planted linear field (known event)
    rng = np.random.default_rng(0)
    Dx, Dv, N = 15, 6, 400
    Phi = np.concatenate([np.ones((N, 1)), rng.standard_normal((N, Dx))], axis=1)
    W_true = rng.standard_normal((Dx + 1, Dv))
    V = Phi @ W_true + 1e-4 * rng.standard_normal((N, Dv))
    W = GF.fit_ridge(Phi, V, lam=1e-6)
    assert np.abs(W - W_true).max() < 1e-2, "ridge failed to recover the planted map"

    # feature shapes + exact param counts
    x_flat = np.zeros(10)
    zf = np.zeros(64)
    for lvl in GF.LADDER:
        f = GF.features(x_flat, 0.5, zf, lvl)
        want = GF.surrogate_params(lvl, 10, 1)
        assert f.shape[0] == want, f"{lvl}: feature dim {f.shape[0]} != count {want}"
    # field_diff: identical -> 0; scaled orthogonal-ish sanity
    vA = rng.standard_normal((32, 5, 3))
    dd = GF.field_diff(vA, vA, adim_valid=3, n_boot=200)
    assert dd["d_rel"] == 0.0, "field_diff of identical fields != 0"
    dB = GF.field_diff(vA, -vA, adim_valid=3, n_boot=200)
    assert abs(dB["d_rel"] - 2.0) < 1e-9, "field_diff scale wrong for v vs -v"
    rep["F_logic"] = {"ridge_max_err": float(np.abs(W - W_true).max()),
                      "d_rel_identical": dd["d_rel"], "d_rel_anti": dB["d_rel"]}
    print("[gestalt-offline] F-logic PASS — integrator exact; ridge recovers; probes deterministic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "GESTALT offline gate (weight-free dry-run, spec §9 step 2)"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    _w()
    r_frozen = check_p1_plants(rep)
    rep["r_frozen"] = r_frozen
    _w()
    check_p2_twin(rep, r_frozen)
    _w()
    check_p3_swap(rep, r_frozen)
    _w()
    check_p4_corr(rep)
    _w()
    check_field_logic(rep)
    _w()
    print(f"GESTALT_OFFLINE_OK r_frozen={r_frozen}")


if __name__ == "__main__":
    main()
