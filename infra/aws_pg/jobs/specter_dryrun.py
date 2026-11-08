"""SPECTER D-suite (CPU, weight-free) — every pure surface unit-tested against dense references,
every §6 branch constructed (INCLUDING each rule's failure branch), the differentiation paths
validated on a toy model against a dense GGN, and the historical events replayed through the new
instruments:

  * the ATLAS R-T1 range-guard misfire (mismatched floor) — classified correctly under the
    construction-matched-reference rule                       → matched floor passes, foreign VOIDs
  * REGRAIN R1's saturated em-null through `saturation_guard` → must VOID
  * the ATLAS R12 d_eff convention event                      → raw reads ≈1, the frozen log
    convention reads the corrected (higher) dimension
  * the A3 monotone-proxy event                               → one-sided proxy inert on symmetric
    damage, live on a truly one-sided family
  * a PLANTED ITEM-CONDITIONAL operator                       → split-half MUST fail (H_field is
    reachable, not rhetorical)

Run on a pg r7i CPU instance (never the laptop):
  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/specter_dryrun --budget-min 60
Fail-loud; `green` is written LAST.
"""
import argparse
import json
import math
import pathlib
import time

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ===========================================================================
# The toy model — a 2-layer residual stack with a genuine nonlinearity, so J is
# input-dependent and the GGN ≠ Hessian gap is real. d=8, V=13, T=6.
# ===========================================================================
def _toy(seed=0, d=8, V=13, T=6, n_layers=3):
    import torch

    class Block(torch.nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.lin = torch.nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.lin.weight.copy_(torch.randn(d, d, generator=gen) / math.sqrt(d))

        def forward(self, hs):
            return hs + torch.tanh(self.lin(hs))

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            gen = torch.Generator().manual_seed(seed)
            self.layers = torch.nn.ModuleList([Block(gen) for _ in range(n_layers)])
            self.head = torch.nn.Linear(d, V, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(torch.randn(V, d, generator=gen) / math.sqrt(d))
            self.x0 = torch.randn(1, T, d, generator=gen)

        def forward(self):
            h = self.x0
            for blk in self.layers:
                h = blk(h)
            return self.head(h)

    return Toy().eval()


def _toy_item(model, L=1, uniform_w=True):
    import torch

    from infra.pg import specter_core as SC
    T = model.x0.shape[1]
    V = model.head.out_features
    install, holder = SC.make_residual_injector(model.layers, L)
    holder["delta"] = torch.zeros(T, model.x0.shape[2])

    def forward_fn():
        with install():
            return model()
    w = torch.full((T,), 1.0 / T) if uniform_w else torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    gen = torch.Generator().manual_seed(99)
    labels = torch.randint(0, V, (T,), generator=gen)
    labels[w == 0] = -100
    return {"forward_fn": forward_fn, "holder": holder, "loss_w": w, "T": T, "labels": labels}


def _dense_ggn(model, item, d=8):
    """Dense JᵀH_out J over the (T·d) site space, from autograd jacobian — the reference."""
    import torch
    T = item["T"]
    holder = item["holder"]

    def f(dflat):
        holder["delta"] = dflat.reshape(T, d)
        return item["forward_fn"]().reshape(-1)
    J = torch.autograd.functional.jacobian(f, torch.zeros(T * d), vectorize=True)  # (T*V, T*d)
    with torch.no_grad():
        holder["delta"] = torch.zeros(T, d)
        logits = item["forward_fn"]()[0]
    V = logits.shape[1]
    p = torch.softmax(logits.double(), dim=-1)
    G = torch.zeros(T * d, T * d, dtype=torch.float64)
    for t in range(T):
        wt = float(item["loss_w"][t])
        if wt == 0:
            continue
        H = torch.diag(p[t]) - torch.outer(p[t], p[t])
        Jt = J.double()[t * V:(t + 1) * V]
        G += wt * Jt.transpose(0, 1) @ H @ Jt
    return G


# ===========================================================================
# 1) streaming Fisher product vs dense (toy vocab).
# ===========================================================================
@check
def fisher_streaming_vs_dense():
    import torch
    from infra.pg.specter_core import fisher_logits_product, fisher_quad_from_tangent
    gen = torch.Generator().manual_seed(1)
    T, V = 6, 50
    logits = torch.randn(T, V, generator=gen) * 3
    u = torch.randn(T, V, generator=gen)
    w = torch.tensor([0.0, 0.25, 0.25, 0.25, 0.25, 0.0])
    p = torch.softmax(logits.double(), dim=-1)
    ref = torch.zeros(T, V, dtype=torch.float64)
    qref = 0.0
    for t in range(T):
        H = torch.diag(p[t]) - torch.outer(p[t], p[t])
        ref[t] = float(w[t]) * (H @ u[t].double())
        qref += float(w[t] * (u[t].double() @ H @ u[t].double()))
    got = fisher_logits_product(logits, u, w, chunk=7)          # chunk < V forces streaming
    assert float((got.double() - ref).abs().max()) < 1e-5, "streaming Fisher product != dense"
    q = fisher_quad_from_tangent(logits, u, w, chunk=7)
    assert abs(q - qref) < 1e-5 * max(abs(qref), 1.0), f"streaming quadform {q} != dense {qref}"


# ===========================================================================
# 2) GGN-vp + quadform: BOTH differentiation paths vs the dense reference.
# ===========================================================================
@check
def ggn_paths_vs_dense():
    import torch

    from infra.pg import specter_core as SC
    model = _toy()
    item = _toy_item(model, uniform_w=False)
    d, T = 8, item["T"]
    G = _dense_ggn(model, item)
    gen = torch.Generator().manual_seed(3)
    delta = torch.randn(T, d, generator=gen)
    ref_w = (G @ delta.double().reshape(-1)).reshape(T, d)
    ref_q = float(delta.double().reshape(-1) @ G @ delta.double().reshape(-1))
    for path in ("func", "manual"):
        w = SC.ggn_vp_item(item, delta, path=path)
        assert float((w.double() - ref_w).abs().max()) < 1e-4, f"ggn_vp[{path}] != dense"
        q = SC.ggn_quadform(item, delta, path=path)
        assert abs(q - ref_q) < 1e-4 * max(abs(ref_q), 1.0), f"quadform[{path}] {q} != {ref_q}"
    # mode_matvec assembles (1/T)·E_ωᵀ G E_ω v
    v = torch.randn(d, generator=gen)
    for mode, seed in (("dc", None), ("sin4", None), ("white", 211)):
        c = SC.env_coeffs(mode, T, sign_seed=seed)
        E = torch.zeros(T * d, d, dtype=torch.float64)
        for t in range(T):
            E[t * d:(t + 1) * d] = float(c[t]) * torch.eye(d, dtype=torch.float64)
        ref = (E.transpose(0, 1) @ G @ E @ v.double()) / T
        got = SC.mode_matvec([item], v, mode, sign_seed=seed)
        assert float((got.double() - ref).abs().max()) < 1e-4, f"mode_matvec[{mode}] != dense"
        qm = SC.mode_quadform([item], v, mode, sign_seed=seed)
        qr = float(v.double() @ ref)
        assert abs(qm - qr) < 1e-4 * max(abs(qr), 1e-9), f"mode_quadform[{mode}] {qm} != {qr}"


@check
def fd_wiring_check_both_paths():
    from infra.pg import specter_core as SC
    import torch
    model = _toy()
    item = _toy_item(model)
    gen = torch.Generator().manual_seed(4)
    for path in ("func", "manual"):
        for k in range(4):
            delta = torch.randn(item["T"], 8, generator=gen) * 0.5
            # h_rel=5e-2 keeps the KL well above the toy's fp32 log-prob noise floor while the
            # symmetric difference kills the cubic term; the live Gate-0 check runs at h=1
            q, q_fd, rel = SC.fd_directional_check(item, delta, path=path, h_rel=5e-2)
            assert rel < SC.FD_REL_TOL, f"FD check[{path}] rel={rel} (q={q}, fd={q_fd})"


@check
def hessian_vp_and_gap():
    import torch

    from infra.pg import specter_core as SC
    model = _toy()
    item = _toy_item(model)
    T, d = item["T"], 8
    holder = item["holder"]

    def loss_of(dflat):
        holder["delta"] = dflat.reshape(T, d)
        logits = item["forward_fn"]()[0]
        lab = item["labels"]
        live = lab != -100
        lse = torch.logsumexp(logits[live].float(), dim=-1)
        nll = lse - logits[live].float().gather(1, lab[live].unsqueeze(1)).squeeze(1)
        return (nll * item["loss_w"][live]).sum()
    Href = torch.autograd.functional.hessian(loss_of, torch.zeros(T * d)).double()
    gen = torch.Generator().manual_seed(5)
    delta = torch.randn(T, d, generator=gen)
    hv = SC.hessian_vp_item(item, delta)
    ref = (Href @ delta.double().reshape(-1)).reshape(T, d)
    assert float((hv.double() - ref).abs().max()) < 1e-4, "hessian_vp != autograd hessian"
    # the GGN−Hessian gap is real on a nonlinear model: the two quadforms differ
    G = _dense_ggn(model, item)
    qh = float(delta.double().reshape(-1) @ Href @ delta.double().reshape(-1))
    qg = float(delta.double().reshape(-1) @ G @ delta.double().reshape(-1))
    assert abs(qh - qg) > 1e-9, "toy model shows no GGN gap — reference degenerate"


# ===========================================================================
# 3) Lanczos: planted spikes + planted kernel recovered; full reorth holds.
# ===========================================================================
def _planted_operator(d=96, spikes=((0, 50.0), (1, 20.0)), bulk=1.0, kernel_dim=8, seed=7):
    import torch
    gen = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    lam = torch.full((d,), bulk, dtype=torch.float64)
    for i, v in spikes:
        lam[i] = v
    if kernel_dim > 0:                      # NB: lam[-0:] would zero EVERYTHING
        lam[-kernel_dim:] = 0.0
    H = Q @ torch.diag(lam) @ Q.transpose(0, 1)
    return H, Q, lam


@check
def lanczos_planted_spike_and_kernel():
    import torch

    from infra.pg import specter_core as SC
    H, Q, lam = _planted_operator()
    mv = lambda v: (H.float() @ v.double().float())
    res = SC.lanczos(mv, 96, m=60, seed=11)
    assert abs(float(res["evals"][0]) - 50.0) < 1e-3, f"top eval {res['evals'][0]} != 50"
    assert abs(float(res["evals"][1]) - 20.0) < 1e-3
    ov0 = abs(float(res["evecs"][:, 0].double() @ Q[:, 0]))
    ov1 = abs(float(res["evecs"][:, 1].double() @ Q[:, 1]))
    assert ov0 > 0.999 and ov1 > 0.999, f"planted spike eigvecs not recovered ({ov0}, {ov1})"
    # kernel: quadform of a planted kernel vector is ~0 while bulk is ~1 (tolerance sized to
    # fp32 matmul roundoff at ‖H‖ ≈ 50: ~1e-5; still 3 orders below the bulk)
    kq = float(Q[:, -1].float() @ mv(Q[:, -1].float()))
    assert abs(kq) < 1e-3, f"planted kernel not near-zero ({kq})"
    # orthonormality of the Ritz basis (full reorth held)
    E = res["evecs"][:, :30].double()
    gram_err = float((E.transpose(0, 1) @ E - torch.eye(30, dtype=torch.float64)).abs().max())
    assert gram_err < 1e-4, f"Ritz basis lost orthonormality ({gram_err})"


@check
def split_half_stability_and_planted_item_conditional_failure():
    import torch

    from infra.pg import specter_core as SC
    # (a) a STABLE operator: two half-estimates = same operator + small noise → spike survives
    H, Q, lam = _planted_operator(seed=21)
    gen = torch.Generator().manual_seed(22)
    N1 = torch.randn(96, 96, generator=gen, dtype=torch.float64) * 0.02
    N2 = torch.randn(96, 96, generator=gen, dtype=torch.float64) * 0.02
    Ha = H + N1 @ N1.transpose(0, 1) * 0.01
    Hb = H + N2 @ N2.transpose(0, 1) * 0.01
    ra = SC.lanczos(lambda v: (Ha.float() @ v), 96, m=60, seed=1)
    rb = SC.lanczos(lambda v: (Hb.float() @ v), 96, m=60, seed=2)
    spikes, rows = SC.stable_spikes(ra, rb, lambda u: float(u.double() @ H @ u.double()), k=8)
    assert len(spikes) >= 2, f"stable planted spikes not promoted ({len(spikes)})"
    # (b) the PLANTED ITEM-CONDITIONAL operator: halves see DIFFERENT spike directions →
    # split-half must FAIL and R-O1 must land H_field (the reachable failure branch)
    gen2 = torch.Generator().manual_seed(23)
    Qa, _ = torch.linalg.qr(torch.randn(96, 96, generator=gen2, dtype=torch.float64))
    Qb, _ = torch.linalg.qr(torch.randn(96, 96, generator=gen2, dtype=torch.float64))
    lam_f = torch.ones(96, dtype=torch.float64)
    lam_f[0] = 50.0
    Hfa = Qa @ torch.diag(lam_f) @ Qa.transpose(0, 1)
    Hfb = Qb @ torch.diag(lam_f) @ Qb.transpose(0, 1)
    rfa = SC.lanczos(lambda v: (Hfa.float() @ v), 96, m=60, seed=3)
    rfb = SC.lanczos(lambda v: (Hfb.float() @ v), 96, m=60, seed=4)
    spikes_f, _ = SC.stable_spikes(rfa, rfb,
                                   lambda u: float(u.double() @ (0.5 * (Hfa + Hfb)) @ u.double()),
                                   k=8)
    assert len(spikes_f) == 0, "item-conditional operator produced a 'stable' spike — split-half broken"
    r = SC.adjudicate_ro1(n_stable=0, converged=True, restart_used=False)
    assert r["verdict"] == "H_field", r


@check
def near_degenerate_cluster_span_repair():
    import torch

    from infra.pg import specter_core as SC
    # two near-equal eigenvalues whose eigvecs ROTATE within the shared span across halves:
    # vector overlap flaps, cluster-SPAN overlap stays ≈1 (the frozen repair)
    d = 64
    gen = torch.Generator().manual_seed(31)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    lam = torch.ones(d, dtype=torch.float64)
    lam[0], lam[1] = 30.0, 30.0001                        # a degenerate pair
    H = Q @ torch.diag(lam) @ Q.transpose(0, 1)
    th = 0.7
    R = torch.eye(d, dtype=torch.float64)
    R[:2, :2] = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]],
                             dtype=torch.float64)
    H2 = (Q @ R) @ torch.diag(lam) @ (Q @ R).transpose(0, 1)
    ra = SC.lanczos(lambda v: (H.float() @ v), d, m=48, seed=5)
    rb = SC.lanczos(lambda v: (H2.float() @ v), d, m=48, seed=6)
    rows = SC.match_cluster_overlaps(ra, rb, k=8)
    top = rows[0]
    assert top["hi"] - top["lo"] == 2, f"degenerate pair not clustered: {top}"
    assert top["overlap"] > 0.99, f"cluster-span overlap must survive rotation ({top['overlap']})"
    vec_ov = abs(float(ra["evecs"][:, 0].double() @ rb["evecs"][:, 0].double()))
    assert vec_ov < 0.9, f"vector overlap should flap under rotation ({vec_ov}) — test degenerate"


# ===========================================================================
# 4) spectral completion + K-FAC pullback vs dense.
# ===========================================================================
@check
def spectral_completion_vs_dense():
    import torch

    from infra.pg import specter_core as SC
    H, Q, lam = _planted_operator(d=96, spikes=((0, 40.0), (1, 15.0)), bulk=0.5, kernel_dim=0,
                                  seed=41)
    basis = Q                                              # pretend Σ-eigenbasis = H's basis
    sb = [(0, 8), (8, 32), (32, 96)]
    traces = [float(lam[lo:hi].sum()) for lo, hi in sb]
    sU = Q[:, :2].float()
    sL = [40.0, 15.0]
    bulk = SC.band_bulk_from_traces(traces, sb, sU, sL, basis)
    gen = torch.Generator().manual_seed(42)
    errs = []
    for _ in range(8):
        v = torch.randn(96, generator=gen, dtype=torch.float64)
        v = v / v.norm()
        qhat = SC.spectral_completion_quadform(v, sU, sL, bulk, basis, sb)
        qref = float(v @ H @ v)
        errs.append(abs(qhat - qref) / max(qref, 1e-12))
    # bulk here is EXACTLY flat within bands, so completion is near-exact
    assert max(errs) < 0.05, f"spectral completion errs {errs}"


@check
def kfac_stiffness_vs_dense():
    import torch

    from infra.pg import specter_core as SC
    d_in, d_out = 24, 96
    H, Q, lam = _planted_operator(d=d_out, spikes=((0, 10.0),), bulk=0.3, kernel_dim=0, seed=51)
    gen = torch.Generator().manual_seed(52)
    A = torch.randn(d_in, d_in, generator=gen, dtype=torch.float64)
    Sigma_y = A @ A.transpose(0, 1) / d_in
    dW = torch.randn(d_out, d_in, generator=gen, dtype=torch.float64) * 0.1
    ref = float(torch.trace(dW @ Sigma_y @ dW.transpose(0, 1) @ H))
    sb = [(0, d_out)]
    bulk = SC.band_bulk_from_traces([float(lam.sum())], sb, Q[:, :1].float(), [10.0], Q)
    got = SC.kfac_stiffness(dW, Sigma_y, Q[:, :1].float(), [10.0], bulk, Q, sb)
    assert abs(got - ref) / abs(ref) < 0.05, f"kfac stiffness {got} vs dense {ref}"


# ===========================================================================
# 5) envelopes, sign-sets, projections, links.
# ===========================================================================
@check
def envelope_conventions():
    import torch

    from infra.pg import specter_core as SC
    for mode, seed in (("dc", None), ("sin1", None), ("sin4", None), ("white", 211)):
        c = SC.env_coeffs(mode, 512, sign_seed=seed)
        ms = float((c.double() ** 2).mean())
        assert abs(ms - 1.0) < 1e-6, f"{mode} not unit mean-square ({ms})"
    c1 = SC.env_coeffs("white", 512, sign_seed=211)
    c2 = SC.env_coeffs("white", 271, sign_seed=211)
    assert torch.equal(c1[:271], c2), "white sign-set prefix not stable across T"
    c3 = SC.env_coeffs("white", 512, sign_seed=212)
    assert not torch.equal(c1, c3), "the two sign-sets must differ"


@check
def energy_projections_and_link():
    import numpy as np
    import torch

    from infra.pg import specter_core as SC
    gen = torch.Generator().manual_seed(61)
    delta = torch.randn(7, 16, generator=gen)
    out = SC.project_pertoken_energy(delta, 0.35)
    n = out.norm(dim=-1)
    assert float((n - 0.35).abs().max()) < 1e-6, "per-token projection inexact"
    q = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    y = np.log10(q) * 2 + 10 + np.array([0.01, -0.01, 0.02, 0.0, -0.02, 0.01])
    link = SC.fit_link(q, y)
    pred = SC.predict_link(link, q)
    assert np.all(np.diff(pred) >= -1e-12), "link not monotone"
    rho = np.corrcoef(pred, y)[0, 1]
    assert rho > 0.99, f"link fit degenerate (rho={rho})"


# ===========================================================================
# 6) every §6 branch constructed (each failure branch reachable).
# ===========================================================================
@check
def all_rule_branches_construct():
    from infra.pg import specter_core as SC
    A = SC.adjudicate_ro1
    assert A(n_stable=3, converged=True, restart_used=False)["verdict"] == "H_spikes"
    assert A(n_stable=0, converged=True, restart_used=False)["verdict"] == "H_field"
    assert A(n_stable=0, converged=False, restart_used=True)["verdict"] == "VOID-nonconverged"

    K = SC.adjudicate_ro2_key
    assert K(mean_rayleigh=1e-5, bulk_mean=1.0)["verdict"] == "H_theorem-verified"
    assert K(mean_rayleigh=0.01, bulk_mean=1.0)["verdict"] == "H_measured-leak"
    assert K(mean_rayleigh=0.5, bulk_mean=1.0)["verdict"] == "HALT-instrument"
    R = SC.adjudicate_ro2_radial
    assert R(mean_rayleigh=0.01, bulk_mean=1.0)["verdict"] == "H_gauge-confirmed"
    assert R(mean_rayleigh=0.1, bulk_mean=1.0)["verdict"] == "H_skip-share-measured"
    assert R(mean_rayleigh=0.5, bulk_mean=1.0)["verdict"] == "H_radial-active-quadratic"

    C1 = SC.adjudicate_rc1
    common = dict(bh_pass_ppl=True, bh_pass_vqa=True, u_dc_stable=True)
    assert C1(tau_ppl=0.73, p_ppl=0.008, tau_vqa=0.5, p_vqa=0.1,
              regressor_span_ratio=50, **common)["verdict"] == "H_carrier-named"
    assert C1(tau_ppl=0.1, p_ppl=0.5, tau_vqa=0.2, p_vqa=0.4,
              regressor_span_ratio=50, **common)["verdict"] == "H_spike-not-carrier"
    assert C1(tau_ppl=0.45, p_ppl=0.1, tau_vqa=0.4, p_vqa=0.2,
              regressor_span_ratio=50, **common)["verdict"] == "H_partial"
    assert C1(tau_ppl=0.9, p_ppl=0.001, tau_vqa=0.9, p_vqa=0.001,
              regressor_span_ratio=2, **common)["verdict"] == "VOID-no-range"

    C2 = SC.adjudicate_rc2
    assert C2(band1_white_ratio=40, band4_reversed_ratio=5)["verdict"] == "H_reversal-quadratic"
    assert C2(band1_white_ratio=40, band4_reversed_ratio=1)["verdict"] == "H_one-side"
    assert C2(band1_white_ratio=2, band4_reversed_ratio=1)["verdict"] == "H_reversal-nonquadratic"

    C3 = SC.adjudicate_rc3
    assert C3(R_l0=1.0, ratio_l0=1.2, R_l14=2.0, ratio_l14=2.3)["verdict"] == "H_dial-quadratic"
    assert C3(R_l0=1.0, ratio_l0=1.2, R_l14=2.0, ratio_l14=3.0)["verdict"] == "H_dial-partial"
    assert C3(R_l0=1.0, ratio_l0=2.0, R_l14=2.0, ratio_l14=3.0)["verdict"] == "H_dial-miss"

    C4 = SC.adjudicate_rc4
    assert C4({"u_dc~mean": 0.5}, {"u_dc~mean": 0.05})["verdict"] == "H_identified"
    assert C4({"u_dc~mean": 0.08}, {"u_dc~mean": 0.05})["verdict"] == "H_new-structure"

    K1 = SC.adjudicate_rk1
    assert K1(kfac_rho=0.9, qbar=0.98, haar_p05=0.05, haar_p95=0.95,
              quartile=0.9)["verdict"] == "H_core-stiff"
    assert K1(kfac_rho=0.9, qbar=0.02, haar_p05=0.05, haar_p95=0.95,
              quartile=0.1)["verdict"] == "H_core-sloppy"
    r = K1(kfac_rho=0.5, qbar=0.5, haar_p05=0.05, haar_p95=0.95, quartile=0.5)
    assert r["verdict"] == "H_core-generic" and r["scope"] == "direct-subsample-only"

    S1 = SC.adjudicate_rs1
    assert S1(damage_cilo_above_null=True, rotated_inside_band=True)["verdict"] == "H_behaviourally-real"
    assert S1(damage_cilo_above_null=True, rotated_inside_band=False)["verdict"] == "H_not-direction-specific"
    assert S1(damage_cilo_above_null=False, rotated_inside_band=True)["verdict"] == "H_spectral-only"
    assert S1(damage_cilo_above_null=True, rotated_inside_band=True, void=True)["verdict"] == "VOID"

    H1 = SC.adjudicate_rh1
    assert H1(G=1.5, Q_ratio=1.0, improved_factor=3.0)["verdict"] == "H_concentration-certified"
    assert H1(G=4.0, Q_ratio=1.2, improved_factor=3.0)["verdict"] == "H_quadratic-worst-case"
    assert H1(G=4.0, Q_ratio=5.0, improved_factor=3.0)["verdict"] == "H_boundary-dominated"
    assert H1(G=4.0, Q_ratio=0.2, improved_factor=3.0)["verdict"] == "H_subquadratic-found"
    assert H1(G=4.0, Q_ratio=5.0, improved_factor=1.05)["verdict"] == "VOID-flat"

    M1 = SC.adjudicate_rm1
    assert M1(drho=0.3, drho_cilo=0.05, drho_cihi=0.5, rho_m4=0.7, rho_m0=0.4,
              n_void_backups=2)["verdict"] == "H_operator-predicts"
    assert M1(drho=0.0, drho_cilo=-0.2, drho_cihi=0.05, rho_m4=0.5, rho_m0=0.62,
              n_void_backups=2)["verdict"] == "H_energy-sufficient-again"
    assert M1(drho=0.1, drho_cilo=-0.1, drho_cihi=0.3, rho_m4=0.6, rho_m0=0.55,
              n_void_backups=2)["verdict"] == "H_partial"
    assert M1(drho=0.1, drho_cilo=-0.1, drho_cihi=0.3, rho_m4=0.3, rho_m0=0.2,
              n_void_backups=2)["verdict"] == "H_neither"
    assert M1(drho=0.1, drho_cilo=0.2, drho_cihi=0.3, rho_m4=0.9, rho_m0=0.2,
              n_void_backups=11)["verdict"] == "VOID-battery"

    B1 = SC.adjudicate_rb1_collapse
    flat = {L: [0.5, 0.4, 0.3] for L in range(18)}
    assert B1(flat)["verdict"] == "H_collapse"
    bad = dict(flat)
    bad[6] = [0.9, 0.4, 0.3]
    r = B1(bad)
    assert r["verdict"] == "H_no-collapse" and r["peel_off_layer"] == 6, r


# ===========================================================================
# 7) historical replays.
# ===========================================================================
@check
def replay_rt1_range_guard_misfire():
    from infra.pg.specter_core import range_guard
    tang_response = 0.03
    floor_matched = 0.001        # measured on the SAME construction family
    floor_foreign = 0.02         # the misfire: a Gaussian-family floor gating a tangential family
    assert range_guard(tang_response, floor_matched), "matched floor must pass a live family"
    assert not range_guard(tang_response, floor_foreign), \
        "the mismatched floor must (wrongly) void — the event must be constructible"


@check
def replay_regrain_r1_saturated_em_null():
    from infra.pg.bispec_core import saturation_guard
    g = saturation_guard(damage_treat=0.001, damage_null=0.002, floor=0.0, intact=0.6,
                         name="replay_R1")
    assert g["void"], "the R1 saturated pair must VOID"
    g2 = saturation_guard(damage_treat=0.30, damage_null=0.55, floor=0.0, intact=0.6, name="live")
    assert not g2["void"]


@check
def replay_r12_deff_convention():
    from infra.pg.atlas_core import d_eff_from_damages
    from infra.pg.specter_core import d_eff_log_damage
    triple = [2624.17, 23.82, 52.42]                      # the banked P1@mid coherent triple
    raw = d_eff_from_damages(triple)["d_eff"]
    logd = d_eff_log_damage(triple)["d_eff"]
    assert raw < 2.0, f"raw convention must read ≈1 on the heavy-tailed triple ({raw})"
    assert logd > 3.0, f"the frozen log convention must read the corrected dimension ({logd})"


@check
def replay_a3_monotone_proxy():
    import numpy as np

    from infra.pg.atlas_fit import perm_p_tau
    alphas = np.array([-3, -2, -1, 1, 2, 3], float)
    onesided = np.maximum(alphas, 0.0)
    tau_sym, _ = perm_p_tau(np.abs(alphas) ** 2, onesided)
    assert tau_sym < 0.6, f"one-sided proxy must be inert on symmetric damage (τ={tau_sym})"
    tau_asym, p_asym = perm_p_tau(np.array([0.01, 0.012, 0.011, 0.1, 0.5, 2.0]), onesided)
    assert tau_asym >= 0.6 and p_asym <= 0.06


# ===========================================================================
# 8) census nulls + C-1 statistics machinery.
# ===========================================================================
@check
def census_haar_null_and_identification():
    import torch

    from infra.pg import specter_core as SC
    d = 2048
    gen = torch.Generator().manual_seed(71)
    lib = torch.randn(d, generator=gen, dtype=torch.float64)
    lib = lib / lib.norm()
    draws = torch.randn(4000, d, generator=gen, dtype=torch.float64)
    draws = draws / draws.norm(dim=1, keepdim=True)
    ovs = (draws @ lib).abs()
    p95 = float(ovs.quantile(0.95))
    assert 0.03 < p95 < 0.06, f"Haar p95 out of expected band ({p95})"
    aligned = float(lib @ lib)
    r = SC.adjudicate_rc4({"planted": aligned}, {"planted": p95})
    assert r["verdict"] == "H_identified"
    stray = float((draws[0] @ lib).abs())
    r2 = SC.adjudicate_rc4({"stray": stray}, {"stray": p95})
    assert r2["verdict"] == "H_new-structure"


@check
def c1_exact_permutation_legs():
    import numpy as np

    from infra.pg.atlas_fit import bh_pass, kendall_tau_b, perm_p_tau
    rng = np.random.default_rng(81)
    x6 = rng.normal(size=6)
    y6 = x6 * 2 + rng.normal(size=6) * 0.05
    t6, p6 = perm_p_tau(y6, x6)
    assert t6 >= 0.6 and p6 <= 0.06, f"6-pair exact leg ({t6}, {p6})"
    x9 = rng.normal(size=9)
    y9 = x9 + rng.normal(size=9) * 0.05
    t9, p9 = perm_p_tau(y9, x9)
    assert t9 >= 0.6 and p9 <= 0.05, f"9-pair exact leg ({t9}, {p9})"
    passed = bh_pass([p6, p9], q=0.10)
    assert passed.all(), "BH must pass two strong legs"
    # emitted p-values are the binding repair: the driver must be able to report them
    assert isinstance(p6, float) and isinstance(p9, float)


@check
def stiffness_quantile_machinery():
    import numpy as np

    from infra.pg.specter_core import stiffness_quantile
    rng = np.random.default_rng(91)
    ref = rng.normal(size=2000)
    hi, _ = stiffness_quantile(ref.max() + np.ones(4), ref)
    lo, _ = stiffness_quantile(ref.min() - np.ones(4), ref)
    mid, _ = stiffness_quantile(np.zeros(4), ref)
    assert hi == 1.0 and lo == 0.0 and 0.4 < mid < 0.6


# ===========================================================================
# 9) M4 q-functional mapping + adversarial selection determinism (offline-gate §0.5-4 half
# that needs no S3: the formulas and the algorithm, exercised on synthetic inputs).
# ===========================================================================
@check
def q_functional_production_mapping():
    from infra.aws_pg.jobs.specter_m_fit import q_m4_for
    permode = {"dc": [10.0] + [1.0] * 15, "sin4": [5.0] + [1.0] * 15,
               "white": [0.1] + [2.0] * 15}
    chan = {"k_low/dc": 0.001, "k_low/white": 3.0, "k_high/dc": 0.5, "k_high/white": 1.0,
            "v/dc": 2.0, "v/white": 0.2, "q/dc": 4.0, "q/white": 1.5}
    base = {k: 0.0 for k in ("is_delete", "log_energy", "dc_frac", "shared_dir", "mean_shift",
                             "radial_frac", "chan_none", "chan_k_low", "chan_k_high", "chan_v",
                             "chan_q", "retention", "v1_overlap_pos")}
    sb0 = [1.0] + [0.0] * 15
    dc = dict(base, log_energy=-2.0, dc_frac=1.0, shared_dir=1.0, chan_none=1.0)
    assert abs(q_m4_for(dc, sb0, permode, 7.0, chan) - 1e-2 * 10.0) < 1e-12
    wh = dict(base, log_energy=-2.0, chan_none=1.0)
    assert abs(q_m4_for(wh, sb0, permode, 7.0, chan) - 1e-2 * 0.1) < 1e-12
    rad = dict(dc, radial_frac=1.0)
    assert abs(q_m4_for(rad, sb0, permode, 7.0, chan) - 1e-2 * 7.0) < 1e-12
    kc = dict(base, log_energy=-2.0, dc_frac=1.0, chan_k_low=1.0)
    assert abs(q_m4_for(kc, [0.0] * 16, permode, 7.0, chan) - 1e-2 * 0.001) < 1e-12
    dl = dict(base, log_energy=-1.0, is_delete=1.0, shared_dir=1.0, dc_frac=1.0, chan_none=1.0)
    assert abs(q_m4_for(dl, sb0, permode, 7.0, chan) - 1e-1 * 10.0) < 1e-12


@check
def selection_deterministic_with_caps():
    import numpy as np

    from infra.pg.atlas_fit import COVERAGE_MAX_BANDMODE, enumerate_grammar, select_sealed
    sb_ranks = [1, 3, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    rmass = [0.5] * 16
    cands, _ = enumerate_grammar(sb_ranks, rmass, set())
    rng = np.random.default_rng(5)
    m4 = rng.normal(size=len(cands))
    m0 = rng.normal(size=len(cands))
    a = select_sealed(cands, m4, m0)
    b = select_sealed(cands, m4, m0)
    assert a == b, "selection not deterministic"
    from collections import Counter
    bm = Counter((cands[i]["subband"], cands[i]["mode"]) for i in a)
    assert max(bm.values()) <= COVERAGE_MAX_BANDMODE, "coverage cap violated"


# ===========================================================================
# runner
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "specter_dryrun_v1", "checks": []}

    def _w():
        (out / "dryrun.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()
    t_all = time.time()
    for fn in CHECKS:
        t0 = time.time()
        fn()
        rep["checks"].append({"name": fn.__name__, "ok": True, "seconds": time.time() - t0})
        print(f"[dryrun] PASS {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        _w()
    rep["n_checks"] = len(rep["checks"])
    rep["gpu_seconds"] = {"total": 0.0, "note": "CPU job"}
    rep["wall_seconds"] = time.time() - t_all
    print("SPECTER_DRYRUN_OK", json.dumps({"n": rep["n_checks"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
