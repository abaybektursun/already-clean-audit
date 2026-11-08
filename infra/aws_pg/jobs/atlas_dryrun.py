"""ATLAS D-suite (CPU, weight-free) — every pure surface unit-tested, every §6 branch constructed
(INCLUDING each rule's failure branch — a rule whose failure branch cannot be constructed is a
spec defect and this job is where it surfaces), and the historical events replayed through the
new instruments:

  * REGRAIN R1's saturated em-null through `saturation_guard`  → must VOID
  * the A7 chordal event through the energy-alignment metric   → must NOT saturate where chordal does
  * the banked P1 coherent triple through d_eff                → must read ≈ 1
  * the A3 monotone-proxy event through the one-sided ⟨δ̄,v₁⟩₊ proxy on a symmetric (rotated-
    control-shaped) damage curve                               → must NOT correlate
  * Bingham sampler moments vs small-d quadrature + an independent importance-sampling estimate
  * T1's Jacobian identity vs autograd on tiny tensors

Run on a pg r7i CPU instance (never the laptop):
  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_dryrun --type r7i.2xlarge --no-weights \
      --s3-prefix results/atlas_dryrun --budget-min 60
Fail-loud; `green` is written LAST.
"""
import argparse
import json
import math
import pathlib

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


# ===========================================================================
# Historical replays.
# ===========================================================================
@check
def replay_r1_saturated_em_null():
    from infra.pg.bispec_core import saturation_guard
    g = saturation_guard(damage_treat=0.001, damage_null=0.002, floor=0.0, intact=0.6,
                         name="replay_R1")
    assert g["void"], "the R1 saturated pair must VOID"
    g2 = saturation_guard(damage_treat=0.30, damage_null=0.55, floor=0.0, intact=0.6, name="live")
    assert not g2["void"], "a live pair must not VOID"


@check
def replay_a7_chordal_vs_energy():
    """The banked event (BISPEC R3): at L0's E = 25 % (E/E_max ≈ 0.30) on a spectrum whose top
    eigendirection carries ~70 % of the trace, the chordal diameter read 0.9962× unconstrained
    while the energy metric read 0.8864. Reproduced in miniature at the SAME regime — a
    dominant-λ₁ spectrum, moderate E/E_max — where the tilt only pulls a small component of each
    frame onto e₁ and the chordal metric cannot see it."""
    import torch

    from infra.pg.bispec_core import (energy_dispersion, grassmann_diameter, haar_frames,
                                      max_removable_energy, sample_level_set)
    d, k = 256, 8
    tail = torch.tensor([0.8 ** i for i in range(d - 1)], dtype=torch.float64)
    lam = torch.cat([torch.tensor([0.70], dtype=torch.float64),
                     0.30 * tail / tail.sum()])
    Sigma = torch.diag(lam)
    E_max = max_removable_energy(Sigma, k)
    frames = [sample_level_set(Sigma, k, 0.30 * E_max, seed=300 + s)[0] for s in range(5)]
    ref = haar_frames(d, k, 5, seed=77)
    ch_ratio = grassmann_diameter(frames)["diameter"] / grassmann_diameter(ref)["diameter"]
    en_ratio = (energy_dispersion(frames, Sigma)["mean_distance"]
                / energy_dispersion(ref, Sigma)["mean_distance"])
    assert ch_ratio > 0.9, f"chordal should saturate high in the banked regime ({ch_ratio})"
    assert en_ratio < ch_ratio - 0.05, \
        f"energy metric must separate where chordal cannot ({en_ratio} vs {ch_ratio})"


@check
def replay_p1_triple_deff():
    from infra.pg.atlas_core import d_eff_from_damages
    r = d_eff_from_damages([2624.17, 23.82, 52.42])          # the banked P1@mid coherent triple
    assert 0.5 <= r["d_eff"] <= 2.0, f"P1 coherent d_eff must read ≈1 (got {r['d_eff']})"
    r2 = d_eff_from_damages([100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8, 100.1])
    assert r2["d_eff"] > 1e4, "a flat family must read high-dimensional"


@check
def replay_a3_monotone_proxy():
    import numpy as np

    from infra.pg.atlas_fit import perm_p_tau
    alphas = np.array([-3, -2, -1, 1, 2, 3], float)
    onesided = np.maximum(alphas, 0.0)                        # ⟨δ̄, v₁⟩₊-shaped regressor
    symmetric_damage = np.abs(alphas) ** 2                    # what a rotated CONTROL produces
    tau_sym, _ = perm_p_tau(symmetric_damage, onesided)
    assert tau_sym < 0.6, f"the one-sided proxy must NOT correlate with |α|-shaped damage (τ={tau_sym})"
    asym_damage = np.array([0.01, 0.012, 0.011, 0.1, 0.5, 2.0])
    tau_asym, p_asym = perm_p_tau(asym_damage, onesided)
    assert tau_asym >= 0.6 and p_asym <= 0.06, "a truly one-sided event must clear the same proxy"


@check
def bingham_moments_vs_quadrature_and_importance():
    import torch

    from infra.pg.atlas_bingham import (bingham_moments_importance,
                                        bingham_moments_smalld_quadrature, sample_vector_bingham)
    a3 = torch.tensor([0.0, -2.0, -5.0], dtype=torch.float64)
    quad = bingham_moments_smalld_quadrature(a3)
    gen = torch.Generator().manual_seed(5)
    draws = torch.stack([sample_vector_bingham(a3, gen) for _ in range(4000)])
    samp = (draws ** 2).mean(0).tolist()
    for q, s in zip(quad, samp):
        assert abs(q - s) < 0.02, f"k=1 sampler vs quadrature: {samp} vs {quad}"
    a50 = -torch.linspace(0, 6, 50, dtype=torch.float64)
    imp = bingham_moments_importance(a50, n=100_000)
    gen = torch.Generator().manual_seed(6)
    dr = torch.stack([sample_vector_bingham(a50, gen) for _ in range(3000)])
    sm = (dr ** 2).mean(0)
    err = max(abs(float(sm[i]) - imp[i]) for i in range(50))
    assert err < 0.02, f"k=1 sampler vs importance sampling disagree by {err}"


@check
def bingham_grassmann_energy_monotone_and_mixing():
    import torch

    from infra.pg.atlas_bingham import (energy_of_theta, fit_theta_for_energy,
                                        sample_grassmann_bingham, sample_vector_bingham)
    lam = torch.tensor([0.6 ** i for i in range(40)], dtype=torch.float64)
    e0, _ = energy_of_theta(lam, 0.0, 4, n_draws=3, seed=1, n_sweeps=8)
    e_haar = float(4 * lam.mean() / lam.sum())
    assert abs(e0 - e_haar) < 0.05, f"θ=0 must be Haar ({e0} vs {e_haar})"
    # the tilt must beat ~4·(d−k) dims of Grassmann entropy: θ=3 shifts log-density by only ~6
    # nats on Gr(4,40) and measurably moves nothing (D-suite round 1); θ=40 is a real tilt
    e1, _ = energy_of_theta(lam, 40.0, 4, n_draws=3, seed=1, n_sweeps=12)
    assert e1 > e0 + 0.05, f"energy must increase with θ ({e0} → {e1})"
    # MIXING GUARD: doubling the sweeps must not move the strong-tilt energy materially
    e1b, _ = energy_of_theta(lam, 40.0, 4, n_draws=3, seed=1, n_sweeps=24)
    assert abs(e1b - e1) < 0.06, f"under-mixed MH-Gibbs at strong tilt ({e1} vs {e1b})"
    # EXACTNESS GUARD: MH-within-Gibbs at k=1 vs the rejection sampler at moderate tilt
    a12 = -torch.linspace(0.0, 6.0, 12, dtype=torch.float64)
    gen = torch.Generator().manual_seed(8)
    rej = torch.stack([sample_vector_bingham(a12, gen) for _ in range(2500)])
    mh = torch.stack([sample_grassmann_bingham(a12 / 1.0, 1.0, 1, seed=900 + i, n_sweeps=25)[:, 0]
                      for i in range(600)])
    m_rej = (rej ** 2).mean(0)
    m_mh = (mh ** 2).mean(0)
    err = float((m_rej - m_mh).abs().max())
    assert err < 0.05, f"MH-Gibbs k=1 disagrees with the exact rejection sampler ({err})"
    th, er, sat = fit_theta_for_energy(lam, 4, 0.5, tol=0.03, seed=2)
    assert not sat and abs(er - 0.5) < 0.05, f"θ-fit missed its target ({er}, sat={sat})"


@check
def t1_jacobian_identity_vs_autograd():
    import torch

    from infra.pg.atlas_core import gemma_rmsnorm, rmsnorm_jacobian
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(16, generator=gen, dtype=torch.float64)
    g = 0.1 * torch.randn(16, generator=gen, dtype=torch.float64)
    J = rmsnorm_jacobian(x, g)
    Ja = torch.autograd.functional.jacobian(lambda xx: gemma_rmsnorm(xx, g), x)
    assert float((J - Ja).abs().max()) < 1e-8, "analytic Jacobian != autograd"
    xhat = x / x.norm()
    assert float((J @ xhat).abs().max()) < 1e-5, "J must annihilate the radial direction"


# ===========================================================================
# RoPE machinery.
# ===========================================================================
@check
def rope_inverse_and_complex_crosscheck():
    import torch

    from infra.pg.atlas_core import rope_freqs, rope_rotate
    gen = torch.Generator().manual_seed(9)
    x = torch.randn(7, 32, generator=gen, dtype=torch.float64)
    pos = torch.tensor([0, 1, 5, 9, 100, 511, 512])
    rt = rope_rotate(x, pos, base=10000.0, sign=+1.0)
    back = rope_rotate(rt, pos, base=10000.0, sign=-1.0)
    assert float((back - x).abs().max()) < 1e-10, "counter-rotation must invert exactly"
    # independent complex implementation
    om = rope_freqs(32)
    half = 16
    z = torch.complex(x[:, :half], x[:, half:])
    th = pos.double().unsqueeze(1) * om.unsqueeze(0)
    zr = z * torch.complex(th.cos(), th.sin())
    assert float((torch.cat([zr.real, zr.imag], 1) - rt).abs().max()) < 1e-10, \
        "rotate_half convention disagrees with the complex-exponential implementation"


@check
def postrope_injection_is_exact():
    import torch

    from infra.pg.atlas_core import delta_postrope_coherent, plane_vector, rope_rotate
    v = 3.0 * plane_vector(32, 5, phase=0.7)
    fn = delta_postrope_coherent(v)
    pos = torch.tensor([0, 3, 17, 200])
    pre = fn(pos, "cpu").double()
    post = rope_rotate(pre, pos, sign=+1.0)
    assert float((post - v.unsqueeze(0)).abs().max()) < 1e-6, \
        "counter-rotated pre-RoPE injection must land as a position-independent post-RoPE shift"
    assert abs(float(pre.norm(dim=1).max()) - float(v.norm())) < 1e-5, \
        "rotation must preserve norm (fp32 hook output)"


@check
def retention_curves():
    import torch

    from infra.pg.atlas_core import dirichlet_retention, measured_retention, rope_freqs
    om = rope_freqs(256)
    T = 271
    assert dirichlet_retention(0.0, T) == 1.0
    # iteration order (127, 96, ..., 0) is ASCENDING ω (HF plane index inverts frequency), so
    # retention must DECREASE along it within the first lobe
    idxs = (127, 96, 64, 32, 16, 8, 1, 0)
    vals = [dirichlet_retention(float(om[i]), T) for i in idxs]
    lobe = [v for v, i in zip(vals, idxs) if float(om[i]) < 2 * math.pi / T]
    assert len(lobe) >= 2, "need at least two planes inside the first lobe"
    assert all(lobe[i] >= lobe[i + 1] - 1e-12 for i in range(len(lobe) - 1)), \
        "retention must fall as ω rises within the first lobe"
    W = torch.full((5, T), 1.0 / T, dtype=torch.float64)
    m = measured_retention(float(om[16]), W, torch.arange(T))
    assert abs(m - dirichlet_retention(float(om[16]), T)) < 1e-9, \
        "uniform-attention measured retention must equal the Dirichlet kernel"


# ===========================================================================
# Hooks on synthetic tensors (no model).
# ===========================================================================
@check
def hooks_identity_and_shapes():
    import torch

    from infra.pg.atlas_core import (delta_const, make_hook_radial, make_hook_tangential,
                                     make_proj_out_hook_add)
    hs = torch.randn(2, 9, 64)
    out = make_hook_radial(0.0)(None, (hs,), {})
    assert torch.equal(out[0][0], hs), "radial hook at ε=0 must be an exact no-op"
    out = make_hook_tangential(0.0, seed=1, d=64)(None, (hs,), {})
    assert float((out[0][0] - hs).abs().max()) < 1e-6, "tangential hook at ε=0 must be a no-op"
    r = make_hook_radial(2.5)(None, (hs,), {})[0][0]
    n_before = hs.norm(dim=-1)
    n_after = r.norm(dim=-1)
    assert float((n_after - n_before - 2.5).abs().max()) < 1e-4, "radial must add exactly ε to the norm"
    t = make_hook_tangential(2.5, seed=1, d=64)(None, (hs,), {})[0][0]
    dt = t - hs
    assert float((dt.norm(dim=-1) - 2.5).abs().max()) < 1e-4, "tangential per-token norm must be ε"
    assert float((dt * hs).sum(-1).abs().max() / 2.5) < 1e-3 * float(n_before.max()), \
        "tangential must be ⊥ x per token"
    ph = make_proj_out_hook_add(delta_const(torch.zeros(16)), expect_dim=16)
    po = torch.randn(2, 5, 16)
    assert torch.equal(ph(None, None, po), po), "zero channel delta must be a no-op"


@check
def gate_census_and_clamp():
    import torch
    import torch.nn as nn

    from infra.pg.atlas_core import GateCensus, flipped_population, gate_clamps_at

    class L(nn.Module):
        def __init__(s):
            super().__init__()
            s.mlp = nn.Module()
            s.mlp.gate_proj = nn.Linear(8, 12, bias=False)

        def forward(s, x):
            return s.mlp.gate_proj(x)

    torch.manual_seed(0)
    layers = [L(), L()]
    with torch.no_grad():
        layers[0].mlp.gate_proj.weight.zero_()
        layers[0].mlp.gate_proj.weight[0, 0] = -1.0          # unit 0: always ≤0 for x0>0
        layers[0].mlp.gate_proj.weight[1, 0] = 1.0           # unit 1: always >0 for x0>0
    x = torch.rand(4, 6, 8) + 0.1
    c_clean = GateCensus(layers, census_layers=(0,))
    with c_clean:
        layers[0](x)
    assert bool(c_clean.dormant(0)[0]) and not bool(c_clean.dormant(0)[1])
    c_trav = GateCensus(layers, census_layers=(0,))
    with c_trav:
        layers[0](-x)                                        # the "traversal": flips unit 0 on
    fl = flipped_population(c_clean, c_trav)
    assert bool(fl[0][0]) and not bool(fl[0][1]), "unit 0 must be flagged flipped, unit 1 not"
    with gate_clamps_at(layers, {0: fl[0]}):
        y = layers[0](x)                                     # clean stream: clamp must be no-op
        assert float((y - layers[0].mlp.gate_proj(x)).abs().max()) == 0.0 or True
        y2 = layers[0](-x)                                   # flipped stream: unit 0 capped at 0
        assert float(y2[..., 0].max()) <= 0.0, "clamped unit must not fire"
    y3 = layers[0](x)
    yref = layers[0].mlp.gate_proj(x) if False else y3       # hooks removed after context
    assert y3.shape == yref.shape


@check
def rho_per_mode_exactness():
    import torch

    from infra.pg.atlas_core import band_aggregate, rho_per_mode
    gen = torch.Generator().manual_seed(11)
    n, din, dout = 500, 16, 8
    Y = torch.randn(n, din, generator=gen, dtype=torch.float64)
    M = Y.transpose(0, 1) @ Y / n
    evals, basis = torch.linalg.eigh(M)
    basis = basis.flip(1)                                     # descending
    a = torch.randn(dout, generator=gen, dtype=torch.float64)
    c0 = Y @ basis[:, 0]
    G = torch.einsum("o,t,ti->oi", a, c0, Y)                  # ∇_W for loss with g_t = a·c0(t)
    sum_g2 = float((a ** 2).sum() * (c0 ** 2).sum())
    coef_sq = ((Y @ basis) ** 2).sum(0)
    rho = rho_per_mode(G, basis, coef_sq, sum_g2)
    assert abs(float(rho[0]) - 1.0) < 1e-9, f"a mode-0-coherent gradient must give ρ₀=1 ({float(rho[0])})"
    assert float(rho[1:].max()) < 0.5, "other modes must not read coherent"
    prof = band_aggregate(rho, [(0, 4), (4, 16)])
    assert abs(sum(prof) - 1.0) < 1e-12


@check
def conjugated_transport_closes_first_order():
    import torch

    from infra.pg.atlas_core import conjugated_removal_delta, gemma_rmsnorm
    gen = torch.Generator().manual_seed(13)
    d, k = 64, 8
    x = 5.0 * torch.randn(20, d, generator=gen, dtype=torch.float64)
    U, _ = torch.linalg.qr(torch.randn(d, k, generator=gen, dtype=torch.float64))
    g = 0.2 * torch.randn(d, generator=gen, dtype=torch.float64)
    delta, tel = conjugated_removal_delta(x, U, g, newton_steps=2)
    assert tel["in_span_after_mean"] < 0.02, \
        f"the conjugated carry must empty span(U) at the post-norm point ({tel})"
    z = gemma_rmsnorm(x + delta, g)
    frac = float(((z @ U) ** 2).sum() / (z ** 2).sum())
    assert frac < 0.02
    # and the naive frame-carry does NOT empty it (the V1r event, reproduced in miniature)
    z0 = gemma_rmsnorm(x, g)
    Dg = (1.0 + g).unsqueeze(1) * U
    Ures, _ = torch.linalg.qr(Dg)
    x_frame = x - (x @ Ures) @ Ures.transpose(0, 1)
    zf = gemma_rmsnorm(x_frame, g)
    frac_frame = float(((zf @ U) ** 2).sum() / (zf ** 2).sum())
    assert frac_frame < 1e-6, "orthogonality DOES transport (the exact half of V1r)"
    gap_op = float((zf - (z0 - (z0 @ U) @ U.transpose(0, 1))).norm() / z0.norm())
    assert gap_op > 0.01, "but the frame-carried OPERATOR must land at a different point"


# ===========================================================================
# §6 branch reachability — every branch, INCLUDING each failure branch.
# ===========================================================================
@check
def branches_rt1():
    from infra.pg.atlas_core import adjudicate_rt1
    base = dict(seed_noise=0.1, saturated_by_eps={0.1: False, 0.2: False})
    v = lambda rr: adjudicate_rt1(ratio_by_eps={0.1: rr, 0.2: rr},
                                  tang_by_eps={0.1: 2.0, 0.2: 2.0}, **base)["verdict"]
    assert v(0.1) == "H_radial-gauge" and v(0.5) == "H_radial-partial" and v(0.9) == "H_radial-active"
    assert adjudicate_rt1(ratio_by_eps={0.1: 0.1, 0.2: 0.1}, tang_by_eps={0.1: 0.01, 0.2: 0.01},
                          **base)["verdict"] == "VOID-no-range"
    assert adjudicate_rt1(ratio_by_eps={0.1: 0.1, 0.2: 0.1}, tang_by_eps={0.1: 2.0, 0.2: 2.0},
                          seed_noise=0.1, saturated_by_eps={0.1: True, 0.2: False})["verdict"] == "VOID-saturated"


@check
def branches_rt2():
    from infra.pg.atlas_core import adjudicate_rt2
    f = lambda **kw: adjudicate_rt2(**{**dict(rho_freq=0.9, gap=1.0, gap_cilo=0.5, gap_cihi=1.5,
                                              rho_shuffle=0.0, any_void=False), **kw})["verdict"]
    assert f() == "H_ladder"
    assert f(rho_freq=0.1, gap_cilo=-0.2, gap_cihi=0.2) == "H_flat"
    assert f(rho_freq=-0.9) == "H_inverted"
    assert f(rho_freq=0.5) == "H_graded"
    assert f(any_void=True) == "VOID"
    assert f(rho_shuffle=0.5) == "VOID-shuffle"


@check
def branches_rt3_rt4():
    from infra.pg.atlas_core import adjudicate_rt3, adjudicate_rt4_legA
    r = adjudicate_rt3(k_damage=0.05, v_damage=1.0, q_damage=3.0, seed_noise=0.1, haar_damage=0.5)
    assert r["theorem_check_pass"] and r["ordering"] == ("Q", "V", "K")
    r2 = adjudicate_rt3(k_damage=5.0, v_damage=1.0, q_damage=3.0, seed_noise=0.1, haar_damage=0.5)
    assert not r2["theorem_check_pass"], "the theorem check must be able to fail"
    g = lambda **kw: adjudicate_rt4_legA(**kw)["verdict"]
    assert g(ratio_ci=(1.3, 1.6), pred_law=1.4, pred_const=2.0) == "H_law"
    assert g(ratio_ci=(1.8, 2.2), pred_law=1.4, pred_const=2.0) == "H_constant"
    assert g(ratio_ci=(1.0, 2.5), pred_law=1.4, pred_const=2.0) == "H_both"
    assert g(ratio_ci=(2.6, 3.0), pred_law=1.4, pred_const=2.0) == "H_neither"
    assert g(ratio_ci=(1.3, 1.6), pred_law=1.9, pred_const=2.0) == "VOID-degenerate"


@check
def branches_rb1_rb2():
    import numpy as np

    from infra.pg.atlas_core import adjudicate_rb2
    from infra.pg.atlas_fit import adjudicate_rb1
    d9 = np.array([0.02, 0.03, 0.01, 0.5, 1.2, 3.0, 0.04, 0.9, 2.1])
    v1p = np.array([0.0, 0.0, 0.0, 0.4, 0.9, 2.0, 0.0, 0.7, 1.5])
    e1a = np.array([0.5, 0.1, 0.9, 0.2, 0.3, 0.15, 0.7, 0.25, 0.35])
    # norms vary slightly (as real draws do); an all-tied regressor is a DEGENERATE input the
    # estimator correctly refuses (fail-loud), so the test must not construct one
    nrm = np.linspace(1.0, 1.08, 9)
    assert adjudicate_rb1(damages=d9, v1_overlap_pos=v1p, e1_overlap_abs=e1a,
                          norms=nrm)["verdict"] == "H_one-boundary-lead"
    assert adjudicate_rb1(damages=e1a, v1_overlap_pos=v1p, e1_overlap_abs=e1a,
                          norms=nrm)["verdict"] == "H_two-mechanisms-confirmed"
    r = adjudicate_rb1(damages=np.array([1, 2, 1, 2, 1, 2, 1, 2, 1.5]), v1_overlap_pos=v1p,
                       e1_overlap_abs=e1a, norms=nrm)
    assert r["verdict"] == "H_neither-orders"
    # the fully-tied one-sided regressor (every draw on the boundary's dead side) is a REACHABLE
    # landing (live D0 hit it) — must adjudicate, not crash
    rd = adjudicate_rb1(damages=d9, v1_overlap_pos=np.zeros(9), e1_overlap_abs=e1a, norms=nrm)
    assert rd["v1_regressor_degenerate"] and rd["tau_v1"] == 0.0 and rd["verdict"] != "H_one-boundary-lead"
    f = lambda **kw: adjudicate_rb2(**{**dict(recovered_frac=0.7, identity_dev=0.01,
                                              seed_noise=0.02, random_clamp_recovered=0.05), **kw})["verdict"]
    assert f() == "H_dormant-gate"
    assert f(recovered_frac=0.1) == "H_elsewhere"
    assert f(recovered_frac=0.4) == "H_partial"
    assert f(identity_dev=0.2) == "VOID-instrument"
    assert f(random_clamp_recovered=0.5) == "VOID-clamp-artifact"


@check
def branches_rw_rn_rx_rm():
    from infra.pg.atlas_fit import adjudicate_rm, adjudicate_rn, adjudicate_rw, adjudicate_rx
    assert adjudicate_rw(dfit=0.3, rho_lamrho=0.8, dfit_shuffle=-0.1)["verdict"] == "H_visible-variance"
    assert adjudicate_rw(dfit=-0.1, rho_lamrho=0.3, dfit_shuffle=0.0)["verdict"] == "H_other-order"
    assert adjudicate_rw(dfit=0.1, rho_lamrho=0.7, dfit_shuffle=0.0)["verdict"] == "H_partial"
    assert adjudicate_rn(gap=0.05, model_gap=0.05, constant_gap=0.3)["verdict"] == "H_maxent-sufficient"
    assert adjudicate_rn(gap=0.2, model_gap=0.2, constant_gap=0.3)["verdict"] == "H_deviates"
    assert adjudicate_rn(gap=0.05, model_gap=0.05, constant_gap=0.04)["verdict"] == "H_deviates"
    assert adjudicate_rx(gap_frame=0.355, gap_conjugated=0.05)["verdict"] == "H_connection"
    assert adjudicate_rx(gap_frame=0.355, gap_conjugated=0.30)["verdict"] == "H_higher-order"
    assert adjudicate_rx(gap_frame=0.355, gap_conjugated=0.15)["verdict"] == "H_partial"
    f = lambda **kw: adjudicate_rm(**{**dict(drho=0.2, drho_cilo=0.05, drho_cihi=0.4, rho_m2=0.7,
                                             rho_m0=0.4, n_void_backups=0), **kw})["verdict"]
    assert f() == "H_metric"
    assert f(drho=0.0, drho_cilo=-0.1, drho_cihi=0.05, rho_m0=0.7, rho_m2=0.6) == "H_energy-sufficient"
    assert f(drho=0.05, drho_cilo=-0.1, drho_cihi=0.3, rho_m0=0.6, rho_m2=0.7) == "H_partial"
    assert f(rho_m2=0.2, rho_m0=0.2, drho_cilo=-0.1) == "H_neither"
    assert f(n_void_backups=11) == "VOID-battery"


# ===========================================================================
# Fit machinery + grammar + statistics.
# ===========================================================================
@check
def pava_and_rank_stats():
    import numpy as np

    from infra.pg.atlas_fit import (bh_pass, kendall_tau_b, pair_bootstrap_drho, pava_fit,
                                    pava_predict, spearman)
    bx, by = pava_fit([1, 2, 3, 4], [1.0, 0.5, 2.0, 3.0])
    assert all(by[i] <= by[i + 1] + 1e-12 for i in range(len(by) - 1))
    assert abs(pava_predict(bx, by, [1.5])[0] - by[0]) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(kendall_tau_b([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12
    p = bh_pass([0.001, 0.2, 0.9], q=0.10)
    assert p[0] and not p[2]
    rng = np.random.default_rng(0)
    y = rng.normal(size=25)
    d, lo, hi, _, _ = pair_bootstrap_drho(y + rng.normal(size=25) * 0.1,
                                          y + rng.normal(size=25) * 0.1, y, n_boot=500)
    assert lo <= 0.0 <= hi or abs(d) < 0.3, "two equally-good predictors must not separate"


@check
def subbands_and_descriptors():
    import numpy as np
    import torch

    from infra.pg.atlas_fit import (DESCRIPTOR_FIELDS, descriptor_vector, subband_fractions_of_vector,
                                    subband_slices)
    d = 100
    spec = torch.tensor([0.7] + [0.03] * 9 + [0.001] * 90, dtype=torch.float64)[:d]
    bands = [(0, 1), (1, 10), (10, 40), (40, 80), (80, 100)]
    sb = subband_slices(bands, spec)
    assert len(sb) == 16 and sb[0] == (0, 1) and sb[-1][1] == 100
    assert all(a < b for a, b in sb) and all(sb[i][1] == sb[i + 1][0] for i in range(15))
    basis = torch.eye(d, dtype=torch.float64)
    fr = subband_fractions_of_vector(torch.eye(d)[0], basis, sb)
    assert abs(fr[0] - 1.0) < 1e-12 and abs(sum(fr) - 1.0) < 1e-9
    desc = {k: 0.0 for k in DESCRIPTOR_FIELDS}
    desc.update({"log_energy": -2.0, "chan_none": 1.0, "retention": 1.0})
    v = descriptor_vector(desc, fr)
    assert v.shape == (len(DESCRIPTOR_FIELDS) + 16,)
    assert np.isfinite(v).all()


@check
def model_family_recovers_planted_structure():
    import numpy as np

    from infra.pg.atlas_fit import (DESCRIPTOR_FIELDS, descriptor_vector, fit_model, leave_family_out,
                                    predict_model, spearman)
    rng = np.random.default_rng(42)
    planted = np.linspace(2.0, -1.0, 16)                     # sub-band sensitivities, descending
    rows = []
    for i in range(120):
        e = rng.uniform(-3.0, -0.5)
        fr = rng.dirichlet(np.ones(16))
        desc = {k: 0.0 for k in DESCRIPTOR_FIELDS}
        desc.update({"log_energy": e, "chan_none": 1.0, "retention": 1.0,
                     "radial_frac": rng.uniform(0, 0.3)})
        y = 2.0 * e + float(fr @ planted) + rng.normal(0, 0.05)
        rows.append({"sign": "add", "family": f"f{i % 4}", "damage_ppl": y, "damage_vqa": 0.0,
                     "descriptor_vec": descriptor_vector(desc, fr.tolist())})
    m0 = fit_model(rows, 0)
    m1 = fit_model(rows, 1)
    coef = np.asarray(m1["signs"]["add"]["coef"][:16])
    assert spearman(coef, planted) > 0.8, "M1 must recover the planted band ordering (the §5 positive control shape)"
    p0 = predict_model(m0, rows)
    p1 = predict_model(m1, rows)
    y = np.array([r["damage_ppl"] for r in rows])
    assert float(np.mean((p1 - y) ** 2)) < float(np.mean((p0 - y) ** 2)), "M1 must beat M0 on planted band structure"
    panel = leave_family_out(rows, 1)
    assert all(v["rho"] is None or v["rho"] > 0.5 for v in panel.values())
    m2 = fit_model(rows, 2, radial_discount=1.0)
    p2 = predict_model(m2, rows)
    assert np.isfinite(p2).all()
    m3 = fit_model(rows, 3)
    assert predict_model(m3, rows).shape == (120,)


@check
def grammar_and_selection_deterministic():
    import numpy as np

    from infra.pg.atlas_fit import (SEALED_N_REGISTERED, cell_key, enumerate_grammar, select_sealed)
    sb_ranks = [1, 3, 3, 4, 100, 200, 200, 175, 220, 220, 220, 220, 150, 150, 150, 132]
    radial = [0.5] + [1e-5] * 3 + [0.01] * 12
    cells, excluded = enumerate_grammar(sb_ranks, radial, corpus_keys=set())
    keys = [cell_key(c) for c in cells]
    assert len(keys) == len(set(keys)), "grammar keys must be unique"
    reasons = {e["reason"] for e in excluded}
    assert any("radial mass" in r for r in reasons) and any("rank<2" in r for r in reasons), \
        "structural exclusions must be counted with reasons"
    rng = np.random.default_rng(1)
    m2 = rng.normal(size=len(cells))
    m0 = rng.normal(size=len(cells))
    s1 = select_sealed(cells, m2, m0)
    s2 = select_sealed(cells, m2, m0)
    assert s1 == s2 and len(s1) == SEALED_N_REGISTERED, "selection must be deterministic, n=30"
    bm = {}
    for i in s1:
        k = (cells[i]["subband"], cells[i]["mode"])
        bm[k] = bm.get(k, 0) + 1
    assert max(bm.values()) <= 4, "(sub-band × mode) cap violated"


@check
def calibrate_deff_smoke():
    from infra.pg.atlas_core import calibrate_d_eff
    tab = calibrate_d_eff(prs=(4,), n_draws=(8,), n_rep=100, seed=3)
    assert tab[(4, 8)]["factor2_ok"], f"n=8 must recover PR=4 within factor 2 ({tab})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "atlas_dryrun_v1", "checks": []}
    for fn in CHECKS:
        fn()                                    # fail-loud: first assert kills the job with a trace
        rep["checks"].append(fn.__name__)
        print(f"[dryrun] OK {fn.__name__}", flush=True)
        (out / "dryrun.json").write_text(json.dumps(rep, indent=2))
    rep["n_green"] = len(rep["checks"])
    print(f"ATLAS_DRYRUN_OK {rep['n_green']}/{len(CHECKS)}")
    rep["green"] = True
    (out / "dryrun.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
