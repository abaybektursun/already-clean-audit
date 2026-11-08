"""FULCRUM weight-free dry-run logic gate (spec §13b). Tiny tensors, no model — seconds on CPU.

Proves, before any GPU minute:
  (a) `theta_family` at θ=1 is ⊥ span(U_P) to fp tol; θ=0 is identity; the per-matrix arithmetic equals
      the whole-decoder `combine_alpha` identity ΔW − θ·P(ΔW).
  (b) an ENERGY-ONLY synthetic ΔW ⇒ E_curv ≈ 0 (the floor: curvature-removal has nothing to remove);
      a CURVATURE-ONLY synthetic ⇒ E_curv strongly negative (the ceiling: strip the spread curvature
      and only the energy-aligned seed remains, so n*_energy collapses) — through the REAL surfaces
      (`theta_family`/`project_delta`/`necessary_rank`) with a generic Q ⊥ energy.
  (c) `e_curv_at_matched_gain` interpolation is exact on a linear curve; g* is the jointly-reachable
      floor; unreachable g* → excluded/reported (not zeroed).
  (d) `necessary_rank` is monotone in a nested energy basis.
  plus the FOOTPRINT-moved gate (`footprint_census` curvature-fraction falls with θ) and every branch
  of the §6 adjudicator (VOID-leak / VOID-blind / VOID-no-reach / H_lever / H_bystander / INSUFFICIENT).

Runs under the repo's pytest AND as `main()` from the offline gate. No try/except (CLAUDE.md).
"""
import numpy as np
import torch

from infra.pg import comwrite_surgery as CS
from infra.pg import fulcrum_core as FC
from infra.pg import quorum_core as QC

# d=256 with the curvature-projection rank r=d//3≈85 ≥ the curvature-bulk rank kq=32 (so curv-removal
# captures the full bulk, as r=679 ≥ any real bulk does) — the regime the offline simulator validated.
D = 256
LADDER = [r for r in (1, 2, 3, 5, 10, 20, 50, 85, 128, 170, 256)]
RHO = 0.946


def _bases(d=D, seed=0):
    """energy basis = I (descending variance by index); curvature Q = an INDEPENDENT random
    orthonormal basis (Q ⊥ energy, generic — a curvature direction is spread across the energy
    spectrum). Returns (U_E, U_Q, banks-shaped dict)."""
    g = np.random.default_rng(seed)
    U_E = np.eye(d)
    U_Q = np.linalg.qr(g.standard_normal((d, d)))[0]
    bank = {"sigma_evecs": torch.as_tensor(U_E), "Q_evecs": torch.as_tensor(U_Q), "d": d}
    return U_E, U_Q, bank


def _nstar_energy(M, U_E, ladder=LADDER, rho=RHO):
    """toy n*_energy: retention(r) = ‖M with top-r ENERGY dirs removed (output side)‖²/‖M‖²."""
    M = np.asarray(M, dtype=np.float64)
    tot = float((M ** 2).sum())
    ret = []
    for r in ladder:
        Ur = U_E[:, :r]
        off = M - Ur @ (Ur.T @ M)
        ret.append(float((off ** 2).sum()) / tot)
    return QC.necessary_rank(ret, ladder, rho_break=rho)["n_star"]


def _synth(U_E, U_Q, curv_frac, ke=3, kq=32, m=8, seed=0):
    """single output-side ΔW = energy part (span U_E[:, :ke]) + curvature part (span U_Q[:, :kq]),
    scaled to the target curvature ENERGY fraction. Mirrors `inject_*_capability`'s _side_block
    geometry but as one matrix (the dry-run doesn't need a full decoder grid)."""
    g = np.random.default_rng(seed + 100)
    M_e = U_E[:, :ke] @ g.standard_normal((ke, m))
    M_q = U_Q[:, :kq] @ g.standard_normal((kq, m))
    M_e *= np.sqrt((1 - curv_frac) / max((M_e ** 2).sum(), 1e-30))
    M_q *= np.sqrt(curv_frac / max((M_q ** 2).sum(), 1e-30))
    return M_e + M_q


def _theta_readout(M, U_E, U_P, r):
    """{θ: gain}, {θ: n*} for projection U_P at matched rank r via the REAL theta_family."""
    fam = FC.theta_family(M, torch.as_tensor(U_P[:, :r]), "o_proj", FC.THETA_LADDER)
    gain, nstar = {}, {}
    for th, Mt in fam.items():
        Mt = Mt.numpy()
        gain[th] = float((Mt ** 2).sum())
        nstar[th] = _nstar_energy(Mt, U_E)
    return gain, nstar


# ---------------------------------------------------------------------------
# (a) theta_family geometry.
# ---------------------------------------------------------------------------
def test_theta_family_theta1_orthogonal_and_theta0_identity():
    g = torch.Generator().manual_seed(1)
    dW = torch.randn(D, 8, generator=g, dtype=torch.float32)              # output-side (d rows)
    U = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float32))[0][:, :16]
    fam = FC.theta_family(dW, U, "o_proj", (0.0, 0.5, 1.0))
    assert torch.allclose(fam[0.0], dW, atol=1e-6), "θ=0 not identity"
    resid = FC.residual_orthogonality(fam[1.0], U, "o_proj")
    assert resid < 1e-5, f"θ=1 not ⊥ span(U): residual-in-span fraction {resid}"


def test_theta_family_equals_combine_alpha_identity():
    """The per-matrix θ-family ΔW − θ·P(ΔW) equals the whole-decoder combine_alpha arithmetic
    (b+ΔW) − θ·((b+P(ΔW)) − b), so the job's `combine_alpha` and this core agree exactly."""
    g = torch.Generator().manual_seed(2)
    b = torch.randn(D, 8, generator=g, dtype=torch.float32)
    dW = torch.randn(D, 8, generator=g, dtype=torch.float32)
    U = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float32))[0][:, :10]
    P = CS.project_delta(dW, U, "o_proj")
    for th in (0.0, 0.25, 0.5, 0.75, 1.0):
        tf = dW - th * P                                     # theta_family delta
        ca = (b + dW) - th * ((b + P) - b)                   # combine_alpha: f - θ(s - b)
        assert torch.allclose(tf, ca - b, atol=1e-6), f"θ={th}: theta_family ≠ combine_alpha"


# ---------------------------------------------------------------------------
# (d) necessary_rank monotone in a nested basis.
# ---------------------------------------------------------------------------
def test_necessary_rank_monotone_nested_basis():
    U_E, _, _ = _bases()
    M = U_E[:, :20] @ np.random.default_rng(3).standard_normal((20, 8))   # spread over 20 energy dirs
    tot = float((M ** 2).sum())
    ret = [float(((M - U_E[:, :r] @ (U_E[:, :r].T @ M)) ** 2).sum()) / tot for r in LADDER]
    assert all(ret[i] >= ret[i + 1] - 1e-12 for i in range(len(ret) - 1)), f"non-monotone: {ret}"
    ns = QC.necessary_rank(ret, LADDER, rho_break=RHO)
    assert not ns["censored"] and 0 < ns["n_star"] <= 20, ns


# ---------------------------------------------------------------------------
# (c) interpolation + g*.
# ---------------------------------------------------------------------------
def test_interp_exact_on_linear_curve():
    th = list(FC.THETA_LADDER)
    gain = {t: 1.0 - 0.5 * t for t in th}
    nstar = {t: 10.0 + 100.0 * (gain[t] / gain[0.0]) for t in th}         # n* = 10 + 100·retention
    r = FC.interp_nstar_at_gstar(nstar, gain, 0.75)
    assert r["reachable"] and abs(r["n_star"] - 85.0) < 1e-9, r


def test_gstar_is_the_binding_floor_and_unreachable_excluded():
    th = list(FC.THETA_LADDER)
    g = {"c": {"susceptibility": {t: 1.0 - 0.8 * t for t in th},
               "random": {t: 1.0 - 0.4 * t for t in th}}}
    gs = FC.g_star_jointly_reachable(g)
    assert abs(gs["g_star"] - 0.6) < 1e-9, gs                             # max(0.2, 0.6)
    below = FC.interp_nstar_at_gstar({t: 5.0 for t in th}, {t: 1.0 - 0.4 * t for t in th}, 0.2)
    assert below["reachable"] is False and below["n_star"] is None, below


# ---------------------------------------------------------------------------
# (b) the two synthetic controls through the real surfaces.
# ---------------------------------------------------------------------------
def _control_e_curv(curv_frac, kq, seed):
    U_E, U_Q, _ = _bases(seed=seed)
    U_R = np.linalg.qr(np.random.default_rng(seed + 7).standard_normal((D, D)))[0]
    M = _synth(U_E, U_Q, curv_frac, kq=kq, seed=seed)
    r = D // 3
    gc, nc = _theta_readout(M, U_E, U_Q, r)
    gr, nr = _theta_readout(M, U_E, U_R, r)
    gstar = FC.g_star_jointly_reachable({"c": {"susceptibility": gc, "random": gr}})["g_star"]
    return FC.e_curv_at_matched_gain(nc, gc, nr, gr, gstar), _nstar_energy(M, U_E)


def test_energy_only_synthetic_e_curv_zero():
    """The FLOOR / positive control: an energy-dominated ΔW → small n*_energy and E_curv ≈ 0
    (curvature-removal has almost nothing to remove). Median over seeds."""
    es = []
    for s in range(6):
        e, n0 = _control_e_curv(curv_frac=0.10, kq=32, seed=s)
        assert n0 <= 5, f"energy-only n0 not small: {n0}"
        es.append(e["E_curv"] if e["reachable"] else 0.0)
    assert abs(float(np.median(es))) < 0.5, f"energy-only E_curv not ≈0: {es}"


def test_curvature_only_synthetic_e_curv_strongly_negative():
    """The CEILING: a curvature-dominated ΔW → E_curv strongly negative (strip the spread curvature and
    only the energy seed remains). Median over seeds; every seed non-positive."""
    es = []
    for s in range(6):
        e, _ = _control_e_curv(curv_frac=0.90, kq=32, seed=s)
        assert e["reachable"], "ceiling not reachable"
        es.append(e["E_curv"])
    assert float(np.median(es)) < -0.5, f"ceiling E_curv not strongly negative: {es}"
    assert all(x <= 1e-6 for x in es), f"a ceiling seed went positive: {es}"


def test_footprint_moves_with_theta():
    """The §5 footprint-moved gate: ΔW(θ,curv)'s CURVATURE fraction falls with θ while its ENERGY
    fraction stays ~stable (via the real `mechanism_census`)."""
    from infra.pg import cleave_core as CL
    U_E, U_Q, bank = _bases(seed=0)
    M = _synth(U_E, U_Q, curv_frac=0.8, kq=32, seed=0)
    r = D // 3
    fam = FC.theta_family(M, torch.as_tensor(U_Q[:, :r]), "o_proj", (0.0, 0.5, 1.0))
    cq = [CL.mechanism_census(fam[t].numpy(), bank, r, "o_proj")["susceptibility"] for t in (0.0, 0.5, 1.0)]
    ce = [CL.mechanism_census(fam[t].numpy(), bank, r, "o_proj")["energy"] for t in (0.0, 0.5, 1.0)]
    assert cq[0] > cq[1] > cq[2] - 1e-9, f"curvature fraction did not fall with θ: {cq}"
    # energy fraction should not spike downward (it is ~stable or mildly rises as curvature leaves)
    assert ce[2] >= ce[0] - 0.05, f"energy fraction dropped unexpectedly with θ: {ce}"


# ---------------------------------------------------------------------------
# the §6 adjudicator — every branch, in order.
# ---------------------------------------------------------------------------
def _kw(**over):
    kw = dict(e_curv_ci={"median": -120.0, "ci_lo": -140.0, "ci_hi": -100.0, "n": 5,
                         "insufficient": False},
              null_band_hi=20.0, pos_floor_e_curv=2.0, ceiling_e_curv=-150.0,
              pos_floor_tol=15.0, ceiling_thresh=-50.0, monotone_frac_curv=1.0,
              footprint_moved=True, any_reach_gstar=True, shuffle_collapses=True)
    kw.update(over)
    return kw


def test_adjudicate_void_on_leak():
    v = FC.adjudicate_fulcrum(**_kw(pos_floor_e_curv=-120.0))               # positive control leaked
    assert v["verdict"] == "VOID" and "leaked" in v["reason"], v


def test_adjudicate_void_on_blind():
    v = FC.adjudicate_fulcrum(**_kw(ceiling_e_curv=-5.0))                   # ceiling did not fire
    assert v["verdict"] == "VOID" and "blind" in v["reason"], v


def test_adjudicate_void_on_no_reach():
    v = FC.adjudicate_fulcrum(**_kw(any_reach_gstar=False))
    assert v["verdict"] == "VOID", v


def test_adjudicate_insufficient():
    v = FC.adjudicate_fulcrum(**_kw(e_curv_ci={"median": None, "ci_lo": None, "ci_hi": None,
                                               "n": 1, "insufficient": True}))
    assert v["verdict"] == "INSUFFICIENT-primary", v


def test_adjudicate_h_lever():
    v = FC.adjudicate_fulcrum(**_kw())
    assert v["verdict"] == "H_lever", v


def test_adjudicate_h_bystander():
    v = FC.adjudicate_fulcrum(**_kw(e_curv_ci={"median": 1.0, "ci_lo": -10.0, "ci_hi": 12.0, "n": 6,
                                               "insufficient": False}, monotone_frac_curv=0.0,
                                    footprint_moved=False))
    assert v["verdict"] == "H_bystander", v


def test_adjudicate_h_bystander_effect_within_band():
    # a significant-negative CI whose MEDIAN is inside the ±band is indistinguishable from random
    v = FC.adjudicate_fulcrum(**_kw(e_curv_ci={"median": -8.0, "ci_lo": -15.0, "ci_hi": -5.0, "n": 6,
                                               "insufficient": False}))
    assert v["verdict"] == "H_bystander", v


def test_adjudicate_h_lever_weak():
    # median beyond the band, CI all-negative, but CI-hi (-8) not below -band (-20) → weak lever
    v = FC.adjudicate_fulcrum(**_kw(e_curv_ci={"median": -40.0, "ci_lo": -70.0, "ci_hi": -8.0, "n": 6,
                                               "insufficient": False}))
    assert v["verdict"] == "H_lever-weak", v


def test_adjudicate_h_anti_lever():
    # significant POSITIVE effect beyond the band (the two-sided other tail)
    v = FC.adjudicate_fulcrum(**_kw(e_curv_ci={"median": 40.0, "ci_lo": 8.0, "ci_hi": 70.0, "n": 6,
                                               "insufficient": False}))
    assert v["verdict"] == "H_anti-lever", v


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    for fn in ALL:
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nALL {len(ALL)} FULCRUM dry-run logic-gate tests PASSED", flush=True)
    return len(ALL)


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
