"""QUORUM weight-free dry-run logic gate (spec §13). Tiny tensors, no model — seconds on CPU.

Proves, before any GPU minute:
  * `necessary_rank` returns ~1 for a rank-1 retention CLIFF and censors to d for a FLAT curve;
  * isotonic smoothing makes n* robust to a non-monotone wiggle;
  * `rho_break_from_null` reads the knee off the null's own curve;
  * the injected rank-1 ΔW is annihilated by removing the top-1 energy direction (n*=1 end-to-end
    through the pure surgery logic — the positive control);
  * a rank-sweep of a nested-basis removal is monotone-nonincreasing in retention;
  * `mechanism_census` energy-fractions + the Haar floor r/d are exact;
  * `bimodality_test` fires on two clusters and stays quiet on one;
  * every branch of the §6 adjudicator, in order.

Runs under the repo's pytest (infra/pg/tests). No try/except (CLAUDE.md).
"""
import numpy as np
import torch

from infra.pg import comwrite_surgery as CS
from infra.pg import quorum_core as QC
from infra.pg.gauge_common import PROJ

D = 32
INNER = 8


def _bank(d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    S, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return {"Q_evecs": Q, "sigma_evecs": S, "d": d}


def _banks(n_layers=2, d=D):
    return {L: _bank(d, seed=L) for L in range(n_layers)}


# ---------------------------------------------------------------------------
# 1. necessary_rank — the cliff, the flat curve (censoring), the wiggle.
# ---------------------------------------------------------------------------
def test_necessary_rank_rank1_cliff():
    """A capability stored in one direction: retention holds at 1 until rank 1 removes it, then 0.
    n* must be ~1."""
    ranks = [1, 5, 10, 50]
    ret = [0.0, 0.0, 0.0, 0.0]                 # already gone at the smallest rank tested
    r = QC.necessary_rank(ret, ranks, rho_break=0.5)
    assert r["n_star"] == 1.0 and not r["censored"], r
    # a curve that holds then cliffs between 1 and 5
    r2 = QC.necessary_rank([1.0, 0.0, 0.0, 0.0], ranks, rho_break=0.5)
    assert 1.0 <= r2["n_star"] <= 5.0 and not r2["censored"], r2


def test_necessary_rank_flat_curve_censors_to_d():
    """A capability that survives every removal on the ladder → right-censored to d, flagged, never a
    silent 'n* = max ladder'."""
    ranks = [5, 50, 679]
    ret = [1.0, 0.98, 0.97]
    r = QC.necessary_rank(ret, ranks, rho_break=0.5, d=2048)
    assert r["censored"] is True and r["n_star"] == 2048.0 and r["crossed_at_rank"] is None, r


def test_necessary_rank_isotonic_robust_to_wiggle():
    """A non-monotone sampling wiggle must not create a spurious early crossing; the isotonic fit is
    non-increasing so a single dip that recovers is smoothed."""
    ranks = [5, 10, 20, 50, 150]
    ret = [1.0, 0.45, 0.9, 0.8, 0.2]           # the 0.45 dip at r=10 recovers to 0.9 — noise
    smoothed = QC.necessary_rank(ret, ranks, rho_break=0.5, smooth=True)
    raw = QC.necessary_rank(ret, ranks, rho_break=0.5, smooth=False)
    assert raw["n_star"] < smoothed["n_star"], "smoothing should push the crossing later than the raw dip"
    fit = smoothed["retention_fit"]
    assert all(fit[i] >= fit[i + 1] - 1e-9 for i in range(len(fit) - 1)), "isotonic fit not non-increasing"


def test_rho_break_reads_the_null_knee():
    ranks = [5, 10, 20, 50, 150, 350, 679]
    # a curve that holds high then bends down — the knee is at the bend
    ret = [0.99, 0.98, 0.97, 0.9, 0.6, 0.3, 0.1]
    rb = QC.rho_break_from_null(ret, ranks)
    assert 0.1 < rb["rho_break"] < 0.99, rb
    assert 20 <= rb["knee_rank"] <= 350, rb


# ---------------------------------------------------------------------------
# 2. the injected rank-1 positive control — n*=1 end-to-end through the surgery.
# ---------------------------------------------------------------------------
def test_injected_rank1_annihilated_by_energy_top1():
    """The rank-1 ΔW along the top energy direction is fully captured (removed) by the top-1 energy
    projector — so its off-projection at r=1 is zero ⇒ retention 0 ⇒ n*=1. This is the positive
    control's analytic core, run through the real projection surface."""
    bank = _bank(seed=5)
    U = bank["sigma_evecs"]
    scale = 3.0
    for mtype in ("o_proj", "q_proj"):
        dW = QC.inject_rank1_delta(U, layer=0, mtype=mtype, scale=scale)
        # remove the top-1 energy subspace on the residual-facing side. project_delta is fp32 by
        # design, so the annihilation floor is ~eps32·‖dW‖ ≈ 1e-7·scale, not 0.
        U1 = U[:, :1].numpy() if hasattr(U, "numpy") else np.asarray(U)[:, :1]
        off = np.asarray(dW) - CS.project_delta(np.asarray(dW), U1, mtype).numpy()
        assert float(np.abs(off).max()) < 1e-4 * scale, f"{mtype}: rank-1 energy injection not removed at r=1"
        # ... and NOT removed by an off-top direction (removal must be of ENERGY-top specifically)
        U_other = U[:, 5:6].numpy() if hasattr(U, "numpy") else np.asarray(U)[:, 5:6]
        off_other = np.asarray(dW) - CS.project_delta(np.asarray(dW), U_other, mtype).numpy()
        assert float(np.abs(off_other).max()) > 0.1 * scale, f"{mtype}: a non-top direction removed it (should not)"


# ---------------------------------------------------------------------------
# 3. rank-sweep monotonicity of removed energy (nested basis).
# ---------------------------------------------------------------------------
def test_rank_sweep_removed_energy_monotone():
    """Removing the top-r subspace of a nested basis removes MORE energy as r grows (retention can
    only fall) — the property `necessary_rank`'s isotonic fit assumes."""
    g = torch.Generator().manual_seed(9)
    bank = _bank(seed=3)
    U = bank["sigma_evecs"]
    dW = torch.randn(D, INNER, generator=g, dtype=torch.float64)          # o_proj (d, inner)
    prev = -1.0
    for r in (1, 2, 4, 8, 16, 32):
        Ur = U[:, :r].numpy()
        captured = float(((Ur.T @ dW.numpy()) ** 2).sum())               # ‖Uᵣᵀ ΔW‖² on the output side
        assert captured >= prev - 1e-9, f"captured energy not monotone at r={r}"
        prev = captured


# ---------------------------------------------------------------------------
# 4. census exactness (reused, re-checked at QUORUM's ranks).
# ---------------------------------------------------------------------------
def test_census_energy_fraction_and_floor_exact():
    from infra.pg import cleave_core as CL
    bank = _bank(seed=6)
    g = torch.Generator().manual_seed(1)
    r = 4
    Ue = bank["sigma_evecs"][:, :r]
    M = (Ue @ torch.randn(r, INNER, generator=g, dtype=torch.float64)).numpy()   # entirely in U_energy
    cen = CL.mechanism_census(M, bank, rank=r, mtype="o_proj")
    assert abs(cen["energy"] - 1.0) < 1e-8, cen
    assert abs(cen["chance_floor"] - r / D) < 1e-12, cen


# ---------------------------------------------------------------------------
# 5. bimodality.
# ---------------------------------------------------------------------------
def test_bimodality_fires_on_two_clusters():
    lo = list(np.random.default_rng(0).normal(2, 0.3, 8))
    hi = list(np.random.default_rng(1).normal(40, 0.3, 8))
    r = QC.bimodality_test(lo + hi, seed=0)
    assert r["p_value"] < 0.05 and r["modes_hint"] == "bimodal", r
    assert r["low_mode_n"] == 8 and r["high_mode_n"] == 8, r


def test_bimodality_quiet_on_one_cluster():
    x = list(np.random.default_rng(2).normal(10, 2.0, 16))
    r = QC.bimodality_test(x, seed=0)
    assert r["p_value"] > 0.05 and r["modes_hint"] == "unimodal", r


# ---------------------------------------------------------------------------
# 6. Δn* + paired CI.
# ---------------------------------------------------------------------------
def test_delta_and_paired_ci():
    assert QC.delta_nstar(679, 20) == 659.0
    assert abs(QC.redundancy_ratio(679) - 33.95) < 0.01
    # compact bank: energy breaks much earlier than random -> large positive Δn*
    d = QC.paired_delta_ci([600, 640, 580, 610])
    assert d["ci_lo"] > 0 and not d["insufficient"], d
    # insufficient
    d2 = QC.paired_delta_ci([1.0, 2.0])
    assert d2["insufficient"] and d2["ci_lo"] is None, d2


# ---------------------------------------------------------------------------
# 7. the §6 adjudicator — every branch, in order.
# ---------------------------------------------------------------------------
def _kw(**over):
    kw = dict(delta_ci={"median": 5.0, "ci_lo": -5.0, "ci_hi": 12.0, "n": 8, "insufficient": False},
              nstar_energy_median=700.0, search_rank=20,
              pos_control_nstar=1.0, pos_control_fired=True,
              null_band_hi=3.0, dip_p=0.5, n_compact=0, n_redundant=8,
              any_learner=True, shuffle_matches_random=True)
    kw.update(over)
    return kw


def test_adjudicate_void_on_blind_instrument():
    v = QC.adjudicate_quorum(**_kw(pos_control_fired=False, pos_control_nstar=900.0))
    assert v["verdict"] == "VOID", v
    v2 = QC.adjudicate_quorum(**_kw(any_learner=False))
    assert v2["verdict"] == "VOID"


def test_adjudicate_insufficient_not_a_verdict_of_absence():
    v = QC.adjudicate_quorum(**_kw(delta_ci={"ci_lo": None, "n": 2, "insufficient": True}))
    assert v["verdict"] == "INSUFFICIENT-primary", v


def test_adjudicate_capability_dependent():
    v = QC.adjudicate_quorum(**_kw(dip_p=0.01, n_compact=3, n_redundant=4))
    assert v["verdict"] == "H_capability-dependent", v


def test_adjudicate_compact():
    v = QC.adjudicate_quorum(**_kw(delta_ci={"median": 600, "ci_lo": 500, "ci_hi": 660, "n": 8,
                                             "insufficient": False}, null_band_hi=50))
    assert v["verdict"] == "H_compact", v


def test_adjudicate_redundant_is_the_hypothesis_to_kill():
    v = QC.adjudicate_quorum(**_kw(delta_ci={"median": 0.0, "ci_lo": -8.0, "ci_hi": 8.0, "n": 8,
                                             "insufficient": False}, nstar_energy_median=900.0))
    assert v["verdict"] == "H_redundant", v


def test_adjudicate_borderline_when_nstar_near_search_rank():
    v = QC.adjudicate_quorum(**_kw(delta_ci={"median": 2.0, "ci_lo": -3.0, "ci_hi": 6.0, "n": 8,
                                             "insufficient": False}, nstar_energy_median=25.0))
    assert v["verdict"] == "H_compact-weak", v


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    for fn in ALL:
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nALL {len(ALL)} QUORUM dry-run logic-gate tests PASSED", flush=True)
    return len(ALL)


if __name__ == "__main__":
    import sys
    main()
    sys.exit(0)
