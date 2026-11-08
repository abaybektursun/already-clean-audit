"""HANDLE D-suite — weight-free rule tests (spec §9.1). Constructs EVERY §6 branch, the
degenerate-input completions, the delta-mixture marginal-matching, the module-matrix binomial
reachability, the projector nesting (tiny tensors), and the historical replays (the CARRY
raw-fraction rule-shape event; the saturated em-null; the all-tied corner). Pure numpy/torch,
runs on a CPU r7i instance (or `pytest infra/pg/tests/test_handle_core.py`).

Every rule in handle_core must classify each synthetic scenario correctly BEFORE any GPU minute.
"""
import math

import numpy as np
import torch

from infra.pg import carry_core as CC
from infra.pg import handle_core as HC


# ---------------------------------------------------------------------------
# ladder / nesting / bands
# ---------------------------------------------------------------------------
def _orthobasis(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return Q


def test_ladder_ks():
    assert HC.ladder_ks(679) == (5, 10, 20, 50, 150, 350, 679)
    assert HC.ladder_ks(32) == (5, 10, 20, 32)          # drops fixed rungs ≥ k_stiff
    assert HC.ladder_ks(200) == (5, 10, 20, 50, 150, 200)


def test_nesting_ok_and_negative():
    E = _orthobasis(64, 1)
    ks = HC.ladder_ks(32)
    P32 = E[:, :32] @ E[:, :32].T
    out = HC.assert_nesting(E, ks, 32, stiff_projector=P32)
    assert out["max_nest_err"] < 1e-8 and out["stiff_proj_err"] < 1e-6
    # non-orthonormal basis breaks the projector identity → must RAISE
    bad = torch.randn(64, 64, dtype=torch.float64)
    try:
        HC.assert_nesting(bad, ks, 32)
        raise SystemExit("nesting assert did not catch a non-orthonormal basis")
    except AssertionError:
        pass
    # wrong stiff projector → must RAISE
    try:
        HC.assert_nesting(E, ks, 32, stiff_projector=torch.eye(64, dtype=torch.float64))
        raise SystemExit("stiff-projector assert did not catch a wrong projector")
    except AssertionError:
        pass


def test_band_basis():
    E = _orthobasis(64, 2)
    U, idx = HC.band_basis(E, 10, 7)
    assert U.shape == (64, 10) and len(idx) == 10
    Us, idxs = HC.band_basis(E, 8, 7, within_stiff=32)
    assert Us.shape == (64, 8) and all(i < 32 for i in idxs)     # drawn from the stiff block only
    # degenerate top rung: random k-of-k = the full pool (must not crash random_eigsplit's k<d)
    Uf, idxf = HC.band_basis(E, 32, 7, within_stiff=32)
    assert Uf.shape == (64, 32) and idxf == list(range(32))


# ---------------------------------------------------------------------------
# k_half + median_ci degenerate completions
# ---------------------------------------------------------------------------
def test_k_half():
    ks = (5, 10, 20, 50, 150, 350, 679)
    compact = {5: 0.1, 10: 0.6, 20: 0.75, 50: 0.78, 150: 0.79, 350: 0.79, 679: 0.8}
    assert HC.k_half(compact, ks) == 10                    # half of 0.8 = 0.4 first at k=10
    broad = {5: 0.02, 10: 0.05, 20: 0.1, 50: 0.2, 150: 0.3, 350: 0.6, 679: 0.8}
    assert HC.k_half(broad, ks) == 350                     # reaches 0.4 only at 350
    noncarry = {k: -0.1 for k in ks}
    assert HC.k_half(noncarry, ks) is None                 # H-A1 completion


def test_median_ci_void():
    assert HC.median_ci([0.1, 0.2])["void"] is True        # < 3 → VOID
    m = HC.median_ci([0.1, 0.2, 0.3, 0.4])
    assert not m["void"] and "ci95" in m


# ---------------------------------------------------------------------------
# R-H — every branch
# ---------------------------------------------------------------------------
KS = (5, 10, 20, 50, 150, 350, 679)


def _rows(core_fn, band_fn, bstiff_fn, n=12, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        j = jitter * rng.standard_normal()
        rows.append({"ft": f"ft{i}",
                     "core": {k: core_fn(k) + j for k in KS},
                     "band": {k: band_fn(k) for k in KS},
                     "band_stiff": {k: bstiff_fn(k) for k in KS}})
    return rows


def test_rh_compact():
    core = lambda k: 0.2 if k == 5 else 0.79
    rows = _rows(core, lambda k: 0.30, lambda k: 0.30, jitter=0.01, seed=1)
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v["verdict"] == "H_compact", v["verdict"]


def test_rh_ordered_broad():
    prof = {5: 0.1, 10: 0.15, 20: 0.25, 50: 0.35, 150: 0.38, 350: 0.6, 679: 0.8}
    rows = _rows(lambda k: prof[k], lambda k: 0.30, lambda k: 0.15, jitter=0.005, seed=2)
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v["verdict"] == "H_ordered-broad", (v["verdict"], v["median_k_half"])


def test_rh_set_property():
    core = lambda k: 0.2 if k == 5 else 0.79
    # band^stiff EQUALS core at every rung ⇒ excess straddles everywhere; band low ⇒ top clears
    rows = _rows(core, lambda k: 0.30, core)
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v["verdict"] == "H_set-property", v["verdict"]


def test_rh_carry_fails():
    # top rung: core ≈ band ⇒ excess_top straddles ⇒ substructure did not reproduce
    rows = _rows(lambda k: 0.30, lambda k: 0.30, lambda k: 0.10)
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v["verdict"] == "H_carry-fails-to-redetect", v["verdict"]


def test_rh_void_range():
    rows = _rows(lambda k: 0.8, lambda k: 0.3, lambda k: 0.3, n=5)   # 7 of 12 VOID
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v["verdict"] == "VOID-range", v["verdict"]


def test_rh_partial():
    prof = {5: 0.1, 10: 0.2, 20: 0.3, 50: 0.35, 150: 0.45, 350: 0.6, 679: 0.8}
    bstiff = {5: 0.5, 10: 0.5, 20: 0.5, 50: 0.5, 150: 0.5, 350: 0.5, 679: 0.7}  # clears only at top
    rows = _rows(lambda k: prof[k], lambda k: 0.30, lambda k: bstiff[k], jitter=0.003, seed=3)
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v["verdict"] == "H_partial", (v["verdict"], v["n_rungs_stiff_cleared"],
                                         v["median_k_half"])


def test_select_kstar():
    core = lambda k: 0.2 if k == 5 else 0.79
    rows = _rows(core, lambda k: 0.30, lambda k: 0.30, jitter=0.01, seed=1)
    v = HC.adjudicate_rh(rows, ks=KS, k_stiff=679, n_sealed_total=12)
    assert HC.select_kstar(v, KS) == 10                    # first rung clearing both bands
    setp = {"verdict": "H_set-property", "per_rung": v["per_rung"]}
    assert HC.select_kstar(setp, KS) == HC.KSTAR_SETPROP   # 50 under set-property


# ---------------------------------------------------------------------------
# R-T — module matrix, binomial reachability, every branch
# ---------------------------------------------------------------------------
def test_diag_dominance_and_binom():
    M = np.eye(6) + 0.2 * (1 - np.eye(6))                  # strong diagonal
    diag, npairs = HC.diag_dominance_count(M)
    assert diag == 30 and npairs == 30
    assert HC.binom_p_ge(30, 30) < 1e-6
    assert HC.binom_p_ge(27, 30) <= 0.01                   # the 27/30 bar is reachable-significant
    assert HC.binom_p_ge(0, 5) == 1.0 and abs(HC.binom_p_ge(5, 5) - 0.5 ** 5) < 1e-12


def test_rt_modules():
    M = np.eye(6) * 1.0 + 0.2 * (1 - np.eye(6))
    deletion = [{"own_drop": 1.0, "ppl_drop": 0.1, "nontarget_drop": 0.1} for _ in range(6)]
    census = {"cross_overlap_med": 0.3, "haar_p95": 0.3}
    v = HC.adjudicate_rt(module_M=M, deletion_rows=deletion, census=census, n_disc_void=0)
    assert v["verdict"] == "H_modules", v


def test_rt_shared_substrate():
    # gain depends only on the task-COLUMN, not which FT's core (row-independent) ⇒ no diagonal
    # preference ⇒ diag_dominance = 0; cores overlap ≫ Haar ⇒ shared substrate
    M = np.tile(np.array([0.3, 0.5, 0.4, 0.6, 0.35, 0.55]), (6, 1))
    deletion = [{"own_drop": 0.2, "ppl_drop": 0.2, "nontarget_drop": 0.2} for _ in range(6)]
    census = {"cross_overlap_med": 0.6, "haar_p95": 0.3}
    v = HC.adjudicate_rt(module_M=M, deletion_rows=deletion, census=census, n_disc_void=0)
    assert v["verdict"] == "H_shared-substrate", v


def test_rt_neither():
    M = np.tile(np.array([0.3, 0.5, 0.4, 0.6, 0.35, 0.55]), (6, 1))
    deletion = [{"own_drop": 0.2, "ppl_drop": 0.2, "nontarget_drop": 0.2} for _ in range(6)]
    census = {"cross_overlap_med": 0.28, "haar_p95": 0.30}   # census at Haar
    v = HC.adjudicate_rt(module_M=M, deletion_rows=deletion, census=census, n_disc_void=0)
    assert v["verdict"] == "H_neither", v


def test_rt_void():
    v = HC.adjudicate_rt(module_M=np.eye(6), deletion_rows=[{"own_drop": 1, "ppl_drop": 0,
                         "nontarget_drop": 0}], census={"cross_overlap_med": 0.3, "haar_p95": 0.3},
                         n_disc_void=3)
    assert v["verdict"] == "VOID", v


# ---------------------------------------------------------------------------
# R-Q — race + delta ladder, every branch + marginal matching
# ---------------------------------------------------------------------------
def _race(scores, damages):
    return HC.property_race(scores, damages, list(scores), n_shuffle=2000, seed=0)


def test_rq_dependence_mechanism():
    dmg = list(np.linspace(1.0, 20.0, 12))                       # span 20× > 10
    scores = {"dependence_fraction": list(dmg),                  # ρ=1
              "feature_alignment": list(np.random.default_rng(0).permutation(dmg)),
              "massive_coord": list(np.random.default_rng(1).permutation(dmg)),
              "sae_feature": list(np.random.default_rng(2).permutation(dmg))}
    race = _race(scores, dmg)
    deltas = HC.RQ_DELTA_GRID
    ladder = HC.monotone_negative(deltas, [5.0, 4.0, 3.0, 2.0, 1.0])       # mono-neg
    twin = HC.monotone_negative(deltas, [3.0, 3.1, 2.9, 3.2, 3.0])         # not mono-neg
    v = HC.adjudicate_rq(race=race, delta_ladder=ladder, delta_ladder_twin=twin, damage_span=20.0)
    assert v["verdict"] == "H_dependence-mechanism", v


def test_rq_none_ranks():
    dmg = list(np.linspace(1.0, 20.0, 12))
    rng = np.random.default_rng(3)
    scores = {p: list(rng.permutation(dmg)) for p in
              ("dependence_fraction", "feature_alignment", "massive_coord", "sae_feature")}
    race = _race(scores, dmg)
    ladder = HC.monotone_negative(HC.RQ_DELTA_GRID, [3, 3, 3, 3, 3])
    v = HC.adjudicate_rq(race=race, delta_ladder=ladder, delta_ladder_twin=ladder, damage_span=20.0)
    assert v["verdict"] == "H_none-ranks", v


def test_rq_void_no_range():
    dmg = [1.0] * 12
    scores = {p: dmg for p in ("dependence_fraction", "feature_alignment",
                               "massive_coord", "sae_feature")}
    race = _race(scores, dmg)
    ladder = HC.monotone_negative(HC.RQ_DELTA_GRID, [5, 4, 3, 2, 1])
    v = HC.adjudicate_rq(race=race, delta_ladder=ladder, delta_ladder_twin=ladder, damage_span=1.0)
    assert v["verdict"] == "VOID-no-range", v


def test_delta_mix_marginal_matching():
    rng = torch.Generator().manual_seed(5)
    R_own = torch.randn(8, 16, generator=rng, dtype=torch.float64)
    R_off = torch.randn(8, 16, generator=rng, dtype=torch.float64)
    own_norm = R_own.norm(dim=1)
    for d in HC.RQ_DELTA_GRID:
        e = HC.delta_mix(R_own, R_off, d)
        assert torch.allclose(e.norm(dim=1), own_norm, atol=1e-9), f"marginal broken at δ={d}"
    twin = HC.rotate_offtoken(R_off, 9)
    assert torch.allclose(twin.norm(dim=1), R_off.norm(dim=1), atol=1e-9)   # rotation preserves norm


# ---------------------------------------------------------------------------
# R-N-a / R-N-b / R-B / R-X
# ---------------------------------------------------------------------------
def test_rna():
    named = HC.adjudicate_rna({"massive": {"tau": 0.7, "p": 0.01}, "hbar": {"tau": 0.1, "p": 0.6},
                               "spike_p8": {"tau": 0.2, "p": 0.5}, "gamma": {"tau": 0.0, "p": 0.9},
                               "n_eff": {"tau": -0.1, "p": 0.7}})
    assert named["verdict"] == "H_carrier-named" and named["carrier"] == "massive"
    miss = HC.adjudicate_rna({n: {"tau": 0.1, "p": 0.6} for n in
                              ("massive", "hbar", "spike_p8", "gamma", "n_eff")})
    assert miss["verdict"] == "H_carrier-escapes"


def test_rnb():
    ov = {"patch_harmonic": 0.9, "image_mean": 0.2, "projector_rowspace": 0.15}
    p95 = {"patch_harmonic": 0.25, "image_mean": 0.25, "projector_rowspace": 0.25}
    assert HC.adjudicate_rnb(ov, p95, rayleigh_drop=0.6)["verdict"] == "H_spikes-identified-causal"
    assert HC.adjudicate_rnb(ov, p95, rayleigh_drop=0.3)["verdict"] == \
        "H_spikes-identified-descriptive"
    assert HC.adjudicate_rnb({"patch_harmonic": 0.2}, {"patch_harmonic": 0.25})["verdict"] == \
        "H_spikes-new-object"


def test_rb():
    lo = HC.adjudicate_rb(d_eff=4.0, mean_min_angle=0.2, haar_mean_min_angle=1.2,
                          overlap_census={}, install_rho=0.6, install_shuffle_p=0.02)
    assert lo["verdict"] == "H_low-dim-surface" and lo["install_tracking"]["tracks"]
    sc = HC.adjudicate_rb(d_eff=40.0, mean_min_angle=1.1, haar_mean_min_angle=1.2,
                          overlap_census={}, install_rho=0.0, install_shuffle_p=0.9)
    assert sc["verdict"] == "H_scattered"


def test_rx():
    v = HC.adjudicate_rx({"two_regime": {"pg1": "A", "pg2": "A"},
                          "operator_order": {"pg1": "A", "pg2": "B"},
                          "kernel": {"pg1": "A", "pg2": "A", "void": True}})
    assert v["per_law"] == {"two_regime": "replicates", "operator_order": "lands-elsewhere",
                            "kernel": "VOID"}
    assert v["n_lands_elsewhere"] == 1


# ---------------------------------------------------------------------------
# degenerate completions + historical replays (one-sided-thresholds law)
# ---------------------------------------------------------------------------
def test_all_tied_corner():
    assert CC.spearman0([1, 1, 1], [1, 2, 3]) == 0.0
    assert CC.kendall0([1, 2, 3], [2, 2, 2]) == 0.0


def test_replay_carry_rule_shape():
    """The CARRY raw-fraction rule-shape event: raw gf(top)≈0.76 alone cannot say whether the
    substructure carries; the NULL-REFERENCED excess must. (a) CARRY-like — stiff beats its band
    (excess_top clears) ⇒ R-H proceeds PAST the carry-fails guard. (b) saturated em-null — band
    equals core at top (excess_top straddles) ⇒ H_carry-fails-to-redetect. Same raw gf, opposite
    verdicts — exactly the two-axis truth a raw-fraction rule (CARRY R-C's frozen shape) could
    not express."""
    core = lambda k: 0.2 if k == 5 else 0.76               # CARRY-like top gf
    carry_like = _rows(core, lambda k: 0.55, lambda k: 0.55, jitter=0.01, seed=7)
    v1 = HC.adjudicate_rh(carry_like, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v1["verdict"] != "H_carry-fails-to-redetect", v1["verdict"]
    saturated = _rows(core, core, lambda k: 0.3)           # band == core at top ⇒ excess≈0
    v2 = HC.adjudicate_rh(saturated, ks=KS, k_stiff=679, n_sealed_total=12)
    assert v2["verdict"] == "H_carry-fails-to-redetect", v2["verdict"]


ALL = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    passed = 0
    for t in ALL:
        t()
        passed += 1
        print(f"[handle-dsuite] {t.__name__} OK", flush=True)
    print(f"HANDLE_DSUITE_OK {passed}/{len(ALL)}")
    return passed


if __name__ == "__main__":
    main()
