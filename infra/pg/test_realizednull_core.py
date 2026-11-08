"""REALIZEDNULL weight-free dry-run tests (SPEC §9.2 N2, §9.3 step 5): pure logic on tiny tensors.

One test per behaviour. Every registered failing input of §6.1 that N1 evaluates fails its gate on a
baseline that passes every gate (test_good_inputs_pass_every_gate). Every §6.4 branch is reached on
synthetic per-window NLL arrays. The registered tests that need the job or the analysis script
(test_probe_hooks_out_of_place, test_no_confirm_stat_in_job_outputs, test_noleak_script_flags_range,
test_unblind_refuses_without_committed_sha) belong to N3 and N5.
Run: python3 -m pytest infra/pg/test_realizednull_core.py -q
"""
import copy
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats as sps
from scipy.optimize import brentq

from infra.pg import gutenberg_core as G
from infra.pg import realizednull_core as R

ROOT = pathlib.Path(__file__).resolve().parents[2]


# =====================================================================================================
# §2.2 — eigen-coordinates, realized quantity, grid scan, construction
# =====================================================================================================

def test_realized_quantity_equals_direct_sum():
    rng = np.random.default_rng(1)
    n, k = 16, 3
    H_all = rng.standard_normal((50, n))
    H_T = H_all[:30]
    W = rng.standard_normal((n, n))
    ec = R.eigencoords(H_all.T @ H_all, W.T @ W, H_T.T @ H_T)
    assert np.all(np.diff(ec["mu"]) <= 0) and ec["mu"].min() >= 0
    assert np.allclose(ec["V"].T @ ec["V"], np.eye(n), atol=1e-12)
    U = np.linalg.qr(rng.standard_normal((n, k)))[0]
    Ue = ec["V"].T @ U
    direct_N = float(((H_T @ U @ U.T @ W.T) ** 2).sum())
    assert R.realized_quantity(Ue, ec["Mp"], ec["Sp"]) == pytest.approx(direct_N, rel=1e-10)
    assert ec["denom"] == pytest.approx(float(((H_T @ W.T) ** 2).sum()), rel=1e-10)
    ec_I = R.eigencoords(H_all.T @ H_all, np.eye(n), H_T.T @ H_T)
    direct_E = float(((H_T @ U) ** 2).sum())
    assert R.realized_quantity(Ue, ec_I["Mp"], ec_I["Sp"]) == pytest.approx(direct_E, rel=1e-10)


def test_core_rejects_bad_shapes_and_nonfinite():
    I4 = np.eye(4)
    U = np.eye(4)[:, :2]
    with pytest.raises(ValueError):
        R.realized_quantity(U, I4, np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        R.realized_quantity(U, np.eye(5), I4)
    with pytest.raises(ValueError):
        R.eigencoords(np.ones((4, 3)), I4, I4)
    with pytest.raises(ValueError):
        R.per_draw_ratios(np.ones((3, 5)), np.ones(4))


def test_reach_grid_log_spaced_and_deterministic():
    rng = np.random.default_rng(2)
    n = 64
    A = rng.standard_normal((n, n))
    Mp, Sp = A @ A.T, np.diag(np.exp(-np.arange(n) / 8.0))
    g1 = R.reach_grid(Mp, Sp, k=4, n_grid=16, n_probe=4, seed=190_000)
    g2 = R.reach_grid(Mp, Sp, k=4, n_grid=16, n_probe=4, seed=190_000)
    assert g1["m"][0] == 4 and g1["m"][-1] == n and np.unique(g1["m"]).size == 16
    assert all(np.array_equal(g1[key], g2[key]) for key in g1)
    assert g1["median_top"].shape == g1["median_bottom"].shape == (16,)
    with pytest.raises(ValueError):                 # 60 log-spaced integers in [4, 64] repeat
        R.reach_grid(Mp, Sp, k=4, n_grid=60, n_probe=4, seed=0)


def _planted(n=64, k=4, n_layers=6, seed=0, light=(0,), light_scale=0.03):
    """A 64-dim problem per layer: activations with a decaying spectrum, a random W_q (scaled by light_scale at
    the light layers), and a band drawn as a Haar frame inside the top-12 eigenvectors, so its N is reachable."""
    rng = np.random.default_rng(seed)
    Mp, Sp, targets, raw = [], [], [], []
    for L in range(n_layers):
        scale = np.sqrt(np.exp(-np.arange(n) / 8.0))
        H_T = rng.standard_normal((300, n)) * scale
        H_all = np.vstack([H_T, rng.standard_normal((100, n)) * scale])
        W = rng.standard_normal((n, n)) / math.sqrt(n) * (light_scale if L in light else 1.0)
        ec = R.eigencoords(H_all.T @ H_all, W.T @ W, H_T.T @ H_T)
        Ub = np.zeros((n, k))
        Ub[:12] = np.linalg.qr(rng.standard_normal((12, k)))[0]
        targets.append(R.realized_quantity(Ub, ec["Mp"], ec["Sp"]))
        Mp.append(ec["Mp"])
        Sp.append(ec["Sp"])
        raw.append((H_T, W, ec["V"]))
    return Mp, Sp, np.array(targets), raw


KW = dict(k=4, n_grid=16, n_probe=8)


@pytest.fixture(scope="module")
def planted():
    Mp, Sp, tg, raw = _planted()
    grids = [R.reach_grid(Mp[L], Sp[L], k=4, n_grid=16, n_probe=8, seed=R.GRID_SEED_BASE + L)
             for L in range(len(Mp))]
    return Mp, Sp, tg, raw, grids


def test_rt_construction_planted_64dim(planted):
    """§2.2: the summed realized perturbation within 0.5%, heavy layers within 5%, frames orthonormal to
    1e-10, and N equal to the hook-side sum Σ_t ‖W_q U Uᵀ h_t‖² in model coordinates."""
    Mp, Sp, tg, raw, grids = planted
    heavy = tg >= 0.01 * tg.sum()
    assert not heavy[0] and heavy[1:].all()          # layer 0 is light, so it is free
    for s in range(5):
        res = R.construct_draw(Mp, Sp, tg, 200_000 + 1000 * s, grids=grids, **KW)
        assert res["matched"] and not res["exhausted"] and res["reachable"]
        assert abs(res["N_sum"] / tg.sum() - 1) <= 0.005
        assert np.all(np.abs(res["N"][heavy] / tg[heavy] - 1) <= 0.05)
        for L, (H_T, W, V) in enumerate(raw):
            assert np.abs(res["Ue"][L].T @ res["Ue"][L] - np.eye(4)).max() < 1e-10
            U = V @ res["Ue"][L]
            assert float(((H_T @ U @ U.T @ W.T) ** 2).sum()) == pytest.approx(res["N"][L], rel=1e-9)
        assert len(res["family"]) == len(res["m"]) == res["draws"].size == 6


def test_rt_construction_closes_tight_sum_without_pivot(planted):
    """§2.2 step 3 does the closing: at tol_sum 0.1% the attempt matches through the closing rounds."""
    Mp, Sp, tg, _, grids = planted
    res = R.realized_matched_subspace(Mp, Sp, tg, 200_000, grids=grids, tol_sum=0.001, **KW)
    assert res["matched"] and not res["pivot"]
    assert abs(res["N_sum"] / tg.sum() - 1) <= 0.001


def test_closing_absorbs_light_error_at_heaviest_layer():
    """§2.2 step 3: five light layers (free, 0.7–1.0% of N* each) leave a summed error; the heaviest layer is
    redrawn toward its residual target N* − Σ_{L≠hl} N_L, so it ends off its own target by minus the light
    error, and the attempt closes without the pivot."""
    Mp, Sp, tg, _ = _planted(light=(0, 1, 2, 3, 4), light_scale=0.1)
    grids = [R.reach_grid(Mp[L], Sp[L], k=4, n_grid=16, n_probe=8, seed=R.GRID_SEED_BASE + L) for L in range(6)]
    res = R.realized_matched_subspace(Mp, Sp, tg, 53, grids=grids, tol_sum=0.001, **KW)
    light_err = (res["N"][:5] - tg[:5]).sum() / tg.sum()
    hl_dev = (res["N"][5] - tg[5]) / tg.sum()
    assert abs(light_err) > 0.001                                # the light layers alone miss the sum
    assert res["matched"] and not res["pivot"]
    assert abs(hl_dev + light_err) <= 0.001


def test_heavy_layer_uses_grid_neighbour():
    """§2.2 step 2: when no frame at m*_L lands within 5% the grid neighbours are tried. The grid here is
    synthetic: index 2 claims the target at m = 64, where real frames cost a quarter of it; neighbour 3 is
    m = 18, where they match."""
    Mp, Sp, tg, _ = _planted(n_layers=1, light=())
    t = tg[0]
    grid = {"m": np.array([64, 64, 64, 18, 64, 64]), "median_top": t * np.array([0.5, 0.5, 1.0, 2.0, 2.0, 2.0]),
            "median_bottom": np.full(6, 100 * t)}
    res = R.realized_matched_subspace(Mp, Sp, tg, 11, k=4, max_resample=200, grids=[grid])
    assert res["matched"] and res["m"] == [18] and res["family"] == ["top"]
    assert res["draws"][0] >= 400                               # 200 at index 2, 200 at index 1, then index 3


def test_construction_deterministic_per_seed(planted):
    Mp, Sp, tg, _, grids = planted
    a = R.realized_matched_subspace(Mp, Sp, tg, 7, grids=grids, **KW)
    b = R.realized_matched_subspace(Mp, Sp, tg, 7, grids=grids, **KW)
    c = R.realized_matched_subspace(Mp, Sp, tg, 8, grids=grids, **KW)
    assert all(np.array_equal(x, y) for x, y in zip(a["Ue"], b["Ue"]))
    assert not np.array_equal(a["Ue"][1], c["Ue"][1])


def test_grids_none_uses_registered_seeds(planted):
    Mp, Sp, tg, _, grids = planted
    a = R.realized_matched_subspace(Mp, Sp, tg, 7, grids=None, **KW)
    b = R.realized_matched_subspace(Mp, Sp, tg, 7, grids=grids, **KW)
    assert np.array_equal(a["N"], b["N"])


def test_reach_target_outside_range(planted):
    """G-REACH registered input: a target above every grid median returns reachable = False, no raise."""
    Mp, Sp, tg, _, grids = planted
    res = R.realized_matched_subspace(Mp, Sp, tg * 1e3, 1, grids=grids, **KW)
    assert not res["reachable"] and not res["matched"] and res["Ue"] is None
    assert set(res["unreachable_layers"]) >= {1, 2, 3, 4, 5, "sum"}
    jm, dc = _good()
    jm["construction"]["rt"] = [_rec(reachable=False, matched=False, unreachable_layers=[1, "sum"])] * 40
    g = R.gates(jm, dc)
    assert not g["G-REACH:rt"]["pass"] and g["G-REACH:rt"]["label"] == "RT-DOES-NOT-COUNT"


def test_reach_nonmonotone_medians():
    """G-REACH registered input: medians that alternate 1, 10, 1, 10 … span the target 5, but no grid point's
    two neighbours bracket it, so the heavy layer is unreachable (a bisection on the range would accept it).
    A non-monotone profile that does bracket picks the bracketing point with the closest median."""
    n = 8
    alt = {"m": np.arange(2, 8), "median_top": np.array([1.0, 10, 1, 10, 1, 10]),
           "median_bottom": np.array([1.0, 10, 1, 10, 1, 10])}
    res = R.realized_matched_subspace([np.eye(n)], [np.eye(n)], np.array([5.0]), 0, k=2, grids=[alt])
    assert not res["reachable"] and res["unreachable_layers"] == [0]
    bumpy = {"m": np.arange(2, 8), "median_top": np.array([1.0, 7, 3, 5.2, 9, 2]),
             "median_bottom": np.full(6, 100.0)}
    assert R._bracket_choice(bumpy, 5.0) == ("top", 3)     # neighbours 3 and 9 bracket; 5.2 is closest


def test_construction_exhausts_without_raise(planted):
    """§2.2 step 6 and G-EXHAUST registered input: four failed attempts end the draw index as exhausted with
    redraw = 3, for a target the attempts cannot match and for a target outside the reach; no raise."""
    Mp, Sp, tg, _, grids = planted
    hard = R.construct_draw(Mp, Sp, tg, 5, grids=grids, tol_layer=1e-12, max_resample=2, **KW)
    assert hard["exhausted"] and hard["redraw"] == 3 and hard["reachable"] and not hard["matched"]
    assert "heavy layer" in hard["fail_reason"]
    far = R.construct_draw(Mp, Sp, tg * 1e3, 5, grids=grids, **KW)
    assert far["exhausted"] and far["redraw"] == 3 and not far["reachable"]
    jm, dc = _good()
    jm["construction"]["rt"] = [_rec(matched=False, exhausted=True)] * 11 + [_rec()] * 29
    g = R.gates(jm, dc)
    assert not g["G-EXHAUST:rt"]["pass"] and g["G-EXHAUST:rt"]["value"] == pytest.approx(11 / 40)
    jm["construction"]["rt"] = [_rec(matched=False, exhausted=True)] * 10 + [_rec()] * 30
    assert R.gates(jm, dc)["G-EXHAUST:rt"]["pass"]


def test_pivot_closes_sum_and_is_flagged(planted):
    """§2.2 step 4: with no closing rounds and tol_sum 1e-6 the φ pivot closes the sum and marks the draw."""
    Mp, Sp, tg, _, grids = planted
    res = R.realized_matched_subspace(Mp, Sp, tg, 1000, grids=grids, tol_sum=1e-6, max_closing=0,
                                      tol_layer=0.02, **KW)
    assert res["matched"] and res["pivot"]
    assert abs(res["N_sum"] / tg.sum() - 1) <= 1e-6


def test_pivot_cannot_bracket_ends_attempt(planted):
    """§2.2 step 4: one frame drawn at the heaviest layer cannot bracket its residual target."""
    Mp, Sp, tg, _, grids = planted
    res = R.realized_matched_subspace(Mp, Sp, tg, 3, grids=grids, tol_sum=1e-6, max_closing=0,
                                      tol_layer=1e6, max_resample=1, **KW)
    assert not res["matched"] and not res["pivot"] and "cannot bracket" in res["fail_reason"]


def test_pivot_share_cap():
    """G-PIVOT registered input: 11 of 40 pivoted draws exceed 25%; 10 of 40 do not."""
    jm, dc = _good()
    jm["construction"]["rt"] = [_rec(pivot=True)] * 11 + [_rec()] * 29
    g = R.gates(jm, dc)
    assert not g["G-PIVOT:rt"]["pass"]
    assert _verdict_on_good(jm, dc)["label"] == "RT-DOES-NOT-COUNT (G-PIVOT:rt)"
    jm["construction"]["rt"] = [_rec(pivot=True)] * 10 + [_rec()] * 30
    assert R.gates(jm, dc)["G-PIVOT:rt"]["pass"]


# =====================================================================================================
# §2.3, §3.6 — means, centered Grams, arm statistics
# =====================================================================================================

def _positions(rng, n=10, n_windows=4):
    """Per window: 3 prefix positions (classes 0, 1, 2), then a 6-position text block (class 3) whose first
    4 positions are the scored set T."""
    H, cls, inT = [], [], []
    for _ in range(n_windows):
        for c in range(3):
            H.append(rng.standard_normal(n) + c)
            cls.append(c)
            inT.append(False)
        for j in range(6):
            H.append(rng.standard_normal(n) * 2 + 0.5)
            cls.append(3)
            inT.append(j < 4)
    return np.array(H), np.array(cls), np.array(inT)


def test_class_means_and_centered_gram():
    rng = np.random.default_rng(3)
    H, cls, _ = _positions(rng)
    sums = {c: H[cls == c].sum(axis=0) for c in range(4)}
    counts = {c: int((cls == c).sum()) for c in range(4)}
    mu = R.class_means(sums, counts)
    assert np.allclose(mu[3], H[cls == 3].mean(axis=0))
    Hc = H - np.array([mu[c] for c in cls])
    assert np.allclose(R.centered_gram(H.T @ H, sums, counts), Hc.T @ Hc)
    with pytest.raises(ValueError):
        R.class_means(sums, {**counts, 3: 0})
    with pytest.raises(ValueError):
        R.class_means(sums, {c: counts[c] for c in range(3)})


def test_arm_stats_matches_direct_sums():
    rng = np.random.default_rng(4)
    n, k = 10, 2
    H, cls, inT = _positions(rng, n)
    W = rng.standard_normal((n, n))
    U = np.linalg.qr(rng.standard_normal((n, k)))[0]
    S_all, S_T = H.T @ H, H[inT].T @ H[inT]
    M = W.T @ W
    ec = R.eigencoords(S_all, M, S_T)
    sums = {c: H[cls == c].sum(axis=0) for c in range(4)}
    counts = {c: int((cls == c).sum()) for c in range(4)}
    mu = R.class_means(sums, counts)
    grams = {"all": [S_all], "T": [S_T], "V": [ec["V"]], "denom_T": [ec["denom"]],
             "denom_all": [R.eigencoords(S_all, M, S_all)["denom"]]}
    means = {"mu": [np.array([mu[c] for c in range(4)])], "counts": [np.array([counts[c] for c in range(4)])],
             "text": [3], "T_sum": [H[inT].sum(axis=0)], "T_count": [int(inT.sum())]}
    st = R.arm_stats([U], grams, [W], means)
    P = U @ U.T
    HT = H[inT]
    assert st["e_all"][0] == pytest.approx(((H @ U) ** 2).sum() / (H ** 2).sum())
    assert st["e_T"][0] == pytest.approx(((HT @ U) ** 2).sum() / (HT ** 2).sum())
    assert st["E_T"][0] == pytest.approx(((HT @ U) ** 2).sum())
    assert st["N_T"][0] == pytest.approx(((HT @ P @ W.T) ** 2).sum())
    assert st["rho_T"][0] == pytest.approx(((HT @ P @ W.T) ** 2).sum() / ((HT @ W.T) ** 2).sum())
    assert st["N_all"][0] == pytest.approx(((H @ P @ W.T) ** 2).sum())
    assert st["rho_all"][0] == pytest.approx(((H @ P @ W.T) ** 2).sum() / ((H @ W.T) ** 2).sum())
    assert st["w_top1"][0] == pytest.approx(((U.T @ ec["V"][:, 0]) ** 2).sum())
    assert st["w_top5"][0] == pytest.approx(((U.T @ ec["V"][:, :5]) ** 2).sum())
    assert st["leverage_max"][0] == pytest.approx((U ** 2).sum(axis=1).max())
    assert st["gain"][0] == pytest.approx(((W @ U) ** 2).sum() / k)
    Hc = H - np.array([mu[c] for c in cls])
    assert st["e_all_centered"][0] == pytest.approx(((Hc @ U) ** 2).sum() / (Hc ** 2).sum())
    assert st["N_T_centered"][0] == pytest.approx((((HT - mu[3]) @ P @ W.T) ** 2).sum())
    assert st["N_T_sum"] == pytest.approx(st["N_T"][0])
    assert "e_all_centered" not in R.arm_stats([U], grams, [W], None)


# =====================================================================================================
# §2.4 — attention-mask identification
# =====================================================================================================

PROBES = {"causal": dict(d_before=0.0, d_after=1.5, r_prior=0.0, r_self=0.2),
          "bidir": dict(d_before=0.9, d_after=1.5, r_prior=0.04, r_self=0.2),
          "prefixlm": dict(d_before=0.0, d_after=1.5, r_prior=0.04, r_self=0.2),
          "unresolved": dict(d_before=0.01, d_after=1.5, r_prior=0.0, r_self=0.2)}


def test_mask_probe_four_outputs():
    assert R.mask_probe_verdict(**PROBES["causal"]) == "causal"
    assert R.mask_probe_verdict(**PROBES["prefixlm"]) == "prefixlm"
    assert R.mask_probe_verdict(**PROBES["bidir"]) == "bidir"
    assert R.mask_probe_verdict(**PROBES["unresolved"]) == "MASK-DOES-NOT-COUNT"
    # §2.4 maps BIDIRECTIONAL to bidir whatever the image label says (read literally)
    assert R.mask_probe_verdict(d_before=0.9, d_after=1.5, r_prior=0.0, r_self=0.2) == "bidir"
    with pytest.raises(ValueError):
        R.mask_probe_verdict(d_before=float("nan"), d_after=1.0, r_prior=0.0, r_self=0.2)


def test_mask_probe_unresolved():
    """G-MASK registered input: d_before between 1e-3 and 0.05 does not count."""
    for d in (2e-3, 0.01, 0.049):
        assert R.mask_probe_verdict(d_before=d, d_after=1.5, r_prior=0.0, r_self=0.2) == "MASK-DOES-NOT-COUNT"
    assert R.mask_probe_verdict(d_before=1e-3, d_after=1.5, r_prior=0.0, r_self=0.2) == "causal"


def test_mask_probe_dead_edit():
    """G-MASK registered input: the edit moved nothing after it (d_after < 0.05), the positive control is dead."""
    assert R.mask_probe_verdict(d_before=0.0, d_after=0.01, r_prior=0.0, r_self=0.2) == "MASK-DOES-NOT-COUNT"


def test_image_probe_units():
    """G-MASK registered input, in relative Frobenius units: r_self < 1e-3 (dead perturbation) and
    1e-6 < r_prior < 1e-3 (between the thresholds) do not count."""
    assert R.mask_probe_verdict(d_before=0.0, d_after=1.5, r_prior=0.0, r_self=5e-4) == "MASK-DOES-NOT-COUNT"
    assert R.mask_probe_verdict(d_before=0.0, d_after=1.5, r_prior=1e-5, r_self=0.2) == "MASK-DOES-NOT-COUNT"
    assert R.mask_probe_verdict(d_before=0.0, d_after=1.5, r_prior=1e-6, r_self=0.2) == "causal"
    assert R.mask_probe_verdict(d_before=0.0, d_after=1.5, r_prior=1e-3, r_self=0.2) == "prefixlm"


def _mask(jm, native, forced):
    """forced: None = forced masks off; else {mode: probe-case} for the three forced modes."""
    jm["mask_probe"] = {"native": PROBES[native], "forced_masks": "off" if forced is None else "on",
                        "forced": {} if forced is None else {m: PROBES[c] for m, c in forced.items()}}


def test_mask_unknown_cases():
    """§2.4 cases a–d give MASK-UNKNOWN; a causal native mode, or a working forced-causal mode, does not."""
    ok = {"causal": "causal", "bidir": "bidir", "prefixlm": "prefixlm"}
    broken_causal = {"causal": "unresolved", "bidir": "bidir", "prefixlm": "prefixlm"}
    cases = {"a": ("bidir", None), "b": ("unresolved", broken_causal), "c": ("unresolved", None),
             "d": ("prefixlm", broken_causal)}
    for name, (native, forced) in cases.items():
        jm, dc = _good()
        _mask(jm, native, forced)
        g = R.gates(jm, dc)["G-MASK"]
        assert g["r3_required"] and g["mask_unknown"], name
        assert "MASK-UNKNOWN" in _verdict_on_good(jm, dc)["notes"], name
    for native, forced in (("causal", None), ("bidir", ok), ("unresolved", ok)):
        jm, dc = _good()
        _mask(jm, native, forced)
        g = R.gates(jm, dc)["G-MASK"]
        assert not g["mask_unknown"]
        assert g["pass"] == (native != "unresolved")
        assert g["r3_required"] == (native != "causal")
    jm, dc = _good()
    jm["run"], jm["mask"], jm["pre1_reference"] = "R3", "causal", R.GB1_PPL_Q["none"]
    _mask(jm, "bidir", None)
    assert not R.gates(jm, dc)["G-MASK"]["mask_unknown"]      # R3 is itself the forced-causal run


# =====================================================================================================
# §2.1, §6.4 — ratios and intervals
# =====================================================================================================

def test_per_draw_ratios_and_raise():
    Yb = np.array([0.1, 0.2, 0.3])
    Y = np.array([2 * Yb, 3 * Yb])
    assert np.allclose(R.per_draw_ratios(Y, Yb), [2.0, 3.0])
    with pytest.raises(ValueError):
        R.per_draw_ratios(Y, np.array([0.1, -0.2, 0.1]))


def test_decision_interval_is_hull_and_seeded():
    rng = np.random.default_rng(5)
    Yb = 0.15 * (1 + 0.3 * rng.standard_normal(20) ** 2)
    r = np.exp(0.2 + 0.1 * rng.standard_normal(30))
    Y = r[:, None] * Yb[None, :] * (1 + 0.05 * rng.standard_normal((30, 20)))
    a = R.decision_interval(Y, Yb, n_boot=2000)
    b = R.decision_interval(Y, Yb, n_boot=2000)
    assert a == b
    lr = np.log(R.per_draw_ratios(Y, Yb))
    assert a["theta"] == pytest.approx(math.exp(lr.mean()))
    h = sps.t.ppf(0.975, 29) * lr.std(ddof=1) / math.sqrt(30)
    assert a["logt"] == pytest.approx((math.exp(lr.mean() - h), math.exp(lr.mean() + h)))
    assert a["lo"] == min(a["logt"][0], a["boot"][0]) and a["hi"] == max(a["logt"][1], a["boot"][1])
    assert a["n_degenerate"] == 0 and a["n_boot"] == 2000


def test_difference_and_prediction_intervals():
    c = np.array([0.31, 0.29, 0.30, 0.33, 0.27])
    m, lo, hi = R.difference_interval(c, 0.1)
    h = sps.t.ppf(0.975, 4) * c.std(ddof=1) / math.sqrt(5)
    assert (m, lo, hi) == pytest.approx((0.2, 0.2 - h, 0.2 + h))
    x = np.log(c)
    plo, phi = R.prediction_interval(x)
    w = sps.t.ppf(0.975, 4) * x.std(ddof=1) * math.sqrt(1 + 1 / 5)
    assert (plo, phi) == pytest.approx((x.mean() - w, x.mean() + w))


def test_boot_degenerate_denominator():
    """G-BOOT registered input: a band cost near 0 with window noise makes > 1% of replicates degenerate."""
    rng = np.random.default_rng(6)
    Yb = 0.002 + 0.05 * rng.standard_normal(30)
    Y = 0.45 + 0.01 * rng.standard_normal((40, 30))
    di = R.decision_interval(Y, Yb, n_boot=2000)
    assert di["n_degenerate"] / di["n_boot"] > 0.01
    jm, dc = _good()
    dc["boot"]["E"] = {"n_degenerate": di["n_degenerate"], "n_boot": di["n_boot"]}
    g = R.gates(jm, dc)
    assert not g["G-BOOT:E"]["pass"]
    assert _verdict_on_good(jm, dc)["label"] == "E-DOES-NOT-COUNT (G-BOOT:E)"


# =====================================================================================================
# §6.2 — power, thresholds, pseudo-band calibration
# =====================================================================================================

# §3.2's table: D, s, P(CONTROL-RULE | 1), P(lo > δ | 1.30), P(BEYOND-CHEAPER | 1.30), smallest θ the PI admits
POWER_TABLE = [(20, 0.15, 0.93, 0.97, 0.04, 1.38), (20, 0.22, 0.44, 0.72, 0.00, 1.60),
               (40, 0.10, 1.00, 1.00, 1.00, 1.23), (40, 0.15, 1.00, 1.00, 0.03, 1.36),
               (40, 0.22, 0.92, 0.96, 0.00, 1.57), (40, 0.255, 0.78, 0.89, 0.00, 1.69),
               (60, 0.22, 0.99, 1.00, 0.00, 1.56), (60, 0.255, 0.95, 0.98, 0.00, 1.67),
               (60, 0.30, 0.84, 0.92, 0.00, 1.83)]


@pytest.mark.parametrize("D,s,p_cr,p_lo,p_by,th", POWER_TABLE)
def test_power_table_cells(D, s, p_cr, p_lo, p_by, th):
    p = R.power_figures(s, D, R.DELTA0)
    assert round(p["t"], 3) == {20: 2.093, 40: 2.023, 60: 2.001}[D]
    assert round(p["p_control_rule_at_1"], 2) == p_cr
    assert round(p["p_lo_at_1.30"], 2) == p_lo
    assert round(p["p_beyond_at_1.30"], 2) == p_by
    assert round(p["smallest_theta_pi_clause"], 2) == th


def test_power_statements():
    """§3.2: CONTROL-RULE has 80% power at θ_R = 1 for s ≤ 0.251 (D = 40) and s ≤ 0.31 (D = 60); BEYOND-* at
    1.30 needs s < 0.128 (D = 40) and 0.130 (D = 60), whatever the power."""
    for D, s80 in ((40, 0.251), (60, 0.31)):
        root = brentq(lambda s: R.power_figures(s, D, R.DELTA0)["p_control_rule_at_1"] - 0.8, 0.05, 0.6)
        assert round(root, 3 if D == 40 else 2) == s80
    for D, s_h2 in ((40, 0.128), (60, 0.130)):
        assert R.power_figures(s_h2 - 0.0005, D, R.DELTA0)["h2_reachable_at_1.30"]
        assert not R.power_figures(s_h2 + 0.0015, D, R.DELTA0)["h2_reachable_at_1.30"]


def test_thresholds_from_controls():
    jm, dc = _good()
    th = R.thresholds_from_controls(dc)
    assert round(th["delta"], 5) == 1.14039 and th["eps_M"] == 0.002 and th["f_H2"] == 0.05
    assert round(th["theta_gb1"], 3) == 2.842
    dc["c_D"]["band_dup"] = 0.153
    assert R.thresholds_from_controls(dc)["eps_M"] == pytest.approx(0.006)
    dc["c_D"]["ident_mean"] = -0.004
    assert R.thresholds_from_controls(dc)["eps_M"] == pytest.approx(0.008)
    dc["families"]["M"] = {"s": 0.30, "D": 40}
    th = R.thresholds_from_controls(dc)
    assert th["power_M"] == pytest.approx(R.power_figures(0.30, 40, R.DELTA0)["power_E"])
    assert th["power_M"] != th["power_E"]
    assert th["topup"] == {"fires": True, "stats_exceeding": ["M"], "run_families": ["em"]}
    dc["families"]["M"]["s"] = 0.22
    assert not R.thresholds_from_controls(dc)["topup"]["fires"]          # "exceeds 0.22"
    dc["families"]["R"] = {"s": 0.30, "D": 60}
    th = R.thresholds_from_controls(dc)
    assert th["topup"]["run_families"] == ["rt"]
    assert th["p_control_rule_at_1"] == pytest.approx(R.power_figures(0.30, 60, R.DELTA0)["p_control_rule_at_1"])


YB_PSEUDO = 0.15 * (1 + 0.2 * np.sin(np.arange(12)))


def _pseudo_f(x):
    return R.pseudo_band_calibration(np.exp(x)[:, None] * YB_PSEUDO[None, :], R.DELTA0)["f_H2"]


def test_pseudo_heavy_tails():
    """G-PSEUDO registered input (Cauchy log-costs). Measured: at D = 10, Cauchy log-costs (evenly spaced
    quantiles) give f_H2 = 0.20 and fail the gate; at D = 40, the registered draw count, they give 0.05 and
    pass it (the outliers inflate s, so only the two extremes fall outside the PI; over 60 random seeds the
    mean was 0.049 and none exceeded 0.10). A contaminated sample (a tight core and 15% of draws at ±1 in
    log cost) gives 0.15 at D = 40 and fails; that is the input fed to the verdict."""
    z10 = np.tan(np.pi * ((np.arange(10) + 0.5) / 10 - 0.5))
    assert _pseudo_f(0.3 * z10) > 0.10
    z40 = np.tan(np.pi * ((np.arange(40) + 0.5) / 40 - 0.5))
    assert _pseudo_f(0.3 * z40) <= 0.10
    x = 0.05 * np.random.default_rng(3).standard_normal(40)
    x[:3], x[3:6] = 1.0, -1.0
    f = _pseudo_f(x)
    assert f > 0.10
    gauss = 0.3 * sps.norm.ppf((np.arange(40) + 0.5) / 40)
    assert _pseudo_f(gauss) <= 0.10
    assert _case(R_=(1.3, 0.10), f_H2=f)["label"] == "INCONCLUSIVE-R (RULE-MISCALIBRATED)"


# =====================================================================================================
# §6.1 — gates on a baseline that passes all of them, each registered failing input changing one thing
# =====================================================================================================

FAMILY_SIZES = {"rt": 40, "emt": 40, "rtall": 5, "rthalf": 3, "rt2x": 3}
TARGETS = [10.0, 20.0, 30.0, 40.0]


def _rec(**kw):
    r = {"reachable": True, "matched": True, "pivot": False, "exhausted": False, "unreachable_layers": [],
         "redraw": 0}
    r.update(kw)
    return r


def _good():
    ppl_R = {arm: R.GB1_PPL_Q[gb1] for arm, gb1 in R.PRE_ARMS.items()}
    ppl_R["none"] = R.GB1_PPL_Q["none"]
    jm = {
        "env": {"transformers": "5.9.0", "torch": "2.12.0", "numpy": "2.4.6", "cuda": "13.0",
                "gpu_name": "NVIDIA L4", "instance_type": "g6.xlarge", "_attn_implementation": "eager"},
        "run": "R2", "mask": "native",
        "mask_probe": {"native": PROBES["causal"], "forced_masks": "on", "forced": {m: PROBES[m] for m in R.MODES}},
        "ppl_R": ppl_R, "pre1_reference": None,
        "ematch": [None] * 3 + [[{"e_target": 0.02, "e_realized": 0.0205} for _ in range(4)] for _ in range(37)],
        "construction": {f: [_rec() for _ in range(n)] for f, n in FAMILY_SIZES.items()},
        "verify_C": {f: [{"N": list(TARGETS), "targets": list(TARGETS)} for _ in range(n)]
                     for f, n in FAMILY_SIZES.items()},
        "gen_D": {f: [{"N_sum": 100.0, "target_sum": 100.0} for _ in range(n)] for f, n in FAMILY_SIZES.items()},
        "readout_validity": {s: {"fit": 0.001, "positive_control": 6.2, "identity": 0.0} for s in "EMRT"},
    }
    dc = {
        "c_D": {"band": 0.15, "band_dup": 0.15, "ident_mean": 0.0, "topenergy32": 6.2, "topsigma32_mean": 2.0},
        "c_D_draws": {"haar": [0.006, 0.009, 0.007, 0.005, 0.008], "haar_mean": [0.004, 0.005, 0.003],
                      "em": [0.45] * 40, "em_mean": [0.30] * 40, "rt": [0.15] * 40, "emt": [0.20] * 40,
                      "rthalf": [0.08] * 3, "rt2x": [0.30] * 3, "rtall": [0.16] * 5, "spec": [0.40] * 5},
        "Y_band_mean_D": list(0.10 * (1 + 0.1 * np.sin(np.arange(66)))),
        "id_max": {"ident_zero": 0.0, "ident_mean": 0.0}, "id_windows": "0-97",
        "boot": {s: {"n_degenerate": 0, "n_boot": 10000} for s in "EMRT"},
        "f_H2": 0.05, "noleak": {"leaks": []}, "planned_draws": {s: 40 for s in "EMRT"},
        "families": {s: {"s": 0.10, "D": 40} for s in "EMRT"},
    }
    return jm, dc


def test_good_inputs_pass_every_gate():
    g = R.gates(*_good())
    assert [k for k, v in g.items() if not v["pass"]] == []
    assert not g["G-MDENOM"]["fires"] and g["G-DOSE"]["n_steps"] == 2
    assert not g["G-MASK"]["mask_unknown"] and g["G-MASK"]["value"]["native"] == "causal"
    assert all(g[f"G-DRAWS:{s}"]["value"]["valid"] == 40 for s in "EMRT")


def _fails(edit, gate_id, label=None):
    jm, dc = _good()
    edit(jm, dc)
    g = R.gates(jm, dc)[gate_id]
    assert not g["pass"], (gate_id, g)
    if label is not None:
        assert g["label"] == label
    return g


def test_env_gate_rejects_sdpa():
    _fails(lambda jm, dc: jm["env"].update(_attn_implementation="sdpa"), "G-ENV", "RUN-DOES-NOT-COUNT")


def test_env_gate_rejects_missing_version():
    g = _fails(lambda jm, dc: jm["env"].pop("transformers"), "G-ENV")
    assert g["value"]["missing"] == ["transformers"]


PRE0 = {"none": [1322.76, 1644.10, 1500.0, 1400.0, 1450.0, 1600.0],
        "band": [1387.11, 1719.03, 1560.0, 1460.0, 1510.0, 1660.0],
        "em_s0": [1523.36, 1880.47, 1700.0, 1600.0, 1650.0, 1800.0]}


def _pre0(jm, shift=False):
    run = {a: list(v) for a, v in PRE0.items()}
    if shift:
        run["none"] = run["none"][1:] + [1500.0]
    jm["pre0"] = {"run": run, "gb1": {"none": PRE0["none"], "core32": PRE0["band"],
                                      "energymatched32_s0": PRE0["em_s0"]}}


def test_pre0_rejects_shifted_window():
    jm, dc = _good()
    _pre0(jm)
    assert R.gates(jm, dc)["G-PRE0"]["pass"]
    _fails(lambda jm, dc: _pre0(jm, shift=True), "G-PRE0", "pivot P1-smoke")


def test_pre1_rejects_sdpa_value_36_8():
    _fails(lambda jm, dc: jm["ppl_R"].update(none=36.8), "G-PRE1", "pivot P1")
    _fails(lambda jm, dc: jm["ppl_R"].update(none=22.10), "G-PRE1")         # inside [18, 28], off by 0.32%
    jm, dc = _good()
    jm["run"], jm["mask"] = "R3", "causal"
    jm["ppl_R"]["none"], jm["pre1_reference"] = 22.5, 22.5
    for arm in R.PRE_ARMS:                           # keep G-PRE2–4 at GB-1's ΔNLL on the new none
        jm["ppl_R"][arm] = 22.5 * R.GB1_PPL_Q[R.PRE_ARMS[arm]] / R.GB1_PPL_Q["none"]
    assert R.gates(jm, dc)["G-PRE1"]["pass"]
    jm["pre1_reference"] = 22.0
    assert not R.gates(jm, dc)["G-PRE1"]["pass"]


# the bands printed in §6.1 for G-PRE2–4
PRE_BANDS = {"band": (0.149161, 0.155250), "em_s0": (0.444819, 0.462975), "em_s1": (0.321908, 0.335048),
             "em_s2": (0.532018, 0.553733), "spec_s0": (0.435909, 0.453702), "spec_s1": (0.263437, 0.274189),
             "spec_s2": (0.390815, 0.406767), "topenergy32": (6.076418, 6.324435),
             "topsigma32": (4.015479, 4.179377), "anti32": (3.997623, 4.160791)}


def _set_dnll(jm, arm, d):
    jm["ppl_R"][arm] = jm["ppl_R"]["none"] * math.exp(d)


def test_pre2_rejects_off_band():
    g = R.gates(*_good())
    rows = {**g["G-PRE2"]["value"], **g["G-PRE3"]["value"], **g["G-PRE4"]["value"]}
    for arm, (lo, hi) in PRE_BANDS.items():
        r = rows[arm]
        assert (r["dnll_gb1"] - r["tol"], r["dnll_gb1"] + r["tol"]) == pytest.approx((lo, hi), abs=1e-6)
    _fails(lambda jm, dc: _set_dnll(jm, "band", 0.1555), "G-PRE2", "RUN-DOES-NOT-COUNT (G-PRE2: band)")
    jm, dc = _good()
    _set_dnll(jm, "band", 0.1552)
    assert R.gates(jm, dc)["G-PRE2"]["pass"]


def test_pre3_rejects():
    _fails(lambda jm, dc: _set_dnll(jm, "em_s0", 0.4635), "G-PRE3", "RUN-DOES-NOT-COUNT (G-PRE3: em_s0)")


def test_pre4_one_arm_off():
    g = _fails(lambda jm, dc: _set_dnll(jm, "spec_s1", 0.30), "G-PRE4", "RUN-DOES-NOT-COUNT (G-PRE4: spec_s1)")
    assert not g["value"]["spec_s1"]["pass"]
    jm, dc = _good()
    _set_dnll(jm, "haar_s0", 0.05)                   # recorded, not gating
    g = R.gates(jm, dc)["G-PRE4"]
    assert g["pass"] and not g["value"]["haar_s0"]["pass"] and g["value"]["haar_s0"]["recorded_only"]
    jm, dc = _good()
    _set_dnll(jm, "spec_s1", 0.30)
    jm["run"], jm["mask"], jm["pre1_reference"] = "R3", "causal", R.GB1_PPL_Q["none"]
    g = R.gates(jm, dc)["G-PRE4"]
    assert not g["pass"] and not g["gating"]           # R3: recorded, not gating
    assert _verdict_on_good(jm, dc)["label"] == "CONTROL-RULE"


def test_identity_gate_rejects_leaky_hook():
    _fails(lambda jm, dc: dc["id_max"].update(ident_zero=2e-3), "G-ID1", "RUN-DOES-NOT-COUNT")
    _fails(lambda jm, dc: dc["id_max"].update(ident_mean=2e-3), "G-ID3", "RUN-DOES-NOT-COUNT")
    jm, dc = _good()
    dc["id_windows"] = "98-195"
    assert R.gates(jm, dc)["G-ID1"]["value"]["windows"] == "98-195"


def test_id2_nondeterministic():
    g = _fails(lambda jm, dc: dc["c_D"].update(band_dup=0.15 * 1.03), "G-ID2", "RUN-DOES-NOT-COUNT")
    assert g["value"] == pytest.approx(math.log(1.03))


def test_scale_band_equals_haar():
    _fails(lambda jm, dc: dc["c_D"].update(band=0.009, band_dup=0.009), "G-SCALE", "RUN-DOES-NOT-COUNT")


def test_scale_haar_zero():
    g = _fails(lambda jm, dc: dc["c_D_draws"].update(haar=[0.0] * 5), "G-SCALE")
    assert g["value"] is None


def test_pos_control_dead():
    _fails(lambda jm, dc: dc["c_D"].update(topenergy32=0.05), "G-POS", "RUN-DOES-NOT-COUNT")


def test_pos_control_nan():
    _fails(lambda jm, dc: dc["c_D"].update(topenergy32=float("nan")), "G-POS", "RUN-DOES-NOT-COUNT")
    _fails(lambda jm, dc: dc["c_D"].update(topenergy32=25.0), "G-POS")       # above the 20 nats ceiling


def test_bracket_family_fails():
    g = _fails(lambda jm, dc: dc["c_D_draws"].update(rt=[7.0] * 5 + [0.15] * 35), "G-BRACKET:R",
               "RT-DOES-NOT-COUNT")
    assert g["value"]["flagged"] == [0, 1, 2, 3, 4]
    jm, dc = _good()
    dc["c_D_draws"]["rt"] = [7.0] * 4 + [0.15] * 36                          # exactly 10%: passes
    assert R.gates(jm, dc)["G-BRACKET:R"]["pass"]


def test_bracket_flag_keeps_draw():
    jm, dc = _good()
    dc["c_D_draws"]["em"] = [0.001] + [0.45] * 39                          # below the Haar floor
    g = R.gates(jm, dc)
    assert g["G-BRACKET:E"]["pass"] and g["G-BRACKET:E"]["value"]["flagged"] == [0]
    assert g["G-DRAWS:E"]["value"]["valid"] == 40


def test_bracket_mean_uses_mean_refs():
    """Mean-ablation draws are bracketed by haar_mean and topsigma32_mean: 3.0 is inside the zero-ablation
    references (0.009, 6.2) but above topsigma32_mean = 2.0. With G-MDENOM firing only the ceiling applies."""
    g = _fails(lambda jm, dc: dc["c_D_draws"].update(em_mean=[3.0] * 5 + [0.3] * 35), "G-BRACKET:M",
               "M-DOES-NOT-COUNT")
    assert g["value"]["references"] == [0.005, 2.0]
    jm, dc = _good()
    dc["c_D_draws"]["em_mean"] = [0.001] * 5 + [0.3] * 35                  # below the haar_mean floor
    assert not R.gates(jm, dc)["G-BRACKET:M"]["pass"]
    dc["Y_band_mean_D"] = [0.0] * 66                                        # G-MDENOM fires
    g = R.gates(jm, dc)
    assert g["G-MDENOM"]["fires"] and g["G-BRACKET:M"]["pass"]


def test_ematch_draw_rejected_not_raised():
    jm, dc = _good()
    jm["ematch"][5][2] = {"e_target": 0.02, "e_realized": 0.025}            # 25% off at a layer above k/d
    jm["ematch"][6][1] = {"e_target": 0.001, "e_realized": 0.01}            # below k/d: exempt
    g = R.gates(jm, dc)
    assert not g["G-EMATCH"]["pass"] and g["G-EMATCH"]["value"] == [{"slot": 5, "layers": [2]}]
    assert g["G-DRAWS:E"]["value"]["valid"] == 39 and g["G-DRAWS:M"]["value"]["valid"] == 39


def test_match_rejects_2pct():
    jm, dc = _good()
    jm["verify_C"]["rt"][0] = {"N": [10.2, 20.4, 30.6, 40.8], "targets": TARGETS}      # sum 2% off
    jm["verify_C"]["rt"][1] = {"N": [10.09, 20.18, 30.27, 40.36], "targets": TARGETS}  # 0.9% off: passes
    jm["verify_C"]["rt"][2] = {"N": [10.0, 20.0, 28.2, 41.8], "targets": TARGETS}     # heavy layer 6% off
    jm["verify_C"]["rt"][3] = {"N": [0.06, 20.0, 30.0, 49.97], "targets": [0.04, 20.0, 30.0, 49.99]}  # light
    g = R.gates(jm, dc)
    assert g["G-MATCH:rt"]["value"] == [0, 2]
    assert g["G-DRAWS:R"]["value"]["valid"] == 38


def test_gen_rejects_overfit_match():
    jm, dc = _good()
    jm["gen_D"]["rt"][0] = {"N_sum": 106.0, "target_sum": 100.0}
    jm["gen_D"]["rt"][1] = {"N_sum": 104.0, "target_sum": 100.0}
    g = R.gates(jm, dc)
    assert g["G-GEN:rt"]["value"] == [0] and g["G-DRAWS:R"]["value"]["valid"] == 39


def test_draw_count_gate():
    def bad_match(n):
        def edit(jm, dc):
            for i in range(n):
                jm["verify_C"]["rt"][i] = {"N": [12.0, 20.0, 30.0, 40.0], "targets": TARGETS}
        return edit
    _fails(bad_match(9), "G-DRAWS:R", "RT-DOES-NOT-COUNT")                  # 31 of 40
    jm, dc = _good()
    bad_match(8)(jm, dc)
    assert R.gates(jm, dc)["G-DRAWS:R"]["pass"]                             # 32 of 40
    jm, dc = _good()
    dc["planned_draws"]["R"] = 60
    jm["construction"]["rt"] = [_rec() for _ in range(60)]
    jm["verify_C"]["rt"] = [{"N": TARGETS, "targets": TARGETS} for _ in range(60)]
    jm["gen_D"]["rt"] = [{"N_sum": 100.0, "target_sum": 100.0} for _ in range(60)]
    dc["c_D_draws"]["rt"] = [0.15] * 60
    bad_match(12)(jm, dc)
    assert R.gates(jm, dc)["G-DRAWS:R"]["pass"]                             # 48 of 60
    bad_match(13)(jm, dc)
    assert not R.gates(jm, dc)["G-DRAWS:R"]["pass"]                         # 47 of 60


def _unreachable(jm, fam):
    jm["construction"][fam] = [_rec(reachable=False, matched=False, unreachable_layers=["sum"])
                               for _ in jm["construction"][fam]]


def test_dose_flat():
    g = _fails(lambda jm, dc: dc["c_D_draws"].update(rthalf=[0.15] * 3), "G-DOSE", "RT-DOES-NOT-COUNT")
    assert g["value"]["steps"]["rt/rthalf"] == pytest.approx(1.0)


def test_dose_runaway():
    _fails(lambda jm, dc: dc["c_D_draws"].update(rt2x=[0.75] * 3), "G-DOSE")                 # step 5
    _fails(lambda jm, dc: dc.update(c_D={**dc["c_D"], "topenergy32": 0.5}) or
           dc["c_D_draws"].update(rt2x=[0.30, 0.30, 0.55]), "G-DOSE")                         # rt2x ≥ topenergy32


def test_dose_single_family_checked():
    def edit(jm, dc):
        _unreachable(jm, "rthalf")
        dc["c_D_draws"]["rt2x"] = [0.75] * 3
    g = _fails(edit, "G-DOSE")
    assert list(g["value"]["steps"]) == ["rt2x/rt"] and g["n_steps"] == 1
    jm, dc = _good()
    _unreachable(jm, "rthalf")
    g = R.gates(jm, dc)["G-DOSE"]
    assert g["pass"] and g["n_steps"] == 1


def test_dose_none_reachable_is_note():
    jm, dc = _good()
    _unreachable(jm, "rthalf")
    _unreachable(jm, "rt2x")
    g = R.gates(jm, dc)["G-DOSE"]
    assert g["pass"] and g["n_steps"] == 0
    v = _verdict_on_good(jm, dc)
    assert v["label"] == "CONTROL-RULE" and "DOSE-DOES-NOT-COUNT" in v["notes"]


def test_meanpos_dead():
    _fails(lambda jm, dc: dc["c_D"].update(topsigma32_mean=0.05), "G-MEANPOS", "M-DOES-NOT-COUNT")
    _fails(lambda jm, dc: dc["c_D"].update(topsigma32_mean=float("nan")), "G-MEANPOS")


def test_meanpos_haar_negative():
    """A negative haar_mean cost must not lower the floor below the absolute 0.1 nats/token."""
    def edit(jm, dc):
        dc["c_D_draws"]["haar_mean"] = [-0.01, -0.02, -0.005]
        dc["c_D"]["topsigma32_mean"] = 0.08
    g = _fails(edit, "G-MEANPOS")
    assert g["value"]["floor"] == 0.1
    jm, dc = _good()
    dc["c_D_draws"]["haar_mean"] = [-0.01, -0.02, -0.005]
    dc["c_D"]["topsigma32_mean"] = 0.12
    assert R.gates(jm, dc)["G-MEANPOS"]["pass"]


def test_meanrow_zero_denominator():
    jm, dc = _good()
    dc["Y_band_mean_D"] = [0.0] * 66
    g = R.gates(jm, dc)
    assert g["G-MDENOM"]["fires"] and g["G-MDENOM"]["pass"]
    assert g["G-BOOT:M"]["pass"] and g["G-BOOT:M"]["value"] is None
    dc["Y_band_mean_D"] = list(0.003 + 0.0005 * np.sin(np.arange(66)))      # above 0 but below haar_mean
    assert R.gates(jm, dc)["G-MDENOM"]["fires"]
    dc["Y_band_mean_D"] = list(0.005 + 0.01 * np.sin(np.arange(66)))        # above 0, straddles haar_mean
    g = R.gates(jm, dc)["G-MDENOM"]
    assert g["value"]["ci"][0] > 0 and g["value"]["ci"][0] < 0.005 < g["value"]["ci"][1]
    assert not g["fires"]


def test_delta_m_equivalence_margin():
    """§6.2, §6.4 step M under G-MDENOM: H0-MEAN iff hi(I_Δ) ≤ ε_M; lo(I_Δ) > ε_M continues; else INCONCLUSIVE-M."""
    z = _z(40)
    assert _case(mdenom=True, em_mean_c=0.0005 + 0.0001 * z)["label"] == "H0-MEAN"
    assert _case(mdenom=True, em_mean_c=0.002 + 0.001 * z)["notes"] == ["INCONCLUSIVE-M", "T-QUANTITY"]
    v = _case(mdenom=True, em_mean_c=0.3 + 0.01 * z)
    assert v["label"] == "CONTROL-RULE" and v["notes"] == ["T-QUANTITY"]
    # a larger identity-pair reading widens ε_M: the same Δ ≈ 0.002 now sits inside the margin
    v = _case(mdenom=True, em_mean_c=0.002 + 0.0001 * z, edit=lambda jm, dc: dc["c_D"].update(ident_mean=0.0012))
    assert v["label"] == "H0-MEAN" and v["numbers"]["eps_M"] == pytest.approx(0.0024)


def test_readout_missing_identity():
    _fails(lambda jm, dc: jm["readout_validity"]["R"].update(identity=None), "G-READOUT:R", "RT-DOES-NOT-COUNT")
    jm, dc = _good()
    jm["readout_validity"]["R"]["identity"] = None
    assert _verdict_on_good(jm, dc)["label"] == "RT-DOES-NOT-COUNT (G-READOUT:R)"


def test_crossjob_drift():
    jm, dc = _good()
    assert "G-CROSSJOB" not in R.gates(jm, dc)
    dc["c_D_band_R4"] = 0.15 * 1.004
    assert R.gates(jm, dc)["G-CROSSJOB"]["pass"]
    _fails(lambda jm, dc: dc.update(c_D_band_R4=0.15 * 1.006), "G-CROSSJOB", "R4 draws do not count")


def test_noleak_marks_every_label():
    jm, dc = _good()
    dc["noleak"]["leaks"] = ["metrics.json: c_band over windows 0-195"]
    assert _verdict_on_good(jm, dc)["notes"][-1] == "SEAL-BROKEN"
    jm["env"]["_attn_implementation"] = "sdpa"
    v = _verdict_on_good(jm, dc)
    assert v["label"] == "RUN-DOES-NOT-COUNT (G-ENV)" and v["notes"] == ["SEAL-BROKEN"]


# =====================================================================================================
# §6.4 — every verdict branch, on synthetic per-window NLL arrays
# =====================================================================================================

W_F = 12
NTOK = np.full(W_F, 511)
NLL_NONE = 2.6 * 511 + 10 * np.cos(np.arange(W_F))
YB = 0.15 * (1 + 0.2 * np.sin(np.arange(W_F)))          # band, per-window ΔNLL per token
YBM = 0.10 * (1 + 0.2 * np.cos(np.arange(W_F)))         # band_mean


def _z(D):
    z = sps.norm.ppf((np.arange(D) + 0.5) / D)
    return (z - z.mean()) / z.std(ddof=1)


def _nll(Y):
    """Per-window NLL sums of an arm whose per-window ΔNLL per token is Y."""
    return NLL_NONE + np.asarray(Y, dtype=np.float64) * NTOK


def _Y(nll):
    return G.dnll_per_window(nll, NLL_NONE, NTOK)


def _family(theta, s, Yb, D=40):
    """Draw d costs θ·exp(s·z_d) times the band in every window; z_d are standardized normal quantiles, so the
    planted geometric-mean ratio is θ and the planted sd(ln r) is s exactly."""
    return [_nll(theta * math.exp(s * z) * Yb) for z in _z(D)]


def _stats(F, mdenom):
    Yb, Ybm = _Y(F["band"]), _Y(F["band_mean"])
    Yf = {k: np.array([_Y(x) for x in F[k]]) for k in ("em", "em_mean", "rt", "emt")}
    st = {"E": R.decision_interval(Yf["em"], Yb), "R": R.decision_interval(Yf["rt"], Yb),
          "T": R.decision_interval(Yf["emt"], Yb), "x_band": math.log(Yb.mean()),
          "x_rt": list(np.log(Yf["rt"].mean(axis=1))), "M": None, "Delta_M": None}
    if mdenom:
        m, lo, hi = R.difference_interval(Yf["em_mean"].mean(axis=1), float(Ybm.mean()))
        st["Delta_M"] = {"mean": m, "lo": lo, "hi": hi}
    else:
        st["M"] = R.decision_interval(Yf["em_mean"], Ybm)
    return st


def _case(R_=(1.0, 0.10), E=(2.9, 0.10), M=(2.9, 0.10), T=(2.9, 0.10), s_D=None, mdenom=False,
          em_mean_c=None, f_H2=0.05, edit=None):
    """Planted per-window NLL on F → N1 statistics; gates on the passing baseline (edited); thresholds from
    the discovery spreads s_D (default: the planted F spreads); then the §6.4 verdict."""
    F = {"band": _nll(YB), "band_mean": _nll(0 * YBM if mdenom else YBM), "em": _family(*E, YB),
         "rt": _family(*R_, YB), "emt": _family(*T, YB),
         "em_mean": [_nll(np.full(W_F, c)) for c in em_mean_c] if mdenom else _family(*M, YBM)}
    jm, dc = _good()
    if mdenom:
        dc["Y_band_mean_D"] = [0.0] * 66
    dc["f_H2"] = f_H2
    sD = {"E": E[1], "M": M[1], "R": R_[1], "T": T[1], **(s_D or {})}
    dc["families"] = {k: {"s": v, "D": 40} for k, v in sD.items()}
    if edit is not None:
        edit(jm, dc)
    return R.verdict(_stats(F, mdenom), R.gates(jm, dc), R.thresholds_from_controls(dc))


def _verdict_on_good(jm, dc):
    """The verdict for an edited baseline on the CONTROL-RULE planted data."""
    F = {"band": _nll(YB), "band_mean": _nll(YBM), "em": _family(2.9, 0.10, YB), "rt": _family(1.0, 0.10, YB),
         "emt": _family(1.0, 0.10, YB), "em_mean": _family(2.9, 0.10, YBM)}
    return R.verdict(_stats(F, R.gates(jm, dc)["G-MDENOM"]["fires"]), R.gates(jm, dc),
                     R.thresholds_from_controls(dc))


@pytest.mark.parametrize("theta,s,label,notes", [
    (1.0, 0.10, "CONTROL-RULE", ["T-QUANTITY"]),
    (1.0, 0.25, "CONTROL-RULE", ["T-QUANTITY"]),
    (1.3, 0.10, "BEYOND-CHEAPER", []),
    (1.3, 0.25, "UNDERPOWERED-R (H2-UNREACHABLE)", []),
    (2.9, 0.10, "BEYOND-CHEAPER", []),
    (2.9, 0.25, "BEYOND-CHEAPER", ["LOW-POWER"]),
])
def test_verdict_branches(theta, s, label, notes):
    """§6.3, §5: planted ratios 1.0, 1.3 and 2.9 at draw spreads 0.10 and 0.25 (the EM contrast at 2.9)."""
    v = _case(R_=(theta, s), E=(2.9, s), M=(2.9, s), T=(2.9, s))
    assert (v["label"], v["notes"]) == (label, notes)


def _edit_env(jm, dc):
    jm["env"]["_attn_implementation"] = "sdpa"


def _edit_em_draws(jm, dc):
    for slot in range(3, 12):
        jm["ematch"][slot][0] = {"e_target": 0.02, "e_realized": 0.03}


def _edit_unreachable(*fams):
    def edit(jm, dc):
        for f in fams:
            _unreachable(jm, f)
    return edit


INV = 1 / 1.3
VERDICT_CASES = {
    "RUN-DOES-NOT-COUNT": (dict(edit=_edit_env), "RUN-DOES-NOT-COUNT (G-ENV)", []),
    "E-DOES-NOT-COUNT": (dict(edit=_edit_em_draws), "E-DOES-NOT-COUNT (G-DRAWS:E)", []),
    "H0-WINDOWS": (dict(E=(1.0, 0.10)), "H0-WINDOWS", []),
    "INCONCLUSIVE-E": (dict(E=(1.14, 0.10)), "INCONCLUSIVE-E", []),
    "UNDERPOWERED-E": (dict(E=(1.14, 0.10), s_D={"E": 1.0}), "UNDERPOWERED-E", []),
    "M-DOES-NOT-COUNT": (dict(edit=lambda jm, dc: dc["c_D"].update(topsigma32_mean=25.0)), "CONTROL-RULE",
                         ["M-DOES-NOT-COUNT (G-MEANPOS)", "T-QUANTITY"]),
    "H0-MEAN (theta_M)": (dict(M=(1.0, 0.10)), "H0-MEAN", []),
    "H0-MEAN (Delta_M)": (dict(mdenom=True, em_mean_c=0.0005 + 0.0001 * _z(40)), "H0-MEAN", []),
    "INCONCLUSIVE-M (Delta_M)": (dict(mdenom=True, em_mean_c=0.002 + 0.001 * _z(40)), "CONTROL-RULE",
                                 ["INCONCLUSIVE-M", "T-QUANTITY"]),
    "INCONCLUSIVE-M (theta_M)": (dict(M=(1.14, 0.10)), "CONTROL-RULE", ["INCONCLUSIVE-M", "T-QUANTITY"]),
    "UNDERPOWERED-M": (dict(M=(1.14, 0.10), s_D={"M": 1.0}), "CONTROL-RULE", ["UNDERPOWERED-M", "T-QUANTITY"]),
    "RT-DOES-NOT-COUNT": (dict(edit=_edit_unreachable("rt")), "RT-DOES-NOT-COUNT (G-REACH:rt, G-DRAWS:R)", []),
    "DOSE-DOES-NOT-COUNT": (dict(edit=_edit_unreachable("rthalf", "rt2x")), "CONTROL-RULE",
                            ["DOSE-DOES-NOT-COUNT", "T-QUANTITY"]),
    "CONTROL-RULE": (dict(), "CONTROL-RULE", ["T-QUANTITY"]),
    "LOW-POWER": (dict(s_D={"R": 0.30}), "CONTROL-RULE", ["LOW-POWER", "T-QUANTITY"]),
    "INCONCLUSIVE-R (RULE-MISCALIBRATED)": (dict(R_=(1.3, 0.10), f_H2=0.2), "INCONCLUSIVE-R (RULE-MISCALIBRATED)",
                                            []),
    "BEYOND-CHEAPER": (dict(R_=(1.3, 0.10)), "BEYOND-CHEAPER", []),
    "BEYOND-COSTLIER": (dict(R_=(INV, 0.10)), "BEYOND-COSTLIER", []),
    "UNDERPOWERED-R (H2-UNREACHABLE)": (dict(R_=(1.3, 0.25)), "UNDERPOWERED-R (H2-UNREACHABLE)", []),
    "BEYOND-WITHIN-SPREAD": (dict(R_=(1.3, 0.25), s_D={"R": 0.05}), "BEYOND-WITHIN-SPREAD", []),
    "UNDERPOWERED-R (H1-POWER)": (dict(R_=(1.14, 0.10), s_D={"R": 0.30}), "UNDERPOWERED-R (H1-POWER)", []),
    "INCONCLUSIVE-R": (dict(R_=(1.14, 0.10)), "INCONCLUSIVE-R", []),
    "T-DOES-NOT-COUNT": (dict(edit=_edit_unreachable("emt")), "CONTROL-RULE",
                         ["T-DOES-NOT-COUNT (G-REACH:emt, G-DRAWS:T)"]),
    "T-QUANTITY": (dict(T=(2.9, 0.10)), "CONTROL-RULE", ["T-QUANTITY"]),
    "T-CHEAPER": (dict(T=(1 / 2.9, 0.10)), "CONTROL-RULE", ["T-CHEAPER"]),
    "T-POSITIONS-OR-SCHEME": (dict(T=(1.0, 0.10)), "CONTROL-RULE", ["T-POSITIONS-OR-SCHEME"]),
    "T-INCONCLUSIVE": (dict(T=(1.14, 0.10)), "CONTROL-RULE", ["T-INCONCLUSIVE"]),
    "T-UNDERPOWERED": (dict(T=(1.14, 0.10), s_D={"T": 1.0}), "CONTROL-RULE", ["T-UNDERPOWERED"]),
    "MASK-UNKNOWN": (dict(edit=lambda jm, dc: _mask(jm, "bidir", None)), "CONTROL-RULE",
                     ["MASK-UNKNOWN", "T-QUANTITY"]),
    "SEAL-BROKEN": (dict(edit=lambda jm, dc: dc["noleak"]["leaks"].append("job.log: mean over 98-195")),
                    "CONTROL-RULE", ["T-QUANTITY", "SEAL-BROKEN"]),
}


@pytest.mark.parametrize("name", list(VERDICT_CASES))
def test_verdict_every_label(name):
    kw, label, notes = VERDICT_CASES[name]
    v = _case(**kw)
    assert (v["label"], v["notes"]) == (label, notes)
    assert v["numbers"]["stats"]["E"]["theta"] is not None                 # every statistic is reported


def _direct(**over):
    """The verdict on the passing baseline with statistics given directly (for the boundaries of §6.4)."""
    jm, dc = _good()
    if "Delta_M" in over:
        dc["Y_band_mean_D"] = [0.0] * 66
    d = R.DELTA0
    st = {"E": {"theta": 2.9, "lo": 2.7, "hi": 3.1}, "M": {"theta": 2.9, "lo": 2.7, "hi": 3.1}, "Delta_M": None,
          "R": {"theta": 1.0, "lo": 0.95, "hi": 1.05}, "T": {"theta": 2.9, "lo": 2.7, "hi": 3.1},
          "x_band": math.log(0.15), "x_rt": list(math.log(0.15) + 0.1 * _z(40))}
    st.update(over)
    if "Delta_M" in over:
        st["M"] = None
    return R.verdict(st, R.gates(jm, dc), R.thresholds_from_controls(dc)), d


def test_verdict_inequality_boundaries():
    """§6.4's inequalities exactly: hi(I_E) < δ and lo(I_E) > δ are strict; hi(I_Δ) ≤ ε_M is not; the
    CONTROL-RULE and T-POSITIONS-OR-SCHEME bounds 1/δ ≤ lo and hi ≤ δ are inclusive."""
    d = R.DELTA0
    assert _direct(E={"theta": 1.0, "lo": 0.9, "hi": d})[0]["label"] == "INCONCLUSIVE-E"
    assert _direct(E={"theta": 1.3, "lo": d, "hi": 1.5})[0]["label"] == "INCONCLUSIVE-E"
    assert _direct(Delta_M={"mean": 0.001, "lo": 0.0, "hi": 0.002})[0]["label"] == "H0-MEAN"
    v = _direct(Delta_M={"mean": 0.003, "lo": 0.002, "hi": 0.004})[0]
    assert v["label"] == "CONTROL-RULE" and v["notes"][0] == "INCONCLUSIVE-M"
    assert _direct(R={"theta": 1.0, "lo": 1 / d, "hi": d})[0]["label"] == "CONTROL-RULE"
    v = _direct(T={"theta": 1.0, "lo": 1 / d, "hi": d})[0]
    assert v["notes"] == ["T-POSITIONS-OR-SCHEME"]


def test_verdict_reports_stop_step_and_numbers():
    v = _case(E=(1.0, 0.10))
    assert v["numbers"]["stopped_at"] == "E" and v["numbers"]["PI"] is not None
    assert _case()["numbers"]["stopped_at"] is None


# =====================================================================================================
# §10.1 samples, provenance, N4 stamp
# =====================================================================================================

def test_sample_record_right_and_wrong():
    window = {"index": 32, "text_seen": "<image>×256 <bos> a b c d\\n", "n_text": 5,
              "target_ids": [10, 11, 12, 13], "target_strs": ["▁a", "▁b", "▁c", "▁d"]}
    lp_none = np.log([0.6, 0.2, 0.5, 0.1])
    lp_arm = np.log([0.5, 0.1, 0.4, 0.05])
    top_none = ([10, 99, 12, 98], ["▁a", "▁x", "▁c", "▁y"], [0.6, 0.3, 0.5, 0.4])
    top_arm = ([10, 99, 77, 98], ["▁a", "▁x", "▁z", "▁y"], [0.5, 0.3, 0.45, 0.4])
    rec = R.sample_record(window, lp_none, lp_arm, top_none, top_arm)
    assert rec["right_example"]["pos"] == 0 and rec["right_example"]["top1_arm_is_target"]
    assert rec["wrong_example"]["pos"] == 1 and not rec["wrong_example"]["top1_arm_is_target"]
    assert rec["n_right"] == {"none": 2, "arm": 1} and rec["ntok"] == 4 and len(rec["positions"]) == 4
    assert rec["nll_sum_none"] == pytest.approx(-lp_none.sum())
    assert rec["Y"] == pytest.approx((lp_none.sum() - lp_arm.sum()) / 4)
    assert rec["positions"][2]["top1_none"] == "▁c" and rec["positions"][2]["logp_arm"] == pytest.approx(math.log(0.4))
    json.dumps(rec)
    with pytest.raises(ValueError):
        R.sample_record({**window, "n_text": 6}, lp_none, lp_arm, top_none, top_arm)
    with pytest.raises(ValueError):
        R.sample_record(window, -lp_none, lp_arm, top_none, top_arm)


def test_provenance_refuses_other_program(tmp_path, monkeypatch):
    p = tmp_path / "realizednull_provenance.json"
    monkeypatch.setattr(R, "PROVENANCE_FILE", p)
    p.write_text(json.dumps({"program": "gutenberg", "commit_sha": "x"}))
    with pytest.raises(ValueError):
        R.provenance()
    p.write_text(json.dumps({"program": "realizednull", "commit_sha": "x"}))
    assert R.provenance()["commit_sha"] == "x"


def test_stamp_script_writes_realizednull(tmp_path, monkeypatch):
    out = tmp_path / "realizednull_provenance.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "realizednull_stamp.py"), "--out", str(out)],
                   check=True, capture_output=True, cwd=ROOT)
    monkeypatch.setattr(R, "PROVENANCE_FILE", out)
    d = R.provenance()
    assert d["source"] == "scripts/realizednull_stamp.py" and len(d["commit_sha"]) == 40
