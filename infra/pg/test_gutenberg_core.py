"""GUTENBERG weight-free dry-run gate (research-spec #13): pure logic on tiny tensors.

Every frozen-rule BRANCH is constructed and reached (MENDEL lesson: a rule branch that cannot
fire is a defect found at freeze, not at unblind). Aggregations are executed on synthetic cells
(MENDEL lesson: test the aggregation, not just the helpers). Run: pytest infra/pg/test_gutenberg_core.py -q
"""
import json
import struct

import numpy as np
import pytest

from infra.pg import gutenberg_core as G


# ---------------- GB-A ----------------

def test_jaccard_identity_orthogonal_partial():
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((8, 4)))
    U, V = Q[:, :2], Q[:, 2:]
    assert G.subspace_jaccard(U, U) == pytest.approx(1.0)
    assert G.subspace_jaccard(U, V) == pytest.approx(0.0, abs=1e-12)
    # k=2 in R^4 sharing exactly one direction: t=1 -> J = 1/(4-1) = 1/3
    E = np.eye(4)
    A, B = E[:, [0, 1]], E[:, [1, 2]]
    assert G.subspace_jaccard(A, B) == pytest.approx(1.0 / 3.0)


def test_jaccard_rotation_invariance():
    rng = np.random.default_rng(1)
    U, _ = np.linalg.qr(rng.standard_normal((16, 3)))
    R, _ = np.linalg.qr(rng.standard_normal((16, 16)))
    V, _ = np.linalg.qr(rng.standard_normal((16, 3)))
    assert G.subspace_jaccard(R @ U, R @ V) == pytest.approx(G.subspace_jaccard(U, V))


def test_rotation_null_scale():
    med, per = G.rotation_null_median_jaccard(n=64, k=4, n_layers=6, n_draws=50, seed=7)
    assert med.shape == (50,) and per.shape == (300,)
    assert 0 < np.median(med) < 0.15          # k/n-scale, far below 0.5
    assert per.max() < 0.5


def test_gba_verdict_all_branches_reachable():
    null = np.full(100, 0.05)
    assert G.gba_verdict([0.7] * 5, null)["verdict"] == "SAME-OBJECT"
    assert G.gba_verdict([0.1] * 5, null)["verdict"] == "DISTINCT"
    assert G.gba_verdict([0.4] * 5, null)["verdict"] == "AMBIGUOUS"
    # >= 0.5 but not above the null p95 is NOT same-object
    assert G.gba_verdict([0.6] * 5, np.full(100, 0.9))["verdict"] == "AMBIGUOUS"


def test_gba2_ceiling_all_branches_reachable():
    # collapse: halves disagree as much as unrelated subspaces
    r = G.gba2_ceiling_verdict([0.05] * 18, 0.0586)
    assert r["verdict"] == "CEILING-COLLAPSE" and r["gba_reregistration"].startswith("AMBIGUOUS")
    # clear: halves agree well, GB-A far below half of it
    r = G.gba2_ceiling_verdict([0.60] * 18, 0.0586)
    assert r["verdict"] == "CEILING-CLEAR"
    # intermediate: above collapse bar but GB-A not far enough below (or ceiling < 0.30)
    r = G.gba2_ceiling_verdict([0.20] * 18, 0.0586)
    assert r["verdict"] == "INTERMEDIATE"
    r = G.gba2_ceiling_verdict([0.60] * 18, 0.40)
    assert r["verdict"] == "INTERMEDIATE"
    assert r["ratio_gba_over_ceiling"] == pytest.approx(0.4 / 0.6)


def test_build_gba_metrics_structure():
    med, per = G.rotation_null_median_jaccard(16, 2, 3, 40, seed=3)
    m = G.build_gba_metrics({0: 0.8, 1: 0.7, 2: 0.9}, med, per, {"src": "test"})
    assert m["verdict"] == "SAME-OBJECT" and m["n_layers"] == 3
    assert set(m["per_layer_jaccard"]) == {"0", "1", "2"}
    json.dumps(m)  # JSON-serializable end to end


# ---------------- GB-2 ----------------

def test_reader_gain_diag():
    W = np.diag([3.0, 2.0, 1.0, 0.5])
    U = G.coordinate_bank(4, [0, 1])
    assert G.reader_gain(W, U) == pytest.approx((9 + 4) / 2)
    assert G.allmean_gain(W) == pytest.approx((9 + 4 + 1 + 0.25) / 4)


def test_overlap_with_singular_subspace():
    E = np.eye(6)
    U = E[:, [0, 5]]
    assert G.overlap_with_singular_subspace(U, E, 2) == pytest.approx(0.5)


def test_safetensors_range_read_bf16_and_f32():
    a32 = np.arange(6, dtype=np.float32).reshape(2, 3)
    bf = (np.arange(4, dtype=np.float32) / 3.0).astype(np.float32)
    bf16_raw = (bf.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    hdr = {"a": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
           "b": {"dtype": "BF16", "shape": [4], "data_offsets": [24, 32]}}
    hj = json.dumps(hdr).encode()
    blob = struct.pack("<Q", len(hj)) + hj + a32.tobytes() + bf16_raw
    fetch = lambda s, e: blob[s:e]
    h, d0 = G.safetensors_header(fetch)
    got_a = G.range_read_tensor(fetch, h, d0, "a")
    got_b = G.range_read_tensor(fetch, h, d0, "b")
    np.testing.assert_allclose(got_a, a32)
    np.testing.assert_allclose(got_b, bf, atol=2 ** -8)  # bf16 truncation


# ---------------- GB-1 / GB-3 ----------------

def test_dnll_and_bootstrap():
    dn = G.dnll_per_window([12, 14], [10, 10], [2, 2])
    np.testing.assert_allclose(dn, [1.0, 2.0])
    m, lo, hi = G.boot_mean_ci(np.full(50, 3.0), n_boot=100, seed=0)
    assert m == lo == hi == 3.0


def test_paired_ratio_ci():
    r, lo, hi = G.paired_ratio_ci(np.full(40, 5.0), np.full(40, 1.0), n_boot=100, seed=0)
    assert r == lo == hi == pytest.approx(5.0)


def test_inversion_verdict_all_branches():
    assert G.inversion_verdict((0.1, 0.08, 0.12), (0.5, 0.4, 0.6)) == "INVERSION-HOLDS"
    assert G.inversion_verdict((0.5, 0.4, 0.6), (0.3, 0.2, 0.4)) == "REVERSED"
    assert G.inversion_verdict((0.30, 0.20, 0.42), (0.4, 0.35, 0.5)) == "AMBIGUOUS"


def test_am1b_confirm_all_branches():
    assert G.am1b_confirm_verdict((5.0, 4.2, 6.0)) == "AM1B-CONFIRMED"
    assert G.am1b_confirm_verdict((0.5, 0.3, 0.8)) == "REVERSED"
    assert G.am1b_confirm_verdict((1.1, 0.9, 1.4)) == "AMBIGUOUS"


def _synth_wnll(rng, dnll_mean, n=60, base=2.0, ntok=100):
    none_nll = base * ntok + rng.normal(0, 1.0, n)
    arm_nll = none_nll + dnll_mean * ntok + rng.normal(0, 0.5, n)
    return none_nll, arm_nll


def test_build_gb1_metrics_synthetic_inversion():
    rng = np.random.default_rng(42)
    n, ntok = 60, 100
    none = 2.0 * ntok + rng.normal(0, 1.0, n)
    wnll = {"none": {"nll": none.tolist(), "ntok": [ntok] * n}}
    means = {"core32": 0.05, "energymatched32_s0": 0.30, "energymatched32_s1": 0.32,
             "energymatched32_s2": 0.28, "haar32_s0": 0.01, "haar32_s1": 0.011,
             "haar32_s2": 0.009, "specmatched32_s0": 0.31, "specmatched32_s1": 0.29,
             "specmatched32_s2": 0.30, "bottomsigma32": 0.001, "topsigma32": 1.5,
             "topenergy32": 2.5}
    for arm, mu in means.items():
        wnll[arm] = {"nll": (none + mu * ntok + rng.normal(0, 0.3, n)).tolist(),
                     "ntok": [ntok] * n}
    rows = {arm: float(np.exp(np.sum(w["nll"]) / (n * ntok))) for arm, w in wnll.items()}
    m = G.build_gb1_metrics({"q": rows, "qk": rows}, {"q": wnll, "qk": wnll},
                            gates={"ppl_none_ok": True}, ematch_diag=[], arm_shares={},
                            n_boot=200, seed=1)
    assert m["verdict"] == "INVERSION-HOLDS"
    q = m["sites"]["q"]
    assert q["E_over_C"][0] == pytest.approx(6.0, rel=0.25)
    assert q["core_cheaper_in_n_windows"] > 0.9 * n
    assert "S_over_C" in q and "qk" in m["sites"]
    json.dumps(m)


def test_build_gb1_metrics_reversed_branch():
    rng = np.random.default_rng(7)
    n, ntok = 40, 50
    none = 2.0 * ntok + rng.normal(0, 0.5, n)
    wnll = {"none": {"nll": none.tolist(), "ntok": [ntok] * n}}
    for arm, mu in {"core32": 0.5, "energymatched32_s0": 0.05, "haar32_s0": 0.01}.items():
        wnll[arm] = {"nll": (none + mu * ntok + rng.normal(0, 0.2, n)).tolist(),
                     "ntok": [ntok] * n}
    m = G.build_gb1_metrics({"q": {}}, {"q": wnll}, gates={}, ematch_diag=[],
                            arm_shares={}, n_boot=100, seed=2)
    assert m["verdict"] == "REVERSED"


# ---------------- GB-4 ----------------

def _gb4_arms(rng, law=True, n=98):
    """Synthetic cost = (E*G)^1.2 (law) or E^1.2 (no law), small window noise."""
    arms = {}
    Es = [0.02, 0.05, 0.09, 0.15, 0.25, 0.40, 0.60, 0.10, 0.30, 0.20]
    Gs = [0.13, 1.10, 3.00, 0.50, 15.0, 2.00, 0.80, 5.00, 0.25, 1.50]
    names = [f"arm{i}" for i in range(8)] + ["heldA", "heldB"]
    for name, E, Gv in zip(names, Es, Gs):
        mu = (E * Gv) ** 1.2 if law else E ** 1.2
        arms[name] = {"dnll_w": mu * (1 + rng.normal(0, 0.03, n)), "E": E, "G": Gv}
    return arms


def test_two_factor_law_fires():
    r = G.two_factor_analysis(_gb4_arms(np.random.default_rng(0), law=True),
                              ["heldA", "heldB"], n_boot=200, seed=0)
    assert r["verdict"] == "LAW" and r["delta_lo95"] > 0


def test_two_factor_curve_when_gain_irrelevant():
    r = G.two_factor_analysis(_gb4_arms(np.random.default_rng(1), law=False),
                              ["heldA", "heldB"], n_boot=200, seed=0)
    assert r["verdict"] == "CURVE"


def test_two_factor_auto_curve_too_few_banks():
    arms = _gb4_arms(np.random.default_rng(2))
    small = {k: arms[k] for k in list(arms)[:5] + ["heldA", "heldB"]}
    r = G.two_factor_analysis(small, ["heldA", "heldB"], n_boot=50, seed=0)
    assert r["verdict"] == "CURVE" and "auto-CURVE" in r["reason"]


def test_two_factor_nonpositive_fit_arm_excluded():
    arms = _gb4_arms(np.random.default_rng(3))
    arms["dead"] = {"dnll_w": np.full(98, -0.001), "E": 0.01, "G": 1e-5}
    r = G.two_factor_analysis(arms, ["heldA", "heldB"], n_boot=100, seed=0)
    assert "dead" in r["excluded_nonpositive"] and r["verdict"] == "LAW"


def test_two_factor_nonpositive_heldout_fails_loud():
    arms = _gb4_arms(np.random.default_rng(4))
    arms["heldA"]["dnll_w"] = np.full(98, -0.01)
    with pytest.raises(AssertionError):
        G.two_factor_analysis(arms, ["heldA", "heldB"], n_boot=10, seed=0)


# ---------------- GB-3 aggregation (the post-loop block, on synthetic cells) ----------------

def _synth_gb3(rng, n=98, ntok=511, mu=None):
    """One synthetic run in v1_ablation's artifact shape: fresh seed slots s3/s4/s5."""
    mu = mu or {}
    none = 2.0 * ntok + rng.normal(0, 1.0, n)
    tbl = {"none": {"nll": none.tolist(), "ntok": [ntok] * n}}
    for arm, m in mu.items():
        tbl[arm] = {"nll": (none + m * ntok + rng.normal(0, 0.2, n)).tolist(),
                    "ntok": [ntok] * n}
    return tbl


def test_build_gb3_metrics_confirms_on_synthetic_inversion():
    rng = np.random.default_rng(11)
    mu = {"core32": 0.05}
    for s_ in (3, 4, 5):
        mu[f"energymatched32_s{s_}"] = 0.30
        mu[f"specmatched32_s{s_}"] = 0.28
        mu[f"magmatched32_s{s_}"] = 0.80
    for s_ in range(3):
        mu[f"haar32_s{s_}"] = 0.01
    wn = _synth_gb3(rng, mu=mu)
    wn_qk = _synth_gb3(rng, mu={"core32": 0.4, "energymatched32_s3": 0.9,
                                "energymatched32_s4": 0.95, "energymatched32_s5": 0.88})
    rows = {"reproduction_gate": {"tol": 1e-3, "checked": ["none", "core32"]}}
    m = G.build_gb3_metrics(wn, wn_qk, rows, n_boot=200)
    assert m["verdict"] == "AM1B-CONFIRMED", m["verdict"]
    assert m["S_over_C"][1] > 1.0
    assert m["inversion_verdict_fresh"] == "INVERSION-HOLDS"
    # the two counts are DIFFERENT statistics and must not be conflated (the 94-vs-91 defect)
    assert "core_cheaper_than_spec_in_n_windows" in m
    assert "core_cheaper_than_ematch_in_n_windows" in m
    assert m["fresh_draw_arms"]["spec"] == ["specmatched32_s3", "specmatched32_s4",
                                            "specmatched32_s5"]
    assert m["qk_bridge_exploratory"]["E_over_C"][0] > 1.0
    assert m["reproduction_gate"]["tol"] == 1e-3
    json.dumps(m)


def test_build_gb3_metrics_reversed_branch_reachable():
    rng = np.random.default_rng(12)
    mu = {"core32": 0.5, "energymatched32_s3": 0.05, "specmatched32_s3": 0.04,
          "haar32_s0": 0.01}
    m = G.build_gb3_metrics(_synth_gb3(rng, mu=mu), None, {}, n_boot=100)
    assert m["verdict"] == "REVERSED"
    assert m["inversion_verdict_fresh"] == "REVERSED"
    assert "qk_bridge_exploratory" not in m


def test_build_gb3_metrics_fails_loud_without_a_prefix_arm():
    rng = np.random.default_rng(13)
    with pytest.raises(AssertionError):
        G.build_gb3_metrics(_synth_gb3(rng, mu={"core32": 0.1}), None, {}, n_boot=50)
