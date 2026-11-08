"""ERLANGEN offline gate — every §6 decision replayed on a KNOWN event before any GPU minute.

Runbook step 2 ("weight-free dry-runs … each replaying a known event"). Numpy only, seconds on the
control plane. Runs under the repo's pytest AND standalone:

    python -m infra.pg.tests.test_erlangen_core

Each test plants an event whose answer is known by construction and asserts the instrument reports
it — and, for every gate, plants the event that must make it FAIL and asserts it fails
(`feedback_signals_that_cannot_fail`: a gate that only ever passes is not a gate).

No try/except (CLAUDE.md).
"""
import numpy as np

from infra.pg import erlangen_core as EC

RNG = np.random.default_rng(20260803)


# ===========================================================================
# P1 — the MDE gate (amendment A1).
# ===========================================================================
def _planted_types(n_per=4, n_read=24, sep=1.0, noise=0.25, seed=0):
    """3 types × n_per components. Each type has its own random RESPONSE PATTERN; components of a
    type share it up to noise, and each component gets its own random overall GAIN — so the types
    are separable by pattern (direction) but NOT by magnitude. That is exactly the object A1's
    normalized statistic is supposed to find, and the object the raw statistic should struggle on."""
    rng = np.random.default_rng(seed)
    patterns = rng.standard_normal((3, n_read)) * sep
    rows, labels = [], []
    for t in range(3):
        for _ in range(n_per):
            gain = rng.uniform(0.2, 5.0)
            rows.append(gain * (patterns[t] + noise * rng.standard_normal(n_read)))
            labels.append(t)
    return np.asarray(rows), np.asarray(labels)


def test_p1_recovers_planted_types():
    F, y = _planted_types(seed=1)
    Fb = F + 0.02 * RNG.standard_normal(F.shape)          # the second seed's measurement
    sc = EC.column_scale(F)["scale"]
    v = EC.p1_verdict(F, Fb, y, sc, k=3, n_perm=2000, n_split=40)
    assert v["verdict"] == "PASS", v
    assert v["ari_normalized"] > 0.9, v
    assert v["perm_p"] < 0.01, v
    assert v["stability_ari_median"] >= EC.ARI_STABLE, v


def test_p1_fails_on_structureless_fingerprints():
    """The event that MUST make the gate fail: fingerprints with no type structure at all."""
    F = RNG.standard_normal((12, 24))
    Fb = F + 0.02 * RNG.standard_normal(F.shape)
    y = np.array([0] * 4 + [1] * 4 + [2] * 4)
    sc = EC.column_scale(F)["scale"]
    v = EC.p1_verdict(F, Fb, y, sc, k=3, n_perm=2000, n_split=40)
    assert v["verdict"] == "FAIL", v
    assert v["perm_p"] > 0.01 or v["ari_normalized"] < EC.ARI_PASS, v


def test_p1_voids_on_unmeasurable_component():
    """A component the instrument cannot move must VOID the gate, not be scored as a mis-cluster."""
    F, y = _planted_types(seed=2)
    Fb = F + 0.02 * RNG.standard_normal(F.shape)
    F[0] = 0.0                       # dead component: signal ≤ noise by construction
    Fb[0] = 0.01 * RNG.standard_normal(F.shape[1])
    sc = EC.column_scale(F)["scale"]
    v = EC.p1_verdict(F, Fb, y, sc, k=3, n_perm=500, n_split=20)
    assert v["verdict"] == "VOID", v
    assert v["noise_floor"]["n_unmeasurable"] >= 1, v


def test_magnitude_only_structure_is_caught_by_the_normalized_statistic():
    """The A1 confound, planted: types that differ ONLY in how much damage they do. The raw
    (magnitude-carrying) statistic recovers them; the normalized one must NOT — that dissociation
    is the whole reason the normalized statistic is the headline."""
    base = RNG.standard_normal(24)
    rows, y = [], []
    for t, g in enumerate([0.5, 2.0, 8.0]):
        for _ in range(4):
            rows.append(g * base + 0.02 * RNG.standard_normal(24))
            y.append(t)
    F = np.asarray(rows)
    y = np.asarray(y)
    sc = EC.column_scale(F)["scale"]
    ari_raw = EC.adjusted_rand(EC.cluster(EC.preprocess_raw(F, sc), 3), y)
    ari_norm = EC.adjusted_rand(EC.cluster(EC.preprocess(F, sc), 3), y)
    assert ari_raw > 0.7, ari_raw
    assert ari_norm < 0.5, ari_norm


def test_ward_and_ari_are_sane():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    lab = EC.cluster(X, 2)
    assert lab[0] == lab[1] and lab[2] == lab[3] and lab[0] != lab[2], lab
    assert EC.adjusted_rand([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert abs(EC.adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1])) < 0.6


def test_select_k_finds_the_planted_k():
    F, _ = _planted_types(n_per=6, seed=3)
    sc = EC.column_scale(F)["scale"]
    r = EC.select_k(F, sc, range(2, 7), n_split=30)
    assert r["k_star"] == 3, r


def test_construct_validity_gate_both_directions():
    a = np.arange(12).astype(float)
    good = EC.construct_validity_rho(a, a + 0.1 * RNG.standard_normal(12))
    assert good["passes"], good
    bad = EC.construct_validity_rho(a, RNG.standard_normal(12))
    assert not bad["passes"] or bad["lo95"] <= 0, bad


# ===========================================================================
# P2 — Arm A algebra (amendment A2).
# ===========================================================================
def test_algebra_positive_and_null():
    pos = RNG.normal(1.0, 0.1, 20)
    null0 = RNG.normal(0.01, 0.002, 20)
    nullr = RNG.normal(0.05, 0.01, 20)
    v = EC.algebra_verdict(pos, null0, nullr, np.ones(20))
    assert v["verdict"] == "PASS", v


def test_algebra_voids_when_reharvest_saturates():
    """The saturation guard: if re-harvest alone produces commutators as large as the ordered
    pairs, the positive control could not have failed → VOID."""
    pos = RNG.normal(1.0, 0.1, 20)
    v = EC.algebra_verdict(pos, RNG.normal(0.01, 0.002, 20), RNG.normal(1.2, 0.1, 20), np.ones(20))
    assert v["verdict"] == "VOID-saturated", v


def test_algebra_fails_without_order_structure():
    x = RNG.normal(0.01, 0.002, 20)
    v = EC.algebra_verdict(x, RNG.normal(0.01, 0.002, 20), RNG.normal(0.005, 0.001, 20), -np.ones(20))
    assert v["verdict"] == "FAIL", v


def test_absorption_and_dependency_graph():
    base = np.zeros(4)
    b_i = np.array([1.0, 0, 0, 0])
    b_j = np.array([0, 1.0, 0, 0])
    b_ji = b_i.copy()                       # i fully absorbs j
    a_ij = EC.absorption(b_ji, b_i, b_j, base)
    assert a_ij["ratio"] == 0.0, a_ij
    b_ij = b_j + b_i                        # j does NOT absorb i
    a_ji = EC.absorption(b_ij, b_j, b_i, base)
    assert abs(a_ji["ratio"] - 1.0) < 1e-9, a_ji
    g = EC.dependency_graph({("i", "j"): a_ij, ("j", "i"): a_ji})
    assert g["edges"] == [{"from": "i", "to": "j", "ratio": 0.0}], g
    assert g["n_mutual"] == 0, g


def test_commutator_is_zero_for_identical_behaviour():
    b = RNG.standard_normal(6)
    assert EC.commutator(b, b) == 0.0


# ===========================================================================
# Arm M — metric axioms (amendment A3).
# ===========================================================================
def _sym_matrix(n=5, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 1, (n, 3))
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))   # a REAL metric by construction
    return D


def test_triangle_test_clean_metric_has_no_violations():
    D = _sym_matrix()
    sig = np.full_like(D, 0.01)
    tasks = list("ABCDE")
    t = EC.triangle_test(D.tolist(), sig.tolist(), tasks)
    assert t["n_confirmed"] == 0, t
    assert t["n_evaluable"] == 60, t
    s = EC.symmetry_test(D.tolist(), sig.tolist(), tasks)
    assert s["n_confirmed"] == 0, s


def test_triangle_test_detects_a_planted_violation():
    D = _sym_matrix()
    D[0, 2] = D[0, 1] + D[1, 2] + 1.0        # planted shortcut through B
    sig = np.full_like(D, 0.01)
    tasks = list("ABCDE")
    t = EC.triangle_test(D.tolist(), sig.tolist(), tasks)
    assert t["n_confirmed"] >= 1, t
    assert t["best_shortcut"]["a"] == "A" and t["best_shortcut"]["c"] == "C", t
    assert t["best_shortcut"]["shortcut_saving"] > 0.9, t


def test_symmetry_test_detects_a_planted_asymmetry():
    D = _sym_matrix()
    D[1, 0] = D[0, 1] + 2.0
    sig = np.full_like(D, 0.01)
    s = EC.symmetry_test(D.tolist(), sig.tolist(), list("ABCDE"))
    assert s["n_confirmed"] == 1, s
    # the DESCRIPTIVE binomial cannot reach 0.05 on one pair out of ten (1−0.975^10 = 0.22) —
    # that is exactly why the adjudicator is the magnitude-aware bootstrap, not the count.
    assert s["p"] > 0.05, s
    b = EC.axiom_bootstrap(D, sig, "symmetry", n_boot=2000, seed=0)
    assert b["p"] < 0.05, b
    assert not b["underpowered"], b


def test_metric_closure_is_the_triangle_projection():
    D = _sym_matrix()
    D[0, 2] = D[0, 1] + D[1, 2] + 1.0
    C = EC.metric_closure(D)
    assert C[0, 2] <= D[0, 1] + D[1, 2] + 1e-12, (C[0, 2], D[0, 1] + D[1, 2])
    assert (C <= D + 1e-12).all()
    # a matrix that already satisfies the triangle inequality is its own closure
    M = _sym_matrix(seed=7)
    assert np.abs(EC.metric_closure(M) - M).max() < 1e-9


def test_axiom_bootstrap_detects_and_rejects_and_reports_power():
    tasks = list("ABCDE")
    D = _sym_matrix()
    sig = np.full_like(D, 0.01)
    clean = EC.axiom_bootstrap(D, sig, "triangle", n_boot=2000, seed=0)
    assert clean["p"] > 0.05, clean
    assert not clean["underpowered"], clean
    Dv = D.copy()
    Dv[0, 2] = Dv[0, 1] + Dv[1, 2] + 1.0
    hit = EC.axiom_bootstrap(Dv, sig, "triangle", n_boot=2000, seed=0)
    assert hit["p"] < 0.05, hit
    assert len(tasks) == 5


def test_axiom_bootstrap_voids_when_underpowered():
    """The guard that stops 'no violations found' from meaning 'could not have found any'."""
    D = _sym_matrix()
    huge = np.full_like(D, 5.0)
    r = EC.axiom_bootstrap(D, huge, "triangle", n_boot=500, seed=0)
    assert r["underpowered"], r
    assert r["min_detectable_over_scale"] > 1.0, r


def test_delta_min_suppresses_a_negligible_violation():
    """Effect-size floor: a statistically resolvable but negligible violation must not read as
    non-geometric."""
    D = _sym_matrix()
    # A violation of size ε means the direct edge exceeds the SHORTEST two-hop path by ε — not the
    # path through some arbitrary node (that plants ε against B and a large excess against every
    # node whose path is shorter, which is what this test caught the first time).
    best = min(D[0, b] + D[b, 2] for b in (1, 3, 4))
    D[0, 2] = best + 0.001
    tiny_sigma = np.full_like(D, 1e-5)
    raw = EC.axiom_bootstrap(D, tiny_sigma, "triangle", n_boot=1000, seed=0, delta_min=0.0)
    guarded = EC.axiom_bootstrap(D, tiny_sigma, "triangle", n_boot=1000, seed=0,
                                 delta_min=0.05 * float(np.median(D[~np.eye(5, dtype=bool)])))
    assert raw["stat"] > 0, raw
    assert guarded["stat"] == 0.0, guarded


def test_noise_swamps_the_violation_when_sigma_is_large():
    """The other direction: with honest (large) uncertainty the same planted violation must NOT be
    confirmed — the test is not a one-sided ratchet."""
    D = _sym_matrix()
    D[0, 2] = D[0, 1] + D[1, 2] + 1.0
    sig = np.full_like(D, 2.0)
    t = EC.triangle_test(D.tolist(), sig.tolist(), list("ABCDE"))
    assert t["n_confirmed"] == 0, t


def test_censored_triples_are_excluded_in_both_directions():
    D = _sym_matrix()
    D[0, 2] = D[0, 1] + D[1, 2] + 1.0
    sig = np.full_like(D, 0.01)
    tasks = list("ABCDE")
    # censoring a LEG can only shrink the excess → a positive excess is no longer evaluable
    t = EC.triangle_test(D.tolist(), sig.tolist(), tasks, censored={("A", "B"): True})
    assert not any(x["a"] == "A" and x["b"] == "B" and x["c"] == "C" for x in t["triples"]), t
    assert t["n_non_evaluable"] > 0, t
    # censoring the DIRECT edge can only grow the excess → non-violations stop being evaluable
    t2 = EC.triangle_test(D.tolist(), sig.tolist(), tasks, censored={("A", "D"): True})
    assert t2["n_non_evaluable"] > 0, t2


def test_pooled_seed_sd_voids_when_underpowered():
    assert EC.pooled_seed_sd({"p": [1.0, 1.2]})["sd"] is None
    r = EC.pooled_seed_sd({"p": [1.0, 1.2], "q": [2.0, 2.4], "r": [3.0, 3.1]})
    assert r["sd"] is not None and r["sd"] > 0, r
    assert EC.cell_sigma(0.3, 0.4) == 0.5


def _verdict(D, sig, tasks, nb=1500):
    """The full M pipeline exactly as the job runs it: descriptive tables + both gauges."""
    Dl = np.log1p(D)
    sigl = sig / (1.0 + D)                       # delta-method transport of σ into the log gauge
    tri_d = EC.triangle_test(D.tolist(), sig.tolist(), tasks)
    sym_d = EC.symmetry_test(D.tolist(), sig.tolist(), tasks)
    mask = np.asarray(tri_d["evaluable_mask"])
    dmin = 0.05 * float(np.median(D[~np.eye(len(tasks), dtype=bool)]))
    kw = dict(n_boot=nb, seed=0, delta_min=dmin)
    return EC.metric_verdict(
        EC.axiom_bootstrap(D, sig, "symmetry", **kw),
        EC.axiom_bootstrap(D, sig, "triangle", censored_mask=mask, **kw),
        EC.axiom_bootstrap(Dl, sigl, "symmetry", n_boot=nb, seed=0, delta_min=np.log1p(dmin)),
        EC.axiom_bootstrap(Dl, sigl, "triangle", censored_mask=mask, n_boot=nb, seed=0,
                           delta_min=np.log1p(dmin)),
        "ok", sym_desc=sym_d, tri_desc=tri_d)


def test_metric_verdict_three_ways_and_void():
    tasks = list("ABCDE")
    sig = np.full((5, 5), 0.01)
    D = _sym_matrix()
    geo = _verdict(D, sig, tasks)
    assert geo["verdict"] == "geometric", geo
    # A PURE quasi-metric: add a path-additive potential h. d(i,j) = D[i,j] + h[j] − h[i] is
    # asymmetric by 2(h[j]−h[i]) while every triple's excess is algebraically unchanged, so this
    # isolates symmetry-failure from triangle-failure. (Inflating individual directed edges — the
    # obvious construction — breaks BOTH, which is what this test caught.)
    h = np.array([0.0, 0.30, -0.20, 0.10, 0.05])
    Da = D + h[None, :] - h[:, None]
    np.fill_diagonal(Da, 0.0)
    assert (Da[~np.eye(5, dtype=bool)] > 0).all(), Da
    qm = _verdict(Da, sig, tasks)
    assert qm["verdict"] == "quasi-metric", qm
    Dt = D.copy()
    for i, j, k in [(0, 1, 2), (0, 2, 3), (1, 3, 4)]:
        Dt[i, k] = Dt[i, j] + Dt[j, k] + 1.0
    nong = _verdict(Dt, sig, tasks)
    assert nong["verdict"] == "non-geometric", nong
    assert nong["best_shortcut"]["shortcut_saving"] > 0.9, nong
    void = EC.metric_verdict(None, None, None, None, "VOID — fewer than 2 pairs with ≥2 seeds")
    assert void["verdict"] == "VOID", void


def test_metric_verdict_voids_a_powerless_geometric_reading():
    """A 'geometric' verdict from a test that could not have detected a violation is VOID."""
    tasks = list("ABCDE")
    D = _sym_matrix()
    v = _verdict(D, np.full((5, 5), 5.0), tasks, nb=400)
    assert v["verdict"] == "VOID-underpowered", v
    assert v["verdict_before_power_gate"] == "geometric", v


# ===========================================================================
# Arm B.
# ===========================================================================
def test_boundary_verdict_three_branches():
    n = 40
    arch = EC.boundary_verdict(RNG.normal(1.0, 0.05, n), RNG.normal(1.0, 0.05, n),
                               RNG.normal(1.0, 0.05, n), RNG.normal(1.0, 0.05, n))
    assert arch["verdict"] == "H_architecture-wall", arch
    rep = EC.boundary_verdict(RNG.normal(5.0, 0.2, n), RNG.normal(1.0, 0.05, n),
                              RNG.normal(1.0, 0.05, n), RNG.normal(1.0, 0.05, n))
    assert rep["verdict"] == "H_repelled", rep
    opt = EC.boundary_verdict(RNG.normal(5.0, 0.2, n), RNG.normal(1.0, 0.05, n),
                              RNG.normal(4.0, 0.2, n), RNG.normal(1.0, 0.05, n))
    assert opt["verdict"] == "H_optimizer-wall", opt


# ===========================================================================
# Arm X — mirror-v2 + P3.
# ===========================================================================
def test_mantel_detects_and_rejects():
    n = 6
    A = RNG.standard_normal((n, n))
    A = (A + A.T) / 2
    B = 3 * A + 0.05 * RNG.standard_normal((n, n))
    B = (B + B.T) / 2
    hit = EC.mantel(A, B, n_perm=2000)
    assert hit["r"] > 0.9 and hit["p_two_sided"] < 0.05, hit
    C = RNG.standard_normal((n, n))
    C = (C + C.T) / 2
    miss = EC.mantel(A, C, n_perm=2000)
    assert miss["p_two_sided"] > 0.05 or abs(miss["r"]) < 0.5, miss


def test_self_transplant_gate():
    assert EC.self_transplant_gate([0.0, 1e-9, -1e-9], 1e-6)["passes"]
    bad = EC.self_transplant_gate([0.0, 0.4], 1e-6)
    assert not bad["passes"] and bad["max_abs_delta"] == 0.4


# ===========================================================================
# P5 — Arm D ddmin.
# ===========================================================================
def test_ddmin_recovers_a_planted_witness():
    universe = [f"e{i}" for i in range(32)]
    truth = {"e3", "e11", "e29"}
    calls = {"n": 0}

    def oracle(sub):
        calls["n"] += 1
        return truth.issubset(set(sub))

    r = EC.ddmin(universe, oracle, max_evals=600, seed=0)
    assert set(r["witness"]) == truth, r
    assert r["size"] == 3 and not r["budget_exhausted"], r
    assert r["n_monotonicity_violations"] == 0, r


def test_ddmin_reports_truncation_instead_of_a_false_minimum():
    universe = [f"e{i}" for i in range(64)]
    truth = {"e5", "e40"}

    def oracle(sub):
        return truth.issubset(set(sub))

    r = EC.ddmin(universe, oracle, max_evals=6, seed=0)
    assert r["budget_exhausted"] and "TRUNCATED" in r["note"], r


def test_witness_family_reports_non_uniqueness():
    universe = [f"e{i}" for i in range(24)]

    def oracle(sub):
        # ANY two of these three restore — the witness is genuinely non-unique
        return len({"e1", "e2", "e3"} & set(sub)) >= 2

    fam = EC.witness_family(universe, oracle, n_restart=3, max_evals=600, seed=0)
    assert all(s == 2 for s in fam["sizes"]), fam
    assert set(fam["union"]) <= {"e1", "e2", "e3"}, fam


def test_ddmin_precondition_raises_when_nothing_restores():
    caught = False
    try:
        EC.ddmin(["a", "b"], lambda s: False, max_evals=10)
    except AssertionError:
        caught = True
    assert caught, "ddmin must RAISE when the full set does not restore"


# ===========================================================================
# P6 — Arm G grammar description length.
# ===========================================================================
def test_grammar_compresses_a_planted_lowrank_plus_interface_edit():
    m, n = 2048, 2048
    r = 8
    prog = [{"kind": "rank", "dl": EC.dl_rank(m, n, r)},
            {"kind": "interface", "dl": EC.dl_dense(256, 2048)}]
    baseline = EC.dl_dense(m, n) * 1.0
    ratio = EC.compression_ratio(EC.program_dl(prog), baseline)
    assert ratio < 0.2, ratio


def test_grammar_does_not_compress_a_dense_edit():
    m = n = 512
    prog = [{"kind": "dense", "dl": EC.dl_dense(m, n)}]
    assert EC.compression_ratio(EC.program_dl(prog), EC.dl_dense(m, n)) >= 1.0


def test_sparse_dl_charges_for_the_index():
    assert EC.dl_sparse(100, 2048, 2048) > 100 * EC.BITS_PER_PARAM


def test_program_dl_rejects_an_unknown_term():
    caught = False
    try:
        EC.program_dl([{"kind": "wormhole", "dl": 1.0}])
    except AssertionError:
        caught = True
    assert caught


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for name, fn in fns:
        fn()
        print(f"  ok  {name}")
    print(f"\n[erlangen-offline-gate] {len(fns)}/{len(fns)} pure-logic checks PASS")
