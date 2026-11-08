"""Weight-free unit tests for MERCATOR's NEW instruments (§8 step 3, §11 reuse-ledger row "NEW").

Run: PYTHONPATH=. python3 infra/pg/tests/test_mercator_instruments.py

Covered here: M-B1 row-Gram, M-8 predictor statistics, M-12 head restriction + two-clock,
M-4 erosion fitters, M-6 format round-trip, M-5 compression hook, M-C1/C2 assemblers, and the
pure-numpy half of M-10/M-11.

The torch half of M-10/M-11 (model build, gauge rotation, equivariance gate, BlockAdam) is
exercised by infra/aws_pg/jobs/mercator_dryrun.py on an instance — torch is deliberately absent on
the control plane, and the house rule is that nothing scientific runs locally.

Every instrument gets BOTH a signal case and a null case wherever it has a decision statistic. An
estimator that only ever sees planted signal is not tested; it is demonstrated.
"""
import numpy as np

from infra.pg import mercator_arch as A
from infra.pg import mercator_assemble as AS
from infra.pg import mercator_compress as C
from infra.pg import mercator_erosion as E
from infra.pg import mercator_format as F
from infra.pg import mercator_induction as I
from infra.pg import mercator_predict as PR
from infra.pg import mercator_rowgram as RG
from infra.pg import mercator_train as TR


# ---------------------------------------------------------------------------
# M-B1 — the output-side census.
# ---------------------------------------------------------------------------
def test_rowgram_planted_null_recovers_signal_and_stays_at_chance_on_noise():
    r = RG.planted_rowspace_null()
    assert r["planted_capture"] > 0.9, "the estimator failed to recover a planted row-space block"
    assert abs(r["null_capture"] - r["chance_capture"]) < 0.05, \
        "the estimator recovered structure from isotropic noise — it cannot be trusted on real data"


def test_rowgram_memory_budget_refuses_an_impossible_partition():
    RG.assert_fits(["q_proj"], range(18), {"q_proj": 2048}, budget_gb=10)
    try:
        RG.assert_fits(["gate_proj", "up_proj"], range(18), {"gate_proj": 16384, "up_proj": 16384},
                       budget_gb=64)
        raise SystemExit("77 GB of Grams must not pass a 64 GB budget")
    except AssertionError as ex:
        assert "partition the type list" in str(ex)


def test_row_banks_are_the_full_battery_and_orthonormal():
    rng = np.random.default_rng(0)
    m, n = 40, 60
    W0 = rng.standard_normal((m, n))
    G = np.cov(rng.standard_normal((m, 200)))
    _lam, V = RG._eig_ascending(G)
    banks = RG.build_row_banks(V, W0, k=8, seed_base=1)
    from infra.pg import darwin_core as D
    assert set(banks) == set(D.BANK_IDS)
    for bid, U in banks.items():
        assert U.shape == (m, 8), f"{bid}: {U.shape}"


def test_eig_extremes_matches_the_full_decomposition_it_replaces():
    """The wide types cannot afford a full eigendecomposition; the subset driver must agree with it
    exactly on the part the banks actually use."""
    rng = np.random.default_rng(21)
    m, k = 80, 8
    X = rng.standard_normal((m, 200))
    G = X @ X.T / 200.0
    lam, V = RG._eig_ascending(G)
    lb, Vb, lt, Vt, trace, full = RG.eig_extremes(G, k)
    assert full is not None                                  # small enough for the full path
    assert np.allclose(lb, lam[:k]) and np.allclose(lt, lam[-k:])
    assert abs(trace - float(np.trace(G))) < 1e-12
    # eigenvectors agree up to per-column sign, which is all an eigenbasis is defined to
    assert np.allclose(np.abs(Vb.T @ V[:, :k]), np.eye(k), atol=1e-8)
    # and the avoidance statistic computed from the extremes matches the full-spectrum form
    a_full = RG.row_avoidance(G, k=k)
    a_ext = RG.row_avoidance_from_extremes(lb, trace, m, k)
    assert abs(a_full["ratio"] - a_ext["ratio"]) < 1e-10
    # the battery built from the extremes is the same battery
    W0 = rng.standard_normal((m, 60))
    b1 = RG.build_row_banks(V, W0, k=k, seed_base=5)
    b2 = RG.build_row_banks_from_extremes(Vb, Vt, W0, k=k, seed_base=5)
    assert set(b1) == set(b2)
    for bid in ("bottomsigma32", "topsigma32", "haar32_s0"):
        assert np.allclose(b1[bid], b2[bid])


def test_krylov_extremes_agree_with_the_dense_decomposition():
    """The wide types use Lanczos, not a dense driver — a subset driver still pays the full O(n^3)
    tridiagonalisation. Forced on a small matrix so the two paths can be compared directly."""
    rng = np.random.default_rng(31)
    m, k = 120, 6
    X = rng.standard_normal((m, 300))
    G = X @ X.T / 300.0
    lam, V = RG._eig_ascending(G)
    lb, Vb, lt, Vt, trace, full = RG.eig_extremes(G, k, force_krylov=True)
    assert full is None                                   # the Krylov path banks no full spectrum
    assert np.allclose(lb, lam[:k], atol=1e-8), (lb, lam[:k])
    assert np.allclose(lt, lam[-k:], atol=1e-8)
    assert abs(trace - float(np.trace(G))) < 1e-10
    # eigenvectors agree up to per-column sign
    assert np.allclose(np.abs(Vb.T @ V[:, :k]), np.eye(k), atol=1e-6)
    assert np.allclose(np.abs(Vt.T @ V[:, -k:]), np.eye(k), atol=1e-6)
    # and the statistic the arm actually reports is unchanged
    assert abs(RG.row_avoidance(G, k=k)["ratio"]
               - RG.row_avoidance_from_extremes(lb, trace, m, k)["ratio"]) < 1e-8


def test_row_avoidance_is_one_at_chance_and_below_one_when_suppressed():
    rng = np.random.default_rng(1)
    m, k = 64, 8
    iso = np.zeros((m, m))
    for _ in range(20):
        B = rng.standard_normal((m, 96))
        iso += (B @ B.T) / float((B ** 2).sum())
    a = RG.row_avoidance(iso, k=k)
    assert 0.5 < a["ratio"] < 1.5, f"isotropic cloud should sit near ratio 1, got {a['ratio']}"
    Q = np.linalg.qr(rng.standard_normal((m, m)))[0]
    d = np.ones(m)
    d[:k] = 0.02
    sup = np.zeros((m, m))
    for _ in range(20):
        R = rng.standard_normal((m, 96))
        X = Q @ (d[:, None] * R)
        sup += (X @ X.T) / float((X ** 2).sum())
    assert RG.row_avoidance(sup, k=k)["ratio"] < 0.1


# ---------------------------------------------------------------------------
# M-8 — the predictor.
# ---------------------------------------------------------------------------
def test_consistency_is_bounded_and_catches_mismatched_accumulators():
    assert np.allclose(PR.consistency([2.0, 1.0], [2.0, 4.0]), [1.0, 0.25])
    try:
        PR.consistency([5.0], [4.0])
        raise SystemExit("‖Σg‖ > Σ‖g‖ is impossible and must fail loud")
    except AssertionError as ex:
        assert "different step counts" in str(ex)


def test_zero_gradient_columns_are_counted_not_crashed_on_or_hidden():
    """bf16 underflow makes some columns exactly zero at every step; 0/0 is undefined."""
    cons, counts = PR.consistency([0.0, 1.0, 2.0], [0.0, 4.0, 2.0], return_counts=True)
    assert cons[0] == 0.0 and np.allclose(cons[1:], [0.25, 1.0])
    assert counts["n_zero_gradient_columns"] == 1 and counts["n_columns"] == 3
    # a column with no gradient scores 0 through the MAGNITUDE factor, so the convention above
    # cannot manufacture a high suppression score out of an undefined ratio
    assert PR.suppression_score([0.0, 1.0], [0.0, 0.0])[0] == 0.0
    # ...but a mostly-dead layer is a broken backward pass, and that still fails loud
    try:
        PR.consistency([0.0] * 8 + [1.0], [0.0] * 8 + [1.0])
        raise SystemExit("a mostly zero-gradient matrix must fail loud")
    except AssertionError as ex:
        assert "broken backward pass" in str(ex)


def test_suppression_score_ranks_large_and_cancelling_first():
    mag = np.array([1.0, 1.0, 0.1, 0.1])
    cons = np.array([0.0, 1.0, 0.0, 1.0])
    s = PR.suppression_score(mag, cons)
    assert PR.predict_address(s, 1) == [0], "large-and-cancelling must rank first"
    assert s[1] == 0.0 and s[3] == 0.0


def test_predictor_beats_the_baseline_when_the_mechanism_holds_and_not_otherwise():
    rng = np.random.default_rng(7)
    n, k = 128, 16
    truth = np.arange(k)
    U = np.zeros((n, k))
    U[truth, np.arange(k)] = 1.0                       # the "measured bank": exactly the truth
    mag = rng.uniform(0.1, 0.2, n)
    cons = rng.uniform(0.8, 1.0, n)
    mag[truth] = 1.0                                   # sink coords: large gradient...
    cons[truth] = 0.05                                 # ...that cancels
    act = rng.uniform(0.1, 0.2, n)                     # activations carry NO information here
    good = PR.compare_to_baselines(score=PR.suppression_score(mag, cons), act_rms=act,
                                   grad_mag=mag, U_bank=U, k=k, seed=3)
    assert good["predictor"]["jaccard"] == 1.0 and good["beats_activation_baseline"]
    # the harder control is in the SAME table: here gradient magnitude alone also finds the
    # address, so the predictor does not beat it — and that must be visible, not hidden
    assert good["gradient_magnitude"]["jaccard"] == 1.0
    assert not good["beats_gradient_magnitude_baseline"]
    # ...and when the score is uninformative, the comparison must NOT report a win
    flat = PR.compare_to_baselines(score=rng.uniform(0, 1, n), act_rms=act, grad_mag=mag,
                                   U_bank=U, k=k, seed=3)
    assert flat["predictor"]["jaccard"] < 0.5


def test_heldout_family_cell_refuses_a_family_outside_the_roster():
    r = PR.heldout_family_cell(["gemma-2b", "llama-3.2-3b"], "llama-3.2-3b")
    assert r["fit_on"] == ["gemma-2b"]
    try:
        PR.heldout_family_cell(["gemma-2b"], "mistral-7b")
        raise SystemExit("a held-out family outside the roster must fail loud")
    except AssertionError:
        pass


def test_sparsity_row_refuses_an_impossible_address_size():
    rows = PR.sparsity_scaling_row({"gemma-2b": {"address_size": 39, "d": 2048}})
    assert abs(rows[0]["fraction"] - 39 / 2048) < 1e-12
    try:
        PR.sparsity_scaling_row({"gemma-2b": {"address_size": 4096, "d": 2048}})
        raise SystemExit("an address larger than the model width must fail loud")
    except AssertionError:
        pass


# ---------------------------------------------------------------------------
# M-12 — induction heads.
# ---------------------------------------------------------------------------
def test_prefix_matching_scores_a_planted_induction_head_and_not_a_uniform_one():
    L, H_ = 6, 2
    T = 2 * L
    attn = np.zeros((H_, T, T))
    for i in range(L + 1, T):                    # head 0: a perfect induction head
        attn[0, i, i - L + 1] = 1.0
    attn[1] = np.ones((T, T)) / T                # head 1: uniform
    s = I.prefix_matching_score(attn, L)
    assert s[0] > 0.99 and s[1] < 0.2
    sel = I.induction_heads(s.reshape(1, -1))
    assert sel["heads"] == [(0, 0)] and sel["n_heads_total"] == 2
    assert set(sel["sweep"]) == set(I.THRESHOLD_SWEEP)


def test_prefix_matching_accepts_a_wrapped_sequence_but_refuses_a_short_one():
    """The processor appends a trailing newline after the repeat block, so the block FITS in the
    sequence rather than being all of it — requiring equality asserted the wrapper away."""
    L, T = 6, 2 * 6 + 3            # block plus a trailing token or two
    attn = np.zeros((2, T, T))
    for i in range(L + 1, 2 * L):
        attn[0, i, i - L + 1] = 1.0
    s = I.prefix_matching_score(attn, L, offset=0)
    assert s[0] > 0.99 and s[1] == 0.0
    try:
        I.prefix_matching_score(np.zeros((1, 11, 11)), 6)
        raise SystemExit("a sequence too short for the block must fail loud")
    except AssertionError as ex:
        assert "too short" in str(ex)


def test_head_slice_refuses_an_index_the_matrix_cannot_have():
    W = np.zeros((8 * 4, 32))
    assert I.head_slice(W, 7, 4).shape == (4, 32)
    try:
        I.head_slice(W, 8, 4)
        raise SystemExit("head 8 does not exist in an 8-head matrix (the MQA/GQA trap)")
    except AssertionError:
        pass


def test_head_read_energy_sits_at_k_over_n_for_a_random_bank():
    rng = np.random.default_rng(11)
    n, k = 256, 32
    W = rng.standard_normal((16, n))
    from infra.pg import humboldt_core as H
    e = I.head_read_energy(W, H.haar_subspace(n, k, 5))
    assert abs(e - k / n) < 0.05, f"a Haar bank must sit near k/n, got {e}"


def test_two_clock_overlay_reports_separation_with_its_grid_resolution():
    steps = [1, 2, 4, 8, 16, 32]
    early = [0.0, 0.1, 0.9, 1.0, 1.0, 1.0]
    late = [0.0, 0.0, 0.0, 0.1, 0.9, 1.0]
    r = I.two_clock_overlay((steps, early), (steps, late))
    assert r["separation_steps"] > 0
    assert r["grid_resolution"] > 0
    # a flat curve has no onset and must not be reported as onset at step 0
    try:
        I.onset_step(steps, [1.0] * 6)
        raise SystemExit("a curve with no rise has no onset")
    except AssertionError as ex:
        assert "no rise" in str(ex)


# ---------------------------------------------------------------------------
# M-4 — erosion.
# ---------------------------------------------------------------------------
def test_sweep_shape_is_enforced():
    E.check_sweep(range(6), [10, 100, 1000], ["a", "b"])
    try:
        E.check_sweep(range(5), [10, 100, 1000], ["a", "b"])
        raise SystemExit("a 5-subspace sweep must fail loud (§4 requires >=6)")
    except AssertionError as ex:
        assert "subspaces" in str(ex)


def test_stamp_lives_inside_the_subspace_and_has_the_requested_norm():
    from infra.pg import humboldt_core as H
    U = H.haar_subspace(64, 8, 3)
    D, d = E.stamp(U, amplitude=0.25, seed=1, out_dim=32)
    assert abs(np.linalg.norm(D) - 0.25) < 1e-12
    assert abs(np.linalg.norm(d) - 1.0) < 1e-12
    assert H.bank_energy_fraction(D, U) > 1 - 1e-12


def test_retained_fraction_is_signed_and_unclipped():
    from infra.pg import humboldt_core as H
    U = H.haar_subspace(32, 4, 0)
    D, d = E.stamp(U, amplitude=1.0, seed=2, out_dim=16)
    assert abs(E.retained_fraction(D, d) - 1.0) < 1e-12
    assert abs(E.retained_fraction(-D, d) + 1.0) < 1e-12       # reversal is an observation
    assert abs(E.retained_fraction(2 * D, d) - 2.0) < 1e-12    # reinforcement too


def test_decay_fit_refuses_to_silently_drop_a_fully_eroded_point():
    r = E.decay_constant([1.0, 2.0, 4.0], [np.exp(-0.5), np.exp(-1.0), np.exp(-2.0)])
    assert abs(r["lambda"] - 0.5) < 1e-9 and r["r2_log"] > 0.999
    try:
        E.decay_constant([1.0, 2.0, 4.0], [0.5, 0.0, 0.1])
        raise SystemExit("a non-positive retention must not be dropped from the fit")
    except AssertionError as ex:
        assert "fully_eroded" in str(ex)


def test_traffic_fit_finds_a_planted_power_law_and_reports_monotonicity():
    traffic = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
    lam = 0.3 * traffic ** 1.5
    f = E.fit_decay_vs_traffic(traffic, lam)
    assert abs(f["slope"] - 1.5) < 1e-6 and f["r2_log"] > 0.999 and f["spearman"] > 0.99
    rng = np.random.default_rng(5)
    noise = E.fit_decay_vs_traffic(traffic, rng.uniform(0.1, 0.2, 6))
    assert abs(noise["spearman"]) < 0.95, "unrelated data must not read as a monotone relation"


def test_amplitude_ladder_is_log_spaced_and_brackets_the_edit_scale():
    from infra.pg import mercator_core as M
    lad = E.amplitude_ladder(0.05, 4)
    M.check_dose_grid(lad, arm="M-4")
    assert lad[0] < 0.05 < lad[-1]


def test_matched_motion_partner_refuses_an_unmatched_control():
    r = E.matched_motion_partner({"a": 1.0, "b": 2.0}, target_motion=2.02)
    assert r["partner"] == "b"
    try:
        E.matched_motion_partner({"a": 1.0, "b": 5.0}, target_motion=2.0)
        raise SystemExit("A1 requires MATCHED realized motion, not the nearest available cell")
    except AssertionError as ex:
        assert "motion-unmatched" in str(ex)


# ---------------------------------------------------------------------------
# M-6 — the checkpoint format.
# ---------------------------------------------------------------------------
def test_format_roundtrip_is_exact_on_the_stable_block_and_improves_with_rank():
    rng = np.random.default_rng(13)
    m, n, k = 24, 32, 6
    Wb = rng.standard_normal((m, n))
    U = np.linalg.qr(rng.standard_normal((n, k)))[0]
    Wf = Wb + 0.05 * rng.standard_normal((m, n))
    lo = F.roundtrip(Wb, Wf, U, r=2)
    hi = F.roundtrip(Wb, Wf, U, r=16)
    assert hi["rel_error_delta"] < lo["rel_error_delta"]
    assert hi["delta_energy_captured"] > lo["delta_energy_captured"]
    assert lo["core_action_error"] < 1e-10, "the vault must be exact on the stable block"


def test_roundtrip_gate_fails_when_the_format_does_not_reconstruct():
    rng = np.random.default_rng(17)
    Wb = rng.standard_normal((16, 20))
    U = np.linalg.qr(rng.standard_normal((20, 4)))[0]
    Wf = Wb + rng.standard_normal((16, 20))
    rt = F.roundtrip(Wb, Wf, U, r=1)
    try:
        F.assert_roundtrip(rt, tol_total=1e-12, tol_core=1e-10)
        raise SystemExit("the round-trip gate must be able to fail")
    except AssertionError as ex:
        assert "does not reconstruct" in str(ex)


def test_bytes_breakdown_charges_the_family_parts_once():
    b1 = F.bytes_breakdown(m=2048, n=2048, k_block=64, r=8, n_finetunes=1)
    b24 = F.bytes_breakdown(m=2048, n=2048, k_block=64, r=8, n_finetunes=24)
    assert b24["total_bits"] - b1["total_bits"] == 23 * b1["motion_bits_per_ft"]
    # the format's ONE amortisation claim: the stable block is shipped once per family
    shared = F.bytes_breakdown(m=2048, n=2048, k_block=64, r=8, n_finetunes=24,
                               n_matrices_sharing_block=18)
    assert shared["total_bits"] < b24["total_bits"]
    assert abs((b24["family_bits"] - shared["family_bits"])
               - b1["basis_bits"] * (1 - 1 / 18)) < 1e-6
    # the conservative default charges the whole basis per matrix
    assert b1["n_matrices_sharing_block"] == 1


# ---------------------------------------------------------------------------
# M-5 — gradient compression.
# ---------------------------------------------------------------------------
def test_compression_arms_are_at_matched_bandwidth_and_exclusions_are_disjoint():
    rng = np.random.default_rng(19)
    g = rng.standard_normal((8, 32))
    sink = list(range(4))
    res = [C.compress(g, 40, arm=a, sink_coords=sink, seed=1) for a in C.ARMS]
    assert C.matched_bandwidth_check(res)["k"] == 40
    for r in res:
        assert int((r["sent"] != 0).sum()) == 40
    excl = C.excluded_columns("random_excluded", n_cols=32, sink_coords=sink, seed=1)
    assert len(set(excl) & set(sink)) == 0 and len(excl) == len(sink)


def test_error_feedback_conserves_what_was_not_sent():
    rng = np.random.default_rng(23)
    g = rng.standard_normal((4, 16))
    r = C.compress(g, 10, arm="naive", sink_coords=[0], seed=1)
    assert np.allclose(r["sent"] + r["residual"], g)
    r2 = C.compress(g, 10, arm="naive", sink_coords=[0], seed=1, residual=r["residual"])
    assert np.allclose(r2["sent"] + r2["residual"], g + r["residual"])


def test_sink_energy_share_is_measured_not_assumed():
    rng = np.random.default_rng(29)
    g = rng.standard_normal((8, 64))
    flat = C.sink_energy_share(g, list(range(8)))
    assert 0.5 < flat["ratio"] < 1.6
    g[:, :8] *= 10
    assert C.sink_energy_share(g, list(range(8)))["ratio"] > 5


def test_convergence_readout_requires_every_arm():
    curves = {a: ([0, 10], [3.0, 2.0]) for a in C.ARMS}
    r = C.convergence_readout(curves, at_step=5)
    assert abs(r["loss_by_arm"]["naive"] - 2.5) < 1e-9
    try:
        C.convergence_readout({"naive": ([0, 10], [3.0, 2.0])}, at_step=5)
        raise SystemExit("a missing control arm is a defect, not an omission")
    except AssertionError as ex:
        assert "missing arms" in str(ex)


# ---------------------------------------------------------------------------
# M-C1 / M-C2 — the assemblers.
# ---------------------------------------------------------------------------
def test_damage_forecast_has_no_default_law_parameters():
    try:
        AS.damage_forecast(1.0, coefficient=0.0, exponent=1.42)
        raise SystemExit("a law parameter of 0 must fail loud, not forecast 0 damage")
    except AssertionError as ex:
        assert "recomputed" in str(ex)
    assert abs(AS.damage_forecast(2.0, coefficient=1.0, exponent=2.0)["damage_forecast"] - 4) < 1e-9


def test_recipe_class_separates_low_rank_from_full_rank_deltas():
    rng = np.random.default_rng(31)
    full = rng.standard_normal((32, 32))
    lora = rng.standard_normal((32, 4)) @ rng.standard_normal((4, 32))
    assert not AS.recipe_class(full)["is_low_rank"]
    r = AS.recipe_class(lora)
    assert r["is_low_rank"] and r["numerical_rank"] == 4


def test_leave_one_out_reports_the_no_model_baseline_in_the_same_table():
    x = np.linspace(0, 1, 12).reshape(-1, 1)
    y = (3 * x.ravel() + 1)

    def fit_predict(tx, ty, testx):
        a = np.linalg.lstsq(np.c_[tx, np.ones(len(tx))], ty, rcond=None)[0]
        return float(np.r_[testx, 1.0] @ a)

    loo = AS.leave_one_out(x, y, fit_predict)
    assert loo["mae_model"] < 1e-8 and loo["mae_baseline"] > 0.5
    assert loo["factor_vs_baseline"] > 1e6
    w = AS.within_factor(loo, factor=2.0)
    assert w["model_better_fraction"] == 1.0


def test_merge_pre_check_requires_ten_pairs_and_ships_its_incumbent():
    rng = np.random.default_rng(37)
    U = np.linalg.qr(rng.standard_normal((16, 4)))[0]
    pairs = {f"p{i}": (rng.standard_normal((8, 16)), rng.standard_normal((8, 16)))
             for i in range(10)}
    rows = AS.merge_prediction_table(pairs, U)
    assert len(rows) == 10 and "norm_product_baseline" in rows[0]
    try:
        AS.merge_prediction_table({k: v for k, v in list(pairs.items())[:9]}, U)
        raise SystemExit("§4 M-C2 requires >=10 pairs")
    except AssertionError:
        pass


# ---------------------------------------------------------------------------
# M-10 / M-11 — the pure-numpy half.
# ---------------------------------------------------------------------------
def test_grid_inventory_refuses_an_incomplete_or_unbounded_declaration():
    good = {k: {"status": A.NEUTRALIZED, "how": "x"} for k in A.GRID_INVENTORY}
    assert A.audit_inventory(good)["n_neutralized"] == 6
    short = {k: v for k, v in good.items() if k != "rope"}
    try:
        A.audit_inventory(short)
        raise SystemExit("an unlisted symmetry breaker IS the leak the REFUTER predicts")
    except AssertionError as ex:
        assert "rope" in str(ex)
    unbounded = dict(good)
    unbounded["fp_rounding"] = {"status": A.DECLARED, "how": "bf16 rounding"}
    try:
        A.audit_inventory(unbounded)
        raise SystemExit("a DECLARED breaker must carry a numeric bound")
    except AssertionError as ex:
        assert "bound" in str(ex)


def test_arch_config_refuses_a_non_composable_switch_pair():
    A.ArchConfig(reserved_k=8, reserve_gamma_fixed=True)
    try:
        A.ArchConfig(norm="scalar", reserved_k=8, reserve_gamma_fixed=True)
        raise SystemExit("switch (a) has no meaning with a scalar gain and must be refused")
    except AssertionError as ex:
        assert "not composable" in str(ex)
    try:
        A.ArchConfig(writer_init_scale=0.5)
        raise SystemExit("switch (c) is s in {0, 0.1, 1}")
    except AssertionError:
        pass


def test_adoption_angles_separate_an_adopted_block_from_the_haar_band():
    n, k = 64, 8
    E_ = np.zeros((n, k))
    E_[np.arange(k), np.arange(k)] = 1.0
    adopted = A.adoption_angles(E_, reserved_k=k)
    assert adopted["mean_angle_rad"] < 1e-8
    from infra.pg import humboldt_core as H
    ignored = A.adoption_angles(H.haar_subspace(n, k, 41), reserved_k=k)
    assert ignored["haar_band_p2.5"] <= ignored["mean_angle_rad"] <= ignored["haar_band_p97.5"]


def test_leverage_in_out_is_one_at_chance():
    from infra.pg import humboldt_core as H
    r = A.leverage_in_out(H.haar_subspace(512, 32, 3), reserved_k=64)
    assert 0.5 < r["ratio"] < 1.7


def test_equal_val_loss_match_refuses_non_overlapping_curves():
    r = A.equal_val_loss_match([1, 2, 3], [3.0, 2.0, 1.0], [1, 2, 3], [3.5, 2.5, 1.5])
    assert r["overlap_lo"] == 1.5 and r["overlap_hi"] == 3.0
    assert len(r["steps_a"]) == len(r["steps_b"]) == 5
    try:
        A.equal_val_loss_match([1, 2], [3.0, 2.5], [1, 2], [2.0, 1.5])
        raise SystemExit("non-overlapping loss curves cannot be compared at equal loss")
    except AssertionError as ex:
        assert "do not overlap" in str(ex)


def test_max_abs_activation_reports_inside_and_outside_separately():
    X = np.zeros((4, 16))
    X[:, 2] = 100.0
    X[:, 10] = 3.0
    r = A.max_abs_activation(X, reserved_k=8)
    assert r["max_abs_inside"] == 100.0 and r["max_abs_outside"] == 3.0


# ---------------------------------------------------------------------------
# M-Q — the gauge.
# ---------------------------------------------------------------------------
def test_fold_sets_the_gain_to_one_and_preserves_the_function():
    from infra.pg import mercator_gauge as G
    from infra.pg.gauge_fix import rmsnorm
    rng = np.random.default_rng(3)
    d, m = 12, 7
    weight = rng.standard_normal(d) * 0.2          # Gemma stores weight; applied gain is 1+weight
    W = rng.standard_normal((m, d))
    x = rng.standard_normal(d)
    new_w, (W2,), st = G.fold_gain_into(weight, [W])
    assert np.allclose(new_w, 0.0)                  # gain 1 means weight 0 for Gemma
    assert st["used_positive_helper"] and st["n_negative_gain"] == 0
    y0 = W @ rmsnorm(x, G.gemma_effective_gain(weight))
    y1 = W2 @ rmsnorm(x, G.gemma_effective_gain(new_w))
    assert np.abs(y0 - y1).max() < 1e-10


def test_fold_handles_negative_and_zero_gains_which_pg1_actually_has():
    """The positive-diagonal helper asserts D > 0; real PG1 gains are not all positive."""
    from infra.pg import mercator_gauge as G
    from infra.pg.gauge_fix import rmsnorm
    rng = np.random.default_rng(9)
    d, m = 10, 6
    weight = rng.standard_normal(d) * 0.2
    weight[0] = -2.0        # effective gain 1 + w = -1.0  (negative)
    weight[1] = -1.0        # effective gain 0.0           (annihilated coordinate)
    W = rng.standard_normal((m, d))
    x = rng.standard_normal(d)
    new_w, (W2,), st = G.fold_gain_into(weight, [W])
    assert not st["used_positive_helper"]
    assert st["n_negative_gain"] == 1 and st["n_zero_gain"] == 1
    # the identity is still EXACT for any sign
    y0 = W @ rmsnorm(x, G.gemma_effective_gain(weight))
    y1 = W2 @ rmsnorm(x, G.gemma_effective_gain(new_w))
    assert np.abs(y0 - y1).max() < 1e-10
    assert np.allclose(W2[:, 1], 0.0)               # a zero gain gives a zero column


def test_fold_plan_refuses_an_undeclared_norm_and_names_the_unfoldable_case():
    from infra.pg import mercator_gauge as G
    pre = G.plan_fold(["input_layernorm", "post_attention_layernorm"])
    assert pre["fully_foldable"] and pre["architecture"] == "pre-norm"
    post = G.plan_fold(["input_layernorm", "post_attention_layernorm",
                        "pre_feedforward_layernorm", "post_feedforward_layernorm"])
    assert not post["fully_foldable"] and "no matrix consumer" in post["consequence"]
    try:
        G.plan_fold(["input_layernorm", "some_new_norm"])
        raise SystemExit("an undeclared norm must fail loud, not be skipped")
    except AssertionError as ex:
        assert "some_new_norm" in str(ex)


def test_rotation_is_a_gauge_on_a_reader_writer_pair():
    from infra.pg import mercator_gauge as G
    rng = np.random.default_rng(5)
    d = 16
    R = G.hadamard_rotation(d)
    assert G.orthogonality_error(R) < 1e-10
    Wr = rng.standard_normal((5, d))               # reader: y = Wr x
    Ww = rng.standard_normal((d, 5))               # writer: x + Ww a
    x, a = rng.standard_normal(d), rng.standard_normal(5)
    xr = R @ x
    assert np.abs(G.rotate_pair(Wr, "reader", R) @ xr - Wr @ x).max() < 1e-10
    assert np.abs(G.rotate_pair(Ww, "writer", R) @ a - R @ (Ww @ a)).max() < 1e-10


def test_permutation_gauge_is_orthogonal_and_commutes_with_elementwise_ops():
    """The diagnostic that separates 'an unfolded diagonal remains' from 'a tensor is unrotated'."""
    from infra.pg import mercator_gauge as G
    rng = np.random.default_rng(4)
    d = 16
    P = G.permutation_rotation(d, 7)
    assert G.orthogonality_error(P) < 1e-12
    assert np.allclose(P.sum(axis=0), 1) and np.allclose(P.sum(axis=1), 1)
    x, gamma = rng.standard_normal(d), rng.standard_normal(d)
    # a permutation commutes with an elementwise gain (up to permuting the gain itself);
    # a dense rotation does not — which is exactly what makes the pair diagnostic
    assert np.allclose(P @ (gamma * x), (P @ gamma) * (P @ x))
    Hd = G.hadamard_rotation(d)
    assert not np.allclose(Hd @ (gamma * x), (Hd @ gamma) * (Hd @ x))


def test_align_rotation_puts_the_sink_on_the_first_k_axes():
    from infra.pg import mercator_gauge as G
    from infra.pg import humboldt_core as H
    U = H.haar_subspace(32, 4, 11)
    R = G.align_rotation(U, seed=2)
    img = R @ U
    assert np.abs(img[4:]).max() < 1e-8            # nothing outside the first k axes
    assert G.orthogonality_error(R) < 1e-8


def test_function_preservation_gate_can_fail():
    from infra.pg import mercator_gauge as G
    a = np.array([[1.0, 2.0, 3.0]])
    G.assert_function_preserved(G.function_preservation(a, a + 1e-6), what="fold")
    try:
        G.assert_function_preserved(G.function_preservation(a, a + 1.0), what="fold")
        raise SystemExit("a gauge that changes the function must fail loud")
    except AssertionError as ex:
        assert "is not a gauge" in str(ex)


def test_noise_floor_gate_is_relative_but_still_fails_on_a_broken_gauge():
    """The absolute 1e-4 is tighter than fp32 at PG1's logit scale, so the binding comparison is
    against the MEASURED identity-path floor — and that comparison must still be able to fail."""
    from infra.pg import mercator_gauge as G
    a = np.array([[10.0, 20.0, 53.0]])
    floor = 1e-4                                    # what the identity path itself costs
    ok = G.assert_function_preserved(G.function_preservation(a, a + 7.7e-4), what="fold",
                                     noise_floor=floor)
    assert ok["passed"] and ok["ratio_to_noise_floor"] < 10
    assert not ok["passes_absolute_tol"], "the spec's absolute gate is still recorded per row"
    try:                                            # 100x the floor is not dtype noise
        G.assert_function_preserved(G.function_preservation(a, a + 1e-2), what="fold",
                                    noise_floor=floor)
        raise SystemExit("a gauge far above the arithmetic floor must fail loud")
    except AssertionError as ex:
        assert "beyond what the dtype explains" in str(ex)
    try:                                            # a bit-exact identity path means no floor
        G.assert_function_preserved(G.function_preservation(a, a), what="fold", noise_floor=0.0)
        raise SystemExit("a zero floor must be refused, not divided by")
    except AssertionError as ex:
        assert "bit-exact" in str(ex)


# ---------------------------------------------------------------------------
# M-A3 — the settling diagnostic.
# ---------------------------------------------------------------------------
def test_settling_ratio_separates_agreeing_from_cancelling_gradients():
    """Var(g)/mean(g)^2 is the gradient sink in one number: large gradient that cancels has a large
    ratio, large gradient that agrees has a small one."""
    from infra.aws_pg.jobs.mercator_a3 import settling_ratio, subspace_mean
    n_steps = 10
    # coordinate 0: consistent (sum large, sum-of-squares just above sum^2/n) -> small ratio
    # coordinate 1: cancelling (sum near zero, sum-of-squares large)          -> huge ratio
    # coordinate 2: dead (never any gradient)                                 -> undefined
    from infra.aws_pg.jobs.mercator_a3 import per_coordinate
    # ELEMENTWISE, as Adam sees it: an (out=2, in=3) weight matrix.
    #   column 0: consistent          -> small ratio
    #   column 1: cancelling          -> huge ratio
    #   column 2: dead                -> undefined
    s1 = np.array([[10.0, 0.1, 0.0], [10.0, -0.1, 0.0]])
    s2 = np.array([[10.5, 50.0, 0.0], [10.5, 50.0, 0.0]])
    r = settling_ratio(s1, s2, n_steps)
    assert r["ratio"].shape == (2, 3)
    coord = per_coordinate(r["ratio"])
    assert coord[0] < coord[1] / 100, (coord[0], coord[1])
    assert not np.isfinite(coord[2]) and r["n_dead"] == 2
    # THE POINT OF GOING ELEMENTWISE: collapsing to column SUMS first hides the cancellation.
    # Column 1's elements are +0.1 and -0.1, so their sum is 0 and the column-sum route would
    # call the column dead instead of noisy — the failure mode of this arm's first run.
    col_sum = settling_ratio(s1.sum(axis=0), s2.sum(axis=0), n_steps)
    assert not np.isfinite(col_sum["ratio"][1]), "column sums lose the cancellation"
    assert np.isfinite(coord[1]), "the elementwise route keeps it"
    # a subspace of only-undefined coordinates must fail loud, not average to nan
    U = np.zeros((3, 1))
    U[2, 0] = 1.0
    try:
        subspace_mean(coord, U)
        raise SystemExit("a subspace of only-undefined coordinates must fail loud")
    except AssertionError as ex:
        assert "undefined" in str(ex)
    U2 = np.zeros((3, 1))
    U2[0, 0] = 1.0
    assert abs(subspace_mean(coord, U2) - coord[0]) < 1e-12


def test_epsilon_position_reports_where_sqrt_v_sits():
    """If sqrt(v) is comparable to Adam's epsilon, epsilon limits the motion, not the second
    moment — and the mechanism claim would be about a hyperparameter."""
    from infra.aws_pg.jobs.mercator_a3 import epsilon_position
    ep = epsilon_position(np.array([1e-9, 1e-9, 1e-3, 1e-3]))
    assert ep["fraction_below_eps"]["1e-08"] == 0.5
    assert ep["fraction_below_eps"]["1e-06"] == 0.5
    assert ep["sqrt_v_min"] == 1e-9


# ---------------------------------------------------------------------------
# M-A2 — the breathe valve.
# ---------------------------------------------------------------------------
def test_gain_ladder_continues_w1s_and_meets_the_dose_rule():
    from infra.pg import mercator_core as M
    lad = TR.gain_ladder(8.0, 5)
    assert lad == [8.0, 16.0, 32.0, 64.0, 128.0]
    M.check_dose_grid(lad, arm="M-A2")            # log-spaced, >= 4 doses
    try:
        TR.gain_ladder(8.0, 3)
        raise SystemExit("a 3-dose ladder must be refused")
    except AssertionError:
        pass


def test_damage_onset_returns_a_bracket_and_admits_no_onset():
    gains = [8.0, 16.0, 32.0, 64.0]
    r = TR.damage_onset(gains, [0.01, 0.02, 0.40, 0.90], band_hi=0.05)
    assert r["onset"] == 32.0 and r["bracket"] == [16.0, 32.0]
    # a ladder that never leaves the band must say so, not invent an onset at the last rung
    flat = TR.damage_onset(gains, [0.01, 0.02, 0.03, 0.04], band_hi=0.05)
    assert flat["onset"] is None and "no damage onset" in flat["reading"]
    assert flat["max_gain_tested"] == 64.0


def test_breathe_valve_at_gain_one_is_an_exact_no_op():
    """The 'off' position must not perturb the run, or every dose is confounded with the valve."""
    U = {"k": np.eye(4)[:, :2]}
    v = TR.BreatheValve.__new__(TR.BreatheValve)   # construct without torch
    v.gain, v.U = 1.0, U
    assert v.apply({}) == 0


def test_power_law_pickup_needs_its_scale_and_scores_the_prediction():
    motion = np.array([1.0, 2.0, 4.0])
    exact = 0.3 * motion ** 1.42
    r = TR.power_law_pickup(motion, exact, exponent=1.42, coefficient=0.3)
    assert r["r2_against_referee_law"] > 0.999
    assert abs(r["mean_ratio_observed_over_predicted"] - 1.0) < 1e-9
    off = TR.power_law_pickup(motion, exact * 3.0, exponent=1.42, coefficient=0.3)
    assert abs(off["mean_ratio_observed_over_predicted"] - 3.0) < 1e-9




def test_damage_onset_separates_a_crossing_from_a_sustained_onset():
    """M-A2's real series: one rung crosses the band by 0.0008 and falls back below at the next
    two. First-crossing calls that an onset; sustained correctly calls it none (defect M-D22)."""
    gains = [8.0, 16.0, 32.0, 64.0, 128.0]
    wobble = [0.9999, 1.0157, 1.0088, 1.0030, 1.0127]
    r = TR.damage_onset(gains, wobble, band_hi=1.0149248)
    assert r["onset"] == 16.0, r
    assert r["sustained_onset"] is None, r
    assert r["n_rungs_over_band"] == 1, r
    assert "does not stay above it" in r["reading"], r
    # and a real onset must report the same rung on both readings
    real = [1.00, 1.00, 1.05, 1.20, 1.60]
    r2 = TR.damage_onset(gains, real, band_hi=1.02)
    assert r2["onset"] == 32.0 and r2["sustained_onset"] == 32.0, r2
    assert r2["sustained_bracket"] == [16.0, 32.0], r2


def test_valve_saturation_scores_full_delivery_and_full_absorption():
    gains = [8.0, 16.0, 32.0, 64.0, 128.0]
    full = TR.valve_saturation(gains, gains)                       # motion tracks the request
    assert abs(full["log_log_slope"] - 1.0) < 1e-9, full
    assert abs(full["delivered_over_requested_at_max_gain"] - 1.0) < 1e-12, full
    absorbed = TR.valve_saturation(gains, [3.0] * 5)               # nothing gets through
    assert abs(absorbed["log_log_slope"]) < 1e-9, absorbed
    # M-A2's measured series: heavy absorption, but not total
    measured = TR.valve_saturation(gains, [5.308, 6.481, 6.843, 8.606, 8.948])
    assert 0.1 < measured["log_log_slope"] < 0.3, measured
    assert measured["delivered_over_requested_at_max_gain"] < 0.1, measured



def test_block_adam_ends_of_the_ladder_are_the_two_known_rules():
    """block=1 must be ExactAdam and block=n must be ScalarDenomAdam. That identity is what makes
    M-10's switch a LADDER between two measured points rather than a third unrelated rule."""
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        return                                    # instance-only; the job asserts it there too
    import torch
    from infra.pg import darwin_train as DT
    from infra.pg import mercator_pretrain as PT
    g = torch.Generator().manual_seed(20260815)
    grads = [torch.randn(8, 16, generator=g) for _ in range(4)]

    def run(make):
        p = torch.nn.Parameter(torch.zeros(8, 16))
        opt = make([p])
        for gr in grads:
            p.grad = gr.clone()
            opt.step()
        return p.detach().clone()

    a = run(lambda ps: DT.ExactAdam(ps, lr=1e-2))
    b = run(lambda ps: PT.BlockAdam(ps, lr=1e-2, block=1))
    assert torch.allclose(a, b, atol=1e-7), (a - b).abs().max()
    c = run(lambda ps: __import__("infra.pg.mercator_train", fromlist=["x"])
            .ScalarDenomAdam(ps, lr=1e-2))
    d = run(lambda ps: PT.BlockAdam(ps, lr=1e-2, block=16))
    assert torch.allclose(c, d, atol=1e-6), (c - d).abs().max()


def test_adoption_angle_reports_the_independent_level_not_just_the_overlap():
    from infra.pg import humboldt_core as HC
    from infra.pg import mercator_pretrain as PT
    U = PT.sanctioned_block(64, 8, seed=0)
    same = PT.adoption_angle(U, U)
    assert abs(same["mean_cos2"] - 1.0) < 1e-9, same
    assert abs(same["independent_level"] - 8 / 64) < 1e-12, same
    far = PT.adoption_angle(HC.haar_subspace(64, 8, 7), U)
    # a random block overlaps a coordinate block at about the independent level, not at zero
    assert 0.2 * far["independent_level"] < far["mean_cos2"] < 5 * far["independent_level"], far
    band = PT.haar_adoption_band(U, n_draws=8, seed=100)
    assert band["lo"] <= band["mean"] <= band["hi"] and band["n_draws"] == 8, band


def test_consensus_from_snapshots_uses_the_same_normalisation_as_the_census():
    """A cell whose delta is huge must not dominate: each term is divided by its own Frobenius
    energy, exactly as census_grams does, so the Gram is a consensus and not a sum of scales."""
    import numpy as np
    from infra.pg import mercator_pretrain as PT
    rng = np.random.default_rng(3)
    n, k = 24, 4
    base = rng.standard_normal((10, n))
    snaps = [{("q", 0): base}]
    for i in range(4):
        step = rng.standard_normal((10, n)) * (10.0 ** i)      # wildly different scales
        snaps.append({("q", 0): snaps[-1][("q", 0)] + step})
    got = PT.consensus_from_snapshots(snaps, [("q", 0)], k)
    U = got[("q", 0)]["U"]
    assert U.shape == (n, k) and got[("q", 0)]["n_terms"] == 4
    assert np.abs(U.T @ U - np.eye(k)).max() < 1e-8


def test_format_incumbent_is_byte_matched_and_two_sided():
    """M-D40. The format's price only means something against an incumbent it is tied with.

    Two ways to get this wrong, and both were live in the shipped jobs: an fp16-backbone incumbent
    makes the format ~3x cheaper on backbone quantization alone (the gate cannot FAIL), and a
    same-bits backbone at equal RANK is strictly cheaper than the format (the gate cannot PASS).
    The byte-matched incumbent spends the format's surplus on motion rank, so capability decides.
    """
    from infra.pg import mercator_format as MF
    m = n = 2048
    kb, ns, nft = 64, 18, 8
    ranks = [8, 16, 32, 64, 128]

    # the two broken framings, stated as facts so a regression trips here
    for r in ranks:
        b = MF.bytes_breakdown(m=m, n=n, k_block=kb, r=r, n_finetunes=nft, bits=4,
                               n_matrices_sharing_block=ns)
        assert b["bits_ratio_vs_fp16"] < 1.0, "the fp16 incumbent must look free — that is the bug"
        assert b["bits_ratio_vs_matched"] > 1.0, "at equal rank the format is the dearer scheme"

    aff = {r: MF.affordable_incumbent_rank(r, m=m, n=n, k_block=kb, n_finetunes=nft, bits=4,
                                           n_matrices_sharing_block=ns) for r in ranks}
    assert all(aff[r] > r for r in ranks), f"the surplus must buy real extra rank, got {aff}"

    # a LOSS readout (M-6's polarity) and a SCORE readout (M-C3's) must both land either way
    for higher_better, cap in ((False, {r: 1.0 - r / 300 for r in ranks}),
                               (True, {r: 0.3 + r / 1000 for r in ranks})):
        sign = 1.0 if higher_better else -1.0
        for helps in (True, False):
            d = sign * (0.02 if helps else -0.02)
            inc = {aff[r]: cap[r] - d for r in ranks}
            curve = MF.capability_vs_bytes(ranks, cap, inc, m=m, n=n, k_block=kb,
                                           n_finetunes=nft, bits=4,
                                           n_matrices_sharing_block=ns,
                                           higher_is_better=higher_better)
            wins = MF.pareto_wins(curve)
            assert (wins == ranks) if helps else (wins == []), (
                f"higher_is_better={higher_better} helps={helps} gave {wins}")
        # a dead tie goes to the incumbent, never to the format
        tie = {aff[r]: cap[r] for r in ranks}
        assert MF.pareto_wins(MF.capability_vs_bytes(
            ranks, cap, tie, m=m, n=n, k_block=kb, n_finetunes=nft, bits=4,
            n_matrices_sharing_block=ns, higher_is_better=higher_better)) == []


def test_format_refuses_an_unmeasured_incumbent():
    """The incumbent's capability cannot be assumed: it is the half that decides the arm."""
    from infra.pg import mercator_format as MF
    ranks = [8, 16]
    cap = {8: 1.0, 16: 0.9}
    try:
        MF.capability_vs_bytes(ranks, cap, {8: 1.0}, m=2048, n=2048, k_block=64, n_finetunes=8,
                               bits=4, n_matrices_sharing_block=18)
    except AssertionError as e:
        assert "affords rank" in str(e), str(e)
    else:
        raise AssertionError("a curve was built without the incumbent's measured capability")


if __name__ == "__main__":
    # THE COUNT IS CHECKED AGAINST THE SOURCE. Appending a test below this block defines it after
    # the block has already run, so it is silently skipped — which is what happened to the two
    # damage-onset tests (defect M-D23). Counting `def test_` in the file and comparing makes that
    # impossible to miss instead of merely unlikely.
    import pathlib as _pl
    import re as _re
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    in_source = len(_re.findall(r"^def (test_\w+)", _pl.Path(__file__).read_text(), _re.M))
    assert in_source == len(fns), (
        f"{in_source} tests are defined in this file but only {len(fns)} were collected — a test "
        f"defined BELOW the __main__ block never runs. Move it above.")
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
