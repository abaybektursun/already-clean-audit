"""Weight-free tests for the CEQ instruments (ADDENDUM arm W1-N4).

Run: PYTHONPATH=. python3 infra/pg/tests/test_darwin_ceq.py

These re-derive, against this repo's implementation, the properties the addendum's dry-run suite
established — and, just as importantly, they pin the two claims the dry-run KILLED, so the code
cannot quietly start asserting them again.
"""
import numpy as np

from infra.pg import darwin_ceq as C
from infra.pg import humboldt_core as H


def _setup(m=48, n=128, k=8, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((m, n))
    U = H.haar_subspace(n, k, seed + 77)
    return W, U, rng


# ---------------------------------------------------------------------------
# T1 / T3 — what the arm MAY assert.
# ---------------------------------------------------------------------------
def test_vault_split_annihilates_the_core_exactly():
    W, U, _ = _setup()
    A, bulk = C.vault_split(W, U)
    assert np.abs(bulk @ U).max() < 1e-12
    assert np.allclose(A, W @ U)
    assert np.allclose(A @ U.T + bulk, W)                    # the split is lossless


def test_core_action_is_exact_under_brutal_bulk_quantization():
    """T1: routing makes the core action exact even at 3 bits; the naive form leaks."""
    W, U, _ = _setup()
    A, bulk = C.vault_split(W, U)
    Q = C.quantize_uniform(bulk, bits=3)
    assert C.core_action_error(W, U, A, Q) < 1e-12
    # The naive comparison: quantize the whole matrix and read its core action.
    naive = C.quantize_uniform(W, bits=3)
    leak = np.linalg.norm(naive @ U - W @ U) / np.linalg.norm(W @ U)
    assert leak > 1e-3, leak                                  # a real, large leak


def test_routing_projection_is_load_bearing():
    """Dropping the trailing (x - U Uᵀx) projection destroys exactness — so the test proves the
    implementation choice matters rather than taking the docstring's word for it."""
    W, U, rng = _setup()
    A, bulk = C.vault_split(W, U)
    Q = C.quantize_uniform(bulk, bits=3)
    x = U[:, 0]                                               # a pure core-direction input
    good = C.ceq_apply(A, Q, U, x)
    bad = A @ (U.T @ x) + Q @ x                               # no perp projection
    assert np.linalg.norm(good - W @ x) < 1e-12
    assert np.linalg.norm(bad - W @ x) > 1e-6


def test_rotated_block_agrees_with_projector_routing():
    """T3, re-checked here rather than assumed: the two forms must agree on the core action."""
    W, U, _ = _setup()
    A, bulk = C.vault_split(W, U)
    Q = C.quantize_uniform(bulk, bits=4)
    Wr = C.rotated_block_quantize(W, U, bits=4)
    assert np.linalg.norm(Wr @ U - W @ U) / np.linalg.norm(W @ U) < 1e-10
    assert C.core_action_error(W, U, A, Q) < 1e-12


# ---------------------------------------------------------------------------
# The claims the dry-run KILLED. These tests exist so the code cannot re-acquire them.
# ---------------------------------------------------------------------------
def test_end_to_end_lane_purity_is_NOT_claimed_and_is_only_first_order():
    """T4b. Two stacked layers: protection helps, but nothing like exactly. If this ever came
    out exact, the module would be computing something other than what it claims."""
    rng = np.random.default_rng(3)
    n, m, k = 96, 96, 6
    U = H.haar_subspace(n, k, 5)
    W1, W2 = rng.standard_normal((m, n)), rng.standard_normal((m, n))
    x = rng.standard_normal((64, n))
    A1, b1 = C.vault_split(W1, U)
    A2, b2 = C.vault_split(W2, U)
    Q1, Q2 = C.quantize_uniform(b1, 4), C.quantize_uniform(b2, 4)
    exact = (W2 @ (W1 @ x.T)).T
    routed = C.ceq_apply(A2, Q2, U, C.ceq_apply(A1, Q1, U, x))
    ambient = (C.quantize_uniform(W2, 4) @ (C.quantize_uniform(W1, 4) @ x.T)).T
    e_routed = np.linalg.norm((routed - exact) @ U)
    e_ambient = np.linalg.norm((ambient - exact) @ U)
    assert e_routed > 0, "end-to-end core error must NOT be zero — that claim was refuted (T4b)"
    assert e_ambient / e_routed < 20, "a >20x end-to-end ratio would contradict the dry-run"
    # No function in the module reports an 'end-to-end purity' statistic.
    assert not [f for f in dir(C) if "purity" in f.lower() or "end_to_end" in f.lower()]


# ---------------------------------------------------------------------------
# T6 — the leak law, which the graduation clause turns on.
# ---------------------------------------------------------------------------
def test_leak_law_is_exact_and_scales_linearly_over_decades():
    W, U, rng = _setup()
    ratios = []
    for scale in (1e-4, 1e-3, 1e-2, 1e-1):
        dW = scale * rng.standard_normal(W.shape)
        r = C.leak_law_check(W, W + dW, U)
        assert r["within_2x"]
        ratios.append(r["ratio"])
    assert max(abs(x - 1.0) for x in ratios) < 1e-9, ratios     # exact, not merely proportional


def test_perfect_avoidance_makes_vault_reuse_exact():
    """A fine-tune that never touches the vaulted subspace reuses it at zero cost."""
    W, U, rng = _setup()
    dW = rng.standard_normal(W.shape)
    dW = dW - (dW @ U) @ U.T                                   # project the update off the core
    r = C.leak_law_check(W, W + dW, U)
    assert r["predicted_leak_norm"] < 1e-10 and r["realized_vault_error"] < 1e-10


# ---------------------------------------------------------------------------
# T5 / T7.
# ---------------------------------------------------------------------------
def test_union_basis_rank_is_shared_plus_private():
    n = 64
    shared = H.haar_subspace(n, 4, 1)
    a = np.linalg.qr(np.concatenate([shared, H.haar_subspace(n, 3, 2)], axis=1))[0][:, :7]
    b = np.linalg.qr(np.concatenate([shared, H.haar_subspace(n, 3, 3)], axis=1))[0][:, :7]
    Uu = C.union_basis([a, b])
    assert Uu.shape[1] == 10                                   # 4 shared + 3 + 3
    assert np.abs(Uu.T @ Uu - np.eye(Uu.shape[1])).max() < 1e-10
    for B in (a, b):                                           # both cores lie inside the union
        assert np.linalg.norm(B - Uu @ (Uu.T @ B)) < 1e-9
    # Disjoint worst case: rank is the sum, which is what the arm must budget for.
    assert C.union_basis([H.haar_subspace(n, 5, 9), H.haar_subspace(n, 5, 10)]).shape[1] == 10


def test_routed_activations_are_blind_to_spiky_core_content():
    """T7: build activations with 10x mass along the core and check the routed path wins."""
    rng = np.random.default_rng(11)
    n, k = 128, 8
    U = H.haar_subspace(n, k, 21)
    X = rng.standard_normal((256, n))
    X = X + 10.0 * (rng.standard_normal((256, k)) @ U.T)       # spiky core content
    r = C.routed_activation_error(X, U, bits=8)
    assert r["ratio"] > 2.0, r
    # Control: with no spiky core content the advantage must largely vanish.
    flat = C.routed_activation_error(rng.standard_normal((256, n)), U, bits=8)
    assert flat["ratio"] < r["ratio"]


# ---------------------------------------------------------------------------
# Matched bit budget — the rule that keeps the comparison honest.
# ---------------------------------------------------------------------------
def test_matched_budget_gives_the_column_list_arm_a_wider_list():
    """A k-vault stores m*k + n*k fp16 values; a k-column list stores only m*k. Arm b must get
    the bigger list or CEQ wins on budget rather than on mechanism."""
    m, n, k = 2048, 2048, 32
    assert C.matched_k_for_vault(m=m, n=n, k_vault=k) == 64
    assert (C.fp16_parameter_count(m=m, n=n, k=64, scheme="column_list")
            == C.fp16_parameter_count(m=m, n=n, k=k, scheme="vault"))


def test_storage_accounting_charges_the_vault_for_its_basis():
    m, n, k = 512, 512, 32
    uni = C.storage_bits(m=m, n=n, k=k, scheme="uniform")
    col = C.storage_bits(m=m, n=n, k=k, scheme="column_list")
    vau = C.storage_bits(m=m, n=n, k=k, scheme="vault")
    rot1 = C.storage_bits(m=m, n=n, k=k, scheme="rotated_block", n_layers_sharing_rotation=1)
    rot99 = C.storage_bits(m=m, n=n, k=k, scheme="rotated_block", n_layers_sharing_rotation=108)
    assert abs(uni["bits_per_weight"] - 4.0) < 1e-9
    assert col["bits_per_weight"] > uni["bits_per_weight"]        # fp16 columns cost more
    assert vau["fp16_values"] == m * k + n * k                    # the basis IS charged
    assert vau["bits_per_weight"] > col["bits_per_weight"]
    # The rotation amortises across the matrices that share it — arm e's real advantage.
    assert rot99["bits_per_weight"] < rot1["bits_per_weight"]
    assert rot99["shared_rotation_bits"] < rot1["shared_rotation_bits"]


def test_quantizer_preserves_exact_zeros():
    """Symmetric quantization must leave W_bulk's exact zeros alone; an asymmetric zero-point
    would smear them and silently break the exactness claim."""
    W, U, _ = _setup()
    _A, bulk = C.vault_split(W, U)
    Q = C.quantize_uniform(bulk, bits=4)
    z = np.zeros((3, bulk.shape[1]))
    assert np.abs(C.quantize_uniform(z, 4)).max() == 0.0
    assert np.isfinite(Q).all()


def test_fp16_column_arm_keeps_those_columns_bit_exact_in_fp16():
    W, U, _ = _setup()
    cols = [0, 5, 9]
    Q = C.quantize_with_fp16_columns(W, cols, bits=4)
    assert np.allclose(Q[:, cols], np.asarray(W[:, cols], dtype=np.float16).astype(np.float64))
    other = [c for c in range(W.shape[1]) if c not in cols]
    assert not np.allclose(Q[:, other], W[:, other])              # the rest really is quantized


# ---------------------------------------------------------------------------
# The frozen graduation clause.
# ---------------------------------------------------------------------------
def test_ceq_graduation_clause_all_paths():
    lo, hi = 9.9, 10.1                                            # CI half-width 0.1
    beat = C.ceq_graduates(ppl_ceq=10.0, ppl_ceq_ci_lo=lo, ppl_ceq_ci_hi=hi,
                           ppl_incumbent_b=10.5, leak_within_2x=False)
    assert beat["graduates"] and beat["path"] == "beats_b"
    tie_ok = C.ceq_graduates(ppl_ceq=10.0, ppl_ceq_ci_lo=lo, ppl_ceq_ci_hi=hi,
                             ppl_incumbent_b=10.05, leak_within_2x=True)
    assert tie_ok["graduates"] and tie_ok["path"] == "tie_plus_leak_law"
    tie_no = C.ceq_graduates(ppl_ceq=10.0, ppl_ceq_ci_lo=lo, ppl_ceq_ci_hi=hi,
                             ppl_incumbent_b=10.05, leak_within_2x=False)
    assert not tie_no["graduates"] and tie_no["path"] == "neither"
    worse = C.ceq_graduates(ppl_ceq=11.0, ppl_ceq_ci_lo=10.9, ppl_ceq_ci_hi=11.1,
                            ppl_incumbent_b=10.0, leak_within_2x=True)
    assert not worse["graduates"]                                 # lower ppl is better
    assert worse["signed_delta_ppl"] < 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
