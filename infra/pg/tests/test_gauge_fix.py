"""Weight-free CPU dry-run gate for GAUGE-FIX (P2) — §12.1 P2 suite.

NO model, NO GPU — tiny planted numpy layers only. Proves the load-bearing science before any GPU/CPU
weight-run: (1) every symmetry generator computes the IDENTICAL function (the frozen catalogue
deliverable — attention checked with RoPE on and off); (2) the aligners RECOVER a known planted gauge
(γ≈1) and the identity check gives γ=0 exactly; (3) γ is bounded in [0,1] and a structureless
(spectrum-matched) Δ yields only a small overfit γ (the δ_GF floor); (4) the summary + decision
predicates. Run remotely: python -m pytest infra/pg/tests/test_gauge_fix.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_fix as GF


def _mlp_base(m=5, n=4, hidden=9, seed=0):
    rng = np.random.default_rng(seed)
    return {"gate_proj": rng.standard_normal((hidden, n)), "up_proj": rng.standard_normal((hidden, n)),
            "down_proj": rng.standard_normal((m, hidden))}


def _attn_base(n_heads=4, n_kv=2, head_dim=4, d_model=6, seed=0):
    rng = np.random.default_rng(seed)
    return {"q_proj": rng.standard_normal((n_heads * head_dim, d_model)),
            "k_proj": rng.standard_normal((n_kv * head_dim, d_model)),
            "v_proj": rng.standard_normal((n_kv * head_dim, d_model)),
            "o_proj": rng.standard_normal((d_model, n_heads * head_dim))}


# ---------------------------------------------------------------------------
# (1) the frozen symmetry catalogue — each generator is function-preserving.
# ---------------------------------------------------------------------------
def test_catalogue_dry_run_all_function_preserving():
    dry = GF.catalogue_dry_run()
    assert set(dry) == {"mlp_hidden_perm", "attn_head_perm", "norm_scale_transfer", "residual_channel_perm"}
    for name, d in dry.items():
        assert d < 1e-6, f"{name} changed the function: max|Δ|={d}"


def test_mlp_permutation_is_exact_identity_of_function():
    assert GF.check_mlp_perm_equivariant(m=6, n=5, hidden=11, seed=3) < 1e-10


def test_attn_head_permutation_preserves_function_with_and_without_rope():
    # MQA (n_kv=1) and true GQA (n_kv=2) both preserve the attention function, RoPE on and off.
    assert GF.check_attn_perm_equivariant(n_heads=8, n_kv=1, head_dim=4, d_model=8, seed=4) < 1e-9
    assert GF.check_attn_perm_equivariant(n_heads=6, n_kv=3, head_dim=4, d_model=6, seed=5) < 1e-9


def test_norm_scale_transfer_preserves_function():
    assert GF.check_norm_transfer_equivariant(d=8, n_out=5, seed=6) < 1e-10


def test_residual_channel_perm_preserves_function():
    assert GF.check_residual_perm_equivariant(d=7, seed=7) < 1e-10


def test_attn_head_perm_indices_keep_query_with_kv_partner():
    # a query head must stay paired with its kv head: new query head (G,q) sources old (kv_src[G], .).
    group = 2
    kv_src = np.array([1, 0])                      # swap the two kv groups
    within = np.array([[0, 1], [1, 0]])
    q_src, kv = GF.attn_head_perm_indices(kv_src, within, group)
    # new group 0 sources old group 1 → its query heads come from old heads {2,3}.
    assert set(q_src[0:2].tolist()) == {2, 3}
    assert set(q_src[2:4].tolist()) == {0, 1}
    assert kv.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# (2) aligners — planted-gauge recovery + identity check.
# ---------------------------------------------------------------------------
def test_mlp_align_recovers_planted_permutation_gamma_one():
    # ft = a KNOWN hidden-unit permutation of base (noise=0) ⇒ ALL of ΔW is gauge ⇒ γ≈1.
    base = _mlp_base(m=6, n=5, hidden=12, seed=1)
    ft = GF.planted_gauge_mlp(base, seed=2, noise=0.0)
    r = GF.mlp_align(base, ft)
    assert GF.gauge_fraction(r["aligned2"], r["raw2"]) > 0.999
    assert r["aligned2"] < 1e-8                     # residual essentially zero


def test_mlp_align_partial_gauge_with_noise_between_zero_and_one():
    # a planted permutation + noise ⇒ some ΔW is gauge, some physical ⇒ 0 < γ < 1.
    base = _mlp_base(m=8, n=6, hidden=16, seed=3)
    ft = GF.planted_gauge_mlp(base, seed=4, noise=0.05)
    r = GF.mlp_align(base, ft)
    g = GF.gauge_fraction(r["aligned2"], r["raw2"])
    assert 0.2 < g < 1.0


def test_attn_align_recovers_planted_head_permutation():
    base = _attn_base(n_heads=6, n_kv=2, head_dim=4, d_model=6, seed=5)
    group = 3
    rng = np.random.default_rng(9)
    kv_src = rng.permutation(2)
    within = np.stack([rng.permutation(group) for _ in range(2)])
    q_src, kv_src2 = GF.attn_head_perm_indices(kv_src, within, group)
    q2, k2, v2, o2 = GF.attn_permute(base["q_proj"], base["k_proj"], base["v_proj"], base["o_proj"],
                                     q_src, kv_src2, 4)
    ft = {"q_proj": q2, "k_proj": k2, "v_proj": v2, "o_proj": o2}
    r = GF.attn_align(base, ft, n_heads=6, n_kv=2, head_dim=4)
    assert GF.gauge_fraction(r["aligned2"], r["raw2"]) > 0.999


def test_attn_align_mqa_single_group():
    # MQA (n_kv=1): a permutation of the 8 query heads is fully recovered.
    base = _attn_base(n_heads=8, n_kv=1, head_dim=3, d_model=5, seed=6)
    rng = np.random.default_rng(11)
    q_src = rng.permutation(8)
    q2, k2, v2, o2 = GF.attn_permute(base["q_proj"], base["k_proj"], base["v_proj"], base["o_proj"],
                                     q_src, np.zeros(1, int), 3)
    ft = {"q_proj": q2, "k_proj": k2, "v_proj": v2, "o_proj": o2}
    r = GF.attn_align(base, ft, n_heads=8, n_kv=1, head_dim=3)
    assert GF.gauge_fraction(r["aligned2"], r["raw2"]) > 0.999


def test_identity_check_gamma_zero():
    base = _mlp_base(seed=7)
    r = GF.mlp_align(base, base)                     # align to itself → nothing changed
    assert GF.gauge_fraction(r["aligned2"], r["raw2"]) == 0.0
    assert GF.gauge_fraction(0.0, 0.0) == 0.0        # raw²=0 convention


def test_norm_align_recovers_planted_scale():
    # ft gain = D·base gain and ft matrix = base/D ⇒ pure gauge ⇒ γ≈1.
    rng = np.random.default_rng(8)
    d, n_out = 6, 4
    gb = np.abs(rng.standard_normal(d)) + 0.5
    Wb = rng.standard_normal((n_out, d))
    D = np.exp(rng.standard_normal(d) * 0.4)
    gf, (Wf,) = GF.norm_scale_transfer(gb, [Wb], D)
    r = GF.norm_align(gb, [Wb], gf, [Wf])
    assert GF.gauge_fraction(r["aligned2"], r["raw2"]) > 0.999


# ---------------------------------------------------------------------------
# (3) γ bounds + the overfit-null floor.
# ---------------------------------------------------------------------------
def test_gauge_fraction_bounded_zero_one_on_random_ft():
    # γ is in [0,1] BY CONSTRUCTION (identity ∈ catalogue): a random ft never gives γ<0.
    base = _mlp_base(m=8, n=6, hidden=20, seed=10)
    rng = np.random.default_rng(12)
    ft = {k: v + 0.3 * rng.standard_normal(v.shape) for k, v in base.items()}
    g = GF.gauge_fraction(*[GF.mlp_align(base, ft)[k] for k in ("aligned2", "raw2")])
    assert 0.0 <= g <= 1.0


def test_overfit_null_gamma_is_small():
    # a structureless (spectrum-matched) Δ: the aligner still over-fits SOME γ, but far below the planted
    # γ≈1 — this small floor is exactly what δ_GF captures (§5.3).
    base = _mlp_base(m=10, n=8, hidden=24, seed=13)
    gammas = []
    for s in range(6):
        nft = {t: base[t] + GF.spectrum_matched_null(0.2 * np.random.default_rng(s).standard_normal(base[t].shape), s)
               for t in base}
        r = GF.mlp_align(base, nft)
        gammas.append(GF.gauge_fraction(r["aligned2"], r["raw2"]))
    assert all(0.0 <= g < 0.6 for g in gammas)       # over-fit, but nowhere near a real planted gauge
    assert GF.delta_gf(gammas) >= 0.0


def test_spectrum_matched_null_preserves_spectrum():
    rng = np.random.default_rng(14)
    A = rng.standard_normal((12, 7))
    N = GF.spectrum_matched_null(A, seed=1)
    assert np.allclose(np.sort(np.linalg.svd(A, compute_uv=False)),
                       np.sort(np.linalg.svd(N, compute_uv=False)), atol=1e-9)
    assert N.shape == A.shape


def test_planted_beats_overfit_null_band():
    # the decisive comparison: planted γ (≈1) sits ABOVE the overfit-null half-width δ_GF; structureless
    # Δ does not clear it.
    base = _mlp_base(m=10, n=8, hidden=24, seed=15)
    planted = GF.gauge_fraction(*[GF.mlp_align(base, GF.planted_gauge_mlp(base, 16, 0.02))[k]
                                  for k in ("aligned2", "raw2")])
    null_g = [GF.gauge_fraction(*[GF.mlp_align(base, {t: base[t] + GF.spectrum_matched_null(
        0.2 * np.random.default_rng(s).standard_normal(base[t].shape), s) for t in base})[k]
        for k in ("aligned2", "raw2")]) for s in range(6)]
    d = GF.delta_gf(null_g)
    assert planted - float(np.median(null_g)) > d


# ---------------------------------------------------------------------------
# (4) summary + decision predicates.
# ---------------------------------------------------------------------------
def test_median_excess_ci_positive_signal():
    ci = GF.median_excess_ci([0.3, 0.28, 0.33, 0.31, 0.29], n_boot=1000)
    assert ci["ci_lo"] > 0 and ci["median"] > 0.25


def test_discovery_clear_and_detection_predicates():
    assert GF.discovery_clear(ci_lo=0.2, delta=0.05) is True
    assert GF.discovery_clear(ci_lo=0.02, delta=0.05) is False
    assert GF.detection(sealed_ci_lo=0.2, delta=0.05, bh_ok=True) is True
    assert GF.detection(sealed_ci_lo=0.2, delta=0.05, bh_ok=False) is False
    assert GF.detection(sealed_ci_lo=0.02, delta=0.05, bh_ok=True) is False


def test_delta_gf_needs_three_draws():
    with pytest.raises(AssertionError):
        GF.delta_gf([0.1, 0.2])
