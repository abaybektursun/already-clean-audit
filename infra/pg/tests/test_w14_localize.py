"""Weight-free CPU unit tests for the W14 localization/restoration science core
(infra/pg/w14_localize.py). NO model, NO datasets, NO GPU — small random tensors only,
so this runs in <1s on a laptop and gates the harness logic BEFORE any GPU job spends a
minute (prereg §12 fail-fast preflight, in spirit, at the unit level).

    python -m pytest infra/pg/tests/test_w14_localize.py -q
    # or:  python infra/pg/tests/test_w14_localize.py
"""
from __future__ import annotations

import pytest

from infra.pg import w14_localize as W


# ---------------------------------------------------------------------------
# Decoder-key partition (Arm 2) — and the anti-drift check vs the jobs layer.
# ---------------------------------------------------------------------------

def test_decoder_key_counts():
    assert len(W.decoder_keys()) == 182
    assert len(set(W.decoder_keys())) == 182                # no dupes
    assert len(W.module_keys("attn")) == 104                # q/k/v/o × 26
    assert len(W.module_keys("mlp")) == 78                  # gate/up/down × 26
    assert len(W.module_keys("all")) == 182
    # the two families partition the 182 with no overlap
    assert set(W.module_keys("attn")) | set(W.module_keys("mlp")) == set(W.decoder_keys())
    assert set(W.module_keys("attn")) & set(W.module_keys("mlp")) == set()
    assert all(".self_attn." in k for k in W.module_keys("attn"))
    assert all(".mlp." in k for k in W.module_keys("mlp"))


def test_n_layers_firewall():
    # §0 generation firewall: Gemma-2 is 26L, NOT the Gemma-1/18L sibling.
    assert W.N_LAYERS == 26
    assert {int(k.split(".")[1]) for k in W.decoder_keys()} == set(range(26))


def test_decoder_keys_match_jobs_layer():
    # w14_localize MIRRORS e4_compress_curve's key list; assert they can never drift.
    from infra.aws_pg.jobs.e4_compress_curve import _decoder_keys, N_LAYERS, LM_MATRIX_TYPES
    assert W.decoder_keys() == _decoder_keys()
    assert W.N_LAYERS == N_LAYERS
    assert W.LM_MATRIX_TYPES == LM_MATRIX_TYPES


def test_norm_keys_excludes_matrices_and_embed():
    keys = (W.decoder_keys()
            + ["layers.0.input_layernorm.weight", "layers.5.post_feedforward_layernorm.weight",
               "norm.weight", "embed_tokens.weight", "lm_head.weight"])
    nk = W.norm_keys(keys)
    assert "norm.weight" in nk
    assert "layers.0.input_layernorm.weight" in nk
    assert "embed_tokens.weight" not in nk and "lm_head.weight" not in nk
    assert not any(k in nk for k in W.decoder_keys())       # no matrices


# ---------------------------------------------------------------------------
# Subset reset overrides + matched-numel null (Arm 2).
# ---------------------------------------------------------------------------

def _toy_sd(scale=1.0):
    import torch
    torch.manual_seed(0)
    sd = {}
    for L in range(2):                                      # tiny 2-layer stand-in
        for mt in W.LM_MATRIX_TYPES:
            out_d = 6 if mt in ("q_proj", "gate_proj", "up_proj") else 3   # GQA-style asymmetry
            sd[W._decoder_key(L, mt)] = torch.randn(out_d, 4) * scale
    return sd


def test_reset_overrides_subset_and_shapes():
    after, base = _toy_sd(1.0), _toy_sd(2.0)
    keys = [W._decoder_key(0, "gate_proj"), W._decoder_key(1, "down_proj")]
    ov = W.reset_overrides(after, base, keys)
    assert set(ov) == set(keys)
    import torch
    for k in keys:
        assert torch.equal(ov[k], base[k])                  # override = base, not after


def test_reset_overrides_fail_loud_on_missing_and_shape():
    after, base = _toy_sd(), _toy_sd()
    with pytest.raises(KeyError):
        W.reset_overrides(after, base, ["layers.9.mlp.up_proj.weight"])   # absent
    import torch
    bad = dict(base); bad[W._decoder_key(0, "q_proj")] = torch.randn(99, 99)
    with pytest.raises(RuntimeError):
        W.reset_overrides(after, bad, [W._decoder_key(0, "q_proj")])      # shape mismatch


def test_matched_numel_null_is_gqa_asymmetric():
    sd = _toy_sd()                                          # 2-layer toy → scope keys to it
    toy_keys = list(sd.keys())
    kn = {k: int(v.numel()) for k, v in sd.items()}
    # q_proj (6×4=24) must outweigh k_proj (3×4=12): match Σ numel, not key-count.
    assert kn[W._decoder_key(0, "q_proj")] == 24
    assert kn[W._decoder_key(0, "k_proj")] == 12
    target = W.numel_of(sd, [k for k in toy_keys if ".mlp." in k])
    chosen, achieved = W.matched_numel_random_keys(toy_keys, kn, target, seed=1337)
    # achieved is within one matrix of target (whole-matrix granularity), and ≤ a loose bound.
    assert achieved > 0
    assert abs(achieved - target) <= max(kn.values())
    assert len(set(chosen)) == len(chosen)


def test_shuffled_reset_preserves_frobenius_but_destroys_structure():
    import torch
    after, base = _toy_sd(1.0), _toy_sd(2.0)
    keys = W.module_keys("mlp")[:6]
    keys = [k for k in keys if k in after]                  # scope to the 2-layer toy
    ov = W.shuffled_reset_overrides(after, base, keys, seed=1337)
    for k in keys:
        removed_true = (after[k] - base[k]).norm()          # true reset removes this
        removed_shuf = (after[k] - ov[k]).norm()            # shuffle removes same Frobenius
        assert torch.allclose(removed_true, removed_shuf, atol=1e-5)
        assert not torch.allclose(ov[k], base[k])           # but the result is NOT base (structure gone)


# ---------------------------------------------------------------------------
# Arm 1 — in/out-span construction + the audit gate (the crucial new science).
# ---------------------------------------------------------------------------

def _toy_dW(out_d=40, in_d=30, true_rank=12, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(out_d, true_rank, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(in_d, true_rank, generator=g, dtype=torch.float64))[0]
    s = torch.linspace(5.0, 0.5, true_rank, dtype=torch.float64)
    return (U * s) @ V.transpose(0, 1)                      # rank-12 dW in R^{40x30}


def test_topk_init_reproduces_rankk_edit():
    import torch
    dW = _toy_dW()
    g = torch.Generator().manual_seed(1)
    A, B = W.subspace_AB(dW, rank=8, kind="topk", gen=g)
    # A@B must equal the rank-8 SVD truncation of dW (Eckart–Young optimum).
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    dW8 = (U[:, :8] * S[:8]) @ Vh[:8, :]
    assert torch.allclose(A @ B, dW8, atol=1e-8)
    aud = W.construction_audit(A, B, dW, rank=8, kind="topk")
    assert aud["ok"], aud


def test_inspan_lies_in_span_and_matched_norm():
    import torch
    dW = _toy_dW()
    g = torch.Generator().manual_seed(2)
    A, B = W.subspace_AB(dW, rank=8, kind="inspan", gen=g)
    aud = W.construction_audit(A, B, dW, rank=8, kind="inspan")
    assert aud["A_subspace_residual"] < W.CONSTRUCTION_TOL, aud
    assert aud["B_subspace_residual"] < W.CONSTRUCTION_TOL, aud
    assert aud["matched_norm_abs_err"] < W.MATCHED_NORM_TOL, aud
    assert aud["ok"]
    # ‖A@B‖_F matches the top-k edit's norm (the magnitude control).
    assert abs(aud["ab_frobenius"] - aud["topk_frobenius"]) < W.MATCHED_NORM_TOL


def test_oospan_lies_in_complement_and_matched_norm():
    import torch
    dW = _toy_dW()
    g = torch.Generator().manual_seed(3)
    A, B = W.subspace_AB(dW, rank=8, kind="oospan", gen=g)
    aud = W.construction_audit(A, B, dW, rank=8, kind="oospan")
    assert aud["A_subspace_residual"] < W.CONSTRUCTION_TOL, aud   # in-span part vanishes
    assert aud["B_subspace_residual"] < W.CONSTRUCTION_TOL, aud
    assert aud["matched_norm_abs_err"] < W.MATCHED_NORM_TOL, aud
    assert aud["ok"]


def test_oospan_audit_rejects_an_inspan_construction():
    # the gate must FAIL if we feed an in-span (A,B) but claim it is out-of-span.
    import torch
    dW = _toy_dW()
    g = torch.Generator().manual_seed(4)
    A, B = W.subspace_AB(dW, rank=8, kind="inspan", gen=g)
    aud = W.construction_audit(A, B, dW, rank=8, kind="oospan")
    assert not aud["ok"]                                    # in-span energy is large ⇒ HALT


# A wide, low-energy-rank ΔW (out_d=1024) — the matrix shape that EXPOSED the
# `max(R_eff, k)` clamp bug (§A1.4 FINDING #1): clamping made out-of-span unconstructable
# and padded "the span" with off-energy-span directions. R_eff here is small (≈4).
def _toy_dW_wide(out_d=1024, in_d=256, true_rank=8, seed=11):
    import torch
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(out_d, true_rank, generator=g, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(in_d, true_rank, generator=g, dtype=torch.float64))[0]
    s = torch.logspace(0.0, -2.0, true_rank, dtype=torch.float64)   # sharp decay ⇒ small R_eff
    return (U * s) @ V.transpose(0, 1)


def test_oospan_constructable_on_wide_low_rank_dW():
    # FINDING #1(a): out-of-span MUST be constructable at a feasible k on a wide (out_d=1024),
    # low-energy-rank ΔW — the old `max(R_eff, k)` clamp made the complement dim collapse.
    import torch
    dW = _toy_dW_wide()
    R_eff = W.effective_rank_energy(torch.linalg.svd(dW, full_matrices=False)[1])
    feas = W.subspace_feasible(dW, rank=16, kind="oospan")
    assert feas["feasible"] and 16 <= feas["max_rank_oospan"], feas   # complement is large
    g = torch.Generator().manual_seed(20)
    A, B = W.subspace_AB(dW, rank=16, kind="oospan", gen=g)
    aud = W.construction_audit(A, B, dW, rank=16, kind="oospan")
    assert aud["R_eff"] == R_eff                            # unclamped: R_eff, NOT max(R_eff, k)
    assert aud["A_subspace_residual"] < W.CONSTRUCTION_TOL, aud   # in-span part of A vanishes
    assert aud["B_subspace_residual"] < W.CONSTRUCTION_TOL, aud
    assert aud["matched_norm_abs_err"] < W.MATCHED_NORM_TOL, aud
    assert aud["ok"]


def test_subspace_feasible_oospan_infeasible_at_out_d():
    # FINDING #1(b): out-of-span at k=out_d is infeasible — the complement has only
    # out_d − R_eff dims (k is clamped to min(out_d, in_d) first, still > the complement dim).
    import torch
    dW = _toy_dW_wide()                                     # out_d=1024, in_d=256
    R_eff = W.effective_rank_energy(torch.linalg.svd(dW, full_matrices=False)[1])
    feas = W.subspace_feasible(dW, rank=1024, kind="oospan")
    assert feas["feasible"] is False
    assert feas["R_eff"] == R_eff
    assert feas["out_d"] == 1024 and feas["in_d"] == 256
    assert feas["max_rank_oospan"] == min(1024 - R_eff, 256 - R_eff)
    assert feas["max_rank_inspan"] == R_eff
    assert feas["reason"]                                   # a non-empty diagnostic


def test_subspace_feasible_inspan_infeasible_above_R_eff():
    # in-span draws k vectors from the R_eff-dim span ⇒ infeasible once k > R_eff (no clamp).
    import torch
    dW = _toy_dW_wide()
    R_eff = W.effective_rank_energy(torch.linalg.svd(dW, full_matrices=False)[1])
    assert W.subspace_feasible(dW, rank=R_eff, kind="inspan")["feasible"] is True
    feas = W.subspace_feasible(dW, rank=R_eff + 1, kind="inspan")
    assert feas["feasible"] is False
    assert feas["max_rank_inspan"] == R_eff
    assert feas["reason"]


def test_inspan_columns_project_onto_TRUE_energy_span_no_leak():
    # FINDING #1(c): with the clamp removed, in-span A columns sit entirely in ΔW's TRUE
    # top-R_eff energy span (residual < 1e-5) — no off-energy-span leak from padding the span.
    import torch
    dW = _toy_dW_wide()
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    R_eff = W.effective_rank_energy(S)
    g = torch.Generator().manual_seed(21)
    A, B = W.subspace_AB(dW, rank=R_eff, kind="inspan", gen=g)   # k = R_eff (the largest feasible)
    Pu = U[:, :R_eff] @ U[:, :R_eff].transpose(0, 1)             # projector onto the TRUE span
    Pv = (Vh[:R_eff, :].transpose(0, 1)) @ Vh[:R_eff, :]
    a_off = float(((A - Pu @ A).norm() / A.norm()).item())
    b_off = float(((B.transpose(0, 1) - Pv @ B.transpose(0, 1)).norm() / B.norm()).item())
    assert a_off < 1e-5, a_off
    assert b_off < 1e-5, b_off


# ---------------------------------------------------------------------------
# Arm 3 — restore-direction rank-k graft.
# ---------------------------------------------------------------------------

def test_restore_full_rank_returns_base():
    import torch
    g = torch.Generator().manual_seed(5)
    after = torch.randn(20, 16, generator=g, dtype=torch.float64)
    base = torch.randn(20, 16, generator=g, dtype=torch.float64)
    restored = W.restore_svd_graft(after, base, rank=16)    # full rank
    assert torch.allclose(restored, base, atol=1e-8)        # → base decoder (positive control)


def test_restore_rankk_is_monotone_and_eckart_young():
    import torch
    g = torch.Generator().manual_seed(6)
    after = torch.randn(24, 18, generator=g, dtype=torch.float64)
    base = after + _toy_dW(24, 18, true_rank=10, seed=7)    # base = after + a rank-10 change
    dW = base - after
    prev = float("inf")
    for k in (1, 3, 6, 10):
        restored = W.restore_svd_graft(after, base, rank=k)
        err = float((restored - base).norm())               # residual to the full restore
        assert err <= prev + 1e-9                            # monotone ↓ in rank
        prev = err
    # at the true rank the restore is essentially exact.
    assert float((W.restore_svd_graft(after, base, rank=10) - base).norm()) < 1e-6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
