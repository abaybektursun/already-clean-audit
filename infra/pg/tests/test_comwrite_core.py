"""Weight-free dry-run gate for the U_wr core (comwrite spec §10.1) — planted-structure Gram
correctness INCLUDING the NEW o/down output-side variant, LOO exclusion actually excludes,
ladder/k* selector, and the Arm-C gradient-projection partition identity.

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_comwrite_core.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from infra.pg import comwrite_core as CC


def _unit(d, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _planted_family(d=12, m=6, n_ft=5, side="input", c_seed=0, noise=0.02):
    """n_ft ΔWs that all write mostly into ONE planted residual-facing direction c."""
    c = _unit(d, c_seed)
    rng = np.random.default_rng(100 + c_seed)
    fam = []
    for i in range(n_ft):
        a = rng.standard_normal(m)
        if side == "input":                       # (m, d): rows live in the input space
            dW = np.outer(a, c) + noise * rng.standard_normal((m, d))
        else:                                     # (d, m): columns live in the output space
            dW = np.outer(c, a) + noise * rng.standard_normal((d, m))
        fam.append(dW)
    return c, fam


# ---------------------------------------------------------------------------
# residual-facing side convention.
# ---------------------------------------------------------------------------
def test_residual_side_map_is_the_frozen_convention():
    for t in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        assert CC.residual_side(t) == "input"
    for t in ("o_proj", "down_proj"):
        assert CC.residual_side(t) == "output"
    with pytest.raises(AssertionError):
        CC.residual_side("embed")


def test_residual_dim_picks_the_2048_analog_side():
    assert CC.residual_dim((256, 8), "k_proj") == 8       # input side
    assert CC.residual_dim((8, 256), "down_proj") == 8    # output side


# ---------------------------------------------------------------------------
# consensus Gram (both sides) on planted structure.
# ---------------------------------------------------------------------------
def test_gram_term_unit_trace_and_input_side_matches_invariant_core_form():
    rng = np.random.default_rng(0)
    dW = rng.standard_normal((6, 12))
    T = CC.gram_term(dW, "q_proj")
    assert T.shape == (12, 12)
    assert abs(np.trace(T) - 1.0) < 1e-12
    ref = dW.T @ dW / (dW ** 2).sum()             # invariant_core._consensus_gram summand
    assert np.allclose(T, ref)


def test_gram_term_output_side_variant_for_o_down():
    rng = np.random.default_rng(1)
    dW = rng.standard_normal((12, 6))             # down-like: out=12 is the residual side
    T = CC.gram_term(dW, "down_proj")
    assert T.shape == (12, 12)
    assert np.allclose(T, dW @ dW.T / (dW ** 2).sum())


def test_consensus_gram_trace_is_n_and_zero_dW_raises():
    _c, fam = _planted_family()
    G = CC.consensus_gram(fam, "q_proj")
    assert abs(np.trace(G) - len(fam)) < 1e-9
    with pytest.raises(AssertionError):
        CC.gram_term(np.zeros((4, 4)), "q_proj")


def test_u_wr_recovers_the_planted_direction_input_side():
    c, fam = _planted_family(side="input")
    U = CC.u_wr(CC.consensus_gram(fam, "q_proj"), 1)
    assert CC.abs_cos(U[:, 0], c) > 0.99


def test_u_wr_recovers_the_planted_direction_output_side():
    c, fam = _planted_family(side="output")
    U = CC.u_wr(CC.consensus_gram(fam, "down_proj"), 1)
    assert CC.abs_cos(U[:, 0], c) > 0.99


def test_u_wr_is_the_top_mirror_not_the_bottom():
    c, fam = _planted_family()
    G = CC.consensus_gram(fam, "q_proj")
    w = np.linalg.eigvalsh(G)
    U = CC.u_wr(G, 1)
    assert abs(float(U[:, 0] @ G @ U[:, 0]) - w[-1]) < 1e-9      # top eigenvalue, not bottom


# ---------------------------------------------------------------------------
# LOO discipline: gram_minus excludes the donor — a donor-private direction cannot
# enter its own basis.
# ---------------------------------------------------------------------------
def test_loo_exclusion_actually_excludes():
    d = 12
    c, fam = _planted_family(d=d, n_ft=4, noise=0.0)
    p = _unit(d, 999)
    p = p - (p @ c) * c
    p /= np.linalg.norm(p)                        # donor-private direction ⊥ c
    donor = 50.0 * np.outer(np.ones(6), p)        # donor 0 writes ONLY into p, hugely
    fam = [donor] + fam
    terms = [CC.gram_term(dW, "q_proj") for dW in fam]
    G_all = sum(terms[1:], terms[0])
    U_full = CC.u_wr(G_all, 1)
    U_loo = CC.u_wr(CC.gram_minus(G_all, terms[0]), 1)
    assert CC.abs_cos(U_loo[:, 0], c) > 0.99      # LOO basis = the family direction
    assert CC.abs_cos(U_loo[:, 0], p) < 0.05      # the donor's private direction is GONE
    assert np.allclose(CC.gram_minus(G_all, terms[0]),
                       CC.consensus_gram(fam[1:], "q_proj"))


# ---------------------------------------------------------------------------
# energy capture + ladder + k*.
# ---------------------------------------------------------------------------
def test_energy_fraction_exact_on_planted_spans():
    d = 10
    v = np.zeros(d)
    v[3] = 1.0
    dW = np.outer(np.arange(1, 5, dtype=float), v)          # input-side energy all in e3
    assert CC.energy_fraction(dW, v[:, None], "q_proj") == pytest.approx(1.0)
    u = np.zeros(d)
    u[7] = 1.0
    assert CC.energy_fraction(dW, u[:, None], "q_proj") == pytest.approx(0.0)
    dWo = np.outer(v, np.arange(1, 5, dtype=float))         # output-side analog
    assert CC.energy_fraction(dWo, v[:, None], "o_proj") == pytest.approx(1.0)


def test_chance_floor_is_k_over_d():
    assert CC.chance_floor(16, 2048) == pytest.approx(16 / 2048)


def test_ladder_and_k_star_on_planted_rank2_structure():
    d = 32
    rng = np.random.default_rng(7)
    c1, c2 = _unit(d, 1), _unit(d, 2)
    c2 = c2 - (c2 @ c1) * c1
    c2 /= np.linalg.norm(c2)
    fam = [np.outer(rng.standard_normal(6), c1) + np.outer(rng.standard_normal(6), c2)
           + 1e-3 * rng.standard_normal((6, d)) for _ in range(6)]
    G = CC.consensus_gram(fam[:3], "q_proj")                # "discovery" half
    U = CC.u_wr(G, 16)
    ladder = CC.concentration_ladder([(dW, "q_proj") for dW in fam[3:]], U,
                                     k_grid=(1, 2, 4, 8, 16))
    assert ladder[2] > 0.99                                 # rank-2 planted → saturates at k=2
    assert CC.k_star(ladder, frac=0.9) == 2


def test_k_star_frozen_selector_on_a_synthetic_ladder():
    ladder = {1: 0.40, 2: 0.70, 4: 0.87, 8: 0.93, 16: 0.97, 32: 0.99, 64: 1.00, 128: 1.00}
    assert CC.k_star(ladder, frac=0.9) == 8


# ---------------------------------------------------------------------------
# structure statistics.
# ---------------------------------------------------------------------------
def test_mean_direction_matches_u_wr_on_a_rank1_star():
    d = 16
    c = _unit(d, 3)
    rng = np.random.default_rng(11)
    fam = [np.outer(rng.standard_normal(5), c) for _ in range(5)]
    md = CC.mean_direction(fam, "q_proj")
    U = CC.u_wr(CC.consensus_gram(fam, "q_proj"), 1)
    assert CC.abs_cos(md, c) > 0.99
    assert CC.abs_cos(U[:, 0], md) > 0.99


def test_subspace_overlap_bounds():
    d = 10
    Q = np.linalg.qr(np.random.default_rng(0).standard_normal((d, 4)))[0]
    assert CC.subspace_overlap(Q, Q) == pytest.approx(1.0)
    assert CC.subspace_overlap(Q[:, :2], Q[:, 2:4]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Arm C — the one-sided gradient-projection partition identity (§10.1).
# ---------------------------------------------------------------------------
def test_project_grad_residual_partitions_exactly_input_side():
    g = torch.randn(6, 12, generator=torch.Generator().manual_seed(0))
    U = torch.linalg.qr(torch.randn(12, 3, generator=torch.Generator().manual_seed(1)))[0]
    into = CC.project_grad_residual(g, U, "q_proj", "into")
    off = CC.project_grad_residual(g, U, "q_proj", "off")
    assert torch.allclose(into + off, g, atol=1e-6)                      # partition
    assert abs(float((into * off).sum())) < 1e-4                        # Frobenius-orthogonal
    assert torch.allclose(CC.project_grad_residual(into, U, "q_proj", "into"), into,
                          atol=1e-5)                                    # idempotent
    assert float(CC.project_grad_residual(off, U, "q_proj", "into").norm()) < 1e-5


def test_project_grad_residual_output_side_for_down():
    g = torch.randn(12, 6, generator=torch.Generator().manual_seed(2))
    U = torch.linalg.qr(torch.randn(12, 3, generator=torch.Generator().manual_seed(3)))[0]
    into = CC.project_grad_residual(g, U, "down_proj", "into")
    off = CC.project_grad_residual(g, U, "down_proj", "off")
    assert torch.allclose(into + off, g, atol=1e-6)
    assert torch.allclose(into, U @ (U.T @ g), atol=1e-6)               # LEFT application
    with pytest.raises(AssertionError):
        CC.project_grad_residual(g, torch.eye(6)[:, :2], "down_proj", "into")  # wrong side


def test_project_grad_residual_bf16_grad_partitions_in_fp32():
    """The deployed dtype: bf16 grads (the Arm-C trainer) must not leak the partition —
    the projection is computed internally in fp32 (a bf16 projection leaves ~0.3% RMS
    in-span residue in the 'off' output, an accumulating bias over SGD steps)."""
    g32 = torch.randn(64, 128, generator=torch.Generator().manual_seed(4))
    U = torch.linalg.qr(torch.randn(128, 8, generator=torch.Generator().manual_seed(5)))[0]
    g16 = g32.to(torch.bfloat16)
    off16 = CC.project_grad_residual(g16, U, "q_proj", "off")
    assert off16.dtype == torch.bfloat16
    # the in-span residue of 'off' must sit at the fp32→bf16 ROUNDING floor of the output
    # (isotropic, ∝√(k/d)·round-off), far below a systematic bf16-projection leak.
    residue = CC.project_grad_residual(off16.float(), U, "q_proj", "into").norm()
    assert float(residue) < 0.02 * float(g32.norm())
    into16 = CC.project_grad_residual(g16, U, "q_proj", "into")
    # into16 and off16 are each independently bf16-rounded (≈0.8% ulp), so the sum can
    # differ from g by ~2 ulps per element — bound the RELATIVE Frobenius error instead
    # (a wrong-side/wrong-basis projection errs at O(1), two decades above this).
    rel = float((into16.float() + off16.float() - g16.float()).norm() / g16.float().norm())
    assert rel < 0.02, rel


def test_project_grad_residual_rejects_unknown_mode():
    g = torch.randn(4, 4)
    U = torch.eye(4)[:, :2]
    with pytest.raises(ValueError):
        CC.project_grad_residual(g, U, "q_proj", "sideways")
