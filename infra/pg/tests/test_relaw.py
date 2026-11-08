"""Weight-free CPU dry-run gate for RELAW (the prereg §9.1 gate).

NO model, NO datasets, NO GPU — tiny planted tensors only, <2s on a CPU box. Proves the
PURE logic BEFORE any GPU minute is spent:
  * principal_survival identities (W↔W = 1; scale-invariance; chance ≈ r/dim; monotone in
    perturbation; pooled mean), built on raw fp64 SVD not mean-centred basis;
  * the insulation projector algebra (into ⊥ off; ‖into‖²+‖off‖²=‖g‖²; off carries no
    in-subspace energy; into reproduces an in-block gradient);
  * the law-fit predicates (select_predictor_r, loo_sign_stable, bh_fdr, law_verdict);
  * the end-to-end law-fit + verdict lockstep on synthetic units (predictor-exists vs
    H-probe-null vs the permutation collapse), reusing savb.delta_loo_perm verbatim;
  * the generation-firewall layer counter (18 vs 26).
Distinct from the runtime preflight (§9.3, which proves the WIRING on real weights).

Run (on a remote `pg` CPU instance — never locally; torch is a large lib):
    python -m pytest infra/pg/tests/test_relaw.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from infra.pg import relaw as RL
from infra.pg import savb as SB
from infra.pg import w15_carrier as C


def _rand_W(m=8, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# principal_survival — the predictor identities.
# ---------------------------------------------------------------------------
def test_principal_survival_identical_is_one():
    W = _rand_W()
    assert RL.principal_survival(W, W, r=3) == pytest.approx(1.0, abs=1e-9)


def test_principal_survival_scale_invariant():
    # scaling a matrix leaves its singular SUBSPACES unchanged ⇒ survival = 1.
    W = _rand_W(seed=4)
    assert RL.principal_survival(W, 7.5 * W, r=3) == pytest.approx(1.0, abs=1e-9)
    assert RL.principal_survival(W, -2.0 * W, r=4) == pytest.approx(1.0, abs=1e-9)


def test_principal_survival_full_rank_is_one():
    # only a SQUARE matrix at r=m=n has BOTH left and right top-r spanning the whole space ⇒
    # overlap=1 for any two matrices. (For non-square, the larger-dim subspace is never full at
    # r=min(m,n), so survival<1 — the meaningful regime; see test_..._bounded_for_independent.)
    W, Wf = _rand_W(m=6, n=6, seed=1), _rand_W(m=6, n=6, seed=2)
    assert RL.principal_survival(W, Wf, r=6) == pytest.approx(1.0, abs=1e-9)


def test_principal_survival_nonsquare_full_min_rank_below_one():
    # at r=min(m,n) of a NON-square matrix, the right subspace is full but the left is a strict
    # subspace of the larger dim ⇒ two independent matrices do NOT overlap fully (guards the
    # earlier faulty assumption that this should be 1).
    o = RL.principal_survival(_rand_W(m=12, n=6, seed=1), _rand_W(m=12, n=6, seed=2), r=6)
    assert o < 0.999


def test_principal_survival_bounded_and_chance_for_independent():
    # two independent random matrices: top-r subspaces overlap near chance r/dim, well below 1.
    m, n, r = 64, 64, 8
    o = RL.principal_survival(_rand_W(m, n, 11), _rand_W(m, n, 12), r=r)
    assert 0.0 <= o <= 1.0
    assert o < 0.4                       # nowhere near the survival=1 rail
    assert o > 0.5 * (r / m) * 0.25      # but not zero — finite-sample chance overlap


def test_principal_survival_monotone_in_perturbation():
    W = _rand_W(m=40, n=40, seed=7)
    gn = torch.Generator().manual_seed(3)
    N = torch.randn(40, 40, generator=gn, dtype=torch.float64)
    near = RL.principal_survival(W, W + 0.02 * N, r=5)
    far = RL.principal_survival(W, W + 1.50 * N, r=5)
    assert near > far                    # a smaller fine-tune preserves more principal subspace
    assert near > 0.8


def test_pooled_principal_survival_means_over_matrices():
    base = {"a": _rand_W(seed=1), "b": _rand_W(seed=2)}
    ft = {"a": base["a"].clone(), "b": _rand_W(seed=9)}   # 'a' identical, 'b' perturbed
    res = RL.pooled_principal_survival(base, ft, ["a", "b"], r=3)
    assert res["per_matrix"]["a"] == pytest.approx(1.0, abs=1e-9)
    assert res["pooled"] == pytest.approx(
        0.5 * (res["per_matrix"]["a"] + res["per_matrix"]["b"]), rel=1e-12)
    assert res["n_mat"] == 2 and res["r_used"] == 3


def test_pooled_principal_survival_shape_mismatch_raises():
    base = {"a": _rand_W(8, 6, 1)}
    ft = {"a": _rand_W(6, 8, 1)}
    with pytest.raises(AssertionError):
        RL.pooled_principal_survival(base, ft, ["a"], r=2)


# ---------------------------------------------------------------------------
# project_grad — the insulation projector algebra (Arm-3).
# ---------------------------------------------------------------------------
def test_project_grad_orthogonal_energy_split():
    W = _rand_W(m=20, n=14, seed=5)
    Ur, Vr = RL.topk_subspace(W, r=4)
    gg = torch.Generator().manual_seed(8)
    g = torch.randn(20, 14, generator=gg, dtype=torch.float64)
    into = RL.project_grad(g, Ur, Vr, "into")
    off = RL.project_grad(g, Ur, Vr, "off")
    # into + off = g, and the two are Frobenius-orthogonal ⇒ ‖into‖²+‖off‖²=‖g‖².
    assert torch.allclose(into + off, g, atol=1e-9)
    assert float((into * off).sum()) == pytest.approx(0.0, abs=1e-9)
    assert float(into.norm() ** 2 + off.norm() ** 2) == pytest.approx(float(g.norm() ** 2), rel=1e-9)


def test_project_grad_off_carries_no_in_block_energy():
    W = _rand_W(m=18, n=12, seed=6)
    Ur, Vr = RL.topk_subspace(W, r=3)
    gg = torch.Generator().manual_seed(2)
    g = torch.randn(18, 12, generator=gg, dtype=torch.float64)
    off = RL.project_grad(g, Ur, Vr, "off")
    # U_rᵀ (off) V_r must be ~0 — the insulated gradient has nothing in the principal block.
    resid = Ur.transpose(0, 1) @ off @ Vr
    assert float(resid.norm()) == pytest.approx(0.0, abs=1e-9)


def test_project_grad_into_reproduces_in_block_gradient():
    W = _rand_W(m=16, n=10, seed=9)
    Ur, Vr = RL.topk_subspace(W, r=3)
    # a gradient that lives ENTIRELY in the block: g = U_r M V_rᵀ ⇒ project_grad(into) == g.
    gm = torch.Generator().manual_seed(1)
    M = torch.randn(3, 3, generator=gm, dtype=torch.float64)
    g = Ur @ M @ Vr.transpose(0, 1)
    assert torch.allclose(RL.project_grad(g, Ur, Vr, "into"), g, atol=1e-9)
    assert float(RL.project_grad(g, Ur, Vr, "off").norm()) == pytest.approx(0.0, abs=1e-9)


def test_project_grad_bad_mode_raises():
    W = _rand_W(seed=0)
    Ur, Vr = RL.topk_subspace(W, r=2)
    with pytest.raises(ValueError):
        RL.project_grad(_rand_W(seed=1), Ur, Vr, "sideways")


def test_insulated_update_anti_is_magnitude_matched_to_off():
    W = _rand_W(m=24, n=18, seed=4)
    Ur, Vr = RL.topk_subspace(W, r=4)
    gg = torch.Generator().manual_seed(5)
    g = torch.randn(24, 18, generator=gg, dtype=torch.float64)
    std = RL.insulated_update(g, Ur, Vr, "standard")
    off = RL.insulated_update(g, Ur, Vr, "off")
    into = RL.insulated_update(g, Ur, Vr, "into")
    assert torch.allclose(std, g)                                   # standard = raw gradient
    # anti is magnitude-matched to the insulated step but points INTO the protected block.
    assert float(into.norm()) == pytest.approx(float(off.norm()), rel=1e-9)
    # off carries no in-block energy; the matched anti lives ENTIRELY in the block.
    assert float((Ur.transpose(0, 1) @ off @ Vr).norm()) == pytest.approx(0.0, abs=1e-9)
    off_resid = into - Ur @ (Ur.transpose(0, 1) @ into @ Vr) @ Vr.transpose(0, 1)
    assert float(off_resid.norm()) == pytest.approx(0.0, abs=1e-9)


def test_insulated_update_bad_mode_raises():
    W = _rand_W(seed=0)
    Ur, Vr = RL.topk_subspace(W, r=2)
    with pytest.raises(ValueError):
        RL.insulated_update(_rand_W(seed=1), Ur, Vr, "elsewhere")


# ---------------------------------------------------------------------------
# law-fit predicates.
# ---------------------------------------------------------------------------
def test_select_predictor_r_argmax_ties_to_smaller():
    assert RL.select_predictor_r([8, 16, 32], {8: 0.1, 16: 0.5, 32: 0.5}) == 16   # tie → smaller
    assert RL.select_predictor_r([8, 16, 32], {8: 0.1, 16: 0.2, 32: 0.9}) == 32


def test_bh_fdr_cases():
    assert RL.bh_fdr([0.001, 0.002, 0.003], q=0.05).tolist() == [True, True, True]
    assert RL.bh_fdr([0.9, 0.8, 0.7], q=0.05).tolist() == [False, False, False]
    # one tiny p among large ones survives at its BH rank.
    rej = RL.bh_fdr([0.001, 0.9, 0.8], q=0.05)
    assert rej.tolist() == [True, False, False]


def test_law_verdict_truth_table():
    assert RL.law_verdict(perm_p=0.01, fdr_ok=True, heldout_lift=0.1, loo_stable=True) is True
    assert RL.law_verdict(perm_p=0.20, fdr_ok=True, heldout_lift=0.1, loo_stable=True) is False
    assert RL.law_verdict(perm_p=0.01, fdr_ok=False, heldout_lift=0.1, loo_stable=True) is False
    assert RL.law_verdict(perm_p=0.01, fdr_ok=True, heldout_lift=-0.1, loo_stable=True) is False
    assert RL.law_verdict(perm_p=0.01, fdr_ok=True, heldout_lift=0.1, loo_stable=False) is False


# ---------------------------------------------------------------------------
# end-to-end law-fit + verdict on synthetic units (reuses savb.delta_loo_perm).
# ---------------------------------------------------------------------------
def _synth_predictive(n=24, seed=0):
    """R_prin truly predicts G beyond a noise R_dist baseline (the planted positive control)."""
    rng = np.random.default_rng(seed)
    R_prin = rng.uniform(0.4, 1.0, n)
    G = RL.planted_g(R_prin, slope=2.0, noise_sd=0.05, seed=seed + 1)
    R_dist = rng.normal(0, 1, n)                       # independent of G
    return R_prin, R_dist, G


def test_law_fit_recovers_planted_predictor():
    R_prin, R_dist, G = _synth_predictive()
    res = SB.delta_loo_perm(R_dist.reshape(-1, 1), R_prin, G, n_perm=500, seed=1337)
    assert res["delta"] > 0.5                          # R_prin adds large LOO power
    assert res["p_perm"] < 0.05                        # beyond the permutation null
    stab = RL.loo_sign_stable(R_dist.reshape(-1, 1), R_prin, G)
    assert stab["stable"] and stab["base_delta"] > 0
    assert RL.law_verdict(res["p_perm"], fdr_ok=True, heldout_lift=res["delta"],
                          loo_stable=stab["stable"]) is True


def test_law_fit_null_adds_no_power():
    # G is driven ENTIRELY by R_dist; R_prin is independent noise ⇒ adding R_prin must NOT improve
    # out-of-sample LOO-R² (deterministic: ΔLOO-R² is small/≤0, no reliance on a lucky perm draw).
    rng = np.random.default_rng(42)
    n = 24
    R_dist = rng.normal(0, 1, n)
    G = 3.0 * R_dist + rng.normal(0, 0.05, n)          # R_dist already explains G
    R_prin = rng.normal(0, 1, n)                        # unrelated to G
    res = SB.delta_loo_perm(R_dist.reshape(-1, 1), R_prin, G, n_perm=500, seed=1337)
    assert res["delta"] < 0.1                           # the noise read adds no real LOO power
    # and the §6 verdict rejects it (no out-of-sample lift to clear).
    assert RL.law_verdict(res["p_perm"], fdr_ok=True, heldout_lift=-abs(res["delta"]),
                          loo_stable=True) is False


def test_law_fit_collapses_under_permutation():
    R_prin, R_dist, G = _synth_predictive(seed=7)
    real = SB.loo_r2(np.column_stack([R_dist, R_prin]), G)
    perm = np.random.default_rng(0).permutation(G.size)
    shuffled = SB.loo_r2(np.column_stack([R_dist, R_prin]), G[perm])
    assert real > 0.8 and shuffled < real - 0.5        # permuting G destroys the fit


# ---------------------------------------------------------------------------
# generation firewall — layer counter (18 Gemma-1 vs 26 Gemma-2).
# ---------------------------------------------------------------------------
def test_n_decoder_layers_counts_distinct_indices():
    g1 = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(18)]
    g2 = [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(26)]
    assert RL.n_decoder_layers(g1) == 18
    assert RL.n_decoder_layers(g2) == 26
    # multiple matrices per layer still count the layer once.
    multi = ([f"model.layers.{i}.self_attn.q_proj.weight" for i in range(3)]
             + [f"model.layers.{i}.mlp.up_proj.weight" for i in range(3)])
    assert RL.n_decoder_layers(multi) == 3


def test_n_decoder_layers_raises_without_layers():
    with pytest.raises(AssertionError):
        RL.n_decoder_layers(["model.embed_tokens.weight", "lm_head.weight"])
