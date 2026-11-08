"""Weight-free CPU unit tests for the W15 carrier-gate science core
(infra/pg/w15_carrier.py). NO model, NO datasets, NO GPU — small random tensors only,
so this runs in <1s on a laptop and IS the prereg §9.1 OFFLINE DRY-RUN gate: every pure
logic the GPU jobs depend on is proved here BEFORE any GPU minute is spent.

    python -m pytest infra/pg/tests/test_w15_carrier.py -q
    # or:  python infra/pg/tests/test_w15_carrier.py

Maps directly onto the §9.1 checklist:
  * planted disjoint → O≈0 / identical → O≈1            (test_plant_*)
  * permuted-null overlap matches k/N                    (test_permuted_null_is_kN)
  * M = Uᵀ ΔW V round-trips < 1e-10                       (test_recon_round_trip)
  * oracle/leverage arithmetic on a known carrier        (test_direction_scores_*)
  * graft selection ≡ e4 rank-k on the energy ordering   (test_graft_energy_is_rankk)
"""
from __future__ import annotations

import pytest
import torch

from infra.pg import w15_carrier as C


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------
# Key-list firewall (re-exported from w14_localize — must never drift).
# ---------------------------------------------------------------------------

def test_decoder_keys_182_and_firewall():
    assert C.N_LAYERS == 26                                     # §0 Gemma-2 firewall
    assert len(C.decoder_keys()) == 182
    assert len(set(C.decoder_keys())) == 182
    assert len(C.module_keys("attn")) == 104
    assert len(C.module_keys("mlp")) == 78


# ---------------------------------------------------------------------------
# INSTALL — the shared instrument recovers a known carrier; selection grafts.
# ---------------------------------------------------------------------------

def test_direction_scores_recover_planted_carrier():
    # ΔW with clean orthonormal SVD directions; a loss gradient sensitive ONLY to dir d*.
    torch.manual_seed(1)
    m, n = 8, 6
    A = torch.randn(m, m, dtype=torch.float64)
    B = torch.randn(n, n, dtype=torch.float64)
    U, _ = torch.linalg.qr(A); U = U[:, :n]                     # (m, n) orthonormal cols
    V, _ = torch.linalg.qr(B)                                   # (n, n) orthonormal
    S = torch.arange(n, 0, -1, dtype=torch.float64)            # descending σ
    dW = (U * S) @ V.transpose(0, 1)
    Uc, Sc, Vc = C.dw_svd(dW)
    # gradient that cares only about SVD direction d*=2: G = u_d* v_d*ᵀ
    dstar = 2
    G = torch.outer(Uc[:, dstar], Vc[:, dstar])
    scores = C.direction_scores(G, Uc, Vc)
    assert scores.shape == (n,)
    # one-hot at d*: score_d ≈ δ_{d,d*}
    assert scores[dstar].item() == pytest.approx(1.0, abs=1e-9)
    others = scores.clone(); others[dstar] = 0.0
    assert others.abs().max().item() < 1e-9
    assert C.rank_select(scores, 1).tolist() == [dstar]


def test_graft_energy_is_rankk():
    # graft_from_indices on the energy ordering == the rank-r SVD truncation (e4 _rankk).
    torch.manual_seed(2)
    m, n, r = 7, 5, 3
    dW = torch.randn(m, n, dtype=torch.float64)
    U, S, V = C.dw_svd(dW)
    idx = C.energy_indices(r, S.numel())
    assert idx.tolist() == [0, 1, 2]
    graft = C.graft_from_indices(torch.zeros(m, n, dtype=torch.float64), U, S, V, idx)
    rankk = (U[:, :r] * S[:r]) @ V[:, :r].transpose(0, 1)       # the reference rank-r recon
    assert (graft - rankk).norm().item() < 1e-12


def test_selection_graft_adds_to_W_before():
    torch.manual_seed(3)
    m, n, r = 6, 6, 2
    Wb = torch.randn(m, n, dtype=torch.float64)
    dW = torch.randn(m, n, dtype=torch.float64)
    U, S, V = C.dw_svd(dW)
    idx = C.rank_select(torch.randn(S.numel()), r)
    graft = C.graft_from_indices(Wb, U, S, V, idx)
    expect = Wb + (U[:, idx] * S[idx]) @ V[:, idx].transpose(0, 1)
    assert (graft - expect).norm().item() < 1e-12


def test_random_rotation_graft_matched_norm():
    # the null graft's Frobenius mass must equal the energy rank-r graft's (matched-norm).
    torch.manual_seed(4)
    m, n, r = 12, 10, 3
    dW = torch.randn(m, n, dtype=torch.float64)
    U, S, V = C.dw_svd(dW)
    energy_fro = C.selection_norm(S, C.energy_indices(r, S.numel()))
    Wb = torch.zeros(m, n, dtype=torch.float64)
    null = C.random_rotation_graft(Wb, dW, r, _gen(7))
    null_fro = (null - Wb).norm().item()
    assert null_fro == pytest.approx(energy_fro, rel=1e-3)      # MATCHED_NORM_TOL regime


def test_base_after_rank_corr_bounds():
    s = torch.randn(20, dtype=torch.float64)
    assert C.base_after_rank_corr(s, s) == pytest.approx(1.0, abs=1e-9)
    assert C.base_after_rank_corr(s, -s) == pytest.approx(1.0, abs=1e-9)   # |·| ranking
    assert -1.0 <= C.base_after_rank_corr(s, torch.randn(20, dtype=torch.float64)) <= 1.0


# ---------------------------------------------------------------------------
# RETENTION — basis round-trip, saliency, overlap, planted controls, null.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(6, 6), (8, 5), (5, 8)])
def test_recon_round_trip(shape):
    # M = Uᵀ ΔW V reconstructs the in-span part of ΔW to < 1e-10 (fp64 mandate gate).
    torch.manual_seed(5)
    Wb = torch.randn(*shape, dtype=torch.float64)
    dW = torch.randn(*shape, dtype=torch.float64)
    U, s, V = C.svd_fp64(Wb)
    assert C.recon_residual(dW, U, V) < C.RECON_TOL


def test_edge_saliency_is_weighted_movement():
    M = torch.tensor([[2.0, -1.0], [0.0, 3.0]], dtype=torch.float64)
    Gb = torch.tensor([[1.0, 4.0], [5.0, -1.0]], dtype=torch.float64)
    I = C.edge_saliency(M, Gb)
    assert torch.allclose(I, torch.tensor([[2.0, 4.0], [0.0, 3.0]], dtype=torch.float64))


def test_energy_topk_count_and_mask():
    # a map with one dominant edge: 50% energy reached by the single top edge.
    M = torch.zeros(4, 4, dtype=torch.float64)
    M[0, 0] = 10.0
    M[1, 1] = M[2, 2] = M[3, 3] = 1.0
    k = C.energy_topk_count(M, frac=0.50)
    assert k == 1
    mask = C.topk_mask(M.abs(), k)
    assert int(mask.sum()) == 1 and mask[0]                     # flat index 0 == M[0,0]


def test_plant_identical_is_O1():
    Ia, Ib, k = C.plant_identical((6, 6), k=5, gen=_gen(11))
    assert C.overlap_maps(Ia, Ib, k) == pytest.approx(1.0, abs=1e-6)


def test_plant_disjoint_is_O0():
    Ia, Ib, k = C.plant_disjoint((6, 6), k=5, gen=_gen(12))
    assert C.overlap_maps(Ia, Ib, k) == pytest.approx(0.0, abs=1e-6)


def test_permuted_null_is_kN():
    # function-permuted null centres on the analytic chance overlap k/N.
    torch.manual_seed(6)
    Ia = torch.rand(12, 12, dtype=torch.float64)
    Ib = torch.rand(12, 12, dtype=torch.float64)
    n = Ia.numel()
    k = 20
    samples, kN = C.permuted_overlap(Ia, Ib, k, n_draws=3000, seed=99)
    assert kN == pytest.approx(k / n, abs=1e-12)
    assert float(samples.mean()) == pytest.approx(k / n, abs=0.02)


def test_pooled_nulls_match_analytic():
    # the hypergeometric pooled null centres on the analytic pooled chance overlap.
    km = [20, 50, 5]
    Nm = [144, 400, 64]
    analytic = C.pooled_analytic_null(km, Nm)
    samples = C.pooled_permuted_null(km, Nm, n_draws=4000, seed=3)
    assert float(samples.mean()) == pytest.approx(analytic, abs=0.01)
    # single-matrix analytic = k/N
    assert C.pooled_analytic_null([7], [100]) == pytest.approx(7 / 100, abs=1e-12)


def test_pooled_overlap_sums_intersections():
    a1 = torch.tensor([True, True, False, False])
    b1 = torch.tensor([True, False, True, False])              # inter 1, k 2
    a2 = torch.tensor([True, True, False, False])
    b2 = torch.tensor([True, True, False, False])              # inter 2, k 2
    O, ksum, nm = C.pooled_overlap([(a1, b1), (a2, b2)])
    assert (O, ksum, nm) == (pytest.approx(3 / 4), 4, 2)


def test_f_perp_in_span_zero_out_of_span_one():
    torch.manual_seed(7)
    Wb = torch.randn(6, 4, dtype=torch.float64)                # m>n → left complement exists
    U, s, V = C.svd_fp64(Wb)
    R_eff = U.shape[1]                                         # full span
    # in-span gradient → f_perp ≈ 0
    Gin = U @ torch.randn(U.shape[1], V.shape[1], dtype=torch.float64) @ V.transpose(0, 1)
    assert C.grad_out_of_span_frac(Gin, U, V, R_eff) < 1e-10
    # gradient orthogonal to U's column span → f_perp ≈ 1
    Z = torch.randn(6, 4, dtype=torch.float64)
    Gout = Z - U @ (U.transpose(0, 1) @ Z)
    assert C.grad_out_of_span_frac(Gout, U, V, R_eff) > 1.0 - 1e-10


# ---------------------------------------------------------------------------
# Frozen decision predicates (the §9.6 unblind logic must be auditable + tested).
# ---------------------------------------------------------------------------

def test_install_verdict_rank_tail_kill():
    # oracle at/below null+δ → KILL regardless of leverage.
    v = C.install_verdict(r_oracle=0.01, r_leverage=0.9, r_energy=0.0,
                          r_wholeloss=0.0, null_hi=0.02, delta_install=0.02,
                          seeds_ok=True, fdr_ok=True)
    assert v == "I-RANK-TAIL"


def test_install_verdict_orderable():
    v = C.install_verdict(r_oracle=0.6, r_leverage=0.5, r_energy=0.0,
                          r_wholeloss=0.1, null_hi=0.1, delta_install=0.05,
                          seeds_ok=True, fdr_ok=True)
    assert v == "I-ORDERABLE"


def test_install_verdict_murky_when_leverage_ties_wholeloss():
    v = C.install_verdict(r_oracle=0.6, r_leverage=0.3, r_energy=0.0,
                          r_wholeloss=0.3, null_hi=0.1, delta_install=0.05,
                          seeds_ok=True, fdr_ok=True)
    assert v == "I-MURKY"


def test_retention_verdict_separable_coincident_anti_instrumentnull():
    # separable: O below null p5 AND below both magnitude controls
    assert C.retention_verdict(0.05, null_p5=0.10, null_p95=0.30,
                               o_forget_mag_p5=0.20, o_install_mag_p5=0.20,
                               seeds_ok=True, fdr_ok=True) == "R-SEPARABLE"
    # instrument-null: O not below the magnitude controls → maps are magnitude
    assert C.retention_verdict(0.25, null_p5=0.10, null_p95=0.30,
                               o_forget_mag_p5=0.20, o_install_mag_p5=0.20,
                               seeds_ok=True, fdr_ok=True) == "R-INSTRUMENT-NULL"
    # anti: O above null p95 (and below magnitude controls so it isn't instrument-null)
    assert C.retention_verdict(0.35, null_p5=0.10, null_p95=0.30,
                               o_forget_mag_p5=0.40, o_install_mag_p5=0.40,
                               seeds_ok=True, fdr_ok=True) == "R-ANTI"
    # coincident: in-band
    assert C.retention_verdict(0.20, null_p5=0.10, null_p95=0.30,
                               o_forget_mag_p5=0.40, o_install_mag_p5=0.40,
                               seeds_ok=True, fdr_ok=True) == "R-COINCIDENT"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
