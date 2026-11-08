"""Weight-free CPU dry-run gate for ENDOW-O (endow spec §9.1) — Procrustes transport + Site-R.

NO model, NO datasets, NO GPU — tiny planted tensors, seconds on a CPU box. Proves: the orthogonal
Procrustes fit recovers a planted rotation and is identity on matched streams; transport_basis maps
a src span into the frame where it removes the same variance; ‖R−I‖_F behaves; the Site-R KV-branch
projector obeys the mean-ablation contract.

Run (on a remote `pg` CPU instance — never locally; torch is a large lib):
    python -m pytest infra/pg/tests/test_endow_o.py -q
"""
from __future__ import annotations

import pytest
import torch

from infra.pg import endow_o as EO
from infra.pg.ctrliface_core import principal_angle_overlap, orthonormalize


def _ortho(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return Q


def _basis(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64)).Q[:, :k].contiguous()


def test_procrustes_recovers_planted_rotation():
    d, n = 8, 200
    src = torch.randn(n, d, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    Q = _ortho(d, seed=2)
    dst = src @ Q
    R = EO.orthogonal_procrustes(src, dst, center=False)
    assert EO.is_orthogonal(R)
    assert torch.allclose(R, Q, atol=1e-6)


def test_procrustes_identity_on_matched_streams():
    d, n = 6, 100
    src = torch.randn(n, d, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    R = EO.orthogonal_procrustes(src, src.clone(), center=True)
    assert torch.allclose(R, torch.eye(d, dtype=torch.float64), atol=1e-6)
    assert EO.frob_rel_identity(R) < 1e-6


def test_transport_basis_maps_span_to_the_right_frame():
    d, k, n = 10, 3, 300
    src = torch.randn(n, d, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    Q = _ortho(d, seed=5)
    dst = src @ Q
    R = EO.orthogonal_procrustes(src, dst, center=False)
    V = _basis(d, k, seed=6)                                   # a src-coordinate span
    Vt = EO.transport_basis(R, V)
    assert torch.allclose(Vt.transpose(0, 1) @ Vt, torch.eye(k, dtype=torch.float64), atol=1e-9)
    # transported span == the true image Qᵀ·V of the src span (overlap ≈ 1)
    truth = orthonormalize(Q.transpose(0, 1) @ V)
    ov, _ = principal_angle_overlap(Vt, truth)
    assert ov > 1 - 1e-8
    # and it removes the SAME variance: dst·V' ≈ src·V up to the orthonormal gauge (norms match)
    assert torch.allclose((dst @ Vt).pow(2).sum(), (src @ V).pow(2).sum(), rtol=1e-6)


def test_frob_rel_identity_grows_with_rotation():
    d = 8
    assert EO.frob_rel_identity(torch.eye(d, dtype=torch.float64)) == pytest.approx(0.0, abs=1e-12)
    R = _ortho(d, seed=9)
    assert EO.frob_rel_identity(R) > 0.0


def test_transported_overlap_identity_is_one():
    d, k = 12, 4
    V = _basis(d, k, seed=7)
    assert EO.transported_overlap(torch.eye(d, dtype=torch.float64), V) == pytest.approx(1.0, abs=1e-9)


def test_kv_branch_ablate_obeys_projector_contract():
    d, k, T = 9, 3, 5
    Q = _basis(d, k, seed=8)
    hbar = torch.randn(d, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    kv = torch.randn(T, d, dtype=torch.float64, generator=torch.Generator().manual_seed(12))
    kv2 = EO.kv_branch_ablate(kv, Q, hbar)
    # after ablation the residual (kv' − h̄) is orthogonal to span(Q): Qᵀ(kv'−h̄) ≈ 0
    resid = (kv2 - hbar) @ Q
    assert float(resid.abs().max()) < 1e-9
    # and the complement is untouched: kv' agrees with kv off span(Q)
    Pc = torch.eye(d, dtype=torch.float64) - Q @ Q.transpose(0, 1)
    assert torch.allclose(kv2 @ Pc, kv @ Pc, atol=1e-9)


def test_procrustes_rejects_unpaired():
    with pytest.raises(AssertionError):
        EO.orthogonal_procrustes(torch.randn(5, 3, dtype=torch.float64),
                                 torch.randn(6, 3, dtype=torch.float64))   # unpaired n


def test_procrustes_underdetermined_still_orthogonal():
    # n < d is allowed: AᵀB is d×d, so R stays orthogonal (arbitrary on the unspanned complement).
    R = EO.orthogonal_procrustes(torch.randn(3, 8, dtype=torch.float64),
                                 torch.randn(3, 8, dtype=torch.float64))
    assert R.shape == (8, 8) and EO.is_orthogonal(R)
