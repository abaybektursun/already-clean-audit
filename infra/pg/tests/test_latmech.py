"""Weight-free CPU dry-run gate for LATMECH (the prereg §9.1 gate).

NO model, NO datasets, NO GPU — tiny planted tensors only, <1s on a CPU box. Proves the
PURE logic (project-out ablation algebra + removed-energy accounting, energy-matched-random
draw determinism, shrink-perturb magnitude, the recovery-fraction MDE + bootstrap lockstep,
carrier-k selection, the position-alignment hash tripwire, the cloze λ-flip, and the §9.6
end-to-end unblind verdict on synthetic banked cells) BEFORE any GPU minute is spent. Distinct
from the runtime preflight (§9.3, which proves the WIRING on real weights).

Run (on a remote `pg` CPU instance — never locally; torch is a large lib):
    python -m pytest infra/pg/tests/test_latmech.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from infra.pg import latmech as L
from infra.pg import w15_carrier as C
from infra.pg import reprice_stats as R


def _rand_W(m=7, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# project_out_subspace + removed_energy — the ablation algebra (§9.1).
# ---------------------------------------------------------------------------
def test_removed_energy_equals_frobenius_change():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    idx = torch.tensor([0, 2])
    W_abl = L.project_out_subspace(W, U, S, V, idx)
    # ‖W − W_abl‖²_F = Σ σ_i² over idx (u_iv_iᵀ orthonormal in the trace inner product).
    assert L.removed_energy(S, idx) == pytest.approx(float((W - W_abl).norm() ** 2), rel=1e-9)
    assert L.removed_energy(S, idx) == pytest.approx(float((S[idx] ** 2).sum()), rel=1e-12)


def test_project_out_energy_identity():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    idx = torch.tensor([1, 3])
    W_abl = L.project_out_subspace(W, U, S, V, idx)
    # orthogonality ⇒ ‖W_abl‖²_F = ‖W‖²_F − removed_energy.
    assert float(W_abl.norm() ** 2) == pytest.approx(
        float(W.norm() ** 2) - L.removed_energy(S, idx), rel=1e-9)


def test_project_out_all_directions_gives_zero():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    idx = torch.arange(S.numel())
    W_abl = L.project_out_subspace(W, U, S, V, idx)
    assert float(W_abl.norm()) == pytest.approx(0.0, abs=1e-9)


def test_project_out_empty_idx_is_identity():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    W_abl = L.project_out_subspace(W, U, S, V, torch.tensor([], dtype=torch.long))
    assert torch.allclose(W, W_abl)


# ---------------------------------------------------------------------------
# carrier_direction_idx — the content carrier picks the gradient-loaded direction.
# ---------------------------------------------------------------------------
def test_carrier_picks_the_content_loaded_direction():
    W = _rand_W(m=6, n=6, seed=3)
    U, S, V = C.svd_fp64(W)
    j = 4
    # a gradient that loads ONLY direction j: G = u_j v_jᵀ ⇒ score_d = u_dᵀ G v_d = δ_dj.
    G = torch.outer(U[:, j], V[:, j])
    idx = L.carrier_direction_idx(G, U, V, r=1)
    assert idx.tolist() == [j]
    # top-2 must still include j.
    assert j in L.carrier_direction_idx(G, U, V, r=2).tolist()


# ---------------------------------------------------------------------------
# energy_matched_random_idx — matched energy, determinism, exclusion.
# ---------------------------------------------------------------------------
def test_energy_matched_random_hits_target_within_one_quantum():
    S = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    target = float((S[[0, 3]] ** 2).sum())          # = 25 + 4 = 29
    idx, achieved = L.energy_matched_random_idx(S, target, seed=1)
    # greedy match: the gap cannot exceed the largest single σ² (else a swap would help).
    assert abs(achieved - target) <= float((S ** 2).max()) + 1e-9
    assert L.removed_energy(S, idx) == pytest.approx(achieved, rel=1e-12)


def test_energy_matched_random_is_deterministic_and_seed_sensitive():
    S = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    t = 20.0
    a, _ = L.energy_matched_random_idx(S, t, seed=7)
    b, _ = L.energy_matched_random_idx(S, t, seed=7)
    assert a.tolist() == b.tolist()                  # same seed → identical
    diffs = {tuple(L.energy_matched_random_idx(S, t, seed=s)[0].tolist()) for s in range(8)}
    assert len(diffs) > 1                             # seed actually varies the draw


def test_energy_matched_random_excludes_carrier():
    S = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    excl = torch.tensor([0, 1])
    idx, _ = L.energy_matched_random_idx(S, 6.0, seed=2, exclude=excl)
    assert set(idx.tolist()).isdisjoint(set(excl.tolist()))


# ---------------------------------------------------------------------------
# shrink_perturb — identity at (1,0), determinism, magnitude monotone in σ.
# ---------------------------------------------------------------------------
def test_shrink_perturb_identity_and_determinism():
    W = _rand_W()
    assert torch.allclose(L.shrink_perturb(W, 1.0, 0.0, seed=0), W)
    a = L.shrink_perturb(W, 0.9, 0.1, seed=5)
    b = L.shrink_perturb(W, 0.9, 0.1, seed=5)
    assert torch.allclose(a, b)


def test_shrink_perturb_magnitude_monotone_in_sigma():
    W = _rand_W(m=40, n=40, seed=11)
    d_small = L.delta_frob(W, L.shrink_perturb(W, 1.0, 0.02, seed=1))
    d_big = L.delta_frob(W, L.shrink_perturb(W, 1.0, 0.20, seed=1))
    assert d_big > d_small > 0


# ---------------------------------------------------------------------------
# mde_contrast — Δ_content − Δ_emrand = RF(emrand) − RF(content); bootstrap lockstep.
# ---------------------------------------------------------------------------
def _planted_mde(content_rate, emrand_rate, n=200):
    """floor=0, ceiling=1 ⇒ RF = mean(S); plant exact per-item rates."""
    content_S = np.array([1.0 if (i % 10) < content_rate * 10 else 0.0 for i in range(n)])
    emrand_S = np.array([1.0 if (i % 10) < emrand_rate * 10 else 0.0 for i in range(n)])
    floor = np.zeros(n)
    ceiling = np.ones(n)
    return content_S, emrand_S, floor, ceiling


def test_mde_point_equals_rf_difference():
    cS, eS, F, Cc = _planted_mde(0.3, 0.6)
    res = L.mde_contrast(cS, eS, F, Cc, n_boot=200, seed=1337)
    # RF(emrand) − RF(content) with floor 0, ceiling 1 = mean(emrand) − mean(content) = 0.3.
    assert res["point"] == pytest.approx(eS.mean() - cS.mean(), abs=1e-9)
    assert res["point"] == pytest.approx(0.3, abs=1e-9)


def test_mde_hstore_signal_ci_above_zero():
    cS, eS, F, Cc = _planted_mde(0.3, 0.6)          # content hurts re-entry MORE → H-store
    res = L.mde_contrast(cS, eS, F, Cc, n_boot=2000, seed=1337)
    assert res["lo"] > 0                              # CI entirely > 0


def test_mde_null_brackets_zero():
    cS, eS, F, Cc = _planted_mde(0.5, 0.5)          # content == emrand → no privilege
    res = L.mde_contrast(cS, eS, F, Cc, n_boot=2000, seed=1337)
    assert res["lo"] <= 0 <= res["hi"]


def test_mde_bootstrap_is_deterministic():
    cS, eS, F, Cc = _planted_mde(0.3, 0.6)
    r1 = L.mde_contrast(cS, eS, F, Cc, n_boot=500, seed=99)
    r2 = L.mde_contrast(cS, eS, F, Cc, n_boot=500, seed=99)
    assert (r1["lo"], r1["hi"]) == (r2["lo"], r2["hi"])


# ---------------------------------------------------------------------------
# select_carrier_k — the §6 energy-controlled validity gate.
# ---------------------------------------------------------------------------
def test_select_carrier_k_smallest_passing():
    k_grid = [8, 16, 32, 64]
    content_drop = [0.05, 0.20, 0.30, 0.40]
    emrand_drop = [0.02, 0.05, 0.08, 0.20]
    assert L.select_carrier_k(k_grid, content_drop, emrand_drop, thresh=0.15) == 16


def test_select_carrier_k_none_when_emrand_also_drops():
    # content drops but the energy-matched random ALSO drops past thresh → gate fails (no carrier).
    assert L.select_carrier_k([8], [0.6], [0.6], thresh=0.15) is None


def test_select_carrier_k_none_when_content_too_weak():
    assert L.select_carrier_k([8, 16], [0.05, 0.08], [0.01, 0.02], thresh=0.15) is None


# ---------------------------------------------------------------------------
# position_align_tripwire — pairing reorder guard (qa_item_hash NOT a join key).
# ---------------------------------------------------------------------------
def test_position_tripwire_passes_on_identical():
    seq = ["a1", "b2", "c3"]
    assert L.position_align_tripwire(seq, list(seq)) is True


def test_position_tripwire_raises_on_reorder():
    with pytest.raises(AssertionError):
        L.position_align_tripwire(["a1", "b2", "c3"], ["a1", "c3", "b2"])


def test_position_tripwire_raises_on_length_mismatch():
    with pytest.raises(AssertionError):
        L.position_align_tripwire(["a1", "b2"], ["a1", "b2", "c3"])


# ---------------------------------------------------------------------------
# cloze λ-flip — the reused content readout's length-norm actually bites.
# ---------------------------------------------------------------------------
def test_cloze_lambda_flip():
    # choice 0: 1 token, logprob −3; choice 1: 3 tokens, logprob −4.
    # λ=0 (raw sum): 0 (−3) beats 1 (−4). λ=1 (per-token): 1 (−1.33) beats 0 (−3).
    logprobs = np.array([[-3.0, -4.0, -9.0, -9.0]])
    ntoks = np.array([[1, 3, 1, 1]])
    assert R.cloze_predict(logprobs, ntoks, lam=0.0)[0] == 0
    assert R.cloze_predict(logprobs, ntoks, lam=1.0)[0] == 1


# ---------------------------------------------------------------------------
# mde_verdict — the §9.6 end-to-end unblind on synthetic banked cells (each H-*).
# ---------------------------------------------------------------------------
_BRACKET = {"lo": -0.05, "hi": 0.05}          # CI brackets 0
_POS = {"lo": 0.10, "hi": 0.30}               # CI entirely > 0
_EXCLUDES = {"lo": 0.10, "hi": 0.30}          # CI excludes 0 (shrink-perturb moves)


def test_verdict_h_store():
    assert L.mde_verdict(_POS, _BRACKET, _BRACKET, seed_signs=[1, 1, 1]) == "H_store"


def test_verdict_h_store_rejected_when_seeds_flip():
    # positive CI but sign flips across seeds → NOT H_store (seed-instability ⇒ non-result).
    assert L.mde_verdict(_POS, _BRACKET, _BRACKET, seed_signs=[1, -1, 1]) != "H_store"


def test_verdict_h_basin():
    assert L.mde_verdict(_BRACKET, _EXCLUDES, _BRACKET, seed_signs=[1, -1, 1]) == "H_basin"


def test_verdict_h_data():
    assert L.mde_verdict(_BRACKET, _BRACKET, _BRACKET, seed_signs=[1, -1]) == "H_data"


def test_verdict_h_mixed_when_fdr_fails():
    # positive CI + stable seeds but FDR not cleared → not promotable to H_store.
    assert L.mde_verdict(_POS, _BRACKET, _EXCLUDES, seed_signs=[1, 1], mde_fdr_ok=False) != "H_store"


# ---------------------------------------------------------------------------
# interpolate_weights — Arm-2 spectrum endpoints + midpoint.
# ---------------------------------------------------------------------------
def test_interpolate_weights_endpoints():
    Wd = _rand_W(seed=1)
    Wf = _rand_W(seed=2)
    assert torch.allclose(L.interpolate_weights(Wd, Wf, 0.0), Wd)
    assert torch.allclose(L.interpolate_weights(Wd, Wf, 1.0), Wf)
    assert torch.allclose(L.interpolate_weights(Wd, Wf, 0.5), 0.5 * (Wd + Wf))
