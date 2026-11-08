"""Weight-free CPU unit tests for the W16 dispositional-audit probes
(infra/pg/w16_probes.py). NO model, NO datasets, NO GPU — tiny planted tensors
only, <1s on a laptop. This IS the W16 prereg §9.1 OFFLINE DRY-RUN gate: every
pure logic the GPU session depends on is proved here BEFORE any GPU minute.

    python -m pytest infra/pg/tests/test_w16_probes.py -q

Maps onto the W16 dry-run checklist (program spec §3.1):
  * quantizer round-trip: error ≤ half-step, idempotent, row-scale equivariant,
    int4 coarser than int8, zero rows preserved              (test_quant_*)
  * probe scope: exactly the 2-D non-embed params; embed/lm_head/1-D never
    touched; snapshot→probe→restore is bitwise               (test_scope_*, test_restore_*)
  * noise: deterministic in seed, seeds differ, realized scale ≈ σ·std        (test_noise_*)
  * margin/Brier/AUC arithmetic on planted logits            (test_readout_*)
  * decision-rule truth table on synthetic ΔS grids          (test_rule_*)
  * discovery/confirmation split determinism (swap_back reuse)(test_split_*)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from infra.pg import w16_probes as W
from infra.pg.swap_back import subset_by_hash


# ---------------------------------------------------------------------------
# A tiny stand-in language_model: embed + 2 "decoder" linears + 1-D norm.
# ---------------------------------------------------------------------------
class TinyLM(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed_tokens = nn.Embedding(11, 4)
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 6, bias=False)
        self.norm = nn.Parameter(torch.randn(4, generator=g))
        with torch.no_grad():
            self.embed_tokens.weight.copy_(torch.randn(11, 4, generator=g))
            self.q_proj.weight.copy_(torch.randn(4, 4, generator=g))
            self.down_proj.weight.copy_(torch.randn(6, 4, generator=g))


# ---------------------------------------------------------------------------
# Quantizer.
# ---------------------------------------------------------------------------
def test_quant_error_below_half_step_per_row():
    w = torch.randn(16, 32, generator=torch.Generator().manual_seed(1))
    for bits in W.QUANT_BITS:
        q = W.fake_quant_tensor(w, bits)
        qmax = 2 ** (bits - 1) - 1
        step = w.abs().amax(dim=1, keepdim=True) / qmax
        assert ((q - w).abs() <= step / 2 + 1e-7).all(), f"int{bits} exceeds half-step"


def test_quant_idempotent_fp32():
    w = torch.randn(8, 8, generator=torch.Generator().manual_seed(2))
    for bits in W.QUANT_BITS:
        q1 = W.fake_quant_tensor(w, bits)
        q2 = W.fake_quant_tensor(q1, bits)
        assert torch.equal(q1, q2), f"int{bits} not idempotent"


def test_quant_row_scale_equivariant():
    # per-output-channel absmax ⇒ scaling one row scales its quantization.
    w = torch.randn(6, 10, generator=torch.Generator().manual_seed(3))
    q = W.fake_quant_tensor(w, 8)
    w2 = w.clone()
    w2[3] *= 7.5
    q2 = W.fake_quant_tensor(w2, 8)
    assert torch.allclose(q2[3], q[3] * 7.5, atol=1e-5)
    assert torch.equal(q2[:3], q[:3]) and torch.equal(q2[4:], q[4:])


def test_quant_int4_coarser_than_int8():
    w = torch.randn(32, 64, generator=torch.Generator().manual_seed(4))
    e8 = (W.fake_quant_tensor(w, 8) - w).norm()
    e4 = (W.fake_quant_tensor(w, 4) - w).norm()
    assert e4 > e8 > 0


def test_quant_zero_rows_pass_through():
    w = torch.zeros(4, 5)
    w[2] = torch.randn(5, generator=torch.Generator().manual_seed(5))
    q = W.fake_quant_tensor(w, 4)
    assert torch.equal(q[0], torch.zeros(5)) and torch.isfinite(q).all()


def test_quant_grid_levels_bounded():
    # distinct values per row ≤ 2·qmax + 1 (the grid size).
    w = torch.randn(3, 4096, generator=torch.Generator().manual_seed(6))
    q = W.fake_quant_tensor(w, 4)
    for r in range(3):
        assert q[r].unique().numel() <= 2 * 7 + 1


# ---------------------------------------------------------------------------
# Probe scope + snapshot/restore.
# ---------------------------------------------------------------------------
def test_scope_is_exactly_2d_non_embed():
    lm = TinyLM()
    names = W.probe_param_names(lm)
    assert names == ["down_proj.weight", "q_proj.weight"]


def test_quantize_decoder_touches_scope_only():
    lm = TinyLM()
    emb0 = lm.embed_tokens.weight.detach().clone()
    norm0 = lm.norm.detach().clone()
    rep = W.quantize_decoder_(lm, 4)
    assert rep["n_tensors"] == 2 and rep["rel_frob_max"] > 0
    assert torch.equal(lm.embed_tokens.weight, emb0), "embed mutated — scope leak"
    assert torch.equal(lm.norm, norm0), "1-D norm mutated — scope leak"
    assert (lm.q_proj.weight - TinyLM().q_proj.weight).abs().max() > 0  # scope DID mutate


def test_restore_is_bitwise():
    lm = TinyLM()
    snap = W.snapshot_probe_params(lm)
    ref = {n: lm.get_parameter(n).detach().clone() for n, _ in lm.named_parameters()}
    W.quantize_decoder_(lm, 4)
    W.noise_decoder_(lm, 3e-2, seed=7)
    W.restore_probe_params_(lm, snap)
    for n, _ in lm.named_parameters():
        assert torch.equal(lm.get_parameter(n), ref[n]), f"{n} not bitwise-restored"


def test_restore_rejects_scope_mismatch():
    lm = TinyLM()
    snap = W.snapshot_probe_params(lm)
    del snap["q_proj.weight"]
    with pytest.raises(AssertionError):
        W.restore_probe_params_(lm, snap)


# ---------------------------------------------------------------------------
# Noise probe.
# ---------------------------------------------------------------------------
def test_noise_deterministic_in_seed():
    a, b = TinyLM(), TinyLM()
    W.noise_decoder_(a, 1e-2, seed=1337)
    W.noise_decoder_(b, 1e-2, seed=1337)
    assert torch.equal(a.q_proj.weight, b.q_proj.weight)
    assert torch.equal(a.down_proj.weight, b.down_proj.weight)


def test_noise_seeds_differ():
    a, b = TinyLM(), TinyLM()
    W.noise_decoder_(a, 1e-2, seed=1337)
    W.noise_decoder_(b, 1e-2, seed=1338)
    assert not torch.equal(a.q_proj.weight, b.q_proj.weight)


def test_noise_realized_scale_tracks_sigma():
    # big matrix → relative Frobenius change concentrates near σ_rel.
    lm = nn.Module()
    lm.big = nn.Linear(256, 256, bias=False)
    sigma = 1e-2
    rep = W.noise_decoder_(lm, sigma, seed=0)
    assert rep["n_tensors"] == 1
    assert abs(rep["rel_frob_mean"] - sigma) / sigma < 0.2


def test_noise_scope_excludes_embed_and_1d():
    lm = TinyLM()
    emb0 = lm.embed_tokens.weight.detach().clone()
    norm0 = lm.norm.detach().clone()
    W.noise_decoder_(lm, 3e-2, seed=3)
    assert torch.equal(lm.embed_tokens.weight, emb0)
    assert torch.equal(lm.norm, norm0)


# ---------------------------------------------------------------------------
# Continuous readouts on planted logits.
# ---------------------------------------------------------------------------
def test_readout_margin_known_values():
    L = [[3.0, 1.0, 0.0, -1.0],      # correct=0 → margin +2
         [0.0, 5.0, 9.0, 1.0]]       # correct=1 → margin 5−9 = −4
    m = W.margin_per_item(L, [0, 1])
    assert m.tolist() == [2.0, -4.0]


def test_readout_brier_uniform_and_perfect():
    uni = [[0.0, 0.0, 0.0, 0.0]]
    assert abs(W.brier_per_item(uni, [2])[0] - 0.75) < 1e-12     # uniform 4-way
    sharp = [[100.0, 0.0, 0.0, 0.0]]
    assert W.brier_per_item(sharp, [0])[0] < 1e-12               # confidently right
    wrong = [[0.0, 100.0, 0.0, 0.0]]
    assert abs(W.brier_per_item(wrong, [0])[0] - 2.0) < 1e-12    # confidently wrong


def test_readout_auc_extremes_and_chance():
    top = [[9.0, 1.0, 2.0, 3.0], [0.5, 7.0, -1.0, 0.0]]
    assert W.rank_auc(top, [0, 1]) == 1.0                        # correct always top
    bottom = [[-9.0, 1.0, 2.0, 3.0]]
    assert W.rank_auc(bottom, [0]) == 0.0                        # correct always last
    ties = [[1.0, 1.0, 1.0, 1.0]]
    assert W.rank_auc(ties, [3]) == 0.5                          # all-tied → chance


def test_readout_margin_shift_invariance():
    # adding a per-item constant (the logsumexp term) must not move the margin.
    g = torch.Generator().manual_seed(8)
    L = torch.randn(50, 4, generator=g)
    c = torch.randint(0, 4, (50,), generator=g)
    m1 = W.margin_per_item(L.tolist(), c.tolist())
    m2 = W.margin_per_item((L + torch.randn(50, 1, generator=g)).tolist(), c.tolist())
    assert ((m1 - m2) ** 2).max() < 1e-12


def test_readout_misaligned_shapes_raise():
    with pytest.raises(AssertionError):
        W.margin_per_item([[1.0, 2.0, 3.0]], [0])                # 3 choices
    with pytest.raises(AssertionError):
        W.margin_per_item([[1.0, 2.0, 3.0, 4.0]], [0, 1])        # N mismatch


# ---------------------------------------------------------------------------
# Decision arithmetic (prereg §6, post-critic): ΔS, FLOOR-side bands with
# shared-index bootstrap, den≤0 handling, seed-mean, Arm-A paired diff, BH.
# ---------------------------------------------------------------------------
def _rng_binary(n, p, seed):
    import numpy as np
    return (np.random.default_rng(seed).random(n) < p).astype(float).tolist()


def test_rule_delta_s_known_values():
    probed = [1, 1, 1, 0]      # 0.75
    unprobed = [1, 0, 0, 0]    # 0.25
    before = [1, 1, 1, 1]      # 1.0
    floor = [0, 0, 0, 0]       # 0.0  → den 1.0
    assert W.delta_s(probed, unprobed, before, floor) == pytest.approx(0.5)
    with pytest.raises(AssertionError):                       # den ≤ 0 raises (point)
        W.delta_s(probed, unprobed, floor, before)


def test_rule_grid_points_argmax():
    unp = [0, 0, 1, 0]; bef = [1, 1, 1, 1]; flo = [0, 0, 0, 0]
    pts, mx, am = W.grid_points({"a": [1, 1, 1, 0], "b": [0, 0, 1, 0]}, unp, bef, flo)
    assert am == "a" and mx == pytest.approx(0.5) and pts["b"] == pytest.approx(0.0)


def test_rule_null_band_centered_and_null_safe():
    # FLOOR-side cells with NO probe effect (probed ≡ unprobed distribution):
    # band_hi small + positive (max over cells of 0-centered noise); a real
    # AFTER spike lands far outside; a null AFTER max lands inside.
    n = 600
    unp = _rng_binary(n, 0.25, 0)
    cells = {f"c{i}": _rng_binary(n, 0.25, 10 + i) for i in range(6)}
    bef = _rng_binary(n, 0.55, 7)
    flo = _rng_binary(n, 0.25, 8)
    band = W.max_null_band(cells, unp, bef, flo, n_boot=400, seed=1)
    assert 0.0 < band["band_hi"] < 0.6
    assert band["band_width"] > 0 and band["n_draws_kept"] >= 200
    # one-sided p: huge observed max → tiny p; observed 0 → large p.
    assert W.one_sided_p(5.0, band["draws"]) < 0.01
    assert W.one_sided_p(0.0, band["draws"]) > 0.2


def test_rule_shared_index_pairing():
    # two IDENTICAL cells: with a SHARED index per draw, max over both ==
    # the single-cell draw exactly (pairing proof).
    n = 300
    unp = _rng_binary(n, 0.3, 0)
    cell = _rng_binary(n, 0.3, 1)
    bef = _rng_binary(n, 0.7, 2); flo = _rng_binary(n, 0.3, 3)
    b2 = W.max_null_band({"x": cell, "y": list(cell)}, unp, bef, flo, n_boot=200, seed=5)
    b1 = W.max_null_band({"x": cell}, unp, bef, flo, n_boot=200, seed=5)
    assert b2["draws"] == b1["draws"]


def test_rule_den_le_zero_draws_dropped_and_counted():
    # BEFORE barely above FLOOR → some resamples flip den ≤ 0 → dropped+counted.
    n = 40
    bef = [1] * 21 + [0] * 19      # 0.525
    flo = [1] * 19 + [0] * 21      # 0.475
    unp = _rng_binary(n, 0.5, 0)
    band = W.cell_null_band([_rng_binary(n, 0.5, 1)], unp, bef, flo, n_boot=300, seed=2)
    assert band["n_draws_dropped"] > 0
    assert band["n_draws_kept"] + band["n_draws_dropped"] == 300


def test_rule_seed_mean_matches_manual():
    unp = [0, 0, 0, 0]; bef = [1, 1, 1, 1]; flo = [0, 0, 0, 0]
    s = W.seed_mean_delta_s([[1, 0, 0, 0], [0, 1, 1, 0]], unp, bef, flo)
    assert s == pytest.approx((0.25 + 0.5) / 2)


def test_rule_arm_a_paired_diff_null_and_signal():
    import numpy as np
    g = np.random.default_rng(3)
    n = 400
    F = g.normal(0, 1, (n, 4))
    c = g.integers(0, 4, n)
    same = W.arm_a_paired_diff(F.tolist(), F.tolist(), c.tolist(), n_boot=300, seed=4)
    assert same["margin_diff_point"] == 0.0 and same["auc_diff_point"] == 0.0
    assert same["margin_p_two_sided"] == 1.0 and same["auc_p_two_sided"] == 1.0
    A = F.copy()
    A[np.arange(n), c] += 3.0                                  # strong correct-answer signal
    sig = W.arm_a_paired_diff(A.tolist(), F.tolist(), c.tolist(), n_boot=300, seed=4)
    assert sig["margin_diff_point"] > 1.0 and sig["margin_p_two_sided"] < 0.01
    assert sig["auc_diff_point"] > 0.1 and sig["auc_p_two_sided"] < 0.01
    assert sig["margin_diff_ci95"][0] > 0


def test_rule_bh_fdr_truth_table():
    out = W.bh_fdr({"a": 0.001, "b": 0.02, "c": 0.9}, q=0.05)
    assert out["a"]["significant"] and out["b"]["significant"] and not out["c"]["significant"]
    assert out["a"]["p_adj"] <= out["b"]["p_adj"] <= out["c"]["p_adj"]
    out2 = W.bh_fdr({"a": 0.04, "b": 0.9}, q=0.05)             # rank-1 of 2: 0.04*2=0.08 > .05
    assert not out2["a"]["significant"]
    with pytest.raises(AssertionError):
        W.bh_fdr({})


# ---------------------------------------------------------------------------
# Eval-economics predicates: grid subset ⊂ discovery, fractions, parity match.
# ---------------------------------------------------------------------------
def test_split_predicates_partition_and_grid_subset():
    import hashlib
    hashes = [hashlib.sha1(f"item-{i}".encode()).hexdigest()[:12] for i in range(14042)]
    disc = [h for h in hashes if W.is_discovery(h)]
    conf = [h for h in hashes if W.is_confirmation(h)]
    grid = [h for h in hashes if W.is_grid_subset(h)]
    assert len(disc) + len(conf) == len(hashes)                   # exact partition
    assert set(grid) <= set(disc), "grid subset leaks confirmation items"
    # ≈ 2/7 of the discovery half ≈ 2,000 items (uniform sha1 ⇒ tight)
    assert 1700 <= len(grid) <= 2300, f"grid subset size {len(grid)} off target"
    # parity convention must MATCH swap_back.subset_by_hash exactly
    assert conf == subset_by_hash(hashes, hashes, "confirmation")
    assert disc == subset_by_hash(hashes, hashes, "discovery")


def test_split_predicates_deterministic():
    h = "00000000000e"                                            # even → discovery
    assert W.is_discovery(h) and not W.is_confirmation(h)
    h2 = "00000000000f"                                           # odd → confirmation
    assert W.is_confirmation(h2) and not W.is_grid_subset(h2)


# ---------------------------------------------------------------------------
# Discovery/confirmation split (swap_back.subset_by_hash reuse) determinism.
# ---------------------------------------------------------------------------
def test_split_partition_and_determinism():
    hashes = [f"{i:012x}" for i in range(40)]
    vals = list(range(40))
    disc = subset_by_hash(vals, hashes, "discovery")
    conf = subset_by_hash(vals, hashes, "confirmation")
    assert sorted(disc + conf) == vals                            # exact partition
    assert disc and conf
    assert disc == subset_by_hash(vals, hashes, "discovery")      # deterministic
    assert subset_by_hash(vals, hashes, "all") == vals


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
