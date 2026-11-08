"""Weight-free CPU dry-run gate for invariant_core_func — the prereg §9.1 logic test.
NO model, NO datasets, NO GPU — pure logic on tiny synthetic tensors, seconds on a laptop.

    python3 -m pytest infra/aws_pg/tests/test_invariant_core_func.py -q

Covers every §9.1 checklist item:
  * planted-band recovery (U_unt overlap == 1)               test_bottom_eigvecs_recovers_planted_band
  * LIVE-activations -> ratio >> 2; INERT -> ratio ~= 0      test_act_energy_ratio_live_vs_inert
  * projector idempotent (resid < 1e-9) + removes the band   test_projector_*, test_input_project_hook
  * top-sigma positive control selects MOST-used directions  test_sigma_ends_select_largest_smallest
  * paired-bootstrap CI brackets a planted contrast / 0      test_paired_contrast_ci_*
  * random-band averaging + FDR + random subspace plumbing   test_avg_per_window_*, test_bh_*, test_random_*
"""
import math

import numpy as np
import pytest
import torch

from infra.aws_pg.jobs import invariant_core_func as J


# ---------------------------------------------------------------------------
# Band construction: bottom-k eigvecs recover a planted low-energy subspace.
# ---------------------------------------------------------------------------
def test_bottom_eigvecs_recovers_planted_band():
    rng = np.random.default_rng(0)
    n, k = 64, 8
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eig = np.full(n, 10.0)
    eig[:k] = 1e-3                                  # first k columns = the planted "untouched" band
    G = (Q * eig) @ Q.T
    G = 0.5 * (G + G.T)
    U, w = J._bottom_eigvecs(G, k)
    M = Q[:, :k].T @ U                              # k x k overlap of the two spans
    overlap = float(np.sum(M ** 2)) / k             # == 1 iff spans coincide
    assert overlap > 0.999, overlap
    assert float(w.max()) < 1e-2                     # recovered the low-eigenvalue band


# ---------------------------------------------------------------------------
# ACT energy gate: LIVE activation -> ratio >> ACT_LIVE_RATIO; INERT -> ~0.
# Replicates main()'s energy read: e(U) = trace(U^T gram U)/trace(gram).
# ---------------------------------------------------------------------------
def _energy_ratio(gram, U, n, k, n_rand=5):
    tot = float(np.trace(gram))
    e_unt = float(np.trace(U.T @ gram @ U)) / tot
    rr = [float(np.trace(Qr.T @ gram @ Qr)) / tot
          for Qr in (J._random_subspace(n, k, s) for s in range(n_rand))]
    return e_unt / float(np.mean(rr))


def test_act_energy_ratio_live_vs_inert():
    rng = np.random.default_rng(1)
    n, k = 64, 8
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    U = Q[:, :k]                                     # treat as the band
    gram_live = U @ U.T                              # all activation energy inside span(U)
    gram_inert = Q[:, k:] @ Q[:, k:].T               # all energy OUTSIDE span(U)
    assert _energy_ratio(gram_live, U, n, k) > J.ACT_LIVE_RATIO     # proceeds
    assert _energy_ratio(gram_inert, U, n, k) < 0.5                  # stops (INERT)


# ---------------------------------------------------------------------------
# Projector logic (what _InputProject's hook computes): idempotent + removes band.
# ---------------------------------------------------------------------------
def test_projector_idempotent_and_removes_band():
    rng = np.random.default_rng(2)
    n, k = 64, 8
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Qk = torch.tensor(Q[:, :k], dtype=torch.float64)
    x = torch.randn(3, 5, n, dtype=torch.float64)
    x1 = x - (x @ Qk) @ Qk.t()
    x2 = x1 - (x1 @ Qk) @ Qk.t()
    assert (x2 - x1).abs().max().item() < 1e-9       # idempotent
    assert (x1 @ Qk).abs().max().item() < 1e-9       # span(Q) component removed


def test_input_project_hook_end_to_end():
    """Exercise the REAL _InputProject hook on a tiny stand-in layer (identity q_proj weight ->
    its output equals the projected input)."""
    import torch.nn as nn
    n, k = 32, 4
    Q, _ = np.linalg.qr(np.random.default_rng(3).standard_normal((n, n)))
    Qk = Q[:, :k]

    class Attn(nn.Module):
        def __init__(s):
            super().__init__()
            s.q_proj = nn.Linear(n, n, bias=False)
            s.k_proj = nn.Linear(n, n, bias=False)
            with torch.no_grad():
                s.q_proj.weight.copy_(torch.eye(n))
                s.k_proj.weight.copy_(torch.eye(n))

    class Layer(nn.Module):
        def __init__(s):
            super().__init__()
            s.self_attn = Attn()

    layer = Layer()
    x = torch.randn(2, 3, n)
    with J._InputProject([layer], {0: Qk}, "cpu", torch.float32):
        y = layer.self_attn.q_proj(x)                # identity weight -> y == projected input
    assert (y.double() @ torch.tensor(Qk)).abs().max().item() < 1e-5     # band removed from what q_proj read


# ---------------------------------------------------------------------------
# Positive control vs bottom-sigma null: top-sigma picks the MOST-used input dirs.
# ---------------------------------------------------------------------------
def test_sigma_ends_select_largest_smallest():
    rng = np.random.default_rng(4)
    n, m = 32, 40
    Uo, _ = np.linalg.qr(rng.standard_normal((m, m)))
    Vo, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.linspace(10.0, 0.1, n)                    # descending sigma spectrum
    W0 = torch.tensor(Uo[:, :n] @ np.diag(s) @ Vo.T)  # m x n with known sigma
    k = 5
    top = J._base_top_sigma(W0, k)
    bot = J._base_bottom_sigma(W0, k)
    Wn = W0.double().numpy()
    e_top = np.linalg.norm(Wn @ top)                 # ~ sqrt(sum of top-k sigma^2)
    e_bot = np.linalg.norm(Wn @ bot)                 # ~ sqrt(sum of bottom-k sigma^2)
    assert e_top > 10.0 * e_bot, (e_top, e_bot)
    assert np.abs(top.T @ top - np.eye(k)).max() < 1e-10     # orthonormal
    assert np.abs(bot.T @ bot - np.eye(k)).max() < 1e-10


# ---------------------------------------------------------------------------
# Paired-contrast bootstrap: brackets a planted truth and excludes 0 when real;
# straddles 0 when the arms are exchangeable.
# ---------------------------------------------------------------------------
def test_paired_contrast_ci_brackets_planted_truth():
    rng = np.random.default_rng(5)
    W = 200
    n_tok = rng.integers(50, 80, W).astype(np.float64)
    base_nll = 2.0 + 0.3 * rng.standard_normal(W)    # per-window avg nll/token
    delta = 0.10                                     # treat is +0.10 nll/token everywhere -> worse
    null_pw = np.stack([base_nll * n_tok, n_tok], axis=1)
    treat_pw = np.stack([(base_nll + delta) * n_tok, n_tok], axis=1)
    out = J._paired_contrast_ci(treat_pw, null_pw, n_boot=2000, seed=7)
    ppl_null = math.exp(null_pw[:, 0].sum() / null_pw[:, 1].sum())
    ppl_treat = math.exp(treat_pw[:, 0].sum() / treat_pw[:, 1].sum())
    true_c = ppl_treat - ppl_null
    assert abs(out["contrast"] - true_c) < 1e-9      # point estimate is exact on the full sample
    assert out["lo"] < true_c < out["hi"]            # CI brackets the truth
    assert out["lo"] > 0                              # treat clearly worse -> excludes 0
    assert out["p_le0"] < 0.05


def test_paired_contrast_ci_null_case_straddles_zero():
    rng = np.random.default_rng(6)
    W = 200
    n_tok = rng.integers(50, 80, W).astype(np.float64)
    nll = (2.0 + 0.3 * rng.standard_normal(W)) * n_tok
    pw = np.stack([nll, n_tok], axis=1)
    treat = pw.copy()
    treat[:, 0] += 0.01 * n_tok * rng.standard_normal(W)   # zero-mean per-token wobble
    out = J._paired_contrast_ci(treat, pw, n_boot=2000, seed=8)
    assert out["lo"] < 0 < out["hi"]                 # straddles zero
    assert 0.2 < out["p_le0"] < 0.8


def test_paired_contrast_ci_fails_loud_on_mismatched_tokens():
    a = np.array([[1.0, 10.0], [2.0, 20.0]])
    b = np.array([[1.0, 11.0], [2.0, 20.0]])         # n_tok differs -> not paired
    with pytest.raises(AssertionError):
        J._paired_contrast_ci(a, b)


# ---------------------------------------------------------------------------
# Random-band averaging, BH-FDR, and random subspace plumbing.
# ---------------------------------------------------------------------------
def test_avg_per_window_averages_nll_keeps_ntok():
    n_tok = np.array([10.0, 20.0, 30.0])
    a = np.stack([np.array([1.0, 2.0, 3.0]), n_tok], axis=1)
    b = np.stack([np.array([3.0, 4.0, 5.0]), n_tok], axis=1)
    out = J._avg_per_window([a, b])
    assert np.allclose(out[:, 0], [2.0, 3.0, 4.0])
    assert np.array_equal(out[:, 1], n_tok)
    c = np.stack([np.array([1.0, 2.0, 3.0]), n_tok + 1], axis=1)
    with pytest.raises(AssertionError):
        J._avg_per_window([a, c])                    # mismatched windows must FAIL LOUD


def test_bh_adjust_matches_hand_computation():
    adj = J._bh_adjust([0.001, 0.5, 0.02])           # sorted [0.001,0.02,0.5]*3/[1,2,3]=[0.003,0.03,0.5]
    assert abs(adj[0] - 0.003) < 1e-9
    assert abs(adj[1] - 0.5) < 1e-9
    assert abs(adj[2] - 0.03) < 1e-9
    assert all(0.0 <= x <= 1.0 for x in adj)


def test_locate_text_block_finds_block_past_trailing_newline():
    prefix = torch.arange(100, 110)                  # fake image+bos prefix
    text = torch.tensor([7, 8, 9, 10, 11])
    trailing = torch.tensor([13])                    # the processor's appended "\n"
    in_ids = torch.cat([prefix, text, trailing])
    pos = J._locate_text_block(in_ids, text)
    assert pos == 10
    assert torch.equal(in_ids[pos:pos + 5], text)
    with pytest.raises(AssertionError):              # absent -> FAIL LOUD
        J._locate_text_block(in_ids, torch.tensor([99, 98]))


def test_locate_text_block_rightmost_when_duplicated():
    text = torch.tensor([7, 8, 9])
    in_ids = torch.cat([text, torch.tensor([1, 2]), text, torch.tensor([13])])  # text at idx 0 and 5
    assert J._locate_text_block(in_ids, text) == 5   # rightmost (the real suffix, not prefix echo)


def test_random_subspace_orthonormal_deterministic():
    Q1 = J._random_subspace(64, 8, 123)
    Q2 = J._random_subspace(64, 8, 123)
    assert np.array_equal(Q1, Q2)                    # seed-deterministic
    assert np.abs(Q1.T @ Q1 - np.eye(8)).max() < 1e-10
