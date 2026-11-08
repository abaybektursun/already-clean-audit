"""Weight-free CPU dry-run gate for REPRICE (the prereg §7 'dry-run gate').
NO model, NO datasets, NO GPU — tiny planted tensors only, <1s on a laptop.
Every pure logic the GPU session depends on is proved here BEFORE any GPU minute.

    python -m pytest infra/pg/tests/test_reprice.py -q

Covers:
  * _score_cloze_window   — continuation-gather index math, mixed lengths
  * _q_block_cloze        — cloze prompt format (exemplar vs scored item)
  * degenerate_cloze_filter — empty / duplicate choice text drop
  * cloze_predict/correct — length-norm λ flips a long-wrong vs short-right item
  * recovery_score/cost   — reached / unreached τ·ceiling
  * concordance           — J = P(cloze|mcq), n_cond=0 guard
  * paired_cluster_bootstrap — determinism, paired draw, degenerate zero-width
  * bootstrap_cost_ratio  — lockstep → 1.0; cloze-plateau → undefined
  * bootstrap_concordance — with / without null subtraction
  * reinit_lm_layer_      — lobotomy hits exactly the target layer's 2-D weights
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from infra.pg import reprice_stats as R
from infra.pg import w16_probes as W
from infra.pg.text_eval import _q_block_cloze, _score_cloze_window


# --------------------------------------------------------------------------- #
# _score_cloze_window — the risky index math, unit-tested without a model.
# --------------------------------------------------------------------------- #
def test_score_cloze_window_mixed_lengths():
    V, B, K = 5, 2, 3                       # max continuation length = K-1 = 2
    lw = torch.full((B, K, V), -9.0)
    # row 0: m=2, cont=[1,3] → predicted by window positions [K-1-2:K-1]=[0,1]
    lw[0, 0, 1] = -0.5
    lw[0, 1, 3] = -0.25
    # row 1: m=1, cont=[4] → predicted by window position [K-1-1:K-1]=[1]
    lw[1, 1, 4] = -2.0
    out = _score_cloze_window(lw, [[1, 3], [4]], [2, 1])
    assert out.dtype == torch.float64
    assert out[0].item() == pytest.approx(-0.75)
    assert out[1].item() == pytest.approx(-2.0)


def test_score_cloze_window_rejects_overlong():
    lw = torch.zeros((1, 3, 4))
    with pytest.raises(AssertionError):
        _score_cloze_window(lw, [[0, 1, 2]], [3])     # m=3 > K-1=2


def test_score_cloze_window_trailing_separator():
    # PaliGemma path: continuation sits BEFORE a 1-token trailing '\n' separator.
    # K=4, trailing=1 → end=K-1-1=2 → predicting slice window[0:2] for m=2.
    V, K = 5, 4
    lw = torch.full((1, K, V), -9.0)
    lw[0, 0, 1] = -0.5
    lw[0, 1, 3] = -0.25
    out = _score_cloze_window(lw, [[1, 3]], [2], trailing=1)
    assert out[0].item() == pytest.approx(-0.75)


# --------------------------------------------------------------------------- #
# _q_block_cloze — content format, no lettered choices, no letter read.
# --------------------------------------------------------------------------- #
def test_q_block_cloze_format():
    shot = _q_block_cloze("What is 2+2?", "four")
    assert shot == "What is 2+2?\nAnswer: four\n\n"
    item = _q_block_cloze("What is 2+2?")
    assert item == "What is 2+2?\nAnswer:"
    assert "A." not in shot and "B." not in shot       # no MCQ scaffold


# --------------------------------------------------------------------------- #
# degenerate_cloze_filter — model-free, pre-registered.
# --------------------------------------------------------------------------- #
def test_degenerate_filter_empty_and_duplicate():
    items = [
        {"question": "q good", "choices": ["a", "b", "c", "d"], "answer": 0},
        {"question": "q empty", "choices": ["", "b", "c", "d"], "answer": 0},   # empty gold
        {"question": "q dup", "choices": ["x", "x", "c", "d"], "answer": 0},    # gold duplicated
        {"question": "q dup-elsewhere", "choices": ["x", "y", "y", "d"], "answer": 0},  # gold unique
    ]
    res = R.degenerate_cloze_filter(items)
    kept_q = [r["question"] for r in res["kept"]]
    assert kept_q == ["q good", "q dup-elsewhere"]
    assert res["n_dropped"] == 2 and res["n_in"] == 4
    assert len(res["dropped_hashes"]) == 2


# --------------------------------------------------------------------------- #
# cloze_predict / correct — length normalization actually matters.
# --------------------------------------------------------------------------- #
def test_length_norm_flips_prediction():
    logprobs = np.array([[-3.0, -2.0, -5.0, -5.0]])   # idx1 wins raw sum
    ntoks = np.array([[6, 1, 3, 3]])                  # idx0 wins per-token mean
    assert R.cloze_predict(logprobs, ntoks, lam=0.0)[0] == 1
    assert R.cloze_predict(logprobs, ntoks, lam=1.0)[0] == 0
    assert R.cloze_correct(logprobs, ntoks, [0], lam=1.0)[0] == 1
    assert R.cloze_correct(logprobs, ntoks, [0], lam=0.0)[0] == 0


def test_cloze_predict_rejects_bad_ntok():
    with pytest.raises(AssertionError):
        R.cloze_predict(np.zeros((1, 4)), np.array([[0, 1, 1, 1]]), lam=1.0)


# --------------------------------------------------------------------------- #
# recovery_score / recovery_cost.
# --------------------------------------------------------------------------- #
def test_recovery_cost_reached_and_unreached():
    budgets = [1000, 4000, 16000]
    hit = R.recovery_cost(budgets, [0.10, 0.30, 0.65], tau=0.9, ceiling=0.65)
    assert hit["reached"] and hit["cost"] == 16000
    miss = R.recovery_cost(budgets, [0.10, 0.30, 0.50], tau=0.9, ceiling=0.65)
    assert not miss["reached"] and miss["cost"] is None


def test_recovery_cost_requires_ascending():
    with pytest.raises(AssertionError):
        R.recovery_cost([16000, 1000], [0.5, 0.5], tau=0.5, ceiling=1.0)


def test_recovery_score_mask():
    per = [1, 0, 1, 1]
    assert R.recovery_score(per) == pytest.approx(0.75)
    assert R.recovery_score(per, mask=[True, True, False, False]) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# concordance J.
# --------------------------------------------------------------------------- #
def test_concordance_basic_and_empty():
    j = R.concordance([1, 1, 0, 1], [1, 0, 0, 1])
    assert j["n_cond"] == 3 and j["J"] == pytest.approx(2 / 3)
    empty = R.concordance([0, 0], [1, 1])
    assert empty["n_cond"] == 0 and empty["J"] is None


# --------------------------------------------------------------------------- #
# paired_cluster_bootstrap — determinism + pairing.
# --------------------------------------------------------------------------- #
def test_bootstrap_deterministic_and_paired():
    a = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    b = np.array([1, 0, 0, 1, 0, 1, 0, 0])

    def diff(arrs):
        return float(arrs[0].mean() - arrs[1].mean())

    r1 = R.paired_cluster_bootstrap([a, b], diff, n=500, seed=42)
    r2 = R.paired_cluster_bootstrap([a, b], diff, n=500, seed=42)
    assert r1["lo"] == r2["lo"] and r1["hi"] == r2["hi"]   # seed-deterministic
    assert r1["lo"] <= r1["point"] <= r1["hi"]


def test_bootstrap_zero_width_on_constant():
    a = np.ones(10)

    def mean(arrs):
        return float(arrs[0].mean())

    r = R.paired_cluster_bootstrap([a], mean, n=200, seed=1)
    assert r["lo"] == pytest.approx(1.0) and r["hi"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# bootstrap_cost_ratio — the headline C_cloze/C_MCQ.
# --------------------------------------------------------------------------- #
def _planted(frac_per_budget, n=200):
    """(n, B) per-item correctness whose column means ≈ frac_per_budget."""
    B = len(frac_per_budget)
    M = np.zeros((n, B), dtype=int)
    for j, f in enumerate(frac_per_budget):
        M[: int(round(f * n)), j] = 1
    return M


def test_cost_ratio_lockstep_is_one():
    budgets = [1000, 4000, 16000]
    mcq = _planted([0.10, 0.40, 0.70])
    cloze = mcq.copy()                                  # axes recover in lockstep
    r = R.bootstrap_cost_ratio(mcq, cloze, budgets, tau=0.9,
                               mcq_ceiling=0.70, cloze_ceiling=0.70,
                               n=300, seed=0)
    assert r["point"] == pytest.approx(1.0)
    assert r["frac_undef"] < 0.05                        # both axes reach τ most draws


def test_cost_ratio_cloze_plateau_is_undefined():
    budgets = [1000, 4000, 16000]
    mcq = _planted([0.10, 0.40, 0.70])
    cloze = _planted([0.05, 0.08, 0.10])                 # never reaches τ·ceiling
    r = R.bootstrap_cost_ratio(mcq, cloze, budgets, tau=0.9,
                               mcq_ceiling=0.70, cloze_ceiling=0.70,
                               n=300, seed=0)
    assert r["point"] is None                            # ratio undefined (H_head signal)
    assert r["frac_undef"] > 0.95


# --------------------------------------------------------------------------- #
# bootstrap_concordance — with / without null.
# --------------------------------------------------------------------------- #
def test_bootstrap_concordance_minus_null():
    mcq = np.array([1, 1, 1, 1, 0, 0])
    cloze = np.array([1, 1, 1, 0, 0, 0])                 # J = 3/4
    null_mcq = np.array([1, 1, 1, 1, 0, 0])
    null_cloze = np.array([1, 0, 0, 0, 0, 0])            # J_null = 1/4
    plain = R.bootstrap_concordance(mcq, cloze, n=300, seed=0)
    assert plain["point"] == pytest.approx(0.75)
    minus = R.bootstrap_concordance(mcq, cloze, null_mcq, null_cloze, n=300, seed=0)
    assert minus["point"] == pytest.approx(0.5)          # 0.75 − 0.25


# --------------------------------------------------------------------------- #
# reinit_lm_layer_ — lobotomy hits exactly the target layer's 2-D weights.
# --------------------------------------------------------------------------- #
class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


def test_lobotomy_scope_and_determinism():
    m = _TinyLM()
    w0_before = m.layers[0].weight.detach().clone()
    w1_before = m.layers[1].weight.detach().clone()
    rep = W.reinit_lm_layer_(m, layer_idx=1, seed=7)
    assert rep["n_tensors"] == 1 and rep["names"] == ["layers.1.weight"]
    assert torch.equal(m.layers[0].weight, w0_before)            # layer 0 untouched
    assert not torch.equal(m.layers[1].weight, w1_before)        # layer 1 destroyed
    assert m.layers[1].weight.shape == w1_before.shape           # shape preserved

    a = copy.deepcopy(_TinyLM())
    b = copy.deepcopy(a)
    W.reinit_lm_layer_(a, 1, seed=7)
    W.reinit_lm_layer_(b, 1, seed=7)
    assert torch.equal(a.layers[1].weight, b.layers[1].weight)   # seed-deterministic


def test_lobotomy_missing_layer_raises():
    with pytest.raises(AssertionError):
        W.reinit_lm_layer_(_TinyLM(), layer_idx=9, seed=0)


# --------------------------------------------------------------------------- #
# Unblind aggregator — the §8 grid + frozen §3 verdict, end-to-end on synthetic
# banked cells (no GPU). Proves the verdict path BEFORE any cell is run.
# --------------------------------------------------------------------------- #
from infra.aws_pg.jobs import reprice_unblind as U     # noqa: E402

_HASHES = [f"it{i:04d}" for i in range(200)]


def _seq(mcq_correct, cloze_correct, N=200):
    """Ordered cell sequence (position-aligned, hash is a tripwire only — mirrors the
    895-way-collision-safe real format)."""
    mc, cc = set(mcq_correct), set(cloze_correct)
    return {"hash": _HASHES,
            "mcq": [1 if i in mc else 0 for i in range(N)],
            "cloze": [1 if i in cc else 0 for i in range(N)]}


def test_unblind_H_know():
    ceiling = _seq(range(140), range(140))                            # 0.70 both
    cap = {                                                            # lockstep recovery
        1000:  _seq(range(20),  range(20)),
        4000:  _seq(range(80),  range(80)),
        16000: _seq(range(140), range(140)),                          # both recover to ceiling
    }
    null = _seq(range(50), range(100, 150))                           # floor 0.25 both; J_null=0
    res = U.aggregate(cap, null, ceiling, budgets=[1000, 4000, 16000],
                      tau=0.9, n_boot=2000, seed=0)
    rf = res["recovery_fraction"]
    assert rf["mcq"]["point"] == pytest.approx(1.0) and rf["cloze"]["point"] == pytest.approx(1.0)
    assert rf["cloze"]["lo"] > 0                                      # cloze ALSO recovers
    assert res["J"]["raw"] == pytest.approx(1.0) and res["J"]["lo"] > 0   # J ≫ null
    assert res["verdict"]["verdict"] == "H_know"


def test_unblind_H_head_cloze_plateau():
    ceiling = _seq(range(140), range(140))
    cap = {                                                            # MCQ recovers, cloze flat
        1000:  _seq(range(20),  range(10)),
        4000:  _seq(range(80),  range(16)),
        16000: _seq(range(140), range(20)),                           # cloze ~0.10, below floor
    }
    null = _seq(range(50), range(45))                                 # floor cloze 0.225; J_null=0.9
    res = U.aggregate(cap, null, ceiling, budgets=[1000, 4000, 16000],
                      tau=0.9, n_boot=2000, seed=0)
    rf = res["recovery_fraction"]
    assert rf["mcq"]["point"] > 0.5 and rf["mcq"]["lo"] > 0           # MCQ recovers
    assert rf["cloze"]["hi"] <= 0                                     # cloze does NOT recover
    assert res["verdict"]["cloze_no_recover"] is True
    assert res["verdict"]["verdict"] == "H_head"


def test_unblind_graded_J_above_null_but_cloze_flat():
    # cloze does NOT recover, but J IS above null (discovery-like) → neither clean → graded
    rf_mcq = {"point": 0.75, "lo": 0.6, "hi": 0.9}
    rf_cloze = {"point": -0.2, "lo": -0.4, "hi": -0.05}              # cloze_no_recover
    J = {"lo": 0.013, "hi": 0.116}                                   # J above null
    assert U._verdict(rf_mcq, rf_cloze, J)["verdict"] == "graded"


def test_unblind_halts_on_sequence_drift():
    """Position alignment: a reordered item sequence MUST HALT (not silently align)."""
    ceiling = _seq(range(140), range(140))
    cap = {1000: _seq(range(20), range(20)), 4000: _seq(range(80), range(80)),
           16000: _seq(range(140), range(140))}
    null = _seq(range(50), range(50))
    null["hash"] = ["X" + h for h in null["hash"]]                    # corrupt the sequence
    with pytest.raises(AssertionError):
        U.aggregate(cap, null, ceiling, budgets=[1000, 4000, 16000], tau=0.9, n_boot=10, seed=0)
