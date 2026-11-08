"""RE-EXPRESS — weight-free dry-run (prereg §9.2: unit-test pure logic on tiny
tensors BEFORE any GPU). Laptop-runnable; no model, no datasets, no network.

Covers exactly the §9.2 list that this freeze owns:
  * NEW keystone graders — _match_to_option (no-match/tie CONTRACT), _normalize_answer,
    _free_gen_em, mmlu_freegen.
  * PORT cloze pure logic — _q_block_cloze, _score_cloze_window (mixed lengths /
    rejects-overlong / trailing separator).
  * PORT reprice_stats — degenerate_cloze_filter, recovery_cost, recovery_fraction,
    concordance, paired_cluster_bootstrap (deterministic + paired),
    bootstrap_cost_ratio (lockstep=1, cloze-plateau undefined).
  * PORT reinit_lm_layer_ — scope / determinism / missing-layer / in-place.
  * Positional cross-axis alignment vs qa_item_hash + ladder_subset nesting + the
    discovery/confirmation/grid-subset partition (the §2 alignment-by-POSITION,
    hash-as-tripwire invariant).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
import torch.nn as nn

from infra.pg import reprice_stats as R
from infra.pg import w16_probes as W
from infra.pg.relearn import qa_item_hash, ladder_subset
from infra.pg.text_eval import (
    _free_gen_em,
    _match_to_option,
    _normalize_answer,
    _q_block_cloze,
    _score_cloze_window,
    mmlu_freegen,
)

OPTS = ["the mitochondria", "the nucleus", "the ribosome", "cell wall"]


# ---------------------------------------------------------------------------
# NEW keystone graders.
# ---------------------------------------------------------------------------
def test_normalize_answer_squad_style():
    assert _normalize_answer("The  Mitochondria!!") == "mitochondria"
    assert _normalize_answer("A cell, the wall.") == "cell wall"
    assert _normalize_answer("   ") == ""
    assert _normalize_answer("THE") == ""          # a bare article normalizes to empty


def test_match_to_option_exact_and_contained():
    # exact gold content (article-stripped option ⊆ generation)
    assert _match_to_option("mitochondria", OPTS) == (0, "ok")
    # generation wraps the gold content
    assert _match_to_option("The answer is the nucleus.", OPTS) == (1, "ok")
    # letter prefix + content → content wins
    assert _match_to_option("B. cell wall", OPTS) == (3, "ok")


def test_match_to_option_unmatched_policy():
    # bare letter (the §2 example) → unmatched, scored incorrect
    assert _match_to_option("A", OPTS) == (None, "unmatched")
    assert _match_to_option("B", OPTS) == (None, "unmatched")
    # no-overlap paraphrase → unmatched
    assert _match_to_option("photosynthesis", OPTS) == (None, "unmatched")
    # empty generation → unmatched
    assert _match_to_option("", OPTS) == (None, "unmatched")


def test_match_to_option_tie_and_longest_wins():
    # nested candidates: 'cell' ⊂ 'cell wall' → the LONGER (more specific) wins, not a tie
    nested = ["cell", "cell wall", "nucleus", "ribosome"]
    assert _match_to_option("the cell wall is rigid", nested) == (1, "ok")
    # genuine tie: two distinct equal-length candidates both contained → incorrect
    tie = ["red", "blue", "green", "amber"]
    assert _match_to_option("either red or blue", tie) == (None, "tie")


def test_free_gen_em_and_freegen():
    assert _free_gen_em("the  Nucleus.", "nucleus") == 1
    assert _free_gen_em("nucleus", "ribosome") == 0
    out = mmlu_freegen(["nucleus", "A", "cell wall"], ["the nucleus", "nucleus", "cell wall"])
    assert out["per_item"] == [1, 0, 1]
    assert out["n"] == 3 and abs(out["acc"] - 2 / 3) < 1e-12


def test_freegen_misalignment_raises():
    with pytest.raises(AssertionError):
        mmlu_freegen(["a", "b"], ["a"])


def test_match_keystone_is_directional_not_recognition():
    """The whole point of (P): a model that emits only the MCQ letter has NOT produced
    content → must read as incorrect, even though the letter readout 'knew' the answer."""
    idx, flag = _match_to_option("C", OPTS)        # 'C' would be a correct MCQ letter
    assert flag == "unmatched" and int(idx == 2) == 0


# ---------------------------------------------------------------------------
# PORT — cloze pure logic.
# ---------------------------------------------------------------------------
def test_q_block_cloze_format():
    # scored item ends exactly at 'Answer:' (no answer text)
    assert _q_block_cloze("What is 2+2?") == "What is 2+2?\nAnswer:"
    # few-shot exemplar appends ' <answer>\n\n'
    assert _q_block_cloze("Q?", "four") == "Q?\nAnswer: four\n\n"


def test_score_cloze_window_mixed_lengths():
    B, K, V = 2, 4, 5
    w = torch.zeros(B, K, V, dtype=torch.float64)
    # trailing=0 → end = K-1 = 3. Row0 ntok=2 → predict positions {1,2}; Row1 ntok=1 → {2}.
    w[0, 1, 3] = 1.0
    w[0, 2, 4] = 2.0
    w[1, 2, 1] = 5.0
    out = _score_cloze_window(w, [[3, 4], [1]], [2, 1], trailing=0)
    assert out.tolist() == [3.0, 5.0]


def test_score_cloze_window_trailing_separator():
    B, K, V = 1, 5, 4
    w = torch.zeros(B, K, V, dtype=torch.float64)
    # trailing=1 → end = K-1-1 = 3. ntok=2 → predict positions {1,2}.
    w[0, 1, 2] = 0.5
    w[0, 2, 0] = 0.25
    out = _score_cloze_window(w, [[2, 0]], [2], trailing=1)
    assert out.tolist() == [0.75]


def test_score_cloze_window_rejects_overlong():
    w = torch.zeros(1, 3, 4, dtype=torch.float64)   # end = 2; ntok 3 → end-m = -1 < 0
    with pytest.raises(AssertionError):
        _score_cloze_window(w, [[0, 1, 2]], [3], trailing=0)


def test_degenerate_cloze_filter_empty_and_duplicate():
    items = [
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 0},   # keep
        {"question": "q2", "choices": ["", "b", "c", "d"], "answer": 0},    # empty gold → drop
        {"question": "q3", "choices": ["x", "x", "c", "d"], "answer": 1},   # gold dup → drop
    ]
    out = R.degenerate_cloze_filter(items)
    assert out["n_in"] == 3 and out["n_dropped"] == 2
    assert [r["question"] for r in out["kept"]] == ["q1"]
    assert out["dropped_hashes"] == [qa_item_hash("q2"), qa_item_hash("q3")]


# ---------------------------------------------------------------------------
# PORT — reprice_stats estimators.
# ---------------------------------------------------------------------------
def test_recovery_cost_reached_unreached_and_ascending():
    budgets = [1000, 2000, 4000, 8000]
    # ceiling 0.5, tau 0.8 → threshold 0.40; first rung ≥0.40 is budget 4000
    hit = R.recovery_cost(budgets, [0.20, 0.35, 0.42, 0.49], tau=0.8, ceiling=0.5)
    assert hit == {"cost": 4000.0, "reached": True, "threshold": pytest.approx(0.40)}
    miss = R.recovery_cost(budgets, [0.20, 0.25, 0.30, 0.31], tau=0.8, ceiling=0.5)
    assert miss["reached"] is False and miss["cost"] is None
    with pytest.raises(AssertionError):
        R.recovery_cost([4000, 2000], [0.5, 0.5], tau=0.8, ceiling=0.5)   # not ascending


def test_recovery_fraction_planted():
    # floor mean 0.2, ceiling mean 0.6 → gap 0.4; S mean 0.4 → fraction (0.4-0.2)/0.4 = 0.5
    S = np.array([0.4] * 10)
    floor = np.array([0.2] * 10)
    ceil = np.array([0.6] * 10)
    assert R.recovery_fraction(S, floor, ceil) == pytest.approx(0.5)


def test_concordance_basic_and_empty():
    mcq = np.array([1, 1, 1, 0])
    cloze = np.array([1, 0, 1, 1])
    out = R.concordance(mcq, cloze)
    assert out["n_cond"] == 3 and out["J"] == pytest.approx(2 / 3)
    empty = R.concordance(np.zeros(4, int), np.ones(4, int))
    assert empty["J"] is None and empty["n_cond"] == 0


def test_paired_bootstrap_deterministic_and_paired():
    a = np.arange(20.0)
    b = a * 2.0                       # perfectly paired
    s1 = R.paired_cluster_bootstrap([a, b], lambda arr: float((arr[1] - 2 * arr[0]).mean()),
                                    n=200, seed=1337)
    s2 = R.paired_cluster_bootstrap([a, b], lambda arr: float((arr[1] - 2 * arr[0]).mean()),
                                    n=200, seed=1337)
    assert s1 == s2                                   # same seed → identical
    # pairing holds under every resample → b-2a is identically 0
    assert s1["point"] == pytest.approx(0.0) and s1["lo"] == pytest.approx(0.0)
    assert s1["hi"] == pytest.approx(0.0)


def test_cost_ratio_lockstep_one_and_plateau_undefined():
    budgets = [1000, 2000, 4000]
    # both axes identical, monotone past threshold → ratio 1
    rising = np.tile(np.array([0.1, 0.5, 0.9]), (30, 1))
    lock = R.bootstrap_cost_ratio(rising, rising.copy(), budgets, tau=0.8,
                                  mcq_ceiling=1.0, cloze_ceiling=1.0, n=200, seed=0)
    assert lock["point"] == pytest.approx(1.0) and lock["frac_undef"] == 0.0
    # cloze permanently at chance → never reaches τ → ratio undefined (the H_head signal)
    flat = np.tile(np.array([0.25, 0.25, 0.25]), (30, 1))
    plateau = R.bootstrap_cost_ratio(rising, flat, budgets, tau=0.8,
                                     mcq_ceiling=1.0, cloze_ceiling=1.0, n=200, seed=0)
    assert plateau["frac_undef"] == 1.0 and plateau["point"] is None


# ---------------------------------------------------------------------------
# PORT — reinit_lm_layer_ (partial-damage lobotomy control).
# ---------------------------------------------------------------------------
class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({"a": nn.Linear(4, 4, bias=False), "n": nn.LayerNorm(4)}),
            nn.ModuleDict({"a": nn.Linear(4, 4, bias=False)}),
        ])
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)


def test_lobotomy_scope_determinism_and_inplace():
    m1, m2 = _TinyLM(), _TinyLM()
    # same init so we can check determinism of the reinit
    m2.load_state_dict(m1.state_dict())
    before_layer1 = m1.layers[1].a.weight.detach().clone()
    before_layer0 = m1.layers[0].a.weight.detach().clone()

    out1 = W.reinit_lm_layer_(m1, layer_idx=0, seed=1337)
    out2 = W.reinit_lm_layer_(m2, layer_idx=0, seed=1337)

    # scope: only the 2-D weight under layers.0, never the LayerNorm/embed/lm_head
    assert out1["names"] == ["layers.0.a.weight"] and out1["n_tensors"] == 1
    # in-place mutation of the targeted layer
    assert not torch.equal(m1.layers[0].a.weight, before_layer0)
    # untouched: layer 1 unchanged
    assert torch.equal(m1.layers[1].a.weight, before_layer1)
    # determinism: same seed → identical reinit
    assert torch.equal(m1.layers[0].a.weight, m2.layers[0].a.weight)


def test_lobotomy_missing_layer_raises():
    with pytest.raises(AssertionError):
        W.reinit_lm_layer_(_TinyLM(), layer_idx=9, seed=0)


# ---------------------------------------------------------------------------
# Positional cross-axis alignment + split partition + ladder nesting.
# ---------------------------------------------------------------------------
def test_qa_item_hash_is_the_one_recipe():
    """Every axis banks per_item_hash = sha1(question[:64])[:12]; relearn.qa_item_hash
    is that recipe. A single recipe is what makes by-POSITION alignment auditable."""
    for q in ["short", "x" * 200, "Which organelle generates ATP in eukaryotic cells?"]:
        assert qa_item_hash(q) == hashlib.sha1(q[:64].encode()).hexdigest()[:12]


def test_reorder_is_detectable_by_hash_sequence():
    qs = ["q-alpha", "q-beta", "q-gamma", "q-delta"]
    seq = [qa_item_hash(q) for q in qs]
    reordered = [qa_item_hash(q) for q in [qs[2], qs[0], qs[1], qs[3]]]
    assert seq != reordered                       # the §2 reorder tripwire fires


def test_split_partition_disjoint_and_grid_subset_of_discovery():
    hs = [hashlib.sha1(f"item-{i}".encode()).hexdigest()[:12] for i in range(500)]
    for h in hs:
        # discovery / confirmation are a partition (exactly one true)
        assert W.is_discovery(h) != W.is_confirmation(h)
        # grid subset never escapes the discovery half
        if W.is_grid_subset(h):
            assert W.is_discovery(h)
    assert any(W.is_grid_subset(h) for h in hs)    # non-empty


def test_ladder_subset_is_nested():
    rows = [{"question": f"q-{i}"} for i in range(50)]
    small = ladder_subset(rows, 5)
    big = ladder_subset(rows, 12)
    small_q = {r["question"] for r in small}
    big_q = {r["question"] for r in big}
    assert len(small) == 5 and len(big) == 12
    assert small_q <= big_q                        # subset(5) ⊂ subset(12)
