"""Weight-free CPU dry-run gate for the GAUGE shared spine (gauge_common).

NO model, NO GPU — tiny planted vectors and key lists only. Proves the pure roster/split/parse/
concentration primitives every GAUGE probe depends on. Run on a remote `pg` CPU box (never
locally; numpy is a large lib):
    python -m pytest infra/pg/tests/test_gauge_common.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import gauge_common as GC


# ---------------------------------------------------------------------------
# key parsing + generation firewall.
# ---------------------------------------------------------------------------
def test_matrix_type_and_layer_index():
    k = "model.layers.7.self_attn.q_proj.weight"
    assert GC.matrix_type(k) == "q_proj"
    assert GC.layer_index(k) == 7
    assert GC.matrix_type("model.layers.0.mlp.down_proj.weight") == "down_proj"


def test_matrix_type_unknown_raises():
    with pytest.raises(ValueError):
        GC.matrix_type("model.layers.0.self_attn.rotary.weight")


def test_layer_index_missing_raises():
    with pytest.raises(AssertionError):
        GC.layer_index("model.embed_tokens.weight")


def test_assert_gemma1_accepts_18_rejects_26():
    g1 = [f"model.layers.{i}.self_attn.q_proj.weight" for i in range(18)]
    assert GC.assert_gemma1(g1) == 18
    g2 = [f"model.layers.{i}.mlp.gate_proj.weight" for i in range(26)]
    with pytest.raises(AssertionError):
        GC.assert_gemma1(g2)


# ---------------------------------------------------------------------------
# deterministic 12/12 split.
# ---------------------------------------------------------------------------
def _roster(n=24):
    return [f"google/paligemma-3b-ft-task{i:02d}-224" for i in range(n)]


def test_sha1_split_is_12_12_disjoint_covering():
    disc, conf = GC.sha1_split(_roster())
    assert len(disc) == 12 and len(conf) == 12
    assert set(disc).isdisjoint(conf)
    assert set(disc) | set(conf) == set(_roster())
    # sorted for stable iteration.
    assert disc == sorted(disc) and conf == sorted(conf)


def test_sha1_split_is_deterministic_and_id_only():
    # same roster in a DIFFERENT input order → identical split (depends only on the ids).
    r = _roster()
    a = GC.sha1_split(r)
    b = GC.sha1_split(list(reversed(r)))
    assert a == b


def test_sha1_split_duplicate_raises():
    r = _roster()
    r[0] = r[1]
    with pytest.raises(AssertionError):
        GC.sha1_split(r)


def test_sha1_split_too_small_raises():
    with pytest.raises(AssertionError):
        GC.sha1_split(_roster(5), n_discovery=12)


# ---------------------------------------------------------------------------
# concentration primitives (Gini, participation ratio).
# ---------------------------------------------------------------------------
def test_gini_uniform_is_zero():
    assert GC.gini(np.ones(50)) == pytest.approx(0.0, abs=1e-12)


def test_gini_one_hot_approaches_one():
    x = np.zeros(1000)
    x[0] = 1.0
    assert GC.gini(x) == pytest.approx(1.0 - 1.0 / 1000, abs=1e-9)   # (n-1)/n for a spike


def test_gini_monotone_in_concentration():
    n = 200
    spread = np.abs(np.random.default_rng(0).normal(1.0, 0.01, n))   # near-uniform
    peaked = np.abs(np.random.default_rng(1).standard_cauchy(n))     # heavy-tailed
    assert GC.gini(peaked) > GC.gini(spread)


def test_gini_zero_vector_is_zero():
    assert GC.gini(np.zeros(10)) == 0.0


def test_gini_negative_raises():
    with pytest.raises(AssertionError):
        GC.gini(np.array([1.0, -1.0, 2.0]))


def test_participation_ratio_bounds():
    n = 64
    assert GC.participation_ratio(np.ones(n)) == pytest.approx(float(n), rel=1e-12)   # uniform → n
    onehot = np.zeros(n)
    onehot[3] = 5.0
    assert GC.participation_ratio(onehot) == pytest.approx(1.0, rel=1e-12)            # spike → 1


def test_participation_ratio_and_gini_disagree_on_direction():
    # PR high ⇔ spread; Gini high ⇔ concentrated. Sanity: concentration ↑ ⇒ PR ↓ and Gini ↑.
    n = 128
    spread = np.abs(np.random.default_rng(2).normal(1.0, 0.01, n))
    peaked = np.abs(np.random.default_rng(3).standard_cauchy(n))
    assert GC.participation_ratio(peaked) < GC.participation_ratio(spread)
    assert GC.gini(peaked) > GC.gini(spread)
