"""Tests for the single discovery↔confirmation firewall utility (prereg §9-1c).

Guards the recurring "held tensor captured but never used" leak (W9 v1 + GraftLM E4): the only
real failure mode is an EMPTY half, so both non-empty asserts must fire. Pure CPU / numpy.

Run: python -m pytest infra/pg/tests/test_enforce_split.py    (or python infra/pg/tests/test_enforce_split.py)
"""
import numpy as np
import pytest

from infra.pg.diff_metrics import enforce_split


def test_normal_split_indices_and_counts():
    held = np.array([False, True, False, True, True])      # 2 disc, 3 conf
    s = enforce_split(held, name="t")
    assert s["n"] == 5 and s["n_disc"] == 2 and s["n_conf"] == 3
    assert np.array_equal(s["disc_mask"], ~held)
    assert np.array_equal(s["disc_idx"], [0, 2])
    assert np.array_equal(s["conf_idx"], [1, 3, 4])


def test_stratum_is_anded_with_held():
    held = np.array([False, True, False, True, True])
    text = np.array([True, True, True, False, False])      # held & text → only index 1
    s = enforce_split(held, name="t", text=text)
    assert np.array_equal(s["conf_text_idx"], [1])


def test_empty_confirmation_half_fails_loud():
    with pytest.raises(AssertionError, match="confirmation"):
        enforce_split(np.zeros(8, dtype=bool), name="t")   # nothing held → firewall is a no-op


def test_empty_discovery_half_fails_loud():
    with pytest.raises(AssertionError, match="discovery"):
        enforce_split(np.ones(8, dtype=bool), name="t")    # everything held → nothing to fit on


def test_misshaped_stratum_fails_loud():
    held = np.array([False, True, True])
    with pytest.raises(AssertionError, match="size"):
        enforce_split(held, name="t", text=np.array([True, False]))


def test_accepts_torch_bool_tensor():
    torch = pytest.importorskip("torch")
    held = torch.tensor([False, True, False, True])
    s = enforce_split(held, name="t", text=torch.tensor([True, True, False, False]))
    assert s["n_disc"] == 2 and s["n_conf"] == 2
    assert np.array_equal(s["conf_text_idx"], [1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
