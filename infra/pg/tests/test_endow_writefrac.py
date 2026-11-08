"""Weight-free CPU dry-run gate for ENDOW-J's write-fraction battery (endow spec §6, §9.1).

NO model, NO GPU — tiny planted ΔW / span tensors. Proves: write_fraction is 1 when the change
lives entirely in the span and 0 when orthogonal to it; the matched-random null; the cluster-by-layer
bootstrap; and the frozen J3 specialness rule (strict non-overlap → special; overlap → generic).

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_endow_writefrac.py -q
"""
from __future__ import annotations

import pytest
import torch

from infra.pg import endow_writefrac as WF


def _std_basis(d, cols):
    U = torch.zeros(d, len(cols), dtype=torch.float64)
    for j, c in enumerate(cols):
        U[c, j] = 1.0
    return U


def test_write_fraction_bounds():
    d, out = 8, 5
    U = _std_basis(d, [0, 1, 2])                       # span = first 3 input axes
    dW_in = torch.zeros(out, d, dtype=torch.float64)
    dW_in[:, :3] = torch.randn(out, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    assert WF.write_fraction(dW_in, U) == pytest.approx(1.0, abs=1e-9)   # all energy in span
    dW_out = torch.zeros(out, d, dtype=torch.float64)
    dW_out[:, 3:] = torch.randn(out, d - 3, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    assert WF.write_fraction(dW_out, U) == pytest.approx(0.0, abs=1e-9)  # orthogonal to span


def test_write_fraction_rejects_zero_and_mismatch():
    U = _std_basis(6, [0, 1])
    with pytest.raises(AssertionError):
        WF.write_fraction(torch.zeros(4, 6, dtype=torch.float64), U)
    with pytest.raises(AssertionError):
        WF.write_fraction(torch.randn(4, 5, dtype=torch.float64), U)     # in-dim 5 != U rows 6


def test_matched_random_null_in_range():
    dW = torch.randn(6, 32, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    fr = WF.matched_random_fractions(dW, k=4, n_draws=5, seed=9)
    assert len(fr) == 5 and all(0.0 < f < 1.0 for f in fr)


def test_layer_bootstrap():
    per = {L: 0.2 for L in range(18)}
    b = WF.layer_bootstrap(per, n_boot=500, seed=1)
    assert b["mean"] == pytest.approx(0.2) and b["n_layers"] == 18
    assert b["ci_lo"] == pytest.approx(0.2, abs=1e-6) and b["ci_hi"] == pytest.approx(0.2, abs=1e-6)
    # a spread of per-layer fractions gives a CI that brackets the mean
    b2 = WF.layer_bootstrap({L: 0.1 * (L % 4) for L in range(18)}, n_boot=1000, seed=2)
    assert b2["ci_lo"] <= b2["mean"] <= b2["ci_hi"]


def test_specialness_strict_vs_overlap():
    unt = {"mean": 0.10, "ci_lo": 0.05, "ci_hi": 0.15, "n_layers": 18}
    low = {"S_vl": {"mean": 0.5, "ci_lo": 0.4, "ci_hi": 0.6, "n_layers": 18},
           "rand": {"mean": 0.4, "ci_lo": 0.3, "ci_hi": 0.5, "n_layers": 18}}
    v = WF.specialness(unt, low)
    assert v["special"] is True and v["verdict"] == "conservation_special"
    overlap = {"S_vl": {"mean": 0.18, "ci_lo": 0.12, "ci_hi": 0.25, "n_layers": 18}}
    v2 = WF.specialness(unt, overlap)                       # CIs overlap (0.15 vs 0.12)
    assert v2["special"] is False and v2["verdict"] == "low_write_generic"
