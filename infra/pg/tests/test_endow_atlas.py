"""Weight-free CPU dry-run gate for ENDOW-F's atlas axes (endow spec §5, §9.1).

NO model, NO GPU — tiny planted curves/bases. Proves: E(k) and the saturation-guarded k_90
(compact vs distributed labels); normalized canonicity (1 for identical draws, ≈0 for random);
participation ratio; and the threshold-free two-kinds-law predicate (holds / named counterexample).

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_endow_atlas.py -q
"""
from __future__ import annotations

import pytest
import torch

from infra.pg import endow_atlas as EA


def _basis(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64)).Q[:, :k].contiguous()


def test_effect_curve():
    E = EA.effect_curve({8: 1.0, 32: 0.5, 128: 0.1})
    assert E == {8: 0.0, 32: 0.5, 128: pytest.approx(0.9)}


def test_k90_compact_label():
    r = EA.k90({4: 1.0, 8: 0.15, 16: 0.1, 32: 0.1}, null_width=0.05)   # E: 0, .85, .9, .9
    assert r["saturated"] is False and r["k90"] == 8


def test_k90_distributed_when_rising_at_grid_max():
    r = EA.k90({4: 0.9, 8: 0.7, 16: 0.4, 32: 0.1}, null_width=0.05)    # E rises to .9 at k=32
    assert r["saturated"] is True and r["k90"] is None


def test_canonicity_identical_vs_random():
    d, k = 64, 8
    U = _basis(d, k, seed=1)
    assert EA.canonicity([U, U.clone(), U.clone()], d) == pytest.approx(1.0, abs=1e-6)
    rnd = [_basis(d, k, seed=s) for s in (2, 3, 4)]
    assert EA.canonicity(rnd, d) < 0.5                                  # unrelated draws ≈ chance


def test_normalized_canonicity_endpoints():
    assert EA.normalized_canonicity(8 / 64, 8, 64) == pytest.approx(0.0, abs=1e-12)
    assert EA.normalized_canonicity(1.0, 8, 64) == pytest.approx(1.0, abs=1e-12)


def test_participation_ratio():
    assert EA.participation_ratio({0: 1.0, 1: 0.0, 2: 0.0}) == pytest.approx(1.0)
    assert EA.participation_ratio({0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}) == pytest.approx(4.0)


def test_two_kinds_law_holds():
    prims = {"U_unt": {"k90": 8, "c": 0.9}, "U_unt-vo": {"k90": 16, "c": 0.8}}
    reads = {"S_action": {"k90": None, "c": 0.3}, "S_vl": {"k90": 128, "c": 0.2}}
    v = EA.two_kinds_law(prims, reads)
    assert v["holds"] is True and v["label"] == "H_two_kinds"


def test_two_kinds_law_counterexample():
    prims = {"U_unt": {"k90": 64, "c": 0.5}}
    reads = {"S_vl": {"k90": 32, "c": 0.6}}          # read is MORE compact AND more canonical
    v = EA.two_kinds_law(prims, reads)
    assert v["holds"] is False
    axes = {c["axis"] for c in v["counterexamples"]}
    assert axes == {"compactness", "canonicity"}


def test_distributed_read_is_consistent_but_distributed_primitive_violates():
    assert EA.two_kinds_law({"P": {"k90": 8, "c": 0.9}},
                            {"R": {"k90": None, "c": 0.1}})["holds"] is True
    assert EA.two_kinds_law({"P": {"k90": None, "c": 0.9}},
                            {"R": {"k90": 64, "c": 0.1}})["holds"] is False
