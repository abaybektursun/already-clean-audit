"""Weight-free CPU dry-run gate for TRAINSCAFFOLD (prereg §9.1).

NO model, NO datasets, NO GPU — tiny planted tensors only, <2s on a CPU box. Proves the PURE logic
BEFORE any GPU minute: the removed-diagonal-dyad gradient fraction; the JOINT-bootstrap RF_novel
anchor + its degenerate guard; the PINNED §6 verdict predicates (generalizes / at_floor_tost /
cell_state / go_nogo / reachability_verdict); and the hierarchical-gatekeeping 4-way verdict on
synthetic banked cells (each H-* outcome incl. INSTRUMENT-NULL / ABORT / MATCHED-NULL).

Run (remote `pg` CPU — never locally; torch is a large lib):
    python -m pytest infra/pg/tests/test_trainscaffold.py -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from infra.pg import trainscaffold as TS
from infra.pg import w15_carrier as C


def _rand_W(m=8, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
# grad_in_removed_dyad_fraction — diagonal-dyad energy accounting.
# ---------------------------------------------------------------------------
def test_grad_fraction_one_when_g_is_in_removed_dyads():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    idx = [0, 2]
    # g = u_0 v_0ᵀ + u_2 v_2ᵀ lives entirely in the removed diagonal dyads ⇒ fraction = 1.
    g = torch.outer(U[:, 0], V[:, 0]) + torch.outer(U[:, 2], V[:, 2])
    assert TS.grad_in_removed_dyad_fraction(g, U, V, idx) == pytest.approx(1.0, abs=1e-9)


def test_grad_fraction_zero_when_g_outside_removed_dyads():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    g = 3.0 * torch.outer(U[:, 1], V[:, 1])      # dyad 1, NOT in idx=[0,2]
    assert TS.grad_in_removed_dyad_fraction(g, U, V, [0, 2]) == pytest.approx(0.0, abs=1e-9)


def test_grad_fraction_empty_idx_is_zero_and_bounded():
    W = _rand_W()
    U, S, V = C.svd_fp64(W)
    gg = torch.Generator().manual_seed(3)
    g = torch.randn(8, 6, generator=gg, dtype=torch.float64)
    assert TS.grad_in_removed_dyad_fraction(g, U, V, []) == 0.0
    f = TS.grad_in_removed_dyad_fraction(g, U, V, [0, 1, 2])
    assert 0.0 <= f <= 1.0                          # a strict subset of ‖g‖² (diagonal only)


# ---------------------------------------------------------------------------
# rf_novel_joint_boot — anchored RF + joint resample + degenerate guard.
# ---------------------------------------------------------------------------
def _planted(mean, n=400):
    """0/1 array with exactly mean fraction of 1s."""
    a = np.zeros(n)
    a[: int(round(mean * n))] = 1.0
    return a


def test_rf_novel_point_equals_anchored_fraction():
    cell, R, B = _planted(0.55), _planted(0.25), _planted(0.85)
    rf = TS.rf_novel_joint_boot(cell, R, B, n_boot=500, seed=1)
    # (0.55−0.25)/(0.85−0.25) = 0.5
    assert rf["point"] == pytest.approx(0.5, abs=1e-9)
    assert rf["ci95"][0] <= rf["point"] <= rf["ci95"][1]


def test_rf_novel_floor_is_zero_ceiling_is_one():
    R, B = _planted(0.25), _planted(0.85)
    assert TS.rf_novel_joint_boot(_planted(0.25), R, B, n_boot=300, seed=2)["point"] == pytest.approx(0.0, abs=1e-9)
    assert TS.rf_novel_joint_boot(_planted(0.85), R, B, n_boot=300, seed=2)["point"] == pytest.approx(1.0, abs=1e-9)


def test_rf_novel_degenerate_no_dynamic_range_raises():
    # B == R everywhere ⇒ no dynamic range ⇒ the bootstrap drops every draw and fails loud.
    same = _planted(0.3)
    with pytest.raises(AssertionError):
        TS.rf_novel_joint_boot(_planted(0.5), same, same, n_boot=200, seed=3)


# ---------------------------------------------------------------------------
# §6 PINNED verdict predicates.
# ---------------------------------------------------------------------------
def _rf(point, lo95, hi95, lo90=None, hi90=None):
    return {"point": point, "ci95": [lo95, hi95],
            "ci90": [lo90 if lo90 is not None else lo95, hi90 if hi90 is not None else hi95]}


def test_generalizes_requires_ci_above_half_and_beats_shuffle():
    assert TS.generalizes(_rf(0.8, 0.6, 0.95), shuffle_rf_point=0.1) is True
    assert TS.generalizes(_rf(0.8, 0.45, 0.95), shuffle_rf_point=0.1) is False   # CI-lo ≤ 0.5
    assert TS.generalizes(_rf(0.8, 0.6, 0.95), shuffle_rf_point=0.7) is False     # shuffle too close


def test_at_floor_tost_band():
    assert TS.at_floor_tost(_rf(0.02, -0.3, 0.3, lo90=-0.15, hi90=0.15)) is True    # 90% CI ⊂ band
    assert TS.at_floor_tost(_rf(0.02, -0.3, 0.3, lo90=-0.25, hi90=0.15)) is False   # 90% CI exceeds band


def test_cell_state_three_way():
    assert TS.cell_state(_rf(0.85, 0.7, 0.95), 0.1) == "GENERALIZES"
    assert TS.cell_state(_rf(0.0, -0.15, 0.15, lo90=-0.12, hi90=0.12), 0.0) == "AT_FLOOR"
    assert TS.cell_state(_rf(0.4, 0.2, 0.6, lo90=0.25, hi90=0.55), 0.1) == "GRADED"   # neither


def test_go_nogo_pinned():
    assert TS.go_nogo(_planted(0.7), _planted(0.27))["go"] is True
    assert TS.go_nogo(_planted(0.5), _planted(0.27))["go"] is False    # acc_B < 0.55
    assert TS.go_nogo(_planted(0.7), _planted(0.45))["go"] is False    # acc_R above floor band


def test_reachability_verdict():
    assert TS.reachability_verdict([0.0, 0.10], 0.7, [-0.05, 0.05]) == "H-severance"
    assert TS.reachability_verdict([0.0, 0.10], 0.7, [0.3, 0.6]) == "H-capacity"
    assert TS.reachability_verdict([0.0, 0.5], 0.3, [-0.05, 0.1]) == "REACH-graded"


# ---------------------------------------------------------------------------
# trainscaffold_verdict — the §6 hierarchical-gatekeeping 4-way.
# ---------------------------------------------------------------------------
_BASE = {"R_free": "AT_FLOOR", "R_letter": "AT_FLOOR",
         "B_free": "GENERALIZES", "B_letter": "GENERALIZES"}


def _cells(af, al):
    return {**_BASE, "A_free": af, "A_letter": al}


def test_verdict_instrument_null_when_R_learns():
    cells = {**_cells("GENERALIZES", "GENERALIZES"), "R_free": "GENERALIZES"}
    v = TS.trainscaffold_verdict(cells, True, False, "H-severance", 3)
    assert v["named"] == "INSTRUMENT-NULL"


def test_verdict_abort_when_B_fails():
    cells = {**_cells("GENERALIZES", "GENERALIZES"), "B_free": "AT_FLOOR"}
    assert TS.trainscaffold_verdict(cells, True, False, "H-severance", 3)["named"] == "ABORT"


def test_verdict_h_capability_gated_on_severance():
    assert TS.trainscaffold_verdict(_cells("GENERALIZES", "GENERALIZES"), True, False, "H-severance", 3)["named"] == "H-capability"
    # same cells, but REACHABILITY = capacity ⇒ NOT the diamond
    assert TS.trainscaffold_verdict(_cells("GENERALIZES", "GENERALIZES"), True, False, "H-capacity", 3)["named"] == "capacity/difficulty-gradient"


def test_verdict_h_readout():
    assert TS.trainscaffold_verdict(_cells("GENERALIZES", "AT_FLOOR"), True, False, "H-severance", 3)["named"] == "H-readout"


def test_verdict_plasticity_vs_total_by_collateral():
    assert TS.trainscaffold_verdict(_cells("AT_FLOOR", "AT_FLOOR"), True, False, "H-capacity", 3)["named"] == "H-plasticity-loss"
    assert TS.trainscaffold_verdict(_cells("AT_FLOOR", "AT_FLOOR"), False, False, "H-capacity", 3)["named"] == "H-lobotomy-total"


def test_verdict_matched_null_blocks_lobotomy():
    # A at floor in both formats BUT emrand also reproduces the failure ⇒ generic removal, not top-σ.
    assert TS.trainscaffold_verdict(_cells("AT_FLOOR", "AT_FLOOR"), True, True, "H-capacity", 3)["named"] == "MATCHED-NULL"


def test_verdict_unresolved_and_non_result():
    assert TS.trainscaffold_verdict(_cells("GRADED", "GENERALIZES"), True, False, "H-severance", 3)["named"] == "format-axis-UNRESOLVED"
    assert TS.trainscaffold_verdict(_cells("GENERALIZES", "GENERALIZES"), True, False, "H-severance", 1)["named"] == "NON-RESULT"
