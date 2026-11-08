"""Weight-free CPU dry-run gate for SAV-C — prereg §9.1 / §13 gate-1 logic test.
NO model, NO GPU — pure logic on planted rows, seconds on a laptop.

    python3 -m pytest infra/aws_pg/tests/test_savc.py -q

Covers: new-unit row extraction (savb schema mirror) + asserts; the within-family (overlap>cut) masking +
full-pool analysis (reuses infra/pg/savb.py); the FROZEN §6 verdict rule mapping each scenario to the right
label (within-resolution / between-regime-only / saturated / no-gauge / instrument-broken).
"""
import numpy as np
import pytest

from infra.pg import savb as SBV
from infra.aws_pg.jobs import savc_analyze as J


# ---------------------------------------------------------------------------
# new-unit row extraction
# ---------------------------------------------------------------------------
def _synthetic_unit(task="paligemma-3b-ft-scicap-224", overlap=0.70, s16=0.40, s1=0.10):
    return {"unit": task,
            "outcomes": {"16000": {"S_recover": s16}, "1000": {"S_recover": s1}},
            "predictors": {"overlap": overlap, "l2_base": 350.0, "lmc_barrier": -1.2}}


def _cloud(task="paligemma-3b-ft-scicap-224"):
    return {task: {"A": {"64": {"a_pooled_full": 0.3}}, "rhoL_enrichment": {"64": 1.1},
                   "frob_dW_total": 192.8}}


def test_new_unit_row_extracts_savb_schema():
    row = J._new_unit_row(_synthetic_unit(), _cloud())
    assert row["unit"] == "paligemma-3b-ft-scicap-224"
    assert row["overlap"] == 0.70 and row["S_16k"] == 0.40 and row["S_1k"] == 0.10
    assert row["l2_base"] == 350.0 and row["lmc_barrier"] == -1.2
    assert row["a64"] == 0.3 and row["rhoL"] == 1.1 and row["frob_cached"] == 192.8


def test_new_unit_row_fails_loud_off_cloud():
    with pytest.raises(KeyError):
        J._new_unit_row(_synthetic_unit(task="paligemma-3b-ft-unknown-224"), _cloud())


def test_new_unit_row_fails_loud_bad_overlap():
    with pytest.raises(AssertionError):
        J._new_unit_row(_synthetic_unit(overlap=1.7), _cloud())


# ---------------------------------------------------------------------------
# _analyze — within-family masking + the real savb stats
# ---------------------------------------------------------------------------
def _row(unit, overlap, s16, s1=None, l2=340.0, lmc=-1.0):
    return {"unit": unit, "kind": "x", "overlap": overlap, "S_16k": s16,
            "S_1k": s1 if s1 is not None else 0.5 * s16, "l2_base": l2, "lmc_barrier": lmc,
            "a64": 0.3, "rhoL": 1.0, "frob_cached": 50.0}


def _pool_overlap_predicts(n=16, seed=0):
    """A pool where S16 increases with overlap, INCLUDING within the family (overlap>0.62).
    l2_base/lmc jittered (the real within-family L2 band is 340.0-341.2 — non-constant, as SAV-B notes)."""
    rng = np.random.default_rng(seed)
    rows = []
    # 3 low-overlap outliers + an in-family gradient
    for i, ov in enumerate(np.concatenate([[0.49, 0.50, 0.61], np.linspace(0.65, 0.84, n - 3)])):
        s16 = float(np.clip(ov + 0.02 * rng.standard_normal(), 0, 1))   # S16 ~ overlap
        l2 = 340.0 + (290.0 if ov < 0.62 else 0.0) + 0.6 * rng.standard_normal()  # jitter, big jump for outliers
        lmc = -1.0 + 0.3 * rng.standard_normal()
        rows.append(_row(f"u{i}", float(ov), s16, l2=l2, lmc=lmc))
    return rows


def test_analyze_within_family_masks_and_detects():
    pool = _pool_overlap_predicts()
    a = J._analyze(pool, SBV, np, n_boot=300, n_perm=300, cut=0.62)
    wf = a["within_family"]
    assert all(o > 0.62 for o in [pool[i]["overlap"] for i in range(len(pool)) if pool[i]["overlap"] > 0.62])
    assert wf["n"] == sum(1 for r in pool if r["overlap"] > 0.62)
    assert wf["overlap_min"] > 0.62
    # within-family overlap DOES predict S16 here -> positive Spearman
    assert wf["S_16k"]["spearman_overlap"]["rho"] > 0.3, wf["S_16k"]["spearman_overlap"]


def test_analyze_within_family_null_when_flat():
    rng = np.random.default_rng(1)
    # in-family S16 independent of overlap (flat) + 3 outliers carry the between-regime signal
    rows = [_row("o0", 0.49, 0.10, l2=630.0 + rng.standard_normal()),
            _row("o1", 0.50, 0.07, l2=1127.0 + rng.standard_normal()),
            _row("o2", 0.61, 0.59, l2=352.0 + rng.standard_normal())]
    for i, ov in enumerate(np.linspace(0.66, 0.84, 12)):
        l2 = 340.0 + 0.6 * rng.standard_normal()
        lmc = -1.0 + 0.3 * rng.standard_normal()
        rows.append(_row(f"c{i}", float(ov), float(np.clip(0.82 + 0.03 * rng.standard_normal(), 0, 1)),
                         l2=l2, lmc=lmc))
    a = J._analyze(rows, SBV, np, n_boot=300, n_perm=300, cut=0.62)
    assert abs(a["within_family"]["S_16k"]["spearman_overlap"]["rho"]) < 0.5  # no within-family gradient


# ---------------------------------------------------------------------------
# the FROZEN §6 verdict rule
# ---------------------------------------------------------------------------
def _analysis(full_delta, full_p, wf_rho, wf_ci_lo, wf_ci_hi, wf_delta, wf_p, wf_n=15):
    return {"full_pool": {"S_16k": {"loo_overlap": {"delta": full_delta, "p_perm": full_p}}},
            "within_family": {"n": wf_n,
                              "S_16k": {"loo_overlap": {"delta": wf_delta, "p_perm": wf_p},
                                        "spearman_overlap": {"rho": wf_rho, "ci95": [wf_ci_lo, wf_ci_hi]}}}}


def _new(overlaps):
    return [{"overlap": o} for o in overlaps]


def test_verdict_instrument_broken_few_units():
    v = J._verdict(_analysis(0.5, 0.01, 0.6, 0.2, 0.9, 0.4, 0.01), _new([0.7, 0.6]), n_new=2)
    assert v["named"] == "INSTRUMENT_BROKEN", v


def test_verdict_no_gauge_when_full_pool_fails():
    v = J._verdict(_analysis(-0.1, 0.40, 0.6, 0.2, 0.9, 0.4, 0.01), _new([0.7, 0.6, 0.75]), n_new=3)
    assert v["named"] == "H_no-gauge", v


def test_verdict_saturated_when_gap_not_filled():
    # full-pool replicates but new units all cluster >=0.78
    v = J._verdict(_analysis(0.5, 0.01, 0.1, -0.2, 0.4, 0.0, 0.5), _new([0.82, 0.83, 0.84]), n_new=3)
    assert v["named"] == "H_saturated", v


def test_verdict_within_resolution():
    v = J._verdict(_analysis(0.5, 0.01, 0.55, 0.12, 0.88, 0.30, 0.04), _new([0.66, 0.72, 0.78]), n_new=3)
    assert v["named"] == "H_within-resolution", v


def test_verdict_between_regime_only():
    # gap filled, full replicates, but within-family null (CI brackets 0)
    v = J._verdict(_analysis(0.5, 0.01, 0.10, -0.30, 0.45, -0.05, 0.6), _new([0.66, 0.72, 0.78]), n_new=3)
    assert v["named"] == "H_between-regime-only", v
