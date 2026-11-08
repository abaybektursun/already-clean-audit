"""Weight-free truth-table tests for the W17 §6 verdict path (w17_verdict.py).
The spec-§3.1 'decision-rule truth table on synthetic inputs' gate — every
branch (savings any-of, no-savings qualified, indeterminate, displacement,
format, instability) exercised on planted arrays. <1s, no model/GPU.

    python -m pytest infra/pg/tests/test_w17_verdict.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import w17_verdict as V


def _bern(n, p, seed):
    return (np.random.default_rng(seed).random(n) < p).astype(float).tolist()


N = 800
BEF = _bern(N, 0.55, 1)
FLO = _bern(N, 0.25, 2)


def _seeds(p, s0):
    return [_bern(N, p, s0 + k) for k in range(3)]


def _tests(cap1k, rnd1k, shf1k, cap16, rnd16, shf16, ret16):
    return {
        "sav_sup_1k": V.paired_arm_diff_s(cap1k, rnd1k, BEF, FLO, n_boot=300, seed=3),
        "sav_floor_1k": V.paired_arm_diff_s(cap1k, shf1k, BEF, FLO, n_boot=300, seed=4),
        "sav_sup_16k": V.paired_arm_diff_s(cap16, rnd16, BEF, FLO, n_boot=300, seed=5),
        "sav_floor_16k": V.paired_arm_diff_s(cap16, shf16, BEF, FLO, n_boot=300, seed=6),
        "disp_16k": V.paired_arm_diff_s(cap16, ret16, BEF, FLO, n_boot=300, seed=7),
    }


def _deq():
    # tight δ from low-noise band sources
    src = [_seeds(0.25, 100), _seeds(0.25, 200)]
    d = V.delta_eq(src, BEF, FLO, n_boot=300, seed=8)
    return {"1k": d["delta_eq"], "16k": d["delta_eq"]}


def test_diff_s_point_and_p_directions():
    hi, lo = _seeds(0.50, 10), _seeds(0.25, 20)
    up = V.paired_arm_diff_s(hi, lo, BEF, FLO, n_boot=300, seed=9)
    assert up["point"] > 0.5 and up["p_one_sided"] < 0.01 and up["ci95"][0] > 0
    null = V.paired_arm_diff_s(_seeds(0.25, 30), _seeds(0.25, 40), BEF, FLO,
                               n_boot=300, seed=9)
    assert null["p_one_sided"] > 0.05 and abs(null["point"]) < 0.1


def test_savings_supported_any_of():
    # 16k separates, 1k does not → H_savings supported via 16k.
    t = _tests(cap1k=_seeds(0.25, 50), rnd1k=_seeds(0.25, 60), shf1k=_seeds(0.25, 70),
               cap16=_seeds(0.50, 80), rnd16=_seeds(0.25, 90), shf16=_seeds(0.25, 95),
               ret16=_seeds(0.27, 97))
    out = V.w17_verdicts(t, _deq(), spreads={})
    assert out["H_savings"] == "supported"
    assert out["savings_per_nstar"] == {"1k": False, "16k": True}
    assert out["H_no_savings"] == "killed"
    assert "killed" in out["H_displacement"]            # cap ≫ retain too


def test_no_savings_qualified_when_all_null_and_tight():
    t = _tests(*[_seeds(0.25, 50 + i) for i in range(7)])
    out = V.w17_verdicts(t, _deq(), spreads={})
    assert out["H_savings"] == "not demonstrated"
    assert out["H_no_savings"].startswith("supported (qualified")
    assert out["H_format"].startswith("supported (qualified")


def test_indeterminate_when_ns_but_wide():
    t = _tests(*[_seeds(0.25, 150 + i) for i in range(7)])
    deq_tiny = {"1k": 1e-6, "16k": 1e-6}                # CIs cannot fit inside
    out = V.w17_verdicts(t, deq_tiny, spreads={})
    assert out["H_no_savings"] == "indeterminate-savings"
    assert out["H_displacement"] == "indeterminate"


def test_instability_overrides_to_non_result():
    t = _tests(cap1k=_seeds(0.25, 50), rnd1k=_seeds(0.25, 60), shf1k=_seeds(0.25, 70),
               cap16=_seeds(0.50, 80), rnd16=_seeds(0.25, 90), shf16=_seeds(0.25, 95),
               ret16=_seeds(0.27, 97))
    spreads = {"cap16": {"spread": 99.0, "n_star": "16k",
                         "tests": ["sav_sup_16k", "sav_floor_16k", "disp_16k"]}}
    out = V.w17_verdicts(t, _deq(), spreads=spreads)
    assert out["test_results"]["sav_sup_16k"] == "non-result"
    assert out["H_savings"] == "not demonstrated"        # 16k leg voided, 1k null
    assert out["H_no_savings"] == "non-result (instability)"
    assert out["H_displacement"] == "non-result (instability)"


def test_family_must_match_exactly():
    with pytest.raises(AssertionError):
        V.w17_verdicts({"sav_sup_1k": {}}, {"1k": 0.1, "16k": 0.1}, {})


def test_delta_eq_monotone_in_noise():
    quiet = V.delta_eq([_seeds(0.25, 300)], BEF, FLO, n_boot=200, seed=11)["delta_eq"]
    # noisy: seeds at very different p → big seed-pair gaps
    noisy_cell = [_bern(N, 0.25, 1), _bern(N, 0.45, 2), _bern(N, 0.30, 3)]
    noisy = V.delta_eq([noisy_cell], BEF, FLO, n_boot=200, seed=11)["delta_eq"]
    assert noisy > quiet > 0


def test_seed_mean_and_per_seed_s():
    arrs = [[1, 0, 1, 1], [1, 1, 0, 1]]
    m = V.seed_mean_items(arrs)
    assert m.tolist() == [1.0, 0.5, 0.5, 1.0]
    ss = V.per_seed_s(arrs, [1, 1, 1, 1], [0, 0, 0, 0])
    assert ss == [0.75, 0.75]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
