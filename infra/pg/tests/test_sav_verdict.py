"""Weight-free truth-table tests for SAV-A §6 (sav_verdict.py) — every §1
branch incl. the pre-registered precedence (collapsed-with-qualifier),
instability, indeterminate. <2s, no model/GPU.

    python -m pytest infra/pg/tests/test_sav_verdict.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from infra.pg import sav_verdict as SV

N = 600
BASE = (np.random.default_rng(1).random(N) < 0.55).astype(float).tolist()
FLOOR = (np.random.default_rng(2).random(N) < 0.25).astype(float).tolist()


def _arr(p, seed):
    return (np.random.default_rng(seed).random(N) < p).astype(float).tolist()


def _ladders(pg1, pi0, pi05, rnd=0.25, seed0=10, n_seeds=2):
    """Per-seed array LISTS per n (the post-critic items+seeds engine input)."""
    def ladder(p16, s):
        return {n: [_arr(0.25 + (p16 - 0.25) * (i + 1) / 4, s + i * 7 + j)
                    for j in range(n_seeds)]
                for i, n in enumerate(SV.LADDER)}
    return ({"PG1": ladder(pg1, seed0), "pi0": ladder(pi0, seed0 + 100),
             "pi05": ladder(pi05, seed0 + 200)}, ladder(rnd, seed0 + 300))


def _run(pg1, pi0, pi05, dl=0.05, dp=0.05, unstable=()):
    ck, rnd = _ladders(pg1, pi0, pi05)
    sd = SV.savings_draws(ck, rnd, BASE, FLOOR, n_boot=300, seed=3)
    t = SV.family_tests(sd)
    return SV.sav_verdicts(t, delta_level=dl, delta_pair=dp, unstable_tests=unstable), t


def test_auc_log2n_constant_is_identity():
    assert SV._auc_log2n([0.4, 0.4, 0.4, 0.4], SV.LADDER) == pytest.approx(0.4)
    assert SV._auc_log2n([0.0, 1.0], (4000, 16000)) == pytest.approx(0.5)


def test_conserved_branch():
    v, t = _run(0.50, 0.50, 0.50, dp=0.30)
    assert v["VERDICT"].startswith("H_conserved")
    assert all(v["test_results"][x] == "sig" for x in ("T1_sav_PG1", "T2_sav_pi0", "T3_sav_pi05"))


def test_consumed_branch():
    v, t = _run(0.55, 0.42, 0.33, dl=0.02, dp=0.02)
    assert v["VERDICT"] == "H_consumed"
    # strict hop-1: EXACTLY equal PG1/pi0 (identical arrays ⇒ identical points)
    # must NOT qualify as consumed (m10 fold)
    ck, rnd = _ladders(0.45, 0.45, 0.30)
    ck["pi0"] = {n: [list(a) for a in ck["PG1"][n]] for n in SV.LADDER}
    sd = SV.savings_draws(ck, rnd, BASE, FLOOR, n_boot=300, seed=6)
    t2 = SV.family_tests(sd)
    v2 = SV.sav_verdicts(t2, delta_level=0.02, delta_pair=0.02)
    assert v2["points"]["PG1"] == v2["points"]["pi0"]
    assert v2["VERDICT"] != "H_consumed"
    assert v["points"]["PG1"] > v["points"]["pi0"] > v["points"]["pi05"]


def test_collapsed_with_consumed_qualifier_precedence():
    v, t = _run(0.55, 0.25, 0.25, dl=0.12)
    assert v["VERDICT"].startswith("H_collapsed")
    assert "consumed-to-floor" in v["VERDICT"]          # precedence + qualifier


def test_collapsed_plain_when_pg1_also_flat():
    v, t = _run(0.26, 0.25, 0.25, dl=0.12)
    assert v["VERDICT"].startswith("H_collapsed")
    assert "qualifier" not in v["VERDICT"]


def test_indeterminate_when_nonmonotone():
    v, t = _run(0.40, 0.25, 0.45, dl=0.01, dp=0.01)     # dip then rise: no branch fits
    assert v["VERDICT"] == "indeterminate"


def test_instability_per_test_voiding():
    # T6 voided: H_consumed cannot fire (needs T6); pattern falls through to
    # indeterminate WITH the instability note (per-test rule, critic M3 fold) —
    # NOT a global non-result.
    v, t = _run(0.55, 0.42, 0.33, dl=0.02, dp=0.02, unstable=("T6_droptotal",))
    assert v["VERDICT"].startswith("indeterminate (instability-touched")
    # all six voided → global non-result
    allv = ("T1_sav_PG1", "T2_sav_pi0", "T3_sav_pi05", "T4_drop1", "T5_drop2", "T6_droptotal")
    v2, _ = _run(0.55, 0.42, 0.33, unstable=allv)
    assert v2["VERDICT"] == "non-result (instability: all tests voided)"


def test_seed_resampling_widens_with_seed_noise():
    # two seeds at very different p → seed-resampled draws must be wider than
    # a single-seed run (the items+seeds engine carries seed variance).
    import numpy as np
    ck1, rnd1 = _ladders(0.50, 0.40, 0.30, n_seeds=1)
    sd1 = SV.savings_draws(ck1, rnd1, BASE, FLOOR, n_boot=300, seed=4)
    ck2, rnd2 = _ladders(0.50, 0.40, 0.30, n_seeds=1)
    ck2["PG1"] = {n: [ _arr(0.30, 50+i), _arr(0.65, 60+i) ] for i, n in enumerate(SV.LADDER)}
    sd2 = SV.savings_draws(ck2, rnd2, BASE, FLOOR, n_boot=300, seed=4)
    assert np.std(sd2["draws"]["PG1"]) > 1.5 * np.std(sd1["draws"]["PG1"])


def test_auc_weight_l2_value():
    w = SV.auc_weight_l2(SV.LADDER)
    assert 0.4 < w < 0.7        # ≈0.53 for the 4-point log2 ladder


def test_family_must_match():
    with pytest.raises(AssertionError):
        SV.sav_verdicts({"T1_sav_PG1": {}}, 0.1, 0.1)


def test_t5_reported_regardless():
    v, t = _run(0.55, 0.42, 0.33, dl=0.02, dp=0.02)
    assert "pi0_vs_pi05_T5" in v and "ci95" in v["pi0_vs_pi05_T5"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
