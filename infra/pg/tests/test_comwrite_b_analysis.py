"""Weight-free dry-run gate for the Arm-B adjudication job (comwrite spec §5.3/§7) —
synthetic banked-metrics dicts prove: the band recipe, donor_verdict's label wiring, and
the sealed BH-FDR step (a label with a clean score leg SURVIVES; a noisy leg FAILS BH and
the confirm() verdict falls, no rule swap).

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_comwrite_b_analysis.py -q
"""
from __future__ import annotations

import pytest

from infra.aws_pg.jobs import comwrite_b_analysis as BA

A_METRICS = {"a3_pooled_ladder": {"16": 0.40}, "per_matrix": [{"d": 2048}]}
K = 16


def _readout(score, per_item, ppl):
    return {"donor": {"score": score, "ci95": [score, score], "per_item": per_item},
            "ppl": {"ppl": ppl}}


def _with_cross(r, cross_scores, cross_items, nlls):
    for s, sc in cross_scores.items():
        r[f"cross_{s}"] = {"score": sc, "ci95": [sc, sc], "per_item": cross_items[s]}
        r[f"nll_{s}"] = {"nll": nlls[s], "ci95": [nlls[s], nlls[s]]}
    return r


def _donor_metrics(ident_pi, ft_pi, sh_pi, re_pi, *, ft_ppl, sh_ppl, re_ppl,
                   capture=0.35):
    """A complete b_cells metrics dict: identity/ft/shared/residual + 3 quiet em draws."""
    def mean(v):
        return sum(v) / len(v)
    cross = {"textvqa": 0.30, "cococap": 0.90}
    citems = {s: [v] * 40 for s, v in cross.items()}
    nll = {"textvqa": 1.0, "cococap": 2.0}
    rd = {"identity@0": _with_cross(_readout(mean(ident_pi), ident_pi, 22.4),
                                    cross, citems, nll),
          "ft@0": _with_cross(_readout(mean(ft_pi), ft_pi, ft_ppl), cross, citems, nll),
          "shared@0": {**_with_cross(_readout(mean(sh_pi), sh_pi, sh_ppl),
                                     cross, citems, nll),
                       "book_totals": {"total_captured_fro2": capture,
                                       "total_fro2": 1.0}},
          "residual@0": _with_cross(_readout(mean(re_pi), re_pi, re_ppl),
                                    cross, citems, nll)}
    for j, eps in enumerate((0.004, -0.004, 0.0)):
        rd[f"em_random@{j}"] = _with_cross(
            _readout(mean(ident_pi) + eps, [x + eps for x in ident_pi], 22.4 + 0.1 * j),
            {s: v + eps for s, v in cross.items()},
            {s: [v + eps] * 40 for s, v in cross.items()},
            {s: v + eps for s, v in nll.items()})
    return {"donor": "vqav2", "half": "sealed", "readouts": rd}


CLEAN = _donor_metrics([0.5] * 40, [0.9] * 40, [0.52] * 40, [0.75] * 40,
                       ft_ppl=2000.0, sh_ppl=500.0, re_ppl=30.0)


def test_donor_verdict_damage_carrier_on_a_clean_separable_pattern():
    v = BA.donor_verdict(CLEAN, A_METRICS, K)
    assert v["verdict"]["label"] == "H_damage_carrier"
    assert v["energy_ok"] and v["verdict"]["flags"]["task_carried"]
    lo, hi = v["bands"]["score"]
    assert lo < 0 < hi and hi < 0.01                      # em envelope is tight + two-sided


def test_donor_verdict_inert_when_nothing_moves():
    quiet = _donor_metrics([0.5] * 40, [0.9] * 40, [0.5] * 40, [0.88] * 40,
                           ft_ppl=22.4, sh_ppl=22.4, re_ppl=22.4)
    v = BA.donor_verdict(quiet, A_METRICS, K)
    assert v["verdict"]["label"] == "H_inert"


def test_bh_pass_keeps_the_sealed_label():
    v = BA.donor_verdict(CLEAN, A_METRICS, K)
    rep = {"disc": {"vqav2": v}, "sealed": {"vqav2": v}, "confirm": {},
           "_sealed_raw": {"vqav2": CLEAN}}
    BA.apply_bh_and_confirm(rep)
    assert rep["bh"] and all(t["bh_pass"] for t in rep["bh"])
    assert rep["confirm"]["vqav2"]["stands"]


def test_bh_fail_drops_the_confirmation_without_rule_swap():
    # residual's score leg is beyond the band in POINT estimate but item-noisy →
    # bootstrap p is large → BH rejects nothing → the label does NOT stand.
    ident = [1.0 if i % 2 == 0 else 0.0 for i in range(40)]          # mean .5, noisy
    resid = [1.0 if (i % 2 == 0 or i < 4) else 0.0 for i in range(40)]   # mean .55
    ft = [1.0 if (i % 2 == 0 or i < 6) else 0.0 for i in range(40)]      # mean .575
    noisy = _donor_metrics(ident, ft, [0.52] * 40, resid,
                           ft_ppl=2000.0, sh_ppl=500.0, re_ppl=30.0)
    v = BA.donor_verdict(noisy, A_METRICS, K)
    assert v["verdict"]["label"] == "H_damage_carrier"    # the point pattern says carrier
    rep = {"disc": {"vqav2": v}, "sealed": {"vqav2": v}, "confirm": {},
           "_sealed_raw": {"vqav2": noisy}}
    BA.apply_bh_and_confirm(rep)
    assert rep["bh"] and not any(t["bh_pass"] for t in rep["bh"])
    c = rep["confirm"]["vqav2"]
    assert not c["stands"] and "BH-failed" in c["result_label"]


def test_band_from_draws_is_an_envelope_not_a_best_draw():
    band = BA.band_from_draws([(0.01, 0.002), (-0.02, 0.001), (0.005, 0.0)])
    assert band == (-0.021, 0.012)
