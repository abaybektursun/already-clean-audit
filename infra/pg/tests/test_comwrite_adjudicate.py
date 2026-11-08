"""Weight-free dry-run gate for the §5.3 pattern-rule adjudicator (comwrite spec §10.1) —
synthetic cell tables prove every identity label REACHABLE and the precedence order resolves
co-occurring antecedents deterministically (mutual exclusivity of the primary label).

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_comwrite_adjudicate.py -q
"""
from __future__ import annotations

import pytest

from infra.pg import comwrite_adjudicate as CA


def _quiet(**over):
    """A synthetic donor table where NOTHING moves; tests perturb one pattern at a time."""
    kw = dict(s_base=0.50, s_ft=0.50, s_shared=0.50, s_resid=0.50,
              score_band=(-0.02, 0.02),
              dppl_shared=0.0, dppl_resid=0.0, f_ppl=0.0, ppl_band=(-0.05, 0.05),
              cross_score_effects=[0.0, 0.0],
              cross_score_bands=[(-0.02, 0.02), (-0.02, 0.02)],
              cross_nll_effects=[0.0, 0.0],
              cross_nll_bands=[(-0.10, 0.10), (-0.10, 0.10)],
              energy_median_ok=True)
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# the damage variable.
# ---------------------------------------------------------------------------
def test_capped_log10_ppl_caps_at_the_dead_model_reference():
    assert CA.capped_log10_ppl(1e6, cap=1e4) == pytest.approx(4.0)
    assert CA.capped_log10_ppl(100.0, cap=1e4) == pytest.approx(2.0)
    with pytest.raises(AssertionError):
        CA.capped_log10_ppl(0.0, cap=1e4)


def test_d_ppl_is_capped_log_difference():
    assert CA.d_ppl(1000.0, 10.0, cap=1e4) == pytest.approx(2.0)
    assert CA.d_ppl(1e9, 10.0, cap=1e4) == pytest.approx(3.0)          # cap engaged


def test_interaction_term():
    assert CA.interaction(0.6, 0.55, 0.5, 0.7) == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# every label reachable.
# ---------------------------------------------------------------------------
def test_h_inert_reachable():
    r = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.68))
    assert r["label"] == "H_inert"


def test_h_entangled_reachable():
    r = CA.adjudicate(**_quiet(s_ft=0.70))
    assert r["label"] == "H_entangled"
    assert r["flags"]["ft_differs"] and not r["flags"]["shared_moves"]


def test_h_damage_carrier_reachable():
    r = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.62, f_ppl=1.0, dppl_shared=0.6))
    assert r["label"] == "H_damage_carrier"
    assert r["flags"]["damage_carried"] and r["flags"]["task_carried"]


def test_h_damage_entangled_reachable():
    r = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.53, f_ppl=1.0, dppl_shared=0.6))
    assert r["label"] == "H_damage_entangled"
    assert r["flags"]["damage_carried"] and not r["flags"]["task_carried"]


def test_h_shared_skill_reachable_needs_two_tasks():
    two = CA.adjudicate(**_quiet(cross_score_effects=[0.05, 0.06]))
    assert two["label"] == "H_shared_skill"
    one = CA.adjudicate(**_quiet(cross_score_effects=[0.05, 0.0]))
    assert one["label"] != "H_shared_skill"                    # ≥2 tasks, frozen


def test_h_interface_reachable_via_nll_texture_without_score_gains():
    r = CA.adjudicate(**_quiet(cross_nll_effects=[0.3, 0.0]))
    assert r["label"] == "H_interface"
    assert r["flags"]["generic"] and r["flags"]["n_cross_gains"] == 0


def test_h_mixed_reachable():
    r = CA.adjudicate(**_quiet(s_ft=0.70, s_shared=0.60))      # shared moves donor score only
    assert r["label"] == "H_mixed"


def test_one_cross_gain_is_a_mixture_not_interface():
    """§5.3: H_interface requires generic movement WITHOUT cross-task score gains —
    exactly one gain is an unlicensed pattern and must stay H_mixed."""
    r = CA.adjudicate(**_quiet(cross_score_effects=[0.05, 0.0]))
    assert r["label"] == "H_mixed" and r["flags"]["n_cross_gains"] == 1


def test_damage_carried_needs_the_ft_to_carry_damage():
    """f_ppl inside the band (or a ppl IMPROVEMENT) must never fire the damage branch —
    the ≥frac·F_ppl conjunct is vacuous at f_ppl≈0 and inverted at f_ppl<0."""
    near_zero = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.62, f_ppl=0.02,
                                       dppl_shared=0.06))
    assert not near_zero["flags"]["damage_carried"]
    improving = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.62, f_ppl=-0.15,
                                       dppl_shared=-0.06))
    assert not improving["flags"]["damage_carried"]
    assert improving["label"] != "H_damage_carrier"


# ---------------------------------------------------------------------------
# frozen precedence on co-occurring antecedents (mutual exclusivity of the primary).
# ---------------------------------------------------------------------------
def test_damage_precedes_shared_skill():
    r = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.62, f_ppl=1.0, dppl_shared=0.6,
                               cross_score_effects=[0.05, 0.06]))
    assert r["label"] == "H_damage_carrier"
    assert r["flags"]["n_cross_gains"] == 2                    # the mixture stays visible


def test_entangled_precedes_inert_when_both_hold():
    r = CA.adjudicate(**_quiet(s_ft=0.70))                     # nothing moves + energy ok
    assert r["label"] == "H_entangled"


def test_inert_requires_energy_majority():
    r = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.68, energy_median_ok=False))
    assert r["label"] == "H_mixed"


def test_label_is_single_valued_over_a_flag_sweep():
    import itertools
    keys = ("damage_carried", "task_carried", "generic", "shared_moves", "resid_moves",
            "ft_differs", "energy_median_ok")
    for bits in itertools.product([False, True], repeat=len(keys)):
        flags = dict(zip(keys, bits))
        flags["n_cross_gains"] = 2 if flags["generic"] else 0
        lab = CA.label(flags)
        assert lab in CA.LABELS                                # total function, one label


# ---------------------------------------------------------------------------
# sealed confirmation + family rule.
# ---------------------------------------------------------------------------
def test_confirm_stands_only_on_sealed_reproduction():
    disc = CA.adjudicate(**_quiet(s_ft=0.70, s_resid=0.62, f_ppl=1.0, dppl_shared=0.6))
    sealed_same = CA.adjudicate(**_quiet(s_ft=0.72, s_resid=0.63, f_ppl=1.1, dppl_shared=0.7))
    sealed_diff = CA.adjudicate(**_quiet(s_ft=0.70))
    assert CA.confirm(disc, sealed_same)["stands"]
    c = CA.confirm(disc, sealed_diff)
    assert not c["stands"] and c["result_label"] == sealed_diff["label"]   # no rule swap


def test_family_label_needs_two_agreeing_donors():
    r = CA.family_label(["H_damage_entangled", "H_damage_entangled", "H_inert"])
    assert r["robust"] and r["family_label"] == "H_damage_entangled"
    r2 = CA.family_label(["H_inert", "H_interface", "H_entangled"])
    assert not r2["robust"] and r2["family_label"] is None
