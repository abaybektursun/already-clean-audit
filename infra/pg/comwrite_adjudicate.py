"""COMWRITE §5.3 frozen decision rules — the pattern→identity-label adjudicator.

Two pure layers, both weight-free-tested on synthetic cell tables (§10.1 dry-run: every label
reachable + mutually exclusive):
  predicates(...)  numbers → the frozen boolean pattern flags (damage-carried,
                   task-carried-by-residual, generic-movement, …).
  label(flags)     flags → ONE primary identity label under a FROZEN precedence order.

PRECEDENCE (frozen at authoring — §1 says the labels are "mutually exclusive as primary
labels, graded mixtures reported as the measured pattern"; where §5.3's iff-conditions can
co-occur the order below resolves them, and the full flag dict is always reported alongside
so mixtures stay visible):
  1. H_entangled        FT ≠ base while BOTH single parts ≈ base on every measured family
                        (the interaction owns the function — checked first because it entails
                        the H_inert antecedent).
  2. H_damage_carrier   damage-carried ∧ task-carried-by-residual (the separable practical
                        branch).
  3. H_damage_entangled damage-carried ∧ ¬task-carried-by-residual (the LATABLATE pattern in
                        weight space).
  4. H_shared_skill     cross-task SCORE gains clear their bands at ≥2 tasks.
  5. H_interface        generic movement without cross-task score gains and without
                        damage-carried.
  6. H_inert            no readout family of base+shared moves beyond its band while the
                        energy bookkeeping confirms the shared cell carries ≥ the
                        family-median captured energy.
  7. H_mixed            none of the above — the measured pattern is reported without a name.

Damage variable: capped log₁₀ ppl (pre-mortem ii — 4-digit FT ppl saturates the raw scale).
Bands are (lo, hi) tuples calibrated on discovery control cells (identity + em-random draws)
and recorded pre-unblind (§2). Pure python/numpy, NO try/except.
"""
from __future__ import annotations

LABELS = ("H_entangled", "H_damage_carrier", "H_damage_entangled", "H_shared_skill",
          "H_interface", "H_inert", "H_mixed")


# ---------------------------------------------------------------------------
# the damage variable.
# ---------------------------------------------------------------------------
def capped_log10_ppl(ppl, cap):
    """log₁₀ of a perplexity, capped at the frozen dead-model reference `cap` (§3). Cap hits
    are the caller's to disclose per cell (§5.4)."""
    import math
    p = float(ppl)
    assert p > 0 and float(cap) > 0, "ppl/cap must be positive"
    return math.log10(min(p, float(cap)))


def d_ppl(ppl_cell, ppl_base, cap):
    """D_ppl(cell) = capped-log₁₀-ppl(cell) − capped-log₁₀-ppl(base) (§5.3)."""
    return capped_log10_ppl(ppl_cell, cap) - capped_log10_ppl(ppl_base, cap)


# ---------------------------------------------------------------------------
# band helpers. A band is (lo, hi) — the calibrated null interval for an EFFECT (cell − base).
# ---------------------------------------------------------------------------
def beyond(effect, band):
    """Effect outside the null band, either direction."""
    lo, hi = float(band[0]), float(band[1])
    assert lo <= hi, f"malformed band {band}"
    return bool(effect < lo or effect > hi)


def gain_beyond(effect, band):
    """Effect above the band's UPPER edge (a score GAIN, the H_shared-skill direction)."""
    return bool(float(effect) > float(band[1]))


def interaction(s_shared, s_resid, s_base, s_ft):
    """I = [S(+shared) + S(+residual)] − [S(base) + S(FT)] — reported with CI for every
    readout family regardless of label (§3/§5.3)."""
    return float(s_shared) + float(s_resid) - float(s_base) - float(s_ft)


# ---------------------------------------------------------------------------
# layer 1 — numbers → frozen pattern flags.
# ---------------------------------------------------------------------------
def predicates(*, s_base, s_ft, s_shared, s_resid, score_band,
               dppl_shared, dppl_resid, f_ppl, ppl_band,
               cross_score_effects, cross_score_bands,
               cross_nll_effects=(), cross_nll_bands=(),
               resid_cross_effects=None, resid_cross_bands=None,
               energy_median_ok, frac=0.5):
    """The §5.3 predicate set for ONE donor at k*.

    Scores are the donor-task readout per cell; score_band / ppl_band are the em-random null
    bands for score-effects (cell − base) and damage-effects respectively. cross_* are the
    base+shared cell's effects on the OTHER donors' tasks (score) and NLL texture, each with
    its band. resid_cross_* (optional) extend the H_entangled "all families" check to the
    residual cell where measured. energy_median_ok = the bookkeeping flag that the shared
    cell carries ≥ the family-median captured energy (job-computed).
    Returns the flag dict `label()` consumes + the measured numbers it derived them from."""
    assert len(cross_score_effects) == len(cross_score_bands), "cross score effect/band drift"
    assert len(cross_nll_effects) == len(cross_nll_bands), "cross NLL effect/band drift"
    gain_ft = float(s_ft) - float(s_base)
    gain_shared = float(s_shared) - float(s_base)
    gain_resid = float(s_resid) - float(s_base)

    # Damage is an INCREASE in capped-log ppl: both the FT and the shared cell must sit
    # above the band's UPPER edge. Without the F_ppl guard the ≥frac·F_ppl conjunct is
    # vacuous when the FT carries no damage (f_ppl≈0 ⇒ threshold≈0), and a two-sided
    # `beyond` would fire damage-carried on a ppl IMPROVEMENT.
    damage_carried = bool(float(f_ppl) > float(ppl_band[1])
                          and float(dppl_shared) >= frac * float(f_ppl)
                          and float(dppl_shared) > float(ppl_band[1]))
    task_carried = bool(gain_resid >= frac * gain_ft and beyond(gain_resid, score_band))
    n_cross_gains = sum(gain_beyond(e, b)
                        for e, b in zip(cross_score_effects, cross_score_bands))
    generic = bool(any(beyond(e, b) for e, b in zip(cross_score_effects, cross_score_bands))
                   or any(beyond(e, b) for e, b in zip(cross_nll_effects, cross_nll_bands)))

    shared_moves = bool(beyond(gain_shared, score_band) or generic
                        or beyond(dppl_shared, ppl_band))
    resid_moves = bool(beyond(gain_resid, score_band) or beyond(dppl_resid, ppl_band))
    if resid_cross_effects is not None:
        assert resid_cross_bands is not None and \
            len(resid_cross_effects) == len(resid_cross_bands), "resid cross effect/band drift"
        resid_moves = bool(resid_moves or any(
            beyond(e, b) for e, b in zip(resid_cross_effects, resid_cross_bands)))
    ft_differs = bool(beyond(gain_ft, score_band) or beyond(f_ppl, ppl_band))

    return {"damage_carried": damage_carried, "task_carried": task_carried,
            "n_cross_gains": int(n_cross_gains), "generic": generic,
            "shared_moves": shared_moves, "resid_moves": resid_moves,
            "ft_differs": ft_differs, "energy_median_ok": bool(energy_median_ok),
            "gain_ft": gain_ft, "gain_shared": gain_shared, "gain_resid": gain_resid,
            "dppl_shared": float(dppl_shared), "dppl_resid": float(dppl_resid),
            "f_ppl": float(f_ppl), "frac": float(frac)}


# ---------------------------------------------------------------------------
# layer 2 — flags → the primary label (frozen precedence, module docstring).
# ---------------------------------------------------------------------------
def label(flags):
    """ONE primary identity label from a predicate dict. Deterministic; the caller reports
    the full flag dict alongside so co-occurring patterns stay visible (§1 graded mixtures)."""
    f = flags
    if f["ft_differs"] and not f["shared_moves"] and not f["resid_moves"]:
        return "H_entangled"
    if f["damage_carried"] and f["task_carried"]:
        return "H_damage_carrier"
    if f["damage_carried"]:
        return "H_damage_entangled"
    if f["n_cross_gains"] >= 2:
        return "H_shared_skill"
    if f["generic"] and f["n_cross_gains"] == 0:
        return "H_interface"                    # §5.3: WITHOUT cross-task score gains —
                                                # exactly one gain is an unlicensed mixture

    if not f["shared_moves"] and f["energy_median_ok"]:
        return "H_inert"
    return "H_mixed"


def adjudicate(**kwargs):
    """predicates(...) + label(...) in one call: {'label': …, 'flags': {…}}."""
    f = predicates(**kwargs)
    return {"label": label(f), "flags": f}


def confirm(discovery, sealed):
    """§5.3 sealed confirmation: the discovery label STANDS only if the sealed-half pattern
    reproduces it (same primary label from the sealed numbers, under the caller's BH gate).
    Else the discovery label is exploratory texture and the sealed pattern is the result —
    no rule swap. Pure comparison of two adjudicate() outputs."""
    stands = bool(discovery["label"] == sealed["label"])
    return {"stands": stands, "discovery_label": discovery["label"],
            "sealed_label": sealed["label"],
            "result_label": sealed["label"]}


def family_label(per_donor_labels):
    """§5.4 anecdote-risk rule: a FAMILY label only when ≥2 donors agree in pattern
    ('robust'); else the family read is 'per-donor' (labels stated per donor only)."""
    from collections import Counter
    labels = list(per_donor_labels)
    assert labels, "no donor labels"
    top, n = Counter(labels).most_common(1)[0]
    return {"family_label": top if n >= 2 else None, "agree_n": int(n),
            "per_donor": labels, "robust": bool(n >= 2)}
