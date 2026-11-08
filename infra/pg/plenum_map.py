"""PLENUM — the map itself: UNITS × PROBES, the role partition, and the terminal-state predicate.

Spec: studies/lineage/paligemma1-pi0/research/2026-07-28_plenum_program_spec.md §1.1, §3.4.

The program's deliverable is a COMPLETED map: every cell in exactly one terminal state, none
`pending`. This module is the single definition of what the cells ARE, so "filled" is a mechanical
predicate over a data structure rather than a judgment made in prose (§1.1).

Terminal states (§1.1 + §3.4):
  measured           a verdict exists, with a scope and an artifact pointer
  impossible         structural — the probe does not exist for this unit (frozen in
                     map_impossible.json BEFORE measurement begins; §1.1 forbids declaring a cell
                     impossible afterwards)
  VOID               a NAMED gate ran and failed (the gate id is carried on the cell)
  rule_selected_out  the §3.4 frozen selection rule excludes it; the rule is printed in MAP.md
  pending            none of the above — the run is not done

No try/except (CLAUDE.md). Pure data + pure functions: no torch, no network, no I/O.
"""
from __future__ import annotations

TERMINAL = ("measured", "impossible", "VOID", "rule_selected_out")

# ===========================================================================
# UNITS (§1.1, as amended 2026-07-28-b/-c: 32 named units + synthetic controls)
# ===========================================================================
# roles:
#   fit        the 26 PG1-hop children — the ONLY rows the law is fit on
#   reference  PG1 (ceiling) and gemma-2b (pre-install reference); never fit rows
#   heldout    U3, U3c — predicted from the fitted law, scored out-of-population
#   control    STITCH-0 (pre-point control unit), U4 (random-VLM floor), synthetics
FT_SLUGS = (
    "ai2d", "aokvqa-da", "aokvqa-mc", "coco35l", "cococap", "docvqa", "gqa", "infovqa",
    "nlvr2", "ocrvqa", "okvqa", "refcoco-seg", "rsvqa-hr", "rsvqa-lr", "scicap", "science-qa",
    "screen2words", "stvqa", "tallyqa", "textcaps", "textvqa", "vizwizvqa", "vqav2", "widgetcap",
)
FT_REPO = "google/paligemma-3b-ft-{slug}-224"

# Faculty-typed supervision per FT recipe (§2.1 "recipe-membership"). PROVISIONAL until the freeze
# job cite-verifies every card claim-level (§8 threat row: a card that cannot be verified drops the
# unit from the LAW FIT but keeps it measured). `recipe_verified` starts False for every unit and is
# set only by `plenum_freeze`.
FT_FACULTIES = {
    "ai2d": ("vqa",),
    "aokvqa-da": ("vqa",),
    "aokvqa-mc": ("vqa",),
    "coco35l": ("caption",),
    "cococap": ("caption",),
    "docvqa": ("vqa", "ocr"),
    "gqa": ("vqa",),
    "infovqa": ("vqa", "ocr"),
    "nlvr2": ("vqa",),
    "ocrvqa": ("vqa", "ocr"),
    "okvqa": ("vqa",),
    "refcoco-seg": ("grounding",),
    "rsvqa-hr": ("vqa",),
    "rsvqa-lr": ("vqa",),
    "scicap": ("caption", "ocr"),
    "science-qa": ("vqa",),
    "screen2words": ("caption",),
    "stvqa": ("vqa", "ocr"),
    "tallyqa": ("vqa",),
    "textcaps": ("caption", "ocr"),
    "textvqa": ("vqa", "ocr"),
    "vizwizvqa": ("vqa",),
    "vqav2": ("vqa",),
    "widgetcap": ("caption",),
}

# The six behavioral faculties the law is defined over (§1.3 "26-unit × 6-faculty grid").
# `control` is a seventh faculty, measured on policy units only and NOT part of the law grid.
LAW_FACULTIES = ("vqa", "ocr", "grounding", "caption", "text_recall", "reading")
ALL_FACULTIES = LAW_FACULTIES + ("control",)

# faculty -> the battery task that reads it (§2.1)
FACULTY_TASK = {
    "vqa": "vqav2", "ocr": "textvqa", "grounding": "refcoco", "caption": "coco_captions",
    "text_recall": "mmlu", "reading": "rfp", "control": "libero",
}
MM_FACULTIES = ("vqa", "ocr", "grounding", "caption")
TEXT_FACULTIES = ("text_recall", "reading")


def _unit(uid, role, kind, **kw):
    u = {"id": uid, "role": role, "kind": kind,
         # structural capabilities — these drive the IMPOSSIBLE enumeration
         "has_vision": True, "has_projector": True, "has_expert": False, "has_recipe": True,
         "recipe_faculties": (), "recipe_source": None, "recipe_verified": False}
    u.update(kw)
    return u


def units():
    """The 32 named units + the 2 synthetic controls, in map order."""
    out = [
        _unit("gemma-2b", "reference", "base_lm", has_vision=False, has_projector=False,
              has_recipe=False),
        _unit("stitch0", "control", "stitch", has_recipe=False),
        _unit("pg1", "reference", "install", recipe_faculties=("vqa", "ocr", "caption", "grounding"),
              recipe_source="google/paligemma-3b-pt-224 (mixture pretrain)"),
    ]
    for slug in FT_SLUGS:
        out.append(_unit(f"ft-{slug}", "fit", "pg1_finetune",
                         recipe_faculties=FT_FACULTIES[slug],
                         recipe_source=FT_REPO.format(slug=slug)))
    out += [
        _unit("pi0", "fit", "vla_policy", has_expert=True,
              recipe_faculties=("control",), recipe_source="physical-intelligence/pi0 (VLAT-verified)"),
        _unit("pi05", "fit", "vla_policy", has_expert=True,
              recipe_faculties=("control", "vqa", "caption"),
              recipe_source="physical-intelligence/pi0.5 (VLAT-verified: VL co-training)"),
        _unit("u3", "heldout", "pg1_plus_fresh_expert", has_expert=True,
              recipe_faculties=("control",), recipe_source="PLENUM-constructed (LACUNA Arm-3 F0)"),
        _unit("u4", "control", "random_vlm_plus_fresh_expert", has_expert=True, has_recipe=False,
              recipe_source="PLENUM-constructed floor (LACUNA Arm-3 F0)"),
        _unit("u3c", "heldout", "pg1_plus_expert_cotrained", has_expert=True,
              recipe_faculties=("control", "vqa"),
              recipe_source="PLENUM-constructed (U3 + VL co-training, §2.3)"),
        _unit("matched_dw", "control", "synthetic", has_recipe=False),
        _unit("rand_dec", "control", "synthetic", has_recipe=False),
    ]
    ids = [u["id"] for u in out]
    assert len(ids) == len(set(ids)), "duplicate unit id"
    named = [u for u in out if u["kind"] != "synthetic"]
    assert len(named) == 32, f"§1.1 says 32 named units, got {len(named)}"
    assert sum(1 for u in out if u["role"] == "fit") == 26, "the law fit population must be 26"
    return out


def unit_by_id():
    return {u["id"]: u for u in units()}


# ===========================================================================
# PROBES (§1.1) — component × layer × faculty × readout, each with an instrument id.
# ===========================================================================
COMPONENTS = ("lm", "vision", "projector", "embedding", "expert")
FEATURE_SITES = ("L9", "L14")
# The Arm-2 streams the spec names (§2.2). Every other unit's feature cells are rule-selected-out.
FEATURE_STREAMS = ("gemma-2b", "pg1", "pi05", "pi0")
# The film units (§2.3) — dynamics is measured here and rule-selected-out elsewhere (§3.4).
FILM_UNITS = ("pi0", "u3", "u3c")


def _probe(pid, layer, **kw):
    p = {"id": pid, "layer": layer, "faculty": None, "readout": None, "component": None,
         "site": None, "instrument": None, "tier": 1}
    p.update(kw)
    return p


def probes():
    """Every probe in the map, in render order."""
    out = []
    # --- behavior -------------------------------------------------------------------
    for f in MM_FACULTIES:
        for ro in ("forced_choice", "generation"):      # forced-choice PRIMARY (§2.1)
            out.append(_probe(f"behavior/{f}/{ro}", "behavior", faculty=f, readout=ro,
                              component="whole_system", instrument="plenum_fc" if
                              ro == "forced_choice" else "lacuna_ladder"))
    for f in TEXT_FACULTIES:
        for ro in ("mcq", "cloze"):
            out.append(_probe(f"behavior/{f}/{ro}", "behavior", faculty=f, readout=ro,
                              component="whole_system", instrument="lacuna_text"))
    out.append(_probe("behavior/text_ppl/ppl", "behavior", faculty="text_ppl", readout="ppl",
                      component="whole_system", instrument="lacuna_text"))
    out.append(_probe("behavior/control/task_native", "behavior", faculty="control",
                      readout="task_native", component="whole_system", instrument="lerobot-eval"))
    # --- weights + geometry census ---------------------------------------------------
    for c in COMPONENTS:
        out.append(_probe(f"weights/{c}/bytes", "weights", component=c,
                          instrument="plenum_census"))
        out.append(_probe(f"geometry/{c}/srank", "geometry", component=c,
                          instrument="plenum_census"))
    # --- features (Arm 2) -------------------------------------------------------------
    for s in FEATURE_SITES:
        out.append(_probe(f"features/{s}/identity", "features", site=s, component="lm",
                          instrument="plenum_sae", tier=2))
    # --- dynamics ---------------------------------------------------------------------
    out.append(_probe("dynamics/film/onset", "dynamics", component="whole_system",
                      instrument="plenum_film", tier=2))
    out.append(_probe("dynamics/relearn/curve", "dynamics", component="lm",
                      instrument="relearn", tier=2))
    # --- mechanism (Arm 4, Tier-3: triggers already fired by LACUNA) --------------------
    out.append(_probe("mechanism/locus/depth_patch", "mechanism", component="lm",
                      instrument="plenum_patch", tier=3))
    out.append(_probe("mechanism/swap/embedding", "mechanism", component="embedding",
                      instrument="plenum_patch", tier=3))
    out.append(_probe("mechanism/swap/projector", "mechanism", component="projector",
                      instrument="plenum_patch", tier=3))
    ids = [p["id"] for p in out]
    assert len(ids) == len(set(ids)), "duplicate probe id"
    return out


# ===========================================================================
# The structural-impossibility rule (§1.1) — frozen BEFORE measurement.
# ===========================================================================
def impossible_reason(unit, probe):
    """Why this (unit, probe) cell cannot exist, or None if it can.

    STRUCTURAL only: an absent component or an absent faculty. Never a budget or a difficulty
    judgment — those are `rule_selected_out` (§3.4) or `VOID` (a gate fired), which are different
    terminal states and are recorded as such.
    """
    f, layer, comp = probe["faculty"], probe["layer"], probe["component"]
    if layer == "behavior":
        if f in MM_FACULTIES and not unit["has_vision"]:
            return f"{unit['id']} has no vision tower — the image faculties do not exist for it"
        if f == "control" and not unit["has_expert"]:
            return f"{unit['id']} has no action expert — control is not defined"
    if layer in ("weights", "geometry"):
        if comp == "expert" and not unit["has_expert"]:
            return f"{unit['id']} has no action expert"
        if comp == "vision" and not unit["has_vision"]:
            return f"{unit['id']} has no vision tower"
        if comp == "projector" and not unit["has_projector"]:
            return f"{unit['id']} has no projector"
    if layer == "mechanism" and comp == "projector" and not unit["has_projector"]:
        return f"{unit['id']} has no projector"
    # Note: a mechanism cell on a unit OUTSIDE the §2.4 contrast set is NOT impossible — the
    # forward exists. It is rule-selected-out, which is a different (and honest) terminal state.
    return None


def rule_selected_out_reason(unit, probe):
    """The §3.4 frozen selection rule. Printed verbatim in MAP.md so the one place the map is
    complete BY RULE rather than by measurement is visible, not buried."""
    if probe["layer"] == "features":
        if unit["id"] not in FEATURE_STREAMS:
            return ("§2.2 names four feature streams (gemma-2b text, PG1, π0.5, π0); the SAE "
                    "instrument is not fanned out over the roster")
    if probe["id"] == "dynamics/film/onset" and unit["id"] not in FILM_UNITS:
        return "§2.3 films U1/U3/U3c only — the checkpoint battery needs a training run"
    if probe["id"] == "dynamics/relearn/curve":
        return ("§3.4: relearnability cells exist for the SAV-18; the rule adds none — the film "
                "supplies this program's dynamics layer")
    if probe["layer"] == "mechanism" and unit["id"] not in ("pg1", "pi05", "pi0"):
        return "§2.4 defines the mechanism contrasts on the PG1↔π0.5 pair (π0 for the swap arms)"
    return None


def cell_id(unit_id, probe_id):
    return f"{unit_id}|{probe_id}"


def all_cells():
    """Every (unit, probe) cell with its PRE-MEASUREMENT disposition.

    Returns a list of dicts carrying `frozen_state` ∈ {impossible, rule_selected_out, expected}.
    `expected` cells are the ones the program must actually measure (or VOID via a named gate).
    """
    out = []
    for u in units():
        for p in probes():
            imp = impossible_reason(u, p)
            rule = rule_selected_out_reason(u, p) if imp is None else None
            out.append({
                "cell": cell_id(u["id"], p["id"]), "unit": u["id"], "probe": p["id"],
                "layer": p["layer"], "tier": p["tier"], "role": u["role"],
                "faculty": p["faculty"], "readout": p["readout"], "component": p["component"],
                "instrument": p["instrument"],
                "frozen_state": ("impossible" if imp else
                                 "rule_selected_out" if rule else "expected"),
                "reason": imp or rule,
            })
    return out


def state_of(frozen, measured):
    """The terminal-state predicate (§1.1), computed mechanically.

    `frozen`: one `all_cells()` entry. `measured`: the aggregator's record for that cell, or None.
    A measured record must carry a verdict, a scope and a pointer (§10: cells are pointers, never
    numbers) — a record missing any of the three is NOT terminal, it is pending, which is what
    makes `pending=0` a real gate rather than a formality.
    """
    if frozen["frozen_state"] in ("impossible", "rule_selected_out"):
        return frozen["frozen_state"]
    if measured is None:
        return "pending"
    if measured.get("state") == "VOID":
        assert measured.get("gate"), f"{frozen['cell']}: VOID without a named gate is not terminal"
        return "VOID"
    if all(measured.get(k) for k in ("verdict", "scope", "pointer")):
        return "measured"
    return "pending"


def render_counts(frozen_cells, records):
    """{state: n} over the whole map + the pending list (the program's definition of done)."""
    counts, pending = {}, []
    for fz in frozen_cells:
        st = state_of(fz, records.get(fz["cell"]))
        counts[st] = counts.get(st, 0) + 1
        if st == "pending":
            pending.append(fz["cell"])
    counts.setdefault("pending", 0)
    return counts, sorted(pending)
