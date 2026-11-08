"""PLENUM G0 — the offline gate: weight-free dry-runs of every NEW logic path (spec §9 step 2).

`feedback_offline_gate_before_gpu`: the full decision surface runs on the control plane, on real
logic and synthetic data, BEFORE a GPU minute is spent. LACUNA's own G0 found a frozen label rule
that could not fire (AMENDMENT-5); this gate exists to find PLENUM's equivalent.

Every check below names the observation that makes it report FAILURE (`feedback_signals_that_cannot_fail`),
and the law's instrument gets BOTH directions: a PLANTED predictor it must recover, and a RANDOM
predictor it must NOT fire on. A gate that can only pass is not a gate.

    python -m infra.aws_pg.jobs.plenum_offline --out /tmp/plenum_offline
"""
from __future__ import annotations

import argparse
import json
import pathlib

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# ===========================================================================
# 0. Static: every called local helper must exist.
# ===========================================================================
@check("static/no_undefined_calls")
def _no_undefined_calls():
    """`py_compile` passes on a NameError. This catches the ones that reach GPU jobs.

    Added after the aggregator's control block shipped a call to `_read_json`, a helper that does
    not exist in that module (it is `_load`). It compiled, and it would have crashed the whole
    aggregation the first time a --control file was passed — AFTER the census and forced-choice work
    was already done. Cheap static check, real class of bug, and exactly what the offline gate is
    for (`memory/feedback_offline_gate_before_gpu.md`).
    """
    import ast
    import builtins
    import pathlib as _p
    root = _p.Path(__file__).resolve().parents[3]
    files = sorted((root / "infra/aws_pg/jobs").glob("plenum_*.py")) + [
        root / "infra/pg/plenum_units.py", root / "infra/pg/plenum_core.py",
        root / "infra/pg/plenum_map.py", root / "scripts/plenum_map_render.py"]
    offenders = {}
    for f in files:
        tree = ast.parse(f.read_text())
        bound = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(n.name)
            elif isinstance(n, ast.Import):
                bound |= {(al.asname or al.name.split(".")[0]) for al in n.names}
            elif isinstance(n, ast.ImportFrom):
                bound |= {(al.asname or al.name) for al in n.names}
            elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                bound.add(n.id)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
        miss = sorted(c.func.id for c in ast.walk(tree)
                      if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                      and c.func.id not in bound and not hasattr(builtins, c.func.id))
        if miss:
            offenders[f.name] = miss
    assert not offenders, f"undefined calls: {offenders}"
    return {"files_scanned": len(files)}


# ===========================================================================
# 1. The MAP — the terminal-state predicate and the frozen impossible enumeration.
# ===========================================================================
@check("map/cell_algebra")
def _map_algebra():
    from infra.pg import plenum_map as PM
    cells = PM.all_cells()
    units, probes = PM.units(), PM.probes()
    assert len(cells) == len(units) * len(probes), "the map is not a full cross product"
    states = {c["frozen_state"] for c in cells}
    assert states <= {"impossible", "rule_selected_out", "expected"}, states
    n_exp = sum(1 for c in cells if c["frozen_state"] == "expected")
    assert n_exp > 0, "nothing to measure"
    # every non-expected cell carries a REASON — an unexplained terminal state is a gap in disguise
    for c in cells:
        if c["frozen_state"] != "expected":
            assert c["reason"], f"{c['cell']}: {c['frozen_state']} without a reason"
    return {"n_cells": len(cells), "n_units": len(units), "n_probes": len(probes),
            "n_expected": n_exp,
            "n_impossible": sum(1 for c in cells if c["frozen_state"] == "impossible"),
            "n_rule_out": sum(1 for c in cells if c["frozen_state"] == "rule_selected_out")}


@check("map/structural_impossibility")
def _map_impossible():
    """The known structural facts must land as `impossible`, and the ones that are NOT structural
    must not. Both directions — a rule that calls everything impossible would 'complete' the map
    trivially."""
    from infra.pg import plenum_map as PM
    idx = {c["cell"]: c for c in PM.all_cells()}
    must_be_impossible = [
        "gemma-2b|behavior/vqa/forced_choice",      # no vision tower
        "gemma-2b|weights/vision/bytes",
        "gemma-2b|weights/projector/bytes",
        "pg1|behavior/control/task_native",         # no action expert
        "ft-vqav2|weights/expert/bytes",
    ]
    must_be_possible = [
        "gemma-2b|behavior/text_recall/mcq",        # a text model HAS text faculties
        "gemma-2b|behavior/text_ppl/ppl",
        "ft-vqav2|behavior/grounding/forced_choice",  # the cross-survival cell — must be MEASURED
        "ft-refcoco-seg|behavior/vqa/forced_choice",
        "pi05|behavior/control/task_native",
        "pi0|weights/expert/bytes",
    ]
    for c in must_be_impossible:
        assert idx[c]["frozen_state"] == "impossible", f"{c} should be structurally impossible"
    for c in must_be_possible:
        assert idx[c]["frozen_state"] == "expected", \
            f"{c} must be MEASURED, not {idx[c]['frozen_state']} ({idx[c]['reason']})"
    return {"checked_impossible": len(must_be_impossible), "checked_possible": len(must_be_possible)}


@check("map/pending_is_real")
def _map_pending():
    """`pending=0` must be reachable AND must actually fire when a cell is missing or malformed."""
    from infra.pg import plenum_map as PM
    cells = PM.all_cells()
    expected = [c for c in cells if c["frozen_state"] == "expected"]
    full = {c["cell"]: {"verdict": "measured-ok", "scope": "dry-run", "pointer": "s3://x"}
            for c in expected}
    counts, pending = PM.render_counts(cells, full)
    assert counts["pending"] == 0, f"a complete registry must give pending=0, got {counts['pending']}"
    # drop one -> pending must be 1
    partial = dict(full)
    partial.pop(expected[0]["cell"])
    counts2, pending2 = PM.render_counts(cells, partial)
    assert counts2["pending"] == 1 and pending2 == [expected[0]["cell"]], \
        "a missing cell did not register as pending — the completion predicate cannot fail"
    # a record missing its pointer is NOT terminal (the no-snapshot rule has teeth)
    partial2 = dict(full)
    partial2[expected[0]["cell"]] = {"verdict": "x", "scope": "y"}
    assert PM.render_counts(cells, partial2)[0]["pending"] == 1, \
        "a record without an artifact pointer was accepted as measured"
    # a VOID with a named gate IS terminal; without one it is not
    v_ok = dict(full)
    v_ok[expected[0]["cell"]] = {"state": "VOID", "gate": "G_identity"}
    assert PM.render_counts(cells, v_ok)[0]["pending"] == 0, "a gated VOID must be terminal"
    return {"n_expected": len(expected)}


@check("map/verdicts_carry_no_numbers")
def _map_no_numbers():
    """§10: MAP cells are verdict+scope+pointer, never numbers. The renderer must REJECT a numeric
    verdict — this is the mechanical form of the no-snapshot rule."""
    import importlib
    R = importlib.import_module("scripts.plenum_map_render")
    R.check_records({"a|b": {"verdict": "consumed", "scope": "fc", "pointer": "s3://x"}})
    for bad in ({"a|b": {"verdict": "S=0.42", "scope": "fc", "pointer": "s3://x"}},
                {"a|b": {"verdict": "consumed", "scope": "fc"}},
                {"a|b": {"state": "VOID"}}):
        raised = False
        try:
            R.check_records(bad)
        except AssertionError:
            raised = True
        assert raised, f"renderer accepted a malformed record: {bad}"
    return {"rejected_forms": 3}


# ===========================================================================
# 2. Label rules — REPLAY the LACUNA known events (spec §9 step 2, §14 principle 13).
# ===========================================================================
@check("labels/replay_lacuna_known_events")
def _labels_replay():
    """The frozen v2 label rules must reproduce LACUNA's own adjudicated cases, including the
    S>ceiling case that the `exceeds_ceiling` disclosure exists for."""
    from infra.pg import lacuna_core as LC
    n = 400
    theta = 0.12
    out = {}
    # (a) a row AT the floor must emit `consumed` — the event AMENDMENT-5 was written for
    at_floor = LC.label_row({"S_point": 0.05, "S_lo95": 0.00, "S_hi95": 0.20,
                             "raw_excess_over_ceiling": -0.4}, theta)
    assert at_floor["label"] == "consumed", at_floor
    # (b) a row at the ceiling must emit `retained`
    at_ceil = LC.label_row({"S_point": 0.95, "S_lo95": 0.88, "S_hi95": 1.00,
                            "raw_excess_over_ceiling": -0.01}, theta)
    assert at_ceil["label"] == "retained", at_ceil
    # (c) a row inside the band but with a CI that cannot exclude retention is `partial`+underpowered
    weak = LC.label_row({"S_point": 0.10, "S_lo95": 0.00, "S_hi95": 0.72,
                         "raw_excess_over_ceiling": -0.2}, theta)
    assert weak["label"] == "partial" and weak["underpowered"], weak
    # (d) the S>CEILING case (LACUNA R5 VQAv2): S clamps at 1.0 and the excess must be DISCLOSED
    over = LC.label_row({"S_point": 1.0, "S_lo95": 0.90, "S_hi95": 1.0,
                         "raw_excess_over_ceiling": +0.08}, theta)
    assert over["label"] == "retained" and over["exceeds_ceiling"], over
    out["cases"] = {"at_floor": at_floor["label"], "at_ceiling": at_ceil["label"],
                    "underpowered": weak["label"], "exceeds_ceiling": over["exceeds_ceiling"]}
    # (e) the null band must SATURATE when the floor and ceiling are indistinguishable — the VOID
    # branch that keeps `consumed` from becoming unfailable
    import numpy as np
    rng = np.random.default_rng(0)
    same = rng.random(n)
    sat = LC.theta_null(same, same.copy(), n_boot=500)
    assert sat["saturated"], "a degenerate floor/ceiling pair did not saturate the null band"
    good = LC.theta_null(rng.random(n) * 0.2, rng.random(n) * 0.2 + 0.8, n_boot=500)
    assert not good["saturated"], "a healthy band was wrongly reported saturated"
    out["theta_null"] = {"degenerate_saturates": sat["saturated"],
                         "healthy_theta": round(good["theta_null"], 4)}
    return out


@check("labels/power_gate_both_directions")
def _power_gate():
    import numpy as np
    from infra.pg import lacuna_core as LC
    rng = np.random.default_rng(1)
    lo = (rng.random(500) < 0.25).astype(float)
    hi = (rng.random(500) < 0.80).astype(float)
    assert LC.power_gate(lo, hi)["passes"], "a wide floor→ceiling gap failed the power gate"
    assert not LC.power_gate(lo, lo.copy())["passes"], "a zero gap passed the power gate"
    inv = LC.power_gate(hi, lo)                     # floor ABOVE ceiling = instrument bug
    assert inv["inverted"] and not inv["passes"], inv
    return {"ok": True}


# ===========================================================================
# 3. The law — planted-predictor recovery AND random-predictor rejection.
# ===========================================================================
def _synth_law(n_units=26, n_fac=6, effect=1.0, seed=0):
    """A synthetic grid where survival IS recipe membership plus noise."""
    import numpy as np
    rng = np.random.default_rng(seed)
    facs = [f"f{i}" for i in range(n_fac)]
    rows = []
    for u in range(n_units):
        own = {facs[u % n_fac]}
        dw = float(rng.random())
        rows.append({
            "unit": f"u{u}", "dw_norm": dw, "recipe_faculties": own,
            "similarity": {f: (1.0 if f in own else float(rng.random() * 0.5)) for f in facs},
            "S": {f: float(np.clip(effect * (1.0 if f in own else 0.0)
                                   + 0.15 * rng.standard_normal() + 0.2 * dw, 0, 1))
                  for f in facs},
        })
    return rows, facs


@check("law/planted_predictor_recovered")
def _law_planted():
    from infra.pg import plenum_core as PC
    import numpy as np
    rows, facs = _synth_law(effect=1.0, seed=3)
    mem, dw, sim, y, grp, info = PC.predictor_matrix(rows, facs)
    assert info["n_rows"] == 26 * 6, info
    d = PC.delta_loo_grouped(np.column_stack([dw]), mem, y, grp, n_perm=400, seed=5)
    assert d["delta"] > 0 and d["p_perm"] < 0.05, \
        f"the law fit could not recover a PLANTED membership effect: {d}"
    ci = PC.boot_delta_loo(np.column_stack([dw]), mem, y, grp, n_boot=200, seed=6)
    assert ci["lo95"] > 0, f"planted effect CI does not exclude 0: {ci}"
    return {"delta": round(d["delta"], 4), "p_perm": d["p_perm"], "ci_lo95": round(ci["lo95"], 4)}


@check("law/random_predictor_rejected")
def _law_null():
    """The instrument-null: with NO membership effect, ΔLOO must not fire. This is the check that
    makes a positive law result meaningful — without it, `ΔLOO>0` could not fail."""
    from infra.pg import plenum_core as PC
    import numpy as np
    rows, facs = _synth_law(effect=0.0, seed=11)
    mem, dw, sim, y, grp, _ = PC.predictor_matrix(rows, facs)
    d = PC.delta_loo_grouped(np.column_stack([dw]), mem, y, grp, n_perm=400, seed=7)
    assert d["p_perm"] > 0.05, f"a null membership predictor FIRED (p={d['p_perm']}) — false positive"
    # and a pure-noise predictor with the real grid must also not fire
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(y.size)
    d2 = PC.delta_loo_grouped(np.column_stack([dw]), noise, y, grp, n_perm=400, seed=8)
    assert d2["p_perm"] > 0.05, f"a pure-noise predictor FIRED (p={d2['p_perm']})"
    return {"p_null_membership": d["p_perm"], "p_noise": d2["p_perm"]}


@check("law/grouped_loo_beats_row_loo_leakage")
def _law_grouping():
    """Row-wise LOO leaks a unit's other faculties into its own prediction. The grouped fold must
    give a STRICTLY more conservative R² on a grid built from unit-level means alone — if the two
    agree, the grouping is not doing anything and the law's ΔLOO is not what §2.1 says it is."""
    from infra.pg import plenum_core as PC
    from infra.pg import savb
    import numpy as np
    rng = np.random.default_rng(4)
    facs = [f"f{i}" for i in range(6)]
    rows = []
    for u in range(26):
        base = float(rng.standard_normal())          # a pure UNIT effect, unpredictable from X
        rows.append({"unit": f"u{u}", "dw_norm": float(rng.random()),
                     "recipe_faculties": set(),
                     "similarity": {f: float(rng.random()) for f in facs},
                     "S": {f: base + 0.01 * rng.standard_normal() for f in facs}})
    mem, dw, sim, y, grp, _ = PC.predictor_matrix(rows, facs)
    X = np.column_stack([dw, sim])
    r2_grouped, _ = PC.loo_r2_grouped(X, y, grp)
    r2_row = savb.loo_r2(X, y)
    assert r2_grouped < r2_row, \
        f"grouped LOO ({r2_grouped:.3f}) is not more conservative than row LOO ({r2_row:.3f})"
    return {"r2_grouped": round(r2_grouped, 4), "r2_row_loo": round(r2_row, 4)}


@check("law/mde_classifier_splits_three_hypotheses")
def _law_mde():
    from infra.pg import plenum_core as PC
    anti = PC.mde_pattern({("ft-vqav2", "vqa"): 0.9, ("ft-vqav2", "grounding"): 0.1,
                           ("ft-refcoco-seg", "vqa"): 0.1, ("ft-refcoco-seg", "grounding"): 0.9})
    same = PC.mde_pattern({("ft-vqav2", "vqa"): 0.9, ("ft-vqav2", "grounding"): 0.2,
                           ("ft-refcoco-seg", "vqa"): 0.8, ("ft-refcoco-seg", "grounding"): 0.3})
    unif = PC.mde_pattern({("ft-vqav2", "vqa"): 0.6, ("ft-vqav2", "grounding"): 0.62,
                           ("ft-refcoco-seg", "vqa"): 0.55, ("ft-refcoco-seg", "grounding"): 0.5})
    assert anti["supports"] == "H_recipe", anti
    assert same["supports"] == "H_fragility", same
    assert unif["supports"] == "H_magnitude", unif
    return {"anti_diagonal": anti["pattern"], "same_order": same["pattern"],
            "uniform": unif["pattern"]}


# ===========================================================================
# 4. The film's order statistic — including the UNDEFINED branch.
# ===========================================================================
@check("film/onset_statistic")
def _film_onset():
    from infra.pg import plenum_core as PC
    steps = [0, 500, 1000, 1500, 2000]
    # consumption first: faculty collapses at 500, control acquired at 1500
    r = PC.onset_statistic(steps,
                           {"vqa": [0.8, 0.3, 0.1, 0.05, 0.02]},
                           [0.0, 0.0, 0.1, 0.5, 0.9])
    assert r["per_faculty"]["vqa"]["reading"] == "consume_first", r
    # install first: control acquired early, faculty holds then falls late
    r2 = PC.onset_statistic(steps,
                            {"vqa": [0.8, 0.8, 0.75, 0.7, 0.2]},
                            [0.0, 0.6, 0.8, 0.9, 1.0])
    assert r2["per_faculty"]["vqa"]["reading"] == "install_first", r2
    # never collapses -> UNDEFINED, and an undefined term contributes no sign
    r3 = PC.onset_statistic(steps, {"vqa": [0.8] * 5}, [0.0, 0.1, 0.5, 0.9, 1.0])
    assert r3["per_faculty"]["vqa"]["delta"] is None, r3
    assert r3["per_faculty"]["vqa"]["reading"] == "undefined" and r3["n_undefined"] == 1, r3
    # control never acquired -> also undefined (the OTHER term failing)
    r4 = PC.onset_statistic(steps, {"vqa": [0.8, 0.1, 0.1, 0.1, 0.1]}, [0.0] * 5)
    assert r4["per_faculty"]["vqa"]["delta"] is None, r4
    return {"consume_first": r["per_faculty"]["vqa"]["delta"],
            "install_first": r2["per_faculty"]["vqa"]["delta"],
            "undefined_paths": 2}


# ===========================================================================
# 5. SAE gates — each must FAIL on the input it exists to catch.
# ===========================================================================
@check("sae/gates_fail_when_they_should")
def _sae_gates():
    from infra.pg import plenum_core as PC
    ok = PC.sae_gates(dead_frac=0.1, same_seed_cos=0.85, cross_stream_cos=0.40,
                      behav_native=0.60, behav_patched=0.57)
    assert ok["all_pass"], ok
    dead = PC.sae_gates(0.9, 0.85, 0.40, 0.60, 0.57)
    assert not dead["all_pass"] and dead["void_gate"] == "G_dead", dead
    # the identity gate's real failure mode: seeds that match EVERY stream equally (no
    # discrimination). The margin here is 0.01 — pure sampling noise. The pre-AMENDMENT gate PASSED
    # this input, which is why the amendment exists; it must now FAIL.
    ident = PC.sae_gates(0.1, 0.45, 0.44, 0.60, 0.57)
    assert not ident["all_pass"] and ident["void_gate"] == "G_identity", ident
    assert ident["G_identity"]["fails_on"] == "no_discrimination", ident
    # the OTHER identity failure: a featurizer that does not reproduce itself across seeds, even
    # though it "beats" a still-lower cross-stream match
    selfd = PC.sae_gates(0.1, 0.30, 0.05, 0.60, 0.57)
    assert not selfd["all_pass"] and \
        selfd["G_identity"]["fails_on"] == "seeds_disagree_with_themselves", selfd
    behav = PC.sae_gates(0.1, 0.85, 0.40, 0.60, 0.20)
    assert not behav["all_pass"] and behav["void_gate"] == "G_behavior", behav
    ef = PC.emergent_frac([0.9, 0.85, 0.2, 0.1], identity_floor=0.5)
    assert abs(ef["emergent_frac"] - 0.5) < 1e-9, ef
    return {"pass": ok["all_pass"], "void_gates_fired": ["G_dead", "G_identity", "G_behavior"]}


# ===========================================================================
# 6. Forced choice — the construction, on synthetic rows (no images, no model).
# ===========================================================================
@check("fc/choice_construction")
def _fc_choices():
    from infra.pg import plenum_core as PC
    rows = [{"answers": [{"answer": a} for a in ([f"ans{i%7}"] * 3 + ["other"])]}
            for i in range(60)]
    ch = PC.choices_for_task("vqav2", rows, seed=1)
    assert len(ch) == 60
    for (cands, gold), r in zip(ch, rows):
        assert len(cands) == PC.N_CAND and len(set(cands)) == PC.N_CAND, cands
        banned = {a["answer"].strip().lower() for a in r["answers"]}
        wrong = [c for i, c in enumerate(cands) if i != gold]
        assert not (set(wrong) & banned), \
            f"a distractor is an annotator answer — chance is no longer the floor: {wrong}"
    return {"n_items": len(ch), "chance": PC.CHANCE}


@check("fc/refcoco_geometric_exclusion")
def _fc_refcoco():
    """RefCOCO's 'already correct' test is IoU, not string equality. A distractor box overlapping
    the gold above the threshold would be scored correct by the task's own scorer."""
    from infra.pg import battery_tasks as BT
    from infra.pg import plenum_core as PC
    assert BT._iou((0, 0, .1, .1), (0, 0, .1, .1)) > 0.99
    assert BT._iou((0, 0, .1, .1), (.5, .5, .6, .6)) == 0.0
    # the task's own scorer counts IoU ≥ 0.5 as CORRECT, so the distractor cap must sit below it
    assert PC.REFCOCO_MAX_DISTRACTOR_IOU < 0.5, "a distractor could be scored correct"

    class _Img:                       # `_refcoco_gt_box` needs only `.size`
        size = (100, 100)

    # a RefCOCO-shaped population of well-separated 20×20 boxes on a 100×100 image
    rows = [{"image": _Img(), "bbox": [(i % 5) * 20, (i // 5) * 20, 18, 18], "answer": "the thing"}
            for i in range(25)]
    ch = PC.choices_for_task("refcoco", rows, seed=2)
    assert len(ch) == 25
    gt = [BT._refcoco_gt_box(r) for r in rows]
    for i, (cands, gold) in enumerate(ch):
        assert len(cands) == PC.N_CAND and len(set(cands)) == PC.N_CAND, cands
        assert cands[gold] == BT._refcoco_target(rows[i])
        for k, c in enumerate(cands):
            if k == gold:
                continue
            box = BT._decode_loc_box(c)
            assert box is not None, f"distractor is not a decodable loc string: {c}"
            iou = BT._iou(box, gt[i])
            assert iou <= PC.REFCOCO_MAX_DISTRACTOR_IOU + 1e-6, \
                f"item {i}: distractor IoU {iou:.3f} would be scored CORRECT — chance is not the floor"
    return {"n_items": len(ch), "max_iou_allowed": PC.REFCOCO_MAX_DISTRACTOR_IOU}


# ===========================================================================
# 7. Prediction scoring + BH families.
# ===========================================================================
@check("predictions/scoring_table")
def _pred_score():
    from infra.pg import plenum_core as PC
    sealed = {"a": "consumed", "b": "retained", "c": "partial"}
    obs = {"a": "consumed", "b": "partial"}
    r = PC.score_predictions(sealed, obs)
    assert r["hit"] == 1 and r["miss"] == 1 and r["n_unmeasured"] == 1, r
    assert abs(r["hit_rate"] - 0.5) < 1e-9, r
    return {"hit": r["hit"], "miss": r["miss"], "unmeasured": r["n_unmeasured"]}


@check("bh/families_exclude_void_and_disclose_m")
def _bh():
    from infra.pg import lacuna_core as LC
    fam = {"t1": 0.001, "t2": 0.30, "t3": None, "t4": 0.02}
    r = LC.bh_reject(fam, q=0.05)
    assert r["m"] == 3 and r["void"] == ["t3"], r
    assert "t1" in r["rejected"], r
    none_r = LC.bh_reject({"t1": None}, q=0.05)
    assert none_r["m"] == 0 and none_r["rejected"] == [], none_r
    return {"m": r["m"], "void": r["void"], "rejected": r["rejected"]}


# ===========================================================================
# 8. DATA STAGE — real datasets + the real processor, everything except the model forward.
#    `feedback_offline_gate_before_gpu`: this is the half that needs the network, so it runs as a
#    CPU pg job. A schema assumption that is wrong here would otherwise surface mid-GPU-run.
# ===========================================================================
DATA_CHECKS = []


def data_check(name):
    def deco(fn):
        DATA_CHECKS.append((name, fn))
        return fn
    return deco


@data_check("data/forced_choice_on_real_rows")
def _data_fc(n):
    """Build the forced choice on REAL rows of all four tasks. This is where a wrong row-schema
    assumption dies cheaply instead of 3 hours into a GPU run."""
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    tasks = LS.mm_tasks()
    out = {}
    for name in ("vqav2", "textvqa", "refcoco", "coco_captions"):
        rows, hashes = IP.load_items(tasks[name], n)
        assert len(set(hashes)) == len(hashes), f"{name}: duplicate content hashes"
        ch = PC.choices_for_task(name, rows, seed=20260728)
        assert len(ch) == len(rows), f"{name}: choice count mismatch"
        for i, (cands, gold) in enumerate(ch):
            assert len(cands) == PC.N_CAND and len(set(cands)) == PC.N_CAND, \
                f"{name}[{i}]: degenerate candidate set {cands}"
            g, banned = PC.gold_and_banned(name, rows[i])
            assert cands[gold] == g, f"{name}[{i}]: gold index does not point at the gold"
            wrong = {c for k, c in enumerate(cands) if k != gold}
            assert not (wrong & set(banned)), \
                f"{name}[{i}]: a distractor is a correct answer — chance is not the floor"
        out[name] = {"n": len(rows), "example_cands": ch[0][0], "gold_idx": ch[0][1]}
        print(f"[G0-data] {name:>14s} n={len(rows)} ok", flush=True)
    return out


@data_check("data/candidate_framing_through_the_real_processor")
def _data_framing(n):
    """`score_item` asserts every candidate shares one [prompt | continuation] framing.

    AMENDED 2026-07-28 after the MDE crashed on RefCOCO with `lcp=261 lcs=0`. The first version of
    this check tokenized TEXT ONLY (`tok(prompt)`), which is not the path `score_item` takes: the
    real path is `proc(text=…, images=…)`, which prepends 256 image tokens + BOS and appends a
    trailing '\\n' (`reference_paligemma_text_trailing_newline`). A text-only check therefore passed
    a construction that the real processor rejects — the gate was measuring the wrong object.

    This version drives the ACTUAL processor with a real image, so it fails on the control plane
    for the same reason the GPU job would. No model is loaded; the processor is enough.
    """
    from PIL import Image
    import numpy as np
    from transformers import AutoProcessor
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    from infra.pg.text_eval import _common_prefix_len, _common_suffix_len
    proc = AutoProcessor.from_pretrained(LS.PG1_REPO)
    proc.tokenizer.padding_side = "left"
    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    tasks = LS.mm_tasks()
    out, bad = {}, []
    for name in ("vqav2", "textvqa", "refcoco", "coco_captions"):
        rows, _ = IP.load_items(tasks[name], n)
        ch = PC.choices_for_task(name, rows, seed=20260728)
        frames, n_ok = set(), 0
        for r, (cands, _g) in list(zip(rows, ch))[:6]:
            prompt = tasks[name].prompt_fn(r)
            pe = proc(text=[prompt], images=[img], return_tensors="pt", padding="longest")
            p_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
            enc = proc(text=[prompt + " " + c for c in cands], images=[img] * len(cands),
                       return_tensors="pt", padding="longest")
            real = enc["attention_mask"].sum(dim=1).tolist()
            for k, c in enumerate(cands):
                f_real = enc["input_ids"][k][enc["input_ids"].shape[1] - int(real[k]):].tolist()
                lcp = _common_prefix_len(p_real, f_real)
                lcs = _common_suffix_len(p_real, f_real)
                if lcp + lcs != len(p_real):
                    bad.append({"task": name, "cand": str(c)[:60], "lcp": lcp, "lcs": lcs,
                                "len_prompt": len(p_real), "len_full": len(f_real),
                                "prompt_tail": p_real[-4:], "full_tail": f_real[-4:],
                                "prompt": prompt[:60]})
                else:
                    frames.add((lcp, lcs))
                    n_ok += 1
        out[name] = {"n_ok": n_ok, "distinct_frames": len(frames)}
    assert not bad, ("the real processor breaks the [shared|sep] framing on "
                     f"{len(bad)} candidates — score_item would raise. First 3: {bad[:3]}")
    return out


@data_check("data/fc_scorer_matches_the_P1_verified_scorer")
def _data_scorer_equiv(n):
    """`score_item_fc` must extract the SAME continuation as the P1-verified `score_item` wherever
    the latter applies, or P1 stops being a check of the code actually in use.

    The continuation extraction is a pure tokenizer property — no model, no GPU. This check runs on
    the control plane and covers BOTH regimes:
      * VQAv2 / TextVQA (no trailing newline in the prompt): the two must agree token-for-token;
      * RefCOCO / COCO (prompt already ends in '\\n'): `score_item` RAISES (that is the bug that
        killed the MDE) while `score_item_fc` must still produce a non-empty continuation whose
        tokens are shared-prefix-free. A check that only covered the first regime would have
        blessed the broken v1 fix.
    """
    from PIL import Image
    import numpy as np
    from transformers import AutoProcessor
    from infra.pg import inherit_pg1 as IP
    from infra.pg import lacuna_shell as LS
    from infra.pg import plenum_core as PC
    from infra.pg.text_eval import _common_prefix_len, _common_suffix_len
    proc = AutoProcessor.from_pretrained(LS.PG1_REPO)
    proc.tokenizer.padding_side = "left"
    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    tasks = LS.mm_tasks()
    out = {}
    for name in ("vqav2", "textvqa", "refcoco", "coco_captions"):
        rows, _ = IP.load_items(tasks[name], n)
        ch = PC.choices_for_task(name, rows, seed=20260728)
        agree, legacy_raises, checked = 0, 0, 0
        for r, (cands, _g) in list(zip(rows, ch))[:6]:
            prompt = PC.fc_prompt(tasks[name].prompt_fn(r))
            pe = proc(text=[prompt], images=[img], return_tensors="pt", padding="longest")
            p_real = pe["input_ids"][0][int((pe["attention_mask"][0] == 0).sum()):].tolist()
            enc = proc(text=[prompt + " " + c for c in cands], images=[img] * len(cands),
                       return_tensors="pt", padding="longest")
            real = enc["attention_mask"].sum(dim=1).tolist()
            for k in range(len(cands)):
                f_real = enc["input_ids"][k][enc["input_ids"].shape[1] - int(real[k]):].tolist()
                lcp = _common_prefix_len(p_real, f_real)
                lcs = _common_suffix_len(p_real, f_real)
                new_cont = f_real[lcp:len(f_real) - 1]          # score_item_fc, trailing=1
                assert len(new_cont) >= 1, f"{name}: score_item_fc produced an empty continuation"
                if lcp + lcs == len(p_real):                    # the legacy scorer applies here
                    old_cont = f_real[lcp:len(f_real) - lcs]
                    assert old_cont == new_cont, (
                        f"{name}: the two scorers disagree — P1 would no longer check the code in "
                        f"use. old={old_cont[:6]} new={new_cont[:6]}")
                    agree += 1
                else:
                    legacy_raises += 1                          # RefCOCO / COCO regime
                checked += 1
        out[name] = {"checked": checked, "agree_with_legacy": agree,
                     "legacy_would_raise": legacy_raises}
        print(f"[G0-data] scorer {name:>14s} agree={agree} legacy_raises={legacy_raises}",
              flush=True)
    # both regimes must actually occur, or the check is not covering what it claims to
    assert any(v["agree_with_legacy"] for v in out.values()), "no task exercised the agreement path"
    assert any(v["legacy_would_raise"] for v in out.values()), \
        "no task exercised the newline-collision path — the check is not covering the MDE failure"
    return out


@data_check("data/roster_is_the_invariant_core_24")
def _data_roster(n):
    """The roster is re-derived from HF and must be exactly the frozen 24, with both MDE units in
    it. A roster that drifted is a different population."""
    from huggingface_hub import HfApi
    from infra.pg.plenum_map import FT_REPO, FT_SLUGS
    api = HfApi()
    live = sorted(m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                  if m.id.endswith("-224") and "-jax" not in m.id
                  and m.id.startswith("google/paligemma-3b-ft-"))
    want = sorted(FT_REPO.format(slug=s) for s in FT_SLUGS)
    assert live == want, (f"roster drift: {len(live)} live vs {len(want)} frozen; "
                          f"only-live={sorted(set(live)-set(want))[:4]} "
                          f"only-frozen={sorted(set(want)-set(live))[:4]}")
    for mde in ("vqav2", "refcoco-seg"):
        assert FT_REPO.format(slug=mde) in live, f"the MDE needs {mde} in the roster"
    shas = {r: api.model_info(r).sha for r in want}
    return {"n": len(want), "shas": shas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/plenum_offline")
    ap.add_argument("--stage", default="pure", choices=("pure", "data", "all"))
    ap.add_argument("--n", type=int, default=64, help="real items per task in the data stage")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "plenum_offline_g0_v1", "stage": a.stage, "checks": {}}
    if a.stage in ("pure", "all"):
        for name, fn in CHECKS:
            rep["checks"][name] = fn()
            print(f"[G0] {name:>52s}  OK  {json.dumps(rep['checks'][name], default=str)[:110]}",
                  flush=True)
    if a.stage in ("data", "all"):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from infra.pg.diff_harvest import hf_online_setup
        hf_online_setup()
        for name, fn in DATA_CHECKS:
            rep["checks"][name] = fn(a.n)
            print(f"[G0] {name:>52s}  OK  {json.dumps(rep['checks'][name], default=str)[:110]}",
                  flush=True)
    rep["n_checks"] = len(rep["checks"])
    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"PLENUM_OFFLINE_OK {rep['n_checks']} checks (stage={a.stage})", flush=True)


if __name__ == "__main__":
    main()
