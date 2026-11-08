"""PLENUM aggregation — banks → S → labels → the law → BH families → the MAP records registry.

Everything here is recomputed from banked PER-ITEM scores. No number is read from a job's log, a
prior report, or a summary (§0: verify from the artifact). The label rules are LACUNA's v2 objects
imported unchanged — this job does not own a decision rule, it applies frozen ones.

Outputs:
  metrics.json                 every gate, band, label, contrast and family, with CIs
  cells.json                   {map_cell_id: {verdict, scope, pointer}} for the renderer
  plenum_prediction_score.json the §11 row-3 deliverable

  python -m infra.aws_pg.jobs.plenum_aggregate --out <dir> --fc <fc_banks.json> \
      --census <census metrics.json> --predictions <plenum_predictions.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib

FLOOR_UNIT = "matched_dw"       # the achievable-damage FLOOR (§2.1)
CEIL_UNIT = "pg1"               # PG1-in-shell CEILING (§2.1)
FACULTY_OF_TASK = {"vqav2": "vqa", "textvqa": "ocr", "refcoco": "grounding",
                   "coco_captions": "caption"}


# Tolerance for two ALIGNED banks of the same cell disagreeing. Near-tie argmax flips under
# nondeterministic GPU reduction order are expected at the 0.1–2% level (measured); anything past
# these limits is two different measurements and stops the run.
DUP_MAX_ITEM_FRAC = 0.05
DUP_MAX_MEAN_SHIFT = 0.02


def _same_series(a, b):
    """Exact equality for per-item score lists, treating NaN as equal to NaN.

    `float("nan") == float("nan")` is False, so plain list equality reports two identical banks as
    different — which would turn a verified replication into a spurious contradiction.
    """
    if a is None or b is None or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y and not (isinstance(x, float) and isinstance(y, float)
                           and x != x and y != y):
            return False
    return True


def _load(p):
    p = str(p)
    if p.startswith("s3://"):
        import subprocess
        return json.loads(subprocess.run(["aws", "s3", "cp", p, "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def main():
    import os
    import sys
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc", nargs="+", default=[], help="forced-choice fc_banks.json shards")
    ap.add_argument("--gen", nargs="+", default=[],
                    help="generation fc_banks.json shards. Both readouts key banks the same way "
                         "({unit}|{task}|{cond}), so they are kept in SEPARATE namespaces here "
                         "rather than merged — mixing them would silently compare a greedy-decode "
                         "score against a 4-way ranking as though they were one quantity.")
    ap.add_argument("--install-deps", action="store_true",
                    help="pip install pycocoevalcap (CIDEr). Explicit rather than auto-detected: "
                         "the corpus path is the only part needing more than numpy, and the "
                         "control plane must never install it. An env-var guess (PG_JOB) silently "
                         "did nothing and cost a job.")
    ap.add_argument("--cider-boot", type=int, default=200,
                    help="corpus-metric bootstrap draws (each draw recomputes CIDEr)")
    ap.add_argument("--sae", nargs="+", default=None,
                    help="one or more plenum_sae metrics.json (the gemma-2b text stream is its own "
                         "run, so a single-file flag would silently drop a stream -- the same shape "
                         "of defect as passing only the FT census). Arm 2's gate verdicts exist but "
                         "were never "
                         "emitted as MAP records, leaving its feature cells PENDING over a "
                         "terminal VOID -- the same wiring gap as the census and the text axes.")
    ap.add_argument("--patch", default=None,
                    help="plenum_patch metrics.json. Same for Arm 4's P5 VOID.")
    ap.add_argument("--text", nargs="+", default=[],
                    help="plenum_text text_banks.json shards. The text axes have their own bank "
                         "shape, so without this the 170 behavior/text_* cells stay PENDING over "
                         "data that exists — the same wiring gap the census had.")
    ap.add_argument("--control", nargs="+", default=None,
                    help="plenum_control metrics.json — the §2.3 step-0 zero-shot control cells")
    ap.add_argument("--control-pointer",
                    default="s3://pg-lineage-769063704696/results/plenum_ctl")
    ap.add_argument("--census", nargs="+", default=None,
                    help="one or more census metrics.json (C6: FT census alone drops pi0/pi05)")
    ap.add_argument("--sae-pointer",
                    default="s3://pg-lineage-769063704696/results/plenum_sae")
    ap.add_argument("--patch-pointer",
                    default="s3://pg-lineage-769063704696/results/plenum_patch")
    ap.add_argument("--text-pointer",
                    default="s3://pg-lineage-769063704696/results/plenum_text_1")
    ap.add_argument("--census-pointer",
                    default="s3://pg-lineage-769063704696/results/plenum_census3")
    ap.add_argument("--predictions", default=None)
    ap.add_argument("--exclude-faculty", default="",
                    help="comma faculties dropped from the LAW FIT only (cells stay "
                         "measured). C2: the decisive test of the claimed dissociation "
                         "is the routed fit with `grounding` excluded — its generation "
                         "FLOOR is exactly 0.0 and only one unit emits the format, so "
                         "it may carry the entire effect.")
    ap.add_argument("--half", default="confirmation",
                    choices=("discovery", "confirmation", "full"))
    ap.add_argument("--pointer", default="s3://pg-lineage-769063704696/results/plenum_fc")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.install_deps:
        from infra.aws_pg.jobs._inherit_shared import install_pg
        install_pg()

    import numpy as np
    from infra.pg import lacuna_core as LC
    from infra.pg import plenum_core as PC
    from infra.pg import plenum_map as PM

    rep = {}
    # unit -> the shard files that supplied its rows. Provenance the bank files do not carry: a
    # bank is {cell_key: scores} with no record of WHICH weights produced it, so any claim about a
    # unit's weights (contamination, re-measurement, supersession) has to be resolved from the
    # shard's sibling metrics.json. Written by load_banks, read by the u3c contamination block.
    sources = {}

    def load_banks(files):
        """(banks, hashes, gold) for ONE readout namespace."""
        banks, hashes, gold = {}, {}, {}
        overrides = []
        for f in files:
            b = _load(f)
            for k, v in b.items():
                if k == "item_hashes":
                    # Shards are only poolable if they drew the SAME items in the SAME order. The
                    # frozen sample + choice seed should guarantee it; asserting it means a shard
                    # that silently drew a different sample cannot be merged into one grid as
                    # though it were the same population.
                    for task, hs in v.items():
                        if task in hashes:
                            assert hashes[task] == hs, (
                                f"{f}: shard item sample for {task!r} differs from an earlier "
                                f"shard (n={len(hs)} vs {len(hashes[task])}) — not poolable")
                        else:
                            hashes[task] = hs
                elif k == "gold_idx":
                    for task, g in v.items():
                        if task in gold:
                            assert gold[task] == g, f"{f}: gold indices differ for {task!r}"
                        else:
                            gold[task] = g
                else:
                    sources.setdefault(k.split("|")[0], set()).add(str(f))
                    if k in banks:
                        # A key can legitimately appear twice when a task was RE-RUN to repair a
                        # defect (the caption re-runs carry their own band rows alongside the
                        # original shards' misaligned ones). Resolve by POSITIONAL ALIGNMENT, which
                        # is checkable, rather than by file order, which is not: the bank whose
                        # length matches the frozen item count is the valid one. If both or neither
                        # align, the collision is unexplained and must fail loud.
                        task = k.split("|")[1]
                        n_h = len(hashes.get(task, []))
                        # CORPUS ROWS have no `per_item` at all — COCO under the generation readout
                        # is scored by corpus CIDEr and banks {cand, ref} dicts instead. Measuring
                        # their alignment by `per_item` length made BOTH banks look misaligned
                        # (0 != 2000) and the guard refused a collision it should have resolved. The
                        # analogue of "positionally aligned" for a corpus row is "has the frozen
                        # number of candidates".
                        def _n_scored(bank):
                            if bank.get("corpus"):
                                return len(bank.get("cand") or {})
                            return len(bank.get("per_item") or [])
                        old_ok = _n_scored(banks[k]) == n_h
                        new_ok = _n_scored(v) == n_h
                        if old_ok and new_ok:
                            # BOTH aligned. The original guard called this "unexplained" and failed
                            # — but it is the most benign case there is, and it fired on precisely
                            # the cells where two runs agree PERFECTLY. Several units were measured
                            # twice (a shard plus a per-unit re-run launched when the shard was
                            # believed reaped), so identical duplicates are expected and are
                            # evidence, not corruption.
                            #
                            # Proving they are the same measurement is not guessing. Only an
                            # aligned duplicate that DISAGREES is unexplained, and that still fails
                            # loud — it would mean one cell has two different values.
                            if banks[k].get("corpus") or v.get("corpus"):
                                # Identity for a corpus row is its candidate/reference text, not a
                                # score vector.
                                if ((banks[k].get("cand") or {}) == (v.get("cand") or {})
                                        and (banks[k].get("ref") or {}) == (v.get("ref") or {})):
                                    rep.setdefault("verified_replications", []).append(
                                        {"key": k, "n_items": n_h, "corpus": True})
                                    continue
                                rep.setdefault("duplicate_disagreements", []).append(
                                    {"key": k, "n_items": n_h, "corpus": True,
                                     "kept": "later-listed (repair pass)",
                                     "note": "corpus candidates differ; generation is sampled text "
                                             "so exact equality is not expected"})
                                banks[k] = v
                                continue
                            if _same_series(banks[k].get("per_item"), v.get("per_item")):
                                rep.setdefault("verified_replications", []).append(
                                    {"key": k, "n_items": n_h})
                                continue
                            # ALIGNED BUT DIFFERENT. This is real: the forced-choice battery is
                            # NOT bit-deterministic across runs. A small fraction of items sit close
                            # enough to a tie that nondeterministic GPU reduction order flips the
                            # argmax, and the flips are 0<->1 on binary correctness. Measured on
                            # ft-vqav2: 2/4000 on vqav2-real, 11/4000 vqav2-gray, 10/2000
                            # textvqa-real, 36/1975 refcoco-gray. (ft-tallyqa and pi0 happened to
                            # have no flips at all, which is why an earlier reading of this called
                            # the battery deterministic — it is not.)
                            #
                            # The resolution is a documented PRECEDENCE, not a coin flip: keep the
                            # later-listed bank, because the launcher lists the original shards
                            # before the per-unit repair runs and the repair pass is canonical. The
                            # disagreement is recorded per cell so it is visible in the artifact
                            # rather than absorbed.
                            #
                            # And the tolerance CAN FAIL. Float nondeterminism at near-ties is
                            # small; a large divergence would mean two genuinely different
                            # measurements (wrong items, wrong unit, wrong prompt) and must stop the
                            # run rather than be logged as noise.
                            _a = banks[k].get("per_item") or []
                            _b = v.get("per_item") or []
                            _nd = sum(1 for x, y in zip(_a, _b)
                                      if x != y and not (isinstance(x, float) and isinstance(y, float)
                                                         and x != x and y != y))
                            _fa = [x for x in _a if x == x]
                            _fb = [x for x in _b if x == x]
                            _ma = (sum(_fa) / len(_fa)) if _fa else float("nan")
                            _mb = (sum(_fb) / len(_fb)) if _fb else float("nan")
                            _frac = _nd / max(len(_a), 1)
                            _shift = abs(_ma - _mb)
                            assert _frac <= DUP_MAX_ITEM_FRAC and _shift <= DUP_MAX_MEAN_SHIFT, (
                                f"duplicate bank key {k}: two aligned banks disagree far beyond "
                                f"near-tie nondeterminism ({_nd}/{len(_a)} items = {_frac:.3%}, "
                                f"mean shift {_shift:.4f}; limits {DUP_MAX_ITEM_FRAC:.1%} and "
                                f"{DUP_MAX_MEAN_SHIFT}). These are not the same measurement.")
                            rep.setdefault("duplicate_disagreements", []).append(
                                {"key": k, "n_items": n_h, "n_differing": _nd,
                                 "frac_differing": _frac, "mean_existing": _ma, "mean_kept": _mb,
                                 "mean_shift": _shift, "kept": "later-listed (repair pass)"})
                            banks[k] = v
                            continue
                        assert old_ok != new_ok, (
                            f"duplicate bank key {k} and alignment does not disambiguate "
                            f"(old_aligned={old_ok}, new_aligned={new_ok}) — refusing to guess")
                        if new_ok:
                            banks[k] = v
                        overrides.append({"key": k, "kept": "new" if new_ok else "existing",
                                          "n_items": n_h})
                    else:
                        banks[k] = v
        if overrides:
            rep.setdefault("bank_overrides", []).extend(overrides)
        return banks, hashes, gold

    def completeness(banks, tag):
        """Which units have a FULL (task × condition) set, and which are partial.

        `exit=0` does NOT mean a shard finished: a job reaped at its budget deadline shuts down
        cleanly and stamps a zero exit code, leaving correct-but-partial banks (observed:
        plenum_fcg_d, 1 of 6 units, exit=0). Completion must therefore be judged from the DATA.
        A unit missing cells that its shard-mates have is partial, and a partial unit is not
        silently folded into the grid — it is reported and excluded, because a grid quietly
        containing 1 unit where 6 were requested is exactly the silent truncation this program is
        built to make impossible.
        """
        seen = {}
        for k in banks:
            u, task, cond = k.split("|")
            seen.setdefault(u, set()).add((task, cond))
        full = max((len(v) for v in seen.values()), default=0)
        partial = {u: sorted(v) for u, v in seen.items() if len(v) < full}
        if partial:
            rep.setdefault("partial_units", {})[tag] = {
                u: {"n_cells": len(v), "expected": full} for u, v in partial.items()}
        return set(partial)

    FC_B, FC_H, _g1 = load_banks(a.fc)
    GEN_B, GEN_H, _g2 = load_banks(a.gen)
    # CIDEr needs pycocoevalcap, which is not in the AMI. `plenum_fc` installs it for the generation
    # readout; the AGGREGATOR never did, because until the corpus-row alignment fix above, the coco
    # corpus rows never reached the CIDEr path from here at all. Installed here rather than at import
    # so a forced-choice-only aggregation does not pay for it.
    if any(isinstance(v, dict) and v.get("corpus") for v in GEN_B.values()):
        from infra.aws_pg.jobs._inherit_shared import install_pg
        install_pg()
        rep["cider_deps"] = "pycocoevalcap installed for the corpus CIDEr path"
    assert FC_B or GEN_B, "no banks supplied"
    hashes = dict(GEN_H); hashes.update(FC_H)
    for t in set(FC_H) & set(GEN_H):
        assert FC_H[t] == GEN_H[t], (
            f"{t}: the forced-choice and generation runs used DIFFERENT items — their cells are "
            f"not two readouts of one population and must not share a band")
    # The aggregation IS the analysis, and an aggregation whose inputs are not recorded cannot be
    # reproduced or audited. `agg_z` did not record its own argv, so establishing which bank shards
    # it had been given meant listing S3 and inferring. Never again.
    rep.update({"argv": list(sys.argv[1:]),
                "object": "plenum_aggregate_v1", "half": a.half,
           "n_banks": {"forced_choice": len(FC_B), "generation": len(GEN_B)},
           "readout_policy": ("forced choice is PRIMARY (§2.1); a faculty whose forced-choice band "
                              "fails the N1 power gate is read on generation instead, and its "
                              "forced-choice cells terminate as VOID/N1_power_gate")})
    cells_out, records = [], {}

    def _put(unit, probe, rec):
        """Write a MAP record, unless `unit` is a film checkpoint.

        A checkpoint is a point on a trajectory, not a roster unit. Without this guard the film's
        rows would mint ids like `ckpt:s3://...|behavior/vqa/forced_choice` and the map would report
        cells for units it does not contain.
        """
        if str(unit).startswith("ckpt:"):
            return
        records[PM.cell_id(unit, probe)] = rec

    def _save(stage):
        """Write the artifacts NOW, after every expensive stage.

        This job wrote metrics.json and cells.json only at the very end. The corpus-CIDEr stage runs
        200 bootstrap resamples over every unit with caption rows and takes hours, so a budget reap
        mid-CIDEr destroyed the entire run's output — including the forced-choice results that had
        already been computed. That is `memory/feedback_fleet_incremental_save.md` applied to the one
        job where it costs the most.

        `stage` is recorded so a partial artifact says WHICH stages completed, rather than looking
        like a finished aggregation with cells missing.
        """
        rep.setdefault("stages_completed", []).append(stage)
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
        (out / "cells.json").write_text(json.dumps(records, indent=2))
        print(f"[AGG] saved after stage: {stage} (cells={len(records)})", flush=True)
    misaligned = set()

    # ---- discovery / confirmation split on the FULL content hashes (§2.1, REUSED) -----------
    masks = {}
    for task, h in hashes.items():
        disc, conf = LC.parity_split(h)
        masks[task] = np.asarray(LC.mask_of(a.half, disc, conf), dtype=bool)
        rep.setdefault("split", {})[task] = {"n_total": len(h), "n_discovery": int(sum(disc)),
                                             "n_confirmation": int(sum(conf)),
                                             "n_used": int(masks[task].sum())}

    def items(banks, unit, task, cond):
        """Per-item scores for one cell, restricted to the analysis half. Corpus tasks (COCO
        CIDEr under generation) have no per-item score, so they return None and are handled by
        the corpus path rather than being silently coerced to zeros."""
        k = f"{unit}|{task}|{cond}"
        if k not in banks or banks[k].get("corpus"):
            return None
        v = np.asarray(banks[k]["per_item"], dtype=float)
        m = masks[task]
        if v.size != m.size:
            # A bank written before unscorable items were recorded IN PLACE is shorter than the
            # frozen item order, so its positions no longer correspond to the hashes and no paired
            # contrast built from it is valid. That is not a crash and it is not something to
            # silently truncate past: the cell terminates as VOID under a NAMED gate, and the other
            # tasks in the same shard — whose banks ARE aligned — are analysed normally.
            misaligned.add((task, k))
            return None
        return v[m]      # may contain NaN for unscorable items; the caller masks jointly

    # ---- per-cell S, labels, Δ_perception — ONCE PER READOUT ------------------------------
    # The two readouts get their OWN band. A generation cell must never be scored against a
    # forced-choice FLOOR/CEILING: they are different quantities on different scales, and sharing
    # a band would manufacture an S out of a unit conversion.
    bands, power = {}, {}

    def run_readout(banks, readout):
        if not banks:
            return
        partial = completeness(banks, readout)
        # Band rows are exempt from the partial check only if they are themselves complete; a
        # partial FLOOR or CEILING would corrupt every cell scored against it, so it is fatal.
        for bu in (FLOOR_UNIT, CEIL_UNIT):
            assert bu not in partial, (
                f"{readout}: the {bu} band row is PARTIAL — every S scored against it would be "
                f"computed from a different item population than the rows it scores")
        tasks = sorted({k.split("|")[1] for k in banks})
        units = sorted({k.split("|")[0] for k in banks})
        # THE FILM. §2.3 runs "the full Tier-1 battery at EVERY checkpoint", and a checkpoint enters
        # the battery as `ckpt:<s3-uri>` through the one loader — so its rows arrive here mixed in
        # with the roster's. They are points on a trajectory, not roster units: emitting map cells
        # for them would mint ids like `ckpt:s3://...|behavior/vqa/forced_choice`. Split them out
        # once, here, so both the CIDEr path and the S loop below inherit the guard.
        film_units = [u for u in units if u.startswith("ckpt:")]
        if film_units:
            rep.setdefault("film", {})["checkpoint_rows"] = film_units
        # Checkpoint rows STAY in `units`: they must flow through the same S computation the grid
        # rows use, because that identity is the film's whole validity claim. What they must not do
        # is mint map cells — so the guard lives on the record write, not on this loop.
        for task in tasks:
            fl = items(banks, FLOOR_UNIT, task, "real")
            ce = items(banks, CEIL_UNIT, task, "real")
            key = f"{readout}|{task}"
            if fl is None or ce is None:
                rep.setdefault("missing_band", []).append(key)
                continue
            # GATES FIRST (§6), and for a load-bearing reason: `theta_null` ASSERTS on a degenerate
            # band ("CEILING ≤ FLOOR in most draws"). On a readout with no power — grounding under
            # forced choice, where PG1 sits at chance — that assert would kill the entire
            # aggregation instead of voiding one faculty. The power gate is computed from
            # `boot_diff_ci`, which has no such precondition, so it can always be asked.
            okb = np.isfinite(fl) & np.isfinite(ce)
            fl, ce = fl[okb], ce[okb]
            power[key] = LC.power_gate(fl, ce)
            if not power[key]["passes"]:
                bands[key] = {"theta_null": None, "saturated": False,
                              "note": "band NOT computed — the readout failed the N1 power gate; "
                                      "θ_null is undefined when CEILING ≤ FLOOR"}
                continue
            bands[key] = LC.theta_null(fl, ce)
        # ---- CORPUS tasks (COCO captions under generation): CIDEr has no per-item score, so the
        # per-item path above cannot reach these cells and they would stay PENDING forever —
        # silently blocking `pending=0` on a faculty the law needs. REUSED verbatim:
        # `lacuna_aggregate.cider_S_multi`, the item-paired corpus bootstrap that computes the
        # floor and ceiling ONCE per draw so the rows are genuinely paired.
        for task in tasks:
            fk = f"{FLOOR_UNIT}|{task}|real"
            ck = f"{CEIL_UNIT}|{task}|real"
            if not (banks.get(fk, {}).get("corpus") and banks.get(ck, {}).get("corpus")):
                continue
            from infra.aws_pg.jobs.lacuna_aggregate import cider_S_multi
            m = masks[task]
            all_keys = sorted(banks[ck]["cand"], key=lambda x: int(x))
            keys = [k for k, keep in zip(all_keys, m) if keep]
            cfgs = {u: (banks[f"{u}|{task}|real"]["cand"], banks[f"{u}|{task}|real"]["ref"])
                    for u in units if banks.get(f"{u}|{task}|real", {}).get("corpus")}
            res = cider_S_multi(cfgs, (banks[fk]["cand"], banks[fk]["ref"]),
                                (banks[ck]["cand"], banks[ck]["ref"]), keys,
                                n_boot=a.cider_boot, seed=7)
            # θ_null = 0 for the corpus metric — the LACUNA precedent, and it is exact rather than
            # conventional here: the achievable-damage floor scores exactly 0 CIDEr, so "inside the
            # floor's band" means "scores nothing".
            for u, s_ci in res.items():
                lab = LC.label_row(s_ci, 0.0)
                lab.update({"unit": u, "task": task, "readout": readout,
                            "faculty": FACULTY_OF_TASK[task], "corpus": True,
                            "acc_real": s_ci["raw_mean"], "n": len(keys),
                            "band_saturated": False, "power_passes": True})
                cells_out.append(lab)
                band_role = ("defines the CEILING" if u == CEIL_UNIT else
                             "defines the FLOOR" if u == FLOOR_UNIT else None)
                _put(u, f"behavior/{FACULTY_OF_TASK[task]}/{readout}", {
                    "verdict": lab["label"] + (" (band row)" if band_role else ""),
                    "scope": (f"{a.half} half, {readout}, corpus CIDEr, n={len(keys)}"
                              + (f" — {band_role}; S is definitional, not a finding"
                                 if band_role else "")),
                    "pointer": a.pointer})

        for unit in units:
            if unit in partial:
                continue      # reported in rep["partial_units"], never silently folded in
            for task in tasks:
                key = f"{readout}|{task}"
                if key not in bands:
                    continue
                real = items(banks, unit, task, "real")
                gray = items(banks, unit, task, "gray")
                if real is None:
                    continue
                if not power[key]["passes"] or bands[key].get("saturated"):
                    # Dead readout: emit the VOID record and bank the raw accuracy, but compute NO
                    # label. Every unit including the CEILING sits in the band here, so a label
                    # would be an artifact of the readout rather than a fact about the unit.
                    cells_out.append({"unit": unit, "task": task, "readout": readout,
                                      "faculty": FACULTY_OF_TASK[task],
                                      "acc_real": float(real.mean()), "n": int(real.size),
                                      "label": None, "S": None,
                                      "power_passes": power[key]["passes"],
                                      "band_saturated": bool(bands[key].get("saturated")),
                                      "void_gate": ("null_band_saturated"
                                                    if bands[key].get("saturated")
                                                    else "N1_power_gate")})
                    _put(unit, f"behavior/{FACULTY_OF_TASK[task]}/{readout}", {
                        "state": "VOID", "scope": f"{a.half} half, {readout}",
                        "gate": ("null_band_saturated" if bands[key].get("saturated")
                                 else "N1_power_gate"),
                        "pointer": a.pointer})
                    continue
                fl = items(banks, FLOOR_UNIT, task, "real")
                ce = items(banks, CEIL_UNIT, task, "real")
                # Joint finite mask: an item unscorable in ANY of {row, FLOOR, CEILING} is dropped
                # from all three together, so the paired bootstrap stays paired.
                ok = np.isfinite(real) & np.isfinite(fl) & np.isfinite(ce)
                n_drop = int((~ok).sum())
                real, fl, ce = real[ok], fl[ok], ce[ok]
                if gray is not None:
                    gray = gray[ok]
                s_ci = LC.boot_S_paired(real, fl, ce)
                dperc = LC.delta_perception(real, gray) if gray is not None else None
                lab = LC.label_row(s_ci, bands[key]["theta_null"], dperc)
                lab["p_above_band"] = LC.p_above_band(
                    LC.boot_S_draws(real, fl, ce), bands[key]["theta_null"])
                lab.update({"unit": unit, "task": task, "readout": readout,
                            "faculty": FACULTY_OF_TASK[task], "acc_real": float(real.mean()),
                            "n": int(real.size), "n_unscorable_dropped": n_drop,
                            "band_saturated": bands[key]["saturated"],
                            "power_passes": power[key]["passes"]})
                cells_out.append(lab)
                band_role = ("defines the CEILING" if unit == CEIL_UNIT else
                             "defines the FLOOR" if unit == FLOOR_UNIT else None)
                _put(unit, f"behavior/{FACULTY_OF_TASK[task]}/{readout}", {
                    "verdict": lab["label"] + (" (band row)" if band_role else ""),
                    "scope": (f"{a.half} half, {readout}, n={lab['n']}"
                              + (f" — {band_role}; S is definitional, not a finding"
                                 if band_role else "")),
                    "pointer": a.pointer})

    run_readout(FC_B, "forced_choice")
    _save("forced_choice")
    if misaligned:
        rep["misaligned_banks"] = sorted(f"{t}|{k}" for t, k in misaligned)
        for task, k in misaligned:
            unit = k.split("|")[0]
            cid = PM.cell_id(unit, f"behavior/{FACULTY_OF_TASK[task]}/forced_choice")
            records.setdefault(cid, {"state": "VOID", "gate": "bank_positionally_misaligned",
                                     "scope": f"{a.half} half, forced choice",
                                     "pointer": a.pointer})
    run_readout(GEN_B, "generation")
    _save("generation")
    rep["N1_power_gate"] = power
    rep["null_bands"] = bands
    # Which readout carries each faculty's LAW outcome: forced choice unless its band died.
    law_readout = {}
    for c in cells_out:
        f, ro = c["faculty"], c["readout"]
        ok = c["power_passes"] and not c["band_saturated"] and c.get("label") is not None
        if ro == "forced_choice" and ok:
            law_readout[f] = "forced_choice"
        elif f not in law_readout and ro == "generation" and ok:
            law_readout[f] = "generation"
    rep["law_readout_per_faculty"] = law_readout
    rep["cells"] = cells_out

    # ---- the MDE (§3.1): the 2×2 cross-survival pattern, classified mechanically -------------
    MDE_UNITS = ("ft-vqav2", "ft-refcoco-seg")
    MDE_FACS = ("vqa", "grounding")
    grid = {(c["unit"], c["faculty"]): c["S"] for c in cells_out
            if c["unit"] in MDE_UNITS and c["faculty"] in MDE_FACS
            and law_readout.get(c["faculty"]) == c["readout"]}
    if len(grid) == 4:
        rep["MDE"] = PC.mde_pattern(grid)
        rep["MDE"]["grid_S"] = {f"{u}|{f}": v for (u, f), v in sorted(grid.items())}
        rep["MDE"]["labels"] = {f"{c['unit']}|{c['faculty']}": c["label"] for c in cells_out
                                if c["unit"] in MDE_UNITS and c["faculty"] in MDE_FACS}
        rep["MDE"]["registered_prediction"] = "anti_diagonal (H_recipe) — journalled pre-unblind"
        rep["MDE"]["prediction_hit"] = bool(rep["MDE"]["pattern"] == "anti_diagonal")
    elif grid:
        rep["MDE"] = {"incomplete": True, "n_cells": len(grid),
                      "note": "the 2×2 is not complete; no pattern is claimed"}

    # ---- the law (§2.1, §6): ΔLOO over the 26 hop-children only -----------------------------
    if a.census and a.predictions:
        # C6: merge ALL census files. Passing only the FT census silently dropped
        # pi0/pi05 from the fit population (§1.1 fixes it at 26) for want of a ‖ΔW‖.
        cen_cells = []
        for _cf in a.census:
            cen_cells.extend(_load(_cf).get("cells", []))
        cen = {"cells": cen_cells}
        sealed = _load(a.predictions)
        dw = {c["unit"]: c["dw_norm_lm"] for c in cen["cells"]}
        sim = sealed["similarity"]
        ub = PM.unit_by_id()
        S_by = {}
        for c in cells_out:
            if law_readout.get(c["faculty"]) == c["readout"]:
                S_by.setdefault(c["unit"], {})[c["faculty"]] = c["S"]
        fit_ids = [u["id"] for u in PM.units() if u["role"] == "fit"]
        rows, excluded = [], []
        for uid in fit_ids:
            if uid not in S_by or uid not in dw or uid not in sim:
                excluded.append({"unit": uid, "why": "no measured grid row / no census / no sim"})
                continue
            rows.append({"unit": uid, "dw_norm": dw[uid],
                         "recipe_faculties": set(ub[uid]["recipe_faculties"]),
                         "similarity": sim[uid], "S": S_by[uid]})
        drop = {x.strip() for x in a.exclude_faculty.split(",") if x.strip()}
        facs = [f for f in PM.LAW_FACULTIES
                if any(f in r["S"] for r in rows) and f not in drop]
        if drop:
            rep["law_excluded_faculties"] = sorted(drop)
        mem, dwv, simv, y, grp, info = PC.predictor_matrix(rows, facs)
        law = {"n_units_fit": len(rows), "faculties": facs, "excluded": excluded, **info}
        if len(rows) >= 4 and y.size > 0:
            base_other = np.column_stack([dwv, simv])
            law["delta_loo_membership"] = PC.delta_loo_grouped(base_other, mem, y, grp)
            law["delta_loo_membership_ci"] = PC.boot_delta_loo(base_other, mem, y, grp)
            law["delta_loo_dw"] = PC.delta_loo_grouped(np.column_stack([mem, simv]), dwv, y, grp)
            law["delta_loo_similarity"] = PC.delta_loo_grouped(np.column_stack([mem, dwv]), simv,
                                                               y, grp)
            d, ci = law["delta_loo_membership"], law["delta_loo_membership_ci"]
            law["H_recipe_supported"] = bool(d["delta"] > 0 and ci["lo95"] > 0)
            law["note"] = ("§6: H_recipe is supported iff ΔLOO(membership | ‖ΔW‖, similarity) > 0 "
                           "with a bootstrap CI excluding 0 AND the MDE anti-diagonal holds "
                           "(the MDE is adjudicated in its own job).")
        rep["law"] = law

        # ---- prediction scoring (§11 row 3) --------------------------------------------
        obs = {f"{c['unit']}|{c['faculty']}": c["label"] for c in cells_out}
        for h in ("H_recipe", "H_fragility", "H_similarity", "PROGRAM"):
            pred = {k: v[h] for k, v in sealed["predictions"].items() if v.get(h)}
            rep.setdefault("prediction_score", {})[h] = PC.score_predictions(pred, obs)
        (out / "plenum_prediction_score.json").write_text(
            json.dumps(rep["prediction_score"], indent=2, default=str))

    # ---- CENSUS -> MAP records (§7: exploratory, never in a BH family) --------------------
    # The census completed for the whole roster but its cells were never emitted as map records,
    # leaving ~240 weights/geometry cells PENDING over data that already exists. That is a wiring
    # gap, not a measurement gap, and leaving it would let `pending` misreport what was actually
    # measured.
    #
    # Verdicts are DESCRIPTIVE and digit-free (the renderer enforces that). The geometry cells
    # explicitly carry the N2 withdrawal: over all 3024 LM matrices, 100% of updates AND 96.8% of
    # BASE matrices fall below the Frobenius-matched Gaussian band, so the band is withdrawn and
    # these cells are census-only rather than a low-rank verdict.
    if a.census:
        n_cen = 0
        for c in cen.get("cells", []):
            unit = c["unit"]
            for comp, block in c.get("components", {}).items():
                # A component the unit does not HAVE is structurally impossible and already terminal
                # in the frozen map. Emitting a record for it would overwrite that with a verdict
                # about a measurement nobody made.
                if block.get("absent"):
                    continue
                # THE ABSOLUTE-CENSUS BRANCH. The action expert has no pre-install state (§2.3
                # "no 'before' exists"), so there is no ΔW and `median_rel_frob` is absent. The old
                # single-expression verdict would have read that absence as `rel = None` and printed
                # `unmoved` — asserting a comparison against a reference that does not exist. The
                # cell is real and the measurement is real; only the contrast is different, and the
                # verdict has to say so.
                if block.get("absolute"):
                    records[PM.cell_id(unit, f"weights/{comp}/bytes")] = {
                        "verdict": "present (absolute census; no pre-install reference exists)",
                        "scope": (f"median and total Frobenius of the component's own matrices, "
                                  f"n_matrices={block.get('n_matrices')}; EXPLORATORY"),
                        "pointer": a.census_pointer}
                    records[PM.cell_id(unit, f"geometry/{comp}/srank")] = {
                        "verdict": "census-only (absolute stable rank; no pre-install reference)",
                        "scope": ("stable rank of the component's own matrices against a "
                                  "Frobenius-matched Gaussian band; no delta contrast exists so no "
                                  "low-rank-of-the-update claim is available here; EXPLORATORY"),
                        "pointer": a.census_pointer}
                    continue
                rel = block.get("median_rel_frob")
                self_ref = block.get("self_reference_check")
                records[PM.cell_id(unit, f"weights/{comp}/bytes")] = {
                    "verdict": ("unmoved (self-reference; exact structural zero)" if self_ref
                                else "moved" if rel and rel > 0 else "unmoved"),
                    "scope": (f"median relative Frobenius of ΔW vs PG1, n_matrices={block.get('n_matrices')}"
                              + ("; this unit IS the reference, so the zero is by construction and "
                                 "doubles as the load-determinism check" if self_ref else "")),
                    "pointer": a.census_pointer}
                records[PM.cell_id(unit, f"geometry/{comp}/srank")] = {
                    # The verdict names the control WITHOUT a digit: the renderer's no-numbers
                    # rule fired on the literal "N2", which is a gate id rather than a measurement.
                    # Rewording is the right fix — weakening a guard that has been catching real
                    # problems, in order to smuggle a gate name past it, is not.
                    "verdict": "census-only (spectral band withdrawn under the base-matrix control)",
                    "scope": ("stable-rank census; the Gaussian band is withdrawn because the BASE "
                              "matrices fail it too (§5 N2), so no low-rank verdict is drawn"),
                    "pointer": a.census_pointer}
                n_cen += 2
        rep["census_records_emitted"] = n_cen
        _save("census")

    # ---- CONTROL cells (§2.3 step-0) ---------------------------------------------------
    # The verdict is digit-free by the renderer's rule, so the ordering it encodes is stated in
    # words: `at floor` means the unit did not complete a single episode, which is what a freshly
    # initialised action expert must do and is the floor the savings contrast is measured from.
    if a.control:
        n_ctl = 0
        for cf in a.control:
            cd = _load(cf)
            for c in cd.get("cells", []):
                pc = float(c["pc_success"])
                if c.get("is_checkpoint"):
                    # film acquisition clock, not a map cell
                    rep.setdefault("film_acquisition", []).append(
                        {"source": c["unit"], "pc_success": pc})
                    continue
                records[PM.cell_id(c["unit"], "behavior/control/task_native")] = {
                    "verdict": ("at floor (no episode completed)" if pc <= 0
                                else "nonzero zero-shot control"),
                    "scope": (f"zero-shot pc_success, {cd['suite']}, "
                              f"n_episodes={c['n_episodes']}; each unit's OWN normalization stats "
                              f"and no unit is LIBERO-trained — NOT a statement about trained "
                              f"control ability; EXPLORATORY"
                              + (f"; read from a feature-adapted near-step-0 checkpoint "
                                 f"({c['from_checkpoint'].rsplit('/', 3)[-3]} steps) because this "
                                 f"unit's PUBLISHED config declares cameras the LIBERO env does not "
                                 f"supply, so it cannot be evaluated cold"
                                 if c.get("from_checkpoint") else "")),
                    "pointer": a.control_pointer}
                n_ctl += 1
        rep["control_records_emitted"] = n_ctl
        _save("control")

    # ---- CONTROL cells the FROZEN INSTRUMENT cannot measure -------------------------------
    # `behavior/control/task_native` names `lerobot-eval` as its instrument (the map's probe table).
    # That instrument REFUSES π0 and π0.5: their published configs declare
    # `observation.images.{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}` — their own training mix's
    # cameras — while the LIBERO env supplies `{image,image2}`, and it raises
    # "Feature mismatch between dataset/environment and policy config" rather than guessing a
    # correspondence. Measured three separate ways: cold (plenum_ctl_a), in training
    # (plenum_a3_u1c3, "All image features are missing from the batch"), and from a
    # feature-adapted step-0 checkpoint (plenum_ctl_pi0_s0, same mismatch).
    #
    # The step-0 route was the last idea and it does NOT work, for a reason worth stating: the camera
    # remap adapts the DATA at training time (renaming LIBERO's keys to the policy's), which makes
    # training run — but it never changes the CHECKPOINT's declared input_features, so the saved
    # policy still asks for cameras LIBERO does not have. The banked U3/U4 evaluate because
    # `lacuna_units` built their configs LIBERO-shaped from the start.
    #
    # This is a terminal state under a named gate, not a pending cell: the frozen instrument cannot
    # present these units' observations. Forcing a number would mean either editing a checkpoint's
    # declared features to make an eval run, or swapping the instrument after unblinding — and the
    # second is exactly what `memory/feedback_no_post_unblind_rule_swap` forbids.
    for _u in ("pi0", "pi05"):
        _cid = PM.cell_id(_u, "behavior/control/task_native")
        if _cid not in records:
            records[_cid] = {
                "state": "VOID", "gate": "feature_schema_incompatible",
                "scope": ("the frozen instrument (lerobot-eval) refuses this unit: its published "
                          "config declares its own training mix's cameras while the LIBERO env "
                          "supplies image/image2. Observed cold, in training, and from a "
                          "feature-adapted step-0 checkpoint. Adapting the DATA at training time does "
                          "not change the checkpoint's declared input_features. U3/U4 are unaffected "
                          "because lacuna_units built their configs LIBERO-shaped."),
                "pointer": a.control_pointer}
    rep["control_void_feature_schema"] = ["pi0", "pi05"]

    # ---- π0's FILM ONSET: the acquisition clock is unmeasurable, so the statistic is undefined ----
    # §2.3's order statistic is `onset(faculty) − acquisition(control)`, and `acquisition` is the
    # first checkpoint reaching half the final pc_success — a `lerobot-eval` quantity. That instrument
    # refuses every π0-configured checkpoint (`Feature mismatch...`), which also blocked the B*
    # calibration ladder itself (`plenum_a3_u1eval3`, first rung).
    #
    # The faculty half IS measurable — the mm battery re-shells the VLM into the PG1 shell and never
    # touches lerobot-eval — but a difference of two clocks needs both. So the CELL is terminal, not
    # partial, and it terminates under the same named gate as the control cells rather than a new one.
    #
    # NOTE the distinction from VOID-budget: the frozen procedure's VOID-budget fires when the 16k cap
    # is reached without crossing the pc_success threshold. The cap was never reached here — the
    # instrument refused at the first rung. Reporting this as VOID-budget would claim π0 failed to
    # learn control, which was never measured.
    _film_cid = PM.cell_id("pi0", "dynamics/film/onset")
    if _film_cid not in records:
        records[_film_cid] = {
            "state": "VOID", "gate": "feature_schema_incompatible",
            "scope": ("onset(faculty) − acquisition(control) requires a pc_success clock, and "
                      "lerobot-eval refuses π0-configured checkpoints (the same refusal that blocked "
                      "the B* ladder at its first rung). The faculty half is measurable via the "
                      "re-shelled mm battery; the difference of two clocks is not. NOT VOID-budget: "
                      "the frozen cap was never reached, so no claim is made about whether π0 learns "
                      "control."),
            "pointer": a.control_pointer}
    # ---- u3c's VQA cells are CONTAMINATED by the co-training stream -----------------------
    # The VL supervision drew from `load_items(vqav2, 4000)` and the battery's frozen vqav2 count is
    # 4000 — the same call, the same items. At cotrain_ratio=4 over 4096 steps that is ~1024 draws
    # from the evaluated set, so u3c saw roughly a QUARTER of its own VQA eval items during training.
    # Visible in the number: forced-choice vqav2 reads 0.8618 against PG1's 0.7747 ceiling, while
    # u3c's LM has moved only 0.35% from PG1.
    #
    # The cells stay MEASURED — the measurement is real and reproducible — but they carry the
    # contamination in their scope, and the §6 U3c−U3 contrast must not be read on this faculty: U3
    # never saw those items.
    #
    # CONTAMINATION IS A PROPERTY OF THE WEIGHTS, NOT OF THE CELL ID. It was fixed at source: the
    # re-trained U3c draws its supervision from rows [4000,8000) while the battery evaluates
    # [0,4000), disjoint by construction. Stamping every `u3c` VQA cell unconditionally therefore
    # marked the CLEAN re-measurement contaminated, and its scope text — "U3c−U3 must not be read on
    # this faculty" — would have suppressed the §6 contrast on the one faculty Arm 3 intervenes on.
    # That is the same error class as a census cell that cannot distinguish "no change" from "no
    # comparison": a disclosure written for a superseded run, asserted over the run that replaced it.
    #
    # So resolve it from the shard's own provenance. A bank file is {cell_key: scores} and carries no
    # weight identity, so read the sibling metrics.json each contributing shard wrote and compare the
    # published U3c shard hash. Unknown provenance FAILS rather than defaulting either way: guessing
    # "clean" would hide a real leak, and guessing "contaminated" would discard a real result.
    U3C_CLEAN_SHA = "6b930e59ce8ca3c2"      # the disjoint-slice re-train (plenum_a3, post-fix)
    _u3c_shas = {}
    for _f in sorted(sources.get("u3c", ())):
        _m = _load(str(_f).rsplit("/", 1)[0] + "/metrics.json")
        _sha = (_m.get("unit_info", {}).get("u3c", {})
                  .get("shard_sha256", {}).get("model.safetensors"))
        assert _sha, (f"{_f}: no u3c shard sha in its sibling metrics.json — which U3c weights "
                      f"produced these rows is undetermined, and the contamination verdict on the "
                      f"faculty Arm 3 intervenes on depends on exactly that")
        _u3c_shas[str(_f)] = _sha
    _dirty = sorted({s for s in _u3c_shas.values() if s != U3C_CLEAN_SHA})
    assert not (_dirty and len(set(_u3c_shas.values())) > 1), (
        f"u3c rows pooled from BOTH the clean re-train and superseded weights {_dirty} — one cell "
        f"would carry two different models' scores; re-run with only the clean banks")
    rep["u3c_vqa_contamination"] = {
        "faculty": "vqa", "readouts": ["forced_choice", "generation"],
        "overlap": "VL stream and battery both drew load_items(vqav2, 4000)",
        "expected_items_seen": 1024, "of": 4000,
        "clean_sha": U3C_CLEAN_SHA, "shard_shas": _u3c_shas,
        "applies": bool(_dirty),
        "fixed_at_source": "plenum_a3 now draws the supervision slice above the battery's sample"}
    if _dirty:
        for _ro in ("forced_choice", "generation"):
            _cid = PM.cell_id("u3c", f"behavior/vqa/{_ro}")
            if _cid in records and isinstance(records[_cid], dict):
                records[_cid]["scope"] = (records[_cid].get("scope", "")
                                          + " — CONTAMINATED: the VL co-training stream drew from "
                                            "the battery's own vqav2 sample, so ~1/4 of these items "
                                            "were seen in training. U3c−U3 must not be read on this "
                                            "faculty.")
                records[_cid]["contaminated"] = True

    # ---- FILM ONSET when control was never acquired ---------------------------------------
    # §2.3's order statistic is `onset(faculty) − acquisition(control)`, and `plenum_core.
    # acquisition_step` returns None on its FIRST branch when the FINAL pc_success is 0 — it never
    # consults the intermediate checkpoints. So a unit whose endpoint control is 0.0 has an undefined
    # acquisition, hence an undefined order statistic, no matter what the faculty half shows.
    #
    # Measured, not VOID: the measurement RAN and the statistic has no value on this grid. §8's threat
    # row prescribes exactly this — "onset defined on the frozen grid; if undefined → reported undefined
    # (no interpolation)". Recording it as VOID would imply a gate fired; recording it as pending would
    # imply more data could change it, and no amount of faculty measurement can.
    #
    # THE READING, fixed before these numbers landed: U3, U4 and U3c each scored 0.0 over 50
    # LIBERO-Spatial episodes (150 endpoint episodes, zero successes) after 4096 steps at batch 1. That
    # is a statement about the BUDGET being below the scale at which control emerges — NOT about the
    # absence of a consumption-before-acquisition ordering, which would require an arm that acquired
    # control to compare against.
    _final_pc = {}
    for _r in (rep.get("film_acquisition") or []):
        _u = str(_r.get("source", ""))
        for _k in ("U3c", "U3", "U1"):
            if f"/{_k}/" in _u:
                _final_pc.setdefault(_k, []).append(_r["pc_success"])
    for _cd in (rep.get("control_shards_units") or []):
        pass
    _FILM_MAP = {"U3": "u3", "U3c": "u3c"}
    for _fu, _mu in _FILM_MAP.items():
        _cid = PM.cell_id(_mu, "dynamics/film/onset")
        if _cid in records:
            continue
        # the unit's own endpoint control cell is the final clock (the published unit IS the last ckpt)
        _own = records.get(PM.cell_id(_mu, "behavior/control/task_native"))
        _at_floor = bool(_own and "at floor" in str(_own.get("verdict", "")))
        if _at_floor:
            records[_cid] = {
                "verdict": "undefined (control never acquired — acquisition step is None)",
                "scope": ("§2.3 order statistic onset(faculty) − acquisition(control); "
                          "acquisition_step returns None because the FINAL pc_success is 0.0 over 50 "
                          "LIBERO-Spatial episodes, so the difference has no value on this grid. "
                          "Reported undefined, never interpolated (§8). The supported reading is that "
                          "4096 steps at batch 1 is below the scale at which LIBERO control emerges — "
                          "NOT that no consumption-before-acquisition ordering exists; EXPLORATORY"),
                "pointer": a.control_pointer}
    rep["film_onset_undefined"] = [u for u in ("u3", "u3c")
                                   if "undefined" in str(records.get(
                                       PM.cell_id(u, "dynamics/film/onset"), {}).get("verdict", ""))]

    rep["B_star"] = {
        "value": None, "state": "VOID-instrument",
        "why": ("the frozen LACUNA procedure calibrates B* on U1 via a 3-task pc_success probe; "
                "lerobot-eval refuses π0-configured checkpoints, so no rung is evaluable. Distinct "
                "from VOID-budget, which would assert the cap was reached without crossing "
                "threshold — it was not reached."),
        "consequence": ("§2.3 savings are stated at {B*/4, B*} and cannot be computed at the "
                        "pre-registered budgets; U3 and U3c ran at a matched 4096 steps, which is "
                        "§2.3's own disclosed matched-budget fallback.")}

    # ---- THE FILM: onset(faculty) − acquisition(control) (§2.3) -------------------------
    # The checkpoint battery rows were split out of map-cell emission above; here they become the
    # trajectory they are. `plenum_core.onset_statistic` is the frozen statistic — this block only
    # assembles its inputs on a COMMON step grid, which is the whole point of the order statistic
    # (onset and acquisition must be read on one clock or their difference means nothing).
    film_rows = rep.get("film", {}).get("checkpoint_rows") or []
    if film_rows:
        def _unit_step(uri):
            """(film unit, step) from a checkpoint URI. Fails loudly rather than guessing a step."""
            parts = [x for x in uri.replace("ckpt:", "").rstrip("/").split("/") if x]
            steps = [x for x in parts if x.isdigit()]
            assert steps, f"{uri}: no numeric step component — cannot place it on the film clock"
            known = [x for x in parts if x in ("U1", "U2", "U3", "U3c", "U4")]
            assert known, f"{uri}: no film unit component"
            return known[-1], int(steps[-1])

        FILM_MAP = {"U1": "pi0", "U3": "u3", "U3c": "u3c"}     # §2.3 films U1/U3/U3c
        acq_by_src = {r["source"]: r["pc_success"]
                      for r in rep.get("film_acquisition", [])}
        series = {}
        for uri in film_rows:
            fu, st = _unit_step(uri)
            series.setdefault(fu, {})[st] = uri
        rep["film"]["series"] = {k: sorted(v) for k, v in series.items()}
        for fu, by_step in sorted(series.items()):
            if fu not in FILM_MAP:
                continue
            # ALREADY DECIDED. The block above terminates a film cell as measured-undefined when the
            # unit's final pc_success is 0 (acquisition_step returns None from that alone). Recomputing
            # the statistic here then crashes on `onset_step`'s grid assert, because the film battery
            # filmed 512..4096 and never step 0 — a real gap in the grid, but one that cannot change a
            # cell whose acquisition clock is undefined. Skip what is already terminal.
            if PM.cell_id(FILM_MAP[fu], "dynamics/film/onset") in records:
                rep.setdefault("film", {}).setdefault("already_terminal", []).append(fu)
                continue
            steps = sorted(by_step)
            # S per faculty at each step, from the SAME bands the grid rows used.
            # S comes from `cells_out` — the SAME rows the grid's S was computed from, keyed by the
            # checkpoint's unit id. (An earlier draft of this block referenced a `film_S` dict that
            # never existed, which would have made every film unit report "incomplete" forever: a
            # status that cannot fail. Reading the real rows is the fix.)
            by_uf = {}
            for row in cells_out:
                if str(row.get("unit", "")).startswith("ckpt:") and row.get("S") is not None:
                    by_uf.setdefault((row["unit"], row["faculty"]), []).append(float(row["S"]))
            s_by_fac = {}
            for fac in ("vqa", "ocr", "grounding", "caption"):
                ser = []
                for st in steps:
                    vals = by_uf.get((f"ckpt:{by_step[st]}", fac)) or by_uf.get((by_step[st], fac))
                    ser.append(float(sum(vals) / len(vals)) if vals else None)
                s_by_fac[fac] = ser
            pcs = [acq_by_src.get(f"ckpt:{by_step[st]}", acq_by_src.get(by_step[st]))
                   for st in steps]
            if any(x is None for x in pcs):
                rep["film"].setdefault("incomplete", {})[fu] = (
                    "the acquisition clock is missing at least one checkpoint; the order statistic "
                    "is undefined on a partial grid and is NOT interpolated (§8 threat row)")
                continue
            drop = {f: v for f, v in s_by_fac.items() if not any(x is None for x in v)}
            if not drop:
                rep["film"].setdefault("incomplete", {})[fu] = (
                    "no faculty has a complete S series across the checkpoint grid")
                continue
            stat = PC.onset_statistic(steps, drop, pcs)
            rep["film"].setdefault("onset_statistic", {})[fu] = stat
            neg = [f for f, d in stat["per_faculty"].items()
                   if d.get("delta") is not None and d["delta"] < 0]
            und = [f for f, d in stat["per_faculty"].items() if d.get("delta") is None]
            records[PM.cell_id(FILM_MAP[fu], "dynamics/film/onset")] = {
                "verdict": ("consumption precedes acquisition in at least one faculty" if neg
                            else "no faculty onset precedes control acquisition" if not und
                            else "order statistic undefined for at least one faculty"),
                "scope": (f"onset(faculty) − acquisition(control) on the frozen checkpoint grid, "
                          f"faculties with a complete series: {sorted(drop)}; undefined deltas are "
                          f"reported undefined and never interpolated"),
                "pointer": a.pointer}
        rep["film"]["note"] = ("§2.3 films U1/U3/U3c; a film unit with no complete series is "
                              "reported incomplete rather than given an onset")

    # ---- TEXT axes -> S, labels, MAP records --------------------------------------------
    # Axis -> map probe. Both MCQ axes are expected to VOID under N1 (the CEILING itself reads at
    # chance on the letter axis — reproduced across the roster), which is why each axis is banded
    # and gated independently rather than pooled into a single "text" verdict.
    TEXT_PROBE = {"mcq": ("text_recall", "mcq"), "cloze": ("text_recall", "cloze"),
                  "rfp_mcq": ("reading", "mcq"), "rfp_cloze": ("reading", "cloze")}
    if a.text:
        tb = {}
        for f in a.text:
            tb.update(_load(f))
        rep["text_units"] = sorted(tb)
        def _axis_items(unit, axis):
            """Per-item correctness for a text axis.

            The MCQ and RFP axes bank `per_item` directly. The MMLU **cloze** axis banks
            (logprobs, ntoks, correct_idx) instead — so a naive `.get("per_item")` returns None and
            the band is never computed, which silently left all 34 cloze cells PENDING. Derive it
            the way LACUNA's aggregate does: length-normalised argmax over the candidate logprobs
            (λ=1), scored against the banked gold index.
            """
            b = tb.get(unit, {}).get(axis)
            if not b:
                return None
            if b.get("per_item") is not None:
                return np.asarray(b["per_item"], dtype=float)
            lp, nt, gold = b.get("logprobs"), b.get("ntoks"), b.get("correct_idx")
            if lp is None or nt is None or gold is None:
                return None
            L = np.asarray(lp, dtype=float)
            N = np.asarray(nt, dtype=float)
            g = np.asarray(gold, dtype=int)
            return (np.argmax(L / np.maximum(N, 1e-9), axis=1) == g).astype(float)

        for axis, (fac, ro) in TEXT_PROBE.items():
            fl = _axis_items(FLOOR_UNIT, axis)
            ce = _axis_items(CEIL_UNIT, axis)
            if fl is None or ce is None:
                rep.setdefault("missing_band", []).append(f"text|{axis}")
                continue
            
            pw = LC.power_gate(fl, ce)
            rep.setdefault("N1_text", {})[axis] = pw
            band = LC.theta_null(fl, ce) if pw["passes"] else {"theta_null": None,
                                                               "saturated": False}
            for u in tb:
                v = _axis_items(u, axis)
                if v is None:
                    continue
                cid = PM.cell_id(u, f"behavior/{fac}/{ro}")
                if not pw["passes"]:
                    records[cid] = {"state": "VOID", "gate": "N1_power_gate",
                                    "scope": f"text axis {axis}", "pointer": a.text_pointer}
                    continue
                n = min(v.size, fl.size, ce.size)
                s_ci = LC.boot_S_paired(v[:n], fl[:n], ce[:n])
                lab = LC.label_row(s_ci, band["theta_null"])
                role = ("defines the CEILING" if u == CEIL_UNIT else
                        "defines the FLOOR" if u == FLOOR_UNIT else None)
                cells_out.append({**lab, "unit": u, "task": axis, "readout": ro,
                                  "faculty": fac, "n": int(n), "power_passes": True,
                                  "band_saturated": False})
                records[cid] = {"verdict": lab["label"] + (" (band row)" if role else ""),
                                "scope": f"text axis {axis}, n={n}" + (f" — {role}" if role else ""),
                                "pointer": a.text_pointer}
        # perplexity is a corpus-level scalar with no per-item array: recorded descriptively
        for u, b in tb.items():
            if "ppl" in b:
                records[PM.cell_id(u, "behavior/text_ppl/ppl")] = {
                    "verdict": "measured", "scope": "wikitext perplexity, clean-text window",
                    "pointer": a.text_pointer}

    # ---- Arm 2 / Arm 4 terminal VOIDs -> MAP records --------------------------------------
    # Both arms REACHED a verdict and both verdicts are VOID under a named gate. Those are terminal
    # states (§1.4 E1: "measured or VOID, never blocked"), but the verdicts lived only in each job's
    # metrics.json and never became map cells -- so the map reported them as PENDING, i.e. as though
    # the arms had not run. They ran; they failed their gates; that is the result.
    if a.sae:
        # MERGE every SAE shard's gates. Streams were trained in separate runs (gemma-2b is its own
        # job), so reading one file would leave the others' feature cells PENDING over data that
        # exists -- the wiring gap this program has now hit six times.
        sae = {"gates": {}, "cells": []}
        for _sf in a.sae:
            _sd = _load(_sf)
            sae["gates"].update(_sd.get("gates") or {})
            sae["cells"].extend(_sd.get("cells") or [])
        STREAM_UNIT = {"pg1_mm": "pg1", "pg1_text": "pg1", "gemma2b": "gemma-2b",
                       "pi05_mm": "pi05", "pi0_mm": "pi0"}
        n_sae = 0
        for key, g in (sae.get("gates") or {}).items():
            stream, site = key.split("@L")
            unit = STREAM_UNIT.get(stream)
            if not unit:
                continue
            cid = PM.cell_id(unit, f"features/L{site}/identity")
            records[cid] = ({"verdict": "measured", "scope": f"SAE identity, stream {stream}",
                             "pointer": a.sae_pointer} if g.get("all_pass") else
                            {"state": "VOID", "gate": g.get("void_gate"),
                             "scope": f"SAE stream {stream}, site L{site}",
                             "pointer": a.sae_pointer})
            n_sae += 1
        rep["sae_records_emitted"] = n_sae
    if a.patch:
        patch = _load(a.patch)
        p5 = patch.get("P5") or {}
        if not p5.get("passes", True):
            for unit in ("pg1", "pi05", "pi0"):
                for probe in ("mechanism/locus/depth_patch", "mechanism/swap/embedding",
                              "mechanism/swap/projector"):
                    records[PM.cell_id(unit, probe)] = {
                        "state": "VOID", "gate": "P5",
                        "scope": "Arm 4 positive control failed: full-depth transplant does not "
                                 "reproduce the endpoint gap (lm_head is not in the residual stream)",
                        "pointer": a.patch_pointer}
            rep["patch_records_emitted"] = 9

    # ---- BH family A (§7) --------------------------------------------------------------
    famA = {}
    if "law" in rep and "delta_loo_membership" in rep.get("law", {}):
        famA["law_membership"] = rep["law"]["delta_loo_membership"]["p_perm"]
        famA["law_dw"] = rep["law"]["delta_loo_dw"]["p_perm"]
        famA["law_similarity"] = rep["law"]["delta_loo_similarity"]["p_perm"]
    rep["BH_family_A"] = LC.bh_reject(famA, q=0.05) if famA else {"m": 0, "note": "no A tests yet"}
    rep["BH_family_A_disclosure"] = (
        "§7 specifies Family A with m=6 (three ΔLOO contrasts + MDE anti-diagonal + "
        "T6-forced-choice + R5b-vs-R5); this build has m=%d. The MDE test contributes NO "
        "p-value — `mde_pattern` classifies a pattern and never computes one — so it cannot "
        "structurally enter a p-value family as §7 describes. That is a gap between the "
        "spec's prose and the instrument, disclosed rather than papered over. T6/R5b are "
        "absent because Arm 4 VOIDed under P5." % rep["BH_family_A"].get("m", 0))

    (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    (out / "cells.json").write_text(json.dumps(records, indent=2))
    print(f"PLENUM_AGGREGATE_OK cells={len(cells_out)} map_records={len(records)}", flush=True)
    if "law" in rep and "H_recipe_supported" in rep["law"]:
        print(f"  law: ΔLOO(membership)={rep['law']['delta_loo_membership']['delta']:.4f} "
              f"p={rep['law']['delta_loo_membership']['p_perm']:.4f} "
              f"supported={rep['law']['H_recipe_supported']}", flush=True)


if __name__ == "__main__":
    main()
