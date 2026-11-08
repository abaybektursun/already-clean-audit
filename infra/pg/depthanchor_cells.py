"""DEPTHANCHOR offline cells — the nine edits of spec §3 run over the frame pack, and the aggregation
into the frozen decision rules of §6.

The actor is injected, not imported: anything with `natural(frame, seed=)`, `inject(frame, bins, seed=)`
and `force_gate(value)` will do. That is what lets `infra.aws_pg.jobs.depthanchor_dryrun` run every
cell, every gate branch and every rule on the control plane with a stub actor whose only job is to be
deterministic and to react to the bins it is given — the offline gate the repo requires before a GPU-hour.

ORDER MATTERS AND IS FIXED. Cells run FRAME-MAJOR: everything about one frame, then the next. Two
reasons. The depth-decode CUDA graph is keyed on the prompt length, so a frame-major order captures one
graph per frame instead of thrashing; and every Δ for a frame is then computed against a reference
measured a few seconds earlier on the same resident model, with the identity call REPEATED at the end of
the frame's block and required to agree with the first one (`repeat_ident`). A drifting model, a leaked
gate override or a graph replaying stale inputs all show up there rather than as a small effect.

THE REFERENCE IS THE IDENTITY REPLAY, NOT THE NATURAL CALL (amendment A5). Every Δ is measured against
`a_ref` = the action produced by replaying the model's OWN grid through the injection path, so an edit
and its reference travel the identical code path and differ only in the bins. What the natural
(`enable_adaptive_depth=False`) call is for is G0: the injection path must reproduce it to 1e-4, which is
what makes the path itself trustworthy. Comparing edits against the natural call instead would put the
forced `<depth_start>`, the single-forward replay and the static cache inside every Δ.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from infra.pg import depthanchor_core as core

CELL_NAMES = ("natural", "ident", "seed", "ceiling", "gate0", "gate1", "shuffle", "const",
              "target", "bg", "random", "swap", "cell", "repeat_ident")
# Which cells each session runs, here rather than in the job files, because the preflight projects the
# cost of both sessions and a second copy of these lists would eventually disagree with the first.
GATE_CELLS = ("natural", "ident", "seed", "ceiling", "gate0", "gate1", "shuffle", "const",
              "repeat_ident")
CONTRAST_CELLS = ("target", "bg", "random", "swap", "cell", "natural", "ident", "repeat_ident")

# The model calls one frame costs. `natural`, the identity reference and the repeat at the end of the
# frame run whatever the cell list says, because every Δ is measured against that reference and the
# repeat is the per-frame drift check.
ALWAYS_CALLS = 3
CONDITIONAL_CALLS = {"seed": core.SEED_FLOOR_N - 1, "ceiling": 1, "gate0": 1, "gate1": 1,
                     "shuffle": core.SHUFFLE_N, "const": 1, "target": len(core.K_LEVELS),
                     "bg": len(core.K_LEVELS), "random": 1, "swap": 4}
PER_CELL_MAP_CALLS = 2 * core.N_CELLS


def projected_calls(cells: Sequence[str], n_frames: int, *, per_cell_frames: int = 0,
                    n_suites: int = len(core.SUITES)) -> int:
    """Exactly the calls `run_offline` will make, derived from the cell list instead of typed. The
    preflight uses this to project each session's hours from its own measured seconds per call."""
    per_frame = ALWAYS_CALLS + sum(n for c, n in CONDITIONAL_CALLS.items() if c in cells)
    total = per_frame * int(n_frames)
    if "cell" in cells:
        total += PER_CELL_MAP_CALLS * int(per_cell_frames) * int(n_suites)
    return total


def _inject(actor, frame, bins, *, seed: int = 0):
    """`(result, error)`. The released code raises when the model will not emit `<depth_end>` after an
    injected grid; that is a property of the edit, so it is recorded as one — the record carries
    `edit_not_completed` and no Δ, the cell counts it, and the session continues. Every other exception
    propagates."""
    try:
        return actor.inject(frame, bins, seed=seed), None
    except core.EditNotCompleted as exc:
        return None, str(exc)


# ------------------------------------------------------------------ records
def _failed_record(frame, cell: str, variant: str, edit: Optional[core.Edit], error: str
                   ) -> Dict[str, object]:
    rec = {"frame_id": frame["spec"]["frame_id"], "suite": frame["spec"]["suite"],
           "task_key": frame["spec"]["task_key"], "cell": cell, "variant": variant,
           "edit_not_completed": True, "error": error[:400]}
    if edit is not None:
        rec.update({"n_changed": int(edit.n_changed), "k": edit.meta.get("k"),
                    "n_cells_aimed": len(edit.cells)})
    return rec


def _record(frame, cell: str, variant: str, ref, res, edit: Optional[core.Edit],
            axis: np.ndarray) -> Dict[str, object]:
    rec = {
        "edit_not_completed": False,
        "frame_id": frame["spec"]["frame_id"],
        "suite": frame["spec"]["suite"],
        "task_key": frame["spec"]["task_key"],
        "cell": cell,
        "variant": variant,
        "delta_norm_cont": core.delta_norm_cont(ref.a_norm, res.a_norm),
        "delta_robot_all": core.delta_robot_all(ref.a_robot, res.a_robot),
        "grip_flip": core.grip_flip(ref.a_robot, res.a_robot),
        "delta_z": core.delta_z(ref.a_robot, res.a_robot, axis),
        "n_tail_tokens": int(res.n_tail_tokens),
        "n_tail_matches_reference": bool(int(res.n_tail_tokens) == int(ref.n_tail_tokens)),
        "n_grid_tokens": int(res.n_grid_tokens),
        "seconds": float(res.seconds),
        "seed": int(res.meta.get("seed", 0)),
        "gate_forced": res.meta.get("gate_forced"),
        "gate_mean": None if res.gate is None else float(np.mean(res.gate)),
    }
    if edit is not None:
        rec.update({"n_changed": int(edit.n_changed),
                    "realized_rank_shift": float(edit.realized_rank_shift),
                    "realized_depth_shift": float(edit.realized_depth_shift),
                    "n_cells_aimed": len(edit.cells),
                    "k": edit.meta.get("k")})
    return rec


# ------------------------------------------------------------------ the offline sweep
def run_offline(actor, pack: Dict[str, object], ladder: core.CodeLadder, *,
                cells: Sequence[str] = CELL_NAMES,
                per_cell_frames: int = 5,
                expect_bins: Optional[Dict[str, Sequence[int]]] = None,
                expect_reference: Optional[Dict[str, Sequence[Sequence[float]]]] = None,
                on_progress: Optional[Callable[[Dict[str, object]], None]] = None
                ) -> Dict[str, object]:
    """Every offline cell, frame-major. Returns records, per-frame diagnostics and the banked samples.

    `per_cell_frames` frames PER SUITE get the 200-call per-cell importance map (O-cell). The spec's
    "every cell singly shifted by +8 and by -8" on all 120 frames is 24,000 calls; the budget in §12 is
    3,600, so the map runs on a pre-declared prefix of the frozen frame list and the choice is recorded
    here rather than made after seeing anything (amendment A6).

    `expect_bins` / `expect_reference` come from an EARLIER session's bank. Every Δ is paired inside its
    own session, so nothing depends on the two sessions agreeing — but if they disagree, the frames of
    the later session are not the frames of the earlier one and the merged aggregation would be
    comparing two different models. Both agreements are measured per frame and banked; they are not
    silently assumed.
    """
    frames = pack["frames"]
    records: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    samples: Dict[str, object] = {}
    references: Dict[str, object] = {}
    per_cell_maps: List[Dict[str, object]] = []
    seen_per_suite: Dict[str, int] = {}
    skipped_short: List[Dict[str, object]] = []

    for fi, frame in enumerate(frames):
        fid = frame["spec"]["frame_id"]
        suite = frame["spec"]["suite"]
        axis = np.asarray(frame["axis_into_scene"], dtype=np.float64)
        partner = frames[_ceiling_partner(fi, frames)]

        # --- the model's own grid, and the reference the whole frame is measured against
        nat = actor.natural(frame, seed=0)
        if nat.bins.shape != (core.N_CELLS,):
            # `_generate_depth_prefix` stops when the model emits `<depth_end>`, and at some states it
            # emits it before the grid is full. There is then no 100-cell grid to replay or edit, so the
            # frame carries no records — counted and named, never silently dropped or padded.
            skipped_short.append({"frame_id": fid, "suite": suite,
                                  "n_grid_cells": int(nat.bins.size),
                                  "n_grid_tokens": int(nat.n_grid_tokens),
                                  "n_tail_tokens": int(nat.n_tail_tokens)})
            print(f"[cells] {fid}: the model emitted {int(nat.bins.size)} depth codes, not "
                  f"{core.N_CELLS}; the frame has no editable grid and is skipped", flush=True)
            if on_progress is not None:
                on_progress({"frames_done": fi + 1, "n_frames": len(frames),
                             "records": len(records), "calls": getattr(actor, "calls", None),
                             "frames_skipped_short_grid": len(skipped_short)})
            continue
        b = nat.bins.copy()
        ref = actor.inject(frame, b, seed=0)     # the reference must not be optional: a model that
        # cannot replay its OWN grid is a broken instrument, so this one is deliberately not guarded
        g0_delta = float(np.max(np.abs(ref.a_norm - nat.a_norm)))
        bins_match = None if not expect_bins or fid not in expect_bins else int(
            np.sum(np.asarray(expect_bins[fid], dtype=np.int64) != b))
        ref_drift = None if not expect_reference or fid not in expect_reference else float(
            np.max(np.abs(np.asarray(expect_reference[fid], dtype=np.float64) - ref.a_norm)))
        records.append(_record(frame, "natural", "adaptive_off", ref, nat, None, axis))
        records.append(_record(frame, "ident", "replay_own_bins", ref, ref, core.edit_identity(b), axis))

        if "seed" in cells:
            for s in range(1, core.SEED_FLOOR_N):
                res = actor.inject(frame, b, seed=s)
                records.append(_record(frame, "seed", f"seed{s}", ref, res, core.edit_identity(b), axis))

        if "ceiling" in cells:
            swapped = dict(frame)
            swapped["agentview"] = partner["agentview"]
            swapped["wrist"] = partner["wrist"]
            res, err = _inject(actor, swapped, b)
            rec = (_failed_record(frame, "ceiling", "rgb_swap", None, err) if err else
                   _record(frame, "ceiling", "rgb_swap", ref, res, core.edit_identity(b), axis))
            rec["partner_frame_id"] = partner["spec"]["frame_id"]
            records.append(rec)

        if "gate0" in cells or "gate1" in cells:
            for value, cell in ((0.0, "gate0"), (1.0, "gate1")):
                if cell not in cells:
                    continue
                actor.force_gate(value)
                try:
                    res, err = _inject(actor, frame, b)
                finally:
                    actor.force_gate(None)
                records.append(_failed_record(frame, cell, f"depth_gate={value}", None, err) if err
                               else _record(frame, cell, f"depth_gate={value}", ref, res,
                                            core.edit_identity(b), axis))

        if "shuffle" in cells:
            for p in range(core.SHUFFLE_N):
                edit = core.edit_shuffle(b, core.rng_for("shuffle", fid, p), ladder)
                res, err = _inject(actor, frame, edit.bins)
                records.append(_failed_record(frame, "shuffle", f"perm{p}", edit, err) if err
                               else _record(frame, "shuffle", f"perm{p}", ref, res, edit, axis))

        if "const" in cells:
            edit = core.edit_const(b, ladder)
            res, err = _inject(actor, frame, edit.bins)
            records.append(_failed_record(frame, "const", "median_code", edit, err) if err
                           else _record(frame, "const", "median_code", ref, res, edit, axis))

        target = np.asarray(frame["target_cells"], dtype=np.int64)
        bg = np.asarray(frame["bg_cells"], dtype=np.int64)
        for k in core.K_LEVELS:
            for cell_name, which in (("target", target), ("bg", bg)):
                if cell_name not in cells:
                    continue
                edit = core.edit_shift_cells(b, which, k, ladder, cell_name)
                res, err = _inject(actor, frame, edit.bins)
                records.append(_failed_record(frame, cell_name, f"k{k:+d}", edit, err) if err
                               else _record(frame, cell_name, f"k{k:+d}", ref, res, edit, axis))

        if "swap" in cells:
            # the §8 pivot, run always: each aimed cell takes the code of this frame's own farthest or
            # nearest cell, which needs no ordinal ladder and injects only codes the model itself chose
            for which in ("far", "near"):
                for cell_name, aimed in (("target_swap", target), ("bg_swap", bg)):
                    edit = core.edit_swap_extreme(b, aimed, which, ladder)
                    res, err = _inject(actor, frame, edit.bins)
                    records.append(_failed_record(frame, cell_name, which, edit, err) if err
                                   else _record(frame, cell_name, which, ref, res, edit, axis))

        if "random" in cells:
            edit = core.edit_random_cells(b, target, core.rng_for("random", fid), ladder)
            res, err = _inject(actor, frame, edit.bins)
            records.append(_failed_record(frame, "random", "match_target_count", edit, err) if err
                           else _record(frame, "random", "match_target_count", ref, res, edit, axis))

        take_map = ("cell" in cells and seen_per_suite.get(suite, 0) < int(per_cell_frames))
        if take_map:
            seen_per_suite[suite] = seen_per_suite.get(suite, 0) + 1
            grid = np.full((core.N_CELLS, 2), np.nan, dtype=np.float64)
            n_failed_cells = 0
            for c in range(core.N_CELLS):
                for j, k in enumerate((+8, -8)):
                    edit = core.edit_single_cell(b, c, k, ladder)
                    res, err = _inject(actor, frame, edit.bins)
                    if err:
                        n_failed_cells += 1
                        continue
                    grid[c, j] = core.delta_norm_cont(ref.a_norm, res.a_norm)
            per_cell_maps.append({"frame_id": fid, "suite": suite,
                                  "target_cells": target.astype(int).tolist(),
                                  "delta_plus8": grid[:, 0].tolist(),
                                  "delta_minus8": grid[:, 1].tolist(),
                                  "n_cells_not_completed": int(n_failed_cells)})

        rep = actor.inject(frame, b, seed=0)
        records.append(_record(frame, "repeat_ident", "replay_own_bins", ref, rep,
                               core.edit_identity(b), axis))
        diagnostics.append({
            "frame_id": fid, "suite": suite,
            "g0_max_abs_delta_norm": g0_delta,
            "repeat_ident_max_abs_delta_norm": float(np.max(np.abs(rep.a_norm - ref.a_norm))),
            "natural_n_tail": int(nat.n_tail_tokens), "ident_n_tail": int(ref.n_tail_tokens),
            "natural_n_grid": int(nat.n_grid_tokens),
            "gate_trained_mean": (None if ref.meta.get("gate_trained") is None
                                  else float(np.mean(ref.meta["gate_trained"]))),
            "gate_trained": ref.meta.get("gate_trained"),
            "depth_positions": ref.meta.get("depth_positions"),
            "n_target_cells": int(target.size), "bg_min_distance": int(frame["bg_min_distance"]),
            "seconds_natural": float(nat.seconds), "seconds_inject": float(ref.seconds),
            "bins": b.astype(int).tolist(),
            "per_cell_map": bool(take_map),
            "edits_not_completed": int(sum(1 for r in records
                                           if r["frame_id"] == fid and r.get("edit_not_completed"))),
            "cells_differing_from_earlier_session": bins_match,
            "reference_drift_vs_earlier_session": ref_drift,
        })
        references[fid] = {"a_norm": ref.a_norm.tolist(), "a_robot": ref.a_robot.tolist(),
                           "bins": b.astype(int).tolist(), "n_tail": int(ref.n_tail_tokens)}
        if fid not in samples and len(samples) < 8:
            samples[fid] = _sample_bank(frame, nat, ref, b)
        if on_progress is not None:
            on_progress({"frames_done": fi + 1, "n_frames": len(frames),
                         "records": len(records), "calls": getattr(actor, "calls", None)})

    return {"records": records, "diagnostics": diagnostics, "per_cell_maps": per_cell_maps,
            "samples": samples, "references": references,
            "frames_skipped_short_grid": skipped_short,
            "per_cell_frames_per_suite": int(per_cell_frames), "cells_run": list(cells)}


def _ceiling_partner(i: int, frames: Sequence[Dict[str, object]]) -> int:
    """The RGB the ceiling swaps in: a frame of a DIFFERENT task, chosen deterministically by walking
    the frozen list until the task differs, so the ceiling is a task change and not a nearby frame of
    the same scene."""
    n = len(frames)
    mine = frames[i]["spec"]["task_key"]
    for step in range(1, n):
        j = (i + step * 7) % n
        if frames[j]["spec"]["task_key"] != mine:
            return j
    raise RuntimeError("every frame in the pack belongs to the same task; no ceiling partner exists")


def _sample_bank(frame, nat, ref, bins) -> Dict[str, object]:
    """Spec §10: the real input and output beside every readout — the prompt's task string, the state,
    the bins, the generated token ids and both action chunks."""
    return {"task": frame["task"], "state": np.asarray(frame["state"]).tolist(),
            "spec": frame["spec"], "bins": np.asarray(bins).astype(int).tolist(),
            "natural_token_ids": np.asarray(nat.token_ids).astype(int).tolist(),
            "ident_token_ids": np.asarray(ref.token_ids).astype(int).tolist(),
            "natural_a_norm": np.asarray(nat.a_norm).tolist(),
            "ident_a_norm": np.asarray(ref.a_norm).tolist(),
            "natural_a_robot": np.asarray(nat.a_robot).tolist(),
            "ident_a_robot": np.asarray(ref.a_robot).tolist(),
            "target_cells": np.asarray(frame["target_cells"]).astype(int).tolist(),
            "bg_cells": np.asarray(frame["bg_cells"]).astype(int).tolist()}


# ------------------------------------------------------------------ banking the extremes
def bank_extremes(actor, pack: Dict[str, object], ladder: core.CodeLadder,
                  records: Sequence[Dict[str, object]], cells: Sequence[str],
                  key: str = "delta_norm_cont") -> Dict[str, object]:
    """SPEC §10 and the repo's rule 6: for every cell, the real input and output of the frame it scored
    HIGHEST and the frame it scored LOWEST — the task string, the state, the grid before and after, the
    generated token ids, and both action chunks. A number without samples is not quotable, and a mean
    without its extremes hides the two records most worth reading.

    The extremes are re-run rather than kept from the sweep, because keeping every chunk would multiply
    the record size by the chunk; two extra calls per cell is nothing against 2,600.
    """
    by_id = {f["spec"]["frame_id"]: f for f in pack["frames"]}
    out: Dict[str, object] = {}
    for cell in cells:
        rows = [r for r in records if r["cell"] == cell and not r.get("edit_not_completed")
                and key in r]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: float(r[key]))
        for label, r in (("lowest", rows[0]), ("highest", rows[-1])):
            frame = by_id.get(r["frame_id"])
            if frame is None:
                continue
            nat = actor.natural(frame, seed=0)
            b = nat.bins.copy()
            edit = _rebuild_edit(cell, r, b, frame, ladder)
            ref = actor.inject(frame, b, seed=0)
            res, err = (_inject(actor, frame, edit.bins, seed=int(r.get("seed") or 0))
                        if edit is not None else (ref, None))
            out[f"{cell}:{label}"] = {
                "cell": cell, "which": label, "variant": r["variant"], key: float(r[key]),
                "frame_id": r["frame_id"], "suite": r["suite"], "task": frame["task"],
                "state": np.asarray(frame["state"]).tolist(),
                "bins_before": b.astype(int).tolist(),
                "bins_after": None if edit is None else edit.bins.astype(int).tolist(),
                "n_cells_changed": None if edit is None else int(edit.n_changed),
                "realized_depth_shift": None if edit is None else float(edit.realized_depth_shift),
                "target_cells": np.asarray(frame["target_cells"]).astype(int).tolist(),
                "bg_cells": np.asarray(frame["bg_cells"]).astype(int).tolist(),
                "token_ids_reference": ref.token_ids.astype(int).tolist(),
                "token_ids_edited": None if err else res.token_ids.astype(int).tolist(),
                "a_norm_reference": ref.a_norm.tolist(),
                "a_norm_edited": None if err else res.a_norm.tolist(),
                "a_robot_reference": ref.a_robot.tolist(),
                "a_robot_edited": None if err else res.a_robot.tolist(),
                "delta_norm_cont_rerun": (None if err else
                                          core.delta_norm_cont(ref.a_norm, res.a_norm)),
                "edit_not_completed": bool(err), "error": err,
                "gate_forced": r.get("gate_forced"),
            }
    return out


def _rebuild_edit(cell: str, record: Dict[str, object], bins, frame, ladder: core.CodeLadder):
    """Reconstruct the exact edit a record came from, from its cell and variant plus the frame's own
    frozen cell sets and the master seed. Deterministic by construction: every edit in `run_offline` is
    a pure function of those."""
    fid = frame["spec"]["frame_id"]
    target = np.asarray(frame["target_cells"], dtype=np.int64)
    bg = np.asarray(frame["bg_cells"], dtype=np.int64)
    variant = str(record["variant"])
    if cell in ("ident", "seed", "ceiling", "gate0", "gate1", "repeat_ident", "natural"):
        return None
    if cell == "shuffle":
        return core.edit_shuffle(bins, core.rng_for("shuffle", fid, int(variant[4:])), ladder)
    if cell == "const":
        return core.edit_const(bins, ladder)
    if cell in ("target", "bg"):
        return core.edit_shift_cells(bins, target if cell == "target" else bg,
                                     int(variant[1:]), ladder, cell)
    if cell in ("target_swap", "bg_swap"):
        return core.edit_swap_extreme(bins, target if cell == "target_swap" else bg, variant, ladder)
    if cell == "random":
        return core.edit_random_cells(bins, target, core.rng_for("random", fid), ladder)
    raise ValueError(f"no rebuild rule for cell {cell!r}")


# ------------------------------------------------------------------ aggregation
def _frame_table(records: Sequence[Dict[str, object]], cell: str, key: str = "delta_norm_cont",
                 variant: Optional[str] = None, reduce: str = "mean") -> Dict[str, float]:
    """One number per frame for one cell: the mean (or median) over that cell's variants."""
    by_frame: Dict[str, List[float]] = {}
    for r in records:
        if r["cell"] != cell or r.get("edit_not_completed"):
            continue
        if variant is not None and r["variant"] != variant:
            continue
        by_frame.setdefault(r["frame_id"], []).append(float(r[key]))
    fn = {"mean": np.mean, "median": np.median, "max": np.max}[reduce]
    return {f: float(fn(v)) for f, v in by_frame.items()}


def aggregate(offline: Dict[str, object], gates: Dict[str, core.Gate], *,
              ceiling_cell: str = "ceiling", require_rule2: bool = True,
              frozen_m: Optional[float] = None) -> Dict[str, object]:
    """Everything §6 and §7 ask for, computed from the records. `ceiling_cell` selects which ceiling
    normalizes the ratios: "ceiling" is the spec's RGB swap, "gate0" the depth-matched ablation
    (amendment A1). Both are reported; the frozen rules read the spec's."""
    records = offline["records"]
    diagnostics = offline["diagnostics"]
    frame_ids = [d["frame_id"] for d in diagnostics]
    suite_of = {d["frame_id"]: d["suite"] for d in diagnostics}
    out: Dict[str, object] = {"ceiling_cell": ceiling_cell}

    seed_floor = _frame_table(records, "seed")
    ceiling_rgb = _frame_table(records, "ceiling")
    ceiling_gate0 = _frame_table(records, "gate0")
    gate1 = _frame_table(records, "gate1")
    shuffle = _frame_table(records, "shuffle")
    const = _frame_table(records, "const")
    random_ = _frame_table(records, "random")
    ceiling = {"ceiling": ceiling_rgb, "gate0": ceiling_gate0}[ceiling_cell]

    # A frame drops out of a statistic when the model refused to finish an edited prefix for that cell
    # (`edit_not_completed`). Dropping is right — there is no Δ to score — but it has to be COUNTED, and
    # losing most of the frames means the instrument, not the frame, is the problem.
    usable = [f for f in frame_ids if f in seed_floor and f in ceiling]
    out["frames_total"] = len(frame_ids)
    out["frames_usable"] = len(usable)
    out["frames_dropped"] = [f for f in frame_ids if f not in usable]
    if len(usable) < max(3, len(frame_ids) // 2):
        raise RuntimeError(f"only {len(usable)} of {len(frame_ids)} frames have both a seed floor and a "
                           f"ceiling; there is nothing to normalize against")
    frame_ids = usable
    if not shuffle:
        # Rule 1 reads the shuffle cell. With no shuffle records the interval would be zero and the
        # label would come out H_epi — a missing measurement must never become a negative result.
        raise RuntimeError("no O-shuffle records: rule 1 has no input, and an absent shuffle cell "
                           "would otherwise be scored as 'the prefix is not read'")

    out["n_frames"] = len(frame_ids)
    out["absolute"] = {}
    for name, table in (("seed_floor", seed_floor), ("ceiling_rgb", ceiling_rgb),
                        ("ceiling_gate0", ceiling_gate0), ("gate1", gate1), ("shuffle", shuffle),
                        ("const", const), ("random", random_)):
        have = [f for f in frame_ids if f in table]
        if len(have) >= 3:
            iv = core.bootstrap_stat([table[f] for f in have], [suite_of[f] for f in have],
                                     "median", seed_label=name)
            out["absolute"][name] = dict(iv.as_dict(), n_frames_used=len(have))

    # the room checks: both ceilings against the same seed floor
    rooms = {}
    for label, table in (("G4a_rgb_ceiling", ceiling_rgb), ("G4b_depth_ceiling", ceiling_gate0)):
        have = [f for f in frame_ids if f in table]
        if len(have) < 3:
            continue
        if min(seed_floor[f] for f in have) <= 0:
            raise RuntimeError("a frame has a zero seed floor: the flow sampler produced identical "
                               "actions for different seeds, which means the seed floor is not a "
                               "noise measurement (check that `generator` reaches the sampler)")
        ratio = [table[f] / seed_floor[f] for f in have]
        iv = core.bootstrap_stat(ratio, [suite_of[f] for f in have], "median", seed_label=label)
        rooms[label] = iv
        out.setdefault("room", {})[label] = dict(iv.as_dict(), n_frames_used=len(have))

    # rule 1: the shuffled prefix against the frozen margin
    m_recomputed = core.margin_m([seed_floor[f] / ceiling[f] for f in frame_ids])
    out["m_n_frames"] = len(frame_ids)
    # The margin is FROZEN in the gate session. Recomputing it here from the same merged control
    # records must give the same number; if it does not, the frozen one still decides and the
    # discrepancy is banked, because a margin that moves after unblinding is not a margin.
    m = m_recomputed if frozen_m is None else float(frozen_m)
    shuffle_frames = [f for f in frame_ids if f in shuffle]
    shuffle_ratio = [shuffle[f] / ceiling[f] for f in shuffle_frames]
    out["m"] = m
    out["m_recomputed"] = m_recomputed
    out["m_frozen"] = frozen_m
    out["m_matches_frozen"] = None if frozen_m is None else bool(abs(m_recomputed - frozen_m) < 1e-9)
    out["ratios"] = {}
    for name, table in (("shuffle", shuffle), ("const", const), ("random", random_),
                        ("gate0", ceiling_gate0), ("gate1", gate1)):
        have = [f for f in frame_ids if f in table]
        if len(have) >= 3:
            iv = core.bootstrap_stat([table[f] / ceiling[f] for f in have],
                                     [suite_of[f] for f in have], "median",
                                     seed_label=f"ratio:{name}")
            out["ratios"][name] = dict(iv.as_dict(), n_frames_used=len(have))
    shuffle_iv = (core.bootstrap_stat(shuffle_ratio, [suite_of[f] for f in shuffle_frames], "median",
                                      seed_label="rule1") if len(shuffle_ratio) >= 3 else None)
    if shuffle_iv is None:
        raise RuntimeError(f"only {len(shuffle_ratio)} frames have a scorable O-shuffle cell; rule 1 "
                           "cannot be read, and an absent shuffle must never be scored as 'not read'")

    # rule 2a: target minus background at each k, paired within frame
    paired: Dict[int, core.Interval] = {}
    bh: Dict[int, float] = {}
    per_k: Dict[str, object] = {}
    for k in core.K_LEVELS:
        t = _frame_table(records, "target", variant=f"k{k:+d}")
        g = _frame_table(records, "bg", variant=f"k{k:+d}")
        common = [f for f in frame_ids if f in t and f in g]
        if len(common) < 3:
            continue
        gsub = [suite_of[f] for f in common]
        paired[k] = core.bootstrap_paired_difference([t[f] for f in common], [g[f] for f in common],
                                                     gsub, "median", seed_label=f"k{k}")
        bh[k] = core.permutation_pvalue([t[f] - g[f] for f in common], seed_label=f"k{k}")
        per_k[str(k)] = {
            "target": core.bootstrap_stat([t[f] for f in common], gsub, "median",
                                          seed_label=f"t{k}").as_dict(),
            "bg": core.bootstrap_stat([g[f] for f in common], gsub, "median",
                                      seed_label=f"g{k}").as_dict(),
            "target_over_ceiling": core.bootstrap_stat([t[f] / ceiling[f] for f in common], gsub,
                                                       "median", seed_label=f"tc{k}").as_dict(),
            "paired_target_minus_bg": paired[k].as_dict(),
            "permutation_p": bh[k],
            "n_frames": len(common),
        }
    out["per_k"] = per_k

    # the §8 pivot contrast: target minus background under the within-frame extreme swap
    paired_swap: Dict[str, core.Interval] = {}
    swap_p: Dict[str, float] = {}
    swap_report: Dict[str, object] = {}
    for which in ("far", "near"):
        t = _frame_table(records, "target_swap", variant=which)
        g = _frame_table(records, "bg_swap", variant=which)
        common = [f for f in frame_ids if f in t and f in g]
        if len(common) < 3:
            continue
        gsub = [suite_of[f] for f in common]
        paired_swap[which] = core.bootstrap_paired_difference([t[f] for f in common],
                                                             [g[f] for f in common], gsub, "median",
                                                             seed_label=f"swap{which}")
        swap_p[which] = core.permutation_pvalue([t[f] - g[f] for f in common],
                                                seed_label=f"swap{which}")
        swap_report[which] = {
            "target": core.bootstrap_stat([t[f] for f in common], gsub, "median",
                                          seed_label=f"st{which}").as_dict(),
            "bg": core.bootstrap_stat([g[f] for f in common], gsub, "median",
                                      seed_label=f"sg{which}").as_dict(),
            "paired_target_minus_bg": paired_swap[which].as_dict(),
            "permutation_p": swap_p[which], "n_frames": len(common)}
    out["pivot_swap"] = swap_report

    # rule 2b: the directional Spearman, target against the background null. Every family is computed;
    # the one that decides is the one whose manipulation the spatial clause used (A20).
    g2_gate = gates.get("G2")
    # three-valued on purpose: the gate session does not pass G2 at all, and "not supplied" must not be
    # banked as "failed" — the deciding family is the pivot either way, and in the gate session it has no
    # records to read
    g2_verdict = None if g2_gate is None else str(g2_gate.verdict)
    deciding_family = "k_ladder" if g2_verdict == "pass" else "pivot_swap"
    families = {name: _directional(records, suite_of, family=name) for name in DIRECTIONAL_FAMILIES}
    sp_target, sp_bg_p95, sp_perm_p, directional = families[deciding_family]
    out["directional"] = dict(directional)
    out["directional"]["deciding_family"] = deciding_family
    out["directional"]["G2_verdict"] = g2_verdict
    out["directional_by_family"] = {name: v[3] for name, v in families.items()}

    # rule 4: anchor concentration
    anchor, count_share = _anchor(offline, suite_of)
    out["anchor"] = {"share": None if anchor is None else anchor.as_dict(),
                     "count_share": count_share}

    # POWER, banked beside every contrast: the half-width of each paired interval against the two
    # ceilings and the noise floor. A null whose resolution is wider than the whole depth channel's
    # dynamic range is not a negative result, and this is where a reader sees that without recomputing
    # anything (closeout §4.2: "a recovery-fraction gate at n 73 cannot decide anything").
    med_seed = float(np.median([seed_floor[f] for f in frame_ids]))
    g0_have = [f for f in frame_ids if f in ceiling_gate0]
    med_depth_ceiling = float(np.median([ceiling_gate0[f] for f in g0_have])) if g0_have else None
    rgb_have = [f for f in frame_ids if f in ceiling_rgb]
    med_rgb_ceiling = float(np.median([ceiling_rgb[f] for f in rgb_have])) if rgb_have else None
    halfwidth = {str(k): (iv.hi - iv.lo) / 2.0 for k, iv in paired.items()}
    out["power"] = {
        "seed_floor_median": med_seed,
        "depth_ceiling_median": med_depth_ceiling,
        "rgb_ceiling_median": med_rgb_ceiling,
        "paired_ci_halfwidth": halfwidth,
        "mde_over_depth_ceiling": (None if not med_depth_ceiling else
                                   {k: v / med_depth_ceiling for k, v in halfwidth.items()}),
        "mde_over_seed_floor": (None if med_seed <= 0 else
                                {k: v / med_seed for k, v in halfwidth.items()}),
    }
    primary_iv = paired.get(core.PRIMARY_ABS_K)
    if primary_iv is not None and med_depth_ceiling:
        hw = (primary_iv.hi - primary_iv.lo) / 2.0
        out["power"]["primary_resolves_within_depth_channel"] = bool(hw <= med_depth_ceiling)

    out["gates"] = {k: g.as_dict() for k, g in gates.items()}
    def _iv_from(name):
        d = out["ratios"].get(name)
        return None if not d else core.Interval(point=float(d["point"]), lo=float(d["ci95"][0]),
                                                hi=float(d["ci95"][1]), n=int(d["n"]),
                                                statistic=str(d["statistic"]))

    decision = core.decide(gates=gates, m=m,
                           shuffle_over_ceiling=shuffle_iv,
                           const_over_ceiling=_iv_from("const"),
                           ablation_over_ceiling=_iv_from("gate0"),
                           paired_target_minus_bg=paired, bh_family=bh,
                           paired_swap=paired_swap or None, swap_pvalues=swap_p or None,
                           spearman_target=sp_target, spearman_bg_p95=sp_bg_p95,
                           spearman_permutation_p=sp_perm_p,
                           anchor_share=anchor, anchor_count_share=count_share,
                           require_rule2=require_rule2)
    if (primary_iv is not None and not primary_iv.excludes(0.0) and med_depth_ceiling
            and (primary_iv.hi - primary_iv.lo) / 2.0 > med_depth_ceiling):
        decision.notes.append(
            "the primary contrast's interval half-width exceeds the median depth-ablation ceiling: "
            "this null does not resolve an effect smaller than removing the depth prefix entirely, "
            "and is reported as a resolution limit rather than as an absence.")
    out["decision"] = decision.as_dict()
    out["tail_mismatches"] = int(sum(0 if r.get("n_tail_matches_reference", True) else 1
                                     for r in records))
    out["edits_not_completed"] = int(sum(1 for r in records if r.get("edit_not_completed")))
    out["records_scored"] = int(sum(1 for r in records if not r.get("edit_not_completed")))
    out["per_cell_edits_not_completed"] = int(sum(int(m.get("n_cells_not_completed") or 0)
                                                 for m in (offline.get("per_cell_maps") or [])))
    out["frames_skipped_short_grid"] = offline.get("frames_skipped_short_grid") or []
    return out


# Which edit family the DIRECTIONAL clause reads. Rule 2b is written on the k-ladder, and A2/A3 amended
# its predictor to the REALIZED decoded-depth shift, which every aimed edit has — including the §8 pivot's
# within-frame extreme swap. So when G2 fails and the pivot becomes the spatial contrast, the directional
# clause reads the same manipulation the spatial clause does, instead of a ladder whose ranks are
# undefined (A20). All three readings are computed and banked; the gate's verdict picks which one decides.
DIRECTIONAL_FAMILIES = {
    "k_ladder": {"target": ("target",), "bg": ("bg",)},
    "pivot_swap": {"target": ("target_swap",), "bg": ("bg_swap",)},
    "pooled": {"target": ("target", "target_swap"), "bg": ("bg", "bg_swap")},
}


def _directional(records: Sequence[Dict[str, object]], suite_of: Dict[str, str],
                 family: str = "k_ladder"
                 ) -> Tuple[Optional[core.Interval], Optional[float], Optional[float],
                            Dict[str, object]]:
    """Spearman between the REALIZED decoded-depth shift and Δ_z, pooled over a frame's aimed edits and
    bootstrapped over frames; the same statistic on the background cells gives the null band. The
    predictor is the realized shift in decoded depth, not the requested k, because the ladder clamps at
    its ends — and because under the §8 pivot there is no k at all.

    `family` selects which aimed edits are read: the k-ladder (rule 2b as written), the pivot swap (rule
    2b under §8), or both pooled. Every family is computed by `aggregate`; this returns one of them."""
    out: Dict[str, object] = {"family": family}
    ivs: Dict[str, core.Interval] = {}
    reps: Dict[str, np.ndarray] = {}
    cells_of = DIRECTIONAL_FAMILIES[family]
    for side in ("target", "bg"):
        rows = [r for r in records if r["cell"] in cells_of[side]
                and r.get("realized_depth_shift") is not None
                and not r.get("edit_not_completed")]
        if len(rows) < 12:
            continue
        fids = [r["frame_id"] for r in rows]
        # "farther from the camera" is a NEGATIVE change in decoded disparity (higher decoded value =
        # nearer). The sign convention is an instrument fact confirmed in G1B; `sign_convention` is
        # recorded here so a reversal is a one-line correction with an artifact, not a re-analysis.
        x = [-float(r["realized_depth_shift"]) for r in rows]
        y = [float(r["delta_z"]) for r in rows]
        iv, rep = core.spearman_by_frame_bootstrap(fids, suite_of, x, y, seed_label=f"{family}:{side}")
        perm = core.spearman_within_frame_permutation(fids, x, y, seed_label=f"{family}:{side}")
        ivs[side] = iv
        reps[side] = rep
        out[side] = iv.as_dict()
        out[f"{side}_permutation"] = perm
        out[f"{side}_n_records"] = len(rows)
        out[f"{side}_predictor_range"] = [float(np.min(x)), float(np.max(x))]
        out[f"{side}_predictor_nonzero_fraction"] = float(np.mean(np.abs(x) > 1e-12))
    out["sign_convention"] = ("predictor = -(realized change in decoded depth) = 'farther from the "
                              "camera'; confirmed against simulator ground truth in G1B")
    if "target" not in ivs or "bg" not in ivs:
        return None, None, None, out
    bg_p95 = float(np.percentile(np.abs(reps["bg"]), 95))
    out["bg_abs_p95"] = bg_p95
    return ivs["target"], bg_p95, out["target_permutation"]["p_two_sided"], out


def _anchor(offline: Dict[str, object], suite_of: Dict[str, str]
            ) -> Tuple[Optional[core.Interval], Optional[float]]:
    """§6 rule 4 (descriptive): the share of the total per-cell importance that sits in the target
    cells, against the share their count alone would give."""
    maps = offline.get("per_cell_maps") or []
    if not maps:
        return None, None
    shares, groups, counts = [], [], []
    for entry in maps:
        imp = (np.abs(np.asarray(entry["delta_plus8"], dtype=np.float64))
               + np.abs(np.asarray(entry["delta_minus8"], dtype=np.float64)))
        imp = np.nan_to_num(imp, nan=0.0)      # a cell the model would not finish contributes nothing
        total = float(imp.sum())
        if total <= 0:
            continue
        cells = np.asarray(entry["target_cells"], dtype=np.int64)
        shares.append(float(imp[cells].sum() / total))
        counts.append(float(cells.size / core.N_CELLS))
        groups.append(suite_of.get(entry["frame_id"], entry.get("suite", "?")))
    if not shares:
        return None, None
    return (core.bootstrap_stat(shares, groups, "median", seed_label="anchor"),
            float(np.median(counts)))
