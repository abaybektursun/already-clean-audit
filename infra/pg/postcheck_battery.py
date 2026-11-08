"""POSTCHECK — the edit battery P3 and N1 both run, and the two things N1 needs that P3 does not.

The battery is deliberately DEPTHANCHOR's: the same released injection path, the same primary unit
(mean |Δ| over the 10 chunk steps and the 6 continuous dimensions of the flow solver's output, captured
before the released clip), the same per-frame ratios. A follow-up that measured the same thing a
different way would not be comparable with the study it is following up, and comparability is the point.
What is new here is the frames it runs on — rendered, unseen — and the model it runs on second.

Per frame, 12 calls (spec §2 P3):

    natural            the model's own grid                        -> b_f
    ident              that grid replayed through the injection    -> the reference every Δ is against
    seed x3            the same grid at three further seeds        -> the seed floor S(f)
    ceiling            another task's rendered image               -> the ceiling C(f)
    shuffle x4         four spatial permutations of the grid       -> the numerator of P3
    const              every cell at the median-depth code
    remove             the depth channel removed

REMOVING THE CHANNEL means two different things on the two checkpoints, and that is the point of N1:
on the fine-tune it is the checkpoint's own gate forced to 0, which leaves the 102 depth positions in
place carrying nothing; on the base there is no gate, so it is `enable_depth_reasoning=False`, which
removes the positions. Neither is a deletion of the other's kind and the artifact says which ran.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from infra.pg import depthanchor_core as core

SEEDS = (1, 2, 3)
N_SHUFFLE = 4


def call_no_depth(actor, frame, seed: int = 0):
    """`enable_depth_reasoning=False`: the released path with no depth prefix at all.

    Mirrors `depthanchor_model.Actor._call` — same processor, same generator, same capture of the
    pre-clip action — with the one flag changed, because the Actor hard-codes depth reasoning on and
    this is the one cell that needs it off.
    """
    import torch
    from infra.pg.depthanchor_model import NORM_TAG, _frame_images, _split_prefix, ActResult

    images, _agentview = _frame_images(frame)
    generator = torch.Generator(device=actor.model.device).manual_seed(int(seed))
    for key in ("a_norm", "gate", "gate_trained", "depth_positions", "depth_cache_out"):
        actor._captured.pop(key, None)
    t0 = time.perf_counter()
    out = actor.model.predict_action(
        processor=actor.processor, images=images, task=str(frame["task"]),
        state=np.asarray(frame["state"], dtype=np.float32),
        norm_tag=getattr(actor, "norm_tag", NORM_TAG),
        inference_action_mode="continuous",
        enable_depth_reasoning=False,          # the whole point of this call
        enable_adaptive_depth=False,
        depth_cache=None,
        num_steps=actor.num_steps,
        generator=generator,
        normalize_language=True,
        enable_cuda_graph=actor.enable_cuda_graph,
    )
    seconds = time.perf_counter() - t0
    actor.calls += 1
    a_norm = actor._captured.get("a_norm")
    if a_norm is None:
        raise RuntimeError("the normalized action chunk was not captured on the no-depth call")
    a_norm = np.asarray(a_norm, dtype=np.float64)
    a_norm = a_norm.reshape(a_norm.shape[-2], a_norm.shape[-1]) if a_norm.ndim == 3 else a_norm
    a_robot = out.actions.detach().float().cpu().numpy()
    a_robot = a_robot.reshape(a_robot.shape[-2], a_robot.shape[-1]) if a_robot.ndim == 3 else a_robot
    token_ids = out.generated_token_ids.detach().cpu().numpy().reshape(-1) \
        if getattr(out, "generated_token_ids", None) is not None else np.zeros(0, dtype=np.int64)
    bins = out.depth_bins.detach().cpu().numpy().reshape(-1).astype(np.int64) \
        if getattr(out, "depth_bins", None) is not None else np.zeros(0, dtype=np.int64)
    n_grid, n_tail = _split_prefix(token_ids, actor.model.config) if token_ids.size else (0, 0)
    return ActResult(a_norm=a_norm, a_robot=a_robot, bins=bins, token_ids=token_ids,
                     n_grid_tokens=n_grid, n_tail_tokens=n_tail, seconds=seconds,
                     gate=actor._captured.get("gate"),
                     meta={"seed": int(seed), "enable_depth_reasoning": False,
                           "n_depth_bins_returned": int(bins.size)})


def add_norm_tag(model, tag: str, metadata: Dict[str, object]) -> Dict[str, object]:
    """Give a checkpoint a normalization tag it does not ship with, through the released code's own
    constructor rather than a reimplementation of it.

    N1 needs this: `MolmoAct2-Think` has no `libero` tag, and the tag is not cosmetic — it selects the
    state normalizer, the action unnormalizer AND the setup/control text that goes into the prompt. So
    the base is read under both its own nearest tag and the fine-tune's `libero` tag, and both are
    banked. `norm_mode` is compared, because copying stats across a different mode would silently
    reinterpret them.
    """
    import sys
    stats = model._get_robot_stats()
    mod = sys.modules[type(stats).__module__]
    before = sorted(stats.metadata_by_tag)
    if tag in stats.metadata_by_tag:
        return {"added": False, "already_present": True, "tags": before}
    stats.metadata_by_tag[tag] = dict(metadata)
    if metadata.get("action_stats") is not None:
        stats.action_normalizers[tag] = mod._FeatureNormalizer.from_stats(
            metadata["action_stats"], stats.norm_mode)
    if metadata.get("state_stats") is not None:
        stats.state_normalizers[tag] = mod._FeatureNormalizer.from_stats(
            metadata["state_stats"], stats.norm_mode)
    return {"added": True, "tag": tag, "tags_before": before,
            "tags_after": sorted(stats.metadata_by_tag),
            "norm_mode_of_this_checkpoint": str(stats.norm_mode),
            "action_horizon": metadata.get("action_horizon"),
            "n_action_steps": metadata.get("n_action_steps"),
            "setup_type": metadata.get("setup_type"),
            "control_mode": metadata.get("control_mode")}


def _record(frame, cell: str, variant: str, ref, res, edit: Optional[core.Edit]) -> Dict[str, object]:
    rec = {
        "frame_id": frame["spec"]["frame_id"], "suite": frame["spec"].get("suite"),
        "task_key": frame["spec"].get("task_key"), "cell": cell, "variant": variant,
        "delta_norm_cont": core.delta_norm_cont(ref.a_norm, res.a_norm),
        "delta_robot_all": core.delta_robot_all(ref.a_robot, res.a_robot),
        "grip_flip": core.grip_flip(ref.a_robot, res.a_robot),
        "n_tail_tokens": int(res.n_tail_tokens),
        "n_tail_matches_reference": bool(int(res.n_tail_tokens) == int(ref.n_tail_tokens)),
        "n_grid_tokens": int(res.n_grid_tokens),
        "seconds": float(res.seconds), "edit_not_completed": False,
        "gate_mean": None if res.gate is None else float(np.mean(res.gate)),
    }
    if edit is not None:
        rec.update({"n_changed": int(edit.n_changed),
                    "realized_depth_shift": float(edit.realized_depth_shift)})
    return rec


def run_battery(actor, frames: Sequence[Dict[str, object]], ladder: core.CodeLadder, *,
                remove_mode: str, on_progress: Optional[Callable] = None) -> Dict[str, object]:
    """The battery over a frame list. `remove_mode` is "gate0" or "no_depth"; anything else is a typo
    that would silently change what "removing the prefix" means, so it raises."""
    if remove_mode not in ("gate0", "no_depth"):
        raise ValueError(f"remove_mode must be 'gate0' or 'no_depth', got {remove_mode!r}")
    records: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    for fi, frame in enumerate(frames):
        fid = frame["spec"]["frame_id"]
        partner = frames[_partner(fi, frames)]
        nat = actor.natural(frame, seed=0)
        if nat.bins.shape != (core.N_CELLS,):
            skipped.append({"frame_id": fid, "n_grid_cells": int(nat.bins.size)})
            continue
        b = nat.bins.copy()
        ref = actor.inject(frame, b, seed=0)
        g0 = float(np.max(np.abs(ref.a_norm - nat.a_norm)))
        records.append(_record(frame, "natural", "adaptive_off", ref, nat, None))
        records.append(_record(frame, "ident", "replay_own_bins", ref, ref, core.edit_identity(b)))

        for s in SEEDS:
            records.append(_record(frame, "seed", f"seed{s}", ref, actor.inject(frame, b, seed=s),
                                   core.edit_identity(b)))

        swapped = dict(frame)
        swapped["agentview"] = partner["agentview"]
        swapped["wrist"] = partner["wrist"]
        rec = _record(frame, "ceiling", "rgb_swap", ref, actor.inject(swapped, b),
                      core.edit_identity(b))
        rec["partner_frame_id"] = partner["spec"]["frame_id"]
        records.append(rec)

        for p in range(N_SHUFFLE):
            edit = core.edit_shuffle(b, core.rng_for("postcheck-shuffle", fid, p), ladder)
            records.append(_record(frame, "shuffle", f"perm{p}", ref, actor.inject(frame, edit.bins),
                                   edit))

        edit = core.edit_const(b, ladder)
        records.append(_record(frame, "const", "median_code", ref, actor.inject(frame, edit.bins), edit))

        if remove_mode == "gate0":
            actor.force_gate(0.0)
            try:
                res = actor.inject(frame, b)
            finally:
                actor.force_gate(None)
            rec = _record(frame, "remove", "depth_gate=0", ref, res, core.edit_identity(b))
        else:
            res = call_no_depth(actor, frame, seed=0)
            rec = _record(frame, "remove", "enable_depth_reasoning=False", ref, res, None)
            rec["n_depth_bins_returned"] = int(res.meta.get("n_depth_bins_returned", -1))
        rec["remove_mode"] = remove_mode
        records.append(rec)

        diagnostics.append({
            "frame_id": fid, "suite": frame["spec"].get("suite"),
            "g0_max_abs_delta_norm": g0,
            "natural_n_tail": int(nat.n_tail_tokens), "ident_n_tail": int(ref.n_tail_tokens),
            "unique_codes": int(len(set(b.tolist()))),
            "gate_trained_mean": (None if ref.meta.get("gate_trained") is None
                                  else float(np.mean(ref.meta["gate_trained"]))),
            "gate_trained": ref.meta.get("gate_trained"),
            "depth_positions": ref.meta.get("depth_positions"),
            "bins": b.astype(int).tolist(),
        })
        if on_progress is not None:
            on_progress({"frames_done": fi + 1, "n_frames": len(frames), "records": len(records),
                         "elapsed_s": round(time.perf_counter() - t0, 1)})
    return {"records": records, "diagnostics": diagnostics, "frames_skipped_short_grid": skipped,
            "remove_mode": remove_mode, "seconds": round(time.perf_counter() - t0, 1)}


def _partner(i: int, frames: Sequence[Dict[str, object]]) -> int:
    """The RGB the ceiling swaps in: a frame of a different task, chosen deterministically."""
    n = len(frames)
    mine = frames[i]["spec"].get("task_key")
    for step in range(1, n):
        j = (i + step * 7) % n
        if frames[j]["spec"].get("task_key") != mine:
            return j
    raise RuntimeError("every frame belongs to the same task; no ceiling partner exists")


def aggregate(battery: Dict[str, object]) -> Dict[str, object]:
    """Per-frame ratios and their medians, in DEPTHANCHOR's units and with its own margin.

    `m` here is this cell's own margin — the 95th percentile over frames of seed floor over ceiling —
    never the frozen one, because these are different frames and a margin from other frames would be
    an assumption rather than a measurement.
    """
    records = battery["records"]
    frames = sorted({r["frame_id"] for r in records})
    suite_of = {r["frame_id"]: r.get("suite") or "all" for r in records}

    def table(cell: str) -> Dict[str, float]:
        by: Dict[str, List[float]] = {}
        for r in records:
            if r["cell"] == cell and not r.get("edit_not_completed"):
                by.setdefault(r["frame_id"], []).append(float(r["delta_norm_cont"]))
        return {k: float(np.mean(v)) for k, v in by.items()}

    floor, ceil = table("seed"), table("ceiling")
    cells = {name: table(name) for name in ("shuffle", "const", "remove", "ident")}
    usable = [f for f in frames if f in floor and f in ceil and ceil[f] > 0]
    out: Dict[str, object] = {"n_frames": len(frames), "n_frames_usable": len(usable),
                              "remove_mode": battery["remove_mode"]}
    if len(usable) < 3:
        out["error"] = "too few frames with both a seed floor and a ceiling"
        return out
    groups = [suite_of[f] for f in usable]
    out["m"] = core.margin_m([floor[f] / ceil[f] for f in usable])
    out["absolute"] = {}
    out["ratio_to_ceiling"] = {}
    out["ratio_to_seed_floor"] = {}
    for name, t in (("seed_floor", floor), ("ceiling_rgb", ceil), *cells.items()):
        have = [f for f in usable if f in t]
        if len(have) < 3:
            continue
        out["absolute"][name] = core.bootstrap_stat([t[f] for f in have],
                                                    [suite_of[f] for f in have], "median",
                                                    seed_label=f"abs:{name}").as_dict()
        out["ratio_to_ceiling"][name] = core.bootstrap_stat(
            [t[f] / ceil[f] for f in have], [suite_of[f] for f in have], "median",
            seed_label=f"ratio:{name}").as_dict()
        out["ratio_to_seed_floor"][name] = core.bootstrap_stat(
            [t[f] / floor[f] for f in have], [suite_of[f] for f in have], "median",
            seed_label=f"floor:{name}").as_dict()
    sh = out["ratio_to_ceiling"].get("shuffle")
    if sh:
        out["shuffle_over_ceiling"] = sh
        out["shuffle_clears_margin"] = bool(sh["ci95"][0] > out["m"])
        out["reads_the_prefix"] = bool(sh["ci95"][0] > out["m"])
    out["g0_max"] = float(max(d["g0_max_abs_delta_norm"] for d in battery["diagnostics"]))
    out["g0_passes_1e-4"] = bool(out["g0_max"] < 1e-4)
    # the `remove` cell under `no_depth` emits NO depth block, so its tail differs from the reference
    # by construction — counting it as a tail mismatch says the instrument is broken when it is doing
    # exactly what the cell asks. Counted separately, and the headline count excludes it.
    mism = [r for r in records if not r.get("n_tail_matches_reference", True)]
    out["tail_mismatches"] = int(sum(1 for r in mism if r["cell"] != "remove"))
    out["tail_mismatches_by_cell"] = {c: sum(1 for r in mism if r["cell"] == c)
                                      for c in sorted({r["cell"] for r in mism})}
    out["remove_cell_emits_no_depth_block"] = bool(
        battery["remove_mode"] == "no_depth"
        and all(r["n_tail_tokens"] == 0 for r in records if r["cell"] == "remove"))
    return out
