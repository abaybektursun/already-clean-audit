"""DEPTHANCHOR closed loop — spec §3 and §6 rule 3: seven depth regimes, one suite per instance.

Arms, and what each one is for:

  C-normal   the released deployment path (`enable_adaptive_depth=True`, the caller-owned cache threaded
             from step to step) — the reference every other arm is compared with
  C-fresh    full regeneration every step — the cost of adaptivity
  C-stale    the grid from step 0 replayed for the whole episode
  C-shuffle  the model's own grid each step, spatially permuted with one permutation per episode
  C-const    every cell at the median-depth code
  C-far      the model's own grid with the RUNTIME target cells shifted +8 ranks, recomputed each step
  C-near     the same at −8

The runtime target cells are the offline definition evaluated on the model's own prediction: integrate
the ten translation deltas of the chunk it just produced from the current end-effector position with the
controller's own output scale, project with the banked camera, dilate by one cell. That keeps the closed
loop's "target" the same object as the offline cells' target instead of a second, looser definition.

Episodes are one per task (10 tasks per suite = 10 episodes per arm per suite), fixed before the first
rollout, each from the task's own banked init state 0, so every arm sees exactly the same starting states.

  pg run infra.aws_pg.jobs.depthanchor_loop --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --s3-prefix results/depthanchor/loop/spatial/1 \\
     --budget-min 300 --args " --suite libero_spatial --film"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.pg import depthanchor_core as core

ARM_ORDER = ("C-normal", "C-fresh", "C-stale", "C-shuffle", "C-const", "C-far", "C-near")
TASKS_PER_SUITE = 10


def build_arms(names, ladder, cam_of_task, cell_flips, opengl: bool, notes: Dict[str, object],
               ladder_usable: bool = True):
    from infra.pg import depthanchor_libero as lib
    const_bins = np.full(core.N_CELLS, ladder.median_code, dtype=np.int64)

    def shuffle_edit(natural, frame, step, ctx):
        # one permutation per episode, drawn from the master seed and the task, so the arm is a
        # spatial control and not a new random grid at every step
        rng = core.rng_for("loop-shuffle", str(ctx["task"].language), int(ctx.get("episode", 0)))
        perm = rng.permutation(core.N_CELLS)
        return natural.bins[perm]

    def const_edit(natural, frame, step, ctx):
        return const_bins

    def shift_edit(k: int):
        """C-far / C-near. With a usable code ladder (G2 pass) the target cells move ±8 ranks; without
        one, "k ranks" is undefined and the §8 pivot applies here too — each target cell takes the code
        of the frame's own farthest (for C-far) or nearest (for C-near) cell. Which one ran is recorded."""
        which = "far" if k > 0 else "near"
        def edit(natural, frame, step, ctx):
            cells = lib.runtime_target_cells(ctx["env"], ctx["obs"], natural.a_robot, cam_of_task,
                                             u_flip=cell_flips[0], v_flip=cell_flips[1],
                                             opengl=opengl)
            notes.setdefault("runtime_target_cell_counts", []).append(int(cells.size))
            notes["edit_kind"] = "k_ladder" if ladder_usable else "within_frame_extreme_swap"
            if ladder_usable:
                return core.edit_shift_cells(natural.bins, cells, k, ladder, f"shift{k}").bins
            return core.edit_swap_extreme(natural.bins, cells, which, ladder).bins
        return edit

    built = {
        "C-normal": lib.ArmNormal(),
        "C-fresh": lib.ArmFresh(),
        "C-stale": lib.ArmStale(),
        "C-shuffle": lib.ArmEdit("C-shuffle", shuffle_edit),
        "C-const": lib.ArmEdit("C-const", const_edit, needs_model_bins=False),
        "C-far": lib.ArmEdit("C-far", shift_edit(+8)),
        "C-near": lib.ArmEdit("C-near", shift_edit(-8)),
    }
    return [built[n] for n in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthanchor")
    ap.add_argument("--suite", required=True, choices=list(core.SUITES))
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--tasks", type=int, default=TASKS_PER_SUITE)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--cuda-graph", choices=("on", "off"), default="off",
                    help="off by default: the graph is bit-identical to the eager loop and 1.85x "
                         "faster, and measuring it left the next graph-enabled call raising an illegal "
                         "memory access, so no session runs on it (SPEC A17)")
    ap.add_argument("--film", action="store_true", help="keep frames and depth for task 0 of each arm")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = the suite's own horizon")
    ap.add_argument("--preflight-prefix", required=True,
                    help="read G2's verdict from here: it decides whether C-far/C-near shift ranks or "
                         "swap in the frame's own extreme code (SPEC §8 pivot). Required, because a "
                         "default would silently read a stale preflight")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = sh.Bank(out, f"loop:{args.suite}")
    bank.top("args", vars(args))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_ORDER:
            raise ValueError(f"unknown arm {a!r}; known: {ARM_ORDER}")

    from infra.pg import depthanchor_libero as lib
    env_report = lib.install(extra_numpy=None)
    bank.stage("env", {"install": env_report, "numpy_pin_needed": None})
    smoke_ok = True
    try:
        e, b, t = lib.make_env(args.suite, 0, camera_depths=True, image_size=lib.IMAGE_SIZE)
        e.close()
    except Exception as exc:
        print(f"[loop] simulator import failed without the numpy pin: {exc}", flush=True)
        smoke_ok = False
    if not smoke_ok:
        env_report = lib.install(extra_numpy="numpy<2")
        e, b, t = lib.make_env(args.suite, 0, camera_depths=True, image_size=lib.IMAGE_SIZE)
        e.close()
    bank.stage("env", {"install": env_report, "numpy_pin_needed": not smoke_ok})

    cam = json.loads(sh.pull_camera_bank(work).read_text())
    orient = json.loads(sh.pull_json(f"{sh.S3_CAMERA}/orientation.json",
                                     work / "orientation.json").read_text())
    # Every convention read from the banked artifact, none defaulted: the transform that makes a
    # RENDERED image look like the DATASET image the checkpoint was trained on (`render_flip`,
    # `render_hflip`), the pixel-sign convention (`opengl`), and where a projected point lands in a
    # dataset image, which is what the target cells are indexed by (`dataset_u_flip`, `dataset_v_flip`).
    for key in ("render_flip", "render_hflip", "opengl", "dataset_u_flip", "dataset_v_flip"):
        if orient.get(key) is None:
            raise RuntimeError(f"the banked orientation artifact has no `{key}`; run the preflight's "
                               "orient, pack and loop stages first")
    flip = bool(orient["render_flip"])
    hflip = bool(orient["render_hflip"])
    opengl = bool(orient["opengl"])
    ds_u, ds_v = bool(orient["dataset_u_flip"]), bool(orient["dataset_v_flip"])
    ladder, _ = sh.load_ladder(work)
    preflight = json.loads(sh.pull_json(
        f"s3://{sh.BUCKET}/{args.preflight_prefix}/out/metrics.json",
        work / "preflight_metrics.json").read_text())
    g2 = (((preflight.get("stages") or {}).get("refs") or {}).get("gate_G2") or {}).get("verdict")
    ladder_usable = g2 == "pass"
    bank.stage("inputs", {"render_flip": flip, "render_hflip": hflip, "opengl": opengl,
                          "dataset_u_flip": ds_u, "dataset_v_flip": ds_v,
                          "render_transform_decided_by": orient.get("render_transform_decided_by"),
                          "gate_G2": g2, "ladder_usable": ladder_usable,
                          "edit_kind": "k_ladder" if ladder_usable else "within_frame_extreme_swap",
                          "chosen_convention": orient.get("chosen_convention"),
                          "gate_G3": orient.get("gate_G3", {}).get("verdict"),
                          "ladder_median_code": ladder.median_code, "arms": arms,
                          "suite": args.suite, "horizon": lib.HORIZON[args.suite],
                          "settle_steps": lib.SETTLE_STEPS})

    from infra.pg import depthanchor_model as dam
    model, processor, path = dam.load_model()
    actor = dam.Actor(model, processor, enable_cuda_graph=(args.cuda_graph == "on"))
    bank.stage("model", {"snapshot": str(path), "cuda_graph": args.cuda_graph})

    t0 = time.perf_counter()
    rows: List[Dict[str, object]] = []
    films: Dict[str, object] = {}
    notes: Dict[str, object] = {}
    for arm_name in arms:
        for task_id in range(args.tasks):
            cam_of_task = cam["tasks"][f"{args.suite}/{task_id}"]
            arm = build_arms([arm_name], ladder, cam_of_task, (ds_u, ds_v), opengl, notes,
                             ladder_usable=ladder_usable)[0]
            for ep in range(args.episodes_per_task):
                film = bool(args.film and task_id == 0 and ep == 0)
                r = lib.rollout(actor, args.suite, task_id=task_id, episode=ep, policy=arm,
                                flip=flip, hflip=hflip, film=film, want_depth=film,
                                max_steps=args.max_steps or None)
                rec = r.as_record()
                rec["episode_index_used"] = ep
                rows.append(rec)
                print(f"[loop] {arm_name} {args.suite}/{task_id} ep{ep}: success={r.success} "
                      f"steps={r.n_steps} calls={rec['meta']['calls']} {r.seconds:.0f}s", flush=True)
                if film and r.frames:
                    films[arm_name] = {"frames": np.stack(r.frames),
                                      "depth": np.stack(r.depth_maps) if r.depth_maps else None,
                                      "eef": np.asarray(r.eef_path, dtype=np.float32),
                                      "bins": np.asarray(r.bins_per_call, dtype=np.int16)}
                bank.stage("progress", {"rollouts_done": len(rows), "arm": arm_name,
                                        "elapsed_s": round(time.perf_counter() - t0, 1),
                                        "seconds_per_rollout": round(
                                            (time.perf_counter() - t0) / len(rows), 1)})
                bank.artifact("rollouts.json", [{k: v for k, v in x.items()
                                                 if k not in ("bins_per_call", "gate_per_call")}
                                                for x in rows])

    if films:
        payload = {}
        for arm_name, f in films.items():
            for key, value in f.items():
                if value is not None:
                    payload[f"{arm_name}:{key}"] = value
        np.savez_compressed(out / "films.npz", **payload)

    summary = _summarize(rows, args.suite, notes)
    bank.stage("summary", summary)
    bank.top("verdict", {"suite": args.suite,
                         "by_arm": {a: summary["by_arm"][a]["rate"] for a in summary["by_arm"]},
                         "confirmation": summary["confirmation"],
                         "seconds_total": round(time.perf_counter() - t0, 1),
                         "calls": actor.calls, "rollouts": len(rows)})
    print(json.dumps(bank.payload["verdict"], indent=1, default=str), flush=True)


def _summarize(rows, suite: str, notes: Dict[str, object]) -> Dict[str, object]:
    by_arm: Dict[str, Dict[str, object]] = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm]
        ok = sum(1 for r in mine if r["success"])
        rate, lo, hi = core.wilson(ok, len(mine))
        by_arm[arm] = {"successes": ok, "n": len(mine), "rate": rate, "ci95": [lo, hi],
                       "median_steps": float(np.median([r["n_steps"] for r in mine])),
                       "median_seconds": float(np.median([r["seconds"] for r in mine])),
                       "tail_token_max": int(max((max(r["tail_per_call"]) for r in mine
                                                 if r["tail_per_call"]), default=0)),
                       "done_check_disagreements": int(sum(
                           1 for r in mine if r["meta"].get("agree") is False)),
                       # per arm, not just per session: an arm whose edits were refused on many steps
                       # ran the model's own grid instead of its own condition and is not that arm
                       "edits_not_completed": int(sum(
                           int((r["meta"] or {}).get("edits_not_completed") or 0) for r in mine)),
                       # the checkpoint closing its own depth block before the grid is full: no edit is
                       # defined on a short grid, so that step ran the model's own and is counted here
                       "short_natural_grids": int(sum(
                           int((r["meta"] or {}).get("short_natural_grids") or 0) for r in mine)),
                       "calls_with_short_grid": int(sum(
                           int((r["meta"] or {}).get("n_short_grid_calls") or 0) for r in mine)),
                       "steps_without_target_cells": int(sum(
                           int((r["meta"] or {}).get("steps_without_target_cells") or 0)
                           for r in mine)),
                       "calls": int(sum(int((r["meta"] or {}).get("calls") or 0) for r in mine))}
    normal = by_arm.get("C-normal")
    confirmation = None
    if normal:
        confirmation = core.closed_loop_confirmation(
            (normal["rate"], normal["ci95"][0], normal["ci95"][1]),
            {a: (v["rate"], v["ci95"][0], v["ci95"][1]) for a, v in by_arm.items() if a != "C-normal"},
            label="H_geo")
        confirmation["note"] = ("`label` is passed as H_geo so every clause is evaluated and reported; "
                               "the offline label decides which clause is the confirmation, and that "
                               "reading is done in the closeout against the contrast session's label.")
    return {"suite": suite, "by_arm": by_arm, "confirmation": confirmation,
            "n_rollouts": len(rows), "edit_kind": notes.get("edit_kind"),
            "edits_not_completed": int(sum(int((r["meta"] or {}).get("edits_not_completed") or 0)
                                           for r in rows)),
            "short_natural_grids": int(sum(int((r["meta"] or {}).get("short_natural_grids") or 0)
                                           for r in rows)),
            "calls_with_short_grid": int(sum(int((r["meta"] or {}).get("n_short_grid_calls") or 0)
                                             for r in rows)),
            "steps_without_target_cells": int(sum(
                int((r["meta"] or {}).get("steps_without_target_cells") or 0) for r in rows)),
            "runtime_target_cells": {
                "n": len(notes.get("runtime_target_cell_counts") or []),
                "median": (float(np.median(notes["runtime_target_cell_counts"]))
                           if notes.get("runtime_target_cell_counts") else None),
                "min": (int(min(notes["runtime_target_cell_counts"]))
                        if notes.get("runtime_target_cell_counts") else None),
                "max": (int(max(notes["runtime_target_cell_counts"]))
                        if notes.get("runtime_target_cell_counts") else None)}}


if __name__ == "__main__":
    main()
