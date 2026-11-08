"""DEPTHANCHOR preflight — spec §9.1/§9.2 and everything the frozen rules will need to exist.

Ten stages, each banked into `out/metrics.json` as it finishes, each runnable alone through
`--stages`, so a failure costs one stage and not the instance-hour:

  env      the simulator installed and PROVED to work, with the numpy decision measured rather than
           assumed (robosuite 1.4.0 predates numpy 2; the pin is applied only if a smoke test needs it)
  cam      the 40 LIBERO tasks' agentview intrinsics/extrinsics, banked; the suite of every task string
  pack     the 120 frames, drawn from the master seed, read out of the seeded parquet tree, with their
           future end-effector paths projected to target cells; pushed to S3 for every later cell
  model    the checkpoint loaded, the card's own example reproduced, and the WEIGHTS-ONLY readout of
           `action_expert_depth_gate` (config init bias -4.0 -> sigmoid 0.018: how much of the depth
           prefix reaches the action expert at all is a fact about the safetensors)
  latency  per-call cost, and two instrument checks — CUDA-graph replay against the eager loop at the
           same seed, and two identical calls against each other
  g0       THE pass-through gate: replaying the model's own grid must reproduce the natural call; plus
           the seed floor, which must be non-zero or the whole design has no denominator
  refs     the released VQ-VAE and Depth-Anything: the code ladder (G2) and depth fidelity (G1)
  orient   G3: which image orientation and which pixel-sign convention agree with the rendered depth
           buffer, measured over all four combinations plus a deliberately wrong camera. Runs BEFORE
           the pack, because the pack's target cells are the output of exactly this projection
  polarity G1B: the SIGN of decoded depth against metric ground truth, which the directional clause of
           §6 rule 2b depends on; measured before any directional number exists
  loop     five episodes of LIBERO-Object under the released deployment path; the spec requires at
           least four successes before anything else is trusted (the released card reports 100.0)

  pg run infra.aws_pg.jobs.depthanchor_preflight --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO,models--allenai--MolmoAct-7B-D-0812 \\
     --s3-prefix results/depthanchor/preflight/1 --budget-min 180
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import traceback
from typing import Dict, List, Optional, Sequence

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.pg import depthanchor_core as core
from infra.pg import depthanchor_frames as frames_mod

ALL_STAGES = ("env", "cam", "orient", "model", "vertical", "pack", "latency", "g0", "refs",
              "polarity", "loop")
PROJECTION_TOL_M = 0.15      # physical: the eef site sits between the fingers
PROJECTION_TOL_PX = 1.0      # numerical: agreement with robosuite's own projection
G0_FRAMES = 20
FIDELITY_FRAMES = 20
VERTICAL_FRAMES = 12
VERTICAL_TASKS = 6
LOOP_EPISODES = 5
LOOP_SUITE = "libero_object"
CARD_TASK = ("put the white mug on the left plate and put the yellow and white mug on the right plate")
CARD_STATE = [-0.05338004603981972, 0.007029631175100803, 0.6783280968666077, 3.1407692432403564,
              0.0017593271331861615, -0.08994418382644653, 0.03878866136074066, -0.03878721222281456]


# ------------------------------------------------------------------ env
def stage_env(bank: sh.Bank) -> Dict[str, object]:
    from infra.pg import depthanchor_libero as lib
    report: Dict[str, object] = {}
    attempt = lib.install(extra_numpy=None)
    report["install_no_pin"] = attempt
    bank.stage("env", dict(report))          # banked BEFORE the smoke, so a crash leaves the install visible
    smoke = _env_smoke(lib)
    report["smoke_no_pin"] = smoke
    bank.stage("env", dict(report))
    if not smoke["ok"]:
        # Measured, not assumed: robosuite 1.4.0 predates numpy 2. The failure is printed in full and
        # banked verbatim before the pin is applied, so the reason for the pin is in the artifact.
        print("[env] simulator smoke FAILED without the numpy pin; error banked, applying numpy<2",
              flush=True)
        report["install_pinned"] = lib.install(extra_numpy="numpy<2")
        smoke = _env_smoke(lib)
        report["smoke_pinned"] = smoke
        bank.stage("env", dict(report))
        if not smoke["ok"]:
            raise RuntimeError(f"the LIBERO simulator does not run with or without the numpy pin: "
                               f"{smoke['error']}")
    import pyarrow
    import PIL
    import torch
    import transformers
    report["after_install"] = {"pyarrow": pyarrow.__version__, "pillow": PIL.__version__,
                              "torch": torch.__version__, "numpy": np.__version__,
                              "transformers": transformers.__version__,
                              "cuda_available": bool(torch.cuda.is_available()),
                              "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; this job cannot run on CPU")
    return report


def _env_smoke(lib) -> Dict[str, object]:
    try:
        env, bench, task = lib.make_env("libero_spatial", 0, camera_depths=True, image_size=lib.IMAGE_SIZE)
        try:
            obs = env.reset()
            obs, _, _, _ = env.step(lib.NOOP)
            keys = sorted(k for k in obs if "image" in k or "depth" in k or "eef" in k or "gripper" in k)
            frame = lib.obs_to_frame(obs, flip=True, task=task.language)
            depth = lib.real_depth(env, obs, flip=True)
            return {"ok": True, "obs_keys": keys, "language": str(task.language),
                    "agentview_shape": list(np.asarray(frame["agentview"]).shape),
                    "state": [float(x) for x in frame["state"]],
                    "depth_min_m": float(np.min(depth)), "depth_max_m": float(np.max(depth)),
                    "n_init_states": int(len(bench.get_task_init_states(0)))}
        finally:
            env.close()
    except Exception as exc:  # the error is the measurement here; it is printed and banked, not hidden
        traceback.print_exc()
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]}


# ------------------------------------------------------------------ cam
def stage_cam(bank: sh.Bank) -> Dict[str, object]:
    from infra.pg import depthanchor_libero as lib
    t0 = time.perf_counter()
    cam = lib.camera_bank()
    path = bank.artifact("camera_bank.json", cam)
    sh.push_camera_bank(path)
    first = cam["tasks"]["libero_spatial/0"]
    return {"n_tasks": len(cam["tasks"]), "seconds": time.perf_counter() - t0,
            "image_size": cam["image_size"],
            "example": {k: first[k] for k in ("suite", "task_id", "language", "K",
                                              "cam_pos_world", "view_dir_world", "znear", "zfar",
                                              "n_init_states")},
            "suites": sorted({v["suite"] for v in cam["tasks"].values()})}


# ------------------------------------------------------------------ pack

# ------------------------------------------------------------------ which way is up in a dataset image
def stage_vertical(bank: sh.Bank, state: Dict[str, object], work: pathlib.Path) -> Dict[str, object]:
    """WHICH WAY IS UP in a dataset image, measured from DEPTH instead of from pixel intensities or from
    where a projected point lands.

    Three probes disagreed about this one bit, so it gets its own stage and its own instrument (A18):

      * the row profile of dataset frames against rendered ones said the dataset is the VERTICAL MIRROR
        of the raw buffer (+0.662 mirrored against -0.469 raw) — but it correlates scenes that do not
        match, because the released episodes do not start from LIBERO's shipped init states;
      * the motion percentile said no vertical mirror (horizontal mirror 0.925 against 180 deg 0.68) — but
        the LIBERO arm is a tall structure, so the vertical mirror of the gripper's pixel often lands on
        the arm as well, which biases exactly this bit;
      * `preflight/12`'s polarity stage said the model reads our transformed render UPSIDE DOWN: its own
        decoded grid agrees with DepthAnything-2 on dataset frames (+0.966) and is INVERTED on the
        rendered frames we built (-0.805) while DepthAnything-2 on the same image is right (-0.968 against
        metric truth). DepthAnything-2 is orientation-robust; a depth head trained on upright frames is
        not.

    The measurement here needs no scene match and no vision-model assumption about orientation. A
    tabletop scene's depth is monotone in the image row: the near edge of the table is at one end and the
    background at the other. So the SIGN of (depth, row) tells which end is which, in each image source:

      dataset image  ->  DepthAnything-2 disparity per cell, and the MODEL's own decoded grid per cell
      raw render     ->  the metric depth buffer per cell, as a disparity (-metres)

    If the two sources disagree in sign, a rendered image has to be flipped vertically to look like a
    dataset image, which is exactly `render_flip`. `dataset_v_flip` follows by XOR with `depth_v_flip`.
    """
    from infra.pg import depthanchor_libero as lib
    cam = json.loads((bank.out / "camera_bank.json").read_text())
    orient = _orientation(bank)
    actor = _need(state, "actor", "model")
    refs = state.get("refs")
    if refs is None:
        from infra.pg import depthanchor_model as dam
        refs = dam.load_depth_refs()
        state["refs"] = refs
    pack_dir = bank.out / "pack"
    if not (pack_dir / "frames_index.json").is_file():
        sh.pull_pack(pack_dir)
    pack = frames_mod.load_pack(pack_dir)
    rows_cell = np.repeat(np.arange(core.GRID), core.GRID).astype(np.float64)   # cell row index, 0 at top

    dataset_rows = []
    for f in pack["frames"][:VERTICAL_FRAMES]:
        da2 = _cell_means(refs.depth_anything_map(f["agentview"]))
        nat = actor.natural(f, seed=0)
        decoded = refs.cell_depths(nat.bins)
        dataset_rows.append({"frame_id": f["spec"]["frame_id"], "suite": f["spec"]["suite"],
                             "task_key": f["spec"]["task_key"],
                             "spearman_da2_vs_row": core.spearman(da2, rows_cell),
                             "spearman_decoded_vs_row": core.spearman(decoded, rows_cell),
                             "spearman_decoded_vs_da2": core.spearman(decoded, da2)})

    render_rows = []
    for key in sorted({r["task_key"] for r in dataset_rows})[:VERTICAL_TASKS]:
        entry = cam["tasks"][key]
        env, bench, task = lib.make_env(entry["suite"], entry["task_id"], camera_depths=True,
                                        image_size=lib.IMAGE_SIZE)
        try:
            init_states = bench.get_task_init_states(entry["task_id"])
            env.reset()
            env.set_init_state(init_states[0])
            for _ in range(lib.SETTLE_STEPS):
                obs, _, _, _ = env.step(lib.NOOP)
            metres = _cell_means(lib.real_depth(env, obs, flip=False))
            raw_rgb = np.asarray(obs["agentview_image"], dtype=np.uint8)
            da2_raw = _cell_means(refs.depth_anything_map(raw_rgb))
            render_rows.append({"task_key": key,
                                "spearman_disparity_vs_row": core.spearman(-metres, rows_cell),
                                "spearman_da2_vs_row": core.spearman(da2_raw, rows_cell),
                                "spearman_da2_vs_metric": core.spearman(da2_raw, -metres)})
        finally:
            env.close()

    ds_da2 = float(np.median([r["spearman_da2_vs_row"] for r in dataset_rows]))
    ds_model = float(np.median([r["spearman_decoded_vs_row"] for r in dataset_rows]))
    rn_metric = float(np.median([r["spearman_disparity_vs_row"] for r in render_rows]))
    rn_da2 = float(np.median([r["spearman_da2_vs_row"] for r in render_rows]))
    # the render's own two references must agree with each other, or the metric buffer is not the image's
    control_ok = bool(np.sign(rn_metric) == np.sign(rn_da2)
                      and float(np.median([r["spearman_da2_vs_metric"] for r in render_rows])) > 0.5)
    render_flip = bool(np.sign(ds_da2) != np.sign(rn_metric))
    agree = bool(np.sign(ds_da2) == np.sign(ds_model))
    ds_v = bool(render_flip != bool(orient.get("depth_v_flip", False)))
    bank.artifact("vertical_rows.json", {"dataset": dataset_rows, "render": render_rows})
    payload = {
        "median_spearman_dataset_da2_vs_row": ds_da2,
        "median_spearman_dataset_model_vs_row": ds_model,
        "median_spearman_render_metric_disparity_vs_row": rn_metric,
        "median_spearman_render_da2_vs_row": rn_da2,
        "median_spearman_render_da2_vs_metric": float(np.median(
            [r["spearman_da2_vs_metric"] for r in render_rows])),
        "dataset_references_agree": agree,
        "render_references_agree": control_ok,
        "render_flip": render_flip,
        "dataset_v_flip": ds_v,
        "depth_v_flip": bool(orient.get("depth_v_flip", False)),
        "n_dataset_frames": len(dataset_rows), "n_render_tasks": len(render_rows),
        "decisive": bool(agree and control_ok and min(abs(ds_da2), abs(rn_metric)) > 0.3),
    }
    _update_orientation(bank, vertical=payload)
    return payload


def stage_pack(bank: sh.Bank, work: pathlib.Path) -> Dict[str, object]:
    """The 120 frames, in three steps whose order is the whole point: read the frames, decide from the
    DATASET whether the image the model sees is the rendered buffer or its mirror, and only then project
    the future end-effector path into target cells."""
    from infra.pg import depthanchor_libero as lib
    cam = json.loads((bank.out / "camera_bank.json").read_text())
    orient = _orientation(bank)
    meta_dir = sh.pull_dataset_meta(work / "lds")
    meta = frames_mod.read_meta(meta_dir)
    task_strings = frames_mod.read_task_strings(meta_dir)
    row_index = _row_index(work)
    mismatch = _row_index_vs_meta(row_index, meta)
    print(f"[pack] row index vs meta/episodes: {mismatch['episodes_with_wrong_file']} of "
          f"{mismatch['n_episodes']} episodes are assigned to the wrong file by meta", flush=True)
    specs = frames_mod.select_frames(row_index, task_strings, cam["by_language"])
    template = str(row_index["data_path"])
    rels = sorted({template.format(file_index=int(sp.file_index)) for sp in specs})
    print(f"[pack] {len(specs)} frames need {len(rels)} of {row_index['n_files']} parquet files",
          flush=True)
    sh.pull_dataset_files(work / "lds", rels)
    frames = frames_mod.read_frames(specs, row_index, work / "lds")

    # WHICH VERTICAL CONVENTION the model's image uses is decided on the DATASET's own frames, by the
    # projection check that licenses the target cells (`motion_check`). The simulator probe runs too and
    # is banked, but it cannot decide this: the released episodes do not start from LIBERO's shipped
    # init states, so no rendered init state reproduces a dataset frame.
    motion = frames_mod.motion_check(frames, cam, opengl=bool(orient["opengl"]))
    agree = _dataset_orientation(lib, cam, frames)
    bank.artifact("dataset_agreement.json", agree)
    bank.artifact("motion_check.json", motion)
    if not motion.get("decisive"):
        raise RuntimeError("the projection does not land on the moving arm under any of the four "
                           "dihedral conventions, so the target cells cannot be trusted: "
                           f"{ {k: v for k, v in motion.items() if k != 'per_frame'} }")
    # THE TARGET CELLS are indexed in the DATASET's images, so they use the dataset's own flips. The
    # HORIZONTAL bit is the motion check's: it separates u=True (0.925 / 0.68) from u=False (0.43 / 0.38)
    # by a wide margin. The VERTICAL bit is the `vertical` stage's, measured from depth, because the
    # motion percentile is biased on exactly that bit — the arm is a tall structure, so the vertical
    # mirror of the gripper's pixel often lands on the arm too (A18). Both are banked, and so is the
    # disagreement.
    vertical = (_orientation(bank).get("vertical") or {})
    if not vertical.get("decisive"):
        raise RuntimeError("the `vertical` stage did not decide which way up a dataset image is, and the "
                           f"motion check cannot decide it alone: {vertical}")
    ds_u = bool(motion["u_flip"])
    ds_v = bool(vertical["dataset_v_flip"])
    motion["v_flip_from_motion"] = bool(motion["v_flip"])
    motion["v_flip_used"] = ds_v
    motion["v_flip_disagrees_with_depth_probe"] = bool(bool(motion["v_flip"]) != ds_v)
    # THE RENDER TRANSFORM for the closed loop is the difference between where the projection lands in a
    # dataset image and where it lands in a rendered one: rendered -> dataset. XOR, because each flip is
    # its own inverse.
    render_hflip = bool(ds_u != bool(orient.get("depth_u_flip", False)))
    render_flip = bool(vertical["render_flip"])
    _update_orientation(bank, render_flip=render_flip, render_hflip=render_hflip,
                        dataset_u_flip=ds_u, dataset_v_flip=ds_v,
                        motion_check={k: v for k, v in motion.items() if k != "per_frame"},
                        dataset_agreement={k: v for k, v in agree.items() if k != "per_task"})

    pack = frames_mod.attach_cells(frames, cam, opengl=bool(orient["opengl"]),
                                   v_flip=ds_v, u_flip=ds_u)
    bank.artifact("row_index_vs_meta.json", mismatch)
    payload = frames_mod.save_pack(pack, bank.out / "pack")
    sh.push_pack(bank.out / "pack")
    n_target = [int(np.asarray(f["target_cells"]).size) for f in pack["frames"]]
    depths = np.concatenate([np.asarray(f["future_depth"]) for f in pack["frames"]])
    return {"n_frames": pack["n_frames"], "n_files_pulled": len(rels),
            "row_index_vs_meta": mismatch,
            "motion_check": {k: v for k, v in motion.items() if k != "per_frame"},
            "vertical": vertical,
            "dataset_u_flip": ds_u, "dataset_v_flip": ds_v,
            "render_hflip": render_hflip, "render_vflip": render_flip,
            "candidates": pack.get("candidates"),
            "skipped_per_suite": pack.get("skipped_per_suite"),
            "n_skipped": len(pack.get("skipped") or []),
            "projection_opengl": pack["projection_opengl"],
            "projection_v_flip": pack["projection_v_flip"],
            "dataset_agreement": {k: v for k, v in agree.items() if k != "per_task"},
            "per_suite": {s: sum(1 for f in pack["frames"] if f["spec"]["suite"] == s)
                          for s in core.SUITES},
            "n_target_cells": {"min": int(min(n_target)), "median": float(np.median(n_target)),
                               "max": int(max(n_target))},
            "future_depth_m": {"min": float(depths.min()), "median": float(np.median(depths)),
                               "max": float(depths.max())},
            "bg_min_distance_counts": _counts([int(f["bg_min_distance"]) for f in pack["frames"]]),
            "first_frame_ids": [f["spec"]["frame_id"] for f in pack["frames"][:6]],
            "index_entries": len(payload["index"])}


def _profile_corr(rendered: Sequence[np.ndarray], dataset_image: np.ndarray) -> float:
    """Correlation between one dataset image's mean row profile and the MEAN row profile of a set of
    rendered images in a given orientation. Init-state independent: it reads the scene's vertical
    structure, not its contents."""
    def prof(img):
        a = np.asarray(img, dtype=np.float64)
        return a.mean(axis=(1, 2)) if a.ndim == 3 else a.mean(axis=1)
    ref = np.mean([prof(r) for r in rendered], axis=0)
    mine = prof(dataset_image)
    ref = ref - ref.mean()
    mine = mine - mine.mean()
    den = float(np.sqrt((ref @ ref) * (mine @ mine)))
    return 0.0 if den < 1e-12 else float((ref @ mine) / den)


def _row_index(work: pathlib.Path) -> Dict[str, object]:
    """The banked row index, in order of preference: the copy committed in the repo, the copy in S3, or
    built here from the data files themselves and pushed. Building it in-region takes about a minute for
    378 files; from a laptop it takes half an hour, which is why the job can do it."""
    if frames_mod.ROW_INDEX.is_file():
        print(f"[pack] row index from the repo: {frames_mod.ROW_INDEX}", flush=True)
        return frames_mod.load_row_index()
    uri = f"{sh.S3_ROOT}/rowindex/v1/depthanchor_rowindex.json"
    local = work / "depthanchor_rowindex.json"
    if sh.s3_exists(uri):
        print(f"[pack] row index from {uri}", flush=True)
        sh.pull_json(uri, local)
        return frames_mod.load_row_index(local)
    print("[pack] building the row index from the data files (in-region scan of 378 files)", flush=True)
    import pyarrow.fs as pafs
    fs = pafs.S3FileSystem(region=sh.REGION)
    base = f"{sh.BUCKET}/depthanchor/liberods/v1/data/chunk-000"
    payload = frames_mod.build_row_index(
        open_parquet=lambda k: fs.open_input_file(f"{base}/file-{k:03d}.parquet"),
        n_files=378, data_path="data/chunk-000/file-{file_index:03d}.parquet",
        source=f"s3://{sh.BUCKET}/depthanchor/liberods/v1")
    local.write_text(json.dumps(payload))
    sh.push_json(local, uri)
    print(f"[pack] row index built and pushed to {uri}", flush=True)
    return frames_mod.load_row_index(local)


def _row_index_vs_meta(row_index: Dict[str, object], meta: Dict[str, object]) -> Dict[str, object]:
    """Quantify the released metadata's row-location defect, because "meta is wrong" is a claim and this
    is its number. Compares the banked row index — built from the data files' own columns — against
    `meta/episodes`' `data/file_index` and `dataset_from_index` for every episode."""
    eps = meta["episodes"]
    by_ep = row_index["by_episode"]
    wrong_file = wrong_start = missing = 0
    examples = []
    for i, e in enumerate(eps["episode_index"]):
        rec = by_ep.get(int(e))
        if rec is None:
            missing += 1
            continue
        bad_file = int(eps["data/file_index"][i]) != int(rec["file_index"])
        bad_start = int(eps["dataset_from_index"][i]) != int(rec["global_start"])
        wrong_file += int(bad_file)
        wrong_start += int(bad_start)
        if (bad_file or bad_start) and len(examples) < 5:
            examples.append({"episode": int(e), "meta_file": int(eps["data/file_index"][i]),
                             "actual_file": int(rec["file_index"]),
                             "meta_global_start": int(eps["dataset_from_index"][i]),
                             "actual_global_start": int(rec["global_start"])})
    return {"n_episodes": len(eps["episode_index"]), "episodes_with_wrong_file": wrong_file,
            "episodes_with_wrong_global_start": wrong_start,
            "episodes_absent_from_row_index": missing,
            "meta_file_index_max": int(max(eps["data/file_index"])),
            "actual_n_files": int(row_index["n_files"]), "examples": examples,
            "note": ("row locations come from infra/pg/depthanchor_rowindex.json, built from the data "
                     "files' own episode_index/frame_index/index columns; meta/episodes is used only "
                     "for provenance")}


def _dataset_orientation(lib, cam: Dict[str, object], frames: Sequence[Dict[str, object]],
                         n_tasks: int = 3, n_episodes: int = 2) -> Dict[str, object]:
    """Is the dataset's agentview image the rendered buffer, or its vertical mirror? Render every init
    state of a task and correlate each dataset episode's OWN first frame against both orientations. The
    dataset defines what the checkpoint was trained on, so this — not a convention from a docstring —
    is what decides the orientation the model is fed and therefore which row of the image a depth cell
    covers. Also reports the 8-d state agreement at the matched init state, which is the same question
    asked of the state layout."""
    per_task = []
    keys = sorted({f["spec"]["task_key"] for f in frames})[:n_tasks]
    for key in keys:
        entry = cam["tasks"][key]
        mine = [f for f in frames if f["spec"]["task_key"] == key][:n_episodes]
        env, bench, task = lib.make_env(entry["suite"], entry["task_id"], camera_depths=False,
                                        image_size=int(cam["image_size"]))
        rows = []
        try:
            init_states = bench.get_task_init_states(entry["task_id"])
            rendered = []
            for i in range(len(init_states)):
                env.reset()
                env.set_init_state(init_states[i])
                obs, _, _, _ = env.step(lib.NOOP)
                raw = np.asarray(obs["agentview_image"])
                rendered.append((raw.copy(), raw[::-1].copy(),
                                 lib.obs_to_frame(obs, flip=False, task=task.language)["state"]))
            for f in mine:
                ds = np.asarray(f["first_agentview"], dtype=np.float64)
                c_raw = [lib._corr(r[0], ds) for r in rendered]
                c_mirror = [lib._corr(r[1], ds) for r in rendered]
                b_raw, b_mirror = int(np.argmax(c_raw)), int(np.argmax(c_mirror))
                best_is_mirror = c_mirror[b_mirror] > c_raw[b_raw]
                b = b_mirror if best_is_mirror else b_raw
                rows.append({
                    "frame_id": f["spec"]["frame_id"], "episode": f["spec"]["episode"],
                    "n_init_states": len(init_states), "best_init_state": b,
                    "corr_raw": float(c_raw[b_raw]), "corr_mirror": float(c_mirror[b_mirror]),
                    "second_best": float(sorted(c_mirror if best_is_mirror else c_raw)[-2]),
                    "dataset_is_mirror_of_render": bool(best_is_mirror),
                    "row_profile_corr_raw": float(_profile_corr(
                        [r[0] for r in rendered], ds)),
                    "row_profile_corr_mirror": float(_profile_corr(
                        [r[1] for r in rendered], ds)),
                    "state_max_abs_diff": float(np.max(np.abs(
                        np.asarray(rendered[b][2], dtype=np.float64)
                        - np.asarray(f["first_state"], dtype=np.float64))))})
        finally:
            env.close()
        per_task.append({"task_key": key, "episodes": rows})
    allrows = [r for t in per_task for r in t["episodes"]]
    if not allrows:
        return {"render_flip": None, "why": "no dataset frames to compare", "per_task": per_task}
    votes = [r["dataset_is_mirror_of_render"] for r in allrows]
    best = [max(r["corr_raw"], r["corr_mirror"]) for r in allrows]
    unanimous = len(set(votes)) == 1
    strong = float(np.median(best)) > 0.95
    # A SECOND, WEAKER SIGNAL that needs no matched init state: the mean row profile. The scene's
    # vertical structure (table low, background high) is the same whatever the objects are doing, so
    # correlating the dataset frames' mean row profile against the rendered frames' tells the two
    # orientations apart even if no init state reproduces a particular episode. Used only when the
    # per-episode match is inconclusive, and banked either way so a reader can see both.
    profile = _row_profile_vote(per_task)
    render_flip = bool(votes[0]) if (unanimous and strong) else profile["render_flip"]
    return {"render_flip": render_flip, "decided_by": ("episode_match" if (unanimous and strong)
                                                       else "row_profile"),
            "row_profile": profile,
            "unanimous": unanimous, "median_best_corr": float(np.median(best)),
            "median_corr_raw": float(np.median([r["corr_raw"] for r in allrows])),
            "median_corr_mirror": float(np.median([r["corr_mirror"] for r in allrows])),
            "median_second_best": float(np.median([r["second_best"] for r in allrows])),
            "state_max_abs_diff_median": float(np.median([r["state_max_abs_diff"] for r in allrows])),
            "state_max_abs_diff_max": float(np.max([r["state_max_abs_diff"] for r in allrows])),
            "n_episodes": len(allrows), "same_camera": strong, "per_task": per_task}


def _counts(values: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


# ------------------------------------------------------------------ model
def stage_model(bank: sh.Bank, state: Dict[str, object]) -> Dict[str, object]:
    from infra.pg import depthanchor_model as dam
    import torch
    t0 = time.perf_counter()
    model, processor, path = dam.load_model()
    load_seconds = time.perf_counter() - t0
    # eager, like every other Actor in this study: the CUDA graph is bit-identical and 1.85x faster and
    # is still not used, because measuring it once left the next graph-enabled call raising an illegal
    # memory access (A17, `stage_latency`)
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    state["model"], state["processor"], state["actor"], state["model_path"] = model, processor, actor, path
    prov = dam.provenance(model, processor, path)
    gate = dam.gate_weights_readout(model)
    bank.artifact("provenance.json", {"provenance": prov, "depth_gate_weights": gate})

    # the card's own example, verbatim, as a smoke test with a banked artifact
    from PIL import Image
    frame = {"agentview": np.asarray(Image.open(path / "assets" / "sample_agentview_rgb.png").convert("RGB")),
             "wrist": np.asarray(Image.open(path / "assets" / "sample_wrist_rgb.png").convert("RGB")),
             "task": CARD_TASK, "state": np.asarray(CARD_STATE, dtype=np.float32),
             "spec": {"frame_id": "card_example", "suite": "card", "task_key": "card/0"}}
    res = actor.natural(frame, seed=0)
    _check_action_sanity(res)
    card = {"bins": res.bins.astype(int).tolist(), "n_grid_tokens": res.n_grid_tokens,
            "n_tail_tokens": res.n_tail_tokens, "seconds": res.seconds,
            "a_robot": res.a_robot.tolist(), "a_norm": res.a_norm.tolist(),
            "gate_trained": res.meta.get("gate_trained"),
            "depth_positions": res.meta.get("depth_positions"),
            "unique_codes": int(len(set(res.bins.tolist())))}
    bank.artifact("card_example.json", card)
    # the recording wrapper must be exactly invertible: unnormalize(captured) == what was returned
    stats = model._get_robot_stats()
    reconstructed = np.asarray(stats.unnormalize_action(res.a_norm, dam.NORM_TAG), dtype=np.float64)
    recording_error = float(np.max(np.abs(reconstructed - res.a_robot)))
    return {"load_seconds": load_seconds, "snapshot": str(path), "provenance": prov,
            "depth_gate_weights": gate, "card_example": {k: card[k] for k in
                ("bins", "n_grid_tokens", "n_tail_tokens", "seconds", "unique_codes",
                 "depth_positions")},
            "card_action_first_step": res.a_robot[0].tolist(),
            "recording_wrapper_max_abs_error": recording_error,
            "recording_wrapper_ok": bool(recording_error < 1e-5),
            "cuda_max_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9)}


def _check_action_sanity(res) -> None:
    if res.bins.shape != (core.N_CELLS,):
        raise RuntimeError(f"the model returned {res.bins.shape} depth bins, expected {core.N_CELLS}")
    if res.a_robot.shape != (core.CHUNK, core.ACTION_DIM):
        raise RuntimeError(f"action chunk {res.a_robot.shape}, expected ({core.CHUNK},{core.ACTION_DIM})")
    if not np.all(np.isfinite(res.a_robot)) or not np.all(np.isfinite(res.a_norm)):
        raise RuntimeError("the action chunk contains a non-finite value")
    if float(np.max(np.abs(res.a_robot))) > 5.0:
        raise RuntimeError(f"action magnitude {float(np.max(np.abs(res.a_robot))):.3f} is outside any "
                           "plausible LIBERO delta-pose range; the norm tag or the state is wrong")


# ------------------------------------------------------------------ latency and determinism
def stage_latency(bank: sh.Bank, state: Dict[str, object], pack,
                  graph_check: bool = False) -> Dict[str, object]:
    """Per-call cost and the two determinism checks, on the path every session will actually run.

    `graph_check` re-measures the CUDA graph against the eager loop. It is OFF by default and no
    session runs on the graph: see A17 and the `graph_hazard` note below. The equivalence itself was
    measured — `results/depthanchor/preflight/11` stage `latency`, `graph_vs_eager_max_abs_delta_norm`
    = 0.0, i.e. bit-identical, graph 0.485 s against eager 0.898 s per injected call.
    """
    from infra.pg import depthanchor_model as dam
    model, processor = state["model"], state["processor"]
    frames = pack["frames"][:3]
    out: Dict[str, object] = {}
    paths = (True, False) if graph_check else (False,)
    for graph in paths:
        actor = dam.Actor(model, processor, enable_cuda_graph=graph)
        nat_t, inj_t = [], []
        for f in frames:
            r = actor.natural(f, seed=0)
            nat_t.append(r.seconds)
            b = r.bins
            for _ in range(3):
                inj_t.append(actor.inject(f, b, seed=0).seconds)
        out[f"cuda_graph={graph}"] = {
            "natural_seconds": [round(x, 3) for x in nat_t],
            "inject_seconds_median": float(np.median(inj_t)),
            "inject_seconds_first": float(inj_t[0]), "inject_seconds_last": float(inj_t[-1])}
    f = frames[0]
    a_eager = dam.Actor(model, processor, enable_cuda_graph=False)
    b = a_eager.natural(f, seed=0).bins
    e = a_eager.inject(f, b, seed=0)
    if graph_check:
        # the graph is a performance path, not a different computation — and measuring that is what
        # poisoned the context in preflight/11, so it is opt-in
        a_graph = dam.Actor(model, processor, enable_cuda_graph=True)
        g = a_graph.inject(f, b, seed=0)
        out["graph_vs_eager_max_abs_delta_norm"] = float(np.max(np.abs(g.a_norm - e.a_norm)))
        rep_g = a_graph.inject(f, b, seed=0)
        out["graph_repeat_max_abs_delta_norm"] = float(np.max(np.abs(rep_g.a_norm - g.a_norm)))
    else:
        out["graph_vs_eager_max_abs_delta_norm"] = None
        out["graph_check"] = "skipped"
        out["graph_equivalence_measured_in"] = ("results/depthanchor/preflight/11 stage latency: "
                                                "graph_vs_eager_max_abs_delta_norm = 0.0")
    rep = a_eager.inject(f, b, seed=0)
    out["eager_repeat_max_abs_delta_norm"] = float(np.max(np.abs(rep.a_norm - e.a_norm)))
    seed_spread = float(np.max(np.abs(a_eager.inject(f, b, seed=1).a_norm - e.a_norm)))
    out["seed_change_max_abs_delta_norm"] = seed_spread
    if seed_spread <= out["eager_repeat_max_abs_delta_norm"]:
        raise RuntimeError("changing the generator seed moved the action no more than repeating the "
                           "same call: the flow sampler is not being reseeded, and the seed floor "
                           "would be a measurement of nothing")
    # The frozen consequence: how many frames get the 200-call per-cell map, and what each session's
    # cost is, decided from THIS instance's latency before any contrast number exists (A6) — and both
    # read the seconds of the path that will actually run.
    #
    # THE CUDA GRAPH IS NOT USED, and that is a measurement too (A17). In
    # `results/depthanchor/preflight/11` the graph/eager alternation this stage used to run
    # unconditionally — five Actors over one resident model — left the released depth-decode graph cache
    # in a state where the NEXT graph-enabled `predict_action` raised `CUDA error: an illegal memory
    # access was encountered`, in stage `g0`, on the same frame and the same bins the graph path had run
    # correctly two minutes earlier. A CUDA context does not recover from that, so the failure costs
    # whatever session it lands in. That run also measured the graph's output as BIT-IDENTICAL to the
    # eager loop's (0.0) at 0.485 s against 0.898 s per injected call, so running eager costs 0.41 s a
    # call and nothing scientific.
    out["graph_output_identical_to_eager"] = (
        None if out["graph_vs_eager_max_abs_delta_norm"] is None
        else bool(out["graph_vs_eager_max_abs_delta_norm"] < 1e-4))
    out["graph_hazard"] = ("results/depthanchor/preflight/11 stage g0: CUDA illegal memory access on "
                           "the first graph-enabled inject after this stage's graph/eager alternation")
    out["chosen_cuda_graph"] = False
    chosen_seconds = float(out["cuda_graph=False"]["inject_seconds_median"])
    out["seconds_per_call_used"] = round(chosen_seconds, 3)
    out["per_cell_frames_per_suite"] = 5 if chosen_seconds <= 0.6 else 2
    # Both sessions' cost, from the cell lists themselves and this instance's own seconds per call, so
    # the budget each is launched with is a measurement and not a guess. `pg` self-terminates at the
    # budget, so a session under-budgeted here loses its frames.
    from infra.pg import depthanchor_cells as cells_mod
    n_frames = int(pack["n_frames"])
    # one call per frame GENERATES the grid a code at a time (4.7 s eager) and the rest replay one
    # (0.9 s), so the two are priced separately instead of at the cheaper rate
    natural_seconds = float(np.median(out["cuda_graph=False"]["natural_seconds"]))
    out["natural_seconds_median"] = round(natural_seconds, 3)

    def hours(calls: int) -> float:
        return round((n_frames * natural_seconds + (calls - n_frames) * chosen_seconds) / 3600.0, 2)

    out["projected_gate_calls"] = cells_mod.projected_calls(cells_mod.GATE_CELLS, n_frames)
    out["projected_gate_hours"] = hours(out["projected_gate_calls"])
    out["projected_contrast_calls"] = cells_mod.projected_calls(
        cells_mod.CONTRAST_CELLS, n_frames, per_cell_frames=out["per_cell_frames_per_suite"])
    out["projected_contrast_hours"] = hours(out["projected_contrast_calls"])
    # the eager Actor that has just made a dozen successful calls, rather than a new one: one fewer
    # thing between this measurement and the stages that depend on it
    state["actor"] = a_eager
    return out


# ------------------------------------------------------------------ G0 and the seed floor
def stage_g0(bank: sh.Bank, state: Dict[str, object], pack) -> Dict[str, object]:
    actor = state["actor"]
    rows = []
    for f in pack["frames"][:G0_FRAMES]:
        nat = actor.natural(f, seed=0)
        ref = actor.inject(f, nat.bins, seed=0)
        seeds = [actor.inject(f, nat.bins, seed=s) for s in range(1, 4)]
        rows.append({
            "frame_id": f["spec"]["frame_id"], "suite": f["spec"]["suite"],
            "g0_max_abs_delta_norm": float(np.max(np.abs(ref.a_norm - nat.a_norm))),
            "g0_delta_norm_cont": core.delta_norm_cont(nat.a_norm, ref.a_norm),
            "seed_delta_norm_cont": [core.delta_norm_cont(ref.a_norm, s.a_norm) for s in seeds],
            "natural_n_tail": nat.n_tail_tokens, "ident_n_tail": ref.n_tail_tokens,
            "unique_codes": int(len(set(nat.bins.tolist()))),
            "gate_trained_mean": (None if ref.meta.get("gate_trained") is None
                                  else float(np.mean(ref.meta["gate_trained"]))),
        })
    worst = max(r["g0_max_abs_delta_norm"] for r in rows)
    g0 = core.gate_g0(worst, len(rows))
    seed_floor = [float(np.mean(r["seed_delta_norm_cont"])) for r in rows]
    bank.artifact("g0_rows.json", rows)
    return {"gate_G0": g0.as_dict(), "worst_max_abs_delta_norm": worst,
            "seed_floor_median": float(np.median(seed_floor)),
            "seed_floor_min": float(np.min(seed_floor)),
            "seed_floor_is_nonzero": bool(np.min(seed_floor) > 0),
            "n_frames": len(rows),
            "tail_token_counts": _counts([r["ident_n_tail"] for r in rows]),
            "natural_tail_token_counts": _counts([r["natural_n_tail"] for r in rows])}


# ------------------------------------------------------------------ the depth references
def stage_refs(bank: sh.Bank, state: Dict[str, object], pack) -> Dict[str, object]:
    from infra.pg import depthanchor_model as dam
    refs = dam.load_depth_refs()
    state["refs"] = refs
    t0 = time.perf_counter()
    ladder = refs.code_ladder()
    state["ladder"] = ladder
    ladder_path = bank.artifact("code_ladder.json", {"d_primary": ladder.d_primary.tolist(),
                                                     "d_local": ladder.d_local.tolist(),
                                                     "order": ladder.order.tolist(),
                                                     "median_code": ladder.median_code})
    sh.push_json(ladder_path, f"{sh.S3_LADDER}/code_ladder.json")
    gap = np.abs(ladder.d_primary - ladder.d_local)
    noise = float(np.median(gap))
    sorted_d = ladder.d_primary[ladder.order]
    left = np.concatenate([[np.inf], np.abs(np.diff(sorted_d))])
    right = np.concatenate([np.abs(np.diff(sorted_d)), [np.inf]])
    sep = np.minimum(left, right)
    n_distinct = int(np.sum(sep > noise))
    g2 = core.gate_g2(n_distinct, noise)

    actor = state["actor"]
    rows = []
    for f in pack["frames"][:FIDELITY_FRAMES]:
        nat = actor.natural(f, seed=0)
        ref_bins = refs.reference_bins(f["agentview"])
        decoded = refs.cell_depths(nat.bins)
        da2 = refs.depth_anything_map(f["agentview"])
        da2_cells = _cell_means(da2)
        shuffled = refs.cell_depths(core.edit_shuffle(nat.bins,
                                                      core.rng_for("g1shuffle", f["spec"]["frame_id"]),
                                                      ladder).bins)
        rows.append({
            "frame_id": f["spec"]["frame_id"], "suite": f["spec"]["suite"],
            "spearman_decoded_vs_da2": core.spearman(decoded, da2_cells),
            "spearman_shuffled": core.spearman(shuffled, da2_cells),
            "spearman_reference_vs_da2": core.spearman(refs.cell_depths(ref_bins), da2_cells),
            "bin_agreement": float(np.mean(nat.bins == ref_bins)),
            "reference_bins": ref_bins.astype(int).tolist(),
            "model_bins": nat.bins.astype(int).tolist(),
        })
    med = float(np.median([r["spearman_decoded_vs_da2"] for r in rows]))
    p95 = float(np.percentile([r["spearman_shuffled"] for r in rows], 95))
    g1 = core.gate_g1(med, p95, len(rows))
    bank.artifact("fidelity_rows.json", rows)
    return {"gate_G1": g1.as_dict(), "gate_G2": g2.as_dict(),
            "ladder_seconds": time.perf_counter() - t0,
            "ladder_range": [float(ladder.d_primary.min()), float(ladder.d_primary.max())],
            "ladder_placement_noise_median": noise, "ladder_codes_distinct": n_distinct,
            "median_spearman_decoded_vs_da2": med, "shuffled_p95": p95,
            "median_bin_agreement": float(np.median([r["bin_agreement"] for r in rows])),
            "median_spearman_reference_vs_da2": float(np.median(
                [r["spearman_reference_vs_da2"] for r in rows])),
            "n_frames": len(rows),
            "fidelity_per_frame": [{k: r[k] for k in ("frame_id", "suite",
                                                      "spearman_decoded_vs_da2", "spearman_shuffled",
                                                      "bin_agreement")} for r in rows]}


def _cell_means(image: np.ndarray) -> np.ndarray:
    """A full-resolution map reduced to the 10x10 grid the depth codes live on."""
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    if h % core.GRID or w % core.GRID:
        yi = (np.arange(core.GRID) * h / core.GRID).astype(int)
        xi = (np.arange(core.GRID) * w / core.GRID).astype(int)
        ys = np.append(yi, h)
        xs = np.append(xi, w)
        return np.array([[img[ys[r]:ys[r + 1], xs[c]:xs[c + 1]].mean() for c in range(core.GRID)]
                         for r in range(core.GRID)]).reshape(-1)
    return img.reshape(core.GRID, h // core.GRID, core.GRID, w // core.GRID).mean(axis=(1, 3)).reshape(-1)


# ------------------------------------------------------------------ orient (G3)
def stage_orient(bank: sh.Bank) -> Dict[str, object]:
    """G3, and the two conventions it decides: which pixel-sign convention the banked camera matrices
    want (against robosuite's own projection), and which row of the rendered buffers a projected point
    indexes (against the rendered depth). Runs BEFORE the pack, because the pack's target cells are the
    output of exactly this projection. The third vertical convention — whether the image the MODEL sees
    is the rendered buffer or its mirror — is not decidable here; the dataset decides it, in `pack`."""
    from infra.pg import depthanchor_libero as lib
    cam = json.loads((bank.out / "camera_bank.json").read_text())
    probes: Dict[str, object] = {}
    for key in ("libero_spatial/0", "libero_object/0"):
        entry = cam["tasks"][key]
        probes[key] = lib.probe_orientation(entry["suite"], entry["task_id"], entry, n_frames=20)
    bank.artifact("projection_probes.json", probes)

    def med(field, name, sub=None):
        vals = []
        for k in probes:
            v = probes[k][field][name]
            v = v if sub is None else v[sub]
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return float(np.median(vals)) if np.all(np.isfinite(vals)) else float("inf")

    real = ("opengl", "opencv")
    wrong = ("wrong_camera", "wrong_camera_opencv")
    pixel = {n: med("pixel_error_px", n) for n, *_ in lib.PROJECTION_VARIANTS}
    keys = [k for k, *_ in lib.DEPTH_INDEXINGS]
    depth = {n: {sub: med("depth_error_m", n, sub) for sub in keys}
             for n, *_ in lib.PROJECTION_VARIANTS}
    best = min(real, key=lambda n: (np.inf if pixel[n] is None else pixel[n]))
    passing = [n for n in real if pixel[n] is not None and pixel[n] < PROJECTION_TOL_PX]
    depth_key = min(keys, key=lambda k: (np.inf if depth[best][k] is None else depth[best][k]))
    depth_flips = {k: (uf, vf) for k, uf, vf in lib.DEPTH_INDEXINGS}[depth_key]
    wrong_pixel = min((pixel[w] for w in wrong if pixel[w] is not None), default=None)
    wrong_depth = min((depth[w][depth_key] for w in wrong if depth[w][depth_key] is not None),
                      default=None)
    g3 = core.gate_g3(pixel[best], wrong_pixel, depth[best][depth_key], wrong_depth,
                      tol_px=PROJECTION_TOL_PX, tol_depth_m=PROJECTION_TOL_M,
                      n_conventions_passing=len(passing))
    payload = {"chosen_convention": best, "opengl": best == "opengl",
               "depth_buffer_indexing": depth_key,
               "depth_u_flip": bool(depth_flips[0]), "depth_v_flip": bool(depth_flips[1]),
               "depth_flip": bool(depth_flips[1]),
               "pixel_error_px": pixel, "depth_error_m": depth,
               "passing_conventions": passing, "gate_G3": g3.as_dict(),
               "tol_px": PROJECTION_TOL_PX, "tol_depth_m": PROJECTION_TOL_M,
               "render_flip": None, "render_hflip": None,
               "note": ("the transform from a RENDERED image to the image the model expects is decided "
                        "in the `pack` stage: the dataset's own frames say where the projection lands "
                        "in a DATASET image, this stage says where it lands in a RENDERED one, and the "
                        "render transform is the difference between the two")}
    path = bank.artifact("orientation.json", payload)
    sh.push_json(path, f"{sh.S3_CAMERA}/orientation.json")
    return payload


def _orientation(bank: sh.Bank) -> Dict[str, object]:
    """The banked conventions, pulled if this invocation did not measure them. Never defaulted."""
    local = bank.out / "orientation.json"
    if not local.is_file():
        sh.pull_json(f"{sh.S3_CAMERA}/orientation.json", local)
    return json.loads(local.read_text())


def _update_orientation(bank: sh.Bank, **fields) -> Dict[str, object]:
    payload = dict(_orientation(bank))
    payload.update(fields)
    path = bank.artifact("orientation.json", payload)
    sh.push_json(path, f"{sh.S3_CAMERA}/orientation.json")
    return payload


def _row_profile_vote(per_task: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pool the per-episode row-profile correlations recorded by `_dataset_orientation` and vote. Returns
    `render_flip=None` when the two orientations are within 0.02 of each other, which is a refusal, not a
    guess."""
    raw = [r["row_profile_corr_raw"] for t in per_task for r in t["episodes"]
           if r.get("row_profile_corr_raw") is not None]
    mir = [r["row_profile_corr_mirror"] for t in per_task for r in t["episodes"]
           if r.get("row_profile_corr_mirror") is not None]
    if not raw or not mir:
        return {"render_flip": None, "why": "no row profiles recorded"}
    a, b = float(np.median(raw)), float(np.median(mir))
    if abs(a - b) < 0.02:
        return {"render_flip": None, "median_corr_raw": a, "median_corr_mirror": b,
                "why": "the two orientations are indistinguishable by row profile"}
    return {"render_flip": bool(b > a), "median_corr_raw": a, "median_corr_mirror": b}


# ------------------------------------------------------------------ G1B: the sign of decoded depth
def stage_polarity(bank: sh.Bank, state: Dict[str, object]) -> Dict[str, object]:
    """On simulator frames the depth is known in metres, so the SIGN of the VQ-VAE's decoded scale is
    measurable instead of assumed. A negative rank correlation with metric depth means the decoded
    value is a disparity (higher = nearer), which is the convention the directional predictor of §6
    rule 2b is registered under. Runs before any directional number exists."""
    from infra.pg import depthanchor_libero as lib
    cam = json.loads((bank.out / "camera_bank.json").read_text())
    orient = _orientation(bank)
    actor = _need(state, "actor", "model")
    refs = _need(state, "refs", "refs")
    entry = cam["tasks"][f"{LOOP_SUITE}/0"]
    env, bench, task = lib.make_env(entry["suite"], entry["task_id"], camera_depths=True,
                                    image_size=lib.IMAGE_SIZE)
    rows = []
    try:
        init_states = bench.get_task_init_states(entry["task_id"])
        for i in range(3):
            env.reset()
            env.set_init_state(init_states[i])
            for _ in range(lib.SETTLE_STEPS):
                obs, _, _, _ = env.step(lib.NOOP)
            frame = lib.obs_to_frame(obs, flip=bool(orient["render_flip"]),
                                     hflip=bool(orient.get("render_hflip")), task=task.language)
            frame["spec"] = {"frame_id": f"sim:{entry['suite']}/{entry['task_id']}:{i}",
                             "suite": entry["suite"],
                             "task_key": f"{entry['suite']}/{entry['task_id']}"}
            res = actor.natural(frame, seed=0)
            decoded = refs.cell_depths(res.bins)
            # the metric depth must sit in the SAME orientation as the image the model was given, or
            # this compares two mirrored 10x10 grids (A15). Both mirrors of it are kept as the
            # registered failing inputs: DepthAnything-2 on the model's own image has to agree with
            # the aligned grid more than with either mirror.
            aligned = lib.model_space_depth(env, obs, flip=bool(orient["render_flip"]),
                                            hflip=bool(orient.get("render_hflip")))
            truth = _cell_means(aligned)
            da2 = _cell_means(refs.depth_anything_map(frame["agentview"]))
            rows.append({"frame_id": frame["spec"]["frame_id"],
                         "spearman_decoded_vs_truth_m": core.spearman(decoded, truth),
                         "spearman_da2_vs_truth_m": core.spearman(da2, truth),
                         "spearman_da2_vs_truth_hmirror": core.spearman(
                             da2, _cell_means(aligned[:, ::-1])),
                         "spearman_da2_vs_truth_vmirror": core.spearman(
                             da2, _cell_means(aligned[::-1])),
                         "spearman_decoded_vs_da2": core.spearman(decoded, da2),
                         "truth_range_m": [float(truth.min()), float(truth.max())],
                         "decoded_range": [float(decoded.min()), float(decoded.max())],
                         "bins": res.bins.astype(int).tolist()})
    finally:
        env.close()
    bank.artifact("polarity_rows.json", rows)
    rho = float(np.median([r["spearman_decoded_vs_truth_m"] for r in rows]))
    align = {k: float(np.median([abs(r[k]) for r in rows])) for k in
             ("spearman_da2_vs_truth_m", "spearman_da2_vs_truth_hmirror",
              "spearman_da2_vs_truth_vmirror")}
    # A19. The gate keeps the clause that has room to fire. Depth in these scenes is monotone in the
    # image ROW, so mirroring the grid vertically destroys the agreement (0.995 against 0.867 here, 0.968
    # against 0.752 in preflight/12's wrong orientation) — that clause discriminates. Mirroring it
    # horizontally barely moves it, because the rendered scene is nearly symmetric about the camera axis:
    # 0.995 against 0.978 here and 0.968 against 0.946 in /12, i.e. a margin of 0.017 and 0.022 either
    # side of the 0.02 threshold. A clause that decides on 0.005 of a nuisance margin has no resolution,
    # so it is REPORTED and does not gate. The horizontal bit is settled by two instruments that do have
    # resolution: the motion percentile on the dataset's own frames (u=True 0.925/0.68 against u=False
    # 0.43/0.38) and the five-episode known number (5/5 at hflip=True against 0/5 at hflip=False).
    v_margin = align["spearman_da2_vs_truth_m"] - align["spearman_da2_vs_truth_vmirror"]
    h_margin = align["spearman_da2_vs_truth_m"] - align["spearman_da2_vs_truth_hmirror"]
    aligned = bool(v_margin > 0.02)
    payload = {"measured": True, "n_frames": len(rows),
               "median_spearman_decoded_vs_truth_m": rho,
               "median_spearman_da2_vs_truth_m": float(np.median(
                   [r["spearman_da2_vs_truth_m"] for r in rows])),
               "median_spearman_decoded_vs_da2": float(np.median(
                   [r["spearman_decoded_vs_da2"] for r in rows])),
               "depth_grid_alignment_abs_spearman": align,
               "depth_grid_alignment_vertical_margin": float(v_margin),
               "depth_grid_alignment_horizontal_margin": float(h_margin),
               "depth_grid_aligned": aligned,
               "depth_grid_alignment_note": ("|rho| of DepthAnything-2 on the model's image against the "
                                             "aligned metric grid must beat its VERTICAL mirror by 0.02, "
                                             "or the depth grid and the image the model saw are not the "
                                             "same way up and no per-cell number is readable. The "
                                             "horizontal margin is reported and does not gate: depth is "
                                             "monotone in the image row, not the column, so this "
                                             "instrument has no resolution on that axis (A19)"),
               "decoded_is_disparity_higher_is_nearer": bool(rho < 0),
               "predictor_sign_as_registered": bool(rho < 0),
               "registered_convention": ("decoded depth is a disparity, higher = nearer, so the "
                                         "directional predictor is -(realized decoded-depth change)")}
    merged = dict(_orientation(bank))
    merged["polarity"] = payload
    path = bank.artifact("orientation.json", merged)
    sh.push_json(path, f"{sh.S3_CAMERA}/orientation.json")
    return payload


# ------------------------------------------------------------------ the closed-loop known number
def stage_loop(bank: sh.Bank, state: Dict[str, object], sweep: bool = False) -> Dict[str, object]:
    """The §9.1 known number, and the arbiter of the render transform (A15, A18). A policy given a
    mirrored world does not reach for anything, so five episodes of the released deployment path are what
    decide which of the four (vertical, horizontal) transforms the checkpoint was trained on. With
    `--loop-transform-sweep` all four are run and banked; otherwise only the derived one. Whichever
    clears the bar is written back into the banked orientation, with the reason."""
    from infra.pg import depthanchor_libero as lib
    orient = _orientation(bank)
    derived = (bool(orient["render_flip"]), bool(orient.get("render_hflip")))
    variants = [derived]
    if sweep:
        variants += [t for t in ((False, False), (False, True), (True, False), (True, True))
                     if t != derived]
    actor = _need(state, "actor", "model")
    per_variant: List[Dict[str, object]] = []
    all_rows: List[Dict[str, object]] = []
    for flip, hflip in variants:
        policy = lib.ArmNormal()
        rows = []
        for ep in range(LOOP_EPISODES):
            r = lib.rollout(actor, LOOP_SUITE, task_id=ep, episode=0, policy=policy, flip=flip,
                            hflip=hflip, max_steps=lib.HORIZON[LOOP_SUITE])
            rec = r.as_record()
            rec["render_flip"], rec["render_hflip"] = bool(flip), bool(hflip)
            rows.append(rec)
            print(f"[loop] vflip={flip} hflip={hflip} {LOOP_SUITE} task {ep}: success={r.success} "
                  f"steps={r.n_steps} {r.seconds:.0f}s", flush=True)
            bank.stage("loop_partial", {"variants_done": len(per_variant),
                                        "rows": [{k: v for k, v in x.items()
                                                  if k not in ("eef_path", "bins_per_call",
                                                               "gate_per_call")} for x in rows]})
        n_ok = sum(1 for r in rows if r["success"])
        per_variant.append({
            "render_flip": bool(flip), "render_hflip": bool(hflip), "is_derived": (flip, hflip) == derived,
            "successes": n_ok, "episodes": LOOP_EPISODES,
            "steps": [r["n_steps"] for r in rows],
            "seconds_per_episode": [round(r["seconds"], 1) for r in rows],
            "cells_regenerated_median": float(np.median(
                [np.median(_regen(r)) if _regen(r) else np.nan for r in rows])),
            "done_vs_check_agree": [r["meta"].get("agree") for r in rows]})
        all_rows.extend(rows)
        print(f"[loop] vflip={flip} hflip={hflip}: {n_ok}/{LOOP_EPISODES}", flush=True)
    bank.artifact("loop_rows.json", all_rows)
    best = max(per_variant, key=lambda v: v["successes"])
    n_ok = int(best["successes"])
    if len(per_variant) > 1 and n_ok >= 4 and not best["is_derived"]:
        # the winning transform is not the derived one, so the banked convention is corrected by this
        # measurement rather than by an argument about which probe to believe
        _update_orientation(bank, render_flip=best["render_flip"], render_hflip=best["render_hflip"],
                            render_transform_decided_by=(
                                f"the {LOOP_EPISODES}-episode known number swept over all four "
                                f"transforms: {n_ok}/{LOOP_EPISODES} at "
                                f"vflip={best['render_flip']} hflip={best['render_hflip']}"))
    return {"suite": LOOP_SUITE, "episodes": LOOP_EPISODES, "successes": n_ok,
            "required": 4, "pass": bool(n_ok >= 4),
            "swept": bool(len(per_variant) > 1),
            "render_flip": best["render_flip"], "render_hflip": best["render_hflip"],
            "render_flip_derived": derived[0], "render_hflip_derived": derived[1],
            "winner_is_derived": bool(best["is_derived"]),
            "by_transform": per_variant,
            "published_card_rate": 100.0,
            "steps": best["steps"], "seconds_per_episode": best["seconds_per_episode"],
            "cells_regenerated_median": best["cells_regenerated_median"],
            "done_vs_check_agree": best["done_vs_check_agree"]}


def _need(state: Dict[str, object], key: str, stage: str):
    if key not in state:
        raise RuntimeError(f"this stage needs `{key}`, which stage `{stage}` produces; run it in the "
                           f"same invocation (--stages ...,{stage},...)")
    return state[key]


def _regen(record) -> List[int]:
    bins = record.get("bins_per_call") or []
    out = []
    for a, b in zip(bins, bins[1:]):
        out.append(int(np.sum(np.asarray(a) != np.asarray(b))))
    return out


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--graph-check", action="store_true",
                    help="re-measure the CUDA graph against the eager loop in the latency stage. Off by "
                         "default: no session runs on the graph, and measuring it is what poisoned the "
                         "CUDA context in preflight/11 (A17)")
    ap.add_argument("--loop-transform-sweep", action="store_true",
                    help="run the five known-number episodes under all four (vertical, horizontal) "
                         "render transforms and bank the table. This is A15's arbiter: a policy given a "
                         "mirrored world does not reach for anything, so the success rate decides the "
                         "transform when the projection probes disagree")
    ap.add_argument("--work", default="/work/depthanchor")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            raise ValueError(f"unknown stage {s!r}; known: {ALL_STAGES}")
    bank = sh.Bank(out, "preflight")
    bank.top("stages_requested", stages)
    state: Dict[str, object] = {}
    pack = None

    if "env" in stages:
        bank.stage("env", stage_env(bank))
    if "cam" in stages:
        bank.stage("cam", stage_cam(bank))
    if any(s in stages for s in ("orient", "pack", "polarity", "vertical")) \
            and not (out / "camera_bank.json").is_file():
        sh.pull_camera_bank(out)
    if "orient" in stages:
        bank.stage("orient", stage_orient(bank))
    if "model" in stages:
        bank.stage("model", stage_model(bank, state))
    if "vertical" in stages:
        bank.stage("vertical", stage_vertical(bank, state, work))
    if "pack" in stages:
        bank.stage("pack", stage_pack(bank, work))
    if any(s in stages for s in ("latency", "g0", "refs")):
        pack_dir = out / "pack"
        if not (pack_dir / "frames_index.json").is_file():
            sh.pull_pack(pack_dir)
        pack = frames_mod.load_pack(pack_dir)
    if "latency" in stages:
        bank.stage("latency", stage_latency(bank, state, pack, graph_check=args.graph_check))
    if "g0" in stages:
        bank.stage("g0", stage_g0(bank, state, pack))
    if "refs" in stages:
        bank.stage("refs", stage_refs(bank, state, pack))
    if "polarity" in stages:
        bank.stage("polarity", stage_polarity(bank, state))
    if "loop" in stages:
        bank.stage("loop", stage_loop(bank, state, sweep=args.loop_transform_sweep))

    verdict = _verdict(bank.payload["stages"])
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1), flush=True)
    if not verdict["ready_for_gates"]:
        raise SystemExit(f"preflight is not ready for the gate session: {verdict['blockers']}")
    if not verdict["ready_for_loop"]:
        print(f"[preflight] READY for the offline sessions, NOT for the closed loop: "
              f"{verdict['loop_blockers']}", flush=True)


def _verdict(stages: Dict[str, object]) -> Dict[str, object]:
    """The verdict reads its gates, and names the ones that are MISSING rather than assuming they
    passed. G1 or G2 failing does not block: G1 cancels every clause stated in decoded metres and G2
    triggers the §5 pivot, and both are pre-registered responses rather than stops."""
    blockers: List[str] = []
    checks: Dict[str, object] = {}
    g0stage = stages.get("g0") or {}
    checks["G0"] = g0stage.get("gate_G0", {}).get("verdict")
    if checks["G0"] != "pass":
        blockers.append(f"G0 is {checks['G0']}")
    if g0stage and not g0stage.get("seed_floor_is_nonzero"):
        blockers.append("the seed floor is zero on some frame: the flow sampler is not being reseeded")
    refs = stages.get("refs") or {}
    for name in ("G1", "G2"):
        checks[name] = refs.get(f"gate_{name}", {}).get("verdict")
        if checks[name] is None:
            blockers.append(f"{name} not measured")
    orient = stages.get("orient") or {}
    checks["G3"] = orient.get("gate_G3", {}).get("verdict")
    if checks["G3"] != "pass":
        blockers.append(f"G3 is {checks['G3']}")
    checks["chosen_convention"] = orient.get("chosen_convention")
    checks["depth_buffer_indexing"] = orient.get("depth_buffer_indexing")
    checks["passing_conventions"] = orient.get("passing_conventions")
    pack = stages.get("pack") or {}
    motion = pack.get("motion_check") or {}
    agree = pack.get("dataset_agreement") or {}
    checks["dataset_flips_u_v"] = [pack.get("dataset_u_flip"), pack.get("dataset_v_flip")]
    checks["render_flips_h_v"] = [pack.get("render_hflip"), pack.get("render_vflip")]
    vert = stages.get("vertical") or {}
    checks["vertical_render_flip"] = vert.get("render_flip")
    checks["vertical_dataset_v_flip"] = vert.get("dataset_v_flip")
    checks["vertical_decisive"] = vert.get("decisive")
    checks["vertical_dataset_da2_vs_row"] = vert.get("median_spearman_dataset_da2_vs_row")
    checks["vertical_render_metric_vs_row"] = vert.get("median_spearman_render_metric_disparity_vs_row")
    checks["motion_v_flip_disagrees"] = (motion or {}).get("v_flip_disagrees_with_depth_probe")
    if vert and not vert.get("decisive"):
        blockers.append("the vertical stage did not decide which way up a dataset image is")
    checks["motion_winner"] = motion.get("winner")
    checks["motion_median_percentile"] = (motion.get("median_percentile") or {}).get(
        motion.get("winner")) if isinstance(motion.get("median_percentile"), dict) else None
    checks["motion_margin"] = motion.get("margin")
    checks["episodes_start_from_shipped_init_states"] = agree.get("same_camera")
    if pack and not motion.get("decisive"):
        blockers.append("the projection does not land on the moving arm in the dataset's own frames, "
                        "so the target cells cannot be trusted")
    pol = stages.get("polarity") or {}
    checks["decoded_is_disparity"] = pol.get("decoded_is_disparity_higher_is_nearer")
    checks["depth_grid_aligned"] = pol.get("depth_grid_aligned")
    checks["depth_grid_alignment_abs_spearman"] = pol.get("depth_grid_alignment_abs_spearman")
    checks["depth_grid_alignment_vertical_margin"] = pol.get("depth_grid_alignment_vertical_margin")
    checks["depth_grid_alignment_horizontal_margin"] = pol.get("depth_grid_alignment_horizontal_margin")
    if not pol:
        blockers.append("G1B (the sign of decoded depth) not measured")
    elif not pol.get("depth_grid_aligned"):
        blockers.append("the metric depth grid does not beat its VERTICAL mirror against the depth "
                        "reference on the model's image: the grid and the image the model saw are not "
                        "the same way up, so no per-cell number is readable")
    loop = stages.get("loop") or {}
    checks["loop_successes"] = loop.get("successes")
    loop_blockers: List[str] = []
    if loop and not loop.get("pass"):
        loop_blockers.append(f"the released deployment path scored {loop.get('successes')}/"
                             f"{loop.get('episodes')} on {loop.get('suite')}, below the required 4")
    elif not loop:
        loop_blockers.append("the closed-loop known number was not measured")
    model = stages.get("model") or {}
    checks["recording_wrapper_ok"] = model.get("recording_wrapper_ok")
    if model and not model.get("recording_wrapper_ok"):
        blockers.append("the normalized-action recording wrapper is not invertible")
    lat = stages.get("latency") or {}
    for k in ("graph_vs_eager_max_abs_delta_norm", "chosen_cuda_graph", "seed_change_max_abs_delta_norm",
              "per_cell_frames_per_suite", "projected_gate_hours", "projected_contrast_hours"):
        checks[k] = lat.get(k)
    # TWO readinesses, because they license different conclusions. The offline cells read dataset frames
    # and need G0, the depth references, the projection and the dataset-camera agreement; they do not
    # need the policy to succeed at the task. The closed-loop session does, and the §9.1 known number is
    # what says our rollout loop is the released one. A rollout that underperforms therefore blocks the
    # loop session and not the offline science, and saying so is more honest than one flag for both.
    return {"ready_for_gates": not blockers, "blockers": blockers,
            "ready_for_loop": not (blockers or loop_blockers),
            "loop_blockers": loop_blockers, "checks": checks}


if __name__ == "__main__":
    main()
