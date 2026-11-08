"""DEPTHANCHOR offline gate — the whole pipeline minus the model forward, on the control plane.

The repo rule this file exists for: "The full pipeline minus the model forward runs green locally. The
first GPU run reproduces one known number before any new one is trusted." So this job runs, with no
torch, no weights and no network:

  1. `infra.pg.depthanchor_core` self-test (every gate branch, every edit, every statistic).
  2. THE FRAME PACK, end to end, on a synthetic LeRobot-v3 dataset written here: `read_meta` ->
     `select_frames` -> `build_pack` -> `save_pack` -> `load_pack`, including the file-offset
     assumption that `_read_rows` checks per row. A registered failing input (a corrupted offset) must
     make it raise.
  3. THE NINE OFFLINE CELLS, with a stub actor whose response to the depth grid is planted:
        `epi`  ignores the grid           -> the pipeline must return H_epi
        `gate` reads the grid unstructured-> must return H_gate
        `geo`  reads the target cells with a direction -> must return H_geo
     A pipeline that cannot tell these three apart cannot be trusted to tell them apart on the model,
     and a pipeline that reports H_geo on `epi` is worse than no pipeline.
  4. THE CLOSED-LOOP aggregation and its confirmation rules, on planted success rates.
  5. The figures, drawn from the planted records, so a figure bug is found here and not after 6 GPU-hours.

  python -m infra.aws_pg.jobs.depthanchor_dryrun --out /tmp/da_dryrun
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
from typing import Dict, List, Optional

import numpy as np

from infra.pg import depthanchor_cells as cells
from infra.pg import depthanchor_core as core
from infra.pg import depthanchor_frames as frames_mod

SUITE_TASKS = {s: [f"{s} pretend task {i}" for i in range(10)] for s in core.SUITES}
IMG = 64


# ------------------------------------------------------------------ a stub actor with planted physics
class StubResult:
    def __init__(self, a_norm, a_robot, bins, token_ids, n_grid, n_tail, seconds, gate, meta):
        self.a_norm = a_norm
        self.a_robot = a_robot
        self.bins = bins
        self.token_ids = token_ids
        self.n_grid_tokens = n_grid
        self.n_tail_tokens = n_tail
        self.seconds = seconds
        self.gate = gate
        self.meta = meta


class StubActor:
    """Deterministic, bins-sensitive, and honest about its own conventions: it reproduces the parts of
    the real actor the pipeline depends on (identity replay reproduces the natural call to 1e-9, a new
    seed moves the action a little, a forced gate of 0 removes the depth term entirely, an RGB change
    moves it a lot) and nothing else."""

    def __init__(self, mode: str, ladder: core.CodeLadder, *, noise: float = 0.004,
                 rgb_scale: float = 0.40, depth_scale: float = 0.12, geo_scale: float = 0.35,
                 reject_rate: float = 0.0):
        if mode not in ("epi", "gate", "geo"):
            raise ValueError(mode)
        self.mode = mode
        self.reject_rate = float(reject_rate)
        self.rejected = 0
        self.ladder = ladder
        self.noise = noise
        self.rgb_scale = rgb_scale
        self.depth_scale = depth_scale
        self.geo_scale = geo_scale
        self.calls = 0
        self._forced: Optional[float] = None
        self._cell_weight = core.rng_for("stub", "cellweight").normal(size=core.N_CELLS)

    def force_gate(self, value):
        self._forced = None if value is None else float(value)

    def natural(self, frame, seed: int = 0):
        return self._act(frame, self._own_bins(frame), seed, adaptive=False)

    def inject(self, frame, bins, seed: int = 0):
        want = np.asarray(bins, dtype=np.int64).reshape(-1)
        # only EDITED grids can be refused: the model has just produced its own, so replaying it is
        # exactly the case the released code cannot fail on, and the reference call is unguarded for
        # that reason
        if self.reject_rate > 0 and not np.array_equal(want, self._own_bins(frame)):
            # deterministic per (frame, grid): stands in for the released code refusing to close its own
            # depth block after an out-of-distribution prefix
            r = core.rng_for("stub-reject", frame["spec"]["frame_id"], int(want.sum())).random()
            if r < self.reject_rate:
                self.rejected += 1
                raise core.EditNotCompleted(
                    "Depth generation did not emit <depth_end> within 10 steps after adaptive depth "
                    "tokens. [stub]")
        return self._act(frame, want, seed, adaptive=True)

    def adaptive(self, frame, cache, seed: int = 0):
        bins = self._own_bins(frame) if cache is None else np.asarray(cache["depth_bins"],
                                                                     dtype=np.int64)
        res = self._act(frame, bins, seed, adaptive=True)
        return res, {"image": frame["agentview"], "depth_bins": res.bins}

    def _own_bins(self, frame) -> np.ndarray:
        return core.rng_for("stub-bins", frame["spec"]["frame_id"]).integers(0, core.N_CODES,
                                                                            size=core.N_CELLS)

    def _act(self, frame, bins: np.ndarray, seed: int, adaptive: bool) -> StubResult:
        self.calls += 1
        fid = frame["spec"]["frame_id"]
        base = core.rng_for("stub-base", fid).normal(size=(core.CHUNK, core.ACTION_DIM)) * 0.2
        base[:, core.GRIP_DIM] = np.where(np.arange(core.CHUNK) < 5, -1.0, 1.0)
        jitter = core.rng_for("stub-seed", fid, seed).normal(size=base.shape) * self.noise
        rgb_key = int(np.asarray(frame["agentview"], dtype=np.int64).sum() % 1_000_003)
        rgb = core.rng_for("stub-rgb", rgb_key).normal(size=base.shape) * self.rgb_scale
        depth = np.zeros_like(base)
        if self._forced != 0.0:
            # The depth term is ABSOLUTE in the grid, not a difference from the model's own grid.
            # That matters: with a difference the identity replay and the gate-0 ablation would both
            # be zero, the depth-matched ceiling would be 0 by construction, and the dry run would
            # never exercise the branch where removing the prefix moves the action.
            d_now = self.ladder.depth_of(bins)
            if self.mode in ("gate", "geo"):
                unstructured = float(np.dot(self._cell_weight, d_now) / core.N_CELLS)
                depth = depth + self.depth_scale * unstructured * np.ones_like(base)
            if self.mode == "geo":
                t = np.asarray(frame["target_cells"], dtype=np.int64)
                # farther (a smaller decoded depth = less disparity) pushes the action along +axis
                signed = -float(np.mean(d_now[t]))
                axis = np.asarray(frame["axis_into_scene"], dtype=np.float64)
                axis = axis / np.linalg.norm(axis)
                depth[:, 0:3] = depth[:, 0:3] + self.geo_scale * signed * axis[None, :]
                depth = depth + 0.5 * self.geo_scale * abs(signed) * np.ones_like(base)
        a_norm = base + jitter + rgb + depth
        a_robot = a_norm.copy()
        a_robot[:, core.GRIP_DIM] = np.clip(a_robot[:, core.GRIP_DIM], -1.0, 1.0)
        n_tail = 1
        token_ids = np.concatenate([[153983], 153985 + bins, [153984]])
        gate = None if self._forced is None else np.full(36, self._forced)
        return StubResult(a_norm=a_norm, a_robot=a_robot, bins=bins.copy(), token_ids=token_ids,
                          n_grid=core.N_CELLS, n_tail=n_tail, seconds=0.001, gate=gate,
                          meta={"seed": seed, "adaptive": adaptive, "gate_forced": self._forced,
                                "gate_trained": np.full(36, 0.5).tolist(), "depth_positions": 102,
                                "cuda_graph": False})


# ------------------------------------------------------------------ a synthetic LeRobot v3 dataset
def _png(rng, size=IMG) -> bytes:
    import io
    from PIL import Image
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def write_synthetic_dataset(root: pathlib.Path, *, n_episodes_per_task: int = 6,
                            episode_length: int = 14, rows_per_file: int = 98) -> Dict[str, object]:
    """A dataset with the released schema: images as `struct<bytes, path>`, an 8-vector state, a
    7-vector action, and `meta/episodes` carrying `dataset_from_index` / `data/file_index`, so the
    offset arithmetic the real pack depends on is exercised on files whose contents are known."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = pathlib.Path(root)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    rng = core.rng_for("dryrun", "dataset")
    languages = [t for s in core.SUITES for t in SUITE_TASKS[s]]
    task_index = {t: i for i, t in enumerate(languages)}

    rows: List[Dict[str, object]] = []
    episodes: List[Dict[str, object]] = []
    ep = 0
    for language in languages:
        for _ in range(n_episodes_per_task):
            start = len(rows)
            for f in range(episode_length):
                rows.append({
                    "observation.images.image": {"bytes": _png(rng), "path": None},
                    "observation.images.wrist_image": {"bytes": _png(rng), "path": None},
                    # a straight-line path in front of the synthetic camera
                    "observation.state": [0.02 * f - 0.1, 0.01 * f, 0.9 + 0.005 * f,
                                          3.14, 0.0, 0.0, 0.04, -0.04],
                    "action": list(rng.normal(size=7).astype(float)),
                    "timestamp": 0.1 * f, "frame_index": f, "episode_index": ep,
                    "index": start + f, "task_index": task_index[language],
                })
            episodes.append({"episode_index": ep, "dataset_from_index": start,
                             "dataset_to_index": start + episode_length, "tasks": [language],
                             "length": episode_length})
            ep += 1

    schema = pa.schema([
        ("observation.images.image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("observation.images.wrist_image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("observation.state", pa.list_(pa.float32(), 8)),
        ("action", pa.list_(pa.float32(), 7)),
        ("timestamp", pa.float32()), ("frame_index", pa.int64()), ("episode_index", pa.int64()),
        ("index", pa.int64()), ("task_index", pa.int64())])
    file_of_row: Dict[int, int] = {}
    n_files = 0
    for fi, start in enumerate(range(0, len(rows), rows_per_file)):
        chunk = rows[start:start + rows_per_file]
        table = pa.Table.from_pylist(chunk, schema=schema)
        # two row groups per file, so `_read_rows` has to pick the right one
        pq.write_table(table, root / "data" / "chunk-000" / f"file-{fi:03d}.parquet",
                       row_group_size=max(1, len(chunk) // 2))
        for r in chunk:
            file_of_row[int(r["index"])] = fi
        n_files = fi + 1
    for e in episodes:
        e["data/file_index"] = file_of_row[int(e["dataset_from_index"])]
        e["data/chunk_index"] = 0
        if file_of_row[int(e["dataset_to_index"]) - 1] != e["data/file_index"]:
            raise RuntimeError("a synthetic episode straddles two files; raise rows_per_file so the "
                               "synthetic dataset keeps the released layout's one-file-per-episode "
                               "property")
    ep_schema = pa.schema([("episode_index", pa.int64()), ("data/chunk_index", pa.int64()),
                           ("data/file_index", pa.int64()), ("dataset_from_index", pa.int64()),
                           ("dataset_to_index", pa.int64()),
                           ("tasks", pa.list_(pa.string())), ("length", pa.int64())])
    pq.write_table(pa.Table.from_pylist(episodes, schema=ep_schema),
                   root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({"task_index": list(task_index.values()),
                             "task": list(task_index.keys())}),
                   root / "meta" / "tasks.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": 10, "total_episodes": len(episodes),
        "total_frames": len(rows), "total_tasks": len(languages), "chunks_size": 1000,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}))
    return {"n_rows": len(rows), "n_episodes": len(episodes), "n_files": n_files}


def _synthetic_row_index(root: pathlib.Path, meta: Dict[str, object]) -> Dict[str, object]:
    """The synthetic dataset's row index, built the same way the real one is — by scanning the data
    files' own columns — so the dry run exercises `load_row_index`'s invariants and not a shortcut."""
    import pyarrow.parquet as pq
    files, episodes = [], []
    paths = sorted((root / "data" / "chunk-000").glob("*.parquet"))
    for k, path in enumerate(paths):
        tb = pq.read_table(path, columns=["episode_index", "frame_index", "task_index", "index"])
        ei = np.asarray(tb.column("episode_index"))
        fi = np.asarray(tb.column("frame_index"))
        ti = np.asarray(tb.column("task_index"))
        gi = np.asarray(tb.column("index"))
        files.append({"file_index": k, "n_rows": int(ei.size), "index_min": int(gi.min()),
                      "index_max": int(gi.max()), "episode_min": int(ei.min()),
                      "episode_max": int(ei.max())})
        for e in np.unique(ei):
            rows = np.flatnonzero(ei == e)
            episodes.append({"episode": int(e), "file_index": k, "local_start": int(rows[0]),
                             "length": int(rows.size), "frame_first": int(fi[rows[0]]),
                             "frame_last": int(fi[rows[-1]]), "task_index": int(ti[rows[0]]),
                             "global_start": int(gi[rows[0]])})
    payload = {"source": str(root), "built_utc": "dryrun", "n_files": len(files),
               "n_episodes": len(episodes), "total_rows": files[-1]["index_max"] + 1,
               "data_path": "data/chunk-000/file-{file_index:03d}.parquet",
               "files": files, "episodes": episodes}
    path = root / "rowindex.json"
    path.write_text(json.dumps(payload))
    return frames_mod.load_row_index(path)


def synthetic_camera_bank() -> Dict[str, object]:
    """A camera one metre in front of the synthetic path, looking down -z of its own frame, so the
    projected path lands inside the image and the target cells are non-degenerate."""
    K = [[80.0, 0.0, IMG / 2], [0.0, 80.0, IMG / 2], [0.0, 0.0, 1.0]]
    bank: Dict[str, object] = {"image_size": IMG, "tasks": {}, "by_language": {}}
    for suite in core.SUITES:
        for tid, language in enumerate(SUITE_TASKS[suite]):
            key = f"{suite}/{tid}"
            bank["tasks"][key] = {
                "suite": suite, "task_id": tid, "language": language, "bddl": "synthetic.bddl",
                "K": K, "R_world_to_cam": np.eye(3).tolist(), "cam_pos_world": [0.0, 0.0, 0.0],
                "view_dir_world": [0.0, 0.0, 1.0], "znear": 0.01, "zfar": 10.0, "n_init_states": 5}
            bank["by_language"][frames_mod._norm(language)] = key
    return bank


# ------------------------------------------------------------------ stages
def stage_frames(out: pathlib.Path) -> Dict[str, object]:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="da_ds_"))
    try:
        info = write_synthetic_dataset(tmp)
        meta = frames_mod.read_meta(tmp / "meta")
        task_strings = frames_mod.read_task_strings(tmp / "meta")
        row_index = _synthetic_row_index(tmp, meta)
        bank = synthetic_camera_bank()
        specs = frames_mod.select_frames(row_index, task_strings, bank["by_language"])
        # `select_frames` returns CANDIDATES (oversampled); `attach_cells` keeps the usable ones
        want = int(round(frames_mod.N_PER_SUITE * frames_mod.OVERSAMPLE))
        assert len(specs) == 4 * want, len(specs)
        by_suite: Dict[str, int] = {}
        for s in specs:
            by_suite[s.suite] = by_suite.get(s.suite, 0) + 1
        assert set(by_suite.values()) == {want}, by_suite
        assert len({s.frame_id for s in specs}) == len(specs), "duplicate frame ids"
        for s in specs:
            assert s.frame_index + frames_mod.FUTURE_STEPS < s.episode_length
        again = frames_mod.select_frames(row_index, task_strings, bank["by_language"])
        assert [x.frame_id for x in again] == [x.frame_id for x in specs], "the draw is not reproducible"

        # the synthetic camera sits at the origin looking down +z with the path ahead of it, i.e. the
        # OpenCV convention; building with the OpenGL one must RAISE rather than mirror the grid
        try:
            frames_mod.build_pack(specs[:2], row_index, tmp, bank, opengl=True)
            raise AssertionError("the wrong projection convention did not raise")
        except RuntimeError as exc:
            assert "behind the camera" in str(exc), exc
        frames = frames_mod.read_frames(specs, row_index, tmp)
        # the synthetic frames are random noise, so the percentile test cannot be decisive here; what
        # the offline gate checks is that it RUNS, counts, and refuses rather than guessing
        motion = frames_mod.motion_check(frames, bank, opengl=False, min_eef_move_m=0.0, n_null=40)
        assert set(motion) >= {"n", "n_skipped", "decisive", "u_flip", "v_flip", "winner",
                               "median_percentile"}, sorted(motion)
        assert motion["n"] + motion["n_skipped"] == motion["n_frames_seen"]
        assert set(motion["median_percentile"]) == {n for n, *_ in frames_mod.DIHEDRAL}
        pack = frames_mod.attach_cells(frames, bank, opengl=False, v_flip=False)
        assert pack["n_frames"] == 4 * frames_mod.N_PER_SUITE, pack["n_frames"]
        assert pack["candidates"] == len(specs) >= pack["n_frames"]
        for f in pack["frames"]:
            t = np.asarray(f["target_cells"]); b = np.asarray(f["bg_cells"])
            assert t.size == b.size and t.size > 0
            assert not set(t.tolist()) & set(b.tolist())
            assert np.asarray(f["future_eef"]).shape == (frames_mod.FUTURE_STEPS, 3)
            assert np.asarray(f["agentview"]).shape == (IMG, IMG, 3)
        payload = frames_mod.save_pack(pack, out / "pack")
        back = frames_mod.load_pack(out / "pack")
        assert back["n_frames"] == pack["n_frames"]
        assert np.array_equal(back["frames"][3]["agentview"], pack["frames"][3]["agentview"])
        assert np.array_equal(back["frames"][3]["target_cells"], pack["frames"][3]["target_cells"])

        # registered failing input: a corrupted row location must raise, not pack someone else's
        # picture. This is the check that caught the released metadata being wrong.
        bad = json.loads(json.dumps({k: v for k, v in row_index.items() if k != "by_episode"}))
        for e in bad["episodes"]:
            e["local_start"] = int(e["local_start"]) + 1
        bad["by_episode"] = {int(e["episode"]): e for e in bad["episodes"]}
        bad_specs = [frames_mod.FrameSpec(**{**s.as_dict(), "local_row": s.local_row + 1})
                     for s in specs[:2]]
        try:
            frames_mod.build_pack(bad_specs, bad, tmp, bank, opengl=False)
            raise AssertionError("a corrupted row location did not raise")
        except RuntimeError as exc:
            assert "expected episode" in str(exc), exc
        return {"synthetic_dataset": info, "n_specs": len(specs),
                "pack_index_entries": len(payload["index"]),
                "wrong_row_location_raises": True, "wrong_projection_convention_raises": True,
                "motion_check_n": motion["n"], "candidates": pack["candidates"],
                "skipped": len(pack["skipped"]),
                "projection_opengl": pack["projection_opengl"],
                "projection_v_flip": pack["projection_v_flip"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def stage_cells(out: pathlib.Path, n_frames_per_suite: int, per_cell_frames: int) -> Dict[str, object]:
    """Every cell, every rule, under three planted physics — twice more with 15 % of edits refused and
    with G2 failing, which is the branch the real checkpoint takes. The labels are the assertion."""
    d = np.sort(core.rng_for("dryrun", "ladder").normal(size=core.N_CODES))
    ladder = core.CodeLadder(d_primary=d, d_local=d + 1e-3 * core.rng_for("dryrun", "l2").normal(size=core.N_CODES))
    pack = _small_pack(n_frames_per_suite)
    results: Dict[str, object] = {}
    # The last case is the branch the REAL checkpoint takes: G2 failed on it (10 of 128 codes distinct),
    # so "k ranks" is undefined, the §8 pivot supplies the spatial clause and — per A20 — the directional
    # clause too. A live branch with no offline coverage is a branch that has never been run.
    for mode, expected, reject, g2_pass in (("epi", "H_epi", 0.0, True), ("gate", "H_gate", 0.0, True),
                                            ("geo", "H_geo", 0.0, True), ("geo", "H_geo", 0.15, True),
                                            ("geo", "H_geo", 0.0, False)):
        actor = StubActor(mode, ladder, reject_rate=reject)
        offline = cells.run_offline(actor, pack, ladder, per_cell_frames=per_cell_frames)
        gates = {
            "G0": core.gate_g0(max(dd["g0_max_abs_delta_norm"] for dd in offline["diagnostics"]),
                               len(offline["diagnostics"])),
            "G1": core.gate_g1(0.62, 0.11, len(offline["diagnostics"])),
            "G2": core.gate_g2(121, 0.002) if g2_pass else core.gate_g2(10, 0.032),
            "G3": core.gate_g3(0.3, 41.0, 0.02, 1.4, n_conventions_passing=1),
        }
        agg = cells.aggregate(offline, gates)
        gates["G4a"] = core.gate_room(_iv(agg["room"]["G4a_rgb_ceiling"]), "G4a")
        gates["G4b"] = core.gate_room(_iv(agg["room"]["G4b_depth_ceiling"]), "G4b")
        agg = cells.aggregate(offline, gates)
        label = agg["decision"]["label"]
        rule2 = agg["decision"]["rule2_read_as_map"]
        want_contrast = "k_ladder" if g2_pass else "within_frame_extreme_swap"
        want_family = "k_ladder" if g2_pass else "pivot_swap"
        # rule 2 is not built at all when rule 1 returns H_epi, and saying so beats asserting on {}
        if rule2:
            assert rule2.get("primary_contrast") == want_contrast, rule2.get("primary_contrast")
        assert agg["directional"]["deciding_family"] == want_family, agg["directional"]["deciding_family"]
        assert set(agg["directional_by_family"]) == set(cells.DIRECTIONAL_FAMILIES)
        assert label == expected, (f"planted {mode} (G2 {'pass' if g2_pass else 'fail'}): pipeline said "
                                  f"{label!r}, wanted {expected!r}")
        assert agg["tail_mismatches"] == 0
        if reject > 0:
            counted = agg["edits_not_completed"] + agg["per_cell_edits_not_completed"]
            assert actor.rejected > 0 and counted == actor.rejected, (
                f"the rejected edits were not recorded: stub raised {actor.rejected}, aggregate "
                f"counted {counted} ({agg['edits_not_completed']} records + "
                f"{agg['per_cell_edits_not_completed']} single-cell)")
            assert agg["records_scored"] + agg["edits_not_completed"] == len(offline["records"])
        else:
            assert agg["edits_not_completed"] == 0
        key = mode if reject == 0 else f"{mode}_reject{reject:g}"
        if not g2_pass:
            key = f"{key}_pivot"
        results[key] = {"label": label, "calls": actor.calls,
                        "edits_not_completed": agg["edits_not_completed"],
                        "per_cell_edits_not_completed": agg["per_cell_edits_not_completed"],
                        "records_scored": agg["records_scored"],
                        "frames_usable": agg["frames_usable"], "frames_total": agg["frames_total"],
                         "m": agg["m"], "room": agg["room"],
                         "shuffle_over_ceiling": agg["ratios"].get("shuffle"),
                         "paired_k8": agg["per_k"].get("8", {}).get("paired_target_minus_bg"),
                         "directional": agg["directional"].get("target"),
                         "directional_family": agg["directional"].get("deciding_family"),
                         "primary_contrast": rule2.get("primary_contrast"),
                         "pivot_swap": agg.get("pivot_swap"),
                         "anchor": agg["anchor"],
                        "gates": {k: v.verdict for k, v in gates.items()}}
        if reject == 0:
            ex = cells.bank_extremes(actor, pack, ladder, offline["records"],
                                     cells=("shuffle", "const", "target", "bg", "random",
                                            "target_swap", "bg_swap"))
            assert ex, "no extreme samples banked"
            worst = 0.0
            for key, rec in ex.items():
                if rec["delta_norm_cont_rerun"] is None:
                    continue
                # the rebuilt edit must be the edit the record came from, to floating point
                worst = max(worst, abs(rec["delta_norm_cont_rerun"] - rec["delta_norm_cont"]))
            assert worst < 1e-9, f"rebuilt edits do not reproduce their records (worst {worst})"
            results.setdefault("extremes", {})[mode] = {"n": len(ex), "max_delta_mismatch": worst}
        if mode == "geo" and reject == 0:
            (out / "dryrun_geo_records.json").write_text(json.dumps(offline["records"][:40], indent=1))
            results["figures"] = _stage_figures(out, offline, agg, ladder)
    return results


def _iv(d: Dict[str, object]) -> core.Interval:
    return core.Interval(point=float(d["point"]), lo=float(d["ci95"][0]), hi=float(d["ci95"][1]),
                         n=int(d["n"]), statistic=str(d["statistic"]))


def _small_pack(n_per_suite: int) -> Dict[str, object]:
    """A pack shaped exactly like the real one, small enough for a stub sweep."""
    rng = core.rng_for("dryrun", "pack")
    frames = []
    for suite in core.SUITES:
        for i in range(n_per_suite):
            fid = f"{suite}:e{i}:f7"
            t_seed = core.rng_for("dryrun-target", fid)
            hit = t_seed.choice(core.N_CELLS, size=3, replace=False)
            target = core.dilate_cells(hit, radius=1)
            bg, dist = core.background_cells(target, int(target.size), core.rng_for("dryrun-bg", fid))
            frames.append({
                "spec": {"frame_id": fid, "suite": suite, "task_key": f"{suite}/{i % 10}",
                         "task_index": i % 10, "language": f"{suite} pretend task {i % 10}",
                         "episode": i, "frame_index": 7, "file_index": 0, "global_row": i * 14 + 7,
                         "episode_length": 14},
                "agentview": rng.integers(0, 255, size=(IMG, IMG, 3), dtype=np.uint8),
                "wrist": rng.integers(0, 255, size=(IMG, IMG, 3), dtype=np.uint8),
                "task": f"{suite} pretend task {i % 10}",
                "state": rng.normal(size=8).astype(np.float32),
                "action": rng.normal(size=7).astype(np.float32),
                "future_eef": rng.normal(size=(10, 3)).astype(np.float32),
                "future_uv": rng.normal(size=(10, 2)).astype(np.float32),
                "future_depth": np.abs(rng.normal(size=10)).astype(np.float32),
                "target_cells": target, "bg_cells": np.asarray(bg), "bg_min_distance": int(dist),
                "axis_into_scene": np.array([0.3, -0.5, 0.8]),
                "camera": {"K": [[80.0, 0, 32.0], [0, 80.0, 32.0], [0, 0, 1.0]]},
            })
    return {"version": "dryrun", "n_frames": len(frames), "image_size": IMG,
            "master_seed": core.MASTER_SEED, "future_steps": 10, "frames": frames}


def stage_closed_loop() -> Dict[str, object]:
    """The confirmation rules on planted rates, including the case they must refuse."""
    normal = core.wilson(38, 40)
    strong = core.closed_loop_confirmation(normal, {
        "C-shuffle": core.wilson(9, 40), "C-const": core.wilson(6, 40),
        "C-far": core.wilson(18, 40), "C-near": core.wilson(21, 40),
        "C-stale": core.wilson(33, 40), "C-fresh": core.wilson(37, 40)}, "H_geo")
    assert strong["verdict"] == "confirmed", strong
    # §6 rule 3 reads BOTH shift arms in the predicted direction; one of the two is not confirmation
    one_arm = core.closed_loop_confirmation(normal, {
        "C-shuffle": core.wilson(9, 40), "C-const": core.wilson(6, 40),
        "C-far": core.wilson(18, 40), "C-near": core.wilson(36, 40)}, "H_geo")
    assert one_arm["confirms_rule1"] and not one_arm["confirms_rule2b"], one_arm
    assert one_arm["verdict"] == "offline only", one_arm
    flat = core.closed_loop_confirmation(normal, {
        "C-shuffle": core.wilson(37, 40), "C-const": core.wilson(38, 40),
        "C-far": core.wilson(38, 40), "C-near": core.wilson(37, 40)}, "H_epi")
    assert flat["verdict"] == "confirmed"
    mixed = core.closed_loop_confirmation(normal, {
        "C-shuffle": core.wilson(37, 40), "C-const": core.wilson(38, 40)}, "H_geo")
    assert mixed["verdict"] == "offline only", mixed
    return {"strong": strong, "one_arm": one_arm, "flat": flat, "mixed": mixed}


def _stage_figures(out: pathlib.Path, offline, agg, ladder) -> Dict[str, object]:
    """Every panel of §10, on synthetic inputs shaped like the real ones. The point is that a figure bug
    is found here and not after six GPU-hours, so all nine are drawn, not the three the offline cells
    happen to supply."""
    from infra.pg import depthanchor_figures as figs
    rng = core.rng_for("dryrun", "figs")
    fidelity = {"per_frame": [{"frame_id": f"f{i}", "suite": core.SUITES[i % 4],
                               "spearman_decoded_vs_da2": float(0.4 + 0.3 * rng.random()),
                               "spearman_shuffled": float(-0.1 + 0.2 * rng.random()),
                               "bin_agreement": float(rng.random())} for i in range(20)]}
    strips = [{"frame_id": f"f{i}", "rgb": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
               "decoded_pred": rng.normal(size=(320, 320)),
               "decoded_edited": rng.normal(size=(320, 320)),
               "target_cells": rng.choice(core.N_CELLS, 6, replace=False)} for i in range(3)]
    arms = ("C-normal", "C-fresh", "C-stale", "C-shuffle", "C-const", "C-far", "C-near")
    # C-normal is planted at 10/10 on every suite on purpose: a rate of exactly 1.0 makes the upper
    # error bar zero, and the 1e-16 version of that killed `figures/1` in matplotlib
    by_arm_suite = {a: {s: dict(zip(("successes", "n", "rate", "ci95"),
                                    (k := (10 if a == "C-normal" else int(2 + 8 * rng.random())), 10,
                                     k / 10, list(core.wilson(k, 10)[1:]))))
                        for s in core.SUITES} for a in arms}
    pooled = {a: ({"successes": 40, "n": 40, "rate": 1.0, "ci95": list(core.wilson(40, 40)[1:])}
                  if a == "C-normal" else
                  {"successes": 20, "n": 40, "rate": 0.5, "ci95": [0.35, 0.65]}) for a in arms}
    closed_loop = {"by_arm_suite": by_arm_suite, "by_suite": {s: {} for s in core.SUITES},
                   "pooled": pooled, "episodes_per_cell": 10, "suite": "all",
                   "traces": {a: (np.cumsum(rng.normal(size=(40, 3)), axis=0) * 0.01).tolist()
                              for a in ("C-normal", "C-shuffle", "C-far")}}
    films = {a: {"frames": rng.integers(0, 255, (6, 48, 48, 3), dtype=np.uint8),
                 "decoded": rng.normal(size=(6, 32, 32))}
             for a in ("C-normal", "C-shuffle", "C-far")}
    made = figs.draw_all(out / "figures", offline=offline, aggregate=agg, ladder=ladder,
                         gates=agg["gates"], fidelity=fidelity, strips=strips,
                         closed_loop=closed_loop, films=films)
    want = {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"}
    missing = want - set(made)
    assert not missing, f"figures not drawn: {sorted(missing)}"
    for name, paths in made.items():
        for path in paths:
            assert pathlib.Path(path).is_file() and pathlib.Path(path).stat().st_size > 1000, \
                f"{name} wrote a file that is empty or missing: {path}"
    return {"drawn": sorted(made), "files": sum(len(v) for v in made.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-suite", type=int, default=4)
    ap.add_argument("--per-cell-frames", type=int, default=1)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[dryrun] 1/5 core self-test", flush=True)
    core._selftest()
    print("[dryrun] 2/5 frame pack on a synthetic LeRobot v3 dataset", flush=True)
    frames_report = stage_frames(out)
    print("[dryrun] 3/5 offline cells under three planted physics, refused edits and the G2-fail pivot",
          flush=True)
    cells_report = stage_cells(out, args.frames_per_suite, args.per_cell_frames)
    print("[dryrun] 4/5 closed-loop confirmation rules", flush=True)
    loop_report = stage_closed_loop()
    print("[dryrun] 5/5 writing metrics", flush=True)
    figures = cells_report.pop("figures", None)
    extremes = cells_report.pop("extremes", None)
    metrics = {"cell": "dryrun", "status": "ok", "frames": frames_report, "cells": cells_report,
               "closed_loop": loop_report, "figures": figures, "extremes": extremes,
               "labels_recovered": {m: r["label"] for m, r in cells_report.items()},
               "edits_not_completed": {m: r["edits_not_completed"] for m, r in cells_report.items()}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics["labels_recovered"], indent=1))
    print("[dryrun] GREEN")


if __name__ == "__main__":
    main()
