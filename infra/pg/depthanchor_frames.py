"""DEPTHANCHOR frame pack — the 120 offline frames, fixed by seed before any cell runs.

`allenai/MolmoAct2-LIBERO-Dataset` is a LeRobot v3 dataset: 1,693 episodes / 273,465 frames in 378
parquet files, images as encoded bytes in a `struct<bytes, path>` column, `observation.state` an 8-vector
and `action` a 7-vector (`meta/info.json`). This module turns it into one small, self-describing pack —
frames, task strings, states, the next ten recorded end-effector positions, the target and background
cells, the camera matrix each frame's projection used — so every later cell reads a 15 MB artifact
instead of 24 GB of parquet, and so the frame list is a banked object that cannot drift between cells.

WHY THE FRAME LIST IS FIXED FIRST (spec §3): the released fine-tune trained on the full LIBERO mixture,
so no frame is truly held out. What is held out is OUR CHOICES: the list is drawn from one master seed
before any Δ is computed and is never reselected. The pack records the draw so a reader can check it.

ROW LOCATIONS DO NOT COME FROM `meta/episodes`. Measured against the data (2026-09-08): its
`dataset_from_index` and `length` are correct for every episode that shipped, but its `data/file_index` is
wrong for 372 of 1,693 episodes and runs 0..378 while only 378 files exist — and three episodes it lists
are absent from the files entirely (1,690 episodes / 273,083 rows against `info.json`'s 1,693 / 273,465).
Deriving a file's offset from the smallest `dataset_from_index` among the episodes meta assigns to it is
exactly the combination that breaks, and the first pack attempt died on it
(`IndexError: file-334.parquet: local row 783 outside 0..765`) — which is what the per-row check below is
for. Every row location instead comes from `infra/pg/depthanchor_rowindex.json`,
built by scanning the data files' OWN `episode_index` / `frame_index` / `task_index` / `index` columns
(~9 MB of int64 for the whole dataset) and checked for contiguity, for one file per episode, for one task
per episode, and for `index` running 0..273,464 across the files in order.

Two things are still checked per frame rather than trusted: after reading a row, `_read_rows` asserts it
carries exactly the episode and frame index that was asked for; and the preflight's dataset-agreement
probe correlates each selected episode's own first frame against the rendered init states of the task the
row index says it belongs to, which is also what would catch a shifted episode-to-task association.
"""
from __future__ import annotations

import io
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from infra.pg import depthanchor_core as core

N_PER_SUITE = 30
MAX_PER_EPISODE = 3
OVERSAMPLE = 2.0           # candidates per kept frame; see `select_frames` and SPEC A13
FUTURE_STEPS = 10          # "the next 10 recorded steps" (spec §2)
PACK_VERSION = "v1"
ARRAY_KEYS = ("agentview", "wrist", "state", "action", "future_eef", "future_uv", "future_depth",
              "target_cells", "bg_cells", "axis_into_scene", "first_agentview", "first_state",
              "next_agentview")


ROW_INDEX = pathlib.Path(__file__).resolve().parent / "depthanchor_rowindex.json"


def load_row_index(path: pathlib.Path = ROW_INDEX) -> Dict[str, object]:
    """The banked row index, with its own invariants re-checked on load: one entry per episode, files in
    order, `index` contiguous from 0, and every episode's frames numbered 0..length-1."""
    payload = json.loads(pathlib.Path(path).read_text())
    files = sorted(payload["files"], key=lambda r: r["file_index"])
    if files[0]["index_min"] != 0:
        raise RuntimeError("the row index does not start at global row 0")
    for a, b in zip(files, files[1:]):
        if a["index_max"] + 1 != b["index_min"]:
            raise RuntimeError(f"the row index is not contiguous between files {a['file_index']} and "
                               f"{b['file_index']}")
    eps = payload["episodes"]
    if len({e["episode"] for e in eps}) != len(eps):
        raise RuntimeError("the row index has a duplicate episode")
    for e in eps:
        if e["frame_first"] != 0 or e["frame_last"] != e["length"] - 1:
            raise RuntimeError(f"episode {e['episode']} does not run 0..length-1: {e}")
    payload["by_episode"] = {int(e["episode"]): e for e in eps}
    return payload


def build_row_index(open_parquet, n_files: int, data_path: str, source: str) -> Dict[str, object]:
    """Build the row index by scanning the data files' own int64 columns. `open_parquet(file_index)`
    returns something `pyarrow.parquet.ParquetFile` accepts, so the same function runs over a local
    directory or straight off S3 — in-region, where it takes about a minute for 378 files.

    Every invariant is asserted here rather than assumed: rows of one episode contiguous, one file per
    episode, one task per episode, frames 0..length-1, files in order, `index` contiguous from 0.
    """
    import pyarrow.parquet as pq
    cols = ["episode_index", "frame_index", "task_index", "index"]
    files: List[Dict[str, object]] = []
    episodes: Dict[int, Dict[str, object]] = {}
    for k in range(int(n_files)):
        tb = pq.ParquetFile(open_parquet(k)).read(columns=cols)
        ei = np.asarray(tb.column("episode_index"))
        fi = np.asarray(tb.column("frame_index"))
        ti = np.asarray(tb.column("task_index"))
        gi = np.asarray(tb.column("index"))
        files.append({"file_index": k, "n_rows": int(ei.size), "index_min": int(gi.min()),
                      "index_max": int(gi.max()), "episode_min": int(ei.min()),
                      "episode_max": int(ei.max())})
        for e in np.unique(ei):
            rows = np.flatnonzero(ei == e)
            if not np.all(np.diff(rows) == 1):
                raise RuntimeError(f"episode {int(e)} is not contiguous in file {k}")
            if len(set(ti[rows].tolist())) != 1:
                raise RuntimeError(f"episode {int(e)} spans more than one task in file {k}")
            if int(e) in episodes:
                raise RuntimeError(f"episode {int(e)} appears in files "
                                   f"{episodes[int(e)]['file_index']} and {k}")
            episodes[int(e)] = {"episode": int(e), "file_index": k, "local_start": int(rows[0]),
                                "length": int(rows.size), "frame_first": int(fi[rows[0]]),
                                "frame_last": int(fi[rows[-1]]), "task_index": int(ti[rows[0]]),
                                "global_start": int(gi[rows[0]])}
    files.sort(key=lambda r: r["file_index"])
    for a, b in zip(files, files[1:]):
        if a["index_max"] + 1 != b["index_min"]:
            raise RuntimeError(f"global row index is not contiguous between files "
                               f"{a['file_index']} and {b['file_index']}")
    if files[0]["index_min"] != 0:
        raise RuntimeError("the first file does not start at global row 0")
    for rec in episodes.values():
        if rec["frame_first"] != 0 or rec["frame_last"] != rec["length"] - 1:
            raise RuntimeError(f"episode {rec['episode']} does not run 0..length-1: {rec}")
    import time as _time
    return {"source": source, "built_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            "why": ("meta/episodes' data/file_index and dataset_from_index do not locate rows in this "
                    "release (file 0 holds episodes 0-2 while meta says 0-3; meta references a file-378 "
                    "that does not exist), so every row location here is read from the data files' own "
                    "episode_index / frame_index / index columns"),
            "n_files": len(files), "n_episodes": len(episodes),
            "total_rows": files[-1]["index_max"] + 1, "data_path": data_path,
            "files": files, "episodes": [episodes[e] for e in sorted(episodes)]}


def read_task_strings(meta_dir: pathlib.Path) -> Dict[int, str]:
    """`meta/tasks.parquet` is an index -> string table and is the only place the task text lives; the
    data files' own `task_index` column keys into it. Nothing about row locations comes from meta."""
    import pyarrow.parquet as pq
    t = pq.read_table(pathlib.Path(meta_dir) / "tasks.parquet").to_pydict()
    return {int(i): str(s) for i, s in zip(t["task_index"], t["task"])}


@dataclass
class FrameSpec:
    frame_id: str
    suite: str
    task_key: str            # "<suite>/<task_id>" into the camera bank
    task_index: int
    language: str
    episode: int
    frame_index: int
    file_index: int
    local_row: int
    global_row: int
    episode_length: int

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in
                ("frame_id", "suite", "task_key", "task_index", "language", "episode",
                 "frame_index", "file_index", "local_row", "global_row", "episode_length")}


# ------------------------------------------------------------------ metadata
def read_meta(meta_dir: pathlib.Path) -> Dict[str, object]:
    """Read `meta/` for the record. NOTHING here locates a row any more: this release's
    `data/file_index` and `dataset_from_index` are wrong (see the module docstring), so the row index is
    built from the data files themselves. Kept because `info.json` and the episode `tasks` field are
    still worth banking as provenance, and because the mismatch is itself a finding worth quantifying."""
    import pyarrow.parquet as pq
    meta_dir = pathlib.Path(meta_dir)
    info = json.loads((meta_dir / "info.json").read_text())
    tasks = pq.read_table(meta_dir / "tasks.parquet").to_pydict()
    ep_files = sorted((meta_dir / "episodes").rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"no episode metadata under {meta_dir / 'episodes'}")
    cols = ["episode_index", "data/chunk_index", "data/file_index", "dataset_from_index",
            "dataset_to_index", "tasks", "length"]
    eps = pq.read_table(ep_files[0], columns=cols).to_pydict()
    for extra in ep_files[1:]:
        more = pq.read_table(extra, columns=cols).to_pydict()
        for k in cols:
            eps[k].extend(more[k])
    n = len(eps["episode_index"])
    if n != int(info["total_episodes"]):
        raise RuntimeError(f"episode metadata has {n} rows, info.json says {info['total_episodes']}")
    return {"info": info, "tasks": tasks, "episodes": eps}


def data_path(data_root: pathlib.Path, chunk_index: int, file_index: int,
              template: str) -> pathlib.Path:
    return pathlib.Path(data_root) / template.format(chunk_index=int(chunk_index),
                                                     file_index=int(file_index))


# ------------------------------------------------------------------ the draw
def select_frames(row_index: Dict[str, object], task_strings: Dict[int, str],
                  suite_of_language: Dict[str, str], n_per_suite: int = N_PER_SUITE,
                  max_per_episode: int = MAX_PER_EPISODE,
                  oversample: float = OVERSAMPLE) -> List[FrameSpec]:
    """The CANDIDATE list, drawn from the master seed off the banked row index. Per suite: shuffle that
    suite's episodes, take up to `max_per_episode` frames from each in a shuffled order until
    `oversample x n_per_suite` candidates are drawn. A frame is eligible only if `FUTURE_STEPS` further
    recorded steps exist in its episode, because the target cells are defined by them.

    Why candidates rather than the final list: a frame whose future end-effector path projects OUTSIDE
    the agentview image has no target cells, and therefore cannot answer the question this study asks —
    measured on the first real pack, `libero_10:e249:f210` is one. `attach_cells` walks this seeded order
    and keeps the first `n_per_suite` per suite whose path is in view, so the list stays a function of the
    master seed alone and the skipped frames are counted rather than quietly replaced (amendment A13).
    """
    by_suite: Dict[str, List[Dict[str, object]]] = {s: [] for s in core.SUITES}
    unmatched = set()
    for e in row_index["episodes"]:
        language = task_strings.get(int(e["task_index"]))
        if language is None:
            raise RuntimeError(f"episode {e['episode']} has task_index {e['task_index']}, which is not "
                               f"in meta/tasks.parquet")
        key = _norm(language)
        if key not in suite_of_language:
            unmatched.add(language)
            continue
        by_suite.setdefault(suite_of_language[key].split("/")[0], []).append(e)
    if unmatched:
        raise RuntimeError(f"{len(unmatched)} dataset task strings matched no LIBERO task: "
                           f"{sorted(unmatched)[:3]}")
    out: List[FrameSpec] = []
    for suite in core.SUITES:
        eps = by_suite.get(suite, [])
        if not eps:
            raise RuntimeError(f"no episodes found for suite {suite}")
        rng = core.rng_for("frames", suite)
        quota = int(round(n_per_suite * float(oversample)))
        taken = 0
        for oi in rng.permutation(len(eps)):
            e = eps[int(oi)]
            eligible = int(e["length"]) - FUTURE_STEPS
            if eligible <= 0:
                continue
            picks = rng.choice(eligible, size=min(max_per_episode, eligible, quota - taken),
                               replace=False)
            language = task_strings[int(e["task_index"])]
            task_key = suite_of_language[_norm(language)]
            for f in sorted(int(x) for x in np.atleast_1d(picks)):
                out.append(FrameSpec(
                    frame_id=f"{suite}:e{int(e['episode'])}:f{f}", suite=suite, task_key=task_key,
                    task_index=int(e["task_index"]), language=language,
                    episode=int(e["episode"]), frame_index=f, file_index=int(e["file_index"]),
                    local_row=int(e["local_start"]) + f,
                    global_row=int(e["global_start"]) + f, episode_length=int(e["length"])))
                taken += 1
            if taken >= quota:
                break
        if taken < n_per_suite:
            raise RuntimeError(f"suite {suite}: drew {taken} candidate frames, needs at least "
                               f"{n_per_suite}")
    return out


def _norm(text: str) -> str:
    return " ".join(str(text).replace("_", " ").lower().split()).strip(" .")


# ------------------------------------------------------------------ reading rows
def _read_rows(path: pathlib.Path, wanted: Sequence[Tuple[int, int, int]]) -> Dict[Tuple[int, int], Dict[str, object]]:
    """`wanted` = (local_row, episode_index, frame_index). Only the row groups that contain a wanted
    row are materialized — the image columns are ~50 MB per row group per camera, so reading the whole
    file for three frames would cost a hundred times what it needs to."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    cols = ["observation.images.image", "observation.images.wrist_image", "observation.state",
            "action", "frame_index", "episode_index", "index", "task_index"]
    by_group: Dict[int, List[Tuple[int, int, int]]] = {}
    offsets = []
    off = 0
    for rg in range(pf.num_row_groups):
        offsets.append(off)
        off += pf.metadata.row_group(rg).num_rows
    total = off
    for row, ep, fi in wanted:
        if not (0 <= row < total):
            raise IndexError(f"{path.name}: local row {row} outside 0..{total - 1}")
        rg = int(np.searchsorted(offsets, row, side="right") - 1)
        by_group.setdefault(rg, []).append((row, ep, fi))
    out: Dict[Tuple[int, int], Dict[str, object]] = {}
    for rg, items in sorted(by_group.items()):
        table = pf.read_row_group(rg, columns=cols)
        base = offsets[rg]
        for row, ep, fi in items:
            local = row - base
            rec = {c: table.column(c)[local].as_py() for c in cols}
            if int(rec["episode_index"]) != int(ep) or int(rec["frame_index"]) != int(fi):
                raise RuntimeError(
                    f"{path.name} row {row}: expected episode {ep} frame {fi}, found episode "
                    f"{rec['episode_index']} frame {rec['frame_index']} — the file offset assumption "
                    "in read_meta is wrong and no frame from this pack is trustworthy")
            out[(int(ep), int(fi))] = rec
    return out


def _decode_image(cell: Dict[str, object]) -> np.ndarray:
    from PIL import Image
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    if raw is None:
        raise ValueError("image cell has no bytes")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


DIHEDRAL = (("id", False, False), ("vflip", False, True), ("hflip", True, False),
            ("rot180", True, True))


def motion_check(frames: Sequence[Dict[str, object]], camera_bank: Dict[str, object], *,
                 opengl: bool, window_cells: float = 1.0, min_eef_move_m: float = 0.008,
                 gap: int = 5, n_null: int = 200) -> Dict[str, object]:
    """WHERE does the banked camera matrix put the end-effector in the DATASET's own images? This is the
    only measurement that ties the projection to the images the offline cells actually read: gate G3 ties
    it to the SIMULATOR, and matching a dataset episode against LIBERO's shipped init states does not
    work because the released episodes do not start from them.

    The statistic is a percentile against a null, and all four dihedral variants are scored, because the
    answer turned out to be one nobody would have assumed:

      * only frames whose end-effector moves at least `min_eef_move_m` over `gap` frames are used, so a
        paused gripper beside a moving object cannot count against the camera;
      * the mean of |image(t+gap) − image(t)| in a one-cell window at the projected pixel is compared
        with the same statistic at `n_null` random windows in the SAME frame;
      * ties are counted at half weight — the images are PNG, so most pixels are EXACTLY equal and a
        strict comparison scores every static window 0, which made a first version read every projection
        as a miss;
      * identity, vertical mirror, horizontal mirror and 180° are all scored, and the winner has to beat
        the runner-up by `min_margin`.

    Measured on this release (file-000, 144 frames): identity 0.357, vflip 0.323, **hflip 0.910**,
    rot180 0.805 — the dataset's images are the horizontal mirror of the projection and are not
    vertically mirrored.
    """
    image_size = int(camera_bank["image_size"])
    half = max(2, int(round(window_cells * image_size / core.GRID / 2)))
    rng = core.rng_for("motion-null", image_size, n_null)
    scores: Dict[str, List[float]] = {name: [] for name, *_ in DIHEDRAL}
    used = skipped = 0
    per_frame: List[Dict[str, object]] = []
    for f in frames:
        img_a = np.asarray(f["agentview"], dtype=np.float64)
        img_b = np.asarray(f.get("next_agentview"), dtype=np.float64)
        future = np.asarray(f["future_eef"], dtype=np.float64)
        if img_b.shape != img_a.shape or future.shape[0] < gap:
            skipped += 1
            continue
        eef_now = np.asarray(f["state"], dtype=np.float64)[0:3]
        eef_then = future[gap - 1]
        if float(np.linalg.norm(eef_then - eef_now)) < min_eef_move_m:
            skipped += 1
            continue
        # `next_agentview` is t+1; the pack does not carry t+gap, so the motion image is t vs t+1 and the
        # eef filter uses the longer horizon only to select frames where the arm is actually moving.
        motion = np.abs(img_b - img_a).sum(axis=2)
        h, w = motion.shape
        cam = camera_bank["tasks"][f["spec"]["task_key"]]
        uv, _ = core.camera_project(np.asarray(cam["K"]), np.asarray(cam["R_world_to_cam"]),
                                   np.asarray(cam["cam_pos_world"]),
                                   np.stack([eef_now, future[0]]), opengl=opengl)
        null = np.array([_window_mean(motion, int(r), int(c), half)
                         for r, c in zip(rng.integers(half, h - half, n_null),
                                         rng.integers(half, w - half, n_null))])
        row = {"frame_id": f["spec"]["frame_id"], "projected_uv": [float(uv[0, 0]), float(uv[0, 1])]}
        good = True
        for name, uf, vf in DIHEDRAL:
            best = 0.0
            for i in range(uv.shape[0]):
                u = (w - 1.0 - uv[i, 0]) if uf else uv[i, 0]
                v = (h - 1.0 - uv[i, 1]) if vf else uv[i, 1]
                if not (half <= u < w - half and half <= v < h - half):
                    good = False
                    break
                best = max(best, _window_mean(motion, int(round(v)), int(round(u)), half))
            if not good:
                break
            row[name] = float(((null < best).sum() + 0.5 * (null == best).sum()) / null.size)
        if not good:
            skipped += 1
            continue
        for name, *_ in DIHEDRAL:
            scores[name].append(row[name])
        per_frame.append(row)
        used += 1
    out: Dict[str, object] = {"n": used, "n_frames_seen": len(frames), "n_skipped": skipped,
                              "window_px": 2 * half + 1, "n_null_windows": n_null, "gap": gap,
                              "min_eef_move_m": min_eef_move_m, "per_frame": per_frame[:40]}
    if used < 20:
        return {**out, "u_flip": None, "v_flip": None, "decisive": False,
                "why": "fewer than 20 frames with a moving end-effector and an in-frame projection"}
    med = {name: float(np.median(scores[name])) for name, *_ in DIHEDRAL}
    order = sorted(med, key=lambda k: med[k], reverse=True)
    win, runner = order[0], order[1]
    flips = {name: (uf, vf) for name, uf, vf in DIHEDRAL}
    return {**out, "median_percentile": med, "winner": win, "runner_up": runner,
            "margin": med[win] - med[runner],
            "u_flip": bool(flips[win][0]), "v_flip": bool(flips[win][1]),
            "decisive": bool(med[win] >= 0.75 and (med[win] - med[runner]) >= 0.05)}


def _window_mean(image: np.ndarray, row: int, col: int, half: int) -> float:
    r0, r1 = max(0, row - half), min(image.shape[0], row + half + 1)
    c0, c1 = max(0, col - half), min(image.shape[1], col + half + 1)
    return float(image[r0:r1, c0:c1].mean())


# ------------------------------------------------------------------ the pack, in two passes
def read_frames(specs: Sequence[FrameSpec], row_index: Dict[str, object],
                data_root: pathlib.Path) -> List[Dict[str, object]]:
    """Pass one: read every selected frame, its next `FUTURE_STEPS` recorded states, and its episode's
    first frame — located by the banked row index, not by `meta/episodes`. No projection here: the
    convention is decided by G3 and by the dataset-agreement probe, and the probe needs these images."""
    by_episode = row_index["by_episode"]
    template = str(row_index["data_path"])
    wanted: Dict[int, List[Tuple[int, int, int]]] = {}
    for s in specs:
        e = by_episode[int(s.episode)]
        if int(e["file_index"]) != int(s.file_index):
            raise RuntimeError(f"{s.frame_id}: the spec says file {s.file_index}, the row index says "
                               f"{e['file_index']}")
        want = wanted.setdefault(int(s.file_index), [])
        want.append((int(s.local_row), s.episode, s.frame_index))
        # frame_index + 1 is read for its state anyway; its IMAGE is what the motion check needs
        for step in range(1, FUTURE_STEPS + 1):
            want.append((int(s.local_row) + step, s.episode, s.frame_index + step))
        want.append((int(e["local_start"]), s.episode, 0))
    rows: Dict[Tuple[int, int], Dict[str, object]] = {}
    for file_index, items in sorted(wanted.items()):
        path = pathlib.Path(data_root) / template.format(file_index=file_index)
        rows.update(_read_rows(path, sorted(set(items))))
    frames: List[Dict[str, object]] = []
    for s in specs:
        rec = rows[(s.episode, s.frame_index)]
        first = rows[(s.episode, 0)]
        future = np.stack([np.asarray(rows[(s.episode, s.frame_index + k)]["observation.state"],
                                      dtype=np.float64)[0:3] for k in range(1, FUTURE_STEPS + 1)])
        if int(rec["task_index"]) != int(s.task_index):
            raise RuntimeError(f"{s.frame_id}: the row's own task_index is {rec['task_index']}, the "
                               f"spec says {s.task_index}")
        frames.append({
            "spec": s.as_dict(),
            "agentview": _decode_image(rec["observation.images.image"]),
            "wrist": _decode_image(rec["observation.images.wrist_image"]),
            "first_agentview": _decode_image(first["observation.images.image"]),
            "next_agentview": _decode_image(
                rows[(s.episode, s.frame_index + 1)]["observation.images.image"]),
            "first_state": np.asarray(first["observation.state"], dtype=np.float32),
            "task": s.language,
            "state": np.asarray(rec["observation.state"], dtype=np.float64).astype(np.float32),
            "action": np.asarray(rec["action"], dtype=np.float32),
            "future_eef": future.astype(np.float32)})
    return frames


def attach_cells(frames: List[Dict[str, object]], camera_bank: Dict[str, object], *,
                 opengl: bool, v_flip: bool, u_flip: bool = False,
                 n_per_suite: int = N_PER_SUITE) -> Dict[str, object]:
    """Pass two: project each frame's future end-effector path with the convention gate G3 chose, derive
    the target cells and their count-matched background, and derive the viewing axis from the SAME
    convention. Two per-frame assertions, because a silent sign error here mirrors the whole study:
    every projected point must be in front of the camera, and the viewing axis must point from the camera
    towards the path it just projected."""
    image_size = int(camera_bank["image_size"])
    kept: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []
    per_suite: Dict[str, int] = {}
    for f in frames:
        spec = f["spec"]
        if per_suite.get(spec["suite"], 0) >= n_per_suite:
            continue
        cam = camera_bank["tasks"][spec["task_key"]]
        R = np.asarray(cam["R_world_to_cam"], dtype=np.float64)
        t = np.asarray(cam["cam_pos_world"], dtype=np.float64)
        future = np.asarray(f["future_eef"], dtype=np.float64)
        uv, z = core.camera_project(np.asarray(cam["K"]), R, t, future, opengl=opengl)
        if not np.all(np.isfinite(z)) or float(np.min(z)) <= 0:
            raise RuntimeError(f"{spec['frame_id']}: the future end-effector path projects behind the "
                               f"camera under the chosen convention (min depth {float(np.min(z)):.3f} m)")
        axis = core.axis_into_scene(R, opengl=opengl)
        towards = np.asarray(future.mean(axis=0)) - t
        if float(axis @ towards) <= 0:
            raise RuntimeError(f"{spec['frame_id']}: the viewing axis points away from the scene "
                               f"(axis·(path − camera) = {float(axis @ towards):.3f}); the sign "
                               "convention and the banked extrinsics disagree")
        hit = core.cells_from_pixels(uv, (image_size, image_size), v_flip=v_flip, u_flip=u_flip)
        if hit.size == 0:
            # No target cells: the gripper's next ten positions are all outside the agentview image, so
            # this frame cannot answer the question. Skipped and COUNTED, never silently replaced.
            skipped.append({"frame_id": spec["frame_id"], "suite": spec["suite"],
                            "why": "future end-effector path projects outside the agentview image",
                            "uv_first": [float(uv[0, 0]), float(uv[0, 1])],
                            "depth_m_first": float(z[0])})
            continue
        target = core.dilate_cells(hit, radius=1)
        bg, bg_distance = core.background_cells(target, int(target.size),
                                                core.rng_for("bg", spec["frame_id"]))
        f.update({"future_uv": uv.astype(np.float32), "future_depth": z.astype(np.float32),
                  "target_cells": target.astype(np.int64), "bg_cells": np.asarray(bg, dtype=np.int64),
                  "bg_min_distance": int(bg_distance),
                  "axis_into_scene": axis, "camera": cam})
        kept.append(f)
        per_suite[spec["suite"]] = per_suite.get(spec["suite"], 0) + 1
    short = {s: n_per_suite - per_suite.get(s, 0) for s in core.SUITES
             if per_suite.get(s, 0) < n_per_suite}
    if short:
        raise RuntimeError(f"not enough frames with a visible future end-effector path: {short} short "
                           f"(skipped {len(skipped)} of {len(frames)} candidates); raise OVERSAMPLE")
    return {"version": PACK_VERSION, "n_frames": len(kept), "image_size": image_size,
            "master_seed": core.MASTER_SEED, "future_steps": FUTURE_STEPS, "frames": kept,
            "projection_opengl": bool(opengl), "projection_v_flip": bool(v_flip),
            "projection_u_flip": bool(u_flip),
            "candidates": len(frames), "skipped": skipped,
            "skipped_per_suite": {s: sum(1 for x in skipped if x["suite"] == s) for s in core.SUITES}}


def build_pack(specs: Sequence[FrameSpec], row_index: Dict[str, object], data_root: pathlib.Path,
               camera_bank: Dict[str, object], opengl: bool = True, v_flip: bool = False,
               u_flip: bool = False) -> Dict[str, object]:
    """The two passes together, for callers that already know the convention (the dry run)."""
    return attach_cells(read_frames(specs, row_index, data_root), camera_bank, opengl=opengl,
                        v_flip=v_flip, u_flip=u_flip)


def save_pack(pack: Dict[str, object], out_dir: pathlib.Path) -> Dict[str, object]:
    """One npz for the arrays, one json for everything a human should be able to read without numpy."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    index: List[Dict[str, object]] = []
    for i, f in enumerate(pack["frames"]):
        for key in ARRAY_KEYS:
            arrays[f"{i}:{key}"] = np.asarray(f[key])
        index.append({**f["spec"], "task": f["task"], "bg_min_distance": f["bg_min_distance"],
                      "n_target_cells": int(np.asarray(f["target_cells"]).size),
                      "camera": f["camera"]})
    np.savez_compressed(out_dir / "frames.npz", **arrays)
    payload = {"version": pack["version"], "n_frames": pack["n_frames"],
               "image_size": pack["image_size"], "master_seed": pack["master_seed"],
               "future_steps": pack["future_steps"],
               "projection_opengl": pack.get("projection_opengl"),
               "projection_v_flip": pack.get("projection_v_flip"),
               "projection_u_flip": pack.get("projection_u_flip"),
               "candidates": pack.get("candidates"), "skipped": pack.get("skipped"),
               "skipped_per_suite": pack.get("skipped_per_suite"), "index": index}
    (out_dir / "frames_index.json").write_text(json.dumps(payload, indent=1))
    return payload


def load_pack(pack_dir: pathlib.Path) -> Dict[str, object]:
    pack_dir = pathlib.Path(pack_dir)
    payload = json.loads((pack_dir / "frames_index.json").read_text())
    arrays = np.load(pack_dir / "frames.npz")
    frames = []
    for i, entry in enumerate(payload["index"]):
        f = {"spec": entry, "task": entry["task"], "camera": entry["camera"],
             "bg_min_distance": entry["bg_min_distance"]}
        for key in ARRAY_KEYS:
            f[key] = arrays[f"{i}:{key}"]
        frames.append(f)
    payload["frames"] = frames
    return payload
