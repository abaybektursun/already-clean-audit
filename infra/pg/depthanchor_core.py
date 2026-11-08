"""DEPTHANCHOR pure core — every rule of `studies/depthanchor/SPEC.md` §2/§5/§6/§7 as code.

Pure numpy. No torch, no model, no GPU, no network: everything here runs on the control plane, which
is what makes the offline gate (`python -m infra.aws_pg.jobs.depthanchor_dryrun`) able to exercise
every cell, every gate branch and every decision rule before a GPU-hour is spent.

WHAT LIVES HERE
  * the frozen unit definitions (§2): which number is Δ, in which units, on which dimensions;
  * the nine offline edits (§3) as pure functions on a 100-vector of depth-code bins;
  * the projection arithmetic that turns a banked camera matrix + a future end-effector path into
    target cells and their count-matched background (§2);
  * the statistics (§7): frame-level suite-stratified bootstrap, Spearman with a bg null band,
    Wilson intervals, Benjamini-Hochberg (reused from `matrix_core`);
  * every gate as a function with a floor, a ceiling and a CANCELLED branch, plus the registered
    input that makes it fail (§5, and closeout §4.1: "thresholds frozen without a floor measured on
    control data; three gates could not fire");
  * the decision rules of §6, which read the gates and refuse to produce a label when a gate is
    CANCELLED.

`python -m infra.pg.depthanchor_core` runs the self-test: every branch of every gate reached, every
edit checked against a property that would catch it being a no-op, the bootstrap checked against a
planted effect, the decision rules checked on synthetic populations built to give each label.

UNITS, STATED ONCE (spec §2, amended — see studies/depthanchor/SPEC.md §13 amendment A2/A3):
  `a_norm`  the action chunk in the checkpoint's NORMALIZED units, exactly the tensor the flow solver
            produced, captured BEFORE `_FeatureNormalizer.unnormalize` clips it to [-1,1]. Shape
            (10, 7). Dimensions 0..5 are q01_q99-normalized; dimension 6 (gripper) is not normalized
            for `norm_tag="libero"` (`normalize_gripper: false`), and its raw values sit at ±1, so it
            is a saturating flip indicator, not a continuous coordinate.
  `a_robot` the same chunk in robot units, as `predict_action` returns it: clip to [-1,1] then the
            affine q01/q99 map. This is what the simulator executes. Dimensions 0..2 share one unit
            (±0.9375), so this — never `a_norm` — is what a geometric projection may use.
  PRIMARY   `delta_norm_cont` = mean over the 10 chunk steps and dimensions 0..5 of |Δa_norm|.
  SECONDARY `delta_robot_all` = mean over 10 steps and dimensions 0..6 of |Δa_robot|;
            `grip_flip` = fraction of chunk steps whose gripper sign differs.
  DIRECTION `delta_z` = mean over the 10 steps of (Δa_robot[:, 0:3] · axis_into_scene), robot units,
            positive = the edited action pushes the end-effector FARTHER from the agentview camera.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from infra.pg.matrix_core import bh_reject

# ------------------------------------------------------------------ frozen shapes and constants
GRID = 10                       # config.json num_depth_codes = 100 = 10 x 10
N_CELLS = GRID * GRID
N_CODES = 128                   # config.json num_depth_tokens
ACTION_DIM = 7                  # norm_stats libero action_stats width
CHUNK = 10                      # norm_stats libero n_action_steps / action_horizon
CONT_DIMS = tuple(range(6))     # the six q01_q99-normalized dimensions
GRIP_DIM = 6
DEPTH_PATCH = 32                # modeling_molmoact2._DEPTH_REASONING_PATCH_SIZE
DEPTH_SIM_THRESHOLD = 0.996     # modeling_molmoact2._DEPTH_REASONING_THRESHOLD
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MASTER_SEED = 20260908
N_BOOT = 2000
K_LEVELS = (-16, -8, -4, 4, 8, 16)
PRIMARY_ABS_K = 8
SEED_FLOOR_N = 8                # O-seed: 8 generator seeds
SHUFFLE_N = 8                   # O-shuffle: 8 permutations
CANCELLED = "CANCELLED"         # a readout whose preconditions failed; never scored, always stamped


class EditNotCompleted(RuntimeError):
    """The released code raises when, after an injected depth grid, the model does not emit `<depth_end>`
    within `max(8, action_horizon)` unconstrained steps. That is a fact about the edit — an
    out-of-distribution prefix can leave the model unable to close its own depth block — not a bug, so it
    is raised as its own type: a cell records it and carries on instead of losing a three-hour session,
    while anything else that goes wrong still propagates. Lives here, in the torch-free core, so the
    offline gate can exercise the path without importing torch."""


class NoTargetCells(RuntimeError):
    """At this state there is nothing to aim at: the policy's predicted path AND the end-effector both
    project outside the image, so `T(f)` is empty. A13 met the same fact in the offline pack and skipped
    those frames; a rollout cannot pause, so the step runs the model's own grid and the arm counts it.
    Its own type, because "no cells to edit" and "the model would not close its block after my edit" are
    different facts about different things."""


# ------------------------------------------------------------------ deterministic seeding
def rng_for(*labels) -> np.random.Generator:
    """One master seed, one derived stream per named draw, so any cell can be re-run alone and land
    on the same frames, permutations and background cells. The label is part of the stream, so
    'frame 7's background cells' cannot silently become 'frame 8's'."""
    key = "|".join(str(x) for x in (MASTER_SEED,) + labels).encode()
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


# ------------------------------------------------------------------ §2 the primary quantities
def delta_norm_cont(a_ref_norm: np.ndarray, a_edit_norm: np.ndarray) -> float:
    """PRIMARY Δ: mean |difference| over the chunk and the six continuous dimensions, normalized units."""
    _check_chunk(a_ref_norm)
    _check_chunk(a_edit_norm)
    d = np.abs(np.asarray(a_edit_norm, dtype=np.float64) - np.asarray(a_ref_norm, dtype=np.float64))
    return float(d[:, list(CONT_DIMS)].mean())


def delta_robot_all(a_ref_robot: np.ndarray, a_edit_robot: np.ndarray) -> float:
    """SECONDARY Δ: mean |difference| over the chunk and all seven dimensions, robot units."""
    _check_chunk(a_ref_robot)
    _check_chunk(a_edit_robot)
    d = np.abs(np.asarray(a_edit_robot, dtype=np.float64) - np.asarray(a_ref_robot, dtype=np.float64))
    return float(d.mean())


def grip_flip(a_ref_robot: np.ndarray, a_edit_robot: np.ndarray) -> float:
    """Fraction of chunk steps whose gripper command changes sign. Reported beside every Δ because
    the gripper dimension saturates at ±1 and a mean-absolute Δ would hide a flip in six smooth
    dimensions (and would count a flip as 2.0 in a mean of seven)."""
    _check_chunk(a_ref_robot)
    _check_chunk(a_edit_robot)
    ref = np.sign(np.asarray(a_ref_robot)[:, GRIP_DIM])
    edit = np.sign(np.asarray(a_edit_robot)[:, GRIP_DIM])
    return float(np.mean(ref != edit))


def delta_z(a_ref_robot: np.ndarray, a_edit_robot: np.ndarray, axis_into_scene: np.ndarray) -> float:
    """DIRECTIONAL Δ: the chunk-mean change in translation projected on the camera's viewing
    direction, robot units, positive = farther from the camera. Robot units only: the three
    normalized translation dimensions have different scales and would not project geometrically."""
    _check_chunk(a_ref_robot)
    _check_chunk(a_edit_robot)
    axis = np.asarray(axis_into_scene, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"axis_into_scene must be a 3-vector, got {axis.shape}")
    n = float(np.linalg.norm(axis))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("axis_into_scene is degenerate")
    axis = axis / n
    d = np.asarray(a_edit_robot, dtype=np.float64)[:, 0:3] - np.asarray(a_ref_robot, dtype=np.float64)[:, 0:3]
    return float((d @ axis).mean())


def _check_chunk(a) -> None:
    arr = np.asarray(a)
    if arr.shape != (CHUNK, ACTION_DIM):
        raise ValueError(f"action chunk must be ({CHUNK}, {ACTION_DIM}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action chunk contains a non-finite value")


# ------------------------------------------------------------------ §2 the code ladder
@dataclass
class CodeLadder:
    """`d(c)`: the decoded depth of each of the 128 codes, and the rank order it induces.

    `d_primary` is the mean decoded value of a uniform grid of the code (the spec's definition).
    `d_local` is the mean over the code's own 32x32 patch when it is placed alone in a grid of the
    median code — the second placement, which measures how much of `d_primary` is the decoder's own
    context rather than the code. G2 compares neighbour separation against |d_primary - d_local|.
    Higher decoded value = larger Depth-Anything disparity = NEARER the camera; the sign is an
    instrument fact, confirmed against simulator ground truth in G1B before any directional clause
    is read (§5 G1, amendment A4).
    """
    d_primary: np.ndarray            # (128,)
    d_local: np.ndarray              # (128,)
    order: np.ndarray = field(init=False)      # code ids sorted by d_primary ascending (farther first)
    rank: np.ndarray = field(init=False)       # rank[code] = its position in `order`

    def __post_init__(self):
        self.d_primary = np.asarray(self.d_primary, dtype=np.float64).reshape(-1)
        self.d_local = np.asarray(self.d_local, dtype=np.float64).reshape(-1)
        if self.d_primary.shape != (N_CODES,) or self.d_local.shape != (N_CODES,):
            raise ValueError("a code ladder needs one value per code, for both placements")
        if not np.all(np.isfinite(self.d_primary)) or not np.all(np.isfinite(self.d_local)):
            raise ValueError("code ladder contains a non-finite decoded depth")
        self.order = np.argsort(self.d_primary, kind="stable").astype(np.int64)
        self.rank = np.empty(N_CODES, dtype=np.int64)
        self.rank[self.order] = np.arange(N_CODES, dtype=np.int64)

    @property
    def median_code(self) -> int:
        """The code at the middle of the ladder — O-const's 'median-depth code'."""
        return int(self.order[N_CODES // 2])

    def shift(self, code: int, k: int) -> int:
        """The code `k` ranks farther along the ladder, clamped at the ends. Clamping is why every
        edit reports its REALIZED shift: at the ends of the ladder a requested +16 is not a +16."""
        r = int(self.rank[int(code)]) + int(k)
        r = max(0, min(N_CODES - 1, r))
        return int(self.order[r])

    def depth_of(self, code) -> np.ndarray:
        return self.d_primary[np.asarray(code, dtype=np.int64)]


# ------------------------------------------------------------------ §3 the nine offline edits
@dataclass
class Edit:
    """An edit and what it REALIZED, not what it intended (repo rule: audit realized effects)."""
    name: str
    bins: np.ndarray                 # (100,) int64, the injected grid
    cells: Tuple[int, ...]           # the cells the edit was aimed at
    n_changed: int                   # how many cells actually hold a different code
    realized_rank_shift: float       # mean signed rank shift over the aimed cells
    realized_depth_shift: float      # mean signed change in d(c) over the aimed cells (decoded units)
    meta: Dict[str, object] = field(default_factory=dict)


def _as_bins(b) -> np.ndarray:
    arr = np.asarray(b, dtype=np.int64).reshape(-1)
    if arr.shape != (N_CELLS,):
        raise ValueError(f"depth bins must be a {N_CELLS}-vector, got {arr.shape}")
    if arr.min() < 0 or arr.max() >= N_CODES:
        raise ValueError(f"depth bins out of range [0,{N_CODES}): [{arr.min()},{arr.max()}]")
    return arr


def _finish(name: str, b0: np.ndarray, b1: np.ndarray, cells: Sequence[int],
            ladder: Optional[CodeLadder], meta=None) -> Edit:
    cells = tuple(int(c) for c in cells)
    n_changed = int(np.sum(b0 != b1))
    if ladder is None or not cells:
        rs = ds = 0.0
    else:
        idx = np.asarray(cells, dtype=np.int64)
        rs = float(np.mean(ladder.rank[b1[idx]].astype(np.float64) - ladder.rank[b0[idx]].astype(np.float64)))
        ds = float(np.mean(ladder.d_primary[b1[idx]] - ladder.d_primary[b0[idx]]))
    return Edit(name=name, bins=b1, cells=cells, n_changed=n_changed,
                realized_rank_shift=rs, realized_depth_shift=ds, meta=dict(meta or {}))


def edit_identity(b) -> Edit:
    b0 = _as_bins(b)
    return _finish("ident", b0, b0.copy(), (), None)


def edit_shuffle(b, rng: np.random.Generator, ladder: Optional[CodeLadder] = None) -> Edit:
    """Same multiset of codes, different places. Anything it moves is not spatial. Retried until the
    permutation actually moves something — an identity permutation would silently become a second
    O-ident and read as 'the prefix does nothing'."""
    b0 = _as_bins(b)
    if len(np.unique(b0)) == 1:
        return _finish("shuffle", b0, b0.copy(), tuple(range(N_CELLS)), ladder,
                       {"degenerate_uniform_grid": True})
    for _ in range(64):
        perm = rng.permutation(N_CELLS)
        b1 = b0[perm]
        if np.any(b1 != b0):
            return _finish("shuffle", b0, b1, tuple(range(N_CELLS)), ladder,
                           {"perm": perm.astype(int).tolist()})
    raise RuntimeError("could not draw a non-identity permutation in 64 tries")


def edit_const(b, ladder: CodeLadder) -> Edit:
    """Every cell set to the median-depth code: a whole-prefix change with no spatial structure."""
    b0 = _as_bins(b)
    code = ladder.median_code
    b1 = np.full(N_CELLS, code, dtype=np.int64)
    return _finish("const", b0, b1, tuple(range(N_CELLS)), ladder, {"code": int(code)})


def edit_shift_cells(b, cells: Sequence[int], k: int, ladder: CodeLadder, name: str) -> Edit:
    b0 = _as_bins(b)
    b1 = b0.copy()
    for c in cells:
        b1[int(c)] = ladder.shift(int(b0[int(c)]), int(k))
    return _finish(name, b0, b1, cells, ladder, {"k": int(k)})


def edit_random_cells(b, cells: Sequence[int], rng: np.random.Generator, ladder: CodeLadder) -> Edit:
    """Same count, arbitrary codes: separates 'the code changed' from 'the depth changed'. Each
    drawn code is forced to differ from the one it replaces, so the count-match is real."""
    b0 = _as_bins(b)
    b1 = b0.copy()
    for c in cells:
        c = int(c)
        choices = np.delete(np.arange(N_CODES, dtype=np.int64), int(b0[c]))
        b1[c] = int(rng.choice(choices))
    return _finish(name="random", b0=b0, b1=b1, cells=cells, ladder=ladder)


def edit_swap_extreme(b, cells: Sequence[int], which: str, ladder: CodeLadder) -> Edit:
    """The §8 PIVOT for a failed G2: instead of "k ranks along the ladder", each aimed cell takes the code
    of the FARTHEST or NEAREST cell already present in this frame's own grid. It needs no ordinal
    structure beyond "which of these 100 codes decodes deepest", which survives a ladder too coarse for
    ±k to mean anything — and it is a within-frame edit, so the injected codes are ones the model itself
    produced for this image. Run always, so the pivot exists whichever way G2 lands; it is the primary
    contrast only when G2 does not pass."""
    if which not in ("far", "near"):
        raise ValueError(f"which must be 'far' or 'near', got {which!r}")
    b0 = _as_bins(b)
    d = ladder.d_primary[b0]
    # higher decoded value = nearer the camera (the registered convention, confirmed by G1B)
    pick = int(b0[int(np.argmin(d))]) if which == "far" else int(b0[int(np.argmax(d))])
    b1 = b0.copy()
    for c in cells:
        b1[int(c)] = pick
    return _finish(f"swap_{which}", b0, b1, cells, ladder,
                   {"which": which, "code": pick,
                    "code_depth": float(ladder.d_primary[pick])})


def edit_single_cell(b, cell: int, k: int, ladder: CodeLadder) -> Edit:
    return edit_shift_cells(b, [int(cell)], int(k), ladder, name=f"cell{int(cell)}_{int(k):+d}")


# ------------------------------------------------------------------ §2 target and background cells
def camera_project(K: np.ndarray, R: np.ndarray, t: np.ndarray, points_world: np.ndarray,
                   opengl: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """World points -> (pixel uv, depth along the optical axis).

    `R` is world->camera and `t` the camera position in world coordinates, as banked from robosuite's
    `get_camera_extrinsic_matrix`. Two sign conventions are possible for what happens next and only one
    is right for this simulator: MuJoCo/OpenGL cameras look down their own -z with +y up, so the pixel
    projection uses (x, -y, -z) (`opengl=True`); a plain OpenCV camera looks down +z with +y down and
    uses (x, y, z) (`opengl=False`). Which one holds is not asserted here — gate G3 projects the
    end-effector under BOTH and compares each against the rendered depth buffer, and the convention
    that agrees is banked and then used everywhere. `depth` is positive in front of the camera.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    cam = (R @ (pts - t[None, :]).T).T                     # world -> camera frame
    if opengl:
        x, y, z = cam[:, 0], -cam[:, 1], -cam[:, 2]
    else:
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
    return np.stack([u, v], axis=1), z


def cells_from_pixels(uv: np.ndarray, image_hw: Tuple[int, int],
                     v_flip: bool = False, u_flip: bool = False) -> np.ndarray:
    """Pixel coordinates -> flat cell index in the 10x10 depth grid. The grid tiles the agentview image
    after an isotropic resize to 320x320 (`_compute_depth_update_mask`), so a cell is exactly one tenth
    of the image in each direction whatever the source resolution. Points outside the image are dropped.

    `u_flip` and `v_flip` are the measured relation between the projection and the image the MODEL is
    given, and both had to be measured. In this release the dataset's agentview images are the HORIZONTAL
    mirror of the projection (correlation between the projected column and the moving arm's column:
    −0.875 as projected, +0.875 mirrored; median |u + cx − 255| = 21 px against |u − cx| = 84 px) and are
    NOT vertically mirrored (correlation +0.767 as projected). Getting either wrong mirrors every target
    cell while every number still looks plausible.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    u = (w - 1.0) - uv[:, 0] if u_flip else uv[:, 0]
    v = (h - 1.0) - uv[:, 1] if v_flip else uv[:, 1]
    col = np.floor(u / w * GRID)
    row = np.floor(v / h * GRID)
    ok = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < GRID) & (row >= 0) & (row < GRID)
    return (row[ok] * GRID + col[ok]).astype(np.int64)


def axis_into_scene(R_world_to_cam: np.ndarray, opengl: bool) -> np.ndarray:
    """The camera's viewing direction in world coordinates, under the SAME sign convention the pixel
    projection uses. `R_world_to_cam` is the banked world->camera rotation, so the camera's own z axis in
    world coordinates is its third ROW; the viewing direction is that axis for an OpenCV-style camera
    (+z into the scene) and its negation for an OpenGL-style one (-z into the scene). Deriving it here
    rather than banking it means the axis cannot disagree with the projection that produced the cells."""
    R = np.asarray(R_world_to_cam, dtype=np.float64).reshape(3, 3)
    axis = R[2, :] if not opengl else -R[2, :]
    n = float(np.linalg.norm(axis))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("degenerate camera rotation: no viewing direction")
    return axis / n


def dilate_cells(cells: Sequence[int], radius: int = 1) -> np.ndarray:
    out = set()
    for c in cells:
        r0, c0 = divmod(int(c), GRID)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, cc = r0 + dr, c0 + dc
                if 0 <= r < GRID and 0 <= cc < GRID:
                    out.add(r * GRID + cc)
    return np.array(sorted(out), dtype=np.int64)


def cell_distance_matrix() -> np.ndarray:
    rows, cols = np.divmod(np.arange(N_CELLS), GRID)
    dr = np.abs(rows[:, None] - rows[None, :])
    dc = np.abs(cols[:, None] - cols[None, :])
    return np.maximum(dr, dc)                              # Chebyshev: 'two cells away' in the grid


_CELL_DIST = cell_distance_matrix()


def background_cells(target: Sequence[int], n: int, rng: np.random.Generator,
                     min_distance: int = 2) -> np.ndarray:
    """A count-matched set at least `min_distance` cells from every target cell. If the far pool is
    smaller than the target count the pool is relaxed one ring at a time and the relaxation is
    reported by the caller — a silently short background set would break the count match that makes
    O-bg-k a control at all."""
    target = np.asarray(sorted({int(c) for c in target}), dtype=np.int64)
    if target.size == 0:
        raise ValueError("no target cells: a background set has nothing to be matched against")
    for relax in range(0, int(min_distance)):
        d = int(min_distance) - relax
        pool = np.array([c for c in range(N_CELLS) if int(_CELL_DIST[c, target].min()) >= d],
                        dtype=np.int64)
        if pool.size >= n:
            return np.sort(rng.choice(pool, size=int(n), replace=False)), d
    raise ValueError(f"cannot draw {n} background cells: target covers too much of the grid "
                     f"({target.size} cells)")


# ------------------------------------------------------------------ §7 statistics
def _boot_indices(groups: Sequence[str], n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Suite-stratified frame resampling: each bootstrap replicate redraws frames within each suite,
    keeping the per-suite counts fixed, so a suite with more frames cannot come and go."""
    groups = np.asarray(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    out = np.empty((n_boot, len(groups)), dtype=np.int64)
    pos = 0
    for idx in idx_by_group:
        draw = rng.integers(0, idx.size, size=(n_boot, idx.size))
        out[:, pos:pos + idx.size] = idx[draw]
        pos += idx.size
    return out


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int
    statistic: str

    def excludes(self, value: float) -> bool:
        return bool(self.lo > value or self.hi < value)

    def as_dict(self) -> Dict[str, object]:
        return {"point": self.point, "ci95": [self.lo, self.hi], "n": self.n,
                "statistic": self.statistic}


def bootstrap_stat(values: Sequence[float], groups: Sequence[str], statistic: str = "median",
                   n_boot: int = N_BOOT, seed_label: str = "boot") -> Interval:
    """Frame-level suite-stratified bootstrap of the median (or mean) of a per-frame quantity."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("bootstrap of an empty sample")
    if not np.all(np.isfinite(v)):
        raise ValueError("bootstrap input contains a non-finite value (a CANCELLED readout must not be scored)")
    fn = {"median": np.median, "mean": np.mean}[statistic]
    rng = rng_for("bootstrap", statistic, seed_label, len(v))
    idx = _boot_indices(groups, n_boot, rng)
    reps = fn(v[idx], axis=1)
    return Interval(point=float(fn(v)), lo=float(np.percentile(reps, 2.5)),
                    hi=float(np.percentile(reps, 97.5)), n=int(v.size), statistic=statistic)


def bootstrap_paired_difference(a: Sequence[float], b: Sequence[float], groups: Sequence[str],
                                statistic: str = "median", n_boot: int = N_BOOT,
                                seed_label: str = "paired") -> Interval:
    """Interval on the median/mean of the WITHIN-FRAME difference a - b. Pairing is the point: every
    edit is compared with its own frame's reference, so frame-to-frame scale drops out."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired difference needs matched samples, got {a.shape} and {b.shape}")
    return bootstrap_stat(a - b, groups, statistic=statistic, n_boot=n_boot,
                          seed_label=f"{seed_label}:diff")


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho with average ranks for ties. Written out rather than imported so the dry-run
    gate needs nothing but numpy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 3:
        raise ValueError(f"spearman needs matched samples of length >= 3, got {x.shape}, {y.shape}")
    rx, ry = _avg_rank(x), _avg_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if den < 1e-12:
        return 0.0
    return float((rx * ry).sum() / den)


def _avg_rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    ranks[order] = np.arange(v.size, dtype=np.float64)
    # average the ranks of ties
    sv = v[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i, j + 1, dtype=np.float64))
        i = j + 1
    return ranks


def spearman_by_frame_bootstrap(frame_ids: Sequence[str], groups_by_frame: Dict[str, str],
                                x: Sequence[float], y: Sequence[float], n_boot: int = N_BOOT,
                                seed_label: str = "spearman") -> Tuple[Interval, np.ndarray]:
    """Spearman over pooled (frame, k) records, with the bootstrap resampling FRAMES (all of a
    frame's records move together), because the k-levels within a frame are not independent draws.
    Returns the interval and the replicate distribution, so a second statistic can be compared
    against this one's 95th percentile (§6 rule 2b)."""
    frame_ids = np.asarray(frame_ids)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (frame_ids.shape == x.shape == y.shape):
        raise ValueError("frame ids, x and y must be matched per record")
    frames = np.unique(frame_ids)
    groups = np.asarray([groups_by_frame[f] for f in frames])
    rows_by_frame = {f: np.flatnonzero(frame_ids == f) for f in frames}
    rng = rng_for("spearman-boot", seed_label, len(frames))
    idx = _boot_indices(groups, n_boot, rng)
    reps = np.empty(n_boot, dtype=np.float64)
    for r in range(n_boot):
        rows = np.concatenate([rows_by_frame[frames[i]] for i in idx[r]])
        reps[r] = spearman(x[rows], y[rows])
    point = spearman(x, y)
    return (Interval(point=point, lo=float(np.percentile(reps, 2.5)),
                     hi=float(np.percentile(reps, 97.5)), n=int(x.size), statistic="spearman"),
            reps)


def spearman_within_frame_permutation(frame_ids: Sequence[str], x: Sequence[float],
                                      y: Sequence[float], n_perm: int = 2000,
                                      seed_label: str = "spperm") -> Dict[str, float]:
    """Two-sided p-value for a pooled Spearman when the records are CLUSTERED by frame, under the null
    "the predictor carries no information about the response within a frame": permute x within each
    frame and recompute the pooled statistic.

    Why this exists rather than the bootstrap interval alone: it is EXACT under the clustered null by
    construction, while the frame-level percentile bootstrap of a rank statistic has no such guarantee.
    Both were calibrated on 500 pure-noise trials of this study's shape and came out indistinguishable
    (bootstrap 6.4 % [4.6, 8.9], permutation 5.8 % [4.1, 8.2], nominal 5 %;
    `studies/depthanchor/checks/interval_calibration.py`) — so this is a design choice, not a repair of a
    measured defect. An earlier 100-trial pass suggested the bootstrap sat at 9 %; that was noise and the
    claim is withdrawn in SPEC A10. Rule 2b decides here; the interval reports the effect size.
    """
    fids = np.asarray(frame_ids)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (fids.shape == x.shape == y.shape) or x.size < 12:
        raise ValueError(f"clustered permutation needs >= 12 matched records, got {x.shape}")
    rows_by_frame = [np.flatnonzero(fids == f) for f in np.unique(fids)]
    rng = rng_for("spearman-perm", seed_label, x.size)
    obs = spearman(x, y)
    ge = 0
    for _ in range(int(n_perm)):
        xp = x.copy()
        for rows in rows_by_frame:
            xp[rows] = x[rng.permutation(rows)]
        if abs(spearman(xp, y)) >= abs(obs) - 1e-15:
            ge += 1
    return {"rho": float(obs), "p_two_sided": float((1.0 + ge) / (n_perm + 1.0)),
            "n_records": int(x.size), "n_frames": len(rows_by_frame), "n_perm": int(n_perm)}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float, float]:
    if n <= 0:
        raise ValueError("Wilson interval on zero episodes")
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # At the ends the algebra gives exactly 0 and exactly 1 — for successes == n, centre + half is
    # (1 + z^2/2n) / (1 + z^2/n) + (z^2/2n) / (1 + z^2/n) = 1 — and floating point leaves a 1e-16 gap
    # instead. That gap is harmless as an interval and poison downstream: it makes `rate - hi` a small
    # NEGATIVE number, which is what killed `figures/1` in matplotlib's errorbar.
    if successes >= n:
        hi = 1.0
    if successes <= 0:
        lo = 0.0
    return p, lo, hi


def intervals_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def overlap_fraction(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Overlap as a fraction of the narrower width — the §3 rule for raising n from 10 to 20."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    width = min(a[1] - a[0], b[1] - b[0])
    if width <= 0:
        return 1.0 if inter > 0 else 0.0
    return float(inter / width)


def permutation_pvalue(diff: Sequence[float], n_perm: int = 10000, seed_label: str = "perm") -> float:
    """Two-sided sign-flip p-value for a paired difference (the BH family of §7 rule 2a). Exact
    under the paired null 'the label target/background is exchangeable within a frame'."""
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("permutation test on an empty sample")
    rng = rng_for("perm", seed_label, d.size)
    obs = abs(float(np.mean(d)))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    reps = np.abs((signs * d[None, :]).mean(axis=1))
    return float((1.0 + np.sum(reps >= obs - 1e-15)) / (n_perm + 1.0))


# ------------------------------------------------------------------ §5 gates
@dataclass
class Gate:
    name: str
    verdict: str                     # "pass" | "fail" | CANCELLED
    value: Optional[float]
    floor: Optional[float]
    ceiling: Optional[float]
    detail: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "verdict": self.verdict, "value": self.value,
                "floor": self.floor, "ceiling": self.ceiling, "detail": self.detail}


def gate_g0(max_abs_delta_norm: Optional[float], n_frames: int, tol: float = 1e-4) -> Gate:
    """G0 pass-through: replaying the model's OWN bins through the injection path must reproduce the
    natural (adaptive-off) action. Floor 0 (bit-identical is fine), ceiling `tol`. CANCELLED when no
    frame produced a comparison at all — a gate with no input must not read 'pass'."""
    if max_abs_delta_norm is None or n_frames <= 0 or not np.isfinite(max_abs_delta_norm):
        return Gate("G0", CANCELLED, max_abs_delta_norm, 0.0, tol,
                    {"why": "no frame produced an identity comparison"})
    verdict = "pass" if max_abs_delta_norm <= tol else "fail"
    return Gate("G0", verdict, float(max_abs_delta_norm), 0.0, tol, {"n_frames": int(n_frames)})


def gate_g1(median_rho: Optional[float], shuffle_p95: Optional[float], n_frames: int,
            floor_rho: float = 0.0) -> Gate:
    """G1 depth fidelity: per-frame Spearman between the model's decoded bins and the Depth-Anything
    reference on the same frame, against the same statistic under spatially shuffled bins. Pass
    needs BOTH bounds: above the shuffle 95th percentile (it is depth, not chance) and above
    `floor_rho` (a negative median would mean the sign convention is inverted, which the directional
    clauses would then read backwards)."""
    if median_rho is None or shuffle_p95 is None or n_frames <= 0 \
            or not np.isfinite(median_rho) or not np.isfinite(shuffle_p95):
        return Gate("G1", CANCELLED, median_rho, floor_rho, shuffle_p95,
                    {"why": "fidelity or its shuffle null is missing"})
    verdict = "pass" if (median_rho > shuffle_p95 and median_rho > floor_rho) else "fail"
    return Gate("G1", verdict, float(median_rho), floor_rho, float(shuffle_p95),
                {"n_frames": int(n_frames)})


def gate_g2(n_distinct: Optional[int], noise: Optional[float], required: int = 100) -> Gate:
    """G2 code ladder: at least `required` of the 128 codes must decode to a depth separated from
    their rank neighbours by more than the decoder's own placement noise. Failure pivots the edits
    to within-frame swaps and cancels every 'k ranks' clause."""
    if n_distinct is None or noise is None or not np.isfinite(noise):
        return Gate("G2", CANCELLED, None, float(required), None,
                    {"why": "ladder or its placement noise is missing"})
    verdict = "pass" if int(n_distinct) >= int(required) else "fail"
    return Gate("G2", verdict, float(n_distinct), float(required), float(N_CODES),
                {"placement_noise": float(noise)})


def gate_g3(pixel_px: Optional[float], wrong_pixel_px: Optional[float],
            depth_m: Optional[float], wrong_depth_m: Optional[float], *,
            tol_px: float = 1.0, wrong_min_px: float = 5.0, tol_depth_m: float = 0.15,
            wrong_depth_ratio: float = 5.0, n_conventions_passing: Optional[int] = None) -> Gate:
    """G3 projection, in two halves, each with its registered failing input.

    (a) ARITHMETIC: the study's own `camera_project` must agree with robosuite's own projection of the
        same point to under `tol_px` pixels, while the same check with the extrinsic rotation transposed
        must be off by more than `wrong_min_px`. This is what licenses using the banked camera matrices
        in the offline cells, which cannot call robosuite.
    (b) PHYSICS: the projected depth must agree with the RENDERED depth at that pixel to `tol_depth_m`,
        and the transposed-camera variant must be at least `wrong_depth_ratio` times worse. The
        tolerance is physical, not numerical: the end-effector site sits between the gripper fingers, so
        the surface the ray hits can be centimetres away — which is why the ratio to a wrong camera
        carries the weight here and the absolute number is only a sanity bound.

    `n_conventions_passing` (when given) must be exactly 1: two conventions that both agree would mean
    the check does not distinguish them, and the study would be choosing one by preference.
    """
    vals = (pixel_px, wrong_pixel_px, depth_m, wrong_depth_m)
    if any(v is None for v in vals) or not np.isfinite(pixel_px) or not np.isfinite(depth_m):
        return Gate("G3", CANCELLED, pixel_px, 0.0, tol_px,
                    {"why": "the projection check or its wrong-camera control is missing",
                     "pixel_px": pixel_px, "wrong_pixel_px": wrong_pixel_px,
                     "depth_error_m": depth_m, "wrong_depth_error_m": wrong_depth_m})
    arithmetic = bool(pixel_px < tol_px)
    arithmetic_control = bool(not np.isfinite(wrong_pixel_px) or wrong_pixel_px > wrong_min_px)
    physics = bool(depth_m <= tol_depth_m)
    physics_control = bool(not np.isfinite(wrong_depth_m)
                           or wrong_depth_m >= wrong_depth_ratio * max(depth_m, 1e-9))
    unique = True if n_conventions_passing is None else bool(n_conventions_passing == 1)
    verdict = "pass" if (arithmetic and arithmetic_control and physics and physics_control
                         and unique) else "fail"
    return Gate("G3", verdict, float(pixel_px), 0.0, tol_px,
                {"pixel_px": float(pixel_px),
                 "wrong_pixel_px": None if wrong_pixel_px is None else float(wrong_pixel_px),
                 "depth_error_m": float(depth_m),
                 "wrong_depth_error_m": None if wrong_depth_m is None else float(wrong_depth_m),
                 "tol_depth_m": tol_depth_m, "wrong_min_px": wrong_min_px,
                 "wrong_depth_ratio": wrong_depth_ratio,
                 "arithmetic_ok": arithmetic, "arithmetic_control_fired": arithmetic_control,
                 "physics_ok": physics, "physics_control_fired": physics_control,
                 "n_conventions_passing": n_conventions_passing, "unique_convention": unique})


def gate_room(ratio_interval: Optional[Interval], name: str, floor: float = 1.0,
              healthy: float = 3.0) -> Gate:
    """The room check of §5, with all three branches. `ratio_interval` is the interval on the median
    of ceiling/seed-floor over frames. Above `healthy` the readout has room; between `floor` and
    `healthy` it has reduced power and every effect must be read against that; with the whole
    interval below `floor` the ceiling itself is inside the sampler noise and no edit can be
    measured — which for the depth-matched ceiling is a RESULT (H_epi with its mechanism), not a
    broken instrument."""
    if ratio_interval is None:
        return Gate(name, CANCELLED, None, floor, healthy, {"why": "no ceiling/floor ratio measured"})
    if ratio_interval.lo >= healthy:
        verdict = "pass"
    elif ratio_interval.hi < floor:
        verdict = "fail"
    else:
        verdict = "reduced"
    return Gate(name, verdict, ratio_interval.point, floor, healthy,
                {"ci95": [ratio_interval.lo, ratio_interval.hi], "n_frames": ratio_interval.n})


# ------------------------------------------------------------------ §6 decision rules
@dataclass
class Decision:
    label: str
    rule1: Dict[str, object]
    rule2: Dict[str, object]
    rule4: Dict[str, object]
    gates: Dict[str, object]
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"label": self.label, "rule1_read_at_all": self.rule1,
                "rule2_read_as_map": self.rule2, "rule4_anchor_concentration": self.rule4,
                "gates": self.gates, "notes": self.notes}


def margin_m(seed_over_ceiling: Sequence[float]) -> float:
    """`m` of §6: the 95th percentile over frames of S(f)/C(f), frozen on the control cells."""
    v = np.asarray(seed_over_ceiling, dtype=np.float64)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("m must be frozen on a finite control sample")
    return float(np.percentile(v, 95))


def decide(*, gates: Dict[str, Gate], m: float,
           shuffle_over_ceiling: Interval,
           const_over_ceiling: Optional[Interval] = None,
           ablation_over_ceiling: Optional[Interval] = None,
           paired_target_minus_bg: Dict[int, Interval],
           bh_family: Dict[int, float],
           paired_swap: Optional[Dict[str, Interval]] = None,
           swap_pvalues: Optional[Dict[str, float]] = None,
           spearman_target: Optional[Interval],
           spearman_bg_p95: Optional[float],
           anchor_share: Optional[Interval],
           anchor_count_share: Optional[float],
           spearman_permutation_p: Optional[float] = None,
           require_rule2: bool = True) -> Decision:
    """The frozen rules, reading their gates. No rule may produce a label while G0 is not passing:
    if the injected bins are not the bins the expert saw, every offline number is meaningless (§8)."""
    notes: List[str] = []
    gate_dict = {k: g.as_dict() for k, g in gates.items()}

    g0 = gates.get("G0")
    if g0 is None or g0.verdict != "pass":
        notes.append("G0 did not pass: the offline label is CANCELLED by §5, not decided.")
        return Decision(CANCELLED, {}, {}, {}, gate_dict, notes)

    # rule 1 — is the prefix read at all
    # RULE 1, AMENDED (A11). The spec reads "is the prefix read at all" off the SHUFFLE alone. But a
    # shuffle preserves the multiset of codes, so an expert that reads the prefix as a bag — no position
    # sensitivity — would move by nothing under it while reading the prefix perfectly well, and the
    # spec's rule would return H_epi, which is exactly false. Two other cells change the prefix without
    # any spatial structure: O-const (every cell at the median code) and the depth-gate ablation
    # (O-gate0, the prefix carrying nothing). "Read at all" is therefore ANY of the three clearing the
    # margin; "read POSITIONALLY" is the shuffle clearing it, and only that licenses asking whether the
    # prefix is a map. The spec's shuffle-only reading is reported beside the amended one.
    def _clears(iv: Optional[Interval]) -> Optional[bool]:
        if iv is None:
            return None
        return bool(iv.excludes(m) and iv.point > m)

    positional = bool(_clears(shuffle_over_ceiling))
    whole_prefix = {"shuffle": _clears(shuffle_over_ceiling),
                    "const": _clears(const_over_ceiling),
                    "ablation": _clears(ablation_over_ceiling)}
    read = any(v for v in whole_prefix.values() if v is not None)
    rule1 = {"median_shuffle_over_ceiling": shuffle_over_ceiling.as_dict(), "m": float(m),
             "median_const_over_ceiling": None if const_over_ceiling is None
                                          else const_over_ceiling.as_dict(),
             "median_ablation_over_ceiling": None if ablation_over_ceiling is None
                                             else ablation_over_ceiling.as_dict(),
             "clears_margin": whole_prefix,
             "read": bool(read), "read_positionally": positional,
             "spec_shuffle_only_read": bool(_clears(shuffle_over_ceiling))}
    g4b = gates.get("G4b")
    if read and g4b is not None and g4b.verdict == "fail":
        # Removing the depth prefix entirely moved the action less than the flow sampler's own noise,
        # yet editing the grid moved it more. Not impossible — the model is not linear and an edited
        # grid is out of distribution in a way an absent one is not — but it is a contradiction between
        # two readouts of the same channel, so it is stamped on the decision rather than resolved here.
        notes.append("depth-ceiling contradiction: G4b (the depth ablation ceiling) is inside the "
                     "seed floor while rule 1 reads the prefix as read; the shuffle effect is larger "
                     "than removing the prefix altogether.")
        rule1["depth_ceiling_contradiction"] = True
    if not read:
        notes.append("rule 1: no whole-prefix change (shuffle, constant grid, or removing the prefix "
                     "altogether) clears the seed-floor margin -> H_epi.")
        return Decision("H_epi", rule1, {}, _anchor(anchor_share, anchor_count_share), gate_dict, notes)
    if not positional:
        notes.append("rule 1 amended (A11): the prefix IS read — a whole-prefix change clears the "
                     "margin — but the spatial permutation does not, so it is read without position "
                     "sensitivity. That is H_gate by §1's own definition, and the map clauses of rule 2 "
                     "are computed and reported but cannot promote the label.")
        d = Decision("H_gate", rule1, {}, _anchor(anchor_share, anchor_count_share), gate_dict, notes)
        return d

    if not require_rule2:
        # The gate session decides rule 1 and freezes `m`; the map is not yet measurable and saying so
        # is not the same as a cancelled readout.
        notes.append("rule 1 decided; the map clauses are not measurable until the contrast cells run.")
        return Decision("read; map undecided (contrast not run)", rule1, {},
                        _anchor(anchor_share, anchor_count_share), gate_dict, notes)

    # rule 2 — is it read as a map
    primary = {}
    for k in (PRIMARY_ABS_K, -PRIMARY_ABS_K):
        iv = paired_target_minus_bg.get(k)
        primary[k] = None if iv is None else iv
    # WHICH CONTRAST IS PRIMARY depends on G2, and the choice is pre-registered (§8): with a usable code
    # ladder it is the ±8-rank shift; without one, "k ranks" is undefined and the pivot — each aimed cell
    # taking the code of the frame's own farthest or nearest cell — takes its place.
    g2 = gates.get("G2")
    ladder_usable = bool(g2 is not None and g2.verdict == "pass")
    have_primary = all(v is not None for v in primary.values())
    rejected_ks = set()
    if bh_family:
        ks_all = sorted(bh_family)
        mask_all = bh_reject([bh_family[k] for k in ks_all], 0.05)
        rejected_ks = {k for k, r in zip(ks_all, mask_all) if r}
    if not ladder_usable:
        notes.append(f"rule 2a: G2 is {None if g2 is None else g2.verdict}, so 'k ranks' is undefined "
                     "and the §8 pivot decides — the paired target-minus-background contrast under the "
                     "within-frame extreme swap. The k-ladder contrasts are computed and reported but "
                     "are CANCELLED as clauses.")
        if not paired_swap or any(paired_swap.get(w) is None for w in ("far", "near")):
            notes.append("rule 2a: the pivot contrast is missing too -> spatial clause CANCELLED.")
            spatial = None
        else:
            spatial = all(paired_swap[w].lo > 0.0 for w in ("far", "near"))
            if swap_pvalues:
                spatial = spatial and all(float(swap_pvalues.get(w, 1.0)) < 0.05
                                          for w in ("far", "near"))
    elif not have_primary:
        notes.append("rule 2a: a |k|=8 contrast is missing -> spatial clause CANCELLED.")
        spatial = None
    else:
        # BOTH the interval and the paired permutation test: the percentile bootstrap of a median on
        # right-skewed Δ data covered the truth in 187 of 200 null trials (93.5 %) rather than 95, the
        # permutation test is exact under the paired null, and BH controls the six-level family of §7.
        spatial = (all(iv.lo > 0.0 for iv in primary.values())
                   and all(k in rejected_ks for k in primary))
    rule2 = {"paired_target_minus_bg": {str(k): (None if v is None else v.as_dict())
                                        for k, v in sorted(paired_target_minus_bg.items())},
             "primary_abs_k": PRIMARY_ABS_K,
             "primary_contrast": "k_ladder" if ladder_usable else "within_frame_extreme_swap",
             "ladder_usable_G2": ladder_usable,
             "paired_swap": None if not paired_swap else
                            {w: (None if v is None else v.as_dict()) for w, v in paired_swap.items()},
             "swap_pvalues": None if not swap_pvalues else
                             {w: float(v) for w, v in swap_pvalues.items()},
             "spatial": None if spatial is None else bool(spatial),
             "bh_family_pvalues": {str(k): float(p) for k, p in sorted(bh_family.items())},
             "bh_rejected_k": sorted(rejected_ks)}

    g1 = gates.get("G1")
    directional_measurable = (spearman_target is not None and spearman_bg_p95 is not None
                              and g1 is not None and g1.verdict == "pass")
    if spearman_target is None or spearman_bg_p95 is None:
        rule2["directional"] = None
        notes.append("rule 2b: Δ_z or its background null is not measurable -> directional clause CANCELLED.")
    elif g1 is not None and g1.verdict != "pass":
        rule2["directional"] = None
        rule2["spearman_target"] = spearman_target.as_dict()
        notes.append(f"rule 2b: G1 is {g1.verdict}; decoded depth is not established as depth, so the "
                     "directional clause is CANCELLED (§5 G1).")
    else:
        # The calibrated test decides; the interval reports the size. `spearman_permutation_p` is the
        # within-frame permutation p-value when the caller measured it (it always does in this study);
        # without it the interval is used and the fallback is recorded.
        p = None if spearman_permutation_p is None else float(spearman_permutation_p)
        if p is None:
            notes.append("rule 2b: no clustered permutation p-value supplied; the bootstrap interval "
                         "decides, which is not exact under the clustered null.")
            passes = spearman_target.excludes(0.0)
        else:
            passes = p < 0.05
        directional = bool(passes and abs(spearman_target.point) > abs(spearman_bg_p95))
        rule2["directional"] = bool(directional)
        rule2["spearman_target"] = spearman_target.as_dict()
        rule2["spearman_permutation_p"] = p
        rule2["spearman_bg_p95"] = float(spearman_bg_p95)
        # §6 rule 2b is written on the MAGNITUDE against the background null, and the background null is
        # itself a magnitude (the 95th percentile of |rho| over bg replicates), so a reversed sign can
        # satisfy the clause. Reading the prefix backwards is still reading it as a map, so the label
        # stands; but it contradicts the registered predictor ("farther" should command motion farther),
        # so the sign is stamped on the decision instead of being averaged into a magnitude (A16).
        rule2["directional_sign_as_predicted"] = bool(spearman_target.point > 0)
        if directional and spearman_target.point < 0:
            notes.append("rule 2b: the directional clause clears the background null on MAGNITUDE but "
                         f"its sign is reversed (rho = {spearman_target.point:+.3f}): asking the model "
                         "to see the target cells as FARTHER moved the commanded translation NEARER. "
                         "The label stands as written, and this sign is a finding, not a pass.")

    if spatial and rule2.get("directional") is True:
        label = "H_geo"
    elif spatial:
        label = "spatial, not directional"
    elif spatial is None:
        label = CANCELLED
        notes.append("the spatial clause could not be evaluated -> no label.")
    else:
        label = "H_gate"
    if label == "H_geo" and not directional_measurable:
        raise AssertionError("H_geo cannot be assigned while the directional clause is not measurable")
    return Decision(label, rule1, rule2, _anchor(anchor_share, anchor_count_share), gate_dict, notes)


def _anchor(anchor_share: Optional[Interval], count_share: Optional[float]) -> Dict[str, object]:
    if anchor_share is None or count_share is None:
        return {"measured": False}
    return {"measured": True, "share_of_importance_in_T": anchor_share.as_dict(),
            "share_expected_from_cell_count": float(count_share),
            "excess": float(anchor_share.point - count_share)}


def closed_loop_confirmation(normal: Tuple[float, float, float],
                             arms: Dict[str, Tuple[float, float, float]],
                             label: str) -> Dict[str, object]:
    """§6 rule 3. `normal` and each arm are (rate, lo, hi) Wilson triples on the same suite set.
    Confirmation of rule 1 needs C-shuffle AND C-const below C-normal with non-overlapping
    intervals; of rule 2b, C-far or C-near differing from C-normal with non-overlapping intervals."""
    out: Dict[str, object] = {"normal": list(normal), "arms": {}}
    for name, tri in arms.items():
        out["arms"][name] = {"rate": tri[0], "ci95": [tri[1], tri[2]],
                             "overlaps_normal": intervals_overlap((tri[1], tri[2]), (normal[1], normal[2])),
                             "below_normal": bool(tri[0] < normal[0]),
                             "overlap_fraction": overlap_fraction((tri[1], tri[2]), (normal[1], normal[2]))}
    def _sep_below(name):
        a = out["arms"].get(name)
        return bool(a and a["below_normal"] and not a["overlaps_normal"])
    def _sep(name):
        a = out["arms"].get(name)
        return bool(a and not a["overlaps_normal"])
    out["confirms_rule1"] = _sep_below("C-shuffle") and _sep_below("C-const")
    # §6 rule 3, read literally: "C-far AND C-near differing from C-normal IN THE PREDICTED DIRECTION".
    # On a success rate the predicted direction for both shift arms is below C-normal — a geometric map
    # told the wrong distance makes the arm reach short or long either way — so a shift arm that beats
    # C-normal is not confirmation of anything. The looser reading the code carried before (either arm,
    # either direction) is reported beside it, because it is the one that could be mistaken for the rule.
    out["confirms_rule2b"] = _sep_below("C-far") and _sep_below("C-near")
    out["either_shift_arm_separated_either_direction"] = _sep("C-far") or _sep("C-near")
    if label == "H_epi":
        out["verdict"] = ("confirmed" if not (_sep("C-shuffle") or _sep("C-const"))
                          else "offline only — the loop separates arms the offline cells did not")
    elif label in ("H_geo", "spatial, not directional", "H_gate"):
        need = out["confirms_rule1"] and (out["confirms_rule2b"] or label != "H_geo")
        out["verdict"] = "confirmed" if need else "offline only"
    else:
        out["verdict"] = CANCELLED
    return out


# ------------------------------------------------------------------ self-test (the offline gate)
def _selftest() -> None:
    print("[core] units")
    a = np.zeros((CHUNK, ACTION_DIM))
    b = a.copy()
    b[:, 0] = 0.5
    b[:, GRIP_DIM] = 2.0
    assert abs(delta_norm_cont(a, b) - 0.5 / 6) < 1e-12, delta_norm_cont(a, b)
    assert abs(delta_robot_all(a, b) - (0.5 + 2.0) / 7) < 1e-12
    ar = np.zeros((CHUNK, ACTION_DIM)); ar[:, GRIP_DIM] = -1.0
    br = ar.copy(); br[:5, GRIP_DIM] = 1.0
    assert abs(grip_flip(ar, br) - 0.5) < 1e-12
    d = np.zeros((CHUNK, ACTION_DIM)); d[:, 2] = 0.1
    assert abs(delta_z(a, d, np.array([0.0, 0.0, 1.0])) - 0.1) < 1e-12
    assert abs(delta_z(a, d, np.array([0.0, 0.0, -1.0])) + 0.1) < 1e-12
    for bad in (np.zeros((CHUNK, ACTION_DIM + 1)), np.full((CHUNK, ACTION_DIM), np.nan)):
        try:
            delta_norm_cont(a, bad); raise AssertionError("bad chunk accepted")
        except ValueError:
            pass

    print("[core] code ladder")
    rng = rng_for("selftest", "ladder")
    d_primary = np.sort(rng.normal(size=N_CODES))
    ladder = CodeLadder(d_primary=d_primary, d_local=d_primary + 1e-4 * rng.normal(size=N_CODES))
    assert ladder.rank[ladder.order[0]] == 0
    assert ladder.shift(int(ladder.order[0]), -50) == int(ladder.order[0]), "clamp at the low end"
    assert ladder.shift(int(ladder.order[-1]), +50) == int(ladder.order[-1]), "clamp at the high end"
    mid = int(ladder.order[N_CODES // 2])
    assert ladder.median_code == mid
    assert ladder.rank[ladder.shift(mid, +8)] == ladder.rank[mid] + 8

    print("[core] edits")
    bins = rng.integers(0, N_CODES, size=N_CELLS)
    e = edit_identity(bins)
    assert e.n_changed == 0
    sh = edit_shuffle(bins, rng_for("selftest", "shuffle"), ladder)
    assert sh.n_changed > 0 and sorted(sh.bins.tolist()) == sorted(np.asarray(bins).tolist())
    uni = edit_shuffle(np.full(N_CELLS, 5), rng_for("selftest", "shuffle-uniform"), ladder)
    assert uni.meta.get("degenerate_uniform_grid") is True and uni.n_changed == 0
    co = edit_const(bins, ladder)
    assert len(set(co.bins.tolist())) == 1
    cells = [0, 1, 11, 55]
    up = edit_shift_cells(bins, cells, +8, ladder, "target")
    assert up.realized_rank_shift > 0 and up.realized_depth_shift > 0
    dn = edit_shift_cells(bins, cells, -8, ladder, "target")
    assert dn.realized_rank_shift < 0 and dn.realized_depth_shift < 0
    assert np.all(up.bins[[c for c in range(N_CELLS) if c not in cells]]
                  == np.asarray(bins)[[c for c in range(N_CELLS) if c not in cells]])
    rc = edit_random_cells(bins, cells, rng_for("selftest", "rand"), ladder)
    assert rc.n_changed == len(cells), "every aimed cell must actually change"
    sc = edit_single_cell(bins, 42, +8, ladder)
    assert sc.n_changed <= 1
    far = edit_swap_extreme(bins, cells, "far", ladder)
    near = edit_swap_extreme(bins, cells, "near", ladder)
    assert len(set(far.bins[list(cells)].tolist())) == 1, "every aimed cell takes the same code"
    assert far.meta["code"] in set(np.asarray(bins).tolist()), "the pivot uses a code from this frame"
    assert far.meta["code_depth"] <= near.meta["code_depth"], "far must decode shallower than near"
    try:
        edit_swap_extreme(bins, cells, "sideways", ladder); raise AssertionError("bad direction")
    except ValueError:
        pass

    print("[core] projection")
    K = np.array([[100.0, 0, 64.0], [0, 100.0, 64.0], [0, 0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1.0])
    uv, z = camera_project(K, R, t, np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    assert abs(z[0] - 1.0) < 1e-12 and abs(uv[0, 0] - 64.0) < 1e-9
    assert uv[1, 0] > uv[0, 0]
    uv_cv, z_cv = camera_project(K, R, t, np.array([[0.0, 0.0, 0.0]]), opengl=False)
    assert abs(z_cv[0] + 1.0) < 1e-12, "the OpenCV convention must give the opposite depth sign here"
    assert abs(uv_cv[0, 1] - 64.0) < 1e-9
    cells_hit = cells_from_pixels(np.array([[5.0, 5.0], [120.0, 120.0], [-1.0, 5.0]]), (128, 128))
    assert cells_hit.tolist() == [0, 99], cells_hit
    flipped = cells_from_pixels(np.array([[5.0, 5.0], [120.0, 120.0]]), (128, 128), v_flip=True)
    assert flipped.tolist() == [90, 9], flipped        # the mirror maps the top row to the bottom
    hflipped = cells_from_pixels(np.array([[5.0, 5.0], [120.0, 120.0]]), (128, 128), u_flip=True)
    assert hflipped.tolist() == [9, 90], hflipped      # and the left column to the right
    ax_cv = axis_into_scene(np.eye(3), opengl=False)
    ax_gl = axis_into_scene(np.eye(3), opengl=True)
    assert np.allclose(ax_cv, [0, 0, 1]) and np.allclose(ax_gl, [0, 0, -1])
    dil = dilate_cells([0])
    assert sorted(dil.tolist()) == [0, 1, 10, 11]
    bg, used = background_cells([0, 1, 10, 11], 4, rng_for("selftest", "bg"))
    assert used == 2 and bg.size == 4 and int(_CELL_DIST[np.ix_(bg, [0, 1, 10, 11])].min()) >= 2
    big = list(range(N_CELLS - 4))
    try:
        background_cells(big, 90, rng_for("selftest", "bg2")); raise AssertionError("bg drew from nothing")
    except ValueError:
        pass

    print("[core] statistics")
    groups = ["a"] * 30 + ["b"] * 30
    base = np.concatenate([rng.normal(1.0, 0.1, 30), rng.normal(1.0, 0.1, 30)])
    iv = bootstrap_stat(base, groups, "median")
    assert iv.lo < 1.0 < iv.hi and iv.n == 60
    planted = base + 0.5
    pd = bootstrap_paired_difference(planted, base, groups, "median")
    assert pd.lo > 0.0 and abs(pd.point - 0.5) < 1e-9, pd
    nul = bootstrap_paired_difference(base, base, groups, "median")
    assert nul.lo == 0.0 == nul.hi
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert abs(spearman([1, 1, 2, 2], [1, 1, 2, 2]) - 1.0) < 1e-12
    fids = np.array([f"f{i // 6}" for i in range(60)])
    gbf = {f"f{i}": ("a" if i < 5 else "b") for i in range(10)}
    ks = np.tile(np.array(K_LEVELS, dtype=float), 10)
    ys = 0.01 * ks + rng.normal(0, 0.001, 60)
    sp, reps = spearman_by_frame_bootstrap(fids, gbf, ks, ys)
    assert sp.lo > 0 and reps.size == N_BOOT
    spn, repsn = spearman_by_frame_bootstrap(fids, gbf, ks, rng.normal(0, 1, 60))
    assert spn.lo < 0 < spn.hi or not spn.excludes(0.0)
    perm_planted = spearman_within_frame_permutation(fids, ks, 0.02 * ks + rng.normal(0, 0.02, 60),
                                                     n_perm=200, seed_label="planted")
    assert perm_planted["p_two_sided"] < 0.02, perm_planted
    perm_flat = spearman_within_frame_permutation(fids, ks, np.zeros(60), n_perm=200,
                                                  seed_label="flat")
    assert perm_flat["p_two_sided"] > 0.5, perm_flat
    try:
        spearman_within_frame_permutation(fids[:6], ks[:6], ks[:6]); raise AssertionError("too few")
    except ValueError:
        pass
    p, lo, hi = wilson(8, 10)
    assert lo < 0.8 < hi and 0 <= lo and hi <= 1
    try:
        wilson(0, 0); raise AssertionError("Wilson on zero episodes accepted")
    except ValueError:
        pass
    assert intervals_overlap((0.0, 0.5), (0.4, 1.0)) and not intervals_overlap((0.0, 0.3), (0.4, 1.0))
    assert abs(overlap_fraction((0.0, 0.4), (0.2, 1.0)) - 0.5) < 1e-12
    assert permutation_pvalue(np.full(30, 0.5)) < 0.01
    assert permutation_pvalue(rng.normal(0, 1, 200)) > 0.01

    print("[core] gates — pass, fail and CANCELLED for each")
    assert gate_g0(1e-7, 120).verdict == "pass"
    assert gate_g0(1e-2, 120).verdict == "fail"
    assert gate_g0(None, 120).verdict == CANCELLED and gate_g0(1e-7, 0).verdict == CANCELLED
    assert gate_g1(0.7, 0.2, 120).verdict == "pass"
    assert gate_g1(0.1, 0.2, 120).verdict == "fail"
    assert gate_g1(-0.7, -0.9, 120).verdict == "fail", "an inverted sign must not pass G1"
    assert gate_g1(None, 0.2, 120).verdict == CANCELLED
    assert gate_g2(110, 0.01).verdict == "pass" and gate_g2(40, 0.01).verdict == "fail"
    assert gate_g2(None, None).verdict == CANCELLED
    assert gate_g3(0.3, 40.0, 0.02, 1.4, n_conventions_passing=1).verdict == "pass"
    assert gate_g3(4.0, 40.0, 0.02, 1.4, n_conventions_passing=1).verdict == "fail", "bad arithmetic"
    assert gate_g3(0.3, 0.4, 0.02, 1.4, n_conventions_passing=1).verdict == "fail", \
        "a gate whose wrong-camera pixel control passes is not a gate"
    assert gate_g3(0.3, 40.0, 0.9, 1.4, n_conventions_passing=1).verdict == "fail", "bad physics"
    assert gate_g3(0.3, 40.0, 0.02, 0.03, n_conventions_passing=1).verdict == "fail", \
        "a gate whose wrong-camera depth control passes is not a gate"
    assert gate_g3(0.3, 40.0, 0.02, 1.4, n_conventions_passing=2).verdict == "fail", \
        "two conventions agreeing means the check does not distinguish them"
    assert gate_g3(0.3, np.inf, 0.02, np.inf, n_conventions_passing=1).verdict == "pass", \
        "an out-of-frame wrong camera is a fired control, not a missing measurement"
    assert gate_g3(None, 40.0, 0.02, 1.4).verdict == CANCELLED
    room_hi = gate_room(Interval(9.0, 5.0, 14.0, 120, "median"), "G4a")
    room_mid = gate_room(Interval(2.0, 1.5, 2.5, 120, "median"), "G4b")
    room_lo = gate_room(Interval(0.4, 0.2, 0.7, 120, "median"), "G4b")
    assert (room_hi.verdict, room_mid.verdict, room_lo.verdict) == ("pass", "reduced", "fail")
    assert gate_room(None, "G4b").verdict == CANCELLED

    print("[core] decision rules — every label reachable")
    gg = {"G0": gate_g0(1e-7, 120), "G1": gate_g1(0.7, 0.2, 120), "G2": gate_g2(110, 0.01),
          "G3": gate_g3(0.3, 41.0, 0.02, 1.4, n_conventions_passing=1), "G4a": room_hi,
          "G4b": room_hi}
    up_iv = Interval(0.05, 0.02, 0.08, 120, "median")
    zero_iv = Interval(0.0, -0.02, 0.02, 120, "median")
    sp_up = Interval(0.4, 0.2, 0.6, 720, "spearman")
    sp_zero = Interval(0.0, -0.2, 0.2, 720, "spearman")
    fam = {k: 0.001 for k in K_LEVELS}
    d_geo = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                   paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                   spearman_target=sp_up, spearman_bg_p95=0.1, spearman_permutation_p=0.001,
                   anchor_share=Interval(0.5, 0.4, 0.6, 120, "median"), anchor_count_share=0.16)
    assert d_geo.label == "H_geo", d_geo.label
    # a directional Spearman whose CALIBRATED p-value does not clear 0.05 must not read as directional
    d_uncal = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                     paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                     spearman_target=sp_up, spearman_bg_p95=0.1, spearman_permutation_p=0.4,
                     anchor_share=None, anchor_count_share=None)
    assert d_uncal.label == "spatial, not directional", d_uncal.label
    # a spatial interval that clears 0 but whose BH-corrected p-value does not must not read as spatial
    d_nobh = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                    paired_target_minus_bg={k: up_iv for k in K_LEVELS},
                    bh_family={k: 0.4 for k in K_LEVELS}, spearman_target=sp_up,
                    spearman_bg_p95=0.1, spearman_permutation_p=0.001, anchor_share=None,
                    anchor_count_share=None)
    assert d_nobh.label == "H_gate", d_nobh.label
    d_sp = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                  paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                  spearman_target=sp_zero, spearman_bg_p95=0.1, anchor_share=None,
                  anchor_count_share=None)
    assert d_sp.label == "spatial, not directional"
    # A16: rule 2b is written on the magnitude against a magnitude null, so a REVERSED sign clears it.
    # The label stands, and the reversal is stamped on the decision rather than hidden in an absolute
    # value — a passing directional clause whose sign contradicts the registered predictor.
    d_rev = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                   paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                   spearman_target=Interval(-0.4, -0.6, -0.2, 720, "spearman"), spearman_bg_p95=0.1,
                   spearman_permutation_p=0.001, anchor_share=None, anchor_count_share=None)
    assert d_rev.label == "H_geo", d_rev.label
    assert d_rev.rule2["directional_sign_as_predicted"] is False
    assert any("sign is reversed" in n for n in d_rev.notes), d_rev.notes
    assert d_geo.rule2["directional_sign_as_predicted"] is True
    d_gate = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                    paired_target_minus_bg={k: zero_iv for k in K_LEVELS}, bh_family=fam,
                    spearman_target=sp_zero, spearman_bg_p95=0.1, anchor_share=None,
                    anchor_count_share=None)
    assert d_gate.label == "H_gate"
    # read as a bag: the shuffle stays inside the margin but a constant grid clears it -> H_gate, and
    # NOT H_epi, which is what the spec's shuffle-only rule 1 would have returned
    d_bag = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.05, 0.02, 0.1, 120, "median"),
                   const_over_ceiling=Interval(0.8, 0.7, 0.9, 120, "median"),
                   paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                   spearman_target=sp_up, spearman_bg_p95=0.1, spearman_permutation_p=0.001,
                   anchor_share=None, anchor_count_share=None)
    assert d_bag.label == "H_gate", d_bag.label
    assert d_bag.rule1["read"] and not d_bag.rule1["read_positionally"]
    d_epi = decide(gates=gg, m=0.7, shuffle_over_ceiling=Interval(0.1, 0.05, 0.2, 120, "median"),
                   const_over_ceiling=Interval(0.1, 0.05, 0.2, 120, "median"),
                   ablation_over_ceiling=Interval(0.12, 0.06, 0.2, 120, "median"),
                   paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                   spearman_target=sp_up, spearman_bg_p95=0.1, anchor_share=None,
                   anchor_count_share=None)
    assert d_epi.label == "H_epi"
    gg_bad = dict(gg); gg_bad["G0"] = gate_g0(1e-2, 120)
    assert decide(gates=gg_bad, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                  paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                  spearman_target=sp_up, spearman_bg_p95=0.1, anchor_share=None,
                  anchor_count_share=None).label == CANCELLED
    gg_g1f = dict(gg); gg_g1f["G1"] = gate_g1(0.1, 0.2, 120)
    d_g1f = decide(gates=gg_g1f, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                   paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                   spearman_target=sp_up, spearman_bg_p95=0.1, anchor_share=None,
                   anchor_count_share=None)
    assert d_g1f.label == "spatial, not directional" and d_g1f.rule2["directional"] is None
    # G2 failing hands the spatial clause to the pivot, and the k-ladder can no longer decide it
    gg_g2f = dict(gg); gg_g2f["G2"] = gate_g2(40, 0.01)
    d_pivot = decide(gates=gg_g2f, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                     paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                     paired_swap={"far": up_iv, "near": up_iv},
                     swap_pvalues={"far": 0.001, "near": 0.002},
                     spearman_target=sp_up, spearman_bg_p95=0.1, spearman_permutation_p=0.001,
                     anchor_share=None, anchor_count_share=None)
    assert d_pivot.label == "H_geo" and d_pivot.rule2["primary_contrast"] == \
        "within_frame_extreme_swap", d_pivot.rule2["primary_contrast"]
    d_pivot_flat = decide(gates=gg_g2f, m=0.2,
                          shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                          paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                          paired_swap={"far": zero_iv, "near": zero_iv},
                          swap_pvalues={"far": 0.6, "near": 0.7},
                          spearman_target=sp_up, spearman_bg_p95=0.1, spearman_permutation_p=0.001,
                          anchor_share=None, anchor_count_share=None)
    assert d_pivot_flat.label == "H_gate", d_pivot_flat.label
    d_pivot_missing = decide(gates=gg_g2f, m=0.2,
                             shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                             paired_target_minus_bg={k: up_iv for k in K_LEVELS}, bh_family=fam,
                             spearman_target=sp_up, spearman_bg_p95=0.1,
                             spearman_permutation_p=0.001, anchor_share=None, anchor_count_share=None)
    assert d_pivot_missing.label == CANCELLED, d_pivot_missing.label
    d_miss = decide(gates=gg, m=0.2, shuffle_over_ceiling=Interval(0.6, 0.5, 0.7, 120, "median"),
                    paired_target_minus_bg={4: up_iv}, bh_family={4: 0.01},
                    spearman_target=sp_up, spearman_bg_p95=0.1, anchor_share=None,
                    anchor_count_share=None)
    assert d_miss.label == CANCELLED

    print("[core] closed-loop confirmation")
    normal = wilson(38, 40)
    conf = closed_loop_confirmation(normal, {"C-shuffle": wilson(10, 40), "C-const": wilson(8, 40),
                                             "C-far": wilson(20, 40), "C-near": wilson(18, 40)}, "H_geo")
    assert conf["confirms_rule1"] and conf["confirms_rule2b"], conf
    assert conf["verdict"] == "confirmed", conf
    # ONE shift arm separated and the other not is not confirmation: §6 rule 3 reads both arms
    half = closed_loop_confirmation(normal, {"C-shuffle": wilson(10, 40), "C-const": wilson(8, 40),
                                             "C-far": wilson(20, 40), "C-near": wilson(35, 40)}, "H_geo")
    assert half["confirms_rule1"] and not half["confirms_rule2b"], half
    assert half["either_shift_arm_separated_either_direction"], half
    assert half["verdict"] == "offline only", half
    # a shift arm that BEATS C-normal with a separated interval is not the predicted direction either
    up = closed_loop_confirmation(wilson(20, 40), {"C-shuffle": wilson(4, 40), "C-const": wilson(3, 40),
                                                   "C-far": wilson(38, 40), "C-near": wilson(37, 40)},
                                  "H_geo")
    assert up["confirms_rule1"] and not up["confirms_rule2b"], up
    assert up["either_shift_arm_separated_either_direction"], up
    flat = closed_loop_confirmation(normal, {"C-shuffle": wilson(37, 40), "C-const": wilson(38, 40)},
                                    "H_epi")
    assert flat["verdict"] == "confirmed"
    assert abs(margin_m([0.1, 0.2, 0.3, 0.4]) - float(np.percentile([0.1, 0.2, 0.3, 0.4], 95))) < 1e-12
    # the ends are exact, so an error bar built from them is never negative
    for n_ in (1, 5, 10, 40, 137):
        assert wilson(n_, n_)[2] == 1.0, wilson(n_, n_)
        assert wilson(0, n_)[1] == 0.0, wilson(0, n_)
        assert wilson(n_, n_)[2] - wilson(n_, n_)[0] >= 0.0
        assert wilson(0, n_)[0] - wilson(0, n_)[1] >= 0.0
    print("[core] SELFTEST OK")


if __name__ == "__main__":
    _selftest()
