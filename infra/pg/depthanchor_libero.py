"""DEPTHANCHOR simulator layer — the LIBERO environment, the camera bank, and the rollout loop.

The public LeRobot port does not support the Think (depth-reasoning) policy, so the closed loop is
ours: per step render, build the observation the checkpoint expects, call `predict_action`, execute
`n_action_steps`, record. Everything the loop needs to be trusted is measured here rather than
assumed, because each of these conventions has exactly one right answer and no error message:

  * THREE VERTICAL/SIGN CONVENTIONS, none of which may be assumed, and all of which are measured:
    (1) `opengl` — does the banked camera matrix want the camera's own z axis pointing into the scene or
        out of it? Decided by `probe_orientation` against robosuite's OWN projection of the same point.
    (2) `depth_flip` — which row of the rendered depth buffer does a projected point index? Decided by
        the same probe against the rendered depth.
    (3) `render_flip` — is the image the MODEL is given the rendered buffer or its vertical mirror?
        Decided by the dataset, which is what the checkpoint was trained on. This one sets which row of
        the image a depth cell covers, so getting it wrong mirrors every target cell while every number
        still looks plausible.
    G3 gates (1) and (2), each with a transposed-camera control that must fail.
  * STATE LAYOUT. `[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]` — checked against the
    dataset's own `observation.state` on a frame rendered from the same init state (the preflight's
    `_dataset_orientation`, which also decides which of the two vertical conventions the dataset uses),
    not against a docstring.
  * GROUND-TRUTH DEPTH. robosuite's own `get_real_depth_map` converts the depth buffer to metres, so
    the model's decoded depth can be compared with metric truth (G1B) instead of only with
    Depth-Anything, which fixes the sign of "nearer" before any directional clause is read.
  * SETTLE STEPS AND HORIZON. LIBERO objects fall for the first few frames; every published harness
    steps a no-op for ~10 frames first. Both numbers are recorded with every rollout.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from infra.pg import depthanchor_core as core

APT_PACKAGES = ("libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 "
                "libxext6 libxrender1 libgomp1")
HF_LIBERO_PIN = "hf-libero==0.1.4"
IMAGE_SIZE = 256          # the dataset's own frame size (meta/info.json shape 256x256x3)
SETTLE_STEPS = 10         # published LIBERO harnesses step a no-op while objects fall
HORIZON = {"libero_spatial": 220, "libero_object": 220, "libero_goal": 220, "libero_10": 520}
NOOP = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def libero_package_root() -> pathlib.Path:
    """The installed `libero` package directory, found WITHOUT importing it — importing
    `libero.libero` triggers both an interactive prompt and the asset download."""
    import importlib.util
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hf-libero is not installed")
    return pathlib.Path(list(spec.submodule_search_locations)[0])


ASSET_PROBE = ("libero", "assets", "scenes", "libero_tabletop_base_style.xml")
ASSETS_REPO = "lerobot/libero-assets"          # a DATASET repo whose root IS the assets tree
S3_ASSETS = "s3://pg-lineage-769063704696/depthanchor/libero_assets/v1"


def ensure_libero_assets() -> Dict[str, object]:
    """`hf-libero` ships NO scene assets. LIBERO fetches them from `lerobot/libero-assets` the first
    time something builds an arena — and `run_job.sh` exports `HF_HUB_OFFLINE=1` for the weight cache,
    so that fetch degrades to a warning and the failure surfaces two hundred lines later as a missing
    `libero_tabletop_base_style.xml` (measured, preflight attempt 1). Importing `libero.libero` is not
    enough to trigger it either (attempt 3): the import only writes the config file.

    So the assets are fetched EXPLICITLY, and from S3 when a previous job already put them there —
    this runs in every session, and a hub round-trip for a few hundred megabytes of meshes costs more
    than the science does. Whichever path is taken, the file the arena needs is CHECKED afterwards;
    offline mode is restored so model loading keeps reading the S3-restored weight cache.
    """
    root = libero_package_root()
    assets = root / "libero" / "assets"
    probe = root.joinpath(*ASSET_PROBE)
    if probe.is_file():
        return {"source": "already_present", "probe": str(probe)}

    report: Dict[str, object] = {}
    have_mirror = subprocess.run(["aws", "s3", "ls", f"{S3_ASSETS}/DONE", "--region", "us-east-1"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if have_mirror:
        assets.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", f"{S3_ASSETS}/tree", str(assets),
                        "--region", "us-east-1"], check=True)
        report["source"] = "s3_mirror"
    if not probe.is_file():
        from infra.pg.diff_harvest import hf_online_setup
        was_offline = os.environ.get("HF_HUB_OFFLINE")
        hf_online_setup()          # fetches the token from SSM and sets HF_HUB_OFFLINE=0 in os.environ
        # THE DOWNLOAD RUNS IN A CHILD PROCESS. `huggingface_hub` reads HF_HUB_OFFLINE into a module
        # constant at IMPORT time, so flipping the environment variable in a process that has already
        # imported it changes nothing — attempt 4 died on exactly that, with LocalEntryNotFoundError
        # while the variable said "0". A fresh interpreter imports the constant as online.
        code = ("import sys\n"
                "from huggingface_hub import snapshot_download\n"
                "print(snapshot_download(sys.argv[1], repo_type='dataset', local_dir=sys.argv[2]))\n")
        proc = subprocess.run([sys.executable, "-c", code, ASSETS_REPO, str(assets)],
                              capture_output=True, text=True, env=os.environ.copy())
        if was_offline is not None:
            os.environ["HF_HUB_OFFLINE"] = was_offline
        report["source"] = "hub"
        report["snapshot"] = proc.stdout.strip()[-300:]
        if proc.returncode != 0:
            raise RuntimeError(f"downloading {ASSETS_REPO} failed (rc={proc.returncode}): "
                               f"{proc.stderr[-1500:]}")
        if probe.is_file():
            # publish the mirror for every later session, DONE written last
            subprocess.run(["aws", "s3", "sync", str(assets), f"{S3_ASSETS}/tree",
                            "--exclude", ".cache/*", "--region", "us-east-1"], check=True)
            subprocess.run(["bash", "-lc",
                            f"date -u > /tmp/libero_assets_done && aws s3 cp "
                            f"/tmp/libero_assets_done {S3_ASSETS}/DONE --region us-east-1"], check=True)
            report["mirror_published"] = True
    # LIBERO asks an interactive question on its first import and there is no stdin in a job; answer it
    # once now so nothing later hits an EOFError.
    subprocess.run('echo "N" | %s -c "import libero.libero"' % sys.executable, shell=True, check=False)
    if not probe.is_file():
        raise RuntimeError(f"LIBERO's scene assets are still missing: {probe} does not exist "
                           f"(report so far: {report})")
    report["probe"] = str(probe)
    report["asset_files"] = sum(1 for x in assets.rglob("*") if x.is_file())
    return report


def install(extra_numpy: Optional[str] = None) -> Dict[str, object]:
    """Install the simulator into the job's venv and PROVE the pieces are importable.

    `extra_numpy` is the numpy pin, and it is a decision the caller makes from a measurement, not a
    default: `hf-libero` pins `robosuite==1.4.0`, which predates numpy 2, but the AMI carries numpy 2
    for torch, and downgrading numpy for no reason risks the torch and pyarrow builds that were made
    against it. The caller runs a smoke test first and only passes a pin if it fails.
    The returned dict goes into `metrics.json` so a later reader sees the exact environment."""
    # `-o DPkg::Lock::Timeout` because the DLAMI runs unattended-upgrades at boot and holds the dpkg
    # lock for the first few minutes. Measured: with a 22 GB weight sync ahead of it the lock is free by
    # the time apt runs, and with `--no-weights` it is not — attempt 2 of the preflight died on
    # "Could not get lock /var/lib/dpkg/lock-frontend ... held by process (unattended-upgr)". Waiting is
    # correct; retrying blind would race the same lock.
    apt = f"apt-get -o DPkg::Lock::Timeout=900 update -y && DEBIAN_FRONTEND=noninteractive " \
          f"apt-get -o DPkg::Lock::Timeout=900 install -y {APT_PACKAGES}"
    subprocess.run(["bash", "-lc", apt], check=True)
    cmd = [sys.executable, "-m", "pip", "install", "-q", HF_LIBERO_PIN]
    if extra_numpy:
        cmd.append(extra_numpy)
    subprocess.run(cmd, check=True)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    assets = ensure_libero_assets()
    import numpy as _np
    import torch as _torch
    bridge = _torch.from_numpy(_np.arange(3, dtype=_np.float32)).sum().item()
    if abs(bridge - 3.0) > 1e-6:
        raise RuntimeError("torch<->numpy bridge broke after the numpy pin")
    import pyarrow
    import robosuite
    import mujoco
    # `pip install hf-libero` can in principle resolve a different torch (robomimic depends on it
    # unpinned). A replaced CPU-only or mismatched-CUDA wheel would not break any import — it would
    # quietly move the science to the CPU — so the CUDA build is part of the banked report and the
    # caller's `after_install` check raises when CUDA is gone.
    return {"numpy": _np.__version__, "torch": _torch.__version__,
            "torch_cuda_build": _torch.version.cuda,
            "torch_cuda_available": bool(_torch.cuda.is_available()),
            "pyarrow": pyarrow.__version__,
            "robosuite": robosuite.__version__, "mujoco": mujoco.__version__,
            "hf_libero_pin": HF_LIBERO_PIN, "numpy_pin": extra_numpy,
            "torch_numpy_bridge_ok": True, "assets": assets}


# ------------------------------------------------------------------ benchmark and camera bank
def _benchmark(suite: str):
    from libero.libero import benchmark as libero_benchmark
    return libero_benchmark.get_benchmark_dict()[suite]()


def _bddl_path(suite: str, task) -> str:
    from libero.libero import get_libero_path
    return os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def make_env(suite: str, task_id: int, *, camera_depths: bool = True,
             image_size: int = IMAGE_SIZE):
    """LIBERO's own off-screen environment, at the dataset's frame size, with the depth buffer on."""
    from libero.libero.envs import OffScreenRenderEnv
    bench = _benchmark(suite)
    task = bench.get_task(task_id)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": _bddl_path(suite, task),
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_depths": bool(camera_depths),
    })
    return env, bench, task


def camera_bank(suites: Sequence[str] = core.SUITES, image_size: int = IMAGE_SIZE
                ) -> Dict[str, object]:
    """One entry per (suite, task): the agentview intrinsics and extrinsics from robosuite's own
    helpers, the viewing direction in the world frame, the language string, and the number of init
    states. Read once, banked, and used by the offline cells for the target-cell projection."""
    from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix,
                                              get_camera_intrinsic_matrix)
    out: Dict[str, object] = {"image_size": int(image_size), "tasks": {}, "by_language": {}}
    for suite in suites:
        for task_id in range(len(_benchmark(suite).tasks)):
            env, bench, task = make_env(suite, task_id, camera_depths=True, image_size=image_size)
            try:
                env.reset()
                sim = env.env.sim
                K = get_camera_intrinsic_matrix(sim, "agentview", image_size, image_size)
                E = get_camera_extrinsic_matrix(sim, "agentview")     # camera-in-world, 4x4
                R_cw = np.asarray(E)[:3, :3]
                t = np.asarray(E)[:3, 3]
                # MuJoCo cameras look down their own -z; the world-frame viewing direction is the
                # third column of the camera-in-world rotation, negated.
                view_dir = -R_cw[:, 2]
                key = f"{suite}/{task_id}"
                entry = {
                    "suite": suite, "task_id": int(task_id), "language": str(task.language),
                    "bddl": os.path.basename(_bddl_path(suite, task)),
                    "K": np.asarray(K)[:3, :3].tolist(),
                    "R_world_to_cam": R_cw.T.tolist(),      # camera_project() wants world->camera
                    "cam_pos_world": t.tolist(),
                    "view_dir_world": (view_dir / np.linalg.norm(view_dir)).tolist(),
                    # Provenance only: NOTHING measures with these. The depth buffer is converted to
                    # metres by robosuite's own `get_real_depth_map`, so a binding that does not expose
                    # the near/far planes under these names must not take the camera bank down with it.
                    **_near_far(sim),
                    "n_init_states": int(len(bench.get_task_init_states(task_id))),
                }
                out["tasks"][key] = entry
                out["by_language"][_normalize_language(task.language)] = key
            finally:
                env.close()
    if len(out["tasks"]) != 40:
        raise RuntimeError(f"expected 40 LIBERO tasks across {list(suites)}, banked {len(out['tasks'])}")
    return out


def _near_far(sim) -> Dict[str, object]:
    try:
        return {"znear": float(sim.model.vis.map.znear * sim.model.stat.extent),
                "zfar": float(sim.model.vis.map.zfar * sim.model.stat.extent)}
    except Exception as exc:      # recorded, and unused by any measurement (see the caller's comment)
        return {"znear": None, "zfar": None, "near_far_error": f"{type(exc).__name__}: {exc}"}


def _normalize_language(text: str) -> str:
    return " ".join(str(text).replace("_", " ").lower().split()).strip(" .")


# ------------------------------------------------------------------ observations
def obs_to_frame(obs: Dict[str, np.ndarray], *, flip: bool, task: str,
                 hflip: bool = False) -> Dict[str, object]:
    """The observation the checkpoint expects: agentview first, wrist second, and the 8-d state
    `[eef_pos, quat2axisangle(eef_quat), gripper_qpos]`.

    `flip` and `hflip` are the transform that makes a RENDERED image look like a DATASET image, which is
    what the checkpoint was trained on. Both are measured, never assumed, and the five-episode known
    number is what confirms the pair: a policy given a mirrored world does not reach for anything."""
    from robosuite.utils.transform_utils import quat2axisangle
    agent = np.asarray(obs["agentview_image"], dtype=np.uint8)
    wrist = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    if flip:
        agent = agent[::-1].copy()
        wrist = wrist[::-1].copy()
    if hflip:
        agent = agent[:, ::-1].copy()
        wrist = wrist[:, ::-1].copy()
    state = np.concatenate([np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                            np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float64),
                            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)])
    if state.shape != (8,):
        raise RuntimeError(f"LIBERO state must be 8-d, built {state.shape}")
    return {"agentview": agent, "wrist": wrist, "task": str(task),
            "state": state.astype(np.float32)}


def real_depth(env, obs: Dict[str, np.ndarray], *, flip: bool) -> np.ndarray:
    """The agentview depth buffer in metres, via robosuite's own conversion."""
    from robosuite.utils.camera_utils import get_real_depth_map
    d = np.asarray(obs["agentview_depth"], dtype=np.float64)
    if d.ndim == 3:
        d = d[..., 0]
    real = np.asarray(get_real_depth_map(env.env.sim, d[..., None]), dtype=np.float64)[..., 0]
    return real[::-1].copy() if flip else real


def model_space_depth(env, obs: Dict[str, np.ndarray], *, flip: bool, hflip: bool) -> np.ndarray:
    """The metric depth under the SAME transform `obs_to_frame` applies to the RGB. Any per-cell
    comparison between a depth grid and the model's own code grid has to be in this orientation; the
    rendered buffer's own orientation is a projection-space question and a different one."""
    real = real_depth(env, obs, flip=flip)
    return real[:, ::-1].copy() if hflip else real


# ------------------------------------------------------------------ G3: orientation and projection
DEPTH_INDEXINGS = (("raw", False, False), ("vflip", False, True), ("hflip", True, False),
                   ("rot180", True, True))

PROJECTION_VARIANTS = (
    # (name, use the transposed - i.e. WRONG - rotation, OpenGL sign convention)
    ("opengl", False, True),
    ("opencv", False, False),
    ("wrong_camera", True, True),
    ("wrong_camera_opencv", True, False),
)


def probe_orientation(suite: str, task_id: int, cam: Dict[str, object], *, n_frames: int = 20,
                      image_size: int = IMAGE_SIZE) -> Dict[str, object]:
    """G3. Two questions, both measured, neither assumed.

    (1) WHICH PIXEL CONVENTION does the banked camera matrix want? `core.camera_project` can read the
        camera's own z axis as pointing into the scene (OpenCV) or out of it (OpenGL), and only one is
        right for these extrinsics. The reference is robosuite's OWN projection
        (`project_points_from_world_to_camera` through `get_camera_transform_matrix`), so this is a
        comparison against the library that rendered the image rather than against a docstring. The
        offline cells cannot call robosuite — they only have the banked matrices — which is why
        `camera_project` exists and why this check is what licenses it.
    (2) WHICH ROW of the rendered buffers does a projected point index? Measured by reading the rendered
        DEPTH at the projected pixel under both indexings and comparing with the projected depth. This
        one has a physical tolerance rather than a numerical one: the end-effector site sits between the
        gripper fingers, so the surface the ray actually hits can be a few centimetres away.

    Both come with a registered failing input: the same checks with the extrinsic rotation transposed,
    which must be far worse on both counts or the gate is not a gate.
    """
    from robosuite.utils.camera_utils import (get_camera_transform_matrix,
                                              project_points_from_world_to_camera)
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R_world_to_cam"], dtype=np.float64)
    t = np.asarray(cam["cam_pos_world"], dtype=np.float64)
    env, bench, task = make_env(suite, task_id, camera_depths=True, image_size=image_size)
    px: Dict[str, List[float]] = {n: [] for n, *_ in PROJECTION_VARIANTS}
    dz: Dict[str, Dict[str, List[float]]] = {n: {k: [] for k, *_ in DEPTH_INDEXINGS}
                                             for n, *_ in PROJECTION_VARIANTS}
    ref_rows: List[List[float]] = []
    try:
        init_states = bench.get_task_init_states(task_id)
        env.reset()
        env.set_init_state(init_states[0])
        T = get_camera_transform_matrix(env.env.sim, "agentview", image_size, image_size)
        for step in range(n_frames + SETTLE_STEPS):
            obs, _, _, _ = env.step(NOOP)
            if step < SETTLE_STEPS:
                continue
            eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(1, 3)
            ref = np.asarray(project_points_from_world_to_camera(eef, T, image_size,
                                                                image_size)).reshape(-1)
            ref_rows.append([float(ref[0]), float(ref[1])])
            depth = real_depth(env, obs, flip=False)
            for name, wrong, opengl in PROJECTION_VARIANTS:
                uv, z = core.camera_project(K, R.T if wrong else R, t, eef, opengl=opengl)
                u0, v0 = float(uv[0, 0]), float(uv[0, 1])
                # robosuite returns (row, column); ours is (u = column, v = row)
                px[name].append(float(np.hypot(v0 - ref[0], u0 - ref[1])))
                # all four dihedral indexings of the rendered buffer, not just the vertical pair: the
                # dataset turned out to be HORIZONTALLY mirrored relative to the projection, so the same
                # question has to be asked of the simulator's buffers or the two cannot be compared
                for key, uf, vf in DEPTH_INDEXINGS:
                    u = (image_size - 1.0 - u0) if uf else u0
                    v = (image_size - 1.0 - v0) if vf else v0
                    if not (0 <= u < image_size and 0 <= v < image_size) or not np.isfinite(z[0]) \
                            or z[0] <= 0:
                        dz[name][key].append(np.inf)
                    else:
                        dz[name][key].append(abs(float(depth[int(v), int(u)]) - float(z[0])))
    finally:
        env.close()

    def _med(values):
        v = [x for x in values]
        if not v:
            return None
        return float(np.median(v)) if np.all(np.isfinite(v)) else float("inf")

    return {"n_frames": len(ref_rows),
            "robosuite_reference_rowcol_first": ref_rows[:3],
            "pixel_error_px": {n: _med(px[n]) for n in px},
            "depth_error_m": {n: {k: _med(dz[n][k]) for k, *_ in DEPTH_INDEXINGS} for n in dz}}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"cannot correlate images of different size: {x.size} vs {y.size}")
    x = x - x.mean(); y = y - y.mean()
    den = float(np.sqrt((x @ x) * (y @ y)))
    return 0.0 if den < 1e-12 else float((x @ y) / den)


# ------------------------------------------------------------------ runtime target cells
def controller_output_scale(env) -> np.ndarray:
    """The metres-per-unit of the OSC controller's translation output, read from the controller itself
    rather than assumed, so integrating a predicted action chunk into future positions is exact."""
    ctrl = env.env.robots[0].controller
    out_max = np.asarray(getattr(ctrl, "output_max", None), dtype=np.float64)
    if out_max.size < 3:
        raise RuntimeError(f"the OSC controller exposes no 3-d output_max: {out_max}")
    return out_max[:3]


def runtime_target_cells(env, obs, chunk_robot: np.ndarray, cam: Dict[str, object], *,
                         v_flip: bool, u_flip: bool, opengl: bool,
                         image_size: int = IMAGE_SIZE) -> np.ndarray:
    """The closed-loop counterpart of the offline target cells, and matched to it by construction: the
    offline definition is "the cells the end-effector occupies in the next 10 recorded steps", and at
    runtime the model's OWN predicted chunk is that trajectory. Integrate its ten translation deltas
    from the current end-effector position with the controller's output scale, project with the banked
    camera, dilate by one cell.

    `opengl` is the pixel-sign convention gate G3 chose; `v_flip` is whether the image the MODEL is
    given is the vertical mirror of the rendered buffer, which is what decides the row a cell covers.
    Both come from the banked orientation artifact, never from a default.
    """
    scale = controller_output_scale(env)
    eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    deltas = np.asarray(chunk_robot, dtype=np.float64)[:, 0:3] * scale[None, :]
    future = eef[None, :] + np.cumsum(deltas, axis=0)
    uv, _ = core.camera_project(np.asarray(cam["K"]), np.asarray(cam["R_world_to_cam"]),
                               np.asarray(cam["cam_pos_world"]), future, opengl=opengl)
    hit = core.cells_from_pixels(uv, (image_size, image_size), v_flip=v_flip, u_flip=u_flip)
    if hit.size == 0:
        # The predicted path leaves the frame: fall back to the current end-effector cell, and the
        # caller records that it happened rather than silently editing a different set of cells.
        uv0, _ = core.camera_project(np.asarray(cam["K"]), np.asarray(cam["R_world_to_cam"]),
                                    np.asarray(cam["cam_pos_world"]), eef[None, :], opengl=opengl)
        hit = core.cells_from_pixels(uv0, (image_size, image_size), v_flip=v_flip,
                                     u_flip=u_flip)
    if hit.size == 0:
        raise core.NoTargetCells("neither the predicted path nor the end-effector projects into the "
                                 "image, so this state has no cells to aim at")
    return core.dilate_cells(hit, radius=1)


# ------------------------------------------------------------------ the rollout loop
@dataclass
class Rollout:
    suite: str
    task_id: int
    episode: int
    arm: str
    success: bool
    n_steps: int
    seconds: float
    eef_path: List[List[float]] = field(default_factory=list)
    bins_per_call: List[List[int]] = field(default_factory=list)
    tail_per_call: List[int] = field(default_factory=list)
    gate_per_call: List[List[float]] = field(default_factory=list)
    frames: List[np.ndarray] = field(default_factory=list)     # kept only when filming
    depth_maps: List[np.ndarray] = field(default_factory=list)  # kept only when filming
    meta: Dict[str, object] = field(default_factory=dict)

    def as_record(self) -> Dict[str, object]:
        return {"suite": self.suite, "task_id": self.task_id, "episode": self.episode,
                "arm": self.arm, "success": bool(self.success), "n_steps": self.n_steps,
                "seconds": self.seconds, "eef_path": self.eef_path,
                "bins_per_call": self.bins_per_call, "tail_per_call": self.tail_per_call,
                "gate_per_call": self.gate_per_call, "meta": dict(self.meta)}


def check_success(env) -> Optional[bool]:
    """LIBERO exposes success in two places depending on the wrapper generation. Probe both and say
    which answered; a rollout records BOTH this and the `done` flag, and a disagreement between them
    is banked rather than resolved silently."""
    for name in ("check_success",):
        fn = getattr(env, name, None)
        if callable(fn):
            return bool(fn())
    inner = getattr(env, "env", None)
    fn = getattr(inner, "_check_success", None)
    if callable(fn):
        return bool(fn())
    return None


class ArmPolicy:
    not_completed = 0
    last_error = None

    """One closed-loop depth regime. `act` gets the resident actor and this step's frame and returns
    the ActResult whose action chunk the environment will execute. Each arm states, in its own words,
    how many model calls per step it costs — the closed-loop budget is dominated by the arms that need
    the model's own grid before they can edit it."""
    name = "abstract"
    calls_per_step = 1

    def reset(self, ctx: Dict[str, object]) -> None:
        pass

    def act(self, actor, frame, step: int, ctx: Dict[str, object]):
        raise NotImplementedError


class ArmNormal(ArmPolicy):
    """C-normal: the released deployment path — `enable_adaptive_depth=True` with the caller-owned
    cache threaded from step to step, so unchanged RGB patches reuse cached codes."""
    name = "C-normal"

    def reset(self, ctx):
        self._cache = None

    def act(self, actor, frame, step, ctx):
        res, self._cache = actor.adaptive(frame, self._cache)
        return res


class ArmFresh(ArmPolicy):
    """C-fresh: full regeneration every step (`enable_adaptive_depth=False`) — the reference, and the
    cost of adaptivity."""
    name = "C-fresh"

    def act(self, actor, frame, step, ctx):
        return actor.natural(frame)


class ArmStale(ArmPolicy):
    """C-stale: the grid from step 0, replayed for the whole episode."""
    name = "C-stale"

    def reset(self, ctx):
        self._bins = None
        self.short_grid = 0

    def act(self, actor, frame, step, ctx):
        if self._bins is None:
            res = actor.natural(frame)
            if res.bins.shape == (core.N_CELLS,):
                self._bins = res.bins.copy()
            else:
                # the checkpoint closed its own depth block early, so there is no full grid to hold on
                # to; this step runs the model's own grid and the next one tries again, counted either way
                self.short_grid += 1
            return res
        return actor.inject(frame, self._bins)


class ArmEdit(ArmPolicy):
    """C-shuffle / C-const / C-far / C-near: the model's own grid for this step, edited.

    `edit(natural, frame, step, ctx) -> bins` gets the whole natural ActResult (so an arm can use the
    model's own predicted action chunk, which is what defines the runtime target cells) and the rollout
    context (env, obs). Two model calls per step for the arms that need the model's grid first;
    `needs_model_bins=False` arms (C-const) cost one."""
    calls_per_step = 2

    def __init__(self, name: str, edit, needs_model_bins: bool = True):
        self.name = name
        self._edit = edit
        self.needs_model_bins = bool(needs_model_bins)
        self.calls_per_step = 2 if needs_model_bins else 1
        self.not_completed = 0
        self.short_grid = 0
        self.no_target_cells = 0
        self.last_error = None

    def act(self, actor, frame, step, ctx):
        natural = actor.natural(frame) if self.needs_model_bins else None
        if natural is not None and natural.bins.shape != (core.N_CELLS,):
            # THE MODEL'S OWN GRID IS SHORT. `_generate_depth_prefix` stops when the model emits
            # `<depth_end>`, and at some states it emits it before the grid is full — 99 codes, in
            # `results/depthanchor/loop/goal/1` at rollout 54. There is then no 100-cell grid for this
            # arm to edit, which is a fact about the checkpoint at that state and not about the edit, so
            # it is counted SEPARATELY from a refused edit and the step runs the model's own grid.
            self.short_grid += 1
            self.last_error = (f"the model emitted {int(natural.bins.size)} depth codes, not "
                               f"{core.N_CELLS}; no edit is defined on a short grid")
            return natural
        try:
            return actor.inject(frame, self._edit(natural, frame, step, ctx))
        except core.NoTargetCells as exc:
            # The arm's cells are not defined at this state (the long-horizon suites walk the gripper out
            # of the camera's view). Counted, and the step runs the model's own grid.
            self.no_target_cells += 1
            self.last_error = str(exc)[:200]
            return natural if natural is not None else actor.natural(frame)
        except core.EditNotCompleted as exc:
            # The model will not close its own depth block after this edit. An episode cannot pause, so
            # the step falls back to the model's own grid and the fallback is COUNTED — an arm whose
            # rollouts are half fallbacks is not that arm, and the count is what says so.
            self.not_completed += 1
            self.last_error = str(exc)[:200]
            return natural if natural is not None else actor.natural(frame)


def rollout(actor, suite: str, task_id: int, episode: int, policy: ArmPolicy, *,
            flip: bool, hflip: bool = False, image_size: int = IMAGE_SIZE,
            max_steps: Optional[int] = None, film: bool = False, seed: int = 0,
            want_depth: bool = False) -> Rollout:
    """One episode under one depth regime. The action chunk is executed in full (`n_action_steps` = 10
    for `norm_tag='libero'`) before the next prediction, which is the checkpoint's own metadata."""
    env, bench, task = make_env(suite, task_id, camera_depths=want_depth or film,
                                image_size=image_size)
    steps_cap = int(max_steps if max_steps is not None else HORIZON[suite])
    t0 = time.perf_counter()
    out = Rollout(suite=suite, task_id=task_id, episode=episode, arm=policy.name, success=False,
                  n_steps=0, seconds=0.0)
    done_flag = False
    try:
        init_states = bench.get_task_init_states(task_id)
        env.reset()
        obs = env.set_init_state(init_states[episode % len(init_states)])
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(NOOP)
        ctx = {"env": env, "task": task, "episode": int(episode), "suite": suite,
               "task_id": int(task_id), "obs": obs}
        policy.reset(ctx)
        step = 0
        while step < steps_cap and not done_flag:
            frame = obs_to_frame(obs, flip=flip, hflip=hflip, task=task.language)
            ctx["obs"] = obs
            ctx["step"] = step
            res = policy.act(actor, frame, step, ctx)
            out.bins_per_call.append(res.bins.astype(int).tolist())
            out.tail_per_call.append(int(res.n_tail_tokens))
            if res.gate is not None:
                out.gate_per_call.append([float(x) for x in res.gate])
            if film:
                out.frames.append(frame["agentview"].copy())
                if want_depth:
                    out.depth_maps.append(model_space_depth(env, obs, flip=flip, hflip=hflip))
            for a in res.a_robot:
                obs, _, done, _ = env.step(np.asarray(a, dtype=np.float64))
                out.eef_path.append([float(x) for x in np.asarray(obs["robot0_eef_pos"])])
                step += 1
                if bool(done):
                    done_flag = True
                if done_flag or step >= steps_cap:
                    break
        out.n_steps = step
        checked = check_success(env)
        # `done` from robosuite is also raised at the horizon, so it is NOT a success signal on its
        # own; the task's own `_check_success` decides when it is available, and both are recorded so
        # a disagreement is visible instead of averaged away.
        out.success = bool(checked) if checked is not None else bool(done_flag)
        out.meta = {"done_flag": bool(done_flag), "check_success": checked,
                    "edits_not_completed": int(getattr(policy, "not_completed", 0)),
                    "short_natural_grids": int(getattr(policy, "short_grid", 0)),
                    "steps_without_target_cells": int(getattr(policy, "no_target_cells", 0)),
                    "n_short_grid_calls": int(sum(1 for b in out.bins_per_call
                                                  if len(b) != core.N_CELLS)),
                    "edit_error": getattr(policy, "last_error", None),
                    "agree": None if checked is None else bool(done_flag == checked),
                    "steps_cap": steps_cap, "settle_steps": SETTLE_STEPS,
                    "render_flip": bool(flip), "render_hflip": bool(hflip),
                    "calls": len(out.bins_per_call)}
    finally:
        env.close()
    out.seconds = time.perf_counter() - t0
    return out
