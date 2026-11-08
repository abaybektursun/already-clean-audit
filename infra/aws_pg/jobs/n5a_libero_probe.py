"""§5.3 N5a — AWS LIBERO feasibility probe (genuine attempt to port the Modal-only robot eval).

The N5a function arm needs a LIBERO sim-rollout eval; it exists only as a Modal app
(`modal_libero.py`) and Modal credits are exhausted. Before declaring it blocked, ATTEMPT the AWS
port. The decisive risk is headless **EGL rendering + the LIBERO/MuJoCo env on the pg AMI** — if that
works, the rest (lerobot π0 policy + rollout) is software; if it fails, the blocker is *demonstrated*
with a real error, not asserted.

Staged + fail-loud: (1) apt/pip install the LIBERO+MuJoCo+robosuite stack (the proven modal_libero
recipe); (2) MUJOCO_GL=egl; (3) create a libero_spatial env, reset, step, RENDER → check the EGL image
is non-degenerate. Reports how far the AWS stack gets.

  pg run infra.aws_pg.jobs.n5a_libero_probe --type g6e.8xlarge --no-weights --budget-min 60
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys


def _run(cmd, **kw):
    print(f"[N5A-PROBE] $ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"stages": {}}

    # ---- Stage 1: system EGL libs (AMI may already have them; try, don't hard-fail on apt) ----
    try:
        _run(["bash", "-lc",
              "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
              "libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 "
              "libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 git "
              "python3-dev python3.12-dev build-essential"])   # Python.h for evdev/robosuite C-ext builds
        rep["stages"]["apt"] = "ok"
    except Exception as e:
        rep["stages"]["apt"] = f"FAILED (continuing — AMI may already have GL): {e}"
        print(f"[N5A-PROBE] apt failed: {e}", flush=True)

    # ---- Stage 2: pip the LIBERO/MuJoCo/robosuite stack (modal_libero recipe) ----
    pip = [sys.executable, "-m", "pip", "install", "-q",
           "numpy<2", "mujoco>=3.1.6", "robosuite==1.4.1", "gym", "gymnasium",
           "bddl", "easydict", "hydra-core", "opencv-python", "imageio", "scipy",
           "termcolor", "future", "pyyaml"]
    _run(pip)
    rep["stages"]["pip"] = "ok"
    if not pathlib.Path("/opt/libero").exists():
        _run(["git", "clone", "--depth", "1",
              "https://github.com/Lifelong-Robot-Learning/LIBERO.git", "/opt/libero"])
    _run(["bash", "-lc", "cd /opt/libero && %s -m pip install --no-deps -e ." % sys.executable])
    sys.path.insert(0, "/opt/libero")                  # robust: import libero regardless of egg-link
    os.environ["MUJOCO_GL"] = "egl"                    # set BEFORE first mujoco/libero import
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # LIBERO writes an interactive config on first import — pre-answer 'N'
    subprocess.run('echo "N" | %s -c "import sys; sys.path.insert(0,\'/opt/libero\'); import libero.libero"'
                   % sys.executable, shell=True)
    rep["stages"]["libero_install"] = "ok"

    # ---- Stage 3: EGL render of a real LIBERO env ----
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = "libero_spatial"
    bm = benchmark.get_benchmark_dict()[suite]()
    task = bm.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[N5A-PROBE] suite={suite} task0={task.name} bddl={bddl}", flush=True)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
    env.seed(0)
    obs = env.reset()
    img = np.asarray(obs["agentview_image"])
    rnd = env.action_space.sample() if hasattr(env, "action_space") else np.zeros(7)
    obs2, r, done, info = env.step(rnd)
    img2 = np.asarray(obs2["agentview_image"])
    nondegen = bool(img.std() > 1.0 and img2.std() > 1.0)
    env.close()
    rep["stages"]["egl_render"] = {
        "ok": nondegen, "img_shape": list(img.shape),
        "img_std_reset": float(img.std()), "img_std_step": float(img2.std()),
        "note": "non-degenerate EGL render iff std>1 (a black/constant frame = EGL fallback failure)"}
    print(f"[N5A-PROBE] EGL render: shape={img.shape} std_reset={img.std():.2f} "
          f"std_step={img2.std():.2f} nondegen={nondegen}", flush=True)

    rep["FEASIBLE"] = bool(nondegen)
    rep["verdict"] = ("AWS LIBERO/EGL FEASIBLE — env renders; π0 policy integration is the remaining software"
                      if nondegen else
                      "AWS LIBERO/EGL render degenerate (EGL fallback) — the sim eval does not run on this AMI")
    (out / "n5a_libero_probe.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[N5A-PROBE] FEASIBLE={rep['FEASIBLE']} → {rep['verdict']}", flush=True)


if __name__ == "__main__":
    main()
