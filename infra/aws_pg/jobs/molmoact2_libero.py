"""MENDELEEV N15 — LIBERO success rate for MolmoAct2 (row 3c-Act's O) through the released LeRobot
checkpoint `allenai/MolmoAct2-LIBERO-LeRobot`, run by lerobot's OWN evaluator (`lerobot-eval`), exactly
as this repo's π0.5 harness did (`n5a_pi0_libero_eval.py`: apt EGL libs + pip lerobot extras + the
console script). Spec B5 / §3.1 3c-Act / §7 item 22 / §8b "MolmoAct2 LeRobot checkpoint bind" / §10 N15.

RUNTIME (determined 2026-09-03, read-only; nothing here was executed):
  * checkpoint card f0c5a956 (`library_name: lerobot`, policy `lerobot.policies.molmoact2`):
      pip install "lerobot[molmoact2,libero] @ git+https://github.com/huggingface/lerobot.git"
  * lerobot v0.6.1 pyproject (tag 7e241bd6; PyPI 0.6.1 exists; `policies/molmoact2/` present; LiberoEnv
    has `task_ids` + `camera_name_mapping`): requires-python >= 3.12; torch >= 2.7, < 2.12 (the box pins
    torch 2.12.0 → lerobot MUST live in its own venv, never in /opt/pg/venv); transformers >= 5.4, < 5.6
    (lerobot vendors its own MolmoAct2 HF model under `molmoact2_hf_model`, so the hub's remote code and
    its 4.57-style masking calls are NOT used here); peft; scipy; gymnasium >= 1.1.1;
    libero extra → `hf-libero` 0.1.4 (PyPI, 3.2 MB) → robosuite == 1.4.0, robomimic == 0.2.0,
    bddl == 1.0.1, mujoco >= 3.0, < 3.9, hf-egl-probe, opencv-python. Import name stays `libero.libero`.
  * headless rendering: apt libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6
    libxext6 libxrender1 libgomp1 + MUJOCO_GL=egl PYOPENGL_PLATFORM=egl (the n5a recipe); LIBERO asks
    an interactive question on first import (answered 'N' once, as n5a did).
  * downloads (HF, online): the LeRobot checkpoint (model.safetensors 10.9 GB bf16, prefix
    `model.model.*`), PLUS `allenai/MolmoAct2-LIBERO` (21.8 GB F32 — lerobot's `_resolve_checkpoint_location`
    snapshot_downloads the whole `checkpoint_path` repo for config/processor/tokenizer and loads its
    weights before the LeRobot state dict overrides them), PLUS `allenai/MolmoAct2-FAST-Tokenizer`
    (discrete tokenizer, `action_mode: both`). ≈ 33 GB → root disk 200 GB is fine; budget ≥ 15 min for it.
  * GPU: bf16 policy ≈ 11 GB + inference CUDA graph → g6e.2xlarge (L40S 48 GB) is enough.

PROTOCOL (frozen): 4 suites × 20 episodes = 10 tasks × 2 episodes per suite (spec B5 "4 suites × 20
episodes"); `--seed 1000` and the card's `--policy.per_episode_seed=true --policy.eval_seed=1000`;
continuous inference (`inference_action_mode=continuous`, the card's reported setting); camera mapping
`agentview_image→image, robot0_eye_in_hand_image→wrist_image` (the card's command, the checkpoint's
`image_keys`). One `lerobot-eval` per suite; its `eval_info.json` (schema: `per_task[i].metrics.successes`
per episode, `per_group`, `overall`; lerobot_eval.eval_policy_all) is parsed into per-episode records and
banked into `out/metrics.json` after EVERY suite (temp file + os.replace), so a kill mid-run keeps the
finished suites. `--smoke` = 1 episode of libero_spatial task 0 — the bind gate (spec §8b "MolmoAct2
LeRobot checkpoint bind fails → O = not measured, stated").

PUBLISHED (card f0c5a956, 50 episodes/task, continuous, per-episode seeding): Spatial 98.4, Object 100.0,
Goal 98.0, LIBERO-10 96.6, average 98.25. Our 20 episodes/suite give a ± ≈ 10-point Wilson CI per suite
(spec §7 item 22) — this cell is a bind check and an O for the row, not a re-measurement.

NEVER fakes a rate: a success is lerobot's own per-episode `success` flag; a missing/short eval_info.json
raises. Fail-loud, no try/except. `python -m infra.aws_pg.jobs.molmoact2_libero` runs `_dryrun()`
(parses a synthetic eval_info.json, aggregates, banks, re-reads) — no lerobot, no torch, no network.

  pg run infra.aws_pg.jobs.molmoact2_libero --type g6e.2xlarge,g6e.4xlarge --no-weights \\
     --s3-prefix results/mdl/S4/3c-act-O/1 --budget-min 180 --args="--smoke"     # bind gate first
  pg run infra.aws_pg.jobs.molmoact2_libero --type g6e.2xlarge,g6e.4xlarge --no-weights \\
     --s3-prefix results/mdl/S4/3c-act-O/2 --budget-min 180
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys

POLICY = "allenai/MolmoAct2-LIBERO-LeRobot"
LEROBOT_PIN = "lerobot[molmoact2,libero] @ git+https://github.com/huggingface/lerobot.git@v0.6.1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
EPISODES_PER_SUITE = 20
SEED = 1000
CAMERA_MAP = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
PUBLISHED_CARD = {"libero_spatial": 98.4, "libero_object": 100.0, "libero_goal": 98.0, "libero_10": 96.6,
                  "average": 98.25, "source": f"{POLICY} card f0c5a956, 50 episodes/task, continuous inference"}
VENV = "/work/lerobot-venv"
APT = ("libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 libxext6 libxrender1 "
       "libgomp1 git python3-dev python3-venv build-essential")


def _run(cmd, **kw):
    print(f"[N15] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _install(venv: str) -> str:
    """Own venv (lerobot's torch < 2.12 must not touch the box's torch 2.12.0). Returns the venv python."""
    _run(["bash", "-lc", f"apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y {APT}"])
    py = f"{venv}/bin/python"
    if not os.path.exists(py):
        _run([sys.executable, "-m", "venv", venv])
    _run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    _run([py, "-m", "pip", "install", "-q", LEROBOT_PIN])
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    # LIBERO writes an interactive config on FIRST import ("custom dataset path? (Y/N)"); no stdin under
    # lerobot-eval → answer 'N' once now (n5a recipe).
    _run(["bash", "-lc", f'echo "N" | {py} -c "import libero.libero"'])
    return py


def _env_versions(py: str) -> dict:
    code = ("import torch, transformers, lerobot, mujoco, robosuite; "
            "print(torch.__version__, transformers.__version__, getattr(lerobot, '__version__', '?'), "
            "mujoco.__version__, robosuite.__version__)")
    out = subprocess.run([py, "-c", code], check=True, capture_output=True, text=True).stdout.split()
    return dict(zip(("torch", "transformers", "lerobot", "mujoco", "robosuite"), out))


def eval_cmd(py: str, out_dir: str, suite: str, task_ids: list, n_episodes: int, seed: int) -> list:
    """The card's `lerobot-eval` invocation, restricted to (suite, task_ids, n_episodes per task)."""
    binp = pathlib.Path(py).parent / "lerobot-eval"
    base = [str(binp)] if binp.exists() else [py, "-m", "lerobot.scripts.lerobot_eval"]
    return base + [
        f"--output_dir={out_dir}",
        f"--policy.path={POLICY}",
        "--policy.inference_action_mode=continuous",
        "--policy.model_dtype=bfloat16",
        "--policy.use_amp=true",
        "--policy.enable_inference_cuda_graph=true",
        "--policy.device=cuda",
        "--policy.per_episode_seed=true",
        f"--policy.eval_seed={seed}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids={json.dumps(task_ids, separators=(',', ':'))}",
        f"--env.camera_name_mapping={json.dumps(CAMERA_MAP, separators=(',', ':'))}",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=1",
        f"--eval.n_episodes={n_episodes}",
        f"--seed={seed}",
    ]


def parse_eval_info(path: pathlib.Path, suite: str, expect_tasks: list, expect_per_task: int) -> list:
    """Per-episode records from lerobot's eval_info.json; asserts the requested tasks/episodes are all there."""
    assert path.exists(), f"lerobot-eval wrote no eval_info.json at {path}"
    info = json.loads(path.read_text())
    eps = []
    seen = {}
    for t in info["per_task"]:
        assert t["task_group"] == suite, (t["task_group"], suite)
        m = t["metrics"]
        succ = m["successes"]
        assert len(succ) == expect_per_task, f"{suite} task {t['task_id']}: {len(succ)} episodes != {expect_per_task}"
        seen[int(t["task_id"])] = len(succ)
        for k, s in enumerate(succ):
            eps.append({"suite": suite, "task_id": int(t["task_id"]), "episode_ix": k, "success": int(bool(s)),
                        "sum_reward": float(m["sum_rewards"][k]), "max_reward": float(m["max_rewards"][k])})
    assert sorted(seen) == sorted(int(i) for i in expect_tasks), (sorted(seen), expect_tasks)
    return eps


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def aggregate(episodes: list) -> dict:
    out = {"suites": {}, "overall": None}
    for s in SUITES:
        e = [x for x in episodes if x["suite"] == s]
        if not e:
            continue
        k = sum(x["success"] for x in e)
        out["suites"][s] = {"n": len(e), "k": k, "success_rate": k / len(e), "wilson95": wilson(k, len(e)),
                            "published_card": PUBLISHED_CARD.get(s)}
    if out["suites"]:
        rates = [v["success_rate"] for v in out["suites"].values()]
        n = sum(v["n"] for v in out["suites"].values())
        k = sum(v["k"] for v in out["suites"].values())
        out["overall"] = {"n": n, "k": k, "success_rate": k / n, "wilson95": wilson(k, n),
                          "suite_mean": sum(rates) / len(rates), "n_suites": len(rates)}
    return out


def bank(out: pathlib.Path, rep: dict) -> None:
    tmp = out / "metrics.json.tmp"
    tmp.write_text(json.dumps(rep, indent=2, default=str))
    os.replace(tmp, out / "metrics.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--episodes-per-suite", type=int, default=EPISODES_PER_SUITE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--venv", default=VENV)
    ap.add_argument("--smoke", action="store_true", help="1 episode of libero_spatial task 0 (the bind gate)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    suites = [s.strip() for s in a.suites.split(",") if s.strip()]
    assert all(s in SUITES for s in suites), suites
    if a.smoke:
        plan = [("libero_spatial", [0], 1)]
    else:
        assert a.episodes_per_suite % TASKS_PER_SUITE == 0, "episodes per suite must be a multiple of 10 tasks"
        plan = [(s, list(range(TASKS_PER_SUITE)), a.episodes_per_suite // TASKS_PER_SUITE) for s in suites]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                              # HF_TOKEN + online: three hub downloads (docstring)
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    py = _install(a.venv)
    rep = {"experiment": "mendeleev_n15_molmoact2_libero", "policy": POLICY, "smoke": a.smoke,
           "protocol": {"suites": suites, "episodes_per_suite": a.episodes_per_suite, "seed": a.seed,
                        "inference_action_mode": "continuous", "per_episode_seed": True,
                        "camera_name_mapping": CAMERA_MAP, "plan": plan},
           "env": _env_versions(py), "lerobot_pin": LEROBOT_PIN, "published_card": PUBLISHED_CARD,
           "episodes": [], "aggregate": aggregate([]), "status": "running"}
    bank(out, rep)
    print(f"[N15] env {rep['env']}", flush=True)

    for suite, task_ids, n_per_task in plan:
        run_dir = out / "lerobot_eval" / suite
        cmd = eval_cmd(py, str(run_dir), suite, task_ids, n_per_task, a.seed)
        _run(cmd, env={**os.environ})
        eps = parse_eval_info(run_dir / "eval_info.json", suite, task_ids, n_per_task)
        rep["episodes"].extend(eps)
        rep["aggregate"] = aggregate(rep["episodes"])
        bank(out, rep)                             # incremental: finished suites survive a kill
        print(f"[N15] {suite}: {rep['aggregate']['suites'][suite]}", flush=True)

    rep["status"] = "done"
    bank(out, rep)
    print("N15_OK", json.dumps(rep["aggregate"], default=str), flush=True)


# ------------------------------------------------------------------------------------------- dry-run
def _fake_eval_info(suite: str, task_ids: list, n: int, seed: int) -> dict:
    """lerobot_eval.eval_policy_all schema with deterministic successes (task 3 always fails)."""
    per_task = []
    for tid in task_ids:
        succ = [bool((tid + k + seed) % 5 != 0) and tid != 3 for k in range(n)]
        per_task.append({"task_group": suite, "task_id": tid,
                         "metrics": {"sum_rewards": [float(s) for s in succ], "max_rewards": [float(s) for s in succ],
                                     "successes": succ, "video_paths": [], "predicted_video_paths": []}})
    return {"per_task": per_task, "per_group": {suite: {"n_episodes": n * len(task_ids)}}, "overall": {}}


def _dryrun():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = pathlib.Path(d)
        cmd = eval_cmd("/nonexistent/bin/python", str(out / "x"), "libero_10", list(range(10)), 2, SEED)
        assert cmd[:3] == ["/nonexistent/bin/python", "-m", "lerobot.scripts.lerobot_eval"]
        assert "--env.task_ids=[0,1,2,3,4,5,6,7,8,9]" in cmd and "--eval.n_episodes=2" in cmd
        assert '--env.camera_name_mapping={"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' in cmd
        assert "--policy.inference_action_mode=continuous" in cmd and f"--policy.eval_seed={SEED}" in cmd
        rep = {"episodes": [], "aggregate": aggregate([]), "status": "running"}
        assert rep["aggregate"]["overall"] is None
        bank(out, rep)
        for suite in SUITES[:2]:
            p = out / suite / "eval_info.json"
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(_fake_eval_info(suite, list(range(10)), 2, SEED)))
            eps = parse_eval_info(p, suite, list(range(10)), 2)
            assert len(eps) == 20 and all(e["success"] in (0, 1) for e in eps)
            assert all(e["success"] == 0 for e in eps if e["task_id"] == 3)
            rep["episodes"].extend(eps)
            rep["aggregate"] = aggregate(rep["episodes"])
            bank(out, rep)
        again = json.loads((out / "metrics.json").read_text())
        assert len(again["episodes"]) == 40 and set(again["aggregate"]["suites"]) == set(SUITES[:2])
        ag = again["aggregate"]
        s0 = ag["suites"]["libero_spatial"]
        assert s0["n"] == 20 and s0["k"] == sum(e["success"] for e in rep["episodes"][:20])
        assert abs(s0["success_rate"] - s0["k"] / 20) < 1e-12 and s0["published_card"] == 98.4
        lo, hi = s0["wilson95"]
        assert 0 <= lo <= s0["success_rate"] <= hi <= 1
        assert ag["overall"]["n"] == 40 and ag["overall"]["n_suites"] == 2
        # a short eval_info raises (never a silent partial rate)
        short = out / "short" / "eval_info.json"
        short.parent.mkdir()
        short.write_text(json.dumps(_fake_eval_info("libero_goal", [0, 1], 2, SEED)))
        raised = False
        try:
            parse_eval_info(short, "libero_goal", list(range(10)), 2)
        except AssertionError:
            raised = True
        assert raised
        assert math.isnan(wilson(0, 0)[0])            # NaN for an empty suite
        assert wilson(20, 20)[1] == 1.0 and wilson(20, 20)[0] > 0.8
    print("PASS molmoact2_libero dry-run (lerobot-eval command; eval_info parse; per-suite banking; Wilson CI)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _dryrun()
    else:
        main()
