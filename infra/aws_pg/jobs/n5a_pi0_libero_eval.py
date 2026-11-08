"""§5.3 N5a function arm — OFFICIAL lerobot LIBERO eval of π0.5 on AWS.

Earlier iterations hand-rolled the obs→action mapping for `lerobot/pi0_base` (zero-shot, guessed
state layout + norm stats) and hit a lerobot↔transformers version drift (`create_causal_mask`
`cache_position`). The clean path, per the lerobot LIBERO docs, is to use lerobot's OWN eval harness
(`lerobot-eval`) with the LIBERO-finetuned checkpoint `lerobot/pi05_libero_finetuned`:
  - lerobot's harness builds the LIBERO env + the correct obs/action/normalization itself (no guesswork);
  - the `[libero,pi]` extras pull the maintainer-pinned transformers (5.4–5.6) → no version drift;
  - the checkpoint is LIBERO-finetuned → a REAL success rate (reported ~97.5%; community repro ~72%),
    not a zero-shot number.

Connects to §5.1: π0.5's free-standing text reasoning is genuinely deleted (own-embed ≈ floor). Is its
ROBOT control intact? A LIBERO success rate is the function-arm readout. Fail-loud, no try/except around
the science.

  pg run infra.aws_pg.jobs.n5a_pi0_libero_eval --type g6e.8xlarge --no-weights --budget-min 120 \
     --args="--suite libero_spatial --task-ids [0,1,2] --episodes 3"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys


def _run(cmd, **kw):
    print(f"[N5A] $ {' '.join(cmd) if isinstance(cmd, list) else cmd}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _install():
    # EGL + build deps (LIBERO/robosuite/mujoco C-ext rendering on the headless pg AMI)
    _run(["bash", "-lc",
          "apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y "
          "libegl1 libgles2 libglvnd0 libgl1 libosmesa6 libglfw3 libglib2.0-0 libsm6 "
          "libxext6 libxrender1 libgomp1 git python3-dev python3.12-dev build-essential"])
    # lerobot main with the libero + pi extras → the maintainer-pinned transformers (5.4–5.6),
    # lerobot's LIBERO integration, and the π0/π0.5 policies. This is the documented eval recipe.
    _run([sys.executable, "-m", "pip", "install", "-q",
          "lerobot[libero,pi] @ git+https://github.com/huggingface/lerobot.git"])
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    # LIBERO writes an interactive config on FIRST import ("custom dataset path? (Y/N)") — there is no
    # stdin under lerobot-eval (EOFError). Pre-answer 'N' once now so ~/.libero/config.yaml exists.
    subprocess.run('echo "N" | %s -c "import libero.libero"' % sys.executable, shell=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-ids", default="[0,1,2]")   # restrict to a few tasks → cheap first number
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    ap.add_argument("--n-action-steps", type=int, default=10)  # matches OpenPI / lerobot LIBERO repro
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"suite": a.suite, "task_ids": a.task_ids, "episodes": a.episodes, "policy": a.policy}

    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                  # HF_TOKEN + online (checkpoint download); run_job sets OFFLINE=1
    import transformers
    import lerobot
    rep["transformers"] = transformers.__version__
    rep["lerobot"] = getattr(lerobot, "__version__", "?")
    print(f"[N5A] transformers={rep['transformers']} lerobot={rep['lerobot']}", flush=True)

    eval_out = out / "lerobot_eval"
    # lerobot-eval console script is installed in the venv bin next to the interpreter
    binp = pathlib.Path(sys.executable).parent / "lerobot-eval"
    base = [str(binp)] if binp.exists() else [sys.executable, "-m", "lerobot.scripts.eval"]
    cmd = base + [
        f"--output_dir={eval_out}",
        "--env.type=libero",
        f"--env.task={a.suite}",
        f"--env.task_ids={a.task_ids}",
        "--eval.batch_size=1",
        f"--eval.n_episodes={a.episodes}",
        f"--policy.path={a.policy}",
        f"--policy.n_action_steps={a.n_action_steps}",
        "--env.max_parallel_tasks=1",
    ]
    _run(cmd, env={**os.environ})

    # lerobot-eval writes an eval_info.json under output_dir; surface its success metric(s).
    found = sorted(glob.glob(str(eval_out / "**" / "*.json"), recursive=True))
    rep["eval_jsons"] = found
    agg = {}
    for f in found:
        d = json.loads(pathlib.Path(f).read_text())
        if isinstance(d, dict) and ("aggregated" in d or "avg_sum_reward" in d or "pc_success" in str(d)):
            agg[f] = d.get("aggregated", d)
    rep["aggregated"] = agg
    (out / "n5a_pi05_libero_eval.json").write_text(json.dumps(rep, indent=2, default=str))
    print(f"[N5A] DONE — eval jsons: {found}", flush=True)
    print(f"[N5A] aggregated: {json.dumps(agg, default=str)[:2000]}", flush=True)


if __name__ == "__main__":
    main()
