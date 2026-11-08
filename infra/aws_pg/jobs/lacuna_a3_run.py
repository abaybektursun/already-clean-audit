"""LACUNA Arm 3 — fine-tune ONE unit with ONE seed on LIBERO-Spatial and evaluate it at EVERY
requested budget checkpoint (spec §2.3, §6.3).

One training run serves both budget points: `--steps B*` with `--save-freq B*/4` writes checkpoints
along the way and `--eval-steps B*/4,B*` evaluates each. That halves the arm's training cost versus
one run per budget and — more importantly — makes the two budget points *nested* rather than
independent runs, which is the correct comparison for a savings curve.

The SAME job is the budget calibrator (spec §6.3): `--unit U1 --steps 8000
--eval-steps 500,1000,2000,4000,8000 --task-ids [0,1,2] --eval-episodes 3` traces the probe curve
from which B* = the smallest checkpoint reaching pc_success ≥ 0.5 is read.

Training and evaluation both go through lerobot's OWN harness — the N5a lesson: a hand-rolled
obs→action mapping cost five rollout attempts and ended in a version-drift wall.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.lacuna_a3_run --type g6e.8xlarge --no-weights \
      --args "--unit U3 --steps 4000 --eval-steps 1000,4000 --seed 0" \
      --s3-prefix results/lacuna_a3_U3_s0 --budget-min 600
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

S3 = "s3://pg-lineage-769063704696"
DATASET_REPO = "HuggingFaceVLA/libero"
SUITE = "libero_spatial"
HF_UNITS = {}   # every unit is now a local LIBERO-shaped checkpoint synced from S3 (see F0)


def _sh(cmd, **kw):
    print(f"[A3] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def unit_path(unit, work):
    """U1/U2 come from the Hub; U3/U4 from the S3 copy F0 wrote (built once, reused by every cell)."""
    if unit in HF_UNITS:
        return HF_UNITS[unit]
    d = pathlib.Path(work) / unit
    d.mkdir(parents=True, exist_ok=True)
    _sh(["aws", "s3", "sync", f"{S3}/weights/lacuna_units/{unit}/", str(d), "--region", "us-east-1"])
    assert (d / "config.json").exists(), f"{unit}: no config.json synced from S3 — run F0 first"
    return str(d)


def find_checkpoint(train_dir, step):
    """lerobot writes checkpoints/<zero-padded step>/pretrained_model (and a `last` symlink)."""
    ck = pathlib.Path(train_dir) / "checkpoints"
    assert ck.exists(), f"no checkpoints dir at {ck}"
    cands = [p for p in ck.iterdir() if p.name.isdigit() and int(p.name) == int(step)]
    if not cands:
        cands = [p for p in ck.iterdir() if p.name.lstrip("0") == str(step)]
    assert cands, (f"no checkpoint for step {step} — present: "
                   f"{sorted(p.name for p in ck.iterdir())}")
    pm = cands[0] / "pretrained_model"
    assert pm.exists(), f"{cands[0]} has no pretrained_model"
    return pm


def parse_success(eval_dir):
    """Pull pc_success (and the per-episode successes when present) out of lerobot's eval json."""
    out = {"files": [], "pc_success": None, "per_episode": None, "aggregated": {}}
    for f in sorted(pathlib.Path(eval_dir).rglob("*.json")):
        j = json.loads(f.read_text())
        if not isinstance(j, dict):
            continue
        out["files"].append(str(f))
        agg = j.get("aggregated", j)
        out["aggregated"][f.name] = agg
        for k in ("pc_success", "success_rate", "avg_success"):
            if isinstance(agg, dict) and k in agg:
                out["pc_success"] = float(agg[k])
        for k in ("per_episode", "episodes"):
            if k in j:
                out["per_episode"] = j[k]
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--unit", required=True, choices=("U1", "U2", "U3", "U4"))
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--eval-steps", default="", help="comma list; default = --steps only")
    ap.add_argument("--save-freq", type=int, default=0, help="0 → min(eval-steps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--task-ids", default="[0,1,2,3,4,5,6,7,8,9]")
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all LIBERO-Spatial episodes")
    ap.add_argument("--work", default="/work/a3")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    eval_steps = ([int(x) for x in a.eval_steps.split(",") if x.strip()] if a.eval_steps
                  else [a.steps])
    save_freq = a.save_freq or min(eval_steps)

    from infra.aws_pg.jobs.n5a_pi0_libero_eval import _install
    _install()
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.pg import ctrliface_lerobot as CL

    t0 = time.time()
    rep = {"object": "lacuna_a3_run_v1", "unit": a.unit, "steps": a.steps, "seed": a.seed,
           "eval_steps": eval_steps, "save_freq": save_freq, "batch_size": a.batch_size,
           "suite": SUITE, "dataset": DATASET_REPO, "task_ids": a.task_ids,
           "eval_episodes": a.eval_episodes, "gpu": torch.cuda.get_device_name(0)}
    cells = []
    rep["cells"] = cells                      # bound BEFORE the eval loop (fleet lesson)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    src = unit_path(a.unit, a.work)
    rep["unit_path"] = src
    meta = CL.dataset_meta(DATASET_REPO)
    eps = CL._episodes_for_suites(meta, ("spatial",))
    assert eps, "no LIBERO-Spatial episodes matched (task naming drift)"
    if a.max_episodes:
        eps = eps[: a.max_episodes]
    rep["n_train_episodes"] = len(eps)
    _w()

    train_dir = pathlib.Path(a.work) / f"train_{a.unit}_{a.steps}_{a.seed}"
    bindir = pathlib.Path(sys.executable).parent
    tcmd = ([str(bindir / "lerobot-train")] if (bindir / "lerobot-train").exists()
            else [sys.executable, "-m", "lerobot.scripts.train"])
    _sh(tcmd + [f"--dataset.repo_id={DATASET_REPO}", f"--dataset.episodes={json.dumps(eps)}",
                f"--policy.path={src}", "--policy.device=cuda", f"--output_dir={train_dir}",
                f"--steps={a.steps}", f"--batch_size={a.batch_size}", f"--seed={a.seed}",
                f"--save_freq={save_freq}", "--eval_freq=0", "--log_freq=100",
                "--wandb.enable=false"], env={**os.environ})
    rep["train_elapsed_s"] = round(time.time() - t0, 1)
    rep["checkpoints_present"] = sorted(p.name for p in (train_dir / "checkpoints").iterdir())
    _w()

    ecmd = ([str(bindir / "lerobot-eval")] if (bindir / "lerobot-eval").exists()
            else [sys.executable, "-m", "lerobot.scripts.eval"])
    for step in eval_steps:
        ck = find_checkpoint(train_dir, step)
        d = out / f"eval_{step}"
        _sh(ecmd + [f"--output_dir={d}", "--env.type=libero", f"--env.task={SUITE}",
                    f"--env.task_ids={a.task_ids}", "--eval.batch_size=1",
                    f"--eval.n_episodes={a.eval_episodes}", f"--policy.path={ck}",
                    f"--policy.n_action_steps={a.n_action_steps}", "--env.max_parallel_tasks=1"],
            env={**os.environ})
        r = parse_success(d)
        cells.append({"unit": a.unit, "seed": a.seed, "steps": int(step),
                      "pc_success": r["pc_success"], "per_episode": r["per_episode"],
                      "aggregated": r["aggregated"], "checkpoint": str(ck),
                      "elapsed_s": round(time.time() - t0, 1)})
        print(f"[A3] {a.unit} s{a.seed} step={step}: pc_success={r['pc_success']}", flush=True)
        _w()

    rep["elapsed_s"] = round(time.time() - t0, 1)
    _w()
    print("LACUNA_A3_RUN_OK", json.dumps([{k: c[k] for k in ("unit", "seed", "steps", "pc_success")}
                                          for c in cells], default=str), flush=True)


if __name__ == "__main__":
    main()
