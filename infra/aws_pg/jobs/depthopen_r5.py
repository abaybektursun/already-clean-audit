"""DEPTHOPEN R5 — one model on LIBERO-plus's camera-viewpoint and object-layout dimensions.

SPEC §3 R5: the benchmark's own task files, one trial per task as its protocol states, a fixed stratified
sample of 200 camera tasks and 200 layout tasks drawn once by seed across its five levels. Success rate
with a Wilson interval; the arm-minus-control difference is computed later, from two of these runs.

THE SAMPLE IS DRAWN ONCE AND BANKED. Every model — the released checkpoint, A0, A3 — is evaluated on the
same task ids, read from the banked sample, because a re-drawn sample would put a second source of
variance inside a 200-task difference. The first run of this job uploads the sample; later runs read it
and refuse to draw their own.

SPEC §5 also asks for the released checkpoint twice on the camera sample, to bank the run-to-run flip
rate before any arm difference is read. That is this job run twice with the same `--sample-s3`.

  pg run infra.aws_pg.jobs.depthopen_r5 --type g6e.2xlarge --budget-min 360 \\
     --weights models--allenai--MolmoAct2-Think-LIBERO \\
     --s3-prefix results/depthopen/r5/released/camera/1 \\
     --args " --model-name released --dimension camera"
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
# Xet left on: the flag forces the plain LFS path and the 6.4 GB asset archive comes down at
# about 5 MB/s with it. DEPTHOPEN_DISABLE_XET=1 restores the old behaviour.
if os.environ.get("DEPTHOPEN_DISABLE_XET") == "1":
    os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.aws_pg.jobs.depthopen_rollout import wilson
from infra.aws_pg.jobs.postcheck_files import Bank
from infra.pg import depthopen_plus as plus
from infra.pg import depthopen_train as dt

RELEASED = "allenai/MolmoAct2-Think-LIBERO"
S3_SAMPLE = f"s3://{dt.BUCKET}/depthopen/r5_sample/v1"
DIMENSION = {"camera": plus.CAMERA, "layout": plus.LAYOUT}
N_PER_DIMENSION = 200
MASTER_SEED = 20260909


def _hf_token() -> Optional[str]:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    r = subprocess.run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                        "--with-decryption", "--region", dt.REGION,
                        "--query", "Parameter.Value", "--output", "text"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        os.environ["HF_TOKEN"] = r.stdout.strip()
        return os.environ["HF_TOKEN"]
    return None


def stage_env(bank: Bank, work: pathlib.Path) -> Dict[str, object]:
    """The DEPTHANCHOR simulator environment first, then LIBERO-plus over it."""
    from infra.pg import depthanchor_libero as lib
    base = lib.install(extra_numpy=None)
    bank.stage("env_base", base)
    lp = plus.install(work, token=_hf_token())
    lp["init_state_loading"] = plus.enable_init_state_loading()
    bank.stage("env_plus", lp)
    print(f"[env] libero -> {lp['libero_file']}", flush=True)
    return {"base": base, "plus": lp}


def stage_sample(bank: Bank, work: pathlib.Path, root: pathlib.Path, *, sample_s3: str,
                 seed: int, n: int) -> Dict[str, object]:
    """Read the banked sample, or draw it once and upload it."""
    local = work / "r5_sample.json"
    if sample_s3:
        dt.s3("cp", f"{sample_s3}/sample.json", str(local))
        payload = json.loads(local.read_text())
        bank.stage("sample", {"read_from": f"{sample_s3}/sample.json",
                              "n": {k: len(v["tasks"]) for k, v in payload["dimensions"].items()},
                              "seed": payload["seed"]})
        return payload
    labels = plus.classification(root)
    table = plus.task_table(sorted({v["suite"] for v in labels.values()}), labels)
    bank.artifact("r5_task_table.json", {k: v for k, v in table.items() if k != "rows"})
    payload = {"seed": seed, "n_per_dimension": n, "lp_sha": plus.LP_SHA,
               "table": {k: v for k, v in table.items() if k != "rows"},
               "dimensions": {}}
    for key, category in DIMENSION.items():
        payload["dimensions"][key] = plus.sample_dimension(table["rows"], category=category,
                                                           n=n, seed=seed)
    local.write_text(json.dumps(payload, indent=1))
    dt.s3("cp", str(local), f"{S3_SAMPLE}/sample.json")
    bank.stage("sample", {"drawn": True, "uploaded_to": f"{S3_SAMPLE}/sample.json",
                          "n": {k: len(v["tasks"]) for k, v in payload["dimensions"].items()},
                          "per_level": {k: v["by_level_final"] for k, v in payload["dimensions"].items()},
                          "table": payload["table"]})
    print(f"[sample] {json.dumps({k: v['by_level_final'] for k, v in payload['dimensions'].items()})}",
          flush=True)
    return payload


def stage_model(bank: Bank, work: pathlib.Path, *, hf_s3: str, repo: str):
    """The arm's exported checkpoint from S3, or a released repo from the weight cache."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from infra.pg import depthanchor_model as dam
    if hf_s3:
        hf_dir = work / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)
        dt.s3("sync", hf_s3, str(hf_dir))
        path = hf_dir
    else:
        root = pathlib.Path(os.environ.get("HF_HOME", "/opt/hf-cache/hf")) / "hub"
        d = root / ("models--" + repo.replace("/", "--"))
        snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
        if not snaps:
            raise FileNotFoundError(f"no restored snapshot for {repo} under {root}")
        path = snaps[-1]
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(str(path), trust_remote_code=True,
                                                        dtype=torch.float32).to("cuda").eval()
    dam.NORM_TAG = "libero"
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    gate = dam.gate_weights_readout(model)
    bank.stage("model", {"path": str(path), "from_s3": bool(hf_s3), "repo": repo,
                         "depth_gate_weights": gate})
    return actor, model


def stage_rollout(bank: Bank, actor, work: pathlib.Path, tasks: List[Dict[str, object]], *,
                  dimension: str, model_name: str, limit: int, out_dir: pathlib.Path,
                  film_index: int = -1) -> Dict[str, object]:
    from infra.pg import depthanchor_libero as lib
    orient = json.loads(sh.pull_json(f"{sh.S3_CAMERA}/orientation.json",
                                     work / "orientation.json").read_text())
    flip, hflip = bool(orient["render_flip"]), bool(orient["render_hflip"])
    bank.stage("rollout_inputs", {"render_flip": flip, "render_hflip": hflip,
                                  "dimension": dimension, "n_tasks": len(tasks),
                                  "trials_per_task": 1,
                                  "why": "the benchmark's protocol is one trial per task"})
    policy = lib.ArmNormal()
    rows: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    films: Dict[str, object] = {}
    for k, task in enumerate(tasks[:limit] if limit else tasks):
        # SPEC §10 wants two films of the SAME perturbed task under A0 and A3, so the index into the
        # banked sample is the caller's, identical across models, not "whichever task happened to fail"
        film = (k == film_index)
        r = lib.rollout(actor, task["suite"], task_id=int(task["task_id"]), episode=0, policy=policy,
                        flip=flip, hflip=hflip, film=film, want_depth=film)
        rec = r.as_record()
        rec.update({"suite": task["suite"], "task_id": int(task["task_id"]), "name": task["name"],
                    "level": task["level"], "category": task["category"]})
        rows.append(rec)
        done = len(rows)
        print(f"[r5] {model_name}/{dimension} {task['suite']}/{task['task_id']} L{task['level']}: "
              f"success={r.success} steps={r.n_steps} {r.seconds:.0f}s ({done}/{len(tasks)})", flush=True)
        bank.stage("progress", {"tasks": done, "of": len(tasks),
                                "successes": sum(1 for x in rows if x.get("success")),
                                "elapsed_s": round(time.perf_counter() - t0, 1),
                                "seconds_per_task": round((time.perf_counter() - t0) / done, 1)})
        if film and r.frames:
            films = {"frames": np.stack(r.frames),
                     "bins": np.asarray(r.bins_per_call, dtype=np.int16),
                     "eef": np.asarray(r.eef_path, dtype=np.float32)}
            if r.depth_maps:
                films["depth"] = np.stack(r.depth_maps)
            np.savez_compressed(out_dir / f"film_{model_name}_{dimension}_{k}.npz", **films)
            bank.stage("film", {"task": task, "index": k, "n_frames": len(r.frames),
                                "success": bool(r.success),
                                "file": f"film_{model_name}_{dimension}_{k}.npz"})
        bank.artifact(f"r5_rollouts_{dimension}.json",
                      [{k2: v for k2, v in x.items() if k2 not in ("bins_per_call", "gate_per_call")}
                       for x in rows])
    by_level = {}
    for level in sorted({str(x["level"]) for x in rows}):
        sel = [x for x in rows if str(x["level"]) == level]
        by_level[level] = wilson(sum(1 for x in sel if x.get("success")), len(sel))
    overall = wilson(sum(1 for x in rows if x.get("success")), len(rows))
    by_suite = {}
    for suite in sorted({x["suite"] for x in rows}):
        sel = [x for x in rows if x["suite"] == suite]
        by_suite[suite] = wilson(sum(1 for x in sel if x.get("success")), len(sel))
    out = {"model": model_name, "dimension": dimension, "overall": overall, "by_level": by_level,
           "by_suite": by_suite, "n_tasks": len(rows),
           "seconds_total": round(time.perf_counter() - t0, 1),
           "seconds_per_task": round((time.perf_counter() - t0) / max(1, len(rows)), 1),
           "successes": [x["name"] for x in rows if x.get("success")][:50]}
    bank.stage("summary", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthopen")
    ap.add_argument("--model-name", required=True, help="released, A0, A3, ... — the row's label")
    ap.add_argument("--hf-s3", default="", help="an arm's exported checkpoint; empty = released repo")
    ap.add_argument("--repo", default=RELEASED)
    ap.add_argument("--dimension", default="camera", choices=sorted(DIMENSION))
    ap.add_argument("--sample-s3", default="", help="read the banked sample from here instead of drawing")
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--n", type=int, default=N_PER_DIMENSION)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many tasks (smoke)")
    ap.add_argument("--film-index", type=int, default=-1,
                    help="keep frames, depth and grids for this index into the banked sample, so the "
                         "same perturbed task can be filmed under two models")
    ap.add_argument("--stages", default="env,sample,rollout")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bank = Bank(pathlib.Path(args.out), f"depthopen_r5:{args.model_name}:{args.dimension}")
    bank.top("args", vars(args))
    bank.top("pins", {"lp_sha": plus.LP_SHA, "lp_assets": plus.LP_ASSETS_REPO,
                      "dimension": DIMENSION[args.dimension], "trials_per_task": 1})

    env = stage_env(bank, work) if "env" in stages else {}
    root = pathlib.Path(env["plus"]["root"]) if env else (work / "LIBERO-plus")
    sample = stage_sample(bank, work, root, sample_s3=args.sample_s3, seed=args.seed, n=args.n) \
        if "sample" in stages else None
    if "rollout" in stages:
        if sample is None:
            raise RuntimeError("rollout needs the sample stage")
        actor, model = stage_model(bank, work, hf_s3=args.hf_s3, repo=args.repo)
        tasks = sample["dimensions"][args.dimension]["tasks"]
        summary = stage_rollout(bank, actor, work, tasks, dimension=args.dimension,
                                model_name=args.model_name, limit=args.limit,
                                out_dir=pathlib.Path(args.out), film_index=args.film_index)
        bank.top("verdict", {**summary, "calls": actor.calls})
        print(json.dumps({k: v for k, v in summary.items() if k != "successes"},
                         indent=1, default=str), flush=True)
    else:
        bank.top("verdict", {"stages_run": stages,
                             "sample": None if sample is None else
                             {k: v["by_level_final"] for k, v in sample["dimensions"].items()}})
        print(json.dumps(bank.payload["verdict"], indent=1, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
