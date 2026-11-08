"""DEPTHOPEN R4 — standard LIBERO for one arm, and G-known for A0.

SPEC §3 R4: 4 suites × 10 tasks × 5 episodes = 200 episodes per arm, Wilson intervals. This is a collapse
check, not a comparison: the band is ±8 points on the pooled rate, and §0 already says nothing here is
claimed against the released checkpoint, which trained on roughly forty times more samples.

SPEC §5 G-known: A0 must reach at least 8 of 10 on LIBERO-Object under the 180° transform, or our
fine-tune is broken and no arm is evaluated. The transform is not chosen here — it is read from
DEPTHANCHOR's banked orientation artifact, the same one its own closed loop used, which POSTCHECK's P5
then confirmed independently at 0.864 image correlation and 10/10 in closed loop.

The rollout engine, the environment install, the horizons and the settle steps are DEPTHANCHOR's, so an
arm's success rate is measured by the instrument that produced the released checkpoint's.

  pg run infra.aws_pg.jobs.depthopen_rollout --type g6e.2xlarge --no-weights --budget-min 360 \\
     --s3-prefix results/depthopen/r4/A3/1 \\
     --args " --arm A3 --hf-s3 s3://pg-lineage-769063704696/depthopen/arms/v1/A3/seed20260909/hf"
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Dict, List

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.aws_pg.jobs.postcheck_files import Bank
from infra.pg import depthanchor_core as core
from infra.pg import depthopen_train as dt

G_KNOWN_SUITE = "libero_object"
G_KNOWN_FLOOR = 8            # SPEC §5: at least 8 of 10 for A0
BAND_POINTS = 8.0            # SPEC §3 R4: the collapse band on the pooled rate


def wilson(k: int, n: int, z: float = 1.96) -> Dict[str, float]:
    """The interval DEPTHANCHOR used, with exact ends at 0 and n so a figure never gets a negative bar."""
    if n == 0:
        return {"rate": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return {"rate": round(p, 4), "lo": round(lo, 4), "hi": round(hi, 4), "k": int(k), "n": int(n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthopen")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--hf-s3", required=True, help="the arm's exported HF checkpoint in S3")
    ap.add_argument("--suites", default=",".join(core.SUITES))
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--episodes-per-task", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    for s in suites:
        if s not in core.SUITES:
            raise ValueError(f"unknown suite {s!r}")
    bank = Bank(pathlib.Path(args.out), f"depthopen_r4:{args.arm}")
    bank.top("args", vars(args))

    from infra.pg import depthanchor_libero as lib
    env_report = lib.install(extra_numpy=None)
    e, b, t = lib.make_env(suites[0], 0, camera_depths=True, image_size=lib.IMAGE_SIZE)
    e.close()
    bank.stage("env", {"install": env_report})

    orient = json.loads(sh.pull_json(f"{sh.S3_CAMERA}/orientation.json",
                                     work / "orientation.json").read_text())
    for key in ("render_flip", "render_hflip"):
        if orient.get(key) is None:
            raise RuntimeError(f"the banked orientation artifact has no `{key}`")
    flip, hflip = bool(orient["render_flip"]), bool(orient["render_hflip"])
    bank.stage("inputs", {"render_flip": flip, "render_hflip": hflip,
                          "chosen_convention": orient.get("chosen_convention"),
                          "suites": suites, "tasks": args.tasks,
                          "episodes_per_task": args.episodes_per_task,
                          "horizons": {s: lib.HORIZON[s] for s in suites}})

    hf_dir = work / f"hf_{args.arm}"
    hf_dir.mkdir(parents=True, exist_ok=True)
    dt.s3("sync", args.hf_s3, str(hf_dir))
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from infra.pg import depthanchor_model as dam
    processor = AutoProcessor.from_pretrained(str(hf_dir), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(str(hf_dir), trust_remote_code=True,
                                                        dtype=torch.float32).to("cuda").eval()
    dam.NORM_TAG = "libero"
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    gate = dam.gate_weights_readout(model)
    bank.stage("model", {"arm": args.arm, "uri": args.hf_s3, "depth_gate_weights": gate,
                         "norm_tags_shipped": sorted(model._get_robot_stats().metadata_by_tag)})

    policy = lib.ArmNormal()
    rows: List[Dict[str, object]] = []
    t0 = time.perf_counter()
    for suite in suites:
        for task_id in range(args.tasks):
            for ep in range(args.episodes_per_task):
                r = lib.rollout(actor, suite, task_id=task_id, episode=ep, policy=policy,
                                flip=flip, hflip=hflip, film=False, want_depth=False,
                                max_steps=args.max_steps or None)
                rec = r.as_record()
                rec.update({"suite": suite, "task_id": task_id, "episode": ep})
                rows.append(rec)
                done = len(rows)
                print(f"[r4] {args.arm} {suite}/{task_id} ep{ep}: success={r.success} "
                      f"steps={r.n_steps} {r.seconds:.0f}s ({done} done)", flush=True)
                bank.stage("progress", {"rollouts": done, "elapsed_s": round(time.perf_counter() - t0, 1),
                                        "seconds_per_rollout": round((time.perf_counter() - t0) / done, 1),
                                        "successes": sum(1 for x in rows if x.get("success"))})
                bank.artifact("rollouts.json", [{k: v for k, v in x.items()
                                                 if k not in ("bins_per_call", "gate_per_call")}
                                                for x in rows])

    by_suite = {}
    for suite in suites:
        sel = [x for x in rows if x["suite"] == suite]
        by_suite[suite] = wilson(sum(1 for x in sel if x.get("success")), len(sel))
    pooled = wilson(sum(1 for x in rows if x.get("success")), len(rows))
    g_known = None
    if G_KNOWN_SUITE in by_suite:
        sel = [x for x in rows if x["suite"] == G_KNOWN_SUITE and x["episode"] == 0]
        k = sum(1 for x in sel if x.get("success"))
        g_known = {"suite": G_KNOWN_SUITE, "episode_0_only": True, "k": k, "n": len(sel),
                   "floor": G_KNOWN_FLOOR, "passes": bool(k >= G_KNOWN_FLOOR),
                   "why": ("SPEC §5 reads G-known on 10 tasks of one suite; episode 0 of each task is the "
                           "same set of initial states DEPTHANCHOR's own loop used")}
    summary = {"arm": args.arm, "by_suite": by_suite, "pooled": pooled, "g_known": g_known,
               "band_points": BAND_POINTS, "n_rollouts": len(rows),
               "seconds_total": round(time.perf_counter() - t0, 1),
               "seconds_per_rollout": round((time.perf_counter() - t0) / max(1, len(rows)), 1),
               "calls": actor.calls}
    bank.stage("summary", summary)
    bank.top("verdict", summary)
    print(json.dumps(summary, indent=1, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
