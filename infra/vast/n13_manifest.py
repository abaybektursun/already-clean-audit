#!/usr/bin/env python3
"""The Vast lane manifest for one N13 launch, from that launch's steps.json (prepared here; renting is controller.py's).

    python3 infra/vast/n13_manifest.py --attempt s3gate-01v --steps infra/aws_pg/steps/n13_s3gate-01.json --gpu H100_NVL
        [--from-attempt s4armsA-01] [--budget-min M] [--note TEXT]

Writes, under infra/vast/manifests/:
  n13_steps/<attempt>.json   the steps, unchanged except every results/next_n13/<from-attempt>/ path (S4's inputs) -> <attempt>/;
  n13_<attempt>.json         the lane manifest: the n13 image, one GPU of --gpu, the n13_pipeline cell on
                             s3://<bucket>/results/next_n13/<attempt>/steps.json with budget_min = the steps' --allowance-min (the
                             pipeline's hard kill, as pg run --budget-min on AWS) unless --budget-min sets it lower, and
                             watch.max_billed_h = budget_min / 60 + 0.25.
Every generation step must carry the same --allowance-min. Fail-loud: no fallback.
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET = "pg-lineage-769063704696"
# the image built by n13_build.sh (record: s3://<bucket>/code/shackleton/images/image_n13-v1.json)
IMAGE = "769063704696.dkr.ecr.us-east-1.amazonaws.com/shackleton:n13-v1"
IMAGE_DIGEST = "sha256:fb6b5525279e0711e79dda41941b0197e344c32fe66bf1e61611067f8c4ce770"
IMAGE_PULL_GB = 41.9         # compressed bytes the host pulls (ECR imageSizeInBytes); billed at the host's inet_down_cost
# One GPU model per launch set (the plan's lane rule 5). Per model: the Vast query, the name lifecycle.sh greps in nvidia-smi
# (the controller also filters offers on it), and the bandwidth cap ($/GB; no L40S or H100 host had free bandwidth on 2026-11-05/06).
# Every query: one GPU, CUDA >= 13.0 (the venv's torch 2.13 + nvidia-cuda-runtime 13.0), host RAM of the smallest AWS box N13 runs
# on (g6e.2xlarge, 64 GiB), a pipe that pulls the 41.9 GB image inside the 30 min load bound. The head's decision of 2026-11-06:
# S3 and S4 likely on an H100, NVL 94 GB preferred, SXM 80 GB acceptable.
GPUS = {
    "L40S": ("gpu_name=L40S num_gpus=1 cuda_vers>=13.0 cpu_ram>=64 inet_down>=600", "L40S", 0.01),
    "H100_NVL": ("gpu_name=H100_NVL num_gpus=1 cuda_vers>=13.0 cpu_ram>=64 inet_down>=400", "H100 NVL", 0.01),
    "H100_SXM": ("gpu_name=H100_SXM num_gpus=1 cuda_vers>=13.0 cpu_ram>=64 inet_down>=400", "H100", 0.04),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", required=True, help="the Vast attempt name, e.g. s3gate-01v")
    ap.add_argument("--steps", required=True, help="the launch's steps.json")
    ap.add_argument("--from-attempt", default=None, help="the AWS attempt whose S3 paths the steps name (S4: s4armsA-01)")
    ap.add_argument("--note", default="", help="appended to the manifest's _comment (e.g. that its numbers are provisional)")
    ap.add_argument("--gpu", required=True, choices=sorted(GPUS))
    ap.add_argument("--budget-min", type=float, default=None,
                    help="the pipeline's hard kill, when a spend cap needs it below the steps' --allowance-min (default: equal)")
    a = ap.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9]+-\d\dv", a.attempt), a.attempt
    text = open(a.steps).read()
    if a.from_attempt:
        text = text.replace(f"results/next_n13/{a.from_attempt}/", f"results/next_n13/{a.attempt}/")
    steps = json.loads(text)
    assert all("results/next_n13/" not in s.get("args", "") or f"results/next_n13/{a.attempt}/" in s["args"] for s in steps), \
        "a step names an S3 path outside this attempt's prefix"
    allow = {float(m) for s in steps for m in re.findall(r"--allowance-min (\S+)", s.get("args", ""))}
    assert len(allow) == 1, f"the generation steps must carry one --allowance-min, found {sorted(allow)}"
    allowance = allow.pop()
    budget_min = a.budget_min if a.budget_min is not None else allowance
    assert budget_min <= allowance, (budget_min, allowance)
    query, expected, cost_max = GPUS[a.gpu]
    smoke = a.attempt.startswith("s2smoke")
    out_steps = os.path.join(HERE, "manifests", "n13_steps", f"{a.attempt}.json")
    with open(out_steps, "w") as f:
        f.write(json.dumps(steps, indent=1) + "\n")
    rental = f"n13_{a.attempt}"
    man = {
        "rental": rental, "session": "N13", "image": IMAGE, "image_digest": IMAGE_DIGEST,
        "disk": 140 if smoke else 200, "gpu_query": query, "expected_gpu": expected, "inet_cost_max": cost_max,
        "bound": "overhead", "weights_mode": "baked",
        "estimated_billed_h": round(budget_min / 60 / 1.3, 2), "estimated_download_gb": IMAGE_PULL_GB,
        "code_paths": ["studies/reproductions/alwayson/code"], "env": {},
        "cells": [{"id": "n13", "module": "infra.vast.cells.n13_pipeline", "attempt": a.attempt,
                   "steps_s3": f"s3://{BUCKET}/results/next_n13/{a.attempt}/steps.json", "budget_min": budget_min,
                   "upload_probe_gb": 1 if smoke else 0}],
        "watch": {"max_billed_h": round(budget_min / 60 + 0.25, 2), "load_backstop_min": 30,
                  "progress_stall_min": 20 if smoke else 30, "progress_kill_min": 45 if smoke else 120},
        "_comment": (f"N13 {a.attempt} on the Vast lane (infra/vast/operations/2026-11-05_n13_lane.md), generated by n13_manifest.py "
                     f"from {a.steps}" + (f" with results/next_n13/{a.from_attempt}/ -> {a.attempt}/" if a.from_attempt else "")
                     + f". GPU {a.gpu}. budget_min {budget_min:g}"
                     + (" = the steps' --allowance-min" if budget_min == allowance else f" (the steps' --allowance-min is {allowance:g})")
                     + "; the cell writes the AWS out/ layout to "
                     f"s3://{BUCKET}/results/next_n13/{a.attempt}/. Image {IMAGE} = {IMAGE_DIGEST}; weights baked." + (f" {a.note}" if a.note else "")),
    }
    out_man = os.path.join(HERE, "manifests", f"{rental}.json")
    with open(out_man, "w") as f:
        f.write(json.dumps(man, indent=2) + "\n")
    print(f"wrote {out_steps} ({len(steps)} steps) and {out_man}: budget_min {budget_min:g}, "
          f"max_billed_h {man['watch']['max_billed_h']}")


if __name__ == "__main__":
    main()
