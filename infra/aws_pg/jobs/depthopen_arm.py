"""DEPTHOPEN — one arm, end to end: train it, export it, then R1, R2 and R3 on it.

SPEC §3's three model-side readouts, all on the SAME 40 rendered frames POSTCHECK used, with the same
instruments, so an arm's numbers sit beside the released checkpoint's without a translation step:

  R1  the gate after training — the bias out of the safetensors (POSTCHECK's weights-only reader) and the
      gate the model actually computes on those frames (from the battery's own per-call record)
  R2  does it read the grid — the DEPTHANCHOR battery through the checkpoint's adaptive-replay path:
      identity, 3 generator seeds, RGB ceiling, 4 shuffles, constant grid, removal. Removal is the gate
      forced to 0 for a gated arm and `enable_depth_reasoning=False` for A4, which has no gate.
  R3  where the expert's cross-attention goes — POSTCHECK's N2 probe on 20 of those frames

The exported HF checkpoint is uploaded before the readouts run, so R4 and R5 are separate jobs that can
be re-run or parallelized without retraining, and so a readout defect never costs a training run.

  pg run infra.aws_pg.jobs.depthopen_arm --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think --budget-min 300 \\
     --s3-prefix results/depthopen/arm/A3/1 --args " --arm A3 --steps 1000 --device-batch 2"
"""
from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from infra.aws_pg.jobs.postcheck_files import Bank
from infra.pg import depthopen_train as dt

BUCKET = dt.BUCKET
REGION = dt.REGION
S3_ARMS = f"s3://{BUCKET}/depthopen/arms/v1"
S3_FRAMES = f"s3://{BUCKET}/postcheck/frames/v1"
R3_FRAMES = 20


def _load_hf(hf_dir: pathlib.Path):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(hf_dir), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(str(hf_dir), trust_remote_code=True,
                                                        dtype=torch.float32).to("cuda").eval()
    return model, processor


def _free(model) -> None:
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def stage_frames(work: pathlib.Path) -> List[Dict[str, object]]:
    """POSTCHECK's 40 frames from held-out initial states — the same pack, so R2 is comparable."""
    from infra.aws_pg.jobs.postcheck_render import load_frames
    dest = work / "frames"
    dest.mkdir(parents=True, exist_ok=True)
    dt.s3("cp", f"{S3_FRAMES}/frames.npz", str(dest / "frames.npz"))
    dt.s3("cp", f"{S3_FRAMES}/frames_index.json", str(dest / "frames_index.json"))
    return load_frames(dest)


def stage_train(bank: Bank, src: pathlib.Path, work: pathlib.Path, data: Dict[str, object],
                *, arm: str, steps: int, device_batch: int, global_batch: int,
                seed: int) -> Dict[str, object]:
    save = work / f"train_{arm}_s{seed}"
    log = work / f"train_{arm}_s{seed}.log"
    argv = dt.train_argv(arm, steps=steps, device_batch=device_batch, global_batch=global_batch,
                         save_folder=save, seed=seed)
    print(f"[train] {arm}: {steps} steps, device_batch={device_batch}, global_batch={global_batch}",
          flush=True)

    def on_line(step, line):
        if step["step"] % 25 == 0:
            bank.stage("train_progress", step)
            print(f"[train] {step}", flush=True)

    run = dt.run_training(src, work, argv, data=data, log_path=log,
                          artifact_dir=bank.out, on_line=on_line)
    probe = run.get("grad_probe") or {}
    payload = {"arm": arm, "flags": dt.ARMS[arm], "steps": steps, "seed": seed,
               "device_batch_size": device_batch, "global_batch_size": global_batch,
               "seconds": run["seconds"], "seconds_per_step": run.get("seconds_per_step"),
               "loss_first_half_mean": run.get("loss_first_half_mean"),
               "loss_second_half_mean": run.get("loss_second_half_mean"),
               "loss_decreasing": run.get("loss_decreasing"),
               "peak_reserved_gb": probe.get("cuda_max_memory_reserved_gb"),
               # dedupe by name: `get_param_groups` is called more than once, so the raw record holds
               # each of the 72 gate tensors twice (A0's artifact shows the undeduped 144)
               "n_gate_trainable": len({e["name"] for e in probe.get("gate_params", [])
                                        if e.get("requires_grad")}) or probe.get("n_gate_trainable"),
               "n_gate_trainable_raw": probe.get("n_gate_trainable"),
               "gate_grad_norm_max": (round(max(probe.get("grad_norms", {}).values()), 8)
                                      if probe.get("grad_norms") else 0.0),
               "checkpoint_dirs": dt.checkpoint_listing(save)}
    bank.artifact(f"train_steps_{arm}.json", run["steps"])
    bank.stage("train", payload)
    return {**payload, "save_folder": save}


def stage_export(bank: Bank, src: pathlib.Path, work: pathlib.Path, save_folder: pathlib.Path,
                 *, arm: str, seed: int, upload: bool, label: str = "") -> pathlib.Path:
    merged = sorted(save_folder.glob("step*-merged"))
    if not merged:
        raise RuntimeError(f"no merged checkpoint in {save_folder}: "
                           f"{sorted(p.name for p in save_folder.glob('*'))}")
    hf_dir = work / f"hf_{arm}_s{seed}{label}"
    conv = dt.convert_to_hf(src, merged[-1], hf_dir, work=work)
    bank.stage("export", conv)
    if upload:
        # the label keeps a diagnostic run (e.g. the doubled-budget A0) from overwriting the equal-budget
        # checkpoint every comparison in this study is made against
        uri = f"{S3_ARMS}/{arm}/seed{seed}{label}/hf"
        t0 = time.perf_counter()
        dt.s3("sync", str(hf_dir), uri)
        bank.stage("upload", {"uri": uri, "seconds": round(time.perf_counter() - t0, 1),
                              "bytes": sum(p.stat().st_size for p in hf_dir.rglob("*") if p.is_file())})
        print(f"[upload] {uri}", flush=True)
    return hf_dir


def stage_r1(bank: Bank, hf_dir: pathlib.Path, arm: str) -> Dict[str, object]:
    """The trained gate, from the weights alone — no forward pass, POSTCHECK's P1a instrument."""
    from infra.pg import postcheck_weights as pcw
    read = pcw.read_gate(hf_dir)
    summary = pcw.summarize(read, init_bias=float(dt.ARMS[arm]["init_bias"]))
    payload = {"arm": arm, "init_bias_declared": dt.ARMS[arm]["init_bias"], **summary}
    bank.artifact(f"r1_gate_layers_{arm}.json", read)
    bank.stage("r1", payload)
    print(f"[r1] {json.dumps({k: v for k, v in payload.items() if not isinstance(v, (list, dict))})[:400]}",
          flush=True)
    return payload


def stage_r2(bank: Bank, hf_dir: pathlib.Path, work: pathlib.Path, frames: List[Dict[str, object]],
             *, arm: str) -> Dict[str, object]:
    """The battery. `remove_mode` follows the arm's architecture, not a default."""
    from infra.aws_pg.jobs import _depthanchor_shared as sh
    from infra.pg import depthanchor_model as dam
    from infra.pg import postcheck_battery as pb
    ladder, _ = sh.load_ladder(work)
    model, processor = _load_hf(hf_dir)
    dam.NORM_TAG = "libero"
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    remove_mode = "gate0" if dt.ARMS[arm]["gate"] else "no_depth"

    def on_progress(p):
        bank.stage("r2_progress", p)
        print(f"[r2] {p}", flush=True)

    bat = pb.run_battery(actor, frames, ladder, remove_mode=remove_mode, on_progress=on_progress)
    agg = pb.aggregate(bat)
    bank.artifact(f"r2_records_{arm}.json", bat["records"])
    bank.artifact(f"r2_diagnostics_{arm}.json", bat["diagnostics"])

    # R1's realized half: the gate the model computed on these frames. The value lives on the per-call
    # RECORDS, not on the diagnostics — reading the wrong one left this null for A0, A3 and A4, and the
    # number had to be recomputed from the banked records afterwards.
    gates = [r.get("gate_mean") for r in bat["records"] if r.get("gate_mean") is not None]
    realized = ({"n": len(gates), "min": float(np.min(gates)), "median": float(np.median(gates)),
                 "max": float(np.max(gates))} if gates else None)
    payload = {"arm": arm, "remove_mode": remove_mode, "ladder_median_code": ladder.median_code,
               "realized_gate_on_frames": realized,
               **{k: v for k, v in agg.items() if k != "per_frame"}}
    bank.stage("r2", payload)
    print(f"[r2] {arm}: m={agg.get('m')} shuffle/C={(agg.get('shuffle_over_ceiling') or {}).get('point')} "
          f"reads={agg.get('reads_the_prefix')}", flush=True)
    _free(model)
    return payload


def stage_r3(bank: Bank, hf_dir: pathlib.Path, work: pathlib.Path, frames: List[Dict[str, object]],
             *, arm: str) -> Dict[str, object]:
    """Cross-attention mass on the depth positions, POSTCHECK's N2 probe, unchanged."""
    from infra.aws_pg.jobs.postcheck_mech import AttentionProbe
    from infra.pg import depthanchor_model as dam
    model, processor = _load_hf(hf_dir)
    n_layers = int(model.config.action_expert_config.num_layers)
    dam.NORM_TAG = "libero"
    probe = AttentionProbe(model, n_layers)
    info = probe.install()
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    try:
        for f in frames[:R3_FRAMES]:
            probe.start_frame()
            actor.natural(f, seed=0)
    finally:
        probe.remove()
    out = dict(probe.summary(), arm=arm, n_expert_layers=n_layers, n_frames=min(R3_FRAMES, len(frames)),
               **info)
    # the cap is stated inside the artifact: a checker who re-derives the median from a truncated file
    # gets a different number and has no way to know why (measured: 0.1398 from 2,000 rows against 0.1441
    # from all 7,200)
    bank.artifact(f"r3_rows_{arm}.json",
                  {"why": "a sample of the probe's rows; the summary in metrics.json uses ALL of them",
                   "n_rows_recorded": len(probe.rows), "n_rows_in_this_file": min(2000, len(probe.rows)),
                   "rows": probe.rows[:2000]})
    bank.stage("r3", out)
    print(f"[r3] {arm}: {json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))})[:400]}",
          flush=True)
    _free(model)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthopen")
    ap.add_argument("--arm", required=True, choices=sorted(dt.ARMS))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device-batch", type=int, default=2)
    ap.add_argument("--global-batch", type=int, default=0, help="0 = same as device batch")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--stages", default="train,export,r1,r2,r3")
    ap.add_argument("--from-hf-s3", default="",
                    help="skip training and read the arm's exported checkpoint from this URI")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--upload-label", default="",
                    help="suffix on the S3 checkpoint path, so a diagnostic run (a doubled step budget, "
                         "say) cannot overwrite the equal-budget checkpoint the comparisons use")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bank = Bank(pathlib.Path(args.out), f"depthopen_arm:{args.arm}")
    bank.top("args", vars(args))
    bank.top("pins", {"arm": args.arm, "flags": dt.ARMS[args.arm], "molmoact2_sha": dt.MO2_SHA,
                      "base": dt.BASE_CKPT, "subset_repo": dt.SUBSET_REPO, "seed": args.seed})

    src: Optional[pathlib.Path] = None
    hf_dir: Optional[pathlib.Path] = None
    if "train" in stages or "export" in stages:
        t0 = time.perf_counter()
        src = dt.fetch_source(work)
        installed = dt.install_stack(src)
        bank.stage("stack", {"root": str(src), "seconds": round(time.perf_counter() - t0, 1),
                             **installed})

    if "train" in stages:
        data = dt.sync_data(work)
        bank.stage("data", data)
        trained = stage_train(bank, src, work, data, arm=args.arm, steps=args.steps,
                              device_batch=args.device_batch,
                              global_batch=args.global_batch or args.device_batch, seed=args.seed)
        hf_dir = stage_export(bank, src, work, trained["save_folder"], arm=args.arm, seed=args.seed,
                              upload=not args.no_upload, label=args.upload_label)
    elif args.from_hf_s3:
        hf_dir = work / f"hf_{args.arm}_s{args.seed}{args.upload_label}"
        hf_dir.mkdir(parents=True, exist_ok=True)
        dt.s3("sync", args.from_hf_s3, str(hf_dir))
        bank.stage("fetch_checkpoint", {"uri": args.from_hf_s3, "dir": str(hf_dir),
                                        "files": sorted(p.name for p in hf_dir.iterdir())[:12]})
    if hf_dir is None:
        raise RuntimeError("nothing to read: pass --stages with train, or --from-hf-s3")

    frames = stage_frames(work) if {"r2", "r3"} & set(stages) else []
    verdict: Dict[str, object] = {"arm": args.arm, "flags": dt.ARMS[args.arm], "seed": args.seed}
    if "r1" in stages:
        r1 = stage_r1(bank, hf_dir, args.arm)
        verdict["r1"] = {k: r1.get(k) for k in
                         ("present", "n_layers", "bias_min", "bias_max", "sigmoid_min", "sigmoid_max",
                          "max_abs_bias_deviation", "verdict")}
    if "r2" in stages:
        r2 = stage_r2(bank, hf_dir, work, frames, arm=args.arm)
        verdict["r2"] = {k: r2.get(k) for k in
                         ("m", "reads_the_prefix", "remove_mode", "n_frames_usable", "g0_max",
                          "realized_gate_on_frames")}
        for key in ("shuffle_over_ceiling", "ratio_to_ceiling", "ratio_to_seed_floor"):
            verdict["r2"][key] = r2.get(key)
    if "r3" in stages:
        r3 = stage_r3(bank, hf_dir, work, frames, arm=args.arm)
        verdict["r3"] = {k: r3.get(k) for k in
                         ("measured", "n_layers", "depth_mass", "uniform_share", "ratio_to_uniform",
                          "probabilities_normalized")}
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
