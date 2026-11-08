"""POSTCHECK cells N1 and P3 — the same battery on the two checkpoints.

P3 asks whether the released fine-tune's depth grid is inert on frames it has never seen. N1 asks the
question the message depends on: does the UN-GATED base read its grid? If it does, the fine-tune's gate
closed a channel post-training had trained open, and the message to the authors is about their recipe.
If it does not, the prefix was inert before the gate existed and the gate is incidental. The draft
message is wrong under the first outcome, which is why this runs before anything is posted.

The two checkpoints do not fit on one card at fp32, so each is loaded, run and freed in turn.

`--plan` names what to run as `tag:frames` pairs, because N1 needs the base under two normalizations and
the tag is not cosmetic — it selects the state normalizer, the action unnormalizer AND the setup/control
text in the prompt. Both are banked; the readout is within-tag, so the tag cannot manufacture a
difference between edits.

  pg run infra.aws_pg.jobs.postcheck_battery_job --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --s3-prefix results/postcheck/p3/1 \\
     --budget-min 120 --args " --model tuned --plan libero:rendered"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time
from typing import Dict, List, Optional

import numpy as np

from infra.aws_pg.jobs.postcheck_files import Bank
from infra.aws_pg.jobs.postcheck_render import load_frames
from infra.pg import postcheck_battery as pb

BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
S3_FRAMES = f"s3://{BUCKET}/postcheck/frames/v1"
S3_DA_PACK = f"s3://{BUCKET}/depthanchor/frames/v1"
S3_LADDER = f"s3://{BUCKET}/depthanchor/ladder/v1"
S3_CODE = f"s3://{BUCKET}/postcheck/code/v1"
MODELS = {"tuned": "allenai/MolmoAct2-Think-LIBERO", "base": "allenai/MolmoAct2-Think"}
REMOVE_MODE = {"tuned": "gate0", "base": "no_depth"}


def _s3(*args: str) -> None:
    subprocess.run(["aws", "s3", *args, "--region", REGION], check=True)


def _pull_rendered(work: pathlib.Path) -> List[Dict[str, object]]:
    dest = work / "rendered"
    dest.mkdir(parents=True, exist_ok=True)
    _s3("cp", f"{S3_FRAMES}/frames.npz", str(dest / "frames.npz"))
    _s3("cp", f"{S3_FRAMES}/frames_index.json", str(dest / "frames_index.json"))
    return load_frames(dest)


def _pull_dataset_pack(work: pathlib.Path) -> List[Dict[str, object]]:
    """DEPTHANCHOR's frozen 120 frames, so N1's base numbers sit beside the fine-tune's on the same
    images. Read from the parent study's banked pack; nothing here writes to it."""
    from infra.pg import depthanchor_frames as frames_mod
    dest = work / "da_pack"
    dest.mkdir(parents=True, exist_ok=True)
    _s3("cp", f"{S3_DA_PACK}/frames.npz", str(dest / "frames.npz"))
    _s3("cp", f"{S3_DA_PACK}/frames_index.json", str(dest / "frames_index.json"))
    pack = frames_mod.load_pack(dest)
    return list(pack["frames"])


def _snapshot_on_disk(repo: str) -> pathlib.Path:
    """The restored snapshot directory, found on disk rather than resolved through the hub cache.

    `snapshot_download(local_files_only=True)` needs `refs/main` to resolve a revision, and the POSTCHECK
    seeder downloaded the base checkpoint at an explicit pinned revision, which writes no `refs/main`.
    That killed `results/postcheck/n1/1` with `LocalEntryNotFoundError` on a checkpoint that was sitting
    on the disk. Loading from the directory needs no cache bookkeeping at all.
    """
    import os
    root = pathlib.Path(os.environ.get("HF_HOME", "/opt/hf-cache/hf")) / "hub"
    d = root / ("models--" + repo.replace("/", "--"))
    snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").is_dir() else []
    if not snaps:
        raise FileNotFoundError(f"no restored snapshot for {repo} under {root}")
    return snaps[-1]


def _load(which: str):
    """One checkpoint at a time; `predict_action` needs the whole thing resident at fp32."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    path = _snapshot_on_disk(MODELS[which])
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(path), trust_remote_code=True, dtype=torch.float32).to("cuda").eval()
    return model, processor, path


def _free(model) -> None:
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/postcheck")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--plan", required=True,
                    help="comma-separated `tag:frames` runs, frames in {rendered,dataset,both}; "
                         "e.g. `libero:both,franka_molmoact:rendered`")
    ap.add_argument("--limit-frames", type=int, default=0)
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = Bank(pathlib.Path(args.out), f"postcheck_battery:{args.model}")
    bank.top("args", vars(args))

    from infra.aws_pg.jobs import _depthanchor_shared as sh
    ladder, ladder_payload = sh.load_ladder(work)

    sets: Dict[str, List[Dict[str, object]]] = {}
    plan = [p.strip() for p in args.plan.split(",") if p.strip()]
    need = {p.split(":", 1)[1] for p in plan}
    if need & {"rendered", "both"}:
        sets["rendered"] = _pull_rendered(work)
    if need & {"dataset", "both"}:
        sets["dataset"] = _pull_dataset_pack(work)
    sets["both"] = list(sets.get("rendered", [])) + list(sets.get("dataset", []))
    bank.stage("inputs", {"plan": plan, "model": MODELS[args.model],
                          "remove_mode": REMOVE_MODE[args.model],
                          "frames": {k: len(v) for k, v in sets.items()},
                          "ladder_median_code": ladder.median_code})

    model, processor, path = _load(args.model)
    from infra.pg import depthanchor_model as dam
    actor = dam.Actor(model, processor, enable_cuda_graph=False)
    gate = dam.gate_weights_readout(model)
    stats = model._get_robot_stats()
    tags_shipped = sorted(stats.metadata_by_tag)
    bank.stage("model", {"repo": MODELS[args.model], "snapshot": str(path),
                         "depth_gate_weights": gate, "norm_tags_shipped": tags_shipped,
                         "norm_mode": str(stats.norm_mode)})
    print(f"[model] {MODELS[args.model]} tags={tags_shipped} gate_present={gate.get('present')}",
          flush=True)

    # the LIBERO tag, if this checkpoint does not ship one
    tag_notes: Dict[str, object] = {}
    if any(p.split(":", 1)[0] == "libero" for p in plan) and "libero" not in tags_shipped:
        local = work / "tuned_norm_stats.json"
        _s3("cp", f"{S3_CODE}/think_libero/norm_stats.json", str(local))
        payload = json.loads(local.read_text())
        meta = (payload.get("metadata_by_tag") or {}).get("libero")
        if meta is None:
            raise RuntimeError("the fine-tune's norm_stats.json has no `libero` tag to copy")
        tag_notes = pb.add_norm_tag(model, "libero", meta)
        tag_notes["source_norm_mode"] = str(payload.get("norm_mode"))
        tag_notes["norm_modes_match"] = bool(str(payload.get("norm_mode")) == str(stats.norm_mode))
        bank.stage("norm_tag_added", tag_notes)
        print(f"[tag] libero copied into {args.model}: {json.dumps(tag_notes)[:300]}", flush=True)

    t0 = time.perf_counter()
    runs: Dict[str, object] = {}
    for entry in plan:
        tag, which = entry.split(":", 1)
        frames = sets[which]
        if args.limit_frames:
            frames = frames[:args.limit_frames]
        actor.norm_tag = tag              # `call_no_depth` reads it; `Actor._call` uses NORM_TAG
        _set_actor_tag(actor, tag)
        print(f"[run] {entry}: {len(frames)} frames, tag={tag}, "
              f"remove={REMOVE_MODE[args.model]}", flush=True)

        def on_progress(p, entry=entry):
            p = dict(p, run=entry)
            bank.stage("progress", p)

        bat = pb.run_battery(actor, frames, ladder, remove_mode=REMOVE_MODE[args.model],
                             on_progress=on_progress)
        agg = pb.aggregate(bat)
        key = entry.replace(":", "_")
        bank.artifact(f"records_{key}.json", bat["records"])
        bank.artifact(f"diagnostics_{key}.json", bat["diagnostics"])
        runs[entry] = agg
        bank.stage(f"aggregate_{key}", agg)
        print(f"[run] {entry}: m={agg.get('m')} shuffle/C="
              f"{(agg.get('shuffle_over_ceiling') or {}).get('point')} "
              f"reads={agg.get('reads_the_prefix')}", flush=True)

    _free(model)
    verdict = {
        "model": MODELS[args.model], "remove_mode": REMOVE_MODE[args.model],
        "gate_present": gate.get("present"),
        "norm_tag_added": tag_notes or None,
        "runs": {k: {"m": v.get("m"),
                     "shuffle_over_ceiling": (v.get("shuffle_over_ceiling") or {}).get("point"),
                     "shuffle_ci95": (v.get("shuffle_over_ceiling") or {}).get("ci95"),
                     "reads_the_prefix": v.get("reads_the_prefix"),
                     "remove_over_ceiling": (v.get("ratio_to_ceiling", {}).get("remove") or {}).get("point"),
                     "remove_over_seed_floor": (v.get("ratio_to_seed_floor", {}).get("remove") or {}).get("point"),
                     "const_over_ceiling": (v.get("ratio_to_ceiling", {}).get("const") or {}).get("point"),
                     "g0_max": v.get("g0_max"), "g0_passes": v.get("g0_passes_1e-4"),
                     "n_frames": v.get("n_frames_usable"),
                     "tail_mismatches": v.get("tail_mismatches")}
                 for k, v in runs.items()},
        "seconds_total": round(time.perf_counter() - t0, 1),
        "calls": actor.calls,
    }
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1, default=str), flush=True)


def _set_actor_tag(actor, tag: str) -> None:
    """`depthanchor_model.Actor._call` passes the module-level NORM_TAG. Rebind it for this run so both
    the injection path and the no-depth path use the tag this run is named after."""
    from infra.pg import depthanchor_model as dam
    dam.NORM_TAG = tag
    actor.norm_tag = tag


if __name__ == "__main__":
    main()
