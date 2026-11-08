"""DEPTHOPEN J1 — the training subset and its depth companions, from the released dataset.

WHAT THIS JOB PRODUCES, in the order the study needs it:

  1. `enumerate`   every data parquet path at the pinned revision, from the Hub listing. Never a count.
  2. `scan`        the four index columns of EVERY file, and with them the corrected version of
                   POSTCHECK's P4 comparison — that claim was measured over a hand-typed 378 of the 379
                   files, so the same instrument is re-run here over the enumerated list.
  3. `select`      whole files, stratified by suite, systematically spread, seeded: SPEC §4's 400
                   episodes without downloading three quarters of the release.
  4. `build`       a fresh self-consistent LeRobot v3 dataset of those files, with `meta` derived from
                   the files' own columns, then verified against itself and against the source bytes.
  5. `companions`  the makers' own `generate_depth_annotation.py`, vendored and driven per file, writing
                   `depth_codes` / `buffer_codes` / `depth_updated_mask`. Throughput is measured on the
                   first file before the rest run, because that rate decides whether the SPEC's subset
                   or the whole release gets annotated.

Both datasets are uploaded whole: every arm's training job reads them, and the exact bytes trained on
are then a banked artifact rather than a re-derivation.

  pg run infra.aws_pg.jobs.depthopen_data --type g5.xlarge --budget-min 180 \\
     --weights models--allenai--MolmoAct-7B-D-0812 \\
     --s3-prefix results/depthopen/data/1 --args " --per-suite-episodes 100"
"""
from __future__ import annotations

import os

# huggingface_hub reads the offline flag at import time and run_job.sh exports 1; this job's whole
# purpose is to read the release, so the flag is flipped before any hub import happens.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
# NOT disabling Xet here, and that is a measurement: attempt 1 ran with HF_HUB_DISABLE_XET=1 (inherited
# from the lineage program's note) and the 91 selected files came down at 1-4 MB/s, ~40 minutes for 8 GB.
# The flag forces the plain LFS path on a Xet-backed repository. Set DEPTHOPEN_DISABLE_XET=1 to restore it.
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

from infra.aws_pg.jobs.postcheck_files import Bank
from infra.pg import depthopen_subset as ds
from infra.pg import postcheck_dataset as pcd

BUCKET = "pg-lineage-769063704696"
REGION = "us-east-1"
DATASET = "allenai/MolmoAct2-LIBERO-Dataset"
DATASET_REV = "fe3ead447f44c0ea950396360b304cc2fb6be8f8"   # POSTCHECK's pinned revision
SUBSET_REPO = "allenai/MolmoAct2-LIBERO-depthopen"          # local-only id: LEROBOT_DATA_ROOT/<this>
S3_SUBSET = f"s3://{BUCKET}/depthopen/subset/v1"
S3_DEPTH = f"s3://{BUCKET}/depthopen/subset_depth/v1"
CAMERA_KEY = "observation.images.image"
MASTER_SEED = 20260909


def _hf_token() -> Optional[str]:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    r = subprocess.run(["aws", "ssm", "get-parameter", "--name", "/pg-lineage/hf-token",
                        "--with-decryption", "--region", REGION,
                        "--query", "Parameter.Value", "--output", "text"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        os.environ["HF_TOKEN"] = r.stdout.strip()
        return os.environ["HF_TOKEN"]
    return None


def _s3(*args: str) -> None:
    subprocess.run(["aws", "s3", *args, "--region", REGION], check=True)


def stage_enumerate(bank: Bank) -> List[str]:
    names = ds.enumerate_data_files(DATASET, DATASET_REV)
    bank.stage("enumerate", {"repo": DATASET, "revision": DATASET_REV, "n_data_files": len(names),
                             "first": names[0], "last": names[-1],
                             "why": "the Hub listing, checked contiguous; POSTCHECK's 378 was hand-typed"})
    print(f"[enumerate] {len(names)} data files, {names[0]} .. {names[-1]}", flush=True)
    return names


def stage_meta(bank: Bank, work: pathlib.Path) -> Dict[str, object]:
    """`meta/` at the pinned revision, into a source root the build reads from."""
    from huggingface_hub import hf_hub_download
    src = work / "source"
    src.mkdir(parents=True, exist_ok=True)
    got = {}
    for rel in (pcd.META_INFO, pcd.META_EPISODES, ds.META_TASKS, ds.META_STATS):
        p = hf_hub_download(DATASET, rel, repo_type="dataset", revision=DATASET_REV,
                            token=_hf_token(), local_dir=str(src))
        got[rel] = pathlib.Path(p).stat().st_size
    import pyarrow.parquet as pq
    info = json.loads((src / pcd.META_INFO).read_text())
    meta_rows = pq.read_table(src / pcd.META_EPISODES).to_pylist()
    tasks_rows = pq.read_table(src / ds.META_TASKS).to_pylist()
    bank.stage("meta", {"files": got, "total_episodes": info["total_episodes"],
                        "total_frames": info["total_frames"], "n_meta_rows": len(meta_rows),
                        "n_tasks": len(tasks_rows), "source_root": str(src)})
    return {"src": src, "info": info, "meta_rows": meta_rows, "tasks_rows": tasks_rows}


def stage_scan(bank: Bank, names: List[str], meta_rows: List[Dict[str, object]],
               info: Dict[str, object]) -> Dict[str, object]:
    """Every file's own `episode_index` / `index`, over HTTP range reads, and the corrected P4 counts."""
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=_hf_token())

    def opener(i: int):
        path = f"datasets/{DATASET}@{DATASET_REV}/" + names[i]
        return lambda: fs.open(path, "rb")

    t0 = time.perf_counter()

    def on_progress(p):
        if p["files_done"] % 40 == 0 or p["files_done"] == p["n_files"]:
            print(f"[scan] {p} {time.perf_counter() - t0:.0f}s", flush=True)
            bank.stage("scan_progress", dict(p, seconds=round(time.perf_counter() - t0, 1)))

    scan = pcd.scan_data_files(opener, len(names), on_progress=on_progress)
    comparison = pcd.compare(meta_rows, scan, info)
    keep = {k: v for k, v in comparison.items() if not isinstance(v, list)}
    bank.artifact("p4_recheck.json", comparison)
    bank.artifact("scan_per_file.json", scan["per_file"])
    bank.stage("scan", {"n_files": scan["n_files"], "n_episodes_present": scan["n_episodes_present"],
                        "n_rows_present": scan["n_rows_present"],
                        "seconds": round(time.perf_counter() - t0, 1),
                        "p4_recheck": keep,
                        "why": ("POSTCHECK measured this over 378 of 379 files from a hand-typed count; "
                                "these are the same four counts over the enumerated list")})
    print(f"[scan] {scan['n_episodes_present']} episodes, {scan['n_rows_present']} rows; "
          f"p4={json.dumps(keep)[:400]}", flush=True)
    return scan


def stage_select(bank: Bank, scan: Dict[str, object], meta_rows: List[Dict[str, object]],
                 tasks_rows: List[Dict[str, object]], per_suite_episodes: int,
                 seed: int, selection_s3: str = "", work: Optional[pathlib.Path] = None
                 ) -> Dict[str, object]:
    """The banked selection if one is named, else a fresh draw.

    The subset every arm trains on is a banked artifact, not a re-derivation: `selection.json` from the
    run that built it is the authoritative file list, and a later run that needs the same subset reads it
    rather than trusting that the draw reproduces.
    """
    if selection_s3:
        local = (work or pathlib.Path(".")) / "selection_in.json"
        _s3("cp", selection_s3, str(local))
        selection = json.loads(local.read_text())
        bank.stage("select", {"read_from": selection_s3, "n_files": selection["n_files"],
                              "n_episodes": selection["n_episodes"], "seed": selection.get("seed")})
        return selection
    episodes_in_file = ds.episodes_by_file(scan)
    suite_map = ds.load_suite_map()
    # the dataset's own task strings must all match the LIBERO task lists, or a stratum is unbalanced
    ds.suite_by_task_index({int(r["task_index"]): r["task"] for r in tasks_rows}, suite_map)
    episode_suite = ds.episode_suites(meta_rows, suite_map)
    f_suite = ds.file_suites(episodes_in_file, episode_suite)
    n_straddling = sum(1 for v in f_suite.values() if v is None)
    selection = ds.select_files(f_suite, episodes_in_file,
                                per_suite_episodes=per_suite_episodes, seed=seed)
    selection["n_files_straddling_suites"] = n_straddling
    selection["episodes_selected"] = sorted(e for f in selection["files"] for e in episodes_in_file[f])
    bank.artifact("selection.json", selection)
    bank.stage("select", {k: v for k, v in selection.items() if k != "episodes_selected"})
    print(f"[select] {selection['n_files']} files, {selection['n_episodes']} episodes; "
          f"per suite {json.dumps({s: v['n_episodes'] for s, v in selection['per_suite'].items()})}",
          flush=True)
    return selection


def stage_download(bank: Bank, work: pathlib.Path, names: List[str],
                   files: List[int]) -> Dict[str, object]:
    from huggingface_hub import snapshot_download
    src = work / "source"
    t0 = time.perf_counter()
    # snapshot_download with allow_patterns and max_workers: one call, parallel transfers, straight into
    # the layout the build reads. Attempt 1 fetched the same files one at a time and spent 40 minutes on
    # 8 GB; the files are ~92 MB each and the bottleneck was per-file, not bandwidth.
    snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REV, token=_hf_token(),
                      local_dir=str(src), allow_patterns=[names[i] for i in files],
                      max_workers=8)
    got = [src / names[i] for i in files]
    missing = [str(p) for p in got if not p.is_file()]
    if missing:
        raise RuntimeError(f"{len(missing)} selected files did not land: {missing[:3]}")
    total = sum(p.stat().st_size for p in got)
    out = {"n_files": len(files), "bytes": total, "gb": round(total / 1e9, 2),
           "seconds": round(time.perf_counter() - t0, 1),
           "mb_per_s": round(total / 1e6 / max(1e-9, time.perf_counter() - t0), 1)}
    print(f"[download] {out}", flush=True)
    bank.stage("download", out)
    return out


def stage_build(bank: Bank, work: pathlib.Path, ctx: Dict[str, object],
                selection: Dict[str, object]) -> Dict[str, object]:
    dst = work / "subset" / SUBSET_REPO
    dst.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def on_progress(p):
        print(f"[build] {p}", flush=True)
        bank.stage("build_progress", p)

    built = ds.build_subset(ctx["src"], dst, files=selection["files"], info=ctx["info"],
                            episodes_meta=ctx["meta_rows"], tasks_rows=ctx["tasks_rows"],
                            on_progress=on_progress)
    verified = ds.verify_subset(dst, src_root=ctx["src"], mapping=built["mapping"], n_byte_checks=8)
    bank.artifact("subset_mapping.json", built["mapping"])
    payload = {k: v for k, v in built.items() if k != "mapping"}
    payload["verified"] = verified
    payload["seconds"] = round(time.perf_counter() - t0, 1)
    payload["root"] = str(dst)
    payload["repo_id"] = SUBSET_REPO
    bank.stage("build", payload)
    print(f"[build] {payload['n_episodes']} episodes / {payload['n_frames']} frames verified", flush=True)
    # `root` LAST: payload carries a str root for the artifact, and `{"root": dst, **payload}` let the
    # string win, which is what killed attempt 1 one line into the companions stage
    return {**payload, "root": dst}


def stage_companions(bank: Bank, work: pathlib.Path, subset_root: pathlib.Path,
                     batch_size: int = 16, smoke_first: bool = True) -> Dict[str, object]:
    """The makers' generator, driven per file with the vendored module DEPTHANCHOR already validated."""
    from infra.pg import depthanchor_model as dam
    subset_root = pathlib.Path(subset_root)
    refs = dam.load_depth_refs("cuda")
    module = refs.module
    info = json.loads((subset_root / ds.META_INFO).read_text())
    out_root = work / "subset_depth" / SUBSET_REPO
    grid = (320 // 32) ** 2
    module.setup_output_dataset(subset_root, out_root, info, CAMERA_KEY, grid_size=grid,
                               vae_image_size=320, threshold=0.996, depth_input_size=518,
                               overwrite=True)
    files = module.iter_data_files(subset_root)
    print(f"[companions] {len(files)} files, out={out_root}", flush=True)
    rates = []
    t_all = time.perf_counter()
    for k, path in enumerate(files):
        t0 = time.perf_counter()
        out_path = out_root / path.relative_to(subset_root)
        module.process_parquet_file(
            parquet_path=path, output_path=out_path, dataset_root=subset_root, info=info,
            camera_key=CAMERA_KEY, camera_dtype="image", depth_model=refs.depth_model, vae=refs.vae,
            depth_input_size=518, vae_image_size=320, threshold=0.996, batch_size=batch_size,
            num_workers=2,
            device=refs.device, video_metadata=None, overwrite=True)
        import pyarrow.parquet as pq
        n_rows = pq.read_metadata(out_path).num_rows
        dt = time.perf_counter() - t0
        rates.append({"file": path.name, "rows": n_rows, "seconds": round(dt, 2),
                      "frames_per_s": round(n_rows / dt, 1)})
        if k == 0:
            bank.stage("companions_first_file", rates[0])
            print(f"[companions] first file: {rates[0]}", flush=True)
            if smoke_first:
                check = verify_companions(out_path)
                bank.stage("companions_first_check", check)
        if (k + 1) % 5 == 0 or k + 1 == len(files):
            print(f"[companions] {k + 1}/{len(files)} at {rates[-1]['frames_per_s']} f/s", flush=True)
            bank.stage("companions_progress", {"files": k + 1, "of": len(files),
                                               "seconds": round(time.perf_counter() - t_all, 1)})
    total_rows = sum(r["rows"] for r in rates)
    seconds = time.perf_counter() - t_all
    checks = verify_companions(*[out_root / p.relative_to(subset_root) for p in files[:3]])
    payload = {"n_files": len(files), "rows": total_rows, "seconds": round(seconds, 1),
               "frames_per_s": round(total_rows / seconds, 1),
               "projected_full_release_hours": round(273465 / max(total_rows / seconds, 1e-9) / 3600, 2),
               "per_file": rates, "checks": checks, "root": str(out_root)}
    bank.stage("companions", payload)
    return {**payload, "root": out_root}


def verify_companions(*paths: pathlib.Path) -> Dict[str, object]:
    """The three columns must be there, be the right width, and hold codes in range.

    A silently short buffer or an out-of-range code would train the model on a prefix the released
    checkpoint's codebook cannot represent, and nothing downstream would say so.
    """
    import pyarrow.parquet as pq
    out = []
    for path in paths:
        t = pq.read_table(path, columns=["episode_index", "frame_index", "depth_codes",
                                         "buffer_codes", "depth_updated_mask"])
        codes = np.asarray(t.column("depth_codes").to_pylist(), dtype=np.int64)
        buf = np.asarray(t.column("buffer_codes").to_pylist(), dtype=np.int64)
        mask = np.asarray(t.column("depth_updated_mask").to_pylist(), dtype=bool)
        fi = np.asarray(t.column("frame_index").to_pylist(), dtype=np.int64)
        if codes.shape[1] != 100 or buf.shape[1] != 100 or mask.shape[1] != 100:
            raise RuntimeError(f"{path.name}: widths {codes.shape} {buf.shape} {mask.shape}, want 100")
        if codes.min() < 0 or codes.max() > 127 or buf.min() < 0 or buf.max() > 127:
            raise RuntimeError(f"{path.name}: codes out of [0,127]: {codes.min()}..{codes.max()}")
        first = np.flatnonzero(fi == 0)
        out.append({"file": path.name, "rows": int(codes.shape[0]),
                    "codes_range": [int(codes.min()), int(codes.max())],
                    "buffer_equals_codes_on_first_frame":
                        bool(np.array_equal(codes[first], buf[first])) if first.size else None,
                    "updated_cells_first_frames": [int(m.sum()) for m in mask[first][:3]],
                    "updated_cells_median_all": int(np.median(mask.sum(axis=1)))})
    return {"files": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthopen")
    ap.add_argument("--per-suite-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=MASTER_SEED)
    ap.add_argument("--depth-batch", type=int, default=16,
                    help="DepthAnythingV2 batch; 16 fits an A10G at 518 with room to spare")
    ap.add_argument("--selection-s3", default="",
                    help="read the banked selection.json instead of drawing: the subset every arm trains "
                         "on is an artifact, not a re-derivation")
    ap.add_argument("--smoke-files", type=int, default=0,
                    help="keep only this many selected files: the whole path on a few files first")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = Bank(pathlib.Path(args.out), "depthopen_data")
    bank.top("args", vars(args))
    bank.top("pins", {"dataset": DATASET, "revision": DATASET_REV, "subset_repo": SUBSET_REPO,
                      "camera_key": CAMERA_KEY, "seed": args.seed})

    names = stage_enumerate(bank)
    ctx = stage_meta(bank, work)
    scan = stage_scan(bank, names, ctx["meta_rows"], ctx["info"])
    selection = stage_select(bank, scan, ctx["meta_rows"], ctx["tasks_rows"],
                             args.per_suite_episodes, args.seed,
                             selection_s3=args.selection_s3, work=work)
    files = list(selection["files"])
    if args.smoke_files:
        files = files[:args.smoke_files]
        bank.stage("smoke", {"files": files, "why": "the whole path on a few files before the full run"})
    stage_download(bank, work, names, files)
    selection_for_build = dict(selection, files=files)
    built = stage_build(bank, work, ctx, selection_for_build)
    companions = stage_companions(bank, work, built["root"], batch_size=args.depth_batch)

    uploaded = {}
    if not args.skip_upload:
        for label, root, uri in (("subset", built["root"], f"{S3_SUBSET}/{SUBSET_REPO}"),
                                 ("companions", companions["root"], f"{S3_DEPTH}/{SUBSET_REPO}")):
            t0 = time.perf_counter()
            _s3("sync", str(root), uri)
            uploaded[label] = {"uri": uri, "seconds": round(time.perf_counter() - t0, 1)}
            print(f"[upload] {label} -> {uri} in {uploaded[label]['seconds']}s", flush=True)
        marker = work / "DONE"
        marker.write_text(json.dumps({"subset": f"{S3_SUBSET}/{SUBSET_REPO}",
                                      "companions": f"{S3_DEPTH}/{SUBSET_REPO}",
                                      "n_episodes": built["n_episodes"],
                                      "n_frames": built["n_frames"]}, indent=1))
        _s3("cp", str(marker), f"{S3_SUBSET}/DONE")
        bank.stage("upload", uploaded)

    bank.top("verdict", {
        "n_data_files_enumerated": len(names),
        "p4_recheck": bank.payload["stages"]["scan"]["p4_recheck"],
        "subset": {"repo_id": SUBSET_REPO, "n_files": built["n_files"],
                   "n_episodes": built["n_episodes"], "n_frames": built["n_frames"],
                   "per_suite": {s: v["n_episodes"] for s, v in selection["per_suite"].items()}},
        "companions": {k: companions[k] for k in ("n_files", "rows", "frames_per_s",
                                                  "projected_full_release_hours")},
        "uploaded": uploaded,
    })
    print(json.dumps(bank.payload["verdict"], indent=1, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
