"""Shared job plumbing for DEPTHANCHOR: S3 paths, atomic banking, and the frame-pack round trip.

Every job writes `out/metrics.json` after EVERY stage through `Bank`, with a temp file and
`os.replace`, because `run_job.sh` syncs `out/` to S3 every 30 s and a job killed by the reaper must
leave the stages it finished readable (skill gotcha: save inside the loop, not after it).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from typing import Dict, Sequence

BUCKET = "pg-lineage-769063704696"
S3_ROOT = f"s3://{BUCKET}/depthanchor"
S3_LIBERODS = f"{S3_ROOT}/liberods/v1"
S3_PACK = f"{S3_ROOT}/frames/v1"
S3_CAMERA = f"{S3_ROOT}/camera/v1"
S3_LADDER = f"{S3_ROOT}/ladder/v1"
S3_GATES = f"{S3_ROOT}/gates/v1"
REGION = "us-east-1"


def s3(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["aws", "s3", *args, "--region", REGION]
    print(f"[s3] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=check)


def s3_exists(uri: str) -> bool:
    return subprocess.run(["aws", "s3", "ls", uri, "--region", REGION],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


class Bank:
    """One growing `metrics.json` plus per-stage side artifacts."""

    def __init__(self, out: pathlib.Path, cell: str):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.payload: Dict[str, object] = {"cell": cell, "started_utc": _utc(), "stages": {}}
        self.flush()

    def stage(self, name: str, payload) -> None:
        self.payload["stages"][name] = payload
        self.payload["updated_utc"] = _utc()
        self.flush()
        print(f"[bank] stage {name} banked", flush=True)

    def top(self, key: str, value) -> None:
        self.payload[key] = value
        self.flush()

    def artifact(self, name: str, payload) -> pathlib.Path:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=_default))
        os.replace(tmp, path)
        return path

    def flush(self) -> None:
        tmp = self.out / "metrics.json.tmp"
        tmp.write_text(json.dumps(self.payload, indent=1, default=_default))
        os.replace(tmp, self.out / "metrics.json")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


# ------------------------------------------------------------------ the frame pack round trip
def pull_pack(dest: pathlib.Path) -> pathlib.Path:
    """Fetch the banked frame pack. Every cell after the preflight reads the SAME pack, so the frame
    list, the target cells and the background draws cannot drift between cells."""
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not s3_exists(f"{S3_PACK}/frames_index.json"):
        raise FileNotFoundError(f"no frame pack at {S3_PACK}; run depthanchor_preflight first")
    s3("cp", f"{S3_PACK}/frames.npz", str(dest / "frames.npz"))
    s3("cp", f"{S3_PACK}/frames_index.json", str(dest / "frames_index.json"))
    return dest


def push_pack(src: pathlib.Path) -> None:
    src = pathlib.Path(src)
    s3("cp", str(src / "frames.npz"), f"{S3_PACK}/frames.npz")
    s3("cp", str(src / "frames_index.json"), f"{S3_PACK}/frames_index.json")


def pull_camera_bank(dest: pathlib.Path) -> pathlib.Path:
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "camera_bank.json"
    if not s3_exists(f"{S3_CAMERA}/camera_bank.json"):
        raise FileNotFoundError(f"no camera bank at {S3_CAMERA}; run depthanchor_preflight first")
    s3("cp", f"{S3_CAMERA}/camera_bank.json", str(path))
    return path


def push_camera_bank(path: pathlib.Path) -> None:
    s3("cp", str(path), f"{S3_CAMERA}/camera_bank.json")


def pull_dataset_files(dest: pathlib.Path, relative_paths: Sequence[str]) -> pathlib.Path:
    """Only the parquet files the frozen frame list needs, from the seeded plain-path tree."""
    dest = pathlib.Path(dest)
    for rel in sorted(set(relative_paths)):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file():
            continue
        s3("cp", f"{S3_LIBERODS}/{rel}", str(target))
    return dest


def pull_dataset_meta(dest: pathlib.Path) -> pathlib.Path:
    dest = pathlib.Path(dest)
    (dest / "meta").mkdir(parents=True, exist_ok=True)
    s3("sync", f"{S3_LIBERODS}/meta/", str(dest / "meta"))
    return dest / "meta"


def push_dir(src: pathlib.Path, uri: str) -> None:
    s3("sync", str(src), uri)


def push_json(path: pathlib.Path, uri: str) -> None:
    s3("cp", str(path), uri)


def pull_json(uri: str, dest: pathlib.Path) -> pathlib.Path:
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not s3_exists(uri):
        raise FileNotFoundError(f"missing banked artifact {uri}")
    s3("cp", uri, str(dest))
    return dest


def load_ladder(dest: pathlib.Path):
    """The code ladder is an INPUT to every edit, so it is banked once and reused, never recomputed:
    a different `d(c)` would mean a different "+8 ranks" and the sessions would not be comparable."""
    from infra.pg import depthanchor_core as core
    path = pull_json(f"{S3_LADDER}/code_ladder.json", pathlib.Path(dest) / "code_ladder.json")
    payload = json.loads(path.read_text())
    return core.CodeLadder(d_primary=payload["d_primary"], d_local=payload["d_local"]), payload
