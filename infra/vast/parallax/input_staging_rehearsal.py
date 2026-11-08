"""Prepare and stage the exact PG1 first-cell inputs without running science."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .artifacts import atomic_json, sha256_file


class StagingError(RuntimeError):
    pass


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_staging_job_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise StagingError("input staging job is not frozen")
    if spec.get("scope") != "pg1_input_staging_rehearsal_only":
        raise StagingError("input staging scope is invalid")
    run_id = str(spec.get("run_id", ""))
    if not run_id or re.fullmatch(r"[a-z0-9_-]+", run_id) is None:
        raise StagingError("input staging run id is unsafe")
    if spec.get("uses_gpu") is not False or spec.get("scientific_claim") is not False:
        raise StagingError("input staging cannot use GPU work or make a scientific claim")
    if int(spec.get("model_reads", -1)) != 0 or int(spec.get("science_data_reads", -1)) != 0:
        raise StagingError("input staging cannot load a model or read science examples")
    runtime = spec.get("runtime", {})
    job = runtime.get("job", {})
    if (
        job.get("execution_site") != "gateway"
        or job.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != ""
        or "infra.vast.parallax.input_staging_rehearsal" not in job.get("command", [])
        or runtime.get("required_sentinels") != ["input-staging/result.json"]
        or runtime.get("artifact_roots") != [
            {"remote_root": "/tmp/parallax-input-staging-v1", "durable_prefix": "input-staging"}
        ]
    ):
        raise StagingError("input staging runtime is not the frozen gateway-only command")
    staging = spec.get("staging")
    if not isinstance(staging, dict):
        raise StagingError("input staging contract is missing")
    for key in ("task_source_root", "task_subset_root", "model_source_root", "worker_root"):
        path = PurePosixPath(str(staging.get(key, "")))
        if not path.is_absolute() or ".." in path.parts:
            raise StagingError(f"unsafe staging path: {key}")
        if any(re.fullmatch(r"[A-Za-z0-9._-]+", part) is None for part in path.parts if part != "/"):
            raise StagingError(f"staging path contains unsafe characters: {key}")
    for key in ("task_source_root", "task_subset_root", "model_source_root"):
        if not PurePosixPath(str(staging[key])).is_relative_to(PurePosixPath("/volume/parallax")):
            raise StagingError(f"persistent source escapes PARALLAX volume: {key}")
    if not PurePosixPath(str(staging["worker_root"])).is_relative_to(PurePosixPath("/root")):
        raise StagingError("worker root must be disposable /root scratch")
    expected_task = staging.get("expected_task", {})
    for key in ("content_sha256", "audit_sha256", "items_jsonl_sha256"):
        digest = str(expected_task.get(key, ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise StagingError(f"invalid task digest: {key}")
    panel = staging.get("panel", {})
    if panel != {
        "task_id": "vr_a_beans_balanced",
        "train_per_target": 32,
        "discovery_per_target": 8,
    }:
        raise StagingError("first-cell task panel is not exact")
    model = staging.get("model", {})
    files = model.get("files")
    if model.get("repo") != "google/paligemma-3b-pt-224" or not isinstance(files, list) or len(files) != 12:
        raise StagingError("PG1 model subset is not exact")
    seen: set[str] = set()
    total = 0
    for row in files:
        relative = PurePosixPath(str(row.get("path", "")))
        digest = str(row.get("sha256", ""))
        size = int(row.get("bytes", -1))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in seen:
            raise StagingError("unsafe or duplicate PG1 model path")
        if size < 0 or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise StagingError("malformed PG1 model binding")
        seen.add(relative.as_posix())
        total += size
    if total != int(model.get("total_bytes", -1)) or total != 11_715_866_017:
        raise StagingError("PG1 model byte total is not exact")
    if not 12_000_000_000 <= int(staging.get("minimum_worker_free_bytes", 0)) <= 20_000_000_000:
        raise StagingError("worker free-space gate is outside the frozen range")
    if not total < int(staging.get("maximum_transfer_bytes", 0)) <= 12_000_000_000:
        raise StagingError("transfer byte ceiling is invalid")
    if not 300 <= int(staging.get("transfer_timeout_seconds", 0)) <= 2400:
        raise StagingError("transfer timeout is invalid")
    return spec


def _write_lines_atomic(path: Path, lines: list[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".partial")
    with temporary.open("wb") as handle:
        for line in lines:
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _select(rows: list[dict[str, Any]], *, task_id: str, split: str, per_target: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("task_id") == task_id and row.get("split") == split:
            grouped[str(row["target"])].append(row)
    if not grouped:
        raise StagingError(f"no rows for {task_id}:{split}")
    selected: list[dict[str, Any]] = []
    for target, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: str(row["item_id"]))
        if len(ordered) < per_target:
            raise StagingError(f"underfilled target {target}:{split}")
        selected.extend(ordered[:per_target])
    selected.sort(key=lambda row: (str(row["target"]), str(row["item_id"])))
    return selected


def _verify_exact_files(root: Path, bindings: list[dict[str, Any]]) -> tuple[int, int]:
    expected = {str(row["path"]): row for row in bindings}
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    if actual != sorted(expected):
        raise StagingError("exact file-set verification failed")
    total = 0
    for relative, row in expected.items():
        path = root / relative
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise StagingError(f"file binding mismatch: {relative}")
        total += path.stat().st_size
    return len(actual), total


def prepare_task_subset(spec: dict[str, Any]) -> dict[str, Any]:
    staging = spec["staging"]
    source = Path(staging["task_source_root"])
    final = Path(staging["task_subset_root"])
    expected = staging["expected_task"]
    manifests = source / "manifests"
    for name, key in (("content.json", "content_sha256"), ("audit.json", "audit_sha256"), ("items.jsonl", "items_jsonl_sha256")):
        if sha256_file(manifests / name) != expected[key]:
            raise StagingError(f"cloud task source mismatch: {name}")
    audit = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "GREEN" or audit.get("errors"):
        raise StagingError("cloud task source audit is not green")

    if final.exists():
        package = json.loads((final / "manifests/package.json").read_text(encoding="utf-8"))
        package_path = final / "manifests/package.json"
        package_binding = {
            "path": "manifests/package.json",
            "bytes": package_path.stat().st_size,
            "sha256": sha256_file(package_path),
        }
        package["files_with_package"] = [*package["files"], package_binding]
        _verify_exact_files(final, package["files_with_package"])
        if package.get("source_items_jsonl_sha256") != expected["items_jsonl_sha256"]:
            raise StagingError("existing task subset belongs to another source")
        return {**package, "reused_existing": True}

    temporary = final.with_name("." + final.name + ".materializing")
    if temporary.exists():
        raise StagingError("task subset staging path already exists")
    temporary.mkdir(parents=True)
    try:
        raw_discovery: list[bytes] = []
        rows: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        with (manifests / "items.jsonl").open("rb") as handle:
            for raw in handle:
                # Route the sealed split from the canonical compact JSON bytes
                # without deserializing any confirmation target or prompt.
                match = re.search(rb'"split":"([^"]+)"', raw)
                if match is None:
                    raise StagingError("task row lacks canonical split routing field")
                split = match.group(1).decode("ascii")
                counts[split] += 1
                if split in {"train", "discovery_eval"}:
                    raw_discovery.append(raw)
                    rows.append(json.loads(raw))
        if counts != Counter({"train": 2048, "discovery_eval": 1024, "confirmation_eval": 1024}):
            raise StagingError("task split counts differ from freeze")
        discovery_path = temporary / "manifests/items.discovery.jsonl"
        _write_lines_atomic(discovery_path, raw_discovery)
        discovery_hash = sha256_file(discovery_path)
        seal = {
            "schema_version": 1,
            "status": "GREEN",
            "source_manifest_sha256": expected["items_jsonl_sha256"],
            "discovery_manifest_sha256": discovery_hash,
            "source_rows": 4096,
            "discovery_rows": 3072,
            "routed_counts": dict(sorted(counts.items())),
            "confirmation_values_emitted": 0,
            "policy": "training_runtimes_must_open_only_items.discovery.jsonl",
        }
        atomic_json(temporary / "manifests/discovery-seal.json", seal)
        panel = staging["panel"]
        selected = _select(rows, task_id=panel["task_id"], split="train", per_target=panel["train_per_target"])
        selected += _select(rows, task_id=panel["task_id"], split="discovery_eval", per_target=panel["discovery_per_target"])
        if len(selected) != 120 or any(row["split"] == "confirmation_eval" for row in selected):
            raise StagingError("first-cell panel is not the exact 120 discovery-safe images")
        selected_bindings = []
        for row in selected:
            relative = str(row["image"])
            if "confirmation_eval" in relative:
                raise StagingError("confirmation image entered first-cell panel")
            source_image = source / relative
            if source_image.stat().st_size != int(row["image_bytes"]) or sha256_file(source_image) != row["image_sha256"]:
                raise StagingError(f"selected source image mismatch: {row['item_id']}")
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_image, destination)
            selected_bindings.append({
                "item_id": row["item_id"], "split": row["split"], "target": row["target"],
                "path": relative, "bytes": destination.stat().st_size, "sha256": sha256_file(destination),
            })
        atomic_json(temporary / "manifests/selected-images.json", {"schema_version": 1, "files": selected_bindings})
        payload_files = []
        for path in sorted(p for p in temporary.rglob("*") if p.is_file() and p.name != "package.json"):
            payload_files.append({
                "path": path.relative_to(temporary).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        if any("confirmation" in row["path"] or row["path"].endswith("items.jsonl") for row in payload_files):
            raise StagingError("confirmation-capable artifact entered first-cell subset")
        package = {
            "schema_version": 1,
            "status": "GREEN",
            "source_items_jsonl_sha256": expected["items_jsonl_sha256"],
            "discovery_manifest_sha256": discovery_hash,
            "discovery_seal_sha256": sha256_file(temporary / "manifests/discovery-seal.json"),
            "selected_image_count": 120,
            "confirmation_rows_emitted": 0,
            "confirmation_capable_artifacts_present": False,
            "files": payload_files,
        }
        atomic_json(temporary / "manifests/package.json", package)
        package_binding = {
            "path": "manifests/package.json",
            "bytes": (temporary / "manifests/package.json").stat().st_size,
            "sha256": sha256_file(temporary / "manifests/package.json"),
        }
        package["files_with_package"] = [*payload_files, package_binding]
        _verify_exact_files(temporary, package["files_with_package"])
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final)
        os.sync()
        return {**package, "reused_existing": False}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_model_source(spec: dict[str, Any]) -> dict[str, Any]:
    staging = spec["staging"]
    root = Path(staging["model_source_root"])
    files = staging["model"]["files"]
    verified_files = 0
    verified_bytes = 0
    for row in files:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise StagingError(f"PG1 model source mismatch: {row['path']}")
        verified_files += 1
        verified_bytes += path.stat().st_size
    return {"verified_files": verified_files, "verified_bytes": verified_bytes}


def _ssh_base(host: str, port: int, key: Path) -> list[str]:
    return [
        "ssh", "-i", str(key), "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10", "-p", str(port), f"root@{host}",
    ]


def _publish_result(ssh: list[str], remote_root: str, result: dict[str, Any]) -> None:
    script = """import os,pathlib,sys
root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
tmp=root/'.result.json.partial';dst=root/'result.json'
with tmp.open('wb') as h:h.write(sys.stdin.buffer.read());h.flush();os.fsync(h.fileno())
os.replace(tmp,dst)"""
    completed = subprocess.run(
        [*ssh, shlex.join(["python3", "-c", script, remote_root])],
        input=(json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise StagingError("could not publish staging result to worker")


def _rsync(source: Path, destination: str, *, host: str, port: int, key: Path, timeout: int) -> dict[str, Any]:
    transport = (
        f"ssh -i {shlex.quote(str(key))} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "
        f"-o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p {port}"
    )
    started = time.monotonic()
    completed = subprocess.run(
        ["rsync", "-aL", "--delete", "--partial", "--stats", "-e", transport, str(source) + "/", f"root@{host}:{destination}/"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-2000:],
    }


REMOTE_VERIFY = r"""import hashlib,json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); expected=json.loads(sys.stdin.read()); errors=[]; total=0; count=0
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while True:
   b=f.read(8*1024*1024)
   if not b:break
   h.update(b)
 return h.hexdigest()
for group in ('task','model'):
 base=root/group; wanted={r['path']:r for r in expected[group]}
 actual=sorted(p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file())
 if actual!=sorted(wanted):errors.append(group+':file_set')
 for rel,row in wanted.items():
  p=base/rel
  if not p.is_file():errors.append(group+':missing:'+rel);continue
  size=p.stat().st_size; total+=size; count+=1
  if size!=int(row['bytes']):errors.append(group+':size:'+rel)
  elif digest(p)!=row['sha256']:errors.append(group+':sha256:'+rel)
print(json.dumps({'status':'GREEN' if not errors else 'RED','errors':errors,'verified_files':count,'verified_bytes':total,'worker_free_bytes_after':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize},sort_keys=True))"""


def run(spec: dict[str, Any], *, worker_host: str, worker_port: int, worker_key: Path, worker_result_root: str) -> dict[str, Any]:
    validate_staging_job_spec(spec)
    started = time.monotonic()
    ssh = _ssh_base(worker_host, worker_port, worker_key)
    task = prepare_task_subset(spec)
    model = verify_model_source(spec)
    staging = spec["staging"]
    worker_root = str(staging["worker_root"])
    required_free = int(staging["minimum_worker_free_bytes"])
    preflight = subprocess.run(
        [*ssh, f"test ! -e {worker_root} && mkdir -p {worker_root}/task {worker_root}/model && df --output=avail -B1 /root | tail -1"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if preflight.returncode != 0:
        raise StagingError("worker scratch preflight failed or destination already exists")
    free_before = int(preflight.stdout.strip().splitlines()[-1])
    if free_before < required_free:
        raise StagingError(f"worker scratch below frozen free-space gate: {free_before}")
    task_transfer = _rsync(Path(staging["task_subset_root"]), worker_root + "/task", host=worker_host, port=worker_port, key=worker_key, timeout=int(staging["transfer_timeout_seconds"]))
    if task_transfer["returncode"] != 0:
        raise StagingError("task subset transfer failed: " + task_transfer["stderr_tail"][-500:])
    model_transfer = _rsync(Path(staging["model_source_root"]), worker_root + "/model", host=worker_host, port=worker_port, key=worker_key, timeout=int(staging["transfer_timeout_seconds"]))
    if model_transfer["returncode"] != 0:
        raise StagingError("PG1 model transfer failed: " + model_transfer["stderr_tail"][-500:])
    expected = {"task": task["files_with_package"], "model": staging["model"]["files"]}
    verification = subprocess.run(
        [*ssh, shlex.join(["python3", "-c", REMOTE_VERIFY, worker_root])],
        input=json.dumps(expected, separators=(",", ":")), capture_output=True, text=True,
        timeout=int(staging["transfer_timeout_seconds"]), check=False,
    )
    if verification.returncode != 0:
        raise StagingError("worker verification command failed: " + verification.stderr[-500:])
    verified = json.loads(verification.stdout)
    if verified.get("status") != "GREEN":
        raise StagingError("worker exact-byte verification failed: " + repr(verified.get("errors", [])[:8]))
    total_seconds = task_transfer["elapsed_seconds"] + model_transfer["elapsed_seconds"]
    packet = {
        "schema_version": 1,
        "status": "GREEN",
        "run_id": spec["run_id"],
        "created_utc": _utc(),
        "mode": "NO_SCIENCE_PG1_INPUT_STAGING_REHEARSAL",
        "task_subset": {
            "persistent_root": staging["task_subset_root"],
            "source_items_jsonl_sha256": task["source_items_jsonl_sha256"],
            "discovery_manifest_sha256": task["discovery_manifest_sha256"],
            "discovery_seal_sha256": task["discovery_seal_sha256"],
            "selected_image_count": task["selected_image_count"],
            "confirmation_capable_artifacts_present": False,
            "reused_existing": task["reused_existing"],
        },
        "model": {**model, "repo": staging["model"]["repo"], "revision": staging["model"]["revision"]},
        "worker": {
            "root": worker_root,
            "free_bytes_before": free_before,
            "free_bytes_after": verified["worker_free_bytes_after"],
            "verified_files": verified["verified_files"],
            "verified_bytes": verified["verified_bytes"],
        },
        "transfer": {
            "task": task_transfer,
            "model": model_transfer,
            "payload_seconds": total_seconds,
            "payload_mib_per_second": verified["verified_bytes"] / (1024 * 1024) / total_seconds,
        },
        "elapsed_seconds": time.monotonic() - started,
        "gpu_work": False,
        "model_loaded": False,
        "science_items_evaluated": 0,
        "confirmation_items_read": 0,
        "network_downloads": 0,
        "promotion_boundary": "STAGING_MECHANICS_ONLY_NO_H100_NO_MODEL_LOAD_NO_SCIENCE",
    }
    _publish_result(ssh, worker_result_root, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--worker-key", type=Path, required=True)
    parser.add_argument("--worker-result-root", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    try:
        result = run(
            spec,
            worker_host=args.worker_host,
            worker_port=args.worker_port,
            worker_key=args.worker_key,
            worker_result_root=args.worker_result_root,
        )
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "RED",
            "run_id": spec.get("run_id"),
            "created_utc": _utc(),
            "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            "gpu_work": False,
            "model_loaded": False,
            "science_items_evaluated": 0,
            "confirmation_items_read": 0,
        }
        ssh = _ssh_base(args.worker_host, args.worker_port, args.worker_key)
        _publish_result(ssh, args.worker_result_root, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
