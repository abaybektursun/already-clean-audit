"""Hydrate one exact versioned S3 model snapshot on H100 scratch without loading it."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, urlsplit


class HydrationError(RuntimeError):
    pass


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise HydrationError("hydration spec is not frozen")
    if spec.get("scope") != "s3_worker_cache_hydration_only":
        raise HydrationError("hydration scope changed")
    files = spec.get("files")
    if not isinstance(files, list) or len(files) != 12:
        raise HydrationError("hydration must bind all twelve PG1 files")
    if int(spec.get("expected_total_bytes", -1)) != 11_715_866_017:
        raise HydrationError("hydration total changed")
    if sum(int(row.get("bytes", -1)) for row in files) != 11_715_866_017:
        raise HydrationError("hydration file bytes do not sum exactly")
    if int(spec.get("parallel_streams", 0)) != 3:
        raise HydrationError("hydration must use exactly three streams")
    if not 300 <= int(spec.get("runtime_timeout_seconds", 0)) <= 600:
        raise HydrationError("hydration runtime must remain bounded in [300, 600] seconds")
    expected_names = {
        "added_tokens.json", "config.json", "generation_config.json",
        "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors", "model.safetensors.index.json",
        "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json",
    }
    if {str(row.get("path")) for row in files} != expected_names:
        raise HydrationError("hydration PG1 file set changed")
    for row in files:
        digest = str(row.get("sha256", ""))
        if (
            row.get("bucket") != "pg-lineage-769063704696"
            or row.get("region") != "us-east-1"
            or not str(row.get("key", "")).startswith(
                f"parallax/input-staging/sha256/{digest}/"
            )
            or not str(row.get("version_id", ""))
            or int(row.get("bytes", 0)) <= 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise HydrationError("hydration source is not exact and versioned")
    destination = PurePosixPath(str(spec.get("worker_snapshot_root", "")))
    if not destination.is_absolute() or "snapshots" not in destination.parts:
        raise HydrationError("hydration destination is not an exact snapshot root")
    if any(int(spec.get(field, -1)) != 0 for field in (
        "confirmation_items_read", "science_items_evaluated", "model_loads", "gpu_work"
    )):
        raise HydrationError("hydration cannot read science or use a GPU")
    runtime = spec.get("runtime", {}).get("job", {})
    secret = f"/tmp/parallax-lifecycle-{spec.get('run_id')}/s3-get-urls.secret.json"
    if (
        runtime.get("execution_site") != "gateway"
        or runtime.get("environment", {}).get("CUDA_VISIBLE_DEVICES") != ""
        or secret not in runtime.get("command", [])
    ):
        raise HydrationError("hydration runtime boundary changed")
    return spec


def validate_url(url: str, row: dict[str, Any]) -> str:
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    if (
        parsed.scheme != "https"
        or parsed.netloc != f"{row['bucket']}.s3.{row['region']}.amazonaws.com"
        or parsed.path != "/" + row["key"]
        or query.get("versionId") != [row["version_id"]]
        or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
        or query.get("X-Amz-SignedHeaders") != ["host"]
        or not 60 <= int(query.get("X-Amz-Expires", ["0"])[0]) <= 1800
        or len(query.get("X-Amz-Signature", [""])[0]) != 64
    ):
        raise HydrationError("hydration URL is not exact, version-bound, and presigned")
    return url


REMOTE_HYDRATE = r'''import concurrent.futures,hashlib,json,os,pathlib,shutil,sys,time,urllib.request
payload=json.load(sys.stdin); destination=pathlib.Path(sys.argv[1]); streams=int(sys.argv[2])
if destination.exists(): raise SystemExit('destination already exists')
destination.parent.mkdir(parents=True,exist_ok=True); partial=destination.parent/(destination.name+'.partial')
if partial.exists(): shutil.rmtree(partial)
partial.mkdir()
def one(row):
 target=partial/row['path']; target.parent.mkdir(parents=True,exist_ok=True); temp=target.with_name(target.name+'.part')
 digest=hashlib.sha256(); size=0; started=time.monotonic()
 with urllib.request.urlopen(row['url'],timeout=120) as response, temp.open('xb') as handle:
  while True:
   chunk=response.read(8*1024*1024)
   if not chunk: break
   handle.write(chunk); digest.update(chunk); size+=len(chunk)
  handle.flush(); os.fsync(handle.fileno())
 match=size==row['bytes'] and digest.hexdigest()==row['sha256']
 if not match: raise RuntimeError('mismatch:'+row['path'])
 os.replace(temp,target)
 return {'path':row['path'],'bytes':size,'sha256':digest.hexdigest(),'version_id':row['version_id'],'elapsed_seconds':time.monotonic()-started,'status':'GREEN'}
started=time.monotonic()
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool: rows=list(pool.map(one,payload))
 os.sync(); os.replace(partial,destination)
 result={'status':'GREEN_HYDRATED','elapsed_seconds':time.monotonic()-started,'file_count':len(rows),'verified_bytes':sum(x['bytes'] for x in rows),'files':rows,'snapshot_root':str(destination)}
except Exception:
 shutil.rmtree(partial,ignore_errors=True); raise
print(json.dumps(result,sort_keys=True))'''


def run(
    spec: dict[str, Any], *, url_file: Path, worker_host: str, worker_port: int,
    worker_key: Path, worker_result_root: str,
) -> dict[str, Any]:
    validate_spec(spec)
    secret = json.loads(url_file.read_text(encoding="utf-8"))
    url_file.unlink(missing_ok=True)
    if not isinstance(secret, list) or len(secret) != len(spec["files"]):
        raise HydrationError("hydration URL secret is incomplete")
    by_path = {str(row.get("path")): row for row in secret}
    payload = []
    for row in spec["files"]:
        supplied = by_path.get(row["path"], {})
        payload.append({**row, "url": validate_url(str(supplied.get("url", "")), row)})
    ssh = [
        "ssh", "-i", str(worker_key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10", "-p", str(worker_port), f"root@{worker_host}",
    ]
    command = (
        "CUDA_VISIBLE_DEVICES='' python3 -c " + shlex.quote(REMOTE_HYDRATE) + " "
        + shlex.quote(spec["worker_snapshot_root"]) + " "
        + shlex.quote(str(spec["parallel_streams"]))
    )
    completed = subprocess.run(
        [*ssh, command], input=(json.dumps(payload) + "\n").encode(), capture_output=True,
        timeout=int(spec["runtime_timeout_seconds"]),
    )
    payload.clear()
    if completed.returncode != 0:
        raise HydrationError("worker hydration command failed")
    result = json.loads(completed.stdout.decode("utf-8"))
    result.update({
        "schema_version": 1,
        "run_id": spec["run_id"],
        "mode": "NO_SCIENCE_S3_CACHE_HYDRATION",
        "url_file_deleted_before_download": True,
        "url_persisted": False,
        "aws_credentials_present_on_worker": False,
        "automatic_retry": False,
        "gpu_work": 0,
        "model_loads": 0,
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
    })
    if (
        result.get("status") != "GREEN_HYDRATED"
        or int(result.get("file_count", -1)) != 12
        or int(result.get("verified_bytes", -1)) != 11_715_866_017
    ):
        raise HydrationError("worker hydration result is not exact")
    root = worker_result_root.rstrip("/")
    publish = (
        f"umask 077; mkdir -p {shlex.quote(root)}; test ! -e {shlex.quote(root + '/result.json')}; "
        f"cat > {shlex.quote(root + '/.result.json.partial')}; "
        f"sync {shlex.quote(root + '/.result.json.partial')}; "
        f"mv {shlex.quote(root + '/.result.json.partial')} {shlex.quote(root + '/result.json')}"
    )
    published = subprocess.run(
        [*ssh, publish], input=(json.dumps(result, sort_keys=True) + "\n").encode(),
        capture_output=True, timeout=30,
    )
    if published.returncode != 0:
        raise HydrationError("could not atomically publish hydration result")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--worker-key", type=Path, required=True)
    parser.add_argument("--worker-result-root", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    print(json.dumps(run(
        spec, url_file=args.url_file, worker_host=args.worker_host,
        worker_port=args.worker_port, worker_key=args.worker_key,
        worker_result_root=args.worker_result_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
