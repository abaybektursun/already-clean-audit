"""Measure exact S3-to-worker staging throughput without loading a model or using a GPU."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


class DownloadProbeError(RuntimeError):
    pass


def validate_download_probe_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != 1 or spec.get("status") != "FROZEN":
        raise DownloadProbeError("download probe spec is not frozen")
    if spec.get("scope") != "s3_worker_download_probe_only":
        raise DownloadProbeError("download probe scope changed")
    source = spec.get("source", {})
    digest = str(source.get("sha256", ""))
    if (
        source.get("bucket") != "pg-lineage-769063704696"
        or source.get("region") != "us-east-1"
        or not str(source.get("key", "")).startswith("parallax/input-staging/sha256/")
        or not str(source.get("version_id", ""))
        or int(source.get("bytes", 0)) != 17_549_604
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise DownloadProbeError("download source is not the exact qualified object")
    if int(spec.get("repetitions", 0)) != 48 or int(spec.get("parallel_streams", 0)) != 3:
        raise DownloadProbeError("download probe must emulate three shard streams over 48 copies")
    total = int(source["bytes"]) * int(spec["repetitions"])
    if total != int(spec.get("maximum_download_bytes", -1)) or total != 842_380_992:
        raise DownloadProbeError("download byte ceiling changed")
    if int(spec.get("projected_full_cache_bytes", 0)) != 11_715_866_017:
        raise DownloadProbeError("full-cache projection target changed")
    if int(spec.get("maximum_projected_full_cache_seconds", 0)) != 300:
        raise DownloadProbeError("download promotion threshold changed")
    if any(int(spec.get(field, -1)) != 0 for field in (
        "confirmation_items_read", "science_items_evaluated", "model_loads", "gpu_work"
    )):
        raise DownloadProbeError("download probe cannot read science or use a GPU")
    runtime = spec.get("runtime", {})
    if runtime.get("job", {}).get("execution_site") != "gateway":
        raise DownloadProbeError("download probe controller must run on the gateway")
    command = runtime.get("job", {}).get("command", [])
    expected_url_file = f"/tmp/parallax-lifecycle-{spec.get('run_id')}/s3-get-url.secret"
    if not isinstance(command, list) or expected_url_file not in command:
        raise DownloadProbeError("download probe URL secret path is not exact")
    return spec


def validate_download_url(url: str, spec: dict[str, Any]) -> str:
    source = spec["source"]
    parsed = urlsplit(url.strip())
    query = parse_qs(parsed.query)
    expected_host = f"{source['bucket']}.s3.{source['region']}.amazonaws.com"
    if (
        parsed.scheme != "https"
        or parsed.netloc != expected_host
        or parsed.path != "/" + source["key"]
        or query.get("versionId") != [source["version_id"]]
        or query.get("X-Amz-Algorithm") != ["AWS4-HMAC-SHA256"]
        or query.get("X-Amz-SignedHeaders") != ["host"]
        or not 60 <= int(query.get("X-Amz-Expires", ["0"])[0]) <= 900
        or len(query.get("X-Amz-Signature", [""])[0]) != 64
    ):
        raise DownloadProbeError("download URL is not exact, version-bound, and presigned")
    return url.strip()


REMOTE_PROBE = r'''import concurrent.futures,hashlib,json,os,pathlib,sys,time,urllib.request
url=sys.stdin.readline().strip()
root=pathlib.Path(sys.argv[1]); count=int(sys.argv[2]); streams=int(sys.argv[3]); expected_bytes=int(sys.argv[4]); expected_sha=sys.argv[5]
root.mkdir(parents=True,exist_ok=False)
def one(index):
 path=root/(f'copy-{index:03d}.bin')
 digest=hashlib.sha256(); size=0
 with urllib.request.urlopen(url,timeout=60) as response, path.open('wb') as handle:
  while True:
   chunk=response.read(8*1024*1024)
   if not chunk: break
   handle.write(chunk);digest.update(chunk);size+=len(chunk)
  handle.flush();os.fsync(handle.fileno())
 return {'index':index,'bytes':size,'sha256':digest.hexdigest(),'match':size==expected_bytes and digest.hexdigest()==expected_sha}
started=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool: rows=list(pool.map(one,range(count)))
elapsed=time.monotonic()-started
for path in root.iterdir(): path.unlink()
root.rmdir()
total=sum(row['bytes'] for row in rows); errors=[row for row in rows if not row['match']]
print(json.dumps({'status':'GREEN' if not errors else 'RED','elapsed_seconds':elapsed,'total_bytes':total,'bytes_per_second':total/elapsed,'verified_copies':len(rows)-len(errors),'errors':errors[:8]},sort_keys=True))'''


def _ssh_base(host: str, port: int, key: Path) -> list[str]:
    return [
        "ssh", "-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10", "-p", str(port), f"root@{host}",
    ]


def _publish_result(ssh: list[str], worker_root: str, packet: dict[str, Any]) -> None:
    encoded = (json.dumps(packet, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = worker_root.rstrip("/") + "/result.json"
    temporary = worker_root.rstrip("/") + "/.result.json.partial"
    command = (
        f"umask 077; mkdir -p {shlex.quote(worker_root)}; test ! -e {shlex.quote(target)}; "
        f"cat > {shlex.quote(temporary)}; sync {shlex.quote(temporary)}; "
        f"mv {shlex.quote(temporary)} {shlex.quote(target)}"
    )
    completed = subprocess.run([*ssh, command], input=encoded, capture_output=True, timeout=30)
    if completed.returncode != 0:
        raise DownloadProbeError("could not publish worker download-probe result")


def run(
    spec: dict[str, Any], *, url_file: Path, worker_host: str, worker_port: int,
    worker_key: Path, worker_result_root: str,
) -> dict[str, Any]:
    validate_download_probe_spec(spec)
    url = url_file.read_text(encoding="utf-8").strip()
    url_file.unlink(missing_ok=True)
    url = validate_download_url(url, spec)
    source = spec["source"]
    ssh = _ssh_base(worker_host, worker_port, worker_key)
    command = [
        *ssh,
        "CUDA_VISIBLE_DEVICES='' python3 -c " + shlex.quote(REMOTE_PROBE) + " "
        + " ".join(
            shlex.quote(str(value)) for value in (
                spec["worker_scratch_root"], spec["repetitions"], spec["parallel_streams"],
                source["bytes"], source["sha256"],
            )
        ),
    ]
    completed = subprocess.run(
        command, input=(url + "\n").encode("utf-8"), capture_output=True,
        timeout=int(spec["runtime_timeout_seconds"]),
    )
    url = ""
    if completed.returncode != 0:
        raise DownloadProbeError(
            "worker download command failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-500:]
        )
    measured = json.loads(completed.stdout.decode("utf-8"))
    throughput = float(measured.get("bytes_per_second", 0))
    projected_seconds = float(spec["projected_full_cache_bytes"]) / throughput if throughput > 0 else None
    green = (
        measured.get("status") == "GREEN"
        and int(measured.get("total_bytes", -1)) == int(spec["maximum_download_bytes"])
        and int(measured.get("verified_copies", -1)) == int(spec["repetitions"])
        and projected_seconds is not None
        and projected_seconds <= float(spec["maximum_projected_full_cache_seconds"])
    )
    packet = {
        "schema_version": 1,
        "status": "GREEN_RATE_AND_HASH" if green else "RED_RATE_OR_HASH",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": spec["run_id"],
        "source": source,
        "repetitions": spec["repetitions"],
        "parallel_streams": spec["parallel_streams"],
        "measured": measured,
        "projected_full_cache_bytes": spec["projected_full_cache_bytes"],
        "projected_full_cache_seconds": projected_seconds,
        "maximum_projected_full_cache_seconds": spec["maximum_projected_full_cache_seconds"],
        "url_persisted": False,
        "aws_credentials_present": False,
        "confirmation_items_read": 0,
        "science_items_evaluated": 0,
        "model_loads": 0,
        "gpu_work": 0,
    }
    _publish_result(ssh, worker_result_root, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--url-file", type=Path, required=True)
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--worker-key", type=Path, required=True)
    parser.add_argument("--worker-result-root", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    try:
        packet = run(
            spec, url_file=args.url_file, worker_host=args.worker_host,
            worker_port=args.worker_port, worker_key=args.worker_key,
            worker_result_root=args.worker_result_root,
        )
    except Exception as error:
        args.url_file.unlink(missing_ok=True)
        packet = {
            "schema_version": 1, "status": "RED", "run_id": spec.get("run_id"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "failure": {"type": type(error).__name__, "message": str(error)[:1000]},
            "url_persisted": False, "aws_credentials_present": False,
            "confirmation_items_read": 0, "science_items_evaluated": 0,
            "model_loads": 0, "gpu_work": 0,
        }
        try:
            _publish_result(
                _ssh_base(args.worker_host, args.worker_port, args.worker_key),
                args.worker_result_root,
                packet,
            )
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
