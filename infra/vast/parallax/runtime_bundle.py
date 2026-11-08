"""Build and activate a content-addressed Linux runtime without H100 installs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_json, sha256_file


def build(
    repo_root: Path,
    output_dir: Path,
    *,
    lock_path: Path | None = None,
    bundle_name: str = "parallax-runtime-linux-amd64.tar.gz",
    runtime_role: str = "core",
) -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("runtime bundle must be built on a Linux amd64 Vast gateway")
    lock = lock_path or (repo_root / "infra/vast/parallax/requirements.lock")
    lock = lock.resolve()
    if not lock.is_file():
        raise FileNotFoundError(lock)
    if Path(bundle_name).name != bundle_name or not bundle_name.endswith(".tar.gz"):
        raise ValueError("bundle_name must be a plain .tar.gz filename")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parallax-runtime-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        site = stage / "site-packages"
        metadata = stage / "metadata"
        site.mkdir()
        metadata.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--target",
                str(site),
                "-r",
                str(lock),
            ],
            cwd=repo_root,
            check=True,
        )
        shutil.copy2(lock, metadata / "requirements.lock")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join([str(site), str(repo_root)])
        freeze_process = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--path", str(site), "--format", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        if freeze_process.returncode != 0:
            raise RuntimeError(f"pip list failed: {freeze_process.stderr.strip()}")
        freeze = freeze_process.stdout
        (metadata / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
        check_process = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if check_process.returncode != 0:
            raise RuntimeError(f"pip check failed: {check_process.stdout.strip()} {check_process.stderr.strip()}")
        check = check_process.stdout
        (metadata / "pip-check.txt").write_text(check, encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "infra.vast.parallax.image_smoke",
                "--lock",
                str(lock),
                "--output",
                str(metadata / "cpu-smoke.json"),
            ],
            cwd=repo_root,
            check=True,
            env=environment,
        )
        if runtime_role == "pi_sidecar":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "infra.vast.parallax.pi_sidecar_smoke",
                    "--output",
                    str(metadata / "pi-sidecar-smoke.json"),
                ],
                cwd=repo_root,
                check=True,
                env=environment,
            )

        output_dir.mkdir()
        bundle = output_dir / bundle_name
        with tarfile.open(bundle, "w:gz", compresslevel=6) as archive:
            archive.add(site, arcname="site-packages", recursive=True)
            archive.add(metadata, arcname="metadata", recursive=True)
        digest = sha256_file(bundle)
        (output_dir / "parallax-runtime-linux-amd64.tar.gz.sha256").write_text(
            f"{digest}  {bundle.name}\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "GREEN",
            "platform": "linux/amd64",
            "base_image": "vastai/base-image:cuda-13.1.2-auto",
            "base_image_digest": "sha256:4dc9d17083bc5e09dabdcf142db10637823cbf38c4e4253949f508d30d9f1749",
            "requirements_sha256": sha256_file(lock),
            "requirements_name": lock.name,
            "runtime_role": runtime_role,
            "bundle": bundle.name,
            "bundle_sha256": digest,
            "sha256": digest,
            "bundle_bytes": bundle.stat().st_size,
            "installation_location": "persistent Vast gateway only",
            "h100_action": "verify_sha256_and_extract_no_pip_no_internet",
        }
        atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def activate(bundle: Path, expected_sha256: str, target_root: Path) -> dict[str, str]:
    actual = sha256_file(bundle)
    if actual != expected_sha256:
        raise ValueError(f"runtime bundle SHA mismatch: expected={expected_sha256} actual={actual}")
    final = target_root / actual
    if not final.exists():
        target_root.mkdir(parents=True, exist_ok=True)
        temporary = target_root / f".{actual}.partial"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        os.replace(temporary, final)
    site = final / "site-packages"
    smoke = final / "metadata/cpu-smoke.json"
    if not site.is_dir() or not smoke.is_file():
        raise ValueError("runtime bundle is missing site-packages or CPU smoke evidence")
    return {
        "runtime_sha256": actual,
        "runtime_root": str(final),
        "pythonpath": str(site),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    builder = subparsers.add_parser("build")
    builder.add_argument("--repo-root", type=Path, required=True)
    builder.add_argument("--output-dir", type=Path, required=True)
    builder.add_argument("--lock", type=Path)
    builder.add_argument("--bundle-name", default="parallax-runtime-linux-amd64.tar.gz")
    builder.add_argument("--runtime-role", default="core")
    activation = subparsers.add_parser("activate")
    activation.add_argument("--bundle", type=Path, required=True)
    activation.add_argument("--sha256", required=True)
    activation.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(
            args.repo_root.resolve(),
            args.output_dir.resolve(),
            lock_path=args.lock,
            bundle_name=args.bundle_name,
            runtime_role=args.runtime_role,
        )
    else:
        result = activate(args.bundle.resolve(), args.sha256, args.target_root.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
