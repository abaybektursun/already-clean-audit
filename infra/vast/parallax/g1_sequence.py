"""Run paired G1 entry points sequentially with exact, isolated runtime paths."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import ArtifactManifest, atomic_json
from .g1_common import G1GateError, utc_now


def runtime_environment(site_packages: Path, forbidden_site_packages: Path) -> dict[str, str]:
    site = site_packages.resolve()
    forbidden = forbidden_site_packages.resolve()
    if site == forbidden or not site.is_dir() or not forbidden.is_dir():
        raise G1GateError("core and pi runtime sites must exist and be distinct")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(site)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if str(forbidden) in environment["PYTHONPATH"].split(os.pathsep):
        raise G1GateError("runtime firewall includes the forbidden site-packages path")
    return environment


def _run_step(
    *,
    name: str,
    module: str,
    arguments: list[str],
    environment: dict[str, str],
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = output_dir / f"{name}.stdout.log"
    stderr = output_dir / f"{name}.stderr.log"
    stdout.write_text(process.stdout, encoding="utf-8")
    stderr.write_text(process.stderr, encoding="utf-8")
    result = {
        "name": name,
        "module": module,
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": stdout.name,
        "stderr": stderr.name,
    }
    return result


def run(
    *,
    root: Path,
    matrix: Path,
    anchors: Path,
    cache_root: Path,
    task_root: Path,
    core_site: Path,
    pi_site: Path,
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite G1 output: {output_dir}")
    output_dir.mkdir(parents=True)
    core_env = runtime_environment(core_site, pi_site)
    pi_env = runtime_environment(pi_site, core_site)
    common = [
        "--matrix",
        str(matrix),
        "--anchors",
        str(anchors),
        "--cache-root",
        str(cache_root),
        "--task-root",
        str(task_root),
        "--instance-id",
        str(instance_id),
    ]
    steps = []
    error = None
    started = time.monotonic()

    def execute(**kwargs: Any) -> None:
        result = _run_step(**kwargs)
        steps.append(result)
        if result["returncode"] != 0:
            raise G1GateError(
                f"G1 step stopped: {result['name']} returncode={result['returncode']}"
            )

    try:
        execute(
            name="01_primary_core",
            module="infra.vast.parallax.g1_core",
            arguments=[
                *common,
                "--lineage",
                "pg1_pi",
                "--output",
                str(output_dir / "primary-core.json"),
            ],
            environment=core_env,
            root=root,
            output_dir=output_dir,
        )
        execute(
            name="02_pi_sidecar",
            module="infra.vast.parallax.g1_pi",
            arguments=[*common, "--output", str(output_dir / "pi-sidecar.json")],
            environment=pi_env,
            root=root,
            output_dir=output_dir,
        )
        execute(
            name="03_supporting_core",
            module="infra.vast.parallax.g1_core",
            arguments=[
                *common,
                "--lineage",
                "pg2",
                "--output",
                str(output_dir / "supporting-core.json"),
            ],
            environment=core_env,
            root=root,
            output_dir=output_dir,
        )
        execute(
            name="04_paired_adjudication",
            module="infra.vast.parallax.g1_adjudicate",
            arguments=[
                "--primary-core",
                str(output_dir / "primary-core.json"),
                "--pi-sidecar",
                str(output_dir / "pi-sidecar.json"),
                "--supporting-core",
                str(output_dir / "supporting-core.json"),
                "--output",
                str(output_dir / "hour6-decision.json"),
            ],
            environment=core_env,
            root=root,
            output_dir=output_dir,
        )
        status = "GREEN"
    except Exception as caught:
        status = "STOPPED"
        error = f"{type(caught).__name__}: {caught}"
    execution = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": status,
        "provider": "vast.ai",
        "instance_id": instance_id,
        "elapsed_seconds": time.monotonic() - started,
        "runtime_firewall": {
            "core_site": str(core_site.resolve()),
            "pi_site": str(pi_site.resolve()),
            "sites_distinct": core_site.resolve() != pi_site.resolve(),
            "old_mlops_paths_supplied": False,
        },
        "steps": steps,
        "error": error,
    }
    atomic_json(output_dir / "execution.json", execution)
    manifest = ArtifactManifest(output_dir, "paired_g1")
    manifest.add(*(path for path in output_dir.iterdir() if path.name != "manifest.json"))
    manifest.write(output_dir / "manifest.json")
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--core-site", type=Path, required=True)
    parser.add_argument("--pi-site", type=Path, required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        root=args.root.resolve(),
        matrix=args.matrix.resolve(),
        anchors=args.anchors.resolve(),
        cache_root=args.cache_root.resolve(),
        task_root=args.task_root.resolve(),
        core_site=args.core_site.resolve(),
        pi_site=args.pi_site.resolve(),
        instance_id=args.instance_id,
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}))
    return 0 if result["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
