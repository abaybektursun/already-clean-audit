"""Build a minimal, deterministic real-gate bundle with no confirmation payload."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .parallax_tasks import DiscoveryTaskBank
from .training_common import TrainingGateError
from .training_model import select_split_stratified


PACKAGE_INIT_PATHS = (
    "infra/__init__.py",
    "infra/vast/__init__.py",
    "infra/vast/parallax/__init__.py",
    "infra/vast/parallax/dag.py",
    "infra/vast/parallax/launch_packet.py",
)

IMPORT_SMOKE_MODULES = (
    "infra.vast.parallax.training_real_gate_preflight",
    "infra.vast.parallax.training_real_gate",
)


def _normalized_add(archive: tarfile.TarFile, root: Path, relative: str) -> None:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    info = archive.gettarinfo(str(path), arcname=relative)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def _isolated_import_smoke(bundle: Path) -> dict[str, Any]:
    """Prove the archive contains the runtime import closure, outside the repo tree."""
    with tempfile.TemporaryDirectory(prefix="parallax-training-transfer-smoke-") as temporary:
        extracted = Path(temporary)
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall(extracted, filter="data")
        command = [
            sys.executable,
            "-c",
            ";".join(f"import {module}" for module in IMPORT_SMOKE_MODULES),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(extracted)
        completed = subprocess.run(
            command,
            cwd=extracted,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise TrainingGateError(f"isolated transfer-bundle import smoke failed: {detail}")
    return {
        "status": "GREEN",
        "modules": list(IMPORT_SMOKE_MODULES),
        "repo_worktree_on_sys_path": False,
    }


def build(*, root: Path, contract_path: Path, output: Path, receipt: Path) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise FileExistsError("refusing to overwrite training transfer bundle")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "FROZEN":
        raise TrainingGateError("source contract is not frozen")
    entries = {row["path"] for row in contract["artifacts"].values()}
    entries.update(PACKAGE_INIT_PATHS)
    entries.add(str(contract_path.resolve().relative_to(root.resolve())))
    for name, row in contract["artifacts"].items():
        if sha256_file(root / row["path"]) != row["sha256"]:
            raise TrainingGateError(f"cannot bundle source hash mismatch: {name}")

    config = yaml.safe_load((root / contract["artifacts"]["config"]["path"]).read_text())
    task_root = root / "studies/lineage/cross-lineage/data/parallax_tasks_v1"
    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["panel"]["task_id"]))
    selected = []
    for split, count_key in ((config["panel"]["train_split"], "train_per_target"),
                             (config["panel"]["discovery_split"], "discovery_per_target")):
        selected.extend(
            select_split_stratified(
                rows,
                task_id=config["panel"]["task_id"],
                split=split,
                per_target=int(config["panel"][count_key]),
            )
        )
    image_entries = {str((task_root / row["image"]).relative_to(root)) for row in selected}
    entries.update(image_entries)
    forbidden_tokens = ("items.jsonl", "confirmation_eval", "confirmation.sha256")
    forbidden = [entry for entry in entries if any(token in entry for token in forbidden_tokens)]
    if forbidden:
        raise TrainingGateError(f"confirmation-capable artifact entered transfer bundle: {forbidden}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as compressed:
            with tarfile.open(mode="w", fileobj=compressed, format=tarfile.PAX_FORMAT) as archive:
                for relative in sorted(entries):
                    _normalized_add(archive, root, relative)
    import_smoke = _isolated_import_smoke(output)
    manifest_entries = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(entries)
    ]
    packet = {
        "schema_version": 1,
        "status": "GREEN",
        "bundle": output.name,
        "bundle_sha256": sha256_file(output),
        "bundle_bytes": output.stat().st_size,
        "source_contract_sha256": sha256_file(contract_path),
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "selected_item_ids": [row["item_id"] for row in selected],
        "selected_image_count": len(image_entries),
        "isolated_import_smoke": import_smoke,
        "confirmation_manifest_included": False,
        "confirmation_image_included": False,
    }
    atomic_json(receipt, packet)
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    packet = build(
        root=args.root.resolve(),
        contract_path=args.contract.resolve(),
        output=args.output.resolve(),
        receipt=args.receipt.resolve(),
    )
    print(json.dumps({"status": packet["status"], "bundle_sha256": packet["bundle_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
