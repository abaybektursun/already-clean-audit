"""Run the primary PG1 interface cell using the external green processor gate.

The completed PG2 job remains byte-identical.  This adapter temporarily binds
its mature S3 hydration/archive mechanics to the independent PG1 lineage, and
replaces the on-H100 tokenizer preflight with exact validation of the processor
terminal that was made durable before H100 authority existed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:  # Staged beside the legacy job on the disposable H100.
    import ephemeral_interface_control_job as legacy
except ModuleNotFoundError:
    from . import ephemeral_interface_control_job as legacy


RUN_ID = "parallax-pg1-primary-interface-s3-v2"
SOURCE_RUN_ID = "parallax-pg1-primary-interface-s3-v1"
LINEAGE = "pg1_pi"
SOURCE_CONTRACT_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_interface_control_source_contract_pg1_primary_v1.json"
)
CONFIG_PATH = (
    "studies/lineage/cross-lineage/config/"
    "parallax_training_interface_control_pg1_primary_v1.yaml"
)
MATRIX_PATH = "studies/lineage/cross-lineage/config/parallax_matrix.yaml"
TASK_ROOT = "studies/lineage/cross-lineage/data/parallax_tasks_v1"
EXTERNAL_PROCESSOR_TERMINAL_PATH = (
    "studies/lineage/cross-lineage/results/readiness/"
    "parallax_pg1_primary_processor_preflight_v1/terminal.json"
)
EXTERNAL_PROCESSOR_TERMINAL_SHA256 = (
    "f751ef90acd6a96cafe3dea4b491ab8df9d06d6f7faa8388ebda798c5f4b2ba9"
)
EXTERNAL_PROCESSOR_S3_VERSION_ID = "DcMFOhnKRi9raBdh8723ZeDUX2wqjXwC"
MODEL_FILES = 12
MODEL_BYTES = 11_715_866_017
MODEL_NAMES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
WEIGHT_NAMES = {
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
}
OVERLAY_IDENTITIES = {
    "infra/vast/parallax/training_interface_control.py": (
        "81ec1e5b7a6fa50db0a8243c21f03bb9c6a9830a6b25532de763abf8d5497da6"
    ),
    "infra/vast/parallax/training_interface_control_cell.py": (
        "2bdfb7bef9137eccf51211b5708d2bef9f98fe3d2577593479ab2c24f284647e"
    ),
    "infra/vast/parallax/training_interface_control_primary_cell.py": (
        "faf4acbadccbcb730ca7388131611b3facf4e15d70cb4523ad3360973441491a"
    ),
    "infra/vast/parallax/training_interface_interaction.py": (
        "ca95530cb74182c549f18535ab12787660c1dd17608164b1b71a771097ecfbb7"
    ),
    "studies/lineage/cross-lineage/config/parallax_execution_freeze_v111.json": (
        "c6976d93c756dbf5f6e03180e170e99501490c54d1f85309287bbe1f51f169e8"
    ),
    "studies/lineage/cross-lineage/config/parallax_interface_interaction_metric_v1.yaml": (
        "5bffe282c29def40828add88b7d2158fd8a41f3e16b166e02e641bbae3bcf608"
    ),
    "studies/lineage/cross-lineage/config/parallax_pg1_h100_s3_hydration_job_v1.json": (
        "fc186748f7a19ca922aac1a25f09745b366d1b4efb62353dc2254a5bc303fbab"
    ),
    "studies/lineage/cross-lineage/config/parallax_training_interface_control_pg1_primary_v1.yaml": (
        "fa5d8ccefeacfbb57fcd223b3b4486d4adef80388ae06d48452221d738e53078"
    ),
    SOURCE_CONTRACT_PATH: (
        "a95f65a78d5dc6b7f4b80416d40c6b9dcc45cc1c91fde74154da90dd117e5544"
    ),
    "studies/lineage/cross-lineage/research/2026-07-18_parallax_interface_interaction_metric_spec.md": (
        "a92181ec909d12942a59b5e6f11ac63575f3d9fd0a63bbba7b4fc5b94ec6ecba"
    ),
    "studies/lineage/cross-lineage/research/2026-07-18_parallax_pg1_primary_interface_control_spec.md": (
        "e2623b5220c5c86aca3f84580e637d0f98f3fc147e2d5e2941f757f0659cc3d2"
    ),
    "studies/lineage/cross-lineage/results/readiness/parallax_pg1_primary_processor_preflight_v1/adjudication.json": (
        "1d418bb360c401c150562cd2535db3ea95904a6c5b46592d7a357a8dd6c3c01c"
    ),
    EXTERNAL_PROCESSOR_TERMINAL_PATH: EXTERNAL_PROCESSOR_TERMINAL_SHA256,
    "studies/lineage/cross-lineage/results/readiness/parallax_primary_next_cell_audit_v1.json": (
        "c7437eddce4532b71299003c7f6a3b712484a5bfea9185f680c761cde69866b4"
    ),
}
REQUIRED_OVERLAY_PATHS = {
    "infra/vast/parallax/training_interface_control.py",
    "infra/vast/parallax/training_interface_control_cell.py",
    "infra/vast/parallax/training_interface_control_primary_cell.py",
    "infra/vast/parallax/training_interface_interaction.py",
    CONFIG_PATH,
    SOURCE_CONTRACT_PATH,
    EXTERNAL_PROCESSOR_TERMINAL_PATH,
}
EXPECTED_SNAPSHOTS = [0, 1, 2, 4, 8, 16]
EXPECTED_CODEBOOKS = ["cycle_0", "cycle_1", "cycle_2"]


class PrimaryInterfaceJobError(RuntimeError):
    pass


def legacy_compatible_source_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Derive the legacy config label from the one exact PG1 config binding.

    The primary contract deliberately names its only training design
    ``primary_config``.  The reused, hash-frozen PG2 cell still consumes the
    historical ``interface_config`` label.  Keep that compatibility alias
    primary-only and in memory: the frozen source contract remains untouched,
    and either a second label or any path/hash disagreement is fatal.
    """

    artifacts = contract.get("artifacts")
    expected = {
        "path": CONFIG_PATH,
        "sha256": OVERLAY_IDENTITIES[CONFIG_PATH],
    }
    if (
        not isinstance(artifacts, dict)
        or len(artifacts) != 26
        or "interface_config" in artifacts
        or artifacts.get("primary_config") != expected
    ):
        raise PrimaryInterfaceJobError(
            "primary source contract has no unique exact legacy config alias"
        )
    compatible = copy.deepcopy(contract)
    compatible["artifacts"]["interface_config"] = dict(expected)
    return compatible


def materialize_legacy_source_contract(*, source: Path, runtime_root: Path) -> Path:
    original = source / SOURCE_CONTRACT_PATH
    expected_sha256 = OVERLAY_IDENTITIES[SOURCE_CONTRACT_PATH]
    if not original.is_file() or legacy.base.sha256_file(original) != expected_sha256:
        raise PrimaryInterfaceJobError("primary source contract is absent or hash-drifted")
    if not runtime_root.is_dir():
        raise PrimaryInterfaceJobError("primary runtime root is absent")
    output = runtime_root / "primary-interface-source-contract-compat.json"
    if output.exists():
        raise PrimaryInterfaceJobError("primary compatibility contract already exists")
    contract = json.loads(original.read_text(encoding="utf-8"))
    compatible = legacy_compatible_source_contract(contract)
    legacy.base.atomic_json(output, compatible)
    if json.loads(output.read_text(encoding="utf-8")) != compatible:
        raise PrimaryInterfaceJobError("primary compatibility contract write disagrees")
    return output


def _overlay_identities(contract: dict[str, Any]) -> dict[str, str]:
    rows = contract.get("overlays")
    if not isinstance(rows, list):
        raise PrimaryInterfaceJobError("primary interface overlays are absent")
    observed: dict[str, str] = {}
    for row in rows:
        path = str(row.get("path", ""))
        if path in observed:
            raise PrimaryInterfaceJobError("primary interface overlay path is duplicated")
        observed[path] = str(row.get("sha256", ""))
    if observed != OVERLAY_IDENTITIES or not REQUIRED_OVERLAY_PATHS.issubset(observed):
        raise PrimaryInterfaceJobError("primary interface overlay identity set changed")
    return observed


@contextmanager
def _patched_legacy(contract: dict[str, Any]) -> Iterator[None]:
    replacements = {
        "RUN_ID": RUN_ID,
        "LINEAGE": LINEAGE,
        "SOURCE_CONTRACT_PATH": SOURCE_CONTRACT_PATH,
        "SOURCE_CONTRACT_SHA256": str(contract.get("source_contract_sha256", "")),
        "CONFIG_PATH": CONFIG_PATH,
        "MATRIX_PATH": MATRIX_PATH,
        "TASK_ROOT": TASK_ROOT,
        "MODEL_FILES": MODEL_FILES,
        "MODEL_BYTES": MODEL_BYTES,
        "MODEL_NAMES": MODEL_NAMES,
        "WEIGHT_NAMES": WEIGHT_NAMES,
        "OVERLAY_IDENTITIES": _overlay_identities(contract),
        "EXPECTED_SNAPSHOTS": EXPECTED_SNAPSHOTS,
        "EXPECTED_CODEBOOKS": EXPECTED_CODEBOOKS,
    }
    original = {name: getattr(legacy, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(legacy, name, value)


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("run_id") != RUN_ID
        or contract.get("lineage") != LINEAGE
        or contract.get("external_processor_gate") != {
            "path": EXTERNAL_PROCESSOR_TERMINAL_PATH,
            "sha256": EXTERNAL_PROCESSOR_TERMINAL_SHA256,
            "s3_version_id": EXTERNAL_PROCESSOR_S3_VERSION_ID,
            "status": "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT_EXECUTED",
            "h100_instances_created": 0,
            "model_weight_urls": 0,
            "model_loads": 0,
            "gpu_operations": 0,
            "confirmation_items_read": 0,
        }
    ):
        raise PrimaryInterfaceJobError("external primary processor gate is absent or changed")
    model = contract.get("model")
    if (
        not isinstance(model, list)
        or len(model) != MODEL_FILES
        or {str(row.get("path", "")) for row in model} != MODEL_NAMES
        or sum(int(row.get("bytes", 0)) for row in model) != MODEL_BYTES
    ):
        raise PrimaryInterfaceJobError("primary PG1 model inventory changed")
    digest = str(contract.get("source_contract_sha256", ""))
    if digest != OVERLAY_IDENTITIES[SOURCE_CONTRACT_PATH]:
        raise PrimaryInterfaceJobError("primary source-contract digest changed")
    _overlay_identities(contract)
    with _patched_legacy(contract):
        legacy.validate_contract(contract)
    return contract


def _external_processor_preflight(
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(contract["paths"]["source_root"])
    path = source / EXTERNAL_PROCESSOR_TERMINAL_PATH
    if not path.is_file() or legacy.base.sha256_file(path) != EXTERNAL_PROCESSOR_TERMINAL_SHA256:
        raise PrimaryInterfaceJobError("external processor terminal is absent or hash-drifted")
    terminal = json.loads(path.read_text(encoding="utf-8"))
    payload = terminal.get("job", {}).get("result", {}).get("payload", {})
    preflight = payload.get("preflight")
    if (
        terminal.get("status") != "GREEN_JOB_TERMINAL"
        or terminal.get("run_id") != "parallax-pg1-primary-processor-preflight-v1"
        or payload.get("status") != "GREEN_PRIMARY_PG1_PROCESSOR_PREFLIGHT"
        or payload.get("h100_instances_created") != 0
        or payload.get("model_weight_urls") != 0
        or payload.get("model_loads") != 0
        or payload.get("gpu_operations") != 0
        or payload.get("confirmation_items_read") != 0
        or not isinstance(preflight, dict)
        or preflight.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or preflight.get("lineage") != LINEAGE
        or preflight.get("candidate_suffix_tokens") != {"A": 2, "B": 2, "C": 2}
        or len({tuple(value) for value in preflight.get("candidate_suffix_token_ids", {}).values()}) != 3
        or preflight.get("prefix_tokens") != 302
        or preflight.get("image_tokens") != 256
        or preflight.get("model_loads") != 0
        or preflight.get("gpu_operations") != 0
        or preflight.get("confirmation_items_read") != 0
    ):
        raise PrimaryInterfaceJobError("external processor terminal is not exact green")
    output = Path(contract["paths"]["tokenizer_preflight_output"])
    legacy.base.atomic_json(output, preflight)
    return preflight, {
        "returncode": 0,
        "elapsed_seconds": 0.0,
        "stdout_tail": "external processor gate verified before H100 authorization",
        "external_terminal_sha256": EXTERNAL_PROCESSOR_TERMINAL_SHA256,
        "external_terminal_s3_version_id": EXTERNAL_PROCESSOR_S3_VERSION_ID,
        "processor_calls_repeated_on_h100": 0,
    }


def _run_primary_control(
    contract: dict[str, Any], tokenizer_result: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(contract["paths"]["source_root"])
    output_dir = Path(contract["paths"]["control_output_dir"])
    cache_root = Path(contract["paths"]["snapshot_root"]).parents[2]
    compatible_source_contract = materialize_legacy_source_contract(
        source=source,
        runtime_root=Path(contract["paths"]["runtime_root"]),
    )
    command = [
        sys.executable,
        "-m",
        "infra.vast.parallax.training_interface_control_primary_cell",
        "--root", str(source),
        "--source-contract", str(compatible_source_contract),
        "--matrix", str(source / MATRIX_PATH),
        "--config", str(source / CONFIG_PATH),
        "--tokenizer-preflight", str(Path(contract["paths"]["tokenizer_preflight_output"])),
        "--cache-root", str(cache_root),
        "--task-root", str(source / TASK_ROOT),
        "--instance-id", str(contract["instance_id"]),
        "--output-dir", str(output_dir),
    ]
    started = time.monotonic()
    run = legacy.base._run_command(
        command,
        cwd=source,
        environment=legacy._runtime_environment(source),
        timeout=int(contract["bounds"]["control_timeout_seconds"]),
        label="primary PG1 three-codebook interface control",
    )
    result = json.loads((output_dir / "interface-control.json").read_text(encoding="utf-8"))
    if (
        result.get("status") != "GREEN_MECHANICAL_PRIMARY_INTERFACE_EXECUTION"
        or result.get("lineage") != LINEAGE
        or result.get("label") != "PRIMARY_PG1_INTERFACE_DISCOVERY_NOT_PROMOTION_ELIGIBLE"
        or result.get("supporting_pg2_dependency") != "none_within_lineage"
        or result.get("paired_or_shared_range_claim") is not False
        or result.get("confirmation_items_read") != 0
        or result.get("promotion_eligible") is not False
        or result.get("fused_base_reset_exact") is not True
        or [row.get("codebook_id") for row in result.get("codebooks", [])] != EXPECTED_CODEBOOKS
        or any(
            [int(snapshot["step"]) for snapshot in row.get("snapshots", [])] != EXPECTED_SNAPSHOTS
            for row in result.get("codebooks", [])
        )
        or result.get("tokenizer_preflight", {}).get("candidate_suffix_token_ids")
        != tokenizer_result["candidate_suffix_token_ids"]
    ):
        raise PrimaryInterfaceJobError("primary PG1 interface output escaped its frozen boundary")
    run["elapsed_seconds_including_primary_metric"] = time.monotonic() - started
    return result, run


def run(input_contract_path: Path, artifact_post_path: Path) -> dict[str, Any]:
    raw = json.loads(input_contract_path.read_text(encoding="utf-8"))
    validate_contract(raw)
    original_preflight = legacy._run_tokenizer_preflight
    original_control = legacy._run_control
    try:
        with _patched_legacy(raw):
            legacy._run_tokenizer_preflight = _external_processor_preflight
            legacy._run_control = _run_primary_control
            return legacy.run(input_contract_path, artifact_post_path)
    finally:
        legacy._run_tokenizer_preflight = original_preflight
        legacy._run_control = original_control


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--artifact-post", type=Path, required=True)
    args = parser.parse_args()
    packet = run(args.input_contract.resolve(), args.artifact_post.resolve())
    print(json.dumps({"run_id": packet["run_id"], "status": packet["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
