"""PG1-native cell with explicit immutable policy and authorization inputs."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml

from .artifacts import atomic_json, sha256_file
from .g1_common import assert_attention_eager, release_cuda, resolve_snapshot, utc_now
from .parallax_entities import frozen_entities
from .parallax_tasks import DiscoveryTaskBank
from .training_common import (
    TrainingGateError,
    atomic_training_checkpoint,
    deterministic_batches,
    named_parameter_sha256,
    optimizer_state_sha256,
    schedule_sha256,
)
from .training_interface_control import (
    CODEBOOKS,
    CONTROL_TASK_ID,
    SEMANTIC_LABELS,
    audit_equal_suffix_counts,
    causal_suffix_loss_per_example,
    one_update_per_example,
    remap_rows,
    semantic_summary,
    validate_codebooks,
)
from .training_model import (
    _attention_configs,
    attach_free_lora_connector,
    evaluate_candidates,
    real_training_batch,
    select_split_stratified,
    suffix_anchor,
)
from .pg1_execution_capsule import (
    ENTITY_ROLE,
    LINEAGE,
    LINEAGE_ROLE,
    PRIMARY_CONFIG_PATH,
    SOURCE_CONTRACT_PATH,
    CapsuleError,
    consumer_receipt,
    load_capsule,
    validate_authorization_receipt,
    validate_consumer,
    validate_execution_authorization,
)
from .pg1_native_source_gate import capsule_preflight as source_preflight
from .pg1_native_source_gate import validate_full_source
from .training_interface_control_primary_cell import (
    _validate_primary_config,
    adjudicate_primary_packet,
)


def _exact_primary_config(mapping: Any, *, owner: str) -> dict[str, Any]:
    expected_path = str(PRIMARY_CONFIG_PATH)
    if not isinstance(mapping, dict):
        raise CapsuleError(f"{owner} artifact mapping is absent")
    candidates = [
        (label, row)
        for label, row in mapping.items()
        if label in {"primary_config", "interface_config"}
        or (isinstance(row, dict) and row.get("path") == expected_path)
    ]
    if (
        len(candidates) != 1
        or candidates[0][0] != "primary_config"
        or not isinstance(candidates[0][1], dict)
        or candidates[0][1].get("path") != expected_path
    ):
        raise CapsuleError(f"{owner} requires exactly one unambiguous primary_config")
    return candidates[0][1]


def capsule_preflight(
    capsule: dict[str, Any],
    *,
    root: Path,
    source_contract: Path,
    config_path: Path,
    allow_parameterized_fixture: bool = False,
) -> dict[str, Any]:
    if source_contract.resolve() != (root.resolve() / SOURCE_CONTRACT_PATH).resolve():
        raise CapsuleError("native cell source-contract path drift")
    frozen = json.loads(source_contract.read_text(encoding="utf-8"))
    frozen_config = _exact_primary_config(frozen.get("artifacts"), owner="source contract")
    capsule_contract = capsule.get("contract")
    runtime_config = _exact_primary_config(
        capsule_contract.get("artifacts") if isinstance(capsule_contract, dict) else None,
        owner="capsule",
    )
    if runtime_config != {**frozen_config, "origin": runtime_config.get("origin")}:
        raise CapsuleError("native cell capsule/source config binding disagrees")
    if config_path.resolve() != (root.resolve() / PRIMARY_CONFIG_PATH).resolve():
        raise CapsuleError("native cell primary_config path drift")
    validate_consumer(capsule, "cell", Path(__file__).resolve())
    source_preflight(
        capsule,
        root,
        allow_parameterized_fixture=allow_parameterized_fixture,
    )
    _validate_primary_config(config_path)
    return consumer_receipt(
        capsule,
        "cell",
        {
            "config_label": "primary_config",
            "config_path": str(PRIMARY_CONFIG_PATH),
            "unique_config_candidates": 1,
            "primary_config_validated_before_ml_imports": True,
        },
    )


def _objective(model: Any, batch: dict[str, Any]) -> float:
    import torch

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch, use_cache=False, return_dict=True)
        loss = causal_suffix_loss_per_example(output.logits, batch["labels"])
    value = float(loss)
    if not math.isfinite(value):
        raise TrainingGateError("interface-control objective is nonfinite")
    return value


def _mean_score_items(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["item_id"])].append(row)
    items = []
    for item_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: str(row["candidate"]))
        values = [float(row["log_likelihood_mean"]) for row in ordered]
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
        probabilities = [weight / sum(weights) for weight in weights]
        target = str(ordered[0]["target"])
        target_index = next(index for index, row in enumerate(ordered) if row["candidate"] == target)
        predicted = max(range(len(values)), key=lambda index: (values[index], -index))
        items.append(
            {
                "item_id": item_id,
                "target": target,
                "prediction": str(ordered[predicted]["candidate"]),
                "correct": int(predicted == target_index),
                "target_probability": probabilities[target_index],
            }
        )
    return items


def _cell_adjudication(snapshots: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    first, final = snapshots[0], snapshots[-1]
    sum_gain = (
        final["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
        - first["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
    )
    mean_gain = (
        final["semantic_summary_mean"]["aggregate"]["mean_target_probability"]
        - first["semantic_summary_mean"]["aggregate"]["mean_target_probability"]
    )
    objective_drop = float(first["objective"] - final["objective"])
    semantic_gains = {
        semantic: (
            final["semantic_summary_sum"]["by_semantic_target"][semantic]["mean_target_probability"]
            - first["semantic_summary_sum"]["by_semantic_target"][semantic]["mean_target_probability"]
        )
        for semantic in SEMANTIC_LABELS
    }
    return {
        "objective_drop": objective_drop,
        "registered_control_probability_gain": sum_gain,
        "diagnostic_mean_score_probability_gain": mean_gain,
        "semantic_probability_gains": semantic_gains,
        "objective_clears_threshold": objective_drop > threshold,
        "positive_heldout_gain": sum_gain > 0.0,
        "all_semantic_gains_positive": all(value > 0.0 for value in semantic_gains.values()),
    }


def _aggregate_adjudication(
    cells: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gate = config["provisional_mechanical_adjudication"]
    objective_threshold = float(gate["minimum_objective_drop_per_passing_codebook"])
    gain_threshold = float(gate["minimum_mapping_marginalized_probability_gain"])
    initial = [
        cell["snapshots"][0]["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
        for cell in cells
    ]
    final = [
        cell["snapshots"][-1]["semantic_summary_sum"]["aggregate"]["mean_target_probability"]
        for cell in cells
    ]
    pooled_gain = sum(final) / len(final) - sum(initial) / len(initial)
    semantic_gains = {
        semantic: sum(
            cell["snapshots"][-1]["semantic_summary_sum"]["by_semantic_target"][semantic][
                "mean_target_probability"
            ]
            - cell["snapshots"][0]["semantic_summary_sum"]["by_semantic_target"][semantic][
                "mean_target_probability"
            ]
            for cell in cells
        )
        / len(cells)
        for semantic in SEMANTIC_LABELS
    }
    positive_cells = sum(cell["adjudication"]["positive_heldout_gain"] for cell in cells)
    objective_cells = sum(
        cell["adjudication"]["objective_drop"] > objective_threshold for cell in cells
    )
    finite = all(
        math.isfinite(value)
        for value in [pooled_gain, *semantic_gains.values()]
        + [
            value
            for cell in cells
            for value in (
                cell["adjudication"]["objective_drop"],
                cell["adjudication"]["registered_control_probability_gain"],
            )
        ]
    )
    baseline = sum(initial) / len(initial)
    if not finite:
        verdict = "ABORT_NONFINITE"
    elif baseline <= float(gate["floor_probability"]):
        verdict = "VOID_READOUT_FLOOR"
    elif baseline >= float(gate["ceiling_probability"]):
        verdict = "VOID_READOUT_CEILING"
    elif objective_cells < int(gate["minimum_codebooks_with_positive_probability_gain"]):
        verdict = "RED_OBJECTIVE_NOT_MOVING"
    elif positive_cells < int(gate["minimum_codebooks_with_positive_probability_gain"]):
        verdict = "RED_MAPPING_FRAGILE"
    elif pooled_gain <= gain_threshold:
        verdict = "NO_HELDOUT_RANGE"
    elif not all(value > 0.0 for value in semantic_gains.values()):
        verdict = "RED_SEMANTIC_NONUNIFORM"
    else:
        verdict = "GREEN_PROVISIONAL_INTERFACE_CONTROL"
    return {
        "verdict": verdict,
        "mapping_marginalized_initial_probability": baseline,
        "mapping_marginalized_final_probability": sum(final) / len(final),
        "mapping_marginalized_probability_gain": pooled_gain,
        "mapping_marginalized_semantic_gains": semantic_gains,
        "positive_gain_codebooks": positive_cells,
        "objective_moving_codebooks": objective_cells,
        "thresholds": gate,
        "promotion_eligible": False,
    }


def _validate_bound_inputs(capsule: dict[str, Any]) -> None:
    """Reject staged scientific-input drift before importing ML libraries."""

    contract = capsule["contract"]
    science = contract["science_inputs"]
    paths = contract["canonical_paths"]
    bound_files = (
        ("matrix", Path(paths["matrix"]), science["matrix"]["sha256"]),
        (
            "primary_config",
            Path(paths["primary_config"]),
            science["primary_config"]["sha256"],
        ),
        (
            "discovery_manifest",
            Path(paths["discovery_manifest"]),
            science["discovery_manifest"]["sha256"],
        ),
        (
            "tokenizer_terminal",
            Path(paths["tokenizer_terminal"]),
            science["tokenizer_terminal"]["sha256"],
        ),
        (
            "model_manifest",
            Path(paths["model_manifest"]),
            science["model"]["manifest_sha256"],
        ),
    )
    for label, path, digest in bound_files:
        if not path.is_file() or sha256_file(path) != digest:
            raise CapsuleError(f"capsule-bound {label} bytes changed")
    if not Path(paths["task_root"]).is_dir():
        raise CapsuleError("capsule-bound task root is absent")
    entities = {
        (row.lineage, row.role): row for row in frozen_entities(Path(paths["matrix"]))
    }
    entity = entities.get((science["lineage"], science["entity_role"]))
    if (
        entity is None
        or entity.repo != science["model"]["repo"]
        or entity.revision != science["model"]["revision"]
    ):
        raise CapsuleError("capsule-bound fused model identity changed")


def _validate_hydration_evidence(capsule: dict[str, Any]) -> dict[str, Any]:
    path = Path(capsule["contract"]["canonical_paths"]["hydration_evidence"])
    if not path.is_file():
        raise CapsuleError("exact runtime hydration evidence is absent")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema") != "pg1-native-runtime-hydration-evidence/v1"
        or evidence.get("status")
        != "GREEN_EXACT_RUNTIME_BODIES_VERIFIED_BEFORE_SCIENCE"
        or evidence.get("capsule_sha256") != capsule["content_address"]["digest"]
        or evidence.get("attempt_run_id")
        != capsule["contract"]["attempt_identity"]["attempt_run_id"]
        or evidence.get("science_started") is not False
        or evidence.get("confirmation_items_read") != 0
        or not evidence.get("model_files")
        or not evidence.get("task_images")
        or evidence.get("tokenizer_asset_count", 0) < 3
    ):
        raise CapsuleError("exact runtime hydration evidence changed")
    return evidence


def _run_science(
    *,
    capsule: dict[str, Any],
    authorization: dict[str, Any],
    authorization_receipt: dict[str, Any],
    root: Path,
    source_contract: Path,
    matrix: Path,
    config_path: Path,
    tokenizer_preflight: Path,
    cache_root: Path,
    task_root: Path,
    lineage: str,
    lineage_policy: Mapping[str, str],
    instance_id: int,
    output_dir: Path,
) -> dict[str, Any]:
    validate_execution_authorization(authorization, capsule)
    validate_authorization_receipt(authorization_receipt, authorization, capsule)
    contract = validate_full_source(capsule, root)
    if config_path != (root / contract["artifacts"]["primary_config"]["path"]).resolve():
        raise TrainingGateError("native primary config escaped its source contract")
    _validate_primary_config(config_path)
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    started = time.monotonic()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite interface-control result: {output_dir}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise TrainingGateError("interface-control cell requires an H100")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    allowed = lineage_policy
    lineage_rows = {str(row["id"]): row for row in config.get("lineage_order", [])}
    if (
        config.get("status") != "frozen_design_before_tokenizer_preflight"
        or config.get("paid_authorization") is not False
        or allowed is None
        or lineage not in lineage_rows
        or lineage_rows[lineage]["role"] != allowed["role"]
        or config["population"]["confirmation_reads"] != 0
    ):
        raise TrainingGateError("interface-control lineage/design is not exact")
    preflight = json.loads(tokenizer_preflight.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "GREEN_PROCESSOR_ONLY_INTERFACE_GATE"
        or preflight.get("lineage") != lineage
        or preflight.get("model_loads") != 0
        or preflight.get("gpu_operations") != 0
        or preflight.get("confirmation_items_read") != 0
        or len(set(preflight.get("candidate_suffix_tokens", {}).values())) != 1
    ):
        raise TrainingGateError("processor-only interface preflight is absent or not green")
    validate_codebooks()
    bank = DiscoveryTaskBank(task_root / "manifests/items.discovery.jsonl")
    rows = list(bank.iter_rows(task_id=config["population"]["source_task_id"]))
    train_source = select_split_stratified(
        rows,
        task_id=config["population"]["source_task_id"],
        split=config["population"]["train_split"],
        per_target=int(config["population"]["train_per_semantic_target"]),
    )
    discovery_source = select_split_stratified(
        rows,
        task_id=config["population"]["source_task_id"],
        split=config["population"]["discovery_split"],
        per_target=int(config["population"]["discovery_per_semantic_target"]),
    )
    schedule = deterministic_batches(
        item_count=len(train_source),
        batch_size=int(config["optimizer"]["batch_size"]),
        steps=int(config["optimizer"]["steps_per_codebook"]),
        seed=int(config["optimizer"]["seed"]),
    )
    schedule_hash = schedule_sha256(schedule)
    if schedule_hash != config["optimizer"]["exact_same_schedule_sha256_as_registered_seed17"]:
        raise TrainingGateError("interface-control schedule differs from registered seed17")
    snapshots_requested = [int(value) for value in config["optimizer"]["snapshots"]]
    entities = {(row.lineage, row.role): row for row in frozen_entities(matrix)}
    entity = entities[(lineage, allowed["entity_role"])]
    snapshot = resolve_snapshot(cache_root, entity.repo, entity.revision)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    output_dir.mkdir(parents=True)
    cells = []
    initial_frozen_hashes = []
    for codebook_id in CODEBOOKS:
        torch.manual_seed(int(config["optimizer"]["seed"]))
        torch.cuda.manual_seed_all(int(config["optimizer"]["seed"]))
        torch.use_deterministic_algorithms(True)
        train_rows = remap_rows(train_source, codebook_id=codebook_id)
        discovery_rows = remap_rows(discovery_source, codebook_id=codebook_id)
        release_cuda()
        torch.cuda.reset_peak_memory_stats()
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")
        model.config.use_cache = False
        attention = assert_attention_eager(_attention_configs(model))
        anchor_rows = select_split_stratified(
            train_rows,
            task_id=CONTROL_TASK_ID,
            split="train",
            per_target=1,
        )
        anchor_batch = real_training_batch(processor, anchor_rows, task_root)
        baseline_logits = suffix_anchor(model, anchor_batch)
        model, trainable_report = attach_free_lora_connector(model, config["adapter"])
        attached_logits = suffix_anchor(model, anchor_batch)
        if not all(torch.equal(left, right) for left, right in zip(baseline_logits, attached_logits)):
            raise TrainingGateError(f"zero-delta adapter attachment failed: {codebook_id}")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(lineage_rows[lineage]["optimizer_learning_rate"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
        )
        frozen_initial = named_parameter_sha256(model, trainable=False)
        initial_frozen_hashes.append(frozen_initial)
        cell_dir = output_dir / codebook_id
        cell_dir.mkdir()
        snapshot_records = []
        update_records = []

        def bank_snapshot(step: int) -> None:
            checkpoint = cell_dir / f"step-{step:06d}.pt"
            checkpoint_sha = atomic_training_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                step=step,
                schedule_sha256=schedule_hash,
                metadata={
                    "lineage": lineage,
                    "codebook_id": codebook_id,
                    "interface_control": True,
                },
            )
            evaluation = evaluate_candidates(
                model=model,
                processor=processor,
                rows=discovery_rows,
                task_root=task_root,
                batch_size=4,
            )
            interface_audit = audit_equal_suffix_counts(evaluation["candidate_rows"])
            sum_semantic = semantic_summary(evaluation["items"], discovery_rows)
            mean_items = _mean_score_items(evaluation["candidate_rows"])
            mean_semantic = semantic_summary(mean_items, discovery_rows)
            frozen_hash = named_parameter_sha256(model, trainable=False)
            if frozen_hash != frozen_initial:
                raise TrainingGateError(f"frozen bytes changed: {codebook_id} step {step}")
            snapshot_records.append(
                {
                    "step": step,
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": checkpoint_sha,
                        "bytes": checkpoint.stat().st_size,
                    },
                    "objective": _objective(model, anchor_batch),
                    "evaluation": evaluation,
                    "interface_audit": interface_audit,
                    "semantic_summary_sum": sum_semantic,
                    "semantic_summary_mean": mean_semantic,
                    "trainable_sha256": named_parameter_sha256(model, trainable=True),
                    "optimizer_sha256": optimizer_state_sha256(optimizer),
                    "frozen_sha256": frozen_hash,
                }
            )

        bank_snapshot(0)
        for step, indices in enumerate(schedule, start=1):
            batch_rows = [train_rows[index] for index in indices]
            batch = real_training_batch(processor, batch_rows, task_root)
            update = one_update_per_example(model=model, optimizer=optimizer, batch=batch)
            update_records.append(
                {"step": step, "item_ids": [row["item_id"] for row in batch_rows], **update}
            )
            del batch
            if step in snapshots_requested:
                bank_snapshot(step)
        if [row["step"] for row in snapshot_records] != snapshots_requested:
            raise TrainingGateError(f"snapshot schedule changed: {codebook_id}")
        cell = {
            "codebook_id": codebook_id,
            "mapping": CODEBOOKS[codebook_id],
            "attention": attention,
            "trainable_contract": trainable_report,
            "train_item_ids": [row["item_id"] for row in train_rows],
            "discovery_item_ids": [row["item_id"] for row in discovery_rows],
            "updates": update_records,
            "snapshots": snapshot_records,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
        cell["adjudication"] = _cell_adjudication(
            snapshot_records,
            float(
                config["provisional_mechanical_adjudication"][
                    "minimum_objective_drop_per_passing_codebook"
                ]
            ),
        )
        cells.append(cell)
        del model, optimizer, anchor_batch, baseline_logits, attached_logits
        release_cuda()
    if len(set(initial_frozen_hashes)) != 1:
        raise TrainingGateError("codebook cells did not reset to the same exact fused base")
    adjudication = _aggregate_adjudication(cells, config)
    packet = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "stage": "training_interface_control",
        "status": "GREEN_MECHANICAL_EXECUTION",
        "label": "INTERFACE_CONTROL_PROVISIONAL_NOT_PROMOTION_ELIGIBLE",
        "provider": "vast.ai",
        "instance_id": instance_id,
        "gpu": torch.cuda.get_device_name(0),
        "lineage": lineage,
        "role": allowed["role"],
        "checkpoint": {"repo": entity.repo, "revision": entity.revision},
        "tokenizer_preflight": {
            "path": str(tokenizer_preflight),
            "sha256": sha256_file(tokenizer_preflight),
            "candidate_suffix_tokens": preflight["candidate_suffix_tokens"],
        },
        "schedule": {"sha256": schedule_hash, "batches": schedule},
        "codebooks": cells,
        "fused_base_reset_exact": True,
        "initial_frozen_sha256": initial_frozen_hashes[0],
        "adjudication": adjudication,
        "confirmation_items_read": 0,
        "promotion_eligible": False,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "checkpoint_bytes": sum(
                row["checkpoint"]["bytes"]
                for cell in cells
                for row in cell["snapshots"]
            ),
        },
        "source_contract": {"path": str(source_contract), "sha256": sha256_file(source_contract)},
        "claim_boundary": config["claim_boundary"],
    }
    atomic_json(output_dir / "interface-control.json", packet)
    del processor
    release_cuda()
    return packet


def run(
    *,
    capsule: dict[str, Any],
    authorization: dict[str, Any],
    authorization_receipt: dict[str, Any],
) -> dict[str, Any]:
    from .pg1_native_authority import ProductionAuthorizationStore

    validate_consumer(capsule, "cell", Path(__file__).resolve())
    paths = capsule["contract"]["canonical_paths"]
    root = Path(paths["staged_root"])
    source_contract = Path(paths["source_contract"])
    matrix = Path(paths["matrix"])
    config_path = Path(paths["primary_config"])
    tokenizer_preflight = Path(paths["tokenizer_evidence"])
    cache_root = Path(paths["cache_root"])
    task_root = Path(paths["task_root"])
    output_dir = Path(paths["output_dir"])
    validate_execution_authorization(authorization, capsule)
    validate_authorization_receipt(authorization_receipt, authorization, capsule)
    ProductionAuthorizationStore().consume_owner(
        capsule, authorization, authorization_receipt, owner="cell"
    )
    _validate_bound_inputs(capsule)
    _validate_hydration_evidence(capsule)
    capsule_preflight(
        capsule,
        root=root,
        source_contract=source_contract,
        config_path=config_path,
    )
    instance_id = authorization["contract"]["instance_id"]
    packet = _run_science(
        capsule=capsule,
        authorization=authorization,
        authorization_receipt=authorization_receipt,
        root=root,
        source_contract=source_contract,
        matrix=matrix,
        config_path=config_path,
        tokenizer_preflight=tokenizer_preflight,
        cache_root=cache_root,
        task_root=task_root,
        lineage=LINEAGE,
        lineage_policy={"role": LINEAGE_ROLE, "entity_role": ENTITY_ROLE},
        instance_id=instance_id,
        output_dir=output_dir,
    )
    result = adjudicate_primary_packet(packet)
    atomic_json(output_dir / "interface-control.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-receipt", type=Path)
    parser.add_argument("--staged-root", type=Path)
    args = parser.parse_args()
    capsule = load_capsule(args.capsule)
    if args.capsule_preflight:
        if args.staged_root is None:
            raise CapsuleError("native cell preflight requires staged root")
        root = args.staged_root.resolve()
        result = capsule_preflight(
            capsule,
            root=root,
            source_contract=root / SOURCE_CONTRACT_PATH,
            config_path=root / PRIMARY_CONFIG_PATH,
        )
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    if not args.execute or args.authorization is None or args.authorization_receipt is None:
        raise CapsuleError("native cell execution arguments are incomplete")
    result = run(
        capsule=capsule,
        authorization=json.loads(args.authorization.read_text(encoding="utf-8")),
        authorization_receipt=json.loads(
            args.authorization_receipt.read_text(encoding="utf-8")
        ),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "verdict": result["adjudication"]["verdict"],
                "lineage": result["lineage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
