"""Weight-free checkpoint genealogy and shell-contract audit.

This downloads only model metadata and ``config.json`` files at pinned commits.
It can establish structural compatibility, but it deliberately cannot promote
an initialization claim: exact ancestry remains UNPROVEN until the H100 stage
performs key-level and weight-level tests.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, hf_hub_download

from .artifacts import atomic_json


SIGNATURE_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "image_size",
    "patch_size",
)


def signature(config: dict[str, Any]) -> dict[str, Any]:
    return {field: config.get(field) for field in SIGNATURE_FIELDS if field in config}


def _load(repo: str, revision: str) -> tuple[dict[str, Any], dict[str, Any]]:
    info = HfApi().model_info(repo, revision=revision, files_metadata=False)
    if info.sha != revision:
        raise ValueError(f"revision drift for {repo}: requested={revision} resolved={info.sha}")
    path = hf_hub_download(repo, "config.json", revision=revision)
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = {
        "repo": repo,
        "revision": revision,
        "gated": info.gated,
        "private": info.private,
        "config": signature(config),
    }
    if isinstance(config.get("vision_config"), dict):
        metadata["vision_config"] = signature(config["vision_config"])
    if isinstance(config.get("text_config"), dict):
        metadata["text_config"] = signature(config["text_config"])
    for field in (
        "type",
        "paligemma_variant",
        "action_expert_variant",
        "image_resolution",
        "tokenizer_max_length",
        "chunk_size",
        "n_action_steps",
        "dtype",
    ):
        if field in config:
            metadata[field] = config[field]
    return config, metadata


def _same_except(
    left: dict[str, Any], right: dict[str, Any], ignored: set[str]
) -> tuple[bool, dict[str, list[Any]]]:
    keys = (set(left) | set(right)) - ignored
    differences = {
        key: [left.get(key), right.get(key)]
        for key in sorted(keys)
        if left.get(key) != right.get(key)
    }
    return not differences, differences


def audit(matrix_path: Path) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    loaded: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for lineage_name, lineage in matrix["lineages"].items():
        for role in ("vision_parent_candidate", "language_parent", "fused"):
            item = lineage[role]
            key = f"{lineage_name}.{role}"
            configs[key], loaded[key] = _load(item["repo"], item["revision"])
        for index, item in enumerate(lineage.get("later", [])):
            key = f"{lineage_name}.later[{index}]"
            configs[key], loaded[key] = _load(item["repo"], item["revision"])

    siglip = signature(configs["pg1_pi.vision_parent_candidate"]["vision_config"])
    checks: dict[str, Any] = {}
    for lineage_name in ("pg1_pi", "pg2"):
        fused = configs[f"{lineage_name}.fused"]
        vision = signature(fused["vision_config"])
        vision_green, vision_diff = _same_except(
            siglip,
            vision,
            {"projection_dim", "vocab_size"},
        )
        language = signature(configs[f"{lineage_name}.language_parent"])
        fused_text = signature(fused["text_config"])
        language_green, language_diff = _same_except(
            language,
            fused_text,
            {"vocab_size"},
        )
        checks[lineage_name] = {
            "siglip_to_paligemma_shell": {
                "status": "GREEN_CONFIG_ONLY" if vision_green else "RED",
                "differences_excluding_connector_projection": vision_diff,
                "exact_weight_ancestry": "UNPROVEN",
            },
            "gemma_to_paligemma_shell": {
                "status": "GREEN_CONFIG_ONLY" if language_green else "RED",
                "differences_excluding_vocab_expansion": language_diff,
                "parent_vocab_size": language.get("vocab_size"),
                "fused_vocab_size": fused_text.get("vocab_size"),
                "exact_weight_ancestry": "UNPROVEN",
            },
        }

    pi_rows = [configs["pg1_pi.later[0]"], configs["pg1_pi.later[1]"]]
    checks["pg1_pi"]["paligemma_to_pi_declared_shell"] = {
        "status": (
            "GREEN_CONFIG_ONLY"
            if all(row.get("paligemma_variant") == "gemma_2b" for row in pi_rows)
            else "RED"
        ),
        "pi0_paligemma_variant": pi_rows[0].get("paligemma_variant"),
        "pi05_paligemma_variant": pi_rows[1].get("paligemma_variant"),
        "strict_key_and_forward_parity": "UNPROVEN",
    }

    all_config_green = all(
        check["status"] == "GREEN_CONFIG_ONLY"
        for lineage in checks.values()
        for check in lineage.values()
    )
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "metadata_and_config_only_no_model_weights",
        "matrix": str(matrix_path),
        "resolved_models": loaded,
        "checks": checks,
        "verdict": "GREEN_CONFIG_ONLY_WEIGHTS_UNPROVEN" if all_config_green else "NO_GO",
        "claim_boundary": (
            "Config compatibility supports spending the gated weight-level preflight; "
            "it does not establish either SigLIP+Gemma initialization jump or PG1-to-pi parity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.matrix)
    atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "verdict": report["verdict"]}))
    return 0 if report["verdict"].startswith("GREEN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
