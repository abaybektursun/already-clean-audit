"""Adjudicate paired PARALLAX G1 packets without crossing the Gemma firewall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_json, sha256_file
from .g1_common import G1GateError, assert_comparison_legal, utc_now


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "GREEN":
        raise G1GateError(f"G1 input is not green: {path}")
    return payload


def _case(packet: dict[str, Any], model: str, case: str) -> dict[str, Any]:
    return packet["outputs"][model]["cases"][case]


def _same_sample(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["shape"] == right["shape"]
        and left["sample_values"] == right["sample_values"]
        and left["sample_sha256_float32"] == right["sample_sha256_float32"]
    )


def _identity_rows(packet: dict[str, Any]):
    for model in packet["outputs"].values():
        if "identity" in model:
            yield model["identity"]
        for case in model.get("cases", {}).values():
            if "identity" in case:
                yield case["identity"]


def adjudicate(primary: dict[str, Any], pi: dict[str, Any], supporting: dict[str, Any]) -> dict[str, Any]:
    if primary.get("lineage") != "pg1_pi" or primary.get("runtime_role") != "core":
        raise G1GateError("primary core packet identity mismatch")
    if pi.get("lineage") != "pg1_pi" or pi.get("runtime_role") != "pi_sidecar":
        raise G1GateError("π sidecar packet identity mismatch")
    if supporting.get("lineage") != "pg2" or supporting.get("runtime_role") != "core":
        raise G1GateError("supporting core packet identity mismatch")
    if any(row.get("provider") != "vast.ai" for row in (primary, pi, supporting)):
        raise G1GateError("G1 packets must come only from Vast.ai")
    if len({row.get("instance_id") for row in (primary, pi, supporting)}) != 1:
        raise G1GateError("paired G1 packets must be bound to one H100 instance")
    if set(primary.get("outputs", {})) != {"siglip", "language_parent", "fused"}:
        raise G1GateError("primary G1 packet is incomplete")
    if set(supporting.get("outputs", {})) != {"siglip", "language_parent", "fused", "mix"}:
        raise G1GateError("supporting G1 packet is incomplete")
    if set(pi.get("outputs", {})) != {"pi0", "pi05"} or pi.get("pg2_imported") is not False:
        raise G1GateError("π G1 packet is incomplete or crossed the PG2 firewall")
    hashes = {row["anchor_config_sha256"] for row in (primary, pi, supporting)}
    gates = {row["cache_gate_manifest_sha256"] for row in (primary, pi, supporting)}
    if len(hashes) != 1 or len(gates) != 1:
        raise G1GateError("G1 packets do not share one anchor/cache contract")

    assert_comparison_legal("pg1_pi", "pg2", "siglip_pixels")
    assert_comparison_legal("pg1_pi", "pg2", "siglip_patch_geometry")
    for forbidden in ("language_logits", "language_weights", "raw_language_depth"):
        try:
            assert_comparison_legal("pg1_pi", "pg2", forbidden)
        except G1GateError:
            pass
        else:
            raise AssertionError(f"firewall unexpectedly licensed {forbidden}")

    checks: dict[str, bool] = {}
    for case in ("blank", "real"):
        p_siglip = _case(primary, "siglip", case)
        s_siglip = _case(supporting, "siglip", case)
        checks[f"shared_siglip_pixels_{case}"] = _same_sample(
            p_siglip["input"]["pixels"], s_siglip["input"]["pixels"]
        )
        checks[f"shared_siglip_anchor_{case}"] = _same_sample(p_siglip["anchor"], s_siglip["anchor"])
        p_fused = _case(primary, "fused", case)
        s_pt = _case(supporting, "fused", case)
        s_mix = _case(supporting, "mix", case)
        checks[f"shared_processor_pixels_pg1_pg2pt_{case}"] = _same_sample(
            p_fused["input"]["pixels"], s_pt["input"]["pixels"]
        )
        checks[f"pg2_pt_mix_pixels_{case}"] = _same_sample(
            s_pt["input"]["pixels"], s_mix["input"]["pixels"]
        )
        checks[f"pg2_pt_mix_tokens_{case}"] = s_pt["token_ids_sha256"] == s_mix["token_ids_sha256"]
        checks[f"all_image_token_counts_256_{case}"] = all(
            row["input"]["image_token_count"] == 256 for row in (p_fused, s_pt, s_mix)
        )
    checks["shared_patch_geometry"] = all(
        row["outputs"]["fused"]["vision_patch_size"] == 14
        and row["outputs"]["fused"]["vision_num_image_tokens"] == 256
        for row in (primary, supporting)
    )
    checks["pi_three_camera_streams"] = all(
        case["camera_streams"] == 3
        for policy in pi["outputs"].values()
        for case in policy["cases"].values()
    )
    checks["pi_native_cache_disabled"] = all(not policy["use_cache"] for policy in pi["outputs"].values())
    checks["pi_strict_load"] = all(
        policy["load"]["missing_keys"] == 0 and policy["load"]["unexpected_keys"] == 0
        for policy in pi["outputs"].values()
    )
    core_identities = [row for packet in (primary, supporting) for row in _identity_rows(packet)]
    checks["identity_hook_count"] = len(core_identities) == 12
    checks["identity_hooks_all_green"] = all(row["status"] == "GREEN" for row in core_identities) and all(
        case["identity"]["status"] == "GREEN"
        for policy in pi["outputs"].values()
        for case in policy["cases"].values()
    )
    status = "GREEN" if all(checks.values()) else "NO_GO"
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": status,
        "hour_6_decision": "GO" if status == "GREEN" else "NO_GO",
        "checks": checks,
        "lineage_claims": {
            "primary": "SigLIP+Gemma1_to_PaliGemma1_to_pi0_with_pi05_optional_extension",
            "supporting": "SigLIP+Gemma2_to_PaliGemma2_pt_to_PaliGemma2_mix",
            "single_chain_across_both": False,
            "direct_cross_gemma_language_comparison_performed": False,
            "shared_siglip_bridge_licensed": status == "GREEN",
        },
        "comparison_scope": {
            "shared": ["pixel_preprocessing", "patch_geometry", "standalone_siglip_anchor"],
            "within_lineage_only": ["language_parent", "fused_language", "later_checkpoint"],
            "cross_lineage_future": "normalized_verdicts_only",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-core", type=Path, required=True)
    parser.add_argument("--pi-sidecar", type=Path, required=True)
    parser.add_argument("--supporting-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = adjudicate(_load(args.primary_core), _load(args.pi_sidecar), _load(args.supporting_core))
    packet["inputs"] = {
        "primary_core": {"path": str(args.primary_core), "sha256": sha256_file(args.primary_core)},
        "pi_sidecar": {"path": str(args.pi_sidecar), "sha256": sha256_file(args.pi_sidecar)},
        "supporting_core": {"path": str(args.supporting_core), "sha256": sha256_file(args.supporting_core)},
    }
    atomic_json(args.output, packet)
    print(json.dumps({"status": packet["status"], "decision": packet["hour_6_decision"], "output": str(args.output)}))
    return 0 if packet["status"] == "GREEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
