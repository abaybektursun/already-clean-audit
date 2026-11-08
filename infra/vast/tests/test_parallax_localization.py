from __future__ import annotations

import torch
import json
from pathlib import Path

import yaml

from infra.vast.parallax.behavioral_common import make_field
from infra.vast.parallax.audit_localization_panel import audit
from infra.vast.parallax.audit_localization_rescue_panel import audit as audit_rescue
from infra.vast.parallax.behavioral_localization import (
    adjudicate_localization,
    equal_total_energy_rms,
    prefix_location_masks,
    select_localization_amplitude,
    total_energy_audit,
)
from infra.vast.parallax.materialize_localization_panel import materialize
from infra.vast.parallax.localization_preflight import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]


def test_location_masks_exactly_partition_prefix_and_exclude_suffix_padding() -> None:
    inputs = {
        "input_ids": torch.tensor([[99, 99, 3, 4, 5, 0], [99, 99, 7, 8, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, -100, 4, 5, -100], [-100, -100, -100, 8, -100, -100]]),
    }
    masks = prefix_location_masks(inputs, image_token_index=99, expected_image_tokens=2)
    assert torch.equal(masks["image_tokens"], torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool))
    assert torch.equal(masks["prompt_tokens"], torch.tensor([[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=torch.bool))
    assert torch.equal(masks["full_prefix"], masks["image_tokens"] | masks["prompt_tokens"])


def test_location_fields_match_full_prefix_total_energy_after_bfloat16_cast() -> None:
    full = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]], dtype=torch.float32)
    image = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    rms = torch.tensor([0.8, 1.2])
    scaled = equal_total_energy_rms(rms, reference_mask=full, location_mask=image)
    field = make_field(
        construction="sensitive_coherent",
        sensitive_direction=torch.arange(1, 65, dtype=torch.float32),
        active_mask=image,
        activation_rms=scaled,
        amplitude=0.003,
        namespace="localization-test",
        lineage="pg2",
        item_ids=["one", "two"],
    ).to(torch.bfloat16)
    audit = total_energy_audit(
        field,
        location_mask=image,
        reference_mask=full,
        activation_rms=rms,
        amplitude=0.003,
        relative_tolerance=0.005,
    )
    assert audit["status"] == "GREEN"
    assert audit["location_tokens"] == [3, 2]
    assert audit["reference_tokens"] == [5, 4]


def _cells(full: float, image: float, prompt: float, n: int = 16):
    zero = {str(i): 0.0 for i in range(n)}
    return {
        "full_sensitive": {str(i): full for i in range(n)},
        "full_rotated": zero,
        "full_white": zero,
        "image_sensitive": {str(i): image for i in range(n)},
        "image_rotated": zero,
        "prompt_sensitive": {str(i): prompt for i in range(n)},
        "prompt_rotated": zero,
    }


def test_localization_adjudicator_exercises_decisive_and_void_worlds() -> None:
    kwargs = {"frozen_mde": 0.05, "seed": 7, "draws": 500, "lower_quantile": 0.0125}
    assert adjudicate_localization(_cells(0.2, 0.2, 0.01), **kwargs)["status"] == "IMAGE_TOKEN_POSITION_DOMINANT"
    assert adjudicate_localization(_cells(0.2, 0.01, 0.2), **kwargs)["status"] == "PROMPT_POSITION_DOMINANT"
    assert adjudicate_localization(_cells(0.2, 0.2, 0.2), **kwargs)["status"] == "MIXED_POSITION_RANGE"
    assert adjudicate_localization(_cells(0.01, 0.2, 0.01), **kwargs)["status"] == "VOID_NO_FULL_PREFIX_REPLICATION"


def test_full_prefix_replication_must_beat_both_registered_nulls() -> None:
    cells = _cells(0.2, 0.2, 0.01)
    cells["full_white"] = {str(i): 0.19 for i in range(16)}
    result = adjudicate_localization(
        cells, frozen_mde=0.05, seed=7, draws=500, lower_quantile=0.0125
    )
    assert result["status"] == "VOID_NO_FULL_PREFIX_REPLICATION"
    assert not result["full_prefix_replication_green"]


def test_localization_amplitude_requires_full_readability_and_all_location_safety() -> None:
    rows = []
    for amplitude in (0.0003, 0.001, 0.003):
        for arm in ("full_sensitive", "image_sensitive", "prompt_sensitive"):
            median = {0.0003: 0.0002, 0.001: 0.005, 0.003: 0.02}[amplitude]
            fraction = amplitude * (3.5 if arm == "prompt_sensitive" else 1.0)
            rows.extend(
                {
                    "amplitude": amplitude,
                    "arm": arm,
                    "kl_nats": median,
                    "per_token_activation_rms_fraction": fraction,
                }
                for _ in range(8)
            )
    result = select_localization_amplitude(
        rows,
        [0.0003, 0.001, 0.003],
        median_kl_min=0.001,
        median_kl_max=0.05,
        p90_kl_max=0.2,
        per_token_fraction_max=0.005,
    )
    assert result["status"] == "GREEN"
    assert result["selected_amplitude"] == 0.001
    assert not result["diagnostics"]["0.003"]["usable"]


def test_missing_unsafe_location_cell_marks_rung_ineligible_without_aborting_ladder() -> None:
    rows = []
    for amplitude in (0.0003, 0.001, 0.003):
        for arm in ("full_sensitive", "image_sensitive", "prompt_sensitive"):
            if amplitude == 0.003 and arm == "prompt_sensitive":
                continue
            rows.extend(
                {
                    "amplitude": amplitude,
                    "arm": arm,
                    "kl_nats": 0.005 if amplitude >= 0.001 else 0.0002,
                    "per_token_activation_rms_fraction": amplitude,
                }
                for _ in range(8)
            )
    result = select_localization_amplitude(
        rows,
        [0.0003, 0.001, 0.003],
        median_kl_min=0.001,
        median_kl_max=0.05,
        p90_kl_max=0.2,
        per_token_fraction_max=0.005,
    )
    assert result["selected_amplitude"] == 0.001
    skipped = result["diagnostics"]["0.003"]
    assert not skipped["all_location_cells_evaluated"]
    assert skipped["arms"]["prompt_sensitive"]["status"] == "SKIPPED_UNSAFE_OR_MISSING"


def test_new_localization_panel_is_balanced_disjoint_and_content_hashed(tmp_path: Path) -> None:
    config = ROOT / "studies/lineage/cross-lineage/config/parallax_pg2_localization_v1.yaml"
    panel_root = tmp_path / "panel"
    content = materialize(config, panel_root)
    report = audit(panel_root)
    assert content["status"] == "GREEN_MATERIALIZED"
    assert report["status"] == "GREEN"
    assert report["split_counts"] == {"confirmation_eval": 32, "discovery_eval": 16}
    assert report["cross_split_image_hashes"] == 0
    assert report["cross_split_source_identities"] == 0


def test_localization_calibration_and_freeze_cannot_access_confirmation_items() -> None:
    package = ROOT / "infra/vast/parallax"
    calibration = (package / "localization_calibrate.py").read_text(encoding="utf-8")
    freeze = (package / "localization_freeze_confirmation.py").read_text(encoding="utf-8")
    assert "iter_confirmation" not in calibration
    assert "bank.iter_discovery" in calibration
    assert '"confirmation_metadata_rows_parsed_by_calibration": 0' in calibration
    assert '"confirmation_unique_items_evaluated_by_calibration": 0' in calibration
    assert "items.jsonl" not in freeze
    assert "iter_discovery" not in freeze
    assert "iter_confirmation" not in freeze


def test_localization_confirmation_is_bound_to_all_scientific_sources() -> None:
    package = ROOT / "infra/vast/parallax"
    confirmation = (package / "localization_confirm.py").read_text(encoding="utf-8")
    freezer = (package / "localization_freeze_confirmation.py").read_text(encoding="utf-8")
    assert "iter_discovery" not in confirmation
    assert "iter_confirmation" in confirmation
    assert "SOURCE_FILES" in confirmation
    for name in (
        "artifacts.py",
        "g1_common.py",
        "parallax_entities.py",
        "parallax_tasks.py",
        "materialize_localization_panel.py",
        "audit_localization_panel.py",
        "localization_calibrate.py",
        "localization_confirm.py",
        "localization_freeze_confirmation.py",
        "behavioral_localization.py",
        "behavioral_localization_model.py",
        "behavioral_model.py",
        "behavioral_common.py",
    ):
        assert name in freezer


def test_localization_gateway_rehearsal_has_no_scientific_panel_access() -> None:
    source = (
        ROOT / "infra/vast/parallax/localization_gateway_rehearsal.py"
    ).read_text(encoding="utf-8")
    assert "FrozenTaskBank" not in source
    assert "iter_discovery" not in source
    assert "iter_confirmation" not in source
    assert '"discovery_items_read": 0' in source
    assert '"confirmation_metadata_rows_parsed": 0' in source
    assert '"confirmation_items_evaluated": 0' in source


def test_localization_preflight_is_pg2_only_no_science_and_instance_bound() -> None:
    source = (ROOT / "infra/vast/parallax/localization_preflight.py").read_text(encoding="utf-8")
    assert "MAX_PREFLIGHT_MINUTES = 20" in source
    assert '"mode": "NO_SCIENCE_PG2_LOCALIZATION_PREFLIGHT"' in source
    assert '"destination_instance_id": instance_id' in source
    assert '"science_items_evaluated": 0' in source
    assert '"confirmation_items_evaluated": 0' in source
    assert "len(repositories) != 1" in source
    assert "snapshot_download" not in source
    assert "huggingface_hub" not in source


def test_localization_source_and_cache_contracts_are_exact() -> None:
    config_root = ROOT / "studies/lineage/cross-lineage/config"
    source = json.loads(
        (config_root / "parallax_pg2_localization_source_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for row in source["entries"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        import hashlib

        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    assert source["runtime_archive"] == {
        "bytes": 3_017_704_443,
        "sha256": "1894ac562f36af701e1918566c283f0a5601a6110813dd76e79e236f7ea0471b",
    }
    cache = json.loads(
        (config_root / "parallax_pg2_localization_cache_contract_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert cache["content_manifest_sha256"] == canonical_sha256(
        {"repositories": cache["repositories"]}
    )
    assert [row["repo"] for row in cache["repositories"]] == [
        "google/paligemma2-3b-pt-224"
    ]
    assert sum(row["bytes"] for row in cache["repositories"][0]["files"]) == 6_099_504_596


def test_localization_rescue_is_fixed_one_shot_and_conservative() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/parallax_pg2_localization_rescue_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["status"] == "frozen_before_new_panel_materialization"
    assert config["energy"]["fixed_reference_amplitude"] == 0.001
    assert config["energy"]["amplitude_selection_or_ladder"] == "forbidden"
    assert config["mde"]["fixed_nats"] == 0.021652735769748688
    assert not config["mde"]["may_be_reestimated"]
    assert config["branch_rules"]["any_second_rescue"] == "forbidden"
    assert config["scope"]["pg1_site_remains_canceled"]
    assert config["scope"]["pi_and_pi05_implications"] == "none"


def test_localization_rescue_prerequisites_are_content_addressed() -> None:
    import hashlib

    config = yaml.safe_load(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/parallax_pg2_localization_rescue_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    for name in ("behavioral_confirmation", "localization_calibration", "localization_void"):
        row = config["prerequisites"][name]
        path = ROOT / row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_localization_rescue_panel_is_fresh_relative_to_v1(tmp_path: Path) -> None:
    original_config = (
        ROOT / "studies/lineage/cross-lineage/config/parallax_pg2_localization_v1.yaml"
    )
    rescue_config = (
        ROOT
        / "studies/lineage/cross-lineage/config/parallax_pg2_localization_rescue_v1.yaml"
    )
    original_root = tmp_path / "original"
    rescue_root = tmp_path / "rescue"
    materialize(original_config, original_root)
    materialize(rescue_config, rescue_root)
    assert audit(rescue_root)["status"] == "GREEN"
    rescue_audit = audit_rescue(rescue_root, original_root)
    assert rescue_audit["status"] == "GREEN"
    assert rescue_audit["consumed_v1_overlap"] == {
        "item_ids": 0,
        "image_hashes": 0,
        "source_identities": 0,
    }

    def rows(root: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in (root / "manifests/items.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    original = rows(original_root)
    rescue = rows(rescue_root)
    assert {row["item_id"] for row in original}.isdisjoint(
        {row["item_id"] for row in rescue}
    )
    assert {row["image_sha256"] for row in original}.isdisjoint(
        {row["image_sha256"] for row in rescue}
    )
    assert {
        json.dumps(row["source_identity"], sort_keys=True) for row in original
    }.isdisjoint({json.dumps(row["source_identity"], sort_keys=True) for row in rescue})


def test_localization_rescue_freezer_has_no_item_level_access() -> None:
    package = ROOT / "infra/vast/parallax"
    freezer = (package / "localization_rescue_freeze.py").read_text(encoding="utf-8")
    confirmation = (package / "localization_rescue_confirm.py").read_text(encoding="utf-8")
    assert "items.jsonl" not in freezer
    assert "iter_discovery" not in freezer
    assert "iter_confirmation" not in freezer
    assert "iter_discovery" not in confirmation
    assert "iter_confirmation" in confirmation
    assert '== 0.001' in confirmation
    assert '== 0.021652735769748688' in confirmation


def test_localization_rescue_confirmation_binds_every_scientific_source() -> None:
    package = ROOT / "infra/vast/parallax"
    freezer = (package / "localization_rescue_freeze.py").read_text(encoding="utf-8")
    confirmation = (package / "localization_rescue_confirm.py").read_text(encoding="utf-8")
    assert "SOURCE_FILES" in confirmation
    for name in (
        "materialize_localization_panel.py",
        "audit_localization_panel.py",
        "audit_localization_rescue_panel.py",
        "localization_rescue_confirm.py",
        "localization_rescue_freeze.py",
        "behavioral_localization.py",
        "behavioral_localization_model.py",
        "behavioral_model.py",
        "behavioral_common.py",
    ):
        assert name in freezer


def test_localization_rescue_rehearsal_is_fixed_0001_and_has_no_panel_access() -> None:
    source = (
        ROOT / "infra/vast/parallax/localization_rescue_gateway_rehearsal.py"
    ).read_text(encoding="utf-8")
    assert "FrozenTaskBank" not in source
    assert "iter_discovery" not in source
    assert "iter_confirmation" not in source
    assert "amplitudes=[0.001]" in source
    assert '"discovery_items_read": 0' in source
    assert '"confirmation_items_evaluated": 0' in source


def test_localization_rescue_preflight_is_zero_science_and_rejects_overlap() -> None:
    source = (
        ROOT / "infra/vast/parallax/localization_rescue_preflight.py"
    ).read_text(encoding="utf-8")
    assert '"NO_SCIENCE_PG2_LOCALIZATION_RESCUE_PREFLIGHT"' in source
    assert '!= 0.001' in source
    assert '"consumed_v1_overlap"' in source
    assert '"confirmation_items_evaluated"' not in source


def test_localization_rescue_source_contract_is_exact() -> None:
    import hashlib

    contract = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/"
            "parallax_pg2_localization_rescue_source_contract_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["stage"] == "pg2_prefix_position_localization_rescue"
    for row in contract["entries"]:
        path = ROOT / row["path"]
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    paths = {row["path"] for row in contract["entries"]}
    assert "infra/vast/parallax/localization_rescue_confirm.py" in paths
    assert "infra/vast/parallax/localization_rescue_preflight.py" in paths
    assert (
        "studies/lineage/cross-lineage/config/"
        "parallax_pg2_localization_rescue_confirmation_freeze_v1.json"
    ) in paths
