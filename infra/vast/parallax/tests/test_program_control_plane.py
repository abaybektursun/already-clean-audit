from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from infra.vast.parallax import program_control as control


ROOT = Path(__file__).resolve().parents[4]
AMENDMENT = ROOT / control.AMENDMENT_PATH
OVERLAY = ROOT / control.LAUNCH_OVERLAY_PATH


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_frozen_historical_control_files_remain_byte_identical() -> None:
    expected = {
        "studies/lineage/cross-lineage/research/2026-07-15_parallax_spec.md":
            "61ac281a9b0b16496651b2166b0494e49e7f73fd890e0625d9d79e4e20f67f6e",
        "studies/lineage/cross-lineage/research/2026-07-15_parallax_design_audit.md":
            "67032269b4f9b4cc18d25f3e7b0c3d2dfb5abc0adf1722cb8976a46c6f8bd9ba",
        "studies/lineage/cross-lineage/config/parallax_matrix.yaml":
            "9df29f6e559f1d2d12651dec3623faa1d515139504e69ef5b1bcfaff325db91c",
        "studies/lineage/cross-lineage/config/parallax_launch_packet.template.yaml":
            "ad72548f2d10d9febeb8cbc88f30e25697da86e4a2b842bfb84c4ce44e8cb0e2",
    }
    for relative, digest in expected.items():
        assert control.sha256_file(ROOT / relative) == digest


def test_effective_policy_is_s3_only_ephemeral_and_unpaid() -> None:
    result = control.validate_local(ROOT)

    assert result["status"] == "GREEN_EFFECTIVE_PROGRAM_CONTROL"
    assert result["historical_inputs_verified"] == 5
    assert result["scientific_design_changed"] is False
    assert result["paid_action"] is False
    assert result["offer_selection"] == "literal_cheapest_eligible_all_regions_at_mutation_time"
    assert result["scratch"]["fixed_minimum"] == "forbidden"
    assert result["scratch"]["current_rehearsed_range_cell_gb"] == 80
    assert result["durability"] == {
        "sole_system_of_record": "amazon_s3",
        "aws_profile": "fuelos",
        "region": "us-east-1",
        "vast_persistent_storage": "forbidden",
    }
    assert result["automatic_paid_retry"] is False


def test_effective_policy_rejects_hash_rewriting_of_frozen_history() -> None:
    amendment = _load(AMENDMENT)
    overlay = _load(OVERLAY)
    amendment["frozen_historical_inputs"]["matrix"]["sha256"] = "0" * 64

    with pytest.raises(control.ProgramControlError, match="historical matrix hash mismatch"):
        control.validate_policy_data(ROOT, amendment, overlay)


def test_effective_policy_rejects_paid_or_persistent_storage_regression() -> None:
    amendment = _load(AMENDMENT)
    overlay = _load(OVERLAY)
    bad_paid = deepcopy(amendment)
    bad_paid["authorization"]["paid_action"] = True
    with pytest.raises(control.ProgramControlError, match="authorizes paid"):
        control.validate_policy_data(ROOT, bad_paid, overlay)

    bad_storage = deepcopy(amendment)
    bad_storage["durability"]["vast_persistent_volumes"] = "required"
    with pytest.raises(control.ProgramControlError, match="durability policy"):
        control.validate_policy_data(ROOT, bad_storage, overlay)


def test_effective_policy_rejects_retired_gateway_lifecycle() -> None:
    amendment = _load(AMENDMENT)
    overlay = _load(OVERLAY)
    overlay["lifecycle"]["continuous_mirror"] = {
        "status": "REQUIRED_GREEN",
        "evidence": "stale",
    }

    with pytest.raises(control.ProgramControlError, match="retired lifecycle"):
        control.validate_policy_data(ROOT, amendment, overlay)
