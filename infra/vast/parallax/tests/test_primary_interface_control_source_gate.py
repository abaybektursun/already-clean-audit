from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infra.vast.parallax import primary_interface_control_source_gate as gate
from infra.vast.parallax import primary_interface_control_overlay_stage as stage
from infra.vast.parallax.training_common import TrainingGateError


ROOT = Path(__file__).resolve().parents[4]


def _inputs() -> tuple[dict, list[dict], dict[str, str], str]:
    contract_path = ROOT / gate.SOURCE_CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / gate.BUNDLE_RECEIPT).read_text(encoding="utf-8"))
    overlays = gate.expected_overlay_identities(ROOT, contract)
    return contract, receipt["entries"], overlays, gate.sha256_file(contract_path)


def test_primary_source_gate_is_cwd_independent_and_s3_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = gate.validate_local(ROOT)
    assert result["status"] == "GREEN_EXACT_PRIMARY_PG1_INTERFACE_SOURCE"
    assert result["lineage"] == "pg1_pi"
    assert result["role"] == (
        "primary_within_lineage_discovery_independent_of_supporting_pg2"
    )
    assert result["source_closure"]["contract_artifacts"] == 26
    assert result["source_closure"]["overlay_objects_including_contract"] == 14
    assert result["confirmation_items_read"] == 0
    assert result["paid_authorization"] is False
    assert result["durable_provider"] == "amazon_s3"
    assert result["vast_persistent_storage"] == "FORBIDDEN"


def test_primary_source_closure_rejects_missing_extra_or_drifted_overlay() -> None:
    contract, entries, overlays, digest = _inputs()
    exact = gate.validate_closure(
        contract=contract,
        bundle_entries=entries,
        overlays=overlays,
        source_contract_sha256=digest,
    )
    assert exact["status"] == "GREEN_EXACT_PRIMARY_BUNDLE_OVERLAY_SOURCE_CLOSURE"

    missing = copy.deepcopy(overlays)
    missing.pop(next(path for path in missing if path != gate.SOURCE_CONTRACT_PATH))
    with pytest.raises(TrainingGateError):
        gate.validate_closure(
            contract=contract,
            bundle_entries=entries,
            overlays=missing,
            source_contract_sha256=digest,
        )

    extra = {**overlays, "unexpected.txt": "0" * 64}
    with pytest.raises(TrainingGateError):
        gate.validate_closure(
            contract=contract,
            bundle_entries=entries,
            overlays=extra,
            source_contract_sha256=digest,
        )

    drifted = copy.deepcopy(overlays)
    drifted[gate.SOURCE_CONTRACT_PATH] = "0" * 64
    with pytest.raises(TrainingGateError):
        gate.validate_closure(
            contract=contract,
            bundle_entries=entries,
            overlays=drifted,
            source_contract_sha256=digest,
        )


def test_primary_overlay_inventory_is_small_unique_and_s3_content_addressed() -> None:
    contract = json.loads((ROOT / gate.SOURCE_CONTRACT_PATH).read_text(encoding="utf-8"))
    identities = gate.expected_overlay_identities(ROOT, contract)
    rows = [stage._object(ROOT, Path(path)) for path in sorted(identities)]
    assert len(rows) == len({row["key"] for row in rows}) == 14
    assert sum(int(row["bytes"]) for row in rows) < 512 * 1024
    assert {str(row["path"]): str(row["sha256"]) for row in rows} == identities
    assert all(str(row["key"]).startswith("parallax/input-staging/sha256/") for row in rows)
