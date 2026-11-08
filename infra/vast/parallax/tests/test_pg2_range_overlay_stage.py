from __future__ import annotations

from pathlib import Path

from infra.vast.parallax import pg2_range_overlay_stage as stage


ROOT = Path(__file__).resolve().parents[4]


def test_pg2_overlay_stage_exact_set_without_cloud_mutation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(stage, "RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(
        stage,
        "_stage_one",
        lambda _root, row: {
            **row,
            "bucket": stage.BUCKET,
            "region": stage.REGION,
            "version_id": "fixture-version",
            "server_side_encryption": "AES256",
            "controller_head_exact_match": True,
            "reused_exact_existing_version": False,
        },
    )
    result = stage.run(ROOT)
    assert result["status"] == "GREEN_PG2_SEED17_RANGE_OVERLAYS_S3"
    assert result["object_count"] == 8
    assert result["source_parity"]["status"] == "GREEN_EXACT_PG2_SEED17_RANGE_PARITY"
    assert result["paid_compute_authorization"] is False
    assert result["vast_resources_created"] == 0
