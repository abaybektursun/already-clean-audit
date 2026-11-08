from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_text_image_pixel_audit_launch as v1
from infra.aws_parallax import pi0_text_image_pixel_audit_v2_launch as v2


ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / v2.SPEC_PATH
USER_DATA = ROOT / v2.USER_DATA_PATH


def test_v2_is_fresh_paid_off_identity_bound_to_the_predecessor_mechanical_void() -> None:
    spec = v2.validate_spec(ROOT, SPEC)
    assert spec["run_id"] == v2.RUN_ID
    assert spec["destination"]["durable_run_prefix"] == f"{v2.STAGE_PREFIX}/"
    assert spec["destination"]["result_key"] == f"{v2.STAGE_PREFIX}/pixel-audit-receipt.json"
    assert spec["authorizations"]["paid_action"] is False
    assert spec["authorizations"]["AWS_CPU_create"] is False
    assert spec["authorizations"]["H100"] is False
    assert spec["authorizations"]["automatic_retry"] is False
    predecessor = json.loads(
        (
            ROOT
            / spec["bound_artifacts"]["predecessor_pixel_audit_void"]["path"]
        ).read_text(encoding="utf-8")
    )
    assert predecessor["status"] == "VOID_INSTRUMENT_COMPACT_DEPENDENCY_ROW_INDEX"
    assert predecessor["scientific_input_package_licensed"] is False
    assert predecessor["failure"]["pixel_comparisons_completed"] == 0


def test_v2_changes_no_semantic_raw_decoded_or_estimand_input() -> None:
    v2_spec = v2.validate_spec(ROOT, SPEC)
    v1_spec = json.loads(
        (
            ROOT
            / "studies/lineage/cross-lineage/config/"
            "parallax_pi0_p2_text_image_pixel_audit_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert v2_spec["input_envelope"] == v1_spec["input_envelope"]
    assert v2_spec["audit_contract"] == v1_spec["audit_contract"]
    assert v2_spec["bound_artifacts"]["semantic_manifest"] == v1_spec["bound_artifacts"][
        "semantic_manifest"
    ]
    assert v2_spec["bound_artifacts"]["first_worker_receipt"] == v1_spec["bound_artifacts"][
        "first_worker_receipt"
    ]
    assert v2_spec["bound_artifacts"]["packaging_amendment"] == v1_spec["bound_artifacts"][
        "packaging_amendment"
    ]


def test_v2_rendered_control_uses_named_fields_and_decodes_all_thirteen_rows() -> None:
    rendered = v2.render_user_data(ROOT, SPEC)
    encoded = rendered.split("CONTROL_B64=", 1)[1].splitlines()[0]
    control = json.loads(gzip.decompress(base64.b64decode(encoded)))
    assert control["run_id"] == v2.RUN_ID
    assert len(control["d"]) == 3
    assert len(control["i"]) == 10
    assert all(set(row) == {"n", "v", "b", "s", "p"} for row in control["d"])
    assert all(set(row) == {"j", "t", "x", "v", "b", "s", "r", "q"} for row in control["i"])
    dependencies = {row["n"]: Path("dependencies") / row["p"] for row in control["d"]}
    assert set(dependencies) == {
        "fashion_train_images_idx_gz",
        "fashion_train_labels_idx_gz",
        "pillow_12_1_1_cp312_linux_x86_64",
    }
    assert all(path.name for path in dependencies.values())
    text = USER_DATA.read_text(encoding="utf-8")
    assert 'deps={r["n"]:root/"dependencies"/r["p"] for r in control["d"]}' in text
    assert "r[5]" not in text
    assert len(rendered.encode("utf-8")) < 16_000


def test_v2_profile_restores_v1_globals_after_success_and_failure() -> None:
    names = ("RUN_ID", "STAGE_PREFIX", "SCHEDULE_NAME", "SPEC_PATH", "LAUNCHER_PATH")
    before = {name: getattr(v1, name) for name in names}
    with v2._v2_profile():
        assert v1.RUN_ID == v2.RUN_ID
        assert v1.STAGE_PREFIX == v2.STAGE_PREFIX
    assert all(getattr(v1, name) == value for name, value in before.items())
    with pytest.raises(RuntimeError, match="plant"):
        with v2._v2_profile():
            raise RuntimeError("plant")
    assert all(getattr(v1, name) == value for name, value in before.items())


def test_v2_no_create_delegation_is_profiled_and_cannot_create(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = {}

    def fake_no_create(root, spec_path, output):
        observed.update(
            run_id=v1.RUN_ID,
            stage_prefix=v1.STAGE_PREFIX,
            spec_path=v1.SPEC_PATH,
            launcher_path=v1.LAUNCHER_PATH,
        )
        return {
            "status": "GREEN_PI0_TEXT_IMAGE_PIXEL_AUDIT_NO_CREATE",
            "mutation": {"AWS_instances_created": 0, "S3_objects_written": 0},
            "paid_authorization": False,
        }

    monkeypatch.setattr(v1, "no_create", fake_no_create)
    packet = v2.no_create(ROOT, SPEC, tmp_path / "receipt.json")
    assert observed == {
        "run_id": v2.RUN_ID,
        "stage_prefix": v2.STAGE_PREFIX,
        "spec_path": v2.SPEC_PATH,
        "launcher_path": v2.LAUNCHER_PATH,
    }
    assert set(packet["mutation"].values()) == {0}
    assert packet["paid_authorization"] is False


def test_v2_paid_launch_without_separate_freeze_is_impossible(tmp_path: Path) -> None:
    missing = tmp_path / "missing-freeze.json"
    with v2._v2_profile():
        spec = v1.validate_spec(ROOT, SPEC)
        with pytest.raises(FileNotFoundError):
            v1._validate_freeze(ROOT, SPEC, missing)


def test_worker_remains_cpu_only_nonproducing_and_independent_of_producer() -> None:
    text = USER_DATA.read_text(encoding="utf-8")
    launcher = (ROOT / "infra/aws_parallax/pi0_text_image_pixel_audit_launch.py").read_text(
        encoding="utf-8"
    )
    assert text.count("aws s3api put-object") == 1
    assert "new_or_rewritten_image_objects\":0" in text
    assert "gpu_work\":0" in text
    assert "confirmation_items_read\":0" in text
    assert "model_body_bytes\":0" in text
    assert "pi0_text_image_stage_launch" not in launcher
    assert "audit_pi0_text_image_stage_result" not in launcher
