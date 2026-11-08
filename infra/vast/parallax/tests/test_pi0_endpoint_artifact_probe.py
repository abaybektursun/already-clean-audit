from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import stat
import struct
import zipfile
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_endpoint_artifact_probe as probe


ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = ROOT / "studies/lineage/cross-lineage/config/parallax_pi0_endpoint_artifact_p1b_v1.json"


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        include_length: bool = True,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = {"Content-Type": "application/octet-stream"}
        if include_length:
            self.headers["Content-Length"] = str(len(body))
        self.headers.update(headers or {})
        self.closed = False

    def getcode(self) -> int:
        return self.status

    def read(self, amount: int = -1) -> bytes:
        return self.body if amount < 0 else self.body[:amount]

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, routes: dict[str, FakeResponse]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    def open(self, request, timeout: int):  # noqa: ANN001
        assert timeout <= 30
        assert request.method == "GET"
        assert request.get_header("Authorization") is None
        self.requested.append(request.full_url)
        return self.routes[request.full_url]


def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _record_hash(body: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def make_wheel(
    *,
    extra: dict[str, bytes | zipfile.ZipInfo] | None = None,
    break_record: bool = False,
) -> bytes:
    files: dict[str, bytes | zipfile.ZipInfo] = {
        "libero/libero/benchmark/task_maps.py": (
            b'TASK_ORDER = {"libero_object": ["object_0"], '
            b'"libero_spatial": ["spatial_0"], "libero_goal": ["goal_0"], '
            b'"libero_10": ["long_0"]}\n'
        ),
        "libero/libero/bddl_files/libero_object/object_0.bddl": (
            b"(define (problem object_0) (:language pick up the object))\n"
        ),
        "libero/libero/bddl_files/libero_spatial/spatial_0.bddl": b"(define (problem spatial_0))\n",
        "libero/libero/bddl_files/libero_goal/goal_0.bddl": b"(define (problem goal_0))\n",
        "libero/libero/bddl_files/libero_10/long_0.bddl": b"(define (problem long_0))\n",
        "hf_libero-0.1.4.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: hf-libero\nVersion: 0.1.4\n"
            b"Requires-Dist: numpy\n\n"
        ),
        "hf_libero-0.1.4.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
    }
    files.update(extra or {})
    record_path = "hf_libero-0.1.4.dist-info/RECORD"
    rows = []
    for path, value in files.items():
        body = value if isinstance(value, bytes) else b"symlink-target"
        rows.append([path, _record_hash(body), str(len(body))])
    rows.append([record_path, "", ""])
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    record_body = record.getvalue().encode()
    if break_record:
        record_body = record_body.replace(b"sha256=", b"sha256=X", 1)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, value in files.items():
            if isinstance(value, zipfile.ZipInfo):
                archive.writestr(value, b"symlink-target")
            else:
                archive.writestr(path, value)
        archive.writestr(record_path, record_body)
    return output.getvalue()


def frozen_limits() -> dict:
    return spec()["wheel_inspection_limits"]


def patch_wheel_identity(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    monkeypatch.setattr(probe, "WHEEL_BYTES", len(body))
    monkeypatch.setattr(probe, "WHEEL_SHA256", hashlib.sha256(body).hexdigest())


def make_safetensors(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    header = {}
    payload = bytearray()
    for name, (dtype, shape, values) in tensors.items():
        start = len(payload)
        payload.extend(values)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    header_body = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header_body)) + header_body + bytes(payload)


def test_frozen_spec_is_exact_and_inert() -> None:
    cfg = spec()
    probe.validate_spec(cfg)
    assert sum(row["bytes"] for row in cfg["network"]["requests"]) == 3_194_781
    assert cfg["authorizations"]["paid_action"] is False
    assert cfg["authorizations"]["exact_wheel_body_read"] is False
    assert all("model.safetensors" not in row["url"] for row in cfg["network"]["requests"])


@pytest.mark.parametrize(
    "url",
    [
        "http://files.pythonhosted.org/package.whl",
        "https://user:secret@files.pythonhosted.org/package.whl",
        "https://huggingface.co/lerobot/pi0_libero_finetuned_v044/resolve/main/config.json",
        "https://huggingface.co/lerobot/pi0_libero_finetuned_v044/resolve/"
        f"{probe.CONTROL_REVISION}/model.safetensors",
        "https://evil.example/control.json",
    ],
)
def test_initial_url_gate_rejects_every_nonfrozen_surface(url: str) -> None:
    row = spec()["network"]["requests"][1]
    with pytest.raises(probe.ProbeError):
        probe.validate_initial_url(url, row)


def test_redirect_gate_allows_only_content_network_for_control() -> None:
    row = spec()["network"]["requests"][1]
    probe.validate_redirect_url(
        f"https://cas-bridge.xethub.hf.co/xet-bridge-us/{probe.CONTROL_REVISION}/signed?x=1", row
    )
    with pytest.raises(probe.ProbeError, match="escaped"):
        probe.validate_redirect_url("https://evil.example/signed", row)


def test_fetcher_requires_exact_size_and_git_blob_identity() -> None:
    body = b'{"small":true}\n'
    row = {
        "id": "fixture",
        "url": "https://huggingface.co/exact",
        "bytes": len(body),
        "digest_kind": "git_blob_sha1",
        "digest": probe._git_blob_sha1(body),
        "kind": "json",
        "body_class": "fixture",
    }
    fetcher = probe.SafeFetcher(
        total_byte_limit=3_200_000,
        max_redirects=1,
        max_requests=7,
        timeout_seconds=5,
        user_agent="test",
        opener=FakeOpener({row["url"]: FakeResponse(body)}),
    )
    result = fetcher.fetch(row)
    assert result.body == body
    changed = dict(row, digest="0" * 40)
    fetcher = probe.SafeFetcher(
        total_byte_limit=3_200_000,
        max_redirects=1,
        max_requests=7,
        timeout_seconds=5,
        user_agent="test",
        opener=FakeOpener({row["url"]: FakeResponse(body)}),
    )
    with pytest.raises(probe.ProbeError, match="digest"):
        fetcher.fetch(changed)


def test_fetcher_rejects_declared_or_actual_size_drift_without_partial_success() -> None:
    body = b"abc"
    row = {
        "id": "fixture",
        "url": "https://huggingface.co/exact",
        "bytes": 3,
        "digest_kind": "sha256",
        "digest": hashlib.sha256(body).hexdigest(),
        "kind": "json",
        "body_class": "fixture",
    }
    wrong_declared = FakeResponse(body, headers={"Content-Length": "4"})
    fetcher = probe.SafeFetcher(
        total_byte_limit=3_200_000,
        max_redirects=1,
        max_requests=7,
        timeout_seconds=5,
        user_agent="test",
        opener=FakeOpener({row["url"]: wrong_declared}),
    )
    with pytest.raises(probe.ProbeError, match="declared artifact size"):
        fetcher.fetch(row)
    assert wrong_declared.closed


def test_wheel_inspection_proves_record_and_literal_task_order(monkeypatch: pytest.MonkeyPatch) -> None:
    body = make_wheel()
    patch_wheel_identity(monkeypatch, body)
    result = probe.inspect_wheel(body, frozen_limits())
    assert result["record_exact_coverage"] is True
    assert result["project_name"] == "hf-libero"
    assert result["version"] == "0.1.4"
    assert result["runtime_task_order_evidence"]["all_four_runtime_suite_orders_proven"] is True
    assert result["bddl_counts_by_suite"] == {"object": 1, "spatial": 1, "goal": 1, "long": 1}
    assert result["execution"] == {
        "installed": False,
        "imported": False,
        "extracted_to_disk": False,
        "simulator_started": False,
    }


def test_wheel_rejects_zip_slip_symlink_record_drift_and_zip_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for body, match in (
        (make_wheel(extra={"../escape": b"x"}), "unsafe"),
        (make_wheel(break_record=True), "RECORD identity"),
        (make_wheel(extra={"compressed-bomb.bin": b"0" * 300_000}), "compression-ratio"),
    ):
        patch_wheel_identity(monkeypatch, body)
        with pytest.raises(probe.ProbeError, match=match):
            probe.inspect_wheel(body, frozen_limits())
    link = zipfile.ZipInfo("libero/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    body = make_wheel(extra={"libero/link": link})
    patch_wheel_identity(monkeypatch, body)
    with pytest.raises(probe.ProbeError, match="symbolic"):
        probe.inspect_wheel(body, frozen_limits())


def test_safetensors_parser_returns_exact_values_and_rejects_bad_layout() -> None:
    body = make_safetensors(
        {
            "action.mean": ("F32", [2], struct.pack("<ff", 1.5, -2.0)),
            "action.count": ("I64", [1], struct.pack("<q", 42)),
        }
    )
    result = probe.inspect_safetensors(body)
    assert result["tensor_count"] == 2
    assert result["tensors"][0]["values_flat"] == [1.5, -2.0]
    header = {"x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}}
    encoded = json.dumps(header).encode()
    malformed = struct.pack("<Q", len(encoded)) + encoded + b"12345"
    with pytest.raises(probe.ProbeError, match="unaccounted|overlap"):
        probe.inspect_safetensors(malformed)


def test_safetensors_parser_rejects_duplicate_header_keys_and_nonfinite_values() -> None:
    header = b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},' \
        b'"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    duplicate = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
    with pytest.raises(probe.ProbeError, match="duplicate JSON key"):
        probe.inspect_safetensors(duplicate)
    nonfinite = make_safetensors({"x": ("F32", [1], struct.pack("<f", float("nan")))})
    with pytest.raises(probe.ProbeError, match="non-finite"):
        probe.inspect_safetensors(nonfinite)


def test_strict_json_rejects_nonfinite_constants_and_unsafe_double_slash_path() -> None:
    with pytest.raises(probe.ProbeError, match="non-finite JSON"):
        probe._strict_json_bytes(b'{"value":NaN}', "fixture")
    with pytest.raises(probe.ProbeError, match="unsafe wheel member"):
        probe._safe_member_name("libero//escape.py")


def test_spec_mutations_cannot_authorize_model_gpu_install_or_retry() -> None:
    for field in ("H100", "model_or_checkpoint_body_read", "package_install_or_import", "automatic_retry"):
        changed = copy.deepcopy(spec())
        changed["authorizations"][field] = True
        with pytest.raises(probe.ProbeError, match="must not authorize"):
            probe.validate_spec(changed)
    changed = copy.deepcopy(spec())
    changed["network"]["requests"].append(
        {
            "id": "model",
            "url": "https://huggingface.co/x/resolve/main/model.safetensors",
            "bytes": 1,
        }
    )
    with pytest.raises(probe.ProbeError, match="request set"):
        probe.validate_spec(changed)
