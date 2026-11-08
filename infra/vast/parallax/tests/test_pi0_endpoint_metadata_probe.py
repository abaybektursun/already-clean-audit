from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from infra.aws_parallax import pi0_endpoint_metadata_probe as probe


ROOT = Path(__file__).resolve().parents[4]
SPEC_PATH = ROOT / "studies/lineage/cross-lineage/config/parallax_pi0_endpoint_metadata_p1_v2.json"


class FakeResponse:
    def __init__(
        self,
        body: bytes = b"",
        *,
        status: int = 200,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
        declared_length: int | None = None,
    ) -> None:
        self.body = body
        self.status = status
        self.headers = {"Content-Type": content_type}
        if declared_length is not None:
            self.headers["Content-Length"] = str(declared_length)
        else:
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
    def __init__(self, routes: dict[str, FakeResponse | list[FakeResponse]]) -> None:
        self.routes = routes
        self.requested: list[str] = []

    def open(self, request, timeout: int):  # noqa: ANN001
        assert timeout <= 30
        assert request.method == "GET"
        assert request.get_header("Authorization") is None
        url = request.full_url
        self.requested.append(url)
        response = self.routes[url]
        if isinstance(response, list):
            return response.pop(0)
        return response


def as_json(value) -> bytes:  # noqa: ANN001
    return json.dumps(value, separators=(",", ":")).encode()


def spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def response_for_request(row: dict) -> FakeResponse:
    if row["kind"] == "text":
        body = ("\n".join(spec()["source_needles"].get(row["id"], ["source"])) + "\n").encode()
        return FakeResponse(body, content_type="text/plain")
    raise AssertionError(row)


def full_routes() -> dict[str, FakeResponse]:
    cfg = spec()
    routes = {}
    hf = {
        "id": "lerobot/pi0_libero_finetuned_v044",
        "sha": "1" * 40,
        "siblings": [
            {"rfilename": "config.json", "size": 123, "blobId": "blob-config"},
            {
                "rfilename": "model.safetensors",
                "size": 1000,
                "lfs": {"size": 1000, "sha256": "a" * 64},
            },
        ],
    }
    pypi = {
        "releases": {
            "0.1.4": [
                {
                    "filename": "hf_libero-0.1.4-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "python_version": "py3",
                    "size": 101,
                    "url": "https://files.pythonhosted.org/hf_libero-0.1.4.whl",
                    "digests": {"sha256": "4" * 64},
                    "yanked": False,
                }
            ],
            "0.1.3": [
                {
                    "filename": "hf_libero-0.1.3-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "python_version": "py3",
                    "size": 99,
                    "url": "https://files.pythonhosted.org/hf_libero-0.1.3.whl",
                    "digests": {"sha256": "3" * 64},
                    "yanked": False,
                }
            ],
            "0.1.6": [
                {
                    "filename": "hf_libero-0.1.6-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "python_version": "py3",
                    "size": 106,
                    "url": "https://files.pythonhosted.org/hf_libero-0.1.6.whl",
                    "digests": {"sha256": "6" * 64},
                    "yanked": False,
                }
            ],
            "0.2.0": [
                {
                    "filename": "hf_libero-0.2.0.whl",
                    "packagetype": "bdist_wheel",
                    "python_version": "py3",
                    "size": 200,
                    "url": "https://files.pythonhosted.org/hf_libero-0.2.0.whl",
                    "digests": {"sha256": "2" * 64},
                    "yanked": False,
                }
            ],
        }
    }
    commit = "6" * 40
    tags = [{"name": "v0.1.6", "commit": {"sha": commit}}]
    for row in cfg["network"]["requests"]:
        if row["id"] == "hf_control_alias":
            routes[row["url"]] = FakeResponse(as_json(hf))
        elif row["id"] == "pypi_hf_libero":
            routes[row["url"]] = FakeResponse(as_json(pypi))
        elif row["id"] == "libero_tags":
            routes[row["url"]] = FakeResponse(as_json(tags))
        else:
            routes[row["url"]] = response_for_request(row)
    tree_url = f"https://api.github.com/repos/huggingface/lerobot-libero/git/trees/{commit}?recursive=1"
    tree = {
        "sha": "7" * 40,
        "truncated": False,
        "tree": [
            {
                "path": "libero/libero/benchmark/task_maps.py",
                "type": "blob",
                "size": 80,
                "sha": "8" * 40,
            },
            *[
                {
                    "path": f"libero/libero/bddl_files/{folder}/task_{index}.bddl",
                    "type": "blob",
                    "size": 20,
                    "sha": str(index + 1) * 40,
                }
                for index, folder in enumerate(probe.SUITE_PATHS.values())
            ],
        ],
    }
    routes[tree_url] = FakeResponse(as_json(tree))
    raw_url = (
        "https://raw.githubusercontent.com/huggingface/lerobot-libero/"
        f"{commit}/libero/libero/benchmark/task_maps.py"
    )
    routes[raw_url] = FakeResponse(
        b'TASK_ORDER = {"libero_object": ["task_0"], "libero_spatial": ["task_1"], '
        b'"libero_goal": ["task_2"], "libero_10": ["task_3"]}\n',
        content_type="text/plain",
    )
    return routes


@pytest.mark.parametrize(
    "url",
    [
        "http://pypi.org/pypi/hf-libero/json",
        "https://evil.example/pypi/hf-libero/json",
        "https://huggingface.co/lerobot/pi0_libero_finetuned/resolve/main/model.safetensors",
        "https://huggingface.co/api/models/lerobot/pi0_libero_finetuned?blobs=false",
        "https://raw.githubusercontent.com/huggingface/lerobot/main/pyproject.toml",
        "https://api.github.com/repos/huggingface/lerobot-libero/zipball/main",
        "https://user:secret@pypi.org/pypi/hf-libero/json",
    ],
)
def test_url_gate_rejects_nonmetadata_surfaces(url: str) -> None:
    with pytest.raises(probe.ProbeError):
        probe.validate_fetch_url(url)


def test_allowed_url_surface_accepts_only_exact_metadata() -> None:
    for row in spec()["network"]["requests"]:
        probe.validate_fetch_url(row["url"])


def test_redirect_to_forbidden_host_is_rejected_before_second_request() -> None:
    url = "https://pypi.org/pypi/hf-libero/json"
    opener = FakeOpener(
        {
            url: FakeResponse(
                status=302,
                headers={"Location": "https://evil.example/steal"},
            )
        }
    )
    fetcher = probe.SafeFetcher(
        total_byte_limit=1000,
        max_redirects=2,
        max_requests=3,
        timeout_seconds=5,
        user_agent="test",
        opener=opener,
    )
    with pytest.raises(probe.ProbeError, match="host is not allowlisted"):
        fetcher.fetch(url, max_bytes=500, expected_kind="json")
    assert opener.requested == [url]


def test_allowed_alias_redirect_remains_bounded() -> None:
    alias = "https://huggingface.co/api/models/lerobot/pi0_libero_finetuned?blobs=true"
    canonical = "https://huggingface.co/api/models/lerobot/pi0_libero_finetuned_v044?blobs=true"
    opener = FakeOpener(
        {
            alias: FakeResponse(status=307, headers={"Location": canonical}),
            canonical: FakeResponse(as_json({"ok": True})),
        }
    )
    fetcher = probe.SafeFetcher(
        total_byte_limit=1000,
        max_redirects=1,
        max_requests=2,
        timeout_seconds=5,
        user_agent="test",
        opener=opener,
    )
    result = fetcher.fetch(alias, max_bytes=500, expected_kind="json")
    assert result.final_url == canonical
    assert result.redirects == (canonical,)


def test_content_length_overrun_is_rejected_without_body_read() -> None:
    url = "https://pypi.org/pypi/hf-libero/json"
    response = FakeResponse(b"{}", declared_length=501)
    opener = FakeOpener({url: response})
    fetcher = probe.SafeFetcher(
        total_byte_limit=1000,
        max_redirects=0,
        max_requests=1,
        timeout_seconds=5,
        user_agent="test",
        opener=opener,
    )
    with pytest.raises(probe.ProbeError, match="per-response"):
        fetcher.fetch(url, max_bytes=500, expected_kind="json")
    assert response.closed


def test_actual_body_overrun_is_rejected_when_length_is_absent() -> None:
    url = "https://pypi.org/pypi/hf-libero/json"
    response = FakeResponse(b"x" * 501)
    response.headers.pop("Content-Length")
    fetcher = probe.SafeFetcher(
        total_byte_limit=1000,
        max_redirects=0,
        max_requests=1,
        timeout_seconds=5,
        user_agent="test",
        opener=FakeOpener({url: response}),
    )
    with pytest.raises(probe.ProbeError, match="response body"):
        fetcher.fetch(url, max_bytes=500, expected_kind="json")


def test_cumulative_budget_is_enforced() -> None:
    first = "https://pypi.org/pypi/hf-libero/json"
    second = "https://api.github.com/repos/huggingface/lerobot-libero/tags?per_page=100"
    fetcher = probe.SafeFetcher(
        total_byte_limit=10,
        max_redirects=0,
        max_requests=2,
        timeout_seconds=5,
        user_agent="test",
        opener=FakeOpener({first: FakeResponse(b"123456"), second: FakeResponse(b"12345")}),
    )
    fetcher.fetch(first, max_bytes=10, expected_kind="json")
    with pytest.raises(probe.ProbeError, match="cumulative"):
        fetcher.fetch(second, max_bytes=10, expected_kind="json")


def test_hf_parser_freezes_canonical_revision_and_manifest() -> None:
    payload = {
        "id": "lerobot/pi0_libero_finetuned_v044",
        "sha": "a" * 40,
        "siblings": [{"rfilename": "model.safetensors", "lfs": {"size": 5, "sha256": "b" * 64}}],
    }
    result = probe.parse_hf_control(payload)
    assert result["canonical_repo"].endswith("_v044")
    assert result["revision"] == "a" * 40
    assert result["file_bodies_read"] == 0
    assert result["manifest"][0]["lfs_sha256"] == "b" * 64


def test_hf_parser_rejects_moving_or_foreign_identity() -> None:
    with pytest.raises(probe.ProbeError, match="canonical"):
        probe.parse_hf_control({"id": "attacker/model", "sha": "a" * 40, "siblings": [{"rfilename": "x"}]})
    with pytest.raises(probe.ProbeError, match="immutable"):
        probe.parse_hf_control(
            {"id": "lerobot/pi0_libero_finetuned_v044", "sha": "main", "siblings": [{"rfilename": "x"}]}
        )


def test_pypi_parser_uses_frozen_highest_compatible_rule() -> None:
    payload = json.loads(full_routes()["https://pypi.org/pypi/hf-libero/json"].body)
    result = probe.parse_pypi(payload)
    assert result["selected_version"] == "0.1.6"
    assert "0.2.0" not in result["compatible_versions"]
    assert "0.1.3" not in result["compatible_versions"]
    assert result["selected_artifact"]["sha256"] == "6" * 64
    assert result["artifact_body_read"] is False


def test_full_probe_captures_metadata_without_blob_or_package_fetches() -> None:
    routes = full_routes()
    opener = FakeOpener(routes)
    packet = probe.run_probe(spec(), opener=opener)
    assert packet["status"] == "GREEN_P1_METADATA_PROBE_COMPLETE"
    assert packet["robot_positive_control"]["revision"] == "1" * 40
    assert packet["hf_libero_distribution"]["selected_version"] == "0.1.6"
    assert packet["hf_libero_source_tag"]["commit"] == "6" * 40
    assert packet["hf_libero_source_tree"]["bddl_counts_by_suite"] == {
        "object": 1,
        "spatial": 1,
        "goal": 1,
        "long": 1,
    }
    assert packet["benchmark_literal_assignments"][0]["assignments"][0]["name"] == "TASK_ORDER"
    assert packet["runtime_task_order_evidence"]["all_four_runtime_suite_orders_proven"] is True
    assert packet["unresolved_fields"] == []
    assert packet["network"]["model_checkpoint_package_dataset_simulator_container_bodies_read"] == 0
    assert packet["p2_licensed"] is False
    assert all("resolve" not in url and "files.pythonhosted.org" not in url for url in opener.requested)


def test_source_evidence_failure_is_explicit_not_smoothed_over() -> None:
    routes = full_routes()
    config_url = next(
        row["url"] for row in spec()["network"]["requests"] if row["id"] == "lerobot_env_configs"
    )
    routes[config_url] = FakeResponse(b"class LiberoEnv:\n    pass\n", content_type="text/plain")
    packet = probe.run_probe(spec(), opener=FakeOpener(routes))
    assert any(item.startswith("missing_source_evidence:lerobot_env_configs:") for item in packet["unresolved_fields"])
    assert packet["p2_licensed"] is False


def test_spec_mutation_cannot_authorize_paid_or_model_work() -> None:
    changed = copy.deepcopy(spec())
    changed["authorizations"]["AWS_CPU_create"] = True
    with pytest.raises(probe.ProbeError, match="must not authorize"):
        probe.validate_spec(changed)
    changed = copy.deepcopy(spec())
    changed["network"]["forbidden_body_classes"].remove("model_weights")
    with pytest.raises(probe.ProbeError, match="forbidden-body"):
        probe.validate_spec(changed)


def test_malformed_json_is_void_not_partial_output() -> None:
    result = probe.FetchResult(
        request_url="https://pypi.org/pypi/hf-libero/json",
        final_url="https://pypi.org/pypi/hf-libero/json",
        status=200,
        content_type="application/json",
        body=b"{broken",
        redirects=(),
    )
    with pytest.raises(probe.ProbeError, match="malformed"):
        probe._json(result, "test")
