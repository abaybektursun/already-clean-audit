from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from infra.vast.parallax.g1_adjudicate import adjudicate
from infra.vast.parallax.g1_common import (
    G1GateError,
    assert_comparison_legal,
    identity_replay,
    tensor_fingerprint,
)
from infra.vast.parallax.g1_pi import _inner_targets, pi05_prompt
from infra.vast.parallax.g1_sequence import runtime_environment


ROOT = Path(__file__).resolve().parents[3]
ANCHORS = ROOT / "studies/lineage/cross-lineage/config/parallax_g1_anchors_v2.yaml"
FREEZE = ROOT / "studies/lineage/cross-lineage/config/parallax_g1_freeze_v3.json"


def _fp(token: str, shape: list[int] | None = None) -> dict[str, object]:
    return {
        "shape": shape or [1],
        "sample_values": 1,
        "sample_sha256_float32": token,
    }


def _case(token: str, *, image_tokens: int = 256) -> dict[str, object]:
    return {
        "input": {"pixels": _fp(token, [1, 3, 224, 224]), "image_token_count": image_tokens},
        "anchor": _fp(token + "-anchor", [1, 4, 8]),
        "token_ids_sha256": token + "-tokens",
        "identity": {"status": "GREEN"},
    }


def _core(lineage: str) -> dict[str, object]:
    base = {
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": 123,
        "lineage": lineage,
        "runtime_role": "core",
        "anchor_config_sha256": "anchor",
        "cache_gate_manifest_sha256": "cache",
        "outputs": {
            "siglip": {"cases": {"blank": _case("blank"), "real": _case("real")}},
            "language_parent": {"identity": {"status": "GREEN"}},
            "fused": {
                "vision_patch_size": 14,
                "vision_num_image_tokens": 256,
                "cases": {"blank": _case("blank"), "real": _case("real")},
            },
        },
    }
    if lineage == "pg2":
        base["outputs"]["mix"] = {
            "vision_patch_size": 14,
            "vision_num_image_tokens": 256,
            "cases": {"blank": _case("blank"), "real": _case("real")},
        }
    return base


def _pi() -> dict[str, object]:
    case = {
        "camera_streams": 3,
        "identity": {"status": "GREEN"},
    }
    policy = {
        "use_cache": False,
        "load": {"missing_keys": 0, "unexpected_keys": 0},
        "cases": {"blank": dict(case), "real": dict(case)},
    }
    return {
        "status": "GREEN",
        "provider": "vast.ai",
        "instance_id": 123,
        "lineage": "pg1_pi",
        "runtime_role": "pi_sidecar",
        "pg2_imported": False,
        "anchor_config_sha256": "anchor",
        "cache_gate_manifest_sha256": "cache",
        "outputs": {"pi0": policy, "pi05": policy},
    }


def test_g1_freeze_binds_anchor_bytes() -> None:
    from infra.vast.parallax.artifacts import sha256_file

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_BEFORE_G1"
    assert freeze["anchor_contract"]["sha256"] == sha256_file(ANCHORS)


def test_g1_v2_makes_paligemma_image_marker_explicit_only_on_processor_path() -> None:
    import yaml

    anchors = yaml.safe_load(ANCHORS.read_text(encoding="utf-8"))
    assert anchors["prompts"]["paligemma"].startswith("<image>")
    assert not anchors["prompts"]["pi0"].startswith("<image>")
    assert not anchors["prompts"]["pi05_task"].startswith("<image>")


def test_tensor_fingerprint_is_deterministic_and_finite() -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert tensor_fingerprint(value) == tensor_fingerprint(value.clone())
    assert tensor_fingerprint(value)["finite"] is True


def test_identity_hook_fires_and_is_bit_exact() -> None:
    module = torch.nn.Linear(3, 2, bias=False)
    value = torch.tensor([[1.0, 2.0, 3.0]])
    output, result = identity_replay(lambda: module(value), lambda row: row, {"linear": module})
    assert output.shape == (1, 2)
    assert result["status"] == "GREEN"
    assert result["hook_counts"] == {"linear": 1}
    assert result["max_abs"] == 0.0


def test_identity_hook_refuses_non_firing_target() -> None:
    used = torch.nn.Linear(2, 2)
    unused = torch.nn.Linear(2, 2)
    with pytest.raises(G1GateError, match="did not fire"):
        identity_replay(lambda: used(torch.ones(1, 2)), lambda row: row, {"unused": unused})


def test_cross_gemma_firewall_licenses_only_registered_shared_objects() -> None:
    assert_comparison_legal("pg1_pi", "pg2", "siglip_pixels")
    assert_comparison_legal("pg1_pi", "pg2", "normalized_verdict")
    with pytest.raises(G1GateError, match="cross-Gemma"):
        assert_comparison_legal("pg1_pi", "pg2", "language_logits")


def test_pi05_prompt_is_state_conditioned_not_pi0_newline_grammar() -> None:
    prompt = pi05_prompt("pick_object\n", [0.0] * 32)
    assert prompt.startswith("Task: pick object, State: ")
    assert prompt.count("128") == 32
    assert prompt.endswith(";\nAction: ")


def test_pi_runner_uses_native_discriminated_config_dispatch() -> None:
    source = (ROOT / "infra/vast/parallax/g1_pi.py").read_text(encoding="utf-8")
    assert "PreTrainedConfig.from_pretrained(snapshot" in source
    assert "config = Config.from_pretrained(snapshot" not in source


def test_pi_identity_hook_targets_an_actually_invoked_language_submodule() -> None:
    vision = object()
    language_mlp = object()
    paligemma = SimpleNamespace(
        vision_tower=SimpleNamespace(
            vision_model=SimpleNamespace(encoder=SimpleNamespace(layers=[vision]))
        ),
        language_model=SimpleNamespace(layers=[SimpleNamespace(mlp=language_mlp)]),
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(model=paligemma)
            )
        )
    )
    assert _inner_targets(policy) == {
        "vision_l0": vision,
        "language_l0_mlp": language_mlp,
    }


def test_all_planted_paired_g1_checks_green() -> None:
    result = adjudicate(_core("pg1_pi"), _pi(), _core("pg2"))
    assert result["status"] == "GREEN"
    assert result["hour_6_decision"] == "GO"
    assert result["lineage_claims"]["single_chain_across_both"] is False
    assert result["lineage_claims"]["direct_cross_gemma_language_comparison_performed"] is False


def test_paired_g1_adjudicator_stops_on_processor_confound() -> None:
    supporting = _core("pg2")
    supporting["outputs"]["fused"]["cases"]["real"]["input"]["pixels"] = _fp(
        "different", [1, 3, 224, 224]
    )
    result = adjudicate(_core("pg1_pi"), _pi(), supporting)
    assert result["status"] == "NO_GO"
    assert result["hour_6_decision"] == "NO_GO"
    assert result["checks"]["shared_processor_pixels_pg1_pg2pt_real"] is False


def test_runtime_firewall_exposes_exactly_one_site(tmp_path: Path) -> None:
    core = tmp_path / "core"
    pi = tmp_path / "pi"
    core.mkdir()
    pi.mkdir()
    environment = runtime_environment(core, pi)
    assert environment["PYTHONPATH"] == str(core.resolve())
    assert str(pi.resolve()) not in environment["PYTHONPATH"].split(":")
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_runtime_firewall_refuses_same_site(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    with pytest.raises(G1GateError, match="distinct"):
        runtime_environment(site, site)
