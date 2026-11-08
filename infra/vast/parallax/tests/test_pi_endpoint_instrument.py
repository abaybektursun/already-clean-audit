from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from infra.vast.parallax.pi_endpoint_instrument import (
    INTERNAL_VLM_PATH,
    PiEndpointGateError,
    adjudicate_pi_endpoint_gate,
    compare_logit_routes,
    paired_action_contrast,
    resolve_internal_vlm,
    score_candidate_suffix,
    validate_internal_vlm_contract,
    validate_text_route_trace,
    wilson_interval,
)


class Weight:
    def __init__(self, shape):
        self.shape = shape


class Embeddings:
    def __init__(self, shape):
        self.weight = Weight(shape)


class LanguageModel:
    def __init__(self, shape):
        self._embeddings = Embeddings(shape)

    def get_input_embeddings(self):
        return self._embeddings


def _policy(*, alias_action_head: bool = False, shape=(257_152, 2_048)):
    lm_head = SimpleNamespace(weight=Weight(shape))
    vlm_model = SimpleNamespace(
        vision_tower=object(),
        multi_modal_projector=object(),
        language_model=LanguageModel(shape),
    )
    vlm = SimpleNamespace(model=vlm_model, lm_head=lm_head, forward=lambda: None)
    paired = SimpleNamespace(paligemma=vlm, gemma_expert=object())
    action_head = lm_head if alias_action_head else object()
    return SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=paired,
            action_in_proj=object(),
            action_out_proj=action_head,
        )
    )


def _green_evidence():
    return {
        "identity": {
            "pi0_revision_exact": True,
            "pg1_revision_exact": True,
            "lerobot_commit_exact": True,
            "strict_load_zero_missing_unexpected": True,
        },
        "text": {
            "exact_internal_vlm_path": True,
            "tokenizer_identity": True,
            "finite_logits": True,
            "inherited_manual_logit_identity": True,
            "candidate_scorer_replay": True,
            "zero_action_module_calls": True,
            "pg1_reference_positive_control": "GREEN",
            "pi_text_vqa_range": "GREEN",
        },
        "robot": {
            "exact_environment_revision": True,
            "task_ids_frozen_before_rollout": True,
            "reset_step_contract": True,
            "paired_noise_state_images": True,
            "finite_native_action_shape": True,
            "native_positive_control": "GREEN",
            "pi0_base_instruction_range": "GREEN",
        },
    }


def test_internal_vlm_contract_resolves_only_exact_pinned_path():
    policy = _policy()
    assert resolve_internal_vlm(policy) is policy.model.paligemma_with_expert.paligemma
    result = validate_internal_vlm_contract(policy)
    assert result == {
        "status": "GREEN_EXACT_INTERNAL_VLM_CONTRACT",
        "path": INTERNAL_VLM_PATH,
        "embedding_shape": [257_152, 2_048],
        "lm_head_shape": [257_152, 2_048],
        "action_component_alias": False,
    }


def test_internal_vlm_contract_voids_missing_path():
    with pytest.raises(PiEndpointGateError) as caught:
        resolve_internal_vlm(SimpleNamespace(model=SimpleNamespace()))
    assert caught.value.verdict == "VOID_TEXT_ROUTE_CONTRACT"


def test_internal_vlm_contract_voids_shape_drift_and_action_alias():
    with pytest.raises(PiEndpointGateError, match="vocabulary/width"):
        validate_internal_vlm_contract(_policy(shape=(10, 20)))
    with pytest.raises(PiEndpointGateError, match="aliases"):
        validate_internal_vlm_contract(_policy(alias_action_head=True))


def test_text_route_trace_requires_vlm_and_forbids_action_modules():
    result = validate_text_route_trace(
        {
            "vision_tower": 1,
            "language_model": 1,
            "lm_head": 1,
            "gemma_expert": 0,
            "action_in_proj": 0,
            "action_out_proj": 0,
        }
    )
    assert result["status"] == "GREEN_VLM_ONLY_TRACE"
    assert result["action_modules_called"] is False

    with pytest.raises(PiEndpointGateError) as missing:
        validate_text_route_trace({"vision_tower": 1, "lm_head": 1})
    assert missing.value.verdict == "VOID_TEXT_ROUTE_CONTRACT"
    with pytest.raises(PiEndpointGateError, match="action_contamination"):
        validate_text_route_trace(
            {"vision_tower": 1, "language_model": 1, "lm_head": 1, "gemma_expert": 1}
        )


def test_logit_route_identity_accepts_roundoff_and_voids_disagreement():
    result = compare_logit_routes(
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0 + 1e-7, 2.0], [3.0, 4.0]],
    )
    assert result["status"] == "GREEN_INHERITED_EQUALS_MANUAL_HEAD"
    assert result["max_abs"] == pytest.approx(1e-7)
    with pytest.raises(PiEndpointGateError) as caught:
        compare_logit_routes([[1.0, 2.0]], [[1.0, 2.1]])
    assert caught.value.verdict == "VOID_TEXT_LOGIT_IDENTITY"


def test_candidate_suffix_scorer_uses_causal_alignment_and_sum():
    logits = [
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
        [9.0, 0.0, 0.0],
    ]
    result = score_candidate_suffix(logits, [0, 1, 2], prefix_tokens=1)
    expected = (3.0 - math.log(math.exp(3.0) + 2.0)) + (
        4.0 - math.log(math.exp(4.0) + 2.0)
    )
    assert result["log_likelihood_sum"] == pytest.approx(expected)
    assert result["log_likelihood_mean"] == pytest.approx(expected / 2)
    assert result["suffix_tokens"] == 2


def test_candidate_suffix_scorer_is_invariant_to_additive_logit_shift():
    base = score_candidate_suffix([[0.0, 2.0], [1.0, 0.0]], [0, 1], prefix_tokens=1)
    shifted = score_candidate_suffix([[50.0, 52.0], [1.0, 0.0]], [0, 1], prefix_tokens=1)
    assert shifted["log_likelihood_sum"] == pytest.approx(base["log_likelihood_sum"])
    with pytest.raises(PiEndpointGateError) as caught:
        score_candidate_suffix([[0.0, 1.0]], [0], prefix_tokens=1)
    assert caught.value.verdict == "VOID_CANDIDATE_SCORER"


def test_paired_action_contrast_preserves_shape_and_reports_instruction_shift():
    result = paired_action_contrast([[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]])
    assert result["shape"] == [2, 2]
    assert result["l2_delta"] == pytest.approx(2.0)
    assert result["normalized_l2_delta"] == pytest.approx(math.sqrt(2.0))
    assert result["cosine"] == pytest.approx(0.0)
    with pytest.raises(PiEndpointGateError) as caught:
        paired_action_contrast([[1.0]], [[1.0, 2.0]])
    assert caught.value.verdict == "VOID_ROBOT_ACTION_CONTRACT"


def test_wilson_interval_handles_boundary_success_counts():
    zero = wilson_interval(0, 10)
    full = wilson_interval(10, 10)
    assert zero["lower"] == 0.0
    assert 0.0 < zero["upper"] < 0.5
    assert 0.5 < full["lower"] < 1.0
    assert full["upper"] == 1.0
    with pytest.raises(PiEndpointGateError) as caught:
        wilson_interval(11, 10)
    assert caught.value.verdict == "VOID_ROBOT_SUCCESS_COUNTS"


def test_adjudicator_requires_both_independent_endpoint_instruments_green():
    result = adjudicate_pi_endpoint_gate(_green_evidence())
    assert result["text_verdict"] == "GREEN_PI_TEXT_VQA_MECHANICAL_RANGE"
    assert result["robot_verdict"] == "GREEN_PI0_BASE_ROBOT_MECHANICAL_RANGE"
    assert result["integrated_verdict"] == "GREEN_PI0_ENDPOINT_MECHANICAL_GATE"
    assert result["cross_endpoint_scope"] == "PI_AND_PG2_MIX_ARE_ANALOGOUS_NOT_REPLICATIONS"


@pytest.mark.parametrize(
    ("mutation", "expected_text", "expected_robot", "expected_integrated"),
    [
        (
            lambda row: row["identity"].update(pi0_revision_exact=False),
            "VOID_ENDPOINT_IDENTITY",
            "VOID_ENDPOINT_IDENTITY",
            "VOID_PI0_ENDPOINT_GATE",
        ),
        (
            lambda row: row["text"].update(zero_action_module_calls=False),
            "VOID_TEXT_ROUTE_CONTRACT",
            "GREEN_PI0_BASE_ROBOT_MECHANICAL_RANGE",
            "VOID_PI0_ENDPOINT_GATE",
        ),
        (
            lambda row: row["text"].update(pg1_reference_positive_control="RED"),
            "VOID_TEXT_REFERENCE_POSITIVE_CONTROL",
            "GREEN_PI0_BASE_ROBOT_MECHANICAL_RANGE",
            "VOID_PI0_ENDPOINT_GATE",
        ),
        (
            lambda row: row["text"].update(pi_text_vqa_range="NO_RANGE"),
            "RED_PI_TEXT_VQA_NO_RANGE",
            "GREEN_PI0_BASE_ROBOT_MECHANICAL_RANGE",
            "RED_PI0_ENDPOINT_RANGE_BOUNDARY",
        ),
        (
            lambda row: row["robot"].update(exact_environment_revision=False),
            "GREEN_PI_TEXT_VQA_MECHANICAL_RANGE",
            "VOID_ROBOT_ENVIRONMENT_CONTRACT",
            "VOID_PI0_ENDPOINT_GATE",
        ),
        (
            lambda row: row["robot"].update(native_positive_control="RED"),
            "GREEN_PI_TEXT_VQA_MECHANICAL_RANGE",
            "VOID_ROBOT_NATIVE_POSITIVE_CONTROL",
            "VOID_PI0_ENDPOINT_GATE",
        ),
        (
            lambda row: row["robot"].update(pi0_base_instruction_range="NO_RANGE"),
            "GREEN_PI_TEXT_VQA_MECHANICAL_RANGE",
            "RED_PI0_BASE_ROBOT_NO_INSTRUCTION_RANGE",
            "RED_PI0_ENDPOINT_RANGE_BOUNDARY",
        ),
        (
            lambda row: row["robot"].update(pi0_base_instruction_range="NOT_RUN"),
            "GREEN_PI_TEXT_VQA_MECHANICAL_RANGE",
            "PENDING_PI0_BASE_ROBOT_RANGE",
            "PENDING_PI0_ENDPOINT_GATE",
        ),
    ],
)
def test_adjudicator_exercises_void_red_and_pending_worlds(
    mutation, expected_text, expected_robot, expected_integrated
):
    evidence = _green_evidence()
    mutation(evidence)
    result = adjudicate_pi_endpoint_gate(evidence)
    assert result["text_verdict"] == expected_text
    assert result["robot_verdict"] == expected_robot
    assert result["integrated_verdict"] == expected_integrated
