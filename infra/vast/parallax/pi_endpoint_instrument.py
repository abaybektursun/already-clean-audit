"""Weight-free contracts for the PARALLAX pi0 endpoint instrument.

This module deliberately contains no torch, transformers, LeRobot, model, or
environment dependency.  It validates planted fixtures locally; the real-model
runner must execute in the isolated cloud pi sidecar and emit the same evidence
shape before an endpoint arm can freeze.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


LEROBOT_COMMIT = "3f2179f3b69708b6ad009b2e7685dd9d05269ee1"
PI0_REPO = "lerobot/pi0_base"
PI0_REVISION = "25c379b52ba2ff8788cab921758a3cc3fe3f77f2"
PG1_REPO = "google/paligemma-3b-pt-224"
PG1_REVISION = "35e4f46485b4d07967e7e9935bc3786aad50687c"
INTERNAL_VLM_PATH = "model.paligemma_with_expert.paligemma"
EXPECTED_VOCAB_SIZE = 257_152
EXPECTED_HIDDEN_SIZE = 2_048

TEXT_REQUIRED_HOOKS = ("vision_tower", "language_model", "lm_head")
TEXT_FORBIDDEN_HOOKS = ("gemma_expert", "action_in_proj", "action_out_proj")


class PiEndpointGateError(ValueError):
    """A stable, non-promotional pi endpoint instrument failure."""

    def __init__(self, verdict: str, message: str):
        super().__init__(message)
        self.verdict = verdict


def _fail(verdict: str, message: str) -> None:
    raise PiEndpointGateError(verdict, message)


def _shape(value: Any, *, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        _fail("VOID_TEXT_ROUTE_CONTRACT", f"{label} has no shape")
    try:
        result = tuple(int(item) for item in shape)
    except (TypeError, ValueError) as error:
        _fail("VOID_TEXT_ROUTE_CONTRACT", f"{label} shape is invalid: {error}")
    if not result or min(result) <= 0:
        _fail("VOID_TEXT_ROUTE_CONTRACT", f"{label} shape is empty or nonpositive")
    return result


def resolve_internal_vlm(policy: Any) -> Any:
    """Resolve only the pinned LeRobot pi0 internal PaliGemma path."""
    current = policy
    for component in INTERNAL_VLM_PATH.split("."):
        if not hasattr(current, component):
            _fail(
                "VOID_TEXT_ROUTE_CONTRACT",
                f"missing exact internal VLM component: {component}",
            )
        current = getattr(current, component)
    return current


def validate_internal_vlm_contract(
    policy: Any,
    *,
    expected_vocab_size: int = EXPECTED_VOCAB_SIZE,
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE,
) -> dict[str, Any]:
    """Prove that the resolved readout is the VLM, not the action expert."""
    vlm = resolve_internal_vlm(policy)
    for name in ("model", "lm_head", "forward"):
        if not hasattr(vlm, name):
            _fail("VOID_TEXT_ROUTE_CONTRACT", f"internal VLM lacks {name}")
    model = vlm.model
    for name in ("vision_tower", "multi_modal_projector", "language_model"):
        if not hasattr(model, name):
            _fail("VOID_TEXT_ROUTE_CONTRACT", f"internal VLM model lacks {name}")
    language_model = model.language_model
    if not hasattr(language_model, "get_input_embeddings"):
        _fail("VOID_TEXT_ROUTE_CONTRACT", "language model lacks input embeddings")
    embeddings = language_model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        _fail("VOID_TEXT_ROUTE_CONTRACT", "language embedding weight is absent")
    if not hasattr(vlm.lm_head, "weight"):
        _fail("VOID_TEXT_ROUTE_CONTRACT", "VLM lm_head weight is absent")

    embedding_shape = _shape(embeddings.weight, label="embedding weight")
    head_shape = _shape(vlm.lm_head.weight, label="lm_head weight")
    expected_shape = (int(expected_vocab_size), int(expected_hidden_size))
    if embedding_shape != expected_shape or head_shape != expected_shape:
        _fail(
            "VOID_TEXT_ROUTE_CONTRACT",
            f"VLM vocabulary/width mismatch: embeddings={embedding_shape} head={head_shape}",
        )

    core = getattr(policy, "model", None)
    paired = getattr(core, "paligemma_with_expert", None)
    expert = getattr(paired, "gemma_expert", None)
    action_head = getattr(core, "action_out_proj", None)
    if vlm is expert or vlm.lm_head is action_head or model is expert:
        _fail("VOID_TEXT_ROUTE_CONTRACT", "text route aliases an action component")

    return {
        "status": "GREEN_EXACT_INTERNAL_VLM_CONTRACT",
        "path": INTERNAL_VLM_PATH,
        "embedding_shape": list(embedding_shape),
        "lm_head_shape": list(head_shape),
        "action_component_alias": False,
    }


def validate_text_route_trace(hook_counts: Mapping[str, Any]) -> dict[str, Any]:
    """Require a VQA pass to use the VLM and never call action modules."""
    normalized: dict[str, int] = {}
    for name in TEXT_REQUIRED_HOOKS + TEXT_FORBIDDEN_HOOKS:
        try:
            count = int(hook_counts.get(name, 0))
        except (TypeError, ValueError):
            _fail("VOID_TEXT_ROUTE_CONTRACT", f"noninteger hook count: {name}")
        if count < 0:
            _fail("VOID_TEXT_ROUTE_CONTRACT", f"negative hook count: {name}")
        normalized[name] = count
    missing = [name for name in TEXT_REQUIRED_HOOKS if normalized[name] <= 0]
    contaminated = [name for name in TEXT_FORBIDDEN_HOOKS if normalized[name] != 0]
    if missing or contaminated:
        _fail(
            "VOID_TEXT_ROUTE_CONTRACT",
            f"text trace missing={missing} action_contamination={contaminated}",
        )
    return {
        "status": "GREEN_VLM_ONLY_TRACE",
        "hook_counts": normalized,
        "action_modules_called": False,
    }


def _as_float_vector(value: Any, *, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("VOID_PLANTED_NUMERIC_FIXTURE", f"{label} is not a numeric sequence")
    output: list[float] = []
    for raw in value:
        try:
            item = float(raw)
        except (TypeError, ValueError):
            _fail("VOID_PLANTED_NUMERIC_FIXTURE", f"{label} contains a nonnumber")
        if not math.isfinite(item):
            _fail("VOID_PLANTED_NUMERIC_FIXTURE", f"{label} contains a nonfinite value")
        output.append(item)
    if not output:
        _fail("VOID_PLANTED_NUMERIC_FIXTURE", f"{label} is empty")
    return output


def _as_float_matrix(value: Any, *, label: str) -> list[list[float]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        _fail("VOID_PLANTED_NUMERIC_FIXTURE", f"{label} is not a nonempty matrix")
    rows = [_as_float_vector(row, label=f"{label}[{index}]") for index, row in enumerate(value)]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        _fail("VOID_PLANTED_NUMERIC_FIXTURE", f"{label} is ragged")
    return rows


def compare_logit_routes(
    inherited_logits: Any,
    manual_head_logits: Any,
    *,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare inherited conditional-generation logits with a manual lm_head route."""
    inherited = _as_float_matrix(inherited_logits, label="inherited_logits")
    manual = _as_float_matrix(manual_head_logits, label="manual_head_logits")
    if len(inherited) != len(manual) or len(inherited[0]) != len(manual[0]):
        _fail("VOID_TEXT_LOGIT_IDENTITY", "inherited and manual logits differ in shape")
    max_abs = 0.0
    max_scale = 0.0
    for left_row, right_row in zip(inherited, manual, strict=True):
        for left, right in zip(left_row, right_row, strict=True):
            max_abs = max(max_abs, abs(left - right))
            max_scale = max(max_scale, abs(left), abs(right))
    threshold = float(absolute_tolerance) + float(relative_tolerance) * max_scale
    if max_abs > threshold:
        _fail(
            "VOID_TEXT_LOGIT_IDENTITY",
            f"logit routes disagree: max_abs={max_abs:g} threshold={threshold:g}",
        )
    return {
        "status": "GREEN_INHERITED_EQUALS_MANUAL_HEAD",
        "shape": [len(inherited), len(inherited[0])],
        "max_abs": max_abs,
        "threshold": threshold,
    }


def score_candidate_suffix(
    logits: Any,
    token_ids: Sequence[Any],
    *,
    prefix_tokens: int,
) -> dict[str, Any]:
    """Teacher-force one candidate suffix using causal next-token alignment."""
    matrix = _as_float_matrix(logits, label="logits")
    try:
        tokens = [int(token) for token in token_ids]
    except (TypeError, ValueError):
        _fail("VOID_CANDIDATE_SCORER", "token ids are not integers")
    if len(matrix) != len(tokens):
        _fail("VOID_CANDIDATE_SCORER", "logit/token sequence lengths differ")
    if not 1 <= int(prefix_tokens) < len(tokens):
        _fail("VOID_CANDIDATE_SCORER", "candidate suffix is empty or has no causal prefix")
    vocab = len(matrix[0])
    total = 0.0
    for token_index in range(int(prefix_tokens), len(tokens)):
        target = tokens[token_index]
        if not 0 <= target < vocab:
            _fail("VOID_CANDIDATE_SCORER", f"target token outside vocabulary: {target}")
        row = matrix[token_index - 1]
        maximum = max(row)
        log_partition = maximum + math.log(sum(math.exp(value - maximum) for value in row))
        total += row[target] - log_partition
    suffix_tokens = len(tokens) - int(prefix_tokens)
    return {
        "status": "GREEN_CAUSAL_SUFFIX_SCORE",
        "log_likelihood_sum": total,
        "log_likelihood_mean": total / suffix_tokens,
        "suffix_tokens": suffix_tokens,
        "prefix_tokens": int(prefix_tokens),
    }


def paired_action_contrast(correct_instruction: Any, shuffled_instruction: Any) -> dict[str, Any]:
    """Summarize an action change under a paired instruction intervention."""
    correct = _as_float_matrix(correct_instruction, label="correct_instruction_actions")
    shuffled = _as_float_matrix(shuffled_instruction, label="shuffled_instruction_actions")
    if len(correct) != len(shuffled) or len(correct[0]) != len(shuffled[0]):
        _fail("VOID_ROBOT_ACTION_CONTRACT", "paired action tensors differ in shape")
    left = [item for row in correct for item in row]
    right = [item for row in shuffled for item in row]
    delta = [a - b for a, b in zip(left, right, strict=True)]
    l2_delta = math.sqrt(sum(value * value for value in delta))
    l2_left = math.sqrt(sum(value * value for value in left))
    l2_right = math.sqrt(sum(value * value for value in right))
    denominator = 0.5 * (l2_left + l2_right)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    cosine = dot / (l2_left * l2_right) if l2_left > 0 and l2_right > 0 else None
    return {
        "status": "GREEN_PAIRED_ACTION_CONTRAST",
        "shape": [len(correct), len(correct[0])],
        "l2_delta": l2_delta,
        "normalized_l2_delta": l2_delta / denominator if denominator > 0 else None,
        "cosine": cosine,
        "mean_signed_delta": sum(delta) / len(delta),
    }


def wilson_interval(successes: int, episodes: int, *, z: float = 1.959963984540054) -> dict[str, float]:
    """Binomial Wilson interval used by the later frozen robot arm."""
    if not isinstance(successes, int) or not isinstance(episodes, int):
        _fail("VOID_ROBOT_SUCCESS_COUNTS", "success and episode counts must be integers")
    if episodes <= 0 or successes < 0 or successes > episodes or not math.isfinite(float(z)) or z <= 0:
        _fail("VOID_ROBOT_SUCCESS_COUNTS", "invalid success counts or interval z")
    proportion = successes / episodes
    z2 = z * z
    denominator = 1 + z2 / episodes
    center = (proportion + z2 / (2 * episodes)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / episodes + z2 / (4 * episodes**2)) / denominator
    return {
        "proportion": proportion,
        "lower": 0.0 if successes == 0 else max(0.0, center - radius),
        "upper": 1.0 if successes == episodes else min(1.0, center + radius),
    }


def _all_true(row: Mapping[str, Any], fields: Sequence[str]) -> bool:
    return all(row.get(field) is True for field in fields)


def adjudicate_pi_endpoint_gate(evidence: Mapping[str, Any]) -> dict[str, str]:
    """Adjudicate text and robot instruments independently, never as replications."""
    identity = evidence.get("identity", {})
    identity_green = isinstance(identity, Mapping) and _all_true(
        identity,
        (
            "pi0_revision_exact",
            "pg1_revision_exact",
            "lerobot_commit_exact",
            "strict_load_zero_missing_unexpected",
        ),
    )

    text = evidence.get("text", {})
    if not identity_green:
        text_verdict = "VOID_ENDPOINT_IDENTITY"
    elif not isinstance(text, Mapping) or not _all_true(
        text,
        (
            "exact_internal_vlm_path",
            "tokenizer_identity",
            "finite_logits",
            "inherited_manual_logit_identity",
            "candidate_scorer_replay",
            "zero_action_module_calls",
        ),
    ):
        text_verdict = "VOID_TEXT_ROUTE_CONTRACT"
    elif text.get("pg1_reference_positive_control") != "GREEN":
        text_verdict = "VOID_TEXT_REFERENCE_POSITIVE_CONTROL"
    elif text.get("pi_text_vqa_range") == "GREEN":
        text_verdict = "GREEN_PI_TEXT_VQA_MECHANICAL_RANGE"
    elif text.get("pi_text_vqa_range") == "NO_RANGE":
        text_verdict = "RED_PI_TEXT_VQA_NO_RANGE"
    else:
        text_verdict = "PENDING_PI_TEXT_VQA_RANGE"

    robot = evidence.get("robot", {})
    if not identity_green:
        robot_verdict = "VOID_ENDPOINT_IDENTITY"
    elif not isinstance(robot, Mapping) or not _all_true(
        robot,
        (
            "exact_environment_revision",
            "task_ids_frozen_before_rollout",
            "reset_step_contract",
            "paired_noise_state_images",
            "finite_native_action_shape",
        ),
    ):
        robot_verdict = "VOID_ROBOT_ENVIRONMENT_CONTRACT"
    elif robot.get("native_positive_control") != "GREEN":
        robot_verdict = "VOID_ROBOT_NATIVE_POSITIVE_CONTROL"
    elif robot.get("pi0_base_instruction_range") == "GREEN":
        robot_verdict = "GREEN_PI0_BASE_ROBOT_MECHANICAL_RANGE"
    elif robot.get("pi0_base_instruction_range") == "NO_RANGE":
        robot_verdict = "RED_PI0_BASE_ROBOT_NO_INSTRUCTION_RANGE"
    else:
        robot_verdict = "PENDING_PI0_BASE_ROBOT_RANGE"

    verdicts = (text_verdict, robot_verdict)
    if all(value.startswith("GREEN_") for value in verdicts):
        integrated = "GREEN_PI0_ENDPOINT_MECHANICAL_GATE"
    elif any(value.startswith("VOID_") for value in verdicts):
        integrated = "VOID_PI0_ENDPOINT_GATE"
    elif any(value.startswith("PENDING_") for value in verdicts):
        integrated = "PENDING_PI0_ENDPOINT_GATE"
    else:
        integrated = "RED_PI0_ENDPOINT_RANGE_BOUNDARY"
    return {
        "text_verdict": text_verdict,
        "robot_verdict": robot_verdict,
        "integrated_verdict": integrated,
        "claim_scope": "MECHANICAL_DISCOVERY_ONLY_NO_ENDPOINT_SCIENCE_OR_PROMOTION",
        "cross_endpoint_scope": "PI_AND_PG2_MIX_ARE_ANALOGOUS_NOT_REPLICATIONS",
    }
