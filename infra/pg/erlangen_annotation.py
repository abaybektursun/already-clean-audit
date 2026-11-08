"""ERLANGEN mirror-v2 — the task-composition annotation, FROZEN BEFORE ANY TRANSPLANT RUNS.

Spec §2-X: "association between a pre-registered task-composition annotation (frozen BEFORE any
transplant runs; recorded with its author and date — a disclosed judgment layer) and
compatibility, matrix-permutation p".

Author:  the executor (assistant, Claude Opus 5)
Date:    2026-08-03 UTC, before Arm X was written to disk and before any transplant was measured
Basis:   what each unit's TRAINING OBJECTIVE demands of the model — not what any measurement said.

Every candidate unit is annotated (not just the two that Gate-0 selects from weights), precisely so
that the selection cannot influence the annotation. This is a judgment layer and is labelled as
one; §8's threat table lists annotation subjectivity, and `ANNOTATION_B` is the pre-registered
sensitivity — an independent second annotation derived from the published task definitions alone,
on a different axis set, with no reference to the first.

No try/except (CLAUDE.md).
"""
from __future__ import annotations

# --- annotation A: sub-processes the task's objective demands ---------------------------------
SUBPROCESSES_A = (
    "object_recognition",     # identify what is in the image
    "scene_text_ocr",         # read rendered text
    "spatial_localization",   # produce or use image coordinates
    "open_ended_generation",  # emit a free-form multi-token string
    "short_answer_selection",  # emit one short answer from a closed-ish space
    "external_world_knowledge",  # facts not present in the image
    "continuous_action",      # emit motor commands
    "proprioceptive_state",   # consume robot joint/gripper state
)

ANNOTATION_A = {
    "pg1":     ("object_recognition", "open_ended_generation", "spatial_localization",
                "scene_text_ocr", "short_answer_selection"),
    "pi0":     ("object_recognition", "continuous_action", "proprioceptive_state"),
    "pi05":    ("object_recognition", "continuous_action", "proprioceptive_state",
                "short_answer_selection"),
    "cococap":  ("object_recognition", "open_ended_generation"),
    "refcoco":  ("object_recognition", "spatial_localization"),
    "textvqa":  ("object_recognition", "scene_text_ocr", "short_answer_selection"),
    "vqav2":    ("object_recognition", "short_answer_selection"),
    "okvqa":    ("object_recognition", "short_answer_selection", "external_world_knowledge"),
}

# --- annotation B (the pre-registered sensitivity) --------------------------------------------
# A different axis set: the OUTPUT INTERFACE each objective writes through, plus the input
# modality it must consume. Derived from the published task definitions; deliberately does not
# reuse annotation A's decomposition.
SUBPROCESSES_B = (
    "out_free_text", "out_closed_label", "out_coordinate_tokens", "out_action_vector",
    "in_image", "in_text_query", "in_robot_state",
)

ANNOTATION_B = {
    "pg1":     ("out_free_text", "out_closed_label", "out_coordinate_tokens", "in_image",
                "in_text_query"),
    "pi0":     ("out_action_vector", "in_image", "in_text_query", "in_robot_state"),
    "pi05":    ("out_action_vector", "out_closed_label", "in_image", "in_text_query",
                "in_robot_state"),
    "cococap":  ("out_free_text", "in_image"),
    "refcoco":  ("out_coordinate_tokens", "in_image", "in_text_query"),
    "textvqa":  ("out_closed_label", "in_image", "in_text_query"),
    "vqav2":    ("out_closed_label", "in_image", "in_text_query"),
    "okvqa":    ("out_closed_label", "in_image", "in_text_query"),
}


def similarity_matrix(units, annotation):
    """Jaccard similarity over the annotated sub-process sets — the predictor mirror-v2 tests."""
    S = []
    for u in units:
        row = []
        a = set(annotation[_key(u)])
        for v in units:
            b = set(annotation[_key(v)])
            row.append(float(len(a & b) / max(len(a | b), 1)))
        S.append(row)
    return S


def _key(unit):
    """Unit ids carry the roster slot (`ft_max`); the annotation is keyed by the TASK. The caller
    passes `ft_max:textvqa` so the mapping is explicit and auditable in the artifact."""
    return unit.split(":")[-1]


def manifest():
    import hashlib
    import json
    body = {"author": "executor (assistant, Claude Opus 5)", "utc": "2026-08-03",
            "frozen_before": "any Arm-X transplant measurement",
            "subprocesses_a": list(SUBPROCESSES_A), "annotation_a": {k: list(v) for k, v in ANNOTATION_A.items()},
            "subprocesses_b": list(SUBPROCESSES_B), "annotation_b": {k: list(v) for k, v in ANNOTATION_B.items()},
            "basis_a": "sub-processes the training objective demands",
            "basis_b": "output interface + input modality of the objective (independent axis set)",
            "disclosure": "a judgment layer, not a measurement; §8 lists it as a threat and B is "
                          "its pre-registered sensitivity"}
    body["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in body.items() if k != "sha256"}, sort_keys=True).encode()
    ).hexdigest()
    return body
