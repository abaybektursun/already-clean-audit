"""NEURONSHOW frozen rules (studies/neuronshow/SPEC.md). The job and the scorer both import this file; the plan data
(poles, labels, top-20 tokens, decoys, prompts, doses, rms) is studies/neuronshow/plan/plan.json, written by
studies/neuronshow/plan/build_plan.py.

Rules (SPEC §5):
- non-repetitive: neuron_rw_analysis.distinct_bigram_ratio(completion) >= repeat_cut (0.5), the review's metric.
- answer form (gate): the stripped completion has >= answer_min_chars characters, contains no code fence (```), and its
  first copy_head_chars characters (casefolded, whitespace collapsed) are not a substring of the prompt (same
  normalization).
- gate: a prompt is kept for a model when >= gate_min_nonrep of its n_samples unsteered samples are non-repetitive
  and >= gate_min_answer_form are in answer form.
- dose: the largest grid dose at which every target pole of the model has >= dose_min_nonrep of its n_samples P1
  samples non-repetitive; when no grid dose qualifies, the smallest grid dose with met = False.
- control: the first neuron of default_rng([seed, model_index, layer, neuron, control_purpose]).permutation(d_mlp)
  that is not a study neuron of that layer and whose ||W_down[:, j]|| is within control_norm_tol of the target's; the
  target's sign.
- M3: promoted = any of the pole's top-20 token ids among the completion's token ids; label = the pole's label regex
  (case-insensitive) matches the completion text. M4: non-repetitive share and mean distinct-bigram ratio.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np

from infra.pg.neuron_rw_analysis import distinct_bigram_ratio

PLAN_PATH = pathlib.Path(__file__).resolve().parents[2] / "studies/neuronshow/plan/plan.json"


def load(path=PLAN_PATH) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def model_poles(plan: dict, model_key: str) -> list:
    return [p for p in plan["poles"] if p["model_key"] == model_key]


def pole_id(layer: int, neuron: int, sign: int) -> str:
    assert sign in (1, -1), sign
    return f"L{layer:02d}N{neuron}{'+' if sign == 1 else '-'}"


def nonrep(text: str, rules: dict) -> bool:
    return distinct_bigram_ratio(text) >= rules["repeat_cut"]


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def answer_form(completion: str, prompt: str, rules: dict) -> dict:
    c = _norm(completion)
    head = c[:rules["copy_head_chars"]]
    out = {"chars": len(completion.strip()), "fence": "```" in completion,
           "copy": bool(head) and head in _norm(prompt)}
    out["ok"] = out["chars"] >= rules["answer_min_chars"] and not out["fence"] and not out["copy"]
    return out


def gate(rows: list, prompt: str, rules: dict) -> dict:
    """rows: the prompt's unsteered sample records (completion text)."""
    assert len(rows) == rules["n_samples"], f"gate needs {rules['n_samples']} samples, got {len(rows)}"
    nr = [nonrep(r["completion"], rules) for r in rows]
    af = [answer_form(r["completion"], prompt, rules) for r in rows]
    out = {"n": len(rows), "n_nonrep": sum(nr), "n_answer_form": sum(a["ok"] for a in af),
           "n_copy": sum(a["copy"] for a in af), "n_fence": sum(a["fence"] for a in af),
           "n_short": sum(a["chars"] < rules["answer_min_chars"] for a in af)}
    out["kept"] = out["n_nonrep"] >= rules["gate_min_nonrep"] and out["n_answer_form"] >= rules["gate_min_answer_form"]
    return out


def choose_dose(counts: dict, grid: list, rules: dict) -> dict:
    """counts[dose][pole] = non-repetitive P1 samples of n_samples."""
    ok = {d: all(n >= rules["dose_min_nonrep"] for n in counts[d].values()) for d in grid}
    good = [d for d in grid if ok[d]]
    return {"grid": grid, "counts": {str(d): counts[d] for d in grid}, "qualifies": {str(d): ok[d] for d in grid},
            "chosen": max(good) if good else min(grid), "met": bool(good)}


def draw_control(norms: np.ndarray, layer: int, neuron: int, sign: int, exclude: set, model_index: int, plan: dict) -> dict:
    """norms: ||W_down[:, j]|| for every neuron j of the layer (fp32 numpy)."""
    r = plan["rules"]
    seed = [plan["seed"], model_index, layer, neuron, r["control_purpose"]]
    t = float(norms[neuron])
    order = np.random.default_rng(seed).permutation(len(norms))
    for rank, j in enumerate(order.tolist()):
        if j == neuron or j in exclude:
            continue
        if abs(float(norms[j]) - t) <= r["control_norm_tol"] * t:
            return {"control": pole_id(layer, j, sign), "neuron": j, "norm": float(norms[j]), "target_norm": t,
                    "ratio": float(norms[j]) / t, "seed": seed, "rank_in_permutation": rank}
    raise AssertionError(f"L{layer}N{neuron}: no neuron of the layer within {r['control_norm_tol']} of the norm")


def m3_promoted(token_ids: list, top_ids: list) -> bool:
    return bool(set(token_ids) & set(top_ids))


def m3_label(text: str, regex: str) -> bool:
    return re.search(regex, text, flags=re.IGNORECASE) is not None
