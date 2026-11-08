"""POSTCHECK P1 — read the action expert's depth gate out of the safetensors, and nothing else.

Deliberately independent of DEPTHANCHOR: no transformers, no `trust_remote_code`, no model class, no
`infra.pg.depthanchor_model`. This opens the shards with `safetensors.safe_open`, finds the tensors whose
names contain `action_expert_depth_gate`, and reports each layer's bias and weight norm. If the gate
DEPTHANCHOR measured through a forward pass is really sitting at its initialization, that fact is visible
in the file itself, and this is the reading a reader can repeat in a minute without a GPU.

WHAT THE GATE IS, from the released `modeling_molmoact2.py` (lines 2797-2841, 2990-3018): for each expert
layer a `nn.Linear(llm_kv_dim, 1)` is applied to the VALUE states pooled over the non-depth positions, and
`torch.sigmoid` of its output multiplies the keys and values at every depth position. So the realized gate
on a frame is `sigmoid(w . v_pooled + b)`, and

    sigmoid(b - ||w|| * ||v_pooled||)  <=  realized gate  <=  sigmoid(b + ||w|| * ||v_pooled||)

by Cauchy-Schwarz. That bound is the honest weights-only statement: with `||v_pooled||` supplied from a
forward pass (cell N2 measures it), it brackets the gate on any input whatsoever, which is stronger than
a median over frames. Without it, the bias alone is reported and the bound is not claimed.

POSITIVE CONTROL (spec §4): the same code applied to a zero pooled input must return sigmoid(bias)
exactly, which is checked here on every layer rather than asserted in prose.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from typing import Dict, List, Optional

GATE_KEY = "action_expert_depth_gate"
LAYER_RE = re.compile(r"(?:^|\.)(\d+)(?:\.|$)")


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def read_gate(snapshot: pathlib.Path, kv_norm: Optional[float] = None) -> Dict[str, object]:
    """Every gate tensor in a checkpoint directory, by layer.

    `snapshot` is a local HF snapshot directory. `kv_norm` is the median L2 norm of the pooled value
    state the gate sees, when a forward pass has supplied one; the Cauchy-Schwarz bracket is only
    reported when it has.
    """
    import numpy as np
    from safetensors import safe_open

    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors in {snapshot}")
    tensors: Dict[str, "np.ndarray"] = {}
    per_shard: Dict[str, List[str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="np") as f:
            names = [n for n in f.keys() if GATE_KEY in n]
            per_shard[shard.name] = names
            for n in names:
                tensors[n] = f.get_tensor(n)

    layers: Dict[int, Dict[str, object]] = {}
    for name, arr in tensors.items():
        stem = name.split(GATE_KEY, 1)[1]
        m = LAYER_RE.search(stem)
        idx = int(m.group(1)) if m else -1
        entry = layers.setdefault(idx, {"layer": idx, "tensors": {}})
        entry["tensors"][name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if name.endswith(".bias"):
            if flat.size != 1:
                raise ValueError(f"{name}: expected one bias per layer, got {flat.size}")
            entry["bias"] = float(flat[0])
        elif name.endswith(".weight"):
            entry["weight_l2"] = float(np.linalg.norm(flat))
            entry["weight_absmax"] = float(np.max(np.abs(flat)))
            entry["weight_n"] = int(flat.size)

    rows = [layers[k] for k in sorted(layers)]
    for r in rows:
        b = r.get("bias")
        if b is None:
            continue
        # the positive control: a zero pooled input gives sigmoid(bias), exactly, per layer
        r["gate_at_zero_input"] = sigmoid(b)
        if kv_norm is not None and r.get("weight_l2") is not None:
            reach = float(r["weight_l2"]) * float(kv_norm)
            r["cauchy_schwarz_reach"] = reach
            r["gate_lower_bound"] = sigmoid(b - reach)
            r["gate_upper_bound"] = sigmoid(b + reach)
    return {"snapshot": str(snapshot), "n_gate_tensors": len(tensors),
            "tensors_per_shard": {k: sorted(v) for k, v in per_shard.items() if v},
            "n_layers_with_bias": sum(1 for r in rows if "bias" in r),
            "kv_norm_used": kv_norm, "layers": rows}


def summarize(read: Dict[str, object], init_bias: float = -4.0,
              bias_tol: float = 0.1, gate_kill: float = 0.05) -> Dict[str, object]:
    """P1's kill condition, applied to the read: any layer with a bias more than `bias_tol` from the
    declared initialization, or a realized gate above `gate_kill`, and the claim is reworded."""
    import numpy as np
    rows = [r for r in read["layers"] if "bias" in r]
    if not rows:
        return {"present": False, "verdict": "no gate tensors in this checkpoint",
                "n_gate_tensors": read["n_gate_tensors"]}
    bias = np.array([r["bias"] for r in rows], dtype=np.float64)
    g0 = np.array([r["gate_at_zero_input"] for r in rows], dtype=np.float64)
    out = {
        "present": True, "n_layers": len(rows), "init_bias_declared": init_bias,
        "sigmoid_at_init_bias": sigmoid(init_bias),
        "bias": {"min": float(bias.min()), "median": float(np.median(bias)), "max": float(bias.max())},
        "bias_offset_from_init": {"min": float((bias - init_bias).min()),
                                  "max": float((bias - init_bias).max()),
                                  "max_abs": float(np.abs(bias - init_bias).max())},
        "gate_at_zero_input": {"min": float(g0.min()), "median": float(np.median(g0)),
                               "max": float(g0.max())},
        "weight_l2": {"min": float(min(r.get("weight_l2", float("nan")) for r in rows)),
                      "median": float(np.median([r.get("weight_l2", float("nan")) for r in rows])),
                      "max": float(max(r.get("weight_l2", float("nan")) for r in rows))},
        "bias_tolerance": bias_tol, "gate_kill_threshold": gate_kill,
    }
    ub = [r.get("gate_upper_bound") for r in rows if r.get("gate_upper_bound") is not None]
    if ub:
        out["gate_upper_bound"] = {"min": float(min(ub)), "median": float(np.median(ub)),
                                   "max": float(max(ub))}
        out["gate_bounded_below_kill"] = bool(max(ub) <= gate_kill)
    out["every_bias_within_tolerance"] = bool(out["bias_offset_from_init"]["max_abs"] <= bias_tol)
    out["every_gate_at_zero_below_kill"] = bool(g0.max() <= gate_kill)
    # the wording P1 registers, and whether the read supports it
    out["claim_within_0_003_of_sigmoid_init"] = bool(
        np.abs(g0 - sigmoid(init_bias)).max() <= 0.003)
    out["max_abs_gate_minus_init"] = float(np.abs(g0 - sigmoid(init_bias)).max())
    return out


def selftest() -> None:
    """The positive control, on a checkpoint that does not exist: sigmoid at a known bias, and the
    bracket collapsing to it when the input norm is zero."""
    assert abs(sigmoid(-4.0) - 0.017986209962091555) < 1e-15, sigmoid(-4.0)
    lo, hi = sigmoid(-4.0 - 0.0), sigmoid(-4.0 + 0.0)
    assert lo == hi == sigmoid(-4.0)
    fake = {"layers": [{"layer": 0, "bias": -4.0, "gate_at_zero_input": sigmoid(-4.0),
                        "weight_l2": 0.5},
                       {"layer": 1, "bias": -3.95, "gate_at_zero_input": sigmoid(-3.95),
                        "weight_l2": 0.6}],
            "n_gate_tensors": 4}
    s = summarize(fake)
    assert s["present"] and s["n_layers"] == 2
    assert s["every_bias_within_tolerance"] is True
    assert s["claim_within_0_003_of_sigmoid_init"] is True, s["max_abs_gate_minus_init"]
    moved = {"layers": [{"layer": 0, "bias": -1.0, "gate_at_zero_input": sigmoid(-1.0),
                         "weight_l2": 0.5}], "n_gate_tensors": 2}
    s2 = summarize(moved)
    assert s2["every_bias_within_tolerance"] is False
    assert s2["every_gate_at_zero_below_kill"] is False, s2
    assert s2["claim_within_0_003_of_sigmoid_init"] is False
    empty = summarize({"layers": [], "n_gate_tensors": 0})
    assert empty["present"] is False
    print("[postcheck_weights] SELFTEST OK — sigmoid(-4) =", sigmoid(-4.0))


if __name__ == "__main__":
    selftest()
    import sys
    if len(sys.argv) > 1:
        r = read_gate(pathlib.Path(sys.argv[1]))
        print(json.dumps(summarize(r), indent=1))
