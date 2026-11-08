"""DEPTHANCHOR exploratory cell — the same edits with the checkpoint's depth gate forced WIDE OPEN.

NOT pre-registered. The frozen rules of SPEC §6 decided the headline on `results/depthanchor/gates/1` and
`results/depthanchor/contrast/1`, and nothing here can change that label; this is the dated exploratory
addendum SPEC §6 allows, and it exists to answer the strongest objection to H_epi.

THE OBJECTION. `action_expert_depth_gate` multiplies the KEYS and VALUES at every depth position by a
per-layer sigmoid before the action expert cross-attends to them, and training left that sigmoid at its
initialization: median 0.01799 over 36 layers against sigmoid(-4.0) = 0.017986 (`preflight/14`
`stages.model.depth_gate_weights`). So "the prefix is not read" might be a fact about the gate rather than
about the prefix: perhaps the depth tokens carry perfectly good geometry that the expert never receives.

THE MEASUREMENT. Force the gate to 1.0 for EVERY call — the edits, the identity reference and the seed
floor alike, so the readout is self-contained at that gate — and re-run the cells rule 1 and rule 2 read.
The gate is ~55x wider than the trained one, so an effect linear in the gate would be ~55x larger. Its own
margin `m`, its own ceilings, its own floor, all recomputed at gate = 1.

WHAT EACH OUTCOME WOULD MEAN. If the shuffle now clears its margin, "the prefix is unread" is a statement
about the gate and the geometry is there behind it. If it still does not, the prefix carries nothing the
expert could use even at full weight, which is a stronger claim than the frozen one and is reported as the
exploratory finding it is.

  pg run infra.aws_pg.jobs.depthanchor_ungated --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --s3-prefix results/depthanchor/ungated/1 \\
     --budget-min 240 --args " --cuda-graph off --force-gate 1.0"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.pg import depthanchor_cells as cells_mod
from infra.pg import depthanchor_core as core
from infra.pg import depthanchor_frames as frames_mod

# Everything rule 1 and rule 2 read, and nothing that would touch the gate: `run_offline`'s own gate0 and
# gate1 cells set and RESET the override, which would undo the global force mid-frame.
UNGATED_CELLS = ("natural", "ident", "seed", "ceiling", "shuffle", "const", "target", "bg", "swap",
                 "random", "repeat_ident")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthanchor")
    ap.add_argument("--cuda-graph", choices=("on", "off"), default="off")
    ap.add_argument("--force-gate", type=float, default=1.0,
                    help="the constant every layer's depth gate is forced to, for every call in this "
                         "session including the reference and the seed floor")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--preflight-prefix", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = sh.Bank(out, "ungated")
    bank.top("args", vars(args))
    bank.top("status", "EXPLORATORY — not pre-registered, cannot change the frozen label")

    pack = frames_mod.load_pack(sh.pull_pack(work / "pack"))
    ladder, _ = sh.load_ladder(work)
    preflight = json.loads(sh.pull_json(
        f"s3://{sh.BUCKET}/{args.preflight_prefix}/out/metrics.json",
        work / "preflight_metrics.json").read_text())
    frozen = json.loads(sh.pull_json(f"{sh.S3_GATES}/freeze.json", work / "freeze.json").read_text())
    if args.limit_frames:
        pack = dict(pack)
        pack["frames"] = pack["frames"][:args.limit_frames]
        pack["n_frames"] = len(pack["frames"])
    bank.stage("inputs", {"n_frames": pack["n_frames"], "cells": list(UNGATED_CELLS),
                          "force_gate": args.force_gate,
                          "frozen_m_at_trained_gate": frozen["m"],
                          "trained_gate_sigmoid_median": _median_sigmoid(preflight)})

    from infra.pg import depthanchor_model as dam
    model, processor, path = dam.load_model()
    actor = dam.Actor(model, processor, enable_cuda_graph=(args.cuda_graph == "on"))
    actor.force_gate(args.force_gate)          # every call in this session, until the process ends
    bank.stage("model", {"snapshot": str(path), "cuda_graph": args.cuda_graph,
                         "forced_gate": args.force_gate,
                         "depth_gate_weights": dam.gate_weights_readout(model)})

    t0 = time.perf_counter()

    def on_progress(p):
        p = dict(p)
        p["elapsed_s"] = round(time.perf_counter() - t0, 1)
        p["seconds_per_frame"] = round(p["elapsed_s"] / max(1, p["frames_done"]), 2)
        bank.stage("progress", p)

    offline = cells_mod.run_offline(actor, pack, ladder, cells=UNGATED_CELLS, per_cell_frames=0,
                                    on_progress=on_progress)
    bank.artifact("records.json", offline["records"])
    bank.artifact("diagnostics.json", offline["diagnostics"])
    bank.artifact("samples.json", offline["samples"])

    # every call ran at the forced gate — checked, not assumed
    gates_seen = sorted({round(float(r["gate_mean"]), 6) for r in offline["records"]
                         if r.get("gate_mean") is not None})
    diag = offline["diagnostics"]
    gates: Dict[str, core.Gate] = {
        "G0": core.gate_g0(max(d["g0_max_abs_delta_norm"] for d in diag), len(diag)),
        "G1": _gate_from(preflight, "refs", "G1"),
        "G2": _gate_from(preflight, "refs", "G2"),
        "G3": _gate_from(preflight, "orient", "G3"),
    }
    agg = cells_mod.aggregate(offline, gates, require_rule2=True)
    gates["G4a"] = core.gate_room(_iv(agg["room"]["G4a_rgb_ceiling"]), "G4a")
    agg = cells_mod.aggregate(offline, gates, require_rule2=True)
    bank.artifact("aggregate.json", agg)
    bank.stage("aggregate", agg)

    verdict = {
        "status": "EXPLORATORY",
        "forced_gate": args.force_gate,
        "gate_mean_values_seen": gates_seen,
        "every_call_at_the_forced_gate": bool(len(gates_seen) == 1
                                              and abs(gates_seen[0] - args.force_gate) < 1e-6),
        "m_at_this_gate": agg["m"],
        "m_at_the_trained_gate": frozen["m"],
        "label_under_this_gate": agg["decision"]["label"],
        "rule1": agg["decision"]["rule1_read_at_all"],
        "ratios": agg["ratios"],
        "absolute": agg["absolute"],
        "room_G4a": agg["room"].get("G4a_rgb_ceiling"),
        "per_k_8": agg["per_k"].get("8", {}).get("paired_target_minus_bg"),
        "pivot_swap": agg.get("pivot_swap"),
        "seconds_total": round(time.perf_counter() - t0, 1),
        "calls": actor.calls,
    }
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1, default=str), flush=True)


def _median_sigmoid(preflight: Dict[str, object]):
    w = (((preflight.get("stages") or {}).get("model") or {}).get("depth_gate_weights") or {})
    sig = w.get("sigmoid_at_bias") or []
    return None if not sig else float(np.median(sig))


def _gate_from(preflight: Dict[str, object], stage: str, name: str) -> core.Gate:
    payload = ((preflight.get("stages") or {}).get(stage) or {}).get(f"gate_{name}")
    if not payload:
        return core.Gate(name, core.CANCELLED, None, None, None,
                         {"why": f"the preflight bank has no {name}"})
    return core.Gate(name=payload["name"], verdict=payload["verdict"], value=payload["value"],
                     floor=payload["floor"], ceiling=payload["ceiling"],
                     detail=dict(payload.get("detail") or {}, source=f"preflight.{stage}"))


def _iv(d: Dict[str, object]) -> core.Interval:
    return core.Interval(point=float(d["point"]), lo=float(d["ci95"][0]), hi=float(d["ci95"][1]),
                         n=int(d["n"]), statistic=str(d["statistic"]))


if __name__ == "__main__":
    main()
