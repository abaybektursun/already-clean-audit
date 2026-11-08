"""DEPTHANCHOR gate session — spec §9.3: G0, O-seed, O-ceiling, O-shuffle, O-const, the depth-gate
ablation cells, the two room checks, and the frozen margin `m`.

This session runs every CONTROL cell and nothing that could answer the question. That order is the
point: `m` (the 95th percentile of seed-floor / ceiling over frames) and the room verdicts are frozen
from these numbers in a tagged commit, and only then do the target and background cells run. Cells:

  natural      the model's own grid, `enable_adaptive_depth=False`  -> defines b_f
  ident        that grid replayed through the injection path        -> G0, and the reference for every Δ
  seed         the same grid at 7 further generator seeds           -> the seed floor S(f)
  ceiling      the same grid with another task's RGB                -> the spec's ceiling C(f)
  gate0/gate1  the same grid with the checkpoint's per-layer depth gate forced to 0 and to 1
                                                                    -> the depth-matched ceiling (A1)
  shuffle      8 spatial permutations of the grid                   -> rule 1
  const        every cell at the median-depth code                  -> a whole-prefix change, no structure
  repeat_ident the identity call again at the end of the frame      -> per-frame drift check

  pg run infra.aws_pg.jobs.depthanchor_gates --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --s3-prefix results/depthanchor/gates/1 \\
     --budget-min 300 --args " --cuda-graph off"
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import numpy as np

from infra.aws_pg.jobs import _depthanchor_shared as sh
from infra.pg import depthanchor_cells as cells_mod
from infra.pg import depthanchor_core as core
from infra.pg import depthanchor_frames as frames_mod

GATE_CELLS = cells_mod.GATE_CELLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthanchor")
    ap.add_argument("--cuda-graph", choices=("on", "off"), default="off",
                    help="off by default: the graph is bit-identical to the eager loop and 1.85x "
                         "faster, and measuring it left the next graph-enabled call raising an illegal "
                         "memory access, so no session runs on it (SPEC A17)")
    ap.add_argument("--limit-frames", type=int, default=0, help="0 = the whole frozen pack")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = sh.Bank(out, "gates")
    bank.top("args", vars(args))

    pack = frames_mod.load_pack(sh.pull_pack(work / "pack"))
    ladder, ladder_payload = sh.load_ladder(work)
    bank.stage("inputs", {"n_frames": pack["n_frames"], "pack_version": pack["version"],
                          "master_seed": pack["master_seed"],
                          "ladder_median_code": ladder.median_code,
                          "ladder_range": [float(ladder.d_primary.min()),
                                           float(ladder.d_primary.max())],
                          "cells": list(GATE_CELLS)})
    if args.limit_frames:
        pack = dict(pack)
        pack["frames"] = pack["frames"][:args.limit_frames]
        pack["n_frames"] = len(pack["frames"])

    from infra.pg import depthanchor_model as dam
    model, processor, path = dam.load_model()
    actor = dam.Actor(model, processor, enable_cuda_graph=(args.cuda_graph == "on"))
    bank.stage("model", {"snapshot": str(path), "cuda_graph": args.cuda_graph,
                         "depth_gate_weights": dam.gate_weights_readout(model)})

    t0 = time.perf_counter()
    progress = {"frames_done": 0}

    def on_progress(p):
        progress.update(p)
        progress["elapsed_s"] = round(time.perf_counter() - t0, 1)
        progress["seconds_per_frame"] = round(progress["elapsed_s"] / max(1, p["frames_done"]), 2)
        bank.stage("progress", dict(progress))

    offline = cells_mod.run_offline(actor, pack, ladder, cells=GATE_CELLS, per_cell_frames=0,
                                   on_progress=on_progress)
    bank.artifact("records.json", offline["records"])
    bank.artifact("diagnostics.json", offline["diagnostics"])
    bank.artifact("references.json", offline["references"])
    bank.artifact("samples.json", offline["samples"])
    extremes = cells_mod.bank_extremes(actor, pack, ladder, offline["records"],
                                      cells=("shuffle", "const", "ceiling", "gate0", "gate1", "seed"))
    bank.artifact("samples_extremes.json", extremes)
    sh.push_json(out / "records.json", f"{sh.S3_GATES}/records.json")
    sh.push_json(out / "references.json", f"{sh.S3_GATES}/references.json")
    sh.push_json(out / "diagnostics.json", f"{sh.S3_GATES}/diagnostics.json")

    diag = offline["diagnostics"]
    g0 = core.gate_g0(max(d["g0_max_abs_delta_norm"] for d in diag), len(diag))
    gates: Dict[str, core.Gate] = {"G0": g0}
    agg = cells_mod.aggregate(offline, gates, require_rule2=False)
    gates["G4a"] = core.gate_room(_iv(agg["room"]["G4a_rgb_ceiling"]), "G4a")
    gates["G4b"] = core.gate_room(_iv(agg["room"]["G4b_depth_ceiling"]), "G4b")
    agg = cells_mod.aggregate(offline, gates, require_rule2=False)

    drift = [d["repeat_ident_max_abs_delta_norm"] for d in diag]
    tails = [d["ident_n_tail"] for d in diag]
    freeze = {
        "m": agg["m"],
        "m_definition": "the 95th percentile over frames of seed-floor / RGB-ceiling, primary units",
        "ceiling_cell": agg["ceiling_cell"],
        "gates": {k: v.as_dict() for k, v in gates.items()},
        "room": agg["room"],
        "rule1": agg["decision"]["rule1_read_at_all"],
        "n_frames": agg["n_frames"],
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    bank.artifact("freeze.json", freeze)
    sh.push_json(out / "freeze.json", f"{sh.S3_GATES}/freeze.json")

    bank.stage("aggregate", agg)
    bank.stage("instrument", {
        "repeat_ident_max": float(np.max(drift)), "repeat_ident_median": float(np.median(drift)),
        "repeat_ident_all_below_1e-5": bool(np.max(drift) < 1e-5),
        "ident_tail_token_counts": _counts(tails),
        "tail_mismatches": agg["tail_mismatches"],
        "gate_trained_mean_median": float(np.median([d["gate_trained_mean"] for d in diag
                                                     if d.get("gate_trained_mean") is not None]))
        if any(d.get("gate_trained_mean") is not None for d in diag) else None,
        "gate_trained_per_layer_median": _per_layer_median(diag),
        "depth_positions": sorted({d.get("depth_positions") for d in diag}),
        "seconds_total": round(time.perf_counter() - t0, 1),
        "calls": actor.calls,
        "seconds_per_call": round((time.perf_counter() - t0) / max(1, actor.calls), 3),
    })
    bank.top("freeze", freeze)
    verdict = {
        "G0": g0.verdict,
        "room_G4a": gates["G4a"].verdict,
        "room_G4b": gates["G4b"].verdict,
        "m": agg["m"],
        "rule1_read": agg["decision"]["rule1_read_at_all"].get("read"),
        "ready_for_contrast": g0.verdict == "pass",
        "blockers": [] if g0.verdict == "pass" else [f"G0 is {g0.verdict}"],
    }
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1), flush=True)
    if not verdict["ready_for_contrast"]:
        raise SystemExit(f"gates are not ready for the contrast session: {verdict['blockers']}")


def _iv(d: Dict[str, object]) -> core.Interval:
    return core.Interval(point=float(d["point"]), lo=float(d["ci95"][0]), hi=float(d["ci95"][1]),
                         n=int(d["n"]), statistic=str(d["statistic"]))


def _counts(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def _per_layer_median(diag: List[Dict[str, object]]):
    rows = [d["gate_trained"] for d in diag if d.get("gate_trained")]
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    return {"median": np.median(arr, axis=0).tolist(), "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(), "n_frames": int(arr.shape[0])}


if __name__ == "__main__":
    main()
