"""DEPTHANCHOR contrast session — spec §9.4: O-target-k, O-bg-k, O-random-k, O-cell, then unblind once.

Reads the FROZEN margin from the gate session's `freeze.json` and the control records from its bank,
runs only the cells that can answer the question, merges the two record sets, and applies the §6 rules
exactly once. Three things are checked across the session boundary rather than assumed:

  * the model's own grid for each frame must equal the grid the gate session recorded (greedy decoding
    on the same frame at the same revision);
  * the identity-replay reference must reproduce the gate session's to `--drift-tol`;
  * the recomputed margin must equal the frozen one — and if it does not, the frozen one still decides.

  pg run infra.aws_pg.jobs.depthanchor_contrast --type g6e.2xlarge \\
     --weights models--allenai--MolmoAct2-Think-LIBERO --s3-prefix results/depthanchor/contrast/1 \\
     --budget-min 420 --args " --per-cell-frames 2 --cuda-graph off"
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

CONTRAST_CELLS = cells_mod.CONTRAST_CELLS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/work/depthanchor")
    ap.add_argument("--cuda-graph", choices=("on", "off"), default="off",
                    help="off by default: the graph is bit-identical to the eager loop and 1.85x "
                         "faster, and measuring it left the next graph-enabled call raising an illegal "
                         "memory access, so no session runs on it (SPEC A17)")
    ap.add_argument("--per-cell-frames", type=int, default=5)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--drift-tol", type=float, default=1e-4)
    ap.add_argument("--preflight-prefix", required=True,
                    help="the preflight whose G1/G2/G3 verdicts this session reads; required, because "
                         "a default would silently read a stale preflight")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bank = sh.Bank(out, "contrast")
    bank.top("args", vars(args))

    pack = frames_mod.load_pack(sh.pull_pack(work / "pack"))
    ladder, _ = sh.load_ladder(work)
    freeze = json.loads(sh.pull_json(f"{sh.S3_GATES}/freeze.json", work / "freeze.json").read_text())
    gate_records = json.loads(sh.pull_json(f"{sh.S3_GATES}/records.json",
                                           work / "gates_records.json").read_text())
    gate_refs = json.loads(sh.pull_json(f"{sh.S3_GATES}/references.json",
                                        work / "gates_references.json").read_text())
    preflight = json.loads(sh.pull_json(
        f"s3://{sh.BUCKET}/{args.preflight_prefix}/out/metrics.json",
        work / "preflight_metrics.json").read_text())
    # The contrast artifact is self-contained: the freeze it read and a hash of the gate records it
    # merged are copied into this run's own output, so a later reader never has to trust that the
    # fixed S3 path still holds what this session used.
    import hashlib
    bank.artifact("gates_freeze_used.json", freeze)
    gate_hash = hashlib.sha256(json.dumps(gate_records, sort_keys=True).encode()).hexdigest()
    bank.stage("inputs", {
        "frozen_m": freeze["m"], "frozen_utc": freeze["frozen_utc"],
        "gate_records_sha256": gate_hash,
        "gate_records": len(gate_records), "gate_reference_frames": len(gate_refs),
        "room": freeze["room"], "gate_verdicts": {k: v["verdict"] for k, v in freeze["gates"].items()},
        "preflight_gates": {k: (preflight["stages"].get(s) or {}).get(f"gate_{k}", {}).get("verdict")
                            for k, s in (("G1", "refs"), ("G2", "refs"), ("G3", "orient"))},
        "n_frames": pack["n_frames"], "cells": list(CONTRAST_CELLS)})

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

    def on_progress(p):
        p = dict(p)
        p["elapsed_s"] = round(time.perf_counter() - t0, 1)
        p["seconds_per_frame"] = round(p["elapsed_s"] / max(1, p["frames_done"]), 2)
        bank.stage("progress", p)

    offline = cells_mod.run_offline(
        actor, pack, ladder, cells=CONTRAST_CELLS, per_cell_frames=args.per_cell_frames,
        expect_bins={k: v["bins"] for k, v in gate_refs.items()},
        expect_reference={k: v["a_norm"] for k, v in gate_refs.items()},
        on_progress=on_progress)
    bank.artifact("records.json", offline["records"])
    bank.artifact("diagnostics.json", offline["diagnostics"])
    bank.artifact("per_cell_maps.json", offline["per_cell_maps"])
    bank.artifact("samples.json", offline["samples"])
    extremes = cells_mod.bank_extremes(actor, pack, ladder, offline["records"],
                                      cells=("target", "bg", "random", "target_swap", "bg_swap"))
    bank.artifact("samples_extremes.json", extremes)

    # cross-session agreement, before any contrast number is read
    diag = offline["diagnostics"]
    bins_diff = [d["cells_differing_from_earlier_session"] for d in diag
                 if d["cells_differing_from_earlier_session"] is not None]
    ref_drift = [d["reference_drift_vs_earlier_session"] for d in diag
                 if d["reference_drift_vs_earlier_session"] is not None]
    cross = {
        "frames_compared": len(bins_diff),
        "frames_with_identical_grid": int(sum(1 for x in bins_diff if x == 0)),
        "max_cells_differing": int(max(bins_diff)) if bins_diff else None,
        "reference_drift_max": float(max(ref_drift)) if ref_drift else None,
        "reference_drift_median": float(np.median(ref_drift)) if ref_drift else None,
        "reproducible": bool(bins_diff and max(bins_diff) == 0
                             and ref_drift and max(ref_drift) <= args.drift_tol),
        "drift_tol": args.drift_tol,
    }
    bank.stage("cross_session", cross)

    merged = {
        "records": list(gate_records) + list(offline["records"]),
        "diagnostics": diag,
        "per_cell_maps": offline["per_cell_maps"],
        "samples": offline["samples"],
        "references": offline["references"],
    }
    bank.artifact("merged_record_count.json", {"gates": len(gate_records),
                                               "contrast": len(offline["records"]),
                                               "merged": len(merged["records"])})

    gates: Dict[str, core.Gate] = {
        "G0": core.gate_g0(max(d["g0_max_abs_delta_norm"] for d in diag), len(diag)),
        "G1": _gate_from_bank(preflight, "refs", "G1"),
        "G2": _gate_from_bank(preflight, "refs", "G2"),
        "G3": _gate_from_bank(preflight, "orient", "G3"),
        "G4a": _gate_from_freeze(freeze, "G4a"),
        "G4b": _gate_from_freeze(freeze, "G4b"),
    }
    agg = cells_mod.aggregate(merged, gates, require_rule2=True, frozen_m=freeze["m"])
    bank.stage("aggregate", agg)
    bank.artifact("aggregate.json", agg)

    figures = _figures(bank, merged, agg, ladder, preflight)
    bank.stage("figures", figures)

    verdict = {
        "label": agg["decision"]["label"],
        "notes": agg["decision"]["notes"],
        "m_frozen": freeze["m"], "m_matches_frozen": agg["m_matches_frozen"],
        "gates": {k: v.verdict for k, v in gates.items()},
        "cross_session_reproducible": cross["reproducible"],
        "tail_mismatches": agg["tail_mismatches"],
        "primary_paired_k8": agg["per_k"].get("8", {}).get("paired_target_minus_bg"),
        "primary_paired_k-8": agg["per_k"].get("-8", {}).get("paired_target_minus_bg"),
        "primary_contrast": agg["decision"]["rule2_read_as_map"].get("primary_contrast"),
        "pivot_swap": agg.get("pivot_swap"),
        "directional": agg["directional"].get("target"),
        "anchor": agg["anchor"],
        "seconds_total": round(time.perf_counter() - t0, 1),
        "calls": actor.calls,
    }
    bank.top("verdict", verdict)
    print(json.dumps(verdict, indent=1, default=str), flush=True)


def _gate_from_bank(preflight: Dict[str, object], stage: str, name: str) -> core.Gate:
    payload = ((preflight.get("stages") or {}).get(stage) or {}).get(f"gate_{name}")
    if not payload:
        return core.Gate(name, core.CANCELLED, None, None, None,
                         {"why": f"the preflight bank has no {name} (stage {stage} did not run)"})
    return core.Gate(name=payload["name"], verdict=payload["verdict"], value=payload["value"],
                     floor=payload["floor"], ceiling=payload["ceiling"],
                     detail=dict(payload.get("detail") or {}, source=f"preflight.{stage}"))


def _gate_from_freeze(freeze: Dict[str, object], name: str) -> core.Gate:
    payload = (freeze.get("gates") or {}).get(name)
    if not payload:
        return core.Gate(name, core.CANCELLED, None, None, None, {"why": "not in freeze.json"})
    return core.Gate(name=payload["name"], verdict=payload["verdict"], value=payload["value"],
                     floor=payload["floor"], ceiling=payload["ceiling"],
                     detail=dict(payload.get("detail") or {}, source="gates.freeze"))


def _figures(bank: sh.Bank, merged, agg, ladder, preflight) -> Dict[str, object]:
    from infra.pg import depthanchor_figures as figs
    fidelity = None
    rows = ((preflight.get("stages") or {}).get("refs") or {}).get("fidelity_per_frame")
    if rows:
        fidelity = {"per_frame": rows}
    made = figs.draw_all(bank.out / "figures", offline=merged, aggregate=agg, ladder=ladder,
                         gates=agg["gates"], fidelity=fidelity)
    return {"drawn": sorted(made), "paths": made}


if __name__ == "__main__":
    main()
