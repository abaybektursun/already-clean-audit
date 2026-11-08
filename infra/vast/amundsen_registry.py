#!/usr/bin/env python3
"""AMUNDSEN artifact registry — every reported number back to its input hash.

Walks the banked per-cell `_done.json` markers (which already carry a sha256 per
artifact, written by run_cells at bank time), pairs them with the verdict file
each arm was scored into, and emits three closeout artifacts required by §10:

  amundsen_registry.json   arm -> {verdict token, S3 prefix, per-artifact sha256,
                                   scorer + code SHA, register tag}
  amundsen_metrics.json    the numbers each verdict rests on, arm by arm
  amundsen_controls.json   every control/validity gate and whether it held

Register tags: CONFIRMATORY for an arm scored by a rule frozen before its run;
DISCOVERY for anything computed after seeing data (each such entry names itself).

  python3 amundsen_registry.py --out ../../studies/lineage/paligemma1-pi0/research
"""
import argparse
import json
import os
import subprocess

S3 = "s3://pg-lineage-769063704696/results"
AWS = ["aws", "--profile", "fuelos"]

ARMS = {
    "AM-1": {"rental": "am1_abl14b", "verdict": "amundsen_am1_verdict.json",
             "question": "is 'band LIVE at 14B' a core statement or an energy statement?"},
    "AM-2": {"rental": "am2_lrsweep", "verdict": "amundsen_am2_verdict.json",
             "question": "is the rotated arm's wikitext-ppl cost intrinsic or untuned lr?"},
    "AM-3": {"rental": "am3_softread", "verdict": "amundsen_am3_verdict.json",
             "question": "does the model READ the sink coordinates at inference?"},
    "AM-4": {"rental": "am4_llama70b", "verdict": "amundsen_am4_verdict.json",
             "question": "does the 72B first-token relocation replicate off-family?"},
    "AM-5": {"rental": "am5_cont3000", "verdict": "amundsen_am5_verdict.json",
             "question": "scar or slow re-formation at >= 3000 steps?"},
    "AM-5b": {"rental": "am5b_cont5000", "verdict": "amundsen_am5b_verdict.json",
              "question": "the §6.5 pre-authorised continuation to 5000 steps"},
    "AM-6": {"rental": "am6_v3seeds", "verdict": "amundsen_am6_verdict.json",
             "question": "is V3's wrong-signed noise trend real?"},
    "AM-1b": {"rental": "am1b_spectral", "verdict": "amundsen_am1b_verdict.json",
              "question": "EXPLORATORY rider: does AM-1 survive a SPECTRUM-matched control?"},
    "AM-7": {"rental": "am7_evals", "verdict": "amundsen_am7_verdict.json",
             "question": "does de-sinking buy any actual task capability?"},
}

# the numbers each arm's verdict rests on: (json path in the verdict file, label)
METRICS = {
    "AM-1": [("C.mean", "mean paired dppl core32"), ("E.mean", "mean paired dppl energymatched32"),
             ("planted.mean", "mean paired dppl magmatched32"), ("E_over_C.ratio", "E/C"),
             ("E_over_C.lo", "E/C CI lo"), ("E_over_C.hi", "E/C CI hi")],
    "AM-2": [("ratio_best_rotated_over_best_unrotated", "best-rotated / best-unrotated ppl"),
             ("R", "best rotated ppl"), ("U", "best unrotated ppl"),
             ("n_seeds_ratio_gt_1", "seeds with ratio > 1")],
    "AM-3": [("spearman_signed.rho", "signed Spearman rho"),
             ("spearman_signed.p", "signed Spearman p"),
             ("spearman_magnitude.rho", "magnitude Spearman rho"),
             ("spearman_magnitude.p", "magnitude Spearman p")],
    "AM-4": [("early_max_min_over_batches", "max early-layer (L<=2) concentration"),
             ("late_max_min_over_batches", "max last-3-layer concentration")],
    "AM-5": [("ratio", "hadamard/unrotated fresh-bottom-32 motion share"),
             ("loglog_slope_last3", "log-log slope, last 3 checkpoints")],
    "AM-5b": [("ratio", "hadamard/unrotated fresh-bottom-32 motion share"),
              ("loglog_slope_last3", "log-log slope, last 3 checkpoints")],
    "AM-1b": [("S_over_C.ratio", "specmatched / core dppl"), ("S_over_C.lo", "CI lo"),
              ("S_over_C.hi", "CI hi"),
              ("arms.specmatched32.median_energy_share", "specmatched removed energy share"),
              ("arms.core32.median_energy_share", "core32 removed energy share")],
    "AM-6": [("n_sign_kept", "cells keeping the banked slope sign"),
             ("n_cells", "cells"), ("threshold_real", "REAL bar")],
    "AM-7": [("benchmarks.mmlu_blankimg.diff", "MMLU blank-image acc diff"),
             ("benchmarks.mmlu_cloze.diff", "MMLU cloze acc diff")],
}

CONTROLS = {
    "AM-1": "validity", "AM-2": "coverage", "AM-3": "validity", "AM-4": None,
    "AM-5": "validity", "AM-5b": "validity", "AM-6": "haar_flat", "AM-7": "bh_reject",
}


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def cell_hashes(rental):
    r = subprocess.run([*AWS, "s3", "ls", "--recursive", f"{S3}/amundsen_{rental}/cells/"],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        key = line.split()[-1]
        if not key.endswith("_done.json"):
            continue
        cid = key.split("/cells/")[1].split("/")[0]
        j = subprocess.run([*AWS, "s3", "cp", f"s3://pg-lineage-769063704696/{key}", "-"],
                           capture_output=True, text=True)
        if j.returncode == 0:
            d = json.loads(j.stdout)
            out[cid] = {"seconds": d.get("seconds"),
                        "artifacts": {k: v["sha256"][:16] for k, v in d["artifacts"].items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-sha", default=None)
    a = ap.parse_args()
    sha = a.code_sha or subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                       capture_output=True, text=True).stdout.strip()
    reg, met, con = {}, {}, {}
    for arm, info in ARMS.items():
        vpath = os.path.join(a.out, info["verdict"])
        v = json.load(open(vpath)) if os.path.exists(vpath) else None
        reg[arm] = {
            "question": info["question"],
            "verdict": (v or {}).get("verdict", "NOT-RUN"),
            "s3_prefix": f"{S3}/amundsen_{info['rental']}",
            "verdict_artifact": info["verdict"] if v else None,
            "scored_by": f"infra/vast/analysis_amundsen.py::{arm.lower().replace('-', '')} @ {sha}",
            "collected_by": f"infra/vast/collect_amundsen.py @ {sha}",
            "register": ("EXPLORATORY (rider, expectation registered before the run)"
                         if arm.endswith("b") and arm != "AM-5b" else
                         "CONFIRMATORY (rule frozen before the run)") if v else "NOT-RUN",
            "cells": cell_hashes(info["rental"]) if v else {},
        }
        if v:
            met[arm] = {label: dig(v, path) for path, label in METRICS.get(arm, [])}
            key = CONTROLS.get(arm)
            con[arm] = {"gate_block": key, "value": v.get(key) if key else None,
                        "verdict_consumed_the_gate": True}
    for name, obj in (("amundsen_registry.json", reg), ("amundsen_metrics.json", met),
                      ("amundsen_controls.json", con)):
        p = os.path.join(a.out, name)
        json.dump(obj, open(p, "w"), indent=2, default=float)
        print("wrote", p)


if __name__ == "__main__":
    main()
