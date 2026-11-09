"""m09: the gradient prediction of switching a side off, and what 16-bit broke.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.M09_DATA = {...}. Prints `path sha256` for every input.

Holds, per switched-off pair of both halves, the four numbers the picture toggles between (16-bit prediction, 16-bit
measurement, 32-bit measurement, 32-bit prediction), the rank agreements and medians recomputed from those rows, and
the two facts behind the 32-bit method: the memory of a 32-bit backward pass against the GPU's, and forward mode
against reverse mode on a small model. f09 holds the same rows with more fields and the hover text.

    python3 extract.py
"""
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []

BANK = "studies/next/n2v/banks/score-02/n2v"
VERDICT = "studies/next/n2v/verdict_n2v.json"
THRESH = "studies/next/n2v/thresholds.json"
N12_THRESH = "studies/next/n12/thresholds.json"
N12_VERDICT = "studies/next/n12/verdict_n12.json"
N12_GVAL = "studies/next/n12/read/r2_gval.json"
MEM = "studies/next/n2v/offline_gate/mem_fp32.json"
DRYRUN = "studies/next/n2v/offline_gate/dryrun.log"
HALF = {"freeze": "rule_setting", "test": "held_out"}
ROLE = ["carrier", "other side ranked 1st", "other side ranked 2nd", "other side ranked 3rd"]


def src(rel):
    """Open a repository file by its path from the root, record its sha256 once, print `path sha256`."""
    p = ROOT / rel
    b = p.read_bytes()
    assert not b.startswith(b"version https://git-lfs"), f"{rel} is an LFS pointer: git lfs pull --include={rel}"
    if rel not in [s["path"] for s in SOURCES]:
        h = hashlib.sha256(b).hexdigest()
        SOURCES.append({"path": rel, "sha256": h})
        print(rel, h)
    return p


def compare(name, recomputed, banked, where, tol):
    """Store a recomputed summary beside the banked one; print both and whether they differ beyond rounding."""
    same = abs(recomputed - banked) <= tol
    print(f"CHECK {name}: recomputed {recomputed} banked {banked} ({where}) {'same' if same else 'DIFFERS'}")
    return {"recomputed": recomputed, "banked": banked, "banked_at": where, "same_within": tol, "same": same}


def write(data, notes):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    data["meta"] = {"commit": commit, "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def rankdata(x):
    """Average ranks, ties share the mean rank (n12._rankdata)."""
    o = np.argsort(x, kind="stable")
    r = np.empty(x.size)
    r[o] = np.arange(x.size, dtype=np.float64)
    _, start, cnt = np.unique(x[o], return_index=True, return_counts=True)
    for s0, c in zip(start, cnt):
        if c > 1:
            r[o[s0:s0 + c]] = s0 + (c - 1) / 2.0
    return r


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def col(rows, k):
    return np.array([r[k] for r in rows], np.float64)


def main():
    rows = {h: [json.loads(l) for l in src(f"{BANK}/n2v_{h}.jsonl").read_text().splitlines() if l.strip()]
            for h in HALF}
    F, T = rows["freeze"], rows["test"]
    assert len(F) == 328 and len(T) == 284
    v = json.loads(src(VERDICT).read_text())["test"]
    th = json.loads(src(THRESH).read_text())
    n12_th = json.loads(src(N12_THRESH).read_text())
    n12_v = json.loads(src(N12_VERDICT).read_text())["test"]
    gval = json.loads(src(N12_GVAL).read_text())
    mem = json.loads(src(MEM).read_text())
    line = next(l for l in src(DRYRUN).read_text().splitlines()
                if l.startswith("CHECK INSTR forward-mode prediction equals reverse-mode"))
    m = re.search(r"PASS neuron (\d+): jvp ([0-9.]+) reverse ([0-9.]+)", line)
    assert m, line
    fwd, rev = float(m[2]), float(m[3])

    fields = [("half", None), ("birth", "unit"), ("side", "pole"), ("place_in_list", "rank"),
              ("prediction_16bit", "pred_bf16"), ("measurement_16bit", "meas_bf16"),
              ("measurement_32bit", "dL"), ("prediction_32bit", "pred_jvp")]
    pairs = {f: [HALF[h] if k is None else r[k] for h in HALF for r in rows[h]] for f, k in fields}
    pairs["role_of_place"] = ROLE

    va, vf = v["window"]["validation"], th["freeze"]["window"]["validation"]
    assert va["prediction"] == vf["prediction"] == "pred_bf16" and va["measurement"] == vf["measurement"] == "dL"
    a, b = col(T, "pred_bf16"), col(T, "pred_jvp")
    both = F + T
    data = {
        "pairs": pairs,
        "toggle_16bit_measurement": {
            "rank_agreement_rule_setting": compare("rank agreement, 16-bit prediction vs 16-bit measurement, rule-setting",
                                                   spearman(col(F, "pred_bf16"), col(F, "meas_bf16")),
                                                   n12_th["validation_freeze"]["rho"],
                                                   f"{N12_THRESH} -> validation_freeze.rho", 1e-12),
            "rank_agreement_held_out": compare("rank agreement, 16-bit prediction vs 16-bit measurement, held-out",
                                               spearman(col(T, "pred_bf16"), col(T, "meas_bf16")),
                                               n12_v["validation"]["rho"], f"{N12_VERDICT} -> test.validation.rho", 1e-12),
            "median_abs_measurement_both_halves": compare("median |16-bit measurement|, both halves",
                                                          float(np.median(np.abs(col(both, "meas_bf16")))),
                                                          gval["both"]["abs_dL"]["0.5"],
                                                          f"{N12_GVAL} -> both.abs_dL.0.5", 1e-12)},
        "toggle_32bit_measurement": {
            "rank_agreement_rule_setting": compare("rank agreement, 16-bit prediction vs 32-bit measurement, rule-setting",
                                                   spearman(col(F, "pred_bf16"), col(F, "dL")), vf["rho"],
                                                   f"{THRESH} -> freeze.window.validation.rho", 1e-12),
            "rank_agreement_held_out": compare("rank agreement, 16-bit prediction vs 32-bit measurement, held-out",
                                               spearman(col(T, "pred_bf16"), col(T, "dL")), va["rho"],
                                               f"{VERDICT} -> test.window.validation.rho", 1e-12),
            "band_held_out": [va["lo"], va["hi"]], "shuffle_reference_held_out": va["shuffle_q975"],
            "median_abs_measurement_held_out": compare("median |32-bit measurement|, held-out",
                                                       float(np.median(np.abs(col(T, "dL")))), va["median_abs_meas"],
                                                       f"{VERDICT} -> test.window.validation.median_abs_meas", 1e-12)},
        "prediction": {
            "median_abs_16bit_both_halves": compare("median |16-bit prediction|, both halves",
                                                    float(np.median(np.abs(col(both, "pred_bf16")))),
                                                    gval["both"]["abs_dLhat"]["0.5"], f"{N12_GVAL} -> both.abs_dLhat.0.5",
                                                    1e-9),
            "rank_agreement_16bit_vs_32bit_held_out": compare("rank agreement, 16-bit vs 32-bit prediction, held-out",
                                                              spearman(a, b), v["bf16_vs_jvp_prediction"]["spearman"],
                                                              f"{VERDICT} -> test.bf16_vs_jvp_prediction.spearman", 1e-12),
            "median_relative_difference_held_out": compare(
                "median |16-bit - 32-bit| / |32-bit| prediction, held-out",
                float(np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))),
                v["bf16_vs_jvp_prediction"]["median_rel_diff"],
                f"{VERDICT} -> test.bf16_vs_jvp_prediction.median_rel_diff", 1e-12)},
        "threshold_set_in_advance": th["rules"]["val_rho_min"],
        "why_forward_mode": {
            "backward_32bit_peak_GB_layer0": mem["peak_lowest_layer_0_GB"]["fp32_math"],
            "backward_32bit_peak_GB_max_over_births": mem["fp32_math_real_units"]["max_peak_GB"],
            "gpu_memory_GB": mem["l40s_GB"],
            "births_that_would_fit": mem["fp32_math_real_units"]["fit_under_l40s_less_2GB"],
            "births": mem["fp32_math_real_units"]["n"],
            "small_model_check": {"unit": int(m[1]), "forward_mode": fwd, "reverse_mode": rev,
                                  "relative_difference": abs(fwd - rev) / abs(rev)}},
    }
    for num, key, val in [
            ("0.100", "toggle_16bit_measurement.rank_agreement_rule_setting",
             data["toggle_16bit_measurement"]["rank_agreement_rule_setting"]["recomputed"]),
            ("0.043", "toggle_16bit_measurement.rank_agreement_held_out",
             data["toggle_16bit_measurement"]["rank_agreement_held_out"]["recomputed"]),
            ("0.65", "toggle_16bit_measurement.median_abs_measurement_both_halves",
             data["toggle_16bit_measurement"]["median_abs_measurement_both_halves"]["recomputed"]),
            ("0.960", "toggle_32bit_measurement.rank_agreement_held_out",
             data["toggle_32bit_measurement"]["rank_agreement_held_out"]["recomputed"]),
            ("0.915 to 0.990", "toggle_32bit_measurement.band_held_out", (va["lo"], va["hi"])),
            ("0.022", "prediction.median_abs_16bit_both_halves",
             data["prediction"]["median_abs_16bit_both_halves"]["recomputed"]),
            ("0.995", "prediction.rank_agreement_16bit_vs_32bit_held_out",
             data["prediction"]["rank_agreement_16bit_vs_32bit_held_out"]["recomputed"]),
            ("2.6 %", "prediction.median_relative_difference_held_out",
             data["prediction"]["median_relative_difference_held_out"]["recomputed"]),
            ("67.2 GB", "why_forward_mode.backward_32bit_peak_GB_layer0", mem["peak_lowest_layer_0_GB"]["fp32_math"]),
            ("48.3 GB", "why_forward_mode.gpu_memory_GB", mem["l40s_GB"]),
            ("3e-6", "why_forward_mode.small_model_check.relative_difference", abs(fwd - rev) / abs(rev))]:
        print(f"PLAN {num} -> {key} ({val})")
    notes = {
        "pairs": "One array per field, index i = one (birth, switched-off side) pair: first the 328 rows of "
                 "n2v_freeze.jsonl (half = rule_setting), then the 284 rows of n2v_test.jsonl (half = held_out), in "
                 "file order.",
        "place_in_list": "rank in the bank: 0 the carrier, 1 to 3 the three other sides of the layer with the highest "
                         "ranking score one snapshot before the birth.",
        "prediction_16bit": "pred_bf16: N12's sum over the group's positions of -h(i) G(i), G the derivative of the "
                            "window's summed loss with respect to the side's input to the down matrix; 16-bit forward "
                            "and backward; nats summed over the birth's windows. The picture's x axis in both states.",
        "measurement_16bit": "meas_bf16: N12's 16-bit window-summed switch-off measurement (per-token 16-bit rows were "
                             "never saved).",
        "measurement_32bit": "dL: the 32-bit window-summed switch-off measurement (N2V).",
        "prediction_32bit": "pred_jvp: the same prediction computed in 32-bit by forward-mode differentiation along -h.",
        "rank agreements": "Spearman rank correlation with average ranks for ties (n12.spearman), recomputed from the "
                           "rows. The band (percentile bootstrap over births) and the shuffle reference (97.5th "
                           "percentile over 1,000 shuffles) are banked, not recomputed.",
        "median_abs_measurement_both_halves": "The 0.65 nats of READ_n12.md lines 23-27: median |meas_bf16| over both "
                                              "halves (612 pairs).",
        "median_abs_16bit_both_halves": "The 0.022 nats of READ_n12.md: median |pred_bf16| over both halves "
                                        "(0.0216; the held-out half alone is 0.0219).",
        "threshold_set_in_advance": "The rank agreement the check needed: 0.7 (n2v thresholds.json rules.val_rho_min).",
        "why_forward_mode": "mem_fp32.json: peak memory of a 32-bit backward pass for a layer-0 birth "
                            "(peak_lowest_layer_0_GB.fp32_math) and the largest over the 153 births, against the "
                            "L40S's 48.31 GB (l40s_GB); births_that_would_fit = fit_under_l40s_less_2GB.",
        "small_model_check": "The dry run's INSTR line in n2v/offline_gate/dryrun.log: forward-mode prediction against "
                             "reverse-mode sum of -h G on a tiny 32-bit model, one unit; relative_difference = "
                             "|forward - reverse| / |reverse| of the logged 7-digit values.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
