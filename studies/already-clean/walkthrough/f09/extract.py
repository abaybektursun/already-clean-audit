"""f09: the 16-bit gradient prediction against the 16-bit and the 32-bit switch-off measurement, per switched-off side.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.F09_DATA = {...}. Prints `path sha256` for every input.

Rows: N2V's per-pair records, 328 on the half that set the rules and 284 on the held-out half (n2v_<half>.jsonl). Each
pair is a birth and one of four switched-off sides: the carrier, then the three other sides of the layer that scored
highest one snapshot before the birth. The rank agreements, medians and the 16-bit against 32-bit prediction comparison
are recomputed from the rows and compared with the banked values; pred_bf16 and meas_bf16 are checked against N12's
own records (n12_<half>.jsonl, abl = [side, prediction, measurement]).

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
ASSETS = "studies/next/00_ASSETS.md"   # the sixteen stage-1 checkpoints and their branches (as m03, m05)

BANK = "studies/next/n2v/banks/score-02/n2v"
N12_BANK = "studies/next/n12/banks/score-02/n12"
VERDICT = "studies/next/n2v/verdict_n2v.json"
THRESH = "studies/next/n2v/thresholds.json"
N12_THRESH = "studies/next/n12/thresholds.json"
N12_VERDICT = "studies/next/n12/verdict_n12.json"
N12_GVAL = "studies/next/n12/read/r2_gval.json"
BIRTHS = "studies/next/m2/births.json"
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
    data["meta"] = {"commit": commit, "link": "studies/next/n2v/banks/score-02/n2v/n2v_test.jsonl",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def jsonl(rel):
    return [json.loads(l) for l in src(rel).read_text().splitlines() if l.strip()]


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
    rows = {h: jsonl(f"{BANK}/n2v_{h}.jsonl") for h in HALF}
    assert len(rows["freeze"]) == 328 and len(rows["test"]) == 284
    v = json.loads(src(VERDICT).read_text())["test"]
    th = json.loads(src(THRESH).read_text())
    n12_th = json.loads(src(N12_THRESH).read_text())
    n12_v = json.loads(src(N12_VERDICT).read_text())["test"]
    gval = json.loads(src(N12_GVAL).read_text())
    names = {b["id"]: b for b in json.loads(src(BIRTHS).read_text())["births"]}

    # N12's own records carry the 16-bit prediction and measurement; they must equal the N2V copies
    n12_check = {}
    for h in HALF:
        abl = {(r["unit"], p[0]): p for r in jsonl(f"{N12_BANK}/n12_{h}.jsonl") if r["kind"] == "real" for p in r["abl"]}
        bad = [r["pole"] for r in rows[h] if abl[(r["unit"], r["pole"])][1:] != [r["pred_bf16"], r["meas_bf16"]]]
        assert len(abl) == len(rows[h])
        n12_check[HALF[h]] = {"pairs": len(rows[h]), "pairs_differing": len(bad)}
        print(f"CHECK {h}: {len(rows[h])} pairs, {len(bad)} differ from N12's abl entries")
        assert not bad, bad

    # the pair table, one array per field (column order = rule-setting rows, then held-out rows, as banked)
    fields = [("half", None), ("birth", "unit"), ("side", "pole"), ("place_in_list", "rank"), ("layer", "layer"),
              ("snapshot_read", "t"), ("unit_windows", "n_windows"), ("prediction_16bit", "pred_bf16"),
              ("measurement_16bit", "meas_bf16"), ("measurement_32bit", "dL"), ("prediction_32bit", "pred_jvp"),
              ("measurement_own_positions_32bit", "dL_own"), ("prediction_own_positions_32bit", "pred_jvp_own"),
              ("rounding_check", "probe"), ("losses_after_first_position", "n_later"),
              ("sum_abs_input_at_positions", "sum_abs_h")]
    pairs = {f: [HALF[h] if k is None else r[k] for h in HALF for r in rows[h]] for f, k in fields}
    pairs["role_of_place"] = ROLE

    F, T = rows["freeze"], rows["test"]
    va, vf = v["window"]["validation"], th["freeze"]["window"]["validation"]
    assert va["prediction"] == vf["prediction"] == "pred_bf16" and va["measurement"] == vf["measurement"] == "dL"
    agree = {
        "measurement_16bit": {
            "rule_setting": compare("rank agreement, 16-bit prediction vs 16-bit measurement, rule-setting half",
                                    spearman(col(F, "pred_bf16"), col(F, "meas_bf16")),
                                    n12_th["validation_freeze"]["rho"], f"{N12_THRESH} -> validation_freeze.rho", 1e-12),
            "held_out": compare("rank agreement, 16-bit prediction vs 16-bit measurement, held-out half",
                                spearman(col(T, "pred_bf16"), col(T, "meas_bf16")),
                                n12_v["validation"]["rho"], f"{N12_VERDICT} -> test.validation.rho", 1e-12),
            "band_held_out": [n12_v["validation"]["lo"], n12_v["validation"]["hi"]],
            "shuffle_reference_held_out": n12_v["validation"]["shuffle_q975"]},
        "measurement_32bit": {
            "rule_setting": compare("rank agreement, 16-bit prediction vs 32-bit measurement, rule-setting half",
                                    spearman(col(F, "pred_bf16"), col(F, "dL")), vf["rho"],
                                    f"{THRESH} -> freeze.window.validation.rho", 1e-12),
            "held_out": compare("rank agreement, 16-bit prediction vs 32-bit measurement, held-out half",
                                spearman(col(T, "pred_bf16"), col(T, "dL")), va["rho"],
                                f"{VERDICT} -> test.window.validation.rho", 1e-12),
            "band_held_out": [va["lo"], va["hi"]], "shuffle_reference_held_out": va["shuffle_q975"],
            "band_rule_setting": [vf["lo"], vf["hi"]], "shuffle_reference_rule_setting": vf["shuffle_q975"]},
        "threshold_set_in_advance": th["rules"]["val_rho_min"],
    }
    a, b = col(T, "pred_bf16"), col(T, "pred_jvp")
    pred = {"rank_agreement": compare("rank agreement, 16-bit vs 32-bit prediction, held-out half", spearman(a, b),
                                      v["bf16_vs_jvp_prediction"]["spearman"],
                                      f"{VERDICT} -> test.bf16_vs_jvp_prediction.spearman", 1e-12),
            "median_relative_difference": compare("median |16-bit - 32-bit| / |32-bit| prediction, held-out half",
                                                  float(np.median(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))),
                                                  v["bf16_vs_jvp_prediction"]["median_rel_diff"],
                                                  f"{VERDICT} -> test.bf16_vs_jvp_prediction.median_rel_diff", 1e-12)}
    both = F + T
    med = {
        "abs_measurement_16bit_both_halves": compare("median |16-bit measurement|, both halves",
                                                     float(np.median(np.abs(col(both, "meas_bf16")))),
                                                     gval["both"]["abs_dL"]["0.5"], f"{N12_GVAL} -> both.abs_dL.0.5", 1e-12),
        "abs_prediction_16bit_both_halves": compare("median |16-bit prediction|, both halves",
                                                    float(np.median(np.abs(col(both, "pred_bf16")))),
                                                    gval["both"]["abs_dLhat"]["0.5"], f"{N12_GVAL} -> both.abs_dLhat.0.5",
                                                    1e-9),
        "abs_prediction_16bit_held_out": compare("median |16-bit prediction|, held-out half",
                                                 float(np.median(np.abs(col(T, "pred_bf16")))), va["median_abs_pred"],
                                                 f"{VERDICT} -> test.window.validation.median_abs_pred", 1e-12),
        "abs_measurement_32bit_held_out": compare("median |32-bit measurement|, held-out half",
                                                  float(np.median(np.abs(col(T, "dL")))), va["median_abs_meas"],
                                                  f"{VERDICT} -> test.window.validation.median_abs_meas", 1e-12),
        "abs_rounding_check_held_out": compare("median |rounding check| over held-out births",
                                               float(np.median([abs(r["probe"]) for r in T if r["rank"] == 0])),
                                               v["probe_median_abs"], f"{VERDICT} -> test.probe_median_abs", 1e-15)}

    hover = []
    for r in jsonl(f"{BANK}/samples/raw_examples_test.jsonl"):
        hover.append({"birth": r["unit"], "half": HALF[r["half"]], "snapshot_read": r["t"],
                      "group_name": names[r["unit"]]["name"], "group_contexts": r["exemplars"]})
    births = {u: {"group_name": b["name"], "layer": b["layer"], "birth_snapshot": b["k"], "half": HALF[b["half"]]}
              for u, b in names.items()}
    assert set(pairs["birth"]) == set(births)

    data = {"pairs": pairs, "rank_agreement": agree, "prediction_16bit_vs_32bit_held_out": pred, "medians": med,
            "check_against_n12_records": n12_check, "births": births, "hover_examples_held_out": hover}
    for num, key, val in [("0.100", "rank_agreement.measurement_16bit.rule_setting", agree["measurement_16bit"]["rule_setting"]),
                          ("0.043", "rank_agreement.measurement_16bit.held_out", agree["measurement_16bit"]["held_out"]),
                          ("0.960", "rank_agreement.measurement_32bit.held_out", agree["measurement_32bit"]["held_out"]),
                          ("0.995", "prediction_16bit_vs_32bit_held_out.rank_agreement", pred["rank_agreement"]),
                          ("2.6 %", "prediction_16bit_vs_32bit_held_out.median_relative_difference",
                           pred["median_relative_difference"]),
                          ("0.65", "medians.abs_measurement_16bit_both_halves", med["abs_measurement_16bit_both_halves"]),
                          ("0.022", "medians.abs_prediction_16bit_both_halves", med["abs_prediction_16bit_both_halves"]),
                          ("1.47e-4", "medians.abs_rounding_check_held_out", med["abs_rounding_check_held_out"])]:
        print(f"PLAN {num} -> {key}.recomputed ({val['recomputed']})")
    print(f"PLAN 0.915 to 0.990 -> rank_agreement.measurement_32bit.band_held_out ({va['lo']}, {va['hi']})")
    print(f"PLAN 0.118 -> rank_agreement.measurement_32bit.shuffle_reference_held_out ({va['shuffle_q975']})")
    print(f"PLAN 0.7 -> rank_agreement.threshold_set_in_advance ({th['rules']['val_rho_min']})")
    print(f"PLAN 328 / 284 -> pairs (rule_setting {len(F)}, held_out {len(T)})")
    notes = {
        "pairs": "One array per field, index i = one (birth, switched-off side) pair: first the 328 rows of "
                 "n2v_freeze.jsonl (half = rule_setting, 82 births), then the 284 rows of n2v_test.jsonl (half = "
                 "held_out, 71 births), in file order; 4 sides per birth.",
        "place_in_list": "rank in the bank: 0 the carrier, 1 to 3 the three other sides of the layer with the "
                         "highest ranking score one snapshot before the birth (N12's switched-off list); "
                         "role_of_place names them.",
        "prediction_16bit": "pred_bf16: N12's first-order prediction, the sum over the group's positions of -h G, 16-bit "
                            "forward and 16-bit backward, nats summed over the birth's windows.",
        "measurement_16bit": "meas_bf16: N12's 16-bit window-summed switch-off measurement (abl[i][2]); the 16-bit "
                             "per-token rows were never saved.",
        "measurement_32bit": "dL: the 32-bit switch-off measurement, sum over each unit window's 2,048 losses of "
                             "(switched off - clean), summed over the birth's windows, nats.",
        "prediction_32bit": "pred_jvp: the 32-bit forward-mode prediction of the same quantity.",
        "measurement_own_positions_32bit": "dL_own: the same sum over the group's own positions only.",
        "prediction_own_positions_32bit": "pred_jvp_own: the 32-bit forward-mode prediction of dL_own.",
        "rounding_check": "probe: the carrier's output change added and subtracted in one forward (zero in exact "
                          "arithmetic), its summed loss change; the same value on the four rows of a birth.",
        "losses_after_first_position": "n_later: the number of loss terms at or after the first switched-off position, "
                                       "summed over the birth's windows.",
        "sum_abs_input_at_positions": "sum_abs_h: the sum of |h|, the side's input to the down matrix, over the group's "
                                      "positions in 32-bit.",
        "snapshot_read": "t in the bank: the snapshot before the birth, at which every row is read.",
        "rank_agreement": "Spearman rank correlation with average ranks for ties (n12.spearman), recomputed from the rows; "
                          "bands (percentile bootstrap over births) and shuffle references (97.5th percentile over "
                          "1,000 shuffles) are banked values, not recomputed.",
        "threshold_set_in_advance": "The rank agreement the check needed (0.7; n2v thresholds.json rules.val_rho_min, "
                                    "N12's bar raised from 0.5).",
        "medians": "Median absolute values recomputed from the rows; the 16-bit measurement's 0.65 and the "
                   "prediction's 0.022 are over both halves (READ_n12.md lines 23-27; r2_gval.json both).",
        "prediction_16bit_vs_32bit_held_out": "Spearman of pred_bf16 with pred_jvp and the median of "
                                              "|pred_bf16 - pred_jvp| / |pred_jvp| over the 284 held-out pairs.",
        "check_against_n12_records": "Every pair's pred_bf16 and meas_bf16 equal N12's abl entry for that side in "
                                     "n12_<half>.jsonl (count of differing pairs).",
        "births": "Group name, layer, birth snapshot and half for the 153 births, verbatim from m2/births.json.",
        "hover_examples_held_out": "The 10 held-out births of n2v samples/raw_examples_test.jsonl; group_contexts are the "
                                   "banked exemplars verbatim, the group's position marked [[ ]].",
    }
    # the training step of each checkpoint, so that the page names a checkpoint by its step and branch
    steps = [int(re.search(r"`stage1-step(\d+)`", s)[1]) for s in src(ASSETS).read_text().splitlines()
             if re.match(r"\| T_\d+ \| `stage1-step", s)]
    assert len(steps) == 16, steps
    data["snapshot_steps"] = steps
    notes["snapshot_steps"] = ("Training step of each of the sixteen stage-1 checkpoints 0 ... 15; the checkpoint's Hugging Face "
                               "branch is stage1-step<step> (studies/next/00_ASSETS.md, the table rows T_0 ... T_15).")
    write(data, notes)


if __name__ == "__main__":
    main()
