"""f11: at the birth, the carrier's weight change against a matched comparison unit's, 67 held-out pairs.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.F11_DATA = {...}. Prints `path sha256` for every input.

Per pair: e of the carrier and of its comparison unit (input rows and output column) from n1_test.jsonl. The mean
paired difference, its percentile bootstrap band (m2.paired: 10,000 resamples, seed 0) and the counts above and below
zero are recomputed and compared with verdict_n1.json; the half that set the rules with thresholds.json; the natural
units (mean log r over the rise window, scale.py's bootstrap) with read_n1/scale.json.

    python3 extract.py
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []

N1 = "studies/next/n1"
ROWS = {"held_out": f"{N1}/banks/weights-03/n1/n1_test.jsonl", "rule_setting": f"{N1}/banks/weights-03/n1/n1_freeze.jsonl"}
VERDICT = f"{N1}/verdict_n1.json"
THRESH = f"{N1}/thresholds.json"
SCALE = f"{N1}/read_n1/scale.json"
RECOUNT = f"{N1}/read_n1/verdict_recount.json"
FRAGILITY = f"{N1}/read_n1/fragility.json"
BIRTHS = "studies/next/m2/births.json"
N_BOOT, SEED = 10_000, 0                     # m2.N_BOOT, m2.SEED; scale.py boot() uses the same


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
    data["meta"] = {"commit": commit, "link": "studies/next/n1/banks/weights-03/n1/n1_test.jsonl",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


def boot_means(d):
    """The percentile bootstrap of m2.paired and scale.py boot(): N_BOOT resampled means, seed SEED."""
    rng = np.random.default_rng(SEED)
    return np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(N_BOOT)])


def paired(x, y, banked, where, label):
    """m2.paired recomputed (mean of x - y, band, counts above and below zero) beside the banked dict."""
    d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
    lo, hi = np.quantile(boot_means(d), [0.025, 0.975])
    return {"pairs": compare(f"{label} pairs", int(d.size), banked["n"], f"{where}.n", 0),
            "mean_difference": compare(f"{label} mean difference", float(d.mean()), banked["diff"], f"{where}.diff", 1e-12),
            "band_low": compare(f"{label} band low", float(lo), banked["lo"], f"{where}.lo", 1e-12),
            "band_high": compare(f"{label} band high", float(hi), banked["hi"], f"{where}.hi", 1e-12),
            "carrier_above": compare(f"{label} carrier above", int((d > 0).sum()), banked["x_only"], f"{where}.x_only", 0),
            "carrier_below": compare(f"{label} carrier below", int((d < 0).sum()), banked["y_only"], f"{where}.y_only", 0)}


def matched(rows):
    R = {r["birth"]: r for r in rows if r["kind"] == "real"}
    T = {r["birth"]: r for r in rows if r["kind"] == "dt"}
    ids = sorted(b for b in R if b in T)
    return R, T, ids


def main():
    names = {b["id"]: b["name"] for b in json.loads(src(BIRTHS).read_text())["births"]}
    rows = {h: [json.loads(l) for l in src(p).read_text().splitlines() if l.strip()] for h, p in ROWS.items()}
    v = json.loads(src(VERDICT).read_text())["test"]
    th = json.loads(src(THRESH).read_text())["discovery"]
    sc = json.loads(src(SCALE).read_text())
    rc = json.loads(src(RECOUNT).read_text())["confounds"]["b_test"]["maxabs_pre_log"]
    fr = json.loads(src(FRAGILITY).read_text())["test"]["adjusted_for_dlogmaxabs_in"]

    R, T, ids = matched(rows["held_out"])
    n_real = len(R)
    pairs = [{"birth": b, "group_name": names[b], "layer": R[b]["layer"], "birth_snapshot": R[b]["k"],
              "carrier_side": R[b]["pole"], "comparison_side": T[b]["pole"],
              "carrier_e_input_rows": R[b]["in"]["e_window"], "comparison_e_input_rows": T[b]["in"]["e_window"],
              "difference_input_rows": R[b]["in"]["e_window"] - T[b]["in"]["e_window"],
              "carrier_e_output_column": R[b]["out"]["e_window"], "comparison_e_output_column": T[b]["out"]["e_window"]}
             for b in ids]
    w = f"{VERDICT} -> test.pair_in_real_dt"
    inp = paired([R[b]["in"]["e_window"] for b in ids], [T[b]["in"]["e_window"] for b in ids], v["pair_in_real_dt"], w,
                 "held-out input rows")
    out = paired([R[b]["out"]["e_window"] for b in ids], [T[b]["out"]["e_window"] for b in ids], v["pair_out_real_dt"],
                 f"{VERDICT} -> test.pair_out_real_dt", "held-out output column")
    Rf, Tf, idf = matched(rows["rule_setting"])
    rule = paired([Rf[b]["in"]["e_window"] for b in idf], [Tf[b]["in"]["e_window"] for b in idf], th["pair_in_real_dt"],
                  f"{THRESH} -> discovery.pair_in_real_dt", "rule-setting input rows")

    def wlog(r):
        k = r["k"]
        return float(np.mean(np.log([r["in"]["r"][j - 1] for j in (k - 2, k - 1, k)])))
    dl = np.array([wlog(R[b]) - wlog(T[b]) for b in ids])
    lo, hi = np.quantile(boot_means(dl), [0.025, 0.975])
    s = sc["test_pair_mean_log_r_window_in"]
    ws = f"{SCALE} -> test_pair_mean_log_r_window_in"
    natural = {"mean_log_r_difference": compare("natural units: mean log r difference", float(dl.mean()), s["mean"],
                                                f"{ws}.mean", 1e-12),
               "relative_change": compare("natural units: relative change", float(np.exp(dl.mean()) - 1),
                                          s["as_relative"][0], f"{ws}.as_relative[0]", 1e-12),
               "relative_band_low": compare("natural units: band low", float(np.exp(lo) - 1), s["as_relative"][1],
                                            f"{ws}.as_relative[1]", 1e-12),
               "relative_band_high": compare("natural units: band high", float(np.exp(hi) - 1), s["as_relative"][2],
                                             f"{ws}.as_relative[2]", 1e-12)}
    activation = {"log_ratio_of_max_activation": rc["paired_mean_diff"], "log_ratio_band": rc["paired_pct_seed0"],
                  "ratio_of_max_activation": float(np.exp(rc["paired_mean_diff"])),
                  "banked_at": f"{RECOUNT} -> confounds.b_test.maxabs_pre_log",
                  "adjusted_for_activation_size": {"mean_difference": fr["intercept"], "band": fr["intercept_pct"],
                                                   "banked_at": f"{FRAGILITY} -> test.adjusted_for_dlogmaxabs_in"}}

    data = {"pairs": pairs, "held_out_births": n_real, "held_out_input_rows": inp, "held_out_output_column": out,
            "rule_setting_input_rows": rule, "natural_units_input_rows": natural, "activation_size": activation}
    for num, key, val in [("67", "held_out_input_rows.pairs", inp["pairs"]["recomputed"]),
                          ("-0.432", "held_out_input_rows.mean_difference", inp["mean_difference"]["recomputed"]),
                          ("-0.817", "held_out_input_rows.band_low", inp["band_low"]["recomputed"]),
                          ("-0.059", "held_out_input_rows.band_high", inp["band_high"]["recomputed"]),
                          ("23", "held_out_input_rows.carrier_above", inp["carrier_above"]["recomputed"]),
                          ("44", "held_out_input_rows.carrier_below", inp["carrier_below"]["recomputed"]),
                          ("-0.258", "held_out_output_column.mean_difference", out["mean_difference"]["recomputed"]),
                          ("-0.654", "held_out_output_column.band_low", out["band_low"]["recomputed"]),
                          ("0.130", "held_out_output_column.band_high", out["band_high"]["recomputed"]),
                          ("-0.217", "rule_setting_input_rows.mean_difference", rule["mean_difference"]["recomputed"]),
                          ("-0.638", "rule_setting_input_rows.band_low", rule["band_low"]["recomputed"]),
                          ("0.204", "rule_setting_input_rows.band_high", rule["band_high"]["recomputed"]),
                          ("-1.6 %", "natural_units_input_rows.relative_change", natural["relative_change"]["recomputed"]),
                          ("-3.2 %", "natural_units_input_rows.relative_band_low",
                           natural["relative_band_low"]["recomputed"]),
                          ("+0.03 %", "natural_units_input_rows.relative_band_high",
                           natural["relative_band_high"]["recomputed"]),
                          ("1.39", "activation_size.ratio_of_max_activation", activation["ratio_of_max_activation"])]:
        print(f"PLAN {num} -> {key} ({val})")
    notes = {
        "pairs": "The 67 held-out births that have a comparison unit (71 births; 4 have none), sorted by birth id, from "
                 "n1_test.jsonl (bank kinds real and dt).",
        "comparison unit": "Bank kind dt: a random group of another unit's contexts, of the same size and layer, dated to "
                           "the same snapshot by the same rule, and matched on the number of distinct tokens.",
        "carrier_e_input_rows": "in.e_window: e of the gate row and up row (8,192 numbers) = mean z over steps k-2, k-1, "
                                "k minus mean z over steps 2 ... k-3; z = log r standardised within the layer and step.",
        "carrier_e_output_column": "out.e_window: the same for the down-matrix column (4,096 numbers).",
        "difference_input_rows": "carrier e minus comparison e, input rows.",
        "held_out_input_rows": "m2.paired recomputed: mean of the 67 differences, percentile bootstrap band (10,000 "
                               "resamples of pairs, seed 0, 2.5 and 97.5 %), carrier_above = differences > 0 (x_only), "
                               "carrier_below = differences < 0 (y_only); banked pair_in_real_dt.",
        "held_out_output_column": "The same for the output column; banked pair_out_real_dt.",
        "rule_setting_input_rows": "The same on the half that set the rules (n1_freeze.jsonl, 73 pairs); banked "
                                   "thresholds.json discovery.pair_in_real_dt.",
        "natural_units_input_rows": "Mean over pairs of (carrier - comparison) of the mean log r over steps k-2, k-1, k "
                                    "(no reference subtracted), and exp(.) - 1 as a relative change with its bootstrap "
                                    "band (scale.py: 10,000 resamples, seed 0); banked scale.json.",
        "activation_size": "Banked only (not recomputed; the per-pair max activations are in N12's ranks_test.jsonl): "
                           "mean paired log ratio of max |activation| at the snapshot before the birth, carrier over "
                           "comparison, its bootstrap band, and exp of the mean; adjusted_for_activation_size is the "
                           "intercept of the regression of the paired e gap on the log ratio (fragility.json).",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
