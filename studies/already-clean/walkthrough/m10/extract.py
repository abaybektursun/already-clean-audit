"""m10: the weight-change statistic, one carrier's relative weight change across the sixteen snapshots.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.M10_DATA = {...}. Prints `path sha256` for every input.

For the 10 held-out births of N1's banked raw examples (the first is the default), the carrier's and its comparison
unit's r_j and z_j for the input rows and the output column over the fifteen steps between the sixteen snapshots
(n1_test.jsonl), the rise window k-2, k-1, k and the reference steps 2 ... k-3, and e recomputed from z against the
banked e. The layer's own median and spread of log r per step are not banked (only z, which is log r measured against
them); the spread summarised over the 32 layers is in read_n1/scale.json and is carried.

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
ROWS = f"{N1}/banks/weights-03/n1/n1_test.jsonl"
EXAMPLES = f"{N1}/banks/weights-03/n1/samples/raw_examples_test.jsonl"
METRICS = f"{N1}/banks/weights-03/n1/metrics.json"
VERDICT = f"{N1}/verdict_n1.json"
SCALE = f"{N1}/read_n1/scale.json"
BIRTHS = "studies/next/m2/births.json"
SIDES = {"in": "input_rows", "out": "output_column"}


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


def e_of(z, k):
    """N1_SPEC §0.5: mean z over steps k-2, k-1, k minus mean z over steps 2 ... k-3 (z[j-1] is step j)."""
    return float(np.mean([z[j - 1] for j in (k - 2, k - 1, k)]) - np.mean([z[j - 1] for j in range(2, k - 2)]))


def unit_block(r, label):
    out = {"side": r["pole"], "unit": r["neuron"]}
    for s, name in SIDES.items():
        assert len(r[s]["r"]) == len(r[s]["z"]) == 15
        out[name] = {"r": r[s]["r"], "z": r[s]["z"],
                     "e": compare(f"{label} {r['unit']} e ({name})", e_of(r[s]["z"], r["k"]), r[s]["e_window"],
                                  f"{ROWS} unit {r['unit']} {s}.e_window", 1e-5)}
    return out


def main():
    rows = {r["unit"]: r for r in map(json.loads, src(ROWS).read_text().splitlines())}
    ex = [json.loads(l) for l in src(EXAMPLES).read_text().splitlines() if l.strip()]
    pts = json.loads(src(METRICS).read_text())["points"]
    steps = [int(pts[str(t)]["download"]["branch"].split("step")[1]) for t in range(16)]
    names = {b["id"]: b["name"] for b in json.loads(src(BIRTHS).read_text())["births"]}
    births = []
    for e in ex:
        c, d = e["real"], e["dt"]
        assert c == rows[c["unit"]] and d == rows[d["unit"]], e["birth"]
        assert c["kind"] == "real" and d["kind"] == "dt" and c["k"] == d["k"] and c["layer"] == d["layer"]
        k = c["k"]
        births.append({"birth": e["birth"], "group_name": names[e["birth"]], "layer": c["layer"], "birth_snapshot": k,
                       "rise_window_steps": [k - 2, k - 1, k], "reference_steps": list(range(2, k - 2)),
                       "carrier": unit_block(c, "carrier"), "comparison_unit": unit_block(d, "comparison")})

    sc = json.loads(src(SCALE).read_text())
    spread = {name: [{"step": int(j), "median_r_over_layers": v["median_r"],
                      "one_z_in_log_r_median_over_layers": v["z_unit_in_log_r_median"],
                      "one_z_in_log_r_min_over_layers": v["min"], "one_z_in_log_r_max_over_layers": v["max"]}
                     for j, v in sorted(sc[f"scale_{s}"].items(), key=lambda x: int(x[0]))] for s, name in SIDES.items()}

    test = [r for r in rows.values()]
    R = {r["birth"]: r for r in test if r["kind"] == "real"}
    T = {r["birth"]: r for r in test if r["kind"] == "dt"}
    ids = sorted(b for b in R if b in T)
    dif = np.array([R[b]["in"]["e_window"] - T[b]["in"]["e_window"] for b in ids])
    hv = json.loads(src(VERDICT).read_text())["test"]["pair_in_real_dt"]
    headline = {"pairs": compare("held-out pairs with a comparison unit", len(ids), hv["n"],
                                 f"{VERDICT} -> test.pair_in_real_dt.n", 0),
                "mean_difference_input_rows": compare("mean e(carrier) - e(comparison), input rows", float(dif.mean()),
                                                      hv["diff"], f"{VERDICT} -> test.pair_in_real_dt.diff", 1e-12),
                "band": [hv["lo"], hv["hi"]], "band_banked_at": f"{VERDICT} -> test.pair_in_real_dt.lo, hi"}

    data = {"snapshot_steps": steps, "step_index": list(range(1, 16)), "default_birth": births[0]["birth"],
            "births": births, "layer_spread_over_32_layers": spread, "headline_held_out": headline}
    b0 = births[0]
    print(f"PLAN 16 snapshots -> snapshot_steps ({len(steps)}: {steps[0]} ... {steps[-1]})")
    print(f"PLAN e of the default carrier -> births[0].carrier.input_rows.e ({b0['carrier']['input_rows']['e']['recomputed']})")
    print(f"PLAN 67 -> headline_held_out.pairs ({len(ids)})")
    print(f"PLAN -0.432 -> headline_held_out.mean_difference_input_rows ({dif.mean()})")
    print(f"PLAN -0.817 to -0.059 -> headline_held_out.band ({hv['lo']}, {hv['hi']})")
    notes = {
        "snapshot_steps": "Training step of each of the sixteen snapshots 0 ... 15 (n1 metrics.json points.*.download "
                          "branch names).",
        "step_index": "Step j = 1 ... 15 is the change from snapshot j-1 to snapshot j (the bank's bracket j); "
                      "r[j-1] and z[j-1] belong to step j.",
        "input_rows": "The unit's gate row and up row as one vector of 8,192 numbers (bank key `in`).",
        "output_column": "The unit's down-matrix column, 4,096 numbers (bank key `out`).",
        "r": "r_j = ||M(j) - M(j-1)|| / ||M(j-1)||, 64-bit from the 16-bit snapshots, as banked.",
        "z": "z_j = (log r_j - the layer's median of log r_j) / (1.4826 x the layer's median absolute deviation), over "
             "the layer's 11,008 units, per step; banked rounded to 5 decimals. z = 0 is the layer's median.",
        "e": "e = mean z over rise_window_steps minus mean z over reference_steps, recomputed from the banked z (rounded "
             "to 5 decimals, so each z is within 5e-6 and e within 1e-5) against the banked e_window.",
        "rise_window_steps": "The three steps k-2, k-1, k, shaded in the picture.",
        "reference_steps": "Steps 2 ... k-3 of the same unit; step 1 (random start to step 1,000) is excluded.",
        "births": "The 10 held-out births of N1's samples/raw_examples_test.jsonl, in file order (selection by the "
                  "bank, not by this script); the records equal the n1_test.jsonl rows (asserted).",
        "carrier / comparison_unit": "Bank kinds real and dt: the comparison unit is a random group of another unit's "
                                     "contexts of the same size and layer, dated to the same snapshot by the same rule, "
                                     "and matched on the number of distinct tokens.",
        "group_name": "The group's name verbatim from m2/births.json.",
        "layer_spread_over_32_layers": "read_n1/scale.json: per step, the median over the 32 layers of each layer's "
                                       "median r, and the size of one z unit in log r (1.4826 x MAD) as the median, "
                                       "min and max over the 32 layers. Not the carrier's own layer.",
        "gap": "The per-layer median and median absolute deviation of log r for each step are not in the repository "
               "(only in stats.npz on S3, results/next_n1/weights-03/out/weights/n1/stats.npz); the unit's own r and z "
               "are banked and carried, and z is r measured against its layer's spread.",
        "headline_held_out": "The mean over the 67 held-out pairs of e(carrier) - e(comparison unit), input rows, "
                             "recomputed from n1_test.jsonl; the band is banked (percentile bootstrap, m2.paired).",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
