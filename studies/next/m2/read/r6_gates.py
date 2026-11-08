"""Reader of M2, phase 1, step 6: every gate and report in the job's metrics.json, both sides.

Input: banks/score-06/m2/metrics.json (sha256 equal to the S3 copy). Output: read/r6_gates.json.
"""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
M2 = ROOT / "studies/next/m2"
F = M2 / "banks/score-06/m2/metrics.json"


def main():
    m = json.loads(F.read_text())
    G = m["gates"]
    rows, bad = [], []
    for name, g in G.items():
        fi = g.get("failing_input")
        row = {"gate": name, "pass": g["pass"], "failing_input_pass": None if fi is None else fi["pass"]}
        for f in ("p99", "median", "max", "n", "n_gated", "exact_share", "max_ratio", "carrier_top10_flips_B"):
            if f in g:
                row[f] = g[f]
        if "tol" in g:
            row["tol_p99"] = g["tol"]["p99"]
        if fi is not None:
            for f in ("p99", "max_ratio", "min_ratio", "n_gated"):
                if f in fi:
                    row[f"fail_{f}"] = fi[f]
        if g["pass"] is not True or (fi is not None and fi["pass"] is not False):
            bad.append(name)
        if fi is None:
            bad.append(name + " (no failing input)")
        if "p99" in g and "tol" in g and not (g["p99"] <= g["tol"]["p99"]):
            bad.append(name + " (p99 above tol but pass)")
        if fi is not None and "p99" in fi and "tol" in fi and not (fi["p99"] > fi["tol"]["p99"]):
            bad.append(name + " (failing p99 not above tol)")
        if name.endswith("_G0"):
            if not (g["max_ratio"] <= 0.01 and fi["min_ratio"] > 0.01):
                bad.append(name + " (G0 sides)")
        rows.append(row)
    rep = []
    for name, r in m["reports"].items():
        x = {"report": name}
        for f in ("p99", "max", "n", "n_gated", "n_above_tol", "max_ratio", "n_layers_over_tol", "under_tol"):
            if f in r:
                x[f] = r[f]
        if "tol" in r:
            x["tol_p99"] = r["tol"]["p99"]
            x["above_tol"] = r["p99"] > r["tol"]["p99"]
        if "failing_input" in r:
            x["fail_p99"] = r["failing_input"]["p99"]
            x["fail_pass"] = r["failing_input"]["pass"]
        rep.append(x)
    pts = {t: {"degenerate": p["degenerate_poles"], "records": p["records"], "windows": p["windows"]}
           for t, p in m["points"].items()}
    out = {"sha256": hashlib.sha256(F.read_bytes()).hexdigest(), "complete": m["complete"], "parked": m["parked"],
           "n_gates": len(rows), "problems": bad, "gates": rows, "reports": rep, "points": pts,
           "cross_device_gex_above_tol": [x["report"] for x in rep if x.get("above_tol")],
           "records_total": {h: sum(p["records"][h] for p in m["points"].values()) for h in ("freeze", "test")}}
    (M2 / "read/r6_gates.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("gates",)}, indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
