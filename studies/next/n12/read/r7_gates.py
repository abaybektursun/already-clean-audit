"""Every gate of the job's metrics.json: pass, and whether its registered failing input failed; the G-EX reports at
T_7-T_14 against the cross-device tolerance; I-GRAD; G-ABL-ID per point."""
import json, pathlib

here = pathlib.Path(__file__).resolve().parent
m = json.load(open(here.parent / "banks" / "score-02" / "n12" / "metrics.json"))
g = m["gates"]
rows, bad = [], []
for k, v in g.items():
    fi = v.get("failing_input")
    fpass = None if fi is None else fi.get("pass")
    rows.append([k, v.get("pass"), fpass])
    if v.get("pass") is not True or fpass is not False:
        bad.append([k, v.get("pass"), fpass])
gex = {k: {"pass_vs_tol": v["pass"], "p99": v["p99"], "tol_p99": v["tol"]["p99"], "n": v["n"],
           "n_gated": v["n_gated"], "max": v["max"], "exact_share": v["exact_share"],
           "failing_input_pass": v["failing_input"]["pass"]}
       for k, v in m["reports"].items() if k.endswith("G_EX")}
gated_ex = {k: {"pass": v["pass"], "p99": v["p99"], "exact_share": v["exact_share"]} for k, v in g.items()
            if k.endswith("G_EX")}
abl = {k: {"max_abs": v["max_abs"], "n": v["n"], "failing_min_abs": v["failing_input"]["min_abs"]}
       for k, v in g.items() if k.endswith("G_ABL_ID")}
m2v = {k: {"p99": v["p99"], "max": v["max"], "exact_share": v["exact_share"], "n": v["n"],
           "failing_p99": v["failing_input"]["p99"]} for k, v in g.items() if k.endswith("G_M2V")}
topk = {k: {"p99": v["p99"], "tol": v["tol"]["p99"], "failing_p99": v["failing_input"]["p99"]}
        for k, v in g.items() if k.endswith("G_TOPK")}
out = {"n_gates": len(g), "n_pass_and_failing_input_failed": len(rows) - len(bad), "exceptions": bad,
       "complete": m["complete"], "parked": m["parked"], "phase": m["phase"], "n_unit_records": m["n_unit_records"],
       "K0": {k: g["K0"][k] for k in ("pass", "max_abs_dB", "tol_B")} | {
           "activations_p99": g["K0"]["activations"]["p99"], "failing_min_abs_dB": g["K0"]["failing_input"]["min_abs_dB"]},
       "G_M2V": m2v, "G_TOPK": topk, "G_ABL_ID": abl, "G_EX_gated": gated_ex, "G_EX_reported": gex,
       "I_ID": {k: g["I_ID"][k] for k in ("pass", "p99", "max", "exact_share", "carrier_top10_flips_B", "n_units")}
                | {"failing_p99": g["I_ID"]["failing_input"]["p99"]},
       "I_GRAD": g["I_GRAD"], "grad_pass_settings": m["grad_pass_settings"], "measured": m["measured"],
       "elapsed_s": m["elapsed_s"],
       "degenerate_poles": {t: p["n_degenerate_poles"] for t, p in m["points"].items()},
       "grad_max_mem_gb": {t: p.get("grad_max_mem_gb") for t, p in m["points"].items()},
       "abl_rows": {t: p.get("abl_rows") for t, p in m["points"].items()}}
(here / "r7_gates.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("n_gates", "n_pass_and_failing_input_failed", "exceptions", "complete", "parked",
                                      "K0", "I_ID")}, indent=0))
for k, v in gex.items():
    print(k, v)
print(abl)
print({k: (v["p99"], v["exact_share"]) for k, v in m2v.items()})
print(out["degenerate_poles"], out["grad_max_mem_gb"], out["abl_rows"])
