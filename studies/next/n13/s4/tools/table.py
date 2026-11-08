"""One line per banked S4 arm x benchmark from arm_record.json and engine_check.json (no accuracy)."""
import json, pathlib, sys
R = pathlib.Path("/Users/abaybektursun/projects/interp_research-n13/studies/next/n13/s4")
rows = []
for ar in sorted(R.glob("*/*/arm_record.json")):
    d = json.loads(ar.read_text())
    ec = json.loads((ar.parent / "engine_check.json").read_text())
    b = d.get("build", {})
    h = b.get("hashes_vs_R1_stage6") or {}
    for bn, x in d["bench"].items():
        gc = x["G-COMPLETE_recount"] or {}
        e = b.get("intended_mlp_energy")
        er = (b["families"]["mlp"]["realized_sq"] / e) if e else None
        rows.append([d["prefix"], d["step"], bn, x["status"],
                     f"{h.get('n_equal')}/{h.get('n_tensors')}" if h else "-", b.get("G-MASK"),
                     None if b.get("kept_sd_units") is None else round(b["kept_sd_units"], 2), b.get("kept_pass"),
                     None if er is None else round(er, 4),
                     gc.get("pass"), gc.get("n_empty"), x["G-COMPLETE_failing_input_failed"], x["n_records"], x["cap_hits"],
                     round(x["mean_new_tokens"], 1), x["screen_fired"], x["wall_min"], d["projection_min"], round(b.get("build_seconds") or 0),
                     "equal" if ec["equal"] else "DIFF"])
hdr = ["prefix", "step", "bench", "status", "hash=R1", "G-MASK", "kept_sd", "kept_ok", "E/E_int", "G-COMPL", "n_empty", "FI_fails",
       "n", "cap", "mean_tok", "screen", "wall_min", "proj_min", "build_s", "engine"]
print("\t".join(hdr))
for r in rows:
    print("\t".join(str(v) for v in r))
