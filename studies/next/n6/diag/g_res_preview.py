"""N6: a preview of G-RES (spec section 3.4, n6.freeze) on L1-02's ten column edits, and the cause of its failure.

G-RES passes when F_eps <= min(1e-6, 0.01 x q10), F_eps = the largest |dL| of the fp32 rounding probes, q10 = the 10th percentile
(numpy, linear) of |dL_col| over both members' column edits of the freeze pairs. The registered gate reads L2's freeze half at freeze
2; this is a preview on L1-02's banks, not the gate. Also written: the path check (the probe against the zero edit, which runs the
same path) and the eval-set noise of each L1 pair's contrast (n6.boot_sd_win, the per-pair input of tau_s). The rule previewed is
freeze 1's (factor 100, EPS_REL_FREEZE1); amendment A4 set the factor to 50 after this preview (diag/a4_preview.py).

    PYTHONPATH=. python3 studies/next/n6/diag/g_res_preview.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import n6  # noqa: E402

BANK = pathlib.Path("studies/next/n6/banks/L1-02")
OUT = pathlib.Path("studies/next/n6/diag/L1-02/g_res_preview.json")
EPS_REL_FREEZE1 = 0.01      # the factor this preview read (n6.EPS_REL is 0.02 after amendment A4)


def main():
    m = json.loads((BANK / "metrics.json").read_text())
    rec = n6.index([json.loads(x) for x in open(BANK / "edits.jsonl") if x.strip()])
    cols = [r for r in rec.values() if r["kind"] == "col"]
    dl = np.array([abs(r["dL"]) for r in cols])
    q10 = float(np.percentile(dl, 10))
    F = max(abs(v) for v in m["gates"]["G_EPS"]["fp32"].values())
    bound = min(n6.EPS_ABS, EPS_REL_FREEZE1 * q10)
    zero = rec["P002_late_zero"]
    probe = rec["P002_late_probe"]
    pairs = []
    for pid in sorted({r["pair"] for r in cols}):
        e, l_ = rec[f"{pid}_early_col"], rec[f"{pid}_late_col"]
        dw = np.array(e["dL_win"]) - np.array(l_["dL_win"])
        sd = n6.boot_sd_win(dw, n6.SEED)
        pairs.append({"pair_stage1": pid, "layer": e["layer"], "early": e["pole"], "late": l_["pole"], "D_col": e["dL"] - l_["dL"],
                      "boot_sd_D_col": sd, "abs_D_over_2sd": abs(e["dL"] - l_["dL"]) / (2 * sd),
                      "edits": [{"edit": r["edit"], "dL": r["dL"], "boot_sd": n6.boot_sd_win(np.array(r["dL_win"]), n6.SEED),
                                 "abs_dL_over_sd": abs(r["dL"]) / n6.boot_sd_win(np.array(r["dL_win"]), n6.SEED)} for r in (e, l_)]})
    out = {"what": "N6 G-RES preview on L1-02's ten column edits (spec section 3.4; the registered gate reads L2's freeze half)",
           "source": {"metrics.json": n6.sha256_file(BANK / "metrics.json"), "edits.jsonl": n6.sha256_file(BANK / "edits.jsonl")},
           "rule": "pass when F_eps <= min(1e-6, 0.01 x q10); F_eps = max |dL| of the fp32 probes; q10 = np.percentile(|dL_col|, 10)",
           "abs_dL_col_sorted": sorted(dl.tolist()), "median_abs_dL_col": float(np.median(dl)), "q10": q10, "bound": bound,
           "F_eps": F, "F_eps_by_probe": m["gates"]["G_EPS"]["fp32"], "F_eps_over_bound": F / bound, "q10_over_F_eps": q10 / F,
           "pass": bool(F <= bound),
           "path_check": {"note": "the zero edit and the probe run the same path (the progression, read against base pass 1); the zero "
                                  "edit's dL is the path offset, its largest per-token difference equals G_CACHE's",
                          "zero_dL": zero["dL"], "zero_max_token_abs": max(t["abs"] for t in zero["top5"]),
                          "G_CACHE_max": m["gates"]["G_CACHE"]["value_max_abs"], "zero_dL_over_F_eps": abs(zero["dL"]) / F,
                          "probe_R": probe["R"], "probe_top_token": probe["top5"][0]},
           "eval_set_noise": {"note": "window-bootstrap sd of D_col over the 96 windows (n6.boot_sd_win, seed 0): the per-pair input of "
                                      "tau_s; R counts pairs with |D_col| > 2 tau_s",
                              "pairs": pairs, "median_boot_sd_D_col": float(np.median([p["boot_sd_D_col"] for p in pairs])),
                              "n_resolved_at_2sd": sum(p["abs_D_over_2sd"] > 1 for p in pairs),
                              "n_edits_within_1sd_of_zero": sum(x["abs_dL_over_sd"] <= 1 for p in pairs for x in p["edits"])}}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("q10", "bound", "F_eps", "F_eps_over_bound", "q10_over_F_eps", "pass")}))
    print(json.dumps({k: out["eval_set_noise"][k] for k in ("median_boot_sd_D_col", "n_resolved_at_2sd", "n_edits_within_1sd_of_zero")}))


if __name__ == "__main__":
    main()
