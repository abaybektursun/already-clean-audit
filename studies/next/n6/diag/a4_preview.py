"""N6 amendment A4: the two rules previewed on L1-02's banks (spec section 14 A4). Not a gate; the registered gates read L2's freeze
half at freeze 2.

  G-RES at the factor 50: F_eps <= min(1e-6, 0.02 x q10), q10 = np.percentile(|dL_col|, 10) over the ten L1 column edits.
  G-POWER: tau_s <= 0.5 x median |D_col| over the pairs, tau_s = the median over the pairs of n6.boot_sd_win(D_col per window,
  SEED + i), the formulas of n6.freeze, on the five L1 pairs at 96 windows; and the same ratio with tau_s scaled by sqrt(96 / 1,024)
  (independent windows, |D_col| held at its 96-window value), the expectation for L2's eval set.

    PYTHONPATH=. python3 studies/next/n6/diag/a4_preview.py
"""
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import n6  # noqa: E402

BANK = pathlib.Path("studies/next/n6/banks/L1-02")
OUT = pathlib.Path("studies/next/n6/diag/L1-02/a4_preview.json")


def main():
    m = json.loads((BANK / "metrics.json").read_text())
    rec = n6.index([json.loads(x) for x in open(BANK / "edits.jsonl") if x.strip()])
    cols = [r for r in rec.values() if r["kind"] == "col"]
    q10 = float(np.percentile([abs(r["dL"]) for r in cols], 10))
    F = n6.gate_floor(m["gates"])["F_eps"]
    bound = min(n6.EPS_ABS, n6.EPS_REL * q10)
    pids = sorted({r["pair"] for r in cols})
    D, sds = [], []
    for i, pid in enumerate(pids):
        e, l_ = rec[f"{pid}_early_col"], rec[f"{pid}_late_col"]
        D.append(e["dL"] - l_["dL"])
        sds.append(n6.boot_sd_win(np.array(e["dL_win"]) - np.array(l_["dL_win"]), n6.SEED + i))
    tau_s, med = float(np.median(sds)), float(np.median(np.abs(D)))
    scale = math.sqrt(n6.W_EVAL / n6.W_EVAL_L2)
    out = {"what": "N6 amendment A4 previewed on L1-02 (not the gates; they read L2's freeze half at freeze 2)",
           "source": {"metrics.json": n6.sha256_file(BANK / "metrics.json"), "edits.jsonl": n6.sha256_file(BANK / "edits.jsonl")},
           "G_RES_factor_50": {"rule": "F_eps <= min(1e-6, 0.02 x q10)", "q10": q10, "bound": bound, "F_eps": F,
                               "F_eps_over_bound": F / bound, "q10_over_F_eps": q10 / F, "pass": bool(F <= bound),
                               "planted_failing_floor_0.05_q10": {"F_eps": 0.05 * q10, "pass": bool(0.05 * q10 <= bound)}},
           "G_POWER": {"rule": "tau_s <= 0.5 x median |D_col| over the pairs (n6.freeze)", "w_eval": n6.W_EVAL,
                       "pairs_stage1": pids, "D_col": D, "boot_sd_D_col": sds, "tau_s": tau_s, "median_abs_D_col": med,
                       "ratio": tau_s / med, "pass_at_96_windows": bool(tau_s <= n6.POWER_MAX * med),
                       "at_1024_windows_if_independent": {"sd_scale": scale, "tau_s": tau_s * scale, "ratio": tau_s * scale / med,
                                                          "pass": bool(tau_s * scale <= n6.POWER_MAX * med),
                                                          "assumes": "windows independent; |D_col| equal to its 96-window value"}}}
    OUT.write_text(json.dumps(out, indent=1))
    g, p = out["G_RES_factor_50"], out["G_POWER"]
    print(json.dumps({"G_RES_50": {k: g[k] for k in ("bound", "F_eps", "F_eps_over_bound", "pass")},
                      "G_POWER": {k: p[k] for k in ("tau_s", "median_abs_D_col", "ratio", "pass_at_96_windows")},
                      "at_1024": p["at_1024_windows_if_independent"]}))


if __name__ == "__main__":
    main()
