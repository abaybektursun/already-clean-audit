"""N6 offline gate (M2 amendment A5's method): both sides of every gate whose quantity the banks hold, at the one point N6 edits
(T_15), before any launch. Reads the laptop banks, three small S3 files (two T_15 controls.json and the T_14 one, the T_15 manifest)
and M2's banked scan of T_15's layer files (studies/next/m2/offline_gate/bank_scan.json). No model, no download.

    PYTHONPATH=. python3 studies/next/n6/offline_gate/gate_sides.py --out studies/next/n6/offline_gate/gate_sides.json
"""
import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np  # noqa: E402

from infra.aws_pg.jobs import n6_edit as J  # noqa: E402
from infra.pg import n6  # noqa: E402

B = "s3://pg-lineage-769063704696/results/neurons"
AWS = ["--profile", "fuelos", "--region", "us-east-1"]


def s3json(key):
    return json.loads(subprocess.run(["aws", "s3", "cp", f"{B}/{key}", "-", *AWS], capture_output=True, text=True, check=True).stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    pairs = json.loads(pathlib.Path("studies/next/n6/pairs.json").read_text())
    # ---- G-BAL (bank side): the matched set and the unmatched failing input
    b, u = pairs["balance"], pairs["failing_input_unmatched"]["balance"]
    out["G_BAL"] = {"pass_side": {k: {m: b[k][m] for m in ("rho_pair", "smd", "pooled_within_layer")} for k in ("ln_max_abs", "ln_density")},
                    "passes": n6.balanced(b),
                    "failing_input_unmatched": {k: {m: u[k][m] for m in ("rho_pair", "smd")} for k in ("ln_max_abs", "ln_density")},
                    "failing_input_fails": not n6.balanced(u), "bounds": {"rho": n6.RHO_MAX, "smd": n6.SMD_MAX},
                    "unmatched_600_within_layer": b["unmatched_all_600"], "n_pairs": pairs["n_pairs"]}
    # ---- K1: E1's banked value, its arithmetic, its failing input
    e1 = json.loads(pathlib.Path("studies/next/n6/inputs/e1_known.json").read_text())
    ab, aa = np.array(e1["a_before"]), np.array(e1["a_after"])
    jf = float(((ab - aa) / ab)[ab > 0].mean())
    met = json.loads(pathlib.Path("studies/instruments/results/e1/1/t15/metrics.json").read_text())
    out["K1"] = {"banked_job_fall": e1["job_fall"], "recomputed_from_banked_contexts": jf, "equal": jf == e1["job_fall"],
                 "failing_input_cross": e1["failing_input_cross_job_fall"],
                 "failing_distance_over_tol": abs(e1["failing_input_cross_job_fall"] - e1["job_fall"]) / J.K1_TOL,
                 "e1_run": {k: met[k] for k in ("device_name", "torch", "transformers", "sha", "dtype", "pg_git_sha")}}
    # ---- K2: two independent T_15 harvests and the T_14 failing input
    w3 = s3json("w3/grid/1/out/t15_step1413814/controls.json")["nll_corpus_first_120k"]
    w1 = s3json("w1/o7_t15/1/out/o7_t15/controls.json")["nll_corpus_first_120k"]
    t14 = s3json("w3/grid/1b/out/t14_step1000000/controls.json")["nll_corpus_first_120k"]
    out["K2"] = {"w3_grid_T15": w3, "w1_50M_T15": w1, "pass_side_two_runs_equal": w3["mean_nll"] == w1["mean_nll"] == J.K2_TARGET,
                 "failing_input_T14": t14["mean_nll"], "failing_distance_over_tol": abs(t14["mean_nll"] - J.K2_TARGET) / J.K2_TOL}
    # ---- G-REV and G-SAME from the banked manifests
    man = s3json("w3/grid/1/out/t15_step1413814/manifest.json")
    out["G_REV"] = {"banked_revision": man["revision"], "registered": n6.SNAPSHOT, "pass_side": man["revision"] == n6.SNAPSHOT,
                    "failing_input_zeros_fails": "0" * 40 != n6.SNAPSHOT}
    out["G_SAME"] = {"registered": J.VERSIONS, "harvest_T15": {k: man[k] for k in ("torch", "transformers", "device_name")},
                     "e1_run": {k: met[k] for k in ("torch", "transformers", "device_name")},
                     "pass_side": man["torch"] == met["torch"] == J.VERSIONS["torch"] and
                     man["transformers"] == met["transformers"] == J.VERSIONS["transformers"] and
                     man["device_name"] == met["device_name"] == J.VERSIONS["device"]}
    # ---- G-POLE: the bank's two records of the T_15 exemplars agree; the T_14 values at the same positions fail
    z = np.load(n6.FILES["arrays"])
    row = {str(p): i for i, p in enumerate(z["poles"])}
    ctx = {json.loads(x)["pole"]: json.loads(x) for x in open(n6.FILES["ctx15"])}
    pass_e, fail_e = [], []
    for p in pairs["poles"]:
        i = row[p]
        a15 = z["a"][15, i, :J.POLE_EX].astype(np.float64)
        ctxa = np.array([k["a"] for k in ctx[p]["topk"][:J.POLE_EX]])
        sc = np.abs(a15).max()
        pass_e.append(float(np.abs(a15 - ctxa).max() / sc))
        fail_e.append(float(np.abs(z["a"][14, i, :J.POLE_EX] - a15).max() / sc))
    out["G_POLE"] = {"n_poles": len(pass_e), "pass_side_max": max(pass_e), "failing_input_T14_min": min(fail_e),
                     "failing_input_T14_median": float(np.median(fail_e)), "tol": J.POLE_TOL,
                     "failing_fails_for_every_pole": min(fail_e) > J.POLE_TOL,
                     "n_poles_failing_input_at_or_below_tol": int(sum(f <= J.POLE_TOL for f in fail_e))}
    # ---- degenerate poles at T_15 (M2's scan of the 32 layer files): every pole has tokens of its sign
    scan = json.loads(pathlib.Path("studies/next/m2/offline_gate/bank_scan.json").read_text())
    p15 = scan["points"]["15"]
    out["degenerate_T15"] = {"n_degenerate_all_layers": p15["n_degenerate_all_layers"], "min_other_poles_scored_layer": p15["min_other_poles_scored_layer"],
                             "scan_control_T15": scan["control"]["15"], "source": "studies/next/m2/offline_gate/bank_scan.json"}
    # ---- the eval set: band 3, outside the grid harvest's 10,044 windows
    w = pairs["eval_windows"]
    out["eval_windows"] = {"n": len(w), "first": w[0], "last": w[-1], "outside_grid_windows": min(w) >= man["n_windows"],
                           "grid_n_windows": man["n_windows"], "in_band3": w[0] * 2048 >= n6.BAND3_TOK[0] and (w[-1] + 1) * 2048 <= n6.BAND3_TOK[1]}
    out["not_in_the_banks"] = ["K3 (no fp32 forward is banked)", "G-FP32", "G-ID, G-ZN, G-CACHE (bitwise identities of this job's forward)",
                               "G-EPS both sides (no forward with a rounding probe is banked; N12's bf16 window-sum noise is the evidence for "
                               "the failing side)", "G-NORM, G-INST (E+ on the eval windows and the column norms are not banked)",
                               "G-PC, G-DOSE (no edit's loss is banked)", "G-POLE's neighbour-neuron failing input (neuron n + 1 is not a grid pole)",
                               "the pass side of K1, K2 and G-POLE on this job's forward (same device and versions as the banked runs)"]
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v)[:300])
    ok = (out["G_BAL"]["passes"] and out["G_BAL"]["failing_input_fails"] and out["K1"]["equal"] and out["K1"]["failing_distance_over_tol"] > 10
          and out["K2"]["pass_side_two_runs_equal"] and out["K2"]["failing_distance_over_tol"] > 10 and out["G_REV"]["pass_side"]
          and out["G_SAME"]["pass_side"] and out["G_POLE"]["pass_side_max"] == 0.0 and out["G_POLE"]["failing_fails_for_every_pole"]
          and out["degenerate_T15"]["n_degenerate_all_layers"] == 0 and out["eval_windows"]["outside_grid_windows"]
          and out["eval_windows"]["in_band3"])
    print("GATE_SIDES_OK" if ok else "GATE_SIDES_FAILED")
    assert ok


if __name__ == "__main__":
    main()
