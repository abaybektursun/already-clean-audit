"""score-05 stopped at t06_G0: the gate passed, but its registered failing input (the activation swapped, silu ->
gelu_tanh, `neurons_harvest.WRONG_ACT`) also passed, max ratio below G0_TOL = 0.01, so `Run.gate` asserted "its failing
input passed". This diagnostic puts that value beside (1) the same failing input at every point of the five M2
attempts (banks/score-0N/metrics.json) and (2) the banked grid harvest's own G0 controls at T_5-T_15 (S3
results/neurons/w3/grid/<dir>/controls.json, 3 KB each), which also hold a second failing input (u scaled by 1.1) whose
power does not depend on |g|. Reads metrics.json files and the 11 controls.json files only; no rank, example or vector
file of any attempt.

    PYTHONPATH=. python studies/next/m2/diag/score05_g0_fail.py studies/next/m2/diag/score05_g0_fail.json
"""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import m2  # noqa: E402
from infra.pg.neurons_harvest import G0_TOL  # noqa: E402

BANKS = pathlib.Path("studies/next/m2/banks")
AWS = ["--profile", "fuelos", "--region", "us-east-1"]


def main():
    out = pathlib.Path(sys.argv[1])
    jobs = {}
    for a in sorted(BANKS.glob("score-0*")):
        g = json.loads((a / "metrics.json").read_text())["gates"]
        jobs[a.name] = {k[1:3]: {"G0_max": v["max_ratio"], "fail_act_max": v["failing_input"]["max_ratio"],
                                 "fail_act_pass": v["failing_input"]["pass"]} for k, v in g.items() if k.endswith("_G0")}
    banked = {}
    with tempfile.TemporaryDirectory() as d:
        for t in range(5, 16):
            key = f"results/neurons/w3/grid/{m2.POINTS[t][1]}/controls.json"
            f = pathlib.Path(d) / f"t{t:02d}.json"
            subprocess.run(["aws", "s3", "cp", f"s3://pg-lineage-769063704696/{key}", str(f), *AWS, "--only-show-errors"],
                           check=True)
            c = json.loads(f.read_text())["G0"]
            banked[f"{t:02d}"] = {"s3": key, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "tol": c["tol"],
                                  "n_windows": c["n_windows"], "G0_max": max(c["ratio"]),
                                  "fail_act_max": max(c["failing_input"]["ratio"]),
                                  "fail_act_pass": c["failing_input"]["pass"],
                                  "fail_act_layers_over_tol": c["failing_input"]["n_layers_over_tol"],
                                  "fail_u1.1_max": max(c["failing_input_2"]["ratio"]),
                                  "fail_u1.1_min": min(c["failing_input_2"]["ratio"]),
                                  "fail_u1.1_pass": c["failing_input_2"]["pass"]}
    res = {"what": __doc__.split("\n\n")[0], "G0_TOL": G0_TOL, "jobs": jobs, "banked_harvest": banked}
    out.write_text(json.dumps(res, indent=1))
    print(f"G0_TOL {G0_TOL}")
    print("point | " + " | ".join(f"{a} act-swap max" for a in jobs) + " | banked act-swap max (pass) | banked u x1.1 min-max")
    for t in sorted(banked, reverse=True):
        row = [f"{jobs[a][t]['fail_act_max']:.5f}" if t in jobs[a] else "-" for a in jobs]
        b = banked[t]
        print(f"T_{int(t)} | " + " | ".join(row) + f" | {b['fail_act_max']:.5f} ({b['fail_act_pass']}) | "
              f"{b['fail_u1.1_min']:.4f}-{b['fail_u1.1_max']:.4f}")


if __name__ == "__main__":
    main()
