"""N2V reader, step 1: every gate in the job's metrics.json (S3 copy) with its value, its pass rule and its failing input,
cross-checked against the npz recount (r1_recount.json, r6_checks.json), N12's records and the registered constants.

    python -I r8_gates.py --s3 <dir holding score-02/n2v> --repo <worktree> --recount r1_recount.json --checks r6_checks.json --out r8_gates.json

The registered constants are typed here from the worktree's code (infra/pg/m2.py POINTS, infra/aws_pg/jobs/m2_score.py
VERSIONS and TOKENS_SHA, infra/aws_pg/jobs/n2v_score.py UNITS_SHA and N12_SHA) and compared with what the job wrote.
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np

COMMITS = {5: "ebfe7b62b31bfadff464a8926a48a256c9431db7", 7: "04eeee4fa74f9f3d2cb11c31e155545dad32365e",
           8: "d1dbb0e2d4d3fbdc9c185a34c4e99f13ef6ae2a6", 9: "08b5da69ff99e0858f18d9870ac87f43d7602cf0",
           10: "1d074dbc15d91544abe47d5508ac1670ef8c7129", 11: "c515906cd0f5e30e9e2f56f5f28870c375d8bef0",
           12: "af83fc98693475b503de0a86407f0260a7281786", 13: "db8c654b5cbe06311f92e86d68cd4f7d31642a5a",
           14: "cb286dcdaaf3f92b3c419dd77439078fdebffe53"}
TOKENS_SHA = "2aca80ca6537f46b8f60a550b4a3f2f13be2d18c5402be8f1af5c00caa86f531"
VERSIONS = {"torch": "2.12.0+cu130", "transformers": "5.9.0"}
INPUT_SHA = {"units.json": "c223dfee261c3f6019ad46d08cccc23859143e6f59698aa0ef24de769f95508a",
             "n12_freeze.jsonl": "30f31cb1b96403aa1e0344d86fb607022d77786b7f333f7a5943556bcaadb391",
             "n12_test.jsonl": "877926da34620a121067788b3c4736b986cba7c94fdc5bbc4f98e40975599d12"}
RULE = {"G_SAME": "torch 2.12.0+cu130, transformers 5.9.0, NVIDIA L40S",
        "INPUTS": "units.json and N12's two record files at their registered sha256",
        "G_CORPUS": "tokens sha256 2aca80ca...",
        "G_REV": "the downloaded commit equals the registered one", "G_CFG": "config equal to the reference, no differing key",
        "K_A": "every one of 40 pairs' bf16 dL within 1e-6 nats of N12's banked abl[i][2]",
        "G_FP32": "every parameter fp32, matmul precision 'highest', TF32 off",
        "G_CAUSAL": "every changed row (zero, fd, probe) bitwise equal to its clean row at j < i0",
        "I_FWD": "the T_13 clean rows again, reversed, clean-only batches: bitwise",
        "I_JVP": "the first 16 T_13 unit windows' forward-mode rows of poles 0 and 1 again, reversed: bitwise"}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def read_jsonl(p):
    return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--recount", required=True)
    ap.add_argument("--checks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s3 = pathlib.Path(a.s3) / "score-02" / "n2v"
    repo = pathlib.Path(a.repo)
    jm = json.loads((s3 / "metrics.json").read_text())
    rc = json.loads(pathlib.Path(a.recount).read_text())
    ck = json.loads(pathlib.Path(a.checks).read_text())
    rows = []
    for name, g in jm["gates"].items():
        base = name.split("_", 1)[1] if name[0] == "t" and name[1:3].isdigit() else name
        fi = g.get("failing_input")
        val = {k: v for k, v in g.items() if k not in ("failing_input", "worst10", "pass")}
        rows.append({"gate": name, "pass": g["pass"], "rule": RULE.get(base), "value": val,
                     "failing_input": fi, "failing_input_fails": None if fi is None else (not fi["pass"])})
    out = {"complete": jm["complete"], "parked": jm["parked"], "parts": jm["parts"], "n_gates": len(rows),
           "n_pass": sum(r["pass"] for r in rows), "n_failing_inputs": sum(r["failing_input"] is not None for r in rows),
           "n_failing_inputs_fail": sum(bool(r["failing_input_fails"]) for r in rows), "gates": rows}
    # cross-checks against registered constants and the inputs in the worktree
    xc = {}
    xc["G_SAME_versions_registered"] = jm["gates"]["G_SAME"]["versions"] == VERSIONS and jm["gates"]["G_SAME"]["device"] == "NVIDIA L40S"
    xc["G_CORPUS_sha_registered"] = jm["gates"]["G_CORPUS"]["got"] == TOKENS_SHA
    xc["G_REV_commits_equal_m2_POINTS"] = {t: jm["gates"][f"t{t:02d}_G_REV"]["commit"] == c for t, c in COMMITS.items()}
    xc["inputs_in_worktree_sha"] = {
        "units.json": sha(repo / "studies/next/n12/units.json") == INPUT_SHA["units.json"],
        "n12_freeze.jsonl": sha(repo / "studies/next/n12/banks/score-02/n12/n12_freeze.jsonl") == INPUT_SHA["n12_freeze.jsonl"],
        "n12_test.jsonl": sha(repo / "studies/next/n12/banks/score-02/n12/n12_test.jsonl") == INPUT_SHA["n12_test.jsonl"]}
    # K-A's failing input from N12's banked dL: the poles rolled inside each of the 10 K-A units; K-A's measurement
    # equals the banked dL (n_exact 40), so the failing input's smallest gap is min |dL(pole j) - dL(pole j+1)|
    n12 = {r["unit"]: r for h in ("freeze", "test") for r in read_jsonl(repo / f"studies/next/n12/banks/score-02/n12/n12_{h}.jsonl")}
    gaps = []
    for u in jm["gates"]["K_A"]["units"]:
        d = [x[2] for x in n12[u]["abl"]]
        gaps += [abs(d[j] - d[(j + 1) % 4]) for j in range(4)]
    xc["K_A_failing_input_min_gap_from_N12_records"] = float(min(gaps))
    xc["K_A_failing_input_min_gap_in_metrics"] = jm["gates"]["K_A"]["failing_input"]["min_abs_diff"]
    # K-A's units: the first 10 real units by id whose pre point is T_14
    units = json.loads((repo / "studies/next/n12/units.json").read_text())["units"]
    first10 = sorted(u["unit"] for u in units if u["kind"] == "real" and u["roles"]["pre"] == 14)[:10]
    xc["K_A_units_are_first_10_real_T14"] = first10 == jm["gates"]["K_A"]["units"]
    # G-CAUSAL per point: rows and the failing input's count against the npz
    by_t_rows = {}
    for p in rc["pairs"]:
        if p["rank"] == 0:
            by_t_rows[p["t"]] = by_t_rows.get(p["t"], 0) + 9 * p["n_windows"]
    xc["G_CAUSAL_rows_per_point_npz_vs_metrics"] = {t: [n, jm["gates"][f"t{t:02d}_G_CAUSAL"]["n_rows"]] for t, n in sorted(by_t_rows.items())}
    fz = ck["h_rows_equal_everywhere"]["zero_rows_equal_through_i0_by_point"]
    xc["G_CAUSAL_failing_input_n_equal_npz_vs_metrics"] = {t: [fz[str(t)], jm["gates"][f"t{t:02d}_G_CAUSAL"]["failing_input"]["n_equal"]]
                                                           for t in sorted(by_t_rows)}
    xc["G_CAUSAL_npz_all_points"] = {k: rc["G_CAUSAL_from_npz"][k] for k in ("rows", "bitwise_before_i0", "differ_at_or_after_i0",
                                                                             "differ_at_i0", "equal_everywhere")}
    t13_uw = sum(p["n_windows"] for p in rc["pairs"] if p["t"] == 13 and p["rank"] == 0)
    xc["I_FWD_n_vs_T13_unit_windows"] = [jm["gates"]["I_FWD"]["n"], t13_uw]
    xc["I_FWD_failing_input_from_npz"] = rc["I_FWD_failing_input_from_npz"]
    xc["I_JVP_n"] = [jm["gates"]["I_JVP"]["n"], 2 * 16]
    xc["I_JVP_failing_input_from_npz"] = rc["I_JVP_failing_input_from_npz"]
    xc["identity_clean_same_window_other_unit_npz"] = rc["identity_clean_same_window_other_unit"]
    xc["jvp_tangent_exactly_0_before_i0_npz"] = rc["jvp_tangent_before_i0"]
    xc["slot_control"] = jm["reports"]["slot_control"]
    xc["jvp_max_mem_gb"] = {t: v.get("jvp_max_mem_gb") for t, v in jm["points"].items()}
    out["cross_checks"] = xc
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("complete", "parked", "parts", "n_gates", "n_pass", "n_failing_inputs", "n_failing_inputs_fail")}))
    print(json.dumps(xc, indent=1)[:4000])


if __name__ == "__main__":
    main()
