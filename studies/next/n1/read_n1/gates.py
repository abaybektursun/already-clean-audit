"""N1 independent read, step 1: every gate and report in metrics.json, its value, its pass rule and its registered
failing input's value; the revisions against m2.POINTS (the registry); the banked copies against S3 by sha256.

    python3 -I gates.py --metrics S3/metrics.json --repo <worktree> --out studies/next/n1/read_n1/gates.json
"""
import argparse
import hashlib
import json
import pathlib
import sys


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--s3dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from infra.pg import m2                    # the registry of revisions only
    m = json.loads(pathlib.Path(a.metrics).read_text())
    G, Rp = m["gates"], m["reports"]
    out = {"complete": m["complete"], "n_gates": len(G), "elapsed_s": m["elapsed_s"], "n_records": m["n_records"],
           "parked": m["parked"], "rows": []}
    bad = []
    for name, g in G.items():
        fi = g.get("failing_input")
        row = {"gate": name, "pass": g["pass"], "failing_input_pass": None if fi is None else fi["pass"]}
        if name.endswith("_G_REV"):
            t = int(name[1:3])
            row["value"] = g["commit"]
            row["rule"] = "commit == registered revision"
            row["registered_in_m2_POINTS"] = m2.POINTS[t][3]
            row["match_registry"] = g["commit"] == g["registered"] == m2.POINTS[t][3]
            row["branch"] = m["points"][str(t)]["download"]["branch"]
            row["branch_registry"] = m2.POINTS[t][2]
        elif name.endswith("_G_SHAPE"):
            row["value"] = {"bad_layers": g["bad_layers"], "n_layers": g["n_layers"]}
            row["rule"] = "96 tensors, [N,H],[N,H],[H,N], bf16"
        elif name.endswith("_PC"):
            row["value"] = {"in": g["in"], "out": g["out"]}
            row["failing_input_value"] = {"in_size0": g["in_size0"], "out_size0": g["out_size0"]}
            row["rule"] = f"min(in, out) >= {g['min']}"
            row["recheck"] = min(g["in"], g["out"]) >= g["min"] and not (min(g["in_size0"], g["out_size0"]) >= g["min"])
        elif name == "I_ID":
            row["value"] = {"norms_bitwise": g["norms_bitwise"], "self_change_zero": g["self_change_zero"]}
            row["rule"] = "norms bitwise equal on a second read and T_15 - T_15 = 0 for every neuron"
            row["failing_input_value"] = "T_14 vs T_15 all zero: " + str(fi["pass"])
        elif name == "G_SAME":
            row["value"], row["rule"] = g["versions"], f"== {g['registered']}"
        elif name == "INPUTS":
            row["value"], row["rule"] = g["mismatch"], "sha256 of units.json, kb_bank.json, n6 array as registered"
        if not g["pass"] or (fi is not None and fi["pass"]):
            bad.append(name)
        out["rows"].append(row)
    out["gates_all_pass"] = all(g["pass"] for g in G.values())
    out["failing_inputs_all_fail"] = all(not g["failing_input"]["pass"] for g in G.values() if "failing_input" in g)
    out["n_gates_with_failing_input"] = sum("failing_input" in g for g in G.values())
    out["problems"] = bad
    out["rev_all_match_registry"] = all(r.get("match_registry", True) for r in out["rows"])
    kb = {k: {"pass": v["pass"], "n": v["n"], "max_rel": v["max_rel"], "tol": v["tol"],
              "failing_input_max_rel": v["failing_input"]["max_rel"], "failing_input_pass": v["failing_input"]["pass"],
              "max_rel_fp32_cpu": v["max_rel_fp32_cpu"], "n_exact_fp32": v["n_exact_fp32"]}
          for k, v in Rp.items() if k.endswith("_K_B")}
    out["K_B_reported"] = kb
    out["K_B_all_pass"] = all(v["pass"] for v in kb.values())
    out["K_B_failing_inputs_all_fail"] = all(not v["failing_input_pass"] for v in kb.values())
    out["TRI"] = {k: v for k, v in Rp.items() if k.endswith("_TRI")}
    pcs = [r for r in out["rows"] if r["gate"].endswith("_PC")]
    out["PC_summary"] = {"n": len(pcs), "planted_min": min(min(r["value"].values()) for r in pcs),
                         "size0_min": min(min(r["failing_input_value"].values()) for r in pcs),
                         "size0_max": max(max(r["failing_input_value"].values()) for r in pcs)}
    s3, bk = pathlib.Path(a.s3dir), pathlib.Path(a.repo) / "studies/next/n1/banks/weights-03/n1"
    out["sha256"] = {f: {"s3": sha(s3 / f), "banked": sha(bk / f) if (bk / f).exists() else "not banked"}
                     for f in ("metrics.json", "n1_freeze.jsonl", "samples/raw_examples_freeze.jsonl",
                               "n1_test.jsonl", "samples/raw_examples_test.jsonl", "stats.npz")}
    th = json.loads((pathlib.Path(a.repo) / "studies/next/n1/thresholds.json").read_text())
    out["thresholds_sha_match"] = {"n1_freeze": th["sha256_n1_freeze"] == out["sha256"]["n1_freeze.jsonl"]["s3"],
                                   "metrics": th["sha256_job_metrics"] == out["sha256"]["metrics.json"]["s3"]}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows",)}, indent=1))


if __name__ == "__main__":
    main()
