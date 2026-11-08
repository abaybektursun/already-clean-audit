"""G-M2V and K0 rechecked over every unit M2 banked: the job's activations a (arrays/tNN/<unit>_<role>.npz) against
M2's banked vectors (S3 results/next_m2/score-06/out/score/m2/vectors/tNN/), and the job's B at the unit pole against
M2's banked value.B (studies/next/m2/banks/score-06/m2/ranks_{freeze,test}.jsonl). Also the job's M2-format records
(ranks_*.jsonl) against M2's, per (unit, role), on rank.B; and the dt units' part-B record against this reader's r0."""
import argparse, json, pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--arrays", required=True)
ap.add_argument("--m2vec", required=True)
a = ap.parse_args()
here = pathlib.Path(__file__).resolve().parent
n12 = here.parent
root, m2v = pathlib.Path(a.arrays), pathlib.Path(a.m2vec)
units = json.load(open(n12 / "units.json"))["units"]
V = json.load(open(here / "r1_units.json"))["units"]
m2 = {}
for h in ("freeze", "test"):
    for line in open(n12.parent / "m2" / "banks" / "score-06" / "m2" / f"ranks_{h}.jsonl"):
        r = json.loads(line)
        m2[(r["unit"], r["role"])] = r
job = {}
for h in ("freeze", "test"):
    for line in open(n12 / "banks" / "score-02" / "n12" / f"ranks_{h}.jsonl"):
        r = json.loads(line)
        job[(r["unit"], r["role"], r["t"])] = r
out = {"vectors": {"n_files": 0, "n_values": 0, "n_unequal": 0, "max_stat": 0.0, "missing_in_m2": []},
       "B_vs_m2": {"n": 0, "max_abs_dB": 0.0}, "rankB_vs_m2": {"n": 0, "n_unequal": 0, "examples": []},
       "job_ranks_vs_reader": {"n": 0, "n_unequal": 0, "examples": []}}
for u in units:
    unit, kind = u["unit"], u["kind"]
    m2unit = None if kind == "dt" else unit
    roles = [("pre", u["roles"]["pre"]), ("birth", u["roles"]["birth"])] + (
        [("end", 15)] if kind == "dt" else [])
    for role, t in roles:
        z = np.load(root / f"t{t:02d}" / f"{unit.replace(':', '_')}_{role}.npz")
        jr = job.get((unit, role, t))
        if jr is not None:
            n, s = u["carrier"]
            mine = V[unit]["r0"] if role == "pre" else (V[unit]["r1"] if role == "birth" else V[unit]["end_rank"])
            out["job_ranks_vs_reader"]["n"] += 1
            if jr["rank"]["B"] != mine:
                out["job_ranks_vs_reader"]["n_unequal"] += 1
                out["job_ranks_vs_reader"]["examples"].append([unit, role, jr["rank"]["B"], mine])
        if m2unit is None:
            continue
        mr = m2.get((unit, role))  # M2's unit ids are N12's for real and pb units
        fn = f"{unit.replace(':', '_')}_{role}.npz"
        f = m2v / f"t{t:02d}" / fn
        if not f.exists():
            out["vectors"]["missing_in_m2"].append(f"t{t:02d}/{fn}")
            continue
        zm = np.load(f)
        assert list(zm["tok"]) == list(z["tok"]), (unit, role)
        st = np.abs(z["a"].astype(np.float64) - zm["a"]) / (0.05 + np.abs(zm["a"].astype(np.float64)))
        out["vectors"]["n_files"] += 1
        out["vectors"]["n_values"] += int(st.size)
        out["vectors"]["n_unequal"] += int((z["a"] != zm["a"]).sum())
        out["vectors"]["max_stat"] = max(out["vectors"]["max_stat"], float(st.max()))
        if mr is not None:
            n, s = u["carrier"]
            bj = float(z["B"][n, s])
            out["B_vs_m2"]["n"] += 1
            out["B_vs_m2"]["max_abs_dB"] = max(out["B_vs_m2"]["max_abs_dB"], abs(bj - mr["value"]["B"]))
            mine = V[unit]["r0"] if role == "pre" else V[unit]["r1"] if role == "birth" else None
            if mine is not None:
                out["rankB_vs_m2"]["n"] += 1
                if mr["rank"]["B"] != mine:
                    out["rankB_vs_m2"]["n_unequal"] += 1
                    if len(out["rankB_vs_m2"]["examples"]) < 10:
                        out["rankB_vs_m2"]["examples"].append([unit, role, mr["rank"]["B"], mine])
out["m2_record_keys_sample"] = [list(k) for k in list(m2)[:3]]
(here / "r5_m2v_k0.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:3000])
