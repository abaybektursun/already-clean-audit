"""N6 G-POLE diagnosis (amendment A1): the job's recomputed g and u at each pole's first 10 banked T_15 exemplars
(g_pole.npz, column 0) beside the harvest's banked bf16 g and u at the same positions (the T_15 contexts.jsonl), in bf16
steps (one step = 2^(floor(log2 |bank value|) - 7), the spacing of bf16 at the bank's value).

    PYTHONPATH=. python3 studies/next/n6/diag/g_pole_join.py <g_pole.npz> <out.json>
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import n6  # noqa: E402


def steps(new, bank):
    ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(bank), 1e-30))) - 7)
    return np.abs(new - bank) / ulp


def main():
    z = np.load(sys.argv[1])
    ctx = {}
    for line in open(n6.FILES["ctx15"]):
        c = json.loads(line)
        ctx[c["pole"]] = c["topk"]
    rows = []
    for i, p in enumerate(str(x) for x in z["poles"]):
        tk = ctx[p][:z["tok"].shape[1]]
        assert [k["tok"] for k in tk] == z["tok"][i].tolist(), f"{p}: positions differ from contexts.jsonl"
        assert np.array_equal(np.array([k["a"] for k in tk]), z["a_bank"][i]), f"{p}: a differs from contexts.jsonl"
        gb, ub = np.array([k["g"] for k in tk]), np.array([k["u"] for k in tk])
        g, u, a = z["g"][i, :, 0].astype(np.float64), z["u"][i, :, 0].astype(np.float64), z["a"][i, :, 0].astype(np.float64)
        sg, su = steps(g, gb), steps(u, ub)
        stat = float(np.abs(a - z["a_bank"][i]).max() / np.abs(z["a_bank"][i]).max())
        rows.append({"pole": p, "layer": int(z["layer"][i]), "g_pole_value": stat,
                     "n_g_differ": int((g != gb).sum()), "n_u_differ": int((u != ub).sum()),
                     "max_g_steps": float(sg.max()), "max_u_steps": float(su.max()),
                     "argmax_exemplar": int(np.argmax(np.abs(a - z["a_bank"][i])))})
    v = np.array([r["g_pole_value"] for r in rows])
    gu = np.array([r["n_g_differ"] + r["n_u_differ"] > 0 for r in rows])
    out = {"what": "G-POLE diagnosis: recomputed g, u (job, prefix path) against the harvest's banked bf16 g, u (full windows)",
           "source": {"g_pole.npz": n6.sha256_file(sys.argv[1]), "contexts.jsonl": str(n6.FILES["ctx15"])},
           "n_poles": len(rows), "n_poles_g_or_u_differ": int(gu.sum()),
           "n_poles_value_above_1e-6": int((v > 1e-6).sum()),
           "n_poles_value_above_1e-6_with_g_or_u_differ": int(((v > 1e-6) & gu).sum()),
           "n_poles_value_at_most_1e-6_with_g_or_u_differ": int(((v <= 1e-6) & gu).sum()),
           "max_g_steps": max(r["max_g_steps"] for r in rows), "max_u_steps": max(r["max_u_steps"] for r in rows),
           "max_value": float(v.max()), "rows": rows}
    pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in out if k not in ("rows", "source")}))


if __name__ == "__main__":
    main()
