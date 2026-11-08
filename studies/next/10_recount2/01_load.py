"""Load the 16 banked grid checkpoints into one array file and check alignment.

Inputs (read-only): <BANK>/w3/grid/{1,1b}/out/tNN_stepN/poles/exemplar_a.npz and samples/contexts.jsonl.
Outputs: <BIG>/arrays.npz (a [16,640,50], q999 [16,640], max_abs [16,640], poles, tok15 [640,50]),
         out/arrays.pointer.json, out/01_load_checks.json.
Assumption checked here: row i of exemplar_a.npz is line i of the same directory's contexts.jsonl, and
column k is T_15 rank k.
"""
import json, hashlib, os
import numpy as np

BANK = "/Users/abaybektursun/projects/interp_research_banks/neurons"
BIG = BANK + "/next/recount2"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DIRS = ["1/out/t00_step0", "1/out/t01_step1000", "1/out/t02_step2000", "1/out/t03_step3000",
        "1/out/t04_step5000", "1/out/t05_step8000", "1/out/t06_step12000", "1b/out/t07_step20000",
        "1b/out/t08_step35000", "1b/out/t09_step60000", "1b/out/t10_step100000", "1b/out/t11_step180000",
        "1b/out/t12_step320000", "1b/out/t13_step560000", "1b/out/t14_step1000000", "1/out/t15_step1413814"]
STEPS = [0, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 35000, 60000, 100000, 180000, 320000, 560000,
         1000000, 1413814]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


A, Q, M, inputs = [], [], [], []
poles = None
ctx15 = None
checks = {"per_t": []}
for t, d in enumerate(DIRS):
    assert d.endswith("_step%d" % STEPS[t]), d
    p_npz = f"{BANK}/w3/grid/{d}/poles/exemplar_a.npz"
    p_ctx = f"{BANK}/w3/grid/{d}/samples/contexts.jsonl"
    z = np.load(p_npz)
    a = z["a"].astype(np.float64)
    rows = [json.loads(l) for l in open(p_ctx)]
    pl = [r["pole"] for r in rows]
    if poles is None:
        poles = pl
    assert pl == poles, f"pole order differs at t={t}"
    assert a.shape == (len(poles), 50), a.shape
    q = np.array([r["q999"] for r in rows], dtype=np.float64)
    m = np.array([r["max_abs"] for r in rows], dtype=np.float64)
    # alignment check: where a T_15 exemplar position is also in this checkpoint's own top-50, values agree
    A.append(a); Q.append(q); M.append(m)
    inputs.append({"t": t, "exemplar_a": p_npz, "sha256_npz": sha(p_npz), "contexts": p_ctx,
                   "sha256_ctx": sha(p_ctx)})
    checks["per_t"].append({"t": t, "n_nan": int(np.isnan(a).sum()), "q999_min": float(q.min()),
                            "q999_max": float(q.max()), "max_abs_min": float(m.min()),
                            "n_rows": len(rows), "_rows": rows})

A = np.stack(A); Q = np.stack(Q); M = np.stack(M)
rows15 = checks["per_t"][15]["_rows"]
tok15 = np.array([[e["tok"] for e in r["topk"]] for r in rows15], dtype=np.int64)
a15c = np.array([[e["a"] for e in r["topk"]] for r in rows15], dtype=np.float64)
assert tok15.shape == (640, 50)
# check 1: T_15 exemplar a equals the T_15 top-50 a in rank order
d15 = np.abs(A[15] - a15c).max()
checks["T15_exemplar_vs_topk_max_abs_diff"] = float(d15)
assert d15 < 1e-6, d15
# check 2: at every t, a T_15 exemplar token that is also in t's own top-50 has the same a
for t in range(16):
    rows = checks["per_t"][t].pop("_rows")
    n_overlap, maxdiff = 0, 0.0
    for i, r in enumerate(rows):
        own = {e["tok"]: e["a"] for e in r["topk"]}
        for k in range(50):
            if tok15[i, k] in own:
                n_overlap += 1
                maxdiff = max(maxdiff, abs(own[tok15[i, k]] - A[t, i, k]))
    checks["per_t"][t]["n_overlap_with_own_top50"] = n_overlap
    checks["per_t"][t]["overlap_max_abs_diff"] = maxdiff
    assert maxdiff < 1e-5, (t, maxdiff)
# check 3: the sign of each pole's T_15 exemplars matches the pole sign
sgn = np.array([1.0 if p.endswith("+") else -1.0 for p in poles])
checks["T15_wrong_sign_entries"] = int(((A[15] * sgn[:, None]) <= 0).sum())
# check 4: q999 values are bin upper edges 10^(-6 + k/40)
k = np.log10(Q) * 40 + 240
checks["q999_off_bin_edge_max"] = float(np.abs(k - np.round(k)).max())
# check 5: T_15 rank order is by |a| descending
checks["T15_rank_order_violations"] = int((np.diff(np.abs(A[15]), axis=1) > 1e-7).sum())

np.savez(BIG + "/arrays.npz", a=A, q999=Q, max_abs=M, poles=np.array(poles), tok15=tok15,
         steps=np.array(STEPS))
json.dump({"path": BIG + "/arrays.npz", "sha256": sha(BIG + "/arrays.npz"),
           "bytes": os.path.getsize(BIG + "/arrays.npz"), "inputs": inputs,
           "written_by": "studies/next/10_recount2/01_load.py"}, open(OUT + "/arrays.pointer.json", "w"), indent=1)
json.dump(checks, open(OUT + "/01_load_checks.json", "w"), indent=1)
print(json.dumps({k: v for k, v in checks.items() if k != "per_t"}, indent=1))
for c in checks["per_t"]:
    print(c)
