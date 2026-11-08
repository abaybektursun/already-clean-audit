"""b (SPEC_D1 line 506): the fraction of poles with m >= 2 whose main cluster is earliest-born
(ties at the earliest date count as earliest, the complement of 'late-born' at line 292).
Interval: percentile bootstrap over poles, 10,000 resamples. The spec's own permutation band
(line 507: birth ranks permuted within each pole, 10,000 permutations) is computed too.
Outputs: out/04_b.json, out/04_per_pole.csv.
"""
import json, os
import numpy as np
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
mine = [json.loads(l) for l in open(C.BIG + "/births.jsonl")]
poles = {}
for l in mine:
    if l["sample"] == 0:
        poles.setdefault(l["pole"], []).append(l)
P = [p for p in sorted(poles) if len(poles[p]) >= 2]
INF = 99
rows, ind, ind_strict, kmin_over_m = ["pole,m,main_birth,min_birth,n_at_min,main_earliest"], [], [], []
for p in P:
    cl = poles[p]
    b = np.array([INF if c["birth_pct"] is None else c["birth_pct"] for c in cl])
    mb = [c["birth_pct"] for c in cl if c["is_main"]]
    assert len(mb) == 1
    mb = INF if mb[0] is None else mb[0]
    ind.append(mb == b.min())
    ind_strict.append(mb == b.min() and (b == b.min()).sum() == 1)
    kmin_over_m.append((b == b.min()).sum() / len(b))
    rows.append(f"{p},{len(b)},{mb},{b.min()},{(b == b.min()).sum()},{int(mb == b.min())}")
ind = np.array(ind, float); ind_strict = np.array(ind_strict, float); kmin_over_m = np.array(kmin_over_m)
rng = np.random.default_rng(1105)
NB = 10000
boot = np.array([ind[rng.integers(0, len(ind), len(ind))].mean() for _ in range(NB)])
boot_s = np.array([ind_strict[rng.integers(0, len(ind), len(ind))].mean() for _ in range(NB)])
# permutation band: each pole's main receives a random one of the pole's own birth dates
births = [np.array([INF if c["birth_pct"] is None else c["birth_pct"] for c in poles[p]]) for p in P]
perm = np.zeros(NB)
for p_i, b in enumerate(births):
    idx = rng.integers(0, len(b), NB)
    perm += (b[idx] == b.min())
perm /= len(P)
res = {"n_poles_m_ge_2": len(P), "b": float(ind.mean()), "k": int(ind.sum()),
       "b_ci95_bootstrap_poles": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
       "b_strict_no_ties": float(ind_strict.mean()),
       "b_strict_ci95": [float(np.percentile(boot_s, 2.5)), float(np.percentile(boot_s, 97.5))],
       "n_poles_main_tied_at_earliest": int(ind.sum() - ind_strict.sum()),
       "permutation_band": {"mean_exact": float(kmin_over_m.mean()), "mean_sim": float(perm.mean()),
                            "p97_5": float(np.percentile(perm, 97.5)),
                            "p_one_sided": float((1 + (perm >= ind.mean()).sum()) / (1 + NB))},
       "n_boot": NB, "seed": 1105, "interval": "percentile bootstrap over poles"}
json.dump(res, open(OUT + "/04_b.json", "w"), indent=1)
open(OUT + "/04_per_pole.csv", "w").write("\n".join(rows) + "\n")
print(json.dumps(res, indent=1))

# sensitivity: b when the even-size median is the lower or the upper middle value (02_births.py fields)
sens = {}
for fld in ["birth_pct_lower", "birth_pct_upper"]:
    ii = []
    for p in P:
        b = np.array([INF if c[fld] is None else c[fld] for c in poles[p]])
        mb = [INF if c[fld] is None else c[fld] for c in poles[p] if c["is_main"]][0]
        ii.append(mb == b.min())
    sens[fld] = float(np.mean(ii))
res["b_median_convention_sensitivity"] = sens
json.dump(res, open(OUT + "/04_b.json", "w"), indent=1)
print(sens)
