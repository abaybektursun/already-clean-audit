"""Two questions the spec did not anticipate, from r1_units.json.
(a) Does F's sign follow from the dating rule? The rule puts the pole's median below its q999 at T_(k-1) and at or
    above it at T_k, so B_birth > B_pre for every unit; with the others held at T_(k-1), r_c <= r0 follows. Then
    F = log r_o - log r_c >= log r_o - log r0, so F >= 0 whenever the others' move alone does not lift the pole.
    The twins carry the same rule; their F, paired and matched on the starting rank, is the reference.
(b) Is the competitors' smaller drop at real births a token-repetition effect? drop against the distinct-token share
    d / size per kind; real - pb within repeated (< 0.67) and varied births; real - dt (matched on d).
Bootstrap 10,000, seed 7."""
import json, pathlib
import numpy as np
from scipy.stats import spearmanr

here = pathlib.Path(__file__).resolve().parent
V = json.load(open(here / "r1_units.json"))["units"]
SEED = 7


def boot(x, nb=10000):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    rng = np.random.default_rng(SEED)
    m = x[rng.integers(0, x.size, (nb, x.size))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def U(kind, half):
    return {v["birth"]: v for v in V.values() if v["kind"] == kind and (half == "both" or v["half"] == half)}


out = {"a_F_and_the_dating_rule": {}, "b_drop_and_token_repetition": {}}
A = out["a_F_and_the_dating_rule"]
for half in ("freeze", "test", "both"):
    h = {}
    for kind in ("real", "pb", "dt"):
        X = list(U(kind, half).values())
        r0 = np.array([u["r0"] for u in X]); F = np.array([u["F"] for u in X])
        h[kind] = {"n": len(X),
                   "share_B_birth_gt_B_pre": float(np.mean([u["B_birth"] > u["B_pre"] for u in X])),
                   "share_r_c_le_r0": float(np.mean([u["r_c"] <= u["r0"] for u in X])),
                   "share_r_o_ge_r0": float(np.mean([u["r_o"] >= u["r0"] for u in X])),
                   "share_F_ge_0": float(np.mean(F >= 0)),
                   "share_F_gt_0_given_r0_gt_1": float(np.mean(F[r0 > 1] > 0)) if (r0 > 1).any() else None,
                   "rho_F_log_r0": float(spearmanr(F, np.log10(r0)).statistic),
                   "F_minus_move": boot([u["F"] - u["move"] for u in X]),
                   "F_by_r0": {lab: boot(F[(r0 >= a) & (r0 <= b)]) for lab, a, b in
                               (("r0=1", 1, 1), ("2-10", 2, 10), ("11-100", 11, 100), (">100", 101, 10 ** 9))},
                   "negatives": [[u["unit"], u["r0"], u["r1"], u["r_c"], u["r_o"], u["F"]] for u in X if u["F"] < 0]}
    # real minus twin F matched on the starting rank stratum (strata as above), pooled with stratum weights of real
    for tw in ("pb", "dt"):
        R, T = U("real", half), U(tw, half)
        ids = sorted(set(R) & set(T))
        d_all = [R[i]["F"] - T[i]["F"] for i in ids]
        # residual of F on log10 r0 within each kind (OLS per kind), then paired residual difference
        def resid(D):
            x = np.array([np.log10(u["r0"]) for u in D.values()]); y = np.array([u["F"] for u in D.values()])
            b = np.polyfit(x, y, 1)
            return {k: u["F"] - np.polyval(b, np.log10(u["r0"])) for k, u in D.items()}, b.tolist()
        pooled = {**{("R", k): v for k, v in R.items()}, **{("T", k): v for k, v in T.items()}}
        x = np.array([np.log10(u["r0"]) for u in pooled.values()]); y = np.array([u["F"] for u in pooled.values()])
        b = np.polyfit(x, y, 1)
        res = {k: u["F"] - np.polyval(b, np.log10(u["r0"])) for k, u in pooled.items()}
        d_res = [res[("R", i)] - res[("T", i)] for i in ids]
        h[f"F_real_minus_{tw}"] = {"raw": boot(d_all), "after_common_fit_on_log10_r0": boot(d_res),
                                   "common_fit_slope_intercept": b.tolist(),
                                   "median_log10_r0_real": float(np.median([np.log10(R[i]["r0"]) for i in ids])),
                                   "median_log10_r0_twin": float(np.median([np.log10(T[i]["r0"]) for i in ids]))}
    A[half] = h

B = out["b_drop_and_token_repetition"]
for half in ("freeze", "test", "both"):
    h = {}
    for kind in ("real", "pb", "dt"):
        X = list(U(kind, half).values())
        sh = np.array([u["d"] / u["size"] for u in X]); dr = np.array([u["drop"] for u in X])
        h[kind] = {"n": len(X), "rho_drop_distinct_share": float(spearmanr(dr, sh).statistic),
                   "drop_repeated_lt_0.67": boot(dr[sh < 0.67]), "drop_varied_ge_0.67": boot(dr[sh >= 0.67]),
                   "n_repeated": int((sh < 0.67).sum()),
                   "hit_pre_repeated": float(np.mean([u["hit_pre"] for u, s in zip(X, sh) if s < 0.67]))
                   if (sh < 0.67).any() else None,
                   "hit_pre_varied": float(np.mean([u["hit_pre"] for u, s in zip(X, sh) if s >= 0.67]))
                   if (sh >= 0.67).any() else None,
                   "median_bar_repeated": float(np.median([u["bar_pre"] for u, s in zip(X, sh) if s < 0.67]))
                   if (sh < 0.67).any() else None,
                   "median_bar_varied": float(np.median([u["bar_pre"] for u, s in zip(X, sh) if s >= 0.67]))
                   if (sh >= 0.67).any() else None,
                   "median_Cmin_B_pre_repeated": float(np.median([u["Cmin_B_pre"] for u, s in zip(X, sh) if s < 0.67]))
                   if (sh < 0.67).any() else None,
                   "median_Cmin_B_pre_varied": float(np.median([u["Cmin_B_pre"] for u, s in zip(X, sh) if s >= 0.67]))
                   if (sh >= 0.67).any() else None}
    R, P, D = U("real", half), U("pb", half), U("dt", half)
    for tw, T in (("pb", P), ("dt", D)):
        ids = sorted(set(R) & set(T))
        rep = [i for i in ids if R[i]["d"] / R[i]["size"] < 0.67]
        var = [i for i in ids if R[i]["d"] / R[i]["size"] >= 0.67]
        h[f"drop_real_minus_{tw}"] = {
            "all": boot([R[i]["drop"] - T[i]["drop"] for i in ids]),
            "repeated_real_births": boot([R[i]["drop"] - T[i]["drop"] for i in rep]),
            "varied_real_births": boot([R[i]["drop"] - T[i]["drop"] for i in var])}
        h[f"hit_real_minus_{tw}"] = {
            "all": boot([float(R[i]["hit_pre"]) - float(T[i]["hit_pre"]) for i in ids]),
            "repeated_real_births": boot([float(R[i]["hit_pre"]) - float(T[i]["hit_pre"]) for i in rep]),
            "varied_real_births": boot([float(R[i]["hit_pre"]) - float(T[i]["hit_pre"]) for i in var])}
    # pooled OLS: drop ~ 1 + share + is_real + is_dt over all units of the half
    rows = [u for u in V.values() if half == "both" or u["half"] == half]
    Xm = np.array([[1.0, u["d"] / u["size"], u["kind"] == "real", u["kind"] == "dt"] for u in rows], float)
    y = np.array([u["drop"] for u in rows])
    coef = np.linalg.lstsq(Xm, y, rcond=None)[0]
    Xm2 = Xm[:, [0, 2, 3]]
    coef2 = np.linalg.lstsq(Xm2, y, rcond=None)[0]
    h["ols_drop_on_share_and_kind"] = {"intercept": coef[0], "share": coef[1], "real_vs_pb": coef[2],
                                       "dt_vs_pb": coef[3], "without_share_real_vs_pb": coef2[1],
                                       "without_share_dt_vs_pb": coef2[2]}
    h["ols_drop_on_share_and_kind"] = {k: float(v) for k, v in h["ols_drop_on_share_and_kind"].items()}
    B[half] = h
(here / "r4_unanticipated.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:12000])

# (a2) where the competitors sit, by token repetition of the unit's own positions (d / size < 0.67)
A2 = {}
for half in ("test", "both"):
    for kind in ("real", "pb", "dt"):
        for lab_, f in (("repeated", lambda u: u["d"] / u["size"] < 0.67), ("varied", lambda u: u["d"] / u["size"] >= 0.67)):
            X = [u for u in U(kind, half).values() if f(u)]
            if not X:
                continue
            A2[f"{half}_{kind}_{lab_}"] = {
                "n": len(X), "median_B_birth": float(np.median([u["B_birth"] for u in X])),
                "median_Cmax_B_pre": float(np.median([u["Cmax_B_pre"] for u in X])),
                "median_Cmin_B_pre": float(np.median([u["Cmin_B_pre"] for u in X])),
                "share_Cmin_B_pre_ge_0.9999": float(np.mean([u["Cmin_B_pre"] >= 0.9999 for u in X])),
                "share_pole_birth_above_all_C_pre": float(np.mean([u["pole_birth_above_allC_pre"] for u in X])),
                "share_r_o_ge_r0": float(np.mean([u["r_o"] >= u["r0"] for u in X])),
                "F": boot([u["F"] for u in X]), "median_r0": float(np.median([u["r0"] for u in X]))}
out["a2_where_the_competitors_sit"] = A2
(here / "r4_unanticipated.json").write_text(json.dumps(out, indent=1))
for k, v in A2.items():
    print(k, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in v.items() if a != "F"},
          "F", round(v["F"]["mean"], 3), [round(v["F"]["lo"], 3), round(v["F"]["hi"], 3)])
