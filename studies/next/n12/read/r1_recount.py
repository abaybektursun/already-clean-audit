"""N12 independent recount, phase 1 (written before opening verdict_n12.json, thresholds.json's DISCOVERY block,
n12.py's verdict and label functions, the N12 JOURNAL.md, or the spec's section 12).

Inputs: the job's per-unit arrays (S3 results/next_n12/score-02/out/score/n12/arrays/tNN/, synced to --arrays),
units.json, the job's records n12_{freeze,test}.jsonl (only for the measured ablation dL and for per-unit
comparison), and q999 / max_abs members of the grid stats files (fetch_stats.py) for the pull and the matched poles.
Everything else is computed here from B, P, Pp, G, H, Glm.

Definitions follow N12_SPEC.md section 0.5 as written. Choices of this reader where the spec is silent:
  competitor ties at the 20th place: descending B, then the lower flat index 2n+s (counted and reported);
  bar = the 10th-highest B among the layer's other 22,015 poles (the unit pole excluded);
  Spearman with average ranks (scipy.stats.spearmanr); undefined when either side is constant.
Bootstrap: percentile, 10,000 resamples over units (or births when paired), seed 7 (this reader's).
"""
import argparse, json, pathlib
import numpy as np
from scipy.stats import spearmanr

SEED = 7
NB = 10000
NC = 20


def fname(unit):
    return unit.replace(":", "_")


def load(root, t, unit, role):
    z = np.load(root / f"t{t:02d}" / f"{fname(unit)}_{role}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def ranks_of(B, idx):
    """layer rank = number of poles (all 22,016) with B >= B[idx]; ties count against idx."""
    srt = np.sort(B)
    return B.size - np.searchsorted(srt, B[np.atleast_1d(idx)], "left")


def rank1(B, i):
    return int(ranks_of(B, i)[0])


def boot_mean(x, seed=SEED, nb=NB):
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, (nb, x.size))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def wilson(h, n, z=1.959964):
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"hits": int(h), "n": int(n), "p": float(p), "lo": float(max(0, c - w)), "hi": float(min(1, c + w))}


def lab3(r, names):
    return names[0] if r["lo"] > 0 else (names[2] if r["hi"] < 0 else names[1])


def rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def unit_values(u, root, stats_dir):
    n, s = u["carrier"]
    i = 2 * n + s
    t0, t1 = u["roles"]["pre"], u["roles"]["birth"]
    pre, bir = load(root, t0, u["unit"], "pre"), load(root, t1, u["unit"], "birth")
    assert list(pre["tok"]) == list(u["tok"]) == list(bir["tok"]), u["unit"]
    Bp, Bb = pre["B"].astype(np.float64).ravel(), bir["B"].astype(np.float64).ravel()
    assert Bp.size == 22016 and not np.isnan(Bp).any() and not np.isnan(Bb).any()
    r0, r1 = rank1(Bp, i), rank1(Bb, i)
    Bc = Bp.copy(); Bc[i] = Bb[i]
    Bo = Bb.copy(); Bo[i] = Bp[i]
    r_c, r_o = rank1(Bc, i), rank1(Bo, i)
    # competitors: top 20 of B at T_(k-1), B > 0, the pole's neuron (both signs) excluded
    ok = Bp > 0
    ok[2 * n] = ok[2 * n + 1] = False
    cand = np.flatnonzero(ok)
    order = cand[np.argsort(-Bp[cand], kind="stable")]
    C = order[:NC]
    tie20 = bool(order.size > NC and Bp[order[NC - 1]] == Bp[order[NC]])
    rp, rb = ranks_of(Bp, C), ranks_of(Bb, C)
    gainC = np.log10(rp) - np.log10(rb)
    gain_pole = np.log10(r0) - np.log10(r1)
    others = np.delete(Bp, i)
    bar = float(np.sort(others)[::-1][9])
    leaders = int((Bp[C] > Bp[i]).sum())
    v = {"unit": u["unit"], "kind": u["kind"], "half": u["half"], "birth": u["birth"], "layer": u["layer"],
         "k": u["k"], "size": u["size"], "d": u["d"], "d_real": u["d_real"], "pole": u["pole"], "i": i,
         "r0": r0, "r1": r1, "r_c": r_c, "r_o": r_o, "F": float(np.log10(r_o) - np.log10(r_c)),
         "move": float(gain_pole), "c_own": float(np.log10(r0) - np.log10(r_c)),
         "c_oth": float(np.log10(r0) - np.log10(r_o)),
         "B_pre": float(Bp[i]), "B_birth": float(Bb[i]), "bar_pre": bar, "n_leaders": leaders,
         "C": [int(c) for c in C], "C_B_pre": Bp[C].tolist(), "C_B_birth": Bb[C].tolist(),
         "C_rank_pre": rp.tolist(), "C_rank_birth": rb.tolist(), "tie20": tie20, "n_cand": int(order.size),
         "drop": float(np.median(-gainC)), "u_gain": float(np.mean(gainC >= gain_pole)),
         "hit_pre": r0 <= 10, "hit_birth": r1 <= 10, "hit_rc": r_c <= 10, "hit_ro": r_o <= 10,
         "Cmax_B_pre": float(Bp[C].max()), "Cmin_B_pre": float(Bp[C].min()),
         "pole_birth_above_allC_pre": bool(Bb[i] > Bp[C].max())}
    # identity: T_k replaced by T_(k-1)
    Bc2 = Bp.copy(); Bo2 = Bp.copy()
    v["id_r_c"], v["id_r_o"] = rank1(Bc2, i), rank1(Bo2, i)
    # pull at pre
    pu = load(root, t0, u["unit"], "pull")
    P, Pp = pu["P"].astype(np.float64).ravel(), pu["Pp"].astype(np.float64).ravel()
    S = P - Pp
    v.update({"S_pole": float(S[i]), "P_pole": float(P[i]), "Pp_pole": float(Pp[i]),
              "u_S": float(np.mean(S[C] >= S[i])), "u_P": float(np.mean(P[C] >= P[i])),
              "u_Pp": float(np.mean(Pp[C] >= Pp[i])), "rho_S_gain": rho(S[C], gainC),
              "rho_S_Bpre": rho(S[C], Bp[C]), "S_gap_min": float(np.min(np.abs(S[C] - S[i]))),
              "n_valid": int(pu["valid"].sum()), "C_S": S[C].tolist()})
    # P and Pp recomputed from G, Glm and the banked q999 at T_(k-1)
    st = np.load(stats_dir / f"t{t0:02d}_L{u['layer']:02d}.npz")
    q = st["q999"].astype(np.float64)
    val = pu["valid"].astype(bool)
    G, Glm, H = pu["G"].astype(np.float64), pu["Glm"].astype(np.float64), pu["H"].astype(np.float64)
    sg = np.array([1.0, -1.0])
    Pm = (-G[val].mean(0))[:, None] * sg[None, :] * q
    Ppm = (-Glm[val].mean(0))[:, None] * sg[None, :] * q
    fin = np.isfinite(Pm.ravel())
    scale = np.abs(P[fin]).max()
    v["P_recalc_maxabs_err"] = float(np.abs(Pm.ravel()[fin] - P[fin]).max() / scale)
    v["Pp_recalc_maxabs_err"] = float(np.abs(Ppm.ravel()[fin] - Pp[fin]).max() / max(np.abs(Pp[fin]).max(), 1e-30))
    v["n_q999_nan"] = int((~fin).sum())
    v["P_nan_where_q_nan"] = bool(np.all(np.isnan(P[~fin])) if (~fin).any() else True)
    v["P_at_q_nan"] = P[~fin].tolist()[:4]
    # dLhat for every neuron: sum over valid positions of -h G (float64)
    v["_dLhat_neuron"] = (-(H[val] * G[val]).sum(0))
    v["_h_absmax_neuron"] = np.abs(H[val]).max(0)
    v["_h_sumabs_neuron"] = np.abs(H[val]).sum(0)
    v["_G_absmax_neuron"] = np.abs(G[val]).max(0)
    v["_HG_f32"] = (-(pu["H"][val].astype(np.float32) * pu["G"][val].astype(np.float32)).sum(0, dtype=np.float32))
    # matched poles: 100 nearest in log max_abs at T_(k-1), the pole's neuron excluded, max_abs > 0, stable order
    if u["kind"] == "real":
        ma = st["max_abs"].astype(np.float64)
        assert ma[n, s] > 0
        okm = ma > 0
        okm[n, :] = False
        dd = np.full(ma.shape, np.inf)
        dd[okm] = np.abs(np.log(ma[okm]) - np.log(ma[n, s]))
        mi = np.argsort(dd.ravel(), kind="stable")[:100]
        v["u_matched"] = [float(np.mean(S[C] >= S[m])) for m in mi]
        v["matched_in_C"] = [j for j, m in enumerate(mi) if int(m) in set(int(c) for c in C)]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, sd, out = pathlib.Path(a.arrays), pathlib.Path(a.stats), pathlib.Path(a.out)
    units = json.load(open(a.units))["units"]
    recs = {}
    for h in ("freeze", "test"):
        for line in open(pathlib.Path(a.records) / f"n12_{h}.jsonl"):
            r = json.loads(line)
            recs[r["unit"]] = r
    assert len(recs) == len(units) == 442
    V = {u["unit"]: unit_values(u, root, sd) for u in units}

    # ---- the dt construction control at T_15 (end role) and the opposite pole
    for u in units:
        if u["kind"] != "dt":
            continue
        n, s = u["carrier"]
        e = load(root, 15, u["unit"], "end")
        assert list(e["tok"]) == list(u["tok"])
        Be = e["B"].astype(np.float64).ravel()
        V[u["unit"]]["end_rank"] = rank1(Be, 2 * n + s)
        V[u["unit"]]["end_rank_opp"] = rank1(Be, 2 * n + (1 - s))

    # ---- per-unit comparison against the job's records
    cmp = {"n": 0, "fields": {}}
    def chk(name, a_, b_, tol=0.0):
        d = cmp["fields"].setdefault(name, {"n": 0, "mismatch": 0, "max_abs_diff": 0.0, "examples": []})
        d["n"] += 1
        if a_ is None or b_ is None:
            ok = a_ is None and b_ is None
            diff = 0.0 if ok else float("inf")
        else:
            diff = float(np.max(np.abs(np.asarray(a_, float) - np.asarray(b_, float)))) if np.size(a_) else 0.0
            ok = diff <= tol
        d["max_abs_diff"] = max(d["max_abs_diff"], diff)
        if not ok:
            d["mismatch"] += 1
            if len(d["examples"]) < 5:
                d["examples"].append([unit, a_ if np.size(a_) < 25 else "...", b_ if np.size(b_) < 25 else "..."])
    abl = []
    for unit, v in V.items():
        r = recs[unit]
        cmp["n"] += 1
        for f in ("r0", "r1", "r_c", "r_o"):
            chk(f, v[f], r["cf"][f])
        chk("F", v["F"], r["cf"]["F"], 1e-12)
        chk("move", v["move"], r["cf"]["move"], 1e-12)
        chk("B_pre", v["B_pre"], r["B_pre"])
        chk("B_birth", v["B_birth"], r["B_birth"])
        chk("bar_pre", v["bar_pre"], r["bar_pre"])
        chk("C_set_ordered", 0 if [f"L{v['layer']:02d}N{c // 2}{'+-'[c % 2]}" for c in v["C"]] ==
            [c["pole"] for c in r["comp"]] else 1, 0)
        chk("C_set_unordered", 0 if sorted(f"L{v['layer']:02d}N{c // 2}{'+-'[c % 2]}" for c in v["C"]) ==
            sorted(c["pole"] for c in r["comp"]) else 1, 0)
        chk("C_B_pre", v["C_B_pre"], [c["B_pre"] for c in r["comp"]])
        chk("C_B_birth", v["C_B_birth"], [c["B_birth"] for c in r["comp"]])
        chk("C_rank_pre", v["C_rank_pre"], [c["rank_pre"] for c in r["comp"]])
        chk("C_rank_birth", v["C_rank_birth"], [c["rank_birth"] for c in r["comp"]])
        chk("drop", v["drop"], r["drop"], 1e-12)
        chk("gain_pole", v["move"], r["gain_pole"], 1e-12)
        chk("u_gain", v["u_gain"], r["pull"]["u_gain"], 1e-12)
        chk("S_pole", v["S_pole"], r["pull"]["S"], 1e-15)
        chk("C_S", v["C_S"], [c["S"] for c in r["comp"]], 1e-15)
        chk("u_S", v["u_S"], r["pull"]["u_S"], 1e-12)
        chk("u_P", v["u_P"], r["pull"]["u_P"], 1e-12)
        chk("u_Pp", v["u_Pp"], r["pull"]["u_Pp"], 1e-12)
        chk("n_valid", v["n_valid"], r["pull"]["n_valid"])
        chk("rho_S_gain", v["rho_S_gain"], r["pull"]["rho_S_gain"], 1e-12)
        chk("rho_S_Bpre", v["rho_S_Bpre"], r["pull"]["rho_S_Bpre"], 1e-12)
        chk("cf_identity", [v["id_r_c"], v["id_r_o"], 0.0], [r["cf_identity"]["r_c"], r["cf_identity"]["r_o"],
                                                            r["cf_identity"]["F"]])
        chk("cf_identity_equals_r0", [v["id_r_c"], v["id_r_o"]], [v["r0"], v["r0"]])
        chk("d", v["d"], r["d"])
        if v["kind"] == "real":
            chk("u_matched", v["u_matched"], r["pull"]["u_matched"], 1e-12)
            lay = v["layer"]
            for pole, x1, x2 in r["abl"]:
                nn = int(pole[4:-1])
                abl.append({"unit": unit, "half": v["half"], "pole": pole, "is_carrier": pole == v["pole"],
                            "rec_x1": x1, "rec_x2": x2, "dLhat_mine": float(v["_dLhat_neuron"][nn]),
                            "dLhat_mine_f32": float(v["_HG_f32"][nn]),
                            "h_absmax": float(v["_h_absmax_neuron"][nn]), "h_sumabs": float(v["_h_sumabs_neuron"][nn]),
                            "G_absmax": float(v["_G_absmax_neuron"][nn])})
                assert pole.startswith(f"L{lay:02d}N")
            ok3 = [p for p, _, _ in r["abl"]] == [v["pole"]] + [f"L{v['layer']:02d}N{c // 2}{'+-'[c % 2]}" for c in v["C"][:3]]
            chk("abl_poles_are_carrier_and_top3", 0 if ok3 else 1, 0)
    for v in V.values():  # the per-competitor lists were compared above; not written, to keep the file small
        for k in [k for k in v if k.startswith("_") or k in ("C_B_pre", "C_B_birth", "C_S", "C_rank_pre",
                                                               "C_rank_birth")]:
            del v[k]
    (out / "r1_units.json").write_text(json.dumps({"units": V, "compare_with_records": cmp}, separators=(",", ":")))
    (out / "r1_abl.json").write_text(json.dumps(abl, separators=(",", ":")))
    print(json.dumps({k: (d["mismatch"], d["max_abs_diff"]) for k, d in cmp["fields"].items()}, indent=0))


if __name__ == "__main__":
    main()
