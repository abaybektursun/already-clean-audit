"""G-VAL recount and the look at why it fails. Input: r1_abl.json (measured dL from the job's records, dLhat
recomputed here from the pull arrays H and G), units.json (positions -> window index i = tok % 2048 + 1).
Spearman with average ranks; bootstrap over units 10,000 (seed 7); 1,000 shuffles of dLhat (seed 7)."""
import json, pathlib, sys
import numpy as np
from scipy.stats import spearmanr

here = pathlib.Path(__file__).resolve().parent
abl = json.load(open(here / "r1_abl.json"))
units = {u["unit"]: u for u in json.load(open(here.parent / "units.json"))["units"]}
SEED = 7


def sp(x, y):
    return float(spearmanr(x, y).statistic)


def q(x, ps=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return {str(p): float(np.quantile(x, p)) for p in ps}


out = {"rule": "dL = the job's measured ablation (record abl[1]); dLhat = sum over valid positions of -h*G from the "
               "pull arrays (matches record abl[0] within 7e-8)"}
for half in ("freeze", "test", "both"):
    rows = [r for r in abl if half == "both" or r["half"] == half]
    dL = np.array([r["rec_x2"] for r in rows])
    dh = np.array([r["dLhat_mine"] for r in rows])
    dh_rec = np.array([r["rec_x1"] for r in rows])
    uid = np.array([r["unit"] for r in rows])
    ul = sorted(set(uid))
    idx = {u: np.flatnonzero(uid == u) for u in ul}
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(10000):
        pick = np.concatenate([idx[ul[j]] for j in rng.integers(0, len(ul), len(ul))])
        bs.append(sp(dh[pick], dL[pick]))
    rng = np.random.default_rng(SEED)
    sh = [sp(rng.permutation(dh), dL) for _ in range(1000)]
    rho = sp(dh, dL)
    lo = float(np.quantile(bs, 0.025))
    sh975 = float(np.quantile(sh, 0.975))
    # downstream loss terms touched by each pair: sum over the unit's valid positions of (2047 - tok % 2048)
    down = np.array([sum(2047 - t % 2048 for t in units[r["unit"]]["tok"]) for r in rows])
    nwin = np.array([len({t // 2048 for t in units[r["unit"]]["tok"]}) for r in rows])
    hsum = np.array([r["h_sumabs"] for r in rows])
    car = np.array([r["is_carrier"] for r in rows])
    ratio = np.abs(dL) / np.maximum(np.abs(dh), 1e-12)
    agree = np.sign(dL) == np.sign(dh)
    top = np.abs(dh) >= np.quantile(np.abs(dh), 0.9)
    big = np.abs(dh) >= 0.1
    bins = np.quantile(hsum, [0, 0.25, 0.5, 0.75, 1.0])
    hb = np.clip(np.searchsorted(bins, hsum, "right") - 1, 0, 3)
    o = {"n_pairs": int(dL.size), "n_units": len(ul), "rho": rho, "rho_record_dLhat": sp(dh_rec, dL),
         "boot_lo": lo, "boot_hi": float(np.quantile(bs, 0.975)), "shuffle_q975": sh975,
         "rho_sign_flipped": sp(-dh, dL),
         "pass_rule": bool(rho >= 0.7 and lo > sh975 and rho <= 0.999),
         "slope_through_origin": float((dL * dh).sum() / (dh * dh).sum()),
         "slope_ols": float(np.polyfit(dh, dL, 1)[0]),
         "abs_dL": q(np.abs(dL)), "abs_dLhat": q(np.abs(dh)),
         "ratio_abs_dL_over_abs_dLhat": q(ratio), "share_ratio_gt_10": float((ratio > 10).mean()),
         "sign_agreement": float(agree.mean()),
         "rho_absdL_absdLhat": sp(np.abs(dL), np.abs(dh)),
         "rho_absdL_hsumabs": sp(np.abs(dL), hsum),
         "rho_absdL_downstream_terms": sp(np.abs(dL), down),
         "rho_absdLhat_downstream_terms": sp(np.abs(dh), down),
         "rho_ratio_hsumabs": sp(ratio, hsum),
         "by_hsumabs_quartile": [{"hsum_range": [float(bins[b]), float(bins[b + 1])], "n": int((hb == b).sum()),
                                  "rho": sp(dh[hb == b], dL[hb == b]),
                                  "sign_agree": float(agree[hb == b].mean()),
                                  "median_ratio": float(np.median(ratio[hb == b])),
                                  "median_abs_dL": float(np.median(np.abs(dL[hb == b]))),
                                  "median_abs_dLhat": float(np.median(np.abs(dh[hb == b])))} for b in range(4)],
         "top_decile_abs_dLhat": {"n": int(top.sum()), "rho": sp(dh[top], dL[top]),
                                  "sign_agree": float(agree[top].mean()),
                                  "median_ratio": float(np.median(ratio[top]))},
         "abs_dLhat_ge_0.1": {"n": int(big.sum()), "rho": sp(dh[big], dL[big]) if big.sum() > 2 else None,
                              "sign_agree": float(agree[big].mean()) if big.any() else None},
         "carrier_rows": {"n": int(car.sum()), "rho": sp(dh[car], dL[car]), "sign_agree": float(agree[car].mean()),
                          "median_abs_dL": float(np.median(np.abs(dL[car]))),
                          "median_abs_dLhat": float(np.median(np.abs(dh[car])))},
         "competitor_rows": {"n": int((~car).sum()), "rho": sp(dh[~car], dL[~car]),
                             "sign_agree": float(agree[~car].mean()),
                             "median_abs_dL": float(np.median(np.abs(dL[~car]))),
                             "median_abs_dLhat": float(np.median(np.abs(dh[~car])))},
         "abs_dL_per_sqrt_downstream": q(np.abs(dL) / np.sqrt(down)),
         "n_windows_per_unit": q(nwin)}
    # how the measured dL spreads within a unit: the 4 poles of a unit share windows and positions
    within = [np.std(dL[idx[u]]) for u in ul]
    o["median_within_unit_sd_dL"] = float(np.median(within))
    # positions at the window start (few downstream terms) against late positions
    o["median_abs_dL_by_downstream_tercile"] = [float(np.median(np.abs(dL[(down >= a_) & (down <= b_)])))
                                                for a_, b_ in zip(np.quantile(down, [0, 1 / 3, 2 / 3]),
                                                                  np.quantile(down, [1 / 3, 2 / 3, 1]))]
    out[half] = o
(here / "r2_gval.json").write_text(json.dumps(out, indent=1))
for h in ("freeze", "test", "both"):
    o = out[h]
    print(h, {k: o[k] for k in ("n_pairs", "rho", "boot_lo", "shuffle_q975", "slope_through_origin", "sign_agreement",
                                "rho_absdL_absdLhat", "rho_absdL_hsumabs", "rho_absdL_downstream_terms",
                                "share_ratio_gt_10")})
    print("  ratio", o["ratio_abs_dL_over_abs_dLhat"], "\n  |dL|", o["abs_dL"], "\n  |dLhat|", o["abs_dLhat"])
    print("  quartiles", json.dumps(o["by_hsumabs_quartile"]))
    print("  top", o["top_decile_abs_dLhat"], o["abs_dLhat_ge_0.1"], "\n  car", o["carrier_rows"], "\n  comp", o["competitor_rows"])
    print("  within-unit sd", o["median_within_unit_sd_dL"], "terciles", o["median_abs_dL_by_downstream_tercile"])

# ---- power of G-VAL at this noise level: if dL were exactly dLhat plus noise of the measured size, what rho?
# noise sample: the measured dL of the rows whose |dLhat| < 0.005 nats (their first-order effect is below 1 % of
# the median |dL|); drawn with replacement, with a random sign, scaled by sqrt(downstream terms / their median).
rng = np.random.default_rng(SEED)
rows = abl
dL = np.array([r["rec_x2"] for r in rows])
dh = np.array([r["dLhat_mine"] for r in rows])
down = np.array([sum(2047 - t % 2048 for t in units[r["unit"]]["tok"]) for r in rows], float)
low = np.abs(dh) < 0.005
z = dL[low] / np.sqrt(down[low])
sim = {"n_noise_rows": int(low.sum()), "median_abs_dL_noise_rows": float(np.median(np.abs(dL[low]))),
       "sign_agree_noise_rows": float((np.sign(dL[low]) == np.sign(dh[low])).mean())}
for beta in (1.0, 3.0, 10.0):
    r_ = []
    for _ in range(1000):
        eps = rng.choice(z, dL.size) * rng.choice([-1, 1], dL.size) * np.sqrt(down)
        r_.append(sp(dh, beta * dh + eps))
    sim[f"rho_if_dL_eq_{beta:g}x_dLhat_plus_noise"] = {"median": float(np.median(r_)),
                                                         "q975": float(np.quantile(r_, 0.975)),
                                                         "share_ge_0.7": float(np.mean(np.array(r_) >= 0.7))}
out["power_simulation"] = sim
for h in ("freeze", "test", "both"):
    out[h]["rho_ratio_hsumabs_printed"] = out[h]["rho_ratio_hsumabs"]
(here / "r2_gval.json").write_text(json.dumps(out, indent=1))
print(json.dumps(sim, indent=1))
print({h: out[h]["rho_ratio_hsumabs"] for h in ("freeze", "test", "both")})

# ---- the same model, the sign agreement it implies (overall and in the top decile of |dLhat|), against the observed
rng = np.random.default_rng(SEED)
top = np.abs(dh) >= np.quantile(np.abs(dh), 0.9)
for beta in (0.0, 1.0):
    sa, st, rr = [], [], []
    for _ in range(1000):
        eps = rng.choice(z, dL.size) * rng.choice([-1, 1], dL.size) * np.sqrt(down)
        y = beta * dh + eps
        sa.append(np.mean(np.sign(y) == np.sign(dh)))
        st.append(np.mean(np.sign(y[top]) == np.sign(dh[top])))
        rr.append(sp(dh, y))
    sim[f"beta_{beta:g}"] = {"sign_agree_median": float(np.median(sa)), "sign_agree_top_decile_median": float(np.median(st)),
                             "sign_agree_top_decile_q025_q975": [float(np.quantile(st, 0.025)), float(np.quantile(st, 0.975))],
                             "rho_median": float(np.median(rr)), "rho_q025_q975": [float(np.quantile(rr, 0.025)),
                                                                                   float(np.quantile(rr, 0.975))]}
sim["observed_both_halves"] = {"rho": out["both"]["rho"], "sign_agree": out["both"]["sign_agreement"],
                               "sign_agree_top_decile": out["both"]["top_decile_abs_dLhat"]["sign_agree"]}
out["power_simulation"] = sim
(here / "r2_gval.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: sim[k] for k in ("beta_0", "beta_1", "observed_both_halves")}, indent=1))
