"""N13 independent read, step 3: the verdict quantities recomputed by the reader's own code (not verdict_n13.py, no n13 import).

    python -I verdict_read.py <recount_read.json> <thresholds.json> <controls.json> <out.json> [--boot-seed 0] [--q 0.05]
                              [--ci bootstrap|wilson_diff]

N13_SPEC.md 0.5 (sigma, delta, H, Q), 6.3 (labels, k_rec, BEATS, WITHIN, the hypothesis tree, necessity, the GSM8K line), 7 (exact
two-sided McNemar on discordant problems, Benjamini-Hochberg over the 14 primary tests, paired bootstrap over problems, 2,000
resamples, numpy default_rng(seed) drawing integer indices). Labels compare exact fractions: B, L, H = B + G/2 and Q = B + G/4 are
taken as fractions of the base and rl counts, and on the registered banks they are asserted equal to the frozen floats of
thresholds.json. run() is also called by checks_read.py on modified score vectors (thresholds then recomputed from those vectors).
Fail-loud: no try/except.
"""
import argparse
import json
import math
import pathlib
from fractions import Fraction

import numpy as np

KS = ("k0.1", "k0.3", "k1", "k3")
ENDPOINTS = ("base", "rl", "zero", "id")


def acc(x):
    return 100.0 * x.sum() / len(x)


def sigma(xa, xb):
    return 100.0 * math.sqrt(float(((xa - xb) ** 2).sum()) / 2.0) / len(xa)


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = Fraction(sum(math.comb(n, i) for i in range(min(b, c) + 1)), 2 ** n)
    return b, c, float(min(Fraction(1), 2 * tail))


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run_ = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run_ = min(run_, ps[i] * m / rank)
        adj[i] = min(run_, 1.0)
    return adj


def boot(x, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(2000, len(x)))
    d = 100.0 * (x[idx].mean(axis=1) - y[idx].mean(axis=1))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def wilson_diff(x, y):
    # Newcombe's interval for a difference of two proportions built from the two Wilson intervals (ignores pairing): check (g) only.
    z = 1.959964

    def w(k, n):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return c - h, c + h
    n = len(x)
    p1, p2 = x.sum() / n, y.sum() / n
    l1, u1 = w(x.sum(), n)
    l2, u2 = w(y.sum(), n)
    dlt = p1 - p2
    return [100 * (dlt - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 100 * (dlt + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))]


def art_rule(jf, jr):
    if jf >= 0.7 and jr >= 0.7:
        return "REFUTED"
    if jf < 0.4 or jr < 0.4:
        return "SUPPORTED"
    return "INDETERMINATE"


def run(vm, vg, C, T=None, boot_seed=0, q=0.05, ci="bootstrap"):
    """vm, vg: arm -> 0/1 numpy vector on MATH-500 and GSM8K (same problems in every arm). T: the frozen thresholds, checked equal to the
    values recomputed from vm when given; when None the recomputed values are used."""
    xb, xr, xz, xi = vm["base"], vm["rl"], vm["zero"], vm["id"]
    n = len(xb)
    assert all(len(v) == n for v in vm.values())
    sb, sr = sigma(xb, xz), sigma(xr, xi)
    sg = max(sb, sr)
    mine = {"sigma_base": sb, "sigma_rl": sr, "sigma": sg, "delta": 2 * math.sqrt(2) * sg, "B": float(acc(xb)), "L": float(acc(xr)),
            "B_prime": float(acc(xz)), "L_prime": float(acc(xi))}
    mine["G"] = mine["L"] - mine["B"]
    mine["H"] = mine["B"] + mine["G"] / 2
    mine["Q"] = mine["B"] + mine["G"] / 4
    frozen_match = None
    if T is not None:
        frozen_match = {k: bool(abs(mine[k] - T[k]) < 1e-9) for k in mine}
        assert all(frozen_match.values()), frozen_match
    D = mine["delta"] if T is None else T["delta"]
    Bf, Lf = Fraction(100 * int(xb.sum()), n), Fraction(100 * int(xr.sum()), n)
    Hf, Qf = Bf + (Lf - Bf) / 2, Bf + (Lf - Bf) / 4
    if T is not None:
        assert abs(float(Hf) - T["H"]) < 1e-9 and abs(float(Qf) - T["Q"]) < 1e-9 and abs(float(Bf) - T["B"]) < 1e-9

    def label(arm):
        x = vm[arm]
        af = Fraction(100 * int(x.sum()), n)
        if float(af) < float(Bf) - D:
            return "DAMAGED"
        if float(af) > float(Lf) + D:
            return "ABOVE"
        if af >= Hf:
            return "REC"
        if af > Qf:
            return "PART"
        return "NONE"

    arms = {}
    for arm in sorted(vm):
        a_ = float(acc(vm[arm]))
        arms[arm] = {"k": int(vm[arm].sum()), "n": n, "acc": a_, "label": None if arm in ENDPOINTS else label(arm),
                     "near_threshold": any(abs(a_ - t) < 1e-6 for t in (float(Bf) - D, float(Lf) + D, float(Hf), float(Qf)))}

    fam = [(f"top_{k}", f"{c}_{k}") for k in KS for c in ("dare", "rndE", "rndT")] + [("mlp", "attn"), ("top_k1", "comp_k1")]
    cmp = {}
    for x, y in fam:
        vx, vy = vm[x], vm[y]
        b, c, p = mcnemar(vx, vy)
        civ = boot(vx, vy, boot_seed) if ci == "bootstrap" else wilson_diff(vx, vy)
        cmp[f"{x}|{y}"] = {"diff": float(acc(vx) - acc(vy)), "b_x_right_y_wrong": b, "c_x_wrong_y_right": c, "discordant": b + c,
                           "mcnemar_p": p, "ci95": civ}
    adj = bh([cmp[f"{x}|{y}"]["mcnemar_p"] for x, y in fam])
    for (x, y), qq in zip(fam, adj):
        cmp[f"{x}|{y}"]["bh_p"] = qq
    assert len(cmp) == 14

    def beats(x, y):
        return bool(arms[x]["acc"] - arms[y]["acc"] > D and cmp[f"{x}|{y}"]["bh_p"] < q)

    def within(y, x):                              # WITHIN(y, x): acc(y) >= acc(x) - delta
        return bool(arms[y]["acc"] >= arms[x]["acc"] - D)

    lab = lambda z: arms[z]["label"]
    k_rec = next((k for k in KS if lab(f"top_{k}") in ("REC", "ABOVE")), None)
    Hy = {"k_rec": k_rec or "NONE"}
    if k_rec is None:
        Hy["H_local"] = "REFUTED (k_rec NONE)"
    else:
        t = f"top_{k_rec}"
        ctl = [f"{c}_{k_rec}" for c in ("dare", "rndE", "rndT")]
        if any(within(z, t) for z in ctl):
            Hy["H_local"] = "REFUTED (a control within delta)"
        elif lab(f"rndT_{k_rec}") == "DAMAGED":
            Hy["H_local"] = "INDETERMINATE (rndT damaged)"
        elif all(beats(t, z) for z in ctl):
            Hy["H_local"] = "SUPPORTED"
        else:
            Hy["H_local"] = "INDETERMINATE"
    dw = [k for k in KS if lab(f"dare_{k}") in ("REC", "ABOVE") and within(f"dare_{k}", f"top_{k}")]
    df = [k for k in KS if lab(f"dare_{k}") not in ("REC", "ABOVE") or beats(f"top_{k}", f"dare_{k}")]
    assert not set(dw) & set(df)
    Hy["H_diffuse"] = f"SUPPORTED (smallest k {dw[0]})" if dw else ("REFUTED" if len(df) == 4 else "INDETERMINATE")
    Hy["H_diffuse_DW"], Hy["H_diffuse_DF"] = dw, df
    la, lm = lab("attn"), lab("mlp")
    if la in ("REC", "ABOVE") and lm in ("NONE", "DAMAGED"):
        Hy["H_attention"] = "SUPPORTED"
    elif lm in ("REC", "ABOVE") and la in ("NONE", "DAMAGED"):
        Hy["H_attention"] = "REFUTED"
    elif lm in ("REC", "ABOVE") and la in ("REC", "ABOVE"):
        Hy["H_attention"] = "REFUTED (BOTH)"
    else:
        Hy["H_attention"] = "INDETERMINATE (" + ("interaction" if la == lm == "NONE" else "PARTIAL") + ")"
    pc = C["PC-ART"]
    rp = art_rule(pc["P_plus"]["J_floor"]["k1"], pc["P_plus"]["J_round"]["k1"])
    rm = art_rule(pc["P_minus"]["J_floor"]["k1"], pc["P_minus"]["J_round"]["k1"])
    assert rp == pc["rule_P_plus"] and rm == pc["rule_P_minus"]
    Hy["H_artifact"] = (f"CANCELLED (PC-ART: P+ {rp}, P- {rm})" if (rp != "SUPPORTED" or rm != "REFUTED")
                        else art_rule(C["J_floor"]["k1"], C["J_round"]["k1"]))
    Hy["H_artifact_all_k"] = {k: {"J_floor": C["J_floor"][k], "J_round": C["J_round"][k], "rule": art_rule(C["J_floor"][k], C["J_round"][k])}
                              for k in KS}
    Hy["H_artifact_values"] = {"J_floor_k1": C["J_floor"]["k1"], "J_round_k1": C["J_round"]["k1"], "J_sp_k1": C["J_sp"]["k1"],
                               "J_round_P_plus_k1": pc["P_plus"]["J_round"]["k1"], "J_sp_P_plus_k1": pc["P_plus"]["J_sp"]["k1"],
                               "rule_P_plus": rp, "rule_P_minus": rm,
                               "J_round_mean_check": abs(float(np.mean(C["J_round_per_redraw"]["k1"])) - C["J_round"]["k1"]) < 1e-12}
    nec = {}
    for k in KS:
        c = f"comp_{k}"
        nec[k] = "NECESSARY" if lab(c) in ("NONE", "PART", "DAMAGED") else ("NOT NECESSARY" if within(c, "rl") else "neither")
    Hy["necessity"] = nec

    gsm = {}
    for c in ("dare", "rndE", "rndT"):
        x, y = "top_k1", f"{c}_k1"
        gx, gy = vg[x], vg[y]
        dm, dg = arms[x]["acc"] - arms[y]["acc"], float(acc(gx) - acc(gy))
        civ = boot(gx, gy, boot_seed) if ci == "bootstrap" else wilson_diff(gx, gy)
        same = (dm > 0 and dg > 0) or (dm < 0 and dg < 0)
        excl = civ[0] > 0 or civ[1] < 0
        lbl = "REPLICATES" if same and excl else "CONSISTENT" if same else "CONTRADICTS" if excl else "UNDEFINED (opposite sign, CI includes 0)"
        mc = mcnemar(gx, gy)
        gsm[f"{x}|{y}"] = {"diff_math500": dm, "diff_gsm8k": dg, "ci95_gsm8k": civ, "mcnemar_p_gsm8k_exploratory": mc[2],
                           "discordant_gsm8k": mc[0] + mc[1], "label": lbl}
    gsm["acc"] = {arm: {"k": int(vg[arm].sum()), "acc": float(acc(vg[arm]))} for arm in sorted(vg)}

    best = {}
    for k in KS:
        ctl = max(("dare", "rndE", "rndT"), key=lambda c: arms[f"{c}_{k}"]["acc"])
        best[k] = {"best_control": f"{ctl}_{k}", "top_minus_best": arms[f"top_{k}"]["acc"] - arms[f"{ctl}_{k}"]["acc"]}

    expl = {"sigma_mid": sigma(vm["top_k1"], vm["top_k1_B"]), "sigma_mid_over_sigma": sigma(vm["top_k1"], vm["top_k1_B"]) / sg,
            "top_k1_minus_top_k1_B": arms["top_k1"]["acc"] - arms["top_k1_B"]["acc"],
            "rndT_k1_d2_minus_rndT_k1": arms["rndT_k1_d2"]["acc"] - arms["rndT_k1"]["acc"],
            "dare_k1_d2_minus_dare_k1": arms["dare_k1_d2"]["acc"] - arms["dare_k1"]["acc"], "delta": D}
    if Hy["H_local"].startswith("REFUTED (k_rec"):
        head = (f"H_local REFUTED (k_rec NONE): no top_k reaches H = {float(Hf):.1f} (best top_3% {arms['top_k3']['acc']:.1f}); "
                f"H_diffuse {Hy['H_diffuse']}; H_attention {Hy['H_attention']}; H_artifact {Hy['H_artifact']}")
    else:
        head = f"H_local {Hy['H_local']} at k_rec {k_rec}; H_diffuse {Hy['H_diffuse']}; H_attention {Hy['H_attention']}; H_artifact {Hy['H_artifact']}"
    return {"settings": {"boot_seed": boot_seed, "q": q, "ci": ci, "n": n, "thresholds": "frozen" if T is not None else "recomputed"},
            "frozen_recomputed": mine, "frozen_match": frozen_match,
            "thresholds_used": {"delta": D, "B": float(Bf), "L": float(Lf), "H": float(Hf), "Q": float(Qf), "B_minus_delta": float(Bf) - D,
                                "L_plus_delta": float(Lf) + D},
            "arms": arms, "comparisons": cmp, "hypotheses": Hy, "gsm8k": gsm, "top_vs_best_control": best, "exploratory": expl, "headline": head}


def vectors(R, bench):
    return {v["arm"]: np.array([int(c) for c in v["scores_banked"]], dtype=np.int64) for v in R.values() if v["bench"] == bench}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("recount")
    ap.add_argument("thresholds")
    ap.add_argument("controls")
    ap.add_argument("out")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--ci", choices=["bootstrap", "wilson_diff"], default="bootstrap")
    a = ap.parse_args()
    R = json.loads(pathlib.Path(a.recount).read_text())["arms"]
    T = json.loads(pathlib.Path(a.thresholds).read_text())
    C = json.loads(pathlib.Path(a.controls).read_text())
    out = run(vectors(R, "math500"), vectors(R, "gsm8k"), C, T, a.boot_seed, a.q, a.ci)
    out["gsm8k"]["G-GATE-GSM_pass"] = T["gsm8k"]["G-GATE-GSM"]["pass"]
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(out["headline"])
    for arm, v in out["arms"].items():
        print(f"  {arm:11s} {v['acc']:5.1f} {v['label']}")
    for kk, v in out["comparisons"].items():
        print(f"  {kk:22s} diff {v['diff']:6.1f} disc {v['discordant']:3d} p {v['mcnemar_p']:.3g} bh {v['bh_p']:.3g} "
              f"ci [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]")
