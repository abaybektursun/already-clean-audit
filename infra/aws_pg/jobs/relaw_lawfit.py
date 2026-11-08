"""RELAW law-fit — the MDE analytic core (prereg §3.1/§6/§7). Pure numpy (savb + relaw
predicates); reads reads_sav.json, NO model, NO GPU.

Does the cheap weight read R_prin predict banked re-learnability G out-of-sample beyond the
trivial weight-distance baseline R_dist, and beyond / fundamentally-vs the activation overlap
gauge R_ovl?

  * freeze r on the INTERIOR (family minus the named held-out member): argmax_r LOO-R²(R_prin,G);
  * ΔLOO-R²(read | R_dist) + permutation null for read ∈ {R_prin@r*, R_ovl}; BH-FDR over the family;
  * Arm-2b partials: ΔLOO-R²(R_prin|R_ovl) and the symmetric ΔLOO-R²(R_ovl|R_prin);
  * LOO sign-stability (drop each fine-tune) — a flip ⇒ non-law;
  * held-out-member OOS lift: fit on interior, the held-out squared-error reduction from adding the read;
  * within-family (overlap>cut) resolution — does R_prin resolve where SAV-C's R_ovl could not;
  * controls: planted-predictor recovery (instrument-null halt) + permutation collapse of G;
  * apply the §6 law_verdict and fill the §11 grid.

    python -m infra.aws_pg.jobs.relaw_lawfit --out <dir> --reads <reads_sav.json|s3://> \
        [--held-out paligemma-3b-ft-okvqa-224] [--within-cut 0.62] [--n-perm 5000]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

WITHIN_CUT = 0.62   # SAV-C frozen: excludes pi0/pi05/refcoco-seg — the within-family resolution test


def _arrays(units, r):
    import numpy as np
    G = np.array([u["G"] for u in units], float)
    R_dist = np.array([u["R_dist"] for u in units], float)
    R_ovl = np.array([u["R_ovl"] for u in units], float)
    R_prin = np.array([u["R_prin"][str(r)] for u in units], float)
    return G, R_dist, R_ovl, R_prin


def _heldout_lift(interior, held, r, read_key):
    """OOS lift on the single held-out member: squared-error reduction from adding `read` to the
    R_dist baseline, models fit on the interior (intercept + standardized predictors). >0 ⇒ the
    read predicts the held-out fine-tune's G better than weight-distance alone."""
    import numpy as np

    def fit_predict(cols_train, y_train, x_held):
        """OLS (intercept + standardized predictors) on the interior; predict ONE held point.
        x_held is the held point's 1-D feature vector (length = n predictors)."""
        mu = cols_train.mean(0)
        sd = cols_train.std(0)
        assert (sd > 0).all(), "constant predictor in held-out fit"
        Z = (cols_train - mu) / sd
        A = np.column_stack([np.ones(len(y_train)), Z])
        w, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        zh = (np.asarray(x_held, float) - mu) / sd            # (n_feat,)
        return float(w[0] + zh @ w[1:])                       # scalar prediction

    gi = np.array([u["G"] for u in interior], float)
    gh = held["G"]
    di = np.array([u["R_dist"] for u in interior], float)
    dh = held["R_dist"]
    ri = np.array([(u["R_prin"][str(r)] if read_key == "R_prin" else u[read_key]) for u in interior], float)
    rh = held["R_prin"][str(r)] if read_key == "R_prin" else held[read_key]
    pred_base = fit_predict(di.reshape(-1, 1), gi, np.array([dh]))
    pred_aug = fit_predict(np.column_stack([di, ri]), gi, np.array([dh, rh]))
    err_base = float((gh - pred_base) ** 2)
    err_aug = float((gh - pred_aug) ** 2)
    return {"err_base": err_base, "err_aug": err_aug, "lift": err_base - err_aug,
            "pred_base": pred_base, "pred_aug": pred_aug, "g_true": float(gh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--held-out", default="", help="held-out unit name (default: alphabetical median)")
    ap.add_argument("--within-cut", type=float, default=WITHIN_CUT)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import relaw as RL
    from infra.pg import savb as SB

    reads_path = a.reads
    if reads_path.startswith("s3://"):
        reads_path = str(out / "reads_sav.json")
        subprocess.run(["aws", "s3", "cp", a.reads, reads_path, "--region", "us-east-1"], check=True)
    R = json.loads(pathlib.Path(reads_path).read_text())
    units = sorted(R["units"], key=lambda u: u["unit"])
    r_grid = R["r_grid"]
    names = [u["unit"] for u in units]
    held_name = a.held_out or names[len(names) // 2]      # alphabetical median (named blind at freeze)
    assert held_name in names, f"held-out {held_name} not in family {names}"
    interior = [u for u in units if u["unit"] != held_name]
    held = next(u for u in units if u["unit"] == held_name)
    print(f"[RELAW-LAWFIT] n={len(units)} held-out={held_name} interior={len(interior)} "
          f"r_grid={r_grid}", flush=True)

    # ---- freeze r on the interior: argmax LOO-R²(R_prin, G) ----
    Gi = np.array([u["G"] for u in interior], float)
    loo_by_r = {}
    for r in r_grid:
        Rp = np.array([u["R_prin"][str(r)] for u in interior], float)
        loo_by_r[r] = SB.loo_r2(Rp.reshape(-1, 1), Gi)
    r_star = RL.select_predictor_r(r_grid, loo_by_r)
    print(f"[RELAW-LAWFIT] frozen r*={r_star}  LOO-R²(R_prin,G) by r: "
          + " ".join(f"{r}:{loo_by_r[r]:+.3f}" for r in r_grid), flush=True)

    # ---- full-pool law-fit at r* (LOO over all units) ----
    G, R_dist, R_ovl, R_prin = _arrays(units, r_star)
    Xd = R_dist.reshape(-1, 1)
    fit = {
        "R_prin_over_dist": SB.delta_loo_perm(Xd, R_prin, G, n_perm=a.n_perm, seed=1337),
        "R_ovl_over_dist": SB.delta_loo_perm(Xd, R_ovl, G, n_perm=a.n_perm, seed=1337),
        "R_prin_over_ovl": SB.delta_loo_perm(R_ovl.reshape(-1, 1), R_prin, G, n_perm=a.n_perm, seed=1337),
        "R_ovl_over_prin": SB.delta_loo_perm(R_prin.reshape(-1, 1), R_ovl, G, n_perm=a.n_perm, seed=1337),
    }
    spearman = {"R_prin": SB.boot_spearman(R_prin, G, n_boot=a.n_boot, seed=1337),
                "R_ovl": SB.boot_spearman(R_ovl, G, n_boot=a.n_boot, seed=1337),
                "R_dist": SB.boot_spearman(R_dist, G, n_boot=a.n_boot, seed=1337)}
    # BH-FDR over the read family vs R_dist {R_prin, R_ovl}.
    fam = ["R_prin_over_dist", "R_ovl_over_dist"]
    fdr = RL.bh_fdr([fit[k]["p_perm"] for k in fam], q=0.05)
    fdr_ok = {k: bool(fdr[i]) for i, k in enumerate(fam)}

    # ---- LOO sign-stability + held-out lift ----
    stab = RL.loo_sign_stable(Xd, R_prin, G)
    ho = {"R_prin": _heldout_lift(interior, held, r_star, "R_prin"),
          "R_ovl": _heldout_lift(interior, held, r_star, "R_ovl")}

    # ---- within-family resolution (overlap>cut) — where SAV-C's R_ovl could not resolve ----
    wf = [u for u in units if u["R_ovl"] > a.within_cut]
    within = None
    if len(wf) >= 6:
        Gw, Rdw, Row, Rpw = _arrays(wf, r_star)
        within = {"n": len(wf), "units": [u["unit"] for u in wf],
                  "R_prin_over_dist": SB.delta_loo_perm(Rdw.reshape(-1, 1), Rpw, Gw,
                                                        n_perm=a.n_perm, seed=1337),
                  "spearman_R_prin": SB.boot_spearman(Rpw, Gw, n_boot=a.n_boot, seed=1337)}

    # ---- controls ----
    # planted positive control: noise SCALED to R_prin's own spread (R_prin is near-saturated, std~1e-2;
    # a fixed noise_sd=0.05 would swamp the signal and falsely fail the control). SNR≈3 ⇒ the machinery
    # must recover a real predictor of THIS predictor's variance structure.
    planted_noise = max(0.3 * float(R_prin.std()), 1e-6)
    planted = RL.planted_g(R_prin, slope=1.0, noise_sd=planted_noise, seed=2024)
    planted_fit = SB.delta_loo_perm(Xd, R_prin, planted, n_perm=a.n_perm, seed=1337)
    instrument_ok = bool(planted_fit["delta"] > 0 and planted_fit["p_perm"] < 0.05)
    perm = np.random.default_rng(7).permutation(G.size)
    shuffle = {"real_loo_r2": SB.loo_r2(np.column_stack([R_dist, R_prin]), G),
               "perm_loo_r2": SB.loo_r2(np.column_stack([R_dist, R_prin]), G[perm])}

    # ---- §6 law verdict for each read ----
    def verdict(read_key):
        f = fit[f"{read_key}_over_dist"]
        st = stab if read_key == "R_prin" else RL.loo_sign_stable(
            Xd, R_ovl if read_key == "R_ovl" else R_prin, G)
        return {"predicts": RL.law_verdict(f["p_perm"], fdr_ok[f"{read_key}_over_dist"],
                                           ho[read_key]["lift"], st["stable"]),
                "delta": f["delta"], "p_perm": f["p_perm"], "fdr_ok": fdr_ok[f"{read_key}_over_dist"],
                "heldout_lift": ho[read_key]["lift"], "loo_stable": st["stable"]}
    verdicts = {"R_prin": verdict("R_prin"), "R_ovl": verdict("R_ovl")}

    # partials → which read is fundamental (§6 Arm-2b)
    prin_over_ovl_p = fit["R_prin_over_ovl"]["p_perm"]
    ovl_over_prin_p = fit["R_ovl_over_prin"]["p_perm"]
    if prin_over_ovl_p < 0.05 and ovl_over_prin_p >= 0.05:
        partial_verdict = "R_prin_fundamental"
    elif ovl_over_prin_p < 0.05 and prin_over_ovl_p >= 0.05:
        partial_verdict = "R_ovl_fundamental"
    elif prin_over_ovl_p < 0.05 and ovl_over_prin_p < 0.05:
        partial_verdict = "both_carry_unique_signal"
    else:
        partial_verdict = "R_prin≈R_ovl_same_construct"

    # §6 naming: PREDICTOR-EXISTS only if the CANDIDATE weight read R_prin predicts; if instead the
    # activation gauge R_ovl predicts but R_prin does not → H-overlap (the weight read is not the law);
    # if neither beats distance → H-probe. INSTRUMENT-NULL only on a valid positive-control failure.
    if not instrument_ok:
        named = "INSTRUMENT-NULL"
    elif verdicts["R_prin"]["predicts"]:
        named = "PREDICTOR-EXISTS"
    elif verdicts["R_ovl"]["predicts"]:
        named = "H-overlap-only"
    else:
        named = "H-probe"

    grid = {
        "predictor_r": r_star,
        "MDE_dLOOR2_Rprin|Rdist": [round(fit["R_prin_over_dist"]["delta"], 4),
                                   round(fit["R_prin_over_dist"]["p_perm"], 4)],
        "heldout_Rprin_lift_vs_Rdist": round(ho["R_prin"]["lift"], 5),
        "LOO_sign_stable": stab["stable"],
        "Arm2b_dLOOR2_Rprin|Rovl_p": round(prin_over_ovl_p, 4),
        "Arm2b_dLOOR2_Rovl|Rprin_p": round(ovl_over_prin_p, 4),
        "partial_verdict": partial_verdict,
        "within_family_Rprin_p": (round(within["R_prin_over_dist"]["p_perm"], 4) if within else None),
        "named_law_verdict": named,
    }

    rep = {"object": "relaw_lawfit_v1", "family": R.get("family"), "generation": R.get("generation"),
           "n_units": len(units), "held_out": held_name, "within_cut": a.within_cut,
           "r_grid": r_grid, "r_star": r_star, "loo_r2_by_r": {str(r): loo_by_r[r] for r in r_grid},
           "fit": fit, "spearman": spearman, "fdr_ok": fdr_ok, "loo_sign_stable": stab,
           "heldout": ho, "within_family": within, "partial_verdict": partial_verdict,
           "controls": {"planted_predictor": planted_fit, "instrument_ok": instrument_ok,
                        "permutation_collapse": shuffle},
           "verdicts": verdicts, "named": named, "grid": grid,
           "frozen_rule": ("PREDICTOR-EXISTS iff R_prin ΔLOO-R²(|R_dist) perm-p<0.05 (BH-FDR) ∧ "
                           "held-out lift>0 ∧ LOO sign-stable; else H-probe. INSTRUMENT-NULL iff the "
                           "planted-predictor positive control fails to recover.")}
    (out / "relaw_metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("=" * 80, flush=True)
    print(f"[RELAW-LAWFIT] r*={r_star}  R_prin|R_dist ΔLOO-R²={fit['R_prin_over_dist']['delta']:+.3f} "
          f"p={fit['R_prin_over_dist']['p_perm']:.4f} (FDR {fdr_ok['R_prin_over_dist']}) | "
          f"held-out lift={ho['R_prin']['lift']:+.4f} | LOO-stable={stab['stable']}", flush=True)
    print(f"[RELAW-LAWFIT] R_ovl|R_dist ΔLOO-R²={fit['R_ovl_over_dist']['delta']:+.3f} "
          f"p={fit['R_ovl_over_dist']['p_perm']:.4f} | partials: {partial_verdict}", flush=True)
    if within:
        print(f"[RELAW-LAWFIT] within-family(n={within['n']}) R_prin|R_dist "
              f"ΔLOO-R²={within['R_prin_over_dist']['delta']:+.3f} "
              f"p={within['R_prin_over_dist']['p_perm']:.4f}", flush=True)
    print(f"[RELAW-LAWFIT] controls: instrument_ok={instrument_ok} "
          f"perm-collapse {shuffle['real_loo_r2']:+.3f}→{shuffle['perm_loo_r2']:+.3f}", flush=True)
    print(f"[RELAW-LAWFIT] VERDICT: {named}", flush=True)
    print(f"RELAW_LAWFIT_OK verdict={named} r_star={r_star}", flush=True)


if __name__ == "__main__":
    main()
