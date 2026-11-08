"""TENANT aggregate v2 — POST-CRITIC (all folds implemented; supersedes v1, which is kept at
results/tenant_aggregate as the pre-critic snapshot).

Changes vs v1 (critic finding in parens): corrected BH max-index with m = m_planned (m25) ·
N2 row recomputed in one unit system from achieved paired SEs (F2/m20) · live-item fraction +
attainable ceiling per MDE contrast (M5) · anchor operating-point disclosure (M6) · leverage
overlap census (M11) · battery contrasts (M13) · branch-overlap record (M14) · mechanism
members re-reported against the FROZEN wording with run ranges, thresholds labeled
post-unblind (F4/M15) · X binds to the v4 re-runs: SGD gated non-diverged (F1), take-revert
adjudicated against the banked matched-random lag reference (F3), qk with norms + the
0.707-at-balanced no-interaction value (M7/M8) · adamfrozen two-sided discriminator (M17).

    python -m infra.aws_pg.jobs.tenant_aggregate --out <dir>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "s3://pg-lineage-769063704696/results"
N_BOOT = 10000
BOOT_SEED = 20260806
SEEDS = (1337, 1338)
MDE_PREFIX = {"incore": "tenant_mde_s{s}/out/incore",
              "leverage": "tenant_mde2_s{s}/out/leverage",
              "random": "tenant_mde2_s{s}/out/random"}
X_SRC = {"adamw_s0": "tenant_x4/out/adamw_s0", "adamw_s1": "tenant_x_full3/out/adamw_s1",
         "sgd_s0": "tenant_x4/out/sgd_s0", "sgd_s1": "tenant_x4/out/sgd_s1",
         "adamfrozen_s0": "tenant_x4/out/adamfrozen_s0"}
M_PLANNED = 7
S_SLOPE = 0.1580                 # raw acc per S unit (banked PRAGNANZ denom; critic F2)
EFF_S = 0.11                     # registered effect, S scale
ANCHOR = "tenant_anchor2/out/metrics.json"


def _pull(rel, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", f"{BUCKET}/{rel}", str(dest),
                    "--region", "us-east-1"], check=True, capture_output=True)
    return json.loads(dest.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from scipy.stats import norm as ndist

    rep = {"version": "post-critic v2", "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
           "families": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    cells = {(arm, s): _pull(f"{MDE_PREFIX[arm].format(s=s)}/metrics.json",
                             out / f"in/{arm}_s{s}.json")
             for arm in MDE_PREFIX for s in SEEDS}

    # ---- MDE censuses the doc must print (M5, M6, M11) --------------------------------------
    P = {(arm, s): np.asarray(cells[(arm, s)]["endpoint"]["per_item"], float)
         for arm in MDE_PREFIX for s in SEEDS}
    mask_all = ~np.any([np.isnan(P[k]) for k in P], axis=0)
    stack = np.stack([P[k][mask_all] for k in P])
    all_right = float((stack == 1).all(axis=0).mean())
    all_wrong = float((stack == 0).all(axis=0).mean())
    rep["mde_census"] = {
        "n_scorable": int(mask_all.sum()),
        "frac_all_correct": all_right, "frac_all_wrong": all_wrong,
        "frac_live": float(1 - all_right - all_wrong),
        "attainable_ceiling": float(1 - all_wrong),
        "leverage_overlap_core": {
            str(s): {"mean": float(np.mean([v.get("overlap_core", np.nan)
                                            for v in cells[("leverage", s)]
                                            ["basis_prov"].values()])),
                     "min": float(np.min([v.get("overlap_core", np.nan)
                                          for v in cells[("leverage", s)]
                                          ["basis_prov"].values()])),
                     "max": float(np.max([v.get("overlap_core", np.nan)
                                          for v in cells[("leverage", s)]
                                          ["basis_prov"].values()]))} for s in SEEDS},
        "haar_overlap_expectation": 32 / 2048,
        "constrained_modules": "18 of 126 (q_proj B only; ~14% of trainable capacity)"}
    anch = _pull(ANCHOR, out / "in/anchor.json")["tasks"]["cococap"]
    rep["mde_census"]["anchor_range"] = {"base": anch["base_acc"], "ft": anch["ft_acc"],
                                         "note": "all six arms operate 0.15 ABOVE the "
                                                 "anchor-validated range (critic M6)"}
    _w()

    # ---- F-placement (corrected BH; paired SEs; N2 re-print) --------------------------------
    members = []
    paired_ses = []
    for ctrl in ("leverage", "random"):
        ps, det = [], []
        for s in SEEDS:
            A, B = P[("incore", s)], P[(ctrl, s)]
            m = ~(np.isnan(A) | np.isnan(B))
            d = A[m] - B[m]
            paired_ses.append(float(d.std(ddof=1) / np.sqrt(d.size)))
            rng = np.random.default_rng(BOOT_SEED)
            idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
            boots = d[idx].mean(axis=1)
            p = float((1 + (boots <= 0).sum()) / (N_BOOT + 1))
            det.append({"seed": s, "d": float(d.mean()),
                        "lo95": float(np.percentile(boots, 2.5)),
                        "hi95": float(np.percentile(boots, 97.5)), "n": int(d.size), "p": p})
            ps.append(p)
        members.append({"name": f"incore_vs_{ctrl}_cococap", "p": max(ps), "seeds": det})
    # battery contrasts (M13): in-core vs each control on the two other tasks
    battery = []
    for ctrl in ("leverage", "random"):
        for t in ("textvqa", "vqav2"):
            for s in SEEDS:
                A = np.asarray(cells[("incore", s)]["battery"][t]["per_item"], float)
                B = np.asarray(cells[(ctrl, s)]["battery"][t]["per_item"], float)
                m = ~(np.isnan(A) | np.isnan(B))
                d = A[m] - B[m]
                rng = np.random.default_rng(BOOT_SEED)
                idx = rng.integers(0, d.size, size=(4000, d.size))
                boots = d[idx].mean(axis=1)
                battery.append({"contrast": f"incore_vs_{ctrl}", "task": t, "seed": s,
                                "d": float(d.mean()),
                                "lo95": float(np.percentile(boots, 2.5)),
                                "hi95": float(np.percentile(boots, 97.5))})
    se = float(np.mean(paired_ses))
    eff_raw = EFF_S * S_SLOPE
    n2 = {}
    for alpha, tag in ((0.05, "alpha_05"), (0.05 / M_PLANNED, "alpha_bh_m7")):
        z = ndist.ppf(1 - alpha)
        n2[tag] = float(1 - ndist.cdf(z - eff_raw / se))
    n2["achieved_paired_se"] = se
    n2["registered_effect_raw"] = eff_raw
    n2["floor80_raw"] = float((ndist.ppf(0.95) + ndist.ppf(0.80)) * se)
    rep["n2_reprint_post_critic"] = {
        **n2, "note": "F2 fold: one unit system (raw acc), achieved item-paired SEs; the MDE "
                      "was UNDERPOWERED at the BH threshold against the registered effect"}
    srt = sorted((m_["p"], m_["name"]) for m_ in members)
    k_star = max([i for i, (p, _) in enumerate(srt, 1) if p <= 0.05 * i / M_PLANNED],
                 default=0)
    bh = {m_["name"]: (m_["name"] in {n for _, n in srt[:k_star]}) for m_ in members}
    rep["families"]["F-placement"] = {
        "members": members, "battery_contrasts": battery,
        "m_run": 2, "m_planned": M_PLANNED, "bh": bh,
        "branch_overlap_disclosure": "frozen branches 2 (leverage ties in-core) AND 3 (all "
                                     "tie) BOTH fire; the Q5 census (leverage captures 1.5x "
                                     "the core) is affirmative evidence for the H_leverage "
                                     "reading (critic M14)"}
    _w()

    # ---- F-mechanism (v4 re-runs; frozen wording; run ranges; post-unblind labels) ----------
    X = {st: _pull(f"{src}/metrics.json", out / f"in/x_{st}.json")
         for st, src in X_SRC.items()}
    adam_disp = [X[f"adamw_s{i}"]["summary"]["disp_core_over_rand_median"] for i in (0, 1)]
    adam_grad = [X[f"adamw_s{i}"]["summary"]["grad_core_over_rand_median"] for i in (0, 1)]
    sgd_ok = {st: (X[st].get("final_loss_mean5") is not None) for st in ("sgd_s0", "sgd_s1")}
    sgd_disp = [X[st]["summary"]["disp_core_over_rand_median"]
                for st in ("sgd_s0", "sgd_s1")]
    sgd_grad = [X[st]["summary"]["grad_core_over_rand_median"]
                for st in ("sgd_s0", "sgd_s1")]
    sgd_final = [X[st]["final_loss_mean5"] for st in ("sgd_s0", "sgd_s1")]
    adam_final = [X[st]["final_loss_mean5"] for st in ("adamw_s0", "adamw_s1")]
    frz = X["adamfrozen_s0"]["summary"]
    lag_in = X["adamw_s0"]["lag_autocorr"]
    lag_rn = X["adamw_s0"].get("lag_autocorr_rand", {})
    lag_contrast = {k.split("__")[2]: {"in_core_lag1": lag_in[k][0],
                                       "rand_lag1": (lag_rn.get(k, [None])[0])}
                    for k in lag_in}
    qk_cells = []
    for row in X["adamw_s0"]["qk_index"]:
        for v in row.values():
            if isinstance(v, dict) and v.get("mean") is not None:
                qk_cells.append(v)
    qk_mean = float(np.mean([v["mean"] for v in qk_cells]))
    nq = float(np.mean([v.get("norm_q", np.nan) for v in qk_cells]))
    nk = float(np.mean([v.get("norm_k", np.nan) for v in qk_cells]))
    r_bal = nq / nk if nk else float("nan")
    no_int = float(np.sqrt(nq ** 2 + nk ** 2) / (nq + nk)) if nk else float("nan")
    rep["families"]["F-mechanism"] = {
        "sap_disclosure": "frozen as 3 BH-tested contrasts with run CIs; delivered with "
                          "2-run ranges and thresholds AUTHORED POST-UNBLIND (critic F4/M15) "
                          "— reported against the FROZEN wording below, verdicts labeled",
        "adam_suppression": {
            "adam_disp_range": adam_disp, "adam_grad_range": adam_grad,
            "sgd_disp_range": sgd_disp, "sgd_grad_range": sgd_grad,
            "sgd_final_loss": sgd_final, "adam_final_loss": adam_final,
            "sgd_sanity_gates": "in-job (pilot-match + non-vanishing-grad), frozen pre-re-run",
            "frozen_wording": "SGD moves the core where Adam didn't (disp ratio rises toward "
                              "the gradient ratio)",
            "adamfrozen_s0": {"disp": frz["disp_core_over_rand_median"],
                              "grad": frz["grad_core_over_rand_median"],
                              "reading": "frozen-v Adam: if the freeze persists, v-inflation "
                                         "is not the driver; if the core unfreezes, adaptive "
                                         "v tracking is (two-sided, critic M17)"}},
        "take_revert": {"lag_contrast_adamw_s0": lag_contrast,
                        "frozen_wording": "killed by autocorrelation ~ matched-random "
                                          "directions' (the F3 contrast, now measured)"},
        "qk_cancel": {"adamw_s0_mean": qk_mean, "norm_q_mean": nq, "norm_k_mean": nk,
                      "norm_ratio": r_bal, "no_interaction_value": no_int,
                      "note": "no-interaction is sqrt(nq^2+nk^2)/(nq+nk), NOT 1.0 (critic "
                              "M8); cos-angle implied by the index is reported in the doc"}}
    rep["census"] = {
        "q5_captured_share": {f"{arm}_s{s}": cells[(arm, s)].get("q5_captured_share")
                              for arm in MDE_PREFIX for s in SEEDS},
        "q5_reading": "leverage 0.40 > core 0.26 >> random 0.016 ~ Haar 0.0156 — capture is "
                      "coordinate-leverage, not core-specific (critic M12; grid row 9)",
        "confinement_final": {f"{arm}_s{s}": (cells[(arm, s)]["confinement"][-1]
                                              if cells[(arm, s)]["confinement"] else None)
                              for arm in MDE_PREFIX for s in SEEDS},
        "ppl_band": {f"{arm}_s{s}": cells[(arm, s)]["ppl"]
                     for arm in MDE_PREFIX for s in SEEDS}}
    _w()
    print("TENANT_AGGREGATE2_OK", json.dumps(
        {m_["name"]: round(m_["p"], 4) for m_ in members}
        | {"sgd_final": [round(x, 3) for x in sgd_final],
           "adamfrozen_disp": round(frz["disp_core_over_rand_median"], 3)}), flush=True)


if __name__ == "__main__":
    main()
