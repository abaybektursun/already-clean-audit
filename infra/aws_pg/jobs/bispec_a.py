"""BISPEC Arm A (GPU) — the asymmetric direction of REGRAIN R18, adjudicated against its strongest
competing mechanism.

**The observation.** REGRAIN R18: traversing `x → x + α·σ₁·v₁` along PC1 of the B*-union's
contribution coordinates costs ΔVQA **−0.5733** at α = +3σ (VQA 0.623 → 0.050) but only −0.0317 at
α = −3σ; the *same vector* Haar-rotated costs −0.0833 and matched-norm random directions −0.023 to
−0.058. So the effect is direction-specific AND sign-asymmetric. R18's own frozen statistic could
not see it (every control passed the monotonicity rule up to ρ = 1.000).

**The competitor, promoted to co-equal standing and tested FIRST** — because it is the cheapest, the
strongest, and the author's own favourite (§14 B7), which is exactly why it must be given the first
chance to win:

  H_mean-shift   a data-correlated direction `v` has `E⟨x, v⟩ ≠ 0`, so `+α` and `−α` inflate and
                 deflate `‖x + αv‖` asymmetrically. RMSNorm then converts that norm change into a
                 uniform gain on the token's content. If a PURE PER-TOKEN RESCALE — carrying no
                 direction information whatsoever — reproduces the collapse, then R18's
                 "direction-specific control surface" is a normalization artifact, and every
                 steering claim in the literature acquires a mandatory new control.

  A2(i)   gain-matched cell: `x → c_t(α)·x` with `c_t = ‖x_t + αv‖ / ‖x_t‖`, matched per token.
  A2(ii)  orthogonalized traversal: `v ⊥ h̄`, re-run. Removes the mean-shift cross-term at source.
  A2(iii) `⟨x_t, v⟩` telemetry logged per token — the quantity that makes H_mean-shift testable
          rather than rhetorical.

Surviving asymmetry is then routed by CURVE SHAPE (A1: a kink at α* ⇒ H_boundary; smooth odd-order
growth ⇒ H_cubic) and by the manifold proxy (A3: does damage track the perturbed point's dictionary
FVU?). Survival of all three ⇒ **H_direction-primitive**, the clean version of lead 1.

**Gate-0a makes H_mean-shift a priori strong at L0.** A single eigendirection carries 70.22 % of the
centered residual trace, and the uncentered top eigendirection aligns 0.9293 with the mean direction.
Any high-energy direction at this site is close to the mean direction, so `E⟨x, v⟩` is large by
construction. This is a prediction the arm can falsify, not a reason to skip it.

EXPLORATORY / DISCOVERY (Arm C already confirmed there is no compact handle). Flagged for a
confirmatory successor with sealed items if H_direction-primitive survives.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_a --type g6e.xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/bispec_a --budget-min 420 --args "--site L0"
"""
import argparse
import json
import os
import pathlib

# Amplitude in units of the MEAN RESIDUAL NORM, not of sqrt(lambda1).
# The first launch used alpha * sqrt(lambda1) with sqrt(lambda1)=88.8, so 3*sigma1 was a vector of
# norm 266 -- about 200% of the residual's typical norm. Every direction, including all three
# matched-norm random controls, drove VQA to exactly 0.0000 and Dppl to 1e120, and H_mean-shift
# fired on a dead grid. REGRAIN R18's +3sigma is norm 28.3 = 21% of the residual norm; the top rung
# here reproduces that.
# REGRAIN R18's own grid, in ITS OWN units: alpha * sigma_1 * v_1, where v_1 is PC1 of the
# B*-union's CONTRIBUTION cloud and sigma_1 = 9.4337 is the std of the cloud's projection on it
# (regrain_pipeB/out/f_steer/metrics.json::pca.sigma[0]). alpha=+3 gives a vector of norm 28.3.
ALPHAS = (-3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0)
TOP_ALPHA = 3.0
GAIN_FRAC_CEILING = 1.25       # a matched null must REPRODUCE the collapse, not exceed it
N_PCA_ROWS = 40000
N_RAND = 3
N_ITEMS = 160
GAIN_FRAC_THRESHOLD = 0.8       # frozen (spec §6)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    ap.add_argument("--cdisc-s3", required=True, help="regrain_c_disc/out — the B* atom list")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs.bispec_gate0 import bispec_split
    from infra.pg import bispec_core as BC
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC

    RS.install_pg()
    L = int(a.site[1:])
    rep = {"object": "bispec_a_v1", "site": a.site, "alphas_over_contribution_sigma1": list(ALPHAS),
           "label": "DISCOVERY / EXPLORATORY",
           "precedence": "H_mean-shift tested FIRST — cheapest, strongest, and the author's own "
                         "favourite (B7); it gets the first chance to win",
           "parent_observation": {"regrain_R18_dvqa_plus3sigma": -0.5733,
                                  "regrain_R18_dvqa_minus3sigma": -0.0317,
                                  "regrain_R18_rotated_same_vector": -0.0833},
           "cells": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    bank = torch.load(RS.fetch(f"{a.sigma_s3}/sigma_bank.pt", "/tmp/bispec/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)[a.site]
    Sigma, hbar, basis, spec = bank["Sigma"], bank["hbar"], bank["basis"], bank["spectrum"]

    sha, _ = RS.pinned_pg1_sha("/tmp/bispec")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/bispec/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = bispec_split(hashes, name="bispec_vqa")
    items = [rows[i] for i in sp["disc_idx"]][: a.n_items]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl, "dead_floor_vqa": floor,
                        "n_items": len(items)}
    _w()

    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/bispec/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    # f_steer built the contribution PCA on 40k rows; reproduce that sample size.
    X = torch.load(RS.fetch(f"{a.harvest_s3}/{shard0}", f"/tmp/bispec/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:N_PCA_ROWS].double()
    d = X.shape[1]

    # ---- R18's ACTUAL direction, rebuilt: PC1 of the B*-union's contribution cloud -------------
    # Two earlier launches traversed the wrong vector. The first used sqrt(lambda1)=88.8 as the
    # amplitude scale and saturated every cell including the random controls; the second corrected
    # the scale but still traversed e1, the top eigendirection of Sigma, where a 21%-of-norm
    # perturbation moves VQA by ~0.02. R18's v1 is PC1 of the CONTRIBUTION cloud of the B*-union --
    # a different vector in a different basis -- and its sigma1 = 9.4337 is the std of the cloud's
    # projection on it. Rebuilt here exactly as regrain_f_steer built it.
    cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/bispec/c_disc.json").read_text())
    bstar = [int(x) for x in cd["bstar"]]
    med_s = int(cd["median_seed"])
    assert int(cd["L"]) == L, f"B* lives at L{cd['L']}, not {a.site}"
    nm = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
    ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(src, f"/tmp/bispec/{nm}"))
    ae = ae.to("cuda").eval()
    Db = RC.decoder_matrix(ae)[:, bstar].float().cuda()
    with torch.inference_mode():
        xs = X[:N_PCA_ROWS].float().cuda()
        fcode = ae.encode(xs, use_threshold=True)
        contrib = fcode[:, bstar] @ Db.T
        fired = contrib.abs().sum(1) > 0
        Cc = contrib[fired].double()
        mu = Cc.mean(0)
        _U, S, Vh = torch.linalg.svd(Cc - mu, full_matrices=False)
        v1 = Vh[0].cpu().clone()
        sigma1 = float(S[0] / max(1, Cc.shape[0] - 1) ** 0.5)
    del xs, fcode, contrib, Cc, _U, S, Vh
    torch.cuda.empty_cache()
    xnorm = float(X.norm(dim=1).mean())
    rep["direction_source"] = {
        "object": "PC1 of the B*-union contribution cloud (regrain_f_steer construction, verbatim)",
        "bstar_len": len(bstar), "median_seed": med_s, "n_fired_rows": int(fired.sum()),
        "sigma1": sigma1, "regrain_f_steer_sigma1": 9.433703479082313,
        "top_alpha_vector_norm": TOP_ALPHA * sigma1,
        "regrain_R18_vector_norm_at_plus3sigma": 28.3,
        "mean_residual_norm": xnorm,
        "sqrt_lambda1_of_Sigma": float(spec[0].sqrt()),
        "cos_with_Sigma_e1": float(abs(v1 @ basis[:, 0]) / v1.norm()),
        "note": "the amplitude scale is sigma1 of the CONTRIBUTION coordinate, not sqrt(lambda1) of "
                "Sigma; the two differ by ~9.4x and the earlier runs saturated on the latter."}
    assert abs(sigma1 - 9.4337) < 1.5, f"sigma1 {sigma1} does not reproduce f_steer's 9.4337"
    # A2(iii): the cross-term telemetry that makes H_mean-shift testable
    tele = BC.cross_term_telemetry(X, v1)
    hb_dir = hbar / hbar.norm()
    rep["direction"] = {
        "sigma1": sigma1,
        "cross_term_mean": tele["mean"], "cross_term_std": tele["std"],
        "frac_positive_inner": tele["frac_positive"],
        "alignment_with_mean_direction": float(abs(v1 @ hb_dir)),
        "reading": "|E<x,v>| large ⇒ ±α inflate/deflate the token norm asymmetrically and RMSNorm "
                   "converts that into a gain asymmetry — the mean-shift mechanism, quantified"}
    _w()

    # v ⊥ h̄, the A2(ii) orthogonalized direction (re-normalized)
    v_orth = v1 - (v1 @ hb_dir) * hb_dir
    v_orth = v_orth / v_orth.norm()
    tele_o = BC.cross_term_telemetry(X, v_orth)
    rep["direction_orth"] = {"cross_term_mean": tele_o["mean"], "cross_term_std": tele_o["std"],
                             "frac_positive_inner": tele_o["frac_positive"],
                             "alignment_with_mean_direction": float(abs(v_orth @ hb_dir))}
    _w()

    rand_dirs = []
    for s in range(N_RAND):
        g = torch.Generator().manual_seed(4242 + s)
        r = torch.randn(d, generator=g, dtype=torch.float64)
        rand_dirs.append(r / r.norm())
    Qrot = RC.haar_rotation(d, RS.ROT_SEED, dtype=torch.float64)
    v_rot = Qrot @ v1

    def cell(family, direction_name, alpha, hook_fn, **extra):
        r = IP.eval_pg1(model, proc, vqa, items, "cuda", ctx=BC.hook_at(layers, L, hook_fn()))
        with BC.hook_at(layers, L, hook_fn()):
            p = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        rec = {"family": family, "direction": direction_name, "alpha": alpha,
               "vqa": r["score"], "dvqa": r["score"] - intact["score"],
               "effect": RS.effect_from(r["per_item"], intact["per_item"], floor),
               "ppl": p, "dppl": p - base_ppl, **extra}
        rep["cells"].append(rec)
        _w()
        print(f"[a] {family}/{direction_name} α={alpha:+.2f}: dVQA={rec['dvqa']:+.4f} "
              f"dppl={rec['dppl']:+.3f}", flush=True)
        return rec

    # ---- A1: the two-sign fine dose curve, PC1 + every control -----------------
    dirs = [("pc1", v1), ("pc1_orthogonalized", v_orth), ("pc1_rotated", v_rot)]
    dirs += [(f"rand{i}", r) for i, r in enumerate(rand_dirs)]
    for name, v in dirs:
        for al in ALPHAS:
            amp = al * sigma1
            hk, vv = BC.make_hook_traverse(v, amp)
            cell("A1_dose", name, al, lambda h=hk: h, amp_abs=abs(amp))

    # ---- A2(i): the gain-matched (mean-shift) null ----------------------------
    # c_t(α) = ‖x_t + α σ₁ v₁‖ / ‖x_t‖, computed from the banked rows and applied as a pure rescale.
    # No direction information survives this hook — only the per-token norm trajectory does.
    for al in ALPHAS:
        amp = al * sigma1
        gains = BC.gains_for_alpha(X, v1, amp)          # (n_rows,) — the empirical gain distribution
        gmean = float(gains.mean())

        def gfn(nrm, g=gmean):
            return torch.full_like(nrm, g)
        cell("A2_gain_matched", "gain", al, lambda g=gfn: BC.make_hook_gain_match(g),
             gain=gmean, note="pure per-token rescale at the mean realized gain; carries NO direction")

    # ---- adjudication ---------------------------------------------------------
    def dv(fam, dname, al):
        m = [c["dvqa"] for c in rep["cells"] if c["family"] == fam and c["direction"] == dname
             and abs(c["alpha"] - al) < 1e-9]
        return m[0] if m else float("nan")

    pc1_plus, pc1_minus = dv("A1_dose", "pc1", TOP_ALPHA), dv("A1_dose", "pc1", -TOP_ALPHA)
    gain_plus = dv("A2_gain_matched", "gain", TOP_ALPHA)
    rand_plus = [dv("A1_dose", f"rand{i}", TOP_ALPHA) for i in range(N_RAND)]
    rand_minus = [dv("A1_dose", f"rand{i}", -TOP_ALPHA) for i in range(N_RAND)]
    rand_asym = [abs(p - m) for p, m in zip(rand_plus, rand_minus)]
    rand_asym_hi = float(np.max(rand_asym))

    collapse = abs(pc1_plus)
    gain_frac = abs(gain_plus) / collapse if collapse > 1e-9 else 0.0
    orth_asym = abs(dv("A1_dose", "pc1_orthogonalized", TOP_ALPHA)
                    - dv("A1_dose", "pc1_orthogonalized", -TOP_ALPHA))

    # saturation guards: a cell at the dead-model floor carries no direction information, and a
    # CONTROL at the floor makes the "inside the random band" clause vacuous.
    dead = lambda x: (intact["score"] + x) <= floor + 0.02 * (intact["score"] - floor)
    n_sat = sum(1 for c in rep["cells"] if dead(c["dvqa"]))
    controls_sat = all(dead(x) for x in rand_plus + rand_minus)
    grid_sat = n_sat > 0.5 * len(rep["cells"])
    rep["saturation"] = {"n_cells_at_floor": n_sat, "n_cells": len(rep["cells"]),
                         "controls_saturated": bool(controls_sat), "grid_saturated": bool(grid_sat),
                         "floor": floor, "intact": intact["score"]}

    # curve shape: kink score (second difference at the largest |α|) vs cubic score (odd-order fit)
    als = np.array(sorted(ALPHAS))
    ys = np.array([dv("A1_dose", "pc1", x) for x in als])
    odd = 0.5 * (ys - ys[::-1])                      # the odd part of the dose curve
    cubic_fit = np.polyfit(als, odd, 3)
    cubic_score = float(abs(cubic_fit[0]) * (als.max() ** 3) / (abs(odd).max() + 1e-12))
    d2 = np.abs(np.diff(ys, n=2))
    kink_score = float(d2.max() / (abs(ys).max() + 1e-12))

    rep["asymmetry"] = {
        "pc1_dvqa_plus3": pc1_plus, "pc1_dvqa_minus3": pc1_minus,
        "gain_matched_dvqa_plus3": gain_plus, "gain_reproduces_fraction": gain_frac,
        "gain_frac_threshold": GAIN_FRAC_THRESHOLD,
        "orthogonalized_asymmetry": orth_asym, "random_asymmetry_hi": rand_asym_hi,
        "rotated_dvqa_plus3": dv("A1_dose", "pc1_rotated", 3.0),
        "kink_score": kink_score, "cubic_score": cubic_score}
    rep["adjudication"] = BC.adjudicate_asymmetry(
        gain_frac=gain_frac, gain_frac_threshold=GAIN_FRAC_THRESHOLD,
        gain_frac_ceiling=GAIN_FRAC_CEILING,
        orth_asym=orth_asym, rand_asym_hi=rand_asym_hi,
        kink_score=kink_score, cubic_score=cubic_score,
        proxy_rho=0.0, proxy_rho_cilo=-1.0,          # A3 proxy folded in by bispec_figures if run
        grid_saturated=grid_sat, controls_saturated=controls_sat)
    _w()

    print("BISPEC_A_OK", json.dumps({"verdict": rep["adjudication"]["verdict"],
                                     "gain_frac": gain_frac, "orth_asym": orth_asym,
                                     "rand_asym_hi": rand_asym_hi}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
