"""W15 §9.2 FAIL-FAST PREFLIGHT (~0.3 GPU-h). Reproduce cached numbers + validate the NEW
instrument end-to-end BEFORE the freeze (§9.3) and the ~4 GPU-h Stage-0 arms. Any red gate
HALTs (raise; no try/except swallowing).

Gates (each fail-loud):
  G1  swap gate           build_pair("base-pt"): decoder Δ>0, embed/lm_head/vision/proj PG-identical,
                          forward finite (the BEFORE/AFTER bridge the whole gate stands on).
  G2  fp64 instrument     on REAL matrices (no GPU math): w15_carrier.recon_residual(ΔW, W_before SVD)
                          < 1e-10; graft_from_indices(energy idx) ≡ the cached rank-r SVD recon
                          (< 1e-6); gain_routing planted gain/routing/out-of-span recovered to 1e-6;
                          w15 plant_disjoint→O=0 / plant_identical→O=1.
  G3  energy cliff (a)    reproduce comparator (a): the ENERGY rank-512 no-retrain graft installs
                          R_cap ≈ 0 / emit ≈ 0 on small-n RefCOCO testA (CEILING=PG2-pt, FLOOR=base).
                          This is the design premise — if energy@512 installs here, the premise is void.
  G4  forgetting (e)      reproduce the W11-A direction: blank-image MMLU(BEFORE=base-decoder) >
                          MMLU(AFTER=PG2-pt) on small n (H_forget genuine in-distribution).
  G5  backward harness    the genuinely-new code: refcoco_ce backward → finite, non-zero per-matrix
                          G_install; w15_carrier.direction_scores(G, U, V) finite. De-risks §9.4/§9.5.

    pg run infra.aws_pg.jobs.w15_preflight --type g6e.8xlarge,p5.4xlarge \
        --s3-prefix results/w15_preflight --budget-min 45
    # wiring-only:  --args "--smoke"
"""
from __future__ import annotations

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rstar", type=int, default=512, help="cliff rank r* under test (comparator a/c)")
    ap.add_argument("--eval-n", type=int, default=300, help="small-n RefCOCO testA for the cliff gate")
    ap.add_argument("--mmlu-per-subject", type=int, default=2, help="small-n blank-image MMLU per subject")
    ap.add_argument("--grad-items", type=int, default=8, help="RefCOCO install items for the backward smoke")
    ap.add_argument("--recon-tol", type=float, default=1e-10)
    ap.add_argument("--ctrl-tol", type=float, default=1e-6)
    ap.add_argument("--cliff-tol", type=float, default=0.05, help="energy R_cap/emit must be below this")
    ap.add_argument("--forget-margin", type=float, default=0.05,
                    help="G4: blank-MMLU BEFORE must exceed AFTER by this (directional W11-A reproduction)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg import diff_harvest as H
    H.hf_online_setup()                                    # BEFORE transformers/datasets imports

    import torch
    from infra.pg import battery_tasks as B
    from infra.pg import gain_routing as GR
    from infra.pg import text_eval as TE
    from infra.pg import w15_carrier as C
    from infra.aws_pg.jobs import _w15_shared as S

    assert torch.cuda.is_available(), "W15 preflight needs a GPU (generation + backward)"
    device = "cuda"
    if a.smoke:
        a.eval_n, a.mmlu_per_subject, a.grad_items = 32, 1, 4
    print(f"[w15-pre] gpu={torch.cuda.get_device_name(0)} torch={torch.__version__} "
          f"r*={a.rstar} eval_n={a.eval_n} smoke={a.smoke}", flush=True)

    res = {"experiment": "w15_preflight", "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0), "config": vars(a)}

    # ---- preload eval rows + the split audit BEFORE the model build (eq1 order) ----
    task = B.REFCOCO_TASK
    eval_rows = {"refcoco": S._eval_rows(task, a.eval_n, S.SEED)}
    eval_hashes = {S.img_hash(task.image_fn(r)) for r in eval_rows["refcoco"]}
    install_items, install_hashes = S.refcoco_install_items(B, max(64, a.grad_items * 4),
                                                            S.SEED, eval_hashes)
    S.split_audit(B, install_hashes, eval_rows, out)

    # =====================================================================
    # G1 — the BEFORE/AFTER swap gate.
    # =====================================================================
    pg_after, pg_before, proc, paths, _img, gate = H.build_pair("base-pt", device)
    g1 = bool(gate.get("forward_finite_before") and gate.get("forward_finite_after")
              and float(gate.get("decoder_max_abs_delta_vs_PG", 0.0)) > 0.0
              and gate.get("embed_lm_head_kept_identical") and gate.get("vision_tower_PG_identical")
              and gate.get("projector_PG_identical"))
    res["G1_swap_gate"] = {"gate": gate, "ok": g1}
    print(f"[w15-pre][G1] swap gate ok={g1}", flush=True)

    lm_before = pg_before.get_submodule(paths["language_model"])
    lm_after = pg_after.get_submodule(paths["language_model"])
    dkeys = S._decoder_keys()
    base_sd = {k: v.detach().clone() for k, v in lm_before.state_dict().items()}

    # =====================================================================
    # G2 — fp64 instrument on REAL matrices + planted controls (CPU math).
    # =====================================================================
    probe = "layers.0.self_attn.q_proj.weight"
    Wb = base_sd[probe].detach().double().cpu()
    Wa = lm_after.state_dict()[probe].detach().double().cpu()
    dW = Wa - Wb
    U, s, V = C.svd_fp64(Wb)
    recon = C.recon_residual(dW, U, V)
    # selection graft on the ENERGY ordering ≡ cached rank-r SVD reconstruction
    svd_probe = S.cached_dw_svd(lm_after, lm_before, [probe], device, a.rstar)
    Up, Sp, Vp = svd_probe[probe]
    idx = C.energy_indices(a.rstar, Sp.shape[0])
    graft = C.graft_from_indices(Wb, Up, Sp, Vp, idx)
    rankk = Wb + (Up[:, :idx.numel()].double() * Sp[:idx.numel()].double()) @ Vp[:, :idx.numel()].double().T
    graft_match = float((graft - rankk).norm().item())
    # gain_routing planted controls (rectangular probe → out-of-span valid)
    gen = torch.Generator().manual_seed(S.SEED)
    cg = GR.decompose(Wb, Wb + GR.plant_gain(Wb, gen))
    cr = GR.decompose(Wb, Wb + GR.plant_routing(Wb, gen))
    co = GR.decompose(Wb, Wb + GR.plant_out_of_span(Wb, gen))
    # w15 overlap planted controls
    Ia1, Ib1, k1 = C.plant_identical((32, 32), k=50, gen=torch.Generator().manual_seed(1))
    Ia0, Ib0, k0 = C.plant_disjoint((32, 32), k=50, gen=torch.Generator().manual_seed(2))
    o_ident = C.overlap_maps(Ia1, Ib1, k1)
    o_disj = C.overlap_maps(Ia0, Ib0, k0)
    g2 = bool(recon < a.recon_tol and graft_match < 1e-6
              and abs(cg["gain_frac"] - 1.0) < a.ctrl_tol and abs(cr["route_frac"] - 1.0) < a.ctrl_tol
              and abs(co["f_perp"] - 1.0) < a.ctrl_tol
              and abs(o_ident - 1.0) < 1e-6 and abs(o_disj) < 1e-6)
    res["G2_fp64_instrument"] = {"probe": probe, "recon_residual": recon, "graft_vs_rankk": graft_match,
                                 "plant_gain_frac": cg["gain_frac"], "plant_route_frac": cr["route_frac"],
                                 "plant_oos_f_perp": co["f_perp"], "O_identical": o_ident,
                                 "O_disjoint": o_disj, "ok": g2}
    print(f"[w15-pre][G2] recon={recon:.2e} graft≡rankk={graft_match:.2e} "
          f"plant(gain={cg['gain_frac']:.6f},route={cr['route_frac']:.6f},oos={co['f_perp']:.6f}) "
          f"O(id={o_ident:.3f},disj={o_disj:.3f}) ok={g2}", flush=True)

    # =====================================================================
    # G3 — reproduce the energy-order R_cap ≈ 0 cliff (comparator a) on small-n RefCOCO.
    # =====================================================================
    rows = eval_rows["refcoco"]
    mf, ef, _ = S.score_refcoco(pg_before, proc, task, rows, device, 8)     # FLOOR (base decoder)
    mc, ec, _ = S.score_refcoco(pg_after, proc, task, rows, device, 8)      # CEILING (PG2-pt)
    svd_all = S.cached_dw_svd(lm_after, lm_before, dkeys, device, a.rstar)
    energy_override = S.selection_graft(base_sd, svd_all, S.energy_idx_by_key(svd_all, dkeys, a.rstar),
                                        dkeys, device)
    S._set_decoder(lm_before, base_sd, energy_override, device)
    me, ee, samp = S.score_refcoco(pg_before, proc, task, rows, device, 8)
    denom = mc - mf
    rcap_energy = (me - mf) / denom if denom > 0 else float("nan")
    S._set_decoder(lm_before, base_sd, {}, device)                          # restore base decoder
    g3 = bool(denom > 0 and rcap_energy < a.cliff_tol and ee < a.cliff_tol)
    res["G3_energy_cliff"] = {"floor": mf, "ceiling": mc, "denom": denom,
                              "energy_rstar_metric": me, "energy_rstar_emit": ee,
                              "energy_rstar_R_cap": rcap_energy, "rstar": a.rstar,
                              "cliff_tol": a.cliff_tol, "samples": samp, "ok": g3}
    print(f"[w15-pre][G3] FLOOR={mf:.4f} CEILING={mc:.4f} energy@{a.rstar} "
          f"R_cap={rcap_energy:.4f} emit={ee:.4f} (premise: ≈0) ok={g3}", flush=True)

    # =====================================================================
    # G4 — reproduce the W11-A forgetting direction (blank-image MMLU BEFORE > AFTER).
    # =====================================================================
    mmlu_before = TE.mmlu_blankimg(pg_before, proc, device, n_per_subject=a.mmlu_per_subject)["acc"]
    mmlu_after = TE.mmlu_blankimg(pg_after, proc, device, n_per_subject=a.mmlu_per_subject)["acc"]
    # Directional reproduction of the W11-A forgetting signal: the base decoder retains MMLU, PG2-pt
    # forgot it (BEFORE > AFTER). The ABSOLUTE level is an n-dependent quantity (W11-A: 0.49 vs 0.24
    # on the full 14k set); at the small preflight n the level is noisy, so the gate is the GAP, not a
    # floor on BEFORE. Margin 0.05 ≪ the ~0.24 full-n gap, robust to small-n sampling noise.
    g4 = bool(mmlu_before > mmlu_after + a.forget_margin)
    res["G4_forgetting"] = {"mmlu_blank_before": mmlu_before, "mmlu_blank_after": mmlu_after,
                            "gap": mmlu_before - mmlu_after, "forget_margin": a.forget_margin,
                            "n_per_subject": a.mmlu_per_subject, "ok": g4,
                            "note": "BEFORE=base-decoder retains MMLU, AFTER=PG2-pt forgot it; "
                                    "W11-A cached blank-image ~0.49 vs ~0.24 (gap ~0.24 on full n)"}
    print(f"[w15-pre][G4] blank-MMLU BEFORE={mmlu_before:.4f} > AFTER={mmlu_after:.4f} "
          f"gap={mmlu_before - mmlu_after:+.4f} (margin {a.forget_margin}) ok={g4}", flush=True)

    # =====================================================================
    # G5 — the NEW backward harness: ∂(RefCOCO <loc> CE)/∂W finite + direction_scores runs.
    # =====================================================================
    grad_items = install_items[:a.grad_items]
    batches = S._refcoco_batches(B, proc, grad_items, train_batch=2)
    grads, gloss = S.decoder_weight_grad(pg_before, paths, dkeys, batches, S.refcoco_ce, device)
    gfin = all(torch.isfinite(g).all() for g in grads.values())
    gnz = sum(float(g.norm()) > 0 for g in grads.values())
    # direction_scores on the probe matrix (leverage smoke)
    scores = C.direction_scores(grads[probe], Up.double(), Vp.double())
    sfin = bool(torch.isfinite(scores).all() and scores.numel() == Sp.shape[0])
    g5 = bool(gfin and gnz == len(dkeys) and sfin and (gloss == gloss))   # gloss==gloss → not NaN
    res["G5_backward_harness"] = {"mean_ce_loss": gloss, "grads_finite": bool(gfin),
                                  "matrices_with_nonzero_grad": gnz, "n_matrices": len(dkeys),
                                  "direction_scores_finite": sfin, "ok": g5}
    print(f"[w15-pre][G5] refcoco_ce={gloss:.4f} grads_finite={gfin} nonzero={gnz}/{len(dkeys)} "
          f"direction_scores_ok={sfin} ok={g5}", flush=True)

    ok = bool(g1 and g2 and g3 and g4 and g5)
    res["PREFLIGHT_OK"] = ok
    (out / "preflight.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"[w15-pre] PREFLIGHT_OK={ok} (G1={g1} G2={g2} G3={g3} G4={g4} G5={g5})", flush=True)
    print("W15_PREFLIGHT", json.dumps({"ok": ok, "energy_rcap": rcap_energy,
                                       "mmlu_before": mmlu_before, "mmlu_after": mmlu_after}), flush=True)
    if not ok:
        raise RuntimeError(f"W15 PREFLIGHT HALT: G1={g1} G2={g2} G3={g3} G4={g4} G5={g5}. "
                           f"See {out / 'preflight.json'}.")


if __name__ == "__main__":
    main()
