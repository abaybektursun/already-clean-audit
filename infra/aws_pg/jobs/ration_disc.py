"""RATION discovery (GPU) — spec §9 steps 1+3: preflight → Σ_x → sink → oracle range guard →
discovery rate-distortion sweep → FREEZE (δ_range, b̄*_ppl, b̄*_vqa, sink set, oracle_w, banks). L*=L0
(A-RATION-1). Writes ration_disc.json + ration_banks.pt to S3. Confirmation reads the frozen
selections from here; the analyst logs predictions BEFORE launching confirmation (§11).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_disc --type p5.4xlarge,g6e.8xlarge \
      --s3-prefix results/ration_disc --budget-min 100
"""
import argparse
import json
import os
import pathlib
import time


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                    # BEFORE transformers/datasets (VQA streaming)
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _ration_shared as RS
    from infra.pg import inherit_pg1 as IP
    from infra.pg import ration_core as RC

    rep = {"object": "ration_disc_v1", "L_star": RS.L_STAR, "budget_grid": list(RC.BUDGET_GRID),
           "massive_coords": list(RC.MASSIVE_COORDS), "k_sigma": RC.K_SIGMA, "b_max": RC.B_MAX}
    t00 = time.time()

    def dump():
        (out / "ration_disc.json").write_text(json.dumps(rep, indent=2, default=float))

    def stage(msg):
        print(f"[ration-disc] {msg} (+{time.time() - t00:.0f}s)", flush=True)

    # ---- load ----
    model, proc, layers, sha, base_ppl = RS.load_model()
    rep["pinned_sha"], rep["base_ppl_anchor"] = sha, base_ppl
    stage(f"loaded PG1 eager sha={sha[:8]} base_ppl_anchor={base_ppl:.5f}")

    # ---- STAGE 0: preflight (Principle #12) ----
    # n_seq=120 EXACTLY reproduces the invcore-FUNC anchor instrument (a.n_ppl_seq default=120);
    # a smaller n_seq scores different windows and lands off-anchor (caught: n_seq=64 → 21.36).
    clean_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=120)["ppl"]
    rep["preflight"] = {"clean_ppl_gate": clean_ppl, "base_ppl_anchor": base_ppl,
                        "ppl_abs_err": abs(clean_ppl - base_ppl)}
    assert abs(clean_ppl - base_ppl) < 0.5, f"eager forward off-anchor: {clean_ppl} vs {base_ppl}"
    stage(f"preflight ppl {clean_ppl:.5f} (anchor {base_ppl:.5f}, |Δ|={abs(clean_ppl-base_ppl):.4f})")

    # ---- STAGE 1: splits ----
    ppl_disc, ppl_conf = RS.wikitext_split(proc)
    task, vqa_disc, vqa_conf = RS.vqa_task_and_split()
    rep["n"] = {"ppl_disc": len(ppl_disc), "ppl_conf": len(ppl_conf),
                "vqa_disc": len(vqa_disc), "vqa_conf": len(vqa_conf)}
    stage(f"splits ppl {rep['n']['ppl_disc']}/{rep['n']['ppl_conf']} vqa "
          f"{rep['n']['vqa_disc']}/{rep['n']['vqa_conf']}")

    # passthrough transparency + VQA clean range (Principle #12 continued)
    sub_install, sub_holder = RC.make_residual_substitutor(layers, RS.L_STAR)
    sub_holder["spec"] = None
    clean_vqa = RS.vqa_per_item(model, proc, task, vqa_disc, sub_install, n=RS.N_VQA_CELL)
    nohook_vqa = IP.eval_pg1(model, proc, task, vqa_disc[:RS.N_VQA_CELL], "cuda")["per_item"]
    assert clean_vqa == nohook_vqa, "passthrough hook is NOT transparent on VQA (identity broken)"
    clean_ppl_pw, _ = RS.ppl_per_window(model, proc, ppl_disc, sub_install, sub_holder, collect_mse=False)
    clean_vqa_acc = float(np.mean(clean_vqa))
    clean_ppl_disc = float(np.exp(sum(s for s, _ in clean_ppl_pw) / sum(n for _, n in clean_ppl_pw)))
    vqa_has_range = clean_vqa_acc > 0.30       # NON-fatal: a weak base VQA lets the VQA leg VOID at
    rep["preflight"].update({"passthrough_transparent": True, "clean_vqa_acc": clean_vqa_acc,
                             "vqa_has_range": bool(vqa_has_range),   # the range guard, not kill the ppl MDE
                             "clean_ppl_disc_windows": clean_ppl_disc})
    stage(f"preflight passthrough transparent · clean_vqa={clean_vqa_acc:.4f} "
          f"({'range OK' if vqa_has_range else 'WEAK — VQA leg may VOID'}) · "
          f"clean_ppl(disc-windows)={clean_ppl_disc:.4f}")
    dump()

    # ---- STAGE 2: Σ_x + banks (MIX calibration: ppl windows + real-image VQA items) ----
    sig = RS.accumulate_sigma(model, proc, layers, ppl_disc, vqa_rows=vqa_disc, task=task)
    banks = RS.build_banks(RS.load_q_bank("/tmp/ration"), sig)
    rep["sigma_n_rows"] = sig["n"]
    rep["sigma_n_ppl"], rep["sigma_n_vqa"] = sig["n_ppl"], sig["n_vqa"]
    stage(f"Σ_x accumulated over {sig['n']} rows at L0 (ppl {sig['n_ppl']} + vqa {sig['n_vqa']})")

    # ---- STAGE 3: sink detection ----
    sink_dirs = RS.detect_sink(banks)
    banks["sink_dirs"] = sink_dirs
    rep["sink_dirs"] = [int(x) for x in sink_dirs]
    rep["n_sink_dirs"] = len(sink_dirs)
    stage(f"sink-carrying Σ_x eigendirections: {len(sink_dirs)} ({sink_dirs[:8]})")

    # ---- STAGE 4: oracle band leave-one-out (ppl, N_ORACLE_BANDS) ----
    orc = RS.oracle_band_damage(banks, model, proc, task, ppl_disc, None, layers)   # ppl-only
    band_dppl = [b["dppl_nll"] for b in orc["bands"]]
    banks["oracle_w"] = RC.oracle_importance(band_dppl, RS.D)
    rep["oracle"] = {"intact_nll": orc["intact_nll"], "band_dppl_nll": band_dppl,
                     "n_bands": len(orc["bands"])}
    stage(f"oracle LOO: {len(orc['bands'])} bands, max band Δnll={max(band_dppl):+.4f}")
    dump()

    # ---- STAGE 5: discovery rate-distortion sweep (7 policies × 5 budgets; ppl+vqa+mse) ----
    def loss_ppl(policy_pw):
        g_clean = RC.ppl_goodness(clean_ppl_pw)
        g_pol = RC.ppl_goodness(policy_pw)
        return RC.loss_ci(g_clean, g_pol)               # mean(clean − policy) goodness = damage

    def loss_vqa(policy_vqa):
        return RC.loss_ci([float(x) for x in clean_vqa], [float(x) for x in policy_vqa])

    sweep = {}
    for policy in RC.POLICIES:
        basis = RS.precompute_basis_dev(policy, banks)
        sweep[policy] = {}
        for b in RC.BUDGET_GRID:
            cell = RS.run_cell(policy, b, banks, model, proc, task, ppl_disc, vqa_disc, layers,
                               precomputed_basis=basis, n_vqa=RS.N_VQA_CELL)
            lp, lv = loss_ppl(cell["ppl_per_window"]), loss_vqa(cell["vqa_per_item"])
            sweep[policy][str(b)] = {"ppl_loss": lp, "vqa_loss": lv, "mse": cell["mse"],
                                     "vqa_acc": float(np.mean(cell["vqa_per_item"]))}
            stage(f"  {policy}@{b}b ppl_loss={lp['E']:+.4f} vqa_loss={lv['E']:+.4f} mse={cell['mse']:.4g}")
        dump()
    rep["discovery_sweep"] = sweep

    # ---- STAGE 6: FREEZE selections (A-RATION-5) ----
    # δ_range = MULT × SE of the uniform loss at the FINEST budget (the measurement noise floor).
    dr_ppl = RC.DELTA_RANGE_MULT * sweep["uniform"]["8"]["ppl_loss"]["se"]
    dr_vqa = RC.DELTA_RANGE_MULT * sweep["uniform"]["8"]["vqa_loss"]["se"]

    def headroom(readout, b):                             # POSITIVE when the oracle beats uniform
        return (sweep["uniform"][str(b)][f"{readout}_loss"]["E"]
                - sweep["oracle"][str(b)][f"{readout}_loss"]["E"])

    def pick_bbar(readout, dr):
        # b̄* := the budget with the LARGEST oracle-over-uniform headroom AMONG budgets with a
        # measurable uniform loss (off the benign plateau) — the budget where the allocation axis has
        # the most leverage (the spec's Arm-O criterion, repurposed to budget since L=L0 is fixed).
        cands = [(headroom(readout, b), b) for b in RC.BUDGET_GRID
                 if sweep["uniform"][str(b)][f"{readout}_loss"]["E"] >= dr]
        return max(cands)[1] if cands else None
    bstar_ppl, bstar_vqa = pick_bbar("ppl", dr_ppl), pick_bbar("vqa", dr_vqa)
    head_ppl = headroom("ppl", bstar_ppl) if bstar_ppl is not None else None
    head_vqa = headroom("vqa", bstar_vqa) if bstar_vqa is not None else None
    rep["freeze"] = {"L_star": RS.L_STAR, "delta_range_ppl": dr_ppl, "delta_range_vqa": dr_vqa,
                     "bstar_ppl": bstar_ppl, "bstar_vqa": bstar_vqa,
                     "oracle_headroom_ppl_at_bstar": head_ppl, "oracle_headroom_vqa_at_bstar": head_vqa,
                     "sink_dirs": rep["sink_dirs"], "n_sink_dirs": rep["n_sink_dirs"]}
    stage(f"FREEZE δ_range ppl={dr_ppl:.4f}/vqa={dr_vqa:.4f} · b̄*_ppl={bstar_ppl} b̄*_vqa={bstar_vqa} "
          f"· headroom ppl={head_ppl}/vqa={head_vqa}")

    # ---- bank the frozen ingredients (confirmation loads THIS) ----
    torch.save({"Q_evecs": banks["Q_evecs"], "Q_evals": banks["Q_evals"],
                "sigma_evecs": banks["sigma_evecs"], "sigma_evals": banks["sigma_evals"],
                "mean_x": banks["mean_x"], "C_x": banks["C_x"], "absmean_x": banks["absmean_x"],
                "sink_dirs": banks["sink_dirs"], "oracle_w": banks["oracle_w"], "seed": banks["seed"],
                "freeze": rep["freeze"], "pinned_sha": sha}, out / "ration_banks.pt")
    rep["banks_saved"] = True
    rep["wall_seconds"] = round(time.time() - t00, 1)
    dump()
    print("RATION_DISC_OK", json.dumps({"bstar_ppl": bstar_ppl, "bstar_vqa": bstar_vqa,
                                        "n_sink": rep["n_sink_dirs"]}))


if __name__ == "__main__":
    main()
