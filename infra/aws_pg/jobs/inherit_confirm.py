"""INHERIT §9.6 sealed confirmation — UNBLIND ONCE on the sha1-sealed half. The promoted set is
the frozen {G-primary (k*, L_primary), G-secondary (k*, L_secondary)} (BH later joins O and
P-causal at closeout; this job records the raw one-sided bootstrap p per cell and a conservative
p×4 bound). Full RF(k) curves at both layers (all swept points, all component curves — figure
discipline), 3 em-random seeds (band, not best), anchors, floor, identity. COCO CIDEr runs as the
secondary task on the sealed half at the promoted cells (sign-agreement rule §7).

H_inherited iff a promoted cell's sealed gap ci-lo > δ_G (BH-corrected) AND RF(k) monotone
non-increasing at that layer; H_constructed iff transport gate green and no promoted cell clears.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_confirm --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/inherit_confirm --budget-min 300 \
      --args "--controls-s3 ... --targets-s3 ... --disc-s3 ..."
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--targets-s3", required=True)
    ap.add_argument("--disc-s3", required=True)
    ap.add_argument("--full-bank-s3", required=True, help="re-bank full Gram (saction_redraw arm)")
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--n-coco", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _inherit_shared as SH
    from infra.pg import inherit_core as IC
    from infra.pg import inherit_pg1 as IP

    SH.install_pg()
    ctrl = json.loads(SH.fetch(a.controls_s3, "/tmp/inherit/controls.json").read_text())
    disc_rep = json.loads(SH.fetch(a.disc_s3, "/tmp/inherit/disc_metrics.json").read_text())
    assert ctrl.get("green") and disc_rep.get("green"), "upstream stages not green"
    assert ctrl["transport_gate"]["green"], "transport gate RED — sealed Arm-G must not run"
    tg = torch.load(SH.fetch(a.targets_s3, "/tmp/inherit/pg1_targets.pt"),
                    map_location="cpu", weights_only=False)
    hbars = tg["hbars"]
    Xs = {L: v.double() for L, v in tg["Xs"].items()}
    bases, _h, _t = SH.load_bank()
    redraw = SH.load_redraw_bases(a.full_bank_s3)          # independent 2nd draw (robustness leg)
    sel = ctrl["selection"]
    thr = ctrl["frozen_thresholds"]
    k_grid, k_star = sel["k_grid"], sel["k_star"]
    floor = thr["floor"]
    delta_G = thr["delta_G"]
    n_items = a.n_items or int(ctrl["n_items"])
    promoted = disc_rep["promoted_sealed_set"]
    rep = {"selection": sel, "thresholds": thr, "n_items": n_items, "promoted": promoted}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    vqa, coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, n_items * 2)
    split = IC.sha1_split(hashes, name="inherit_vqa")
    sealed = [rows[i] for i in split["conf_idx"]][:n_items]
    rep["n_sealed_used"] = len(sealed)

    intact = SH.eval_arm(model, proc, layers, vqa, sealed, "identity", sel["primary"], 0, None)
    flr = SH.eval_arm(model, proc, layers, vqa, sealed, "floor", sel["primary"], 0, None, seed=a.seed)
    ii = np.asarray(intact["per_item"], float)
    rep["intact_sealed"] = {"score": intact["score"], "ci": intact["ci95"]}
    rep["floor_sealed"] = {"score": flr["score"], "ci": flr["ci95"],
                           "note": "reported; RF uses the FROZEN controls floor"}
    _w()

    per_item_bank = {"identity": intact["per_item"], "floor": flr["per_item"]}
    curves, pvals = {}, {}
    for L in (sel["primary"], sel["secondary"]):
        for k in k_grid:
            sa = SH.eval_arm(model, proc, layers, vqa, sealed, "saction", L, k, hbars[L],
                             Q=SH.build_arm_Q("saction", L, k, bases, seed=0))
            ems = []
            for s in range(3):
                em = SH.eval_arm(model, proc, layers, vqa, sealed, "random_energy", L, k, hbars[L],
                                 Q=SH.build_arm_Q("random_energy", L, k, bases, seed=200 + s,
                                                  X=Xs[L], hbar=hbars[L]))
                ems.append(np.asarray(em["per_item"], float))
                per_item_bank[f"em_L{L}_k{k}_s{s}"] = em["per_item"]
            per_item_bank[f"sa_L{L}_k{k}"] = sa["per_item"]
            em_mean = np.mean(np.stack(ems, 0), 0)          # per-item mean over the 3 draws
            rd = SH.eval_arm(model, proc, layers, vqa, sealed, "saction_redraw", L, k, hbars[L],
                             Q=SH.build_arm_Q("saction_redraw", L, k, bases, seed=0, redraw_bases=redraw))
            per_item_bank[f"rd_L{L}_k{k}"] = rd["per_item"]
            cell = {"saction": IC.rf_item_bootstrap(np.asarray(sa["per_item"], float), ii, floor),
                    "saction_redraw": IC.rf_item_bootstrap(np.asarray(rd["per_item"], float), ii, floor),
                    "random_energy_band": [IC.rf_item_bootstrap(e, ii, floor)["rf"] for e in ems],
                    "gap": IC.gap_item_bootstrap(np.asarray(sa["per_item"], float), em_mean, ii, floor),
                    "gap_redraw": IC.gap_item_bootstrap(np.asarray(rd["per_item"], float), em_mean,
                                                        ii, floor)}
            p = IC.boot_p_gt(cell["gap"]["boots"], delta_G)
            cell["gap"].pop("boots")
            cell["gap_redraw"].pop("boots")
            cell["p_gap_gt_delta"] = p
            curves[f"L{L}_k{k}"] = cell
            if k == k_star:
                pvals[f"G_{'primary' if L == sel['primary'] else 'secondary'}"] = p
            _w()
    rep["curves_sealed"] = curves
    for L in sel["anchors"]:
        sa = SH.eval_arm(model, proc, layers, vqa, sealed, "saction", L, k_star, hbars[L],
                         Q=SH.build_arm_Q("saction", L, k_star, bases, seed=0))
        r = IC.rf_item_bootstrap(np.asarray(sa["per_item"], float), ii, floor)
        rep[f"anchor_L{L}_sealed"] = {"rf": r["rf"], "ci": r["ci"],
                                      "tripped": bool(r["ci"][1] < thr["tau_high"])}
    torch.save(per_item_bank, out / "per_item_sealed.pt")
    _w()

    # --- COCO secondary (promoted cells only; sign agreement) ---
    crows, chashes = IP.load_items(coco, a.n_coco * 2)
    csplit = IC.sha1_split(chashes, name="inherit_coco")
    csealed = [crows[i] for i in csplit["conf_idx"]][: a.n_coco]
    coco_cells = {"identity": SH.eval_arm(model, proc, layers, coco, csealed, "identity",
                                          sel["primary"], 0, None)}
    for L in (sel["primary"], sel["secondary"]):
        coco_cells[f"sa_L{L}"] = SH.eval_arm(model, proc, layers, coco, csealed, "saction", L,
                                             k_star, hbars[L],
                                             Q=SH.build_arm_Q("saction", L, k_star, bases, seed=0))
        coco_cells[f"em_L{L}"] = SH.eval_arm(model, proc, layers, coco, csealed, "random_energy",
                                             L, k_star, hbars[L],
                                             Q=SH.build_arm_Q("random_energy", L, k_star, bases,
                                                              seed=200, X=Xs[L], hbar=hbars[L]))
    rep["coco_sealed"] = {kn: {"score": v["score"], "ci": v["ci95"]} for kn, v in coco_cells.items()}
    _w()

    # --- frozen verdict (BH here is the conservative p×4 bound; exact BH at closeout with O/P) ---
    verdicts = {}
    for cell in promoted:
        L, k = cell["L"], cell["k"]
        key = f"L{L}_k{k}"
        gap_lo = curves[key]["gap"]["ci_lo"]
        ks = [kk for kk in k_grid if kk <= k]
        mono = all(curves[f"L{L}_k{ks[i]}"]["saction"]["rf"]
                   >= curves[f"L{L}_k{ks[i + 1]}"]["saction"]["rf"] - 0.05
                   for i in range(len(ks) - 1))
        p = curves[key]["p_gap_gt_delta"]
        verdicts[cell["cell"]] = {"gap_ci_lo": gap_lo, "clears_delta": bool(gap_lo > delta_G),
                                  "monotone_ok": mono, "p": p, "p_x4_bound": min(1.0, p * 4)}
    any_clear = any(v["clears_delta"] and v["monotone_ok"] and v["p_x4_bound"] < 0.05
                    for v in verdicts.values())
    rep["verdict"] = {"cells": verdicts,
                      "arm_g": "H_inherited" if any_clear else "H_constructed",
                      "note": "gate was green; exact BH across {G-primary, G-secondary, O, P-causal} "
                              "recomputed at closeout — p_x4 is the conservative bound"}
    rep["law_instance_row"] = {f"L{sel['primary']}_rf_by_k":
                               {str(k): curves[f"L{sel['primary']}_k{k}"]["saction"]["rf"]
                                for k in k_grid}}
    rep["green"] = True
    _w()
    print("CONFIRM_OK", json.dumps(rep["verdict"], default=str))


if __name__ == "__main__":
    main()
