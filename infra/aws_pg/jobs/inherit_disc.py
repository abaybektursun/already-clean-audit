"""INHERIT §9.5 discovery — the S_action-in-PG1 sweep on the DISCOVERY half (exploratory;
the sealed pass is the headline). Requires controls green + transport gate resolved.

Cells: {saction, random_energy(seed 200)} × k-grid × {L_primary, L_secondary}; anchors at k*;
ppl collateral at (L_primary, k*) for saction vs em-random (capability-specificity row).
Promotes the FROZEN sealed set {G-primary (k*, L_primary), G-secondary (k*, L_secondary)} —
fixed by the spec §7, re-affirmed here; predictions re-logged in the journal before §9.6.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_disc --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/inherit_disc --budget-min 300 \
      --args "--controls-s3 s3://.../inherit_controls/out/controls.json --targets-s3 s3://.../inherit_controls/out/pg1_targets.pt"
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--targets-s3", required=True)
    ap.add_argument("--full-bank-s3", required=True, help="re-bank full Gram (saction_redraw arm)")
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
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
    assert ctrl.get("green"), "controls not green"
    # POST-PIVOT MODE (frozen §6: gate red → H_stale-transport). The SEALED forward (inherit_confirm)
    # is skipped; THIS discovery-half sweep runs as a clearly-labelled EXPLORATORY look (never the
    # seal), reporting raw VQA scores + a dead-model-floor RF′ (the scrambled floor is mis-scaled for
    # residual ablations — controls Finding 2). Anchors did not trip → harness is sound.
    rep_gate = {"transport_gate_green": ctrl["transport_gate"]["green"],
                "mode": "EXPLORATORY (gate red → sealed forward skipped; discovery-half look only)",
                "anchor_tripped": ctrl["anchor_tripwire"]["tripped"]}
    assert not ctrl["anchor_tripwire"]["tripped"], "anchor tripwire fired at controls — halt-and-document"
    tg = torch.load(SH.fetch(a.targets_s3, "/tmp/inherit/pg1_targets.pt"),
                    map_location="cpu", weights_only=False)
    hbars = tg["hbars"]
    Xs = {L: v.double() for L, v in tg["Xs"].items()}
    bases, _h, _t = SH.load_bank()
    redraw = SH.load_redraw_bases(a.full_bank_s3)          # independent 2nd draw (robustness leg)
    sel = ctrl["selection"]
    k_grid = sel["k_grid"]
    k_star = sel["k_star"]
    floor = ctrl["frozen_thresholds"]["floor"]          # scrambled floor (frozen; broken for res-ablation)
    floor_dead = 0.0                                    # dead-model floor (Finding 2 repair; RF′)
    n_items = a.n_items or int(ctrl["n_items"])
    rep = {"selection": sel, "n_items": n_items, "controls_s3": a.controls_s3,
           "gate": rep_gate, "floor_scrambled": floor, "floor_deadmodel": floor_dead,
           "promoted_sealed_set": [{"cell": "G-primary", "L": sel["primary"], "k": k_star},
                                   {"cell": "G-secondary", "L": sel["secondary"], "k": k_star}]}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, n_items * 2)
    split = IC.sha1_split(hashes, name="inherit_vqa")
    disc = [rows[i] for i in split["disc_idx"]][:n_items]

    intact = SH.eval_arm(model, proc, layers, vqa, disc, "identity", sel["primary"], 0, None)
    ii = np.asarray(intact["per_item"], float)
    rep["intact"] = {"score": intact["score"], "ci": intact["ci95"]}

    sweep = {}
    for L in (sel["primary"], sel["secondary"]):
        for k in k_grid:
            sa = SH.eval_arm(model, proc, layers, vqa, disc, "saction", L, k, hbars[L],
                             Q=SH.build_arm_Q("saction", L, k, bases, seed=0))
            em = SH.eval_arm(model, proc, layers, vqa, disc, "random_energy", L, k, hbars[L],
                             Q=SH.build_arm_Q("random_energy", L, k, bases, seed=200,
                                              X=Xs[L], hbar=hbars[L]))
            rd = SH.eval_arm(model, proc, layers, vqa, disc, "saction_redraw", L, k, hbars[L],
                             Q=SH.build_arm_Q("saction_redraw", L, k, bases, seed=0, redraw_bases=redraw))
            sa_pi, em_pi, rd_pi = (np.asarray(x["per_item"], float) for x in (sa, em, rd))
            cell = {"raw_score": {"saction": float(sa_pi.mean()), "saction_redraw": float(rd_pi.mean()),
                                  "random_energy": float(em_pi.mean()), "intact": float(ii.mean())},
                    "raw_gap_em_minus_saction": float(em_pi.mean() - sa_pi.mean()),
                    "raw_gap_em_minus_redraw": float(em_pi.mean() - rd_pi.mean()),
                    # RF′ (dead-model floor; the Finding-2 repair) — the interpretable normalization
                    "saction": IC.rf_item_bootstrap(sa_pi, ii, floor_dead),
                    "saction_redraw": IC.rf_item_bootstrap(rd_pi, ii, floor_dead),
                    "random_energy": IC.rf_item_bootstrap(em_pi, ii, floor_dead),
                    "gap": IC.gap_item_bootstrap(sa_pi, em_pi, ii, floor_dead),
                    "gap_redraw": IC.gap_item_bootstrap(rd_pi, em_pi, ii, floor_dead),
                    # RF (scrambled floor; frozen but broken — kept for provenance)
                    "saction_scrambledfloor": IC.rf_item_bootstrap(sa_pi, ii, floor)["rf"]}
            cell["gap"].pop("boots")
            cell["gap_redraw"].pop("boots")
            sweep[f"L{L}_k{k}"] = cell
            _w()
    rep["sweep_discovery"] = sweep

    # anchors at k* (tripwire re-check on the full-n discovery set)
    for L in sel["anchors"]:
        sa = SH.eval_arm(model, proc, layers, vqa, disc, "saction", L, k_star, hbars[L],
                         Q=SH.build_arm_Q("saction", L, k_star, bases, seed=0))
        rep[f"anchor_L{L}"] = IC.rf_item_bootstrap(np.asarray(sa["per_item"], float), ii, floor)
    _w()

    # ppl collateral (capability-specificity, interpretation row)
    Lp = sel["primary"]
    with IP.ablate_at(layers, Lp, SH.build_arm_Q("saction", Lp, k_star, bases, seed=0), hbars[Lp]):
        pp_sa = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    with IP.ablate_at(layers, Lp, SH.build_arm_Q("random_energy", Lp, k_star, bases, seed=200,
                                                 X=Xs[Lp], hbar=hbars[Lp]), hbars[Lp]):
        pp_em = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    pp_base = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    rep["ppl_collateral"] = {"base": pp_base["ppl"], "saction": pp_sa["ppl"], "em": pp_em["ppl"]}
    rep["green"] = True
    _w()
    print("DISC_OK", json.dumps({"promoted": rep["promoted_sealed_set"],
                                 "ppl_collateral": rep["ppl_collateral"]}, default=str))


if __name__ == "__main__":
    main()
