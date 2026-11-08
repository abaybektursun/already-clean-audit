"""ENDOW-O discovery (spec §3, DISCOVERY half) — the exploratory sweep that PROMOTES the sealed set.
Dead-model-floor RF. Two reads per the genesis question:

  * IDENTITY transport S_action at L0 (origin, expect crater) and L14 (secondary, expect flat) — the
    INHERIT forward, re-run under the frozen dead floor on the discovery half.
  * PROCRUSTES transport S_action at L14 (Arm O-B genesis) — ablate span(Rᵀ·V) with R the frozen
    R(L14); if it bites where identity was flat, the mid-stack read pre-existed up to rotation.

Promotes {O-A: (L0, k*), identity} always + {O-B: (L14, k*), procrustes} iff it clears the em-null on
discovery. Requires controls green + the direct-forward transport gate green.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_o_disc --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/endow_o_disc --budget-min 240 \
      --args "--controls-s3 .../endow_o_controls/out/controls.json \
              --targets-s3 .../endow_o_controls/out/pg1_targets.pt \
              --procrustes-s3 .../endow_o_controls/out/procrustes_R14.pt \
              --full-bank-s3 .../inherit_pi05_bank/out/bases_full.pt"
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls-s3", required=True)
    ap.add_argument("--targets-s3", required=True)
    ap.add_argument("--procrustes-s3", required=True)
    ap.add_argument("--full-bank-s3", required=True)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
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
    from infra.pg import endow_o as EO

    SH.install_pg()
    ctrl = json.loads(SH.fetch(a.controls_s3, "/tmp/endow/controls.json").read_text())
    assert ctrl.get("green"), "controls not green"
    assert ctrl["transport_gate"]["green"], "direct-forward transport gate RED — O-A must not proceed"
    tg = torch.load(SH.fetch(a.targets_s3, "/tmp/endow/pg1_targets.pt"), map_location="cpu", weights_only=False)
    hbars = tg["hbars"]
    Xs = {L: v.double() for L, v in tg["Xs"].items()}
    R14 = torch.load(SH.fetch(a.procrustes_s3, "/tmp/endow/procrustes_R14.pt"),
                     map_location="cpu", weights_only=False)["R14"]
    bases, _h, _t = SH.load_bank()
    redraw = SH.load_redraw_bases(a.full_bank_s3)
    sel = ctrl["selection"]
    Lp, Ls, k_grid, k_star = sel["primary"], sel["secondary"], sel["k_grid"], sel["k_star"]
    dead_floor = ctrl["frozen_thresholds"]["floor"]
    delta_O = ctrl["frozen_thresholds"]["delta_O"]
    n_items = a.n_items or int(ctrl["n_items"])
    rep = {"selection": sel, "n_items": n_items, "dead_floor": dead_floor, "delta_O": delta_O,
           "controls_s3": a.controls_s3}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, n_items * 2)
    split = IC.sha1_split(hashes, name="inherit_vqa")
    disc = [rows[i] for i in split["disc_idx"]][:n_items]
    intact = IP.eval_pg1(model, proc, vqa, disc, "cuda")
    ii = np.asarray(intact["per_item"], float)
    rep["intact"] = {"score": intact["score"], "ci": intact["ci95"]}

    def eval_Q(L, Q):
        r = IP.eval_pg1(model, proc, vqa, disc, "cuda", ctx=IP.ablate_at(layers, L, Q, hbars[L]))
        return np.asarray(r["per_item"], float)

    sweep = {}
    for L in (Lp, Ls):
        for k in k_grid:
            sa = eval_Q(L, SH.build_arm_Q("saction", L, k, bases, seed=0))
            em = eval_Q(L, SH.build_arm_Q("random_energy", L, k, bases, seed=200, X=Xs[L], hbar=hbars[L]))
            cell = {"saction": IC.rf_item_bootstrap(sa, ii, dead_floor),
                    "random_energy": IC.rf_item_bootstrap(em, ii, dead_floor),
                    "gap": IC.gap_item_bootstrap(sa, em, ii, dead_floor)}
            cell["gap"].pop("boots")
            # Procrustes-transported S_action (Arm O-B genesis) at the secondary/mid layer
            if L == Ls:
                Vp = EO.transport_basis(R14, SH.build_arm_Q("saction", L, k, bases, seed=0))
                pa = eval_Q(L, Vp)
                cell["procrustes"] = IC.rf_item_bootstrap(pa, ii, dead_floor)
                gp = IC.gap_item_bootstrap(pa, em, ii, dead_floor)
                gp.pop("boots")
                cell["gap_procrustes"] = gp
            sweep[f"L{L}_k{k}"] = cell
            _w()
    rep["sweep_discovery"] = sweep

    # promotion (spec §3.3): the (readout, k) cells that CLEAR the em-null on the discovery half —
    # NOT the fixed CTRLIFACE k* (the L0 origin read is DISTRIBUTED: it bites at high k, not k16).
    promoted = [{"cell": "O-A", "L": Lp, "k": k, "transport": "identity"}
                for k in k_grid if sweep[f"L{Lp}_k{k}"]["gap"]["ci_lo"] > delta_O]
    ob_ks = [k for k in k_grid
             if (sweep[f"L{Ls}_k{k}"].get("gap_procrustes") or {}).get("ci_lo", -9) > delta_O]
    promoted += [{"cell": "O-B", "L": Ls, "k": k, "transport": "procrustes"} for k in ob_ks]
    rep["promoted_sealed_set"] = promoted
    rep["ob_procrustes_clears_disc"] = bool(ob_ks)
    assert promoted, "no O-A cell cleared on discovery — origin read not detectable at this floor (halt)"

    # anchors tripwire recheck (identity S_action at the no-bite layers)
    for L in sel["anchors"]:
        sa = eval_Q(L, SH.build_arm_Q("saction", L, k_star, bases, seed=0))
        rep[f"anchor_L{L}"] = IC.rf_item_bootstrap(sa, ii, dead_floor)
    _w()

    # ppl collateral at L0 (capability-specificity)
    with IP.ablate_at(layers, Lp, SH.build_arm_Q("saction", Lp, k_star, bases, seed=0), hbars[Lp]):
        pp_sa = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    pp_base = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    rep["ppl_collateral"] = {"base": pp_base["ppl"], "saction_L0": pp_sa["ppl"]}
    rep["green"] = True
    _w()
    print("O_DISC_OK", json.dumps({"promoted": promoted, "ob_clears": bool(ob_ks)}, default=str))


if __name__ == "__main__":
    main()
