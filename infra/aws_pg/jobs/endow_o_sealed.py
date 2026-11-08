"""ENDOW-O sealed confirmation (spec §3, UNBLIND ONCE on the sha1-sealed half) — the confirmatory
origin verdict. Spends INHERIT's preserved seal exactly once. Dead-model-floor RF, direct-forward
transport gate, 3 em-random seeds (band), BH over the promoted family {O-A, O-B}.

Frozen §3.5 rules:
  H_inherited-aligned(L0) iff sealed gap(em − S_action) ci-lo > δ_O at ≥1 promoted (VQA,k) cell under
    BH, with RF non-increasing across the promoted k's.
  H_inherited-rotated(L14) iff the aligned cells fail AND a promoted Procrustes cell clears the same
    rule AND the Procrustes guard was null (checked at controls).
  H_constructed(L14) iff both transports fail while the gate is green.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_o_sealed --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/endow_o_sealed --budget-min 300 \
      --args "--controls-s3 ... --targets-s3 ... --procrustes-s3 ... --disc-s3 ... --full-bank-s3 ..."
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
    ap.add_argument("--disc-s3", required=True)
    ap.add_argument("--full-bank-s3", required=True)
    ap.add_argument("--n-items", type=int, default=None)
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
    from infra.pg import endow_o as EO

    SH.install_pg()
    ctrl = json.loads(SH.fetch(a.controls_s3, "/tmp/endow/controls.json").read_text())
    disc = json.loads(SH.fetch(a.disc_s3, "/tmp/endow/disc.json").read_text())
    assert ctrl.get("green") and disc.get("green"), "upstream not green"
    assert ctrl["transport_gate"]["green"], "direct-forward gate RED — sealed origin must not run"
    tg = torch.load(SH.fetch(a.targets_s3, "/tmp/endow/pg1_targets.pt"), map_location="cpu", weights_only=False)
    hbars = tg["hbars"]
    R14 = torch.load(SH.fetch(a.procrustes_s3, "/tmp/endow/procrustes_R14.pt"),
                     map_location="cpu", weights_only=False)["R14"]
    bases, _h, _t = SH.load_bank()
    sel = ctrl["selection"]
    k_grid, k_star = sel["k_grid"], sel["k_star"]
    dead_floor = ctrl["frozen_thresholds"]["floor"]
    delta_O = ctrl["frozen_thresholds"]["delta_O"]
    guard_null_ok = bool(ctrl["procrustes_guard"]["null_ok"])
    promoted = disc["promoted_sealed_set"]
    n_items = a.n_items or int(ctrl["n_items"])
    rep = {"selection": sel, "dead_floor": dead_floor, "delta_O": delta_O, "promoted": promoted,
           "guard_null_ok": guard_null_ok}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, n_items * 2)
    split = IC.sha1_split(hashes, name="inherit_vqa")
    sealed = [rows[i] for i in split["conf_idx"]][:n_items]      # THE SEAL — touched once
    rep["n_sealed_used"] = len(sealed)
    intact = IP.eval_pg1(model, proc, vqa, sealed, "cuda")
    ii = np.asarray(intact["per_item"], float)
    rep["intact_sealed"] = {"score": intact["score"], "ci": intact["ci95"]}
    _w()

    def eval_Q(L, Q):
        r = IP.eval_pg1(model, proc, vqa, sealed, "cuda", ctx=IP.ablate_at(layers, L, Q, hbars[L]))
        return np.asarray(r["per_item"], float)

    def cell_curve(L, transport):
        curve, pvals = {}, {}
        for k in k_grid:
            V = SH.build_arm_Q("saction", L, k, bases, seed=0)
            Q = EO.transport_basis(R14, V) if transport == "procrustes" else V
            sa = eval_Q(L, Q)
            ems = [eval_Q(L, SH.build_arm_Q("random_energy", L, k, bases, seed=200 + s,
                                            X=tg["Xs"][L].double(), hbar=hbars[L])) for s in range(3)]
            em_mean = np.mean(np.stack(ems, 0), 0)
            gap = IC.gap_item_bootstrap(sa, em_mean, ii, dead_floor)
            p = IC.boot_p_gt(gap["boots"], delta_O)
            gap.pop("boots")
            curve[f"k{k}"] = {"saction": IC.rf_item_bootstrap(sa, ii, dead_floor),
                              "random_energy_band": [IC.rf_item_bootstrap(e, ii, dead_floor)["rf"] for e in ems],
                              "gap": gap, "p_gap_gt_delta": p}
            pvals[k] = p
            _w()
        return curve, pvals

    # one full k-curve per unique (layer, transport) lane; verdict reads each promoted (L,k) cell
    lanes = {}
    for cell in promoted:
        lane = f"L{cell['L']}_{cell['transport']}"
        if lane not in lanes:
            lanes[lane], _ = cell_curve(cell["L"], cell["transport"])
    rep["curves_sealed"] = lanes

    def monotone(curve, ks):
        ks = sorted(ks)
        return all(curve[f"k{ks[i]}"]["saction"]["rf"] >= curve[f"k{ks[i + 1]}"]["saction"]["rf"] - 0.05
                   for i in range(len(ks) - 1))

    # BH over the promoted (L,k) family
    fam = [(cell, f"L{cell['L']}_{cell['transport']}") for cell in promoted]
    adj = IC.bh_adjust([lanes[lane][f"k{cell['k']}"]["p_gap_gt_delta"] for cell, lane in fam])
    id_ks = [cell["k"] for cell in promoted if cell["transport"] == "identity" and cell["L"] == sel["primary"]]
    verd = {}
    for i, (cell, lane) in enumerate(fam):
        c = lanes[lane][f"k{cell['k']}"]
        clears = bool(c["gap"]["ci_lo"] > delta_O and adj[i] < 0.05)
        verd[f"{cell['cell']}_L{cell['L']}_k{cell['k']}"] = {
            "gap_ci_lo": c["gap"]["ci_lo"], "bh_p": adj[i], "clears": clears,
            "transport": cell["transport"], "L": cell["L"], "k": cell["k"]}
    id_mono = monotone(lanes.get(f"L{sel['primary']}_identity", {}), id_ks) if id_ks else False
    aligned = id_mono and any(v["clears"] and v["transport"] == "identity" and v["L"] == sel["primary"]
                              for v in verd.values())
    rotated = (not aligned) and guard_null_ok and any(
        v["clears"] and v["transport"] == "procrustes" for v in verd.values())
    if aligned:
        label = "H_inherited-aligned(L0)"
    elif rotated:
        label = "H_inherited-rotated(L14)"
    else:
        label = "H_constructed(L14)_or_unconfirmed"
    rep["verdict"] = {"cells": verd, "origin": label}
    id_lane = lanes.get(f"L{sel['primary']}_identity", {})
    rep["law_instance_L0_rf_by_k"] = {str(k): id_lane[f"k{k}"]["saction"]["rf"] for k in k_grid} \
        if id_lane else {}
    rep["green"] = True
    _w()
    print("O_SEALED_OK", json.dumps(rep["verdict"], default=str))


if __name__ == "__main__":
    main()
