"""ENDOW-O controls (spec §3, DISCOVERY half) — the calibration + FREEZE for the confirmatory origin.
Three upgrades over inherit_controls, all pre-registered:

  1. DEAD-MODEL FLOOR (measured, §2 R10b). Full-read mean-replace at ALL 18 layers → the fully-dead
     PG1 VQA score. Used as the RF denominator program-wide (the scrambled floor mis-scales residual
     ablations → the INHERIT RF=−9 artifact). Floors are measured, never assumed.
  2. DIRECT-FORWARD TRANSPORT GATE (replaces CKA). Green iff the transported top-σ positive craters
     PG1 (RF ≤ τ_pos); the CKA leg (weak instrument, INHERIT R10a) is recorded for provenance only.
  3. PROCRUSTES transport R(L14) + guard (Arm O-B/O-D). Fit the orthogonal map on paired PG1/π0.5
     pooled residuals (pi05 bank + PG1 re-harvest, discovery items); the guard ablates R·(em-random)
     and must stay null (a rotation that manufactures bite voids O-B's rotated branch).

Freezes τ_pos, δ_O, the dead floor, and saves pg1_targets (h̄/X at the selected layers) + R14.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.endow_o_controls --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/endow_o_controls --budget-min 240 \
      --args "--gate0-s3 s3://pg-lineage-769063704696/results/inherit_gate0/out/gate0.json \
              --pi05-bank-s3 s3://pg-lineage-769063704696/results/inherit_pi05_bank/out/cov_cka_bank.pt"
"""
import argparse
import contextlib
import json
import pathlib


def _pil(t):
    from PIL import Image
    arr = t.numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.max() <= 1.0 + 1e-6:
        arr = arr * 255.0
    return Image.fromarray(arr.clip(0, 255).astype("uint8").squeeze()).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--pi05-bank-s3", required=True)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--x-cap", type=int, default=16384)
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
    gate0 = json.loads(SH.fetch(a.gate0_s3, "/tmp/inherit/gate0.json").read_text())
    assert gate0.get("green"), "Gate-0 not green — controls must not run"
    pi05 = torch.load(SH.fetch(a.pi05_bank_s3, "/tmp/inherit/cov_cka_bank.pt"),
                      map_location="cpu", weights_only=False)
    bases, _ph, _pt = SH.load_bank()
    sel = SH.resolve_selection()
    Lp, Ls, k_star = sel["primary"], sel["secondary"], sel["k_star"]     # L0, L14, 16
    n_items = a.n_items or int(gate0["vqa_probe"]["n_frozen"])
    rep = {"selection": sel, "n_items": n_items, "seed": a.seed, "gate": "direct-forward (CKA dropped)"}

    def _w():
        (out / "controls.json").write_text(json.dumps(rep, indent=2, default=str))

    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, n_items * 2)
    split = IC.sha1_split(hashes, name="inherit_vqa")            # SAME split as INHERIT (seal firewall)
    disc = [rows[i] for i in split["disc_idx"]][:n_items]
    rep["n_disc_used"] = len(disc)
    _w()

    # --- targets: PG1-native h̄/X at every layer (all-18 for the dead floor; L_all keep X for em-null) ---
    L_all = sorted({Lp, Ls, *sel["anchors"]})
    batches = SH.vqa_forward_batches(proc, vqa, disc, "cuda")
    harv_all = IP.harvest(layers, lambda b: model(**b), batches, list(range(len(layers))), x_cap=0)
    hbars_all = {L: harv_all[L][0] for L in range(len(layers))}
    harv = IP.harvest(layers, lambda b: model(**b), batches, L_all, x_cap=a.x_cap)
    hbars = {L: harv[L][0] for L in L_all}
    Xs = {L: harv[L][2] for L in L_all}
    torch.save({"hbars": hbars, "Xs": {L: Xs[L].to(torch.float32) for L in L_all},
                "hbars_all": hbars_all}, out / "pg1_targets.pt")

    # --- DEAD-MODEL FLOOR: full-read mean-replace at ALL layers (measured) ---
    with contextlib.ExitStack() as stack:
        for L in range(len(layers)):
            stack.enter_context(IP.replace_at(layers, L, hbars_all[L]))
        dead = IP.eval_pg1(model, proc, vqa, disc, "cuda")
    intact = IP.eval_pg1(model, proc, vqa, disc, "cuda")
    dead_floor = float(dead["score"])
    rep["dead_model_floor"] = {"score": dead_floor, "ci": dead["ci95"],
                               "intact": intact["score"], "intact_ci": intact["ci95"]}
    assert intact["score"] - dead_floor > 0.05, "no dynamic range above the dead-model floor"
    ii = np.asarray(intact["per_item"], float)
    _w()

    # --- control arms at the primary bite layer L0 (dead-floor RF) ---
    def run(kind, L, k, Q=None, seed=0):
        return SH.eval_arm(model, proc, layers, vqa, disc, kind, L, k, hbars.get(L), Q=Q, seed=seed)

    arms = {}
    for s in range(3):
        arms[f"random_energy_s{s}"] = run("random_energy", Lp, k_star,
                                          Q=SH.build_arm_Q("random_energy", Lp, k_star, bases,
                                                           seed=200 + s, X=Xs[Lp], hbar=hbars[Lp]))
    arms["topsig_native"] = run("topsig_native", Lp, 128,
                                Q=SH.build_arm_Q("topsig_native", Lp, 128, bases, seed=0,
                                                 X=Xs[Lp], hbar=hbars[Lp]))
    tsb = {L: pi05["topsig"][L].double() for L in pi05["topsig"]}
    arms["topsig_transported"] = run("topsig_transported", Lp, 128,
                                     Q=SH.build_arm_Q("topsig_transported", Lp, 128, bases,
                                                      seed=0, topsig_bank=tsb))
    for L in sel["anchors"]:
        arms[f"anchor_L{L}"] = run("saction", L, k_star, Q=SH.build_arm_Q("saction", L, k_star, bases, seed=0))

    def rf_of(kn):
        return IC.rf_item_bootstrap(np.asarray(arms[kn]["per_item"], float), ii, dead_floor)
    rf = {kn: rf_of(kn) for kn in arms}
    rep["rf_by_arm"] = {kn: {"rf": v["rf"], "ci": v["ci"]} for kn, v in rf.items()}
    rf_null = min(rf[f"random_energy_s{s}"]["rf"] for s in range(3))
    rf_pos = rf["topsig_native"]["rf"]
    rf_tt = rf["topsig_transported"]["rf"]
    thr = {"tau_pos": round(min(0.5, rf_pos + 0.10), 4),
           "tau_high": round(max(0.5, rf_null - 0.10), 4),
           "delta_O": round(max(0.05, (rf_null - rf_pos) / 3.0), 4),
           "floor": dead_floor, "rf_null": rf_null, "rf_pos": rf_pos, "intact": intact["score"]}
    rep["frozen_thresholds"] = thr

    # --- DIRECT-FORWARD transport gate (CKA dropped; recorded for provenance) ---
    NL = len(layers)
    pooled_pg1 = {L: [] for L in range(NL)}
    bs = 8
    imgs, tasks = pi05["images"], pi05["tasks"]
    for s in range(0, imgs.shape[0], bs):
        ims = [_pil(imgs[j]) for j in range(s, min(s + bs, imgs.shape[0]))]
        txt = [tasks[j] for j in range(s, min(s + bs, imgs.shape[0]))]
        inp = proc(text=txt, images=ims, return_tensors="pt", padding="longest").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        cap = IP.harvest(layers, lambda b: model(**b), [inp], list(range(NL)), x_cap=10 ** 9)
        for L in range(NL):
            pooled_pg1[L].append(cap[L][2].reshape(len(ims), -1, IP.D).mean(1))
    pooled_pg1 = {L: torch.cat(v, 0).double() for L, v in pooled_pg1.items()}
    cka_matched = {str(L): float(IC.linear_cka(pooled_pg1[L], pi05["pooled"][L].double())) for L in L_all}
    rep["transport_gate"] = {"rf_topsig_transported": rf_tt, "tau_pos": thr["tau_pos"],
                             "green": bool(rf_tt <= thr["tau_pos"]),
                             "cka_matched_provenance": cka_matched}
    _w()

    # --- PROCRUSTES fit R(L14) + guard (Arm O-B / O-D) ---
    A = pi05["pooled"][Ls].double()           # π0.5 residuals (src)
    B = pooled_pg1[Ls]                         # PG1 residuals (dst), same items
    R14 = EO.orthogonal_procrustes(A, B, center=True)
    V14 = SH.build_arm_Q("saction", Ls, 128, bases, seed=0)          # S_action(L14) top-128 (π0.5 coords)
    rep["procrustes_L14"] = {"is_orthogonal": EO.is_orthogonal(R14),
                             "frob_rel_identity": EO.frob_rel_identity(R14),
                             "transported_overlap_saction": EO.transported_overlap(R14, V14)}
    torch.save({"R14": R14, "Ls": Ls}, out / "procrustes_R14.pt")
    # guard: transport an em-random span, ablate R·(em-random) in PG1 at L14 → must stay null (RF≈1)
    emr = SH.build_arm_Q("random_energy", Ls, k_star, bases, seed=777, X=Xs[Ls], hbar=hbars[Ls])
    emr_transported = EO.transport_basis(R14, emr)
    guard = IP.eval_pg1(model, proc, vqa, disc, "cuda",
                        ctx=IP.ablate_at(layers, Ls, emr_transported, hbars[Ls]))
    guard_rf = IC.rf_item_bootstrap(np.asarray(guard["per_item"], float), ii, dead_floor)
    # null_ok iff the transported em-random does NOT crater (RF stays above τ_high): the rotation did
    # not manufacture bite → O-B's rotated branch is valid. A crater here voids the rotated branch.
    rep["procrustes_guard"] = {"rf": guard_rf["rf"], "ci": guard_rf["ci"],
                               "n_pairs": int(A.shape[0]),
                               "null_ok": bool(guard_rf["rf"] > thr["tau_high"])}
    rep["green"] = True
    _w()
    print("O_CONTROLS_OK", json.dumps({"thresholds": thr, "transport_gate": rep["transport_gate"],
                                       "procrustes": rep["procrustes_L14"],
                                       "guard": rep["procrustes_guard"]}, default=str))


if __name__ == "__main__":
    main()
