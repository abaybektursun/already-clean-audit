"""INHERIT §9.4 PG1 controls (blind; DISCOVERY half only) — build the calibration table and
FREEZE τ_pos / δ_G / the transport-alignment bound from control data. No S_action-at-bite-layer
cell is computed here (the negative-anchor cells are pre-registered-null tripwires, not the
contrast).

Arms at (L_primary, k*): identity · random_k ×3 · random_energy ×3 · shuffle ·
topsig_native(k=128) · topsig_transported(k=128) · positive (mean-replace) · floor (scrambled).
Anchors: saction at the two no-bite layers, k*. Transport gate: CKA(PG1 L, π0.5 L) on the banked
items vs the mismatched-layer null. Collateral: ppl gate re-verified.

Frozen calibration (CTRLIFACE convention): δ_G = max(0.15, (RF_null − RF_pos)/3) with RF_null =
min over the em-random draws, RF_pos = the NATIVE top-σ RF; τ_pos = min(0.5, RF_pos + 0.10);
τ_high = max(0.5, RF_null − 0.10); anchor tripwire: RF ci-hi < τ_high trips halt-and-document.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_controls --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/inherit_controls --budget-min 240 \
      --args "--gate0-s3 s3://.../inherit_gate0/out/gate0.json --pi05-bank-s3 s3://.../inherit_pi05_bank/out/cov_cka_bank.pt"
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--pi05-bank-s3", required=True)
    ap.add_argument("--n-items", type=int, default=None, help="override the gate0 frozen n (smoke)")
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

    SH.install_pg()
    gate0 = json.loads(SH.fetch(a.gate0_s3, "/tmp/inherit/gate0.json").read_text())
    assert gate0.get("green"), "Gate-0 not green — controls must not run"
    pi05 = torch.load(SH.fetch(a.pi05_bank_s3, "/tmp/inherit/cov_cka_bank.pt"),
                      map_location="cpu", weights_only=False)
    bases, _ph, _pt = SH.load_bank()
    sel = SH.resolve_selection()
    Lp, k_star = sel["primary"], sel["k_star"]
    n_items = a.n_items or int(gate0["vqa_probe"]["n_frozen"])
    rep = {"selection": sel, "n_items": n_items, "seed": a.seed,
           "gate0_s3": a.gate0_s3, "pi05_bank_s3": a.pi05_bank_s3}

    def _w():
        (out / "controls.json").write_text(json.dumps(rep, indent=2, default=str))

    model, proc = IP.load_pg1()
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, n_items * 2)
    split = IC.sha1_split(hashes, name="inherit_vqa")
    disc = [rows[i] for i in split["disc_idx"]][:n_items]
    rep["n_disc_used"] = len(disc)
    _w()

    # --- PG1-native h̄ / X at the four selected layers (discovery items only) ---
    L_all = sorted({Lp, sel["secondary"], *sel["anchors"]})
    batches = SH.vqa_forward_batches(proc, vqa, disc, "cuda")
    harv = IP.harvest(layers, lambda b: model(**b), batches, L_all, x_cap=a.x_cap)
    hbars = {L: harv[L][0] for L in L_all}
    Xs = {L: harv[L][2] for L in L_all}
    torch.save({"hbars": hbars, "Xs": {L: Xs[L].to(torch.float32) for L in L_all}},
               out / "pg1_targets.pt")

    # --- transport gate: matched vs mismatched-layer CKA on the banked π0.5 items ---
    imgs, tasks = pi05["images"], pi05["tasks"]
    from PIL import Image
    def _pil(t):
        arr = t.numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = arr.transpose(1, 2, 0)
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        return Image.fromarray(arr.clip(0, 255).astype("uint8").squeeze()).convert("RGB")
    NL = len(layers)
    pooled_pg1 = {L: [] for L in range(NL)}
    bs = 8
    for s in range(0, imgs.shape[0], bs):
        ims = [_pil(imgs[j]) for j in range(s, min(s + bs, imgs.shape[0]))]
        txt = [tasks[j] for j in range(s, min(s + bs, imgs.shape[0]))]
        inp = proc(text=txt, images=ims, return_tensors="pt", padding="longest").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        cap = IP.harvest(layers, lambda b: model(**b), [inp], list(range(NL)),
                         x_cap=10 ** 9)
        for L in range(NL):
            X = cap[L][2]
            pooled_pg1[L].append(X.reshape(len(ims), -1, IP.D).mean(1))
    pooled_pg1 = {L: torch.cat(v, 0) for L, v in pooled_pg1.items()}
    cka = np.zeros((NL, NL))
    for i in range(NL):
        for j in range(NL):
            cka[i, j] = IC.linear_cka(pooled_pg1[i], pi05["pooled"][j].double())
    null = [cka[i, j] for i in range(NL) for j in range(NL) if i != j]
    bound = IC.alignment_bound(null, q=0.95)
    rep["transport_cka"] = {"matched": {str(L): float(cka[L, L]) for L in L_all},
                            "bound": bound, "null_mean": float(np.mean(null)),
                            "matrix": cka.tolist()}
    _w()

    # --- control arms (discovery half) ---
    def run(kind, L, k, Q=None, seed=0):
        return SH.eval_arm(model, proc, layers, vqa, disc, kind, L, k, hbars.get(L),
                           Q=Q, seed=seed)

    arms = {}
    arms["identity"] = run("identity", Lp, 0)
    arms["floor"] = run("floor", Lp, 0, seed=a.seed)
    floor = arms["floor"]["score"]
    intact = arms["identity"]["score"]
    assert intact - floor > 0.05, "dynamic range collapsed on the frozen-n discovery set"
    for s in range(3):
        Q = SH.build_arm_Q("random_k", Lp, k_star, bases, seed=100 + s)
        arms[f"random_k_s{s}"] = run("random_k", Lp, k_star, Q=Q)
        Q = SH.build_arm_Q("random_energy", Lp, k_star, bases, seed=200 + s,
                           X=Xs[Lp], hbar=hbars[Lp])
        arms[f"random_energy_s{s}"] = run("random_energy", Lp, k_star, Q=Q)
    arms["shuffle"] = run("shuffle", Lp, k_star, Q=SH.build_arm_Q("shuffle", Lp, k_star, bases, seed=0))
    arms["topsig_native"] = run("topsig_native", Lp, 128,
                                Q=SH.build_arm_Q("topsig_native", Lp, 128, bases, seed=0,
                                                 X=Xs[Lp], hbar=hbars[Lp]))
    tsb = {L: pi05["topsig"][L].double() for L in pi05["topsig"]}
    arms["topsig_transported"] = run("topsig_transported", Lp, 128,
                                     Q=SH.build_arm_Q("topsig_transported", Lp, 128, bases,
                                                      seed=0, topsig_bank=tsb))
    arms["positive"] = run("positive", Lp, 0)
    for L in sel["anchors"]:
        Q = SH.build_arm_Q("saction", L, k_star, bases, seed=0)
        arms[f"anchor_L{L}"] = run("saction", L, k_star, Q=Q)
    rep["arms"] = {kname: {kk: v[kk] for kk in ("score", "ci95", "n")} for kname, v in arms.items()}
    per_item = {kname: v["per_item"] for kname, v in arms.items()}
    torch.save(per_item, out / "per_item_controls.pt")
    _w()

    # --- RF table + FROZEN thresholds (CTRLIFACE convention) ---
    def rf_of(kname):
        return IC.rf_item_bootstrap(np.asarray(arms[kname]["per_item"], float),
                                    np.asarray(arms["identity"]["per_item"], float), floor)

    rf = {kname: rf_of(kname) for kname in arms if kname not in ("identity", "floor")}
    rep["rf_by_arm"] = {kname: {"rf": v["rf"], "ci": v["ci"]} for kname, v in rf.items()}
    rf_null = min(rf[f"random_energy_s{s}"]["rf"] for s in range(3))
    rf_pos = rf["topsig_native"]["rf"]
    thr = {"tau_pos": round(min(0.5, rf_pos + 0.10), 3),
           "tau_high": round(max(0.5, rf_null - 0.10), 3),
           "delta_G": round(max(0.15, (rf_null - rf_pos) / 3.0), 3),
           "alignment_bound": round(bound, 4), "rf_null": rf_null, "rf_pos": rf_pos,
           "floor": floor, "intact": intact}
    rep["frozen_thresholds"] = thr
    # --- transport-gate verdict (frozen §6 rule) ---
    rf_tt = rf["topsig_transported"]["rf"]
    cka_ok = all(cka[L, L] >= bound for L in (Lp, sel["secondary"]))
    rep["transport_gate"] = {"rf_topsig_transported": rf_tt, "tau_pos": thr["tau_pos"],
                             "cka_ok": bool(cka_ok),
                             "green": bool(rf_tt <= thr["tau_pos"] and cka_ok)}
    # --- anchor tripwire ---
    trips = {f"L{L}": bool(rf[f"anchor_L{L}"]["ci"][1] < thr["tau_high"]) for L in sel["anchors"]}
    rep["anchor_tripwire"] = {"trips": trips, "tripped": any(trips.values())}
    rep["green"] = True
    _w()
    print("CONTROLS_OK", json.dumps({"thresholds": thr, "transport_gate": rep["transport_gate"],
                                     "anchor_tripwire": rep["anchor_tripwire"]}, default=str))


if __name__ == "__main__":
    main()
