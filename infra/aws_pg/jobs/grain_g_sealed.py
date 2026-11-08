"""GRAIN G2 (GPU, SEALED half, touched ONCE) — evaluate the promoted cells frozen in
g_disc.json, compute the paired statistics + BH, adjudicate via the frozen §5.2 table
(grain_bsf.adjudicate_grain), report rank-efficiency + collateral ppl. No screening, no
calibration, no rule changes here — every construction is reconstructed bit-for-bit from the
g_disc artifact (block ids, em seed/rank, δ_G, ρ).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_g_sealed --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_g_sealed_saction --budget-min 240 --args "\
      --read saction --gdisc-s3 .../grain_g_disc_saction/out --dicts-s3 .../grain_train_primary/out \
      --harvest-s3 .../grain_harvest/out --gate0-s3 .../grain_gate0/out"
"""
import argparse
import json
import pathlib

BH_Q = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--read", required=True, choices=("saction", "svl"))
    ap.add_argument("--gdisc-s3", required=True)
    ap.add_argument("--dicts-s3", required=True)
    ap.add_argument("--gates-s3", default=None)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--n-sealed-items", type=int, default=400)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg.ctrliface_core import energy_matched_random
    from infra.pg.inherit_core import (bh_adjust, boot_p_gt, gap_item_bootstrap,
                                       rf_item_bootstrap, slice_k)

    GS.install_pg()
    disc = json.loads(GS.fetch(f"{a.gdisc_s3}/g_disc.json", "/tmp/grain/g_disc.json").read_text())
    rep = {"object": "grain_g_sealed_v1", "read": a.read, "gdisc_s3": a.gdisc_s3,
           "mode": disc.get("mode")}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()
    if disc.get("mode") != "full":
        rep["verdict"] = "UNRUN"
        _w()
        print("G_SEALED_UNRUN", disc.get("mode"))
        return

    L = int(disc["L"])
    m_star = int(disc["m_star"])
    med_s = int(disc["median_seed"])
    cells_spec = disc["promoted_cells"]
    delta_G = float(disc["random_unions"]["delta_G"])
    rho = float(disc["rho"])
    banks = GS.load_read_banks()
    V = banks[a.read]["V"]
    assert banks[a.read]["L"] == L

    gates = json.loads(GS.fetch(f"{a.gates_s3 or a.dicts_s3}/gates.json", "/tmp/grain/gates.json").read_text())
    ae_med, _c, _q, msha = GB.load_dict_bank(GS.fetch(
        f"{a.dicts_s3}/{GS.dict_bank_name(gates['site'], m_star, med_s)}", "/tmp/grain/d_med.pt"))
    assert msha == gates["manifest_sha"]
    Wmed = ae_med.decoder.weight.detach()
    ae_m1 = None
    if cells_spec.get("grain1_union", {}).get("latents"):
        ae_m1, _c1, _q1, _s1 = GB.load_dict_bank(GS.fetch(
            f"{a.dicts_s3}/{GS.dict_bank_name(gates['site'], 1, 0)}", "/tmp/grain/d_m1.pt"))

    sha, _ppl_ref = GS.pinned_pg1_sha()
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(GS.fetch(f"{a.gate0_s3}/items_vqa.json", "/tmp/grain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift vs the gate0 bank — halt"
    split = GS.grain_split(hashes, name="grain_vqa")
    sealed_rows = [rows[i] for i in split["conf_idx"]][: a.n_sealed_items]
    rep["n_sealed"] = len(sealed_rows)
    stats = torch.load(GS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/grain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]

    # --- instrument gate BEFORE any cell is read (§2 ppl discipline) ---
    _sha2, ppl_ref = GS.pinned_pg1_sha()
    pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": ppl_ref, "delta": pp["ppl"] - ppl_ref}
    assert abs(pp["ppl"] - ppl_ref) <= GS.PPL_TOL, "ppl instrument RED before sealed read — halt"
    _w()

    # --- reconstruct the frozen cell constructions ---
    bstar = cells_spec["bstar"]["blocks"]
    r_bstar = len(bstar) * m_star
    em = cells_spec["em_span"]
    Qem, r_e = energy_matched_random(IP.D, disc["em_span"]["E_target"],
                                     disc["em_span"]["trace"], em["seed"], "cpu", torch.float64)
    assert int(r_e) == int(em["r_e"]), "em-span reconstruction drift"
    constructions = {
        "full_read": lambda: IP.ablate_at(layers, L, slice_k(V, GS.K_READ), hbar),
        "bstar": lambda: IP.ablate_at(layers, L, GB.union_Q(Wmed, m_star, bstar), hbar),
        "rand_union_0": lambda: IP.ablate_at(layers, L, GB.union_Q(
            Wmed, m_star, cells_spec["rand_union_0"]["blocks"]), hbar),
        "rand_union_1": lambda: IP.ablate_at(layers, L, GB.union_Q(
            Wmed, m_star, cells_spec["rand_union_1"]["blocks"]), hbar),
        "rand_union_2": lambda: IP.ablate_at(layers, L, GB.union_Q(
            Wmed, m_star, cells_spec["rand_union_2"]["blocks"]), hbar),
        "em_span": lambda: IP.ablate_at(layers, L, Qem, hbar),
        "equal_rank_span": lambda: IP.ablate_at(layers, L, slice_k(V, r_bstar), hbar),
        "floor_replace": lambda: IP.replace_at(layers, L, hbar),
    }
    if ae_m1 is not None:
        W1 = ae_m1.decoder.weight.detach()
        g1 = cells_spec["grain1_union"]["latents"]
        constructions["grain1_union"] = lambda: IP.ablate_at(layers, L,
                                                             GB.union_Q(W1, 1, g1), hbar)

    # --- the sealed pass (once) ---
    per_item = {}
    intact = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda")
    per_item["intact"] = np.asarray(intact["per_item"], float)
    # identity hook cell: harness inertness ON THE SEALED forward (must equal intact exactly)
    h = layers[L].register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(), with_kwargs=True)
    ident = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda")
    h.remove()
    per_item["identity_hook"] = np.asarray(ident["per_item"], float)
    assert np.array_equal(per_item["identity_hook"], per_item["intact"]), \
        "identity-hook cell diverged from intact — harness not inert"
    for name, ctx in constructions.items():
        r = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda", ctx=ctx())
        per_item[name] = np.asarray(r["per_item"], float)
        rep.setdefault("cells_raw", {})[name] = {"score": r["score"], "ci": r["ci95"]}
        _w()
    np.savez(out / "per_item.npz", **{k: v for k, v in per_item.items()})

    # --- statistics (paired, dead-floor RF; effects vs matched nulls) ---
    ii = per_item["intact"]
    floor = float(per_item["floor_replace"].mean())
    rf = {name: rf_item_bootstrap(pi, ii, floor) for name, pi in per_item.items()
          if name not in ("intact", "floor_replace", "identity_hook")}
    rep["rf"] = rf
    rand_mean_pi = np.mean([per_item[f"rand_union_{i}"] for i in range(3)], axis=0)
    gaps = {"rand_minus_bstar": gap_item_bootstrap(per_item["bstar"], rand_mean_pi, ii, floor),
            "em_minus_bstar": gap_item_bootstrap(per_item["bstar"], per_item["em_span"], ii, floor),
            "span_minus_bstar": gap_item_bootstrap(per_item["bstar"],
                                                   per_item["equal_rank_span"], ii, floor)}
    if "grain1_union" in per_item:
        gaps["rand_minus_grain1"] = gap_item_bootstrap(per_item["grain1_union"], rand_mean_pi,
                                                       ii, floor)
    pvals = {k: boot_p_gt(v["boots"], delta_G) for k, v in gaps.items()}
    order = sorted(pvals)
    adj = dict(zip(order, bh_adjust([pvals[k] for k in order])))
    for k in gaps:
        gaps[k].pop("boots")
        gaps[k]["p_gt_deltaG"] = pvals[k]
        gaps[k]["p_bh"] = adj[k]
    rep["gaps"] = gaps
    rep["bh"] = {"q": BH_Q, "family": order}
    _w()

    # --- adjudication (frozen table; effects = 1 − RF on the sealed half) ---
    eff = {name: 1.0 - rf[name]["rf"] for name in rf}
    support = disc["support_spread"]
    verdict = GB.adjudicate_grain(
        effect_bstar=eff["bstar"], effect_full=eff["full_read"], effect_em=eff["em_span"],
        effect_grain1=eff.get("grain1_union", 0.0),
        effect_rand_mean=float(np.mean([eff[f"rand_union_{i}"] for i in range(3)])),
        gap_cilo=gaps["rand_minus_bstar"]["ci_lo"], delta_G=delta_G, rho=rho,
        bh_pass=bool(adj["rand_minus_bstar"] <= BH_Q),
        support_frac_90=float(support["frac_active_for_90"]))
    rep["effects"] = eff
    rep["support_spread"] = support
    rep["verdict"] = verdict
    # rank-efficiency (reported regardless, §5.2): B* vs the old instrument at matched rank
    er = eff["equal_rank_span"]
    rep["rank_efficiency"] = {"effect_bstar": eff["bstar"], "effect_equal_rank_span": er,
                              "ratio": (eff["bstar"] / er) if er > 1e-9 else None,
                              "gap_ci": gaps["span_minus_bstar"]["ci"]}
    _w()

    # --- collateral ppl under B* ablation (capability-specificity) ---
    with IP.ablate_at(layers, L, GB.union_Q(Wmed, m_star, bstar), hbar):
        pp_b = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
    rep["ppl_collateral"] = {"base": pp["ppl"], "bstar_ablate": pp_b["ppl"]}
    rep["green"] = True
    _w()
    print("G_SEALED_OK", json.dumps({"read": a.read, "verdict": verdict,
                                     "eff_bstar": eff["bstar"], "eff_full": eff["full_read"],
                                     "gap_cilo": gaps["rand_minus_bstar"]["ci_lo"],
                                     "delta_G": delta_G}, default=str))


if __name__ == "__main__":
    main()
