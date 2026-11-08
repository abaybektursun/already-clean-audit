"""REGRAIN Arm C2 (GPU, SEALED half, touched ONCE) — the MDE's confirmatory pass.

Evaluates exactly the 12 cells frozen in c_disc.json, reconstructing every construction
bit-for-bit from that artifact (atom ids, em seed/rank, δ_C, ρ). No screening, no calibration,
no rule change here.

Cells: intact · dead floor · identity hook · full-read k=128 · B* · 3× uniform random-atom
unions · 3× usage-matched random-atom unions (A2) · em-random span at rank(B*) · equal-rank
top read-span directions.

EVERY cell also reports blank-image ppl at n_seq=120 (A3) — the off-target readout that
separates "compact control of the read" from "general damage to the stream".

Frozen adjudication: regrain_core.adjudicate_atom_handle.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_c_sealed --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_c_sealed --budget-min 300 \
      --args "--cdisc-s3 s3://…/results/regrain_c_disc/out"
"""
import argparse
import json
import os
import pathlib


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cdisc-s3", required=True)
    ap.add_argument("--n-sealed-items", type=int, default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.inherit_core import (bh_adjust, boot_p_gt, gap_item_bootstrap,
                                       rf_item_bootstrap, slice_k)

    RS.install_pg()
    disc = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/regrain/c_disc.json").read_text())
    assert disc.get("green"), "c_disc artifact not green — halt"
    L = int(disc["L"])
    med_s = int(disc["median_seed"])
    cells_spec = disc["promoted_cells"]
    delta_C = float(disc["delta_C"]["value"])
    rho = float(disc["rho"])
    bstar = [int(x) for x in cells_spec["bstar"]["atoms"]]
    r_bstar = len(bstar)
    n_sealed = a.n_sealed_items or RS.N_SEALED

    rep = {"object": "regrain_c_sealed_v1", "read": "saction", "L": L, "cdisc_s3": a.cdisc_s3,
           "median_seed": med_s, "bstar": bstar, "bstar_rank": r_bstar, "delta_C": delta_C,
           "rho": rho, "n_ppl_seq": RS.N_PPL_SEQ}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    banks = RS.load_read_banks("/tmp/regrain")
    V = banks["saction"]["V"]
    assert banks["saction"]["L"] == L

    # the median-seed dictionary: seed 0 from the GRAIN bank, seeds 1-2 from c_disc's own out/
    name = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{name}" if med_s == 0 else f"{a.cdisc_s3}/{name}"
    ae_med, _c, _q, msha = GB.load_dict_bank(RS.fetch(src, f"/tmp/regrain/{name}"))
    assert msha == disc["manifest_sha"], "dictionary trained on a different data manifest"
    Wmed = RC.decoder_matrix(ae_med)

    sha, ppl_ref = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift vs the gate0 bank — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    sealed_rows = [rows[i] for i in split["conf_idx"]][:n_sealed]
    rep["n_sealed"] = len(sealed_rows)
    stats = torch.load(RS.fetch(f"{RS.GRAIN_HARVEST}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]

    # --- instrument gate BEFORE any sealed cell is read (§2 ppl discipline) ---
    pp = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)
    rep["ppl_gate"] = {"ppl": pp["ppl"], "ref": ppl_ref, "delta": pp["ppl"] - ppl_ref}
    assert abs(pp["ppl"] - ppl_ref) <= RS.PPL_TOL, "ppl instrument RED before sealed read — halt"
    base_ppl = pp["ppl"]
    _w()

    # --- reconstruct the frozen constructions ---
    em = cells_spec["em_span"]
    Qem = RS.em_span(em["r_e"], em["seed"], IP.D)
    ctx = {
        "full_read": lambda: IP.ablate_at(layers, L, slice_k(V, RS.K_READ), hbar),
        "bstar": lambda: IP.ablate_at(layers, L, GB.union_Q(Wmed, 1, bstar), hbar),
        "em_span": lambda: IP.ablate_at(layers, L, Qem, hbar),
        "equal_rank_span": lambda: IP.ablate_at(layers, L, slice_k(V, r_bstar), hbar),
        "floor_replace": lambda: IP.replace_at(layers, L, hbar),
    }
    for fam in ("uniform", "usage"):
        for i in range(3):
            atoms = [int(x) for x in cells_spec[f"rand_{fam}_{i}"]["atoms"]]
            assert len(atoms) == r_bstar, f"null union {fam}_{i} rank drift"
            ctx[f"rand_{fam}_{i}"] = (lambda at=atoms:
                                      IP.ablate_at(layers, L, GB.union_Q(Wmed, 1, at), hbar))

    # --- the sealed pass (once) ---
    per_item, ppl = {}, {}
    intact = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda")
    per_item["intact"] = np.asarray(intact["per_item"], float)
    ppl["intact"] = base_ppl

    h = layers[L].register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(), with_kwargs=True)
    ident = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda")
    h.remove()
    per_item["identity_hook"] = np.asarray(ident["per_item"], float)
    assert np.array_equal(per_item["identity_hook"], per_item["intact"]), \
        "identity-hook cell diverged from intact — harness not inert"

    for name_, mk in ctx.items():
        r = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda", ctx=mk())
        per_item[name_] = np.asarray(r["per_item"], float)
        with mk():                                    # A3: off-target ppl on EVERY cell
            ppl[name_] = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        rep.setdefault("cells_raw", {})[name_] = {"vqa": r["score"], "ci": r["ci95"],
                                                  "ppl": ppl[name_],
                                                  "dppl": ppl[name_] - base_ppl}
        _w()
    np.savez(out / "per_item.npz", **per_item)

    # --- statistics (paired, dead-floor RF; effects vs matched nulls) ---
    ii = per_item["intact"]
    floor = float(per_item["floor_replace"].mean())
    rf = {k: rf_item_bootstrap(v, ii, floor) for k, v in per_item.items()
          if k not in ("intact", "floor_replace", "identity_hook")}
    eff = {k: 1.0 - rf[k]["rf"] for k in rf}
    uni_mean_pi = np.mean([per_item[f"rand_uniform_{i}"] for i in range(3)], axis=0)
    usg_mean_pi = np.mean([per_item[f"rand_usage_{i}"] for i in range(3)], axis=0)
    gaps = {
        "uniform_minus_bstar": gap_item_bootstrap(per_item["bstar"], uni_mean_pi, ii, floor),
        "usage_minus_bstar": gap_item_bootstrap(per_item["bstar"], usg_mean_pi, ii, floor),
        "em_minus_bstar": gap_item_bootstrap(per_item["bstar"], per_item["em_span"], ii, floor),
        "span_minus_bstar": gap_item_bootstrap(per_item["bstar"], per_item["equal_rank_span"],
                                               ii, floor)}
    pvals = {k: boot_p_gt(v["boots"], delta_C) for k, v in gaps.items()}
    order = sorted(pvals)
    adj = dict(zip(order, bh_adjust([pvals[k] for k in order])))
    for k in gaps:
        gaps[k].pop("boots")
        gaps[k]["p_gt_deltaC"] = pvals[k]
        gaps[k]["p_bh"] = adj[k]
    rep["rf"] = rf
    rep["effects"] = eff
    rep["gaps"] = gaps
    rep["bh"] = {"q": RS.BH_Q, "family": order}
    _w()

    # --- A3: the selectivity band from the six random unions' off-target ppl ---
    rand_dppl = [ppl[f"rand_{f}_{i}"] - base_ppl for f in ("uniform", "usage") for i in range(3)]
    dppl_band_hi = (max(rand_dppl) - min(rand_dppl)) / 2
    dppl_bstar = ppl["bstar"] - base_ppl
    dppl_full = ppl["full_read"] - base_ppl
    rep["selectivity"] = {"dppl_bstar": dppl_bstar, "dppl_full_read": dppl_full,
                          "dppl_random_unions": rand_dppl, "dppl_rand_band_hi": dppl_band_hi,
                          "selective": bool(dppl_bstar <= dppl_full + dppl_band_hi)}
    _w()

    # --- frozen adjudication ---
    verdict = RC.adjudicate_atom_handle(
        effect_bstar=eff["bstar"], effect_full=eff["full_read"], effect_em=eff["em_span"],
        effect_rand_uniform_mean=float(np.mean([eff[f"rand_uniform_{i}"] for i in range(3)])),
        effect_rand_usage_mean=float(np.mean([eff[f"rand_usage_{i}"] for i in range(3)])),
        gap_uniform_cilo=gaps["uniform_minus_bstar"]["ci_lo"],
        gap_usage_cilo=gaps["usage_minus_bstar"]["ci_lo"],
        delta_C=delta_C, rho=rho,
        bh_pass=bool(adj["uniform_minus_bstar"] <= RS.BH_Q
                     and adj["usage_minus_bstar"] <= RS.BH_Q),
        dppl_bstar=dppl_bstar, dppl_full=dppl_full, dppl_rand_band_hi=dppl_band_hi)
    rep["verdict"] = verdict
    er = eff["equal_rank_span"]
    rep["rank_efficiency"] = {"effect_bstar": eff["bstar"], "effect_equal_rank_span": er,
                              "ratio": (eff["bstar"] / er) if er > 1e-9 else None,
                              "gap_ci": gaps["span_minus_bstar"]["ci"]}
    rep["rho_check"] = {"effect_bstar": eff["bstar"], "rho_times_full": rho * eff["full_read"],
                        "clears_rho": bool(eff["bstar"] >= rho * eff["full_read"])}
    print("C_SEALED_OK", json.dumps(
        {"verdict": verdict, "eff_bstar": eff["bstar"], "eff_full": eff["full_read"],
         "eff_rand_uniform": float(np.mean([eff[f"rand_uniform_{i}"] for i in range(3)])),
         "eff_rand_usage": float(np.mean([eff[f"rand_usage_{i}"] for i in range(3)])),
         "gap_uni_cilo": gaps["uniform_minus_bstar"]["ci_lo"],
         "gap_usg_cilo": gaps["usage_minus_bstar"]["ci_lo"], "delta_C": delta_C,
         "selective": rep["selectivity"]["selective"], "dppl_bstar": dppl_bstar,
         "dppl_full": dppl_full}, default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
