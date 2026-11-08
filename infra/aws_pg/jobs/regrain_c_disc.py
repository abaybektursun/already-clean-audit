"""REGRAIN Arm C1 (GPU, DISCOVERY half only) — the MDE's screening + null calibration.

Everything the sealed pass consumes is FROZEN here and journaled before any sealed statistic:
the per-seed B* unions, the seed-median union, BOTH random-union families (uniform + usage-
matched, A2), δ_C, the em-span construction, the equal-rank read-span cell, and the promoted
sealed cell list.

Phase A (no model on GPU): retrain m=1 seeds 1,2 from the banked shards — they were never
trained (A5a: GRAIN's instrument_null pivot fired before its extra-seed round).
Phase B (model on GPU): re-gate all three seeds at the PROGRAM-FROZEN n_seq=120, then screen.

Screening recipe (frozen, A4-amended):
  a. alignment ranking ‖u_jᵀV‖² → top-TOP_ALIGN atoms per seed;
  b. causal screening: those atoms ablated singly on the screen slice (N_SCREEN=320 items);
  c. B*_s = greedy cumulative union in descending single-atom-effect order, ONE ATOM AT A TIME
     (v1 used chunks of 8), cap CAP_ATOMS=48, pool extended by alignment past TOP_ALIGN;
     stop at the first prefix with effect ≥ ρ·effect(full read) on the screen slice.
     THE FULL DOSE CURVE TO 48 IS ALWAYS MEASURED AND PUBLISHED (spec §14 F-C2), whether or not
     the stopping rule fires early;
  d. the screen-slice bootstrap SE of `effect` is journaled BEFORE B* is fixed (A4).
  e. seed-median union = the B*_s with median discovery effect.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_c_disc --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_c_disc --budget-min 400
"""
import argparse
import gc
import json
import os
import pathlib


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-disc-items", type=int, default=None)
    ap.add_argument("--n-screen-items", type=int, default=None)
    ap.add_argument("--n-usage-rows", type=int, default=40000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=4000)
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
    from infra.pg.ctrliface_core import energy_matched_random, removed_energy
    from infra.pg.inherit_core import slice_k

    n_disc = a.n_disc_items or RS.N_DISC
    n_screen = a.n_screen_items or RS.N_SCREEN
    seeds = [int(s) for s in a.seeds.split(",")]

    RS.install_pg()
    banks = RS.load_read_banks("/tmp/regrain")
    L, V = banks["saction"]["L"], banks["saction"]["V"]
    gates_v2 = RS.load_v2_gates("/tmp/regrain")
    assert gates_v2["L"] == L
    meta = json.loads(RS.fetch(f"{RS.GRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    rep = {"object": "regrain_c_disc_v1", "read": "saction", "L": L, "rho": RS.RHO,
           "n_disc": n_disc, "n_screen": n_screen, "cap_atoms": RS.CAP_ATOMS,
           "top_align": RS.TOP_ALIGN, "n_ppl_seq": RS.N_PPL_SEQ, "seeds": seeds,
           "manifest_sha": meta["manifest_sha"]}

    def _w():
        (out / "c_disc.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def mem(tag):
        if torch.cuda.is_available():
            print(f"[mem] {tag}: {torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)

    # ============ Phase A: seed 0 from the bank, seeds 1,2 retrained (A5a) ============
    X, labs = RS.load_shards(RS.GRAIN_HARVEST, meta, f"L{L}", workdir="/tmp/regrain/shards")
    train_idx, hold_idx = RS.disc_holdout_split(X.shape[0])
    Xtr, Xho = X[train_idx], X[hold_idx]
    dict_paths, train_rep = {}, {}

    name0 = RS.dict_bank_name("primary", 1, 0)
    p0 = RS.fetch(f"{RS.GRAIN_DICTS_L0}/{name0}", f"/tmp/regrain/{name0}")
    _ae0, cfg0, q0, msha0 = GB.load_dict_bank(p0)
    assert msha0 == meta["manifest_sha"]
    dict_paths[0] = p0
    train_rep["0"] = {"source": "banked (grain_t2)", "quality": q0, "cfg_seed": cfg0["seed"]}
    del _ae0
    _w()

    for s in seeds:
        if s == 0:
            continue
        mem(f"train_start_s{s}")
        ae, cfg = GB.train_block_sae(Xtr, m=1, seed=s, device="cuda", steps=a.steps)
        q = GB.block_quality(ae, Xho, "cuda")
        path = out / RS.dict_bank_name("primary", 1, s)
        GB.save_dict_bank(ae, cfg, q, path, meta["manifest_sha"])
        dict_paths[s] = path
        train_rep[str(s)] = {"source": "retrained (A5a)", "quality": q, "cfg_seed": cfg["seed"],
                             "threshold": cfg["threshold"]}
        del ae
        gc.collect()
        torch.cuda.empty_cache()
        mem(f"train_done_s{s}")
        _w()
    rep["dicts"] = train_rep
    _w()

    # ============ Phase B: model on GPU ============
    sha, _ = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    disc_rows = [rows[i] for i in split["disc_idx"]][:n_disc]
    screen_rows = disc_rows[:n_screen]
    stats = torch.load(RS.fetch(f"{RS.GRAIN_HARVEST}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    mem("model_loaded")

    # --- re-gate every seed at n_seq=120 (the C0 discipline, per seed) ---
    gate_rows = disc_rows[:200]
    intact_gate = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda")
    gate_out = {}
    for s in seeds:
        ae, _c, _q, _m = GB.load_dict_bank(dict_paths[s])
        ae = ae.to("cuda").eval()
        spl = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=GB.splice_at(layers, L, ae))
        with GB.splice_at(layers, L, ae):
            spl_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        # em band at THIS dict's reconstruction-error energy
        n_e = min(20000, Xho.shape[0])
        E_t = tr = 0.0
        hb = hbar.to("cuda").float()
        with torch.inference_mode():
            for st in range(0, n_e, 4096):
                xe = Xho[st:st + 4096].to("cuda").float()
                rc = ae.decode(ae.encode(xe, use_threshold=True))
                E_t += float(((xe - rc) ** 2).sum())
                tr += float(((xe - hb) ** 2).sum())
        dv, dp = [], []
        for es in RS.EM_SEEDS:
            Qem, r_e = energy_matched_random(IP.D, E_t, tr, es, "cpu", torch.float64)
            r = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=IP.ablate_at(layers, L, Qem, hbar))
            dv.append(r["score"] - intact_gate["score"])
            with IP.ablate_at(layers, L, Qem, hbar):
                dp.append(IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"] - base_ppl)
        dvqa, dppl = spl["score"] - intact_gate["score"], spl_ppl - base_ppl
        vhw, phw = (max(dv) - min(dv)) / 2, (max(dp) - min(dp)) / 2
        ok = (dvqa >= float(np.mean(dv)) - max(vhw, 0.01)) and \
             (dppl <= float(np.mean(dp)) + max(phw, RS.PPL_TOL))
        gate_out[str(s)] = {"delta_vqa": dvqa, "delta_ppl": dppl, "r_e": int(r_e),
                            "em_vqa": {"mean": float(np.mean(dv)), "halfwidth": vhw},
                            "em_ppl": {"mean": float(np.mean(dp)), "halfwidth": phw},
                            "gate_pass": bool(ok)}
        ae = ae.to("cpu")
        del ae
        gc.collect()
        torch.cuda.empty_cache()
        rep["c0_regate_per_seed"] = gate_out
        _w()
    passing = [s for s in seeds if gate_out[str(s)]["gate_pass"]]
    rep["seeds_passing_gate"] = passing
    assert passing, "no m=1 seed is v2-faithful at n_seq=120 — INSTRUMENT-DRIFT (§12 pivot)"
    _w()

    # --- reference cells on the discovery slice ---
    intact = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda")
    ii = np.asarray(intact["per_item"], float)
    fl = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda", ctx=IP.replace_at(layers, L, hbar))
    floor = float(np.asarray(fl["per_item"], float).mean())
    full = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda",
                       ctx=IP.ablate_at(layers, L, slice_k(V, RS.K_READ), hbar))
    fi = np.asarray(full["per_item"], float)
    assert ii.mean() - floor > 1e-3, "no dynamic range vs dead floor on discovery — halt"

    def effect(pi, n=None):
        return RS.effect_from(pi[:n] if n else pi, ii[:n] if n else ii, floor)

    eff_full_screen = effect(fi, n_screen)
    eff_full_disc = effect(fi)
    # A4: the screen-slice noise floor, journaled BEFORE B* is fixed
    se_screen = RS.effect_se(fi[:n_screen], ii[:n_screen], floor)
    rep["reference"] = {"intact": intact["score"], "floor": floor, "full_read": full["score"],
                        "effect_full_disc": eff_full_disc, "effect_full_screen": eff_full_screen,
                        "screen_effect_se": se_screen,
                        "rho_threshold_on_screen": RS.RHO * eff_full_screen,
                        "rho_stopped_before_cap": None,
                        "rho_note": "A4: rho gates only |B*|, never pass/fail. The binding rules "
                                    "are the two null gaps."}
    _w()

    def eval_Q(rows_, Q, n=None):
        r = IP.eval_pg1(model, proc, vqa, rows_, "cuda", ctx=IP.ablate_at(layers, L, Q, hbar))
        return np.asarray(r["per_item"], float)

    # --- C1: per-seed screening + per-atom greedy union ---
    per_seed = {}
    for s in passing:
        ae, _c, _q, _m = GB.load_dict_bank(dict_paths[s])
        Wd = RC.decoder_matrix(ae)
        pool = RC.top_aligned_atoms(Wd, V, RS.CAP_ATOMS)      # ranked by alignment
        screened = pool[:RS.TOP_ALIGN]
        singles = {}
        for at in screened:
            singles[at] = effect(eval_Q(screen_rows, GB.union_Q(Wd, 1, [at])), n_screen)
        order = sorted(screened, key=lambda x: -singles[x])
        order += [x for x in pool if x not in order]           # extend by alignment past the pool
        curve, union, bstar, stop_n = [], [], None, None
        for n_at in range(1, RS.CAP_ATOMS + 1):
            union = order[:n_at]
            e = effect(eval_Q(screen_rows, GB.union_Q(Wd, 1, union)), n_screen)
            curve.append({"n_atoms": n_at, "effect": e, "atom": order[n_at - 1]})
            if bstar is None and e >= RS.RHO * eff_full_screen:
                bstar, stop_n = list(union), n_at             # frozen stopping rule; curve continues
        if bstar is None:
            bstar, stop_n = order[:RS.CAP_ATOMS], RS.CAP_ATOMS
        e_disc = effect(eval_Q(disc_rows, GB.union_Q(Wd, 1, bstar)))
        per_seed[str(s)] = {"align_pool": pool, "screened": screened, "singles": singles,
                            "greedy_order": order, "dose_curve": curve, "bstar": bstar,
                            "bstar_len": len(bstar), "stop_n": stop_n,
                            "bstar_effect_disc": e_disc}
        rep["per_seed"] = per_seed
        _w()
        del ae
        gc.collect()
        torch.cuda.empty_cache()

    med_s = sorted(passing, key=lambda s: per_seed[str(s)]["bstar_effect_disc"])[len(passing) // 2]
    ae_med, _c, _q, _m = GB.load_dict_bank(dict_paths[med_s])
    Wmed = RC.decoder_matrix(ae_med)
    bstar = per_seed[str(med_s)]["bstar"]
    r_bstar = len(bstar)
    rep["median_seed"] = med_s
    rep["bstar"] = bstar
    rep["bstar_rank"] = r_bstar
    if len(passing) >= 2:
        rep["bstar_seed_jaccard"] = GB.pairwise_jaccard(
            [per_seed[str(s)]["bstar"] for s in passing])[0]
    rep["reference"]["rho_stopped_before_cap"] = bool(per_seed[str(med_s)]["stop_n"] < RS.CAP_ATOMS)
    rep["reference"]["effect_at_cap_screen"] = per_seed[str(med_s)]["dose_curve"][-1]["effect"]
    # stability cross-check vs GRAIN's banked screening pool (never a selection input)
    gdisc = json.loads(RS.fetch(RS.GRAIN_GDISC, "/tmp/regrain/g_disc.json").read_text())
    v1_pool = [int(k) for k in gdisc["grain1_descriptive"]["singles"]]
    rep["v1_crosscheck"] = {"v1_screened_pool": v1_pool,
                            "overlap_bstar_v1pool": sorted(set(bstar) & set(v1_pool))}
    _w()

    # --- both null families, calibrated on discovery (A2) ---
    usage = GB.block_usage(ae_med, X[: a.n_usage_rows], labs[: a.n_usage_rows].long(), "cuda")
    active_mask = usage["freq"] > 0
    active_atoms = torch.nonzero(active_mask).flatten().tolist()
    strata = RC.usage_strata(usage["latent_mean_act"], active_mask, n_strata=10)
    rep["n_active_atoms"] = len(active_atoms)

    nulls = {"uniform": [], "usage_matched": []}
    null_rf = {"uniform": [], "usage_matched": []}
    for i in range(3):
        uni = RC.uniform_random_union(active_atoms, r_bstar, seed=7 + i, exclude=bstar)
        usg = RC.usage_matched_union(bstar, strata, seed=7 + i)
        nulls["uniform"].append(uni)
        nulls["usage_matched"].append(usg)
        null_rf["uniform"].append(1.0 - effect(eval_Q(disc_rows, GB.union_Q(Wmed, 1, uni))))
        null_rf["usage_matched"].append(1.0 - effect(eval_Q(disc_rows, GB.union_Q(Wmed, 1, usg))))
        rep["nulls"] = {"draws": nulls, "rf_disc": null_rf}
        _w()
    hw_u = (max(null_rf["uniform"]) - min(null_rf["uniform"])) / 2
    hw_g = (max(null_rf["usage_matched"]) - min(null_rf["usage_matched"])) / 2
    delta_C = max(hw_u, hw_g)                       # A2: the WIDER band sets δ_C
    rep["delta_C"] = {"value": delta_C, "halfwidth_uniform": hw_u, "halfwidth_usage": hw_g}
    _w()

    # --- em-span at rank(B*)'s removed energy ---
    Xnull = X[:20000].double()
    Qb = GB.union_Q(Wmed, 1, bstar)
    E_target = removed_energy(Xnull, Qb, hbar)
    trace = float(((Xnull - hbar) ** 2).sum())
    _Qem, r_e = energy_matched_random(IP.D, E_target, trace, 900, "cpu", torch.float64)
    rep["em_span"] = {"E_target": E_target, "trace": trace, "r_e": int(r_e), "seed": 900}

    # --- the frozen promoted sealed cells (12) ---
    rep["promoted_cells"] = {
        "intact": {}, "floor_replace": {}, "identity_hook": {},
        "full_read": {"k": RS.K_READ},
        "bstar": {"atoms": bstar, "seed": med_s},
        **{f"rand_uniform_{i}": {"atoms": nulls["uniform"][i]} for i in range(3)},
        **{f"rand_usage_{i}": {"atoms": nulls["usage_matched"][i]} for i in range(3)},
        "em_span": {"seed": 900, "r_e": int(r_e)},
        "equal_rank_span": {"k": r_bstar}}
    print("C_DISC_OK", json.dumps(
        {"median_seed": med_s, "bstar_len": r_bstar, "stop_n": per_seed[str(med_s)]["stop_n"],
         "eff_full_disc": eff_full_disc, "bstar_eff_disc": per_seed[str(med_s)]["bstar_effect_disc"],
         "delta_C": delta_C, "screen_se": se_screen,
         "rho_stopped_before_cap": rep["reference"]["rho_stopped_before_cap"],
         "n_cells": len(rep["promoted_cells"])}, default=str))
    # `green` is written LAST, after every statement that can raise. A job that crashes on its
    # summary line must not leave a green artifact -- the drivers gate on `green`.
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
