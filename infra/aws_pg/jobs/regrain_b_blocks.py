"""REGRAIN Arm B (GPU) — block-grain adjudication, gated on I4. Discovery + sealed in one job.

Runs ONLY on I4-promoted (gate-passing) block configs. If I4 returned `H_class-boundary` this job
records the certificate and exits without touching the sealed half — the budget does not roll.

Inherits Arm C's frozen rules verbatim, at block grain:
  screening  alignment ‖Q_bᵀV‖²_F/m → top-TOP_ALIGN blocks → single-block causal screening →
             greedy union one block at a time, cap ⌈CAP_ATOMS/m⌉ blocks, stop at ρ·effect(full)
  nulls      uniform random-block unions ×3 AND usage-matched random-block unions ×3 (A2)
  sealed     12 cells, off-target ppl on every cell (A3), BH q=0.10
  verdict    regrain_core.adjudicate_atom_handle (the same table; "atom" reads "block")

H_tiling leg 2 (NEW): do the gate-passing blocks CONTAIN Arm C's B* atoms?
  collapse_count = the number of top-aligned blocks needed to cover 90% of the B*-span energy.
  The leg lands iff collapse_count ≤ 4. Reported with a Haar floor.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_b_blocks --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_b --budget-min 420 --args "\
      --factorial-s3 s3://…/regrain_i_factorial/out --cdisc-s3 s3://…/regrain_c_disc/out \
      --harvest-s3 s3://…/regrain_harvest/out"
"""
import argparse
import gc
import json
import math
import os
import pathlib


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--factorial-s3", required=True)
    ap.add_argument("--cdisc-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--config", default=None, help="override the promoted config name")
    ap.add_argument("--n-disc-items", type=int, default=None)
    ap.add_argument("--n-screen-items", type=int, default=None)
    ap.add_argument("--n-sealed-items", type=int, default=None)
    ap.add_argument("--n-usage-rows", type=int, default=40000)
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
    from infra.pg.ctrliface_core import (energy_matched_random, orthonormalize,
                                         principal_angle_overlap, removed_energy)
    from infra.pg.inherit_core import (bh_adjust, boot_p_gt, gap_item_bootstrap,
                                       rf_item_bootstrap, slice_k)

    RS.install_pg()
    fg = json.loads(RS.fetch(f"{a.factorial_s3}/gates.json", "/tmp/regrain/i_gates.json").read_text())
    promoted = [a.config] if a.config else (fg.get("promoted_to_arm_b") or [])
    rep = {"object": "regrain_b_v1", "factorial_s3": a.factorial_s3,
           "sign_verdict": fg.get("sign_verdict"), "promoted": promoted}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    if not promoted:
        rep["verdict"] = "UNRUN"
        rep["certificate"] = (
            "I4 produced no gate-passing block (b>1) featurizer at L0. The block-grain "
            "adjudication is CLOSED WITH A CERTIFICATE, not with a null result: the natural "
            "constructive instrument for block grain fails behavioural splice-faithfulness at "
            "this budget/mix, so no block-union claim — positive or negative — can be made "
            "about the sealed read. Arm B's budget does not roll (spec §6).")
        print("REGRAIN_B_CERTIFICATE", json.dumps({"sign_verdict": fg.get("sign_verdict")}))
        # green written LAST: a crash after this point must not leave a green
        # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
        rep["green"] = True
        _w()
        return

    cid = promoted[0]
    cell = fg["configs"][cid]
    m = int(cell["b"])
    assert m > 1, f"Arm B is the BLOCK arm; {cid} has b={m}"
    L = int(fg["L"])
    n_disc = a.n_disc_items or RS.N_DISC
    n_screen = a.n_screen_items or RS.N_SCREEN
    n_sealed = a.n_sealed_items or RS.N_SEALED
    max_blocks = max(1, math.ceil(RS.CAP_ATOMS / m))
    rep.update({"config": cid, "m": m, "L": L, "max_blocks": max_blocks, "rho": RS.RHO})
    _w()

    banks = RS.load_read_banks("/tmp/regrain")
    V = banks["saction"]["V"]
    assert banks["saction"]["L"] == L
    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    key = f"L{L}"
    X, labs, _st = RS.load_shards_stream(a.harvest_s3, meta, key, workdir="/tmp/regrain/shards")
    stats = torch.load(RS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    ae, _c, _q, msha = RC.load_bank_any(RS.fetch(f"{a.factorial_s3}/{cell['bank']}",
                                                 f"/tmp/regrain/{cell['bank']}"))
    assert msha == meta["manifest_sha"]
    Wd = RC.decoder_matrix(ae)

    sha, ppl_ref = RS.pinned_pg1_sha("/tmp/regrain")
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
    sealed_rows = [rows[i] for i in split["conf_idx"]][:n_sealed]

    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    assert abs(base_ppl - ppl_ref) <= RS.PPL_TOL, "ppl instrument RED — halt"

    # ---------------- discovery: screening + nulls ----------------
    intact = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda")
    ii = np.asarray(intact["per_item"], float)
    fl = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda", ctx=IP.replace_at(layers, L, hbar))
    floor = float(np.asarray(fl["per_item"], float).mean())
    full = IP.eval_pg1(model, proc, vqa, disc_rows, "cuda",
                       ctx=IP.ablate_at(layers, L, slice_k(V, RS.K_READ), hbar))
    fi = np.asarray(full["per_item"], float)

    def effect(pi, n=None):
        return RS.effect_from(pi[:n] if n else pi, ii[:n] if n else ii, floor)

    eff_full_screen, eff_full_disc = effect(fi, n_screen), effect(fi)
    rep["reference"] = {"intact": intact["score"], "floor": floor,
                        "effect_full_disc": eff_full_disc,
                        "effect_full_screen": eff_full_screen,
                        "screen_effect_se": RS.effect_se(fi[:n_screen], ii[:n_screen], floor)}
    _w()

    def eval_Q(rows_, Q):
        r = IP.eval_pg1(model, proc, vqa, rows_, "cuda", ctx=IP.ablate_at(layers, L, Q, hbar))
        return np.asarray(r["per_item"], float)

    align = GB.block_alignment(Wd, m, V)
    pool = torch.topk(align, min(RS.TOP_ALIGN, ae.n_blocks)).indices.tolist()
    singles = {b: effect(eval_Q(screen_rows, GB.block_subspace(Wd, m, b)), n_screen) for b in pool}
    order = sorted(pool, key=lambda b: -singles[b])
    curve, bstar, stop_n = [], None, None
    for n_b in range(1, min(max_blocks, len(order)) + 1):
        u = order[:n_b]
        e = effect(eval_Q(screen_rows, GB.union_Q(Wd, m, u)), n_screen)
        curve.append({"n_blocks": n_b, "effect": e, "block": order[n_b - 1]})
        if bstar is None and e >= RS.RHO * eff_full_screen:
            bstar, stop_n = list(u), n_b
    if bstar is None:
        bstar, stop_n = order[: min(max_blocks, len(order))], min(max_blocks, len(order))
    r_bstar = len(bstar) * m
    rep["screening"] = {"align_pool": pool, "singles": singles, "order": order,
                        "dose_curve": curve, "bstar_blocks": bstar, "stop_n": stop_n,
                        "bstar_rank": r_bstar}
    _w()

    usage = GB.block_usage(ae, X[: a.n_usage_rows], labs[: a.n_usage_rows].long(), "cuda")
    active_mask = usage["freq"] > 0
    active = torch.nonzero(active_mask).flatten().tolist()
    # block-level usage scale = mean |activation| summed over the block's latents
    lat = usage["latent_mean_act"].reshape(ae.n_blocks, m).sum(1)
    strata = RC.usage_strata(lat, active_mask, n_strata=10)
    nulls, null_rf = {"uniform": [], "usage_matched": []}, {"uniform": [], "usage_matched": []}
    for i in range(3):
        uni = RC.uniform_random_union(active, len(bstar), seed=7 + i, exclude=bstar)
        usg = RC.usage_matched_union(bstar, strata, seed=7 + i)
        nulls["uniform"].append(uni)
        nulls["usage_matched"].append(usg)
        null_rf["uniform"].append(1.0 - effect(eval_Q(disc_rows, GB.union_Q(Wd, m, uni))))
        null_rf["usage_matched"].append(1.0 - effect(eval_Q(disc_rows, GB.union_Q(Wd, m, usg))))
    hw_u = (max(null_rf["uniform"]) - min(null_rf["uniform"])) / 2
    hw_g = (max(null_rf["usage_matched"]) - min(null_rf["usage_matched"])) / 2
    delta_B = max(hw_u, hw_g)
    Qb = GB.union_Q(Wd, m, bstar)
    E_t = removed_energy(X[:20000].double(), Qb, hbar)
    trace = float(((X[:20000].double() - hbar) ** 2).sum())
    _Qem, r_e = energy_matched_random(IP.D, E_t, trace, 900, "cpu", torch.float64)
    rep["nulls"] = {"draws": nulls, "rf_disc": null_rf, "delta_B": delta_B,
                    "em_span": {"E_target": E_t, "trace": trace, "r_e": int(r_e), "seed": 900}}
    rep["support_spread"] = GB.support_spread(align, n_active=len(active))
    _w()

    # ---------------- H_tiling leg 2: do blocks CONTAIN Arm C's B* atoms? ----------------
    cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/regrain/c_disc.json").read_text())
    c_atoms = [int(x) for x in cd["bstar"]]
    med_s = int(cd["median_seed"])
    nm = RS.dict_bank_name("primary", 1, med_s)
    src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
    ae1, _c1, _q1, _m1 = GB.load_dict_bank(RS.fetch(src, f"/tmp/regrain/med_{nm}"))
    Ustar = orthonormalize(RC.decoder_matrix(ae1)[:, c_atoms].double())      # (d, |B*|)
    blk_align = GB.block_alignment(Wd, m, Ustar)
    ranked = torch.argsort(blk_align, descending=True).tolist()
    cum, collapse = 0.0, None
    energies = []
    for n_b in range(1, min(16, ae.n_blocks) + 1):
        Qn = GB.union_Q(Wd, m, ranked[:n_b])
        cov = float((Qn.transpose(0, 1) @ Ustar).pow(2).sum()) / Ustar.shape[1]
        energies.append({"n_blocks": n_b, "bstar_span_energy_covered": cov})
        if collapse is None and cov >= 0.90:
            collapse = n_b
    haar = []
    for s in range(8):
        Qr = orthonormalize(torch.randn(IP.D, 4 * m, generator=torch.Generator().manual_seed(s),
                                        dtype=torch.float64))
        haar.append(float((Qr.transpose(0, 1) @ Ustar).pow(2).sum()) / Ustar.shape[1])
    rep["tiling_leg2"] = {"collapse_curve": energies, "collapse_count": collapse,
                          "rule": "collapse_count <= 4", "holds": bool(collapse is not None
                                                                       and collapse <= 4),
                          "haar_coverage_at_4_blocks": float(np.mean(haar)),
                          "overlap_top_block_with_bstar": float(
                              principal_angle_overlap(GB.block_subspace(Wd, m, ranked[0]),
                                                      Ustar)[0])}
    _w()

    # ---------------- sealed pass (once) ----------------
    ctx = {"full_read": lambda: IP.ablate_at(layers, L, slice_k(V, RS.K_READ), hbar),
           "bstar": lambda: IP.ablate_at(layers, L, GB.union_Q(Wd, m, bstar), hbar),
           "em_span": lambda: IP.ablate_at(layers, L, RS.em_span(int(r_e), 900, IP.D), hbar),
           "equal_rank_span": lambda: IP.ablate_at(layers, L, slice_k(V, r_bstar), hbar),
           "floor_replace": lambda: IP.replace_at(layers, L, hbar)}
    for fam in ("uniform", "usage_matched"):
        for i in range(3):
            blocks = nulls[fam][i]
            tag = f"rand_{'uniform' if fam == 'uniform' else 'usage'}_{i}"
            ctx[tag] = (lambda bl=blocks: IP.ablate_at(layers, L, GB.union_Q(Wd, m, bl), hbar))

    per_item, ppl = {}, {}
    si = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda")
    per_item["intact"] = np.asarray(si["per_item"], float)
    ppl["intact"] = base_ppl
    for nm_, mk in ctx.items():
        r = IP.eval_pg1(model, proc, vqa, sealed_rows, "cuda", ctx=mk())
        per_item[nm_] = np.asarray(r["per_item"], float)
        with mk():
            ppl[nm_] = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        rep.setdefault("cells_raw", {})[nm_] = {"vqa": r["score"], "ppl": ppl[nm_],
                                                "dppl": ppl[nm_] - base_ppl}
        _w()
    np.savez(out / "per_item.npz", **per_item)

    iis = per_item["intact"]
    floor_s = float(per_item["floor_replace"].mean())
    rf = {k: rf_item_bootstrap(v, iis, floor_s) for k, v in per_item.items()
          if k not in ("intact", "floor_replace")}
    eff = {k: 1.0 - rf[k]["rf"] for k in rf}
    uni_pi = np.mean([per_item[f"rand_uniform_{i}"] for i in range(3)], axis=0)
    usg_pi = np.mean([per_item[f"rand_usage_{i}"] for i in range(3)], axis=0)
    gaps = {"uniform_minus_bstar": gap_item_bootstrap(per_item["bstar"], uni_pi, iis, floor_s),
            "usage_minus_bstar": gap_item_bootstrap(per_item["bstar"], usg_pi, iis, floor_s),
            "em_minus_bstar": gap_item_bootstrap(per_item["bstar"], per_item["em_span"], iis,
                                                 floor_s),
            "span_minus_bstar": gap_item_bootstrap(per_item["bstar"],
                                                   per_item["equal_rank_span"], iis, floor_s)}
    pv = {k: boot_p_gt(v["boots"], delta_B) for k, v in gaps.items()}
    ordk = sorted(pv)
    adj = dict(zip(ordk, bh_adjust([pv[k] for k in ordk])))
    for k in gaps:
        gaps[k].pop("boots")
        gaps[k]["p_bh"] = adj[k]
    rand_dppl = [ppl[f"rand_{f}_{i}"] - base_ppl for f in ("uniform", "usage") for i in range(3)]
    dppl_hi = (max(rand_dppl) - min(rand_dppl)) / 2
    verdict = RC.adjudicate_atom_handle(
        effect_bstar=eff["bstar"], effect_full=eff["full_read"], effect_em=eff["em_span"],
        effect_rand_uniform_mean=float(np.mean([eff[f"rand_uniform_{i}"] for i in range(3)])),
        effect_rand_usage_mean=float(np.mean([eff[f"rand_usage_{i}"] for i in range(3)])),
        gap_uniform_cilo=gaps["uniform_minus_bstar"]["ci_lo"],
        gap_usage_cilo=gaps["usage_minus_bstar"]["ci_lo"], delta_C=delta_B, rho=RS.RHO,
        bh_pass=bool(adj["uniform_minus_bstar"] <= RS.BH_Q
                     and adj["usage_minus_bstar"] <= RS.BH_Q),
        dppl_bstar=ppl["bstar"] - base_ppl, dppl_full=ppl["full_read"] - base_ppl,
        dppl_rand_band_hi=dppl_hi)
    # The frozen A3 clause compares Δppl(B*) to the FULL READ, which is itself catastrophic at this
    # site, so it reports `selective` on almost any input. The informative comparator is the
    # random-union band. Reported beside the frozen clause, never in place of it (Arm C used the
    # same adjudicator, so both arms stay comparable).
    rand_mean = float(np.mean(rand_dppl))
    rep.update({"rf": rf, "effects": eff, "gaps": gaps, "verdict": verdict,
                "selectivity": {"dppl_bstar": ppl["bstar"] - base_ppl,
                                "dppl_full": ppl["full_read"] - base_ppl,
                                "dppl_rand_band_hi": dppl_hi,
                                "frozen_clause_selective": bool(
                                    (ppl["bstar"] - base_ppl) <= (ppl["full_read"] - base_ppl)
                                    + dppl_hi),
                                "dppl_random_unions": rand_dppl,
                                "dppl_random_mean": rand_mean,
                                "bstar_over_random_dppl": ((ppl["bstar"] - base_ppl) / rand_mean)
                                if abs(rand_mean) > 1e-9 else None,
                                "note": "the frozen clause's comparator (full read) saturates at "
                                        "this site; bstar_over_random_dppl is the informative "
                                        "off-target ratio"},
                "green": True})
    _w()
    print("REGRAIN_B_OK", json.dumps({"config": cid, "verdict": verdict,
                                      "eff_bstar": eff["bstar"], "eff_full": eff["full_read"],
                                      "tiling_leg2": rep["tiling_leg2"]["holds"],
                                      "collapse_count": rep["tiling_leg2"]["collapse_count"]},
                                     default=str))


if __name__ == "__main__":
    main()
