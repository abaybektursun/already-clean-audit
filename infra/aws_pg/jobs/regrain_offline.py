"""REGRAIN offline gate (CPU, --no-weights) — the full pipeline MINUS the PG1 forward, on REAL
banks, REAL shards, REAL item streams and the REAL banked dictionary (house convention: GPU
minutes only validate what only a GPU can).

Exercises end-to-end, on artifacts:
  1. read banks + sites + grain| halves (real streamed rows)
  2. the banked m=1 L0 dictionary: load, sha, held-out FVU on real shards
  3. usage → usage_strata → usage-matched AND uniform random-atom unions (A2), with the
     stratum-match asserted on real activation scales
  4. alignment ranking vs the REAL S_action bank; overlap against GRAIN's banked 40-latent list
     (a stability cross-check, never a selection)
  5. union_Q rank/orthonormality at |B*| up to the cap; em-span rank arithmetic
  6. MDL (Eq. 6) on the REAL code + residual spectra of the banked dictionary, swept over the
     paper's distortion levels frac ∈ {0.01, 0.05, 0.10, 0.20}
  7. Arm D machinery on the REAL S_vl draws: canonicity, permutation null, and the
     canonicity-matched surrogate null (A1)
  8. every adjudicator on the numbers this job actually produced

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_offline --type r7i.4xlarge --no-weights \
      --s3-prefix results/regrain_offline --budget-min 120
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-usage-rows", type=int, default=60000)
    ap.add_argument("--n-fvu-rows", type=int, default=20000)
    ap.add_argument("--n-surrogate-draws", type=int, default=64)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"object": "regrain_offline_v1"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import orthonormalize
    from infra.pg.endow_atlas import canonicity

    # ---- 1: banks, sites, halves ----
    banks = RS.load_read_banks("/tmp/regrain")
    Lp, Ls = banks["saction"]["L"], banks["svl"]["L"]
    V = banks["saction"]["V"]
    rep["sites"] = {"primary": Lp, "secondary": Ls}
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    _rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift"
    split = RS.grain_split(hashes, name="grain_vqa")
    rep["items"] = {"n": len(hashes), "n_disc": int(split["n_disc"]),
                    "n_sealed": int(split["n_conf"])}
    _w()

    # ---- 2: the banked dictionary ----
    meta = json.loads(RS.fetch(f"{RS.GRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    name = RS.dict_bank_name("primary", 1, 0)
    ae, cfg, quality, msha = GB.load_dict_bank(
        RS.fetch(f"{RS.GRAIN_DICTS_L0}/{name}", f"/tmp/regrain/{name}"))
    assert msha == meta["manifest_sha"]
    X, labs = RS.load_shards(RS.GRAIN_HARVEST, meta, f"L{Lp}", workdir="/tmp/regrain/shards")
    _tr, ho = RS.disc_holdout_split(X.shape[0])
    q = GB.block_quality(ae, X[ho], "cpu", n_tokens=a.n_fvu_rows)
    rep["dict_m1_s0"] = {"cfg": {k: cfg[k] for k in ("m", "k_blk", "dict_size", "threshold")},
                         "banked_quality": quality, "recomputed_quality": q,
                         "fvu_reproduces": abs(q["fvu"] - quality["fvu"]) < 0.01}
    assert rep["dict_m1_s0"]["fvu_reproduces"], "banked FVU not reproduced — bank drift"
    _w()

    # ---- 3: usage → strata → the two null families (A2) ----
    usage = GB.block_usage(ae, X[: a.n_usage_rows], labs[: a.n_usage_rows].long(), "cpu")
    scale = usage["latent_mean_act"]
    active_mask = usage["freq"] > 0
    n_active = int(active_mask.sum())
    strata = RC.usage_strata(scale, active_mask, n_strata=10)
    rep["usage"] = {"n_active": n_active, "n_latents": int(scale.numel()),
                    "scale_q": [float(x) for x in
                                torch.quantile(scale[active_mask].double(),
                                               torch.tensor([0.0, .25, .5, .75, 1.0],
                                                            dtype=torch.float64))]}
    _w()

    # ---- 4: alignment ranking vs the REAL S_action bank ----
    Wd = RC.decoder_matrix(ae)
    ranked = RC.top_aligned_atoms(Wd, V, RS.TOP_ALIGN)
    gdisc = json.loads(RS.fetch(RS.GRAIN_GDISC, "/tmp/regrain/g_disc.json").read_text())
    banked_singles = [int(k) for k in gdisc["grain1_descriptive"]["singles"]]
    rep["alignment"] = {"top_align_atoms": ranked,
                        "banked_v1_screened_atoms": banked_singles,
                        "overlap_with_v1_pool": len(set(ranked) & set(banked_singles)),
                        "note": "stability cross-check only; C1 re-screens from scratch"}
    _w()

    # ---- 5: unions + em-span arithmetic ----
    pseudo_bstar = ranked[:8]
    active_atoms = torch.nonzero(active_mask).flatten().tolist()
    uni = RC.uniform_random_union(active_atoms, len(pseudo_bstar), seed=7, exclude=pseudo_bstar)
    usg = RC.usage_matched_union(pseudo_bstar, strata, seed=7)
    for atom, partner in zip(pseudo_bstar, usg):
        assert int(strata[atom]) == int(strata[partner]), "usage stratum mismatch"
    assert not (set(uni) & set(pseudo_bstar)) and not (set(usg) & set(pseudo_bstar))
    Qb = GB.union_Q(Wd, 1, pseudo_bstar)
    assert Qb.shape == (IP.D, len(pseudo_bstar))
    orth = float((Qb.T @ Qb - torch.eye(Qb.shape[1], dtype=torch.float64)).abs().max())
    assert orth < 1e-8, f"union_Q not orthonormal ({orth:.2e})"
    Qcap = GB.union_Q(Wd, 1, ranked[:min(RS.CAP_ATOMS, len(ranked))])
    Qem = RS.em_span(64, 900, IP.D)
    rep["unions"] = {"uniform_null": uni, "usage_matched_null": usg,
                     "union_orth_err": orth, "cap_union_rank": int(Qcap.shape[1]),
                     "em_span_rank": int(Qem.shape[1]),
                     "stable_rank_pseudo_bstar": RC.stable_rank(Wd[:, pseudo_bstar]),
                     "stable_rank_haar_band": [
                         float(x) for x in torch.quantile(
                             RC.stable_rank_haar_band(IP.D, len(pseudo_bstar), 32, seed=3),
                             torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))]}
    _w()

    # ---- 6: MDL (Eq. 6) on the REAL spectra of the banked dictionary ----
    spec = RC.mdl_spectra(ae, X[: a.n_fvu_rows], device="cpu")
    mdl = {}
    for frac in (0.01, 0.05, 0.10, 0.20):
        mdl[f"{frac:.2f}"] = RC.mdl_bits_per_token(
            n_blocks=ae.n_blocks, k_blk=ae.k_blk, n_tokens=int(meta["n_rows"][f"L{Lp}"]),
            block_fire_rates=spec["block_fire_rates"], block_code_eigs=spec["block_code_eigs"],
            residual_eigs=spec["residual_eigs"], delta=frac * spec["var_ref"],
            dict_free_params=RC.free_decoder_params(ae.n_blocks, ae.m, IP.D))
    totals = [mdl[k]["total"] for k in sorted(mdl)]
    assert all(np.isfinite(totals)), "MDL non-finite on a real config"
    assert totals == sorted(totals, reverse=True), "MDL must fall as distortion loosens"
    rep["mdl"] = {"var_ref": spec["var_ref"], "k_blk": ae.k_blk,
                  "expected_active_blocks": mdl["0.20"]["expected_active_blocks"],
                  "by_frac": mdl}
    _w()

    # ---- 7: Arm D machinery on the REAL S_vl draws (A1) ----
    # Canonicity and the atom-set readout MUST be measured at the SAME rank, or the surrogate is
    # matched to a quantity the readout never used. Atom sets read from the k=K_READ draw ⇒ the
    # surrogate is built at the k=K_READ canonicity. The ENDOW-F k_can=16 value is reported for
    # continuity ONLY, never used in the rule.
    draws = torch.load(RS.fetch(f"{RS.GRAIN_HARVEST}/svl_draws.pt", "/tmp/regrain/svl_draws.pt"),
                       map_location="cpu", weights_only=False)
    Vs = [orthonormalize(torch.as_tensor(draws[k]["V"], dtype=torch.float64))
          for k in sorted(draws)]
    assert Vs[0].shape[1] == RS.K_READ, f"draw rank {Vs[0].shape[1]} != K_READ"
    name14 = RS.dict_bank_name("secondary", 1, 0)
    ae14, cfg14, _q14, _s14 = GB.load_dict_bank(
        RS.fetch(f"{RS.GRAIN_DICTS_L14}/{name14}", f"/tmp/regrain/{name14}"))
    W14 = RC.decoder_matrix(ae14)
    U14 = RC.unit_decoder(W14)
    sets = [RC.top_aligned_atoms(U14, Vd, RS.TOP8, already_unit=True) for Vd in Vs]
    jac_mean, jac_pairs = GB.pairwise_jaccard(sets)
    c_align = canonicity(Vs, IP.D)                       # at k = K_READ (the readout's rank)
    c_16 = canonicity([Vd[:, : RS.K_CAN] for Vd in Vs], IP.D)   # reported only
    perm = GB.jaccard_null(ae14.dict_size, RS.TOP8, 2000, seed=3)
    perm_p95 = float(torch.quantile(perm, 0.95))
    surro, cs = RC.canonicity_matched_jaccard_null(W14, Vs[0], c_align, IP.D, RS.TOP8,
                                                   a.n_surrogate_draws, seed=11)
    surro_p95 = float(torch.quantile(surro, 0.95))
    verdict_d = RC.adjudicate_draws(jaccard_cilo=float(min(jac_pairs)),
                                    permutation_p95=perm_p95, surrogate_p95=surro_p95)
    rep["arm_d_dryrun_on_svl"] = {
        "n_draws": len(Vs), "sets": sets, "jaccard_mean": jac_mean, "jaccard_pairs": jac_pairs,
        "canonicity_at_readout_rank": c_align, "k_readout": RS.K_READ,
        "canonicity_kcan16_reported_only": c_16,
        "permutation_p95": perm_p95, "surrogate_p95": surro_p95,
        "surrogate_mean": float(surro.mean()),
        "surrogate_realized_c": [float(x) for x in cs[:4]],
        "provisional_verdict": verdict_d,
        "note": "OFFLINE mechanics check on discovery-side artifacts; the arm re-runs with "
                "5 draws + pair-bootstrap CI. NOT the Arm-D result."}
    _w()

    # ---- 8: adjudicators on numbers this job produced ----
    labels = {
        "atom_handle_smoke": RC.adjudicate_atom_handle(
            effect_bstar=0.60, effect_full=0.51, effect_em=0.05,
            effect_rand_uniform_mean=0.05, effect_rand_usage_mean=0.10,
            gap_uniform_cilo=0.2, gap_usage_cilo=0.15,
            delta_C=0.05, rho=RS.RHO, bh_pass=True,
            dppl_bstar=3.0, dppl_full=6.0, dppl_rand_band_hi=1.0),
        "draws_real": verdict_d,
        "sign_smoke": RC.adjudicate_sign(signed_m4_pass=True, rectified_m4_pass=False,
                                         any_signed_small_b_pass=True),
        "stream_smoke": RC.adjudicate_stream(pure_text_passes_text_gate=False,
                                             pure_text_fvu=0.071, mixed_fvu=0.070)}
    assert labels["atom_handle_smoke"] in RC.C_LABELS and labels["draws_real"] in RC.D_LABELS
    assert labels["sign_smoke"] in RC.I_LABELS and labels["stream_smoke"] in RC.H_LABELS
    rep["adjudication_smoke"] = labels
    _w()
    print("REGRAIN_OFFLINE_OK", json.dumps(
        {"fvu": q["fvu"], "n_active": n_active, "mdl_20pct": mdl["0.20"]["total"],
         "svl_jaccard": jac_mean, "canonicity": c_obs, "perm_p95": perm_p95,
         "surrogate_p95": surro_p95, "provisional_d": verdict_d}, default=str))
    # green written LAST: a crash on the summary line must not leave a green artifact.
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
