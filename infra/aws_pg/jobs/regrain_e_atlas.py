"""REGRAIN Arm E + Arm C3 (CPU, banks only) — internal geometry, matched-null disciplined.

C3 (the handle's geometry, H_tiling leg 1):
  * stable rank of the B* decoder matrix vs a matched Haar band of |B*| random UNIT columns
    (NOT an orthonormal basis — that has stable rank exactly |B*| and the band would be degenerate)
  * position-type purity of the B* atoms
  * contribution-PCA spectrum of the B*-union (the coordinates Arm F traverses)

E (the atlas):
  * per-block CODE-covariance rank triplet — stable rank, participation ratio, effective rank
    (the paper's C.4) — and utilization = effective rank / b, for every gate-passing dictionary
  * usage frequency + position-type purity by block
  * seed canonicity: Hungarian matching of the top-used decoder atoms across m=1 seeds, against a
    Haar floor computed the SAME way

Every geometry statistic is reported beside its **matched-noise null** (App. E design): the SAME
trained featurizer run on coordinate-shuffled activations, which preserve each coordinate's
marginal exactly and destroy every cross-coordinate correlation. No geometry panel ships without
its null. No verdicts here — frozen summary statistics and figures.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_e_atlas --type r7i.8xlarge,r7i.4xlarge \
      --no-weights --s3-prefix results/regrain_e --budget-min 150 --args "\
      --harvest-s3 s3://…/regrain_harvest/out --cdisc-s3 s3://…/regrain_c_disc/out \
      --factorial-s3 s3://…/regrain_i_factorial/out"
"""
import argparse
import json
import pathlib

TOP_MATCH = 512          # atoms entering the Hungarian seed-canonicity match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--cdisc-s3", default=None)
    ap.add_argument("--factorial-s3", default=None)
    ap.add_argument("--n-rows", type=int, default=8192)
    ap.add_argument("--n-usage-rows", type=int, default=30000)
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

    rep = {"object": "regrain_e_atlas_v1", "n_rows": a.n_rows}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    banks = RS.load_read_banks("/tmp/regrain")
    Lp = banks["saction"]["L"]
    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    key = f"L{Lp}"
    X, labs, _st = RS.load_shards_stream(a.harvest_s3, meta, key, workdir="/tmp/regrain/shards")
    x_scale = float(meta["x_scale"][key])
    Xa = X[: a.n_rows]
    Xnull = RC.shuffled_null_rows(Xa.float(), seed=99).half()
    rep["matched_null"] = {"kind": "coordinate-shuffle (App. E design)", "seed": 99,
                           "note": "per-coordinate marginals preserved exactly; all "
                                   "cross-coordinate structure destroyed"}
    _w()

    def dict_paths():
        """Every dictionary this atlas covers: the three m=1 seeds + every I2 config."""
        paths = {}
        n0 = RS.dict_bank_name("primary", 1, 0)
        paths["m1_s0"] = (f"{RS.GRAIN_DICTS_L0}/{n0}", 1, 1.0)
        if a.cdisc_s3:
            cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json",
                                     "/tmp/regrain/c_disc.json").read_text())
            for s in cd.get("seeds_passing_gate", []):
                if int(s) == 0:
                    continue
                nm = RS.dict_bank_name("primary", 1, int(s))
                paths[f"m1_s{s}"] = (f"{a.cdisc_s3}/{nm}", 1, 1.0)
        if a.factorial_s3:
            fg = json.loads(RS.fetch(f"{a.factorial_s3}/gates.json",
                                     "/tmp/regrain/i_gates.json").read_text())
            for cid, cell in fg["configs"].items():
                if cell.get("state") == "ok":
                    paths[cid] = (f"{a.factorial_s3}/{cell['bank']}", cell["b"],
                                  float(fg["x_scale"]))
        return paths

    # ---------------- E: the atlas ----------------
    atlas, seed_dicts = {}, {}
    for cid, (src, b, xs) in dict_paths().items():
        local = f"/tmp/regrain/atlas_{cid}.pt"
        ae, cfg, q, _m = RC.load_bank_any(RS.fetch(src, local))
        inner = ae.ae if isinstance(ae, RC.ScaledAE) else ae
        s_eff = float(cfg.get("x_scale", 1.0))
        spec = RC.mdl_spectra(inner, Xa, device="cpu", x_scale=s_eff)
        spec_n = RC.mdl_spectra(inner, Xnull, device="cpu", x_scale=s_eff)

        def triplets(sp):
            eigs = sp["block_code_eigs"]
            live = sp["block_fire_rates"] > 0
            tri = [RC.rank_triplet(eigs[i]) for i in torch.nonzero(live).flatten().tolist()
                   if float(eigs[i].sum()) > 0]
            if not tri:
                return None, 0
            arr = {k: np.asarray([t[k] for t in tri]) for k in tri[0]}
            return {k: {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                        "q75": float(np.percentile(v, 75)), "mean": float(v.mean())}
                    for k, v in arr.items()}, len(tri)

        real, n_live = triplets(spec)
        null, _ = triplets(spec_n)
        util = None
        if real is not None and b > 1:
            util = real["effective_rank"]["median"] / b
        usage = GB.block_usage(ae, X[: a.n_usage_rows], labs[: a.n_usage_rows].long(), "cpu")
        counts = usage["counts"]
        purity = counts / counts.sum(1, keepdim=True).clamp_min(1e-12)
        freq = usage["freq"]
        live = freq > 0
        atlas[cid] = {
            "b": b, "n_blocks": int(inner.n_blocks), "n_live_blocks": int(n_live),
            "fvu": q.get("fvu") if isinstance(q, dict) else None,
            "rank_triplet": real, "rank_triplet_matched_noise_null": null,
            "utilization_eff_over_b": util,
            "purity_image_gt80": int(((purity[:, RS.POS_IMAGE] > 0.8) & live).sum()),
            "purity_text_gt80": int(((purity[:, RS.POS_TEXT] > 0.8) & live).sum()),
            "mean_fire_rate": float(spec["block_fire_rates"][live].mean())}
        if b == 1 and cid.startswith("m1_s"):
            seed_dicts[cid] = (RC.decoder_matrix(ae).clone(), freq.clone())
        rep["atlas"] = atlas
        _w()
        del ae, inner
    _w()

    # ---------------- E: seed canonicity vs a Haar floor, same estimator ----------------
    if len(seed_dicts) >= 2:
        ids = sorted(seed_dicts)
        pairs = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                Wa, fa = seed_dicts[ids[i]]
                Wb, fb = seed_dicts[ids[j]]
                ta = torch.topk(fa, TOP_MATCH).indices
                tb = torch.topk(fb, TOP_MATCH).indices
                pairs[f"{ids[i]}|{ids[j]}"] = RC.hungarian_seed_canonicity(Wa[:, ta], Wb[:, tb])
        g = torch.Generator().manual_seed(5)
        haar = []
        for t in range(3):
            A = torch.randn(IP.D, TOP_MATCH, generator=g, dtype=torch.float64)
            B = torch.randn(IP.D, TOP_MATCH, generator=g, dtype=torch.float64)
            haar.append(RC.hungarian_seed_canonicity(A, B))
        rep["seed_canonicity"] = {"pairs": pairs, "mean": float(np.mean(list(pairs.values()))),
                                  "haar_floor_mean": float(np.mean(haar)), "haar_draws": haar,
                                  "n_atoms_matched": TOP_MATCH}
        _w()

    # ---------------- C3: the handle's geometry ----------------
    if a.cdisc_s3:
        cd = json.loads(RS.fetch(f"{a.cdisc_s3}/c_disc.json", "/tmp/regrain/c_disc.json").read_text())
        bstar = [int(x) for x in cd["bstar"]]
        med_s = int(cd["median_seed"])
        nm = RS.dict_bank_name("primary", 1, med_s)
        src = f"{RS.GRAIN_DICTS_L0}/{nm}" if med_s == 0 else f"{a.cdisc_s3}/{nm}"
        ae, _c, _q, _m = GB.load_dict_bank(RS.fetch(src, f"/tmp/regrain/med_{nm}"))
        Wd = RC.decoder_matrix(ae)
        Ub = RC.unit_decoder(Wd)[:, bstar]
        sr = RC.stable_rank(Ub)
        band = RC.stable_rank_haar_band(IP.D, len(bstar), 64, seed=3)
        usage = GB.block_usage(ae, X[: a.n_usage_rows], labs[: a.n_usage_rows].long(), "cpu")
        counts = usage["counts"]
        purity = counts / counts.sum(1, keepdim=True).clamp_min(1e-12)

        # contribution cloud of the B* union → PCA spectrum (the coordinates Arm F traverses)
        with torch.inference_mode():
            xs = Xa.float()
            f = ae.encode(xs, use_threshold=True)
            contrib = f[:, bstar] @ Ub.float().T                     # (n, d)
            fired = (f[:, bstar].abs().sum(1) > 0)
            Cc = contrib[fired].double()
            Cc = Cc - Cc.mean(0)
            evals = torch.linalg.svdvals(Cc) ** 2 / max(1, Cc.shape[0])
        # matched-noise null for the SAME statistic
        with torch.inference_mode():
            fn = ae.encode(Xnull.float(), use_threshold=True)
            cn = fn[:, bstar] @ Ub.float().T
            firedn = (fn[:, bstar].abs().sum(1) > 0)
            Cn = cn[firedn].double()
            Cn = Cn - Cn.mean(0) if Cn.shape[0] > 1 else Cn
            evals_n = (torch.linalg.svdvals(Cn) ** 2 / max(1, Cn.shape[0])
                       if Cn.shape[0] > 1 else torch.zeros(1, dtype=torch.float64))

        rep["c3_geometry"] = {
            "bstar_len": len(bstar), "median_seed": med_s,
            "stable_rank_bstar_decoders": sr,
            "stable_rank_haar_band": {"p05": float(torch.quantile(band, 0.05)),
                                      "median": float(torch.quantile(band, 0.5)),
                                      "p95": float(torch.quantile(band, 0.95))},
            "tiling_leg1_rule": "stable_rank(B* decoders) <= 8",
            "tiling_leg1_holds": bool(sr <= 8.0),
            "tiling_leg1_beats_haar": bool(sr < float(torch.quantile(band, 0.05))),
            "purity_bstar": {"image": [float(purity[i, RS.POS_IMAGE]) for i in bstar],
                             "text": [float(purity[i, RS.POS_TEXT]) for i in bstar],
                             "bos": [float(purity[i, RS.POS_BOS]) for i in bstar]},
            "fire_rate_bstar": [float(usage["freq"][i]) for i in bstar],
            "contribution_pca": {
                "n_fired_rows": int(fired.sum()),
                "spectrum_top16": [float(x) for x in evals[:16]],
                "rank_triplet": RC.rank_triplet(evals),
                "matched_noise_null_rank_triplet": (RC.rank_triplet(evals_n)
                                                    if float(evals_n.sum()) > 0 else None)}}
        _w()

    print("REGRAIN_E_OK", json.dumps(
        {"n_dicts": len(atlas),
         "c3_stable_rank": rep.get("c3_geometry", {}).get("stable_rank_bstar_decoders"),
         "c3_haar_p05": rep.get("c3_geometry", {}).get("stable_rank_haar_band", {}).get("p05"),
         "seed_canonicity": rep.get("seed_canonicity", {}).get("mean")}, default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
