"""GRAIN S1+S2 (CPU, banks only — no model forward) — the structure atlas + the canonicity
re-read at block grain.

S1 (descriptive, seed-banded): per banked m* dictionary (both sites, all passing seeds):
per-block stable rank of the ACTIVATION-WEIGHTED decoder block, activation frequency,
position-type purity, pairwise block-subspace overlaps (top-used blocks) with the Haar floor.
No verdict; frozen summary statistics only. Per-block tables land as JSON (parquet substitute,
disclosed — pandas/pyarrow are not pinned on the AMI).

S2 (adjudicated on S_vl; descriptive on S_action if <3 banked draws — the Gate-0 §C branch):
per re-estimation draw of the read, the top-8 aligned blocks from the ONE frozen m* seed-0
dictionary at the read's home site; block-set Jaccard across draws vs the permutation null;
the parent subspace-overlap canonicity recomputed on the SAME draws (k_can=16, the ENDOW-F
convention); frozen rule = grain_bsf.adjudicate_s2.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_s_atlas --type r7i.4xlarge --no-weights \
      --s3-prefix results/grain_s_atlas --budget-min 120 --args "\
      --train-primary-s3 .../grain_train_primary/out --train-secondary-s3 .../grain_train_secondary/out \
      --harvest-s3 .../grain_harvest/out"
"""
import argparse
import json
import pathlib

TOP8 = 8
K_CAN = 16
N_NULL = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-primary-s3", required=True)
    ap.add_argument("--gates-primary-s3", default=None)
    ap.add_argument("--train-secondary-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--n-usage-rows", type=int, default=60000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg.ctrliface_core import matched_random_overlap, principal_angle_overlap
    from infra.pg.endow_atlas import canonicity

    banks = GS.load_read_banks()
    meta = json.loads(GS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/grain/harvest_meta.json").read_text())
    rep = {"object": "grain_s_atlas_v1", "sites": {}, "s2": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    dicts = {}                      # (site_name, seed) → ae
    gates_by_site = {}
    for site_name, s3 in (("primary", a.train_primary_s3), ("secondary", a.train_secondary_s3)):
        gsrc = a.gates_primary_s3 if (site_name == "primary" and a.gates_primary_s3) else s3
        gates = json.loads(GS.fetch(f"{gsrc}/gates.json", f"/tmp/grain/gates_{site_name}.json").read_text())
        gates_by_site[site_name] = gates
        m_star = gates.get("m_star")
        if not m_star:
            rep["sites"][site_name] = {"state": "UNRUN", "pivot": gates.get("pivot")}
            continue
        L = gates["L"]
        X, labs = GS.load_shards(a.harvest_s3, meta, f"L{L}",
                                 workdir=f"/tmp/grain/shards_{site_name}")
        X, labs = X[: a.n_usage_rows], labs[: a.n_usage_rows]
        haar = matched_random_overlap(2048, m_star, m_star, 300, 11)
        site_rep = {"L": L, "m_star": m_star, "seeds": {}, "haar_overlap":
                    {"mean": float(haar.mean()), "p95": float(torch.quantile(haar, 0.95))}}
        for seed in (0, 1, 2):
            cell = gates["configs"].get(f"m{m_star}_s{seed}", {})
            if not cell.get("gate_pass"):
                continue
            name = GS.dict_bank_name(gates["site"], m_star, seed)
            ae, cfg, q, msha = GB.load_dict_bank(GS.fetch(f"{s3}/{name}", f"/tmp/grain/{name}"))
            assert msha == gates["manifest_sha"]
            dicts[(site_name, seed)] = ae
            usage = GB.block_usage(ae, X, labs.long(), "cpu")
            Wd = ae.decoder.weight.detach()
            sr = GB.block_stable_rank(Wd, m_star, usage["latent_mean_act"])
            freq = usage["freq"]
            active = torch.nonzero(freq > 0).flatten()
            counts = usage["counts"]
            purity = torch.zeros(ae.n_blocks, 3, dtype=torch.float64)
            tot = counts.sum(1, keepdim=True).clamp_min(1e-12)
            purity = counts / tot
            # pairwise subspace overlaps for the top-used blocks (Haar floor overlaid)
            top_used = torch.topk(freq, min(48, int(active.numel()))).indices.tolist()
            ovs = []
            for i in range(len(top_used)):
                for j in range(i + 1, len(top_used)):
                    ov, _ = principal_angle_overlap(
                        GB.block_subspace(Wd, m_star, top_used[i]),
                        GB.block_subspace(Wd, m_star, top_used[j]))
                    ovs.append(ov)
            act_sr = sr[active]
            site_rep["seeds"][str(seed)] = {
                "n_active": int(active.numel()),
                "stable_rank": {"median": float(act_sr.median()),
                                "q25": float(torch.quantile(act_sr, 0.25)),
                                "q75": float(torch.quantile(act_sr, 0.75))},
                "pairwise_overlap": {"mean": float(np.mean(ovs)), "p95":
                                     float(np.percentile(ovs, 95))},
                "purity_image_gt80": int(((purity[:, GS.POS_IMAGE] > 0.8)
                                          & (freq > 0)).sum()),
                "purity_text_gt80": int(((purity[:, GS.POS_TEXT] > 0.8)
                                         & (freq > 0)).sum()),
                "stable_rank_by_type": {
                    "image_pure": float(sr[(purity[:, 0] > 0.8) & (freq > 0)].median())
                    if bool(((purity[:, 0] > 0.8) & (freq > 0)).any()) else None,
                    "text_pure": float(sr[(purity[:, 1] > 0.8) & (freq > 0)].median())
                    if bool(((purity[:, 1] > 0.8) & (freq > 0)).any()) else None},
            }
            (out / f"blocks_{site_name}_s{seed}.json").write_text(json.dumps(
                {"stable_rank": [float(x) for x in sr], "freq": [float(x) for x in freq],
                 "purity": purity.tolist()}, indent=None))
            _w()
        rep["sites"][site_name] = site_rep
        _w()

    # ---------------- S2: canonicity re-read ----------------
    # S_vl (adjudicated): 3 re-estimation draws from the T1 stage-2 bank, blocks from the
    # frozen SECONDARY m* seed-0 dictionary.
    if ("secondary", 0) in dicts:
        ae = dicts[("secondary", 0)]
        m_star = gates_by_site["secondary"]["m_star"]
        Wd = ae.decoder.weight.detach()
        draws = torch.load(GS.fetch(f"{a.harvest_s3}/svl_draws.pt", "/tmp/grain/svl_draws.pt"),
                           map_location="cpu", weights_only=False)
        Ls = gates_by_site["secondary"]["L"]
        X, labs = GS.load_shards(a.harvest_s3, meta, f"L{Ls}",
                                 workdir="/tmp/grain/shards_secondary")
        usage = GB.block_usage(ae, X[:a.n_usage_rows], labs[:a.n_usage_rows].long(), "cpu")
        n_active = int((usage["freq"] > 0).sum())
        sets, Vs = [], []
        from infra.pg.ctrliface_core import orthonormalize
        for di in sorted(draws):
            V = orthonormalize(torch.as_tensor(draws[di]["V"], dtype=torch.float64))
            Vs.append(V)
            align = GB.block_alignment(Wd, m_star, V)
            sets.append(torch.topk(align, TOP8).indices.tolist())
        jac_mean, jac_vals = GB.pairwise_jaccard(sets)
        # bootstrap-free CI substitute: the pairwise values' min as the conservative CI-lo
        null = GB.jaccard_null(n_active, TOP8, N_NULL, seed=3)
        null_p95 = float(torch.quantile(null, 0.95))
        sub_can = canonicity([V[:, :K_CAN] for V in Vs], 2048)
        verdict = GB.adjudicate_s2(jaccard_cilo=min(jac_vals), null_p95=null_p95,
                                   subspace_canonicity=sub_can)
        rep["s2"]["svl"] = {"adjudicated": True, "sets": sets, "jaccard_mean": jac_mean,
                            "jaccard_pairs": jac_vals, "jaccard_cilo_rule": "min-pair",
                            "null_p95": null_p95, "n_active": n_active,
                            "subspace_canonicity_kcan16": sub_can, "verdict": verdict}
        _w()

    # S_action (descriptive if <3 draws): bases.pt + bases_full.pt at the primary site.
    # Pivot fallback (journal E7): if the primary has no faithful m* dictionary, the block
    # sets come from the FAITHFUL grain-1 dictionary (m=1 blocks = single latents; labeled).
    sa_dict = None
    if ("primary", 0) in dicts:
        sa_dict = (dicts[("primary", 0)], gates_by_site["primary"]["m_star"], "m_star_blocks")
    elif gates_by_site["primary"]["configs"].get("m1_s0", {}).get("gate_pass"):
        name = GS.dict_bank_name(gates_by_site["primary"]["site"], 1, 0)
        ae1, _c, _q, _m = GB.load_dict_bank(GS.fetch(f"{a.train_primary_s3}/{name}",
                                                     f"/tmp/grain/{name}"))
        sa_dict = (ae1, 1, "grain1_latents (no faithful m* at L0 — E7 pivot)")
    if sa_dict is not None:
        ae, m_star, sa_label = sa_dict
        Wd = ae.decoder.weight.detach()
        Lp = gates_by_site["primary"]["L"]
        from infra.aws_pg.jobs._inherit_shared import load_bank, load_redraw_bases
        bases, _h, _t = load_bank("/tmp/grain")
        from infra.pg.inherit_core import slice_k
        draws_sa = [slice_k(bases[Lp], GS.K_READ)]
        import subprocess
        if subprocess.run(["aws", "s3", "ls", GS.FULL_BANK, "--region", "us-east-1"],
                          capture_output=True).returncode == 0:
            redraw = load_redraw_bases(GS.FULL_BANK, "/tmp/grain")
            draws_sa.append(redraw[Lp])
        X, labs = GS.load_shards(a.harvest_s3, meta, f"L{Lp}",
                                 workdir="/tmp/grain/shards_primary")
        usage = GB.block_usage(ae, X[:a.n_usage_rows], labs[:a.n_usage_rows].long(), "cpu")
        n_active = int((usage["freq"] > 0).sum())
        sets = []
        for V in draws_sa:
            align = GB.block_alignment(Wd, m_star, V)
            sets.append(torch.topk(align, TOP8).indices.tolist())
        entry = {"n_draws": len(draws_sa), "adjudicated": len(draws_sa) >= 3, "sets": sets,
                 "n_active": n_active, "dictionary": sa_label, "m": int(m_star)}
        if len(draws_sa) >= 2:
            jm, jv = GB.pairwise_jaccard(sets)
            null = GB.jaccard_null(n_active, TOP8, N_NULL, seed=5)
            entry.update({"jaccard_mean": jm, "jaccard_pairs": jv,
                          "null_p95": float(torch.quantile(null, 0.95)),
                          "subspace_canonicity_kcan16":
                          canonicity([V[:, :K_CAN] for V in draws_sa], 2048)})
        if not entry["adjudicated"]:
            entry["note"] = ("<3 banked S_action draws — DESCRIPTIVE only (Gate-0 §C branch); "
                             "the adjudicated S2 rule runs on S_vl")
        rep["s2"]["saction"] = entry

    rep["green"] = True
    _w()
    print("S_ATLAS_OK", json.dumps({"s2": {k: v.get("verdict", v.get("jaccard_mean"))
                                           for k, v in rep["s2"].items()}}, default=str))


if __name__ == "__main__":
    main()
