"""HANDLE Arm T (GPU) — the demonstration: is a compact core a portable, task-specific module?

Three legs on the 6 discovery FTs (K* frozen from Arm H; under H_set-property, cores = random
rank-K*=50 subspaces of the stiff space — the pre-specified set-property module definition):
  (a) module matrix  M[i][j] = gain of core_{K*}(FT_i) evaluated on task_j (FT_j's assigned task),
                     + a Haar-core floor row (random rank-K* projection, per FT) as the column floor.
  (b) deletion       FT_i − core_i (the residual cell): own-task drop vs ppl-drop vs non-target drop.
  (c) core census    per-FT core directions at the primary site (0, o_proj) = top-r left singular
                     vectors of the projected core ΔW; pairwise principal angles + subspace overlap
                     vs U_wr / U_unt, against the WITHIN-STIFF matched-rank Haar null (H-A5).

Feeds handle_core.adjudicate_rt. Swap-shell reused from comwrite_surgery; Q eigvecs from carry_scope;
U_wr/U_unt from specter_o library_bank. One FT snapshot on disk at a time.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_t --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/handle_t --budget-min 420 --args "--ladder-prefix results/handle_h_sealed"
"""
import argparse
import hashlib
import json
import os
import pathlib

CENSUS_SITE = (0, "o_proj")        # where U_wr_o_proj / U_unt_residual are banked (L0)
R_CENSUS = 16                      # matched census rank (min with K* applied at runtime)


def _floor_seed(slug):
    return int(hashlib.sha1(f"handle_haarcore|{slug}".encode()).hexdigest()[:8], 16)


def _setprop_seed(slug):
    return int(hashlib.sha1(f"handle_setprop|{slug}".encode()).hexdigest()[:8], 16)


def _gain(res_cell, res_base):
    from infra.pg import carry_core as CC
    if res_cell["per_item"] is not None:
        return CC.paired_gain(res_cell["per_item"], res_base["per_item"])
    se = ((res_cell["ci95"][1] - res_cell["ci95"][0]) ** 2
          + (res_base["ci95"][1] - res_base["ci95"][0]) ** 2) ** 0.5 / (2 * 1.96)
    return {"gain": res_cell["score"] - res_base["score"], "se": se}


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder-prefix", default="results/handle_h_sealed",
                    help="S3 prefix of the sealed Arm-H metrics (for the R-H verdict + K*)")
    ap.add_argument("--kstar", type=int, default=0, help="override K* (0 = derive from ladder)")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.pg import bispec_core as BC
    from infra.pg import carry_core as CC
    from infra.pg import comwrite_core as CoreC
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import handle_core as HC
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import build_tasks, random_basis

    freeze = HS.assert_frozen("t")
    HS.install_pg()
    rep = {"object": "handle_t_v1", "freeze": freeze}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/handle_t"
    pathlib.Path(wd).mkdir(parents=True, exist_ok=True)
    fts = HS.load_carry_fts(wd)
    qb = HS.load_q_bank(wd)
    k_stiff = int(qb["k_stiff"])
    ks = HC.ladder_ks(k_stiff)
    evecs = torch.as_tensor(qb["evecs"])

    # ---- K* from the sealed Arm-H verdict (or override) ---------------------------------
    ladder = json.loads(HS.fetch(f"{HS.S3_BUCKET}/{a.ladder_prefix}/out/metrics.json",
                                 pathlib.Path(wd) / "ladder.json").read_text())
    assert ladder.get("green"), "sealed Arm-H not green"
    rh = ladder["rh"]
    setprop = rh["verdict"] == "H_set-property"
    kstar = a.kstar or HC.select_kstar(rh, ks)
    r_cen = min(R_CENSUS, kstar)
    rep.update({"k_stiff": k_stiff, "kstar": kstar, "rh_verdict": rh["verdict"],
                "setprop_module_def": setprop, "r_census": r_cen})
    _w()

    disc = fts["discovery"]
    tnames = sorted({fts["assigned_task"][s] for s in disc})       # {vqav2, textvqa}

    # ---- U_wr / U_unt from the SPECTER library bank (matched rank) -----------------------
    lib = torch.load(HS.fetch(f"{HS.SPECTER_O}/library_bank.pt", pathlib.Path(wd) / "library_bank.pt"),
                     map_location="cpu", weights_only=False)
    U_wr = torch.as_tensor(lib["U_wr_o_proj"], dtype=torch.float64)[:, :r_cen].contiguous()
    U_unt = torch.as_tensor(lib["U_unt_residual"], dtype=torch.float64)[:, :r_cen].contiguous()

    from transformers import PaliGemmaForConditionalGeneration  # noqa: F401  (via HS.ft_decoder_sd)
    with tm.stage("load_base"):
        model, proc = IP.load_pg1(sha=fts["base_sha"])
        paths = resolve_submodule_paths(model)
        lm = model.get_submodule(paths["language_model"])
        dev = next(model.parameters()).device
        base_lm_sd = {k: v.detach().clone().cpu() for k, v in lm.state_dict().items()}
        vt_orig = {k: v.detach().clone().cpu()
                   for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
        g0 = json.loads(HS.fetch(f"{HS.HANDLE_GATE0}/gate0.json", pathlib.Path(wd) / "g0.json").read_text())
        assert g0.get("green"), "handle gate0 not green"
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        assert abs(pp["ppl"] - g0["ppl_anchor"]["ppl"]) <= 0.5, "base ppl drifted vs gate0"
        rep["base_ppl"] = pp["ppl"]
    _w()

    grid = SURG.grid_of(base_lm_sd)
    SURG.assert_full_grid(grid)
    tasks = {t.name: t for t in build_tasks()}
    census_key = grid[CENSUS_SITE]

    def install(cell_sd, tag):
        missing, unexpected = lm.load_state_dict({k: v.to(dev) for k, v in cell_sd.items()}, strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt[k].detach().cpu(), vt_orig[k]) for k in vt_orig), f"{tag}: LEAK"

    rows_cache, base_res, floor_res = {}, {}, {}
    with tm.stage("base_floor"):
        for tname in tnames:
            hashes, nstream = HS.load_handle_items(wd, tname, "disc", n=a.n_items)
            rows_cache[tname] = HS.rows_for_banked(tasks[tname], hashes, nstream)
            base_res[tname] = IP.eval_pg1(model, proc, tasks[tname], rows_cache[tname], "cuda")
            floor_res[tname] = IP.eval_pg1(model, proc, tasks[tname], rows_cache[tname], "cuda",
                                           scramble_seed=7)
    _w()

    def core_bases(slug):
        """core_{K*}(FT) basis dict: top-K* Q eigvecs (default) or a random-within-stiff K* subspace
        under the H_set-property module definition."""
        if setprop:
            R = random_basis(k_stiff, kstar, _setprop_seed(slug), "cpu", torch.float32)
            U = evecs[:, :k_stiff].contiguous() @ R                    # random K* directions in stiff
            U, _ = torch.linalg.qr(U)
        else:
            U = evecs[:, :kstar].contiguous()
        return HS.rung_bases(U, grid), U

    def eval_cell(cellname, bases, task, rows):
        sd, book = SURG.build_decoder(cellname, base_lm_sd, ft_sd, bases)
        install(sd, cellname); del sd
        return IP.eval_pg1(model, proc, task, rows, "cuda"), book

    M = {}
    deletion_rows, core_dirs, haar_floor = [], {}, {}
    for slug in disc:
        own = fts["assigned_task"][slug]
        nontarget = next(t for t in tnames if t != own) if len(tnames) > 1 else own
        with tm.stage(f"ft_{slug}"):
            ft_sd = HS.ft_decoder_sd(fts["repo_of"][slug], fts["shas"][fts["repo_of"][slug]])
            cbases, Ucore = core_bases(slug)
            # (a) module row: core_i on every discovery task
            M[slug] = {}
            for tname in tnames:
                res, _b = eval_cell("shared", cbases, tasks[tname], rows_cache[tname])
                M[slug][tname] = _gain(res, base_res[tname])["gain"]
            # Haar-core floor (random rank-K* projection of THIS FT's ΔW), on the own task
            Rf = random_basis(2048, kstar, _floor_seed(slug), "cpu", torch.float32)
            hres, _b = eval_cell("shared", HS.rung_bases(Rf, grid), tasks[own], rows_cache[own])
            haar_floor[slug] = _gain(hres, base_res[own])["gain"]
            # (c) census directions at (0,o_proj): top-r left singular vecs of the projected core ΔW
            dW = ft_sd[census_key].detach().float() - base_lm_sd[census_key].float()
            core_dW = SURG.project_delta(dW, Ucore, CENSUS_SITE[1]).double()
            Usvd, _S, _V = torch.linalg.svd(core_dW, full_matrices=False)
            core_dirs[slug] = Usvd[:, :r_cen].contiguous()
            # (b) deletion: the residual cell (FT − core), own-task + non-target + ppl
            del_res_own, _b = eval_cell("residual", cbases, tasks[own], rows_cache[own])
            del_res_nt, _b = eval_cell("residual", cbases, tasks[nontarget], rows_cache[nontarget])
            pg = HS.safe_ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
            ft_res_own, _b = eval_cell("ft", cbases, tasks[own], rows_cache[own])
            ft_res_nt, _b = eval_cell("ft", cbases, tasks[nontarget], rows_cache[nontarget])
            own_drop = _gain(ft_res_own, base_res[own])["gain"] - _gain(del_res_own, base_res[own])["gain"]
            nt_drop = abs(_gain(ft_res_nt, base_res[nontarget])["gain"]
                          - _gain(del_res_nt, base_res[nontarget])["gain"])
            ppl_drop = abs(HS.safe_log1p_dppl(pg["mean_nll"], rep["base_ppl"]))
            deletion_rows.append({"ft": slug, "own_drop": own_drop, "ppl_drop": ppl_drop,
                                  "nontarget_drop": nt_drop})
            HS.drop_snapshot(fts["repo_of"][slug]); del ft_sd
            rep["module_matrix"] = M
            rep["deletion_rows"] = deletion_rows
            rep["haar_core_floor"] = haar_floor
            print(f"[handle-t] {slug}: own_drop={own_drop:+.4f} ppl_drop={ppl_drop:.4f} "
                  f"nt_drop={nt_drop:.4f}", flush=True)
            _w()
    lm.load_state_dict({k: v.to(dev) for k, v in base_lm_sd.items()}, strict=True)

    # ---- census statistics (zero-forward) ----------------------------------------------
    with tm.stage("census"):
        slugs = list(disc)
        frames = [core_dirs[s] for s in slugs]
        gd = BC.grassmann_diameter(frames)
        pair_overlaps = [CoreC.subspace_overlap(core_dirs[slugs[i]].numpy(),
                                                 core_dirs[slugs[j]].numpy())
                         for i in range(len(slugs)) for j in range(i + 1, len(slugs))]
        import numpy as np
        cross_overlap_med = float(np.median(pair_overlaps))
        # within-stiff matched-rank Haar null (H-A5): random r-frames inside span(U_stiff)
        Ust = evecs[:, :k_stiff].double()
        haar_ov = []
        for n in range(200):
            Rs = [Ust @ torch.linalg.qr(torch.randn(k_stiff, r_cen,
                  generator=torch.Generator().manual_seed(1000 + n * 2 + t), dtype=torch.float64))[0]
                  for t in range(2)]
            haar_ov.append(CoreC.subspace_overlap(Rs[0].numpy(), Rs[1].numpy()))
        haar_p95 = float(np.quantile(haar_ov, 0.95))
        uwr_ov = float(np.median([CoreC.subspace_overlap(core_dirs[s].numpy(), U_wr.numpy())
                                  for s in slugs]))
        uunt_ov = float(np.median([CoreC.subspace_overlap(core_dirs[s].numpy(), U_unt.numpy())
                                   for s in slugs]))
        census = {"cross_overlap_med": cross_overlap_med, "haar_p95": haar_p95,
                  "core_uwr_overlap_med": uwr_ov, "core_uunt_overlap_med": uunt_ov,
                  "mean_min_angle": gd["mean_min_angle"], "n_pairs": gd["n_pairs"]}
        rep["census"] = census
    _w()

    # ---- module matrix as a numpy 6×6 + R-T verdict ------------------------------------
    import numpy as np
    Mmat = np.array([[M[i][fts["assigned_task"][j]] for j in disc] for i in disc])
    rep["module_matrix_np"] = Mmat.tolist()
    rep["rt"] = HC.adjudicate_rt(module_M=Mmat, deletion_rows=deletion_rows, census=census,
                                 n_disc_void=0)
    HS.assert_gpu_seconds(rep)
    print("HANDLE_T_OK", json.dumps({"verdict": rep["rt"]["verdict"], "kstar": kstar}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
