"""CARRY Arm C (GPU) — the MDE: which component of the write carries the gain.

Three phases (one job each; strict §9 order):
  --phase assign   base + floor per wired task (discovery half), then per FT the full-write
                   (`ft`) swap-shell cell read on ALL 4 wired tasks → per-FT gains → the
                   frozen A1 assignment (gain task := argmax |gain|) + the frozen §0.5-3
                   discovery selection (first 6 by ascending |gain| among range-passing).
  --phase cells    per FT (--fts, --half): the 9-cell grid on the ASSIGNED task —
                   {ft, stiff, flat, randA@s, randB@s for s in 0,1,2} — each cell scored on
                   the half's 200 banked items + the n_seq=120 ppl collateral. Discovery
                   rehearses the rule on the 6 frozen FTs; sealed (18 FTs) runs ONCE.

Surgery = comwrite_surgery.build_decoder on the PG1 language_model state-dict inside the
full multimodal model (identity-inertness asserted; vision-tower leak asserted). The split
bases come from the carry_scope Q bank (A2): stiff = `shared` cell with U_stiff, flat =
`residual` with U_stiff, randA/randB = `shared`/`residual` with a matched-rank random
subset of Q's OWN eigvecs (seed = sha1("carry_rand|{slug}|{s}"), frozen). One FT snapshot
on disk at a time; the HF cache entry is deleted after its cells complete (Arm-W lesson).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_c --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_c_assign --budget-min 420 --args "--phase assign"
  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_c --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_c_disc --budget-min 300 --args "\
      --phase cells --half disc --fts <6 slugs from the assign artifact>"
"""
import argparse
import hashlib
import json
import os
import pathlib


def _rand_seed(slug, s):
    """Frozen per-(FT, draw) seed for the random eigsplit (sha1, never hash())."""
    return int(hashlib.sha1(f"carry_rand|{slug}|{s}".encode()).hexdigest()[:8], 16)


def _slug_of(repo):
    return repo.split("-ft-")[1].rsplit("-", 1)[0]


def _eval_cell(model, proc, task, rows, floor_seed=None):
    from infra.pg import inherit_pg1 as IP
    return IP.eval_pg1(model, proc, task, rows, "cuda", scramble_seed=floor_seed)


def _gain(res_cell, res_base):
    """{gain, se} per A5: paired-item bootstrap SE; corpus tasks (per_item None) use the
    conservative unpaired combination of the two cells' resampled CIs (disclosed)."""
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
    ap.add_argument("--phase", required=True, choices=("assign", "cells"))
    ap.add_argument("--half", default="disc", choices=("disc", "sealed"))
    ap.add_argument("--fts", default="", help="comma slugs (cells phase)")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from transformers import PaliGemmaForConditionalGeneration

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import build_tasks

    arm = "c_assign" if a.phase == "assign" else f"c_{a.half}"
    freeze = CS.assert_frozen(arm)
    CS.install_pg()
    rep = {"object": f"carry_c_{a.phase}_{a.half}_v1", "freeze": freeze, "phase": a.phase,
           "half": a.half}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/carry_c"
    g0 = json.loads(CS.fetch(f"{CS.CARRY_GATE0}/gate0.json",
                             pathlib.Path(wd) / "gate0.json").read_text())
    assert g0.get("green"), "carry gate0 not green"
    anat = CS.load_roster(wd)
    assert anat["excluded_lt"] == [], \
        "CARRY splits every (L,t); a non-empty COMWRITE exclusion set needs a §0.5 note"
    qb = CS.load_q_bank(wd)
    k_stiff = int(qb["k_stiff"])
    U_stiff = qb["evecs"][:, :k_stiff].contiguous()
    rep["k_stiff"] = k_stiff

    with tm.stage("load_base"):
        model, proc = IP.load_pg1(sha=anat["base_sha"])
        paths = resolve_submodule_paths(model)
        lm = model.get_submodule(paths["language_model"])
        base_lm_sd = {k: v.detach().clone().cpu() for k, v in lm.state_dict().items()}
        vt_orig = {k: v.detach().clone().cpu()
                   for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        assert abs(pp["ppl"] - g0["ppl_anchor"]["ppl"]) <= 0.5, \
            f"base ppl drifted: {pp['ppl']:.3f} vs gate0 {g0['ppl_anchor']['ppl']:.3f}"
        rep["base_ppl"] = pp["ppl"]
    _w()

    grid = SURG.grid_of(base_lm_sd)
    SURG.assert_full_grid(grid)
    bases_stiff = {lt: U_stiff for lt in grid}

    # harness inertness (once per job — the Gate-0 assert, re-asserted before any contrast).
    with tm.stage("identity"):
        dummy_ft = {k: v.clone() for k, v in base_lm_sd.items()}
        ident_sd, _ = SURG.build_decoder("identity", base_lm_sd, dummy_ft, bases_stiff)
        assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd), \
            "identity cell not bitwise base — surgery pipeline NOT inert"
        del dummy_ft, ident_sd
        rep["identity_gate"] = "bitwise_base_ok"
    _w()

    # ---- items + base/floor per task on this phase's half ------------------------------
    tasks = {t.name: t for t in build_tasks()}
    half = "disc" if a.phase == "assign" else a.half
    rows_cache, base_res, floor_res = {}, {}, {}
    with tm.stage("base_floor"):
        for tname in CS.WIRED_TASKS:
            hashes, n_stream = CS.load_gate0_items(wd, tname, half, n=a.n_items)
            rows_cache[tname] = CS.rows_for_banked(tasks[tname], hashes, n_stream)
            base_res[tname] = _eval_cell(model, proc, tasks[tname], rows_cache[tname])
            floor_res[tname] = _eval_cell(model, proc, tasks[tname], rows_cache[tname],
                                          floor_seed=7)
            span = base_res[tname]["score"] - floor_res[tname]["score"]
            rep.setdefault("base", {})[tname] = {
                "score": base_res[tname]["score"], "ci95": base_res[tname]["ci95"],
                "floor": floor_res[tname]["score"], "span": span}
            print(f"[carry-c] base {tname}: {base_res[tname]['score']:.4f} "
                  f"floor {floor_res[tname]['score']:.4f}", flush=True)
    _w()

    dev = next(model.parameters()).device

    def install(cell_sd, tag):
        missing, unexpected = lm.load_state_dict(
            {k: v.to(dev) for k, v in cell_sd.items()}, strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt_now = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt_now[k].detach().cpu(), vt_orig[k]) for k in vt_orig), \
            f"{tag}: LEAK — vision tower changed"

    def ft_decoder_sd(repo):
        ft = PaliGemmaForConditionalGeneration.from_pretrained(
            repo, revision=anat["shas"][repo], torch_dtype=torch.bfloat16,
            device_map="cpu", attn_implementation="eager").eval()
        sd = {k: v.detach().clone()
              for k, v in ft.get_submodule(
                  resolve_submodule_paths(ft)["language_model"]).state_dict().items()}
        del ft
        return sd

    def drop_snapshot(repo):
        from infra.aws_pg.jobs.invariant_core import _delete_cache
        _delete_cache(repo)

    # =====================================================================================
    if a.phase == "assign":
        rep["assign"] = {}
        for repo in anat["roster"]:
            slug = _slug_of(repo)
            with tm.stage(f"assign_{slug}"):
                ft_sd = ft_decoder_sd(repo)
                cell_sd, book = SURG.build_decoder("ft", base_lm_sd, ft_sd, bases_stiff)
                del ft_sd
                install(cell_sd, f"{slug}/ft")
                del cell_sd
                row = {"stiff_energy_fraction":
                       book["total_captured_fro2"] / book["total_fro2"]}
                for tname in CS.WIRED_TASKS:
                    res = _eval_cell(model, proc, tasks[tname], rows_cache[tname])
                    row[tname] = {**_gain(res, base_res[tname]), "score": res["score"]}
                # the frozen A1 assignment + A5 range guard on the assigned task
                best = max(CS.WIRED_TASKS, key=lambda t: abs(row[t]["gain"]))
                row["assigned_task"] = best
                row["range_ok"] = CC.range_guard_ok(row[best]["gain"], row[best]["se"])
                rep["assign"][slug] = row
                drop_snapshot(repo)
                print(f"[carry-c] {slug}: task={best} gain={row[best]['gain']:+.4f} "
                      f"se={row[best]['se']:.4f} range_ok={row['range_ok']} "
                      f"stiff_frac={row['stiff_energy_fraction']:.3f}", flush=True)
                _w()
        # frozen §0.5-3 discovery selection: first 6 ascending |gain| among range-passing.
        passing = [(abs(rep["assign"][s][rep["assign"][s]["assigned_task"]]["gain"]), s)
                   for s in rep["assign"] if rep["assign"][s]["range_ok"]]
        passing.sort()
        rep["discovery_fts"] = [s for _g, s in passing[:CC.N_DISCOVERY_FTS]]
        rep["sealed_fts"] = sorted(s for s in rep["assign"]
                                   if s not in rep["discovery_fts"])
        rep["n_range_passing"] = len(passing)
        # H_no-range guarded FIRST (the boring branch): if fewer than 6 FTs clear the range
        # guard the corpus-level swap-shell range IS the finding — recorded, not crashed
        # (rep is banked; the sealed phase / R-C VOID-range branch confirms it).
        rep["range_finding"] = ("sufficient" if len(passing) >= CC.N_DISCOVERY_FTS
                                else "H_no-range-at-discovery")
        restore = {k: v.to(dev) for k, v in base_lm_sd.items()}
        lm.load_state_dict(restore, strict=True)

    # =====================================================================================
    else:
        assign = json.loads(CS.fetch(f"{CS.CARRY_C_ASSIGN}/metrics.json",
                                     pathlib.Path(wd) / "assign.json").read_text())
        assert assign.get("green"), "assign phase not green"
        want = [s.strip() for s in a.fts.split(",") if s.strip()]
        pool = assign["discovery_fts"] if a.half == "disc" else assign["sealed_fts"]
        assert set(want) <= set(pool), f"--fts not all in the frozen {a.half} pool"
        rep["cells"] = {}
        for slug in want:
            repo = next(r for r in anat["roster"] if _slug_of(r) == slug)
            tname = assign["assign"][slug]["assigned_task"]
            task, rows, bres = tasks[tname], rows_cache[tname], base_res[tname]
            with tm.stage(f"cells_{slug}"):
                ft_sd = ft_decoder_sd(repo)
                cells = [("ft", "ft", bases_stiff), ("stiff", "shared", bases_stiff),
                         ("flat", "residual", bases_stiff)]
                for s in range(CC.N_RAND_SEEDS):
                    U_rand, idx = CC.random_eigsplit(qb["evecs"], k_stiff,
                                                     _rand_seed(slug, s))
                    assert U_rand.shape[1] == k_stiff        # split-rank parity (§6 R-C)
                    br = {lt: U_rand for lt in grid}
                    cells += [(f"randA@{s}", "shared", br), (f"randB@{s}", "residual", br)]
                frow = {"task": tname}
                for tag, cellname, br in cells:
                    cell_sd, book = SURG.build_decoder(cellname, base_lm_sd, ft_sd, br)
                    install(cell_sd, f"{slug}/{tag}")
                    del cell_sd
                    res = _eval_cell(model, proc, task, rows)
                    pg = CS.safe_ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
                    span = bres["score"] - floor_res[tname]["score"]
                    assert span > 1e-3, f"{tname}: no dynamic range vs the dead floor"
                    frow[tag] = {**_gain(res, bres), "score": res["score"],
                                 "ci95": res["ci95"],
                                 "saturated_dead":
                                     bool((res["score"] - floor_res[tname]["score"])
                                          / span < 0.02),
                                 "injected_fro2": book["total_injected_fro2"],
                                 "total_fro2": book["total_fro2"],
                                 "log1p_dppl": CS.safe_log1p_dppl(pg["mean_nll"],
                                                                  rep["base_ppl"]),
                                 "saturated_ppl": pg["saturated"]}
                    print(f"[carry-c] {slug}/{tag}: gain={frow[tag]['gain']:+.4f} "
                          f"dppl={frow[tag]['log1p_dppl']:+.3f}", flush=True)
                    rep["cells"][slug] = frow
                    _w()
                del ft_sd
                drop_snapshot(repo)
                # per-FT guard + gain fractions (rule mechanics live in carry_core;
                # adjudication happens downstream on the banked metrics, not here)
                full = frow["ft"]
                frow["range_ok"] = CC.range_guard_ok(full["gain"], full["se"])
                if frow["range_ok"]:
                    for tag in [t for t, _, _ in cells if t != "ft"]:
                        frow[tag]["gain_frac"] = CC.gain_frac(frow[tag]["gain"],
                                                              full["gain"])
                _w()
        restore = {k: v.to(dev) for k, v in base_lm_sd.items()}
        lm.load_state_dict(restore, strict=True)

        # R-C adjudication on the banked cells (house style: the verdict lives in the
        # artifact). band_stiff_max = max gain_frac over the 3 randA (k_stiff projection)
        # draws; band_flat_max = max over the 3 randB (complement) draws. Only range-passing
        # FTs enter `rows`; n_void = n_sealed_total − len(rows) (the discovery pool is the
        # frozen 6, the sealed pool the frozen 18 — n_sealed_total = the pool size).
        rows = []
        for slug, frow in rep["cells"].items():
            if not frow.get("range_ok"):
                continue
            rows.append({
                "ft": slug, "gf_stiff": frow["stiff"]["gain_frac"],
                "gf_flat": frow["flat"]["gain_frac"],
                "band_stiff_max": max(frow[f"randA@{s}"]["gain_frac"]
                                      for s in range(CC.N_RAND_SEEDS)),
                "band_flat_max": max(frow[f"randB@{s}"]["gain_frac"]
                                     for s in range(CC.N_RAND_SEEDS))})
        n_total = len(pool)
        rep["rc"] = CC.adjudicate_rc(rows, n_sealed_total=n_total)
        print(f"[carry-c] R-C ({a.half}): {rep['rc']['verdict']} "
              f"Δcarry={rep['rc'].get('dcarry', 'n/a')}", flush=True)

    CS.assert_gpu_seconds(rep)
    summary = ({"discovery_fts": rep.get("discovery_fts")} if a.phase == "assign"
               else {"verdict": rep.get("rc", {}).get("verdict"),
                     "range_ok": {s: rep["cells"][s].get("range_ok")
                                  for s in rep.get("cells", {})}})
    print("CARRY_C_OK", json.dumps(summary))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
