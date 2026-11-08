"""HANDLE Arm H (GPU) — the MDE: the compact-core ladder. Does a small core inside the 679-dim
stiff space carry most of the direction-specific gain, and is the Q-ordering load-bearing?

Per FT, per rung k ∈ ladder = {5,10,20,50,150,350,k_stiff}:
  * core_k        — build_decoder('shared', bases={(L,t): Q_evecs[:,:k]})   (the nested top-k slice)
  * band_k (×3)   — 'shared' with random_eigsplit(Q_evecs, k, seed)         (matched-rank, ALL d)
  * band_k^stiff  — 'shared' with random_eigsplit(Q_evecs[:,:k_stiff], k, seed)  (within-stiff)
    (×3)
plus ONE ft (full-write) cell per FT — the gain_frac denominator + the range guard. band_k /
band_k^stiff at a rung = the MAX gain_frac over the 3 seeds (the CARRY null convention). Seeds are
frozen sha1(handle_band|slug|k|btype|s). One FT snapshot on disk at a time (dropped after its
cells). Surgery = comwrite_surgery.build_decoder on the PG1 language_model inside the full
multimodal model (identity-inertness + vision-tower-leak asserted). Q eigvectors LOADED from
carry_scope (never re-derived).

Phases:
  --phase disc     rule rehearsal on CARRY's 6 discovery FTs.
  --phase sealed   the 12 CARRY-scored sealed FTs (--fts subset ⇒ run as a map; each banks cells).
  --phase adjudicate --sealed-prefixes p1,p2,...  CPU: merge sealed cells + R-H verdict.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_h --type g6e.8xlarge,p5.4xlarge --no-weights \
      --s3-prefix results/handle_h_disc --budget-min 480 --args "--phase disc"
"""
import argparse
import hashlib
import json
import os
import pathlib


def _band_seed(slug, k, btype, s):
    return int(hashlib.sha1(f"handle_band|{slug}|{k}|{btype}|{s}".encode()).hexdigest()[:8], 16)


def _gain(res_cell, res_base):
    """{gain, se}: paired-item bootstrap SE for per-item tasks; corpus tasks combine the two
    cells' resampled CIs conservatively (the carry_c convention, disclosed)."""
    from infra.pg import carry_core as CC
    if res_cell["per_item"] is not None:
        return CC.paired_gain(res_cell["per_item"], res_base["per_item"])
    se = ((res_cell["ci95"][1] - res_cell["ci95"][0]) ** 2
          + (res_base["ci95"][1] - res_base["ci95"][0]) ** 2) ** 0.5 / (2 * 1.96)
    return {"gain": res_cell["score"] - res_base["score"], "se": se}


def _adjudicate(rep, fts, scored, ks, k_stiff):
    """R-H on the merged sealed cells (rep['cells'])."""
    from infra.pg import handle_core as HC
    rows = []
    for slug in scored:
        frow = rep["cells"].get(slug)
        if not frow or not frow.get("range_ok"):
            continue
        rows.append({"ft": slug,
                     "core": {k: frow["rungs"][str(k)]["core"]["gain_frac"] for k in ks},
                     "band": {k: frow["rungs"][str(k)]["band_max"] for k in ks},
                     "band_stiff": {k: frow["rungs"][str(k)]["band_stiff_max"] for k in ks}})
    rep["rh"] = HC.adjudicate_rh(rows, ks=ks, k_stiff=k_stiff, n_sealed_total=len(scored))
    return rep["rh"]


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", required=True, choices=("disc", "sealed", "adjudicate"))
    ap.add_argument("--fts", default="", help="comma slugs (disc/sealed); empty = full pool")
    ap.add_argument("--sealed-prefixes", default="", help="adjudicate: comma S3 result prefixes")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import handle_core as HC

    arm = f"h_{a.phase}"
    freeze = HS.assert_frozen(arm)
    HS.install_pg()
    rep = {"object": f"handle_h_{a.phase}_v1", "freeze": freeze, "phase": a.phase}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = "/tmp/handle_h"
    pathlib.Path(wd).mkdir(parents=True, exist_ok=True)
    fts = HS.load_carry_fts(wd)
    qb = HS.load_q_bank(wd)
    k_stiff = int(qb["k_stiff"])
    ks = HC.ladder_ks(k_stiff)
    rep["k_stiff"], rep["ladder"] = k_stiff, list(ks)
    scored = fts["sealed_scored"]

    # ---- adjudicate (CPU): merge the sealed subset outputs + R-H ------------------------
    if a.phase == "adjudicate":
        prefixes = [p.strip() for p in a.sealed_prefixes.split(",") if p.strip()]
        assert prefixes, "adjudicate needs --sealed-prefixes"
        rep["cells"] = {}
        for pref in prefixes:
            m = json.loads(HS.fetch(f"{HS.S3_BUCKET}/{pref}/out/metrics.json",
                                    pathlib.Path(wd) / f"{pref.replace('/', '_')}.json").read_text())
            assert m.get("green"), f"sealed subset {pref} not green"
            rep["cells"].update(m.get("cells", {}))
        rep["gpu_seconds"] = {"adjudicate": 0.0}
        v = _adjudicate(rep, fts, scored, ks, k_stiff)
        print("HANDLE_H_ADJ_OK", json.dumps({"verdict": v["verdict"]}))
        rep["green"] = True
        _w()
        return

    # ---- GPU phases: the ladder ---------------------------------------------------------
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.pg import comwrite_surgery as SURG
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import build_tasks

    pool = fts["discovery"] if a.phase == "disc" else scored
    want = [s.strip() for s in a.fts.split(",") if s.strip()] or list(pool)
    assert set(want) <= set(pool), f"--fts not all in the frozen {a.phase} pool"
    rep["fts_requested"] = want

    tasks = {t.name: t for t in build_tasks()}
    evecs = torch.as_tensor(qb["evecs"])
    U_stiff = evecs[:, :k_stiff].contiguous()

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

    def install(cell_sd, tag):
        missing, unexpected = lm.load_state_dict({k: v.to(dev) for k, v in cell_sd.items()}, strict=True)
        assert not missing and not unexpected, f"{tag}: LM load not clean"
        vt = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt[k].detach().cpu(), vt_orig[k]) for k in vt_orig), \
            f"{tag}: LEAK — vision tower changed"

    # identity inertness (once)
    with tm.stage("identity"):
        ident_sd, _ = SURG.build_decoder("identity", base_lm_sd,
                                         {k: v.clone() for k, v in base_lm_sd.items()},
                                         HS.rung_bases(U_stiff, grid))
        assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd), \
            "identity cell not bitwise base"
        del ident_sd
    _w()

    # base/floor per needed task
    need_tasks = sorted({fts["assigned_task"][s] for s in want})
    half = "disc" if a.phase == "disc" else "sealed"
    rows_cache, base_res, floor_res = {}, {}, {}
    with tm.stage("base_floor"):
        for tname in need_tasks:
            hashes, nstream = HS.load_handle_items(wd, tname, half, n=a.n_items)
            rows_cache[tname] = HS.rows_for_banked(tasks[tname], hashes, nstream)
            base_res[tname] = IP.eval_pg1(model, proc, tasks[tname], rows_cache[tname], "cuda")
            floor_res[tname] = IP.eval_pg1(model, proc, tasks[tname], rows_cache[tname], "cuda",
                                           scramble_seed=7)
    _w()

    def eval_cell(cellname, bases, task, rows):
        cell_sd, book = SURG.build_decoder(cellname, base_lm_sd, ft_sd, bases)
        install(cell_sd, cellname)
        del cell_sd
        res = IP.eval_pg1(model, proc, task, rows, "cuda")
        return res, book

    rep["cells"] = {}
    for slug in want:
        tname = fts["assigned_task"][slug]
        task, rows, bres, fres = tasks[tname], rows_cache[tname], base_res[tname], floor_res[tname]
        span = bres["score"] - fres["score"]
        assert span > 1e-3, f"{tname}: no dynamic range vs the dead floor"
        with tm.stage(f"ft_{slug}"):
            ft_sd = HS.ft_decoder_sd(fts["repo_of"][slug], fts["shas"][fts["repo_of"][slug]])
            # the ft (full-write) cell: gain_frac denominator + range guard
            res_ft, book_ft = eval_cell("ft", HS.rung_bases(U_stiff, grid), task, rows)
            g_full = _gain(res_ft, bres)
            frow = {"task": tname, "ft": {**g_full, "score": res_ft["score"]},
                    "range_ok": CC.range_guard_ok(g_full["gain"], g_full["se"]), "rungs": {}}
            rep["cells"][slug] = frow
            print(f"[handle-h] {slug}: full gain={g_full['gain']:+.4f} se={g_full['se']:.4f} "
                  f"range_ok={frow['range_ok']}", flush=True)
            _w()
            if not frow["range_ok"]:
                HS.drop_snapshot(fts["repo_of"][slug]); del ft_sd
                continue
            for k in ks:
                # core_k
                core_res, core_book = eval_cell("shared", HS.rung_bases(evecs[:, :k].contiguous(), grid),
                                                task, rows)
                pg = HS.safe_ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
                gf_core = CC.gain_frac(_gain(core_res, bres)["gain"], g_full["gain"])
                rung = {"core": {"gain_frac": gf_core, "score": core_res["score"],
                                 "log1p_dppl": HS.safe_log1p_dppl(pg["mean_nll"], rep["base_ppl"]),
                                 "saturated_ppl": pg["saturated"],
                                 "injected_fro2": core_book["total_injected_fro2"]},
                        "band": [], "band_stiff": []}
                for btype, pool_ev in (("band", None), ("band_stiff", k_stiff)):
                    # band^stiff at k==k_stiff is "random 679-of-679" = the full stiff space =
                    # core exactly (tautological); skip those evals and set it to gf_core.
                    if btype == "band_stiff" and k == k_stiff:
                        rung["band_stiff"] = [gf_core] * HC.N_RAND_SEEDS
                        continue
                    for s in range(HC.N_RAND_SEEDS):
                        U, _idx = HC.band_basis(evecs, k, _band_seed(slug, k, btype, s),
                                                within_stiff=pool_ev)
                        r, _b = eval_cell("shared", HS.rung_bases(U, grid), task, rows)
                        rung[btype].append(CC.gain_frac(_gain(r, bres)["gain"], g_full["gain"]))
                rung["band_max"] = max(rung["band"])
                rung["band_stiff_max"] = max(rung["band_stiff"])
                frow["rungs"][str(k)] = rung
                rep["cells"][slug] = frow
                print(f"[handle-h] {slug} k={k}: core={gf_core:+.3f} "
                      f"band={rung['band_max']:+.3f} band^stiff={rung['band_stiff_max']:+.3f}",
                      flush=True)
                _w()
            HS.drop_snapshot(fts["repo_of"][slug]); del ft_sd
    lm.load_state_dict({k: v.to(dev) for k, v in base_lm_sd.items()}, strict=True)

    # inline adjudication only when this job covered the FULL scored pool (else a CPU adjudicate step)
    if a.phase == "sealed" and set(want) >= set(scored):
        v = _adjudicate(rep, fts, scored, ks, k_stiff)
        print("HANDLE_H_SEALED_VERDICT", json.dumps({"verdict": v["verdict"]}))
    elif a.phase == "disc":
        v = _adjudicate(rep, fts, fts["discovery"], ks, k_stiff)  # rehearsal only (not sealed)
        rep["rh_rehearsal"] = rep.pop("rh")
        print("HANDLE_H_DISC_REHEARSAL", json.dumps({"verdict": v["verdict"]}))

    HS.assert_gpu_seconds(rep)
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
