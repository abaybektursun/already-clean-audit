"""HANDLE Gate-0 (GPU) — spec §3.1 row 1. Five checks, any miss halts the arm (journal first):
  1. anchor      — PG1 sha + arch/MQA asserts + ppl gate vs the CARRY anchor (PPL_TOL).
  2. q_algebra   — §0.5-1/§0.5-2: load carry_scope's Q eigvectors (NEVER re-derive); assert the
                   top-k_stiff half-trace split reproduces (cum evals crosses 0.5·tr at k_stiff)
                   and the ladder projectors nest (P_k ⊂ P_{k'}) + P_{k_stiff} is the CARRY stiff
                   projector (handle_core.assert_nesting).
  3. items       — bank the handle| halves for all 4 wired tasks + the re-salt INDEPENDENCE audit
                   (the handle| sealed mask ~independent of carry|/comwrite| — the §4.1 firewall,
                   operationalized as split-independence: H-A2, dated 2026-07-13).
  4. preflight   — the swap-shell WIRING control: rebuild ONE banked CARRY sealed `stiff` cell
                   (= core_{k_stiff}) through HANDLE's exact pipeline and reproduce its banked
                   greedy-decode score to PREFLIGHT_TOL (+ identity-cell bitwise-base inertness).
  5. range_recal — recompute the full-write gain + construction-matched SE on handle| DISCOVERY
                   items for the preflight FT → the handle| range instrument (§3.1); the per-FT
                   range guards themselves run in Arm H.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_gate0 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/handle_gate0 --budget-min 150
"""
import argparse
import hashlib
import json
import os
import pathlib

PPL_TOL = 0.5              # inherit/carry gate convention
PREFLIGHT_TOL = 0.01      # |reproduced − banked| CARRY-sealed-cell score (cross-hardware bf16)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    ap.add_argument("--n-items", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _handle_shared as HS
    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import carry_core as CC
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import handle_core as HC
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import _content_hash, build_tasks, load_eval_items

    freeze = HS.assert_frozen("gate0")
    HS.install_pg()
    rep = {"object": "handle_gate0_v1", "freeze": freeze}
    tm = HS.StageTimer(rep)

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/handle_gate0")
    wd.mkdir(parents=True, exist_ok=True)
    fts = HS.load_carry_fts(str(wd))
    anat = fts["anat"]
    scope = json.loads(HS.fetch(f"{HS.CARRY_SCOPE}/metrics.json", wd / "scope.json").read_text())
    assert scope.get("green"), "carry_scope not green"
    carry_g0 = json.loads(HS.fetch(f"{HS.CARRY_GATE0}/gate0.json", wd / "carry_gate0.json").read_text())
    assert carry_g0.get("green"), "carry gate0 not green"
    carry_sealed = json.loads(HS.fetch(f"{HS.CARRY_C_SEALED}/metrics.json",
                                       wd / "carry_sealed.json").read_text())
    rep["roster"] = {"n_discovery": len(fts["discovery"]), "discovery": fts["discovery"],
                     "n_sealed_pool": len(fts["sealed_pool"]),
                     "n_sealed_scored": fts["n_sealed_scored"], "sealed_scored": fts["sealed_scored"]}
    _w()

    # ---- 1. anchor ----------------------------------------------------------------------
    with tm.stage("anchor"):
        sha, ppl_ref = HS.pinned_pg1_sha(str(wd))
        assert sha == fts["base_sha"], \
            f"provenance fork: invcore sha {sha[:8]} != anatomy {fts['base_sha'][:8]}"
        model, proc = IP.load_pg1(sha=sha)
        cfg = model.config.text_config
        assert cfg.num_hidden_layers == 18 and cfg.hidden_size == 2048, \
            f"architecture drift: {cfg.num_hidden_layers}L d={cfg.hidden_size}"
        assert cfg.num_key_value_heads == 1, f"MQA assert failed: {cfg.num_key_value_heads} KV heads"
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        rep["ppl_anchor"] = {"ppl": pp["ppl"], "ref": ppl_ref, "base_sha": sha}
        assert abs(pp["ppl"] - float(ppl_ref)) <= PPL_TOL, \
            f"base ppl gate RED: {pp['ppl']:.4f} vs {ppl_ref:.4f}"
    _w()

    # ---- 2. Q algebra (§0.5-1 half-trace reproduction, §0.5-2 nesting) ------------------
    with tm.stage("q_algebra"):
        qb = HS.load_q_bank(str(wd))
        k_stiff = int(qb["k_stiff"])
        evals = torch.as_tensor(qb["evals"], dtype=torch.float64)
        evecs = torch.as_tensor(qb["evecs"], dtype=torch.float64)
        tr = float(evals.sum())
        cum = torch.cumsum(evals, 0)
        # half-trace split: k_stiff is the SMALLEST k with cum ≥ 0.5·tr (CARRY stiff_split)
        assert float(cum[k_stiff - 1]) >= 0.5 * tr > float(cum[k_stiff - 2]), \
            "banked k_stiff is not the half-trace split of the banked evals (§0.5-1 FAIL)"
        ks = HC.ladder_ks(k_stiff)
        stiff_proj = evecs[:, :k_stiff] @ evecs[:, :k_stiff].transpose(0, 1)
        nest = HC.assert_nesting(evecs, ks, k_stiff, stiff_projector=stiff_proj)
        rep["q_algebra"] = {"k_stiff": k_stiff, "ladder": list(ks), "tr_Q": tr,
                            "stiff_trace_frac": float(cum[k_stiff - 1] / tr), **nest}
    _w()

    # ---- 3. items + handle| re-salt independence audit ---------------------------------
    with tm.stage("items"):
        tasks = {t.name: t for t in build_tasks()}
        for tname in CS.WIRED_TASKS:
            rows = load_eval_items(tasks[tname], CS.N_STREAM)
            hashes = [_content_hash(tasks[tname], r) for r in rows]
            split = HS.handle_split(hashes, name=f"handle_{tname}")
            sealed_idx = split["conf_idx"].tolist()
            (out / f"items_handle_{tname}.json").write_text(json.dumps(
                {"hashes": hashes, "sealed_idx": sealed_idx}))
            held = np.zeros(len(hashes), dtype=bool)
            held[sealed_idx] = True
            audit = {"n": len(hashes), "n_sealed": len(sealed_idx)}
            for prior in ("carry", "comwrite", "grain"):
                pm = np.array([int(hashlib.sha1(f"{prior}|{h}".encode()).hexdigest(), 16) % 2 == 1
                               for h in hashes])
                agree = float((pm == held).mean())
                audit[f"agreement_vs_{prior}"] = agree
                assert 0.4 < agree < 0.6, \
                    f"{tname}: handle| split correlates with {prior}| ({agree:.3f}) — salt bug"
            rep.setdefault("resalt_audit", {})[tname] = audit
    _w()

    # ---- 4. swap-shell wiring preflight: reproduce one CARRY sealed `stiff` cell --------
    with tm.stage("preflight"):
        paths = resolve_submodule_paths(model)
        lm = model.get_submodule(paths["language_model"])
        dev = next(model.parameters()).device
        base_lm_sd = {k: v.detach().clone().cpu() for k, v in lm.state_dict().items()}
        vt_orig = {k: v.detach().clone().cpu()
                   for k, v in model.get_submodule(paths["vision_tower"]).state_dict().items()}
        grid = SURG.grid_of(base_lm_sd)
        SURG.assert_full_grid(grid)
        assert anat["excluded_lt"] == [], "CARRY splits every (L,t); a non-empty exclusion needs a note"
        U_stiff = torch.as_tensor(qb["evecs"])[:, :k_stiff].contiguous()
        bases_stiff = HS.rung_bases(U_stiff, grid)

        repro_slug = sorted(fts["sealed_scored"])[0]
        repro_task = fts["assigned_task"][repro_slug]
        banked = carry_sealed["cells"][repro_slug]["stiff"]
        rep["preflight_target"] = {"ft": repro_slug, "task": repro_task,
                                   "banked_stiff_score": banked["score"]}

        # identity inertness FIRST (delta 0.0 → bitwise base), then the stiff cell.
        ft_sd = HS.ft_decoder_sd(fts["repo_of"][repro_slug], anat["shas"][fts["repo_of"][repro_slug]])
        ident_sd, _ = SURG.build_decoder("identity", base_lm_sd, ft_sd, bases_stiff)
        assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd), \
            "identity cell not bitwise base — surgery pipeline NOT inert"
        del ident_sd

        # CARRY's own sealed items for this FT's task (reproduce its banked number, not ours).
        c_hashes, c_nstream = CS.load_gate0_items(str(wd), repro_task, "sealed", n=a.n_items)
        c_rows = CS.rows_for_banked(tasks[repro_task], c_hashes, c_nstream)
        cell_sd, book = SURG.build_decoder("shared", base_lm_sd, ft_sd, bases_stiff)
        missing, unexpected = lm.load_state_dict({k: v.to(dev) for k, v in cell_sd.items()}, strict=True)
        assert not missing and not unexpected, "preflight stiff cell LM load not clean"
        vt_now = model.get_submodule(paths["vision_tower"]).state_dict()
        assert all(torch.equal(vt_now[k].detach().cpu(), vt_orig[k]) for k in vt_orig), \
            "preflight LEAK — vision tower changed"
        del cell_sd
        res = IP.eval_pg1(model, proc, tasks[repro_task], c_rows, "cuda")
        rep["preflight"] = {"reproduced_stiff_score": res["score"],
                            "banked_stiff_score": banked["score"],
                            "delta": res["score"] - banked["score"], "n": res["n"],
                            "stiff_energy_fraction": book["total_captured_fro2"] / book["total_fro2"]}
        assert abs(res["score"] - banked["score"]) <= PREFLIGHT_TOL, \
            (f"preflight RED: reproduced {res['score']:.4f} vs banked CARRY-sealed stiff "
             f"{banked['score']:.4f} — the shell does not reproduce its own parent artifact")
        lm.load_state_dict({k: v.to(dev) for k, v in base_lm_sd.items()}, strict=True)
    _w()

    # ---- 5. range recalibration under handle| (the reproduction FT, disc items) ---------
    with tm.stage("range_recal"):
        h_hashes, h_nstream = HS.load_handle_items(str(wd), repro_task, "disc", n=a.n_items)
        h_rows = HS.rows_for_banked(tasks[repro_task], h_hashes, h_nstream)
        base_res = IP.eval_pg1(model, proc, tasks[repro_task], h_rows, "cuda")
        ft_full_sd, _ = SURG.build_decoder("ft", base_lm_sd, ft_sd, bases_stiff)
        lm.load_state_dict({k: v.to(dev) for k, v in ft_full_sd.items()}, strict=True)
        del ft_full_sd
        ft_res = IP.eval_pg1(model, proc, tasks[repro_task], h_rows, "cuda")
        if base_res["per_item"] is not None:
            g = CC.paired_gain(ft_res["per_item"], base_res["per_item"])
        else:
            g = {"gain": ft_res["score"] - base_res["score"],
                 "se": ((ft_res["ci95"][1] - ft_res["ci95"][0]) ** 2
                        + (base_res["ci95"][1] - base_res["ci95"][0]) ** 2) ** 0.5 / (2 * 1.96)}
        rep["range_recal"] = {"ft": repro_slug, "task": repro_task, "handle_full_gain": g["gain"],
                              "handle_se": g["se"], "range_ok": CC.range_guard_ok(g["gain"], g["se"])}
        lm.load_state_dict({k: v.to(dev) for k, v in base_lm_sd.items()}, strict=True)
        HS.drop_snapshot(fts["repo_of"][repro_slug])
        del ft_sd
    _w()

    HS.assert_gpu_seconds(rep)
    print("HANDLE_GATE0_OK", json.dumps({"ppl": rep["ppl_anchor"]["ppl"],
                                         "preflight_delta": rep["preflight"]["delta"],
                                         "k_stiff": k_stiff}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
