"""CARRY Gate-0 (GPU) — spec §3.1 row 1: ppl anchor + model sha; the swap-shell wiring
preflight (reproduce one banked COMWRITE surgery cell's gain from its artifact BEFORE any
new cell); the RMSNorm-γ readout at the sink coordinates (the S11 check); MQA asserts; and
the `carry|` re-salt audit + item banking for all 4 wired tasks.

Any miss halts the arm (nonzero exit) — journal first, per §9.2.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.carry_gate0 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/carry_gate0 --budget-min 180
"""
import argparse
import json
import os
import pathlib

PPL_TOL = 0.5                 # comwrite gate convention
PREFLIGHT_TOL = 0.01          # |reproduced − banked| on the donor score
N_SINK_COORDS = 8             # top massive-activation coordinates read (S17/S11 check)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ppl-seq", type=int, default=120)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from transformers import PaliGemmaForConditionalGeneration

    from infra.aws_pg.jobs import _carry_shared as CS
    from infra.pg import comwrite_surgery as SURG
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg.projector_ablation import _content_hash, build_tasks, load_eval_items

    freeze = CS.assert_frozen("gate0")
    CS.install_pg()
    rep = {"object": "carry_gate0_v1", "freeze": freeze}
    tm = CS.StageTimer(rep)

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    wd = pathlib.Path("/tmp/carry_gate0")
    wd.mkdir(parents=True, exist_ok=True)
    anat = CS.load_roster(wd)
    scope = json.loads(CS.fetch(f"{CS.CARRY_SCOPE}/metrics.json",
                                wd / "scope.json").read_text())
    assert scope.get("green"), "carry_scope not green"

    # ---- ppl anchor + sha + MQA asserts -------------------------------------------------
    with tm.stage("anchor"):
        sha, ppl_ref = CS.pinned_pg1_sha(str(wd))
        assert sha == anat["base_sha"], \
            f"provenance fork: invcore sha {sha[:8]} != anatomy {anat['base_sha'][:8]}"
        model, proc = IP.load_pg1(sha=sha)
        cfg = model.config.text_config
        assert cfg.num_hidden_layers == 18 and cfg.hidden_size == 2048, \
            f"architecture drift: {cfg.num_hidden_layers}L d={cfg.hidden_size}"
        assert cfg.num_key_value_heads == 1, \
            f"MQA assert failed: {cfg.num_key_value_heads} KV heads (want 1)"
        pp = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)
        rep["ppl_anchor"] = {"ppl": pp["ppl"], "ref": ppl_ref, "base_sha": sha}
        assert abs(pp["ppl"] - float(ppl_ref)) <= PPL_TOL, \
            f"base ppl gate RED: {pp['ppl']:.4f} vs {ppl_ref:.4f}"
    _w()

    # ---- RMSNorm-γ readout at the sink coordinates (S11/S17, minutes) -------------------
    with tm.stage("gamma_sink"):
        layers = IP.decoder_layers(model)
        wins = CS.wikitext_windows(proc, 8)
        cap = IP.harvest(layers, lambda b: model(**b).logits,
                         [_ppl_inputs(proc, w, model) for w in wins],
                         [0, 9, 17], x_cap=2048)
        sink_rows = {}
        for L, (hbar, _trc, X, _n) in cap.items():
            mag = X.abs().amax(0) if X is not None else hbar.abs()
            top = torch.topk(mag, N_SINK_COORDS)
            gamma = layers[L].input_layernorm.weight.detach().float().cpu()
            sink_rows[f"L{L}"] = {
                "coords": [int(i) for i in top.indices],
                "max_abs_activation": [float(v) for v in top.values],
                "median_abs_activation": float(mag.median()),
                "gamma_at_coords": [float(gamma[i]) for i in top.indices],
                "gamma_median": float(gamma.median())}
        rep["gamma_sink_readout"] = sink_rows
    _w()

    # ---- item banking + the carry| re-salt audit ----------------------------------------
    with tm.stage("items"):
        tasks = {t.name: t for t in build_tasks()}
        import hashlib as _hl
        for tname in CS.WIRED_TASKS:
            rows = load_eval_items(tasks[tname], CS.N_STREAM)
            hashes = [_content_hash(tasks[tname], r) for r in rows]
            split = CS.carry_split(hashes, name=f"carry_{tname}")   # enforce_split → dict
            sealed_idx = split["conf_idx"].tolist()
            (out / f"items_carry_{tname}.json").write_text(json.dumps(
                {"hashes": hashes, "sealed_idx": sealed_idx}))
            # re-salt audit: the carry| SEALED mask must be ~independent of the comwrite| one
            held_mask = np.zeros(len(hashes), dtype=bool)
            held_mask[sealed_idx] = True
            cw = np.array([int(_hl.sha1(f"comwrite|{h}".encode()).hexdigest(), 16) % 2 == 1
                           for h in hashes])
            agree = float((cw == held_mask).mean())
            rep.setdefault("resalt_audit", {})[tname] = {
                "n": len(hashes), "n_sealed": len(sealed_idx),
                "agreement_vs_comwrite_salt": agree}
            assert 0.4 < agree < 0.6, \
                f"{tname}: carry| split correlates with comwrite| ({agree:.3f}) — salt bug"
    _w()

    # ---- swap-shell wiring preflight: reproduce the banked COMWRITE cell ----------------
    with tm.stage("preflight"):
        tgt = scope["gate0_reproduction_target"]
        donor_repo = "google/paligemma-3b-ft-vqav2-224"
        donor_idx = anat["roster"].index(donor_repo)
        excluded = {(int(L), t) for L, t in anat["excluded_lt"]}
        k_star = int(anat["k_star"])
        paths = resolve_submodule_paths(model)
        lm = model.get_submodule(paths["language_model"])
        base_lm_sd = {k: v.detach().clone().cpu() for k, v in lm.state_dict().items()}
        grid = SURG.grid_of(base_lm_sd)
        SURG.assert_full_grid(grid)
        bases = {}
        for (L, t) in grid:
            if (L, t) in excluded:
                continue
            f = CS.fetch(f"{CS.COMWRITE_ANATOMY}/scratch/loo__{t}__L{L}__{donor_idx}.npy",
                         wd / f"bank/loo__{t}__L{L}.npy")
            U = np.load(f).astype(np.float32)
            bases[(L, t)] = torch.from_numpy(U[:, :k_star].copy())
        ft = PaliGemmaForConditionalGeneration.from_pretrained(
            donor_repo, revision=anat["shas"][donor_repo], torch_dtype=torch.bfloat16,
            device_map="cpu", attn_implementation="eager").eval()
        ft_lm_sd = {k: v.detach().clone()
                    for k, v in ft.get_submodule(
                        resolve_submodule_paths(ft)["language_model"]).state_dict().items()}
        del ft
        # identity inertness FIRST, then the banked ft@0 cell on the banked comwrite items
        ident_sd, _ = SURG.build_decoder("identity", base_lm_sd, ft_lm_sd, bases,
                                         excluded=excluded)
        assert all(torch.equal(ident_sd[k], base_lm_sd[k]) for k in base_lm_sd), \
            "identity cell not bitwise base — surgery pipeline NOT inert"
        del ident_sd
        it = json.loads(CS.fetch(f"{CS.COMWRITE_GATE0}/items_vqav2.json",
                                 wd / "items_vqav2_comwrite.json").read_text())
        sealed = set(it["sealed_idx"])
        disc_hashes = [h for i, h in enumerate(it["hashes"]) if i not in sealed][:800]
        rows = CS.rows_for_banked(tasks["vqav2"], disc_hashes, len(it["hashes"]))
        cell_sd, _book = SURG.build_decoder("ft", base_lm_sd, ft_lm_sd, bases,
                                            excluded=excluded)
        missing, unexpected = lm.load_state_dict(
            {k: v.to(next(model.parameters()).device) for k, v in cell_sd.items()},
            strict=True)
        assert not missing and not unexpected
        del cell_sd
        res = IP.eval_pg1(model, proc, tasks["vqav2"], rows, "cuda")
        rep["preflight"] = {"banked_ft_score": tgt["ft_score"],
                            "reproduced_ft_score": res["score"],
                            "delta": res["score"] - tgt["ft_score"], "n": res["n"]}
        assert abs(res["score"] - tgt["ft_score"]) <= PREFLIGHT_TOL, \
            (f"preflight RED: reproduced {res['score']:.4f} vs banked "
             f"{tgt['ft_score']:.4f} — the shell does not reproduce its own artifact")
        lm.load_state_dict({k: v.to(next(model.parameters()).device)
                            for k, v in base_lm_sd.items()}, strict=True)
        from infra.aws_pg.jobs.invariant_core import _delete_cache
        _delete_cache(donor_repo)
    _w()

    CS.assert_gpu_seconds(rep)
    print("CARRY_GATE0_OK", json.dumps({"ppl": rep["ppl_anchor"]["ppl"],
                                        "preflight_delta": rep["preflight"]["delta"]}))
    rep["green"] = True
    _w()


def _ppl_inputs(proc, wdict, model):
    """One blank-image forward batch dict for the harvest pass."""
    import torch
    inp = proc(text=[wdict["text"]], images=[wdict["blank"]], return_tensors="pt",
               padding=False).to(next(model.parameters()).device)
    inp["pixel_values"] = inp["pixel_values"].to(next(model.parameters()).dtype)
    return dict(inp)


if __name__ == "__main__":
    main()
