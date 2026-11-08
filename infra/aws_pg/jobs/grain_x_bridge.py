"""GRAIN X (GPU; exploratory, labels only, no confirmatory weight) — the COMWRITE bridges.

X1 damage blocks: rebuild the vqav2 `shared` and `residual` surgical models (the COMWRITE
deterministic builder, REUSED verbatim at the anatomy artifact's own k*/sha), harvest the
entering-residual at both GRAIN sites under {base, shared, residual, ft} over the SAME
discovery-slice batches, project each per-row activation shift (state − base) into block
coordinates of the site's m* dictionary. Statistic: top-8 blocks' share of the shift energy —
damage footprint (shared/residual) vs task footprint (ft).

X2 U_wr footprint: block alignment of the donor-LOO U_wr bases (o_proj/down_proj at the layer
writing into the secondary site) against the secondary dictionary. Descriptive.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_x_bridge --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_x_bridge --budget-min 180 --args "\
      --dicts-primary-s3 .../grain_train_primary/out --dicts-secondary-s3 .../grain_train_secondary/out \
      --gdisc-s3 .../grain_g_disc_saction/out --gate0-s3 .../grain_gate0/out"
"""
import argparse
import json
import pathlib

TOP8 = 8
DONOR = "vqav2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dicts-primary-s3", required=True)
    ap.add_argument("--gates-primary-s3", default=None)
    ap.add_argument("--dicts-secondary-s3", required=True)
    ap.add_argument("--gdisc-s3", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--n-items", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    from infra.pg import comwrite_surgery as CS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg.entities import resolve_submodule_paths

    GS.install_pg()
    rep = {"object": "grain_x_bridge_v1", "register": "EXPLORATORY", "donor": DONOR}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # --- dictionaries (median-seed m* per site) ---
    disc = json.loads(GS.fetch(f"{a.gdisc_s3}/g_disc.json", "/tmp/grain/g_disc.json").read_text())
    med_s = int(disc.get("median_seed", 0))
    dicts, m_star_by, L_by = {}, {}, {}
    rep["site_unrun"] = {}
    for site_name, s3 in (("primary", a.dicts_primary_s3), ("secondary", a.dicts_secondary_s3)):
        gsrc = a.gates_primary_s3 if (site_name == "primary" and a.gates_primary_s3) else s3
        gates = json.loads(GS.fetch(f"{gsrc}/gates.json", f"/tmp/grain/gx_{site_name}.json").read_text())
        m_star = gates.get("m_star")
        if not m_star:                 # §11.3 per-site pivot: no faithful block dict HERE
            rep["site_unrun"][site_name] = gates.get("pivot")
            continue
        seed = med_s if gates["configs"].get(f"m{m_star}_s{med_s}", {}).get("gate_pass") else 0
        ae, _c, _q, _m = GB.load_dict_bank(GS.fetch(
            f"{s3}/{GS.dict_bank_name(gates['site'], m_star, seed)}", f"/tmp/grain/xd_{site_name}.pt"))
        dicts[site_name] = ae
        m_star_by[site_name] = m_star
        L_by[site_name] = gates["L"]
    assert dicts, "no site has a faithful block dictionary — X UNRUN"
    sites = sorted(set(L_by.values()))
    rep["sites"] = L_by
    _w()

    # --- COMWRITE anatomy + surgical builds (deterministic, REUSED) ---
    anat = json.loads(GS.fetch(f"{GS.COMWRITE_ANATOMY}/metrics.json",
                               "/tmp/grain/anatomy.json").read_text())
    k_star = int(anat["k_star"])
    donor_repo = f"google/paligemma-3b-ft-{DONOR}-224"
    donor_idx = anat["roster"].index(donor_repo)
    excluded = {(int(L), t) for L, t in anat["excluded_lt"]}
    grid = [(m["layer"], m["type"]) for m in anat["per_matrix"]]
    bases = {}
    for L, t in grid:
        if (L, t) in excluded:
            continue
        f = GS.fetch(f"{GS.COMWRITE_ANATOMY}/scratch/loo__{t}__L{L}__{donor_idx}.npy",
                     f"/tmp/grain/bank/loo__{t}__L{L}.npy")
        U = np.load(f).astype(np.float32)
        bases[(L, t)] = torch.from_numpy(U[:, :k_star].copy())
    rep["anatomy"] = {"k_star": k_star, "excluded": sorted(excluded), "base_sha": anat["base_sha"]}
    _w()

    sha, _ = GS.pinned_pg1_sha()
    assert sha == anat["base_sha"], "provenance fork: grain pin != anatomy base sha"
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    paths = resolve_submodule_paths(model)
    lm = model.get_submodule(paths["language_model"])
    base_sd = {k: v.detach().clone().cpu() for k, v in lm.state_dict().items()}
    ft = PaliGemmaForConditionalGeneration.from_pretrained(
        donor_repo, revision=anat["shas"][donor_repo], torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager").eval()
    ft_sd = {k: v.detach().clone()
             for k, v in ft.get_submodule(resolve_submodule_paths(ft)["language_model"])
             .state_dict().items()}
    del ft
    ident_sd, _b = CS.build_decoder("identity", base_sd, ft_sd, bases, excluded=excluded)
    assert all(torch.equal(ident_sd[k], base_sd[k]) for k in base_sd), \
        "identity surgery not bitwise base — builder wiring broken"
    state_sds = {"shared": CS.build_decoder("shared", base_sd, ft_sd, bases,
                                            excluded=excluded)[0],
                 "residual": CS.build_decoder("residual", base_sd, ft_sd, bases,
                                              excluded=excluded)[0],
                 "ft": ft_sd}
    rep["identity_gate"] = "bitwise_base_ok"
    _w()

    # --- paired harvest at both sites under each state ---
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(GS.fetch(f"{a.gate0_s3}/items_vqa.json", "/tmp/grain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift vs the gate0 bank — halt"
    split = GS.grain_split(hashes, name="grain_vqa")
    disc_rows = [rows[i] for i in split["disc_idx"]][: a.n_items]
    batches = vqa_forward_batches(proc, vqa, disc_rows, "cuda")
    dev = next(model.parameters()).device

    def harvest_state():
        stores = {L: [] for L in sites}
        handles = [layers[L].register_forward_pre_hook(GS.make_row_capture(stores[L], IP.D),
                                                       with_kwargs=True) for L in sites]
        with torch.no_grad():
            for inp in batches:
                model(**inp)
                mask = inp["attention_mask"].reshape(-1).bool().cpu()
                for L in sites:
                    stores[L][-1] = stores[L][-1][mask]
        for h in handles:
            h.remove()
        return {L: torch.cat(stores[L], 0) for L in sites}

    acts = {"base": harvest_state()}
    for state, sd in state_sds.items():
        lm.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
        acts[state] = harvest_state()
    lm.load_state_dict({k: v.to(dev) for k, v in base_sd.items()}, strict=True)
    rep["n_rows"] = {str(L): int(acts["base"][L].shape[0]) for L in sites}
    _w()

    # --- X1: block-concentration of the shift energy ---
    def block_energies(site_name, D):
        ae = dicts[site_name]
        m = m_star_by[site_name]
        Wd = ae.decoder.weight.detach()
        Dg = D.to("cuda").float()
        out_e = torch.empty(ae.n_blocks, dtype=torch.float64)
        for b in range(ae.n_blocks):
            Qb = GB.block_subspace(Wd, m, b).float().to("cuda")
            out_e[b] = float(((Dg @ Qb) ** 2).sum().item())
        return out_e

    rep["x1"] = {}
    for site_name in dicts:
        L = L_by[site_name]
        entry = {}
        for state in ("shared", "residual", "ft"):
            D = (acts[state][L].float() - acts["base"][L].float())
            e = block_energies(site_name, D)
            tot = float(e.sum())
            top = torch.topk(e, TOP8)
            entry[state] = {"total_energy": tot,
                            "top8_share": float(top.values.sum()) / max(tot, 1e-12),
                            "top8_blocks": top.indices.tolist(),
                            "shift_fro2": float((D ** 2).sum())}
        rep["x1"][site_name] = entry
        _w()

    # --- X2: U_wr write-basis footprint at the secondary site ---
    assert "secondary" in dicts, "X2 needs the secondary block dictionary"
    Ls = L_by["secondary"]
    m2 = m_star_by["secondary"]
    W2 = dicts["secondary"].decoder.weight.detach()
    rep["x2"] = {}
    for t in ("o_proj", "down_proj"):
        lt = (Ls - 1, t)
        if lt in bases:
            align = GB.block_alignment(W2, m2, GB.orthonormalize(bases[lt].double()))
            tot = float(align.sum())
            top = torch.topk(align, TOP8)
            rep["x2"][f"L{Ls-1}_{t}"] = {"top8_share": float(top.values.sum()) / max(tot, 1e-12),
                                         "top8_blocks": top.indices.tolist()}
        else:
            rep["x2"][f"L{Ls-1}_{t}"] = {"state": "excluded_pair_or_missing"}

    rep["green"] = True
    _w()
    print("X_BRIDGE_OK", json.dumps({"x1": {s: {st: v["top8_share"] for st, v in e.items()}
                                            for s, e in rep["x1"].items()},
                                     "x2": rep["x2"]}, default=str))


if __name__ == "__main__":
    main()
