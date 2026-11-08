"""GRAIN T1 harvest (GPU) — PG1 forwards over DISCOVERY-half VQAv2 items (+ the blank-image
wikitext stream at the frozen 4:1 VL:text row mix), entering-residual rows captured at BOTH
sites (primary L0 + the S_vl home layer), fp16 shards + position labels banked to out/
(streamed per batch — the x_cap OOM lesson). Side channels:

  * site_stats.pt — exact fp64 h̄ + tr(C) per site over the VL stream (the ablation target +
    em-null trace, the ENDOW harvest convention);
  * svl_draws.pt — 3 disjoint-discovery-subset re-estimations of the S_vl gradient Gram at the
    secondary site (the S2 canonicity draws; the inherit_svl recipe verbatim).

Featurizer training data = discovery items ONLY (pre-mortem iv); the gate0 item bank is
FETCHED and the local split re-derivation asserted against it (drift tripwire).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_harvest --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_harvest --budget-min 240 \
      --args "--gate0-s3 s3://pg-lineage-769063704696/results/grain_gate0/out"
"""
import argparse
import json
import pathlib

VL_TARGET = 160_000
TEXT_TARGET = 40_000            # 4:1 VL:text (frozen §4 T1)
SHARD_ROWS = 50_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--n-vl-items", type=int, default=700)
    ap.add_argument("--n-text-windows", type=int, default=90)
    ap.add_argument("--n-svl-draw-items", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    from infra.pg import inherit_pg1 as IP
    from infra.pg.ctrliface_core import topk_eigvecs
    from infra.pg.inherit_core import make_pg1_pre_hook_capture, finalize_capture

    GS.install_pg()
    banks = GS.load_read_banks()
    Lp, Ls = banks["saction"]["L"], banks["svl"]["L"]
    sites = [Lp, Ls]
    rep = {"object": "grain_harvest_v1", "sites": sites, "seed": a.seed,
           "vl_target": VL_TARGET, "text_target": TEXT_TARGET}

    def _w():
        (out / "harvest_meta.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # --- items: fetch the gate0 bank; re-derive + assert (drift tripwire) ---
    it = json.loads(GS.fetch(f"{a.gate0_s3}/items_vqa.json", "/tmp/grain/items_vqa.json").read_text())
    vqa, coco = IP.pg1_tasks()
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift vs the gate0 bank — halt"
    split = GS.grain_split(hashes, name="grain_vqa")
    assert sorted(int(i) for i in split["conf_idx"]) == sorted(it["sealed_idx"]), \
        "grain split drift vs the gate0 bank — halt"
    disc_rows = [rows[i] for i in split["disc_idx"]][: a.n_vl_items]
    disc_hashes = [hashes[i] for i in split["disc_idx"]][: a.n_vl_items]
    rep["n_vl_items"] = len(disc_rows)

    sha, _ppl_ref = GS.pinned_pg1_sha()
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    rep["pg1_sha"] = sha
    image_id = model.config.image_token_index
    bos_id = proc.tokenizer.bos_token_id
    assert image_id is not None and bos_id is not None

    # --- VL stream: rows + labels at both sites; exact h̄/trace side channel ---
    row_store = {L: [] for L in sites}
    lab_store = {L: [] for L in sites}
    accs = {L: {} for L in sites}
    row_handles = [layers[L].register_forward_pre_hook(GS.make_row_capture(row_store[L], IP.D),
                                                       with_kwargs=True) for L in sites]
    sum_handles = [layers[L].register_forward_pre_hook(
        make_pg1_pre_hook_capture(accs[L], IP.D), with_kwargs=True) for L in sites]
    batches = vqa_forward_batches(proc, vqa, disc_rows, "cuda", batch_size=a.batch_size)
    with torch.no_grad():
        for inp in batches:
            model(**inp)
            lab = GS.position_labels(inp["input_ids"], image_id, bos_id).reshape(-1).cpu()
            mask = inp["attention_mask"].reshape(-1).bool().cpu()
            for L in sites:
                assert row_store[L][-1].shape[0] == lab.shape[0], "row/label misalignment"
                row_store[L][-1] = row_store[L][-1][mask]
            lab = lab[mask]
            for L in sites:
                lab_store[L].append(lab.clone())
    for h in row_handles + sum_handles:
        h.remove()
    site_stats = {}
    for L in sites:
        hbar, trace, _X, n = finalize_capture(accs[L])
        site_stats[L] = {"hbar": hbar, "trace": float(trace), "n": int(n)}
    torch.save(site_stats, out / "site_stats.pt")
    rep["site_stats_n"] = {str(L): site_stats[L]["n"] for L in sites}
    _w()

    # --- text stream: blank-image wikitext windows; keep text+BOS rows only ---
    from datasets import load_dataset
    tok = proc.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blank = _blank_image()
    text_rows = {L: [] for L in sites}
    text_labs = {L: [] for L in sites}
    tstore = {L: [] for L in sites}
    handles = [layers[L].register_forward_pre_hook(GS.make_row_capture(tstore[L], IP.D),
                                                   with_kwargs=True) for L in sites]
    n_win = 0
    with torch.no_grad():
        for start in range(0, ids.size(0), 512):
            if n_win >= a.n_text_windows:
                break
            chunk = ids[start:start + 512]
            if chunk.size(0) < 64:
                break
            inp = proc(text=[tok.decode(chunk.tolist())], images=[blank],
                       return_tensors="pt", padding=False).to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"])
            lab = GS.position_labels(inp["input_ids"], image_id, bos_id).reshape(-1).cpu()
            keep = lab != GS.POS_IMAGE
            for L in sites:
                text_rows[L].append(tstore[L].pop()[keep])
                text_labs[L].append(lab[keep])
            n_win += 1
    for h in handles:
        h.remove()
    rep["n_text_windows"] = n_win
    _w()

    # --- subsample to the frozen mix, shard, bank ---
    g = torch.Generator().manual_seed(a.seed)
    rep["shards"], rep["n_rows"], rep["mix_realized"] = {}, {}, {}
    for L in sites:
        Xvl = torch.cat(row_store[L], 0)
        Lvl = torch.cat(lab_store[L], 0)
        Xtx = torch.cat(text_rows[L], 0)
        Ltx = torch.cat(text_labs[L], 0)
        pv = torch.randperm(Xvl.shape[0], generator=g)[: min(VL_TARGET, Xvl.shape[0])]
        pt = torch.randperm(Xtx.shape[0], generator=g)[: min(TEXT_TARGET, Xtx.shape[0])]
        X = torch.cat([Xvl[pv], Xtx[pt]], 0)
        lab = torch.cat([Lvl[pv], Ltx[pt]], 0).to(torch.int8)
        mix = torch.randperm(X.shape[0], generator=g)
        X, lab = X[mix], lab[mix]
        key = f"L{L}"
        names = []
        for i in range(0, X.shape[0], SHARD_ROWS):
            name = f"shard_{key}_{i // SHARD_ROWS}.pt"
            torch.save({"X": X[i:i + SHARD_ROWS].clone(),
                        "labels": lab[i:i + SHARD_ROWS].clone()}, out / name)
            names.append(name)
        rep["shards"][key] = names
        rep["n_rows"][key] = int(X.shape[0])
        rep["mix_realized"][key] = {"vl": int(pv.numel()), "text": int(pt.numel())}
        _w()
    cfg = {"vl_target": VL_TARGET, "text_target": TEXT_TARGET, "seed": a.seed,
           "n_vl_items": len(disc_rows), "n_text_windows": n_win, "sites": sites}
    rep["manifest_sha"] = GS.manifest_sha(disc_hashes, cfg)
    _w()

    # --- stage 2: S_vl re-estimation draws (3 disjoint discovery COCO subsets; S2 machinery) ---
    itc = json.loads(GS.fetch(f"{a.gate0_s3}/items_coco.json", "/tmp/grain/items_coco.json").read_text())
    crows, chashes = IP.load_items(coco, itc["stream_window"])
    assert chashes == itc["hashes"], "coco item-stream drift vs the gate0 bank — halt"
    csplit = GS.grain_split(chashes, name="grain_coco")
    cdisc = [crows[i] for i in csplit["disc_idx"]]
    need = 3 * a.n_svl_draw_items
    assert len(cdisc) >= need, f"coco discovery half {len(cdisc)} < {need} draw items"

    def loss_fn(inp):
        o = model(**{k: v for k, v in inp.items()})
        assert o.loss is not None, "no VL loss (suffix labels missing)"
        return o.loss

    draws = {}
    for di in range(3):
        chunk_rows = cdisc[di * a.n_svl_draw_items:(di + 1) * a.n_svl_draw_items]
        bt = []
        for s in range(0, len(chunk_rows), 4):
            cc = chunk_rows[s:s + 4]
            inp = proc(text=[coco.prompt_fn(r) for r in cc],
                       images=[coco.image_fn(r).convert("RGB") for r in cc],
                       suffix=[coco.refs_fn(r)[0] for r in cc],
                       return_tensors="pt", padding="longest").to("cuda")
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            bt.append(inp)
        G, npos = IP.accumulate_gram_pre(model, layers, loss_fn, bt, [Ls], d=IP.D)
        V, evals = topk_eigvecs(G[Ls].cpu(), GS.K_READ)
        draws[di] = {"V": V.to(torch.float32), "evals": evals[:16], "n_pos": int(npos)}
    torch.save(draws, out / "svl_draws.pt")
    rep["svl_draws"] = {str(k): draws[k]["n_pos"] for k in draws}
    rep["green"] = True
    _w()
    print("GRAIN_HARVEST_OK", json.dumps({"n_rows": rep["n_rows"],
                                          "mix": rep["mix_realized"]}, default=str))


if __name__ == "__main__":
    main()
