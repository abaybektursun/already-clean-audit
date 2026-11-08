"""ATLAS Gate-0e — the L8 residual harvest: `regrain_harvest`'s recipe, verbatim, at a
PARAMETERIZED site list (the parent derives its sites from the read banks and cannot be pointed
at L8). Same widened item stream, same GRAIN-sealed firewall assert, same 4:1 VL:text mix, same
fp16 shards with per-row position labels + stream provenance — so `bispec_sigma --sites L8`
consumes these shards exactly as it consumed the L0/L14 ones and the L8 Σ bank is
convention-identical to the banked pair (A2: centered, BOS-excluded happens downstream in sigma).

Dropped relative to the parent: the S_vl re-estimation draws (Arm-D machinery, not an ATLAS
input). Everything else is the same code path.

Pre-mortem (§9.4): OOM at L8 → re-run with --sites L4 (the pre-named substitute), journal.
"""
import argparse
import json
import pathlib

VL_TARGET = 800_000
TEXT_TARGET = 200_000
SHARD_ROWS = 100_000
STREAM_VL, STREAM_TEXT = 0, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sites", default="L8")
    ap.add_argument("--n-stream", type=int, default=8000)
    ap.add_argument("--n-vl-items", type=int, default=3400)
    ap.add_argument("--n-text-windows", type=int, default=440)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.inherit_core import finalize_capture, make_pg1_pre_hook_capture

    AS.assert_frozen("gate0")
    RS.install_pg()
    sites = [int(s.strip()[1:]) for s in a.sites.split(",")]
    rep = {"object": "atlas_harvest_v1", "sites": sites, "seed": a.seed,
           "vl_target": VL_TARGET, "text_target": TEXT_TARGET, "n_stream": a.n_stream,
           "recipe": "regrain_harvest_v1 verbatim at a parameterized site (S_vl draws dropped)"}

    def _w():
        (out / "harvest_meta.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/atlas/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, a.n_stream)
    assert hashes[: it["stream_window"]] == it["hashes"], \
        "widened stream does not extend the GRAIN item stream — halt"
    sealed_hashes = {it["hashes"][i] for i in it["sealed_idx"]}
    split = RS.grain_split(hashes, name="regrain_vqa_wide")
    disc_idx = list(split["disc_idx"])
    disc_hashes = [hashes[i] for i in disc_idx]
    leaked = sealed_hashes & set(disc_hashes)
    assert not leaked, f"{len(leaked)} sealed items leaked into the widened discovery half — halt"
    disc_rows = [rows[i] for i in disc_idx][: a.n_vl_items]
    disc_hashes = disc_hashes[: a.n_vl_items]
    rep["firewall"] = {"n_stream": len(hashes), "n_disc": int(split["n_disc"]),
                       "n_sealed": int(split["n_conf"]), "grain_sealed_n": len(sealed_hashes),
                       "leaked": 0, "n_vl_items_used": len(disc_rows)}
    _w()

    sha, _ppl_ref = RS.pinned_pg1_sha("/tmp/atlas")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    rep["pg1_sha"] = sha
    image_id = model.config.image_token_index
    bos_id = proc.tokenizer.bos_token_id
    assert image_id is not None and bos_id is not None

    # ---- VL stream ----
    row_store = {L: [] for L in sites}
    lab_store = {L: [] for L in sites}
    accs = {L: {} for L in sites}
    rh = [layers[L].register_forward_pre_hook(RS.make_row_capture(row_store[L], IP.D),
                                              with_kwargs=True) for L in sites]
    sh = [layers[L].register_forward_pre_hook(make_pg1_pre_hook_capture(accs[L], IP.D),
                                              with_kwargs=True) for L in sites]
    with torch.no_grad():
        for bi, inp in enumerate(vqa_forward_batches(proc, vqa, disc_rows, "cuda",
                                                     batch_size=a.batch_size)):
            model(**inp)
            lab = RS.position_labels(inp["input_ids"], image_id, bos_id).reshape(-1).cpu()
            mask = inp["attention_mask"].reshape(-1).bool().cpu()
            for L in sites:
                assert row_store[L][-1].shape[0] == lab.shape[0], "row/label misalignment"
                row_store[L][-1] = row_store[L][-1][mask]
            lab = lab[mask]
            for L in sites:
                lab_store[L].append(lab.clone())
            if bi % 40 == 0:
                print(f"[vl] batch {bi} rows={sum(t.shape[0] for t in row_store[sites[0]])}", flush=True)
    for h in rh + sh:
        h.remove()
    site_stats = {}
    for L in sites:
        hbar, trace, _X, n = finalize_capture(accs[L])
        site_stats[L] = {"hbar": hbar, "trace": float(trace), "n": int(n)}
    torch.save(site_stats, out / "site_stats.pt")
    rep["site_stats_n"] = {str(L): site_stats[L]["n"] for L in sites}
    _w()

    # ---- text stream (blank image, wikitext windows; drop image rows) ----
    from datasets import load_dataset
    tok = proc.tokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    blank = _blank_image()
    text_rows = {L: [] for L in sites}
    text_labs = {L: [] for L in sites}
    tstore = {L: [] for L in sites}
    th = [layers[L].register_forward_pre_hook(RS.make_row_capture(tstore[L], IP.D),
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
            lab = RS.position_labels(inp["input_ids"], image_id, bos_id).reshape(-1).cpu()
            keep = lab != RS.POS_IMAGE
            for L in sites:
                text_rows[L].append(tstore[L].pop()[keep])
                text_labs[L].append(lab[keep])
            n_win += 1
    for h in th:
        h.remove()
    rep["n_text_windows"] = n_win
    _w()

    # ---- subsample to the frozen mix, shard, bank ----
    g = torch.Generator().manual_seed(a.seed)
    rep["shards"], rep["n_rows"], rep["mix_realized"], rep["x_scale"] = {}, {}, {}, {}
    for L in sites:
        Xvl = torch.cat(row_store[L], 0)
        Lvl = torch.cat(lab_store[L], 0)
        Xtx = torch.cat(text_rows[L], 0)
        Ltx = torch.cat(text_labs[L], 0)
        row_store[L], lab_store[L], text_rows[L], text_labs[L] = [], [], [], []
        pv = torch.randperm(Xvl.shape[0], generator=g)[: min(VL_TARGET, Xvl.shape[0])]
        pt = torch.randperm(Xtx.shape[0], generator=g)[: min(TEXT_TARGET, Xtx.shape[0])]
        X = torch.cat([Xvl[pv], Xtx[pt]], 0)
        lab = torch.cat([Lvl[pv], Ltx[pt]], 0).to(torch.int8)
        stream = torch.cat([torch.full((pv.numel(),), STREAM_VL, dtype=torch.int8),
                            torch.full((pt.numel(),), STREAM_TEXT, dtype=torch.int8)], 0)
        del Xvl, Xtx, Lvl, Ltx
        mix = torch.randperm(X.shape[0], generator=g)
        X, lab, stream = X[mix], lab[mix], stream[mix]
        key = f"L{L}"
        rep["x_scale"][key] = RC.sqrt_d_scale(X, IP.D)
        names = []
        for i in range(0, X.shape[0], SHARD_ROWS):
            name = f"shard_{key}_{i // SHARD_ROWS}.pt"
            torch.save({"X": X[i:i + SHARD_ROWS].clone(),
                        "labels": lab[i:i + SHARD_ROWS].clone(),
                        "stream": stream[i:i + SHARD_ROWS].clone()}, out / name)
            names.append(name)
        rep["shards"][key] = names
        rep["n_rows"][key] = int(X.shape[0])
        rep["mix_realized"][key] = {"vl": int(pv.numel()), "text": int(pt.numel())}
        del X, lab, stream
        _w()
    cfg = {"vl_target": VL_TARGET, "text_target": TEXT_TARGET, "seed": a.seed,
           "n_vl_items": len(disc_rows), "n_text_windows": n_win, "sites": sites,
           "n_stream": a.n_stream}
    rep["manifest_sha"] = RS.manifest_sha(disc_hashes, cfg)
    print("ATLAS_HARVEST_OK", json.dumps({f"L{L}": rep["n_rows"][f"L{L}"] for L in sites}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
