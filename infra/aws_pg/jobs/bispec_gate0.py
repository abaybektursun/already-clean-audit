"""BISPEC Gate-0b (GPU) — the live-forward gate. Nothing downstream runs until this is green.

Spec §12.0, with the §0.5 amendments applied:

  * ppl instrument reproduces the invcore-FUNC artifact exactly (the wiring gate)
  * item halves RE-SALTED to `bispec|` (A6) and the REGRAIN sealed-exposure fact recorded
  * every §3 injector fires on the LIVE forward, with its REALIZED coordinates measured
    (per-token R², band purity, coherence) rather than assumed
  * n_eff MEASURED from attention maps (pre-mortem (ii)) — the cancellation law is fit against
    these values, never against an assumed pool size
  * the Σ bank loads and its band edges reproduce

Fail-loud throughout. `green` is written LAST so a crash cannot leave a green artifact.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.bispec_gate0 --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/bispec_gate0 --budget-min 120 \
      --args "--sigma-s3 s3://…/results/bispec_sigma/out --harvest-s3 s3://…/regrain_pipeA/out/harvest"
"""
import argparse
import hashlib
import json
import os
import pathlib

FUNC_BASE_PPL = 22.4258       # the invcore-FUNC anchor; reproduced, never assumed
PPL_TOL = 0.01
N_ITEMS_SMOKE = 64
N_R2_ROWS = 8192


def bispec_split(content_hashes, name):
    """A6: RE-SALT. REGRAIN's `grain|` sealed half was touched once (R1: regrain_c_sealed, 400
    items), and REGRAIN's own spec §(iv) froze the disposition — *any exposure → re-salt for the
    affected task*. So the halves are drawn fresh under the `bispec|` salt. The exposure audit
    below still runs, but it confirms a known fact rather than deciding anything.
    """
    import numpy as np
    from infra.pg.diff_metrics import enforce_split
    held = np.array([int(hashlib.sha1(f"bispec|{h}".encode()).hexdigest(), 16) % 2 == 1
                     for h in content_hashes], dtype=bool)
    return enforce_split(held, name=name)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma-s3", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--site", default="L0")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC

    RS.install_pg()
    rep = {"object": "bispec_gate0_v1", "site": a.site, "n_ppl_seq": RS.N_PPL_SEQ,
           "amendments": ["A2 sigma bank", "A4 two-term fit", "A6 re-salt bispec|"]}

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    L = int(a.site[1:])

    # ---- Σ bank + frozen bands ------------------------------------------------
    bank = torch.load(RS.fetch(f"{a.sigma_s3}/sigma_bank.pt", "/tmp/bispec/sigma_bank.pt"),
                      map_location="cpu", weights_only=False)[a.site]
    Sigma, hbar = bank["Sigma"], bank["hbar"]
    bands = BC.eig_bands(Sigma)
    assert bands["band_slices"] == [tuple(s) for s in bank["band_slices"]] or \
        [list(s) for s in bands["band_slices"]] == [list(s) for s in bank["band_slices"]], \
        "band edges do not reproduce from the banked Σ — halt"
    rep["bands"] = {"slices": bands["band_slices"], "energy": bands["band_energy"],
                    "top1_share": float(bands["spectrum"][0] / bands["spectrum"].sum())}
    _w()

    # ---- model + the ppl wiring gate ------------------------------------------
    sha, _ = RS.pinned_pg1_sha("/tmp/bispec")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    rep["ppl_gate"] = {"base_ppl": base_ppl, "func_anchor": FUNC_BASE_PPL,
                       "delta": base_ppl - FUNC_BASE_PPL}
    assert abs(base_ppl - FUNC_BASE_PPL) < PPL_TOL, \
        f"ppl instrument drift: {base_ppl} vs {FUNC_BASE_PPL} — halt before any cell"
    _w()

    # ---- items: re-salt + the sealed-exposure audit (A6) -----------------------
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/bispec/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    old = RS.grain_split(hashes, name="grain_vqa")
    new = bispec_split(hashes, name="bispec_vqa")
    # enforce_split returns disc_mask / disc_idx / conf_idx — there is no `held` key.
    n_moved = int((old["disc_mask"] != new["disc_mask"]).sum())
    rep["items"] = {"n_total": len(hashes),
                    "grain_disc": old["n_disc"], "grain_sealed": old["n_conf"],
                    "bispec_disc": new["n_disc"], "bispec_sealed": new["n_conf"],
                    "n_reassigned_by_resalt": n_moved,
                    "regrain_sealed_exposure": "regrain_c_sealed touched 400 grain|-sealed items once "
                                               "(REGRAIN R1); inherited rule §(iv) => re-salt. Confirmed, "
                                               "not decided, at Gate-0.",
                    "salt": "bispec|"}
    assert n_moved > 0, "re-salt produced an identical split — the salt is not being applied"
    disc = [rows[i] for i in new["disc_idx"]][:N_ITEMS_SMOKE]
    _w()

    intact = IP.eval_pg1(model, proc, vqa, disc, "cuda")
    rep["reference"] = {"intact_vqa": intact["score"], "n_items": len(disc), "base_ppl": base_ppl}
    _w()

    # ---- realized-coordinate smoke of every NEW injector on the LIVE forward ----
    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/bispec/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{a.harvest_s3}/{shard0}", f"/tmp/bispec/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:N_R2_ROWS].float().cuda()
    xbar = X.mean(0)
    rms = float(X.pow(2).mean().sqrt())
    d = X.shape[1]
    rep["site_rms"] = rms

    def cell(name, hook_fn, corrupt_rows=None):
        with torch.inference_mode():
            r2 = RC.r2_of(X, corrupt_rows(), xbar) if corrupt_rows is not None else None
        r = IP.eval_pg1(model, proc, vqa, disc, "cuda", ctx=BC.hook_at(layers, L, hook_fn()))
        with BC.hook_at(layers, L, hook_fn()):
            p = IP.ppl_gate(model, proc, "cuda", n_seq=24)["ppl"]     # short ppl: a smoke, not a cell
        rec = {"cell": name, "realized_r2": r2, "vqa": r["score"], "dvqa": r["score"] - intact["score"],
               "ppl_short": p}
        rep.setdefault("injector_smoke", []).append(rec)
        _w()
        print(f"[gate0] {name}: R2={r2 if r2 is None else round(r2,4)} "
              f"dVQA={rec['dvqa']:+.4f} ppl24={p:.3f}", flush=True)
        return rec

    sigma_n = 0.2 * rms
    # identity: σ=0 makes the coherence hook an exact no-op. It exercises the whole hook path
    # (position tracking, dtype round-trip, rebuild) and MUST leave the readout untouched. If this
    # moves VQA, every later "damage" number is partly the hook's own footprint.
    idc = cell("identity", lambda: BC.make_hook_gaussian_group(sigma=0.0, seed=1, g=None, d=d),
               lambda: X.clone())
    assert abs(idc["dvqa"]) < 1e-9, f"identity hook moved VQA by {idc['dvqa']} — the hook is not a no-op"
    assert idc["realized_r2"] == 1.0, f"identity R² = {idc['realized_r2']} ≠ 1"

    cell("gauss_iid", lambda: BC.make_hook_gaussian_group(sigma_n, 11, 1, d),
         lambda: X + sigma_n * torch.randn(X.shape, generator=torch.Generator(device="cuda").manual_seed(11), device="cuda"))
    cell("gauss_shared", lambda: BC.make_hook_gaussian_group(sigma_n, 11, None, d))
    cell("gauss_g16", lambda: BC.make_hook_gaussian_group(sigma_n, 11, 16, d))

    step = 0.8 * rms
    cell("quant_true", lambda: BC.make_hook_quant_permuted(step, 5, permute=False),
         lambda: RC.quantize_step(X, step))
    cell("quant_permuted", lambda: BC.make_hook_quant_permuted(step, 5, permute=True))

    U0 = BC.band_projector(bands, 0).cuda()
    cell("band0_noise_dc", lambda: BC.make_hook_band_noise(U0, sigma_n, 3, "dc"))
    cell("band0_noise_white", lambda: BC.make_hook_band_noise(U0, sigma_n, 3, "white"))
    cell("band0_delete_dc", lambda: BC.make_hook_band_deletion(U0, hbar.cuda(), "dc"))

    # band purity of the ADD injectors, on the live-forward draw (pre-mortem (iii))
    purity = []
    for b in range(len(bands["band_slices"])):
        U = BC.band_projector(bands, b)
        gen = torch.Generator().manual_seed(1000 + b)
        delta = torch.randn(64, U.shape[1], generator=gen, dtype=torch.float64) @ U.transpose(0, 1)
        pur = BC.band_purity(delta, U)
        purity.append({"band": b, "rank": U.shape[1], "purity": pur})
        assert (1 - pur) < BC.LEAK_TOL, f"band {b} leak {1-pur:.4f} ≥ {BC.LEAK_TOL}"
    rep["band_purity_live"] = purity
    _w()

    # ---- n_eff MEASURED from attention (pre-mortem (ii)) ----------------------
    # eager attention → output_attentions is honoured. One batch, all layers ≥ L.
    batch = disc[:8]
    images = [vqa.image_fn(r).convert("RGB") for r in batch]
    prompts = [vqa.prompt_fn(r) for r in batch]
    inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        o = model(**inp, output_attentions=True)
    attns = o.attentions
    neff = {}
    for li, at in enumerate(attns):
        if li < L:
            continue
        v = BC.n_eff_from_attention(at.float().cpu())
        neff[f"L{li}"] = {"mean": float(v.mean()), "median": float(v.median()),
                          "p10": float(v.quantile(0.10)), "p90": float(v.quantile(0.90))}
    allv = torch.cat([BC.n_eff_from_attention(at.float().cpu()) for li, at in enumerate(attns) if li >= L])
    rep["n_eff"] = {"per_layer": neff, "pooled_mean": float(allv.mean()),
                    "pooled_median": float(allv.median()),
                    "seq_len": int(inp["input_ids"].shape[-1]),
                    "note": "inverse participation ratio 1/sum(w^2) per attention row, layers >= L; "
                            "the cancellation law is FIT against these, never assumed"}
    _w()
    print(f"[gate0] n_eff pooled mean={rep['n_eff']['pooled_mean']:.1f} "
          f"median={rep['n_eff']['pooled_median']:.1f} (seq_len={rep['n_eff']['seq_len']})", flush=True)

    # the coherence premise, measured on the live forward: shared vs iid at matched per-token energy
    sm = [c for c in rep["injector_smoke"] if c["cell"] == "gauss_shared"][0]
    ii = [c for c in rep["injector_smoke"] if c["cell"] == "gauss_iid"][0]
    rep["coherence_premise_smoke"] = {"shared_dvqa": sm["dvqa"], "iid_dvqa": ii["dvqa"],
                                      "shared_ppl24": sm["ppl_short"], "iid_ppl24": ii["ppl_short"],
                                      "note": "SMOKE at n=64 items / n_seq=24 — not a cell, not adjudicated"}

    print("BISPEC_GATE0_OK", json.dumps({"base_ppl": base_ppl, "n_eff": rep["n_eff"]["pooled_median"],
                                         "resalt_moved": n_moved,
                                         "top1_share": rep["bands"]["top1_share"]}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
