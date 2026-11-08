"""ATLAS Gate-0 (GPU) — anchors + the RoPE-convention certification (0d) + the `atlas|` re-salt
audit. Nothing downstream runs until this is green; T-2/T-3 drivers additionally refuse to inject
unless THIS artifact's `rope_convention.certified` is true.

Anchors, each recomputed from its named artifact at run time (spec §3.2):
  * blank-image wikitext ppl (n_seq=120) vs the invcore-FUNC anchor (22.4258, tol 0.01)
  * identity hooks: the σ=0 Gaussian hook AND every NEW ATLAS injector at zero amplitude must be
    exact no-ops on the readout (bit-level for the residual hooks)
  * gauss_iid σ=0.2·rms realized R² vs the REGRAIN frontier artifact's gauss row (recomputed
    from frontier.json at run time, never from prose)
  * model sha vs the invcore-FUNC pin; MQA layout asserts (num_key_value_heads=1, k/v width 256)
  * Σ bank loads; band edges reproduce from the banked Σ

Gate-0d (NEW measurement): the RoPE convention. The model's own (cos, sin) are captured from the
live attention call and compared against `atlas_core.rope_freqs` (< 1e-4 fp32); a manual eager
attention recomputation built on `atlas_core.rope_rotate` must reproduce the layer's attention
pattern (bf16 forward → weights tolerance 2e-2, disclosed) — this certifies WHERE pre-RoPE
injection sits and that counter-rotation lands post-RoPE shifts exactly.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_gate0 --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_gate0 --budget-min 120
"""
import argparse
import json
import os
import pathlib

N_SMOKE = 64
N_R2_ROWS = 8192
PPL_TOL = 0.01
ROPE_TOL_FP32 = 1e-4
ATTN_TOL_BF16 = 2e-2


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--rope-layer", type=int, default=2)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import atlas_core as AC
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC

    freeze = AS.assert_frozen("gate0")
    RS.install_pg()
    L = int(a.site[1:])
    rep = {"object": "atlas_gate0_v1", "site": a.site, "freeze": freeze,
           "n_ppl_seq": RS.N_PPL_SEQ}

    def _w():
        (out / "gate0.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- Σ bank + frozen bands reproduce --------------------------------------
    bank = AS.load_sigma_bank(a.site)
    Sigma, hbar = bank["Sigma"], bank["hbar"]
    bands = BC.eig_bands(Sigma)
    assert [list(s) for s in bands["band_slices"]] == [list(s) for s in bank["band_slices"]], \
        "band edges do not reproduce from the banked Σ — halt"
    rep["bands"] = {"slices": bands["band_slices"],
                    "top1_share": float(bands["spectrum"][0] / bands["spectrum"].sum())}
    _w()

    # ---- model + sha + MQA asserts --------------------------------------------
    sha, _ = RS.pinned_pg1_sha("/tmp/atlas")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    tc = model.config.text_config
    assert tc.num_hidden_layers == 18 and tc.hidden_size == 2048
    assert tc.num_key_value_heads == 1, f"MQA assert failed: kv heads {tc.num_key_value_heads}"
    kp = layers[0].self_attn.k_proj
    vp = layers[0].self_attn.v_proj
    qp = layers[0].self_attn.q_proj
    assert kp.out_features == 256 and vp.out_features == 256, "k/v width must be 256"
    head_dim = tc.head_dim
    n_heads = tc.num_attention_heads
    assert head_dim == 256 and n_heads == 8 and qp.out_features == n_heads * head_dim
    rope_theta = float(getattr(tc, "rope_theta", 10000.0))
    rep["model"] = {"sha": sha, "head_dim": head_dim, "n_heads": n_heads,
                    "num_key_value_heads": tc.num_key_value_heads, "rope_theta": rope_theta}
    _w()

    # ---- ppl wiring anchor -----------------------------------------------------
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    rep["ppl_gate"] = {"base_ppl": base_ppl, "func_anchor": AS.FUNC_BASE_PPL,
                       "delta": base_ppl - AS.FUNC_BASE_PPL}
    assert abs(base_ppl - AS.FUNC_BASE_PPL) < PPL_TOL, \
        f"ppl instrument drift: {base_ppl} vs {AS.FUNC_BASE_PPL} — halt"
    _w()

    # ---- items: atlas| re-salt + exposure audit --------------------------------
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/atlas/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    import hashlib as _hl

    import numpy as np
    new = AS.atlas_split(hashes)
    old_b = np.array([int(_hl.sha1(f"bispec|{h}".encode()).hexdigest(), 16) % 2 == 1 for h in hashes])
    n_moved = int((np.asarray(new["disc_mask"]) != ~old_b).sum())
    atlas_sealed = {hashes[i] for i in new["conf_idx"]}
    bispec_sealed = {h for h, s in zip(hashes, old_b) if s}
    rep["items"] = {"n_total": len(hashes), "atlas_disc": int(new["n_disc"]),
                    "atlas_sealed": int(new["n_conf"]),
                    "n_reassigned_vs_bispec": n_moved,
                    "atlas_sealed_inter_bispec_sealed": len(atlas_sealed & bispec_sealed),
                    "audit": "banked cells enter ONLY as fit data measured under other salts; the "
                             "atlas| sealed half is touched once by atlas_m_sealed",
                    "salt": "atlas|"}
    assert n_moved > 0, "re-salt produced an identical split — the salt is not being applied"
    disc = [rows[i] for i in new["disc_idx"]][:N_SMOKE]
    intact = IP.eval_pg1(model, proc, vqa, disc, "cuda")
    rep["reference"] = {"intact_vqa": intact["score"], "n_items": len(disc)}
    _w()

    # ---- identity hooks: σ=0 Gaussian + every NEW injector at zero amplitude ----
    d = IP.D
    meta = json.loads(RS.fetch(f"{AS.REGRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/atlas/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{AS.REGRAIN_HARVEST}/{shard0}", f"/tmp/atlas/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:N_R2_ROWS].float().cuda()
    xbar = X.mean(0)
    rms = float(X.pow(2).mean().sqrt())
    rep["site_rms"] = rms

    def smoke(name, ctx_factory):
        r = IP.eval_pg1(model, proc, vqa, disc, "cuda", ctx=ctx_factory())
        dv = r["score"] - intact["score"]
        rep.setdefault("identity_smoke", []).append({"cell": name, "dvqa": dv})
        _w()
        print(f"[gate0] {name}: dVQA={dv:+.6f}", flush=True)
        return dv

    assert smoke("id_gauss0", lambda: BC.hook_at(layers, L, BC.make_hook_gaussian_group(0.0, 1, None, d))) == 0.0
    assert smoke("id_radial0", lambda: BC.hook_at(layers, L, AC.make_hook_radial(0.0))) == 0.0
    assert smoke("id_tang0", lambda: BC.hook_at(layers, L, AC.make_hook_tangential(0.0, seed=1, d=d))) == 0.0
    zed = AC.delta_const(torch.zeros(256))
    assert smoke("id_kproj0", lambda: AC.proj_hook_at(layers[L], "k_proj",
                                                      AC.make_proj_out_hook_add(zed, expect_dim=256))) == 0.0

    # ---- gauss_iid realized R² vs the frontier artifact ------------------------
    front = json.loads(RS.fetch(AS.REGRAIN_FRONTIER, "/tmp/atlas/frontier.json").read_text())
    gpt = [p for p in front["points"] if p["family"] == "gauss" and abs(p["level"] - 0.2) < 1e-9]
    assert len(gpt) == 1, "frontier gauss@0.2 row missing"
    sig = 0.2 * rms
    with torch.inference_mode():
        Xc = X + sig * torch.randn(X.shape, generator=torch.Generator(device="cuda").manual_seed(11),
                                   device="cuda")
        r2 = RC.r2_of(X, Xc, xbar)
    rep["gauss_iid_r2"] = {"measured": r2, "frontier": gpt[0]["r2"], "delta": r2 - gpt[0]["r2"]}
    assert abs(r2 - gpt[0]["r2"]) < 0.01, f"gauss_iid R² drift: {r2} vs {gpt[0]['r2']}"
    _w()

    # ---- Gate-0d: the RoPE convention, certified on the live forward -----------
    Lr = a.rope_layer
    cap = {}

    def pre_attn(_m, args, kwargs):
        cap["hs"] = (args[0] if args else kwargs["hidden_states"]).detach()
        cap["mask"] = kwargs.get("attention_mask")
        if cap["mask"] is None and len(args) > 1:
            cap["mask"] = args[1]
        pe = kwargs.get("position_embeddings")
        if pe is not None:
            cap["cos"], cap["sin"] = pe[0].detach(), pe[1].detach()
        if kwargs.get("position_ids") is not None:
            cap["pos_ids"] = kwargs["position_ids"].detach()

    def post_attn(_m, _args, outp):
        cap["attn_out"] = outp[0].detach()
        if len(outp) > 1 and outp[1] is not None:
            cap["attn_w"] = outp[1].detach()

    site_cap = {}

    def pre_site(_m, args, kwargs):
        site_cap["hs"] = (args[0] if args else kwargs["hidden_states"]).detach()

    h1 = layers[Lr].self_attn.register_forward_pre_hook(pre_attn, with_kwargs=True)
    h2 = layers[Lr].self_attn.register_forward_hook(post_attn)
    h3 = layers[L].self_attn.register_forward_pre_hook(pre_site, with_kwargs=True)
    b0 = disc[:1]                       # batch of ONE: no padding, positions are arange
    images = [vqa.image_fn(r).convert("RGB") for r in b0]
    prompts = [vqa.prompt_fn(r) for r in b0]
    inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        model(**inp, output_attentions=True)
    h1.remove()
    h2.remove()
    h3.remove()
    assert "hs" in cap and "attn_w" in cap, "attention capture failed — hook surface changed"
    T = cap["hs"].shape[1]
    pos = (cap["pos_ids"][0] if "pos_ids" in cap else torch.arange(T, device="cuda")).cpu()

    # (i) the model's own cos/sin vs the config-derived frequencies
    if "cos" in cap:
        om = AC.rope_freqs(head_dim, base=rope_theta)
        th = pos.double().unsqueeze(1) * om.unsqueeze(0)
        ours = torch.cat([th.cos(), th.cos()], dim=1)
        theirs = cap["cos"][0].cpu() if cap["cos"].dim() == 3 else cap["cos"].cpu()
        # the model's table is cast to the MODEL dtype (bf16 here: half-ULP at 1.0 = 0.0039); a
        # wrong FREQUENCY convention errs at O(1), so the gate is table-precision-aware: compare
        # against our fp64 table quantized to the captured dtype, and bound by the dtype's ULP
        ulp = {torch.bfloat16: 4e-3, torch.float16: 5e-4, torch.float32: 1e-4,
               torch.float64: 1e-9}[theirs.dtype]
        err_raw = float((ours - theirs.double()).abs().max())
        err_cast = float((ours.to(theirs.dtype).double() - theirs.double()).abs().max())
        rep["rope_convention"] = {"cos_table_dtype": str(theirs.dtype),
                                  "cos_max_err_raw": err_raw, "cos_max_err_after_cast": err_cast,
                                  "cos_tol_ulp": ulp}
        assert err_raw < ulp, f"cos table mismatch {err_raw} > {ulp} — wrong frequency convention"
    else:
        rep["rope_convention"] = {"cos_max_err_fp32": None,
                                  "note": "position_embeddings not passed as kwarg on this "
                                          "transformers version; certified via (ii) alone"}

    # (ii) manual eager attention on OUR rope_rotate vs the model's attention pattern
    with torch.inference_mode():
        hs = cap["hs"].float()
        q = layers[Lr].self_attn.q_proj(hs.to(torch.bfloat16)).float()[0]
        k = layers[Lr].self_attn.k_proj(hs.to(torch.bfloat16)).float()[0]
        qh = q.view(T, n_heads, head_dim).permute(1, 0, 2)
        q_rot = torch.stack([AC.rope_rotate(qh[h].double().cpu(), pos, base=rope_theta).float()
                             for h in range(n_heads)])
        k_rot = AC.rope_rotate(k.double().cpu(), pos, base=rope_theta).float().unsqueeze(0)
        logits = (q_rot.cuda() @ k_rot.cuda().transpose(-1, -2)) / (head_dim ** 0.5)
        if cap["mask"] is not None:
            logits = logits + cap["mask"][0, 0].float()
        w_manual = torch.softmax(logits, dim=-1)
        w_model = cap["attn_w"][0].float()
        err_w = float((w_manual - w_model).abs().max())
    rep["rope_convention"]["attn_weight_max_err_bf16"] = err_w
    rep["rope_convention"]["certified"] = bool(err_w < ATTN_TOL_BF16)
    assert err_w < ATTN_TOL_BF16, f"manual attention reproduction failed ({err_w}) — DO NOT run T-2/T-3"
    _w()

    # ---- channel output scales (T-2/T-3 energy anchors, measured AT THE SITE) ---
    assert "hs" in site_cap, "site-layer capture failed"
    with torch.inference_mode():
        hs16 = site_cap["hs"].to(torch.bfloat16)
        kv = layers[L].self_attn.k_proj(hs16).float()
        vv = layers[L].self_attn.v_proj(hs16).float()
        qv = layers[L].self_attn.q_proj(hs16).float()
    rep["channel_rms_at_site"] = {"k": float(kv.pow(2).mean().sqrt()),
                                  "v": float(vv.pow(2).mean().sqrt()),
                                  "q": float(qv.pow(2).mean().sqrt()),
                                  "note": "per-dim rms of the projection outputs at layers[site]; "
                                          "T-2/T-3 amplitudes are set in these units"}
    _w()

    # ---- banked n_eff pointer (AM-1) -------------------------------------------
    g0 = json.loads(RS.fetch(AS.BISPEC_GATE0, "/tmp/atlas/bispec_gate0.json").read_text())
    rep["n_eff_banked"] = {"pooled_median_L0": g0["n_eff"]["pooled_median"],
                           "per_layer_median": {k: v["median"] for k, v in g0["n_eff"]["per_layer"].items()},
                           "source": AS.BISPEC_GATE0}
    print("ATLAS_GATE0_OK", json.dumps({"base_ppl": base_ppl,
                                        "rope_attn_err": rep["rope_convention"]["attn_weight_max_err_bf16"],
                                        "resalt_moved": n_moved}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
