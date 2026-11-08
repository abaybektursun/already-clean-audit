"""GRAIN S3 (GPU, causal register, small) — steering dose-response for the implicated block.

Target block = the top-single-effect block of B* (from the g_disc screening table); if G
implicated no block (mode != full or empty B*), the highest-usage block runs as PURE
EXPLORATION (labeled). Traversal: the block's top-2 activation principal coordinates on the
frozen ±ασ grid, via the shifted-reconstruction splice (grain_bsf.make_pg1_pre_hook_steer).
Control = the same traversal (top pc) in 3 usage-matched random blocks. Task = COCO captions
(discovery half; §3 Variable): CIDEr + caption-shift rate vs the α=0 splice baseline.
Pre-registered signature: monotone dose-response for the implicated block, control flat.
Saturated grid / degenerate generation → SUPPORT-UNRESOLVED (§11.3).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_s_steer --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_s_steer --budget-min 240 --args "\
      --gdisc-s3 .../grain_g_disc_saction/out --dicts-s3 .../grain_train_primary/out \
      --harvest-s3 .../grain_harvest/out --gate0-s3 .../grain_gate0/out"
"""
import argparse
import json
import pathlib

ALPHAS = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gdisc-s3", required=True)
    ap.add_argument("--dicts-s3", required=True)
    ap.add_argument("--gates-s3", default=None)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--n-items", type=int, default=200)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP

    GS.install_pg()
    disc = json.loads(GS.fetch(f"{a.gdisc_s3}/g_disc.json", "/tmp/grain/g_disc.json").read_text())
    gates = json.loads(GS.fetch(f"{a.gates_s3 or a.dicts_s3}/gates.json", "/tmp/grain/gates.json").read_text())
    meta = json.loads(GS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/grain/harvest_meta.json").read_text())
    m_star = gates.get("m_star")
    assert m_star, "no m* — S3 UNRUN upstream (instrument_null pivot)"
    L = gates["L"]
    rep = {"object": "grain_s_steer_v1", "L": L, "m_star": m_star, "alphas": list(ALPHAS)}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    med_s = int(disc.get("median_seed", 0))
    ae, _c, _q, msha = GB.load_dict_bank(GS.fetch(
        f"{a.dicts_s3}/{GS.dict_bank_name(gates['site'], m_star, med_s)}", "/tmp/grain/d.pt"))
    assert msha == gates["manifest_sha"]
    Wd = ae.decoder.weight.detach()

    # --- target block: top-single-effect block of B*, else highest-usage (exploration) ---
    X, labs = GS.load_shards(a.harvest_s3, meta, f"L{L}")
    usage = GB.block_usage(ae, X[:60000], labs[:60000].long(), "cpu")
    if disc.get("mode") == "full" and disc.get("bstar"):
        singles = disc["per_seed"][str(med_s)]["singles"]
        target = max((b for b in disc["bstar"]), key=lambda b: singles[str(b)]
                     if str(b) in singles else singles[b])
        rep["target"] = {"block": int(target), "source": "bstar_top_single_effect"}
    else:
        target = int(torch.argmax(usage["freq"]).item())
        rep["target"] = {"block": target, "source": "highest_usage_EXPLORATION"}
    _w()

    # --- block principal coordinates + σ from the harvest codes ---
    ae_gpu = ae.to("cuda").eval()
    lo, hi = target * m_star, (target + 1) * m_star
    codes = []
    with torch.inference_mode():
        for s in range(0, min(120000, X.shape[0]), 8192):
            f = ae_gpu.encode(X[s:s + 8192].to("cuda").float(), use_threshold=True)
            blk = f[:, lo:hi]
            sel = ae_gpu.block_norms(f)[:, target] > 0
            if bool(sel.any()):
                codes.append(blk[sel].cpu())
    codes = torch.cat(codes, 0)
    assert codes.shape[0] >= 100, f"target block active on only {codes.shape[0]} rows"
    mu = codes.mean(0)
    C = ((codes - mu).T @ (codes - mu)) / (codes.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(C.double())
    pcs = evecs[:, -2:].flip(1).float()            # top-2 principal coordinates
    sigmas = evals[-2:].flip(0).clamp_min(0).sqrt().float()
    rep["target"]["n_code_rows"] = int(codes.shape[0])
    rep["target"]["sigmas"] = [float(x) for x in sigmas]
    _w()

    # --- usage-matched random control blocks (freq within ×2 of the target's) ---
    freq = usage["freq"]
    tfreq = float(freq[target])
    pool = [b for b in torch.nonzero(freq > 0).flatten().tolist()
            if b != target and tfreq / 2 <= float(freq[b]) <= tfreq * 2]
    if len(pool) < 3:
        pool = [b for b in torch.nonzero(freq > 0).flatten().tolist() if b != target]
    g = torch.Generator().manual_seed(0)
    controls = [pool[i] for i in torch.randperm(len(pool), generator=g)[:3].tolist()]
    rep["controls"] = controls
    _w()

    def control_pc_sigma(b):
        lo_b, hi_b = b * m_star, (b + 1) * m_star
        cc = []
        with torch.inference_mode():
            for s in range(0, min(120000, X.shape[0]), 8192):
                f = ae_gpu.encode(X[s:s + 8192].to("cuda").float(), use_threshold=True)
                sel = ae_gpu.block_norms(f)[:, b] > 0
                if bool(sel.any()):
                    cc.append(f[sel][:, lo_b:hi_b].cpu())
        cc = torch.cat(cc, 0)
        m0 = cc.mean(0)
        Cc = ((cc - m0).T @ (cc - m0)) / max(1, cc.shape[0] - 1)
        ev, evec = torch.linalg.eigh(Cc.double())
        return evec[:, -1].float(), float(ev[-1].clamp_min(0).sqrt())

    # --- eval: COCO captions, discovery half ---
    sha, _ = GS.pinned_pg1_sha()
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    _vqa, coco = IP.pg1_tasks()
    it = json.loads(GS.fetch(f"{a.gate0_s3}/items_coco.json", "/tmp/grain/items_coco.json").read_text())
    rows, hashes = IP.load_items(coco, it["stream_window"])
    assert hashes == it["hashes"], "coco item-stream drift vs the gate0 bank — halt"
    split = GS.grain_split(hashes, name="grain_coco")
    disc_rows = [rows[i] for i in split["disc_idx"]][: a.n_items]

    def captions_under(ctx):
        preds = []
        with ctx:
            for s in range(0, len(disc_rows), 16):
                batch = disc_rows[s:s + 16]
                inp = proc(text=[coco.prompt_fn(r) for r in batch],
                           images=[coco.image_fn(r).convert("RGB") for r in batch],
                           return_tensors="pt", padding="longest").to("cuda")
                inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
                n_in = inp["input_ids"].shape[-1]
                with torch.inference_mode():
                    gen = model.generate(**inp, max_new_tokens=coco.max_new_tokens,
                                         do_sample=False)
                preds += [p.strip() for p in proc.batch_decode(gen[:, n_in:],
                                                               skip_special_tokens=True)]
        return preds

    def cider_of(preds):
        from infra.pg.projector_ablation import _cider_boot
        cand = {i: [p] for i, p in enumerate(preds)}
        ref = {i: coco.refs_fn(r) for i, r in enumerate(disc_rows)}
        s, lo_, hi_ = _cider_boot(cand, ref, n_boot=200)
        return {"cider": s, "ci": [lo_, hi_]}

    import contextlib
    base_preds = captions_under(GB.splice_at(layers, L, ae_gpu))          # α=0 baseline
    rep["baseline"] = {"splice": cider_of(base_preds),
                       "intact": cider_of(captions_under(contextlib.nullcontext()))}
    _w()

    def sweep(block, pc, sigma, tag):
        curve = {}
        for alpha in ALPHAS:
            preds = captions_under(GB.steer_at(layers, L, ae_gpu, block, pc,
                                               alpha * sigma))
            shift = float(np.mean([p != b for p, b in zip(preds, base_preds)]))
            curve[str(alpha)] = {"shift_rate": shift, **cider_of(preds)}
            rep.setdefault("curves", {})[tag] = curve
            _w()
        return curve

    for j in range(2):
        sweep(target, pcs[:, j], float(sigmas[j]), f"real_pc{j}")
    for b in controls:
        pc_c, sig_c = control_pc_sigma(b)
        sweep(b, pc_c, sig_c, f"control_b{b}")

    # --- pre-registered signature: monotone on both sides + above the control band at |α|=2 ---
    def side_monotone(curve, sgn):
        seq = [curve[str(sgn * x)]["shift_rate"] for x in (0.5, 1.0, 2.0)]
        return all(seq[i] <= seq[i + 1] + 1e-9 for i in range(len(seq) - 1))

    real = rep["curves"]["real_pc0"]
    ctrl_max2 = max(max(rep["curves"][f"control_b{b}"][str(s)]["shift_rate"]
                        for s in ("2.0", "-2.0")) for b in controls)
    saturated = all(real[str(x)]["shift_rate"] > 0.98 for x in ALPHAS)
    if saturated:
        rep["signature"] = "SUPPORT-UNRESOLVED (grid saturates)"
    else:
        mono = side_monotone(real, 1) and side_monotone(real, -1)
        above = max(real["2.0"]["shift_rate"], real["-2.0"]["shift_rate"]) > ctrl_max2
        rep["signature"] = {"monotone_both_sides": mono, "above_control_at_2sigma": above,
                            "pass": bool(mono and above), "control_max_at_2sigma": ctrl_max2}
    rep["green"] = True
    _w()
    print("S_STEER_OK", json.dumps({"target": rep["target"], "signature": rep["signature"]},
                                   default=str))


if __name__ == "__main__":
    main()
