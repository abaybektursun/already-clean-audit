"""REGRAIN I2+I4 (GPU) — the corrected instrument, the attribution factorial, and the gates.

Frozen grid (spec §5.2), all on the SAME expanded shards, the SAME steps/batch/LR/init protocol,
and the SAME √d input normalization — so the only moved variables are the code's SIGN and the
block dimension:

  Attribution 2×2 (vanilla, house BatchTopK selection):
      {rectified, signed} × {m=1, m=4}      W=16384, R_active=96
  Grassmannian (signed by construction, tied encoder, per-sample Π_ℓ, QR every 20 steps):
      b ∈ {1, 2, 3, 4}                      ℓ = 96/b, W ≈ 16384 (16386 at b=3, divisibility)

Per config, the three-metric discipline (§2), gates riding on (b) only:
  (a) held-out FVU + cosine                     [selection / description]
  (b) forward-splice ΔVQA + Δppl @ n_seq=120, judged against the em-calibrated band AND the
      ROTATED-ERROR NULL (same error norm, destroyed direction)                       [the gate]
  (c) MDL (Eq. 6) at the paper's four distortion levels, in √d-normalized units    [description]

Frozen adjudications (§5.5, `regrain_core.adjudicate_sign`):
  H_sign-fix       iff signed-m4 passes the L0 splice gate while rectified-m4 fails it
  H_class-boundary iff NO signed b≤4 config (vanilla or Grassmannian) passes at L0
  H_grain-cost     otherwise

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_i_factorial --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_i_factorial --budget-min 480 \
      --args "--harvest-s3 s3://…/results/regrain_harvest/out"
"""
import argparse
import gc
import json
import os
import pathlib

# (name, kind, block_dim, dict_size, signed)
GRID = (
    ("vanilla_rect_m1", "blocktopk", 1, 16384, False),
    ("vanilla_sign_m1", "blocktopk", 1, 16384, True),
    ("vanilla_rect_m4", "blocktopk", 4, 16384, False),
    ("vanilla_sign_m4", "blocktopk", 4, 16384, True),
    ("grass_b1", "grassmann", 1, 16384, True),
    ("grass_b2", "grassmann", 2, 16384, True),
    ("grass_b3", "grassmann", 3, 16386, True),      # 16386 = 3·5462 (divisibility)
    ("grass_b4", "grassmann", 4, 16384, True),
)
MDL_FRACS = (0.01, 0.05, 0.10, 0.20)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--site", default="primary", choices=("primary", "secondary"))
    ap.add_argument("--n-gate-items", type=int, default=200)
    ap.add_argument("--n-mdl-rows", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--only", default=None, help="comma list of config names (debug)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg import regrain_core as RC
    from infra.pg.ctrliface_core import energy_matched_random

    grid = [g for g in GRID if (a.only is None or g[0] in a.only.split(","))]
    RS.install_pg()
    banks = RS.load_read_banks("/tmp/regrain")
    L = banks["saction"]["L"] if a.site == "primary" else banks["svl"]["L"]
    key = f"L{L}"
    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    x_scale = float(meta["x_scale"][key])
    X, _labs, _stream = RS.load_shards_stream(a.harvest_s3, meta, key,
                                              workdir="/tmp/regrain/shards")
    stats = torch.load(RS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    tr_idx, ho_idx = RS.disc_holdout_split(X.shape[0])
    Xtr, Xho = X[tr_idx], X[ho_idx]
    rep = {"object": "regrain_i_factorial_v1", "site": a.site, "L": L, "x_scale": x_scale,
           "n_train": int(Xtr.shape[0]), "n_holdout": int(Xho.shape[0]),
           "manifest_sha": meta["manifest_sha"], "steps": a.steps, "n_ppl_seq": RS.N_PPL_SEQ,
           "rot_seed": RS.ROT_SEED, "configs": {}}

    def _w():
        (out / "gates.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def mem(tag):
        if torch.cuda.is_available():
            print(f"[mem] {tag}: {torch.cuda.memory_allocated()/2**30:.2f}GiB", flush=True)

    # ================= Phase A: train + bank + FVU + MDL (NO model on GPU) =================
    for name, kind, b, W, signed in grid:
        cid = name
        mem(f"train_start {cid}")
        if kind == "blocktopk":
            inner, cfg = GB.train_block_sae(Xtr, m=b, seed=0, device="cuda", dict_size=W,
                                            r_active=RC.R_ACTIVE, steps=a.steps, signed=signed,
                                            x_scale=x_scale)
        else:
            inner, cfg = RC.train_grassmann_bsf(Xtr, b=b, seed=0, device="cuda", dict_size=W,
                                                r_active=RC.R_ACTIVE, steps=a.steps,
                                                x_scale=x_scale)
        ae = RC.ScaledAE(inner, x_scale)
        q = GB.block_quality(ae, Xho, "cuda")                       # original units; returns to cpu
        bank = f"dict_{a.site}_{name}.pt"
        RC.save_bank_any(ae, cfg, q, out / bank, meta["manifest_sha"], x_scale=x_scale)
        # MDL lives in the featurizer's OWN √d-normalized units → feed the INNER + x_scale.
        inner = inner.to("cuda").eval()
        spec = RC.mdl_spectra(inner, Xho[: a.n_mdl_rows], device="cuda", x_scale=x_scale)
        dfp = (RC.grassmann_free_params(inner.n_blocks, b, IP.D) if kind == "grassmann"
               else RC.free_decoder_params(inner.n_blocks, b, IP.D))
        mdl = {f"{f:.2f}": RC.mdl_bits_per_token(
            n_blocks=inner.n_blocks, k_blk=inner.k_blk, n_tokens=int(meta["n_rows"][key]),
            block_fire_rates=spec["block_fire_rates"], block_code_eigs=spec["block_code_eigs"],
            residual_eigs=spec["residual_eigs"], delta=f * spec["var_ref"],
            dict_free_params=dfp) for f in MDL_FRACS}
        rep["configs"][cid] = {
            "state": "trained", "kind": kind, "b": b, "dict_size": W, "signed": bool(signed),
            "n_blocks": inner.n_blocks, "k_blk": inner.k_blk, "active_rank": inner.k_blk * b,
            "cfg": {k: cfg[k] for k in ("m", "k_blk", "seed", "threshold", "final_loss")},
            "quality": q, "mdl": mdl, "mdl_var_ref": spec["var_ref"],
            "dict_free_params": dfp, "bank": bank}
        del inner, ae
        gc.collect()
        torch.cuda.empty_cache()
        mem(f"train_done {cid}")
        _w()

    # ================= Phase B: behavioral gates (model on GPU) =================
    sha, _ = RS.pinned_pg1_sha("/tmp/regrain")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _coco = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json",
                             "/tmp/regrain/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    split = RS.grain_split(hashes, name="grain_vqa")
    gate_rows = [rows[i] for i in split["disc_idx"]][: a.n_gate_items]
    intact = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    Q = RC.haar_rotation(IP.D, RS.ROT_SEED, device="cuda", dtype=torch.float32)

    # inertness tripwire on the real forward, once
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    inp = vqa_forward_batches(proc, vqa, gate_rows[:8], "cuda")[0]
    with torch.inference_mode():
        lg0 = model(**inp).logits[:, -1, :].float().cpu()
    h = layers[L].register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(), with_kwargs=True)
    with torch.inference_mode():
        lg1 = model(**inp).logits[:, -1, :].float().cpu()
    h.remove()
    assert torch.equal(lg0, lg1), "pass-through hook NOT a bitwise no-op"
    del inp, lg0, lg1
    rep["gate_slice"] = {"n": len(gate_rows), "intact": intact["score"], "base_ppl": base_ppl,
                         "inertness": True}
    _w()

    em_cache = {}

    def em_band(ae):
        # `Xho[s:s+4096]` on the last iteration over-reads past n_e (Xho has 100k rows), so the
        # inherited `cos /= n_e` averaged 20480 summands over 20000 — producing cosine 1.0043, an
        # impossible value, in the first factorial run. E_t/tr are ratios and were unaffected.
        n_e = min(20000, Xho.shape[0])
        E_t = tr = cos = 0.0
        seen = 0
        hb = hbar.to("cuda").float()
        with torch.inference_mode():
            for s in range(0, n_e, 4096):
                xe = Xho[s:min(s + 4096, n_e)].to("cuda").float()
                seen += xe.shape[0]
                rc = ae.decode(ae.encode(xe, use_threshold=True))
                E_t += float(((xe - rc) ** 2).sum())
                tr += float(((xe - hb) ** 2).sum())
                cos += float(torch.nn.functional.cosine_similarity(xe, rc, dim=1).sum())
        assert seen == n_e, f"em_band read {seen} rows, expected {n_e}"
        cos /= seen
        r_e = int(energy_matched_random(IP.D, E_t, tr, 500, "cpu", torch.float64)[1])
        r_key = int(round(r_e / 8.0) * 8) or 1
        if r_key not in em_cache:
            dv, dp = [], []
            for es in RS.EM_SEEDS:
                Qem, _ = energy_matched_random(IP.D, E_t, tr, es, "cpu", torch.float64)
                r = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda",
                                ctx=IP.ablate_at(layers, L, Qem, hbar))
                dv.append(r["score"] - intact["score"])
                with IP.ablate_at(layers, L, Qem, hbar):
                    dp.append(IP.ppl_gate(model, proc, "cuda",
                                          n_seq=RS.N_PPL_SEQ)["ppl"] - base_ppl)
            em_cache[r_key] = (dv, dp)
        dv, dp = em_cache[r_key]
        return ({"r_e": r_e, "deltas": dv, "mean": float(np.mean(dv)),
                 "halfwidth": float((max(dv) - min(dv)) / 2)},
                {"deltas": dp, "mean": float(np.mean(dp)),
                 "halfwidth": float((max(dp) - min(dp)) / 2)},
                cos, E_t / max(tr, 1e-12))

    for name, kind, b, W, signed in grid:
        cell = rep["configs"][name]
        ae, _cfg, _q, msha = RC.load_bank_any(out / cell["bank"])
        assert msha == meta["manifest_sha"]
        ae = ae.to("cuda").eval()
        mem(f"gate_start {name}")
        vband, pband, cos, energy_frac = em_band(ae)

        spl = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=GB.splice_at(layers, L, ae))
        with GB.splice_at(layers, L, ae):
            spl_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        rot_hook = RC.make_pre_hook_rotated_splice(ae, Q)
        rot = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=RC.hook_at(layers, L, rot_hook))
        with RC.hook_at(layers, L, rot_hook):
            rot_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]

        dvqa, dppl = spl["score"] - intact["score"], spl_ppl - base_ppl
        rvqa, rppl = rot["score"] - intact["score"], rot_ppl - base_ppl
        vqa_ok = dvqa >= vband["mean"] - max(vband["halfwidth"], 0.01)
        ppl_ok = dppl <= pband["mean"] + max(pband["halfwidth"], RS.PPL_TOL)
        cell.update({
            "cosine_recon": cos, "recon_energy_frac": energy_frac,
            "splice": {"vqa": spl["score"], "dvqa": dvqa, "ppl": spl_ppl, "dppl": dppl},
            "rotated_error_null": {"vqa": rot["score"], "dvqa": rvqa, "ppl": rot_ppl,
                                   "dppl": rppl},
            "direction_carried": {"dvqa": dvqa - rvqa, "dppl": dppl - rppl,
                                  "note": "true splice minus its own rotated-error null: the "
                                          "part of the damage carried by the error DIRECTION"},
            "em_vqa_band": vband, "em_ppl_band": pband,
            "gate_parts": {"vqa_ok": bool(vqa_ok), "ppl_ok": bool(ppl_ok)},
            "gate_pass": bool(vqa_ok and ppl_ok), "state": "ok"})
        ae = ae.to("cpu")
        del ae
        gc.collect()
        torch.cuda.empty_cache()
        mem(f"gate_done {name}")
        _w()

    # ================= I4: frozen adjudication + promotion =================
    def passed(n):
        return bool(rep["configs"].get(n, {}).get("gate_pass"))

    signed_small_b = [n for n, k, b, W, s in grid if s and b <= 4]
    verdict = RC.adjudicate_sign(
        signed_m4_pass=passed("vanilla_sign_m4"),
        rectified_m4_pass=passed("vanilla_rect_m4"),
        any_signed_small_b_pass=any(passed(n) for n in signed_small_b))
    rep["sign_verdict"] = verdict
    rep["attribution_2x2"] = {n: {"gate_pass": passed(n),
                                  "dvqa": rep["configs"][n]["splice"]["dvqa"],
                                  "dppl": rep["configs"][n]["splice"]["dppl"]}
                              for n, k, b, W, s in grid if k == "blocktopk"}

    # promote ≤2 gate-passing BLOCK (b>1) configs to Arm B, ranked by MDL at the 20% floor
    block_pass = [n for n, k, b, W, s in grid if b > 1 and passed(n)]
    ranked = sorted(block_pass, key=lambda n: rep["configs"][n]["mdl"]["0.20"]["total"])
    rep["promoted_to_arm_b"] = ranked[:2]
    rep["arm_b_status"] = "runs" if ranked else "certificate (no gate-passing block config)"
    rep["mdl_ranking_20pct"] = {n: rep["configs"][n]["mdl"]["0.20"]["total"]
                                for n, k, b, W, s in grid}
    print("I_FACTORIAL_OK", json.dumps(
        {"sign_verdict": verdict, "gates": {n: passed(n) for n, k, b, W, s in grid},
         "promoted": rep["promoted_to_arm_b"],
         "dvqa": {n: round(rep["configs"][n]["splice"]["dvqa"], 4) for n, k, b, W, s in grid},
         "dppl": {n: round(rep["configs"][n]["splice"]["dppl"], 3) for n, k, b, W, s in grid}},
        default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
