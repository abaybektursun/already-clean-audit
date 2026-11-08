"""REGRAIN Arm H (GPU) — anatomy of the L14 text-stream splice collapse.

GRAIN R4: at L14 the splice is VL-adequate (m1/m4 inside the em-VQA band) but its TEXT-stream
fidelity collapses (Δppl +16 … +42 against em bands of +3–6), at every grain. Two hypotheses:

  H_mixture        the 4:1 VL:text mix starved the text stream — a PURE-TEXT dictionary fixes it
  H_prior-mismatch it fails even trained on its own stream, at matched reconstruction quality —
                   block/flat sparsity is the wrong prior for language, which is the BSF authors'
                   own conjecture (§7: "applying block sparsity directly to language … would
                   mismatch the prior"), landing empirically on a multimodal LM
  H_stream-null    it fails AND reconstructs worse — undertrained; the anatomy cannot adjudicate

H1  train the best I-config recipe on PURE-TEXT and PURE-VL L14 rows (the harvest's per-row
    `stream` provenance selects them from the same shards; nothing is stored twice)
H2  splice-gate each on its OWN stream and CROSS-stream, at the program-frozen n_seq=120, each
    against its em band AND its own rotated-error null
H3  error anatomy (descriptive): per-position-type decomposition of splice error and of the
    behavioural damage — which tokens' corruption carries the ppl collapse (BOS/sink vs content)

  ./infra/aws_pg/pg run infra.aws_pg.jobs.regrain_h_stream --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/regrain_h --budget-min 360 \
      --args "--harvest-s3 s3://…/results/regrain_harvest/out --recipe vanilla_rect_m1"
"""
import argparse
import gc
import json
import os
import pathlib

STREAM_VL, STREAM_TEXT = 0, 1


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--factorial-s3", default=None,
                    help="I2 gates.json; the promoted recipe is read from it when given")
    ap.add_argument("--recipe", default="vanilla_rect_m1",
                    help="fallback recipe name if --factorial-s3 is absent")
    ap.add_argument("--n-gate-items", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--fvu-tol", type=float, default=0.02)
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

    RS.install_pg()
    banks = RS.load_read_banks("/tmp/regrain")
    L = banks["svl"]["L"]                           # L14, the S_vl home site
    key = f"L{L}"
    meta = json.loads(RS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/regrain/harvest_meta.json").read_text())
    x_scale = float(meta["x_scale"][key])
    X, labs, stream = RS.load_shards_stream(a.harvest_s3, meta, key,
                                            workdir="/tmp/regrain/shards")
    stats = torch.load(RS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/regrain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]

    # ---- recipe: the I4-promoted config if available, else the frozen fallback ----
    recipe = a.recipe
    if a.factorial_s3:
        fg = json.loads(RS.fetch(f"{a.factorial_s3}/gates.json",
                                 "/tmp/regrain/i_gates.json").read_text())
        promoted = fg.get("promoted_to_arm_b") or []
        recipe = promoted[0] if promoted else a.recipe
        rcfg = fg["configs"][recipe]
        kind, b, W, signed = rcfg["kind"], rcfg["b"], rcfg["dict_size"], rcfg["signed"]
    else:
        from infra.aws_pg.jobs.regrain_i_factorial import GRID
        kind, b, W, signed = next((k, bb, w, s) for n, k, bb, w, s in GRID if n == recipe)

    rep = {"object": "regrain_h_v1", "L": L, "recipe": recipe, "kind": kind, "b": b,
           "dict_size": W, "signed": bool(signed), "x_scale": x_scale, "steps": a.steps,
           "n_ppl_seq": RS.N_PPL_SEQ, "manifest_sha": meta["manifest_sha"], "arms": {}}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ---- H1: per-stream training sets from the SAME shards ----
    is_text = stream == STREAM_TEXT
    idx_text = torch.nonzero(is_text).flatten()
    idx_vl = torch.nonzero(~is_text).flatten()
    rep["rows"] = {"text": int(idx_text.numel()), "vl": int(idx_vl.numel()),
                   "total": int(X.shape[0])}
    _w()

    def train_one(rows_idx, tag):
        Xs = X[rows_idx]
        tr, ho = RS.disc_holdout_split(Xs.shape[0])
        if kind == "blocktopk":
            inner, cfg = GB.train_block_sae(Xs[tr], m=b, seed=0, device="cuda", dict_size=W,
                                            r_active=RC.R_ACTIVE, steps=a.steps, signed=signed,
                                            x_scale=x_scale)
        else:
            inner, cfg = RC.train_grassmann_bsf(Xs[tr], b=b, seed=0, device="cuda", dict_size=W,
                                                r_active=RC.R_ACTIVE, steps=a.steps,
                                                x_scale=x_scale)
        ae = RC.ScaledAE(inner, x_scale)
        q = GB.block_quality(ae, Xs[ho], "cuda")
        bank = f"dict_{tag}.pt"
        RC.save_bank_any(ae, cfg, q, out / bank, meta["manifest_sha"], x_scale=x_scale)
        rep["arms"][tag] = {"n_train": int(tr.numel()), "n_holdout": int(ho.numel()),
                            "quality": q, "bank": bank}
        _w()
        del inner, ae, Xs
        gc.collect()
        torch.cuda.empty_cache()
        return out / bank

    p_text = train_one(idx_text, "pure_text")
    p_vl = train_one(idx_vl, "pure_vl")
    # the mixed baseline, same recipe, same site (the comparator for the FVU clause)
    tr_m, ho_m = RS.disc_holdout_split(X.shape[0])
    if kind == "blocktopk":
        inner, cfg = GB.train_block_sae(X[tr_m], m=b, seed=0, device="cuda", dict_size=W,
                                        r_active=RC.R_ACTIVE, steps=a.steps, signed=signed,
                                        x_scale=x_scale)
    else:
        inner, cfg = RC.train_grassmann_bsf(X[tr_m], b=b, seed=0, device="cuda", dict_size=W,
                                            r_active=RC.R_ACTIVE, steps=a.steps, x_scale=x_scale)
    ae = RC.ScaledAE(inner, x_scale)
    q_mixed = GB.block_quality(ae, X[ho_m], "cuda")
    RC.save_bank_any(ae, cfg, q_mixed, out / "dict_mixed.pt", meta["manifest_sha"],
                     x_scale=x_scale)
    rep["arms"]["mixed"] = {"quality": q_mixed, "bank": "dict_mixed.pt"}
    del inner, ae
    gc.collect()
    torch.cuda.empty_cache()
    _w()

    # cross-stream held-out FVU: does each dict reconstruct the OTHER stream?
    for tag, path in (("pure_text", p_text), ("pure_vl", p_vl),
                      ("mixed", out / "dict_mixed.pt")):
        aei, _c, _q, _m = RC.load_bank_any(path)
        aei = aei.to("cuda").eval()
        rep["arms"][tag]["fvu_on_text"] = GB.block_quality(aei, X[idx_text[:20000]], "cuda")["fvu"]
        rep["arms"][tag]["fvu_on_vl"] = GB.block_quality(aei, X[idx_vl[:20000]], "cuda")["fvu"]
        del aei
        gc.collect()
        torch.cuda.empty_cache()
        _w()

    # ---- H2: splice gates on the live forward, own-stream and cross-stream ----
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
    rep["gate_slice"] = {"intact": intact["score"], "base_ppl": base_ppl}
    _w()

    Xho_all = X[ho_m]

    def gate(tag, path):
        aei, _c, _q, _m = RC.load_bank_any(path)
        aei = aei.to("cuda").eval()
        n_e = min(20000, Xho_all.shape[0])
        E_t = tr = 0.0
        hb = hbar.to("cuda").float()
        with torch.inference_mode():
            for s in range(0, n_e, 4096):
                xe = Xho_all[s:s + 4096].to("cuda").float()
                rc = aei.decode(aei.encode(xe, use_threshold=True))
                E_t += float(((xe - rc) ** 2).sum())
                tr += float(((xe - hb) ** 2).sum())
        dv, dp = [], []
        for es in RS.EM_SEEDS:
            Qem, _r = energy_matched_random(IP.D, E_t, tr, es, "cpu", torch.float64)
            r = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda",
                            ctx=IP.ablate_at(layers, L, Qem, hbar))
            dv.append(r["score"] - intact["score"])
            with IP.ablate_at(layers, L, Qem, hbar):
                dp.append(IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"] - base_ppl)
        spl = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda", ctx=GB.splice_at(layers, L, aei))
        with GB.splice_at(layers, L, aei):
            spl_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        rot_h = RC.make_pre_hook_rotated_splice(aei, Q)
        with RC.hook_at(layers, L, rot_h):
            rot_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
        dvqa, dppl = spl["score"] - intact["score"], spl_ppl - base_ppl
        vhw, phw = (max(dv) - min(dv)) / 2, (max(dp) - min(dp)) / 2
        cell = {
            "delta_vqa": dvqa, "delta_ppl": dppl,
            "rotated_error_dppl": rot_ppl - base_ppl,
            "direction_carried_dppl": dppl - (rot_ppl - base_ppl),
            "em_vqa": {"mean": float(np.mean(dv)), "halfwidth": vhw},
            "em_ppl": {"mean": float(np.mean(dp)), "halfwidth": phw},
            "vqa_ok": bool(dvqa >= float(np.mean(dv)) - max(vhw, 0.01)),
            "ppl_ok": bool(dppl <= float(np.mean(dp)) + max(phw, RS.PPL_TOL))}
        cell["gate_pass"] = bool(cell["vqa_ok"] and cell["ppl_ok"])
        rep["arms"][tag]["gate"] = cell
        del aei
        gc.collect()
        torch.cuda.empty_cache()
        _w()
        return cell

    g_text = gate("pure_text", p_text)
    g_vl = gate("pure_vl", p_vl)
    g_mixed = gate("mixed", out / "dict_mixed.pt")

    # ---- H2 frozen adjudication ----
    verdict = RC.adjudicate_stream(
        pure_text_passes_text_gate=bool(g_text["ppl_ok"]),
        pure_text_fvu=float(rep["arms"]["pure_text"]["fvu_on_text"]),
        mixed_fvu=float(rep["arms"]["mixed"]["fvu_on_text"]), fvu_tol=a.fvu_tol)
    rep["verdict"] = verdict
    rep["adjudication_inputs"] = {
        "pure_text_ppl_ok": g_text["ppl_ok"], "pure_text_dppl": g_text["delta_ppl"],
        "pure_text_fvu_on_text": rep["arms"]["pure_text"]["fvu_on_text"],
        "mixed_fvu_on_text": rep["arms"]["mixed"]["fvu_on_text"], "fvu_tol": a.fvu_tol}
    rep["promotion"] = (
        "temporal-prior featurizer PROMOTED in the parked-frames ledger (Bhalla et al. 2026b; "
        "Lubana et al. 2025) — the BSF authors' §7 language-mismatch conjecture lands on a "
        "multimodal LM" if verdict == "H_prior-mismatch" else
        "L14 re-enters Arm B's scope with per-stream dictionaries" if verdict == "H_mixture" else
        "undertrained on its own stream — the anatomy cannot adjudicate; no promotion")
    _w()

    # ---- H3: error anatomy (descriptive) ----
    aei, _c, _q, _m = RC.load_bank_any(out / "dict_mixed.pt")
    aei = aei.to("cuda").eval()
    anat = {}
    with torch.inference_mode():
        for pt, nm in ((RS.POS_IMAGE, "image"), (RS.POS_TEXT, "text"), (RS.POS_BOS, "bos")):
            sel = torch.nonzero(labs == pt).flatten()[:20000]
            if sel.numel() < 100:
                continue
            xs = X[sel].to("cuda").float()
            rc = aei.decode(aei.encode(xs, use_threshold=True))
            err = (rc - xs)
            anat[nm] = {"n": int(sel.numel()),
                        "mean_err_norm": float(err.norm(dim=1).mean()),
                        "mean_act_norm": float(xs.norm(dim=1).mean()),
                        "rel_err": float((err.norm(dim=1) / xs.norm(dim=1).clamp_min(1e-6)).mean()),
                        "cosine": float(torch.nn.functional.cosine_similarity(xs, rc, dim=1).mean())}
    rep["error_anatomy_by_position_type"] = anat
    print("REGRAIN_H_OK", json.dumps(
        {"verdict": verdict, "pure_text_dppl": g_text["delta_ppl"],
         "pure_vl_dppl": g_vl["delta_ppl"], "mixed_dppl": g_mixed["delta_ppl"],
         "fvu_text": rep["arms"]["pure_text"]["fvu_on_text"],
         "fvu_mixed_on_text": rep["arms"]["mixed"]["fvu_on_text"],
         "anatomy": {k: round(v["rel_err"], 4) for k, v in anat.items()}}, default=str))
    # green written LAST: a crash after this point must not leave a green
    # artifact -- the drivers gate on `green` (regrain_drive*.sh wait_for_green).
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
