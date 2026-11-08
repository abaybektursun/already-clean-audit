"""ATLAS Arm B (GPU) — T5b: is the one-sided boundary's substrate a dormant GeGLU gate
population? Census → flip map → causal clamp, with the integrity pair adjudicated BEFORE the
branch (R-B2), plus the two curiosity cells that assign A9 a mechanism class (C-1, C-2).

The clamp is cap-at-zero (prevent switching on), NOT set-to-−∞: gelu(x) < 0 for mildly negative
x, so zeroing dormant units outright would change the clean forward and the identity-at-α=0 check
could never pass. With the cap, identity-at-α=0 holds by construction and any violation is an
instrument bug (fail loud).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_b --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_b --budget-min 300
"""
import argparse
import json
import math
import os
import pathlib

N_ITEMS = 200
N_CENSUS_ITEMS = 48
TOP_ALPHA = 3.0
DOSE_ALPHAS = (-3.0, -1.0, 1.0, 3.0)
N_RAND_CLAMP = 3
QUANT_DELTA = 0.8            # ×rms — the A9-active amplitude (inert at 0.4, banked)
N_QUANTILE_BINS = 10


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--site", default="L0")
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
    from infra.pg import atlas_core as AC
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg.inherit_core import _split_hidden

    AS.assert_frozen("b")
    RS.install_pg()
    g0 = json.loads(RS.fetch(f"{AS.ATLAS_GATE0}/gate0.json", "/tmp/atlas/gate0.json").read_text())
    assert g0.get("green"), "Gate-0 not green — Arm B refuses to run"
    L = int(a.site[1:])
    rep = {"object": "atlas_b_v1", "site": a.site, "cells": [],
           "clamp": "cap-at-zero on gate_proj pre-activations (identity at alpha=0 by construction)"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    sha, _ = RS.pinned_pg1_sha("/tmp/atlas")
    model, proc = IP.load_pg1(sha=sha)
    layers = IP.decoder_layers(model)
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/atlas/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = AS.atlas_split(hashes)
    items = [rows[i] for i in sp["disc_idx"]][: a.n_items]
    intact = IP.eval_pg1(model, proc, vqa, items, "cuda")
    base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["ppl"]
    assert abs(base_ppl - AS.FUNC_BASE_PPL) < 0.01, f"ppl anchor drift {base_ppl}"
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl,
                        "dead_floor_vqa": floor, "n_items": len(items), "salt": "atlas|"}
    _w()
    run = AS.CellRunner(model, proc, vqa, items, intact, floor, base_ppl, rep, _w)

    # ---- v₁ + the traversal hook -------------------------------------------------
    v1, sigma1, v1_meta = AS.build_v1(AS.REGRAIN_HARVEST, AS.REGRAIN_CDISC, device="cuda")
    rep["v1"] = v1_meta
    _w()

    def traverse_ctx(alpha_sigma):
        hook, _v = BC.make_hook_traverse(v1, alpha_sigma * sigma1)
        return BC.hook_at(layers, L, hook)

    # ---- census: clean vs +3σ traversal ------------------------------------------
    # THE pre-specified re-run repair (R-B2 VOID-instrument, run 1): a 48-item census
    # undersampled the eval distribution — "dormant" units fired on unseen eval tokens and the
    # cap-at-zero clamp carried a 0.056 footprint at α=0. The census now runs over the FULL eval
    # item set, making the α=0 cap a no-op on the VQA readout by construction (decode-time
    # tokens remain the disclosed residual; the identity cell still adjudicates it).
    census_rows = items

    def census_pass(ctx):
        cen = AC.GateCensus(layers)
        with cen, ctx, torch.inference_mode():
            for inp in vqa_forward_batches(proc, vqa, census_rows, "cuda", batch_size=16):
                model(**inp)
        return cen

    import contextlib
    cen_clean = census_pass(contextlib.nullcontext())
    cen_trav = census_pass(traverse_ctx(TOP_ALPHA))
    flipped = AC.flipped_population(cen_clean, cen_trav)
    dormant_counts = {f"L{Li}": int(cen_clean.dormant(Li).sum()) for Li in cen_clean.Ls}
    flip_counts = {f"L{Li}": int(flipped[Li].sum()) for Li in flipped}
    rep["census"] = {"n_rows": {f"L{Li}": cen_clean.n[Li] for Li in cen_clean.Ls},
                     "n_units": {f"L{Li}": int(cen_clean.frac_positive(Li).shape[0])
                                 for Li in cen_clean.Ls},
                     "dormant": dormant_counts, "flipped_by_plus3sigma": flip_counts,
                     "dormant_frac_positive_hist_note": "the reservoir map ships regardless of the "
                                                        "clamp verdict"}
    torch.save({"flipped": flipped,
                "dormant": {Li: cen_clean.dormant(Li) for Li in cen_clean.Ls}}, out / "gate_masks.pt")
    _w()
    n_flipped_total = sum(flip_counts.values())
    print(f"[b] census: dormant={dormant_counts} flipped={flip_counts}", flush=True)

    # ---- the clamp quartet ---------------------------------------------------------
    for al in DOSE_ALPHAS:
        run.run(f"B_dose_a{al:+.0f}", lambda aa=al: traverse_ctx(aa),
                family="B_dose", arm="unclamped", alpha=al)
    unclamped = next(c for c in rep["cells"] if c["family"] == "B_dose" and c["alpha"] == 3.0)

    def clamp_ctx(masks, alpha_sigma):
        @contextlib.contextmanager
        def ctx():
            with AC.gate_clamps_at(layers, masks):
                if alpha_sigma == 0.0:
                    yield
                else:
                    with traverse_ctx(alpha_sigma):
                        yield
        return ctx()

    idcell = run.run("B_clamp_identity_a0", lambda: clamp_ctx(flipped, 0.0),
                     family="B_clamp", arm="identity_alpha0", alpha=0.0)
    clamped = run.run("B_clamp_flipped_a3", lambda: clamp_ctx(flipped, TOP_ALPHA),
                      family="B_clamp", arm="clamped", alpha=3.0)

    rand_recovered = []
    for s in range(N_RAND_CLAMP):
        masks_r = {}
        gen = torch.Generator().manual_seed(900 + s)
        for Li in cen_clean.Ls:
            never = cen_clean.dormant(Li) & (~flipped[Li])          # dormant AND never-flipping
            idx = torch.nonzero(never).flatten()
            k = int(flipped[Li].sum())
            pick = idx[torch.randperm(idx.numel(), generator=gen)[:k]]
            m = torch.zeros_like(flipped[Li])
            m[pick] = True
            masks_r[Li] = m
        rc = run.run(f"B_clamp_random_s{s}", lambda mm=masks_r: clamp_ctx(mm, TOP_ALPHA),
                     family="B_clamp", arm="random_clamp", draw=s, alpha=3.0)
        rand_recovered.append(1.0 - rc["effect"] / max(unclamped["effect"], 1e-9))

    seed_noise = float(np.std([c["effect"] for c in rep["cells"]
                               if c["family"] == "B_clamp" and c["arm"] == "random_clamp"]))
    recovered = 1.0 - clamped["effect"] / max(unclamped["effect"], 1e-9)
    rep["rb2"] = {"n_flipped_total": n_flipped_total,
                  "recovered_frac": recovered,
                  "identity_dev": abs(idcell["effect"]),
                  "seed_noise": seed_noise,
                  "random_clamp_recovered": rand_recovered,
                  "adjudication": AC.adjudicate_rb2(
                      recovered_frac=recovered, identity_dev=abs(idcell["effect"]),
                      seed_noise=max(seed_noise, 0.005),
                      random_clamp_recovered=float(np.mean(rand_recovered)))}
    print(f"[b] R-B2: {rep['rb2']['adjudication']['verdict']} recovered={recovered:.3f} "
          f"rand={np.mean(rand_recovered):.3f}", flush=True)
    _w()

    # ---- curiosity cells (C-1, C-2; exploratory, protected slice) -------------------
    meta = json.loads(RS.fetch(f"{AS.REGRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/atlas/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{AS.REGRAIN_HARVEST}/{shard0}", f"/tmp/atlas/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:8192].float().cuda()
    rms = float(X.pow(2).mean().sqrt())
    step = QUANT_DELTA * rms

    def make_hook_quant_signflip(st):
        def hook(_m, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            x = hs.float()
            e = BC.quant_error(x, st)
            return rebuild((x - e).to(hs.dtype))
        return hook

    def make_hook_quant_qshuffle(st, seed):
        def hook(_m, args, kwargs):
            hs, rebuild = _split_hidden(args, kwargs)
            B, T, D = hs.shape
            x = hs.float()
            e = BC.quant_error(x, st)
            if T > 1:
                stat = x.mean(-1)                                # ⟨x⟩ per token
                gen = torch.Generator(device="cpu").manual_seed(int(seed))
                for b in range(B):
                    q = torch.quantile(stat[b].double().cpu(),
                                       torch.linspace(0, 1, N_QUANTILE_BINS + 1, dtype=torch.float64))
                    binid = torch.bucketize(stat[b].double().cpu(), q[1:-1])
                    perm = torch.arange(T)
                    for bin_ in range(N_QUANTILE_BINS):
                        idx = torch.nonzero(binid == bin_).flatten()
                        if idx.numel() > 1:
                            perm[idx] = idx[torch.randperm(idx.numel(), generator=gen)]
                    e[b] = e[b][perm.to(e.device)]
            return rebuild((x + e).to(hs.dtype))
        return hook

    run.run("C1_signflip_q0.8", lambda: BC.hook_at(layers, L, make_hook_quant_signflip(step)),
            family="C", arm="signflip", level=QUANT_DELTA)
    run.run("C0_true_q0.8", lambda: BC.hook_at(layers, L, BC.make_hook_quant_permuted(step, 5, permute=False)),
            family="C", arm="true", level=QUANT_DELTA)
    for s in range(3):
        run.run(f"C2_qshuffle_s{s}", lambda ss=s: BC.hook_at(layers, L, make_hook_quant_qshuffle(step, 40 + ss)),
                family="C", arm="quantile_shuffle", draw=s, level=QUANT_DELTA)
        run.run(f"C0_perm_s{s}", lambda ss=s: BC.hook_at(layers, L, BC.make_hook_quant_permuted(step, 5 + ss, permute=True)),
                family="C", arm="permuted", draw=s, level=QUANT_DELTA)

    def cmean(arm, ro="log1p_dppl"):
        return float(np.mean([c[ro] for c in rep["cells"] if c["family"] == "C" and c["arm"] == arm]))

    rep["c_reading"] = {
        "true": cmean("true"), "permuted": cmean("permuted"), "signflip": cmean("signflip"),
        "quantile_shuffle": cmean("quantile_shuffle"),
        "registered_readings": "C-1 ≈ permuted ⇒ lattice landing is the mechanism; C-1 ≈ true ⇒ "
                               "the error–token statistical dependence is; C-2 interpolating ⇒ "
                               "dependence-strength is the axis"}
    print("ATLAS_B_OK", json.dumps({"rb2": rep["rb2"]["adjudication"]["verdict"],
                                    "flipped": n_flipped_total,
                                    "c": {k: round(v, 3) for k, v in rep["c_reading"].items()
                                          if isinstance(v, float)}}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
