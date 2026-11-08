"""GRAIN T2+T3 (GPU) — train the frozen grain grid on the T1 shards at ONE site, gate every
config behaviorally (FLENS rule: behavior + cosine, never variance-EV), select m*, train the
extra seeds, bank everything (T4: downstream loads, never retrains).

Frozen grid (spec §4): W=16384, R_active=96, lr 1e-4, 4000 steps, batch 2048.
  --site primary   : m ∈ {1,4,8,16} @ seed 0 → T3 gates → m* = best held-out FVU among m>1
                     passing T3 → extra seeds {1,2} for m=1 and m* → gates for extras.
  --site secondary --mstar-from <primary gates.json> : {m=1@0, m*@{0,1,2}} (never re-selected).

T3 gates per config:
  1. splice-in on a discovery slice: ΔVQA within the em-noise band (energy-matched random
     ablation at the reconstruction-error energy, 3 seeds; δ_splice = band half-width;
     frozen rule: ΔVQA_splice ≥ em_mean − max(δ_splice, 0.01)) AND |Δppl| ≤ house tol;
     cosine(x̂, x) reported.
  2. identity/inertness: pass-through hook = bitwise no-op (once per job).
  (3. planted-structure recovery runs in the CPU dry-run suite, pre-freeze.)

Memory discipline (the 2026-07-09 OOM lesson): trainings and model NEVER coexist on the GPU —
phase A trains + banks with the model unloaded; phase B loads PG1 once and gates every banked
config; the extra-seed round unloads/reloads. cuda memory printed at every boundary.

Intercurrent rule (§3): training divergence → one retry at seed+100, then FAILED (never imputed).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.grain_train --type g6e.8xlarge,p5.4xlarge \
      --no-weights --s3-prefix results/grain_train_primary --budget-min 360 \
      --args "--site primary --harvest-s3 .../grain_t/out/harvest --gate0-s3 .../grain_gate0/out"
"""
import argparse
import json
import os
import pathlib

M_GRID = (1, 4, 8, 16)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--harvest-s3", required=True)
    ap.add_argument("--gate0-s3", required=True)
    ap.add_argument("--site", required=True, choices=("primary", "secondary"))
    ap.add_argument("--mstar", type=int, default=None, help="secondary only: the primary's m*")
    ap.add_argument("--mstar-from", default=None,
                    help="secondary only: path/URI of the primary's gates.json to read m* from")
    ap.add_argument("--full-grid", action="store_true",
                    help="secondary only: run the full grid with SITE-LOCAL m* selection "
                         "(the journal E6 per-site fork when the primary yields no m*)")
    ap.add_argument("--n-gate-items", type=int, default=200)
    ap.add_argument("--n-ppl-seq", type=int, default=60)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import gc
    import numpy as np
    import torch
    from infra.aws_pg.jobs import _grain_shared as GS
    from infra.pg import grain_bsf as GB
    from infra.pg import inherit_pg1 as IP
    from infra.pg.ctrliface_core import energy_matched_random, energy_matched_rank

    def mem(tag):
        if torch.cuda.is_available():
            print(f"[mem] {tag}: alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB "
                  f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB", flush=True)

    GS.install_pg()
    banks = GS.load_read_banks()
    L = banks["saction"]["L"] if a.site == "primary" else banks["svl"]["L"]
    key = f"L{L}"
    meta = json.loads(GS.fetch(f"{a.harvest_s3}/harvest_meta.json",
                               "/tmp/grain/harvest_meta.json").read_text())
    X, _labels = GS.load_shards(a.harvest_s3, meta, key)
    stats = torch.load(GS.fetch(f"{a.harvest_s3}/site_stats.pt", "/tmp/grain/site_stats.pt"),
                       map_location="cpu", weights_only=False)
    hbar = stats[L]["hbar"]
    train_idx, hold_idx = GS.disc_holdout_split(X.shape[0])
    Xtr, Xho = X[train_idx], X[hold_idx]
    rep = {"object": "grain_train_v2", "site": a.site, "L": L, "n_train": int(Xtr.shape[0]),
           "n_holdout": int(Xho.shape[0]), "manifest_sha": meta["manifest_sha"],
           "steps": a.steps, "configs": {}}
    if a.site == "secondary" and not a.full_grid:
        if a.mstar_from is not None:
            pg = json.loads(GS.fetch(a.mstar_from, "/tmp/grain/primary_gates.json").read_text())
            assert pg.get("m_star"), \
                f"primary pivot={pg.get('pivot')} — secondary grid has no m* (halt)"
            a.mstar = int(pg["m_star"])
        assert a.mstar is not None, "--mstar/--mstar-from/--full-grid required for secondary"

    def _w():
        (out / "gates.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    # ====================== Phase A: train + bank (NO model on GPU) ======================
    def train_config(m, seed):
        """Train + quality + bank one config. Returns cid on success, None on FAILED."""
        cid = f"m{m}_s{seed}"
        mem(f"train_start {cid}")
        ae, cfg = None, None
        for attempt_seed in (seed, seed + 100):        # one retry on divergence, then FAILED
            try:
                ae, cfg = GB.train_block_sae(Xtr, m=m, seed=attempt_seed, device="cuda",
                                             steps=a.steps)
                break
            except AssertionError as e:               # sanctioned §3 intercurrent boundary
                if "non-finite loss" not in str(e):
                    raise
                rep["configs"][cid] = {"state": "diverged", "diverged_seed": attempt_seed}
                _w()
        if ae is None:
            rep["configs"][cid] = {"state": "FAILED", "reason": "divergence after retry"}
            _w()
            return None
        q = GB.block_quality(ae, Xho, "cuda")
        name = GS.dict_bank_name(a.site, m, seed)
        GB.save_dict_bank(ae, cfg, q, out / name, meta["manifest_sha"])
        rep["configs"][cid] = {"state": "trained",
                               "cfg": {k: cfg[k] for k in ("m", "k_blk", "seed", "threshold",
                                                           "final_loss")},
                               "quality": q, "bank": name}
        del ae
        gc.collect()
        torch.cuda.empty_cache()
        mem(f"train_done {cid}")
        _w()
        return cid

    first_round = [(m, 0) for m in M_GRID] if (a.site == "primary" or a.full_grid) \
        else [(1, 0), (a.mstar, 0), (a.mstar, 1), (a.mstar, 2)]
    for m, s in first_round:
        train_config(m, s)

    # ====================== Phase B: behavioral gates (model on GPU) ======================
    model = proc = layers = None
    gate_ctx = {}

    def load_model_ctx():
        nonlocal model, proc, layers
        sha, _ = GS.pinned_pg1_sha()
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        mem("model_loaded")

    def unload_model():
        nonlocal model, proc, layers
        del model, proc, layers
        model = proc = layers = None
        gc.collect()
        torch.cuda.empty_cache()
        mem("model_unloaded")

    def build_gate_ctx():
        """Shared gate cells (intact, base ppl, inertness) — measured once; the pinned-sha
        reload is weight-identical so later rounds reuse them (deterministic greedy evals)."""
        vqa, _coco = IP.pg1_tasks()
        it = json.loads(GS.fetch(f"{a.gate0_s3}/items_vqa.json",
                                 "/tmp/grain/items_vqa.json").read_text())
        rows, hashes = IP.load_items(vqa, it["stream_window"])
        assert hashes == it["hashes"], "item-stream drift vs the gate0 bank — halt"
        split = GS.grain_split(hashes, name="grain_vqa")
        gate_rows = [rows[i] for i in split["disc_idx"]][: a.n_gate_items]
        intact = IP.eval_pg1(model, proc, vqa, gate_rows, "cuda")
        mem("intact_eval_done")
        base_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)["ppl"]
        mem("base_ppl_done")
        from infra.aws_pg.jobs._inherit_shared import vqa_forward_batches
        inp = vqa_forward_batches(proc, vqa, gate_rows[:8], "cuda")[0]
        with torch.inference_mode():
            base_logits = model(**inp).logits[:, -1, :].float().cpu()
        h = layers[L].register_forward_pre_hook(GB.make_pg1_pre_hook_passthrough(),
                                                with_kwargs=True)
        with torch.inference_mode():
            hooked = model(**inp).logits[:, -1, :].float().cpu()
        h.remove()
        assert torch.equal(base_logits, hooked), "pass-through hook NOT bitwise no-op"
        del inp, base_logits, hooked
        gc.collect()
        torch.cuda.empty_cache()
        rep["gate_slice"] = {"n": len(gate_rows), "intact": intact["score"],
                             "intact_ci": intact["ci95"], "base_ppl": base_ppl}
        rep["inertness"] = True
        _w()
        gate_ctx.update(vqa=vqa, gate_rows=gate_rows, intact=intact, base_ppl=base_ppl)

    em_cache = {}                      # r_e (rounded) → em-noise ΔVQA values (3 seeds)

    def em_band(ae):
        """Energy-matched-noise band: em-random ablation at the splice's reconstruction-error
        energy, 3 seeds, same slice. Chunked encode (bounded peak memory)."""
        n_e = min(20000, Xho.shape[0])
        E_target = trace = cos_sum = 0.0
        hb = hbar.to("cuda").float()
        with torch.inference_mode():
            for s in range(0, n_e, 4096):
                xe = Xho[s:s + 4096].to("cuda").float()
                rec = ae.decode(ae.encode(xe, use_threshold=True))
                E_target += float(((xe - rec) ** 2).sum().item())
                trace += float(((xe - hb) ** 2).sum().item())
                cos_sum += float(torch.nn.functional.cosine_similarity(xe, rec, dim=1)
                                 .sum().item())
        cos = cos_sum / n_e
        r_e = energy_matched_rank(E_target, trace, IP.D)
        r_key = int(round(r_e / 8.0) * 8) or 1
        if r_key not in em_cache:
            deltas, ppl_deltas = [], []
            for s in range(3):
                Qem, _ = energy_matched_random(IP.D, E_target, trace, 500 + s, "cpu",
                                               torch.float64)
                r = IP.eval_pg1(model, proc, gate_ctx["vqa"], gate_ctx["gate_rows"], "cuda",
                                ctx=IP.ablate_at(layers, L, Qem, hbar))
                deltas.append(r["score"] - gate_ctx["intact"]["score"])
                with IP.ablate_at(layers, L, Qem, hbar):
                    pem = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)["ppl"]
                ppl_deltas.append(pem - gate_ctx["base_ppl"])
            em_cache[r_key] = (deltas, ppl_deltas)
        d, pd = em_cache[r_key]
        return ({"r_e": r_e, "deltas": d, "mean": float(np.mean(d)),
                 "halfwidth": float((max(d) - min(d)) / 2)},
                {"deltas": pd, "mean": float(np.mean(pd)),
                 "halfwidth": float((max(pd) - min(pd)) / 2)},
                cos, E_target / max(trace, 1e-12))

    def gate_banked(cid):
        cell = rep["configs"][cid]
        if cell.get("state") != "trained":
            return
        ae, _cfg, _q, msha = GB.load_dict_bank(out / cell["bank"])
        assert msha == meta["manifest_sha"]
        ae = ae.to("cuda").eval()
        mem(f"gate_start {cid}")
        band, pband, cos, fvu_energy = em_band(ae)
        spliced = IP.eval_pg1(model, proc, gate_ctx["vqa"], gate_ctx["gate_rows"], "cuda",
                              ctx=GB.splice_at(layers, L, ae))
        with GB.splice_at(layers, L, ae):
            splice_ppl = IP.ppl_gate(model, proc, "cuda", n_seq=a.n_ppl_seq)["ppl"]
        del ae
        gc.collect()
        torch.cuda.empty_cache()
        dvqa = spliced["score"] - gate_ctx["intact"]["score"]
        dppl = splice_ppl - gate_ctx["base_ppl"]
        delta_splice = band["halfwidth"]
        vqa_ok = dvqa >= band["mean"] - max(delta_splice, 0.01)
        # T3-v2 ppl leg (journal E6): em-calibrated, house tol as the minimum slack.
        ppl_ok = dppl <= pband["mean"] + max(pband["halfwidth"], GS.PPL_TOL)
        cell.update({"state": "ok", "cosine_recon": cos, "recon_energy_frac": fvu_energy,
                     "splice_vqa": spliced["score"], "delta_vqa": dvqa,
                     "splice_ppl": splice_ppl, "delta_ppl": dppl, "em_band": band,
                     "em_ppl_band": pband, "t3_rule": "v2",
                     "delta_splice": delta_splice, "gate_pass": bool(vqa_ok and ppl_ok),
                     "gate_parts": {"vqa_ok": bool(vqa_ok), "ppl_ok_v2": bool(ppl_ok)}})
        mem(f"gate_done {cid}")
        _w()

    load_model_ctx()
    build_gate_ctx()
    for m, s in first_round:
        gate_banked(f"m{m}_s{s}")

    # ============== m* selection + extra seeds (primary, or secondary --full-grid) ==============
    if a.site == "primary" or a.full_grid:
        passing = {m: rep["configs"][f"m{m}_s0"] for m in M_GRID
                   if rep["configs"][f"m{m}_s0"].get("gate_pass")}
        m_gt1 = {m: c for m, c in passing.items() if m > 1}
        if m_gt1:
            m_star = min(m_gt1, key=lambda m: m_gt1[m]["quality"]["fvu"])
            rep["m_star"] = int(m_star)
            rep["pivot"] = "none"
            extra = [(m, s) for m in sorted({1, m_star} & set(passing)) for s in (1, 2)]
        elif 1 in passing:
            rep["m_star"] = None
            rep["pivot"] = "grain1_only"     # §11.3: block class unfaithful HERE
            extra = [(1, 1), (1, 2)]
        else:
            rep["m_star"] = None
            rep["pivot"] = "instrument_null"  # §11.3: program-wide INSTRUMENT-NULL
            extra = []
        _w()
        if extra:
            unload_model()
            for m, s in extra:
                train_config(m, s)
            load_model_ctx()
            for m, s in extra:
                gate_banked(f"m{m}_s{s}")
    else:
        rep["m_star"] = int(a.mstar)
        passing = [c for c in rep["configs"].values() if c.get("gate_pass")]
        rep["pivot"] = "none" if passing else "secondary_unfaithful"

    rep["green"] = True
    _w()
    print("GRAIN_TRAIN_OK", json.dumps(
        {"site": a.site, "m_star": rep.get("m_star"), "pivot": rep.get("pivot"),
         "gates": {k: v.get("gate_pass") for k, v in rep["configs"].items()}}, default=str))


if __name__ == "__main__":
    main()
