"""HELMHOLTZ-2 R0 — the wire rung: known-change pair + identity band + plant calibration.

Pair: gemma-3-4b-pt vs gemma-3-4b-it (instruction tuning = a REAL, benign-ish change the
diff pipeline MUST detect). Per swept residual site it computes CKA, the shipped Scope-2
SAE's diff summary + per-side FVU, the identity (split-half) band, and the plant controls
at c ∈ {0.5, 1, 2} × spectrum dose — everything the §6 thresholds freeze needs, and the
fires-on-known-change gate that licenses R1. Behavior: MCQ + cloze on both models, 570
items. BINDs (layers, SAE dirs, d_model) come from the seed job's metrics.json on S3.

GPU torch is used for forwards and SAE math; every SAE computation is parity-checked
against the pure-numpy core on a small slice (single-implementation discipline).

  pg run infra.aws_pg.jobs.helmholtz2_r0 \
      --args "--bind-s3 s3://pg-lineage-769063704696/results/h2_seed/out/metrics.json" \
      --type g6e.2xlarge --s3-prefix results/h2_r0 --budget-min 240
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time


def resolve_gemma3_layers(model):
    """Walk nested wrappers (Gemma3ForConditionalGeneration -> .model -> .language_model
    -> .layers, tolerant to transformers-version nesting) to the decoder ModuleList."""
    m = model
    for _ in range(5):
        if hasattr(m, "layers"):
            return m.layers
        for attr in ("model", "language_model"):
            if hasattr(m, attr):
                m = getattr(m, attr)
                break
        else:
            break
    raise RuntimeError(f"no .layers under {type(model).__name__}")


def harvest_multi(model, layers_ml, layer_idxs, texts, tok, device, micro=4):
    """Paired-ready multi-layer residual harvest: forward once per micro-batch, capture
    every requested layer. Returns {layer_idx: (n_tokens, d) float32 CPU} with identical
    token order across calls given identical texts."""
    import torch
    from infra.pg.gemma_scope import ResidualHarvest
    outs = {li: [] for li in layer_idxs}
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    hs = [ResidualHarvest(layers_ml, li) for li in layer_idxs]
    for h in hs:
        h.__enter__()
    try:
        for s in range(0, len(texts), micro):
            enc = tok(texts[s:s + micro], return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device)
            km = enc["attention_mask"].reshape(-1).bool()
            with torch.inference_mode():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            for li, h in zip(layer_idxs, hs):
                a = h.captured.reshape(-1, h.captured.shape[-1])[km].float().cpu()
                outs[li].append(a)
    finally:
        for h in hs:
            h.__exit__(None, None, None)
    import torch as t
    return {li: t.cat(v, 0) for li, v in outs.items()}


def sae_rates_torch(acts_cpu, sae, device, chunk=8192):
    """Firing stats + FVU for one SAE over one side's activations, GPU-chunked, numpy-core
    parity-checked on the first 128 rows. Returns (rates, mags, fvu, n_tokens)."""
    import numpy as np
    import torch
    from infra.pg import helmholtz2_core as HC
    torch.backends.cuda.matmul.allow_tf32 = False   # exact-fp32 parity with the numpy core
    # VRAM: pre-act chunk is rows x n_features fp32 -> scale rows inversely with width
    # (8192 rows @ 262k = 8.6GB, the iteration-6 OOM). 65k keeps 8192; 262k gets 2048.
    chunk = max(256, min(chunk, int(8192 * 65536 / sae["n_features"])))
    W_enc = torch.from_numpy(sae["W_enc"]).to(device)
    b_enc = torch.from_numpy(sae["b_enc"]).to(device)
    W_dec = torch.from_numpy(sae["W_dec"]).to(device)
    b_dec = torch.from_numpy(sae["b_dec"]).to(device)
    thr = torch.from_numpy(sae["threshold"]).to(device)
    acc = HC.FiringAccumulator(sae["n_features"])
    se_sum = 0.0
    n = acts_cpu.shape[0]
    mean = acts_cpu.mean(0, keepdim=True)
    var_sum = float(((acts_cpu - mean) ** 2).sum())
    for s in range(0, n, chunk):
        x = acts_cpu[s:s + chunk].to(device)
        pre = x @ W_enc + b_enc
        codes = pre * (pre > thr)
        recon = codes @ W_dec + b_dec
        se_sum += float(((x - recon) ** 2).sum().item())
        acc.add(codes.cpu().numpy())
    # parity vs the pure core on a slice: PRE-activations under fp32 tolerance, and the
    # JumpReLU gate only AWAY from the knife edge — an entry with pre ~ threshold flips
    # legitimately between BLAS backends (at 262k width a flip is ~guaranteed), and its
    # code jumps 0 <-> pre, which no elementwise tolerance can absorb.
    xs = acts_cpu[:128].numpy()
    pre_ref = xs @ sae["W_enc"] + sae["b_enc"]
    pre_t = (acts_cpu[:128].to(device) @ W_enc + b_enc).cpu().numpy()
    # Wiring-bug detector, not a BLAS-noise detector: a transpose/dtype/orientation bug wrecks
    # correlation and most entries; fp32 accumulation wobble does neither. Diagnostics printed
    # so a failure is self-describing in the log.
    _dev = np.abs(pre_ref - pre_t)
    _scale = np.abs(pre_ref) + 1.0
    _frac_bad = float((_dev > 0.05 * _scale).mean())
    _corr = float(np.corrcoef(pre_ref.ravel(), pre_t.ravel())[0, 1])
    _margin = 0.05 * (1.0 + np.abs(sae["threshold"]))
    _clear = np.abs(pre_ref - sae["threshold"]) > _margin
    _gate_mismatch = float(((pre_ref > sae["threshold"]) != (pre_t > sae["threshold"]))[_clear].mean())
    print(f"[parity] F={sae['n_features']} max_dev={_dev.max():.4f} frac_bad={_frac_bad:.2e} "
          f"corr={_corr:.8f} gate_mismatch_offedge={_gate_mismatch:.2e}", flush=True)
    assert _corr > 0.99999 and _frac_bad < 1e-4, \
        f"torch/np PRE parity FAILED (corr={_corr}, frac_bad={_frac_bad})"
    assert _gate_mismatch < 1e-4, f"GATE parity FAILED off knife edge ({_gate_mismatch})"
    assert var_sum > 0
    return acc.rates(), acc.mags(), se_sum / var_sum, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bind-s3", required=True)
    ap.add_argument("--pt-repo", default="google/gemma-3-4b-pt")
    ap.add_argument("--it-repo", default="google/gemma-3-4b-it")
    ap.add_argument("--n-per-subject", type=int, default=10)
    ap.add_argument("--n-harvest-items", type=int, default=270)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--plant-cs", default="0.5,1.0,2.0")
    ap.add_argument("--behavior-from-s3", default=None,
                    help="metrics.json URI of a prior run; reuse its behavior_pt/behavior_it (provenance recorded)")
    ap.add_argument("--tau", type=float, default=1e-3, help="active-feature rate floor (draft; frozen post-R0)")
    a = ap.parse_args()
    t0 = time.time()
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    bind_p = out / "bind.json"
    subprocess.run(["aws", "s3", "cp", a.bind_s3, str(bind_p), "--region", "us-east-1"], check=True)
    bind = json.loads(bind_p.read_text())
    d_model = bind["bind"]["d_model"]
    scope_repo = bind["scope_repo"]["repo"]
    chosen = bind["scope_repo"]["chosen_dirs"]
    import re
    lay_of = lambda d: int(re.search(r"layer_(\d+)", d).group(1))
    single_primary = sorted([d for d in chosen if "resid_post_all" not in d
                             and "width_65k" in d], key=lay_of)
    stability = [d for d in chosen if "resid_post_all" not in d and "width_65k" not in d]
    final_dir = [d for d in chosen if "resid_post_all" in d]
    assert single_primary and final_dir, (single_primary, final_dir)
    mid_layer = bind["scope_repo"]["mid_layer"]

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()

    # SAE PREFLIGHT FIRST (fail in minute 1, not minute 45): load every chosen dir once.
    from infra.pg import scope2_sae as _S2
    _widths_seen = set()
    for _d in sorted(set(chosen)):
        _sae = _S2.load_scope2_sae(scope_repo, _d, d_model)
        if _sae["n_features"] not in _widths_seen:   # minute-1 parity for EVERY width
            _widths_seen.add(_sae["n_features"])
            import torch as _t
            _r, _, _fv, _n = sae_rates_torch(_t.randn(256, d_model), _sae, "cuda")
            print(f"[R0][preflight] parity OK on random acts F={_sae['n_features']}", flush=True)
        print(f"[R0][preflight] {_d}: F={_sae['n_features']} OK", flush=True)
        del _sae

    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from infra.pg import helmholtz2_core as HC
    from infra.pg import scope2_sae as S2
    from infra.pg import text_eval as TE

    assert torch.cuda.is_available()
    device = "cuda"
    rep = {"object": "HELMHOLTZ-2 R0 wire (pt vs it, identity band, plant calibration)",
           "pt": a.pt_repo, "it": a.it_repo, "d_model": d_model, "tau_draft": a.tau,
           "sae_primary_dirs": single_primary, "sae_stability_dirs": stability,
           "sae_final_dir": final_dir, "seed": a.seed,
           "n_per_subject": a.n_per_subject, "n_harvest_items": a.n_harvest_items}
    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    print(f"[R0] loading {a.pt_repo} + {a.it_repo} (eager, bf16)", flush=True)
    m_pt = AutoModelForImageTextToText.from_pretrained(
        a.pt_repo, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    m_it = AutoModelForImageTextToText.from_pretrained(
        a.it_repo, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(a.pt_repo)

    # ---- behavior first (both models, both readouts, 570 items) ----
    beh = {}
    if a.behavior_from_s3:
        bc = out / "behavior_prev.json"
        subprocess.run(["aws", "s3", "cp", a.behavior_from_s3, str(bc), "--region", "us-east-1"], check=True)
        prev = json.loads(bc.read_text())
        for nm in ("pt", "it"):
            rep[f"behavior_{nm}"] = prev[f"behavior_{nm}"]
            beh[nm] = dict(prev[f"behavior_{nm}"])
        rep["behavior_source"] = a.behavior_from_s3
        print(f"[R0] behavior reused from artifact {a.behavior_from_s3}", flush=True)
        _w()
    for name, m in (() if a.behavior_from_s3 else (("pt", m_pt), ("it", m_it))):
        mcq = TE.mmlu_accuracy(m, tok, device, n_per_subject=a.n_per_subject)
        clz = TE.mmlu_cloze(m, tok, device, processor=None, n_per_subject=a.n_per_subject)
        beh[name] = {"mcq_acc": mcq["acc"], "mcq_pct": mcq["acc_pct"], "n": mcq["n"],
                     "cloze_acc": clz["acc"], "cloze_pct": clz["acc_pct"],
                     "mcq_per_item": mcq["per_item"], "item_hash": mcq["per_item_hash"],
                     "cloze_per_item": clz.get("per_item")}
        print(f"[R0] {name}: MCQ {mcq['acc_pct']:.2f}% cloze {clz['acc_pct']:.2f}% n={mcq['n']}", flush=True)
        rep[f"behavior_{name}"] = {k: beh[name][k] for k in ("mcq_acc", "mcq_pct", "cloze_acc", "cloze_pct", "n")}
        _w()

    # ---- harvest texts: deterministic item subset of the SAME eval family ----
    from collections import defaultdict
    from datasets import load_dataset
    ds = load_dataset(TE.MMLU_REPO, "all")
    by_subj = defaultdict(list)
    for r in ds["test"]:
        by_subj[r["subject"]].append(r)
    all_items = []
    for subj in sorted(by_subj):
        all_items += by_subj[subj][: a.n_per_subject]
    rng = np.random.default_rng(a.seed)
    idx = sorted(rng.choice(len(all_items), size=min(a.n_harvest_items, len(all_items)),
                            replace=False).tolist())
    texts = [TE._q_block(all_items[i]["question"], all_items[i]["choices"]) for i in idx]
    item_ids = [i for i in idx]
    rep["harvest"] = {"n_texts": len(texts)}

    layer_idxs = sorted({lay_of(d) for d in single_primary} | {lay_of(d) for d in final_dir})
    lm_pt = resolve_gemma3_layers(m_pt)
    lm_it = resolve_gemma3_layers(m_it)
    print(f"[R0] harvesting layers {layer_idxs} over {len(texts)} texts", flush=True)
    acts_pt = harvest_multi(m_pt, lm_pt, layer_idxs, texts, tok, device)
    acts_it = harvest_multi(m_it, lm_it, layer_idxs, texts, tok, device)
    n_tok = acts_pt[layer_idxs[0]].shape[0]
    rep["harvest"]["n_tokens"] = int(n_tok)
    for li in layer_idxs:
        assert acts_pt[li].shape == acts_it[li].shape == (n_tok, d_model)
    # per-token item id for split-half + block bootstrap
    tid = []
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for it_i, t_ in zip(item_ids, texts):
        tid += [it_i] * len(tok(t_, add_special_tokens=True)["input_ids"])
    tid = np.array(tid)
    assert len(tid) == n_tok, (len(tid), n_tok)
    half1 = np.isin(tid, np.array(item_ids)[::2])
    _w()

    # ---- per-site: CKA + SAE diff + identity band + FVU ----
    dir_by_layer = {lay_of(d): d for d in single_primary} | {lay_of(d): d for d in final_dir}
    sites = {}
    for li in layer_idxs:
        A = acts_pt[li].numpy()
        B = acts_it[li].numpy()
        cka = HC.linear_cka(A, B)
        sae = S2.load_scope2_sae(scope_repo, dir_by_layer[li], d_model)
        r_pt, _, fvu_pt, _ = sae_rates_torch(acts_pt[li], sae, device)
        r_it, _, fvu_it, _ = sae_rates_torch(acts_it[li], sae, device)
        # band and diff at IDENTICAL estimation statistics: band = pt_h1 vs pt_h2 (16k vs 16k),
        # diff = pt_h1 vs it_h2 (16k vs 16k). tau swept on CONTROL data (R0's whole purpose);
        # a low tau makes rare-feature rate noise dominate 'lost' and kills the test's power.
        r_h1, _, _, _ = sae_rates_torch(acts_pt[li][half1], sae, device)
        r_h2, _, _, _ = sae_rates_torch(acts_pt[li][~half1], sae, device)
        r_it_h2, _, _, _ = sae_rates_torch(acts_it[li][~half1], sae, device)
        by_tau = {}
        for tau in (3e-3, 1e-2, 3e-2):
            band = HC.identity_band_from_split(r_h1, r_h2, tau)
            diff = HC.sae_diff_from_rates(r_h1, r_it_h2, tau)
            by_tau[str(tau)] = {"band": band, "diff": diff,
                                "fires": bool(diff["lost"] > 2 * band["lost"] + 1e-4
                                              or diff["dlog_rate"] > 2 * band["dlog_rate"])}
        sites[str(li)] = {"dir": dir_by_layer[li], "cka_pt_it": cka,
                          "fvu_pt": fvu_pt, "fvu_it": fvu_it, "by_tau": by_tau,
                          "diff_full_tau1e2": HC.sae_diff_from_rates(r_pt, r_it, 1e-2),
                          "sae_fires_known_change": any(v["fires"] for v in by_tau.values())}
        print(f"[R0] L{li}: CKA={cka:.4f} fvu {fvu_pt:.3f}/{fvu_it:.3f} " +
              " ".join(f"tau={t}: lost={v['diff']['lost']:.3f} vs band={v['band']['lost']:.3f}"
                       + (" FIRES" if v["fires"] else "")
                       for t, v in by_tau.items()), flush=True)
        rep["sites"] = sites
        _w()
        del sae
        torch.cuda.empty_cache()

    # ---- frozen tau selection rule (control data only): smallest tau with >=3 sites firing
    tau_star = None
    for t in ("0.003", "0.01", "0.03"):
        if sum(1 for sv in sites.values() if sv["by_tau"][t]["fires"]) >= 3:
            tau_star = float(t)
            break
    rep["tau_star"] = tau_star
    print(f"[R0] tau_star={tau_star}", flush=True)
    _w()

    # ---- stability widths at the mid layer ----
    stab = {}
    for d in stability:
        sae = S2.load_scope2_sae(scope_repo, d, d_model)
        r_pt2, _, fv_p, _ = sae_rates_torch(acts_pt[mid_layer], sae, device)
        r_it2, _, fv_i, _ = sae_rates_torch(acts_it[mid_layer], sae, device)
        stab[d] = {"diff": HC.sae_diff_from_rates(r_pt2, r_it2, 1e-2),
                   "fvu_pt": fv_p, "fvu_it": fv_i}
        del sae
        torch.cuda.empty_cache()
    rep["stability_mid_layer"] = stab
    _w()

    # ---- plant calibration at the mid layer (both instruments, dose curve) ----
    plant = {}
    sae_mid = S2.load_scope2_sae(scope_repo, dir_by_layer[mid_layer], d_model)
    A_mid = acts_pt[mid_layer]
    r_clean, _, _, _ = sae_rates_torch(A_mid, sae_mid, device)
    g = torch.Generator().manual_seed(a.seed)
    u = torch.randn(d_model, generator=g)
    u = u / u.norm()
    for c in [float(x) for x in a.plant_cs.split(",")]:
        alpha = HC.plant_alpha_from_spectrum(A_mid.numpy(), c=c)
        planted = A_mid.clone()
        n_inj = int(0.5 * n_tok)
        pidx = torch.randperm(n_tok, generator=g)[:n_inj]
        planted[pidx] += alpha * u
        cka_p = HC.linear_cka(A_mid.numpy(), planted.numpy())
        r_pl, _, _, _ = sae_rates_torch(planted, sae_mid, device)
        tau_p = rep.get("tau_star") or 1e-2
        pd = HC.sae_diff_from_rates(r_clean, r_pl, tau_p)
        band = sites[str(mid_layer)]["by_tau"][str(tau_p)]["band"]
        plant[str(c)] = {"alpha": alpha, "cka_planted": cka_p, "sae_diff_planted": pd,
                         "cka_fires": bool(cka_p < 0.995),  # amended 2026-09-01: relative to identity floor (gemma-3 outlier-dominated variance; 0.9 was gemma-2-calibrated)
                         "sae_fires": bool(pd["lost"] > band["lost"] * 2 + 1e-4
                                           or pd["emergent"] > band["emergent"] * 2 + 1e-4
                                           or pd["dlog_rate"] > band["dlog_rate"] * 2)}
        print(f"[R0] plant c={c}: alpha={alpha:.1f} CKA={cka_p:.4f} "
              f"sae(lost={pd['lost']:.4f} emerg={pd['emergent']:.4f} dlog={pd['dlog_rate']:.4f}) "
              f"cka_fires={plant[str(c)]['cka_fires']} sae_fires={plant[str(c)]['sae_fires']}", flush=True)
        rep["plant_mid_layer"] = plant
        _w()

    # ---- wiring verdict ----
    any_sae_fires = rep.get("tau_star") is not None
    any_cka = any(v["cka_fires"] for v in plant.values())      # each instrument validates at its own dose
    any_sae_pl = any(v["sae_fires"] for v in plant.values())
    wiring_valid = bool(any_sae_fires and any_cka and any_sae_pl
                        and beh["pt"]["mcq_acc"] > 0.30 and beh["it"]["mcq_acc"] > 0.30)
    rep["wiring_valid"] = wiring_valid
    rep["elapsed_s"] = time.time() - t0
    # draft thresholds artifact (frozen by commit AFTER review, BEFORE R1 scoring)
    _ts = str(rep.get("tau_star") or 1e-2)
    (out / "metrics_thresholds_draft.json").write_text(json.dumps(
        {"tau": rep.get("tau_star") or 1e-2,
         "tau_rule": "smallest tau in {3e-3,1e-2,3e-2} with >=3 sites firing on the known change (control data only)",
         "per_site_identity_bands": {k: v["by_tau"][_ts]["band"] for k, v in sites.items()},
         "fires_rule": "stat > 2x identity band (and CI excl 0 where per-item units exist)",
         "plant_c_primary": 1.0, "source_run": "results/h2_r0"}, indent=1))
    _w()
    print("HELMHOLTZ2_R0_OK " + json.dumps({"wiring_valid": wiring_valid,
          "sae_fires_known_change_any": any_sae_fires,
          "cka_plant_c2": plant["2.0"]["cka_planted"]}), flush=True)
    assert wiring_valid, "R0 wiring INVALID — R1 must not launch; see metrics.json"


if __name__ == "__main__":
    main()
