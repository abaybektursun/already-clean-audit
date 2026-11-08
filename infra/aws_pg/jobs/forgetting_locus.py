"""forgetting-locus — is the load-bearing invariant core where fine-tuning forgets?

Child of invariant_core_func.py (the untouched q/k input subspace U_unt is LIVE / load-bearing).
invariant_core says fine-tunes write LITTLE to U_unt; FUNC says U_unt is load-bearing. So when a
fine-tune forgets a base capability, the forgetting is either CARRIED BY ΔW's small U_unt-reading
(core-locus) or lives entirely in U_unt's COMPLEMENT (complement-locus). FUNC cannot tell these apart.

POST-HOC WEIGHT SURGERY (no retraining). For a fine-tune W_ft = W0 + ΔW, removing its reading of an
input subspace span(Q) while KEEPING the base's reading is:
    W_surg = W0 + ΔW (I − Q Qᵀ) = W_ft − ΔW Q Qᵀ        (on q_proj AND k_proj, every layer)
RECOVERY = ppl(W_ft) − ppl(W_surg) (positive ⇒ base competence restored). REMOVAL FRACTION =
‖ΔW Q Qᵀ‖²_F / ‖ΔW‖²_F. The question is recovery PER UNIT removed: if removing the tiny U_unt-component
recovers disproportionately more base capability than matched-random / bottom-σ ⇒ the invariant core
is the forgetting locus (core-locus) ⇒ an invariant-core-protected fine-tuning method is motivated.

PRECISION. U_unt and ΔW are built from the canonical fp64 decoder weights (`_load_decoder`, exactly as
FUNC defined U_unt). The forward (ppl) runs in bf16, matching FUNC. Per fine-tune we load the decoder
ONCE (precise weights, downloads the snapshot) and the full model ONCE (bf16 forward, cache hit) — one
download per fine-tune. The surgical weight W_surg = bf16(W_ft − ΔW QQᵀ) is set on the live model; its
bf16-cast baseline equals the live model's own loaded q/k (both bf16 of the same checkpoint).

Reuses the FUNC substrate VERBATIM (imported). FAIL LOUD (no try/except, no synthetic fallback).

  # smoke (3-ft cloud, tiny eval, FORGET_MIN forced negative so the surgery path runs):
  pg run infra.aws_pg.jobs.forgetting_locus --args="--smoke" --no-weights \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/forget_locus_smoke --budget-min 120
  # full:
  pg run infra.aws_pg.jobs.forgetting_locus --args="--k-grid 32,128 --n-ppl-seq 120" --no-weights \
      --type p5.4xlarge,g6e.8xlarge --s3-prefix results/forget_locus --budget-min 420
"""
import argparse
import json
import pathlib

from infra.aws_pg.jobs.invariant_core import (
    PG1_REPO,
    N_LAYERS,
    _load_decoder,
    _delete_cache,
)
from infra.aws_pg.jobs.invariant_core_func import (
    _bottom_eigvecs,
    _base_bottom_sigma,
    _base_top_sigma,
    _random_subspace,
    _ppl_blankimg,
    _mmlu_blankimg,
    _decoder_layers,
)

FORGET_MIN = 1.0        # a fine-tune "forgets recoverably via q/k" iff ppl_ft − ppl_fullqk >= this.
N_RAND_DRAWS = 3        # random null is a band over >=3 draws (matches FUNC).


def _load_full(repo, sha, device, dtype, attn="eager"):
    """Load any full PaliGemma model (base OR fine-tune) for the forward. (FUNC's _load_full_pg1
    hardcodes the base repo; fine-tunes need their own repo id.) MUST use EAGER: PaliGemma's prefix-LM
    attention is bidirectional over the image+prefix and causal over the suffix; sdpa mishandles that
    mask and gives WRONG forwards (base blank-image ppl 36.8 vs the FUNC-validated eager 22.4)."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    return PaliGemmaForConditionalGeneration.from_pretrained(
        repo, revision=sha, torch_dtype=dtype, device_map=device,
        attn_implementation=attn).eval()


def _qk(layers, L):
    return layers[L].self_attn.q_proj, layers[L].self_attn.k_proj


def _snapshot_qk(layers, want_layers):
    """Clone the live (bf16) q/k weights to restore between surgical arms."""
    snap = {}
    for L in want_layers:
        q, k = _qk(layers, L)
        snap[(L, "q")] = q.weight.detach().clone()
        snap[(L, "k")] = k.weight.detach().clone()
    return snap


def _restore_qk(layers, snap, want_layers):
    import torch
    for L in want_layers:
        q, k = _qk(layers, L)
        with torch.no_grad():
            q.weight.copy_(snap[(L, "q")])
            k.weight.copy_(snap[(L, "k")])


def _set_qk_to_base(layers, W0q, W0k, want_layers, device):
    """Set q/k of every layer to BASE weights (remove ALL ΔW on q/k = max q/k-recoverable forgetting)."""
    import torch
    for L in want_layers:
        q, k = _qk(layers, L)
        with torch.no_grad():
            q.weight.copy_(torch.as_tensor(W0q[L], device=device, dtype=q.weight.dtype))
            k.weight.copy_(torch.as_tensor(W0k[L], device=device, dtype=k.weight.dtype))


def _surgery(layers, Wftq, Wftk, W0q, W0k, Q_by_layer, want_layers, device):
    """Patch q/k in place to W_surg = W_ft − ΔW Q Qᵀ (ΔW = W_ft − W0), using the PRECISE fp64 decoder
    weights for the math and casting the result to the live model dtype. Returns the removal fraction
    ‖ΔW Q Qᵀ‖²/‖ΔW‖² over q+k, all layers."""
    import numpy as np
    import torch
    num = 0.0
    den = 0.0
    for L in want_layers:
        Q = np.asarray(Q_by_layer[L], dtype=np.float64)           # n x k
        QQt = Q @ Q.T                                             # n x n
        for tag, proj, Wft, W0 in (("q", layers[L].self_attn.q_proj, Wftq[L], W0q[L]),
                                   ("k", layers[L].self_attn.k_proj, Wftk[L], W0k[L])):
            dW = np.asarray(Wft, dtype=np.float64) - np.asarray(W0, dtype=np.float64)   # m x n
            removed = dW @ QQt                                                          # m x n
            num += float((removed ** 2).sum())
            den += float((dW ** 2).sum())
            Wsurg = np.asarray(Wft, dtype=np.float64) - removed
            with torch.no_grad():
                proj.weight.copy_(torch.as_tensor(Wsurg, device=device, dtype=proj.weight.dtype))
    return num / den if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ft-tasks", default="", help="comma slugs; empty => all -224 (the N5c cloud)")
    ap.add_argument("--layers", default="", help="comma layer idxs; empty => all 18")
    ap.add_argument("--k-grid", default="32,128")
    ap.add_argument("--n-ppl-seq", type=int, default=120, help="windows for the surgery-arm ppl")
    ap.add_argument("--n-precheck", type=int, default=40, help="windows for the Pass-1 forgetting ranking")
    ap.add_argument("--max-surgery", type=int, default=10, help="cap on #forgetting ft taken to surgery")
    ap.add_argument("--ppl-window", type=int, default=512)
    ap.add_argument("--forget-min", type=float, default=FORGET_MIN)
    ap.add_argument("--n-mmlu-per", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cached-mmlu", type=float, default=25.49)
    ap.add_argument("--cached-tol", type=float, default=3.0)
    ap.add_argument("--cached-ppl", type=float, default=22.43,
                    help="FUNC-validated base blank-image ppl anchor (eager); hard preflight gate")
    ap.add_argument("--cached-ppl-tol", type=float, default=3.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    k_grid = [int(x) for x in a.k_grid.split(",")]
    forget_min = a.forget_min
    if a.smoke:
        k_grid = [32]
        a.n_ppl_seq = 8
        a.n_precheck = 6
        a.max_surgery = 3
        want_layers = [0, 9, 17]
        forget_min = -1e9      # force every ft into the forgetting set so the surgery path is exercised
        if not a.ft_tasks.strip():
            a.ft_tasks = "vqav2,gqa,ai2d"
    else:
        want_layers = ([int(x) for x in a.layers.split(",") if x.strip()]
                       if a.layers.strip() else list(range(N_LAYERS)))

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"
    bf16 = torch.bfloat16
    api = HfApi()
    n = 2048

    if a.ft_tasks.strip():
        repos = [f"google/paligemma-3b-ft-{t.strip()}-224" for t in a.ft_tasks.split(",") if t.strip()]
    else:
        repos = sorted({m.id for m in api.list_models(author="google", search="paligemma-3b-ft")
                        if (m.id.split("/")[-1].endswith("-224") and "-jax" not in m.id
                            and m.id.split("/")[-1].startswith("paligemma-3b-ft-"))})
    shas = {r: api.model_info(r).sha for r in repos}
    base_sha = api.model_info(PG1_REPO).sha
    print(f"[forget] base={PG1_REPO}@{base_sha[:8]} cloud={len(repos)} layers={want_layers} "
          f"k={k_grid} forget_min={forget_min} smoke={a.smoke} gpu={torch.cuda.get_device_name(0)}",
          flush=True)

    # ---- canonical fp64 base q/k (FUNC-consistent U_unt + ΔW definition).
    w_base = _load_decoder(PG1_REPO, base_sha)
    W0q = {L: w_base[f"layers.{L}.self_attn.q_proj.weight"].to(torch.float64).cpu().numpy()
           for L in want_layers}
    W0k = {L: w_base[f"layers.{L}.self_attn.k_proj.weight"].to(torch.float64).cpu().numpy()
           for L in want_layers}
    base_W0q_torch = {L: w_base[f"layers.{L}.self_attn.q_proj.weight"] for L in want_layers}

    rep = {"object": "forgetting_locus_v1", "base_repo": PG1_REPO, "base_sha": base_sha,
           "cloud_repos": repos, "cloud_shas": shas, "n_ft": len(repos), "layers": want_layers,
           "k_grid": k_grid, "seed": a.seed, "smoke": a.smoke, "forget_min": forget_min,
           "n_rand_draws": N_RAND_DRAWS,
           "surgery": "W_surg = W_ft − ΔW Q Qᵀ on q_proj+k_proj (fp64 weights); recovery = ppl_ft − ppl_surg"}

    processor = AutoProcessor.from_pretrained(PG1_REPO, revision=base_sha)

    # ---- PRE: base full model (bf16) — preflight + base ppl.
    base_model = _load_full(PG1_REPO, base_sha, device, bf16)
    Qr = _random_subspace(n, k_grid[0], a.seed)
    Qt = torch.as_tensor(Qr, device=device, dtype=torch.float64)
    xr = torch.randn(3, 5, n, device=device, dtype=torch.float64)
    x1 = xr - (xr @ Qt) @ Qt.transpose(0, 1)
    proj_resid = float(((x1 - (x1 @ Qt) @ Qt.transpose(0, 1)) - x1).abs().max())
    assert proj_resid < 1e-9, f"projector not idempotent: {proj_resid}"
    pre_mmlu = _mmlu_blankimg(base_model, processor, device,
                              n_per_subject=(2 if a.smoke else a.n_mmlu_per))
    d_cached = abs(pre_mmlu["acc_pct"] - a.cached_mmlu)
    base_ppl = _ppl_blankimg(base_model, processor, device, n_seq=a.n_ppl_seq, window=a.ppl_window)["ppl"]
    d_ppl = abs(base_ppl - a.cached_ppl)
    preflight_ok = bool(a.smoke or (d_cached <= a.cached_tol and d_ppl <= a.cached_ppl_tol))
    rep["preflight"] = {"projector_resid": proj_resid, "native_blankimg_mmlu_pct": pre_mmlu["acc_pct"],
                        "cached_mmlu": a.cached_mmlu, "delta": d_cached, "tol": a.cached_tol,
                        "base_ppl": base_ppl, "cached_ppl": a.cached_ppl, "ppl_delta": d_ppl,
                        "ppl_tol": a.cached_ppl_tol, "ok": preflight_ok}
    (out / "forgetting_locus.json").write_text(json.dumps(rep, indent=2))
    print(f"[forget][PRE] proj={proj_resid:.2e} MMLU={pre_mmlu['acc_pct']:.2f} base_ppl={base_ppl:.3f} "
          f"(cached {a.cached_ppl}, |d|={d_ppl:.2f}) ok={preflight_ok}", flush=True)
    assert preflight_ok, (f"PREFLIGHT FAIL: MMLU {pre_mmlu['acc_pct']} vs {a.cached_mmlu} (|d|{d_cached:.2f}) "
                          f"or base_ppl {base_ppl:.2f} vs {a.cached_ppl} (|d|{d_ppl:.2f}) — wrong forward?")
    del base_model
    _delete_cache(PG1_REPO)
    torch.cuda.empty_cache()

    # ---- GRAM + PRECHECK (Pass 1): per ft load decoder (fp64 Gram) + full model (bf16 ppl_ft).
    import time
    G = {L: np.zeros((n, n), dtype=np.float64) for L in want_layers}
    ppl_ft = {}
    for ridx, repo in enumerate(repos):
        sha = shas[repo]
        print(f"[forget][gram] {repo}@{sha[:8]} ({ridx+1}/{len(repos)}) ...", flush=True)
        t0 = time.time()
        w_ft = _load_decoder(repo, sha)
        for L in want_layers:
            Wq = w_ft[f"layers.{L}.self_attn.q_proj.weight"].to(torch.float64).cpu().numpy()
            dWq = Wq - W0q[L]
            fro2 = float((dWq ** 2).sum())
            assert fro2 > 0, f"zero ΔW q_proj L{L} in {repo}"
            G[L] += (dWq.T @ dWq) / fro2
        del w_ft
        t1 = time.time()
        model = _load_full(repo, sha, device, bf16)
        t2 = time.time()
        ppl_ft[repo] = _ppl_blankimg(model, processor, device, n_seq=a.n_precheck, window=a.ppl_window)["ppl"]
        t3 = time.time()
        del model
        _delete_cache(repo)
        torch.cuda.empty_cache()
        print(f"[forget][gram]   decoder+gram {t1-t0:.0f}s | full-load {t2-t1:.0f}s | ppl({a.n_precheck}w) "
              f"{t3-t2:.0f}s | ppl_ft={ppl_ft[repo]:.2f}", flush=True)
    rep["ppl_ft_precheck"] = ppl_ft
    rep["base_ppl"] = base_ppl
    rep["n_precheck"] = a.n_precheck
    all_forgetting = sorted([r for r in repos if ppl_ft[r] - base_ppl >= forget_min],
                            key=lambda r: ppl_ft[r] - base_ppl, reverse=True)
    forgetting = all_forgetting[:a.max_surgery]      # cap the surgery set (no silent truncation: logged)
    dropped = all_forgetting[a.max_surgery:]
    rep["whole_model_forgetting_precheck"] = {r: ppl_ft[r] - base_ppl for r in repos}
    rep["forgetting_all"] = all_forgetting
    rep["forgetting_surgered"] = forgetting
    rep["forgetting_dropped_over_cap"] = dropped
    (out / "forgetting_locus.json").write_text(json.dumps(rep, indent=2))
    print(f"[forget][PRECHECK] {len(all_forgetting)}/{len(repos)} forget (precheck ppl_ft − base >= "
          f"{forget_min}); surgering top {len(forgetting)} (cap {a.max_surgery}); dropped {len(dropped)}. "
          f"top: {[(r.split('ft-')[-1], round(ppl_ft[r]-base_ppl,2)) for r in all_forgetting[:8]]}", flush=True)

    if not forgetting:
        rep["verdict"] = (f"H_no-signal: no fine-tune forgets general LM ppl by >= {forget_min} "
                          f"(max {max(ppl_ft[r]-base_ppl for r in repos):.3f}). VL fine-tuning does not "
                          f"degrade general blank-image perplexity; the q/k surgery has nothing to "
                          f"recover -> pivot to a deliberate-deletion retraining setup.")
        (out / "forgetting_locus.json").write_text(json.dumps(rep, indent=2))
        print(f"\nFORGET_LOCUS_OK verdict=NO_SIGNAL", flush=True)
        return

    # ---- build U_unt / bottom-σ / top-σ per (L,k).
    U_unt, Q_bots, Q_tops = {}, {}, {}
    for L in want_layers:
        for k in k_grid:
            Uk, _ = _bottom_eigvecs(G[L], k)
            assert abs((Uk.T @ Uk) - np.eye(k)).max() < 1e-8, f"U_unt not orthonormal L{L} k{k}"
            U_unt[(L, k)] = Uk
        W0_torch = base_W0q_torch[L].float().cpu()
        for k in k_grid:
            Q_bots[(L, k)] = _base_bottom_sigma(W0_torch, k)
            Q_tops[(L, k)] = _base_top_sigma(W0_torch, k)

    # ---- SURGERY (Pass 2): per forgetting ft reload decoder (precise W_ft) + full model (forward).
    def _ppl_now(model):
        return _ppl_blankimg(model, processor, device, n_seq=a.n_ppl_seq, window=a.ppl_window)["ppl"]

    surg = {}
    for ridx, repo in enumerate(forgetting):
        sha = shas[repo]
        print(f"[forget][surg] {repo}@{sha[:8]} ({ridx+1}/{len(forgetting)}) ...", flush=True)
        w_ft = _load_decoder(repo, sha)
        Wftq = {L: w_ft[f"layers.{L}.self_attn.q_proj.weight"].to(torch.float64).cpu().numpy()
                for L in want_layers}
        Wftk = {L: w_ft[f"layers.{L}.self_attn.k_proj.weight"].to(torch.float64).cpu().numpy()
                for L in want_layers}
        del w_ft
        model = _load_full(repo, sha, device, bf16)
        layers = _decoder_layers(model)
        snap = _snapshot_qk(layers, want_layers)
        ft_ppl = _ppl_now(model)            # full n_ppl_seq baseline (unpatched), the recovery contrast anchor
        _set_qk_to_base(layers, W0q, W0k, want_layers, device)
        ppl_fullqk = _ppl_now(model)
        _restore_qk(layers, snap, want_layers)
        rec_full = ft_ppl - ppl_fullqk
        per_k = {}
        for k in k_grid:
            arms = {}
            rem = _surgery(layers, Wftq, Wftk, W0q, W0k,
                           {L: U_unt[(L, k)] for L in want_layers}, want_layers, device)
            arms["U_unt"] = {"ppl": _ppl_now(model), "removal_frac": rem}
            _restore_qk(layers, snap, want_layers)
            for name, QB in (("bottom_sigma", Q_bots), ("top_sigma", Q_tops)):
                rem = _surgery(layers, Wftq, Wftk, W0q, W0k,
                               {L: QB[(L, k)] for L in want_layers}, want_layers, device)
                arms[name] = {"ppl": _ppl_now(model), "removal_frac": rem}
                _restore_qk(layers, snap, want_layers)
            rand = []
            for d in range(N_RAND_DRAWS):
                QR = {L: _random_subspace(n, k, a.seed * 104729 + 1000 * d + L) for L in want_layers}
                rem = _surgery(layers, Wftq, Wftk, W0q, W0k, QR, want_layers, device)
                rand.append({"ppl": _ppl_now(model), "removal_frac": rem})
                _restore_qk(layers, snap, want_layers)
            arms["random_band"] = rand
            per_k[str(k)] = arms
            print(f"[forget][surg]   k={k} ft_ppl={ft_ppl:.2f} fullqk_rec={rec_full:+.2f} "
                  f"U_unt rec={ft_ppl-arms['U_unt']['ppl']:+.2f}(rem {arms['U_unt']['removal_frac']:.3f}) "
                  f"rand rec={ft_ppl-np.mean([r['ppl'] for r in rand]):+.2f}", flush=True)
        surg[repo] = {"ppl_ft": ft_ppl, "ppl_fullqk": ppl_fullqk, "rec_full_qk": rec_full, "k": per_k}
        rep["surgery"] = surg
        (out / "forgetting_locus.json").write_text(json.dumps(rep, indent=2))
        del model, layers, snap, Wftq, Wftk
        _delete_cache(repo)
        torch.cuda.empty_cache()

    # ---- VERDICT: across ft with valid q/k-recoverable forgetting, is U_unt the efficient locus?
    def _eff(rec, rem):
        return rec / rem if rem > 1e-6 else float("nan")

    valid = [r for r in forgetting if surg[r]["rec_full_qk"] >= forget_min]
    rep["valid_forgetting_ft"] = valid
    verdict_k = {}
    for k in k_grid:
        ks = str(k)
        rows = []
        for r in valid:
            arms = surg[r]["k"][ks]
            ft_ppl = surg[r]["ppl_ft"]
            rec_full = surg[r]["rec_full_qk"]
            rec_unt = ft_ppl - arms["U_unt"]["ppl"]
            rec_bots = ft_ppl - arms["bottom_sigma"]["ppl"]
            rand_recs = [ft_ppl - x["ppl"] for x in arms["random_band"]]
            rec_rand = float(np.mean(rand_recs))
            rem_rand = float(np.mean([x["removal_frac"] for x in arms["random_band"]]))
            rows.append({"ft": r, "rec_full": rec_full,
                         "rec_unt": rec_unt, "rem_unt": arms["U_unt"]["removal_frac"],
                         "eff_unt": _eff(rec_unt, arms["U_unt"]["removal_frac"]),
                         "frac_unt": rec_unt / rec_full if rec_full > 0 else float("nan"),
                         "rec_rand": rec_rand, "rem_rand": rem_rand, "eff_rand": _eff(rec_rand, rem_rand),
                         "frac_rand": rec_rand / rec_full if rec_full > 0 else float("nan"),
                         "rec_bots": rec_bots, "rem_bots": arms["bottom_sigma"]["removal_frac"],
                         "eff_bots": _eff(rec_bots, arms["bottom_sigma"]["removal_frac"]),
                         "frac_bots": rec_bots / rec_full if rec_full > 0 else float("nan")})

        def _boot(ku, kn):
            d = np.array([row[ku] - row[kn] for row in rows
                          if np.isfinite(row[ku]) and np.isfinite(row[kn])], dtype=np.float64)
            if d.size == 0:
                return {"n": 0, "median": float("nan"), "lo": float("nan"), "hi": float("nan")}
            rng = np.random.default_rng(a.seed)
            meds = [np.median(d[rng.integers(0, d.size, d.size)]) for _ in range(1000)]
            return {"n": int(d.size), "median": float(np.median(d)),
                    "lo": float(np.percentile(meds, 2.5)), "hi": float(np.percentile(meds, 97.5))}

        verdict_k[ks] = {"rows": rows, "eff_contrast_vs_rand": _boot("eff_unt", "eff_rand"),
                         "eff_contrast_vs_bots": _boot("eff_unt", "eff_bots"),
                         "median_frac_unt": float(np.nanmedian([r["frac_unt"] for r in rows])) if rows else float("nan"),
                         "median_frac_rand": float(np.nanmedian([r["frac_rand"] for r in rows])) if rows else float("nan"),
                         "median_frac_bots": float(np.nanmedian([r["frac_bots"] for r in rows])) if rows else float("nan")}
    rep["verdict_by_k"] = verdict_k

    kk = str(k_grid[-1])
    v = verdict_k[kk]
    core = (v["eff_contrast_vs_rand"]["lo"] > 0 and v["eff_contrast_vs_bots"]["lo"] > 0
            and v["median_frac_unt"] >= 1.5 * max(v["median_frac_rand"], 1e-9)
            and v["median_frac_unt"] >= 1.5 * max(v["median_frac_bots"], 1e-9))
    rep["verdict_basis"] = {"k": int(kk), "n_valid": len(valid), **v["eff_contrast_vs_rand"],
                            "median_frac_unt": v["median_frac_unt"]}
    if core:
        rep["verdict"] = (f"CORE-LOCUS: across {len(valid)} forgetting fine-tunes, removing ΔW's U_unt "
                          f"reading recovers base LM competence MORE per unit ‖ΔW‖ removed than matched-"
                          f"random AND bottom-σ (eff contrast vs rand median {v['eff_contrast_vs_rand']['median']:.2f} "
                          f"CI[{v['eff_contrast_vs_rand']['lo']:.2f},{v['eff_contrast_vs_rand']['hi']:.2f}]); "
                          f"U_unt captures median {v['median_frac_unt']:.2f} of q/k-recoverable forgetting vs "
                          f"{v['median_frac_rand']:.2f}/{v['median_frac_bots']:.2f}. The invariant core IS the "
                          f"forgetting locus -> invariant-core-protected fine-tuning is motivated.")
    else:
        rep["verdict"] = (f"COMPLEMENT-LOCUS: across {len(valid)} forgetting fine-tunes, removing ΔW's U_unt "
                          f"reading does NOT recover base competence more efficiently than matched-random / "
                          f"bottom-σ (eff contrast vs rand CI[{v['eff_contrast_vs_rand']['lo']:.2f},"
                          f"{v['eff_contrast_vs_rand']['hi']:.2f}]; U_unt frac {v['median_frac_unt']:.2f} vs "
                          f"{v['median_frac_rand']:.2f}/{v['median_frac_bots']:.2f}). The invariant core is "
                          f"load-bearing but NOT where general-competence forgetting lives.")
    (out / "forgetting_locus.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78, flush=True)
    print(rep["verdict"], flush=True)
    print(f"FORGET_LOCUS_OK verdict={'CORE' if core else 'COMPLEMENT'} n_valid={len(valid)} k={k_grid}",
          flush=True)


if __name__ == "__main__":
    main()
