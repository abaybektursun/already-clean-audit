"""ATLAS Arm W (GPU) — T6, the visible-variance write law, tested against the banked Ω profiles.

T6: ΔW accumulates Σ_t g_t y_tᵀ, so mode-i write intensity ∝ λ_i ρ_i with ρ_i the squared
error–input-coefficient correlation. The top data mode is ≈ DC across tokens, so its error
coefficient is the batch-mean error signal — near zero for a calibrated model — predicting
ρ₁ ≈ 0 and exactly the "data-aligned in direction, starved on the top mode" shape Arm W measured.

Measurement: the frozen base's OWN gradient under two proxy losses (blank-image wikitext CE; the
VQA-answer CE on discovery items), per (layer × matrix) in the SAME banded input eigenbases
bispec_w profiled Ω in (recomputed here by the same capture, asserted against the banked ranks).
ρ_i = ‖∇_W u_i‖² / (Σ‖g_t‖² · Σc_it²) ∈ [0,1]. Scope disclosed in the estimand: ρ comes from
proxy losses on the frozen base, not the 24 FTs' own objectives — the fit is a transfer claim at
exactly that scope (spec §6 R-W).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_w --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_w --budget-min 360
"""
import argparse
import json
import os
import pathlib

ATTN_TYPES = ("q_proj", "k_proj", "v_proj")
MLP_TYPES = ("gate_proj", "up_proj")
N_WIN = 60
N_VQA = 200
N_SHUFFLE = 20


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0,4,9,14,17")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs.conserved_core import _wikitext_windows
    from infra.aws_pg.jobs.invariant_core_func import _blank_image
    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP

    AS.assert_frozen("w")
    RS.install_pg()
    g0 = json.loads(RS.fetch(f"{AS.ATLAS_GATE0}/gate0.json", "/tmp/atlas/gate0.json").read_text())
    assert g0.get("green"), "Gate-0 not green — Arm W refuses to run"
    want_L = [int(x) for x in a.layers.split(",")]
    w_art = json.loads(RS.fetch(AS.BISPEC_W, "/tmp/atlas/bispec_w.json").read_text())
    assert w_art["green"] and w_art["layers"] == want_L, "banked Ω battery layer set mismatch"
    rep = {"object": "atlas_w_v1", "layers": want_L,
           "estimand": "rho_i = |grad_W u_i|^2 / (sum|g_t|^2 * sum c_it^2) per mode of each "
                       "matrix's OWN input second moment; profiles = band-aggregated lambda*rho",
           "scope": "proxy losses on the frozen base — a transfer claim at exactly that scope"}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    sha, _ = RS.pinned_pg1_sha("/tmp/atlas")
    model, proc = IP.load_pg1(sha=sha)
    model = model.float()                                  # fp32: ρ needs cancellation fidelity
    layers = IP.decoder_layers(model)
    for p in model.parameters():
        p.requires_grad_(False)
    mods = {}
    for L in want_L:
        for t in ATTN_TYPES:
            mods[(L, t)] = getattr(layers[L].self_attn, t)
        for t in MLP_TYPES:
            mods[(L, t)] = getattr(layers[L].mlp, t)
    for m in mods.values():
        m.weight.requires_grad_(True)

    # ---- input Grams (one pass, both spaces), banded — the bispec_w convention -----
    windows = _wikitext_windows(proc, N_WIN, 512)
    blank = _blank_image(128)

    grams = {("attn", L): None for L in want_L}
    grams.update({("mlp", L): None for L in want_L})
    counts = {k: 0 for k in grams}
    handles = []

    def mk_gram(space, L):
        def pre_hook(_m, args):
            x = args[0].reshape(-1, args[0].shape[-1]).double()
            g = (x.transpose(0, 1) @ x).cpu()
            grams[(space, L)] = g if grams[(space, L)] is None else grams[(space, L)] + g
            counts[(space, L)] += x.shape[0]
        return pre_hook

    for L in want_L:
        handles.append(layers[L].self_attn.q_proj.register_forward_pre_hook(mk_gram("attn", L)))
        handles.append(layers[L].mlp.gate_proj.register_forward_pre_hook(mk_gram("mlp", L)))
    with torch.inference_mode():
        for wt, _wi in windows:
            inp = proc(text=[wt], images=[blank], return_tensors="pt", padding=False).to("cuda")
            model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                  pixel_values=inp["pixel_values"].to(torch.float32))
    for h in handles:
        h.remove()

    bands = {}
    rank_drift = {}
    for (space, L), G in grams.items():
        M = torch.as_tensor(G, dtype=torch.float64) / counts[(space, L)]
        bb = BC.eig_bands(M)
        banked = w_art["bands"][space][str(L)]["ranks"]
        got = [hi - lo for lo, hi in bb["band_slices"]]
        # this job runs the forward in FP32 (gradient fidelity); bispec_w captured in bf16, so
        # energy-quantile cuts drift a few indices (~2–5 %). Gross drift (>12 % of d per band)
        # would mean a convention break and still halts; the realized drift is disclosed.
        for b_i, (g_, b_) in enumerate(zip(got, banked)):
            assert abs(g_ - b_) <= max(0.12 * 2048 / 5, 0.35 * max(b_, 8)), \
                f"band ranks drift beyond precision at {space}/L{L} band{b_i}: {got} vs {banked}"
        rank_drift[f"{space}/L{L}"] = {"got": got, "banked": banked}
        bands[(space, L)] = {"bb": bb, "n": counts[(space, L)], "M": M}
    rep["band_rank_drift_fp32_vs_banked_bf16"] = rank_drift
    _w()

    # ---- the two proxy backward passes ------------------------------------------
    vqa, _c = IP.pg1_tasks()
    it = json.loads(RS.fetch(f"{RS.GRAIN_GATE0}/items_vqa.json", "/tmp/atlas/items_vqa.json").read_text())
    rows, hashes = IP.load_items(vqa, it["stream_window"])
    assert hashes == it["hashes"], "item-stream drift — halt"
    sp = AS.atlas_split(hashes)
    items = [rows[i] for i in sp["disc_idx"]][:N_VQA]

    def majority_answer(r):
        from collections import Counter
        ans = r.get("answers") or [r.get("answer", "")]
        vals = [x["answer"] if isinstance(x, dict) else str(x) for x in ans]
        return Counter(vals).most_common(1)[0][0]

    def run_proxy(name):
        for m in mods.values():
            m.weight.grad = None
        cap = AC.GradNormCapture({k: v for k, v in mods.items()})
        # per-proxy input Grams accumulated on the SAME tokens the gradient sees — Cauchy–Schwarz
        # (ρ ≤ 1) only holds with same-token accounting; normalizing the VQA proxy's gradient by
        # the wikitext Gram let ρ escape to 1e274 (caught by the accounting assert, run 2)
        gram_p = {("attn", L): None for L in want_L}
        gram_p.update({("mlp", L): None for L in want_L})

        def mk_gp(space, L):
            def pre_hook(_m, args):
                with torch.no_grad():                      # grad-enabled pass: keep the Gram
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
                    g = (x.transpose(0, 1) @ x).cpu()
                gram_p[(space, L)] = g if gram_p[(space, L)] is None else gram_p[(space, L)] + g
            return pre_hook

        gh = []
        for L in want_L:
            gh.append(layers[L].self_attn.q_proj.register_forward_pre_hook(mk_gp("attn", L)))
            gh.append(layers[L].mlp.gate_proj.register_forward_pre_hook(mk_gp("mlp", L)))
        n_batches = 0
        with cap:
            if name == "ppl_proxy":
                for wt, _wi in windows:
                    # empty prefix + the window as SUFFIX = LM loss over the window itself
                    # (text=wt AND suffix=wt would double the sequence and score a copy task)
                    inp = proc(text=[""], images=[blank], suffix=[wt], return_tensors="pt",
                               padding=False).to("cuda")
                    o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                              pixel_values=inp["pixel_values"].to(torch.float32),
                              labels=inp["labels"])
                    o.loss.backward()
                    n_batches += 1
            else:
                B = 8
                for s0 in range(0, len(items), B):
                    b = items[s0:s0 + B]
                    ims = [vqa.image_fn(r).convert("RGB") for r in b]
                    prompts = [vqa.prompt_fn(r) for r in b]
                    sufs = [majority_answer(r) for r in b]
                    inp = proc(text=prompts, images=ims, suffix=sufs, return_tensors="pt",
                               padding="longest").to("cuda")
                    o = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                              pixel_values=inp["pixel_values"].to(torch.float32),
                              labels=inp["labels"])
                    o.loss.backward()
                    n_batches += 1
        for h in gh:
            h.remove()
        res = {}
        for (L, t), m in mods.items():
            assert m.weight.grad is not None, f"no grad for L{L}/{t}"
            space = "attn" if t in ATTN_TYPES else "mlp"
            res[(L, t)] = {"nabla": m.weight.grad.double().cpu().clone(),
                           "sum_g2": cap.sum_g2[(L, t)], "n_pos": cap.n_pos[(L, t)],
                           "gram_sum": gram_p[(space, L)]}
            m.weight.grad = None
        print(f"[w] proxy {name}: {n_batches} backward batches", flush=True)
        return res

    proxies = {"ppl_proxy": run_proxy("ppl_proxy"), "vqa_proxy": run_proxy("vqa_proxy")}

    # ---- ρ per mode, λρ profiles, the R-W fit ------------------------------------
    rng = np.random.default_rng(4242)

    def coef_sq_of(res_lt, bb):
        """Σ_t (y_tᵀu_i)² on the PROXY'S OWN tokens: diag(Uᵀ (Σ y yᵀ) U) in the banded basis."""
        Mp = torch.as_tensor(res_lt["gram_sum"], dtype=torch.float64)
        U = bb["basis"]
        return (U * (Mp @ U)).sum(0).clamp_min(0.0)

    rw = {}
    rho_bank = {}
    for pname, res in proxies.items():
        pts_omega, pts_lamrho, pts_lam = [], [], []
        rho_bank[pname] = {}
        for L in want_L:
            for t in ATTN_TYPES + MLP_TYPES:
                space = "attn" if t in ATTN_TYPES else "mlp"
                bb = bands[(space, L)]["bb"]
                lam = bb["spectrum"].clamp_min(0)
                coef_sq = coef_sq_of(res[(L, t)], bb)
                rho = AC.rho_per_mode(res[(L, t)]["nabla"], bb["basis"], coef_sq,
                                      res[(L, t)]["sum_g2"])
                rho_bank[pname][f"L{L}/{t}"] = {"rho_top8": rho[:8].tolist(),
                                                "rho_median": float(rho.median())}
                lam_rho = (lam * rho)
                prof_lr = AC.band_aggregate(lam_rho, bb["band_slices"])
                om_prof = w_art["profiles"][f"L{L}/{t}"]["profile"]
                lam_prof = w_art["profiles"][f"L{L}/{t}"]["ntk_reference"]
                pts_omega.extend(om_prof)
                pts_lamrho.extend(prof_lr)
                pts_lam.extend(lam_prof)
        rho_s_lr = AF.spearman(pts_omega, pts_lamrho)
        rho_s_l = AF.spearman(pts_omega, pts_lam)
        dfit = rho_s_lr - rho_s_l
        # shuffled-ρ control: permute ρ across modes within each matrix, re-aggregate
        dfits_sh = []
        for si in range(N_SHUFFLE):
            pts_sh = []
            for L in want_L:
                for t in ATTN_TYPES + MLP_TYPES:
                    space = "attn" if t in ATTN_TYPES else "mlp"
                    bb = bands[(space, L)]["bb"]
                    lam = bb["spectrum"].clamp_min(0)
                    rho = AC.rho_per_mode(proxies[pname][(L, t)]["nabla"], bb["basis"],
                                          coef_sq_of(proxies[pname][(L, t)], bb),
                                          proxies[pname][(L, t)]["sum_g2"])
                    perm = torch.as_tensor(rng.permutation(len(rho)))
                    pts_sh.extend(AC.band_aggregate(lam * rho[perm], bb["band_slices"]))
            dfits_sh.append(AF.spearman(pts_omega, pts_sh) - rho_s_l)
        rw[pname] = {"rho_spearman_lamrho": rho_s_lr, "rho_spearman_lam": rho_s_l,
                     "dfit": dfit, "dfit_shuffle_mean": float(np.mean(dfits_sh)),
                     "adjudication": AF.adjudicate_rw(dfit=dfit, rho_lamrho=rho_s_lr,
                                                      dfit_shuffle=float(np.mean(dfits_sh)))}
        print(f"[w] {pname}: dfit={dfit:+.4f} rho_lamrho={rho_s_lr:.4f} "
              f"→ {rw[pname]['adjudication']['verdict']}", flush=True)
    rep["rw"] = rw
    rep["rho_per_matrix"] = rho_bank
    rep["rho1_prediction_check"] = {
        f"{p}/L{L}/{t}": rho_bank[p][f"L{L}/{t}"]["rho_top8"][0]
        for p in proxies for L in want_L for t in ("q_proj", "gate_proj")}
    _w()

    # bank per-mode ρ and λρ vectors (raw ∇W would be ~5 GB of fp64; the per-mode statistics are
    # the analysis object — every figure recomputes from these)
    bank_out = {}
    for p in proxies:
        for L in want_L:
            for t in ATTN_TYPES + MLP_TYPES:
                space = "attn" if t in ATTN_TYPES else "mlp"
                bb = bands[(space, L)]["bb"]
                lam = bb["spectrum"].clamp_min(0)
                rho = AC.rho_per_mode(proxies[p][(L, t)]["nabla"], bb["basis"],
                                      coef_sq_of(proxies[p][(L, t)], bb),
                                      proxies[p][(L, t)]["sum_g2"])
                bank_out[f"{p}/L{L}/{t}"] = {"rho": rho, "lam": lam,
                                             "sum_g2": proxies[p][(L, t)]["sum_g2"]}
    torch.save(bank_out, out / "rho_bank.pt")
    print("ATLAS_W_OK", json.dumps({p: rw[p]["adjudication"]["verdict"] for p in rw}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
