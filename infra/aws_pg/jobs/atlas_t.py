"""ATLAS Arm T (GPU) — the theory block's live tests. T-1 is the MDE: one person, one job, one
contrast (radial vs coherence-matched tangential at matched per-token energy) on which H_gauge and
H_energy predict opposite outcomes.

  T-1  radial ± vs tangential (coherence-matched twin, AM-6) at ε ∈ {0.1, 0.2, 0.4}·rms; iid
       tangential + Haar-iid support rows; 3 seeds where stochastic. Adjudicated on log(1+Δppl).
  T-2  coherent PRE-RoPE K-plane injection per rotary frequency plane (8 planes, matched energy)
       + the Haar-plane-pair control (physically breaks the frequency ladder at identical
       marginals — the §5 "frequency-shuffled" control, implemented as real cells that CAN fail).
  T-3  coherent injection by channel: K post-RoPE (the exact-gauge THEOREM CHECK, injected as the
       counter-rotated pre-RoPE vector), V, Q (counter-rotated, energy split across heads), + the
       Haar 256-dim channel null.
  T-4  leg A: P3-style shared vs per-token rank-128 removal at L14 — n_eff(L14) measured and the
       P4 prediction UPLOADED before any L14 cell runs (§9.3). leg B (AM-2): per-window Spearman
       between n_eff and the shared-minus-iid damage gap at L0, permutation p.

Refuses to run unless Gate-0 is green AND its RoPE convention is certified.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_t --type p5.4xlarge,g6e.8xlarge --no-weights \
      --s3-prefix results/atlas_t --budget-min 600
"""
import argparse
import json
import math
import os
import pathlib

T1_EPS = (0.1, 0.2, 0.4)
T2_PLANES = (0, 2, 5, 10, 20, 40, 80, 127)
T2_KAPPA = 0.2                 # channel amplitude, × channel rms × √dim
N_SEEDS = 3
P3_RANK = 128
N_POOL = 512
N_ITEMS = 200


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
    from infra.pg import atlas_core as AC
    from infra.pg import atlas_fit as AF
    from infra.pg import bispec_core as BC
    from infra.pg import inherit_pg1 as IP
    from infra.pg.ctrliface_core import random_basis

    AS.assert_frozen("t")
    RS.install_pg()
    g0 = json.loads(RS.fetch(f"{AS.ATLAS_GATE0}/gate0.json", "/tmp/atlas/gate0.json").read_text())
    assert g0.get("green") and g0["rope_convention"]["certified"], \
        "Gate-0 not green / RoPE not certified — Arm T refuses to run"
    L = int(a.site[1:])
    rope_theta = float(g0["model"]["rope_theta"])
    head_dim, n_heads = g0["model"]["head_dim"], g0["model"]["n_heads"]
    rep = {"object": "atlas_t_v1", "site": a.site, "gate0_ppl": g0["ppl_gate"]["base_ppl"],
           "cells": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    def _upload_now(path):
        os.system(f"aws s3 cp {path} {AS.S3_BUCKET}/results/atlas_t/out/{path.name} --region us-east-1")

    # ---- model, refs -----------------------------------------------------------
    bank = AS.load_sigma_bank(a.site)
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
    base = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)
    base_ppl = base["ppl"]
    assert abs(base_ppl - AS.FUNC_BASE_PPL) < 0.01, f"ppl anchor drift {base_ppl}"
    floor = IP.eval_pg1(model, proc, vqa, items, "cuda", scramble_seed=7)["score"]
    rep["reference"] = {"intact_vqa": intact["score"], "base_ppl": base_ppl,
                        "dead_floor_vqa": floor, "n_items": len(items), "salt": "atlas|"}
    _w()
    run = AS.CellRunner(model, proc, vqa, items, intact, floor, base_ppl, rep, _w)

    meta = json.loads(RS.fetch(f"{AS.REGRAIN_HARVEST}/harvest_meta.json",
                               "/tmp/atlas/harvest_meta.json").read_text())
    shard0 = meta["shards"][a.site][0]
    X = torch.load(RS.fetch(f"{AS.REGRAIN_HARVEST}/{shard0}", f"/tmp/atlas/shards/{shard0}"),
                   map_location="cpu", weights_only=False)["X"][:8192].float().cuda()
    rms = float(X.pow(2).mean().sqrt())
    d = X.shape[1]
    rep["site_rms"] = rms

    def hk(hook):
        return lambda: BC.hook_at(layers, L, hook)

    # =========================================================================
    # T-1 — the MDE.
    # =========================================================================
    for eps in T1_EPS:
        ea = eps * rms * math.sqrt(d)
        for sign, nm in ((+1.0, "out"), (-1.0, "in")):
            run.run(f"T1_radial_{nm}_e{eps}", hk(AC.make_hook_radial(ea, sign=sign)),
                    family="T1", arm="radial", eps=eps, sign_dir=nm)
        for s in range(N_SEEDS):
            run.run(f"T1_tang_coh_e{eps}_s{s}", hk(AC.make_hook_tangential(ea, seed=100 + s, d=d)),
                    family="T1", arm="tangential_coherent", draw=s, eps=eps)
    for s in range(N_SEEDS):
        ea = 0.2 * rms * math.sqrt(d)
        run.run(f"T1_tang_iid_e0.2_s{s}",
                hk(AC.make_hook_tangential(ea, seed=200 + s, shared=False, d=d)),
                family="T1", arm="tangential_iid", draw=s, eps=0.2)
        run.run(f"T1_haar_iid_e0.2_s{s}",
                hk(AC.make_hook_tangential(ea, seed=300 + s, shared=False, orthogonalize=False, d=d)),
                family="T1", arm="haar_iid", draw=s, eps=0.2)

    def t1_stats(readout):
        ratio, tang, sat = {}, {}, {}
        for eps in T1_EPS:
            rads = [c for c in rep["cells"] if c["family"] == "T1" and c["arm"] == "radial"
                    and c["eps"] == eps]
            tcs = [c for c in rep["cells"] if c["family"] == "T1"
                   and c["arm"] == "tangential_coherent" and c["eps"] == eps]
            rm = float(np.mean([c[readout] for c in rads]))
            tm = float(np.mean([c[readout] for c in tcs]))
            ratio[eps] = rm / max(tm, 1e-9)
            tang[eps] = tm
            sat[eps] = any(c.get("saturated_dead") for c in rads + tcs)
        noise = float(np.std([c[readout] for c in rep["cells"]
                              if c["family"] == "T1" and c["arm"] == "tangential_coherent"
                              and c["eps"] == 0.2]))
        return ratio, tang, sat, noise

    ratio_p, tang_p, sat_p, noise_p = t1_stats("log1p_dppl")
    rep["rt1"] = {"ppl": {"ratio_by_eps": ratio_p, "tang_by_eps": tang_p, "seed_noise": noise_p,
                          "adjudication": AC.adjudicate_rt1(ratio_by_eps=ratio_p, tang_by_eps=tang_p,
                                                            seed_noise=noise_p, saturated_by_eps=sat_p)}}
    ratio_v, tang_v, sat_v, noise_v = t1_stats("effect")
    rep["rt1"]["vqa"] = {"ratio_by_eps": ratio_v, "tang_by_eps": tang_v, "seed_noise": noise_v,
                         "adjudication": AC.adjudicate_rt1(ratio_by_eps=ratio_v, tang_by_eps=tang_v,
                                                           seed_noise=noise_v, saturated_by_eps=sat_v)}
    print(f"[t] R-T1 ppl: {rep['rt1']['ppl']['adjudication']['verdict']} "
          f"ratios={ {k: round(v, 3) for k, v in ratio_p.items()} }", flush=True)
    _w()

    # =========================================================================
    # T-2 — the RoPE frequency ladder (pre-RoPE K-plane injection).
    # =========================================================================
    kv_rms = float(g0["channel_rms_at_site"]["k"])
    eps_k = T2_KAPPA * kv_rms * math.sqrt(head_dim)
    om = AC.rope_freqs(head_dim, base=rope_theta)
    for p_i in T2_PLANES:
        for s in range(2):
            phase = float(torch.rand(1, generator=torch.Generator().manual_seed(500 + 10 * p_i + s)) * 2 * math.pi)
            v = eps_k * AC.plane_vector(head_dim, p_i, phase=phase)
            run.run(f"T2_plane{p_i}_s{s}",
                    lambda vv=v: AC.proj_hook_at(layers[L], "k_proj",
                                                 AC.make_proj_out_hook_add(AC.delta_const(vv), expect_dim=head_dim)),
                    family="T2", arm="plane", draw=s, plane=p_i, omega=float(om[p_i]),
                    retention_T=AC.dirichlet_retention(float(om[p_i]), 271))
    for j in range(8):                                        # Haar plane-pair control cells
        gen = torch.Generator().manual_seed(600 + j)
        A = torch.randn(head_dim, 2, generator=gen, dtype=torch.float64)
        Q, _ = torch.linalg.qr(A)
        phase = float(torch.rand(1, generator=gen) * 2 * math.pi)
        v = eps_k * (Q[:, 0] * math.cos(phase) + Q[:, 1] * math.sin(phase))
        run.run(f"T2_haarpair{j}",
                lambda vv=v: AC.proj_hook_at(layers[L], "k_proj",
                                             AC.make_proj_out_hook_add(AC.delta_const(vv), expect_dim=head_dim)),
                family="T2", arm="haar_pair", draw=0, plane=T2_PLANES[j], shuffled_label=True)

    def rt2_stats(readout):
        # ρ_freq is Spearman against the ACTUAL angular rate ω (plane index 0 is the FASTEST plane
        # in the HF convention — ω_i = θ^(−2i/dim) — so index order INVERTS frequency order).
        # T2's derivation: the retained (row-mean) component of a pre-RoPE coherent shift is the
        # gauge component and does nothing; damage rides the non-retained energy ε²(1−retention²),
        # which GROWS with ω. H_ladder ⇔ damage increasing in ω.
        pl = [(c["omega"], c[readout]) for c in rep["cells"]
              if c["family"] == "T2" and c["arm"] == "plane"]
        hp = [(c["plane"], c[readout]) for c in rep["cells"]
              if c["family"] == "T2" and c["arm"] == "haar_pair"]
        rho = AF.spearman([w for w, _ in pl], [v for _, v in pl])
        rho_sh = AF.spearman([p for p, _ in hp], [v for _, v in hp])
        om_hi, om_lo = max(w for w, _ in pl), min(w for w, _ in pl)
        hi = [v for w, v in pl if w == om_hi]
        lo = [v for w, v in pl if w == om_lo]
        gap = float(np.mean(hi) - np.mean(lo))
        se = float(np.hypot(np.std(hi) / max(len(hi), 1) ** 0.5, np.std(lo) / max(len(lo), 1) ** 0.5))
        void = any(c.get("saturated_dead") for c in rep["cells"] if c["family"] == "T2")
        return dict(rho_freq=rho, gap=gap, gap_cilo=gap - 1.96 * se, gap_cihi=gap + 1.96 * se,
                    rho_shuffle=rho_sh, any_void=void)

    st2 = rt2_stats("log1p_dppl")
    rep["rt2"] = {"ppl": {**st2, "adjudication": AC.adjudicate_rt2(**st2)}}
    _w()
    print(f"[t] R-T2 ppl: {rep['rt2']['ppl']['adjudication']['verdict']} rho={st2['rho_freq']:.3f}",
          flush=True)

    # =========================================================================
    # T-3 — the channel triple + the theorem check.
    # =========================================================================
    for s in range(N_SEEDS):
        gen = torch.Generator().manual_seed(700 + s)
        u = torch.randn(head_dim, generator=gen, dtype=torch.float64)
        u = eps_k * u / u.norm()
        run.run(f"T3_K_postrope_s{s}",
                lambda vv=u: AC.proj_hook_at(layers[L], "k_proj",
                                             AC.make_proj_out_hook_add(
                                                 AC.delta_postrope_coherent(vv, base=rope_theta),
                                                 expect_dim=head_dim)),
                family="T3", arm="K_postrope", draw=s)
        run.run(f"T3_V_s{s}",
                lambda vv=u: AC.proj_hook_at(layers[L], "v_proj",
                                             AC.make_proj_out_hook_add(AC.delta_const(vv), expect_dim=head_dim)),
                family="T3", arm="V", draw=s)
        run.run(f"T3_Q_postrope_s{s}",
                lambda vv=u: AC.proj_hook_at(layers[L], "q_proj",
                                             AC.make_proj_out_hook_add(
                                                 AC.delta_multihead(AC.delta_postrope_coherent(vv, base=rope_theta),
                                                                    n_heads, head_dim),
                                                 expect_dim=n_heads * head_dim)),
                family="T3", arm="Q", draw=s)
        gen2 = torch.Generator().manual_seed(800 + s)
        h = torch.randn(head_dim, generator=gen2, dtype=torch.float64)
        h = eps_k * h / h.norm()
        run.run(f"T3_haar_chan_s{s}",
                lambda vv=h: AC.proj_hook_at(layers[L], "k_proj",
                                             AC.make_proj_out_hook_add(AC.delta_const(vv), expect_dim=head_dim)),
                family="T3", arm="haar_channel_preRoPE", draw=s)

    def t3m(arm, readout="log1p_dppl"):
        return float(np.mean([c[readout] for c in rep["cells"]
                              if c["family"] == "T3" and c["arm"] == arm]))

    seed_noise_t3 = float(np.std([c["log1p_dppl"] for c in rep["cells"]
                                  if c["family"] == "T3" and c["arm"] == "V"]))
    rep["rt3"] = AC.adjudicate_rt3(k_damage=t3m("K_postrope"), v_damage=t3m("V"),
                                   q_damage=t3m("Q"), seed_noise=max(seed_noise_t3, 1e-4),
                                   haar_damage=t3m("haar_channel_preRoPE"))
    assert rep["rt3"]["theorem_check_pass"], \
        "K-coherent post-RoPE moved the readout > 3× seed-noise — INSTRUMENT BUG (T2 is a theorem); halt"
    _w()
    print(f"[t] R-T3: ordering={rep['rt3']['ordering']} theorem={rep['rt3']['theorem_check_pass']}",
          flush=True)

    # =========================================================================
    # T-4 — n_eff(L14) measured + P4/P5 predictions UPLOADED, then the cells.
    # =========================================================================
    batch = items[:8]
    images = [vqa.image_fn(r).convert("RGB") for r in batch]
    prompts = [vqa.prompt_fn(r) for r in batch]
    inp = proc(text=prompts, images=images, return_tensors="pt", padding="longest").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    with torch.inference_mode():
        o = model(**inp, output_attentions=True)
    v14 = torch.cat([BC.n_eff_from_attention(at.float().cpu())
                     for li, at in enumerate(o.attentions) if li >= 14])
    n_eff_l14 = float(v14.median())
    del o
    torch.cuda.empty_cache()
    ks = json.loads(RS.fetch(AS.BISPEC_K_SEALED, "/tmp/atlas/ks.json").read_text())
    co = [c["dppl"] for c in ks["cells"] if c["family"] == "P3" and c["arm"] == "coherent"]
    io_ = [c["dppl"] for c in ks["cells"] if c["family"] == "P3" and c["arm"] == "independent"]
    pred_const = float(np.mean(co) / np.mean(io_))
    pred_law = math.sqrt(n_eff_l14)
    pred = {"object": "atlas_t_pre_unblind", "rule": "R-T4",
            "n_eff_L14_pooled_median": n_eff_l14,
            "prediction_law_sqrt_neff_L14": pred_law,
            "prediction_constant_L0_ratio": pred_const,
            "grid_rows": {"P4": f"L14 ratio within ±0.4 of {pred_law:.3f}, excludes {pred_const:.3f}",
                          "P5": "per-window Spearman(n_eff, shared−iid gap) > 0, perm p ≤ 0.05 (AM-2)"},
            "note": "written and uploaded BEFORE any L14 coherence cell runs (§9.3)"}
    pp = out / "pre_unblind_predictions.json"
    pp.write_text(json.dumps(pred, indent=2))
    _upload_now(pp)
    rep["rt4_predictions"] = pred
    _w()
    print(f"[t] T-4 predictions journaled: law={pred_law:.3f} const={pred_const:.3f}", flush=True)

    # leg A — L14 shared vs per-token subspace removal (rank 128); per-window nll banked so the
    # ratio CI is a PAIRED window bootstrap (n=120 windows × 3 seeds), not a 3-point resample
    bank14 = AS.load_sigma_bank("L14")
    hb14 = bank14["hbar"]
    legA_pw = {"shared": [], "pertoken": []}
    for s in range(N_SEEDS):
        Qsh = random_basis(d, P3_RANK, 777 + s, "cpu", torch.float64)
        with BC.hook_at(layers, 14, BC.make_hook_subspace_removal(Qsh, hb14, False, 0, d, P3_RANK)):
            r_sh = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)
        legA_pw["shared"].append(r_sh["per_window"])
        rep["cells"].append({"cell": f"T4_L14_shared_s{s}", "family": "T4_L14", "arm": "shared",
                             "draw": s, "site": "L14", "ppl": r_sh["ppl"],
                             "dppl": r_sh["ppl"] - base_ppl,
                             "log1p_dppl": math.log1p(max(r_sh["ppl"] - base_ppl, -0.99))})
        pool = BC.pertoken_subspace_pool(d, P3_RANK, N_POOL, 9 + s, device="cuda")
        with BC.hook_at(layers, 14, BC.make_hook_subspace_removal(None, hb14, True, 0, d, P3_RANK,
                                                                  pool=pool)):
            r_pt = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)
        legA_pw["pertoken"].append(r_pt["per_window"])
        rep["cells"].append({"cell": f"T4_L14_pertoken_s{s}", "family": "T4_L14", "arm": "pertoken",
                             "draw": s, "site": "L14", "ppl": r_pt["ppl"],
                             "dppl": r_pt["ppl"] - base_ppl,
                             "log1p_dppl": math.log1p(max(r_pt["ppl"] - base_ppl, -0.99))})
        del pool
        torch.cuda.empty_cache()
        _w()
        print(f"[t] T4 L14 s{s}: shared dppl={r_sh['ppl']-base_ppl:+.3f} "
              f"pertoken dppl={r_pt['ppl']-base_ppl:+.3f}", flush=True)
    bw = np.asarray(base["per_window"], float)
    shw = np.mean([np.asarray(p, float)[:, 0] for p in legA_pw["shared"]], axis=0)
    ptw = np.mean([np.asarray(p, float)[:, 0] for p in legA_pw["pertoken"]], axis=0)
    ntokw = bw[:, 1]
    rng = np.random.default_rng(1337)
    boots = []
    nw = len(ntokw)
    for _ in range(10000):
        idx = rng.integers(0, nw, nw)
        dsh = math.log1p(max(math.exp((shw[idx].sum()) / ntokw[idx].sum())
                             - math.exp(bw[idx, 0].sum() / ntokw[idx].sum()), -0.99))
        dpt = math.log1p(max(math.exp((ptw[idx].sum()) / ntokw[idx].sum())
                             - math.exp(bw[idx, 0].sum() / ntokw[idx].sum()), -0.99))
        if dpt > 1e-9:
            boots.append(dsh / dpt)
    sh = np.array([c["log1p_dppl"] for c in rep["cells"]
                   if c["family"] == "T4_L14" and c["arm"] == "shared"])
    pt = np.array([c["log1p_dppl"] for c in rep["cells"]
                   if c["family"] == "T4_L14" and c["arm"] == "pertoken"])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    n_eff_l0_banked = float(g0["n_eff_banked"]["pooled_median_L0"])
    rep["rt4_legA"] = {"ratio": float(sh.mean() / pt.mean()), "ci": ci,
                       "n_eff_L14": n_eff_l14, "pred_law": pred_law, "pred_const": pred_const,
                       "degenerate_gap_check": abs(pred_law - math.sqrt(n_eff_l0_banked)),
                       "adjudication": AC.adjudicate_rt4_legA(ratio_ci=ci, pred_law=pred_law,
                                                              pred_const=pred_const)}
    _w()
    print(f"[t] R-T4 legA: {rep['rt4_legA']['adjudication']['verdict']} "
          f"ratio={rep['rt4_legA']['ratio']:.3f} ci={ci}", flush=True)

    # leg B — per-window n_eff at L0 vs per-window shared−iid gap (AM-2)
    win_neff = []
    caps = {}

    def _cap_attn(li):
        def hook(_m, _a, outp):
            if outp[1] is not None:
                caps.setdefault(li, []).append(
                    float(BC.n_eff_from_attention(outp[1].float().cpu()).median()))
        return hook

    hs_ = [layers[li].self_attn.register_forward_hook(_cap_attn(li)) for li in range(18)]
    # eager attention returns weights only when output_attentions reaches the layer call; set it
    # on every config object that might be consulted, and FAIL LOUD if capture still comes back
    # empty (leg B may carry the R-T4 family alone — it must not be lost silently)
    model.config.output_attentions = True
    model.config.text_config.output_attentions = True
    if hasattr(model, "language_model"):
        model.language_model.config.output_attentions = True
    with torch.inference_mode():
        base_pw = IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)
    for h in hs_:
        h.remove()
    model.config.output_attentions = False
    model.config.text_config.output_attentions = False
    if hasattr(model, "language_model"):
        model.language_model.config.output_attentions = False
    n_win = base_pw["n_windows"]
    assert caps and all(len(v) == n_win for v in caps.values()), \
        f"attention capture failed for leg B (got layers {sorted(caps)}) — halt"
    per_win_neff = [float(np.median([caps[li][wi] for li in caps])) for wi in range(n_win)]
    rep["rt4_legB_capture"] = {"n_windows": n_win, "attn_capture_ok": True,
                               "layers_captured": sorted(caps)}
    _w()

    if per_win_neff is not None:
        pw_gap = None
        sh_pw, pt_pw = [], []
        for s in range(N_SEEDS):
            Qsh = random_basis(d, P3_RANK, 777 + s, "cpu", torch.float64)
            with BC.hook_at(layers, L, BC.make_hook_subspace_removal(Qsh, bank["hbar"], False, 0,
                                                                     d, P3_RANK)):
                sh_pw.append(IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["per_window"])
            pool = BC.pertoken_subspace_pool(d, P3_RANK, N_POOL, 9 + s, device="cuda")
            with BC.hook_at(layers, L, BC.make_hook_subspace_removal(None, bank["hbar"], True, 0,
                                                                     d, P3_RANK, pool=pool)):
                pt_pw.append(IP.ppl_gate(model, proc, "cuda", n_seq=RS.N_PPL_SEQ)["per_window"])
            del pool
            torch.cuda.empty_cache()
        b = np.asarray(base_pw["per_window"], float)
        shm = np.mean([np.asarray(p, float)[:, 0] for p in sh_pw], axis=0)
        ptm = np.mean([np.asarray(p, float)[:, 0] for p in pt_pw], axis=0)
        gap_w = (shm - ptm) / b[:, 1]                        # per-window per-token nll gap
        rho, p_perm = AF.perm_p_spearman(np.asarray(per_win_neff), gap_w)
        rep["rt4_legB"] = {"rho": rho, "perm_p": p_perm, "n_windows": int(len(gap_w)),
                           "verdict": "H_law_legB" if (rho > 0 and p_perm <= 0.05) else
                                      "H_no-within-site-ordering",
                           "note": "AM-2: per-window statistic replaces the unadjudicable "
                                   "3-tercile ordering; terciles remain descriptive (F-T4)"}
        terc = np.argsort(np.argsort(per_win_neff)) * 3 // len(per_win_neff)
        rep["rt4_legB"]["tercile_gaps"] = [float(np.mean(gap_w[terc == t])) for t in range(3)]
        rep["rt4_legB"]["tercile_neff"] = [float(np.mean(np.asarray(per_win_neff)[terc == t]))
                                           for t in range(3)]
        print(f"[t] R-T4 legB: rho={rho:.3f} p={p_perm:.4f}", flush=True)
    _w()

    print("ATLAS_T_OK", json.dumps({"rt1": rep["rt1"]["ppl"]["adjudication"]["verdict"],
                                    "rt2": rep["rt2"]["ppl"]["adjudication"]["verdict"],
                                    "rt3": rep["rt3"]["ordering"],
                                    "rt4A": rep["rt4_legA"]["adjudication"]["verdict"]}, default=str))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
