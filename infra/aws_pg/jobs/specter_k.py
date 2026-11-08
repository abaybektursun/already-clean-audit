"""SPECTER Arm K (GPU) — CORE⟂SPIKE: stiffness spectra of Σ_y⊗F̄_z per (layer, matrix-type),
stiffness-quantile overlaps of the named objects vs matched-rank Haar bands, and the K-FAC
fidelity gate (32 direct weight-space GGN products at 2 layers).

Amendments live here (journal E2): A1 — Σ_y re-harvested on the operator batch with the
bispec_w recipe (uncentered x.T@x/n fp64; q_proj input = attn space, gate_proj input = mlp
space); A2 — residual-space objects (U_unt_residual, U_wr, task footprint, v1) read on the F̄
side via the gated spectral completion + a 32-probe direct-quadform fidelity check; input-space
objects (U_unt_qk) read on the Σ_y side. R-K1 headline = U_unt_residual's F̄_dc(L0, ppl)
quantile.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_k --type p5.4xlarge,g6e.8xlarge \
      --no-weights --s3-prefix results/specter_k --budget-min 180
"""
import argparse
import json
import os
import pathlib

N_HAAR_BAND = 4000
N_DIRECT = 32
N_KFAC_PROBES = 32
KFAC_LAYERS = (0, 14)
MATRIX_TYPES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.aws_pg.jobs import _specter_shared as SS
    from infra.pg import atlas_fit as AF
    from infra.pg import inherit_pg1 as IP
    from infra.pg import specter_core as SC

    freeze = SS.assert_frozen("k")
    RS.install_pg()
    rep = {"object": "specter_k_v1", "freeze": freeze}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))
    _w()

    g0 = json.loads(SS.fetch(f"{SS.SPECTER_GATE0}/gate0.json", "/tmp/specter/g0.json").read_text())
    o_metrics = json.loads(SS.fetch(f"{SS.SPECTER_O}/metrics.json",
                                    "/tmp/specter/o_metrics.json").read_text())
    assert g0.get("green") and o_metrics.get("green"), "parents not green"
    path = g0["diff_path"]

    with tm.stage("load"):
        sha, _ = RS.pinned_pg1_sha("/tmp/specter")
        model, proc = IP.load_pg1(sha=sha)
        layers = IP.decoder_layers(model)
        bank = torch.load(SS.fetch(f"{SS.SPECTER_O}/operator_bank.pt",
                                   "/tmp/specter/operator_bank.pt"),
                          map_location="cpu", weights_only=False)
        # the L14 rows live in the second operator pass — required (KFAC_LAYERS = (0, 14))
        bank2 = torch.load(SS.fetch(f"{SS.S3_BUCKET}/results/specter_o_l8l14/out/operator_bank.pt",
                                    "/tmp/specter/operator_bank2.pt"),
                           map_location="cpu", weights_only=False)
        bank2.pop("_meta", None)
        bank.update(bank2)
        assert "L14/dc/ppl" in bank, "L14 operator rows missing — run the L8/L14 pass first"
        lib = torch.load(SS.fetch(f"{SS.SPECTER_O}/library_bank.pt",
                                  "/tmp/specter/library_bank.pt"),
                         map_location="cpu", weights_only=False)
    d = IP.D
    dev = "cuda"
    meta = lib["band_projectors_meta"]
    sb = [tuple(x) for x in meta["sb_slices"]]
    basis = lib["sigma_basis_L0"].double()

    def completion_env(key):
        e = bank[key]
        sp = e["spikes"][:8]
        if sp:
            U = e["evecs_A1"][:, [s["idx"] for s in sp]].double()
            lam = [s["eval_a1"] for s in sp]
        else:
            U, lam = None, []
        bulk = SC.band_bulk_from_traces(e["traces_sb"], sb, U, lam, basis)
        return U, lam, bulk

    U_dc, lam_dc, bulk_dc = completion_env("L0/dc/ppl")

    def qhat(v):
        return SC.spectral_completion_quadform(v, U_dc, lam_dc, bulk_dc, basis, sb)

    # ---- operator batch (ppl A) for direct quadforms + Σ_y harvest -----------------
    with tm.stage("items"):
        wins_a = SS.wikitext_windows(proc, SS.N_OP_PPL, skip=0)
        install, holder = SC.make_residual_injector(layers, 0)
        items = [SS.build_ppl_item(model, proc, dev, wd, install, holder) for wd in wins_a]
    eps = 0.05 * g0["identity"]["site_rms"]

    def direct_q(v):
        return SC.mode_quadform(items, v.float().to(dev) * eps, "dc", path=path) / (eps ** 2)

    # ---- A1: Σ_y re-harvest (bispec_w recipe) on the operator batch -----------------
    with tm.stage("sigma_y"):
        grams = {}
        hooks = []

        def _mk(key_):
            def hook(_m, args, _kw=None):
                x = args[0] if isinstance(args, tuple) else args
                xx = x.detach().reshape(-1, x.shape[-1]).double().cpu()
                grams[key_] = grams.get(key_, torch.zeros(xx.shape[1], xx.shape[1],
                                                          dtype=torch.float64)) + xx.T @ xx
                cnt[key_] = cnt.get(key_, 0) + xx.shape[0]
            return hook
        cnt = {}
        for L in KFAC_LAYERS:
            hooks.append(layers[L].self_attn.q_proj.register_forward_pre_hook(_mk(f"attn/L{L}")))
            hooks.append(layers[L].mlp.gate_proj.register_forward_pre_hook(_mk(f"mlp/L{L}")))
        with torch.no_grad():
            for it in items[:12]:
                SS.clean_forward(it, d)                   # zero the shared holder (stale-probe bug)
        for h in hooks:
            h.remove()
        for k in grams:
            grams[k] /= cnt[k]
        rep["sigma_y"] = {k: {"n_rows": cnt[k], "trace": float(torch.trace(g))}
                          for k, g in grams.items()}
    _w()

    # ---- stiffness spectra of Σ_y ⊗ F̄_z per (layer, matrix-type) --------------------
    with tm.stage("spectra"):
        spec_rows = {}
        f_evals = {}
        for L in KFAC_LAYERS:
            key = f"L{L}/dc/ppl"
            if key in bank:
                e = bank[key]
                spx = e["spikes"][:8]
                top = [s["eval_a1"] for s in spx]
                Ux = e["evecs_A1"][:, [s["idx"] for s in spx]].double() if spx else None
                bulkx = SC.band_bulk_from_traces(e["traces_sb"], sb, Ux,
                                                 top, basis)
                f_evals[L] = {"top": top, "bulk_per_band": bulkx,
                              "trace": e["trace_total"]}
        for L in KFAC_LAYERS:
            if L not in f_evals:
                continue
            for space, types in (("attn", ("q_proj", "k_proj", "v_proj")),
                                 ("mlp", ("gate_proj", "up_proj"))):
                G = grams[f"{space}/L{L}"]
                ev = torch.linalg.eigvalsh(0.5 * (G + G.T)).flip(0).clamp_min(0)
                spec_rows[f"L{L}/{space}"] = {
                    "sigma_y_top8": [float(x) for x in ev[:8]],
                    "sigma_y_trace": float(ev.sum()),
                    "f_top": f_evals[L]["top"], "f_trace": f_evals[L]["trace"],
                    "kron_top_eig": (float(ev[0]) * f_evals[L]["top"][0]
                                     if f_evals[L]["top"] else None),
                    "types": list(types)}
        rep["stiffness_spectra"] = spec_rows
    _w()

    # ---- completion↔direct fidelity (32 probes) + K-FAC fidelity gate ---------------
    with tm.stage("fidelity"):
        gen = torch.Generator(device="cpu").manual_seed(1201)
        probes, qc, qd = [], [], []
        Uur = lib["U_unt_residual"].double()
        for j in range(N_DIRECT // 2):
            probes.append(Uur[:, j % Uur.shape[1]])
        for j in range(N_DIRECT - len(probes)):
            g = torch.randn(d, generator=gen, dtype=torch.float64)
            probes.append(g / g.norm())
        for v in probes:
            qc.append(qhat(v))
            qd.append(direct_q(v))
        rho_completion = AF.spearman(np.array(qc), np.array(qd))
        rep["completion_fidelity"] = {"rho": rho_completion, "q_completion": qc, "q_direct": qd}
        # K-FAC gate: 32 random rank-1 ΔW at q_proj of the 2 layers, direct vs kron form
        kf_direct, kf_kron = [], []
        for L in KFAC_LAYERS:
            mod = layers[L].self_attn.q_proj
            inst_w, hold_w = SC.make_weight_injector(mod)
            items_w = [SS.build_ppl_item(model, proc, dev, wd, inst_w, hold_w)
                       for wd in wins_a[:6]]
            G = grams[f"attn/L{L}"]
            key = f"L{L}/dc/ppl"
            e = bank[key]
            spx = e["spikes"][:8]
            Ux = e["evecs_A1"][:, [s["idx"] for s in spx]].double() if spx else None
            lamx = [s["eval_a1"] for s in spx]
            bulkx = SC.band_bulk_from_traces(e["traces_sb"], sb, Ux, lamx, basis)
            d_out = mod.out_features
            for j in range(N_KFAC_PROBES // 2):
                u = torch.randn(d_out, generator=gen, dtype=torch.float64)
                y = torch.randn(d, generator=gen, dtype=torch.float64)
                dW = 1e-3 * torch.outer(u / u.norm(), y / y.norm())
                q_direct_w = 0.0
                for it in items_w:
                    q_direct_w += SC.ggn_quadform(it, dW.float().to(dev), path=path) / it["T"]
                kf_direct.append(q_direct_w / len(items_w))
                # NOTE: q_proj output is 2048-wide (8 heads × 256) but F̄_z is measured on the
                # RESIDUAL 2048 space — the K-FAC form here treats the q output in the residual
                # completion basis; the fidelity ρ is exactly the test of that identification.
                kf_kron.append(SC.kfac_stiffness(dW, G, Ux.float() if Ux is not None else
                                                 torch.zeros(2048, 0),
                                                 lamx, bulkx, basis, sb))
        kfac_rho = AF.spearman(np.array(kf_kron), np.array(kf_direct))
        rep["kfac_gate"] = {"rho": kfac_rho, "n": len(kf_direct),
                            "direct": kf_direct, "kron": kf_kron}
        print(f"[k] completion rho={rho_completion:.3f} kfac rho={kfac_rho:.3f}", flush=True)
    _w()

    # ---- stiffness quantiles vs matched-rank Haar bands -----------------------------
    with tm.stage("quantiles"):
        (out / "pre_unblind_k.json").write_text(json.dumps(
            {"row": "P8", "prediction": "H_core-stiff", "credence": "med-low (favourite #2)"},
            indent=2))
        gen = torch.Generator(device="cpu").manual_seed(1301)
        haar_q = []
        for i in range(N_HAAR_BAND):
            g = torch.randn(d, generator=gen, dtype=torch.float64)
            haar_q.append(qhat(g / g.norm()))
        haar_q = np.array(haar_q)

        def object_row(name, Q):
            cols = [Q[:, j] / Q[:, j].norm() for j in range(Q.shape[1])]
            vals = np.array([qhat(v) for v in cols])
            qbar, per = SC.stiffness_quantile(vals, haar_q)
            k = len(cols)
            rng = np.random.default_rng(7)
            null_means = np.array([rng.choice(len(haar_q), k, replace=False).astype(int)
                                   for _ in range(2000)])
            means = np.array([np.searchsorted(np.sort(haar_q), haar_q[idx]).mean() / len(haar_q)
                              for idx in null_means])
            # (count+1)/(n+1): a resampled p is never exactly 0 (the parent bh_pass has a
            # thresh>0 boundary that voids p=0 families — and 0 is statistically wrong anyway)
            p_hi = float(((means >= qbar).sum() + 1) / (len(means) + 1))
            p_lo = float(((means <= qbar).sum() + 1) / (len(means) + 1))
            return {"name": name, "rank": k, "qbar_quantile": qbar,
                    "haar_p05": float(np.quantile(means, 0.05)),
                    "haar_p95": float(np.quantile(means, 0.95)),
                    "p_two_sided": min(1.0, 2 * min(p_hi, p_lo)),
                    "stiffness_values": [float(v) for v in vals]}
        objs = {"U_unt_residual": lib["U_unt_residual"].double(),
                "U_wr_o_proj": lib["U_wr_o_proj"].double(),
                "U_wr_down_proj": lib["U_wr_down_proj"].double(),
                "task_footprint": lib["task_footprint"].double(),
                "v1": lib["v1"].double().reshape(-1, 1)}
        rows = {n: object_row(n, Q) for n, Q in objs.items()}
        # input-space side: U_unt_qk vs Σ_y(attn, L0) Haar band
        Gq = grams["attn/L0"]
        Uqk = lib["U_unt_qk"].double()
        vals_in = np.array([float(Uqk[:, j] @ Gq @ Uqk[:, j] / (Uqk[:, j] @ Uqk[:, j]))
                            for j in range(Uqk.shape[1])])
        haar_in = []
        for i in range(N_HAAR_BAND // 4):
            g = torch.randn(d, generator=gen, dtype=torch.float64)
            g = g / g.norm()
            haar_in.append(float(g @ Gq @ g))
        qbar_in, _ = SC.stiffness_quantile(vals_in, np.array(haar_in))
        rows["U_unt_qk_sigma_side"] = {"name": "U_unt_qk_sigma_side", "rank": Uqk.shape[1],
                                       "qbar_quantile": qbar_in}
        rep["stiffness_rows"] = rows
        SS.emit_family(rep, "k_secondary",
                       [rows[n].get("p_two_sided", 1.0) for n in
                        ("U_wr_o_proj", "U_wr_down_proj", "task_footprint")],
                       ["U_wr_o_proj", "U_wr_down_proj", "task_footprint"])
        head = rows["U_unt_residual"]
        rep["rk1"] = dict(SC.adjudicate_rk1(
            kfac_rho=rep["kfac_gate"]["rho"], qbar=head["qbar_quantile"],
            haar_p05=head["haar_p05"], haar_p95=head["haar_p95"],
            quartile=head["qbar_quantile"]),
            completion_rho=rep["completion_fidelity"]["rho"])
        print(f"[k] R-K1 {rep['rk1']['verdict']} qbar={head['qbar_quantile']:.3f}", flush=True)
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_K_OK", json.dumps({"rk1": rep["rk1"]["verdict"],
                                      "kfac_rho": rep["kfac_gate"]["rho"]}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
