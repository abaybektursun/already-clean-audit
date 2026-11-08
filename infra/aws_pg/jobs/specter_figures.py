"""SPECTER figure suite (CPU) — every §10 figure from banked metrics + the operator bank; each
figure: claim in the title backed by number + CI + null; one-line "how to read"; null overlaid;
provenance footer (input hashes); PNG + PDF; colourblind-safe (Okabe–Ito); the full-sweep rule
(every swept point + every component curve).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.specter_figures --type r7i.2xlarge --no-weights \
      --s3-prefix results/specter_figures --budget-min 60
"""
import argparse
import json
import math
import os
import pathlib

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _save(fig, out, name, provenance):
    fig.text(0.01, 0.005, provenance, fontsize=5, color="0.4")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    print(f"[fig] {name}", flush=True)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = _style()
    import numpy as np
    import torch

    from infra.aws_pg.jobs import _specter_shared as SS

    rep = {"object": "specter_figures_v1", "made": []}
    tm = SS.StageTimer(rep)

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    def _fetch_json(uri, name):
        return json.loads(SS.fetch(uri, f"/tmp/specter/{name}").read_text())

    with tm.stage("load"):
        o_m = _fetch_json(f"{SS.SPECTER_O}/metrics.json", "fo.json")
        bank = torch.load(SS.fetch(f"{SS.SPECTER_O}/operator_bank.pt", "/tmp/specter/ob.pt"),
                          map_location="cpu", weights_only=False)
        prov_o = f"operator_bank {o_m.get('bank_sha256', '')[:12]} | specter_o green={o_m.get('green')}"
        c_m = _fetch_json(f"{SS.S3_BUCKET}/results/specter_c/out/metrics.json", "fc.json")
        art = {"o": o_m, "c": c_m}
        for nm in ("s", "h", "k", "b", "m_sealed", "m_unblind"):
            art[nm] = _fetch_json(f"{SS.S3_BUCKET}/results/specter_{nm}/out/metrics.json",
                                  f"f{nm}.json")
        # the battery's rows live in m_sealed; the R-M1 statistics in the unblind artifact
        art["m_sealed"]["rm1"] = art["m_unblind"]["rm1"]

    # ---- F-O1: spectrum panorama -------------------------------------------------------
    with tm.stage("F-O1"):
        sites = sorted({k.split("/")[0] for k in bank})
        modes = ("dc", "sin1", "sin4", "white")
        fig, axes = plt.subplots(len(sites), 2, figsize=(9, 3 * len(sites)), squeeze=False)
        for si, site in enumerate(sites):
            for li, loss in enumerate(("ppl", "vqa")):
                ax = axes[si][li]
                for mi, m in enumerate(modes):
                    k = f"{site}/{m}/{loss}"
                    if k not in bank:
                        continue
                    ev1 = np.asarray(bank[k]["evals_A1"][:64], float)
                    ev2 = np.asarray(bank[k]["evals_A2"][:64], float)
                    ax.semilogy(np.maximum(ev1, 1e-12), color=OI[mi], label=f"{m} A1", lw=1.2)
                    ax.semilogy(np.maximum(ev2, 1e-12), color=OI[mi], ls="--", alpha=0.6,
                                label=f"{m} A2", lw=0.8)
                    bulk = bank[k]["trace_total"] / 2048.0
                    ax.axhline(bulk, color=OI[mi], ls=":", alpha=0.4, lw=0.7)
                ax.set_title(f"{site} / {loss}", fontsize=8)
                ax.set_xlabel("Ritz index")
                if li == 0:
                    ax.set_ylabel("eigenvalue (per-token-energy units)")
        axes[0][0].legend(fontsize=5, ncol=4)
        fig.suptitle("F-O1 — the spectrum panorama: eigenvalue ladders per (site × mode × loss); "
                     "dashed = split-half A2 (overlap = stability), dotted = bulk mean tr/d.\n"
                     "How to read: a spike is real where solid and dashed agree above the dotted bulk.",
                     fontsize=8)
        _save(fig, out, "F-O1_spectrum_panorama", prov_o)
        plt.close(fig)
        rep["made"].append("F-O1")
    _w()

    # ---- F-O2: kernel censuses ----------------------------------------------------------
    with tm.stage("F-O2"):
        rows = [(k, v) for k, v in o_m["ro2"].items()]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        labels, vals, cols = [], [], []
        for k, v in rows:
            labels.append(k + f"\n[{v['verdict']}]")
            vals.append(max(v["ratio"], 1e-8))
            cols.append(OI[0] if "radial" in k else OI[1])
        ax.bar(range(len(vals)), vals, color=cols)
        ax.set_yscale("log")
        ax.axhline(1e-3, color="k", ls=":", lw=0.8)
        ax.axhline(0.05, color=OI[2], ls=":", lw=0.8)
        ax.axhline(0.1, color="r", ls="--", lw=0.8)
        ax.axhline(0.25, color=OI[3], ls=":", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5, rotation=30, ha="right")
        ax.set_ylabel("mean Rayleigh / bulk")
        ax.set_title("F-O2 — gauge-kernel censuses vs bulk (log): the theorem checks as bars.\n"
                     "How to read: key-coherent must sit below 1e-3 of bulk (exact theorem); "
                     "radial below 0.05 (gauge) — thresholds dotted, halt line dashed red.",
                     fontsize=8)
        _save(fig, out, "F-O2_kernel_census", prov_o)
        plt.close(fig)
        rep["made"].append("F-O2")
    _w()

    # ---- F-C1: the carrier regression ---------------------------------------------------
    with tm.stage("F-C1"):
        rows = c_m["c1_rows"]
        reg = c_m["c1"].get("regressor", "reg_udc")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
        for ax, (leg, ykey, sel) in zip(axes, (
                ("ppl (6 unique)", "log1p_dppl", lambda r: r["src"] == "disc"),
                ("vqa (9 draws)", "effect", lambda r: r["effect"] is not None))):
            rr = [r for r in rows if sel(r)]
            x = [r[reg] for r in rr]
            y = [r[ykey] for r in rr]
            ax.plot(x, y, "o", color=OI[0])
            for r in rr:
                ax.annotate(f"s{r['draw']}/{r['level']}", (r[reg], r[ykey]), fontsize=5)
            ax.set_xscale("log")
            ax.set_xlabel(f"⟨δ̄, u_dc⟩² ({reg})")
            ax.set_ylabel(ykey)
            ax.set_title(leg, fontsize=8)
        t = c_m["c1"]
        fig.suptitle(f"F-C1 — the carrier regression [{t['verdict']}]: τ_ppl={t['tau_ppl']:.2f}, "
                     f"τ_vqa={t['tau_vqa']:.2f} (exact permutation p emitted in metrics).\n"
                     "How to read: H_carrier-named needs monotone ordering (τ ≥ 0.6) on a leg.",
                     fontsize=8)
        _save(fig, out, "F-C1_carrier_regression", f"specter_c green={c_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-C1")
    _w()

    # ---- F-C2: band × mode trace heatmap -------------------------------------------------
    with tm.stage("F-C2"):
        modes = ("dc", "sin1", "sin4", "white")
        M = np.array([[math.log10(max(x, 1e-12)) for x in bank[f"L0/{m}/ppl"]["permode_sb"]]
                      for m in modes])
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(modes)))
        ax.set_yticklabels(modes)
        ax.set_xlabel("sub-band (energy-ordered)")
        fig.colorbar(im, ax=ax, label="log10 per-direction susceptibility")
        ax.set_title("F-C2 — the quadratic bi-spectral surface: tr(H_ω P_b)/k_b at L0 (ppl).\n"
                     f"How to read: the S3 reversal is quadratic iff band1 white≫dc and band4 "
                     f"dc>white [{c_m['c2']['verdict']}].", fontsize=8)
        _save(fig, out, "F-C2_trace_heatmap", prov_o)
        plt.close(fig)
        rep["made"].append("F-C2")
    _w()

    # ---- F-C3: coherence dial -------------------------------------------------------------
    with tm.stage("F-C3"):
        c3 = c_m["c3"]
        fig, ax = plt.subplots(figsize=(4.5, 3.4))
        if "R_l14" in c3 and c3.get("R_l14") is not None:
            xs = ["L0", "L14"]
            Rv = [c3["R_l0"], c3["R_l14"]]
            bv = [c3["ratio_l0"], c3["ratio_l14"]]
        else:
            xs = ["L0"]
            Rv = [c3["R_l0"]]
            bv = [c3["ratio_l0"]]
        x = np.arange(len(xs))
        ax.bar(x - 0.17, Rv, width=0.34, color=OI[0], label="R = tr(H_dc)/tr(H_white)")
        ax.bar(x + 0.17, bv, width=0.34, color=OI[1], label="behavioural ratio (recomputed)")
        for xi, (r, b) in enumerate(zip(Rv, bv)):
            ax.plot([xi - 0.17, xi - 0.17], [r - 0.4, r + 0.4], color="k", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(xs)
        ax.legend(fontsize=6)
        ax.set_title(f"F-C3 — the coherence dial [{c3['verdict']}]; band = the frozen ±0.4.\n"
                     "How to read: bars agreeing within the whisker at both sites ⇒ the dial is "
                     "quadratic-order.", fontsize=8)
        _save(fig, out, "F-C3_dial", f"specter_c green={c_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-C3")
    _w()

    # ---- F-C4: census matrix ---------------------------------------------------------------
    with tm.stage("F-C4"):
        cen = c_m["c4_census"]
        nulls = c_m["c4_haar_p95"]
        probes = sorted(cen)
        objs = sorted(next(iter(cen.values())))
        M = np.array([[cen[p][o] / max(3.0 * nulls[o], 1e-12) for o in objs] for p in probes])
        fig, ax = plt.subplots(figsize=(10, 0.35 * len(probes) + 2))
        im = ax.imshow(np.minimum(M, 3.0), aspect="auto", cmap="magma", vmin=0, vmax=3)
        ax.set_xticks(range(len(objs)))
        ax.set_xticklabels(objs, fontsize=5, rotation=60, ha="right")
        ax.set_yticks(range(len(probes)))
        ax.set_yticklabels(probes, fontsize=5)
        fig.colorbar(im, ax=ax, label="overlap / (3× Haar p95); ≥1 = identification")
        ax.set_title("F-C4 — census: stable spikes × the named library, matched-rank Haar nulls.\n"
                     "How to read: any cell ≥ 1 is an identification (R-C4); an all-dark row is "
                     "new structure.", fontsize=8)
        _save(fig, out, "F-C4_census", f"specter_c green={c_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-C4")
    _w()

    # ---- F-K1: stiffness quantiles ----------------------------------------------------------
    with tm.stage("F-K1"):
        k_m = art["k"]
        rows = k_m["stiffness_rows"]
        names = [n for n in rows if "qbar_quantile" in rows[n]]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for i, n in enumerate(names):
            r = rows[n]
            ax.plot([i], [r["qbar_quantile"]], "o", color=OI[0], ms=7)
            if "haar_p05" in r:
                ax.plot([i, i], [r["haar_p05"], r["haar_p95"]], color="0.6", lw=5, alpha=0.5)
        ax.axhline(0.5, color="k", ls=":", lw=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=6, rotation=25, ha="right")
        ax.set_ylabel("mean stiffness quantile in the Haar band")
        ax.set_ylim(0, 1)
        ax.set_title(f"F-K1 — CORE⟂SPIKE [{k_m['rk1']['verdict']}]: named objects' stiffness "
                     f"quantiles vs matched-rank Haar bands (grey); K-FAC gate "
                     f"ρ={k_m['kfac_gate']['rho']:.2f}.\nHow to read: dot above the grey band = "
                     "stiff, below = sloppy, inside = generic.", fontsize=8)
        _save(fig, out, "F-K1_stiffness", f"specter_k green={k_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-K1")
    _w()

    # ---- F-S1: per-spike amplitude scans -----------------------------------------------------
    with tm.stage("F-S1"):
        s_m = art["s"]
        cells = s_m["cells"]
        preds = {p["spike"]: p for p in s_m["quadratic_predictions"]}
        n_sp = len(s_m["promoted"])
        fig, axes = plt.subplots(2, (n_sp + 1) // 2, figsize=(3 * ((n_sp + 1) // 2), 6),
                                 squeeze=False)
        for i in range(n_sp):
            ax = axes[i % 2][i // 2]
            sp = [c for c in cells if c.get("spike") == i and c["arm"] == "spike"]
            al = sorted({abs(c["alpha"]) for c in sp})
            meas = [np.mean([c["log1p_dppl"] for c in sp if abs(abs(c["alpha"]) - a) < 1e-9])
                    for a in al]
            pred = [preds[i][f"pred_log1p_dppl@{a}"] for a in al]
            nl = [c["log1p_dppl"] for c in cells
                  if c.get("spike") == i and c["arm"] == "bingham_null"]
            rot = [c["log1p_dppl"] for c in cells
                   if c.get("spike") == i and c["arm"] == "rotated"]
            ax.plot(al, meas, "o-", color=OI[0], label="measured")
            ax.plot(al, pred, "s--", color=OI[1], label="quadratic pred")
            ax.fill_between(al, [p / 2 for p in pred], [p * 2 for p in pred], color=OI[1],
                            alpha=0.15, label="2× band")
            if nl:
                ax.axhspan(min(nl), max(nl), color=OI[2], alpha=0.2, label="Bingham null @0.1")
            if rot:
                ax.axhline(rot[0], color=OI[3], ls=":", label="rotated @0.1")
            rq = s_m["r_quad"].get(f"spike{i}", 0.0)
            ax.axvline(max(rq, 1e-3), color="0.5", lw=0.7)
            ax.set_title(f"spike{i} [{s_m['rs1'].get(f'spike{i}/ppl', {}).get('verdict', '')}]",
                         fontsize=7)
            ax.set_xlabel("α (·rms)")
            ax.set_ylabel("log(1+Δppl)")
            if i == 0:
                ax.legend(fontsize=5)
        fig.suptitle("F-S1 — all 8 promoted spikes: amplitude scan, quadratic band, em-null, "
                     "rotated control, r_quad (grey vline). Full-sweep rule: every point + every "
                     "component curve.", fontsize=8)
        _save(fig, out, "F-S1_spike_scans", f"specter_s green={s_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-S1")
    _w()

    # ---- F-H1: worst vs typical -----------------------------------------------------------
    with tm.stage("F-H1"):
        h_m = art["h"]
        rows = h_m["remeasure"]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for i, (k, r) in enumerate(sorted(rows.items())):
            ax.plot([i], [r["battery_log1p_dppl"]], "o", color=OI[1], ms=8, label="found max"
                    if i == 0 else None)
            ax.plot([i] * len(r["null_draws"]), r["null_draws"], ".", color=OI[0],
                    label="matched-energy null" if i == 0 else None)
            ax.plot([i], [r["quad_pred_log1p"]], "s", color=OI[2],
                    label="quadratic pred" if i == 0 else None)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(sorted(rows), fontsize=6)
        ax.set_ylabel("log(1+Δppl), standard battery")
        ax.set_title(f"F-H1 — the worst-vs-typical frontier [{h_m['rh1']['verdict']}]: found "
                     "directions re-measured (the number of record) vs Bingham-typical + the "
                     "quadratic prediction.\nHow to read: G = found/typical; G≤2 certifies "
                     "concentration.", fontsize=8)
        ax.legend(fontsize=6)
        _save(fig, out, "F-H1_frontier", f"specter_h green={h_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-H1")
    _w()

    # ---- F-M1: sealed parity -----------------------------------------------------------------
    with tm.stage("F-M1"):
        m_m = art["m_sealed"]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        for ax, ro, mkey in ((axes[0], "ppl", "measured_log1p_dppl"),
                             (axes[1], "vqa", "measured_vqa_effect")):
            sc = m_m["scored"]
            y = [s[mkey] for s in sc]
            m4 = [s["pred"][f"M4_{ro}"] for s in sc]
            m0 = [s["pred"][f"M0_{ro}"] for s in sc]
            ax.plot(m4, y, "o", color=OI[0], label="M4 (operator)")
            ax.plot(m0, y, "^", color=OI[1], label="M0 (energy)")
            lim = [min(min(m4), min(m0), min(y)), max(max(m4), max(m0), max(y))]
            ax.plot(lim, lim, color="0.7", lw=0.7)
            r = m_m["rm1"][ro]
            ax.set_title(f"{ro}: Δρ={r['drho']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] "
                         f"ρ_M4={r['rho_m4']:.2f} ρ_M0={r['rho_m0']:.2f}\n"
                         f"[{r['adjudication']['verdict']}]", fontsize=7)
            ax.set_xlabel("predicted")
            ax.set_ylabel("measured (sealed)")
            ax.legend(fontsize=6)
        fig.suptitle("F-M1 — sealed parity, M4 and M0 side by side (20 valid never-run cells, "
                     "adversarially selected).", fontsize=8)
        _save(fig, out, "F-M1_sealed_parity", f"specter_m_sealed green={m_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-M1")
    _w()

    # ---- F-B1: 18-layer fan + collapse matrix + hybrid panel -----------------------------------
    with tm.stage("F-B1"):
        b_m = art["b"]
        curves = {int(k): v for k, v in b_m["curves"].items()}
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        cmap = plt.get_cmap("viridis")
        rungs = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
        for L in sorted(curves):
            axes[0].plot(rungs, curves[L], color=cmap(L / 17), lw=1,
                         label=f"L{L}" if L in (0, 4, 8, 14, 17) else None)
        axes[0].legend(fontsize=5)
        axes[0].set_xlabel("E / E_max(25)")
        axes[0].set_ylabel("energy dispersion")
        axes[0].set_title("the 18-layer fan", fontsize=8)
        Ls = sorted(curves)
        Mx = np.zeros((18, 18))
        for i in Ls:
            for j in Ls:
                Mx[i, j] = float(np.max(np.abs(np.asarray(curves[i]) - np.asarray(curves[j]))))
        im = axes[1].imshow(Mx, cmap="magma")
        fig.colorbar(im, ax=axes[1], label="pairwise max|Δ|")
        axes[1].set_title(f"collapse matrix [{b_m['rb1']['verdict']}, "
                          f"peel={b_m['rb1']['peel_off_layer']}]", fontsize=8)
        hy = b_m["hybrid_swap"]["rows"]
        names = sorted(hy)
        axes[2].bar(range(len(names)), [hy[n]["dev_to_ref"] for n in names], color=OI[0])
        axes[2].axhline(list(hy.values())[0]["dev_before"], color=OI[1], ls="--",
                        label="before swap")
        axes[2].set_xticks(range(len(names)))
        axes[2].set_xticklabels(names, fontsize=5, rotation=20, ha="right")
        axes[2].legend(fontsize=6)
        axes[2].set_title("hybrid-swap sensitivity", fontsize=8)
        fig.suptitle("F-B1 — the R10 invariant mapped: dispersion-curve fan, collapse matrix, "
                     "and which spectral feature moves the curve.", fontsize=8)
        _save(fig, out, "F-B1_collapse", f"specter_b green={b_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-B1")
    _w()

    # ---- F-G1: whitening scatter ----------------------------------------------------------------
    with tm.stage("F-G1"):
        b_m = art["b"]
        spectra = torch.load(SS.fetch(f"{SS.S3_BUCKET}/results/specter_b/out/spectra_18.pt",
                                      "/tmp/specter/spectra18.pt"),
                             map_location="cpu", weights_only=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        for i, (site, fit) in enumerate(sorted(b_m["whitening"].items())):
            L = int(site[1:])
            key = f"{site}/dc/ppl"
            if key not in bank or fit.get("slope") is None:
                continue
            fe = np.asarray(bank[key]["evals_A1"][:64], float)
            se = np.asarray(spectra[L][:64], float)
            ok = (fe > 0) & (se > 0)
            ax.loglog(se[ok], fe[ok], ".", color=OI[i], ms=4,
                      label=f"{site}: slope {fit['slope']:.2f} "
                            f"[{fit['ci95'][0]:.2f},{fit['ci95'][1]:.2f}]")
        ax.set_xlabel("Σ eigenvalue")
        ax.set_ylabel("F̄ eigenvalue (dc, ppl)")
        ax.legend(fontsize=6)
        ax.set_title("F-G1 — whitening scatter: F̄ vs Σ spectra, fitted exponent + CI per site.\n"
                     "Registered expectation: no single exponent survives.", fontsize=8)
        _save(fig, out, "F-G1_whitening", f"specter_b green={b_m.get('green')}")
        plt.close(fig)
        rep["made"].append("F-G1")
    _w()

    SS.assert_gpu_seconds(rep)
    print("SPECTER_FIGURES_OK", json.dumps({"n": len(rep['made'])}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
