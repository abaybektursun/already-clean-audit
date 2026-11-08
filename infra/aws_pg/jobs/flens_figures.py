"""flens figure suite (prereg §10) — weight-free; reads the banked flens JSONs and writes the figures.
PNG (review) + PDF (paper), colourblind-safe (Okabe-Ito), bbox_inches='tight'. Run on a pg CPU
instance (`--no-weights`); no model, no GPU.

F1 is the MANDATED COMPLETE k-sweep (`feedback_complete_sweep_visualization`): every K on the grid, ALL
component curves overlaid — deleted-floor (RF=0), intact ceiling (RF=1), the K=full recoverable ceiling,
real SAE-feature RF(K)+CI, the matched-random-K band, the shuffle null, and the SVD-component patch
(the BASIS control), with the K* marker and the δ_rand margin. No K hidden, no component dropped.

    python -m infra.aws_pg.jobs.flens_figures --discovery-prefix results/flens_disc \
        --confirm-prefix results/flens_confirm --sha <git-sha>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


# Okabe-Ito colourblind-safe palette.
C_REAL = "#0072B2"      # real SAE-feature RF(K) (blue)
C_RAND = "#999999"      # matched-random-K null band (grey)
C_SHUF = "#CC79A7"      # item-shuffle null (pink)
C_SVD = "#A6761D"       # SVD-component patch — the BASIS control (brown)
C_FLOOR = "#000000"     # deleted floor (black)
C_CEIL = "#009E73"      # intact ceiling (green)
C_RECOV = "#56B4E9"     # K=full recoverable ceiling (sky)
C_KSTAR = "#D55E00"     # K* marker (vermillion)
C_TAU = "#E69F00"       # τ_restore threshold (orange)
C_SUPP = "#0072B2"      # suppressed features (intact>deleted) (blue)
C_CORR = "#D55E00"      # corrupting features (deleted>intact) (vermillion)


def _pull(prefix, dest):
    """Download <prefix>/out/flens.json from S3 to dest; return the parsed dict (or None)."""
    if not prefix:
        return None
    from infra.aws_pg import config as C
    uri = f"s3://{C.BUCKET}/{prefix}/out/flens.json"
    subprocess.run(["aws", "s3", "cp", uri, str(dest), "--region", C.REGION], check=True)
    return json.loads(pathlib.Path(dest).read_text())


def _footer(ax, text):
    ax.text(0.0, -0.17, text, transform=ax.transAxes, fontsize=6, color="#555555", va="top")


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=140)


def fig_complete_sweep(disc, out, sha):
    """F1 — the MANDATED complete K-sweep. Every K; ALL components: deleted-floor (RF=0), intact ceiling
    (RF=1), the K=full recoverable ceiling (all features→intact), real SAE-feature RF(K)+95% CI, the
    5-seed matched-random-K band (mean±sd), the shuffle null, the SVD-component patch (basis control),
    the K* marker, the τ_restore line + δ_rand margin. The lens claim is the real (blue) curve above the
    SVD (brown) curve at matched small K; the artifact gate is real above the random (grey) band by δ."""
    import matplotlib.pyplot as plt
    import numpy as np

    d = disc["discovery"]
    sweep = d["sweep"]
    ks = sorted(int(k) for k in sweep)
    x = np.array(ks)
    real = np.array([sweep[str(k)]["rf_real"] for k in ks])
    real_lo = np.array([sweep[str(k)].get("rf_real_ci_lo", sweep[str(k)]["rf_real"]) for k in ks])
    real_hi = np.array([sweep[str(k)].get("rf_real_ci_hi", sweep[str(k)]["rf_real"]) for k in ks])
    rmean = np.array([sweep[str(k)]["rf_random_mean"] for k in ks])
    rsd = np.array([sweep[str(k)]["rf_random_sd"] for k in ks])
    smean = np.array([sweep[str(k)]["rf_shuffle_mean"] for k in ks])
    ssd = np.array([sweep[str(k)]["rf_shuffle_sd"] for k in ks])
    svd = np.array([sweep[str(k)].get("rf_svd", np.nan) for k in ks])
    svdw = np.array([sweep[str(k)].get("rf_svd_white", np.nan) for k in ks])
    tau = disc["tau_restore"]
    delta = disc["delta_rand"]
    kstar = d.get("minimal_k_star")
    recoverable = disc.get("rf_full")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    # reference lines (anchors)
    ax.axhline(0.0, color=C_FLOOR, lw=1.0, ls="-", label="deleted floor (RF=0)")
    ax.axhline(1.0, color=C_CEIL, lw=1.0, ls="-", label="intact ceiling (RF=1)")
    if recoverable is not None:
        ax.axhline(recoverable, color=C_RECOV, lw=1.2, ls="-.",
                   label=f"K=full recoverable ceiling (all features→intact) = {recoverable:.2f}")
    ax.axhline(tau, color=C_TAU, lw=1.0, ls="--", label=f"τ_restore = {tau}")
    # matched-random-K null band (LOAD-BEARING) + the δ margin shaded where real clears it
    ax.fill_between(x, rmean - rsd, rmean + rsd, color=C_RAND, alpha=0.30,
                    label="matched-random-K null (5-seed mean±sd)")
    ax.plot(x, rmean, "o--", color=C_RAND, ms=4, lw=1.2)
    ax.fill_between(x, rmean + delta, np.maximum(rmean + delta, real), where=(real > rmean + delta),
                    color=C_REAL, alpha=0.10, label=f"real − random margin > δ_rand={delta}")
    # shuffle null (item-specificity characterization)
    ax.plot(x, smean, "s:", color=C_SHUF, ms=4, lw=1.2, label="item-shuffle null (5-seed mean)")
    ax.fill_between(x, smean - ssd, smean + ssd, color=C_SHUF, alpha=0.12)
    # SVD-component patches — the BASIS controls (same operation, linear basis): raw + the
    # massive-activation-robust WHITENED control (audit B1). The lens claim is real ABOVE BOTH.
    if np.isfinite(svd).any():
        ax.plot(x, svd, "D-.", color=C_SVD, ms=4, lw=1.4,
                label="SVD-patch RAW (linear basis control)")
    if np.isfinite(svdw).any():
        ax.plot(x, svdw, "x:", color=C_SVD, ms=6, lw=1.6,
                label="SVD-patch WHITENED (massive-activation-robust control)")
    # real SAE-feature RF(K) with bootstrap CI
    ax.errorbar(x, real, yerr=[real - real_lo, real_hi - real], fmt="o-", color=C_REAL, ms=6, lw=2.0,
                capsize=3, label="real SAE-feature RF(K) (top-K |Δf| features→intact) + 95% CI")
    if kstar is not None:
        ax.axvline(kstar, color=C_KSTAR, ls=":", lw=1.6, label=f"K* = {kstar} (minimal τ+random)")
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.axvline(disc["k_small"], color="#888888", ls=(0, (1, 3)), lw=1.0)
    ax.text(disc["k_small"], ax.get_ylim()[1], f" K_small={disc['k_small']}", fontsize=6, color="#888888", va="top")
    sparse = kstar is not None and kstar <= disc["k_small"]
    verdict = "SPARSE in the feature basis (lens was blind)" if sparse else "NOT feature-sparse at small K"
    ax.set_title(f"Is the L16 deletion sparse in the SAE feature basis? — {verdict}", fontsize=11, pad=10)
    ax.set_xlabel("K  (number of SAE features set to their intact per-item values at L16, last token)")
    ax.set_ylabel("RF(K) = (acc_patch(K) − floor) / (intact − floor)")
    ax.legend(fontsize=6.3, loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    ax.grid(True, alpha=0.2)
    _footer(ax, f"F1 COMPLETE K-SWEEP · flens {disc['eval_split']} · n_eval={disc['n_eval_items']} "
                f"d_model={disc['d_model']} d_sae={disc['d_sae']} · SAE {disc['sae_choice']['rel_path']} "
                f"· real>SVD ⇒ lens matters; real>random by δ ⇒ feature-specific · SHA {sha} · DISCOVERY")
    _save(fig, out, "flens_F1_complete_k_sweep")
    plt.close(fig)


def fig_sae_fidelity(disc, out, sha):
    """F2 — SAE faithfulness at L16. The GATES (study-aligned + robust): the intact round-trip MCQ
    (readout preserved) and the directional cosine floor. EV is REPORTED only — it is OOD-sensitive on
    the PG2 blank-image multimodal forward (text-trained Gemma Scope SAE: in-dist text EV≈+0.55, but the
    multimodal residual is mean-dominated → EV<0 while cosine≈0.9 and the readout is preserved)."""
    import matplotlib.pyplot as plt

    f = disc["sae_faithfulness"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.0, 3.9))
    # GATE panel: round-trip MCQ + cosine
    a2_ok = f.get("cosine_pass", True)
    a1.bar([0, 1], [f["intact_mcq"], f["roundtrip_mcq"]], width=0.5, color=[C_CEIL, C_REAL])
    a1.axhline(f["intact_mcq"] - f["roundtrip_retain_delta"], color=C_TAU, ls="--", lw=1,
               label=f"gate: intact − {f['roundtrip_retain_delta']}")
    a1.set_xticks([0, 1]); a1.set_xticklabels(["intact MCQ", "SAE round-trip\nMCQ"])
    a1.set_ylabel("blank-image MMLU-MCQ accuracy")
    a1.set_title(f"round-trip MCQ = {f['roundtrip_mcq']:.3f}  ({'PASS' if f['roundtrip_pass'] else 'ABORT'})", fontsize=9.5)
    a1.legend(fontsize=7)
    # cosine + EV (reported) panel
    cos = f.get("recon_cosine", float("nan"))
    a2.bar([0], [cos], width=0.45, color=C_REAL, label="recon cosine (GATE)")
    a2.axhline(f.get("cosine_min", 0.5), color=C_TAU, ls="--", lw=1, label=f"cosine_min = {f.get('cosine_min', 0.5)}")
    a2.bar([1], [f["explained_variance"]], width=0.45, color=C_RAND, label="EV (reported, OOD-sensitive)")
    a2.axhline(0, color="#000000", lw=0.6)
    a2.set_xticks([0, 1]); a2.set_xticklabels(["recon\ncosine", "EV\n(reported)"])
    a2.set_ylabel("directional fidelity / explained var")
    a2.set_title(f"cosine = {cos:.3f} ({'PASS' if a2_ok else 'ABORT'});  EV = {f['explained_variance']:+.3f}", fontsize=9.5)
    a2.legend(fontsize=6.5)
    dev = f.get("deleted_explained_variance")
    fig.suptitle("F2 — SAE faithfulness at L16: GATES = round-trip MCQ + cosine; EV reported (OOD)", fontsize=9.5)
    _footer(a1, f"F2 · flens · SAE {disc['sae_choice']['rel_path']}"
                + (f" · deleted-side EV={dev:.2f}" if dev is not None else "")
                + f" · multimodal-OOD: EV<0 while cosine≈{cos:.2f} & readout preserved · SHA {sha} · DISCOVERY")
    _save(fig, out, "flens_F2_sae_fidelity")
    plt.close(fig)


def fig_feature_survival(disc, out, sha, topn=40):
    """F3 — feature survival / direction for the top-N differing features: mean f_int vs mean f_del,
    coloured suppressed (intact>deleted: silenced a present feature) vs corrupting (deleted>intact:
    activated a disabling feature). The suppressed/corrupting fractions NAME the mechanism."""
    import matplotlib.pyplot as plt
    import numpy as np

    sp = disc["direction_split"]
    fi = np.array(sp["mean_f_int_on_S"][:topn])
    fd = np.array(sp["mean_f_del_on_S"][:topn])
    supp = (np.array(sp["signed_mean_df"][:topn]) > 0)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    mx = float(max(fi.max(initial=1e-6), fd.max(initial=1e-6))) * 1.05
    ax.plot([0, mx], [0, mx], color="#aaaaaa", lw=0.8, ls="--", label="f_int = f_del (no change)")
    ax.scatter(fd[supp], fi[supp], color=C_SUPP, s=28, alpha=0.8, label="suppressed (intact>deleted)")
    ax.scatter(fd[~supp], fi[~supp], color=C_CORR, s=28, alpha=0.8, marker="^", label="corrupting (deleted>intact)")
    ax.set_xlabel("mean deleted activation  f_del")
    ax.set_ylabel("mean intact activation  f_int")
    ax.set_title(f"Top-{topn} differing features: "
                 f"{sp['frac_suppressed']*100:.0f}% suppressed / {sp['frac_corrupting']*100:.0f}% corrupting",
                 fontsize=9.5)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.2)
    _footer(ax, f"F3 · flens feature survival · top-{topn} of K_max={disc['k_max']} · "
                f"present (not f_del≈0) ⇒ NOT overwritten · SHA {sha} · DISCOVERY")
    _save(fig, out, "flens_F3_feature_survival")
    plt.close(fig)


def fig_headline(conf, out, sha):
    """F4 — confirmation headline. LEFT: RF(K*) for the feature patch vs BOTH linear basis controls
    (raw + whitened SVD) and the random descriptor, vs τ. RIGHT: the LENS GATE — the paired-bootstrap CI
    of (RF_feature − RF_svd) for each control vs δ_lens. H_sparse needs BOTH margin CI-lowers > δ_lens."""
    import matplotlib.pyplot as plt

    c = conf["confirmation"]
    dl = c.get("delta_lens", conf.get("delta_lens", 0.20))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    # LEFT — RF levels
    a1.axhline(conf["tau_restore"], color=C_TAU, ls="--", lw=1, label=f"τ_restore = {conf['tau_restore']}")
    a1.axhline(0, color="#000000", lw=0.6)
    a1.bar([0], [c["rf_real"]], width=0.55, color=C_REAL,
           yerr=[[c["rf_real"] - c["rf_ci_lo"]], [c["rf_ci_hi"] - c["rf_real"]]], capsize=5,
           label=f"real SAE-feature RF(K*={c['k_star']})")
    a1.bar([1], [c["rf_svd_raw"]], width=0.55, color=C_SVD, label="SVD raw")
    a1.bar([2], [c["rf_svd_white"]], width=0.55, color=C_SVD, alpha=0.55, hatch="//", label="SVD whitened")
    a1.bar([3], [c["rf_random_mean"]], width=0.55, color=C_RAND,
           yerr=[[c["rf_random_sd"]], [c["rf_random_sd"]]], capsize=5, label="random (descriptor)")
    a1.set_xticks([0, 1, 2, 3]); a1.set_xticklabels(["feat K*", "SVD\nraw", "SVD\nwhite", "random"])
    a1.set_ylabel("RF = (acc_patch − floor)/(intact − floor)")
    a1.set_title(f"RF at frozen K*={c['k_star']}", fontsize=9.5)
    a1.legend(fontsize=6.5, loc="best")
    # RIGHT — the lens gate (paired margin CIs vs δ_lens)
    mr = [(c["lens_margin_raw"], c["lens_margin_raw_ci_lo"], c["lens_margin_raw_ci_hi"], "vs raw-SVD"),
          (c["lens_margin_white"], c["lens_margin_white_ci_lo"], c["lens_margin_white_ci_hi"], "vs whitened-SVD")]
    a2.axhline(dl, color=C_KSTAR, ls="--", lw=1.2, label=f"δ_lens = {dl}")
    a2.axhline(0, color="#000000", lw=0.6)
    for i, (pt, lo, hi, lab) in enumerate(mr):
        ok = lo > dl
        a2.errorbar([i], [pt], yerr=[[pt - lo], [hi - pt]], fmt="o", ms=8, capsize=6,
                    color=(C_CEIL if ok else C_CORR), label=f"{lab} ({'PASS' if ok else 'no'})")
    a2.set_xticks([0, 1]); a2.set_xticklabels(["RF_feat −\nRF_svd_raw", "RF_feat −\nRF_svd_white"])
    a2.set_ylabel("paired lens margin  (RF_feature − RF_svd)")
    a2.set_title("LENS GATE: margin CI-lo > δ_lens?", fontsize=9.5)
    a2.legend(fontsize=6.5, loc="best")
    v = "H_sparse CONFIRMED (lens beats BOTH linear controls)" if c["verdict_H_sparse"] else "NOT a feature-lens win"
    fig.suptitle(f"Confirmation K*={c['k_star']}: {v}", fontsize=10.5)
    _footer(a1, f"F4 · flens confirmation · n_eval={conf['n_eval_items']} · "
                f"lens gate = paired CI-lo(RF_feat−RF_svd) > δ_lens for BOTH controls · SHA {sha} · CONFIRMATORY")
    _save(fig, out, "flens_F4_headline")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--discovery-prefix", default="")
    ap.add_argument("--confirm-prefix", default="")
    ap.add_argument("--sha", default="uncommitted")
    args = ap.parse_args()

    import importlib.util
    import sys
    if importlib.util.find_spec("matplotlib") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib"], check=True)

    out = pathlib.Path(args.out or ".")
    out.mkdir(parents=True, exist_ok=True)

    disc = _pull(args.discovery_prefix, out / "_disc.json")
    conf = _pull(args.confirm_prefix, out / "_confirm.json")
    if disc and disc.get("discovery"):
        fig_complete_sweep(disc, out, args.sha)
        fig_sae_fidelity(disc, out, args.sha)
        fig_feature_survival(disc, out, args.sha)
        print("[flens-fig] F1 (complete K-sweep) + F2 (SAE fidelity) + F3 (feature survival) written")
    if conf and conf.get("confirmation"):
        fig_headline(conf, out, args.sha)
        print("[flens-fig] F4 (headline) written")
    print("[flens-fig] done")


if __name__ == "__main__":
    main()
