"""ATLAS figure suite (CPU) — every figure recomputed from `metrics.json` artifacts, never from
prose. House rules (spec §10): claim in the title backed by number + CI + null; a one-line
"how to read"; the null overlaid; provenance footer (input hashes + freeze tag); colourblind-safe
(Okabe–Ito, fixed assignment order); PNG + PDF; the FULL-SWEEP rule — every swept point and every
component curve, never the headline diff alone. No dual axes (the spec's F-B2 "twin axis" is
rendered as stacked shared-x panels).

  ./infra/aws_pg/pg run infra.aws_pg.jobs.atlas_figures --type r7i.2xlarge --no-weights \
      --s3-prefix results/atlas_figs --budget-min 60 --args "--stage postT"
`--stage postT` renders F-T1..F-T4 + F-B1/F-D1/F-N1 (+F-G1/G2); `--stage final` adds
F-B2/F-W1/F-X1/F-C1/F-M1 once those arms have landed.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib

# Okabe–Ito, fixed assignment order (CVD-safe by construction; identity never cycled)
C_BLUE = "#0072B2"      # treatment / primary curve
C_ORANGE = "#E69F00"    # comparator / twin
C_GREEN = "#009E73"     # third series
C_VERM = "#D55E00"      # fourth / emphasis
C_PURPLE = "#CC79A7"    # fifth
C_SKY = "#56B4E9"       # sixth
C_GRAY = "#7F7F7F"      # nulls / references
C_INK = "#1a1a1a"


def _fetch_json(RS, uri, name):
    p = RS.fetch(uri, f"/tmp/atlasfig/{name}")
    return json.loads(p.read_text()), hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _finish(fig, out, name, how_to_read, provenance):
    # reserve a real footer band so captions never collide with axis labels
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    fig.text(0.01, 0.030, f"how to read: {how_to_read}", fontsize=7.5, color="#444444",
             wrap=True)
    fig.text(0.99, 0.006, provenance, fontsize=6.5, color="#888888", ha="right")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    print(f"[figs] wrote {name}", flush=True)


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="postT", choices=["postT", "final"])
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from infra.aws_pg.jobs import _atlas_shared as AS
    from infra.aws_pg.jobs import _regrain_shared as RS
    from infra.pg import atlas_core as AC

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                         "axes.titlesize": 9.5, "figure.facecolor": "white"})

    man = json.loads(AS.FREEZE_MANIFEST.read_text())
    tag0 = man["gate0"]["tag"]

    t_art, t_sha = _fetch_json(RS, f"{AS.S3_BUCKET}/results/atlas_t/out/metrics.json", "t.json")
    d0_art, d0_sha = _fetch_json(RS, f"{AS.ATLAS_D0}/metrics.json", "d0.json")
    prov_t = f"{tag0} · atlas_t@{t_sha} · atlas_d0@{d0_sha}"

    cells = t_art["cells"]

    # ================= F-T1 — the MDE =================
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    DEAD_CAP = 1e3          # dead cells (Δppl → 1e165) drawn at a capped open marker, labeled

    def arm_rows(arm, e, ro, extra=None):
        return [(min(c.get(ro), DEAD_CAP) if ro == "log1p_dppl" else c.get(ro),
                 bool(c.get("saturated_dead")))
                for c in cells if c["family"] == "T1" and c["arm"] == arm and c["eps"] == e
                and c.get(ro) is not None and (extra is None or extra(c))]

    for ax, ro, lab in ((axes[0], "log1p_dppl", "log(1+Δppl)"), (axes[1], "effect", "VQA effect (1−RF)")):
        eps_all = sorted({c["eps"] for c in cells if c["family"] == "T1"})
        series = (("radial", C_BLUE, "o", "radial OUT (+x̂)", lambda c: c.get("sign_dir") == "out"),
                  ("radial", C_SKY, "v", "radial IN (−x̂)", lambda c: c.get("sign_dir") == "in"),
                  ("tangential_coherent", C_ORANGE, "s", "tangential, coherence-matched twin", None),
                  ("tangential_iid", C_GREEN, "^", "tangential iid", None),
                  ("haar_iid", C_GRAY, "x", "Haar iid (matched-energy null)", None))
        for arm, col, mk, name, extra in series:
            xs, ys = [], []
            for e in eps_all:
                rows = arm_rows(arm, e, ro, extra)
                if not rows:
                    continue
                vals = [v for v, _ in rows]
                xs.append(e)
                ys.append(np.mean(vals))
                for v, dead in rows:
                    ax.scatter([e], [v], s=26 if dead else 14, color=col, alpha=0.6, marker=mk,
                               facecolors="none" if dead else col)
            if xs:
                ax.plot(xs, ys, color=col, lw=1.8, marker=mk, ms=5, label=name)
        ax.set_xlabel("per-token amplitude ε (× rms)")
        ax.set_ylabel(lab)
        ax.set_xscale("log")
        if ro == "log1p_dppl":
            ax.set_yscale("symlog", linthresh=0.5)
            ax.axhline(DEAD_CAP, color="#999999", lw=0.8, ls=":")
            ax.text(0.105, DEAD_CAP, "dead cells capped here (open markers)", fontsize=6.5,
                    va="bottom", color="#666666")
        ax.axvspan(0.35, 0.45, color="#000000", alpha=0.05, lw=0)
        ax.text(0.4, ax.get_ylim()[0], "0.4: disclosed,\nnot adjudicated", fontsize=6.5,
                ha="center", va="bottom", color="#666666")
    adj = t_art["rt1"]["ppl"]["adjudication"]
    r = t_art["rt1"]["ppl"]["ratio_by_eps"]
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"F-T1 (MDE): radial vs tangential at matched per-token energy — "
                 f"ratio {min(r.values()):.3f}–{max(r.values()):.3f} → {adj['verdict']}",
                 fontsize=10)
    _finish(fig, out, "F-T1_radial_vs_tangential",
            "each dot is one cell; lines join per-ε means; radial ≪ tangential ⇒ RMSNorm's "
            "radial gauge is load-bearing (T1)", prov_t)
    plt.close(fig)

    # ================= F-T2 — the RoPE ladder =================
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pl = [(c["omega"], c["log1p_dppl"]) for c in cells if c["family"] == "T2" and c["arm"] == "plane"]
    hp = [c["log1p_dppl"] for c in cells if c["family"] == "T2" and c["arm"] == "haar_pair"]
    oms = sorted({w for w, _ in pl})
    means = [np.mean([v for w, v in pl if w == o]) for o in oms]
    ax.scatter([w for w, _ in pl], [v for _, v in pl], s=16, color=C_BLUE, alpha=0.5, label="plane cells")
    ax.plot(oms, means, color=C_BLUE, lw=2)
    if hp:
        ax.axhspan(float(np.min(hp)), float(np.max(hp)), color=C_GRAY, alpha=0.25, lw=0,
                   label="Haar plane-pair control band")
    # zero-parameter derived curve: damage ∝ live (non-gauge) energy ε²(1 − retention²),
    # scaled to the highest-ω plane's mean (one pinned point, no fit)
    T = 271
    live = np.array([1.0 - AC.dirichlet_retention(o, T) ** 2 for o in oms])
    if live[-1] > 0 and means[-1] > 0:
        ax.plot(oms, live / live[-1] * means[-1], color=C_VERM, lw=1.5, ls="--",
                label="derived retention curve (zero-parameter, pinned at max ω)")
    ax.set_xscale("log")
    ax.set_xlabel("rotary plane angular rate ω (rad/token)")
    ax.set_ylabel("log(1+Δppl)")
    st2 = t_art["rt2"]["ppl"]
    ax.set_title(f"F-T2: coherent pre-RoPE K-plane injection — ρ_S(ω, damage) = "
                 f"{st2['rho_freq']:+.3f}, gap CI [{st2['gap_cilo']:+.3f}, {st2['gap_cihi']:+.3f}] "
                 f"→ {st2['adjudication']['verdict']}")
    ax.legend(fontsize=7)
    _finish(fig, out, "F-T2_rope_ladder",
            "the retained (row-mean) post-RoPE component is the gauge direction and is harmless; "
            "damage should track the live energy ε²(1−retention²) if addressing-decoherence-limited",
            prov_t)
    plt.close(fig)

    # ================= F-T3 — the channel triple =================
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    arms3 = [("K_postrope", "K post-RoPE\n(exact gauge)", C_BLUE),
             ("V", "V (transport)", C_ORANGE), ("Q", "Q (addressing)", C_GREEN),
             ("haar_channel_preRoPE", "Haar 256-d\nchannel null", C_GRAY)]
    for i, (arm, lab, col) in enumerate(arms3):
        v = [c["log1p_dppl"] for c in cells if c["family"] == "T3" and c["arm"] == arm]
        ax.bar(i, np.mean(v), width=0.6, color=col, edgecolor="white")
        ax.scatter([i] * len(v), v, s=14, color=C_INK, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([lab for _, lab, _ in arms3], fontsize=7.5)
    ax.set_ylabel("log(1+Δppl)")
    rt3 = t_art["rt3"]
    ax.set_title(f"F-T3: coherent injection by channel — ordering {'>'.join(rt3['ordering'])}, "
                 f"theorem check {'PASS' if rt3['theorem_check_pass'] else 'FAIL'}")
    _finish(fig, out, "F-T3_channel_triple",
            "K-coherent post-RoPE must sit at seed-noise (T2 is a theorem; a violation is an "
            "instrument bug, not a finding); the K/V/Q ordering is the finding", prov_t)
    plt.close(fig)

    # ================= F-T4 — √n_eff =================
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    la = t_art["rt4_legA"]
    ax = axes[0]
    ax.errorbar([la["n_eff_L14"] ** 0.5], [la["ratio"]],
                yerr=[[la["ratio"] - la["ci"][0]], [la["ci"][1] - la["ratio"]]],
                fmt="o", color=C_BLUE, ms=8, capsize=4, label="measured L14 ratio (window bootstrap CI)")
    xs = np.linspace(1.0, 2.4, 50)
    ax.plot(xs, xs, color=C_GREEN, lw=1.5, ls="-", label="law: ratio = √n_eff")
    ax.axhline(la["pred_const"], color=C_ORANGE, lw=1.5, ls="--",
               label=f"site constant (L0 sealed ratio {la['pred_const']:.2f})")
    ax.set_xlabel("√n_eff (site)")
    ax.set_ylabel("coherent / iid damage ratio")
    ax.set_title(f"leg A: {la['adjudication']['verdict']}")
    ax.legend(fontsize=7)
    ax = axes[1]
    lb = t_art.get("rt4_legB")
    if lb:
        ax.bar(range(3), lb["tercile_gaps"], width=0.6, color=C_BLUE)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"tercile {i+1}\nn_eff≈{v:.2f}" for i, v in
                            enumerate(lb["tercile_neff"])], fontsize=7.5)
        ax.set_ylabel("per-token nll gap (shared − per-token)")
        ax.set_title(f"leg B (descriptive terciles): ρ = {lb['rho']:+.3f}, perm p = {lb['perm_p']:.4f} "
                     f"→ {lb['verdict']} (AM-2)")
    fig.suptitle("F-T4: does the coherence transport factor move with the site's own attention pooling?",
                 fontsize=10)
    _finish(fig, out, "F-T4_sqrt_neff",
            "left: the measured L14 ratio against BOTH frozen predictions (law line vs L0 constant); "
            "right: within-site ordering, adjudicated on the per-window statistic", prov_t)
    plt.close(fig)

    # ================= F-B1 — the banked-draw regression (zero forwards) =================
    rb1 = d0_art["rb1"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    dr = rb1["draws"]
    for ax, key, xl in ((axes[0], "v1_overlap_pos", "⟨δ̄, v₁⟩₊"),
                        (axes[1], "e1_overlap_abs", "|⟨δ̄, e₁⟩|")):
        for half, col, mk in (("discovery", C_BLUE, "o"), ("sealed", C_ORANGE, "s")):
            xs = [r[key] for r in dr if r["half"] == half]
            ys = [r["log1p_dppl"] for r in dr if r["half"] == half]
            ax.scatter(xs, ys, s=40, color=col, marker=mk, label=half)
        ax.set_xlabel(xl)
    adjb = rb1["adjudication"]
    axes[0].set_ylabel("log(1+Δppl)")
    axes[0].set_title(f"τ_v1 = {adjb['tau_v1']:+.3f} (exact p = {adjb['p_v1']:.4f})")
    axes[1].set_title(f"τ_e1 = {adjb['tau_e1']:+.3f} · τ_norm = {adjb['tau_norm']:+.3f}")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"F-B1: the 9 banked coherent draws, regenerated bit-exactly (AM-3) — "
                 f"{adjb['verdict']}", fontsize=10)
    _finish(fig, out, "F-B1_banked_draws",
            "same 9 damages on both panels; if the left panel orders them and the right does not, "
            "Arm K's 110× spread and Arm A's asymmetry are ONE object (T5a)", prov_t)
    plt.close(fig)

    # ================= F-N1 — max-ent prediction vs house sampler =================
    rn = d0_art["rn"]
    ver = d0_art["bingham_verify"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, name, blk in ((axes[0], "L0 (verification)", ver["L0"]),
                          (axes[1], "L14 (verification)", ver["L14"]),
                          (axes[2], "L8 (HELD OUT — predicted before sampled)",
                           {"model": rn["prediction"], "house": rn["sampled_L8"]})):
        mc = blk["model"]["curve"]
        xs = [r["E_over_Emax"] for r in mc]
        ax.plot(xs, [r["energy_mean_distance"] for r in mc],
                color=C_VERM, lw=2, ls="-", label="max-ent (Bingham) prediction")
        for r in mc:
            missed = r.get("target_missed")
            ax.scatter([r["E_over_Emax"]], [r["energy_mean_distance"]], s=30, marker="o",
                       color=C_VERM, facecolors="none" if missed else C_VERM, zorder=3)
        if any(r.get("target_missed") for r in mc):
            ax.text(0.98, 0.62, "open circles: tilt could not\ncertify the rung (excluded\nfrom the gap, counted)",
                    fontsize=6.2, ha="right", transform=ax.transAxes, color="#8a4a00")
        ax.plot(xs, [r["energy_mean_distance"] for r in blk["house"]["curve"]],
                color=C_BLUE, lw=2, marker="s", ms=4, label="house level-set sampler")
        ax.set_xlabel("E / E_max (rank 25)")
        ax.set_title(name, fontsize=9)
    # constant baseline on the L8 panel (the null's null)
    const = d0_art["rn"].get("constant_baseline") or []
    if const:
        axes[2].plot([c["E_over_Emax"] for c in const],
                     [c["energy_mean_distance"] for c in const],
                     color=C_GRAY, lw=1.5, ls="--", label="constant baseline (L0/L14 site-average)")
    axes[0].set_ylabel("energy dispersion (1 − alignment)")
    axes[2].legend(fontsize=7)
    radj = rn["adjudication"]
    fig.suptitle(f"F-N1: level-set dispersion from the spectrum alone — max gap "
                 f"{max(rn['abs_gaps']):.3f} vs constant {max(rn['constant_gaps']):.3f} → "
                 f"{radj['verdict']}", fontsize=10)
    _finish(fig, out, "F-N1_bingham_curves",
            "the L8 curve was PREDICTED from its spectrum before being sampled (§9.3); "
            "H_maxent-sufficient needs gap ≤ 0.10 AND beating the constant baseline", prov_t)
    plt.close(fig)

    # ================= F-D1 — d_eff calibration + the family map =================
    cal = d0_art["deff_calibration"]
    sweep = d0_art["deff_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    prs = (1, 4, 32, 256)
    for n, col in ((3, C_ORANGE), (8, C_BLUE)):
        med = [cal[f"pr{p}_n{n}"]["median"] for p in prs]
        lo = [cal[f"pr{p}_n{n}"]["q05"] for p in prs]
        hi = [cal[f"pr{p}_n{n}"]["q95"] for p in prs]
        ax.plot(prs, med, color=col, lw=2, marker="o", label=f"n = {n} draws (median)")
        ax.fill_between(prs, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(prs, prs, color=C_GRAY, ls="--", lw=1.2, label="truth (d_eff = PR)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("true participation ratio")
    ax.set_ylabel("recovered d_eff = 2/CV²")
    ax.set_title("synthetic calibration (q05–q95 band)")
    ax.legend(fontsize=7)
    ax = axes[1]
    names = [s["family"] for s in sweep]
    vals = [min(s["d_eff"], 1e4) for s in sweep]
    ax.barh(range(len(sweep)), vals, color=C_BLUE, height=0.6)
    ax.set_yticks(range(len(sweep)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("d_eff (ordinal at n=3 — calibrated interval in metrics)")
    ax.set_title("the banked-family susceptibility-dimension map")
    fig.suptitle("F-D1: dispersion dimension — CV²(vᵀMv) = 2/PR(M), zero forwards", fontsize=10)
    _finish(fig, out, "F-D1_deff",
            "left: how wide n=3 readings are (the point of R-D); right: every reconstructible "
            "banked multi-draw family placed on the dimension axis", prov_t)
    plt.close(fig)

    # ================= F-G1/F-G2 — explanatory geometry (3D) =================
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(10.5, 4.2))
    ax = fig.add_subplot(121, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color="#dddddd", alpha=0.25, linewidth=0)
    x0 = np.array([0.55, 0.6, 0.58]); x0 = x0 / np.linalg.norm(x0)
    ax.quiver(0, 0, 0, *x0, color=C_INK, lw=2, arrow_length_ratio=0.08)
    ax.quiver(*x0, *(0.45 * x0), color=C_BLUE, lw=2.5, arrow_length_ratio=0.15)
    t1 = np.cross(x0, [0, 0, 1.0]); t1 /= np.linalg.norm(t1)
    ax.quiver(*x0, *(0.45 * t1), color=C_ORANGE, lw=2.5, arrow_length_ratio=0.15)
    th = np.linspace(0, 2 * np.pi, 100)
    t2 = np.cross(x0, t1)
    circ = np.outer(np.cos(th), t1) + np.outer(np.sin(th), t2)
    ax.plot(*(x0[:, None] + 0.45 * circ.T), color=C_ORANGE, alpha=0.4, lw=1)
    for s_ in np.linspace(0.7, 1.5, 5):
        ax.scatter(*(s_ * x0), color=C_BLUE, s=8, alpha=0.5)
    ax.set_title("F-G1: the T1 gauge — RMSNorm reads direction, not radius\n"
                 "radial (blue) moves along the annihilated ray; tangential (orange) moves the read",
                 fontsize=8.5)
    ax.set_axis_off()
    rr = t_art["rt1"]["ppl"]["ratio_by_eps"]
    ax.text2D(0.02, 0.02, f"measured radial/tangential damage ratio @ε=0.2: {rr.get(0.2, rr.get('0.2', float('nan'))):.3f}",
              transform=ax.transAxes, fontsize=7.5)
    ax2 = fig.add_subplot(122, projection="3d")
    T = 60
    pos = np.arange(T)
    for om_, col, lab in ((1.0, C_VERM, "plane 0 (ω=1.0): decoheres in a few tokens"),
                          (0.01, C_BLUE, "plane 64 (ω=0.01): coherent across the window")):
        ang = pos * om_
        ax2.plot(pos, np.cos(ang), np.sin(ang), color=col, lw=1.8, label=lab)
    ax2.set_xlabel("token position", fontsize=7.5)
    ax2.set_title("F-G2: what RoPE does to a coherent pre-RoPE key shift —\n"
                  "the position-helix of the injected vector in one rotary plane", fontsize=8.5)
    ax2.legend(fontsize=6.5, loc="upper right")
    _finish(fig, out, "F-G1_G2_geometry",
            "left: geometry of the exact radial gauge (T1); right: geometry of the RoPE ladder "
            "(T2) — the retained row-mean of the helix is the harmless gauge component", prov_t)
    plt.close(fig)

    if a.stage == "final":
        b_art, b_sha = _fetch_json(RS, f"{AS.S3_BUCKET}/results/atlas_b/out/metrics.json", "b.json")
        w_art, w_sha = _fetch_json(RS, f"{AS.S3_BUCKET}/results/atlas_w/out/metrics.json", "w.json")
        x_art, x_sha = _fetch_json(RS, f"{AS.S3_BUCKET}/results/atlas_x/out/metrics.json", "x.json")
        m_art, m_sha = _fetch_json(RS, f"{AS.S3_BUCKET}/results/atlas_m_sealed/out/metrics.json", "m.json")
        wb_art, _ = _fetch_json(RS, AS.BISPEC_W, "bw.json")
        prov_f = f"{tag0} · b@{b_sha} · w@{w_sha} · x@{x_sha} · m@{m_sha}"

        # ---- F-B2: the clamp quartet (stacked panels, no dual axis) ----
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        dose = [(c["alpha"], c["effect"]) for c in b_art["cells"] if c["family"] == "B_dose"]
        dose.sort()
        ax.plot([d[0] for d in dose], [d[1] for d in dose], color=C_BLUE, lw=2, marker="o",
                label="unclamped traversal (±ασ₁ along v₁)")
        cl = next(c for c in b_art["cells"] if c.get("arm") == "clamped")
        ax.scatter([3.0], [cl["effect"]], color=C_VERM, s=70, marker="D", zorder=4,
                   label="flipped population clamped")
        rndv = [c["effect"] for c in b_art["cells"] if c.get("arm") == "random_clamp"]
        ax.scatter([3.0] * len(rndv), rndv, color=C_GRAY, s=30, marker="s",
                   label="random-clamp control (never-flipping units)")
        idv = next(c for c in b_art["cells"] if c.get("arm") == "identity_alpha0")
        ax.scatter([0.0], [idv["effect"]], color=C_GREEN, s=40, marker="^",
                   label="identity check (clamp, α=0)")
        rb2 = b_art["rb2"]
        ax.set_xlabel("traversal amplitude α (× σ₁)")
        ax.set_ylabel("VQA effect (1−RF)")
        ax.set_title(f"F-B2: the dormant-gate clamp — recovered {rb2['recovered_frac']:.2f} of the "
                     f"collapse (random {np.mean(rb2['random_clamp_recovered']):.2f}) → "
                     f"{rb2['adjudication']['verdict']}")
        ax.legend(fontsize=7)
        _finish(fig, out, "F-B2_clamp_quartet",
                "if capping the +αv₁-flipped GeGLU gates recovers the collapse while the "
                "matched random clamp does not, the boundary's substrate is the gate population "
                f"(census: {b_art['census']['flipped_by_plus3sigma']})", prov_f)
        plt.close(fig)

        # ---- F-W1: the write-profile overlay, 5×5 small multiples ----
        import torch as _t
        rho_bank = _t.load(RS.fetch(f"{AS.S3_BUCKET}/results/atlas_w/out/rho_bank.pt",
                                    "/tmp/atlasfig/rho_bank.pt"), map_location="cpu",
                           weights_only=False)
        layers_w = w_art["layers"]
        types_w = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
        fig, axes = plt.subplots(len(layers_w), len(types_w), figsize=(13, 10),
                                 sharex=True, sharey=True)
        for i, L in enumerate(layers_w):
            for j, t in enumerate(types_w):
                ax = axes[i][j]
                key = f"L{L}/{t}"
                prof = wb_art["profiles"][key]
                x5 = range(5)
                ax.plot(x5, prof["profile"], color=C_INK, lw=2, marker="o", ms=3, label="Ω (measured)")
                ax.plot(x5, prof["ntk_reference"], color=C_GREEN, lw=1.4, ls="--", label="λ (NTK)")
                ax.plot(x5, prof["random_reference"], color=C_GRAY, lw=1.2, ls=":", label="Haar ΔW")
                rb = rho_bank[f"ppl_proxy/{key}"]
                lr = (rb["lam"] * rb["rho"])
                from infra.pg.atlas_core import band_aggregate
                bb5 = None
                # recompute 5-band aggregate of λρ on the banked slices implied by profile length
                # (the atlas_w metrics carries the Spearman fit; this panel shows the curve)
                n_modes = len(rb["lam"])
                # equal-energy 5-band split of the stored spectrum (matches eig_bands convention)
                cum = np.cumsum(rb["lam"].numpy()) / float(rb["lam"].sum())
                edges, lo = [], 0
                for e_ in (0.5, 0.8, 0.95, 0.99, 1.0):
                    hi_ = int(np.searchsorted(cum, e_)) + 1
                    hi_ = min(max(hi_, lo + 1), n_modes)
                    edges.append((lo, hi_)); lo = hi_
                lrp = band_aggregate(lr, edges)
                ax.plot(x5, lrp, color=C_VERM, lw=1.6, marker="s", ms=3, label="λρ (T6 prediction)")
                if i == 0:
                    ax.set_title(t, fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"L{L}", fontsize=8)
                ax.set_yscale("log")
        axes[0][0].legend(fontsize=6, loc="lower left")
        rw = w_art["rw"]
        fig.suptitle(f"F-W1: where fine-tuning writes vs what the gradient sees — Δfit(ppl proxy) "
                     f"= {rw['ppl_proxy']['dfit']:+.3f} → {rw['ppl_proxy']['adjudication']['verdict']}, "
                     f"Δfit(vqa proxy) = {rw['vqa_proxy']['dfit']:+.3f}", fontsize=10)
        _finish(fig, out, "F-W1_write_profiles",
                "per (layer × matrix): the banked Ω profile against three references — NTK (λ), "
                "Haar ΔW, and the T6 visible-variance curve (λρ)", prov_f)
        plt.close(fig)

        # ---- F-X1: transport identity gaps ----
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        rx = x_art["rx"]
        bars = [("frame carry\n(orth(D_γU))", x_art["frame_carry"]["gap_frame_live"], C_ORANGE),
                ("conjugated carry\n(J-least-squares)", x_art["conjugated"]["gap_conjugated"], C_BLUE)]
        for i, (lab, v, col) in enumerate(bars):
            ax.bar(i, v, width=0.55, color=col)
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhspan(0, 0.10, color=C_GREEN, alpha=0.12, lw=0)
        ax.text(1.4, 0.05, "identity band (≤0.10)", fontsize=7, color=C_GREEN)
        ax.set_xticks(range(2))
        ax.set_xticklabels([b[0] for b in bars], fontsize=8)
        ax.set_ylabel("|native − transported| position-corr gap")
        ax.set_title(f"F-X1: closure = {rx['closure']:.2f} → {rx['verdict']}")
        _finish(fig, out, "F-X1_transport",
                "the V1r frame carry left a 0.355 gap (recomputed live); conjugation through the "
                "RMSNorm Jacobian closes the fraction shown (T8)", prov_f)
        plt.close(fig)

        # ---- F-C1: the A9 mechanism triptych ----
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        cr = b_art["c_reading"]
        labs = [("true", "true error e(x)"), ("permuted", "permuted\n(across tokens)"),
                ("signflip", "sign-flipped −e(x)"), ("quantile_shuffle", "shuffled within\n⟨x⟩-quantile")]
        for i, (k, lab) in enumerate(labs):
            ax.bar(i, cr[k], width=0.6, color=[C_BLUE, C_ORANGE, C_VERM, C_PURPLE][i])
            ax.text(i, cr[k], f"{cr[k]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels([lab for _, lab in labs], fontsize=7.5)
        ax.set_ylabel("log(1+Δppl)")
        ax.set_title("F-C1: which property of quantization error carries A9's alignment axis")
        _finish(fig, out, "F-C1_mechanism_triptych",
                "C-1 ≈ permuted ⇒ lattice landing; C-1 ≈ true ⇒ error–token dependence; "
                "C-2 interpolating ⇒ dependence-strength is the axis", prov_f)
        plt.close(fig)

        # ---- F-M1: the headline sealed parity plot ----
        sc = m_art["scored"]
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True, sharex=True)
        y = [s["measured_log1p_dppl"] for s in sc]
        mk_by_sign = {"add": "o", "delete": "s"}
        for ax, mdl, col in ((axes[0], "M0", C_GRAY), (axes[1], "M2", C_BLUE)):
            p = [s["pred"][f"{mdl}_ppl_log1p_dppl"] for s in sc]
            for s_, pi, yi in zip(sc, p, y):
                ax.scatter(pi, yi, s=46, color=col, marker=mk_by_sign[s_["cell"]["sign"]],
                           edgecolor="white", linewidth=0.6)
            lim = [min(min(p), min(y)) - 0.3, max(max(p), max(y)) + 0.3]
            ax.plot(lim, lim, color="#bbbbbb", lw=1, ls="--")
            from infra.pg.atlas_fit import spearman
            ax.set_title(f"{mdl}: ρ_S = {spearman(p, y):+.3f}")
            ax.set_xlabel(f"{mdl} predicted log(1+Δppl)")
        axes[0].set_ylabel("measured log(1+Δppl), SEALED cells")
        rm = m_art["rm"]["ppl"]
        fig.suptitle(f"F-M1 (headline): sealed forward-prediction — Δρ = {rm['drho']:+.3f} "
                     f"CI [{rm['ci'][0]:+.3f}, {rm['ci'][1]:+.3f}] → {rm['adjudication']['verdict']} "
                     f"(circles add · squares delete; {m_art['n_void']} VOIDs replaced)", fontsize=10)
        _finish(fig, out, "F-M1_sealed_parity",
                "20 never-run cells, CHOSEN ADVERSARIALLY to maximize M2−M0 disagreement, predicted "
                "before any sealed forward; the diagonal is perfect prediction", prov_f)
        plt.close(fig)

    print("ATLAS_FIGS_OK", a.stage)
    (out / "figs_done.json").write_text(json.dumps({"stage": a.stage, "green": True}))


if __name__ == "__main__":
    main()
