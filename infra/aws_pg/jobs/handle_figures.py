"""HANDLE figures (CPU) — the §10 figure suite (15 figures). Reads every arm's banked metrics
from S3 and renders each with a plain-language claim title backed by number + CI + null, a
how-to-read line, the null overlaid, the full-sweep rule (every rung + every component curve),
colourblind-safe colours, a provenance footer, PNG + PDF. Robust to a missing arm OR a missing
field (that figure is skipped + logged). Regenerated post-critic.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.handle_figures --type c7i.2xlarge \
      --no-weights --s3-prefix results/handle_figures --budget-min 40
"""
import argparse
import json
import math
import os
import pathlib

CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "yellow": "#F0E442", "sky": "#56B4E9", "grey": "#999999"}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from infra.aws_pg.jobs import _handle_shared as HS

    rep = {"object": "handle_figures_v1", "gpu_seconds": {"total": 0.0, "note": "CPU job"},
           "figures": []}

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=2, default=str))

    def load(prefix, name="metrics.json"):
        import subprocess
        p = out / f"_{prefix.replace('/', '_')}_{name}"
        r = subprocess.run(["aws", "s3", "cp", f"{HS.S3_BUCKET}/results/{prefix}/out/{name}",
                            str(p), "--region", "us-east-1", "--only-show-errors"],
                           capture_output=True, text=True)
        return json.loads(p.read_text()) if (r.returncode == 0 and p.exists()) else None

    def footer(ax, fid, src):
        ax.text(0.5, -0.18, f"{fid} · HANDLE · src: {src} · handle_figures_v1",
                transform=ax.transAxes, ha="center", va="top", fontsize=6, color=CB["grey"])

    def save(fig, fid):
        fig.savefig(out / "figs" / f"{fid}.png", dpi=150, bbox_inches="tight")
        fig.savefig(out / "figs" / f"{fid}.pdf", bbox_inches="tight")
        plt.close(fig)
        rep["figures"].append(fid)
        print(f"[handle-fig] {fid} rendered", flush=True)
        _w()

    def skip(fid, why):
        rep["figures"].append(f"{fid}:SKIPPED({why})")
        print(f"[handle-fig] {fid} SKIPPED: {why}", flush=True)
        _w()

    sealed = load("handle_h_sealed") or load("handle_h_adjudicate")
    disc = load("handle_h_disc")
    tm = load("handle_t")
    qm = load("handle_q")
    nm_c = load("handle_n_carrier")
    nm_s = load("handle_n_spikes")
    bm = load("handle_b")
    xm = load("handle_x")

    def rungs_of(cell):
        return sorted((int(k) for k in cell.get("rungs", {})), key=int)

    # ================= F-0 program schema =================
    try:
        fig, ax = plt.subplots(figsize=(9, 3.5)); ax.axis("off")
        arms = ["Gate-0", "H (MDE ladder)", "T (modules)", "Q (benignity)", "N (naming)",
                "B (witness)", "X (PG2 repl.)"]
        for i, arm in enumerate(arms):
            ax.add_patch(plt.Rectangle((i * 1.25, 0), 1.1, 1, fc=CB["sky"] if i < 2 else CB["grey"],
                                       ec="k", alpha=0.5))
            ax.text(i * 1.25 + 0.55, 0.5, arm, ha="center", va="center", fontsize=8)
            if i:
                ax.annotate("", (i * 1.25, 0.5), (i * 1.25 - 0.15, 0.5),
                            arrowprops=dict(arrowstyle="->"))
        ax.set_xlim(-0.3, len(arms) * 1.25); ax.set_ylim(-0.4, 1.4)
        ax.set_title("F-0 · HANDLE program: from map to handle — the MDE (Gate-0→H) then four hunts (T,Q,N,B) + a second-model replication (X)",
                     fontsize=9)
        footer(ax, "F-0", "design")
        save(fig, "F-0")
    except Exception as e:
        skip("F-0", str(e)[:60])

    # ================= F-H1 headline: per-FT gain-vs-k ladders (small multiples) =================
    if sealed and sealed.get("cells"):
        cells = {s: c for s, c in sealed["cells"].items() if c.get("range_ok") and c.get("rungs")}
        khalf = sealed.get("rh", {}).get("k_half", {})
        n = len(cells)
        if n:
            ncol = 4; nrow = math.ceil(n / ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow), squeeze=False)
            for idx, (slug, c) in enumerate(sorted(cells.items())):
                ax = axes[idx // ncol][idx % ncol]
                ks = rungs_of(c)
                core = [c["rungs"][str(k)]["core"]["gain_frac"] for k in ks]
                band = [c["rungs"][str(k)]["band_max"] for k in ks]
                bstiff = [c["rungs"][str(k)]["band_stiff_max"] for k in ks]
                ax.plot(ks, core, "-o", color=CB["blue"], ms=3, label="core_k")
                ax.plot(ks, band, "--", color=CB["grey"], lw=1, label="band_k")
                ax.plot(ks, bstiff, ":", color=CB["orange"], lw=1.2, label="band_k^stiff")
                kh = khalf.get(slug)
                if kh:
                    ax.axvline(kh, color=CB["red"], lw=0.8, alpha=0.7)
                ax.set_xscale("log"); ax.set_title(slug, fontsize=7)
                ax.tick_params(labelsize=6); ax.set_ylim(-0.1, 1.15)
                if idx == 0:
                    ax.legend(fontsize=5, loc="lower right")
            for j in range(n, nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            mkh = sealed.get("rh", {}).get("median_k_half", {})
            mtxt = mkh.get("median", "?") if isinstance(mkh, dict) else "?"
            fig.suptitle(f"F-H1 · The compact-core ladder, FT by FT (median k_half={mtxt}) — "
                         f"core_k vs both matched-random bands; red = k_half", fontsize=9)
            fig.text(0.5, 0.005, "How to read: each panel is one sealed FT; x=rank k (log), "
                     "y=gain_frac; core above BOTH bands ⇒ the ordering carries.", ha="center",
                     fontsize=7, color=CB["grey"])
            save(fig, "F-H1")
        else:
            skip("F-H1", "no range-passing sealed FTs")
    else:
        skip("F-H1", "handle_h_sealed absent")

    # ================= F-H2 median curves + hypothesis regions =================
    if sealed and sealed.get("rh", {}).get("per_rung"):
        pr = sealed["rh"]["per_rung"]
        ks = sorted((int(k) for k in pr), key=int)
        cells = {s: c for s, c in sealed["cells"].items() if c.get("range_ok") and c.get("rungs")}
        med_core = [float(np.median([c["rungs"][str(k)]["core"]["gain_frac"]
                    for c in cells.values()])) for k in ks]
        med_band = [float(np.median([c["rungs"][str(k)]["band_max"] for c in cells.values()])) for k in ks]
        med_bs = [float(np.median([c["rungs"][str(k)]["band_stiff_max"] for c in cells.values()])) for k in ks]
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.axvspan(ks[0], 50, color=CB["green"], alpha=0.08, label="H_compact zone (k_half≤50)")
        ax.axvspan(150, ks[-1], color=CB["orange"], alpha=0.08, label="H_ordered-broad zone")
        ax.plot(ks, med_core, "-o", color=CB["blue"], label="median core_k")
        ax.plot(ks, med_band, "--", color=CB["grey"], label="median band_k")
        ax.plot(ks, med_bs, ":", color=CB["orange"], lw=2, label="median band_k^stiff")
        ax.set_xscale("log"); ax.set_xlabel("rank k"); ax.set_ylabel("median gain_frac")
        v = sealed["rh"].get("verdict", "?")
        ax.set_title(f"F-H2 · Which world: {v} — median gain-vs-k with both bands", fontsize=9)
        ax.legend(fontsize=7)
        footer(ax, "F-H2", "handle_h_sealed.rh")
        save(fig, "F-H2")
    else:
        skip("F-H2", "handle_h_sealed rh.per_rung absent")

    # ================= F-H3 excess_k forest plot =================
    if sealed and sealed.get("rh", {}).get("per_rung"):
        pr = sealed["rh"]["per_rung"]
        ks = sorted((int(k) for k in pr), key=int)
        fig, ax = plt.subplots(figsize=(7, 5))
        for j, (key, col, off) in enumerate((("excess_band", CB["grey"], -0.12),
                                             ("excess_stiff", CB["orange"], 0.12))):
            for i, k in enumerate(ks):
                mc = pr[str(k)][key]
                if mc.get("void"):
                    continue
                lo, hi = mc["ci95"]; med = mc["median"]
                ax.plot([lo, hi], [i + off, i + off], color=col, lw=2)
                ax.plot(med, i + off, "o", color=col, ms=4,
                        label=key if i == 0 else None)
        ax.axvline(0, color=CB["red"], lw=0.8)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks)
        ax.set_xlabel("median excess (gain_frac − band)  [CI-lo>0 ⇒ cleared]")
        ax.set_ylabel("rank k")
        ax.set_title("F-H3 · The null-referenced statistic: excess_k vs both bands (95% CI)", fontsize=9)
        ax.legend(fontsize=7)
        footer(ax, "F-H3", "handle_h_sealed.rh.per_rung")
        save(fig, "F-H3")
    else:
        skip("F-H3", "handle_h_sealed rh.per_rung absent")

    # ================= F-T1 module matrix heatmap =================
    if tm and tm.get("module_matrix_np"):
        M = np.array(tm["module_matrix_np"])
        floor = tm.get("haar_core_floor", {})
        fr = np.array([[list(floor.values())[j] if j < len(floor) else np.nan
                        for j in range(M.shape[1])]])
        full = np.vstack([M, fr])
        fig, ax = plt.subplots(figsize=(6.5, 6))
        im = ax.imshow(full, cmap="viridis", aspect="auto")
        for i in range(M.shape[0]):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=CB["red"], lw=2))
        ax.set_yticks(range(full.shape[0]))
        ax.set_yticklabels([f"core_{i}" for i in range(M.shape[0])] + ["Haar-core floor"], fontsize=7)
        ax.set_xlabel("evaluated on task_j"); plt.colorbar(im, ax=ax, label="gain")
        dd = tm.get("rt", {}).get("diag_dominance", "?")
        ax.set_title(f"F-T1 · Module matrix M (diag={dd}/30) — core_i × task_j; red=diagonal", fontsize=9)
        footer(ax, "F-T1", "handle_t")
        save(fig, "F-T1")
    else:
        skip("F-T1", "handle_t module_matrix absent")

    # ================= F-T2 deletion-selectivity dumbbells =================
    if tm and tm.get("deletion_rows"):
        rows = tm["deletion_rows"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, r in enumerate(rows):
            other = max(r["ppl_drop"], r["nontarget_drop"])
            ax.plot([other, r["own_drop"]], [i, i], "-", color=CB["grey"])
            ax.plot(r["own_drop"], i, "o", color=CB["blue"], ms=6)
            ax.plot(other, i, "s", color=CB["orange"], ms=5)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["ft"] for r in rows], fontsize=7)
        ax.set_xlabel("drop from deleting the core")
        ax.plot([], [], "o", color=CB["blue"], label="own-task drop")
        ax.plot([], [], "s", color=CB["orange"], label="max(ppl, non-target) drop")
        ax.set_title("F-T2 · Deletion selectivity — surgical iff own-task ≥ 3× the rest", fontsize=9)
        ax.legend(fontsize=7)
        footer(ax, "F-T2", "handle_t.deletion_rows")
        save(fig, "F-T2")
    else:
        skip("F-T2", "handle_t deletion_rows absent")

    # ================= F-T3 core census =================
    if tm and tm.get("census"):
        c = tm["census"]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        labels = ["cross-FT", "vs U_wr", "vs U_unt"]
        vals = [c.get("cross_overlap_med", np.nan), c.get("core_uwr_overlap_med", np.nan),
                c.get("core_uunt_overlap_med", np.nan)]
        ax.bar(labels, vals, color=[CB["blue"], CB["green"], CB["purple"]], alpha=0.8)
        ax.axhline(c.get("haar_p95", np.nan), color=CB["red"], ls="--", label="Haar p95 (within-stiff)")
        ax.set_ylabel("subspace overlap")
        ax.set_title("F-T3 · Are the cores new objects? overlap vs {each other, U_wr, U_unt} vs Haar", fontsize=9)
        ax.legend(fontsize=7)
        footer(ax, "F-T3", "handle_t.census")
        save(fig, "F-T3")
    else:
        skip("F-T3", "handle_t census absent")

    # ================= F-Q1 property race (4 panels) =================
    if qm and qm.get("race", {}).get("per_property") and qm.get("fam_damage") and qm.get("fam_props"):
        pp = qm["race"]["per_property"]
        fams = list(qm["fam_damage"])
        dmg = [qm["fam_damage"][f] for f in fams]
        props = list(pp)
        fig, axes = plt.subplots(1, len(props), figsize=(3.2 * len(props), 3.4), squeeze=False)
        for j, p in enumerate(props):
            ax = axes[0][j]
            sc = [qm["fam_props"][f][p] for f in fams]
            ax.scatter(sc, dmg, s=18, color=CB["blue"])
            ax.set_xlabel(p, fontsize=7); ax.set_ylabel("log(1+Δppl)" if j == 0 else "", fontsize=7)
            r = pp[p]
            ax.set_title(f"ρ={r['rho']:+.2f}\nshuf_p={r['shuffle_p']:.2f} {'✓' if r.get('bh_pass') else ''}",
                         fontsize=7)
            ax.tick_params(labelsize=6)
        v = qm.get("rq", {}).get("verdict", "?")
        fig.suptitle(f"F-Q1 · Which ruler ranks benignity: {v} — score vs damage per family", fontsize=9)
        save(fig, "F-Q1")
    else:
        skip("F-Q1", "handle_q race absent")

    # ================= F-Q2 δ-ladder dose-response =================
    if qm and qm.get("delta_damage"):
        dd = qm["delta_damage"]; td = qm.get("twin_damage", {})
        ks = sorted(float(k) for k in dd)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(ks, [dd[str(k) if str(k) in dd else k] for k in ks], "-o", color=CB["blue"],
                label="δ-mixture")
        if td:
            ax.plot(ks, [td.get(str(k), td.get(k, np.nan)) for k in ks], "--s", color=CB["orange"],
                    label="rotated twin")
        tau = qm.get("rq", {}).get("delta_tau", "?")
        ax.set_xlabel("dependence fraction δ"); ax.set_ylabel("log(1+Δppl)")
        ax.set_title(f"F-Q2 · The causal dose-response (Kendall τ={tau}) — more dependence ⇒ less damage?", fontsize=9)
        ax.legend(fontsize=7)
        footer(ax, "F-Q2", "handle_q.delta_damage")
        save(fig, "F-Q2")
    else:
        skip("F-Q2", "handle_q delta_damage absent")

    # ================= F-N1 carrier regression (5 candidate panels) =================
    if nm_c and nm_c.get("candidate_stats") and nm_c.get("carrier_rows"):
        cs = nm_c["candidate_stats"]; rows = nm_c["carrier_rows"]
        names = list(cs)
        fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 3.2), squeeze=False)
        dmg = [r.get("damage", r.get("dppl", np.nan)) for r in rows]
        for j, nname in enumerate(names):
            ax = axes[0][j]
            sc = [r.get(nname, r.get("features", {}).get(nname, np.nan)) for r in rows]
            ax.scatter(sc, dmg, s=20, color=CB["purple"])
            ax.set_title(f"{nname}\nτ={cs[nname].get('tau', '?')}, p={cs[nname].get('p', '?')}", fontsize=6)
            ax.tick_params(labelsize=6)
        v = nm_c.get("rna", {}).get("verdict", "?")
        fig.suptitle(f"F-N1 · Is the carrier named? {v} — 5 candidates on the 9 draws", fontsize=9)
        save(fig, "F-N1")
    else:
        skip("F-N1", "handle_n_carrier absent")

    # ================= F-N2 L0-spike census + causal ablation =================
    if nm_s and nm_s.get("rnb", {}).get("overlaps"):
        cen = nm_s["rnb"]["overlaps"]                 # {elem: {max_overlap, haar_p95, ratio}}
        elems = list(cen)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ratios = [float(cen[e]["ratio"]) for e in elems]
        ax.bar(elems, ratios, color=CB["sky"], alpha=0.8)
        ax.axhline(3.0, color=CB["red"], ls="--", label="3× Haar identification bar")
        cc = nm_s.get("causal_cell") or {}
        drop = cc.get("rayleigh_drop") if isinstance(cc, dict) else None
        v = nm_s.get("rnb", {}).get("verdict", "?")
        ax.set_ylabel("max-overlap / Haar p95"); ax.tick_params(axis="x", rotation=30, labelsize=6)
        ax.set_title(f"F-N2 · What are the L0 spikes? {v}" + (f" (Rayleigh drop {drop})" if drop else ""),
                     fontsize=9)
        ax.legend(fontsize=7)
        footer(ax, "F-N2", "handle_n_spikes.census")
        save(fig, "F-N2")
    else:
        skip("F-N2", "handle_n_spikes census absent")

    # ================= F-B1 witness geometry =================
    if bm and bm.get("geometry"):
        g = bm["geometry"]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        mm = g.get("mean_min_angle"); hm = g.get("haar_mean_min_angle")
        ax.bar(["witness set", "Haar band"], [mm, hm], color=[CB["blue"], CB["grey"]], alpha=0.8)
        de = bm.get("d_eff_detail", {}).get("witnesses_only", {}).get("d_eff", "?")
        v = bm.get("rb", {}).get("verdict", "?")
        ax.set_ylabel("mean smallest principal angle (rad)")
        ax.set_title(f"F-B1 · Is the boundary an object? {v} (d_eff={de}) — angles vs Haar", fontsize=9)
        footer(ax, "F-B1", "handle_b.geometry")
        save(fig, "F-B1")
    else:
        skip("F-B1", "handle_b geometry absent")

    # ================= F-B2 install-tracking scatter =================
    if bm and bm.get("install_tracking", {}).get("probes"):
        it = bm["install_tracking"]
        pts = it["probes"]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        xs = [float(p["alignment"]) for p in pts]
        ys = [float(p["install_success"]) for p in pts]
        ax.scatter(xs, ys, s=30, color=CB["green"])
        ax.set_xlabel("witness-subspace alignment"); ax.set_ylabel("install success")
        ax.set_title(f"F-B2 · Does installability ride the witness structure? "
                     f"ρ={it.get('rho', '?')} (shuffle_p={it.get('shuffle_p', '?')})", fontsize=9)
        footer(ax, "F-B2", "handle_b.install_tracking")
        save(fig, "F-B2")
    else:
        skip("F-B2", "handle_b install_tracking absent")

    # ================= F-X1 replication scoreboard =================
    if xm and xm.get("rx", {}).get("per_law"):
        pl = xm["rx"]["per_law"]
        laws = list(pl)
        colmap = {"replicates": CB["green"], "lands-elsewhere": CB["orange"], "VOID": CB["grey"]}
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(laws) + 1.5))
        for i, law in enumerate(laws):
            ax.barh(i, 1, color=colmap.get(pl[law], CB["grey"]), alpha=0.85)
            ax.text(0.5, i, pl[law], ha="center", va="center", fontsize=8)
        ax.set_yticks(range(len(laws))); ax.set_yticklabels(laws, fontsize=7)
        ax.set_xticks([])
        ne = xm["rx"].get("n_lands_elsewhere", "?")
        ax.set_title(f"F-X1 · First cross-model scope map on PG2 ({ne} laws land elsewhere)", fontsize=9)
        footer(ax, "F-X1", "handle_x.rx")
        save(fig, "F-X1")
    else:
        skip("F-X1", "handle_x rx absent")

    # ================= F-X2 PG2 operator panorama =================
    if xm and xm.get("law_verdicts"):
        lv = xm["law_verdicts"]
        arch = xm.get("pg2_arch", {})
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); ax.axis("off")
        txt = [f"PG2 arch: {arch.get('num_hidden_layers','?')}L d={arch.get('hidden_size','?')} "
               f"KV={arch.get('num_key_value_heads','?')} softcap={arch.get('attn_logit_softcapping','?')}",
               "PG1 label  →  PG2 label   (verdict)"]
        for lname, d in lv.items():
            txt.append(f"{lname:26s} {str(d.get('pg1'))[:16]:16s} → {str(d.get('pg2'))[:16]:16s}"
                       f"  {'VOID' if d.get('void') else ('replicates' if d.get('pg1')==d.get('pg2') else 'lands-elsewhere')}")
        ax.text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=7,
                family="monospace", transform=ax.transAxes)
        ax.set_title("F-X2 · PG2 vs PG1 core-law panorama (verdict labels only — firewall)", fontsize=9)
        save(fig, "F-X2")
    else:
        skip("F-X2", "handle_x legs absent")

    print("HANDLE_FIGURES_OK", json.dumps({"n": len([f for f in rep["figures"] if "SKIP" not in f])}))
    rep["green"] = True
    _w()


if __name__ == "__main__":
    main()
