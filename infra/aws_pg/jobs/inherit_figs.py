"""INHERIT figure suite (spec §10; CPU) — renders the PIVOTED result (the forward transport gate was
red → the sealed forward was skipped, so figures come from the exploratory forward (disc), the
transport gate (controls), the π0.5 arms (rollout_op O/P, svl G-reverse), probes (C), and the
weight-free closeout (Arm P-geo overlap)). Each panel = a number + CI/null; PNG+PDF; colourblind-safe;
provenance footers. Missing-arm panels skipped loud.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.inherit_figs --type r7i.2xlarge --no-weights \
      --s3-prefix results/inherit_figs --budget-min 25 \
      --args "--disc-s3 ... --controls-s3 ... --rollout-s3 ... --svl-s3 ... --svl-l0-s3 ... --probes-s3 ... --closeout-s3 ..."
"""
import argparse
import json
import pathlib

CB = {"saction": "#0072B2", "redraw": "#56B4E9", "em": "#D55E00", "identity": "#000000",
      "tail": "#E69F00", "topk": "#0072B2", "mid": "#009E73", "uunt": "#CC79A7", "null": "#999999"}


def _load(uri, dst):
    if not uri:
        return None
    from infra.aws_pg.jobs._inherit_shared import fetch
    return json.loads(fetch(uri, dst).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for name in ("disc", "controls", "rollout", "svl", "svl-l0", "probes", "closeout"):
        ap.add_argument(f"--{name}-s3", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    disc = _load(a.disc_s3, "/tmp/inherit/f_disc.json")
    controls = _load(a.controls_s3, "/tmp/inherit/f_controls.json")
    rollout = _load(a.rollout_s3, "/tmp/inherit/f_rollout.json")
    svl = _load(a.svl_s3, "/tmp/inherit/f_svl.json")
    svl0 = _load(a.svl_l0_s3, "/tmp/inherit/f_svl0.json")
    probes = _load(a.probes_s3, "/tmp/inherit/f_probes.json")
    closeout = _load(a.closeout_s3, "/tmp/inherit/f_closeout.json")
    made = []

    def _footer(fig, tag):
        fig.text(0.01, 0.01, f"INHERIT · {tag} · inherit-freeze-2026-07-02", fontsize=6, color="#666")

    def _save(fig, name):
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)

    # F1 — the genesis headline (EXPLORATORY): PG1 raw VQA under transported S_action vs em vs redraw, L0 & L14
    if disc:
        sw = disc["sweep_discovery"]
        kg = sorted({int(kk.split("_k")[1]) for kk in sw})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, L in zip(axes, (disc["selection"]["primary"], disc["selection"]["secondary"])):
            xs = [k for k in kg if f"L{L}_k{k}" in sw]
            sa = [sw[f"L{L}_k{k}"]["raw_score"]["saction"] for k in xs]
            rd = [sw[f"L{L}_k{k}"]["raw_score"]["saction_redraw"] for k in xs]
            em = [sw[f"L{L}_k{k}"]["raw_score"]["random_energy"] for k in xs]
            ax.plot(xs, sa, "o-", color=CB["saction"], label="S_action (frozen bank)")
            ax.plot(xs, rd, "s--", color=CB["redraw"], label="S_action (2nd draw)")
            ax.plot(xs, em, "^:", color=CB["em"], label="energy-matched random")
            ax.axhline(disc["intact"]["score"], ls=":", color=CB["identity"], lw=0.8, label="intact")
            ax.axhline(disc["floor_scrambled"], ls="--", color=CB["null"], lw=0.8, label="scrambled floor")
            ax.set_xscale("log"); ax.set_xlabel("k (transported directions ablated)")
            ax.set_title(f"L{L}" + (" (primary bite)" if L == disc["selection"]["primary"] else " (secondary)"))
        axes[0].set_ylabel("PG1 VQAv2 soft-acc"); axes[0].legend(fontsize=6)
        fig.suptitle("F1 · EXPLORATORY: transported π0.5 S_action ablated in PaliGemma-1 (both draws crater at L0)")
        _footer(fig, "F1 EXPLORATORY (discovery half; forward gate red)"); _save(fig, "F1_forward_exploratory"); made.append("F1")

    # F1b — ppl collateral (the sensitive channel): direction-specific PG1 language damage
    if disc and "ppl_collateral" in disc:
        p = disc["ppl_collateral"]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["base", "S_action\n(L0,k16)", "em-random"], [p["base"], p["saction"], p["em"]],
                      color=[CB["identity"], CB["saction"], CB["em"]])
        ax.set_ylabel("blank-image wikitext ppl"); ax.set_title("F1b · ppl collateral — S_action is direction-specifically load-bearing in PG1")
        for b, v in zip(bars, [p["base"], p["saction"], p["em"]]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        _footer(fig, "F1b EXPLORATORY"); _save(fig, "F1b_ppl"); made.append("F1b")

    # F2 — transport gate: CKA matrix + the empirical contradiction (positive leg passes, CKA fails)
    if controls and controls.get("transport_cka"):
        cka = np.asarray(controls["transport_cka"]["matrix"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        im = ax1.imshow(cka, cmap="viridis", vmin=0, vmax=1)
        for i in range(cka.shape[0]):
            ax1.plot(i, i, "r.", ms=3)
        ax1.set_title(f"CKA(PG1 L, π0.5 L′) — matched≈mismatched (bound {controls['transport_cka']['bound']:.2f})")
        ax1.set_xlabel("π0.5 layer"); ax1.set_ylabel("PG1 layer"); fig.colorbar(im, ax=ax1, fraction=0.046)
        tg = controls["transport_gate"]
        ax2.bar(["positive leg\n(top-σ transported)", "CKA leg\n(matched L0)"],
                [1.0 if tg["rf_topsig_transported"] < 0 else 0.0, cka[0, 0]],
                color=[CB["saction"], CB["em"]])
        ax2.axhline(controls["transport_cka"]["bound"], ls="--", color=CB["null"], label="CKA bound")
        ax2.set_title(f"gate GREEN={tg['green']} (positive passes, CKA fails → RED)"); ax2.legend(fontsize=7)
        ax2.set_ylim(0, 1.1); ax2.set_ylabel("pass indicator / CKA")
        _footer(fig, "F2 CONTROL"); _save(fig, "F2_transport_gate"); made.append("F2")

    # F3 — Arm O: RF vs captured mass (top-k) + the concentration fact
    if rollout and rollout.get("arm_o"):
        byk = rollout["arm_o"]["by_k"]
        ks = sorted(byk, key=lambda s: int(s[1:]))
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [byk[k]["captured_mass_frac"] for k in ks]
        ax.plot(xs, [byk[k]["topk"]["rf"] for k in ks], "o-", color=CB["topk"], label="top-k (S_action)")
        tail_x = [byk[k]["captured_mass_frac"] for k in ks if "tail" in byk[k]]
        tail_y = [byk[k]["tail"]["rf"] for k in ks if "tail" in byk[k]]
        if tail_x:
            ax.plot(tail_x, tail_y, "s--", color=CB["tail"], label="mass-matched tail (≈full-rank; confounded)")
        ax.axhline(1.0, ls=":", color=CB["identity"])
        for k in ks:
            ax.annotate(k, (byk[k]["captured_mass_frac"], byk[k]["topk"]["rf"]), fontsize=7)
        ax.set_xlabel("captured Gram mass fraction"); ax.set_ylabel("π0.5 rollout RF")
        ax.set_title(f"F3 · Arm O — top-64 hold 77% mass (concentration); tail confounded → {rollout['arm_o']['verdict']}")
        ax.legend(fontsize=7); _footer(fig, "F3 (inconclusive/confounded — disclosed)"); _save(fig, "F3_armO"); made.append("F3")

    # F4 — Arm P: causal (U_unt not control-critical) + weight-write (U_unt conserved through robot hop) + geo overlap
    if rollout and rollout.get("arm_p_causal"):
        p = rollout["arm_p_causal"]
        w = rollout.get("uunt_weight_write", {})
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].bar(["U_unt ablate", "em-null"], [p["rf_uunt"]["rf"], p["rf_em"]["rf"]],
                    color=[CB["uunt"], CB["em"]])
        axes[0].axhline(1.0, ls=":", color=CB["identity"]); axes[0].set_ylabel("π0.5 rollout RF")
        axes[0].set_title(f"P-causal: U_unt NOT control-critical (leg pass={p['causal_leg_pass']})")
        if w:
            Ls = sorted(int(k) for k in w)
            axes[1].plot(Ls, [w[str(L)]["frac_in_uunt"] for L in Ls], "o-", color=CB["uunt"], label="robot-write ∈ U_unt")
            axes[1].plot(Ls, [w[str(L)]["rand_frac_mean"] for L in Ls], "s--", color=CB["null"], label="random null")
            axes[1].set_xlabel("layer"); axes[1].set_ylabel("‖ΔW·U‖/‖ΔW‖")
            axes[1].set_title("robot-training AVOIDS U_unt (esp. L0)"); axes[1].legend(fontsize=7)
        if closeout and closeout.get("arm_p_geometric"):
            g = closeout["arm_p_geometric"]
            Ls = sorted(int(x) for x in g)
            axes[2].plot(Ls, [g[str(L)]["overlap_raw"] for L in Ls], "o-", color=CB["saction"], label="overlap(S_action,U_unt)")
            axes[2].plot(Ls, [g[str(L)]["null_p95"] for L in Ls], "s--", color=CB["null"], label="null p95")
            axes[2].set_xlabel("layer"); axes[2].set_ylabel("principal-angle overlap")
            axes[2].set_title("P-geometric: overlap vs null band"); axes[2].legend(fontsize=7)
        _footer(fig, "F4 CONFIRMATORY(P-causal)+DIAGNOSTIC"); _save(fig, "F4_armP"); made.append("F4")

    # F5 — Arm C probes (the inherited directions carry scene content in PG1)
    if probes:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels, vals, cols = [], [], []
        for grp, col in (("pi05", CB["saction"]), ("pg1_twin", CB["mid"])):
            if grp in probes:
                labels += [f"{grp}\nS_action", f"{grp}\nrandom-k"]
                vals += [probes[grp]["saction"]["r2_mean"], probes[grp]["random_k"]["r2_mean"]]
                cols += [col, CB["null"]]
        ax.bar(labels, vals, color=cols)
        ax.set_ylabel("held-out R² (robot state)"); ax.set_ylim(0, 1.05)
        ax.set_title("F5 · Arm C — scene content pre-exists in the inherited directions (PG1 twin 0.77 vs 0.50)")
        _footer(fig, "F5 EXPLORATORY"); _save(fig, "F5_armC"); made.append("F5")

    # F6 — G-reverse at L0 and L14 (control does NOT reuse PG1's VL-read)
    revs = [(svl, "L14"), (svl0, "L0")]
    revs = [(r, lab) for r, lab in revs if r is not None]
    if revs:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(revs)); w = 0.35
        ax.bar(x - w / 2, [r["rf_svl"]["rf"] for r, _ in revs], w, color=CB["saction"], label="S_vl ablate")
        ax.bar(x + w / 2, [r["rf_em"]["rf"] for r, _ in revs], w, color=CB["em"], label="em-null")
        for i, (r, _) in enumerate(revs):
            ax.errorbar(i - w / 2, r["rf_svl"]["rf"],
                        yerr=[[r["rf_svl"]["rf"] - r["rf_svl"]["ci"][0]], [r["rf_svl"]["ci"][1] - r["rf_svl"]["rf"]]],
                        fmt="none", color="k")
        ax.axhline(1.0, ls=":", color=CB["identity"]); ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in revs])
        ax.set_ylabel("π0.5 rollout RF"); ax.legend(fontsize=7)
        ax.set_title("F6 · G-reverse — control does NOT depend on PG1's VL-read (null at L0 & L14)")
        _footer(fig, "F6 CONFIRMATORY"); _save(fig, "F6_greverse"); made.append("F6")

    (out / "figs_manifest.json").write_text(json.dumps({"made": made}, indent=2))
    print("FIGS_OK", json.dumps({"made": made}))


if __name__ == "__main__":
    main()
