"""RATION figures (CPU) — the 6-figure suite (spec §10). Weight-free: reads the banked JSONs from S3
and renders PNG+PDF. Every panel: all policies, all budgets (feedback_complete_sweep_visualization),
CI bands, null/oracle overlays, b̄* marked, plain-language title, provenance footer, colourblind-safe.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.ration_figures --type r7i.2xlarge \
      --no-weights --s3-prefix results/ration_figures --budget-min 20 \
      --args "--run-prefix results/ration_run"
"""
import argparse
import json
import os
import pathlib


COL = {"susceptibility": "#0072B2", "variance": "#D55E00", "sink_aware_variance": "#E69F00",
       "magnitude": "#CC79A7", "uniform": "#999999", "random": "#000000", "oracle": "#009E73"}
LAB = {"susceptibility": "susceptibility (Q)", "variance": "variance", "sink_aware_variance": "sink-aware variance",
       "magnitude": "magnitude", "uniform": "uniform (floor)", "random": "random (null)", "oracle": "oracle (ceiling)"}


def _fetch(uri, local):
    import subprocess
    subprocess.run(["aws", "s3", "cp", uri, str(local), "--region", "us-east-1"], check=True)
    return json.loads(pathlib.Path(local).read_text())


def _save(fig, out, name, foot):
    fig.text(0.005, 0.005, foot, fontsize=5, color="#666", ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"[ration-fig] {name}", flush=True)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-prefix", default="results/ration_run")
    ap.add_argument("--disc-prefix", default="results/ration_disc")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    B = "s3://pg-lineage-769063704696"
    wd = pathlib.Path("/tmp/rationfig")
    wd.mkdir(parents=True, exist_ok=True)
    rate = _fetch(f"{B}/{a.run_prefix}/out/conf/ration_rate.json", wd / "rate.json")
    conf = _fetch(f"{B}/{a.run_prefix}/out/conf/ration_conf.json", wd / "conf.json")
    layer = _fetch(f"{B}/{a.run_prefix}/out/layer/ration_layer.json", wd / "layer.json")
    try:
        kv = _fetch(f"{B}/{a.run_prefix}/out/kv/ration_kv.json", wd / "kv.json")
    except Exception:
        kv = None
    sweep = rate["sweep"]
    fr = rate["freeze"]
    grid = rate["budget_grid"]
    foot = f"RATION · PG1 L0 · b̄*_ppl={fr['bstar_ppl']} b̄*_vqa={fr['bstar_vqa']} · CONFIRMATORY sealed half"

    def curve(ax, readout, policies, marker_bstar, logy=False):
        for p in policies:
            xs, ys, los, his = [], [], [], []
            for b in grid:
                c = sweep[p][str(b)][f"{readout}_loss"]
                xs.append(b); ys.append(c["E"]); los.append(c["ci_lo"]); his.append(c["ci_hi"])
            ax.plot(xs, ys, "-o", color=COL[p], label=LAB[p], lw=1.6, ms=4)
            ax.fill_between(xs, los, his, color=COL[p], alpha=0.15)
        if logy:                                             # ppl loss spans 0.2→40+ (oracle starves)
            ax.set_yscale("symlog", linthresh=0.1)
        if marker_bstar is not None:
            ax.axvline(marker_bstar, color="k", ls=":", lw=1, alpha=0.6)
            ax.text(marker_bstar, ax.get_ylim()[1], " b̄*", fontsize=8, va="top")
        ax.set_xlabel("average bits  b̄"); ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_ylabel(f"{readout} behavior loss  (clean − policy)" + ("  [symlog]" if logy else ""))

    # ---- Fig 1: headline rate-distortion (both readouts, all policies) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, ro, bs in ((axes[0], "ppl", fr["bstar_ppl"]), (axes[1], "vqa", fr["bstar_vqa"])):
        curve(ax, ro, list(COL), bs, logy=(ro == "ppl"))
        ax.set_title(f"{ro.upper()}: behavior lost vs bit budget, per allocation policy")
    axes[0].legend(fontsize=6, ncol=2, loc="upper right")
    fig.suptitle("RATION headline — does allocating bits by susceptibility preserve behavior "
                 "better than by energy?", fontsize=11)
    _save(fig, out, "fig1_headline_rate_distortion", foot)
    plt.close(fig)

    # ---- Fig 2: the mirror — behavior (susc wins) vs MSE (variance wins) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for p in ("susceptibility", "variance", "sink_aware_variance", "uniform"):
        xs = grid
        axes[0].plot(xs, [sweep[p][str(b)]["ppl_loss"]["E"] for b in xs], "-o", color=COL[p], label=LAB[p], ms=4)
        axes[1].plot(xs, [sweep[p][str(b)]["mse"] for b in xs], "-o", color=COL[p], label=LAB[p], ms=4)
    axes[0].set_title("behavior loss (ppl) by allocation policy"); axes[0].set_xlabel("b̄")
    axes[0].set_ylabel("ppl behavior loss  [symlog]"); axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].legend(fontsize=7)
    axes[1].set_title("reconstruction MSE (variance is MSE-optimal, Eckart-Young)"); axes[1].set_xlabel("b̄")
    axes[1].set_ylabel("residual MSE"); axes[1].set_yscale("log")
    fig.suptitle("RATION mirror — do behavior and reconstruction agree or dissociate?", fontsize=11)
    _save(fig, out, "fig2_mirror", foot)
    plt.close(fig)

    # ---- Fig 3: sink adjudication at b̄* ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ro, bs in ((axes[0], "ppl", fr["bstar_ppl"]), (axes[1], "vqa", fr["bstar_vqa"])):
        if bs is None:
            continue
        pol = ["variance", "sink_aware_variance", "susceptibility"]
        vals = [sweep[p][str(bs)][f"{ro}_loss"]["E"] for p in pol]
        los = [sweep[p][str(bs)][f"{ro}_loss"]["E"] - sweep[p][str(bs)][f"{ro}_loss"]["ci_lo"] for p in pol]
        his = [sweep[p][str(bs)][f"{ro}_loss"]["ci_hi"] - sweep[p][str(bs)][f"{ro}_loss"]["E"] for p in pol]
        ax.bar(range(3), vals, yerr=[los, his], color=[COL[p] for p in pol], capsize=4)
        ax.set_xticks(range(3)); ax.set_xticklabels(["naive var", "sink-aware var", "suscept."], fontsize=8)
        ax.set_title(f"{ro.upper()} loss at b̄*={bs}  (lower = better)"); ax.set_ylabel("behavior loss")
    fig.suptitle("RATION sink adjudication — is the win more than 'don't quantise the sink'?", fontsize=11)
    _save(fig, out, "fig3_sink_adjudication", foot)
    plt.close(fig)

    # ---- Fig 4: layer leverage — E across 18 layers ----
    pl = layer["per_layer"]
    Ls = sorted(int(k) for k in pl)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for ro, col in (("ppl", "#0072B2"), ("vqa", "#D55E00")):
        key = f"E_{ro}"
        if key in pl[str(Ls[0])]:
            es = [pl[str(L)][key]["E"] for L in Ls]
            lo = [pl[str(L)][key]["ci_lo"] for L in Ls]
            hi = [pl[str(L)][key]["ci_hi"] for L in Ls]
            ax.plot(Ls, es, "-o", color=col, label=f"{ro} excess (susc − sink-aware)", ms=4)
            ax.fill_between(Ls, lo, hi, color=col, alpha=0.15)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="green", ls=":", label="L0 (matched Q)")
    ax.set_xlabel("decoder layer L"); ax.set_ylabel("excess E (susc − sink-aware)")
    ax.set_title("RATION Arm L — where does the L0 susceptibility map retain leverage? (transfer)")
    ax.legend(fontsize=7)
    _save(fig, out, "fig4_layer_leverage", foot)
    plt.close(fig)

    # ---- Fig 5: oracle headroom — how much of the ceiling susceptibility captures ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, ro, bs in ((axes[0], "ppl", fr["bstar_ppl"]), (axes[1], "vqa", fr["bstar_vqa"])):
        if bs is None:
            continue
        pol = ["uniform", "variance", "sink_aware_variance", "susceptibility", "oracle"]
        vals = [sweep[p][str(bs)][f"{ro}_loss"]["E"] for p in pol]
        ax.bar(range(len(pol)), vals, color=[COL[p] for p in pol])
        ax.set_xticks(range(len(pol)))
        ax.set_xticklabels(["uniform", "var", "sink-var", "susc", "oracle"], fontsize=8, rotation=20)
        ax.set_title(f"{ro.upper()} loss at b̄*={bs}: uniform → oracle span"); ax.set_ylabel("behavior loss")
    fig.suptitle("RATION oracle headroom — how much of the available headroom the map captures", fontsize=11)
    _save(fig, out, "fig5_oracle_headroom", foot)
    plt.close(fig)

    # ---- Fig 6: KV extension (machinery-reach) ----
    if kv is not None:
        ksw = kv["kv_sweep"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, ro in ((axes[0], "ppl"), (axes[1], "vqa")):
            for p in ksw:
                xs = sorted(int(b) for b in ksw[p])
                ys = [ksw[p][str(b)][f"{ro}_loss"]["E"] for b in xs]
                ax.plot(xs, ys, "-o", label=p, ms=4)
            ax.set_xlabel("average KV bits  b̄"); ax.set_ylabel(f"{ro} behavior loss")
            ax.set_title(f"{ro.upper()} loss vs KV-cache bit budget"); ax.axhline(0, color="k", lw=0.5)
        axes[0].legend(fontsize=8)
        fig.suptitle("RATION Arm K — the quantiser machinery reaches the KV cache (energy axes; "
                     "map-transfer is a follow-up)", fontsize=10)
        _save(fig, out, "fig6_kv_extension", foot)
        plt.close(fig)

    (out / "figures_done.json").write_text(json.dumps(
        {"figures": ["fig1_headline_rate_distortion", "fig2_mirror", "fig3_sink_adjudication",
                     "fig4_layer_leverage", "fig5_oracle_headroom"] + (["fig6_kv_extension"] if kv else []),
         "verdict_ppl": conf.get("verdict_ppl", {}).get("verdict"),
         "verdict_vqa": conf.get("verdict_vqa", {}).get("verdict")}, indent=2))
    print("RATION_FIGURES_OK")


if __name__ == "__main__":
    main()
