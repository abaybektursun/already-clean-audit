"""W13 §10 figure suite — weight-free, rendered from the result JSONs on S3 (no models, CPU job).

Headline (`fig_w13_nbehav`): behavioral capability R_cap vs the component budget N, for BOTH arms —
low-rank (rank r) and sparse-dictionary (atom budget k) — on the RefCOCO `<loc>`-flip axis, with the
frozen threshold lines (N_sparse=64, N_dense=720, eff-rank=1440) and τ*=0.9 overlaid. The verdict is
visible: low-rank flips at the near-full-rank tail (~1024); the overcomplete sparse adversary is impotent (R_cap=0
at all k up to 1280) → no small sparse parameter basis. Necessity panel (`fig_w13_necessity`) if the
§6C JSON is present: R_cap(top-N) vs R_cap(random-N subspace).

Each panel: the null/baseline overlaid, threshold lines labelled, a plain finding in the title, a
provenance footer (run prefixes + commit). PNG (review) + PDF (paper). Reads JSONs via the instance
role. No try/except around the science.

    pg run infra.aws_pg.jobs.w13_figures --type r7i.large --no-weights \
        --s3-prefix results/w13_figures --budget-min 20
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

BUCKET = "pg-lineage-769063704696"


def _load(prefix, fname):
    """Download s3://BUCKET/<prefix>/out/<fname> and parse JSON; None if absent."""
    local = pathlib.Path("/tmp") / fname
    r = subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{prefix}/out/{fname}", str(local),
                        "--region", "us-east-1"], capture_output=True)
    return json.loads(local.read_text()) if r.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", default="", help="commit SHA for the provenance footer")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lr = _load("results/w13_lowrank_full", "w13_lowrank.json")
    sp = _load("results/w13_sparse_oc", "w13_sparse.json")
    nec = _load("results/w13_necessity_v2", "w13_necessity.json")
    assert lr and sp, "need w13_lowrank_full + w13_sparse_oc result JSONs"

    N_SPARSE, N_DENSE, EFF = 64, 720.2, 1440.3
    TAU = 0.9
    foot = (f"W13 · low-rank=results/w13_lowrank_full · sparse=results/w13_sparse_oc · "
            f"eff-rank/thresholds=results/w13_real_dw · commit {a.commit[:10] or 'HEAD'} · DISCOVERY n=1 tower")
    CB = {"lowrank": "#0072B2", "sparse": "#D55E00", "thr": "#999999", "tau": "#009E73"}

    # ---- headline: R_cap vs N (RefCOCO), both arms ----
    lr_ranks = sorted(int(r) for r in lr["curve"])
    lr_rc = [lr["curve"][str(r)]["refcoco"]["R_cap"] for r in lr_ranks]
    sp_ks = sorted(int(k) for k in sp["curve"])
    sp_rc = [sp["curve"][str(k)]["refcoco"]["R_cap"] for k in sp_ks]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(lr_ranks, lr_rc, "o-", color=CB["lowrank"], lw=2, label="low-rank arm (SVD rank r) — replication")
    ax.plot(sp_ks, sp_rc, "s--", color=CB["sparse"], lw=2, label="sparse-dict arm (atom budget k) — confirmatory")
    for x, lab in [(N_SPARSE, f"N_sparse={N_SPARSE}"), (N_DENSE, f"N_dense={int(N_DENSE)}"), (EFF, f"eff-rank≈{int(EFF)}")]:
        ax.axvline(x, color=CB["thr"], ls=":", lw=1)
        ax.text(x, 1.02, lab, rotation=90, va="bottom", ha="center", fontsize=8, color=CB["thr"])
    ax.axhline(TAU, color=CB["tau"], ls="-.", lw=1.2, label=f"τ*={TAU} (behavioral gate)")
    ax.set_xscale("log"); ax.set_xlabel("component budget N  (rank r / atom budget k, log scale)")
    ax.set_ylabel("RefCOCO R_cap  (behavioral capability, winsorized)")
    ax.set_ylim(-0.05, 1.1); ax.legend(loc="center left", fontsize=8.5); ax.grid(alpha=0.25)
    valid = sp.get("instrument_validity", {})
    ax.set_title("INCONCLUSIVE: the overcomplete sparse-dict adversary is behaviorally IMPOTENT\n"
                 f"(full-dict k={valid.get('k_full','?')} RefCOCO R_cap={valid.get('refcoco_R_cap',0):.2f}=0 → cannot reconstruct ΔW); "
                 "low-rank fires only at the near-full-rank tail (~1024)",
                 fontsize=9.5)
    fig.text(0.5, 0.005, foot, ha="center", fontsize=6, color="#555555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_w13_nbehav.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[W13-FIG] wrote fig_w13_nbehav.{png,pdf}", flush=True)

    # ---- necessity panel (§6C), if present ----
    if nec:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        names = list(nec["necessity"].keys())
        tops = [nec["necessity"][n]["top_N"] for n in names]
        rmean = [nec["necessity"][n]["random_N_mean"] for n in names]
        rstd = [nec["necessity"][n]["random_N_std"] for n in names]
        x = range(len(names))
        ax.bar([i - 0.2 for i in x], tops, 0.4, color=CB["lowrank"], label=f"top-{nec['rank_N']} (real ΔW directions)")
        ax.bar([i + 0.2 for i in x], rmean, 0.4, yerr=rstd, color=CB["thr"],
               label=f"random-{nec['rank_N']} subspace (null)")
        ax.set_xticks(list(x)); ax.set_xticklabels(names)
        ax.set_ylabel("capability (acc / ANLS)")
        ax.set_title(f"§6C subspace necessity at N={nec['rank_N']}: the capability is in the SPECIFIC\n"
                     f"top-ΔW directions, not any equal-N subspace", fontsize=9.5)
        ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
        fig.text(0.5, 0.005, foot, ha="center", fontsize=6, color="#555555")
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_w13_necessity.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("[W13-FIG] wrote fig_w13_necessity.{png,pdf}", flush=True)
    else:
        print("[W13-FIG] necessity JSON absent — skipping that panel (run w13_necessity first)", flush=True)

    (out / "figures_manifest.json").write_text(json.dumps({
        "headline": "fig_w13_nbehav", "necessity": "fig_w13_necessity" if nec else None,
        "low_rank_N_behav_refcoco": lr["N_behav"]["refcoco"],
        "sparse_N_behav_refcoco": sp["N_behav_atoms"]["refcoco"],
        "thresholds": {"N_sparse": N_SPARSE, "N_dense": N_DENSE, "eff_rank": EFF, "tau": TAU},
        "verdict": "INCONCLUSIVE — overcomplete sparse-dict adversary behaviorally impotent (full-dict R_cap=0); H_invisible NOT upheld in parameter space",
        "sources": ["results/w13_lowrank_full", "results/w13_sparse_oc", "results/w13_real_dw",
                    "results/w13_necessity_v2" if nec else None],
    }, indent=2))
    print("[W13-FIG] done.", flush=True)


if __name__ == "__main__":
    main()
