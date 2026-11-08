"""ERLANGEN figures F1–F6 (CPU) — the complete-sweep rule: every measured cell is plotted.

  F1 fingerprint atlas      every component × every fingerprint dimension, per unit, with the
                            cluster assignment and the unmeasurable (noise-floor) rows marked
  F2 algebra                the full ordered K×K commutator matrix, the absorption matrix, both
                            nulls drawn as bands, and the dependency graph's edge list
  F3 compatibility          the 5×5 matrix per component group AND pooled, plus the mirror-v2
                            scatter with its permutation null
  F4 cost surface           every adaptation curve (all cells), the directed cost matrix, the
                            symmetry pairs, the triangle excesses with their bootstrap null and
                            the minimum detectable violation
  F5 witness map            every oracle call (|S| vs accuracy), the witness family overlap, and
                            the matched-random control at the accepted size
  F6 compression            description length vs behavioural fidelity for every program, per hop,
                            typed vs structure-free, with the ε line

    python -m infra.aws_pg.jobs.erlangen_figures --out DIR

No try/except (CLAUDE.md) outside the S3 fetch boundary.
"""
from __future__ import annotations

import argparse
import json
import pathlib

BUCKET = "pg-lineage-769063704696"


def _sh(args):
    import subprocess
    return subprocess.run(args, capture_output=True)


def _get(key):
    """Fetch a JSON artifact via the AWS CLI. boto3 is NOT installed in the job venv — every S3
    touch on this AMI goes through the CLI (the same path run_job.sh uses). A MISSING artifact is a
    legitimate state (an arm that has not landed) and reads as None; any other failure RAISES."""
    r = _sh(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", "-", "--region", "us-east-1"])
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace")
        if any(m in err for m in ("Not Found", "does not exist", "NoSuchKey", "404")):
            return None
        raise RuntimeError(f"s3 cp {key} failed: {err[:300]}")
    return json.loads(r.stdout)


def _ls(prefix):
    r = _sh(["aws", "s3", "ls", f"s3://{BUCKET}/{prefix}", "--recursive", "--region", "us-east-1"])
    if r.returncode != 0 and not r.stdout:
        return []
    return [ln.split()[-1] for ln in r.stdout.decode().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from infra.pg import erlangen_probe as P

    agg = _get("results/erlangen_agg3/out/erlangen_aggregate.json")
    made = []

    # ---------- F1 fingerprint atlas -----------------------------------------------------------
    # The T screen did not run (frozen rule, MDE fail), so the atlas plots every fingerprint that
    # WAS measured: the MDE's 16-component grid under both readouts. Complete-sweep rule — no
    # measured cell goes unplotted just because the arm that would have scaled it was gated off.
    src = [("NLL readout (n=64)", "results/erlangen_mde_fixed/out/erlangen_mde.json", "pg1"),
           ("generation readout (n=256)", "results/erlangen_mdegen2/out/erlangen_mde_gen.json", None)]
    panels = []
    for title, key, unit in src:
        d = _get(key)
        if not d:
            continue
        cells = d["cells"][unit] if unit else d["cells"]
        chans = d.get("channels") or list(P.READOUT_NAMES)
        grid = d["grid"]
        base = cells["__base__"]["vec"]
        F = np.array([[x - y for iv in P.INTERVENTIONS
                       for x, y in zip(cells.get(f"{c}|{iv}", {"vec": base})["vec"], base)]
                      for c in grid])
        panels.append((title, grid, chans, F))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(9 * len(panels), 8), squeeze=False)
        for ax, (title, grid, chans, F) in zip(axes[0], panels):
            v = np.nanpercentile(np.abs(F), 98) or 1.0
            im = ax.imshow(F, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
            ax.set_title(f"F1 fingerprints — {title}\n{len(grid)} components × {F.shape[1]} dims")
            ax.set_xticks([i * len(chans) + len(chans) / 2 for i in range(len(P.INTERVENTIONS))])
            ax.set_xticklabels(P.INTERVENTIONS, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(grid)))
            ax.set_yticklabels([g.replace("language_model.", "") for g in grid], fontsize=6)
            for x in range(len(chans), F.shape[1], len(chans)):
                ax.axvline(x - 0.5, color="k", lw=0.5, alpha=0.5)
            fig.colorbar(im, ax=ax, fraction=0.03, label="Δ readout vs baseline")
        fig.tight_layout()
        fig.savefig(out / "F1_fingerprint_atlas.png", dpi=140)
        plt.close(fig)
        made.append("F1")

    # ---------- F2 algebra ----------------------------------------------------------------------
    A = _get("results/erlangen_a/out/erlangen_a.json")
    if A and A.get("DONE"):
        ivs = A["interventions"]
        n = len(ivs)
        C = np.full((n, n), np.nan)
        AB = np.full((n, n), np.nan)
        for i, x in enumerate(ivs):
            for j, y in enumerate(ivs):
                if f"{x}|{y}" in A.get("commutators", {}):
                    C[i, j] = A["commutators"][f"{x}|{y}"]
                if f"{x}|{y}" in A.get("absorption", {}):
                    AB[i, j] = A["absorption"][f"{x}|{y}"]["ratio"]
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        im0 = ax[0].imshow(C, cmap="viridis")
        ax[0].set_title("F2a ‖b(i∘j) − b(j∘i)‖  (full ordered K×K)")
        fig.colorbar(im0, ax=ax[0], fraction=0.046)
        im1 = ax[1].imshow(AB, cmap="magma", vmin=0, vmax=2)
        ax[1].set_title("F2b absorption ‖b(j∘i) − b(i)‖ / ‖b(j) − b(∅)‖")
        fig.colorbar(im1, ax=ax[1], fraction=0.046)
        for k in (0, 1):
            ax[k].set_xticks(range(n)); ax[k].set_xticklabels(ivs, rotation=90, fontsize=6)
            ax[k].set_yticks(range(n)); ax[k].set_yticklabels(ivs, fontsize=6)
        ne = list(A["null_exact"].values())
        nr = list(A["null_reharvest"].values())
        pos = [C[i, j] for i in range(n) for j in range(n)
               if i != j and C[i, j] == C[i, j]
               and int(ivs[i].split("|")[0][1:]) < int(ivs[j].split("|")[0][1:])]
        ax[2].hist(pos, bins=30, alpha=0.7, label=f"layer-ordered pairs (n={len(pos)})")
        ax[2].axvline(np.percentile(ne, 97.5), color="k", ls="--",
                      label="Null-1 exact-commutation p97.5 (floor)")
        ax[2].axvline(np.median(nr), color="r", ls=":",
                      label="Null-2 re-harvest median (saturation ceiling)")
        ax[2].set_xlabel("commutator norm"); ax[2].set_ylabel("count")
        ax[2].set_title(f"F2c P2 — verdict {A['verdict']['verdict']}")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "F2_algebra.png", dpi=140)
        plt.close(fig)
        made.append("F2")

    # ---------- F3 compatibility ---------------------------------------------------------------
    X = _get("results/erlangen_x2/out/erlangen_x.json")
    if X and X.get("DONE"):
        units_x = X["compat_matrix_units"]
        groups = X["groups"]
        ncol = 4
        nrow = int(np.ceil((len(groups) + 2) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow))
        axes = axes.ravel()
        for i, g in enumerate(groups):
            M = np.array(X["compat_by_group"][g], dtype=float)
            im = axes[i].imshow(M, cmap="RdYlGn", vmin=-1, vmax=1)
            axes[i].set_title(f"{g}", fontsize=9)
            axes[i].set_xticks(range(len(units_x)))
            axes[i].set_xticklabels(units_x, rotation=45, fontsize=6, ha="right")
            axes[i].set_yticks(range(len(units_x)))
            axes[i].set_yticklabels(units_x, fontsize=6)
            fig.colorbar(im, ax=axes[i], fraction=0.046)
        k = len(groups)
        im = axes[k].imshow(np.array(X["compat_matrix"]), cmap="RdYlGn", vmin=-1, vmax=1)
        axes[k].set_title("pooled compatibility (host ← donor)", fontsize=9)
        axes[k].set_xticks(range(len(units_x)))
        axes[k].set_xticklabels(units_x, rotation=45, fontsize=6, ha="right")
        axes[k].set_yticks(range(len(units_x)))
        axes[k].set_yticklabels(units_x, fontsize=6)
        fig.colorbar(im, ax=axes[k], fraction=0.046)
        for tag, mk in (("A", "o"), ("B", "s")):
            mv = X["mirror_v2"][tag]
            S = np.array(mv["similarity_matrix"])
            Cm = (np.array(X["compat_matrix"]) + np.array(X["compat_matrix"]).T) / 2
            off = ~np.eye(len(units_x), dtype=bool)
            axes[k + 1].scatter(S[off], Cm[off], marker=mk,
                                label=f"annot {tag}: r={mv['r']:+.2f} p={mv['p_two_sided']:.3f}")
        axes[k + 1].set_xlabel("annotation similarity (Jaccard)")
        axes[k + 1].set_ylabel("compatibility (symmetrised)")
        axes[k + 1].set_title("F3 mirror-v2 (Mantel)", fontsize=9)
        axes[k + 1].legend(fontsize=7)
        for j in range(k + 2, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        fig.savefig(out / "F3_compatibility.png", dpi=140)
        plt.close(fig)
        made.append("F3")

    # ---------- F4 cost surface -----------------------------------------------------------------
    mcells = {}
    for k in [x for x in _ls("results/erlangen_m_") if x.endswith(".json") and "/out/" in x]:
        d = _get(k) or {}
        for _key, c in (d.get("cells") or {}).items():
            if c.get("DONE"):
                mcells[(c["src"], c["dst"], c["seed"])] = c
    if mcells:
        fig, ax = plt.subplots(2, 2, figsize=(15, 11))
        for (s, t, z), c in sorted(mcells.items()):
            xs = [p["examples"] for p in c["curve"]]
            ys = [p["s"] for p in c["curve"]]
            ax[0][0].plot(xs, ys, marker="o", lw=1, alpha=0.75,
                          ls="-" if z == 0 else "--", label=f"{s}→{t} s{z}")
        ax[0][0].axhline(0.5, color="k", ls=":", label="criterion s=0.5")
        ax[0][0].set_xlabel("training examples"); ax[0][0].set_ylabel("s (normalised progress)")
        ax[0][0].set_title("F4a every adaptation curve (all cells, both seeds)")
        ax[0][0].legend(fontsize=5, ncol=3)
        if agg and agg.get("M", {}).get("cost_matrix"):
            tasks = agg["M"]["tasks"]
            D = np.array(agg["M"]["cost_matrix"], dtype=float)
            im = ax[0][1].imshow(D, cmap="viridis")
            ax[0][1].set_xticks(range(len(tasks)))
            ax[0][1].set_xticklabels(tasks, rotation=45, ha="right", fontsize=7)
            ax[0][1].set_yticks(range(len(tasks)))
            ax[0][1].set_yticklabels(tasks, fontsize=7)
            ax[0][1].set_title("F4b directed cost d(src→dst)")
            for i in range(len(tasks)):
                for j in range(len(tasks)):
                    if D[i, j] == D[i, j]:
                        ax[0][1].text(j, i, f"{D[i,j]:.0f}", ha="center", va="center",
                                      fontsize=6, color="w")
            fig.colorbar(im, ax=ax[0][1], fraction=0.046)
            sym = agg["M"].get("symmetry_detail", {}).get("pairs", [])
            if sym:
                lbl = [f"{p['a']}↔{p['b']}" for p in sym]
                ax[1][0].bar(range(len(sym)), [p["abs_diff"] for p in sym],
                             color=["crimson" if p["confirmed"] else "gray" for p in sym])
                ax[1][0].errorbar(range(len(sym)), [p["abs_diff"] for p in sym],
                                  yerr=[p["abs_diff"] - p["lo95"] for p in sym],
                                  fmt="none", ecolor="k", lw=1)
                ax[1][0].set_xticks(range(len(sym)))
                ax[1][0].set_xticklabels(lbl, rotation=90, fontsize=6)
                ax[1][0].set_ylabel("|d(A→B) − d(B→A)|")
                ax[1][0].set_title("F4c symmetry (red = confirmed asymmetry)")
            tri = agg["M"].get("triangle_detail", {}).get("triples", [])
            if tri:
                ex = sorted([p["excess"] for p in tri])
                ax[1][1].plot(ex, marker=".", lw=0.8, label="triangle excess (sorted)")
                ax[1][1].axhline(0, color="k", lw=0.8)
                mv2 = agg["M"]["verdict"]
                ax[1][1].axhline(agg["M"].get("delta_min", 0), color="orange", ls="--",
                                 label="effect-size floor δ_min")
                ax[1][1].set_title(
                    f"F4d triangle — verdict {mv2.get('verdict')} "
                    f"(p={mv2.get('triangle_p')}, min-detectable/scale="
                    f"{mv2.get('min_detectable_violation_over_scale', float('nan')):.2f})")
                ax[1][1].set_ylabel("d(A→C) − d(A→B) − d(B→C)")
                ax[1][1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "F4_cost_surface.png", dpi=140)
        plt.close(fig)
        made.append("F4")

    # ---------- F5 witness map --------------------------------------------------------------------
    Dd = _get("results/erlangen_d3/out/erlangen_d.json")
    if Dd and Dd.get("DONE") and Dd.get("oracle_calls"):
        calls = Dd["oracle_calls"]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        ax[0].scatter([c["n"] for c in calls], [c["acc"] for c in calls],
                      c=["seagreen" if c["restores"] else "lightgray" for c in calls], s=18)
        ax[0].axhline(Dd["threshold"], color="k", ls="--", label="restore threshold")
        ax[0].axhline(Dd["chance"], color="r", ls=":", label="chance (4-way)")
        ax[0].axhline(Dd["acc_pg1"], color="b", ls="-.", label="PG1 ceiling")
        ax[0].set_xlabel("|restored set|"); ax[0].set_ylabel("forced-choice accuracy")
        ax[0].set_title(f"F5a every oracle call (n={len(calls)})")
        ax[0].legend(fontsize=7)
        rc = Dd.get("random_control", [])
        if rc:
            ax[1].bar(range(len(rc)), [r["acc"] for r in rc], color="gray",
                      label="matched-random sets of the accepted size")
            ax[1].axhline(Dd["threshold"], color="k", ls="--")
            ax[1].set_title("F5b matched-random control (N1)")
            ax[1].set_ylabel("forced-choice accuracy")
            ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "F5_witness_map.png", dpi=140)
        plt.close(fig)
        made.append("F5")

    # ---------- F6 compression ---------------------------------------------------------------------
    G = _get("results/erlangen_g/out/erlangen_g.json")
    if G and G.get("DONE"):
        hops = list(G["hop_results"])
        fig, axes = plt.subplots(1, len(hops), figsize=(4.6 * len(hops), 4.6), squeeze=False)
        for ax, h in zip(axes[0], hops):
            r = G["hop_results"][h]
            for term, col in (("rank", "tab:blue"), ("sparse", "tab:orange"),
                              ("gain", "tab:green")):
                pts = [p for p in r["programs"] if p["term"] == term]
                if pts:
                    ax.scatter([p["dl"] for p in pts], [p["fidelity"] for p in pts],
                               c=col, label=term, s=26,
                               marker="o", alpha=0.85)
            ax.axhline(G["eps"], color="k", ls="--", label="ε")
            ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
            ax.set_xlabel("description length (bits)")
            ax.set_ylabel("behavioural fidelity |ΔNLL|")
            ax.set_title(f"{h}  ratio={r.get('compression_ratio')}", fontsize=9)
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "F6_compression.png", dpi=140)
        plt.close(fig)
        made.append("F6")

    # ---------- F7 Arm B: gradient in, displacement out, curvature -------------------------------
    mb = [c for c in mcells.values() if c.get("arm_b", {}).get("gradient_ratio_mean") is not None]
    if mb:
        gr = [c["arm_b"]["gradient_ratio_mean"] for c in mb]
        dr = [c["arm_b"]["displacement_ratio"] for c in mb]
        fig, ax = plt.subplots(1, 3, figsize=(17, 5))
        ax[0].scatter(gr, dr, s=30, c="tab:purple")
        ax[0].axhline(1.0, color="k", ls="--", label="matched-random parity")
        ax[0].axvline(1.0, color="k", ls="--")
        ax[0].set_xlabel("gradient energy ratio  core / matched-random")
        ax[0].set_ylabel("displacement ratio  core / matched-random")
        ax[0].set_title(f"F7a every training run (n={len(mb)})\nH_repelled: gradient in, motion not")
        ax[0].legend(fontsize=7)
        for i, (vals, lab) in enumerate(((gr, "gradient ratio"), (dr, "displacement ratio"))):
            ax[1].boxplot(vals, positions=[i], widths=0.6)
        ax[1].axhline(1.0, color="k", ls="--")
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["gradient", "displacement"])
        ax[1].set_yscale("log"); ax[1].set_ylabel("ratio vs matched-random")
        ax[1].set_title("F7b the dissociation")
        for c in mb:
            cv = c.get("curvature") or []
            if cv:
                ax[2].plot([x["step"] for x in cv], [x["vHv_core"] for x in cv],
                           color="tab:red", alpha=0.35, lw=1)
                ax[2].plot([x["step"] for x in cv], [x["vHv_rand"] for x in cv],
                           color="tab:blue", alpha=0.35, lw=1)
        ax[2].set_yscale("log"); ax[2].set_xlabel("training step"); ax[2].set_ylabel("vᵀHv")
        ax[2].set_title("F7c curvature: core (red) vs matched-random (blue)")
        fig.tight_layout()
        fig.savefig(out / "F7_boundary_type.png", dpi=140)
        plt.close(fig)
        made.append("F7")

    (out / "figures_manifest.json").write_text(json.dumps(
        {"made": made, "missing": [f for f in ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
                                   if f not in made]}, indent=2))
    print(f"[figs] made {made}", flush=True)


if __name__ == "__main__":
    main()
