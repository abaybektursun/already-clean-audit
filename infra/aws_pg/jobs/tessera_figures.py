"""TESSERA figures — spec §10. Every panel carries its null, its CI, and a provenance footer.

Figures are never evidence on their own: each panel illustrates a claim that is separately backed
by a number + CI + null in the arm artifact, and the panel SHOWS that null rather than asserting
the claim. Panels for arms that have not run render as an explicit "PENDING" placeholder rather
than being omitted — the same rule the aggregator applies to grid rows.

  F2  the 26-unit consensus core: real bottom-k vs the matched-rotation null band, per q/k matrix
  F7  cross-hop subspace overlap: EVERY pair × EVERY matrix, against BOTH nulls (Haar and the
      unrelated-FT reference) — the full sweep, not a summary, per the house complete-sweep rule
  F3  K1 footprint vs survival, per faculty, with the chance floor r/d drawn
  F4  crosscoder fractions per pair and site, with the identity band AND the per-pair shuffle null
  F5  n* ladders with the planted rank-1 control

  python -m infra.aws_pg.jobs.tessera_figures --out fig --s <arm_s.json> [--k1 ...] [--x ...] [--k3 ...]

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

FOOT = "TESSERA · science/tessera · tessera-freeze-20260801"


def _read(p):
    if p is None:
        return None
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def _pending(ax, title, why):
    ax.text(0.5, 0.5, f"PENDING\n{why}", ha="center", va="center", fontsize=11,
            transform=ax.transAxes)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def f2_core(ax, s):
    """26-unit consensus core vs the matched-rotation null, per q/k matrix at k=8."""
    if not s:
        return _pending(ax, "F2 — 26-unit consensus core", "Arm S not run")
    import numpy as np
    rows = s["core"]["n26"]
    xs = np.arange(len(rows))
    real = [r["bottom_k"]["8"] for r in rows]
    nlo = [r["null_bottom"]["8"]["p2.5"] for r in rows]
    nhi = [r["null_bottom"]["8"]["p97.5"] for r in rows]
    nmu = [r["null_bottom"]["8"]["mean"] for r in rows]
    ax.fill_between(xs, nlo, nhi, alpha=0.35, label="matched-rotation null 95% band", zorder=1)
    ax.plot(xs, nmu, lw=1.0, ls="--", label="null mean", zorder=2)
    ax.plot(xs, real, lw=0, marker="o", ms=3.5, label="26-unit consensus (real)", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("q/k matrix (18 layers × {q_proj, k_proj})")
    ax.set_ylabel("mean of bottom-8 eigenvalues of G")
    n_beyond = sum(1 for r in rows if r["verdict"]["8"]["verdict"] == "BEYOND_NULL_below")
    ax.set_title(f"F2 — the untouched q/k core SURVIVES the VLA writes\n"
                 f"{n_beyond}/{len(rows)} matrices below the matched-rotation null at k=8",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.text(0.5, -0.22, "How to read:each point is one weight matrix. Real BELOW the band ⇒ the 26 "
                        "units jointly leave that subspace more untouched than matched rotations do.",
            transform=ax.transAxes, ha="center", fontsize=6.5, wrap=True)


def f7_overlap(ax, s):
    """The COMPLETE cross-hop sweep: every pair × every matrix, against BOTH nulls."""
    if not s:
        return _pending(ax, "F7 — cross-hop overlap (both nulls)", "Arm S not run")
    import numpy as np
    ovs = s["overlaps"]
    haar = s.get("haar_bands", {})
    ftref = (s.get("ft_reference_null") or {}).get("per_key", {})
    off = 0
    for o in ovs:
        # Within each pair, order matrices by their OWN reference band so the band reads as a band
        # rather than as noise. Interleaved matrix types (q_proj vs down_proj) have reference values
        # that differ by 3x, and plotting them in key order makes the reference look like a comb.
        # Every cell is still drawn — the complete-sweep rule is about coverage, not ordering.
        keyed = [r for r in o["per_key"] if r["key"] in ftref]
        rest = [r for r in o["per_key"] if r["key"] not in ftref]
        keyed.sort(key=lambda r: ftref[r["key"]]["median"])
        seq = keyed + rest
        xs = np.arange(len(seq)) + off
        if keyed:
            kx = np.arange(len(keyed)) + off
            ax.fill_between(kx, [ftref[r["key"]]["p2.5"] for r in keyed],
                            [ftref[r["key"]]["p97.5"] for r in keyed],
                            color="0.75", alpha=0.45,
                            label="unrelated-FT reference band, p2.5–p97.5" if off == 0 else None)
            ax.plot(kx, [ftref[r["key"]]["p2.5"] for r in keyed], lw=0.9, color="crimson",
                    alpha=0.85,
                    label="p2.5 — the branch A12 lacked (C4)" if off == 0 else None)
        below = [(x, r["overlap"]) for x, r in zip(xs, seq)
                 if r["key"] in ftref and r["overlap"] < ftref[r["key"]]["p2.5"]]
        ax.plot(xs, [r["overlap"] for r in seq], lw=0, marker=".", ms=2.5,
                label="~".join(o["pair"]))
        if below:
            ax.plot([b[0] for b in below], [b[1] for b in below], lw=0, marker="o", ms=3.5,
                    mfc="none", mec="crimson", mew=0.6,
                    label="BELOW the reference" if off == 0 else None)
        off += len(seq) + 4
    hb = [b["p97.5"] for b in haar.values()]
    if hb:
        ax.axhline(max(hb), ls="--", lw=1.0, color="0.35", label="Haar p97.5 (max over n)")
    ax.set_xlabel("matrix, grouped by pair")
    ax.set_ylabel("subspace overlap (mean squared principal cosine, r=64)")
    n_sh = sum(o["n_shared_channel"] for o in ovs)
    n_en = sum(o.get("n_energy_aligned", 0) for o in ovs)
    # C4: A12's demotion was one-sided, so cells BELOW the reference's p2.5 were labelled
    # ENERGY-ALIGNED ("inside the reference") while sitting below it. Draw the lower edge and
    # count against it, or this panel keeps asserting the pre-critic reading.
    n_below = n_scored = 0
    for o in ovs:
        for r in o["per_key"]:
            ref = ftref.get(r["key"])
            if ref:
                n_scored += 1
                n_below += int(r["overlap"] < ref["p2.5"])
    # Denominator is the count of cells that HAVE a reference — not n_shared + n_energy, which
    # silently drops the ANTI-ALIGNED cells and under-reports the sweep by 2.
    ax.set_title(f"F7 — above Haar everywhere; BELOW the unrelated-FT reference on "
                 f"{n_below} of {n_scored} cells\n"
                 f"the lineage writes avoid each other MORE than unrelated fine-tunes do (C4)",
                 fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.text(0.5, -0.24, "How to read: every point clears the dashed Haar line, so against Haar alone "
                        "all cells would read 'shared channel'. INSIDE the grey band is what any two "
                        "large writes produce. RINGED points fall below the band's lower edge (red) "
                        "— those two writes avoid each other more than two unrelated fine-tunes do, "
                        "which is the branch A12 lacked. Matrices are ordered within each pair by "
                        "their own reference median, so the band reads as a band.",
            transform=ax.transAxes, ha="center", fontsize=6.5)


def f3_k1(ax, k1):
    if not (k1 and "law" in k1):
        return _pending(ax, "F3 — K1 footprint vs survival", "K1 fit not complete")
    import numpy as np
    fps = k1["footprints"]
    r = str(k1["registered_rank"])
    facs = k1["law"]["faculties"]
    floor = None
    for f in facs:
        xs, ys = [], []
        for u, rec in fps.items():
            if f not in rec:
                continue
            xs.append(rec[f]["footprint"][r] if isinstance(rec[f]["footprint"], dict)
                      else rec[f]["footprint"])
            floor = rec[f]["chance_floor"][r] if isinstance(rec[f]["chance_floor"], dict) else floor
            ys.append(u)
        ax.plot(xs, np.arange(len(xs)), lw=0, marker="o", ms=4, alpha=0.75, label=f)
    if floor is not None:
        ax.axvline(float(floor), ls="--", color="0.35", lw=1.0, label="chance floor r/d")
    d = k1["law"]["delta_loo_footprint"]
    ci = k1["law"]["delta_loo_footprint_ci"]
    ax.set_xlabel(f"curvature footprint at r={r}")
    ax.set_ylabel("fine-tune (index)")
    n3 = (k1["law"].get("N3_permuted_labels") or {}).get("delta")
    sub = (f"{k1['law']['verdict']['verdict']} — but N3 (permuted labels) = {n3:+.4f}, "
           f"{abs(n3 / d['delta']):.0%} of the real effect: INSTRUMENT-SUSPECT"
           if n3 is not None else k1["law"]["verdict"]["verdict"])
    ax.set_title(f"F3 — K1: ΔLOO(footprint) = {d['delta']:+.4f}, "
                 f"CI [{ci['lo95']:+.4f}, {ci['hi95']:+.4f}]\n{sub}", fontsize=10)
    ax.legend(fontsize=7)


def f4_x(ax, x, c1=None):
    """Every measured cell, with BOTH shuffle nulls drawn on the quantity each one bounds.

    The emergent bar is read against the shuffle-EMERGENT null and the LOST bar against the
    shuffle-LOST null. Drawing one null across both bars would be the plotted form of the schema
    defect this arm actually shipped — `lost_fraction` bounded by a null belonging to a different
    quantity. Cells whose LOST null was never banked show a hollow marker, not a missing one.
    """
    if not (x and x.get("pairs")):
        return _pending(ax, "F4 — crosscoder fractions", "Arm X not complete")
    import numpy as np
    labels, em, lo, sh_e, sh_l = [], [], [], [], []
    for pname, prec in x["pairs"].items():
        for L, s in sorted(prec.get("sites", {}).items(), key=lambda kv: int(kv[0])):
            if "seeds" not in s:
                continue
            labels.append(f"{pname}\nL{L}")
            em.append(np.mean([q["emergent_fraction"] for q in s["seeds"]]))
            lo.append(np.mean([q["lost_fraction"] for q in s["seeds"]]))
            sh_e.append(np.mean([q["emergent_fraction"]
                                 for q in s.get("shuffle_null", [])] or [np.nan]))
            full = s.get("shuffle_null_full")
            sh_l.append(np.mean([q["lost_fraction"] for q in full]) if full else np.nan)
    xs = np.arange(len(labels))
    ax.bar(xs - 0.20, em, width=0.18, label="emergent (after-only)")
    ax.bar(xs + 0.02, lo, width=0.18, label="LOST (before-only) — the erasure signal")
    ax.plot(xs - 0.20, sh_e, lw=0, marker="_", ms=11, color="k",
            label="shuffle null — emergent")
    got = ~np.isnan(sh_l)
    if got.any():
        ax.plot(xs[got] + 0.02, np.asarray(sh_l)[got], lw=0, marker="_", ms=11, color="crimson",
                label="shuffle null — LOST (recovered)")
    if (~got).any():
        ax.plot(xs[~got] + 0.02, np.zeros((~got).sum()), lw=0, marker="x", ms=5, color="crimson",
                alpha=0.55, label="LOST null not banked for this cell")
    # The C1 sibling band is the null these readings are actually verdicted against (spec §6).
    # Without it the panel shows only the shuffle markers, which bound the instrument's range but
    # are not the decision boundary -- exactly the confusion C0 exposed for the identity band.
    if c1:
        sib = c1["pairs"]["sibling"]["sites"]
        for st, col, off_ in (("emergent_fraction", "tab:blue", -0.20),
                              ("lost_fraction", "darkorange", 0.02)):
            seg_x, seg_y = [], []
            for i, lab in enumerate(labels):
                L = lab.split("L")[-1]
                if L in sib and "seeds" in sib[L]:
                    hi = max(q[st] for q in sib[L]["seeds"])
                    seg_x += [i + off_ - 0.09, i + off_ + 0.09, None]
                    seg_y += [hi, hi, None]
            if seg_x:
                ax.plot(seg_x, seg_y, lw=1.6, color=col, alpha=0.95,
                        label=f"C1 sibling band, upper edge ({st.split('_')[0]})")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("fraction of LIVE latents")
    ax.set_title("F4 — crosscoder fractions per pair × site, against the C1 sibling band "
                 "(the decision boundary) and each pair's own shuffle null (the range check)",
                 fontsize=9.5)
    ax.legend(fontsize=6.5)
    n_pairs = len(x["pairs"])
    n_zero = sum(1 for p in x["pairs"].values()
                 if all(sum(q["lost_fraction"] for q in s["seeds"]) == 0
                        for s in p.get("sites", {}).values() if "seeds" in s))
    # Counted from the artifact, never typed in. An earlier hand-written caption said "six of
    # seven" for what is 5 of 6 pairs / 10 of 12 cells -- a miscount that survived several
    # retellings precisely because it was prose rather than a computed field.
    ax.text(0.5, -0.26, "How to read: a bar is REAL only if it clears the C1 band line of its own "
                        "colour at its own site — that band is two unrelated fine-tunes, and its "
                        "own shuffle fires at 0.13–0.32, so it could have been nonzero. The "
                        "grey/crimson markers are shuffle nulls: range checks, not the boundary. "
                        f"colour. {n_zero} of {n_pairs} pairs lose exactly zero, which no null "
                        "can make smaller; π0's LOST is the only bar with anything to bound.",
            transform=ax.transAxes, ha="center", fontsize=6.5)


def f6_core_magnitude(ax, d):
    """D1 — the complete 36-matrix sweep behind Arm S's 36/36 count.

    Per the house complete-sweep rule: every matrix, both component curves (what the VLA units
    added, what matched-random units add), and the ratio that is the actual finding.
    """
    if not (d and d.get("D1_core_magnitude")):
        return _pending(ax, "F6 — how much core did the VLA writes consume?", "derived not run")
    import numpy as np
    per = d["D1_core_magnitude"]["per_matrix"]
    xs = np.arange(len(per))
    vla = [p["vla_increase"] for p in per]
    rnd = [p["random_increase"] for p in per]
    ax.plot(xs, rnd, lw=1.1, marker="o", ms=3, color="0.35",
            label="two MATCHED-RANDOM units of equal energy")
    ax.plot(xs, vla, lw=1.1, marker="o", ms=3, color="crimson",
            label="the actual π0 + π0.5 writes")
    ax.set_yscale("log")
    ax.set_xlabel("q/k matrix (18 layers × {q_proj, k_proj})")
    ax.set_ylabel("increase in bottom-8 eigenvalue mean of G")
    m = d["D1_core_magnitude"]["vla_over_random_increase"]
    ax.set_title("F6 — the VLA hops enter the consensus core at %.0f%% the rate of matched-random writes\n"
                 "between-null denominator shown here: median %.3f (range %.3f–%.3f), %d/%d below 1.0 "
                 "— C3's same-base control agrees at 0.3883"
                 % (100 * m["median"], m["median"], m["min"], m["max"], m["n"], len(per)),
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.text(0.5, -0.24, "How to read: adding ANY two units raises these eigenvalues. The grey curve "
                        "is a BETWEEN-NULL difference (26-unit null mean minus 24-unit null mean), "
                        "which re-randomises the 24 shared units between its terms — critic finding "
                        "8. C3 replaces it with a same-base control that holds the real 24-FT Gram "
                        "fixed and randomises only the two VLA orientations; it lands at 0.3883 "
                        "against this panel's 0.3880, so the shape shown here is the right one.",
            transform=ax.transAxes, ha="center", fontsize=6.5)


def f5_k3(ax, k3):
    if not (k3 and k3.get("arms_result")):
        return _pending(ax, "F5 — n* ladders", "K3 not complete")
    for name, rec in k3["arms_result"].items():
        sw = rec.get("sweep")
        if not sw:
            continue
        ax.plot([s["rank"] for s in sw], [s["retention"] for s in sw], marker="o", ms=3,
                label=f"{name} (n*={rec.get('n_star')})")
    ax.axhline(k3.get("rho_break", 0.5), ls="--", color="0.35", lw=1.0, label="ρ_break")
    ax.set_xscale("log"); ax.set_xlabel("removal rank"); ax.set_ylabel("retention R(r)")
    ax.set_title("F5 — n* ladders (removal from the INSTALL write, A10)", fontsize=10)
    ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for k in ("s", "k1", "x", "k3", "derived", "c1"):
        ap.add_argument(f"--{k}", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    art = {k: _read(getattr(a, k)) for k in ("s", "k1", "x", "k3", "derived", "c1")}
    panels = [("F2_core", f2_core, art["s"]), ("F7_overlap", f7_overlap, art["s"]),
              ("F3_k1", f3_k1, art["k1"]), ("F4_x", lambda ax, d: f4_x(ax, d, art["c1"]), art["x"]),
              ("F5_k3", f5_k3, art["k3"]),
              ("F6_core_magnitude", f6_core_magnitude, art["derived"])]
    for name, fn, data in panels:
        fig, ax = plt.subplots(figsize=(9, 5.0))
        fn(ax, data)
        fig.text(0.99, 0.005, FOOT, ha="right", fontsize=5.5, color="0.4")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"tessera_{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {name}", flush=True)
    print("TESSERA_FIGURES_OK", flush=True)


if __name__ == "__main__":
    main()
