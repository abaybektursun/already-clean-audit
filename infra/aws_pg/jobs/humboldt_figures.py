"""HUMBOLDT — atlas assembly, coverage table, anomaly log, and the full-sweep figure suite (§10, §11).

Pulls every arm's output from s3://…/humboldt/, concatenates the per-arm atlas.jsonl into ONE
out/atlas.parquet, and derives:
  * out/coverage.parquet + coverage.md — every arm × cell: recorded / skipped-with-pivot / VOID
  * out/anomaly_log.parquet — MECHANICAL (§11): one row per (statistic, cell) where the real bank
    sits outside the [min, max] of its twins by more than DEVIATION_K × the twin spread. Filled by
    code, not judgement; no prose column exists.
  * out/figures/ — one full-sweep figure per swept parameter, with ALL points and ALL component
    curves (real + every twin + CI), and a provenance footer on each (house rule
    feedback_complete_sweep_visualization).

Zero interpretation (§0.1). Nothing here decides anything.

  pg run infra.aws_pg.jobs.humboldt_figures --type r7i.2xlarge --no-weights \
      --s3-prefix results/humboldt_figs --budget-min 120
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import humboldt_core as H

DEVIATION_K = 2.0                  # §7: "outside the [min, max] of its 4 twins by >2× twin spread"
REAL_BANKS = ("core32", "anti32")
BUCKET = "pg-lineage-769063704696"
ARM_PREFIXES = ("humboldt_a1", "humboldt_a2", "humboldt_a3", "humboldt_a4", "humboldt_a5",
                "humboldt_b1", "humboldt_b2", "humboldt_b3", "humboldt_b4")
# The frozen plan, for the coverage table. (arm, unit-of-work description, expected cell count expr)
PLANNED = {
    "A1": "7 types × 18 layers: spectrum + banks + leverage + PR + G1; 15 type-pairs × 7 bank-pairs × 18 layers of angles",
    "A2": "2 rope types × 18 layers × 7 banks: band coupling (+ ERLANGEN gradient sub-arm, pivot)",
    "A3": "7 types × 18 layers × (base + 3 init seeds): σ spectra, leverage, PR, bank-vs-σ angles",
    "A4": "≤8 external models × 7 types × 18 layers × 7 banks: ΔW energy fraction + ΔW spectra",
    "A5": "OPTIONAL closed-form-editor deltas (4h cap; skip permitted)",
    "B1": "5 models × 6-8 corpora × 18 layers × 3 sites × 42 banks: per-token e_U + class means",
    "B2": "B1 blocks → ACF, spectral peak, parallel/vqav2/position/period contrasts",
    "B3": "2 models × 3 tasks × 3 shot-deltas × 18 layers × 42 banks: ICL shift fraction",
    "B4": "18 layers × 21 banks × 6 ε × 3 seeds: attn JS + next-token KL + Δppl",
}


# B1's vectors are the per-token e_U blocks — ~10 GB that no figure reads (the figure suite plots
# scalars plus A2's 128-band coupling curves). Skip them here; `scripts/humboldt_verify.py` does a
# full sync when it checks that every vector_ref resolves.
HEAVY_VECTOR_ARMS = ("humboldt_b1",)


def _sync(dest, prefix):
    """Pull an arm's output. Returns True only if it produced atlas rows; a skipped arm still leaves
    its manifest and gates behind, which the coverage table reads."""
    cmd = ["aws", "s3", "sync", f"s3://{BUCKET}/humboldt/{prefix}", str(dest / prefix),
           "--region", "us-east-1"]
    if prefix in HEAVY_VECTOR_ARMS:
        cmd += ["--exclude", "vectors/*"]
    subprocess.run(cmd, capture_output=True, text=True)
    f = dest / prefix / "atlas.jsonl"
    return f.exists() and f.stat().st_size > 0


def _bank_identity(bid):
    """(identity, subfamily) for a bank_id.

    Plain ids compare directly. COMPOSITE ids — `A|B`, used by every angle statistic — must not be
    thrown away: A1's cross-type angles and A3's bank-vs-σ angles carry their twins in exactly that
    column. Two shapes occur and they group differently:
      * symmetric `A|A` (A1's cross-type diagonal): the family is the diagonal, identity A.
      * `A|R` against a fixed reference R (A3's `core32|base_bottomsigma32`): the family is R,
        identity A.
    Off-diagonal pairs such as `core32|anti32` land alone in their family and therefore produce no
    comparison — which is correct: they are cross-object readings with no twin analogue, and
    inventing one would manufacture anomalies.
    """
    if bid in H.BANK_IDS:
        return bid, ""
    if isinstance(bid, str) and "|" in bid:
        a, b = bid.split("|", 1)
        if a in H.BANK_IDS:
            return a, ("DIAG" if a == b else b)
    return None, None


def _anomaly_log(df):
    """§11's mechanical schema. Scalar-valued rows only (a vector has no scalar deviation)."""
    import numpy as np
    import pandas as pd
    d = df[df["value"].notna() & df["bank_id"].notna()].copy()
    ident = d["bank_id"].map(lambda b: _bank_identity(b))
    d["_ident"] = [x[0] for x in ident]
    d["_subfam"] = [x[1] for x in ident]
    d = d[d["_ident"].notna()].copy()
    keys = ["arm", "statistic", "model_id", "layer", "type", "corpus_id", "token_class", "seed",
            "_subfam"]
    for k in keys:
        d[k] = d[k].astype("object").where(d[k].notna(), "")
    rows = []
    for key, g in d.groupby(keys, dropna=False):
        vals = dict(zip(g["_ident"], g["value"]))
        raw = dict(zip(g["_ident"], g["bank_id"]))
        for real in REAL_BANKS:
            if real not in vals:
                continue
            twins = {b: v for b, v in vals.items() if b != real}
            if len(twins) < 2:
                continue
            lo, hi = min(twins.values()), max(twins.values())
            spread = hi - lo
            v = vals[real]
            out = (v - hi) if v > hi else ((lo - v) if v < lo else 0.0)
            mult = float(out / spread) if spread > 0 else (np.inf if out > 0 else 0.0)
            if mult > DEVIATION_K:
                rows.append({**dict(zip(keys, key)), "bank_id": real,
                             "bank_id_raw": raw[real], "real_value": float(v),
                             "twin_min": float(lo), "twin_max": float(hi),
                             "twin_spread": float(spread), "deviation_multiple": mult,
                             "n_twins": len(twins),
                             "twins": json.dumps({k2: float(v2) for k2, v2 in twins.items()},
                                                 sort_keys=True)})
    df = pd.DataFrame(rows, columns=(keys + ["bank_id", "bank_id_raw", "real_value", "twin_min",
                                             "twin_max", "twin_spread", "deviation_multiple",
                                             "n_twins", "twins"]))
    # The grouping keys are IDENTIFIERS in this log, and they were filled with "" where null, so a
    # column like `seed` or `layer` holds both "" and ints — object dtype that pyarrow refuses.
    # Cast every key to str: lossless for a log whose purpose is to point at atlas rows.
    for k in keys + ["bank_id", "bank_id_raw", "twins"]:
        df[k] = df[k].astype(str)
    for k in ("real_value", "twin_min", "twin_max", "twin_spread", "deviation_multiple"):
        df[k] = pd.to_numeric(df[k], errors="coerce").astype("float64")
    df["n_twins"] = pd.to_numeric(df["n_twins"], errors="coerce").astype("Int64")
    return df


def _footer(fig, provenance):
    fig.text(0.005, 0.005, provenance, fontsize=5, ha="left", va="bottom", family="monospace",
             alpha=0.75)


def _sweep_figure(d, x, series, title, xlabel, ylabel, path, provenance, logx=False, logy=False,
                  facet=None, chance=None):
    """One full-sweep panel set: every point of the sweep, every component curve (real + all twins),
    CI bands where present. facet → one subplot per facet value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    facets = sorted(d[facet].dropna().unique()) if facet else [None]
    n = len(facets)
    if n == 0 or d.empty:
        return None
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.4 * nrow), squeeze=False)
    for i, fv in enumerate(facets):
        ax = axes[i // ncol][i % ncol]
        sub = d if fv is None else d[d[facet] == fv]
        for name, g in sub.groupby(series):
            g = g.sort_values(x)
            # the REAL bank is drawn heavy and its twins light, whatever composite the series label
            # is built from (`core32`, `core32|core32`, `core32 · 16shot`) — a reader must be able
            # to tell the object from its chance column at a glance.
            head = str(name).split(" · ")[0].split("|")[0]
            style = (dict(lw=1.8, marker="o", ms=3) if head in REAL_BANKS
                     else dict(lw=1.0, ms=2, marker=".", alpha=0.75))
            ln, = ax.plot(g[x], g["value"], label=str(name), **style)
            if g["ci_lo"].notna().any():
                ax.fill_between(g[x], g["ci_lo"], g["ci_hi"], alpha=0.15, color=ln.get_color(),
                                linewidth=0)
        if chance is not None and chance in sub.columns and sub[chance].notna().any():
            ax.axhline(float(sub[chance].dropna().iloc[0]), color="k", ls=":", lw=0.9,
                       label="chance k/n")
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{title}" if fv is None else f"{title} — {fv}", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0][0].legend(fontsize=6, ncol=2, framealpha=0.6)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _footer(fig, provenance)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return str(path.with_suffix(".png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="humboldt_figs")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    import numpy as np
    import pandas as pd

    src = H.cache_dir(out, "arms")
    present, frames = {}, []
    for pref in ARM_PREFIXES:
        ok = _sync(src, pref)
        present[pref] = ok
        if ok:
            df = pd.read_json(src / pref / "atlas.jsonl", lines=True)
            df["_source"] = pref
            frames.append(df)
            print(f"[figs] {pref}: {len(df)} rows", flush=True)
        else:
            print(f"[figs] {pref}: ABSENT", flush=True)
    assert frames, "no arm produced an atlas — nothing to assemble"
    atlas = pd.concat(frames, ignore_index=True)
    # Concatenating arms leaves mixed dtypes: a column that is all-null in one arm and integer in
    # another lands as `object`, which pyarrow refuses to infer. Coerce to the frozen schema's
    # types explicitly — nullable Int64 for the integer columns, float64 for the numeric ones,
    # str for the rest — so the combined parquet is written with the same types every arm used.
    for c in ("layer", "n", "seed"):
        atlas[c] = pd.to_numeric(atlas[c], errors="coerce").astype("Int64")
    for c in ("value", "ci_lo", "ci_hi"):
        atlas[c] = pd.to_numeric(atlas[c], errors="coerce").astype("float64")
    for c in ("arm", "model_id", "model_sha", "type", "bank_id", "statistic", "vector_ref",
              "corpus_id", "item_id", "token_class", "flags", "run_id", "commit_sha", "tag",
              "_source"):
        atlas[c] = atlas[c].astype("object").where(atlas[c].notna(), None)
    atlas.to_parquet(out / "atlas.parquet", index=False)

    # ---- flags → columns the figures can facet/annotate on, without touching the frozen schema.
    fl = atlas["flags"].apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    atlas["chance"] = fl.apply(lambda d: d.get("chance"))
    atlas["site"] = fl.apply(lambda d: d.get("site"))
    atlas["eps_frac"] = fl.apply(lambda d: d.get("eps_frac"))
    atlas["shots"] = fl.apply(lambda d: d.get("shots"))
    atlas["mqa_kernel_caveat"] = fl.apply(lambda d: bool(d.get("mqa_kernel_caveat", False)))

    # ---- anomaly log (§11).
    anom = _anomaly_log(atlas)
    anom.to_parquet(out / "anomaly_log.parquet", index=False)
    print(f"[figs] anomaly log: {len(anom)} rows (deviation > {DEVIATION_K}× twin spread)",
          flush=True)

    # ---- coverage table (§6): what each arm actually recorded, and what is missing.
    cov = []
    for pref in ARM_PREFIXES:
        arm = pref.split("_")[-1].upper()
        if not present[pref]:
            # §6 requires "skipped-with-pivot-reason", not just a hole. An arm that executed its
            # pre-specified pivot writes a manifest saying so even though it wrote no atlas rows.
            mf = src / pref / "manifest.json"
            detail = {"note": "no atlas.jsonl under humboldt/" + pref}
            status = "ABSENT"
            if mf.exists():
                m = json.loads(mf.read_text())
                status = m.get("status", "ABSENT")
                detail = {k: m[k] for k in ("status", "skip_reason", "pivot_triggered", "reach")
                          if k in m}
                if status == "SKIPPED":
                    status = "SKIPPED_WITH_PIVOT_REASON"
            gp = src / pref / "gates.json"
            gj = json.loads(gp.read_text()) if gp.exists() else {}
            cov.append({"arm": arm, "planned": PLANNED.get(arm, ""), "status": status,
                        "n_rows": 0, "n_cells": 0,
                        "detail": json.dumps({**detail, "gates": {k: v.get("ok") for k, v in gj.items()}},
                                             sort_keys=True, default=str)})
            continue
        d = atlas[atlas["_source"] == pref]
        cells = d.groupby(["statistic"]).size().to_dict()
        gates_p = src / pref / "gates.json"
        gates = json.loads(gates_p.read_text()) if gates_p.exists() else {}
        # A gate that was RECORDED rather than REQUIRED reading false is a flagged observation the
        # spec asks for (A2's unreachable sub-arm, B3's weak contrast), not a failure. Calling both
        # "GATE_FAIL" says something stronger than the evidence supports.
        failed = [k for k, v in gates.items() if not v.get("ok") and v.get("_required", True)]
        flagged = [k for k, v in gates.items() if not v.get("ok") and not v.get("_required", True)]
        status = ("RECORDED_WITH_REQUIRED_GATE_FAIL" if failed
                  else "RECORDED_WITH_FLAGGED_OBSERVATION" if flagged else "RECORDED")
        cov.append({"arm": arm, "planned": PLANNED.get(arm, ""), "status": status,
                    "n_rows": int(len(d)), "n_cells": int(len(cells)),
                    "detail": json.dumps({"statistics": cells, "required_gates_failed": failed,
                                          "flagged_observations": flagged}, sort_keys=True)})
    covdf = pd.DataFrame(cov)
    covdf.to_parquet(out / "coverage.parquet", index=False)
    (out / "coverage.md").write_text(
        "# HUMBOLDT coverage table\n\n"
        "arm · planned unit of work · status · rows · distinct statistics\n\n"
        + "\n".join(f"- **{r['arm']}** — {r['planned']}\n  - status: `{r['status']}`, "
                    f"rows: {r['n_rows']}, statistics: {r['n_cells']}\n  - {r['detail'][:600]}"
                    for r in cov) + "\n")

    prov = (f"HUMBOLDT atlas · commit {H.commit_sha()[:12]} · rows={len(atlas)} · "
            f"arms={[k for k, v in present.items() if v]} · DISCOVERY-grade, no verdicts")
    figs = []

    def _f(name):
        return out / "figures" / name

    # ---------------- A1
    d = atlas[(atlas["statistic"] == "leverage_max") & (atlas["_source"] == "humboldt_a1")]
    figs.append(_sweep_figure(d, "layer", "bank_id", "A1 · coordinate-leverage max", "layer",
                              "max ℓ_i = max ‖U[i,:]‖²", _f("A1_leverage_max"), prov,
                              facet="type"))
    d = atlas[(atlas["statistic"] == "participation_ratio") & (atlas["_source"] == "humboldt_a1")]
    figs.append(_sweep_figure(d, "layer", "bank_id", "A1 · leverage participation ratio", "layer",
                              "PR", _f("A1_participation_ratio"), prov, facet="type", logy=True))
    for k in (8, 32, 128):
        d = atlas[atlas["statistic"].isin([f"gram_bottom{k}_mean", f"gram_top{k}_mean"])
                  & (atlas["_source"] == "humboldt_a1")]
        if not d.empty:
            figs.append(_sweep_figure(d, "layer", "statistic",
                                      f"A1 · consensus-Gram bottom/top-{k} mean eigenvalue",
                                      "layer", "mean eigenvalue", _f(f"A1_gram_k{k}"), prov,
                                      facet="type", logy=True))
    d = atlas[(atlas["statistic"] == "cross_type_principal_angle_mean")]
    if not d.empty:
        d = d.assign(pair=d["type"])
        figs.append(_sweep_figure(d, "layer", "bank_id", "A1 · cross-type principal angle (mean)",
                                  "layer", "angle (rad)", _f("A1_cross_type_angles"), prov,
                                  facet="pair"))
    # ---------------- A2
    for st, yl in (("rope_coupling_participation_ratio", "PR over rotary bands"),
                   ("rope_coupling_max", "max band coupling")):
        d = atlas[atlas["statistic"] == st]
        figs.append(_sweep_figure(d, "layer", "bank_id", f"A2 · {st}", "layer", yl,
                                  _f(f"A2_{st}"), prov, facet="type"))
    # the coupling curves themselves are vectors — plot them from the banked .npy.
    d = atlas[(atlas["statistic"] == "rope_band_coupling")]
    if not d.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for t in sorted(d["type"].unique()):
            sub = d[d["type"] == t]
            layers = sorted(sub["layer"].unique())
            ncol = 6
            nrow = int(np.ceil(len(layers) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.2 * nrow), squeeze=False)
            for i, L in enumerate(layers):
                ax = axes[i // ncol][i % ncol]
                for _, r in sub[sub["layer"] == L].iterrows():
                    v = np.load(src / r["_source"] / r["vector_ref"])
                    ax.plot(v, lw=1.6 if r["bank_id"] in REAL_BANKS else 0.8,
                            alpha=1.0 if r["bank_id"] in REAL_BANKS else 0.7,
                            label=r["bank_id"] if i == 0 else None)
                ax.set_title(f"L{L}", fontsize=8)
                ax.set_yscale("log")
                ax.tick_params(labelsize=6)
                ax.grid(alpha=0.2, lw=0.3)
            axes[0][0].legend(fontsize=5, ncol=2)
            for j in range(len(layers), nrow * ncol):
                axes[j // ncol][j % ncol].axis("off")
            fig.suptitle(f"A2 · RoPE band coupling — {t} (x = rotary band, y = energy fraction)",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0.02, 1, 0.97))
            _footer(fig, prov)
            fig.savefig(_f(f"A2_coupling_curves_{t}").with_suffix(".png"), dpi=150)
            fig.savefig(_f(f"A2_coupling_curves_{t}").with_suffix(".pdf"))
            plt.close(fig)
            figs.append(str(_f(f"A2_coupling_curves_{t}").with_suffix(".png")))
    # ---------------- A3
    for st in ("bank_vs_base_sigma_angle_mean", "bank_vs_init_sigma_angle_mean",
               "init_cross_type_angle_mean"):
        d = atlas[atlas["statistic"] == st]
        figs.append(_sweep_figure(d, "layer", "bank_id", f"A3 · {st}", "layer", "angle (rad)",
                                  _f(f"A3_{st}"), prov, facet="type"))
    # ---------------- A4
    d = atlas[atlas["statistic"] == "ext_dW_bank_energy_fraction"]
    if not d.empty:
        for mid in sorted(d["model_id"].unique()):
            figs.append(_sweep_figure(d[d["model_id"] == mid], "layer", "bank_id",
                                      f"A4 · external ΔW energy in bank — {mid}", "layer",
                                      "‖ΔW U‖²/‖ΔW‖²", _f(f"A4_{mid.replace('/', '_')}"), prov,
                                      facet="type", logy=True, chance="chance"))
    # ---------------- B1 / B2
    d = atlas[atlas["statistic"] == "e_U_mean"]
    if not d.empty:
        for mid in sorted(d["model_id"].unique()):
            for cid in sorted(d[d["model_id"] == mid]["corpus_id"].dropna().unique()):
                sub = d[(d["model_id"] == mid) & (d["corpus_id"] == cid)]
                figs.append(_sweep_figure(sub, "layer", "bank_id",
                                          f"B1 · mean e_U — {mid} / {cid}", "layer",
                                          "e_U = ‖UUᵀh‖²/‖h‖²",
                                          _f(f"B1_eU_{mid}_{cid}"), prov, facet="type",
                                          logy=True, chance="chance"))
    for st, yl in (("parallel_paired_delta_vs_reference", "Δ e_U vs reference language"),
                   ("vqav2_real_minus_gray", "Δ e_U (real − gray-128)"),
                   ("repetition_period_response", "Δ e_U (periodic − shuffled)"),
                   ("vqav2_image_minus_text", "Δ e_U (image − text)")):
        d = atlas[atlas["statistic"] == st]
        if not d.empty:
            for mid in sorted(d["model_id"].unique()):
                figs.append(_sweep_figure(d[d["model_id"] == mid], "layer", "bank_id",
                                          f"B2 · {st} — {mid}", "layer", yl,
                                          _f(f"B2_{st}_{mid}"), prov, facet="type"))
    # ---------------- B3
    d = atlas[atlas["statistic"].isin(["icl_shift_bank_fraction",
                                       "cross_query_shift_bank_fraction"])]
    if not d.empty:
        for mid in sorted(d["model_id"].unique()):
            for task in sorted(d["corpus_id"].dropna().unique()):
                sub = d[(d["model_id"] == mid) & (d["corpus_id"] == task)]
                if sub.empty:
                    continue
                sub = sub.assign(series=sub["bank_id"] + " · " + sub["token_class"].astype(str))
                figs.append(_sweep_figure(sub, "layer", "series",
                                          f"B3 · ICL shift in bank — {mid} / {task}", "layer",
                                          "‖UUᵀΔh‖²/‖Δh‖²", _f(f"B3_{mid}_{task}"), prov,
                                          facet="type", logy=True, chance="chance"))
    # ---------------- B4
    for st, yl in (("perturb_attn_js", "attention-map JS (nats)"),
                   ("perturb_next_token_kl", "next-token KL (nats)"),
                   ("perturb_dppl", "Δ perplexity")):
        d = atlas[atlas["statistic"] == st]
        if not d.empty:
            figs.append(_sweep_figure(d, "eps_frac", "bank_id", f"B4 · {st}",
                                      "ε as a fraction of ‖h‖", yl, _f(f"B4_{st}"), prov,
                                      facet="layer", logx=True))

    figs = [f for f in figs if f]
    (out / "figures_index.json").write_text(json.dumps(figs, indent=2))
    (out / "figs_summary.json").write_text(json.dumps(
        {"n_rows": int(len(atlas)), "arms_present": present, "n_anomalies": int(len(anom)),
         "n_figures": len(figs), "deviation_k": DEVIATION_K,
         "commit_sha": H.commit_sha()}, indent=2))
    H.s3_sync(out, f"humboldt/{a.run_id}")
    print("=" * 78)
    print(f"HUMBOLDT_FIGS_OK rows={len(atlas)} anomalies={len(anom)} figures={len(figs)} "
          f"arms={ {k: v for k, v in present.items()} }")


if __name__ == "__main__":
    main()
