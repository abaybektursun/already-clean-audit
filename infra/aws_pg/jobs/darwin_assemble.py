"""DARWIN assembly (§10) — one atlas, one coverage table, one surprise log, the figure suite.

Pulls every DARWIN arm prefix from s3://…/darwin/, concatenates the per-arm atlas.jsonl into ONE
`out/atlas.parquet` (both registers, schema §3), and derives:

  out/coverage.parquet + coverage.md   every arm x cell: planned / pivot-skipped / VOID-expected,
                                       joined against what the atlas ACTUALLY contains, so a
                                       silently-shrunk arm shows up as a row-count of zero rather
                                       than as an absence nobody notices (§0.1).
  out/surprise_log.parquet             every arm's mechanical surprise rows, concatenated (§10).
  out/graduation.parquet               Register-B verdicts + formula inputs (§5); empty until the
                                       GRAD arms land, and recorded as empty rather than omitted.
  out/figures/                         per-sweep complete figures: all points, all control
                                       columns, CI bands, provenance footers (§10, house rule).

Zero interpretation (§0.1). Nothing here decides anything; the only verdicts written are the
three §5 literals computed by darwin_core.graduate from recorded rows.

  pg run infra.aws_pg.jobs.darwin_assemble --type r7i.2xlarge --no-weights \
      --s3-prefix results/darwin_figs --budget-min 120
"""
import argparse
import json
import pathlib
import subprocess

from infra.pg import darwin_core as D
from infra.pg import darwin_figures as F
from infra.pg import darwin_program as P
from infra.pg import humboldt_core as H

# Arm prefixes under s3://…/darwin/. Smoke prefixes are excluded deliberately: a smoke's rows are
# real rows from a deliberately truncated cell set, and concatenating them would inflate coverage.
SKIP_SUFFIXES = ("_smoke",)
SKIP_SUBSTRINGS = ("smoke",)     # `_smoke2` etc. escape a suffix test; a real arm is never named
                                 # with "smoke" in it, so the substring rule is the safe one
NON_ARM_PREFIXES = ("banks", "steps", "darwin_figs")


def discover_arm_prefixes():
    """List s3://…/darwin/ and return the arm run prefixes present."""
    r = subprocess.run(["aws", "s3", "ls", f"{D.DARWIN_S3}/", "--region", "us-east-1"],
                       check=True, capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "PRE":
            name = parts[1].rstrip("/")
            if (name in NON_ARM_PREFIXES or name.endswith(SKIP_SUFFIXES)
                    or any(x in name for x in SKIP_SUBSTRINGS)):
                continue
            out.append(name)
    return sorted(out)


def sync_arms(dest, prefixes, with_vectors: bool):
    """Pull each arm. Vector blocks are large and only some figures read them."""
    got = []
    for p in prefixes:
        # An ARM writes its artifacts at the prefix ROOT (D.s3_sync of its out/ dir). A PIPELINE
        # runner writes `job.log` and `out/` there instead, and those logs run to hundreds of MB.
        # Excluding both makes a non-arm prefix free to walk rather than a download, and costs an
        # arm nothing because an arm has neither.
        cmd = ["aws", "s3", "sync", f"{D.DARWIN_S3}/{p}", str(pathlib.Path(dest) / p),
               "--region", "us-east-1", "--exclude", "job.log", "--exclude", "out/*"]
        if not with_vectors:
            cmd += ["--exclude", "vectors/*"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        f = pathlib.Path(dest) / p / "atlas.jsonl"
        if f.exists() and f.stat().st_size > 0:
            got.append(p)
        else:
            print(f"[assemble] {p}: no atlas rows (arm not complete)", flush=True)
    return got


def concat_atlas(dest, prefixes, out):
    import pandas as pd
    frames = []
    for p in prefixes:
        df = pd.read_json(pathlib.Path(dest) / p / "atlas.jsonl", lines=True)
        missing = set(D.ATLAS_COLUMNS) - set(df.columns)
        assert not missing, f"{p}: atlas is missing DARWIN columns {sorted(missing)}"
        frames.append(df[list(D.ATLAS_COLUMNS)])
    assert frames, "no arm produced atlas rows"
    all_df = pd.concat(frames, ignore_index=True)
    path = pathlib.Path(out) / "atlas.parquet"
    all_df.to_parquet(path, index=False)
    return all_df, {"path": str(path), "n_rows": int(len(all_df)),
                    "n_arms": int(all_df["arm"].nunique())}


def concat_surprises(dest, prefixes, out):
    import pandas as pd
    frames = []
    for p in prefixes:
        f = pathlib.Path(dest) / p / "surprise_log.jsonl"
        if f.exists() and f.stat().st_size:
            frames.append(pd.read_json(f, lines=True))
    path = pathlib.Path(out) / "surprise_log.parquet"
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["arm", "expected_ref", "cell", "statistic", "observed", "expected",
                 "control_spread", "deviation", "threshold", "is_surprise", "claimed_by"])
    df.to_parquet(path, index=False)
    return df, {"path": str(path), "n_rows": int(len(df)),
                "n_surprises": int(df["is_surprise"].sum()) if len(df) else 0}


def coverage(all_df, out):
    """§3's enumeration, joined against what actually landed."""
    import pandas as pd
    # Every arm value in the atlas must be enumerated. An atlas arm the coverage table does not
    # know about is coverage that silently does not exist (§0.1) — it was G1, which writes rows
    # but is a preflight gate, and the row-count mismatch is how it surfaced.
    unregistered = sorted(set(all_df["arm"].dropna()) - set(P.ARMS))
    assert not unregistered, (
        f"atlas contains arms absent from the registry: {unregistered}. Add them to "
        f"infra/pg/darwin_program.py so the coverage enumeration accounts for their rows")
    seen = all_df.groupby("arm").agg(n_rows=("statistic", "size"),
                                     n_statistics=("statistic", "nunique"),
                                     n_banks=("bank_id", "nunique")).to_dict("index")
    rows = []
    # `status` here is the REGISTRY's static field (always "planned" — the registry is the
    # frozen enumeration, not a run log). Whether an arm ran is the derived `executed` column.
    # Naming the column `status` next to a row with 7825 landed rows reads as "planned" being the
    # arm's state, which it is not.
    for r in P.coverage_table():
        s = seen.get(r["arm"], {})
        rows.append({**r, "n_rows": int(s.get("n_rows", 0)),
                     "n_statistics": int(s.get("n_statistics", 0)),
                     "n_banks": int(s.get("n_banks", 0)),
                     "executed": bool(s.get("n_rows", 0) > 0)})
    df = pd.DataFrame(rows)
    df.to_parquet(pathlib.Path(out) / "coverage.parquet", index=False)
    md = ["# DARWIN — coverage, filled (§3)", "",
          f"{int(df['executed'].sum())} of {len(df)} arms have landed rows. "
          "`status` is the frozen plan; `n_rows` is what the atlas actually contains.", "",
          "| arm | wave | register | registry_status | executed | n_rows | n_statistics | n_banks |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['arm']} | {r['wave']} | {r['register']} | {r['status']} | "
                  f"{'yes' if r['executed'] else 'no'} | {r['n_rows']} | {r['n_statistics']} | "
                  f"{r['n_banks']} |")
    (pathlib.Path(out) / "coverage.md").write_text("\n".join(md) + "\n")
    return df


# (arm, statistic, kind, group_cols, x, reference_fn, title, ctrl) — the figure plan, data-driven
# so a new arm adds a row here rather than a new code path. `ctrl` says whether `bank_id` holds
# the BANK BATTERY (so control-completeness applies) or arms/schemes (so it does not): asserting
# the twin set on a quantization-arm figure would demand curves that have no meaning there.
FIGURE_PLAN = [
    ("W0-0", "bank_derivation_max_principal_angle_rad", "heatmap", (), None, None,
     "W0-0 bank re-derivation: max principal angle vs HUMBOLDT's banked bottomsigma32", True),
    ("W0-0", "bank_energy_fraction", "sweep", ("type",), "layer", "haar",
     # ctrl=False: this statistic exists only for topsigma32 — W0-0 is the arm that DERIVES it,
     # so a one-curve figure is the honest content, not a missing control set.
     "W0-0 base-matrix energy on topsigma32", False),
    ("W0-1a", "leverage_spike_count", "sweep", ("type",), "layer", None,
     "W0-1a leverage spikes (> 5x median) per bank", True),
    ("W0-1a", "leverage_participation_ratio", "sweep", ("type",), "layer", None,
     "W0-1a leverage participation ratio per bank", True),
    ("W0-2", "cross_layer_mean_cos2_summary", "bars", ("type",), None, "haar",
     "W0-2 cross-layer subspace agreement (mean cos^2)", True),
    ("W0-2", "cross_layer_shared_coordinate_fraction", "bars", ("type",), None, None,
     # ctrl=False for a SUBSTANTIVE reason, recorded rather than waived: this statistic is
     # undefined for a bank with no spiky coordinates, and W0-1a measured exactly 0.00 spikes for
     # haar32_s{0,1,2} and complement32. Their absence from the curve set is the finding; the
     # companion `cross_layer_shared_coordinate_fraction_undefined` rows carry their status.
     "W0-2 cross-layer shared spiky-coordinate fraction", False),
    ("W0-1b", "outlier_coord_count", "sweep_nobank", ("model_id",), "layer", None,
     "W0-1b massive-activation coordinates (> 5x median |h_i|) per layer", False),
    ("W0-1c", "outlier_overlap_jaccard", "bars", ("model_id",), None, None,
     "W0-1c bank-spike vs activation-outlier overlap (Jaccard)", True),
    ("W0-1c", "outlier_overlap_enrichment", "bars", ("model_id",), None, 1.0,
     # ctrl=False for the same reason as W0-2's shared fraction (defect D5): enrichment divides
     # by the expected intersection, which is 0 for a bank with no spiky coordinates, so it is
     # recorded only where defined. `outlier_overlap_jaccard` above KEEPS ctrl=True — Jaccard is
     # defined for every bank, which is precisely why D5 made it the statistic the twin band and
     # the arm's REFUTER are decided on.
     "W0-1c overlap enrichment over the hypergeometric expectation", False),
    ("W0-7", "ood_band_deviation_auc", "sweep", ("model_id", "corpus_id"), "layer", 0.5,
     # ctrl=False, and this one is a RECORDED COVERAGE GAP rather than a convenience: W0-7
     # re-analyses HUMBOLDT's B1 series, which was computed before topsigma32 existed, so its twin
     # set is B1's seven banks. The structured top-k control is genuinely absent from this arm and
     # asserting it would only hide that.
     "W0-7 OOD band-deviation AUC vs wikitext", False),
    # --- census arms: bank_id is the bank battery, but only core/anti/haar are built per base.
    ("W0-3", "census_leverage_spike_count", "bars", ("model_id",), None, None,
     "W0-3 census: >5x-median leverage spikes per bank", False),
    ("W0-3", "census_leverage_participation_ratio", "bars", ("model_id",), None, None,
     "W0-3 census: leverage participation ratio (lower = spikier)", False),
    ("W0-3", "census_bottom_eigenvalue_fraction", "bars", ("model_id",), None, None,
     "W0-3 census: bottom-32 share of consensus write energy", False),
    ("W0-4", "census_leverage_spike_count", "bars", ("model_id",), None, None,
     "W0-4 scale census (gemma-7b): leverage spikes per bank", False),
    ("W0-4", "census_bottom_eigenvalue_fraction", "bars", ("model_id",), None, None,
     "W0-4 scale census: bottom-32 share of consensus write energy", False),
    # --- quantization / surgery arms: bank_id holds the ARM id.
    ("W1-N4", "wikitext_ppl", "sweep", (), "dose", None,
     "W1-N4 wikitext ppl vs quantizer bit-depth, per arm", False),
    ("W1-N4", "wikitext_ppl_contrast_vs_incumbent", "sweep", (), "dose", 0.0,
     "W1-N4 paired contrast vs incumbent b (positive = worse), 95% CI", False),
    ("W1-N4", "core_action_error", "sweep", ("type",), "layer", None,
     "W1-N4 core action error under bulk quantization (T1 exactness)", False),
    ("W1-N4", "kv_routed_activation_error_ratio", "bars", ("dose",), None, 1.0,
     "W1-N4 KV routed-vs-ambient activation error ratio (T7)", False),
    ("W1-O", "wikitext_ppl", "sweep", (), "dose", None,
     "W1-O wikitext ppl vs sparsity: doubly-cold vs magnitude", False),
]


def _slug(*parts):
    return "__".join(str(p).replace("/", "-").replace(" ", "_") for p in parts if p is not None)


def render_figures(all_df, out, strict_controls: bool):
    import numpy as np
    F.use_agg()
    made = {}
    for arm, stat, kind, group_cols, x, ref, title, ctrl in FIGURE_PLAN:
        sub = all_df[(all_df["arm"] == arm) & (all_df["statistic"] == stat)]
        if not len(sub):
            made[_slug(arm, stat)] = {"skipped": "no rows", "arm": arm, "statistic": stat}
            continue
        groups = ([((), sub)] if not group_cols
                  else list(sub.groupby(list(group_cols), dropna=True)))
        for key, g in groups:
            key = key if isinstance(key, tuple) else (key,)
            name = _slug(arm, stat, *key) + ".png"
            path = pathlib.Path(out) / "figures" / name
            t = f"{title}  [{' / '.join(str(k) for k in key)}]" if key else title
            reference, rlabel = None, None
            if ref == "haar":
                ty = key[0] if group_cols and group_cols[0] == "type" else None
                if ty in H.IN_DIM:
                    reference, rlabel = H.BANK_K / H.IN_DIM[ty], f"Haar k/n = 32/{H.IN_DIM[ty]}"
            elif isinstance(ref, float):
                reference, rlabel = ref, f"reference {ref}"
            try_ctrl = ctrl and kind in ("sweep", "bars")
            if kind == "heatmap":
                info = F.layer_type_heatmap(
                    g[g["bank_id"] == "bottomsigma32"] if (g["bank_id"] == "bottomsigma32").any()
                    else g, value="value", title=t, out_path=path,
                    cbar_label="max principal angle (rad)")
                made[name] = {"kind": kind, "arm": arm, "statistic": stat, **info}
                continue
            if kind == "sweep":
                drawn = F.bank_sweep(g, x=x, y="value", title=t, xlabel=x, ylabel=stat,
                                     out_path=path, reference=reference, reference_label=rlabel)
            elif kind == "sweep_nobank":
                # No bank dimension (activation-side): group by layer, one curve per site.
                gg = g.copy()
                gg["bank_id"] = gg["flags"].map(lambda s: json.loads(s).get("site", "site"))
                drawn = F.bank_sweep(gg, x=x, y="value", title=t, xlabel=x, ylabel=stat,
                                     out_path=path, reference=reference, reference_label=rlabel)
                try_ctrl = False
            else:
                drawn = F.bank_bars(g, y="value", title=t, ylabel=stat, out_path=path,
                                    reference=reference, reference_label=rlabel)
            if strict_controls and try_ctrl:
                F.assert_controls_drawn(drawn, figure=name)
            made[name] = {"kind": kind, "arm": arm, "statistic": stat,
                          "banks": sorted(str(b) for b in drawn), "n_rows": int(len(g)),
                          "group": [str(k) for k in key]}
    # Cross-arm view: how many rows each (arm, register) contributed.
    import matplotlib.pyplot as plt
    counts = all_df.groupby(["wave", "arm"]).size().sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * max(4, len(counts)) + 1.2), dpi=170)
    ax.barh([f"{w}/{a}" for w, a in counts.index], counts.to_numpy(), color="#2563EB", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("atlas rows (log)", fontsize=8)
    ax.set_title("DARWIN atlas: rows per arm", fontsize=9)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    F.provenance_footer(fig, all_df)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(pathlib.Path(out) / "figures" / "atlas_rows_per_arm.png")
    plt.close(fig)
    made["atlas_rows_per_arm.png"] = {"kind": "summary", "n_arms": int(len(counts))}
    np.seterr(all="warn")
    return made


def graduation(all_df, out):
    """§10 out/graduation.parquet — arm x criteria x verdict, with every formula input.

    Evaluated MECHANICALLY from recorded rows by darwin_core.graduate; nothing here decides
    anything. A Register-B arm that has landed a paired contrast against its named incumbent gets
    a verdict; one that has not is absent, and the summary says how many of each.

    COMPOSITION is W2-F's interaction matrix, which has not run. Its absence is carried as
    `composition_evaluable: false` rather than silently passed — an arm cannot GRADUATE on two of
    three criteria, so the verdicts below are NOT-GRADUATED-pending-composition wherever EFFECT
    and ENVELOPE both hold.
    """
    import pandas as pd
    rows = []
    # W2-F's interaction, converted to the EFFECT scale graduate() expects. The recorded value is
    # on the ppl scale where lower is better, so worse-than-additive is POSITIVE; graduate()'s
    # ">= -CI" rule is written for higher-is-better, so the sign is flipped here exactly once
    # (defect D20 — the first version compared the raw ppl-scale value and would have passed a
    # stack that is significantly worse than additive).
    inter = all_df[all_df["statistic"] == "interaction_AxB"]
    inter_terms = []
    for _, ir in inter.iterrows():
        ihw = (float(ir["ci_hi"]) - float(ir["ci_lo"])) / 2.0
        inter_terms.append({"name": f"W2-F {ir['bank_id']}", "estimate": -float(ir["value"]),
                            "ci_lo": -float(ir["value"]) - ihw,
                            "ci_hi": -float(ir["value"]) + ihw})
    c = all_df[all_df["statistic"] == "wikitext_ppl_contrast_vs_incumbent"]
    for (arm, bank, dose), g in c.groupby(["arm", "bank_id", "dose"], dropna=False):
        r = g.iloc[0]
        hw = (float(r["ci_hi"]) - float(r["ci_lo"])) / 2.0
        # The contrast is (this arm - incumbent) in ppl, where LOWER ppl is better, so a negative
        # contrast means the arm beat its incumbent. Expressed for effect_criterion as a
        # higher-is-better quantity by negating.
        eff = D.effect_criterion(
            treatment_mean=-float(r["value"]), treatment_ci_lo=-float(r["value"]) - hw,
            treatment_ci_hi=-float(r["value"]) + hw, incumbent_mean=0.0,
            higher_is_better=True, cost_ratio=1.0, mde=0.0)
        verdict = D.graduate(
            f"{arm}/{bank}@bits={dose}", effect=eff, envelope_measured=True,
            envelope_evidence=f"arm-local dose curve over {int(g['dose'].nunique())} bit depths",
            interaction_terms=inter_terms, incumbent_id=str(r["incumbent_id"]))
        rows.append({"arm": arm, "cell": f"{bank}@bits={dose}", "incumbent_id": r["incumbent_id"],
                     "contrast_ppl": float(r["value"]), "ci_lo": float(r["ci_lo"]),
                     "ci_hi": float(r["ci_hi"]), "ci_halfwidth": hw,
                     "effect_passed": eff["passed"], "effect_path": eff["path"],
                     "composition_evaluable": bool(inter_terms),
                     "composition_passed": verdict["composition"]["passed"],
                     "verdict": (verdict["verdict"] if inter_terms
                                 else ("NOT-GRADUATED-pending-composition" if eff["passed"]
                                       else verdict["verdict"])),
                     "claimed_by": D.CLAIMED_BY})
    df = pd.DataFrame(rows, columns=["arm", "cell", "incumbent_id", "contrast_ppl", "ci_lo",
                                     "ci_hi", "ci_halfwidth", "effect_passed", "effect_path",
                                     "composition_evaluable", "composition_passed",
                                     "verdict", "claimed_by"])
    df.to_parquet(pathlib.Path(out) / "graduation.parquet", index=False)
    summary = {"n_cells": int(len(df)),
               "n_effect_passed": int(df["effect_passed"].sum()) if len(df) else 0,
               "composition_evaluable": bool(inter_terms),
               "n_graduated": int((df["verdict"] == D.GRADUATED).sum()) if len(df) else 0,
               "interaction_terms": inter_terms,
               "note": ("COMPOSITION from W2-F, sign-converted to the effect scale; "
                        "worse-than-additive on ppl is positive and fails the criterion")}
    D.write_json(out, "graduation_summary.json", summary)
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="darwin_figs")
    ap.add_argument("--s3-prefix", default="darwin/darwin_figs")
    ap.add_argument("--with-vectors", action="store_true")
    ap.add_argument("--no-strict-controls", action="store_true",
                    help="record instead of failing when a figure lacks its control curves; "
                         "used only while a wave is still in flight")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    dest = H.cache_dir(out, "arms")

    prefixes = discover_arm_prefixes()
    print(f"[assemble] discovered {len(prefixes)} arm prefixes: {prefixes}", flush=True)
    got = sync_arms(dest, prefixes, a.with_vectors)
    all_df, arep = concat_atlas(dest, got, out)
    sdf, srep = concat_surprises(dest, got, out)
    cov = coverage(all_df, out)
    figs = render_figures(all_df, out, strict_controls=not a.no_strict_controls)
    F.manifest(figs, out)

    grad_df, grad_rows = graduation(all_df, out)

    # §5 G0 — evaluated MECHANICALLY from the graduation table and WP-1's presence. This is the
    # gate that decides whether stage 2 launches at all; §5 and §0.1 both define a G0 failure as a
    # complete stage-1 deliverable, not a shortfall, so it is recorded plainly either way.
    graduated = sorted(set(grad_df[grad_df["verdict"] == D.GRADUATED]["arm"])) if len(grad_df) else []
    wp1_rows = int((all_df["arm"] == "WP-1").sum())
    g0 = D.gate_g0(pilot_ci_widths_finite=bool(wp1_rows), graduated_arms=graduated)
    g0["wp1_atlas_rows"] = wp1_rows
    g0["arms_executed"] = sorted(set(all_df["arm"].dropna()))
    D.write_json(out, "g0.json", g0)
    print(f"[assemble] G0: passed={g0['passed']} — {g0['consequence']}", flush=True)

    summary = {"program": D.PROGRAM, "run_id": a.run_id, "commit_sha": D.commit_sha(),
               "provenance": D.provenance(), "atlas": arep, "surprises": srep,
               "arms_discovered": prefixes, "arms_with_rows": got,
               "coverage": {"n_arms": int(len(cov)), "n_executed": int(cov["executed"].sum())},
               "figures": {"n": len([k for k, v in figs.items() if "skipped" not in v]),
                           "n_skipped": len([k for k, v in figs.items() if "skipped" in v])},
               "g0": g0,
               "graduation": {"path": str(out / "graduation.parquet"),
                              "n_rows": int(len(grad_df)),
                              "verdicts": grad_rows},
               "finished_utc": D.utc_now()}
    D.write_json(out, "assemble_summary.json", summary)
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_ASSEMBLE_OK rows={arep['n_rows']} arms={arep['n_arms']} "
          f"figures={summary['figures']['n']} surprises={srep['n_surprises']}")


if __name__ == "__main__":
    main()
