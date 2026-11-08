"""DARWIN Wave-0 re-analysis arms (§4). CPU. Pure joins over rows other arms already wrote.

  W0-1c  The overlap join (§4/W0-1). Bank-side leverage spikes (W0-1a) x activation outliers
         (W0-1b): Jaccard + hypergeometric enrichment per (type, layer, bank, model), with every
         twin in the same table so the REFUTER — "twin overlap bands indistinguishable" — is
         decidable from the artifact. This is the arm §8 says scopes the program: if core
         coordinates ARE the outlier dimensions, affected claims carry an `outlier_overlap` flag.
  W0-7   OOD ROC from the existing atlas (§4/W0-7, §7.4). Band-deviation of the per-token e_U
         series against a wikitext normal band, used as a classifier: wikitext vs repetition vs
         code vs parallel, per model. REFUTER: AUC <= 0.6 everywhere.

Both arms read `humboldt/humboldt_b1` and the DARWIN W0-1a/W0-1b prefixes from S3 and never take
a number from prose (§0.18).

  pg run infra.aws_pg.jobs.darwin_w0_reanalysis --args "--arm W0-7" --type r7i.2xlarge ...
"""
import argparse
import json
import pathlib

from infra.pg import darwin_core as D
from infra.pg import darwin_w0 as DW
from infra.pg import humboldt_core as H

B1_PREFIX = "humboldt/humboldt_b1"
W0_8_PREFIX = "darwin/darwin_w0_8"
W0_1A_PREFIX = "darwin/darwin_w0_1a"
W0_1B_PREFIX = "darwin/darwin_w0_1b"

# §4/W0-7's frozen band + split rules.
BAND_LO_PCT, BAND_HI_PCT = 2.5, 97.5
AUC_REFUTER = 0.6
# The band is FIT on the first half of the wikitext tokens and EVALUATED on the second half.
# Without the split the negatives are the very tokens the band was built from, so ~95% of them sit
# inside it by construction and the AUC is inflated for every model — an instrument that cannot
# report a low number. The split makes the in-distribution class an honest held-out sample.
FIT_FRACTION = 0.5

SPEC_EXPECTATIONS = {
    "W1-J": ("detrended spectra show no peak above the shuffled-series null",
             "the detrended e_U series carries autocorrelation structure above its shuffled null"),
    "W0-8": ("the universal basis captures no more delta energy than a matched-rank random basis",
             "the highway basis captures delta energy well above the k/n Haar reference at every "
             "rank"),
    "W0-1c": ("twin overlap bands indistinguishable",
              "core32's overlap with the massive-activation coordinate set exceeds every twin's "
              "overlap band"),
    "W0-7": ("AUC <= 0.6 everywhere",
             "band-deviation of the core32 e_U series separates repetition and code from "
             "wikitext above AUC 0.6 in at least some layers"),
}


# ---------------------------------------------------------------------------
# W0-1c — the overlap join.
# ---------------------------------------------------------------------------
def run_w0_1c(out, atlas, gates, surprises, args):
    import numpy as np

    a_dir = D.fetch_prior_arm(H.cache_dir(out, "w0_1a"), args.w0_1a_prefix)
    b_dir = D.fetch_prior_arm(H.cache_dir(out, "w0_1b"), args.w0_1b_prefix)
    da, db = D.load_atlas(a_dir), D.load_atlas(b_dir)

    spikes = da[da["statistic"] == "leverage_spike_coords"]
    outl = db[db["statistic"] == "outlier_coords"]
    assert len(spikes) and len(outl), f"empty join inputs: {len(spikes)} spike rows, {len(outl)} outlier rows"

    n_rows, skipped, by_bank = 0, [], {}
    for _, orow in outl.iterrows():
        flags_o = json.loads(orow["flags"])
        site, corpus, L = flags_o["site"], orow["corpus_id"], int(orow["layer"])
        if corpus != args.criterion_corpus:
            continue
        o_coords = D.vector_of(b_dir, orow)
        types_here = [t for t in H.TYPES if H.SITE[t] == site]
        for t in types_here:
            n_amb = H.IN_DIM[t]
            for bid in D.bank_ids_for(t):
                sel = spikes[(spikes["type"] == t) & (spikes["layer"] == L)
                             & (spikes["bank_id"] == bid)]
                if not len(sel):
                    skipped.append({"type": t, "layer": L, "bank_id": bid,
                                    "reason": "no W0-1a row"})
                    continue
                s_coords = D.vector_of(a_dir, sel.iloc[0])
                st = DW.overlap_stats(s_coords, o_coords, n_amb)
                fl = {**H.type_flags(t), "site": site, "corpus": corpus,
                      "model": orow["model_id"], "rule_bank": f"{DW.LEVERAGE_MULT}x median lev",
                      "rule_act": f"{DW.OUTLIER_MULT}x median |h_i|", **st}
                # A bank with NO spiky coordinates (W0-1a records exactly that for the haar and
                # complement twins) has a well-defined Jaccard of 0 but an UNDEFINED enrichment
                # (its expected intersection is 0, so the ratio divides by zero) and an undefined
                # hypergeometric tail. Record each statistic where it is defined and record an
                # explicit undefined-row where it is not; writing NaN into `value` is refused by
                # the atlas, and dropping the cell would silently delete the twin band that the
                # REFUTER is decided against.
                if st["n_a"] == 0 or st["n_b"] == 0:
                    atlas.add("outlier_overlap_undefined", value=1.0, layer=L, type=t,
                              bank_id=bid, model_id=orow["model_id"], corpus_id=corpus,
                              flags={**fl, "reason": ("bank has no spiky coordinates"
                                                      if st["n_a"] == 0
                                                      else "no massive-activation coordinates")})
                    n_rows += 1
                if st["jaccard"] == st["jaccard"]:          # defined unless BOTH sets are empty
                    atlas.add("outlier_overlap_jaccard", value=st["jaccard"], layer=L, type=t,
                              bank_id=bid, model_id=orow["model_id"], corpus_id=corpus, flags=fl)
                    by_bank.setdefault((orow["model_id"], t, bid), []).append(st["jaccard"])
                    n_rows += 1
                if st["expected_intersection"] > 0:
                    atlas.add("outlier_overlap_enrichment", value=st["enrichment"], layer=L,
                              type=t, bank_id=bid, model_id=orow["model_id"], corpus_id=corpus,
                              flags=fl)
                    atlas.add("outlier_overlap_log_p", value=st["log_p_hypergeom"], layer=L,
                              type=t, bank_id=bid, model_id=orow["model_id"], corpus_id=corpus,
                              flags=fl)
                    n_rows += 2

    # The REFUTER is decided against the TWIN BAND, in the same table (§3 structured controls).
    twin_ids = list(D.TWIN_IDS)
    verdicts = {}
    for (model, t), _ in {(k[0], k[1]): None for k in by_bank}.items():
        core = by_bank.get((model, t, "core32"))
        twins = [np.mean(by_bank[(model, t, b)]) for b in twin_ids
                 if (model, t, b) in by_bank]
        if not core or len(twins) < 2:
            continue
        cm, tb = float(np.mean(core)), np.asarray(twins, dtype=np.float64)
        s = D.surprise(observed=cm, expected=float(tb.mean()), control_values=tb)
        surprises.add(arm="W0-1c", expected_ref=atlas.expected_ref, cell=f"{model}/{t}",
                      statistic="outlier_overlap_jaccard", s=s)
        # A twin band of exactly zero width makes the 2-sigma test degenerate: it can never fire,
        # however large the core's overlap is. That is a one-sided threshold, so it is recorded as
        # its OWN outcome (feedback_one_sided_thresholds) rather than being allowed to read as a
        # quiet "not a surprise". The absolute gap is recorded alongside so the cell stays
        # decidable by the director without re-running the arm.
        degenerate = bool(s["control_spread"] == 0.0)
        verdicts[f"{model}/{t}"] = {"core_mean_jaccard": cm, "twin_mean": float(tb.mean()),
                                    "twin_sd": float(tb.std(ddof=1)),
                                    "twin_max": float(tb.max()),
                                    "core_minus_twin_max": cm - float(tb.max()),
                                    "twin_band_degenerate": degenerate,
                                    "outside_twin_band": (None if degenerate
                                                          else s["is_surprise"])}
        atlas.add("outlier_overlap_core_vs_twin_band", value=cm, type=t, model_id=model,
                  bank_id="core32", flags={**H.type_flags(t), **verdicts[f"{model}/{t}"],
                                           "twin_ids": twin_ids})
    D.s3_sync(out, args.s3_prefix)

    n_out = sum(1 for v in verdicts.values() if v["outside_twin_band"] is True)
    n_degen = sum(1 for v in verdicts.values() if v["twin_band_degenerate"])
    gates.record("W0_1c_outlier_overlap_scope", True,
                 n_cells=len(verdicts), n_core_outside_twin_band=n_out,
                 n_twin_band_degenerate=n_degen,
                 scope_flag_required=bool(n_out),
                 note=("cells where core32's overlap with the massive-activation set exceeds its "
                       "own twin band; every affected downstream claim carries outlier_overlap"))
    D.write_json(out, "w0_1c_verdicts.json", verdicts)
    return {"n_rows": n_rows, "n_cells": len(verdicts), "n_outside_twin_band": n_out,
            "n_twin_band_degenerate": n_degen,
            "skipped": skipped[:50], "n_skipped": len(skipped),
            "coverage_gap_down_proj": "down_proj_in is not a Phase-B site (§2.1 cost exclusion), "
                                      "so down_proj banks have no activation counterpart"}


# ---------------------------------------------------------------------------
# W0-7 — OOD ROC from the B1 series.
# ---------------------------------------------------------------------------
def run_w0_7(out, atlas, gates, surprises, args):
    import numpy as np

    b1 = D.fetch_prior_arm(H.cache_dir(out, "b1"), args.b1_prefix)
    df = D.load_atlas(b1)
    ser = df[df["statistic"] == "e_U_series"]
    assert len(ser), "no e_U_series rows in the B1 atlas"

    models = sorted(ser["model_id"].dropna().unique())
    if args.models:
        models = [m for m in models if m in args.models]
    aucs, n_rows = {}, 0

    for model in models:
        sm = ser[ser["model_id"] == model]
        ood_corpora = sorted(set(sm["corpus_id"].dropna()) - {"wikitext"})
        if args.smoke:
            ood_corpora = ood_corpora[:1]
        for t in sorted(sm["type"].dropna().unique()):
            for L in sorted(sm["layer"].dropna().unique()):
                for bid in sorted(sm["bank_id"].dropna().unique()):
                    base = sm[(sm["type"] == t) & (sm["layer"] == L) & (sm["bank_id"] == bid)]
                    wk = base[base["corpus_id"] == "wikitext"]
                    if not len(wk):
                        continue
                    x = D.vector_of(b1, wk.iloc[0]).astype(np.float64)
                    cut = int(len(x) * FIT_FRACTION)
                    assert cut > 10 and len(x) - cut > 10, \
                        f"wikitext series too short to split at {model}/{t}/L{L}/{bid}: {len(x)}"
                    fit, held = x[:cut], x[cut:]
                    lo = float(np.percentile(fit, BAND_LO_PCT))
                    hi = float(np.percentile(fit, BAND_HI_PCT))
                    neg = DW.band_deviation(held, lo, hi)
                    for corpus in ood_corpora:
                        cr = base[base["corpus_id"] == corpus]
                        if not len(cr):
                            continue
                        pos = DW.band_deviation(D.vector_of(b1, cr.iloc[0]).astype(np.float64),
                                                lo, hi)
                        r = DW.roc_auc(pos, neg)
                        fl = {**H.type_flags(t), "band_lo": lo, "band_hi": hi,
                              "band_pct": [BAND_LO_PCT, BAND_HI_PCT],
                              "fit_fraction": FIT_FRACTION, "n_fit": cut,
                              "tie_fraction": r["tie_fraction"], "site": H.SITE[t],
                              "refuter_threshold": AUC_REFUTER}
                        atlas.add("ood_band_deviation_auc", value=r["auc"], layer=int(L), type=t,
                                  bank_id=bid, model_id=model, corpus_id=corpus,
                                  n=r["n_pos"] + r["n_neg"], flags=fl)
                        aucs.setdefault((model, corpus, bid), []).append(r["auc"])
                        n_rows += 1
        print(f"[W0-7] {model}: {n_rows} rows so far", flush=True)
        D.s3_sync(out, args.s3_prefix)

    # Summary + the REFUTER's decision, per (model, corpus), core against its own twin band.
    best = 0.0
    for (model, corpus, bid), v in sorted(aucs.items()):
        s = atlas.add_ci("ood_band_deviation_auc_summary", v, model_id=model, corpus_id=corpus,
                         bank_id=bid, flags={"n_cells": len(v),
                                             "refuter_threshold": AUC_REFUTER})
        best = max(best, s["mean"])
    for (model, corpus) in sorted({(m, c) for m, c, _ in aucs}):
        core = aucs.get((model, corpus, "core32"))
        twins = [float(np.mean(aucs[(model, corpus, b)])) for b in D.TWIN_IDS
                 if (model, corpus, b) in aucs]
        if not core or len(twins) < 2:
            continue
        s = D.surprise(observed=float(np.mean(core)), expected=float(np.mean(twins)),
                       control_values=np.asarray(twins, dtype=np.float64))
        surprises.add(arm="W0-7", expected_ref=atlas.expected_ref, cell=f"{model}/{corpus}",
                      statistic="ood_band_deviation_auc", s=s)

    gates.record("W0_7_auc_above_refuter", bool(best > AUC_REFUTER),
                 best_mean_auc=best, threshold=AUC_REFUTER, n_summaries=len(aucs),
                 note="RECORDED, not blocking: AUC <= 0.6 everywhere is W0-7's named REFUTER and "
                      "a legitimate recorded outcome, not a validity failure")
    return {"n_rows": n_rows, "best_mean_auc": best, "n_summaries": len(aucs),
            "refuted": bool(best <= AUC_REFUTER)}



# ---------------------------------------------------------------------------
# W0-8 — the delta-compression frontier, derived exactly from the banked Gram spectrum.
# ---------------------------------------------------------------------------
def run_w0_8(out, atlas, gates, surprises, args):
    """Mean relative DELTA energy captured vs basis rank (§4/W0-8).

    The first W0-8 pass recorded the BASE matrix's energy on each basis — the wrong object; the
    spec asks what fraction of the FINE-TUNES' updates a rank-r basis captures (defect D15).

    No re-stream is needed, because the consensus Gram already IS that quantity:

        mean_i ‖dW_i U‖²_F / ‖dW_i‖²_F  =  trace(Uᵀ G U) / N,     G = Σ dW_iᵀ dW_i / ‖dW_i‖²_F

    and for the eigenbasis of G that trace is just a sum of eigenvalues. So the highway (top-r)
    and core (bottom-r) curves are partial sums of the spectrum this arm already banked, and the
    Haar reference is analytic: E[trace(UᵀGU)]/N = r/n. Exact, not approximate — re-downloading
    144 GB of checkpoints would recompute the same numbers.
    """
    import numpy as np
    d = D.fetch_prior_arm(H.cache_dir(out, "w0_8"), args.w0_8_prefix)
    df = D.load_atlas(d)
    spec = df[df["statistic"] == "gram_eigenvalue_spectrum"]
    assert len(spec), "no gram_eigenvalue_spectrum rows — run darwin_w0_gram --arm W0-8 first"

    n_rows = 0
    for _, row in spec.iterrows():
        lam = np.sort(np.asarray(D.vector_of(d, row), dtype=np.float64))   # ascending
        tot = float(lam.sum())
        assert tot > 0, f"degenerate spectrum at {row['type']} L{row['layer']}"
        n = int(lam.size)
        t, L = row["type"], int(row["layer"])
        fl = {**H.type_flags(t), "n_ambient": n, "trace_G": tot,
              "identity": "mean_i ||dW_i U||^2_F / ||dW_i||^2_F = trace(U^T G U) / N"}
        for r in DW.rank_ladder():
            if r > n:
                continue
            for bid, frac in (("highway_topr", float(lam[-r:].sum() / tot)),
                              ("core_bottomr", float(lam[:r].sum() / tot)),
                              ("haar_r", r / n)):
                atlas.add("delta_energy_captured", value=frac, layer=L, type=t, bank_id=bid,
                          dose=float(r), model_id=row["model_id"], model_sha=row["model_sha"],
                          flags={**fl, "rank": r, "haar_reference": r / n,
                                 "vs_haar": frac / (r / n)})
                n_rows += 1
        D.s3_sync(out, args.s3_prefix)

    # The REFUTER is decided against the matched-rank random basis (haar_r), which rides in the
    # same table at every rank, so the comparison needs no separate artifact.
    gates.record("W0_8_frontier_recorded", True, n_rows=n_rows, n_cells=len(spec),
                 ladder=DW.rank_ladder(),
                 note="delta-energy frontier derived from the banked Gram spectrum; the identity "
                      "is exact, so this is not an approximation of a re-stream")
    return {"n_rows": n_rows, "n_cells": int(len(spec)), "ladder": DW.rank_ladder()}



# ---------------------------------------------------------------------------
# W1-J — heartbeat redo: detrended autocorrelation / spectra of the e_U series.
# ---------------------------------------------------------------------------
DETREND_NOTE = ("first-difference + Hann window, frozen in §4/W1-J. The raw e_U series is "
                "dominated by a slow drift, which puts almost all periodogram power in the "
                "lowest bins and leaves the spectral statistic unable to distinguish anything — "
                "the 'dead spectral statistic' this arm exists to fix.")


def detrend(series):
    """First-difference then Hann-window, per §4/W1-J's frozen rule."""
    import numpy as np
    x = np.asarray(series, dtype=np.float64).ravel()
    assert x.size >= 8, f"series too short to detrend: {x.size}"
    d = np.diff(x)
    return d * np.hanning(d.size)


def run_w1_j(out, atlas, gates, surprises, args):
    """Autocorrelation and spectral peaks of the DETRENDED series, against a shuffled null.

    The null is the SAME series shuffled: identical marginal distribution, destroyed time
    structure. That is the control the REFUTER is decided against — a peak that survives
    shuffling is a property of the histogram, not of the dynamics.
    """
    import numpy as np
    b1 = D.fetch_prior_arm(H.cache_dir(out, "b1"), args.b1_prefix)
    df = D.load_atlas(b1)
    ser = df[df["statistic"] == "e_U_series"]
    assert len(ser), "no e_U_series rows in the B1 atlas"
    if args.models:
        ser = ser[ser["model_id"].isin(args.models)]

    rng = np.random.default_rng(args.seed)
    n_rows, peaks_by = 0, {}
    for model in sorted(ser["model_id"].dropna().unique()):
        sm = ser[ser["model_id"] == model]
        for corpus in sorted(sm["corpus_id"].dropna().unique()):
            sc = sm[sm["corpus_id"] == corpus]
            for bid in sorted(sc["bank_id"].dropna().unique()):
                sb = sc[sc["bank_id"] == bid]
                for L in sorted(sb["layer"].dropna().unique())[::args.layer_stride]:
                    row = sb[sb["layer"] == L].iloc[0]
                    x = D.vector_of(b1, row).astype(np.float64)
                    if x.size < 64:
                        continue
                    d = detrend(x)
                    sh = detrend(rng.permutation(x))
                    ac, ac0 = H.autocorr(d, 32), H.autocorr(sh, 32)
                    sp, sp0 = H.spectral_peaks(d, top=5), H.spectral_peaks(sh, top=5)
                    top_pow = float(sp["power"][0]) if sp["power"] else 0.0
                    null_pow = float(sp0["power"][0]) if sp0["power"] else 0.0
                    fl = {"site": H.SITE[row["type"]] if row["type"] in H.SITE else None,
                          "detrend": DETREND_NOTE, "n_raw": int(x.size), "n_detrended": int(d.size),
                          "top_period": (float(sp["period"][0]) if sp["period"] else None),
                          "null_top_power": null_pow,
                          "excess_over_null": top_pow - null_pow}
                    atlas.add("detrended_spectral_top_power", value=top_pow, layer=int(L),
                              type=row["type"], bank_id=bid, model_id=model, corpus_id=corpus,
                              incumbent_id="shuffled_null", flags=fl)
                    atlas.add("detrended_spectral_top_power_shuffled", value=null_pow,
                              layer=int(L), type=row["type"], bank_id=bid, model_id=model,
                              corpus_id=corpus, flags=fl)
                    atlas.add("detrended_autocorr_lag1", value=float(ac[0]), layer=int(L),
                              type=row["type"], bank_id=bid, model_id=model, corpus_id=corpus,
                              incumbent_id="shuffled_null",
                              flags={**fl, "shuffled_lag1": float(ac0[0])})
                    peaks_by.setdefault(bid, []).append(top_pow - null_pow)
                    n_rows += 3
            D.s3_sync(out, args.s3_prefix)
        print(f"[W1-J] {model}: {n_rows} rows", flush=True)

    # REFUTER decided against the shuffled null, per bank, with the twin band in the same table.
    summary = {}
    for bid, v in sorted(peaks_by.items()):
        s = atlas.add_ci("detrended_spectral_excess_over_null", v, bank_id=bid,
                         incumbent_id="shuffled_null", flags={"n_cells": len(v)})
        summary[bid] = {"mean_excess": s["mean"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                        "above_null": bool(s["ci_lo"] > 0)}
    n_above = sum(1 for v in summary.values() if v["above_null"])
    gates.record("W1J_structure_above_shuffled_null", bool(n_above > 0),
                 n_banks_above_null=n_above, n_banks=len(summary), per_bank=summary,
                 note="RECORDED, not blocking: no peak above the shuffled null is W1-J's named "
                      "REFUTER and a legitimate outcome")
    return {"n_rows": n_rows, "n_banks": len(summary), "n_banks_above_null": n_above,
            "per_bank": summary}


RUNNERS = {"W0-1c": run_w0_1c, "W0-7": run_w0_7, "W0-8": run_w0_8,
           "W1-J": run_w1_j}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--s3-prefix", default=None)
    ap.add_argument("--b1-prefix", default=B1_PREFIX)
    ap.add_argument("--w0-1a-prefix", default=W0_1A_PREFIX)
    ap.add_argument("--w0-1b-prefix", default=W0_1B_PREFIX)
    ap.add_argument("--w0-8-prefix", default=W0_8_PREFIX)
    ap.add_argument("--criterion-corpus", default="wikitext")
    ap.add_argument("--models", default="")
    ap.add_argument("--layer-stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    a.run_id = a.run_id or f"darwin_{a.arm.lower().replace('-', '_')}"
    a.s3_prefix = a.s3_prefix or f"{D.PROGRAM}/{a.run_id}"
    a.models = [m.strip() for m in a.models.split(",") if m.strip()]

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    utc = D.utc_now()
    exp = D.Expectations(out.parent / "expectations.jsonl")
    if not exp.by_arm(a.arm):
        refuter, direction = SPEC_EXPECTATIONS[a.arm]
        exp.log(a.arm, refuter=refuter, direction=direction, utc=utc)
    ref = exp.require(a.arm)

    gates = H.Gates(out)
    atlas = D.Atlas(out, arm=a.arm, run_id=a.run_id, register="ATLAS", wave="W0",
                    expected_ref=ref)
    surprises = D.SurpriseLog(out)
    manifest = {"program": D.PROGRAM, "arm": a.arm, "run_id": a.run_id, "register": "ATLAS",
                "wave": "W0", "commit_sha": D.commit_sha(), "provenance": D.provenance(),
                "inputs": {"b1": a.b1_prefix, "w0_1a": a.w0_1a_prefix, "w0_1b": a.w0_1b_prefix},
                "rules": {"BAND_PCT": [BAND_LO_PCT, BAND_HI_PCT], "FIT_FRACTION": FIT_FRACTION,
                          "AUC_REFUTER": AUC_REFUTER, "OUTLIER_MULT": DW.OUTLIER_MULT,
                          "LEVERAGE_MULT": DW.LEVERAGE_MULT},
                "expected_ref": ref, "started_utc": utc, "seed": a.seed}
    D.write_json(out, "manifest.json", manifest)

    result = RUNNERS[a.arm](out, atlas, gates, surprises, a)

    atlas.close()
    rep = atlas.to_parquet()
    srep = surprises.to_parquet()
    D.write_json(out, f"{a.arm.lower().replace('-', '_')}_summary.json",
                 {"manifest": manifest, "atlas": rep, "surprises": srep, "gates": gates.gates,
                  "result": result, "finished_utc": D.utc_now()})
    D.s3_sync(out, a.s3_prefix)
    print("=" * 78)
    print(f"DARWIN_{a.arm.replace('-', '_')}_OK rows={rep['n_rows']} "
          f"surprises={srep['n_rows']} {json.dumps(result, default=str)[:300]}")


if __name__ == "__main__":
    main()
