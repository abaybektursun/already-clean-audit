"""MERCATOR Wave 0 — the three arms that read existing artifacts and nothing else.

  M-1     (queue 1)      MQA prediction: is the sink's q_proj concentration explained by
                         multi-query asymmetry? Classify every census base MHA / GQA / MQA from
                         its own recorded geometry, then compare k_proj avoidance across classes.
                         REFUTER: k_proj at chance in a true-MHA family kills the MQA story.
  M-9a    (queue 9)      The universality table's CPU cells: per-type scoping and leverage
                         profiles across the five census families, each against its own recorded
                         uniform reference and the twin banks.
  M-mine  (standing W2-A) Mine W2-A's per-bank motion rows for the Muon regime.

All three read `darwin/darwin_figs2/atlas.parquet` and nothing else, so they are CPU-only and
cannot be confounded by a re-measurement: the object is DARWIN's own recorded numbers.

Two properties are enforced rather than assumed:
  * the arm's expectation must already be pre-logged in the FROZEN store on S3 (§3), which the job
    pulls — an arm cannot run before its expected-pattern row exists;
  * §3's dual-basis rule applies to every coordinate claim. Avoidance and leverage ARE coordinate
    claims, so each is written in the coordinate basis and again after conjugating the same
    spectrum by one random reference basis.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import darwin_w0 as W0
from infra.pg import humboldt_core as H
from infra.pg import mercator_core as M
from infra.pg import mercator_program as P

ATLAS_S3 = f"s3://{M.BUCKET}/darwin/darwin_figs2/atlas.parquet"
ARMS = ("M-1", "M-9a", "M-mine", "M-C6")
ATTN_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
FULL_BATTERY = ("core32", "anti32", "topsigma32", "bottomsigma32", "complement32",
                "haar32_s0", "haar32_s1", "haar32_s2")


def _flags(row) -> dict:
    f = row["flags"]
    return json.loads(f) if isinstance(f, str) else (f or {})


def attention_class(geom: dict) -> dict:
    """MHA / GQA / MQA from a base's OWN recorded geometry.

    The classification is read from `num_attention_heads` vs `num_key_value_heads` as the census
    recorded them in each row's flags, not from a hand-kept table of model names — a name-keyed
    table is how a mislabelled family silently becomes evidence.
    """
    h = int(geom["num_attention_heads"])
    kv = int(geom["num_key_value_heads"])
    assert h > 0 and kv > 0 and h % kv == 0, f"implausible head geometry {h}/{kv}"
    cls = "MHA" if kv == h else ("MQA" if kv == 1 else "GQA")
    hid = int(geom["hidden_size"])
    # head_dim is DECLARED by some configs (Gemma sets it independently of hidden/heads) and
    # omitted by others, where the standard relation hidden = n_heads x head_dim holds. Which of
    # the two applied is recorded, because a derived head_dim is an assumption and the shape
    # confound below is computed from it.
    if "head_dim" in geom and geom["head_dim"]:
        hd, hd_src = int(geom["head_dim"]), "declared"
    else:
        assert hid % h == 0, f"cannot derive head_dim: hidden {hid} not divisible by {h} heads"
        hd, hd_src = hid // h, "derived (hidden_size / n_heads)"
    return {"n_heads": h, "n_kv_heads": kv, "class": cls, "kv_share": h // kv,
            "head_dim": hd, "head_dim_source": hd_src, "hidden_size": hid}


def shape_confound(cls: dict, t: str, n_ft: int) -> dict:
    """The rank facts a k_proj comparison MUST carry (the ERLANGEN k_proj lesson).

    Under MQA/GQA the k/v projections are SHORT — out_dim = n_kv_heads x head_dim, which can be far
    below the residual width — so the consensus Gram Sum_i dW_i^T dW_i has rank at most
    n_ft x out_dim. If that bound falls below the ambient dimension, the Gram has an EXACT kernel,
    its smallest eigenvalues are structurally zero, and a bottom-eigenvalue fraction of ~0 means
    "this matrix cannot write there at all", NOT "the cloud avoids writing there".

    Reporting an avoidance ratio for such a cell without this flag is how a shape becomes a
    finding. Every number here comes from the base's own recorded geometry.
    """
    hid = cls["hidden_size"]
    out_dim = {"q_proj": cls["n_heads"] * cls["head_dim"],
               "k_proj": cls["n_kv_heads"] * cls["head_dim"],
               "v_proj": cls["n_kv_heads"] * cls["head_dim"],
               "o_proj": hid}[t]
    in_dim = cls["n_heads"] * cls["head_dim"] if t == "o_proj" else hid
    max_rank = min(in_dim, int(n_ft) * out_dim)
    return {"out_dim": int(out_dim), "in_dim": int(in_dim), "n_full_ft": int(n_ft),
            "max_gram_rank": int(max_rank),
            "structurally_rank_deficient": bool(max_rank < in_dim),
            "kernel_dims": int(max(0, in_dim - max_rank)),
            "is_short_kv": bool(t in ("k_proj", "v_proj") and out_dim < in_dim)}


def _dual_basis_pair(values, *, seed: int):
    """The same spectral statistic, measured again in one random reference basis (§3).

    An avoidance fraction computed from a Gram's EIGENVALUES is basis-free by construction: a
    rotation moves the eigenvectors, not the spectrum. Writing the reference-basis row therefore
    tests exactly the right thing — if the coordinate-basis number and the reference-basis number
    ever differ, the statistic was reading the grid rather than the operator, and the grid-artifact
    guard has caught it. Here the reference row is produced by conjugating a diagonal operator
    built from the SAME values by a Haar rotation and re-extracting its spectrum, so the numerical
    path is genuinely different even though the mathematics says it must agree.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    Q = M.random_reference_basis(n, seed)
    lam = np.linalg.eigvalsh(Q @ np.diag(v) @ Q.T)
    return float(v.sum()), float(lam.sum()), float(np.abs(np.sort(lam) - np.sort(v)).max())


# ---------------------------------------------------------------------------
# M-1 — the MQA prediction.
# ---------------------------------------------------------------------------
def run_m1(df, at, sur) -> dict:
    rows = df[(df["arm"].isin(["W0-3", "W0-4"]))
              & (df["statistic"] == "census_bottom_eigenvalue_fraction")]
    assert len(rows), "no census_bottom_eigenvalue_fraction rows in the DARWIN atlas"
    out, classes = {}, {}
    for model, g in rows.groupby("model_id"):
        geom = _flags(g.iloc[0])["geometry"]
        cls = attention_class(geom)
        classes[str(model)] = cls
        per_type = {}
        for t, gt in g.groupby("type"):
            if t not in ATTN_TYPES:
                continue
            frac = gt["value"].astype(float).to_numpy()
            ref = np.array([float(_flags(r)["uniform_reference"]) for _, r in gt.iterrows()])
            assert (ref > 0).all(), f"{model}/{t}: missing uniform_reference"
            n_ft = int(_flags(gt.iloc[0])["n_full_ft"])
            shape = shape_confound(cls, str(t), n_ft)
            ratio = frac / ref
            s = at.add_ci("census_avoidance_ratio", ratio, model_id=str(model), type=str(t),
                          bank_id="core32", basis=M.BASIS_COORD,
                          flags={"attention_class": cls["class"], "n_heads": cls["n_heads"],
                                 "n_kv_heads": cls["n_kv_heads"], "n_layers_measured": int(len(gt)),
                                 "uniform_reference_mean": float(ref.mean()),
                                 "source": ATLAS_S3, **shape})
            # §3 dual basis: the spectrum-derived statistic re-read in a random reference basis.
            coord_sum, ref_sum, max_dev = _dual_basis_pair(frac, seed=H.cell_seed(1337, model, t))
            at.add("census_avoidance_ratio", value=float((frac.sum() / ref.sum())
                                                         * (ref_sum / coord_sum)),
                   model_id=str(model), type=str(t), bank_id="core32",
                   basis=M.BASIS_RANDOM_REF,
                   flags={"attention_class": cls["class"], "spectrum_max_abs_dev": max_dev,
                          "note": "same spectrum after Haar conjugation; a difference would mean "
                                  "the statistic reads the coordinate grid, not the operator"})
            per_type[str(t)] = {"mean_ratio": s["mean"], "ci": [s["ci_lo"], s["ci_hi"]],
                                "n": s["n"], "shape": shape}
        assert set(ATTN_TYPES) <= set(per_type), \
            f"{model}: missing attention types {sorted(set(ATTN_TYPES) - set(per_type))}"
        out[str(model)] = {"attention": cls, "by_type": per_type}

    # The REFUTER's decision cell: k_proj in a TRUE-MHA family. "At chance" means the avoidance
    # ratio's CI covers 1.0.
    mha = [m for m, d in out.items() if d["attention"]["class"] == "MHA"]
    verdict_cells = {}
    for m in mha:
        k = out[m]["by_type"]["k_proj"]
        q = out[m]["by_type"]["q_proj"]
        at_chance = bool(k["ci"][0] <= 1.0 <= k["ci"][1])
        verdict_cells[m] = {
            "k_proj_mean_ratio": k["mean_ratio"], "ci": k["ci"], "covers_one": at_chance,
            "refuter_fires": at_chance,
            "k_proj_shape": k["shape"],
            "cell_is_clean": not k["shape"]["structurally_rank_deficient"],
            "q_proj_mean_ratio": q["mean_ratio"],
            "k_over_q": k["mean_ratio"] / q["mean_ratio"]}
    # Which cells are readable at all: a structurally rank-deficient k/v cell has an exact kernel,
    # so its bottom-eigenvalue fraction is a SHAPE, not an avoidance. Recorded per cell so a reader
    # can see which zeros are measurements and which are geometry.
    shape_flags = {m: {t: d["by_type"][t]["shape"] for t in ATTN_TYPES}
                   for m, d in out.items()}
    unreadable = {m: sorted(t for t in ATTN_TYPES
                            if out[m]["by_type"][t]["shape"]["structurally_rank_deficient"])
                  for m in out}
    return {"arm": "M-1", "by_model": out, "classes": classes,
            "true_mha_families": mha,
            "refuter_cells": verdict_cells,
            "refuter_evaluable": bool(mha),
            "shape_confound": {"per_model_per_type": shape_flags,
                               "structurally_unreadable_cells": unreadable,
                               "why": ("under MQA/GQA the k/v projections are short, so a bottom-"
                                       "eigenvalue fraction near 0 can be an exact kernel rather "
                                       "than avoidance; the ERLANGEN k_proj lesson")},
            "note": ("REFUTER: k_proj at chance in a true-MHA family kills the MQA story. "
                     "'At chance' = the avoidance ratio's bootstrap CI covers 1.0.")}


# ---------------------------------------------------------------------------
# M-9a — the universality table's CPU cells.
# ---------------------------------------------------------------------------
LEVERAGE_STATS = ("census_leverage_max", "census_leverage_participation_ratio",
                  "census_leverage_spike_count")


def run_m9a(df, at, sur) -> dict:
    rows = df[df["arm"].isin(["W0-3", "W0-4"])]
    assert len(rows), "no census rows"
    table, banks_seen = {}, {}
    for model, g in rows.groupby("model_id"):
        cell = {}
        # (i) per-type scoping: is avoidance concentrated on q_proj?
        av = g[g["statistic"] == "census_bottom_eigenvalue_fraction"]
        by_type = {}
        for t, gt in av.groupby("type"):
            if t not in ATTN_TYPES:
                continue
            ref = np.array([float(_flags(r)["uniform_reference"]) for _, r in gt.iterrows()])
            by_type[str(t)] = float((gt["value"].astype(float).to_numpy() / ref).mean())
        assert by_type, f"{model}: no typed avoidance rows"
        strongest = min(by_type, key=by_type.get)          # lowest ratio = strongest avoidance
        cell["scoping"] = {"by_type_mean_ratio": by_type, "strongest_avoidance_type": strongest,
                           "q_proj_is_strongest": strongest == "q_proj"}

        # (ii) leverage profiles, with the twin banks present in the SAME cell.
        prof = {}
        for st in LEVERAGE_STATS:
            gs = g[g["statistic"] == st]
            if not len(gs):
                continue
            per_bank = {}
            for b, gb in gs.groupby("bank_id"):
                s = at.add_ci(st, gb["value"].astype(float).to_numpy(), model_id=str(model),
                              bank_id=str(b), basis=M.BASIS_COORD,
                              flags={"source": ATLAS_S3, "n_cells": int(len(gb))})
                per_bank[str(b)] = {"mean": s["mean"], "ci": [s["ci_lo"], s["ci_hi"]]}
                banks_seen.setdefault(st, set()).add(str(b))
            prof[st] = per_bank
        cell["leverage"] = prof
        table[str(model)] = cell

    # The census carried 5 of the 8 standing banks (critic F8): record the gap as data.
    battery_gap = {st: sorted(set(FULL_BATTERY) - bs) for st, bs in banks_seen.items()}

    # (iii) The cells that CANNOT be filled from existing artifacts, with the evidence.
    # DARWIN's census banked only the Gram SPECTRA (`W0-3__lam__<model>__<type>__L<k>.npy`), never
    # the eigenvectors — so cross-layer verticality and the published-outlier-index comparison,
    # both of which need per-COORDINATE leverage, are not derivable here. Recording this as a named
    # pivot with its evidence is the alternative to letting the cells quietly go missing.
    impossible = {
        "cross_layer_verticality": {
            "outcome": M.COVERAGE_IMPOSSIBLE,
            "evidence": ("the census published per-cell eigenvalue vectors only "
                         "(darwin/darwin_w0_3*/vectors/W0-3__lam__*.npy); verticality is a "
                         "statement about shared COORDINATES and needs the eigenvectors"),
            "unlock": "re-run the census for the foreign families with eigenvector banking"},
        "published_outlier_index_comparison": {
            "outcome": M.COVERAGE_IMPOSSIBLE,
            "evidence": ("census_leverage_spike_count is a COUNT; the spike INDICES were never "
                         "banked, so they cannot be compared with a published index list"),
            "unlock": "same re-census, plus a cite-verified index list from the source paper"},
    }
    return {"arm": "M-9a", "table": table, "battery_gap_in_source_rows": battery_gap,
            "not_derivable_from_existing_artifacts": impossible}


# ---------------------------------------------------------------------------
# M-mine — the Muon rows in W2-A.
# ---------------------------------------------------------------------------
def run_mmine(df, at, sur, expected_ref: str) -> dict:
    w = df[df["arm"] == "W2-A"]
    assert len(w), "no W2-A rows"
    mf = w[w["statistic"] == "motion_fraction"].copy()
    assert len(mf), "no motion_fraction rows in W2-A"
    mf["regime"] = [str(_flags(r).get("regime")) for _, r in mf.iterrows()]
    regimes = sorted(mf["regime"].unique())
    assert "muon" in regimes, f"no muon regime in W2-A; regimes present: {regimes}"

    by = {}
    for (reg, bank), g in mf.groupby(["regime", "bank_id"]):
        s = at.add_ci("w2a_motion_fraction", g["value"].astype(float).to_numpy(),
                      bank_id=str(bank), basis=M.BASIS_COORD,
                      flags={"regime": reg, "source": ATLAS_S3, "n_cells": int(len(g))})
        by.setdefault(reg, {})[str(bank)] = {"mean": s["mean"], "ci": [s["ci_lo"], s["ci_hi"]],
                                             "n": s["n"]}
    present = {b for r in by.values() for b in r}
    M.check_controls(present & set(FULL_BATTERY), arm="M-mine")

    # The pre-logged prediction: Muon MOVES the sink. Read as core32 motion relative to the twin
    # band, per regime — so "moves" is measured against the same controls in every regime rather
    # than against zero.
    def _twin_band(d):
        tw = [d[b]["mean"] for b in ("haar32_s0", "haar32_s1", "haar32_s2") if b in d]
        assert len(tw) == 3, f"expected 3 haar twins, got {len(tw)}"
        return float(np.mean(tw)), float(np.std(tw, ddof=1))

    comparison = {}
    for reg, d in sorted(by.items()):
        if "core32" not in d:
            continue
        tmean, tstd = _twin_band(d)
        comparison[reg] = {"core32_mean": d["core32"]["mean"], "twin_mean": tmean,
                           "twin_sd": tstd,
                           "core_over_twin": d["core32"]["mean"] / tmean,
                           "core32_ci": d["core32"]["ci"]}
        s = M.surprise(d["core32"]["mean"], tmean,
                       [d[b]["mean"] for b in ("haar32_s0", "haar32_s1", "haar32_s2") if b in d])
        sur.add(arm="M-mine", expected_ref=expected_ref, cell=f"regime={reg}",
                statistic="w2a_motion_fraction/core32", s=s)

    # Does damage follow motion? Join retention against realized motion, per regime.
    ret = w[w["statistic"] == "retention_ppl"].copy()
    damage = {}
    if len(ret):
        ret["regime"] = [str(_flags(r).get("regime")) for _, r in ret.iterrows()]
        for reg, g in ret.groupby("regime"):
            damage[str(reg)] = {"retention_ppl_mean": float(g["value"].astype(float).mean()),
                                "n": int(len(g))}
    return {"arm": "M-mine", "regimes": regimes, "by_regime_bank": by,
            "core_vs_twin": comparison, "retention_by_regime": damage,
            "prediction": ("pre-logged: Muon moves the sink, and the damage that follows tracks "
                           "the motion power law")}


def run_mc6(df, at, sur) -> dict:
    """M-C6 stage 1 — RECOMPUTE the OOD baseline before anything is asked to beat it.

    The queue names "the AUC 0.69 baseline to beat". Recomputed from W0-7's own rows, 0.6904 is
    the MAXIMUM over 196 summary cells whose MEAN is 0.5102 — chance. Treating the maximum of 196
    selections as the level a new feature must clear is a selection effect in the baseline's
    favour in one direction and against it in the other: it is simultaneously too hard (a new
    feature must beat the luckiest of 196 cells) and too easy (no correction is applied to the new
    feature's own selection). So the baseline is reported as a DISTRIBUTION with its multiplicity,
    and the level to beat is stated as a percentile of that distribution rather than its maximum.

    Stage 2 — the actual rescoring with resonance-informed features — is NOT run here and the
    reason is recorded: W0-7 banked per-cell AUCs, not per-item scores, and an AUC of a RECOMBINED
    score cannot be computed from per-cell AUCs. It needs the corpora forward passes again.
    """
    w = df[df["arm"] == "W0-7"]
    assert len(w), "no W0-7 rows in the DARWIN atlas"
    summ = w[w["statistic"] == "ood_band_deviation_auc_summary"]
    raw = w[w["statistic"] == "ood_band_deviation_auc"]
    assert len(summ) and len(raw), "W0-7 is missing its AUC statistics"

    v = summ["value"].astype(float).to_numpy()
    order = np.argsort(v)[::-1]
    best = summ.iloc[int(order[0])]
    pct = {f"p{p}": float(np.percentile(v, p)) for p in (50, 75, 90, 95, 99)}
    thr = float(_flags(summ.iloc[0]).get("refuter_threshold", float("nan")))

    s = at.add_ci("ood_auc_baseline", v, corpus_id="W0-7_summary_cells",
                  flags={"source": ATLAS_S3, "n_summary_cells": int(len(summ)),
                         "n_raw_cells": int(len(raw)), "chance": 0.5,
                         "w0_7_refuter_threshold": thr})
    for name, val in pct.items():
        at.add(f"ood_auc_baseline_{name}", value=val, corpus_id="W0-7_summary_cells",
               flags={"n_summary_cells": int(len(summ))})
    at.add("ood_auc_baseline_max", value=float(v.max()), corpus_id=str(best["corpus_id"]),
           bank_id=str(best["bank_id"]),
           flags={"note": "the queue's cited 0.69; it is the MAX over the summary cells",
                  "n_summary_cells": int(len(summ))})

    # How many cells clear the queue's cited level by chance alone? Reported so the level's
    # meaning is visible rather than assumed.
    n_over = int((v >= 0.69).sum())
    return {"arm": "M-C6", "stage": "baseline recompute only",
            "source": ATLAS_S3, "source_arm": "W0-7",
            "n_summary_cells": int(len(summ)), "n_raw_cells": int(len(raw)),
            "mean": s["mean"], "ci": [s["ci_lo"], s["ci_hi"]],
            "percentiles": pct, "max": float(v.max()),
            "max_cell": {"corpus_id": str(best["corpus_id"]), "bank_id": str(best["bank_id"]),
                         "value": float(best["value"])},
            "n_cells_at_or_above_0_69": n_over,
            "w0_7_refuter_threshold": thr,
            "baseline_correction": (
                "the cited 0.69 is the maximum of %d summary cells whose mean is %.4f (chance is "
                "0.50). The level a resonance-informed feature must clear is stated here as the "
                "95th percentile of the same distribution, %.4f, with the per-cell matched "
                "comparison as the primary readout." % (len(summ), s["mean"], pct["p95"])),
            "max_cell_note": ("the best cell uses bank anti32, not core32 — a resonance-informed "
                              "feature motivated by the CORE is not the same object as the cell "
                              "that produced 0.69"),
            "stage_2_requirement": {
                "outcome": M.COVERAGE_IMPOSSIBLE,
                "evidence": ("W0-7 banked per-cell AUCs (%d summary, %d raw) and no per-ITEM "
                             "scores; the AUC of a recombined resonance-weighted score cannot be "
                             "computed from per-cell AUCs" % (len(summ), len(raw))),
                "unlock": "re-score the B1 corpora with the resonance features (needs forwards)"}}


def main():
    ap = argparse.ArgumentParser(description="MERCATOR Wave 0 — atlas-derived arms")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arm = a.arm
    meta = P.ARMS[arm]

    cache = H.cache_dir(out, "mercator_w0")
    exp = M.fetch_expectations(cache / "expectations.jsonl")
    expected_ref = exp.require(arm)          # §3: the arm cannot run before its row exists

    import pandas as pd
    import subprocess
    atlas_local = cache / "darwin_atlas.parquet"
    if not atlas_local.exists():
        subprocess.run(["aws", "s3", "cp", ATLAS_S3, str(atlas_local), "--region", "us-east-1"],
                       check=True)
    df = pd.read_parquet(atlas_local)
    print(f"[{arm}] darwin atlas: {len(df)} rows, {df['arm'].nunique()} arms", flush=True)

    at = M.Atlas(out, arm=arm, run_id=a.run_id or arm.lower().replace("-", "_"),
                 register=meta["register"], wave=meta["wave"], expected_ref=expected_ref,
                 queue_item=meta["queue_item"])
    sur = M.SurpriseLog(out)
    gates = H.Gates(out)

    res = {"M-1": lambda: run_m1(df, at, sur),
           "M-9a": lambda: run_m9a(df, at, sur),
           "M-mine": lambda: run_mmine(df, at, sur, expected_ref),
           "M-C6": lambda: run_mc6(df, at, sur)}[arm]()

    if arm == "M-1":
        at.check_dual_basis(["census_avoidance_ratio"])
        gates.require("m1_all_attention_types_present",
                      all(set(ATTN_TYPES) <= set(d["by_type"]) for d in res["by_model"].values()),
                      n_models=len(res["by_model"]))
        gates.record("m1_refuter_evaluable", res["refuter_evaluable"],
                     true_mha_families=res["true_mha_families"])

    info = at.to_parquet()
    sur.to_parquet()
    M.write_json(out, f"{arm}_result.json",
                 {"arm": arm, "queue_item": meta["queue_item"], "register": meta["register"],
                  "wave": meta["wave"], "refuter": meta["refuter"], "expected_ref": expected_ref,
                  "provenance": M.provenance(), "utc": M.utc_now(),
                  "atlas": info, "source": ATLAS_S3, "result": res})
    print(f"{arm}_OK " + json.dumps({"rows": info["n_rows"], "expected_ref": expected_ref}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
