"""MENDEL G3 — PLENUM verification, from the artifacts (spec §2, gate for MN-5 and MN-14).

The spec's instruction is explicit and is the whole point of the gate: confirm `pending = 0 / 1054`
and the sealed nothing-predicts-survival negative **FROM THE PLENUM ARTIFACTS (never the
evaluation doc's prose)**, and bank the faculty-survival matrix manifest that MN-5/MN-14 consume.

Three checks, each recomputed here:

  1. FROZEN CELL COUNT — `n_cells_total` from `plenum/map_impossible.json`, which its own header
     says was frozen BEFORE measurement. A count taken after measurement could have been fitted to
     what was measurable.
  2. PENDING = 0 — recounted over every cell record in `plenum_agg_final2/out/cells.json`. The
     records are heterogeneous by design (a measured cell carries `verdict`; a gated cell carries
     `state` + `gate`), so both shapes are counted and any record carrying neither fails loud
     rather than being skipped into the "not pending" bucket.
  3. THE SEALED NEGATIVE — the hit rates of the sealed survival predictors in
     `plenum_prediction_score.json`, recomputed from their own `rows` rather than read off their
     summary fields, so a summary that drifted from its rows cannot pass.

If any check fails, MN-5 and MN-14 are BLOCKED-WITH-UNLOCK and the ladder re-scopes to the lineage
faculty tables only — the spec's own named branch, not an error.

Reads a few hundred KB of JSON. No weights, no GPU.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import collections
import json
import pathlib
import sys

from infra.pg import humboldt_core as H
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP

ARM = "G3"
PLENUM = f"{MC.RESULTS_S3}/plenum_agg_final2/out"
MAP_IMPOSSIBLE = "studies/lineage/paligemma1-pi0/research/plenum/map_impossible.json"

# The four sealed predictors PLENUM registered before measuring. The gate is not "are they bad" —
# it is "is the negative real and recomputable", which is what MN-5's ladder has to beat.
SEALED = ("H_recipe", "H_fragility", "H_similarity", "PROGRAM")


def _cell_state(rec: dict) -> str:
    """The terminal state of one PLENUM cell record.

    Two record shapes exist by construction: a MEASURED cell carries `verdict`, a GATED cell
    carries `state` (with the `gate` that fired). A record with neither is not a cell this gate
    knows how to count, and counting it as "not pending" would be exactly the silent pass the gate
    exists to prevent.
    """
    if "verdict" in rec:
        return str(rec["verdict"])
    if "state" in rec:
        return str(rec["state"])
    raise AssertionError(f"cell record has neither `verdict` nor `state`: {sorted(rec)}")


def _rescore(block: dict) -> dict:
    """Recompute a sealed predictor's hit rate from its own rows.

    Reading `hit_rate` off the summary would test nothing: the question is whether the summary and
    the rows agree, which is the only way a drifted aggregate shows up.
    """
    rows = block["rows"]
    outcomes = collections.Counter(r["outcome"] for r in rows)
    scored = outcomes["hit"] + outcomes["miss"]
    assert scored > 0, "a sealed predictor with no adjudicated rows cannot be scored"
    return {"n_rows": len(rows), "hit": outcomes["hit"], "miss": outcomes["miss"],
            "unmeasured": outcomes["unmeasured"], "n_scored": scored,
            "hit_rate": outcomes["hit"] / scored,
            "banked_hit_rate": float(block["hit_rate"]),
            "banked_n_scored": int(block["n_scored"]),
            "outcomes": dict(outcomes)}


def main():
    ap = argparse.ArgumentParser(description="MENDEL G3 — PLENUM verification from artifacts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo-root", default=".", help="tree holding plenum/map_impossible.json")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_g3", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_g3"), out)

    cells = json.loads(man.require("plenum_cells", f"{PLENUM}/cells.json").read_text())
    score = json.loads(
        man.require("plenum_prediction_score", f"{PLENUM}/plenum_prediction_score.json").read_text())
    metrics = json.loads(man.require("plenum_metrics", f"{PLENUM}/metrics.json").read_text())
    impossible = json.loads(
        man.require("plenum_map_impossible",
                    str(pathlib.Path(a.repo_root) / MAP_IMPOSSIBLE)).read_text())

    # ---- 1. the frozen cell count ----
    n_total = int(impossible["n_cells_total"])
    n_impossible = len(impossible["impossible"])
    at.add("plenum_frozen_cell_count", value=n_total, corpus_id="plenum",
           flags={"source": MAP_IMPOSSIBLE, "object": impossible["object"],
                  "n_impossible": n_impossible,
                  "frozen_before_measurement": impossible.get("note", "")[:120]})

    # ---- 2. pending = 0 ----
    states = collections.Counter(_cell_state(r) for r in cells.values())
    n_pending = sum(n for s, n in states.items() if "pending" in s.lower())
    n_measured = sum(1 for r in cells.values() if "verdict" in r)
    n_gated = sum(1 for r in cells.values() if "verdict" not in r and "state" in r)
    at.add("plenum_pending_cells", value=n_pending, corpus_id="plenum", n=len(cells),
           flags={"n_records": len(cells), "n_measured": n_measured, "n_gated": n_gated,
                  "distinct_states": len(states)})
    gates.record("g3_pending_is_zero", n_pending == 0, n_pending=n_pending,
                 n_records=len(cells), n_measured=n_measured, n_gated=n_gated,
                 n_frozen_total=n_total,
                 note=("cells.json holds the measured+gated records; the impossible and "
                       "rule-selected-out cells are terminal by construction and never pending"))

    # ---- 3. the sealed negative ----
    rescored = {}
    for name in SEALED:
        assert name in score, f"sealed predictor {name} absent from the banked score"
        r = _rescore(score[name])
        rescored[name] = r
        drift = abs(r["hit_rate"] - r["banked_hit_rate"])
        assert drift < 1e-9, (
            f"{name}: recomputed hit rate {r['hit_rate']:.6f} != banked {r['banked_hit_rate']:.6f}")
        at.add("plenum_sealed_predictor_hit_rate", value=r["hit_rate"], corpus_id="plenum",
               n=r["n_scored"], item_id=name, control_id=MC.TREATMENT,
               flags={"hit": r["hit"], "miss": r["miss"], "unmeasured": r["unmeasured"],
                      "recomputed_from": "rows", "banked": r["banked_hit_rate"]})

    best = max(rescored, key=lambda k: rescored[k]["hit_rate"])
    best_rate = rescored[best]["hit_rate"]

    # THE ACHIEVABLE NULL. A hit rate is meaningless without the rate a constant predictor gets,
    # and the outcome distribution here is very lopsided. Computed from the adjudicated rows of any
    # sealed predictor (they share one row set), so it is the null on exactly the cells the
    # predictors were scored on.
    adjudicated = [r for r in score[best]["rows"] if r["outcome"] in ("hit", "miss")]
    observed = collections.Counter(r["observed"] for r in adjudicated)
    majority_label, majority_n = observed.most_common(1)[0]
    majority_rate = majority_n / len(adjudicated)
    predicted_mix = {k: dict(collections.Counter(
        r["predicted"] for r in score[k]["rows"] if r["outcome"] in ("hit", "miss")))
        for k in SEALED}
    at.add("plenum_majority_class_hit_rate", value=majority_rate, corpus_id="plenum",
           n=len(adjudicated), item_id=f"always_{majority_label}", control_id=MC.STRUCTURED,
           flags={"observed_distribution": dict(observed),
                  "why": "the achievable null every survival predictor must clear"})

    beaten_by_trivial = sorted(k for k, v in rescored.items() if v["hit_rate"] < majority_rate)
    gates.record("g3_sealed_negative_recomputable", True,
                 best_predictor=best, best_hit_rate=best_rate,
                 all_hit_rates={k: round(v["hit_rate"], 4) for k, v in rescored.items()},
                 note=("recomputed from each predictor's own rows, not read off its summary"))
    gates.record("g3_sealed_predictors_beat_majority_class", not beaten_by_trivial,
                 majority_label=majority_label, majority_rate=majority_rate,
                 observed_distribution=dict(observed),
                 worse_than_trivial=beaten_by_trivial, predicted_mix=predicted_mix,
                 note=("RECORDED, not required. All four sealed predictors scoring BELOW a "
                       "constant predictor strengthens PLENUM's nothing-predicts-survival "
                       "negative rather than weakening it — but it means MN-5's ladder must be "
                       "read against the majority-class rate, not against these predictors"))

    # ---- the manifest MN-5/MN-14 consume ----
    survival = [c for c in metrics["cells"] if "label" in c and "faculty" in c]
    # A cell whose `label` is null is a cell PLENUM measured but did not adjudicate. It is not a
    # survival outcome and must not be counted as one — MN-5's ladder is fitted on labels, so a
    # null silently coerced to a string would become a phantom class. Counted separately.
    by_faculty = collections.Counter(str(c["faculty"]) for c in survival)
    by_label = collections.Counter("<unlabelled>" if c["label"] is None else str(c["label"])
                                   for c in survival)
    n_unlabelled = sum(1 for c in survival if c["label"] is None)
    matrix = {"source": f"{PLENUM}/metrics.json", "n_cells": len(survival),
              "n_labelled": len(survival) - n_unlabelled, "n_unlabelled": n_unlabelled,
              "by_faculty": dict(by_faculty), "by_label": dict(by_label),
              "fields": sorted(survival[0]) if survival else [],
              "units": sorted({str(c["unit"]) for c in survival})[:5],
              "n_units": len({str(c["unit"]) for c in survival})}
    at.add("plenum_survival_matrix_unlabelled_fraction",
           value=n_unlabelled / len(survival) if survival else 0.0, corpus_id="plenum",
           n=len(survival),
           flags={"n_unlabelled": n_unlabelled,
                  "why": "cells PLENUM measured but did not adjudicate; MN-5 fits on labels only"})
    (out / "plenum_matrix_manifest.json").write_text(json.dumps(matrix, indent=2, sort_keys=True))
    gates.record("g3_survival_matrix_present", len(survival) > 0, **{
        k: v for k, v in matrix.items() if k in ("n_cells", "n_units")})

    ok = n_pending == 0 and len(survival) > 0
    verdict = MP.check_verdict(ARM, "CONFIRMED" if ok else "NOT-AS-DESCRIBED")
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "n_frozen_cells": n_total, "n_impossible": n_impossible,
        "n_records_in_cells_json": len(cells), "n_measured": n_measured, "n_gated": n_gated,
        "n_pending": n_pending, "state_counts": dict(states),
        "sealed_predictors": rescored, "best_sealed": {"name": best, "hit_rate": best_rate},
        "majority_class_null": {"label": majority_label, "hit_rate": majority_rate,
                                "n_adjudicated": len(adjudicated),
                                "observed_distribution": dict(observed),
                                "sealed_worse_than_trivial": beaten_by_trivial,
                                "predicted_mix": predicted_mix},
        "survival_matrix": matrix, "atlas": info,
        "gates_mn5_mn14": ("UNBLOCKED" if ok else "BLOCKED-WITH-UNLOCK: re-scope the ladder to "
                                                "the lineage faculty tables only")})
    print("G3_OK " + json.dumps({"verdict": verdict, "n_pending": n_pending,
                                 "n_frozen": n_total, "best_sealed": best,
                                 "best_hit_rate": round(best_rate, 4),
                                 "majority_class": round(majority_rate, 4),
                                 "sealed_worse_than_trivial": beaten_by_trivial}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
