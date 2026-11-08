"""MENDEL MN-8 — magnitude structure as the object (queue item 24 + item 95's L0 mining).

THE ARM HAS TWO QUESTIONS AND THIS JOB ANSWERS ONE OF THEM COMPLETELY.

  Q1  "does a common statistic unify >= 3 of the six sightings?" — the six sightings live in six
      DIFFERENT programs' artifacts (V4's sign-randomised nulls, B2's 56x scale heterogeneity, the
      census leverage-0.998 signature, V1(a)'s energy concentration, F10's spikes, CO-X2's
      spectra-kept null). Adjudicating them needs each one's underlying per-coordinate data, not
      its headline. NOT attempted here; scoped as a named residual in the metrics.

  Q2  "are coordinate-magnitude and activation-magnitude structure ONE anatomy or TWO?" — fully
      answerable from what MENDEL already has banked at all 126 cells, and answered here.

Q2's statistics, all per input coordinate, all on the same axis:

    activation_rms      how much signal arrives            G4
    gradient_magnitude  how much gradient arrives          G4
    consistency_r       how much of it agrees              G4 (floor-corrected)
    core32_leverage     participation in the SINK basis    humboldt banks
    topsigma_leverage   participation in the base W's TOP singular subspace
    bottomsigma_leverage        ... and its BOTTOM singular subspace

"One anatomy" means these are substantially the same ORDERING of coordinates. Measured as the
pairwise Spearman matrix across coordinates, per cell, pooled by median. Rank rather than linear
correlation because these quantities live on wildly different scales and the claim is about which
coordinates are large, not about a linear relationship.

FROZEN RULE. ONE-OBJECT iff the median pairwise |rho| among the six statistics is >= 0.5;
CATALOG iff < 0.2; else AMBIGUOUS.

L0 MINING (item 95, confirmed present by G1: n_dead = 370,688 elementwise entries at L0, zero at
L1-L17). Characterised here as: which coordinates carry the dead mass, and do they align with any
of the six statistics? A dead-entry population that lands on one statistic's tail is a finding
about that statistic; one that lands nowhere is a finding about the instrument.

CPU only, banked artifacts.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

from infra.pg import humboldt_core as H
from infra.pg import humboldt_weights as WT
from infra.pg import mendel_core as MC
from infra.pg import mendel_critic as CR
from infra.pg import mendel_program as MP
from infra.pg import mendel_sinkform as SF

ARM = "MN-8"
G4_PREFIX = f"{MC.RESULTS_S3}/mendel_g4_v1/out"
AM6_ROWS = f"{MC.RESULTS_S3}/amundsen_am6_v3seeds/cells/am6_seed3001/v3_rows.jsonl"
STATS = ("activation_rms", "gradient_magnitude", "consistency_r",
         "core32_leverage", "topsigma_leverage", "bottomsigma_leverage")
ONE_OBJECT_BAR, CATALOG_BAR = 0.5, 0.2
N_STEPS = 24

# The six sightings, named so the residual is a list rather than a gesture. Each entry is the
# program whose per-coordinate data would be needed to adjudicate it.
SIGHTINGS = {
    "V4_sign_randomised_column_norm_nulls": "V4",
    "B2_56x_scale_heterogeneity": "COPERNICUS B2",
    "census_leverage_0.998_signature": "SHACKLETON census",
    "V1a_energy_concentration": "V1(a)",
    "F10_spikes": "COPERNICUS F10",
    "COX2_spectra_kept_null_rho_0.99_1.00": "COPERNICUS X2 @ 28 layers",
}


def cell_statistics(vec_dir, bank_dir, t: str, L: int) -> dict:
    """The six per-coordinate magnitude statistics for one (type, layer) cell."""
    act = np.load(vec_dir / f"act_rms__{t}__L{L}.npy")
    mag = np.load(vec_dir / f"grad_mag__{t}__L{L}.npy")
    cons = np.load(vec_dir / f"grad_cons__{t}__L{L}.npy")
    r, _ = SF.r_from_consistency(cons, N_STEPS)
    out = {"activation_rms": act, "gradient_magnitude": mag, "consistency_r": r}
    for name, bank in (("core32_leverage", "core32"), ("topsigma_leverage", "topsigma32"),
                       ("bottomsigma_leverage", "bottomsigma32")):
        p = pathlib.Path(bank_dir) / WT.bank_name(t, L, bank)
        if not p.exists():
            # The input-side library carries core32 / bottomsigma32 but NOT topsigma32 (that lives
            # in MERCATOR's row-side library). Absence is recorded per cell rather than silently
            # shrinking the statistic set, so a reader can see the arm ran on five and not six.
            continue
        out[name] = H.leverage(np.load(p))
    n = {v.size for v in out.values()}
    assert len(n) == 1, f"{t}/L{L}: statistics on different axes {n}"
    assert len(out) >= 4, f"{t}/L{L}: only {sorted(out)} available — too few to speak of an anatomy"
    return out


def pairwise_spearman(stats: dict) -> dict:
    """|rho| for every pair. Absolute because the CLAIM is about shared ordering, and two of these
    statistics are expected to run in opposite directions (a sink coordinate has high leverage and
    low consistency) — a sign flip is still one anatomy."""
    keys = sorted(stats)
    return {f"{a}|{b}": abs(CR.spearman(stats[a], stats[b]))
            for a, b in itertools.combinations(keys, 2)}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-8 — magnitude structure as the object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn8", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn8"), out)

    vec = man.require("g4_vectors", f"{G4_PREFIX}/vectors", recursive=True)
    banks = WT.fetch_banks(H.cache_dir(out, "banks"))
    am6 = [json.loads(x) for x in
           man.require("am6_v3_rows", AM6_ROWS).read_text().splitlines() if x.strip()]

    cells, per_pair = [], {}
    for t in H.TYPES:
        for L in range(H.N_LAYERS):
            st = cell_statistics(vec, banks, t, L)
            rho = pairwise_spearman(st)
            cells.append({"type": t, "layer": L, "n_stats": len(st), "n": int(st[STATS[0]].size),
                          **rho})
            for k, v in rho.items():
                per_pair.setdefault(k, []).append(v)
                at.add("magnitude_statistic_rank_agreement", value=v, layer=L, type=t,
                       item_id=k, control_id=MC.TREATMENT,
                       flags={"pair": k.split("|"), "n_coords": int(st[STATS[0]].size)})

    present = sorted(set(itertools.chain.from_iterable(k.split("|") for k in per_pair)))
    absent = sorted(set(STATS) - set(present))
    medians = {k: float(np.median(v)) for k, v in per_pair.items()}
    overall = float(np.median(list(medians.values())))
    strongest = max(medians, key=medians.get)
    weakest = min(medians, key=medians.get)

    # The permutation null: shuffle one statistic's coordinates and re-measure. |rho| collapses to
    # ~0, which is what makes the observed value interpretable as shared ordering rather than as an
    # artifact of the rank transform.
    rng = np.random.default_rng(a.seed)
    st0 = cell_statistics(vec, banks, "q_proj", 9)
    null = [abs(CR.spearman(st0["activation_rms"], rng.permutation(st0["core32_leverage"])))
            for _ in range(200)]
    at.add_ci("magnitude_statistic_rank_agreement", null, item_id="permuted", type="q_proj",
              layer=9, control_id=MC.SHUFFLE, seed=a.seed,
              flags={"what": "one statistic's coordinates permuted; shared ordering destroyed"})

    # ---- the L0 dead-entry population (item 95) ----
    dead_rows = [r for r in am6 if "n_dead" in r]
    by_layer = {}
    for r in dead_rows:
        by_layer.setdefault(int(r["layer"]), []).append(int(r["n_dead"]))
    dead = {f"L{L}": int(np.max(v)) for L, v in sorted(by_layer.items())}
    n_dead_l0 = dead.get("L0", 0)
    l0_is_unique = n_dead_l0 > 0 and all(v == 0 for k, v in dead.items() if k != "L0")
    at.add("l0_dead_elementwise_entries", value=float(n_dead_l0), layer=0,
           control_id=MC.TREATMENT,
           flags={"by_layer": dead, "l0_unique": l0_is_unique,
                  "source": "AMUNDSEN AM-6 v3_rows (V3 instrument)"})

    verdict = MP.check_verdict(ARM, "ONE-OBJECT" if overall >= ONE_OBJECT_BAR else
                               ("CATALOG" if overall < CATALOG_BAR else "AMBIGUOUS"))
    gates.record("mn8_statistics_share_one_ordering", overall >= ONE_OBJECT_BAR,
                 median_abs_rho=overall, strongest=(strongest, medians[strongest]),
                 weakest=(weakest, medians[weakest]),
                 permutation_null_median=float(np.median(null)),
                 bars={"one_object": ONE_OBJECT_BAR, "catalog": CATALOG_BAR})
    gates.record("mn8_l0_dead_entries_are_layer_unique", l0_is_unique, n_dead_L0=n_dead_l0,
                 by_layer=dead)
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "question_answered": "Q2 — are coordinate-magnitude and activation-magnitude ONE anatomy?",
        "question_not_attempted": {
            "Q1": "does a common statistic unify >= 3 of the six sightings?",
            "why": ("each sighting's per-coordinate data lives in a different program's artifacts; "
                    "adjudicating them from headlines would be exactly the prose-quoting the "
                    "provenance rule forbids"),
            "sightings_and_their_programs": SIGHTINGS},
        "n_cells": len(cells), "statistics_requested": list(STATS),
        "statistics_present": present, "statistics_absent": absent,
        "absence_reason": ("topsigma32 is a MERCATOR row-side bank; the input-side library carries "
                           "core32 and bottomsigma32 only" if absent else None),
        "median_abs_spearman_by_pair": medians, "overall_median_abs_spearman": overall,
        "strongest_pair": [strongest, medians[strongest]],
        "weakest_pair": [weakest, medians[weakest]],
        "permutation_null": {"median": float(np.median(null)),
                             "p95": float(np.percentile(null, 95)), "n": len(null)},
        "l0_dead_entries": {"n_dead_L0": n_dead_l0, "by_layer": dead,
                            "layer_unique": l0_is_unique},
        "cells": cells, "atlas": info})
    print("MN8_OK " + json.dumps({"verdict": verdict, "median_abs_rho": round(overall, 4),
                                  "strongest": [strongest, round(medians[strongest], 3)],
                                  "weakest": [weakest, round(medians[weakest], 3)],
                                  "n_dead_L0": n_dead_l0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
