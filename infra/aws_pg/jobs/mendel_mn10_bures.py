"""MENDEL MN-10 — re-score the banked GESTALT atlases with the repaired Bures cost (item 20A).

THE ARM. GESTALT's shape-diffing payoff went VOID at L9 on an instrument failure. This re-scores
the SAME banked atlases with four cost variants and asks which, if any, passes its own split-half
stability gate. It is the INSTRUMENT for MN-13 and MN-18, so a total failure here kills those two
as well — which is why the gate has a separation clause that refuses a variant for being
uninformative rather than only for being unstable.

THE SUBSTRATE. `results/gestalt_arm_d/out/atlas_{pg1,pi0,ftmax}/site{9,14}/` and
`gestalt_arm_d2/out/atlas_ftmin/...`, each a K=32, d=2048, rank-2 splat atlas with `full_s0`,
`full_s1`, `halfA_s0`, `halfB_s0` — the split halves the identity band needs.

**site14 is EXCLUDED** by the queue's own rider ("L14 stays EXCLUDED — real and twin both floor, no
readout range; rule-formal verdicts there say nothing"). It is still SCORED here and reported
separately, because "excluded for lack of range" is a claim about the data that this arm is in a
position to check rather than inherit.

THE GATE (mendel_bures.band_statistics): ordering tau_move < tau_chance; tail p97.5/p50 <= 10;
separation tau_chance/tau_move >= 1.5. A variant passes only on all three.

KILL: no variant passes -> the shape-diffing instrument class is dead at these sites.

CPU only, banked artifacts.

FAIL LOUD: no try/except, no synthetic fallback.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

from infra.pg import gestalt_corr as GC
from infra.pg import humboldt_core as H
from infra.pg import mendel_bures as MB
from infra.pg import mendel_core as MC
from infra.pg import mendel_program as MP

ARM = "MN-10"
ARM_D = f"{MC.RESULTS_S3}/gestalt_arm_d/out"
ARM_D2 = f"{MC.RESULTS_S3}/gestalt_arm_d2/out"
MODELS = {"pg1": (ARM_D, "atlas_pg1"), "pi0": (ARM_D, "atlas_pi0"),
          "ftmax": (ARM_D, "atlas_ftmax"), "ftmin": (ARM_D2, "atlas_ftmin")}
SITES = (9, 14)
EXCLUDED_SITE = 14
N_SHUFFLE = 200


def load_atlas(path) -> dict:
    """A banked .npz splat atlas -> the correspondence view {mu, S, weight}.

    S = U * sqrt(lam) so that S Sᵀ is the rank-r covariance part, matching `gestalt_corr.atlas_np`
    exactly — the banked file is npz rather than a torch bundle, so the conversion is repeated here
    on the same formula rather than the loader being reused around a different container.
    """
    z = np.load(path)
    mu = z["mu"].astype(np.float64)
    U = z["U"].astype(np.float64)
    lam = z["lam"].astype(np.float64)
    w = np.exp(z["log_pi"].astype(np.float64))
    assert U.ndim == 3 and mu.ndim == 2 and lam.shape == U.shape[:1] + U.shape[2:], \
        f"{path}: shapes mu{mu.shape} U{U.shape} lam{lam.shape}"
    return {"mu": mu, "S": U * np.sqrt(np.clip(lam, 0.0, None))[:, None, :], "weight": w}


def main():
    ap = argparse.ArgumentParser(description="MENDEL MN-10 — the Bures repair on banked atlases")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = MP.arm(ARM)
    at = MC.Atlas(out, arm=ARM, run_id="mendel_mn10", wave=meta["wave"], queue_item=meta["item"])
    gates = H.Gates(out)
    man = MC.Manifest(H.cache_dir(out, "mendel_mn10"), out)

    cells, by_variant = [], {v: [] for v in MB.VARIANTS}
    for model, (prefix, folder) in MODELS.items():
        for site in SITES:
            names = {half: f"{model}_site{site}_{half}"
                     for half in ("halfA_s0", "halfB_s0", "full_s0")}
            paths = {h: man.require(n, f"{prefix}/{folder}/site{site}/{h}.npz")
                     for h, n in names.items()}
            A = load_atlas(paths["halfA_s0"])
            B = load_atlas(paths["halfB_s0"])
            for variant in MB.VARIANTS:
                C = MB.cost_matrix_variant(A, B, variant)
                _ri, ci, matched = GC.hungarian(C)
                chance = GC.shuffle_chance_costs(C, assign_cols=ci, n_draws=N_SHUFFLE,
                                                 seed=H.cell_seed(a.seed, ARM, model, site))
                st = MB.band_statistics(matched, chance, variant)
                row = {"model": model, "site": site, "excluded_by_queue": site == EXCLUDED_SITE,
                       **st}
                cells.append(row)
                if site != EXCLUDED_SITE:
                    by_variant[variant].append(st["passes"])
                at.add("bures_band_width_ratio", value=st["width_ratio"], layer=site,
                       item_id=f"{model}/{variant}", control_id=MC.TREATMENT,
                       flags={"variant": variant, "model": model, "passes": st["passes"],
                              "ordering_ok": st["ordering_ok"], "tail_ok": st["tail_ok"],
                              "separation_ok": st["separation_ok"],
                              "excluded_by_queue": site == EXCLUDED_SITE})
                at.add("bures_band_separation", value=st["separation"], layer=site,
                       item_id=f"{model}/{variant}", control_id=MC.STRUCTURED,
                       flags={"variant": variant, "tau_move": st["tau_move"],
                              "tau_chance": st["tau_chance"]})
            print(f"[mn10] {model} site{site}: " + " ".join(
                f"{v}={'PASS' if [c for c in cells if c['model']==model and c['site']==site and c['variant']==v][0]['passes'] else 'fail'}"
                for v in MB.VARIANTS), flush=True)

    # Does log1p leave the assignment untouched, as its claim requires? Checked on real atlases,
    # not only on the synthetic dry-run.
    A = load_atlas(man.entries["pg1_site9_halfA_s0"]["local"])
    B = load_atlas(man.entries["pg1_site9_halfB_s0"]["local"])
    same = MB.assignment_is_unchanged(MB.cost_matrix_variant(A, B, "original"),
                                      MB.cost_matrix_variant(A, B, "log1p"))
    gates.record("mn10_log1p_preserves_the_assignment", same,
                 note=("RECORDED, not required. A monotone map on entries does NOT preserve a "
                       "Hungarian assignment, because Hungarian minimises a SUM — this module "
                       "claimed otherwise until the banked atlases falsified it. Whether log1p "
                       "changed any decision here is an empirical fact about these atlases."))

    passing = {v: (sum(p for p in b), len(b)) for v, b in by_variant.items()}
    winners = sorted(v for v, (n, tot) in passing.items() if tot > 0 and n == tot)
    verdict = MP.check_verdict(ARM, "REPAIRED" if winners else "NO-VARIANT-PASSES")
    gates.record("mn10_a_variant_passes_everywhere", bool(winners), winners=winners,
                 passing=passing, excluded_site=EXCLUDED_SITE,
                 note="site14 is scored but excluded from the verdict by the queue's own rider")
    info = at.to_parquet()
    man.write()
    MC.write_metrics(out, {
        "arm": ARM, "verdict": verdict, "queue_item": meta["item"],
        "variants": list(MB.VARIANTS), "bars": {"width_ratio": MB.WIDTH_RATIO_BAR,
                                                "separation": MB.SEPARATION_BAR,
                                                "relative_tau_mult": MB.RELATIVE_TAU_MULT},
        "models": sorted(MODELS), "sites": list(SITES), "excluded_site": EXCLUDED_SITE,
        "passing_by_variant": passing, "winners": winners,
        "median_separation_by_cell": {f"{c['model']}/site{c['site']}/{c['variant']}":
                                      c["median_separation"] for c in cells},
        "n_cells_where_median_separates": sum(1 for c in cells if c["median_separates"]),
        "log1p_preserves_assignment": bool(same),
        "cells": cells, "atlas": info,
        "instrument_for": ["MN-13", "MN-18"],
        "consequence": ("MN-13 and MN-18 have an instrument" if winners else
                        "the shape-diffing instrument class is dead at these sites; MN-13 and "
                        "MN-18 lose their instrument")})
    print("MN10_OK " + json.dumps({"verdict": verdict, "winners": winners,
                                   "passing": {k: list(v) for k, v in passing.items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
