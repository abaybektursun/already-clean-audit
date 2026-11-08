"""GESTALT Arm D adjudication — correspondence, bands, labels, survival, S_norm. CPU-only.
Inputs: the gestalt_atlas banks for PG1 + the targets. Frozen rules: journal FR-6; sealed
predictions: gestalt_sealed_predictions.json (P6, committed at freeze 1c7ae99).

Per (target, site):
  tau_move   = p97.5 of PG1's split-half self-match costs (P4; ZERO width => Arm D VOID).
  tau_chance = p2.5 of the shuffled-assignment distribution of the PG1->target cost matrix.
  survival(target) = frac of PG1 full-atlas components labeled `survived` vs target's full atlas.
  ceiling = PG1 split-half self-survival; floor = survival of PG1 vs twin-of-TARGET.
  S_norm = (survival - floor)/(ceiling - floor); seed-pairs -> band.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.gestalt_diff --type r7i.2xlarge --no-weights \
      --args "--targets pi0,pi05 --atlas-prefixes pg1=results/gestalt_atlas_pg1,pi0=..." \
      --s3-prefix results/gestalt_diff --budget-min 120

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

SITES = (9, 14)


def _pull(prefix, dest):
    """`prefix` = FULL path under the bucket to an atlas OUT dir, or `local:<dir>`."""
    if prefix.startswith("local:"):
        return pathlib.Path(prefix[len("local:"):])
    subprocess.run(["aws", "s3", "sync",
                    f"s3://pg-lineage-769063704696/{prefix}", str(dest),
                    "--region", "us-east-1", "--only-show-errors"], check=True)
    return dest


def _load_atlas(npz_path):
    import numpy as np
    import torch
    z = np.load(npz_path)
    return {k: torch.from_numpy(z[k]) for k in ("log_pi", "mu", "U", "lam", "sigma2")}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", required=True, help="comma list, e.g. pi0,pi05,ft-x,ft-n")
    ap.add_argument("--atlas-prefixes", required=True,
                    help="unit=prefix comma list incl. pg1")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import gestalt_corr as GC
    from infra.pg import gestalt_splat as GS

    targets = [t.strip() for t in a.targets.split(",") if t.strip()]
    prefixes = dict(kv.split("=", 1) for kv in a.atlas_prefixes.split(","))
    assert "pg1" in prefixes, "need the pg1 atlas bank"
    rep = {"object": "GESTALT Arm D adjudication", "targets": targets,
           "atlas_prefixes": prefixes, "sealed_predictions": "gestalt_sealed_predictions.json @ freeze 1c7ae99"}
    cells = []
    rep["cells"] = cells

    def _w():
        (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    _w()

    banks = {}
    for unit, prefix in prefixes.items():
        banks[unit] = _pull(prefix, pathlib.Path(f"/tmp/gestalt_atlases/{unit}"))

    for L in SITES:

        def np_atlas(unit, tag, seed):
            p = banks[unit] / f"site{L}" / f"{tag}_s{seed}.npz"
            if not p.exists():
                return None
            return GC.atlas_np(_load_atlas(p))

        pgA = np_atlas("pg1", "halfA", 0)
        pgB = np_atlas("pg1", "halfB", 0)
        assert pgA is not None and pgB is not None, f"pg1 split-half atlases missing at site {L}"
        Cself = GC.cost_matrix(pgA, pgB)
        _, ci_self, mc_self = GC.hungarian(Cself)
        band = GC.selfhalf_band(mc_self)
        site_rec = {"site": L, "p4_band": band}
        if band["zero_width"]:
            site_rec["VOID"] = "P4 zero-width band — Arm D VOID at this site (identity-band rule)"
            cells.append(site_rec)
            _w()
            continue
        tau_move = band["tau_move"]
        # ceiling: PG1 split-half self-survival under its own tau (chance from the self matrix)
        chance_self = float(np.percentile(GC.shuffle_chance_costs(Cself, assign_cols=ci_self),
                                          GC.TAU_CHANCE_PCTL))
        lab_self = GC.label_pair(Cself, tau_move, chance_self)
        ceiling = lab_self["frac_survived"]
        site_rec["ceiling_selfhalf_survival"] = ceiling
        site_rec["self_label"] = {k: lab_self[k] for k in
                                  ("frac_survived", "frac_moved", "frac_died", "void_banded")}
        site_rec["self_labels_list"] = lab_self["labels"]     # component-bootstrap input
        for tgt in targets:
            trec = {"target": tgt, "site": L, "seed_pairs": []}
            floors = []
            for s_pg in (0, 1):
                pg_full = np_atlas("pg1", "full", s_pg)
                if pg_full is None:
                    continue
                for s_t in (0, 1):
                    t_full = np_atlas(tgt, "full", s_t)
                    t_twin = np_atlas(tgt, "twin", s_t)
                    if t_full is None or t_twin is None:
                        trec["seed_pairs"].append({"s_pg": s_pg, "s_t": s_t,
                                                   "VOID": "missing/void atlas"})
                        continue
                    Cx = GC.cost_matrix(pg_full, t_full)
                    _, ci_x, _ = GC.hungarian(Cx)
                    tau_chance = float(np.percentile(
                        GC.shuffle_chance_costs(Cx, assign_cols=ci_x), GC.TAU_CHANCE_PCTL))
                    lab = GC.label_pair(Cx, tau_move, tau_chance)
                    Cf = GC.cost_matrix(pg_full, t_twin)
                    _, ci_f, _ = GC.hungarian(Cf)
                    tau_chance_f = float(np.percentile(
                        GC.shuffle_chance_costs(Cf, assign_cols=ci_f), GC.TAU_CHANCE_PCTL))
                    lab_f = GC.label_pair(Cf, tau_move, tau_chance_f)
                    floors.append(lab_f["frac_survived"])
                    s_norm = GC.survival_norm(lab["frac_survived"], lab_f["frac_survived"],
                                              ceiling)
                    trec["seed_pairs"].append({
                        "s_pg": s_pg, "s_t": s_t,
                        "survival": lab["frac_survived"], "moved": lab["frac_moved"],
                        "died": lab["frac_died"], "merged": lab["frac_merged"],
                        "void_banded": lab["void_banded"],
                        "twin_floor": lab_f["frac_survived"], "s_norm": s_norm,
                        "labels": lab["labels"], "labels_floor": lab_f["labels"],
                        "matched_costs": lab["matched_costs"],
                        "tau_move": tau_move, "tau_chance": tau_chance})
                    print(f"[GD] site{L} {tgt} s{s_pg}/{s_t}: surv={lab['frac_survived']:.3f} "
                          f"floor={lab_f['frac_survived']:.3f} ceil={ceiling:.3f} "
                          f"S_norm={s_norm:.3f}", flush=True)
            valid = [p for p in trec["seed_pairs"] if "s_norm" in p and p["s_norm"] == p["s_norm"]]
            if valid:
                sn = [p["s_norm"] for p in valid]
                sv = [p["survival"] for p in valid]
                trec["summary"] = {"s_norm_mean": float(np.mean(sn)),
                                   "s_norm_min": float(np.min(sn)),
                                   "s_norm_max": float(np.max(sn)),
                                   "survival_mean": float(np.mean(sv)),
                                   "n_valid_pairs": len(valid)}
            site_rec.setdefault("targets", []).append(trec)
            _w()
        cells.append(site_rec)
        _w()

    rep["elapsed_s"] = time.time() - t0
    _w()
    print("GESTALT_DIFF_OK", flush=True)


if __name__ == "__main__":
    main()
