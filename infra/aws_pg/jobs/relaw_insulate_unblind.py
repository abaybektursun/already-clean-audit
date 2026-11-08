"""RELAW Arm-3 unblind — aggregate the insulation cells → RF contrasts + the §6 causal verdict.

Reads the insulate cells (mode ∈ {standard, off, into} × seeds) + the LATMECH floor/ceiling cells
(random_floor@16k, deleted@16k, SAME eval-split) banked per item; computes RF per mode and the
paired item cluster-bootstrap CIs of RF(off)−RF(standard) and RF(off)−RF(into)
(latmech.rf_diff_contrast — items paired by canonical position), and applies the §6 rule:
INSULATION WORKS iff both CIs > 0, seed-stable. Pure CPU, weight-free, fail-loud.

    python -m infra.aws_pg.jobs.relaw_insulate_unblind --out <dir> \
        --cells-s3 s3://.../relaw-insulate/out --split confirmation \
        --floor-s3 s3://.../random_floor_n16000/cell_random_floor_cap_r512_n16000_s1337_confirmation.json \
        --ceiling-s3 s3://.../deleted_n16000/cell_deleted_cap_r512_n16000_s1337_confirmation.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def _cp(s3, dest):
    subprocess.run(["aws", "s3", "cp", s3, str(dest), "--region", "us-east-1"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells-s3", required=True, help="prefix holding cell_insulate_*.json")
    ap.add_argument("--split", default="confirmation")
    ap.add_argument("--floor-s3", required=True)
    ap.add_argument("--ceiling-s3", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from infra.pg import latmech as Lm

    cdir = out / "cells"
    cdir.mkdir(exist_ok=True)
    # --cells-s3 may be a comma-list of prefixes (one per mode/seed cell lives in its own job prefix).
    for pre in [p.strip() for p in a.cells_s3.split(",") if p.strip()]:
        subprocess.run(["aws", "s3", "cp", pre, str(cdir), "--recursive",
                        "--exclude", "*", "--include", "*cell_insulate_*.json", "--region", "us-east-1"],
                       check=True)
    _cp(a.floor_s3, out / "floor.json")
    _cp(a.ceiling_s3, out / "ceiling.json")

    floor = np.array(json.loads((out / "floor.json").read_text())["mcq"]["per_item"], float)
    ceil = np.array(json.loads((out / "ceiling.json").read_text())["mcq"]["per_item"], float)

    cells = {}
    for p in sorted(cdir.rglob("cell_insulate_*.json")):
        r = json.loads(p.read_text())
        if r["eval_split"] != a.split:
            continue
        cells[(r["mode"], int(r["seed"]))] = np.array(r["mcq"]["per_item"], float)
    modes = sorted({m for m, _ in cells})
    seeds = sorted({s for _, s in cells})
    print(f"[RELAW-INSULATE-UNBLIND] modes={modes} seeds={seeds} cells={len(cells)}", flush=True)
    assert all(m in modes for m in ("standard", "off", "into")), f"missing modes: {modes}"

    def rf(S):
        denom = ceil.mean() - floor.mean()
        assert denom > 0, "ceiling≈floor — RF undefined"
        return float((S.mean() - floor.mean()) / denom)

    per_seed = {}
    for s in seeds:
        if not all((m, s) in cells for m in ("standard", "off", "into")):
            continue
        off_S, std_S, into_S = cells[("off", s)], cells[("standard", s)], cells[("into", s)]
        assert off_S.shape == std_S.shape == into_S.shape == floor.shape == ceil.shape, \
            "per-item arrays not position-aligned across modes/floor/ceiling — HALT"
        per_seed[s] = {
            "RF_standard": rf(std_S), "RF_off": rf(off_S), "RF_into": rf(into_S),
            "off_minus_standard": Lm.rf_diff_contrast(off_S, std_S, floor, ceil, n_boot=a.n_boot, seed=s),
            "off_minus_into": Lm.rf_diff_contrast(off_S, into_S, floor, ceil, n_boot=a.n_boot, seed=s),
        }
        d = per_seed[s]
        print(f"[RELAW-INSULATE-UNBLIND] seed {s}: RF off={d['RF_off']:+.3f} std={d['RF_standard']:+.3f} "
              f"into={d['RF_into']:+.3f} | off−std=[{d['off_minus_standard']['lo']:+.3f},"
              f"{d['off_minus_standard']['hi']:+.3f}] off−into=[{d['off_minus_into']['lo']:+.3f},"
              f"{d['off_minus_into']['hi']:+.3f}]", flush=True)

    # pooled across seeds (concatenate per-item arrays) — the headline CI.
    off_all = np.concatenate([cells[("off", s)] for s in seeds if ("off", s) in cells])
    std_all = np.concatenate([cells[("standard", s)] for s in seeds if ("standard", s) in cells])
    into_all = np.concatenate([cells[("into", s)] for s in seeds if ("into", s) in cells])
    floor_all = np.tile(floor, len([s for s in seeds if ("off", s) in cells]))
    ceil_all = np.tile(ceil, len([s for s in seeds if ("off", s) in cells]))
    pooled = {
        "off_minus_standard": Lm.rf_diff_contrast(off_all, std_all, floor_all, ceil_all, n_boot=a.n_boot, seed=1337),
        "off_minus_into": Lm.rf_diff_contrast(off_all, into_all, floor_all, ceil_all, n_boot=a.n_boot, seed=1337),
    }

    def signs(key):
        return [1 if per_seed[s][key]["point"] > 0 else -1 for s in per_seed]
    seed_stable_std = len(set(signs("off_minus_standard"))) == 1
    seed_stable_into = len(set(signs("off_minus_into"))) == 1
    works = bool(pooled["off_minus_standard"]["lo"] > 0 and pooled["off_minus_into"]["lo"] > 0
                 and seed_stable_std and seed_stable_into)
    only_std = bool(pooled["off_minus_standard"]["lo"] > 0 and not (pooled["off_minus_into"]["lo"] > 0))
    named = ("INSULATION-WORKS" if works
             else "INSULATION-CONFOUNDED-BY-SURGERY" if only_std
             else "INSULATION-NULL")

    rep = {"object": "relaw_insulate_unblind_v1", "split": a.split, "modes": modes, "seeds": seeds,
           "per_seed": per_seed, "pooled": pooled,
           "seed_stable": {"off_minus_standard": seed_stable_std, "off_minus_into": seed_stable_into},
           "named": named,
           "grid": {"Arm3_G_off_minus_standard": [round(pooled["off_minus_standard"]["lo"], 4),
                                                  round(pooled["off_minus_standard"]["hi"], 4)],
                    "Arm3_G_off_minus_into": [round(pooled["off_minus_into"]["lo"], 4),
                                              round(pooled["off_minus_into"]["hi"], 4)],
                    "named_arm3_verdict": named},
           "frozen_rule": ("INSULATION-WORKS iff RF(off)−RF(standard) CI>0 AND RF(off)−RF(into) CI>0, "
                           "seed-stable; only >standard ⇒ CONFOUNDED-BY-SURGERY; else NULL.")}
    (out / "relaw_insulate_metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("=" * 78, flush=True)
    print(f"[RELAW-INSULATE-UNBLIND] pooled off−standard=[{pooled['off_minus_standard']['lo']:+.3f},"
          f"{pooled['off_minus_standard']['hi']:+.3f}] off−into=[{pooled['off_minus_into']['lo']:+.3f},"
          f"{pooled['off_minus_into']['hi']:+.3f}]", flush=True)
    print(f"[RELAW-INSULATE-UNBLIND] VERDICT: {named}", flush=True)
    print(f"RELAW_INSULATE_UNBLIND_OK verdict={named}", flush=True)


if __name__ == "__main__":
    main()
