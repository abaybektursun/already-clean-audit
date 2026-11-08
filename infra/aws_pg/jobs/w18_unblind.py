"""W18 UNBLIND AGGREGATOR (control-plane; frozen ordinal procedure; ONE read).

  * RESTORATION gate (critic M4 label-split; distinct from the controls-job
    RECIPE gate): the RMU r=512 × n_max held-out S, 3 seeds + shared-index
    bootstrap — HARNESS-INDICTED HALT only if (a) the controls positive
    control failed (it gates controls already) or (b) this S sits inside the
    FLOOR band (point < 0.15 AND CI upper < 0.35; power-calibrated). Failure
    to cross 0.5 with a healthy harness
    routes to the REVERSAL branch as a science datum (with the RTT dose-gap
    disclosure: their anchor is full-FT × 6 epochs).
  * ordinal family (m = 2 + promoted probe test if any; BH q = 0.05):
      T_data: P(S_rmu(r512, n*=375, heldout, 3 seeds) ≥ 0.5). Anchor phrasing
              (critic M6, frozen): the incidental side's CONFIRMATION-grade
              S(1k) = 0.019 ≪ 0.5 ⇒ incidental n_half > 1,000 > 375 with no
              cross-system score arithmetic; the DISCOVERY-grade interpolation
              (8.4–10.4k, CI lower bound 6,095 log-space) is cited by pointer.
      T_rank: same at r=64 × n_max (engineered r_min ≤ 64 vs incidental
              no-recovery-below-r512-at-its-frozen-recipe — DISCOVERY-grade,
              capacity-vs-optimization unresolved per W17 critic M4).
      r=8: scheduled and read with the same machinery, DESCRIPTIVE.
  * verdict: SUPPORTED iff both BH-significant; PARTIAL if data-axis only;
    REVERSAL otherwise (W14 re-grade gate fires per §1).

    python -m infra.aws_pg.jobs.w18_unblind --out results/w18 \
        --controls <w18_controls.json> --cells-dir <dir> [--probes <w18_probes.json>]
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEEDS = (1337, 1338, 1339)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--probes", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import w16_probes as W
    from infra.pg.w16_probes import _boot_draws, bh_fdr

    C = json.loads(pathlib.Path(args.controls).read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    hashes = C["item_hashes"]
    pool_h = set(C["pool_hashes"])
    b_pi, f_pi = C["BASE"]["per_item"], C["FLOOR"]["per_item"]

    def ep(h):
        return h not in pool_h

    rows = {str(p): json.loads(p.read_text())
            for p in sorted(pathlib.Path(args.cells_dir).rglob("cell_*.json"))}
    ids = [r["cell_id"] for r in rows.values()]
    assert len(ids) == len(set(ids)), "duplicate cell_ids"

    def find(model, rank, n, seed, split):
        cid = f"{model}_cap_r{rank}_n{n}_s{seed}_{split}"
        m = [r for r in rows.values() if r["cell_id"] == cid]
        assert len(m) == 1, f"{cid}: {len(m)}"
        return m[0]

    def masked(arr, pred):
        return [v for v, h in zip(arr, hashes) if ep(h) and pred(h)]

    # ---- populations ----
    held_h = set(C["heldout_hashes"])
    held = lambda h: h in held_h
    b_c = np.asarray(masked(b_pi, held), float)
    f_c = np.asarray(masked(f_pi, held), float)
    held_hashes = [h for h in hashes if held(h) and ep(h)]
    den_held = float(b_c.mean() - f_c.mean())
    assert den_held > 0.05, f"held-out denom {den_held:.4f} ≤ 0.05 — scale HALT"

    def cross_test(rank, n):
        cells = [find("rmu", rank, n, s, "heldout") for s in SEEDS]
        for c in cells:
            assert c["per_item_hash"] == held_hashes, f"{c['cell_id']}: identity drift — HALT"
        per_seed = [np.asarray(c["per_item"], float) for c in cells]
        k = len(per_seed)
        rng = np.random.default_rng(424242)        # child seed ≠ item-index stream (critic m2)

        def stat(idx):
            den = b_c[idx].mean() - f_c[idx].mean()
            if den <= 0:
                return None
            picks = rng.integers(0, k, k)
            a = np.mean([per_seed[j][idx].mean() for j in picks])
            return (a - f_c[idx].mean()) / den

        draws, dropped = _boot_draws(stat, b_c.size, args.n_boot, 1337)
        point = float((np.mean([p.mean() for p in per_seed]) - f_c.mean())
                      / (b_c.mean() - f_c.mean()))
        p = float((1 + (draws < 0.5).sum()) / (1 + draws.size))
        return {"point_S": point, "ci95": [float(np.percentile(draws, 2.5)),
                                           float(np.percentile(draws, 97.5))],
                "p_one_sided_ge_0.5": p, "n_draws_dropped": int(dropped),
                "per_seed_S": [float((p_.mean() - f_c.mean()) / (b_c.mean() - f_c.mean()))
                               for p_ in per_seed]}

    # ---- restoration gate (3 seeds + bootstrap; label-split per critic M4) ----
    g = cross_test(512, 375)
    # floor-band = point < 0.15 AND CI-upper < 0.35 (calibrated to the held-out
    # population's power: at n≈127 the CI half-width alone is ≈0.2, so a bare
    # CI-upper<0.2 criterion could never fire even at a true floor)
    in_floor_band = bool(g["point_S"] < 0.15 and g["ci95"][1] < 0.35)
    harness_ok = not in_floor_band
    result = {"controls": args.controls, "n_cells": len(rows),
              "restoration_gate": {**g, "ci_upper_in_floor_band": in_floor_band}}
    print(f"[W18-UNB] restoration gate: S={g['point_S']:.4f} CI[{g['ci95'][0]:.4f},"
          f"{g['ci95'][1]:.4f}] floor-band={in_floor_band}")
    if not harness_ok:
        result["VERDICT"] = ("HARNESS-INDICTED HALT — RMU r512×n_max sits in the FLOOR "
                             "band despite a passing recipe gate; published direction "
                             "(RTT, this checkpoint) says it restores. No verdict.")
        (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
        print(result["VERDICT"])
        return

    tests = {"T_data_r512_n375": g,
             "T_rank_r64_n375": cross_test(64, 375)}
    r8 = cross_test(8, 375)
    for t, v in tests.items():
        print(f"[W18-UNB] {t}: S={v['point_S']:.4f} CI[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] "
              f"p={v['p_one_sided_ge_0.5']:.4f}")
    print(f"[W18-UNB] r8 (descriptive): S={r8['point_S']:.4f}")

    fam = {k: v["p_one_sided_ge_0.5"] for k, v in tests.items()}
    probes = None
    if args.probes and pathlib.Path(args.probes).exists():
        probes = json.loads(pathlib.Path(args.probes).read_text())
        if probes.get("confirmation"):
            fam["probe_promoted"] = probes["confirmation"]["p_one_sided"]
    bh = bh_fdr(fam, q=args.q)

    data_sig = bh["T_data_r512_n375"]["significant"]
    rank_sig = bh["T_rank_r64_n375"]["significant"]
    if data_sig and rank_sig:
        verdict = ("H_deeper-than-engineered SUPPORTED: engineered crosses S≥0.5 at "
                   "n*=375 on its own scale (3 seeds, BH-sig) — incidental confirmation-"
                   "grade S(1k)=0.019 ≪ 0.5 ⇒ incidental n_half > 1,000 > 375, no "
                   "cross-system arithmetic (DISCOVERY interpolation 8.4–10.4k by "
                   "pointer) — AND at r=64 (incidental: no recovery below r512 at its "
                   "frozen recipe, DISCOVERY-grade)")
    elif data_sig:
        verdict = ("PARTIAL (data axis only): engineered crosses at n*=375; rank axis "
                   "ns — supported on data, indeterminate on rank")
    else:
        verdict = ("REVERSAL BRANCH (science datum, dose-gap disclosed: RTT anchor is "
                   "full-FT × 6 epochs): engineered does not cross at its max pre-stated "
                   "budget — W14 re-grade gate per §1")

    result.update({"ordinal_tests": tests, "r8_descriptive": r8,
                   "bh": {k: {kk: vv for kk, vv in v.items()} for k, v in bh.items()},
                   "probes_summary": (probes or {}).get("screening"),
                   "probes_confirmation": (probes or {}).get("confirmation"),
                   "VERDICT": verdict})
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[W18-UNB] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
