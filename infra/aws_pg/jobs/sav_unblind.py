"""SAV-A UNBLIND AGGREGATOR (control-plane; frozen §6 over banked rows; ONE read).

Assembles {PG1, pi0, pi05, random} × ladder × 3 seeds confirmation rows into
seed-mean ladders, runs `sav_verdict.savings_draws` + `family_tests` +
`sav_verdicts` (m=6 BH, precedence collapsed→consumed→conserved), per-cell
spreads vs δ (instability), n_half per checkpoint in BOTH interpolation
spaces, AUC secondaries at 16k (binding seed, labeled), and writes
results/sav/metrics.json — finalization COMMITTED here (no manual steps).

    python -m infra.aws_pg.jobs.sav_unblind --out results/sav \
        --controls <sav_controls.json> --cells-dir <dir with cell_*.json>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

from infra.pg.sav_verdict import LADDER          # single source (critic m4)

CKPTS = ("PG1", "pi0", "pi05", "random")
SEEDS = (1337, 1338, 1339)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import sav_verdict as SV
    from infra.pg import w16_probes as W
    from infra.pg import w17_verdict as V17

    C = json.loads(pathlib.Path(args.controls).read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    hashes = C["item_hashes"]
    conf = W.is_confirmation
    b_conf = W.mask_by(C["BASE"]["per_item"], hashes, conf)
    f_conf = W.mask_by(C["FLOOR"]["per_item"], hashes, conf)
    correct_conf = W.mask_by(C["correct_idx"], hashes, conf)
    delta = float(C["delta"]["level_and_pair"])

    rows = {str(p): json.loads(p.read_text())
            for p in sorted(pathlib.Path(args.cells_dir).rglob("cell_*.json"))}
    assert rows, "no cell rows"
    ids = [r["cell_id"] for r in rows.values()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate cell_ids across files (re-run hygiene): {sorted(dupes)}"

    def find(ck, n, s):
        cid = f"{ck}_cap_r512_n{n}_s{s}_confirmation"
        m = [r for r in rows.values() if r["cell_id"] == cid]
        assert len(m) == 1, f"{cid}: {len(m)} rows"
        return m[0]

    # recipe + correct_idx + per-checkpoint source constancy (critic m11/m2/m7)
    rec_ref = None
    src_by_ck = {}
    for r in rows.values():
        if "confirmation" not in r["cell_id"]:
            continue
        rr = r.get("train", {}).get("recipe")
        if rr:
            rk = (rr["lr"], rr.get("epochs"), rr["batch_size"], rr["warmup_ratio"])
            assert rec_ref in (None, rk), f"recipe drift across rows: {rec_ref} vs {rk}"
            rec_ref = rk
        ck = r["checkpoint"]
        src_by_ck.setdefault(ck, set()).add(str(r.get("decoder_source_sha")))
        if "correct_idx" in r:
            assert r["correct_idx"] == correct_conf, f"{r['cell_id']}: correct_idx != controls"
    for ck, shas in src_by_ck.items():
        assert len(shas) == 1, f"{ck}: decoder source drifted across cells: {shas}"

    ladders_seed = {}        # per-seed arrays (the bootstrap input — items AND seeds)
    ladders_mean = {}        # seed means (n_half + reporting)
    spreads = {}
    for ck in CKPTS:
        ladders_seed[ck] = {}
        ladders_mean[ck] = {}
        for n in LADDER:
            per_seed = [find(ck, n, s)["per_item"] for s in SEEDS]
            for a in per_seed:
                assert len(a) == len(b_conf), f"{ck}@{n} misaligned"
            ladders_seed[ck][n] = per_seed
            ladders_mean[ck][n] = np.asarray(per_seed, float).mean(axis=0).tolist()
            ss = V17.per_seed_s(per_seed, b_conf, f_conf)
            spreads[f"{ck}_{n}"] = {"per_seed_S": ss, "spread": max(ss) - min(ss)}

    sd = SV.savings_draws({c: ladders_seed[c] for c in ("PG1", "pi0", "pi05")},
                          ladders_seed["random"], b_conf, f_conf,
                          ns=LADDER, n_boot=args.n_boot, seed=1337)
    tests = SV.family_tests(sd)
    for t, v in tests.items():
        print(f"[SAV-UNB] {t}: {v['point']:.4f} CI[{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}] p={v['p']:.4f}")

    # instability: any cell entering a test with spread > δ voids that test
    unstable = set()
    test_cells = {"T1_sav_PG1": ["PG1"], "T2_sav_pi0": ["pi0"], "T3_sav_pi05": ["pi05"],
                  "T4_drop1": ["PG1", "pi0"], "T5_drop2": ["pi0", "pi05"],
                  "T6_droptotal": ["PG1", "pi05"]}
    for t, cks in test_cells.items():
        for ck in cks + ["random"]:
            if any(spreads[f"{ck}_{n}"]["spread"] > delta for n in LADDER):
                unstable.add(t)
    # δ on the savings-AUC scale for CI-inclusion legs (critic M1a: ‖w‖₂ scaling);
    # the per-cell spread guard above keeps the raw S-unit δ (unit-consistent there).
    delta_auc = delta * SV.auc_weight_l2(LADDER)
    verdicts = SV.sav_verdicts(tests, delta_level=delta_auc, delta_pair=delta_auc,
                               unstable_tests=sorted(unstable), q=args.q)

    # n_half per checkpoint, both spaces (item bootstrap, seed-mean ladders)
    def n_half(ck):
        import numpy as _np
        b = _np.asarray(b_conf, float); f = _np.asarray(f_conf, float)
        Ls = {n: _np.asarray(ladders_mean[ck][n], float) for n in LADDER}
        rng = _np.random.default_rng(1337)
        lin, lg = [], []
        n_valid = 0
        side = {"below": 0, "above": 0}
        for _ in range(args.n_boot):
            idx = rng.integers(0, b.size, b.size)
            den = b[idx].mean() - f[idx].mean()
            if den <= 0:
                continue                                   # den≤0 excluded from denominator (m3)
            n_valid += 1
            S = [(Ls[n][idx].mean() - f[idx].mean()) / den for n in LADDER]
            cross = None
            for i in range(len(LADDER) - 1):
                if S[i] < 0.5 <= S[i + 1]:
                    cross = i
                    break
            if cross is None:
                side["above" if S[0] >= 0.5 else "below"] += 1
                continue
            t = (0.5 - S[cross]) / (S[cross + 1] - S[cross])
            lin.append(LADDER[cross] + t * (LADDER[cross + 1] - LADDER[cross]))
            lg.append(2 ** (math.log2(LADDER[cross]) + t *
                            (math.log2(LADDER[cross + 1]) - math.log2(LADDER[cross]))))
        frac = (len(lin) / n_valid) if n_valid else 0.0
        def s(d):
            return (None if not d else
                    {"median": float(_np.median(d)),
                     "ci95": [float(_np.percentile(d, 2.5)), float(_np.percentile(d, 97.5))],
                     "note": "CI conditional on crossing" if frac < 1.0 else None})
        return {"linear_n": s(lin), "linear_log2n": s(lg),
                "frac_valid_draws_crossing": frac,
                "non_crossing_side": side, "n_valid_draws": n_valid}

    nhalf = {ck: n_half(ck) for ck in ("PG1", "pi0", "pi05")}

    # AUC secondaries at 16k, binding seed (labeled)
    sec = {}
    for ck in ("PG1", "pi0", "pi05"):
        a = find(ck, 16000, 1337)["choice_logits"]
        r = find("random", 16000, 1337)["choice_logits"]
        d = W.arm_a_paired_diff(a, r, correct_conf, n_boot=args.n_boot, seed=1337)
        sec[f"auc_{ck}_vs_random_16k"] = {"auc_diff": d["auc_diff_point"],
                                          "ci95": d["auc_diff_ci95"],
                                          "p_two_sided": d["auc_p_two_sided"]}
    sec_bh = W.bh_fdr({k: v["p_two_sided"] for k, v in sec.items()}, q=args.q)

    result = {"controls": args.controls, "n_cells": len(rows), "delta_S": delta,
              "delta_auc_ci_legs": delta_auc, "auc_weight_l2": SV.auc_weight_l2(LADDER),
              "savings_observed": sd["observed"],
              "primary_tests": tests, "unstable_tests": sorted(unstable),
              "spreads": spreads, "VERDICTS": verdicts, "n_half": nhalf,
              "secondary_auc": {"tests": sec, "bh": sec_bh,
                                "label": "SECONDARY (binding seed 1337; never verdict-bearing)"},
              "bootstrap": {"n_draws_kept": sd["n_draws_kept"],
                            "n_draws_dropped": sd["n_draws_dropped"]}}
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    print(f"[SAV-UNB] VERDICT: {verdicts['VERDICT']}")
    print(f"[SAV-UNB] savings: {sd['observed']}")


if __name__ == "__main__":
    main()
