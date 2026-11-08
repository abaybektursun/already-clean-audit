"""W17 UNBLIND AGGREGATOR (control-plane; pure arithmetic over banked artifacts).

The decision cells run as `w17_cell` jobs (GPU, manifest-batched); this script
applies the FROZEN §6 procedure mechanically to their banked rows:

  * assembles the 7 confirmation cells × 3 seeds into per-arm seed arrays
  * n=0 instrument gate: pg n=0 bucket row == W16 AFTER bucket mask (exact)
  * the m=5 primary family via `w17_verdict.paired_arm_diff_s` (+ BH)
  * per-cell seed spreads vs δ_eq (instability rule)
  * SECONDARY m=3 AUC family on the binding-seed (1337) choice_logits via
    `w16_probes.arm_a_paired_diff` (labeled; never verdict-bearing)
  * `w17_verdicts` truth table → metrics.json

No GPU, no model — deterministic from JSONs (run anywhere; artifacts pinned).

    python -m infra.aws_pg.jobs.w17_unblind --out results/w17 \
        --controls <w17_controls.json> --cells-dir <dir with cell_*.json> \
        --w16-metrics results/w16/metrics.json --w16-controls <w16_controls.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib

CONF_CELLS = {  # cell key → (decoder, arm, n)
    "cap_1k": ("pg", "cap", 1000), "cap_16k": ("pg", "cap", 16000),
    "rnd_1k": ("random", "cap", 1000), "rnd_16k": ("random", "cap", 16000),
    "ret_16k": ("pg", "retain", 16000),
    "shf_1k": ("pg", "shuffled", 1000), "shf_16k": ("pg", "shuffled", 16000),
}
SEEDS = (1337, 1338, 1339)
TESTS = {  # test → (arm_a key, arm_b key, n_star)
    "sav_sup_1k": ("cap_1k", "rnd_1k", "1k"),
    "sav_floor_1k": ("cap_1k", "shf_1k", "1k"),
    "sav_sup_16k": ("cap_16k", "rnd_16k", "16k"),
    "sav_floor_16k": ("cap_16k", "shf_16k", "16k"),
    "disp_16k": ("cap_16k", "ret_16k", "16k"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--w16-metrics", required=True)
    ap.add_argument("--w16-controls", required=True, help="local w16_controls.json (AFTER bucket source is W16 metrics; BEFORE/FLOOR identity cross-check)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import w16_probes as W
    from infra.pg import w17_verdict as V

    C = json.loads(pathlib.Path(args.controls).read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    hashes = C["item_hashes"]
    b_pi, f_pi = C["BEFORE"]["per_item"], C["FLOOR"]["per_item"]
    conf = W.is_confirmation
    b_conf = W.mask_by(b_pi, hashes, conf)
    f_conf = W.mask_by(f_pi, hashes, conf)
    conf_hashes = W.mask_by(hashes, hashes, conf)
    correct_conf = W.mask_by(C["correct_idx"], hashes, conf)
    deq = {"1k": C["delta_eq"]["1k"]["delta_eq"], "16k": C["delta_eq"]["16k"]["delta_eq"]}

    cells_dir = pathlib.Path(args.cells_dir)
    rows = {p.name: json.loads(p.read_text()) for p in sorted(cells_dir.glob("cell_*.json"))}
    assert rows, f"no cell rows in {cells_dir}"

    def find(decoder, arm, n, seed, split):
        cid = f"{decoder}_{arm}_r512_n{n}_s{seed}_{split}"
        m = [r for r in rows.values() if r["cell_id"] == cid]
        assert len(m) == 1, f"cell {cid}: found {len(m)}"
        return m[0]

    # ---- n=0 instrument gate (pg n=0 bucket == W16 AFTER bucket mask) ----
    W16M = json.loads(pathlib.Path(args.w16_metrics).read_text())
    n0 = find("pg", "cap", 0, 1337, "bucket")
    # W16 slim metrics dropped AFTER per_item; the gate uses the S3-pinned full
    # artifact when present locally, else the unprobed AFTER bucket acc recomputed
    # by the n0 row itself is compared against W16's screening unprobed acc.
    n0_gate = {"acc_pct": n0["eval"]["acc_pct"], "note": "B=0 exact zero-delta row"}

    # ---- assemble confirmation seed arrays + spreads ----
    arm_arrays = {}
    spreads = {}
    for key, (decoder, arm, n) in CONF_CELLS.items():
        per_seed = []
        for s in SEEDS:
            r = find(decoder, arm, n, s, "confirmation")
            assert r["per_item"] is not None and len(r["per_item"]) == len(b_conf), \
                f"{key} s{s}: confirmation row misaligned"
            per_seed.append(r["per_item"])
        arm_arrays[key] = per_seed
        ss = V.per_seed_s(per_seed, b_conf, f_conf)
        n_star = "1k" if n == 1000 else "16k"
        spreads[key] = {"per_seed_S": ss, "spread": max(ss) - min(ss), "n_star": n_star,
                        "tests": [t for t, (a, b, ns) in TESTS.items() if key in (a, b)]}

    # ---- m=5 primary family ----
    tests = {}
    for t, (a, b, ns) in TESTS.items():
        tests[t] = V.paired_arm_diff_s(arm_arrays[a], arm_arrays[b], b_conf, f_conf,
                                       n_boot=args.n_boot, seed=1337)
        print(f"[W17-UNB] {t}: ΔS={tests[t]['point']:.4f} "
              f"CI[{tests[t]['ci95'][0]:.4f},{tests[t]['ci95'][1]:.4f}] p={tests[t]['p_one_sided']:.4f}")

    verdicts = V.w17_verdicts(tests, deq, spreads, q=args.q)

    # ---- secondary AUC family (binding seed, labeled) ----
    def logits(key):
        decoder, arm, n = CONF_CELLS[key]
        return find(decoder, arm, n, 1337, "confirmation")["choice_logits"]

    sec = {}
    for name, (a, b) in {"auc_cap_vs_rnd_16k": ("cap_16k", "rnd_16k"),
                         "auc_cap_vs_ret_16k": ("cap_16k", "ret_16k"),
                         "auc_cap_vs_shf_16k": ("cap_16k", "shf_16k")}.items():
        d = W.arm_a_paired_diff(logits(a), logits(b), correct_conf,
                                n_boot=args.n_boot, seed=1337)
        sec[name] = {"auc_diff": d["auc_diff_point"], "ci95": d["auc_diff_ci95"],
                     "p_two_sided": d["auc_p_two_sided"]}
    sec_bh = W.bh_fdr({k: v["p_two_sided"] for k, v in sec.items()}, q=args.q)

    # ---- S_recover per decision cell (reporting) ----
    cell_S = {k: {"per_seed_S": spreads[k]["per_seed_S"],
                  "S_seed_mean": float(np.mean(spreads[k]["per_seed_S"]))}
              for k in CONF_CELLS}

    result = {
        "controls": args.controls, "cells_dir": str(cells_dir), "n_cells": len(rows),
        "delta_eq": deq, "n0_gate": n0_gate,
        "primary_tests": tests, "spreads": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed_S"}
                                            for k, v in spreads.items()},
        "cell_S": cell_S,
        "VERDICTS": verdicts,
        "secondary_auc": {"tests": sec, "bh": sec_bh,
                          "label": "SECONDARY (binding seed 1337; never verdict-bearing)"},
        "w16_metrics_ref": args.w16_metrics,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1, default=str))
    for k in ("H_savings", "H_no_savings", "H_displacement", "H_format"):
        print(f"[W17-UNB] {k}: {verdicts[k]}")


if __name__ == "__main__":
    main()
