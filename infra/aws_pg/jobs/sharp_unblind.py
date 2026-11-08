"""SHARP unblind — ONE read for both sharpeners (control-plane; frozen rules).

SHARP-1: r∈{128,256} cap@16k conf cells (w17_cell rows) vs W17 controls BEFORE/FLOOR
         → m=2 BH → mechanical gate bracket.
SHARP-2: {pi0, random}@64k conf cells (sav_cell rows) + sealed pi0 16k arrays vs SAV
         controls BASE/FLOOR → m=2 BH → mechanical π0-asymptote verdict.

    python -m infra.aws_pg.jobs.sharp_unblind --out results/sharpen \
        --w17-controls /tmp/w17_controls_a3.json --sav-controls /tmp/sav_controls_a2.json \
        --sharp1-dir /tmp/sharp1_cells --sharp2-dir /tmp/sharp2_cells \
        --sav-sealed-dir /tmp/sav_cells
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEEDS = (1337, 1338, 1339)


def _cells(d, pat):
    return {p.name: json.loads(p.read_text()) for p in sorted(pathlib.Path(d).rglob(pat))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w17-controls", required=True)
    ap.add_argument("--sav-controls", required=True)
    ap.add_argument("--sharp1-dir", required=True)
    ap.add_argument("--sharp2-dir", required=True)
    ap.add_argument("--sav-sealed-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import sharpen as SH
    from infra.pg import w16_probes as W
    from infra.pg.w16_probes import bh_fdr

    # ---------------- SHARP-1 ----------------
    C1 = json.loads(pathlib.Path(args.w17_controls).read_text())
    assert C1["CONTROLS_OK"] is True and C1["recipe"]["FROZEN"] is True
    h1 = C1["item_hashes"]
    conf1 = W.mask_by(h1, h1, W.is_confirmation)
    b1 = np.asarray(W.mask_by(C1["BEFORE"]["per_item"], h1, W.is_confirmation), float)
    f1 = np.asarray(W.mask_by(C1["FLOOR"]["per_item"], h1, W.is_confirmation), float)
    rows1 = _cells(args.sharp1_dir, "cell_*.json")
    assert len(rows1) == 6, f"SHARP-1 expects 6 cells, found {len(rows1)}"

    def w17_seed_arrays(rank):
        arrs, ces = [], []
        for s in SEEDS:
            m = [r for r in rows1.values()
                 if r["rank"] == rank and r["seed"] == s and r["n"] == 16000
                 and r["arm"] == "cap" and r["decoder"] == "pg"
                 and r["eval_split"] == "confirmation"]
            assert len(m) == 1, f"r{rank} s{s}: {len(m)} rows"
            # identity vs controls asserted IN-JOB (w17_cell line 217; rows don't
            # persist the hash list) — length-check here as the cheap tripwire
            assert len(m[0]["per_item"]) == len(conf1), f"r{rank} s{s}: length drift — HALT"
            arrs.append(m[0]["per_item"])
            ces.append(m[0]["train"]["mean_loss_last_epoch"])
        return arrs, ces

    s1 = {}
    for rank in (128, 256):
        arrs, ces = w17_seed_arrays(rank)
        t = SH.s_vs_threshold(arrs, b1, f1, thr=0.5)
        t["fit_ce_last_mean"] = float(np.mean(ces))
        t["fit_ok_le_0.8"] = bool(np.mean(ces) <= 0.8)
        s1[f"T_r{rank}"] = t
        print(f"[SHARP1] r{rank}: S={t['point_S']:.4f} CI[{t['ci95'][0]:.4f},"
              f"{t['ci95'][1]:.4f}] p={t['p_one_sided']:.4f} CE={t['fit_ce_last_mean']:.3f}")
    bh1 = bh_fdr({k: v["p_one_sided"] for k, v in s1.items()}, q=0.05)
    bracket = SH.gate_bracket(bh1["T_r128"]["significant"], bh1["T_r256"]["significant"])
    print(f"[SHARP1] BRACKET: {bracket}")

    # ---------------- SHARP-2 ----------------
    C2 = json.loads(pathlib.Path(args.sav_controls).read_text())
    assert C2["CONTROLS_OK"] is True and C2["recipe"]["FROZEN"] is True
    h2 = C2["item_hashes"]
    conf2 = W.mask_by(h2, h2, W.is_confirmation)
    b2 = np.asarray(W.mask_by(C2["BASE"]["per_item"], h2, W.is_confirmation), float)
    f2 = np.asarray(W.mask_by(C2["FLOOR"]["per_item"], h2, W.is_confirmation), float)
    rows2 = _cells(args.sharp2_dir, "cell_*.json")
    assert len(rows2) == 6, f"SHARP-2 expects 6 cells, found {len(rows2)}"

    def sav_seed_arrays(d, ck, n):
        arrs = []
        for s in SEEDS:
            cid = f"{ck}_cap_r512_n{n}_s{s}_confirmation"
            m = [r for r in d.values() if r["cell_id"] == cid]
            assert len(m) == 1, f"{cid}: {len(m)} rows"
            assert m[0]["per_item_hash"] == conf2, f"{cid}: identity drift — HALT"
            if n == 64000:
                assert m[0]["train"]["n_examples"] == 64000, f"{cid}: pool shortfall"
            arrs.append(m[0]["per_item"])
        return arrs

    pi64 = sav_seed_arrays(rows2, "pi0", 64000)
    rnd64 = sav_seed_arrays(rows2, "random", 64000)
    sealed = _cells(args.sav_sealed_dir, "cell_pi0_cap_r512_n16000_*_confirmation.json")
    pi16 = sav_seed_arrays(sealed, "pi0", 16000)

    t_sav = SH.paired_diff_s(pi64, rnd64, b2, f2)
    t_trend = SH.paired_diff_s(pi64, pi16, b2, f2)
    s64 = SH.s_vs_threshold(pi64, b2, f2, thr=0.5)   # point + CI texture (thr unused in verdict)
    print(f"[SHARP2] sav64: Δ={t_sav['point_delta']:+.4f} CI[{t_sav['ci95'][0]:+.4f},"
          f"{t_sav['ci95'][1]:+.4f}] p={t_sav['p_one_sided']:.4f}")
    print(f"[SHARP2] trend: Δ={t_trend['point_delta']:+.4f} CI[{t_trend['ci95'][0]:+.4f},"
          f"{t_trend['ci95'][1]:+.4f}] p={t_trend['p_one_sided']:.4f}")
    print(f"[SHARP2] S_pi0(64k)={s64['point_S']:.4f} per-seed {['%.3f' % v for v in s64['per_seed_S']]}")
    bh2 = bh_fdr({"T_sav64": t_sav["p_one_sided"], "T_trend": t_trend["p_one_sided"]}, q=0.05)
    verdict2 = SH.pi0_verdict(bh2["T_sav64"]["significant"], bh2["T_trend"]["significant"],
                              s64["point_S"])
    print(f"[SHARP2] VERDICT: {verdict2}")

    res = {"sharp1": {"tests": s1, "bh": bh1, "bracket": bracket,
                      "anchors": {"r512_conf_S": "0.6455 (W17 sealed, pointer)",
                                  "r64": "dead (W17 DISCOVERY bucket, pointer)"}},
           "sharp2": {"T_sav64": t_sav, "T_trend": t_trend, "S_pi0_64k": s64,
                      "bh": bh2, "VERDICT": verdict2},
           "inputs": vars(args)}
    (out / "sharpen_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print("[SHARP] done")


if __name__ == "__main__":
    main()
