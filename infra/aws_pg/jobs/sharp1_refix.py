"""SHARP-1 CORRECTED unblind — rank ladder at α=r (SCALE 1, the frozen recipe).

Amendment 2026-06-13 (twice-corrected; see prereg). Reads {r64, r128, r256} cap@16k
confirmation cells (each α==rank ⇒ scale α/r == 1, 3 seeds) + the W17 sealed r512
anchor (pointer 0.6455, also scale 1). Per-rank P(S≥0.5) via items+seeds bootstrap;
m=3 BH q=0.05; bracket = (highest dead rank, lowest recovering rank]. The scale-1 guard
(α==rank) prevents repeating the α=2r / fixed-α=512 defect.

    python -m infra.aws_pg.jobs.sharp1_refix --out results/sharpfix \
        --w17-controls /tmp/w17_controls_a3.json --cells-dir /tmp/sharp1fix_cells
"""
from __future__ import annotations

import argparse
import json
import pathlib

SEEDS = (1337, 1338, 1339)
RANKS = (64, 128, 256)
R512_ANCHOR = 0.6455  # W17 sealed confirmation, α=r=512 ⇒ scale 1 (pointer; not re-read)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--w17-controls", required=True)
    ap.add_argument("--cells-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import numpy as np

    from infra.pg import sharpen as SH
    from infra.pg import w16_probes as W
    from infra.pg.w16_probes import bh_fdr

    C = json.loads(pathlib.Path(args.w17_controls).read_text())
    assert C["CONTROLS_OK"] is True and C["recipe"]["FROZEN"] is True
    h = C["item_hashes"]
    conf = W.mask_by(h, h, W.is_confirmation)
    b = np.asarray(W.mask_by(C["BEFORE"]["per_item"], h, W.is_confirmation), float)
    f = np.asarray(W.mask_by(C["FLOOR"]["per_item"], h, W.is_confirmation), float)

    rows = {p.name: json.loads(p.read_text())
            for p in sorted(pathlib.Path(args.cells_dir).rglob("cell_*.json"))}

    def arrays(rank):
        arrs, ces = [], []
        for s in SEEDS:
            m = [r for r in rows.values()
                 if r["rank"] == rank and r["seed"] == s and r["n"] == 16000
                 and r["arm"] == "cap" and r["decoder"] == "pg"
                 and r["eval_split"] == "confirmation"]
            assert len(m) == 1, f"r{rank} s{s}: {len(m)} rows"
            assert len(m[0]["per_item"]) == len(conf), f"r{rank} s{s}: length drift — HALT"
            al = float(m[0]["adapter"]["alpha"])
            assert al == float(rank), \
                f"r{rank} s{s}: alpha={al} ≠ rank (scale {al/rank:.2f} ≠ 1) — DEFECT GUARD HALT"
            arrs.append(m[0]["per_item"])
            ces.append(m[0]["train"]["mean_loss_last_epoch"])
        return arrs, ces

    tests = {}
    for rank in RANKS:
        arrs, ces = arrays(rank)
        t = SH.s_vs_threshold(arrs, b, f, thr=0.5)
        t["fit_ce_last_mean"] = float(np.mean(ces))
        t["alpha"] = float(rank)
        t["scale"] = 1.0
        tests[f"r{rank}"] = t
        print(f"[SHARP1-FIX] r{rank} (α={rank}, scale 1): S={t['point_S']:.4f} "
              f"CI[{t['ci95'][0]:.4f},{t['ci95'][1]:.4f}] p={t['p_one_sided']:.4f} "
              f"CE={t['fit_ce_last_mean']:.3f}")
    print(f"[SHARP1-FIX] r512 anchor (α=512, scale 1, W17 sealed pointer): S={R512_ANCHOR}")

    bh = bh_fdr({k: v["p_one_sided"] for k, v in tests.items()}, q=0.05)
    sig = {r: bh[f"r{r}"]["significant"] for r in RANKS}
    recovering = [r for r in RANKS if sig[r]] + [512]      # r512 recovers by construction
    dead = [r for r in RANKS if not sig[r]]
    lo_rec = min(recovering)
    hi_dead = max(dead) if dead else None
    if hi_dead is not None and hi_dead > lo_rec:
        bracket = (f"NON-MONOTONIC HALT — dead rank {hi_dead} above recovering rank "
                   f"{lo_rec}; harness-indicted")
    elif hi_dead is None:
        bracket = f"gate ≤ {lo_rec} (all tested ranks recover at scale 1; probe r<64 next)"
    else:
        bracket = f"gate ∈ ({hi_dead}, {lo_rec}] (α=r, scale 1, confirmation half)"
    print(f"[SHARP1-FIX] BRACKET: {bracket}")

    res = {"convention": "alpha=r (scale alpha/r == 1) — the frozen W17 recipe",
           "tests": tests, "bh": bh, "sig": sig, "r512_anchor_S": R512_ANCHOR,
           "bracket": bracket,
           "supersedes": "sharpen_metrics.json sharp1 (α=2r/scale-2 defect) AND the "
                         "killed fixed-α=512/scale-{8,4,2} re-run"}
    (out / "sharp1fix_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print("[SHARP1-FIX] done")


if __name__ == "__main__":
    main()
