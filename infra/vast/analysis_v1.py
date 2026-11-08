#!/usr/bin/env python3
"""V1 offline analysis — share curves + twin bands from banked sufficient stats.

Frozen BEFORE any rung's data is read (blind-analysis discipline): this file
defines every statistic and control; thresholds are then recorded into
shackleton_expectations.jsonl from the twin bands BEFORE the contrast row is
printed (§6). Runs on the local CPU against a downloaded actstats.npz.

Per (layer, site), from sum_sq[d] / pos_energy[w] / pos0_sq[d]:
  top-k share        k in K_GRID, coordinates ranked by sum_sq (the treatment)
  random-k twin band N_DRAWS random coordinate sets of the same k (chance floor)
  bottom-k share     the anti-address (specificity control)
  complement share   1 - top-k (mass conservation check)
  pos0 share         pos_energy[0]/Σ (the BOS spike, cell (c))
  spikiness          participation ratio (Σs)²/(d·Σs²), and kurtosis proxy
                     from sum_quad — the per-coordinate heavy-tail readout

Usage: python3 analysis_v1.py <actstats.npz> <flags.json> --out atlas_v1.jsonl
"""
import argparse
import json

import numpy as np

K_GRID = (8, 32, 128, 512)
N_DRAWS = 1000


def shares(sum_sq, rng):
    d = sum_sq.size
    total = sum_sq.sum()
    order = np.argsort(sum_sq)[::-1]
    rows = {}
    for k in K_GRID:
        if k >= d:
            continue
        top = sum_sq[order[:k]].sum() / total
        bottom = sum_sq[order[-k:]].sum() / total
        draws = np.array([sum_sq[rng.choice(d, k, replace=False)].sum() / total
                          for _ in range(N_DRAWS)])
        rows[k] = {"top_share": float(top), "bottom_share": float(bottom),
                   "complement_share": float(1.0 - top),
                   "random_mean": float(draws.mean()),
                   "random_p99": float(np.quantile(draws, 0.99)),
                   "random_p01": float(np.quantile(draws, 0.01)),
                   "excess_over_random_p99": float(top / np.quantile(draws, 0.99))}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("flags")
    p.add_argument("--out", default="atlas_v1.jsonl")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    z = np.load(a.npz)
    flags = json.load(open(a.flags))
    rng = np.random.default_rng(a.seed)
    out = open(a.out, "w")
    for row in flags["per_site"]:
        key = f"L{row['layer']:02d}_{row['site']}"
        sum_sq = z[f"{key}_sum_sq"]
        sum_quad = z[f"{key}_sum_quad"]
        pos_energy = z[f"{key}_pos_energy"]
        pos0_sq = z[f"{key}_pos0_sq"]
        n_tok = row["tokens"]
        d = sum_sq.size

        mean_sq = sum_sq / n_tok
        # per-coordinate kurtosis proxy: E[h^4] / E[h^2]^2 of the top coordinate
        with np.errstate(divide="ignore", invalid="ignore"):
            kurt = np.where(mean_sq > 0, (sum_quad / n_tok) / mean_sq**2, 0.0)
        top_coord = int(np.argmax(sum_sq))
        pr = float(sum_sq.sum()**2 / (d * (sum_sq**2).sum()))

        rec = {"arm": "V1", "rung": flags["model_dir"].split("/")[-1],
               "layer": row["layer"], "site": row["site"], "d": d,
               "tokens": n_tok, "corpus_sha256": flags["corpus_sha256"],
               "share_curves": shares(sum_sq, rng),
               "pos0_share": float(pos_energy[0] / pos_energy.sum()),
               "pos_share_tail_mean": float(pos_energy[1:].mean()
                                            / pos_energy.sum()),
               "pos0_top_coord_share": float(pos0_sq.max() / pos0_sq.sum()),
               "participation_ratio": pr,
               "top_coord": top_coord, "top_coord_kurtosis": float(kurt[top_coord]),
               "median_kurtosis": float(np.median(kurt[mean_sq > 0]))}
        out.write(json.dumps(rec) + "\n")
    out.close()
    print(f"[analysis_v1] wrote {a.out}")


if __name__ == "__main__":
    main()
