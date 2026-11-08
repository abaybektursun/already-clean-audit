"""GUTENBERG GB-5 [rider, free] — the RMSNorm account of reader gain (Chen et al. Thm 2).

Spec §3 GB-5: does per-coordinate reader gain track the ||gamma||/rms(x) bound? EXPLORATORY —
consistent-with support for §5's mechanism sentence; no frozen claim rides on it (registered
director prediction: tracks in RANK, not magnitude, p=0.6).

Per layer on gemma-2b (the paper's §4-5 substrate), three per-coordinate (d=2048) vectors:
  g_j   = ||W_qproj[:, j]||^2                     per-coordinate reader gain (range-read weights)
  ln_j  = (1 + gamma_j)^2                          effective RMSNorm gain^2 (GEMMA CONVENTION:
                                                   the trap — gemma's LN weight is (1+w))
  e_j   ∝ E[h_j^2] = sum_sq[j]                     carried energy at q_proj_in (banked actstats;
                                                   un-normalised — every statistic here is a
                                                   RANK correlation, so 1/n_tok cancels)
Banked: per-layer Spearman rho(g, ln), rho(g, e), rho(e, ln), partial-rank rho(g, e | ln),
plus the same at the core32 leverage coordinates. CPU, banked artifacts + range-read only.

FREE RIDER (same artifacts, no extra cost) — the §C "not the BOS/attention-sink specifically"
bullet currently carries NO number. The banked actstats holds pos0_sq (per-coordinate energy at
position 0 = the BOS/sink position) beside the all-position sum_sq, so the sink-coordinate
question is answerable here: rho(core32 leverage, pos0 energy) and the top-32 coordinate overlap
between core32's leverage set and the position-0 spike set, against the chance level 32/2048.

    pg run infra.aws_pg.jobs.gutenberg_gb5 --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --s3-prefix results/gutenberg_gb5 --budget-min 40
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import gutenberg_core as G
from infra.pg.humboldt_core import hf_setup

ACTSTATS = ("s3://pg-lineage-769063704696/results/shackleton_r_v1_anchor_gemma2b/"
            "cells/anchor_gemma2b_actstats/actstats.npz")
CENSUS_G2B = "s3://pg-lineage-769063704696/results/shackleton_census_g2b/out/banks/"
N_LAYERS = 18


def _fetch(s3_uri, dst, recursive=False):
    cmd = ["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1", "--only-show-errors"]
    if recursive:
        cmd.insert(3, "--recursive")
    subprocess.run(cmd, check=True)
    return dst


def _rank(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r


def spearman(a, b):
    ra, rb = _rank(np.asarray(a)), _rank(np.asarray(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def partial_spearman(a, b, c):
    """rank rho(a, b | c) via residualized ranks."""
    ra, rb, rc = (_rank(np.asarray(v)) for v in (a, b, c))
    def resid(y, x):
        x = np.stack([np.ones_like(x), x], axis=1)
        return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    return spearman(resid(ra, rc), resid(rb, rc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gb5")
    wk.mkdir(parents=True, exist_ok=True)

    hf_setup()
    from infra.aws_pg.jobs.gutenberg_gb2 import HFRangeReader

    stats = np.load(_fetch(ACTSTATS, wk / "actstats.npz"))
    _fetch(CENSUS_G2B, wk / "banks", recursive=True)
    sq_keys = [k for k in stats.files if "sum_sq" in k and "q_proj" in k]
    assert sq_keys, f"no q_proj sum_sq keys in actstats ({stats.files[:8]}...)"

    def sum_sq(L):
        return np.asarray(stats[f"L{L:02d}_q_proj_in_sum_sq"], dtype=np.float64)

    def pos0_sq(L):
        return np.asarray(stats[f"L{L:02d}_q_proj_in_pos0_sq"], dtype=np.float64)

    reader = HFRangeReader("google/gemma-2b")
    rows = []
    for L in range(N_LAYERS):
        W = reader.tensor(f"model.layers.{L}.self_attn.q_proj.weight")
        gamma = reader.tensor(f"model.layers.{L}.input_layernorm.weight")
        g = (W ** 2).sum(axis=0)                         # (2048,) per-coordinate reader gain
        ln = (1.0 + gamma) ** 2                          # gemma (1+w) convention
        e = sum_sq(L)
        assert g.shape == ln.shape == e.shape == (2048,), (g.shape, ln.shape, e.shape)
        core = np.linalg.qr(np.asarray(
            np.load(wk / "banks" / f"q_proj__L{L}__core32.npy"), dtype=np.float64))[0]
        lev = (core ** 2).sum(axis=1)                    # core32 leverage per coordinate
        p0 = pos0_sq(L)
        top_lev = set(np.argsort(lev)[::-1][:32].tolist())
        top_p0 = set(np.argsort(p0)[::-1][:32].tolist())
        row = {"layer": L,
               "rho_gain_lngain": spearman(g, ln),
               "rho_gain_energy": spearman(g, e),
               "rho_energy_lngain": spearman(e, ln),
               "rho_gain_energy_given_lngain": partial_spearman(g, e, ln),
               "rho_corelev_lngain": spearman(lev, ln),
               "rho_corelev_energy": spearman(lev, e),
               "rho_corelev_gain": spearman(lev, g),
               # free rider: the BOS/sink-coordinate question (§C)
               "rho_corelev_pos0": spearman(lev, p0),
               "n_top32_corelev_in_top32_pos0": len(top_lev & top_p0),
               "pos0_share_of_total_energy": float(p0.sum() / e.sum())}
        rows.append(row)
        print(f"[gb5] L{L}: rho(g,ln)={row['rho_gain_lngain']:.3f} "
              f"rho(g,e)={row['rho_gain_energy']:.3f} "
              f"rho(lev,e)={row['rho_corelev_energy']:.3f} "
              f"lev∩pos0={row['n_top32_corelev_in_top32_pos0']}/32", flush=True)

    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k != "layer"}
    (out / "gutenberg_gb5_rmsnorm.json").write_text(json.dumps({
        "object": "gutenberg_gb5_rmsnorm_account_v1",
        "register": "EXPLORATORY rider — no frozen claim (spec GB-5)",
        "estimand": "per-coordinate Spearman correlations between reader gain "
                    "||W_q[:,j]||^2, effective RMSNorm gain^2 (1+gamma_j)^2 (gemma "
                    "convention), and carried energy E[h_j^2] at q_proj_in; gemma-2b, 18 layers",
        "per_layer": rows, "median": med,
        "chance_top32_overlap_of_2048": 32 * 32 / 2048,
        "actstats": ACTSTATS, "banks": CENSUS_G2B,
        "code": G.provenance(),
    }, indent=2, sort_keys=True))
    print(f"GUTENBERG_GB5_OK median rho(g,ln)={med['rho_gain_lngain']:.3f} "
          f"rho(g,e)={med['rho_gain_energy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
