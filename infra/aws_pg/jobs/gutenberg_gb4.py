"""GUTENBERG GB-3 verdict + GB-4 two-factor surface (14B analysis, CPU, banked artifacts only).

Spec §3 GB-3/GB-4 + AM-0.5.5/9. Consumes ONE coherent run (the GB-3 vast rental: fresh
seed-slot draws, extended shares, banked vectors, qk rider) plus GB-2's gain table (computed
with GB-3's banked vectors as --extra-banks-s3), so costs, energies and gains all describe the
SAME bank objects on the same substrate — no cross-run mixing.

  GB-3 verdict: AM1B-CONFIRMED iff the fresh-draw S/C ratio's 95% paired-bootstrap CI
  excludes 1 (site q, AM-1's instrument); the qk rider rows are the exploratory FUNC-instrument
  bridge. GB-4: LAW/CURVE per gutenberg_core.two_factor_analysis (held-out {anti32,
  topenergy128}, frozen pre-look).

    pg run infra.aws_pg.jobs.gutenberg_gb4 --no-weights --type r7i.2xlarge,m7i.2xlarge \
        --args "--gb3-prefix s3://.../shackleton_gb3_am1b_harden/cells/gb3_full \
                --gb2-uri s3://.../results/gutenberg_gb2b/out/gutenberg_gb2_reader_gain.json" \
        --s3-prefix results/gutenberg_gb4 --budget-min 40
"""
import argparse
import json
import pathlib
import subprocess

import numpy as np

from infra.pg import copernicus_core as C
from infra.pg import gutenberg_core as G

HELDOUT = ["anti32", "topenergy128"]        # AM-0.5.9, frozen before any fit
SEED = 20260819


def _fetch(s3_uri, dst):
    subprocess.run(["aws", "s3", "cp", s3_uri, str(dst), "--region", "us-east-1",
                    "--only-show-errors"], check=True)
    return dst


def _gain_key(arm):
    """gb2 q_proj row key for an ablation arm name."""
    if arm.startswith("haar32_s"):
        return "haar_ablation_s" + arm.rsplit("_s", 1)[1]
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gb3-prefix", required=True, help=".../cells/<gb3_full> S3 prefix")
    ap.add_argument("--gb2-uri", required=True, help="gutenberg_gb2_reader_gain.json S3 uri")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wk = pathlib.Path("/tmp/gutenberg_gb4")
    wk.mkdir(parents=True, exist_ok=True)

    wn = json.loads(_fetch(f"{a.gb3_prefix}/v1abl_window_nll.json",
                           wk / "wnll.json").read_text())["per_window"]
    wn_qk = json.loads(_fetch(f"{a.gb3_prefix}/v1abl_window_nll_qk.json",
                              wk / "wnll_qk.json").read_text())["per_window"]
    shares = json.loads(_fetch(f"{a.gb3_prefix}/v1abl_arm_shares.json",
                               wk / "shares.json").read_text())
    rows = json.loads(_fetch(f"{a.gb3_prefix}/v1abl_rows.json",
                             wk / "rows.json").read_text())
    gb2 = json.loads(_fetch(a.gb2_uri, wk / "gb2.json").read_text())

    # ---- GB-3 verdict: assembled by the TESTED pure function, not inline here ----
    gb3 = G.build_gb3_metrics(wn, wn_qk, rows, seed=SEED)
    (out / "gutenberg_gb3_verdict.json").write_text(
        json.dumps(gb3, indent=2, sort_keys=True, default=float))
    sc = gb3["S_over_C"]
    print(f"[gb4] GB-3 verdict={gb3['verdict']} S/C={sc[0]:.3f} [{sc[1]:.3f},{sc[2]:.3f}] "
          f"E/C={gb3['E_over_C'][0]:.3f}", flush=True)

    # ---- GB-4 surface: (cost, E, G) triples from one run ----
    per_share = shares["per_layer_share"]
    gain_rows = gb2["qwen14b_rows"]
    none = wn["none"]
    ntok = np.asarray(none["ntok"], dtype=np.float64)

    def dn(arm, table):
        return G.dnll_per_window(table[arm]["nll"], none["nll"], ntok)

    def med_gain(arm):
        key = _gain_key(arm)
        return float(np.median([r["q_proj"][key] for r in gain_rows]))

    arms = {}
    skipped = []
    for arm in wn:
        if arm == "none":
            continue
        if arm not in per_share:
            skipped.append(arm)
            continue
        arms[arm] = {"dnll_w": dn(arm, wn),
                     "E": float(np.median(per_share[arm])),
                     "G": med_gain(arm)}
    for h in HELDOUT:
        assert h in arms, f"held-out {h} lacks a (cost,E,G) triple"
    surface = G.two_factor_analysis(arms, HELDOUT, n_boot=1000, seed=SEED + 8)
    surface["skipped_no_energy"] = skipped
    surface["points"] = {k: {"cost_mean_dnll": float(np.mean(v["dnll_w"])),
                             "E": v["E"], "G": v["G"]} for k, v in arms.items()}
    surface["provenance"] = {"gb3_prefix": a.gb3_prefix, "gb2_uri": a.gb2_uri,
                             "code": G.provenance(), "utc": C.utc_now()}
    (out / "gutenberg_gb4_surface.json").write_text(
        json.dumps(surface, indent=2, sort_keys=True, default=float))
    print(f"GUTENBERG_GB4_OK gb3={gb3['verdict']} gb4={surface['verdict']} "
          f"n_fit={surface['n_fit_banks']}", flush=True)


if __name__ == "__main__":
    main()
