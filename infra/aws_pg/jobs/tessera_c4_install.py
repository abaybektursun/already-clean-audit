"""TESSERA-FIX — census the INSTALL write's energy concentration, to test the C4-confound prediction.

THE PREDICTION UNDER TEST, recorded before this job ran (`2026-08-02_tessera_result.md` Result 4,
journal 2026-08-03):

    if write FLATNESS drives C4's below-reference pattern, then `install` — the one hop never
    censused — must be the LEAST flat (MOST concentrated) of the three hops, because the pairs
    containing it are the minima of both classes: install~pi0 = 0.500 is the lowest hop~hop
    fraction and install~ft24 = 0.139 is the lowest hop~ft fraction.

Measured concentrations so far (`tessera_c4_confound`, energy ÷ analytic chance floor at r=20):

    FT donors, median (n=18)   2.372   [1.741, 2.680]
    HOP π0                     1.807
    HOP π0.5                   1.553      ← flattest, and in the most-depressed pairs

WHAT EACH OUTCOME MEANS. This is a real fork and the confound loses on one branch:

  install > π0 > π0.5            CONFIRMED — concentration order reproduces the depression order
                                 across all six pairs; the confound explains C4 completely.
  install between π0 and π0.5    PARTIAL — order broken but the family separation stands.
  install < π0.5 (flattest)      REFUTED — the flattest hop would sit in the LEAST depressed pairs,
                                 which is the opposite of what flatness predicts. The confound would
                                 then not explain C4's ordering and the avoidance reading would be
                                 back on the table.

INSTRUMENT. `cleave_core.mechanism_census` at ranks {20, 50, 350, 679} — the identical function,
ranks and floors QUORUM used on the FT donors and TESSERA's K4 used on π0/π0.5. The install write is
`PG1 − gemma-2b` over the 126 LM decoder matrices (the creation direction, matching Arm S's
convention and amendment A10). The bank is one of K1's banked faculty banks, and **the bank is
recorded**: K4 used `caption`, so this job defaults to it — a different bank would not be comparable
to the numbers this test is scored against.

  ./infra/aws_pg/pg run infra.aws_pg.jobs.tessera_c4_install --type r7i.2xlarge --no-weights \\
      --args "--bank s3://.../tessera_k1_banks_caption2/out/bank.pt --confound s3://.../tessera_c4_confound/out/metrics.json" \\
      --s3-prefix results/tessera_c4_install --budget-min 240

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

PG1_REPO = "google/paligemma-3b-pt-224"
GEMMA1_REPO = "google/gemma-2b"
RANKS = (20, 50, 350, 679)
BANK_FACULTY = "caption"          # what K4 used; changing it breaks comparability


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", required=True, help="K1 faculty bank (.pt) — must be the one K4 used")
    ap.add_argument("--confound", required=True, help="tessera_c4_confound metrics.json")
    a = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from infra.aws_pg.jobs.tessera_s_core import _ftype, _hf_lm_sd
    from infra.pg import cleave_core as CL

    subprocess.run(["aws", "s3", "cp", a.bank, "/tmp/c4_bank.pt", "--only-show-errors"], check=True)
    bank = torch.load("/tmp/c4_bank.pt", map_location="cpu", weights_only=False)

    w_pg1 = _hf_lm_sd(PG1_REPO)
    w_g1 = _hf_lm_sd(GEMMA1_REPO)
    keys = sorted(set(w_pg1) & set(w_g1))
    assert len(keys) == 126, f"{len(keys)} shared LM decoder matrices, want 126"

    acc = {p: {r: 0.0 for r in RANKS} for p in ("energy", "susceptibility", "random")}
    floor, wsum = {}, 0.0
    for k in keys:
        L = int(k.split(".")[1])
        # install = PG1 - gemma-2b (the CREATION direction, matching Arm S and A10)
        dW = (w_pg1[k].to(torch.float64) - w_g1[k].to(torch.float64)).numpy()
        fro2 = float((dW ** 2).sum())
        if fro2 <= 0:
            continue
        for r in RANKS:
            cen = CL.mechanism_census(dW, bank[L], r, _ftype(k))
            for p in acc:
                acc[p][r] += cen[p] * fro2
            floor[r] = cen["chance_floor"]
        wsum += fro2
    frac = {p: {str(r): acc[p][r] / wsum for r in RANKS} for p in acc}
    conc = {str(r): frac["energy"][str(r)] / float(floor[r]) for r in RANKS}

    cf = _read(a.confound)
    hop = {u.replace("HOP/", ""): v["concentration"]
           for u, v in [(x["unit"], x) for x in cf["per_unit"]] if u.startswith("HOP/")}
    r20 = {"install": conc["20"], **{u: v["20"] for u, v in hop.items()}}
    order = sorted(r20, key=lambda u: r20[u])            # ascending = flattest first
    flattest, most_conc = order[0], order[-1]
    if most_conc == "install":
        verdict = ("CONFIRMED — install is the MOST concentrated (least flat) hop, so the "
                   "concentration order reproduces the depression order across all six C4 pairs")
    elif flattest == "install":
        verdict = ("REFUTED — install is the FLATTEST hop, yet sits in the LEAST depressed pairs. "
                   "Flatness does not explain C4's ordering and the avoidance reading returns to "
                   "the table")
    else:
        verdict = ("PARTIAL — install sits between the two VLA hops; the FT-vs-hop family "
                   "separation stands but the within-hop order does not reproduce C4's ordering")

    rep = {"object": "TESSERA-FIX — install write energy census, testing the C4-confound prediction",
           "prediction": "install must be the MOST concentrated of the three hops if flatness "
                         "drives C4's below-reference pattern",
           "bank_faculty": BANK_FACULTY, "bank": a.bank,
           "write": "install = PG1 - gemma-2b over 126 LM decoder matrices",
           "n_matrices": len(keys), "total_fro2": wsum,
           "fractions": frac, "chance_floor": {str(r): float(floor[r]) for r in RANKS},
           "install_concentration": conc,
           "hop_concentration_r20": r20,
           "order_ascending_flattest_first": order,
           "ft_median_r20": cf["by_rank"]["20"]["ft_median"],
           "verdict": verdict, "elapsed_s": time.time() - t0}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))
    print("install concentration (energy ÷ floor): "
          + " ".join(f"r={r}:{conc[str(r)]:.3f}" for r in RANKS))
    print("hop order at r=20 (flattest first): " + " < ".join(f"{u} {r20[u]:.3f}" for u in order))
    print(f"FT median r=20: {cf['by_rank']['20']['ft_median']:.3f}")
    print(f"\n{verdict}")
    print("TESSERA_C4_INSTALL_OK", flush=True)


if __name__ == "__main__":
    main()
