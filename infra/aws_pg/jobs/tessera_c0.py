"""TESSERA-FIX C0 — is the identity band's [0,0] a decoder-symmetry artifact? Spec §1.2, §2, §7.2.

THE SPEC'S FORM, AND WHY IT CHANGED (amendment AF-2, recorded before the read).
§2 defines C0 as `max|W_dec[0] − W_dec[1]|` over "the banked identity crosscoder". No crosscoder
WEIGHTS were ever banked — `tessera_x.py` writes `metrics.json` only, and an S3 listing of
`tessera_x_{c,d}/out/` shows nothing else. Re-training one to read its decoder would cost ~1 GPU-h
and would not answer the question about the crosscoders that actually produced the banked numbers,
only about a fresh one.

The banked `rel_norm_hist` is a SUFFICIENT STATISTIC for the question, and it describes the exact
dictionaries that produced Results 5 and 6:

    r = ‖W_dec[after]‖ / (‖W_dec[before]‖ + ‖W_dec[after]‖)

    W_dec[0] ≡ W_dec[1]  ⟹  r = 0.5 EXACTLY for every live latent
                         ⟹  the histogram is a single spike in the bin containing 0.5

So: mass in more than one bin ⟹ the two decoder halves differ ⟹ H_band-artifact is REBUTTED. This
is free, it is terminal either way, and it reads the real dictionaries rather than a proxy.

WHAT THE READ ALSO DELIVERS, which the spec's form could not. `max|W_dec[0]−W_dec[1]|` answers
"are they identical". The histogram answers the question that actually decides whether the identity
band means anything: **how far from 0.5 does this pair's r-distribution reach, compared with how far
it would have to reach to score?** `delta_crosscoder` classifies at r ≥ 0.90 (emergent) and r ≤ 0.10
(lost). A pair whose entire live population sits within ±0.02 of 0.5 cannot produce a nonzero
fraction no matter what it does — its band is [0,0] by construction of the dictionary, not by
evidence about the model pair. That is the critic's substantive point, and it is checkable here.

  python -m infra.aws_pg.jobs.tessera_c0 --out c0 --x <merged_x.json>

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from infra.pg.delta_crosscoder import RELNORM_EMERGENT_MIN, RELNORM_LOST_MAX

FP_TOL = 1e-12          # "more than one bin" needs no tolerance; kept for the identity-mass check


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def hist_reach(hist):
    """How far the live r-distribution reaches, and how far it would need to reach to score."""
    c, e = hist["counts"], hist["edges"]
    tot = sum(c)
    assert tot > 0, "empty rel-norm histogram"
    nz = [i for i, n in enumerate(c) if n > 0]
    lo_edge, hi_edge = e[nz[0]], e[nz[-1] + 1]
    # Distance from 0.5 that the population actually reaches, and the distance either cut demands.
    reach = max(abs(hi_edge - 0.5), abs(0.5 - lo_edge))
    need = min(RELNORM_EMERGENT_MIN - 0.5, 0.5 - RELNORM_LOST_MAX)
    return {"n_live": tot, "n_nonzero_bins": len(nz),
            "r_min_edge": lo_edge, "r_max_edge": hi_edge,
            "reach_from_half": reach, "reach_needed_to_score": need,
            "headroom_ratio": reach / need,
            "can_reach_a_cut": bool(hi_edge > RELNORM_EMERGENT_MIN or lo_edge < RELNORM_LOST_MAX)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--x", required=True, help="merged Arm X artifact")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    x = _read(a.x)
    rows = []
    for pn, pr in x["pairs"].items():
        for L, s in sorted(pr.get("sites", {}).items(), key=lambda kv: int(kv[0])):
            for q in s.get("seeds", []):
                h = q.get("rel_norm_hist")
                if not h or not h.get("counts"):
                    continue
                rows.append({"pair": pn, "site": int(L), "seed": q["seed"],
                             "emergent_fraction": q["emergent_fraction"],
                             "lost_fraction": q["lost_fraction"], **hist_reach(h)})

    idr = [r for r in rows if r["pair"] == "identity"]
    assert idr, "no identity rows — C0 has nothing to adjudicate"
    multi = all(r["n_nonzero_bins"] > 1 for r in idr)
    # H_band-artifact: the band is [0,0] because W_dec[0] ≡ W_dec[1]. Killed by any spread at all.
    verdict = ("REBUTTED — the identity decoder halves are NOT identical" if multi else
               "SUPPORTED — every identity latent sits in a single bin, consistent with W_dec[0] ≡ W_dec[1]")

    # The separate, sharper question the histogram can answer.
    id_reach = max(r["reach_from_half"] for r in idr)
    id_ratio = max(r["headroom_ratio"] for r in idr)
    structural = not any(r["can_reach_a_cut"] for r in idr)

    rep = {"object": "TESSERA-FIX C0 — identity band: artifact or evidence?",
           "amendment": "AF-2 — crosscoder WEIGHTS were never banked (tessera_x.py writes "
                        "metrics.json only), so the spec's max|W_dec[0]-W_dec[1]| form is not "
                        "runnable on the dictionaries that produced the results. The banked "
                        "rel_norm_hist is a sufficient statistic for the symmetry question AND "
                        "reads the real dictionaries. Recorded before the read.",
           "cuts": {"emergent_min": RELNORM_EMERGENT_MIN, "lost_max": RELNORM_LOST_MAX},
           "H_band_artifact": {"verdict": verdict, "all_identity_multi_bin": multi},
           "identity_reach": {
               "max_reach_from_half": id_reach,
               "reach_needed_to_score": 0.5 - RELNORM_LOST_MAX,
               "headroom_ratio": id_ratio,
               "structurally_unable_to_score": structural,
               "reading": "if the identity population cannot reach either cut, its [0,0] band is a "
                          "property of the dictionary's decoder-norm concentration, not evidence "
                          "about the model pair — which is why a nonzero-width reference pair (C1) "
                          "is required before any nonzero Arm-X reading can be called real"},
           "per_cell": rows}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    print("%-10s %-4s %-4s %6s %8s %8s %9s %9s %s"
          % ("pair", "site", "seed", "bins", "r_min", "r_max", "reach", "ratio", "can_score"))
    for r in rows:
        print("%-10s %-4d %-4d %6d %8.3f %8.3f %9.4f %9.4f %s"
              % (r["pair"], r["site"], r["seed"], r["n_nonzero_bins"], r["r_min_edge"],
                 r["r_max_edge"], r["reach_from_half"], r["headroom_ratio"],
                 r["can_reach_a_cut"]))
    print(f"\nH_band-artifact: {verdict}")
    print(f"identity reach from 0.5: {id_reach:.4f} vs {0.5 - RELNORM_LOST_MAX:.2f} needed to score "
          f"(ratio {id_ratio:.4f}); structurally unable to score = {structural}")
    print("TESSERA_C0_OK", flush=True)


if __name__ == "__main__":
    main()
