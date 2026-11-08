"""TESSERA-FIX C4 — recount Arm S's overlaps with A12's missing symmetric branch. Spec §2, §3 row 13.

FINDING 13. A12 added a second null (the unrelated-FT reference) to stop a lineage-wide "shared
channel" being read off a Haar band that any two large writes clear. It shipped ONE-SIDED: it could
demote SHARED-CHANNEL → ENERGY-ALIGNED when an overlap sat *below* the FT reference's p97.5, but it
had no branch at all for an overlap below the reference's **p2.5** — two writes that avoid each other
MORE than two unrelated fine-tunes do. That is as much a structure claim as the upper tail, and the
rule as written could not report it. `overlap_verdict` was already two-sided against the Haar band;
A12's layer on top was not.

This job is a pure RECOUNT. `tessera_s_full2` banks every per-matrix overlap and the reference null's
`p2.5` alongside its `p97.5`, so nothing is re-measured and no weights are touched — the numbers
existed before the branch did. The fix in `tessera_s_core.py` makes future runs symmetric; this makes
the banked run symmetric.

  python -m infra.aws_pg.jobs.tessera_c4 --out c4 --s <tessera_s_full2.json>

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

# The lineage hop-hop pairs: the ones H_hops-disjoint-below-ft is about. `~ft24` pairs compare a hop
# against the FT family and are reported but are not the hypothesis.
HOP_HOP = {("install", "pi0"), ("install", "pi05"), ("pi0", "pi05")}


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--s", required=True, help="banked tessera_s_full2 metrics.json")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    s = _read(a.s)
    ftref = (s.get("ft_reference_null") or {}).get("per_key", {})
    assert ftref, "no banked ft_reference_null — C4 has nothing to recount against"
    missing_p25 = [k for k, v in ftref.items() if "p2.5" not in v]
    assert not missing_p25, f"{len(missing_p25)} reference keys lack p2.5, e.g. {missing_p25[:3]}"

    rows, per_pair = [], []
    for o in s["overlaps"]:
        pair = tuple(o["pair"])
        n_below = n_above = n_scored = 0
        for r in o["per_key"]:
            ref = ftref.get(r["key"])
            if not ref:
                continue
            n_scored += 1
            below = bool(r["overlap"] < ref["p2.5"])
            above = bool(r["overlap"] > ref["p97.5"])
            n_below += int(below)
            n_above += int(above)
            rows.append({"pair": "~".join(pair), "key": r["key"], "overlap": r["overlap"],
                         "ft_ref_p2_5": ref["p2.5"], "ft_ref_median": ref["median"],
                         "ft_ref_p97_5": ref["p97.5"],
                         "below_ft_ref": below, "above_ft_ref": above})
        # `banked` is what the one-sided rule reported; `recount` is what the symmetric rule reports.
        per_pair.append({
            "pair": list(pair), "is_hop_hop": bool(frozenset(pair) in
                                                   {frozenset(p) for p in HOP_HOP}),
            "n_scored": n_scored, "median_overlap": o["median_ci"]["point"],
            "banked": {"shared": o["n_shared_channel"], "energy": o.get("n_energy_aligned", 0),
                       "anti": o["n_anti_aligned"], "haar_disjoint": o["n_haar_disjoint"]},
            "recount": {"below_ft_reference": n_below, "above_ft_reference": n_above,
                        "inside_ft_reference": n_scored - n_below - n_above},
            "frac_below_ft_reference": (n_below / n_scored) if n_scored else None})

    hop = [p for p in per_pair if p["is_hop_hop"]]
    tot_below = sum(p["recount"]["below_ft_reference"] for p in hop)
    tot_scored = sum(p["n_scored"] for p in hop)
    # H_hops-disjoint-below-ft (spec §1.2): killed by hop-hop overlaps sitting inside/above the FT
    # reference. "Fires" needs the lower branch to actually carry cells, not merely to exist.
    verdict = ("FIRES" if tot_below > 0 else "DIES — the one-sided A12 branch hid nothing")
    rep = {"object": "TESSERA-FIX C4 — A12 symmetric recount (banked data, zero new measurement)",
           "finding": 13,
           "source": a.s,
           "rule": "BELOW-FT-REFERENCE iff overlap < ft_reference_null p2.5, per matrix; the "
                   "symmetric partner of A12's existing overlap > p97.5 branch",
           "H_hops_disjoint_below_ft": {
               "verdict": verdict,
               "hop_hop_below": tot_below, "hop_hop_scored": tot_scored,
               "hop_hop_fraction": (tot_below / tot_scored) if tot_scored else None},
           "per_pair": per_pair, "per_key": rows}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    print("%-16s %-8s %8s %10s %10s %10s %8s"
          % ("pair", "hop-hop", "n", "median", "BELOW", "inside", "ABOVE"))
    for p in per_pair:
        print("%-16s %-8s %8d %10.5f %10d %10d %8d"
              % ("~".join(p["pair"]), p["is_hop_hop"], p["n_scored"], p["median_overlap"],
                 p["recount"]["below_ft_reference"], p["recount"]["inside_ft_reference"],
                 p["recount"]["above_ft_reference"]))
    print(f"\nH_hops-disjoint-below-ft: {verdict} "
          f"({tot_below}/{tot_scored} hop-hop cells below the FT reference)")
    print("TESSERA_C4_OK", flush=True)


if __name__ == "__main__":
    main()
