"""TESSERA-FIX C4-CONFOUND — is the C4 "avoidance" reading explained by write flatness? Spec §8.

THE QUESTION. C4 found 294 of 702 cross-hop overlap cells sitting BELOW the unrelated-FT reference
band, and TESSERA read that as "the lineage writes avoid each other more than unrelated fine-tunes
do". The independent re-audit (B9) named a competing mechanism: `subspace_overlap` compares only the
top-r=64 right-singular subspaces and is **magnitude-free**, so if the hop writes have FLATTER
spectra than a single-task fine-tune, their top-64 captures a smaller, less-concentrated slice of
where the write lives — depressing overlap with no avoidance at all.

That was recorded as "unmeasurable, because per-unit spectra are not banked" (AF-3). It is not
unmeasurable. **A directly comparable concentration statistic IS banked, on both sides:**

  `cleave_core.mechanism_census(dW, bank, r, t)` returns an `energy` fraction and its analytic
  `chance_floor` at ranks {20, 50, 350, 679}.
    · QUORUM `quorum_arm0_{a,b,c}` ran it over the PG1 fine-tune donors
    · TESSERA K4 (`tessera_k24b`) ran the SAME function at the SAME ranks over π0 and π0.5

`energy / chance_floor` is exactly the concentration the confound is about: 1.0 = isotropic (flat),
larger = more of the write's energy inside the top-r energy subspace. No new compute.

WHAT WOULD REFUTE THE CONFOUND. Hop writes at or above the FT concentration distribution — then
flatness cannot depress hop overlap and C4's avoidance reading survives. This is a real fork; the
statistic distinguishes the two.

CAVEATS, stated because they bound the strength and not the direction:
  1. the two censuses use different energy BANKS (QUORUM: its own task banks; K4: the `caption`
     faculty bank). Both are activation-energy bases on the same base model, but they are not the
     identical basis, so this is a strong indication rather than a matched comparison;
  2. `install`'s concentration was never censused — K4 covered π0 and π0.5 only. Any claim about
     where `install` sits in the ordering is a PREDICTION here, not a measurement;
  3. n_hop = 2.

  python -m infra.aws_pg.jobs.tessera_c4_confound --out d --c4 <c4.json> --k24 <k24.json>

No try/except (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import subprocess

QUORUM_PREFIXES = ("quorum_arm0_a", "quorum_arm0_b", "quorum_arm0_c")
RANKS = (20, 50, 350, 679)


def _s3(prefix):
    r = subprocess.run(["aws", "s3", "cp",
                        f"s3://pg-lineage-769063704696/results/{prefix}/out/metrics.json", "-",
                        "--profile", "fuelos", "--region", "us-east-1"],
                       capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def _read(p):
    if str(p).startswith("s3://"):
        return json.loads(subprocess.run(["aws", "s3", "cp", str(p), "-"], check=True,
                                         capture_output=True, text=True).stdout)
    return json.loads(pathlib.Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--c4", required=True)
    ap.add_argument("--k24", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ft = {}
    for pref in QUORUM_PREFIXES:
        d = _s3(pref)
        if d:
            for k, v in d.get("per_donor", {}).items():
                ft[k] = v["census"]
    assert len(ft) >= 10, f"only {len(ft)} FT donors recovered — too few to characterise the family"

    k24 = _read(a.k24)
    hop_rows, ft_rows = [], []
    for u, c in sorted(ft.items()):
        ft_rows.append({"unit": f"FT/{u}", "family": "FT",
                        "concentration": {str(r): c[str(r)]["energy"] / c[str(r)]["chance_floor"]
                                          for r in RANKS}})
    for u, rec in k24["K4_census"].items():
        fr, fl = rec["fractions"], rec["chance_floor"]
        hop_rows.append({"unit": f"HOP/{u}", "family": "HOP",
                         "concentration": {str(r): fr["energy"][str(r)] / float(fl[str(r)])
                                           for r in RANKS}})

    summ = {}
    for r in RANKS:
        f = [x["concentration"][str(r)] for x in ft_rows]
        h = [x["concentration"][str(r)] for x in hop_rows]
        below = [x["unit"] for x in hop_rows if x["concentration"][str(r)] < min(f)]
        summ[str(r)] = {"ft_median": st.median(f), "ft_min": min(f), "ft_max": max(f),
                        "hop_values": {x["unit"]: x["concentration"][str(r)] for x in hop_rows},
                        "hop_median": st.median(h),
                        "hops_below_every_ft": below,
                        "hop_median_over_ft_median": st.median(h) / st.median(f)}

    flatter = all(summ[str(r)]["hop_median"] < summ[str(r)]["ft_median"] for r in RANKS)
    # The confound needs the hops to be FLATTER. If they are not, it cannot depress hop overlap.
    verdict = ("CONFOUND-SUPPORTED — hop writes are less energy-concentrated than FT writes at every "
               "rank, which is exactly the condition under which a magnitude-free top-r overlap is "
               "depressed for hop pairs without any avoidance"
               if flatter else
               "CONFOUND-REFUTED — hop writes are not flatter than FT writes, so flatness cannot "
               "explain the C4 pattern and the avoidance reading survives this test")

    # Does the per-unit flatness ORDER predict C4's per-pair ordering? Flatter ⇒ more depressed.
    c4 = _read(a.c4)
    order = sorted(hop_rows, key=lambda x: x["concentration"]["20"])
    pairs = {"~".join(p["pair"]): p["frac_below_ft_reference"] for p in c4["per_pair"]}
    rep = {"object": "TESSERA-FIX — does write flatness explain C4's below-reference pattern?",
           "statistic": "energy fraction ÷ analytic chance floor (cleave_core.mechanism_census); "
                        "1.0 = isotropic, larger = more concentrated",
           "sources": {"ft": list(QUORUM_PREFIXES), "hop": "tessera_k24b K4_census"},
           "caveats": ["the two censuses use different energy BANKS (QUORUM: task banks; K4: the "
                       "caption faculty bank) — a strong indication, not a matched comparison",
                       "`install` was never censused; its place in the ordering is a PREDICTION",
                       f"n_hop = {len(hop_rows)}"],
           "per_unit": ft_rows + hop_rows, "by_rank": summ,
           "flatness_order_ascending": [x["unit"] for x in order],
           "c4_frac_below": pairs,
           "verdict": verdict,
           "prediction_if_confound": "if flatness drives C4, `install` — the only hop not censused — "
                                     "must be the LEAST flat of the three, because the pairs "
                                     "containing it are the least depressed (install~pi0 0.500 and "
                                     "install~ft24 0.139 are the minima of their classes). "
                                     "Censusing the install write at these ranks tests it directly."}
    (out / "metrics.json").write_text(json.dumps(rep, indent=1, default=str))

    print("ENERGY CONCENTRATION (energy ÷ chance floor); 1.0 = flat/isotropic")
    print("%-18s %8s %8s %8s %8s" % ("unit", *[f"r={r}" for r in RANKS]))
    for x in ft_rows + hop_rows:
        print("%-18s %8.3f %8.3f %8.3f %8.3f"
              % (x["unit"], *[x["concentration"][str(r)] for r in RANKS]))
    print()
    for r in RANKS:
        s = summ[str(r)]
        print("r=%-5d FT median %.3f [%.3f, %.3f] · HOP median %.3f · ratio %.3f · below every FT: %s"
              % (r, s["ft_median"], s["ft_min"], s["ft_max"], s["hop_median"],
                 s["hop_median_over_ft_median"], s["hops_below_every_ft"] or "none"))
    print(f"\n{verdict}")
    print("TESSERA_C4_CONFOUND_OK", flush=True)


if __name__ == "__main__":
    main()
