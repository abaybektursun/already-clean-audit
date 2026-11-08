"""N1 pre-check at zero GPU (studies/next/07_PICKS/N1_SPEC.md section 2.1): where each birth's onset sits on the
16-point grid, from the banked exemplar activations.

For each of the 153 births and its two twins (dt, pb) the trajectory r_t = median over E_c of a+_t divided by the pole's
own q999_t, t = 0..15 (J1's recount arrays: the activation at each T_15 exemplar position read at every grid point). The
dating rule (m2.births_matrix) gives k: present at t when the median is > 0 and r_t >= 1; birth = the first t present at
t and t + 1. Per unit this script writes:
  k, the bracket in steps (T_(k-1), T_k], r at T_(k-3) .. T_(k+1);
  transient      present at some t < k without being present at t + 1 (an earlier crossing the rule skips);
  rise_k         log10 r_k - log10 r_(k-1), decades (r floored at 1e-3);
  rise_pre       log10 r_(k-1) - log10 r_(k-3): the rise over the two brackets before;
  concentrated   rise_k > rise_pre (the dated bracket holds more rise than the two before it together);
  f_cross        exploratory: where r crosses 1 inside (T_(k-1), T_k] by log-linear interpolation of log r against
                 log step (0 = at T_(k-1), 1 = at T_k); it assumes a smooth monotone rise inside the bracket, which no
                 bank can check;
and the counts by kind and half. No bank holds anything finer than the grid: the onset inside a bracket needs forwards
at intermediate checkpoints (section 2.1 of the spec costs them).

    python studies/next/n1/onsets/onsets.py --bank <BANK/neurons> --out studies/next/n1/onsets
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

from infra.pg import m2, n1

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="/Users/abaybektursun/projects/interp_research_banks/neurons")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    z = np.load(pathlib.Path(a.bank) / m2.BANK_FILES["arrays"])
    assert m2.sha256_file(pathlib.Path(a.bank) / m2.BANK_FILES["arrays"]) == m2.BANK_SHA["arrays"]
    U = json.loads((REPO_ROOT / "studies/next/n12/units.json").read_text())["units"]
    rows = [n1.onset_row(u, z) for u in U]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["kind"], r["half"])].append(r)
    summ = {}
    for (kind, half), rs in sorted(by.items()):
        f = np.array([r["f_cross"] for r in rs if r["f_cross"] is not None])
        summ[f"{kind}_{half}"] = {
            "n": len(rs), "k_matches_unit": sum(r["k"] == r["k_unit"] for r in rs),
            "transient_before_k": sum(r["transient"] for r in rs),
            "concentrated": sum(r["concentrated"] for r in rs),
            "r_pre_quantiles_10_25_50_75_90": np.quantile([r["r"]["pre"] for r in rs], [.1, .25, .5, .75, .9]).round(4).tolist(),
            "r_birth_quantiles_10_25_50_75_90": np.quantile([r["r"]["birth"] for r in rs], [.1, .25, .5, .75, .9]).round(4).tolist(),
            "rise_k_median": round(float(np.median([r["rise_k"] for r in rs])), 4),
            "rise_pre_median": round(float(np.median([r["rise_pre"] for r in rs])), 4),
            "f_cross_quantiles_10_25_50_75_90": np.quantile(f, [.1, .25, .5, .75, .9]).round(3).tolist() if f.size else None,
            "bracket_steps_median": float(np.median([r["bracket_steps"] for r in rs]))}
    real = [r for r in rows if r["kind"] == "real"]
    by_k = collections.Counter(r["k"] for r in real)
    res = {"what": __doc__.split("\n\n")[0], "bank_arrays_sha256": m2.BANK_SHA["arrays"],
           "units_sha256": m2.sha256_file(REPO_ROOT / "studies/next/n12/units.json"),
           "summary": summ, "real_by_k": dict(sorted(by_k.items())),
           "bracket_steps_by_k": {k: m2.STEPS[k] - m2.STEPS[k - 1] for k in sorted(by_k)}, "units": rows}
    (out / "onsets.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"summary": summ, "real_by_k": res["real_by_k"], "bracket_steps_by_k": res["bracket_steps_by_k"]},
                     indent=1))
    print("N1_ONSETS_OK")


if __name__ == "__main__":
    main()
