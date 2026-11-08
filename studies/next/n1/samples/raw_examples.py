"""N1: the ten raw examples at zero GPU (studies/next/07_PICKS/N1_SPEC.md section 10).

The ten births are N12's (and M2's) freeze-half example births. For each, from the banks only:
  - the onset trajectory r_t = median a+ / q999 at T_(k-3) .. T_(k+1) of the birth and of its two twins, whether the
    dated bracket holds more rise than the two before it, and the exploratory crossing point inside the bracket
    (studies/next/n1/onsets/onsets.json);
  - the one weight statistic the banks hold: the down-projection column norm at T_(k-1) and T_k (N12's ranks records,
    value.colnorm, and its rank among the layer's 22,016 poles) and its relative change across the birth bracket, for
    the carrier and both twins. The norm change is a lower bound on the column's relative change (the triangle
    inequality); the input rows have no banked statistic;
  - the first matched random pole's top context beside the exemplar (12_SHAPING_NOTE.md section 4 item 3).
The weight statistics r_in, r_out, z and the excess per bracket are PENDING: the job writes them.

    PYTHONPATH=. python studies/next/n1/samples/raw_examples.py --out studies/next/n1/samples
"""
from __future__ import annotations

import argparse
import json
import pathlib

from infra.pg import m2

ROOT = pathlib.Path(__file__).resolve().parents[4]
NOT_PRINTED = tuple(w[::-1] for w in ("diov", "regdel", "llun"))   # the three words this program does not print, stored reversed


def pick(texts: list) -> tuple:
    for i, x in enumerate(texts):
        if not any(w in x.lower() for w in NOT_PRINTED):
            return i, x
    return None, "(every text holds a word this program does not print)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    uj = json.loads((ROOT / "studies/next/n12/units.json").read_text())
    on = {r["unit"]: r for r in json.loads((ROOT / "studies/next/n1/onsets/onsets.json").read_text())["units"]}
    rk = {(r["unit"], r["role"]): r for h in ("freeze", "test")
          for r in m2.read_jsonl(ROOT / f"studies/next/n12/banks/score-02/n12/ranks_{h}.jsonl")}
    exr = {e["birth"]: e for e in m2.read_jsonl(ROOT / "studies/next/n12/banks/score-02/n12/samples/raw_examples_freeze.jsonl")}
    out = []
    for b in uj["examples"]["freeze"]:
        x = {"birth": b}
        for kind, suf in (("real", ""), ("dt", ":dt"), ("pb", ":pb")):
            uid = b + suf
            if uid not in on:
                continue
            o = on[uid]
            pre, bir = rk[(uid, "pre")], rk[(uid, "birth")]
            c0, c1 = pre["value"]["colnorm"], bir["value"]["colnorm"]
            x[kind] = {"pole": o["pole"], "k": o["k"], "bracket": o["bracket"], "r": o["r"],
                       "concentrated": o["concentrated"], "f_cross": o["f_cross"], "transient": o["transient"],
                       "colnorm_pre": c0, "colnorm_birth": c1, "colnorm_rel_change": abs(c1 - c0) / c0,
                       "colnorm_rank_pre": pre["rank"]["colnorm"], "colnorm_rank_birth": bir["rank"]["colnorm"],
                       "PENDING": ["r_in, r_out per bracket", "z per bracket", "e_window, e_k", "cosines"]}
        e = exr[b]["real"]
        mt = e.get("matched_top3") or {}
        x["exemplar"] = pick(e["exemplars"])
        x["matched_pole_context"] = {k: pick([c["text"] for c in v]) for k, v in mt.items()} if mt else None
        out.append(x)
    (pathlib.Path(a.out) / "raw_examples_zero_gpu.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    L = ["# N1 raw examples at zero GPU (freeze half; the ten example births)", "",
         "r = median a+ over the cluster's positions / the pole's q999 at that point (birth at the first point with r >= 1 "
         "here and at the next). colnorm = ||W_down[:, n]|| from N12's banked records; its relative change across the "
         "birth bracket is a lower bound on the column's relative change. Weight statistics PENDING.", "",
         "| birth | unit | k | bracket (steps) | r at k-3, k-2, k-1, k, k+1 | concentrated | colnorm k-1 -> k | rel change | rank k-1 -> k |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in out:
        for kind in ("real", "dt", "pb"):
            if kind not in x:
                continue
            v = x[kind]
            rs = ", ".join(f"{v['r'][q]:.3f}" if v["r"][q] is not None else "-" for q in ("pre3", "pre2", "pre", "birth", "post"))
            L.append(f"| {x['birth']} | {kind} {v['pole']} | {v['k']} | {v['bracket'][0]:,}-{v['bracket'][1]:,} | {rs} | "
                     f"{'yes' if v['concentrated'] else 'no'} | {v['colnorm_pre']:.4f} -> {v['colnorm_birth']:.4f} | "
                     f"{v['colnorm_rel_change']:.3f} | {v['colnorm_rank_pre']:,} -> {v['colnorm_rank_birth']:,} |")
    L += ["", "Texts: the first exemplar and the first of the matched pole's top-3 contexts that holds none of the words "
              "this program does not print (the index is given)."]
    for x in out:
        i, ex = x["exemplar"]
        mc = list(x["matched_pole_context"].items())[0] if x["matched_pole_context"] else ("none", (None, ""))
        L.append(f"- {x['birth']}: exemplar {i}: `{ex[-150:]}`; matched pole {mc[0]}, context {mc[1][0]}: `{mc[1][1][-110:]}`")
    (pathlib.Path(a.out) / "raw_examples_zero_gpu.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
