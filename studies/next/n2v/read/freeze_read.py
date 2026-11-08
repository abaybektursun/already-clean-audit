"""N2V spec section 7 step 3: the freeze-half read before freeze 2 (laptop; banked JSON only; no test-half record opened).

    PYTHONPATH=. python studies/next/n2v/read/freeze_read.py --bank studies/next/n2v/banks/score-02 \
        --out studies/next/n2v/read/freeze_read.json

Answers, from n2v/metrics.json, n2v/n2v_freeze.jsonl, n2v/samples/raw_examples_freeze.jsonl and
studies/next/n2v/samples/raw_examples_zero_gpu.json (N12's exemplars and bf16 values of the same births):
  1. the decoded exemplars are N12's: N12's exemplar text (the zero-GPU file's pick) appears verbatim among the job's
     decoded exemplars (the job lists them in window order, N12 in its own order, so the position is reported);
  2. the rows' losses before i0 equal the clean row's: G-CAUSAL at every point (count bitwise, largest difference);
  3. the carrier's five largest per-position changes sit at or after the zeroed indices: j >= i0 of that window;
  4. the probe is at the fp32 floor: |probe| against |dL| per pair, over the freeze half;
  5. one pair with |dL - pred_bf16| small and one with it large among the examples, with their tokens.
"""
import argparse
import json
import pathlib

import numpy as np

from infra.pg import m2, n2v

ROOT = pathlib.Path(__file__).resolve().parents[4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bank = pathlib.Path(a.bank) / "n2v"
    jm = json.loads((bank / "metrics.json").read_text())
    fz = m2.read_jsonl(bank / "n2v_freeze.jsonl")
    assert all(p["half"] == "freeze" for p in fz)
    exs = m2.read_jsonl(bank / "samples" / "raw_examples_freeze.jsonl")
    zg = {x["birth"]: x for x in json.loads((ROOT / "studies/next/n2v/samples/raw_examples_zero_gpu.json").read_text())}
    units = json.loads((ROOT / "studies/next/n12/units.json").read_text())["units"]
    n12r = [r for h in ("freeze", "test") for r in m2.read_jsonl(ROOT / f"studies/next/n12/banks/score-02/n12/n12_{h}.jsonl")]
    uw = {it["unit"]: dict(it["uw"]) for items in n2v.plan(units, n12r).values() for it in items}
    out = {"examples": []}
    for e in exs:
        b = e["unit"]
        i, txt = zg[b]["exemplar_index_and_text"]
        top = e["carrier_zero_top5"]
        i0 = min(uw[b][top[0]["window"]])
        pr = [dict(p, abs_dev=abs(p["dL"] - p["pred_bf16"])) for p in e["pairs"]]
        out["examples"].append({
            "birth": b, "t": e["t"], "exemplar_equal_to_N12": txt in e["exemplars"], "N12_index": i,
            "job_index": e["exemplars"].index(txt) if txt in e["exemplars"] else None, "exemplar": txt,
            "i0": i0, "top5_j": [x["j"] for x in top], "top5_at_or_after_i0": all(x["j"] >= i0 for x in top),
            "top5": top,
            "pairs": [{k: p[k] for k in ("pole", "rank", "pred_bf16", "meas_bf16", "dL", "dL_own", "dL_fd", "pred_jvp",
                                         "pred_jvp_own", "probe", "probe_own", "abs_dev")} for p in pr]})
    gc = {k: {x: g[x] for x in ("n_rows", "n_bitwise_before_i0", "max_abs_before_i0", "n_over_tol")}
          for k, g in jm["gates"].items() if k.endswith("_G_CAUSAL")}
    car = [p for p in fz if p["rank"] == 0]
    ratio = np.array([abs(p["probe"]) / abs(p["dL"]) for p in fz if p["rank"] == 0 and p["dL"] != 0])
    allp = [(x["birth"], p) for x in out["examples"] for p in x["pairs"]]
    lo, hi = min(allp, key=lambda z: z[1]["abs_dev"]), max(allp, key=lambda z: z[1]["abs_dev"])
    tok = {x["birth"]: (x["exemplar"], x["top5"][0]) for x in out["examples"]}
    out["summary"] = {
        "q1_N12_exemplar_found_verbatim": f"{sum(x['exemplar_equal_to_N12'] for x in out['examples'])} of {len(out['examples'])}",
        "q2_G_CAUSAL": gc,
        "q3_top5_at_or_after_i0": f"{sum(x['top5_at_or_after_i0'] for x in out['examples'])} of {len(out['examples'])}",
        "q4_probe": {"n_carrier_pairs": len(car), "median_abs_probe": float(np.median([abs(p["probe"]) for p in car])),
                     "max_abs_probe": float(max(abs(p["probe"]) for p in car)),
                     "median_abs_dL": float(np.median([abs(p["dL"]) for p in car])),
                     "median_abs_probe_over_abs_dL": float(np.median(ratio)), "max_abs_probe_over_abs_dL": float(ratio.max()),
                     "median_abs_probe_own": float(np.median([abs(p["probe_own"]) for p in car]))},
        "q5_small": {"birth": lo[0], **lo[1], "exemplar": tok[lo[0]][0], "carrier_top_change": tok[lo[0]][1]},
        "q5_large": {"birth": hi[0], **hi[1], "exemplar": tok[hi[0]][0], "carrier_top_change": tok[hi[0]][1]}}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
