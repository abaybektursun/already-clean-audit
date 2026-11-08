#!/usr/bin/env python3
"""V2a offline scoring — FROZEN before the MDE data exists (blind order).

From the banked mq2 atlas.jsonl: final gauge_task_loss per (gauge, seed);
matched-seed deltas hadamard - unrotated on the FRESH seeds.

FROZEN RULE (recorded in shackleton_expectations.jsonl, §6 — thresholds from
THIS run's own spread, never imported):
  band     = max(SD(unrotated losses), SD(hadamard losses)) across fresh seeds
  REPLICATES   if mean_delta < 0, all matched deltas < 0, and |mean_delta| > band
  REFUTED      if |mean_delta| <= band  (inside the fresh-seed noise band)
  MIXED        otherwise (sign agreement without clearing the band, or split
               seeds) — reported as-is, no post-hoc promotion
  VOID-INSTRUMENT if the run's own function-preservation gates failed (read
               from gates.json), regardless of the deltas.

Usage: python3 analysis_v2.py <atlas.jsonl> <gates.json> --out v2a_score.json
"""
import argparse
import json
import statistics as st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("atlas")
    p.add_argument("gates")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gates = json.load(open(a.gates))
    gate_fail = [k for k, v in gates.items()
                 if isinstance(v, dict) and v.get("passed") is False]

    rows = [json.loads(l) for l in open(a.atlas)]
    tl = {}
    for r in rows:
        if r.get("statistic") != "gauge_task_loss":
            continue
        f = r["flags"]
        f = json.loads(f) if isinstance(f, str) else (f or {})
        tl[(f.get("gauge"), int(f.get("seed")))] = float(r["value"])

    seeds = sorted({s for (g, s) in tl if g == "unrotated"})
    u = [tl[("unrotated", s)] for s in seeds]
    h = [tl[("hadamard", s)] for s in seeds]
    deltas = [h[i] - u[i] for i in range(len(seeds))]
    mean_delta = st.mean(deltas)
    band = max(st.stdev(u), st.stdev(h)) if len(seeds) > 1 else float("nan")

    if gate_fail:
        verdict = "VOID-INSTRUMENT"
    elif mean_delta < 0 and all(d < 0 for d in deltas) and abs(mean_delta) > band:
        verdict = "REPLICATES"
    elif abs(mean_delta) <= band:
        verdict = "REFUTED"
    else:
        verdict = "MIXED"

    out = {"fresh_seeds": seeds,
           "unrotated": [round(x, 4) for x in u],
           "hadamard": [round(x, 4) for x in h],
           "matched_deltas": [round(d, 4) for d in deltas],
           "mean_delta": round(mean_delta, 4), "band": round(band, 4),
           "gate_failures": gate_fail, "verdict": verdict}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
