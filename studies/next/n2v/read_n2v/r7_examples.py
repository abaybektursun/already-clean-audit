"""N2V reader, step 7: the test half's raw examples (samples/raw_examples_test.jsonl) against the npz recount.

    python -I r7_examples.py --s3 <dir holding score-02/n2v> --recount r1_recount.json --repo <worktree> --out r7_examples.json

For every example unit: each pair's numbers in the record against my recount from losses/tNN.npz; the carrier's five
largest per-position changes (the record's carrier_zero_top5: window w0 = the unit's first window, l_zero0 - l_clean)
recomputed from the npz. Then the pairs with the smallest and the largest |dL - pred_bf16| among all 40 pairs and among
the 10 carriers, with the five largest per-position changes of that pole's zero row over all its windows (positions and
values from the npz; the predicted token's text is banked by the job only for the carrier's first window).
"""
import argparse
import json
import pathlib

import numpy as np

W = 2048
FIELDS = ["pred_bf16", "meas_bf16", "dL", "dL_own", "dL_fd", "dL_fd_own", "pred_jvp", "pred_jvp_own", "probe", "probe_own"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3", required=True)
    ap.add_argument("--recount", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s3 = pathlib.Path(a.s3) / "score-02" / "n2v"
    ex = [json.loads(x) for x in (s3 / "samples" / "raw_examples_test.jsonl").read_text().splitlines() if x.strip()]
    rc = {(p["unit"], p["pole"]): p for p in json.loads(pathlib.Path(a.recount).read_text())["pairs"]}
    units = {u["unit"]: u for u in json.loads((pathlib.Path(a.repo) / "studies/next/n12/units.json").read_text())["units"]}
    out = {"n_examples": len(ex), "examples": []}
    allpairs = []
    npz = {}
    for e in ex:
        u = e["unit"]
        uw = {}
        for t in units[u]["tok"]:
            i = 1 + t % W
            if i <= W - 1:
                uw.setdefault(t // W, []).append(i)
        uw = dict(sorted(uw.items()))
        z = npz.setdefault(e["t"], np.load(s3 / "losses" / f"t{e['t']:02d}.npz"))
        diffs = []
        for p in e["pairs"]:
            m = rc[(u, p["pole"])]
            mm = {"pred_bf16": m["pred_bf16_n12"], "meas_bf16": m["meas_bf16_n12"], **{f: m[f] for f in FIELDS[2:]}}
            diffs.append(max(abs(mm[f] - p[f]) for f in FIELDS))
            allpairs.append((u, p, e))
        w0 = next(iter(uw))
        dl = z[f"{u}|{w0}|zero|0"].astype(np.float64) - z[f"{u}|{w0}|clean|None"].astype(np.float64)
        top = np.argsort(-np.abs(dl))[:5]
        rec_top = e["carrier_zero_top5"]
        top_ok = all(int(j) == r["j"] and float(dl[j]) == r["dl"] and float(z[f"{u}|{w0}|clean|None"][j]) == r["clean_loss"]
                     and r["window"] == w0 for j, r in zip(top, rec_top))
        out["examples"].append({"unit": u, "t": e["t"], "n_exemplars": len(e["exemplars"]),
                                "max_abs_diff_record_vs_recount_over_pairs_and_fields": float(max(diffs)),
                                "carrier_top5_equal_to_npz": top_ok, "i0_first_window": min(uw[w0]),
                                "carrier_top5_at_or_after_i0": all(r["j"] >= min(uw[w0]) for r in rec_top)})

    def detail(u, p, e):
        z = npz[e["t"]]
        uw = {}
        for t in units[u]["tok"]:
            i = 1 + t % W
            if i <= W - 1:
                uw.setdefault(t // W, []).append(i)
        q = p["rank"]
        ch = []
        for w in sorted(uw):
            d = z[f"{u}|{w}|zero|{q}"].astype(np.float64) - z[f"{u}|{w}|clean|None"].astype(np.float64)
            for j in np.flatnonzero(d):
                ch.append((abs(d[j]), int(w), int(j), float(d[j]), float(z[f"{u}|{w}|clean|None"][j]), min(uw[w])))
        ch.sort(reverse=True)
        m = rc[(u, p["pole"])]
        return {"unit": u, "pole": p["pole"], "rank": q, "t": e["t"],
                "record": {f: p[f] for f in FIELDS}, "recount": {"pred_bf16": m["pred_bf16_n12"], "meas_bf16": m["meas_bf16_n12"],
                                                                 **{f: m[f] for f in FIELDS[2:]}},
                "abs_dL_minus_pred_bf16": abs(p["dL"] - p["pred_bf16"]),
                "exemplar_first": e["exemplars"][0], "n_exemplars": len(e["exemplars"]),
                "zero_row_top5_changes_all_windows": [{"window": c[1], "j": c[2], "dl": c[3], "clean_loss": c[4], "i0": c[5],
                                                       "at_or_after_i0": c[2] >= c[5]} for c in ch[:5]],
                "carrier_zero_top5_record": e["carrier_zero_top5"] if q == 0 else None}
    dev = [abs(p["dL"] - p["pred_bf16"]) for _, p, _ in allpairs]
    car = [k for k, (_, p, _) in enumerate(allpairs) if p["rank"] == 0]
    out["smallest_all"] = detail(*allpairs[int(np.argmin(dev))])
    out["largest_all"] = detail(*allpairs[int(np.argmax(dev))])
    out["smallest_carrier"] = detail(*allpairs[min(car, key=lambda k: dev[k])])
    out["largest_carrier"] = detail(*allpairs[max(car, key=lambda k: dev[k])])
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:9000])


if __name__ == "__main__":
    main()
