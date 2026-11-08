"""N1 independent read, step 4: every number of the registered verdict (infra.pg.n1 verdict, run into the scratchpad)
against this read's own recount (verdict_recount.json).

    python3 -I compare.py --registered <scratchpad>/verdict_n1.json --mine studies/next/n1/read_n1/verdict_recount.json \
        --out studies/next/n1/read_n1/compare.json
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    for a in ("registered", "mine", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    V = json.loads(pathlib.Path(a.registered).read_text())
    M = json.loads(pathlib.Path(a.mine).read_text())
    T, MT = V["test"], M["test"]
    rows = []

    def add(name, reg, mine):
        rows.append({"q": name, "registered": reg, "mine": mine, "abs_diff": abs(reg - mine)})

    for tw in ("dt", "pb"):
        for s in ("in", "out", "gate", "up"):
            for pre, key in (("pair", f"pair_{s}_real_{tw}"), ("pair_k", f"pair_k_{s}_real_{tw}")):
                r = T[key] if pre == "pair" else T[f"pair_k_{s}_real_{tw}"]
                mk = MT.get(key)
                if mk is None:
                    continue
                add(f"{key}.diff", r["diff"], mk["mean"])
                add(f"{key}.lo", r["lo"], mk["pct_seed0"][0])
                add(f"{key}.hi", r["hi"], mk["pct_seed0"][1])
                add(f"{key}.x_only", r["x_only"], mk["n_pos"])
                add(f"{key}.y_only", r["y_only"], mk["n_neg"])
    for s in ("in", "out", "gate", "up"):
        for kind in ("real", "dt", "pb"):
            r, mk = T[f"{kind}_{s}"], MT[f"{kind}_{s}"]
            add(f"{kind}_{s}.mean", r["mean"], mk["mean"])
            add(f"{kind}_{s}.lo", r["lo"], mk["pct_seed0"][0])
            add(f"{kind}_{s}.hi", r["hi"], mk["pct_seed0"][1])
        for d in ("-2", "-1", "0", "1"):
            r, mk = T[f"profile_{s}"][d], MT[f"profile_{s}"][d]
            add(f"profile_{s}[{d}].mean", r["mean"], mk["mean"])
            add(f"profile_{s}[{d}].lo", r["lo"], mk["pct_seed0"][0])
            add(f"profile_{s}[{d}].hi", r["hi"], mk["pct_seed0"][1])
            add(f"profile_{s}[{d}].n", r["n"], mk["n"])
    for s in ("in", "out"):
        for q in ("mean", "q025", "q975"):
            add(f"floor_{s}.{q}", T[f"floor_{s}"][q], MT[f"floor_{s}"][q])
        add(f"ceiling_{s}", T[f"ceiling_{s}"], MT[f"ceiling_{s}"])
        add(f"labels.floor_{s}.bar", V["labels"][f"floor_{s}"]["bar"], M["labels"][s]["bar"])
    labels = {"registered": {"sel_in": V["labels"]["sel_in"]["label"], "sel_out": V["labels"]["sel_out"]["label"],
                             "floor_in": V["labels"]["floor_in"]["label"], "floor_out": V["labels"]["floor_out"]["label"],
                             "hypothesis": V["hypothesis"], "headline": V["headline"]},
              "mine": {"sel_in": M["labels"]["in"]["sel"], "sel_out": M["labels"]["out"]["sel"],
                       "floor_in": M["labels"]["in"]["floor"], "floor_out": M["labels"]["out"]["floor"],
                       "hypothesis": M["hypothesis"], "headline": M["headline"]}}
    out = {"n_compared": len(rows), "max_abs_diff": max(r["abs_diff"] for r in rows),
           "worst": max(rows, key=lambda r: r["abs_diff"])["q"], "labels": labels,
           "labels_equal": labels["registered"] == labels["mine"], "gates": V["gates"],
           "thresholds_sha256": V["thresholds_sha256"], "n": T["n"], "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        if r["q"].startswith(("pair_in_real_dt", "pair_out_real_dt", "floor", "ceiling", "labels", "real_in", "real_out")):
            print(r)


if __name__ == "__main__":
    main()
