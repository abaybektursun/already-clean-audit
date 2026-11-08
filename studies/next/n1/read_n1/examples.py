"""N1 independent read, step 6: the test half's raw examples (samples/raw_examples_test.jsonl). For each of the ten
example births: real minus dt on e_in (the label's direction is real below dt); for one birth in the label's direction
(the largest gap) and the one against it: unit, k, e_in, e_out, the twins' values, this read's recount of the same
numbers from stats.npz, and the first exemplars (N12's banked raw examples of the same births).

    python3 -I examples.py --raw S3/samples/raw_examples_test.jsonl --units-recount studies/next/n1/read_n1/units_recount.json \
        --n12-raw studies/next/n12/banks/score-02/n12/samples/raw_examples_test.jsonl --out studies/next/n1/read_n1/examples.json
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    for a in ("raw", "units-recount", "n12-raw", "out"):
        ap.add_argument(f"--{a}", required=True)
    a = ap.parse_args()
    ex = [json.loads(x) for x in pathlib.Path(a.raw).read_text().split("\n") if x.strip()]
    U = json.loads(pathlib.Path(a.units_recount).read_text())
    nr = {json.loads(x)["birth"]: json.loads(x) for x in pathlib.Path(a.n12_raw).read_text().split("\n") if x.strip()}
    rows = []
    for e in ex:
        row = {"birth": e["birth"], "k": e["real"]["k"]}
        for kind in ("real", "dt", "pb"):
            if kind not in e:
                continue
            r = e[kind]
            m = U[r["unit"]]
            row[kind] = {"unit": r["unit"], "pole": r["pole"], "layer": r["layer"], "neuron": r["neuron"],
                         "e_in": r["in"]["e_window"], "e_out": r["out"]["e_window"], "e_k_in": r["in"]["e_k"],
                         "e_by_bracket_in": r["in"]["e_by_bracket"],
                         "r_in_window": [r["in"]["r"][j - 1] for j in (r["k"] - 2, r["k"] - 1, r["k"])],
                         "cos_in_k": r["in"]["cos"][r["k"] - 1], "cos_out_k": r["out"]["cos"][r["k"] - 1],
                         "recount_e_in": m["in"]["e_window"], "recount_e_out": m["out"]["e_window"],
                         "abs_diff_record_vs_recount": max(abs(r["in"]["e_window"] - m["in"]["e_window"]),
                                                           abs(r["out"]["e_window"] - m["out"]["e_window"]))}
        if "dt" in row:
            row["real_minus_dt_in"] = row["real"]["e_in"] - row["dt"]["e_in"]
            row["real_minus_dt_out"] = row["real"]["e_out"] - row["dt"]["e_out"]
            row["direction_in"] = "with the label (real below dt)" if row["real_minus_dt_in"] < 0 else "against the label"
        n12 = nr.get(e["birth"], {})
        row["exemplars"] = {kind: n12[kind]["exemplars"][:2] for kind in ("real", "dt") if kind in n12}
        rows.append(row)
    withl = [r for r in rows if r.get("real_minus_dt_in", 0) < 0]
    agst = [r for r in rows if r.get("real_minus_dt_in", 0) > 0]
    out = {"n_examples": len(rows), "n_with_label_in": len(withl), "n_against_label_in": len(agst),
           "chosen_with_label": min(withl, key=lambda r: r["real_minus_dt_in"])["birth"],
           "chosen_against_label": max(agst, key=lambda r: r["real_minus_dt_in"])["birth"],
           "max_abs_diff_record_vs_recount": max(r[k]["abs_diff_record_vs_recount"] for r in rows for k in ("real", "dt", "pb") if k in r),
           "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for r in rows:
        print(r["birth"], r["k"], round(r["real"]["e_in"], 3), round(r["real"]["e_out"], 3),
              {k: (round(r[k]["e_in"], 3), round(r[k]["e_out"], 3)) for k in ("dt", "pb") if k in r}, r.get("direction_in"))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
