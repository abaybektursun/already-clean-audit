"""N1 independent read, step 5 (b): are carriers and their dt and pb twins matched on layer, k, group size and number
of distinct tokens (units.json), and does a twin's neuron coincide with its carrier's.

    python3 -I matching.py --units studies/next/n1/units.json --out studies/next/n1/read_n1/matching.json
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    U = json.loads(pathlib.Path(a.units).read_text())["units"]
    out = {}
    for half in ("test", "freeze"):
        rb = {u["birth"]: u for u in U if u["half"] == half and u["kind"] == "real"}
        for kind in ("dt", "pb"):
            tw = {u["birth"]: u for u in U if u["half"] == half and u["kind"] == kind}
            ids = sorted(b for b in rb if b in tw)
            out[f"{half}_{kind}"] = {"n_pairs": len(ids), "n_real_without_twin": len(rb) - len(ids),
                                     **{f"same_{f}": sum(rb[b][f] == tw[b][f] for b in ids) for f in ("layer", "k", "size", "d")},
                                     "same_neuron": sum(rb[b]["neuron"] == tw[b]["neuron"] for b in ids),
                                     "same_sign": sum(rb[b]["sign"] == tw[b]["sign"] for b in ids)}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
