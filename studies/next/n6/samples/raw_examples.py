"""N6's ten raw examples, the zero-GPU part (spec section 10): the five freeze pairs n6.raw_example_pairs picks, both members, from the
banks: pole, partner, ages, the matched quantities, the pole's three strongest T_15 contexts. The Frobenius norm c of each edit, the
corpus loss change and the five tokens whose loss changed most need the forward; they are PENDING until L1 writes
samples/raw_examples.jsonl.

    PYTHONPATH=. python3 studies/next/n6/samples/raw_examples.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
from infra.pg import n6  # noqa: E402

HERE = pathlib.Path("studies/next/n6/samples")


def main():
    pairs = json.loads(pathlib.Path("studies/next/n6/pairs.json").read_text())
    ctx = {json.loads(x)["pole"]: json.loads(x) for x in open(n6.FILES["ctx15"])}
    by = {p["pair"]: p for p in pairs["pairs"]}
    rows = []
    for pid in n6.raw_example_pairs(pairs):
        p = by[pid]
        for m, o in (("early", "late"), ("late", "early")):
            q = pairs["poles"][p[m]]
            c = ctx[p[m]]
            rows.append({"pair": pid, "member": m, "pole": p[m], "partner": p[o], "layer": p["layer"], "half": p["half"],
                         "age": q["age"], "partner_age": pairs["poles"][p[o]]["age"], "adjusted_age": q["adj"],
                         "max_abs": q["max_abs"], "density": q["density"], "q999": q["q999"],
                         "top3_T15": [{"rank": k["rank"], "a": k["a"], "before_tail": k["before"][-90:], "token": k["token"],
                                       "after_head": k["after"][:40]} for k in c["topk"][:3]],
                         "frobenius_norm_c": "PENDING (L1)", "dL_nats_per_token": "PENDING (L1)", "top5_tokens": "PENDING (L1)"})
    (HERE / "raw_examples_zero_gpu.json").write_text(json.dumps(rows, indent=1))
    lines = ["# N6 raw examples, zero-GPU part", "",
             "Written by `studies/next/n6/samples/raw_examples.py` from the banks (`pairs.json`, the T_15 `contexts.jsonl`). "
             "Age is the main cluster's birth index T_k (J1's recount). The edit size c, the corpus loss change and the five tokens "
             "whose loss changed most are PENDING: they need the forward, and L1 writes them to `samples/raw_examples.jsonl`.", ""]
    for r in rows:
        lines.append(f"## {r['pair']} {r['member']}: {r['pole']} (partner {r['partner']})")
        lines.append("")
        lines.append(f"layer {r['layer']}, {r['half']} half; age T_{r['age']} (partner T_{r['partner_age']}); adjusted age "
                     f"{r['adjusted_age']:+.2f}; max|a| {r['max_abs']:.4g}; density {r['density']:,}; q999 {r['q999']:.4g}")
        lines.append("")
        for t in r["top3_T15"]:
            lines.append(f"- rank {t['rank']}, a {t['a']:.4g}: …{json.dumps(t['before_tail'])[1:-1]} **[{json.dumps(t['token'])[1:-1]}]** "
                         f"{json.dumps(t['after_head'])[1:-1]}…")
        lines.append("")
        lines.append("Frobenius norm c: PENDING (L1). Corpus loss change: PENDING (L1). Five tokens: PENDING (L1).")
        lines.append("")
    (HERE / "raw_examples_zero_gpu.md").write_text("\n".join(lines))
    print("written", len(rows), "rows")


if __name__ == "__main__":
    main()
