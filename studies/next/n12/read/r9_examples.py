"""The raw examples (banks/score-02/n12/samples/raw_examples_{freeze,test}.jsonl): (1) each example's record equals
the unit record and this reader's recount; (2) d recounted from the decoded exemplars (the distinct [[token]] strings)
for real, pb and dt units of the 20 example births; (3) the two births read in READ_n12.md section 7, printed."""
import json, pathlib, re

here = pathlib.Path(__file__).resolve().parent
n12 = here.parent
V = json.load(open(here / "r1_units.json"))["units"]
names = {x["id"]: x["name"] for x in json.load(open(n12.parent / "m2" / "births.json"))["births"]}
out = {"d_check": [], "record_check": []}
for h in ("freeze", "test"):
    for line in open(n12 / "banks" / "score-02" / "n12" / "samples" / f"raw_examples_{h}.jsonl"):
        e = json.loads(line)
        for kind in ("real", "pb", "dt"):
            if kind not in e:
                continue
            r = e[kind]
            v = V[r["unit"]]
            allm = [re.findall(r"\[\[(.*?)\]\]", x, re.S) for x in r["exemplars"]]
            toks = [m[-1] for m in allm]  # the text's own [[...]] (wiki links) can precede the marker; take the last
            out["d_check"].append({"unit": r["unit"], "size": v["size"], "n_exemplars": len(toks),
                                   "n_exemplars_with_2_bracket_pairs": sum(len(m) > 1 for m in allm),
                                   "d_from_text": len(set(toks)), "d_record": r["d"], "d_real": r["d_real"]})
            out["record_check"].append({"unit": r["unit"], "equal": all(r["cf"][k] == v[k] for k in
                                                                        ("r0", "r1", "r_c", "r_o"))
                                        and abs(r["drop"] - v["drop"]) < 1e-12})
out["n_d_equal"] = sum(x["d_from_text"] == x["d_record"] for x in out["d_check"])
out["n_d"] = len(out["d_check"])
out["n_dt_d_equals_real"] = sum(x["d_record"] == x["d_real"] for x in out["d_check"] if x["unit"].endswith(":dt"))
out["n_records_equal"] = sum(x["equal"] for x in out["record_check"])
(here / "r9_examples.json").write_text(json.dumps(out, indent=1))
print({k: out[k] for k in ("n_d", "n_d_equal", "n_dt_d_equals_real", "n_records_equal")})
print([x for x in out["d_check"] if x["d_from_text"] != x["d_record"]])
