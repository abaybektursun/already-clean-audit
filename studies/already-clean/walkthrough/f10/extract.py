"""f10: naming the carrier one snapshot early. Per passing birth the carrier's rank among its layer's 22,016 sides one
snapshot before the birth, the rank of a random group of another side's contexts (same size and birth snapshot) on its
own side, and the same with distinct tokens also matched; counts in the top ten; raw examples with text.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # walkthrough/<picture> -> already-clean -> studies -> repository root
ID = HERE.name.upper()
MAX_BYTES = 300_000
SOURCES = []


def src(rel):
    """Open a repository file by its path from the root, record its sha256 once, print `path sha256`."""
    p = ROOT / rel
    b = p.read_bytes()
    assert not b.startswith(b"version https://git-lfs"), f"{rel} is an LFS pointer: git lfs pull --include={rel}"
    if rel not in [s["path"] for s in SOURCES]:
        h = hashlib.sha256(b).hexdigest()
        SOURCES.append({"path": rel, "sha256": h})
        print(rel, h)
    return p


def compare(name, recomputed, banked, where, tol):
    """Store a recomputed summary beside the banked one; print both and whether they differ beyond rounding."""
    same = abs(recomputed - banked) <= tol
    print(f"CHECK {name}: recomputed {recomputed} banked {banked} ({where}) {'same' if same else 'DIFFERS'}")
    return {"recomputed": recomputed, "banked": banked, "banked_at": where, "same_within": tol, "same": same}


def write(data, notes):
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    data["meta"] = {"commit": commit, "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


M2 = "studies/next/m2/"
RANKS = [M2 + "banks/score-06/m2/ranks_test.jsonl", M2 + "banks/score-06/m2/ranks_freeze.jsonl"]
N12_RANKS = ["studies/next/n12/banks/score-02/n12/ranks_test.jsonl", "studies/next/n12/banks/score-02/n12/ranks_freeze.jsonl"]
UNITS = "studies/next/n12/units.json"
BIRTHS = M2 + "births.json"
VERDICT = M2 + "verdict_m2.json"
N12_VERDICT = "studies/next/n12/verdict_n12.json"
SHARE = M2 + "read/r5b_token_share.json"
RAW = M2 + "banks/score-06/m2/samples/raw_examples_test.jsonl"
SIDES_PER_LAYER = 22016
TOP = 10
SHARE_CUT = 0.67                 # r5b_token_share.py: distinct tokens / size below 0.67 = repeated-token birth


def jsonl(rel):
    return [json.loads(s) for s in src(rel).read_text().splitlines() if s.strip()]


def side(r):
    return {"side_id": r["carrier"], "rank": r["rank"]["B"], "score": r["value"]["B"]}


def hits(xs):
    return sum(x["rank"] <= TOP for x in xs)


def main():
    pre = {r["unit"]: r for p in RANKS for r in jsonl(p) if r["role"] == "pre"}
    dt = {r["unit"]: r for p in N12_RANKS for r in jsonl(p) if r["role"] == "pre" and r["kind"] == "dt"}
    n12_real = {r["unit"]: r for p in N12_RANKS for r in jsonl(p) if r["role"] == "pre" and r["kind"] == "real"}
    units = {u["unit"]: u for u in json.loads(src(UNITS).read_text())["units"]}
    bj = json.loads(src(BIRTHS).read_text())
    births = []
    for b in bj["births"]:
        r = pre[b["id"]]
        assert r["t"] == b["k"] - 1 and r["layer"] == b["layer"]
        assert n12_real[b["id"]]["rank"]["B"] == r["rank"]["B"], "the two studies' carrier ranks differ"
        cg, cd = pre.get(b["id"] + ":pb"), dt.get(b["id"] + ":dt")
        births.append({
            "birth": b["id"], "half": "rule_setting" if b["half"] == "freeze" else "held_out", "layer": b["layer"],
            "birth_snapshot": b["k"], "scored_at_snapshot": r["t"], "group_size": b["size"],
            "distinct_tokens": units[b["id"]]["d"], "group_name": b["name"],
            "carrier": side(r),
            "comparison_group": side(cg) if cg else None,
            "comparison_group_distinct_tokens": (side(cd) | {"distinct_tokens": units[b["id"] + ":dt"]["d"]}) if cd else None})
    v = json.loads(src(VERDICT).read_text())
    nv = json.loads(src(N12_VERDICT).read_text())
    summary = {}
    for half in ("held_out", "rule_setting"):
        R = [x for x in births if x["half"] == half]
        car = [x["carrier"] for x in R]
        cg = [x["comparison_group"] for x in R if x["comparison_group"]]
        cd = [x["comparison_group_distinct_tokens"] for x in R if x["comparison_group_distinct_tokens"]]
        summary[half] = {"births": len(R), "carrier_in_top_ten": hits(car),
                         "with_comparison_group": len(cg), "comparison_group_in_top_ten": hits(cg),
                         "with_distinct_token_comparison_group": len(cd),
                         "distinct_token_comparison_group_in_top_ten": hits(cd)}
    h = summary["held_out"]
    t = v["test"]
    summary["held_out"]["checks"] = {
        "carrier_in_top_ten": compare("carrier top ten", h["carrier_in_top_ten"], v["labels"]["H_B"]["hits"],
                                      f"{VERDICT} labels.H_B.hits", 0),
        "carrier_share": compare("carrier share", round(h["carrier_in_top_ten"] / h["births"], 4),
                                 round(v["labels"]["H_B"]["p"], 4), f"{VERDICT} labels.H_B.p", 1e-4),
        "comparison_group_in_top_ten": compare("comparison top ten", h["comparison_group_in_top_ten"], t["pb_pre_B"][3],
                                               f"{VERDICT} test.pb_pre_B[3]", 0),
        "with_comparison_group": compare("births with a comparison group", h["with_comparison_group"], t["n_pb"],
                                         f"{VERDICT} test.n_pb", 0),
        "distinct_token_comparison_share": compare(
            "distinct-token comparison share", round(h["distinct_token_comparison_group_in_top_ten"] / h["with_distinct_token_comparison_group"], 4),
            round(nv["test"]["hits"]["dt"]["pre"][0], 4), f"{N12_VERDICT} test.hits.dt.pre[0]", 1e-4),
    }
    banked = {
        "carrier_share_range": [v["labels"]["H_B"]["lo"], v["labels"]["H_B"]["hi"]],
        "comparison_group_share": t["pb_pre_B"][:3],
        "carrier_minus_comparison_group": {k: t["pair_real_pb_B"][k] for k in ("n", "diff", "lo", "hi", "x_only", "y_only")},
        "carrier_minus_distinct_token_comparison_group": nv["test"]["pair_hit_real_dt"],
        "chance_level": TOP / SIDES_PER_LAYER, "threshold_set_in_advance": v["bars"]["B"],
        "carrier_share_over_chance": round(v["labels"]["H_B"]["p"] / (TOP / SIDES_PER_LAYER), 1),
    }
    print(f"PLAN 42 of 71 -> summary.held_out.carrier_in_top_ten = {h['carrier_in_top_ten']} of {h['births']}; "
          f"PLAN 53 of 70 -> comparison_group_in_top_ten = {h['comparison_group_in_top_ten']} of {h['with_comparison_group']}; "
          f"PLAN 41 of 67 -> {h['distinct_token_comparison_group_in_top_ten']} of {h['with_distinct_token_comparison_group']}; "
          f"PLAN -0.171 (-0.286 to -0.057) -> banked.carrier_minus_comparison_group = {banked['carrier_minus_comparison_group']}; "
          f"PLAN -0.030 (-0.149, +0.090) -> banked.carrier_minus_distinct_token_comparison_group; "
          f"PLAN about 1,300 times chance -> banked.carrier_share_over_chance = {banked['carrier_share_over_chance']}; "
          f"PLAN 0.00045, 0.0122 -> banked.chance_level, threshold_set_in_advance")

    # post hoc split by the share of distinct tokens, both halves, births with a comparison group (the reader's finding)
    sh = json.loads(src(SHARE).read_text())["both"]
    split = {}
    for name, key, pick in (("repeated_token", "real_share_lt_0.67", lambda x: x["distinct_tokens"] / x["group_size"] < SHARE_CUT),
                            ("varied_token", "real_share_ge_0.67", lambda x: x["distinct_tokens"] / x["group_size"] >= SHARE_CUT)):
        R = [x for x in births if x["comparison_group"] and pick(x)]
        split[name] = {
            "births": compare(f"{name} births", len(R), sh[key]["n"], f"{SHARE} both.{key}.n", 0),
            "carrier_in_top_ten": compare(f"{name} carrier", hits([x["carrier"] for x in R]), sh[key]["real_hits"],
                                          f"{SHARE} both.{key}.real_hits", 0),
            "comparison_group_in_top_ten": compare(f"{name} comparison", hits([x["comparison_group"] for x in R]),
                                                   sh[key]["twin_hits"], f"{SHARE} both.{key}.twin_hits", 0),
            "difference": sh[key]["diff"], "difference_range": sh[key]["diff_ci"]}
    print(f"PLAN 22 against 53 of 78; 57 against 58 of 71 -> post_hoc_split")

    raw = []
    for e in jsonl(RAW):
        raw.append({"birth": e["unit"], "scored_at_snapshot": e["t"], "carrier_side": e["carrier"],
                    "carrier_rank": e["rank"]["B"], "carrier_score": e["value"]["B"],
                    "group_contexts": [{"rank_in_fifty": x["rank"], "corpus_token": x["tok"], "text": x["text"]} for x in e["exemplars"]],
                    "top_ten": [{"side_id": s["pole"], "score": s["score"],
                                 "strongest_contexts": [{"rank": c["rank"], "a": c["a"], "text": c["text"]} for c in s["top3"]]}
                                for s in e["top10"]["B"]]})
    data = {"sides_per_layer": SIDES_PER_LAYER, "top": TOP, "births": births, "summary": summary, "banked": banked,
            "post_hoc_split": split, "raw_examples": raw}
    notes = {
        "births": "All 153 passing births (71 held out, 82 in the half that set the rules), in births.json order. rank is the "
                  "number of the layer's 22,016 sides whose score B is at least the side's own (ties count against it), scored "
                  "one snapshot before the birth (scored_at_snapshot = birth_snapshot - 1); score is B itself.",
        "comparison_group": "A random group of another side's own fifty (echo excluded), of the birth's size and layer, dated to "
                            "the same snapshot by the same rule; rank and score are of that group's own side, under the same "
                            "score (internal pb twin, studies/next/m2 score-06). Null where no such group was found.",
        "comparison_group_distinct_tokens": "The same, also matched on distinct_tokens, the number of distinct decoded tokens "
                                            "at the group's positions (internal dt twin, studies/next/n12 score-02).",
        "summary": "Counts recomputed from the rows (in the top ten = rank <= 10); checks compare them with the banked verdicts.",
        "banked": "From verdict_m2.json (labels.H_B, test.pb_pre_B, test.pair_real_pb_B, bars.B) and verdict_n12.json "
                  "(test.pair_hit_real_dt); ranges are the studies' 95 % intervals; carrier_share_over_chance = the held-out "
                  "share divided by chance_level = 10/22,016; threshold_set_in_advance is the matched-side level the share had "
                  "to exceed (internal bar).",
        "post_hoc_split": "The reader's post hoc split, both halves, births with a comparison group: repeated_token = distinct "
                          "tokens / group size below 0.67, varied_token otherwise (studies/next/m2/read/r5b_token_share.json); "
                          "marked post hoc in the record.",
        "raw_examples": "The 10 banked raw examples of the held-out half (raw_examples_test.jsonl), verbatim: the group's contexts "
                        "with the marked token between [[ and ]] (160 characters before), and the layer's ten highest-scoring "
                        "sides one snapshot before the birth with each side's own three strongest contexts and activations a at "
                        "that snapshot.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
