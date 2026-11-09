"""m07: ranking a side one snapshot early. For each held-out passing birth: the score B and rank of the carrier among
its layer's 22,016 sides one snapshot before the birth, the layer's ten highest-scoring sides, and the same for the
comparison group's own side under the same score.

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import ast
import re
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
    data["meta"] = {"commit": commit, "link": "studies/next/m2/banks/score-06/m2/ranks_test.jsonl",  # the file the data link opens
                    "sources": SOURCES,
                    "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "notes": notes}
    out = HERE / "data.js"
    out.write_text(f"window.{ID}_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size <= MAX_BYTES, f"{out} is {size:,} bytes"
    print(f"wrote {out.relative_to(ROOT)}: {size:,} bytes")


RANKS = "studies/next/m2/banks/score-06/m2/ranks_test.jsonl"
BIRTHS = "studies/next/m2/births.json"
VERDICT = "studies/next/m2/verdict_m2.json"
M2_CODE = "infra/pg/m2.py"
M2_SPEC = "studies/next/07_PICKS/M2_SPEC.md"
SIDES_PER_LAYER = 22016


def ranked(r):
    return {"side_id": r["carrier"], "rank": r["rank"]["B"], "score": r["value"]["B"],
            "top_ten": [{"side_id": p, "score": s} for p, s in r["top10"]["B"]]}


def main():
    tree = ast.parse(src(M2_CODE).read_text())
    hist = next(n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                and [e.id for e in n.targets[0].elts] == ["HIST_LO", "HIST_HI", "HIST_B"])
    lo, hi, nb = ast.literal_eval(hist.value)
    rows = [json.loads(s) for s in src(RANKS).read_text().splitlines() if s.strip()]
    pre = {r["unit"]: r for r in rows if r["role"] == "pre"}
    bj = json.loads(src(BIRTHS).read_text())
    v = json.loads(src(VERDICT).read_text())
    spec_chance = float(re.search(r"\*\*Chance\*\* = 10 / 22,016 = (\d+\.\d+)", src(M2_SPEC).read_text())[1])
    births = []
    for b in bj["births"]:
        if b["half"] != "test":
            continue
        r = pre[b["id"]]
        assert r["t"] == b["k"] - 1
        assert all(r["top10"]["B"][i][1] >= r["top10"]["B"][i + 1][1] for i in range(9))
        cg = pre.get(b["id"] + ":pb")
        births.append({"birth": b["id"], "layer": b["layer"], "birth_snapshot": b["k"], "scored_at_snapshot": r["t"],
                       "group_size": b["size"], "group_name": b["name"],
                       "carrier": ranked(r), "comparison_group": ranked(cg) if cg else None})
    n_top = sum(x["carrier"]["rank"] <= 10 for x in births)
    n_cg = sum(1 for x in births if x["comparison_group"])
    n_cg_top = sum(x["comparison_group"]["rank"] <= 10 for x in births if x["comparison_group"])
    data = {
        "sides_per_layer": SIDES_PER_LAYER, "top": 10,
        "chance_level": compare("chance", round(10 / SIDES_PER_LAYER, 5), spec_chance, f"{M2_SPEC} §0.5 '**Chance** = ...'", 5e-6),
        "threshold_set_in_advance": v["bars"]["B"],
        "histogram": {"bins": nb, "log10_low": lo, "log10_high": hi},
        "held_out_births": len(births),
        "carrier_in_top_ten": compare("carrier top ten", n_top, v["labels"]["H_B"]["hits"], f"{VERDICT} labels.H_B.hits", 0),
        "with_comparison_group": compare("with comparison group", n_cg, v["test"]["n_pb"], f"{VERDICT} test.n_pb", 0),
        "comparison_group_in_top_ten": compare("comparison top ten", n_cg_top, v["test"]["pb_pre_B"][3], f"{VERDICT} test.pb_pre_B[3]", 0),
        "births": births,
        "not_banked": "Each layer's full ranking (all 22,016 scores) is not in the repository; the banks hold each scored "
                      "side's own rank and score and the layer's ten highest scores.",
    }
    print(f"PLAN 0.00045 -> chance_level; PLAN 0.0122 -> threshold_set_in_advance = {v['bars']['B']}; "
          f"PLAN 70 of 71 have a comparison group -> with_comparison_group.recomputed = {n_cg}")
    notes = {
        "births": "The 71 held-out passing births in births.json order. For a birth at snapshot k, v = the median over the "
                  "group's positions of a side's own-direction activation at k-1; score B of a side = the share of its own-sign "
                  "tokens at k-1 with |a| below v, read from the side's 400-bin histogram (bins below v's bin whole, v's bin by "
                  "the log-linear share below v; infra/pg/m2.py percentile); rank = the number of the layer's 22,016 sides with "
                  "B at least the side's own (ties count against it).",
        "carrier": "The side that carries the group at the end of training; top_ten lists the layer's ten highest B under this "
                   "group's positions, highest first.",
        "comparison_group": "A random group of another side's own fifty (echo excluded), of the birth's size and layer, dated to "
                            "the same snapshot by the same rule, taken in seeded order up to 2,000 draws (internal pb twin); "
                            "side_id is that group's own side, ranked under B with that group's positions; null if none was found.",
        "histogram": "Each side's |a| histogram has 400 bins evenly spaced in log10 |a| from -6 to 4 (infra/pg/m2.py).",
        "threshold_set_in_advance": "The share in the top ten that the carrier had to exceed, set from sides matched on size "
                                    "(internal bar, verdict_m2.json bars.B).",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
