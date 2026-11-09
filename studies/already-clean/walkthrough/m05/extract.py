"""m05: a birth, the rule. One group's positions read at the sixteen snapshots, the median against the side's own
99.9th percentile, the crossing, and the birth; then the ratio to the threshold for all 153 passing births (births
are gradual).

Reads only banked files in this repository (no network, no GPU) and writes data.js next to this file as
window.<ID>_DATA = {...}. Prints `path sha256` for every input.

    python3 extract.py
"""
import datetime
import hashlib
import json
import csv
import re
import subprocess

import numpy as np
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


EXAMPLES = "studies/next/10_recount2/out/06_examples.json"
BIRTHS_ALL = "studies/next/10_recount2/out/births_all.csv"
BIRTHS = "studies/next/m2/births.json"
ONSETS = "studies/next/n1/onsets/onsets.json"
ASSETS = "studies/next/00_ASSETS.md"
N1_SPEC = "studies/next/07_PICKS/N1_SPEC.md"
EXAMPLE = "beyond_chance_late"


def birth(present):
    """First snapshot present at t and at t+1; a group first present at the last snapshot is born there."""
    for t, p in enumerate(present):
        if p and (t == len(present) - 1 or present[t + 1]):
            return t
    return None


def main():
    src(EXAMPLES)  # first, so the data link opens the examples file (head, 2026-11-08)
    steps = [int(re.search(r"`stage1-step(\d+)`", s)[1]) for s in src(ASSETS).read_text().splitlines()
             if re.match(r"\| T_\d+ \| `stage1-step", s)]
    ex = json.loads(src(EXAMPLES).read_text())[EXAMPLE]
    a_plus = np.array(ex["a_plus_by_t"])                         # [16, size]
    med = np.median(a_plus, axis=1)
    q = np.array(ex["q999_by_t"])
    present = [bool(m > 0 and m >= qq) for m, qq in zip(med, q)]
    b = birth(present)
    med_diff = float(np.max(np.abs(np.round(med, 4) - np.array(ex["median_by_t"]))))
    print(f"CHECK median recomputed from the banked activations: largest difference from the banked median {med_diff}")
    assert present == ex["present_pct"], (present, ex["present_pct"])
    with open(src(BIRTHS_ALL)) as f:
        row = next(r for r in csv.DictReader(f) if r["pole"] == ex["pole"] and r["sample"] == "0" and int(r["c"]) == ex["c"])
    bj = json.loads(src(BIRTHS).read_text())
    rec = next(x for x in bj["births"] if x["id"] == f"{ex['pole']}:c{ex['c']}")
    others = [o["birth"] for o in ex["other_clusters_of_pole"] if o["c"] != ex["c"]]
    example = {
        "id": ex["pole"], "side": "positive" if ex["pole"].endswith("+") else "negative", "group": ex["c"],
        "group_name": ex["name"], "group_size": ex["size"], "ranks_in_fifty": ex["ranks"],
        "positions": [{"rank": p["rank"], "token": p["token"], "before_tail": p["before_tail"], "document": p["doc"],
                       "position_in_document": p["pos"], "corpus_token": p["tok"]} for p in ex["positions_T15"]],
        "activation_own_direction_by_snapshot": ex["a_plus_by_t"],
        "median_by_snapshot": ex["median_by_t"],
        "median_recomputed_by_snapshot": [round(float(m), 6) for m in med],
        "median_check": compare("median, largest |recomputed - banked| over the 16 snapshots", round(med_diff, 6), 0.0,
                                f"{EXAMPLES} {EXAMPLE}.median_by_t", 1.5e-4),
        "threshold_by_snapshot": ex["q999_by_t"], "present_by_snapshot": present,
        "birth_snapshot": compare("birth", b, ex["birth_banked"], f"{EXAMPLES} {EXAMPLE}.birth_banked", 0),
        "birth_by_raw_ratio_reading": int(row["birth_raw"]),
        "birth_in_passing_list": rec["k"], "share_random_born_same_or_later": rec["p_late"], "half": "rule_setting" if rec["half"] == "freeze" else "held_out",
        "other_groups_of_side": ex["other_clusters_of_pole"],
        "late_by_both_readings": bool(b > min(others) and int(row["birth_raw"]) > min(others)),
    }
    assert rec["k"] == b == int(row["birth_pct"])
    print(f"PLAN rule -> example.birth_snapshot = {b} (step {steps[b]:,})")

    # all 153 passing births: the ratio of the median to the threshold at every snapshot (internal r_all)
    on = json.loads(src(ONSETS).read_text())
    assert on["units_sha256"] and len(on["units"]) == 442
    real = [u for u in on["units"] if u["kind"] == "real"]
    assert len(real) == 153 == bj["n_births"]
    ids = {x["id"] for x in bj["births"]}
    assert {u["unit"] for u in real} == ids
    pre = [u["r"]["pre"] for u in real]
    not_conc = sum(not u["concentrated"] for u in real)
    spec = src(N1_SPEC).read_text()
    banked_n = int(re.search(r"In (\d+) of 153 births the dated bracket holds less of the rise", spec)[1])
    banked_r = float(re.search(r"one checkpoint before birth is ([\d.]+) \(", spec)[1])
    gradual = {
        "births": 153,
        "dated_step_holds_less_rise_than_two_before": compare("not concentrated", not_conc, banked_n,
                                                              f"{N1_SPEC} §0.6 item 2", 0),
        "median_ratio_one_snapshot_before": compare("median r_pre", round(float(np.median(pre)), 3), banked_r,
                                                    f"{N1_SPEC} §0.6 item 2", 0.005),
        "median_ratio_one_snapshot_before_by_half": {
            "rule_setting": on["summary"]["real_freeze"]["r_pre_quantiles_10_25_50_75_90"][2],
            "held_out": on["summary"]["real_test"]["r_pre_quantiles_10_25_50_75_90"][2]},
        "per_birth": [{"birth": u["unit"], "half": "rule_setting" if u["half"] == "freeze" else "held_out",
                       "birth_snapshot": u["k"], "ratio_by_snapshot": u["r_all"],
                       "dated_step_holds_more_rise": u["concentrated"]} for u in real],
    }
    print(f"PLAN 93 of 153 -> gradual.dated_step_holds_less_rise_than_two_before.recomputed = {not_conc}; "
          f"PLAN 0.84 -> gradual.median_ratio_one_snapshot_before.recomputed = {np.median(pre):.4f}")
    data = {"snapshot_steps": steps, "example": example, "gradual": gradual}
    notes = {
        "example": "One passing birth from studies/next/10_recount2/out/06_examples.json (the recount's example dated later "
                   "than chance; internal id L10N9794+ cluster 3). Positions are fixed at the end of training; "
                   "activation_own_direction_by_snapshot[t][j] = max(sign x a, 0) at position j and snapshot t, in raw activation units.",
        "threshold_by_snapshot": "The side's own 99.9th percentile of its own-direction activation at each snapshot (internal q999).",
        "share_random_born_same_or_later": "Share of 2,000 random groups of the same size from the same side born at the same snapshot or later (internal p_late, births.json); the birth passes at 0.05 or below.",
        "present_by_snapshot": "Recomputed here: present when the median is above 0 and at least the threshold; asserted equal to "
                               "the banked present_pct.",
        "birth_snapshot": "Recomputed here with the rule (first snapshot present at t and at t+1), against the banked birth.",
        "birth_by_raw_ratio_reading": "The second reading (median at least 0.3 x the side's max |a| at t) from births_all.csv "
                                      "birth_raw; for the primary count both readings must agree that the group is born after "
                                      "the side's first-born group (late_by_both_readings).",
        "median_check": "The banked activations and medians are rounded to 4 decimals, so the median recomputed from the "
                        "rounded activations can differ from the banked median by up to 1e-4.",
        "gradual": "From studies/next/n1/onsets/onsets.json (the pre-check of the weight-change study): ratio_by_snapshot is the "
                   "median own-direction activation over the group's positions divided by the side's threshold, at each "
                   "snapshot (internal r_all); dated_step_holds_more_rise is internal 'concentrated' (log rise in the dated step "
                   "larger than over the two steps before).",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
