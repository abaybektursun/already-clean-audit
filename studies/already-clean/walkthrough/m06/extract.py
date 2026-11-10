"""m06: births that beat chance, and the split. Per group the share of 2,000 random groups of the same size born at
the same snapshot or later (p_late), the 153 of 2,428 groups with p_late <= 0.05 against 22.4 expected, two examples
with their random groups' birth histogram, and the split of the 153 by unit index.

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


BIRTHS = "studies/next/m2/births.json"
PER_GROUP = "studies/next/10_recount2/out/05_per_cluster_R1t.csv"
CHANCE = "studies/next/10_recount2/out/05_chance.json"
EXAMPLES = "studies/next/10_recount2/out/06_examples.json"
SPEC = "studies/already-clean/SPEC_D1.md"   # §4.3: the training step of each of the sixteen checkpoints


def main():
    bj = json.loads(src(BIRTHS).read_text())
    ch = json.loads(src(CHANCE).read_text())
    r1t = ch["summary"]["R1t"]["all"]
    with open(src(PER_GROUP)) as f:
        rows = list(csv.DictReader(f))
    alpha = ch["alpha"]
    p_late = [float(r["p_late"]) for r in rows]
    passing = [r for r in rows if float(r["p_late"]) <= alpha]
    expected = sum(float(r["exp_late"]) for r in rows)
    counts = {
        "groups_tested": compare("groups tested", len(rows), bj["n_clusters_tested"], f"{BIRTHS} n_clusters_tested", 0),
        "passing": compare("passing births", len(passing), bj["n_births"], f"{BIRTHS} n_births", 0),
        "expected_at_chance": compare("expected at chance", round(expected, 1), r1t["late_expected"],
                                      f"{CHANCE} summary.R1t.all.late_expected", 0.05),
        "earlier_than_chance": r1t["early_beat"], "earlier_expected_at_chance": r1t["early_expected"],
        "calibration_passing": r1t["calibration_late_beat"],
        "naive_expected_alpha_times_groups": r1t["naive_expected_alpha_times_n"],
    }
    print(f"PLAN 153 of 2,428 -> counts.passing.recomputed = {len(passing)}, counts.groups_tested.recomputed = {len(rows)}; "
          f"PLAN 22.4 -> counts.expected_at_chance.recomputed = {expected:.2f}")
    ids = {f"{r['pole']}:c{r['c']}" for r in passing}
    births = bj["births"]
    assert ids == {b["id"] for b in births}, "the passing groups differ from births.json"

    # the split by unit index
    halves = {"rule_setting": 0, "held_out": 0}
    for b in births:
        h = "rule_setting" if b["neuron"] % 2 == 0 else "held_out"
        assert h == ("rule_setting" if b["half"] == "freeze" else "held_out"), b["id"]
        halves[h] += 1
    split = {"rule": "unit index even: the half that set the rules; odd: the held-out half",
             "rule_setting": compare("rule-setting half", halves["rule_setting"], bj["halves"]["freeze"], f"{BIRTHS} halves.freeze", 0),
             "held_out": compare("held-out half", halves["held_out"], bj["halves"]["test"], f"{BIRTHS} halves.test", 0),
             "by_birth_snapshot": {"rule_setting": bj["by_k"]["freeze"], "held_out": bj["by_k"]["test"]}}
    print(f"PLAN 82 / 71 -> split.rule_setting / split.held_out = {halves['rule_setting']} / {halves['held_out']}")

    ex = json.loads(src(EXAMPLES).read_text())
    by_id = {f"{r['pole']}:c{r['c']}": r for r in rows}
    examples = []
    for key in ("beyond_chance_late", "at_chance"):
        v = ex[key]
        r = by_id[f"{v['pole']}:c{v['c']}"]
        examples.append({
            "example": key, "id": v["pole"], "group": v["c"], "group_name": v["name"], "group_size": v["size"],
            "ranks_in_fifty": v["ranks"], "birth_snapshot": v["birth_banked"],
            "groups_of_side": [{"group": o["c"], "size": o["size"], "main": o["is_main"], "birth_snapshot": o["birth"]}
                               for o in v["other_clusters_of_pole"]],
            "random_group_births_fresh_draw": {int(k): n for k, n in v["ref_hist_fresh_draw"].items()},
            "share_random_born_same_or_later_fresh_draw": v["p_late_fresh"], "share_random_born_same_or_later": float(r["p_late"]),
            "random_group_median_birth": float(r["ref_median"]), "passes": float(r["p_late"]) <= alpha})
    steps = [int(r.split("|")[2].strip().replace(",", "")) for r in src(SPEC).read_text().splitlines()
             if re.match(r"\| T_\d+ \| [\d,]+ \|", r)]
    assert len(steps) == 16 and steps[-1] == 1413814, steps
    data = {
        "rule": {"random_groups_per_group": ch["reps"], "threshold": alpha, "seed": ch["seed"],
                 "sides_with_two_or_more_groups": ch["n_poles"]},
        "checkpoint_steps": steps,
        "counts": counts, "split": split, "examples": examples,
        "all_groups": {"share_random_born_same_or_later": p_late, "birth_snapshot": [int(r["real"]) for r in rows],
                       "size": [int(r["size"]) for r in rows], "main": [r["is_main"] == "True" for r in rows]},
        "passing_births": [{"birth": b["id"], "layer": b["layer"], "unit": b["neuron"], "group_name": b["name"],
                            "size": b["size"], "birth_snapshot": b["k"], "share_random_born_same_or_later": b["p_late"],
                            "half": "rule_setting" if b["half"] == "freeze" else "held_out"} for b in births],
        "not_banked": "The individual random groups (which of the fifty each holds) are not in the repository (the verifier's "
                      "random_groups.npz was moved to the laptop bank, studies/next/06b_verify_births/out/LARGE_FILES_MOVED.md); "
                      "only their birth histograms (two examples) and each group's share are banked.",
    }
    notes = {
        "rule": "For every group of a side with two or more groups, 2,000 random groups of the same size drawn from the side's "
                "own fifty (echo excluded; the side's top context keeps the main group's label, internal reference R1t) are "
                "dated by the same rule; share_random_born_same_or_later (internal p_late) = the share of them born at the "
                "same snapshot or later (never born counts as latest).",
        "counts": "passing = groups with share_random_born_same_or_later <= 0.05, recomputed from 05_per_cluster_R1t.csv; expected_at_chance = the sum over "
                  "groups of each group's chance probability of passing (column exp_late), recomputed; banked values from "
                  "births.json and 05_chance.json.",
        "examples": "Two groups from 06_examples.json: one born later than chance and one at chance. random_group_births_fresh_draw "
                    "maps a birth snapshot to the count of a fresh draw of 2,000 random groups (seed 7) born there, so "
                    "share_random_born_same_or_later_fresh_draw differs slightly from the main run's value.",
        "all_groups": "Parallel arrays over the 2,428 tested groups in the order of 05_per_cluster_R1t.csv.",
        "checkpoint_steps": "The training step of each of the sixteen checkpoints, indexed by the snapshot numbers used above "
                            "(SPEC_D1.md §4.3); the checkpoint at index k is the Hugging Face branch stage1-step<steps[k]>.",
        "split": "Each passing birth goes to the half that set the rules if its unit index is even, to the held-out half if odd "
                 "(births.json split, internal freeze / test); recomputed from the unit index.",
    }
    write(data, notes)


if __name__ == "__main__":
    main()
