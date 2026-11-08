"""Build data.js for the picture "When a neuron's groups are born".

Reads saved files only: the verdict's per-group record (w5/verdict/samples/q2_clusters.jsonl, one line per group,
read without its judgments), the verdict's metrics.json (q2.births, q2.p_u, q2.status_counts, q2.birth_order), and
the 16 saved training points of the W3 harvest (poles/exemplar_a.npz and samples/contexts.jsonl in each). Since
2026-11-05 it also reads the chance level of the dates: per group, the third recount's random same-size groups
(studies/next/10_recount2/out/05_per_cluster_R1t.csv, 05_chance.json, 04_b.json; 10_RECOUNT2.md §7-8), and per
training point, the verifier's mean count of random same-size groups (studies/next/06b_verify_births/out/
s03_histogram_tv.json; 06b_VERIFY_BIRTHS.md §3). Writes data.js next to this file, prints the checks, and rewrites
the checks block of README.md with them.

    python3 extract.py
"""
import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The local copy on the laptop; README.md names the S3 source of each.
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
GROUPS = BANK / "w5/verdict/samples/q2_clusters.jsonl"
VERDICT = BANK / "w5/verdict/metrics.json"
HARVEST = BANK / "w3/grid"
# The chance level of the dates (added 2026-11-05, after the retag of the birth-order claim; JOURNAL of that day).
STUDIES = HERE.parents[2]
RECOUNT = STUDIES / "next/10_recount2/out"          # third recount, random groups R1t (10_RECOUNT2.md §7)
VERIFY = STUDIES / "next/06b_verify_births/out"      # independent verifier, random groups set A (06b §3)
ALPHA = 0.05                                         # one-sided, stated before the recount ran (10_RECOUNT2.md §8)

MODEL = "OLMo-3 7B"
FINAL_STEP = 1413814
T_FIRST_TRAINED = 1          # SPEC_D1 §2.4: a birth at T_1 has only random init before it
# The neurons the picker names, in this order: id, the label the viewer sees, where the write-up names it.
NAMED = [
    ("L03N6372+", "the lithium neuron", "walkthrough lines 663-664; A_training_dynamics.md §2"),
    ("L03N7155+", "the got neuron", "walkthrough line 659; VISUALS_PLAN.md P6; A_training_dynamics.md §5"),
    ("L04N3936-", "the system neuron", "walkthrough line 659; A_training_dynamics.md §5"),
    ("L01N10284-", "the React neuron", "../judge NAMED; B_clutter.md Story 3; E_skeptic.md §3.4"),
    ("L06N7759+", "the laughter neuron", "../judge NAMED; B_clutter.md Story 4; E_skeptic.md §3.4"),
    ("L23N809-", "the Gul neuron", "../judge NAMED; A_training_dynamics.md §9"),
    ("L31N10276-", "the G neuron", "../judge NAMED; B_clutter.md Story 9"),
]
N_RANDOM, SEED = 33, 0
# From the write-up; printed beside what this run computes.
EXPECTED_GOT_STEPS = [5000, 20000, 35000, 100000]      # A_training_dynamics.md §5, VISUALS_PLAN.md P6
EXPECTED_DROP_AFTER_BIRTH = 44                          # E_skeptic.md §3.2
BEFORE_CHARS, AFTER_CHARS = 64, 28
MAX_BYTES = 2_000_000
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


def birth(present):
    """SPEC_D1 §2.4, a port of birth() in infra/pg/neurons_verdict.py: the first point at which the group is present
    and still present at the next point; presence at the last point alone is a birth there."""
    for t, p in enumerate(present):
        if p and (t == len(present) - 1 or present[t + 1]):
            return t
    return None


def late(births):
    """Late-born: born after the neuron's earliest-born group (neurons_verdict.py late_flags); none if none is born."""
    defined = [b for b in births if b is not None]
    return [bool(defined) and b is not None and b > min(defined) for b in births]


def step_dirs(harvest):
    found = {}
    for sub in ("1", "1b"):
        for d in sorted((harvest / sub / "out").iterdir()):
            m = re.fullmatch(r"(t\d\d)_step(\d+)", d.name)
            if m:
                assert m[1] not in found, f"{m[1]} appears twice: {found[m[1]]} and {d}"
                found[m[1]] = (int(m[2]), d)
    names = [f"t{i:02d}" for i in range(16)]
    assert sorted(found) == names, f"expected t00..t15, found {sorted(found)}"
    return [found[t] for t in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=Path, default=GROUPS)
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--harvest", type=Path, default=HARVEST)
    ap.add_argument("--recount", type=Path, default=RECOUNT)
    ap.add_argument("--verify", type=Path, default=VERIFY)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    q2 = json.loads(a.verdict.read_text())["q2"]
    grid_steps = [int(p.rsplit("_step", 1)[1]) for p in q2["inputs"]["grid"]]
    dirs = step_dirs(a.harvest)
    steps = [s for s, _ in dirs]
    n_scored = set()
    for _, d in dirs:
        m = json.loads((d / "metrics.json").read_text())
        assert m["complete"] is True, f"{d} metrics.json complete = {m['complete']}"
        n_scored.add(m["n_tokens_scored"])
    assert len(n_scored) == 1, f"n_tokens_scored differs across points: {n_scored}"
    n_scored = n_scored.pop()
    assert steps == grid_steps and steps[-1] == FINAL_STEP, (steps, grid_steps)
    NT = len(steps)

    # the per-group record, without the judge's calls
    rows = []
    with a.groups.open() as f:
        for s in f:
            r = json.loads(s)
            r.pop("judgments")
            rows.append(r)
    groups = [r for r in rows if not r["echo"]]
    by_pole = {}
    for r in groups:
        by_pole.setdefault(r["pole"], []).append(r)
    assert all(len(r["present_pct"]) == NT == len(r["med"]) for r in groups)

    # 1. presence recomputed from the activations: at each point, the median over the group's contexts of the
    #    activation in the neuron's own direction (clipped at 0), against the neuron's 99.9th percentile there.
    #    Rows of exemplar_a.npz follow the lines of contexts.jsonl (checked at the last point against the top-50 a).
    A, Q = [], []
    order = None
    for step, d in dirs:
        ctx = [json.loads(s) for s in (d / "samples/contexts.jsonl").read_text().splitlines() if s.strip()]
        poles_here = [c["pole"] for c in ctx]
        assert order is None or poles_here == order, f"{d}: pole order differs"
        order = poles_here
        A.append(np.load(d / "poles/exemplar_a.npz")["a"].astype(float))
        Q.append({c["pole"]: c["q999"] for c in ctx})
        if step == FINAL_STEP:
            final_ctx = {c["pole"]: c for c in ctx}
            for i, c in enumerate(ctx):
                for e in c["topk"]:
                    assert np.isclose(A[-1][i, e["rank"]], e["a"], rtol=1e-5), (c["pole"], e["rank"])
    A = np.stack(A)                                             # [T, 640, 50]
    row = {p: i for i, p in enumerate(order)}
    assert set(by_pole) <= set(row), "a neuron of the record is not in the harvest"

    def recompute(q_of_t):
        cells_same, med_err = 0, 0.0
        for r in groups:
            sgn = 1.0 if r["pole"].endswith("+") else -1.0
            med = np.median(np.maximum(sgn * A[:, row[r["pole"]], r["ranks"]], 0.0), axis=1)
            q = np.array([Q[q_of_t(t)][r["pole"]] for t in range(NT)])
            pres = (med > 0) & (med >= q)
            cells_same += int((pres == np.array(r["present_pct"])).sum())
            med_err = max(med_err, float(np.max(np.abs(med - np.array(r["med"])) / np.maximum(np.abs(med), 1e-30))))
        return cells_same, med_err
    n_cells = len(groups) * NT
    same, med_err = recompute(lambda t: t)
    same_wrong, _ = recompute(lambda t: (t + 1) % NT)            # the registered failing input: the next point's line

    # 2. births from the record's presence, and the late-born count
    births = {r["cluster"]: birth(r["present_pct"]) for r in groups}
    births_same = sum(births[r["cluster"]] == r["birth_pct"] for r in groups)
    counts = [sum(b == t for b in births.values()) for t in range(NT)]
    rec = [x + y for x, y in zip(q2["births"]["main"]["counts"], q2["births"]["non_main"]["counts"])]
    never = sum(b is None for b in births.values())
    late_pct, in_test = 0, 0
    for p, rs in by_pole.items():
        lp = late([births[r["cluster"]] for r in rs])
        lr = late([birth(r["present_raw"]) for r in rs])
        for r, x, y in zip(rs, lp, lr):
            late_pct += x
            in_test += x and y and r["matched"] and births[r["cluster"]] != T_FIRST_TRAINED
    status = Counter(r["status"] for r in groups)
    before_birth = sum(any(r["present_pct"][:births[r["cluster"]]]) for r in groups)
    drop_after = sum(not all(r["present_pct"][births[r["cluster"]]:]) for r in groups)
    # birth order (§6.4), as a check on the reading of births and of the strongest group
    order_hits = [min(births[r["cluster"]] for r in rs) == births[next(r["cluster"] for r in rs if r["is_main"])]
                  for rs in by_pole.values() if len(rs) >= 2]
    b_obs = sum(order_hits) / len(order_hits)

    # 4. the chance level (2026-11-05). Per group: the recount replaced each group of a neuron with 2+ groups by
    #    random groups of the same size from the same neuron's non-echo top-50 contexts (the strongest group's
    #    replacement keeps the neuron's top context), 2,000 times, dated by the same rule. ref_median is the median
    #    birth point of those random groups; p_late and p_early are one-sided against them.
    with (a.recount / "05_per_cluster_R1t.csv").open() as f:
        per = {(r["pole"], int(r["c"])): r for r in csv.DictReader(f)}
    chance_doc = json.loads((a.recount / "05_chance.json").read_text())
    r1t = chance_doc["summary"]["R1t"]
    b_rec = json.loads((a.recount / "04_b.json").read_text())
    multi = [r for r in groups if len(by_pole[r["pole"]]) >= 2]
    joined = [r for r in multi if (r["pole"], r["c"]) in per]
    same_size = sum(int(per[(r["pole"], r["c"])]["size"]) == len(r["ranks"]) for r in joined)
    same_birth = sum(int(per[(r["pole"], r["c"])]["real"]) == births[r["cluster"]] for r in joined)

    def neighbour(r):
        """The registered failing input for the join: the next group of the same neuron (by group number)."""
        cs = sorted(x["c"] for x in by_pole[r["pole"]])
        return per[(r["pole"], cs[(cs.index(r["c"]) + 1) % len(cs)])]
    same_birth_wrong = sum(int(neighbour(r)["real"]) == births[r["cluster"]] for r in joined)
    vs = {}
    for r in joined:
        x = per[(r["pole"], r["c"])]
        late_, early_ = float(x["p_late"]) <= ALPHA, float(x["p_early"]) <= ALPHA
        assert not (late_ and early_), (r["cluster"], x["p_late"], x["p_early"])
        vs[r["cluster"]] = "later" if late_ else "earlier" if early_ else "chance"
    n_later, n_earlier = Counter(vs.values())["later"], Counter(vs.values())["earlier"]
    #    Per training point: the verifier's random regrouping of all 600 neurons (300 draws, same sizes, same rule).
    hist = json.loads((a.verify / "s03_histogram_tv.json").read_text())
    pop_chance = [round(x, 1) for x in hist["random_mean_counts_A"]]

    say(f"Computed by `python3 extract.py` from `{a.groups}`, `{a.verdict}` and the 16 points under `{a.harvest}`.")
    say()
    say("| check | computed | expected |")
    say("|---|---:|---:|")
    say(f"| lines in q2_clusters.jsonl | {len(rows):,} | 2,465 |")
    say(f"| echo groups (left out) | {len(rows) - len(groups)} | {q2['echo_clusters']} (q2.echo_clusters) |")
    say(f"| groups | {len(groups):,} | {q2['n_clusters']:,} (q2.n_clusters) |")
    say(f"| neurons | {len(by_pole)} | {q2['n_poles']} (q2.n_poles) |")
    say(f"| presence cells recomputed from exemplar_a.npz and each point's q999 that equal the record | "
        f"{same:,} of {n_cells:,} | {n_cells:,} |")
    say(f"| largest relative difference, recomputed median against the record's `med` | {med_err:.1e} | below 1e-6 |")
    say(f"| the same with each point given the next point's q999 (must fail) | {same_wrong:,} of {n_cells:,} | "
        f"fewer than {n_cells:,} |")
    say(f"| births recomputed from presence that equal the record's birth point | {births_same:,} of {len(groups):,} | "
        f"{len(groups):,} |")
    say(f"| groups never born | {never} | {q2['births']['main']['never_born'] + q2['births']['non_main']['never_born']} "
        f"(q2.births) |")
    say(f"| late-born groups (born after the neuron's first-born group) | {late_pct:,} | |")
    say(f"| late-born groups in the test (both readings agree, matched in the second sorting, not born at step 1,000) | "
        f"{in_test:,} | {q2['p_u']['n_clusters']:,} (q2.p_u.n_clusters); status late_primary {status['late_primary']:,} |")
    say(f"| groups present at an earlier point that was not followed by presence (before birth) | {before_birth} | |")
    say(f"| groups absent at some point after their birth | {drop_after} | {EXPECTED_DROP_AFTER_BIRTH} (E_skeptic.md §3.2) |")
    say(f"| b, neurons with 2+ groups whose strongest group is among the first born (ties count as first) | "
        f"{sum(order_hits)} of {len(order_hits)} = {b_obs:.4f} | {q2['birth_order']['b']:.4f} of "
        f"{q2['birth_order']['n_poles']} (q2.birth_order.b); {b_rec['b']:.4f} (recount `04_b.json`) |")
    lo, hi = r1t["b_ref_ci95_over_replicates"]
    say(f"| b at the chance level: random groups of the same sizes from the same neuron, the strongest group's "
        f"holding the top context | {r1t['b_ref_mean']:.3f} [{lo:.3f}, {hi:.3f}] over {chance_doc['reps']:,} replicates | "
        f"`05_chance.json` summary.R1t; b is {'below' if b_obs < lo else 'inside' if b_obs <= hi else 'above'} it. "
        f"The registered permutation band (97.5th percentile {q2['birth_order']['null_97_5']:.3f}) keeps the real "
        f"groups' ties and is not a chance level for size and rank (retagged 2026-11-05) |")
    say(f"| groups on neurons with 2+ groups, and of them in the recount's per-group file | {len(multi):,}, "
        f"{len(joined):,} | {r1t['n_clusters']:,} (summary.R1t.n_clusters) |")
    say(f"| of them, the recount's size and birth point equal the record's | {same_size:,} and {same_birth:,} | "
        f"{len(multi):,} |")
    say(f"| the same join with each group given the next group's row of its neuron (must fail) | "
        f"{same_birth_wrong:,} of {len(joined):,} | fewer than {len(joined):,} |")
    say(f"| born later than chance (p_late ≤ {ALPHA}) | {n_later} | {r1t['all']['late_beat']} (summary.R1t.all; "
        f"{r1t['all']['late_expected']} expected at chance) |")
    say(f"| born earlier than chance (p_early ≤ {ALPHA}) | {n_earlier} | {r1t['all']['early_beat']} (summary.R1t.all; "
        f"{r1t['all']['early_expected']} expected at chance) |")
    say(f"| births per point in the verifier's file that equal q2.births | "
        f"{sum(x == y for x, y in zip(hist['real_counts'], rec))} of {NT} | {NT} |")
    say()
    say("Groups born at each training point, all 600 neurons:")
    say()
    say("| point | " + " | ".join(f"T{t}" for t in range(NT)) + " |")
    say("|---|" + "---:|" * NT)
    say("| step | " + " | ".join(f"{s:,}" for s in steps) + " |")
    say("| recomputed | " + " | ".join(str(x) for x in counts) + " |")
    say("| q2.births, main + non_main | " + " | ".join(str(x) for x in rec) + " |")
    say(f"| chance level: random groups of the same sizes, mean of {hist['n_draws']} draws (06b `s03_histogram_tv.json`) | "
        + " | ".join(f"{x:.1f}" for x in pop_chance) + " |")
    say()

    # 3. the neurons in the picker: the named ones, then a seed-0 draw from the rest of the 600
    named = [p for p, _, _ in NAMED]
    assert all(p in by_pole for p in named), [p for p in named if p not in by_pole]
    pool = sorted(set(by_pole) - set(named))
    drawn = random.Random(SEED).sample(pool, N_RANDOM)

    def example(pole, ranks):
        """The group's strongest context at the end of training, cut to BEFORE_CHARS and AFTER_CHARS."""
        e = next(e for e in final_ctx[pole]["topk"] if e["rank"] == min(ranks))
        return [e["before"][-BEFORE_CHARS:], e["token"], e["after"][:AFTER_CHARS]]

    def chance_of(r):
        """[the median birth point of random groups of its size, 'later' / 'earlier' / 'chance'];
        [None, None] on a neuron with one group, whose group holds all its contexts."""
        if len(by_pole[r["pole"]]) < 2:
            return [None, None]
        t = float(per[(r["pole"], r["c"])]["ref_median"])
        return [int(t) if t.is_integer() else t, vs[r["cluster"]]]

    def neuron(pole, kind, label):
        rs = sorted(by_pole[pole], key=lambda r: (births[r["cluster"]], -len(r["ranks"]), r["c"]))
        return {"pole": pole, "kind": kind, "label": label,
                "groups": [[r["name"], len(r["ranks"]), births[r["cluster"]],
                            "".join("1" if x else "0" for x in r["present_pct"]), example(pole, r["ranks"]),
                            *chance_of(r)]
                           for r in rs]}

    neurons = [neuron(p, "named", lab) for p, lab, _ in NAMED] + [neuron(p, "random", None) for p in drawn]

    li = neurons[0]
    say(f"The lithium neuron, {li['pole']}: {len(li['groups'])} groups (present: 1, by point T0 to T15).")
    say()
    say("| group | contexts | born at | step | present | chance level (median of random groups of its size) | against chance |")
    say("|---|---:|---:|---:|---|---:|---|")
    for name, n, b, pres, _, ct, v in li["groups"]:
        say(f"| {name} | {n} | T{b} | {steps[b]:,} | `{pres}` | T{ct} | {v} |")
    say()
    got = neurons[1]
    got_steps = sorted(steps[g[2]] for g in got["groups"])
    say(f"The got neuron, {got['pole']}: groups born at steps {', '.join(f'{s:,}' for s in got_steps)}; expected "
        f"{', '.join(f'{s:,}' for s in EXPECTED_GOT_STEPS)} (A_training_dynamics.md §5).")
    say()
    say(f"The picker holds the {len(NAMED)} named neurons in `extract.py` and {N_RANDOM} drawn by "
        f"`random.Random({SEED}).sample(pool, {N_RANDOM})` from the other {len(pool)} neurons (ids sorted as strings): "
        + ", ".join(drawn) + ".")
    say()

    differ = []
    check = lambda ok, what: differ.append(what) if not ok else None
    check(len(rows) == 2465, "lines")
    check(len(rows) - len(groups) == q2["echo_clusters"], "echo")
    check(len(groups) == q2["n_clusters"] and len(by_pole) == q2["n_poles"], "groups or neurons")
    check(same == n_cells, "presence recomputed")
    check(med_err < 1e-6, "median recomputed")
    check(same_wrong < n_cells, "the wrong point's q999 did not fail")
    check(births_same == len(groups), "births recomputed")
    check(counts == rec, "births per point")
    check(in_test == q2["p_u"]["n_clusters"] == status["late_primary"], "late-born in the test")
    check(drop_after == EXPECTED_DROP_AFTER_BIRTH, "drops after birth")
    check(abs(b_obs - q2["birth_order"]["b"]) < 1e-12 and len(order_hits) == q2["birth_order"]["n_poles"], "b")
    check(got_steps == EXPECTED_GOT_STEPS, "got births")
    check(len(multi) == len(joined) == r1t["n_clusters"] == same_size == same_birth, "recount join")
    check(same_birth_wrong < len(joined), "the neighbour's row did not fail")
    check(n_later == r1t["all"]["late_beat"] and n_earlier == r1t["all"]["early_beat"], "later or earlier than chance")
    check(abs(b_obs - b_rec["b"]) < 1e-12 and abs(b_obs - r1t["b_real"]) < 1e-12, "b against the recount")
    check(hist["real_counts"] == rec and hist["n_clusters"] == len(groups), "the verifier's births per point")
    say("Every expected number above matches, and both failing inputs fail (the next point's line, the next "
        "group's row)." if not differ
        else "These differ from the expected numbers: " + "; ".join(differ) + ".")
    say()

    data = {"model": MODEL, "n_tokens_scored": n_scored, "steps": [{"t": t, "step": s, "pct": round(100 * s / FINAL_STEP, 4)}
                                      for t, s in enumerate(steps)],
            "population": {"births": rec, "n_groups": len(groups), "n_neurons": len(by_pole),
                           "chance": pop_chance, "chance_draws": hist["n_draws"]},
            "chance": {"reps": chance_doc["reps"], "alpha": ALPHA, "n_groups": len(joined),
                       "later": n_later, "later_expected": r1t["all"]["late_expected"],
                       "earlier": n_earlier, "earlier_expected": r1t["all"]["early_expected"]},
            "neurons": neurons}
    out = a.out / "data.js"
    out.write_text("window.BIRTHS_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, f"{out} is {size:,} bytes"
    say(f"Wrote `data.js`: {size:,} bytes, {len(neurons)} neurons, {sum(len(n['groups']) for n in neurons)} groups; "
        f"sha256 `{hashlib.sha256(out.read_bytes()).hexdigest()}`.")
    assert not differ, differ

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, rest = txt.split(CHECKS_START)
    readme.write_text(head + CHECKS_START + "\n" + "\n".join(lines) + "\n" + CHECKS_END + rest.split(CHECKS_END)[1])
    print(f"\nrewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
