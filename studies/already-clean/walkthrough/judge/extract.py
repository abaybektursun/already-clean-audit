"""Build data.js for the picture "The judge's calls".

Reads saved files only: the judge's record (w4/o7_s_v2/explain/judgments.jsonl, one line per call), the verdict's
per-group record (w5/verdict/samples/q2_clusters.jsonl, one line per group with the text the judge saw for each
call), the verdict's saved sample calls (w5/verdict/samples/q2_parent.jsonl), the verdict's metrics.json, and the
explain manifest (the judge model). Writes data.js next to this file, prints the checks, and rewrites the checks
block of README.md with them.

    python3 extract.py
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The local copy on the laptop; README.md names the S3 source of each.
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
JUDGMENTS = BANK / "w4/o7_s_v2/explain/judgments.jsonl"
MANIFEST = BANK / "w4/o7_s_v2/explain/manifest.json"
GROUPS = BANK / "w5/verdict/samples/q2_clusters.jsonl"
PARENT = BANK / "w5/verdict/samples/q2_parent.jsonl"
VERDICT = BANK / "w5/verdict/metrics.json"

MODEL = "OLMo-3 7B"
JUDGE_ID, JUDGE_NAME = "anthropic/claude-sonnet-5-5", "Claude Sonnet 5.5"
CONTROL_BEFORE = 14          # the control groups are judged against the record at T_14 (SPEC_D1 §2.4)
N_BOOT, SEED = 10_000, 0     # the verdict's p_u bootstrap (infra/pg/neurons_verdict.py p_u_bootstrap)
CALL_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]   # (sample, draw): the order of a group's four calls in data.js
MARK = re.compile(r"⟦(.*?)⟧", re.S)
HEAD = "Exemplars of the job at the end of training:\n"
MAX_BYTES = 4_000_000
# The named calls in the picker, in this order: call id, the label the viewer sees, where the write-up names it.
NAMED = [
    ("judge:w4j:3893", "The lithium neuron: its strongest group against the lithium record",
     "stories A_training_dynamics.md §2, E_skeptic.md §3.4"),
    ("judge:w4j:10996", "The lithium neuron: the same group as a control group",
     "stories A_training_dynamics.md §2"),
    ("judge:w4j:8908", "The lithium neuron: a canyon-name group against the licence-URL record",
     "stories A_training_dynamics.md §2"),
    ("judge:w4j:6843", "The got neuron: gettin’ against the gotta and gonna record",
     "stories A_training_dynamics.md §5"),
    ("judge:w4j:2715", "The got neuron: gotta and gonna against the Got record",
     "stories A_training_dynamics.md §5"),
    ("judge:w4j:2287", "The Gul neuron: gulch, right with no gul in the record",
     "stories A_training_dynamics.md §9, E_skeptic.md §4"),
    ("judge:w4j:6627", "The Gul neuron: Gul- names against a record of marketing nouns",
     "stories A_training_dynamics.md §9"),
    ("judge:w4j:4128", "The React neuron: useEffect against the Truly and Exactly record",
     "stories B_clutter.md Story 3, E_skeptic.md §1"),
    ("judge:w4j:5972", "The laughter neuron: haha against the URL question-mark record",
     "stories B_clutter.md Story 4, E_skeptic.md §3.4"),
    ("judge:w4j:5632", "The G neuron: wrong against a record whose next words start with G",
     "stories B_clutter.md Story 9"),
]
LITHIUM = "L03N6372+"
N_PER_STRATUM = 5            # calls drawn at random for each column and outcome
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().split("\n") if s.strip()]


def parse_saw(saw):
    """The judge's user message as three lists of contexts: the group's, record A and record B. Each context
    holds exactly one marked token; it comes back as (before, token, after)."""
    assert saw.startswith(HEAD), saw[:60]
    ex, rest = saw[len(HEAD):].split("\n\nRecord A:\n")
    a, b = rest.split("\n\nRecord B:\n")

    def items(block):
        out = []
        for line in block.split("\n"):
            assert line.startswith("- "), repr(line[:40])
            text = line[2:]
            m = MARK.findall(text)
            assert len(m) == 1, f"{len(m)} marks in {text[:60]!r}"
            before, rest_ = text.split("⟦", 1)
            token, after = rest_.split("⟧", 1)
            out.append((before, token, after))
        return out
    return items(ex), items(a), items(b)


def shares_token(ex, rec):
    """A marked token of the group's contexts is also a marked token of the record, case and surrounding spaces
    ignored (the definition that gives E_skeptic.md §3.1's 50.0% and 18.7%)."""
    norm = lambda t: t.strip().lower()
    return bool({norm(t) for _, t, _ in ex} & {norm(t) for _, t, _ in rec})


def p_u_bootstrap(cluster_acc, ctrl_acc, n_boot, seed):
    """A port of p_u_bootstrap in infra/pg/neurons_verdict.py: poles resampled with replacement, late-born groups
    move with their pole, the control accuracy is recomputed over the same poles. No draw here is degenerate
    (every pole has a control), which the function asserts instead of redrawing."""
    S = np.array([sum(x) for x in cluster_acc], dtype=float)
    n = np.array([len(x) for x in cluster_acc], dtype=float)
    C = np.array(ctrl_acc, dtype=float)
    m = np.ones(len(C))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(n_boot, len(S)))
    assert ((n[idx].sum(1) > 0) & (m[idx].sum(1) > 0)).all(), "a degenerate draw; the verdict would redraw it"
    A_b, a_b = S[idx].sum(1) / n[idx].sum(1), C[idx].sum(1) / m[idx].sum(1)
    pu = np.clip((a_b - A_b) / (a_b - 0.5), 0.0, 1.0)
    lo, hi = np.percentile(pu, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--groups", type=Path, default=GROUPS)
    ap.add_argument("--parent", type=Path, default=PARENT)
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    verdict = json.loads(a.verdict.read_text())
    q2 = verdict["q2"]
    pu_rec = q2["p_u"]
    steps = [int(p.rsplit("_step", 1)[1]) for p in q2["inputs"]["grid"]]
    assert len(steps) == 16 and steps[0] == 0 and steps == sorted(steps), steps

    # the judge's record, its hash against the one the verdict recorded, and the judge model
    sha = hashlib.sha256(a.judgments.read_bytes()).hexdigest()
    assert sha == q2["inputs"]["judgments"]["sha256"], f"{a.judgments} sha256 {sha} is not the verdict's"
    manifest = json.loads(a.manifest.read_text())
    judge_models = {mo for b in manifest["batches"] if "judge" in b["roles"] for mo in b["models"]}
    assert judge_models == {JUDGE_ID}, judge_models
    assert manifest["prompts"]["judge"].startswith("11b5fd81"), manifest["prompts"]["judge"]
    calls = read_jsonl(a.judgments)
    by_id = {j["id"]: j for j in calls}
    assert len(by_id) == len(calls), "a call id twice"
    random_calls = [j for j in calls if j["decoy_type"] == "random"]
    hard_calls = [j for j in calls if j["decoy_type"] == "hard"]
    # every answer agrees with its stored outcome: the picked record is the neuron's own exactly when correct
    for j in calls:
        picked = j["shown_order"][0 if j["answer"] == "A" else 1]
        assert (picked == "true") == j["correct"], j["id"]
    tests, ctrls = defaultdict(dict), defaultdict(dict)
    for j in random_calls:
        key = j["cluster"].split(":")[0] if j["control"] else j["cluster"]
        target = ctrls if j["control"] else tests
        assert (j["sample"], j["draw"]) not in target[key], j["id"]
        target[key][(j["sample"], j["draw"])] = j
    assert all(sorted(d) == sorted(CALL_ORDER) for d in list(tests.values()) + list(ctrls.values())), \
        "a group without its four random-decoy calls"

    # the verdict's own saved sample calls, to compare with the text in the per-group record
    parent = read_jsonl(a.parent)
    parent_saw = {}

    # the per-group record: statuses, births and the text of every call
    rows, texts = [], {}
    overlap = {False: [0, 0], True: [0, 0]}          # control? -> [calls sharing a token with own record, calls]
    overlap_decoy = {False: [0, 0], True: [0, 0]}    # the same against the other neuron's record
    with a.groups.open() as f:
        for s in f:
            r = json.loads(s)
            js = r.pop("judgments")
            r["n_calls"] = len(js)
            rows.append(r)
            for j in js:
                o = by_id[j["id"]]
                assert all(o[k] == j[k] for k in o), f"{j['id']}: the two records of this call differ"
                if any(p["id"] == j["id"] for p in parent):
                    parent_saw[j["id"]] = j["judge_saw"]
                if j["decoy_type"] != "random":
                    continue
                ex, ra, rb = parse_saw(j["judge_saw"])
                assert len(ra) == len(rb) >= 1, j["id"]
                assert len(ex) == min(10, len(r["ranks"])), j["id"]
                own, other = (ra, rb) if j["shown_order"][0] == "true" else (rb, ra)
                overlap[j["control"]][0] += shares_token(ex, own)
                overlap[j["control"]][1] += 1
                overlap_decoy[j["control"]][0] += shares_token(ex, other)
                overlap_decoy[j["control"]][1] += 1
                texts[j["id"]] = (ex, ra, rb, j["t_before"])
    assert len(texts) == len(random_calls), (len(texts), len(random_calls))

    groups = [r for r in rows if not r["echo"]]
    status = Counter(r["status"] for r in groups)
    late_pct = [r for r in groups if r["late_pct"]]
    late = sorted((r for r in groups if r["status"] == "late_primary"), key=lambda r: (r["pole"], r["c"]))
    mains = sorted((r for r in groups if r["is_main"]), key=lambda r: r["pole"])
    assert {r["cluster"] for r in late} == set(tests), "judged groups are not the late-born groups in the test"
    assert {r["pole"] for r in mains} == set(ctrls) and len(mains) == len(ctrls), "controls are not the strongest groups"
    for r in late:
        k = sum(j["correct"] for j in tests[r["cluster"]].values())
        assert r["acc_random"] == k / 4 and r["flagged"] == (k <= 1), r["cluster"]
        assert int(tests[r["cluster"]][(0, 0)]["t_before"][2:]) == r["birth_pct"] - 1 == r["t_before"], r["cluster"]
    for r in mains:
        assert all(j["t_before"] == f"T_{CONTROL_BEFORE}" for j in ctrls[r["pole"]].values()), r["pole"]

    # the readout, recomputed from the calls
    acc = {r["cluster"]: sum(j["correct"] for j in tests[r["cluster"]].values()) / 4 for r in late}
    cacc = {p: sum(j["correct"] for j in d.values()) / 4 for p, d in ctrls.items()}
    n_test_calls, n_test_right = 4 * len(late), sum(j["correct"] for d in tests.values() for j in d.values())
    n_ctrl_calls, n_ctrl_right = 4 * len(ctrls), sum(j["correct"] for d in ctrls.values() for j in d.values())
    A = float(np.mean(list(acc.values())))
    a_ctrl = float(np.mean(list(cacc.values())))
    pu = (a_ctrl - A) / (a_ctrl - 0.5)
    poles = sorted(cacc)
    per_pole = defaultdict(list)
    for r in late:
        per_pole[r["pole"]].append(acc[r["cluster"]])
    lo, hi = p_u_bootstrap([per_pole[p] for p in poles], [cacc[p] for p in poles], N_BOOT, SEED)
    flagged = sum(v <= 0.25 for v in acc.values())
    flagged_ctrl = sum(v <= 0.25 for v in cacc.values())
    hist = {"late": [sum(round(4 * v) == k for v in acc.values()) for k in range(5)],
            "control": [sum(round(4 * v) == k for v in cacc.values()) for k in range(5)]}
    # the same computation on the hard-decoy calls (another decoy rule): it must not give back A
    hard_acc = defaultdict(list)
    for j in hard_calls:
        if not j["control"]:
            hard_acc[j["cluster"]].append(j["correct"])
    A_hard = float(np.mean([np.mean(v) for v in hard_acc.values()]))
    # a fair coin in place of the judge, drawn once with a fixed seed
    coin = random.Random(SEED)
    coin_test = [sum(coin.random() < 0.5 for _ in range(4)) for _ in late]
    coin_ctrl = [sum(coin.random() < 0.5 for _ in range(4)) for _ in mains]

    say(f"Computed by `python3 extract.py` from `{a.judgments}` and `{a.groups}`.")
    say()
    say(f"- The judge's record has sha256 `{sha}`, the value at `q2.inputs.judgments.sha256` in `{a.verdict}`.")
    say(f"- {len(calls):,} calls: {len(random_calls):,} against a random other neuron of the same layer (the "
        f"primary test, shown in the picture) and {len(hard_calls):,} against a hard decoy (not shown). The judge "
        f"model in the manifest is `{JUDGE_ID}`, prompt sha256 `{manifest['prompts']['judge'][:8]}…`.")
    say("- Every call's answer and shown order give back its stored outcome. Every call in the judge's record "
        "appears with the same fields in the per-group record, which also holds the text the judge saw.")
    say(f"- Each judged group has exactly four random-decoy calls: two random neurons, each shown twice "
        f"(`sample` 0 and 1, `draw` 0 and 1). Each context the judge saw carries exactly one marked token; the two "
        f"records of a call have the same length; a group shows min(10, its size) contexts.")
    say()
    say("| check | computed | expected |")
    say("|---|---:|---:|")
    say(f"| lines in q2_clusters.jsonl (groups, with the echo groups) | {len(rows):,} | 2,465 |")
    say(f"| echo groups (left out of every count) | {len(rows) - len(groups)} | {q2['echo_clusters']} (q2.echo_clusters) |")
    say(f"| groups | {len(groups):,} | {q2['n_clusters']:,} (q2.n_clusters) |")
    say(f"| strongest groups at the end (one per neuron; the control groups) | {len(mains)} | {q2['n_poles']} (q2.n_poles) |")
    say(f"| groups first in the top 0.1% at the neuron's earliest point | {status['earliest']} | "
        f"{q2['status_counts']['earliest']} (q2.status_counts) |")
    say(f"| late-born groups, percentile reading | {len(late_pct):,} | |")
    say(f"| of these, not late-born under the raw-ratio reading | {status['readings_disagree']} | "
        f"{q2['status_counts']['readings_disagree']} (q2.status_counts) |")
    say(f"| of these, no match in the second sorting | {status['unmatched_in_sample_1']} | "
        f"{q2['status_counts']['unmatched_in_sample_1']} (q2.status_counts) |")
    say(f"| late-born groups in the test | {len(late):,} | {pu_rec['n_clusters']:,} (q2.p_u.n_clusters) |")
    say(f"| calls on late-born groups, right | {n_test_right:,} of {n_test_calls:,} | |")
    say(f"| A, mean share right per late-born group | {A:.6f} | {pu_rec['A']:.6f} (q2.p_u.A) |")
    say(f"| calls on control groups, right | {n_ctrl_right:,} of {n_ctrl_calls:,} | 2,218 of 2,400 (E_skeptic.md §1) |")
    say(f"| a_ctrl, mean share right per control group | {a_ctrl:.6f} | {pu_rec['a_ctrl']:.6f} (q2.p_u.a_ctrl) |")
    say(f"| p_u = (a_ctrl − A) / (a_ctrl − 0.5) | {pu:.6f} | {pu_rec['point']:.6f} (q2.p_u.point) |")
    say(f"| its 95% interval, {N_BOOT:,} draws over neurons, seed {SEED} | [{lo:.6f}, {hi:.6f}] | "
        f"[{pu_rec['ci95'][0]:.6f}, {pu_rec['ci95'][1]:.6f}] (q2.p_u.ci95) |")
    say(f"| late-born groups flagged (at most 1 of 4 right) | {flagged} | {q2['flags']['n_flagged']} (q2.flags.n_flagged) |")
    say(f"| control groups with at most 1 of 4 right | {flagged_ctrl} | {round(q2['f_ctrl'] * 600)} (q2.f_ctrl × 600) |")
    say(f"| control calls sharing a marked token with the neuron's own record | "
        f"{overlap[True][0]:,} of {overlap[True][1]:,} = {overlap[True][0] / overlap[True][1]:.3f} | 0.500 (E_skeptic.md §3.1) |")
    say(f"| late-born calls sharing a marked token with the neuron's own record | "
        f"{overlap[False][0]:,} of {overlap[False][1]:,} = {overlap[False][0] / overlap[False][1]:.3f} | 0.187 (E_skeptic.md §3.1) |")
    say(f"| the same against the other neuron's record (control, late-born) | "
        f"{overlap_decoy[True][0] / overlap_decoy[True][1]:.3f}, {overlap_decoy[False][0] / overlap_decoy[False][1]:.3f} | "
        f"well below both |")
    say(f"| A from the hard-decoy calls instead (a different input; must differ) | {A_hard:.4f} | "
        f"{q2['hard']['accuracy']:.4f} (q2.hard.accuracy) |")
    say(f"| a fair coin, seed {SEED}: share right, late-born and control | "
        f"{sum(coin_test) / (4 * len(late)):.3f}, {sum(coin_ctrl) / (4 * len(mains)):.3f} | 0.5, 0.5 |")
    say(f"| a fair coin: late-born groups with at most 1 of 4 right | {sum(k <= 1 for k in coin_test)} | "
        f"{len(late) * 5 / 16:.0f} (5/16 of {len(late):,}) |")
    say()
    say("Groups by the number of their four calls the judge got right:")
    say()
    say("| right of 4 | 0 | 1 | 2 | 3 | 4 |")
    say("|---|---:|---:|---:|---:|---:|")
    say("| late-born groups | " + " | ".join(str(x) for x in hist["late"]) + " |")
    say("| control groups | " + " | ".join(str(x) for x in hist["control"]) + " |")
    say()
    b_all = [sum(r["birth_pct"] == t for r in groups) for t in range(16)]
    b_rec = [x + y for x, y in zip(q2["births"]["main"]["counts"], q2["births"]["non_main"]["counts"])]
    b_late = [sum(r["birth_pct"] == t for r in late) for t in range(16)]
    say("Groups by the training point where they were born (first in the neuron's top 0.1% and still there at the "
        "next point):")
    say()
    say("| point | " + " | ".join(f"T{t}" for t in range(16)) + " |")
    say("|---|" + "---:|" * 16)
    say("| step | " + " | ".join(f"{s:,}" for s in steps) + " |")
    say("| all groups | " + " | ".join(str(x) for x in b_all) + " |")
    say("| q2.births, main + non_main | " + " | ".join(str(x) for x in b_rec) + " |")
    say("| late-born groups in the test | " + " | ".join(str(x) for x in b_late) + " |")
    say()

    differ = []
    check = lambda ok, what: differ.append(what) if not ok else None
    check(len(rows) == 2465, f"lines {len(rows)}")
    check(len(groups) == q2["n_clusters"], "groups")
    check(len(rows) - len(groups) == q2["echo_clusters"], "echo")
    check(all(status[k] == v for k, v in q2["status_counts"].items()), "status counts")
    check(len(late) == pu_rec["n_clusters"], "late-born count")
    check(len(mains) == q2["n_poles"] == 600, "controls")
    check(abs(A - pu_rec["A"]) < 1e-12, "A")
    check(abs(a_ctrl - pu_rec["a_ctrl"]) < 1e-12, "a_ctrl")
    check((n_ctrl_right, n_ctrl_calls) == (2218, 2400), "control calls")
    check(abs(pu - pu_rec["point"]) < 1e-12, "p_u")
    check(abs(lo - pu_rec["ci95"][0]) < 1e-12 and abs(hi - pu_rec["ci95"][1]) < 1e-12, "p_u interval")
    check(flagged == q2["flags"]["n_flagged"], "flagged")
    check(flagged_ctrl == round(q2["f_ctrl"] * 600), "control flags")
    check(round(overlap[True][0] / overlap[True][1], 3) == 0.500, "control overlap")
    check(round(overlap[False][0] / overlap[False][1], 3) == 0.187, "late-born overlap")
    check(abs(A_hard - q2["hard"]["accuracy"]) < 1e-12 and abs(A_hard - A) > 0.05, "hard-decoy A")
    check(b_all == b_rec, "births")
    say("Every expected number above matches, and the hard-decoy calls give a different A." if not differ
        else "These differ from the expected numbers: " + "; ".join(differ) + ".")
    say()

    # the verdict's own saved sample calls carry the same text as the per-group record
    same = sum(p["judge_saw"] == parent_saw.get(p["id"]) for p in parent)
    say(f"The {len(parent)} calls the verdict saved with their text in `q2_parent.jsonl` "
        f"({', '.join(p['id'] for p in parent)}) carry the same text in the per-group record: {same} of {len(parent)}.")
    assert same == len(parent), "q2_parent.jsonl and q2_clusters.jsonl disagree on the text of a call"
    say()

    # the calls the picker opens: the named ones, then N_PER_STRATUM per column and outcome, drawn at random
    named_ids = [cid for cid, _, _ in NAMED]
    assert by_id[named_ids[0]]["cluster"].startswith(LITHIUM), "the first example is not from the lithium neuron"
    assert all(cid in texts for cid in named_ids), "a named call is not a random-decoy call"
    rng = random.Random(SEED)
    strata = [(False, True), (False, False), (True, True), (True, False)]
    drawn = []
    for control, correct in strata:
        pool = sorted(j["id"] for j in random_calls
                      if j["control"] is control and j["correct"] is correct and j["id"] not in named_ids)
        drawn += [(cid, control, correct) for cid in rng.sample(pool, N_PER_STRATUM)]
    say(f"The picker holds the {len(NAMED)} named calls in extract.py and {len(drawn)} drawn by "
        f"`random.Random({SEED}).sample(pool, {N_PER_STRATUM})` from each of four pools (late-born right, late-born "
        f"wrong, control right, control wrong; call ids sorted as strings, named calls left out): "
        + ", ".join(f"{cid} ({by_id[cid]['cluster']})" for cid, _, _ in drawn) + ".")
    say()

    # data.js
    def gid(r):
        return r["cluster"]
    group_list = [(r, False) for r in late] + [(r, True) for r in mains]
    index = {}
    out_groups = []
    for i, (r, control) in enumerate(group_list):
        d = ctrls[r["pole"]] if control else tests[r["cluster"]]
        index[(gid(r), control)] = i
        out_groups.append([r["pole"], r["name"], r["birth_pct"], int(control), len(r["ranks"]),
                           "".join("1" if d[k]["correct"] else "0" for k in CALL_ORDER),
                           [d[k]["decoy_pole"] for k in CALL_ORDER]])

    def example(cid, kind, label, source):
        j = by_id[cid]
        ex, ra, rb, tb = texts[cid]
        r = next(r for r in groups if r["cluster"] == j["cluster"])
        gi = index[(gid(r), j["control"])]
        return {"id": cid, "kind": kind, "label": label, "source": source, "g": gi,
                "k": CALL_ORDER.index((j["sample"], j["draw"])), "order": j["shown_order"][0][0] + j["shown_order"][1][0],
                "answer": j["answer"], "t": int(tb[2:]), "ex": ex, "a": ra, "b": rb}

    examples = [example(cid, "named", label, src) for cid, label, src in NAMED]
    examples += [example(cid, ("control" if c else "late") + ("-right" if ok else "-wrong"), None, None)
                 for cid, c, ok in drawn]
    data = {"model": MODEL, "judge": JUDGE_NAME, "steps": steps, "control_before": CONTROL_BEFORE,
            "n_late": len(late), "n_control": len(mains),
            "right": {"late": n_test_right, "control": n_ctrl_right},
            "A": A, "a_ctrl": a_ctrl, "share": pu, "ci95": [lo, hi],
            "overlap": {"control": overlap[True][0] / overlap[True][1], "late": overlap[False][0] / overlap[False][1]},
            "hist": hist, "groups": out_groups, "examples": examples}
    out = a.out / "data.js"
    out.write_text("window.JUDGE_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, f"{out} is {size:,} bytes"
    say(f"Wrote `data.js`: {size:,} bytes, {len(out_groups):,} groups ({len(late):,} late-born, {len(mains)} control) "
        f"with their {4 * len(out_groups):,} calls, and {len(examples)} calls with their text; sha256 "
        f"`{hashlib.sha256(out.read_bytes()).hexdigest()}`.")
    say()
    e = examples[0]
    own = e["a"] if e["order"][0] == "t" else e["b"]
    say(f"The first call in the picker, {e['id']} on {by_id[e['id']]['cluster']}, judged against the records at step "
        f"{steps[e['t']]:,}: the neuron's own record marks " +
        ", ".join(f"`{t.strip()}` {n}" for t, n in Counter(t.strip() for _, t, _ in own).most_common(3)) +
        f"; the judge answered {e['answer']}, which is {'right' if by_id[e['id']]['correct'] else 'wrong'}. "
        f"Its first context:")
    say()
    show = lambda s: s.replace("\n", "⏎")
    b0, t0, a0 = e["ex"][0]
    say("```")
    say(f"…{show(b0[-120:])}[{show(t0)}]{show(a0[:40])}")
    say("```")

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, rest = txt.split(CHECKS_START)
    readme.write_text(head + CHECKS_START + "\n" + "\n".join(lines) + "\n" + CHECKS_END + rest.split(CHECKS_END)[1])
    print(f"\nrewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
