"""Build data.js for the picture "One neuron's 50 contexts, sorted into groups".

Reads saved files only: the clusterer's record (w4/o7_s_v2/explain/clusters.jsonl, two sortings of each of
the 600 study neurons), the end-of-training contexts (w3/grid/1/out/t15_step1413814/samples/contexts.jsonl
and the metrics.json beside samples/), the W5 verdict metrics.json (the record's sha256, the study's group
counts and the gallery) and the study set (w3/o7/draw/S.json). Writes data.js next to this file, prints
the checks, and rewrites the checks block of README.md with them.

    python3 extract.py
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The local copy on the laptop; README.md names the S3 source of each.
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
CLUSTERS = BANK / "w4/o7_s_v2/explain/clusters.jsonl"
T15 = BANK / "w3/grid/1/out/t15_step1413814"
VERDICT = BANK / "w5/verdict/metrics.json"
DRAW = BANK / "w3/o7/draw/S.json"

MODEL, SORTER, FINAL_STEP, K = "OLMo-3 7B", "Claude Sonnet 5.5", 1413814, 50
LITHIUM = "L03N6372+"
# The examples in the picker, in this order: the label the viewer sees and where the write-up names it.
NAMED = [
    ("L03N6372+", "the lithium neuron", "stories B_clutter.md Story 1"),
    ("L26N929-", "one group under both sortings", "chosen for this picture"),
    ("L06N7759+", "laughter (haha, lol)", "stories B_clutter.md Story 4"),
    ("L06N572+", "a rare term seen earlier in the document", "stories B_clutter.md Story 6"),
    ("L12N8337+", "sentence-ending periods in spam text", "stories B_clutter.md Story 6"),
    ("L02N4649-", "Rouhani and Roulette", "stories B_clutter.md Story 6"),
    ("L03N7155+", "forms of got", "stories A_training_dynamics.md §5"),
    ("L01N10284-", "Exactly, and React's useEffect", "stories B_clutter.md Story 3"),
    ("L16N10459+", "latch, hinge, zipper", "stories B_clutter.md Story 5"),
    ("L06N4452-", "new Random, and rhyme", "stories B_clutter.md Story 5"),
    ("L09N10681+", "git head and rawhide", "stories B_clutter.md Story 5"),
    ("L31N10276-", "the next word starts with G", "stories B_clutter.md Story 9"),
    ("L12N1003-", "a modal verb after It", "stories B_clutter.md Story 9"),
    ("L05N924+", "together with", "stories B_clutter.md Story 9"),
    ("L03N10451+", "switch off", "stories B_clutter.md Story 10"),
    ("L05N2634+", "verbs in spam text", "stories B_clutter.md Story 11"),
    ("L01N6100+", "Chief as a job title", "stories B_clutter.md §2"),
]
N_TOTAL, SEED = 40, 0
BEFORE_CHARS, AFTER_CHARS = 180, 36
MAX_BYTES = 6_000_000
# Expected values, printed beside what this run computes. Group counts per sorting are from
# stories B_clutter.md (Stories 1, 4 and 6); the rest are read from the verdict's metrics.json below.
EXPECTED_COUNTS = {"L03N6372+": (15, 3), "L06N572+": (6, 1), "L12N8337+": (9, 2), "L02N4649-": (11, 5),
                   "L06N7759+": (3, None)}
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def groups_of(line):
    """The sorting's groups without the echo group (the study drops it from every count)."""
    return [c for c in line["clusters"] if not c["echo"]]


def find(exemplar, topk):
    """Ranks whose context, as rendered in the record shown to the sorting model, appears in `exemplar`."""
    text = exemplar.replace("↵", "\n")
    return [e["rank"] for e in topk if e["before"][-40:] + "⟦" + e["token"] + "⟧" + e["after"][:12] in text]


def gallery_check(gallery, sorted0, topk, shift):
    """Each exemplar the verdict quotes in q2.gallery must be found at exactly one rank, and that rank must
    lie in its group's ranks. `shift` moves every rank by that many places (a wrong join, to show the
    check failing)."""
    found = total = 0
    for g in gallery:
        groups = {f"{g['pole']}:c{c['c']}": c["exemplar_ranks"] for c in sorted0[g["pole"]]["clusters"]}
        for c in g["clusters"]:
            ranks = {(r + shift) % K for r in groups[c["cluster"]]}
            for ex in c["exemplars"]:
                hits = find(ex, topk[g["pole"]])
                total += 1
                found += len(hits) == 1 and hits[0] in ranks
    return found, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=Path, default=CLUSTERS)
    ap.add_argument("--t15", type=Path, default=T15)
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--draw", type=Path, default=DRAW)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # the record and its hash against the one the verdict recorded
    verdict = json.loads(a.verdict.read_text())
    q2 = verdict["q2"]
    sha = hashlib.sha256(a.clusters.read_bytes()).hexdigest()
    sha_rec = q2["inputs"]["clusters"]["sha256"]
    assert sha == sha_rec, f"{a.clusters} sha256 {sha} differs from q2.inputs.clusters.sha256 {sha_rec}"
    rows = read_jsonl(a.clusters)
    print("clusters.jsonl line keys:", sorted(rows[0]), "; group keys:", sorted(rows[0]["clusters"][0]))
    sortings = ({}, {})
    for r in rows:
        assert r["pole"] not in sortings[r["sample"]], f"{r['pole']} sample {r['sample']} twice"
        sortings[r["sample"]][r["pole"]] = r
    S = json.loads(a.draw.read_text())
    assert len(S) == 600 and set(sortings[0]) == set(sortings[1]) == set(S), "the record's neurons are not S"
    for s in (0, 1):
        for p, r in sortings[s].items():
            ranks = sorted(k for c in r["clusters"] for k in c["exemplar_ranks"])
            assert ranks == list(range(K)), f"{p} sorting {s + 1}: the groups do not hold ranks 0 to 49 once each"
            assert sum(c["echo"] for c in r["clusters"]) <= 1, f"{p} sorting {s + 1}: two echo groups"

    # the contexts at the end of training
    m15 = json.loads((a.t15 / "metrics.json").read_text())
    assert m15["complete"] is True, f"{a.t15} metrics.json complete = {m15['complete']}"
    ctx_rows = read_jsonl(a.t15 / "samples/contexts.jsonl")
    topk = {r["pole"]: r["topk"] for r in ctx_rows}
    assert len(ctx_rows) == 640 == len(topk) and set(S) <= set(topk), (len(ctx_rows), len(topk))
    for p in S:
        t = topk[p]
        sign = 1 if p.endswith("+") else -1
        assert len(t) == K and [e["rank"] for e in t] == list(range(K)), f"{p}: ranks are not 0 to 49 in order"
        assert all(e["tok"] >= 0 for e in t), f"{p}: an empty slot in the top 50"
        assert all(sign * x["a"] >= sign * y["a"] for x, y in zip(t, t[1:])), f"{p}: not in activation order"

    # the counts
    n_groups = [{p: len(groups_of(sortings[s][p])) for p in S} for s in (0, 1)]
    echo = [sum(c["echo"] for r in sortings[s].values() for c in r["clusters"]) for s in (0, 1)]
    echo_poles = sorted({p for s in (0, 1) for p, r in sortings[s].items() if any(c["echo"] for c in r["clusters"])})
    total = [sum(n.values()) for n in n_groups]
    same = sum(n_groups[0][p] == n_groups[1][p] for p in S)
    within1 = sum(abs(n_groups[0][p] - n_groups[1][p]) <= 1 for p in S)
    hist = [Counter(n.values()) for n in n_groups]
    top = max(max(h) for h in hist)

    say(f"Computed by `python3 extract.py` from `{a.clusters}` and `{a.t15}/samples/contexts.jsonl`.")
    say()
    say(f"- The record's sha256 is `{sha}`, the value at `q2.inputs.clusters.sha256` in `{a.verdict}`.")
    say(f"- {len(rows)} lines: 600 neurons (the 600 of S.json), each sorted twice. Every sorting puts each of "
        f"the 50 contexts in exactly one group.")
    say(f"- The contexts file is complete (`complete` true, {m15['n_tokens_scored']:,} tokens scored); every "
        f"study neuron has 50 contexts, ranks 0 to 49 in activation order. A group's `exemplar_ranks` are these ranks.")
    say()
    li = (n_groups[0][LITHIUM], n_groups[1][LITHIUM])
    say("| check | computed | expected |")
    say("|---|---:|---:|")
    say(f"| lithium neuron {LITHIUM}, groups under the first sorting | {li[0]} | 15 |")
    say(f"| lithium neuron {LITHIUM}, groups under the second sorting | {li[1]} | 3 |")
    for p, (e0, e1) in EXPECTED_COUNTS.items():
        if p == LITHIUM:
            continue
        say(f"| {p}, groups under the first and second sorting | {n_groups[0][p]} and {n_groups[1][p]} | "
            f"{e0} and {e1 if e1 is not None else '(not stated)'} |")
    say(f"| groups over the 600, first sorting (echo groups dropped) | {total[0]:,} | {q2['n_clusters']:,} "
        f"(q2.n_clusters) |")
    say(f"| groups over the 600, second sorting | {total[1]:,} | |")
    say(f"| echo groups, first and second sorting | {echo[0]} and {echo[1]} | {q2['echo_clusters']} "
        f"(q2.echo_clusters) |")
    say(f"| neurons whose two counts are equal | {same} of 600 | 182 (stories B_clutter.md Story 6) |")
    say(f"| neurons whose two counts are within one | {within1} of 600 = {within1 / 600:.4f} | "
        f"{q2['B4_m_within_1']:.4f} (q2.B4_m_within_1) |")
    one_both = [p for p in S if n_groups[0][p] == n_groups[1][p] == 1]
    say(f"| neurons with one group under both sortings | {len(one_both)} ({len(set(one_both) - set(echo_poles))} "
        f"without an echo group) | |")
    say(f"| mean groups per neuron, first and second sorting | {total[0] / 600:.3f} and {total[1] / 600:.3f} | "
        f"4.09 and 3.375 (stories B_clutter.md Story 6) |")

    # the join of ranks to contexts, against the exemplars the verdict quotes
    g_ok, g_n = gallery_check(q2["gallery"], sortings[0], topk, 0)
    g_bad, _ = gallery_check(q2["gallery"], sortings[0], topk, 1)
    say(f"| q2.gallery exemplars found at one rank inside their group (3 neurons) | {g_ok} of {g_n} | {g_n} of {g_n} |")
    say(f"| the same with every rank moved by one (a wrong join; must fail) | {g_bad} of {g_n} | well below {g_n} |")
    say()
    say("How many groups the 600 study neurons get (echo groups dropped):")
    say()
    say("| groups | " + " | ".join(str(k) for k in range(1, top + 1)) + " |")
    say("|---|" + "---:|" * top)
    for s in (0, 1):
        say(f"| {('first', 'second')[s]} sorting | " + " | ".join(str(hist[s].get(k, 0)) for k in range(1, top + 1)) + " |")
    say()

    differ = [f"lithium {li} (expected (15, 3))"] if li != (15, 3) else []
    differ += [f"{p} {n_groups[0][p]}, {n_groups[1][p]} (expected {e0}, {e1})" for p, (e0, e1) in EXPECTED_COUNTS.items()
               if n_groups[0][p] != e0 or (e1 is not None and n_groups[1][p] != e1)]
    differ += [f"total {total[0]} (expected {q2['n_clusters']})"] if total[0] != q2["n_clusters"] else []
    differ += [f"echo {echo} (expected {q2['echo_clusters']})"] if echo != [q2["echo_clusters"]] * 2 else []
    differ += [f"within one {within1 / 600} (expected {q2['B4_m_within_1']})"] if abs(within1 / 600 - q2["B4_m_within_1"]) > 1e-9 else []
    differ += [f"equal {same} (expected 182)"] if same != 182 else []
    differ += [f"gallery {g_ok} of {g_n}"] if g_ok != g_n else []
    differ += [f"the wrong join passed {g_bad} of {g_n}"] if g_bad > g_n // 2 else []
    say("Every expected number above matches, and the wrong join fails." if not differ
        else "These differ from the expected numbers: " + "; ".join(differ) + ".")
    say()

    # the neurons in data.js
    named = [p for p, _, _ in NAMED]
    assert len(set(named)) == len(named) and set(named) <= set(S), "a named neuron is repeated or not in S"
    assert not set(named) & set(echo_poles), f"a named neuron has an echo group: {set(named) & set(echo_poles)}"
    pool = sorted(set(S) - set(named) - set(echo_poles))
    rand = random.Random(SEED).sample(pool, N_TOTAL - len(NAMED))
    say(f"The picker holds the {len(NAMED)} examples in extract.py and `random.Random({SEED}).sample(pool, "
        f"{len(rand)})`, where `pool` is S.json minus the examples and the {len(echo_poles)} neurons with an "
        f"echo group, sorted as strings ({len(pool)} neurons): {', '.join(rand)}.")
    say()

    def neuron(p, kind, label, source):
        ctx = [[e["before"][-BEFORE_CHARS:], e["token"], e["after"][:AFTER_CHARS], float(f"{e['a']:.4g}")]
               for e in topk[p]]
        sorts = [[[c["name"], sorted(c["exemplar_ranks"])] for c in groups_of(sortings[s][p])] for s in (0, 1)]
        return {"id": p, "kind": kind, "label": label, "source": source, "ctx": ctx, "sortings": sorts}

    neurons = [neuron(p, "named", label, src) for p, label, src in NAMED]
    neurons += [neuron(p, "random", None, None) for p in rand]
    data = {"model": MODEL, "sorter": SORTER, "step": FINAL_STEP, "n_tokens_scored": m15["n_tokens_scored"],
            "k": K, "n_study": len(S), "context_tokens_before": 256,
            "hist": [[hist[s].get(k, 0) for k in range(1, top + 1)] for s in (0, 1)],
            "total": total, "same": same, "within1": within1, "neurons": neurons}
    out = a.out / "data.js"
    out.write_text("window.GROUPS_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, f"{out} is {size:,} bytes"
    say(f"Wrote `data.js`: {size:,} bytes, {len(neurons)} neurons, sha256 "
        f"`{hashlib.sha256(out.read_bytes()).hexdigest()}`.")
    say()
    e = topk[LITHIUM][0]
    show = lambda s: s.replace("\n", "⏎")
    g0 = next(c for c in groups_of(sortings[0][LITHIUM]) if 0 in c["exemplar_ranks"])
    g1 = next(c for c in groups_of(sortings[1][LITHIUM]) if 0 in c["exemplar_ranks"])
    say(f"The lithium neuron's strongest context (rank 0, activation {e['a']:.3g}, token `{show(e['token'])}`), in "
        f"the group \"{g0['name']}\" ({len(g0['exemplar_ranks'])} contexts) under the first sorting and "
        f"\"{g1['name']}\" ({len(g1['exemplar_ranks'])} contexts) under the second:")
    say()
    say("```")
    say(f"…{show(e['before'][-120:])}[{show(e['token'])}]{show(e['after'][:AFTER_CHARS])}")
    say("```")

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, rest = txt.split(CHECKS_START)
    readme.write_text(head + CHECKS_START + "\n" + "\n".join(lines) + "\n" + CHECKS_END + rest.split(CHECKS_END)[1])
    print(f"\nrewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
