"""Build data.js for the page "One neuron across training" from the banked W3 harvest.

Reads saved JSON only: the 16 grid steps' samples/contexts.jsonl and metrics.json, the W5 verdict
metrics.json (the 17 poles at q2.H_MIG.poles) and the W3 draw (S.json, S_gate.json). Writes data.js
next to this file, prints the checks, and rewrites the checks block of README.md with them. Prints
`path sha256` for every input; the same pairs go into data.js under meta.sources (BUILD.md §2).

    python3 extract.py
"""
import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The local copy on the laptop; README.md names the S3 source of each.
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
HARVEST = BANK / "w3/grid"
VERDICT = BANK / "w5/verdict/metrics.json"
DRAW = BANK / "w3/o7/draw"
# BANK is the laptop copy of this S3 prefix (same relative paths, sizes checked 2026-11-08). The two draw files
# were rebuilt on the laptop and are on no S3 path (README.md, Inputs), so they keep their local path.
S3_BANK = "s3://pg-lineage-769063704696/results/neurons"
LOCAL_ONLY = {DRAW / "S.json", DRAW / "S_gate.json"}
# meta.commit: the public repository's commit that holds this folder (abaybektursun/already-clean-audit main,
# a copy of borrowed-geometry clean-neurons 512951ce9 with equal blob hashes). The inputs are not in that
# repository, so the page's data link opens this folder's README.md, which names every input.
PUBLIC_COMMIT = "2e43474ae1bbfa353e222993cb706283a47bee7b"
LINK = "studies/already-clean/walkthrough/neuron_across_training/README.md"

MODEL = "OLMo-3 7B"
LITHIUM, LITHIUM_TOKEN = "L03N6372+", " lithium"
FINAL_STEP = 1413814
TOP, K = 20, 50
N_RANDOM, SEED = 24, 0
# The decoded strings carry no token boundaries, so "40 tokens of text before" and "8 after" are cut
# as characters at 4.5 characters per token.
BEFORE_CHARS, AFTER_CHARS = 180, 36
MAX_BYTES = 10_000_000
# From an earlier read of the same files (the brief); printed beside what this run computes.
EXPECTED_COUNT20 = {"t09": 20, "t10": 19, "t11": 19, "t12": 19, "t14": 3}
EXPECTED_COUNT50_T15 = 0
EXPECTED_MAX = {"t03": 146, "t09": 5.46, "t15": 0.246}
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


def sig3(x):
    return float(f"{x:.3g}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source(path):
    """{path, sha256} for one input: its S3 URI when it is the copy of an S3 object, else its local path."""
    path = path.resolve()
    shown = str(path) if path in LOCAL_ONLY else f"{S3_BANK}/{path.relative_to(BANK)}"
    return {"path": shown, "sha256": sha256(path)}


def step_dirs(harvest):
    found = {}
    for sub in ("1", "1b"):
        for d in sorted((harvest / sub / "out").iterdir()):
            m = re.fullmatch(r"(t\d\d)_step(\d+)", d.name)
            if m:
                assert m[1] not in found, f"{m[1]} appears twice: {found.get(m[1])} and {d}"
                found[m[1]] = (int(m[2]), d)
    names = [f"t{i:02d}" for i in range(16)]
    assert sorted(found) == names, f"expected t00..t15, found {sorted(found)}"
    assert found["t09"][0] == 60000 and found["t15"][0] == FINAL_STEP, (found["t09"], found["t15"])
    return [(t, *found[t]) for t in names]


def ordered(pole, topk):
    """Valid top-k entries in the pole's sign order (padding slots, tok -1 or a NaN, dropped)."""
    valid = [e for e in topk if e["tok"] >= 0 and not math.isnan(e["a"])]
    sign = 1 if pole.endswith("+") else -1
    return sorted(valid, key=lambda e: -sign * e["a"]), len(topk) - len(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", type=Path, default=HARVEST)
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--draw", type=Path, default=DRAW)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()

    steps = step_dirs(a.harvest)
    n_scored = set()
    lines = {}                                    # tNN -> {pole: line}
    for t, step, d in steps:
        m = json.loads((d / "metrics.json").read_text())
        assert m["complete"] is True, f"{d} metrics.json complete = {m['complete']}"
        n_scored.add(m["n_tokens_scored"])
        rows = [json.loads(s) for s in (d / "samples/contexts.jsonl").read_text().splitlines() if s.strip()]
        assert len(rows) == 640, f"{d}: {len(rows)} lines"
        lines[t] = {r["pole"]: r for r in rows}
        assert len(lines[t]) == 640, f"{d}: repeated pole ids"
        print(f"{t} step {step:>9,}  complete  n_tokens_scored {m['n_tokens_scored']:,}  {d}")
    assert len(n_scored) == 1, f"n_tokens_scored differs across steps: {n_scored}"
    n_scored = n_scored.pop()
    first = lines["t00"][next(iter(lines["t00"]))]
    print("contexts.jsonl line keys:", sorted(first))
    print("topk entry keys:", sorted(first["topk"][0]))
    all_poles = set(lines["t00"])
    assert all(set(v) == all_poles for v in lines.values()), "pole set differs across steps"

    verdict = json.loads(a.verdict.read_text())
    mig = verdict["q2"]["H_MIG"]["poles"]
    print(f"late group: {a.verdict} key q2.H_MIG.poles, {len(mig)} poles, lithium in it: {LITHIUM in mig}")
    S = json.loads((a.draw / "S.json").read_text())
    S_gate = json.loads((a.draw / "S_gate.json").read_text())
    assert len(S) == 600 and len(S_gate) == 40 and not set(S) & set(S_gate), (len(S), len(S_gate))
    assert set(S) | set(S_gate) == all_poles, "S and S_gate do not make up the 640 harvested poles"
    assert LITHIUM in all_poles and set(mig) <= all_poles
    print(f"study set: {a.draw}/S.json (600); gate set S_gate.json (40); lithium in S: {LITHIUM in S}")
    late = [p for p in mig if p != LITHIUM]
    pool = sorted(set(S) - {LITHIUM} - set(late))
    rand = random.Random(SEED).sample(pool, N_RANDOM)
    print(f"random group: random.Random({SEED}).sample(sorted S minus lithium and the late group "
          f"({len(pool)} poles), {N_RANDOM}): {rand}")

    # max |a| over the 50 against the line's own max_abs, every pole and step (instrument check)
    off, n_padded = [], 0
    for t in lines:
        for p, r in lines[t].items():
            ents, pad = ordered(p, r["topk"])
            n_padded += pad
            if abs(max(abs(e["a"]) for e in ents) - r["max_abs"]) > 1e-3 * r["max_abs"]:
                off.append((t, p))
    print(f"padding slots (tok -1 or a NaN) dropped: {n_padded} over {16 * 640} lines")
    print(f"max |a| over the top-50 differs from the line's max_abs by > 0.1% at {len(off)} of "
          f"{16 * 640} (pole, step) pairs {off[:5]}")

    neurons = []
    for group, poles in (("lithium", [LITHIUM]), ("late", late), ("random", rand)):
        for p in poles:
            ctx, index, text, per_step = [], {}, {}, []
            for t, _, _ in steps:
                ents = ordered(p, lines[t][p]["topk"])[0]
                assert len(ents) <= K, (p, t, len(ents))
                top = []
                for e in ents[:TOP]:
                    full = (e["before"], e["token"], e["after"])
                    assert text.setdefault(e["tok"], full) == full, f"{p} tok {e['tok']}: text differs across steps"
                    if e["tok"] not in index:
                        index[e["tok"]] = len(ctx)
                        ctx.append([e["before"][-BEFORE_CHARS:], e["token"], e["after"][:AFTER_CHARS]])
                    top.append([index[e["tok"]], float(f"{e['a']:.4g}")])
                counts = Counter(e["token"] for e in ents[:TOP]).most_common()
                per_step.append({"top": top, "max": float(f"{max(abs(e['a']) for e in ents):.4g}"),
                                 "counts": [[tok, n] for tok, n in counts]})
            neurons.append({"pole": p, "group": group, "ctx": ctx, "steps": per_step})

    inputs = [p for _, _, d in steps for p in (d / "samples/contexts.jsonl", d / "metrics.json")]
    inputs += [a.verdict, a.draw / "S.json", a.draw / "S_gate.json"]
    sources = [source(p) for p in inputs]
    for s in sources:
        print(s["path"], s["sha256"])
    data = {"model": MODEL, "n_tokens_scored": n_scored, "k": K, "top": TOP,
            "steps": [{"t": t, "step": s, "pct": round(100 * s / FINAL_STEP, 4)} for t, s, _ in steps],
            "neurons": neurons,
            "meta": {"commit": PUBLIC_COMMIT, "link": LINK, "sources": sources}}
    out = a.out / "data.js"
    out.write_text("window.NEURON_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, f"{out} is {size:,} bytes"
    n_ctx = sum(len(n["ctx"]) for n in neurons)
    n_slots = sum(len(s["top"]) for n in neurons for s in n["steps"])
    print(f"wrote {out}: {size:,} bytes; {len(neurons)} poles; {n_slots:,} top-20 slots; {n_ctx:,} distinct "
          f"contexts")

    # lithium checks
    li = {t: ordered(LITHIUM, lines[t][LITHIUM]["topk"])[0] for t, _, _ in steps}
    c20 = {t: sum(e["token"] == LITHIUM_TOKEN for e in li[t][:TOP]) for t in li}
    c50 = {t: sum(e["token"] == LITHIUM_TOKEN for e in li[t]) for t in li}
    mx = {t: max(abs(e["a"]) for e in li[t]) for t in li}
    table = ["| point | step | ' lithium' in top 20 | ' lithium' in top 50 | strongest activation | expected |",
            "|---|---:|---:|---:|---:|---|"]
    differ = []
    for t, s, _ in steps:
        exp = []
        if t in EXPECTED_COUNT20:
            exp.append(f"{EXPECTED_COUNT20[t]} of 20")
            differ += [f"{t} top-20 count {c20[t]} (expected {EXPECTED_COUNT20[t]})"] if c20[t] != EXPECTED_COUNT20[t] else []
        if t == "t15":
            exp.append(f"{EXPECTED_COUNT50_T15} of 50")
            differ += [f"t15 top-50 count {c50[t]} (expected 0)"] if c50[t] != EXPECTED_COUNT50_T15 else []
        if t in EXPECTED_MAX:
            exp.append(f"max about {EXPECTED_MAX[t]}")
            differ += [f"{t} max {sig3(mx[t]):g} (expected about {EXPECTED_MAX[t]})"] if sig3(mx[t]) != EXPECTED_MAX[t] else []
        table.append(f"| {t} | {s:,} | {c20[t]} of {min(TOP, len(li[t]))} | {c50[t]} of {len(li[t])} | {sig3(mx[t]):g} | {', '.join(exp)} |")

    def quote(label, t):
        e = li[t][0]
        show = lambda s: s.replace("\n", "⏎")
        return [f"Top context {label} ({t}), rank 1, activation {sig3(e['a']):g}, token `{show(e['token'])}`:", "",
                "```", f"…{show(e['before'][-120:])}[{show(e['token'])}]{show(e['after'])}", "```", ""]
    verdict_line = ("Every expected number above matches." if not differ
                    else "These differ from the expected numbers: " + "; ".join(differ) + ".")
    block = "\n".join([f"Computed by `python3 extract.py` from `{a.harvest}`.", "", *table, "", verdict_line, "",
                       *quote("at step 60,000", "t09"), *quote("at the final step", "t15")])
    print("\n" + block + "\n")

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, rest = txt.split(CHECKS_START)
    readme.write_text(head + CHECKS_START + "\n" + block + "\n" + CHECKS_END + rest.split(CHECKS_END)[1])
    print(f"rewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
