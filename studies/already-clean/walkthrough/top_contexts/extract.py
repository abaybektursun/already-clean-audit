"""Build data.js for the picture "What a top context is" from the banked end-of-training harvest.

Reads saved files only: the final step's samples/contexts.jsonl and metrics.json, its poles/exemplar_a.npz,
and the study set S.json. Writes data.js next to this file, prints the checks, and rewrites the checks
block of README.md with them.

    python3 extract.py
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# The local copy on the laptop; README.md names the S3 source of each.
BANK = Path("/Users/abaybektursun/projects/interp_research_banks/neurons")
FINAL = BANK / "w3/grid/1/out/t15_step1413814"
DRAW = BANK / "w3/o7/draw/S.json"

MODEL, STEP = "OLMo-3 7B", 1413814
LITHIUM, GOT = "L03N6372+", "L03N7155+"
K, N_RANDOM, SEED = 50, 18, 0
WINDOW = 2048                      # harvest window (infra/pg/neurons_harvest.py default, DATA_CONTRACT §2)
BEFORE_CHARS, AFTER_CHARS = 180, 36
MAX_BYTES = 2_000_000
# Expected values from the brief and VISUALS_PLAN.md P1; printed beside what this run computes.
EXPECTED = {"n_tokens_scored": 20560068, "lithium max": 0.246, "got max": 0.663, "got q999": 0.050}
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


def sig(x, n=4):
    return float(f"{x:.{n}g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, default=FINAL)
    ap.add_argument("--draw", type=Path, default=DRAW)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    out_lines = []

    def say(s=""):
        print(s)
        out_lines.append(s)

    m = json.loads((a.final / "metrics.json").read_text())
    assert m["complete"] is True, f"metrics.json complete = {m['complete']}"
    n_scored = m["n_tokens_scored"]
    say(f"- `n_tokens_scored` {n_scored:,} (expected {EXPECTED['n_tokens_scored']:,}): "
        f"{'match' if n_scored == EXPECTED['n_tokens_scored'] else 'DIFFERS'}; `complete` true.")
    assert n_scored % (WINDOW - 1) == 0, f"{n_scored} is not a whole number of {WINDOW}-token windows"
    n_windows = n_scored // (WINDOW - 1)
    n_stream = n_windows * WINDOW
    say(f"- The stream: {n_windows:,} windows of {WINDOW:,} tokens = {n_stream:,} corpus positions; the first "
        f"position of each window is not scored, which leaves {n_windows:,} x {WINDOW - 1:,} = {n_scored:,}.")

    rows = [json.loads(s) for s in (a.final / "samples/contexts.jsonl").read_text().splitlines() if s.strip()]
    lines = {r["pole"]: r for r in rows}
    assert len(rows) == 640 and len(lines) == 640, (len(rows), len(lines))
    keys = sorted(rows[0]["topk"][0])
    say(f"- contexts.jsonl: {len(rows)} lines, one per neuron side. Entry fields: {', '.join(keys)}.")
    has_pos = all(f in rows[0]["topk"][0] for f in ("tok", "doc", "pos"))
    say(f"- Positions exist: {'yes' if has_pos else 'NO'}. `tok` is the position in the corpus token stream, "
        f"`doc` the document, `pos` the position inside the document (DATA_CONTRACT §0, §2).")
    assert has_pos

    # every side: 50 valid entries, rank order, sign order, max_abs at rank 0
    n_full = sum(len(r["topk"]) == K and all(e["tok"] >= 0 and not math.isnan(e["a"]) for e in r["topk"])
                 for r in rows)
    say(f"- Sides with {K} valid contexts (no `tok` -1, no NaN): {n_full} of {len(rows)}.")
    assert n_full == len(rows)
    for r in rows:
        t = r["topk"]
        sgn = 1 if r["pole"].endswith("+") else -1
        assert [e["rank"] for e in t] == list(range(K)), r["pole"]
        assert all(sgn * t[i]["a"] >= sgn * t[i + 1]["a"] for i in range(K - 1)), f"{r['pole']} not in sign order"
        assert all(sgn * e["a"] > 0 for e in t), f"{r['pole']}: an entry of the wrong sign"
        assert abs(abs(t[0]["a"]) - r["max_abs"]) <= 1e-6 * r["max_abs"], f"{r['pole']}: rank 0 is not max_abs"
    say("- Every side's 50 are in rank order 0 to 49, sorted in the side's sign, all of that sign, and rank 0 "
        "equals the line's `max_abs`.")

    toks = np.array([e["tok"] for r in rows for e in r["topk"]], dtype=np.int64)
    docs = np.array([e["doc"] for r in rows for e in r["topk"]], dtype=np.int64)
    n_past = int((toks >= n_scored).sum())
    say(f"- Largest `tok` {toks.max():,}, smallest {toks.min():,}. Every `tok` < {n_stream:,} (the stream): "
        f"{'yes' if toks.max() < n_stream else 'NO'}. None at a window's first position (`tok` % {WINDOW} == 0): "
        f"{'yes' if not (toks % WINDOW == 0).any() else 'NO'}.")
    say(f"- The plan's assert \"every tok < 20,560,068\" fails for {n_past} of {toks.size:,} entries: `tok` "
        f"indexes the {n_stream:,}-position stream, not the count of scored tokens. The picture's axis is the stream.")
    assert toks.max() < n_stream and not (toks % WINDOW == 0).any()
    order = np.argsort(toks, kind="stable")
    mono = bool((np.diff(docs[order]) >= 0).all())
    say(f"- `doc` never decreases as `tok` increases, over all {toks.size:,} entries: {'yes' if mono else 'NO'}.")
    assert mono
    same_tok_text = {}
    for r in rows:
        for e in r["topk"]:
            full = (e["before"], e["token"], e["after"])
            assert same_tok_text.setdefault(e["tok"], full) == full, f"tok {e['tok']}: text differs across sides"
    say(f"- {len(same_tok_text):,} distinct positions; a position that is in several sides' top 50 decodes to "
        f"the same text in each.")

    z = np.load(a.final / "poles/exemplar_a.npz")
    A = z["a"]
    eq = sum(np.allclose(A[i], [e["a"] for e in r["topk"]], rtol=1e-5, atol=1e-7) for i, r in enumerate(rows))
    say(f"- exemplar_a.npz: `a` {A.shape}; row i equals line i's 50 activations for {eq} of {len(rows)}. It holds "
        f"only the top-50 positions, so the record has no activations at random positions; the picture shows "
        f"the neuron's 99.9th-percentile line instead.")
    assert A.shape == (640, K) and eq == len(rows)

    S = json.loads(a.draw.read_text())
    assert len(S) == 600 and set(S) <= set(lines) and LITHIUM in S and GOT in S
    pool = sorted(set(S) - {LITHIUM, GOT})
    rand = random.Random(SEED).sample(pool, N_RANDOM)
    say(f"- Random neurons: `random.Random({SEED}).sample(pool, {N_RANDOM})`, pool = S.json ({len(S)}) minus "
        f"{LITHIUM} and {GOT}, sorted ({len(pool)}): {', '.join(rand)}.")

    neurons = []
    for group, poles in (("lithium", [LITHIUM]), ("got", [GOT]), ("random", rand)):
        for p in poles:
            r = lines[p]
            ctx = [[e["tok"], e["doc"], sig(e["a"]), e["before"][-BEFORE_CHARS:], e["token"], e["after"][:AFTER_CHARS]]
                   for e in r["topk"]]
            neurons.append({"pole": p, "group": group, "max": sig(r["max_abs"]), "q999": sig(r["q999"]),
                            "density": r["density"], "ctx": ctx})

    data = {"model": MODEL, "step": STEP, "n_tokens_scored": n_scored, "window": WINDOW, "n_windows": n_windows,
            "n_stream": n_stream, "k": K, "neurons": neurons}
    out = a.out / "data.js"
    out.write_text("window.TOP_CONTEXTS_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, f"{out} is {size:,} bytes"
    say(f"- Wrote data.js: {size:,} bytes, {len(neurons)} neurons x {K} contexts.")

    li, got = lines[LITHIUM], lines[GOT]
    for name, r, key in (("lithium", li, "lithium max"), ("got", got, "got max")):
        v = sig(r["max_abs"], 3)
        say(f"- The {name} neuron ({r['pole']}): strongest activation {v:g} (expected {EXPECTED[key]}): "
            f"{'match' if v == EXPECTED[key] else 'DIFFERS'}; 99.9th percentile {r['q999']:.3f}; "
            f"50th strongest {sig(r['topk'][-1]['a'], 3):g}; {len({e['doc'] for e in r['topk']})} documents.")
    q = sig(got["q999"], 2)
    say(f"- The got neuron's 99.9th percentile {q:.3f} (expected {EXPECTED['got q999']:.3f}): "
        f"{'match' if q == EXPECTED['got q999'] else 'DIFFERS'}. It is the upper edge of a histogram bin "
        f"(resolution 5.9%, DATA_CONTRACT §2), so at least 99.9% of the side's tokens lie below it.")
    n_li = sum(e["token"] == " lithium" for e in li["topk"])
    say(f"- ' lithium' among the lithium neuron's final 50: {n_li}.")
    top = Counter(e["token"] for e in got["topk"]).most_common(3)
    say(f"- The got neuron's most common marked tokens: {', '.join(f'{t!r} {c}' for t, c in top)}.")

    e = li["topk"][0]
    show = lambda s: s.replace("\n", "⏎")
    say()
    say(f"Lithium neuron, rank 1, activation {sig(e['a'], 3):g}, position {e['tok']:,}, document {e['doc']:,}:")
    say()
    say("```")
    say(f"…{show(e['before'][-120:])}[{show(e['token'])}]{show(e['after'])}")
    say("```")

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, rest = txt.split(CHECKS_START)
    block = "\n".join([f"Computed by `python3 extract.py` from `{a.final}`.", "", *out_lines])
    readme.write_text(head + CHECKS_START + "\n" + block + "\n" + CHECKS_END + rest.split(CHECKS_END)[1])
    print(f"rewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
