"""Build data.js for the picture "One explanation and its test texts" from the banked W5 verdict.

Reads saved files only: the verdict's metrics.json, q1_O7_poles.jsonl (one line per side of each of 300
OLMo-3 7B neurons: the explanation, its test texts and the score of every text at the neuron and at a
random other neuron of the same layer), and the scoring job's scores.npz and tokens.json (the token where
each text peaked, and the text's tokens). Recomputes pass or fail and the class of every neuron side from
the stored scores, with the rule as one_job/extract.py ports it from infra/pg/neurons_verdict.py, checks
both against the stored values and metrics.json, writes data.js next to this file, prints the checks, and
rewrites the checks block of README.md with them.

    python3 extract.py
"""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Local copies on the laptop; README.md names the S3 source of each.
VERDICT = Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict")
PROBE = Path("/Users/abaybektursun/projects/interp_research/.pg_pulls/results/neurons/w2c/o7_probe/1/probe/poles")

# The rule, as the finished picture "How many neurons have one job" ports it (n_strong, explanation_passes,
# job_supported, assign_class, class_at, compact). Loaded from its file, not copied.
sys.dont_write_bytecode = True                  # leave no __pycache__ in one_job/
_spec = importlib.util.spec_from_file_location("one_job_extract", HERE.parent / "one_job" / "extract.py")
OJ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OJ)

KINDS = ("pos", "near", "distant")
THETA_BAND = OJ.THETA_BAND                  # 0.1: the registered stops are θ_s - 0.1, θ_s, θ_s + 0.1
STOP_FIELDS = OJ.STOP_FIELDS                # class_theta_minus, class, class_theta_plus
STOP_BANDS = OJ.STOP_BANDS                  # the same three in metrics.json q1.O7
SLIDER = (0.0, 1.2, 0.005)                  # the strong line's range on the page; positions within half a step of a stop take the stop
SCORE_DECIMALS = 6
N_DRAW = 19                                 # passes and fails each, drawn with random.Random(0)
MAX_BYTES = 4_000_000
GOT = "L03N7155+"                           # the "got" neuron, named in the walkthrough (lines 288, 622)
LITHIUM = "L03N6372+"                       # the "lithium" neuron, named in the walkthrough (lines 293, 622)
EXPLAINERS = OJ.EXPLAINERS
# Printed beside what this run computes (metrics.json q1.O7; VISUALS_PLAN.md P3 and P4).
EXPECTED_STOPS = OJ.EXPECTED_STOPS
EXPECTED_ONE = OJ.EXPECTED_ONE["O7"]        # 370, 311, 235
EXPECTED_STRONG = {"pos": (1794, 2995), "near": (112, 2995), "distant": (2, 2995), "shuffled": (23, 8985)}
EXPECTED_FAIL_STANDIN = "L25N5542+"
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


def sign_of(pole):
    assert pole[-1] in "+-", pole
    return pole[-1]


def char_spans(tokens, text):
    """The character span of each token in the text. Byte-level pieces that split one character decode to
    U+FFFD; a run of such pieces gets the span of the characters they cover together."""
    spans, off, i = [None] * len(tokens), 0, 0
    while i < len(tokens):
        if "�" not in tokens[i]:
            assert text.startswith(tokens[i], off), (text, tokens, i)
            spans[i] = (off, off + len(tokens[i]))
            off += len(tokens[i])
            i += 1
            continue
        j = i
        while j < len(tokens) and "�" in tokens[j]:
            j += 1
        end = text.find(tokens[j], off + 1) if j < len(tokens) else len(text)
        assert end > off, (text, tokens, i, j)
        for k in range(i, j):
            spans[k] = (off, end)
        off, i = end, j
    assert off == len(text), (text, tokens)
    return spans


def tally(scores15, theta):
    return tuple(OJ.n_strong(scores15[5 * k:5 * k + 5], theta) for k in range(3))


def passes(scores15, theta):
    return OJ.explanation_passes(scores15[0:5], scores15[5:10], scores15[10:15], theta)


def slider_positions(stops):
    lo, hi, step = SLIDER
    out = []
    for i in range(round((hi - lo) / step) + 1):
        v = round(lo + i * step, 3)
        near = [s for s in stops if abs(v - s) < step / 2]
        out.append(near[0] if near else v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--probe", type=Path, default=PROBE)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()
    lines, differ = [], []

    def say(s=""):
        print(s)
        lines.append(s)

    metrics = json.loads((a.verdict / "metrics.json").read_text())
    th = metrics["theta"]
    assert th["gate"] == "pass", th["gate"]
    theta = th["theta_s"]
    stops = (theta - THETA_BAND, theta, theta + THETA_BAND)
    assert tuple(round(s, 4) for s in stops) == EXPECTED_STOPS, stops
    m = metrics["q1"]["O7"]

    path = a.verdict / "samples" / "q1_O7_poles.jsonl"
    rows = [json.loads(s) for s in path.read_text().split("\n") if s.strip()]   # not splitlines: texts hold U+2028, U+2029, U+0085
    assert len(rows) == 600 and len({r["pole"] for r in rows}) == 600, len(rows)
    scored = [r for r in rows if "texts" in r]
    unscored = [r["pole"] for r in rows if "texts" not in r]
    assert all(set(r["texts"]) >= {*KINDS, "max"} and all(len(r["texts"][k]) == 5 for k in KINDS)
               and len(r["texts"]["max"]) == 1 for r in scored)

    # ---------- the scoring job's file: the one metrics.json names ----------
    npz = a.probe / "probe" / "scores.npz"
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    assert sha == m["inputs"]["scores"]["sha256"], (sha, m["inputs"]["scores"])
    with np.load(npz) as f:
        z = {k: f[k] for k in f.files}       # read each array once; indexing the NpzFile re-reads it
    tokens = json.loads((a.probe / "probe" / "tokens.json").read_text())
    probe_poles = json.loads((a.probe / "poles.json").read_text())
    assert z["max_a"].shape == (len(z["text_ids"]), len(probe_poles)), (z["max_a"].shape, len(probe_poles))
    tix = {str(t): i for i, t in enumerate(z["text_ids"])}
    pix = {p: j for j, p in enumerate(probe_poles)}
    say(f"scores.npz sha256 {sha} equals metrics.json q1.O7.inputs.scores.sha256; "
        f"{len(tix):,} texts x {len(pix):,} neuron sides")

    def probe_value(text_id, pole, shift=0):
        j = (pix[pole] + shift) % len(pix)
        return float(z["max_a"][tix[text_id], j]) if sign_of(pole) == "+" else -float(z["min_a"][tix[text_id], j])

    def peak(text_id, pole):
        i, j = tix[text_id], pix[pole]
        pos = int(z["argmax_tok"][i, j] if sign_of(pole) == "+" else z["argmin_tok"][i, j])
        n = int(z["n_tokens"][i])
        assert n == len(tokens[text_id]) and 1 <= pos < n, (text_id, pole, pos, n)   # position 0 is excluded (SPEC §2.2)
        return pos

    # instrument check: every stored score is the probe value over one number per neuron side (1 / max|a|).
    # Fed the neighbouring column, the same check must fail.
    def consistent(shift):
        pairs = {}
        for r in scored:
            for k in (*KINDS, "max"):
                for t in r["texts"][k]:
                    pairs.setdefault(r["pole"], []).append((t["score"], probe_value(t["text_id"], r["pole"], shift)))
            for t in r["shuffled"]:
                pairs.setdefault(r["shuffle_pole"], []).append((t["score"], probe_value(t["text_id"], r["shuffle_pole"], shift)))
        bad = 0
        for p, xs in pairs.items():
            big = max(xs, key=lambda x: abs(x[1]))
            c = big[0] / big[1]
            bad += any(abs(s - v * c) > 1e-6 * max(1.0, abs(s)) for s, v in xs)
        return len(pairs), bad
    n_sides, bad = consistent(0)
    assert bad == 0, f"{bad} neuron sides: stored scores are not one multiple of the probe values"
    _, bad_shift = consistent(1)
    assert bad_shift > 0, "the column check cannot fail"
    say(f"stored scores against scores.npz: one multiple per neuron side for all {n_sides} sides "
        f"(the same check on the neighbouring column fails for {bad_shift})")

    # ---------- stored strong flags, pass or fail, classes ----------
    bad = [r["pole"] for r in scored for t in [*(x for v in r["texts"].values() for x in v), *r["shuffled"]]
           if t["strong"] != (t["score"] >= theta)]
    assert not bad, bad[:5]
    assert all(r["shuffle_pole"] != r["pole"] and r["shuffle_pole"].split("N")[0] == r["pole"].split("N")[0]
               for r in scored), "a random neuron is not another neuron of the same layer"

    def own15(r):
        return [x["score"] for k in KINDS for x in r["texts"][k]]

    def shuf15(r):
        sc = {t["text_id"]: t["score"] for t in r["shuffled"]}
        return [sc[x["text_id"]] for k in KINDS for x in r["texts"][k]]

    mine = [passes(own15(r), theta) for r in scored]
    agree = sum(x == r["passes"] for x, r in zip(mine, scored))
    assert agree == len(scored), f"pass or fail differs from the stored value for {len(scored) - agree}"
    wrong = sum(passes(own15(r), stops[2]) != r["passes"] for r in scored)
    assert wrong > 0, "the pass check cannot fail"
    say(f"pass or fail recomputed at the strong line {theta:.4f}: agrees with the stored value for {agree} of "
        f"{len(scored)} neuron sides with texts ({len(unscored)} without texts: {', '.join(unscored)}); "
        f"fed {stops[2]:.4f} instead it disagrees for {wrong}")

    full = [OJ.compact(r, None) for r in rows]
    for s, field in zip(stops, STOP_FIELDS):
        off = [r["pole"] for n, r in zip(full, rows) if OJ.class_at(n, s) != r[field]]
        assert not off, (field, off[:5])
    say(f"class recomputed at {stops[0]:.4f}, {stops[1]:.4f} and {stops[2]:.4f}: equals the stored "
        f"class_theta_minus, class and class_theta_plus for all 600")

    say()
    say("| strong line | passes (this run) | passes in metrics.json | one job (this run) | one job in metrics.json | expected |")
    say("|---:|---:|---:|---:|---:|---:|")
    pass_at = []
    for s, band, e in zip(stops, STOP_BANDS, EXPECTED_ONE):
        k = sum(passes(own15(r), s) for r in scored)
        pass_at.append(k)
        one = sum(OJ.class_at(n, s) == "ONE" for n in full)
        mone = m[band]["ONE"]["k"]
        if s == theta:
            by_model = sum(v["pass_rate"]["k"] for v in m["by_explainer_model"].values())
            x7 = m["X7_pass_rate_by_encoding_flag"]["flagged"]["k"] + m["X7_pass_rate_by_encoding_flag"]["unflagged"]["k"]
            assert k == by_model == x7, (k, by_model, x7)
            mp = f"{by_model} (by_explainer_model) and {x7} (X7)"
        else:
            mp = "not stored"
        assert one == mone == e, (s, one, mone, e)
        say(f"| {s:.4f} | {k} of {len(scored)} | {mp} | {one} | {mone} | {e} |")
    say()

    say("| texts strong at the study's line | this run | metrics.json q1.O7.strong_rates | expected |")
    say("|---|---:|---:|---:|")
    names = {"pos": "texts written to fire it", "near": "near misses", "distant": "distant texts",
             "shuffled": "the same 15 texts at the random neuron"}
    nonrare = [r for r in scored if not r["rare"]]
    for k in ("pos", "near", "distant", "shuffled"):
        xs = [t["score"] for r in nonrare for t in (r["shuffled"] if k == "shuffled" else r["texts"][k])]
        got = (sum(x >= theta for x in xs), len(xs))
        mm = (m["strong_rates"][k]["k"], m["strong_rates"][k]["n"])
        assert got == mm == EXPECTED_STRONG[k], (k, got, mm)
        say(f"| {names[k]} | {got[0]:,} of {got[1]:,} | {mm[0]:,} of {mm[1]:,} | {EXPECTED_STRONG[k][0]:,} of {EXPECTED_STRONG[k][1]:,} |")
    rand_pass = sum(passes(shuf15(r), theta) for r in scored)
    say()
    say(f"Scored at the random neuron instead, {rand_pass} of {len(scored)} explanations pass at {theta:.4f} "
        f"(at their own neuron: {pass_at[1]}).")

    # ---------- which neurons the page holds ----------
    by = {r["pole"]: r for r in scored}
    for p in (GOT, LITHIUM):
        assert p in by and by[p]["passes"], p
    # the walkthrough's fail, L02N3919- of the Qwen3-4B discovery set, failed on its near misses with 5, 5
    # and 0 strong. It is not in these files. Its stand-in: the OLMo neuron side that fails on near misses
    # with the tally nearest 5, 5, 0; ties go to pole-id order.
    nm = sorted((r for r in scored if tally(own15(r), theta)[1] > 2),
                key=lambda r: (abs(5 - tally(own15(r), theta)[0]) + abs(5 - tally(own15(r), theta)[1]), r["pole"]))
    standin = nm[0]["pole"]
    if standin != EXPECTED_FAIL_STANDIN:
        differ.append(f"fail stand-in {standin} (expected {EXPECTED_FAIL_STANDIN})")
    examples = [GOT, standin, LITHIUM]
    rest = sorted(p for p in by if p not in examples)
    rng = random.Random(0)
    drawn_pass = rng.sample([p for p in rest if by[p]["passes"]], N_DRAW)
    drawn_fail = rng.sample([p for p in rest if not by[p]["passes"]], N_DRAW)

    positions = slider_positions(stops)
    assert all(s in positions for s in stops)

    def record(p, group):
        r = by[p]
        texts = [x for k in (*KINDS, "max") for x in r["texts"][k]]
        out_t, out_m = [], []
        for x in texts:
            sp = char_spans(tokens[x["text_id"]], x["text"])
            assert "".join(x["text"][a_:b_] for a_, b_ in dict.fromkeys(sp)) == x["text"]
            a_, b_ = sp[peak(x["text_id"], p)]
            out_t.append([x["text"], round(x["score"], SCORE_DECIMALS), a_, b_])
        sc = {t["text_id"]: t["score"] for t in r["shuffled"]}
        for x in texts[:15]:
            a_, b_ = char_spans(tokens[x["text_id"]], x["text"])[peak(x["text_id"], r["shuffle_pole"])]
            out_m.append([round(sc[x["text_id"]], SCORE_DECIMALS), a_, b_])
        rounded = [t[1] for t in out_t[:15]]
        rounded_r = [t[0] for t in out_m]
        for v in positions:                         # the rounding never changes pass or fail on the page
            assert passes(rounded, v) == passes(own15(r), v), (p, v)
            assert passes(rounded_r, v) == passes(shuf15(r), v), (p, v)
        return {"p": p, "g": group, "e": list(EXPLAINERS).index(r["explainer_model"]), "x": r["explanation"],
                "t": out_t, "r": {"p": r["shuffle_pole"], "t": out_m},
                "check": {"passes": r["passes"], "tally": list(tally(own15(r), theta))}}

    neurons = [record(GOT, 0), record(standin, 0), record(LITHIUM, 0)]
    for n, name in zip(neurons, ("the “got” neuron", "fails on its near misses", "the “lithium” neuron")):
        n["n"] = name                               # the picker's label for the three named sides
    neurons += [record(p, 1) for p in sorted(drawn_pass)] + [record(p, 2) for p in sorted(drawn_fail)]
    data = {"theta": theta, "stops": list(stops), "slider": list(SLIDER), "explainers": list(EXPLAINERS.values()),
            "groups": ["Examples", "Passes at the study’s line, drawn at random", "Fails at the study’s line, drawn at random"],
            "neurons": neurons}
    out = a.out / "data.js"
    out.write_text("window.EXPLANATION_TEST_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, size
    say()
    say(f"data.js: {size:,} bytes, {len(neurons)} neuron sides: {GOT} (the got neuron), {standin} (the fail stand-in), "
        f"{LITHIUM} (the lithium neuron), {N_DRAW} passes and {N_DRAW} fails drawn with random.Random(0) from the other "
        f"{len(rest)} sides with texts. Scores at {SCORE_DECIMALS} decimals give the same pass or fail as full precision "
        f"at all {len(positions)} positions of the strong line ({SLIDER[0]} to {SLIDER[1]} in steps of {SLIDER[2]}), "
        f"at the neuron and at the random neuron.")

    # ---------- samples: one scored pass and one scored fail, with the marked token ----------
    def show(n):
        r = by[n["p"]]
        k = tally([t[1] for t in n["t"][:15]], theta)
        say(f"{n['p']}, explanation by {EXPLAINERS[r['explainer_model']]}, {'passes' if r['passes'] else 'fails'} at "
            f"{theta:.4f} with {k[0]}, {k[1]} and {k[2]} strong; random neuron {r['shuffle_pole']}:")
        say()
        say(f"> {r['explanation']}")
        say()
        for i, kind in ((0, "pos"), (1, "pos"), (5, "near"), (6, "near"), (10, "distant"), (15, "max")):
            text, s, a_, b_ = n["t"][i]
            rs = f"; at the random neuron {n['r']['t'][i][0]:.4f}" if i < 15 else ""
            say(f"- {kind}, score {s:.4f}{' (strong)' if s >= theta else ''}{rs}: "
                f"{text[:a_]}⟦{text[a_:b_]}⟧{text[b_:]}".replace("\n", "⏎"))
        say()
    say()
    say("Two of the page's neuron sides, with the token where each text peaked marked ⟦ ⟧:")
    say()
    show(neurons[0])
    show(neurons[1])
    say("Every expected number above matches." if not differ else "These differ from the expected: " + "; ".join(differ) + ".")

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, tail = txt.split(CHECKS_START)
    block = f"Computed by `python3 extract.py` from `{a.verdict}` and `{a.probe}`.\n\n" + "\n".join(lines)
    readme.write_text(head + CHECKS_START + "\n" + block + "\n" + CHECKS_END + tail.split(CHECKS_END)[1])
    print(f"\nrewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
