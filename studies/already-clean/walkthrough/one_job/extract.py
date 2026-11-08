"""Build data.js for the picture "How many neurons have one job" from the banked W5 verdict.

Reads saved JSON only: the verdict's metrics.json and the three per-neuron sample files
q1_{Q4B,G4,O7}_poles.jsonl (one line per pole: the explainer's label, jobs and explanation, and the
score of every test text). Recomputes each pole's class from the stored scores with a port of the rule
in infra/pg/neurons_verdict.py, checks it against the stored classes and metrics.json, writes data.js
next to this file, prints the checks, and rewrites the checks block of README.md with them.

    python3 extract.py
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The local copy on the laptop; README.md names the S3 source.
VERDICT = Path("/Users/abaybektursun/projects/interp_research_banks/neurons/w5/verdict")

MODELS = (("Q4B", "Qwen3-4B base"), ("G4", "Gemma-3 4B"), ("O7", "OLMo-3 7B"))
EXPLAINERS = {"anthropic/claude-opus-5-5": "Opus 5.5", "anthropic/claude-sonnet-5-5": "Sonnet 5.5"}
THETA_BAND = 0.1                     # neurons_verdict.THETA_BAND: the width control, classes at θ_s ± 0.1
CLASSES = ("ONE", "TWO-REL", "UNREL", "UNK", "RARE")
STOP_FIELDS = ("class_theta_minus", "class", "class_theta_plus")      # stored per pole, at θ_s - 0.1, θ_s, θ_s + 0.1
STOP_BANDS = ("band_theta_minus_0.1", "fractions", "band_theta_plus_0.1")   # the same three in metrics.json q1.<model>
# The slider: min, max, step. Positions within half a step of a stop take the stop's exact value.
SLIDER = (0.05, 0.80, 0.005)
SCORE_DECIMALS = 6                   # scores in data.js; the script fails if rounding changes any class on the slider
ESSAY_ONE, N_POLES = 540, 600        # the essay's "about 90%" of 600
MAX_BYTES = 6_000_000
# From the brief and VISUALS_PLAN.md P4 (RESULTS.md Q1); printed beside what this run computes.
EXPECTED_STOPS = (0.2153, 0.3153, 0.4153)
EXPECTED_ONE = {"Q4B": (383, 358, 313), "G4": (373, 338, 294), "O7": (370, 311, 235)}
EXPECTED_SHUFFLED_STRONG = {"Q4B": 48, "G4": 26, "O7": 23}
EXPECTED_LABEL_ONE = {"Q4B": 525, "G4": 520, "O7": 521}               # E_skeptic.md D9
CHECKS_START, CHECKS_END = "<!-- checks:start -->", "<!-- checks:end -->"


# --- the rule, ported from infra/pg/neurons_verdict.py (n_strong, explanation_passes, job_supported, assign_class)

def n_strong(scores, threshold):
    return sum(s >= threshold for s in scores)


def explanation_passes(pos, near, distant, theta):
    assert len(pos) == len(near) == len(distant) == 5, (len(pos), len(near), len(distant))
    return n_strong(pos, theta) >= 3 and n_strong(near, theta) <= 2 and n_strong(distant, theta) <= 1


def job_supported(job_scores, theta):
    assert len(job_scores) == 5, len(job_scores)
    return n_strong(job_scores, theta) >= 3


def assign_class(rare, label, related, passes, job_support):
    """related[k] is jobs[k]['related_to_others']; job_support(k) is the support clause for job k."""
    if rare:
        return "RARE"
    if label == "ONE" and len(related) == 1 and passes:
        return "ONE"
    if label == "TWO-REL" and 2 <= len(related) <= 3 and all(related) and passes:
        return "TWO-REL"
    if label == "UNREL":
        unrelated = [k for k, r in enumerate(related) if not r]
        if len(unrelated) >= 2 and sum(job_support(k) for k in unrelated) >= 2:
            return "UNREL"
    return "UNK"


def class_at(n, theta):
    """The class of one pole at a strong line, from the compact record written to data.js (the same
    record the page reads). As in neurons_verdict.q1_records: a pole with no texts (never fired on its
    side, or an agent declined) is RARE if rare, else UNK."""
    if n["s"] is None:
        return "RARE" if n["rare"] else "UNK"
    s = n["s"]
    passes = explanation_passes(s[0:5], s[5:10], s[10:15], theta)
    return assign_class(n["rare"], n["l"], n["rel"], passes, lambda k: job_supported(n["u"][k], theta))


def compact(row, decimals):
    """The fields the page needs, scores rounded to `decimals` (None: as stored).
    f: 0 scored, 1 rare, 2 explainer declined, 3 writer declined."""
    rel = [bool(j["related_to_others"]) for j in row["jobs"]]
    d = row["declined"]
    f = 1 if row["rare"] else 0 if not d else 2 if d.startswith("explainer declined") else 3 if d.startswith("writer declined") else None
    assert f is not None, (row["pole"], d)
    s = u = None
    if "texts" in row:
        t = row["texts"]
        assert [x["text_id"].split(":")[1] for k in ("pos", "near", "distant") for x in t[k]] == \
            ["pos"] * 5 + ["near"] * 5 + ["distant"] * 5, row["pole"]
        rd = (lambda x: x) if decimals is None else (lambda x: round(x, decimals))
        s = [rd(x["score"]) for k in ("pos", "near", "distant") for x in t[k]]
        u = {int(k.split(":")[1]): [rd(x["score"]) for x in v] for k, v in t.items() if k.startswith("job_pos:")}
    else:
        assert row["rare"] or d, f"{row['pole']}: no texts, not rare and not declined"
    return {"p": row["pole"], "l": row["label"], "rel": rel, "rare": row["rare"], "f": f, "s": s, "u": u,
            "e": row["explainer_model"], "x": row["explanation"]}


def slider_positions(stops):
    lo, hi, step = SLIDER
    out = []
    for i in range(round((hi - lo) / step) + 1):
        v = round(lo + i * step, 3)
        near = [s for s in stops if abs(v - s) < step / 2]
        out.append(near[0] if near else v)
    return out


def tally(classes):
    c = Counter(classes)
    assert set(c) <= set(CLASSES), c
    return {k: c.get(k, 0) for k in CLASSES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", type=Path, default=VERDICT)
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args()

    metrics = json.loads((a.verdict / "metrics.json").read_text())
    th = metrics["theta"]
    assert th["gate"] == "pass", th["gate"]
    theta = th["theta_s"]
    stops = (theta - THETA_BAND, theta, theta + THETA_BAND)
    assert tuple(round(s, 4) for s in stops) == EXPECTED_STOPS, stops
    print(f"θ_s {theta!r} from {a.verdict / 'metrics.json'} key theta.theta_s (gate {th['gate']}, "
          f"{th['n_scores']} shuffled scores, {th['percentile']}th percentile); stops {[round(s, 6) for s in stops]}")
    positions = slider_positions(stops)
    assert all(s in positions for s in stops)

    out_models, lines, differ = [], [], []
    stop_rows, extra_rows, samples = [], [], []
    for key, name in MODELS:
        path = a.verdict / "samples" / f"q1_{key}_poles.jsonl"
        rows = [json.loads(s) for s in path.read_text().split("\n") if s.strip()]   # not splitlines: texts hold U+2028, U+2029, U+0085
        poles = [r["pole"] for r in rows]
        assert len(rows) == N_POLES and len(set(poles)) == N_POLES, (path, len(rows), len(set(poles)))
        sides = Counter(p[:-1] for p in poles)
        assert len(sides) == N_POLES // 2 and set(sides.values()) == {2}, f"{path}: not 300 neurons x 2 sides"
        m = metrics["q1"][key]
        # Q4B and G4 use every pole of their explain directory (poles_list null); O7 uses a list of 600
        excluded = set(m["poles_list"]["excluded_by_list"]["poles"]) if m["poles_list"] is not None else set()
        assert not excluded & set(poles), f"{path}: holds poles the audit list leaves out"
        print(f"{key} {name}: {path} ({path.stat().st_size:,} bytes), {len(rows)} poles = 300 neurons x 2 sides"
              + (f"; none of the {len(excluded)} poles the audit list leaves out" if excluded else ""))

        # instrument check: each stored strong flag is score >= θ_s
        bad = [r["pole"] for r in rows if "texts" in r for t in [*(x for v in r["texts"].values() for x in v), *r["shuffled"]]
               if t["strong"] != (t["score"] >= theta)]
        assert not bad, f"{key}: stored strong flags disagree with score >= θ_s at {bad[:5]}"
        unknown = set(r["explainer_model"] for r in rows) - set(EXPLAINERS)
        assert not unknown, unknown

        full = [compact(r, None) for r in rows]        # full-precision scores
        small = [compact(r, SCORE_DECIMALS) for r in rows]

        # the class at the three stops, recomputed, against the stored per-pole classes and metrics.json
        per_stop = []
        for s, field, band in zip(stops, STOP_FIELDS, STOP_BANDS):
            mine = [class_at(n, s) for n in full]
            stored = [r[field] for r in rows]
            off = [(r["pole"], x, y) for r, x, y in zip(rows, mine, stored) if x != y]
            assert not off, f"{key} at {s}: {len(off)} classes differ from {field}: {off[:5]}"
            t = tally(mine)
            mt = {c: m[band][c]["k"] for c in CLASSES}
            assert t == mt, f"{key} at {s}: {t} against metrics.json q1.{key}.{band} {mt}"
            per_stop.append(t)
        # the same check fed the wrong stop: the classes at θ_s + 0.1 against the stored class at θ_s
        wrong = sum(class_at(n, stops[2]) != r["class"] for n, r in zip(full, rows))
        assert wrong > 0, f"{key}: the class check cannot fail"

        # the rounded scores written to data.js give the same class at every slider position
        flips = [(v, n["p"]) for v in positions for n, f in zip(small, full) if class_at(n, v) != class_at(f, v)]
        assert not flips, f"{key}: scores at {SCORE_DECIMALS} decimals change {len(flips)} classes: {flips[:5]}"
        curve = [(v, sum(class_at(n, v) == "ONE" for n in small)) for v in positions]
        best = max(curve, key=lambda x: x[1])

        # shuffled-pole control: the stored texts scored at a random other neuron of the same layer
        scored = [r for r in rows if "texts" in r and not r["rare"]]
        shuf_strong = sum(t["score"] >= theta for r in scored for t in r["shuffled"])
        assert shuf_strong == m["strong_rates"]["shuffled"]["k"], (shuf_strong, m["strong_rates"]["shuffled"])
        assert shuf_strong == EXPECTED_SHUFFLED_STRONG[key], (key, shuf_strong)
        shuf_one = 0
        for r in scored:
            sc = {t["text_id"]: t["score"] for t in r["shuffled"]}
            g = [[sc[x["text_id"]] for x in r["texts"][k]] for k in ("pos", "near", "distant")]
            shuf_one += assign_class(False, r["label"], [j["related_to_others"] for j in r["jobs"]],
                                     explanation_passes(*g, theta), lambda k: False) == "ONE"
        label_one = sum(r["label"] == "ONE" for r in rows)

        exp = EXPECTED_ONE[key]
        for s, t, e in zip(stops, per_stop, exp):
            stop_rows.append(f"| {name} | {s:.4f} | {t['ONE']} | {e} | {t['TWO-REL']} | {t['UNREL']} | {t['UNK']} | {t['RARE']} |")
            if t["ONE"] != e:
                differ.append(f"{name} one job at {s:.4f}: {t['ONE']} (expected {e})")
        if label_one != EXPECTED_LABEL_ONE[key]:
            differ.append(f"{name} explainer said one job: {label_one} (expected {EXPECTED_LABEL_ONE[key]})")
        extra_rows.append(f"| {name} | {label_one} ({EXPECTED_LABEL_ONE[key]}) | {shuf_strong} of "
                          f"{sum(len(r['shuffled']) for r in scored):,} ({EXPECTED_SHUFFLED_STRONG[key]}) | {shuf_one} | "
                          f"{best[1]} at {best[0]:.4g} | {wrong} |")

        # one pole scored one job and one scored cannot tell at θ_s, with its texts and scores
        if key == "O7":
            for want in ("ONE", "UNK"):
                r = next(r for r in rows if r["class"] == want and r["label"] == "ONE" and "texts" in r)
                samples.append((name, want, r))

        neurons = []
        for n in small:
            rec = {"p": n["p"], "l": n["l"], "j": "".join("1" if x else "0" for x in n["rel"]), "f": n["f"],
                   "e": list(EXPLAINERS).index(n["e"]), "x": n["x"]}
            if n["s"] is not None:
                rec["s"] = n["s"]
            if n["u"]:
                rec["u"] = {str(k): v for k, v in sorted(n["u"].items())}
            neurons.append(rec)
        out_models.append({"key": key, "name": name, "neurons": neurons,
                           "check": {"stops": [t["ONE"] for t in per_stop]}})

    data = {"theta": theta, "stops": list(stops), "slider": list(SLIDER), "essay_one": ESSAY_ONE, "n": N_POLES,
            "explainers": list(EXPLAINERS.values()), "models": out_models}
    out = a.out / "data.js"
    out.write_text("window.ONE_JOB_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    size = out.stat().st_size
    assert size < MAX_BYTES, f"{out} is {size:,} bytes; shorten the explanations to 160 characters"
    print(f"wrote {out}: {size:,} bytes; full explanations; scores at {SCORE_DECIMALS} decimals")

    def show(t):
        return t.replace("\n", "⏎")

    sample_lines = []
    for name, want, r in samples:
        sample_lines += [f"{name}, {r['pole']}, explainer {EXPLAINERS[r['explainer_model']]} said {r['label']}; "
                         f"class at θ_s {r['class']}:", "", f"> {show(r['explanation'])}", ""]
        for k in ("pos", "near", "distant"):
            for x in r["texts"][k][:2]:
                sample_lines.append(f"- {k}, score {x['score']:.4f}{' (strong)' if x['strong'] else ''}: {show(x['text'])}")
        sample_lines += [f"- scores, pos {[round(x['score'], 3) for x in r['texts']['pos']]}, near "
                         f"{[round(x['score'], 3) for x in r['texts']['near']]}, distant "
                         f"{[round(x['score'], 3) for x in r['texts']['distant']]}", ""]

    verdict_line = ("Every expected number above matches." if not differ
                    else "These differ from the expected numbers: " + "; ".join(differ) + ".")
    block = "\n".join([
        f"Computed by `python3 extract.py` from `{a.verdict}`.", "",
        f"Strong line θ_s = {theta!r} (metrics.json `theta.theta_s`); the stops are θ_s - 0.1, θ_s and θ_s + 0.1.", "",
        "| model | strong line | one job | expected | two related | unrelated | cannot tell (UNK) | too rare (RARE) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|", *stop_rows, "",
        f"At every stop the recomputed class equals the stored per-pole class for all 600 poles of each model, and the "
        f"counts equal metrics.json `q1.<model>.band_theta_minus_0.1`, `fractions` and `band_theta_plus_0.1`. The page "
        f"folds RARE into \"cannot tell\". The essay's estimate drawn on the page: {ESSAY_ONE} of {N_POLES}.", "",
        "| model | explainer said one job (expected) | shuffled texts strong at θ_s (expected) | one job from shuffled scores at θ_s | most one-job poles on the slider | classes at θ_s + 0.1 that differ from the stored class at θ_s |",
        "|---|---:|---:|---:|---:|---:|", *extra_rows, "",
        f"The last column is the class check fed the wrong stop; it fails there, so it can fail. Scores are stored in "
        f"`data.js` at {SCORE_DECIMALS} decimals; at all {len(positions)} slider positions (0.05 to 0.80 in steps of "
        f"0.005, with 0.215, 0.315 and 0.415 replaced by the exact stops) they give the same class as the full-precision "
        f"scores for every pole.", "",
        verdict_line, "",
        "One pole scored one job and one scored cannot tell, at θ_s, with two of each kind of text:", "",
        *sample_lines])
    print("\n" + block + "\n")

    readme = a.out / "README.md"
    txt = readme.read_text()
    assert txt.count(CHECKS_START) == 1 and txt.count(CHECKS_END) == 1, f"{readme} lacks the checks markers"
    head, rest = txt.split(CHECKS_START)
    readme.write_text(head + CHECKS_START + "\n" + block + "\n" + CHECKS_END + rest.split(CHECKS_END)[1])
    print(f"rewrote the checks block of {readme}")


if __name__ == "__main__":
    main()
