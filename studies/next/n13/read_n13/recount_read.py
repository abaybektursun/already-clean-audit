"""N13 independent read, step 2: accuracies from the banks, by the banked scores and by an independent re-extraction and re-grade.

    python -I recount_read.py <banks dir> <out.json>

For every arm x benchmark: n, completeness and seeds, the banked accuracy, the accuracy in metrics.json, the accuracy from re-grading
the banked extracted string against the banked gold with math_verify under the active rule (N13_SPEC.md 2.4 item 3 as amended by A1),
and the accuracy from re-extracting the generated text by 2.4 item 2 as amended by A10 and A26 (this reader's own code) and grading
that. Wilson 95 % intervals (z = 1.959964). Every disagreement is listed. Written by the independent reader; it imports no n13 code.
Fail-loud: no try/except.
"""
import json
import math
import pathlib
import re
import sys

from math_verify import parse, verify

BANKS, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
SEED_B_ARMS = {"zero", "id", "top_k1_B"}
N = {"math500": 500, "gsm8k": 1319}
Z = 1.959964

# 2.4 item 2 as amended by A10 and A26: a label line begins, after optional spaces, with one of four labels (case as written).
LABELS = ("**Answer:**", "*Answer:*", "__Answer:__", "Answer:")


def label_rest(line):
    s = line.lstrip(" ")
    for lab in LABELS:
        if s.startswith(lab):
            return s[len(lab):]
    return None


def boxed_last(text):
    key, out, start = "\\boxed{", None, 0
    while True:
        i = text.find(key, start)
        if i < 0:
            return out
        depth, j = 1, i + len(key)
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out = text[i + len(key):j - 1]
        start = i + 1


def my_extract(text):
    hit = None
    for line in text.split("\n"):
        rest = label_rest(line)
        if rest is not None and rest.strip():
            hit = rest.strip()
    if hit is not None:
        return "answer_line", hit
    b = boxed_last(text)
    if b is not None:
        return "boxed", b
    return "none", None


def wrap(s):
    # A1: unchanged when s holds a "$" not preceded by a backslash, or holds \boxed; otherwise $s$. Same for gold and answer.
    if re.search(r"(?<!\\)\$", s) or "\\boxed" in s:
        return s
    return "$" + s + "$"


CACHE = {}


def grade(gold, x):
    if x is None:
        return 0
    k = (gold, x)
    if k not in CACHE:
        CACHE[k] = int(bool(verify(parse(wrap(gold)), parse(wrap(x)))))
    return CACHE[k]


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


res, golds_by, prompts_by = {}, {}, {}
for gen in sorted(BANKS.rglob("generations.jsonl")):
    bench, arm = gen.parent.name, gen.parent.parent.name
    recs = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["problem_index"])
    m = json.loads((gen.parent / "metrics.json").read_text())
    n = N[bench]
    seed0 = 2_000_000 if arm in SEED_B_ARMS else 1_000_000
    idx = [r["problem_index"] for r in recs]
    complete = {"n": len(recs), "covers": idx == list(range(n)), "seeded": all(r["seed"] == seed0 + r["problem_index"] for r in recs),
                "n_generated_ge1": all(r["n_generated"] >= 1 for r in recs), "finish_reason": all(bool(r["finish_reason"]) for r in recs),
                "n_empty": sum(r["text"] == "" for r in recs), "cap_hit_consistent": all(r["cap_hit"] == (r["n_new"] == 16384) for r in recs)}
    banked = [int(r["score"]) for r in recs]
    regr = [grade(r["gold"], r["extracted"]) for r in recs]
    reex = [my_extract(r["text"]) for r in recs]
    regr2 = [grade(r["gold"], x) for r, (_, x) in zip(recs, reex)]
    ext_diff = [r["problem_index"] for r, (md, x) in zip(recs, reex) if (md, x) != (r["mode"], r["extracted"])]
    k_b, k_r, k_r2 = sum(banked), sum(regr), sum(regr2)
    dis = [{"i": r["problem_index"], "banked": b, "regraded": g, "gold": r["gold"], "extracted": r["extracted"]}
           for r, b, g in zip(recs, banked, regr) if b != g]
    dis2 = [{"i": r["problem_index"], "banked": b, "re_extracted_graded": g, "mode": r["mode"], "my_mode": md, "extracted": r["extracted"],
             "my_extracted": x} for r, b, g, (md, x) in zip(recs, banked, regr2, reex) if b != g]
    for r in recs:
        golds_by.setdefault((bench, r["problem_index"]), set()).add(r["gold"])
        prompts_by.setdefault((bench, r["problem_index"]), set()).add((r["prompt"], r["prompt_n_tokens"], tuple(r["prompt_first20"]),
                                                                       tuple(r["prompt_last20"])))
    res[f"{arm}/{bench}"] = {
        "arm": arm, "bench": bench, "status": m["status"], "complete": complete,
        "k_banked": k_b, "acc_banked": 100 * k_b / n, "wilson_banked": wilson(k_b, n),
        "metrics_correct": m["counts"]["correct"], "metrics_acc": m["counts"].get("acc"),
        "k_regraded": k_r, "acc_regraded": 100 * k_r / n, "wilson_regraded": wilson(k_r, n),
        "k_reextracted": k_r2, "acc_reextracted": 100 * k_r2 / n,
        "n_score_disagree_regrade": len(dis), "score_disagreements_regrade": dis,
        "n_extraction_differ": len(ext_diff), "extraction_differ_idx": ext_diff[:50],
        "n_score_disagree_reextract": len(dis2), "score_disagreements_reextract": dis2[:30],
        "modes_banked": {md: sum(r["mode"] == md for r in recs) for md in ("answer_line", "boxed", "none")},
        "metrics_modes": m["counts"]["modes"],
        "scores_banked": "".join(map(str, banked)), "scores_regraded": "".join(map(str, regr)),
    }
    print(f"{arm:12s} {bench:8s} n={len(recs)} banked {100*k_b/n:6.2f} metrics {m['counts']['correct']} regraded {100*k_r/n:6.2f} "
          f"re-extracted {100*k_r2/n:6.2f} disagree {len(dis)}/{len(dis2)} extract-differ {len(ext_diff)}", flush=True)

cons = {"golds_consistent_across_arms": all(len(v) == 1 for v in golds_by.values()),
        "prompts_consistent_across_arms": all(len(v) == 1 for v in prompts_by.values()),
        "n_items": {b: sum(1 for (bb, _) in golds_by if bb == b) for b in N}}
OUT.write_text(json.dumps({"rule": "dollar_both (A1); extraction 2.4 item 2 as amended by A10, A26", "consistency": cons, "arms": res},
                          indent=1))
print(json.dumps(cons))
