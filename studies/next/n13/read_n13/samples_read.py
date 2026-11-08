"""N13 independent read, step 6: the banked samples of six arms (MATH-500), checked against generations.jsonl and re-graded.

    python -I samples_read.py <banks dir> <out.json>

For each of base, rl, top_k1, dare_k1, rndE_k1, attn: every samples.jsonl row with its role (problems 0-2, lowest-index right, lowest
wrong, a cap hit), the prompt's first line (the problem's first line after the instruction), the extracted answer, the gold, the
banked score, the reader's re-grade (A1 wrap, math_verify), and whether the row equals that problem's generations.jsonl record.
Fail-loud: no try/except.
"""
import json
import pathlib
import re
import sys

from math_verify import parse, verify

BANKS, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
PRE = ("Solve the following problem step by step. The last line of your response should be the answer to the problem in form Answer: "
       "$Answer (without quotes) where $Answer is the answer to the problem.\n\n")


def wrap(s):
    return s if (re.search(r"(?<!\\)\$", s) or "\\boxed" in s) else "$" + s + "$"


def grade(g, x):
    return 0 if x is None else int(bool(verify(parse(wrap(g)), parse(wrap(x)))))


res = {}
for arm in ("base", "rl", "top_k1", "dare_k1", "rndE_k1", "attn"):
    d = BANKS / arm / arm / "math500"
    gens = {r["problem_index"]: r for r in (json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines() if l.strip())}
    rows = [json.loads(l) for l in (d / "samples.jsonl").read_text().splitlines() if l.strip()]
    right = min(i for i, r in gens.items() if r["score"] == 1)
    wrong = min(i for i, r in gens.items() if r["score"] == 0)
    caps = sorted(i for i, r in gens.items() if r["cap_hit"])
    out = []
    for s in rows:
        i = s["problem_index"]
        g = gens[i]
        common = [k for k in g if k in s and k != "utc"]
        problem = s["prompt"][len(PRE):]
        out.append({"i": i, "roles": [x for x, c in (("fixed", i in (0, 1, 2)), ("lowest_right", i == right), ("lowest_wrong", i == wrong),
                                                     ("cap_hit", s["cap_hit"])) if c],
                    "prompt_starts_with_instruction": s["prompt"].startswith(PRE), "problem_first_line": problem.split("\n")[0][:160],
                    "mode": s["mode"], "extracted": s["extracted"], "gold": s["gold"], "score": s["score"], "regrade": grade(s["gold"], s["extracted"]),
                    "n_new": s["n_new"], "cap_hit": s["cap_hit"], "finish_reason": s["finish_reason"], "answer_prob": s.get("answer_prob"),
                    "equals_generations_record": all(s[k] == g[k] for k in common), "n_fields_compared": len(common)})
    have = {r["i"] for r in out}
    res[arm] = {"rows": out, "lowest_right": right, "lowest_wrong": wrong, "first_cap_hit": caps[0] if caps else None,
                "covers_0_1_2_right_wrong": {0, 1, 2, right, wrong} <= have, "has_a_cap_hit_row": any(r["cap_hit"] for r in out) or not caps}
OUT.write_text(json.dumps(res, indent=1))
for arm, v in res.items():
    print(f"== {arm}: lowest right {v['lowest_right']}, lowest wrong {v['lowest_wrong']}, covers {v['covers_0_1_2_right_wrong']}, cap row {v['has_a_cap_hit_row']}")
    for r in v["rows"]:
        print(f"  #{r['i']} {r['roles']} [{r['mode']}] extracted={r['extracted']!r} gold={r['gold']!r} score={r['score']} regrade={r['regrade']} "
              f"n_new={r['n_new']} cap={r['cap_hit']} eqgen={r['equals_generations_record']} | {r['problem_first_line'][:90]!r}")
