"""N13 scoring (studies/next/07_PICKS/N13_SPEC.md §2.2 prompt, §2.4 extraction and grading, §6.1 G-GRADER, G-TEMPLATE, G-COMPLETE,
S-BAND, §6.3 damage screen, §7 statistics). Standard library, NumPy, math-verify 0.9.0 (imported lazily) and a transformers tokenizer
object passed in by the caller.

Grading rules. RULE "registered" is §2.4 item 3 as written: math_verify.verify(parse(gold), parse(x)) with x wrapped as $x$ when it
holds neither "$" nor "\\boxed"; the gold is parsed as stored. RULE "dollar_both" applies the same wrap to the gold as well as to x,
and does not count an escaped dollar ("\\$", a currency sign in MATH-500 golds) as a delimiter. RULE "boxed" wraps both as \\boxed{.}.
The registered rule failed G-GRADER in the offline gate (2026-11-05: 12/24 equivalent pairs, 392/500 golds equal to themselves);
"boxed" passed the 500 golds but judged "45°" unequal to "45^\\circ"; "dollar_both" passed every check. Under G-GRADER's own branch
("before freeze: fix and record") GRADE_RULE is "dollar_both". §2.4 item 3 needs a dated amendment; until Fable records it, this
choice is a deviation (N13_OFFLINE_GATE.md, Spec issues).
Fail-loud: no try/except, no fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from studies.reproductions.alwayson.code.pg.alwayson_edit import boxed_last as alwayson_boxed_last, row_sha256  # noqa: F401 (REUSED, §9.2)

TEMPLATE = {
    "pair1": ("Solve the following problem step by step. The last line of your response should be the answer to the problem in form "
              "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}\n\nRemember to put your answer on "
              "its own line after \"Answer:\""),
    "pair1b": ("Solve the following math problem step by step. The last line of your response should be the answer to the problem in "
               "form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n{problem}\n\nRemember to put your "
               "answer on its own line after \"Answer:\""),
}
BOS_ID, STOP_ID = 100257, 100257
GRADE_RULE = "dollar_both"
SEED_BASE = {"A": 1_000_000, "B": 2_000_000}
BOXED = "\\boxed{"
SCREEN = {"first": 20, "cap_hits_min": 15, "correct_max": 1}


# ------------------------------------------------------------------------------------------------ prompt (G-TEMPLATE)
def render_text(tok, problem: str, add_generation_prompt: bool = True) -> str:
    return tok.apply_chat_template([{"role": "user", "content": problem}], tokenize=False, add_generation_prompt=add_generation_prompt)


def render_ids(tok, problem: str) -> list:
    ids = tok.apply_chat_template([{"role": "user", "content": problem}], tokenize=True, add_generation_prompt=True, return_dict=False)
    return [int(i) for i in ids]


def g_template(tok, problems: list, pair: str = "pair1") -> dict:
    exp = TEMPLATE[pair]
    bad = [i for i, p in enumerate(problems) if render_text(tok, p) != exp.format(problem=p)]
    ids = [render_ids(tok, p) for p in problems]
    off = render_text(tok, problems[0], add_generation_prompt=False)
    return {"pass": not bad, "n": len(problems), "mismatched": bad[:10], "n_mismatched": len(bad),
            "first_id_is_bos": sorted({i[0] == BOS_ID for i in ids}), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "failing_input": {"what": "the render with add_generation_prompt=False", "failed": off != exp.format(problem=problems[0]),
                              "tail": off[-60:]},
            "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest()}, ids


# ------------------------------------------------------------------------------------------------ extraction
def boxed_last(text: str):
    """The content of the last balanced \\boxed{...} in text, or None (the rule of alwayson_edit.boxed_last)."""
    last, i = None, text.find(BOXED)
    while i >= 0:
        depth, j = 1, i + len(BOXED)
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            last = (text[i + len(BOXED):j - 1], i + len(BOXED), j - 1)
        i = text.find(BOXED, i + 1)
    return last


# A26: the label at the start of a line after optional spaces, alone or inside one markdown emphasis pair.
ANSWER_LABEL = re.compile(r" *(?:(\*\*|\*|__)Answer:\1|Answer:)")


def extract(text: str) -> tuple:
    """(mode, extracted string, char span) by §2.4 item 2 as amended (N13_SPEC.md A10, A26). A label line begins, after optional
    spaces, with "Answer:", "**Answer:**", "*Answer:*" or "__Answer:__" (A26). A label line with nothing after it is skipped (A10),
    so the last non-empty one counts, and with none the last box does."""
    pos, hit = 0, None
    for line in text.split("\n"):
        m = ANSWER_LABEL.match(line)
        if m:
            raw = line[m.end():]
            lead = len(raw) - len(raw.lstrip())
            if raw.strip():
                start = pos + m.end() + lead
                hit = (raw.strip(), (start, start + len(raw.strip())))
        pos += len(line) + 1
    if hit is not None:
        return "answer_line", hit[0], hit[1]
    b = boxed_last(text)
    if b is not None:
        return "boxed", b[0], (b[1], b[2])
    return "none", None, None


# ------------------------------------------------------------------------------------------------ grading
def wrap(s: str, rule: str, side: str) -> str:
    if rule == "registered":
        if side == "gold":
            return s
        return s if ("$" in s or "\\boxed" in s) else f"${s}$"
    if rule == "dollar_both":
        return s if (re.search(r"(?<!\\)\$", s) or "\\boxed" in s) else f"${s}$"
    if rule == "boxed":
        return s if "\\boxed" in s else "\\boxed{" + s + "}"
    raise ValueError(rule)


def grade(gold: str, extracted, rule: str = GRADE_RULE) -> dict:
    from math_verify import parse, verify
    if extracted is None:
        return {"score": 0, "parsed_gold": None, "parsed_answer": None}
    g, x = parse(wrap(gold, rule, "gold")), parse(wrap(extracted, rule, "answer"))
    return {"score": int(bool(verify(g, x))), "parsed_gold": [str(v) for v in g], "parsed_answer": [str(v) for v in x]}


# 24 equivalent and 16 non-equivalent (gold, answer) pairs in MATH-500 answer styles (§9.3 item 4).
PLANTED_EQ = [
    ("\\frac{1}{2}", "0.5"), ("\\frac{1}{2}", "\\dfrac{1}{2}"), ("\\frac{3}{4}", "\\frac34"), ("10", "10.0"), ("45^\\circ", "45°"),
    ("[1,3)", "[1, 3)"), ("(-\\infty, 2]", "(-\\infty,2]"), ("30^\\circ", "30"), ("90^\\circ", "90^{\\circ}"), ("2\\sqrt{3}", "\\sqrt{12}"),
    ("\\frac{\\sqrt{2}}{2}", "\\frac{1}{\\sqrt{2}}"), ("\\frac{\\pi}{2}", "\\pi/2"), ("(x+1)^2", "x^2+2x+1"), ("-\\frac{2}{3}", "\\frac{-2}{3}"),
    ("(2,3)", "(2, 3)"), ("\\{1,2,3\\}", "\\{3,2,1\\}"), ("1,2,3", "1, 2, 3"), ("3", "\\boxed{3}"), ("\\frac{7}{2}", "3.5"),
    ("x=5", "5"), ("\\text{(C)}", "C"), ("2^{10}", "1024"), ("\\$36", "36"), ("5\\%", "5"),
]
PLANTED_NEQ = [
    ("\\frac{1}{2}", "\\frac{1}{3}"), ("10", "100"), ("[1,3)", "(1,3)"), ("[1,3)", "[1,3]"), ("2\\sqrt{3}", "3\\sqrt{2}"),
    ("-5", "5"), ("(2,3)", "(3,2)"), ("x+1", "x-1"), ("\\pi", "3.14"), ("\\frac{\\pi}{2}", "\\pi"), ("2^{10}", "1000"),
    ("\\text{(C)}", "D"), ("\\{1,2\\}", "\\{1,2,3\\}"), ("0", "1"), ("\\sqrt{2}", "1.41"), ("7", "\\frac{7}{2}"),
]


def normalise(s: str) -> str:
    return re.sub(r"\s+", "", s.replace("\\left", "").replace("\\right", ""))


def derangement(n: int, seed: int = 0) -> np.ndarray:
    """A permutation with no fixed point: a uniform random cycle order p, each p[i] mapped to p[i+1] (one draw, always deranged)."""
    p = np.random.default_rng(seed).permutation(n)
    d = np.empty(n, dtype=np.int64)
    d[p] = np.roll(p, -1)
    assert (d != np.arange(n)).all()
    return d


def g_grader(golds: list, rule: str = GRADE_RULE, seed: int = 0) -> dict:
    eq = [grade(g, x, rule)["score"] for g, x in PLANTED_EQ]
    neq = [grade(g, x, rule)["score"] for g, x in PLANTED_NEQ]
    self_eq = [grade(g, g, rule)["score"] for g in golds]
    d = derangement(len(golds), seed)
    der = [grade(golds[i], golds[d[i]], rule)["score"] for i in range(len(golds))]
    ident = float(np.mean([normalise(golds[i]) == normalise(golds[d[i]]) for i in range(len(golds))]))
    rate = float(np.mean(der))
    v = {"rule": rule, "planted_eq_true": int(sum(eq)), "planted_eq_n": len(eq), "planted_neq_false": int(len(neq) - sum(neq)),
         "planted_neq_n": len(neq), "eq_failed": [PLANTED_EQ[i] for i, s in enumerate(eq) if not s],
         "neq_failed": [PLANTED_NEQ[i] for i, s in enumerate(neq) if s], "self_equal": int(sum(self_eq)), "n_golds": len(golds),
         "self_unequal": [(i, golds[i]) for i, s in enumerate(self_eq) if not s][:60], "derangement_rate": rate,
         "identical_normalised_rate": ident, "derangement_bound": ident + 0.01, "derangement_seed": seed}
    v["pass"] = bool(v["planted_eq_true"] == len(eq) and v["planted_neq_false"] == len(neq) and v["self_equal"] == len(golds)
                     and rate <= ident + 0.01)
    v["failing_input"] = {"what": "the derangement scored as if it were the identity (needs every pair equal)",
                          "value": int(sum(der)), "failed": int(sum(der)) != len(golds)}
    return v


# ------------------------------------------------------------------------------------------------ GSM8K
def gsm8k_gold(answer_field: str) -> str:
    assert "#### " in answer_field, answer_field[-80:]
    return answer_field.split("#### ")[-1].strip().replace(",", "")


# ------------------------------------------------------------------------------------------------ token bytes (answer-line log-probability)
def _byte_decoder() -> dict:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BD = _byte_decoder()


def token_bytes(tok, ids: list) -> list:
    added = {int(k): v for k, v in tok.added_tokens_decoder.items()}
    out = []
    for i, t in zip(ids, tok.convert_ids_to_tokens(ids)):
        out.append(added[i].content.encode() if i in added else bytes(_BD[c] for c in t))
    return out


def span_logprob(tok, ids: list, logprobs: list, text: str, span) -> dict:
    """Sum of sampled-token log-probabilities over the tokens that overlap the char span of the extracted answer."""
    tb = token_bytes(tok, ids)
    assert b"".join(tb).decode("utf-8", errors="replace") == text, "token bytes do not reproduce the decoded text"
    mean_all = float(np.mean(logprobs)) if logprobs else None
    if span is None:
        return {"answer_logprob": None, "answer_prob": None, "answer_tokens": 0, "mean_logprob": mean_all}
    b0, b1 = len(text[: span[0]].encode()), len(text[: span[1]].encode())
    pos, s, n = 0, 0.0, 0
    for t, lp in zip(tb, logprobs):
        if pos < b1 and pos + len(t) > b0:
            s += lp
            n += 1
        pos += len(t)
    return {"answer_logprob": s, "answer_prob": math.exp(s), "answer_tokens": n, "mean_logprob": mean_all}


# ------------------------------------------------------------------------------------------------ statistics (§7)
def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar(x, y) -> dict:
    """Exact two-sided McNemar on paired 0/1 vectors: binomial on the discordant problems, p = 0.5."""
    x, y = np.asarray(x), np.asarray(y)
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    m = min(b, c)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / 2 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


def bh(pvals: list, q: float = 0.05) -> dict:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    o = np.argsort(p)
    adj = np.empty(n)
    adj[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    return {"adjusted": adj.tolist(), "reject": (adj <= q).tolist(), "q": q}


def paired_bootstrap(x, y, n_boot: int = 2000, seed: int = 0) -> tuple:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    d = 100 * (x[idx].mean(1) - y[idx].mean(1))
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def sigma(xa, xb) -> float:
    """One single-seed accuracy reading's sd, in points, from two seed runs of the same weights (§0.5)."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    return float(100 * math.sqrt(((xa - xb) ** 2).sum() / 2) / len(xa))


def acc(x) -> float:
    return float(100 * np.mean(x))


# ------------------------------------------------------------------------------------------------ screens and completeness
def damage_screen(recs: list) -> dict:
    first = [r for r in recs if r["problem_index"] < SCREEN["first"]]
    if len(first) < SCREEN["first"]:
        return {"ready": False}
    hits, right = sum(r["cap_hit"] for r in first), sum(r["score"] for r in first)
    return {"ready": True, "cap_hits": hits, "correct": right,
            "fired": bool(hits >= SCREEN["cap_hits_min"] and right <= SCREEN["correct_max"])}


def g_complete(recs: list, n: int, seed_set: str) -> dict:
    """§6.1 G-COMPLETE as amended by A25: every problem has one record, seeded as registered, with n_generated >= 1 and a
    finish_reason. Empty text (the stop token alone) is the model's output: counted in n_empty and reported, not gated."""
    idx = sorted(r["problem_index"] for r in recs)
    v = {"n": len(recs), "distinct": len(set(idx)), "covers": idx == list(range(n)),
         "seeded": all(r["seed"] == SEED_BASE[seed_set] + r["problem_index"] for r in recs),
         "generated": all(r["n_generated"] >= 1 for r in recs), "finish_reason": all(bool(r["finish_reason"]) for r in recs),
         "n_empty": sum(r["text"] == "" for r in recs)}
    v["pass"] = bool(v["n"] == n and v["distinct"] == n and v["covers"] and v["seeded"] and v["generated"] and v["finish_reason"])
    return v


def s_band(l_prime: float, lo: float = 81.5, hi: float = 96.7) -> dict:
    return {"value": l_prime, "band": [lo, hi], "pass": bool(lo <= l_prime <= hi)}


def selfcheck():
    assert extract("x\n  Answer: $\\frac{1}{2}$\nmore")[:2] == ("answer_line", "$\\frac{1}{2}$")
    assert extract("Answer: 3\nAnswer:  4 ")[:2] == ("answer_line", "4")
    assert extract("so \\boxed{\\frac{1}{2}} and \\boxed{5}")[:2] == ("boxed", "5")
    assert extract("no answer here")[0] == "none"
    assert extract("Answer:\n\\boxed{5}")[:2] == ("boxed", "5") and extract("Answer: 3\nAnswer:  \nmore")[:2] == ("answer_line", "3")   # A10
    assert extract("x\n**Answer:** 42")[:2] == ("answer_line", "42") and extract("**Answer:**\n\\boxed{5}")[:2] == ("boxed", "5")   # A26
    assert extract("*Answer:* 7")[:2] == ("answer_line", "7") and extract("  __Answer:__ 8")[:2] == ("answer_line", "8")         # A26
    assert extract("Answer: 3\n**Answer:** 4")[:2] == ("answer_line", "4") and extract("**Answer:*\n\\boxed{2}")[:2] == ("boxed", "2")
    for t in ("**Answer**: 9", "**Final Answer:** 9", "answer: 9", "ANSWER: 9", "***Answer:*** 9", "### Answer: 9"):           # A26: unchanged
        assert extract(t)[0] == "none", t
    t = "a\n  **Answer:**  (3, \\pi)  \n"
    m, x, sp = extract(t)
    assert t[sp[0]:sp[1]] == x == "(3, \\pi)"
    t = "ab\nAnswer: 42\n"
    m, x, sp = extract(t)
    assert t[sp[0]:sp[1]] == x == "42"
    assert gsm8k_gold("so\n#### 1,234") == "1234"
    lo, hi = wilson(250, 500)
    assert abs(lo - 45.64) < 0.01 and abs(hi - 54.36) < 0.01, (lo, hi)
    assert abs(mcnemar([1] * 10 + [0] * 2, [0] * 10 + [1] * 2)["p"] - 0.0386) < 0.0001
    assert [round(v, 4) for v in bh([0.01, 0.04, 0.03, 0.2])["adjusted"]] == [0.04, 0.0533, 0.0533, 0.2]
    assert abs(sigma([1, 0, 1, 1], [1, 1, 0, 1]) - 25.0) < 1e-9
    d = derangement(7)
    assert sorted(d.tolist()) == list(range(7))
    assert [wrap(x, "dollar_both", "gold") for x in ("\\$36", "36", "$x$", "\\boxed{3}")] == ["$\\$36$", "$36$", "$x$", "\\boxed{3}"]
    for t in ("a \\boxed{x^{2}} b \\boxed{\\frac{1}{2}}", "none", "\\boxed{unclosed"):
        mine = boxed_last(t)
        assert (mine[0] if mine else None) == alwayson_boxed_last(t), t
    return True
