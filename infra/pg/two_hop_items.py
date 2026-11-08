"""N4 — U (knowledge in use): 100 two-hop items built from MMLU VALIDATION rows (1531; dev had only 62 eligible of 285 — smoke 2026-09-03) by ONE fixed template, plus
the shortcut control (the same items with the first hop removed). Spec §1.2 / §5; contract §2.

Template ("answer-initial"). Each scored item is built from one MMLU dev row (q1, its 4 options, gold a1):

    Inner question: <q1>
    What is the first letter of the correct answer to the inner question?
    A. the letter P
    B. the letter M
    C. the letter T
    D. the letter S
    Answer:

Hop 1 = recall the answer TEXT a1 of q1 (q1's option list is NOT shown, so no letter can be matched from
the prompt — the recall is content recall, as in K). Hop 2 = a property of that text (its initial),
chosen among the four initials of q1's own options in a per-item shuffled order (seed 1337; gold index
tracked). The second hop needs a1 itself: nothing in the prompt names a1, and the four candidate letters
are equally plausible without it. Rejected template ("the answer to q1 is the same as the correct option
of q2"): the second hop there is answerable by matching option lists, with no use of a1's text.

Why initials rather than word counts or letter counts: the initial of a word is the surface property a
small LM reads most reliably, and the question's form ("Which planet…") does not predict it, whereas it
does predict a word count (an uncaught first-hop-free shortcut). Eligibility filter (fail-loud count):
four non-empty options, all initials ASCII letters, all four distinct. Options render as "the letter X"
so the candidate letters are not confused with the A–D labels.

Shortcut control: the SAME items with "Inner question: <q1>" removed and the second hop rephrased as
"What is the first letter of the correct answer to an unstated question about <subject>?" — identical
options, order and gold. Validity rule = `matrix_core.u_valid`: healthy ≥ floor + 0.15 on U and
≤ floor + 0.10 on the shortcut set, else VOID-U.

Scoring: S-style letter read (format_readouts.U / U_shortcut) with 5 exemplars in the same format built
from 5 further eligible dev rows (hash-disjoint from the scored 100, asserted). Items carry the S item
schema {subject, question, choices, answer, hash} with subject = "two-step questions" (the
_subject_header renders "…about two-step questions.") plus the inner-row provenance fields.
Hashes are stable under repeated builds (dry-run asserts). Injectable dev rows for the dry-run.
"""
from __future__ import annotations

import random

from infra.pg import matrix_items as MI

U_SUBJECT = "two-step questions"
N_ITEMS, N_SHOTS, SEED = 100, 5, 1337
INNER = "Inner question: {q}\n"
HOP2 = "What is the first letter of the correct answer to the inner question?"
HOP2_SHORTCUT = "What is the first letter of the correct answer to an unstated question about {subject}?"


def _initials(row: dict):
    """The four option initials if the row is eligible, else None."""
    ch = [str(c).strip() for c in row["choices"]]
    if len(ch) != 4 or any(not c for c in ch):
        return None
    letters = [c[0].upper() for c in ch]
    if not all(ch_.isascii() and ch_.isalpha() for ch_ in letters) or len(set(letters)) != 4:
        return None
    return letters


def _compose(row: dict, order: list, shortcut: bool) -> dict:
    letters = _initials(row)
    assert letters is not None
    gold = order.index(int(row["answer"]))
    subject = row["subject"].replace("_", " ")
    inner_hash = MI.item_hash(row["subject"], row["question"])       # the source row's identity key
    # The tag names the source row inside the prompt (provenance a reader can check). It is no longer
    # load-bearing for uniqueness: MI.item_hash now hashes the FULL question, not a 64-char prefix.
    tag = f"(#{inner_hash}) "
    q = tag + (HOP2_SHORTCUT.format(subject=subject) if shortcut
               else INNER.format(q=row["question"].strip()) + HOP2)
    return {"subject": U_SUBJECT, "question": q, "choices": [f"the letter {letters[j]}" for j in order],
            "answer": gold, "hash": MI.item_hash(U_SUBJECT, q), "inner_subject": row["subject"],
            "inner_question": row["question"], "inner_answer_text": str(row["choices"][int(row["answer"])]).strip(),
            "inner_hash": inner_hash, "shortcut": shortcut}


def build_two_hop(dev_rows=None, load_dataset=None) -> dict:
    rows = dev_rows if dev_rows is not None else MI.mmlu_validation_rows(load_dataset)   # validation split (see matrix_items)
    pool = [r for r in rows if _initials(r) is not None]
    seen = set()                                                   # one item per distinct source row
    pool = [r for r in pool
            if not (MI.item_hash(r["subject"], r["question"]) in seen
                    or seen.add(MI.item_hash(r["subject"], r["question"])))]
    assert len(pool) >= N_ITEMS + N_SHOTS, f"only {len(pool)} eligible pool rows (< {N_ITEMS + N_SHOTS})"
    rng = random.Random(SEED)
    pick = rng.sample(range(len(pool)), N_ITEMS + N_SHOTS)
    items, shortcut, ex, ex_short = [], [], [], []
    for k, i in enumerate(pick):
        order = list(range(4))
        rng.shuffle(order)
        a, b = _compose(pool[i], order, False), _compose(pool[i], order, True)
        (items if k < N_ITEMS else ex).append(a)
        (shortcut if k < N_ITEMS else ex_short).append(b)
    assert not ({it["inner_hash"] for it in items} & {e["inner_hash"] for e in ex}), "exemplar/item leak"
    for name, lst in (("two_hop items", items), ("two_hop shortcut", shortcut),
                      ("two_hop exemplars", ex), ("two_hop shortcut_exemplars", ex_short)):
        MI._assert_unique(lst, name)
    assert not ({it["hash"] for it in items} & {s["hash"] for s in shortcut}), "U and U-shortcut keys collide"
    out = {"items": items, "shortcut": shortcut, "exemplars": ex, "shortcut_exemplars": ex_short}
    out["meta"] = {"template": "answer-initial", "seed": SEED, "pool_split": "validation" if dev_rows is None else "injected", "n_pool_rows": len(rows), "n_eligible": len(pool),
                   "content_sha256": MI._sha256([items, shortcut, ex, ex_short])}
    return out


# ------------------------------------------------------------------------------------------ dry-run
def fake_dev_rows() -> list:
    """57 subjects × 5 rows; rows 3 and 4 of every subject are ineligible (numeric / repeated initials)."""
    words = ["pear", "moon", "tiger", "salt", "quartz", "river", "kite", "lamp", "nest", "oak", "vine", "wolf"]
    rng = random.Random(7)
    rows = []
    for subj in MI.MMLU_SUBJECTS:
        for i in range(5):
            if i == 3:
                choices = ["12", "15", "20", "25"]
            elif i == 4:
                choices = ["pear", "plum", "moon", "tiger"]
            else:
                choices = rng.sample(words, 4)
            rows.append({"subject": subj, "question": f"Dev question {i} of {subj}?", "choices": choices,
                         "answer": rng.randrange(4)})
    return rows


def _dryrun():
    from infra.pg.text_eval import _q_block
    rows = fake_dev_rows()
    b1, b2 = build_two_hop(rows), build_two_hop(rows)
    assert b1["meta"]["content_sha256"] == b2["meta"]["content_sha256"]
    assert [i["hash"] for i in b1["items"]] == [i["hash"] for i in b2["items"]]
    assert b1["meta"]["n_eligible"] == 57 * 3 and len(b1["items"]) == 100 and len(b1["exemplars"]) == 5
    for it, sc in zip(b1["items"] + b1["exemplars"], b1["shortcut"] + b1["shortcut_exemplars"]):
        assert it["choices"][it["answer"]] == "the letter " + it["inner_answer_text"][0].upper()
        assert len(set(it["choices"])) == 4 and it["question"].startswith("(#") and "Inner question: " in it["question"]
        assert sc["choices"] == it["choices"] and sc["answer"] == it["answer"]
        assert "Inner question" not in sc["question"] and it["inner_subject"].replace("_", " ") in sc["question"]
        assert it["hash"] != sc["hash"] and it["inner_question"] in it["question"]
        assert _q_block(it["question"], it["choices"]).endswith("Answer:")
    assert {it["inner_hash"] for it in b1["items"]}.isdisjoint({e["inner_hash"] for e in b1["exemplars"]})
    from infra.pg.matrix_core import u_valid                  # the validity rule lives in the core
    assert u_valid(0.5, 0.3)["U"] == "VALID" and u_valid(0.3, 0.3)["U"] == "VOID-U" and u_valid(0.5, 0.4)["U"] == "VOID-U"
    print(_q_block(b1["items"][0]["question"], b1["items"][0]["choices"]))
    print(f"eligible {b1['meta']['n_eligible']} / {len(rows)}; content sha256 {b1['meta']['content_sha256'][:12]}")
    print("PASS two_hop_items dry-run (100 items + shortcut control, hashes stable)")


if __name__ == "__main__":
    _dryrun()
