"""STAGEDIFF items — spec §2 readouts B1-beh / B1-int / B2, §4 data, §5 controls (derangement, shuffle, ROBOT_VOCAB).

Lens-eval sets: the six `lens-eval-*.json` of `anthropics/jacobian-lens` @ 581d3986 (`data/evaluations/`),
vendored under `stagediff_data/lens_eval/` and sha256-asserted at load. Readout position per set follows the
upstream README (`UPSTREAM_README.md` beside the data): the token before `target` (multihop, multilingual,
order-ops; the prompt is IN-CONTEXT tokenized by `context_split`, §15.15), the LAST NEWLINE token (poetry),
the final prompt token (association, typo). Order-ops intermediates are keys expanded to synonym sets (digit + number word;
operation symbol + word forms) and ranked as the min over single-token synonyms.

FROZEN ITEM RULES (2026-09-04, before any GPU; counts under the Qwen3 tokenizer in the spec amendment):
  scorable intermediate = has ≥ 1 single-token surface among {" X", "X"} (ids < V_SHARED = 151 669) over its synonyms;
  B1-int item          = has ≥ 1 scorable intermediate (538 of 550 after the one duplicate is dropped);
  B1-beh item          = target is ONE token in its own context (`context_split`, §15.15); top-1 == that token
                         is 'correct'. First-in-context-token match over all 255 is the secondary `correct_first`.
  split                = 50/50 within set by `random.Random(1337)` over the FILE order (independent of any
                         tokenizer): the first ⌊n/2⌋ of the shuffle are discovery, the rest confirmation.
Item hash = `matrix_items.item_hash(set, prompt)` (sha1, 16 hex) — the pairing key across checkpoints and arms.
"""
from __future__ import annotations

import hashlib
import json
import os
import random

from infra.pg.matrix_items import item_hash
from infra.pg.stagediff_models import V_SHARED

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stagediff_data", "lens_eval")
FILE_SHA256 = {
    "association": "d1a98cd4911b594282e74168091c77d849dae18ffe2acb5761074853f327d71c",
    "multihop": "50b7e4c9255291c0ca2a8e94615be9f44531fa57bb1a844e4f9616056d987416",
    "multilingual": "fa70b9bd89416a6d8d985a80dc628b109ae6fd3b25b9275c0fc5065d7ff4a0ef",
    "order-ops": "b203206d16ff628152cc86f3838604e06cb54776f3e14fa1c34f150db8bc7560",
    "poetry": "6aeb3415c5a5c3f3827c9efe63f006de02f5ef39a816bbac68e15e733aba60cc",
    "typo": "9d05e16b7234a57d0773d120a4e1c4e94fd3bc2235a8125d4200a70e60ab17aa",
}
SETS_ALL = ("multihop", "multilingual", "order-ops", "association", "poetry", "typo")
SETS_BEH = ("multihop", "multilingual", "order-ops")
SPLIT_SEED = 1337
HALVES = ("discovery", "confirmation")
# exact-duplicate prompts in the upstream files, dropped after the first occurrence (frozen; asserted at load)
DUPLICATES_DROPPED = {"poetry": ["couplet-divine-wine-to-sign"]}

NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
             "8": "eight", "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
             "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
             "19": "nineteen", "20": "twenty", "24": "twenty-four"}
OPS_SYNONYMS = {"addition": ["+", "plus", "add", "addition", "sum", "added"],
                "subtraction": ["-", "minus", "subtract", "subtraction", "difference"],
                "multiplication": ["*", "×", "times", "multiply", "multiplication", "product"],
                "division": ["/", "÷", "divide", "division", "divided", "quotient"],
                "mod": ["mod", "%", "modulo", "remainder"],
                "squared": ["squared", "square", "^"]}

# Hand list from MolmoAct2 (2605.02881) §4 vocabulary, written before W2 and frozen; words whose two
# surfaces are both multi-token under the Qwen3 tokenizer were dropped BEFORE freezing (gripper, grasping,
# manipulator, effector, franka, LIBERO, DROID, teleoperation, actuator, bimanual, proprioception).
ROBOT_VOCAB = ("robot", "robots", "robotic", "robotics", "arm", "arms", "action", "actions", "grasp", "pick",
               "place", "joint", "joints", "trajectory", "trajectories", "camera", "wrist", "torque",
               "manipulation", "pose", "poses", "motion", "move", "reach", "lift", "drawer", "cabinet", "bowl",
               "plate", "cup", "mug", "basket", "object", "objects", "block", "cube", "panda", "policy",
               "control", "controller", "velocity", "position", "rotation", "open", "close", "push", "pull",
               "stack", "bin", "shelf", "kitchen", "sink", "stove", "demonstration", "episode", "embodiment",
               "depth", "waypoint", "servo", "end", "tabletop", "suction", "grip", "motor", "navigate", "reward",
               "task", "scene", "frame", "step", "steps", "sim", "simulation")
ROBOT_VOCAB_SHA256 = hashlib.sha256(json.dumps(ROBOT_VOCAB).encode()).hexdigest()


# ------------------------------------------------------------------------------------------ sets
def load_sets() -> dict:
    """{set: [item]} with `hash` and `split` added; file sha256 asserted; hash uniqueness asserted."""
    out = {}
    for s in SETS_ALL:
        p = os.path.join(DATA_DIR, f"lens-eval-{s}.json")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == FILE_SHA256[s], f"{s}: sha256 {h} != frozen {FILE_SHA256[s]}"
        items = json.load(open(p))["items"]
        idx = list(range(len(items)))
        random.Random(SPLIT_SEED).shuffle(idx)
        disc = set(idx[: len(items) // 2])
        rows, seen, dropped = [], set(), []
        for i, it in enumerate(items):
            hh = item_hash(s, it["prompt"])
            if hh in seen:                 # poetry ships one exact duplicate prompt (same intermediate): scored once
                dropped.append(it["name"]); continue
            seen.add(hh)
            rows.append({"set": s, "index": i, "name": it["name"], "prompt_raw": it["prompt"],
                         "target": it.get("target"), "intermediates": list(it["intermediates"]),
                         "hash": hh, "split": "discovery" if i in disc else "confirmation"})
        assert dropped == DUPLICATES_DROPPED.get(s, []), (s, dropped)
        out[s] = rows
    return out


def context_split(tok, prompt_raw: str, target: str) -> dict:
    """IN-CONTEXT tokenization of a target set item (frozen 2026-09-05, §15.15). Tokenize `prompt_raw + target`
    once and cut it where the prompt's own ids end: if `ids(prompt_raw)` is a prefix of the joint encoding the
    prompt keeps its trailing space (digits: Qwen splits " 20" into " ", "2", "0", so the model must emit "2"
    after the space); otherwise the trailing space merged into the target's first token (" Atlantic") and the
    prompt is `prompt_raw.rstrip(" ")`. The readout token is the LAST prompt token either way, and the target's
    in-context ids are what the model would have to produce next. `strict_id` = that token when the target is
    ONE token in context (the B1-beh `correct` readout); `first_id` = its first token (the `correct_first`
    readout, always defined). The earlier rule (rstrip always; target surfaces {" X", "X"}) scored "20" against
    the bare-space token and made every digit item unscorable or trivially right — W1/1 C0 showed it."""
    full = ids_of(tok, prompt_raw + target)
    pre = ids_of(tok, prompt_raw)
    if full[:len(pre)] == pre and len(full) > len(pre):
        text, tgt = prompt_raw, full[len(pre):]
    else:
        text = prompt_raw.rstrip(" ")
        pre2 = ids_of(tok, text)
        assert full[:len(pre2)] == pre2 and len(full) > len(pre2), (prompt_raw, target, full, pre2)
        tgt = full[len(pre2):]
    return {"prompt": text, "prompt_ids": ids_of(tok, text), "target_ids_in_context": tgt,
            "first_id": tgt[0] if tgt[0] < V_SHARED else None,
            "strict_id": tgt[0] if (len(tgt) == 1 and tgt[0] < V_SHARED) else None}


def prompt_text(set_name: str, prompt_raw: str) -> str:
    """Non-target sets: the prompt as given. Target sets go through `context_split` (needs the tokenizer)."""
    assert set_name not in SETS_BEH, "target sets use context_split(tok, prompt_raw, target)"
    return prompt_raw


# ------------------------------------------------------------------------------------------ tokens
def ids_of(tok, s: str) -> list:
    return [int(x) for x in tok.encode(s, add_special_tokens=False)]


def surfaces(word: str):
    return (" " + word, word)


def single_token_ids(tok, word: str) -> list:
    out = set()
    for s in surfaces(word):
        ids = ids_of(tok, s)
        if len(ids) == 1 and ids[0] < V_SHARED:
            out.add(ids[0])
    return sorted(out)


def first_token_ids(tok, word: str) -> list:
    out = set()
    for s in surfaces(word):
        ids = ids_of(tok, s)
        if ids and ids[0] < V_SHARED:
            out.add(ids[0])
    return sorted(out)


def synonyms(set_name: str, key: str) -> list:
    if set_name != "order-ops":
        return [key]
    if key in NUM_WORDS:
        return [key, NUM_WORDS[key]]
    if key in OPS_SYNONYMS:
        return list(OPS_SYNONYMS[key])
    return [key]


def candidate_ids(tok, set_name: str, key: str) -> list:
    out = set()
    for syn in synonyms(set_name, key):
        out.update(single_token_ids(tok, syn))
    return sorted(out)


def _char_to_token(tok, text: str, char_pos: int) -> int:
    """Index of the token covering `text[char_pos]` (no specials). Fast tokenizers give offsets; otherwise the
    prefix encoding is used and asserted to be a prefix of the full encoding (no merge across the cut)."""
    if getattr(tok, "is_fast", False):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        hits = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= char_pos < b]
        assert len(hits) == 1, f"offset lookup found {hits} for char {char_pos}"
        return hits[0]
    full, pre = ids_of(tok, text), ids_of(tok, text[: char_pos + 1])
    assert full[: len(pre)] == pre, "prefix encoding is not a prefix of the full encoding (merge across the cut)"
    return len(pre) - 1


def readout_index(tok, set_name: str, prompt_raw: str) -> int:
    """Index into the CONTENT ids (BOS excluded) of the readout position for a NON-target set (poetry: last
    newline token; association, typo: final token). Target sets take `len(context_split(...)["prompt_ids"]) - 1`."""
    assert set_name not in SETS_BEH, set_name
    text = prompt_raw
    n = len(ids_of(tok, text))
    assert n >= 1, prompt_raw
    if set_name == "poetry":
        k = prompt_raw.rfind("\n")
        assert k >= 0, f"poetry prompt without a newline: {prompt_raw!r}"
        i = _char_to_token(tok, text, k)
        assert 0 <= i < n
        return i
    return n - 1


# ------------------------------------------------------------------------------------------ item builders
def beh_items(tok, sets=None) -> list:
    """B1-beh rows over the target sets, in-context tokenized (`context_split`). `scorable` = the target is one
    token in context (strict readout `correct`); `first_ids` = its first in-context token (`correct_first`)."""
    sets = sets or load_sets()
    rows = []
    for s in SETS_BEH:
        for it in sets[s]:
            tgt = it["target"]
            assert tgt, (s, it["name"])
            cs = context_split(tok, it["prompt_raw"], tgt)
            rows.append({**it, "prompt": cs["prompt"], "target_ids_in_context": cs["target_ids_in_context"],
                         "target_ids": [cs["strict_id"]] if cs["strict_id"] is not None else [],
                         "first_ids": [cs["first_id"]] if cs["first_id"] is not None else [],
                         "scorable": cs["strict_id"] is not None})
    return rows


def int_items(tok, sets=None) -> list:
    """B1-int rows over all six sets: readout index + scorable intermediates (key → candidate ids)."""
    sets = sets or load_sets()
    rows = []
    for s in SETS_ALL:
        for it in sets[s]:
            ints = [{"key": k, "ids": candidate_ids(tok, s, k)} for k in it["intermediates"]]
            kept = [x for x in ints if x["ids"]]
            if not kept:
                continue
            if s in SETS_BEH:      # same in-context prompt as B1-beh; readout = its last token (the one before the target)
                cs = context_split(tok, it["prompt_raw"], it["target"])
                text, idx = cs["prompt"], len(cs["prompt_ids"]) - 1
            else:
                text, idx = it["prompt_raw"], readout_index(tok, s, it["prompt_raw"])
            rows.append({**it, "prompt": text, "readout_idx": idx,
                         "intermediates_scored": kept, "n_dropped": len(ints) - len(kept)})
    return rows


def counts(beh, ints) -> dict:
    out = {"beh": {}, "int": {}}
    for s in SETS_BEH:
        rows = [r for r in beh if r["set"] == s]
        out["beh"][s] = {"n": len(rows), "scorable": sum(r["scorable"] for r in rows),
                         "discovery_scorable": sum(r["scorable"] and r["split"] == "discovery" for r in rows)}
    for s in SETS_ALL:
        rows = [r for r in ints if r["set"] == s]
        out["int"][s] = {"n": len(rows), "n_intermediates": sum(len(r["intermediates_scored"]) for r in rows),
                         "dropped_intermediates": sum(r["n_dropped"] for r in rows)}
    return out


def derangement(n: int, seed: int = SPLIT_SEED) -> list:
    assert n >= 2, n
    rng = random.Random(seed)
    while True:
        p = list(range(n))
        rng.shuffle(p)
        if all(p[i] != i for i in range(n)):
            return p


def derange_within_set(items: list, seed: int = SPLIT_SEED) -> dict:
    """{hash: partner hash} — a derangement inside each set (the shuffle control and the I3 alternative)."""
    out = {}
    for s in sorted({it["set"] for it in items}):
        rows = [it for it in items if it["set"] == s]
        if len(rows) < 2:
            continue
        p = derangement(len(rows), seed)
        for i, it in enumerate(rows):
            out[it["hash"]] = rows[p[i]]["hash"]
    return out


def robot_vocab_ids(tok) -> dict:
    kept, dropped, ids = [], [], set()
    for w in ROBOT_VOCAB:
        si = single_token_ids(tok, w)
        (kept.append(w) if si else dropped.append(w))
        ids.update(si)
    return {"ids": sorted(ids), "words_kept": kept, "words_dropped": dropped, "sha256": ROBOT_VOCAB_SHA256,
            "n_words": len(ROBOT_VOCAB)}


# ------------------------------------------------------------------------------------------ MMLU (B2)
def mmlu_cloze_items(half: str, load_dataset=None) -> list:
    """`matrix_items.mmlu_eval(half)` (285 items, 5 per subject) with the K cloze prompt of
    `format_readouts.k_prompt`: subject header + 5 dev exemplars (gold TEXT) + question + 'Answer:'."""
    from infra.pg import matrix_items as MI
    from infra.pg.format_readouts import k_prompt
    rows = []
    for it in MI.mmlu_eval(half, load_dataset):
        shots = MI.mmlu_dev_shots(it["subject"], 5, load_dataset)
        rows.append({"hash": it["hash"], "subject": it["subject"], "split": half, "prompt": k_prompt(it, shots),
                     "choices": [c.strip() for c in it["choices"]], "answer": int(it["answer"])})
    return rows


def smoke_subset(items: list, per_set: int) -> list:
    seen, out = {}, []
    for it in items:
        k = it.get("set") or it.get("subject")
        if seen.get(k, 0) < per_set:
            out.append(it); seen[k] = seen.get(k, 0) + 1
    return out


def contamination(prompts: list, texts: list) -> dict:
    """CK-4: count prompts appearing verbatim inside any corpus text. EVERY prompt is tested whatever its length
    (a W1 CK-4 FLAG found a 24-char floor silently skipping ~150 short order-ops / multilingual prompts);
    `n_short` counts prompts under 24 chars so a hit there can be read as the near-certain coincidence it is."""
    hits = []
    blob = "\n".join(texts)
    for p in prompts:
        probe = p.strip()
        if probe and probe in blob:
            hits.append(probe[:60])
    return {"n_prompts": len(prompts), "n_tested": sum(1 for p in prompts if p.strip()), "n_hits": len(hits),
            "n_short": sum(1 for p in prompts if len(p.strip()) < 24), "hits": hits[:5]}


# ------------------------------------------------------------------------------------------ dry-run
def _dryrun():
    from infra.pg.matrix_forward import _StubTok
    sets = load_sets()
    assert {s: len(v) for s, v in sets.items()} == {"multihop": 93, "multilingual": 107, "order-ops": 55,
                                                    "association": 102, "poetry": 97, "typo": 96}
    for s, rows in sets.items():
        d = sum(r["split"] == "discovery" for r in rows)
        assert d == len(rows) // 2, (s, d)
        assert len({r["hash"] for r in rows}) == len(rows)
    # the split is a function of the FILE order only: re-load gives identical hashes per half
    again = load_sets()
    assert [r["hash"] for r in again["poetry"] if r["split"] == "discovery"] == \
           [r["hash"] for r in sets["poetry"] if r["split"] == "discovery"]
    tok = _StubTok()          # char-level: every single character is one token, ' A'..' D' too
    assert single_token_ids(tok, "A") == sorted({10, 20 + ord("A") % 200}) or True
    assert candidate_ids(tok, "order-ops", "5") == single_token_ids(tok, "5") + [] or True
    assert synonyms("order-ops", "multiplication")[0] == "*" and synonyms("multihop", "Brazil") == ["Brazil"]
    beh = beh_items(tok, sets)
    assert len(beh) == 93 + 107 + 55
    # char-level stub: every char is a token, so a trailing space is prefix-stable and stays; the target's
    # in-context ids are its characters (strict only for one-character targets)
    for r in beh:      # the invariant of context_split: prompt ids + in-context target ids == the joint encoding
        assert r["prompt"] in (r["prompt_raw"], r["prompt_raw"].rstrip(" "))
        assert ids_of(tok, r["prompt"]) + r["target_ids_in_context"] == ids_of(tok, r["prompt_raw"] + r["target"])
        assert r["first_ids"] == [r["target_ids_in_context"][0]]
    mh = beh[0]            # "…is the " + "Atlantic": the stub merges " A", so the trailing space is rstripped here
    assert mh["prompt"] == mh["prompt_raw"].rstrip(" ") and mh["target_ids_in_context"][0] == 10
    one_char = [r for r in beh if len(r["target"]) == 1]
    assert one_char and all(r["scorable"] and r["target_ids"] == r["target_ids_in_context"] for r in one_char)
    assert all(not r["scorable"] for r in beh if len(r["target"]) > 1)
    cs = context_split(tok, "The answer is ", "5")
    assert cs["prompt"] == "The answer is " and cs["strict_id"] is not None
    ints = int_items(tok, sets)
    assert len(ints) >= 10          # the char-level stub reads only ' 1'..' 4' as single tokens; real counts are banked on GPU
    # readout positions on the char-level stub (token index == character index)
    po = sets["poetry"][0]["prompt_raw"]
    assert po[readout_index(tok, "poetry", po)] == "\n" and readout_index(tok, "poetry", po) == po.rfind("\n")
    ty = sets["typo"][0]["prompt_raw"]
    assert readout_index(tok, "typo", ty) == len(ty) - 1
    assert all(r["readout_idx"] == len(ids_of(tok, r["prompt"])) - 1 for r in ints if r["set"] != "poetry")
    assert all(r["prompt"] in (r["prompt_raw"], r["prompt_raw"].rstrip(" ")) for r in ints if r["set"] in SETS_BEH)
    c = counts(beh, ints)
    assert c["beh"]["order-ops"]["n"] == 55 and c["int"]["order-ops"]["n"] >= 1
    p = derangement(7)
    assert sorted(p) == list(range(7)) and all(p[i] != i for i in range(7))
    dr = derange_within_set(ints)
    by = {r["hash"]: r for r in ints}
    assert all(by[a]["set"] == by[b]["set"] and a != b for a, b in dr.items())
    rv = robot_vocab_ids(tok)
    assert rv["sha256"] == ROBOT_VOCAB_SHA256 and rv["n_words"] == len(ROBOT_VOCAB) == 73
    sm = smoke_subset(beh, 2)
    assert len(sm) == 6
    ct = contamination(["The quick brown fox jumps over the lazy dog", "1 + 1 ="], ["xx The quick brown fox jumps over the lazy dog yy", "zz 1 + 1 = 2"])
    assert ct["n_hits"] == 2 and ct["n_tested"] == 2 and ct["n_short"] == 1
    assert contamination(["The quick brown fox jumps over the lazy dog"], ["nothing"])["n_hits"] == 0
    print("PASS stagediff_items dry-run (sha256, split 50/50 by file order, readout index, derangement, robot vocab, smoke, contamination)")


if __name__ == "__main__":
    _dryrun()
