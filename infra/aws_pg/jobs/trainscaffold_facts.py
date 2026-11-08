"""TRAINSCAFFOLD invented-knowledge battery (prereg §4). The NOVEL capability the ablated backbone is
asked to acquire — difficulty/breadth-matched to MMLU re-entry, base ≈ chance by construction.

~16k synthetic (entity × relation → object) facts over many entity×relation templates, each with 3
type-matched, length-balanced confusable distractors. Three RENDERINGS of every fact:
  * free-gen TRAIN   — "The {relation} of {entity} is" → target " {object}" (the masked fact-span).
  * letter-MC TRAIN  — a 4-choice MCQ, target = the correct letter (format_qa-style).
  * held-out PROBE   — a DIFFERENT-surface-form cloze-recognition item (prompt + 4 candidate
                       completions, scored by likelihood) — the SAME for both formats, so A_free and
                       A_letter are evaluated identically. Tests recall-generalization across surface
                       form (rendering-disjoint, fact-SHARED), NOT recall of unseen associations.
Plus a SHUFFLE twin (permuted entity→object) whose recognition must fail.

Pure (numpy/stdlib); deterministic in --seed. `main` builds + writes + runs the structural offline
checks (rendering-disjointness, distractor type/length balance, no surface leakage). Base-chance is a
model check (the GPU preflight). Fail-loud.

    python -m infra.aws_pg.jobs.trainscaffold_facts --out <dir> --n-entities 4000 --seed 1337
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

CONS = "bdfgklmnprstvz"
VOW = "aeiou"

# 4 relations, each with its own typed invented-object pool (so distractors are type-matched). The
# free-gen prefix, the MC question, and the held-out PROBE frame are all DIFFERENT surface forms.
RELATIONS = [
    {"key": "warden", "free": "The warden of {e} is", "q": "Who is the warden of {e}?",
     "probe": "In the ledger, the post of warden for {e} is held by"},
    {"key": "sigil", "free": "The sigil of {e} is the", "q": "What is the sigil of {e}?",
     "probe": "Heralds record that {e} bears the emblem of the"},
    {"key": "tithe", "free": "The tithe of {e} is paid in", "q": "In what is the tithe of {e} paid?",
     "probe": "Accounts show that {e} renders its tithe in"},
    {"key": "ward", "free": "The ward of {e} lies in", "q": "Where does the ward of {e} lie?",
     "probe": "Maps place the ward belonging to {e} within"},
]


def _syllables(rng, n):
    import numpy as np
    return "".join(CONS[rng.integers(len(CONS))] + VOW[rng.integers(len(VOW))] for _ in range(n))


def _unique_names(rng, k, n_syl=2):
    out, seen = [], set()
    while len(out) < k:
        nm = _syllables(rng, n_syl).capitalize()
        if nm not in seen:
            seen.add(nm)
            out.append(nm)
    return out


def build_battery(n_entities=4000, pool_size=64, seed=1337):
    """Deterministic battery. Returns {facts:[...], meta:{...}, hash}. ~n_entities×len(RELATIONS) facts."""
    import numpy as np
    rng = np.random.default_rng(seed)
    entities = _unique_names(rng, n_entities, n_syl=2)
    # one typed object pool per relation (all 2-syllable invented words → length-balanced).
    pools = {r["key"]: _unique_names(rng, pool_size, n_syl=2) for r in RELATIONS}

    facts = []
    for ei, e in enumerate(entities):
        for r in RELATIONS:
            pool = pools[r["key"]]
            oi = int(rng.integers(len(pool)))
            obj = pool[oi]
            # 3 type-matched distractors from the SAME pool (≠ object).
            dpool = [o for j, o in enumerate(pool) if j != oi]
            dperm = rng.permutation(len(dpool))[:3]
            distractors = [dpool[j] for j in dperm]
            choices = [obj] + distractors
            cperm = rng.permutation(4)                       # shuffle MC positions
            shuf_choices = [choices[j] for j in cperm]
            ans_idx = int(np.where(cperm == 0)[0][0])
            fid = f"{ei}_{r['key']}"
            facts.append({
                "fid": fid, "entity": e, "relation": r["key"], "object": obj,
                "distractors": distractors,
                "free": {"prompt": r["free"].format(e=e), "target": " " + obj},
                "letter_mc": {"prompt": _mc_prompt(r["q"].format(e=e), shuf_choices),
                              "choices": shuf_choices, "answer_idx": ans_idx},
                "probe": {"prompt": r["probe"].format(e=e),
                          "candidates": shuf_choices, "answer_idx": ans_idx},   # held-out, different form
            })
    # SHUFFLE twin: permute entity→object within each relation (associations broken; recognition must fail).
    shuffled = _shuffle_associations(facts, seed=seed + 1)
    h = hashlib.sha1(json.dumps([f["fid"] + f["object"] for f in facts]).encode()).hexdigest()[:12]
    return {"facts": facts, "shuffle": shuffled,
            "meta": {"n_entities": n_entities, "n_facts": len(facts), "pool_size": pool_size,
                     "relations": [r["key"] for r in RELATIONS], "seed": seed},
            "hash": h}


CHOICES = ("A", "B", "C", "D")


def _mc_prompt(question, choices):
    s = question + "\n"
    for i, c in enumerate(choices):
        s += f"{CHOICES[i]}. {c}\n"
    return s + "Answer:"


def _shuffle_associations(facts, seed):
    """Per relation, derange the object assignment so each fact gets a WRONG object (the no-information
    floor). Renders the free + letter-MC TRAIN forms with the wrong object as the answer (+ 3 distractors
    from the pool ≠ wrong object) — a model trained on these must FAIL the (correct) held-out probe (the
    §5/§6 shuffle gate). Eval is always the real fact's probe (matched by fid). Returns a fid→twin dict."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by_rel = {}
    for f in facts:
        by_rel.setdefault(f["relation"], []).append(f)
    out = {}
    for rel, fs in by_rel.items():
        objs = [f["object"] for f in fs]
        # union pool for this relation = all objects seen (for distractor draws on the wrong object)
        relpool = sorted({o for o in objs} | {d for f in fs for d in f["distractors"]})
        perm = rng.permutation(len(fs))
        for i in range(len(fs)):
            if perm[i] == i and len(fs) > 1:
                perm[i], perm[(i + 1) % len(fs)] = perm[(i + 1) % len(fs)], perm[i]
        for i, f in enumerate(fs):
            wrong = objs[perm[i]]
            dpool = [o for o in relpool if o != wrong]
            dperm = rng.permutation(len(dpool))[:3]
            distractors = [dpool[j] for j in dperm]
            choices = [wrong] + distractors
            cperm = rng.permutation(4)
            shuf_choices = [choices[j] for j in cperm]
            ans_idx = int(np.where(cperm == 0)[0][0])
            # reconstruct the letter question stem from the real letter_mc prompt (before the choices).
            q_stem = f["letter_mc"]["prompt"].split("\n", 1)[0]
            out[f["fid"]] = {
                "fid": f["fid"], "object": wrong,
                "free": {"prompt": f["free"]["prompt"], "target": " " + wrong},
                "letter_mc": {"prompt": _mc_prompt(q_stem, shuf_choices),
                              "choices": shuf_choices, "answer_idx": ans_idx},
            }
    return out


def structural_checks(bat):
    """Offline structural validity (no model): rendering-disjoint, distractor balance, no surface leakage."""
    import numpy as np
    facts = bat["facts"]
    rep = {"n_facts": len(facts)}
    # 1. the three renderings of a fact are textually DISTINCT (rendering-disjoint).
    n_disjoint = sum(1 for f in facts
                     if len({f["free"]["prompt"], f["letter_mc"]["prompt"], f["probe"]["prompt"]}) == 3)
    rep["rendering_disjoint_frac"] = round(n_disjoint / len(facts), 4)
    assert rep["rendering_disjoint_frac"] == 1.0, "renderings not all distinct"
    # 2. exactly 4 choices, the correct object present, 3 distractors, answer_idx points at object.
    ok_mc = all(len(f["letter_mc"]["choices"]) == 4
                and f["letter_mc"]["choices"][f["letter_mc"]["answer_idx"]] == f["object"]
                for f in facts)
    rep["mc_wellformed"] = ok_mc
    assert ok_mc, "MC items malformed"
    # 3. distractor type/length balance: object & distractors from the same pool, same char-length dist.
    objlen = np.array([len(f["object"]) for f in facts])
    dislen = np.array([len(d) for f in facts for d in f["distractors"]])
    rep["len_balance"] = {"obj_mean": float(objlen.mean()), "dis_mean": float(dislen.mean())}
    assert abs(objlen.mean() - dislen.mean()) < 0.5, "object/distractor length imbalance (heuristic leak)"
    # 4. the held-out PROBE candidates == the trained choices set (same options, different frame).
    same_opts = all(set(f["probe"]["candidates"]) == set(f["letter_mc"]["choices"]) for f in facts)
    rep["probe_options_match"] = same_opts
    assert same_opts, "probe candidate set differs from trained choices"
    # 5. answer-position balance across A/B/C/D (no positional shortcut).
    pos = np.bincount([f["letter_mc"]["answer_idx"] for f in facts], minlength=4)
    rep["answer_pos_balance"] = pos.tolist()
    assert pos.min() > 0.15 * len(facts), f"answer-position imbalance: {pos.tolist()}"
    rep["STRUCT_OK"] = True
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-entities", type=int, default=4000)
    ap.add_argument("--pool-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bat = build_battery(a.n_entities, a.pool_size, a.seed)
    rep = structural_checks(bat)
    (out / "battery.json").write_text(json.dumps(bat, indent=1))
    (out / "battery_checks.json").write_text(json.dumps(rep, indent=1))
    print(f"[TS-FACTS] OK — {bat['meta']['n_facts']} facts, hash {bat['hash']}, checks {rep}")
    print(f"TS_FACTS_OK n_facts={bat['meta']['n_facts']}")


if __name__ == "__main__":
    main()
