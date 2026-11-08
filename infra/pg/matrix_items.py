"""MENDELEEV item sets (MATRIX_CONTRACT §2 / spec B5) — every list content-hashed, every loader
weight-free and download-free under an injected `load_dataset`.

Hash recipe (the per-item ALIGNMENT KEY, so it must be an identity):
    item_hash(subject, question) = sha1(f"{subject}|{question}").hexdigest()[:16]

It hashes the FULL question, never a prefix. The old recipe (`relearn.qa_item_hash` =
sha1(question[:64])[:12], still used by pre-MENDELEEV programs) is NOT an identity: 285 MMLU discovery
items produce only 280 distinct keys, because five items share a 64-character prefix. The paired-item
bootstrap and `scripts/matrix_verdicts.align()` pair per-item vectors BY THIS KEY, so a collision makes
pairing ambiguous and no behaviour cell can be scored. Every scored set therefore asserts distinct
hashes at BUILD time (`_assert_unique`, naming the colliding questions).

The gate goes on sets used as a PAIRING KEY, and only those: both MMLU eval halves, WMDP eval, arith,
TriviaQA, RFP, the two-hop U / U-shortcut sets, the vision lists. Sets that are dose pools (MMLU
recovery train, WMDP train), exemplar sources (dev), the two-hop pool (validation) or compared
positionally (probe-train) are exempt and each says why — MMLU genuinely repeats rows: 78 duplicate
(subject, question) pairs in `test` (all outside the scored rows[:10]), 1 in `dev`, 1 in `validation`.
A duplicate SOURCE ROW is not a hash defect; pairing a scored vector by an ambiguous key is.

`question_key(question)` (subject-blind, full text) is a SECOND, different key, used only for the
train-vs-eval leak guards. It must stay subject-blind: an eval question reappears in auxiliary_train
under a different subject, so a subject-scoped guard would never fire.

Sets:
  mmlu_eval(half)            discovery = test rows[:5] per subject (285); confirmation = rows[5:10] (285);
                             text_eval order (sorted subject, dataset row order).
  mmlu_dev_shots(subject, n) first n dev rows of the subject (the 5-shot prefix source).
  held_out_subjects()        random.Random(1337).sample(sorted(MMLU_SUBJECTS), 14) — dataset-free
                             (MMLU_SUBJECTS is the fixed 57-subject list, asserted against the data at load).
  mmlu_train(n, seed, ex)    auxiliary_train + dev, minus held-out subjects, minus every eval QUESTION
                             (`ex` must cover BOTH eval halves — asserted), relearn.ladder_subset order with
                             salt f"mdl-{seed}" (nested ladder within a seed), first n.
  rfp_items()                the frozen read_from_passage set: build_synth(20260620, 2000); content sha1
                             asserted to start with the C2 artifact's prefix.
  wmdp_items(split)          cais/wmdp bio + cyber; per config 60/40 by random.Random(1337); train rows
                             whose question appears in eval are dropped (leakage guard).
  arith_items()              200 two-step integer word problems, 6 fixed templates, random.Random(1337);
                             `options` = gold + 3 offset distractors for the candidate readout.
  triviaqa_items()           500 rc.nocontext validation items by a random.Random(1337) permutation; options =
                             gold + 3 other picked items' golds; 5 further items are the cloze exemplars.
  vision_index_lists()       RefCOCO testA / VQAv2 val / TextVQA val, n = 200: the first 200 of
                             np.random.default_rng(1337).permutation(n_split) — exactly what HF
                             `Dataset.shuffle(seed=1337).select(range(200))` yields, i.e. e4_capability's
                             O eval list (seed 1337; battery_tasks carries no VQAv2 list). n_split comes
                             from the hub metadata (`load_dataset_builder(...).info.splits[split]
                             .num_examples`, no download); the 200 rows' light-column hashes come from the
                             on-disk arrow split via `.select(sorted)` (default, "arrow-select" — the GPU
                             box needs the split for images anyway) or, `light=True`, a streaming iterator
                             capped at the max index ("streaming-light" — reads every shard up to it, so a
                             full-split permutation costs the whole split; never the default). Canonical
                             ids `lmms-lab-encoder/*` (legacy `lmms-lab/*` aliases recorded). The record
                             says which path produced its hashes.
  vision_rows(task)          THE loader for the O column: the 200 scored rows WITH images, in index-list
                             order, each carrying `_index`. Takes the cached eval subset when it is on
                             the box (verified row by row first) and falls back to materialising the
                             split otherwise. Every O-column caller goes through this.
  build_vision_subset()      SEEDER ONLY: writes that subset once, on a CPU box, from the canonical path.
  vqav2_train(n, seed)       VQAv2 train from a 6-shard prefix (≈ 21k rows ≥ the 16k REH dose; transfer_a1
                             pattern), hash-disjoint from the eval list.
  manifest()                 {name: {n, sha256(json)}} for every set above + the U set (two_hop_items).

`load_dataset` (and `split_rows` for the vision lists) is injectable everywhere (defaults:
`datasets.load_dataset`, hub split metadata); `_dryrun()` passes synthetic ones and touches no network.
Fail-loud: asserts, no try/except.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import random

import numpy as np

from infra.pg import read_from_passage as RFP
from infra.pg.relearn import ladder_subset
from infra.pg.text_eval import MMLU_REPO

SEED = 1337
HALVES = {"discovery": (0, 5), "confirmation": (5, 10)}
HELD_OUT_N = 14
MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
    "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
)
assert len(MMLU_SUBJECTS) == 57 and tuple(sorted(MMLU_SUBJECTS)) == MMLU_SUBJECTS

RFP_SEED, RFP_N, RFP_CONTENT_SHA1_PREFIX = 20260620, 2000, "bc8744f5eec2"   # C2 artifact (result doc :122)

WMDP_REPO, WMDP_CONFIGS, WMDP_TRAIN_FRAC = "cais/wmdp", ("wmdp-bio", "wmdp-cyber"), 0.6

TRIVIAQA_REPO, TRIVIAQA_CONFIG, TRIVIAQA_SPLIT = "mandarjoshi/trivia_qa", "rc.nocontext", "validation"
TRIVIAQA_N, TRIVIAQA_N_SHOTS = 500, 5
TRIVIAQA_SUBJECT = "trivia"          # the set tag in the item_hash key (these items carry no MMLU subject)

ARITH_N, ARITH_N_SHOTS = 200, 5
ARITH_SUBJECT = "arithmetic"

# Canonical hub ids (the lmms-lab/* ids now 307-redirect to lmms-lab-encoder/*; kept as legacy aliases).
# `columns` = the light columns hashed per selected row (no image decode) — the ROW3 evaluator verifies
# each selected row against these hashes, which also catches a streaming-vs-arrow order mismatch.
VISION = {"refcoco": {"dataset": "lmms-lab-encoder/RefCOCO", "legacy_ids": ["lmms-lab/RefCOCO"],
                      "split": "testA", "columns": ("answer", "bbox")},
          "vqav2": {"dataset": "lmms-lab-encoder/VQAv2", "legacy_ids": ["lmms-lab/VQAv2"],
                    "split": "validation", "columns": ("question", "question_id")},
          "textvqa": {"dataset": "lmms-lab-encoder/textvqa", "legacy_ids": ["lmms-lab/textvqa"],
                      "split": "validation", "columns": ("question", "question_id")}}
VISION_SEED, VISION_N = 1337, 200
VQAV2_TRAIN_SHARDS = [f"data/train-{i:05d}-of-00136.parquet" for i in range(6)]   # ≈ 3.5k rows/shard ≥ 16k REH dose


def item_hash(subject: str, question: str) -> str:
    """The per-item alignment key: sha1 of the FULL subject + question, 16 hex. An identity, not a
    fingerprint — `align()` and the paired-item bootstrap pair vectors by it."""
    return hashlib.sha1(f"{subject}|{question}".encode()).hexdigest()[:16]


def question_key(question: str) -> str:
    """Subject-BLIND key over the full question text — the train-vs-eval LEAK guard, not the alignment
    key. It must not carry the subject: an eval question reappears in MMLU auxiliary_train under a
    different (usually empty) subject, so a subject-scoped comparison would never fire and the guard
    would silently become a no-op."""
    return hashlib.sha1(question.encode()).hexdigest()[:16]


def _assert_unique(items: list, name: str) -> list:
    """Build-time identity gate: every scored set fails loud here rather than mispairing later. The
    message names the colliding questions so the defect is diagnosable from the traceback alone."""
    seen: dict = {}
    for it in items:
        h = it["hash"]
        if h in seen:
            raise AssertionError(
                f"{name}: item_hash collision {h}\n  A: {seen[h].get('subject')!r} | {seen[h]['question']!r}"
                f"\n  B: {it.get('subject')!r} | {it['question']!r}")
        seen[h] = it
    return items


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _hf(*args, **kw):
    from datasets import load_dataset
    return load_dataset(*args, **kw)


@functools.lru_cache(maxsize=None)
def _rows(load_dataset, repo, config, split):
    """Materialised dict rows of one split (cached per loader). NOT for image datasets."""
    ld = load_dataset or _hf
    ds = ld(repo, config, split=split) if config is not None else ld(repo, split=split)
    rows = [dict(r) for r in ds]
    assert rows, f"{repo}/{config}/{split}: empty split"
    return rows


# ------------------------------------------------------------------------------------------ MMLU
def _mmlu_item(r: dict) -> dict:
    choices = [str(c) for c in r["choices"]]
    assert len(choices) == 4, f"MMLU item must have 4 choices, got {len(choices)}"
    return {"subject": r["subject"], "question": r["question"], "choices": choices,
            "answer": int(r["answer"]), "hash": item_hash(r["subject"], r["question"])}


def _mmlu_by_subject(load_dataset, split) -> dict:
    by: dict = {}
    for r in _rows(load_dataset, MMLU_REPO, "all", split):
        by.setdefault(r["subject"], []).append(r)
    if split in ("test", "dev"):
        assert tuple(sorted(by)) == MMLU_SUBJECTS, f"MMLU {split} subjects != the fixed 57-list"
    return by


def mmlu_eval(half: str, load_dataset=None) -> list:
    lo, hi = HALVES[half]
    by = _mmlu_by_subject(load_dataset, "test")
    items = []
    for subj in sorted(by):
        rows = by[subj][lo:hi]
        assert len(rows) == hi - lo, f"{subj}: {len(rows)} test rows in [{lo},{hi})"
        items += [_mmlu_item(r) for r in rows]
    assert len(items) == (hi - lo) * len(MMLU_SUBJECTS)
    return _assert_unique(items, f"mmlu_eval({half})")


def mmlu_dev_rows(load_dataset=None) -> list:
    # No identity gate: dev is the EXEMPLAR source, never scored or paired, and it genuinely repeats one
    # row (college_physics "A refracting telescope consists of two converging lenses…", identical choices).
    by = _mmlu_by_subject(load_dataset, "dev")
    return [_mmlu_item(r) for subj in sorted(by) for r in by[subj]]


def mmlu_validation_rows(load_dataset=None) -> list:
    """MMLU validation split (1531 rows): the U (two-hop) pool. It is in neither eval half (test) nor the
    recovery training pool (auxiliary_train + dev). Chosen 2026-09-03 after the smoke showed only 62 of the
    285 dev rows pass the two-hop eligibility filter."""
    # No identity gate: this is the two-hop POOL, not a scored set, and the split genuinely repeats one
    # row (high_school_macroeconomics "Stagflation occurs when", different choices). `build_two_hop`
    # deduplicates the pool on this same key, so at most one U item is ever built from such a pair.
    by = _mmlu_by_subject(load_dataset, "validation")
    return [_mmlu_item(r) for subj in sorted(by) for r in by[subj]]


def mmlu_dev_shots(subject: str, n: int, load_dataset=None) -> list:
    rows = _mmlu_by_subject(load_dataset, "dev")[subject][:n]
    assert len(rows) == n, f"{subject}: only {len(rows)} dev shots (<{n})"
    return [_mmlu_item(r) for r in rows]


def held_out_subjects() -> list:
    return random.Random(SEED).sample(sorted(MMLU_SUBJECTS), HELD_OUT_N)


def mmlu_train(n: int | None, seed: int, exclude_hashes, load_dataset=None) -> list:
    ex = set(exclude_hashes)
    assert ex, "exclude_hashes must hold every eval hash (both halves)"
    for half in HALVES:
        missing = {it["hash"] for it in mmlu_eval(half, load_dataset)} - ex
        assert not missing, f"exclude_hashes misses {len(missing)} {half} hashes"
    held = set(held_out_subjects())
    pool = (_rows(load_dataset, MMLU_REPO, "all", "auxiliary_train")
            + _rows(load_dataset, MMLU_REPO, "all", "dev"))
    # No _assert_unique here: this is a training pool, never an alignment key. A repeated question in
    # MMLU auxiliary_train is a duplicate row, not an ambiguous key.
    ev_q = {question_key(it["question"]) for half in HALVES for it in mmlu_eval(half, load_dataset)}
    rows = [_mmlu_item(r) for r in pool
            if r["subject"] not in held and question_key(r["question"]) not in ev_q]
    sub = ladder_subset(rows, len(rows) if n is None else n, salt=f"mdl-{seed}")
    assert not ({it["hash"] for it in sub} & ex)
    return sub


# ------------------------------------------------------------------------------------------ RFP
def rfp_items() -> dict:
    items, exemplars, meta = RFP.build_synth(RFP_SEED, RFP_N)
    assert meta["content_sha1"].startswith(RFP_CONTENT_SHA1_PREFIX), \
        f"RFP set drifted from the C2 artifact: {meta['content_sha1']}"
    # RFP items are keyed by RFP.item_hash (sha1 of passage + question + choices, already full-content),
    # so the identity gate runs over THAT key rather than a "hash" field the items do not carry.
    _assert_unique([{**it, "hash": RFP.item_hash(it)} for it in items], "rfp_items")
    return {"items": items, "exemplars": exemplars, "meta": meta}


# ------------------------------------------------------------------------------------------ WMDP
def wmdp_items(split: str, load_dataset=None) -> list:
    assert split in ("train", "eval"), split
    out = []
    for cfg in WMDP_CONFIGS:
        rows = _rows(load_dataset, WMDP_REPO, cfg, "test")
        idx = list(range(len(rows)))
        random.Random(SEED).shuffle(idx)
        n_tr = int(round(WMDP_TRAIN_FRAC * len(rows)))
        tr, ev = sorted(idx[:n_tr]), sorted(idx[n_tr:])
        ev_q = {question_key(rows[i]["question"]) for i in ev}
        chosen = ev if split == "eval" else [i for i in tr if question_key(rows[i]["question"]) not in ev_q]
        for i in chosen:
            out.append({"config": cfg, "idx": i, **_mmlu_item({**rows[i], "subject": cfg})})
    # The eval split is scored and paired, so it must be an identity; the train split is an unlearning
    # dose pool (duplicate source rows are legitimate there), so it is exempt.
    return _assert_unique(out, "wmdp_eval") if split == "eval" else out


# ------------------------------------------------------------------------------------------ arithmetic
_NAMES = ["Maya", "Liam", "Noah", "Ava", "Ethan", "Zoe", "Omar", "Priya", "Kenji", "Sara", "Leo", "Nina"]
_OBJS = ["apples", "marbles", "stickers", "books", "coins", "pencils", "cookies", "cards", "shells", "stamps"]


def _arith_problem(rng: random.Random):
    t = rng.randrange(6)
    name, name2 = rng.sample(_NAMES, 2)
    obj = rng.choice(_OBJS)
    if t == 0:
        a, b = rng.randint(3, 12), rng.randint(3, 12)
        c = rng.randint(1, a * b - 1)
        q = (f"{name} buys {a} packs of {obj} with {b} {obj} in each pack, then gives away {c} {obj}. "
             f"How many {obj} does {name} have left?")
        ans = a * b - c
    elif t == 1:
        a, b = rng.randint(5, 60), rng.randint(2, 30)
        q = f"{name} has {a} {obj}. {name2} has {b} more {obj} than {name}. How many {obj} do they have together?"
        ans = 2 * a + b
    elif t == 2:
        a, b, c = rng.randint(4, 15), rng.randint(2, 9), rng.randint(1, 20)
        q = (f"A box holds {a} {obj}. {name} fills {b} boxes and then adds {c} extra {obj}. "
             f"How many {obj} does {name} have in total?")
        ans = a * b + c
    elif t == 3:
        a, b = rng.randint(8, 25), rng.randint(2, 9)
        c = rng.randint(1, a * b - 1)
        q = (f"{name} earns {a} dollars per hour and works {b} hours. After spending {c} dollars, "
             f"how many dollars does {name} have left?")
        ans = a * b - c
    elif t == 4:
        c, each = rng.randint(2, 6), rng.randint(2, 15)
        a = rng.randint(1, c * each - 1)
        q = (f"{name} had {a} {obj} and bought {c * each - a} more, then shared them equally among {c} friends. "
             f"How many {obj} did each friend get?")
        ans = each
    else:
        a, b = rng.randint(5, 30), rng.randint(2, 9)
        c = a * b + rng.randint(1, 100)
        q = f"{name} reads {a} pages a day for {b} days. The book has {c} pages. How many pages remain after {b} days?"
        ans = c - a * b
    assert ans >= 0
    return t, q, ans


def _arith_options(rng: random.Random, gold: int):
    offs = (-10, -5, -3, -2, -1, 1, 2, 3, 5, 10)
    ds: set = set()
    while len(ds) < 3:
        v = gold + rng.choice(offs)
        if v >= 0 and v != gold:
            ds.add(v)
    opts = [gold] + sorted(ds)
    rng.shuffle(opts)
    return [str(o) for o in opts], opts.index(gold)


def _arith_build(seed: int, n: int) -> list:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        t, q, ans = _arith_problem(rng)
        options, gold_idx = _arith_options(rng, ans)
        items.append({"subject": ARITH_SUBJECT, "question": q, "answer": str(ans), "options": options,
                      "gold_idx": gold_idx, "template": t, "hash": item_hash(ARITH_SUBJECT, q)})
    return items


def arith_items() -> list:
    return _assert_unique(_arith_build(SEED, ARITH_N), "arith_items")


def arith_exemplars() -> list:
    ex = _assert_unique(_arith_build(SEED + 1, ARITH_N_SHOTS), "arith_exemplars")
    assert not ({e["hash"] for e in ex} & {it["hash"] for it in arith_items()})
    return ex


# ------------------------------------------------------------------------------------------ TriviaQA
def triviaqa_items(load_dataset=None) -> dict:
    rows = _rows(load_dataset, TRIVIAQA_REPO, TRIVIAQA_CONFIG, TRIVIAQA_SPLIT)
    perm = list(range(len(rows)))
    random.Random(SEED).shuffle(perm)
    picked, seen = [], set()
    for i in perm:
        r = rows[i]
        gold, q = str(r["answer"]["value"]).strip(), str(r["question"]).strip()
        # The rc.nocontext validation split repeats questions verbatim (e.g. "Who was Poopdeck Pappys most
        # famous son?"). This IS a scored set, so a repeat is an ambiguous alignment key, not a harmless
        # duplicate row: keep the first occurrence under the fixed permutation and take the next row.
        if not gold or not q or item_hash(TRIVIAQA_SUBJECT, q) in seen:
            continue
        seen.add(item_hash(TRIVIAQA_SUBJECT, q))
        picked.append({"subject": TRIVIAQA_SUBJECT, "question": q, "answer": gold,
                       "aliases": [str(a) for a in r["answer"]["aliases"]],
                       "question_id": r["question_id"], "hash": item_hash(TRIVIAQA_SUBJECT, q)})
        if len(picked) == TRIVIAQA_N + TRIVIAQA_N_SHOTS:
            break
    assert len(picked) == TRIVIAQA_N + TRIVIAQA_N_SHOTS, len(picked)
    _assert_unique(picked, "triviaqa (items + exemplars)")
    items, exemplars = picked[:TRIVIAQA_N], picked[TRIVIAQA_N:]
    rng = random.Random(SEED)
    golds = [it["answer"] for it in items]
    for i, it in enumerate(items):
        others: set = set()
        while len(others) < 3:
            j = rng.randrange(len(items))
            if j != i and golds[j].lower() != it["answer"].lower() and golds[j].lower() not in {o.lower() for o in others}:
                others.add(golds[j])
        opts = [it["answer"]] + sorted(others)
        rng.shuffle(opts)
        it["options"], it["gold_idx"] = opts, opts.index(it["answer"])
    return {"items": items, "exemplars": exemplars}


# ------------------------------------------------------------------------------------------ vision
def _row_hash(cols: dict) -> str:
    return hashlib.sha1(json.dumps(cols, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _perm_indices(seed: int, n_split: int, n: int) -> list:
    """== datasets.Dataset.shuffle(seed=seed).select(range(n)) (np.random.default_rng permutation)."""
    assert n_split >= n, f"split has {n_split} < {n}"
    return np.random.default_rng(seed).permutation(n_split)[:n].tolist()


def _hf_split_rows(repo: str, split: str) -> int:
    """Row count from the hub metadata (README dataset_info) — no data download, nothing on disk."""
    from datasets import load_dataset_builder
    info = load_dataset_builder(repo).info
    assert info.splits and split in info.splits, f"{repo}: no split metadata for {split!r}"
    return int(info.splits[split].num_examples)


def _select_rows(ld, repo: str, split: str, indices: list, columns) -> list:
    """The rows at `indices` from the on-disk (arrow) split via `.select(sorted)` — the default: the GPU box
    holds the split for the images anyway (e4_capability pattern); only `columns` are read."""
    order = sorted(indices)
    sub = ld(repo, split=split).select(order)
    cols = {c: sub[c] for c in columns}
    by_idx = {i: {c: cols[c][k] for c in columns} for k, i in enumerate(order)}
    return [by_idx[i] for i in indices]


def _stream_rows(ld, repo: str, split: str, indices: list, columns) -> list:
    """`light=True` path: the rows at `indices` through a STREAMING iterator restricted to `columns`, stopping
    at the max index. Nothing lands on disk, but every shard up to that index is read (image bytes
    included) — with a permutation over a 214k-row split that is the whole split. Use only when the caller
    wants hashes without the images on disk; never as the default."""
    wanted = set(indices)
    it = ld(repo, split=split, streaming=True).select_columns(list(columns))
    rows = {}
    for i, row in enumerate(it):
        if i in wanted:
            rows[i] = {c: row[c] for c in columns}
        if i >= max(indices):
            break
    assert len(rows) == len(wanted), f"{repo}/{split}: streamed {len(rows)} of {len(wanted)} wanted rows"
    return [rows[i] for i in indices]


# ------------------------------------------------------- the 200 scored rows, not the split they came from
# WHY (2026-09-03, measured): `_select_rows` above materialises the WHOLE split to read 200 rows —
# 214,354 image-bearing rows for VQAv2 — so two jobs each spent > 1 h pulling vision data from the hub
# at ~2.8 MB/s before their first GPU stage, and the seeded image cache came to 181.8 GB, which does not
# fit a 200 GB job volume beside a row's weights and was therefore gated off and unused.
# THE SUBSET is the 200 rows themselves, images included, written once on a CPU box as a
# `datasets.save_to_disk` directory (tens of MB per task) and restored by `run_job.sh` into
# `EVAL_SUBSET_DIR`. It is a CACHE: absent, every path below falls back to materialising the split.
# IDENTITY IS THE BAR, NOT SPEED. The frozen index (`vision_eval_index.json`, git-tracked, built by
# `seed_eval_subsets_cpu.sh` from the canonical arrow-select path) holds the indices and the per-row
# light-column hashes. EVERY load re-derives the permutation from (seed, n_split), checks the subset's
# `_index` column against it, and re-hashes every row's light columns against the frozen hash. Nothing
# restored from S3 is trusted; a single mismatch raises rather than scoring a different item set.
EVAL_SUBSET_DIR = os.environ.get("PG_EVAL_SUBSETS", "/opt/hf-cache/eval_subsets")
FROZEN_VISION_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_eval_index.json")


def frozen_vision_index() -> dict:
    """The git-tracked frozen index lists (indices + per-row hashes). This file is the authority a
    cached subset is checked against, so it is a hard requirement wherever a subset is used."""
    with open(FROZEN_VISION_INDEX) as f:
        return json.load(f)


def verify_vision_subset(ds, frozen: dict, task: str) -> list:
    """Row-by-row identity of a cached subset against the frozen index. Returns the recomputed hashes.
    Three independent checks, all fatal: the frozen indices ARE the (seed, n_split) permutation; the
    subset rows are those indices, in that order; every row's light-column hash is the frozen one."""
    cols = list(frozen["columns"])
    idx = _perm_indices(frozen["seed"], frozen["n_split"], frozen["n"])
    assert idx == list(frozen["indices"]), \
        f"{task}: the frozen indices are not the seed-{frozen['seed']} permutation of {frozen['n_split']} rows"
    assert len(ds) == frozen["n"], f"{task}: subset holds {len(ds)} rows, the frozen index wants {frozen['n']}"
    assert [int(i) for i in ds["_index"]] == idx, f"{task}: subset row order is not the index-list order"
    vals = {c: ds.select_columns(cols)[c] for c in cols}   # light columns only: no image decode
    got = [_row_hash({c: vals[c][k] for c in cols}) for k in range(len(ds))]
    bad = [k for k, (a, b) in enumerate(zip(got, frozen["hashes"])) if a != b]
    assert not bad, (f"{task}: {len(bad)} of {frozen['n']} cached rows do not match the frozen row hash "
                     f"(first offenders at list positions {bad[:5]}) — refusing to score them")
    return got


def _open_vision_subset(task: str):
    """The cached subset for `task`, or None when this box has none. Not a try/except: a subset
    directory that exists but cannot be read or verified must raise, not degrade to a silent refetch."""
    d = os.path.join(EVAL_SUBSET_DIR, task)
    if not os.path.isdir(d):
        return None
    from datasets import load_from_disk
    return load_from_disk(d)


def vision_rows(task: str, load_dataset=None) -> list:
    """The 200 scored rows of `task` WITH images, in index-list order, each carrying `_index`.
    Cached subset first (verified against the frozen index); the full split only as the fallback."""
    assert task in VISION, task
    ds = _open_vision_subset(task)
    if ds is not None:
        verify_vision_subset(ds, frozen_vision_index()[task], task)
        return [dict(ds[k]) for k in range(len(ds))]
    spec = VISION[task]
    full = (load_dataset or _hf)(spec["dataset"], split=spec["split"])
    idx = _perm_indices(VISION_SEED, len(full), VISION_N)
    rows = [dict(full[int(i)]) for i in idx]
    for r, i in zip(rows, idx):
        r["_index"] = int(i)
    return rows


def build_vision_subset(task: str, out_dir: str, load_dataset=None, split_rows=None) -> dict:
    """SEEDER ONLY (`infra/aws_pg/seed_eval_subsets_cpu.sh`, CPU box): materialise the split ONCE, write
    the 200 scored rows (images included) to `out_dir`, and return the frozen index record for `task`.
    The hashes are the CANONICAL ones — `_select_rows` on the same split on the same box — and the
    written subset is reloaded and re-verified against them before this returns."""
    spec = VISION[task]
    ld, nrows = load_dataset or _hf, split_rows or _hf_split_rows
    n_split = nrows(spec["dataset"], spec["split"])
    idx = _perm_indices(VISION_SEED, n_split, VISION_N)
    full = ld(spec["dataset"], split=spec["split"])
    assert len(full) == n_split, f"{task}: split holds {len(full)} rows, `split_rows` says {n_split}"
    hashes = [_row_hash(r) for r in _select_rows(ld, spec["dataset"], spec["split"], idx, spec["columns"])]
    assert len(set(hashes)) == len(hashes), f"{task}: duplicate row hashes in the canonical list"
    sub = full.select(idx).flatten_indices().add_column("_index", [int(i) for i in idx])
    sub.save_to_disk(out_dir)
    frozen = {**spec, "columns": list(spec["columns"]), "seed": VISION_SEED, "n": VISION_N,
              "n_split": n_split, "indices": idx, "hashes": hashes}
    from datasets import load_from_disk
    verify_vision_subset(load_from_disk(out_dir), frozen, task)
    return frozen


def vision_index_lists(load_dataset=None, split_rows=None, light=False) -> dict:
    """Index derivation is free (hub `num_examples`); the 200 row hashes come from the cached subset
    (`hash_path` "subset-cached", verified against the frozen index), else the arrow split
    ("arrow-select"), else with `light=True` the capped streaming path ("streaming-light").
    `hash_path` names the provenance of the hashes and so enters `manifest()`'s sha256; the identity a
    reader compares across runs is `indices` + `hashes`, which the three paths agree on by assertion."""
    ld, nrows = load_dataset or _hf, split_rows or _hf_split_rows
    fetch, path = (_stream_rows, "streaming-light") if light else (_select_rows, "arrow-select")
    out = {}
    for task, spec in VISION.items():
        ds = None if light else _open_vision_subset(task)
        if ds is not None:
            fz = frozen_vision_index()[task]
            n_split, idx, hashes, hp = fz["n_split"], list(fz["indices"]), verify_vision_subset(ds, fz, task), "subset-cached"
        else:
            n_split = nrows(spec["dataset"], spec["split"])
            idx = _perm_indices(VISION_SEED, n_split, VISION_N)
            hashes = [_row_hash(r) for r in fetch(ld, spec["dataset"], spec["split"], idx, spec["columns"])]
            hp = path
        assert len(set(hashes)) == len(hashes), \
            f"vision_index_lists[{task}]: {len(hashes) - len(set(hashes))} duplicate row hashes"
        out[task] = {**spec, "columns": list(spec["columns"]), "seed": VISION_SEED, "n_split": n_split,
                     "indices": idx, "hashes": hashes, "hash_path": hp}
    return out


def vqav2_train(n: int, seed: int, load_dataset=None, split_rows=None) -> dict:
    """VQAv2 train subset for REH from a fixed shard prefix (transfer_a1 pattern; the REH trainer needs these
    rows with images on disk anyway). Hash-disjoint from the eval index list, asserted."""
    ld = load_dataset or _hf
    spec = VISION["vqav2"]
    ds = ld(spec["dataset"], data_files={"train": VQAV2_TRAIN_SHARDS}, split="train", verification_mode="no_checks")
    idx = _perm_indices(seed, len(ds), n)
    sub = ds.select(idx)
    cols = {c: sub[c] for c in spec["columns"]}
    hashes = [_row_hash({c: cols[c][k] for c in spec["columns"]}) for k in range(n)]
    ev = set(vision_index_lists(load_dataset, split_rows)["vqav2"]["hashes"])
    assert not (set(hashes) & ev), "VQAv2 train subset overlaps the eval list"
    return {"dataset": spec["dataset"], "legacy_ids": spec["legacy_ids"], "split": "train",
            "data_files": VQAV2_TRAIN_SHARDS, "seed": seed, "n": n, "indices": idx, "hashes": hashes}


# ------------------------------------------------------------------------------------------ manifest
def manifest(load_dataset=None, split_rows=None) -> dict:
    from infra.pg.two_hop_items import build_two_hop
    ld = load_dataset
    disc, conf = mmlu_eval("discovery", ld), mmlu_eval("confirmation", ld)
    rfp, tq, vis = rfp_items(), triviaqa_items(ld), vision_index_lists(ld, split_rows)
    two = build_two_hop(load_dataset=ld)
    sets = {
        "mmlu_discovery": disc, "mmlu_confirmation": conf,
        "held_out_subjects": held_out_subjects(),
        "mmlu_train_pool": mmlu_train(None, 0, {it["hash"] for it in disc + conf}, ld),
        "rfp": rfp["items"], "rfp_exemplars": rfp["exemplars"],
        "wmdp_train": wmdp_items("train", ld), "wmdp_eval": wmdp_items("eval", ld),
        "arith": arith_items(), "arith_exemplars": arith_exemplars(),
        "triviaqa": tq["items"], "triviaqa_exemplars": tq["exemplars"],
        "two_hop": two["items"], "two_hop_shortcut": two["shortcut"], "two_hop_exemplars": two["exemplars"],
        **{f"vision_{k}": v for k, v in vis.items()},
    }
    return {k: {"n": len(v), "sha256": _sha256(v)} for k, v in sets.items()}


# ------------------------------------------------------------------------------------------ dry-run
class _FakeSplit(list):
    """Enough of `datasets.Dataset` for this module: len, iteration, select, column access."""
    def select(self, idx):
        return _FakeSplit(list.__getitem__(self, i) for i in idx)

    def __getitem__(self, k):
        if isinstance(k, str):
            return [r[k] for r in self]
        return list.__getitem__(self, k)


class _FakeStream:
    """Enough of `datasets.IterableDataset`: select_columns + iteration; no len, no select."""
    def __init__(self, rows, columns=None):
        self.rows, self.columns = rows, columns

    def select_columns(self, columns):
        return _FakeStream(self.rows, list(columns))

    def __iter__(self):
        for r in self.rows:
            yield {c: r[c] for c in self.columns} if self.columns else dict(r)


class _FakeSubset(_FakeSplit):
    """Enough of a `save_to_disk` subset for `verify_vision_subset`: len, column access, select_columns."""
    def select_columns(self, columns):
        return _FakeSubset({c: r[c] for c in columns} for r in self)


FAKE_VISION_ROWS = 1000


def fake_split_rows(repo, split) -> int:
    return FAKE_VISION_ROWS


def fake_load_dataset(repo, config=None, split=None, streaming=False, **kw):
    """Synthetic rows with the real schemas; deterministic; one planted eval/train collision."""
    rng = random.Random(f"{repo}|{config}|{split}|{sorted(kw.items())}")
    words = ["pear", "moon", "tiger", "salt", "quartz", "river", "kite", "lamp", "nest", "oak", "vine", "wolf"]

    def mmlu_row(subj, i, tag):
        return {"subject": subj, "question": f"{tag} question {i} about {subj} {rng.random():.6f}?",
                "choices": rng.sample(words, 4), "answer": rng.randrange(4)}
    if repo == MMLU_REPO:
        if split == "auxiliary_train":
            rows = [mmlu_row("", i, "aux") for i in range(400)]
            rows[8] = {**rows[8], "subject": held_out_subjects()[0]}                     # planted held-out row
            rows[9] = {**rows[9], "question": fake_load_dataset(MMLU_REPO, "all", split="test")[3]["question"]}  # planted eval leak
            return _FakeSplit(rows)
        per = {"dev": 5, "test": 12, "validation": 27}[split]   # validation ≈ 1531 real rows
        return _FakeSplit(mmlu_row(s, i, split) for s in MMLU_SUBJECTS for i in range(per))
    if repo == WMDP_REPO:
        rows = [{"question": f"{config} q{i} {rng.random():.6f}", "choices": [f"c{j}" for j in range(4)],
                 "answer": rng.randrange(4)} for i in range(50)]
        rows[1] = {**rows[1], "question": rows[0]["question"]}                          # planted duplicate
        return _FakeSplit(rows)
    if repo == TRIVIAQA_REPO:
        return _FakeSplit({"question": f"Trivia question {i} {rng.random():.6f}?", "question_id": f"tq{i}",
                           "answer": {"value": f"Answer{i}", "aliases": [f"Answer{i}", f"answer {i}"]}}
                          for i in range(600))
    if repo in {s["dataset"] for s in VISION.values()}:
        assert not (streaming and "data_files" in kw)
        n = 3500 * len(kw["data_files"]["train"]) if "data_files" in kw else FAKE_VISION_ROWS
        tag = "train" if "data_files" in kw else split
        rows = [{"question": f"{tag} q{i}", "question_id": f"{tag}-{i}", "answer": [f"expr {i}"],
                 "bbox": [i, i, 10, 10], "image": object()} for i in range(n)]
        return _FakeStream(rows) if streaming else _FakeSplit(rows)
    raise KeyError((repo, config, split))


def _dryrun():
    global EVAL_SUBSET_DIR
    EVAL_SUBSET_DIR = "/nonexistent/pg-eval-subsets"    # the synthetic paths must never read a real cache
    ld, sr = fake_load_dataset, fake_split_rows
    disc, conf = mmlu_eval("discovery", ld), mmlu_eval("confirmation", ld)
    assert len(disc) == len(conf) == 285 and disc[0]["subject"] == "abstract_algebra"
    assert not ({i["hash"] for i in disc} & {i["hash"] for i in conf})
    assert [i["subject"] for i in disc] == [s for s in MMLU_SUBJECTS for _ in range(5)]
    shots = mmlu_dev_shots("anatomy", 5, ld)
    assert len(shots) == 5 and all(set(s) >= {"question", "choices", "answer", "subject"} for s in shots)
    assert mmlu_dev_shots("anatomy", 5, ld) == shots and len(mmlu_dev_rows(ld)) == 285

    held = held_out_subjects()
    assert len(held) == 14 == len(set(held)) and held == held_out_subjects()
    ev = {i["hash"] for i in disc + conf}
    tr200, tr100 = mmlu_train(200, 0, ev, ld), mmlu_train(100, 0, ev, ld)
    assert tr200[:100] == tr100 and not ({i["hash"] for i in tr200} & ev)
    assert all(i["subject"] not in held for i in mmlu_train(None, 0, ev, ld))
    assert len(mmlu_train(None, 0, ev, ld)) == 400 + 285 - 1 - 14 * 5 - 1   # pool − planted leak − held-out dev − held-out aux
    assert mmlu_train(50, 1, ev, ld) != mmlu_train(50, 0, ev, ld)

    rfp = rfp_items()
    assert len(rfp["items"]) == RFP_N and len(rfp["exemplars"]) == 5

    wt, we = wmdp_items("train", ld), wmdp_items("eval", ld)
    assert not ({i["hash"] for i in wt} & {i["hash"] for i in we})
    assert len(we) == 40 and len(wt) in (59, 60) and {i["config"] for i in we} == set(WMDP_CONFIGS)

    ar, ax = arith_items(), arith_exemplars()
    assert len(ar) == 200 and ar == arith_items() and len(ax) == 5
    for it in ar:
        assert it["options"][it["gold_idx"]] == it["answer"] and len(set(it["options"])) == 4
        assert all(int(o) >= 0 for o in it["options"]) and int(it["answer"]) >= 0
    assert len({it["template"] for it in ar}) == 6

    tq = triviaqa_items(ld)
    assert len(tq["items"]) == 500 and len(tq["exemplars"]) == 5 and tq == triviaqa_items(ld)
    for it in tq["items"]:
        assert it["options"][it["gold_idx"]] == it["answer"] and len({o.lower() for o in it["options"]}) == 4

    vis = vision_index_lists(ld, sr)
    assert set(vis) == set(VISION) and all(len(v["indices"]) == 200 == len(v["hashes"]) for v in vis.values())
    assert all(v["hash_path"] == "arrow-select" for v in vis.values())
    lite = vision_index_lists(ld, sr, light=True)                     # both paths agree row for row
    assert all(lite[t]["hashes"] == vis[t]["hashes"] and lite[t]["hash_path"] == "streaming-light" for t in vis)
    assert vis["refcoco"]["indices"] == np.random.default_rng(1337).permutation(FAKE_VISION_ROWS)[:200].tolist()
    assert all(v["dataset"].startswith("lmms-lab-encoder/") and v["legacy_ids"][0].startswith("lmms-lab/") for v in vis.values())
    i0 = vis["vqav2"]["indices"][0]                                   # streamed rows are the permuted ones
    assert vis["vqav2"]["hashes"][0] == _row_hash({"question": f"validation q{i0}", "question_id": f"validation-{i0}"})
    assert vis["refcoco"]["hashes"][0] == _row_hash({"answer": [f"expr {vis['refcoco']['indices'][0]}"],
                                                     "bbox": [vis["refcoco"]["indices"][0]] * 2 + [10, 10]})
    # the eval-subset cache: the verifier ACCEPTS the canonical rows and REFUSES every tampered one.
    # This gate is what stands between a 30 MB cache and silently scoring a different frozen item set,
    # so it is exercised here rather than trusted.
    fz = {**vis["vqav2"], "n": VISION_N}
    canon = _select_rows(ld, fz["dataset"], fz["split"], fz["indices"], fz["columns"])
    good = _FakeSubset([{**r, "_index": i} for r, i in zip(canon, fz["indices"])])
    assert verify_vision_subset(good, fz, "vqav2") == fz["hashes"]
    tampered = (("swapped rows", [good[1], good[0]] + list(good[2:])),
                ("edited content", [{**good[0], "question": "tampered"}] + list(good[1:])),
                ("short list", list(good[:VISION_N - 1])),
                ("wrong n_split", None))
    for name, rows in tampered:
        f2 = {**fz, "n_split": fz["n_split"] + 1} if rows is None else fz
        try:
            verify_vision_subset(good if rows is None else _FakeSubset(rows), f2, "vqav2")
            raise SystemExit(f"verify_vision_subset accepted a subset with {name}")
        except AssertionError:
            pass
    assert _open_vision_subset("vqav2") is None                       # no cache on this box -> split path

    vt = vqav2_train(16000, 0, ld, sr)                                # the 16k REH dose fits the shard prefix
    assert len(vt["indices"]) == 16000 == len(set(vt["indices"])) and not (set(vt["hashes"]) & set(vis["vqav2"]["hashes"]))
    assert len(VQAV2_TRAIN_SHARDS) >= 6

    # the identity gate must be REACHABLE, not decorative: a planted collision has to raise
    dup = [{"subject": "s", "question": "q", "hash": "h"}, {"subject": "s", "question": "q2", "hash": "h"}]
    try:
        _assert_unique(dup, "planted")
        raise SystemExit("_assert_unique did not fire on a planted collision")
    except AssertionError as e:
        assert "planted: item_hash collision h" in str(e) and "'q2'" in str(e), str(e)
    assert item_hash("a", "bc") != item_hash("ab", "c") and len(item_hash("a", "b")) == 16
    assert question_key("q") != item_hash("", "q")           # leak key and alignment key are distinct

    m1, m2 = manifest(ld, sr), manifest(ld, sr)
    assert m1 == m2 and m1["mmlu_discovery"]["n"] == 285 and m1["two_hop"]["n"] == 100
    print(f"held-out subjects: {held}")
    print(f"manifest sets: {len(m1)}; rfp content_sha1 {rfp['meta']['content_sha1'][:12]}")
    print("PASS matrix_items dry-run (synthetic loader; no download)")


if __name__ == "__main__":
    _dryrun()
