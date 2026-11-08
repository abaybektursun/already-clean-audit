"""ERLANGEN Arm M — the 5 frozen task bindings and their train/eval item loaders.

Gate-0 resolved which splits actually exist (recorded in `erlangen_freeze.json`):

    cococap  lmms-lab/COCO-Caption  ['val', 'test']                  → only `val` is labelled
    refcoco  lmms-lab/RefCOCO       ['val','test','testA','testB']   → train `val`, eval `testA`
    textvqa  lmms-lab/textvqa       ['train','validation','test']    → the only genuine train split
    vqav2    lmms-lab/VQAv2         ['validation','testdev','test']  → NO train split on this mirror
    okvqa    lmms-lab/OK-VQA        ['val2014']                      → a single labelled split

So three of five must take train and eval from DISJOINT SLICES of one split — the convention
`battery_tasks.COCO_TASK` already carries (`same_split_disjoint=True`). The slice is deterministic
(streaming order after content-hash dedup: eval takes the first n_eval rows, train the next
n_train), so every cell of the cost matrix sees the same items and there is no train/eval leakage.

Everything is expressed as (image, prompt, target) triples — the SAME representation the readout
battery scores — so adaptation cost is measured with the program's one instrument rather than five
task-specific metrics.

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

M_TASKS = ("cococap", "refcoco", "textvqa", "vqav2", "okvqa")

BINDINGS = {
    "cococap": {"dataset": "lmms-lab/COCO-Caption", "train_split": "val", "eval_split": "val",
                "disjoint": True},
    "refcoco": {"dataset": "lmms-lab/RefCOCO", "train_split": "val", "eval_split": "testA",
                "disjoint": False},
    "textvqa": {"dataset": "lmms-lab/textvqa", "train_split": "train", "eval_split": "validation",
                "disjoint": False},
    "vqav2": {"dataset": "lmms-lab/VQAv2", "train_split": "validation", "eval_split": "validation",
              "disjoint": True},
    "okvqa": {"dataset": "lmms-lab/OK-VQA", "train_split": "val2014", "eval_split": "val2014",
              "disjoint": True},
}

FT_REPOS = {
    "cococap": "google/paligemma-3b-ft-cococap-224",
    "refcoco": "google/paligemma-3b-ft-refcoco-seg-224",
    "textvqa": "google/paligemma-3b-ft-textvqa-224",
    "vqav2": "google/paligemma-3b-ft-vqav2-224",
    "okvqa": "google/paligemma-3b-ft-okvqa-224",
}


def _prompt_target(task, row):
    """(prompt, target) for one row, reusing the house prompt/target functions VERBATIM.

    No leading space is added to the target. The processor's `suffix=` path builds
    `…prefix \n suffix`, so prepending a space tokenises " yes" as a different token sequence from
    the "yes" a PaliGemma fine-tune was actually trained to emit. That mis-tokenisation punishes a
    SHARPLY-fitted model far more than a loosely-fitted one, which is how the first Arm-M run got a
    criterion anchor WORSE than its own starting points (ft-vqav2 read 5.20 on VQAv2 items while
    ft-cococap read 4.11). `battery_tasks.collate` passes target_fn's output unmodified; so does
    this."""
    from infra.pg import battery_tasks as BT
    from infra.pg import transfer_a1 as TA
    if task == "cococap":
        return BT.COCO_TASK.prompt_fn(row), BT.COCO_TASK.target_fn(row)
    if task == "refcoco":
        return BT.REFCOCO_TASK.prompt_fn(row), BT._refcoco_target(row)
    if task == "textvqa":
        return BT.TEXTVQA_TASK.prompt_fn(row), BT._textvqa_majority(row)
    if task in ("vqav2", "okvqa"):
        from collections import Counter
        ans = [x["answer"] if isinstance(x, dict) else str(x) for x in row["answers"]]
        return f"{TA.VQA_PREFIX}{row['question']}", Counter(ans).most_common(1)[0][0]
    raise AssertionError(f"unknown M task {task!r}")


def _image(task, row):
    im = row["image"]
    return im.convert("RGB")


def _hash(task, row):
    import hashlib
    p, _t = _prompt_target(task, row)
    return hashlib.md5(p.encode() + _image(task, row).resize((32, 32)).tobytes()).hexdigest()


def items(task, kind, n, n_eval_skip=0, resize=224):
    """`kind` ∈ {eval, train}. Streaming + content-hash dedup, deterministic order.

    For a `disjoint` task the eval slice is the FIRST n_eval rows and train starts after them
    (`n_eval_skip`), so the two never overlap; for a task with a real train split they are
    different splits and `n_eval_skip` is ignored.

    Images are resized to the processor's 224 so a 2k-item train pool stays in host RAM."""
    from datasets import load_dataset
    b = BINDINGS[task]
    split = b["train_split"] if kind == "train" else b["eval_split"]
    skip = n_eval_skip if (kind == "train" and b["disjoint"]) else 0
    ds = load_dataset(b["dataset"], split=split, streaming=True)
    rows, seen, taken = [], set(), 0
    for r in ds:
        h = _hash(task, r)
        if h in seen:
            continue
        seen.add(h)
        taken += 1
        if taken <= skip:
            continue
        p, t = _prompt_target(task, r)
        rows.append({"image": _image(task, r).resize((resize, resize)), "prompt": p, "target": t})
        if len(rows) >= n:
            break
    assert len(rows) == n, f"{task}/{kind}: only {len(rows)} unique rows (<{n})"
    return rows
