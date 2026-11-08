"""MENDELEEV N14 — the vision own-skill readouts (O) for row 3: RefCOCO / VQAv2 / TextVQA through the
VLM's OWN processor and prompt conventions (spec §1.2 O, §3.1 rows 3b/3c-ER, §10 N14; contract §1 N14).

Items: the fixed 200-item lists of `matrix_items.vision_index_lists()` (RefCOCO testA / VQAv2 val /
TextVQA val; `np.random.default_rng(1337).permutation(len(split))[:200]`) — imported, never redefined.
NOTE (bind 2026-09-03): the `lmms-lab/*` repos in `matrix_items.VISION` were RENAMED on the hub to
`lmms-lab-encoder/{RefCOCO,VQAv2,textvqa}`; `datasets.load_dataset` follows the 307 redirect, the
datasets-server API does not. Recorded in the row-3 bind doc; not changed here (READOUTS owns the file).

Readouts (each returns the contract §4 record shape via `format_readouts._record`; `pred` is the
per-item correctness for generation readouts and the 4-way argmax for candidate readouts):
  refcoco      generate → parse a POINT (Molmo2's native grounding output, `<points … coords="1 <idx> <x> <y> …"/>`
               (frame 1, then idx x y triples, coordinates ×1000 — regexes verbatim from the Molmo2-4B model card) or a BOX (PaliGemma
               `<locDDDD>` via battery_tasks, or `[x0, y0, x1, y1]` in 0–1000 / fraction / pixel units —
               the generic path). Per item: point → point-in-GT-box; box → IoU ≥ 0.5 (battery_tasks._iou).
               `acc` is the mean; `floor` = the mean GT-box area fraction (a random point's hit rate)
               when the outputs are points, 0.0 when boxes. Molmo2 does NOT emit boxes, so on Molmo2
               this column is point-in-box accuracy, NOT acc@0.5 IoU — labelled in `extra["metric"]`;
               `extra["parse_frac"]` < 0.5 stamps `extra["void"] = "VOID-parse"` (never a silent drop).
  vqav2, textvqa   generate (16 tokens) → `acc` = the OFFICIAL VQA soft accuracy min(1, #matching human
               answers / 3) after the official normalisation (`transfer_a1.vqa_normalize`, verbatim
               vqaEval) — the metric behind the published numbers and this repo's VQAv2 metric.
               Beside it: `extra["em_acc"]` (normalised exact match against any of the 10 human answers —
               the spec's "exact match") and `extra["anls_mean"]` (battery_tasks._anls_one). Floor 0.0.
  vqav2_cand, textvqa_cand   the format-robust O used by every verdict (spec §1.2 O, D4): 4-way
               candidate likelihood over {gold, three OTHER items' golds} through
               `Forward.continuation_logprobs` (per-token-mean log-likelihood, argmax; floor 0.25).
               Distractors: a fixed `random.Random(1337)` draw over the other items' golds, deduplicated
               after `vqa_normalize` (VQAv2 golds are ~40 % "yes"/"no" — without the dedupe a distractor
               could equal the gold). `image_free=True` reruns it on the shell's BLANK image: the
               answer-prior floor of this readout (a candidate set drawn from other items' answers can be
               partly solved from answer-type priors alone — report it beside 0.25).

Prompt conventions (Molmo2-4B, repo sha 042abfa7; `chat_template.jinja` + allenai/molmo2
`olmo/preprocessing/data_formatter.py` + `olmo/data/academic_image_datasets.py`):
  * the template puts `<|image|>` FIRST, then `<|im_start|>user\\n{text}<|im_end|>\\n<|im_start|>assistant\\n`;
  * a text segment with a `style` outside DEMO_STYLES renders as `"{style}: {text}"` — this is the
    training-time `demo_or_style_v2` system prompt; the academic short-answer styles are `vqa2`
    (VQAv2) and `text_vqa` (TextVQA), so the eval prompts are "vqa2: {question}" / "text_vqa: {question}";
  * `pointing` IS a DEMO style (no prefix): RefCOCO uses "Point to {expression}".
  `molmo2_template_parity(processor, question)` asserts at run time that this static rendering equals
  `processor.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
Generic HF-VLM path (DeepSeek-Vision, row 4 — UNTESTED, no weights were run): `generic_prompts(processor)`
renders a plain instruction through the model's chat template; the box parser above scores it. The
matching `Forward` shell for that family does not exist yet (row 4 is GATED on S9-pre, spec B2).

Candidate-scoring caveat (raise, not fix): `Forward.continuation_logprobs` scores `prompt + " " +
option` (the house "Answer:" convention). Molmo2's prompt ends in "assistant\\n", after which the model
would emit "yes", not " yes" (Qwen BPE: different token). All four options share the mismatch, so the
argmax is still a ranking of the options; the generation readouts are unaffected.

Acceptance spec (N14, spec §10; the tolerance was widened to 6 points on 2026-09-03):
  published Molmo2-4B — VQA v2.0 val 86.6, TextVQA val 85.0: arXiv 2601.10611v4 (Molmo2 report), Table 6
  "Image results", Molmo2-4B row, read 2026-09-03 from arxiv.org/html/2601.10611 (the model card gives
  only the 15-video-benchmark average 62.8; spec §14 cites 2605.02881, which is the MolmoAct2 paper).
  RefCOCO: NO published Molmo2 number exists (paper + card grep 2026-09-03) → acceptance N/A, stated.
  Rule: |100·acc(200 items) − published| ≤ ACCEPT_POINTS = 6.0 for vqav2 and textvqa (`acceptance()`).
  Why 6 and not 3: at p ≈ 0.86 the 200-item binomial SE is ≈ 2.5 points, so a ±3-point band fails a
  PERFECT harness about a fifth of the time — a test that fires that often on correct code is noise, not
  a gate. ±6 ≈ 2.4 SE. The record still carries the item-bootstrap half-width so the reader can see the
  noise; the published numbers come from the authors' harness (their prompt/style, full val split).
  This is a GATE, not a note (2026-09-03 ruling). `acceptance()` returns a `verdict` string
  (PASS / FAIL / N/A) beside `pass_`, and a FAIL stamps `status = "VOID-acceptance"` on EVERY O record
  of that member in `matrix_behaviour._o_column_molmo2`, so the offline verdict script refuses to score
  the row's O column instead of scoring it silently. It never kills the job: the failure is banked loud
  and the run continues, because the other columns of the cell are unaffected by a vision-anchor miss.

Dry-run (`python -m infra.pg.vlm_skill_eval`): weight-free, torch-free — a duck-typed stub forward and
synthetic rows; exercises prompt construction, point/box parsing, IoU, VQA normalisation and soft
accuracy, ANLS, candidate-set construction, candidate scoring, the record shape, acceptance, and the
`_perm_indices` == `vision_index_lists` identity. Fail-loud: asserts, no try/except.
"""
from __future__ import annotations

import random
import re

import numpy as np

from infra.pg.battery_tasks import (_anls_one, _decode_loc_box, _iou, _refcoco_expr, _refcoco_gt_box,
                                    _textvqa_answers, _textvqa_majority)
from infra.pg.format_readouts import _record, _samples
from infra.pg.matrix_forward import MOLMO_IMAGE_PLACEHOLDER, Molmo2Forward
from infra.pg.matrix_items import VISION, VISION_N, VISION_SEED, _perm_indices, _row_hash, vision_rows
from infra.pg.transfer_a1 import vqa_normalize, vqa_soft_accuracy

TASKS = ("refcoco", "vqav2", "textvqa")
CAND_TASKS = ("vqav2", "textvqa")
MAX_NEW_TOKENS = {"refcoco": 48, "vqav2": 16, "textvqa": 16}
FLOOR_CAND = 0.25
ACCEPT_POINTS = 6.0   # spec §10 N14 (widened 2026-09-03): ±3 is inside the 200-item sampling noise

# model id -> task -> (published score in points or None, source line)
PUBLISHED = {
    "allenai/Molmo2-4B": {
        "vqav2": (86.6, "arXiv 2601.10611v4 Table 6 'VQA v2.0 val', Molmo2-4B row (read 2026-09-03)"),
        "textvqa": (85.0, "arXiv 2601.10611v4 Table 6 'TextVQA val', Molmo2-4B row (read 2026-09-03)"),
        "refcoco": (None, "no RefCOCO number in arXiv 2601.10611 or the allenai/Molmo2-4B card (grep 2026-09-03)"),
    },
}

# ----------------------------------------------------------------------------------------- prompts
MOLMO2_STYLE = {"vqav2": "vqa2", "textvqa": "text_vqa"}     # academic_image_datasets.Vqa2 / TextVqa styles


def molmo2_prompt(text: str, style: str | None = None) -> str:
    """The exact string `chat_template.jinja` (allenai/Molmo2-4B @042abfa7) renders for one user turn with
    one image, `add_generation_prompt=True`; `style` outside DEMO_STYLES → "{style}: {text}"."""
    seg = f"{style}: {text}" if style else text
    return f"{MOLMO_IMAGE_PLACEHOLDER}<|im_start|>user\n{seg}<|im_end|>\n<|im_start|>assistant\n"


def molmo2_prompts() -> dict:
    """task -> (row -> prompt string) for the Molmo2 family."""
    return {"refcoco": lambda row: molmo2_prompt(f"Point to {_refcoco_expr(row)}"),
            "vqav2": lambda row: molmo2_prompt(row["question"], MOLMO2_STYLE["vqav2"]),
            "textvqa": lambda row: molmo2_prompt(row["question"], MOLMO2_STYLE["textvqa"])}


def molmo2_template_parity(processor, question="What is this?", style="vqa2") -> str:
    """Run-time gate: the processor's own chat template must render exactly `molmo2_prompt`."""
    messages = [{"role": "user", "content": [{"type": "text", "text": question, "style": style},
                                             {"type": "image"}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expect = molmo2_prompt(question, style)
    assert rendered == expect, f"chat template drift:\n{rendered!r}\n!=\n{expect!r}"
    return rendered


GENERIC_TEXT = {   # UNTESTED generic HF-VLM instructions (DeepSeek-Vision, row 4)
    "refcoco": "Locate {expr} in the image and output its bounding box as [x0, y0, x1, y1] "
               "with coordinates scaled to 0-1000.",
    "vqav2": "{question}\nAnswer the question using a single word or phrase.",
    "textvqa": "{question}\nAnswer the question using a single word or phrase.",
}


def generic_prompts(processor) -> dict:
    """task -> (row -> prompt) through `processor.apply_chat_template` (one image + one text). UNTESTED."""
    def render(text):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"refcoco": lambda row: render(GENERIC_TEXT["refcoco"].format(expr=_refcoco_expr(row))),
            "vqav2": lambda row: render(GENERIC_TEXT["vqav2"].format(question=row["question"])),
            "textvqa": lambda row: render(GENERIC_TEXT["textvqa"].format(question=row["question"]))}


# ----------------------------------------------------------------------------------------- shell
class Molmo2SkillForward(Molmo2Forward):
    """Molmo2Forward whose encode uses the bound real image (`.image`, a PIL image) — the blank image
    when `.image is None`. Prompts arrive with their own single `<|image|>` placeholder (the chat
    template puts it first), so nothing is prepended.

    SIGNATURE (fixed 2026-09-03, SEAMS audit): the override MUST take `(texts, images=None)` like
    `matrix_forward.Forward._encode`, because the base class calls it with TWO positional arguments —
    `self._encode(fulls, imgs if any(...) else None)` in `continuation_logprobs` and
    `self._encode(chunk, imgs)` in `generate`. With the old one-argument override every real call
    through this shell would have died with "takes 2 positional arguments but 3 were given"; only the
    module's own duck-typed `_StubForward` (which never constructs this class) hid it. This shell's
    image is the BOUND one, so a caller that passes real images is a contract error, not a fallback."""
    shell = "molmo2"
    image = None

    def _encode(self, texts, images=None):
        texts = list(texts)
        assert images is None or all(im is None for im in images), \
            "Molmo2SkillForward takes its image from `.image`; passing `images=` would be silently ignored"
        bad = [t[:40] for t in texts if t.count(self.placeholder) != 1]
        assert not bad, f"each prompt must carry exactly one {self.placeholder}: {bad}"
        img = self.blank if self.image is None else self.image.convert("RGB")
        return self.proc(text=texts, images=[img] * len(texts), return_tensors="pt", padding=True).to(self.device)


# ----------------------------------------------------------------------------------------- parsing
# Molmo2 point output (allenai/Molmo2-4B card, olmo/preprocessing/point_formatter.py UnifiedPointFormatter,
# coordinate_scale "1000"): <points … coords="<frame> <idx> <x> <y> …"/>, x/y in 0–1000 of width/height.
_COORD_RE = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
_FRAME_RE = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
_POINTS_RE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
_NUM4_RE = re.compile(r"(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)")


def parse_points(text: str, W: int, H: int) -> list:
    """All (x_px, y_px) points in a Molmo2 generation, in output order; [] if none."""
    pts = []
    for coord in _COORD_RE.finditer(text):
        for grp in _FRAME_RE.finditer(coord.group(1)):
            for _idx, x, y in _POINTS_RE.findall(grp.group(2)):
                x, y = float(x) / 1000 * W, float(y) / 1000 * H
                if 0 <= x <= W and 0 <= y <= H:
                    pts.append((x, y))
    return pts


def parse_box(text: str, W: int, H: int):
    """Normalised (y0, x0, y1, x1) from a PaliGemma <loc> string or the first 'x0, y0, x1, y1' quadruple
    (all ≤ 1 → fractions; all ≤ 1000 → the 0–1000 grid; else pixels). None if absent."""
    loc = _decode_loc_box(text)
    if loc is not None:
        return loc
    m = _NUM4_RE.search(text)
    if m is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    mx = max(x0, y0, x1, y1)
    if mx <= 1.0:
        sx = sy = 1.0
    elif mx <= 1000.0:
        sx = sy = 1000.0
    else:
        sx, sy = float(W), float(H)
    return (y0 / sy, x0 / sx, y1 / sy, x1 / sx)


def point_in_box(pt, box, W: int, H: int) -> bool:
    x, y = pt
    y0, x0, y1, x1 = box
    return x0 * W <= x <= x1 * W and y0 * H <= y <= y1 * H


def box_area_frac(box) -> float:
    y0, x0, y1, x1 = box
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)


# ----------------------------------------------------------------------------------------- items
def load_rows(task: str, load_dataset=None) -> list:
    """The fixed 200 rows of `task` (materialised dicts, PIL images), in `vision_index_lists` order.
    Delegated to `matrix_items.vision_rows`, which takes the cached 200-row eval subset when the box
    has one (verified row by row against the frozen index) and materialises the split otherwise."""
    assert task in TASKS, task
    return vision_rows(task, load_dataset)


def item_hashes(task: str, rows: list) -> list:
    """The frozen per-row content hashes: `matrix_items._row_hash` over the task's light columns — the
    same function over the same columns `vision_index_lists` records, so a record's `per_item_hash` is
    identical to the manifest's by construction (which is what this function always promised).
    FIXED 2026-09-03 (SUBSETS agent, on READOUTS' surface): this called `matrix_items._vqa_hash`, a name
    matrix_items DROPPED in d8a3728c when the row hash became content-based. The stale import made this
    whole module unimportable — `python -m infra.pg.vlm_skill_eval` died at line 78 — so the Molmo O
    column could not have run. READOUTS: if a different key was intended, it must still equal the
    manifest hash for the same row, or the O records cannot be aligned to the frozen item set."""
    cols = VISION[task]["columns"]
    return [_row_hash({c: r[c] for c in cols}) for r in rows]


def human_answers(task: str, row: dict) -> list:
    """The 10 human answers (VQAv2: `answers[*].answer`; TextVQA: `answers`)."""
    if task == "vqav2":
        return [a["answer"] for a in row["answers"]]
    assert task == "textvqa", task
    return _textvqa_answers(row)


def gold_answer(task: str, row: dict) -> str:
    if task == "vqav2":
        return str(row["multiple_choice_answer"])
    assert task == "textvqa", task
    return _textvqa_majority(row)


def candidate_sets(task: str, rows: list, seed: int = VISION_SEED) -> list:
    """Per item: 4 options = gold + 3 other items' golds (fixed draw, distinct after vqa_normalize),
    shuffled; `{"options", "gold_idx", "gold"}`."""
    golds = [gold_answer(task, r) for r in rows]
    norm = [vqa_normalize(g) for g in golds]
    rng = random.Random(seed)
    out = []
    for i, g in enumerate(golds):
        order = list(range(len(rows)))
        rng.shuffle(order)
        taken, opts = {norm[i]}, []
        for j in order:
            if j == i or norm[j] in taken or not norm[j]:
                continue
            taken.add(norm[j])
            opts.append(golds[j])
            if len(opts) == 3:
                break
        assert len(opts) == 3, f"item {i}: only {len(opts)} distinct distractors among {len(rows)} golds"
        options = [g] + opts
        rng.shuffle(options)
        out.append({"options": options, "gold_idx": options.index(g), "gold": g})
    return out


# ----------------------------------------------------------------------------------------- readouts
def _generate_one(fwd, prompt: str, image, max_new_tokens: int) -> str:
    fwd.image = image
    out = fwd.generate([prompt], max_new_tokens)[0]
    fwd.image = None
    return str(out)


def _halfwidth(per_item, n_boot=2000, seed=VISION_SEED) -> float:
    """95 % item-bootstrap half-width of the mean, in points."""
    a = np.asarray(per_item, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(n_boot)])
    return float(100.0 * (np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2.0)


def refcoco(fwd, rows: list, prompt_fn, max_new_tokens: int = MAX_NEW_TOKENS["refcoco"]) -> dict:
    per_item, kinds, gens, parsed, gts = [], [], [], [], []
    area = []
    for row in rows:
        W, H = row["image"].size
        gt = _refcoco_gt_box(row)
        gts.append([float(v) for v in gt])
        area.append(box_area_frac(gt))
        text = _generate_one(fwd, prompt_fn(row), row["image"], max_new_tokens)
        gens.append(text)
        pts = parse_points(text, W, H)
        if pts:
            hit, kind, got = point_in_box(pts[0], gt, W, H), "point", [float(pts[0][0]), float(pts[0][1])]
        else:
            box = parse_box(text, W, H)
            if box is not None:
                hit, kind, got = _iou(box, gt) >= 0.5, "box", [float(v) for v in box]
            else:
                hit, kind, got = False, "none", None
        per_item.append(int(hit))
        kinds.append(kind)
        parsed.append(got)
    n = len(rows)
    n_point, n_box = kinds.count("point"), kinds.count("box")
    metric = "point_in_gt_box" if n_point >= n_box else "acc@0.5 IoU"
    floor = float(np.mean(area)) if metric == "point_in_gt_box" else 0.0
    extra = {"metric": metric, "kinds": kinds, "n_point": n_point, "n_box": n_box, "parsed": parsed,
             "generations": gens, "parse_frac": (n_point + n_box) / n,
             "floor_point_in_box_chance": float(np.mean(area)), "floor_iou": 0.0,
             "prompt_example": prompt_fn(rows[0]), "max_new_tokens": max_new_tokens,
             "ci95_halfwidth_points": _halfwidth(per_item),
             # §12.11: the real prompt, the real generation, what it parsed to and the GT box. Without
             # it this column is a number with nothing behind it, and O feeds four recovery hypotheses.
             "samples": _samples([prompt_fn(r) for r in rows], per_item, [str(x) for x in parsed],
                                 [str(g) for g in gts], decoded=gens,
                                 extra_per_item={"parse_kind": kinds, "gt_box_xywh": gts})}
    if extra["parse_frac"] < 0.5:
        extra["void"] = "VOID-parse"
    return _record("refcoco", floor, per_item, item_hashes("refcoco", rows), None, per_item, extra)


def vqa(task: str, fwd, rows: list, prompt_fn, max_new_tokens: int | None = None) -> dict:
    """`acc` = official VQA soft accuracy; exact match and ANLS beside."""
    assert task in CAND_TASKS, task
    mnt = MAX_NEW_TOKENS[task] if max_new_tokens is None else max_new_tokens
    soft, em, anls, gens = [], [], [], []
    for row in rows:
        answers = human_answers(task, row)
        text = _generate_one(fwd, prompt_fn(row), row["image"], mnt)
        gens.append(text)
        soft.append(vqa_soft_accuracy(text, answers))
        em.append(int(vqa_normalize(text) in {vqa_normalize(a) for a in answers}))
        anls.append(max(_anls_one(text, a) for a in answers))
    extra = {"metric": "VQA soft accuracy min(1, matches/3), official normalisation (transfer_a1)",
             "em_acc": float(np.mean(em)), "em_per_item": em, "anls_mean": float(np.mean(anls)),
             "anls_per_item": anls, "generations": gens, "prompt_example": prompt_fn(rows[0]),
             "max_new_tokens": mnt, "ci95_halfwidth_points": _halfwidth(soft),
             # §12.11. `scored_correct` here is the SOFT accuracy (this record's `acc`), and the ten
             # human answers it was scored against ride along — the number cannot be read without them.
             "samples": _samples([prompt_fn(r) for r in rows], soft, gens,
                                 [gold_answer(task, r) for r in rows], decoded=gens,
                                 extra_per_item={"human_answers": [human_answers(task, r) for r in rows],
                                                 "exact_match": em, "anls": anls})}
    return _record(task, 0.0, soft, item_hashes(task, rows), None, em, extra)


def vqa_cand(task: str, fwd, rows: list, prompt_fn, cands: list | None = None, image_free: bool = False) -> dict:
    """4-way candidate likelihood (per-token mean) through `fwd.continuation_logprobs`; floor 0.25.
    `image_free=True` binds the blank image instead of the item's (the answer-prior floor)."""
    assert task in CAND_TASKS, task
    cands = candidate_sets(task, rows) if cands is None else cands
    assert len(cands) == len(rows)
    means, sums, ntoks, pred, per_item = [], [], [], [], []
    for row, c in zip(rows, cands):
        prompt = prompt_fn(row)
        fwd.image = None if image_free else row["image"]
        res = fwd.continuation_logprobs([prompt] * 4, list(c["options"]))
        fwd.image = None
        assert len(res) == 4, res
        s = [float(r[0]) for r in res]
        k = [int(r[1]) for r in res]
        assert all(m >= 1 for m in k) and all(np.isfinite(s)), (s, k)
        m = [a / b for a, b in zip(s, k)]
        p = int(np.argmax(m))
        means.append(m)
        sums.append(s)
        ntoks.append(k)
        pred.append(p)
        per_item.append(int(p == c["gold_idx"]))
    col = f"{task}_cand" + ("_blank" if image_free else "")
    golds = [c["gold"] for c in cands]
    extra = {"metric": "4-way candidate likelihood, per-token mean, argmax", "image_free": image_free,
             "options": [c["options"] for c in cands], "gold_idx": [c["gold_idx"] for c in cands],
             "cloze_logprobs": sums, "cloze_ntoks": ntoks,
             "n_yesno_golds": sum(vqa_normalize(g) in ("yes", "no") for g in golds),
             "prompt_example": prompt_fn(rows[0]), "ci95_halfwidth_points": _halfwidth(per_item),
             # §12.11: this is the O column every verdict consumes, so it is the one that most needed
             # the four candidates and their per-token-mean scores visible beside the accuracy.
             "samples": _samples([prompt_fn(r) for r in rows], per_item, pred, [c["gold_idx"] for c in cands],
                                 means, options=[c["options"] for c in cands],
                                 extra_per_item={"gold_text": golds, "candidate_sum_logprob": sums,
                                                 "candidate_n_tokens": ntoks})}
    return _record(col, FLOOR_CAND, per_item, item_hashes(task, rows), means, pred, extra)


def evaluate(col: str, fwd, rows: list, prompts: dict, image_free: bool = False) -> dict:
    """Dispatcher: col ∈ {refcoco, vqav2, textvqa, vqav2_cand, textvqa_cand}."""
    if col == "refcoco":
        return refcoco(fwd, rows, prompts["refcoco"])
    if col in CAND_TASKS:
        return vqa(col, fwd, rows, prompts[col])
    task = col[:-len("_cand")]
    assert col.endswith("_cand") and task in CAND_TASKS, col
    return vqa_cand(task, fwd, rows, prompts[task], image_free=image_free)


def acceptance(model_id: str, task: str, record: dict) -> dict:
    """N14 acceptance GATE: |100·acc − published| ≤ ACCEPT_POINTS.

    Returns `pass_` (True / False / None) and a human- and script-readable `verdict`:
      "PASS …"  the harness reproduces the published number inside the band;
      "FAIL …"  it does not — the caller must mark this member's O column unusable (VOID-acceptance),
                 not score it. A criterion that cannot fail is not a criterion (2026-09-03 ruling).
      "N/A …"   no published anchor exists for this task (RefCOCO); stated, never scored as a pass.
    """
    published, source = PUBLISHED[model_id][task]
    est = 100.0 * float(record["acc"])
    out = {"model": model_id, "task": task, "published": published, "source": source, "estimate": est,
           "n": record["n"], "ci95_halfwidth_points": record["extra"].get("ci95_halfwidth_points"),
           "band_points": ACCEPT_POINTS}
    if published is None:
        out.update(delta=None, pass_=None, note="no published anchor for this task; acceptance N/A",
                   verdict=f"N/A — no published anchor for {model_id} {task}")
        return out
    delta = est - published
    ok = bool(abs(delta) <= ACCEPT_POINTS)
    out.update(delta=delta, pass_=ok,
               verdict=(f"{'PASS' if ok else 'FAIL'} — {task} {est:.2f} vs published {published:.2f} "
                        f"(delta {delta:+.2f} points, band ±{ACCEPT_POINTS:.1f}, n={record['n']})"))
    return out


# ----------------------------------------------------------------------------------------- dry-run
def _stub_rows(task: str, n: int = 8) -> list:
    """Synthetic rows for the weight-free dry-runs. `n` may exceed the 8 distinct golds (the recovery
    job's O seam test needs ≥ 20 — `matrix_core.preflight_match` asserts n ≥ 20): the gold list CYCLES
    while `question_id` keeps every row hash distinct, which is what `candidate_sets` needs (3 distinct
    normalised distractors) and what `item_hashes` promises."""
    from PIL import Image
    rng = random.Random(f"vlm-{task}")
    rows = []
    for i in range(n):
        W, H = 40 + 10 * (i % 3), 30 + 10 * (i % 2)
        img = Image.new("RGB", (W, H), (i * 20 % 255, 80, 120))
        if task == "refcoco":
            x, y, w, h = 5 + i, 4, 12, 10
            rows.append({"image": img, "bbox": [x, y, w, h], "answer": [f"object {i}"], "_index": i})
        elif task == "vqav2":
            gold = ["yes", "no", "2", "white", "dog", "red", "3", "cat"][i % 8]
            others = [gold] * 7 + ["maybe", "blue", "1"]
            rng.shuffle(others)
            rows.append({"image": img, "question": f"vqa question {i}?", "question_id": 1000 + i,
                         "multiple_choice_answer": gold,
                         "answers": [{"answer": a, "answer_confidence": "yes", "answer_id": k + 1}
                                     for k, a in enumerate(others)], "_index": i})
        else:
            gold = ["stop", "2019", "coca cola", "exit", "5", "open", "10", "sale"][i % 8]
            others = [gold] * 6 + ["unanswerable", "none", gold.upper(), "a"]
            rng.shuffle(others)
            rows.append({"image": img, "question": f"text question {i}?", "question_id": 2000 + i,
                         "answers": others, "_index": i})
    return rows


class _StubForward:
    """Duck-typed forward: `.image`, `.blank`, `.batch_size`, `generate`, `continuation_logprobs`.
    `gens`: prompt -> generation; `golds`: prompt -> gold answer (candidate scoring prefers the gold
    except on `miss_prompt`, where it prefers the non-gold options); the blank image prefers "yes"."""
    shell = "molmo2"
    batch_size = 4
    image = None
    blank = "BLANK"

    def __init__(self, gens: dict, golds: dict | None = None, miss_prompt: str = ""):
        self.gens = gens
        self.golds = golds or {}
        self.miss = miss_prompt
        self.seen_images = []

    def generate(self, prompts, max_new_tokens):
        self.seen_images.append(self.image)
        return [self.gens[p] for p in prompts]

    def continuation_logprobs(self, prompts, continuations):
        self.seen_images.append(self.image)
        gold = self.golds[prompts[0]]
        out = []
        for p, c in zip(prompts, continuations):
            good = (vqa_normalize(c) == vqa_normalize(gold)) != (p == self.miss)
            if self.image is None:                      # blank image: answer prior only
                good = c.strip().lower() == "yes"
            ntok = max(1, len(c.strip().split()))
            out.append(((-1.0 if good else -6.0) * ntok, ntok))
        return out


def _dryrun():
    import inspect
    import math
    from infra.pg.matrix_forward import Forward
    from infra.pg.matrix_items import fake_load_dataset, fake_split_rows, vision_index_lists
    # SHELL CONTRACT (2026-09-03): `_StubForward` is duck-typed, so nothing else in this dry-run ever
    # constructs `Molmo2SkillForward` — the class that every REAL Molmo O readout runs through. Its
    # override must accept exactly what `Forward` calls it with, or the first weighted run dies on an
    # arity error. Checked here because it is the one property of that class a weight-free test can see.
    assert (list(inspect.signature(Molmo2SkillForward._encode).parameters)
            == list(inspect.signature(Forward._encode).parameters)), \
        (f"Molmo2SkillForward._encode{inspect.signature(Molmo2SkillForward._encode)} does not match "
         f"Forward._encode{inspect.signature(Forward._encode)} — the base class calls it positionally")
    P = molmo2_prompts()
    # prompt construction == the chat template's rendering
    assert P["vqav2"]({"question": "How many dogs?"}) == \
        "<|image|><|im_start|>user\nvqa2: How many dogs?<|im_end|>\n<|im_start|>assistant\n"
    assert P["textvqa"]({"question": "What brand?"}).startswith("<|image|><|im_start|>user\ntext_vqa: What brand?")
    assert P["refcoco"]({"answer": ["left guy"]}) == molmo2_prompt("Point to left guy")
    assert molmo2_prompt("x").count(MOLMO_IMAGE_PLACEHOLDER) == 1

    # point parsing (card format) + box parsing + IoU + point-in-box
    W, H = 200, 100
    assert parse_points('<points dogs alt="dogs" coords="1 1 500 500"/>', W, H) == [(100.0, 50.0)]
    assert parse_points('<points alt="dogs" coords="1 1 250 100 2 750 900"/>', W, H) == [(50.0, 10.0), (150.0, 90.0)]
    assert parse_points("There are none.", W, H) == []
    assert parse_points('<points coords="1 1 1500 500"/>', W, H) == []        # out of range dropped
    assert parse_box("[100, 200, 300, 400]", W, H) == (0.2, 0.1, 0.4, 0.3)   # 0-1000 grid → (y0,x0,y1,x1)
    assert parse_box("box: 0.1, 0.2, 0.3, 0.4", W, H) == (0.2, 0.1, 0.4, 0.3)
    assert parse_box("[20, 20, 400, 80]", W, H) == (0.02, 0.02, 0.08, 0.4)
    assert parse_box("[400, 50, 1200, 90]", W, H) == (0.5, 2.0, 0.9, 6.0)      # > 1000 → pixels / (W, H)
    assert parse_box("<loc0100><loc0200><loc0300><loc0400>", W, H) is not None
    assert parse_box("no box here", W, H) is None
    assert _iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0 and _iou((0, 0, .5, .5), (.5, .5, 1, 1)) == 0.0
    assert abs(_iou((0, 0, 1, 1), (0, 0, 1, .5)) - 0.5) < 1e-9
    assert point_in_box((10, 10), (0.0, 0.0, 0.5, 0.5), W, H) and not point_in_box((150, 10), (0.0, 0.0, 0.5, 0.5), W, H)
    # VQA normalisation / soft accuracy / ANLS
    assert vqa_normalize("Two") == "2" and vqa_normalize("A dog.") == "dog" and vqa_normalize("don't") == "don't"
    assert vqa_soft_accuracy("yes", ["yes"] * 3 + ["no"] * 7) == 1.0
    assert abs(vqa_soft_accuracy("yes", ["yes"] + ["no"] * 9) - 1 / 3) < 1e-9
    assert _anls_one("coca cola", "coca cola") == 1.0 and _anls_one("pepsi", "coca cola") == 0.0
    # index identity with the READOUTS list
    fake = fake_load_dataset
    ds = fake(VISION["vqav2"]["dataset"], split="validation")
    idx = _perm_indices(VISION_SEED, len(ds), VISION_N)
    assert idx == vision_index_lists(fake, fake_split_rows)["vqav2"]["indices"]
    # load_rows delegates to matrix_items.vision_rows: same rows, same order, `_index` carried
    lr = load_rows("vqav2", fake)
    assert [r["_index"] for r in lr] == idx and len(lr) == VISION_N

    # candidate sets: 4 distinct normalised options, gold present
    rows_v = _stub_rows("vqav2")
    cs = candidate_sets("vqav2", rows_v)
    for c, r in zip(cs, rows_v):
        assert c["options"][c["gold_idx"]] == r["multiple_choice_answer"]
        assert len({vqa_normalize(o) for o in c["options"]}) == 4
    assert candidate_sets("vqav2", rows_v) == cs

    # readouts through the stub forward
    rows_r = _stub_rows("refcoco")
    ans = {}
    for r in rows_r:
        Wr, Hr = r["image"].size
        x, y, w, h = r["bbox"]
        inside = f'<points alt="o" coords="1 1 {int(1000 * (x + w / 2) / Wr):03d} {int(1000 * (y + h / 2) / Hr):03d}"/>'
        ans[P["refcoco"](r)] = inside
    miss = P["refcoco"](rows_r[3])
    ans[miss] = '<points coords="1 1 990 990"/>'         # outside every box
    ans[P["refcoco"](rows_r[5])] = "There are none."     # unparseable
    fwd = _StubForward(ans)
    rec = refcoco(fwd, rows_r, P["refcoco"])
    assert set(rec) == {"col", "acc", "n", "floor", "per_item", "per_item_hash", "per_item_scores", "pred", "extra"}
    assert rec["n"] == 8 and rec["per_item"][3] == 0 and rec["per_item"][5] == 0 and sum(rec["per_item"]) == 6
    assert rec["extra"]["metric"] == "point_in_gt_box" and rec["extra"]["kinds"][5] == "none"
    assert abs(rec["extra"]["parse_frac"] - 7 / 8) < 1e-9 and "void" not in rec["extra"]
    assert 0 < rec["floor"] < 1 and abs(rec["acc"] - float(np.mean(rec["per_item"]))) < 1e-12
    assert fwd.seen_images == [r["image"] for r in rows_r] and fwd.image is None
    assert len(set(rec["per_item_hash"])) == 8

    golds_v = {P["vqav2"](r): r["multiple_choice_answer"] for r in rows_v}
    ans_v = dict(golds_v)
    miss_v = P["vqav2"](rows_v[1])
    ans_v[miss_v] = "banana"
    ans_v[P["vqav2"](rows_v[2])] = "Two"                 # normalises onto the gold "2"
    fv = _StubForward(ans_v, golds_v, miss_v)
    rv = vqa("vqav2", fv, rows_v, P["vqav2"])
    assert rv["col"] == "vqav2" and rv["floor"] == 0.0 and rv["per_item"][1] == 0.0 and rv["per_item"][2] == 1.0
    assert rv["extra"]["em_per_item"][1] == 0 and rv["extra"]["em_per_item"][2] == 1 and rv["extra"]["em_acc"] == 7 / 8
    assert all(x in (0.0, 1.0) or 0 < x < 1 for x in rv["per_item"]) and math.isfinite(rv["extra"]["anls_mean"])
    # per-item hashes ARE the frozen manifest hashes: same function, same light columns, same values
    ih = item_hashes("vqav2", rows_v)
    assert ih == [_row_hash({"question": r["question"], "question_id": r["question_id"]}) for r in rows_v]
    assert len(set(ih)) == len(ih) and all(len(h) == 12 for h in ih)

    rc = vqa_cand("vqav2", fv, rows_v, P["vqav2"], cs)
    assert rc["col"] == "vqav2_cand" and rc["floor"] == FLOOR_CAND and rc["per_item"][1] == 0 and sum(rc["per_item"]) == 7
    assert len(rc["per_item_scores"]) == 8 and all(len(s) == 4 for s in rc["per_item_scores"])
    assert rc["pred"] == [int(np.argmax(s)) for s in rc["per_item_scores"]]
    assert rc["extra"]["n_yesno_golds"] == 2 and rc["extra"]["options"] == [c["options"] for c in cs]
    rb = vqa_cand("vqav2", fv, rows_v, P["vqav2"], cs, image_free=True)
    assert rb["col"] == "vqav2_cand_blank" and rb["extra"]["image_free"] and fv.seen_images[-1] is None
    assert rb["acc"] < rc["acc"]
    # §12.11 REAL-INFERENCE SAMPLES on every O readout. Until 2026-09-03 none of these three banked
    # `extra.samples`, so under the frozen rule every vision number in the program was unquotable —
    # including the one four recovery hypotheses consume. One right item and one wrong item, the actual
    # prompt, the actual generation / chosen candidate, the gold.
    for r, kind in ((rec, "refcoco"), (rv, "vqa"), (rc, "cand")):
        sm = r["extra"]["samples"]
        assert len(sm) == 3 and {d["scored_correct"] for d in sm} >= {1.0, 0.0}, (kind, sm)
        assert all(d["prompt_tail"] and d["prompt_chars"] and d["gold"] is not None for d in sm), kind
        assert all(d["predicted"] is not None for d in sm), kind
    assert all(d["model_output_text"] for d in rec["extra"]["samples"])          # generation readouts
    assert all(d["model_output_text"] for d in rv["extra"]["samples"])
    assert all(d["parse_kind"] in ("point", "box", "none") for d in rec["extra"]["samples"])
    assert all(d["human_answers"] and "anls" in d for d in rv["extra"]["samples"])
    assert all(len(d["candidates"]) == 4 and len(d["candidate_scores"]) == 4    # the 4-way it chose from
               and d["gold_text"] for d in rc["extra"]["samples"])

    rows_t = _stub_rows("textvqa")
    ans_t = {P["textvqa"](r): _textvqa_majority(r) for r in rows_t}
    ft = _StubForward(ans_t, ans_t)
    rt = evaluate("textvqa", ft, rows_t, P)
    assert rt["col"] == "textvqa" and rt["acc"] == 1.0 and rt["extra"]["em_acc"] == 1.0
    rtc = evaluate("textvqa_cand", ft, rows_t, P)
    assert rtc["col"] == "textvqa_cand" and rtc["acc"] == 1.0
    assert evaluate("refcoco", fwd, rows_r, P)["acc"] == rec["acc"]

    # acceptance — the GATE: a pass, a FAIL, and the no-anchor N/A must each be reachable and labelled.
    assert ACCEPT_POINTS == 6.0, ACCEPT_POINTS                       # spec §10 N14 tolerance, frozen
    a = acceptance("allenai/Molmo2-4B", "vqav2", {**rv, "acc": 0.85})
    assert a["pass_"] is True and abs(a["delta"] - (85.0 - 86.6)) < 1e-9 and a["verdict"].startswith("PASS")
    a_edge = acceptance("allenai/Molmo2-4B", "vqav2", {**rv, "acc": 0.812})   # delta −5.4: inside ±6
    assert a_edge["pass_"] is True and a_edge["verdict"].startswith("PASS")
    bad = acceptance("allenai/Molmo2-4B", "vqav2", {**rv, "acc": 0.80})       # delta −6.6: outside ±6
    assert bad["pass_"] is False and bad["verdict"].startswith("FAIL") and "band ±6.0" in bad["verdict"]
    na = acceptance("allenai/Molmo2-4B", "refcoco", rec)
    assert na["pass_"] is None and na["verdict"].startswith("N/A")

    # generic (untested) path renders through a stub processor's template
    class _Proc:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "<img>" + messages[0]["content"][1]["text"] + "<gen>"
    G = generic_prompts(_Proc())
    assert G["refcoco"](rows_r[0]).startswith("<img>Locate object 0 in the image") and G["vqav2"](rows_v[0]).endswith("<gen>")
    print("published anchors:", {t: PUBLISHED["allenai/Molmo2-4B"][t][0] for t in TASKS})
    print("PASS vlm_skill_eval dry-run (stub forward; Molmo2 prompts; point/box parse; VQA soft/em/ANLS; "
          "candidate sets + scoring; acceptance; §12.11 samples on all three readouts)")


if __name__ == "__main__":
    _dryrun()
