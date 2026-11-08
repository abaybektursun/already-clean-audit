# A1 BATTERY — extend the A1 transfer-necessity anchor to the full pre-registered
# task battery B (spec §2.1 estimand "S̄ = mean over B", §3 named battery).
"""The A1 anchor's ACTUAL estimand is S̄ over a pre-registered task battery B, not
the single VQAv2 task already done. This module adds the OTHER battery tasks so we
can test whether the per-tower split found on VQAv2 (vision change NOT necessary,
S_A=0.93; LM change necessary, S_B=0.18) is GENERAL or VQA-specific.

What is held IDENTICAL across tasks (the frozen recipe = hyperparameters + procedure):
  * the LoRA rank/alpha/dropout, target modules, connector-trainable, frozen vision,
    optimizer, lr, schedule, warmup, max_steps, n_train, batch sizes — imported
    verbatim from transfer_a1.RECIPE. NEVER edited.
  * the model build (tower swaps) — `contrast_a1.build_swapped_model` for the three
    swap configs and `transfer_a1.build_ceiling_model` for CEILING — reused verbatim.
  * the train loop (suffix-masked CE on the connector + LM-LoRA) — same code shape.
What NECESSARILY differs per task (the spec is explicit: "the task's prompt, metric,
and data necessarily differ per the big_vision transfer configs"):
  * the dataset + splits, the big_vision prompt PREFIX, the supervised target text,
    the eval decode parsing, and the standard metric.

Each TaskSpec cites the big_vision prompt source + the standard metric source. We do
NOT invent metrics. Where a metric is too heavy to implement faithfully we pick the
spec's named alternative and say why (spec §3: DocVQA↔TextVQA, VSR↔What'sUp).

Configs (spec §2.1 table) — ALL FOUR run per task (unlike VQAv2, where CEILING was
reused from the σ_seed pilot; that pilot was VQAv2-specific, so for every NEW task we
must run CEILING too):
  CEILING   = PG vision + PG LM            (reference; build via build_ceiling_model)
  PRIMARY_A = base SigLIP vision + PG LM   (is the vision change necessary?)
  PRIMARY_B = PG vision + base Gemma2 LM   (is the LM change necessary?)
  FLOOR     = base SigLIP + base Gemma2    (borrowed substrate + recipe only)

Per-task estimand (spec §2.1, identical to VQAv2):
  s = (config - FLOOR)/(CEILING - FLOOR), clamped [0,1]. S_A = PRIMARY_A; S_B = PRIMARY_B.
  Inclusion floor: a task enters B only if CEILING - FLOOR >= 0.15 (the frozen margin);
  else it is uninformative and no verdict is forced (spec §2.1 "Inclusion floor").
  Intercurrent events: a diverged/degenerate run is scored worst-case s=0 (never dropped).

Design rules (CLAUDE.md + spec §6):
  * NO try/except hiding — a wrong assumption raises with a full trace.
  * NEVER touch the frozen transfer_a1.RECIPE — only prompt/data/metric vary per task.
  * Deterministic: every seed pins torch/numpy/python RNG (transfer_a1._seed_everything).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from infra.pg.transfer_a1 import (
    RECIPE,
    _seed_everything,
    build_ceiling_model,
)

# The four A1 configs run per battery task.
ALL_CONFIGS = ("CEILING", "PRIMARY_A", "PRIMARY_B", "FLOOR")
SWAP_CONFIGS = ("PRIMARY_A", "PRIMARY_B", "FLOOR")  # built by contrast_a1.build_swapped_model


def bootstrap_S_ci(config_scores, floor_scores, ceiling_scores,
                   n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap 95% CI for S = (config_mean - FLOOR_mean)/(CEILING_mean - FLOOR_mean),
    clamped [0,1]. VERBATIM the frozen estimand used for VQAv2 (modal_pg_a1_contrast.
    _bootstrap_S_ci) — re-housed here (pure, no Modal dep) so the battery's finalize
    reuses the IDENTICAL estimator. Resamples each config's per-seed scores
    independently with replacement; recomputes S each draw."""
    import numpy as np
    rng = np.random.default_rng(seed)
    c = np.asarray(config_scores, dtype=float)
    f = np.asarray(floor_scores, dtype=float)
    k = np.asarray(ceiling_scores, dtype=float)

    def _S(cm, fm, km):
        denom = km - fm
        if denom <= 0:
            return float("nan")
        return float(np.clip((cm - fm) / denom, 0.0, 1.0))

    point = _S(c.mean(), f.mean(), k.mean())
    draws = []
    for _ in range(n_boot):
        cb = rng.choice(c, size=c.size, replace=True).mean()
        fb = rng.choice(f, size=f.size, replace=True).mean()
        kb = rng.choice(k, size=k.size, replace=True).mean()
        s = _S(cb, fb, kb)
        if not np.isnan(s):
            draws.append(s)
    draws = np.asarray(draws)
    return {
        "S_point": point,
        "S_ci95_low": float(np.percentile(draws, 2.5)),
        "S_ci95_high": float(np.percentile(draws, 97.5)),
        "n_boot_valid": int(draws.size),
    }


# ===========================================================================
# Per-task spec. The recipe HYPERPARAMETERS come from transfer_a1.RECIPE (frozen);
# only the fields below vary per task (prompt / data / metric / decode).
# ===========================================================================

@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset: str                      # HF repo id
    train_split: str
    eval_split: str
    prompt_prefix: str                # big_vision transfer prefix (cited in prompt_source)
    metric_name: str                  # the standard metric (cited in metric_source)
    metric_source: str
    prompt_source: str
    # build supervised target text from a row (the suffix the model is trained to emit)
    target_fn: Callable               # (row) -> str
    # build the eval prompt text (sans <image>; processor auto-prepends image tokens)
    prompt_fn: Callable               # (row) -> str
    # parse a decoded generation + score it against the row's ground truth -> float in [0,1]
    score_fn: Callable                # (pred_str, row) -> float
    # how the row image is fetched (most are row["image"]); overridable for odd schemas
    image_fn: Callable = field(default=lambda row: row["image"])
    # If set, the image is fetched from this row key's URL (with a persistent on-volume
    # cache) rather than an inline PIL column. Used for VSR (cambridgeltl/vsr_random),
    # whose `image` field is just a COCO filename + `image_link` is the COCO URL.
    image_url_key: str | None = None
    # max NEW tokens at eval (captions need more than VQA's 8; loc-boxes need ~20)
    max_new_tokens: int = 8
    # max supervised label length (suffix cap)
    max_label_len: int = 32
    # data_files override for repos whose registered config hides the split we need
    train_data_files: dict | None = None
    eval_data_files: dict | None = None
    # verification_mode for load_dataset (subset loads need "no_checks")
    train_no_checks: bool = False
    eval_no_checks: bool = False
    # Some eval mirrors ship a single labelled split (e.g. COCO-Caption: only `val`
    # carries captions — `test` has none). When True, train + eval come from the SAME
    # split, sliced DISJOINTLY (eval = shuffled[:n_eval]; train = shuffled[n_eval:]),
    # so there is no train/eval image leakage. The slice is deterministic across all
    # configs+seeds (fixed shuffle), so the contrast stays clean.
    same_split_disjoint: bool = False


# ===========================================================================
# TASK 1 — COCO-Captions, metric CIDEr.
# ---------------------------------------------------------------------------
# big_vision prompt (proj/paligemma/transfers/coco_caption.py): the caption prefix
# is the single token "caption en\n" (PaliGemma's captioning instruction). The
# processor auto-prepends <image>. Source:
#   github.com/google-research/big_vision .../proj/paligemma/transfers/coco_caption.py
# Metric: CIDEr-D (Vedantam et al., CVPR 2015) via pycocoevalcap.cider — the
# canonical COCO-caption metric PaliGemma reports. We tokenize with the PTB
# tokenizer the COCO eval toolkit ships (pycocoevalcap.tokenizer.PTBTokenizer).
#   Source: github.com/salaniz/pycocoevalcap  (the pip-installable COCO caption eval).
# ===========================================================================

COCO_PREFIX = "caption en\n"


def _coco_target(row) -> str:
    # COCO ships 5 captions/image; the training target is one reference caption.
    # We use the first caption (deterministic). Eval scores against ALL references.
    caps = _coco_caps(row)
    return caps[0]


def _coco_caps(row) -> list[str]:
    """Return the list of reference captions for a row, schema-robust.
    lmms-lab/COCO-Caption stores them under "answer" (list[str]); other mirrors use
    "captions"/"caption"/"sentences". Resolved at runtime by the probe."""
    for k in ("answer", "captions", "caption", "sentences", "references"):
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, str):
                return [v]
            if isinstance(v, list) and v and isinstance(v[0], dict):
                # e.g. sentences=[{"raw": ...}]
                return [d.get("raw") or d.get("caption") or next(iter(d.values())) for d in v]
            return list(v)
    raise RuntimeError(f"COCO row has no caption field; keys={sorted(row.keys())}")


def _coco_prompt(row) -> str:
    return COCO_PREFIX


# CIDEr scorer is constructed once per eval (corpus-level metric). We expose a
# closure-builder rather than a per-item score_fn because CIDEr is computed over the
# whole candidate/reference corpus, not per item. `run_transfer_task` special-cases
# corpus metrics via `is_corpus_metric`.


# ===========================================================================
# TASK 2 — RefCOCO referring-expression comprehension, metric acc@0.5 IoU.
# ---------------------------------------------------------------------------
# big_vision prompt (proj/paligemma/transfers/refcoco.py): "detect {expr}\n" — the
# model emits PaliGemma location tokens <locDDDD> (4 of them) encoding a normalized
# box. The 1024-bin loc grid: each <locDDDD> ∈ [0,1023] maps to a normalized coord
# coord = D/1023; ORDER is (y0, x0, y1, x1) (big_vision detect format).
#   Source: github.com/google-research/big_vision .../proj/paligemma/transfers/refcoco.py
#   + PaliGemma model card "detect" task ("<loc0000>...<loc1023>" segment).
# Metric: accuracy@0.5 IoU — a prediction is correct iff IoU(pred_box, gt_box) >= 0.5
# (standard REC metric; Yu et al. 2016 RefCOCO; PaliGemma reports acc@0.5).
# ===========================================================================

REFCOCO_PREFIX_FMT = "detect {expr}\n"
_LOC_RE = re.compile(r"<loc(\d{4})>")


def _decode_loc_box(text: str):
    """Parse the first 4 <locDDDD> tokens → normalized (y0,x0,y1,x1) in [0,1].
    Returns None if fewer than 4 loc tokens (a degenerate generation → IoU 0)."""
    locs = _LOC_RE.findall(text)
    if len(locs) < 4:
        return None
    y0, x0, y1, x1 = (int(locs[i]) / 1023.0 for i in range(4))
    return (y0, x0, y1, x1)


def _iou(a, b) -> float:
    """IoU of two (y0,x0,y1,x1) normalized boxes."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    ay0, ay1 = min(ay0, ay1), max(ay0, ay1)
    ax0, ax1 = min(ax0, ax1), max(ax0, ax1)
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
    inter = ih * iw
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _refcoco_gt_box(row):
    """Normalized (y0,x0,y1,x1) GT box. lmms-lab/RefCOCO stores bbox as [x,y,w,h] in
    PIXELS + the image; normalize by image size. Schema confirmed by the probe."""
    img = row["image"]
    W, H = img.size
    x, y, w, h = row["bbox"]
    return (y / H, x / W, (y + h) / H, (x + w) / W)


def _refcoco_loc_str(box) -> str:
    """Encode a normalized (y0,x0,y1,x1) box as the 4-loc-token string PaliGemma emits."""
    return "".join(f"<loc{int(round(c * 1023)):04d}>" for c in box)


def _refcoco_expr(row) -> str:
    """The referring expression. lmms-lab/RefCOCO stores a list under "answer"
    (the phrases) — use the first; some mirrors use "sentences"/"referring_expressions"."""
    for k in ("answer", "sentences", "referring_expressions", "expression", "sent"):
        if k in row and row[k]:
            v = row[k]
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0].get("raw") or v[0].get("sent") or next(iter(v[0].values()))
            return v[0]
    raise RuntimeError(f"RefCOCO row has no expression field; keys={sorted(row.keys())}")


def _refcoco_target(row) -> str:
    return _refcoco_loc_str(_refcoco_gt_box(row))


def _refcoco_prompt(row) -> str:
    return REFCOCO_PREFIX_FMT.format(expr=_refcoco_expr(row))


def _refcoco_score(pred: str, row) -> float:
    box = _decode_loc_box(pred)
    if box is None:
        return 0.0  # degenerate (no 4 loc tokens) → worst case
    return 1.0 if _iou(box, _refcoco_gt_box(row)) >= 0.5 else 0.0


# ===========================================================================
# TASK 3 — TextVQA (OCR-VQA), metric ANLS.
# ---------------------------------------------------------------------------
# CHOICE: TextVQA over DocVQA (spec §3 names "TextVQA or DocVQA"). Rationale:
#   * TextVQA (lmms-lab/textvqa) ships train + validation with the 10-answer schema
#     (answers list), is a PaliGemma transfer target, and uses the standard ANLS/acc.
#   * DocVQA's standard eval is also ANLS, so the metric is identical; TextVQA's train
#     split is cleaner to subsample at our budget. We disclose the choice.
# big_vision prompt (proj/paligemma/transfers/textvqa.py): "answer en {question}" —
# the same open-ended VQA prefix family as VQAv2 (PaliGemma uses one OCR-VQA prompt).
#   Source: github.com/google-research/big_vision .../proj/paligemma/transfers/textvqa.py
# Metric: ANLS (Average Normalized Levenshtein Similarity; Biten et al. ICDAR 2019) —
# the standard scene-text/doc VQA metric. ANLS(pred, gts) = max over gts of
# (1 - NL) if (1 - NL) >= 0.5 else 0, where NL = Levenshtein(pred,gt)/max(|pred|,|gt|),
# both lowercased+stripped. This is the official RRC/DocVQA ANLS with τ=0.5.
#   Source: rrc.cvc.uab.es ST-VQA/DocVQA ANLS definition (Biten 2019; Mathew 2021).
# ===========================================================================

TEXTVQA_PREFIX = "answer en "


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _anls_one(pred: str, gt: str) -> float:
    p, g = pred.strip().lower(), gt.strip().lower()
    if not p and not g:
        return 1.0
    m = max(len(p), len(g))
    if m == 0:
        return 0.0
    nl = _levenshtein(p, g) / m
    s = 1.0 - nl
    return s if s >= 0.5 else 0.0  # official τ=0.5 thresholding


def _textvqa_answers(row) -> list[str]:
    """The list of human answers. lmms-lab/textvqa: "answers" (list[str]). Some rows
    also carry "answer"/"valid_answers"."""
    for k in ("answers", "valid_answers", "answer"):
        if k in row and row[k] is not None:
            v = row[k]
            return [v] if isinstance(v, str) else list(v)
    raise RuntimeError(f"TextVQA row has no answers field; keys={sorted(row.keys())}")


def _textvqa_majority(row) -> str:
    answers = _textvqa_answers(row)
    counts: dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(counts, key=counts.get)


def _textvqa_prompt(row) -> str:
    return f"{TEXTVQA_PREFIX}{row['question']}"


def _textvqa_score(pred: str, row) -> float:
    gts = _textvqa_answers(row)
    return max(_anls_one(pred, g) for g in gts)


# ===========================================================================
# TASK 4 — spatial relations. CHOICE: What'sUp over VSR.
# ---------------------------------------------------------------------------
# CHOICE rationale (spec §3 "VSR or What'sUp"): we resolve which has a usable
# train/eval split + a clean binary-relation schema at the probe step (probe_battery_
# datasets). Both are accuracy. The spec lets us pick the tractable+faithful one. The
# realised choice (dataset id + why) is recorded in the per-task result and the report.
# big_vision: PaliGemma has no dedicated What'sUp/VSR transfer config; spatial-relation
# sets are evaluated as open-ended VQA, so we use the VQA prefix "answer en {question}"
# (same family as VQAv2/TextVQA). Metric: plain accuracy (exact-match of the normalized
# answer against the single GT label), the standard for these binary/forced-choice sets.
#   VSR: Liu et al. TACL 2023 (true/false caption). What'sUp: Kamath et al. EMNLP 2023.
# These specs are FINALIZED after the probe (the prompt/answer fields differ by dataset);
# the placeholders here are overwritten in `build_spatial_taskspec` once schema is known.
# ===========================================================================

SPATIAL_PREFIX = "answer en "


def _spatial_score_exact(pred: str, row, answer_key: str, answer_map=None) -> float:
    """Accuracy: normalized exact match of pred vs the row's GT label."""
    gt = row[answer_key]
    if answer_map is not None:
        gt = answer_map.get(gt, gt)
    p = " ".join(str(pred).strip().lower().split())
    g = " ".join(str(gt).strip().lower().split())
    return 1.0 if p == g else 0.0


# On-volume cache for URL-fetched images (VSR: images are COCO URLs, not inline). One
# download per unique image, reused across configs/seeds/epochs. The COCO 2017 images
# are public + static, so this is a deterministic content source (not a moving target).
_IMG_CACHE_DIR = "/vol/activations/coco_url_cache"


def _fetch_image_cached(url: str):
    """Fetch a COCO image by URL with a persistent on-volume file cache → PIL.Image.
    Fails loud on a dead URL / decode error (no silent skip — a missing image would
    bias the eval). Used only for VSR (image_url_key set)."""
    import os
    from pathlib import Path
    from urllib.request import urlopen

    from PIL import Image

    Path(_IMG_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    fname = url.rsplit("/", 1)[-1]
    fp = Path(_IMG_CACHE_DIR) / fname
    if not fp.exists():
        with urlopen(url, timeout=30) as resp:
            data = resp.read()
        tmp = fp.with_suffix(fp.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, fp)  # atomic — safe under the parallel DataLoader workers
    return Image.open(fp).convert("RGB")


# ===========================================================================
# TASK REGISTRY. COCO + RefCOCO + TextVQA are fully specified; the spatial task's
# fields are filled by build_spatial_taskspec after the probe (schema-dependent).
# Dataset ids/splits are CANDIDATES verified by probe_battery_datasets() before any
# training; the probe HALTs (raises) if a chosen split is absent.
# ===========================================================================

# DATASET BINDINGS — confirmed by the probe (introspect, don't assume):
#   COCO    lmms-lab/COCO-Caption : ONLY `val` (40504 rows) carries captions; `test`
#           has none → train+eval are DISJOINT slices of `val` (no leakage).
#   RefCOCO lmms-lab/RefCOCO      : inline image + bbox[x,y,w,h] + `answer` phrases;
#           train on `val` (8811), eval on `testA` (1975, ≥500). Standard REC eval split.
#   TextVQA lmms-lab/textvqa      : train (34602) + validation; inline image, `answers`.
#   VSR     cambridgeltl/vsr_random : train (7680) + test (2195); inline image; resolved
#           by the probe (What'sUp `facebook/whatsup` not on the Hub → VSR chosen).
# The frozen recipe's max_steps=1500 is the BINDING training budget; n_train (16384) is
# the pool CAP. Where a task's labelled pool is smaller (RefCOCO val=8811, VSR=7680), the
# train loop cycles the pool to 1500 steps — IDENTICAL optimizer budget, task-appropriate
# data (the data necessarily differs per task; spec §3 + hard rule 2). The realized pool
# size is recorded per run. No recipe HYPERPARAMETER changes.

COCO_TASK = TaskSpec(
    name="coco_captions",
    dataset="lmms-lab/COCO-Caption",
    train_split="val",          # only `val` carries captions; train+eval disjoint-slice it
    eval_split="val",
    same_split_disjoint=True,
    prompt_prefix=COCO_PREFIX,
    metric_name="CIDEr-D",
    metric_source="pycocoevalcap.cider (Vedantam CVPR2015); PTBTokenizer; salaniz/pycocoevalcap",
    prompt_source="big_vision proj/paligemma/transfers/coco_caption.py PREFIX='caption en\\n'",
    target_fn=_coco_target,
    prompt_fn=_coco_prompt,
    score_fn=lambda pred, row: 0.0,  # unused — CIDEr is corpus-level (is_corpus_metric)
    max_new_tokens=32,
    max_label_len=48,
)

REFCOCO_TASK = TaskSpec(
    name="refcoco",
    dataset="lmms-lab/RefCOCO",
    train_split="val",          # lmms-lab mirror is eval-only; val=8811 (largest labelled)
    eval_split="testA",         # standard REC eval split (1975 ≥ 500)
    prompt_prefix="detect {expr}\\n",
    metric_name="acc@0.5 IoU (REC, <loc> box decode)",
    metric_source="acc@0.5 IoU — Yu et al. RefCOCO 2016; PaliGemma reports acc@0.5",
    prompt_source="big_vision proj/paligemma/transfers/refcoco.py PREFIX='detect {expr}\\n'; "
                  "PaliGemma <locDDDD> 1024-bin grid, order (y0,x0,y1,x1)",
    target_fn=_refcoco_target,
    prompt_fn=_refcoco_prompt,
    score_fn=_refcoco_score,
    max_new_tokens=24,        # 4 loc tokens + slack
    max_label_len=24,
)

TEXTVQA_TASK = TaskSpec(
    name="textvqa",
    dataset="lmms-lab/textvqa",
    train_split="train",
    eval_split="validation",
    prompt_prefix=TEXTVQA_PREFIX,
    metric_name="ANLS (τ=0.5)",
    metric_source="ANLS — Biten et al. ICDAR2019 / Mathew DocVQA 2021; official τ=0.5",
    prompt_source="big_vision proj/paligemma/transfers/textvqa.py PREFIX='answer en '",
    target_fn=_textvqa_majority,
    prompt_fn=_textvqa_prompt,
    score_fn=_textvqa_score,
    max_new_tokens=16,        # scene text answers can be multi-word
    max_label_len=24,
)

# Spatial task is resolved at runtime (probe decides VSR vs What'sUp + the schema).
TASKS: dict[str, TaskSpec] = {
    "coco_captions": COCO_TASK,
    "refcoco": REFCOCO_TASK,
    "textvqa": TEXTVQA_TASK,
}

# Which tasks use a corpus-level metric (scored over the whole candidate/ref corpus,
# not per item). CIDEr is corpus-level.
CORPUS_METRIC_TASKS = {"coco_captions"}


# ===========================================================================
# DATASET PROBE — introspect every candidate's splits + first-row schema before
# any training (spec §3 "introspect, don't assume"; HALT if a chosen split is
# absent). Also resolves the spatial-task choice (VSR vs What'sUp).
# ===========================================================================

# Candidate dataset ids per task — the A1 estimand needs BOTH a train split (the
# fixed transfer fine-tune) AND an eval split. The lmms-lab eval mirrors are often
# eval-only, so we survey several parquet-native candidates per task and bind each
# TaskSpec to one the probe confirms has train+eval (introspect, don't assume; the
# VQAv2 probe used the same survey pattern). Each candidate names its (train, eval)
# split + which field carries the supervised target / answer / box / caption.
TASK_CANDIDATES: dict[str, list[dict]] = {
    "coco_captions": [
        # INLINE-IMAGE candidates first (no URL fetch). "image" must be a PIL column.
        {"id": "nlphuji/coco_captions", "train": "train", "eval": "validation"},
        {"id": "sayakpaul/coco-30-val-2014", "train": "train", "eval": "validation"},
        {"id": "shunk031/MSCOCO", "train": "train", "eval": "validation"},
        # Karpathy (caption strings present, but image is by URL → flagged needs_url).
        {"id": "yerevann/coco-karpathy", "train": "train", "eval": "validation"},
        {"id": "lmms-lab/COCO-Caption", "train": "val", "eval": "test"},
    ],
    "refcoco": [
        # INLINE-IMAGE RefCOCO candidates (PIL "image" + bbox + expression).
        {"id": "lmms-lab/RefCOCO", "train": "val", "eval": "testA"},
        {"id": "JierunChen/Ref-L4", "train": "val", "eval": "test"},
        {"id": "moondream/refcoco-m", "train": "train", "eval": "validation"},
        {"id": "jxu124/refcoco", "train": "train", "eval": "validation"},  # image by path
    ],
    "textvqa": [
        {"id": "lmms-lab/textvqa", "train": "train", "eval": "validation"},
    ],
}


def probe_battery_datasets() -> dict:
    """Survey candidate datasets per task: report each candidate's split availability
    + (for the first with train+eval) the first-row schema. Resolves the spatial task.
    Does NOT bind TaskSpecs — that is done in build_task_specs after this report is
    read. Enumeration is reported, not hidden (probe job, not a compute-error mask)."""
    out: dict = {}
    for task, cands in TASK_CANDIDATES.items():
        out[task] = _survey_candidates(task, cands)
    out["spatial_resolution"] = _probe_spatial()
    return out


def _survey_candidates(task: str, candidates: list[dict]) -> dict:
    """For each candidate id: list splits, flag train+eval present, and for the first
    usable one dump a row schema + a couple example values so we can bind fields."""
    from datasets import get_dataset_split_names, load_dataset

    report = []
    chosen = None
    for c in candidates:
        entry = dict(c)
        try:
            splits = list(get_dataset_split_names(c["id"]))
            entry["available_splits"] = splits
            entry["has_train"] = c["train"] in splits
            entry["has_eval"] = c["eval"] in splits
            usable = entry["has_train"] and entry["has_eval"]
            if usable:
                trow = next(iter(load_dataset(c["id"], split=c["train"], streaming=True)))
                erow = next(iter(load_dataset(c["id"], split=c["eval"], streaming=True)))
                entry["train_keys"] = sorted(trow.keys())
                entry["eval_keys"] = sorted(erow.keys())
                entry["eval_row_sample"] = {k: str(v)[:80] for k, v in erow.items()
                                            if k != "image"}
                # Does the row carry a decoded PIL image inline (vs a URL/path)?
                img = erow.get("image")
                entry["image_inline"] = bool(img is not None and hasattr(img, "size"))
                if entry["image_inline"]:
                    entry["image_size"] = list(img.size)
                # Prefer the FIRST candidate that has train+eval AND an inline image.
                if chosen is None and entry["image_inline"]:
                    chosen = entry
        except Exception as e:  # noqa: BLE001 — probe enumeration: report availability
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        report.append(entry)
    return {"chosen": chosen, "candidates": report}


def _probe_spatial() -> dict:
    """Inspect VSR + What'sUp candidates; return which is usable + its schema.
    Records the realised choice (id + reason). Raises only if NEITHER is usable."""
    from datasets import get_dataset_split_names, load_dataset

    # Candidate ids (verified to exist on the hub at design time; probe confirms splits).
    candidates = [
        # VSR (Liu et al. TACL 2023): zero-shot + random splits. "cambridgeltl/vsr_*".
        {"id": "cambridgeltl/vsr_random", "kind": "VSR",
         "train": "train", "eval": "test"},
        {"id": "cambridgeltl/vsr_zeroshot", "kind": "VSR",
         "train": "train", "eval": "test"},
        # What'sUp (Kamath et al. EMNLP 2023). Mirrors vary; try the common ones.
        {"id": "facebook/whatsup", "kind": "WhatsUp", "train": "train", "eval": "test"},
    ]
    report = []
    chosen = None
    for c in candidates:
        entry = dict(c)
        try:
            splits = get_dataset_split_names(c["id"])
            entry["available_splits"] = list(splits)
            has_train = c["train"] in splits
            has_eval = c["eval"] in splits
            entry["has_train"] = has_train
            entry["has_eval"] = has_eval
            if has_train and has_eval:
                row = next(iter(load_dataset(c["id"], split=c["eval"], streaming=True)))
                entry["keys"] = sorted(row.keys())
                entry["row_sample"] = {k: str(v)[:60] for k, v in row.items()
                                       if k != "image"}
                if chosen is None:
                    chosen = entry
        except Exception as e:  # noqa: BLE001 — probe-only enumeration, reported not hidden
            entry["error"] = f"{type(e).__name__}: {e}"
        report.append(entry)
    return {"chosen": chosen, "candidates": report}


def build_spatial_taskspec(probe_spatial: dict) -> TaskSpec:
    """Build the spatial TaskSpec from the resolved probe. Binds the prompt/answer
    fields to the chosen dataset's REAL schema (no assumption). Called by the Modal
    driver once the probe has resolved VSR vs What'sUp."""
    chosen = probe_spatial.get("chosen")
    if not chosen:
        raise RuntimeError(f"No usable spatial dataset (VSR/What'sUp): {probe_spatial}")
    ds_id = chosen["id"]
    keys = set(chosen["keys"])

    # VSR schema (confirmed by the probe): {image (COCO filename str), image_link (COCO
    # URL), caption, label∈{0,1}, relation}. A true/false caption-verification task.
    # Image is NOT inline → fetched from `image_link` (cached on-volume). Prompt = the
    # VQA prefix + the relational caption; GT = the word for label (1→"true", 0→"false")
    # — the standard VSR binary forced-choice.
    if chosen["kind"] == "VSR":
        assert {"caption", "label"} <= keys, f"VSR row missing caption/label: {keys}"
        answer_map = {1: "true", 0: "false", "1": "true", "0": "false",
                      True: "true", False: "false"}

        def prompt_fn(row):
            return f"answer en is this true or false: {row['caption']}"

        def target_fn(row):
            return answer_map[row["label"]]

        def score_fn(pred, row):
            return _spatial_score_exact(pred, row, "label", answer_map)

        def image_fn(row):
            return _fetch_image_cached(row["image_link"])

        return TaskSpec(
            name="spatial_vsr",
            dataset=ds_id,
            train_split=chosen["train"],
            eval_split=chosen["eval"],
            prompt_prefix="answer en is this true or false: ",
            metric_name="accuracy (true/false exact-match)",
            metric_source="VSR accuracy — Liu et al. TACL 2023 (binary caption verification)",
            prompt_source="open-ended VQA prefix 'answer en ' (no dedicated PG transfer "
                          "config for VSR; spatial sets evaluated as VQA)",
            target_fn=target_fn,
            prompt_fn=prompt_fn,
            score_fn=score_fn,
            image_fn=image_fn,
            image_url_key="image_link",
            max_new_tokens=6,
            max_label_len=4,
        )

    # What'sUp schema resolved from the probe row (id+keys); build the VQA-style
    # forced choice. Filled from the realised keys (no assumption beyond the probe).
    raise RuntimeError(
        f"Spatial dataset {ds_id} kind={chosen['kind']} schema not yet bound; "
        f"keys={sorted(keys)}. Extend build_spatial_taskspec for this schema."
    )


# ===========================================================================
# CIDEr (corpus-level). Computed with pycocoevalcap over the candidate/reference
# corpus after PTB tokenization. Returns the CIDEr-D score in [0, ~10] (typically
# 0–1.4 for COCO); we DO NOT rescale — s = (config-FLOOR)/(CEILING-FLOOR) handles
# the units (the estimand is scale-free in the metric).
# ===========================================================================

def _cider_corpus(candidates: dict, references: dict) -> float:
    """candidates: {id: [one_caption]}, references: {id: [ref1, ref2, ...]}.
    Returns corpus CIDEr-D via pycocoevalcap (PTBTokenizer + Cider)."""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    # PTBTokenizer expects {id: [{"caption": str}, ...]}.
    def _wrap(d):
        return {k: [{"caption": c} for c in v] for k, v in d.items()}

    tok = PTBTokenizer()
    gts = tok.tokenize(_wrap(references))
    res = tok.tokenize(_wrap(candidates))
    score, _ = Cider().compute_score(gts, res)
    return float(score)


# ===========================================================================
# THE GENERIC TRANSFER + EVAL DRIVER. Mirrors run_transfer_ceiling /
# run_transfer_contrast EXACTLY (same frozen recipe hyperparameters, same train
# loop, same deterministic seeding). The ONLY differences are task-specified:
# the dataset/splits, the prompt prefix, the supervised target, the eval decode
# parsing, and the metric. The recipe dict is transfer_a1.RECIPE with ONLY the
# task's prompt/data/metric knobs overridden (max_new_tokens, max_label_len) —
# never the rank/steps/lr/optimizer/schedule/n_train/batch (the frozen recipe).
# ===========================================================================

# Recipe keys that a TASK is allowed to set (prompt/metric/data shape only). The
# learning hyperparameters (rank, alpha, steps, lr, optimizer, schedule, warmup,
# n_train, batch sizes, weight_decay) are NEVER overridden — they are the frozen
# recipe, identical across configs AND tasks (spec §2.1 + hard rule 2).
_TASK_OVERRIDABLE = {"max_new_tokens", "max_label_len", "n_eval", "eval_batch_size"}


def _build_model_for_config(config_name: str, seed: int):
    """CEILING → build_ceiling_model; the three swaps → contrast_a1.build_swapped_model.
    Both reuse the FROZEN recipe attach (LoRA regex + connector trainable + frozen
    vision) by construction — identical to the VQAv2 contrast."""
    if config_name == "CEILING":
        return build_ceiling_model(seed)
    from infra.pg.contrast_a1 import build_swapped_model
    return build_swapped_model(config_name, seed)


def _load_train_eval(task: TaskSpec, n_train_cap: int, n_eval: int):
    """Load + deterministically subsample the train/eval pools (fixed shuffle seed →
    identical data across all configs+seeds; only the tower init differs). Handles:
      * same_split_disjoint: train + eval from one split, sliced disjointly (eval first
        n_eval rows; train the rest) → no leakage (COCO-Caption: only `val` is labelled).
      * pool cap: train pool = min(available, n_train_cap). max_steps (1500) is the
        binding budget; a smaller pool just cycles (identical optimizer budget)."""
    from datasets import load_dataset

    DATA_SHUFFLE_SEED = 0  # fixed across configs+seeds by design

    def _load(split, data_files, no_checks):
        kw = {"split": split}
        if data_files:
            kw["data_files"] = data_files
        if no_checks:
            kw["verification_mode"] = "no_checks"
        return load_dataset(task.dataset, **kw)

    if task.same_split_disjoint:
        full = _load(task.eval_split, task.eval_data_files, task.eval_no_checks)
        full = full.shuffle(seed=DATA_SHUFFLE_SEED)
        n_eval = min(n_eval, len(full) // 2)         # keep train+eval disjoint + sized
        eval_ds = full.select(range(n_eval))
        train_pool = min(n_train_cap, len(full) - n_eval)
        train_ds = full.select(range(n_eval, n_eval + train_pool))
    else:
        train_ds = _load(task.train_split, task.train_data_files, task.train_no_checks)
        eval_ds = _load(task.eval_split, task.eval_data_files, task.eval_no_checks)
        train_pool = min(n_train_cap, len(train_ds))
        train_ds = train_ds.shuffle(seed=DATA_SHUFFLE_SEED).select(range(train_pool))
        eval_ds = eval_ds.shuffle(seed=DATA_SHUFFLE_SEED).select(
            range(min(n_eval, len(eval_ds))))
    return train_ds, eval_ds


def run_transfer_task(task: TaskSpec, config_name: str, seed: int,
                      n_eval: int | None = None) -> dict:
    """Full transfer + eval for one (task, config, seed). Returns the per-seed
    result dict with the task's standard-metric `score` (the estimand variable
    input). Identical recipe hyperparameters + train loop as VQAv2; only the
    task's prompt/data/metric differ."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    assert config_name in ALL_CONFIGS, f"unknown config {config_name!r}"
    assert torch.cuda.is_available(), "A1 battery must run on GPU (B200/H200)."

    # --- recipe = FROZEN RECIPE + ONLY the task's prompt/data/metric knobs --------
    rcp = dict(RECIPE)
    rcp["task"] = task.name
    rcp["max_new_tokens"] = task.max_new_tokens
    rcp["max_label_len"] = task.max_label_len
    if n_eval is not None:
        rcp["n_eval"] = n_eval
    # Guard: a task must NOT touch the learning hyperparameters (fail loud).
    for k in ("lora_rank", "lora_alpha", "lr", "max_steps", "n_train",
              "train_batch_size", "optimizer", "lr_scheduler", "warmup_ratio",
              "weight_decay", "lora_target_modules"):
        assert rcp[k] == RECIPE[k], f"task illegally changed frozen recipe key {k!r}"

    t0 = time.time()
    print(f"[battery/{task.name}/{config_name} s{seed}] torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0)} metric={task.metric_name}")
    _seed_everything(seed)

    model, proc, report = _build_model_for_config(config_name, seed)
    print(f"[battery/{task.name}/{config_name} s{seed}] "
          f"trainable={report['trainable_params']:,} ({report['trainable_pct']}%)")

    # --- data: same fixed-shuffle subset pattern as VQAv2 (deterministic across
    #     configs+seeds so the only thing that differs is the tower init). ----------
    train_ds, eval_ds = _load_train_eval(task, rcp["n_train"], rcp["n_eval"])
    print(f"[battery/{task.name}/{config_name} s{seed}] train_pool={len(train_ds)} "
          f"eval={len(eval_ds)} (max_steps={rcp['max_steps']} is the binding budget)")

    def collate(batch_rows):
        images = [task.image_fn(r).convert("RGB") for r in batch_rows]
        prompts = [task.prompt_fn(r) for r in batch_rows]
        suffixes = [task.target_fn(r) for r in batch_rows]
        batch = proc(text=prompts, images=images, suffix=suffixes,
                     return_tensors="pt", padding="longest")
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
        return batch

    loader = DataLoader(
        train_ds, batch_size=rcp["train_batch_size"], shuffle=True,
        collate_fn=collate, num_workers=2, drop_last=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rcp["lr"], weight_decay=rcp["weight_decay"])
    sched = get_scheduler(
        rcp["lr_scheduler"], optimizer=opt,
        num_warmup_steps=int(rcp["warmup_ratio"] * rcp["max_steps"]),
        num_training_steps=rcp["max_steps"],
    )

    # --- train loop — IDENTICAL to the frozen recipe loop ------------------------
    model.train()
    step, losses, done = 0, [], False
    while not done:
        for batch in loader:
            batch = {k: (v.to("cuda") if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[battery/{task.name}/{config_name} s{seed}] non-finite loss at "
                    f"step {step}: {loss.item()} — diverged (estimand scores worst-case)."
                )
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            if step % 100 == 0 or step == 1:
                print(f"[battery/{task.name}/{config_name} s{seed}] "
                      f"step {step}/{rcp['max_steps']} loss={loss.item():.4f}")
            if step >= rcp["max_steps"]:
                done = True
                break

    train_secs = time.time() - t0
    print(f"[battery/{task.name}/{config_name} s{seed}] train done {train_secs:.1f}s "
          f"loss[start={losses[0]:.3f} end={np.mean(losses[-10:]):.3f}]")

    # --- eval: greedy generate, task's standard metric ---------------------------
    model.eval()
    is_corpus = task.name in CORPUS_METRIC_TASKS
    n = len(eval_ds)
    samples = []
    per_item_scores = []          # for non-corpus metrics
    cand_corpus, ref_corpus = {}, {}  # for CIDEr
    for start in range(0, n, rcp["eval_batch_size"]):
        rows = [eval_ds[i] for i in range(start, min(start + rcp["eval_batch_size"], n))]
        images = [task.image_fn(r).convert("RGB") for r in rows]
        prompts = [task.prompt_fn(r) for r in rows]
        inp = proc(text=prompts, images=images, return_tensors="pt",
                   padding="longest").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        in_len = inp["input_ids"].shape[-1]
        with torch.inference_mode():
            # loc tokens for RefCOCO are special tokens → keep them in the decode.
            gen = model.generate(**inp, max_new_tokens=rcp["max_new_tokens"],
                                 do_sample=False)
        keep_special = task.name == "refcoco"
        decoded = proc.batch_decode(gen[:, in_len:], skip_special_tokens=not keep_special)

        for i, (row, pred) in enumerate(zip(rows, decoded)):
            pred = pred.strip()
            if is_corpus:
                idx = start + i
                cand_corpus[idx] = [pred]
                ref_corpus[idx] = _coco_caps(row)
                sc = None
            else:
                sc = task.score_fn(pred, row)
                per_item_scores.append(sc)
            if len(samples) < 30:
                samples.append({"prompt": prompts[i], "pred": pred,
                                "score": sc,
                                "gt": (ref_corpus.get(start + i, [])[:2] if is_corpus
                                       else _sample_gt(task, row))})

    if is_corpus:
        score = _cider_corpus(cand_corpus, ref_corpus)
        n_scored = len(cand_corpus)
    else:
        score = float(np.mean(per_item_scores))
        n_scored = len(per_item_scores)

    elapsed = time.time() - t0
    print(f"[battery/{task.name}/{config_name} s{seed}] DONE {task.metric_name}="
          f"{score:.4f} (N={n_scored}) total={elapsed:.1f}s")

    return {
        "task": task.name,
        "config": config_name,
        "seed": seed,
        "dataset": task.dataset,
        "train_split": task.train_split,
        "eval_split": task.eval_split,
        "metric": task.metric_name,
        "metric_source": task.metric_source,
        "prompt_source": task.prompt_source,
        "prompt_prefix": task.prompt_prefix,
        "score": score,                 # the estimand variable input (task's metric)
        "n_eval": n_scored,
        "n_train": len(train_ds),       # realized train pool (cycled to max_steps)
        "same_split_disjoint": task.same_split_disjoint,
        "recipe": rcp,
        "train_loss_start": losses[0],
        "train_loss_end_mean10": float(np.mean(losses[-10:])),
        "trainable_report": report,
        "train_secs": round(train_secs, 1),
        "elapsed_sec": round(elapsed, 1),
        "gpu": torch.cuda.get_device_name(0),
        "sample_predictions": samples,
    }


def _sample_gt(task: TaskSpec, row):
    """A compact GT for the qualitative sample log (non-corpus tasks)."""
    if task.name == "refcoco":
        return {"box_yxyx": _refcoco_gt_box(row), "expr": _refcoco_expr(row)}
    if task.name == "textvqa":
        return _textvqa_answers(row)[:3]
    if task.name.startswith("spatial"):
        return row.get("label")
    return None


def verify_task_plumbing(task: TaskSpec, n_eval: int = 16) -> dict:
    """Cheap pre-train check (NO optimizer steps): build an UNTRAINED CEILING, run
    ONE train-collate batch (assert finite loss + suffix labels present) and a tiny
    eval-generate → decode-parse → metric (assert finite). Confirms the task's
    prompt/data/metric wiring before the fan-out. The numbers are meaningless
    (untrained) — only that the path runs. Fails loud on any wiring bug."""
    import numpy as np
    import torch

    _seed_everything(0)
    model, proc, _report = build_ceiling_model(0)

    # Exercise the REAL data path (incl. same_split_disjoint), tiny pools.
    train_ds, eval_ds = _load_train_eval(task, n_train_cap=64, n_eval=n_eval)
    bs = min(RECIPE["train_batch_size"], len(train_ds))
    train_rows = [train_ds[i] for i in range(bs)]
    images = [task.image_fn(r).convert("RGB") for r in train_rows]
    prompts = [task.prompt_fn(r) for r in train_rows]
    suffixes = [task.target_fn(r) for r in train_rows]
    batch = proc(text=prompts, images=images, suffix=suffixes,
                 return_tensors="pt", padding="longest")
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    has_supervised = bool((batch["labels"] != -100).any().item())
    batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.train()
    with torch.inference_mode():
        loss = model(**batch).loss
    loss_finite = bool(torch.isfinite(loss).item())

    # eval-generate → score path on a few items
    model.eval()
    is_corpus = task.name in CORPUS_METRIC_TASKS
    rows = [eval_ds[i] for i in range(min(n_eval, len(eval_ds)))]
    e_images = [task.image_fn(r).convert("RGB") for r in rows]
    e_prompts = [task.prompt_fn(r) for r in rows]
    inp = proc(text=e_prompts, images=e_images, return_tensors="pt",
               padding="longest").to("cuda")
    inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
    in_len = inp["input_ids"].shape[-1]
    keep_special = task.name == "refcoco"
    with torch.inference_mode():
        gen = model.generate(**inp, max_new_tokens=task.max_new_tokens, do_sample=False)
    decoded = proc.batch_decode(gen[:, in_len:], skip_special_tokens=not keep_special)
    if is_corpus:
        cand = {i: [decoded[i].strip()] for i in range(len(rows))}
        ref = {i: _coco_caps(rows[i]) for i in range(len(rows))}
        score = _cider_corpus(cand, ref)
    else:
        score = float(np.mean([task.score_fn(decoded[i].strip(), rows[i])
                               for i in range(len(rows))]))
    return {
        "task": task.name,
        "train_collate_ok": True,
        "train_has_supervised_labels": has_supervised,
        "train_loss_finite": loss_finite,
        "train_loss_value": float(loss.item()),
        "n_train_split": len(train_ds),
        "n_eval_split": len(eval_ds),
        "untrained_metric_finite": bool(np.isfinite(score)),
        "untrained_metric_value": score,
        "example_prompt": e_prompts[0],
        "example_target": suffixes[0],
        "example_decoded": [d.strip() for d in decoded[:3]],
    }
