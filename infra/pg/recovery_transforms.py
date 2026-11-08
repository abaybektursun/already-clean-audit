# W14-R — deletion-vs-suppression recovery gate: the frozen transform set + the
# control-anchored decision math (LOCAL, pure-python; NO torch/GPU).
"""The science core for the recovery gate (prereg 2026-06-02_w14_recovery_gate_prereg.md).

Does the adapted model's lost free-standing MMLU capability return under a bounded, frozen set
of prompt/format/conjugate transforms applied at INFERENCE with NO weight change (suppression),
or does it stay at floor, recoverable only by a weight edit (deletion)? This module holds the two
non-model halves the prereg §9 contract specifies:

  1. the FROZEN transform set (§3) + the deterministic prompt renderer `apply_transform`, and
  2. the control-anchored readout math (§6): R_recover, the δ_rec noise band, the τ_rec
     suppression target, the paired max-statistic bootstrap, and the three-way verdict.

The W11 MMLU scorer (TEXT-ONLY forward, answer-letter argmax — `infra.pg.text_eval.mmlu_accuracy`;
see journal C2: W11-A is text-only, NOT blank-image) and the verified pre-translated multilingual
MMLU are REUSED externals wired by the orchestration job (`infra.aws_pg.jobs.w14r_recovery`).

Pure functions, plain JSON-able returns, no try/except (fail loud — a missing translation must
raise, never silently fall back to English). The §9 weight-free dry-run
(`infra/pg/tests/test_recovery_transforms.py`) pins every function below on a laptop, no model.

The transform set is FROZEN here and re-asserted at the §9 freeze tag. |T| = 18: IDENTITY plus
one varied axis at a time (no cross-product — that would be a search, not a pre-registration).
"""
from __future__ import annotations

import hashlib

from infra.pg.text_eval import _subject_header  # W11 subject header — single source of truth (parity)


# ── §3 F1 — format variants (FormatSpread, arXiv:2310.11324) ──────────────────────────────────
# (option-label style, answer cue). `canonical` is IDENTITY's surface and is pinned to the EXACT
# W11 `text_eval._q_block` format (`A.` labels + `Answer:` cue + the subject header) so IDENTITY
# reproduces the §0 anchor acc_after byte-for-byte (preflight gate (a)). The 7 `f1_*` rows are the
# §3 table verbatim — each differs from canonical in the label style, the cue, or both, and none
# coincides with the canonical `dot_upper`/`Answer:` pair (checked in the dry-run).
FORMATS = {
    "canonical":  ("dot_upper",    "Answer:"),
    "f1_colon":   ("colon_upper",  "Answer:"),
    "f1_bracket": ("bracket_upper", "Answer:"),   # was f1_numeric: digit answers ' 1' tokenize to 2
    "f1_lower":   ("paren_lower",  "answer:"),     #   tokens → unscoreable by the single-token-answer
    "f1_correct": ("paren_upper", "Correct option:"),
    "f1_arrow":   ("paren_upper", "=>"),
    "f1_the":     ("paren_upper", "The answer is"),
    "f1_bare":    ("bare_upper",  "Answer:"),
}

# ── §3 F2 — conjugate prompting (Kotha, arXiv:2309.10105) ─────────────────────────────────────
# Two reframing prefixes (pretraining-style text, no external dependency) ...
CONJ_PREFIX = {
    "f2_textbook": "The following is an excerpt from a reference textbook.",
    "f2_booklet":  "From a printed examination booklet, reproduced verbatim:",
}
# ... and five translations. The conj id maps to the language KEY read from `item['i18n'][lang]`
# (supplied by the orchestrator from openai/MMMLU — PROFESSIONAL human translation, NOT model-in-loop,
# which would inject a translator confound). Languages fr/es/de/ja/zh: `ru` was swapped to `ja`
# pre-freeze (ratified 2026-06-02) because no clean professional source covers Russian; Japanese keeps
# the maximal distance-from-English + non-Latin script the Kotha conjugation effect needs (journal C2).
# F2 varies ONLY the conjugation axis: the option-label surface stays canonical so the answer-extraction
# token (the letter the scorer reads) is identical to identity — isolating conjugation from the F1 axis.
CONJ_LANG = {"f2_fr": "fr", "f2_es": "es", "f2_de": "de", "f2_ja": "ja", "f2_zh": "zh"}

# ── §3 F3 — few-shot structure (5-shot default == IDENTITY) ───────────────────────────────────
# shot_id -> (n_shots, exemplar_mode). 'uniform' renders every exemplar in the transform's own
# label style; 'alternating' (the `varied` transform) flips exemplar style by index parity to
# probe few-shot format inconsistency, while the TARGET item stays canonical.
SHOTS = {
    "5_default": (5, "uniform"),
    "0shot":     (0, "uniform"),
    "1shot":     (1, "uniform"),
    "5_varied":  (5, "alternating"),
}


def transform_set():
    """The 18 frozen transforms (§3) as `(id, fmt_id, conj_id|None, shot_id)` tuples: IDENTITY +
    7 F1 + 7 F2 + 3 F3. Each transform moves exactly ONE axis off IDENTITY. Order is frozen; the
    dry-run asserts the count, uniqueness, and per-family tallies."""
    ts = [("identity", "canonical", None, "5_default")]
    for fid in ("f1_colon", "f1_bracket", "f1_lower", "f1_correct", "f1_arrow", "f1_the", "f1_bare"):
        ts.append((fid, fid, None, "5_default"))                 # vary FORMAT only
    for cid in ("f2_textbook", "f2_booklet", "f2_fr", "f2_es", "f2_de", "f2_ja", "f2_zh"):
        ts.append((cid, "canonical", cid, "5_default"))          # vary CONJUGATION only
    ts.append(("f3_0shot", "canonical", None, "0shot"))          # vary SHOTS only
    ts.append(("f3_1shot", "canonical", None, "1shot"))
    ts.append(("f3_5shot_varied", "canonical", None, "5_varied"))
    return ts


def _label(style: str, j: int) -> str:
    """Option-label decoration for choice index j (0..3) in a given style."""
    if style == "dot_upper":
        return f"{chr(65 + j)}."                                 # W11 canonical: 'A.'
    if style == "paren_upper":
        return f"({chr(65 + j)})"
    if style == "colon_upper":
        return f"{chr(65 + j)}:"
    if style == "bracket_upper":
        return f"[{chr(65 + j)}]"
    if style == "paren_lower":
        return f"({chr(97 + j)})"
    if style == "bare_upper":
        return f"{chr(65 + j)}"
    raise ValueError(f"unknown label style {style!r}")


def _answer_label(style: str, idx: int) -> str:
    """The bare answer token the model is expected to emit for choice `idx` under `style` — the
    gold token the scorer reads (decoration stripped). Letters for the alpha styles, digit for
    numeric, lowercase for paren_lower."""
    if style == "paren_lower":
        return chr(97 + idx)
    return chr(65 + idx)                  # dot/paren/colon/bracket/bare _upper — all letter answers


def answer_label(t, idx: int) -> str:
    """Public: the gold answer token for choice `idx` under transform `t` (the job's scorer key)."""
    _id, fmt_id, _conj, _shot = t
    return _answer_label(FORMATS[fmt_id][0], idx)


def answer_candidates(t) -> list:
    """The 4 answer SURFACE tokens the scorer argmaxes over for transform `t`, with the leading
    space the tokenizer sees (W11 reads ' A'..' D'). For numeric/lower formats these become
    ' 1'..' 4' / ' a'..' d' — the C5 parametrization so F1 surface changes stay scoreable while the
    forward/extraction are otherwise the reused W11 path."""
    style = FORMATS[t[1]][0]
    return [" " + _answer_label(style, i) for i in range(4)]


def _block(question: str, choices, style: str, cue: str, answer_idx) -> str:
    """One MMLU item block, byte-identical to W11 `text_eval._q_block` for the canonical
    (dot_upper, 'Answer:') style: `Q.strip()\\n<label> c0\\n...\\n<cue>` and, for an exemplar
    (answer_idx set), a trailing ` <gold>\\n\\n`. The scored item ends exactly at the cue."""
    assert len(choices) == 4, f"MMLU item must have 4 choices, got {len(choices)}"
    s = question.strip() + "\n"
    for i, c in enumerate(choices):
        s += f"{_label(style, i)} {c}\n"
    s += cue
    if answer_idx is not None:
        s += f" {_answer_label(style, answer_idx)}\n\n"
    return s


def apply_transform(item: dict, t, exemplars: list) -> str:
    """Deterministic few-shot prompt for `item` under transform `t`, using `exemplars` (same-subject)
    as the shot pool. item = {question, options[4], answer∈0..3, subject, i18n:{lang:{question,options}}}.

    Assembly is W11-exact for IDENTITY: `[reframe prefix] + subject_header + 5 exemplar blocks +
    item block` concatenated (each block carries its own trailing blank line). F1 sets the
    option-label style + cue (§3 table). F2 either prepends a reframing prefix (f2_textbook/booklet)
    or renders question+options from item['i18n'][lang] (the translations; KeyError fails loud — no
    silent English fallback); the cue + answer surface stay canonical so the answer-extraction token
    is unchanged (C5/§3 isolation). F3 sets the shot count and, for `5_varied`, the alternating
    exemplar style. The subject header stays English (W11 parity; the conjugation signal is in the
    item text). Alignment key (the job pairs across transforms by): transform id × content_hash(item)."""
    _id, fmt_id, conj_id, shot_id = t
    label_style, cue = FORMATS[fmt_id]
    n_shots, ex_mode = SHOTS[shot_id]

    prefix = None
    lang = None
    if conj_id is not None:
        if conj_id in CONJ_PREFIX:
            prefix = CONJ_PREFIX[conj_id]
        elif conj_id in CONJ_LANG:
            lang = CONJ_LANG[conj_id]
        else:
            raise ValueError(f"unknown conj id {conj_id!r}")

    assert len(exemplars) >= n_shots, f"{shot_id} needs {n_shots} exemplars, got {len(exemplars)}"

    def _qc(it):                                                 # (question, options), translated or not
        if lang is None:
            return it["question"], it["options"]
        loc = it["i18n"][lang]                                   # KeyError here == the §9 fail-loud
        return loc["question"], loc["options"]

    parts = []
    if prefix is not None:
        parts.append(prefix + "\n\n")
    parts.append(_subject_header(item["subject"]))
    for i in range(n_shots):
        ex_style = label_style
        if ex_mode == "alternating":
            ex_style = "dot_upper" if i % 2 == 0 else "paren_upper"
        q, c = _qc(exemplars[i])
        parts.append(_block(q, c, ex_style, cue, exemplars[i]["answer"]))
    q, c = _qc(item)
    parts.append(_block(q, c, label_style, cue, None))
    return "".join(parts)


def content_hash(item: dict) -> str:
    """Per-item pairing key, W11-IDENTICAL: `sha1(question[:64])[:12]`. Matches the frozen
    `legA_controls.json['item_hashes']` so the identity per-item arrays align directly and the
    pairing key is shared with W11 (§2 instrument identity). Uses the English question, so a
    translated rendering hashes to the SAME item."""
    return hashlib.sha1(item["question"][:64].encode()).hexdigest()[:12]


# ── §6 — control-anchored readout math ────────────────────────────────────────────────────────

def r_recover(acc: float, acc_base: float, acc_floor: float) -> float:
    """R_recover = (acc − floor) / (base − floor): normalize so floor → 0, base → 1 (§6). The
    denominator is the base-vs-floor headroom; assert it is positive (fail loud on a degenerate
    instrument). R may exceed 1 if a transform beats the untransformed base."""
    denom = acc_base - acc_floor
    assert denom > 0, f"degenerate denominator: base {acc_base} <= floor {acc_floor}"
    return (acc - acc_floor) / denom


def delta_rec(base_acc_by_transform: dict, base: float, floor: float) -> float:
    """δ_rec — the BASE model's recovery spread across the transform set (§6 noise band): the
    capability is fully present in the base, so any spread there is formatting/conjugation noise,
    not signal. δ_rec = max_t R_base(t) − min_t R_base(t). Adapted 'recovery' must clear it.
    Frozen at preflight, before the adapted-model unblind."""
    rs = [r_recover(a, base, floor) for a in base_acc_by_transform.values()]
    return max(rs) - min(rs)


def tau_rec(swap_restore_r: float, frac: float = 0.5) -> float:
    """τ_rec — the suppression target (§6), a frozen fraction of what the weight-swap restore
    brings back. `swap_restore_r` is the W11 decoder-reset restore expressed as an R_recover.
    The constant 0.5 is frozen in the prereg; preflight asserts τ_rec > δ_rec or the experiment
    is underpowered and is escalated, not run."""
    return frac * swap_restore_r


def max_stat_bootstrap(correct_by_transform: dict, base_correct, floor: float,
                       n_boot: int = 2000, seed: int = 1337) -> dict:
    """Paired (per-item) max-statistic bootstrap for max_t R_recover over the transform set (§6/§7).

    correct_by_transform: {transform_id: per-item 0/1 correctness of the ADAPTED model under that
    transform}, every array aligned by item index and to `base_correct` (the BASE model's per-item
    0/1 under identity, the denominator anchor). `floor` is acc_floor (chance).

    Per draw: resample item indices ONCE (paired across base + all transforms), recompute
    acc_base = base_correct[idx].mean() and acc(t) = correct[t][idx].mean() JOINTLY, form
    R(t) = (acc(t) − floor) / (acc_base − floor), then take max_t R(t). The CI is the percentile
    band of that max-statistic — honest under the max selection (NOT a naive per-t CI). The point
    estimate (max_transform / max_r) is computed on the full sample; per_transform_r is the full
    per-t vector for the §10 table.
    """
    import numpy as np

    ids = list(correct_by_transform.keys())
    mats = {k: np.asarray(v, dtype=float) for k, v in correct_by_transform.items()}
    bc = np.asarray(base_correct, dtype=float)
    n = bc.shape[0]
    for k, v in mats.items():
        assert v.shape == (n,), f"transform {k}: shape {v.shape} != ({n},) base-aligned"

    def _r_by_t(idx):
        denom = bc[idx].mean() - floor
        assert denom > 0, f"degenerate denominator on this sample: base {bc[idx].mean()} <= floor {floor}"
        return {k: (mats[k][idx].mean() - floor) / denom for k in ids}

    full = np.arange(n)
    point_r = _r_by_t(full)
    max_id = max(point_r, key=point_r.get)

    rng = np.random.default_rng(seed)
    boot_max = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_max[b] = max(_r_by_t(idx).values())

    return {
        "max_transform": max_id,
        "max_r": float(point_r[max_id]),
        "ci_lo": float(np.percentile(boot_max, 2.5)),
        "ci_hi": float(np.percentile(boot_max, 97.5)),
        "per_transform_r": {k: float(v) for k, v in point_r.items()},
        "n_items": int(n), "n_boot": int(n_boot), "seed": int(seed),
    }


def verdict(max_r_lo: float, max_r_hi: float, heldout_r_lo: float,
            swap_restore_r: float, base_spread: float, frac: float = 0.5) -> str:
    """The §6 three-way decision rule on the confirmation half.

      H_suppression — lower-CI of max_t R_recover ≥ τ_rec AND lower-CI of the HELD-OUT max_t
                      R_recover ≥ τ_rec (recovery generalizes, not format-overfit).
      H_deletion    — upper-CI of max_t R_recover ≤ δ_rec (base_spread): all recovery is within
                      the base's own formatting noise.
      H_partial     — anything between (reported descriptively with the per-t table).

    τ_rec = frac × swap_restore_r. BH-FDR over the per-transform family is applied by the caller
    (the job, reusing w5_verdict._bh_fdr); this rule reads the already-computed max-statistic CIs."""
    tau = tau_rec(swap_restore_r, frac)
    if max_r_lo >= tau and heldout_r_lo >= tau:
        return "H_suppression"
    if max_r_hi <= base_spread:
        return "H_deletion"
    return "H_partial"
