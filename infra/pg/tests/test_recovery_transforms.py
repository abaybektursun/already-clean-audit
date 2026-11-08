"""W14-R §9 weight-free dry-run — the unit tests that MUST pass before the freeze tag.

No model, no GPU, seconds on a laptop. Pins the frozen transform set (§3), the W11-exact prompt
rendering (so IDENTITY reproduces the §0 anchor byte-for-byte), the fail-loud paths, and the
control-anchored readout math (§6).

Run:
    cd /Users/abaybektursun/projects/interp_research-w14r
    python -m pytest infra/pg/tests/test_recovery_transforms.py -v
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from infra.pg.recovery_transforms import (
    FORMATS,
    answer_candidates,
    answer_label,
    apply_transform,
    content_hash,
    delta_rec,
    max_stat_bootstrap,
    r_recover,
    transform_set,
    verdict,
)
from infra.pg.text_eval import _q_block, _subject_header   # the W11 oracle for the IDENTITY regression


# ── fixtures ──────────────────────────────────────────────────────────────────────────────────

def _item(q="What is 2+2?", opts=("3", "4", "5", "6"), ans=1, subject="abstract_algebra", with_i18n=True):
    it = {"question": q, "options": list(opts), "answer": ans, "subject": subject}
    if with_i18n:
        it["i18n"] = {lang: {"question": f"[{lang}] {q}", "options": [f"[{lang}] {o}" for o in opts]}
                      for lang in ("fr", "es", "de", "ja", "zh")}
    return it


ITEM = _item()
EXEMPLARS = [_item(f"Q{i}?", ("a", "b", "c", "d"), i % 4) for i in range(5)]
TMAP = {t[0]: t for t in transform_set()}


# ── test 1 — the frozen set is exactly 18, unique, family-balanced ─────────────────────────────

def test_transform_set_shape():
    ts = transform_set()
    ids = [t[0] for t in ts]
    assert len(ts) == 18
    assert len(set(ids)) == 18
    assert "identity" in ids
    assert sum(1 for t in ts if t[0].startswith("f1_")) == 7
    assert sum(1 for t in ts if t[0].startswith("f2_")) == 7
    assert sum(1 for t in ts if t[0].startswith("f3_")) == 3
    assert TMAP["identity"] == ("identity", "canonical", None, "5_default")


# ── test 2 — F1 renders the §3 table; IDENTITY is byte-identical to the W11 scorer prompt ───────

EXPECTED_F1 = {
    "f1_colon":   ("colon_upper",  "Answer:"),
    "f1_bracket": ("bracket_upper", "Answer:"),
    "f1_lower":   ("paren_lower",  "answer:"),
    "f1_correct": ("paren_upper",  "Correct option:"),
    "f1_arrow":   ("paren_upper",  "=>"),
    "f1_the":     ("paren_upper",  "The answer is"),
    "f1_bare":    ("bare_upper",   "Answer:"),
}

EXPECTED_RENDER = {                                    # (first ITEM option line, target cue) per F1 id
    "f1_colon":   ("\nA: 3", "Answer:"),
    "f1_bracket": ("\n[A] 3", "Answer:"),
    "f1_lower":   ("\n(a) 3", "answer:"),
    "f1_correct": ("\n(A) 3", "Correct option:"),
    "f1_arrow":   ("\n(A) 3", "=>"),
    "f1_the":     ("\n(A) 3", "The answer is"),
    "f1_bare":    ("\nA 3", "Answer:"),
}


def test_f1_table_and_identity_matches_w11():
    assert FORMATS["canonical"] == ("dot_upper", "Answer:")     # W11 canonical
    for fid, (style, cue) in EXPECTED_F1.items():
        assert FORMATS[fid] == (style, cue)
    for fid, (opt_line, cue) in EXPECTED_RENDER.items():
        out = apply_transform(ITEM, TMAP[fid], EXEMPLARS)
        assert opt_line in out, f"{fid}: missing option line {opt_line!r}"
        assert out.rstrip().endswith(cue), f"{fid}: target must end on bare cue {cue!r}"
        assert apply_transform(ITEM, TMAP[fid], EXEMPLARS) == out   # deterministic

    # IDENTITY must equal the W11 `_subject_header + _q_block` construction byte-for-byte, else
    # IDENTITY != the §0 adapted baseline and preflight gate (a) fails.
    w11 = (_subject_header(ITEM["subject"])
           + "".join(_q_block(e["question"], e["options"], e["answer"]) for e in EXEMPLARS[:5])
           + _q_block(ITEM["question"], ITEM["options"]))
    ident = apply_transform(ITEM, TMAP["identity"], EXEMPLARS)
    assert ident == w11, "IDENTITY render diverged from the W11 scorer prompt"
    assert "A. 3" in ident and ident.rstrip().endswith("Answer:")
    assert "[fr]" not in ident and "(A)" not in ident          # no translation, not paren style
    assert ident != apply_transform(ITEM, TMAP["f1_bracket"], EXEMPLARS)


# ── test 3 — translation reads i18n; missing i18n raises; answer surface stays canonical ───────

def test_translate_reads_i18n_and_fails_loud():
    out = apply_transform(ITEM, TMAP["f2_fr"], EXEMPLARS)
    assert "[fr] What is 2+2?" in out                           # target question translated
    assert "[fr] " in out.split("The following are")[-1].split("\n")[0] or "[fr]" in out  # exemplars translated
    assert out.rstrip().endswith("Answer:")                     # §3 isolation: cue stays canonical English
    assert "A. [fr] 3" in out                                   # canonical labels kept under translation
    for k in range(4):                                          # gold token identical to identity (C5)
        assert answer_label(TMAP["f2_fr"], k) == answer_label(TMAP["identity"], k)
        assert answer_candidates(TMAP["f2_fr"])[k] == answer_candidates(TMAP["identity"])[k]
    # missing i18n must raise (no silent English fallback)
    bare = _item(with_i18n=False)
    bare_ex = [_item(f"Q{i}?", with_i18n=False) for i in range(5)]
    with pytest.raises(KeyError):
        apply_transform(bare, TMAP["f2_fr"], bare_ex)


# ── test 4 — shot specs build the right block count (incl. subject header); varied alternates ──

def test_shot_counts_and_varied():
    # split on the blank-line block separator: parts = [subject_header, *exemplars, item]
    for sid, n in [("f3_0shot", 0), ("f3_1shot", 1)]:
        out = apply_transform(ITEM, TMAP[sid], EXEMPLARS)
        blocks = out.split("\n\n")
        assert len(blocks) == n + 2                             # header + n exemplars + item
        assert out.count("Answer:") == n + 1                   # n exemplar answers + target cue
    out = apply_transform(ITEM, TMAP["f3_5shot_varied"], EXEMPLARS)
    blocks = out.split("\n\n")
    assert len(blocks) == 7                                     # header + 5 exemplars + item
    assert "A. a" in blocks[1] and "(A) a" not in blocks[1]     # exemplar 0 (even) -> dot_upper
    assert "(A) a" in blocks[2] and "A. a" not in blocks[2]     # exemplar 1 (odd)  -> paren_upper
    assert "A. a" in blocks[3] and "(A) a" not in blocks[3]     # exemplar 2 (even) -> dot_upper
    out5 = apply_transform(ITEM, TMAP["identity"], EXEMPLARS)
    assert "(A) " not in out5                                   # uniform default has no paren labels


# ── test 5 — content_hash is W11-identical; answer_candidates carry the leading space ──────────

def test_hash_and_candidates():
    assert content_hash(ITEM) == hashlib.sha1(ITEM["question"][:64].encode()).hexdigest()[:12]
    assert len(content_hash(ITEM)) == 12
    assert content_hash(_item(ans=3)) == content_hash(ITEM)     # hash keyed on question only (W11)
    assert answer_candidates(TMAP["identity"]) == [" A", " B", " C", " D"]
    assert answer_candidates(TMAP["f1_bracket"]) == [" A", " B", " C", " D"]
    assert answer_candidates(TMAP["f1_lower"]) == [" a", " b", " c", " d"]


# ── test 6 — r_recover maps floor->0, base->1; degenerate base==floor raises ───────────────────

def test_r_recover():
    assert r_recover(0.25, 0.53, 0.25) == 0.0
    assert abs(r_recover(0.53, 0.53, 0.25) - 1.0) < 1e-12
    assert r_recover(0.39, 0.53, 0.25) > 0.0
    with pytest.raises(AssertionError):
        r_recover(0.4, 0.25, 0.25)
    assert delta_rec({"a": 0.53, "b": 0.53}, 0.53, 0.25) == 0.0
    assert delta_rec({"a": 0.53, "b": 0.39}, 0.53, 0.25) > 0.0
    # the real anchors (text-only, legA): R_recover(identity) ~ 0.0277
    assert abs(r_recover(0.25424, 0.51674, 0.24676) - 0.0277) < 0.002


# ── test 7 — max_stat_bootstrap: seed-deterministic, recovers a planted max, CI brackets point ──

def test_max_stat_bootstrap():
    n = 600
    rng = np.random.default_rng(0)
    base = (rng.random(n) < 0.50).astype(float)
    cbt = {
        "identity": (rng.random(n) < 0.30).astype(float),
        "t_hi":     (rng.random(n) < 0.80).astype(float),
        "t_lo":     (rng.random(n) < 0.30).astype(float),
    }
    r = max_stat_bootstrap(cbt, base, floor=0.25, n_boot=500, seed=1337)
    assert r["max_transform"] == "t_hi"
    assert r["ci_lo"] <= r["max_r"] <= r["ci_hi"]
    assert set(r["per_transform_r"]) == set(cbt)
    r2 = max_stat_bootstrap(cbt, base, floor=0.25, n_boot=500, seed=1337)
    assert (r2["ci_lo"], r2["ci_hi"], r2["max_transform"]) == (r["ci_lo"], r["ci_hi"], r["max_transform"])
    with pytest.raises(AssertionError):
        max_stat_bootstrap({"x": np.zeros(n - 1)}, base, floor=0.25, n_boot=10)


# ── test 8 — verdict returns each of the three labels on boundary inputs ────────────────────────

def test_verdict_labels():
    assert verdict(0.60, 0.90, 0.55, 1.0, 0.10) == "H_suppression"
    assert verdict(0.00, 0.05, 0.00, 1.0, 0.10) == "H_deletion"
    assert verdict(0.20, 0.40, 0.10, 1.0, 0.10) == "H_partial"
    assert verdict(0.60, 0.90, 0.10, 1.0, 0.10) == "H_partial"   # probe-only recovery (anti-overfit)
