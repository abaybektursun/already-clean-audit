"""sink-order — PURE clamp logic (no torch model, no GPU, no datasets).

The reusable weight-free core of the sink-vs-position causal-ordering experiment
(spec studies/.../2026-06-24_sink_order_prereg.md §13). Builds per-(query-row, key-col) ADDITIVE
attention-logit biases (added to the eager 4D `attention_mask` so a positive bias on a key column
raises that column's post-softmax mass) to:
  (a) INDUCE the BOS sink            -> bos_bias       (+b on the BOS key col for text query rows)
  (b) divert the SAME mass elsewhere -> matched_mass_bias  (+b on ONE random visible non-sink text col / row)
  (c) SUPPRESS the sink              -> suppress_bias  (-b on the BOS key col for text query rows)
plus selection helpers (closest achieved-mass grid level). Unit-tested weight-free
(infra/aws_pg/tests/test_sink_order.py) BEFORE any GPU spend. FAIL LOUD: no try/except, no fallback.

GEOMETRY (PaliGemma blank-image forward, ONE window, batch=1). The full input_ids is
    [256 image tokens] [BOS] [text tokens ...] [trailing "\\n" ...]
with pos = index of the first TEXT token, bos_idx = pos-1 (the BOS / sink token), image cols = 0..bos_idx-1,
text cols = pos..pos+n_text-1, T = full length. The TEXT QUERY rows are pos..pos+n_text-1 (the rows whose
downstream residual carries position — the readout). Causal over text: query row t sees key cols 0..t; the
image+BOS prefix is bidirectional (every text query sees the whole prefix, so BOS is always visible).

Each builder returns (bias (T,T) float64, target_cols (T,) int64): bias is added to the additive mask;
target_cols[t] = the key column whose post-clamp attention mass we read for row t (-1 for non-text rows).
"""
import numpy as np


def _check(pos, n_text, bos_idx, T):
    assert isinstance(T, int) and T > 0, T
    assert bos_idx == pos - 1, (bos_idx, pos)
    assert 0 <= bos_idx < pos, (bos_idx, pos)
    assert n_text >= 2, n_text
    assert pos + n_text <= T, (pos, n_text, T)


def _text_rows(pos, n_text):
    return np.arange(pos, pos + n_text, dtype=np.int64)


def bos_bias(pos, n_text, bos_idx, T, b):
    """INDUCE the BOS sink: +b on the BOS key column (bos_idx) for every text query row. target_cols =
    bos_idx for text rows (so the read-back 'target mass' == the text->BOS sink mass)."""
    _check(pos, n_text, bos_idx, T)
    bias = np.zeros((T, T), dtype=np.float64)
    rows = _text_rows(pos, n_text)
    bias[rows, bos_idx] = float(b)
    target = np.full(T, -1, dtype=np.int64)
    target[rows] = bos_idx
    return bias, target


def matched_mass_bias(pos, n_text, bos_idx, T, b, seed):
    """MATCHED-ATTENTION-MASS NULL: +b on ONE random causally-visible TEXT column per text query row —
    for row t, a uniform draw c ~ U[pos, t] (a text col, never the BOS bos_idx=pos-1, never an image col
    0..pos-2). Diverts the SAME injected bias to a NON-sink token. target_cols[t] = the drawn c."""
    _check(pos, n_text, bos_idx, T)
    rng = np.random.default_rng(seed)
    bias = np.zeros((T, T), dtype=np.float64)
    target = np.full(T, -1, dtype=np.int64)
    for t in range(pos, pos + n_text):
        c = int(rng.integers(pos, t + 1))          # visible text col in [pos, t]
        assert pos <= c <= t and c != bos_idx and c >= pos, (c, pos, t, bos_idx)
        bias[t, c] = float(b)
        target[t] = c
    return bias, target


def suppress_bias(pos, n_text, bos_idx, T, b_supp):
    """SUPPRESS the sink: -|b_supp| on the BOS key column for every text query row (push attention OFF
    BOS). Used with U_unt-ablation in the secondary necessity arm. target_cols = bos_idx (read the
    residual BOS mass after suppression)."""
    _check(pos, n_text, bos_idx, T)
    bias = np.zeros((T, T), dtype=np.float64)
    rows = _text_rows(pos, n_text)
    bias[rows, bos_idx] = -abs(float(b_supp))
    target = np.full(T, -1, dtype=np.int64)
    target[rows] = bos_idx
    return bias, target


def closest_level(achieved, target):
    """Index of the grid level whose achieved (per-level) mass is closest to `target`. FAIL LOUD on
    empty input. Used to pick the headline level (sink BOS mass closest to the U_unt-ablation level,
    and the matched-null level whose diverted mass matches the headline sink mass)."""
    a = np.asarray(achieved, dtype=np.float64)
    assert a.size > 0, "no achieved-mass levels to select from"
    return int(np.argmin(np.abs(a - float(target))))


def softmax_mass_after_bias(logits, col, b):
    """Pure check helper (TEST ONLY): post-softmax mass on `col` after adding b to its logit. Validates
    that a positive additive bias RAISES that column's mass (the semantics the GPU clamp relies on)."""
    z = np.asarray(logits, dtype=np.float64).copy()
    z[col] += float(b)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return float(p[col])
