"""conserved-core Tier 1 — PURE probe logic (no torch model, no GPU, no datasets download).

The reusable weight-free core of the "name what U_unt computes" experiment (spec
2026-06-19_conserved_core_prereg.md §13). Everything here is unit-tested on tiny planted tensors
(infra/aws_pg/tests/test_conserved_core.py) BEFORE any GPU spend — distinct from the runtime preflight.
FAIL LOUD: no try/except, no synthetic fallback.

Function-probes (pure numpy, no sklearn dependency):
  position_probe   ridge regression: predict normalized within-window token position from the q/k input
  induction_probe  regularized least-squares linear classifier: detect induction-applicable positions
  (sink-mass is computed directly from attention weights in the job; no probe here)

The ablation-probe paradigm (locked pre-freeze, journal 2026-06-19): the probe is TRAINED ONCE on the
no-ablation (baseline) forward's discovery-window tokens, then EVALUATED on each arm's ablated
activations over the SAME sealed confirmation-window tokens (paired). degradation(arm) =
metric(baseline) - metric(arm); contrast = degradation(U_unt) - degradation(matched-random); the
binding readout is the block-bootstrap (over confirmation windows) CI on that contrast, plus monotone
dose-response over k, plus the synthetic task. Construct validity: the probe scores high on real
structure and collapses to chance under label-shuffle.
"""
import hashlib

import numpy as np


# ===========================================================================
# Discovery / confirmation window split (salted sha1, 50/50). Spec §4/§7.
# ===========================================================================
def is_confirmation(window_text, salt="conserved-core-v1"):
    """Sealed confirmation half iff sha1(salt+text) is even. Discovery = the complement. The probe is
    TUNED on discovery; the headline contrast is the confirmation half (never tuned on)."""
    h = hashlib.sha1((salt + window_text).encode("utf-8")).hexdigest()
    return (int(h, 16) % 2) == 0


# ===========================================================================
# Pure-numpy linear probes (ridge regression + least-squares classifier).
# fp64 closed form; a tiny ridge λ for conditioning. X is (n, d) features, fitted with an intercept.
# ===========================================================================
def _augment(X):
    """Append a bias column of ones -> (n, d+1)."""
    return np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)


def ridge_fit(X, y, lam=1.0):
    """Closed-form ridge: w = (XᵀX + λI)⁻¹ Xᵀy with an unpenalized intercept. Returns w (d+1,)."""
    Xa = _augment(np.asarray(X, dtype=np.float64))
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    d1 = Xa.shape[1]
    R = lam * np.eye(d1, dtype=np.float64)
    R[-1, -1] = 0.0                                   # do not penalize the intercept
    w = np.linalg.solve(Xa.T @ Xa + R, Xa.T @ yv)
    return w


def linear_predict(w, X):
    return _augment(np.asarray(X, dtype=np.float64)) @ w


def r2_score(y, yhat):
    """Coefficient of determination wrt the truth's own mean. Can be negative (worse than the mean)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    assert ss_tot > 0, "position labels are constant — no variance to explain"
    return 1.0 - ss_res / ss_tot


def corr_score(y, yhat):
    """Pearson correlation of a position probe's predictions vs the truth — the position-decodability
    metric for the ablation contrast. BOUNDED in [-1,1] and shift/scale-invariant, so the distribution
    shift a subspace-ablation induces (which sends the unbounded R² to large negatives — smoke 2026-06-19)
    cannot crater it; it drops only when the ablation destroys the actual position SIGNAL. Returns 0 if a
    side is constant (no signal)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    yd, hd = y - y.mean(), yhat - yhat.mean()
    denom = np.sqrt(float(yd @ yd) * float(hd @ hd))
    if denom == 0.0:
        return 0.0
    return float((yd @ hd) / denom)


def lsclf_fit(X, y01, lam=1.0):
    """Regularized least-squares linear classifier: fit ridge on ±1 targets. Returns w (d+1,).
    A probe, not a calibrated classifier — decision score = linear_predict; threshold at 0."""
    y = np.asarray(y01, dtype=np.float64).reshape(-1)
    assert set(np.unique(y).tolist()) <= {0.0, 1.0}, "induction labels must be 0/1"
    ypm = 2.0 * y - 1.0                                # {0,1} -> {-1,+1}
    return ridge_fit(X, ypm, lam=lam)


def f1_at_zero(scores, y01):
    """Binary F1 of the positive class with the score thresholded at 0 (the ±1 LS decision boundary)."""
    y = np.asarray(y01, dtype=np.float64).reshape(-1)
    pred = (np.asarray(scores, dtype=np.float64).reshape(-1) > 0.0).astype(np.float64)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


# ===========================================================================
# Induction-applicability label. Position t is induction-applicable iff its PREVIOUS token has
# occurred earlier in the window — i.e. an induction head at t has a prior occurrence to copy from.
# (Uses the previous token's history, not the current token's identity, to reduce token-identity
# confounding; the label-shuffle control guards the rest.)
# ===========================================================================
def induction_labels(ids):
    """ids: 1-D int array of token ids for one window. Returns 0/1 array, same length.
    label[t] = 1 iff ids[t-1] appears in ids[:t-1]. label[0] = 0 (no previous token)."""
    ids = np.asarray(ids).reshape(-1)
    out = np.zeros(ids.shape[0], dtype=np.int64)
    seen = set()
    for t in range(ids.shape[0]):
        if t >= 1 and int(ids[t - 1]) in seen:
            out[t] = 1
        if t >= 1:
            seen.add(int(ids[t - 1]))
    return out


def position_labels(n_text):
    """Normalized within-window position in [0,1]: arange(n)/(n-1). n>=2 required (variance)."""
    assert n_text >= 2, "need >=2 text tokens for a position label"
    return np.arange(n_text, dtype=np.float64) / (n_text - 1)


# ===========================================================================
# Block bootstrap (over windows) of a probe-metric contrast. Each window contributes a token block;
# resampling whole windows preserves within-window correlation (paired across arms: every arm scored
# the SAME windows/tokens, only the activations differ). Spec §7.
# ===========================================================================
def probe_contrast_block_ci(window_blocks, metric_fn, key_treat, key_null, n_boot=1000, seed=1337):
    """window_blocks: list of dicts, one per confirmation window, each holding the per-token arrays the
    metric needs (e.g. {'y': labels, 'treat': preds_U_unt, 'null': preds_random, ...}). metric_fn(pool,
    key) -> scalar metric for one arm on a pooled token dict. The CONTRAST is
    [metric(null) - metric(treat)]  (= degradation(treat) - degradation(null), since both subtract the
    shared baseline). Positive contrast => the treat arm degrades the probe MORE than the null arm.
    Returns point + 95% CI + one-sided p(contrast <= 0). FAIL LOUD on empty input."""
    assert len(window_blocks) > 0, "no confirmation windows to bootstrap"
    keys = list(window_blocks[0].keys())

    def _pool(idx):
        return {kk: np.concatenate([window_blocks[i][kk] for i in idx]) for kk in keys}

    def _contrast(idx):
        pool = _pool(idx)
        return metric_fn(pool, key_null) - metric_fn(pool, key_treat)

    W = len(window_blocks)
    full = np.arange(W)
    point = _contrast(full)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = _contrast(rng.integers(0, W, W))
    return {"contrast": float(point), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "p_le0": float(np.mean(boots <= 0.0)), "n_windows": int(W)}


def r2_metric(pool, pred_key):
    """metric_fn for position-R²: pooled token dict carries 'y' (labels) + pred_key (arm predictions)."""
    return r2_score(pool["y"], pool[pred_key])


def f1_metric(pool, pred_key):
    """metric_fn for induction-F1: pooled token dict carries 'y' (0/1) + pred_key (arm scores)."""
    return f1_at_zero(pool[pred_key], pool["y"])


# ===========================================================================
# Synthetic probe corpora (NEW, deterministic, tiny vocab so the base sits at ceiling). Each returns a
# list of (prompt_text, gold_text) — the model's next-token argmax at the prompt end is scored == gold.
# The job tokenizes + forwards these under each ablation arm; degradation(arm) = acc(base) - acc(arm).
# H_pos and H_indu make OPPOSITE predictions: ablating a positional core collapses explicit-position;
# ablating an induction core collapses repeat-copy (spec §1 primary opposite-prediction readout).
# ===========================================================================
_LETTERS = list("abcdefghijklmnopqrstuvwxyz")


def copy_corpus(n_items, seq_len=6, seed=20260619):
    """repeat-3 copy (induction). A length-`seq_len` random letter sequence, then its first `seq_len-1`
    letters again; gold = the letter that followed at that point the first time (pure induction copy).
    Prompt: 'a b c d e f a b c d e' -> gold 'f'. Distinct letters per item so the copy target is
    unambiguous (no positional shortcut). Returns [(prompt, gold), ...]."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_items):
        seq = list(rng.choice(_LETTERS, size=seq_len, replace=False))
        prompt = " ".join(seq + seq[:-1])             # full seq, then all but the last
        gold = seq[-1]                                # induction copies the last (the token after seq[-2])
        items.append((prompt, gold))
    return items


def position_corpus(n_items, seq_len=6, seed=20260620):
    """explicit-position retrieval. A list with explicit 1-based position markers, then a query for one
    position; gold = the letter at that position. Prompt: '1 a 2 b 3 c 4 d 5 e ? 3' -> gold 'c'. Tiny
    vocab; tests whether the model resolves an EXPLICIT position index (the positional-addressing
    functional readout). Returns [(prompt, gold), ...]."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_items):
        seq = list(rng.choice(_LETTERS, size=seq_len, replace=False))
        q = int(rng.integers(1, seq_len + 1))         # 1-based queried position
        toks = []
        for i, ch in enumerate(seq, start=1):
            toks += [str(i), ch]
        prompt = " ".join(toks + ["?", str(q)])
        gold = seq[q - 1]
        items.append((prompt, gold))
    return items


def shuffle_labels(y, seed):
    """Permute labels (the §5 construct-validity shuffle). A valid probe collapses to chance/zero-R²."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    rng.shuffle(y)
    return y


# ===========================================================================
# Energy-matched null (the primary energy-confound control, added post-review). U_unt removes ~28% of
# the q/k-input activation energy at k=128 vs a random k-subspace's ~6% — so a bare random null is NOT
# energy-matched and "U_unt degrades the probe more" could be a pure variance effect. The energy-matched
# null is a k-dim subspace built from the ACTIVATION Gram's eigenbasis whose removed-energy fraction
# matches U_unt's. We name a function ONLY if U_unt beats THIS null.
# ===========================================================================
def energy_matched_subspace(V_desc, mu_desc, e_target, k, seed, tol=0.03, max_resample=80):
    """A k-dim RANDOM subspace whose removed-activation-energy fraction matches e_target — the primary
    energy-confound control. V_desc/mu_desc: eigvecs (columns) / eigenvalues of the activation Gram,
    DESCENDING by eigenvalue. Construction: a random k-subspace of span(top-m eigvecs), with m chosen so
    a Haar k-frame's expected energy (k/m)·Σμ[:m]/Σμ ≈ e_target; resample the Haar frame until the actual
    energy is within tol (else take the closest). Distinct per seed (so ematch-vs-ematch calibrates θ),
    energy-matched, and NOT aligned to any function. Returns (Q (n×k orthonormal), actual energy)."""
    mu = np.asarray(mu_desc, dtype=np.float64)
    n = mu.size
    assert k < n, (k, n)
    p = mu / mu.sum()
    cum = np.cumsum(p)
    # (k/m)·cum[m-1] decreases from cum[k-1] (top-k energy, >= e_target) toward k/n; grow m to reach e_target.
    m = k
    while m < n and (k / m) * cum[m - 1] > e_target:
        m += 1
    Vtop = np.asarray(V_desc, dtype=np.float64)[:, :m]
    mtop = mu[:m]
    rng = np.random.default_rng(seed)
    best_R, best_e, best_err = None, None, float("inf")
    for _ in range(max_resample):
        R, _ = np.linalg.qr(rng.standard_normal((m, k)))      # Haar m×k frame
        e = float((R ** 2 * mtop[:, None]).sum() / mu.sum())  # trace(Rᵀ diag(μ_top) R)/Σμ
        if abs(e - e_target) < best_err:
            best_R, best_e, best_err = R, e, abs(e - e_target)
        if best_err < tol:
            break
    Q = (Vtop @ best_R).astype(np.float64)
    return Q, best_e


def synth_contrast_bootstrap(base_items, treat_items, null_items, n_boot=1000, seed=1337):
    """Paired item bootstrap of the synthetic CONTRAST = degradation(treat) - degradation(null), where
    degradation(arm) = acc(base) - acc(arm). The shared base cancels => contrast = acc(null) - acc(treat).
    Positive => treat (U_unt) collapses the task MORE than the null. Same item indices applied to all
    three arms each draw (paired). Returns point + 95% CI + one-sided p(contrast <= 0)."""
    b = np.asarray(base_items, dtype=np.float64)
    t = np.asarray(treat_items, dtype=np.float64)
    u = np.asarray(null_items, dtype=np.float64)
    assert b.shape == t.shape == u.shape, (b.shape, t.shape, u.shape)
    N = b.shape[0]
    point = float(u.mean() - t.mean())                       # = (b.mean-t.mean) - (b.mean-u.mean)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        boots[i] = u[idx].mean() - t[idx].mean()
    return {"contrast": point, "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "p_le0": float(np.mean(boots <= 0.0))}


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values (§7 multiplicity), returned in input order. (Mirror of
    invariant_core_func._bh_adjust, kept here so the pure core is self-contained + unit-tested.)"""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out
