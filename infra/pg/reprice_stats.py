"""REPRICE statistics — pure, importable, weight-free (laptop-runnable).

The recovery-cost / concordance estimators + paired cluster bootstrap that turn
banked per-item correctness (MCQ vs GEN-cloze, paired by `qa_item_hash`) into the
two §8 headline readouts. NO model, NO datasets, NO GPU. Fail-loud; no try/except.

  * degenerate_cloze_filter — model-free drop of items whose cloze answer is
    degenerate (empty / duplicated choice text); set size returned, never hidden.
  * cloze_predict / cloze_correct — length-normalized (logprob / ntok**λ) argmax.
  * recovery_score / recovery_cost (C_x) — per-axis score and the min budget that
    reaches τ·ceiling.
  * concordance (J) — P(cloze recovered | MCQ recovered).
  * paired_cluster_bootstrap — resample ITEMS (the cluster) once, index every axis
    with the same draw (paired); generic stat_fn.
  * bootstrap_cost_ratio / bootstrap_concordance — the two headline CIs, effect
    reported MINUS the random-adapter/shuffle null (never vs zero).
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Degenerate-item filter (pre-registered, model-output-free).
# ---------------------------------------------------------------------------
def degenerate_cloze_filter(items) -> dict:
    """`items`: list of {'question','choices','answer'}. An item is degenerate for
    the cloze axis when its gold answer text is empty (after strip) or duplicates
    another choice text (content-scoring cannot disambiguate). Returns
    {kept, dropped_hashes, n_in, n_dropped}; `kept` preserves input order."""
    import hashlib

    def h(q):
        return hashlib.sha1(q[:64].encode()).hexdigest()[:12]

    kept, dropped = [], []
    for r in items:
        choices = [str(c).strip() for c in r["choices"]]
        gold = choices[int(r["answer"])]
        degenerate = (gold == "") or (choices.count(gold) > 1)
        (dropped if degenerate else kept).append(r)
    return {"kept": kept, "dropped_hashes": [h(r["question"]) for r in dropped],
            "n_in": len(items), "n_dropped": len(dropped)}


# ---------------------------------------------------------------------------
# Cloze prediction under a length-norm exponent λ (tuned on discovery only).
# ---------------------------------------------------------------------------
def cloze_predict(logprobs, ntoks, lam: float = 1.0):
    """argmax_k (logprobs[:,k] / ntoks[:,k]**λ). λ=0 → raw sum (length-biased),
    λ=1 → per-token mean. Returns (N,) int predictions."""
    import numpy as np
    lp = np.asarray(logprobs, dtype=float)
    nt = np.asarray(ntoks, dtype=float)
    assert lp.shape == nt.shape and lp.ndim == 2 and lp.shape[1] == 4, \
        f"logprobs/ntoks must be (N,4), got {lp.shape}/{nt.shape}"
    assert (nt >= 1).all(), "ntoks must be ≥ 1 (a continuation has ≥ 1 token)"
    return (lp / nt ** lam).argmax(axis=1)


def cloze_correct(logprobs, ntoks, correct_idx, lam: float = 1.0):
    """(N,) 0/1: cloze prediction (under λ) equals the gold choice index."""
    import numpy as np
    c = np.asarray(correct_idx, dtype=int)
    return (cloze_predict(logprobs, ntoks, lam) == c).astype(int)


# ---------------------------------------------------------------------------
# Per-axis recovery score and recovery cost C_x.
# ---------------------------------------------------------------------------
def recovery_score(per_item, mask=None) -> float:
    """Mean per-item correctness (the per-axis score S_x). `mask` (bool array)
    restricts to a split (discovery / confirmation)."""
    import numpy as np
    x = np.asarray(per_item, dtype=float)
    if mask is not None:
        x = x[np.asarray(mask, dtype=bool)]
    assert x.size > 0, "recovery_score over empty set"
    return float(x.mean())


def recovery_fraction(S_per_item, floor_per_item, ceiling_per_item) -> float:
    """(mean(S) − mean(floor)) / (mean(ceiling) − mean(floor)) — the τ-independent
    fraction of the deletable gap that is recovered. 1 = full recovery, 0 = none,
    <0 = below the deleted floor. Floor-aware (unlike recovery_cost, which is
    degenerate when the floor already exceeds τ·ceiling)."""
    import numpy as np
    S = np.asarray(S_per_item, float); F = np.asarray(floor_per_item, float)
    C = np.asarray(ceiling_per_item, float)
    denom = C.mean() - F.mean()
    assert denom != 0, "ceiling == floor: recovery fraction undefined"
    return float((S.mean() - F.mean()) / denom)


def bootstrap_recovery_fraction(S_per_item, floor_per_item, ceiling_per_item,
                                n: int = 10000, seed: int = 0) -> dict:
    """Paired item bootstrap of `recovery_fraction` (one resample indexes S, floor,
    ceiling together). Returns {point, lo, hi, n_valid}; a draw with ceiling≈floor
    (denom≤0) is dropped as nan."""
    import numpy as np
    S = np.asarray(S_per_item, float); F = np.asarray(floor_per_item, float)
    C = np.asarray(ceiling_per_item, float)
    N = S.shape[0]
    assert F.shape[0] == N == C.shape[0], "S/floor/ceiling must share item count"

    def f(idx):
        fm = F[idx].mean(); d = C[idx].mean() - fm
        return (S[idx].mean() - fm) / d if d > 0 else np.nan

    pt = f(np.arange(N))
    rng = np.random.default_rng(seed)
    draws = np.array([f(rng.integers(0, N, N)) for _ in range(n)])
    fin = draws[np.isfinite(draws)]
    return {"point": float(pt), "lo": float(np.quantile(fin, 0.025)),
            "hi": float(np.quantile(fin, 0.975)), "n_valid": int(fin.size)}


def recovery_cost(budgets, scores, tau: float, ceiling: float) -> dict:
    """C_x = min budget b with scores[b] ≥ τ·ceiling. `budgets` ascending;
    `scores[i]` = S_x at budget[i]. Returns {cost, reached, threshold}; cost=None
    (reached=False) if no rung clears the bar — NOT silently 0 or inf."""
    import numpy as np
    b = np.asarray(budgets, dtype=float)
    s = np.asarray(scores, dtype=float)
    assert b.shape == s.shape and b.ndim == 1, f"budgets/scores misaligned: {b.shape}/{s.shape}"
    assert (np.diff(b) > 0).all(), "budgets must be strictly ascending"
    thr = tau * ceiling
    hit = np.where(s >= thr)[0]
    if hit.size == 0:
        return {"cost": None, "reached": False, "threshold": thr}
    return {"cost": float(b[hit[0]]), "reached": True, "threshold": thr}


# ---------------------------------------------------------------------------
# Per-item recovery concordance J = P(cloze recovered | MCQ recovered).
# ---------------------------------------------------------------------------
def concordance(mcq_correct, cloze_correct_arr) -> dict:
    """J(b) = P(item correct on cloze | correct on MCQ), paired per item.
    Returns {J, n_cond} where n_cond = #(MCQ correct). J=None if n_cond=0."""
    import numpy as np
    m = np.asarray(mcq_correct, dtype=int)
    c = np.asarray(cloze_correct_arr, dtype=int)
    assert m.shape == c.shape, f"axes misaligned: {m.shape} vs {c.shape}"
    n_cond = int(m.sum())
    if n_cond == 0:
        return {"J": None, "n_cond": 0}
    return {"J": float((m & c).sum() / n_cond), "n_cond": n_cond}


# ---------------------------------------------------------------------------
# Paired cluster bootstrap (items = the cluster; one draw indexes every axis).
# ---------------------------------------------------------------------------
def paired_cluster_bootstrap(arrays, stat_fn, n: int = 10000, seed: int = 0,
                             ci: float = 0.95) -> dict:
    """Resample item indices with replacement (the SAME draw applied to every
    array in `arrays` → paired). `stat_fn(resampled_arrays)` → float (np.nan when
    undefined, e.g. an unreached cost). Returns {point, lo, hi, n_boot, n_valid,
    frac_undef}. Percentile CI over the finite draws."""
    import numpy as np
    arrays = [np.asarray(a) for a in arrays]
    N = arrays[0].shape[0]
    assert all(a.shape[0] == N for a in arrays), "arrays must share the item axis (N)"
    point = stat_fn(arrays)
    rng = np.random.default_rng(seed)
    draws = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        draws[i] = stat_fn([a[idx] for a in arrays])
    finite = draws[np.isfinite(draws)]
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo = float(np.quantile(finite, lo_q)) if finite.size else float("nan")
    hi = float(np.quantile(finite, hi_q)) if finite.size else float("nan")
    return {"point": float(point) if point == point else None,
            "lo": lo, "hi": hi, "n_boot": n, "n_valid": int(finite.size),
            "frac_undef": float((n - finite.size) / n)}


def bootstrap_cost_ratio(mcq_by_budget, cloze_by_budget, budgets, tau,
                         mcq_ceiling, cloze_ceiling, n: int = 10000,
                         seed: int = 0) -> dict:
    """CI for C_cloze / C_MCQ. `*_by_budget` are (N, B) per-item correctness at
    each budget. Per draw: resample items, per-budget means → scores → C_x per
    axis → ratio (np.nan if either axis never reaches τ). `frac_undef` reports how
    often the ratio is undefined (the H_head 'cloze never reaches τ' signal)."""
    import numpy as np
    M = np.asarray(mcq_by_budget, dtype=float)
    C = np.asarray(cloze_by_budget, dtype=float)
    b = np.asarray(budgets, dtype=float)
    assert M.shape == C.shape and M.shape[1] == b.shape[0], \
        f"shapes: mcq{M.shape} cloze{C.shape} budgets{b.shape}"

    def ratio(arrs):
        m, c = arrs
        cm = recovery_cost(b, m.mean(axis=0), tau, mcq_ceiling)
        cc = recovery_cost(b, c.mean(axis=0), tau, cloze_ceiling)
        if not cm["reached"] or not cc["reached"]:
            return float("nan")
        return cc["cost"] / cm["cost"]

    return paired_cluster_bootstrap([M, C], ratio, n=n, seed=seed)


def bootstrap_concordance(mcq_correct, cloze_correct_arr, null_mcq=None,
                          null_cloze=None, n: int = 10000, seed: int = 0) -> dict:
    """CI for J (or J_cap − J_null when a null arm's per-item arrays are given).
    All arrays paired by item; one resample indexes every arm."""
    import numpy as np
    arrs = [np.asarray(mcq_correct, dtype=int), np.asarray(cloze_correct_arr, dtype=int)]
    have_null = null_mcq is not None and null_cloze is not None
    if have_null:
        arrs += [np.asarray(null_mcq, dtype=int), np.asarray(null_cloze, dtype=int)]

    def _J(m, c):
        nc = int(m.sum())
        return (m & c).sum() / nc if nc else np.nan

    def stat(a):
        j = _J(a[0], a[1])
        return j - _J(a[2], a[3]) if have_null else j

    return paired_cluster_bootstrap(arrs, stat, n=n, seed=seed)
