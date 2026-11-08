"""M-5 — sink-aware gradient compression (queue item 5).

The claim under test: the sink carries a large share of gradient ENERGY and produces almost no
MOTION, and naive top-k gradient compression transmits by magnitude — so it spends its bandwidth
on components that will be suppressed anyway. Excluding the sink from the transmitted set should
therefore buy convergence at the same bandwidth.

Three arms, at MATCHED BANDWIDTH (identical entry counts, so the comparison is about WHICH entries,
never how many):

    naive            top-k by |g|
    sink_excluded    top-k by |g| after removing the sink coordinates
    random_excluded  top-k by |g| after removing an equal number of RANDOM coordinates

The third arm is the one that makes this a test rather than a demonstration. Excluding anything
changes which entries are sent; without a matched random exclusion, "sink-excluded beats naive"
could just mean "excluding some coordinates beats excluding none".

ERROR FEEDBACK is included because it is standard in top-k compression and because leaving it out
would make the naive arm artificially bad: without it, untransmitted gradient is discarded rather
than accumulated, and the sink's suppressed-but-large components would be silently dropped by
every arm equally.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

ARMS = ("naive", "sink_excluded", "random_excluded")


def _flat_topk_mask(v, k: int):
    import numpy as np
    a = np.abs(np.asarray(v, dtype=np.float64)).ravel()
    assert 0 < k <= a.size, f"k={k} out of range for {a.size} entries"
    idx = np.argpartition(a, a.size - k)[a.size - k:]
    m = np.zeros(a.size, dtype=bool)
    m[idx] = True
    return m.reshape(np.asarray(v).shape)


def excluded_columns(arm: str, *, n_cols: int, sink_coords, seed: int):
    """Which INPUT coordinates this arm removes before selecting its top-k.

    The exclusion is per-COLUMN because the sink is a set of residual-stream coordinates, and a
    weight matrix's columns are exactly its residual-stream inputs. The random arm removes the
    SAME NUMBER of columns, drawn from the complement of the sink so the two exclusions are
    disjoint and the contrast is not diluted by overlap.
    """
    import numpy as np
    assert arm in ARMS, f"unknown arm {arm!r} (expected one of {ARMS})"
    s = np.unique(np.asarray(sink_coords, dtype=int))
    assert s.size == 0 or (s.min() >= 0 and s.max() < n_cols), "sink coordinate out of range"
    if arm == "naive":
        return np.array([], dtype=int)
    if arm == "sink_excluded":
        return s
    pool = np.setdiff1d(np.arange(n_cols), s)
    assert pool.size >= s.size, (
        f"cannot draw {s.size} random exclusions disjoint from a sink of the same size in "
        f"{n_cols} columns")
    return np.sort(np.random.default_rng(seed).choice(pool, size=s.size, replace=False))


def compress(g, k: int, *, arm: str, sink_coords, seed: int, residual=None) -> dict:
    """One compression step with error feedback.

    Returns the transmitted tensor, the updated residual, and the accounting a matched-bandwidth
    comparison needs. `k` is IDENTICAL across arms by construction of the caller; it is asserted
    here too, because a bandwidth that quietly differs between arms is the one confound that would
    make this whole comparison meaningless.
    """
    import numpy as np
    G = np.asarray(g, dtype=np.float64)
    assert G.ndim == 2, f"expected a weight-shaped gradient, got {G.shape}"
    R = np.zeros_like(G) if residual is None else np.asarray(residual, dtype=np.float64)
    assert R.shape == G.shape, f"residual shape {R.shape} != gradient shape {G.shape}"

    acc = G + R
    excl = excluded_columns(arm, n_cols=G.shape[1], sink_coords=sink_coords, seed=seed)
    cand = acc.copy()
    cand[:, excl] = 0.0
    n_avail = cand.size - excl.size * G.shape[0]
    assert k <= n_avail, (
        f"{arm}: k={k} exceeds the {n_avail} selectable entries after excluding {excl.size} "
        f"columns — the arms would not be at matched bandwidth")
    mask = _flat_topk_mask(cand, k)
    sent = np.where(mask, acc, 0.0)
    return {"arm": arm, "sent": sent, "residual": acc - sent, "k": int(k),
            "n_excluded_cols": int(excl.size),
            "sent_energy_fraction": float((sent ** 2).sum() / max((acc ** 2).sum(), 1e-300)),
            "residual_norm": float(np.linalg.norm(acc - sent))}


def matched_bandwidth_check(results) -> dict:
    """Every arm must have transmitted the same number of entries. This is the arm's ONE structural
    control and it is checked, not assumed."""
    ks = {r["arm"]: int(r["k"]) for r in results}
    assert len(set(ks.values())) == 1, f"bandwidth not matched across arms: {ks}"
    assert set(ks) == set(ARMS), f"missing arms: {sorted(set(ARMS) - set(ks))}"
    return {"k": next(iter(ks.values())), "arms": sorted(ks)}


def sink_energy_share(g, sink_coords) -> dict:
    """How much of a gradient's energy sits in the sink columns, against the flat expectation.

    This is the premise of the whole arm, so it is MEASURED per run rather than cited: if the share
    is not elevated on this model and this data, the arm's motivation does not hold here and the
    row says so.
    """
    import numpy as np
    G = np.asarray(g, dtype=np.float64)
    s = np.unique(np.asarray(sink_coords, dtype=int))
    tot = float((G ** 2).sum())
    assert tot > 0, "zero gradient"
    share = float((G[:, s] ** 2).sum() / tot)
    flat = s.size / G.shape[1]
    return {"sink_energy_share": share, "flat_expectation": flat,
            "ratio": share / flat if flat > 0 else float("nan"),
            "n_sink_cols": int(s.size), "n_cols": int(G.shape[1])}


def convergence_readout(loss_curves: dict, *, at_step: int) -> dict:
    """The primary readout: validation loss at a matched step, per arm.

    Matched STEP is correct here and matched validation loss is not — unlike the M-11 twins, all
    three arms run the identical optimizer on the identical data and differ only in which gradient
    entries survive the wire, so step count is the matched budget and loss is the outcome.
    """
    import numpy as np
    out = {}
    for arm, (steps, loss) in loss_curves.items():
        assert arm in ARMS, f"unknown arm {arm!r}"
        s = np.asarray(steps, dtype=np.float64)
        l = np.asarray(loss, dtype=np.float64)
        assert s.shape == l.shape and s.size >= 2, f"{arm}: need >=2 paired points"
        assert s.max() >= at_step, f"{arm}: curve stops at {s.max()}, before step {at_step}"
        out[arm] = float(np.interp(at_step, s, l))
    assert set(out) == set(ARMS), f"missing arms: {sorted(set(ARMS) - set(out))}"
    return {"at_step": int(at_step), "loss_by_arm": out,
            "sink_excluded_minus_naive": out["sink_excluded"] - out["naive"],
            "random_excluded_minus_naive": out["random_excluded"] - out["naive"]}
