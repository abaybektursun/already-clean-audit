"""M-C1 / M-C2 — the two Register-B assemblers (queue items C1, C2).

Neither arm measures anything new. Every component already exists as a measured quantity; what has
never existed is the PACKAGE and, more importantly, the OUT-OF-SAMPLE VALIDATION of the package.
So this module is deliberately thin on estimation and thick on validation discipline:

  * `lint_report` assembles a weight-diff report from banks + laws whose parameters are passed in,
    never hardcoded — every law coefficient is recomputed from its artifact at preflight (§2).
  * `leave_one_out` is the only thing that turns the report into a claim. It refits the law on
    N−1 fine-tunes and forecasts the held-out one, against the NO-MODEL baseline (the training
    fold's mean). §6's formula needs the forecast to beat that baseline by more than the bootstrap
    CI half-width, so the baseline is computed here rather than assumed to be beaten.
  * `collision_profile` predicts merge damage from highway collision, and `norm_product_baseline`
    is its named incumbent — computed in the same function call so the pair cannot drift apart.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from infra.pg import darwin_ceq as CEQ    # REUSED verbatim (§11 reuse ledger)


# ---------------------------------------------------------------------------
# M-C1 — the fine-tune linter.
# ---------------------------------------------------------------------------
def highway_profile(delta, U_highway) -> dict:
    """Energy fraction of a fine-tune's write inside the highway (the top-consensus subspace)."""
    from infra.pg import humboldt_core as H
    import numpy as np
    D = np.asarray(delta, dtype=np.float64)
    e = H.bank_energy_fraction(D, U_highway)
    k, n = int(U_highway.shape[1]), int(U_highway.shape[0])
    return {"highway_energy_fraction": float(e), "flat_expectation": k / n,
            "ratio": float(e) / (k / n), "frobenius": float(np.linalg.norm(D))}


def sink_leak(W_base, W_ft, U_core) -> dict:
    """How much of the write leaked into the core, and what the leak law says it costs.

    `darwin_ceq.leak_law_check` is the measured relation; it is reused rather than re-derived so
    the linter's forecast and the CEQ arms' forecast are literally the same function.
    """
    return CEQ.leak_law_check(W_base, W_ft, U_core)


def damage_forecast(motion: float, *, coefficient: float, exponent: float) -> dict:
    """damage = c * motion^p.

    Both parameters are ARGUMENTS. REFEREE measured them, M-A7 corrects the dose anchor they were
    read at, and §2 requires them to be recomputed from the artifact at preflight — so a default
    value here would be exactly the transcribed-number failure the provenance rule forbids.
    """
    assert motion >= 0, "motion must be non-negative"
    assert coefficient > 0 and exponent > 0, (
        "the damage law's coefficient and exponent are recomputed from REFEREE's metrics.json at "
        "preflight; there is no default")
    return {"motion": float(motion), "coefficient": float(coefficient),
            "exponent": float(exponent), "damage_forecast": float(coefficient * motion ** exponent)}


def recipe_class(delta, *, rank_tol: float = 1e-6) -> dict:
    """Classify a delta by its own numerical signature: effective rank and spectral concentration.

    This is the flag that catches the adapter/merge contamination §7 names (the k_proj lesson): a
    LoRA-derived delta is low-rank, a merge is a sum of low-rank pieces, a full fine-tune is not.
    The classification is reported with the numbers it was made from, so a reader can disagree with
    the cut without re-running anything.
    """
    import numpy as np
    from infra.pg import humboldt_core as H
    D = np.asarray(delta, dtype=np.float64)
    s = np.linalg.svd(D, compute_uv=False)
    tot = float((s ** 2).sum())
    assert tot > 0, "zero delta"
    numerical_rank = int((s > rank_tol * s[0]).sum())
    pr = H.participation_ratio(s ** 2)
    full = min(D.shape)
    return {"numerical_rank": numerical_rank, "full_rank": int(full),
            "rank_fraction": numerical_rank / full,
            "spectral_participation_ratio": float(pr),
            "is_low_rank": bool(numerical_rank < 0.5 * full),
            "top1_energy_fraction": float(s[0] ** 2 / tot)}


def nearest_checkpoint(delta, known: dict) -> dict:
    """The closest known fine-tune by delta cosine, with the runner-up, so a near-tie is visible."""
    import numpy as np
    D = np.asarray(delta, dtype=np.float64).ravel()
    nd = np.linalg.norm(D)
    assert nd > 0, "zero delta"
    sims = {}
    for name, other in known.items():
        O = np.asarray(other, dtype=np.float64).ravel()
        assert O.shape == D.shape, f"{name}: shape mismatch"
        sims[name] = float(D @ O / (nd * np.linalg.norm(O)))
    order = sorted(sims, key=sims.get, reverse=True)
    assert order, "no known checkpoints to compare against"
    return {"nearest": order[0], "cosine": sims[order[0]],
            "runner_up": order[1] if len(order) > 1 else None,
            "runner_up_cosine": sims[order[1]] if len(order) > 1 else None,
            "all_cosines": sims}


def lint_report(*, W_base, W_ft, U_highway, U_core, damage_coefficient: float,
                damage_exponent: float, known_deltas: dict) -> dict:
    """The C1 deliverable for ONE fine-tune: every component in one dict, each with its inputs."""
    import numpy as np
    D = np.asarray(W_ft, dtype=np.float64) - np.asarray(W_base, dtype=np.float64)
    motion = float(np.linalg.norm(D))
    return {"highway": highway_profile(D, U_highway),
            "leak": sink_leak(W_base, W_ft, U_core),
            "damage": damage_forecast(motion, coefficient=damage_coefficient,
                                      exponent=damage_exponent),
            "recipe": recipe_class(D),
            "nearest": nearest_checkpoint(D, known_deltas) if known_deltas else None,
            "motion_frobenius": motion}


def leave_one_out(features, targets, fit_predict) -> dict:
    """N-fold LOO with the NO-MODEL baseline in the same table.

    `fit_predict(train_x, train_y, test_x) -> yhat`. The baseline is the training fold's mean,
    which is what "no model" means for a forecast: the best you can do knowing only the population.

    Returns per-fold absolute errors for BOTH, so §6's bootstrap-CI comparison runs on the arrays
    rather than on two summary numbers.
    """
    import numpy as np
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    assert X.shape[0] == y.shape[0] and y.size >= 3, \
        f"LOO needs >=3 units, got {y.size}"
    err_model, err_base, preds = [], [], []
    for i in range(y.size):
        tr = np.array([j for j in range(y.size) if j != i])
        yhat = float(fit_predict(X[tr], y[tr], X[i]))
        assert np.isfinite(yhat), f"fold {i}: non-finite forecast"
        preds.append(yhat)
        err_model.append(abs(yhat - y[i]))
        err_base.append(abs(float(y[tr].mean()) - y[i]))
    return {"n": int(y.size), "predictions": preds,
            "abs_err_model": err_model, "abs_err_baseline": err_base,
            "mae_model": float(np.mean(err_model)),
            "mae_baseline": float(np.mean(err_base)),
            "factor_vs_baseline": float(np.mean(err_base) / max(np.mean(err_model), 1e-300))}


def within_factor(loo: dict, *, factor: float) -> dict:
    """§4 M-C1's frozen criterion: 'forecasts within a pre-set factor beat the no-model baseline'.

    `factor` is the pre-set multiple of the realized value the forecast must land inside; it is an
    argument so it lives in the job's frozen config and in the artifact.
    """
    import numpy as np
    assert factor > 1.0, "a forecast factor of <=1 cannot be met by any forecast"
    m = np.asarray(loo["abs_err_model"], dtype=np.float64)
    b = np.asarray(loo["abs_err_baseline"], dtype=np.float64)
    return {"factor": float(factor), "n": int(m.size),
            "model_better_fraction": float((m < b).mean()),
            "mae_model": loo["mae_model"], "mae_baseline": loo["mae_baseline"]}


# ---------------------------------------------------------------------------
# M-C2 — the merge pre-check.
# ---------------------------------------------------------------------------
def collision_profile(delta_a, delta_b, U_highway) -> dict:
    """How much two fine-tunes write in the SAME highway directions.

    The statistic is the cosine between the two deltas' highway-projected components, scaled by
    the geometric mean of their highway energy fractions: two deltas can be aligned in the highway
    while barely using it, and that pair should not be predicted to collide.

    Its named incumbent, the norm product, is computed in the same call so the two can never be
    reported from different runs.
    """
    import numpy as np
    from infra.pg import humboldt_core as H
    A = np.asarray(delta_a, dtype=np.float64)
    B = np.asarray(delta_b, dtype=np.float64)
    U = np.asarray(U_highway, dtype=np.float64)
    assert A.shape == B.shape, f"shape mismatch {A.shape} vs {B.shape}"
    assert U.shape[0] == A.shape[1], f"highway ambient {U.shape[0]} != input dim {A.shape[1]}"
    Pa, Pb = (A @ U) @ U.T, (B @ U) @ U.T
    na, nb = float(np.linalg.norm(Pa)), float(np.linalg.norm(Pb))
    assert na > 0 and nb > 0, "a delta has zero highway component"
    cos = float((Pa * Pb).sum() / (na * nb))
    ea = H.bank_energy_fraction(A, U)
    eb = H.bank_energy_fraction(B, U)
    return {"highway_cosine": cos, "energy_a": float(ea), "energy_b": float(eb),
            "collision": float(cos * np.sqrt(ea * eb)),
            "norm_product_baseline": float(np.linalg.norm(A) * np.linalg.norm(B))}


def merge_prediction_table(pairs: dict, U_highway) -> list:
    """One row per pair/triple: the collision prediction and its incumbent, ready to be joined
    against the MEASURED TIES-merge damage. Refuses fewer than the §4 minimum of 10 pairs."""
    assert len(pairs) >= 10, (
        f"{len(pairs)} pairs, §4 M-C2 requires >=10 pairs/triples of the 24 — a prediction "
        f"validated on fewer is not the validation the arm promised")
    rows = []
    for name, (da, db) in sorted(pairs.items()):
        p = collision_profile(da, db, U_highway)
        rows.append({"pair": name, **p})
    return rows


def rank_correlation(pred, realized) -> dict:
    """Spearman between predicted collision and realized merge damage, and the same for the
    incumbent. The REFUTER ('prediction no better than the norm-product baseline') is read on the
    difference, so both correlations come from one call."""
    from infra.pg.mercator_erosion import _spearman
    import numpy as np
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(realized, dtype=np.float64)
    assert p.shape == r.shape and p.size >= 10, f"need >=10 paired points, got {p.shape}"
    return {"spearman": _spearman(p, r), "n": int(p.size)}
