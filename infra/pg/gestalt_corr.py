"""GESTALT Arm-D correspondence — splat-atlas diffing across model pairs. Pure numpy fp64.

THE FROZEN RULE (journal FR-6; frozen pre-data, changing it is a re-freeze):

  cost(c_i^A, c_j^B) = ||mu_i - mu_j||^2 + Bures^2(Sigma~_i, Sigma~_j)

where Sigma~ = U diag(lam) U^T is the component's rank-r covariance part (the isotropic floor
excluded) and Bures^2(A, B) = tr A + tr B - 2 tr((A^1/2 B A^1/2)^1/2), computed EXACTLY in the
joint span orth([U_i lam_i^1/2, U_j lam_j^1/2]) (dim <= 2r; both operators live entirely inside
it, so the restriction loses nothing). Assignment: Hungarian on the K x K cost matrix.

LABELS (per A-component, mechanical):
  survived : matched cost <= tau_move
  moved    : tau_move < matched cost < tau_chance
  died     : matched cost >= tau_chance
  merged   : flag beside the label — >= 2 A-components share one unconstrained nearest-B at
             cost <= tau_move.
  tau_move   = p97.5 of the SPLIT-HALF self-match cost distribution (P4 band; zero width => the
               identity-band failure mode => Arm D VOID until rebuilt).
  tau_chance = p2.5 of the shuffled-assignment cost distribution (random pairing) of the SAME
               pair — the "indistinguishable from chance pairing" boundary.

FLOORS/CEILINGS for survival fractions (journal FR-6): ceiling = split-half self-survival;
floor = survival of (PG1 atlas vs an atlas fit on the target's moment-matched TWIN stream).

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

TAU_MOVE_PCTL = 97.5
TAU_CHANCE_PCTL = 2.5
N_SHUFFLE_DRAWS = 200
SHUFFLE_SEED = 20260804


def atlas_np(atlas_cpu):
    """Banked torch atlas -> the numpy correspondence view {mu (K,d), S (K,d,r), weight (K,)}
    with S = U * sqrt(lam) (so S S^T = the rank-r covariance part)."""
    import numpy as np
    mu = atlas_cpu["mu"].numpy().astype(np.float64)
    U = atlas_cpu["U"].numpy().astype(np.float64)
    lam = atlas_cpu["lam"].numpy().astype(np.float64)
    w = np.exp(atlas_cpu["log_pi"].numpy().astype(np.float64))
    assert U.ndim == 3 and mu.ndim == 2 and lam.shape == U.shape[:1] + U.shape[2:]
    S = U * np.sqrt(np.clip(lam, 0.0, None))[:, None, :]
    return {"mu": mu, "S": S, "weight": w}


def bures2_in_span(S_i, S_j):
    """Bures^2 between Sigma_i = S_i S_i^T and Sigma_j = S_j S_j^T, exact, via the joint span."""
    import numpy as np
    M = np.concatenate([S_i, S_j], axis=1)                     # (d, 2r)
    Q, _ = np.linalg.qr(M)
    Ai = Q.T @ S_i                                             # (m, r)
    Aj = Q.T @ S_j
    A = Ai @ Ai.T
    B = Aj @ Aj.T
    wa, Va = np.linalg.eigh(A)
    wa = np.clip(wa, 0.0, None)
    Ah = (Va * np.sqrt(wa)) @ Va.T                             # A^1/2
    C = Ah @ B @ Ah
    wc = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    b2 = float(np.trace(A) + np.trace(B) - 2.0 * np.sqrt(wc).sum())
    return max(b2, 0.0)


def cost_matrix(na, nb):
    """(K_A, K_B) frozen-rule costs between two `atlas_np` views."""
    import numpy as np
    Ka, Kb = na["mu"].shape[0], nb["mu"].shape[0]
    C = np.empty((Ka, Kb))
    for i in range(Ka):
        dm = ((nb["mu"] - na["mu"][i]) ** 2).sum(axis=1)       # (K_B,)
        for j in range(Kb):
            C[i, j] = dm[j] + bures2_in_span(na["S"][i], nb["S"][j])
    assert np.isfinite(C).all(), "non-finite correspondence cost"
    return C


def hungarian(C):
    """Hungarian assignment; returns (row_ind, col_ind, matched_costs row-ordered)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(C)
    order = np.argsort(ri)
    ri, ci = ri[order], ci[order]
    return ri, ci, C[ri, ci]


def shuffle_chance_costs(C, assign_cols=None, n_draws=N_SHUFFLE_DRAWS, seed=SHUFFLE_SEED):
    """Matched costs under RANDOM pairings (permutation assignments) — the chance distribution.

    `assign_cols` (the Hungarian assignment, row i -> col assign_cols[i]) entries are EXCLUDED
    from the pool: a random permutation reproduces the true pairing on ~1/K of its entries, and
    those accidental matches drag the p2.5 of "chance" into the matched band (defect D4, caught
    by the offline gate). Chance means "pairings other than the chosen best"."""
    import numpy as np
    rng = np.random.default_rng(seed)
    K = min(C.shape)
    ac = None if assign_cols is None else np.asarray(assign_cols)
    draws = []
    for _ in range(n_draws):
        perm = rng.permutation(C.shape[1])[:K]
        keep = np.ones(K, dtype=bool) if ac is None else (perm != ac[:K])
        draws.append(C[np.arange(K)[keep], perm[keep]])
    pool = np.concatenate(draws)
    assert pool.size >= 50, f"chance pool too small ({pool.size})"
    return pool


def selfhalf_band(matched_costs_selfhalf):
    """P4: tau_move from the split-half self-match costs. ZERO width => VOID (identity-band
    failure mode) — returned as a flag the caller MUST honor, never silently."""
    import numpy as np
    m = np.asarray(matched_costs_selfhalf, float)
    assert m.ndim == 1 and m.size >= 4, f"need >=4 self-match costs, got {m.shape}"
    width = float(m.max() - m.min())
    tau = float(np.percentile(m, TAU_MOVE_PCTL))
    return {"tau_move": tau, "band_width": width, "n": int(m.size),
            "zero_width": bool(width <= 0.0 or tau <= 0.0),
            "p50": float(np.percentile(m, 50.0))}


def label_pair(C, tau_move, tau_chance):
    """Assignment + per-A-component labels + fractions. `tau_chance` must exceed `tau_move`
    for the three bands to be ordered; if it does not, the pair is VOID-banded (returned as a
    flag — the null could not be separated from the noise band).

    FR-6 AMENDMENT (defect D5, caught by the offline gate BEFORE any real data): the assignment
    runs on min(C, tau_chance) — the RAW-C Hungarian optimum can sacrifice a cheap true match to
    save cost on far 'died' columns (dead components steal live matches; observed verbatim in
    the planted-diff event). Capping at tau_chance makes all beyond-chance pairings equally
    expensive, removing that incentive. LABELS are computed from the UNCAPPED costs."""
    import numpy as np
    ri, ci, _mc_capped = hungarian(np.minimum(C, tau_chance))
    mc = C[ri, ci]
    banded_ok = bool(tau_chance > tau_move)
    labels = []
    for c in mc:
        if c <= tau_move:
            labels.append("survived")
        elif c < tau_chance:
            labels.append("moved")
        else:
            labels.append("died")
    nn = C.argmin(axis=1)                                       # unconstrained nearest-B
    nn_cost = C[np.arange(C.shape[0]), nn]
    merged_targets = [j for j in set(nn.tolist())
                      if (nn == j).sum() >= 2 and (nn_cost[nn == j] <= tau_move).all()]
    merged_mask = [bool(nn[i] in merged_targets) for i in range(C.shape[0])]
    n = len(labels)
    return {"assign_rows": ri.tolist(), "assign_cols": ci.tolist(),
            "matched_costs": [float(x) for x in mc],
            "labels": labels, "merged_mask": merged_mask,
            "tau_move": float(tau_move), "tau_chance": float(tau_chance),
            "banded_ok": banded_ok,
            "void_banded": not banded_ok,
            "frac_survived": labels.count("survived") / n,
            "frac_moved": labels.count("moved") / n,
            "frac_died": labels.count("died") / n,
            "frac_merged": float(np.mean(merged_mask))}


def survival_norm(surv, floor, ceiling):
    """S_norm = (survival - floor)/(ceiling - floor); nan when the band is degenerate (the
    caller's VOID branch, never a silent 0)."""
    denom = ceiling - floor
    if denom <= 0:
        return float("nan")
    return float((surv - floor) / denom)


# ===========================================================================
# Synthetic planted-diff atlases for the offline gate (P4 + label recovery).
# ===========================================================================
def synth_atlas(K, d, r, seed, spread=10.0, lam_scale=2.0):
    """A random synthetic atlas_np view: means on a sphere of radius `spread`, random rank-r
    factors. Deterministic."""
    import numpy as np
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal((K, d))
    mu *= spread / np.linalg.norm(mu, axis=1, keepdims=True)
    S = np.empty((K, d, r))
    for c in range(K):
        Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
        S[c] = Q * (lam_scale * (0.5 + rng.random(r)))[None, :]
    w = np.full(K, 1.0 / K)
    return {"mu": mu, "S": S, "weight": w}


def perturb_atlas(na, move_idx, die_idx, jitter, move_dist, seed):
    """A planted diff of `na`: every component jittered by `jitter` (the noise floor), the
    `move_idx` components displaced by `move_dist`, the `die_idx` components REPLACED by fresh
    random ones. The ground-truth labels are returned beside the atlas."""
    import numpy as np
    rng = np.random.default_rng(seed)
    K, d, r = na["S"].shape
    mu = na["mu"] + jitter * rng.standard_normal((K, d))
    S = na["S"] + 0.02 * jitter * rng.standard_normal(na["S"].shape)
    truth = ["survived"] * K
    for i in move_idx:
        v = rng.standard_normal(d)
        mu[i] = na["mu"][i] + move_dist * v / np.linalg.norm(v)
        truth[i] = "moved"
    for i in die_idx:
        v = rng.standard_normal(d)
        mu[i] = 40.0 * v / np.linalg.norm(v)
        Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
        S[i] = Q * 3.0
        truth[i] = "died"
    return {"mu": mu, "S": S, "weight": na["weight"].copy()}, truth
