"""ERLANGEN pure core — every estimand, clustering rule, null band and gate, weight-free.

Spec:  studies/lineage/paligemma1-pi0/research/2026-08-03_erlangen_program_spec.md
Amend: studies/lineage/paligemma1-pi0/research/2026-08-03_erlangen_amendments.md

This module holds ONLY pure functions of banked arrays, so the whole §6 decision surface is
unit-testable on the control plane before a single GPU minute (`feedback_offline_gate_before_gpu`).
Every heavy import lives inside a function; numpy is the only dependency (clustering and ARI are
implemented here rather than pulled from sklearn/scipy so the offline gate has no optional-dep
surface).

House commitments carried in:
  * every floor has a CEILING and every null a SATURATION guard + VOID branch
    (`feedback_one_sided_thresholds`) — see `p1_verdict`, `algebra_verdict`, `metric_verdict`;
  * status flags are computed LAST from numbers already banked
    (`feedback_signals_that_cannot_fail`);
  * no try/except (CLAUDE.md): a shape/alignment error RAISES with a trace.
"""
from __future__ import annotations

# --- frozen constants (amendment A1/A2/A3; pre-set conventions, disclosed as such) ------------
ARI_PASS = 0.50           # P1 clustering agreement with the frozen label map
ARI_STABLE = 0.50         # P1 split-half stability over the readout axis
PERM_ALPHA = 0.05         # permutation-null significance for P1
N_PERM = 10_000
N_SPLIT = 200
CI_ALPHA = 0.05           # two-sided → 95% intervals; one-sided tail 0.025
ABSORB_RHO = 0.50         # Arm A dependency-graph rule: i absorbs j iff marginal drops below this
BITS_PER_PARAM = 16.0     # bf16 — the storage the lineage actually uses (Arm G description length)


# ===========================================================================
# Fingerprints (Arm T) — frozen preprocessing, then distances.
# ===========================================================================
def column_scale(F_ref):
    """Per-readout-dimension scale, computed ONCE on the reference unit and applied to every unit.

    Fingerprint columns are heterogeneous (NLL deltas on different tasks/transformations), so an
    unscaled distance is dominated by whichever readout happens to have the largest units. Fixing
    the scale on the reference unit (PG1) — rather than re-standardising per unit — is what makes
    cross-unit fingerprints comparable at all: a per-unit z-score would remove exactly the
    between-unit differences the transfer test is looking for."""
    import numpy as np
    R = np.asarray(F_ref, dtype=np.float64)
    assert R.ndim == 2 and R.shape[0] >= 2, f"reference fingerprint matrix {R.shape}"
    sd = R.std(axis=0, ddof=1)
    # A dead column (no component moves it) carries no information; scaling it up would amplify
    # pure noise. It is kept at scale 1 and reported, never silently dropped.
    dead = sd <= 1e-12
    sd = np.where(dead, 1.0, sd)
    return {"scale": sd, "n_dead_columns": int(dead.sum()), "dead_mask": dead}


def preprocess(F, scale):
    """Frozen preprocessing (amendment A1): column-scale → row-centre → row-unit-norm.

    After centring and unit-norming a row, squared Euclidean distance is 2(1−Pearson correlation),
    so Ward linkage on these vectors IS agglomeration in the correlation geometry the amendment
    names, with none of Ward's Euclidean assumptions violated.

    A row of exactly zero (a component the instrument cannot move at all) has no direction to
    normalise; it is returned as zeros and flagged by `fingerprint_noise_floor`, which is what
    routes it to the VOID branch instead of into a cluster."""
    import numpy as np
    X = np.asarray(F, dtype=np.float64) / np.asarray(scale, dtype=np.float64)[None, :]
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 1e-12)


def preprocess_raw(F, scale):
    """Column-scaled ONLY — magnitude retained. This is the disclosed A1 sensitivity: clustering
    here can succeed merely because components differ in how much damage they do."""
    import numpy as np
    return np.asarray(F, dtype=np.float64) / np.asarray(scale, dtype=np.float64)[None, :]


def fingerprint_noise_floor(F_a, F_b, scale):
    """VOID branch for P1: a component whose fingerprint does not exceed repeat-measurement noise.

    `F_a`, `F_b` are the SAME grid measured under two seeds. Signal = ‖(a+b)/2‖ in the scaled space,
    noise = ‖a−b‖/2. A component is `unmeasurable` when signal ≤ noise — the instrument cannot see
    it, which is a different fact from "it clustered wrongly" and must not be scored as one."""
    import numpy as np
    A = np.asarray(F_a, float) / np.asarray(scale, float)[None, :]
    B = np.asarray(F_b, float) / np.asarray(scale, float)[None, :]
    assert A.shape == B.shape, f"seed-paired fingerprints misaligned {A.shape} {B.shape}"
    sig = np.linalg.norm((A + B) / 2.0, axis=1)
    noi = np.linalg.norm(A - B, axis=1) / 2.0
    unmeasurable = sig <= noi
    return {"signal": sig.tolist(), "noise": noi.tolist(),
            "snr": (sig / np.maximum(noi, 1e-30)).tolist(),
            "unmeasurable": unmeasurable.tolist(),
            "n_unmeasurable": int(unmeasurable.sum()), "n": int(A.shape[0])}


# ===========================================================================
# Agglomerative clustering (Ward, Lance–Williams) + ARI. Implemented here so the
# offline gate depends on numpy alone.
# ===========================================================================
def ward_tree(X):
    """Ward-linkage merge sequence over rows of X. Returns the list of (i, j, size) merges in
    merge order, with cluster ids ≥ n denoting merged nodes (scipy's convention).

    Vectorised Lance–Williams: the surviving cluster reuses slot i, the absorbed one is masked out.
    O(n²) numpy work per merge — the split-half harness runs hundreds of clusterings, and a
    pure-Python nearest-pair scan made that the offline gate's bottleneck."""
    import numpy as np
    P = np.asarray(X, dtype=np.float64)
    n = P.shape[0]
    assert n >= 2, "need ≥2 points to cluster"
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)     # squared Euclidean (Ward's metric)
    np.fill_diagonal(D, np.inf)
    size = np.ones(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    label = np.arange(n)
    merges = []
    nxt = n
    for _ in range(n - 1):
        M = np.where(active[:, None] & active[None, :], D, np.inf)
        i, j = np.unravel_index(int(np.argmin(M)), M.shape)
        assert np.isfinite(M[i, j]), "no finite pair distance left — clustering is degenerate"
        newd = (((size[i] + size) * D[i] + (size[j] + size) * D[j] - size * D[i, j])
                / (size[i] + size[j] + size))
        D[i, :] = newd
        D[:, i] = newd
        D[i, i] = np.inf
        D[j, :] = np.inf
        D[:, j] = np.inf
        active[j] = False
        size[i] += size[j]
        merges.append((int(label[i]), int(label[j]), int(size[i])))
        label[i] = nxt
        nxt += 1
    return merges


def cut_tree(merges, n, k):
    """Flat k-cluster labelling from a merge sequence: undo the last k−1 merges."""
    assert 1 <= k <= n, f"k={k} out of range for n={n}"
    parent = {}
    kept = merges[: len(merges) - (k - 1)] if k > 1 else merges

    def find(x):
        while x in parent:
            x = parent[x]
        return x

    nxt = n
    for (i, j, _s) in kept:
        parent[i] = nxt
        parent[j] = nxt
        nxt += 1
    roots = {}
    lab = [0] * n
    for p in range(n):
        r = find(p)
        if r not in roots:
            roots[r] = len(roots)
        lab[p] = roots[r]
    return lab


def cluster(X, k):
    """k-cluster labels for rows of X (Ward on the preprocessed fingerprints)."""
    return cut_tree(ward_tree(X), len(X), k)


def adjusted_rand(a, b):
    """Adjusted Rand index between two labellings (Hubert & Arabie 1985)."""
    import numpy as np
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape and a.ndim == 1, f"label arrays misaligned {a.shape} {b.shape}"
    ua, ub = np.unique(a), np.unique(b)
    C = np.zeros((ua.size, ub.size), dtype=np.int64)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            C[i, j] = int(((a == x) & (b == y)).sum())
    def c2(v):
        v = np.asarray(v, dtype=np.float64)
        return (v * (v - 1) / 2).sum()
    sij, sa, sb = c2(C.ravel()), c2(C.sum(1)), c2(C.sum(0))
    n = a.size
    exp = sa * sb / (n * (n - 1) / 2)
    mx = (sa + sb) / 2
    if mx == exp:
        # Degenerate: both labellings put everything in one cluster (or all singletons). ARI is
        # undefined; 0.0 is the conservative reading and the caller's gate treats it as failure.
        return 0.0
    return float((sij - exp) / (mx - exp))


def ari_permutation_p(X, labels, k, n_perm=N_PERM, seed=0):
    """One-sided achieved significance for "the clustering agrees with the frozen labels".

    The CLUSTERING is held fixed and the LABELS are permuted (N2): under the null the label map
    carries no information about fingerprint geometry, so the observed ARI should sit inside the
    permuted distribution."""
    import numpy as np
    lab = cluster(X, k)
    obs = adjusted_rand(lab, labels)
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    draws = np.array([adjusted_rand(lab, rng.permutation(y)) for _ in range(n_perm)])
    return {"ari": obs, "perm_mean": float(draws.mean()), "perm_p95": float(np.percentile(draws, 95)),
            "p": float(((draws >= obs).sum() + 1) / (n_perm + 1)), "n_perm": int(n_perm),
            "k": int(k), "cluster_labels": [int(v) for v in lab]}


def split_half_stability(F, scale, k, n_split=N_SPLIT, seed=0):
    """Median ARI between clusterings built on disjoint halves of the READOUT axis.

    This is the "is the type structure a property of the component or of one readout" test: if the
    inventory is real it survives measuring it with half the instrument."""
    import numpy as np
    A = np.asarray(F, float)
    m = A.shape[1]
    assert m >= 4, f"need ≥4 readout dims to split-half (got {m})"
    rng = np.random.default_rng(seed)
    sc = np.asarray(scale, float)
    out = []
    for _ in range(n_split):
        p = rng.permutation(m)
        h1, h2 = p[: m // 2], p[m // 2:]
        l1 = cluster(preprocess(A[:, h1], sc[h1]), k)
        l2 = cluster(preprocess(A[:, h2], sc[h2]), k)
        out.append(adjusted_rand(l1, l2))
    v = np.asarray(out)
    return {"ari_median": float(np.median(v)), "ari_lo95": float(np.percentile(v, 2.5)),
            "ari_hi95": float(np.percentile(v, 97.5)), "n_split": int(n_split), "k": int(k)}


def select_k(F, scale, k_range, n_split=50, seed=0):
    """k-selection by stability (spec §2-T): the k whose split-half clustering is most reproducible.
    Reported with the whole curve — a flat curve means k is not identified, which is data."""
    curve = {int(k): split_half_stability(F, scale, k, n_split=n_split, seed=seed)["ari_median"]
             for k in k_range}
    best = max(curve, key=lambda k: curve[k])
    vals = sorted(curve.values(), reverse=True)
    return {"k_star": int(best), "curve": curve,
            "margin": float(vals[0] - vals[1]) if len(vals) > 1 else float("nan"),
            "identified": bool(len(vals) > 1 and (vals[0] - vals[1]) > 0.05),
            "note": "flat stability curve — k not identified" if len(vals) > 1
                    and (vals[0] - vals[1]) <= 0.05 else "k identified by stability margin"}


def p1_verdict(F_seed_a, F_seed_b, labels, scale, k=3, seed=0,
               n_perm=N_PERM, n_split=N_SPLIT):
    """The MDE gate, mechanically (amendment A1). PASS / FAIL / VOID, flags computed LAST."""
    import numpy as np
    F = (np.asarray(F_seed_a, float) + np.asarray(F_seed_b, float)) / 2.0
    floor = fingerprint_noise_floor(F_seed_a, F_seed_b, scale)
    X = preprocess(F, scale)
    perm = ari_permutation_p(X, labels, k, n_perm=n_perm, seed=seed)
    stab = split_half_stability(F, scale, k, n_split=n_split, seed=seed)
    raw = adjusted_rand(cluster(preprocess_raw(F, scale), k), labels)
    void = floor["n_unmeasurable"] > 0
    passes = bool(perm["ari"] >= ARI_PASS and perm["p"] < PERM_ALPHA
                  and stab["ari_median"] >= ARI_STABLE and not void)
    return {"verdict": "VOID" if void else ("PASS" if passes else "FAIL"),
            "ari_normalized": perm["ari"], "ari_raw_sensitivity": float(raw),
            "perm_p": perm["p"], "perm_p95": perm["perm_p95"],
            "stability_ari_median": stab["ari_median"], "stability_ci": [stab["ari_lo95"], stab["ari_hi95"]],
            "thresholds": {"ari_pass": ARI_PASS, "ari_stable": ARI_STABLE, "perm_alpha": PERM_ALPHA},
            "noise_floor": floor, "cluster_labels": perm["cluster_labels"],
            "failure_means": ("typing instrument VOID — components below the repeat-measurement "
                              "noise floor; not scored as a clustering failure" if void else
                              "typing instrument works and recovers the known types" if passes else
                              "known types NOT recovered → H_types-converge dead, program pivots "
                              "to D/M/B")}


def construct_validity_rho(nll_damage, gen_damage, n_boot=10_000, seed=0):
    """Amendment A5's gate: does the teacher-forced NLL readout rank component damage the way the
    generation battery does? Spearman ρ with an item-free component bootstrap.
    FAILURE = CI includes 0 → the readout measures something else; T re-scopes."""
    import numpy as np
    a = np.asarray(nll_damage, float)
    b = np.asarray(gen_damage, float)
    assert a.shape == b.shape and a.ndim == 1 and a.size >= 4, \
        f"construct-validity arrays {a.shape} {b.shape}"

    def _rho(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = np.linalg.norm(rx) * np.linalg.norm(ry)
        return float(rx @ ry / den) if den > 0 else 0.0

    rng = np.random.default_rng(seed)
    obs = _rho(a, b)
    d = np.array([_rho(a[i], b[i]) for i in
                  (rng.integers(0, a.size, a.size) for _ in range(n_boot))])
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"rho": obs, "lo95": lo, "hi95": hi, "n": int(a.size),
            "passes": bool(obs >= 0.5 and lo > 0),
            "note": "NLL readout tracks the generation battery" if (obs >= 0.5 and lo > 0)
                    else "CONSTRUCT-VALIDITY FAILURE — re-scope T to the generation battery"}


# ===========================================================================
# Arm A — intervention algebra (amendment A2).
# ===========================================================================
def commutator(b_ij, b_ji):
    """‖b(i∘j) − b(j∘i)‖₂ over behaviour vectors."""
    import numpy as np
    x = np.asarray(b_ij, float)
    y = np.asarray(b_ji, float)
    assert x.shape == y.shape and x.ndim == 1, f"behaviour vectors misaligned {x.shape} {y.shape}"
    return float(np.linalg.norm(x - y))


def absorption(b_ji, b_i, b_j, b_base):
    """How much of j's marginal effect survives once i is applied.

    ratio = ‖b(j∘i) − b(i)‖ / ‖b(j) − b(base)‖ — 0 means i fully absorbs j, 1 means j acts
    independently, >1 means i AMPLIFIES j (reported, not clipped: the ceiling direction matters)."""
    import numpy as np
    num = float(np.linalg.norm(np.asarray(b_ji, float) - np.asarray(b_i, float)))
    den = float(np.linalg.norm(np.asarray(b_j, float) - np.asarray(b_base, float)))
    if den <= 1e-12:
        return {"ratio": float("nan"), "num": num, "den": den,
                "note": "j has no marginal effect on the base — absorption undefined"}
    return {"ratio": num / den, "num": num, "den": den, "note": "ok"}


def dependency_graph(absorb, rho=ABSORB_RHO):
    """Frozen extraction rule over the absorption matrix.

      directed  i→j ("j depends on i")  iff  ratio(i,j) < rho and NOT ratio(j,i) < rho
      mutual    i—j                     iff  both directions absorb — recorded as undirected,
                                             never silently oriented (the orientation would be an
                                             artifact of which cell we happened to read first)
      undefined pairs (j has no marginal effect on the base) are counted, not dropped.
    """
    edges, mutual, undef = [], [], 0
    for (i, j), a in absorb.items():
        r = a["ratio"]
        if r != r:
            undef += 1
            continue
        rev = absorb.get((j, i), {}).get("ratio", float("nan"))
        rev_absorbs = bool(rev == rev and rev < rho)
        if r < rho and not rev_absorbs:
            edges.append({"from": i, "to": j, "ratio": r})
        elif r < rho and rev_absorbs and i < j:
            mutual.append({"a": i, "b": j, "ratio_ij": r, "ratio_ji": rev})
    return {"edges": edges, "mutual": mutual, "rho": rho,
            "n_edges": len(edges), "n_mutual": len(mutual), "n_undefined": undef}


def algebra_verdict(pos_commutators, null_exact, null_reharvest, flow_signs):
    """P2 as frozen (amendment A2), with the saturation guard that makes it able to fail.

    pos_commutators : commutator norms over layer-ORDERED pairs (ℓ<ℓ′)
    null_exact      : same-locus ORTHOGONAL random pairs, parameters harvested once (the floor)
    null_reharvest  : the same pairs WITH re-harvest (the ceiling / saturation guard)
    flow_signs      : +1 where the directed asymmetry carries the information-flow sign, −1 else
    """
    import numpy as np
    p = np.asarray(pos_commutators, float)
    n0 = np.asarray(null_exact, float)
    nr = np.asarray(null_reharvest, float)
    s = np.asarray(flow_signs, float)
    assert p.size and n0.size and nr.size, "algebra verdict needs all three arms measured"
    above_floor = bool(np.median(p) > np.percentile(n0, 97.5))
    saturated = bool(np.median(nr) >= np.median(p))
    sign_frac = float((s > 0).mean())
    ok = bool(above_floor and not saturated and sign_frac > 0.5)
    return {"median_positive": float(np.median(p)), "null_exact_p97_5": float(np.percentile(n0, 97.5)),
            "median_null_reharvest": float(np.median(nr)), "flow_sign_fraction": sign_frac,
            "above_floor": above_floor, "reharvest_saturated": saturated,
            "verdict": "VOID-saturated" if saturated else ("PASS" if ok else "FAIL"),
            "failure_means": ("re-harvest alone manufactures non-commutation — the positive control "
                              "could not have failed; algebra VOID" if saturated else
                              "architecture order is visible in the commutators" if ok else
                              "no order structure above the exact-commutation floor → algebra VOID")}


# ===========================================================================
# Arm M — metric axioms (amendment A3).
# ===========================================================================
def pooled_seed_sd(replicates):
    """Pooled within-pair seed SD from the replicated pairs (amendment A3b).

    `replicates`: {pair_key: [cost_seed1, cost_seed2, ...]}. Returns None when fewer than 2 pairs
    carry ≥2 seeds — which VOIDs the axiom tests rather than letting them run on the item term
    alone (that would be anti-conservative in the direction of killing H_metric)."""
    import numpy as np
    usable = {k: np.asarray(v, float) for k, v in replicates.items() if len(v) >= 2}
    if len(usable) < 2:
        return {"sd": None, "n_pairs": len(usable),
                "note": "VOID — fewer than 2 pairs with ≥2 seeds; axiom tests cannot run"}
    ss, df = 0.0, 0
    for v in usable.values():
        ss += float(((v - v.mean()) ** 2).sum())
        df += v.size - 1
    return {"sd": float((ss / df) ** 0.5), "n_pairs": len(usable), "df": int(df), "note": "ok"}


def cell_sigma(item_sd, seed_sd):
    """Total per-cell SD = item ⊕ seed, in quadrature (amendment A3b)."""
    return float((float(item_sd) ** 2 + float(seed_sd) ** 2) ** 0.5)


def symmetry_test(D, sigma, tasks, z=1.96):
    """|d(A→B) − d(B→A)| per unordered pair vs the (item⊕seed) noise band.

    A pair is a CONFIRMED asymmetry iff the |difference| lower confidence bound clears 0. The test
    statistic is the count of confirmed asymmetries; under the null (true symmetry) each pair
    confirms with probability ≤ 0.025, giving an exact binomial p."""
    import numpy as np
    n = len(tasks)
    pairs, conf = [], 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = D[i][j], D[j][i]
            if a != a or b != b:
                continue
            se = (sigma[i][j] ** 2 + sigma[j][i] ** 2) ** 0.5
            diff = abs(a - b)
            lo = diff - z * se
            idx = diff / max((abs(a) + abs(b)) / 2, 1e-12)
            conf += int(lo > 0)
            pairs.append({"a": tasks[i], "b": tasks[j], "d_ab": float(a), "d_ba": float(b),
                          "abs_diff": float(diff), "lo95": float(lo),
                          "symmetry_index": float(idx), "confirmed": bool(lo > 0)})
    p = binom_tail_p(conf, len(pairs), CI_ALPHA / 2)
    return {"pairs": pairs, "n_pairs": len(pairs), "n_confirmed": conf, "p": p,
            "mean_symmetry_index": float(np.mean([q["symmetry_index"] for q in pairs])) if pairs else float("nan")}


def binom_tail_p(k, n, p0):
    """P(X ≥ k) for X~Binomial(n, p0) — exact, no scipy. Descriptive companion to the bootstrap
    test: it treats triples as independent, which they are NOT (they share edges), so it is
    reported but never the adjudicator."""
    from math import comb
    if n == 0:
        return float("nan")
    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


def metric_closure(D):
    """All-pairs shortest path (Floyd–Warshall) — the LARGEST triangle-satisfying matrix ≤ D, i.e.
    the projection of D onto the triangle-inequality null from below. This is the null the
    bootstrap resamples around: the least-favourable configuration consistent with 'the axiom
    holds' is the one that just barely holds."""
    import numpy as np
    C = np.array(D, dtype=np.float64, copy=True)
    n = C.shape[0]
    np.fill_diagonal(C, 0.0)
    for k in range(n):
        C = np.minimum(C, C[:, [k]] + C[[k], :])
    return C


def axiom_bootstrap(D, sigma, kind, n_boot=N_PERM, seed=0, delta_min=0.0, censored_mask=None):
    """The spec's "violation magnitude vs bootstrap null", with an effect-size floor.

    kind='triangle' : statistic = Σ over EVALUABLE triples of max(0, excess − delta_min);
                      null = the metric closure of D (the axiom holding as tightly as possible),
                      re-noised cell-wise with the measured (item⊕seed) σ.
    kind='symmetry' : statistic = Σ over pairs of max(0, |d_ij − d_ji| − delta_min);
                      null = the symmetrised matrix, re-noised the same way.

    Resampling the EDGES (not the triples) is what makes this valid where a per-triple binomial is
    not: triples share edges, so their violations are dependent, and the edge-level null carries
    that dependence for free.

    `delta_min` is the ceiling-side guard (`feedback_one_sided_thresholds`): with small σ, an
    arbitrarily tiny violation would otherwise be 'significant' and read as non-geometric. A
    violation must ALSO exceed delta_min to enter the statistic.
    """
    import numpy as np
    A = np.asarray(D, float)
    S = np.asarray(sigma, float)
    n = A.shape[0]
    assert A.shape == S.shape == (n, n), f"axiom bootstrap shapes {A.shape} {S.shape}"
    mask = (np.ones((n, n, n), dtype=bool) if censored_mask is None
            else np.asarray(censored_mask, dtype=bool))

    ii = np.arange(n)
    distinct = ((ii[:, None, None] != ii[None, :, None]) & (ii[:, None, None] != ii[None, None, :])
                & (ii[None, :, None] != ii[None, None, :]))

    def _stat_tri(M):
        # ex[a,b,c] = d(a→c) − d(a→b) − d(b→c)
        ex = M[:, None, :] - M[:, :, None] - M[None, :, :]
        return float(np.maximum(ex - delta_min, 0.0)[distinct & mask].sum())

    def _stat_sym(M):
        d = np.abs(M - M.T)
        iu = np.triu_indices(n, 1)
        return float(np.maximum(d[iu] - delta_min, 0.0).sum())

    stat_fn = _stat_tri if kind == "triangle" else _stat_sym
    assert kind in ("triangle", "symmetry"), kind
    null_M = metric_closure(A) if kind == "triangle" else (A + A.T) / 2.0
    obs = stat_fn(A)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        M = null_M + rng.standard_normal((n, n)) * S
        np.fill_diagonal(M, 0.0)
        draws[i] = stat_fn(M)
    p95 = float(np.percentile(draws, 95))
    # POWER GATE (the guard that makes "no violations" mean something). Plant a violation of
    # magnitude v on a representative cell and bisect for the smallest v the test would call at
    # 80% power. If that v exceeds the cost scale itself, "the axiom holds" is unfalsifiable here
    # and the test is UNDERPOWERED, not passing.
    scale = float(np.median(A[~np.eye(n, dtype=bool)]))
    cell = np.unravel_index(int(np.argmax(np.where(np.eye(n, dtype=bool), -np.inf, S))), S.shape)

    def _power(v, n_p=400):
        planted = null_M.copy()
        planted[cell] = planted[cell] + v
        hits = 0
        for _ in range(n_p):
            M = planted + rng.standard_normal((n, n)) * S
            np.fill_diagonal(M, 0.0)
            hits += int(stat_fn(M) > p95)
        return hits / n_p

    lo_v, hi_v = 0.0, max(4.0 * scale, 1e-6)
    for _ in range(12):
        mid = (lo_v + hi_v) / 2
        if _power(mid) >= 0.8:
            hi_v = mid
        else:
            lo_v = mid
    v_star = hi_v
    underpowered = bool(v_star > scale)
    return {"stat": obs, "p": float(((draws >= obs).sum() + 1) / (n_boot + 1)),
            "null_mean": float(draws.mean()), "null_p95": p95,
            "kind": kind, "delta_min": float(delta_min), "n_boot": int(n_boot),
            "cost_scale": scale, "min_detectable_violation": float(v_star),
            "min_detectable_over_scale": float(v_star / max(scale, 1e-30)),
            "underpowered": underpowered,
            "note": ("UNDERPOWERED — the smallest detectable violation exceeds the cost scale; "
                     "'the axiom holds' could not have failed here" if underpowered else
                     "power OK: a violation of one cost-scale would be detected")}


def triangle_test(D, sigma, tasks, censored=None, z=1.96):
    """Triangle violations d(A→C) > d(A→B) + d(B→C) with censoring handled (amendment A3c).

    A triple is EVALUABLE iff its verdict is identical whether censored cells take their measured
    lower bound or +∞; non-evaluable triples are counted separately and enter no statistic. A
    violation is CONFIRMED iff lower95(excess) > 0. The statistic is the confirmed count with the
    exact binomial p — and, for A3d, every violating triple banks its realised saving."""
    import numpy as np
    censored = censored or {}
    n = len(tasks)
    tri, conf, nonev = [], 0, 0
    evaluable = np.zeros((n, n, n), dtype=bool)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if len({a, b, c}) < 3:
                    continue
                dac, dab, dbc = D[a][c], D[a][b], D[b][c]
                if any(v != v for v in (dac, dab, dbc)):
                    nonev += 1
                    continue
                cen_ac = bool(censored.get((tasks[a], tasks[c]), False))
                cen_leg = bool(censored.get((tasks[a], tasks[b]), False) or
                               censored.get((tasks[b], tasks[c]), False))
                excess = dac - (dab + dbc)
                # censoring pushes the CENSORED cell upward only; a triple is evaluable when that
                # direction cannot flip the sign of the verdict.
                if (cen_leg and excess > 0) or (cen_ac and excess <= 0):
                    nonev += 1
                    continue
                se = (sigma[a][c] ** 2 + sigma[a][b] ** 2 + sigma[b][c] ** 2) ** 0.5
                lo = excess - z * se
                ok = bool(lo > 0)
                conf += int(ok)
                evaluable[a, b, c] = True
                tri.append({"a": tasks[a], "b": tasks[b], "c": tasks[c],
                            "d_ac": float(dac), "d_ab": float(dab), "d_bc": float(dbc),
                            "excess": float(excess), "lo95": float(lo), "confirmed": bool(ok),
                            # A3d: a confirmed violation IS a measured intermediate-task shortcut
                            "shortcut_saving": float(excess) if ok else 0.0})
    p = binom_tail_p(conf, len(tri), CI_ALPHA / 2)
    return {"triples": tri, "n_evaluable": len(tri), "n_non_evaluable": nonev,
            "n_confirmed": conf, "p": p, "evaluable_mask": evaluable.tolist(),
            "best_shortcut": max(tri, key=lambda t: t["shortcut_saving"]) if tri else None}


def metric_verdict(sym, tri, sym_log, tri_log, seed_sd_note, sym_desc=None, tri_desc=None):
    """The §6 verdict from the BOOTSTRAP tests, in the PRIMARY gauge (no rule swap), with the
    secondary gauge attached as the pre-registered scope (amendment A3a).

    Outcome space, all four first-class:
      geometric / quasi-metric / non-geometric  — the spec's three
      VOID                                      — seed term missing (A3b) or the axiom test is
                                                  underpowered (a 'geometric' verdict that could
                                                  not have failed is not a verdict)
    """
    if seed_sd_note != "ok":
        return {"verdict": "VOID", "reason": "seed-variance term unavailable (A3b)",
                "note": seed_sd_note}
    sym_bad = bool(sym["p"] < CI_ALPHA)
    tri_bad = bool(tri["p"] < CI_ALPHA)
    if tri_bad:
        v = "non-geometric"
    elif sym_bad:
        v = "quasi-metric"
    else:
        v = "geometric"
    # a NEGATIVE (axiom-holding) reading is only reportable if the test had the power to break it
    void_power = bool((not tri_bad and tri["underpowered"]) or (not sym_bad and sym["underpowered"]))
    gauge_agrees = bool((tri_log["p"] < CI_ALPHA) == tri_bad and (sym_log["p"] < CI_ALPHA) == sym_bad)
    out = {"verdict": "VOID-underpowered" if void_power else v,
           "verdict_before_power_gate": v,
           "symmetry_p": sym["p"], "triangle_p": tri["p"],
           "symmetry_stat": sym["stat"], "triangle_stat": tri["stat"],
           "symmetry_p_loggauge": sym_log["p"], "triangle_p_loggauge": tri_log["p"],
           "gauge_stable": gauge_agrees,
           "min_detectable_violation_over_scale": tri["min_detectable_over_scale"],
           "min_detectable_asymmetry_over_scale": sym["min_detectable_over_scale"],
           "scope": ("verdict in the AUC gauge; " +
                     ("the log gauge agrees" if gauge_agrees else
                      "the log gauge DISAGREES — the verdict is gauge-dependent and is reported "
                      "as such"))}
    if sym_desc is not None:
        out["n_confirmed_asymmetries"] = sym_desc["n_confirmed"]
        out["binomial_p_symmetry_descriptive"] = sym_desc["p"]
    if tri_desc is not None:
        out["n_confirmed_violations"] = tri_desc["n_confirmed"]
        out["n_non_evaluable_triples"] = tri_desc["n_non_evaluable"]
        out["binomial_p_triangle_descriptive"] = tri_desc["p"]
        out["best_shortcut"] = tri_desc["best_shortcut"]
    return out


# ===========================================================================
# Arm B — boundary type.
# ===========================================================================
def boundary_verdict(g_core, g_rand, disp_core, disp_rand, z=1.96):
    """§1.2's rule in its frozen order: gradient ratio first, then displacement ratio.

    H_architecture-wall : no signal exists      — gradient ratio at/below the matched-random floor
    H_repelled          : signal exists, no net motion — gradient above floor, displacement at floor
    H_optimizer-wall    : gradient above floor AND displacement above floor (the core is written to
                          and the 'wall' is not a wall at all — reported, it is a real outcome)
    """
    import numpy as np
    gc, gr = np.asarray(g_core, float), np.asarray(g_rand, float)
    dc, dr = np.asarray(disp_core, float), np.asarray(disp_rand, float)
    assert gc.shape == gr.shape and dc.shape == dr.shape, "boundary arrays misaligned"

    def _ratio(x, y):
        r = x / np.maximum(y, 1e-30)
        m = float(np.median(r))
        boot = np.array([np.median(r[np.random.default_rng(s).integers(0, r.size, r.size)])
                         for s in range(200)])
        return m, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    gm, glo, ghi = _ratio(gc, gr)
    dm, dlo, dhi = _ratio(dc, dr)
    grad_above = bool(glo > 1.0)
    disp_above = bool(dlo > 1.0)
    if not grad_above:
        v = "H_architecture-wall"
    elif not disp_above:
        v = "H_repelled"
    else:
        v = "H_optimizer-wall"
    return {"verdict": v, "gradient_ratio": gm, "gradient_ci": [glo, ghi],
            "displacement_ratio": dm, "displacement_ci": [dlo, dhi],
            "gradient_above_floor": grad_above, "displacement_above_floor": disp_above,
            # this is the number of RUNS pooled, not optimiser steps (critic finding 20)
            "n_runs_pooled": int(gc.size),
            "failure_means": {"H_architecture-wall": "no data-generated signal enters the core",
                              "H_repelled": "signal enters and is cancelled — an optimizer wall",
                              "H_optimizer-wall": "the core is written to during training"}[v]}


# ===========================================================================
# Arm X — mirror-v2: Mantel test between annotation similarity and compatibility.
# ===========================================================================
def mantel(A, B, n_perm=N_PERM, seed=0):
    """Matrix-permutation association between two square matrices, off-diagonal, rows+cols permuted
    jointly (the only permutation that preserves the matrices' own dependence structure)."""
    import numpy as np
    X, Y = np.asarray(A, float), np.asarray(B, float)
    assert X.shape == Y.shape and X.shape[0] == X.shape[1], f"mantel shapes {X.shape} {Y.shape}"
    n = X.shape[0]
    off = ~np.eye(n, dtype=bool)
    finite = off & np.isfinite(X) & np.isfinite(Y)
    assert finite.sum() >= 6, f"only {int(finite.sum())} usable off-diagonal cells"

    def _r(u, v):
        u = u - u.mean(); v = v - v.mean()
        den = np.linalg.norm(u) * np.linalg.norm(v)
        return float(u @ v / den) if den > 0 else 0.0

    obs = _r(X[finite], Y[finite])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        Yp = Y[np.ix_(p, p)]
        f = off & np.isfinite(X) & np.isfinite(Yp)
        draws[i] = _r(X[f], Yp[f])
    from math import factorial
    # POWER FLOOR. The permutation group over n labelled units has only n! members, so the
    # smallest attainable p is 1/n! regardless of n_perm — with n=5 that is 0.0083, and a
    # "not supported" verdict must be read against that ceiling, not against n_perm
    # (critic finding 14). `detectable_r_p05` is the |r| that the permuted distribution's own
    # 95th percentile sits at: nothing weaker could have been called.
    n_fact = factorial(n)
    return {"r": obs, "p_two_sided": float(((np.abs(draws) >= abs(obs)).sum() + 1) / (n_perm + 1)),
            "perm_mean": float(draws.mean()), "n_cells": int(finite.sum()), "n_perm": int(n_perm),
            "n_units": int(n), "min_attainable_p": float(1.0 / n_fact),
            "detectable_r_p05": float(np.percentile(np.abs(draws), 95)),
            "power_note": (f"only {n_fact} distinct permutations exist; |r| below "
                           f"{float(np.percentile(np.abs(draws), 95)):.3f} could not have been "
                           f"detected at α=0.05")}


def self_transplant_gate(deltas, tol):
    """P3: an A→A transplant must be a no-op. Any non-zero delta is a surgery bug → HALT."""
    import numpy as np
    d = np.asarray(deltas, float)
    worst = float(np.max(np.abs(d))) if d.size else 0.0
    return {"max_abs_delta": worst, "tol": float(tol), "passes": bool(worst <= tol),
            "n": int(d.size),
            "failure_means": "self-transplant is NOT a no-op — state-dict surgery bug, HALT"}


# ===========================================================================
# Arm D — delta-debugging witness search (ddmin) with monotonicity audit.
# ===========================================================================
def ddmin(elements, oracle, max_evals=600, seed=0, log=None):
    """Classic ddmin over `elements` with `oracle(subset) -> bool` (True = the subset RESTORES).

    Returns the 1-minimal witness plus the audit trail. `oracle` is charged against `max_evals`;
    exhausting the budget returns the best subset found with `budget_exhausted=True` — a truncated
    search is reported, never presented as minimal (`feedback_signals_that_cannot_fail`).

    Monotonicity (delta-debugging's assumption) is FALSIFIABLE and audited: any observed case where
    a superset fails while one of its subsets passes is recorded in `monotonicity_violations`."""
    import numpy as np
    rng = np.random.default_rng(seed)
    items = list(elements)
    rng.shuffle(items)
    evals = {"n": 0}
    seen = {}
    viol = []

    def test(sub):
        key = tuple(sorted(map(str, sub)))
        if key in seen:
            return seen[key]
        assert evals["n"] < max_evals, "oracle budget exhausted (guarded by caller)"
        evals["n"] += 1
        r = bool(oracle(list(sub)))
        seen[key] = r
        for k, v in seen.items():
            if v and not r and set(k).issubset(set(key)):
                viol.append({"passing_subset": list(k), "failing_superset": list(key)})
        if log is not None:
            log.append({"n": len(sub), "restores": r, "eval": evals["n"]})
        return r

    assert test(items), "ddmin precondition: the FULL set must restore, else there is no witness"
    cur = items
    n = 2
    budget_exhausted = False
    while len(cur) >= 2:
        parts = [[int(i) for i in c]
                 for c in np.array_split(np.arange(len(cur)), min(n, len(cur)))]
        parts = [p for p in parts if p]
        reduced = False
        # ddmin phase 1 — reduce to a single chunk (the big win when it fires)
        for ci in parts:
            if evals["n"] >= max_evals - 1:
                budget_exhausted = True
                break
            if test([cur[i] for i in ci]):
                cur, n, reduced = [cur[i] for i in ci], 2, True
                break
        # ddmin phase 2 — reduce to a complement (granularity stays high)
        if not reduced and not budget_exhausted:
            for ci in parts:
                drop = set(ci)
                comp = [cur[i] for i in range(len(cur)) if i not in drop]
                if not comp:
                    continue
                if evals["n"] >= max_evals - 1:
                    budget_exhausted = True
                    break
                if test(comp):
                    cur, n, reduced = comp, max(n - 1, 2), True
                    break
        if budget_exhausted:
            break
        if not reduced:
            if n >= len(cur):
                break
            n = min(2 * n, len(cur))
    return {"witness": list(cur), "size": len(cur), "n_evals": evals["n"],
            "budget_exhausted": bool(budget_exhausted),
            "monotonicity_violations": viol, "n_monotonicity_violations": len(viol),
            "note": ("TRUNCATED — oracle budget exhausted; the returned set is an upper bound on "
                     "the witness, not a minimal one" if budget_exhausted else "1-minimal witness")}


def witness_family(elements, oracle, n_restart=3, max_evals=600, seed=0):
    """≥3 restarts with different orderings → the witness FAMILY (non-uniqueness is data, §1.3)."""
    per = max_evals // n_restart
    runs = [ddmin(elements, oracle, max_evals=per, seed=seed + r) for r in range(n_restart)]
    sets = [set(map(str, r["witness"])) for r in runs]
    inter = set.intersection(*sets) if sets else set()
    union = set.union(*sets) if sets else set()
    return {"runs": runs, "sizes": [r["size"] for r in runs],
            "core_intersection": sorted(inter), "union": sorted(union),
            "jaccard_min": min((len(a & b) / max(len(a | b), 1) for a in sets for b in sets
                                if a is not b), default=1.0),
            "n_restart": n_restart,
            "unique": bool(len(set(map(frozenset, sets))) == 1)}


# ===========================================================================
# Arm G — edit-grammar description length.
# ===========================================================================
def dl_rank(m, n, r, bits=BITS_PER_PARAM):
    return float(r * (m + n) * bits)


def dl_sparse(nnz, m, n, bits=BITS_PER_PARAM):
    from math import log2
    return float(nnz * (bits + log2(max(m * n, 2))))


def dl_gain(bits=BITS_PER_PARAM):
    return float(bits)


def dl_dense(m, n, bits=BITS_PER_PARAM):
    return float(m * n * bits)


def program_dl(terms):
    """Total description length of an edit program (a list of term dicts with a `dl` field) plus
    the code needed to name each term's kind (log2 of the grammar size, charged honestly)."""
    from math import log2
    kinds = {"rank", "sparse", "interface", "gain", "dense"}
    assert all(t["kind"] in kinds for t in terms), f"unknown term kind in {terms}"
    return float(sum(t["dl"] for t in terms) + len(terms) * log2(len(kinds)))


def compression_ratio(prog_dl, baseline_dl):
    """<1 means the typed program is SHORTER than the structure-free baseline at matched fidelity."""
    assert baseline_dl > 0, "baseline description length must be positive"
    return float(prog_dl / baseline_dl)
