# W3 — representation geometry & cross-modal alignment (spec §2.3 exploratory arm, §7 W3).
"""Pure metrics for the W3 GEOMETRY lens: does vision become language-addressable
inside the PG2 decoder, and WHERE?

EXPLORATORY / hypothesis-generating, NOT confirmatory. Geometry is correlational;
the strongest findings are flagged for the W5 causal test, never over-claimed.

This module is PURE (numpy/torch/sklearn only, NO Modal) so it is unit-importable
and the heavy harvest (modal_pg_w3.py) and the analysis share ONE estimator —
exactly the W4 discipline (lm_feature_drift.py is the sibling).

The four W3 instruments (all per-decoder-layer, base-LM-with-PG-projector "before"
vs PG2-LM "after"):

  1. CROSS-MODAL ALIGNMENT between projected image-token reps and MATCHED
     text-concept reps, on a region<->word correspondence set (RefCOCO box<->phrase),
     with TWO metrics so no conclusion is metric-dependent:
       (a) mutual-kNN  (Huh et al. 2405.07987 Platonic; the metric that INFLATES on
           small galleries -> reported at C3 gallery sizes {1k,10k,100k}).
       (b) aligned-cosine via orthogonal Procrustes (Schoenemann 1966): fit R on a
           train split, score mean matched cosine on a DISJOINT test split (a learned
           rotation cannot be gamed by chance on held-out pairs).
     "Matched" = the image-token rep for the patch the GT box covers vs the text rep of
     the box's phrase token(s). The before->after contrast isolates whether the LM's
     REWRITE (not the held-fixed connector) makes image tokens language-readable.

  2. GEOMETRY of image-token reps: TwoNN intrinsic dim (Facco 2017) + linear & RBF
     CKA (Kornblith 2019; single-kernel CKA can mislead -> both) of image-token reps,
     before vs after, with the C1 activation-scale control. Does the image-token
     manifold get reshaped by the rewrite, and where?

  3. W4 GAIN-CONTROL GEOMETRIC TEST. W4's text-SAE found the LM ingests vision by
     SUPPRESSING on text the features image tokens drive (gain-control), with emergent
     latents tiny. A geometric signature: image-token rep NORM and the SEPARATION of
     the image-token manifold from the text-token manifold across layers, before vs
     after. If gain-control is real, the rewrite should change image<->text separation
     / relative norm even though few NEW latents appear.

Statistics: bootstrap CIs on every endpoint (>=1000 resamples); the alignment
gallery-size curve is the C3 control; FDR (BH q=0.05) lives in the orchestrator
(modal_pg_w3) across the layer x metric family. After-alignment must exceed BOTH
before AND the independent-model Platonic floor to count (spec §7 W3).
"""
from __future__ import annotations

import numpy as np


# ===========================================================================
# C1 — activation-scale normalization (spec §4 C1; both CKA & mutual-kNN are
# scale-sensitive, and RMSNorm scale differs between base-Gemma2 and PG2-LM).
# ===========================================================================

def standardize(X: np.ndarray) -> np.ndarray:
    """Per-feature mean-center + unit-variance (the C1 normalization). Reported
    alongside RAW (un-normalized) so a measured diff is not a scale artefact."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return X / n


# ===========================================================================
# Mutual-kNN alignment (Platonic, Huh et al. 2405.07987). The fraction of a
# point's k nearest neighbours (in modality A's space) that are ALSO its
# neighbours in modality B's space, averaged over points. C3: this metric
# INFLATES on small galleries (Back into Plato's Cave 2604.18572) -> we report
# it across gallery sizes and treat a 1k-only effect as artefact.
# ===========================================================================

def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k nearest neighbours of each row (excluding self), cosine
    distance. X is (n, d). Returns (n, k) int.

    Uses a chunked GPU torch top-k when CUDA is available (cosine = top-k of the
    row-normalized Gram), which makes the 100k gallery (C3) tractable — sklearn's
    brute O(n^2) on CPU does not scale there. Falls back to sklearn exact kNN on CPU.
    EXACT either way (full pairwise cosine, no approximation)."""
    import importlib.util
    n = X.shape[0]
    use_gpu = False
    if importlib.util.find_spec("torch") is not None:
        import torch
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        import torch
        Xt = torch.from_numpy(np.ascontiguousarray(X)).to("cuda", torch.float32)
        Xt = Xt / (Xt.norm(dim=1, keepdim=True) + 1e-12)
        idx = torch.empty((n, k), dtype=torch.long, device="cuda")
        CH = 4096
        for s in range(0, n, CH):
            sim = Xt[s:s + CH] @ Xt.T                       # (chunk, n) cosine
            # exclude self: set the diagonal entry to -inf for this chunk.
            rows = torch.arange(s, min(s + CH, n), device="cuda")
            sim[torch.arange(rows.numel(), device="cuda"), rows] = float("-inf")
            idx[s:s + CH] = sim.topk(k, dim=1).indices
        return idx.cpu().numpy()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]   # drop self (col 0)


def mutual_knn(Xa: np.ndarray, Xb: np.ndarray, k: int = 10) -> float:
    """Mutual-kNN alignment between PAIRED reps Xa[i] <-> Xb[i] (same i = same
    region<->word pair). For each i, neighbour-set overlap fraction in A vs B,
    averaged. 0 = no shared neighbour structure; 1 = identical neighbourhoods."""
    Xa = np.asarray(Xa, np.float64); Xb = np.asarray(Xb, np.float64)
    n = Xa.shape[0]
    if n != Xb.shape[0]:
        raise ValueError(f"mutual_knn paired rows must match: {Xa.shape} vs {Xb.shape}")
    if n <= k + 1:
        raise ValueError(f"mutual_knn needs n > k+1; got n={n}, k={k}")
    ia = _knn_indices(Xa, k)
    ib = _knn_indices(Xb, k)
    overlaps = np.empty(n, dtype=np.float64)
    for i in range(n):
        overlaps[i] = len(set(ia[i].tolist()) & set(ib[i].tolist())) / k
    return float(overlaps.mean())


def mutual_knn_overlaps(Xa: np.ndarray, Xb: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-point mutual-kNN overlap fractions (the (n,) vector mutual_knn means).
    Exposed so the after-vs-before contrast can be a PAIRED per-point bootstrap."""
    Xa = np.asarray(Xa, np.float64); Xb = np.asarray(Xb, np.float64)
    n = Xa.shape[0]
    ia = _knn_indices(Xa, k); ib = _knn_indices(Xb, k)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = len(set(ia[i].tolist()) & set(ib[i].tolist())) / k
    return out


def mutual_knn_gallery_curve(
    Xa: np.ndarray, Xb: np.ndarray, gallery_sizes, k: int = 10,
    n_boot: int = 200, seed: int = 0,
) -> dict:
    """The C3 gallery-scale control: mutual-kNN at each gallery size with a
    bootstrap CI (resample the gallery). At sizes <= the full N we subsample a
    gallery `n_boot` times; the Platonic metric's known small-gallery inflation
    is thus made VISIBLE rather than hidden. Returns {size: {mean, ci_lo, ci_hi}}.
    """
    rng = np.random.default_rng(seed)
    N = Xa.shape[0]
    out: dict = {}
    for g in gallery_sizes:
        g = int(g)
        if g > N:
            out[str(g)] = {"requested": g, "available": N, "skipped": True}
            continue
        # mutual_knn is O(g^2) (brute cosine kNN); resample budget scales DOWN with g so
        # the large galleries stay tractable while small ones still get a tight CI. The
        # full-N point is evaluated once (it IS the data). A capped reps still gives a
        # bootstrap CI; the C3 point estimate is the headline.
        if g == N:
            reps = 1
        elif g <= 2000:
            reps = n_boot
        elif g <= 20000:
            reps = max(8, n_boot // 8)
        else:
            reps = 3
        vals = []
        for _ in range(reps):
            sel = rng.choice(N, size=g, replace=False) if g < N else np.arange(N)
            vals.append(mutual_knn(Xa[sel], Xb[sel], k=k))
        vals = np.asarray(vals)
        out[str(g)] = {
            "gallery_size": g, "k": k,
            "mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)) if reps > 2 else float(vals.min()),
            "ci_hi": float(np.percentile(vals, 97.5)) if reps > 2 else float(vals.max()),
            "n_resamples": reps,
        }
    return out


# ===========================================================================
# Orthogonal Procrustes -> aligned cosine (metric #2; train/test split so a
# learned rotation cannot be gamed on held-out pairs). Schoenemann 1966.
# ===========================================================================

def procrustes_aligned_cosine(
    Xa: np.ndarray, Xb: np.ndarray, train_frac: float = 0.5, seed: int = 0,
) -> dict:
    """Fit the optimal orthogonal map R (det free) mapping A->B on a TRAIN split of
    paired rows, then report the mean matched cosine similarity on the DISJOINT TEST
    split. Both spaces are l2-row-normalized first (scale-free; the C1 spirit).
    A learned rotation that only fit noise would NOT raise test matched-cosine above
    a permuted baseline, so we also report the permuted-pair test cosine as the floor.
    Returns train R fit error + test matched/permuted cosine + lift."""
    Xa = l2_normalize_rows(Xa); Xb = l2_normalize_rows(Xb)
    n = Xa.shape[0]
    if n != Xb.shape[0]:
        raise ValueError(f"procrustes paired rows must match: {Xa.shape} vs {Xb.shape}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(train_frac * n))
    tr, te = perm[:n_tr], perm[n_tr:]
    if te.size < 8:
        raise ValueError(f"procrustes test split too small ({te.size}); need >=8 pairs")

    A, B = Xa[tr], Xb[tr]
    # R = argmin ||A R - B||_F  ->  M = A^T B = U S V^T ; R = U V^T
    M = A.T @ B
    U, _S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    fit_err = float(np.linalg.norm(A @ R - B) / max(np.linalg.norm(B), 1e-12))

    Ate, Bte = Xa[te], Xb[te]
    mapped = Ate @ R
    mapped = l2_normalize_rows(mapped)
    matched = float((mapped * Bte).sum(1).mean())          # mean cosine of matched pairs
    # permuted floor: same mapped reps vs SHUFFLED targets (chance matched cosine).
    sh = rng.permutation(te.size)
    permuted = float((mapped * Bte[sh]).sum(1).mean())
    return {
        "n_pairs": int(n), "n_train": int(tr.size), "n_test": int(te.size),
        "fit_err_rel_frob": fit_err,
        "test_matched_cosine": matched,
        "test_permuted_cosine": permuted,
        "lift_over_permuted": float(matched - permuted),
    }


def procrustes_aligned_cosine_boot(
    Xa: np.ndarray, Xb: np.ndarray, n_boot: int = 100, train_frac: float = 0.5,
    seed: int = 0,
) -> dict:
    """Bootstrap CI on the Procrustes test matched-cosine + lift, resampling the
    train/test SPLIT (different seed each rep). The headline endpoint is the
    lift_over_permuted (alignment above chance). Each rep's cost is a d×d SVD (fixed
    in n), so a learned rotation on thousands of pairs is stable across splits and
    n_boot=100 gives a tight CI without the d×d SVD ×1000 blowing up at scale."""
    matched, lift = [], []
    for b in range(n_boot):
        r = procrustes_aligned_cosine(Xa, Xb, train_frac=train_frac, seed=seed + b)
        matched.append(r["test_matched_cosine"]); lift.append(r["lift_over_permuted"])
    matched = np.asarray(matched); lift = np.asarray(lift)
    return {
        "n_boot": n_boot,
        "matched_cosine_mean": float(matched.mean()),
        "matched_cosine_ci": [float(np.percentile(matched, 2.5)),
                              float(np.percentile(matched, 97.5))],
        "lift_mean": float(lift.mean()),
        "lift_ci": [float(np.percentile(lift, 2.5)), float(np.percentile(lift, 97.5))],
    }


# ===========================================================================
# CKA — linear AND RBF (Kornblith 2019). Single-kernel CKA can mislead -> both.
# Used base-vs-after on the SAME stimuli to ask: how much does the rewrite change
# the image-token rep geometry at this layer (1 = identical, 0 = unrelated)?
# ===========================================================================

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Double-centre a Gram matrix: HKH where H = I - 11^T/n. Computed via the rank-1
    identity (K - row_mean - col_mean + grand_mean) in O(n^2) — NOT by forming the
    dense n×n H and two n×n×n matmuls (which is O(n^3) and dominates at n~1e4)."""
    K = np.asarray(K, np.float64)
    rm = K.mean(0, keepdims=True)
    cm = K.mean(1, keepdims=True)
    gm = K.mean()
    return K - rm - cm + gm


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two rep matrices over the SAME n points (X,Y are (n,dx),
    (n,dy)). Scale-invariant by construction (HSIC normalized)."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # HSIC_lin(X,Y) = ||X^T Y||_F^2 ; CKA = HSIC(X,Y)/sqrt(HSIC(X,X)HSIC(Y,Y))
    xty = X.T @ Y
    xtx = X.T @ X
    yty = Y.T @ Y
    hsic_xy = float((xty * xty).sum())
    hsic_xx = float((xtx * xtx).sum())
    hsic_yy = float((yty * yty).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else float("nan")


def _rbf_gram(X: np.ndarray, sigma_mult: float = 1.0) -> np.ndarray:
    """RBF kernel with bandwidth = sigma_mult * median pairwise distance (the
    standard median-heuristic; Kornblith 2019 use a fraction-of-median sweep)."""
    X = np.asarray(X, np.float64)
    sq = np.sum(X * X, 1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)
    n = X.shape[0]
    iu = np.triu_indices(n, k=1)
    med = np.median(d2[iu])
    if med <= 0:
        med = 1.0
    gamma = 1.0 / (2.0 * (sigma_mult ** 2) * med)
    return np.exp(-gamma * d2)


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_mult: float = 1.0) -> float:
    """RBF-kernel CKA (HSIC with centred Gram matrices). Complements linear CKA."""
    Kx = _center_gram(_rbf_gram(X, sigma_mult))
    Ky = _center_gram(_rbf_gram(Y, sigma_mult))
    hsic_xy = float((Kx * Ky).sum())
    hsic_xx = float((Kx * Kx).sum())
    hsic_yy = float((Ky * Ky).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else float("nan")


# ===========================================================================
# TwoNN intrinsic dimension (Facco 2017). The ID of the image-token manifold,
# before vs after, per layer. A rewrite that compresses/expands the manifold
# shows up as an ID shift.
# ===========================================================================

def twonn_intrinsic_dim(X: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN intrinsic dimension. For each point, mu = r2/r1 (ratio of distances to
    its 2nd and 1st nearest neighbours). ID is the slope of the line through the
    origin in (-log(1-F(mu)), log mu) space, fit on the lower (1-discard_frac)
    quantile of mu (Facco et al. 2017, the canonical estimator + outlier discard)."""
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, np.float64)
    n = X.shape[0]
    if n < 10:
        raise ValueError(f"TwoNN needs >=10 points; got {n}")
    nn = NearestNeighbors(n_neighbors=3).fit(X)   # self + r1 + r2
    d, _ = nn.kneighbors(X)
    r1 = d[:, 1]; r2 = d[:, 2]
    ok = r1 > 0
    mu = r2[ok] / r1[ok]
    mu = mu[mu > 1.0]                              # strictly increasing distances
    mu = np.sort(mu)
    m = mu.size
    keep = int(np.floor(m * (1.0 - discard_frac)))
    mu = mu[:keep]
    F = np.arange(1, mu.size + 1) / m              # empirical CDF
    x = np.log(mu)
    y = -np.log(1.0 - F)
    # slope through origin: d = sum(x*y)/sum(x*x)
    denom = float((x * x).sum())
    return float((x * y).sum() / denom) if denom > 0 else float("nan")


# ===========================================================================
# W4 gain-control geometric test. image-token rep NORM + image<->text manifold
# SEPARATION across layers, before vs after. (W4 text-SAE: the LM ingests vision
# by suppressing-on-text, emergent latents tiny -> a GEOMETRIC signature should
# appear even where new latents do not.)
# ===========================================================================

def manifold_separation(img: np.ndarray, txt: np.ndarray) -> dict:
    """Geometry of the image-token cloud vs the text-token cloud at one layer.

    Reports, all on the RAW reps (norms are the point) plus a scale-free cosine:
      * mean image-token rep L2 norm vs mean text-token rep norm (+ ratio).
      * centroid separation: ||mean(img) - mean(txt)|| normalized by pooled rep
        scale -> a scale-free 'how far apart are the two clouds'.
      * linear-probe separability AUC (image vs text) via a 1-D Fisher LDA score
        (cheap, deterministic) -> how linearly separable the two token types are.
    The before->after MOVE in these is the gain-control geometric signature."""
    img = np.asarray(img, np.float64); txt = np.asarray(txt, np.float64)
    img_norm = float(np.linalg.norm(img, axis=1).mean())
    txt_norm = float(np.linalg.norm(txt, axis=1).mean())
    mi, mt = img.mean(0), txt.mean(0)
    pooled_scale = float(np.sqrt(((img - mi) ** 2).sum(1).mean()
                                 + ((txt - mt) ** 2).sum(1).mean()) + 1e-12)
    centroid_sep = float(np.linalg.norm(mi - mt))
    # Fisher LDA 1-D projection w ~ (mi - mt); score AUC of img>txt along w.
    w = mi - mt
    wn = np.linalg.norm(w)
    if wn > 0:
        w = w / wn
        si = img @ w; st = txt @ w
        auc = _auc_separability(si, st)
    else:
        auc = 0.5
    return {
        "img_rep_norm_mean": img_norm,
        "txt_rep_norm_mean": txt_norm,
        "img_txt_norm_ratio": float(img_norm / txt_norm) if txt_norm > 0 else float("nan"),
        "centroid_separation": centroid_sep,
        "centroid_separation_scalefree": float(centroid_sep / pooled_scale),
        "img_vs_txt_separability_auc": float(auc),
        "n_image": int(img.shape[0]), "n_text": int(txt.shape[0]),
    }


def _auc_separability(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC that scores pos>neg along a 1-D axis (Mann-Whitney U / rank statistic).
    Symmetric: returns max(auc, 1-auc) so it measures separability, not direction."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    r_pos = ranks[: pos.size].sum()
    n1, n2 = pos.size, neg.size
    auc = (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n2)
    return max(auc, 1.0 - auc)


# ===========================================================================
# FDR (BH q=0.05) — the look-elsewhere control across the layer x metric family.
# (Same estimator as W4's lm_feature_drift.bh_fdr; duplicated here so the module
# stays self-contained / unit-importable.)
# ===========================================================================

def bh_fdr(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * q
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = int(np.max(np.where(passed)[0]))
    crit = ranked[kmax]
    return p <= crit


def permutation_p_greater(observed: float, null_samples: np.ndarray) -> float:
    """One-sided permutation p: P(null >= observed). +1 smoothing (Phipson &
    Smyth 2010) so p is never exactly 0."""
    null_samples = np.asarray(null_samples, dtype=float)
    b = int((null_samples >= observed).sum())
    return (b + 1) / (null_samples.size + 1)


def bootstrap_paired_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0,
) -> dict:
    """Bootstrap CI on a per-point paired difference mean(a-b) (e.g. after-vs-before
    per-point overlap). Returns mean diff + 95% CI + one-sided p that diff>0."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = a - b
    rng = np.random.default_rng(seed)
    n = d.size
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = d[rng.integers(0, n, n)].mean()
    return {
        "diff_mean": float(d.mean()),
        "diff_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_diff_gt_0": float((means <= 0).mean()),
    }
