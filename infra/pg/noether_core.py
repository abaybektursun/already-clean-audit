"""NOETHER pure core — affine alignment maps from streamed statistics, nulls, gauge fraction, verdict.

Spec: studies/lineage/cross-lineage/research/2026-09-02_noether_spec.md §2/§6. Pure numpy; asserts;
`python -m infra.pg.noether_core` runs the dry-run gate (synthetic gauge-drift vs damage pairs; every
verdict branch reachable). No model, no GPU.
"""
import numpy as np

from infra.pg.broca_core import null_band, threshold  # noqa: F401  (re-exported, same frozen rules)


class RidgeStats:
    """Streamed sufficient statistics for an affine map X -> Y (rows paired). add() accepts float32/64
    blocks; everything is accumulated in float64. Both directions solve from the same object."""

    def __init__(self, d: int):
        self.d = d
        self.n = 0
        self.sx = np.zeros(d); self.sy = np.zeros(d)
        self.xx = np.zeros((d, d)); self.xy = np.zeros((d, d)); self.yy = np.zeros((d, d))

    def add(self, X, Y):
        X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
        assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == self.d, (X.shape, Y.shape, self.d)
        self.n += X.shape[0]
        self.sx += X.sum(0); self.sy += Y.sum(0)
        self.xx += X.T @ X; self.xy += X.T @ Y; self.yy += Y.T @ Y

    def add_products(self, n: int, sx, sy, xx, xy, yy):
        """Accumulate pre-computed block products (the GPU path: products in fp32 with TF32 off,
        summed here in fp64)."""
        assert xx.shape == xy.shape == yy.shape == (self.d, self.d), (xx.shape, self.d)
        self.n += int(n)
        self.sx += np.asarray(sx, dtype=np.float64); self.sy += np.asarray(sy, dtype=np.float64)
        self.xx += np.asarray(xx, dtype=np.float64); self.xy += np.asarray(xy, dtype=np.float64)
        self.yy += np.asarray(yy, dtype=np.float64)

    def centered(self):
        assert self.n > self.d, f"n={self.n} <= d={self.d}: under-determined"
        mx = self.sx / self.n; my = self.sy / self.n
        cxx = self.xx - self.n * np.outer(mx, mx)
        cxy = self.xy - self.n * np.outer(mx, my)
        cyy = self.yy - self.n * np.outer(my, my)
        return mx, my, cxx, cxy, cyy

    def swapped(self):
        """The same statistics for the reverse direction Y -> X."""
        s = RidgeStats(self.d); s.n = self.n
        s.sx, s.sy = self.sy.copy(), self.sx.copy()
        s.xx, s.yy, s.xy = self.yy.copy(), self.xx.copy(), self.xy.T.copy()
        return s


def ridge_solve(stats: RidgeStats, lam_rel: float):
    """W, b with y ≈ x W + b; λ = lam_rel · trace(Cxx)/d (relative to the mean eigenvalue)."""
    mx, my, cxx, cxy, _ = stats.centered()
    lam = lam_rel * np.trace(cxx) / stats.d
    W = np.linalg.solve(cxx + lam * np.eye(stats.d), cxy)
    b = my - mx @ W
    return W, b


def held_mse(W, b, stats: RidgeStats) -> float:
    """Mean squared reconstruction error of x W + b against y, from (held-out) statistics only."""
    mx, my, cxx, cxy, cyy = stats.centered()
    # E|y - xW - b|^2 with b = my - mx W on the FIT split; on held data the means differ slightly,
    # so expand around the held moments exactly:
    r = my - mx @ W - b                      # residual of the means
    tr = (np.trace(cyy) - 2 * np.trace(W.T @ cxy) + np.trace(W.T @ cxx @ W)) / stats.n
    return float(tr + r @ r)


def select_lambda(fit: RidgeStats, held: RidgeStats, grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Frozen rule: the λ with the smallest held-out MSE. Control data only (no behaviour)."""
    best = None
    for g in grid:
        W, b = ridge_solve(fit, g)
        m = held_mse(W, b, held)
        if best is None or m < best[0]:
            best = (m, g, W, b)
    return {"lam_rel": best[1], "held_mse": best[0], "W": best[2], "b": best[3]}


def baseline_mse(held: RidgeStats) -> dict:
    """Reference MSEs on the held split: predicting the mean, and the identity map (y ≈ x + (my - mx))."""
    mx, my, cxx, cxy, cyy = held.centered()
    d = held.d
    mean_only = float(np.trace(cyy) / held.n)
    ident = float((np.trace(cyy) - 2 * np.trace(cxy) + np.trace(cxx)) / held.n)
    return {"mean_only": mean_only, "identity": ident, "d": d}


# ---------------------------------------------------------------- map families + nulls
def mean_map(stats: RidgeStats):
    mx, my, *_ = stats.centered()
    return np.eye(stats.d), my - mx


def diag_map(stats: RidgeStats):
    """Per-dimension scale + bias (least squares per coordinate)."""
    mx, my, cxx, cxy, _ = stats.centered()
    s = np.diag(cxy) / np.maximum(np.diag(cxx), 1e-12)
    return np.diag(s), my - mx * s


def truncate_delta(W, mx, my, k: int):
    """W_k = I + top-k SVD of (W - I); bias refit so the means always match (k=0 = mean map)."""
    d = W.shape[0]
    D = W - np.eye(d)
    if k <= 0:
        Wk = np.eye(d)
    else:
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        Wk = np.eye(d) + (U[:, :k] * S[:k]) @ Vt[:k]
    return Wk, my - mx @ Wk


def random_matched_delta(W, mx, my, k: int, seed: int):
    """Matched-capacity null: same top-k singular values as (W - I), random orthonormal U, V."""
    d = W.shape[0]
    _, S, _ = np.linalg.svd(W - np.eye(d), full_matrices=False)
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, k)))
    V, _ = np.linalg.qr(rng.standard_normal((d, k)))
    Wr = np.eye(d) + (U * S[:k]) @ V.T
    return Wr, my - mx @ Wr


def delta_spectrum(W) -> np.ndarray:
    return np.linalg.svd(W - np.eye(W.shape[0]), compute_uv=False)


# ---------------------------------------------------------------- readouts + verdict
def damage_fraction(acc_patched: float, acc_after: float, acc_before: float) -> float:
    """D = transported damage as a fraction of the lost margin (healthy recipient, broken donor)."""
    lost = acc_before - acc_after
    assert lost > 1e-6, f"no lost margin (before={acc_before}, after={acc_after})"
    return float((acc_before - acc_patched) / lost)


def gauge_fraction(D_identity: float, D_mapped: float) -> float:
    """G = fraction of the identity-graft damage removed by the map. Raw (may exceed 1 / go < 0)."""
    assert D_identity > 1e-6, f"identity graft transported no damage (D_I={D_identity})"
    return float((D_identity - D_mapped) / D_identity)


def null_recovery_band(D_identity: dict, D_nulls: dict) -> float:
    """Band = max over (layer, null type) of (D_I - D_null): how much the nulls 'recover'."""
    vals = []
    for L, per_type in D_nulls.items():
        for _, dn in per_type.items():
            vals.append(D_identity[L] - dn)
    assert len(vals) >= 3, "need >=3 null cuts"
    return float(max(0.0, max(vals)))


def persistence_depth(layers, D_full, hi: float = 0.7):
    """L_loss = first layer from which D_full stays > hi to the end (content no longer linearly
    recoverable). None if never sustained."""
    L = list(layers); D = list(D_full)
    assert L == sorted(L) and len(L) == len(D)
    for i in range(len(L)):
        if all(d > hi for d in D[i:]):
            return L[i]
    return None


def rank_knee(ks, G_fit, G_full: float, frac: float = 0.8):
    """k* = smallest k with G_k >= frac * G_full. None if no k reaches it."""
    for k, g in sorted(zip(ks, G_fit)):
        if g >= frac * G_full:
            return k
    return None


def bootstrap_gauge(items_before, items_after, items_identity, items_mapped, n_boot: int = 2000,
                    seed: int = 1337):
    """Paired item bootstrap over the four per-item correctness vectors (same items, same order).
    Returns {D_I, D_M, G: (point, lo, hi)} at 95 %."""
    ib, ia, ii, im = (np.asarray(v, dtype=float) for v in (items_before, items_after, items_identity,
                                                          items_mapped))
    n = ib.size
    assert ia.size == ii.size == im.size == n and n >= 20, n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    def _stats(sel):
        b, a, i, m = ib[sel].mean(-1), ia[sel].mean(-1), ii[sel].mean(-1), im[sel].mean(-1)
        lost = np.maximum(b - a, 1e-6)
        DI = (b - i) / lost; DM = (b - m) / lost
        G = (DI - DM) / np.where(np.abs(DI) < 1e-6, np.nan, DI)
        return DI, DM, G
    DI, DM, G = _stats(idx)
    pDI, pDM, pG = _stats(np.arange(n)[None, :])
    def _ci(x):
        x = x[np.isfinite(x)]
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return {"D_I": (float(pDI[0]), *_ci(DI)), "D_M": (float(pDM[0]), *_ci(DM)),
            "G": (float(pG[0]), *_ci(G)), "n_items": int(n), "n_boot": int(n_boot)}


def noether_verdict(band_layers, D_identity: dict, D_full: dict, D_nulls: dict, D_wrong_item: dict,
                    T: float) -> dict:
    """§6 frozen mapping over the band layers. D_* are dicts keyed by layer; D_nulls[L][name]."""
    G = {L: gauge_fraction(D_identity[L], D_full[L]) for L in band_layers}
    beats = {L: (D_identity[L] - D_full[L]) - max(D_identity[L] - dn for dn in D_nulls[L].values()) >= T
             for L in band_layers}
    void = {L: D_wrong_item[L] < 0.7 for L in band_layers}
    valid = [L for L in band_layers if not void[L]]
    n_gauge = sum(1 for L in valid if G[L] >= 0.7 and beats[L])
    all_damage = all(G[L] < 0.3 for L in valid) if valid else False
    if len(valid) < 2:
        v = "VOID (wrong-item-through-map restored at >=2 band layers — the map carries content)"
    elif n_gauge >= 2:
        v = "H_gauge (mid-stack change is linearly compensable: content present in a drifted basis)"
    elif all_damage:
        v = "H_damage (no affine map recovers: mid-stack content is lost)"
    else:
        v = "H_partial (part gauge, part damage — report components, ladder, L_loss)"
    return {"verdict": v, "G": G, "beats_null": beats, "void_layer": void, "n_gauge_layers": n_gauge, "T": T}


# ---------------------------------------------------------------- dry-run gate (synthetic)
def _synthetic_pair(kind: str, d=24, n=6000, seed=0):
    """Healthy H ~ N(0, Σ). 'gauge': B = H Q + c (rotation) — content intact; 'damage': B = H P (rank-
    deficient projection killing the readout subspace) + noise. Readout = fixed linear probe on H."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(d); w /= np.linalg.norm(w)          # the 'reader' direction
    if kind == "gauge":
        A = rng.standard_normal((d, d)); H = rng.standard_normal((n, d)) @ A   # correlated dims
        Q, _ = np.linalg.qr(rng.standard_normal((d, d))); c = rng.standard_normal(d) * 3
        B = H @ Q + c
    elif kind == "damage":
        # isotropic H so the reader coordinate is NOT redundantly encoded elsewhere (with correlated
        # dims a killed direction is still linearly recoverable — that is content redundancy, not
        # damage; the first dry-run caught exactly this). Kill it and add noise: information gone.
        H = rng.standard_normal((n, d))
        P = np.eye(d) - np.outer(w, w)
        B = H @ P + 0.5 * rng.standard_normal((n, d))
    else:
        raise ValueError(kind)
    return H, B, w


def _acc(Z, w, labels):
    return float(((Z @ w > 0).astype(int) == labels).mean())


def _dryrun():
    d = 24
    for kind, want_G in (("gauge", 0.9), ("damage", 0.2)):
        H, B, w = _synthetic_pair(kind, d=d)
        labels = (H @ w > 0).astype(int)
        fit, held = RidgeStats(d), RidgeStats(d)
        fit.add(B[:4000], H[:4000]); held.add(B[4000:], H[4000:])
        sel = select_lambda(fit, held)
        W, b = sel["W"], sel["b"]
        assert sel["held_mse"] < baseline_mse(held)["identity"], "ridge worse than identity"
        acc_before = _acc(H[4000:], w, labels[4000:])            # healthy reader on healthy state
        acc_after = 0.5                                          # the broken model is at chance
        D_I = damage_fraction(_acc(B[4000:], w, labels[4000:]), acc_after, acc_before)
        D_M = damage_fraction(_acc(B[4000:] @ W + b, w, labels[4000:]), acc_after, acc_before)
        G = gauge_fraction(D_I, D_M)
        if kind == "gauge":
            assert D_I > 0.5 and G >= want_G, (kind, D_I, D_M, G)
        else:
            assert D_I > 0.5 and G <= want_G, (kind, D_I, D_M, G)
        # matched-capacity random null at full rank must NOT recover in the gauge case
        mx, my, *_ = fit.centered()
        Wr, br = random_matched_delta(W, mx, my, d, seed=1)
        D_R = damage_fraction(_acc(B[4000:] @ Wr + br, w, labels[4000:]), acc_after, acc_before)
        if kind == "gauge":
            assert gauge_fraction(D_I, D_R) < 0.5, ("random matched map recovered", D_R)
        # wrong-item control: the map on a rolled (mismatched) B must not restore
        D_wrong = damage_fraction(_acc(np.roll(B[4000:], 1, 0) @ W + b, w, labels[4000:]),
                                  acc_after, acc_before)
        assert D_wrong > 0.7, ("map smuggles content", D_wrong)
        # ladder monotone-ish and spectrum shapes
        Wk, bk = truncate_delta(W, mx, my, 0); assert np.allclose(Wk, np.eye(d))
        assert delta_spectrum(W).shape == (d,)
    # verdict branches
    band = [10, 12, 14]
    DI = {L: 1.0 for L in band}
    nulls = {L: {"rand": 0.95, "shuf": 0.9} for L in band}
    ok = {L: 0.9 for L in band}
    T = 0.10
    v = noether_verdict(band, DI, {10: 0.2, 12: 0.25, 14: 0.5}, nulls, ok, T); assert v["verdict"].startswith("H_gauge"), v
    v = noether_verdict(band, DI, {10: 0.9, 12: 0.85, 14: 0.95}, nulls, ok, T); assert v["verdict"].startswith("H_damage"), v
    v = noether_verdict(band, DI, {10: 0.5, 12: 0.6, 14: 0.9}, nulls, ok, T); assert v["verdict"].startswith("H_partial"), v
    v = noether_verdict(band, DI, {10: 0.2, 12: 0.2, 14: 0.2}, nulls, {10: 0.2, 12: 0.3, 14: 0.9}, T)
    assert v["verdict"].startswith("VOID"), v
    # a 'gauge-looking' G that does not beat a strong null is NOT H_gauge
    strong = {L: {"rand": 0.3} for L in band}
    v = noether_verdict(band, DI, {10: 0.25, 12: 0.25, 14: 0.25}, strong, ok, T); assert v["verdict"].startswith("H_partial"), v
    assert persistence_depth([0, 2, 4, 6], [0.1, 0.2, 0.8, 0.9]) == 4
    assert persistence_depth([0, 2, 4, 6], [0.1, 0.9, 0.2, 0.9]) == 6
    assert rank_knee([1, 8, 64], [0.1, 0.7, 0.85], 0.85) == 8
    assert rank_knee([1, 8], [0.1, 0.2], 0.85) is None
    # bootstrap: a clean 50%-restored case has G ≈ 0.5 inside its CI; identical vectors -> G ≈ 0
    rng = np.random.default_rng(0)
    ib = np.ones(300); ia = (rng.random(300) < 0.5).astype(float)
    ii = ia.copy(); im = np.where(rng.random(300) < 0.5, 1.0, ia)
    bs = bootstrap_gauge(ib, ia, ii, im)
    assert bs["G"][1] <= 0.5 <= bs["G"][2], bs
    bs0 = bootstrap_gauge(ib, ia, ii, ii)
    assert abs(bs0["G"][0]) < 1e-9 and bs0["D_I"][0] > 0.99, bs0
    # streamed products path equals the block path
    r1, r2 = RidgeStats(4), RidgeStats(4)
    X = rng.standard_normal((50, 4)); Y = rng.standard_normal((50, 4))
    r1.add(X, Y); r2.add_products(50, X.sum(0), Y.sum(0), X.T @ X, X.T @ Y, Y.T @ Y)
    assert np.allclose(r1.xy, r2.xy) and r1.n == r2.n
    print("NOETHER core dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
