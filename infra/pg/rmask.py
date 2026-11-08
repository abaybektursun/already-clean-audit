"""rmask — pure-logic core for the READOUT-MASK vs INFORMATION-LOSS study.

Companion to `studies/lineage/paligemma2/research/2026-07-01_readout_mask_prereg.md` (slug `rmask`).
PURE numpy — no torch, no sklearn, no model, no GPU — so it py_compiles and unit-tests on the control
plane. The GPU harvest + the sklearn LogisticRegression probe live in `infra/aws_pg/jobs/rmask_cell.py`;
ALL of the gap arithmetic / shuffled-label null / item-bootstrap / data-efficiency / readout-alignment
logic lives HERE and is provable weight-free (prereg §13). The dry-run uses a closed-form numpy ridge
one-hot LINEAR probe as a stand-in — it validates the DESIGN LOGIC (a separable signal decodes, a shuffled
null collapses, floor < deleted < ceiling separates); the exact sklearn call is validated on the box.

The question (prereg §1). LATABLATE/FLENS showed the (deleted−intact) DIFFERENCE is not removable /
feature-sparse (answered). This study reads the DELETED residual ALONE: is the correct MMLU answer still
linearly PRESENT and computed at L16 (a readout MASK of preserved computation) or gone by mid-stack (a
LOSS)? A frozen linear probe on the deleted L16 last-token residual, judged against an intact-L16 CEILING,
an early-layer FLOOR, and a shuffled-label NULL — so no outcome is instrument-ambiguous.

Discipline: no try/except hiding; degenerate shapes RAISE. No model-name or baseline literals baked in.
"""
from __future__ import annotations

import numpy as np

# ===========================================================================
# Frozen thresholds (prereg §6) — SINGLE SOURCE OF TRUTH, imported by the driver and the dry-run.
# ===========================================================================
CHANCE = 0.25                     # 4-way MCQ
DELTA_DECODE = 0.10               # a gap counts only if ≥ this (guards against bootstrap noise)
GAMMA_CEILING = 0.10              # deleted-L16 is "≈ ceiling" iff within this of intact-L16
LAYER_SET = (2, 4, 8, 12, 16, 20, 25)     # the frozen depth profile
L_HEADLINE = 16                   # the LATLOC/LATABLATE/FLENS locus
L_FLOOR = 4                       # the surface-prompt floor layer
L_PIVOT_ORDER = (16, 20, 12, 25, 8)       # if intact-L16 preflight fails, promote L in THIS order
C_GRID = (0.01, 0.1, 1.0, 10.0)   # LogisticRegression C, selected by 5-fold CV on DISCOVERY only
DATA_EFF_SIZES = (25, 50, 100, 200, 500)  # data-efficiency train sizes (+ all-discovery)
PROBE_SEEDS = (0, 1, 2)
SHUFFLE_NULL_SEEDS = (301, 302, 303, 304, 305)
BOOTSTRAP_SEED = 7
N_BOOT = 10000
SHUFFLE_NULL_TOL = 0.03           # shuffled-null must be ≤ CHANCE + this


# ===========================================================================
# Label / null plumbing.
# ===========================================================================

def shuffled_labels(y, seed: int):
    """Permuted labels for the memorization null (prereg §5). A probe that fits these as well as the real
    labels on a HELD-OUT set is memorizing; a linear probe cannot, so shuffled → chance on held-out."""
    y = np.asarray(y)
    return np.random.default_rng(seed).permutation(y)


def accuracy(pred, gold) -> float:
    pred = np.asarray(pred); gold = np.asarray(gold)
    if pred.shape != gold.shape:
        raise ValueError(f"accuracy: pred/gold shape mismatch {pred.shape} vs {gold.shape}")
    return float((pred == gold).mean())


# ===========================================================================
# Gap arithmetic (prereg §2) — the three decodability gaps.
# ===========================================================================

def masking_gap(acc_deleted_L16: float, acc_deleted_mcq: float) -> float:
    """Is the answer more decodable from the residual than the model's saturated readout expresses?"""
    return acc_deleted_L16 - acc_deleted_mcq


def computation_gap(acc_deleted_L16: float, acc_deleted_floor: float) -> float:
    """Did mid-stack add answer signal beyond the surface-prompt floor layer?"""
    return acc_deleted_L16 - acc_deleted_floor


def ceiling_gap(acc_intact_L16: float, acc_deleted_L16: float) -> float:
    """How much of the intact model's L16 answer signal is missing in the deleted model?"""
    return acc_intact_L16 - acc_deleted_L16


# ===========================================================================
# Item-level bootstrap (prereg §7). Accuracies + paired gaps.
# ===========================================================================

def bootstrap_acc(correct, seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT):
    """Item-bootstrap 95% CI of a held-out accuracy from a per-item 0/1 array."""
    c = np.asarray(correct, float)
    n = len(c)
    if n == 0:
        raise ValueError("bootstrap_acc: empty")
    pt = float(c.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = c[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pt, float(lo), float(hi)


def bootstrap_gap(a_correct, b_correct, seed: int = BOOTSTRAP_SEED, n_boot: int = N_BOOT):
    """Paired item-bootstrap 95% CI of acc(a) − acc(b) (the same resample indices each draw). a,b are
    per-item 0/1 arrays over the SAME items (e.g. deleted-L16 vs deleted-floor)."""
    a = np.asarray(a_correct, float); b = np.asarray(b_correct, float)
    n = len(a)
    if len(b) != n:
        raise ValueError(f"bootstrap_gap: a/b must be item-paired ({len(a)} vs {len(b)})")
    pt = float(a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for bi in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[bi] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pt, float(lo), float(hi)


# ===========================================================================
# Data-efficiency + readout-alignment (prereg §5).
# ===========================================================================

def saturation_index(acc_by_size: dict, small_key: int = 100) -> float:
    """acc@small / acc@all — near 1 ⇒ a PRESERVED linear signal read cheaply (H_mask); ≪ 1 and climbing ⇒
    a RE-LEARNED head (REPRICE-class). acc_by_size maps train_size→held-out acc, with key 'all' for the
    full-discovery fit."""
    all_acc = acc_by_size.get("all")
    small = acc_by_size.get(small_key)
    if all_acc is None or small is None or all_acc <= CHANCE:
        return float("nan")
    return float((small - CHANCE) / (all_acc - CHANCE))


def readout_alignment(probe_W, unembed_rows) -> dict:
    """Cosine of each probe class-direction with the corresponding choice-letter unembedding row (prereg
    §5, threat 5). probe_W: (n_classes, d) class weight vectors; unembed_rows: (n_classes, d) the model's
    choice-letter `lm_head` rows. High alignment ⇒ the decoded direction is the one the model's OWN readout
    uses (masking is at the used readout, not an orthogonal decodable-but-unused signal). Returns per-class
    + mean cosine and a random-direction null mean."""
    W = np.asarray(probe_W, float); U = np.asarray(unembed_rows, float)
    if W.shape != U.shape:
        raise ValueError(f"readout_alignment: shape mismatch {W.shape} vs {U.shape}")
    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
    per = [abs(_cos(W[i], U[i])) for i in range(W.shape[0])]
    rng = np.random.default_rng(0)
    null = [abs(_cos(rng.standard_normal(W.shape[1]), U[i])) for i in range(U.shape[0])]
    return {"per_class_abscos": per, "mean_abscos": float(np.mean(per)),
            "random_null_mean": float(np.mean(null))}


# ===========================================================================
# Closed-form numpy LINEAR probe — for the WEIGHT-FREE dry-run only (the real probe is sklearn in the
# cell). Ridge one-hot least-squares: linear, deterministic, no sklearn. Validates the DESIGN LOGIC.
# ===========================================================================

def _fit_ridge_onehot(X, y, n_classes: int, ridge: float = 1.0):
    """W = (XᵀX + ridge·I)⁻¹ Xᵀ Y_onehot on bias-augmented X. Returns W (d+1, n_classes)."""
    X = np.asarray(X, float); y = np.asarray(y, int)
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    Y = np.eye(n_classes)[y]
    A = Xa.T @ Xa + ridge * np.eye(Xa.shape[1])
    return np.linalg.solve(A, Xa.T @ Y)


def probe_accuracy_np(Xtr, ytr, Xte, yte, n_classes: int, ridge: float = 1.0):
    """Fit the ridge one-hot probe on (Xtr,ytr), return (held-out accuracy, per-item correct on Xte)."""
    W = _fit_ridge_onehot(Xtr, ytr, n_classes, ridge)
    Xte_a = np.hstack([np.asarray(Xte, float), np.ones((len(Xte), 1))])
    pred = np.argmax(Xte_a @ W, axis=1)
    correct = (pred == np.asarray(yte, int)).astype(int)
    return float(correct.mean()), correct


class SyntheticProbeStack:
    """Tiny synthetic for the dry-run (prereg §13). Class-separable Gaussian blobs at a controllable signal
    level: 'ceiling' (well separated → high decodability), 'deleted' (same as ceiling if MASK; near floor
    if LOSS — the two modes), 'floor' (weak signal → near chance). A linear probe on a held-out split must
    recover: floor ≈ chance < signal; shuffled labels → chance at every signal level."""

    def __init__(self, d=64, n=1200, n_classes=4, signal=3.0, seed=0):
        rng = np.random.default_rng(seed)
        self.n_classes = n_classes
        centers = rng.standard_normal((n_classes, d))
        self.y = rng.integers(0, n_classes, n)
        self.X = signal * centers[self.y] + rng.standard_normal((n, d))
        # a fixed disc/conf split (hash-parity analog)
        self.disc = np.arange(n) % 2 == 0
        self.conf = ~self.disc

    def held_out_acc(self, ridge=1.0, labels=None):
        y = self.y if labels is None else labels
        return probe_accuracy_np(self.X[self.disc], y[self.disc], self.X[self.conf], y[self.conf],
                                 self.n_classes, ridge)


# ===========================================================================
# Weight-free dry-run (prereg §13) — RAISES on any failed invariant (fail-loud).
# ===========================================================================

def dry_run(verbose: bool = False) -> dict:
    out: dict = {"checks": {}}

    # ---- Check 1: gap arithmetic ----
    assert abs(masking_gap(0.42, 0.24) - 0.18) < 1e-9
    assert abs(computation_gap(0.42, 0.27) - 0.15) < 1e-9
    assert abs(ceiling_gap(0.46, 0.42) - 0.04) < 1e-9
    out["checks"]["gap_arithmetic"] = "pass"

    # ---- Check 2: probe LOGIC — a separable signal decodes; shuffled labels collapse to chance;
    #      floor (weak signal) ≈ chance < ceiling (strong signal). This is the whole design in a nutshell. ----
    ceil = SyntheticProbeStack(signal=3.0, seed=1)
    acc_ceil, corr_ceil = ceil.held_out_acc()
    floor = SyntheticProbeStack(signal=0.0, seed=1)
    acc_floor, _ = floor.held_out_acc()
    acc_shuf = np.mean([ceil.held_out_acc(labels=shuffled_labels(ceil.y, s))[0] for s in SHUFFLE_NULL_SEEDS])
    assert acc_ceil > 0.80, f"separable signal should decode (got {acc_ceil:.3f})"
    assert acc_floor < CHANCE + 0.10, f"weak-signal floor should be ~chance (got {acc_floor:.3f})"
    assert abs(acc_shuf - CHANCE) < 0.06, f"shuffled null should be ~chance (got {acc_shuf:.3f})"
    assert (acc_ceil - acc_floor) > DELTA_DECODE, "ceiling must separate from floor by > δ_decode"
    out["checks"]["probe_logic"] = {"ceiling": round(acc_ceil, 3), "floor": round(acc_floor, 3),
                                    "shuffled_null": round(float(acc_shuf), 3)}

    # ---- Check 3: the mask-vs-loss SEPARATION — a 'deleted' stack that is (a) fully present decodes like
    #      the ceiling (H_mask), (b) absent decodes like the floor (H_loss). Same probe, opposite verdicts. ----
    deleted_mask = SyntheticProbeStack(signal=3.0, seed=2)      # answer present
    deleted_loss = SyntheticProbeStack(signal=0.0, seed=2)      # answer absent
    a_mask, c_mask = deleted_mask.held_out_acc()
    a_loss, c_loss = deleted_loss.held_out_acc()
    _, floor_corr = SyntheticProbeStack(signal=0.0, seed=3).held_out_acc()
    # H_mask signature: deleted ≈ ceiling, computation gap > δ
    g_mask = a_mask - a_loss   # proxy for computation gap using the loss-stack as floor
    assert a_mask > CHANCE + DELTA_DECODE and g_mask > DELTA_DECODE, "mask stack should decode ≫ floor"
    # H_loss signature: deleted ≈ floor, ceiling gap ≫ δ
    assert (acc_ceil - a_loss) > DELTA_DECODE, "loss stack should be ≪ ceiling"
    out["checks"]["mask_vs_loss_separation"] = {"deleted_mask": round(a_mask, 3), "deleted_loss": round(a_loss, 3)}

    # ---- Check 4: bootstrap brackets; paired gap CI ----
    pt, lo, hi = bootstrap_acc(corr_ceil, n_boot=2000)
    assert lo <= pt <= hi and lo > CHANCE, f"ceiling acc CI should clear chance (CI[{lo:.3f},{hi:.3f}])"
    gpt, glo, ghi = bootstrap_gap(c_mask, floor_corr, n_boot=2000)
    assert glo <= gpt <= ghi, "gap CI must bracket the point"
    out["checks"]["bootstrap"] = {"ceiling_ci": [round(lo, 3), round(hi, 3)], "gap_ci_lo": round(glo, 3)}

    # ---- Check 5: data-efficiency — a preserved (separable) signal saturates by few examples; a
    #      memorization-only (random-label) signal never rises above chance at any train size. ----
    big = SyntheticProbeStack(signal=3.0, n=2000, seed=4)
    def acc_at(ntr, labels):
        idx = np.where(big.disc)[0][:ntr]
        return probe_accuracy_np(big.X[idx], labels[idx], big.X[big.conf], labels[big.conf], big.n_classes)[0]
    acc_by_size = {k: acc_at(k, big.y) for k in (50, 100)}
    acc_by_size["all"] = big.held_out_acc()[0]
    sat = saturation_index(acc_by_size, small_key=100)
    assert sat > 0.7, f"preserved signal should saturate early (sat={sat})"
    rand_y = np.random.default_rng(9).integers(0, big.n_classes, len(big.y))
    assert acc_at(500, rand_y) < CHANCE + 0.06, "random-label (memorization-only) never beats chance held-out"
    out["checks"]["data_efficiency"] = {"saturation_index": round(sat, 3)}

    # ---- Check 6: readout-alignment — aligned probe scores ~1, orthogonal ~0 ----
    d = 32
    U = np.eye(4, d)                      # 4 readout rows = first 4 basis vectors
    W_aligned = U + 0.01 * np.random.default_rng(0).standard_normal((4, d))
    W_orth = np.zeros((4, d)); W_orth[:, 4:8] = np.eye(4)   # orthogonal to U
    assert readout_alignment(W_aligned, U)["mean_abscos"] > 0.9, "aligned probe should score high"
    assert readout_alignment(W_orth, U)["mean_abscos"] < 0.1, "orthogonal probe should score ~0"
    out["checks"]["readout_alignment"] = "pass"

    if verbose:
        import json
        print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(dry_run(), indent=2))
