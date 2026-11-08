"""HELMHOLTZ-2 pure core — weight-free logic for the shipped-SAE diffing discrimination test.

Spec: studies/lineage/cross-lineage/research/2026-09-01_helmholtz2_spec.md.
Three arms (B benign / D_know content-damage / D_skill readout-damage) vs one base; per
instrument I the readout is TWO separations Δ_I^know, Δ_I^skill and the dissociation
Δ_I^know − Δ_I^skill. This module holds every decision rule and statistic that does not
need a model: JumpReLU encode, per-feature firing stats, the SAE diff summary, identity
bands from split halves, spectrum-scaled plant dose, bootstrap CIs, fires(), and the
frozen discrimination verdict. Pure numpy; asserts, no try/except.

Run `python -m infra.pg.helmholtz2_core` for the dry-run gate (Phase-2 #13).
"""
import numpy as np

from infra.pg.helmholtz_core import linear_cka  # re-export: validated in HELMHOLTZ-1


# ---------------------------------------------------------------- JumpReLU SAE math
def jumprelu_encode(x: np.ndarray, W_enc: np.ndarray, b_enc: np.ndarray,
                    threshold: np.ndarray) -> np.ndarray:
    """Gemma Scope JumpReLU encode: pre = x @ W_enc + b_enc; code = pre * (pre > threshold).
    x (n, d); W_enc (d, F); threshold (F,) learned per-feature. Returns (n, F) codes."""
    assert x.ndim == 2 and W_enc.shape[0] == x.shape[1], (x.shape, W_enc.shape)
    pre = x @ W_enc + b_enc
    return pre * (pre > threshold)


def sae_decode(codes: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    return codes @ W_dec + b_dec


def fvu_np(x: np.ndarray, recon: np.ndarray) -> float:
    """Fraction of variance unexplained vs the dataset-mean baseline (Gemma Scope §4.1)."""
    assert x.shape == recon.shape
    num = float(((x - recon) ** 2).sum())
    den = float(((x - x.mean(0, keepdims=True)) ** 2).sum())
    assert den > 0, "degenerate activations"
    return num / den


class FiringAccumulator:
    """Streaming per-feature firing stats over token chunks (codes never materialized whole).
    rate_j = P(code_j > 0); mag_j = mean code_j over firing tokens (0 if never fires)."""

    def __init__(self, n_features: int):
        self.n = 0
        self.fire_count = np.zeros(n_features, dtype=np.int64)
        self.mag_sum = np.zeros(n_features, dtype=np.float64)

    def add(self, codes: np.ndarray):
        assert codes.ndim == 2 and codes.shape[1] == self.fire_count.shape[0]
        firing = codes > 0
        self.n += codes.shape[0]
        self.fire_count += firing.sum(0)
        self.mag_sum += np.where(firing, codes, 0.0).sum(0)

    def rates(self) -> np.ndarray:
        assert self.n > 0
        return self.fire_count / self.n

    def mags(self) -> np.ndarray:
        return np.where(self.fire_count > 0, self.mag_sum / np.maximum(self.fire_count, 1), 0.0)


# ---------------------------------------------------------------- SAE diff summary
def sae_diff_from_rates(rate_base: np.ndarray, rate_other: np.ndarray, tau: float) -> dict:
    """The shipped-SAE diff summary between two firing-rate vectors on IDENTICAL inputs.
    tau = minimum firing rate to count a feature as 'active' (frozen from the identity band).
    survival = active-in-base still active in other; lost = active-in-base now inactive;
    emergent = inactive-in-base now active (denominator: base-active count, Scope convention
    n_emergent/n_base_active so it is comparable to lost); dlog = mean |Δ log10 rate| over
    features active on BOTH sides (the graded shift signal)."""
    assert rate_base.shape == rate_other.shape and rate_base.ndim == 1
    act_b = rate_base > tau
    act_o = rate_other > tau
    n_b = int(act_b.sum())
    assert n_b > 0, "no active features on the base side — tau too high or SAE dead"
    surv = act_b & act_o
    lost = act_b & ~act_o
    emergent = ~act_b & act_o
    both = surv
    dlog = float(np.abs(np.log10(rate_base[both]) - np.log10(rate_other[both])).mean()) \
        if both.any() else float("nan")
    return {"n_base_active": n_b,
            "survival": float(surv.sum() / n_b),
            "lost": float(lost.sum() / n_b),
            "emergent": float(emergent.sum() / n_b),
            "dlog_rate": dlog}


def identity_band_from_split(rate_h1: np.ndarray, rate_h2: np.ndarray, tau: float) -> dict:
    """Sampling-noise band: the SAME model's rates on two disjoint item halves. Whatever
    lost/emergent/dlog this manufactures is estimation noise, not change; every 'fires'
    judgement must clear it. Returns the band = the identity-split diff summary."""
    d = sae_diff_from_rates(rate_h1, rate_h2, tau)
    return {"lost": d["lost"], "emergent": d["emergent"], "dlog_rate": d["dlog_rate"],
            "survival_floor": d["survival"]}


# ---------------------------------------------------------------- plant dose
def plant_alpha_from_spectrum(acts: np.ndarray, c: float = 1.0) -> float:
    """Spectrum-scaled plant magnitude: alpha = c * E||x - mean(x)||_2 (the typical
    per-token deviation norm). Replaces the successor's uncalibrated default that
    under-dosed on high-variance activations. c frozen: 1.0 primary; {0.5, 2.0} curve."""
    assert acts.ndim == 2 and acts.shape[0] > 1
    dev = acts - acts.mean(0, keepdims=True)
    return float(c * np.linalg.norm(dev, axis=1).mean())


# ---------------------------------------------------------------- bootstrap + fires
def bootstrap_mean_diff(per_unit_a: np.ndarray, per_unit_b: np.ndarray,
                        n_boot: int = 10_000, seed: int = 0) -> dict:
    """CI on mean(a) − mean(b) with independent resampling of each side's units
    (items or item-blocks). Returns {diff, lo, hi} at 95%."""
    a = np.asarray(per_unit_a, dtype=np.float64)
    b = np.asarray(per_unit_b, dtype=np.float64)
    assert a.ndim == 1 and b.ndim == 1 and len(a) > 1 and len(b) > 1
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = a[ia].mean(1) - b[ib].mean(1)
    return {"diff": float(a.mean() - b.mean()),
            "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975))}


def fires(ci: dict, band_width: float) -> bool:
    """Frozen 'instrument fires' rule (§6): CI excludes 0 AND |effect| exceeds the
    identity-band width for that statistic."""
    excl0 = (ci["lo"] > 0) or (ci["hi"] < 0)
    return bool(excl0 and abs(ci["diff"]) > band_width)


# ---------------------------------------------------------------- frozen verdicts
def gate_dose_know(mcq_margin_drop: float, ce_final_over_initial: float) -> bool:
    """G1k: content-damage dose qualifies."""
    return bool(mcq_margin_drop >= 0.5 and ce_final_over_initial < 0.8)


def gate_dose_skill(mcq_margin_drop: float, ce_final_over_initial: float,
                    cloze_margin_drop: float) -> bool:
    """G1s: readout-damage dose qualifies (adds the readout-purity condition)."""
    return bool(mcq_margin_drop >= 0.5 and ce_final_over_initial < 0.8
                and cloze_margin_drop <= 0.15)


def gate_benign(mcq_margin_drop: float) -> bool:
    """G2: the matched benign arm must stay benign."""
    return bool(mcq_margin_drop < 0.1)


def discrimination_verdict(sae_fires_know: bool, sae_fires_skill: bool,
                           cka_fires_know: bool, cka_fires_skill: bool,
                           beh_fires_know: bool, beh_fires_skill: bool,
                           cloze_drop_know: float, cloze_drop_skill: float,
                           plants_valid: bool) -> str:
    """§6 frozen mapping from the fires-table to the hypothesis tree. sae_fires_* must
    already be aggregated over the swept sites/widths per §6 (fires ⟺ fires ANYWHERE;
    silent ⟺ silent EVERYWHERE). Order of precedence: VOID > gates > tree."""
    if not plants_valid:
        return "VOID-INSTRUMENT (a plant control failed; no 'no change' verdict is valid)"
    if not (beh_fires_know and beh_fires_skill):
        return "GATE-FAIL (behavior does not separate a D arm from B — dose gates should have caught this)"
    skill_pure = cloze_drop_skill <= 0.15
    know_content = cloze_drop_know > 0.15
    if sae_fires_know and not sae_fires_skill and skill_pure and know_content:
        return "H_skill-blind (SAE sees content damage, misses readout damage)"
    if not sae_fires_know and not sae_fires_skill:
        if cka_fires_know or cka_fires_skill:
            return "H_blind (SAE separates neither arm; CKA separates)"
        return "H_repr-blind (nothing at the representation level separates; only behavior)"
    if sae_fires_know and sae_fires_skill:
        return "H_detects (the shipped SAE separates both damage kinds — blindness does not scale)"
    return "MIXED (SAE fires on D_skill but not D_know, or purity failed — outside the tree; report components)"


# ---------------------------------------------------------------- dry-run gate
def _dryrun():
    rng = np.random.default_rng(0)
    n, d, F = 400, 32, 128
    # JumpReLU encode/decode + FVU: a random SAE reconstructs badly (high FVU), an
    # identity-ish SAE (W_enc = W_dec.T = I-padded) reconstructs its span (low FVU).
    x = rng.standard_normal((n, d))
    W_enc = np.zeros((d, F)); W_enc[:d, :d] = np.eye(d)
    W_dec = W_enc.T.copy()
    b_enc = np.zeros(F); b_dec = np.zeros(d); thr = np.full(F, -1e9)  # always pass
    codes = jumprelu_encode(x, W_enc, b_enc, thr)
    assert codes.shape == (n, F)
    recon = sae_decode(codes, W_dec, b_dec)
    assert fvu_np(x, recon) < 1e-12, "identity SAE must reconstruct perfectly"
    thr_hi = np.full(F, 1e9)
    dead = jumprelu_encode(x, W_enc, b_enc, thr_hi)
    assert (dead == 0).all(), "threshold above pre-acts must kill all codes"
    # FiringAccumulator streaming == batch
    acc = FiringAccumulator(F)
    for s in range(0, n, 64):
        acc.add(codes[s:s + 64])
    assert acc.n == n and np.allclose(acc.rates(), (codes > 0).mean(0))
    # diff summary: identical rates -> survival 1, lost 0; kill half -> survival .5
    r1 = np.clip(rng.uniform(0, 0.3, F), 1e-4, None)
    same = sae_diff_from_rates(r1, r1, tau=1e-3)
    assert same["survival"] == 1.0 and same["lost"] == 0.0 and same["dlog_rate"] == 0.0
    r2 = r1.copy(); act = r1 > 1e-3; kill = np.where(act)[0][: act.sum() // 2]; r2[kill] = 0.0
    half = sae_diff_from_rates(r1, r2, tau=1e-3)
    assert abs(half["survival"] - 0.5) < 0.02 and abs(half["lost"] - 0.5) < 0.02, half
    # identity band from a noisy split is small but nonzero
    r1n = np.clip(r1 + rng.normal(0, 0.002, F), 1e-5, None)
    band = identity_band_from_split(r1, r1n, tau=1e-3)
    assert band["lost"] < 0.1 and band["survival_floor"] > 0.9, band
    # plant alpha: for iid N(0,1) rows, E||x-mean|| ~ sqrt(d)
    a1 = plant_alpha_from_spectrum(x, c=1.0)
    assert abs(a1 - np.sqrt(d)) / np.sqrt(d) < 0.15, (a1, np.sqrt(d))
    assert abs(plant_alpha_from_spectrum(x, c=2.0) - 2 * a1) < 1e-9
    # bootstrap + fires
    ci_big = bootstrap_mean_diff(rng.normal(1.0, 0.1, 200), rng.normal(0.0, 0.1, 200), 2000, 1)
    assert fires(ci_big, band_width=0.2), ci_big
    ci_null = bootstrap_mean_diff(rng.normal(0.0, 0.1, 200), rng.normal(0.0, 0.1, 200), 2000, 2)
    assert not fires(ci_null, band_width=0.2), ci_null
    ci_sub_band = bootstrap_mean_diff(rng.normal(0.1, 0.01, 200), rng.normal(0.0, 0.01, 200), 2000, 3)
    assert not fires(ci_sub_band, band_width=0.2), "CI excludes 0 but effect under band -> not fires"
    # gates
    assert gate_dose_know(0.6, 0.5) and not gate_dose_know(0.4, 0.5) and not gate_dose_know(0.6, 0.9)
    assert gate_dose_skill(0.6, 0.5, 0.1) and not gate_dose_skill(0.6, 0.5, 0.3)
    assert gate_benign(0.05) and not gate_benign(0.2)
    # verdicts — one per tree branch + void + mixed
    v = discrimination_verdict(True, False, True, False, True, True, 0.6, 0.05, True)
    assert v.startswith("H_skill-blind"), v
    v = discrimination_verdict(False, False, True, False, True, True, 0.6, 0.05, True)
    assert v.startswith("H_blind"), v
    v = discrimination_verdict(False, False, False, False, True, True, 0.6, 0.05, True)
    assert v.startswith("H_repr-blind"), v
    v = discrimination_verdict(True, True, True, True, True, True, 0.6, 0.05, True)
    assert v.startswith("H_detects"), v
    v = discrimination_verdict(True, False, True, False, True, True, 0.6, 0.5, True)
    assert v.startswith("MIXED"), v
    v = discrimination_verdict(True, False, True, False, True, True, 0.6, 0.05, False)
    assert v.startswith("VOID"), v
    v = discrimination_verdict(True, False, True, False, True, False, 0.6, 0.05, True)
    assert v.startswith("GATE-FAIL"), v
    print("HELMHOLTZ-2 core dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
