"""HELMHOLTZ confirmatory core — pure logic, weight-free dry-runnable.

The question: does a model-diff (representation-similarity AND a crosscoder feature
counter) report a checkpoint's representation as PRESERVED while its MMLU capability
is destroyed — and is the diff SENSITIVE (positive control) so "preserved" is
meaningful rather than coarse?

Pure numpy only; no torch, no model. Run `python -m infra.pg.helmholtz_core` for the
weight-free self-test (the Phase-2 dry-run gate). Fail-loud: asserts, no try/except.
"""
import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear CKA between two (n_tokens, d) activation matrices on PAIRED
    tokens. 1.0 = representations identical up to linear transform; ~0 = unrelated.
    The primary, non-degenerate 'is the representation preserved?' signal."""
    assert X.shape == Y.shape and X.ndim == 2, f"shape {X.shape} vs {Y.shape}"
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    assert xx > 0 and yy > 0, "degenerate (zero-variance) activations"
    return float(xy / (xx * yy))


def blindness_index(behavior_drop: float, repr_change: float, eps: float = 1e-6) -> float:
    """behavior_drop = fraction of MMLU capability lost (0..1);
    repr_change = 1 - representation similarity (0..1).
    High => a large behavioral loss registered as a small representation change =
    the diff is blind. Reported with both raw components, never alone."""
    assert 0.0 <= behavior_drop <= 1.0, behavior_drop
    assert 0.0 <= repr_change <= 1.0 + 1e-9, repr_change
    return float(behavior_drop / (repr_change + eps))


def preservation_verdict(survival: float, lost: float, emergent: float,
                         identity_band: dict) -> dict:
    """Crosscoder feature-counter verdict against the identity-pair band (the null =
    what the tool manufactures on two copies of the same model)."""
    moved = (lost > identity_band["lost"]) or (emergent > identity_band["emergent"])
    return {"survival": float(survival), "lost": float(lost), "emergent": float(emergent),
            "verdict": "MOVED" if moved else "PRESERVED-INSIDE-IDENTITY-BAND"}


def positive_control_ok(planted_lost: float, planted_emergent: float,
                        identity_band: dict) -> bool:
    """The crosscoder MUST register a PLANTED change (diff_metrics.plant_direction)
    as MOVED. If it does not, the instrument is coarse => VOID-INSTRUMENT and the
    'preserved' verdict on the real pair is uninformative (H_coarse guard)."""
    return bool((planted_lost > identity_band["lost"]) or
                (planted_emergent > identity_band["emergent"]))


def cka_positive_control_ok(cka_planted: float, thresh: float = 0.9) -> bool:
    """CKA must DROP below `thresh` for a planted change of realistic magnitude. This is the
    sensitivity gate for the primary (CKA) instrument: if CKA cannot see a planted change, a
    high CKA on the real pair would be uninformative (coarse). Replaces the crosscoder-plant
    positive control, which the 2026-09-01 wiring smoke showed is void (a crosscoder trained on
    the pair learns a shared code for the plant — dated amendment, pre-confirmatory-data)."""
    return bool(cka_planted < thresh)


def adjudicate(cka_real: float, cka_planted: float, behavior_drop_mcq: float,
               behavior_drop_cloze: float, cross_real: dict, identity_band: dict,
               fvu: float) -> dict:
    """Fold the arm into one verdict. Frozen decision rule (§6 + 2026-09-01 amendment):
    CKA is the PRIMARY diff instrument; its positive control is the CKA-plant drop. The
    crosscoder feature-counter is reported as corroboration only (its own positive control is
    void — see cka_positive_control_ok docstring).
    - H_coarse : CKA positive control FAILS (CKA blind to a planted change) -> primary VOID.
    - H_lost   : cloze also destroyed (behavior_drop_cloze high).
    - H_skill  : CKA high (representation preserved) AND MCQ destroyed AND cloze retained.
    """
    cka_pc = cka_positive_control_ok(cka_planted)
    real = preservation_verdict(cross_real["survival"], cross_real["lost"],
                                cross_real["emergent"], identity_band)
    bi = blindness_index(behavior_drop_mcq, 1.0 - cka_real)
    if not cka_pc:
        verdict = "H_coarse / VOID-INSTRUMENT (CKA blind to a planted change)"
    elif behavior_drop_cloze > 0.5:
        verdict = "H_lost (cloze also destroyed — knowledge genuinely gone)"
    elif behavior_drop_mcq > 0.5 and behavior_drop_cloze < 0.2 and cka_real > 0.9:
        verdict = "H_skill (representation preserved, MCQ skill destroyed, diff blind)"
    else:
        verdict = "AMBIGUOUS (see components)"
    return {"verdict": verdict, "cka_real": cka_real, "cka_planted": cka_planted,
            "cka_positive_control_fires": cka_pc, "blindness_index": bi,
            "mcq_drop": behavior_drop_mcq, "cloze_drop": behavior_drop_cloze,
            "crosscoder_real_corroboration": real, "fvu": fvu}


def _dryrun():
    """Weight-free self-test on tiny synthetic tensors. THE Phase-2 gate."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((500, 64))
    # CKA: identical -> 1; tiny noise -> high; independent -> low.
    assert abs(linear_cka(A, A) - 1.0) < 1e-9, "CKA(A,A) must be 1"
    cka_noise = linear_cka(A, A + 0.01 * rng.standard_normal(A.shape))
    assert cka_noise > 0.99, f"CKA under tiny noise should stay high, got {cka_noise}"
    cka_indep = linear_cka(A, rng.standard_normal(A.shape))
    assert cka_indep < 0.2, f"CKA of independent should be low, got {cka_indep}"
    # blindness_index: big behavior drop + tiny repr change -> large; no drop -> 0.
    assert blindness_index(1.0, 0.02) > 20, "blind case must score high"
    assert blindness_index(0.0, 0.02) == 0.0, "no drop -> 0"
    # preservation verdict + positive control against an identity band.
    band = {"lost": 0.005, "emergent": 0.005}
    assert preservation_verdict(1.0, 0.001, 0.0, band)["verdict"].startswith("PRESERVED")
    assert preservation_verdict(0.9, 0.05, 0.0, band)["verdict"] == "MOVED"
    assert positive_control_ok(0.20, 0.0, band) is True, "plant must be detected"
    assert positive_control_ok(0.001, 0.001, band) is False, "sub-band plant = void"
    # CKA positive control: a planted change must drop CKA below threshold.
    assert cka_positive_control_ok(0.5) is True, "CKA must detect a big planted change"
    assert cka_positive_control_ok(0.999) is False, "CKA unchanged by plant = void"
    # full adjudication (CKA-primary): H_skill, H_coarse guard, H_lost.
    band0 = {"lost": 0.005, "emergent": 0.005}
    real0 = {"survival": 1.0, "lost": 0.001, "emergent": 0.0}
    skill = adjudicate(cka_real=0.98, cka_planted=0.55, behavior_drop_mcq=0.95,
                       behavior_drop_cloze=0.05, cross_real=real0, identity_band=band0, fvu=0.06)
    assert skill["verdict"].startswith("H_skill"), skill["verdict"]
    coarse = adjudicate(cka_real=0.98, cka_planted=0.995, behavior_drop_mcq=0.95,
                        behavior_drop_cloze=0.05, cross_real=real0, identity_band=band0, fvu=0.06)
    assert coarse["verdict"].startswith("H_coarse"), coarse["verdict"]
    lost = adjudicate(cka_real=0.6, cka_planted=0.55, behavior_drop_mcq=0.95,
                      behavior_drop_cloze=0.9, cross_real=real0, identity_band=band0, fvu=0.06)
    assert lost["verdict"].startswith("H_lost"), lost["verdict"]
    print("HELMHOLTZ core dry-run: ALL PASS")
    print(f"  CKA identical=1.0 noise={cka_noise:.4f} indep={cka_indep:.4f}")
    print(f"  H_skill={skill['verdict']}\n  H_coarse={coarse['verdict']}\n  H_lost={lost['verdict']}")


if __name__ == "__main__":
    _dryrun()
