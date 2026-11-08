"""INSTRUMENTS E1 first-order prediction (studies/instruments/SPEC_instruments.md §3.2, R4; §10). unite_route is imported unchanged.

The coefficient. unite_route.attribution gives A_k(x) = sum over the unit's terms of w sign a_k(x) s_l(x) [c_g(x) G.d_k + c_u(x) U.d_k]. The same
function on the same terms with every a_k set to 1 gives c_k(x) = sum over terms of w sign s_l(x) [c_g G.d_k + c_u U.d_k], so A = a ⊙ c by
construction. The check: max |A - a ⊙ c| / max |A| <= 1e-5 at every context of every read set (asserted). Its failing input: c built with each
layer's scale s_l taken from the next layer (l + 1 mod L), which must miss by more than 1e-5.
The prediction. Replacing a_k(x) by v_k for k in S moves a_P by sum_{k in S} (v_k - a_k(x)) c_k(x) to first order (the direct path, s_l and the
coefficients held). In the sign of the measured fall (a_P before - a_P after):
    pred_fall(x) = sum_{k in S} (a_k(x) - v_k) c_k(x)          (units of a_P)
A side's predicted fall is reuse_ablate.job_fall's convention: the mean over the side's contexts with a_P > 0 of pred_fall(x) / a_P(x).
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import numpy as np
import torch

from infra.pg import unite_route as UR

C_TOL = 1e-5


def c_error(A: torch.Tensor, a: torch.Tensor, c: torch.Tensor) -> float:
    """max |A - a c| / max |A|."""
    return float((A - a * c).abs().max() / A.abs().max().clamp_min(1e-30))


def _ones(cap: dict, L: int, shift: int = 0) -> dict:
    return {"a": {l: torch.ones_like(cap["a"][l]) for l in range(L)}, "s": {l: cap["s"][(l + shift) % L] for l in range(L)}, "emb": cap["emb"]}


@torch.inference_mode()
def coefficients(rows: UR.Rows, cap: dict, unit: dict, W: int, A: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """c [N, L W] for the unit at the captured contexts, from unite_route.attribution with a = 1; asserts A (reuse_ablate.direct_A on the same
    capture) equals a ⊙ c to C_TOL. Returns c and the check's record (value, failing input value)."""
    L = unit["layer"]
    gc, uc = UR.term_gu(cap, unit)
    terms = UR.unit_terms(rows, unit, gc, uc)
    c = UR.attribution(_ones(cap, L), L, terms, W)
    a = torch.cat([cap["a"][l] for l in range(L)], 1)
    err = c_error(A, a, c)
    err_fail = c_error(A, a, UR.attribution(_ones(cap, L, shift=1), L, terms, W)) if L > 1 else float("nan")
    rec = {"value": err, "failing_input": err_fail, "tol": C_TOL}
    assert err <= C_TOL, (unit["unit"], err, "A is not a ⊙ c")
    return c, rec


def pred_fall(a, c, v) -> np.ndarray:
    """[..., N]: sum_k (a_k - v_k) c_k over the last axis. a, c [N, n]; v [n] or [R, n] (R written tables)."""
    a, c, v = (np.asarray(x, np.float64) for x in (a, c, v))
    if v.ndim == 1:
        return ((a - v[None]) * c).sum(-1)
    return ((a[None] - v[:, None, :]) * c[None]).sum(-1)


def side_fall(pred, before) -> float:
    """job_fall's convention on a prediction: mean over contexts with a_P > 0 of pred / a_P; NaN when none is positive."""
    pred, before = np.asarray(pred, np.float64), np.asarray(before, np.float64)
    pos = before > 0
    return float((pred[..., pos] / before[pos]).mean()) if pos.any() else float("nan")


# ------------------------------------------------------------------------------------------ pure-logic checks on tiny tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    N, L, W, d = 4, 3, 5, 6
    t = lambda *s: torch.tensor(rng.normal(0, 1, s), dtype=torch.float32)
    cap = {"a": {l: t(N, W) for l in range(L)}, "s": {l: torch.tensor(rng.uniform(0.5, 2.0, N), dtype=torch.float32) for l in range(L)}, "emb": t(N, d)}
    terms = [(1.0, s, t(N), t(N), [(t(W), t(W)) for _ in range(L)]) for s in (1.0, -1.0)]
    A = UR.attribution(cap, L, terms, W)
    a = torch.cat([cap["a"][l] for l in range(L)], 1)
    c = UR.attribution(_ones(cap, L), L, terms, W)
    cs = UR.attribution(_ones(cap, L, shift=1), L, terms, W)
    e, ef = c_error(A, a, c), c_error(A, a, cs)
    assert e <= C_TOL < ef, (e, ef)
    out["coefficient"] = f"A = a ⊙ c to {e:.1e} on two terms of opposite sign; c with the next layer's scale misses by {ef:.2f}"
    # a unit linear in its upstream neurons: the replacement's fall equals the prediction exactly, in the measured fall's sign
    n = 6
    aa, cc, b0 = rng.normal(0, 1, (N, n)), rng.normal(0, 1, (N, n)), 5.0
    v = rng.normal(0, 1, n)
    before = (aa * cc).sum(1) + b0
    after = (np.where(np.arange(n) < 3, v[None], aa) * cc).sum(1) + b0
    p = pred_fall(aa[:, :3], cc[:, :3], v[:3])
    assert np.allclose(before - after, p), "the prediction is not the linear unit's fall"
    pr = pred_fall(aa[:, :3], cc[:, :3], np.stack([v[:3], 2 * v[:3]]))
    assert pr.shape == (2, N) and np.allclose(pr[0], p)
    assert pred_fall([[2.0]], [[3.0]], [0.5])[0] == 4.5, "a fall from a_k 2 to 0.5 with c 3 is +4.5"
    assert side_fall([1.0, 2.0, 9.0], [2.0, 4.0, -1.0]) == 0.5 and np.isnan(side_fall([1.0], [-1.0]))
    out["prediction"] = "on a linear unit the fall equals sum (a - v) c exactly; positive when the replacement lowers a_P; side mean over a_P > 0"
    return out
