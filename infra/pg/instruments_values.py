"""INSTRUMENTS E1 written values (studies/instruments/SPEC_instruments.md §3.2, §10). reuse_ablate and unite_route are imported unchanged.

The capture. X [P, L, W] fp32: each MLP neuron's a = act(g) u at the marked token of the window ending at each of the P = 200 background positions
(UNITE's e1 set), from unite_route.capture up to layer L - 1 with no g/u layers and the job's window length, BOS, batch and harvest window: the code
path of reuse_ablate.background_mean_a, which averages the same capture. Known number of the capture: X.mean(0) equals the job's ā to 1e-5 in
torch.allclose's rule |d| <= 1e-5 (1 + |ā|); its failing input is the mean over the first P/2 positions.
Written values per neuron, each a table [L, W]:
  mean      ā                                                     zero    0 (reuse_ablate.ablation_dict "zero")
  median    the median over the P positions (the mean of the two middle order statistics for an even P, numpy's convention)
  res<r>    resample draw r (seeds 0..3): per neuron one position index drawn uniformly from 0..P-1 by numpy default_rng(r).integers(0, P, (L, W)), the
            value X[index, l, k]. One table per draw: every set, context and window position >= the start of draw r gets that value (one draw per
            forward, written as the mean is). A neuron in two sets gets the same value in one draw.
The ablation dict {l: (idx, values)} of any table is reuse_ablate.ablation_dict(S, "mean", table, ...): the code that writes ā writes every table.
Resample known number: with X replaced by the constant ā (every position ā) each res<r> table is ā exactly, so the resample arm reproduces the mean
arm; with 2ā it must not.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import numpy as np
import torch

from infra.pg import reuse_ablate as RA
from infra.pg import unite_route as UR

SEEDS = (0, 1, 2, 3)
DRAWS = tuple(f"res{r}" for r in SEEDS)
VALUES = ("mean", "zero", "median") + DRAWS
MEAN_TOL = 1e-5


@torch.inference_mode()
def capture_background(model, positions, n_layers: int, *, tokens, length, bos, batch: int, hw: int, device) -> torch.Tensor:
    """X [P, n_layers, W] fp32: background_mean_a's capture, kept per position."""
    assert n_layers >= 1
    cap = UR.capture(model, np.asarray(positions), n_layers - 1, set(), tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    return torch.stack([cap["a"][l] for l in range(n_layers)], 1)


def _close(x: torch.Tensor, abar: torch.Tensor) -> tuple[bool, float, float]:
    d = (x - abar).abs()
    return bool((d <= MEAN_TOL * (1 + abar.abs())).all()), float(d.max()), float((d / (1 + abar.abs())).max())


def mean_check(X: torch.Tensor, abar: torch.Tensor) -> dict:
    """X.mean(0) against ā; failing input: the mean over the first half of the positions."""
    ok, mx, rel = _close(X.mean(0), abar)
    fok, fmx, frel = _close(X[: X.shape[0] // 2].mean(0), abar)
    return {"pass": ok, "max_abs": mx, "max_scaled": rel, "tol": MEAN_TOL, "rule": "|mean_P X - ā| <= 1e-5 (1 + |ā|) at every neuron",
            "failing_input": {"input": "the mean over the first half of the positions", "max_abs": fmx, "max_scaled": frel}, "failing_input_failed": not fok,
            "n_positions": int(X.shape[0])}


def median_table(X: torch.Tensor) -> torch.Tensor:
    s = X.sort(0).values
    P = X.shape[0]
    return s[P // 2] if P % 2 else 0.5 * (s[P // 2 - 1] + s[P // 2])


def draw_index(n_pos: int, shape: tuple, seeds=SEEDS) -> np.ndarray:
    """[R, L, W] position indices, one table per seed."""
    return np.stack([np.random.default_rng(r).integers(0, n_pos, shape) for r in seeds])


def resample_tables(X: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """[R, L, W]: X[idx[r, l, k], l, k]."""
    ii = torch.as_tensor(idx, device=X.device, dtype=torch.long)
    assert ii.shape[1:] == X.shape[1:], (tuple(ii.shape), tuple(X.shape))
    return torch.stack([X.gather(0, ii[r][None]).squeeze(0) for r in range(ii.shape[0])])


def tables(X: torch.Tensor, abar: torch.Tensor, seeds=SEEDS) -> tuple[dict, np.ndarray]:
    """{value: [L, W]} for VALUES, and the draw indices [R, L, W]."""
    assert X.shape[1:] == abar.shape, (tuple(X.shape), tuple(abar.shape))
    idx = draw_index(X.shape[0], tuple(X.shape[1:]), seeds)
    res = resample_tables(X, idx)
    out = {"mean": abar.float(), "zero": torch.zeros_like(abar, dtype=torch.float32), "median": median_table(X)}
    out.update({f"res{r}": res[i] for i, r in enumerate(seeds)})
    return out, idx


def constant_tables(value: torch.Tensor, n_pos: int, idx: np.ndarray) -> torch.Tensor:
    """The resample tables of a capture whose every position equals `value` [L, W] (the resample known number's input)."""
    return resample_tables(value.float()[None].expand(n_pos, *value.shape).contiguous(), idx)


def written(S: list, value: str, tab: dict, L: int, device, identity: bool = False) -> dict:
    """{l: (idx, values)} for reuse_ablate.replaced: zero through ablation_dict's own zero, every other table through its mean branch."""
    if value == "zero":
        return RA.ablation_dict(S, "zero", tab["mean"], L, device, identity=identity)
    return RA.ablation_dict(S, "mean", tab[value], L, device, identity=identity)


# ------------------------------------------------------------------------------------------ pure-logic checks on tiny tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    P, L, W = 200, 3, 7
    X = torch.tensor(rng.standard_t(2, (P, L, W)) * rng.uniform(0.1, 5, (1, L, W)), dtype=torch.float32)
    abar = X.mean(0)
    mc = mean_check(X, abar)
    assert mc["pass"] and mc["failing_input_failed"], mc
    out["mean_check"] = f"X.mean(0) = ā to {mc['max_abs']:.1e}; the first-half mean misses by {mc['failing_input']['max_abs']:.2f}"
    for n in (P, P - 1):
        assert np.allclose(median_table(X[:n]).numpy(), np.median(X[:n].numpy(), 0), rtol=0, atol=1e-6), n
    out["median"] = "equals numpy's median for an even and an odd count"
    idx = draw_index(P, (L, W))
    assert idx.shape == (len(SEEDS), L, W) and idx.min() >= 0 and idx.max() < P and (idx == draw_index(P, (L, W))).all() and (idx[0] != idx[1]).any()
    tab, idx2 = tables(X, abar)
    assert (idx2 == idx).all() and set(tab) == set(VALUES)
    Xn = X.numpy()
    for i, r in enumerate(SEEDS):
        want = np.array([[Xn[idx[i, l, k], l, k] for k in range(W)] for l in range(L)])
        assert np.array_equal(tab[f"res{r}"].numpy(), want), r
    out["resample"] = f"{len(SEEDS)} seeded draws, each X[index, l, k] neuron by neuron; reproducible, seeds differ"
    S = [(0, 1), (2, 3), (2, 0), (1, 6)]
    for v in VALUES:
        d = written(S, v, tab, L, "cpu")
        for l, (ix, val) in d.items():
            assert torch.equal(val, tab[v][l, ix].float()), v
    z = written(S, "zero", tab, L, "cpu")
    assert all(torch.equal(v, torch.zeros_like(v)) for _, v in z.values())
    ident = written([], "median", tab, L, "cpu", identity=True)
    assert set(ident) == set(range(L)) and all(len(ix) == 0 for ix, _ in ident.values())
    out["written"] = "every table through ablation_dict's mean branch, zero through its zero branch, identity hooks with nothing written"
    c1 = constant_tables(abar, P, idx)
    c2 = constant_tables(2 * abar, P, idx)
    m = RA.ablation_dict(S, "mean", abar, L, "cpu")
    t1 = {"mean": abar, "c": c1[0]}
    t2 = {"mean": abar, "c": c2[0]}
    assert all(torch.equal(written(S, "c", t1, L, "cpu")[l][1], m[l][1]) for l in m), "the constant-ā resample table does not reproduce the mean arm"
    assert not all(torch.equal(written(S, "c", t2, L, "cpu")[l][1], m[l][1]) for l in m), "the constant-2ā resample table reproduces the mean arm"
    out["resample_known"] = "constant ā: every draw writes ā exactly (the mean arm's values); constant 2ā does not"
    return out
