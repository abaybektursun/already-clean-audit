"""SPEC4 BIRTH instrument additions (studies/reuse/SPEC4_birth.md §2, §3.1, §7). reuse_ablate and unite_route are imported unchanged.

High-background set B_t: over the upstream layers 0..n_up-1 of ā (n_up = 31 on O7: 31 x 11008 = 341,248 entries), the k = round(0.01 x count)
entries with the largest |ā| by torch.topk (3,412 on O7); the cut is the smallest |ā| included.
Exclusion: B_t is removed from the candidate pool before ranking (top_m with B_t at -inf) and before matching (random_matched with the magnitude
set to +inf on B_t, so a B_t neuron is never nearest). The ablated values stay ā. Every set carries a per-member flag "in B_t" (all False).
Top-200 bank: per side, the flat indices of the 200 largest summed attributions over all upstream neurons (no exclusion), their summed A and their ā,
so s' can be re-read offline at other cuts.
Part A bank (§3.1), per (case, side, checkpoint), from a capture of the side's contexts at P's layer L with gu_layers {L}:
    a_mean_abs [L, W]  mean over contexts of |a_k|        s_mean [L]  mean over contexts of the layer scale s_l
    gP, uP [N]         P's g and u at the marked token    GP, UP [d]  P's input rows (fp32)
    loadsG, loadsU [L, W]  G_P.d_k and U_P.d_k (Rows.loads; weights only, so the job banks them on the J side alone)
The statistics (rho_a, rho_omega with omega_k = |cbar_g G.d_k + cbar_u U.d_k|, rho_g, rho_u, rho_P, the row cosines, the share of D'_b(J) in
the pre-birth ranking) are computed offline by studies/reuse/scripts/verdict_birth.py.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from infra.pg import reuse_ablate as RA
from infra.pg import unite_route as UR

FRAC = 0.01
M = 50
TOP_BANK = 200


# ------------------------------------------------------------------------------------------ the high-background set
def high_background(abar: torch.Tensor, n_up: int) -> dict:
    """B_t over the first n_up rows of ā [n_layers, W]: a boolean mask [n_up, W], the cut, the count and the members (layer, neuron, ā)."""
    assert abar.ndim == 2 and abar.shape[0] >= n_up, (tuple(abar.shape), n_up)
    x = abar[:n_up].float()
    W = x.shape[1]
    n = x.numel()
    k = int(round(FRAC * n))
    v, i = torch.topk(x.abs().reshape(-1), k, sorted=True)
    flat = torch.zeros(n, dtype=torch.bool, device=x.device)
    flat[i] = True
    mask = flat.view(n_up, W)
    assert int(mask.sum()) == k, "the mask does not hold k entries"
    il = i.cpu().numpy()
    vals = x.reshape(-1)[i].cpu().numpy()
    members = [[int(j // W), int(j % W), float(a)] for j, a in zip(il, vals)]
    return {"mask": mask, "cut": float(v[-1]), "count": k, "n_entries": n, "n_up": n_up, "frac": FRAC,
            "per_layer": mask.sum(1).cpu().numpy().astype(int).tolist(), "members": members}


def write_high_background(path, hb: dict, t: str):
    """high_background.json: cut, count, per-layer counts and the members as [layer, neuron, ā], largest |ā| first."""
    rec = {"t": t, **{k: hb[k] for k in ("cut", "count", "n_entries", "n_up", "frac", "per_layer", "members")},
           "rule": "the round(frac x n_entries) upstream neurons with the largest |ā| over layers 0..n_up-1, torch.topk"}
    pathlib.Path(path).write_text(json.dumps(rec))


def mask_of_members(members, n_up: int, W: int) -> np.ndarray:
    """The boolean mask [n_up, W] of a high_background.json member list."""
    out = np.zeros((n_up, W), bool)
    for l, k, _ in members:
        out[int(l), int(k)] = True
    return out


# ------------------------------------------------------------------------------------------ exclusion
def in_B(S: list, mask) -> np.ndarray:
    """One flag per member of S: True when the member is in B_t."""
    mk = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    return np.array([bool(mk[l, k]) for l, k in S], bool)


def top_m_excluded(score: torch.Tensor, m: int, W: int, mask: torch.Tensor) -> list:
    """reuse_ablate.top_m on the score [L W] with every B_t entry of layers < L set to -inf."""
    L = score.numel() // W
    assert score.ndim == 1 and score.numel() == L * W and L <= mask.shape[0], (tuple(score.shape), W, tuple(mask.shape))
    s = score.clone()
    s[mask[:L].reshape(-1).to(s.device)] = -float("inf")
    D = RA.top_m(s, m, W)
    assert not in_B(D, mask).any(), "a ranked set holds a B_t neuron"
    return D


def random_matched_excluded(D_other: list, D_own: list, mag, mask) -> list:
    """reuse_ablate.random_matched with the magnitude array set to +inf on B_t: a B_t neuron is never the nearest. `mag` [L', W] is ā, or the mean
    |a| over J for R_J; it is not modified."""
    x = np.abs(np.asarray(mag.cpu() if torch.is_tensor(mag) else mag, dtype=np.float64)).copy()
    mk = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    assert x.shape[0] <= mk.shape[0] and x.shape[1] == mk.shape[1], (x.shape, mk.shape)
    x[mk[:x.shape[0]]] = np.inf
    assert not in_B(D_other, mk).any(), "the set to match holds a B_t neuron"
    R = RA.random_matched(D_other, D_own, x)
    assert not in_B(R, mk).any(), "a matched set holds a B_t neuron"
    return R


def set_flags(arms: list, mask, width: int = RA.MS[-1]) -> np.ndarray:
    """set_in_B [n_arms, width]: per arm and member, True when the member is in B_t (padding and the identity arm False)."""
    out = np.zeros((len(arms), width), bool)
    for i, arm in enumerate(arms):
        out[i, :len(arm["set"])] = in_B(arm["set"], mask)
    return out


def top_bank(score: torch.Tensor, abar: torch.Tensor, W: int, n: int = TOP_BANK) -> dict:
    """The n largest summed attributions over every upstream neuron (no exclusion): flat index l W + k, the summed A, and ā."""
    L = score.numel() // W
    v, i = torch.topk(score, min(n, score.numel()), sorted=True)
    return {"idx": i.cpu().numpy().astype(np.int64), "A": v.cpu().numpy().astype(np.float32),
            "abar": abar[:L].reshape(-1)[i].cpu().numpy().astype(np.float32)}


# ------------------------------------------------------------------------------------------ Part A
@torch.inference_mode()
def part_a(rows: UR.Rows, cap: dict, unit: dict, loads: bool) -> dict:
    """The Part A arrays of one side at one checkpoint. `cap`: a capture at the pole's layer L with gu_layers {L}. loads: G.d_k and U.d_k too."""
    assert len(unit["terms"]) == 1, "Part A is read on a single pole"
    L, n = unit["layer"], unit["terms"][0]["n"]
    out = {"a_mean_abs": torch.stack([cap["a"][l].abs().mean(0) for l in range(L)]).cpu().numpy().astype(np.float32),
           "s_mean": torch.stack([cap["s"][l].mean() for l in range(L)]).cpu().numpy().astype(np.float32),
           "gP": cap["g"][L][:, n].cpu().numpy().astype(np.float32), "uP": cap["u"][L][:, n].cpu().numpy().astype(np.float32)}
    G, U = rows.inputs(L, n)
    out["GP"], out["UP"] = G.cpu().numpy().astype(np.float32), U.cpu().numpy().astype(np.float32)
    if loads:
        lo = rows.loads(L, n)
        out["loadsG"] = torch.stack([x[0] for x in lo]).cpu().numpy().astype(np.float32)
        out["loadsU"] = torch.stack([x[1] for x in lo]).cpu().numpy().astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------ memory
def memory_start(device: str):
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def memory_gb(device: str) -> float:
    """torch.cuda.max_memory_allocated in GB since memory_start; 0.0 on CPU."""
    return float(torch.cuda.max_memory_allocated() / 1e9) if str(device).startswith("cuda") else 0.0


# ------------------------------------------------------------------------------------------ pure-logic checks on random tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    n_up, W = 5, 200
    abar = torch.tensor(rng.standard_cauchy((n_up + 1, W)), dtype=torch.float32)
    hb = high_background(abar, n_up)
    k = round(FRAC * n_up * W)
    mask = hb["mask"]
    assert hb["count"] == k == int(mask.sum()) == len(hb["members"]) == sum(hb["per_layer"]), (hb["count"], k)
    mags = np.abs(abar[:n_up].numpy())
    assert np.isclose(hb["cut"], np.sort(mags.ravel())[-k]), "the cut is not the k-th largest |ā|"
    assert (mags[mask.numpy()] >= hb["cut"]).all() and (mags[~mask.numpy()] <= hb["cut"]).all(), "the mask is not the top k by |ā|"
    assert (mask_of_members(hb["members"], n_up, W) == mask.numpy()).all(), "the member list does not rebuild the mask"
    assert high_background(torch.tensor(rng.normal(0, 1, (31, 11008)), dtype=torch.float32), 31)["count"] == 3412, "the O7 count is not 3,412"
    out["high_background"] = f"count round(0.01 x {n_up * W}) = {k}, cut the k-th largest |ā|, members rebuild the mask; 31 x 11008 gives 3,412"
    # exclusion in the ranking: a score whose largest entries are all in B_t
    L = 4
    score = torch.tensor(rng.normal(0, 1, L * W), dtype=torch.float32)
    score[mask[:L].reshape(-1)] += 100.0
    D = top_m_excluded(score, 20, W, mask)
    D0 = RA.top_m(score, 20, W)
    assert in_B(D0, mask).any() and not in_B(D, mask).any() and len(set(D)) == 20, "top_m_excluded did not remove B_t"
    keep = score.clone()
    keep[mask[:L].reshape(-1)] = -float("inf")
    assert D == RA.top_m(keep, 20, W), "top_m_excluded is not top_m over the rest"
    out["top_m_excluded"] = f"{int(in_B(D0, mask).sum())} of 20 unexcluded members are in B_t; 0 after exclusion; equal to top_m on the rest"
    # exclusion in the matching: a set whose nearest neighbours in |ā| are all in B_t
    ab = np.abs(abar[:n_up].numpy()).astype(np.float64)
    other = [(l, int(np.flatnonzero(~mask[l].numpy())[0])) for l in range(L) if (~mask[l]).any()][:3]
    for l, kk in other:
        for j in np.flatnonzero(mask[l].numpy()):
            ab[l, j] = ab[l, kk] + 1e-9            # B_t members of the layer sit at distance 1e-9 from the neuron to match
    R0 = RA.random_matched(other, [], ab)
    ab_copy = ab.copy()
    R = random_matched_excluded(other, [], ab, mask)
    assert not in_B(R, mask).any() and len(R) == len(other) and [l for l, _ in R] == [l for l, _ in other], "random_matched_excluded returned a B_t neuron"
    assert in_B(R0, mask).any() or not mask[:L].any(), "the constructed case did not put a B_t neuron nearest"
    assert np.array_equal(ab, ab_copy), "the magnitude array was modified"
    for trial in range(50):
        m_ = torch.tensor(rng.random((n_up, W)) < 0.3)
        Do = [(int(rng.integers(L)), int(rng.integers(W))) for _ in range(5)]
        Do = list(dict.fromkeys([x for x in Do if not m_[x]]))
        assert not in_B(random_matched_excluded(Do, [], rng.normal(0, 1, (L, W)), m_), m_).any()
    out["random_matched_excluded"] = f"{int(in_B(R0, mask).sum())} of {len(other)} unexcluded matches in B_t; 0 after; 50 random masks at 30% never return a member"
    arms = [{"arm": "identity", "set": []}, {"arm": "own", "set": D}, {"arm": "raw", "set": D0}]
    fl = set_flags(arms, mask)
    assert fl.shape == (3, RA.MS[-1]) and not fl[0].any() and not fl[1].any() and fl[2].sum() == in_B(D0, mask).sum()
    out["set_flags"] = "identity and excluded sets all False; the unexcluded set flags its B_t members"
    tb = top_bank(score, abar, W, 30)
    assert (tb["idx"] == torch.argsort(score, descending=True)[:30].numpy()).all() and np.allclose(tb["abar"], abar[:L].reshape(-1).numpy()[tb["idx"]])
    first = [j for j in tb["idx"] if not mask[j // W, j % W]][:20]
    assert [(int(j // W), int(j % W)) for j in first] == D, "the excluded set is not recoverable from the top bank"
    out["top_bank"] = "largest summed A first, ā per entry; the first 20 entries outside B_t are top_m_excluded's set"
    return out
