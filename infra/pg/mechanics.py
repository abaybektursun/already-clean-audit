"""MECHANICS screen instrument (studies/mechanics/SPEC_mechanics.md §2, §3): random-token sequences from seeds, online fp32 accumulators for the
induction excess E, the position range P, the token and previous-token variance ratios T and U and the random-text activity R, the per-head
induction score, the per-head ablation hook, the repeat flag and the weight-side numbers. The job (infra/aws_pg/jobs/mechanics_screen.py), its dry
run and the verdict import it.

Positions. Every sequence is [BOS] + 2 * half ids (half = 128); the BOS is model index 0 and is not a position; sequence position p is model index
p + 1. a is the down_proj input (the model's own activation, bf16), cast to fp32 before it is summed.
Sequences (§2), ids from the sampling vocabulary V: every id below len(tok) that is not an added token, has a non-zero count in the harvest
corpus and is not among the corpus's n_top (1,000) most frequent ids (amendment 5; sampling_ids):
  repeated  default_rng(0): x = V[rng.integers(0, |V|, (n, half))]; row = x ‖ x (pass 1 = positions 0..half-1, pass 2 = half..2 half-1).
  A, B      default_rng(1), default_rng(2): V[rng.integers(0, |V|, (n, 2 half))].
  narrow    default_rng(3): sub = V[rng.choice(|V|, 2048, replace=False)], then rows sub[rng.integers(0, 2048, (n, 2 half))].
  natural   default_rng(4): rng.choice(n_windows, 256, replace=False) harvest windows, each window's first 256 corpus tokens (window positions
            0..255, so every prefix is the harvest's own).
Per neuron k, s_k = the larger of the bank's two per-pole q999 values (amendment 1; scale_from_q999; NaN when zero or non-finite):
  D_X = mean a over positions half..2 half-1 - mean a over 0..half-1 on set X;  E = (D_rep - D_A) / s;  E' = (D_B - D_A) / s.
  E_early, E_early': the same with pass 2 cut to positions half..half+15 (amendment 6; the pass-1 baseline stays 0..half-1).
  P = (max - min over 8 bins of 32 consecutive positions of the bin mean on A) / s;  P' = the same on B with the positions assigned to bins
      through default_rng(10).permutation(2 half) (permuting the 8 bin means alone leaves max - min unchanged, so positions are permuted).
  T = between-id variance / total variance of a over the narrow tokens whose id occurs >= 20 times in the narrow set; U = the same with the
      previous id (positions >= 1, previous ids occurring >= 20 times as a previous id); T', U' = the same with the labels permuted across the
      kept tokens by default_rng(11).
  R = mean |a| over A / s.   F = mean |a| over positions 0..3 / mean |a| over positions 4..2 half-1 on A (amendment 7).
Head score per (layer, head): mean attention weight from model index 1 + half + i to i + 2 (sequence positions half + i -> i + 1), i = 1..half-2,
on the repeated set minus the same on A (eager attention: the attention module returns its weights).
Copying score per head (amendment 2): at the same pass-2 query positions of the repeated set, the head's write g_l * (W_O[:, head slice] z_head),
z_head = the head's slice of o_proj's input, dotted with W_U[t] for the correct next token t (the token at the attended position i + 1), minus
the mean of the same dot product over 32 fixed random ids of V (default_rng(15)); g_l = g_final * g_attn_l, the gains of the final norm and the
layer's post_attention_layernorm (the positive per-position 1/rms factors on the path are left out).
Ablation: a forward pre-hook on o_proj sets the head's slice [h * head_dim, (h + 1) * head_dim) of its input (the attention output before o_proj)
to zero; a layer with no listed head still gets the hook, which copies its input and writes nothing.
Repeat flag: the marked token is the last id of its window, and the 3-gram (two preceding ids, marked id) occurs earlier in the window.
Weights: column norm of down_proj, and the mean, standard deviation / ||w_k||, skew (m3 / m2^1.5) and kurtosis (m4 / m2^2, normal = 3) over the
vocabulary of W_U w_k, computed directly in fp32 (wu_columns; amendment 8); the _eff variant puts the gains in: W_U (g_final * g_post * w_k),
g_post = the layer's post_feedforward_layernorm weight, g_final = the final norm's weight, std divided by ||g_post * w_k||.
Fail loud: no try/except, no fallback.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch


N_SEQ, HALF = 512, 128
N_NARROW, MIN_OCC = 2048, 20
N_NAT, NAT_LEN = 256, 256
N_BINS = 8
SCALE_Q = 0.999
N_TOP_EXCL = 1000
EARLY = 16
F_FIRST = 4
N_COPY_RAND = 32
SEEDS = {"rep": 0, "A": 1, "B": 2, "narrow": 3, "natural": 4, "random_set": 5, "bin_perm": 10, "label_perm": 11, "random_heads": 12,
         "known_poles": 13, "shuffle": 14, "copy_rand": 15}
LABELS = ("id", "prev", "id_perm", "prev_perm")


# ------------------------------------------------------------------------------------------ sequences
def nonspecial_ids(tok) -> np.ndarray:
    added = {int(i) for i in tok.added_tokens_decoder}
    ids = np.array([i for i in range(len(tok)) if i not in added], dtype=np.int64)
    assert ids.size > 0 and len(added) > 0, (ids.size, len(added))
    return ids


def sampling_ids(tok, corpus_tokens, n_top: int = N_TOP_EXCL) -> tuple:
    """(V, record): the non-special ids with a non-zero corpus count, minus the n_top most frequent ids of the corpus (stable order)."""
    V0 = nonspecial_ids(tok)
    t = np.asarray(corpus_tokens)
    assert t.min() >= 0 and t.max() < len(tok), (t.min(), t.max(), len(tok))
    cnt = np.bincount(t, minlength=len(tok))
    top = np.argsort(-cnt, kind="stable")[:n_top]
    zero = cnt[V0] == 0
    intop = np.isin(V0, top)
    V = V0[~zero & ~intop]
    rec = {"n_nonspecial": int(V0.size), "n_zero_count": int(zero.sum()), "n_top_excluded": int((intop & ~zero).sum()), "n_top": int(n_top),
           "n_top_special": int((~np.isin(top, V0)).sum()), "n_sampling": int(V.size), "top_count_cut": int(cnt[top[-1]]) if n_top else None}
    return V, rec


def sequences(V, n_seq: int = N_SEQ, half: int = HALF, n_narrow: int = N_NARROW) -> dict:
    V = np.asarray(V, np.int64)
    assert V.ndim == 1 and np.unique(V).size == V.size and V.size >= n_narrow, V.shape
    r = np.random.default_rng(SEEDS["rep"])
    x = V[r.integers(0, V.size, (n_seq, half))]
    out = {"rep": np.concatenate([x, x], 1)}
    for k in ("A", "B"):
        out[k] = V[np.random.default_rng(SEEDS[k]).integers(0, V.size, (n_seq, 2 * half))]
    r = np.random.default_rng(SEEDS["narrow"])
    sub = V[r.choice(V.size, n_narrow, replace=False)]
    out["narrow"] = sub[r.integers(0, n_narrow, (n_seq, 2 * half))]
    out["narrow_ids"] = sub
    return out


def natural_windows(tokens, n_windows: int, hw: int, n: int = N_NAT, length: int = NAT_LEN):
    """(window indices, ids [n, length]): the first `length` tokens of n distinct harvest windows."""
    assert length <= hw and n_windows * hw <= len(tokens), (length, hw, n_windows, len(tokens))
    w = np.random.default_rng(SEEDS["natural"]).choice(n_windows, n, replace=False)
    return w, np.stack([np.asarray(tokens[i * hw:i * hw + length], np.int64) for i in w])


def shuffled_pass2(row, half: int = HALF) -> np.ndarray:
    """The detector's failing input: pass 2 of a repeated row permuted by default_rng(14)."""
    row = np.asarray(row, np.int64)
    p = np.random.default_rng(SEEDS["shuffle"]).permutation(half)
    return np.concatenate([row[:half], row[half:][p]])


def narrow_labels(seqs, sub, min_occ: int = MIN_OCC) -> dict:
    """Per token of the narrow set: the id's index in sub and the previous id's index (position >= 1), -1 where excluded (fewer than min_occ
    occurrences); the permuted copies (labels permuted across the kept tokens); the counts."""
    seqs, sub = np.asarray(seqs, np.int64), np.asarray(sub, np.int64)
    order = np.argsort(sub)
    lab = order[np.searchsorted(sub[order], seqs)]
    assert np.array_equal(sub[lab], seqs), "a narrow id outside the subset"
    K = sub.size
    cnt = np.bincount(lab.ravel(), minlength=K)
    prev = np.full_like(lab, -1)
    prev[:, 1:] = lab[:, :-1]
    pcnt = np.bincount(prev[prev >= 0], minlength=K)
    out = {"id": np.where(cnt[lab] >= min_occ, lab, -1), "prev": np.where((prev >= 0) & (pcnt[np.maximum(prev, 0)] >= min_occ), prev, -1),
           "id_count": cnt, "prev_count": pcnt}
    rng = np.random.default_rng(SEEDS["label_perm"])
    for k in ("id", "prev"):
        x = out[k].copy()
        m = x >= 0
        x[m] = rng.permutation(x[m])
        out[f"{k}_perm"] = x
    return out


def window_ids(tokens, t: int, hw: int, length: int = 2 * HALF) -> np.ndarray:
    """The `length` corpus tokens ending at global token t (t last), never before t's harvest window start (the model saw nothing earlier)."""
    t = int(t)
    return np.asarray(tokens[max(t - length + 1, t - t % hw):t + 1], np.int64)


def repeat_flag(w) -> bool:
    x = np.asarray(w, np.int64)
    m = x.size - 1
    if m < 3:
        return False
    return bool(((x[:m - 2] == x[m - 2]) & (x[1:m - 1] == x[m - 1]) & (x[2:m] == x[m])).any())


def head_index(half: int = HALF):
    """Model indices (query, key) of the head score: (1 + half + i, i + 2) for i = 1..half-2."""
    i = np.arange(1, half - 1)
    return 1 + half + i, i + 2


# ------------------------------------------------------------------------------------------ hooks (each removed on exit)
@contextlib.contextmanager
def tap(layers, fn):
    """Forward pre-hooks on every layer's mlp.down_proj: fn(layer, a) with a its input [B, S, W]; nothing is written."""
    def hook(i):
        def f(m, args):
            assert fn(i, args[0]) is None, "a tap callback returned a value (it would replace down_proj's input)"
        return f
    hs = [l.mlp.down_proj.register_forward_pre_hook(hook(i)) for i, l in enumerate(layers)]
    yield
    for h in hs:
        h.remove()


@contextlib.contextmanager
def attn_tap(layers, fn):
    """Forward hooks on every layer's self_attn: fn(layer, attention weights [B, H, S, S]); None under sdpa, which fn must refuse."""
    def hook(i):
        def f(m, args, out):
            assert fn(i, out[1]) is None
        return f
    hs = [l.self_attn.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    yield
    for h in hs:
        h.remove()


@contextlib.contextmanager
def o_tap(layers, fn):
    """Forward pre-hooks on every layer's self_attn.o_proj: fn(layer, z) with z its input [B, S, H * D] (the attention output); nothing is written."""
    def hook(i):
        def f(m, args):
            assert fn(i, args[0]) is None
        return f
    hs = [l.self_attn.o_proj.register_forward_pre_hook(hook(i)) for i, l in enumerate(layers)]
    yield
    for h in hs:
        h.remove()


@contextlib.contextmanager
def ablate(layers, heads: dict, n_heads: int, head_dim: int):
    """heads {layer: [head, ...]}: those heads' slices of o_proj's input set to zero in every forward; every layer is hooked."""
    assert all(0 <= int(h) < n_heads for v in heads.values() for h in v), heads
    rec = {"calls": 0}

    def hook(i):
        hh = sorted({int(h) for h in heads.get(i, [])})

        def f(m, args):
            x = args[0]
            assert x.shape[-1] == n_heads * head_dim, (x.shape, n_heads, head_dim)
            y = x.clone()
            if hh:
                y.view(*y.shape[:-1], n_heads, head_dim)[..., hh, :] = 0
            rec["calls"] += 1
            return (y, *args[1:])
        return f
    hs = [l.self_attn.o_proj.register_forward_pre_hook(hook(i)) for i, l in enumerate(layers)]
    yield rec
    for h in hs:
        h.remove()


@contextlib.contextmanager
def write_neuron(layers, L: int, k: int, value: float):
    """The identity gate's failing input: value added to neuron k of layer L's down_proj input at every position after the BOS."""
    def f(m, args):
        y = args[0].clone()
        y[:, 1:, k] += value
        return (y, *args[1:])
    h = layers[L].mlp.down_proj.register_forward_pre_hook(f)
    yield
    h.remove()


# ------------------------------------------------------------------------------------------ online accumulators (fp32)
class PosSum:
    """Sums over sequences of a and of |a| at each sequence position, S and Q [L, P, W]; with a scale [L, W], the count of |a| > s over positions
    >= abs_from, X [L, W]."""

    def __init__(self, n_layers: int, n_pos: int, width: int, device, abs_from: int = 0, scale=None):
        self.S = torch.zeros(n_layers, n_pos, width, device=device)
        self.Q = torch.zeros(n_layers, n_pos, width, device=device)
        self.X = None if scale is None else torch.zeros(n_layers, width, device=device)
        self.scale = scale
        self.abs_from = abs_from
        self.n = [0] * n_layers

    def update(self, layer: int, a: torch.Tensor):
        assert a.dim() == 3 and a.shape[1] == self.S.shape[1] + 1 and a.shape[2] == self.S.shape[2], (tuple(a.shape), tuple(self.S.shape))
        x = a[:, 1:].float()
        self.S[layer] += x.sum(0)
        xa = x.abs()
        self.Q[layer] += xa.sum(0)
        if self.X is not None:
            self.X[layer] += (xa[:, self.abs_from:] > self.scale[layer]).sum((0, 1)).float()
        self.n[layer] += a.shape[0]

    def count(self) -> int:
        assert len(set(self.n)) == 1 and self.n[0] > 0, self.n
        return self.n[0]


class Between:
    """Class sums C [L, K, W] of a - c over the tokens with label >= 0, label counts [L, K] and the sum of squares q [L, W]; c is the layer's mean
    over its first batch's labelled tokens (a shift for fp32 accuracy; the ratio does not depend on it)."""

    def __init__(self, n_layers: int, K: int, width: int, device):
        self.C = torch.zeros(n_layers, K, width, device=device)
        self.q = torch.zeros(n_layers, width, device=device)
        self.c = torch.zeros(n_layers, width, device=device)
        self.cnt = torch.zeros(n_layers, K, dtype=torch.long, device=device)
        self.shifted = [False] * n_layers
        self.K = K

    def update(self, layer: int, a: torch.Tensor, lab: torch.Tensor):
        """a [B, 1 + P, W]; lab [B, P] long, -1 excluded."""
        W = self.C.shape[2]
        assert a.dim() == 3 and a.shape[2] == W and tuple(lab.shape) == (a.shape[0], a.shape[1] - 1), (tuple(a.shape), tuple(lab.shape))
        l = lab.reshape(-1)
        m = l >= 0
        x = a[:, 1:].float().reshape(-1, W)[m]
        l = l[m]
        assert l.numel() > 0 and int(l.max()) < self.K
        if not self.shifted[layer]:
            self.c[layer] = x.mean(0)
            self.shifted[layer] = True
        x = x - self.c[layer]
        self.C[layer].index_add_(0, l, x)
        self.q[layer] += (x * x).sum(0)
        self.cnt[layer] += torch.bincount(l, minlength=self.K)

    def ratio(self) -> np.ndarray:
        """[L, W] float64: sum_k n_k (m_k - m)^2 / sum (a - m)^2; NaN where the total is not positive."""
        assert all(self.shifted) and bool((self.cnt == self.cnt[0]).all()), "the layers saw different tokens"
        n = self.cnt[0].double()
        keep = n > 0
        out = []
        for L in range(self.C.shape[0]):
            C = self.C[L].double()
            N = n.sum()
            t2 = C.sum(0) ** 2 / N
            between = (C[keep] ** 2 / n[keep, None]).sum(0) - t2
            total = self.q[L].double() - t2
            out.append(torch.where(total > 0, between / total, torch.full_like(total, float("nan"))).cpu().numpy())
        return np.stack(out)

    def means(self, L: int, k) -> np.ndarray:
        """Per-label means of a for neurons k of layer L, [len(k), K] float64 (NaN where a label has no token)."""
        n = self.cnt[L].double()
        M = (self.C[L][:, k].double() / n[:, None] + self.c[L][k].double()).T
        return torch.where(n[None] > 0, M, torch.full_like(M, float("nan"))).cpu().numpy()


class HeadScore:
    """Sum over sequences and i of the attention weight from model index 1 + half + i to i + 2, per (layer, head)."""

    def __init__(self, n_layers: int, n_heads: int, half: int, device):
        q, k = head_index(half)
        self.q, self.k = torch.as_tensor(q, device=device), torch.as_tensor(k, device=device)
        self.S = torch.zeros(n_layers, n_heads, device=device)
        self.n = [0] * n_layers
        self.S_len = 2 * half + 1

    def update(self, layer: int, w):
        assert w is not None, "no attention weights: eager attention is not active"
        assert w.dim() == 4 and w.shape[1] == self.S.shape[1] and w.shape[2] == w.shape[3] == self.S_len, tuple(w.shape)
        self.S[layer] += w[:, :, self.q, self.k].float().sum((0, 2))
        self.n[layer] += w.shape[0] * self.q.numel()

    def mean(self) -> np.ndarray:
        assert len(set(self.n)) == 1 and self.n[0] > 0, self.n
        return (self.S.double() / self.n[0]).cpu().numpy()


class CopyScore:
    """Per (layer, head): the sum over rows and pass-2 query positions (model index 1 + half + i, i = 1..half-2) of
    W_U[t] . (g_l * W_O_h z_h) minus the mean of W_U[r] . (g_l * W_O_h z_h) over the random ids r; t = the correct next token (row[half + i + 1]).
    W_U [V, d] and W_O[l] [d, H * D] are the model's weights, gains[l] [d] = g_final * g_attn_l."""

    def __init__(self, W_U, W_O: list, gains: list, n_heads: int, head_dim: int, half: int, rand_ids, chunk: int = 1024):
        self.WU, self.WO, self.g = W_U, W_O, [g.float() for g in gains]
        self.WR = W_U[torch.as_tensor(np.asarray(rand_ids), device=W_U.device)].float()
        self.H, self.D, self.half, self.chunk = n_heads, head_dim, half, chunk
        self.q = torch.as_tensor(head_index(half)[0], device=W_U.device)
        self.S = torch.zeros(len(W_O), n_heads, dtype=torch.float64, device=W_U.device)
        self.n = [0] * len(W_O)

    @staticmethod
    def targets(ids, half: int):
        """The correct next token at each scored query position: row[half + i + 1], i = 1..half-2."""
        return np.asarray(ids)[:, half + 2:2 * half]

    def update(self, layer: int, z: torch.Tensor, targets: torch.Tensor):
        B, nq = z.shape[0], self.q.numel()
        assert z.dim() == 3 and z.shape[1] == 2 * self.half + 1 and z.shape[2] == self.H * self.D and tuple(targets.shape) == (B, nq), (tuple(z.shape), tuple(targets.shape))
        d = self.WO[layer].shape[0]
        WO = self.WO[layer].float().view(d, self.H, self.D)
        zq = z[:, self.q].float().reshape(B * nq, self.H, self.D)
        t = targets.reshape(-1)
        for c0 in range(0, B * nq, self.chunk):
            o = torch.einsum("nhd,ehd->nhe", zq[c0:c0 + self.chunk], WO) * self.g[layer]
            tg = torch.einsum("nhe,ne->nh", o, self.WU[t[c0:c0 + self.chunk]].float())
            rn = torch.einsum("nhe,re->nhr", o, self.WR).mean(-1)
            self.S[layer] += (tg - rn).double().sum(0)
        self.n[layer] += B * nq

    def mean(self) -> np.ndarray:
        assert len(set(self.n)) == 1 and self.n[0] > 0, self.n
        return (self.S / self.n[0]).cpu().numpy()


# ------------------------------------------------------------------------------------------ signatures (numpy, float64)
def half_diff(S, n: int, half: int = HALF, stop: int | None = None) -> np.ndarray:
    """[L, W]: mean a over positions half..stop-1 (default 2 half - 1) minus mean over 0..half-1, from position sums S [L, 2 half, W] over n
    sequences."""
    S = np.asarray(S)
    assert S.ndim == 3 and S.shape[1] == 2 * half and (stop is None or half < stop <= 2 * half), (S.shape, stop)
    m = S.astype(np.float64) / n
    return m[:, half:stop].mean(1) - m[:, :half].mean(1)


def first_ratio(Q, n_first: int = F_FIRST) -> np.ndarray:
    """[L, W]: mean |a| over positions 0..n_first-1 over mean |a| over the rest, from per-position |a| sums Q [L, P, W]."""
    Q = np.asarray(Q).astype(np.float64)
    return Q[:, :n_first].mean(1) / Q[:, n_first:].mean(1)


def bin_means(S, n: int, n_bins: int = N_BINS, perm=None) -> np.ndarray:
    """[L, W, n_bins]: mean a over each bin of consecutive positions (after reordering the positions by perm, when given)."""
    m = np.asarray(S).astype(np.float64) / n
    if perm is not None:
        assert sorted(perm) == list(range(m.shape[1])), "perm is not a permutation of the positions"
        m = m[:, np.asarray(perm)]
    L, P, W = m.shape
    assert P % n_bins == 0, (P, n_bins)
    return m.reshape(L, n_bins, P // n_bins, W).mean(2).transpose(0, 2, 1)


def scale_from_q999(q999) -> np.ndarray:
    """s_k = the larger of the bank's two per-pole 99.9th percentiles of |a| (stats q999 [W, 2]; amendment 1); NaN where either is non-finite
    (np.maximum propagates it) or the larger is not positive."""
    q = np.asarray(q999, np.float64)
    assert q.ndim == 2 and q.shape[1] == 2, q.shape
    s = np.maximum(q[:, 0], q[:, 1])
    return np.where(np.isfinite(s) & (s > 0), s, np.nan)


def lines(nulls: dict, valid, q: float, floors: dict) -> dict:
    """Per signature: the q-quantile over the valid neurons of its null, or its floor, whichever is larger."""
    return {k: max(float(np.quantile(v[valid], q)), floors.get(k, -np.inf)) for k, v in nulls.items()}


def top_cells(score, mask, cap: int) -> list:
    """(layer, neuron) of the cells in mask, by score descending (stable), at most cap."""
    W = score.shape[1]
    idx = np.flatnonzero(np.asarray(mask).ravel())
    idx = idx[np.argsort(-np.asarray(score).ravel()[idx], kind="stable")][:cap]
    return [(int(i // W), int(i % W)) for i in idx]


def largest_remainder(weights, n: int) -> np.ndarray:
    w = np.asarray(weights, np.float64)
    q = n * w / w.sum()
    c = np.floor(q).astype(np.int64)
    c[np.argsort(-(q - c), kind="stable")[:n - c.sum()]] += 1
    assert c.sum() == n
    return c


def layer_matched(cells: list, n: int, valid, seed: int = SEEDS["random_set"]) -> list:
    """n (layer, neuron) cells with the layer distribution of `cells` (largest-remainder counts; uniform over layers when cells is empty), drawn
    within each layer without replacement from the valid neurons outside `cells`, by default_rng(seed)."""
    valid = np.asarray(valid, bool)
    L, W = valid.shape
    per = np.bincount([c[0] for c in cells], minlength=L) if cells else np.ones(L)
    counts = largest_remainder(per, n)
    rng = np.random.default_rng(seed)
    taken = set(map(tuple, cells))
    out = []
    for l in range(L):
        pool = np.array([k for k in np.flatnonzero(valid[l]) if (l, int(k)) not in taken], np.int64)
        assert pool.size >= counts[l], (l, pool.size, counts[l])
        out += [(l, int(k)) for k in np.sort(rng.choice(pool, counts[l], replace=False))]
    return out


# ------------------------------------------------------------------------------------------ weight side (direct, fp32)
def moments(X: torch.Tensor) -> tuple:
    """Per column of X [V, c]: mean, population std, skew m3 / m2^1.5, kurtosis m4 / m2^2 (normal = 3)."""
    m = X.mean(0)
    x = X - m
    m2 = (x * x).mean(0)
    return m, m2.sqrt(), (x ** 3).mean(0) / m2 ** 1.5, (x ** 4).mean(0) / m2 ** 2


@torch.no_grad()
def wu_columns(W_U: torch.Tensor, W_down: torch.Tensor, g_final, g_post, chunk: int = 1024) -> dict:
    """Per column w_k of W_down [d, W]: w_norm; wu_mean, wu_std_ratio (std / w_norm), wu_skew, wu_kurtosis of W_U w_k over the vocabulary; the
    same with _eff for W_U (g_final * g_post * w_k), std divided by ||g_post * w_k||. W_U is cast to fp32 by the caller or here."""
    WU = W_U.float()
    Wd = W_down.float()
    gp, gf = g_post.float(), g_final.float()
    out = {k: [] for k in ("w_norm", "wu_mean", "wu_std_ratio", "wu_skew", "wu_kurtosis", "wu_mean_eff", "wu_std_ratio_eff", "wu_skew_eff", "wu_kurtosis_eff")}
    for c0 in range(0, Wd.shape[1], chunk):
        w = Wd[:, c0:c0 + chunk]
        we = gp[:, None] * w
        for suf, col, nrm in (("", w, w.norm(dim=0)), ("_eff", gf[:, None] * we, we.norm(dim=0))):
            m, sd, sk, ku = moments(WU @ col)
            out[f"wu_mean{suf}"].append(m)
            out[f"wu_std_ratio{suf}"].append(sd / nrm)
            out[f"wu_skew{suf}"].append(sk)
            out[f"wu_kurtosis{suf}"].append(ku)
        out["w_norm"].append(w.norm(dim=0))
    return {k: torch.cat(v).double().cpu().numpy() for k, v in out.items()}


# ------------------------------------------------------------------------------------------ selfcheck
class _Toy(torch.nn.Module):
    """A stand-in layer with the two hook points: self_attn.o_proj and mlp.down_proj."""

    def __init__(self, H, D, W, d):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(H * D, d, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(W, d, bias=False)


def selfcheck() -> dict:
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    # sequences
    V = np.arange(10, 400)
    s1, s2 = sequences(V, 6, 8, 32), sequences(V, 6, 8, 32)
    assert all(np.array_equal(s1[k], s2[k]) for k in s1), "sequences are not reproducible"
    assert s1["rep"].shape == (6, 16) and np.array_equal(s1["rep"][:, :8], s1["rep"][:, 8:]) and not np.array_equal(s1["A"], s1["B"])
    assert np.isin(s1["narrow"], s1["narrow_ids"]).all() and np.unique(s1["narrow_ids"]).size == 32 and np.isin(s1["A"], V).all()
    assert np.array_equal(s1["rep"][:, :8], V[np.random.default_rng(0).integers(0, V.size, (6, 8))])
    sh = shuffled_pass2(s1["rep"][0], 8)
    assert np.array_equal(sh[:8], s1["rep"][0, :8]) and sorted(sh[8:]) == sorted(s1["rep"][0, 8:]) and not np.array_equal(sh, s1["rep"][0])
    toks = np.arange(1000)
    w, nat = natural_windows(toks, 10, 100, 4, 16)
    assert nat.shape == (4, 16) and all(np.array_equal(nat[j], np.arange(w[j] * 100, w[j] * 100 + 16)) for j in range(4))
    # labels
    seqs = np.array([[5, 6, 5, 7], [6, 6, 9, 5]])
    lab = narrow_labels(seqs, np.array([9, 5, 6, 7]), 2)
    assert lab["id"].tolist() == [[1, 2, 1, -1], [2, 2, -1, 1]], lab["id"]
    assert lab["prev"].tolist() == [[-1, 1, 2, 1], [-1, 2, 2, -1]], lab["prev"]
    assert sorted(lab["id_perm"][lab["id"] >= 0]) == sorted(lab["id"][lab["id"] >= 0]) and (lab["id_perm"] >= 0).sum() == (lab["id"] >= 0).sum()
    # repeat flag and windows
    assert repeat_flag([1, 2, 3, 9, 1, 2, 3]) and not repeat_flag([1, 2, 4, 9, 1, 2, 3]) and not repeat_flag([2, 3, 9, 1, 2, 3])
    assert repeat_flag([4, 4, 4, 4]) and not repeat_flag([4, 4, 4]) and repeat_flag([1, 2, 3, 1, 2, 3][::-1]) and not repeat_flag([1, 2, 3, 4, 5, 6])
    assert window_ids(toks, 205, 100, 16).tolist() == list(range(200, 206)) and window_ids(toks, 250, 100, 16).tolist() == list(range(235, 251))
    q, k = head_index(8)
    assert q.tolist() == [10, 11, 12, 13, 14, 15] and k.tolist() == [3, 4, 5, 6, 7, 8]
    # hooks on stand-in layers
    H, D, W, d, B, S = 3, 4, 6, 5, 2, 7
    layers = [_Toy(H, D, W, d) for _ in range(2)]
    x = torch.randn(B, S, H * D, generator=g)
    y0 = layers[0].self_attn.o_proj(x)
    with ablate(layers, {0: [1]}, H, D) as rec:
        y1 = layers[0].self_attn.o_proj(x)
        y1b = layers[1].self_attn.o_proj(x)
    x1 = x.clone()
    x1[..., D:2 * D] = 0
    assert torch.allclose(y1, layers[0].self_attn.o_proj(x1)) and torch.equal(y1b, layers[1].self_attn.o_proj(x)) and rec["calls"] == 2
    with ablate(layers, {}, H, D):
        assert torch.equal(layers[0].self_attn.o_proj(x), y0), "the zero-head ablation changed the output"
    assert torch.equal(layers[0].self_attn.o_proj(x), y0), "a hook outlived its context"
    a = torch.randn(B, S, W, generator=g)
    seen = []
    with tap(layers, lambda i, t: seen.append((i, t.clone()))):
        o = layers[1].mlp.down_proj(a)
    assert len(seen) == 1 and seen[0][0] == 1 and torch.equal(seen[0][1], a) and torch.equal(o, layers[1].mlp.down_proj(a))
    with write_neuron(layers, 1, 2, 0.5):
        o2 = layers[1].mlp.down_proj(a)
    a2 = a.clone()
    a2[:, 1:, 2] += 0.5
    assert torch.allclose(o2, layers[1].mlp.down_proj(a2)) and not torch.allclose(o2, o)
    # PosSum and Between against brute force
    Lc, Pn, Wc, K = 2, 8, 5, 4
    acts = [torch.randn(3, Pn + 1, Wc, generator=g) for _ in range(4)]
    scale = torch.full((Lc, Wc), 1.0)
    ps = PosSum(Lc, Pn, Wc, "cpu", abs_from=1, scale=scale)
    labs = [torch.as_tensor(rng.integers(-1, K, (3, Pn))) for _ in range(4)]
    bt = Between(Lc, K, Wc, "cpu")
    for a_, l_ in zip(acts, labs):
        for L in range(Lc):
            ps.update(L, a_ + L)
            bt.update(L, (a_ + 3.0) * (L + 1), l_)
    X = torch.cat(acts).double()[:, 1:]
    assert ps.count() == 12 and torch.allclose(ps.S[1].double(), (X + 1).sum(0), atol=1e-5)
    assert torch.allclose(ps.Q[0].double(), X.abs().sum(0), atol=1e-4) and torch.equal(ps.X[0], (X[:, 1:].abs() > 1).sum((0, 1)).float())
    Fr = first_ratio(ps.Q.numpy(), 2)
    assert np.allclose(Fr[0], X[:, :2].abs().mean((0, 1)).numpy() / X[:, 2:].abs().mean((0, 1)).numpy(), rtol=1e-5)
    lab_all = torch.cat(labs).reshape(-1).numpy()
    for L in range(Lc):
        v = ((X + 3.0) * (L + 1)).reshape(-1, Wc).numpy()[lab_all >= 0]
        ll = lab_all[lab_all >= 0]
        m = v.mean(0)
        bw = sum((ll == c).sum() * (v[ll == c].mean(0) - m) ** 2 for c in range(K) if (ll == c).any())
        assert np.allclose(bt.ratio()[L], bw / ((v - m) ** 2).sum(0), atol=1e-6), L
        assert np.allclose(bt.means(L, [0, 3])[:, 1], [v[ll == 1, 0].mean(), v[ll == 1, 3].mean()], atol=1e-5)
    pure = Between(1, K, 2, "cpu")
    lp = torch.as_tensor(rng.integers(0, K, (4, Pn)))
    vals = torch.tensor([3.0, -1.0, 0.5, 2.0])[lp]
    pure.update(0, torch.cat([torch.zeros(4, 1, 2), torch.stack([vals, vals + 0.25 * torch.randn(4, Pn, generator=g)], -1)], 1), lp)
    r = pure.ratio()[0]
    assert abs(r[0] - 1.0) < 1e-6 and 0.5 < r[1] < 1.0, r
    # HeadScore, half_diff, bin_means
    half = 8
    hs = HeadScore(1, 2, half, "cpu")
    w = torch.zeros(3, 2, 2 * half + 1, 2 * half + 1)
    qq, kk = head_index(half)
    w[:, 0, qq, kk] = 1.0
    w[:, 1, qq, kk] = 0.25
    hs.update(0, w)
    assert np.allclose(hs.mean()[0], [1.0, 0.25])
    Sx = np.zeros((1, 2 * half, 2))
    Sx[0, half:, 0] = 6.0
    Sx[0, :, 1] = np.arange(2 * half) * 3.0
    D_ = half_diff(Sx, 3, half)
    assert np.allclose(D_[0], [2.0, half * 1.0]) and np.allclose(half_diff(Sx, 3, half, half + 2)[0], [2.0, (half + 0.5) - (half - 1) / 2])
    bm = bin_means(Sx, 3, 4)
    assert bm.shape == (1, 2, 4) and np.allclose(bm[0, 1], [1.5, 5.5, 9.5, 13.5])
    perm = np.random.default_rng(10).permutation(2 * half)
    assert np.allclose(np.sort(bin_means(Sx, 3, 4, perm)[0, 1].ravel()).sum() * 4, Sx[0, :, 1].sum() / 3)
    # scale from per-pole q999; the sampling vocabulary
    s = scale_from_q999(np.array([[0.2, 0.5], [3.0, 1.0], [0.0, 0.0], [np.nan, 2.0], [np.inf, 1.0]], np.float32))
    assert np.allclose(s[:2], [0.5, 3.0]) and np.isnan(s[2:]).all(), s

    class _Tok:
        added_tokens_decoder = {0: None, 9: None}

        def __len__(self):
            return 10
    Vs, vrec = sampling_ids(_Tok(), np.array([1, 1, 1, 2, 2, 3, 5, 5, 5, 5, 9, 9, 9, 9, 9, 6]), n_top=2)
    assert Vs.tolist() == [1, 2, 3, 6] and vrec["n_zero_count"] == 3 and vrec["n_top_excluded"] == 1 and vrec["n_top_special"] == 1, (Vs, vrec)
    # copying score on a stand-in: head 0 writes W_U[t] exactly (positive), head 1 writes -W_U[t] (negative)
    Vv, dd, Hh, Dh, hf = 12, 4, 2, 4, 4
    WUc = torch.randn(Vv, dd, generator=g)
    WOc = torch.zeros(dd, Hh * Dh)
    WOc[:, :Dh] = torch.eye(dd)
    WOc[:, Dh:] = -torch.eye(dd)
    ids = np.random.default_rng(1).integers(0, Vv, (3, 2 * hf))
    tg = CopyScore.targets(ids, hf)
    zc = torch.zeros(3, 2 * hf + 1, Hh * Dh)
    qi = head_index(hf)[0]
    for b in range(3):
        for j, qm in enumerate(qi):
            zc[b, qm, :Dh] = WUc[tg[b, j]]
            zc[b, qm, Dh:] = WUc[tg[b, j]]
    cs = CopyScore(WUc, [WOc], [torch.ones(dd)], Hh, Dh, hf, [0, 1, 2], chunk=2)
    cs.update(0, zc, torch.as_tensor(tg))
    want = np.mean([float(WUc[t] @ WUc[t] - (WUc[[0, 1, 2]] @ WUc[t]).mean()) for t in tg.ravel()])
    assert np.allclose(cs.mean()[0], [want, -want], rtol=1e-5) and want > 0, (cs.mean(), want)
    # lines, top cells, layer-matched draws
    nul = {"E": np.array([[0.0, 0.1, 0.05], [0.02, np.nan, 0.03]])}
    val = np.isfinite(nul["E"])
    assert lines(nul, val, 1.0, {"E": 0.2})["E"] == 0.2 and lines(nul, val, 1.0, {})["E"] == 0.1
    sc = np.array([[0.5, 0.9, 0.1], [0.7, 0.2, 0.3]])
    assert top_cells(sc, sc > 0.25, 2) == [(0, 1), (1, 0)]
    vmask = np.ones((3, 50), bool)
    vmask[2, 7] = False
    cells = [(0, 1), (0, 2), (2, 3)]
    lm = layer_matched(cells, 9, vmask)
    assert lm == layer_matched(cells, 9, vmask) and np.bincount([c[0] for c in lm], minlength=3).tolist() == [6, 0, 3]
    assert not set(lm) & set(cells) and (2, 7) not in lm and len(set(lm)) == 9
    assert np.bincount([c[0] for c in layer_matched([], 9, vmask)], minlength=3).tolist() == [3, 3, 3]
    # weight side against numpy float64
    WU = (torch.randn(500, 6, generator=g) + 0.3).numpy()
    Wd = torch.randn(6, 5, generator=g).numpy()
    gf, gp = (torch.rand(6, generator=g) + 0.5).numpy(), (torch.rand(6, generator=g) + 0.5).numpy()
    wc = wu_columns(torch.as_tensor(WU), torch.as_tensor(Wd), torch.as_tensor(gf), torch.as_tensor(gp), chunk=2)
    for suf, lg, nrm in (("", WU.astype(np.float64) @ Wd, np.linalg.norm(Wd.astype(np.float64), axis=0)),
                         ("_eff", (WU.astype(np.float64) * gf) @ (gp[:, None] * Wd), np.linalg.norm(gp[:, None] * Wd.astype(np.float64), axis=0))):
        c = lg - lg.mean(0)
        m2 = (c ** 2).mean(0)
        assert np.allclose(wc[f"wu_mean{suf}"], lg.mean(0), atol=1e-5) and np.allclose(wc[f"wu_std_ratio{suf}"], np.sqrt(m2) / nrm, rtol=1e-4)
        assert np.allclose(wc[f"wu_skew{suf}"], (c ** 3).mean(0) / m2 ** 1.5, atol=1e-4) and np.allclose(wc[f"wu_kurtosis{suf}"], (c ** 4).mean(0) / m2 ** 2, rtol=1e-4)
    assert np.allclose(wc["w_norm"], np.linalg.norm(Wd, axis=0), rtol=1e-6)
    return {"between_pure": float(r[0]), "between_noisy": float(r[1]), "scale": [float(s[0]), float(s[1])], "copy": [float(x) for x in cs.mean()[0]],
            "layer_matched": lm[:3]}


if __name__ == "__main__":
    print(selfcheck())
    print("MECHANICS_SELFCHECK_OK")
