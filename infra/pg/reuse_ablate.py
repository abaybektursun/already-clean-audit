"""REUSE instrument (studies/reuse/SPEC.md §2): mean-replacement ablation of upstream MLP neurons and the fall of a pole's activation.

For pole P (layer L, neuron n, sign sigma) at context x, one forward of the 256-token window ending at x (BOS first), stopped after layer L:
    a_P(x) = sigma silu(g_n(x)) u_n(x)          (a planted unit: the weighted sum of its terms, unite_route.planted_unit).
Ablating a set S of (l, k), l < L: a forward pre-hook on layers[l].mlp.down_proj writes r_k into the down projection's input for every k in S at
every window position. BOS is not a window position and is left as it is: ā_k is a mean over corpus tokens, and BOS is not one.
    r_k = ā_k, k's mean a over UNITE's 200 e1 centring positions at the same checkpoint (mean replacement), or 0 (zero replacement).
    fall(x) = (a_P(x) - a_P^S(x)) / a_P(x);  a job's fall is the mean over its contexts with a_P(x) > 0.
Drivers D(J): the m upstream neurons with the largest sum over x in J of
    direct:  A_k(x) = unite_route.attribution with P's own coefficients (the MLP path at x), or
    grad:    a_k(t) d a_P(x) / d a_k(t) summed over the window positions t (all paths; BOS left out, as the ablation leaves it out).
Random-matched R: for each k in D_other, the same-layer neuron nearest in |ā| outside D_own, D_other and the neurons already picked.

Reuses unite_route (capture, attribution, Rows, pack, upto_layer, background_positions) and neurons_harvest (text_tower, forward).
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from infra.pg import neurons_harvest as NH
from infra.pg import unite_route as UR

MS = (20, 50, 100)
M_FOREIGN = 50
LENGTH = 256
ID_TOL = 1e-3
FLOOR = 0.25
CANCEL_RATIO = 2.0
MIN_POS = 3


def window_start(bos) -> int:
    """First window position in the forwarded row: 1 after a BOS, else 0."""
    return 0 if bos is None else 1


def n_forwarded(toks, length: int, bos, hw: int) -> int:
    """Tokens in the rows `UR.pack` builds for `toks`, BOS included, padding not."""
    t = np.asarray(toks, dtype=np.int64)
    n = t % hw + 1 if length is None else np.minimum(t % hw + 1, length)
    return int(n.sum() + (0 if bos is None else len(t)))


# ------------------------------------------------------------------------------------------ the replacement value
@torch.inference_mode()
def background_mean_a(model, positions, n_layers: int, *, tokens, length, bos, batch: int, hw: int, device) -> torch.Tensor:
    """ā [n_layers, W] fp32: each MLP neuron's a = act(g) u at the marked token of the window ending at each position, averaged over positions."""
    assert n_layers >= 1
    cap = UR.capture(model, np.asarray(positions), n_layers - 1, set(), tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    return torch.stack([cap["a"][l].mean(0) for l in range(n_layers)])


def ablation_dict(S: list, replace: str, abar: torch.Tensor, L: int, device, identity: bool = False) -> dict:
    """{l: (idx, values)} for `replaced`. identity: the hooks on every layer below L, with nothing written."""
    assert replace in ("mean", "zero"), replace
    if identity:
        assert not S, "the identity arm has an empty set"
        e = torch.zeros(0, dtype=torch.long, device=device)
        return {l: (e, torch.zeros(0, device=device)) for l in range(L)}
    assert S and len(set(S)) == len(S), "a set is empty or names a neuron twice"
    by = {}
    for l, k in S:
        assert 0 <= l < L, f"layer {l} is not upstream of layer {L}"
        by.setdefault(l, []).append(k)
    out = {}
    for l, ks in by.items():
        idx = torch.tensor(ks, dtype=torch.long, device=device)
        out[l] = (idx, abar[l, idx].float() if replace == "mean" else torch.zeros(len(ks), device=device))
    return out


@contextlib.contextmanager
def replaced(layers, ablation: dict, start: int):
    """A forward pre-hook on layers[l].mlp.down_proj writes values[j] into its input at positions >= start for neuron idx[j]; a forward hook on the
    same module asserts that the input down_proj received holds exactly those values (in its dtype) and counts the calls."""
    rec = {"calls": 0}
    handles = []
    for l, (idx, val) in ablation.items():
        def pre(m, args, idx=idx, val=val):
            a = args[0].clone()
            a[:, start:, idx] = val.to(a.dtype)
            return (a, *args[1:])

        def post(m, args, out, idx=idx, val=val):
            got = args[0][:, start:, idx]
            assert bool((got == val.to(got.dtype)).all()), "down_proj did not receive the replaced values"
            rec["calls"] += 1
        handles += [layers[l].mlp.down_proj.register_forward_pre_hook(pre), layers[l].mlp.down_proj.register_forward_hook(post)]
    yield rec
    for h in handles:
        h.remove()


# ------------------------------------------------------------------------------------------ the readout
@torch.inference_mode()
def readout(model, toks, unit: dict, ablation: dict | None, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """At the marked token of each context: the unit's g, u [N, T] fp32 and a [N] = sum_terms w sign act(g) u, forward stopped after the unit's layer.
    Rows keep the order of `toks`. Each ablated layer's down_proj must run once per batch with the replacement in place."""
    tower = NH.text_tower(model)
    layers = list(tower.layers)
    L = unit["layer"]
    ablation = ablation or {}
    assert all(l < L for l in ablation), "an ablated layer is not upstream of the unit"
    ns = torch.tensor([t["n"] for t in unit["terms"]], device=device)
    ws = torch.tensor([t["w"] * t["sign"] for t in unit["terms"]], device=device, dtype=torch.float32)
    toks = np.asarray(toks)
    order = np.argsort(toks % hw, kind="stable")
    gs, us = [], []
    for b0 in range(0, len(toks), batch):
        idx = order[b0:b0 + batch]
        x, pos = UR.pack(tokens, toks[idx], length, bos, hw, device)
        rows = torch.arange(len(idx), device=device)
        keep = {}
        hs = [layers[L].mlp.gate_proj.register_forward_hook(lambda m, a, o: keep.__setitem__("g", o)),
              layers[L].mlp.up_proj.register_forward_hook(lambda m, a, o: keep.__setitem__("u", o))]
        with UR.upto_layer(tower, L), replaced(layers, ablation, window_start(bos)) as rec:
            NH.forward(model, x, False)
        for h in hs:
            h.remove()
        assert rec["calls"] == len(ablation), f"{rec['calls']} replacement checks for {len(ablation)} ablated layers"
        gs.append(keep["g"][rows, pos][:, ns].float())
        us.append(keep["u"][rows, pos][:, ns].float())
    inv = torch.as_tensor(np.argsort(order), device=device)
    g, u = torch.cat(gs)[inv], torch.cat(us)[inv]
    a = (ws * layers[L].mlp.act_fn(g) * u).sum(1)
    return {"a": a.cpu().numpy(), "g": g.cpu().numpy(), "u": u.cpu().numpy()}


# ------------------------------------------------------------------------------------------ drivers
def direct_A(rows: UR.Rows, cap: dict, unit: dict, W: int) -> torch.Tensor:
    """[N, L W]: UNITE's direct attribution of the unit's activation, with the unit's own coefficients at each context."""
    gc, uc = UR.term_gu(cap, unit)
    return UR.attribution(cap, unit["layer"], UR.unit_terms(rows, unit, gc, uc), W)


def grad_attr_window(model, toks, unit: dict, *, tokens, length, bos, batch: int, hw: int, device) -> torch.Tensor:
    """[N, L W] fp32: sum over window positions t of a_k(t) d a_unit(x) / d a_k(t). unite_route.grad_attr with BOS left out of the sum."""
    tower = NH.text_tower(model)
    L = unit["layer"]
    layers = list(tower.layers)
    start = window_start(bos)
    toks = np.asarray(toks)
    order = np.argsort(toks % hw, kind="stable")
    W = layers[0].mlp.down_proj.weight.shape[1]
    assert L >= 1, "a unit at layer 0 has no upstream MLP"
    out = []
    for b0 in range(0, len(toks), batch):
        idx = order[b0:b0 + batch]
        x, pos = UR.pack(tokens, toks[idx], length, bos, hw, device)
        rows_ = torch.arange(len(idx), device=device)
        kept, gu, handles = {}, {}, []

        def keep_a(l):
            def h(m, args):
                args[0].retain_grad()
                kept[l] = args[0]
            return h
        for l in range(L):
            handles.append(layers[l].mlp.down_proj.register_forward_pre_hook(keep_a(l)))
        handles.append(layers[L].mlp.gate_proj.register_forward_hook(lambda m, a, o: gu.__setitem__("g", o)))
        handles.append(layers[L].mlp.up_proj.register_forward_hook(lambda m, a, o: gu.__setitem__("u", o)))
        with torch.enable_grad(), UR.upto_layer(tower, L):
            emb = tower.embed_tokens(x).detach().requires_grad_(True)
            model(inputs_embeds=emb, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
            value = 0.0
            for t in unit["terms"]:
                g, u = gu["g"][rows_, pos, t["n"]].float(), gu["u"][rows_, pos, t["n"]].float()
                value = value + t["w"] * t["sign"] * layers[L].mlp.act_fn(g) * u
            value.sum().backward()
        for h in handles:
            h.remove()
        out.append(torch.cat([(kept[l].detach().float() * kept[l].grad.float())[:, start:].sum(1) for l in range(L)], 1).detach())
        del kept, gu, emb, value
    res = torch.cat(out)
    assert res.shape == (len(toks), L * W), res.shape
    return res[torch.as_tensor(np.argsort(order), device=device)]


def top_m(score: torch.Tensor, m: int, W: int) -> list:
    """The m (layer, neuron) with the largest signed score, largest first."""
    assert score.ndim == 1 and m <= score.numel(), (score.shape, m)
    i = torch.topk(score, m, sorted=True).indices.tolist()
    return [(j // W, j % W) for j in i]


def drivers_direct(rows: UR.Rows, cap: dict, unit: dict, W: int, ms=MS):
    """{m: D} under the direct ranking, and the per-context attribution A [N, L W] it was summed from."""
    A = direct_A(rows, cap, unit, W)
    D = top_m(A.sum(0), max(ms), W)
    return {m: D[:m] for m in ms}, A


def drivers_gradxact(model, toks, unit: dict, W: int, ms=MS, **kw):
    """{m: D} under the gradient x activation ranking, and the per-context attribution [N, L W]."""
    A = grad_attr_window(model, toks, unit, **kw)
    D = top_m(A.sum(0), max(ms), W)
    return {m: D[:m] for m in ms}, A


def random_matched(D_other: list, D_own: list, abar) -> list:
    """For each k in D_other, in order: the neuron of k's layer nearest to k in |ā|, outside D_own, D_other and the neurons already picked. Deterministic."""
    mag = np.abs(np.asarray(abar.cpu() if torch.is_tensor(abar) else abar, dtype=np.float64))
    taken = set(D_own) | set(D_other)
    out = []
    for l, k in D_other:
        gap = np.abs(mag[l] - mag[l, k])
        gap[[kk for ll, kk in taken if ll == l]] = np.inf
        j = int(np.argmin(gap))
        assert np.isfinite(gap[j]), f"layer {l}: no neuron left to match"
        out.append((l, j))
        taken.add((l, j))
    return out


# ------------------------------------------------------------------------------------------ falls and the linear prediction
def fall(before, after) -> np.ndarray:
    before, after = np.asarray(before, np.float64), np.asarray(after, np.float64)
    return (before - after) / before


def job_fall(before, after) -> dict:
    """The job's fall: the mean over contexts with a_P > 0 (SPEC §2's cancel rule counts those contexts)."""
    before, after = np.asarray(before, np.float64), np.asarray(after, np.float64)
    pos = before > 0
    f = fall(before[pos], after[pos])
    return {"mean": float(f.mean()) if pos.any() else float("nan"), "median": float(np.median(f)) if pos.any() else float("nan"),
            "n_pos": int(pos.sum()), "n": int(before.size)}


def linear_terms(rows: UR.Rows, cap: dict, unit: dict, S: list, r: torch.Tensor, W: int, loads) -> dict:
    """The direct-path prediction of an ablation at each context, for a single-term unit. Per context:
    sum_A    = sum_{k in S} A_k(x) (SPEC §7's linearity column, the change if a_k went to 0 with s_l and the coefficients held),
    sum_A_c  = the same with a_k - r_k (the replacement actually made),
    pred_g, pred_u = sum_{k in S} s_l (a_k - r_k) G.d_k, and the same with U: the direct-path change in g and u."""
    assert len(unit["terms"]) == 1, "linear terms are for a single pole"
    L, t = unit["layer"], unit["terms"][0]
    gc, uc = UR.term_gu(cap, unit)
    cg, cu = UR.coefficients(gc[:, 0], uc[:, 0])
    dev = cap["emb"].device
    flat = torch.tensor([l * W + k for l, k in S], device=dev)
    ls = torch.tensor([l for l, _ in S], device=dev)
    all_a = torch.cat([cap["a"][l] for l in range(L)], 1)[:, flat]
    s = torch.stack([cap["s"][l] for l in range(L)], 1)[:, ls]
    lg = torch.cat([lo[0] for lo in loads])[flat]
    lu = torch.cat([lo[1] for lo in loads])[flat]
    pg0, pu0 = (s * all_a * lg).sum(1), (s * all_a * lu).sum(1)
    pg, pu = (s * (all_a - r) * lg).sum(1), (s * (all_a - r) * lu).sum(1)
    sg = t["sign"] * t["w"]
    return {"sum_A": (sg * (cg * pg0 + cu * pu0)).cpu().numpy(), "sum_A_c": (sg * (cg * pg + cu * pu)).cpu().numpy(),
            "pred_g": pg.cpu().numpy(), "pred_u": pu.cpu().numpy()}


def linfit(x, y) -> dict:
    """Least squares y = slope x + intercept, with R^2."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3 or np.ptp(x) == 0:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "n": int(x.size)}
    slope, icpt = np.polyfit(x, y, 1)
    res = y - (slope * x + icpt)
    return {"slope": float(slope), "intercept": float(icpt), "r2": float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()), "n": int(x.size)}


# ------------------------------------------------------------------------------------------ the summary the freeze reads
def cancel(own: float, rand: float, n_pos: int) -> dict:
    room = not (own >= CANCEL_RATIO * abs(rand))
    few = n_pos < MIN_POS
    return {"no_room": bool(room), "under_3_positive": bool(few), "cancelled": bool(room or few)}


def freeze_rule(own_by: dict, ms=MS) -> dict:
    """SPEC §2: per ranking, the smallest m whose median own fall over jobs is >= 0.25; the ranking with the larger median own fall at that m (tie: direct).
    `own_by` {ranking: {m: [own fall per job]}}."""
    med = {rk: {m: float(np.nanmedian(v[m])) if len(v.get(m, [])) else float("nan") for m in ms} for rk, v in own_by.items()}
    m_of = {rk: next((m for m in ms if med[rk][m] >= FLOOR), None) for rk in med}
    cand = [m for m in m_of.values() if m is not None]
    if not cand:
        return {"median_own": med, "m_by_ranking": m_of, "m": None, "ranking": None, "why": "no m reaches 0.25 under either ranking: no room"}
    m = min(cand)
    rk = max(med, key=lambda r: (med[r][m], r == "direct"))
    return {"median_own": med, "m_by_ranking": m_of, "m": m, "ranking": rk,
            "why": f"smallest m with median own fall >= {FLOOR} is {m}; at m {m} the larger median own fall is {rk}"}


# ------------------------------------------------------------------------------------------ pure-logic checks on random tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    b = rng.uniform(0.5, 2.0, 50)
    assert np.all(fall(b, b) == 0.0), "fall of identical arrays is not exactly 0"
    assert np.allclose(fall([2.0, 4.0], [1.0, 1.0]), [0.5, 0.75])
    jf = job_fall([2.0, -1.0, 4.0, 0.0], [1.0, -3.0, 1.0, 5.0])
    assert jf["n_pos"] == 2 and abs(jf["mean"] - 0.625) < 1e-12, jf
    out["fall"] = "identity exactly 0; (2,4)->(1,1) gives (0.5, 0.75); job mean over a_P > 0 only"
    # random_matched: count, exclusion, nearest in |ā|, no repeats
    L, W = 4, 40
    abar = rng.normal(0, 1, (L, W))
    D_other = [(1, 3), (1, 4), (2, 10), (3, 0), (3, 1)]
    D_own = [(1, 5), (2, 11), (0, 7)]
    R = random_matched(D_other, D_own, abar)
    assert len(R) == len(D_other), "random-matched count differs"
    assert not set(R) & (set(D_other) | set(D_own)), "random-matched picked a forbidden neuron"
    assert len(set(R)) == len(R), "random-matched picked a neuron twice"
    assert [l for l, _ in R] == [l for l, _ in D_other], "random-matched left a layer"
    mag = np.abs(abar)
    for (l, k), (_, j) in zip(D_other, R):
        allowed = [x for x in range(W) if (l, x) not in set(D_other) | set(D_own) | set(R[:R.index((l, j))])]
        assert j == min(allowed, key=lambda x: (abs(mag[l, x] - mag[l, k]), x)), "random-matched is not the nearest allowed"
    # a constructed case where the nearest neighbour is forbidden
    ab = np.zeros((1, 6))
    ab[0] = [1.0, 1.01, 1.02, 5.0, 9.0, 1.03]
    assert random_matched([(0, 0)], [(0, 1)], ab) == [(0, 2)], random_matched([(0, 0)], [(0, 1)], ab)
    assert random_matched([(0, 0), (0, 2)], [(0, 1)], ab) == [(0, 5), (0, 3)], random_matched([(0, 0), (0, 2)], [(0, 1)], ab)
    out["random_matched"] = "count equal, forbidden sets excluded, no repeats, nearest allowed in |ā|"
    # top_m
    sc = torch.tensor(rng.normal(0, 1, L * W))
    D = top_m(sc, 7, W)
    flat = [l * W + k for l, k in D]
    assert flat == torch.argsort(sc, descending=True)[:7].tolist(), "top_m is not the largest first"
    out["top_m"] = "largest signed score first"
    # linfit and freeze_rule
    x = rng.normal(0, 1, 30)
    f = linfit(x, 2 * x + 1)
    assert abs(f["slope"] - 2) < 1e-9 and abs(f["r2"] - 1) < 1e-9
    fr = freeze_rule({"direct": {20: [0.1, 0.2], 50: [0.3, 0.3], 100: [0.5, 0.6]}, "grad": {20: [0.1, 0.1], 50: [0.4, 0.5], 100: [0.7, 0.8]}})
    assert fr["m"] == 50 and fr["ranking"] == "grad", fr
    fr2 = freeze_rule({"direct": {20: [0.1], 50: [0.2], 100: [0.2]}, "grad": {20: [0.0], 50: [0.1], 100: [0.1]}})
    assert fr2["m"] is None and fr2["ranking"] is None, fr2
    fr3 = freeze_rule({"direct": {20: [0.3], 50: [0.4], 100: [0.5]}, "grad": {20: [0.3], 50: [0.4], 100: [0.5]}})
    assert fr3["m"] == 20 and fr3["ranking"] == "direct", fr3
    assert cancel(0.3, 0.2, 5)["cancelled"] and not cancel(0.5, 0.2, 5)["cancelled"] and cancel(0.5, 0.1, 2)["cancelled"] and cancel(float("nan"), 0.1, 5)["cancelled"]
    out["freeze_rule"] = "smallest m with median own >= 0.25, larger ranking, tie to direct, none -> no room; cancel on room and on positives"
    return out
