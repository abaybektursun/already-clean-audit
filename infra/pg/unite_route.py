"""UNITE Arm 1 instrument (studies/unite/PREREG.md §2): first-order upstream attribution of a pole's activation, per context.

For pole P of neuron n at layer L, sign sigma, context x, in one clean forward (bf16, arithmetic in fp32):
    a = silu(g) u,  c_g = silu'(g) u,  c_u = silu(g),  g = G x_mlp,  u = U x_mlp.
    For neuron k of layer l < L:  A_k(x) = sigma a_k(x) [c_g(x) G.d_k + c_u(x) U.d_k] s_l(x),
    d_k = gamma_l * W_down_l[:, k],  s_l(x) = 1 / rms of post_feedforward_layernorm's input at x.
OLMo 3's residual stream gives x_mlp at layer L = embedding + sum_{l<L} (attention_l + mlp_l) + attention_L, each after its own RMSNorm,
so the same terms add back to g and u: that is the completeness gate. A planted unit is a list of terms with weights, so rows and
attributions add with the same lambda. M_ij = (A(x_i) - e1) . (A(x_j) - e2) is banked by rank, so any partition is scored offline.

Reuses `neurons_harvest.MLPTap`, `TopK`, `forward`, `context` and `neuron_rw_core.norm_gain_exact`, `inv_rms`, `norm_eps`.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn as nn

from infra.pg import neurons_harvest as NH
from infra.pg import unite_stats as US
from infra.pg.neuron_rw_core import inv_rms, norm_eps, norm_gain_exact

HARVEST_WINDOW = 2048
SOURCES = ("mlp", "attention", "embedding", "bias")
TOP_INPUTS = 20
N_E, N_NOISE = 200, 100


# ------------------------------------------------------------------------------------------ units and positions
def pole_unit(pole: str) -> dict:
    L, n, s = NH.parse_pole(pole)
    return {"unit": pole, "layer": L, "terms": [{"n": n, "sign": 1.0 if s == 0 else -1.0, "w": 1.0}]}


def planted_unit(rec: dict) -> dict:
    return {"unit": rec["unit"], "layer": rec["layer"], "terms": [{"n": rec["a"], "sign": 1.0, "w": 1.0}, {"n": rec["b"], "sign": 1.0, "w": float(rec["lam"])}]}


def background_positions(t_index: int, n_windowed: int, hw: int = HARVEST_WINDOW) -> dict:
    """Two disjoint sets of 200 random corpus positions and 100 for the W noise level, window position >= 1, fixed by seed per checkpoint."""
    rng = np.random.default_rng(1000 + t_index)
    n = 2 * N_E + N_NOISE
    pos = rng.choice(n_windowed, size=2 * n, replace=False)
    pos = pos[pos % hw != 0][:n]
    assert pos.size == n, pos.size
    return {"e1": pos[:N_E], "e2": pos[N_E:2 * N_E], "noise": pos[2 * N_E:]}


def pack(tokens: np.ndarray, toks, length, bos, hw: int, device, pad: int = 0):
    """Right-padded ids ending at each position x: the harvest's window prefix, cut to `length` tokens (None = the full prefix), BOS first
    when the tokenizer defines one. Attention is causal, so padding after x cannot reach x."""
    rows = []
    for t in toks:
        t = int(t)
        start = t - t % hw
        if length is not None:
            start = max(start, t - length + 1)
        rows.append(([bos] if bos is not None else []) + np.asarray(tokens[start:t + 1]).astype(np.int64).tolist())
    S = max(map(len, rows))
    x = torch.full((len(rows), S), pad, dtype=torch.long)
    for i, r in enumerate(rows):
        x[i, :len(r)] = torch.tensor(r, dtype=torch.long)
    return x.to(device), torch.tensor([len(r) - 1 for r in rows], device=device)


@contextlib.contextmanager
def upto_layer(tower, L: int):
    """The forward stops after layer L: nothing later can affect L's inputs."""
    full = tower.layers
    tower.layers = nn.ModuleList(list(full)[:L + 1])
    yield
    tower.layers = full


# ------------------------------------------------------------------------------------------ capture
@torch.inference_mode()
def capture(model, toks, upto: int, gu_layers, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """At each position in `toks`, for layers 0..upto: a [N, W] (fp32 recomputed), s [N], attention and MLP vectors after their norms [N, d];
    g, u [N, W] for `gu_layers`; embedding [N, d]. Rows keep the order of `toks`."""
    tower = NH.text_tower(model)
    layers = list(tower.layers)[:upto + 1]
    acts = [l.mlp.act_fn for l in layers]
    tap = NH.MLPTap(layers)
    toks = np.asarray(toks)
    order = np.argsort(toks % hw, kind="stable")                # similar prefix lengths batch together
    out = {"a": {}, "g": {}, "u": {}, "s": {}, "attn": {}, "mlp": {}, "emb": []}
    for b0 in range(0, len(toks), batch):
        idx = order[b0:b0 + batch]
        x, pos = pack(tokens, toks[idx], length, bos, hw, device)
        rows = torch.arange(len(idx), device=device)

        def at(v):
            return v[rows, pos].float()

        def fn(lyr, g, u, a_model):
            gg, uu = at(g), at(u)
            out["a"].setdefault(lyr, []).append(acts[lyr](gg) * uu)
            if lyr in gu_layers:
                out["g"].setdefault(lyr, []).append(gg)
                out["u"].setdefault(lyr, []).append(uu)

        handles = [tower.embed_tokens.register_forward_hook(lambda m, a, o: out["emb"].append(at(o)))]
        for lyr, layer in enumerate(layers):
            eps = norm_eps(layer.post_feedforward_layernorm)

            def on_attn(m, a, o, lyr=lyr):
                out["attn"].setdefault(lyr, []).append(at(o))

            def on_ff(m, a, o, lyr=lyr, eps=eps):
                out["s"].setdefault(lyr, []).append(inv_rms(at(a[0]), eps))
                out["mlp"].setdefault(lyr, []).append(at(o))
            handles += [layer.post_attention_layernorm.register_forward_hook(on_attn), layer.post_feedforward_layernorm.register_forward_hook(on_ff)]
        with upto_layer(tower, upto):
            tap.run(fn, lambda: NH.forward(model, x, False))
        for h in handles:
            h.remove()
    inv = torch.as_tensor(np.argsort(order), device=device)
    res = {k: {l: torch.cat(v)[inv] for l, v in out[k].items()} for k in ("a", "g", "u", "s", "attn", "mlp")}
    res["emb"] = torch.cat(out["emb"])[inv]
    return res


def cap_rows(cap: dict, idx) -> dict:
    idx = torch.as_tensor(np.asarray(idx), device=cap["emb"].device)
    res = {k: {l: v[idx] for l, v in cap[k].items()} for k in ("a", "g", "u", "s", "attn", "mlp")}
    res["emb"] = cap["emb"][idx]
    return res


# ------------------------------------------------------------------------------------------ rows and attributions
class Rows:
    """gamma_l * W_down_l in fp32, once per load, and each neuron's input rows. `shift` != 0 reads the column from the wrong layer: the failing input."""

    def __init__(self, model):
        self.layers = list(NH.text_tower(model).layers)
        self.d = [None] * len(self.layers)
        assert all(l.mlp.gate_proj.bias is None and l.mlp.up_proj.bias is None and l.mlp.down_proj.bias is None for l in self.layers), "an MLP has a bias: the bias term is not 0"

    def column(self, l: int, shift: int = 0):
        src = (l + shift) % len(self.layers)
        if shift:
            return norm_gain_exact(self.layers[l].post_feedforward_layernorm)[:, None] * self.layers[src].mlp.down_proj.weight.float()
        if self.d[l] is None:
            self.d[l] = norm_gain_exact(self.layers[l].post_feedforward_layernorm)[:, None] * self.layers[l].mlp.down_proj.weight.float()
        return self.d[l]

    def inputs(self, L: int, n: int):
        return self.layers[L].mlp.gate_proj.weight[n].float(), self.layers[L].mlp.up_proj.weight[n].float()

    def loads(self, L: int, n: int, shift: int = 0):
        """Per l < L: (G.d_k, U.d_k), each [W]."""
        G, U = self.inputs(L, n)
        return [(self.column(l, shift).T @ G, self.column(l, shift).T @ U) for l in range(L)]

    def foreign(self, L: int, n: int, also_exclude=()) -> int:
        """Another neuron of the layer, nearest in ||G||^2 + ||U||^2. The rule is deterministic, so no seed is needed."""
        m = self.layers[L].mlp
        load = m.gate_proj.weight.float().pow(2).sum(1) + m.up_proj.weight.float().pow(2).sum(1)
        gap = (load - load[n]).abs()
        gap[[n, *also_exclude]] = float("inf")
        return int(gap.argmin())


def coefficients(g, u):
    """c_g = silu'(g) u, c_u = silu(g). The job asserts the activation measures as silu."""
    sig = torch.sigmoid(g)
    return (sig + g * sig * (1 - sig)) * u, g * sig


def attribution(cap: dict, L: int, terms: list, W: int) -> torch.Tensor:
    """[N, L W]: sum over terms of weight * sign * a * s * (c_g G.d + c_u U.d). `terms`: (w, sign, c_g, c_u, loads)."""
    N = cap["emb"].shape[0]
    if L == 0:
        return torch.zeros(N, 0, device=cap["emb"].device)
    parts = []
    for l in range(L):
        base = cap["a"][l] * cap["s"][l][:, None]
        parts.append(sum(w * sgn * base * (cg[:, None] * lo[l][0][None] + cu[:, None] * lo[l][1][None]) for w, sgn, cg, cu, lo in terms))
    return torch.cat(parts, 1)


def gram(A, e1, e2) -> torch.Tensor:
    return (A - e1) @ (A - e2).T


def unit_terms(rows: Rows, unit: dict, cap_g, cap_u, foreign: bool = False, fixed=None):
    """(w, sign, c_g, c_u, loads) per term. foreign: the same coefficients under another neuron's rows. fixed: coefficients held at given values."""
    L, out, skip = unit["layer"], [], [t["n"] for t in unit["terms"]]
    for i, t in enumerate(unit["terms"]):
        cg, cu = coefficients(cap_g[:, i], cap_u[:, i])
        if fixed is not None:
            cg, cu = torch.full_like(cg, fixed[i][0]), torch.full_like(cu, fixed[i][1])
        n = rows.foreign(L, t["n"], skip) if foreign else t["n"]
        out.append((t["w"], t["sign"], cg, cu, rows.loads(L, n)))
    return out


def term_gu(cap: dict, unit: dict):
    idx = torch.tensor([t["n"] for t in unit["terms"]], device=cap["emb"].device)
    L = unit["layer"]
    return cap["g"][L][:, idx], cap["u"][L][:, idx]


def source_terms(rows: Rows, cap: dict, L: int, n: int) -> dict:
    """Each source's part of g and of u at each row: {'mlp': (g [N], u [N]), ...}, and the MLP part per layer."""
    G, U = rows.inputs(L, n)
    N = cap["emb"].shape[0]
    zero = torch.zeros(N, device=cap["emb"].device)

    def part(v):
        return v @ G, v @ U
    mlp_l = [part(cap["mlp"][l]) for l in range(L)]
    attn_l = [part(cap["attn"][l]) for l in range(L + 1)]
    return {"mlp": (sum((x[0] for x in mlp_l), zero), sum((x[1] for x in mlp_l), zero)),
            "attention": (sum(x[0] for x in attn_l), sum(x[1] for x in attn_l)), "embedding": part(cap["emb"]), "bias": (zero, zero),
            "mlp_by_layer": mlp_l}


def neuron_sum(rows: Rows, cap: dict, L: int, n: int, shift: int = 0):
    """sum_l s_l sum_k a_k (G.d_k), the same for U: the MLP terms rebuilt neuron by neuron."""
    zero = torch.zeros(cap["emb"].shape[0], device=cap["emb"].device)
    g, u = zero.clone(), zero.clone()
    for l, (lg, lu) in enumerate(rows.loads(L, n, shift)):
        g += cap["s"][l] * (cap["a"][l] @ lg)
        u += cap["s"][l] * (cap["a"][l] @ lu)
    return g, u


def completeness(rows: Rows, cap: dict, L: int, n: int, shift: int = 0) -> dict:
    """Errors of G.sum(terms) against g and U.sum(terms) against u, each over the pole's max at these contexts."""
    st = source_terms(rows, cap, L, n)
    g_mlp, u_mlp = neuron_sum(rows, cap, L, n, shift)
    g_hat = g_mlp + st["attention"][0] + st["embedding"][0]
    u_hat = u_mlp + st["attention"][1] + st["embedding"][1]
    g, u = cap["g"][L][:, n], cap["u"][L][:, n]
    return {"err_g": ((g_hat - g).abs() / g.abs().max()).cpu().numpy(), "err_u": ((u_hat - u).abs() / u.abs().max()).cpu().numpy()}


# ------------------------------------------------------------------------------------------ one unit at one checkpoint
@torch.inference_mode()
def unit_bank(rows: Rows, unit: dict, cap: dict, bg: dict, sets: dict, noise: bool = False) -> dict:
    """Grams and the per-context table for one unit. `cap`: its contexts; `bg`: the background capture, rows ordered e1, e2, noise."""
    L, W = unit["layer"], rows.layers[0].mlp.down_proj.weight.shape[1]
    nE1, nE2 = len(sets["e1"]), len(sets["e2"])
    b1 = cap_rows(bg, np.arange(nE1))
    b2 = cap_rows(bg, np.arange(nE1, nE1 + nE2))
    gc, uc = term_gu(cap, unit)
    A = attribution(cap, L, unit_terms(rows, unit, gc, uc), W)

    def centres(foreign=False, fixed=None):
        out = []
        for b in (b1, b2):
            gb, ub = term_gu(b, unit)
            out.append(attribution(b, L, unit_terms(rows, unit, gb, ub, foreign=foreign, fixed=fixed), W).mean(0))
        return out
    e1, e2 = centres()
    M = gram(A, e1, e2)
    Af = attribution(cap, L, unit_terms(rows, unit, gc, uc, foreign=True), W)
    f1, f2 = centres(foreign=True)
    Mf = gram(Af, f1, f2)
    fixed = [(float(coefficients(gc[:, i], uc[:, i])[0].median()), float(coefficients(gc[:, i], uc[:, i])[1].median())) for i in range(len(unit["terms"]))]
    Afx = attribution(cap, L, unit_terms(rows, unit, gc, uc, fixed=fixed), W)
    x1, x2 = centres(fixed=fixed)
    Mx = gram(Afx, x1, x2)

    # the table: activation, coefficients, scales, top inputs, the rise over background by source
    N = cap["emb"].shape[0]
    a_terms = torch.stack([t["w"] * t["sign"] * rows.layers[L].mlp.act_fn(gc[:, i]) * uc[:, i] for i, t in enumerate(unit["terms"])], 1)
    rise = torch.zeros(N, len(SOURCES), device=A.device)
    rise_l = torch.zeros(N, L, device=A.device)
    err_g, err_u = [], []
    bg_all = cap_rows(bg, np.arange(nE1 + nE2))
    for i, t in enumerate(unit["terms"]):
        cg, cu = coefficients(gc[:, i], uc[:, i])
        st, sb = source_terms(rows, cap, L, t["n"]), source_terms(rows, bg_all, L, t["n"])
        for j, src in enumerate(SOURCES):
            dg = st[src][0] - sb[src][0].mean()
            du = st[src][1] - sb[src][1].mean()
            rise[:, j] += t["w"] * t["sign"] * (cg * dg + cu * du)
        for l in range(L):
            dg = st["mlp_by_layer"][l][0] - sb["mlp_by_layer"][l][0].mean()
            du = st["mlp_by_layer"][l][1] - sb["mlp_by_layer"][l][1].mean()
            rise_l[:, l] += t["w"] * t["sign"] * (cg * dg + cu * du)
        ce = completeness(rows, cap, L, t["n"])
        err_g.append(ce["err_g"])
        err_u.append(ce["err_u"])
    table = {"a_terms": a_terms.cpu().numpy(), "g": gc.cpu().numpy(), "u": uc.cpu().numpy(), "c_g": coefficients(gc, uc)[0].cpu().numpy(), "c_u": coefficients(gc, uc)[1].cpu().numpy(),
             "s": torch.stack([cap["s"][l] for l in range(L)], 1).cpu().numpy() if L else np.zeros((N, 0), np.float32),
             "rise": rise.cpu().numpy(), "rise_by_layer": rise_l.cpu().numpy(), "err_g": np.stack(err_g, 1), "err_u": np.stack(err_u, 1)}
    if L:
        k = min(TOP_INPUTS, A.shape[1])
        v, i = A.abs().topk(k, dim=1)
        all_a = torch.cat([cap["a"][l] for l in range(L)], 1)
        table.update({"top_layer": (i // W).cpu().numpy().astype(np.int16), "top_neuron": (i % W).cpu().numpy().astype(np.int32),
                      "top_A": A.gather(1, i).cpu().numpy(), "top_a": all_a.gather(1, i).cpu().numpy()})
    out = {"M": M.cpu().numpy(), "M_foreign": Mf.cpu().numpy(), "M_fixed": Mx.cpu().numpy(), **table}
    if noise:
        nb = cap_rows(bg, np.arange(nE1 + nE2, nE1 + nE2 + len(sets["noise"])))
        gn, un = term_gu(nb, unit)
        out["M_noise"] = gram(attribution(nb, L, unit_terms(rows, unit, gn, un), W), e1, e2).cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------ gradient readout
def grad_attr(model, toks, unit: dict, *, tokens, length, bos, batch: int, hw: int, device) -> torch.Tensor:
    """A^grad_k(x) = sigma sum_{t<=x} a_k(t) d a_P(x) / d a_k(t), [N, L W] fp32, one backward per batch. It holds every path, attention included."""
    tower = NH.text_tower(model)
    L = unit["layer"]
    layers = list(tower.layers)
    toks = np.asarray(toks)
    order = np.argsort(toks % hw, kind="stable")
    out = []
    W = layers[0].mlp.down_proj.weight.shape[1]
    if L == 0:
        return torch.zeros(len(toks), 0, device=device)
    for b0 in range(0, len(toks), batch):
        idx = order[b0:b0 + batch]
        x, pos = pack(tokens, toks[idx], length, bos, hw, device)
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
        with torch.enable_grad(), upto_layer(tower, L):
            emb = tower.embed_tokens(x).detach().requires_grad_(True)
            model(inputs_embeds=emb, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1)
            value = 0.0
            for t in unit["terms"]:
                g, u = gu["g"][rows_, pos, t["n"]].float(), gu["u"][rows_, pos, t["n"]].float()
                value = value + t["w"] * t["sign"] * layers[L].mlp.act_fn(g) * u
            value.sum().backward()
        for h in handles:
            h.remove()
        out.append(torch.cat([(kept[l].detach().float() * kept[l].grad.float()).sum(1) for l in range(L)], 1).detach())
        del kept, gu, emb, value
    res = torch.cat(out)
    assert res.shape == (len(toks), L * W), res.shape
    return res[torch.as_tensor(np.argsort(order), device=device)]


def grad_bank(model, unit: dict, toks, sets: dict, noise: bool = False, **kw) -> dict:
    """The Gram under the frozen centring sets; `noise` adds the noise positions' Gram for the W rule."""
    A = grad_attr(model, toks, unit, **kw)
    e1 = grad_attr(model, sets["e1"], unit, **kw).mean(0)
    e2 = grad_attr(model, sets["e2"], unit, **kw).mean(0)
    out = {"M": gram(A, e1, e2).cpu().numpy()}
    if noise:
        out["M_noise"] = gram(grad_attr(model, sets["noise"], unit, **kw), e1, e2).cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------ the planted scan
@torch.inference_mode()
def scan_planted(model, plants: list, *, tokens, n_windows: int, window: int, bos, batch: int, device, progress=None) -> dict:
    """Top 50 of a_plant = a_A + lambda a_B over every full window, forward stopped after the last planted layer. Window position 0 and BOS are excluded,
    as in the harvest. TopK's g and u slots hold the two terms, so the larger term at each context is banked with it."""
    tower = NH.text_tower(model)
    Lmax = max(p["layer"] for p in plants)
    layers = list(tower.layers)[:Lmax + 1]
    tap = NH.MLPTap(layers)
    by_L = {}
    for p in plants:
        by_L.setdefault(p["layer"], []).append(p)
    A = {L: torch.tensor([p["a"] for p in ps], device=device) for L, ps in by_L.items()}
    B = {L: torch.tensor([p["b"] for p in ps], device=device) for L, ps in by_L.items()}
    lam = {L: torch.tensor([p["lam"] for p in ps], device=device, dtype=torch.float32) for L, ps in by_L.items()}
    top = {L: NH.TopK(len(ps), NH.K_TOP, device) for L, ps in by_L.items()}
    off = 0 if bos is None else 1
    for bi in range(0, n_windows, batch):
        w0, w1 = bi, min(bi + batch, n_windows)
        x = torch.from_numpy(np.asarray(tokens[w0 * window:w1 * window]).astype(np.int64)).view(w1 - w0, window)
        if bos is not None:
            x = torch.cat([torch.full((w1 - w0, 1), bos, dtype=torch.long), x], 1)
        x = x.to(device)
        tok_rest = torch.from_numpy((np.arange(w0, w1)[:, None] * window + np.arange(1, window)[None]).reshape(-1)).to(device)

        def fn(lyr, g, u, a_model):
            if lyr in by_L:
                g, u = g[:, off + 1:].reshape(-1, g.shape[-1]), u[:, off + 1:].reshape(-1, u.shape[-1])
                ga, ua = g[:, A[lyr]].float(), u[:, A[lyr]].float()
                gb, ub = g[:, B[lyr]].float(), u[:, B[lyr]].float()
                ta, tb = layers[lyr].mlp.act_fn(ga) * ua, lam[lyr] * layers[lyr].mlp.act_fn(gb) * ub
                top[lyr].update(ta + tb, ta, tb, tok_rest)
        with upto_layer(tower, Lmax):
            tap.run(fn, lambda: NH.forward(model, x, False))
        if progress is not None:
            progress(w1)
    res = {}
    for L, ps in by_L.items():
        arr = top[L].arrays()
        for i, p in enumerate(ps):
            res[p["unit"]] = {"tok": arr["tok"][i, 0], "a_plant": arr["a"][i, 0], "term_a": arr["g"][i, 0], "term_b": arr["u"][i, 0]}
    return res


def select_planted(plants: list, scan: dict, tokens, primary_keep: int = 15, want: int = 20, min_side: int = 15) -> dict:
    """J = contexts where A's term is larger, O = the rest. A pair is kept only if each side has at least 15 contexts; then contexts whose marked token occurs on
    the other side are dropped; a side left under 3 contexts drops the pair. A layer that keeps under 15 primary pairs takes reserve pairs in listed order, up to 20."""
    out, by_L = {}, {}
    for p in plants:
        by_L.setdefault(p["layer"], []).append(p)
    for L, ps in by_L.items():
        n_kept = 0
        for stage in ("primary", "reserve"):
            for p in ps:
                if (stage == "reserve") != bool(p["reserve"]):
                    continue
                if stage == "reserve" and (n_primary_kept >= primary_keep or n_kept >= want):
                    out[p["unit"]] = {**p, "kept": False, "why": "reserve not used"}
                    continue
                s = scan[p["unit"]]
                valid = np.flatnonzero(s["tok"] >= 0)
                J = valid[s["term_a"][valid] > s["term_b"][valid]]
                O = valid[~(s["term_a"][valid] > s["term_b"][valid])]
                rec = {**p, "tok": s["tok"].tolist(), "a_plant": s["a_plant"].tolist(), "term_a": s["term_a"].tolist(), "term_b": s["term_b"].tolist(),
                       "n_J_raw": len(J), "n_O_raw": len(O), "kept": False}
                if len(J) < min_side or len(O) < min_side:
                    out[p["unit"]] = {**rec, "why": "a side under 15 contexts"}
                    continue
                tj, to = np.asarray(tokens[s["tok"][J]]), np.asarray(tokens[s["tok"][O]])
                j2, o2 = J[~np.isin(tj, to)], O[~np.isin(to, tj)]
                if len(j2) < 3 or len(o2) < 3:
                    out[p["unit"]] = {**rec, "why": "a side under 3 contexts after dropping shared tokens"}
                    continue
                out[p["unit"]] = {**rec, "kept": True, "J": j2.tolist(), "O": o2.tolist(), "used_as": stage}
                n_kept += 1
            if stage == "primary":
                n_primary_kept = n_kept
    return out


# ------------------------------------------------------------------------------------------ job readout on a Gram (samples only; the verdict is offline)
def job_rows(jobs: list, grams: dict, field: str = "M") -> list:
    rows = []
    for j in jobs:
        if j["pole"] not in grams:
            continue
        M = grams[j["pole"]][field]
        o = next(o for o in j["older"] if o["c"] == j["o_star"])
        rows.append({"job": j["job"], "pole": j["pole"], "J": j["ranks"], "O": o["ranks"], **US.stat_T(M, j["ranks"], o["ranks"])})
    return rows
