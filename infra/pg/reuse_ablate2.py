"""REUSE heads and output instrument (studies/reuse/SPEC_B_heads.md §3, studies/reuse/SPEC_A_output.md §3).

Heads. OLMo 3 adds its attention block to the residual stream as post_attention_layernorm(o_proj(z)), z the concatenated head outputs at each
position, head h in columns h*hd:(h+1)*hd (the head count and hd read from the model config). With s_attn(l, x) = 1 / rms of o_proj's output at x
(recorded at the norm, as unite_route records s at the post-feedforward norm) and gamma the norm's exact gain, the block is
sum_h s_attn gamma * (W_o[:, h] @ z_h). Head (l, h)'s direct part of the pole's activation at the marked token, with the pole's own coefficients:
    A_(l,h)(x) = sum_terms w sign s_attn(l, x) [c_g (W_o[:, h]^T (gamma * G)) . z_h(x) + c_u (W_o[:, h]^T (gamma * U)) . z_h(x)],
as unite_route.attribution reads an MLP neuron's s_l a_k (G . d_k). Summed over the heads of layer l it equals the captured post-norm attention vector
read through G and U: the head capture check. Ablation: a forward pre-hook on self_attn.o_proj writes z̄_h (the head's mean over UNITE's e1 positions
at this checkpoint) into its columns at every window position; a forward hook asserts the write, as reuse_ablate.replaced does.

Output. P writes raw_a_P(x) d_P into the residual stream, d_P = gamma_ff(L) * W_down(L)[:, P] scaled by s_L(x) (layer L's post-feedforward norm, as
unite_route reads every upstream write); a downstream MLP neuron k at layer l > L reads the raw residual through G_k and U_k (OLMo 3 has no norm before
its MLP), so to first order
    dg_k = s_L raw_a_P (G_k . d_P),  du_k = s_L raw_a_P (U_k . d_P),  C_k = da_k = silu'(g_k) u_k dg_k + silu(g_k) du_k.
raw_a_P is the neuron's own output silu(g) u (no sign, no weight; a planted unit writes both its neurons' outputs). The held forward: every neuron of
P replaced by its background mean at window positions (reuse_ablate.ablation_dict and replaced on layer L itself), forward to the last layer.
    E(S; x) = sum_{k in S} |a_k(x) - a_k^P(x)| / sum_{k in S} |a_k(x)|,  a_k from the identity forward (hook on layer L, nothing written).
    share(S; x) = sum_{k in S} |a_k(x) - a_k^P(x)| / sum over every MLP neuron above L of |a_k(x) - a_k^P(x)|.

Reuses unite_route (capture, cap_rows, coefficients, Rows, term_gu, pack, upto_layer) and reuse_ablate (readout, ablation_dict, replaced, top_m,
random_matched, window_start), all unchanged. Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from infra.pg import neurons_harvest as NH
from infra.pg import reuse_ablate as RA
from infra.pg import unite_route as UR
from infra.pg.neuron_rw_core import inv_rms, norm_eps, norm_gain_exact

HS = (10, 20, 40)          # SPEC B §3: h candidates
RS = (10, 20, 50)          # SPEC A §3: r candidates
HEAD_CHECK_TOL = 1e-2      # head capture check, relative; REUSE's completeness line (SPEC §5), set at freeze: the per-head sum is compared with the bf16-captured attention block, whose rounding puts it at 1e-3–3e-3 on bf16 models; the wrong-layer failing input sits at 0.45 or more
READER_CHECK_TOL = 1e-3    # reader capture check, relative: two fp32 computations of one quantity from the same captured tensors, as SPEC A §5 wrote
CHECK_TOLS = {"head_check": HEAD_CHECK_TOL, "reader_check": READER_CHECK_TOL}
ID_TOL = RA.ID_TOL
TOP_LOGIT = 50


# ------------------------------------------------------------------------------------------ heads: layout and capture
def heads_of(model) -> tuple:
    """(n_heads, head_dim, n_kv_heads) from the config, checked against every layer's o_proj and attention module."""
    cfg = getattr(model.config, "text_config", model.config)
    H, kv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    hd = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // H)
    for i, layer in enumerate(NH.text_tower(model).layers):
        o = layer.self_attn.o_proj
        assert o.in_features == H * hd and o.bias is None, f"layer {i}: o_proj is {o.in_features} wide with bias {o.bias is not None}, not {H} x {hd} without"
        assert getattr(layer.self_attn, "head_dim", hd) == hd, f"layer {i}: attention head_dim {layer.self_attn.head_dim} is not the config's {hd}"
    return H, hd, kv


@torch.inference_mode()
def capture(model, toks, upto: int, gu_layers, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """unite_route.capture's outputs plus, for layers 0..upto at the marked token: z [N, H, hd] (self_attn.o_proj's input, fp32) and s_attn [N]
    (1 / rms of post_attention_layernorm's input). The batches are unite_route.capture's (its order and packing, rebuilt here); a pre-hook on
    embed_tokens asserts each batch's ids before the layer hooks read that batch's positions."""
    tower = NH.text_tower(model)
    layers = list(tower.layers)[:upto + 1]
    H, hd, _ = heads_of(model)
    toks = np.asarray(toks)
    order = np.argsort(toks % hw, kind="stable")
    plan = [UR.pack(tokens, toks[order[b0:b0 + batch]], length, bos, hw, device) for b0 in range(0, len(toks), batch)]
    st = {"i": -1}
    z, s = {l: [] for l in range(len(layers))}, {l: [] for l in range(len(layers))}

    def on_embed(m, args):
        st["i"] += 1
        x, pos = plan[st["i"]]
        assert torch.equal(args[0], x), "the forwarded batch is not unite_route.capture's batch"
        st["rows"], st["pos"] = torch.arange(len(pos), device=device), pos

    handles = [tower.embed_tokens.register_forward_pre_hook(on_embed)]
    for l, layer in enumerate(layers):
        eps = norm_eps(layer.post_attention_layernorm)

        def on_o(m, args, l=l):
            z[l].append(args[0][st["rows"], st["pos"]].float().view(-1, H, hd))

        def on_norm(m, args, out, l=l, eps=eps):
            s[l].append(inv_rms(args[0][st["rows"], st["pos"]], eps))
        handles += [layer.self_attn.o_proj.register_forward_pre_hook(on_o), layer.post_attention_layernorm.register_forward_hook(on_norm)]
    cap = UR.capture(model, toks, upto, gu_layers, tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    for h in handles:
        h.remove()
    assert st["i"] == len(plan) - 1, f"{st['i'] + 1} forwards for {len(plan)} batches"
    inv = torch.as_tensor(np.argsort(order), device=device)
    cap["z"] = {l: torch.cat(v)[inv] for l, v in z.items()}
    cap["s_attn"] = {l: torch.cat(v)[inv] for l, v in s.items()}
    return cap


def cap_rows(cap: dict, idx) -> dict:
    """unite_route.cap_rows, keeping z and s_attn."""
    out = UR.cap_rows(cap, idx)
    i = torch.as_tensor(np.asarray(idx), device=cap["emb"].device)
    for k in ("z", "s_attn"):
        out[k] = {l: v[i] for l, v in cap[k].items()}
    return out


class HeadRows:
    """gamma_attn(l) * W_o(l) in fp32, once per load, and each pole's rows read through it."""

    def __init__(self, model):
        self.layers = list(NH.text_tower(model).layers)
        self.H, self.hd, self.kv = heads_of(model)
        self.g = [None] * len(self.layers)

    def gained(self, l: int):
        if self.g[l] is None:
            self.g[l] = norm_gain_exact(self.layers[l].post_attention_layernorm)[:, None] * self.layers[l].self_attn.o_proj.weight.float()
        return self.g[l]

    def loads(self, rows: UR.Rows, L: int, n: int):
        """Per l < L: (W_o^T (gamma * G), W_o^T (gamma * U)), each [H, hd]."""
        G, U = rows.inputs(L, n)
        return [((self.gained(l).T @ G).view(self.H, self.hd), (self.gained(l).T @ U).view(self.H, self.hd)) for l in range(L)]

    def out_norm(self, l: int, z) -> torch.Tensor:
        """||W_o[:, h] @ z_h|| per head, z [N, H, hd] -> [N, H]: the head's output vector before the norm (SPEC B §3)."""
        Wo = self.layers[l].self_attn.o_proj.weight.float().view(-1, self.H, self.hd)
        return torch.einsum("dhk,nhk->nhd", Wo, z.float()).norm(dim=-1)


def head_terms(hrows: HeadRows, rows: UR.Rows, unit: dict, cap: dict) -> list:
    """(w, sign, c_g, c_u, head loads) per term, the coefficients the unit's own at each context (unite_route.unit_terms for heads)."""
    gc, uc = UR.term_gu(cap, unit)
    out = []
    for i, t in enumerate(unit["terms"]):
        cg, cu = UR.coefficients(gc[:, i], uc[:, i])
        out.append((t["w"], t["sign"], cg, cu, hrows.loads(rows, unit["layer"], t["n"])))
    return out


def head_attribution(cap: dict, L: int, terms: list) -> torch.Tensor:
    """[N, L H]: sum over terms of w sign s_attn (c_g go . z_h + c_u uo . z_h); column l H + h is head h of layer l."""
    assert L >= 1, "a unit at layer 0 has no upstream attention below it"
    parts = []
    for l in range(L):
        zs = cap["z"][l] * cap["s_attn"][l][:, None, None]
        parts.append(sum(w * sgn * (cg[:, None] * (zs * lo[l][0][None]).sum(-1) + cu[:, None] * (zs * lo[l][1][None]).sum(-1)) for w, sgn, cg, cu, lo in terms))
    return torch.cat(parts, 1)


def direct_H(hrows: HeadRows, rows: UR.Rows, cap: dict, unit: dict) -> torch.Tensor:
    return head_attribution(cap, unit["layer"], head_terms(hrows, rows, unit, cap))


def head_completeness(hrows: HeadRows, rows: UR.Rows, cap: dict, unit: dict, shift: int = 0) -> np.ndarray:
    """Per context: the largest, over terms, layers l < L and (g, u), of |sum_h s_attn(l') lo(l') . z_h(l') - attn(l) . G| / max over contexts |g| (u the
    same), with l' = l (the check) or l' = (l + shift) mod L (shift 1: the heads of the wrong layer, the failing input)."""
    L = unit["layer"]
    assert L >= 2 or shift == 0, "the wrong-layer input needs two layers below L"
    worst = torch.zeros(cap["emb"].shape[0], device=cap["emb"].device)
    for t in unit["terms"]:
        G, U = rows.inputs(L, t["n"])
        lo = hrows.loads(rows, L, t["n"])
        for j, R in ((0, G), (1, U)):
            pole = (cap["g"] if j == 0 else cap["u"])[L][:, t["n"]].abs().max()
            hp = [(cap["z"][l] * lo[l][j][None]).sum((-1, -2)) * cap["s_attn"][l] for l in range(L)]
            for l in range(L):
                err = (hp[(l + shift) % L] - cap["attn"][l] @ R).abs() / pole
                worst = torch.maximum(worst, err)
    return worst.cpu().numpy()


# ------------------------------------------------------------------------------------------ heads: replacement and readout
@torch.inference_mode()
def background_mean_z(model, positions, n_layers: int, *, tokens, length, bos, batch: int, hw: int, device) -> torch.Tensor:
    """z̄ [n_layers, H, hd] fp32: each head's o_proj input at the marked token of the window ending at each position, averaged over positions."""
    assert n_layers >= 1
    cap = capture(model, np.asarray(positions), n_layers - 1, set(), tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    return torch.stack([cap["z"][l].mean(0) for l in range(n_layers)])


def head_ablation(S: list, zbar: torch.Tensor, L: int, device, identity: bool = False) -> dict:
    """{l: (columns, values)} for `heads_replaced`: head h's columns h hd .. (h+1) hd - 1 take z̄_h. identity: hooks on every layer below L, nothing written."""
    hd = zbar.shape[-1]
    if identity:
        assert not S, "the identity arm has an empty set"
        return {l: (torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, device=device)) for l in range(L)}
    assert S and len(set(S)) == len(S), "a set is empty or names a head twice"
    by = {}
    for l, h in S:
        assert 0 <= l < L, f"layer {l} is not upstream of layer {L}"
        by.setdefault(l, []).append(h)
    out = {}
    for l, hs in by.items():
        idx = torch.tensor(hs, dtype=torch.long, device=device)
        cols = (idx[:, None] * hd + torch.arange(hd, device=device)[None]).reshape(-1)
        out[l] = (cols, zbar[l, idx].float().reshape(-1))
    return out


@contextlib.contextmanager
def heads_replaced(layers, ablation: dict, start: int):
    """A forward pre-hook on layers[l].self_attn.o_proj writes values into its input columns at positions >= start; a forward hook on the same module
    asserts that o_proj received exactly those values (in its dtype) and counts the calls."""
    rec = {"calls": 0}
    handles = []
    for l, (cols, val) in ablation.items():
        def pre(m, args, cols=cols, val=val):
            a = args[0].clone()
            a[:, start:, cols] = val.to(a.dtype)
            return (a, *args[1:])

        def post(m, args, out, cols=cols, val=val):
            got = args[0][:, start:, cols]
            assert bool((got == val.to(got.dtype)).all()), "o_proj did not receive the replaced head outputs"
            rec["calls"] += 1
        handles += [layers[l].self_attn.o_proj.register_forward_pre_hook(pre), layers[l].self_attn.o_proj.register_forward_hook(post)]
    yield rec
    for h in handles:
        h.remove()


def readout_heads(model, toks, unit: dict, ablation: dict, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """reuse_ablate.readout with the head replacement in place: each ablated layer's o_proj must run once per batch with it."""
    layers = list(NH.text_tower(model).layers)
    assert all(l < unit["layer"] for l in ablation), "an ablated layer is not upstream of the unit"
    with heads_replaced(layers, ablation, RA.window_start(bos)) as rec:
        out = RA.readout(model, toks, unit, None, tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    n_batches = -(-len(toks) // batch)
    assert rec["calls"] == len(ablation) * n_batches, f"{rec['calls']} head replacement checks for {len(ablation)} layers x {n_batches} batches"
    return out


def matched(D_other: list, D_own: list, mag, rows) -> tuple:
    """For each (l, j) in D_other, in order: the unit of row l nearest to (l, j) in mag, outside D_own, D_other and those already picked
    (reuse_ablate.random_matched). When row l has none left: the nearest row of `rows` that has one (the lower row on a tie), nearest in mag there.
    Returns (picked, number picked off their row). Deterministic."""
    mag = np.asarray(mag, np.float64)
    taken = set(D_own) | set(D_other)
    out, off = [], 0
    for l, j in D_other:
        for r in sorted(rows, key=lambda r: (abs(r - l), r)):
            gap = np.abs(mag[r] - mag[l, j])
            gap[[k for rr, k in taken if rr == r]] = np.inf
            k = int(np.argmin(gap))
            if np.isfinite(gap[k]):
                break
        assert np.isfinite(gap[k]), "no unit left to match in the allowed layers"
        out.append((r, k))
        taken.add((r, k))
        off += int(r != l)
    return out, off


# ------------------------------------------------------------------------------------------ output: held forward and reader attribution
def hold_dict(unit: dict, abar: torch.Tensor, device, identity: bool = False) -> dict:
    """{L: (idx, values)} for reuse_ablate.replaced on P's own layer: every neuron of the unit held at its background mean. reuse_ablate.ablation_dict
    is called with the bound L + 1, so its l < bound assertion admits layer L itself; that relaxation exists only here. identity: the hook on layer L
    with nothing written."""
    L = unit["layer"]
    if identity:
        return {L: (torch.zeros(0, dtype=torch.long, device=device), torch.zeros(0, device=device))}
    d = RA.ablation_dict([(L, t["n"]) for t in unit["terms"]], "mean", abar, L + 1, device)
    assert set(d) == {L}, sorted(d)
    return d


@torch.inference_mode()
def capture_down(model, toks, unit: dict, hold: dict | None, gu: bool, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """unite_route.capture through the last layer (g, u at L and every layer above it when `gu`), with `hold` written into layer L's down_proj input
    at window positions (None: no hook at all). The held layer's down_proj must run once per batch with the values in place."""
    layers = list(NH.text_tower(model).layers)
    n, L = len(layers), unit["layer"]
    kw = dict(tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    gl = set(range(L, n)) if gu else set()
    if hold is None:
        return UR.capture(model, np.asarray(toks), n - 1, gl, **kw)
    with RA.replaced(layers, hold, RA.window_start(bos)) as rec:
        cap = UR.capture(model, np.asarray(toks), n - 1, gl, **kw)
    n_batches = -(-len(toks) // batch)
    assert rec["calls"] == len(hold) * n_batches, f"{rec['calls']} hold checks for {len(hold)} layers x {n_batches} batches"
    return cap


def down_a(cap: dict, L: int, n_layers: int) -> torch.Tensor:
    """[N, (n_layers - L - 1) W]: every MLP neuron's a above L; column (l - L - 1) W + k is neuron k of layer l."""
    return torch.cat([cap["a"][l] for l in range(L + 1, n_layers)], 1)


def reader_loads(rows: UR.Rows, L: int, n: int, shift: int = 0) -> list:
    """Per layer l above L: (G_l @ d_P, U_l @ d_P), each [W]; d_P = gamma_ff(L) * W_down[:, n] (shift 1: the down column of the wrong layer)."""
    d = rows.column(L, shift)[:, n]
    return [(rows.layers[l].mlp.gate_proj.weight.float() @ d, rows.layers[l].mlp.up_proj.weight.float() @ d) for l in range(L + 1, len(rows.layers))]


def reader_pre(rows: UR.Rows, cap: dict, unit: dict, shift: int = 0) -> list:
    """Per layer above L: (dg, du), each [N, W]: s_L sum over the unit's neurons of raw_a (G_k . d), the direct change of every reader's g and u."""
    L = unit["layer"]
    out = None
    for t in unit["terms"]:
        sa = cap["s"][L] * cap["a"][L][:, t["n"]]
        part = [(sa[:, None] * g[None], sa[:, None] * u[None]) for g, u in reader_loads(rows, L, t["n"], shift)]
        out = part if out is None else [(a[0] + b[0], a[1] + b[1]) for a, b in zip(out, part)]
    return out


def direct_C(rows: UR.Rows, cap: dict, unit: dict) -> torch.Tensor:
    """[N, (n_layers - L - 1) W]: da_k = silu'(g_k) u_k dg_k + silu(g_k) du_k with every reader's own g and u at the context."""
    L = unit["layer"]
    parts = []
    for i, (dg, du) in enumerate(reader_pre(rows, cap, unit)):
        cg, cu = UR.coefficients(cap["g"][L + 1 + i], cap["u"][L + 1 + i])
        parts.append(cg * dg + cu * du)
    return torch.cat(parts, 1)


def reader_completeness(rows: UR.Rows, cap: dict, unit: dict, shift: int = 0) -> np.ndarray:
    """Per context: the largest, over the neurons of layer L+1 and (g, u), of |dg_k - dg'_k| / max over contexts and neurons |dg'| (u the same), where
    dg comes from reader_pre (the instrument) and dg' = gate_proj(L+1) applied in fp32 to s_L raw_a gamma_ff(L) * W_down[:, P], from the module weights
    alone. shift 1: dg' from the down column of layer L+1 (the wrong layer, the failing input)."""
    L = unit["layer"]
    layers = rows.layers
    assert L + 1 < len(layers), "no MLP layer above L"
    dg, du = reader_pre(rows, cap, unit)[0]
    gam = norm_gain_exact(layers[L].post_feedforward_layernorm)
    v = sum((cap["s"][L] * cap["a"][L][:, t["n"]])[:, None] * (gam * layers[L + shift].mlp.down_proj.weight[:, t["n"]].float())[None] for t in unit["terms"])
    ref_g, ref_u = F.linear(v, layers[L + 1].mlp.gate_proj.weight.float()), F.linear(v, layers[L + 1].mlp.up_proj.weight.float())
    v0 = sum((cap["s"][L] * cap["a"][L][:, t["n"]])[:, None] * (gam * layers[L].mlp.down_proj.weight[:, t["n"]].float())[None] for t in unit["terms"])
    sg, su = F.linear(v0, layers[L + 1].mlp.gate_proj.weight.float()).abs().max(), F.linear(v0, layers[L + 1].mlp.up_proj.weight.float()).abs().max()
    return torch.maximum((dg - ref_g).abs().amax(1) / sg, (du - ref_u).abs().amax(1) / su).cpu().numpy()


def logit_direction(model, rows: UR.Rows, L: int, n: int) -> torch.Tensor:
    """[V]: W_U (gamma_final * d_P), the direct logit direction of P's write (the final norm's 1 / rms is a positive scale per context)."""
    gam = norm_gain_exact(NH.text_tower(model).norm)
    return model.get_output_embeddings().weight.float() @ (gam * rows.column(L)[:, n])


def effect(a_id: torch.Tensor, a_held: torch.Tensor, idx) -> torch.Tensor:
    """E(S; x) [N] for column indices idx: sum |a_id - a_held| / sum |a_id| over S."""
    i = torch.as_tensor(idx, device=a_id.device)
    return (a_id[:, i] - a_held[:, i]).abs().sum(1) / a_id[:, i].abs().sum(1)


# ------------------------------------------------------------------------------------------ pure-logic checks on random tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    # matched: equals reuse_ablate.random_matched while every row has room; falls back to the nearest row, lower first on a tie
    mag = rng.uniform(0, 3, (5, 40))
    D_other, D_own = [(1, 3), (1, 4), (2, 10), (3, 0), (4, 1)], [(1, 5), (2, 11), (0, 7)]
    got, off = matched(D_other, D_own, mag, range(5))
    assert got == RA.random_matched(D_other, D_own, mag) and off == 0, (got, off)
    small = np.array([[1.0, 9.0], [1.0, 1.1], [0.9, 4.0]])
    got, off = matched([(1, 0), (1, 1)], [(0, 1)], small, range(3))   # row 1 full: (1,0) -> row 0 (tie with row 2, lower first); then row 0 full -> row 2
    assert got == [(0, 0), (2, 0)] and off == 2, (got, off)
    got, off = matched([(1, 0), (1, 1)], [(0, 1)], small, range(1, 3))   # rows 1 and 2 only: row 0 is never used
    assert got == [(2, 0), (2, 1)] and off == 2, (got, off)
    out["matched"] = "equals reuse_ablate.random_matched with room; full row -> nearest allowed row, lower first on a tie; off-row count"
    # head_ablation layout: head h's columns are h hd .. (h+1) hd - 1, values z̄[l, h]
    zbar = torch.tensor(rng.normal(0, 1, (3, 4, 5)), dtype=torch.float32)
    ab = head_ablation([(1, 2), (1, 0), (2, 3)], zbar, 3, "cpu")
    assert ab[1][0].tolist() == [10, 11, 12, 13, 14, 0, 1, 2, 3, 4] and torch.equal(ab[1][1], torch.cat([zbar[1, 2], zbar[1, 0]])), ab[1]
    assert ab[2][0].tolist() == list(range(15, 20)) and 0 not in ab
    idn = head_ablation([], zbar, 3, "cpu", identity=True)
    assert set(idn) == {0, 1, 2} and all(v[0].numel() == 0 for v in idn.values())
    raised = False
    try:
        head_ablation([(3, 0)], zbar, 3, "cpu")
    except AssertionError:
        raised = True
    assert raised, "a head at the unit's own layer was accepted"
    out["head_ablation"] = "head h -> columns h hd..(h+1) hd - 1 with z̄[l, h]; identity hooks every layer below L; layer L refused"
    # hold_dict: layer L accepted through the relaxed bound, while ablation_dict with the bound L still refuses layer L
    abar = torch.tensor(rng.normal(0, 1, (6, 30)), dtype=torch.float32)
    unit = {"layer": 4, "terms": [{"n": 7, "sign": 1.0, "w": 1.0}, {"n": 2, "sign": 1.0, "w": 0.5}]}
    hdict = hold_dict(unit, abar, "cpu")
    assert set(hdict) == {4} and hdict[4][0].tolist() == [7, 2] and torch.equal(hdict[4][1], abar[4, [7, 2]]), hdict
    raised = False
    try:
        RA.ablation_dict([(4, 7)], "mean", abar, 4, "cpu")
    except AssertionError:
        raised = True
    assert raised, "reuse_ablate.ablation_dict accepted the unit's own layer: the relaxation would not be needed"
    assert set(hold_dict(unit, abar, "cpu", identity=True)) == {4}
    out["hold_dict"] = "layer L accepted only through the bound L + 1; ablation_dict with bound L still refuses it"
    # head_attribution: the heads of a layer sum to the post-norm attention block read through the rows
    N, H, hd, d, L = 6, 4, 3, 8, 2
    z = {l: torch.tensor(rng.normal(0, 1, (N, H, hd)), dtype=torch.float32) for l in range(L)}
    s = {l: torch.tensor(rng.uniform(0.5, 2, N), dtype=torch.float32) for l in range(L)}
    Wo = [torch.tensor(rng.normal(0, 1, (d, H * hd)), dtype=torch.float32) for _ in range(L)]
    gam = [torch.tensor(rng.uniform(0.5, 1.5, d), dtype=torch.float32) for _ in range(L)]
    G, U = (torch.tensor(rng.normal(0, 1, d), dtype=torch.float32) for _ in range(2))
    cg, cu = (torch.tensor(rng.normal(0, 1, N), dtype=torch.float32) for _ in range(2))
    lo = [(((gam[l][:, None] * Wo[l]).T @ G).view(H, hd), ((gam[l][:, None] * Wo[l]).T @ U).view(H, hd)) for l in range(L)]
    A = head_attribution({"z": z, "s_attn": s}, L, [(1.0, -1.0, cg, cu, lo)])
    for l in range(L):
        block = s[l][:, None] * gam[l][None] * (z[l].reshape(N, -1) @ Wo[l].T)
        want = -(cg * (block @ G) + cu * (block @ U))
        assert torch.allclose(A[:, l * H:(l + 1) * H].sum(1), want, rtol=1e-4, atol=1e-4), l
    out["head_attribution"] = "heads of a layer sum to (s gamma * W_o z) read through G and U with the coefficients and the sign"
    # first-order reader change against the exact SwiGLU change for a small step
    g, u = torch.tensor(rng.normal(0, 2, 200)), torch.tensor(rng.normal(0, 2, 200))
    dg, du = torch.tensor(rng.normal(0, 1e-4, 200)), torch.tensor(rng.normal(0, 1e-4, 200))
    c_g, c_u = UR.coefficients(g, u)
    exact = F.silu(g + dg) * (u + du) - F.silu(g) * u
    assert torch.allclose(c_g * dg + c_u * du, exact, rtol=1e-3, atol=1e-7), "the first-order reader change is wrong"
    out["reader_first_order"] = "silu'(g) u dg + silu(g) du equals the exact change to first order"
    a_id = torch.tensor(rng.normal(0, 1, (4, 30)))
    assert bool((effect(a_id, a_id, [1, 5, 9]) == 0).all()) and torch.allclose(effect(a_id, torch.zeros_like(a_id), [1, 5, 9]), torch.ones(4, dtype=a_id.dtype))
    out["effect"] = "identity gives exactly 0, all readers zeroed gives 1"
    return out
