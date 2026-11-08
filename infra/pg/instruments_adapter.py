"""INSTRUMENTS E1b model adapter (studies/instruments/SPEC_instruments.md §3.6, §10). unite_route, reuse_ablate, instruments_linear and
neurons_harvest are imported unchanged.

Three facts per architecture, read from the module tree (transformers 5.20.2 decoder layers):
  olmo3   the MLP reads h_L itself (no pre-MLP norm); attention adds post_attention_layernorm(attn); the MLP adds post_feedforward_layernorm(mlp).
  olmo2   the same three facts as olmo3 (INSTRUMENTS E6b; modeling_olmo2 Olmo2DecoderLayer.forward: h + post_attention_layernorm(self_attn(h)), then
          + post_feedforward_layernorm(mlp(.)); q_norm and k_norm on the q and k projections); asserted at construction by olmo2_layout: each decoder
          layer's children are exactly self_attn, mlp and the two output norms (no input_layernorm, no pre_feedforward_layernorm), self_attn holds
          q_norm and k_norm, and the config declares no layer_types and no sliding_window (every layer full causal attention).
  gemma3  the MLP reads pre_feedforward_layernorm(h_L); attention adds post_attention_layernorm(attn); the MLP adds post_feedforward_layernorm(mlp).
  qwen3   the MLP reads post_attention_layernorm(h_L) (Qwen 3's name for its pre-MLP norm); attention adds self_attn's output; the MLP adds its own
          output, unnormed (post-MLP gain 1, s_l = 1).
  h_L = embedding + sum_{l<L} (attention_l + mlp_l) + attention_L, each term as it is added to the residual stream.
With a pre-MLP norm of gain gamma_pre and r_L(x) = 1 / rms(h_L(x)) (with that norm's eps), g = (G ⊙ gamma_pre) . h_L r_L, so for neuron k of layer l < L
    A_k(x) = sigma a_k(x) s_l(x) [c_g(x) r_L(x) G'.d_k + c_u(x) r_L(x) U'.d_k],   G' = G ⊙ gamma_pre,  d_k = gamma_post_l ⊙ W_down_l[:, k],
s_l = 1 / rms of the post-MLP norm's input. Without a post-MLP norm gamma_post = 1 and s_l = 1; without a pre-MLP norm gamma_pre = 1 and r_L = 1,
which is unite_route's Olmo formula. So: Rows subclasses unite_route.Rows (column reads the adapter's post gain, inputs fold in the pre gain;
loads and foreign are inherited); unit_terms folds r_L into c_g and c_u and unite_route.attribution is called unchanged; completeness is
unite_route.source_terms and neuron_sum times r_L. Coefficients: c_g = act'(g) u, c_u = act(g), act measured on the module
(neurons_harvest.act_name): silu -> unite_route.coefficients; gelu_tanh (Gemma 3) -> its derivative here (NEW; checked against autograd in selfcheck).
The adapter's code-level known number: on an Olmo 3, `olmo_equality` asserts capture, Rows, direct_A, c, completeness, ā and the background
capture equal to unite_route.capture, unite_route.Rows, reuse_ablate.direct_A, instruments_linear.coefficients, unite_route.completeness,
reuse_ablate.background_mean_a and instruments_values.capture_background, exactly.
Asserted at construction: every hooked module exists and has no bias; the layer count equals the config's and the expected count (34 for the Gemma 3
text tower, 36 for Qwen3-4B-Base, from matrix_models.KNOWN); one activation function on every layer.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from infra.pg import instruments_linear as IL
from infra.pg import neurons_harvest as NH
from infra.pg import unite_route as UR
from infra.pg.neuron_rw_core import inv_rms, norm_eps, norm_gain_exact

ARCH = {"olmo3": {"attn_out": "post_attention_layernorm", "post_mlp_norm": "post_feedforward_layernorm", "pre_mlp_norm": None},
        "gemma3": {"attn_out": "post_attention_layernorm", "post_mlp_norm": "post_feedforward_layernorm", "pre_mlp_norm": "pre_feedforward_layernorm"},
        "qwen3": {"attn_out": "self_attn", "post_mlp_norm": None, "pre_mlp_norm": "post_attention_layernorm"},
        "olmo2": {"attn_out": "post_attention_layernorm", "post_mlp_norm": "post_feedforward_layernorm", "pre_mlp_norm": None}}
OLMO2_CHILDREN = {"self_attn", "mlp", "post_attention_layernorm", "post_feedforward_layernorm"}
OLMO2_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"}
MLP = ("gate_proj", "up_proj", "down_proj", "act_fn")
GELU_K = math.sqrt(2.0 / math.pi)


def no_bias(m, what: str):
    assert getattr(m, "bias", None) is None, f"{what} ({type(m).__name__}) has a bias"


def olmo2_layout(cfg, layers):
    """OLMo-2's post-norm layout read off the module tree and the config (modeling_olmo2, transformers 5.20.2): the attention and the MLP read the
    raw residual (no pre-norm module exists), the two output norms exist, QK-norm exists, and no layer is a sliding-window layer."""
    assert getattr(cfg, "layer_types", None) is None and getattr(cfg, "sliding_window", None) is None, \
        ("olmo2: the config declares layer types or a sliding window", getattr(cfg, "layer_types", None), getattr(cfg, "sliding_window", None))
    for i, layer in enumerate(layers):
        kids = {n for n, _ in layer.named_children()}
        assert kids == OLMO2_CHILDREN, (i, sorted(kids), "olmo2: a decoder layer's modules are not self_attn, mlp and the two output norms")
        att = {n for n, _ in layer.self_attn.named_children()}
        assert att == OLMO2_ATTN, (i, sorted(att), "olmo2: the attention's modules are not q/k/v/o and q_norm, k_norm")


class Adapter:
    def __init__(self, model, n_layers: int | None = None):
        self.arch = model.config.model_type
        assert self.arch in ARCH, f"no adapter for model_type {self.arch}"
        self.spec = ARCH[self.arch]
        self.tower = NH.text_tower(model)
        self.layers = list(self.tower.layers)
        cfg = model.config.get_text_config()
        assert len(self.layers) == cfg.num_hidden_layers, (len(self.layers), cfg.num_hidden_layers)
        assert n_layers is None or len(self.layers) == n_layers, f"{len(self.layers)} decoder layers, expected {n_layers}"
        if self.arch == "olmo2":
            olmo2_layout(cfg, self.layers)
        self.hidden = int(cfg.hidden_size)
        self.width = int(self.layers[0].mlp.down_proj.weight.shape[1])
        for i, layer in enumerate(self.layers):
            for name in MLP:
                assert hasattr(layer.mlp, name), f"layer {i}: no mlp.{name}"
            for name in MLP[:3]:
                no_bias(getattr(layer.mlp, name), f"layer {i} mlp.{name}")
            assert layer.mlp.down_proj.weight.shape == (self.hidden, self.width), (i, tuple(layer.mlp.down_proj.weight.shape))
            for key in ("attn_out", "post_mlp_norm", "pre_mlp_norm"):
                name = self.spec[key]
                if name is not None:
                    assert hasattr(layer, name), f"layer {i}: no {name} ({self.arch})"
                    no_bias(getattr(layer, name), f"layer {i} {name}")
            if self.arch == "qwen3":
                no_bias(layer.self_attn.o_proj, f"layer {i} self_attn.o_proj")
        acts = {NH.act_name(l.mlp.act_fn) for l in self.layers}
        assert len(acts) == 1, f"layers use different activations: {acts}"
        self.act = acts.pop()
        self._post, self._pre = {}, {}

    def attn_out(self, l: int):
        return getattr(self.layers[l], self.spec["attn_out"])

    def post_norm(self, l: int):
        n = self.spec["post_mlp_norm"]
        return None if n is None else getattr(self.layers[l], n)

    def pre_norm(self, l: int):
        n = self.spec["pre_mlp_norm"]
        return None if n is None else getattr(self.layers[l], n)

    def post_gain(self, l: int) -> torch.Tensor:
        """gamma_post_l (fp32): the post-MLP norm's exact gain, or ones without one."""
        if l not in self._post:
            n = self.post_norm(l)
            w = self.layers[l].mlp.down_proj.weight
            self._post[l] = norm_gain_exact(n) if n is not None else torch.ones(self.hidden, device=w.device, dtype=torch.float32)
        return self._post[l]

    def pre_gain(self, L: int) -> torch.Tensor:
        """gamma_pre_L (fp32): the pre-MLP norm's exact gain, or ones without one."""
        if L not in self._pre:
            n = self.pre_norm(L)
            w = self.layers[L].mlp.gate_proj.weight
            self._pre[L] = norm_gain_exact(n) if n is not None else torch.ones(self.hidden, device=w.device, dtype=torch.float32)
        return self._pre[L]

    def describe(self) -> dict:
        return {"arch": self.arch, **self.spec, "n_layers": len(self.layers), "hidden": self.hidden, "width": self.width, "act": self.act}


# ------------------------------------------------------------------------------------------ rows
class Rows(UR.Rows):
    """unite_route.Rows with d_k = gamma_post_l ⊙ W_down_l[:, k] (gain 1 without a post-MLP norm) and the input rows G ⊙ gamma_pre_L, U ⊙ gamma_pre_L."""

    def __init__(self, model, ad: Adapter):
        super().__init__(model)
        assert len(self.layers) == len(ad.layers) and all(a is b for a, b in zip(self.layers, ad.layers)), "Rows and the adapter see different layers"
        self.ad = ad

    def column(self, l: int, shift: int = 0):
        src = (l + shift) % len(self.layers)
        if shift:
            return self.ad.post_gain(l)[:, None] * self.layers[src].mlp.down_proj.weight.float()
        if self.d[l] is None:
            self.d[l] = self.ad.post_gain(l)[:, None] * self.layers[l].mlp.down_proj.weight.float()
        return self.d[l]

    def inputs(self, L: int, n: int):
        G, U = super().inputs(L, n)
        g = self.ad.pre_gain(L)
        return G * g, U * g


# ------------------------------------------------------------------------------------------ capture
@torch.inference_mode()
def capture(ad: Adapter, model, toks, upto: int, gu_layers, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """unite_route.capture's record through the adapter: at each position in `toks`, for layers 0..upto, a [N, W] (fp32 recomputed), s [N]
    (1 / rms of the post-MLP norm's input; ones without one), attention and MLP terms as added to the residual [N, d]; for `gu_layers` g, u [N, W]
    and r [N] (1 / rms of the pre-MLP norm's input; ones without one); embedding [N, d]. Rows keep the order of `toks`."""
    tower = ad.tower
    layers = ad.layers[:upto + 1]
    acts = [l.mlp.act_fn for l in layers]
    tap = NH.MLPTap(layers)
    toks = np.asarray(toks)
    order = np.argsort(toks % hw, kind="stable")
    out = {"a": {}, "g": {}, "u": {}, "s": {}, "r": {}, "attn": {}, "mlp": {}, "emb": []}
    for b0 in range(0, len(toks), batch):
        idx = order[b0:b0 + batch]
        x, pos = UR.pack(tokens, toks[idx], length, bos, hw, device)
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
            def on_attn(m, a, o, lyr=lyr):
                out["attn"].setdefault(lyr, []).append(at(o[0] if isinstance(o, tuple) else o))
            handles.append(ad.attn_out(lyr).register_forward_hook(on_attn))
            post = ad.post_norm(lyr)
            if post is not None:
                def on_ff(m, a, o, lyr=lyr, eps=norm_eps(post)):
                    out["s"].setdefault(lyr, []).append(inv_rms(at(a[0]), eps))
                    out["mlp"].setdefault(lyr, []).append(at(o))
                handles.append(post.register_forward_hook(on_ff))
            else:
                def on_mlp(m, a, o, lyr=lyr):
                    v = at(o)
                    out["s"].setdefault(lyr, []).append(torch.ones(v.shape[0], device=v.device))
                    out["mlp"].setdefault(lyr, []).append(v)
                handles.append(layer.mlp.register_forward_hook(on_mlp))
            pre = ad.pre_norm(lyr)
            if lyr in gu_layers and pre is not None:
                def on_pre(m, a, o, lyr=lyr, eps=norm_eps(pre)):
                    out["r"].setdefault(lyr, []).append(inv_rms(at(a[0]), eps))
                handles.append(pre.register_forward_hook(on_pre))
        with UR.upto_layer(tower, upto):
            tap.run(fn, lambda: NH.forward(model, x, False))
        for h in handles:
            h.remove()
    inv = torch.as_tensor(np.argsort(order), device=device)
    res = {k: {l: torch.cat(v)[inv] for l, v in out[k].items()} for k in ("a", "g", "u", "s", "r", "attn", "mlp")}
    res["emb"] = torch.cat(out["emb"])[inv]
    for l in gu_layers:
        if l not in res["r"]:
            assert ad.pre_norm(l) is None, f"layer {l}: the pre-MLP norm did not fire"
            res["r"][l] = torch.ones(len(toks), device=res["emb"].device)
    n = len(toks)
    assert all(v.shape[0] == n for k in ("a", "s", "attn", "mlp") for v in res[k].values()) and len(res["a"]) == upto + 1, "a hook fired the wrong number of times"
    return res


def cap_rows(cap: dict, idx) -> dict:
    """unite_route.cap_rows with the r record."""
    out = UR.cap_rows(cap, idx)
    ii = torch.as_tensor(np.asarray(idx), device=cap["emb"].device)
    out["r"] = {l: v[ii] for l, v in cap["r"].items()}
    return out


def capture_background(ad: Adapter, model, positions, n_layers: int, **kw) -> torch.Tensor:
    """X [P, n_layers, W] fp32: instruments_values.capture_background through the adapter."""
    assert n_layers >= 1
    cap = capture(ad, model, np.asarray(positions), n_layers - 1, set(), **kw)
    return torch.stack([cap["a"][l] for l in range(n_layers)], 1)


def background_mean_a(ad: Adapter, model, positions, n_layers: int, **kw) -> torch.Tensor:
    """ā [n_layers, W] fp32: reuse_ablate.background_mean_a through the adapter (one capture, averaged over positions)."""
    cap = capture(ad, model, np.asarray(positions), n_layers - 1, set(), **kw)
    return torch.stack([cap["a"][l].mean(0) for l in range(n_layers)])


# ------------------------------------------------------------------------------------------ coefficients and attribution
def gelu_tanh_grad(g: torch.Tensor) -> torch.Tensor:
    """d/dg of 0.5 g (1 + tanh(k (g + 0.044715 g^3))), k = sqrt(2 / pi)."""
    t = torch.tanh(GELU_K * (g + 0.044715 * g ** 3))
    return 0.5 * (1 + t) + 0.5 * g * (1 - t * t) * GELU_K * (1 + 3 * 0.044715 * g * g)


def coefficients(act: str, g: torch.Tensor, u: torch.Tensor):
    """c_g = act'(g) u, c_u = act(g)."""
    if act == "silu":
        return UR.coefficients(g, u)
    assert act == "gelu_tanh", act
    return gelu_tanh_grad(g) * u, torch.nn.functional.gelu(g, approximate="tanh")


def unit_terms(ad: Adapter, rows: Rows, unit: dict, cap_g, cap_u, r, shift: int = 0) -> list:
    """(w, sign, c_g r_L, c_u r_L, loads) per term, for unite_route.attribution. shift != 0: the loads from the wrong layer's columns."""
    L, out = unit["layer"], []
    for i, t in enumerate(unit["terms"]):
        cg, cu = coefficients(ad.act, cap_g[:, i], cap_u[:, i])
        out.append((t["w"], t["sign"], cg * r, cu * r, rows.loads(L, t["n"], shift)))
    return out


def direct_A(ad: Adapter, rows: Rows, cap: dict, unit: dict, W: int) -> torch.Tensor:
    """[N, L W]: reuse_ablate.direct_A through the adapter."""
    L = unit["layer"]
    gc, uc = UR.term_gu(cap, unit)
    return UR.attribution(cap, L, unit_terms(ad, rows, unit, gc, uc, cap["r"][L]), W)


@torch.inference_mode()
def coefficients_c(ad: Adapter, rows: Rows, cap: dict, unit: dict, W: int, A: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """instruments_linear.coefficients through the adapter: c = attribution with a = 1; A = a ⊙ c asserted to C_TOL. Failing input: c from the wrong
    layer's columns (loads with shift 1; the next layer's s_l of the Olmo check cannot fail on Qwen 3, whose s_l is 1)."""
    L = unit["layer"]
    gc, uc = UR.term_gu(cap, unit)
    r = cap["r"][L]
    c = UR.attribution(IL._ones(cap, L), L, unit_terms(ad, rows, unit, gc, uc, r), W)
    a = torch.cat([cap["a"][l] for l in range(L)], 1)
    err = IL.c_error(A, a, c)
    err_fail = IL.c_error(A, a, UR.attribution(IL._ones(cap, L), L, unit_terms(ad, rows, unit, gc, uc, r, shift=1), W))
    rec = {"value": err, "failing_input": err_fail, "tol": IL.C_TOL}
    assert err <= IL.C_TOL, (unit["unit"], err, "A is not a ⊙ c")
    return c, rec


def completeness(ad: Adapter, rows: Rows, cap: dict, L: int, n: int, shift: int = 0) -> dict:
    """unite_route.completeness through the adapter: r_L (G'.(embedding + attention + MLP terms)) against g, the same for u, over the pole's max."""
    st = UR.source_terms(rows, cap, L, n)
    g_mlp, u_mlp = UR.neuron_sum(rows, cap, L, n, shift)
    r = cap["r"][L]
    g_hat = r * (g_mlp + st["attention"][0] + st["embedding"][0])
    u_hat = r * (u_mlp + st["attention"][1] + st["embedding"][1])
    g, u = cap["g"][L][:, n], cap["u"][L][:, n]
    return {"err_g": ((g_hat - g).abs() / g.abs().max()).cpu().numpy(), "err_u": ((u_hat - u).abs() / u.abs().max()).cpu().numpy()}


# ------------------------------------------------------------------------------------------ the code-level known number (an Olmo 3)
def olmo_equality(model, toks, units: list, positions, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """On an Olmo 3 (or an OLMo 2, the same layout): the adapter path equals the REUSED path exactly (torch.equal) on every quantity E1b uses.
    Returns what was compared."""
    from infra.pg import instruments_values as IV
    from infra.pg import reuse_ablate as RA
    ad = Adapter(model)
    assert ad.arch in ("olmo3", "olmo2"), ad.arch
    rows, ref = Rows(model, ad), UR.Rows(model)
    kw = dict(tokens=tokens, length=length, bos=bos, batch=batch, hw=hw, device=device)
    W = ad.width
    out = {"units": [u["unit"] for u in units]}
    for l in range(len(ad.layers)):
        assert torch.equal(rows.column(l), ref.column(l)) and torch.equal(rows.column(l, 1), ref.column(l, 1)), f"column {l}"
    for unit in units:
        L = unit["layer"]
        a, b = capture(ad, model, toks, L, {L}, **kw), UR.capture(model, toks, L, {L}, **kw)
        for k in ("a", "g", "u", "s", "attn", "mlp"):
            assert set(a[k]) == set(b[k]) and all(torch.equal(a[k][l], b[k][l]) for l in b[k]), (unit["unit"], k)
        assert torch.equal(a["emb"], b["emb"]) and torch.equal(a["r"][L], torch.ones_like(a["r"][L])), unit["unit"]
        for t in unit["terms"]:
            for x, y in zip(rows.loads(L, t["n"]), ref.loads(L, t["n"])):
                assert torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]), (unit["unit"], "loads")
        A, A_ref = direct_A(ad, rows, a, unit, W), RA.direct_A(ref, b, unit, W)
        assert torch.equal(A, A_ref), (unit["unit"], "direct_A")
        c, rec = coefficients_c(ad, rows, a, unit, W, A)
        c_ref, rec_ref = IL.coefficients(ref, b, unit, W, A_ref)
        assert torch.equal(c, c_ref) and rec["value"] == rec_ref["value"], (unit["unit"], "c")
        if len(unit["terms"]) == 1:
            n = unit["terms"][0]["n"]
            for shift in (0, 1):
                x, y = completeness(ad, rows, a, L, n, shift), UR.completeness(ref, b, L, n, shift)
                assert np.array_equal(x["err_g"], y["err_g"]) and np.array_equal(x["err_u"], y["err_u"]), (unit["unit"], "completeness", shift)
    n_up = len(ad.layers) - 1
    ab, ab_ref = background_mean_a(ad, model, positions, n_up, **kw), RA.background_mean_a(model, positions, n_up, **kw)
    X, X_ref = capture_background(ad, model, positions, n_up, **kw), IV.capture_background(model, positions, n_up, **kw)
    assert torch.equal(ab, ab_ref) and torch.equal(X, X_ref), "ā or the background capture"
    out["compared"] = ["column (and shift 1)", "capture a, g, u, s, attn, mlp, emb (r = 1)", "loads", "direct_A", "c and its check", "completeness (and shift 1)",
                       "background_mean_a", "capture_background"]
    return out


# ------------------------------------------------------------------------------------------ pure-logic checks on tiny tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    g = torch.linspace(-6, 6, 241, dtype=torch.float64, requires_grad=True)
    y = torch.nn.functional.gelu(g, approximate="tanh")
    (dg,) = torch.autograd.grad(y.sum(), g)
    e = float((gelu_tanh_grad(g.detach()) - dg).abs().max())
    assert e < 1e-12, e
    gs = torch.tensor(rng.normal(0, 2, 50), dtype=torch.float32)
    us = torch.tensor(rng.normal(0, 1, 50), dtype=torch.float32)
    cs = coefficients("silu", gs, us)
    rs = UR.coefficients(gs, us)
    assert torch.equal(cs[0], rs[0]) and torch.equal(cs[1], rs[1])
    cg, cu = coefficients("gelu_tanh", gs, us)
    assert torch.allclose(cu, torch.nn.functional.gelu(gs, approximate="tanh")) and not torch.allclose(cg, rs[0])
    out["coefficients"] = f"gelu_tanh' = autograd to {e:.1e}; silu is unite_route.coefficients exactly"
    N, L, W = 5, 3, 7
    t = lambda *s: torch.tensor(rng.normal(0, 1, s), dtype=torch.float32)
    cap = {"a": {l: t(N, W) for l in range(L)}, "s": {l: torch.tensor(rng.uniform(0.5, 2, N), dtype=torch.float32) for l in range(L)}, "emb": t(N, 4)}
    r = torch.tensor(rng.uniform(0.5, 2, N), dtype=torch.float32)
    loads = [(t(W), t(W)) for _ in range(L)]
    c1, c2 = t(N), t(N)
    A_r = UR.attribution(cap, L, [(1.0, -1.0, c1 * r, c2 * r, loads)], W)
    A_1 = UR.attribution(cap, L, [(1.0, -1.0, c1, c2, loads)], W)
    er = float((A_r - r[:, None] * A_1).abs().max() / A_1.abs().max())
    assert er < 1e-6, er
    out["pre_norm"] = f"r_L folded into the coefficients scales each context's attribution row by r_L (to {er:.1e})"
    assert set(ARCH) == {"olmo3", "gemma3", "qwen3", "olmo2"} and ARCH["qwen3"]["post_mlp_norm"] is None and ARCH["olmo3"]["pre_mlp_norm"] is None
    assert ARCH["olmo2"] == ARCH["olmo3"], "olmo2's hook points are olmo3's"
    return out
