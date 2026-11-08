"""NEURONRW core (studies/neuronrw/SCHEMA.md v5, SPEC §9.1 item 1): push poles' W_down columns into the residual stream, read what the
model writes, and measure the direct path of one neuron's write.

The object (as in NEURONOFF): a = act(g) * u, the input of `mlp.down_proj`, one coordinate per neuron; a pole is
(layer, neuron, sign). u_i = W_down[:, i] / ||W_down[:, i]||_2 (fp32, from the model's bf16 weight); the pole's
written direction is w = sign * u_i.

`ResidualSteer` is a forward hook on the decoder layer module itself, so it edits the layer's output: the residual
stream after the residual add, which follows `post_feedforward_layernorm` in Gemma 3 and Olmo 3 (Qwen 3 has no
post-MLP norm; its residual add follows the MLP directly). HF modeling code (transformers 5.9.0), the decoder layer
forward ends:
    gemma3:  hidden_states = self.mlp(hidden_states)
             hidden_states = self.post_feedforward_layernorm(hidden_states)
             hidden_states = residual + hidden_states
    olmo3:   hidden_states = self.mlp(hidden_states)
             hidden_states = self.post_feedforward_layernorm(hidden_states)
             hidden_states = residual + hidden_states
    qwen3:   hidden_states = self.mlp(hidden_states)
             hidden_states = residual + hidden_states
`placement(model, layer)` locates these module names in the loaded model and fails loud when they are missing;
`placement_check(model, layer, ids)` also runs one forward and reads the order in which the layer's children ran:
the post-MLP norm (Gemma 3, Olmo 3) or the MLP (Qwen 3) must be the last child before the hook point.

The hook (SCHEMA §2) holds one row spec per batch row: (layer or None, w or None, alpha_abs), alpha_abs = alpha * RMS_l
(SCHEMA conventions). On every call at a hooked layer it copies the layer output to fp32, adds alpha_abs_b * w_b to
row b (0 for a row steered at another layer or at none) at every sequence position >= `start` (positions counted
across the forwards of one `with` block, so cached generation step k is position P + k), and casts back to the model
dtype. The layer output may be a bare tensor or a tuple led by one; the hook returns the same type and raises on
anything else (A11). It records, per steered row, the realized change bf16(h + v) - h at P - 1 and decode steps 1-3
(cos against the signed vector alpha_abs * w, A21; norm ratio against |alpha_abs|); `realized_cos` is their minimum (A2). keyed="per_forward" counts positions
within the current forward instead (the G-gen failing input: it never steers a decode step). It runs its add on every
call, including alpha 0, so alpha = 0 exercises the whole add-and-cast path and must
reproduce the unhooked logits exactly. Directions: `pole_direction` (sign * u_i), `random_direction` (a seeded unit
Gaussian), `orthogonal_direction` (a seeded unit direction orthogonal to the effective unembeddings of given tokens).

Also here: per-layer residual RMS on a text sample (b), the vocabulary projection of a pole (c), the direct-effect
readout of one neuron's write at given positions for the self-repair ratio (d; both norm scales frozen as primary,
post-MLP and final norm recomputed as secondaries, A6), batched steered sampling with each row's mean NLL under the
unhooked model (e), the one JSON-lines reader (f), which splits on "\\n" only, the G-place placements, the firing
readout, the literal-overlap rule (A1), the coherence band with its seeded half split (A7), and the statistics:
one-way, paired-over-prompts (Arm 1, A3) and two-way (Arm 2, A10) bootstraps, the A4 rule intervals and the room
checks R-null, R-pass, R-equiv.

Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib
import json
import math
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from infra.pg import neuron_off_core as C
from infra.pg.neurons_harvest import text_tower

SIGNS = (1, -1)
POST_MLP_NORM = {"gemma3": "post_feedforward_layernorm", "olmo3": "post_feedforward_layernorm", "qwen3": None}
LAST_CHILD = {"gemma3": "post_feedforward_layernorm", "olmo3": "post_feedforward_layernorm", "qwen3": "mlp"}
SCORE_POSITIONS = 2048        # rows x positions per fp32 log-softmax when scoring samples
VOCAB_CHUNK = 32768           # W_U rows per fp32 matmul


# ------------------------------------------------------------------------------------------ (f) JSON lines
def jsonl(text: str) -> list:
    """JSON lines split on "\\n" only: str.splitlines() also splits on U+2028/U+2029, which JSON allows raw inside a
    string (the parent verdict file has them)."""
    return [json.loads(x) for x in text.split("\n") if x.strip()]


def read_jsonl(path) -> list:
    return jsonl(pathlib.Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ placement
def model_type(model) -> str:
    mt = model.config.model_type
    assert mt in POST_MLP_NORM, f"model_type {mt!r} not in {sorted(POST_MLP_NORM)}"
    return mt


def d_model(model) -> int:
    return text_tower(model).norm.weight.shape[0]


def module_name(model, module) -> str:
    names = [n for n, m in model.named_modules() if m is module]
    assert len(names) == 1, f"module found under {len(names)} names"
    return names[0]


def placement(model, layer: int) -> dict:
    """The hook module (the decoder layer) and its post-MLP norm, by qualified name in the loaded model."""
    mt = model_type(model)
    layers = text_tower(model).layers
    assert 0 <= layer < len(layers), f"layer {layer} out of range 0..{len(layers) - 1}"
    children = dict(layers[layer].named_children())
    assert "mlp" in children and hasattr(children["mlp"], "down_proj"), f"layer {layer} has no mlp.down_proj"
    norm = POST_MLP_NORM[mt]
    if norm is None:
        assert "post_feedforward_layernorm" not in children, f"{mt} layer {layer} has a post_feedforward_layernorm"
    else:
        assert norm in children, f"{mt} layer {layer}: no child {norm!r} (children {sorted(children)})"
    return {"model_type": mt, "hook_module": module_name(model, layers[layer]),
            "post_mlp_norm": None if norm is None else module_name(model, children[norm]),
            "expected_last_child": LAST_CHILD[mt]}


@torch.no_grad()
def placement_check(model, layer: int, ids: torch.Tensor) -> dict:
    """placement() plus the order in which the layer's direct children ran during one forward of `ids`;
    pass when the last child before the layer's own output hook is the expected one."""
    p = placement(model, layer)
    lyr = text_tower(model).layers[layer]
    order, handles = [], []
    for name, child in lyr.named_children():
        handles.append(child.register_forward_hook(lambda m, a, o, name=name: order.append(name)))
    handles.append(lyr.register_forward_hook(lambda m, a, o: order.append("<layer output>")))
    ids = ids.to(model.device)
    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    for h in handles:
        h.remove()
    assert order.count("<layer output>") == 1, f"layer {layer} output hook fired {order.count('<layer output>')} times"
    last = order[order.index("<layer output>") - 1]
    return {**p, "layer": layer, "call_order": order, "last_child": last, "pass": last == p["expected_last_child"]}


# ------------------------------------------------------------------------------------------ directions
@torch.no_grad()
def w_down_column(model, layer: int, neuron: int) -> torch.Tensor:
    w = text_tower(model).layers[layer].mlp.down_proj.weight
    assert 0 <= neuron < w.shape[1], f"neuron {neuron} out of range 0..{w.shape[1] - 1}"
    return w[:, neuron].float()


def pole_direction(model, layer: int, neuron: int, sign: int) -> torch.Tensor:
    """w = sign * W_down[:, i] / ||W_down[:, i]||_2, fp32, on the model's device."""
    assert sign in SIGNS, f"sign must be +1 or -1, got {sign!r}"
    col = w_down_column(model, layer, neuron)
    n = col.norm()
    assert n > 0, f"L{layer}N{neuron}: zero W_down column"
    return col / n * sign


def random_direction(model, seed: int) -> torch.Tensor:
    """SPEC §6 G-floor-dir: numpy default_rng(seed).standard_normal(d), normalized, fp32."""
    v = torch.from_numpy(np.random.default_rng(seed).standard_normal(d_model(model))).float()
    return (v / v.norm()).to(model.device)


@torch.no_grad()
def orthogonal_direction(model, token_ids: list, seed: int) -> torch.Tensor:
    """A seeded unit direction orthogonal to gain_final * W_U[v] for every v in token_ids (the tokens' effective
    unembeddings: its direct logit effect on those tokens is 0)."""
    E = final_norm_gain(model)[None, :] * unembedding(model)[torch.tensor(token_ids, device=model.device)].float()
    q, _ = torch.linalg.qr(E.T.double())
    v = random_direction(model, seed).double()
    v = v - q @ (q.T @ v)
    v = (v / v.norm()).float()
    assert float((E @ v).abs().max() / E.norm(dim=1).max()) < 1e-5, "orthogonal direction is not orthogonal"
    return v


# ------------------------------------------------------------------------------------------ (a) the hook
N_REALIZED_STEPS = 4            # prefill position P - 1 and decode steps 1-3 (SPEC §2.2, A2)


def hidden_of(out):
    """The hidden state of a decoder layer's output: a bare tensor, or element 0 of a tuple (A11); raises otherwise."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"decoder layer output of type {type(out).__name__} is neither a tensor nor a tuple led by one")


def with_hidden(out, h):
    """`out` with its hidden state replaced by h, the same type."""
    return h if isinstance(out, torch.Tensor) else (h,) + tuple(out[1:])


class ResidualSteer:
    """Forward hooks on the decoder layers named by the row specs. rows: one (layer or None, w or None, alpha_abs)
    per batch row; a row with layer None must have alpha_abs 0 and receives no add.
    keyed "absolute" (the instrument): positions are absolute, a running offset over the forwards of one `with` block
    (prefill, then one position per decode step); the add lands at every position >= start.
    keyed "per_forward" (G-gen's registered failing input): the index within the current forward is compared with
    start, so a decode step (one position, index 0) is never steered.
    `calls` counts forwards (calls of the first hooked layer); `n_edited` counts (row, position) pairs steered;
    `realized[b]` = [(step, cos, ratio)] of the realized bf16(h + v) - h against the signed intended vector v =
    alpha_abs * w, i.e. cos(dh, sign(alpha_abs) * w) (A21), at absolute positions start + step,
    step 0..3 (step 0 = prefill position P - 1, steps 1-3 = decode steps 1-3), ratio = ||dh|| / |alpha_abs|;
    `realized_cos[b]` = the smallest of those cosines (None for an unsteered or alpha = 0 row)."""

    fire = None                     # neuron_off_core.next_token reads ablator.fire; a steer has no firing mask

    def __init__(self, model, rows: list, start: int, keyed: str = "absolute"):
        assert rows, "no rows"
        assert start >= 0, f"start {start} < 0"
        assert keyed in ("absolute", "per_forward"), keyed
        d, dev = d_model(model), model.device
        layers = sorted({r[0] for r in rows if r[0] is not None})
        self.placement = {l: placement(model, l) for l in layers}
        self.delta = {}
        for l in layers:
            rows_l = []
            for layer, w, a in rows:
                assert math.isfinite(a), f"alpha_abs {a} is not finite"
                if layer is None:
                    assert w is None and a == 0.0, "a row with no layer must have no direction and alpha 0"
                if layer == l:
                    assert w.shape == (d,) and w.dtype == torch.float32, f"direction must be fp32 [{d}]"
                    rows_l.append(a * w.to(dev))
                else:
                    rows_l.append(torch.zeros(d, dtype=torch.float32, device=dev))
            self.delta[l] = torch.stack(rows_l)                                    # [B, d]
        self.dirs = [None if w is None else w.to(dev) for _, w, _ in rows]
        self.alphas = [float(a) for _, _, a in rows]
        self.realized = [[] for _ in rows]
        self.n_rows, self.keyed = len(rows), keyed
        self.rows_at = {l: sum(r[0] == l for r in rows) for l in layers}
        self.layer_of = [r[0] for r in rows]
        self.modules = {l: text_tower(model).layers[l] for l in layers}
        self.start, self.handles = start, []
        self.offset = {l: 0 for l in layers}
        self.n_calls = {l: 0 for l in layers}
        self.n_edited = 0

    @property
    def calls(self) -> int:
        return self.n_calls[min(self.n_calls)] if self.n_calls else 0

    @property
    def realized_cos(self) -> list:
        return [min(c for _, c, _ in r) if r else None for r in self.realized]

    def __enter__(self):
        assert not self.handles, "hooks already registered"
        self.offset = {l: 0 for l in self.offset}
        self.realized = [[] for _ in range(self.n_rows)]
        self.n_edited = 0
        self.handles = [m.register_forward_hook(lambda mod, a, o, l=l: self._hook(l, o)) for l, m in self.modules.items()]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _hook(self, l, out):
        h = hidden_of(out)
        B, T = h.shape[:2]
        assert B == self.n_rows, f"batch {B} rows, steer built for {self.n_rows}"
        hf = h.to(torch.float32, copy=True)
        absolute = self.offset[l] + torch.arange(T, device=h.device)
        mask = (absolute if self.keyed == "absolute" else torch.arange(T, device=h.device)) >= self.start
        hf[:, mask] += self.delta[l][:, None, :]
        h2 = hf.to(h.dtype)
        for step in range(N_REALIZED_STEPS):                    # realized dose at P - 1 and decode steps 1-3
            j = self.start + step - self.offset[l]
            if 0 <= j < T:
                dh = h2[:, j].float() - h[:, j].float()
                for b in range(B):
                    if self.layer_of[b] == l and self.alphas[b] != 0.0:
                        n = float(dh[b].norm())
                        sgn = math.copysign(1.0, self.alphas[b])        # A21: against v, not +w
                        self.realized[b].append((step, sgn * float(dh[b] @ self.dirs[b]) / n if n > 0 else 0.0,
                                                 n / abs(self.alphas[b])))
        self.n_edited += self.rows_at[l] * int(mask.sum())
        self.offset[l] += T
        self.n_calls[l] += 1
        return with_hidden(out, h2)


def steer_one(model, layer: int, w: torch.Tensor, alpha_abs: float, start: int, n: int = 1) -> ResidualSteer:
    """n identical rows steered at one layer."""
    return ResidualSteer(model, [(layer, w, alpha_abs)] * n, start)


@torch.no_grad()
def logits_with(model, ids: torch.Tensor, hook=None) -> torch.Tensor:
    """Full-sequence logits (model dtype) with `hook` (a ResidualSteer or PoleAblator, or None) active."""
    ids = ids.to(model.device)
    with hook if hook is not None else contextlib.nullcontext():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits


# ------------------------------------------------------------------------------------------ (b) residual RMS
@torch.no_grad()
def residual_rms(model, windows: list, exclude_first: int) -> dict:
    """SCHEMA §3.3 rms.json body: per decoder layer l, over every position p >= exclude_first of every window ([1, T]
    id tensors), with h_l the layer's output (the steering hook point) in fp32:
      rms = sqrt(mean over positions p and coordinates j of h_l[p, j]^2)   (per-coordinate RMS)
      mean_norm, median_norm = mean and median of ||h_l[p]||_2.
    Units: residual-stream coordinates; alpha * rms * w has L2 norm alpha * rms."""
    layers = text_tower(model).layers
    norms = {i: [] for i in range(len(layers))}
    handles = []
    for i, lyr in enumerate(layers):
        def grab(m, a, o, i=i):
            h = hidden_of(o)
            norms[i].append(h[0, exclude_first:].float().norm(dim=-1).double().cpu())
        handles.append(lyr.register_forward_hook(grab))
    for ids in windows:
        assert ids.shape[0] == 1 and ids.shape[1] > exclude_first, f"window shape {tuple(ids.shape)}"
        ids = ids.to(model.device)
        model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    for h in handles:
        h.remove()
    d, out = d_model(model), []
    for i in range(len(layers)):
        assert len(norms[i]) == len(windows), f"layer {i} hook fired {len(norms[i])} times for {len(windows)} windows"
        x = torch.cat(norms[i])
        out.append({"layer": i, "rms": float((x.pow(2).sum() / (x.numel() * d)).sqrt()), "mean_norm": float(x.mean()),
                    "median_norm": float(x.median())})
    return {"exclude_first": exclude_first, "n_positions": int(x.numel()), "d_model": d, "layers": out}


# ------------------------------------------------------------------------------------------ norms and unembedding
def norm_eps(norm) -> float:
    for k in ("eps", "variance_epsilon"):
        if hasattr(norm, k):
            return float(getattr(norm, k))
    raise AssertionError(f"{type(norm).__name__} has neither eps nor variance_epsilon")


@torch.no_grad()
def norm_gain(norm) -> torch.Tensor:
    """SCHEMA §3.3: norm(ones(d)) in fp32 (Gemma's 1 + weight, Qwen's and Olmo's weight, times 1/sqrt(1 + eps))."""
    return norm(torch.ones(1, norm.weight.shape[0], dtype=torch.float32, device=norm.weight.device))[0].float()


def norm_gain_exact(norm) -> torch.Tensor:
    """gamma with norm(x) = gamma * x * inv_rms(x) exactly: norm_gain times sqrt(1 + eps)."""
    return norm_gain(norm) * math.sqrt(1.0 + norm_eps(norm))


def inv_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """1 / sqrt(mean(x^2) + eps) over the last axis, fp32."""
    return torch.rsqrt(x.float().pow(2).mean(-1) + eps)


def final_norm_gain(model) -> torch.Tensor:
    return norm_gain(text_tower(model).norm)


def unembedding(model) -> torch.Tensor:
    head = model.get_output_embeddings()
    assert getattr(head, "bias", None) is None, "lm_head has a bias"
    return head.weight


def softcap(model, logits: torch.Tensor) -> torch.Tensor:
    """The model's final_logit_softcapping (cap * tanh(logits / cap)) when its config sets one (SPEC §2.3)."""
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return logits if cap is None else cap * torch.tanh(logits / cap)


@torch.no_grad()
def unembed(model, x: torch.Tensor) -> torch.Tensor:
    """W_U @ x for x [K, d] in fp32 -> [K, V], VOCAB_CHUNK rows of W_U at a time."""
    W = unembedding(model)
    x = x.float().to(W.device)
    return torch.cat([x @ W[r:r + VOCAB_CHUNK].float().T for r in range(0, W.shape[0], VOCAB_CHUNK)], 1)


# ------------------------------------------------------------------------------------------ (c) vocab projection
@torch.no_grad()
def vocab_projection(model, tok, poles: list, attested_ids: torch.Tensor, k: int = 20) -> dict:
    """SCHEMA §3.3 projection lines without the pole fields: {pole id: {"w_down_norm", "top", "bottom", "top_full",
    "bottom_full"}} from the fp32 scores W_U @ (gain * w), gain = final_norm(ones), w = sign * u_i; top/bottom over
    `attested_ids` (a 1-D id tensor), *_full over the whole vocabulary; entries {"id", "str": tok.decode([id]),
    "logit"}. poles = [(pole id, layer, neuron, sign)]. The scores are a direction's projection, not a position's
    logits: no softcap."""
    g = final_norm_gain(model)
    keep = torch.zeros(unembedding(model).shape[0], dtype=torch.bool, device=g.device)
    keep[attested_ids.to(g.device)] = True

    def entries(vals, ids):
        return [{"id": int(i), "str": tok.decode([int(i)]), "logit": float(v)} for v, i in zip(vals, ids)]
    out = {}
    for pid, l, n, s in poles:
        scores = unembed(model, (g * pole_direction(model, l, n, s))[None])[0]
        inf = torch.tensor(float("inf"), device=scores.device)
        top, bot = torch.where(keep, scores, -inf).topk(k), torch.where(keep, scores, inf).topk(k, largest=False)
        tf, bf = scores.topk(k), scores.topk(k, largest=False)
        out[pid] = {"w_down_norm": float(w_down_column(model, l, n).norm()),
                    "top": entries(top.values, top.indices), "bottom": entries(bot.values, bot.indices),
                    "top_full": entries(tf.values, tf.indices), "bottom_full": entries(bf.values, bf.indices)}
    return out


# ------------------------------------------------------------------------------------------ (d) direct effect
def nll_rows(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """-log softmax(logits)[target] per row, fp32."""
    return -F.log_softmax(logits.float(), -1).gather(-1, targets.to(logits.device)[:, None])[:, 0]


GAIN_ORDER = {"qwen3": "cast_then_gain", "gemma3": "gain_then_cast", "olmo3": "gain_then_cast"}   # the families' bf16 casts


@torch.no_grad()
def manual_logits_model_dtype(model, h: torch.Tensor, eps=None) -> torch.Tensor:
    """A23: the manual final path in the model dtype for h [K, d] (the model dtype, as captured): fp32 normalization
    h * rsqrt(mean(h^2) + eps) (eps: the final norm's, or a per-row tensor), the gain gamma = norm_gain_exact applied
    with the family's cast (Qwen 3 casts the normalized vector to the model dtype, then multiplies by the gain; Gemma 3
    and Olmo 3 multiply in fp32, then cast), the unembedding W_U in the model dtype, then the softcap. Built from the
    norm's epsilon and gain and W_U, never by calling the model's norm or lm_head modules."""
    norm = text_tower(model).norm
    dt = h.dtype
    e = norm_eps(norm) if eps is None else eps[:, None]
    x = h.float() * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + e)
    g = norm_gain_exact(norm)
    y = g.to(dt) * x.to(dt) if GAIN_ORDER[model_type(model)] == "cast_then_gain" else (g * x).to(dt)
    return softcap(model, F.linear(y[None], unembedding(model).to(dt))[0])


def known_zero_last(model) -> int:
    """A24: G-known's registered failing input zeroes the MLP outputs of layers 0 .. n_layers // 4 inclusive."""
    return len(text_tower(model).layers) // 4


class MlpZeroLayers:
    """A24: neuron_off_core's mlp_zero hook on layers 0 .. last at once (positions >= exclude_first), as one ablator
    for neuron_off_core.next_token (`calls` counts forwards)."""
    fire = None

    def __init__(self, model, last: int, exclude_first: int):
        self.parts = [C.PoleAblator(model, "mlp_zero", l, exclude_first=exclude_first) for l in range(last + 1)]

    @property
    def calls(self) -> int:
        return self.parts[0].calls

    def __enter__(self):
        for x in self.parts:
            x.__enter__()
        return self

    def __exit__(self, *exc):
        for x in self.parts:
            x.__exit__(*exc)


def early_layer(model) -> int:
    """G-repair-fit's registered failing input reads the residual entering this layer: n_layers // 2."""
    return len(text_tower(model).layers) // 2


@torch.no_grad()
def repair_readout(model, ids: torch.Tensor, positions: list, targets: list, layer: int, neuron: int) -> dict:
    """SCHEMA v5 §3.3 repair fields from one clean forward of ids ([1, T]) at `positions` (targets = the next token at
    each), SPEC §2.3 (A6). Returns {"logits": the model's logits at the positions (model dtype, for the t-only identity
    check), "rows": per position}, each row, fp32:
      a_ours            the clean a_i (down_proj input);
      clean_nll_ours    the model's own NLL of the target;
      delta_r (primary) Qwen 3 a_i * W_down[:, i]; Gemma 3, Olmo 3 gain_post * (a_i * W_down[:, i]) * s_post, s_post =
                        1 / sqrt(mean(y^2) + eps) of the clean down_proj output y (post-MLP scale frozen);
      h, s              the clean final residual and s = 1 / sqrt(mean(h^2) + eps); final_rms = sqrt(mean h^2);
      clean_nll_manual  NLL from W_U @ (gamma * h * s), gamma = norm_gain_exact (G-repair-fit, against clean_nll_ours);
      direct_dnll       NLL(W_U @ (gamma * (h - delta_r) * s)) - clean_nll_manual: both scales frozen (PRIMARY);
      direct_dnll_postrenorm   delta_r = postnorm(y) - postnorm(y - a_i * W_down[:, i]) (post-MLP norm recomputed),
                        final scale frozen; equals the primary for Qwen 3;
      direct_dnll_finalrenorm  NLL(W_U @ final_norm(h - delta_r)) - clean_nll_manual (final norm recomputed);
      clean_nll_manual_early   the frozen-scale path on the residual entering layer early_layer(model) = n_layers // 2
                        (G-repair-fit's registered failing input; n_layers - 2 moved NLL by only 0.10-0.20 nats on
                        two of three preflight-2 poles); delta_r_norm = ||delta_r (primary)||;
      clean_nll_manual_model_dtype  the manual formula (final-norm epsilon and gain, unembedding, softcap) evaluated in
                        the model dtype (manual_logits_model_dtype); A23: G-repair-fit compares it with
                        clean_nll_ours (preflight 2: the fp32 manual path differs from the bf16 model by about
                        0.03 x (1 - p(target)), bf16 rounding of the logits);
      clean_nll_manual_model_dtype_wrong_eps  the same with the final-norm epsilon replaced by each context's own
                        mean(h^2), so the normalized vector shrinks by about 1/sqrt(2) (A23's registered failing input).
    Every manual path applies the config's final_logit_softcapping when set (SPEC §2.3)."""
    assert ids.shape[0] == 1 and len(positions) == len(targets) > 0, "a batch of 1 and one target per position"
    tower = text_tower(model)
    lyr = tower.layers[layer]
    post = POST_MLP_NORM[model_type(model)]
    cap, handles = {}, []
    handles.append(lyr.mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__("a", a[0][0, :, neuron].float())))
    handles.append(tower.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__("h_md", a[0][0])))
    handles.append(tower.layers[early_layer(model)].register_forward_pre_hook(
        lambda m, a: cap.__setitem__("early", a[0][0].float())))
    if post:
        handles.append(getattr(lyr, post).register_forward_pre_hook(lambda m, a: cap.__setitem__("y", a[0][0].float())))
    ids = ids.to(model.device)
    pos = torch.tensor(positions, device=model.device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=pos).logits[0]
    for h in handles:
        h.remove()
    tgt = torch.tensor(targets, device=model.device)
    h_md = cap["h_md"][pos]                                         # the model dtype, as the model's own final norm sees it
    a, h, e = cap["a"][pos], h_md.float(), cap["early"][pos]
    removed = a[:, None] * w_down_column(model, layer, neuron)[None, :]
    if post:
        norm = getattr(lyr, post)
        y = cap["y"][pos]
        d_frozen = norm_gain_exact(norm)[None, :] * removed * inv_rms(y, norm_eps(norm))[:, None]
        d_renorm = norm(y) - norm(y - removed)
    else:
        d_frozen = d_renorm = removed
    s = inv_rms(h, norm_eps(tower.norm))[:, None]
    g = norm_gain_exact(tower.norm)
    K = len(positions)
    x = torch.cat([g * h * s, g * (h - d_frozen) * s, g * (h - d_renorm) * s, tower.norm(h - d_frozen).float(),
                   g * e * inv_rms(e, norm_eps(tower.norm))[:, None]])
    nll = nll_rows(softcap(model, unembed(model, x)), tgt.repeat(5)).cpu().view(5, K)
    own = nll_rows(logits, tgt).cpu()
    md = nll_rows(manual_logits_model_dtype(model, h_md), tgt).cpu()
    md_wrong = nll_rows(manual_logits_model_dtype(model, h_md, h.pow(2).mean(-1)), tgt).cpu()    # eps = mean(h^2) (A23)
    rows = [{"a_ours": float(a[k]), "clean_nll_ours": float(own[k]), "clean_nll_manual": float(nll[0, k]),
             "clean_nll_manual_model_dtype": float(md[k]), "clean_nll_manual_model_dtype_wrong_eps": float(md_wrong[k]),
             "direct_dnll": float(nll[1, k] - nll[0, k]), "direct_dnll_postrenorm": float(nll[2, k] - nll[0, k]),
             "direct_dnll_finalrenorm": float(nll[3, k] - nll[0, k]), "clean_nll_manual_early": float(nll[4, k]),
             "delta_r_norm": float(d_frozen[k].norm()), "final_rms": float(h[k].pow(2).mean().sqrt())} for k in range(K)]
    return {"logits": logits, "rows": rows}


# ------------------------------------------------------------------------------------------ literal overlap (§5.1, A1)
NO_SPACE_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "THAI", "LAO", "KHMER", "MYANMAR", "TIBETAN")
SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "er", "ed", "es", "ly", "s")


def _norm_text(t: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", t).casefold()


def _strip_ps(t: str) -> str:
    import unicodedata
    t = t.strip()
    while t and unicodedata.category(t[0])[0] in "PS":
        t = t[1:]
    while t and unicodedata.category(t[-1])[0] in "PS":
        t = t[:-1]
    return t.strip()


def _destem(w: str) -> str:
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def literal_match(label: str, token: str) -> bool:
    """SPEC §5.1 rule for one token against one label (NFKC + casefold; P/S stripped; scripts without spaces match as
    substrings; otherwise word match, one-suffix stems of >= 3 characters, or a >= 4-character prefix either way;
    1-character tokens skipped, 2-3-character tokens only by equality)."""
    import unicodedata
    lab, t = _norm_text(label), _strip_ps(_norm_text(token))
    if not any(unicodedata.category(ch)[0] in "LN" for ch in t):
        return False
    if any(any(sc in unicodedata.name(ch, "") for sc in NO_SPACE_SCRIPTS) for ch in t):
        return t in lab
    if len(t) <= 1:
        return False
    words, cur = [], ""
    for ch in lab:
        if unicodedata.category(ch)[0] in "LN":
            cur += ch
        elif cur:
            words.append(cur)
            cur = ""
    if cur:
        words.append(cur)
    for w in words:
        if t == w:
            return True
        if len(t) <= 3:
            continue
        st, sw = _destem(t), _destem(w)
        if (st != t or sw != w) and st == sw and len(st) >= 3:
            return True
        if w.startswith(t) or (len(w) >= 4 and t.startswith(w)):
            return True
    return False


def literal_overlap(label: str, tokens: list) -> bool:
    """True if the label names any token of the judged top-20 (SPEC §5.1, A1)."""
    return any(literal_match(label, t) for t in tokens)


# ------------------------------------------------------------------------------------------ (e) steered sampling
def prompt_ids(tok, prompt: str) -> list:
    """BOS (when the tokenizer defines one, prepended explicitly) + tok(prompt, add_special_tokens=False)."""
    body = tok(prompt, add_special_tokens=False).input_ids
    assert tok.bos_token_id is None or not body or body[0] != tok.bos_token_id, f"{prompt[:40]!r} tokenizes to a BOS"
    return C.bos_prefix(tok) + body


@torch.no_grad()
def generate_rows(model, tok, prompt: str, rows: list, seed: int, sampling: dict, keyed: str = "absolute") -> list:
    """One generate call: len(rows) continuations of BOS + prompt, row b steered by rows[b] = (layer or None, w or
    None, alpha_abs) from sequence position P - 1 onward (P = prompt length with BOS: the first generated token is
    already steered). torch.manual_seed(seed) right before generate; every sampling knob from `sampling`
    (temperature, top_k, top_p, max_new_tokens; repetition_penalty must be 1.0). A call whose rows all have layer
    None runs with no hook. Each item: completion (new tokens before the first EOS, decoded with
    skip_special_tokens=True), completion_token_ids, n_new_tokens, stopped_eos, realized_cos (the smallest realized
    cosine over P - 1 and decode steps 1-3, None for an unsteered row or alpha 0) and realized_steps (the job keeps
    them for G-gen and drops them from the sample record). keyed "per_forward" is G-gen's failing input."""
    assert sampling.get("repetition_penalty", 1.0) == 1.0, "repetition_penalty other than 1.0 is not supported"
    p = prompt_ids(tok, prompt)
    ids = torch.tensor([p], device=model.device).repeat(len(rows), 1)
    steer = ResidualSteer(model, rows, start=len(p) - 1, keyed=keyed) if any(r[0] is not None for r in rows) else None
    cfg = C.generation_config(tok, sampling["temperature"], sampling["max_new_tokens"], sampling["top_k"], sampling["top_p"])
    torch.manual_seed(seed)
    with steer if steer is not None else contextlib.nullcontext():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
    if steer is not None:
        assert steer.calls >= 1, "the steering hook never fired"
    eos, res = tok.eos_token_id, []
    for b, o in enumerate(out[:, len(p):].tolist()):
        stop = eos in o
        new = o[:o.index(eos)] if stop else o
        res.append({"completion": tok.decode(new, skip_special_tokens=True), "completion_token_ids": new,
                    "n_new_tokens": len(new), "stopped_eos": stop,
                    "realized_cos": None if steer is None else steer.realized_cos[b],
                    "realized_steps": [] if steer is None else steer.realized[b]})
    return res


@torch.no_grad()
def score_rows(model, p: list, completions: list, pad: int) -> list:
    """Per completion (a list of token ids): the mean NLL (nats) of its tokens under the unhooked model with the
    prompt ids p (BOS included) as context, None for an empty completion. Rows are right-padded with `pad` and scored
    together (causal: a pad after a row's last token changes nothing before it), logits only at the scored positions,
    fp32 log-softmax on at most SCORE_POSITIONS (rows x positions) at a time."""
    P, out = len(p), [None] * len(completions)
    live = [j for j, c in enumerate(completions) if c]
    if not live:
        return out
    L = max(len(completions[j]) for j in live)
    step = max(1, SCORE_POSITIONS // L)
    pos = torch.arange(P - 1, P - 1 + L, device=model.device)
    for s in range(0, len(live), step):
        js = live[s:s + step]
        ids = torch.tensor([p + completions[j] + [pad] * (L - len(completions[j])) for j in js], device=model.device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=pos).logits
        lp = F.log_softmax(logits.float(), -1)
        for r, j in enumerate(js):
            n = len(completions[j])
            t = torch.tensor(completions[j], device=model.device)
            out[j] = float(-lp[r, torch.arange(n, device=model.device), t].double().mean())
        del logits, lp
    return out


@torch.no_grad()
def steered_nll(model, ids: list, P: int, rows_hook: ResidualSteer | None) -> torch.Tensor:
    """Teacher-forced NLL [n] of ids[P:] given ids[:P] (one row), the hook active from its own start; the prediction
    columns P - 1 .. len(ids) - 2."""
    n = len(ids) - P
    assert n > 0, "nothing to score"
    r = C.next_token(model, torch.tensor([ids]), rows_hook, positions=list(range(P - 1, P - 1 + n)), targets=ids[P:])
    return r["nll"][0]


# ------------------------------------------------------------------------------------------ G-place placements
@torch.no_grad()
def realized_change(model, ids: list, P: int, layer: int, w: torch.Tensor, alpha_abs: float, where: str) -> dict:
    """SPEC §6 G-place: the realized change dh of decoder layer `layer`'s output at positions >= P - 1 when
    alpha_abs * w is added at `where`: "layer_output" (the steering hook), "mlp_input" (forward pre-hook on
    layer.mlp, the registered failing input), "down_proj_output" (forward hook on mlp.down_proj, before the
    post-MLP norm; reported) or "layer_output_wrong_sign" (the steering hook adding -v; A21's registered failing
    input). Median and range of cos(dh, sign(alpha_abs) * w) (the signed intended vector, A21) and
    ||dh|| / |alpha_abs| over those positions."""
    lyr = text_tower(model).layers[layer]
    t = torch.tensor([ids], device=model.device)
    mask = (torch.arange(t.shape[1], device=t.device) >= P - 1)[None, :, None]
    w = w.to(model.device)
    cap = {}

    def run(key):
        h = lyr.register_forward_hook(lambda m, a, o: cap.__setitem__(key, hidden_of(o)[0].float()))
        model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False, logits_to_keep=1)
        h.remove()
    run("clean")
    add = lambda x: (x.float() + mask * (alpha_abs * w)).to(x.dtype)  # noqa: E731
    if where in ("layer_output", "layer_output_wrong_sign"):
        with ResidualSteer(model, [(layer, w, alpha_abs if where == "layer_output" else -alpha_abs)], P - 1):
            run("steered")
    elif where == "mlp_input":
        hk = lyr.mlp.register_forward_pre_hook(lambda m, a: (add(a[0]),) + tuple(a[1:]))
        run("steered")
        hk.remove()
    else:
        assert where == "down_proj_output", where
        hk = lyr.mlp.down_proj.register_forward_hook(lambda m, a, o: add(o))
        run("steered")
        hk.remove()
    dh = (cap["steered"] - cap["clean"])[P - 1:]
    cos = (dh @ (math.copysign(1.0, alpha_abs) * w)) / dh.norm(dim=-1).clamp_min(1e-30)
    ratio = dh.norm(dim=-1) / abs(alpha_abs)
    return {"placement": where, "n_positions": int(dh.shape[0]), "cos_median": float(cos.median()),
            "cos_min": float(cos.min()), "ratio_median": float(ratio.median()), "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max())}


# ------------------------------------------------------------------------------------------ firing check (§5.5)
@torch.no_grad()
def firing_max(model, ids: list, layer: int, neuron: int, sign: int, first: int) -> tuple:
    """(max over positions >= first of sign * a_i, its position) in one clean forward of ids."""
    down = text_tower(model).layers[layer].mlp.down_proj
    cap = {}
    h = down.register_forward_pre_hook(lambda m, a: cap.__setitem__("a", a[0][0, :, neuron].float() * sign))
    t = torch.tensor([ids], device=model.device)
    model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False, logits_to_keep=1)
    h.remove()
    x = cap["a"][first:]
    assert x.numel(), f"no position >= {first} in {len(ids)} tokens"
    j = int(x.argmax())
    return float(x[j]), j + first


# ------------------------------------------------------------------------------------------ coherence (§5.2, A7)
MIN_NEW = 8
COH_SHARE = 0.90


def split_halves(alpha0: list, model_key: str) -> tuple:
    """SPEC §5.2 (A7): default_rng([20261025, model_index, 12]).permutation over the rows in file order; the first half
    (floor(n/2) rows) calibrates, the rest tests."""
    perm = np.random.default_rng([SEED, MODEL_INDEX[model_key], 12]).permutation(len(alpha0))
    h = len(alpha0) // 2
    return [alpha0[i] for i in perm[:h]], [alpha0[i] for i in perm[h:]]


def coherence_band(calibration: list, natural: list) -> dict:
    """SPEC §5.2: L = 2.5th percentile of nll_clean_mean over the calibration half of the alpha = 0 rows (null rows,
    n_new = 0, counted and not used); U = max(97.5th percentile of the same, 97.5th percentile over the natural
    calibration spans). SCHEMA §3.1 band keys."""
    v = [r["nll_clean_mean"] for r in calibration if r["nll_clean_mean"] is not None]
    s = [r["nll_clean_mean"] for r in natural]
    assert v and s, "coherence band needs alpha = 0 samples and calibration spans"
    ha, hn = float(np.percentile(v, 97.5)), float(np.percentile(s, 97.5))
    return {"lo": float(np.percentile(v, 2.5)), "hi": max(ha, hn), "hi_alpha0": ha, "hi_natural": hn,
            "n_alpha0": len(v), "n_alpha0_empty": len(calibration) - len(v), "n_natural": len(s)}


def is_coherent(row: dict, band: dict) -> bool:
    v = row["nll_clean_mean"]
    return row["n_new_tokens"] >= MIN_NEW and v is not None and band["lo"] <= v <= band["hi"]


def g_coh(band: dict, failing: dict, natural_test: list, alpha0_test: list, label_prompted: list | None = None) -> dict:
    """SPEC §6 G-coh (A7): both failing inputs ({"repeated": row, "random": row}) collapsed; the natural test spans,
    the test half of the alpha = 0 rows and (Arm 2) the label-prompted rows coherent at >= 0.90; L < U.
    SCHEMA §3.1 coherence block."""
    fail = {k: {"nll": r["nll_clean_mean"], "collapsed": not is_coherent(r, band)} for k, r in failing.items()}
    pas = {}
    for k, rows in (("natural_test", natural_test), ("alpha0", alpha0_test), ("label_prompted", label_prompted)):
        if rows is None:
            continue
        share = sum(is_coherent(r, band) for r in rows) / len(rows) if rows else float("nan")
        pas[k] = {"n": len(rows), "share_coherent": share, "pass": bool(rows) and share >= COH_SHARE}
    ok = all(x["collapsed"] for x in fail.values()) and all(x["pass"] for x in pas.values()) and band["lo"] < band["hi"]
    return {"band": band, "failing_inputs": fail, "passing_inputs": pas, "pass": bool(ok)}


# ------------------------------------------------------------------------------------------ statistics (SPEC §7, §7.5)
SEED = 20261025
N_BOOT = 10_000
MODEL_INDEX = {"Q4B": 0, "G4": 1, "O7": 2}
ROOM_OUTER, ROOM_INNER = 2_000, 1_000
ROOM_NULL_MAX, ROOM_PASS_MIN, ROOM_FAIL_N, ROOM_EQUIV_MIN = 0.05, 0.80, 5, 0.50


def num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _pct(v) -> tuple:
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def ci(x, stat=np.mean) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0, "est": None, "lo": None, "hi": None}
    rng = np.random.default_rng(SEED)
    lo, hi = _pct(stat(x[rng.integers(0, len(x), (N_BOOT, len(x)))], axis=1))
    return {"n": len(x), "est": float(stat(x)), "lo": lo, "hi": hi}


def two_group(x, y, f) -> dict:
    """f(mean x, mean y) with x and y resampled separately from one generator."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if not len(x) or not len(y):
        return {"n": [len(x), len(y)], "est": None, "lo": None, "hi": None}
    rng = np.random.default_rng(SEED)
    mx = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    my = y[rng.integers(0, len(y), (N_BOOT, len(y)))].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = f(mx, my)
    lo, hi = _pct(v[np.isfinite(v)]) if np.isfinite(v).any() else (None, None)
    return {"n": [len(x), len(y)], "est": float(f(x.mean(), y.mean())) if y.mean() != 0 or f is not ratio else None,
            "lo": lo, "hi": hi, "n_nonfinite": int((~np.isfinite(v)).sum())}


def diff(a, b):
    return a - b


def ratio(a, b):
    return a / b


def pooled(groups: list, stat=np.mean) -> dict:
    """stat over the concatenation of groups, each group resampled within itself."""
    groups = [np.asarray(g, dtype=np.float64) for g in groups if len(g)]
    if not groups:
        return {"n": 0, "est": None, "lo": None, "hi": None}
    rng = np.random.default_rng(SEED)
    draws = np.concatenate([g[rng.integers(0, len(g), (N_BOOT, len(g)))] for g in groups], axis=1)
    lo, hi = _pct(stat(draws, axis=1))
    return {"n": int(sum(len(g) for g in groups)), "est": float(stat(np.concatenate(groups))), "lo": lo, "hi": hi}


def ranks(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    r = np.empty(len(a))
    r[a.argsort(kind="stable")] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def spearman(x, y) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def spearman_ci(x, y) -> dict:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    v = np.array([spearman(x[i], y[i]) for i in rng.integers(0, len(x), (N_BOOT, len(x)))])
    lo, hi = _pct(v[np.isfinite(v)]) if np.isfinite(v).any() else (None, None)
    return {"n": len(x), "est": spearman(x, y), "lo": lo, "hi": hi}


def kappa(a: list, b: list):
    """Cohen's kappa of two equal-length categorical answer lists; None with fewer than two categories."""
    cats = sorted(set(a) | set(b))
    if len(cats) < 2 or not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def derangement(n: int, model_key: str) -> np.ndarray:
    rng = np.random.default_rng([SEED, MODEL_INDEX[model_key], 11])
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


# ------------------------------------------------------------------------------------------ paired and two-way bootstraps
def paired_ratio_boot(counts: dict, combine, n_boot: int = N_BOOT) -> dict:
    """SPEC §5.4 (A3): one prompt index set per replicate, shared by every rate. counts: {name: (num [n], den [n])}
    per prompt; each rate = sum(num[drawn]) / sum(den[drawn]); combine({name: rate array}) -> statistic array."""
    names = sorted(counts)
    n = len(counts[names[0]][0])
    assert all(len(counts[k][0]) == len(counts[k][1]) == n for k in names), "every rate needs one value per prompt"
    nd = {k: (np.asarray(counts[k][0], dtype=np.float64), np.asarray(counts[k][1], dtype=np.float64)) for k in names}
    with np.errstate(divide="ignore", invalid="ignore"):
        est = float(combine({k: np.array([nd[k][0].sum() / nd[k][1].sum()]) for k in names})[0])
        idx = np.random.default_rng(SEED).integers(0, n, (n_boot, n))
        v = combine({k: nd[k][0][idx].sum(1) / nd[k][1][idx].sum(1) for k in names})
    v = v[np.isfinite(v)]
    lo, hi = _pct(v) if len(v) else (None, None)
    return {"est": est, "lo": lo, "hi": hi, "n_prompts": n, "n_finite": int(len(v))}


def two_way_boot(steer: dict, zero: dict, prompts0: list, stat, n_boot: int = N_BOOT) -> dict:
    """SPEC §5.7 (A10): each replicate resamples poles, then each drawn pole's judged prompts at the steered dose, and
    the shared alpha = 0 prompts once (one draw used for every pole's alpha = 0 term). steer {pole: [0/1 outcomes]};
    zero {pole: {prompt_i: 0/1}}; stat(m_d [P], m_0 [P], poles) -> {name: value}. Returns {name: {est, lo, hi}}."""
    poles = sorted(steer)
    arrs = [np.asarray(steer[p], dtype=np.float64) for p in poles]
    ks = np.array([len(a) for a in arrs])
    Z = np.array([[zero.get(p, {}).get(q, np.nan) for q in prompts0] for p in poles], dtype=np.float64).reshape(len(poles), len(prompts0))
    with np.errstate(invalid="ignore"):
        m0_full = np.nanmean(Z, axis=1) if len(prompts0) else np.full(len(poles), np.nan)
    est = stat(np.array([a.mean() for a in arrs]), m0_full, poles)
    rng = np.random.default_rng(SEED)
    reps = {k: [] for k in est}
    for _ in range(n_boot):
        idx = rng.integers(0, len(poles), len(poles))
        md = np.empty(len(poles))
        for k in np.unique(ks[idx]):
            slots = np.nonzero(ks[idx] == k)[0]
            mat = np.stack([arrs[idx[j]] for j in slots])
            md[slots] = mat[np.arange(len(slots))[:, None], rng.integers(0, k, (len(slots), k))].mean(1)
        p0 = rng.integers(0, len(prompts0), len(prompts0))
        with np.errstate(invalid="ignore"):
            m0 = np.nanmean(Z[idx][:, p0], axis=1) if len(prompts0) else np.full(len(poles), np.nan)
        for k, v in stat(md, m0, [poles[j] for j in idx]).items():
            reps[k].append(v)
    out = {}
    for k, v in reps.items():
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        lo, hi = _pct(v) if len(v) else (None, None)
        out[k] = {"est": est[k], "lo": lo, "hi": hi, "n": len(poles), "n_finite": int(len(v))}
    return out


# ------------------------------------------------------------------------------------------ room checks (§7.5, A4)
EQUIV = {"H_write": (-0.10, 0.10), "H_depth": (-0.15, 0.15), "H_ONE": (-0.10, 0.10), "H_suff": (1 / 3 - 0.08, 1 / 3 + 0.08),
         "H_pred_agree": (0.8, 1.25), "H_pred_conf": (-0.2, 0.2), "B1": (-0.05, 0.05)}


def rule_interval(rule: str, vals, rng, inner: int) -> tuple:
    """(estimate, lo, hi) of the rule's statistic on one outer draw, by an inner bootstrap."""
    if rule in ("H_write", "H_ONE", "H_suff", "B1"):
        m = vals[rng.integers(0, len(vals), (inner, len(vals)))].mean(1)
        return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
    if rule in ("H_depth", "H_pred_agree"):
        x, y = vals
        mx = x[rng.integers(0, len(x), (inner, len(x)))].mean(1)
        my = y[rng.integers(0, len(y), (inner, len(y)))].mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = mx - my if rule == "H_depth" else mx / my
            e = x.mean() - y.mean() if rule == "H_depth" else x.mean() / y.mean()
        r = r[np.isfinite(r)]
        return (float(e), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))) if len(r) else (float("nan"),) * 3
    assert rule == "H_pred_conf", rule
    o, cf = vals
    v = np.array([spearman(o[i], cf[i]) for i in rng.integers(0, len(o), (inner, len(o)))])
    v = v[np.isfinite(v)]
    return (spearman(o, cf), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"),) * 3


def rule_fires(rule: str, est: float, lo: float, hi: float, margin: float = 0.0) -> tuple:
    """(supported, equivalent) of the rule on one interval (A3, A4); H_pred_conf's "supported" here is its directional
    REFUTED, the label the planted relation must make fire (§7.5)."""
    if not (num(lo) and num(hi)):
        return False, False
    a, b = EQUIV[rule]
    eq = a < lo and hi < b
    if rule == "H_write":
        return lo > 0 and est >= margin, eq
    if rule == "H_suff":
        return lo > 1 / 3, eq
    if rule == "H_pred_agree":
        return lo > 1 and est >= 2, eq
    if rule == "H_pred_conf":
        return lo > 0 or hi < 0, eq
    return lo > 0, eq


BERNOULLI_RULES = ("H_write", "H_depth", "H_ONE", "B1")      # A25: the Arm 0 rules whose R-pass plant is a Bernoulli draw


def planted_values(rule: str, plant, n, effect: float, rng):
    """A25: one planted draw of n items (H_depth: a pair of group sizes) from the per-item plant records, each
    [y_s1, y_s2, (y_s1', y_s2' of the H_ONE control,) p] with p the model's pooled shuffled yes-rate. A planted true-label
    answer is Bernoulli(p + effect (1 - p)); an answer under no effect (the H_ONE control, H_depth's tercile 0, B1's
    re-explained label) is Bernoulli(p); shuffled answers stay as judged. Returns the rule's per-item values. p is
    p_sh, the model's pooled shuffled-label yes-rate on the ONE items (SPEC §14.2 A25)."""
    hit = lambda p, e: (rng.random(len(p)) < p + e * (1 - p)).astype(np.float64)  # noqa: E731

    def take(recs, k):
        r = np.asarray(recs, dtype=np.float64)
        return r[rng.integers(0, len(r), k)]
    if rule == "H_depth":
        a, b = take(plant[0], n[0]), take(plant[1], n[1])
        return hit(a[:, -1], effect) - (a[:, 0] + a[:, 1]) / 2, hit(b[:, -1], 0.0) - (b[:, 0] + b[:, 1]) / 2
    r = take(plant, n)
    p = r[:, -1]
    if rule == "H_write":
        return hit(p, effect) - (r[:, 0] + r[:, 1]) / 2
    if rule == "H_ONE":
        return (hit(p, effect) - (r[:, 0] + r[:, 1]) / 2) - (hit(p, 0.0) - (r[:, 2] + r[:, 3]) / 2)
    assert rule == "B1", rule
    return hit(p, effect) - hit(p, 0.0)


def room_draws(rule: str, null, n, effect, margin=0.0, outer=ROOM_OUTER, inner=ROOM_INNER, plant=None, pass_only=False) -> dict:
    """R-null, R-equiv (unplanted draws) and R-pass (planted draws) at size n (an int, or a pair of group sizes). With
    `plant` (A25, BERNOULLI_RULES) every draw comes from planted_values, the unplanted ones at effect +0.0; without it
    the null values are drawn and the effect is added to them (the frozen constant plant)."""
    assert plant is None or rule in BERNOULLI_RULES, f"no Bernoulli plant for {rule}"
    rng = np.random.default_rng(SEED)

    def draw(planted: bool):
        sup = eq = 0
        for _ in range(outer):
            if plant is not None:
                vals = planted_values(rule, plant, n, effect if planted else 0.0, rng)
            elif rule in ("H_depth", "H_pred_agree"):
                x0, y0 = (np.asarray(v, dtype=np.float64) for v in null)
                x, y = x0[rng.integers(0, len(x0), n[0])], y0[rng.integers(0, len(y0), n[1])]
                if planted:
                    x = x * effect if rule == "H_pred_agree" else x + effect
                vals = (x, y)
            elif rule == "H_pred_conf":
                o0, c0 = (np.asarray(v, dtype=np.float64) for v in null)
                i = rng.integers(0, len(o0), n)
                o, cf = o0[i], rng.permutation(c0[i])
                if planted:
                    o = o + effect * (ranks(cf) / n - 0.5)
                vals = (o, cf)
            else:
                v0 = np.asarray(null, dtype=np.float64)
                vals = v0[rng.integers(0, len(v0), n)] + (effect if planted else 0.0)
            f = rule_fires(rule, *rule_interval(rule, vals, rng, inner), margin)
            sup, eq = sup + f[0], eq + f[1]
        return sup / outer, eq / outer
    r_null, r_equiv = (None, None) if pass_only else draw(False)
    r_pass, _ = draw(True)
    return {"n": n, "r_null": r_null, "r_pass": r_pass, "r_equiv": r_equiv}


def room(rule: str, null, n, effect, margin=0.0, outer=ROOM_OUTER, inner=ROOM_INNER, plant=None) -> dict:
    """§7.5 with its registered failing input (R-pass at n = 5 must fail) and stamps. With `plant` (A25): every draw is
    a Bernoulli plant (R-null and R-equiv from the +0.0 plant at the full n); registered failing inputs: the amended
    plant at n = 5 must give R-pass < 0.80, and the +0.0 plant must fire the supported label at most 5% at n = 5 and fail
    the 0.80 power line at the full n (its full-n rate is R-null); the frozen constant-plant reading is kept beside
    ("frozen_constant_plant") and printed in every label's room stamp."""
    if (isinstance(n, tuple) and min(n) < 1) or (not isinstance(n, tuple) and n < 1) or \
            not np.size(null[0] if isinstance(null, tuple) else null):
        return {"n": n, "cancelled": True, "r_pass": None, "r_equiv": None, "stamps": ["room check not computed (no data)"],
                "readable": False, "equiv_readable": False}
    r = room_draws(rule, null, n, effect, margin, outer, inner, plant)
    n5 = (ROOM_FAIL_N, ROOM_FAIL_N) if isinstance(n, tuple) else ROOM_FAIL_N
    f = room_draws(rule, null, n5, effect, margin, outer, inner, plant, pass_only=True)
    can_fail = f["r_pass"] < ROOM_PASS_MIN
    extra, zero_ok, zstamps = {}, True, []
    if plant is not None:
        z5 = room_draws(rule, null, n5, 0.0, margin, outer, inner, plant, pass_only=True)["r_pass"]
        zero_ok = z5 <= ROOM_NULL_MAX and r["r_null"] < ROOM_PASS_MIN
        zstamps = ([] if z5 <= ROOM_NULL_MAX else ["zero plant fires above 5% at n = 5"]) + \
            ([] if r["r_null"] < ROOM_PASS_MIN else ["zero plant reaches the power line"])
        c = room_draws(rule, null, n, effect, margin, outer, inner)
        c5 = room_draws(rule, null, n5, effect, margin, outer, inner, pass_only=True)["r_pass"]
        extra = {"plant": "A25: true-label answers Bernoulli(p_sh + E (1 - p_sh)) on the planted side, Bernoulli(p_sh) on the "
                          "unplanted side; shuffled answers as judged",
                 "failing_input_zero_plant": {"r_pass_n5": z5, "r_pass_full_n": r["r_null"], "fails": zero_ok},
                 "frozen_constant_plant": {"r_null": c["r_null"], "r_pass": c["r_pass"], "r_equiv": c["r_equiv"],
                                           "failing_input_n5": {"r_pass": c5, "fails": c5 < ROOM_PASS_MIN}}}
    stamps = ([] if can_fail else ["room check cannot fail"]) + zstamps + \
        (["rule fires on the null"] if r["r_null"] > ROOM_NULL_MAX else []) + \
        (["underpowered for a direction"] if r["r_pass"] < ROOM_PASS_MIN else []) + \
        (["no room for equivalence"] if r["r_equiv"] < ROOM_EQUIV_MIN else [])
    ok = can_fail and zero_ok
    return {**r, "effect": effect, "failing_input_n5": {**f, "fails": can_fail}, **extra, "cancelled": not ok, "stamps": stamps,
            "readable": ok and r["r_null"] <= ROOM_NULL_MAX, "equiv_readable": r["r_equiv"] >= ROOM_EQUIV_MIN}


def room_stamp(rm: dict | None) -> str:
    """The R-pass and R-equiv stamp printed beside every label (A4)."""
    if rm is None:
        return ""
    f = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
    extra = f"; {'; '.join(rm['stamps'])}" if rm["stamps"] else ""
    fz = rm.get("frozen_constant_plant")
    frozen = "" if fz is None else (f"; frozen constant plant: R-pass {f(fz['r_pass'])}, n = 5 {f(fz['failing_input_n5']['r_pass'])}"
                                    + ("" if fz["failing_input_n5"]["fails"] else " (cannot fail)"))
    return f" [R-pass {f(rm.get('r_pass'))}, R-equiv {f(rm.get('r_equiv'))}{extra}{frozen}]"


def apply_room(label: str, supported: set, rm: dict | None) -> str:
    """A supported (or EQUIVALENT) label whose room check says it cannot be read is replaced by that statement; the
    room stamp follows every label."""
    if rm is None:
        return label
    h = label.split(",")[0]
    if (h in supported and not rm["readable"]) or (h.startswith("EQUIVALENT") and not rm["equiv_readable"]):
        label = f"{h} NOT READABLE" + label[len(h):]
    return label + room_stamp(rm)
