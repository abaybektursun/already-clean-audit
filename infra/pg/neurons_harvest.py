"""D1 neuron harvest (N1): SPEC_D1 §2.1 as amended by DATA_CONTRACT §2. One checkpoint per call.

The object. For neuron i of layer l: g = W_gate x, u = W_up x (the bf16 outputs of the model's own
`mlp.gate_proj` / `mlp.up_proj`, read by forward hooks), and a = act(g) * u recomputed in fp32 with the
layer's own `mlp.act_fn`. Pole axis 0 is a > 0, axis 1 is a < 0. A forward pre-hook on `mlp.down_proj`
reads the model's own input to the down projection; G0 compares it with the recomputed a on the first
batch and the job aborts if any layer differs by more than 1e-2 of that layer's max |a|.

The stream. Windows of `window` corpus tokens taken in order from `tokens.npy`, a BOS prepended when the
tokenizer defines one; BOS positions never enter any statistic. Global token index `tok` = position in
`tokens.npy`. Window position 0 (the first corpus token of a window, which has no preceding corpus
context; on Gemma it is the token after BOS) is excluded from topk/ and stats/ (SPEC §4.2.3); a separate
running top-K over those positions gives `stats/pos0_share`, the share of each pole's pre-exclusion top-K
that sat there. When a band boundary falls inside a batch, the batch is split there, so a band snapshot
holds exactly the tokens before the boundary.

Poles. `--poles`, or with `--sample-neurons` the drawn neurons' poles, are the pole list: `poles/` holds
poles.json, vocab.json, top1000.npz, positions.npz and exemplar_a.npz for them (without `--positions`,
the positions are the poles' own final top-K, DATA_CONTRACT §7).

Per layer, on the GPU: exact running top-K per pole (merged per batch), a histogram of log10|a| per pole
over fixed edges, and for `--poles` an exact running top-1000 and the values at given positions.

Fail-loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

K_TOP = 50
K_POLES = 1000
HIST_B = 400
HIST_LO, HIST_HI = -6.0, 4.0          # log10 |a|; |a| < 1e-6 counts in bin 0, |a| >= 1e4 in bin B-1
G0_TOL = 1e-2
G1_TOL = 0.02
NLL_TOKENS = 120_000
CTX_BEFORE, CTX_AFTER = 64, 16
METRICS_EVERY = 50
FIRST_BATCH_POLES, FIRST_BATCH_CONTEXTS = 3, 10
EXPECTED_MODEL_TYPE = {"Q4B": "qwen3", "Q4": "qwen3", "C0": "qwen3", "G4": "gemma3", "O7": "olmo3", "O2": "olmo2"}
VOCAB_RULE = ("score = sign * W_U @ (gamma_final * W_down[:, i]) in fp32, sign = +1 for the + pole and -1 for "
              "the - pole (the direction the pole writes when it fires); gamma_final = final_norm(ones), which "
              "is the norm's effective gain for every RMSNorm form; the 1/rms factor is a positive scalar and "
              "does not change the ranking")
SAMPLE_RULE = "neuron i of n sits in layer floor(i * n_layers / n); neurons within a layer drawn without " \
              "replacement by numpy default_rng(seed); both poles of each neuron"
POLE_RE = re.compile(r"^L(\d+)N(\d+)([+-])$")
PROBE_RULE = ("max_a, min_a, argmax_tok, argmin_tok over text positions >= 1: BOS (when defined) and text "
              "position 0 excluded, as the harvest excludes window position 0 (SPEC_D1 §4.2); positions index "
              "the text's own tokens; n_tokens is the full count; a text needs >= 2 tokens")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dump(path, obj):
    """JSON written to a temp file and renamed, so the 30 s S3 sync never uploads half a file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------ poles
def pole_id(layer: int, neuron: int, sign: int) -> str:
    return f"L{layer:02d}N{neuron}{'+-'[sign]}"


def parse_pole(p: str) -> tuple[int, int, int]:
    m = POLE_RE.match(p)
    assert m, f"bad pole id {p!r}"
    return int(m[1]), int(m[2]), "+-".index(m[3])


def load_poles(path) -> list:
    """A JSON list of pole ids, or a `samples/sample_draw.json` (its `poles` field)."""
    obj = json.loads(pathlib.Path(path).read_text())
    poles = obj["poles"] if isinstance(obj, dict) else obj
    assert poles and all(isinstance(p, str) for p in poles), f"{path}: no pole list"
    [parse_pole(p) for p in poles]
    assert len(set(poles)) == len(poles), f"{path}: duplicate poles"
    return poles


def draw_neurons(n_layers: int, width: int, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    layers = [(i * n_layers) // n for i in range(n)]
    out = []
    for lyr in sorted(set(layers)):
        for x in rng.choice(width, size=layers.count(lyr), replace=False):
            out.append([lyr, int(x)])
    return out


def positions_from(src_dir, poles: list) -> np.ndarray:
    """tok [P, K_TOP] for `poles` from a previous harvest's out dir: its poles/positions.npz when it has one
    (that harvest had the same --poles), else its topk/ top-K. Every grid directory gets the same columns."""
    src = pathlib.Path(src_dir)
    pos = src / "poles" / "positions.npz"
    if pos.exists():
        prev = json.loads((src / "poles" / "poles.json").read_text())
        assert prev == poles, f"{src}: positions.npz was written for other poles"
        return np.load(pos)["tok"]
    cache = {}
    rows = []
    for p in poles:
        lyr, n, s = parse_pole(p)
        if lyr not in cache:
            cache[lyr] = np.load(src / "topk" / f"layer_{lyr:02d}.npz")["tok"]
        rows.append(cache[lyr][n, s])
    return np.stack(rows)


# ------------------------------------------------------------------------------------------ model
def text_tower(model):
    """The decoder that holds `.layers` and the final `.norm` (Gemma3ForConditionalGeneration and
    Qwen3_5ForConditionalGeneration: its language model)."""
    tower = model.model.language_model if model.config.model_type in ("gemma3", "qwen3_5") else model.model
    assert hasattr(tower, "layers") and hasattr(tower, "norm"), type(tower).__name__
    return tower


def load_model(local_dir: str, model_key: str, device: str = "cuda"):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(local_dir)
    assert cfg.model_type == EXPECTED_MODEL_TYPE[model_key], \
        f"{model_key}: config model_type {cfg.model_type}, expected {EXPECTED_MODEL_TYPE[model_key]}"
    cls = transformers.Gemma3ForConditionalGeneration if cfg.model_type == "gemma3" else AutoModelForCausalLM
    model = cls.from_pretrained(local_dir, dtype=torch.bfloat16, device_map=device,
                                attn_implementation="sdpa").eval()
    return model, AutoTokenizer.from_pretrained(local_dir)


def act_name(act) -> str:
    """Which function the module computes, measured on a grid (not read from a class name)."""
    x = torch.linspace(-8, 8, 321)
    y = act(x)
    if torch.allclose(y, F.silu(x), atol=1e-5):
        return "silu"
    if torch.allclose(y, F.gelu(x, approximate="tanh"), atol=1e-5):
        return "gelu_tanh"
    raise AssertionError(f"activation {type(act).__name__} is neither silu nor gelu_tanh")


WRONG_ACT = {"silu": lambda x: F.gelu(x, approximate="tanh"), "gelu_tanh": F.silu}
WRONG_U_SCALE = 1.1


class MLPTap:
    """Forward hooks on every layer's mlp.gate_proj and mlp.up_proj (outputs g, u) and a forward pre-hook on
    mlp.down_proj (its input, the model's own a). The hooks exist only inside `run(fn, call)`, which calls
    `fn(layer, g, u, a_model)` once per layer during `call()` and asserts every layer fired once, in order;
    any other forward of the model (G1) runs without them."""

    def __init__(self, layers):
        self.layers = layers
        self.n = len(layers)
        self.fn, self.fired, self._gu = None, [], {}
        for lyr, layer in enumerate(layers):
            for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
                assert hasattr(layer.mlp, name), f"layer {lyr}: {type(layer.mlp).__name__} has no {name}"

    def _keep(self, key):
        def hook(module, args, output):
            self._gu[key] = output
        return hook

    def _down(self, lyr):
        def hook(module, args):
            g, u = self._gu.pop("g"), self._gu.pop("u")
            self.fired.append(lyr)
            self.fn(lyr, g, u, args[0])
        return hook

    def run(self, fn, call):
        self.fn, self.fired = fn, []
        handles = []
        for lyr, layer in enumerate(self.layers):
            mlp = layer.mlp
            handles += [mlp.gate_proj.register_forward_hook(self._keep("g")),
                        mlp.up_proj.register_forward_hook(self._keep("u")),
                        mlp.down_proj.register_forward_pre_hook(self._down(lyr))]
        out = call()
        for h in handles:
            h.remove()
        assert self.fired == list(range(self.n)), f"MLP hooks fired for layers {self.fired}"
        self.fn = None
        return out


def forward(model, x, logits: bool):
    return model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                 logits_to_keep=0 if logits else 1)


# ------------------------------------------------------------------------------------------ gates
def g0_ratios(model, tap, acts, x, u_scale: float = 1.0) -> list:
    """Per layer max|act(g) * u_scale * u - a_model| / max|a_model| over one forward of `x` (fp32
    recomputation). u_scale = 1 is the gate; u_scale = 1.1 is its second registered failing input, whose
    power does not depend on |g| (the activation swap's does)."""
    out = {}

    def fn(lyr, g, u, a_model):
        am = a_model.float()
        out[lyr] = float((acts[lyr](g.float()) * (u_scale * u.float()) - am).abs().max() / am.abs().max())
    tap.run(fn, lambda: forward(model, x, False))
    return [out[lyr] for lyr in range(tap.n)]


def g0_pass(ratios) -> bool:
    return all(r <= G0_TOL for r in ratios)


def g1_pass(mean_nll: float, banked: float) -> bool:
    return abs(mean_nll - banked) <= G1_TOL


def g1_wikitext(model, tok, device) -> dict:
    """The STAGEDIFF B3 call for C0 (`_stagediff_shared.run_b3`): `shell_text_perplexity` through
    StageForward with no BOS, window 2048, the first 120,000 wikitext-2 test tokens."""
    from infra.pg.stagediff_models import _text_forward_cls
    from infra.pg.text_eval import shell_text_perplexity
    assert tok.bos_token_id is None, "G1 reproduces the C0 forward (no BOS); this tokenizer defines a BOS"
    r = shell_text_perplexity(_text_forward_cls()(model, tok, device, bos_id=None), window=2048,
                              max_tokens=120_000)
    return {k: r[k] for k in ("mean_nll", "ppl", "n_tokens", "n_dropped_leading_targets", "forward", "protocol",
                              "samples") if k in r}


# ------------------------------------------------------------------------------------------ state
def hist_bin(absval: torch.Tensor) -> torch.Tensor:
    scale = HIST_B / (HIST_HI - HIST_LO)
    return ((torch.log10(absval.clamp_min(1e-30)) - HIST_LO) * scale).floor().clamp(0, HIST_B - 1).long()


def hist_edges() -> np.ndarray:
    return (10.0 ** np.linspace(HIST_LO, HIST_HI, HIST_B + 1)).astype(np.float32)


class TopK:
    """Exact running top-k per pole of n neurons. Score s = a for the + pole and -a for the - pole, kept
    sorted descending; a slot with s <= 0 is empty (fewer than k tokens of that sign)."""

    def __init__(self, n: int, k: int, device):
        self.k = k
        self.s = torch.full((n, 2, k), -math.inf, device=device)
        self.tok = torch.full((n, 2, k), -1, dtype=torch.long, device=device)
        self.g = torch.zeros((n, 2, k), device=device)
        self.u = torch.zeros((n, 2, k), device=device)

    def update(self, a, g, u, tok):
        """a, g, u: [T, n] fp32; tok: [T] global token index."""
        kk = min(self.k, a.shape[0])
        for p, s in ((0, a), (1, -a)):
            v, i = torch.topk(s, kk, dim=0)
            cs = torch.cat([self.s[:, p], v.T], 1)
            j = torch.topk(cs, self.k, dim=1).indices
            self.s[:, p] = cs.gather(1, j)
            self.tok[:, p] = torch.cat([self.tok[:, p], tok[i].T], 1).gather(1, j)
            self.g[:, p] = torch.cat([self.g[:, p], g.gather(0, i).T], 1).gather(1, j)
            self.u[:, p] = torch.cat([self.u[:, p], u.gather(0, i).T], 1).gather(1, j)

    def max_abs(self) -> torch.Tensor:
        return self.s[:, :, 0].clamp_min(0)

    def arrays(self) -> dict:
        """Contract layout: a signed (+ pole descending, - pole ascending); empty slots tok -1, a/g/u NaN."""
        valid = self.s > 0
        nan = torch.tensor(float("nan"), device=self.s.device)
        sign = torch.tensor([1.0, -1.0], device=self.s.device).view(1, 2, 1)
        return {"tok": torch.where(valid, self.tok, -1).cpu().numpy(),
                "a": torch.where(valid, self.s * sign, nan).cpu().numpy(),
                "g": torch.where(valid, self.g, nan).cpu().numpy(),
                "u": torch.where(valid, self.u, nan).cpu().numpy()}


class Hist:
    """Counts of log10|a| per pole over fixed edges; a == 0 belongs to neither pole and is dropped."""

    def __init__(self, n: int, device):
        self.n = n
        self.counts = torch.zeros(n * 2 * HIST_B + 1, dtype=torch.long, device=device)

    def update(self, a):
        n = a.shape[1]
        idx = hist_bin(a.abs())
        idx += (a < 0).long() * HIST_B
        idx += torch.arange(n, device=a.device) * (2 * HIST_B)
        idx[a == 0] = n * 2 * HIST_B
        self.counts += torch.bincount(idx.view(-1), minlength=self.counts.numel())

    def array(self) -> np.ndarray:
        return self.counts[:-1].view(self.n, 2, HIST_B).to(torch.int32).cpu().numpy()


class Exemplars:
    """a, g, u at fixed global token positions, one entry per (pole p, slot e) with its neuron."""

    def __init__(self, p, e, neuron, tok, device):
        self.p, self.e = np.asarray(p), np.asarray(e)
        self.neuron = torch.tensor(neuron, dtype=torch.long, device=device)
        self.tok = torch.tensor(tok, dtype=torch.long, device=device)
        m = len(tok)
        self.a, self.g, self.u = (torch.full((m,), float("nan"), device=device) for _ in range(3))
        self.filled = torch.zeros(m, dtype=torch.bool, device=device)

    def update(self, a, g, u, tok):
        """tok: [T] ascending global indices of the rows of a, g, u (window position 0 is not among them)."""
        r = torch.searchsorted(tok, self.tok).clamp(max=tok.numel() - 1)
        hit = tok[r] == self.tok
        self.a = torch.where(hit, a[r, self.neuron], self.a)
        self.g = torch.where(hit, g[r, self.neuron], self.g)
        self.u = torch.where(hit, u[r, self.neuron], self.u)
        self.filled |= hit


class LayerState:
    def __init__(self, n: int, device, pole_neurons=None, exemplars=None):
        self.topk = TopK(n, K_TOP, device)
        self.pos0 = TopK(n, K_TOP, device)          # window position 0 only; for stats/pos0_share
        self.hist = Hist(n, device)
        self.pole_neurons = None if pole_neurons is None else torch.tensor(pole_neurons, device=device)
        self.ptop = None if pole_neurons is None else TopK(len(pole_neurons), K_POLES, device)
        self.ex = exemplars

    def update(self, a, g, u, tok):
        self.topk.update(a, g, u, tok)
        self.hist.update(a)
        if self.ptop is not None:
            idx = self.pole_neurons
            self.ptop.update(a[:, idx], g[:, idx], u[:, idx], tok)
        if self.ex is not None:
            self.ex.update(a, g, u, tok)

    def snapshot(self):
        return self.topk.arrays(), self.hist.array(), self.pos0.arrays()


# ------------------------------------------------------------------------------------------ finalize
def hist_quantile_edge(hist: np.ndarray, q: float, edges: np.ndarray) -> np.ndarray:
    """Upper edge of the bin holding the q-quantile (inverted CDF: the ceil(q*n)-th smallest |a|)."""
    n = hist.sum(-1)
    k = np.maximum(np.ceil(q * n), 1)
    b = np.minimum((np.cumsum(hist, -1) < k[..., None]).sum(-1), HIST_B - 1)
    return np.where(n > 0, edges[b + 1], np.nan).astype(np.float32)


def same_token_frac(tok: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Share of a pole's filled top-K slots held by its most common corpus token id."""
    valid = tok >= 0
    ids = np.where(valid, tokens[np.where(valid, tok, 0)], -1)
    eq = (ids[..., :, None] == ids[..., None, :]) & valid[..., :, None] & valid[..., None, :]
    nv = valid.sum(-1)
    return np.where(nv > 0, eq.sum(-1).max(-1) / np.maximum(nv, 1), np.nan).astype(np.float32)


def sign_counts(g: np.ndarray, u: np.ndarray, valid: np.ndarray) -> np.ndarray:
    gp, up = g >= 0, u >= 0
    cats = (gp & up, gp & ~up, ~gp & up, ~gp & ~up)
    return np.stack([(c & valid).sum(-1) for c in cats], -1).astype(np.int16)


def pos0_share(tk: dict, p0: dict) -> np.ndarray:
    """Share of each pole's pre-exclusion top-K (top-K over window position 0 and the rest together) that sat
    at window position 0; NaN for a pole with no token of its sign."""
    sign = np.array([1.0, -1.0], np.float32).reshape(1, 2, 1)
    s = np.concatenate([np.where(np.isnan(x), -np.inf, x * sign) for x in (tk["a"], p0["a"])], -1)
    k = tk["a"].shape[-1]
    order = np.argsort(-s, axis=-1, kind="stable")[..., :k]
    valid = np.take_along_axis(s, order, -1) > 0
    n = valid.sum(-1)
    return np.where(n > 0, ((order >= k) & valid).sum(-1) / np.maximum(n, 1), np.nan).astype(np.float32)


def finalize_layer(tk: dict, hist: np.ndarray, p0: dict, tokens: np.ndarray, edges: np.ndarray) -> tuple[dict, dict]:
    valid = tk["tok"] >= 0
    max_abs = np.where(valid[..., 0], np.abs(tk["a"][..., 0]), 0).astype(np.float32)
    thr_bin = hist_bin(torch.from_numpy(0.01 * max_abs)).numpy()
    cum_from = np.cumsum(hist[..., ::-1], -1)[..., ::-1]                 # count in bins >= b
    density = np.where(max_abs > 0, np.take_along_axis(cum_from, thr_bin[..., None], -1)[..., 0], 0)
    stats = {"max_abs": max_abs, "hist": hist, "hist_edges": edges,
             "n_pole_tokens": hist.sum(-1).astype(np.int64),
             "q99": hist_quantile_edge(hist, 0.99, edges), "q999": hist_quantile_edge(hist, 0.999, edges),
             "q9999": hist_quantile_edge(hist, 0.9999, edges), "density": density.astype(np.int64),
             "top_same_token_frac": same_token_frac(tk["tok"], tokens),
             "sign_counts": sign_counts(tk["g"], tk["u"], valid), "pos0_share": pos0_share(tk, p0)}
    topk = {k: (v if k == "tok" else v.astype(np.float32)) for k, v in tk.items()}
    return topk, stats


def write_layers(root, layer_states: dict, tokens: np.ndarray) -> dict:
    """topk/ and stats/ under `root` from {layer: (topk arrays, hist, pos0 arrays)}; returns {layer: (topk, stats)}."""
    root = pathlib.Path(root)
    (root / "topk").mkdir(parents=True, exist_ok=True)
    (root / "stats").mkdir(parents=True, exist_ok=True)
    edges, out = hist_edges(), {}
    for lyr, (tk, hist, p0) in sorted(layer_states.items()):
        topk, stats = finalize_layer(tk, hist, p0, tokens, edges)
        np.savez_compressed(root / "topk" / f"layer_{lyr:02d}.npz", **topk)
        np.savez_compressed(root / "stats" / f"layer_{lyr:02d}.npz", **stats)
        out[lyr] = (topk, stats)
    return out


# ------------------------------------------------------------------------------------------ text
def decoder(tok):
    return lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)


def context(tokens: np.ndarray, offsets: np.ndarray, t: int, decode) -> dict:
    """64 tokens before and 16 after global token t, never crossing its document's bounds."""
    d = int(np.searchsorted(offsets, t, "right") - 1)
    s, e = int(offsets[d]), int(offsets[d + 1])
    return {"doc": d, "pos": t - s, "before": decode(tokens[max(s, t - CTX_BEFORE):t]),
            "token": decode(tokens[t:t + 1]), "after": decode(tokens[t + 1:min(e, t + 1 + CTX_AFTER)])}


def pole_contexts(topk: dict, n: int, s: int, tokens, offsets, decode, limit=None) -> list:
    rows = []
    for r in range(topk["tok"].shape[-1] if limit is None else limit):
        t = int(topk["tok"][n, s, r])
        if t < 0:
            break
        rows.append({"rank": r, "tok": t, **context(tokens, offsets, t, decode),
                     "a": float(topk["a"][n, s, r]), "g": float(topk["g"][n, s, r]),
                     "u": float(topk["u"][n, s, r])})
    return rows


def vocab_table(model, layers, final_norm, tok, poles: list, k: int = 10) -> dict:
    W_U = model.get_output_embeddings().weight
    d = W_U.shape[1]
    gamma = final_norm(torch.ones(1, d, dtype=final_norm.weight.dtype, device=W_U.device)).float()[0]
    cols = []
    for p in poles:
        lyr, n, s = parse_pole(p)
        cols.append(layers[lyr].mlp.down_proj.weight[:, n].float() * gamma * (1.0 if s == 0 else -1.0))
    scores = W_U.float() @ torch.stack(cols, 1)                                   # [V, P]
    top, bot = torch.topk(scores, k, dim=0), torch.topk(scores, k, dim=0, largest=False)
    dec = decoder(tok)
    return {p: {"promoted": [[dec([int(i)]), float(v)] for v, i in zip(top.values[:, j], top.indices[:, j])],
                "suppressed": [[dec([int(i)]), float(v)] for v, i in zip(bot.values[:, j], bot.indices[:, j])]}
            for j, p in enumerate(poles)}


def probe_scores(model, tap, acts, tok, bos, texts: list, poles: list, device) -> tuple[dict, dict]:
    """One forward per text, BOS prepended when defined and excluded. Per (text, pole): max and min of a over
    text positions >= 1 and their positions within the text (PROBE_RULE). Text position 0 has no preceding
    context (Qwen3 has no BOS; on Gemma it is the first token after BOS), so it is not scored."""
    parsed = [parse_pole(p) for p in poles]
    by_layer = {}
    for j, (lyr, n, _) in enumerate(parsed):
        by_layer.setdefault(lyr, []).append((j, n))
    T, P = len(texts), len(poles)
    max_a, min_a = np.full((T, P), np.nan, np.float32), np.full((T, P), np.nan, np.float32)
    argmax, argmin = np.full((T, P), -1, np.int32), np.full((T, P), -1, np.int32)
    n_tok = np.zeros(T, np.int32)
    strings = {}
    off = 0 if bos is None else 1
    dec = decoder(tok)
    for t, rec in enumerate(texts):
        ids = tok(rec["text"], add_special_tokens=False)["input_ids"]
        assert len(ids) >= 2, (f"probe text {rec['text_id']!r} has {len(ids)} token(s); text position 0 is not "
                               f"scored, so at least 2 are needed")
        x = torch.tensor([([bos] if bos is not None else []) + list(ids)], device=device)
        cur = {}

        def fn(lyr, g, u, a_model):
            if lyr in by_layer:
                idx = torch.tensor([n for _, n in by_layer[lyr]], device=g.device)
                cur[lyr] = (acts[lyr](g[0, off + 1:][:, idx].float()) * u[0, off + 1:][:, idx].float()).cpu()
        tap.run(fn, lambda: forward(model, x, False))
        for lyr, entries in by_layer.items():
            for c, (j, _) in enumerate(entries):
                col = cur[lyr][:, c]
                max_a[t, j], argmax[t, j] = float(col.max()), 1 + int(col.argmax())
                min_a[t, j], argmin[t, j] = float(col.min()), 1 + int(col.argmin())
        n_tok[t] = len(ids)
        strings[rec["text_id"]] = [dec([i]) for i in ids]
    return ({"text_ids": np.array([r["text_id"] for r in texts]), "max_a": max_a, "min_a": min_a,
             "argmax_tok": argmax, "argmin_tok": argmin, "n_tokens": n_tok, "probe_rule": np.array(PROBE_RULE)},
            strings)


# ------------------------------------------------------------------------------------------ run
def segments(toks: list, t0: int, t1: int, marks: dict) -> list:
    """A batch covering tokens [t0, t1), cut at every band end inside it. `toks` are the batch's ascending
    token-index arrays (rest rows, position-0 rows). Returns [([(r0, r1) per array], band ending there or
    None)]; a range may be empty."""
    cuts = sorted((e, b) for b, e in marks.items() if t0 < e <= t1)
    edges = [[0] + [int(np.searchsorted(v, e)) for e, _ in cuts] + [len(v)] for v in toks]
    bands = [b for _, b in cuts] + [None]
    return [([(ed[k], ed[k + 1]) for ed in edges], bands[k]) for k in range(len(bands))]


def run(out, *, model, tok, corpus: dict, model_key: str, repo: str, revision: str, branch=None,
        max_tokens, window: int = 2048, batch: int = 8, snapshot_bands=(), poles=None, positions=None,
        probe_texts=None, sample_neurons: int = 0, seed: int = 0, wikitext_parity=None, device="cuda",
        nll_tokens: int = NLL_TOKENS, extra_manifest=None) -> dict:
    """Harvest one checkpoint into `out` (DATA_CONTRACT §2). `corpus` is `neurons_corpus.load_tok(...)`.
    `max_tokens` None = every full window of the corpus."""
    import transformers
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    tokens, offsets = corpus["tokens"], corpus["doc_offsets"]
    tower = text_tower(model)
    layers = tower.layers
    L = len(layers)
    width = int(layers[0].mlp.gate_proj.weight.shape[0])
    hidden = int(layers[0].mlp.gate_proj.weight.shape[1])
    acts = [layer.mlp.act_fn for layer in layers]
    assert len({type(a_).__name__ for a_ in acts}) == 1, "layers use different activation classes"
    act = act_name(acts[0])
    from infra.pg.matrix_models import vocab_sha256
    assert vocab_sha256(tok) == corpus["record"]["vocab_sha256"], \
        "model tokenizer vocabulary differs from the corpus tokenizer's: token ids would not mean the same"
    bos = tok.bos_token_id
    n_windows = (tokens.size if max_tokens is None else max_tokens) // window
    n_windowed = n_windows * window
    n_scored = n_windows * (window - 1)            # tokens entering topk/ and stats/: window position 0 excluded
    assert n_windows >= 1 and n_windowed <= tokens.size, f"{n_windowed} tokens asked, corpus has {tokens.size}"
    marks = {int(b): corpus["band_end"][int(b)] for b in snapshot_bands}
    for b, e in marks.items():
        assert e <= n_windowed, f"band {b} ends at token {e}, past the {n_windowed} windowed tokens"

    if poles is None:
        assert sample_neurons > 0, "give --poles or --sample-neurons"
        drawn = draw_neurons(L, width, sample_neurons, seed)
        selected = [pole_id(lyr, n, s) for lyr, n in drawn for s in (0, 1)]
    else:
        assert sample_neurons == 0, "--poles and --sample-neurons are exclusive"
        drawn, selected = None, list(poles)
    from_poles_arg = poles is not None
    poles = selected                                 # the pole list for poles/, drawn or given
    parsed = {p: parse_pole(p) for p in selected}
    for lyr, n, _ in parsed.values():
        assert lyr < L and n < width, f"pole outside the model: L{lyr} N{n}"
    if positions is not None:
        assert positions.shape[0] == len(poles), "positions need one row per pole"
        assert positions.max() < n_windowed, "a position lies past the windowed tokens"
        assert not (positions[positions >= 0] % window == 0).any(), "a position sits at window position 0"

    manifest = {"model_key": model_key, "repo": repo, "revision": revision, "branch": branch,
                "transformers": transformers.__version__, "torch": torch.__version__,
                "act_fn": type(acts[0]).__name__, "act_fn_measured": act, "model_type": model.config.model_type,
                "model_class": type(model).__name__, "n_layers": L, "width": width, "hidden": hidden,
                "dtype": str(next(model.parameters()).dtype), "attn": model.config._attn_implementation,
                "corpus_manifest_sha256": corpus["manifest_sha256"], "tokname": corpus["tokname"],
                "tokens_sha256": corpus["tokens_sha256"], "bos_id": bos, "n_tokens_scored": n_scored,
                "n_tokens_windowed": n_windowed, "excluded_positions": "BOS and window position 0",
                "probe_excluded_positions": None if probe_texts is None else "text position 0",
                "probe_rule": None if probe_texts is None else PROBE_RULE,
                "max_tokens_arg": max_tokens, "n_windows": n_windows, "batch": batch, "window": window, "topk_K": K_TOP,
                "poles_K": K_POLES, "hist_edges": {"lo_log10": HIST_LO, "hi_log10": HIST_HI, "bins": HIST_B,
                                                   "below_lo": "bin 0", "at_or_above_hi": "bin B-1"},
                "density_rule": "every token in the bin holding 0.01*max_abs and above (bin resolution)",
                "vocab_rule": VOCAB_RULE, "sample_rule": SAMPLE_RULE, "snapshot_bands": sorted(marks),
                "band_end_tok": marks, "started_utc": utc(), **(extra_manifest or {})}
    dump(out / "manifest.json", manifest)
    tap = MLPTap(layers)
    off = 0 if bos is None else 1

    def windows(w0, w1):
        x = torch.from_numpy(tokens[w0 * window:w1 * window].astype(np.int64)).view(w1 - w0, window)
        if bos is not None:
            x = torch.cat([torch.full((w1 - w0, 1), bos, dtype=torch.long), x], 1)
        return x.to(device)

    controls = {}
    with torch.inference_mode():
        # ---- G0 on the first batch, before anything is harvested ------------------------------
        x0 = windows(0, min(batch, n_windows))
        ratios = g0_ratios(model, tap, acts, x0)
        wrong = g0_ratios(model, tap, [WRONG_ACT[act]] * L, x0)
        wrong2 = g0_ratios(model, tap, acts, x0, u_scale=WRONG_U_SCALE)
        controls["G0"] = {"ratio": ratios, "pass": g0_pass(ratios), "tol": G0_TOL, "n_windows": int(x0.shape[0]),
                          "failing_input": {"act": {"silu": "gelu_tanh", "gelu_tanh": "silu"}[act],
                                            "ratio": wrong, "pass": g0_pass(wrong),
                                            "n_layers_over_tol": sum(r > G0_TOL for r in wrong)},
                          "failing_input_2": {"u_scale": WRONG_U_SCALE, "ratio": wrong2, "pass": g0_pass(wrong2),
                                              "n_layers_over_tol": sum(r > G0_TOL for r in wrong2)}}
        dump(out / "controls.json", controls)
        assert controls["G0"]["pass"], f"G0 FAILED: per-layer ratios {ratios}"
        if wikitext_parity is not None:
            g1 = g1_wikitext(model, tok, device)
            controls["G1"] = {"wikitext_mean_nll": g1["mean_nll"], "banked": wikitext_parity,
                              "gap": g1["mean_nll"] - wikitext_parity, "tolerance": G1_TOL,
                              "pass": g1_pass(g1["mean_nll"], wikitext_parity), "detail": g1}
            dump(out / "controls.json", controls)

        # ---- state ----------------------------------------------------------------------------
        pole_neurons, ex_entries = {}, {}
        for j, p in enumerate(poles):
            lyr, n, _ = parse_pole(p)
            pole_neurons.setdefault(lyr, [])
            if n not in pole_neurons[lyr]:
                pole_neurons[lyr].append(n)
            if positions is not None:
                for e, t in enumerate(positions[j]):
                    if t >= 0:
                        ex_entries.setdefault(lyr, []).append((j, e, n, int(t)))
        states = []
        for lyr in range(L):
            ex = None
            if lyr in ex_entries:
                p_, e_, n_, t_ = zip(*ex_entries[lyr])
                ex = Exemplars(p_, e_, n_, t_, device)
            states.append(LayerState(width, device, pole_neurons.get(lyr), ex))

        # ---- the harvest loop -------------------------------------------------------------------
        nll_sum, nll_n = 0.0, 0
        timing = {"harvest_loop_s": 0.0, "snapshot_write_s": 0.0, "metrics_write_s": 0.0}
        n_batches = math.ceil(n_windows / batch)
        done_tokens = 0                              # tokens entered into topk/ and stats/

        def write_metrics(batches_done, complete):
            t = time.time()
            dead = sum(int((st.topk.max_abs() == 0).sum()) for st in states)
            mean_max = [st.topk.max_abs().mean(0).tolist() for st in states]
            dump(out / "metrics.json", {
                "n_tokens_scored": done_tokens, "n_tokens_target": n_scored, "batches_done": batches_done,
                "n_batches": n_batches, "tokens_per_second": done_tokens / max(timing["harvest_loop_s"], 1e-9),
                "complete": complete, "G0_pass": controls["G0"]["pass"],
                "G1_pass_or_null": controls["G1"]["pass"] if "G1" in controls else None,
                "dead_poles": dead, "n_poles": 2 * width * L, "mean_max_abs": mean_max, "updated_utc": utc()})
            timing["metrics_write_s"] += time.time() - t

        for bi in range(n_batches):
            t_b = time.time()
            w0, w1 = bi * batch, min((bi + 1) * batch, n_windows)
            t0, t1 = w0 * window, w1 * window
            x = windows(w0, w1)
            starts = np.arange(w0, w1) * window
            tok_rest = (starts[:, None] + np.arange(1, window)[None]).reshape(-1)
            segs = segments([tok_rest, starts], t0, t1, marks)
            tok_rest_d, tok0_d = torch.from_numpy(tok_rest).to(device), torch.from_numpy(starts).to(device)
            snaps = {}

            def on_layer(lyr, g, u, a_model):
                n = g.shape[-1]
                g, u = g[:, off:], u[:, off:]                    # [B, window, n]: corpus positions
                g0, u0 = g[:, 0].float(), u[:, 0].float()        # window position 0
                g, u = g[:, 1:].reshape(-1, n).float(), u[:, 1:].reshape(-1, n).float()
                a, a0 = acts[lyr](g) * u, acts[lyr](g0) * u0
                for ((r0, r1), (q0, q1)), band in segs:
                    if r1 > r0:
                        states[lyr].update(a[r0:r1], g[r0:r1], u[r0:r1], tok_rest_d[r0:r1])
                    if q1 > q0:
                        states[lyr].pos0.update(a0[q0:q1], g0[q0:q1], u0[q0:q1], tok0_d[q0:q1])
                    if band is not None:
                        snaps.setdefault(band, {})[lyr] = states[lyr].snapshot()

            need_logits = nll_n < nll_tokens
            res = tap.run(on_layer, lambda: forward(model, x, need_logits))
            if need_logits:
                for w in range(x.shape[0]):
                    nll = F.cross_entropy(res.logits[w, :-1].float(), x[w, 1:], reduction="none")
                    take = min(nll_tokens - nll_n, nll.numel())
                    nll_sum += float(nll[:take].sum())
                    nll_n += take
                    if nll_n >= nll_tokens:
                        break
            del res
            if device != "cpu":
                torch.cuda.synchronize()
            timing["harvest_loop_s"] += time.time() - t_b
            done_tokens = w1 * (window - 1)
            for band, by_layer in snaps.items():
                t = time.time()
                assert sorted(by_layer) == list(range(L)), f"band {band} snapshot missing layers"
                write_layers(out / "snapshots" / f"band{band}", by_layer, tokens)
                timing["snapshot_write_s"] += time.time() - t
            if bi == 0:
                first = []
                for p in selected[:FIRST_BATCH_POLES]:
                    lyr, n, s = parsed[p]
                    tk = states[lyr].topk.arrays()
                    first.append({"pole": p, "max_abs_after_first_batch": float(states[lyr].topk.max_abs()[n, s]),
                                  "contexts": pole_contexts(tk, n, s, tokens, offsets, decoder(tok),
                                                            FIRST_BATCH_CONTEXTS)})
                dump(out / "samples" / "first_batch.json", {"n_tokens": done_tokens, "poles": first})
            if (bi + 1) % METRICS_EVERY == 0:
                write_metrics(bi + 1, False)

        controls["nll_corpus_first_120k"] = {"mean_nll": nll_sum / nll_n, "n_tokens": nll_n,
                                             "rule": "targets input[:, 1:] from logits[:, :-1] per window, "
                                                     f"windows in order, first {nll_tokens} targets"}
        assert nll_n == nll_tokens, f"corpus NLL saw {nll_n} targets, needs {nll_tokens}"
        dump(out / "controls.json", controls)

        # ---- final arrays -----------------------------------------------------------------------
        t = time.time()
        final = write_layers(out, {lyr: st.snapshot() for lyr, st in enumerate(states)}, tokens)
        timing["final_write_s"] = time.time() - t
        dec = decoder(tok)
        with open(out / "samples" / "contexts.jsonl", "w") as f:
            for p in selected:
                lyr, n, s = parsed[p]
                topk, stats = final[lyr]
                f.write(json.dumps({
                    "pole": p, "max_abs": float(stats["max_abs"][n, s]), "q99": float(stats["q99"][n, s]),
                    "q999": float(stats["q999"][n, s]), "q9999": float(stats["q9999"][n, s]),
                    "density": int(stats["density"][n, s]), "sign_counts": stats["sign_counts"][n, s].tolist(),
                    "topk": pole_contexts(topk, n, s, tokens, offsets, dec)}) + "\n")
        dump(out / "samples" / "sample_draw.json", {"seed": seed, "rule": SAMPLE_RULE, "neurons": drawn,
                                                    "poles": selected, "from_poles_arg": from_poles_arg})

        # ---- poles/ for the pole list (given or drawn) ------------------------------------------
        pdir = out / "poles"
        dump(pdir / "poles.json", list(poles))
        ptok = np.full((len(poles), K_POLES), -1, np.int64)
        pa = np.full((len(poles), K_POLES), np.nan, np.float32)
        for j, p in enumerate(poles):
            lyr, n, s = parse_pole(p)
            arr = states[lyr].ptop.arrays()
            c = pole_neurons[lyr].index(n)
            ptok[j], pa[j] = arr["tok"][c, s], arr["a"][c, s]
        np.savez_compressed(pdir / "top1000.npz", tok=ptok, a=pa)
        if positions is None:
            # T_15 rule (DATA_CONTRACT §7): the poles' own final top-K, whose values are the exemplar record
            rows = [final[lyr][0] for lyr, _, _ in map(parse_pole, poles)]
            idx = [(n, s) for _, n, s in map(parse_pole, poles)]
            positions = np.stack([r["tok"][n, s] for r, (n, s) in zip(rows, idx)])
            ea, eg, eu = (np.stack([r[k][n, s] for r, (n, s) in zip(rows, idx)]) for k in ("a", "g", "u"))
        else:
            ea, eg, eu = (np.full(positions.shape, np.nan, np.float32) for _ in range(3))
            for st in states:
                if st.ex is not None:
                    assert bool(st.ex.filled.all()), "an exemplar position was never reached"
                    ea[st.ex.p, st.ex.e] = st.ex.a.cpu().numpy()
                    eg[st.ex.p, st.ex.e] = st.ex.g.cpu().numpy()
                    eu[st.ex.p, st.ex.e] = st.ex.u.cpu().numpy()
        np.savez_compressed(pdir / "positions.npz", tok=positions)
        np.savez_compressed(pdir / "exemplar_a.npz", a=ea, g=eg, u=eu)
        dump(pdir / "vocab.json", vocab_table(model, layers, tower.norm, tok, list(poles)))
        if probe_texts is not None:
            t = time.time()
            arrays, strings = probe_scores(model, tap, acts, tok, bos, probe_texts, list(poles), device)
            (pdir / "probe").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pdir / "probe" / "scores.npz", **arrays)
            dump(pdir / "probe" / "tokens.json", strings)
            timing["probe_s"] = time.time() - t

        write_metrics(n_batches, True)
    manifest.update({"tokens_per_second": n_scored / timing["harvest_loop_s"], "timing_s": timing,
                     "wall_s": time.time() - t_start, "finished_utc": utc()})
    dump(out / "manifest.json", manifest)
    return manifest
