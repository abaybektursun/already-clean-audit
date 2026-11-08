"""MENDELEEV N5 — family resolvers: decoder layers (the graft hook points), hook names, the head
tail, state-dict anatomy and key remaps, per model family. Contract: infra/pg/MATRIX_CONTRACT.md §8.

`resolve(model)` dispatches on the OUTER class name + `config.get_text_config().model_type`, then
ASSERTS every layer name against `dict(model.named_modules())` and `len(layers) ==
num_hidden_layers`. Every attribute path was read from the modeling source it names; the URL and
line numbers are next to each family entry (transformers v5.9.0 tag on GitHub; HF remote code for
Molmo2/MolmoAct2). Weight-free: `python -m infra.pg.family_resolvers` builds tiny stubs with the
same module layout and runs the dry-run gate.

Source base (transformers v5.9.0):
  T = https://raw.githubusercontent.com/huggingface/transformers/v5.9.0/src/transformers/models
  gemma      T/gemma/modeling_gemma.py       GemmaModel L375 (embed_tokens L381, layers L384, norm L387);
             GemmaDecoderLayer L303 (input_layernorm L312, post_attention_layernorm L313; returns a TENSOR
             L343); GemmaForCausalLM L451 (model L458, lm_head L460; tail L508-509: lm_head(h[:, -k:]),
             NO softcap).
  gemma2     T/gemma2/modeling_gemma2.py     Gemma2Model L392 (embed L398, layers L401, norm L404);
             Gemma2DecoderLayer L302 (norms L310-314: input/post_attention/pre_feedforward/
             post_feedforward; returns TENSOR L346); Gemma2ForCausalLM L477 (lm_head L486; tail L536-540:
             lm_head then tanh softcap by config.final_logit_softcapping).
  gemma3     T/gemma3/modeling_gemma3.py     Gemma3TextModel L489 (embed L499, layers L502, norm L505);
             Gemma3DecoderLayer L385 (same four norms; q_norm/k_norm inside self_attn L337-338; returns
             TENSOR L428); Gemma3ForCausalLM L586 (softcap L647-650); Gemma3Model L717
             (self.language_model = AutoModel.from_config(text_config) L727-728);
             Gemma3ForConditionalGeneration L879 (lm_head L888; tail L972-973: NO softcap in the shell).
  gemma4     T/gemma4/modeling_gemma4.py     Gemma4TextDecoderLayer L1353 (four norms L1361-1364,
             `layer_scalar` persistent buffer L1365, optional per-layer-input + MoE params L1367-1379;
             returns TENSOR L1439); Gemma4TextModel L1565 (embed L1580, layers L1583, norm L1586, optional
             embed_tokens_per_layer L1596); Gemma4ForCausalLM L1797 (lm_head L1808; softcap L1868-1871);
             Gemma4Model L2099 (language_model L2108-2109); Gemma4ForConditionalGeneration L2419 (lm_head
             L2427; softcap from get_text_config() L2506-2509 — the shell DOES softcap, unlike PaliGemma).
  paligemma  T/paligemma/modeling_paligemma.py PaliGemmaModel L121 (language_model =
             AutoModel.from_config(text_config) L131-132 → GemmaModel for PG1, Gemma2Model for PG2);
             PaliGemmaForConditionalGeneration L293 (lm_head L299 tied to model.language_model.embed_tokens
             L294; tail L366-367: lm_head only, NO softcap).
  qwen3      T/qwen3/modeling_qwen3.py       Qwen3Model L357 (embed L363, layers L364, norm L367);
             Qwen3DecoderLayer L294 (input_layernorm L303, post_attention_layernorm L304; returns TENSOR
             L334); q_norm/k_norm on head_dim inside self_attn L248-249; Qwen3ForCausalLM L442 (lm_head
             L451; tail L504-505).
  llama      T/llama/modeling_llama.py       LlamaModel L355 (embed L361, layers L362, norm L365);
             LlamaDecoderLayer L292 (returns TENSOR L332); LlamaForCausalLM L429 (lm_head L438; tail L486-487).
  mistral    T/mistral/modeling_mistral.py   MistralModel L328 (embed L334, layers L335, norm L338);
             MistralDecoderLayer L202 (returns TENSOR L240); MistralForCausalLM L402 (lm_head L411; tail L459-460).
  deepseek_v4 T/deepseek_v4/modeling_deepseek_v4.py  see DEEPSEEK_V4_NOTE below.
  molmo2     https://huggingface.co/allenai/Molmo2-4B/resolve/main/modeling_molmo2.py
             Molmo2DecoderLayer L751 (attn_norm L762, ff_norm L767; returns a TUPLE `(hidden_states,)`
             L811-815); Molmo2TextModel L925 (wte = Molmo2Embedding(vocab_size, additional_vocab_size)
             L931-934 → params `wte.embedding` [151936, d] and `wte.new_embedding` [128, d], concatenated at
             lookup L879-880; blocks L941; ln_f L944; forward iterates `blocks[: num_hidden_layers]` L1043,
             ln_f L1073); Molmo2Model L1114 (transformer L1124, vision_backbone L1125-1127);
             Molmo2ForConditionalGeneration L1567 (model L1577, lm_head L1578 with text vocab rows;
             `language_model` property → model.transformer L1597-1599; tail L1674-1676: lm_head only).
             Attention `att_proj` is the fused [q; k; v] (split order L684); MLP `ff_proj` chunks into
             (x, gate) with act(gate) * x L744-746 — so ff_proj = [up; gate].
  molmoact2  https://huggingface.co/allenai/MolmoAct2/resolve/main/modeling_molmoact2.py — the same text
             tower (MolmoAct2TextModel L2542, `blocks[: num_hidden_layers]` L2693) under model.transformer,
             plus `model.action_expert.*` (ActionExpert L534) which the LM prefix excludes by construction;
             text vocab 154624 + 128 new rows (config.json) vs Molmo2-ER 151936 + 128.

DEEPSEEK_V4_NOTE (T/deepseek_v4/modeling_deepseek_v4.py, transformers 5.9.0 — the HF repo
`deepseek-ai/DeepSeek-V4-Flash-0731` ships NO transformers remote code; it ships `inference/model.py`
(its own fp8/fp4 stack) and a config with `model_type: deepseek_v4`, `hc_mult: 4`, 43 layers, d 4096):
  * DeepseekV4Model L1237: embeds, then `hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1,
    hc_mult, -1)` L1292 — the residual stream is a TENSOR of shape [B, S, 4, D] for the whole stack.
  * DeepseekV4DecoderLayer L1091 (self_attn, mlp = DeepseekV4SparseMoeBlock, input_layernorm,
    post_attention_layernorm, attn_hc, ffn_hc L1105-1111). forward L1115-1141: each sublayer site runs a
    DeepseekV4HyperConnection (L864): `pre` collapses the 4 streams to one sequence (input-dependent
    sigmoid weights, L928/L939), the sublayer runs on the collapsed sequence, and the new 4-stream state is
    `post ⊗ sublayer_out + comb^T @ streams` with `comb` a per-token 4×4 doubly-stochastic (Sinkhorn) mixer
    L931-936, L1133-1141. `layers[L](...)` RETURNS ONE TENSOR [B, S, 4, D] (not a tuple).
  * Top of stack L1309: `hidden_states = self.norm(self.hc_head(hidden_states))` — DeepseekV4HyperHead
    L943 collapses 4 → 1 with input-dependent sigmoid weights, THEN the shared RMSNorm; lm_head L1470.
  * What the graft means here: replacing the layer-L output replaces ALL FOUR streams at once (one
    tensor). That is well-defined: the recipient's layer L+1 sees exactly what the donor's layer L+1
    would. What is NOT well-defined: (i) grafting one stream — `comb` remixes the streams at every
    sublayer, so a stream has no persistent identity across layers; (ii) a single "residual vector"
    mid-stack — the only canonical collapse is `hc_head` at the top, and its weights are input-dependent.
    Per-position cuts are fine (dim 1 is the sequence). The transplant anatomy: attn bucket = self_attn.*
    + input_layernorm + attn_hc.* (the HC feeding the attention site); mlp bucket = mlp.* (routed experts
    + gate/router + shared_experts — routers move WITH experts, confound 10) + post_attention_layernorm +
    ffn_hc.*; the head = hc_head + norm + lm_head (+ embed).
  * `DeepseekV4Model.layers` holds only the 43 main layers (`range(config.num_hidden_layers)` L1244);
    the `num_nextn_predict_layers` MTP layers are not in this ModuleList.
  * `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/config.json` declares `architectures:
    ["DeepseekV4ForCausalLM"]` plus `vision_*` keys and ships its own `inference/vision.py`; transformers'
    DeepseekV4ForCausalLM owns no vision tower, so under transformers only the text tower loads (vision
    keys unexpected). Row 4's forward shell is B2's (gated) problem; this resolver covers the text tower.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---- per-family anatomy (relative to the LM root; layer sub-keys relative to `<container>.<i>.`) ----
# attn/mlp: prefixes of layer sub-keys that belong to each bucket (the norms travel with the sublayer
# they wrap, as in carrel_core); embed: LM-root modules that are embeddings; head_extra: LM-root modules
# that belong to the head tail; skip: LM-root buffers that are neither (derived from config).
_GEMMA1 = dict(container="layers", norm="norm", embed=("embed_tokens",),
               attn=("self_attn.", "input_layernorm."), mlp=("mlp.", "post_attention_layernorm."),
               head_extra=(), skip=("rotary_emb",), n_streams=1)
_GEMMA2 = dict(_GEMMA1, attn=("self_attn.", "input_layernorm.", "post_attention_layernorm."),
               mlp=("mlp.", "pre_feedforward_layernorm.", "post_feedforward_layernorm."))
_GEMMA4 = dict(_GEMMA2,
               mlp=_GEMMA2["mlp"] + ("layer_scalar", "per_layer_input_gate.", "per_layer_projection.",
                                     "post_per_layer_input_norm.", "router.", "experts.",
                                     "pre_feedforward_layernorm_2.", "post_feedforward_layernorm_1.",
                                     "post_feedforward_layernorm_2."),
               embed=("embed_tokens", "embed_tokens_per_layer", "per_layer_model_projection",
                      "per_layer_projection_norm"))
_LLAMA = dict(_GEMMA1)            # qwen3 / llama / mistral: q_norm,k_norm live under self_attn.
_MOLMO2 = dict(container="blocks", norm="ln_f", embed=("wte",), attn=("self_attn.", "attn_norm."),
               mlp=("mlp.", "ff_norm."), head_extra=(), skip=("rotary_emb", "rotary_embs", "emb_drop"),
               n_streams=1)
_DSV4 = dict(container="layers", norm="norm", embed=("embed_tokens",),
             attn=("self_attn.", "input_layernorm.", "attn_hc."),
             mlp=("mlp.", "post_attention_layernorm.", "ffn_hc."),
             head_extra=("hc_head",), skip=("rotary_emb",), n_streams=4)

FAMILIES = {"gemma": _GEMMA1, "gemma2": _GEMMA2, "gemma3_text": _GEMMA2, "gemma4_text": _GEMMA4,
            "qwen3": _LLAMA, "llama": _LLAMA, "mistral": _LLAMA,
            "molmo2_text": _MOLMO2, "molmoact2_text": _MOLMO2, "deepseek_v4": _DSV4}
# outer classes whose forward applies tanh soft-capping AFTER lm_head (cited above)
SOFTCAP_CLASSES = {"Gemma2ForCausalLM", "Gemma3ForCausalLM", "Gemma4ForCausalLM",
                   "Gemma4ForConditionalGeneration"}
# where the decoder root lives, tried in order (shells first, then bare text models)
_ROOT_CANDIDATES = ("model.language_model", "model.transformer", "model", "language_model", "transformer")
_TAIL_RE = re.compile(r"(?:^|\.)((?:layers|blocks)\.\d+\..+|(?:norm|ln_f)\..+|embed_tokens(?:_per_layer)?\..+"
                      r"|wte\..+|hc_head\..+|per_layer_model_projection\..+|per_layer_projection_norm\..+)$")


@dataclass
class Resolver:
    family: str
    outer_class: str
    layers: list
    layer_names: list
    final_norm: object
    lm_head: object
    embed: object
    pre_head: object            # DeepSeek-V4: hc_head (4 → 1 collapse) applied before the final norm
    lm_prefix: str              # state-dict prefix of the decoder root, e.g. "model.language_model."
    lm_head_key: str
    n_layers: int
    n_streams: int
    softcap: float | None
    vocab_prefix: int | None    # Molmo2: rows shared with the Qwen3 parent (text vocab_size)
    spec: dict = field(repr=False)
    keys: frozenset = field(repr=False)

    # ---- state-dict anatomy ----
    def rel(self, key: str):
        """LM-root-relative key, or None when the key is outside the decoder root."""
        return key[len(self.lm_prefix):] if key.startswith(self.lm_prefix) else None

    def classify(self, key: str):
        """('lm_head',) | ('other',) | ('layer', i, 'attn'|'mlp') | ('final_norm',) | ('embed',) | ('skip',).
        Fails loud on an unknown key under the LM prefix (anatomy drift)."""
        if key == self.lm_head_key:
            return ("lm_head",)
        rel = self.rel(key)
        if rel is None:
            return ("other",)
        s = self.spec
        if rel.startswith(s["container"] + "."):
            i, sub = rel[len(s["container"]) + 1:].split(".", 1)
            if sub.startswith(s["attn"]):
                return ("layer", int(i), "attn")
            if sub.startswith(s["mlp"]):
                return ("layer", int(i), "mlp")
            raise AssertionError(f"unknown layer sub-key (anatomy drift) in {self.family}: {key}")
        if rel.startswith(s["norm"] + "."):
            return ("final_norm",)
        if rel.startswith(tuple(e + "." for e in s["embed"])):
            return ("embed",)
        if rel.startswith(tuple(h + "." for h in s["head_extra"])):
            return ("final_norm",)          # travels with the head tail (DeepSeek hc_head)
        if rel.startswith(s["skip"]):
            return ("skip",)
        raise AssertionError(f"unknown LM key (anatomy drift) in {self.family}: {key}")

    def key_remap(self, donor_key: str) -> str:
        """Donor state-dict key (same family, any shell/prefix) → this model's key. Asserts the target exists."""
        if donor_key.endswith("lm_head.weight"):
            return self.lm_head_key
        m = _TAIL_RE.search(donor_key)
        assert m, f"cannot find the LM tail of donor key {donor_key!r}"
        k = self.lm_prefix + m.group(1)
        assert k in self.keys, f"remapped key {k!r} (from {donor_key!r}) is not in the recipient state dict"
        return k

    # ---- the head tail (the DONOR's own readout on a captured layer output) ----
    def readout(self, residual):
        """Mirror the family's forward tail on a captured decoder-layer output: [pre_head] → final norm on the
        full sequence → slice the LAST position → lm_head → softcap. Returns [B, V] logits."""
        import torch
        h = residual
        if self.pre_head is not None:
            h = self.pre_head(h)
        h = self.final_norm(h)
        logits = self.lm_head(h[:, -1:, :])
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits[:, -1, :]


def _get(obj, dotted: str):
    for a in dotted.split("."):
        if not hasattr(obj, a):
            return None
        obj = getattr(obj, a)
    return obj


def resolve(model, lm_head=None) -> Resolver:
    """Resolve a loaded model (bare text model or VLM shell). `lm_head` overrides the head module when the
    Forward shell reads a different head (π0-in-PG1: its own tied head is primary — the caller passes it)."""
    cfg = model.config
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    mt = text_cfg.model_type
    assert mt in FAMILIES, f"no resolver for model_type {mt!r} (outer {type(model).__name__})"
    spec = FAMILIES[mt]
    root = root_name = None
    for cand in _ROOT_CANDIDATES:
        r = _get(model, cand)
        if r is not None and isinstance(_get(r, spec["container"]), __import__("torch").nn.ModuleList):
            root, root_name = r, cand
            break
    assert root is not None, f"no decoder root with `.{spec['container']}` under {type(model).__name__}"
    named = dict(model.named_modules())
    assert named.get(root_name) is root, root_name
    layers = list(getattr(root, spec["container"]))
    n = text_cfg.num_hidden_layers
    assert len(layers) == n, f"{len(layers)} layers != config.num_hidden_layers {n}"
    names = [f"{root_name}.{spec['container']}.{i}" for i in range(n)]
    for nm, ly in zip(names, layers):
        assert named.get(nm) is ly, f"hook name {nm!r} is not in named_modules() / not the same module"
    head = lm_head if lm_head is not None else _get(model, "lm_head")
    assert head is not None, "no lm_head on the outer model"
    keys = frozenset(model.state_dict().keys())
    lm_head_key = "lm_head.weight" if lm_head is None else next(
        (k for k, v in model.named_parameters() if v is head.weight), "lm_head.weight")
    assert lm_head_key in keys, lm_head_key
    embed = _get(root, spec["embed"][0])
    assert embed is not None, spec["embed"][0]
    pre_head = _get(root, spec["head_extra"][0]) if spec["head_extra"] else None
    if spec["head_extra"]:
        assert pre_head is not None, spec["head_extra"][0]
    softcap = text_cfg.final_logit_softcapping if type(model).__name__ in SOFTCAP_CLASSES else None
    vocab_prefix = text_cfg.vocab_size if mt in ("molmo2_text", "molmoact2_text") else None
    res = Resolver(family=mt, outer_class=type(model).__name__, layers=layers, layer_names=names,
                   final_norm=getattr(root, spec["norm"]), lm_head=head, embed=embed, pre_head=pre_head,
                   lm_prefix=root_name + ".", lm_head_key=lm_head_key, n_layers=n,
                   n_streams=spec["n_streams"], softcap=softcap, vocab_prefix=vocab_prefix, spec=spec,
                   keys=keys)
    for k in keys:                       # anatomy must be fully known before any transplant
        res.classify(k)
    return res


# ---- transplant bookkeeping (carrel_core.partition_keys / config_keys generalised by the resolver) ----
def partition(res: Resolver, keys=None) -> dict:
    """{'layers': {i: {'attn': [...], 'mlp': [...]}}, 'final_norm': [...], 'embed': [...], 'lm_head': [...],
    'skip': [...], 'other': [...]}; every layer must have ≥ 1 attn and ≥ 1 mlp key."""
    part = {"layers": {i: {"attn": [], "mlp": []} for i in range(res.n_layers)},
            "final_norm": [], "embed": [], "lm_head": [], "skip": [], "other": []}
    for k in sorted(keys if keys is not None else res.keys):
        c = res.classify(k)
        if c[0] == "layer":
            part["layers"][c[1]][c[2]].append(k)
        else:
            part[c[0]].append(k)
    for i, d in part["layers"].items():
        assert d["attn"] and d["mlp"], f"layer {i} incomplete: {d}"
    assert len(part["final_norm"]) >= 1 and len(part["embed"]) >= 1 and len(part["lm_head"]) == 1, \
        {k: len(v) for k, v in part.items() if k != "layers"}
    return part


def config_keys(part: dict, spec: dict) -> list:
    """spec = {'layers': [...], 'modules': ('attn','mlp'), 'final_norm': bool, 'embed': bool}; embed
    means embed + lm_head (tied models share storage, so both always move together)."""
    ks = []
    for l in spec["layers"]:
        for mod in spec["modules"]:
            ks += part["layers"][l][mod]
    if spec.get("final_norm"):
        ks += part["final_norm"]
    if spec.get("embed"):
        ks += part["embed"] + part["lm_head"]
    assert len(ks) == len(set(ks))
    return sorted(ks)


def lm_keys(part: dict) -> list:
    ks = []
    for l in sorted(part["layers"]):
        ks += part["layers"][l]["attn"] + part["layers"][l]["mlp"]
    return sorted(ks + part["final_norm"] + part["embed"] + part["lm_head"])


def qwen3_to_molmo2(sd: dict) -> dict:
    """Qwen3 decoder state dict (keys relative to the Qwen3Model root, plus 'lm_head.weight') → Molmo2 text
    tower keys (relative to model.transformer, plus 'lm_head.weight'). Tensor fusions per the Molmo2
    source: att_proj = cat[q; k; v] (split order L684), ff_proj = cat[up; gate] (chunk → (x, gate), act(gate)*x
    L744-746). embed_tokens → wte.embedding (the shared 151 936-row prefix; new_embedding is Molmo2's own)."""
    import torch
    out = {}
    rename = {"input_layernorm": "attn_norm", "post_attention_layernorm": "ff_norm",
              "self_attn.o_proj": "self_attn.attn_out", "self_attn.q_norm": "self_attn.q_norm",
              "self_attn.k_norm": "self_attn.k_norm", "mlp.down_proj": "mlp.ff_out"}
    by_layer = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            out[k] = v
        elif k == "norm.weight":
            out["ln_f.weight"] = v
        elif k == "embed_tokens.weight":
            out["wte.embedding"] = v
        else:
            m = re.match(r"^layers\.(\d+)\.(.+)\.weight$", k)
            assert m, f"unexpected Qwen3 key {k}"
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = v
    for i, d in by_layer.items():
        p = f"blocks.{i}."
        out[p + "self_attn.att_proj.weight"] = torch.cat(
            [d.pop("self_attn.q_proj"), d.pop("self_attn.k_proj"), d.pop("self_attn.v_proj")], 0)
        out[p + "mlp.ff_proj.weight"] = torch.cat([d.pop("mlp.up_proj"), d.pop("mlp.gate_proj")], 0)
        for sub, v in d.items():
            assert sub in rename, f"unexpected Qwen3 layer sub-key {sub}"
            out[p + rename[sub] + ".weight"] = v
    return out


# ---- weight-free stubs with the real module layouts (used by every MENDELEEV graft dry-run) ----
class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get_text_config(self):
        return self


def _stub_modules():
    import torch
    from torch import nn

    class RMS(nn.Module):
        def __init__(self, d):
            super().__init__(); self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

    class Layer(nn.Module):
        """Causal cumulative-mean 'attention' + gated MLP; residual output (tensor, or a tuple when
        `as_tuple`, mirroring Molmo2's `(hidden_states,)`)."""

        def __init__(self, d, as_tuple=False, streams=1, norms=("input_layernorm", "post_attention_layernorm")):
            super().__init__()
            self.self_attn = nn.Linear(d, d, bias=False); self.mlp = nn.Linear(d, d, bias=False)
            self.norm_names = norms                     # Molmo2 names its norms attn_norm / ff_norm
            setattr(self, norms[0], RMS(d)); setattr(self, norms[1], RMS(d))
            self.as_tuple, self.streams = as_tuple, streams

        def forward(self, h, attention_mask=None, **kw):
            n1, n2 = (getattr(self, n) for n in self.norm_names)
            x = h.mean(2) if self.streams > 1 else h
            m = (torch.ones(x.shape[:2], device=x.device) if attention_mask is None else attention_mask).to(x.dtype)[..., None]
            a = self.self_attn(n1(x)) * m
            x = x + a.cumsum(1) / m.cumsum(1).clamp(min=1)
            x = x + torch.tanh(self.mlp(n2(x)))
            if self.streams > 1:
                h = h + (x - h.mean(2))[:, :, None, :] * torch.linspace(0.5, 1.5, self.streams, device=x.device)[None, None, :, None]
                return h
            return (x,) if self.as_tuple else x

    class Head(nn.Module):           # DeepSeek-style 4 → 1 collapse
        def forward(self, h):
            return h.mean(2)

    class Out:
        def __init__(self, logits):
            self.logits = logits

    return nn, RMS, Layer, Head, Out


def stub_model(layout="llama", n_layers=2, d=16, vocab=32, seed=0):
    """A tiny model with the REAL module layout of a family: 'llama' (model.layers/norm/embed_tokens +
    lm_head, model_type llama), 'molmo2' (model.transformer.blocks/ln_f/wte.{embedding,new_embedding},
    tuple-returning blocks), 'deepseek_v4' (4-stream [B,S,4,D] state + hc_head)."""
    import torch
    nn, RMS, Layer, Head, Out = _stub_modules()
    torch.manual_seed(seed)
    streams = 4 if layout == "deepseek_v4" else 1

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            cont = "blocks" if layout == "molmo2" else "layers"
            norms = ("attn_norm", "ff_norm") if layout == "molmo2" else ("input_layernorm", "post_attention_layernorm")
            setattr(self, cont, nn.ModuleList([Layer(d, as_tuple=(layout == "molmo2"), streams=streams, norms=norms)
                                               for _ in range(n_layers)]))
            if layout == "molmo2":
                self.wte = nn.Module(); self.wte.embedding = nn.Parameter(torch.randn(vocab, d))
                self.wte.new_embedding = nn.Parameter(torch.randn(4, d)); self.ln_f = RMS(d)
            else:
                self.embed_tokens = nn.Embedding(vocab, d); self.norm = RMS(d)
            if streams > 1:
                self.hc_head = Head()

        def embed_fn(self, ids):
            if layout == "molmo2":
                return torch.nn.functional.embedding(ids, torch.cat([self.wte.embedding, self.wte.new_embedding], 0))
            return self.embed_tokens(ids)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module(); self.lm_head = nn.Linear(d, vocab, bias=False)
            root = Root()
            if layout == "molmo2":
                self.model.transformer = root
            elif layout == "llama":
                self.model = root
            else:
                self.model = root
            mt = {"llama": "llama", "molmo2": "molmo2_text", "deepseek_v4": "deepseek_v4"}[layout]
            self.config = _Cfg(model_type=mt, num_hidden_layers=n_layers, hidden_size=d, vocab_size=vocab,
                               final_logit_softcapping=None)

        def _root(self):
            return self.model.transformer if layout == "molmo2" else self.model

        def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=1, **kw):
            r = self._root()
            h = r.embed_fn(input_ids)
            if streams > 1:
                h = h.unsqueeze(2).expand(-1, -1, streams, -1).contiguous()
            for ly in (r.blocks if layout == "molmo2" else r.layers):
                o = ly(h, attention_mask=attention_mask)
                h = o[0] if isinstance(o, tuple) else o
            if streams > 1:
                h = r.hc_head(h)
            h = (r.ln_f if layout == "molmo2" else r.norm)(h)
            return Out(self.lm_head(h[:, -logits_to_keep:, :]))

    return Outer().eval()


class StubForward:
    """Contract-shaped Forward (§3) over a stub: char-code tokens, left padding, last real token,
    logits_to_keep=1. `model` is swappable (the graft job swaps a PatchedLM in)."""

    def __init__(self, model, vocab=32, family="stub", shell="text", device="cpu"):
        self.model, self.vocab, self.batch_size, self.device = model, vocab, 4, device
        self.n_layers, self.family, self.shell = model.config.num_hidden_layers, family, shell

    def _encode(self, prompts):
        import torch
        ids = [[5 + (ord(c) % (self.vocab - 5)) for c in p][-24:] for p in prompts]
        T = max(len(x) for x in ids)
        inp = torch.tensor([[0] * (T - len(x)) + x for x in ids])
        am = torch.tensor([[0] * (T - len(x)) + [1] * len(x) for x in ids])
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        return inp, am, pos

    def next_token_logits(self, prompts):
        import torch
        inp, am, pos = self._encode(prompts)
        with torch.inference_mode():
            return self.model(input_ids=inp, attention_mask=am, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()

    def letter_ids(self, letters=("A", "B", "C", "D"), space=True):
        ids = [5 + (ord(l) % (self.vocab - 5)) for l in letters]
        assert len(set(ids)) == 4
        return ids


def _dryrun():
    import torch
    for layout, family, container, norm, embed_leaf, n_streams in (
            ("llama", "llama", "layers", "norm", "embed_tokens", 1),
            ("molmo2", "molmo2_text", "blocks", "ln_f", "wte", 1),
            ("deepseek_v4", "deepseek_v4", "layers", "norm", "embed_tokens", 4)):
        m = stub_model(layout, n_layers=2)
        r = resolve(m)
        assert r.family == family and r.n_layers == 2 and r.n_streams == n_streams, r
        assert r.layer_names == [f"{r.lm_prefix}{container}.0", f"{r.lm_prefix}{container}.1"], r.layer_names
        assert r.rel(r.lm_prefix + norm + ".weight") == norm + ".weight"
        part = partition(r)
        assert set(part["layers"]) == {0, 1} and part["lm_head"] == ["lm_head.weight"], part
        assert all(k.startswith(r.lm_prefix + embed_leaf) for k in part["embed"]), part["embed"]
        assert len(config_keys(part, {"layers": [1], "modules": ("attn",)})) == 2      # self_attn + input_layernorm
        assert len(config_keys(part, {"layers": [1], "modules": ("mlp",)})) == 2       # mlp + post_attention_layernorm
        assert set(config_keys(part, {"layers": [0, 1], "modules": ("attn", "mlp"), "final_norm": True, "embed": True})) == set(lm_keys(part))
        # key remap from a bare-prefix donor of the same family
        k = r.key_remap(f"anything.{container}.1.self_attn.weight")
        assert k == f"{r.lm_prefix}{container}.1.self_attn.weight" and k in r.keys
        assert r.key_remap("language_model.lm_head.weight") == "lm_head.weight"
        # the readout on the LAST layer's captured output reproduces the model's own logits exactly
        fwd = StubForward(m)
        cap = {}
        h = r.layers[-1].register_forward_hook(lambda mod, a, o: cap.__setitem__("v", o[0] if isinstance(o, tuple) else o))
        ref = fwd.next_token_logits(["hello world", "abc"]); h.remove()
        got = r.readout(cap["v"]).float()
        assert torch.equal(got, ref), (layout, (got - ref).abs().max())
        # a WRONG hook point (the attention sub-module) does not reproduce the logits — the positive hook test bites
        cap = {}
        h = r.layers[-1].self_attn.register_forward_hook(lambda mod, a, o: cap.__setitem__("v", o))
        fwd.next_token_logits(["hello world", "abc"]); h.remove()
        if n_streams == 1:
            assert (r.readout(cap["v"]).float() - ref).abs().max() > 1e-2, layout
        # anatomy drift fails loud
        try:
            r.classify(f"{r.lm_prefix}{container}.0.mystery.weight"); raise SystemExit("no raise")
        except AssertionError:
            pass
        try:
            r.classify(f"{r.lm_prefix}unknown_thing.weight"); raise SystemExit("no raise")
        except AssertionError:
            pass
    # Molmo2 vocab prefix + DeepSeek 4-stream shape
    assert resolve(stub_model("molmo2")).vocab_prefix == 32
    md = stub_model("deepseek_v4"); rd = resolve(md); cap = {}
    h = rd.layers[0].register_forward_hook(lambda mod, a, o: cap.__setitem__("v", o))
    StubForward(md).next_token_logits(["xy"]); h.remove()
    assert cap["v"].shape == (1, 2, 4, 16) and rd.pre_head is not None
    # Qwen3 → Molmo2 fusion: round-trip the fused tensors
    d, ff = 8, 12
    q, k, v = (torch.randn(d, d) for _ in range(3)); up, gate, down = torch.randn(ff, d), torch.randn(ff, d), torch.randn(d, ff)
    sd = {"layers.0.self_attn.q_proj.weight": q, "layers.0.self_attn.k_proj.weight": k, "layers.0.self_attn.v_proj.weight": v,
          "layers.0.self_attn.o_proj.weight": torch.eye(d), "layers.0.self_attn.q_norm.weight": torch.ones(2),
          "layers.0.self_attn.k_norm.weight": torch.ones(2), "layers.0.mlp.gate_proj.weight": gate,
          "layers.0.mlp.up_proj.weight": up, "layers.0.mlp.down_proj.weight": down,
          "layers.0.input_layernorm.weight": torch.ones(d), "layers.0.post_attention_layernorm.weight": torch.ones(d),
          "norm.weight": torch.ones(d), "embed_tokens.weight": torch.randn(5, d), "lm_head.weight": torch.randn(5, d)}
    out = qwen3_to_molmo2(sd)
    qq, kk, vv = out["blocks.0.self_attn.att_proj.weight"].split([d, d, d], 0)
    assert torch.equal(qq, q) and torch.equal(kk, k) and torch.equal(vv, v)
    x_half, g_half = out["blocks.0.mlp.ff_proj.weight"].chunk(2, 0)
    assert torch.equal(x_half, up) and torch.equal(g_half, gate) and torch.equal(out["blocks.0.mlp.ff_out.weight"], down)
    assert set(out) == {"blocks.0.self_attn.att_proj.weight", "blocks.0.mlp.ff_proj.weight", "blocks.0.self_attn.attn_out.weight",
                        "blocks.0.self_attn.q_norm.weight", "blocks.0.self_attn.k_norm.weight", "blocks.0.mlp.ff_out.weight",
                        "blocks.0.attn_norm.weight", "blocks.0.ff_norm.weight", "ln_f.weight", "wte.embedding", "lm_head.weight"}, sorted(out)
    print("family_resolvers dry-run: ALL PASS")


if __name__ == "__main__":
    _dryrun()
