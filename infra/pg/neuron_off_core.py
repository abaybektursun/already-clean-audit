"""NEURONOFF core: turn off one pole of one MLP neuron and measure what changes (studies/neuronoff/SCHEMA.md).

The object (as in neurons_harvest): a = act(g) * u, the input of `mlp.down_proj`, one coordinate per neuron.
A pole is (layer, neuron, sign); sign "+" means the pole fires where a_i > 0, "-" where a_i < 0.

`PoleAblator` is a forward pre-hook on the layer's `mlp.down_proj`. It copies the input to fp32, edits it, and
casts it back to the model dtype. Modes:
  pole      a_i := 0 where sign(a_i) matches the pole's sign; the other pole and every other coordinate unchanged.
  identity  the pole path on the pole's real firing mask, writing the original values (a_i := a_i there): the
            where-and-cast path runs exactly as for pole. Must reproduce the unhooked logits.
  floor     the pole path with an empty mask: a_i := a_i everywhere. Must give delta NLL exactly 0.
  mlp_zero  the whole down_proj input := 0, so the layer's MLP output is 0 (down_proj has no bias). Positive control.
  add       a_i := a_i + delta at every editable position (studies/reproductions/evilqwen): the residual stream gains
            delta * W_down[:, i]. delta 0 must reproduce the unhooked logits.
Positions below `exclude_first` (sequence positions, counted across the forwards of one `with` block, so a cached
generation step at position 200 is position 200) are never edited: the job passes BOS + window position 0, the
positions the harvest left out. After each forward the hook holds, for pole, identity and floor mode, `a` (the
clean a_i, fp32, [B, T]), `a_next` (the clean a of neuron i+1 mod width, for G-fit's failing input), `fire`
(sign(a_i) matches the pole's sign) and `edit` (the mask written); `n_edited` counts the written positions since
the `with` block opened.

Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib
import json
import math
import pathlib

import torch
import torch.nn.functional as F

from infra.pg.neurons_harvest import parse_pole, text_tower

MODES = ("pole", "identity", "floor", "mlp_zero", "add")
SIGNS = ("+", "-")
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 128


def pole_id(layer: int, neuron: int, sign: str) -> str:
    """The harvest's pole id, e.g. L07N10202-."""
    assert sign in SIGNS, f"sign must be '+' or '-', got {sign!r}"
    return f"L{layer:02d}N{neuron}{sign}"


def decoder_layers(model):
    return text_tower(model).layers


def bos_prefix(tok) -> list:
    """[BOS] when the tokenizer defines one, else []: prepended explicitly, never by add_special_tokens."""
    return [] if tok.bos_token_id is None else [tok.bos_token_id]


class PoleAblator:
    def __init__(self, model, mode: str, layer: int, neuron: int | None = None, sign: str | None = None,
                 exclude_first: int = 0, delta: float | None = None):
        assert mode in MODES, f"mode {mode!r} not in {MODES}"
        layers = decoder_layers(model)
        assert 0 <= layer < len(layers), f"layer {layer} out of range 0..{len(layers) - 1}"
        self.down = layers[layer].mlp.down_proj
        assert self.down.bias is None, "down_proj has a bias: a zero input would not zero the MLP output"
        if mode in ("pole", "identity", "floor"):
            assert sign in SIGNS, f"mode {mode} needs sign '+' or '-', got {sign!r}"
            assert 0 <= neuron < self.down.in_features, f"neuron {neuron} out of range 0..{self.down.in_features - 1}"
        if mode == "add":
            assert neuron is not None and 0 <= neuron < self.down.in_features, \
                f"neuron {neuron} out of range 0..{self.down.in_features - 1}"
            assert delta is not None and math.isfinite(delta), f"mode add needs a finite delta, got {delta!r}"
        self.mode, self.layer, self.neuron, self.sign, self.exclude_first = mode, layer, neuron, sign, exclude_first
        self.delta = delta
        self.a = self.a_next = self.fire = self.edit = None
        self.handle = None
        self.calls = self.offset = self.n_edited = 0

    def __enter__(self):
        assert self.handle is None, "hook already registered"
        self.offset = self.n_edited = 0
        self.handle = self.down.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        self.handle = None

    def _hook(self, module, args):
        x = args[0]
        B, T = x.shape[:2]
        xf = x.to(torch.float32, copy=True)
        editable = (self.offset + torch.arange(T, device=x.device) >= self.exclude_first).expand(B, T)
        if self.mode == "mlp_zero":
            xf[editable] = 0.0
            self.n_edited += int(editable.sum())
        elif self.mode in ("pole", "identity", "floor"):
            a = xf[..., self.neuron].clone()
            fire = a > 0 if self.sign == "+" else a < 0
            edit = fire & editable if self.mode != "floor" else torch.zeros_like(fire)
            value = a if self.mode == "identity" else torch.zeros_like(a)
            xf[..., self.neuron] = torch.where(edit, value, a)
            self.a, self.fire, self.edit = a, fire, edit
            self.a_next = xf[..., (self.neuron + 1) % xf.shape[-1]].clone()
            self.n_edited += int(edit.sum())
        elif self.mode == "add":
            xf[..., self.neuron] += editable.to(xf.dtype) * self.delta
            self.n_edited += int(editable.sum())
        self.offset += T
        self.calls += 1
        return (xf.to(x.dtype),) + tuple(args[1:])


# ------------------------------------------------------------------------------------------ teacher forcing
CHUNK = 1024      # positions per fp32 log-softmax (SCHEMA §2 memory rule: never more than 4,096 fp32 logit positions)


@torch.no_grad()
def next_token(model, ids: torch.Tensor, ablator: PoleAblator | None = None, keep_logits: bool = False,
               positions: list | None = None, targets: list | None = None) -> dict:
    """One teacher-forced forward, with `ablator` active when given.
    positions None: the prediction columns t = 0..T-2, column t scored against ids[:, t+1].
    positions (batch of 1) with targets: only those positions' logits are computed (`logits_to_keep` index tensor),
    position k scored against targets[k] (a token that need not be in ids).
    Returns nll [B, K] = -log p(target) in nats (fp32 log-softmax, CHUNK positions at a time); top5_id, top5_p;
    logits in the model's own dtype ([B, K, V], only with keep_logits); for a pole/floor ablator, the hook's fire,
    edit, a and a_next ([B, T], indexed by position)."""
    ids = ids.to(model.device)
    if positions is None:
        keep, tgt = 0, ids[:, 1:]
    else:
        assert ids.shape[0] == 1 and len(positions) == len(targets) > 0, "positions need a batch of 1 and one target each"
        keep = torch.tensor(positions, device=ids.device)
        tgt = torch.tensor([targets], device=ids.device)
    calls = ablator.calls if ablator is not None else 0
    with ablator if ablator is not None else contextlib.nullcontext():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=keep).logits
    if ablator is not None:
        assert ablator.calls == calls + 1, f"hook fired {ablator.calls - calls} times in one forward"
    if positions is None:
        logits = logits[:, :-1]
    nll, top_id, top_p = [], [], []
    for s in range(0, logits.shape[1], CHUNK):
        lp = F.log_softmax(logits[:, s:s + CHUNK].float(), -1)
        nll.append(-lp.gather(-1, tgt[:, s:s + CHUNK, None])[..., 0].cpu())
        v, i = lp.topk(5, -1)
        top_id.append(i.cpu())
        top_p.append(v.exp().cpu())
        del lp
    out = {"nll": torch.cat(nll, 1), "top5_id": torch.cat(top_id, 1), "top5_p": torch.cat(top_p, 1),
           "logits": logits if keep_logits else None, "fire": None, "edit": None, "a": None, "a_next": None}
    if ablator is not None and ablator.fire is not None:
        assert ablator.fire.shape == ids.shape, f"mask {tuple(ablator.fire.shape)} vs ids {tuple(ids.shape)}"
        for k in ("fire", "edit", "a", "a_next"):
            out[k] = getattr(ablator, k).cpu()
    return out


def max_abs_dlogit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """max over the vocabulary of |a - b| per position ([B, K]), in fp32, CHUNK positions at a time."""
    assert a.shape == b.shape, f"{tuple(a.shape)} vs {tuple(b.shape)}"
    return torch.cat([(a[:, s:s + CHUNK].float() - b[:, s:s + CHUNK].float()).abs().amax(-1).cpu()
                      for s in range(0, a.shape[1], CHUNK)], 1)


def nll_clean_hooked(model, ids: torch.Tensor, ablator: PoleAblator) -> dict:
    """Clean and hooked next-token NLL [B, T-1] on the same batch, the hook's fire/edit masks and clean a_i
    ([B, T]; None for identity and mlp_zero), and max |hooked - clean| logit per prediction column."""
    clean = next_token(model, ids, keep_logits=True)
    hooked = next_token(model, ids, ablator, keep_logits=True)
    return {"clean": clean["nll"], "hooked": hooked["nll"], "fire": hooked["fire"], "edit": hooked["edit"],
            "a": hooked["a"], "max_abs_dlogit": max_abs_dlogit(hooked["logits"], clean["logits"])}


# ------------------------------------------------------------------------------------------ Arm B
@torch.no_grad()
def teacher(model, tok, prompt: str, continuation: str, ablator: PoleAblator | None) -> dict:
    """Teacher-forced log-probabilities of `continuation` after `prompt` (SPEC §5.3, SCHEMA §3.4 fields). Input =
    BOS (when the tokenizer defines one, prepended explicitly) + tok(prompt) + continuation ids, every tok() call with
    add_special_tokens=False (a tokenizer that adds its own BOS would otherwise give two). Continuation ids =
    tok(prompt + continuation)[len(tok(prompt)):] when tok(prompt) is a strict prefix of it (prefix_property true);
    otherwise tok(continuation), scored the same way and flagged (the analysis excludes such records). cont_logits (model dtype, CPU) are returned for the caller's max |dlogit|
    and not written; a_inputs = the hook's clean a_i at the scored input positions (last prompt token and every
    continuation token but the last), None without a pole/floor hook."""
    p_txt = tok(prompt, add_special_tokens=False).input_ids
    f_txt = tok(prompt + continuation, add_special_tokens=False).input_ids
    prefix = len(f_txt) > len(p_txt) and f_txt[:len(p_txt)] == p_txt
    c_ids = f_txt[len(p_txt):] if prefix else tok(continuation, add_special_tokens=False).input_ids
    p_ids = bos_prefix(tok) + p_txt
    assert c_ids, f"continuation {continuation!r} has no tokens"
    ids = p_ids + c_ids
    cols = list(range(len(p_ids) - 1, len(ids) - 1))
    r = next_token(model, torch.tensor([ids]), ablator, keep_logits=True, positions=cols, targets=c_ids)
    lp = (-r["nll"][0]).tolist()
    return {"prompt_n_tokens": len(p_ids), "prefix_property": prefix,
            "cont_token_ids": c_ids, "cont_tokens": [tok.decode([i]) for i in c_ids],
            "first_token_str": tok.decode([c_ids[0]]), "logp_per_token": lp, "logp_sum": sum(lp), "logp_first": lp[0],
            "top5_end": [[tok.decode([int(i)]), float(p)] for i, p in zip(r["top5_id"][0, 0], r["top5_p"][0, 0])],
            "cont_logits": r["logits"][0].cpu(), "a_inputs": None if r["a"] is None else r["a"][0, cols].tolist()}


def generation_config(tok, temperature: float = TEMPERATURE, max_new_tokens: int = MAX_NEW_TOKENS, top_k: int = 0,
                      top_p: float = 1.0):
    """Every sampling knob set explicitly (SPEC §5.3), so the model's own generation_config defaults never apply."""
    from transformers import GenerationConfig
    eos = tok.eos_token_id
    return GenerationConfig(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=1.0,
                            max_new_tokens=max_new_tokens, eos_token_id=eos,
                            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else eos)


@torch.no_grad()
def sample(model, tok, prompt: str, ablator: PoleAblator | None, n: int, seed: int, temperature: float = TEMPERATURE,
           max_new_tokens: int = MAX_NEW_TOKENS, top_k: int = 0, top_p: float = 1.0) -> list:
    """n continuations of BOS + tok(prompt, no special tokens) from one fixed seed, the hook active for the whole
    generation; top_k 0 and
    top_p 1 are plain temperature sampling. The same seed in two conditions draws the same uniform numbers.
    Each item: completion (new tokens before the first EOS, decoded), completion_token_ids, n_new_tokens,
    stopped_eos."""
    ids = torch.tensor([bos_prefix(tok) + tok(prompt, add_special_tokens=False).input_ids], device=model.device)
    eos = tok.eos_token_id
    torch.manual_seed(seed)
    with ablator if ablator is not None else contextlib.nullcontext():
        out = model.generate(ids.repeat(n, 1), attention_mask=torch.ones_like(ids).repeat(n, 1),
                             generation_config=generation_config(tok, temperature, max_new_tokens, top_k, top_p))
    rows = []
    for o in out[:, ids.shape[1]:].tolist():
        stop = eos in o
        new = o[:o.index(eos)] if stop else o
        rows.append({"completion": tok.decode(new, skip_special_tokens=True), "completion_token_ids": new,
                     "n_new_tokens": len(new), "stopped_eos": stop})
    return rows


# ------------------------------------------------------------------------------------------ control poles
def load_classes(verdict_dir, model_key: str) -> dict:
    """{pole id: {layer, class}} from the verdict's per-pole records, <verdict_dir>/samples/q1_<model>_poles.jsonl.
    (The verdict metrics.json itself holds only aggregates; its per-pole rows live in samples/.)"""
    path = pathlib.Path(verdict_dir) / "samples" / f"q1_{model_key}_poles.jsonl"
    rows = [json.loads(x) for x in path.read_text().split("\n") if x.strip()]     # not splitlines(): U+2028 in records
    out = {r["pole"]: {"layer": int(r["layer"]), "class": r["class"]} for r in rows}
    assert len(out) == len(rows), f"{path}: duplicate poles"
    return out


def pole_features(stats_dir, poles: list, n_tokens_scored: int) -> dict:
    """{pole id: {density, meanabs, rate}} from a harvest's stats/layer_LL.npz (SPEC §4.2 as amended A5, as
    make_plan.py computes them): density = stats density (tokens with |a| >= 0.01 max|a|) / n_tokens_scored;
    meanabs = mean |a| over the pole's sign tokens from the log10|a| histogram at geometric bin centres; rate = the
    pole's sign-token count over the neuron's nonzero-a count (recorded, not matched on)."""
    import numpy as np
    out, cache = {}, {}
    for p in poles:
        lyr, n, s = parse_pole(p)
        if lyr not in cache:
            cache[lyr] = np.load(pathlib.Path(stats_dir) / f"layer_{lyr:02d}.npz")
        st = cache[lyr]
        h, e = st["hist"][n, s].astype(np.float64), st["hist_edges"].astype(np.float64)
        cnt = st["n_pole_tokens"][n]
        assert h.sum() == cnt[s] > 0, f"{p}: no tokens of its sign"
        out[p] = {"density": float(st["density"][n, s] / n_tokens_scored),
                  "meanabs": float((h * np.sqrt(e[:-1] * e[1:])).sum() / h.sum()), "rate": float(cnt[s] / cnt.sum())}
    return out


MATCH_VARS = ("density", "meanabs", "ho_med")


def match_rule(target: dict, candidates: list) -> dict:
    """The control-matching rule, a pure function (SPEC §4.2 distance, no caliper): the candidate nearest to `target`
    by d = max over the MATCH_VARS both carry of |log(x_target / x_candidate)|; a tie goes to the smaller pole id.
    Not the selection authority: plan/make_plan.py is (SPEC §9.1)."""
    assert candidates, f"no candidate for {target['pole']}"

    def dist(c):
        return max(abs(math.log(c[k] / target[k])) for k in MATCH_VARS if k in c and k in target)
    return min(candidates, key=lambda c: (dist(c), c["pole"]))


def match_controls(classes: dict, features: dict, rule=match_rule, ctrl_class: str = "UNK") -> list:
    """For each ONE pole, the rule's pick among the `ctrl_class` poles of the same layer (with replacement;
    SCHEMA §1.1 takes controls from class UNK). Returns [{test, control, n_candidates}] in pole-id order."""
    out = []
    for p in sorted(q for q, c in classes.items() if c["class"] == "ONE"):
        lyr = classes[p]["layer"]
        cands = [{"pole": q, **features[q]} for q, c in sorted(classes.items())
                 if c["layer"] == lyr and c["class"] == ctrl_class]
        c = rule({"pole": p, **features[p]}, cands)
        out.append({"test": {"pole": p, **features[p]}, "control": {**c, "class": classes[c["pole"]]["class"]},
                    "n_candidates": len(cands)})
    return out


if __name__ == "__main__":          # SPEC §9.6 names this module; the commands live in neuron_off_analysis
    from infra.pg.neuron_off_analysis import main
    main()
