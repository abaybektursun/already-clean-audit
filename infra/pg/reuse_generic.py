"""SPEC6 GENERIC instrument additions (studies/reuse/SPEC6_generic.md §2, §3, §5). reuse_ablate, reuse_birth and unite_route are imported unchanged
(reuse_birth.py is SPEC4's frozen module, so the additions live here).

Held scale. OLMo 3's decoder layer adds post_feedforward_layernorm(mlp(x)) to the residual stream; the norm is an RMSNorm computed in fp32 and cast
to the input dtype: out = (weight * (y.float() * rsqrt(mean(y.float()^2, -1) + eps))).to(y.dtype) (transformers Olmo3RMSNorm.forward). held_scale
puts a forward hook on layers[l].post_feedforward_layernorm for every l in `ablated_layers`. record(): during the clean forward each hook stores
inv_l = rsqrt(mean(y.float()^2, -1, keepdim) + eps) [B, T, 1] per batch. hold(layers): during an ablated forward the hooks of `layers` replace the
norm's output at positions >= start by (weight * (y_ablated.float() * inv_recorded)).to(y.dtype), the batches replayed in recording order (the
readout batches a fixed context list in a fixed order); positions before start keep the model's own output. The recorded inv must have the
ablated input's batch and sequence shape (asserted). free(): the hooks pass the model's output through (SPEC4's condition). shift=1 in hold() is
the identity gate's failing input: layer l replays the inv recorded at layer l-1. With an empty set, y_ablated is the clean y bit for bit and the
held output is the model's output bit for bit (same operations in the same order).
norm_probe: an independent check of the held output (the completeness gate). record() stores its own inv per layer and batch from the clean
forward; compare(layers) recomputes, per position >= start, the expected (weight.float() * (y.float() * inv)) in fp32, casts it to the output dtype
as the norm does, and keeps max over positions of max|out - expected| / max|expected| per layer. Registered after held_scale, it sees the held output.
Cuts over the upstream layers 0..n_up-1 of ā: "g1" B_t (reuse_birth.high_background, 3,412 on O7), "g3" the round(0.03 n) largest |ā| (10,237),
"pl1" the round(0.01 W) largest |ā| within each layer (110 per layer, 3,410), "none" no entry.
Arms from the unexcluded cross set D(O*): members_alone = its members in B_t, largest |ā| first; single_largest = its one member with the largest
|ā| (SPEC6 §6; it is in B_t whenever the set holds a B_t member).
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from infra.pg import reuse_ablate as RA
from infra.pg import reuse_birth as RB
from infra.pg.neuron_rw_core import norm_eps

EXCLUSIONS = ("none", "g1", "g3", "pl1")
CUT_RULE = {"g1": (0.01, False), "g3": (0.03, False), "pl1": (0.01, True)}
SCALES = ("free", "held")
COMPLETE_TOL = 1e-4


def ff_norm(layer):
    """layers[l].post_feedforward_layernorm, asserted to be the RMSNorm the held scale assumes (a weight and an eps, no bias)."""
    norm = getattr(layer, "post_feedforward_layernorm")
    assert "RMSNorm" in type(norm).__name__ and norm.weight.ndim == 1 and getattr(norm, "bias", None) is None, (type(norm).__name__, "not an RMSNorm")
    return norm


class _Held:
    def __init__(self, layers, ablated_layers, start: int):
        self.layers, self.start = layers, start
        self.ablated = sorted(set(int(l) for l in ablated_layers))
        self.eps = {l: norm_eps(ff_norm(layers[l])) for l in self.ablated}
        self.mode, self.inv, self.i, self.held, self.shift = "free", {}, {}, set(), 0

    def record(self):
        self.mode, self.inv = "record", {l: [] for l in self.ablated}

    def hold(self, layers=None, shift: int = 0):
        held = set(self.ablated if layers is None else (int(l) for l in layers))
        assert held <= set(self.ablated), ("a held layer was not recorded", sorted(held - set(self.ablated)))
        assert shift in (0, 1), shift
        if shift:
            held = {l for l in held if l - shift in self.inv}
        assert self.inv and all(self.inv[l] for l in self.ablated), "hold() before a recorded clean forward"
        self.mode, self.held, self.shift, self.i = "hold", held, shift, {l: 0 for l in held}

    def free(self):
        self.mode, self.held = "free", set()

    def done(self) -> int:
        """After an ablated forward under hold(): every held layer replayed every recorded batch exactly once. Returns the number of held calls."""
        if self.mode != "hold":
            return 0
        for l in self.held:
            assert self.i[l] == len(self.inv[l - self.shift]), (l, self.i[l], len(self.inv[l - self.shift]), "held batches differ from recorded batches")
        return sum(self.i.values())

    def hook(self, l: int):
        def h(m, args, out):
            y = args[0]
            if self.mode == "record":
                self.inv[l].append(torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + self.eps[l]))
                return None
            if self.mode == "free" or l not in self.held:
                return None
            rec = self.inv[l - self.shift][self.i[l]]
            self.i[l] += 1
            assert rec.shape == (*y.shape[:2], 1), (l, tuple(rec.shape), tuple(y.shape), "the recorded inv is not of the ablated input's batch and sequence")
            held = (m.weight * (y.float() * rec)).to(y.dtype)
            new = out.clone()
            new[:, self.start:] = held[:, self.start:]
            return new
        return h


@contextlib.contextmanager
def held_scale(layers, ablated_layers, start: int):
    """The held-scale hooks on post_feedforward_layernorm of each layer in `ablated_layers`; yields the controller (record, hold, free, done)."""
    hs = _Held(layers, ablated_layers, start)
    handles = [ff_norm(layers[l]).register_forward_hook(hs.hook(l)) for l in hs.ablated]
    yield hs
    for h in handles:
        h.remove()


class _Probe:
    def __init__(self, layers, probe_layers, start: int):
        self.start = start
        self.probed = sorted(set(int(l) for l in probe_layers))
        self.eps = {l: norm_eps(ff_norm(layers[l])) for l in self.probed}
        self.mode, self.inv, self.i, self.on, self.rel = "off", {}, {}, set(), {}

    def record(self):
        self.mode, self.inv = "record", {l: [] for l in self.probed}

    def compare(self, layers):
        self.on = set(int(l) for l in layers) & set(self.probed)
        self.mode, self.i, self.rel = "compare", {l: 0 for l in self.on}, {l: 0.0 for l in self.on}

    def off(self):
        self.mode = "off"

    def result(self) -> dict:
        for l in self.on:
            assert self.i[l] == len(self.inv[l]), (l, self.i[l], len(self.inv[l]), "compared batches differ from recorded batches")
        return {"max": max(self.rel.values()) if self.rel else float("nan"), "by_layer": {int(l): float(v) for l, v in sorted(self.rel.items())}}

    def hook(self, l: int):
        def h(m, args, out):
            y = args[0]
            if self.mode == "record":
                self.inv[l].append(torch.rsqrt(y.float().pow(2).mean(-1, keepdim=True) + self.eps[l]))
            elif self.mode == "compare" and l in self.on:
                inv = self.inv[l][self.i[l]]
                self.i[l] += 1
                exp = m.weight.float() * (y.float() * inv)
                s = self.start
                err = (out[:, s:].float() - exp[:, s:].to(out.dtype).float()).abs().amax(-1)
                rel = err / exp[:, s:].abs().amax(-1).clamp_min(1e-30)
                self.rel[l] = max(self.rel[l], float(rel.max()))
            return None
        return h


@contextlib.contextmanager
def norm_probe(layers, probe_layers, start: int):
    """The completeness probe on post_feedforward_layernorm of each layer in `probe_layers`; register it after held_scale so it sees the held output."""
    pr = _Probe(layers, probe_layers, start)
    handles = [ff_norm(layers[l]).register_forward_hook(pr.hook(l)) for l in pr.probed]
    yield pr
    for h in handles:
        h.remove()


# ------------------------------------------------------------------------------------------ the cuts
def cut(abar: torch.Tensor, n_up: int, frac: float, per_layer: bool) -> dict:
    """A cut over the first n_up rows of ā: global (the round(frac n) largest |ā|, torch.topk) or per layer (the round(frac W) largest within each)."""
    assert abar.ndim == 2 and abar.shape[0] >= n_up, (tuple(abar.shape), n_up)
    x = abar[:n_up].float().abs()
    W = x.shape[1]
    if per_layer:
        k = int(round(frac * W))
        v, i = torch.topk(x, k, dim=1, sorted=True)
        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, i, True)
        cutv = v[:, -1].cpu().numpy().tolist()
    else:
        k = int(round(frac * x.numel()))
        v, i = torch.topk(x.reshape(-1), k, sorted=True)
        flat = torch.zeros(x.numel(), dtype=torch.bool, device=x.device)
        flat[i] = True
        mask = flat.view(n_up, W)
        cutv = float(v[-1])
    return {"mask": mask, "count": int(mask.sum()), "per_layer": mask.sum(1).cpu().numpy().astype(int).tolist(), "frac": frac, "per_layer_rule": per_layer,
            "cut": cutv, "n_entries": int(x.numel()), "n_up": n_up}


def masks(abar: torch.Tensor, n_up: int) -> dict:
    """{exclusion: cut dict}; "g1" is reuse_birth.high_background's B_t itself, "none" an empty mask."""
    hb = RB.high_background(abar, n_up)
    out = {"none": {"mask": torch.zeros_like(hb["mask"]), "count": 0, "per_layer": [0] * n_up, "frac": 0.0, "per_layer_rule": False, "cut": None,
                    "n_entries": hb["n_entries"], "n_up": n_up},
           "g1": {"mask": hb["mask"], "count": hb["count"], "per_layer": hb["per_layer"], "frac": hb["frac"], "per_layer_rule": False, "cut": hb["cut"],
                  "n_entries": hb["n_entries"], "n_up": n_up, "members": hb["members"]}}
    for e in ("g3", "pl1"):
        out[e] = cut(abar, n_up, *CUT_RULE[e])
    return out


# ------------------------------------------------------------------------------------------ the members arms
def members_alone(cross: list, mask, abar) -> list:
    """The members of the unexcluded cross set that are in B_t, largest |ā| first (set order on ties); empty when it holds none."""
    mk = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    mag = np.abs(np.asarray(abar.cpu() if torch.is_tensor(abar) else abar, dtype=np.float64))
    mem = [(l, k) for l, k in cross if mk[l, k]]
    return sorted(mem, key=lambda x: -mag[x[0], x[1]])


def single_largest(cross: list, abar) -> list:
    """The one member of the unexcluded cross set with the largest |ā| (the first in set order on ties)."""
    assert cross, "the cross set is empty"
    mag = np.abs(np.asarray(abar.cpu() if torch.is_tensor(abar) else abar, dtype=np.float64))
    i = int(np.argmax([mag[l, k] for l, k in cross]))
    return [cross[i]]


# ------------------------------------------------------------------------------------------ pure-logic checks on tiny tensors
def selfcheck() -> dict:
    from transformers.models.olmo3.modeling_olmo3 import Olmo3RMSNorm
    rng = np.random.default_rng(0)
    out = {"reuse_birth": RB.selfcheck()}
    d, B, T, start, eps = 16, 3, 7, 1, 1e-6

    class Layer(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            self.post_feedforward_layernorm = Olmo3RMSNorm(d, eps=eps)
            with torch.no_grad():
                self.post_feedforward_layernorm.weight.copy_(torch.tensor(np.random.default_rng(seed).uniform(0.5, 2.0, d)))
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 2 ** -7)):
        layers = [Layer(1).to(dtype), Layer(2).to(dtype)]
        norm = [lyr.post_feedforward_layernorm for lyr in layers]
        yc = [torch.tensor(rng.normal(0, 1, (B, T, d)), dtype=dtype) * (1 + 3 * i) for i in range(2)]
        ya = [y.clone() for y in yc]
        ya[0][:, :, 3] += 25.0                                    # the ablation moves one coordinate of layer 0's MLP output by a lot
        ya[1][:, :, 5] -= 0.5
        with torch.no_grad(), held_scale(layers, [0, 1], start) as hs:
            hs.record()
            clean = [n(y) for n, y in zip(norm, yc)]
            hs.hold()
            held = [n(y) for n, y in zip(norm, ya)]
            assert hs.done() == 2
            hs.free()
            free = [n(y) for n, y in zip(norm, ya)]
            hs.hold()                                              # the empty set: the ablated input is the clean input
            ident = [n(y) for n, y in zip(norm, yc)]
            hs.done()
            hs.hold(shift=1)                                       # the identity gate's failing input: layer 1 replays layer 0's inv
            shifted = norm[1](yc[1])
            hs.done()
        for i in range(2):
            y64, c64, w64 = ya[i].double(), yc[i].double(), norm[i].weight.detach().double()
            hand = w64 * y64 / torch.sqrt(c64.pow(2).mean(-1, keepdim=True) + eps)
            assert torch.equal(clean[i], norm[i](yc[i])), "record() changed the clean output"
            rel = float(((held[i][:, start:].double() - hand[:, start:]).abs().max() / hand[:, start:].abs().max()))
            assert rel <= tol, (str(dtype), i, rel, "the held norm differs from the hand computation")
            assert torch.equal(held[i][:, :start], free[i][:, :start]), "a position before the window start was not left to the model"
            assert torch.equal(ident[i], clean[i]), (str(dtype), i, "an empty set under held scale is not the identity")
        moved = float(((free[0][:, start:].double() - held[0][:, start:].double()).abs().max() / held[0][:, start:].double().abs().max()))
        assert moved > 0.1, (moved, "the constructed ablation did not move layer 0's divisor")
        assert not torch.equal(shifted, clean[1]), "the shifted inv reproduced the clean output"
        out[f"held_norm_{str(dtype).split('.')[-1]}"] = (f"held = weight * y_abl / sqrt(mean(y_clean^2) + eps) to {rel:.1e} (tol {tol:.0e}); positions < {start} left to the model; "
                                                         f"empty set exact; free differs by {moved:.2f}; shifted inv differs")
    # the shape assertion on the recorded inv
    layers = [Layer(1)]
    with torch.no_grad(), held_scale(layers, [0], start) as hs:
        hs.record()
        layers[0].post_feedforward_layernorm(torch.ones(B, T, d))
        hs.hold()
        raised = False
        try:
            layers[0].post_feedforward_layernorm(torch.ones(B, T + 1, d))
        except AssertionError:
            raised = True
    assert raised, "a recorded inv of another sequence length was replayed"
    out["held_shape"] = "a recorded inv of another batch shape stops the forward"
    # the probe: zero on the held output, nonzero on the free output
    layers = [Layer(1), Layer(2)]
    norm = [lyr.post_feedforward_layernorm for lyr in layers]
    yc = [torch.tensor(rng.normal(0, 1, (B, T, d)), dtype=torch.float32) for _ in range(2)]
    ya = [y + torch.tensor(rng.normal(0, 2, (B, T, d)), dtype=torch.float32) for y in yc]
    with torch.no_grad(), held_scale(layers, [0, 1], start) as hs, norm_probe(layers, [0, 1], start) as pr:
        hs.record()
        pr.record()
        [n(y) for n, y in zip(norm, yc)]
        hs.hold()
        pr.compare([0, 1])
        [n(y) for n, y in zip(norm, ya)]
        hs.done()
        r_held = pr.result()
        hs.free()
        pr.compare([0, 1])
        [n(y) for n, y in zip(norm, ya)]
        r_free = pr.result()
    assert r_held["max"] <= 1e-6 and r_free["max"] > COMPLETE_TOL, (r_held, r_free)
    out["norm_probe"] = f"held {r_held['max']:.1e}, free {r_free['max']:.2e} against {COMPLETE_TOL}"
    # the cuts
    n_up, W = 5, 200
    abar = torch.tensor(rng.standard_cauchy((n_up + 1, W)), dtype=torch.float32)
    mk = masks(abar, n_up)
    assert torch.equal(mk["g1"]["mask"], RB.high_background(abar, n_up)["mask"]) and not mk["none"]["mask"].any()
    mags = np.abs(abar[:n_up].numpy())
    g3 = mk["g3"]["mask"].numpy()
    assert g3.sum() == round(0.03 * n_up * W) and mags[g3].min() >= mags[~g3].max(), "g3 is not the top 3% of |ā|"
    assert (mk["g1"]["mask"].numpy() <= g3).all(), "B_t is not inside the 3% cut"
    pl = mk["pl1"]["mask"].numpy()
    for l in range(n_up):
        assert pl[l].sum() == round(0.01 * W) and mags[l][pl[l]].min() >= mags[l][~pl[l]].max(), f"layer {l}: the per-layer cut is not its top 1%"
    big = torch.tensor(rng.normal(0, 1, (32, 11008)), dtype=torch.float32)
    c3, cp = cut(big, 31, 0.03, False), cut(big, 31, 0.01, True)
    assert c3["count"] == 10237 and cp["count"] == 3410 and set(cp["per_layer"]) == {110}, (c3["count"], cp["count"])
    out["cuts"] = f"g1 = B_t; g3 {g3.sum()} = round(0.03 x {n_up * W}), the top |ā|, B_t inside; per layer round(0.01 x {W}) each; on 31 x 11008: 10,237 and 110 x 31 = 3,410"
    # the members arms
    m_ = np.zeros((3, 10), bool)
    m_[1, 4] = m_[2, 7] = True
    ab = np.zeros((3, 10))
    ab[1, 4], ab[2, 7], ab[0, 1], ab[2, 2] = 5.0, -9.0, 0.5, 2.0
    cross = [(0, 1), (1, 4), (2, 2), (2, 7)]
    assert members_alone(cross, m_, ab) == [(2, 7), (1, 4)] and single_largest(cross, ab) == [(2, 7)]
    assert members_alone([(0, 1), (2, 2)], m_, ab) == [] and single_largest([(0, 1), (2, 2)], ab) == [(2, 2)]
    out["members"] = "members in B_t, largest |ā| first; empty when none; the single largest |ā| member, also when the set holds no B_t member"
    return out
