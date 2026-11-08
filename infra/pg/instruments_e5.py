"""INSTRUMENTS E5 instrument additions (studies/instruments/SPEC_instruments.md, section "E5: the cause of Olmo-3's generic term"). reuse_ablate,
unite_route and instruments_adapter are imported unchanged.

T1, the input hold (a "virtual pre-norm" at Olmo 3's readers). Olmo 3's attention reads the raw residual h_l (q_proj, k_proj, v_proj take h_l itself;
modeling_olmo3 Olmo3Attention.forward) and its MLP reads h_l + post_attention_layernorm(attn) itself (gate_proj, up_proj; Olmo3MLP.forward). held_input
puts a forward pre-hook on each of the five readers of every layer given. record(): during the clean forward each hook stores rms(x) [B, T] fp32 of the
input it receives, per batch (rms = sqrt(mean(x.float()^2)) over the hidden dimension, no eps). hold(layers): during an ablated forward the hooks of
`layers` replace the input at window positions >= start by (x.float() * rms_clean / rms(x)).to(x.dtype), the batches replayed in recording order (the
readout batches a fixed context list in a fixed order); positions before start (the BOS) keep the model's input, and so does a zero input (a padding
position whose embedding is zero has no scale; 0/0 would put a NaN into every later attention); the residual stream itself is not changed, only
what each reader sees. The hold is of the INPUT scale: each position's reader input has the clean RMS exactly, so it cannot compound
(SPEC6's held output-norm divisor, reuse_generic._Held, replayed a clean divisor on a moved input and ran away). shift=1 is the identity gate's failing
input (the clean RMS of position t - 1 at position t, torch.roll over the sequence); fill=inf is the non-finite counter's failing input. On the clean
forward rms(x) is computed by the same operations on the same tensor, so the ratio is 1.0 exactly and the held clean forward is the clean forward bit
for bit. Per held forward the hooks count, per layer, the reader calls (5 per layer per batch) whose held input holds a non-finite value.
T2, the marked-token write. replaced_at is reuse_ablate.replaced with the value written at the marked position of each row only (the window's other
positions untouched); its post-hook asserts that down_proj received exactly the original input with the values at (row, marked position, idx).
readout_marked is reuse_ablate.readout with replaced_at in place of replaced (the same packing, order, hooks and assertions).
T5, the members' residual write. For member k of layer l: w_k(x) = |ā_k| ||d_k|| s_l(x) / rms(h_l(x)), d_k = gamma_post_l ⊙ W_down_l[:, k]
(instruments_adapter.Rows.column; gain 1 without a post-MLP norm), s_l(x) = 1 / rms of the post-MLP norm's input (unite_route's s; 1 on Qwen 3),
h_l(x) = embedding + sum_{l' <= l} (attention_l' + mlp_l') the residual stream after layer l as instruments_adapter.capture records its terms (the
composition the completeness gates check), all at the last position of each forwarded window.
Fail loud: no try/except, no fallback; asserts carry their reason.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from infra.pg import instruments_adapter as AD
from infra.pg import instruments_values as IV
from infra.pg import neurons_harvest as NH
from infra.pg import unite_route as UR

READERS = (("q", "self_attn", "q_proj"), ("k", "self_attn", "k_proj"), ("v", "self_attn", "v_proj"), ("g", "mlp", "gate_proj"), ("u", "mlp", "up_proj"))
NAMES = tuple(r for r, _, _ in READERS)
ID_TOL = 1e-4
NONFINITE_LINE = 0.05


def rms(x: torch.Tensor) -> torch.Tensor:
    """[..., d] -> [...]: sqrt(mean(x^2)) over the last dimension in fp32, no eps."""
    return x.float().pow(2).mean(-1).sqrt()


def reader(layer, r: str):
    _, a, b = next(x for x in READERS if x[0] == r)
    m = getattr(getattr(layer, a), b)
    assert isinstance(m, torch.nn.Linear), (r, type(m).__name__)
    return m


# ------------------------------------------------------------------------------------------ T1: the input hold
class _Hold:
    def __init__(self, layers, start: int):
        self.layers, self.start = layers, start
        self.mode, self.rec, self.i, self.held, self.shift, self.fill = "free", {}, {}, set(), 0, None
        self.nonfinite, self.last = {}, {}

    def record(self):
        self.mode, self.rec = "record", {}

    def hold(self, layers, shift: int = 0, fill: float | None = None):
        held = {int(l) for l in layers}
        assert held and all((l, r) in self.rec and self.rec[(l, r)] for l in held for r in NAMES), ("hold() before a recorded clean forward of these layers",
                                                                                                   sorted(held))
        assert shift in (0, 1), shift
        self.mode, self.held, self.shift, self.fill = "hold", held, shift, fill
        self.i = {(l, r): 0 for l in held for r in NAMES}
        self.nonfinite, self.last = {l: 0 for l in sorted(held)}, {}

    def free(self):
        self.mode, self.held = "free", set()

    def done(self) -> dict:
        """After a forward under hold(): every held reader replayed every recorded batch exactly once. Returns the per-layer non-finite reader-call counts."""
        assert self.mode == "hold", self.mode
        for key, n in self.i.items():
            assert n == len(self.rec[key]), (key, n, len(self.rec[key]), "held batches differ from recorded batches")
        out = dict(self.nonfinite)
        self.free()
        return out

    def hook(self, l: int, r: str):
        def pre(m, args):
            x = args[0]
            if self.mode == "record":
                self.rec.setdefault((l, r), []).append(rms(x))
                return None
            if self.mode == "free" or l not in self.held:
                return None
            c = self.rec[(l, r)][self.i[(l, r)]]
            self.i[(l, r)] += 1
            assert c.shape == x.shape[:2], (l, r, tuple(c.shape), tuple(x.shape), "the recorded rms is not of this input's batch and sequence")
            if self.shift:
                c = torch.roll(c, self.shift, 1)
            if self.fill is not None:
                c = torch.full_like(c, self.fill)
            have = rms(x)
            s = self.start
            ratio = torch.where(have > 0, c / have, torch.ones_like(have))     # a zero input (a padding row's zero embedding) has no scale: left as it is
            new = x.clone()
            new[:, s:] = (x[:, s:].float() * ratio[:, s:, None]).to(x.dtype)
            if not bool(torch.isfinite(new[:, s:]).all()):
                self.nonfinite[l] += 1
            self.last.setdefault((l, r), []).append((c, have))
            return (new, *args[1:])
        return pre


@contextlib.contextmanager
def held_input(layers, hold_layers, start: int):
    """Pre-hooks on q_proj, k_proj, v_proj, gate_proj and up_proj of each layer in `hold_layers`; yields the controller (record, hold, free, done)."""
    H = _Hold(layers, start)
    handles = [reader(layers[l], r).register_forward_pre_hook(H.hook(int(l), r)) for l in sorted(set(int(x) for x in hold_layers)) for r in NAMES]
    yield H
    for h in handles:
        h.remove()


def held_layers(S: list, L: int) -> list:
    """T1's layers for set S read at a unit of layer L: from the lowest ablated layer to L, inclusive (the identity arm: 0 to L)."""
    return list(range(min(l for l, _ in S) if S else 0, L + 1))


# ------------------------------------------------------------------------------------------ T2: the marked-token write
@contextlib.contextmanager
def replaced_at(layers, ablation: dict, pos: torch.Tensor):
    """reuse_ablate.replaced at the marked position of each row only: a pre-hook on layers[l].mlp.down_proj writes values[j] at (row, pos[row],
    idx[j]); the post-hook asserts down_proj received exactly the original input with those entries replaced (the written value there, every
    other position and neuron untouched) and counts the calls."""
    rows = torch.arange(len(pos), device=pos.device)
    rec, orig, handles = {"calls": 0}, {}, []
    for l, (idx, val) in ablation.items():
        def pre(m, args, l=l, idx=idx, val=val):
            orig[l] = args[0]
            a = args[0].clone()
            a[rows[:, None], pos[:, None], idx[None, :]] = val.to(a.dtype)[None, :]
            return (a, *args[1:])

        def post(m, args, out, l=l, idx=idx, val=val):
            want = orig.pop(l).clone()
            want[rows[:, None], pos[:, None], idx[None, :]] = val.to(want.dtype)[None, :]
            got = args[0][rows[:, None], pos[:, None], idx[None, :]]
            assert bool((got == val.to(got.dtype)[None, :]).all()), "down_proj did not receive the value at the marked position"
            assert torch.equal(args[0], want), "down_proj's input differs from the clean input away from the marked write"
            rec["calls"] += 1
        handles += [layers[l].mlp.down_proj.register_forward_pre_hook(pre), layers[l].mlp.down_proj.register_forward_hook(post)]
    yield rec
    for h in handles:
        h.remove()


@torch.inference_mode()
def readout_marked(model, toks, unit: dict, ablation: dict, *, tokens, length, bos, batch: int, hw: int, device) -> dict:
    """reuse_ablate.readout with the replacement at the marked token only (replaced_at); a [N], g, u [N, T] at the marked token, rows in `toks` order."""
    tower = NH.text_tower(model)
    layers = list(tower.layers)
    L = unit["layer"]
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
        with UR.upto_layer(tower, L), replaced_at(layers, ablation, pos) as rec:
            NH.forward(model, x, False)
        for h in hs:
            h.remove()
        assert rec["calls"] == len(ablation), f"{rec['calls']} marked-write checks for {len(ablation)} ablated layers"
        gs.append(keep["g"][rows, pos][:, ns].float())
        us.append(keep["u"][rows, pos][:, ns].float())
    inv = torch.as_tensor(np.argsort(order), device=device)
    g, u = torch.cat(gs)[inv], torch.cat(us)[inv]
    a = (ws * layers[L].mlp.act_fn(g) * u).sum(1)
    return {"a": a.cpu().numpy(), "g": g.cpu().numpy(), "u": u.cpu().numpy()}


# ------------------------------------------------------------------------------------------ T5: the members' residual write
def residual_rms(cap: dict, upto: int) -> torch.Tensor:
    """[N, upto + 1]: rms of h_l = embedding + sum_{l' <= l} (attention_l' + mlp_l') at each captured position, fp32."""
    h = cap["emb"].float().clone()
    out = []
    for l in range(upto + 1):
        h = h + cap["attn"][l].float() + cap["mlp"][l].float()
        out.append(rms(h))
    return torch.stack(out, 1)


@torch.inference_mode()
def layer_scales(ad, model, positions, n_up: int, **kw) -> dict:
    """One capture of `positions` through instruments_adapter.capture (layers 0..n_up-1): ā from it [n_up, W] (and the mean over the first half of
    the positions, the ā check's failing input), s [P, n_up], the residual rms [P, n_up]."""
    cap = AD.capture(ad, model, np.asarray(positions), n_up - 1, set(), **kw)
    h = len(positions) // 2
    return {"abar": torch.stack([cap["a"][l].mean(0) for l in range(n_up)]), "abar_half": torch.stack([cap["a"][l][:h].mean(0) for l in range(n_up)]),
            "s": torch.stack([cap["s"][l].float() for l in range(n_up)], 1), "rms": residual_rms(cap, n_up - 1), "n": int(len(positions))}


def abar_check(abar_cap, abar_half, abar_ref) -> dict:
    """The capture's mean against the banked ā in instruments_values' rule |d| <= 1e-5 (1 + |ā|) at every entry; failing input: the first half's mean."""
    ref = np.asarray(abar_ref, np.float64)
    ok = lambda x: bool((np.abs(np.asarray(x, np.float64) - ref) <= IV.MEAN_TOL * (1 + np.abs(ref))).all())
    mx = lambda x: float(np.abs(np.asarray(x, np.float64) - ref).max()) if ref.size else 0.0
    return {"pass": ok(abar_cap), "max_abs": mx(abar_cap), "tol": IV.MEAN_TOL, "n_entries": int(ref.size),
            "failing_input": {"input": "the mean over the first half of the positions", "max_abs": mx(abar_half)}, "failing_input_failed": not ok(abar_half),
            "rule": "|mean_P a - ā_banked| <= 1e-5 (1 + |ā|) at every entry"}


def column_norms(ad, l: int, ks) -> torch.Tensor:
    """||gamma_post_l ⊙ W_down_l[:, k]|| for each k in ks (fp32): instruments_adapter.Rows.column's columns, without caching the whole layer."""
    w = ad.layers[l].mlp.down_proj.weight[:, torch.as_tensor(list(ks), dtype=torch.long, device=ad.layers[l].mlp.down_proj.weight.device)].float()
    return (ad.post_gain(l)[:, None] * w).norm(dim=0)


def member_writes(dnorm, abar_m, layer_m, s: torch.Tensor, rr: torch.Tensor) -> np.ndarray:
    """[P, n]: |ā_k| ||d_k|| s_l(x) / rms(h_l(x)) for member j = (l, k) with ||d_k|| dnorm[j], ā_k abar_m[j], l layer_m[j]."""
    li = torch.as_tensor(np.asarray(layer_m, np.int64), device=s.device)
    d = torch.as_tensor(np.asarray(dnorm, np.float64), device=s.device)
    ab = torch.as_tensor(np.abs(np.asarray(abar_m, np.float64)), device=s.device)
    return (ab[None] * d[None] * s[:, li].double() / rr[:, li].double()).cpu().numpy()


def summary(write: np.ndarray) -> dict:
    """Per member the median over positions; over members their median and 90th percentile (numpy's linear rule)."""
    per = np.median(write, 0) if write.size else np.zeros(0)
    return {"n_members": int(per.size), "median": float(np.median(per)) if per.size else float("nan"),
            "p90": float(np.percentile(per, 90)) if per.size else float("nan"), "per_member_median": per.tolist()}


# ------------------------------------------------------------------------------------------ pure-logic checks on tiny tensors
def selfcheck() -> dict:
    rng = np.random.default_rng(0)
    out = {}
    d, B, T, start = 16, 3, 9, 1

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj = (torch.nn.Linear(d, d, bias=False) for _ in range(3))

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj, self.up_proj = torch.nn.Linear(d, 8, bias=False), torch.nn.Linear(d, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, d, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attn(), MLP()

    def run(layers, xs):
        """Each reader of each layer on its own input: {(l, r): output}; the probe stores the input each reader received."""
        return {(l, r): reader(layers[l], r)(xs[l]) for l in range(len(layers)) for r in NAMES}

    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-6), (torch.bfloat16, 2 ** -7)):
        layers = [Layer().to(dtype) for _ in range(2)]
        clean = [torch.tensor(rng.normal(0, 1 + i, (B, T, d)), dtype=dtype) for i in range(2)]
        abl = [x * torch.tensor(rng.uniform(0.5, 3.0, (B, T, 1)), dtype=dtype) + 0.3 for x in clean]
        seen = {}
        with torch.no_grad(), held_input(layers, [0, 1], start) as H:
            # the probe, registered after the hold, sees the input each reader receives
            probe = [reader(layers[l], r).register_forward_pre_hook(lambda m, a, key=(l, r): seen.__setitem__(key, a[0].clone()))
                     for l in range(2) for r in NAMES]
            H.record()
            ref = run(layers, clean)
            H.hold([0, 1])
            run(layers, abl)
            nf = H.done()
            held_in = dict(seen)
            H.hold([0, 1])
            ident = run(layers, clean)
            H.done()
            H.hold([1])
            run(layers, abl)
            one_in = dict(seen)
            H.done()
            H.hold([0, 1], shift=1)
            shifted = run(layers, clean)
            H.done()
            H.hold([0, 1], fill=float("inf"))
            run(layers, clean)
            nf_inf = H.done()
            H.free()
            free = run(layers, abl)
        for h in probe:
            h.remove()
        for l in range(2):
            for r in NAMES:
                rc, rh = rms(clean[l])[:, start:], rms(held_in[(l, r)])[:, start:]
                e = float(((rh - rc).abs() / rc).max())
                assert e <= tol, (str(dtype), l, r, e, "a held reader's input rms is not the clean rms")
                assert torch.equal(held_in[(l, r)][:, :start], abl[l][:, :start]), "a position before the window start was held"
                assert torch.equal(ident[(l, r)], ref[(l, r)]), (str(dtype), l, r, "the held clean forward is not the clean forward")
                assert not torch.equal(shifted[(l, r)], ref[(l, r)]), "the shifted rms reproduced the clean output"
            assert torch.equal(one_in[(0, "v")], abl[0]) and not torch.equal(one_in[(1, "v")], abl[1]), "the one-layer hold held another layer"
        assert nf == {0: 0, 1: 0} and nf_inf == {0: len(NAMES), 1: len(NAMES)}, (nf, nf_inf)
        assert not torch.equal(free[(0, "v")], ref[(0, "v")]), "the constructed ablation did not move the input"
        out[f"hold_{str(dtype).split('.')[-1]}"] = (f"held reader input rms = clean rms to {e:.1e} (tol {tol:.0e}) at every position >= {start}; position < {start} "
                                                    f"untouched; held clean = clean exactly; one-layer hold leaves layer 0; shifted rms differs; inf fill counted "
                                                    f"{nf_inf} non-finite reader calls, finite run {nf}")
    layers = [Layer()]
    with torch.no_grad(), held_input(layers, [0], start) as H:
        H.record()
        run(layers, [torch.ones(B, T, d)])
        H.hold([0])
        raised = False
        try:
            run(layers, [torch.ones(B, T + 1, d)])
        except AssertionError:
            raised = True
    assert raised, "a recorded rms of another sequence length was replayed"
    raised = False
    with torch.no_grad(), held_input(layers, [0], start) as H:
        H.record()
        run(layers, [torch.ones(B, T, d)])
        H.hold([0])
        reader(layers[0], "v")(torch.ones(B, T, d))
        try:
            H.done()
        except AssertionError:
            raised = True
    assert raised, "done() passed with readers that did not replay"
    out["hold_shape"] = "a recorded rms of another shape stops the forward; done() stops when a reader did not replay"
    assert held_layers([(3, 1), (5, 2)], 7) == [3, 4, 5, 6, 7] and held_layers([], 4) == [0, 1, 2, 3, 4]
    # T2: the marked-token write
    lyr = [Layer()]
    a0 = torch.tensor(rng.normal(0, 1, (B, T, 8)), dtype=torch.float32)
    pos = torch.tensor([T - 1, 4, 2])
    idx, val = torch.tensor([1, 6]), torch.tensor([7.0, -3.0])
    seen = {}
    h0 = lyr[0].mlp.down_proj.register_forward_hook(lambda m, a, o: seen.__setitem__("x", a[0].clone()))
    with torch.no_grad(), replaced_at(lyr, {0: (idx, val)}, pos) as rec:
        lyr[0].mlp.down_proj(a0)
    h0.remove()
    want = a0.clone()
    for i in range(B):
        want[i, pos[i], idx] = val
    assert rec["calls"] == 1 and torch.equal(seen["x"], want), "the marked write is not the original input with the marked entries replaced"
    assert int((seen["x"] != a0).sum()) == B * len(idx), "the marked write touched other entries"

    def bad(m, args):                                     # a second writer that also writes position 0: the post-hook must stop it
        a = args[0].clone()
        a[:, 0, 3] = 99.0
        return (a, *args[1:])
    raised = False
    with torch.no_grad(), replaced_at(lyr, {0: (idx, val)}, pos):
        hb = lyr[0].mlp.down_proj.register_forward_pre_hook(bad)
        try:
            lyr[0].mlp.down_proj(a0)
        except AssertionError:
            raised = True
        hb.remove()
    assert raised, "a write away from the marked position passed the post-hook"
    out["marked_write"] = f"{B * len(idx)} entries changed (row, marked position, idx), every other entry equal; a write at position 0 stops the forward"
    # T5: the residual rms and the member write
    N, Lc, dd = 5, 3, 6
    t = lambda *s_: torch.tensor(rng.normal(0, 1, s_), dtype=torch.float32)
    cap = {"emb": t(N, dd), "attn": {l: t(N, dd) for l in range(Lc)}, "mlp": {l: t(N, dd) for l in range(Lc)}}
    rr = residual_rms(cap, Lc - 1)
    h = cap["emb"].double().numpy().copy()
    for l in range(Lc):
        h = h + cap["attn"][l].double().numpy() + cap["mlp"][l].double().numpy()
        assert np.allclose(rr[:, l].numpy(), np.sqrt((h ** 2).mean(-1)), rtol=1e-5), l
    s = torch.tensor(rng.uniform(0.5, 2, (N, Lc)), dtype=torch.float32)
    w = member_writes([2.0, 0.5], [-3.0, 1.5], [0, 2], s, rr)
    want = np.stack([3.0 * 2.0 * s[:, 0].numpy() / rr[:, 0].numpy(), 1.5 * 0.5 * s[:, 2].numpy() / rr[:, 2].numpy()], 1)
    assert np.allclose(w, want, rtol=1e-6), (w, want)
    sm = summary(w)
    assert sm["n_members"] == 2 and abs(sm["median"] - float(np.median(np.median(want, 0)))) < 1e-6 * abs(sm["median"])
    out["member_write"] = "residual rms = rms(emb + cumulative attention + mlp) to 1e-5; |ā| ||d|| s / rms by hand to 1e-6; per-member median then median and p90"
    return out
