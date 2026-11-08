"""DARWIN — the shared fine-tuning harness for the retention arms (W1-U, W1-T, WP-1).

Every retention arm asks the same question in a different dialect: fine-tune on a task that is
known to damage the base, and see which update rule keeps the base's language ability. So the
arms must differ ONLY in the update rule, never in the model, the data, the optimizer budget or
the eval. That is what this module enforces.

§0.2.7 compliance, which is not optional here:
  * EXACT fp32 Adam. `darwin_core.check_optimizer` refuses an approximate optimizer on
    W1-Q/S/T/U, W2-A/B and S2-A/B, because on those arms the second-moment representation is the
    object under study. This harness keeps fp32 master weights and fp32 Adam moments and applies
    the update in fp32, then casts back to the model's bf16 parameter — no 8-bit states, no
    fused-approximate path.
  * The trainable set is a STATED SCOPE, applied identically to every arm: the decoder's q_proj
    and o_proj across all layers (~151M params on PG1). Exact fp32 Adam over those costs ~1.8 GB
    of optimizer state, which fits a 48 GB L40S natively and so avoids the CPU-offload throughput
    tax entirely. Restricting the trainable set changes what all arms can express EQUALLY, so the
    contrast between update rules is unaffected; it is recorded on every row rather than implied.

FAIL LOUD: no try/except, no silent fallback.
"""
from __future__ import annotations

TRAINABLE_TYPES = ("q_proj", "o_proj")


def trainable_modules(decoder_layers, types=TRAINABLE_TYPES, layers=None):
    """{(type, layer): nn.Linear} — the stated trainable scope, identical across arms."""
    out = {}
    idx = range(len(decoder_layers)) if layers is None else layers
    for L in idx:
        for t in types:
            parent = (decoder_layers[L].self_attn if t.startswith(("q", "k", "v", "o"))
                      else decoder_layers[L].mlp)
            out[(t, int(L))] = getattr(parent, t)
    return out


class ExactAdam:
    """fp32 AdamW over a set of bf16 parameters, with fp32 master weights and moments.

    Written out rather than delegated to `torch.optim.AdamW` because the arms this serves are
    exactly the ones §0.2.7 forbids from using an approximate optimizer, and "we passed
    fused=False" is a weaker guarantee than arithmetic you can read. Decoupled weight decay
    (AdamW), bias correction, no gradient scaling — the model runs in bf16 but every optimizer
    quantity is fp32.
    """

    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        import torch
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.t = 0
        self.master = [p.detach().to(torch.float32).clone() for p in self.params]
        self.m = [torch.zeros_like(x) for x in self.master]
        self.v = [torch.zeros_like(x) for x in self.master]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @property
    def state_bytes(self) -> int:
        return sum(x.numel() * 4 * 3 for x in self.master)     # master + m + v, all fp32

    def step(self):
        import torch
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        with torch.no_grad():
            for p, w, m, v in zip(self.params, self.master, self.m, self.v):
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32)
                m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
                if self.wd:
                    w.add_(w, alpha=-self.lr * self.wd)
                w.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(self.eps), value=-self.lr)
                p.copy_(w.to(p.dtype))


class Muon:
    """W2-A's second optimizer regime: momentum, orthogonalized by Newton-Schulz, applied in fp32.

    Muon's update is the ORTHOGONALIZATION of the momentum buffer — take M, find the nearest
    semi-orthogonal matrix, step along that. It matters to this program specifically because an
    orthogonalized update has a flat singular spectrum BY CONSTRUCTION, so any subspace preference
    it shows cannot come from the optimizer concentrating its step in a few directions. That makes
    it the sharpest available probe of §4/W2-A's question — what breaks the invariant — and the
    reason it is here rather than a third Adam learning rate.

    The Newton-Schulz iteration is the standard quintic (a, b, c) = (3.4445, -4.7750, 2.0315) run
    at fp32 on the row-normalized matrix; five steps is the published default and is frozen here.
    Non-2D parameters fall back to plain momentum SGD, which is also what the reference does.
    """

    NS_STEPS = 5
    NS_COEFFS = (3.4445, -4.7750, 2.0315)

    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True, weight_decay=0.0):
        import torch
        self.params = list(params)
        self.lr, self.mu, self.nesterov, self.wd = lr, momentum, nesterov, weight_decay
        self.buf = [torch.zeros_like(p, dtype=torch.float32) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @property
    def state_bytes(self) -> int:
        return sum(b.numel() * 4 for b in self.buf)

    @classmethod
    def orthogonalize(cls, M):
        """Nearest-semi-orthogonal approximation of M by Newton-Schulz, fp32."""
        import torch
        a, b, c = cls.NS_COEFFS
        X = M.to(torch.float32)
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        X = X / (X.norm() + 1e-7)
        for _ in range(cls.NS_STEPS):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        return X.T if transposed else X

    def step(self):
        import torch
        self.t += 1
        with torch.no_grad():
            for p, buf in zip(self.params, self.buf):
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32)
                buf.mul_(self.mu).add_(g)
                d = g.add(buf, alpha=self.mu) if self.nesterov else buf
                if d.ndim == 2:
                    d = self.orthogonalize(d)
                    # Scale so the update's magnitude is comparable to the plain-momentum path;
                    # the orthogonalized matrix has unit-ish singular values regardless of ‖g‖.
                    d = d * (max(d.shape) ** 0.5)
                if self.wd:
                    p.add_(p.to(torch.float32), alpha=-self.lr * self.wd)
                p.add_((-self.lr * d).to(p.dtype))


def fake_quant(w, bits: int):
    """Symmetric per-tensor absmax fake quantization — the value used by W2-A's QAT regime.

    Returned as a tensor with the STRAIGHT-THROUGH gradient already attached: the forward sees the
    quantized weight, the backward sees an identity. Written as `w + (q - w).detach()` rather than
    with a custom autograd Function because the one-line form is exactly the STE definition and
    has nothing to get wrong.
    """
    import torch
    assert bits >= 2, f"fake_quant needs >= 2 bits, got {bits}"
    qmax = 2 ** (bits - 1) - 1
    scale = w.detach().abs().max().clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    return w + (q - w).detach()


class QATLinear:
    """Wrap a set of nn.Linear so their forward uses `fake_quant(weight, bits)` with STE.

    Swaps the bound `forward`, restores it on exit. A weight-replacing hook cannot express this:
    the quantized weight must be a NODE in the graph for the straight-through gradient to reach
    the real parameter, and a hook that overwrites `.data` breaks that edge silently.
    """

    def __init__(self, modules, bits: int):
        self.mods = modules
        self.bits = int(bits)
        self._saved = {}

    def __enter__(self):
        import torch.nn.functional as F
        for key, mod in self.mods.items():
            self._saved[key] = mod.forward

            def fwd(x, _m=mod):
                return F.linear(x, fake_quant(_m.weight, self.bits), _m.bias)
            mod.forward = fwd
        return self

    def __exit__(self, *exc):
        for key, mod in self.mods.items():
            mod.forward = self._saved[key]
        self._saved = {}
        return False


def motion_fraction(dW, U):
    """‖dW U‖²_F / ‖dW‖²_F — the fraction of a realized update that landed in the bank.

    This is the house "realized motion" object (REFEREE), not a weight-energy gauge: it is
    normalized by the update, so a regime that simply moves further does not automatically look
    like it moves more IN the bank. W2-A's whole question is which regimes change this ratio.
    """
    import numpy as np
    d = np.asarray(dW, dtype=np.float64)
    u = np.asarray(U, dtype=np.float64)
    tot = float((d ** 2).sum())
    assert tot > 0, "motion_fraction on a zero update"
    return float(((d @ u) ** 2).sum() / tot)


class LoRAAdapter:
    """W1-T's named incumbent: rank-r low-rank adapters, W_eff = W0 + (B A) * (alpha / r).

    Only A and B train; W0 is frozen. The scale is alpha/r, the house convention
    (reference_adapter_alpha_scale_convention) — passing alpha explicitly matters because a rank
    sweep at fixed alpha silently changes the effective learning rate per rank.

    B is zero-initialised so the adapter starts as an exact no-op: at step 0 the model IS the
    base, which is what makes the comparison against a full fine-tune fair from the first step.

    The point of implementing this rather than naming it: §3's matched-incumbent rule means a
    Register-B arm with an unimplemented incumbent is not a contrast at all. W1-T sat unexecuted
    for exactly that reason.
    """

    def __init__(self, modules, rank: int, alpha: float, device, seed: int = 1337):
        import torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scale = float(alpha) / float(rank)
        self.mods = modules
        self.A, self.B = {}, {}
        self._handles = []
        for key, mod in modules.items():
            out_f, in_f = mod.weight.shape
            a = torch.randn(int(rank), in_f, generator=g) * (1.0 / max(1, in_f) ** 0.5)
            self.A[key] = a.to(device=device, dtype=torch.float32).requires_grad_(True)
            self.B[key] = torch.zeros(out_f, int(rank), device=device,
                                      dtype=torch.float32).requires_grad_(True)

    def parameters(self):
        return list(self.A.values()) + list(self.B.values())

    @property
    def n_params(self) -> int:
        return sum(t.numel() for t in self.parameters())

    def _mk(self, key):
        import torch          # module convention: heavy imports inside the function that uses them

        def hook(_m, _args, output):
            x = _args[0]
            d = (x.to(torch.float32) @ self.A[key].T) @ self.B[key].T
            return output + (d * self.scale).to(output.dtype)
        return hook

    def __enter__(self):
        for key, mod in self.mods.items():
            self._handles.append(mod.register_forward_hook(self._mk(key)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


class SubspaceProjector:
    """Project each parameter's GRADIENT onto a fixed input subspace: g <- g U Uᵀ.

    This is W1-T's "highway-restricted training": the update is confined to the consensus-written
    subspace, so the complement — which is where the core lives — cannot move at all. Applied to
    the gradient rather than the weight so the restriction holds for every optimizer step
    including momentum, which a post-hoc weight projection would not guarantee.
    """

    def __init__(self, basis_by_key, device):
        import torch
        self.U = {k: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for k, U in basis_by_key.items()}

    def apply(self, modules):
        import torch
        with torch.no_grad():
            for key, mod in modules.items():
                if mod.weight.grad is None or key not in self.U:
                    continue
                U = self.U[key]
                g = mod.weight.grad.to(torch.float32)
                mod.weight.grad = ((g @ U) @ U.T).to(mod.weight.grad.dtype)


class CoreActivationAnchor:
    """W1-U's treatment: penalise drift of the CORE-projected activations from the base model's.

    L_total = L_task + lam * mean_l ‖U_lᵀ(h_l - h_l^base)‖² / ‖U_lᵀ h_l^base‖²

    Activation-space, not weight-space — that is the whole contrast against the EWC-lite control,
    which penalises drift of the core COORDINATES OF THE WEIGHTS. If the anchor only matched
    EWC-lite, the effect would be generic regularisation rather than anything about the core, and
    §4/W1-U's REFUTER says exactly that.

    The reference activations are captured from the FROZEN base on the same batch, so the penalty
    is computed against what the base actually did, not against a stored summary.
    """

    def __init__(self, modules, basis_by_key, device, lam: float):
        import torch
        self.mods = modules
        self.U = {k: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for k, U in basis_by_key.items()}
        self.lam = float(lam)
        self.ref = {}
        self.cur = {}
        self._handles = []
        self._recording = None

    def _hook(self, key):
        def pre_hook(_m, args):
            x = args[0]
            store = self.ref if self._recording == "ref" else self.cur
            store[key] = x.reshape(-1, x.shape[-1])
        return pre_hook

    def __enter__(self):
        for key, mod in self.mods.items():
            if key in self.U:
                self._handles.append(mod.register_forward_pre_hook(self._hook(key)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def record_reference(self):
        self._recording = "ref"
        self.ref = {}

    def record_current(self):
        self._recording = "cur"
        self.cur = {}

    def penalty(self):
        """Relative squared drift of the core-projected activations, averaged over cells."""
        import torch
        assert self.ref, "anchor penalty requested before a reference forward"
        terms = []
        for key, U in self.U.items():
            if key not in self.cur or key not in self.ref:
                continue
            a = self.cur[key].to(torch.float32) @ U
            b = self.ref[key].to(torch.float32) @ U
            den = (b ** 2).sum().clamp_min(1e-12)
            terms.append(((a - b) ** 2).sum() / den)
        assert terms, "anchor has no matching cells"
        return self.lam * torch.stack(terms).mean()


class EwcLiteCore:
    """W1-U's named CONTROL: quadratic penalty on the WEIGHTS' core coordinates.

    L_total = L_task + lam * mean ‖(W - W0) U‖²_F / ‖W0 U‖²_F

    Same subspace, same lambda ladder, same everything — except it constrains weights instead of
    activations. That is the comparison §4/W1-U's REFUTER turns on, so it must share the harness
    down to the normalisation.
    """

    def __init__(self, modules, basis_by_key, device, lam: float):
        import torch
        self.mods = modules
        self.U = {k: torch.as_tensor(U, device=device, dtype=torch.float32)
                  for k, U in basis_by_key.items()}
        self.lam = float(lam)
        self.W0 = {k: m.weight.detach().to(torch.float32).clone()
                   for k, m in modules.items() if k in self.U}
        self.den = {k: ((self.W0[k] @ self.U[k]) ** 2).sum().clamp_min(1e-12) for k in self.W0}

    def penalty(self):
        import torch
        terms = []
        for key, U in self.U.items():
            d = (self.mods[key].weight.to(torch.float32) - self.W0[key]) @ U
            terms.append((d ** 2).sum() / self.den[key])
        assert terms, "ewc-lite has no matching cells"
        return self.lam * torch.stack(terms).mean()


def core_energy_gauge(modules, basis_by_key):
    """§4/W1-U's forgetting gauge (3.2): per-cell core-projected weight-drift energy.

    Recorded at every checkpoint of the NAIVE arm so the director can ask whether the gauge LEADS
    the damage. Returns {key: relative drift}, a pure readout with no effect on training.
    """
    import torch
    out = {}
    with torch.no_grad():
        for key, U in basis_by_key.items():
            if key not in modules:
                continue
            Ut = torch.as_tensor(U, device=modules[key].weight.device, dtype=torch.float32)
            W = modules[key].weight.to(torch.float32)
            out[key] = float(((W @ Ut) ** 2).sum() / (W ** 2).sum().clamp_min(1e-12))
    return out


def lam_ladder(n: int = 4, lo: float = 0.01, hi: float = 10.0):
    """>=4 log-spaced anchor doses (§3). Returned as a plain list so it lands in a manifest."""
    import numpy as np
    return [float(x) for x in np.geomspace(lo, hi, n)]
