"""M-10 / M-11 — the architecture axis: sanctioned registers, and the rotation-equivariant twin.

Two Wave-2 arms share one substrate, so they share one module.

M-10 (queue item 10, "registers-in-width"): reserve the first k residual coordinates at init as a
SANCTIONED sink. Three switches, ABLATED INDEPENDENTLY — never as a bundle, which is the queue's
explicit instruction:
    (a) `reserve_gamma_fixed` — norm gains gamma_i fixed = 1 for i < k
    (b) `reserve_block_v`     — the reserved block gets ONE subspace-level second moment
                                v_blk = EMA(mean_{i<k} g_i^2) instead of per-coordinate v_i
    (c) `writer_init_scale`   — writers' rows into the block scaled s in {0, 0.1, 1}

M-11 (queue item 11, "rotation-equivariant training"): let G = O(d) act on the residual stream
x -> Rx, readers W -> W R^T, writers W -> R W, embeddings absorbing R. Training is G-equivariant
iff every operation AND the optimizer AND the init distribution commute with that action; then
train(R.theta0) = R.train(theta0), so coordinate-spiky structure cannot form except through
explicit symmetry breaking. `GRID_INVENTORY` is the six-item bookkeeping list the queue names, and
`audit_inventory` refuses a run in which any item is neither neutralized nor declared with a bound.

`equivariance_residual` is the gate that makes the twin's claim testable rather than asserted: it
trains the same steps from theta0 and from R.theta0 and measures how far apart the results are
after un-rotating. For a genuinely equivariant configuration this is fp noise; for the grid-aware
one it is large. A gate that only ever passes is not a gate, so both arms are run and reported.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

from infra.pg.darwin_train import ExactAdam, Muon    # REUSED verbatim (§11 reuse ledger)

# The queue's complete grid-aware inventory (item 11). Every entry must be NEUTRALIZED (made
# equivariant) or DECLARED (left grid-aware, with a bound on how much symmetry it breaks).
GRID_INVENTORY = ("norm_gains", "optimizer_second_moment", "dropout", "weight_decay",
                  "fp_rounding", "rope")
NEUTRALIZED = "neutralized"
DECLARED = "declared"


def audit_inventory(decls: dict) -> dict:
    """§11's core bookkeeping, enforced. `decls` maps each inventory item to
    {"status": neutralized|declared, "how": str, "bound": float|None}.

    A DECLARED item must carry a numeric bound on the symmetry breaking it leaves behind (item (5),
    fp rounding, is the designed example) — "we noticed it" is not a declaration.
    """
    missing = [k for k in GRID_INVENTORY if k not in decls]
    assert not missing, (
        f"grid-aware inventory incomplete: {missing}. §11 requires every one of {GRID_INVENTORY} "
        f"to be neutralized or declared — an unlisted symmetry breaker IS the leak the REFUTER "
        f"predicts, and finding it later by surprise is the failure this list prevents")
    extra = [k for k in decls if k not in GRID_INVENTORY]
    assert not extra, f"unknown inventory items {extra}"
    out = {}
    for k in GRID_INVENTORY:
        d = decls[k]
        assert d["status"] in (NEUTRALIZED, DECLARED), f"{k}: bad status {d['status']!r}"
        assert d.get("how", "").strip(), f"{k}: state HOW it was {d['status']}"
        if d["status"] == DECLARED:
            assert d.get("bound") is not None, (
                f"{k}: a DECLARED residual symmetry breaker must carry a numeric bound on its size")
        out[k] = dict(d)
    return {"inventory": out, "n_neutralized": sum(1 for d in out.values()
                                                   if d["status"] == NEUTRALIZED),
            "n_declared": sum(1 for d in out.values() if d["status"] == DECLARED)}


# ---------------------------------------------------------------------------
# The model. A 160M-class pre-norm decoder, written out because every switch above is a change to
# a component that HF's implementations bake in.
# ---------------------------------------------------------------------------
@dataclass
class ArchConfig:
    vocab: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq: int = 512
    # ---- M-11 switches ----
    norm: str = "per_coord"          # "per_coord" (grid-aware) | "scalar" (equivariant)
    dropout: float = 0.0             # per-coordinate Bernoulli masks are grid-aware -> 0
    rope: bool = True                # acts in head space AFTER projection; commutes with G
    # ---- M-10 switches (each independent) ----
    reserved_k: int = 0              # width of the sanctioned block (0 = no block)
    reserve_gamma_fixed: bool = False        # switch (a)
    reserve_block_v: bool = False            # switch (b)
    writer_init_scale: float = 1.0           # switch (c): s in {0, 0.1, 1}
    seed: int = 1337

    def __post_init__(self):
        assert self.norm in ("per_coord", "scalar"), f"unknown norm {self.norm!r}"
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"
        assert 0 <= self.reserved_k <= self.d_model, "reserved block wider than the model"
        assert self.writer_init_scale in (0.0, 0.1, 1.0), \
            f"switch (c) is s in {{0, 0.1, 1}}, got {self.writer_init_scale}"
        if self.reserve_gamma_fixed or self.reserve_block_v:
            assert self.reserved_k > 0, "switches (a)/(b) need a reserved block"
        if self.norm == "scalar":
            assert not self.reserve_gamma_fixed, (
                "switch (a) fixes per-coordinate gains inside the block; with a scalar gain there "
                "are no per-coordinate gains to fix — the two switches are not composable and "
                "silently pretending otherwise would mislabel the arm")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_params(self) -> int:
        emb = self.vocab * self.d_model
        per_layer = 4 * self.d_model ** 2 + 3 * self.d_model * self.d_ff
        return emb + self.n_layers * per_layer


def build_model(cfg: ArchConfig):
    """The switchable decoder. Returns an nn.Module with `residual_spec` attached: the per-parameter
    map the gauge rotation and the block optimizer both need (which axis of each weight faces the
    residual stream)."""
    import math
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        """Per-coordinate gains (grid-aware) or ONE scalar gain (rotation-invariant).

        RMS itself is ‖x‖/sqrt(d), which is rotation-invariant either way; it is the GAIN that
        carries the grid. `fixed_k` pins the first k gains at 1 (M-10 switch (a)) by masking the
        gradient — the parameter still exists so the parameter count is unchanged, and §7's
        confound row ('parameter-count delta from (a) is k params/layer') is satisfied exactly
        rather than approximately.
        """

        def __init__(self, d, scalar: bool, fixed_k: int = 0):
            super().__init__()
            self.scalar = scalar
            self.fixed_k = int(fixed_k)
            self.weight = nn.Parameter(torch.ones(1 if scalar else d))
            if self.fixed_k:
                mask = torch.ones(d)
                mask[:self.fixed_k] = 0.0
                self.register_buffer("grad_mask", mask)
                self.weight.register_hook(lambda g: g * self.grad_mask)

        def forward(self, x):
            r = x.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
            return x * r * self.weight

    class Attention(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            self.h, self.dh, self.rope = cfg.n_heads, cfg.head_dim, cfg.rope
            self.q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.v = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        def forward(self, x, freqs):
            B, T, _ = x.shape
            q, k, v = (t(x).view(B, T, self.h, self.dh).transpose(1, 2) for t in (self.q, self.k, self.v))
            if self.rope:
                q, k = apply_rope(q, freqs), apply_rope(k, freqs)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, -1))

    class MLP(nn.Module):
        """SwiGLU. The elementwise nonlinearity lives in the MLP's own hidden space, NOT in the
        residual stream, so it does not break residual-stream equivariance — the up/gate/down
        projections absorb R."""

        def __init__(self, cfg: ArchConfig):
            super().__init__()
            self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

        def forward(self, x):
            return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

    class Block(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            fk = cfg.reserved_k if cfg.reserve_gamma_fixed else 0
            self.n1 = RMSNorm(cfg.d_model, cfg.norm == "scalar", fk)
            self.attn = Attention(cfg)
            self.n2 = RMSNorm(cfg.d_model, cfg.norm == "scalar", fk)
            self.mlp = MLP(cfg)
            self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        def forward(self, x, freqs):
            x = x + self.drop(self.attn(self.n1(x), freqs))
            return x + self.drop(self.mlp(self.n2(x)))

    class TinyLM(nn.Module):
        def __init__(self, cfg: ArchConfig):
            super().__init__()
            self.cfg = cfg
            torch.manual_seed(cfg.seed)
            self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
            fk = cfg.reserved_k if cfg.reserve_gamma_fixed else 0
            self.nf = RMSNorm(cfg.d_model, cfg.norm == "scalar", fk)
            self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
            self.head.weight = self.emb.weight              # tied
            self.register_buffer("freqs", rope_freqs(cfg.head_dim, cfg.max_seq), persistent=False)
            self.apply(self._init)
            if cfg.reserved_k and cfg.writer_init_scale != 1.0:
                self._scale_writer_rows(cfg.reserved_k, cfg.writer_init_scale)
            self.residual_spec = residual_spec(self)

        def _init(self, m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        def _scale_writer_rows(self, k: int, s: float):
            """M-10 switch (c): scale the rows that WRITE into the reserved block."""
            with torch.no_grad():
                for b in self.blocks:
                    b.attn.o.weight[:k, :].mul_(s)
                    b.mlp.down.weight[:k, :].mul_(s)
                self.emb.weight[:, :k].mul_(s)

        def forward(self, idx, labels=None):
            x = self.emb(idx)
            f = self.freqs[:idx.shape[1]]
            for b in self.blocks:
                x = b(x, f)
            logits = self.head(self.nf(x))
            if labels is None:
                return logits
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
            return logits, loss

    m = TinyLM(cfg)
    assert abs(sum(p.numel() for p in m.parameters()) - cfg.n_params) <= cfg.d_model * (
        2 * cfg.n_layers + 1), "parameter count drifted from ArchConfig.n_params"
    return m


def rope_freqs(head_dim: int, max_seq: int, base: float = 10000.0):
    import torch
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    return torch.outer(t, inv)


def apply_rope(x, freqs):
    """x: (B, H, T, Dh). Rotates coordinate pairs by the position angle."""
    import torch
    cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# The gauge: which axis of which parameter faces the residual stream.
# ---------------------------------------------------------------------------
READER = "reader"     # W x  -> W R^T   (input axis is the residual stream: axis 1)
WRITER = "writer"     # x + W y -> R W  (output axis is the residual stream: axis 0)
GAIN = "gain"         # per-coordinate norm gain -> NOT equivariant (this is inventory item 1)


def residual_spec(model) -> dict:
    """Per-parameter role in the gauge. Built by NAME because that is what the rotation has to
    address, and an unclassified residual-facing parameter would be silently left unrotated —
    which reads as a symmetry break that is really a bug."""
    spec = {}
    for name, p in model.named_parameters():
        if name.endswith(("attn.q.weight", "attn.k.weight", "attn.v.weight",
                          "mlp.gate.weight", "mlp.up.weight")):
            spec[name] = READER
        elif name.endswith(("attn.o.weight", "mlp.down.weight")):
            spec[name] = WRITER
        elif name.endswith(("n1.weight", "n2.weight", "nf.weight")):
            spec[name] = GAIN if p.numel() > 1 else None
        elif name in ("emb.weight", "head.weight"):
            spec[name] = READER          # (vocab, d_model): the residual axis is axis 1
        else:
            raise AssertionError(f"parameter {name} has no declared role in the gauge")
    return spec


def rotate_model_(model, R):
    """Apply the gauge x -> Rx in place. Returns the count of rotated tensors.

    A GAIN parameter cannot be rotated — a per-coordinate gain is not a tensor on the residual
    stream, it is a diagonal operator ON it, and diag(gamma) does not commute with R. That is
    inventory item (1), and this function refuses rather than silently rotating it, because a
    "rotated" grid-aware twin would make the equivariance gate pass for the wrong reason.
    """
    import torch
    spec = model.residual_spec
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            role = spec[name]
            if role is None:
                continue
            assert role != GAIN, (
                f"{name} is a per-coordinate norm gain: diag(gamma) does not commute with R. Set "
                f"ArchConfig.norm='scalar' before rotating (inventory item 1)")
            Rt = R.to(p.dtype)
            if role == READER:
                p.copy_(p @ Rt.T)
            else:
                p.copy_(Rt @ p)
            n += 1
    # tied emb/head are one tensor; count them once
    return n


def random_rotation(d: int, seed: int, dtype=None):
    """A Haar-random orthogonal d x d matrix, in fp64 unless a dtype is asked for.

    The QR always runs in fp64, and the result is DOWNCAST ONLY ON REQUEST. Returning fp32 by
    default and letting callers upcast does not restore orthogonality — ‖RᵀR − I‖ stays at ~1e-7 —
    and every gauge identity checked against such an R then fails at ~1e-6. That is precisely how
    the RoPE commutation gate failed on its first instance run (journal: dry-run 1), and the
    orthonormality assertion below is what turns a repeat into a loud failure at the source.
    """
    import torch
    g = torch.Generator().manual_seed(int(seed))
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    if dtype is not None:
        Q = Q.to(dtype)
    err = float((Q.T @ Q - torch.eye(d, dtype=Q.dtype)).abs().max())
    tol = 1e-10 if Q.dtype == torch.float64 else 1e-5
    assert err < tol, f"rotation not orthonormal at {Q.dtype}: max|QᵀQ − I| = {err:.3e} (tol {tol})"
    return Q


def unrotate_state(state: dict, R, spec: dict) -> dict:
    """Map a trained rotated model's parameters back to the unrotated gauge."""
    import torch
    out = {}
    for name, p in state.items():
        role = spec[name]
        if role is None:
            out[name] = p.clone()
        elif role == READER:
            out[name] = p @ R.to(p.dtype)
        else:
            out[name] = R.to(p.dtype).T @ p
    return out


def equivariance_residual(state_plain: dict, state_rotated: dict, R, spec: dict) -> dict:
    """max |theta_plain - R^{-1}.theta_rotated| over parameters, relative to the parameter scale.

    This is THE gate that decides whether a configuration really is equivariant. It is reported for
    BOTH twins: the equivariant one should sit at fp noise, the grid-aware one far above it. A gate
    that is only ever run on the arm expected to pass cannot fail.
    """
    import torch
    back = unrotate_state(state_rotated, R, spec)
    worst, worst_name = 0.0, None
    for name, p in state_plain.items():
        d = float((p - back[name]).abs().max())
        s = float(p.abs().max()) + 1e-12
        if d / s > worst:
            worst, worst_name = d / s, name
    return {"max_rel_residual": worst, "worst_param": worst_name,
            "n_params_compared": len(state_plain)}


def rope_commutation_check(head_dim: int = 64, seq: int = 16, d_model: int = 128,
                           n_heads: int = 2, seed: int = 1337) -> dict:
    """Inventory item (6), verified ONCE numerically rather than cited.

    Claim: RoPE acts in head space AFTER the q/k projection, so it commutes with a residual-stream
    rotation. Check: rotate the residual stream, absorb R into the reader, and confirm the RoPE'd
    queries are identical to the unrotated path's.
    """
    import torch
    torch.manual_seed(seed)
    R = random_rotation(d_model, seed).to(torch.float64)
    Wq = torch.randn(n_heads * head_dim, d_model, dtype=torch.float64)
    x = torch.randn(1, seq, d_model, dtype=torch.float64)
    f = rope_freqs(head_dim, seq).to(torch.float64)

    def path(xx, W):
        q = (xx @ W.T).view(1, seq, n_heads, head_dim).transpose(1, 2)
        return apply_rope(q, f)

    a = path(x, Wq)
    b = path(x @ R.T, Wq @ R.T)          # rotate the stream, absorb R into the reader
    d = float((a - b).abs().max())
    return {"max_abs_diff": d, "commutes": bool(d < 1e-10), "tol": 1e-10,
            "head_dim": head_dim, "d_model": d_model, "seq": seq}


# ---------------------------------------------------------------------------
# M-10 switch (b): the block second moment.
# ---------------------------------------------------------------------------
class BlockAdam(ExactAdam):
    """ExactAdam, except that the reserved block's second moment is ONE scalar per parameter.

    v_blk = EMA(mean_{i<k} g_i^2) over the reserved slice, exactly as queue item 10(b) writes it.
    Everything outside the block keeps per-coordinate v_i, so this is switch (b) ALONE and not a
    bundle with anything else.

    `roles` maps each parameter's index to its residual role, so the slice is taken on the axis
    that actually faces the residual stream: columns for a reader, rows for a writer, the whole
    vector for a gain.
    """

    def __init__(self, params, roles, reserved_k: int, **kw):
        super().__init__(params, **kw)
        self.roles = list(roles)
        self.k = int(reserved_k)
        assert len(self.roles) == len(self.params), "one role per parameter"
        assert self.k > 0, "BlockAdam with k=0 is just ExactAdam — say so explicitly"
        import torch
        self.v_blk = [torch.zeros((), dtype=torch.float32, device=p.device) for p in self.params]

    @staticmethod
    def _slice(role, k):
        if role == READER:
            return (slice(None), slice(0, k))
        if role == WRITER:
            return (slice(0, k), slice(None))
        if role == GAIN:
            return (slice(0, k),)
        return None

    def step(self):
        import torch
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        with torch.no_grad():
            for i, (p, w, m, v) in enumerate(zip(self.params, self.master, self.m, self.v)):
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32)
                m.mul_(self.b1).add_(g, alpha=1.0 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
                sl = self._slice(self.roles[i], self.k)
                denom = (v / bc2).sqrt().add_(self.eps)
                if sl is not None:
                    self.v_blk[i].mul_(self.b2).add_(g[sl].pow(2).mean(), alpha=1.0 - self.b2)
                    denom[sl] = (self.v_blk[i] / bc2).sqrt().add(self.eps)
                if self.wd:
                    w.add_(w, alpha=-self.lr * self.wd)
                w.addcdiv_(m / bc1, denom, value=-self.lr)
                p.copy_(w.to(p.dtype))


class SGDM:
    """SGD with momentum — the equivariant optimizer of §11(2). Written out so that the arm's
    optimizer is arithmetic you can read, for the same reason ExactAdam is."""

    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=0.0):
        import torch
        self.params = list(params)
        self.lr, self.mu, self.wd = lr, momentum, weight_decay
        # fp32 momentum over bf16/fp16/fp32 parameters (the house convention), but NEVER below the
        # parameter's own precision: an fp32 buffer under fp64 parameters would round every update
        # in the coordinate basis, and rounding does not commute with R — which would make the
        # equivariance gate fail for a reason that has nothing to do with the optimizer.
        lo = (torch.bfloat16, torch.float16, torch.float32)
        self.buf = [torch.zeros_like(p, dtype=torch.float32 if p.dtype in lo else p.dtype)
                    for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        import torch
        with torch.no_grad():
            for p, b in zip(self.params, self.buf):
                if p.grad is None:
                    continue
                g = p.grad.detach().to(torch.float32)
                if self.wd:
                    g = g + self.wd * p.to(torch.float32)      # Frobenius-based: equivariant
                b.mul_(self.mu).add_(g)
                p.add_((-self.lr * b).to(p.dtype))


EQUIVARIANT_OPTIMIZERS = {"sgdm": SGDM, "muon": Muon}
GRID_AWARE_OPTIMIZERS = {"adam": ExactAdam, "blockadam": BlockAdam}


# ---------------------------------------------------------------------------
# Measurements shared by both arms.
# ---------------------------------------------------------------------------
def adoption_angles(V_bottom, reserved_k: int) -> dict:
    """Principal angles between the emergent consensus bottom subspace and the RESERVED block.

    The reserved block is span(e_0..e_{k-1}), so the angles reduce to the singular values of
    V_bottom's first k rows — computed via the general routine anyway, so this number is produced
    by the same code every other subspace comparison in the program uses.

    The REFUTER ('adoption angle stays at the Haar band') is read against `haar_band`, recomputed
    here rather than cited.
    """
    import numpy as np
    from infra.pg import humboldt_core as H
    V = np.asarray(V_bottom, dtype=np.float64)
    n, k = V.shape
    assert 0 < reserved_k <= n, f"reserved_k {reserved_k} out of range for n={n}"
    E = np.zeros((n, reserved_k))
    E[np.arange(reserved_k), np.arange(reserved_k)] = 1.0
    ang = H.principal_angles(V, E)
    band = H.haar_angle_band(n, min(k, reserved_k))
    return {"angles_rad": ang.tolist(), "mean_angle_rad": float(ang.mean()),
            "haar_band_mean": float(np.mean(band["mean"])),
            "haar_band_p2.5": float(np.mean(band["p2.5"])),
            "haar_band_p97.5": float(np.mean(band["p97.5"])),
            "n": int(n), "k": int(k), "reserved_k": int(reserved_k)}


def leverage_in_out(U, reserved_k: int) -> dict:
    """Leverage mass inside vs outside the reserved block, against the flat expectation."""
    import numpy as np
    from infra.pg import humboldt_core as H
    lev = H.leverage(U)
    n = lev.size
    inside = float(lev[:reserved_k].sum())
    total = float(lev.sum())
    flat = reserved_k / n
    return {"leverage_inside": inside, "leverage_total": total,
            "fraction_inside": inside / total, "flat_expectation": flat,
            "ratio": (inside / total) / flat,
            "participation_ratio": H.participation_ratio(lev)}


def max_abs_activation(X, reserved_k: int = 0) -> dict:
    """max |x_i| over the residual stream, split in/out of the reserved block.

    Queue item 10 asks 'does giving outliers a home reduce wild magnitudes elsewhere?', which is a
    question about the OUTSIDE maximum, so both are reported and neither is summarised away.
    """
    import numpy as np
    A = np.asarray(X, dtype=np.float64).reshape(-1, np.asarray(X).shape[-1])
    out = {"max_abs_all": float(np.abs(A).max()),
           "p99_9_abs_all": float(np.percentile(np.abs(A), 99.9)), "n_tokens": int(A.shape[0])}
    if reserved_k:
        out["max_abs_inside"] = float(np.abs(A[:, :reserved_k]).max())
        out["max_abs_outside"] = float(np.abs(A[:, reserved_k:]).max())
    return out


def equal_val_loss_match(steps_a, loss_a, steps_b, loss_b, *, n_points: int = 5) -> dict:
    """§7: compare the twins at equal VALIDATION LOSS, never at equal steps.

    Returns `n_points` loss levels spanning the OVERLAP of the two curves, with the (interpolated)
    step at which each run reaches each level. Refuses when the curves do not overlap: two runs
    that never reach a common loss cannot be compared at equal loss, and reporting a step-matched
    number in that case is exactly the confound this rule exists to stop.
    """
    import numpy as np
    sa, la = np.asarray(steps_a, float), np.asarray(loss_a, float)
    sb, lb = np.asarray(steps_b, float), np.asarray(loss_b, float)
    for s, l, nm in ((sa, la, "a"), (sb, lb, "b")):
        assert s.shape == l.shape and s.size >= 2, f"curve {nm}: need >=2 paired points"
        assert (np.diff(s) > 0).all(), f"curve {nm}: steps must be strictly increasing"
    lo = max(la.min(), lb.min())
    hi = min(la.max(), lb.max())
    assert hi > lo, (
        f"validation-loss curves do not overlap (a: [{la.min():.4f}, {la.max():.4f}], "
        f"b: [{lb.min():.4f}, {lb.max():.4f}]) — an equal-loss comparison is undefined and a "
        f"step-matched one is confounded by the optimizer's convergence speed")
    levels = np.linspace(lo, hi, n_points)

    def step_at(s, l, target):
        order = np.argsort(l)
        return float(np.interp(target, l[order], s[order]))

    return {"levels": levels.tolist(),
            "steps_a": [step_at(sa, la, t) for t in levels],
            "steps_b": [step_at(sb, lb, t) for t in levels],
            "overlap_lo": float(lo), "overlap_hi": float(hi)}
