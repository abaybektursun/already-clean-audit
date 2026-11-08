"""Wave-2 shared substrate: small pretrains with switchable optimizers and architecture.

M-10 and M-11 ask opposite-facing questions about the same object — whether the sink's ADDRESS can
be dictated (M-10: does a sanctioned register block get adopted?) and whether it exists at all
under an optimizer that has no coordinates to prefer (M-11: is the equivariant twin spiky?). They
need the same harness, so they share it, and a difference between the two arms cannot be a
difference of harness.

WHY SMALL MODELS. The question is about what training DOES, not about scale, and a 4-layer 256-wide
transformer trained for a few thousand steps produces a consensus Gram with the same estimand as a
3B one. The alternative — one 3B pretrain — would answer the question once, at one seed, with no
switch arms. Stated plainly: these are small models and the arms are switches on them.

THE ESTIMAND IS THE PROGRAM'S. The emergent sink is the bottom-k of `sum_t dW_t^T dW_t / ||dW_t||^2`
over consecutive training checkpoints — the same normalisation `darwin_census.census_grams` uses
for a fine-tune cloud and `darwin_w0_ext.delta_terms` uses for a checkpoint suite. So "the sink" in
a Wave-2 pretrain and "the sink" in the PG1 census are the same construction.

FAIL LOUD: no try/except, no synthetic fallback.
"""
from __future__ import annotations

# The arms are named here so both jobs and the tests agree on the vocabulary.
OPTIMIZERS = ("adam", "sgd", "block_adam", "scalar_adam")
SWITCHES = ("none", "fixed_gamma", "writer_init_ablated")


def tiny_config(vocab_size: int, d_model: int, n_layers: int, n_heads: int,
                max_positions: int = 512):
    """A small Gemma-shaped decoder. Gemma-shaped so its residual stream, its RMSNorm gains and its
    q_proj input site are the same objects every other arm in this program reads."""
    from transformers import GemmaConfig
    assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
    return GemmaConfig(vocab_size=vocab_size, hidden_size=d_model,
                       intermediate_size=4 * d_model, num_hidden_layers=n_layers,
                       num_attention_heads=n_heads, num_key_value_heads=n_heads,
                       head_dim=d_model // n_heads,
                       max_position_embeddings=max_positions,
                       attn_implementation="eager")


class BlockAdam:
    """Adam whose second moment is shared within contiguous coordinate BLOCKS of the input axis.

    M-A3 stage 2 measured that replacing the per-coordinate denominator with ONE scalar releases
    the core 27.7x. This is the intermediate: a denominator with `block` resolution instead of 1 or
    n. M-10's switch (b) is exactly this, and the block size is the knob — at block=1 it is
    ExactAdam and at block=n it is ScalarDenomAdam, which is why those two are the ladder's ends
    and are asserted to coincide with them.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, block=32):
        import torch
        self.params = list(params)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.block = int(block)
        assert self.block >= 1, f"block {block} must be >= 1"
        self.t = 0
        self.master = [p.detach().to(torch.float32).clone() for p in self.params]
        self.m = [torch.zeros_like(x) for x in self.master]
        self.v = [torch.zeros_like(x) for x in self.master]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

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
                vh = (v / bc2).sqrt()
                if w.ndim == 2 and self.block > 1:
                    n = w.shape[1]
                    nb = (n + self.block - 1) // self.block
                    pad = nb * self.block - n
                    x = torch.nn.functional.pad(vh, (0, pad))
                    x = x.reshape(w.shape[0], nb, self.block).mean(dim=2, keepdim=True)
                    x = x.expand(w.shape[0], nb, self.block).reshape(w.shape[0], -1)[:, :n]
                    vh = x.contiguous()
                if self.wd:
                    w.add_(w, alpha=-self.lr * self.wd)
                w.addcdiv_(m / bc1, vh.add_(self.eps), value=-self.lr)
                p.copy_(w.to(p.dtype))


def build_optimizer(name: str, params, *, lr: float, block: int = 32, momentum: float = 0.9):
    """The four update rules the Wave-2 arms switch between.

    `sgd` is here because it is EXACTLY rotation-equivariant: its update is the gradient, and a
    rotation of the parameter space rotates the gradient with it. That is what makes it M-11's
    twin rather than merely a second optimizer.
    """
    import torch
    from infra.pg import darwin_train as T
    from infra.pg import mercator_train as TR
    assert name in OPTIMIZERS, f"unknown optimizer {name!r} (expected one of {OPTIMIZERS})"
    ps = list(params)
    if name == "adam":
        return T.ExactAdam(ps, lr=lr)
    if name == "scalar_adam":
        return TR.ScalarDenomAdam(ps, lr=lr)
    if name == "block_adam":
        return BlockAdam(ps, lr=lr, block=block)
    return torch.optim.SGD(ps, lr=lr, momentum=momentum)


def sanctioned_block(d_model: int, k: int, seed: int, *, kind: str = "coordinate"):
    """The register block a Wave-2 pretrain is offered.

    `coordinate` returns the FIRST k coordinate axes — a block the architecture can name, which is
    what "sanctioned" means: the switches can give it a fixed gain or a special denominator only if
    it is expressible in coordinates. `haar` returns a random k-subspace and is the null: a block
    nothing about the architecture distinguishes.
    """
    import numpy as np
    assert 0 < k <= d_model, f"k={k} out of range for d_model={d_model}"
    if kind == "coordinate":
        U = np.zeros((d_model, k), dtype=np.float64)
        U[np.arange(k), np.arange(k)] = 1.0
        return U
    assert kind == "haar", f"unknown block kind {kind!r}"
    from infra.pg import humboldt_core as H
    return H.haar_subspace(d_model, k, seed)


def consensus_from_snapshots(snaps, keys, k: int):
    """Bottom-k of sum_t dW_t^T dW_t / ||dW_t||^2_F over consecutive snapshots, per cell.

    Same normalisation as `census_grams` and `delta_terms`, so the emergent sink of a Wave-2
    pretrain and the sink of the PG1 census are the same estimand rather than two things sharing a
    name.
    """
    import numpy as np
    from infra.aws_pg.jobs.invariant_core_func import _bottom_eigvecs
    assert len(snaps) >= 2, f"need >= 2 snapshots to form a delta, got {len(snaps)}"
    out = {}
    for key in keys:
        n = snaps[0][key].shape[1]
        G = np.zeros((n, n), dtype=np.float64)
        used = 0
        for a, b in zip(snaps[:-1], snaps[1:]):
            D = np.asarray(b[key], dtype=np.float64) - np.asarray(a[key], dtype=np.float64)
            f2 = float((D ** 2).sum())
            if f2 == 0:
                continue
            G += (D.T @ D) / f2
            used += 1
        assert used >= 1, f"cell {key} never moved across the snapshot ladder"
        U, w = _bottom_eigvecs(G / used, k)
        out[key] = {"U": U, "eigs": w, "n_terms": used}
    return out


def adoption_angle(U_emergent, U_sanctioned) -> dict:
    """Grassmann overlap between the emergent sink and the sanctioned block.

    Reported as mean cos^2 of the principal angles — 1.0 for identical subspaces, k/n for
    independent ones — with the independent level stated on the row so "adopted" is never read
    against zero.
    """
    import numpy as np
    from infra.pg import darwin_w0 as DW
    A = np.asarray(U_emergent, dtype=np.float64)
    B = np.asarray(U_sanctioned, dtype=np.float64)
    assert A.shape[0] == B.shape[0], f"ambient mismatch {A.shape} vs {B.shape}"
    r = DW.subspace_agreement(A, B)
    n, k = A.shape[0], min(A.shape[1], B.shape[1])
    return {"mean_cos2": float(r["mean_cos2"]),
            "independent_level": float(k) / float(n),
            "ratio_to_independent": float(r["mean_cos2"]) * float(n) / float(k),
            "max_angle_rad": float(r["max_angle_rad"]),
            "min_angle_rad": float(r["min_angle_rad"]), "k": int(k), "n": int(n)}


def haar_adoption_band(U_emergent, n_draws: int, seed: int) -> dict:
    """The band `adoption_angle` would show against random blocks of the same size — the null the
    adoption claim is read against, computed from the SAME emergent subspace so the band reflects
    its actual dimension and conditioning rather than a textbook expectation."""
    import numpy as np
    from infra.pg import humboldt_core as H
    A = np.asarray(U_emergent, dtype=np.float64)
    vals = [adoption_angle(A, H.haar_subspace(A.shape[0], A.shape[1], seed + i))["mean_cos2"]
            for i in range(n_draws)]
    return {"lo": float(np.min(vals)), "hi": float(np.max(vals)),
            "mean": float(np.mean(vals)), "n_draws": int(n_draws), "values": vals}
