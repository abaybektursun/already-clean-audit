"""STAGEDIFF interventions — spec §2 I1 (head transplant), I2 (trace projection), I3 (coordinate swap),
TRACE banks and the ADL random-difference null (§5).

I1  `transplant_head(child, snap)` copies the parent's `ln_f.weight` and `lm_head.weight[:V_SHARED]` into the
    resident child in place and returns a restore closure. Control (§6.4, amended 2026-09-04): the SAME-NORM
    random change — W_C + (W_P − W_C) Q and γ_C + (γ_P − γ_C) Q with Q a random orthogonal d×d matrix — so the
    null asks "does any head change of this size help?", not "does garbage help?".
I2  `ProjectOut(layers, {ℓ: u_ℓ})` removes the unit direction u_ℓ from every position's residual at every
    band layer (fp32 arithmetic, cast back). Control: 8 norm-matched random-difference vectors from the
    parent's null bank (two random (sequence, position) activations subtracted — Minder et al. §3 recipe).
I3  `Swap(lm, lens, band, ...)`: in the lens (final) basis x = J_ℓ h, the component along ŵ_old (the unit
    read-out direction of the old token, g = γ ⊙ W_U[t]) is moved onto ŵ_new at scale α:
    x' = x + α c (ŵ_new − ŵ_old), c = x·ŵ_old, and pulled back, h' = h + (x' − x) J_ℓ⁻ᵀ, at every position of
    every band layer. `new_vec` may be a random unit vector (the norm-matched random-direction null).
TRACE `trace_bank` stores h at content positions 0..4 for every sequence and layer (fp16); `null_bank` stores
    64 activations at random (sequence, position ≥ 5); `delta_bar` = mean_child − mean_parent per (ℓ, j).
"""
from __future__ import annotations

import random

from infra.pg.stagediff_models import BAND, V_SHARED


# ------------------------------------------------------------------------------------- I1: head
def head_snapshot(member: dict) -> dict:
    an = member["an"]
    return {"id": member["id"], "ln_f": an["norm"].weight.detach().float().cpu().clone(),
            "lm_head": an["lm_head"].weight.detach()[:V_SHARED].cpu().clone(), "n_rows": V_SHARED}


def _install_head(member: dict, gamma, W):
    """Copy (gamma, W[:V_SHARED]) into the resident model in place; return restore()."""
    import torch
    an = member["an"]
    g_p, W_p = an["norm"].weight, an["lm_head"].weight
    g_old, W_old = g_p.detach().clone(), W_p.detach()[:V_SHARED].clone()
    with torch.no_grad():
        g_p.copy_(gamma.to(g_p.device, g_p.dtype))
        W_p[:V_SHARED].copy_(W.to(W_p.device, W_p.dtype))
    if getattr(an["embed"], "weight", None) is not None and an["embed"].weight.data_ptr() == W_p.data_ptr():
        pass    # tied embeddings (Qwen3): the input embedding moved too — the transplant covers both, by construction

    def restore():
        with torch.no_grad():
            g_p.copy_(g_old); W_p[:V_SHARED].copy_(W_old)
    return restore


def transplant_head(member: dict, snap: dict):
    return _install_head(member, snap["ln_f"], snap["lm_head"])


def random_orthogonal(d: int, seed: int, device="cpu"):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(device)


def random_head_control(member: dict, snap: dict, seed: int = 0):
    """Same-norm random head change: W_C + (W_P − W_C) Q, γ_C + (γ_P − γ_C) Q. Returns (restore, stats)."""
    import torch
    an = member["an"]
    W_c = an["lm_head"].weight.detach()[:V_SHARED].float().cpu()
    g_c = an["norm"].weight.detach().float().cpu()
    dW = snap["lm_head"].float() - W_c
    dg = snap["ln_f"].float() - g_c
    Q = random_orthogonal(W_c.shape[1], seed)
    W_new = W_c + dW @ Q
    g_new = g_c + dg @ Q
    stats = {"seed": seed, "dW_fro": float(dW.norm()), "dWQ_fro": float((dW @ Q).norm()),
             "dg_norm": float(dg.norm()), "dgQ_norm": float((dg @ Q).norm()),
             "rel_head_change": float(dW.norm() / W_c.norm()), "rel_norm_change": float(dg.norm() / g_c.norm())}
    return _install_head(member, g_new, W_new), stats


# ------------------------------------------------------------------------------------- TRACE banks
def trace_bank(lm, seqs: list, n_pos: int = 5):
    """[n_seq, n_layers, n_pos, d] fp16 CPU at content positions 0..n_pos-1."""
    import torch
    from infra.pg.stagediff_lens import capture
    rows = [capture(lm, s, positions=list(range(n_pos))).to(torch.float16).cpu() for s in seqs]
    return torch.stack(rows, 0)


def null_bank(lm, seqs: list, n: int = 64, seed: int = 0, min_pos: int = 5) -> dict:
    """n activations at random (sequence, content position ≥ min_pos): {"acts": [n, n_layers, d] fp16, "where"}."""
    import torch
    from infra.pg.stagediff_lens import capture
    rng = random.Random(seed)
    where = [(rng.randrange(len(seqs)), None) for _ in range(n)]
    where = [(i, rng.randrange(min_pos, len(seqs[i]))) for i, _ in where]
    acts = torch.stack([capture(lm, seqs[i], positions=[p])[:, 0, :].to(torch.float16).cpu() for i, p in where], 0)
    return {"acts": acts, "where": where}


def delta_bar(bank_child, bank_parent):
    """[n_layers, n_pos, d] fp32: mean over sequences of (h_child − h_parent) = mean_c − mean_p (linear)."""
    assert bank_child.shape == bank_parent.shape, (bank_child.shape, bank_parent.shape)
    return bank_child.float().mean(0) - bank_parent.float().mean(0)


def random_diffs(null_acts_parent, n_draws: int = 8, seed: int = 0):
    """[n_draws, n_layers, d] fp32: differences of two distinct random null-bank activations (ADL null)."""
    import torch
    rng = random.Random(seed)
    n = null_acts_parent.shape[0]
    out = []
    for _ in range(n_draws):
        i, j = rng.sample(range(n), 2)
        out.append(null_acts_parent[i].float() - null_acts_parent[j].float())
    return torch.stack(out, 0)


def norm_match(vecs, target):
    """Scale each [.., n_layers, d] vector per layer to the target's per-layer norm ([n_layers, d] reference)."""
    import torch
    tnorm = target.float().norm(dim=-1)                                  # [n_layers]
    vnorm = vecs.float().norm(dim=-1).clamp_min(1e-12)                   # [.., n_layers]
    return vecs.float() * (tnorm / vnorm).unsqueeze(-1)


def unit_rows(v):
    return v.float() / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------------------------- I2: projection
class ProjectOut:
    """Forward hooks on `layers[ℓ]` for ℓ in `u_by_layer` that remove the unit direction u_ℓ at every position."""

    def __init__(self, layers, u_by_layer: dict):
        self.layers = layers
        self.u = {int(L): u for L, u in u_by_layer.items()}
        self.handles = []
        self.n_calls = 0

    def _hook(self, L):
        u = self.u[L]

        def h(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            uu = u.to(hs.device, dtype=hs.dtype if hs.dtype == u.dtype else None) if False else u.to(hs.device).float()
            hf = hs.float()
            hf = hf - (hf @ uu).unsqueeze(-1) * uu
            self.n_calls += 1
            new = hf.to(hs.dtype)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        return h

    def __enter__(self):
        for L in sorted(self.u):
            self.handles.append(self.layers[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()
        self.handles = []


# ------------------------------------------------------------------------------------- I3: coordinate swap
class Swap:
    """Lens-coordinate swap in RESIDUAL space at every position of every band layer (§15.19).

    Token t's read-out direction at layer ℓ is the gradient of its lens logit with respect to h:
    v_t = J_ℓᵀ g_t with g_t = γ ⊙ W_U[t]; unit-normalised v̂_t. The swap moves the coordinate along the old
    token's direction onto the new token's direction at the same magnitude:
        h' = h + α · c · (v̂_new − v̂_old),   c = h · v̂_old.
    The first construction pulled a final-basis change back through J_ℓ⁻¹; on the real C0 lens cond(J_13)
    = 2.4e8, the pull-back exploded the residual and BOTH the real swap and the random-direction control
    flipped 100 % of the parent's answers (W3 pair-H1/2, I3 VOID by its own ceiling). This form is bounded
    by |c| ≤ ‖h‖ and needs no inverse; per-layer cond(J) and cos(v̂_old, v̂_new) are banked as diagnostics.
    `new_vec=None` with `random_seed` draws an independent random unit direction per layer (norm-matched)."""

    def __init__(self, lm, lens, gamma, W, band=BAND, alpha: float = 1.0):
        import torch
        self.lm, self.lens, self.alpha = lm, lens, float(alpha)
        self.gamma, self.W = gamma.to(lm.device).float(), W
        self.band = [int(L) for L in band if L in lens.jacobians]
        assert self.band, "no band layer has a fitted J"
        self.J, self.cond = {}, {}
        for L in self.band:
            J = lens.jacobians[L].to(lm.device).float()
            self.J[L] = J
            self.cond[L] = float(torch.linalg.cond(J.double()))
        self.v_old = self.v_new = None
        self.cos_old_new = {}
        self.handles = []
        self.n_calls = 0

    def _g(self, token_id: int):
        return self.W[int(token_id)].float().to(self.lm.device) * self.gamma

    def token_dir(self, token_id: int, L: int):
        v = self.J[L].T @ self._g(token_id)
        return v / v.norm().clamp_min(1e-12)

    def random_dir(self, seed: int, L: int):
        import torch
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 7919 + int(L))
        v = torch.randn(self.lm.d_model, generator=g).to(self.lm.device)
        return v / v.norm()

    def set_pair(self, old_id: int, new_id=None, random_seed=None):
        import torch
        assert (new_id is None) != (random_seed is None), "give exactly one of new_id / random_seed"
        self.v_old = {L: self.token_dir(old_id, L) for L in self.band}
        self.v_new = {L: (self.token_dir(new_id, L) if new_id is not None else self.random_dir(random_seed, L)) for L in self.band}
        self.cos_old_new = {L: float(torch.dot(self.v_old[L], self.v_new[L])) for L in self.band}
        return self

    def _hook(self, L):
        vo, vn = self.v_old[L], self.v_new[L]

        def h(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            hf = hs.float()
            c = hf @ vo                                             # [.., S] coordinate along the old token's direction
            hn = (hf + self.alpha * c.unsqueeze(-1) * (vn - vo).unsqueeze(0)).to(hs.dtype)
            self.n_calls += 1
            return (hn,) + tuple(output[1:]) if isinstance(output, tuple) else hn
        return h

    def __enter__(self):
        assert self.v_old is not None and self.v_new is not None, "call set_pair() first"
        for L in self.band:
            self.handles.append(self.lm.layers[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for hd in self.handles:
            hd.remove()
        self.handles = []


# ------------------------------------------------------------------------------------- dry-run
def _dryrun():
    import os
    import tempfile
    import torch
    from infra.pg.stagediff_models import stub_member
    from infra.pg.stagediff_lens import AnatomyLensModel, capture, fit_corpus, fit_lens, lens_logits
    P, C = stub_member(bos_id=1, seed=0), stub_member(bos_id=1, seed=1)
    lmP, lmC = AnatomyLensModel(P), AnatomyLensModel(C)
    prompts = ["Q: 1+1?\nAnswer:", "Q: capital of France? Answer:"]
    base = C["fwd"].next_token_logits(prompts)
    # I1: transplant then restore is exact; transplanted child agrees with the parent on the head applied to
    # the same residual
    snap = head_snapshot(P)
    restore = transplant_head(C, snap)
    assert torch.equal(C["an"]["norm"].weight.detach().cpu(), snap["ln_f"])
    changed = C["fwd"].next_token_logits(prompts)
    assert not torch.allclose(changed, base)
    restore()
    assert torch.allclose(C["fwd"].next_token_logits(prompts), base)
    restore_r, st = random_head_control(C, snap, seed=3)
    assert abs(st["dW_fro"] - st["dWQ_fro"]) < 1e-3 * st["dW_fro"] and abs(st["dg_norm"] - st["dgQ_norm"]) < 1e-4
    assert not torch.allclose(C["fwd"].next_token_logits(prompts), base)
    restore_r()
    assert torch.allclose(C["fwd"].next_token_logits(prompts), base)
    # TRACE banks and the null
    docs = [f" = Doc {i} = \n" + ("word " * 300) + f"x{i}\n" for i in range(12)]
    seqs = fit_corpus(P["tok"], n_seqs=4, seq_len=24, seed=0, docs=docs)["seqs"]
    bP, bC = trace_bank(lmP, seqs, n_pos=5), trace_bank(lmC, seqs, n_pos=5)
    assert bP.shape == (4, 4, 5, 16) and bP.dtype == torch.float16
    d = delta_bar(bC, bP)
    assert d.shape == (4, 5, 16) and torch.allclose(d, bC.float().mean(0) - bP.float().mean(0))
    nb = null_bank(lmP, seqs, n=6, seed=0, min_pos=5)
    assert nb["acts"].shape == (6, 4, 16) and all(p >= 5 for _, p in nb["where"])
    rd = random_diffs(nb["acts"], n_draws=3, seed=0)
    pooled = d.mean(1)                                            # [L, d]
    rdm = norm_match(rd, pooled)
    assert torch.allclose(rdm.norm(dim=-1), pooled.norm(dim=-1).unsqueeze(0).expand(3, -1), atol=1e-4)
    # I2: after projecting u out at layer 2, the layer-2 residual has zero component along u
    u = unit_rows(pooled)                                         # [L, d]
    ids = C["tok"].encode(prompts[1], add_special_tokens=False)
    with ProjectOut(lmC.layers, {2: u[2]}) as po:
        a = capture(lmC, ids)
        assert po.n_calls >= 1
    assert (a[2] @ u[2]).abs().max() < 1e-4, (a[2] @ u[2]).abs().max()
    a0 = capture(lmC, ids)
    assert (a0[2] @ u[2]).abs().max() > 1e-4                      # the direction was genuinely present before
    # I3: fit a lens on the stub; a swap old→old is the identity; old→random moves the logits; the residual-space
    # change is exactly c (v̂_new − v̂_old) at the swapped layer
    with tempfile.TemporaryDirectory() as td:
        lens, _ = fit_lens(lmC, seqs, dim_batch=4, ckpt_path=os.path.join(td, "ck.pt"))
    g, W = lmC.head()
    sw = Swap(lmC, lens, g, W, band=(1, 2), alpha=1.0)
    assert set(sw.band) == {1, 2} and all(c > 0 for c in sw.cond.values())
    tid = int(base[1].argmax())
    with sw.set_pair(tid, new_id=tid):
        same = C["fwd"].next_token_logits(prompts)
    assert torch.allclose(same, base, atol=1e-4), (same - base).abs().max()
    assert all(abs(c - 1) < 1e-5 for c in sw.cos_old_new.values())
    with sw.set_pair(tid, random_seed=0):
        moved = C["fwd"].next_token_logits(prompts)
        assert sw.n_calls >= 2
    assert not torch.allclose(moved, base, atol=1e-4)
    sw1 = Swap(lmC, lens, g, W, band=(2,), alpha=1.0)
    with sw1.set_pair(tid, random_seed=0):
        a_sw = capture(lmC, ids)
    vo, vn = sw1.v_old[2], sw1.v_new[2]
    c = a0[2, -1] @ vo
    assert torch.allclose(a_sw[2, -1] - a0[2, -1], c * (vn - vo), atol=1e-4), (a_sw[2, -1] - a0[2, -1] - c * (vn - vo)).abs().max()
    assert abs(float((a_sw[2, -1] @ vo)) - float(c * (vn @ vo))) < 1e-4       # old coordinate after: c·cos(old,new)
    other = int(base[1].argmin())
    sw2 = Swap(lmC, lens, g, W, band=(1, 2), alpha=1.0).set_pair(tid, new_id=other)
    assert -1 <= sw2.cos_old_new[1] <= 1
    assert C["fwd"].next_token_logits(prompts).allclose(base)     # hooks removed
    print("PASS stagediff_interv dry-run (transplant/restore, same-norm head control, banks, ADL null norm-match, ProjectOut, Swap identity/move/pullback)")


if __name__ == "__main__":
    _dryrun()
