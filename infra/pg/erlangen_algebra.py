"""ERLANGEN Arm A — the intervention algebra (amendment A2).

Why this module exists at all: the spec's commutator ‖b(i∘j) − b(j∘i)‖ is IDENTICALLY ZERO for
weight-space edits on disjoint matrices (either order yields the same state-dict), so the P2
positive control could not fire. Arm A's interventions are therefore **state-dependent
activation-space** ones — mean-ablate span(Q) → h̄ on the residual entering layer L, with BOTH Q
and h̄ harvested from the model AS CURRENTLY CONFIGURED. Composition is sequential:

    i∘j  :=  harvest+install j on the base, then harvest i UNDER j, then evaluate with both live.

Order then matters exactly when information flows between the loci, which is the quantity the arm
exists to measure. Hook registration order is load-bearing: PyTorch runs forward-pre-hooks in
registration order, so installing j's ablation BEFORE registering the capture hook is what makes
"harvest i under j" mean what it says.

Kinds (the three subspace families at each locus, all self-contained — no external bank):
  topk : the top-k principal directions of the entering residual's second moment  (state-dependent)
  botk : the bottom-k principal directions — the low-σ band the untouched core lives in
         (invariant_core-TRIV: "a layer-local, q/k-shared, moderately-low-σ residual band"); a
         PROXY for that core, labelled as such, never as the core itself
  rand : a matched-random k-subspace — the null anchor (NOT state-dependent by construction)

No try/except (CLAUDE.md). Heavy imports inside functions.
"""
from __future__ import annotations

LOCI = (0, 2, 4, 6, 9, 12, 15, 17)
KINDS = ("topk", "botk", "rand")
K_SUB = 32
X_CAP = 4096
D = 2048


def intervention_ids():
    """The frozen K = 24 intervention list, in a fixed order."""
    return [f"L{L}|{k}" for L in LOCI for k in KINDS]


def parse_id(iv):
    L, k = iv.split("|")
    return int(L[1:]), k


# ===========================================================================
# Harvest + basis construction.
# ===========================================================================
def harvest_at(model, layers, L, forward_fn, batches, x_cap=X_CAP):
    """(h̄, X) at layer L UNDER whatever hooks are currently installed."""
    from infra.pg.inherit_pg1 import harvest
    got = harvest(layers, forward_fn, batches, [L], d=D, x_cap=x_cap)
    hbar, _trace, X, n = got[L]
    assert X is not None and X.shape[0] > K_SUB, f"harvest at L{L} banked {None if X is None else X.shape}"
    return hbar, X, n


def basis_from(X, hbar, kind, k=K_SUB, seed=0):
    """Q (d×k) orthonormal, fp32. topk/botk from the centred residual's principal directions;
    rand from a seeded QR (the null anchor)."""
    import torch
    if kind == "rand":
        g = torch.Generator().manual_seed(int(seed))
        A = torch.randn(D, k, generator=g, dtype=torch.float32)
        Q, _ = torch.linalg.qr(A)
        return Q[:, :k].contiguous()
    Xc = (X - hbar[None, :]).float()
    # right singular vectors of the centred rows = principal directions in residual space
    _U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    V = Vh.transpose(0, 1).contiguous()
    assert V.shape[0] == D, f"principal basis has width {V.shape[0]} ≠ {D}"
    assert S.numel() >= k, f"only {S.numel()} singular values at this locus, need {k}"
    if kind == "topk":
        return V[:, :k].contiguous()
    if kind == "botk":
        return V[:, -k:].contiguous()
    raise AssertionError(f"unknown kind {kind!r}")


def orthogonal_random_pair(k=K_SUB, seed=0):
    """Two k-subspaces that are exactly orthogonal — the exact-commutation null (Null-1). Mean
    ablation onto orthogonal subspaces commutes algebraically, so any measured commutator here is
    pipeline noise and nothing else."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    A = torch.randn(D, 2 * k, generator=g, dtype=torch.float32)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k].contiguous(), Q[:, k:2 * k].contiguous()


def subspace_shift(Qa, Qb):
    """1 − mean cos² of the principal angles between two k-subspaces: 0 = identical, 1 = orthogonal.
    This is how much an intervention's OWN parameters moved when harvested under another one."""
    import torch
    M = Qa.transpose(0, 1).float() @ Qb.float()
    s = torch.linalg.svdvals(M)
    return float(1.0 - (s ** 2).mean())


# ===========================================================================
# Install / compose.
# ===========================================================================
def install_ablation(layers, L, Q, hbar):
    """Register the mean-ablation pre-hook and return its handle (caller removes it)."""
    from infra.pg.inherit_core import make_pg1_pre_hook_ablate
    return layers[L].register_forward_pre_hook(
        make_pg1_pre_hook_ablate(Q, hbar), with_kwargs=True)


def realize(model, layers, ivs, forward_fn, batches, seed=0, reharvest=True, fixed=None):
    """Install an ORDERED list of interventions; ivs[0] is applied FIRST.

    reharvest=True  — the real composition: each intervention's (Q, h̄) is harvested with the
                      earlier ones LIVE, so order carries information flow.
    reharvest=False — the order-free control: every (Q, h̄) is harvested on the base model, then
                      all hooks are installed. Any commutator here is pipeline noise.
    fixed           — {iv: Q} pins a subspace (the same-locus orthogonal-pair nulls).

    Returns (handles, params); the caller must `remove(handles)`.
    """
    fixed = fixed or {}
    params, handles = {}, []
    if not reharvest:
        for iv in ivs:
            L, kind = parse_id(iv)
            hbar, X, _n = harvest_at(model, layers, L, forward_fn, batches)
            Q = fixed[iv] if iv in fixed else basis_from(X, hbar, kind, seed=seed + L)
            params[iv] = {"Q": Q, "hbar": hbar}
        for iv in ivs:
            params[iv]["handle_order"] = len(handles)
            handles.append(install_ablation(layers, parse_id(iv)[0],
                                            params[iv]["Q"], params[iv]["hbar"]))
        return handles, params
    for iv in ivs:
        L, kind = parse_id(iv)
        hbar, X, _n = harvest_at(model, layers, L, forward_fn, batches)
        Q = fixed[iv] if iv in fixed else basis_from(X, hbar, kind, seed=seed + L)
        params[iv] = {"Q": Q, "hbar": hbar}
        handles.append(install_ablation(layers, L, Q, hbar))
    return handles, params


def remove(handles):
    for h in handles:
        h.remove()


# ===========================================================================
# Amendment A6 (curiosity reserve) — does the low-σ activation band have a TYPE?
# ===========================================================================
def core_fingerprint(model, proc, device, packs, batch_size=8, base_vec=None, loci=(4, 9, 15)):
    """Fingerprint ACTIVATION subspaces with the same readout axis the weight grid uses, so their
    rows land in the same space and can be clustered beside components.

    Scope note, stated up front: `botk` is a PROXY for the universally-untouched q/k core
    (invariant_core-TRIV places that core in a moderately-low-σ, layer-local band) — it is not that
    core, which is defined by a consensus over 24 fine-tunes and is not recomputed here."""
    import torch

    from infra.pg import erlangen_probe as P
    from infra.pg.inherit_pg1 import decoder_layers
    layers = decoder_layers(model)
    harvest_items = packs["cap_real"][:16]

    def forward_fn(batch):
        inp = proc(text=[b["prompt"] for b in batch], images=[b["image"] for b in batch],
                   return_tensors="pt", padding="longest").to(device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            model(**inp)

    batches = [harvest_items[i:i + 8] for i in range(0, len(harvest_items), 8)]
    base = base_vec if base_vec is not None else P.vec(
        P.readout_vector(model, proc, device, packs, batch_size=batch_size))
    rows = {}
    for L in loci:
        hbar, X, _ = harvest_at(model, layers, L, forward_fn, batches)
        for kind in KINDS:
            Q = basis_from(X, hbar, kind, seed=L)
            h = install_ablation(layers, L, Q, hbar)
            r = P.readout_vector(model, proc, device, packs, batch_size=batch_size)
            h.remove()
            rows[f"activation.L{L}.{kind}"] = {
                "vec": P.vec(r), "delta": [x - y for x, y in zip(P.vec(r), base)],
                "half_a": P.half_vecs(r)[0], "half_b": P.half_vecs(r)[1]}
            print(f"[A6] activation L{L} {kind}: mean Δ="
                  f"{sum(rows[f'activation.L{L}.{kind}']['delta'])/len(base):+.4f}", flush=True)
    return {"rows": rows, "loci": list(loci), "kinds": list(KINDS), "k": K_SUB,
            "scope": "botk is a PROXY for the untouched q/k core (low-σ band), not that core"}
