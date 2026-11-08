"""ENDOW-R transplant builder (endow spec §4.3) — gain-of-function hybrid inits.

Pure state-dict weight surgery, weight-free unit-tested (infra/pg/tests/test_endow_transplant.py)
before any GPU minute. Builds the 8 ladder cells by copying NAMED structure from base gemma-2b
(google/gemma-2b — Gemma-1, 18L, d=2048; the endowment) into an otherwise raw-random gemma-2b
decoder, so the relearn probe measures each named component's MARGINAL re-learnability value:

  EMB   — base embeddings (+tied head) only; every decoder matrix random.
  ATTN  — all attention projections (q,k,v,o), all layers, + base embeddings.
  MLP   — all MLP blocks (gate,up,down), all layers, + base embeddings.
  EARLY — layers 0–5 complete (attn+MLP) + base embeddings.
  MID   — layers 6–11 complete + base embeddings.
  LATE  — layers 12–17 complete + base embeddings.
  SPEC  — statistics-only: every matrix (embeddings included) → U_r Σ V_rᵀ with base singular
          values Σ and seeded Haar U_r, V_r (the statistics-matched-displaced hypothesis).
  UNT   — base q/k weights restricted to the conserved primitive's span
          (W = W_rand·(I−P_U) + W_base·P_U on q/k inputs) + base embeddings; else random.
Plus anchors: BASE (ceiling, verbatim) and RAND (raw-random zero anchor).

RESOLVED DESIGN (Phase-2 Finding-4, ruled option B): base embeddings are a FIXED SHARED INTERFACE
across every cell (`random_anchor_embed=False`, the default). This isolates the DECODER endowment in
R0 (π0 — whose embeddings the port drops — is loaded against the same base embeddings as scratch and
base), removes the embedding confound, and reuses the existing swapback harness unchanged. Under B
the EMB cell ≡ the raw-random anchor and is dropped from the ladder (re-scoped to "the embedding
interface, held fixed as a control, not a treatment"). The builder still returns a FULL state-dict
(embeddings + lm_head + norms + 18×7 matrices) so a job may strict-load a cell directly OR route its
decoder subset through swap_back.swapback_state_dict — both yield base embeddings under B. The flag is
kept because it is the only lever for the alternative design A (random-embedding anchor + a separate
base-embedding scratch for R0), should the author ever want the EMB cell back as a treatment.

Conventions (frozen with the program): 1-D params (RMSNorm scales / biases) stay at BASE in every
cell — randomizing a scale param destroys the forward without testing any reasoning (the SAV/VLAT
raw-random convention, swap_back.random_decoder_state_dict). 2-D matrices carry the per-matrix
std-matched Gaussian for their random init. Every cell passes the 18×7 architecture assert.

Design rules (CLAUDE.md): NO try/except hiding; heavy imports inside functions; simplest surgery.
"""
from __future__ import annotations

CELLS = ("BASE", "RAND", "EMB", "ATTN", "MLP", "EARLY", "MID", "LATE", "SPEC", "UNT")
BANDS = {"EARLY": range(0, 6), "MID": range(6, 12), "LATE": range(12, 18)}
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Key classification (robust to a leading 'model.' prefix — substring match, the
# _bare_decoder / vlat_retention convention).
# ---------------------------------------------------------------------------
def is_embed(k):
    return "embed_tokens" in k or "lm_head" in k


def is_matrix(k, w):
    """A 2-D decoder matrix (attn/MLP projection) — NOT an embedding, NOT a 1-D scale."""
    return getattr(w, "ndim", None) == 2 and not is_embed(k)


def layer_of(k):
    """The decoder layer index in a 'layers.{i}.' key, or None (final norm / embeddings)."""
    import re
    m = re.search(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def is_attn(k):
    return ".self_attn." in k


def is_mlp(k):
    return ".mlp." in k


def is_qk(k):
    return ".self_attn.q_proj." in k or ".self_attn.k_proj." in k


# ---------------------------------------------------------------------------
# Random init (per-matrix std-matched Gaussian; the swap_back convention).
# ---------------------------------------------------------------------------
def random_like(w, seed):
    import torch
    g = torch.Generator().manual_seed(int(seed))
    std = float(w.detach().float().std().item())
    return (torch.randn(tuple(w.shape), generator=g, dtype=torch.float32) * std).to(w.dtype)


def _seed_for(base_seed, k):
    """A per-key deterministic seed (so two matrices never share a random draw)."""
    import hashlib
    return (int(base_seed) * 1000003 + int(hashlib.sha1(k.encode()).hexdigest()[:8], 16)) % (2 ** 31)


# ---------------------------------------------------------------------------
# SPEC — statistics-matched-displaced matrix (base singular values, Haar factors).
# ---------------------------------------------------------------------------
def spec_matrix(w, seed):
    """U_r Σ V_rᵀ with Σ = base singular values and U_r, V_r seeded Haar-orthonormal — a matrix
    with EXACTLY w's spectrum but random singular directions. r = min(out, in)."""
    import torch
    W = torch.as_tensor(w, dtype=torch.float64)
    out_d, in_d = W.shape
    r = min(out_d, in_d)
    s = torch.linalg.svdvals(W)                                    # base spectrum (r,)
    g = torch.Generator().manual_seed(int(seed))
    Ur, _ = torch.linalg.qr(torch.randn(out_d, r, generator=g, dtype=torch.float64))
    g2 = torch.Generator().manual_seed(int(seed) + 1)
    Vr, _ = torch.linalg.qr(torch.randn(in_d, r, generator=g2, dtype=torch.float64))
    M = (Ur[:, :r] * s) @ Vr[:, :r].transpose(0, 1)
    return M.to(getattr(w, "dtype", torch.float32))


# ---------------------------------------------------------------------------
# UNT — base q/k restricted to the conserved primitive's input span.
# ---------------------------------------------------------------------------
def unt_hybrid(w_base, w_rand, P_U):
    """W = W_rand·(I−P_U) + W_base·P_U — base action on the U_unt INPUT span, random on the
    complement. P_U (in,in) is the projector onto span(U_unt) in the q/k input (residual) space."""
    import torch
    Wb = torch.as_tensor(w_base, dtype=torch.float64)
    Wr = torch.as_tensor(w_rand, dtype=torch.float64)
    P = torch.as_tensor(P_U, dtype=torch.float64)
    in_d = Wb.shape[1]
    assert P.shape == (in_d, in_d), f"P_U {tuple(P.shape)} != (in,in)=({in_d},{in_d})"
    I = torch.eye(in_d, dtype=torch.float64)
    return (Wr @ (I - P) + Wb @ P).to(getattr(w_base, "dtype", torch.float32))


def projector_from_basis(U):
    """P = U·Uᵀ from an orthonormal input basis U (in,k) — the U_unt span projector."""
    import torch
    U = torch.as_tensor(U, dtype=torch.float64)
    return U @ U.transpose(0, 1)


# ---------------------------------------------------------------------------
# The builder.
# ---------------------------------------------------------------------------
def build_cell(cell, base_sd, seed=1337, U_unt=None, random_anchor_embed=False):
    """A hybrid decoder state-dict with EXACTLY base_sd's keys/shapes, per the §4.3 cell rule.
    base_sd is the base gemma-2b decoder (any key convention; classified by substring). 1-D params
    stay base; 2-D matrices are base / random / spec / unt-hybrid per the cell. Fail-loud."""
    import torch
    assert cell in CELLS, f"unknown cell {cell!r} (∈ {CELLS})"
    if cell == "BASE":
        return {k: v.detach().clone() for k, v in base_sd.items()}

    # UNT needs the primitive span as an input projector, applied per q/k matrix (its in-dim = d).
    # U_unt may be a single basis (applied to every q/k layer) OR a per-layer dict {L: (in,k)}.
    P_U, P_U_by_layer = None, None
    if cell == "UNT":
        assert U_unt is not None, "UNT cell needs the U_unt basis (in_dim, k) or per-layer dict"
        if isinstance(U_unt, dict):
            P_U_by_layer = {int(L): projector_from_basis(U) for L, U in U_unt.items()}
        else:
            P_U = projector_from_basis(U_unt)

    out = {}
    for k, w in base_sd.items():
        wseed = _seed_for(seed, k)
        if cell == "SPEC":
            out[k] = spec_matrix(w, wseed) if getattr(w, "ndim", None) == 2 else w.detach().clone()
            continue
        if is_embed(k):
            base_embed = not (cell == "RAND" and random_anchor_embed)   # base everywhere except RAND anchor
            out[k] = w.detach().clone() if base_embed else random_like(w, wseed)
            continue
        if not is_matrix(k, w):                                          # 1-D norms/scales: base
            out[k] = w.detach().clone()
            continue
        if _copy_matrix(cell, k):
            out[k] = w.detach().clone()
        elif cell == "UNT" and is_qk(k):
            P = P_U_by_layer[layer_of(k)] if P_U_by_layer is not None else P_U
            out[k] = unt_hybrid(w, random_like(w, wseed), P)
        else:
            out[k] = random_like(w, wseed)
    assert_ladder_cell(out)
    return out


def _copy_matrix(cell, k):
    """Does this cell copy the base 2-D matrix at key k verbatim?"""
    if cell in ("RAND", "EMB", "UNT"):
        return False
    if cell == "ATTN":
        return is_attn(k)
    if cell == "MLP":
        return is_mlp(k)
    if cell in BANDS:
        L = layer_of(k)
        return L is not None and L in BANDS[cell]
    return False


# ---------------------------------------------------------------------------
# Architecture assert (18×7, prefix-robust) — a partial swap would leak base weights.
# ---------------------------------------------------------------------------
def matrix_keys(sd):
    """{(layer, proj): key} for every 2-D attn/MLP projection in the state-dict."""
    out = {}
    for k, w in sd.items():
        if not is_matrix(k, w):
            continue
        L = layer_of(k)
        for p in PROJ:
            if f".{p}." in k:
                out[(L, p)] = k
    return out


def assert_ladder_cell(sd, n_layers=18):
    """Explicit 18×7 presence (a key-intersection would not catch a gap) + a base embedding key."""
    mk = matrix_keys(sd)
    missing = sorted({(i, p) for i in range(n_layers) for p in PROJ} - set(mk))
    assert not missing, f"18×7 incomplete: {len(missing)} missing (e.g. {missing[:5]})"
    assert len(mk) == n_layers * 7, f"expected {n_layers * 7} matrices, got {len(mk)}"
    assert any(is_embed(k) for k in sd), "no embedding key in the ladder cell state-dict"
    return sorted(mk.values())
