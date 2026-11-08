"""COMWRITE Arm-B surgery builder (comwrite spec §2 surgery convention, §5) — pure state-dict
weight surgery, weight-free unit-tested (infra/pg/tests/test_comwrite_surgery.py) before any GPU
minute (endow_transplant precedent).

The frozen convention: P_wr projects ΔW = W_ft − W_base on the RESIDUAL-FACING side
(comwrite_core.RESIDUAL_SIDE) — ΔW′ = ΔW·(V_kV_kᵀ) for q/k/v/gate/up, ΔW′ = (U_kU_kᵀ)·ΔW for
o/down — at all 18 layers × all 7 types simultaneously (partial surgeries = a smaller `bases`
grid + matching `excluded`). Surgery in fp32 on state-dicts, cast back to the base dtype after.

Cells (§5.1/§5.2):
  identity  — empty projector (k=0): W′ = W_base exactly (harness-inertness gate).
  shared    — W′ = W_base + P_wr(ΔW)          (base + the common write).
  residual  — W′ = W_base + (ΔW − P_wr(ΔW))   (base + the idiosyncratic part).
  ft        — W′ = W_base + ΔW                (the FT rebuilt through the same pipeline;
              shared + residual ≡ ft by construction — the additivity dry-run identity).
  em_random — W′ = W_base + s·P_rand(ΔW), s per matrix so the injected energy MATCHES the
              shared cell's captured energy (the decisive null, §2).
  random    — W′ = W_base + P_rand(ΔW), unscaled equal-k (any-projection artifact gate).
  shuffle   — W′ = W_base + P_shuf(ΔW), the (L,t) basis taken from layer (L+roll)%18, same t
              (layer-structure control).

(L,t) pairs in `excluded` (failed the A1 existence gate, §8) carry NO commons: their shared
part is 0 and their residual part is the full ΔW — so shared+residual = ΔW holds globally.

Design rules (CLAUDE.md): NO try/except hiding; heavy imports inside functions; simplest surgery.
"""
from __future__ import annotations

from infra.pg import comwrite_core as CC
from infra.pg.gauge_common import N_LAYERS, PROJ, layer_index, matrix_type

CELLS = ("identity", "shared", "residual", "ft", "em_random", "random", "shuffle")


# ---------------------------------------------------------------------------
# key classification (prefix-robust, endow_transplant convention).
# ---------------------------------------------------------------------------
def is_target(key, w):
    """A 2-D decoder projection matrix inside a 'layers.{i}.' scope — the 18×7 surgery grid."""
    return (getattr(w, "ndim", None) == 2 and any(f".{p}." in key for p in PROJ)
            and "layers." in key)


def grid_of(sd):
    """{(L, mtype): key} for every surgery-target matrix in a decoder state-dict."""
    out = {}
    for k, w in sd.items():
        if is_target(k, w):
            out[(layer_index(k), matrix_type(k))] = k
    return out


def assert_full_grid(grid, n_layers=N_LAYERS):
    """The 18×7 architecture assert (endow_transplant.assert_ladder_cell convention)."""
    want = {(i, p) for i in range(n_layers) for p in PROJ}
    missing = sorted(want - set(grid))
    assert not missing, f"18×7 incomplete: {len(missing)} missing (e.g. {missing[:5]})"
    assert len(grid) == n_layers * 7, f"expected {n_layers * 7} matrices, got {len(grid)}"


# ---------------------------------------------------------------------------
# fp32 projection of one ΔW on the residual-facing side.
# ---------------------------------------------------------------------------
def project_delta(dW, U, mtype):
    """P_wr(ΔW) in fp32: input side ΔW·(UUᵀ); output side (UUᵀ)·ΔW. U (d,k) orthonormal."""
    import torch
    D = torch.as_tensor(dW, dtype=torch.float32)
    Ub = torch.as_tensor(U, dtype=torch.float32)
    d = CC.residual_dim(tuple(D.shape), mtype)
    assert Ub.ndim == 2 and Ub.shape[0] == d, \
        f"U{tuple(Ub.shape)} does not face d={d} of {mtype}"
    if CC.residual_side(mtype) == "input":
        return (D @ Ub) @ Ub.transpose(0, 1)
    return Ub @ (Ub.transpose(0, 1) @ D)


def _rand_basis(d, k, seed):
    from infra.pg.projector_ablation import random_basis
    import torch
    return random_basis(d, k, seed, "cpu", torch.float32)


def _seed_for(base_seed, L, mtype):
    """Deterministic per-(L,t) seed — endow_transplant's sha1 recipe (never hash())."""
    from infra.pg.endow_transplant import _seed_for as _et_seed
    return _et_seed(base_seed, f"{L}.{mtype}")


def shuffle_source(L, t, bases, roll, n_layers=N_LAYERS):
    """The shuffle cell's source pair for target (L,t): layer (L+roll)%18, same type,
    advancing past A1-excluded layers until an INCLUDED one is found — so the shuffle
    cell's injection SUPPORT matches the shared cell's exactly (a control that writes
    where the treatment never does would test the wrong thing). Returns the source (L,t)
    key into `bases`. RAISES if the roll cannot leave the target (roll≡0 mod 18 with a
    single included layer) — a degenerate control, never silently identity."""
    for j in range(1, n_layers):
        src = ((L + int(roll) * j) % n_layers, t)
        if src == (L, t):
            continue
        if src in bases:
            return src
    raise AssertionError(f"shuffle: no included source layer for {(L, t)} at roll {roll}")


# ---------------------------------------------------------------------------
# the builder.
# ---------------------------------------------------------------------------
def build_decoder(cell, base_dec, ft_dec, bases, *, excluded=(), seed=1337, roll=7,
                  n_layers=N_LAYERS):
    """A surgered decoder state-dict with EXACTLY base_dec's keys/shapes/dtypes + the energy
    bookkeeping. `bases` = {(L, mtype): U (d,k)} for every INCLUDED grid pair; `excluded` =
    the A1-failed pairs (no commons: shared→0, residual→ΔW). Non-target keys (embeddings,
    norms) are base VERBATIM — surgery touches only the 18×7 grid (§5.4 multimodal threat).

    Returns (state_dict, book): book has per-key {fro2, injected_fro2, captured_fro2,
    fraction, k} + totals — the §2 energy-capture bookkeeping the dry-run asserts."""
    import torch
    assert cell in CELLS, f"unknown cell {cell!r} (∈ {CELLS})"
    grid = grid_of(base_dec)
    assert_full_grid(grid, n_layers)
    ft_grid = grid_of(ft_dec)
    assert set(ft_grid) == set(grid), "base/ft decoder grids differ"
    excluded = {(int(L), t) for L, t in excluded}
    for lt in excluded:
        assert lt in grid, f"excluded pair {lt} not in the 18×7 grid"
    covered = set(bases) | excluded
    assert covered == set(grid), \
        f"bases ∪ excluded must cover the full grid exactly (missing {sorted(set(grid) - covered)[:5]}, " \
        f"extra {sorted(covered - set(grid))[:5]})"
    assert not (set(bases) & excluded), "a pair is both based and excluded"

    out, book = {}, {"cell": cell, "per_key": {}, "total_fro2": 0.0, "total_injected_fro2": 0.0,
                     "total_captured_fro2": 0.0}
    for key, w in base_dec.items():
        if not is_target(key, w):
            out[key] = w.detach().clone()
            continue
        L, t = layer_index(key), matrix_type(key)
        Wb = w.detach().to(torch.float32)
        dW = ft_dec[ft_grid[(L, t)]].detach().to(torch.float32) - Wb
        assert dW.shape == Wb.shape, f"{key}: base/ft shape mismatch"
        fro2 = float((dW ** 2).sum())

        # the shared (captured) part under U_wr — the energy target every cell books against.
        if (L, t) in excluded:
            shared = torch.zeros_like(dW)
            k_lt = 0
        else:
            U = bases[(L, t)]
            shared = project_delta(dW, U, t)
            k_lt = int(torch.as_tensor(U).shape[1])
        cap2 = float((shared ** 2).sum())

        if cell == "identity":
            inj = torch.zeros_like(dW)
        elif cell == "shared":
            inj = shared
        elif cell == "residual":
            inj = dW - shared
        elif cell == "ft":
            inj = dW
        elif (L, t) in excluded:
            inj = torch.zeros_like(dW)          # no commons here (§8): every null injects 0 too
        elif cell in ("em_random", "random"):
            d = CC.residual_dim(tuple(dW.shape), t)
            R = _rand_basis(d, k_lt, _seed_for(seed, L, t))
            pr = project_delta(dW, R, t)
            if cell == "random":
                inj = pr
            else:
                pr2 = float((pr ** 2).sum())
                assert pr2 > 0, f"{key}: zero random capture (degenerate ΔW?)"
                inj = pr * (cap2 / pr2) ** 0.5      # energy-matched to the captured ΔW energy
        elif cell == "shuffle":
            src = shuffle_source(L, t, bases, roll, n_layers)
            inj = project_delta(dW, bases[src], t)
        out[key] = (Wb + inj).to(w.dtype)
        inj2 = float((inj ** 2).sum())
        book["per_key"][key] = {"L": L, "type": t, "k": k_lt, "fro2": fro2,
                                "captured_fro2": cap2, "injected_fro2": inj2,
                                "fraction": (cap2 / fro2) if fro2 > 0 else 0.0}
        book["total_fro2"] += fro2
        book["total_injected_fro2"] += inj2
        book["total_captured_fro2"] += cap2
    assert_preserved(base_dec, out)
    return out, book


def assert_preserved(base_dec, out):
    """Exact key/shape/dtype preservation + every NON-target key bitwise base (the §2 dry-run
    key-set assert; a leaked embedding or norm would confound the multimodal forward)."""
    import torch
    assert set(out) == set(base_dec), "surgery changed the key set"
    for k, w in base_dec.items():
        assert tuple(out[k].shape) == tuple(w.shape), f"{k}: shape drift"
        assert out[k].dtype == w.dtype, f"{k}: dtype drift"
        if not is_target(k, w):
            assert torch.equal(out[k], w), f"{k}: non-target key modified"
