"""Weight-free dry-run gate for the COMWRITE surgery builder (comwrite spec §10.1) — a tiny but
COMPLETE 18-layer decoder pair (base, ft). Proves: projector idempotence, shared+residual
additivity (≡ ft), exact 18×7 key/shape/dtype preservation with non-target keys bitwise base,
the identity cell's harness inertness, em-random energy matching, shuffle's layer roll, the
excluded-pair rule, and the energy bookkeeping.

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_comwrite_surgery.py -q
"""
from __future__ import annotations

import pytest
import torch

from infra.pg import comwrite_core as CC
from infra.pg import comwrite_surgery as CS

D, KV, INTER, K = 8, 4, 16, 2      # tiny residual dim d=8; every basis lives in d


def _w(shape, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)


def _decoder(seed0, dtype=torch.float32, n_layers=18):
    """A complete bare decoder sd (model.* prefix; the endow_transplant tiny-base pattern)."""
    sd = {"model.embed_tokens.weight": _w((20, D), seed0 + 1, dtype),
          "model.norm.weight": _w((D,), seed0 + 2, dtype)}
    for i in range(n_layers):
        b = seed0 + 10 + i * 10
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = _w((D, D), b + 1, dtype)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = _w((KV, D), b + 2, dtype)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = _w((KV, D), b + 3, dtype)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = _w((D, D), b + 4, dtype)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = _w((INTER, D), b + 5, dtype)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = _w((INTER, D), b + 6, dtype)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = _w((D, INTER), b + 7, dtype)
        sd[f"model.layers.{i}.input_layernorm.weight"] = _w((D,), b + 8, dtype)
    return sd


def _bases(seed=0, k=K):
    """A full 18×7 basis grid — every U is (d=8, k) orthonormal (the residual-facing side)."""
    out = {}
    for L in range(18):
        for j, t in enumerate(CS.PROJ):
            g = torch.Generator().manual_seed(seed + L * 7 + j)
            Q, _ = torch.linalg.qr(torch.randn(D, k, generator=g))
            out[(L, t)] = Q
    return out


BASE, FT, BASES = _decoder(0), _decoder(5000), _bases()


def _cell(name, **kw):
    return CS.build_decoder(name, BASE, FT, BASES, **kw)


# ---------------------------------------------------------------------------
# projector.
# ---------------------------------------------------------------------------
def test_project_delta_idempotent_both_sides():
    dW = _w((D, D), 42)
    for t in ("q_proj", "o_proj"):
        P1 = CS.project_delta(dW, BASES[(0, t)], t)
        assert torch.allclose(CS.project_delta(P1, BASES[(0, t)], t), P1, atol=1e-6)


def test_project_delta_side_convention():
    dW = _w((D, INTER), 43)                                   # down: out=8 is residual side
    U = BASES[(0, "down_proj")]
    assert torch.allclose(CS.project_delta(dW, U, "down_proj"), U @ (U.T @ dW), atol=1e-6)
    dWg = _w((INTER, D), 44)                                  # gate: in=8 is residual side
    Ug = BASES[(0, "gate_proj")]
    assert torch.allclose(CS.project_delta(dWg, Ug, "gate_proj"), (dWg @ Ug) @ Ug.T, atol=1e-6)
    with pytest.raises(AssertionError):
        CS.project_delta(_w((D, INTER), 45), torch.eye(INTER)[:, :2], "down_proj")


# ---------------------------------------------------------------------------
# cells.
# ---------------------------------------------------------------------------
def test_identity_cell_is_base_exactly():
    sd, book = _cell("identity")
    assert all(torch.equal(sd[k], BASE[k]) for k in BASE)
    assert book["total_injected_fro2"] == 0.0


def test_ft_cell_rebuilds_the_ft():
    sd, book = _cell("ft")
    q5 = "model.layers.5.self_attn.q_proj.weight"
    assert torch.allclose(sd[q5], FT[q5], atol=1e-5)
    assert book["total_injected_fro2"] == pytest.approx(book["total_fro2"], rel=1e-6)


def test_shared_plus_residual_is_additive_to_ft():
    sh, _ = _cell("shared")
    re, _ = _cell("residual")
    for key in CS.grid_of(BASE).values():
        rebuilt = sh[key] + re[key] - BASE[key]
        assert torch.allclose(rebuilt, FT[key], atol=1e-5), key


def test_shared_injection_lives_in_span_and_residual_off_span():
    sh, _ = _cell("shared")
    re, _ = _cell("residual")
    key = "model.layers.3.mlp.down_proj.weight"
    U = BASES[(3, "down_proj")]
    inj_sh = sh[key] - BASE[key]
    inj_re = re[key] - BASE[key]
    assert torch.allclose(CS.project_delta(inj_sh, U, "down_proj"), inj_sh, atol=1e-5)
    assert float(CS.project_delta(inj_re, U, "down_proj").norm()) < 1e-4


def test_em_random_matches_captured_energy_and_random_does_not():
    em, em_book = _cell("em_random")
    rd, rd_book = _cell("random")
    for key, rec in em_book["per_key"].items():
        assert rec["injected_fro2"] == pytest.approx(rec["captured_fro2"], rel=1e-4), key
    diffs = [abs(r["injected_fro2"] - r["captured_fro2"]) / max(r["captured_fro2"], 1e-12)
             for r in rd_book["per_key"].values()]
    assert max(diffs) > 0.01                                  # unscaled ⇒ not energy-matched
    key = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.allclose(em[key], rd[key], atol=1e-6)


def test_shuffle_uses_the_rolled_layer_basis():
    sh, _ = _cell("shuffle", roll=7)
    key = "model.layers.2.self_attn.q_proj.weight"
    dW = FT[key] - BASE[key]
    want = BASE[key] + CS.project_delta(dW, BASES[(9, "q_proj")], "q_proj")   # (2+7)%18
    assert torch.allclose(sh[key], want, atol=1e-5)
    with pytest.raises(AssertionError):
        _cell("shuffle", roll=18)                              # maps every pair to itself


def test_shuffle_support_matches_shared_under_exclusions():
    """The structure control must inject EXACTLY where the shared cell does: excluded
    TARGETS get 0 (like shared), and an excluded SOURCE layer is rolled past to the next
    included layer of the same type."""
    ex = [(9, "q_proj"), (4, "v_proj")]
    bases = {lt: U for lt, U in BASES.items() if lt not in set(ex)}
    sh, _ = CS.build_decoder("shuffle", BASE, FT, bases, excluded=ex, roll=7)
    # excluded target (4, v_proj): zero injection, same as shared.
    kv = "model.layers.4.self_attn.v_proj.weight"
    assert torch.equal(sh[kv], BASE[kv])
    # target (2, q_proj) rolls to (9, q_proj) which is EXCLUDED → advances to (16, q_proj).
    assert CS.shuffle_source(2, "q_proj", bases, 7) == (16, "q_proj")
    kq = "model.layers.2.self_attn.q_proj.weight"
    want = BASE[kq] + CS.project_delta(FT[kq] - BASE[kq], BASES[(16, "q_proj")], "q_proj")
    assert torch.allclose(sh[kq], want, atol=1e-5)


def test_excluded_pair_carries_no_commons():
    ex = [(4, "v_proj")]
    bases = {lt: U for lt, U in BASES.items() if lt != (4, "v_proj")}
    sh, book = CS.build_decoder("shared", BASE, FT, bases, excluded=ex)
    re, _ = CS.build_decoder("residual", BASE, FT, bases, excluded=ex)
    key = "model.layers.4.self_attn.v_proj.weight"
    assert torch.equal(sh[key], BASE[key])                     # shared → 0 there
    assert torch.allclose(re[key], FT[key], atol=1e-5)         # residual → full ΔW
    assert book["per_key"][key]["k"] == 0


# ---------------------------------------------------------------------------
# preservation + grid discipline.
# ---------------------------------------------------------------------------
def test_key_shape_dtype_preserved_and_non_targets_bitwise_base():
    base16 = {k: v.to(torch.bfloat16) for k, v in BASE.items()}
    ft16 = {k: v.to(torch.bfloat16) for k, v in FT.items()}
    sd, _ = CS.build_decoder("shared", base16, ft16, BASES)
    assert set(sd) == set(base16)
    for k in base16:
        assert sd[k].dtype == torch.bfloat16 and sd[k].shape == base16[k].shape
    assert torch.equal(sd["model.embed_tokens.weight"], base16["model.embed_tokens.weight"])
    assert torch.equal(sd["model.norm.weight"], base16["model.norm.weight"])
    assert torch.equal(sd["model.layers.0.input_layernorm.weight"],
                       base16["model.layers.0.input_layernorm.weight"])


def test_incomplete_basis_grid_fails_loud():
    bases = dict(BASES)
    del bases[(11, "up_proj")]                                 # neither based nor excluded
    with pytest.raises(AssertionError):
        CS.build_decoder("shared", BASE, FT, bases)
    with pytest.raises(AssertionError):                        # both based AND excluded
        CS.build_decoder("shared", BASE, FT, BASES, excluded=[(11, "up_proj")])


def test_incomplete_18x7_decoder_fails_loud():
    broken = {k: v for k, v in BASE.items() if "layers.9.mlp.up_proj" not in k}
    with pytest.raises(AssertionError):
        CS.build_decoder("shared", broken, FT, BASES)


def test_unknown_cell_fails_loud():
    with pytest.raises(AssertionError):
        _cell("frankenstein")
