"""Weight-free CPU dry-run gate for ENDOW-R's transplant builder (endow spec §4.3, §9.1).

NO model, NO GPU — a tiny but COMPLETE 18-layer decoder state-dict of planted tensors. Proves every
ladder cell's invariant: SPEC has exactly the base spectrum with rotated factors; UNT ≡ base on the
U_unt-span inputs and ≡ random on the complement; layer-band cells copy complete blocks and nothing
else; ATTN/MLP copy their family only; EMB carries only base embeddings; the RAND anchor's embedding
toggles with random_anchor_embed; and every cell passes the 18×7 architecture assert.

Run (remote `pg` CPU — never locally):
    python -m pytest infra/pg/tests/test_endow_transplant.py -q
"""
from __future__ import annotations

import pytest
import torch

from infra.pg import endow_transplant as ET


def _w(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


def _tiny_base(d=8, kv=4, inter=16, vocab=20, n_layers=18):
    """A complete bare-ish gemma-2b decoder sd (model.* prefix) with distinct planted matrices."""
    sd = {"model.embed_tokens.weight": _w((vocab, d), 1),
          "lm_head.weight": _w((vocab, d), 2),
          "model.norm.weight": _w((d,), 3)}
    for i in range(n_layers):
        b = 10 + i * 10
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = _w((d, d), b + 1)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = _w((kv, d), b + 2)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = _w((kv, d), b + 3)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = _w((d, d), b + 4)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = _w((inter, d), b + 5)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = _w((inter, d), b + 6)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = _w((d, inter), b + 7)
        sd[f"model.layers.{i}.input_layernorm.weight"] = _w((d,), b + 8)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = _w((d,), b + 9)
    return sd


def _eq(a, b):
    return torch.equal(a, b)


def _diff(a, b):
    return not torch.allclose(a.float(), b.float(), atol=1e-6)


def test_base_cell_is_verbatim_and_asserts():
    base = _tiny_base()
    cell = ET.build_cell("BASE", base)
    assert set(cell) == set(base) and all(_eq(cell[k], base[k]) for k in base)
    assert len(ET.assert_ladder_cell(cell)) == 18 * 7


def test_rand_anchor_randomizes_matrices_keeps_norms_and_base_embed():
    base = _tiny_base()
    cell = ET.build_cell("RAND", base, seed=7)                    # default = option B (base embed fixed)
    q0 = "model.layers.0.self_attn.q_proj.weight"
    assert _diff(cell[q0], base[q0])                              # matrices random
    assert _eq(cell["model.norm.weight"], base["model.norm.weight"])                 # 1-D norms base
    assert _eq(cell["model.layers.0.input_layernorm.weight"],
               base["model.layers.0.input_layernorm.weight"])
    assert _eq(cell["model.embed_tokens.weight"], base["model.embed_tokens.weight"])   # base embed (B)
    assert _eq(cell["lm_head.weight"], base["lm_head.weight"])


def test_rand_anchor_embed_toggle_to_design_a():
    base = _tiny_base()
    a_design = ET.build_cell("RAND", base, seed=7, random_anchor_embed=True)            # option A
    assert _diff(a_design["model.embed_tokens.weight"], base["model.embed_tokens.weight"])
    assert _diff(a_design["lm_head.weight"], base["lm_head.weight"])


def test_emb_cell_only_base_embeddings():
    base = _tiny_base()
    cell = ET.build_cell("EMB", base, seed=7)
    assert _eq(cell["model.embed_tokens.weight"], base["model.embed_tokens.weight"])
    assert _eq(cell["lm_head.weight"], base["lm_head.weight"])
    assert _diff(cell["model.layers.3.self_attn.q_proj.weight"],
                 base["model.layers.3.self_attn.q_proj.weight"])          # everything else random


def test_attn_and_mlp_copy_their_family_only():
    base = _tiny_base()
    attn = ET.build_cell("ATTN", base, seed=7)
    mlp = ET.build_cell("MLP", base, seed=7)
    for i in (0, 9, 17):
        assert _eq(attn[f"model.layers.{i}.self_attn.v_proj.weight"],
                   base[f"model.layers.{i}.self_attn.v_proj.weight"])
        assert _diff(attn[f"model.layers.{i}.mlp.gate_proj.weight"],
                     base[f"model.layers.{i}.mlp.gate_proj.weight"])
        assert _eq(mlp[f"model.layers.{i}.mlp.down_proj.weight"],
                   base[f"model.layers.{i}.mlp.down_proj.weight"])
        assert _diff(mlp[f"model.layers.{i}.self_attn.q_proj.weight"],
                     base[f"model.layers.{i}.self_attn.q_proj.weight"])
    assert _eq(attn["model.embed_tokens.weight"], base["model.embed_tokens.weight"])   # +base embed


def test_layer_band_copies_complete_blocks_and_nothing_else():
    base = _tiny_base()
    early = ET.build_cell("EARLY", base, seed=7)
    for i in range(0, 6):                                            # in-band: complete block copied
        for p in ("self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.down_proj"):
            k = f"model.layers.{i}.{p}.weight"
            assert _eq(early[k], base[k]), k
    for i in range(6, 18):                                           # out-of-band: random
        assert _diff(early[f"model.layers.{i}.self_attn.q_proj.weight"],
                     base[f"model.layers.{i}.self_attn.q_proj.weight"])
    mid = ET.build_cell("MID", base, seed=7)
    assert _diff(mid["model.layers.0.self_attn.q_proj.weight"],
                 base["model.layers.0.self_attn.q_proj.weight"])
    assert _eq(mid["model.layers.7.mlp.up_proj.weight"], base["model.layers.7.mlp.up_proj.weight"])


def test_spec_matrix_preserves_spectrum_rotates_factors():
    W = _w((10, 6), 21).double()
    M = ET.spec_matrix(W, seed=5)
    assert torch.allclose(torch.linalg.svdvals(M), torch.linalg.svdvals(W), atol=1e-8)
    assert _diff(M, W)                                              # different singular directions


def test_spec_cell_matches_spectrum_everywhere():
    base = _tiny_base()
    cell = ET.build_cell("SPEC", base, seed=7)
    for k in ("model.layers.4.self_attn.o_proj.weight", "model.embed_tokens.weight"):
        assert torch.allclose(torch.linalg.svdvals(cell[k].double()),
                              torch.linalg.svdvals(base[k].double()), atol=1e-6), k
    assert _eq(cell["model.norm.weight"], base["model.norm.weight"])   # norms untouched


def test_unt_hybrid_projector_property():
    d, k = 8, 3
    g = torch.Generator().manual_seed(31)
    U = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64)).Q[:, :k]
    P = ET.projector_from_basis(U)
    Wb = torch.randn(6, d, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
    Wr = torch.randn(6, d, generator=torch.Generator().manual_seed(33), dtype=torch.float64)
    H = ET.unt_hybrid(Wb, Wr, P).double()
    assert torch.allclose(H @ U, Wb @ U, atol=1e-8)                # base action on the U_unt span
    Uperp = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(34),
                                        dtype=torch.float64)).Q
    Uperp = Uperp - U @ (U.transpose(0, 1) @ Uperp)               # project out span(U)
    assert torch.allclose(H @ Uperp, Wr @ Uperp, atol=1e-7)       # random action off the span


def test_unt_cell_hybridizes_qk_only():
    d = 8
    base = _tiny_base(d=d)
    g = torch.Generator().manual_seed(41)
    U = torch.linalg.qr(torch.randn(d, 2, generator=g, dtype=torch.float64)).Q[:, :2]
    cell = ET.build_cell("UNT", base, seed=7, U_unt=U)
    q0 = base["model.layers.0.self_attn.q_proj.weight"].double()
    assert torch.allclose(cell["model.layers.0.self_attn.q_proj.weight"].double() @ U,
                          q0 @ U, atol=1e-6)                        # base on U_unt input span
    assert _diff(cell["model.layers.0.self_attn.v_proj.weight"],
                 base["model.layers.0.self_attn.v_proj.weight"])    # v/o/mlp random
    assert _eq(cell["model.embed_tokens.weight"], base["model.embed_tokens.weight"])


def test_assert_rejects_incomplete_cell():
    base = _tiny_base()
    bad = ET.build_cell("RAND", base, seed=7)
    del bad["model.layers.5.mlp.up_proj.weight"]
    with pytest.raises(AssertionError):
        ET.assert_ladder_cell(bad)
