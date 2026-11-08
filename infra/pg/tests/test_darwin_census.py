"""Weight-free tests for the DARWIN census selection rule (§4 W0-3/W0-4).

Run: PYTHONPATH=. python3 infra/pg/tests/test_darwin_census.py

The selection rule decides which third-party checkpoints enter a census, so its failure mode is
not a crash but a WRONG ROSTER — either silently dropping valid derivatives or admitting repos
that cannot be loaded. Both are tested.
"""
import types

import numpy as np

from infra.pg import darwin_census as CEN
from infra.pg import humboldt_core as H


def _cfg(**kw):
    """A stand-in for a transformers config: attribute access, optional nested text_config."""
    return types.SimpleNamespace(**kw)


def test_base_geometry_unwraps_text_config():
    flat = CEN.base_geometry(_cfg(hidden_size=2048, num_hidden_layers=18, intermediate_size=16384,
                                  head_dim=256, num_attention_heads=8, num_key_value_heads=1,
                                  model_type="gemma"))
    assert flat["hidden_size"] == 2048 and flat["num_hidden_layers"] == 18
    nested = CEN.base_geometry(_cfg(model_type="paligemma", text_config=_cfg(
        hidden_size=2048, num_hidden_layers=18, model_type="gemma")))
    assert nested["hidden_size"] == 2048 and nested["model_type"] == "gemma"


def test_base_geometry_fails_loud_on_an_unreadable_config():
    try:
        CEN.base_geometry(_cfg(vocab_size=1000))
        raise SystemExit("a config with no hidden_size must fail loud")
    except AssertionError as ex:
        assert "hidden_size" in str(ex)


def test_prefilter_accepts_a_matching_derivative_and_rejects_a_foreign_one():
    geom = {"hidden_size": 3072, "num_hidden_layers": 28, "model_type": "llama"}
    ok, why = CEN.cfg_prefilter(_cfg(hidden_size=3072, num_hidden_layers=28, model_type="llama"),
                                geom)
    assert ok, why
    bad, why2 = CEN.cfg_prefilter(_cfg(hidden_size=2048, num_hidden_layers=28,
                                       model_type="llama"), geom)
    assert not bad and "hidden_size" in why2
    # A model of the right size but the wrong family is still foreign.
    bad2, why3 = CEN.cfg_prefilter(_cfg(hidden_size=3072, num_hidden_layers=28,
                                        model_type="qwen2"), geom)
    assert not bad2 and "model_type" in why3


def test_prefilter_rejects_quantised_checkpoints():
    geom = {"hidden_size": 2048, "num_hidden_layers": 18}
    ok, why = CEN.cfg_prefilter(_cfg(hidden_size=2048, num_hidden_layers=18,
                                     quantization_config={"quant_method": "gptq"}), geom)
    assert not ok and "quantised" in why


def test_loadable_entrypoint_rejects_compiled_exports():
    """journal D8: a Neuron/TensorRT export ships .safetensors that transformers cannot open.
    journal D12: the filter must match THIS ARM'S reader, which opens safetensors only."""
    assert CEN.loadable_entrypoint({"model.safetensors", "config.json"})
    assert CEN.loadable_entrypoint({"model.safetensors.index.json", "model-00001-of-2.safetensors"})
    # D13: an index is a POINTER, not data. The exact listing that burned a launch --
    # SipsaLabs/mistral-7b-v0.3-uc-v3-bpw3 -- has an index and no shard at all.
    dangling = {"config.json", "model.safetensors.index.json", "model.uc.bin",
                "ultracompress.json", "tokenizer.json"}
    assert not CEN.loadable_entrypoint(dangling)
    assert not CEN.legacy_bin_only(dangling)      # model.uc.bin is not pytorch_model.bin either
    # .bin-only repos are rejected -- the reader downloads *.safetensors and would find no shards.
    assert not CEN.loadable_entrypoint({"pytorch_model.bin"})
    assert CEN.legacy_bin_only({"pytorch_model.bin", "config.json"})
    assert CEN.legacy_bin_only({"pytorch_model.bin.index.json"})
    # ...and a repo with BOTH is fine, and is not classed as legacy.
    assert CEN.loadable_entrypoint({"pytorch_model.bin", "model.safetensors"})
    assert not CEN.legacy_bin_only({"pytorch_model.bin", "model.safetensors"})
    assert not CEN.legacy_bin_only({"config.json"})
    # The exact shape that burned a launch: sharded compiled graphs, no loadable entrypoint.
    neuron = {"config.json", "compiled/graph_0.safetensors", "compiled/graph_1.safetensors"}
    assert not CEN.loadable_entrypoint(neuron)
    assert any(f.endswith(".safetensors") for f in neuron)     # ...which a suffix test would pass
    assert not CEN.loadable_entrypoint({"config.json", "model.onnx"})
    assert not CEN.loadable_entrypoint(set())


def test_matrix_key_routes_attention_and_mlp():
    assert CEN.matrix_key("q_proj", 3) == "layers.3.self_attn.q_proj.weight"
    assert CEN.matrix_key("gate_proj", 7) == "layers.7.mlp.gate_proj.weight"
    assert CEN.matrix_key("down_proj", 0) == "layers.0.mlp.down_proj.weight"


def test_spikiness_separates_a_spiky_bank_from_a_haar_one():
    n, k = 512, 16
    flat = H.haar_subspace(n, k, 0)
    rng = np.random.default_rng(1)
    Z = np.zeros((n, k))
    hot = rng.choice(n, 32, replace=False)
    Z[hot] = rng.standard_normal((32, k))
    spiky = np.linalg.qr(Z)[0][:, :k]
    sf, ss = CEN.spikiness(flat), CEN.spikiness(spiky)
    assert ss["participation_ratio"] < sf["participation_ratio"] / 4
    assert ss["max_leverage"] > sf["max_leverage"]
    assert sf["haar_expected_leverage"] == k / n
    assert "matched-rotation" in sf["null"]


def test_min_for_census_is_the_spec_threshold():
    assert CEN.MIN_FTS_FOR_CENSUS == 12          # §4/W0-3


def test_highway_basis_takes_the_LARGEST_eigenvalue_columns():
    """journal D9: banks are stored ascending, so the highway top-r is the LAST r columns.
    Taking [:, :r] would ship the least-written directions of the most-written block."""
    import pathlib as _p, tempfile
    from infra.aws_pg.jobs.darwin_w1_surgery import highway_basis
    from infra.pg import humboldt_weights as _W
    with tempfile.TemporaryDirectory() as td:
        d = _p.Path(td)
        # Real ambient dim: humboldt_weights.load_bank asserts (IN_DIM[t], 32).
        # Columns tagged by row index so the slice actually taken is identifiable.
        U = np.zeros((H.IN_DIM["q_proj"], 32))
        for j in range(32):
            U[j, j] = 1.0
        np.save(d / _W.bank_name("q_proj", 0, "anti32"), U)
        got = highway_basis(d, "q_proj", 0, 8, have_wide=False)
        assert got.shape == (H.IN_DIM["q_proj"], 8)
        # The LAST 8 columns are the ones whose 1.0 sits at rows 24..31.
        assert [int(np.argmax(got[:, j])) for j in range(8)] == list(range(24, 32))
        # r == 32 is the degenerate case where both slices agree.
        assert np.allclose(highway_basis(d, "q_proj", 0, 32, have_wide=False), U)
        # The wide-bank path, and its width guard: ask for more rank than the basis carries.
        from infra.aws_pg.jobs.darwin_w1_surgery import WIDE_BANK
        np.save(d / _W.bank_name("q_proj", 0, WIDE_BANK), U[:, :16])
        wide = highway_basis(d, "q_proj", 0, 8, have_wide=True)
        assert [int(np.argmax(wide[:, j])) for j in range(8)] == list(range(8, 16))
        try:
            highway_basis(d, "q_proj", 0, 32, have_wide=True)
            raise SystemExit("a rank wider than the basis must fail loud")
        except AssertionError as ex:
            assert "exceeds available basis width" in str(ex)


def test_resolve_key_matches_by_suffix_across_prefix_conventions():
    """journal D10: checkpoints disagree on the prefix, so the key is found by suffix."""
    for prefix in ("model.", "language_model.model.", ""):
        keys = {f"{prefix}layers.{L}.self_attn.q_proj.weight" for L in range(4)}
        keys |= {f"{prefix}layers.{L}.mlp.gate_proj.weight" for L in range(4)}
        assert CEN.resolve_key(keys, "q_proj", 2) == f"{prefix}layers.2.self_attn.q_proj.weight"
        assert CEN.resolve_key(keys, "gate_proj", 0) == f"{prefix}layers.0.mlp.gate_proj.weight"


def test_resolve_key_fails_loud_on_zero_or_ambiguous_matches():
    try:
        CEN.resolve_key({"model.layers.0.self_attn.k_proj.weight"}, "q_proj", 0)
        raise SystemExit("a missing key must fail loud")
    except AssertionError as ex:
        assert "found 0" in str(ex)
    ambiguous = {"model.layers.1.self_attn.q_proj.weight",
                 "ema.model.layers.1.self_attn.q_proj.weight"}
    try:
        CEN.resolve_key(ambiguous, "q_proj", 1)
        raise SystemExit("an ambiguous key must fail loud rather than guess")
    except AssertionError as ex:
        assert "found 2" in str(ex)
    # Layer 1 must not match layer 11 -- the tail includes the dot-delimited index.
    assert CEN.resolve_key({"model.layers.1.self_attn.q_proj.weight",
                            "model.layers.11.self_attn.q_proj.weight"}, "q_proj", 11) \
        == "model.layers.11.self_attn.q_proj.weight"


def test_quantization_marker_catches_mlx_as_well_as_transformers():
    """journal D11: MLX writes `quantization`, not `quantization_config`; checking only the
    latter let a 4-bit bit-packed checkpoint through selection and it aborted a census."""
    assert CEN.quantization_marker(_cfg(quantization_config={"quant_method": "gptq"}))[0]
    assert CEN.quantization_marker(_cfg(quantization_config={"quant_method": "bitsandbytes"}))[0]
    is_q, desc = CEN.quantization_marker(_cfg(quantization={"group_size": 64, "bits": 4}))
    assert is_q and "quantization=" in desc
    assert not CEN.quantization_marker(_cfg(hidden_size=2048))[0]
    assert not CEN.quantization_marker(_cfg(quantization_config=None, quantization=None))[0]
    # ...and the prefilter must act on it.
    geom = {"hidden_size": 2048, "num_hidden_layers": 18}
    ok, why = CEN.cfg_prefilter(_cfg(hidden_size=2048, num_hidden_layers=18,
                                     quantization={"bits": 4}), geom)
    assert not ok and "quantised" in why


def test_recipe_class_matches_whole_tokens_not_substrings():
    """journal D16: `endpoints_compatible` contains 'dpo' (en-DPO-ints_compatible) and is on
    nearly every HF repo, so a substring rule labelled 14 of 14 derivatives DPO."""
    from infra.aws_pg.jobs.darwin_w0_misc import recipe_class
    # The exact tag list that caused the bug.
    real = ['transformers', 'safetensors', 'gemma', 'text-generation', 'conversational',
            'dataset:HuggingFaceH4/ultrachat_200k', 'base_model:google/gemma-2b',
            'base_model:finetune:google/gemma-2b', 'license:other', 'model-index',
            'text-generation-inference', 'endpoints_compatible', 'region:us']
    assert recipe_class("wandb/gemma-2b-zephyr-sft", real) == "sft"
    assert recipe_class("abhinand/gemma-2b-tamil", real) == "unknown"
    # Priority: a repo carrying both is the later pipeline stage.
    assert recipe_class("x/model-sft-dpo", []) == "dpo"
    assert recipe_class("x/model-ppo", []) == "rlhf"
    # The -it suffix fires on the id's last component only, never on the Italian language tag.
    assert recipe_class("google/gemma-2b-it", []) == "sft"
    assert recipe_class("someone/multilingual-model", ["it", "fr", "de"]) == "unknown"
    assert recipe_class("someone/it-is-a-model", []) == "unknown"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
