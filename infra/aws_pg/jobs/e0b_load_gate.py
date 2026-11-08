"""E0b-load — convention/reshape HALT gate for the ΔW₂ frame (pi-trajectory spec §E0b-load). CPU, fail-loud.

Confirm the π0 backbone tensors LOAD AS THE SAME COMPUTATIONAL MAP as PaliGemma-1 (same Q/K/V reshape,
key scheme, vocab order) — so ΔW₂ = W(π0-backbone) − W(PG1) is a real same-frame delta, not a JAX→PT
artifact. This REPLACES the original logit-equivalence gate (which would fail for the RIGHT reason — π0
genuinely trains the backbone — and confounds real change with a basis artifact). The gate here is
weight-CHANGE-INDEPENDENT: strict-load + a K/V value probe that distinguishes FT from a convention swap.

  pg run infra.aws_pg.jobs.e0b_load_gate --type r7i.4xlarge --no-weights --budget-min 60
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    import torch
    import transformers
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, GemmaModel, PaliGemmaForConditionalGeneration
    from infra.pg.entities import resolve_submodule_paths
    from infra.pg import pi0_port

    rep = {"source": "lerobot/pi0_base", "transformers_version": transformers.__version__}
    rep["pi0_sha"] = HfApi().model_info("lerobot/pi0_base", revision="main").sha

    # 1) load the port + remap the backbone LM (excludes vision tower + action expert; expert-leak asserts).
    pi0_path = hf_hub_download("lerobot/pi0_base", "model.safetensors")
    pi0_sd = load_file(pi0_path)
    backbone = pi0_port.remap_backbone_lm(pi0_sd)
    rep["n_backbone_keys"] = len(backbone)
    rep["expert_excluded_by_construction"] = not any("gemma_expert" in k for k in backbone)

    # 2) PG1 LM decoder (the OTHER endpoint) — bare GemmaModel keys.
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224", torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model
    pg_sd = lm.state_dict()

    # 3) KEY-SET identity — the port must provide EXACTLY PG1's decoder keys minus the documented embed drop.
    only_pg = sorted(set(pg_sd) - set(backbone))
    only_pi0 = sorted(set(backbone) - set(pg_sd))
    rep["key_set"] = {"only_in_pg1": only_pg, "only_in_pi0": only_pi0}
    embed_dropped = only_pg == ["embed_tokens.weight"]
    rep["embed_dropped_caveat"] = embed_dropped
    assert not only_pi0, f"π0 backbone has keys absent from PG1: {only_pi0[:5]}"
    assert set(only_pg) <= {"embed_tokens.weight"}, f"unexpected PG1-only keys: {only_pg}"

    # 4) STRICT-LOAD into a fresh Gemma-1 from PG1's exact text_config (the key-drift / convention guard).
    pg_cfg = AutoConfig.from_pretrained("google/paligemma-3b-pt-224")
    fresh = GemmaModel(pg_cfg.text_config)
    missing, unexpected = fresh.load_state_dict(backbone, strict=False)
    bad_missing = [k for k in missing if not ("embed_tokens" in k or "inv_freq" in k or "rotary" in k)]
    rep["strict_load"] = {"n_missing": len(missing), "n_unexpected": len(unexpected),
                          "missing_sample": list(missing)[:8], "unexpected": list(unexpected)[:8],
                          "bad_missing": bad_missing}
    assert not unexpected, f"FAIL_KEYS: π0 backbone has keys the fresh Gemma-1 rejects: {unexpected[:5]}"
    assert not bad_missing, f"FAIL_KEYS: missing non-embed decoder keys: {bad_missing[:5]}"

    # 5) K/V VALUE probe (NOT shape — a within-head permutation / K↔V slab-swap is shape-identical). The
    #    correct convention keeps the identity mapping CLOSEST (its residual = the FT ~0.3), a swap/permute
    #    reads near √2. Probe layers {0,8,17}; (256,2048) shape kept as a cheap precondition.
    kv = {}
    for li in (0, 8, 17):
        kk = f"layers.{li}.self_attn.k_proj.weight"
        vk = f"layers.{li}.self_attn.v_proj.weight"
        assert list(backbone[kk].shape) == [256, 2048] == list(pg_sd[kk].shape), f"k_proj shape {li}"
        kv[li] = pi0_port.kv_convention_probe(pg_sd[kk], pg_sd[vk], backbone[kk], backbone[vk])
    rep["kv_probe"] = kv
    kv_ok = all(v["identity_is_closest"] for v in kv.values())

    # vocab: embed dropped → not gate-verifiable from weights; record the config value.
    rep["vocab_note"] = {"pg1_text_vocab": int(getattr(pg_cfg.text_config, "vocab_size", -1)),
                         "embedding_delta": "NOT gate-verified (port drops tied embed_tokens; openpi #669)"}

    # verdict.
    if kv_ok and embed_dropped and not unexpected and not bad_missing:
        verdict, fail_action = "PASS_same_frame", None
    else:
        verdict, fail_action = "FAIL", "attempt JAX-converter source (same converter both endpoints); else π0-FAST-vs-π0 fallback; stamp ΔW₂ converter-limited"
    rep["verdict"] = verdict
    rep["fail_action"] = fail_action

    (out / "e0b_load_gate.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78)
    print(f"strict-load: unexpected={len(unexpected)} missing={len(missing)} (embed-only={embed_dropped})")
    for li, v in kv.items():
        print(f"  L{li:2d} K/V probe: id={v['d_identity']:.3f} kv_swap={v['d_kv_swap']:.3f} "
              f"rowperm={v['d_rowperm']:.3f} identity_closest={v['identity_is_closest']}")
    print("=" * 78)
    print(f"E0B_LOAD_VERDICT {verdict}", json.dumps({"verdict": verdict, "fail_action": fail_action}))


if __name__ == "__main__":
    main()
