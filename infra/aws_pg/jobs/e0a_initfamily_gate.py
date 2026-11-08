"""E0a — init-family HALT gate (pi-trajectory spec §E0a).

Certify PaliGemma-1's LM tower was initialised from Gemma-1-2B, so ΔW₁ = PG1.LM − Gemma-1-2B is a real
FT#1 edit and not an artifact of the wrong base. Renamed from "byte-verify": a Frobenius cut certifies
init-FAMILY, never byte-identity (gated repo, v1.0/v1.1 ambiguity — open Q1). CPU, fp32, fail-loud.

The discriminator: per-decoder-matrix relative Frobenius ‖W_PG1 − W_base‖/‖W_base‖. A fine-tune is a
SMALL relative perturbation, so the candidate base reads FAR below a wrong base. The wrong-base FLOOR is
MEASURED in the same job from a config-random Gemma-1 (same shapes, unrelated weights) — NOT hardcoded.
RMSNorm weights are reported on w_eff = 1.0 + stored (HF Gemma stores them zero-centred, forward (1+w)·x,
so the raw ‖Δw‖/‖w‖ divides by ≈0). bf16 storage floors a true same-init residual at ~1e-3 relative.

  pg run infra.aws_pg.jobs.e0a_initfamily_gate --type r7i.4xlarge --no-weights \
      --args "--seed 1337 --base-revision main" --budget-min 60
"""
import argparse
import json
import pathlib


PG1_REPO = "google/paligemma-3b-pt-224"
BASE_REPO = "google/gemma-2b"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _norm_keys(sd):
    """Strip a leading 'model.' so a GemmaForCausalLM .model wrapper aligns with a bare GemmaModel."""
    return {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}


def _lm_decoder_sd(model):
    """The Gemma decoder state_dict of PG1's LM tower (descend through any lm_head wrapper)."""
    from infra.pg.entities import resolve_submodule_paths
    lm = model.get_submodule(resolve_submodule_paths(model)["language_model"])
    if hasattr(lm, "lm_head") and hasattr(lm, "model"):
        lm = lm.model                                   # GemmaForCausalLM → GemmaModel decoder
    return _norm_keys(lm.state_dict())


def _rel_frob(wb, wa):
    import torch
    a = wb.detach().to(torch.float64)
    b = wa.detach().to(torch.float64)
    nb = a.norm().item()
    return (b - a).norm().item() / nb if nb > 0 else float("nan")


def _stats(xs):
    import numpy as np
    a = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(0)
    boots = np.array([np.median(a[rng.integers(0, a.size, a.size)]) for _ in range(1000)])
    return {"n": int(a.size), "median": float(np.median(a)), "max": float(a.max()),
            "mean": float(a.mean()), "ci95_median": [float(np.percentile(boots, 2.5)),
                                                      float(np.percentile(boots, 97.5))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--base-revision", default="main")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()                                   # FIRST — gated metadata 401s otherwise
    import torch
    from huggingface_hub import HfApi
    from transformers import (AutoConfig, AutoModelForCausalLM, GemmaForCausalLM,
                              PaliGemmaForConditionalGeneration)
    from infra.pg.gain_routing import matched_decoder_weight_matrices
    from infra.pg.weight_diff import compute_weight_diff

    rep = {"pg1_repo": PG1_REPO, "base_repo": BASE_REPO, "base_revision": a.base_revision}

    # 0) provenance pin (explicit revisions — the v1.0/v1.1 ambiguity is the whole point).
    api = HfApi()
    rep["pg1_sha"] = api.model_info(PG1_REPO, revision="main").sha
    rep["base_sha"] = api.model_info(BASE_REPO, revision=a.base_revision).sha
    pg_cfg = AutoConfig.from_pretrained(PG1_REPO)
    base_cfg = AutoConfig.from_pretrained(BASE_REPO, revision=a.base_revision)
    txt_cfg = getattr(pg_cfg, "text_config", pg_cfg)
    rep["pg1_text_model_type"] = getattr(txt_cfg, "model_type", None)
    rep["base_model_type"] = getattr(base_cfg, "model_type", None)
    # Gemma-1, NOT Gemma-2 (no final_logit_softcapping / query_pre_attn_scalar).
    assert rep["base_model_type"] == "gemma", f"base is not Gemma-1: {rep['base_model_type']}"
    assert rep["pg1_text_model_type"] == "gemma", f"PG1 text is not Gemma-1: {rep['pg1_text_model_type']}"

    # 1) load both fp32 on CPU (NO torch.cuda assert).
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        PG1_REPO, torch_dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_REPO, revision=a.base_revision, torch_dtype=torch.float32,
        device_map="cpu", attn_implementation="eager")
    pg_sd = _lm_decoder_sd(pg)
    base_sd = _norm_keys(base.model.state_dict())

    # 2) structural key/shape audit — matched_decoder_weight_matrices silently INTERSECTS keys, so it
    #    will not catch a missing layer; assert 18×7 AND every (type, layer 0..17) present explicitly.
    matched = matched_decoder_weight_matrices(pg_sd, base_sd)
    n_layers = int(getattr(base_cfg, "num_hidden_layers"))
    rep["n_layers"] = n_layers
    expect = {f"layers.{i}.{('self_attn' if p.startswith(('q','k','v','o')) else 'mlp')}.{p}.weight"
              for i in range(n_layers) for p in PROJ}
    missing = sorted(expect - set(matched.keys()))
    rep["structural"] = {"n_matched": len(matched), "expected": len(expect),
                         "missing": missing[:20], "n_missing": len(missing)}
    assert not missing, f"E0a HALT: {len(missing)} decoder matrices missing (e.g. {missing[:5]})"
    assert len(matched) == n_layers * 7, f"expected {n_layers*7} matched, got {len(matched)}"

    # 3) MQA layout — only k/v rows = head_dim (single KV head) is informative; q_proj rows are vacuous.
    head_dim = int(getattr(base_cfg, "head_dim", getattr(base_cfg, "hidden_size") // getattr(base_cfg, "num_attention_heads")))
    rep["mqa"] = {"num_key_value_heads": int(getattr(base_cfg, "num_key_value_heads")),
                  "head_dim": head_dim,
                  "k0_rows": list(matched["layers.0.self_attn.k_proj.weight"][0].shape)[0],
                  "v0_rows": list(matched["layers.0.self_attn.v_proj.weight"][0].shape)[0]}
    assert rep["mqa"]["num_key_value_heads"] == 1, "expected MQA single KV head"
    assert rep["mqa"]["k0_rows"] == head_dim == rep["mqa"]["v0_rows"], "k/v rows != head_dim (MQA)"

    # 4) embed alignment by TOKEN-ID (guards against interleaved/prepended pad), classify the extra rows.
    pg_emb = pg.get_input_embeddings().weight.detach().to(torch.float64)
    base_emb = base.get_input_embeddings().weight.detach().to(torch.float64)
    shared = min(pg_emb.shape[0], base_emb.shape[0])
    known_ids = [2, 100, 1000, 10000, 100000]           # BOS + spread of real Gemma ids
    known_ids = [i for i in known_ids if i < shared]
    emb_cos = [torch.nn.functional.cosine_similarity(pg_emb[i], base_emb[i], dim=0).item() for i in known_ids]
    n_pad = pg_emb.shape[0] - base_emb.shape[0]
    pad_rows = pg_emb[base_emb.shape[0]:] if n_pad > 0 else pg_emb[:0]
    rep["embed"] = {"pg_vocab": int(pg_emb.shape[0]), "base_vocab": int(base_emb.shape[0]),
                    "n_pad_rows": int(n_pad), "known_id_cos": emb_cos,
                    "pad_norm_mean": float(pad_rows.norm(dim=1).mean()) if n_pad > 0 else None,
                    "pad_norm_std": float(pad_rows.norm(dim=1).std()) if n_pad > 1 else None}
    # shared-vocab embed rel_frob (pad rows EXCLUDED from any ΔW₁ number).
    rep["embed"]["shared_rel_frob"] = _rel_frob(base_emb[:shared], pg_emb[:shared])

    # 5) discriminator: per-matrix rel_frob (candidate) vs a MEASURED config-random Gemma-1 floor.
    cand_rel = [_rel_frob(wb, wa) for (wb, wa) in matched.values()]
    g = torch.Generator().manual_seed(a.seed)
    rand_model = GemmaForCausalLM(base_cfg)             # default HF init (unrelated weights, same shapes)
    rand_sd = _norm_keys(rand_model.model.state_dict())
    floor_rel = [_rel_frob(base_sd[k], rand_sd[k]) for k in matched]   # base vs random = wrong-base floor
    rep["matrix_rel_frob"] = {"candidate": _stats(cand_rel), "wrong_base_floor": _stats(floor_rel)}

    # RMSNorm on w_eff = 1 + stored (zero-centred storage → raw denom ≈ 0). Reported, not gated.
    norm_keys = [k for k in (set(pg_sd) & set(base_sd)) if "norm" in k and pg_sd[k].ndim == 1]
    norm_rel = [_rel_frob(1.0 + base_sd[k], 1.0 + pg_sd[k]) for k in norm_keys]
    rep["norm_rel_frob_weff"] = _stats(norm_rel) if norm_rel else None

    # 6) bf16 floor acknowledgement (informational).
    rep["bf16_floor_note"] = "stored bf16; fp32 load upcasts but a true same-init residual is bf16-granular (~1e-3 rel)"

    # 7) wiring controls: N1 self-diff exact-0 + permuted-tensor ~√2 (alignment matches the RIGHT tensors).
    compute_weight_diff(base_sd, base_sd, assert_zero=True)             # raises unless exactly 0
    k0 = "layers.0.self_attn.q_proj.weight"
    perm = base_sd[k0][torch.randperm(base_sd[k0].shape[0], generator=g)]
    rep["controls"] = {"n1_selfdiff": "exact-0 (asserted)", "permuted_rel_frob": _rel_frob(base_sd[k0], perm)}

    # verdict: candidate band must sit WELL BELOW the measured wrong-base band (non-overlapping CIs +
    #          candidate max below the wrong-base median) AND all structural/MQA/embed gates passed.
    cand = rep["matrix_rel_frob"]["candidate"]
    floor = rep["matrix_rel_frob"]["wrong_base_floor"]
    embed_ok = all(c > 0.99 for c in emb_cos)
    sep = cand["ci95_median"][1] < floor["ci95_median"][0] and cand["max"] < floor["median"]
    if sep and embed_ok:
        verdict, fail_action = "PASS_init_family_byte_unproven", None
    elif cand["median"] >= floor["ci95_median"][0]:
        verdict, fail_action = "FAIL", "HALT FT#1 axis; trigger π0-FAST-vs-π0 fallback (objective-only)"
    else:
        verdict, fail_action = "AMBIGUOUS", "re-run with fallback base revision (v1.1); downgrade FT#1 to base-init-plausible"
    rep["verdict"] = verdict
    rep["fail_action"] = fail_action
    rep["embed_known_id_match"] = bool(embed_ok)

    (out / "e0a_initfamily_gate.json").write_text(json.dumps(rep, indent=2))
    print("=" * 78)
    print(f"candidate matrix rel_frob: median={cand['median']:.4f} max={cand['max']:.4f} "
          f"CI={cand['ci95_median']}")
    print(f"wrong-base floor:          median={floor['median']:.4f} CI={floor['ci95_median']}")
    print(f"embed known-id cos: {[round(c,5) for c in emb_cos]}  permuted rel_frob: {rep['controls']['permuted_rel_frob']:.4f}")
    print("=" * 78)
    print(f"E0A_VERDICT {verdict}", json.dumps({"verdict": verdict, "fail_action": fail_action}))


if __name__ == "__main__":
    main()
