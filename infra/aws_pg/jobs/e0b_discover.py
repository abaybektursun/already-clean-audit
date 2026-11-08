"""E0b discovery probe — dump the π0 port's key structure so E0b-load / E1.5 / E2b are written against
the REAL layout, not a guess (pi-trajectory spec §E0b-load step 1). CPU, fail-loud.

lerobot/pi0_base is a LeRobot policy checkpoint (safetensors). We read its tensor HEADER (keys + shapes
+ dtypes — no full tensor load) and classify keys into backbone (…paligemma…) vs action-expert
(gemma_expert / einsum '_1') vs other, then surface the Gemma-decoder key pattern + vocab + the MQA k/v
shapes. Also dumps PaliGemma-1's LM decoder key prefix so the E0b-load remap is exact.

  pg run infra.aws_pg.jobs.e0b_discover --type r7i.2xlarge --no-weights --budget-min 40
"""
import argparse
import json
import pathlib
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from infra.pg.diff_harvest import hf_online_setup
    hf_online_setup()
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open

    rep = {"repo": "lerobot/pi0_base"}
    files = list_repo_files("lerobot/pi0_base")
    rep["repo_files"] = files
    st_files = [f for f in files if f.endswith(".safetensors")]
    rep["safetensors_files"] = st_files

    # read HEADERS only (keys + shapes + dtypes), no tensor data.
    keys = {}
    for f in st_files:
        p = hf_hub_download("lerobot/pi0_base", f)
        with safe_open(p, framework="pt") as g:
            for k in g.keys():
                sl = g.get_slice(k)
                keys[k] = {"shape": list(sl.get_shape()), "dtype": sl.get_dtype()}
    rep["n_tensors"] = len(keys)

    def bucket(k):
        if "gemma_expert" in k or re.search(r"(^|[._])_1([._]|$)", k):
            return "action_expert"
        if "paligemma" in k:
            return "backbone_paligemma"
        return "other"

    buckets = {"backbone_paligemma": [], "action_expert": [], "other": []}
    for k in sorted(keys):
        buckets[bucket(k)].append(k)
    rep["bucket_counts"] = {b: len(v) for b, v in buckets.items()}

    # backbone decoder key PATTERN: strip the leading prefix up to the Gemma decoder root.
    def decoder_suffix(k):
        m = re.search(r"(layers\.\d+\..*|embed_tokens\.weight|(?:^|\.)norm\.weight)$", k)
        return m.group(0) if m else None

    bk = buckets["backbone_paligemma"]
    suffixes = sorted({decoder_suffix(k) for k in bk if decoder_suffix(k)})
    # an example prefix (everything before the suffix) — the remap target for E0b-load/E1.5.
    example_prefix = None
    for k in bk:
        s = decoder_suffix(k)
        if s and k.endswith(s):
            example_prefix = k[: len(k) - len(s)]
            break
    rep["backbone_prefix_example"] = example_prefix
    rep["backbone_decoder_suffixes_sample"] = suffixes[:40]
    rep["n_backbone_decoder_suffixes"] = len(suffixes)

    # layer indices present + per-type counts (does it cover 18 layers × 7?).
    layer_ids = sorted({int(m.group(1)) for k in bk for m in [re.search(r"layers\.(\d+)\.", k)] if m})
    rep["backbone_layer_ids"] = layer_ids
    for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        rep[f"n_{t}"] = sum(1 for k in bk if t in k)

    # vocab + MQA k/v shapes (the convention preconditions).
    emb = [k for k in bk if "embed_tokens" in k]
    rep["embed_keys"] = {k: keys[k]["shape"] for k in emb}
    kv = {k: keys[k]["shape"] for k in bk if ("k_proj" in k or "v_proj" in k) and "layers.0." in k}
    rep["layer0_kv_shapes"] = kv
    rep["action_expert_keys_sample"] = buckets["action_expert"][:20]
    rep["other_keys_sample"] = buckets["other"][:30]

    # PaliGemma-1 LM decoder key prefix (the OTHER endpoint of ΔW₂) for the exact remap.
    from transformers import PaliGemmaForConditionalGeneration
    import torch
    from infra.pg.entities import resolve_submodule_paths
    pg = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224", torch_dtype=torch.float32, device_map="cpu",
        attn_implementation="eager")
    lm = pg.get_submodule(resolve_submodule_paths(pg)["language_model"])
    descended = hasattr(lm, "lm_head") and hasattr(lm, "model")
    if descended:
        lm = lm.model
    pg_keys = sorted(lm.state_dict().keys())
    rep["pg1_lm_descended_through_lm_head"] = descended
    rep["pg1_lm_key_sample"] = pg_keys[:12]
    rep["pg1_lm_layer0_kv_shapes"] = {k: list(lm.state_dict()[k].shape)
                                      for k in pg_keys if ("k_proj" in k or "v_proj" in k) and "layers.0." in k}

    (out / "e0b_discover.json").write_text(json.dumps(rep, indent=2))
    print("E0B_DISCOVER_OK", json.dumps(rep["bucket_counts"]),
          "layers=", layer_ids[:3], "...", layer_ids[-3:] if layer_ids else [])


if __name__ == "__main__":
    main()
